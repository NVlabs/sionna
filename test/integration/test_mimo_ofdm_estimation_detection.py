#
# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0#
"""Integration tests various OFDM MIMO Detectors"""

import pytest
import numpy as np
import tensorflow as tf

from sionna.phy import Block
from sionna.phy.mimo import StreamManagement
from sionna.phy.ofdm import ResourceGrid, ResourceGridMapper, LinearDetector, \
                            EPDetector, KBestDetector, MMSEPICDetector, \
                            LMMSEInterpolator, LSChannelEstimator, \
                            tdl_freq_cov_mat, tdl_time_cov_mat
from sionna.phy.channel.tr38901 import TDL
from sionna.phy.channel import OFDMChannel, exp_corr_mat
from sionna.phy.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder
from sionna.phy.mapping import Mapper, BinarySource
from sionna.phy.utils import ebnodb2no, compute_ber

class MIMOOFDMLink(Block):

    def __init__(self, output, det_method, perf_csi, num_tx, num_bits_per_symbol, det_param=None, coderate=0.5, **kwargs):
        super().__init__(**kwargs)

        assert det_method in ('lmmse', 'k-best', 'ep', 'mmse-pic'), "Unknown detection method"

        self._output = output
        self.num_tx = num_tx
        self.num_bits_per_symbol = num_bits_per_symbol
        self.coderate = coderate
        self.det_method = det_method
        self.perf_csi = perf_csi

        self.num_ofdm_symbols = 14
        self.fft_size = 12*4 # 4 PRBs
        self.subcarrier_spacing = 30e3 # Hz
        self.carrier_frequency = 3.5e9 # Hz
        self.speed = 3. # m/s

        # 3GPP UMi channel model is considered
        num_rx_ant = 16
        delay_spread = 300e-9
        rx_corr_mat = exp_corr_mat(0.5, num_rx_ant).numpy()
        tx_corr_mat = exp_corr_mat(0.0, self.num_tx).numpy()
        space_cov_mat = np.kron(rx_corr_mat, tx_corr_mat)
        space_cov_mat = tf.constant(space_cov_mat, tf.complex64)
        rx_corr_mat = tf.constant(rx_corr_mat, tf.complex64)
        self.channel_model = TDL('A', delay_spread=300e-9, carrier_frequency=self.carrier_frequency,
                                 num_rx_ant=num_rx_ant, num_tx_ant=self.num_tx,
                                 spatial_corr_mat=space_cov_mat)

        # Configure the resource grid
        rg = ResourceGrid(num_ofdm_symbols=self.num_ofdm_symbols,
                          fft_size=self.fft_size,
                          subcarrier_spacing=self.subcarrier_spacing,
                          num_tx=1,
                          num_streams_per_tx=self.num_tx,
                          pilot_pattern="kronecker",
                          pilot_ofdm_symbol_indices=[2,11])
        self.rg = rg

        # Stream management
        sm = StreamManagement([[1]], self.num_tx)

        # Codeword length and number of information bits per codeword
        n = int(rg.num_data_symbols*num_bits_per_symbol)
        k = int(coderate*n)
        self.n = n
        self.k = k

        # If output is symbol, then no FEC is used and hard decision are output
        hard_out = (output == "symbol")
        coded = (output == "bit")
        self.hard_out = hard_out
        self.coded = coded

        ##################################
        # Transmitter
        ##################################

        self.binary_source = BinarySource()
        self.mapper = Mapper(constellation_type="qam", num_bits_per_symbol=num_bits_per_symbol, return_indices=True)
        self.rg_mapper = ResourceGridMapper(rg)
        if coded:
            self.encoder = LDPC5GEncoder(k, n, num_bits_per_symbol=num_bits_per_symbol)

        ##################################
        # Channel
        ##################################

        self.channel = OFDMChannel(self.channel_model, rg, return_channel=True)

        ###################################
        # Receiver
        ###################################

        # Channel estimation
        if not self.perf_csi:
            freq_cov_mat = tdl_freq_cov_mat('A', self.subcarrier_spacing, self.fft_size, delay_spread)
            time_cov_mat = tdl_time_cov_mat('A', self.speed, self.carrier_frequency, rg.ofdm_symbol_duration, self.num_ofdm_symbols)
            lmmse_int_time_first = LMMSEInterpolator(rg.pilot_pattern, time_cov_mat, freq_cov_mat, rx_corr_mat, order='t-f-s')
            self.channel_estimator = LSChannelEstimator(rg, interpolator=lmmse_int_time_first)

        # Detection
        if det_method == "lmmse":
            self.detector = LinearDetector("lmmse", output, "app", rg, sm, constellation_type="qam", num_bits_per_symbol=num_bits_per_symbol, hard_out=hard_out)
        elif det_method == 'k-best':
            if det_param is None:
                k = 64
            else:
                k = det_param
            self.detector = KBestDetector(output, num_tx, k, rg, sm, constellation_type="qam", num_bits_per_symbol=num_bits_per_symbol, hard_out=hard_out)
        elif det_method == "ep":
            if det_param is None:
                l = 10
            else:
                l = det_param
            self.detector = EPDetector(output, rg, sm, num_bits_per_symbol, l=l, hard_out=hard_out)
        elif det_method == 'mmse-pic':
            if det_param is None:
                l = 4
            else:
                l = det_param
            self.detector = MMSEPICDetector(output, 'app', rg, sm, num_iter=l, constellation_type="qam", num_bits_per_symbol=num_bits_per_symbol, hard_out=hard_out)

        if coded:
            self.decoder = LDPC5GDecoder(self.encoder, hard_out=True)

    def call(self, batch_size, ebno_db):


        ##################################
        # Transmitter
        ##################################

        if self.coded:
            b = self.binary_source([batch_size, 1, self.num_tx, self.k])
            c = self.encoder(b)
        else:
            c = self.binary_source([batch_size, 1, self.num_tx, self.n])
        bits_shape = tf.shape(c)
        x,x_ind = self.mapper(c)
        x_rg = self.rg_mapper(x)

        ##################################
        # Channel
        ##################################

        no = ebnodb2no(ebno_db, self.num_bits_per_symbol, self.coderate, resource_grid=self.rg)
        y_rg, h_freq = self.channel(x_rg, no)

        ###################################
        # Receiver
        ###################################

        # Channel estimation
        if self.perf_csi:
            h_hat = h_freq
            err_var = 0.0
        else:
            h_hat,err_var = self.channel_estimator(y_rg, no)

        # Detection
        if self.det_method == "mmse-pic":
            if self._output == "bit":
                prior_shape = bits_shape
            elif self._output == "symbol":
                prior_shape = tf.concat([tf.shape(x), [2**self.num_bits_per_symbol]], axis=0)
            prior = tf.zeros(prior_shape)
            det_out = self.detector(y_rg,h_hat,prior,err_var,no)
        else:
            det_out = self.detector(y_rg,h_hat,err_var,no)

        # (Decoding) and output
        if self._output == "bit":
            llr = tf.reshape(det_out, bits_shape)
            b_hat = self.decoder(llr)
            return b, b_hat
        elif self._output == "symbol":
            x_hat = tf.reshape(det_out, tf.shape(x_ind))
            return x_ind, x_hat

@pytest.mark.parametrize("detector", ["lmmse", "ep", "k-best", "mmse-pic"])
@pytest.mark.parametrize("output", ["bit", "symbol"])
@pytest.mark.parametrize("mode", ["eager", "graph"])
def test_all_detectors_in_all_modes(detector, output, mode):
    """Test for all detectors in all execution modes"""
    model = MIMOOFDMLink(output, detector, False, 4, 2)
    if mode=="eager":
        ber = compute_ber(*model(tf.cast(1, tf.int32), tf.cast(40, tf.float32)))
    else:
        ber = compute_ber(*tf.function(model, jit_compile=mode=="xla")(tf.cast(1, tf.int32), tf.cast(40, tf.float32)))
    assert ber == 0
