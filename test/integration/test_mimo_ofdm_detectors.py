#
# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0#
"""Integration tests various OFDM MIMO Detectors"""

import pytest
import numpy as np
import tensorflow as tf
from sionna.phy import Block
from sionna.phy.mimo import StreamManagement
from sionna.phy.ofdm import ResourceGrid, ResourceGridMapper, \
                            RemoveNulledSubcarriers, LinearDetector, \
                            EPDetector, KBestDetector, MaximumLikelihoodDetector
from sionna.phy.channel.tr38901 import AntennaArray, CDL
from sionna.phy.channel import OFDMChannel
from sionna.phy.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder
from sionna.phy.mapping import Mapper, BinarySource
from sionna.phy.utils import compute_ber

class OFDMModel(Block):
    def __init__(self,
                 detector,
                 output,
                 precision="double"):
        super().__init__(precision=precision)
        self.num_tx_ant = 4
        self.num_rx_ant = 8
        self.num_streams_per_tx = self.num_tx_ant
        self.coderate = 0.5
        self.num_bits_per_symbol = 4
        self.carrier_frequency = 2.6e9
        self.sm = StreamManagement(np.array([[1]]), self.num_streams_per_tx)
        self.rg = ResourceGrid(num_ofdm_symbols=14,
                               fft_size=12,
                               subcarrier_spacing=15e3,
                               num_tx=1,
                               num_streams_per_tx=self.num_tx_ant,
                               precision=precision)
        self.n = int(self.rg.num_data_symbols * self.num_bits_per_symbol)
        self.k = int(self.n * self.coderate)

        self.ut_array = AntennaArray(num_rows=1,
                                     num_cols=int(self.num_tx_ant/2),
                                     polarization="dual",
                                     polarization_type="cross",
                                     antenna_pattern="38.901",
                                     carrier_frequency=self.carrier_frequency,
                                     precision=precision)

        self.bs_array = AntennaArray(num_rows=1,
                                     num_cols=int(self.num_rx_ant/2),
                                     polarization="dual",
                                     polarization_type="cross",
                                     antenna_pattern="38.901",
                                     carrier_frequency=self.carrier_frequency,
                                     precision=precision)

        self.cdl = CDL(model="A",
                       delay_spread=100e-9,
                       carrier_frequency=self.carrier_frequency,
                       ut_array=self.ut_array,
                       bs_array=self.bs_array,
                       direction="uplink",
                       min_speed=3.0,
                       precision=precision)

        self.channel = OFDMChannel(self.cdl, self.rg, normalize_channel=True,
                                   return_channel=True,
                                   precision=precision)

        self.binary_source = BinarySource(precision=precision)
        self.encoder = LDPC5GEncoder(self.k, self.n, precision=precision)
        self.decoder = LDPC5GDecoder(self.encoder, hard_out=True, precision=precision)
        self.mapper = Mapper("qam", self.num_bits_per_symbol, return_indices=True, precision=precision)
        self.rg_mapper = ResourceGridMapper(self.rg, precision=precision)
        self.remove_nulled_scs = RemoveNulledSubcarriers(self.rg, precision=precision)

        if output=="symbol":
            hard_out = True
        else:
            hard_out = False

        self._output = output

        if detector in ["mf", "zf", "lmmse"]:
            self.detector = LinearDetector(detector, output, "maxlog", self.rg, self.sm, "qam", self.num_bits_per_symbol, hard_out=hard_out, precision=precision)
        elif detector=="ep":
            self.detector = EPDetector(output, self.rg, self.sm, self.num_bits_per_symbol, hard_out=hard_out, precision=precision)
        elif detector=="kbest":
            self.detector = KBestDetector(output, self.num_tx_ant, 16, self.rg, self.sm, "qam", self.num_bits_per_symbol, hard_out=hard_out, precision=precision)
        elif detector=="ml":
            self.detector = MaximumLikelihoodDetector(output, "maxlog", self.rg, self.sm, "qam", self.num_bits_per_symbol, hard_out=hard_out, precision=precision)

    def call(self, batch_size):
        b = self.binary_source([batch_size, 1, self.num_streams_per_tx, self.k])
        c = self.encoder(b)
        x, x_ind = self.mapper(c)
        x_rg = self.rg_mapper(x)
        y, h_hat = self.channel(x_rg)
        err_var = tf.cast(0.0, y.dtype.real_dtype)
        no = tf.cast(1e-4, y.dtype.real_dtype)
        llr = self.detector(y, h_hat, err_var, no)

        if self._output=="symbol":
            return x_ind, llr

        b_hat = self.decoder(llr)
        return b, b_hat

@pytest.mark.parametrize("mode", ["eager", "graph", "xla"])
@pytest.mark.parametrize("output", ["bit", "symbol"])
@pytest.mark.parametrize("detector", ["mf", "lmmse", "zf", "ep", "kbest", "ml"])
def test_all_detectors_in_all_modes(mode, detector, output):
    """Test for all detectors in all execution modes"""
    model = OFDMModel(detector, output)
    if mode=="eager":
        ber = compute_ber(*model(4))
    elif mode=="graph":
        ber = compute_ber(*tf.function(model)(4))
    elif mode=="xla":
        ber = compute_ber(*tf.function(model, jit_compile=True)(4))
    if detector=="mf":
        assert ber<1
    elif detector=="ep" and mode=="xla":
        assert ber<1
    else:
        assert ber==0
