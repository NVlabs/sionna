#
# SPDX-FileCopyrightText: Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Integration tests various OFDM MIMO Detectors"""

import unittest
import numpy as np
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
print('Number of GPUs available :', len(gpus))
if gpus:
    gpu_num = 0 # Number of the GPU to be used
    try:
        tf.config.set_visible_devices(gpus[gpu_num], 'GPU')
        print('Only GPU number', gpu_num, 'used.')
        tf.config.experimental.set_memory_growth(gpus[gpu_num], True)
    except RuntimeError as e:
        print(e)

try:
    import sionna
except ImportError as e:
    import sys
    sys.path.append("../")
    import sionna

from sionna.mimo import StreamManagement
from sionna.ofdm import ResourceGrid, ResourceGridMapper, RemoveNulledSubcarriers, LinearDetector, EPDetector, KBestDetector, MaximumLikelihoodDetector
from sionna.channel.tr38901 import AntennaArray, CDL
from sionna.channel import OFDMChannel
from sionna.fec.ldpc.encoding import LDPC5GEncoder
from sionna.fec.ldpc.decoding import LDPC5GDecoder
from sionna.mapping import Mapper
from sionna.utils import BinarySource, ebnodb2no, compute_ber

class OFDMModel(tf.keras.Model):
    def __init__(self,
                detector,
                output):
        super().__init__()
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
                               num_streams_per_tx=self.num_tx_ant)
        self.n = int(self.rg.num_data_symbols * self.num_bits_per_symbol)
        self.k = int(self.n * self.coderate)

        self.ut_array = AntennaArray(num_rows=1,
                                     num_cols=int(self.num_tx_ant/2),
                                     polarization="dual",
                                     polarization_type="cross",
                                     antenna_pattern="38.901",
                                     carrier_frequency=self.carrier_frequency)

        self.bs_array = AntennaArray(num_rows=1,
                                     num_cols=int(self.num_rx_ant/2),
                                     polarization="dual",
                                     polarization_type="cross",
                                     antenna_pattern="38.901",
                                     carrier_frequency=self.carrier_frequency)

        self.cdl = CDL(model="A",
                       delay_spread=100e-9,
                       carrier_frequency=self.carrier_frequency,
                       ut_array=self.ut_array,
                       bs_array=self.bs_array,
                       direction="uplink",
                       min_speed=3.0)

        self.channel = OFDMChannel(self.cdl, self.rg, normalize_channel=True, add_awgn=False, return_channel=True)

        self.binary_source = BinarySource()
        self.encoder = LDPC5GEncoder(self.k, self.n)
        self.decoder = LDPC5GDecoder(self.encoder, hard_out=True)
        self.mapper = Mapper("qam", self.num_bits_per_symbol, return_indices=True)
        self.rg_mapper = ResourceGridMapper(self.rg)
        self.remove_nulled_scs = RemoveNulledSubcarriers(self.rg)

        if output=="symbol":
            hard_out = True
        else:
            hard_out = False

        self._output = output

        if detector in ["mf", "zf", "lmmse"]:
            self.detector = LinearDetector(detector, output, "maxlog", self.rg, self.sm, "qam", self.num_bits_per_symbol, hard_out=hard_out)
        elif detector=="ep":
            self.detector = EPDetector(output, self.rg, self.sm, self.num_bits_per_symbol, hard_out=hard_out)
        elif detector=="kbest":
            self.detector = KBestDetector(output, self.num_tx_ant, 16, self.rg, self.sm, "qam", self.num_bits_per_symbol, hard_out=hard_out)
        elif detector=="ml":
            self.detector = MaximumLikelihoodDetector(output, "maxlog", self.rg, self.sm, "qam", self.num_bits_per_symbol, hard_out=hard_out)

    def call(self, batch_size):
        no = 1e-4
        b = self.binary_source([batch_size, 1, self.num_streams_per_tx, self.k])
        c = self.encoder(b)
        x, x_ind = self.mapper(c)
        x_rg = self.rg_mapper(x)
        y, h_hat = self.channel(x_rg)
        err_var = 0.0
        llr = self.detector([y, h_hat, err_var, no])

        if self._output=="symbol":
            return x_ind, llr

        b_hat = self.decoder(llr)
        return b, b_hat

class TestOFDMMIMODetectors(unittest.TestCase):

    def test_all_detectors_in_all_modes(self):
        """Test for all detectors in all execution modes
        """

        tf.random.set_seed(1)

        for detector in ["mf", "lmmse", "zf", "ep", "kbest", "ml"]:
            for output in ["bit", "symbol"]:
                for mode in ["eager", "graph", "xla"]:
                    model = OFDMModel(detector, output)
                    if mode=="eager":
                        ber = compute_ber(*model(4))
                    elif mode=="graph":
                        ber = compute_ber(*tf.function(model)(4))
                    elif mode=="xla":
                        sionna.config.xla_compat=True
                        ber = compute_ber(*tf.function(model, jit_compile=True)(4))
                        sionna.config.xla_compat=False
                    if detector=="mf":
                        self.assertTrue(ber<1)
                    elif detector=="ep" and mode=="xla":
                        self.assertTrue(ber<1)
                    else:
                        self.assertTrue(ber==0)
