#
# SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Integration tests for MIMO transmissions over flat fading channels"""

import unittest
import numpy as np
import tensorflow as tf
import sionna
from sionna.utils import BinarySource, QAMSource, ebnodb2no, compute_ser, compute_ber, sim_ber
from sionna.channel import FlatFadingChannel, KroneckerModel
from sionna.channel.utils import exp_corr_mat
from sionna.mimo import lmmse_equalizer
from sionna.mapping import SymbolDemapper, Mapper, Demapper
from sionna.fec.ldpc.encoding import LDPC5GEncoder
from sionna.fec.ldpc.decoding import LDPC5GDecoder

sionna.config.xla_compat=True
class Model(tf.keras.Model):
    def __init__(self, spatial_corr=None):
        super().__init__()
        self.n = 1024 
        self.k = 512  
        self.coderate = self.k/self.n
        self.num_bits_per_symbol = 4
        self.num_tx_ant = 4
        self.num_rx_ant = 16
        self.binary_source = BinarySource()
        self.encoder = LDPC5GEncoder(self.k, self.n)
        self.mapper = Mapper("qam", self.num_bits_per_symbol)
        self.demapper = Demapper("app", "qam", self.num_bits_per_symbol)
        self.decoder = LDPC5GDecoder(self.encoder, hard_out=True)
        self.channel = FlatFadingChannel(self.num_tx_ant,
                                         self.num_rx_ant,
                                         spatial_corr=spatial_corr,
                                         add_awgn=True,
                                         return_channel=True)
        
    @tf.function(jit_compile=True)
    def call(self, batch_size, ebno_db):
        b = self.binary_source([batch_size, self.num_tx_ant, self.k])
        c = self.encoder(b)
        
        x = self.mapper(c)
        shape = tf.shape(x)
        x = tf.reshape(x, [-1, self.num_tx_ant])
        
        no = ebnodb2no(ebno_db, self.num_bits_per_symbol, self.coderate)
        no *= np.sqrt(self.num_rx_ant)

        y, h = self.channel([x, no])
        s = tf.complex(no*tf.eye(self.num_rx_ant, self.num_rx_ant), 0.0)
        
        x_hat, no_eff = lmmse_equalizer(y, h, s)
        
        x_hat = tf.reshape(x_hat, shape)
        no_eff = tf.reshape(no_eff, shape)
        
        llr = self.demapper([x_hat, no_eff])
        b_hat = self.decoder(llr)
        
        return b,  b_hat

class TestMIMOFlatFading(unittest.TestCase):

    def test_uncorrelated(self):
        model = Model()
        ber, bler = sim_ber(model,
                        [0, 10, 20],
                        batch_size=64,
                        max_mc_iter=10)
        self.assertFalse(np.isnan(ber).any())
        self.assertFalse(np.isnan(bler).any())

    def test_correlated(self):
        num_tx_ant = 4
        num_rx_ant = 16
        r_tx = exp_corr_mat(0.4, num_tx_ant) 
        r_rx = exp_corr_mat(0.7, num_rx_ant)
        model = Model(KroneckerModel(r_tx, r_rx))
        ber, bler = sim_ber(model,
                        [0, 10, 20],
                        batch_size=64,
                        max_mc_iter=10)
        self.assertFalse(np.isnan(ber).any())
        self.assertFalse(np.isnan(bler).any())
