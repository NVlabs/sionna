#
# SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Integration tests for MIMO OFDM transmissions over the CDL channel model"""

import unittest
import numpy as np
import tensorflow as tf
import sionna
from sionna import config
import numpy as np
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer, Dense
from sionna.channel.tr38901 import Antenna, CDL
from sionna.channel import OFDMChannel
from sionna.mimo import StreamManagement
from sionna.ofdm import ResourceGrid, ResourceGridMapper, LSChannelEstimator, LMMSEEqualizer
from sionna.utils import BinarySource, ebnodb2no, log10, expand_to_rank, split_dim, flatten_last_dims
from sionna.fec.ldpc.encoding import LDPC5GEncoder
from sionna.fec.ldpc.decoding import LDPC5GDecoder
from sionna.mapping import Mapper, Constellation


###########################################################
# Simulation parameters
###########################################################

# Number of bit per symbol
num_bits_per_symbol = 4

# Batch size
batch_size = 64

# Receiver-transmitter association matrix
# One stream per transmitter
stream_manager = StreamManagement(np.array([[1]]), 1)

# Resource grid
resource_grid = ResourceGrid(num_ofdm_symbols = 14,
                             fft_size = 12,
                             subcarrier_spacing = 30e3,
                             num_tx = 1,
                             num_streams_per_tx = 1,
                             cyclic_prefix_length = 0,
                             dc_null = False,
                             pilot_pattern = "kronecker",
                             pilot_ofdm_symbol_indices = [2,11],
                             num_guard_carriers = [0,0])

# FEC coderate
coderate = 0.5

# Codeword length. It is calculated from the total number of databits carried
# by the resource grid, and the number of bits transmitted per resource element
n = int(resource_grid.num_data_symbols*num_bits_per_symbol)
# Number of information bits per codeword
k = int(n*coderate)

# UT and BS antennas
ut_antenna = Antenna(polarization="single",
                     polarization_type="V",
                     antenna_pattern="38.901",
                     carrier_frequency=3.5e9)

bs_array = Antenna( polarization="single",
                    polarization_type="V",
                    antenna_pattern="38.901",
                    carrier_frequency=3.5e9)

###########################################################
# Layer implementing a small neural receiver
###########################################################

class NeuralDemapper(Layer):

    def __init__(self):
        super().__init__()

        self._dense_1 = Dense(128, 'relu')
        self._dense_2 = Dense(128, 'relu')
        self._dense_3 = Dense(num_bits_per_symbol, None)

    def call(self, inputs):
        y,no = inputs

        # Using log10 scale helps with the performance
        no_db = log10(no)

        # Stacking the real and imaginary components of the
        # complex received samples and the noise variance
        # [batch size, num_symbols_per_codeword, 3]
        z = tf.stack([tf.math.real(y),
                      tf.math.imag(y),
                      no_db], axis=2)
        # [batch size, num_symbols_per_codeword, num_bits_per_symbol]
        llr = self._dense_1(z)
        llr = self._dense_2(llr)
        llr = self._dense_3(llr)

        return llr

###########################################################
# End-to-end system with a trainable receiver
###########################################################

class E2ESystemTrainableRX(Model):

    def __init__(self):
        super().__init__()

        ################
        ## Transmitter
        ################
        self._binary_source = BinarySource()
        self._encoder = LDPC5GEncoder(k, n)
        # Trainable constellation
        constellation = Constellation("qam", num_bits_per_symbol,
                                        trainable=False)
        self.constellation = constellation
        self._mapper = Mapper(constellation=constellation)
        self._rg_mapper = ResourceGridMapper(resource_grid)

        ################
        ## Channel
        ################
        cdl = CDL("C", 100e-9, 3.5e9, ut_antenna, bs_array, "uplink")
        self._channel = OFDMChannel(cdl, resource_grid,
                                normalize_channel=True, return_channel=False)

        ################
        ## Receiver
        ################
        self._ls_est = LSChannelEstimator(resource_grid,
                                            interpolation_type="nn")
        self._lmmse_equ = LMMSEEqualizer(resource_grid, stream_manager)
        self._demapper = NeuralDemapper()
        self._decoder = LDPC5GDecoder(self._encoder, hard_out=False)

        #################
        # Loss function
        #################
        self._bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    def call(self, batch_size, ebno_db):

        # If `ebno_db` is a scalar, a tensor with shape [batch size] is
        # created as it is what is expected by some layers
        if len(ebno_db.shape) == 0:
            ebno_db = tf.fill([batch_size,], ebno_db)
        no = ebnodb2no(ebno_db, num_bits_per_symbol, coderate)
        no = expand_to_rank(no, 2)

        ################
        ## Transmitter
        ################
        # Outer coding is only performed if not training
        b = self._binary_source([batch_size, 1, 1, k])
        c = self._encoder(b)
        # Modulation
        # [batch size, 1, 1, num_data_symbol]
        x = self._mapper(c)
        # [batch size, 1, 1, fft_size, num_ofdm_symbols]
        x_rg = self._rg_mapper(x)

        ################
        ## Channel
        ################
        no_ = expand_to_rank(no, tf.rank(x_rg))
        y = self._channel([x_rg, no_])

        ################
        ## Receiver
        ################
        h_hat, err_var = self._ls_est([y, no])
        x_hat, no_eff = self._lmmse_equ([y, h_hat, err_var, no])
        x_hat = x_hat[:,0,0]
        no_eff = no_eff[:,0,0]
        llr = self._demapper([x_hat, no_eff])
        llr = tf.reshape(llr, [batch_size, n])
        llr_info = self._decoder(llr)

        #################
        # Compute loss
        #################
        loss = self._bce(b[:,0,0], llr_info)

        return loss

###########################################################
# Layer implementing a small neural transmitter
###########################################################

class NeuralMapper(Layer):

    def __init__(self):
        super().__init__()

        self._dense_1 = Dense(128, 'relu')
        self._dense_2 = Dense(128, 'relu')
        self._dense_3 = Dense(2, None)

    def call(self, bits):

        bits = tf.cast(bits, tf.int32)
        bits = split_dim(bits, [resource_grid.num_data_symbols, num_bits_per_symbol], tf.rank(bits)-1)
        x = tf.one_hot(bits, depth=2, dtype=tf.float32)
        x = flatten_last_dims(x, 2)
        x = self._dense_1(x)
        x = self._dense_2(x)
        x = self._dense_3(x)
        x = tf.complex(x[...,0], x[...,1])

        return x

###########################################################
# End-to-end system with a trainable receiver
# and transmitter
###########################################################

class E2ESystemTrainableRXTX(Model):

    def __init__(self):
        super().__init__()

        ################
        ## Transmitter
        ################
        self._binary_source = BinarySource()
        self._encoder = LDPC5GEncoder(k, n)
        # Trainable constellation
        self._mapper = NeuralMapper()
        self._rg_mapper = ResourceGridMapper(resource_grid)

        ################
        ## Channel
        ################
        cdl = CDL("C", 100e-9, 3.5e9, ut_antenna, bs_array, "uplink")
        self._channel = OFDMChannel(cdl, resource_grid,
                                normalize_channel=True, return_channel=False)

        ################
        ## Receiver
        ################
        self._ls_est = LSChannelEstimator(resource_grid,
                                            interpolation_type="nn")
        self._lmmse_equ = LMMSEEqualizer(resource_grid, stream_manager)
        self._demapper = NeuralDemapper()
        self._decoder = LDPC5GDecoder(self._encoder, hard_out=False)

        #################
        # Loss function
        #################
        self._bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    def call(self, batch_size, ebno_db):

        # If `ebno_db` is a scalar, a tensor with shape [batch size] is
        # created as it is what is expected by some layers
        if len(ebno_db.shape) == 0:
            ebno_db = tf.fill([batch_size,], ebno_db)
        no = ebnodb2no(ebno_db, num_bits_per_symbol, coderate)
        no = expand_to_rank(no, 2)

        ################
        ## Transmitter
        ################
        # Outer coding is only performed if not training
        b = self._binary_source([batch_size, 1, 1, k])
        c = self._encoder(b)
        # Modulation
        # [batch size, 1, 1, num_data_symbol]
        x = self._mapper(c)
        # [batch size, 1, 1, fft_size, num_ofdm_symbols]
        x_rg = self._rg_mapper(x)

        ################
        ## Channel
        ################
        no_ = expand_to_rank(no, tf.rank(x_rg))
        y = self._channel([x_rg, no_])

        ################
        ## Receiver
        ################
        h_hat, err_var = self._ls_est([y, no])
        x_hat, no_eff = self._lmmse_equ([y, h_hat, err_var, no])
        x_hat = x_hat[:,0,0]
        no_eff = no_eff[:,0,0]
        llr = self._demapper([x_hat, no_eff])
        llr = tf.reshape(llr, [batch_size, n])
        llr_info = self._decoder(llr)

        #################
        # Compute loss
        #################
        loss = self._bce(b[:,0,0], llr_info)

        return loss

###########################################################
# Test suite
###########################################################

class TestRxTraining(unittest.TestCase):

    def test_e2e_rx_eager_inference(self):
        model = E2ESystemTrainableRX()
        for _ in range(10):
            ebno_db = config.tf_rng.uniform([batch_size],
                        minval=-20.0, maxval=40.0, dtype=tf.float32)
            loss = model(batch_size, ebno_db)
            loss = loss.numpy()
            self.assertFalse(np.isnan(loss).any())
            self.assertFalse(np.isinf(loss).any())

    def test_e2e_rx_eager_gradient(self):
        model = E2ESystemTrainableRX()
        for _ in range(10):
            ebno_db = config.tf_rng.uniform([batch_size],
                            minval=-20.0, maxval=40.0, dtype=tf.float32)
            with tf.GradientTape() as tape:
                loss = model(batch_size, ebno_db)

            grads = tape.gradient(loss, model.trainable_weights)
            grads = tf.concat([tf.reshape(g, [-1]) for g in grads], axis=0)
            grads = loss.numpy()
            self.assertFalse(np.isnan(grads).any())
            self.assertFalse(np.isinf(grads).any())

    def test_e2e_rx_graph_inference(self):
        model = E2ESystemTrainableRX()

        @tf.function
        def graph_call(batch_size, ebno_db):
            return model(batch_size, ebno_db)

        for _ in range(10):
            ebno_db = config.tf_rng.uniform([batch_size], minval=-20.0, maxval=40.0,
                                            dtype=tf.float32)
            loss = graph_call(batch_size, ebno_db)
            loss = loss.numpy()
            self.assertFalse(np.isnan(loss).any())
            self.assertFalse(np.isinf(loss).any())

    def test_e2e_rx_graph_gradient(self):
        model = E2ESystemTrainableRX()

        @tf.function
        def graph_call(batch_size, ebno_db):
            with tf.GradientTape() as tape:
                loss = model(batch_size, ebno_db)
            grads = tape.gradient(loss, model.trainable_weights)
            grads = tf.concat([tf.reshape(g, [-1]) for g in grads], axis=0)
            return grads

        for _ in range(10):
            ebno_db = config.tf_rng.uniform([batch_size], minval=-20.0, maxval=40.0,
                                            dtype=tf.float32)
            grads = graph_call(batch_size, ebno_db)
            grads = grads.numpy()
            self.assertFalse(np.isnan(grads).any())
            self.assertFalse(np.isinf(grads).any())

    def test_e2e_rx_xla_inference(self):

        sionna.config.xla_compat = True

        model = E2ESystemTrainableRX()

        @tf.function(jit_compile=True)
        def xla_call(batch_size, ebno_db):
            return model(batch_size, ebno_db)

        for _ in range(10):
            ebno_db = config.tf_rng.uniform([batch_size], minval=-20.0, maxval=40.0,
                                            dtype=tf.float32)
            loss = xla_call(batch_size, ebno_db)
            loss = loss.numpy()
            self.assertFalse(np.isnan(loss).any())
            self.assertFalse(np.isinf(loss).any())

        sionna.config.xla_compat = False

    def test_e2e_rx_xla_gradient(self):

        sionna.config.xla_compat = True

        model = E2ESystemTrainableRX()

        @tf.function(jit_compile=True)
        def xla_call(batch_size, ebno_db):
            with tf.GradientTape() as tape:
                loss = model(batch_size, ebno_db)
            grads = tape.gradient(loss, model.trainable_weights)
            grads = tf.concat([tf.reshape(g, [-1]) for g in grads], axis=0)
            return grads

        for _ in range(10):
            ebno_db = config.tf_rng.uniform([batch_size], minval=-20.0, maxval=40.0,
                                            dtype=tf.float32)
            grads = xla_call(batch_size, ebno_db)
            grads = grads.numpy()
            self.assertFalse(np.isnan(grads).any())
            self.assertFalse(np.isinf(grads).any())

        sionna.config.xla_compat = False

class TestTxRxTraining(unittest.TestCase):

    def test_e2e_txrx_eager_inference(self):
        model = E2ESystemTrainableRXTX()
        for _ in range(10):
            ebno_db = config.tf_rng.uniform([batch_size],
                        minval=-20.0, maxval=40.0, dtype=tf.float32)
            loss = model(batch_size, ebno_db)
            loss = loss.numpy()
            self.assertFalse(np.isnan(loss).any())
            self.assertFalse(np.isinf(loss).any())

    def test_e2e_txrx_eager_gradient(self):
        model = E2ESystemTrainableRXTX()
        for _ in range(10):
            ebno_db = config.tf_rng.uniform([batch_size],
                            minval=-20.0, maxval=40.0, dtype=tf.float32)
            with tf.GradientTape() as tape:
                loss = model(batch_size, ebno_db)

            grads = tape.gradient(loss, model.trainable_weights)
            grads = tf.concat([tf.reshape(g, [-1]) for g in grads], axis=0)
            grads = loss.numpy()
            self.assertFalse(np.isnan(grads).any())
            self.assertFalse(np.isinf(grads).any())

    def test_e2e_txrx_graph_inference(self):
        model = E2ESystemTrainableRXTX()

        @tf.function
        def graph_call(batch_size, ebno_db):
            return model(batch_size, ebno_db)

        for _ in range(10):
            ebno_db = config.tf_rng.uniform([batch_size], minval=-20.0, maxval=40.0,
                                            dtype=tf.float32)
            loss = graph_call(batch_size, ebno_db)
            loss = loss.numpy()
            self.assertFalse(np.isnan(loss).any())
            self.assertFalse(np.isinf(loss).any())

    def test_e2e_txrx_graph_gradient(self):
        model = E2ESystemTrainableRXTX()

        @tf.function
        def graph_call(batch_size, ebno_db):
            with tf.GradientTape() as tape:
                loss = model(batch_size, ebno_db)
            grads = tape.gradient(loss, model.trainable_weights)
            grads = tf.concat([tf.reshape(g, [-1]) for g in grads], axis=0)
            return grads

        for _ in range(10):
            ebno_db = config.tf_rng.uniform([batch_size], minval=-20.0, maxval=40.0,
                                            dtype=tf.float32)
            grads = graph_call(batch_size, ebno_db)
            grads = grads.numpy()
            self.assertFalse(np.isnan(grads).any())
            self.assertFalse(np.isinf(grads).any())

    def test_e2e_txrx_xla_inference(self):

        sionna.config.xla_compat = True

        model = E2ESystemTrainableRXTX()

        @tf.function(jit_compile=True)
        def xla_call(batch_size, ebno_db):
            return model(batch_size, ebno_db)

        for _ in range(10):
            ebno_db = config.tf_rng.uniform([batch_size], minval=-20.0, maxval=40.0,
                                            dtype=tf.float32)
            loss = xla_call(batch_size, ebno_db)
            loss = loss.numpy()
            self.assertFalse(np.isnan(loss).any())
            self.assertFalse(np.isinf(loss).any())

        sionna.config.xla_compat = False

    # def test_e2e_txrx_xla_gradient(self):

    #     sionna.config.xla_compat = True

    #     model = E2ESystemTrainableRXTX()

    #     @tf.function(jit_compile=True)
    #     def xla_call(batch_size, ebno_db):
    #         with tf.GradientTape() as tape:
    #             loss = model(batch_size, ebno_db)
    #         grads = tape.gradient(loss, model.trainable_weights)
    #         grads = tf.concat([tf.reshape(g, [-1]) for g in grads], axis=0)
    #         return grads

    #     for _ in range(10):
    #         ebno_db = config.tf_rng.uniform([batch_size], minval=-20.0, maxval=40.0,
    #                                         dtype=tf.float32)
    #         grads = xla_call(batch_size, ebno_db)
    #         grads = grads.numpy()
    #         self.assertFalse(np.isnan(grads).any())
    #         self.assertFalse(np.isinf(grads).any())

    #     sionna.config.xla_compat = False
