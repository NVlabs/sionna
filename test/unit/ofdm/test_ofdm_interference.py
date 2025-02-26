#
# SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

try:
    import sionna
except ImportError as e:
    import sys
    sys.path.append("../")
    import sionna

import unittest
import pytest
import numpy as np
import tensorflow as tf

class TestInterference(unittest.TestCase):

    def setUp(self):
        self.batch_size = 64
        self.num_tx = 3
        self.num_tx_ant = 2
        self.fft_size = 72
        self.num_ofdm_symbols = 14
        self.num_bits_per_symbol = 4
        self.dtype = tf.complex128

    def test_frequency_interference_sampling(self):
        # test variance and dtype of all sampling methods
        constellation = sionna.mapping.Constellation('qam', self.num_bits_per_symbol, dtype=self.dtype)
        for sampler in ["uniform", "gaussian", constellation]:
            itf_src = sionna.ofdm.OFDMInterferenceSource(density_subcarriers=1.0, sampler=sampler, domain="freq", dtype=self.dtype)
            x_itf = itf_src([self.batch_size, self.num_tx, self.num_tx_ant, self.num_ofdm_symbols, self.fft_size])
            self.assertEqual(self.dtype, x_itf.dtype)
            self.assertAlmostEqual(1.0, np.var(x_itf), 2)

    def test_density_subcarriers(self):
        density_subcarriers = 0.5
        itf_src = sionna.ofdm.OFDMInterferenceSource(density_subcarriers=density_subcarriers, sampler="uniform", domain="freq", dtype=self.dtype)
        x_itf = itf_src([self.batch_size, self.num_tx, self.num_tx_ant, self.num_ofdm_symbols, self.fft_size])
        # count number of non-zero subcarriers per batch
        num_non_zero_subcarriers = np.sum(np.sum(np.abs(x_itf), axis=(1, 2, 3)) > 0, axis=-1, dtype=np.float32)
        assert np.all(num_non_zero_subcarriers == np.round(density_subcarriers * self.fft_size))

    def test_time_interference(self):
        # check if cyclic prefix is added correctly for different cp-lengths, including edge-cases (0, fft_size, fft_size+1)
        for cp_length in [0, 16, self.fft_size, self.fft_size+1]:
            itf_src = sionna.ofdm.OFDMInterferenceSource(domain="time", cyclic_prefix_length=cp_length, fft_size=self.fft_size, dtype=self.dtype)
            num_time_samples = self.num_ofdm_symbols * (self.fft_size + cp_length)
            shape = [self.batch_size, self.num_tx, self.num_tx_ant, num_time_samples]
            if cp_length <= self.fft_size:
                x_itf = itf_src(shape)
                assert x_itf.shape == shape
                x_itf_ofdm_symbols = tf.reshape(x_itf, [-1, self.fft_size + cp_length]).numpy()
                if cp_length > 0:
                    assert np.all(x_itf_ofdm_symbols[:, -cp_length:] == x_itf_ofdm_symbols[:, :cp_length])
            else:
                with self.assertRaises(Exception):
                    x_itf = itf_src(shape)


class TestCovarianceEstimation(unittest.TestCase):

    def setUp(self):
        self.batch_size = 64
        self.num_rx = 1
        self.num_rx_ant = 8
        self.fft_size = 72
        self.num_ofdm_symbols = 14
        self.num_bits_per_symbol = 4
        self.dtype = tf.complex128
        self.y = sionna.utils.SymbolSource("qam", self.num_bits_per_symbol, dtype=self.dtype)([self.batch_size, self.num_rx, self.num_rx_ant, self.num_ofdm_symbols, self.fft_size])

    def test_only_pilots(self):
        r'''When there is only one resource element which is a pilot with energy 0, the estimated covariance matrix should be the same for the whole grid: The covariance matrix of this resource element.'''

        mask = np.zeros([1, 1, self.num_ofdm_symbols, self.fft_size], dtype=bool)
        estimation_position = (np.random.randint(0, self.num_ofdm_symbols), np.random.randint(0, self.fft_size))
        mask[0, 0, estimation_position[0], estimation_position[1]] = True
        pilots = tf.zeros([1, 1, 1], dtype=self.dtype)
        pilot_pattern = sionna.ofdm.PilotPattern(mask, pilots)
        cov_est = sionna.ofdm.CovarianceEstimator(pilot_pattern)(self.y)
        data_estimation_position = self.y[..., estimation_position[0], estimation_position[1]][..., tf.newaxis]
        cov_estimation_position = tf.matmul(data_estimation_position, data_estimation_position, adjoint_b=True)
        cov_estimation_position = sionna.utils.insert_dims(cov_estimation_position, 2, 2)
        # broadcast the estimation to the whole grid
        cov_estimation_position = tf.broadcast_to(cov_estimation_position, [self.batch_size, self.num_rx, self.num_ofdm_symbols, self.fft_size, self.num_rx_ant, self.num_rx_ant])
        assert np.allclose(cov_est.numpy(), cov_estimation_position.numpy(), rtol=1e-2)
        

    def test_scattered_pilots(self):
        r'''Test that the estimation does not fail when the pilots are scattered across the grid.'''
        mask = tf.random.uniform([1, 1, self.num_ofdm_symbols, self.fft_size], maxval=2, dtype=tf.int32)
        pilots = tf.zeros([1, 1, tf.math.reduce_sum(mask)], dtype=self.dtype)
        pilot_pattern = sionna.ofdm.PilotPattern(mask, pilots)
        cov_est = sionna.ofdm.CovarianceEstimator(pilot_pattern)(self.y)