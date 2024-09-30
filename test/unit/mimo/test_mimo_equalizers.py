#
# SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
import pytest
import unittest
import numpy as np
import tensorflow as tf
from sionna.utils import QAMSource, matrix_sqrt, complex_normal
from sionna.channel import FlatFadingChannel
from sionna.mimo.equalization import lmmse_equalizer, zf_equalizer, mf_equalizer
from sionna.channel.utils import exp_corr_mat

class Model(tf.keras.Model):
    def __init__(self, 
                 equalizer,
                 num_tx_ant,
                 num_rx_ant,
                 num_bits_per_symbol,
                 colored_noise=False,
                 rho=None):
        super().__init__()
        self.qam_source = QAMSource(num_bits_per_symbol)
        self.channel = FlatFadingChannel(num_tx_ant, num_rx_ant, add_awgn=not colored_noise, return_channel=True)
        self.equalizer = equalizer
        self.colored_noise = colored_noise
        if self.colored_noise:
            self.s = exp_corr_mat(rho, self.channel._num_rx_ant)
        else:
            self.s = tf.eye(self.channel._num_rx_ant, dtype=tf.complex64)
                 
    @tf.function()
    def call(self, batch_size, no):
        x = self.qam_source([batch_size, self.channel._num_tx_ant])
        if self.colored_noise:
            y, h = self.channel(x)
            s = tf.cast(no, y.dtype)*tf.eye(self.channel._num_rx_ant, dtype=tf.complex64) + self.s
            s_12 = matrix_sqrt(s)
            w = complex_normal([batch_size, self.channel._num_rx_ant, 1])
            w = tf.squeeze(tf.matmul(s_12, w), -1)
            y += w
            
        else:
            y, h = self.channel([x, no])
            s = tf.cast(no, y.dtype)*self.s

        x_hat, no_eff = self.equalizer(y, h, s)
        err = x-x_hat
        err_mean = tf.reduce_mean(err)
        err_var = tf.reduce_mean(tf.abs(err)**2)
        no_eff_mean = tf.reduce_mean(no_eff)

        return err_mean, err_var, no_eff_mean

@pytest.mark.usefixtures("only_gpu")
class TestMIMOEqualizers(unittest.TestCase):

    def test_error_statistics_awgn(self):
        num_tx_ant = 4
        num_rx_ant = 8
        num_bits_per_symbol = 4
        batch_size = 1000000
        num_batches = 10
        nos = tf.constant([0.01, 0.1, 1, 3, 10], tf.float32)
        equalizers = [lmmse_equalizer, zf_equalizer, mf_equalizer]
        for eq in equalizers:
            model = Model(eq, num_tx_ant, num_rx_ant, num_bits_per_symbol)
            for no in nos:
                for i in range(num_batches):
                    if i==0:
                        err_mean, err_var, no_eff_mean = [e/num_batches for e in model(batch_size, no)]
                    else:
                        a, b, c = [e/num_batches for e in model(batch_size, no)]
                        err_mean += a
                        err_var += b
                        no_eff_mean +=c

                # Check that the measured error has zero mean
                self.assertTrue(np.abs(err_mean)<1e-3)
                # Check that the estimated error variance matches the measured error variance
                self.assertTrue(np.abs(err_var-no_eff_mean)/no_eff_mean<1e-3)

    def test_error_statistics_colored(self):
        num_tx_ant = 4
        num_rx_ant = 8
        num_bits_per_symbol = 4
        batch_size = 1000000
        num_batches = 10
        nos = tf.constant([0.01, 0.1, 1, 3, 10], tf.float32)
        equalizers = [lmmse_equalizer, zf_equalizer, mf_equalizer]
        for eq in equalizers:
            model = Model(eq, num_tx_ant, num_rx_ant, num_bits_per_symbol, colored_noise=True, rho=0.95)
            for no in nos:
                for i in range(num_batches):
                    if i==0:
                        err_mean, err_var, no_eff_mean = [e/num_batches for e in model(batch_size, no)]
                    else:
                        a, b, c = [e/num_batches for e in model(batch_size, no)]
                        err_mean += a
                        err_var += b
                        no_eff_mean +=c

                # Check that the measured error has zero mean
                self.assertTrue(np.abs(err_mean)<1e-3)
                # Check that the estimated error variance matches the measured error variance
                self.assertTrue(np.abs(err_var-no_eff_mean)/no_eff_mean<1e-3)
