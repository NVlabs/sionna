#
# SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import unittest
import numpy as np
import tensorflow as tf
from sionna.channel import ApplyTimeChannel, ApplyOFDMChannel
from sionna import config

class TestApplyTimeChannel(unittest.TestCase):

    def test_apply_time_channel(self):
        batch_size = 16
        num_rx = 4
        num_rx_ant = 4
        num_tx = 2
        num_tx_ant = 2
        NUM_TIME_SAMPLES = [1, 5, 32, 128]
        L_TOT = [1, 3, 8, 16]
        for num_time_samples in NUM_TIME_SAMPLES:
            for l_tot in L_TOT:
                apply = ApplyTimeChannel(num_time_samples, l_tot, False)
                x = config.tf_rng.normal([batch_size,
                                      num_tx,
                                      num_tx_ant,
                                      num_time_samples])
                h_time = config.tf_rng.normal([batch_size,
                                           num_rx,
                                           num_rx_ant,
                                           num_tx,
                                           num_tx_ant,
                                           num_time_samples+l_tot-1,
                                           l_tot])
                y = apply((x, h_time)).numpy()
                self.assertEqual(y.shape, (batch_size,
                                           num_rx,
                                           num_rx_ant,
                                           num_time_samples+l_tot-1))
                y_ref = np.zeros([batch_size,
                                  num_rx,
                                  num_rx_ant,
                                  num_time_samples+l_tot-1], dtype=np.complex64)
                h_time = h_time.numpy()
                x = x.numpy()
                for b in np.arange(batch_size):
                    for rx in np.arange(num_rx):
                        for ra in np.arange(num_rx_ant):
                            for t in np.arange(num_time_samples+l_tot-1):
                                h_ = h_time[b,rx,ra,:,:,t,:]
                                x_ = x[b]
                                for l in np.arange(l_tot):
                                    if t-l < 0:
                                        break
                                    if t-l > num_time_samples-1:
                                        continue
                                    y_ref[b,rx,ra,t] += np.sum(x_[:,:,t-l]*h_[:,:,l])
                self.assertTrue(np.allclose(y_ref, y, atol=1e-5))

class TestApplyOFDMChannel(unittest.TestCase):

    def test_apply_ofdm_channel(self):
        batch_size = 16
        num_rx = 4
        num_rx_ant = 4
        num_tx = 2
        num_tx_ant = 2
        NUM_OFDM_SYMBOLS = [1, 14, 28, 64]
        FFT_SIZE = [1, 12, 32, 64]
        apply = ApplyOFDMChannel(False)
        for num_ofdm_symbols in NUM_OFDM_SYMBOLS:
            for fft_size in FFT_SIZE:
                x = config.tf_rng.normal([batch_size,
                                      num_tx,
                                      num_tx_ant,
                                      num_ofdm_symbols,
                                      fft_size])
                h_freq = config.tf_rng.normal([batch_size,
                                           num_rx,
                                           num_rx_ant,
                                           num_tx,
                                           num_tx_ant,
                                           num_ofdm_symbols,
                                           fft_size])
                y = apply((x, h_freq)).numpy()
                self.assertEqual(y.shape, (batch_size,
                                           num_rx,
                                           num_rx_ant,
                                           num_ofdm_symbols,
                                           fft_size))
                y_ref = np.zeros([batch_size,
                                  num_rx,
                                  num_rx_ant,
                                  num_ofdm_symbols,
                                  fft_size], dtype=np.complex64)
                h_freq = h_freq.numpy()
                x = x.numpy()
                for b in np.arange(batch_size):
                    for rx in np.arange(num_rx):
                        for ra in np.arange(num_rx_ant):
                            h_ = h_freq[b,rx,ra]
                            x_ = x[b]
                            y_ref[b,rx,ra] += np.sum(x_*h_, axis=(0,1))
                self.assertTrue(np.allclose(y_ref, y, atol=1e-5))
