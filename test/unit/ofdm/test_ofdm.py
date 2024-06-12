#
# SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

try:
    import sionna
except ImportError as e:
    import sys
    sys.path.append("../")

from sionna.ofdm import OFDMModulator, OFDMDemodulator, ResourceGrid, ResourceGridMapper, ResourceGridDemapper
from sionna.mimo import StreamManagement
from sionna.utils import QAMSource
from tensorflow.keras import Model

import pytest
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


class TestOFDMModulator(unittest.TestCase):
    def test_cyclic_prefixes(self):
        batch_size = 64
        fft_size = 72
        num_ofdm_symbols = 14
        qam_source = QAMSource(4)
        for cp_length in range(1,fft_size+1):
            modulator = OFDMModulator(cp_length)
            x = qam_source([batch_size, num_ofdm_symbols, fft_size])
            x_time = modulator(x)
            x_time = tf.reshape(x_time, [batch_size, num_ofdm_symbols, -1])
            self.assertTrue(np.array_equal(x_time[...,:cp_length], x_time[...,-cp_length:]))

        cp_length = fft_size+1
        modulator = OFDMModulator(cp_length)
        x = qam_source([batch_size, num_ofdm_symbols, fft_size])
        with self.assertRaises(AssertionError):
            x_time = modulator(x)

    def test_nonuniform_cyclic_prefix(self):
        batch_size = 64
        fft_size = 128
        num_ofdm_symbols = 14
        cp_length_first_symbol = 24
        cp_length_other_symbols = 16

        for symbols_per_block in [7, 10, 20]:
            modulator = OFDMModulator(cp_length_other_symbols, cp_length_first_symbol,
                                      symbols_per_block)
            qam_source = QAMSource(4)
            x = qam_source([batch_size, num_ofdm_symbols, fft_size])
            x_time = modulator(x)

            x_np = x.numpy()
            x_np = np.fft.ifft(np.fft.ifftshift(x_np, axes=-1), norm="ortho")
            x_time_np = np.empty(shape=(batch_size, 0))
            for i in range(num_ofdm_symbols):
                if i % symbols_per_block == 0:
                    x_time_np = np.concatenate([x_time_np,
                                                x_np[..., i, -cp_length_first_symbol:],
                                                x_np[..., i, :]], axis=-1)
                else:
                    x_time_np = np.concatenate([x_time_np,
                                                x_np[..., i, -cp_length_other_symbols:],
                                                x_np[..., i, :]], axis=-1)

            np.testing.assert_array_almost_equal(x_time, x_time_np)

    def test_higher_dimensions(self):
        batch_size = [64, 12, 6]
        fft_size = 72
        num_ofdm_symbols = 14
        qam_source = QAMSource(4)
        for cp_length in range(1,fft_size+1):
            modulator = OFDMModulator(cp_length)
            x = qam_source(batch_size + [num_ofdm_symbols, fft_size])
            x_time = modulator(x)
            x_time = tf.reshape(x_time, batch_size + [num_ofdm_symbols, -1])
            self.assertTrue(np.array_equal(x_time[...,:cp_length], x_time[...,-cp_length:]))

class TestOFDMDemodulator(unittest.TestCase):
    def test_cyclic_prefixes(self):
        batch_size = 64
        fft_size = 72
        num_ofdm_symbols = 14
        qam_source = QAMSource(4)
        for cp_length in range(0,fft_size+1):
            modulator = OFDMModulator(cp_length)
            demodulator = OFDMDemodulator(fft_size, 0, cp_length)
            x = qam_source([batch_size, num_ofdm_symbols, fft_size])
            x_time = modulator(x)
            x_hat = demodulator(x_time)
            self.assertTrue(np.max(np.abs(x-x_hat))<1e-6)

    def test_nonuniform_cyclic_prefix(self):
        batch_size = 64
        fft_size = 128
        num_ofdm_symbols = 14
        cp_length_first_symbol = 24
        cp_length_other_symbols = 16

        for symbols_per_block in [7, 10, 20]:
            modulator = OFDMModulator(cp_length_other_symbols,
                cp_length_first_symbol, symbols_per_block)
            demodulator = OFDMDemodulator(fft_size, 0, cp_length_other_symbols,
                                          cp_length_first_symbol, symbols_per_block)
            qam_source = QAMSource(4)
            x = qam_source([batch_size, num_ofdm_symbols, fft_size])
            x_time = modulator(x)
            x_hat = demodulator(x_time)

            np.testing.assert_array_almost_equal(x, x_hat)

    def test_higher_dimensions(self):
        batch_size = [64, 12, 6]
        fft_size = 72
        num_ofdm_symbols = 14
        qam_source = QAMSource(4)
        for cp_length in range(1,fft_size+1):
            modulator = OFDMModulator(cp_length)
            demodulator = OFDMDemodulator(fft_size, 0, cp_length)
            x = qam_source(batch_size + [num_ofdm_symbols, fft_size])
            x_time = modulator(x)
            x_hat = demodulator(x_time)
            self.assertTrue(np.max(np.abs(x-x_hat))<1e-6)

    def test_overlapping_input(self):
        batch_size = 64
        fft_size = 72
        num_ofdm_symbols = 14
        qam_source = QAMSource(4)
        for cp_length in [0, 12]:
            modulator = OFDMModulator(cp_length)
            demodulator = OFDMDemodulator(fft_size, 0, cp_length)
            x = qam_source([batch_size, num_ofdm_symbols, fft_size])
            x_time = modulator(x)
            x_time = tf.concat([x_time, x_time[...,:10]], axis=-1)
            x_hat = demodulator(x_time)
            self.assertTrue(np.max(np.abs(x-x_hat))<1e-6)

class TestOFDMModDemod(unittest.TestCase):
    def test_end_to_end(self):
        """E2E test verifying that all shapes can be properly inferred (see Issue #7)"""
        class E2ESystem(Model):
            def __init__(self, cp_length, padding):
                super().__init__()
                self.cp_length = cp_length
                self.padding = padding
                self.fft_size = 72
                self.num_ofdm_symbols = 14
                self.qam = QAMSource(4)
                self.mod = OFDMModulator(self.cp_length)
                self.demod = OFDMDemodulator(self.fft_size, 0, self.cp_length)

            @tf.function(jit_compile=True)
            def call(self, batch_size):
                x_rg = self.qam([batch_size, 1, 1, self.num_ofdm_symbols, self.fft_size])
                x_time = self.mod(x_rg)
                pad = tf.zeros_like(x_time)[...,:self.padding]
                x_time = tf.concat([x_time, pad], axis=-1)
                x_f = self.demod(x_time)
                return x_f

        for cp_length in [0,1,5,12]:
            for padding in [0,1,5,71]:
                e2e = E2ESystem(cp_length, padding)
                self.assertEqual(e2e(128).shape, [128,1,1,e2e.num_ofdm_symbols,e2e.fft_size])

class TestResourceGridDemapper(unittest.TestCase):

    def test_various_params(self):
        """data_dim dimension is omitted"""
        fft_size = 72

        def func(cp_length, num_tx, num_streams_per_tx):
            rg = ResourceGrid(num_ofdm_symbols=14,
                              fft_size=fft_size,
                              subcarrier_spacing=30e3,
                              num_tx=num_tx,
                              num_streams_per_tx=num_streams_per_tx,
                              cyclic_prefix_length=cp_length)
            sm = StreamManagement(np.ones([1, rg.num_tx]), rg.num_streams_per_tx)
            rg_mapper = ResourceGridMapper(rg)
            rg_demapper = ResourceGridDemapper(rg, sm)
            modulator = OFDMModulator(rg.cyclic_prefix_length)
            demodulator = OFDMDemodulator(rg.fft_size, 0, rg.cyclic_prefix_length)
            qam_source = QAMSource(4)
            x = qam_source([128, rg.num_tx, rg.num_streams_per_tx, rg.num_data_symbols])
            x_rg = rg_mapper(x)
            x_time = modulator(x_rg)
            y = demodulator(x_time)
            x_hat = rg_demapper(y)
            return np.max(np.abs(x-x_hat))

        for cp_length in [0,1,12,fft_size]:
            for num_tx in [1,2,3,8]:
                for num_streams_per_tx in [1,2,3]:
                    err = func(cp_length, num_tx, num_streams_per_tx)
                    self.assertTrue(err<1e-6)

    def test_data_dim(self):
        """data_dim dimension is provided"""
        fft_size = 72

        def func(cp_length, num_tx, num_streams_per_tx):
            rg = ResourceGrid(num_ofdm_symbols=14,
                              fft_size=fft_size,
                              subcarrier_spacing=30e3,
                              num_tx=num_tx,
                              num_streams_per_tx=num_streams_per_tx,
                              cyclic_prefix_length=cp_length)
            sm = StreamManagement(np.ones([1, rg.num_tx]), rg.num_streams_per_tx)
            rg_mapper = ResourceGridMapper(rg)
            rg_demapper = ResourceGridDemapper(rg, sm)
            modulator = OFDMModulator(rg.cyclic_prefix_length)
            demodulator = OFDMDemodulator(rg.fft_size, 0, rg.cyclic_prefix_length)
            qam_source = QAMSource(4)
            x = qam_source([128, rg.num_tx, rg.num_streams_per_tx, rg.num_data_symbols])
            x_rg = rg_mapper(x)
            x_time = modulator(x_rg)
            y = demodulator(x_time)
            # Stack inputs to ResourceGridDemapper to simulate the data_dim dimension
            y = tf.stack([y,y,y], axis=-1)
            x_hat = rg_demapper(y)
            x = tf.stack([x,x,x], axis=-1)
            return np.max(np.abs(x-x_hat))

        for cp_length in [0,1,12,fft_size]:
            for num_tx in [1,2,3,8]:
                for num_streams_per_tx in [1,2,3]:
                    err = func(cp_length, num_tx, num_streams_per_tx)
                    self.assertTrue(err<1e-6)
