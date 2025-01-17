#
# SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import unittest
import numpy as np
import tensorflow as tf
from sionna.ofdm import OFDMModulator, OFDMDemodulator, ResourceGrid, ResourceGridMapper, ResourceGridDemapper
from sionna.mimo import StreamManagement
from sionna.utils import QAMSource
from tensorflow.keras import Model

class TestOFDMModulator(unittest.TestCase):
    def test_cyclic_prefixes(self):
        "Test that cyclic prefix is correct implemented"
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
        with self.assertRaises(ValueError):
            x_time = modulator(x)

    def test_variable_cyclic_prefixes(self):
        "Test per-OFDM symbol cyclic prefix length"
        batch_size = 64
        fft_size = 72
        num_ofdm_symbols = fft_size
        qam_source = QAMSource(4)
        cp_lengths = np.arange(fft_size)
        modulator = OFDMModulator(cp_lengths)
        x = qam_source([batch_size, num_ofdm_symbols, fft_size])
        x_time = modulator(x)

        start = 0
        for i in range(num_ofdm_symbols):
            end = start + cp_lengths[i] + fft_size
            x_sym = x_time[...,start:end]
            check = np.array_equal(x_sym[:,:cp_lengths[i]], x_sym[:,x_sym.shape[-1]-cp_lengths[i]:])
            self.assertTrue(check)
            start = end

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

    def test_variable_cyclic_prefixes_higher_dimensions(self):
        "Test per-OFDM symbol cyclic prefix length with multi-dimensional batch size"
        batch_size = [64, 12, 6]
        fft_size = 72
        num_ofdm_symbols = fft_size
        qam_source = QAMSource(4)
        cp_lengths = np.arange(fft_size)
        modulator = OFDMModulator(cp_lengths)
        x = qam_source(batch_size + [num_ofdm_symbols, fft_size])
        x_time = modulator(x)

        start = 0
        for i in range(num_ofdm_symbols):
            end = start + cp_lengths[i] + fft_size
            x_sym = x_time[...,start:end]
            check = np.array_equal(x_sym[...,:cp_lengths[i]], x_sym[...,x_sym.shape[-1]-cp_lengths[i]:])
            self.assertTrue(check)
            start = end

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
            self.assertLess(np.max(np.abs(x-x_hat)), 1e-5)

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
            self.assertLess(np.max(np.abs(x-x_hat)), 1e-5)

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
            self.assertLess(np.max(np.abs(x-x_hat)), 1e-5)

class TestOFDMModDemod(unittest.TestCase):
    def test_end_to_end(self):
        """E2E test verying that all shapes can be properly inferred (see Issue #7)"""
        class E2ESystem(Model):
            def __init__(self, cp_length, padding):
                super().__init__()
                self.cp_length = cp_length
                self.padding = padding
                self.fft_size = 72
                self.num_ofdm_symbols = 14
                self.qam = QAMSource(4)
                self.mod = OFDMModulator(self.cp_length)
                self.demod  = OFDMDemodulator(self.fft_size, 0, self.cp_length)

            @tf.function(jit_compile=True)
            def call(self, batch_size):
                x_rg = self.qam([batch_size, 1, 1, self.num_ofdm_symbols, self.fft_size])
                x_time  = self.mod(x_rg)
                pad = tf.zeros_like(x_time)[...,:self.padding]
                x_time = tf.concat([x_time, pad], axis=-1)
                x_f = self.demod(x_time)
                return x_f

        for cp_length in [0,1,5,12]:
            for padding in [0,1,5,71]:
                e2e = E2ESystem(cp_length, padding)
                self.assertEqual(e2e(128).shape, [128,1,1,e2e.num_ofdm_symbols,e2e.fft_size])

        cp_lengths = np.arange(72)
        for padding in [0,1,5,71]:
                e2e = E2ESystem(cp_lengths, padding)
                e2e.num_ofdm_symbols = 72
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
                    self.assertLess(err, 1e-5)

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
            # Stack inputs to ResourceGridDemppaer to simulate the data_dim dimension
            y = tf.stack([y,y,y], axis=-1)
            x_hat = rg_demapper(y)
            x = tf.stack([x,x,x], axis=-1)
            return np.max(np.abs(x-x_hat))

        for cp_length in [0,1,12,fft_size]:
            for num_tx in [1,2,3,8]:
                for num_streams_per_tx in [1,2,3]:
                    err = func(cp_length, num_tx, num_streams_per_tx)
                    self.assertLess(err, 1e-5)
