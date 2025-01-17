#
# SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
import unittest
import numpy as np
import tensorflow as tf
from sionna.channel import GenerateFlatFadingChannel, ApplyFlatFadingChannel, FlatFadingChannel, exp_corr_mat, KroneckerModel
from sionna.utils import QAMSource

class TestGenerateFlatFading(unittest.TestCase):
    """Unittest for GenerateFlatFading"""

    def test_without_spatial_correlation(self):
        num_tx_ant = 4
        num_rx_ant = 16
        batch_size = 128
        gen_chn = GenerateFlatFadingChannel(num_tx_ant, num_rx_ant)
        h = gen_chn(batch_size)
        self.assertEqual(h.shape, [batch_size, num_rx_ant, num_tx_ant])
        self.assertEqual(h.dtype, tf.complex64)
        gen_chn = GenerateFlatFadingChannel(num_tx_ant, num_rx_ant, dtype=tf.complex128)
        h = gen_chn(batch_size)
        self.assertEqual(h.dtype, tf.complex128)

    def test_with_spatial_correlation(self):
        num_tx_ant = 4
        num_rx_ant = 16
        r_tx = exp_corr_mat(0.4, num_tx_ant)
        r_rx = exp_corr_mat(0.99, num_rx_ant)
        kron = KroneckerModel(r_tx, r_rx)
        gen_chn = GenerateFlatFadingChannel(num_tx_ant, num_rx_ant, spatial_corr=kron)

        @tf.function()
        def func():
            h = gen_chn(1000000)
            r_tx_hat = tf.reduce_mean(tf.matmul(h, h, adjoint_a=True), 0)
            r_rx_hat = tf.reduce_mean(tf.matmul(h, h, adjoint_b=True), 0)
            return r_tx_hat, r_rx_hat

        r_tx_hat = tf.zeros_like(r_tx)
        r_rx_hat = tf.zeros_like(r_rx)
        iterations = 10
        for i in range(iterations):
            tmp = func()
            r_tx_hat += tmp[0]/iterations/num_rx_ant
            r_rx_hat += tmp[1]/iterations/num_tx_ant
        self.assertTrue(np.allclose(r_tx, r_tx_hat, atol=1e-3))
        self.assertTrue(np.allclose(r_rx, r_rx_hat, atol=1e-3))

    def test_property_setter(self):
        num_tx_ant = 4
        num_rx_ant = 16
        r_tx = exp_corr_mat(0.4, num_tx_ant)
        r_rx = exp_corr_mat(0.99, num_rx_ant)
        kron = KroneckerModel(r_tx, r_rx)
        gen_chn = GenerateFlatFadingChannel(num_tx_ant, num_rx_ant)

        @tf.function()
        def func():
            gen_chn.spatial_corr = kron
            h = gen_chn(1000000)
            r_tx_hat = tf.reduce_mean(tf.matmul(h, h, adjoint_a=True), 0)
            r_rx_hat = tf.reduce_mean(tf.matmul(h, h, adjoint_b=True), 0)
            return r_tx_hat, r_rx_hat

        r_tx_hat = tf.zeros_like(r_tx)
        r_rx_hat = tf.zeros_like(r_rx)
        iterations = 10
        for i in range(iterations):
            tmp = func()
            r_tx_hat += tmp[0]/iterations/num_rx_ant
            r_rx_hat += tmp[1]/iterations/num_tx_ant
        self.assertTrue(np.allclose(r_tx, r_tx_hat, atol=1e-3))
        self.assertTrue(np.allclose(r_rx, r_rx_hat, atol=1e-3))

class TestGenerateApplyFading(unittest.TestCase):
    """Unittest for ApplyFlatFading"""
    def test_without_noise(self):
        num_tx_ant = 4
        num_rx_ant = 16
        batch_size = 24
        r_tx = exp_corr_mat(0.4, num_tx_ant)
        r_rx = exp_corr_mat(0.99, num_rx_ant)
        kron = KroneckerModel(r_tx, r_rx)
        gen_chn = GenerateFlatFadingChannel(num_tx_ant, num_rx_ant, spatial_corr=kron)
        app_chn = ApplyFlatFadingChannel(add_awgn=False)
        h = gen_chn(batch_size)
        x = QAMSource(4)([batch_size, num_tx_ant])
        y = app_chn([x, h])
        self.assertTrue(np.array_equal(y, tf.squeeze(tf.matmul(h, tf.expand_dims(x, -1)))))

    def test_with_noise(self):
        num_tx_ant = 4
        num_rx_ant = 16
        batch_size = 100000
        r_tx = exp_corr_mat(0.4, num_tx_ant)
        r_rx = exp_corr_mat(0.99, num_rx_ant)
        kron = KroneckerModel(r_tx, r_rx)
        gen_chn = GenerateFlatFadingChannel(num_tx_ant, num_rx_ant, spatial_corr=kron)
        app_chn = ApplyFlatFadingChannel(add_awgn=True)
        h = gen_chn(batch_size)
        x = QAMSource(4)([batch_size, num_tx_ant])
        no = 0.1
        y = app_chn([x, h, no])
        n = y - tf.squeeze(tf.matmul(h, tf.expand_dims(x, -1)))
        noise_var = np.var(n)
        self.assertAlmostEqual(no, noise_var, places=3)

class TestFlatFadingChannel(unittest.TestCase):
    """Unittest for FlatFading"""
    def test_without_noise(self):
        num_tx_ant = 4
        num_rx_ant = 16
        batch_size = 24
        dtype=tf.complex128
        r_tx = exp_corr_mat(0.4, num_tx_ant, dtype)
        r_rx = exp_corr_mat(0.99, num_rx_ant, dtype)
        kron = KroneckerModel(r_tx, r_rx)
        chn = FlatFadingChannel(num_tx_ant, num_rx_ant, spatial_corr=kron, add_awgn=False, return_channel=True, dtype=dtype)
        x = QAMSource(4, dtype=tf.complex128)([batch_size, num_tx_ant])
        y, h = chn(x)
        self.assertTrue(np.array_equal(y, tf.squeeze(tf.matmul(h, tf.expand_dims(x, -1)))))

    def test_with_noise(self):
        num_tx_ant = 4
        num_rx_ant = 16
        batch_size = 100000
        dtype=tf.complex128
        r_tx = exp_corr_mat(0.4, num_tx_ant, dtype)
        r_rx = exp_corr_mat(0.99, num_rx_ant, dtype)
        kron = KroneckerModel(r_tx, r_rx)
        chn = FlatFadingChannel(num_tx_ant, num_rx_ant, spatial_corr=kron, add_awgn=True, return_channel=True, dtype=dtype)
        x = QAMSource(4, dtype=dtype)([batch_size, num_tx_ant])
        no = 0.2
        y, h = chn([x, no])
        n = y - tf.squeeze(tf.matmul(h, tf.expand_dims(x, -1)))
        noise_var = np.var(n)
        self.assertAlmostEqual(no, noise_var, places=3)

    def test_no_return_channel(self):
        num_tx_ant = 4
        num_rx_ant = 16
        batch_size = 1000000
        dtype=tf.complex64
        chn = FlatFadingChannel(num_tx_ant, num_rx_ant, add_awgn=True, return_channel=False, dtype=dtype)
        x = QAMSource(4, dtype=dtype)([batch_size, num_tx_ant])
        no = 0.2
        y = chn([x, no])
        y_var = np.var(y)
        self.assertAlmostEqual(y_var , num_tx_ant + no, places=2)

    def test_property_setter(self):
        num_tx_ant = 4
        num_rx_ant = 16
        r_tx = exp_corr_mat(0.4, num_tx_ant)
        r_rx = exp_corr_mat(0.99, num_rx_ant)
        kron = KroneckerModel(r_tx, r_rx)
        chn = FlatFadingChannel(num_tx_ant, num_rx_ant, add_awgn=True, return_channel=True)
        qam_source = QAMSource(4)
        @tf.function()
        def func():
            chn.spatial_corr = kron
            x = qam_source([1000000, num_tx_ant])
            no = 0.2
            y, h = chn([x, no])
            r_tx_hat = tf.reduce_mean(tf.matmul(h, h, adjoint_a=True), 0)
            r_rx_hat = tf.reduce_mean(tf.matmul(h, h, adjoint_b=True), 0)
            return r_tx_hat, r_rx_hat

        r_tx_hat = tf.zeros_like(r_tx)
        r_rx_hat = tf.zeros_like(r_rx)
        iterations = 10
        for i in range(iterations):
            tmp = func()
            r_tx_hat += tmp[0]/iterations/num_rx_ant
            r_rx_hat += tmp[1]/iterations/num_tx_ant
        self.assertTrue(np.allclose(r_tx, r_tx_hat, atol=1e-3))
        self.assertTrue(np.allclose(r_rx, r_rx_hat, atol=1e-3))
