#
# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0#
import unittest
import numpy as np
import tensorflow as tf
from sionna.phy import config
from sionna.phy.channel import exp_corr_mat, one_ring_corr_mat, KroneckerModel, PerColumnModel
from sionna.phy.utils import complex_normal

class TestKroneckerModel(unittest.TestCase):
    """Unittest for the KroneckerModel"""

    def test_covariance(self):
        M = 16
        K = 4
        precision = "double"
        r_tx = exp_corr_mat(0.4, K, precision=precision)
        r_rx = exp_corr_mat(0.99, M, precision=precision)
        batch_size = 1000000
        kron = KroneckerModel(r_tx, r_rx)

        @tf.function(jit_compile=True)
        def func():
            h = complex_normal([batch_size, M, K], precision=precision)
            h = kron(h)
            r_tx_hat = tf.reduce_mean(tf.matmul(h, h, adjoint_a=True), 0)
            r_rx_hat = tf.reduce_mean(tf.matmul(h, h, adjoint_b=True), 0)
            return r_tx_hat, r_rx_hat

        r_tx_hat = tf.zeros_like(r_tx)
        r_rx_hat = tf.zeros_like(r_rx)
        iterations = 10
        for i in range(iterations):
            tmp = func()
            r_tx_hat += tmp[0]/iterations/M
            r_rx_hat += tmp[1]/iterations/K
        self.assertTrue(np.allclose(r_tx, r_tx_hat, atol=1e-3))
        self.assertTrue(np.allclose(r_rx, r_rx_hat, atol=1e-3))

    def test_per_example_r_tx(self):
        """Configure a different tx correlation for each example"""
        M = 16
        K = 4
        precision = "double"
        batch_size = 128
        r_tx = exp_corr_mat(config.np_rng.uniform(size=[batch_size]), K, precision=precision)
        r_rx = exp_corr_mat(0.99, M, precision=precision)
        kron = KroneckerModel(r_tx, r_rx)
        h = complex_normal([batch_size, M, K], precision=precision)
        h_corr = kron(h)
        for i in range(batch_size):
            h_test = tf.linalg.cholesky(r_rx)@h[i]@tf.linalg.adjoint(tf.linalg.cholesky(r_tx[i]))
            self.assertTrue(np.allclose(h_corr[i], h_test))

    def test_per_example_r_rx(self):
        """Configure a different rx correlation for each example"""
        M = 16
        K = 4
        precision = "double"
        batch_size = 10
        r_tx = exp_corr_mat(0.4, K, precision=precision)
        r_rx = exp_corr_mat(config.np_rng.uniform(size=[batch_size]), M, precision=precision)
        kron = KroneckerModel(r_tx, r_rx)
        h = complex_normal([batch_size, M, K], precision=precision)
        h_corr = kron(h)
        for i in range(batch_size):
            h_test = tf.linalg.cholesky(r_rx[i])@h[i]@tf.linalg.adjoint(tf.linalg.cholesky(r_tx))
            self.assertTrue(np.allclose(h_corr[i], h_test))

    def test_per_example_corr(self):
        """Configure a different rx/tx correlation for each example"""
        M = 16
        K = 4
        precision = "double"
        batch_size = 10
        r_tx = exp_corr_mat(config.np_rng.uniform(size=[batch_size]), K, precision=precision)
        r_rx = exp_corr_mat(config.np_rng.uniform(size=[batch_size]), M, precision=precision)
        kron = KroneckerModel(r_tx, r_rx)
        h = complex_normal([batch_size, M, K], precision=precision)
        h_corr = kron(h)
        for i in range(batch_size):
            h_test = tf.linalg.cholesky(r_rx[i])@h[i]@tf.linalg.adjoint(tf.linalg.cholesky(r_tx[i]))
            self.assertTrue(np.allclose(h_corr[i], h_test))

    def test_same_channel_with_different_corr(self):
        """Apply different correlation matrices to the same channel"""
        M = 16
        K = 4
        precision = "double"
        batch_size = 10
        r_tx = exp_corr_mat(config.np_rng.uniform(size=[batch_size]), K, precision=precision)
        r_rx = exp_corr_mat(config.np_rng.uniform(size=[batch_size]), M, precision=precision)
        kron = KroneckerModel(r_tx, r_rx)
        h = complex_normal([M, K], precision=precision)
        h_corr = kron(h)
        for i in range(batch_size):
            h_test = tf.linalg.cholesky(r_rx[i])@h@tf.linalg.adjoint(tf.linalg.cholesky(r_tx[i]))
            self.assertTrue(np.allclose(h_corr[i], h_test))

    def test_property_setter(self):
        """Check that correlation matrices can be changed"""
        M = 16
        K = 4
        precision = "double"
        batch_size = 10
        kron = KroneckerModel(None, None)

        @tf.function(jit_compile=True)
        def func():
            r_tx = exp_corr_mat(0.4, K, precision=precision)
            r_rx = exp_corr_mat(0.9, M, precision=precision)
            kron.r_tx = r_tx
            kron.r_rx = r_rx
            h = complex_normal([batch_size, M, K], precision=precision)
            h_corr = kron(h)
            return h, h_corr, r_tx, r_rx

        h, h_corr, r_tx, r_rx = func()
        for i in range(batch_size):
            h_test = tf.linalg.cholesky(r_rx)@h[i]@tf.linalg.adjoint(tf.linalg.cholesky(r_tx))
            self.assertTrue(np.allclose(h_corr[i], h_test, atol=1e-6))

class TestPerColumnModel(unittest.TestCase):
    def test_covariance(self):
        M = 16
        K = 4
        precision = "double"
        r_rx = one_ring_corr_mat([-45, -15, 0, 30], M, precision=precision)
        batch_size = 100000
        onering = PerColumnModel(r_rx)

        @tf.function(jit_compile=True)
        def func():
            h = complex_normal([batch_size, M, K], precision=precision)
            h = onering(h)
            h = tf.transpose(h, [2, 0, 1])
            h = tf.expand_dims(h, -1)
            r_rx_hat = tf.reduce_mean(tf.matmul(h, h, adjoint_b=True), 1)
            return r_rx_hat

        r_rx_hat = tf.zeros_like(r_rx)
        iterations = 100
        for _ in range(iterations):
            r_rx_hat += func()/iterations
        self.assertTrue(np.allclose(r_rx, r_rx_hat, atol=1e-3))

    def test_per_example_corr(self):
        M = 16
        K = 4
        precision = "double"
        batch_size = 24
        r_rx = one_ring_corr_mat(config.np_rng.uniform(size=[batch_size, K]), M, precision=precision)
        onering = PerColumnModel(r_rx)

        @tf.function()
        def func():
            h = complex_normal([batch_size, M, K], precision=precision)
            h_corr = onering(h)
            return h, h_corr

        h, h_corr = func()
        for i in range(batch_size):
            for k in range(K):
                h_test = tf.linalg.cholesky(r_rx[i,k])@tf.expand_dims(h[i,:,k], -1)
                h_test = tf.squeeze(h_test, -1)
                self.assertTrue(np.allclose(h_corr[i,:,k], h_test))

    def test_property_setter(self):
        M = 16
        K = 4
        precision = "double"
        batch_size = 24
        onering = PerColumnModel(None)
        @tf.function()
        def func():
            h = complex_normal([batch_size, M, K], precision=precision)
            r_rx = one_ring_corr_mat(config.tf_rng.uniform([batch_size, K], -70, 70), M, precision=precision)
            onering.r_rx = r_rx
            h_corr = onering(h)
            return h, h_corr, r_rx

        h, h_corr, r_rx = func()
        for i in range(batch_size):
            for k in range(K):
                h_test = tf.linalg.cholesky(r_rx[i,k])@tf.expand_dims(h[i,:,k], -1)
                h_test = tf.squeeze(h_test, -1)
                self.assertTrue(np.allclose(h_corr[i,:,k], h_test, atol=1e-6))
