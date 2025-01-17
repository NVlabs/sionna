#
# SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
import pytest
import unittest
import numpy as np
import tensorflow as tf
import sionna
from sionna import config
from sionna.mimo.utils import complex2real_vector, real2complex_vector
from sionna.mimo.utils import complex2real_matrix, real2complex_matrix
from sionna.mimo.utils import complex2real_covariance, real2complex_covariance
from sionna.mimo.utils import complex2real_channel, whiten_channel
from sionna.utils import matrix_pinv
from sionna.utils import matrix_sqrt, complex_normal
from sionna.channel.utils import exp_corr_mat
from sionna.utils import QAMSource


class Complex2Real(unittest.TestCase):

    def test_vector(self):
        shapes = [
                  [1],
                  [20,1],
                  [30,20],
                  [30,20,40]
                 ]
        for shape in shapes:
            z = config.tf_rng.uniform(shape)
            x = tf.math.real(z)
            y = tf.math.imag(z)

            # complex2real transformation
            zr = complex2real_vector(z)
            x_, y_ = tf.split(zr, 2, -1)
            self.assertTrue(np.array_equal(x, x_))
            self.assertTrue(np.array_equal(y, y_))

            # real2complex transformation
            zc = real2complex_vector(zr)
            self.assertTrue(np.array_equal(z, zc))

    def test_matrix(self):
        shapes = [
                  [1,1],
                  [20,1],
                  [1, 20],
                  [30,20],
                  [30,20,40],
                  [12, 45, 64, 42]
                 ]
        for shape in shapes:
            h = config.tf_rng.uniform(shape)
            h_r = tf.math.real(h)
            h_i = tf.math.imag(h)

            # complex2real transformation
            hr = complex2real_matrix(h)
            self.assertTrue(np.array_equal(h_r, hr[...,:shape[-2], :shape[-1]]))
            self.assertTrue(np.array_equal(h_r, hr[...,shape[-2]:, shape[-1]:]))
            self.assertTrue(np.array_equal(h_i, hr[...,shape[-2]:, :shape[-1]]))
            self.assertTrue(np.array_equal(-h_i, hr[...,:shape[-2], shape[-1]:]))

            # real2complex transformation
            hc = real2complex_matrix(hr)
            self.assertTrue(np.array_equal(h, hc))

    def test_covariance(self):
        ns = [1, 2, 5, 13]
        batch_dims = [
                  [1],
                  [5, 30],
                  [4,5,10]
                 ]
        for shape in batch_dims:
            for n in ns:
                a = config.tf_rng.uniform(shape,minval=0, maxval=1)
                r = exp_corr_mat(a, n)
                r_r = tf.math.real(r)/2
                r_i = tf.math.imag(r)/2

                # complex2real transformation
                rr = complex2real_covariance(r)
                self.assertTrue(np.allclose(r_r, rr[...,:n, :n]))
                self.assertTrue(np.allclose(r_r, rr[...,n:, n:]))
                self.assertTrue(np.allclose(r_i, rr[...,n:, :n]))
                self.assertTrue(np.allclose(-r_i, rr[...,:n, n:]))

                # real2complex transformation
                rc = real2complex_covariance(rr)
                self.assertTrue(np.allclose(r, rc))

    def test_covariance_statistics(self):
        """Test that the statisics of the real-valued equivalent random
           vector match the target statistics"""
        batch_size = 1000000
        num_batches = 100
        n = 8
        r = exp_corr_mat(0.8, 8)
        rr = complex2real_covariance(r)
        r_12 = matrix_sqrt(r)

        @tf.function(jit_compile=True)
        def fun():
            w = tf.matmul(r_12, complex_normal([n, batch_size]))
            w = tf.transpose(w, perm=[1,0])
            wr = complex2real_vector(w)
            r_hat = tf.matmul(wr, wr, transpose_a=True)/tf.cast(batch_size, wr.dtype)
            return r_hat
        r_hat = tf.zeros_like(rr)
        for _ in range(num_batches):
            r_hat += fun()/num_batches

        self.assertTrue(np.max(np.abs(rr-r_hat))<1e-3)

    @pytest.mark.usefixtures("only_gpu")
    def test_whiten_channel_noise_covariance(self):
        # Generate channel outputs
        num_rx = 16
        num_tx = 4
        batch_size = 1000000
        qam_source = QAMSource(8, dtype=tf.complex128)

        r = exp_corr_mat(0.8, num_rx, dtype=tf.complex128)
        r_12 = matrix_sqrt(r)
        s = exp_corr_mat(0.5, num_rx, dtype=tf.complex128) + tf.eye(num_rx, dtype=tf.complex128)
        s_12 = matrix_sqrt(s)

        sionna.config.xla_compat = True
        @tf.function(jit_compile=True)
        def fun():
            x = qam_source([batch_size, num_tx, 1])
            h = tf.matmul(tf.expand_dims(r_12,0), complex_normal([batch_size, num_rx, num_tx], dtype=tf.complex128))
            w = tf.squeeze(tf.matmul(tf.expand_dims(s_12, 0), complex_normal([batch_size, num_rx, 1], dtype=tf.complex128)), -1)
            hx = tf.squeeze(tf.matmul(h, x), -1)
            y = hx+w

            # Compute noise error after whitening the complex channel
            yw, hw, sw = whiten_channel(y, h, s)
            hwx = tf.squeeze(tf.matmul(hw, x), -1)
            ww = yw - hwx
            err_w = tf.matmul(ww, ww, adjoint_a=True)/tf.cast(batch_size, ww.dtype) - sw

            # Compute noise error after whitening the real valued channel
            yr, hr, sr = complex2real_channel(y, h, s)
            yrw, hrw, srw = whiten_channel(yr, hr, sr)
            xr = tf.expand_dims(complex2real_vector(x[...,0]), -1)
            hrwxr = tf.squeeze(tf.matmul(hrw, xr), -1)
            wrw = yrw - hrwxr
            err_rw = tf.matmul(wrw, wrw, transpose_a=True)/tf.cast(batch_size, wrw.dtype) - srw

            # Compute noise covariance after transforming the complex whitened channel to real
            ywr, hwr, swr = complex2real_channel(yw, hw, sw)
            hwrxr = tf.squeeze(tf.matmul(hwr, xr), -1)
            wwr = ywr - hwrxr
            err_wr = tf.matmul(wwr, wwr, transpose_a=True)/tf.cast(batch_size, wwr.dtype) - swr
            return err_w, err_rw, err_wr

        num_iterations = 100
        for i in range(num_iterations):
            if i==0:
                err_w, err_rw, err_wr = [e/num_iterations for e in fun()]
            else:
                a, b, c = fun()
                err_w += a/num_iterations
                err_rw += b/num_iterations
                err_wr += c/num_iterations
        self.assertTrue(np.max(np.abs(err_w))<1e-3)
        self.assertTrue(np.max(np.abs(err_rw))<1e-3)
        self.assertTrue(np.max(np.abs(err_wr))<1e-3)

    @pytest.mark.usefixtures("only_gpu")
    def test_whiten_channel_symbol_recovery(self):
        """Check that the whitened channel can be used to receover the symbols"""
        # Generate channel outputs
        num_rx = 16
        num_tx = 4
        batch_size = 1000000
        qam_source = QAMSource(8, dtype=tf.complex128)
        s = exp_corr_mat(0.5, num_rx, dtype=tf.complex128) + tf.eye(num_rx, dtype=tf.complex128)
        s_12 = matrix_sqrt(s)
        r = exp_corr_mat(0.8, num_rx, dtype=tf.complex128)
        r_12 = matrix_sqrt(r)

        sionna.config.xla_compat = True
        @tf.function(jit_compile=True)
        def fun():
            # Noise free transmission
            x = qam_source([batch_size, num_tx, 1])
            h = tf.matmul(tf.expand_dims(r_12,0), complex_normal([batch_size, num_rx, num_tx], dtype=tf.complex128))
            hx = tf.squeeze(tf.matmul(h, x), -1)
            y = hx

            # Compute symbol error on detection on the complex whitened channel
            yw, hw, sw = whiten_channel(y, h, s)
            xw = tf.matmul(matrix_pinv(hw), tf.expand_dims(yw,-1))

            err_w = tf.reduce_mean(x - xw, axis=0)
            return err_w
        err_w = fun()
        self.assertTrue(np.max(np.abs(err_w))<1e-6)






