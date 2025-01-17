#
# SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import unittest
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from sionna import config
from sionna.utils.misc import complex_normal
from sionna.utils.tensors import matrix_sqrt, matrix_inv, matrix_sqrt_inv, matrix_pinv, flatten_last_dims, flatten_dims, expand_to_rank
from sionna.channel import exp_corr_mat


class TestFlattenLastDims(unittest.TestCase):
    def test_jit_mode(self):
        """Test that all but first dim are not None"""
        class F(Model):
            def __init__(self, dims, num_dims):
                super().__init__()
                self._dims = dims
                self._num_dims = num_dims

            @tf.function(jit_compile=True)
            def call(self, batch_size):
                shape = tf.concat([[batch_size], self._dims], -1)
                x = tf.ones(shape)
                x = flatten_last_dims(x, self._num_dims)
                tf.debugging.assert_equal(x.shape[-1], tf.reduce_prod(shape[-self._num_dims:]))
                return x

        dimsl = [[100],
                 [10, 100],
                 [20, 30, 40]
                ]
        batch_size = 128
        for dims in dimsl:
            for num_dims in range(2,len(dims)+1):
                f = F(dims, num_dims)
                r = f(batch_size)
                shape = [batch_size]+dims
                self.assertEqual(r.shape[-1], np.prod(shape[-num_dims:]))

        f = F([30], 2)
        with self.assertRaises(ValueError):
            f(batch_size)

    def test_full_flatten(self):
        """Test flattening to vector"""
        class F(Model):
            def __init__(self, dims, num_dims):
                super().__init__()
                self._dims = dims
                self._num_dims = num_dims

            @tf.function(jit_compile=True)
            def call(self, batch_size):
                shape = tf.concat([[batch_size], self._dims], -1)
                x = tf.ones(shape)
                x = flatten_last_dims(x, self._num_dims)
                return x

        dimsl = [[100],
                 [10, 100],
                 [20, 30, 40]
                ]
        batch_size = 128
        for dims in dimsl:
            num_dims = len(dims)+1
            f = F(dims, num_dims)
            r = f(batch_size)
            shape = [batch_size]+dims
            self.assertEqual(r.shape[-1], np.prod(shape[-num_dims:]))

class TestFlattenDims(unittest.TestCase):
    def test_jit_mode(self):
        """Test output shapes"""
        class F(Model):
            def __init__(self, dims, num_dims, axis):
                super().__init__()
                self._dims = dims
                self._num_dims = num_dims
                self._axis = axis

            @tf.function(jit_compile=True)
            def call(self, batch_size):
                shape = tf.concat([[batch_size], self._dims], -1)
                x = tf.ones(shape)
                x = flatten_dims(x, self._num_dims, self._axis)
                return x

        dimsl = [[100],
                 [10, 100],
                 [20, 30, 40],
                 [20, 30, 40, 50]
                ]
        batch_size = 128
        for dims in dimsl:
            for axis in range(0, len(dims)+1):
                for num_dims in range(2,len(dims)+2-axis):
                    f = F(dims, num_dims, axis)
                    r = f(batch_size)
                    shape = [batch_size]+dims
                    new_shape = shape[:axis] + [np.prod(shape[axis:axis+num_dims])] + shape[axis+num_dims:]
                    self.assertEqual(r.shape, new_shape)

class TestMatrixSqrt(unittest.TestCase):
    """Unittest for the matrix_sqrt function"""
    def test_identity_matrix(self):
        n = 64
        R = tf.eye(n, dtype=tf.complex128)
        R_sqrt = matrix_sqrt(R)
        R_hat = tf.matmul(R_sqrt, R_sqrt, adjoint_b=True)
        self.assertTrue(np.allclose(R, R_hat))

    def test_single_dim(self):
        a = 0.9
        n = 64
        R = exp_corr_mat(a, n, tf.complex128)
        R_sqrt = matrix_sqrt(R)
        R_hat = tf.matmul(R_sqrt, R_sqrt, adjoint_b=True)
        self.assertTrue(np.allclose(R, R_hat))

    def test_multi_dim(self):
        a = config.np_rng.uniform(0, 1, [2, 4, 3])
        n = 32
        R = exp_corr_mat(a, n, tf.complex128)
        R_sqrt = matrix_sqrt(R)
        R_hat = tf.matmul(R_sqrt, R_sqrt, adjoint_b=True)
        self.assertTrue(np.allclose(R, R_hat))

    def test_xla(self):
        a = config.np_rng.uniform(0, 1, [2, 4, 3])
        n = 32
        R64 = exp_corr_mat(a, n, tf.complex64)
        R128 = exp_corr_mat(a, n, tf.complex128)
        config.xla_compat=True
        @tf.function(jit_compile=True)
        def func(R):
            return matrix_sqrt(R)

        self.assertTrue(func(R64).dtype==tf.complex64)
        self.assertTrue(func(R128).dtype==tf.complex128)

class TestMatrixInv(unittest.TestCase):
    """Unittest for the matrix_inv function"""
    def test_single_dim(self):
        av = [0, 0.2, 0.9, 0.99]
        n = 64
        for a in av: 
            R = exp_corr_mat(a, n, tf.complex128)
            R_inv = matrix_inv(R)
            I = tf.matmul(R, R_inv)
            self.assertTrue(np.allclose(I, tf.eye(n, dtype=R.dtype)))

    def test_multi_dim(self):
        a = config.np_rng.uniform(0, 1, [2, 4, 3])
        n = 32
        R = exp_corr_mat(a, n, tf.complex128)
        R_inv = matrix_inv(R)
        I = tf.matmul(R, R_inv)
        self.assertTrue(np.allclose(I, exp_corr_mat(0, n, tf.complex128)))

    def test_xla(self):
        a = config.np_rng.uniform(0, 1, [2, 4, 3])
        n = 32
        R64 = exp_corr_mat(a, n, tf.complex64)
        R128 = exp_corr_mat(a, n, tf.complex128)
        config.xla_compat=True
        @tf.function(jit_compile=True)
        def func(R):
            return matrix_inv(R)

        self.assertTrue(func(R64).dtype==tf.complex64)
        self.assertTrue(func(R128).dtype==tf.complex128)

class TestMatrixSqrtInv(unittest.TestCase):
    """Unittest for the matrix_sqrt_inv function"""

    def test_single_dim(self):
        av = [0, 0.2, 0.9, 0.99]
        n = 64
        for a in av: 
            R = exp_corr_mat(a, n, tf.complex128)
            R_sqrt = matrix_sqrt(R)
            R_sqrt_inv = matrix_sqrt_inv(R)
            I = tf.matmul(R_sqrt, R_sqrt_inv)
            self.assertTrue(np.allclose(I, tf.eye(n, dtype=R.dtype)))

    def test_multi_dim(self):
        a = config.np_rng.uniform(0, 1, [2, 4, 3])
        n = 32
        R = exp_corr_mat(a, n, tf.complex128)
        R_sqrt = matrix_sqrt(R)
        R_sqrt_inv = matrix_sqrt_inv(R)
        I = tf.matmul(R_sqrt, R_sqrt_inv)
        self.assertTrue(np.allclose(I, exp_corr_mat(0, n, tf.complex128)))

    def test_xla(self):
        a = config.np_rng.uniform(0, 1, [2, 4, 3])
        n = 32
        R64 = exp_corr_mat(a, n, tf.complex64)
        R128 = exp_corr_mat(a, n, tf.complex128)

        config.xla_compat=True
        @tf.function(jit_compile=True)
        def func(R):
            return matrix_sqrt_inv(R)

        self.assertTrue(func(R64).dtype==tf.complex64)
        self.assertTrue(func(R128).dtype==tf.complex128)

class TestMatrixPinv(unittest.TestCase):
    """Unittest for the matrix_pinv function"""
    def test_single_dim(self):
        av = [0, 0.2, 0.9, 0.99]
        n = 64
        for a in av: 
            A = complex_normal([n, n//2], dtype=tf.complex128)
            A_pinv = matrix_pinv(A)
            I = tf.matmul(A_pinv, A)
            self.assertTrue(np.allclose(I, tf.eye(n//2, dtype=A.dtype)))

    def test_multi_dim(self):
        a = [2, 4, 3]
        n = 32
        A = complex_normal(a + [n, n//2], dtype=tf.complex128)
        A_pinv = matrix_pinv(A)
        I = tf.matmul(A_pinv, A)
        I_target = tf.eye(n//2, dtype=A.dtype)
        I_target = expand_to_rank(I_target, tf.rank(I), 0)
        self.assertTrue(np.allclose(I, I_target))

    def test_xla(self):
        a = [2, 4, 3]
        n = 32
        A64 = complex_normal(a + [n, n//2], dtype=tf.complex64)
        A128 = complex_normal(a + [n, n//2], dtype=tf.complex128)
        config.xla_compat=True
        @tf.function(jit_compile=True)
        def func(A):
            return matrix_pinv(A)

        self.assertTrue(func(A64).dtype==tf.complex64)
        self.assertTrue(func(A128).dtype==tf.complex128)
