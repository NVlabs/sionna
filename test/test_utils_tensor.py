#
# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

try:
    import sionna as sn
except ImportError as e:
    import sys
    sys.path.append("../")
    import sionna as sn

from sionna.utils.tensors import matrix_sqrt, matrix_inv, matrix_sqrt_inv
from sionna.channel import exp_corr_mat

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
        a = np.random.uniform(0, 1, [2, 4, 3])
        n = 32
        R = exp_corr_mat(a, n, tf.complex128)
        R_sqrt = matrix_sqrt(R)
        R_hat = tf.matmul(R_sqrt, R_sqrt, adjoint_b=True)
        self.assertTrue(np.allclose(R, R_hat))

    def test_xla(self):
        a = np.random.uniform(0, 1, [2, 4, 3])
        n = 32
        R64 = exp_corr_mat(a, n, tf.complex64)
        R128 = exp_corr_mat(a, n, tf.complex128)
        sn.config.xla_compat=True
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
        a = np.random.uniform(0, 1, [2, 4, 3])
        n = 32
        R = exp_corr_mat(a, n, tf.complex128)
        R_inv = matrix_inv(R)
        I = tf.matmul(R, R_inv)
        self.assertTrue(np.allclose(I, exp_corr_mat(0, n, tf.complex128)))

    def test_xla(self):
        a = np.random.uniform(0, 1, [2, 4, 3])
        n = 32
        R64 = exp_corr_mat(a, n, tf.complex64)
        R128 = exp_corr_mat(a, n, tf.complex128)
        sn.config.xla_compat=True
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
        a = np.random.uniform(0, 1, [2, 4, 3])
        n = 32
        R = exp_corr_mat(a, n, tf.complex128)
        R_sqrt = matrix_sqrt(R)
        R_sqrt_inv = matrix_sqrt_inv(R)
        I = tf.matmul(R_sqrt, R_sqrt_inv)
        self.assertTrue(np.allclose(I, exp_corr_mat(0, n, tf.complex128)))

    def test_xla(self):
        a = np.random.uniform(0, 1, [2, 4, 3])
        n = 32
        R64 = exp_corr_mat(a, n, tf.complex64)
        R128 = exp_corr_mat(a, n, tf.complex128)

        sn.config.xla_compat=True
        @tf.function(jit_compile=True)
        def func(R):
            return matrix_sqrt_inv(R)

        self.assertTrue(func(R64).dtype==tf.complex64)
        self.assertTrue(func(R128).dtype==tf.complex128)
