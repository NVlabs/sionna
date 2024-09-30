#
# SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import os
import unittest
import numpy as np
import tensorflow as tf
from sionna.fec.polar.encoding import PolarEncoder, Polar5GEncoder
from sionna.utils import BinarySource
from sionna.fec.polar.utils import generate_5g_ranking, generate_polar_transform_mat

current_dir = os.path.dirname(os.path.abspath(__file__))
test_dir = os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir))

class TestPolarEncoding(unittest.TestCase):
    """Testcases for the PolarEncoder layer."""

    def test_invalid_inputs(self):
        """Test against invalid values of n and k."""

        # (k, n, k_des, n_des)...last two entries are just used to generate
        # valid channel rankings
        param_invalid = [[-1, 10, 1, 32], [10,-3, 10, 32], ["1.0", 10, 1, 32],
                        [3, "10.", 3, 32], [10, 9, 10, 32]]

        for p in param_invalid:
            frozen_pos,_ = generate_5g_ranking(p[2], p[3])
            with self.assertRaises(AssertionError):
                PolarEncoder(frozen_pos, p[1])

        # no complex-valued input allowed
        with self.assertRaises(ValueError):
            frozen_pos,_ = generate_5g_ranking(32, 64)
            PolarEncoder(frozen_pos, 64, dtype=tf.complex64)

        # test also valid shapes
        # (k, n)
        param_valid = [[0, 32], [10, 32], [32, 32], [100, 256],
                       [123, 1024], [1024, 1024]]

        for p in param_valid:
            frozen_pos, _ = generate_5g_ranking(p[0], p[1])
            PolarEncoder(frozen_pos, p[1])

    def test_output_dim(self):
        """Test that output dims are correct (=n) and output is all-zero
         codeword."""

        bs = 10

        # (k, n)
        param_valid = [[1, 32], [10, 32], [32, 32], [100, 256], [123, 1024],
                      [1024, 1024]]

        for p in param_valid:
            frozen_pos, _ = generate_5g_ranking(p[0], p[1])
            enc = PolarEncoder(frozen_pos, p[1])
            u = np.zeros([bs, p[0]])
            c = enc(u).numpy()
            self.assertTrue(c.shape[-1]==p[1])

            # also check that all-zero input yields all-zero output
            c_hat = np.zeros([bs, p[1]])
            self.assertTrue(np.array_equal(c, c_hat))

    def test_keras(self):
        """Test that Keras model can be compiled (supports dynamic shapes)."""

        bs = 10
        k = 100
        n = 128
        source = BinarySource()
        frozen_pos, _ = generate_5g_ranking(k, n)
        inputs = tf.keras.Input(shape=(k), dtype=tf.float32)
        x = PolarEncoder(frozen_pos, n)(inputs)
        model = tf.keras.Model(inputs=inputs, outputs=x)

        b = source([bs,k])
        model(b)
        # call twice to see that bs can change
        b2 = source([bs+1,k])
        model(b2)
        model.summary()

    def test_multi_dimensional(self):
        """Test against multi-dimensional shapes."""

        k = 120
        n = 256

        frozen_pos, _ = generate_5g_ranking(k, n)
        source = BinarySource()
        enc = PolarEncoder(frozen_pos, n)

        b = source([100, k])
        b_res = tf.reshape(b, [4, 5, 5, k])

        # encode 2D Tensor
        c = enc(b).numpy()
        # encode 4D Tensor
        c_res = enc(b_res).numpy()
        # and reshape to 2D shape
        c_res = tf.reshape(c_res, [100, n])
        # both version should yield same result
        self.assertTrue(np.array_equal(c, c_res))


    def test_tf_fun(self):
        """Test that graph mode works and XLA is supported."""

        @tf.function
        def run_graph(u):
            return enc(u)

        @tf.function(jit_compile=True)
        def run_graph_xla(u):
            return enc(u)

        bs = 10
        k = 100
        n = 128
        source = BinarySource()
        frozen_pos, _ = generate_5g_ranking(k, n)
        enc = PolarEncoder(frozen_pos, n)

        # test that for arbitrary input only 0,1 values are returned
        u = source([bs, k])
        x = run_graph(u).numpy()

        # execute the graph twice
        x = run_graph(u).numpy()

        # and change batch_size
        u = source([bs+1, k])
        x = run_graph(u).numpy()

        #check XLA
        x = run_graph_xla(u).numpy()
        u = source([bs, k])
        x = run_graph_xla(u).numpy()

    def test_ref_implementation(self):
        """Test channel rankings against reference implementation based on
        polar transform matrix.
        """

        bs = 10
        k = 12
        n = 32

        frozen_pos, info_pos = generate_5g_ranking(k, n)
        source = BinarySource()
        enc = PolarEncoder(frozen_pos, n)

        b = source([bs, k]).numpy() # perform array ops in numpy
        u = np.zeros([bs, n])
        u[:, info_pos] = b

        # call reference implementation
        c_ref = np.zeros([bs, n])
        gen_mat = generate_polar_transform_mat(int(np.log2(n)))
        gen_mat = tf.expand_dims(gen_mat, axis=0)

        u = tf.expand_dims(u, axis=1)
        c_ref = tf.linalg.matmul(u, gen_mat)
        c_ref = tf.math.mod(c_ref, 2)
        c_ref = tf.squeeze(c_ref, axis=1)

        # and run tf version (to be tested)
        c = enc(b).numpy()

        self.assertTrue(np.array_equal(c, c_ref))

    def test_batch(self):
        """Test that all samples in batch yield same output (for same input).
        """
        bs = 100
        k = 120
        n = 256

        frozen_pos, _ = generate_5g_ranking(k, n)
        source = BinarySource()
        enc = PolarEncoder(frozen_pos, n)

        b = source([1, 15, k])
        b_rep = tf.tile(b, [bs, 1, 1])
        c = enc(b_rep).numpy()

        for i in range(bs):
            self.assertTrue(np.array_equal(c[0,:,:], c[i,:,:]))

    def test_dtypes_flexible(self):
        """Test that encoder supports variable dtypes and
        yields same result."""

        dt_supported = (tf.float16, tf.float32, tf.float64, tf.int8,
            tf.int32, tf.int64, tf.uint8, tf.uint16, tf.uint32)

        bs = 10
        k = 32
        n = 64

        source = BinarySource()
        frozen_pos, _ = generate_5g_ranking(k, n)
        enc_ref = PolarEncoder(frozen_pos, n, dtype=tf.float32)

        u = source([bs, k])
        c_ref = enc_ref(u)

        for dt in dt_supported:
            enc = PolarEncoder(frozen_pos, n, dtype=dt)
            u_dt = tf.cast(u, dt)
            c = enc(u_dt)

            c_32 = tf.cast(c, tf.float32)

            self.assertTrue(np.array_equal(c_ref.numpy(), c_32.numpy()))


class TestPolarEncoding5G(unittest.TestCase):
    """Test 5G encoder including rate-matching.

    Remark: the layer inherits from PolarEncoder, thus many basic tests are
    already covered by the previous testcases."""

    def test_invalid_inputs(self):
        """Test against invalid values of n and k according to 38.212."""

        # (k, n)
        param_invalid = [[-1, 30], # negative k
                         [12, -3], # negative n
                         ["12.", 30], # non int k
                         [3, "10."], # non int n
                         [10, 9], # r>1
                         [10, 32], # k too small
                         [10, 10], # n too small
                         [1014, 1040], # k too large
                         [1000, 1100], # n too large
                         [100, 110]] # k+k_crc>n

        for p in param_invalid:
            with self.assertRaises((AssertionError, ValueError)):
                Polar5GEncoder(p[0], p[1])

        # no complex-valued input allowed
        with self.assertRaises(ValueError):
            Polar5GEncoder(32, 64, dtype=tf.complex64)

    def test_output_dim(self):
        """Test that output dims are correct (=n) and output is all-zero
         codeword for all-zero inputs."""

        bs = 10
        # (k, n)
        param_valid = [[12, 32], [20, 32], [100, 256],
                       [243, 1024], [1013, 1088]]

        for p in param_valid:
            enc = Polar5GEncoder(p[0], p[1])
            u = np.zeros([bs, p[0]])
            c = enc(u).numpy()
            self.assertTrue(c.shape[-1]==p[1])
            # also check that all-zero input yields all-zero output
            c_hat = np.zeros_like(c)
            self.assertTrue(np.array_equal(c, c_hat))

    def test_keras(self):
        """Test that Keras model can be compiled (supports dynamic shapes)."""

        bs = 10
        k = 45
        n = 67
        source = BinarySource()
        inputs = tf.keras.Input(shape=(k), dtype=tf.float32)
        x = Polar5GEncoder(k, n)(inputs)
        model = tf.keras.Model(inputs=inputs, outputs=x)

        b = source([bs, k])
        model(b)
        # call twice to see that bs can change
        b2 = source([bs+1, k])
        model(b2)
        model.summary()

    def test_multi_dimensional(self):
        """Test against arbitrary shapes."""

        k = 56
        n = 240

        source = BinarySource()
        enc = Polar5GEncoder(k, n)

        b = source([100, k])
        b_res = tf.reshape(b, [4, 5, 5, k])

        # encode 2D Tensor
        c = enc(b).numpy()
        # encode 4D Tensor
        c_res = enc(b_res).numpy()
        # and reshape to 2D shape
        c_res = tf.reshape(c_res, [100, n])
        # both version should yield same result
        self.assertTrue(np.array_equal(c, c_res))


    def test_tf_fun(self):
        """Test that graph mode works and XLA is supported."""

        @tf.function
        def run_graph(u):
            return enc(u)

        @tf.function(jit_compile=True)
        def run_graph_xla(u):
            return enc(u)

        bs = 10
        k = 100
        n = 135
        source = BinarySource()
        enc = Polar5GEncoder(k, n)

        # test that for arbitrary input only 0,1 values are returned
        u = source([bs, k])
        x = run_graph(u).numpy()

        # execute the graph twice
        x = run_graph(u).numpy()

        # and change batch_size
        u = source([bs+1, k])
        x = run_graph(u).numpy()

        # check XLA
        x = run_graph_xla(u).numpy()
        u = source([bs, k])
        x = run_graph_xla(u).numpy()

    def test_batch(self):
        """Test that all samples in batch yield same output (for same input).
        """
        bs = 100
        k = 120
        n = 253

        source = BinarySource()
        enc = Polar5GEncoder(k, n)

        b = source([1, 15, k])
        b_rep = tf.tile(b, [bs, 1, 1])

        # and run tf version (to be tested)
        c = enc(b_rep).numpy()

        for i in range(bs):
            self.assertTrue(np.array_equal(c[0,:,:], c[i,:,:]))

    def test_ref_implementation(self):
        """Test against pre-generated test cases.
        The test-cases include CRC-encoding and rate-matching and
        cover puncturing, shortening and repetition coding.
        """

        ref_path = test_dir + '/codes/polar/'
        filename = ['E45_k30_K41',
                    'E70_k32_K43',
                    'E127_k29_K40',
                    'E1023_k400_K411',
                    'E70_k28_K39']

        for f in filename:
            # load random info bits
            u = np.load(ref_path + f + "_u.npy")
            # load reference codewords
            c_ref = np.load(ref_path + f + "_c.npy")

            # restore dimensions
            k = u.shape[1]
            n = c_ref.shape[1]

            # encode u with Sionna encoder
            enc = Polar5GEncoder(k, n)
            c = enc(u)

            # and compare results
            self.assertTrue(np.array_equal(c, c_ref))

    def test_dtypes_flexible(self):
        """Test that encoder supports variable dtypes and
        yields same result for all dtypes."""

        dt_supported = (tf.float16, tf.float32, tf.float64, tf.int8,
            tf.int32, tf.int64, tf.uint8, tf.uint16, tf.uint32)

        bs = 10
        k = 32
        n = 64

        source = BinarySource()
        enc_ref = Polar5GEncoder(k, n, dtype=tf.float32)

        u = source([bs, k])
        c_ref = enc_ref(u)

        for dt in dt_supported:
            enc = Polar5GEncoder(k, n, dtype=dt)
            u_dt = tf.cast(u, dt)
            c = enc(u_dt)

            c_32 = tf.cast(c, tf.float32)

            self.assertTrue(np.array_equal(c_ref.numpy(), c_32.numpy()))


