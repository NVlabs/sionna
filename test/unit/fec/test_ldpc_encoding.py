#
# SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import unittest
import numpy as np
from os import walk # to load generator matrices from files
import re # regular expressions for generator matrix filenames
import tensorflow as tf
from sionna import config
from sionna.fec.ldpc.encoding import LDPC5GEncoder
from sionna.utils import BinarySource

class TestLDPC5GEncoder(unittest.TestCase):
    """Testcases for the LDPC5GEncoder."""

    def test_invalid_inputs(self):
        """Test against invalid values of n and k."""

        param_invalid = [[-1, 10],[10,-3],["a", 10],[3, "10"],[10,9],
                         [8500,10000],[5000, 30000]] # (k,n)
        for p in param_invalid:
            with self.assertRaises(BaseException):
                LDPC5GEncoder(p[0],p[1])

        param_valid = [[12, 20],[12,30],[1000, 1566],[364, 1013], [948, 1024],
                       [36,100], [12,18],[8448,10000]] # (k,n)
        for p in param_valid:
            LDPC5GEncoder(p[0],p[1])

    def test_output_dim(self):
        """ This test combines multiple checks to avoid unnecessary rebuilds of the graph during testing.

        a) Test that output dims are correct (=n)

        b) Test that allzero input leads to  all-zero output.

        c) Test that the systematic part is part of code
        (first 2z bits are punctured!)
        """

        bs = 10
        # (k,n)
        ks = [12, 20, 100, 1234, 2000, 6244, 8448]
        rs = [0.2, 0.34, 0.4, 0.47, 0.7, 0.85, 0.9]
        for k in ks:
            for r in rs:
                n = int(k/r)
                if k>3840 and r<1/3:
                    continue # range is officially not supported
                enc = LDPC5GEncoder(k, n)

                # a) Test for correct dimensions
                u = tf.zeros([bs, k])
                c = enc(u).numpy()
                self.assertTrue(c.shape[-1]==n)

                # b) test for all zero codeword
                c_hat = np.zeros([bs, n])
                self.assertTrue(np.array_equal(c, c_hat))

                # c) Test that systematic part (excluding first 2z pos) is
                # valid
                z = enc._z # access private attribute
                u = tf.cast(config.tf_rng.uniform([bs, k],
                                            0,
                                            2,
                                            tf.int32), tf.float32)
                c = enc(u).numpy()
                self.assertTrue(np.array_equal(u[:,2*z:],c[:,:k-2*z]))

    def test_invalid_input2(self):
        """Test that error raises if non-binary input or invalid dtype."""

        bs = 20
        k = 100
        n = 200
        u = np.zeros([bs, k])

        enc = LDPC5GEncoder(k, n)
        # test wrong datatype
        with self.assertRaises(TypeError):
            enc(tf.constant(u, dtype=tf.complex64))
        with self.assertRaises(TypeError):
            enc(tf.constant(u, dtype=tf.int32))

        # test for non-binary input
        u[13,37] = 2 # add single invalid number
        with self.assertRaises(BaseException):
            x = enc(tf.constant(u, dtype=tf.float32))

    def test_dim_mismatch(self):
        """Test that error raises if input_shape does not match k"""
        bs = 20
        k = 100
        n = 200
        enc = LDPC5GEncoder(k+1, n)
        # test for non-binary input
        with self.assertRaises(BaseException):
            x = enc(tf.zeros([bs, k]))

    def test_example_matrices(self):
        """test against reference matrices.
        """
        bs = 10 # batch_size (random samples PER generator matrix)

        # (k,n)
        ref_path = '../test/codes/ldpc/'
        f = []
        for (dirpath, dirnames, filenames) in walk(ref_path):
            f.extend(filenames)
            break
        try: # mac os adds DS_store element that should be ignored if it exists
            f.remove('.DS_Store')
        except:
            pass

        # identify all k and n parameters for automatic check
        params = []
        for s in f:
            m = re.match(r'k(.*)_n(.*)_G.npy', s)
            if m is not None: # ignore files that did not match
                params.append([int(m.group(1)), int(m.group(2))])
        # params contains a list of all (k,n) parameters

        source = BinarySource()

        for p in params:
            k = int(p[0])
            n = int(p[1])
            gm_sp = np.array(
                    np.load('../test/codes/ldpc/k{}_n{}_G.npy'.format(k, n),
                    allow_pickle=True))
            gm = np.zeros([k,n])
            for i in range(len(gm_sp[0,:])):
                c = gm_sp[0,i]
                r = gm_sp[1,i]
                gm[c-1, r-1] = 1

            u = source([bs, k]) # random info bits
            enc = LDPC5GEncoder(k, n)
            c = enc(u)

            # direct encoding
            # add dim for matrix/vect. mult.
            c_ref = tf.linalg.matmul(tf.expand_dims(u, axis=1), gm)

            c_ref = tf.math.mod(c_ref, 2)
            c_ref = tf.squeeze(c_ref) # remove new dim
            c = c.numpy()
            c_ref = c_ref.numpy()
            self.assertTrue(np.array_equal(c, c_ref),
                            "not equal for k={}, n={}".format(k, n))

    def test_multi_dimensional(self):
        """Test against arbitrary shapes.
        """
        k = 100
        n = 200
        shapes =[[10, 20, 30, k], [1, 40, k],[10, 2 ,3, 4, 3, k]]
        enc = LDPC5GEncoder(k, n)

        for s in shapes:
            source = BinarySource()
            u = source(s)
            u_ref = tf.reshape(u, [-1, k])

            c = enc(u)
            c_ref = enc(u_ref)
            s[-1] = n
            c_ref = tf.reshape(c_ref, s)
            self.assertTrue(np.array_equal(c.numpy(), c_ref.numpy()))

        # and verify that wrong last dimension raises an error
        with self.assertRaises(tf.errors.InvalidArgumentError):
            s = [10, 2, k-1]
            u = source(s)
            x = enc(u)

    def test_keras(self):
        """Test that Keras model can be compiled (supports dynamic shapes)."""
        bs = 10
        k = 100
        n = 200
        source = BinarySource()

        inputs = tf.keras.Input(shape=(k), dtype=tf.float32)
        x = LDPC5GEncoder(k, n)(inputs)
        model = tf.keras.Model(inputs=inputs, outputs=x)

        b = source([bs, k])
        model(b)
        # call twice to see that bs can change
        b2 = source([bs+1, k])
        model(b2)
        model.summary()

    def test_tf_fun(self):
        """Test that tf.function works as expected and XLA is supported"""

        @tf.function
        def run_graph(u):
            c = enc(u)
            return c

        @tf.function(jit_compile=True)
        def run_graph_xla(u):
            c = enc(u)
            return c

        k = 50
        n = 100
        bs = 10
        enc = LDPC5GEncoder(k, n)
        source = BinarySource()

        u = source([bs, k])
        run_graph(u)
        run_graph_xla(u)

    def test_dtypes_flexible(self):
        """Test that encoder supports variable dtypes and
        yields same result."""

        dt_supported = (tf.float16, tf.float32, tf.float64, tf.int8,
            tf.int32, tf.int64, tf.uint8, tf.uint16, tf.uint32)

        bs = 10
        k = 100
        n = 200

        source = BinarySource()
        enc_ref = LDPC5GEncoder(k, n, dtype=tf.float32)

        u = source([bs, k])
        c_ref = enc_ref(u)

        for dt in dt_supported:
            enc = LDPC5GEncoder(k, n, dtype=dt)
            u_dt = tf.cast(u, dt)
            c = enc(u_dt)

            c_32 = tf.cast(c, tf.float32)

            self.assertTrue(np.array_equal(c_ref.numpy(), c_32.numpy()))

    def test_ldpc_interleaver(self):
        """Test that LDPC output interleaver pattern is correct."""

        enc = LDPC5GEncoder(k=12, n=20)
        #n,m
        params = [[12,4], [100,2], [80, 8]]
        for (n,m) in params:
            s, s_inv = enc.generate_out_int(n, m)

            idx = np.arange(n)
            idx_p = idx[s]
            idx_pp = idx_p[s_inv]
            # test that interleaved vector is not the same
            self.assertFalse(np.array_equal(idx, idx_p))
            # test that interleaver can be inverted
            self.assertTrue(np.array_equal(idx, idx_pp))

        # test that for m=1 no interleaving happens
        m = 1
        for n in [10, 100, 1000]:
            s, s_inv = enc.generate_out_int(n, m)
            idx = np.arange(n)
            self.assertTrue(np.array_equal(idx, s))
            self.assertTrue(np.array_equal(idx, s_inv))
