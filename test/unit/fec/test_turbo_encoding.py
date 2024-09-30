#
# SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
import os
import unittest
import numpy as np
import tensorflow as tf
from sionna.fec.turbo import TurboEncoder
from sionna.utils import BinarySource

current_dir = os.path.dirname(os.path.abspath(__file__))
test_dir = os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir))

class TestTurboEncoding(unittest.TestCase):

    def test_output_dim(self):
        r"""Test with allzero codeword that output dims are correct (=n) and output also equals all-zero."""

        bs = 10
        coderates = [1/2, 1/3]
        ks = [10, 20, 50, 100]
        cl = 5 # constraint length

        for rate in coderates:
            for k in ks:
                for t in [False, True]: # termination

                    n = int(k/rate) # calculate coderate
                    if t:
                        n += int(cl/rate) # account for additional constraint length

                    enc = TurboEncoder(rate=rate,
                                       constraint_length=cl,
                                       terminate=t)
                    u = np.zeros([bs, k])
                    c = enc(u).numpy()

                    # if no termination is used, the output must be k/r
                    if t is False:
                        self.assertTrue(c.shape[-1]==n)

                    # verify that coderate is correct
                    # allow small epsilon due to rounding
                    self.assertTrue(enc.coderate-k/c.shape[-1]<1e-6)

                    # also check that all-zero input yields all-zero output
                    c_hat = np.zeros_like(c)
                    self.assertTrue(np.array_equal(c, c_hat))

                    # test that output dim can change (in eager mode)
                    k += 1 # increase length
                    n = int(k/rate) # calculate coderate
                    u = np.zeros([bs, k])
                    c = enc(u).numpy()

                    # if no termination is used, the output must be k/r
                    if t is False:
                        self.assertTrue(c.shape[-1]==n)

                    # also check that all-zero input yields all-zero output
                    c_hat = np.zeros_like(c)
                    self.assertTrue(np.array_equal(c, c_hat))

                    # verify that coderate is correctly updated
                    # allow small epsilon due to rounding
                    self.assertTrue(enc.coderate-k/c.shape[-1]<1e-6)

    def test_invalid_inputs(self):
        r"""Test with invalid rate values and invalid constraint lengths as
        input. Only rates [1/2, 1/3] and constraint lengths [3, 4, 5, 6]
        are accepted currently."""

        rate_invalid = [0.2, 0.45, 0.01]
        rate_valid = [1/3, 1/2]

        constraint_length_invalid = [2, 9, 0]
        constraint_length_valid = [3, 4, 5, 6]
        for rate in rate_valid:
            for mu in constraint_length_invalid:
                with self.assertRaises(AssertionError):
                    enc = TurboEncoder(rate=rate, constraint_length=mu)

        for rate in rate_invalid:
            for mu in constraint_length_valid:
                with self.assertRaises(AssertionError):
                    enc = TurboEncoder(rate=rate, constraint_length= mu)

        gmat = [['101', '111', '000'], ['000', '010', '011']]
        with self.assertRaises(AssertionError):
            enc = TurboEncoder(gen_poly=gmat)

    def test_polynomial_input(self):
        r"""Test that different formats of input polynomials are accepted and raises exceptions when the generator polynomials fail assertions."""

        bs = 10
        k = 100
        rate = 1/2
        n = int(k/rate) # calculate coderate
        u = np.zeros([bs, k])

        g1 = ['101', '111']
        g2 = np.array(g1)

        g = [g1, g2]
        for gen_poly in g:
            enc = TurboEncoder(gen_poly=gen_poly, rate=rate, terminate=False)
            c = enc(u).numpy()
            self.assertTrue(c.shape[-1]==n)
            # also check that all-zero input yields all-zero output
            c_hat = np.zeros([bs, n])
            self.assertTrue(np.array_equal(c, c_hat))

        def util_check_assertion_err(gen_poly_, msg_):
            with self.assertRaises(AssertionError) as exception_context:
                enc = TurboEncoder(gen_poly=gen_poly_)
                self.assertEqual(str(exception_context.exception), msg_)

        gs = [
            ['1001', '111'],
            ['1001', 111],
            ('1211', '1101')]
        msg_s = [
            "Each polynomial must be of same length.",
                 "Each polynomial must be a string.",
                 "Each Polynomial must be a string of 0/1 s."
                 ]
        for idx, g in enumerate(gs):
            util_check_assertion_err(g,msg_s[idx])

    def test_keras(self):
        """Test that Keras model can be compiled (+supports dynamic shapes)."""
        bs = 10
        k = 100

        source = BinarySource()
        inputs = tf.keras.Input(shape=(k), dtype=tf.float32)
        x = TurboEncoder(rate=0.5, constraint_length=4)(inputs)
        model = tf.keras.Model(inputs=inputs, outputs=x)

        b = source([bs, k])
        model(b)
        # call twice to see that bs can change
        b2 = source([bs+1, k])
        model(b2)

        model.summary()

        source = BinarySource()
        enc = TurboEncoder(rate=0.5, constraint_length=5)
        u = source([1, 32])
        x = enc(u)
        u = source([2, 30])
        x = enc(u)

    def test_multi_dimensional(self):
        """Test against arbitrary shapes
        """
        k = 120
        n = 240 # rate must be 1/2 or 1/3

        source = BinarySource()
        enc = TurboEncoder(rate=k/n, constraint_length=5, terminate=False)

        b = source([100, k])
        b_res = tf.reshape(b, [4, 5, 5, k])

        # encode 2D Tensor
        c = enc(b).numpy()
        # encode 4D Tensor
        c_res = enc(b_res).numpy()
        # test that shape was preserved
        self.assertTrue(c_res.shape[:-1]==b_res.shape[:-1])

        # and reshape to 2D shape
        c_res = tf.reshape(c_res, [100,n])
        # both version should yield same result
        self.assertTrue(np.array_equal(c, c_res))

    def test_batch(self):
        """Test that all samples in batch yield same output (for same input).
        """
        bs = 100
        k = 120

        source = BinarySource()
        enc = TurboEncoder(rate=0.5, constraint_length=6)

        b = source([1, 15, k])
        b_rep = tf.tile(b, [bs, 1, 1])

        # and run tf version (to be tested)
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

        source = BinarySource()

        enc_ref = TurboEncoder(rate=0.5,
                              constraint_length=6,
                              output_dtype=tf.float32)

        u = source([bs, k])
        c_ref = enc_ref(u)

        for dt in dt_supported:
            enc = TurboEncoder(rate=0.5,
                              constraint_length=6,
                              output_dtype=dt)
            u_dt = tf.cast(u, dt)
            c = enc(u_dt)

            c_32 = tf.cast(c, tf.float32)

            self.assertTrue(np.array_equal(c_ref.numpy(), c_32.numpy()))

    def test_tf_fun(self):
        """Test that tf.function decorator works and XLA is supported"""

        bs = 10
        k = 100

        source = BinarySource()

        for t in [False, True]:

            enc = TurboEncoder(rate=0.5, constraint_length=6, terminate=t)

            @tf.function
            def run_graph(u):
                return enc(u)

            @tf.function(jit_compile=True)
            def run_graph_xla(u):
                return enc(u)

            # test that for arbitrary input only 0,1 values are outputed
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
        r"""Test against pre-encoded codewords from reference implementation.
        """
        ref_path = test_dir + '/codes/turbo/'
        ks = [40, 112, 168, 432]
        enc = TurboEncoder(rate=1/3, terminate=True, constraint_length=4)

        for k in ks:
            uref = np.load(ref_path + 'ref_k{}_u.npy'.format(k))
            cref = np.load(ref_path + 'ref_k{}_x.npy'.format(k))
            c = enc(uref).numpy()
            self.assertTrue(np.array_equal(c, cref))
