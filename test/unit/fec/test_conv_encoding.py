#
# SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
import os
from itertools import product
import unittest
import numpy as np
import tensorflow as tf
from sionna.fec.conv import ConvEncoder
from sionna.utils import BinarySource

current_dir = os.path.dirname(os.path.abspath(__file__))
test_dir = os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir))

class TestConvEncoding(unittest.TestCase):

    def test_output_dim(self):
        r"""Test with allzero codeword that output dims are correct (=n) and output also equals all-zero."""

        bs = 10
        mu = 4
        coderates = [1/2, 1/3]
        ks = [10, 20, 50, 100]

        for rate, k in product(coderates, ks):
            n = int(k/rate) # calculate coderate
            for enc in (
                ConvEncoder(rate=rate, constraint_length=mu+1),
                ConvEncoder(rate=rate, constraint_length=mu+1, rsc=True),
                ConvEncoder(rate=rate, constraint_length=mu+1, terminate=True)):

                if enc.terminate:
                    n+= int(mu/rate)

                u = np.zeros([bs, k])
                c = enc(u).numpy()
                self.assertTrue(c.shape[-1]==n)
                # also check that all-zero input yields all-zero output
                c_hat = np.zeros([bs, n])
                self.assertTrue(np.array_equal(c, c_hat))

                # test that output dim can change (in eager mode)
                k = k+1 # increase length
                n = int(k/rate) # calculate coderate
                if enc.terminate is True:
                    n+= int(mu/rate)
                u = np.zeros([bs, k])
                c = enc(u).numpy()
                self.assertTrue(c.shape[-1]==n)

                # also check that all-zero input yields all-zero output
                c_hat = np.zeros([bs, n])
                self.assertTrue(np.array_equal(c, c_hat))

    def test_invalid_inputs(self):
        r"""Test with invalid rate values and invalid constraint lengths as input.
        Only rates [1/2, 1/3] and constraint lengths [3, 4, 5, 6, 7, 8] are accepted currently."""
        rate_invalid = [0.2, 0.45, 0.01]
        rate_valid = [1/3, 1/2]

        constraint_length_invalid = [2, 9, 0]
        constraint_length_valid = [3, 4, 5, 6, 7, 8]
        for rate in rate_valid:
            for mu in constraint_length_invalid:
                with self.assertRaises(AssertionError):
                    enc = ConvEncoder(rate=rate, constraint_length=mu)

        for rate in rate_invalid:
            for mu in constraint_length_valid:
                with self.assertRaises(AssertionError):
                    enc = ConvEncoder(rate=rate, constraint_length= mu)
                    enc = ConvEncoder(rate=rate, rsc=True)
                    enc = ConvEncoder(rate=rate, terminate=True)
                    enc = ConvEncoder(rate=rate, rsc=True, terminate=True)

        gmat = [['101', '111', '000'], ['000', '010', '011']]
        with self.assertRaises(AssertionError):
            enc = ConvEncoder(gen_poly=gmat)
            enc = ConvEncoder(gen_poly=gmat, rsc=True)

    def test_polynomial_input(self):
        r"""Test that different formats of input polynomials are accepted and raises exceptions when the generator polynomials fail assertions."""

        def util_check_assertion_err(gen_poly_, msg_):
            with self.assertRaises(AssertionError) as exception_context:
                enc = ConvEncoder(gen_poly=gen_poly_)
                self.assertEqual(str(exception_context.exception), msg_)

        bs = 10
        k = 100
        rate = 1/2
        n = int(k/rate) # calculate coderate
        u = np.zeros([bs, k])

        g1 = ['101', '111']
        g2 = np.array(g1)

        g = [g1, g2]
        for gen_poly in g:
            for enc in (
                ConvEncoder(gen_poly=gen_poly),
                ConvEncoder(gen_poly=gen_poly, rsc=True)):

                c = enc(u).numpy()
                self.assertTrue(c.shape[-1]==n)
                # also check that all-zero input yields all-zero output
                c_hat = np.zeros([bs, n])
                self.assertTrue(np.array_equal(c, c_hat))

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

        x = ConvEncoder(rate=0.5, constraint_length=4)(inputs)
        model = tf.keras.Model(inputs=inputs, outputs=x)

        xterm = ConvEncoder(rate=1/3, constraint_length=3, terminate=True)(inputs)
        modelterm = tf.keras.Model(inputs=inputs, outputs=xterm)

        b = source([bs, k])
        model(b)
        modelterm(b)

        # call twice to see that bs can change
        b2 = source([bs+1, k])
        model(b2)
        modelterm(b2)

        model.summary()
        modelterm.summary()

        source = BinarySource()
        enc = ConvEncoder(rate=0.5, constraint_length=6)
        u = source([1, 32])
        x = enc(u)
        self.assertTrue(x.shape == [1,64])

        u = source([2, 30])
        x = enc(u)
        self.assertTrue(x.shape == [2,60])

    def test_multi_dimensional(self):
        """Test against arbitrary shapes
        """
        k = 120
        rate = 1/2
        n = int(k/rate)
        mu = 4

        source = BinarySource()
        for enc in (
            ConvEncoder(rate=rate, constraint_length=mu+1),
            ConvEncoder(rate=rate, constraint_length=mu+1, terminate=True)):

            if enc.terminate:
                n += int(mu/rate)

            b = source([100, k])
            b_res = tf.reshape(b, [4, 5, 5, k])

            # encode 2D Tensor
            c = enc(b).numpy()
            # encode 4D Tensor
            c_res = enc(b_res).numpy()

            # test that shape was preserved
            self.assertTrue(c_res.shape[:-1]==b_res.shape[:-1])

            # and reshape to 2D shape
            c_res = tf.reshape(c_res, [100, n])
            # both version should yield same result
            self.assertTrue(np.array_equal(c, c_res))

    def test_ref_implementation(self):
        r"""Test against pre-encoded codewords from reference implementation.
        """
        ref_path = test_dir + '/codes/conv/'
        gs = [
            ['101', '111'],
            ['1101', '1111'],
            ['101', '111', '111'],
            ['101', '111', '111', '111']]
        gen_strs = [
            'conv_rate_half_57_',
            'conv_rate_half_6474_',
            'conv_rate_onethird_577_',
            'conv_rate_onefourth_5777_']
        rs=[1/2, 1/2, 1/3, 1/4]
        mus = [3, 4, 3, 3]
        for idx, gen_poly in enumerate(gs):
            enc = ConvEncoder(gen_poly=gen_poly)
            gen_str = gen_strs[idx]
            u = np.load(ref_path + gen_str + 'ref_u.npy')
            cref = np.load(ref_path + gen_str + 'ref_x.npy')
            c = enc(u).numpy()
            self.assertTrue(np.array_equal(c, cref))

            if idx in [0, 2]:
                enc = ConvEncoder(rate=rs[idx], constraint_length=mus[idx])
                c = enc(u).numpy()
                self.assertTrue(np.array_equal(c, cref))

    def test_batch(self):
        """Test that all samples in batch yield same output (for same input).
        """
        bs = 100
        k = 117

        source = BinarySource()
        for enc in (
            ConvEncoder(rate=0.5, constraint_length=8),
            ConvEncoder(rate=0.5, constraint_length=7, terminate=True)):

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

        enc_ref = ConvEncoder(rate=0.5,
                              constraint_length=7,
                              rsc=True,
                              output_dtype=tf.float32)

        u = source([bs, k])
        c_ref = enc_ref(u)

        for dt in dt_supported:
            enc = ConvEncoder(rate=0.5,
                              constraint_length=7,
                              rsc=True,
                              output_dtype=dt)
            u_dt = tf.cast(u, dt)
            c = enc(u_dt)

            c_32 = tf.cast(c, tf.float32)

            self.assertTrue(np.array_equal(c_ref.numpy(), c_32.numpy()))

    def test_tf_fun(self):
        """Test that tf.function decorator works and XLA is supported"""

        @tf.function
        def run_graph(u):
            return enc(u)

        @tf.function(jit_compile=True)
        def run_graph_xla(u):
            return enc(u)

        bs = 10
        k = 100

        source = BinarySource()
        for enc in (
            ConvEncoder(rate=0.5, constraint_length=7),
            ConvEncoder(rate=0.5, constraint_length=4, terminate=True)):

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

