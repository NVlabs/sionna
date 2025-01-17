#
# SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
import os
from itertools import product
import unittest
import numpy as np
import tensorflow as tf
from sionna import config
from sionna.utils.misc import ebnodb2no
from sionna.fec.conv import ConvEncoder, ViterbiDecoder, BCJRDecoder
from sionna.fec.utils import GaussianPriorSource
from sionna.utils import BinarySource
from sionna.channel import AWGN

current_dir = os.path.dirname(os.path.abspath(__file__))
test_dir = os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir))

class TestViterbiDecoding(unittest.TestCase):

    def test_output_dim(self):
        """Test that output dims are correct (=n) and output is all-zero
         codeword."""
        bs = 10

        coderates = [1/2, 1/3]
        ks = [10, 22, 40]
        muterm = 3
        for k, rate in product(ks, coderates):
            for dec in (
                ViterbiDecoder(rate=rate, constraint_length=5),
                ViterbiDecoder(rate=rate, constraint_length=3, rsc=True),
                ViterbiDecoder(rate=rate, constraint_length=muterm+1, terminate=True)):

                n = int(k/rate)
                if dec.terminate:
                    n += int((muterm)/rate)

                # all-zero with BPSK (no noise);logits
                c = -10. * np.ones([bs, n])
                u = dec(c).numpy()
                # also check that all-zero input yields all-zero output
                self.assertTrue(u.shape[-1]==k)

                u_hat = np.zeros([bs, k])
                self.assertTrue(np.array_equal(u, u_hat))

    def test_numerical_stab(self):
        """Test for numerical stability (no nan or infty as output)"""
        bs = 10
        # (k,n)
        source = GaussianPriorSource()

        coderates = [1/2, 1/3]
        ks = [10, 20, 60]

        for k, rate in product(ks, coderates):
            n = int(k/rate)
            dec = ViterbiDecoder(rate=rate, constraint_length=5)

            # case 1: extremely large inputs
            c = source([[bs, n], 0.0001])
            # llrs
            u1 = dec(c).numpy()
            # no nan
            self.assertFalse(np.any(np.isnan(u1)))
            #no inftfy
            self.assertFalse(np.any(np.isinf(u1)))
            self.assertFalse(np.any(np.isneginf(u1)))

            # case 2: zero input
            c = tf.zeros([bs, n])
            # llrs
            u2 = dec(c).numpy()
            # no nan
            self.assertFalse(np.any(np.isnan(u2)))
            #no inftfy
            self.assertFalse(np.any(np.isinf(u2)))
            self.assertFalse(np.any(np.isneginf(u2)))

    def test_init(self):
        """Test different init methods as described in the docstring.
        Also test that both implementations lead to the same result."""

        bs = 10
        n = 120
        no = 0.1
        source = GaussianPriorSource()

        coderates = [1/3, 1/2]
        constraint_lengths = [3, 4, 5, 6]
        for r, cs in product(coderates, constraint_lengths):

            enc = ConvEncoder(rate=r, constraint_length=cs)

            # method 1: explicitly provide enc
            dec1 = ViterbiDecoder(gen_poly=enc.gen_poly)

            # method 2: provide rate and constraint length
            dec2 = ViterbiDecoder(rate=r, constraint_length=cs)

            llr = source([[bs, n], no])

            x_hat1 = dec1(llr)
            x_hat2 = dec2(llr)

            #verify that both decoders produce the same result
            self.assertTrue(np.array_equal(x_hat1.numpy(), x_hat2.numpy()))

    def test_identity(self):
        """Test that info bits can be recovered if no noise is added."""

        def test_identity_(enc, msg, rsc=False):
            cw = enc(msg)

            # test that encoder can be directly provided
            for api_mode in ("poly", "enc"):

                # BPSK modulation, no noise
                code_syms = 20. * (2. * cw - 1)
                if api_mode=="poly":
                    u_hat = ViterbiDecoder(
                        gen_poly=enc.gen_poly, method='soft_llr', rsc=rsc)(code_syms)
                else:
                    u_hat = ViterbiDecoder(encoder=enc, method='soft_llr')(code_syms)
                self.assertTrue(np.array_equal(msg.numpy(), u_hat.numpy()))

                # No modulation, 0, 1 bits
                code_syms = cw
                if api_mode=="poly":
                    u_hat = ViterbiDecoder(
                        gen_poly=enc.gen_poly, method='hard', rsc=rsc)(code_syms)
                else:
                    u_hat = ViterbiDecoder(encoder=enc, method='hard')(code_syms)
                self.assertTrue(np.array_equal(msg.numpy(), u_hat.numpy()))

                # BPSK symbols with AWGN noise
                bs, n = cw.get_shape().as_list()
                code_syms = 6. * (2. * cw - 1) + config.np_rng.normal(size=[bs,n])
                if api_mode=="poly":
                    u_hat = ViterbiDecoder(
                        gen_poly=enc.gen_poly, method='soft_llr', rsc=rsc)(code_syms)
                else:
                    u_hat = ViterbiDecoder(encoder=enc, method='soft_llr')(code_syms)
                self.assertTrue(np.array_equal(msg.numpy(), u_hat.numpy()))

        bs = 10
        k = 35
        coderates = [1/2, 1/3]
        mus = [3, 8] # constraint length

        for rate, mu in product(coderates, mus):
            u = BinarySource()([bs, k])
            enc = ConvEncoder(rate=rate, constraint_length=mu)
            test_identity_(enc, u)

            enc = ConvEncoder(rate=rate, constraint_length=mu, rsc=True)
            test_identity_(enc, u, rsc=True)

        for gp in (['101', '111', '111', '111'], ['1101', '1111']):
            u = BinarySource()([bs, k])
            enc = ConvEncoder(gen_poly=gp)
            test_identity_(enc, u)

    def test_keras(self):
        """Test that Keras model can be compiled (supports dynamic shapes)"""
        bs = 10
        n1 = 64

        muterm = 3
        rterm = 1/3
        n2 = 96 + int(muterm/rterm)

        source = BinarySource()

        inputs = tf.keras.Input(shape=(n1), dtype=tf.float32)
        x = ViterbiDecoder(rate=1/2, constraint_length=3)(inputs)
        model = tf.keras.Model(inputs=inputs, outputs=x)

        # Keras Model using termination
        inputs = tf.keras.Input(shape=(n2), dtype=tf.float32)
        xterm = ViterbiDecoder(
            rate=rterm, constraint_length=muterm+1, terminate=True)(inputs)
        modelterm = tf.keras.Model(inputs=inputs, outputs=xterm)

        for n, mod in zip((n1,n2),(model, modelterm)):
            b = source([bs, n])
            mod(b)
            # call twice to see that bs can change
            b2 = source([bs+1,n])
            mod(b2)
            mod.summary()

    def test_multi_dimensional(self):
        """Test against arbitrary shapes
        """
        k = 100
        rate = 1/2
        mu = 3

        source = BinarySource()
        for term in (True, False):
            dec = ViterbiDecoder(rate=rate, constraint_length=mu+1, terminate=term)

            n = int(k/rate)
            if dec.terminate:
                n += int(mu/rate)

            b = source([100, n])
            b_res = tf.reshape(b, [4, 5, 5, n])

            # encode 2D Tensor
            c = dec(b).numpy()
            # encode 4D Tensor
            c_res = dec(b_res).numpy()

            # test that shape was preserved
            self.assertTrue(c_res.shape[:-1]==b_res.shape[:-1])

            # and reshape to 2D shape
            c_res = tf.reshape(c_res, [100, k])
            # both version should yield same result
            self.assertTrue(np.array_equal(c, c_res))

    def test_batch(self):
        """Test that all samples in batch yield same output (for same input).
        """
        bs = 100
        n = 240

        source = GaussianPriorSource()
        dec = ViterbiDecoder(rate=1/2, constraint_length=3)

        b = source([[1, n], 1])
        b_rep = tf.tile(b, [bs, 1])

        # and run tf version (to be tested)
        c = dec(b_rep).numpy()

        for i in range(bs):
            self.assertTrue(np.array_equal(c[0,:], c[i,:]))

    def test_ref_implementation(self):
        """Test against pre-decoded results from reference implementation.
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

        for idx, gen_poly in enumerate(gs):
            dec = ViterbiDecoder(gen_poly=gen_poly, method='soft_llr')
            gen_str = gen_strs[idx]

            # yref is generated from AWGN channel with Es/N0=4.95dB
            yref = np.load(ref_path + gen_str + 'ref_y.npy')
            uhat_ref = np.load(ref_path + gen_str + 'ref_uhat.npy')
            no = ebnodb2no(4.95, num_bits_per_symbol=2, coderate=1.)
            yref_soft = 2*(yref) / no
            uhat = dec(yref_soft)
            self.assertTrue(np.array_equal(uhat_ref, uhat))

    def test_dtype_flexible(self):
        """Test that output_dtype can be flexible"""
        batch_size = 100
        k = 30
        n = 64
        source = GaussianPriorSource()

        dtypes_supported = (tf.float16, tf.float32, tf.float64)

        for dt_in in dtypes_supported:
            for dt_out in dtypes_supported:
                llr = source([[batch_size, n], 0.5])
                llr = tf.cast(llr, dt_in)

                dec = ViterbiDecoder(rate=1/2,
                                     constraint_length=3,
                                     output_dtype=dt_out)

                x = dec(llr)

                self.assertTrue(x.dtype==dt_out)

        # test that complex inputs raise error
        llr = source([[batch_size, n], 0.5])
        llr_c = tf.complex(llr, tf.zeros_like(llr))
        dec = ViterbiDecoder(rate=1/2,
                            constraint_length=3,
                            output_dtype=tf.float32)

        with self.assertRaises(TypeError):
            x = dec(llr_c)

    def test_tf_fun(self):
        """Test that tf.function decorator works
        include xla compiler test"""

        @tf.function
        def run_graph(u):
            return dec(u)

        @tf.function(jit_compile=True)
        def run_graph_xla(u):
            return dec(u)

        bs = 10
        k = 100
        n = 128
        source = BinarySource()
        dec = ViterbiDecoder(rate=1/2, constraint_length=5)

        # test that for arbitrary input only 0,1 values are outputed
        u = source([bs, n])
        x = run_graph(u).numpy()

        # execute the graph twice
        x = run_graph(u).numpy()

        # and change batch_size
        u = source([bs+1, n])
        x = run_graph(u).numpy()

        # run same test for XLA (jit_compile=True)
        u = source([bs, n])
        x = run_graph_xla(u).numpy()
        x = run_graph_xla(u).numpy()
        u = source([bs+1, n])
        x = run_graph_xla(u).numpy()


class TestBCJRDecoding(unittest.TestCase):

    def test_output_dim(self):
        """Test that output dims are correct (=n) and output is the all-zero
         codeword."""

        bs = 10
        coderates = [1/2, 1/3]
        ks = [10, 45]
        muterm = 5

        for k, rate in product(ks, coderates):
            for dec in (
                BCJRDecoder(rate=rate, constraint_length=5),
                BCJRDecoder(rate=rate, constraint_length=3, rsc=True),
                BCJRDecoder(rate=rate, constraint_length=muterm+1, terminate=True)):

                n = int(k/rate)
                if dec.terminate:
                    n += int((muterm)/rate)

                # all-zero with BPSK (no noise);logits
                c = -10. * np.ones([bs, n])
                u1 = dec(c).numpy()
                self.assertTrue(u1.shape[-1]==k)
                # also check that all-zero input yields all-zero output
                u_hat = np.zeros([bs, k])
                self.assertTrue(np.array_equal(u1, u_hat))

    def test_numerical_stab(self):
        """Test for numerical stability (no nan or infty as output)"""

        bs = 10
        coderates = [1/2, 1/3]
        ks = [22, 55]

        source = GaussianPriorSource()

        for k, rate in product(ks, coderates):
            n = int(k/rate)
            dec = BCJRDecoder(rate=rate, constraint_length=5)

            # case 1: extremely large inputs
            c = source([[bs, n], 0.0001])
            # llrs
            u1 = dec(c).numpy()
            # no nan
            self.assertFalse(np.any(np.isnan(u1)))
            #no inftfy
            self.assertFalse(np.any(np.isinf(u1)))
            self.assertFalse(np.any(np.isneginf(u1)))

            # case 2: zero input
            c = tf.zeros([bs, n])
            # llrs
            u2 = dec(c).numpy()
            # no nan
            self.assertFalse(np.any(np.isnan(u2)))
            #no inftfy
            self.assertFalse(np.any(np.isinf(u2)))
            self.assertFalse(np.any(np.isneginf(u2)))

    def test_init(self):
        """Test different init methods as described in the docstring.
        Also test that both implementations lead to the same result."""

        bs = 10
        n = 120
        no = 0.1
        source = GaussianPriorSource()

        coderates = [1/3, 1/2]
        constraint_lengths = [3, 4, 5, 6]
        for r, cs in product(coderates, constraint_lengths):

            enc = ConvEncoder(rate=r, constraint_length=cs)

            # method 1: explicitly provide enc
            dec1 = BCJRDecoder(gen_poly=enc.gen_poly)

            # method 2: provide rate and constraint length
            dec2 = BCJRDecoder(rate=r, constraint_length=cs)

            llr = source([[bs, n], no])

            x_hat1 = dec1(llr)
            x_hat2 = dec2(llr)

            #verify that both decoders produce the same result
            self.assertTrue(np.array_equal(x_hat1.numpy(), x_hat2.numpy()))

    def test_identity(self):
        """test that info bits can be recovered if no noise is added"""

        def test_identity_(enc, msg, rsc=False):
            cw = enc(msg)

            # test that encoder can be directly provided
            for api_mode, alg in product(
                ("poly", "enc"), ("map", "log", "maxlog")):

                # BPSK modulation, no noise
                code_syms = 20. * (2. * cw - 1)
                if api_mode=="poly":
                    u_hat = BCJRDecoder(gen_poly=enc.gen_poly,
                                        algorithm=alg, rsc=rsc)(code_syms)
                else:
                    u_hat = BCJRDecoder(encoder=enc, algorithm=alg)(code_syms)
                self.assertTrue(np.array_equal(msg.numpy(), u_hat.numpy()))

                # BPSK symbols with AWGN noise
                bs, n = cw.get_shape().as_list()
                code_syms = 6. * (2. * cw - 1) + config.np_rng.normal(size=[bs,n])
                if api_mode=="poly":
                    u_hat = BCJRDecoder(gen_poly=enc.gen_poly,
                                        algorithm=alg, rsc=rsc)(code_syms)
                else:
                    u_hat = BCJRDecoder(encoder=enc, algorithm=alg)(code_syms)
                self.assertTrue(np.array_equal(msg.numpy(), u_hat.numpy()))

        bs = 5
        coderates = [1/2, 1/3]
        k = 40
        mus = [3, 8] # constraint length

        for rate, mu in product(coderates, mus):
            u = BinarySource()([bs, k])
            enc = ConvEncoder(rate=rate, constraint_length=mu)
            test_identity_(enc, u)

            enc = ConvEncoder(rate=rate, constraint_length=mu, rsc=True)
            test_identity_(enc, u, rsc=True)

        for gp in (
            ['101', '111', '111'],
            ['1101', '1111']):
            u = BinarySource()([bs, k])
            enc = ConvEncoder(gen_poly=gp)
            test_identity_(enc, u)

    def test_keras(self):
        """Test that Keras model can be compiled (supports dynamic shapes)"""
        bs = 10
        n1 = 64

        muterm = 3
        rterm = 1/3
        n2 = 96 + int(muterm/rterm)

        source = BinarySource()

        inputs = tf.keras.Input(shape=(n1), dtype=tf.float32)
        x = BCJRDecoder(rate=1/2, constraint_length=3)(inputs)
        model = tf.keras.Model(inputs=inputs, outputs=x)

        # Keras Model using termination
        inputs = tf.keras.Input(shape=(n2), dtype=tf.float32)
        xterm = BCJRDecoder(
            rate=rterm, constraint_length=muterm+1, terminate=True)(inputs)
        modelterm = tf.keras.Model(inputs=inputs, outputs=xterm)

        for n, mod in zip((n1,n2),(model, modelterm)):
            b = source([bs,n])
            mod(b)
            # call twice to see that bs can change
            b2 = source([bs+1,n])
            mod(b2)
            mod.summary()

    def test_multi_dimensional(self):
        """Test against arbitrary shapes
        """
        k = 40
        rate = 1/2
        mu = 3

        source = BinarySource()
        for term in (True, False):
            dec = BCJRDecoder(rate=rate, constraint_length=mu+1, terminate=term)

            n = int(k/rate)
            if dec.terminate:
                n += int(mu/rate)

            b = source([100, n])
            b_res = tf.reshape(b, [4, 5, 5, n])

            # encode 2D Tensor
            c = dec(b).numpy()
            # encode 4D Tensor
            c_res = dec(b_res).numpy()

            # test that shape was preserved
            self.assertTrue(c_res.shape[:-1]==b_res.shape[:-1])

            # and reshape to 2D shape
            c_res = tf.reshape(c_res, [100, k])
            # both version should yield same result
            self.assertTrue(np.array_equal(c, c_res))

    def test_batch(self):
        """Test that all samples in batch yield same output (for same input).
        """
        bs = 100
        n = 240

        source = GaussianPriorSource()
        dec = BCJRDecoder(rate=1/2, constraint_length=3)

        b = source([[1, n], 1])
        b_rep = tf.tile(b, [bs, 1])

        # and run tf version (to be tested)
        c = dec(b_rep).numpy()

        for i in range(bs):
            self.assertTrue(np.array_equal(c[0,:], c[i,:]))

    def test_ref_implementation(self):
        """Test against pre-decoded results from reference implementation.
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

        for idx, gen_poly in enumerate(gs):
            dec = BCJRDecoder(gen_poly=gen_poly)
            gen_str = gen_strs[idx]

            # yref is generated from AWGN channel with Es/N0=4.95dB
            yref = np.load(ref_path + gen_str + 'ref_y.npy')
            uhat_ref = np.load(ref_path + gen_str + 'ref_uhat.npy')

            yref_soft = 0.5 * (yref+1)
            uhat = dec(yref_soft)
            self.assertTrue(np.array_equal(uhat_ref, uhat))

    def test_dtype_flexible(self):
        """Test that output_dtype can be flexible"""
        batch_size = 100
        k = 30
        n = 64
        source = GaussianPriorSource()

        dtypes_supported = (tf.float16, tf.float32, tf.float64)

        for dt_in in dtypes_supported:
            for dt_out in dtypes_supported:
                llr = source([[batch_size, n], 0.5])
                llr = tf.cast(llr, dt_in)

                dec = BCJRDecoder(rate=1/2,
                                  constraint_length=3,
                                  output_dtype=dt_out)
                x = dec(llr)
                self.assertTrue(x.dtype==dt_out)

        # test that complex inputs raise error
        llr = source([[batch_size, n], 0.5])
        llr_c = tf.complex(llr, tf.zeros_like(llr))
        dec = BCJRDecoder(rate=1/2,
                          constraint_length=3,
                          output_dtype=tf.float32)

        with self.assertRaises(TypeError):
            x = dec(llr_c)

    def test_tf_fun(self):
        """Test that tf.function decorator works
        include xla compiler test"""

        @tf.function
        def run_graph(u):
            return dec(u)

        @tf.function(jit_compile=True)
        def run_graph_xla(u):
            return dec(u)

        bs = 10
        n = 128
        source = BinarySource()
        dec = BCJRDecoder(rate=1/2, constraint_length=5)

        # test that for arbitrary input only 0,1 values are outputed
        u = source([bs, n])
        x = run_graph(u).numpy()

        # execute the graph twice
        x = run_graph(u).numpy()

        # and change batch_size
        u = source([bs+1, n])
        x = run_graph(u).numpy()

        # run same test for XLA (jit_compile=True)
        u = source([bs, n])
        x = run_graph_xla(u).numpy()
        x = run_graph_xla(u).numpy()
        u = source([bs+1, n])
        x = run_graph_xla(u).numpy()

    def test_dynamic_shapes(self):
        """Test for dynamic (=unknown) batch-sizes"""

        n = 1536
        enc = ConvEncoder(gen_poly=('1101', '1011'), rate=1/3, terminate=False)
        dec = BCJRDecoder(enc)

        @tf.function(jit_compile=True)
        def run_graph(batch_size):
            llr_ch = tf.zeros((batch_size, n))
            u_hat = dec(llr_ch)
            return u_hat

        run_graph(tf.constant(1))
