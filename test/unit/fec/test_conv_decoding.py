#
# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
try:
    import sionna
except ImportError as e:
    import sys
    sys.path.append("../")

import unittest
import numpy as np
import tensorflow as tf

from sionna.utils.misc import ebnodb2no
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
from sionna.fec.conv import ConvEncoder, ViterbiDecoder, BCJRDecoder
from sionna.fec.utils import GaussianPriorSource
from sionna.utils import BinarySource
from sionna.channel import AWGN

class ConvExample(tf.keras.Model):
    def __init__(self,
                 k,
                 rate,
                 constraint_length):
        super().__init__()
        self.rate = rate
        self.k = k

        self.binary_source = BinarySource()
        self.encoder = ConvEncoder(rate=rate,
                                   constraint_length=constraint_length)
        self.channel = AWGN()
        self.decoder = ViterbiDecoder(self.encoder.gen_poly, method='soft_llr')

    def call(self, ebno, batch_size):
        # Generate a batch of random bit vectors
        no = tf.cast((1/self.rate) * (10 ** (-ebno / 10)),tf.float32)

        msg = tf.cast(self.binary_source([batch_size, self.k]), tf.int32)
        cw = self.encoder(msg)
        x = 2 * cw - 1

        x_cpx = tf.complex(tf.cast(x, tf.float32), tf.zeros(x.shape))

        y_cpx = self.channel((x_cpx, no))
        y = tf.math.real(y_cpx)
        llr = 2.*y/no

        msghat = tf.cast(self.decoder(llr), tf.int32)

        errs_ = int(tf.math.count_nonzero(msghat-msg))
        return errs_


class TestViterbiDecoding(unittest.TestCase):

    def test_output_dim(self):
        """Test that output dims are correct (=n) and output is all-zero
         codeword."""
        bs = 10

        coderates = [1/2, 1/3]
        ks = [10, 20, 40, 100, 1000]

        for rate in coderates:
            for k in ks:
                n = int(k/rate)
                dec = ViterbiDecoder(rate=rate, constraint_length=5)
                # all-zero with BPSK (no noise);logits
                c = -10. * np.ones([bs, n])
                u = dec(c).numpy()
                self.assertTrue(u.shape[-1]==k)
                # also check that all-zero input yields all-zero output
                u_hat = np.zeros([bs, k])
                self.assertTrue(np.array_equal(u, u_hat))

    def test_numerical_stab(self):
        """Test for numerical stability (no nan or infty as output)"""
        bs = 10
        # (k,n)
        source = GaussianPriorSource()

        coderates = [1/2, 1/3]
        ks = [10, 20, 40, 100, 1000]

        for rate in coderates:
            for k in ks:
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

    def test_identity(self):
        """test that info bits can be recovered if no noise is added"""

        def test_identity_(enc, msg):
            cw = enc(msg)
            # BPSK modulation, no noise
            code_syms = 20. * (2. * cw - 1)
            u_hat = ViterbiDecoder(gen_poly=enc.gen_poly, method='soft_llr')(code_syms)
            self.assertTrue(np.array_equal(msg.numpy(), u_hat.numpy()))

            # No modulation, 0, 1 bits
            code_syms = cw
            u_hat = ViterbiDecoder(gen_poly=enc.gen_poly, method='hard')(code_syms)
            self.assertTrue(np.array_equal(msg.numpy(), u_hat.numpy()))

            # BPSK symbols with AWGN noise
            bs, n = cw.get_shape().as_list()
            code_syms = 6. * (2. * cw - 1) + np.random.randn(bs,n)
            u_hat = ViterbiDecoder(gen_poly=enc.gen_poly, method='soft_llr')(code_syms)
            self.assertTrue(np.array_equal(msg.numpy(), u_hat.numpy()))

            return

        bs = 10
        coderates = [1/2, 1/3]
        ks = [10, 50, 100, 100]
        mus = [3, 4, 5, 6 ,7, 8] # constraint length

        for k in ks:
            for rate in coderates:
                for mu in mus:
                    u = BinarySource()([bs, k])
                    enc = ConvEncoder(rate=rate, constraint_length=mu)
                    test_identity_(enc, u)

            u = BinarySource()([bs, k])

            enc = ConvEncoder(gen_poly=['101', '111', '111', '111'])
            test_identity_(enc, u)

            enc = ConvEncoder(gen_poly=['1101', '1111'])
            test_identity_(enc, u)

    def test_keras(self):
        """Test that Keras model can be compiled (supports dynamic shapes)"""
        bs = 10
        n = 64
        source = BinarySource()
        inputs = tf.keras.Input(shape=(n), dtype=tf.float32)
        x = ViterbiDecoder(rate=1/2, constraint_length=3)(inputs)
        model = tf.keras.Model(inputs=inputs, outputs=x)

        b = source([bs,n])
        model(b)
        # call twice to see that bs can change
        b2 = source([bs+1,n])
        model(b2)
        model.summary()

    def test_multi_dimensional(self):
        """Test against arbitrary shapes
        """
        k = 100
        n = 200

        source = BinarySource()
        dec = ViterbiDecoder(rate=1/2, constraint_length=3)

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
        ref_path = 'codes/conv/'
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
        dec = ViterbiDecoder(rate=1/2, constraint_length=3)

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
        coderates = [1/3, 1/3]
        ks = [10, 20, 40, 100, 1000]

        for rate in coderates:
            for k in ks:
                n = int(k/rate)
                dec = BCJRDecoder(rate=rate, constraint_length=5)
                # all-zero with BPSK (no noise);logits
                c = -10. * np.ones([bs, n])
                u1 = dec(c).numpy()
                self.assertTrue(u1.shape[-1]==k)
                # also check that all-zero input yields all-zero output
                u_hat = np.zeros([bs, k])
                self.assertTrue(np.array_equal(u1, u_hat))

    def test_init(self):
        """Test different init methods as described in the docstring.
        Also test that both implementations lead to the same result."""

        bs = 10
        n = 120
        no = 0.1
        source = GaussianPriorSource()

        coderates = [1/3, 1/2]
        constraint_lengths = [3, 4, 5, 6]
        for r in coderates:
            for cs in constraint_lengths:
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

    def test_numerical_stab(self):
        """Test for numerical stability (no nan or infty as output)"""

        bs = 10
        coderates = [1/2, 1/3]
        ks = [10, 20, 40, 100, 1000]

        source = GaussianPriorSource()

        for rate in coderates:
            for k in ks:
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

    def test_identity(self):
        """test that info bits can be recovered if no noise is added"""

        def test_identity_(enc, msg):
            cw = enc(msg)
            # BPSK modulation, no noise
            code_syms = 20. * (2. * cw - 1)
            u_hat = BCJRDecoder(gen_poly=enc.gen_poly)(code_syms)
            self.assertTrue(np.array_equal(msg.numpy(), u_hat.numpy()))

            # BPSK symbols with AWGN noise
            bs, n = cw.get_shape().as_list()
            code_syms = 6. * (2. * cw - 1) + np.random.randn(bs,n)
            u_hat = BCJRDecoder(gen_poly=enc.gen_poly)(code_syms)
            self.assertTrue(np.array_equal(msg.numpy(), u_hat.numpy()))

            return

        bs = 10
        coderates = [1/2, 1/3]
        ks = [10, 50, 100, 100]
        mus = [3, 4, 5, 6 ,7, 8] # constraint length

        for k in ks:
            for rate in coderates:
                for mu in mus:
                    u = BinarySource()([bs, k])
                    enc = ConvEncoder(rate=rate, constraint_length=mu)
                    test_identity_(enc, u)

            u = BinarySource()([bs, k])

            enc = ConvEncoder(gen_poly=['101', '111', '111', '111'])
            test_identity_(enc, u)

            enc = ConvEncoder(gen_poly=['1101', '1111'])
            test_identity_(enc, u)

    def test_keras(self):
        """Test that Keras model can be compiled (supports dynamic shapes)"""
        bs = 10
        n = 64
        source = BinarySource()
        inputs = tf.keras.Input(shape=(n), dtype=tf.float32)
        x = BCJRDecoder(rate=1/2, constraint_length=3)(inputs)
        model = tf.keras.Model(inputs=inputs, outputs=x)

        b = source([bs,n])
        model(b)
        # call twice to see that bs can change
        b2 = source([bs+1,n])
        model(b2)
        model.summary()

    def test_multi_dimensional(self):
        """Test against arbitrary shapes
        """
        k = 100
        n = 200

        source = BinarySource()
        dec = BCJRDecoder(rate=1/2, constraint_length=3)

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

        ref_path = 'codes/conv/'
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
        k = 100
        n = 128
        source = BinarySource()
        dec = BCJRDecoder(rate=1/2, constraint_length=3)

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
