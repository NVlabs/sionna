#
# SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
import pytest
import os
from itertools import product
import unittest
import numpy as np
import tensorflow as tf
from sionna import config
from sionna.fec.turbo import TurboEncoder, TurboDecoder
from sionna.fec.utils import GaussianPriorSource
from sionna.utils import BinarySource, sim_ber, ebnodb2no
from sionna.channel import AWGN
from sionna.mapping import Mapper, Demapper, Constellation

current_dir = os.path.dirname(os.path.abspath(__file__))
test_dir = os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir))

class TestTurboDecoding(unittest.TestCase):

    def test_output_dim_num_stab(self):
        """Test that output dims are correct (=n) and output is all-zero
         codeword.

        Further, test numerical stability (no nan or infty as output)."""
        bs = 6

        coderates = [1/2, 1/3]
        ks = [12, 60]

        source = GaussianPriorSource()

        for rate in coderates:
            for k in ks:
                n = int(k/rate)
                dec = TurboDecoder(rate=rate,
                                   constraint_length=5,
                                   num_iter=3,
                                   terminate=False)

                # --- test output dimensions ---
                # all-zero with BPSK (no noise);logits
                c = -10. * np.ones([bs, n])
                u = dec(c).numpy()
                self.assertTrue(u.shape[-1]==k)
                # also check that all-zero input yields all-zero output
                u_hat = np.zeros([bs, k])
                self.assertTrue(np.array_equal(u, u_hat))

                # --- test numerical stability ---
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

        def test_identity_(enc, dec, msg):
            cw = enc(msg)
            # BPSK modulation, no noise
            code_syms = 20. * (2. * cw - 1)
            u_hat = dec(code_syms)
            self.assertTrue(np.array_equal(msg.numpy(), u_hat.numpy()))

            # BPSK symbols with AWGN noise
            bs, n = cw.get_shape().as_list()
            code_syms = 6. * (2. * cw - 1) + config.np_rng.normal(size=[bs,n])
            u_hat = dec(code_syms)
            self.assertTrue(np.array_equal(msg.numpy(), u_hat.numpy()))

            return

        bs = 5
        k = 50
        cl = 4 # constraint length
        coderates = [1/3, 1/2]

        for terminate, alg in product([True, False], ("map", "log", "maxlog")):
            for rate in coderates:
                u = BinarySource()([bs, k])
                enc = TurboEncoder(
                    constraint_length=cl, rate=rate, terminate=terminate)
                dec = TurboDecoder(enc, algorithm=alg, num_iter=2)
                test_identity_(enc, dec, u)

    def test_keras(self):
        """Test that Keras model can be compiled (+ supports dynamic shapes)"""
        bs = 10
        n = 64
        source = BinarySource()
        inputs = tf.keras.Input(shape=(n), dtype=tf.float32)
        x = TurboDecoder(rate=1/2, constraint_length=3, terminate=False, num_iter=3)(inputs)
        model = tf.keras.Model(inputs=inputs, outputs=x)

        b = source([bs, n])
        model(b)
        # call twice to see that bs can change in Eager mode
        b2 = source([bs+1, n])
        model(b2)
        model.summary()

    def test_multi_dimensional(self):
        """Test against arbitrary multi-dim input shapes."""
        k = 100
        n = 200

        source = BinarySource()
        dec = TurboDecoder(rate=1/2, constraint_length=3, num_iter=2, terminate=False)

        b = source([30, n])
        b_res = tf.reshape(b, [2, 3, 5, n])

        # encode 2D Tensor
        c = dec(b).numpy()

        # encode 4D Tensor
        c_res = dec(b_res).numpy()

        # test that shape was preserved
        self.assertTrue(c_res.shape[:-1]==b_res.shape[:-1])

        # and reshape to 2D shape
        c_res = tf.reshape(c_res, [30, k])
        # both version should yield same result
        self.assertTrue(np.array_equal(c, c_res))

    def test_batch(self):
        """Test that all samples in batch yield same output (for same input).
        """
        bs = 30
        n = 240

        source = GaussianPriorSource()
        dec = TurboDecoder(rate=1/2, constraint_length=3, terminate=False, num_iter=2)

        b = source([[1, n], 1.])
        b_rep = tf.tile(b, [bs, 1])

        # and run the decoder
        c = dec(b_rep).numpy()

        # test that all samples in the batch are the same
        for i in range(bs):
            self.assertTrue(np.array_equal(c[0,:], c[i,:]))

    @pytest.mark.usefixtures("only_gpu")
    def test_ber_match(self):
        """Test against results from reference implementation.
        """
        def simulation(k, num_iter, snrs):
            r = 1/3
            source = BinarySource()
            enc = TurboEncoder(gen_poly=('1101', '1011'), rate=r, terminate=True)
            dec = TurboDecoder(enc, num_iter=num_iter)
            constellation = Constellation("qam", num_bits_per_symbol=2)
            mapper = Mapper(constellation=constellation)
            demapper = Demapper("app", constellation=constellation)
            channel = AWGN()

            @tf.function(jit_compile=True)
            def run_graph(batch_size, ebno_db):
                no = ebnodb2no(ebno_db, num_bits_per_symbol=2, coderate=r)
                u = source([batch_size, k])
                c = enc(u)
                x = mapper(c)
                y = channel([x, no])
                llr_ch = demapper([y, no])
                u_hat = dec(llr_ch)
                return u, u_hat

            ber, _ = sim_ber(run_graph,
                            ebno_dbs=snrs,
                            max_mc_iter=20,
                            num_target_bit_errors=500,
                            batch_size=10000,
                            soft_estimates=False,
                            early_stop=True,
                            forward_keyboard_interrupt=False)
            return ber
        k = 512
        snrs = [0, 0.5, 1, 1.5, 2]
        ber_lb, ber_ub = {}, {}
        ber_ub[3] = [10.0e-02, 6.0e-02, 5.5e-03, 2.5e-4, 5.0e-06]
        ber_lb[3] = [5.0e-02, 1.0e-02, 1.0e-03, 5.0e-5, 8.0e-07]

        ber_ub[6] = [10.0e-02, 4.0e-02, 6.5e-04, 4.5e-5]
        ber_lb[6] = [5.0e-02, 8.0e-03, 1.0e-04, 2.0e-6]
        for num_iters in [3, 6]:
            if num_iters == 6:
                snrs = snrs[:-1]
            ber = simulation(k, num_iters, snrs)
            for idx in range(len(snrs)):
                self.assertTrue(np.less_equal(ber[idx], ber_ub[num_iters][idx]))
                self.assertTrue(np.greater_equal(ber[idx], ber_lb[num_iters][idx]))

    @pytest.mark.usefixtures("only_gpu")
    def test_ref_implementation(self):
        r"""Test against pre-decoded noisy codewords from reference
        implementation.
        """
        ref_path = test_dir + '/codes/turbo/'
        r = 1/3
        ks = [40, 112, 168]
        enc = TurboEncoder(rate=1/3, terminate=True, constraint_length=4)
        dec = TurboDecoder(enc, num_iter=10)
        ebno = 0.0
        no = 1/(r* (10 ** (-ebno / 10)))

        for k in ks:
            uhatref = np.load(ref_path + 'ref_k{}_uhat.npy'.format(k))
            yref = np.load(ref_path + 'ref_k{}_y.npy'.format(k))
            uhat = dec(-4.*yref/no).numpy()
            self.assertTrue(np.array_equal(uhat, uhatref))

    def test_dtype_flexible(self):
        """Test that output_dtype can be flexible."""
        batch_size = 40
        n = 64
        source = GaussianPriorSource()

        dtypes_supported = (tf.float16, tf.float32, tf.float64)

        for dt_in in dtypes_supported:
            for dt_out in dtypes_supported:
                llr = source([[batch_size, n], 0.5])
                llr = tf.cast(llr, dt_in)

                dec = TurboDecoder(rate=1/2,
                                   constraint_length=3,
                                   output_dtype=dt_out)

                x = dec(llr)

                self.assertTrue(x.dtype==dt_out)

        # test that complex inputs raise error
        llr = source([[batch_size, n], 0.5])
        llr_c = tf.complex(llr, tf.zeros_like(llr))
        dec = TurboDecoder(rate=1/2,
                           constraint_length=3,
                           num_iter=1,
                           output_dtype=tf.float32)

        with self.assertRaises(TypeError):
            x = dec(llr_c)

    @pytest.mark.usefixtures("only_gpu")
    def test_tf_fun(self):
        """Test that tf.function decorator works include xla compiler test."""

        bs = 5
        n = 39 # n should be divisible by 3 for rate=1/3.
        source = BinarySource()

        for t in [False, True]:

            dec = TurboDecoder(rate=1/3, constraint_length=3, terminate=t, num_iter=3)

            @tf.function
            def run_graph(u):
                return dec(u)

            @tf.function(jit_compile=True)
            def run_graph_xla(u):
                return dec(u)

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
            # and change the batch_size again
            u = source([bs+1, n])
            x = run_graph_xla(u).numpy()

    def test_dynamic_shapes(self):
        """Test for dynamic (=unknown) batch-sizes"""

        n = 1536
        enc = TurboEncoder(gen_poly=('1101', '1011'), rate=1/3, terminate=False)
        dec = TurboDecoder(enc, num_iter=3)

        @tf.function(jit_compile=True)
        def run_graph(batch_size):
            llr_ch = tf.zeros((batch_size, n))
            u_hat = dec(llr_ch)
            return u_hat

        run_graph(tf.constant(1))
