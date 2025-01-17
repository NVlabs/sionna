#
# SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import os
import unittest
import pytest # for pytest filterwarnings
import numpy as np
import tensorflow as tf

from sionna.fec.polar.encoding import PolarEncoder, Polar5GEncoder
from sionna.fec.polar.decoding import PolarSCDecoder, PolarSCLDecoder, PolarBPDecoder
from sionna.fec.polar.decoding import Polar5GDecoder
from sionna.fec.crc import CRCEncoder
from sionna.fec.utils import GaussianPriorSource
from sionna.utils import BinarySource
from sionna.fec.polar.utils import generate_5g_ranking
from sionna.channel import AWGN
from sionna.mapping import Mapper, Demapper

current_dir = os.path.dirname(os.path.abspath(__file__))
test_dir = os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir))

class TestPolarDecodingSC(unittest.TestCase):

    def test_invalid_inputs(self):
        """Test against invalid values of n and frozen_pos."""

        # frozen vec to long
        n = 32
        frozen_pos = np.arange(n+1)
        with self.assertRaises(AssertionError):
            PolarSCDecoder(frozen_pos, n)

        # n not a pow of 2
        # frozen vec to long
        n = 32
        k = 12
        frozen_pos,_ = generate_5g_ranking(k, n)
        with self.assertRaises(AssertionError):
            PolarSCDecoder(frozen_pos, n+1)

        # test valid shapes
        # (k, n)
        param_valid = [[0, 32], [10, 32], [32, 32], [100, 256],
                       [123, 1024], [1024, 1024]]

        for p in param_valid:
            frozen_pos, _ = generate_5g_ranking(p[0], p[1])
            PolarSCDecoder(frozen_pos, p[1])

        # no complex-valued input allowed
        with self.assertRaises(ValueError):
            frozen_pos,_ = generate_5g_ranking(32, 64)
            PolarSCDecoder(frozen_pos, 64, output_dtype=tf.complex64)

    def test_output_dim(self):
        """Test that output dims are correct (=n) and output equals all-zero
         codeword."""

        bs = 10

        # (k, n)
        param_valid = [[1, 32], [10, 32], [32, 32], [100, 256], [123, 1024],
                      [1024, 1024]]

        for p in param_valid:
            frozen_pos, _ = generate_5g_ranking(p[0],p[1])
            dec = PolarSCDecoder(frozen_pos, p[1])
            # all-zero with BPSK (no noise);logits
            c = -10. * np.ones([bs, p[1]])
            u = dec(c).numpy()
            self.assertTrue(u.shape[-1]==p[0])
            # also check that all-zero input yields all-zero output
            u_hat = np.zeros([bs, p[0]])
            self.assertTrue(np.array_equal(u, u_hat))

    def test_numerical_stab(self):
        """Test for numerical stability (no nan or infty as output)."""

        bs = 10
        # (k,n)
        param_valid = [[1, 32], [10, 32], [32, 32], [100, 256]]
        source = GaussianPriorSource()

        for p in param_valid:
            frozen_pos, _ = generate_5g_ranking(p[0],p[1])
            dec = PolarSCDecoder(frozen_pos, p[1])

            # case 1: extremely large inputs
            c = source([[bs, p[1]], 0.0001])
            # llrs
            u1 = dec(c).numpy()
            # no nan
            self.assertFalse(np.any(np.isnan(u1)))
            #no inftfy
            self.assertFalse(np.any(np.isinf(u1)))
            self.assertFalse(np.any(np.isneginf(u1)))

            # case 2: zero llr input
            c = tf.zeros([bs, p[1]])
            # llrs
            u2 = dec(c).numpy()
            # no nan
            self.assertFalse(np.any(np.isnan(u2)))
            #no inftfy
            self.assertFalse(np.any(np.isinf(u2)))
            self.assertFalse(np.any(np.isneginf(u2)))

    def test_identity(self):
        """test that info bits can be recovered if no noise is added."""

        bs = 10

        # (k, n)
        param_valid = [[1, 32], [10, 32], [32, 32], [100, 256], [123, 1024],
                      [1024, 1024]]

        for p in param_valid:
            source = BinarySource()
            frozen_pos, _ = generate_5g_ranking(p[0],p[1])
            enc = PolarEncoder(frozen_pos, p[1])
            dec = PolarSCDecoder(frozen_pos, p[1])

            u = source([bs, p[0]])
            c = enc(u)
            llr_ch = 20.*(2.*c-1) # demod BPSK witout noise
            u_hat = dec(llr_ch)

            self.assertTrue(np.array_equal(u.numpy(), u_hat.numpy()))

    def test_keras(self):
        """Test that Keras model can be compiled (supports dynamic shapes)."""

        bs = 10
        k = 100
        n = 128
        source = BinarySource()
        frozen_pos, _ = generate_5g_ranking(k, n)
        inputs = tf.keras.Input(shape=(n), dtype=tf.float32)
        x = PolarSCDecoder(frozen_pos, n)(inputs)
        model = tf.keras.Model(inputs=inputs, outputs=x)

        b = source([bs, n])
        model(b)
        # call twice to see that bs can change
        b2 = source([bs+1, n])
        model(b2)
        model.summary()

    def test_multi_dimensional(self):
        """Test against arbitrary shapes.
        """

        k = 120
        n = 256

        frozen_pos, _ = generate_5g_ranking(k, n)
        source = BinarySource()
        dec = PolarSCDecoder(frozen_pos, n)

        b = source([100, n])
        b_res = tf.reshape(b, [4, 5, 5, n])

        # encode 2D Tensor
        c = dec(b).numpy()
        # encode 4D Tensor
        c_res = dec(b_res).numpy()
        # and reshape to 2D shape
        c_res = tf.reshape(c_res, [100, k])
        # both version should yield same result
        self.assertTrue(np.array_equal(c, c_res))

    def test_batch(self):
        """Test that all samples in batch yield same output (for same input).
        """

        bs = 100
        k = 120
        n = 256

        frozen_pos, _ = generate_5g_ranking(k, n)
        source = BinarySource()
        dec = PolarSCDecoder(frozen_pos, n)

        b = source([1,15,n])
        b_rep = tf.tile(b, [bs, 1, 1])

        # and run tf version (to be tested)
        c = dec(b_rep).numpy()

        for i in range(bs):
            self.assertTrue(np.array_equal(c[0,:,:], c[i,:,:]))


    def test_tf_fun(self):
        """Test that graph mode works and xla is supported."""

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
        frozen_pos, _ = generate_5g_ranking(k, n)
        dec = PolarSCDecoder(frozen_pos, n)

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

    def test_ref_implementation(self):
        """Test against pre-calculated results from internal implementation.
        """

        ref_path = test_dir + '/codes/polar/'
        filename = ["P_128_37", "P_128_110", "P_256_128"]

        for f in filename:
            A = np.load(ref_path + f + "_Avec.npy")
            llr_ch = np.load(ref_path + f + "_Lch.npy")
            u_hat = np.load(ref_path + f + "_uhat.npy")
            frozen_pos = np.array(np.where(A==0)[0])
            info_pos = np.array(np.where(A==1)[0])

            n = len(frozen_pos) + len(info_pos)
            k = len(info_pos)

            dec = PolarSCDecoder(frozen_pos, n)
            l_in = -1. * llr_ch # logits
            u_hat_tf = dec(l_in).numpy()

            # the output should be equal to the reference
            self.assertTrue(np.array_equal(u_hat_tf, u_hat))

    def test_dtype_flexible(self):
        """Test that output_dtype can be flexible."""

        batch_size = 100
        k = 30
        n = 64
        source = GaussianPriorSource()
        frozen_pos, _ = generate_5g_ranking(k, n)
        dtypes_supported = (tf.float16, tf.float32, tf.float64)

        for dt_in in dtypes_supported:
            for dt_out in dtypes_supported:
                llr = source([[batch_size, n], 0.5])
                llr = tf.cast(llr, dt_in)

                dec = PolarSCDecoder(frozen_pos, n, output_dtype=dt_out)

                x = dec(llr)

                self.assertTrue(x.dtype==dt_out)

        # test that complex-valued inputs raise error
        llr = source([[batch_size, n], 0.5])
        llr_c = tf.complex(llr, tf.zeros_like(llr))
        dec = PolarSCDecoder(frozen_pos, n, output_dtype=tf.float32)

        with self.assertRaises(TypeError):
            x = dec(llr_c)


class TestPolarDecodingSCL(unittest.TestCase):

    # Filter warnings related to large ressource allocation
    @pytest.mark.filterwarnings("ignore: Required resource allocation")
    def test_invalid_inputs(self):
        """Test against invalid values of n and frozen_pos."""

        # frozen vec to long
        n = 32
        frozen_pos = np.arange(n+1)
        with self.assertRaises(AssertionError):
            PolarSCLDecoder(frozen_pos, n)

        # n not a pow of 2
        # frozen vec to long
        n = 32
        k = 12
        frozen_pos,_ = generate_5g_ranking(k, n)
        with self.assertRaises(AssertionError):
            PolarSCLDecoder(frozen_pos, n+1)

        # also test valid shapes
        # (k, n)
        param_valid = [[0, 32], [10, 32], [32, 32], [100, 256],
                       [123, 1024], [1024, 1024]]

        for p in param_valid:
            frozen_pos, _ = generate_5g_ranking(p[0],p[1])
            PolarSCLDecoder(frozen_pos, p[1])

        # no complex-valued input allowed
        with self.assertRaises(ValueError):
            frozen_pos,_ = generate_5g_ranking(32, 64)
            PolarSCLDecoder(frozen_pos, 64, output_dtype=tf.complex64)

    # Filter warnings related to large resource allocation
    @pytest.mark.filterwarnings("ignore: Required resource allocation")
    @pytest.mark.usefixtures("only_gpu")
    def test_output_dim(self):
        """Test that output dims are correct (=n) and output is the all-zero
         codeword."""

        bs = 10
        # (k, n)
        param_valid = [[1, 32], [10, 32], [32, 32], [100, 256], [123, 1024],
                      [1024, 1024]]

        # use_hybrid, use_fast_scl, cpu_only, use_scatter
        for p in param_valid:
            frozen_pos, _ = generate_5g_ranking(p[0], p[1])
            for use_fast_scl in [False, True]:
                for cpu_only in [False, True]:
                    for use_scatter in [False, True]:
                        dec = PolarSCLDecoder(frozen_pos,
                                            p[1],
                                            use_fast_scl=use_fast_scl,
                                            cpu_only=cpu_only,
                                            use_scatter=use_scatter)

                        # all-zero with BPSK (no noise);logits
                        c = -10. * np.ones([bs, p[1]])
                        u = dec(c).numpy()
                        # check shape
                        self.assertTrue(u.shape[-1]==p[0])
                        # also check that all-zero input yields all-zero
                        u_hat = np.zeros([bs, p[0]])
                        self.assertTrue(np.array_equal(u, u_hat))

        # also test different list sizes
        n = 32
        k = 16
        frozen_pos, _ = generate_5g_ranking(k, n)
        list_sizes = [1, 2, 8, 32]
        for list_size in list_sizes:
            for use_fast_scl in [False, True]:
                for cpu_only in [False, True]:
                    for use_scatter in [False, True]:
                        dec = PolarSCLDecoder(frozen_pos,
                                         n,
                                         list_size=list_size,
                                         use_fast_scl=use_fast_scl,
                                         cpu_only=cpu_only,
                                         use_scatter=use_scatter)

                        # all-zero with BPSK (no noise);logits
                        c = -10. * np.ones([bs, n])
                        u = dec(c).numpy()
                        self.assertTrue(u.shape[-1]==k)
                        # also check that all-zero input yields all-zero
                        u_hat = np.zeros([bs, k])
                        self.assertTrue(np.array_equal(u, u_hat))

    # Filter warnings related to large resource allocation
    @pytest.mark.filterwarnings("ignore: Required ressource allocation")
    def test_numerical_stab(self):
        """Test for numerical stability (no nan or infty as output)"""

        bs = 10

        # (k, n)
        param_valid = [[1, 32], [10, 32], [32, 32], [100, 256]]
        source = GaussianPriorSource()

        for p in param_valid:
            frozen_pos, _ = generate_5g_ranking(p[0], p[1])
            for use_fast_scl in [False, True]:
                for cpu_only in [False, True]:
                    for use_scatter in [False, True]:
                        dec = PolarSCLDecoder(frozen_pos,
                                            p[1],
                                            use_fast_scl=use_fast_scl,
                                            cpu_only=cpu_only,
                                            use_scatter=use_scatter)
                        # case 1: extremely large inputs
                        c = source([[bs, p[1]], 0.0001])
                        # llrs
                        u1 = dec(c).numpy()
                        # no nan
                        self.assertFalse(np.any(np.isnan(u1)))
                        #no infty
                        self.assertFalse(np.any(np.isinf(u1)))
                        self.assertFalse(np.any(np.isneginf(u1)))

                        # case 2: zero input
                        c = tf.zeros([bs, p[1]])
                        # llrs
                        u2 = dec(c).numpy()
                        # no nan
                        self.assertFalse(np.any(np.isnan(u2)))
                        # no infty
                        self.assertFalse(np.any(np.isinf(u2)))
                        self.assertFalse(np.any(np.isneginf(u2)))

    # Filter warnings related to large ressource allocation
    @pytest.mark.filterwarnings("ignore: Required ressource allocation")
    def test_identity(self):
        """Test that info bits can be recovered if no noise is added."""

        bs = 10
        # (k,n)
        param_valid = [[1, 32], [10, 32], [32, 32], [100, 256]]

        source = BinarySource()

        # use_hybrid, use_fast_scl, cpu_only, use_scatter
        for p in param_valid:
            frozen_pos, _ = generate_5g_ranking(p[0], p[1])
            enc = PolarEncoder(frozen_pos, p[1])
            u = source([bs, p[0]])
            c = enc(u)
            llr_ch = 200.*(2.*c-1) # demod BPSK without noise
            for use_fast_scl in [False, True]:
                for cpu_only in [False, True]:
                    for use_scatter in [False, True]:
                        dec = PolarSCLDecoder(frozen_pos,
                                         p[1],
                                         use_fast_scl=use_fast_scl,
                                         cpu_only=cpu_only,
                                         use_scatter=use_scatter)

                        u_hat = dec(llr_ch)
                        self.assertTrue(np.array_equal(u.numpy(),
                                                       u_hat.numpy()))
        # also test different list sizes
        n = 32
        k = 16
        crc_degree = "CRC11"
        frozen_pos, _ = generate_5g_ranking(k, n)
        enc = PolarEncoder(frozen_pos, n)
        enc_crc = CRCEncoder(crc_degree)
        u = source([bs, k-enc_crc.crc_length])
        u_crc = enc_crc(u)
        c = enc(u_crc)
        llr_ch = 200.*(2.*c-1) # demod BPSK witout noise
        list_sizes = [1, 2, 8, 32]
        for list_size in list_sizes:
            for use_fast_scl in [False, True]:
                for cpu_only in [False, True]:
                    for use_scatter in [False, True]:
                        dec = PolarSCLDecoder(frozen_pos,
                                         n,
                                         list_size=list_size,
                                         use_fast_scl=use_fast_scl,
                                         cpu_only=cpu_only,
                                         use_scatter=use_scatter,
                                         crc_degree=crc_degree)
                        u_hat = dec(llr_ch)
                        self.assertTrue(np.array_equal(u_crc.numpy(),
                                                       u_hat.numpy()))

    def test_keras(self):
        """Test that Keras model can be compiled (supports dynamic shapes)."""

        bs = 10
        k = 16
        n = 32
        for use_fast_scl in [False, True]:
            for cpu_only in [False, True]:
                for use_scatter in [False, True]:
                    source = BinarySource()
                    frozen_pos, _ = generate_5g_ranking(k, n)
                    inputs = tf.keras.Input(shape=(n), dtype=tf.float32)
                    x = PolarSCLDecoder(frozen_pos,
                                        n,
                                        use_fast_scl=use_fast_scl,
                                        cpu_only=cpu_only,
                                        use_scatter=use_scatter)(inputs)
                    model = tf.keras.Model(inputs=inputs, outputs=x)

                    b = source([bs,n])
                    model(b)
                    # call twice to see that bs can change
                    b2 = source([bs+1,n])
                    model(b2)
                    model.summary()

    # Filter warnings related to large ressource allocation
    @pytest.mark.filterwarnings("ignore: Required ressource allocation")
    def test_multi_dimensional(self):
        """Test against multi-dimensional input shapes.

        As reshaping is done before calling the actual decoder, no exhaustive
        testing against all decoder options is required.
        """
        k = 120
        n = 256

        frozen_pos, _ = generate_5g_ranking(k, n)
        source = BinarySource()
        dec = PolarSCLDecoder(frozen_pos, n)

        b = source([100, n])
        b_res = tf.reshape(b, [4, 5, 5, n])

        # encode 2D Tensor
        c = dec(b).numpy()
        # encode 4D Tensor
        c_res = dec(b_res).numpy()
        # and reshape to 2D shape
        c_res = tf.reshape(c_res, [100, k])
        # both version should yield same result
        self.assertTrue(np.array_equal(c, c_res))

    @pytest.mark.usefixtures("only_gpu")
    def test_batch(self):
        """Test that all samples in batch yield same output (for same input).
        """
        bs = 100
        k = 78
        n = 128

        frozen_pos, _ = generate_5g_ranking(k, n)
        source = BinarySource()
        for use_fast_scl in [False, True]:
            for cpu_only in [False, True]:
                for use_scatter in [False, True]:
                    dec = PolarSCLDecoder(frozen_pos,
                                     n,
                                     use_fast_scl=use_fast_scl,
                                     cpu_only=cpu_only,
                                     use_scatter=use_scatter)

                    b = source([1,15,n])
                    b_rep = tf.tile(b, [bs, 1, 1])

                    # and run tf version (to be tested)
                    c = dec(b_rep).numpy()

                    for i in range(bs):
                        self.assertTrue(np.array_equal(c[0,:,:], c[i,:,:]))

    @pytest.mark.usefixtures("only_gpu")
    def test_tf_fun(self):
        """Test that graph mode works and XLA is supported."""

        bs = 10
        k = 16
        n = 32
        source = BinarySource()
        frozen_pos, _ = generate_5g_ranking(k, n)
        crc_degrees = [None, "CRC11"]
        for crc_degree in crc_degrees:
            for use_fast_scl in [False, True]:
                for cpu_only in [False, True]:
                    for use_scatter in [False, True]:
                        @tf.function
                        def run_graph(u):
                            return dec(u)

                        @tf.function(jit_compile=True)
                        def run_graph_xla(u):
                            return dec(u)
                        dec = PolarSCLDecoder(frozen_pos,
                                        n,
                                        use_fast_scl=use_fast_scl,
                                        cpu_only=cpu_only,
                                        use_scatter=use_scatter,
                                        crc_degree=crc_degree)

                        # test that for arbitrary input only binary values are
                        # returned
                        u = source([bs, n])
                        x = run_graph(u).numpy()

                        # execute the graph twice
                        x = run_graph(u).numpy()

                        # and change batch_size
                        u = source([bs+1, n])
                        x = run_graph(u).numpy()
                        if not cpu_only: # cpu only does not support XLA
                            # run same test for XLA (jit_compile=True)
                            u = source([bs, n])
                            x = run_graph_xla(u).numpy()
                            x = run_graph_xla(u).numpy()
                            u = source([bs+1, n])
                            x = run_graph_xla(u).numpy()

    # Filter warnings related to large ressource allocation
    @pytest.mark.filterwarnings("ignore: Required ressource allocation")
    @pytest.mark.usefixtures("only_gpu")
    def test_ref_implementation(self):
        """Test against pre-calculated results from internal implementation.

        Also verifies that all decoding options yield same results.

        Remark: results are for SC only, i.e., list_size=1.
        """

        ref_path = test_dir + '/codes/polar/'
        filename = ["P_128_37", "P_128_110", "P_256_128"]

        for f in filename:
            A = np.load(ref_path + f + "_Avec.npy")
            llr_ch = np.load(ref_path + f + "_Lch.npy")
            u_hat = np.load(ref_path + f + "_uhat.npy")
            frozen_pos = np.array(np.where(A==0)[0])
            info_pos = np.array(np.where(A==1)[0])

            n = len(frozen_pos) + len(info_pos)
            k = len(info_pos)

            for use_fast_scl in [False, True]:
                for cpu_only in [False, True]:
                    for use_scatter in [False, True]:
                        dec = PolarSCLDecoder(frozen_pos,
                                         n,
                                         list_size=1,
                                         use_fast_scl=use_fast_scl,
                                         cpu_only=cpu_only,
                                         use_scatter=use_scatter)
                        l_in = -1. * llr_ch # logits
                        u_hat_tf = dec(l_in).numpy()

                        # the output should be equal to the reference
                        self.assertTrue(np.array_equal(u_hat_tf, u_hat))

    def test_hybrid_scl(self):
        """Verify hybrid SC decoding option.

        Remark: XLA is currently not supported.
        """

        bs = 10
        n = 32
        k = 16
        crc_degree = "CRC11"
        list_sizes = [1, 2, 8, 32]

        frozen_pos, _ = generate_5g_ranking(k, n)
        source = BinarySource()
        enc = PolarEncoder(frozen_pos, n)
        enc_crc = CRCEncoder(crc_degree)
        k_crc = enc_crc.crc_length

        u = source([bs, k-k_crc])
        u_crc = enc_crc(u)
        c = enc(u_crc)
        llr_ch = 20.*(2.*c-1) # demod BPSK witout noise

        for list_size in list_sizes:
            dec = PolarSCLDecoder(frozen_pos,
                                n,
                                list_size=list_size,
                                use_hybrid_sc=True,
                                crc_degree=crc_degree)
            u_hat = dec(llr_ch)
            self.assertTrue(np.array_equal(u_crc.numpy(), u_hat.numpy()))

            # verify that graph can be executed
            @tf.function
            def run_graph(u):
                return dec(u)

            u = source([bs, n])
            # execute the graph twice
            x = run_graph(u).numpy()
            x = run_graph(u).numpy()
            # and change batch_size
            u = source([bs+1, n])
            x = run_graph(u).numpy()

    def test_dtype_flexible(self):
        """Test that output_dtype is variable."""

        batch_size = 100
        k = 30
        n = 64
        source = GaussianPriorSource()
        frozen_pos, _ = generate_5g_ranking(k, n)
        dtypes_supported = (tf.float16, tf.float32, tf.float64)

        for dt_in in dtypes_supported:
            for dt_out in dtypes_supported:
                llr = source([[batch_size, n], 0.5])
                llr = tf.cast(llr, dt_in)

                dec = PolarSCLDecoder(frozen_pos, n, output_dtype=dt_out)

                x = dec(llr)

                self.assertTrue(x.dtype==dt_out)

        # test that complex-valued inputs raise error
        llr = source([[batch_size, n], 0.5])
        llr_c = tf.complex(llr, tf.zeros_like(llr))
        dec = PolarSCLDecoder(frozen_pos, n, output_dtype=tf.float32)

        with self.assertRaises(TypeError):
            x = dec(llr_c)

    def test_return_crc(self):
        """Test that correct CRC status is returned."""

        k = 32
        n = 64
        bs = 100
        no = 1.
        num_mc_iter = 10

        channel = AWGN()
        source = BinarySource()
        mapper = Mapper("qam", 2)
        demapper = Demapper("app", "qam", 2)

        frozen_pos, _ = generate_5g_ranking(k, n)
        enc = PolarEncoder(frozen_pos, n)
        crc_enc = CRCEncoder("CRC24A")
        dec = PolarSCLDecoder(frozen_pos, n, crc_degree="CRC24A", return_crc_status=True)
        for _ in range(num_mc_iter):
            u = source((bs, 3, k-24))
            u_crc = crc_enc(u)
            c = enc(u_crc)
            x = mapper(c)
            y = channel((x, no))
            llr_ch = demapper((y, no))
            u_hat, crc_status = dec(llr_ch)

            # test for individual error patterns
            err_patt = tf.reduce_any(tf.not_equal(u_hat, u_crc), axis=-1)
            diffs = tf.cast(err_patt, tf.float32) - (1. - crc_status)
            num_diffs = tf.reduce_sum(tf.abs(diffs))
            self.assertTrue(num_diffs < 3) # allow a few CRC mis-detections

class TestPolarDecodingBP(unittest.TestCase):
    """Test Polar BP decoder."""

    def test_invalid_inputs(self):
        """Test against invalid values of n and frozen_pos."""

        # frozen vec to long
        n = 32
        frozen_pos = np.arange(n+1)
        with self.assertRaises(AssertionError):
            PolarBPDecoder(frozen_pos, n)

        # n not a pow of 2
        # frozen vec to long
        n = 32
        k = 12
        frozen_pos,_ = generate_5g_ranking(k, n)
        with self.assertRaises(AssertionError):
            PolarBPDecoder(frozen_pos, n+1)

        # test also valid shapes
        # (k, n)
        param_valid = [[0, 32], [10, 32], [32, 32], [100, 256],
                       [123, 1024], [1024, 1024]]

        for p in param_valid:
            frozen_pos, _ = generate_5g_ranking(p[0],p[1])
            PolarBPDecoder(frozen_pos, p[1])

        # no complex-valued input allowed
        with self.assertRaises(ValueError):
            frozen_pos,_ = generate_5g_ranking(32, 64)
            PolarBPDecoder(frozen_pos, 64, output_dtype=tf.complex64)

    def test_output_dim(self):
        """Test that output dims are correct (=n) and output is all-zero
         codeword."""

        # batch size
        bs = 10

        # (k, n)
        param_valid = [[1, 32],[10, 32], [32, 32], [100, 256], [123, 1024],
                      [1024, 1024]]
        for hard_out in [True, False]:
            for p in param_valid:
                frozen_pos, _ = generate_5g_ranking(p[0],p[1])
                dec = PolarBPDecoder(frozen_pos,
                                     p[1],
                                     hard_out=hard_out)
                # all-zero with BPSK (no noise);logits
                c = -10. * np.ones([bs, p[1]])
                u = dec(c).numpy()
                self.assertTrue(u.shape[-1]==p[0])
                if hard_out:
                    # also check that all-zero input yields all-zero output
                    u_hat = np.zeros([bs, p[0]])
                    self.assertTrue(np.array_equal(u, u_hat))

    def test_identity(self):
        """Test that info bits can be recovered if no noise is added."""

        bs = 10

        # (k, n)
        param_valid = [[1, 32], [10, 32], [32, 32], [100, 256], [123, 1024],
                      [1024, 1024]]

        for p in param_valid:
            source = BinarySource()
            frozen_pos, _ = generate_5g_ranking(p[0], p[1])
            enc = PolarEncoder(frozen_pos, p[1])
            dec = PolarBPDecoder(frozen_pos, p[1])

            u = source([bs, p[0]])
            c = enc(u)
            llr_ch = 20.*(2.*c-1) # demod BPSK witout noise
            u_hat = dec(llr_ch)

            self.assertTrue(np.array_equal(u.numpy(), u_hat.numpy()))

    def test_keras(self):
        """Test that Keras model can be compiled (supports dynamic shapes)."""

        bs = 10
        k = 100
        n = 128
        source = BinarySource()
        frozen_pos, _ = generate_5g_ranking(k, n)
        inputs = tf.keras.Input(shape=(n), dtype=tf.float32)
        x = PolarBPDecoder(frozen_pos, n)(inputs)
        model = tf.keras.Model(inputs=inputs, outputs=x)

        b = source([bs, n])
        model(b)
        # call twice to see that bs can change
        b2 = source([bs+1, n])
        model(b2)
        model.summary()

    def test_multi_dimensional(self):
        """Test against arbitrary shapes."""

        k = 120
        n = 256

        frozen_pos, _ = generate_5g_ranking(k, n)
        source = BinarySource()
        dec = PolarBPDecoder(frozen_pos, n)

        b = source([100, n])
        b_res = tf.reshape(b, [4, 5, 5, n])

        # encode 2D Tensor
        c = dec(b).numpy()
        # encode 4D Tensor
        c_res = dec(b_res).numpy()
        # and reshape to 2D shape
        c_res = tf.reshape(c_res, [100, k])
        # both version should yield same result
        self.assertTrue(np.array_equal(c, c_res))

    def test_batch(self):
        """Test that all samples in batch yield same output (for same input).
        """

        bs = 100
        k = 120
        n = 256

        frozen_pos, _ = generate_5g_ranking(k, n)
        source = BinarySource()
        dec = PolarBPDecoder(frozen_pos, n)

        b = source([1, 15, n])
        b_rep = tf.tile(b, [bs, 1, 1])

        # and run tf version (to be tested)
        c = dec(b_rep).numpy()

        for i in range(bs):
            self.assertTrue(np.array_equal(c[0,:,:], c[i,:,:]))

    def test_numerics(self):
        """Test for numerical stability with large llrs and many iterations.
        """

        bs = 100
        k = 120
        n = 256
        num_iter = 200

        for hard_out in [False, True]:
            frozen_pos, _ = generate_5g_ranking(k, n)
            source = GaussianPriorSource()
            dec = PolarBPDecoder(frozen_pos,
                                 n,
                                 hard_out=hard_out,
                                 num_iter=num_iter)

            b = source([[bs,n], 0.001]) # very large llrs

            c = dec(b).numpy()

            # all values are finite (not nan and not inf)
            self.assertTrue(np.sum(np.abs(1 - np.isfinite(c)))==0)

    @pytest.mark.usefixtures("only_gpu")
    def test_tf_fun(self):
        """Test that graph mode works and XLA is supported."""

        @tf.function
        def run_graph(u):
            return dec(u)

        @tf.function(jit_compile=True)
        def run_graph_xla(u):
            return dec(u)

        bs = 10
        k = 32
        n = 64
        num_iter = 10
        source = BinarySource()
        frozen_pos, _ = generate_5g_ranking(k, n)
        dec = PolarBPDecoder(frozen_pos, n, num_iter=num_iter)

        # test that for arbitrary input only 0,1 values are returned
        u = source([bs, n])
        x = run_graph(u).numpy()

        # execute the graph twice
        x = run_graph(u).numpy()

        # and change batch_size
        u = source([bs+1, n])
        x = run_graph(u).numpy()
        x = run_graph(u).numpy()

        # Currently not supported
        # run same test for XLA (jit_compile=True)
        #u = source([bs, n])
        #x = run_graph_xla(u).numpy()
        #x = run_graph_xla(u).numpy()
        #u = source([bs+1, n])
        #x = run_graph_xla(u).numpy()

    @pytest.mark.usefixtures("only_gpu")
    def test_ref_implementation(self):
        """Test against Numpy reference implementation.

        Test hard and soft output.
        """

        def boxplus_np(x, y):
            """Check node update (boxplus) for LLRs in numpy.

            See [Stimming_LLR]_ and [Hashemi_SSCL]_ for detailed equations.
            """
            x_in = np.maximum(np.minimum(x, llr_max), -llr_max)
            y_in = np.maximum(np.minimum(y, llr_max), -llr_max)
            # avoid division for numerical stability
            llr_out = np.log(1 + np.exp(x_in + y_in))
            llr_out -= np.log(np.exp(x_in) + np.exp(y_in))

            return llr_out

        def decode_bp(llr_ch, n_iter, frozen_pos, info_pos):

            n = llr_ch.shape[-1]
            bs = llr_ch.shape[0]
            n_stages = int(np.log2(n))

            msg_r = np.zeros([bs, n_stages+1, n])
            msg_l = np.zeros([bs, n_stages+1, n])

            # init llr_ch
            msg_l[:, n_stages, :] = -1*llr_ch.numpy()

            # init frozen positions with infty
            msg_r[:, 0, frozen_pos] = llr_max

            # and decode
            for iter in range(n_iter):

                # update r messages
                for s in range(n_stages):
                    # calc indices
                    ind_range = np.arange(int(n/2))
                    ind_1 = ind_range * 2 - np.mod(ind_range, 2**(s))
                    ind_2 = ind_1 + 2**s

                    # load messages
                    l1_in = msg_l[:, s+1, ind_1]
                    l2_in = msg_l[:, s+1, ind_2]
                    r1_in = msg_r[:, s, ind_1]
                    r2_in = msg_r[:, s, ind_2]
                    # r1_out
                    msg_r[:, s+1, ind_1] = boxplus_np(r1_in, l2_in + r2_in)
                    # r2_out
                    msg_r[:, s+1, ind_2] = boxplus_np(r1_in, l1_in) + r2_in

                # update l messages
                for s in range(n_stages-1, -1, -1):
                    ind_range = np.arange(int(n/2))
                    ind_1 = ind_range * 2 - np.mod(ind_range, 2**(s))
                    ind_2 = ind_1 + 2**s

                    l1_in = msg_l[:, s+1, ind_1]
                    l2_in = msg_l[:, s+1, ind_2]
                    r1_in = msg_r[:, s, ind_1]
                    r2_in = msg_r[:, s, ind_2]

                    # l1_out
                    msg_l[:, s, ind_1] = boxplus_np(l1_in, l2_in + r2_in)
                    # l2_out
                    msg_l[:, s, ind_2] = boxplus_np(r1_in, l1_in) + l2_in

            # recover u_hat
            u_hat_soft = msg_l[:, 0, info_pos]
            u_hat = 0.5 * (1 - np.sign(u_hat_soft))
            return u_hat, u_hat_soft

        # generate llr_ch
        noise_var = 0.3
        num_iters = [5, 10, 20, 40]
        llr_max = 19.3
        bs = 100
        n = 128
        k = 64
        frozen_pos, info_pos = generate_5g_ranking(k, n)

        for num_iter in num_iters:

            source = GaussianPriorSource()
            llr_ch = source([[bs, n], noise_var])

            # and decode
            dec_bp = PolarBPDecoder(frozen_pos, n,
                                    hard_out=True, num_iter=num_iter)
            dec_bp_soft = PolarBPDecoder(frozen_pos, n,
                                         hard_out=False, num_iter=num_iter)

            u_hat_bp = dec_bp(llr_ch).numpy()
            u_hat_bp_soft = dec_bp_soft(llr_ch,).numpy()

            # and run BP decoder
            u_hat_ref, u_hat_ref_soft = decode_bp(llr_ch,
                                                num_iter,
                                                frozen_pos,
                                                info_pos)

            # the output should be equal to the reference
            self.assertTrue(np.array_equal(u_hat_bp, u_hat_ref))
            self.assertTrue(np.allclose(-u_hat_bp_soft,
                                        u_hat_ref_soft,
                                        rtol=5e-2,
                                        atol=5e-3))

    def test_dtype_flexible(self):
        """Test that output dtype is variable."""

        batch_size = 100
        k = 30
        n = 64
        source = GaussianPriorSource()
        frozen_pos, _ = generate_5g_ranking(k, n)
        dtypes_supported = (tf.float16, tf.float32, tf.float64)

        for dt_in in dtypes_supported:
            for dt_out in dtypes_supported:
                llr = source([[batch_size, n], 0.5])
                llr = tf.cast(llr, dt_in)

                dec = PolarBPDecoder(frozen_pos, n, output_dtype=dt_out)

                x = dec(llr)

                self.assertTrue(x.dtype==dt_out)

        # test that complex inputs raise error
        llr = source([[batch_size, n], 0.5])
        llr_c = tf.complex(llr, tf.zeros_like(llr))
        dec = PolarBPDecoder(frozen_pos, n, output_dtype=tf.float32)

        with self.assertRaises(TypeError):
            x = dec(llr_c)

class TestPolarDecoding5G(unittest.TestCase):

    def test_invalid_inputs(self):
        """Test against invalid input values.

        Note: consistency of code parameters is already checked by the encoder.
        """
        enc = Polar5GEncoder(40, 60)
        with self.assertRaises(AssertionError):
            Polar5GDecoder(enc, dec_type=1)
        with self.assertRaises(ValueError):
            Polar5GDecoder(enc, dec_type="ABC")
        with self.assertRaises(AssertionError):
            Polar5GDecoder("SC")

    # Filter warnings related to large ressource allocation
    @pytest.mark.filterwarnings("ignore: Required ressource allocation")
    def test_identity_de_ratematching(self):
        """Test that info bits can be recovered if no noise is added and
        dimensions are correct."""

        bs = 10

        # Uplink scenario
        # (k,n)
        param_valid_ul = [[12, 20], [20, 44], [100, 257], [123, 897],
                       [1013, 1088]]
        # Uplink scenario
        # (k,n)
        param_valid_dl = [[1, 25], [20, 44], [140, 576]]

        for ch_type in ["uplink", "downlink"]:
            if ch_type=="uplink":
                param_valid = param_valid_ul
            else:
                param_valid = param_valid_dl

            for p in param_valid:
                for dec_type in ["SC", "SCL", "hybSCL", "BP"]:
                    source = BinarySource()
                    enc = Polar5GEncoder(p[0], p[1], channel_type=ch_type)
                    dec = Polar5GDecoder(enc, dec_type=dec_type)

                    u = source([bs, p[0]])
                    c = enc(u)
                    self.assertTrue(c.numpy().shape[-1]==p[1])
                    llr_ch = 20.*(2.*c-1) # demod BPSK without noise
                    u_hat = dec(llr_ch)

                    self.assertTrue(np.array_equal(u.numpy(), u_hat.numpy()))
                    # Uplink scenario


    # Filter warnings related to large ressource allocation
    @pytest.mark.filterwarnings("ignore: Required ressource allocation")
    def test_keras(self):
        """Test that Keras model can be compiled (supports dynamic shapes)."""

        bs = 10
        k = 100
        n = 145
        source = BinarySource()
        enc = Polar5GEncoder(k, n)

        for dec_type in ["SC", "SCL", "hybSCL", "BP"]:
            inputs = tf.keras.Input(shape=(n), dtype=tf.float32)
            x = Polar5GDecoder(enc, dec_type=dec_type)(inputs)
            model = tf.keras.Model(inputs=inputs, outputs=x)

            b = source([bs,n])
            model(b)
            # call twice to see that bs can change
            b2 = source([bs+1,n])
            model(b2)
            model.summary()

    # Filter warnings related to large ressource allocation
    @pytest.mark.filterwarnings("ignore: Required ressource allocation")
    def test_multi_dimensional(self):
        """Test against arbitrary shapes."""

        k = 120
        n = 237

        enc = Polar5GEncoder(k, n)
        source = BinarySource()

        # also verifies that interleaver support n-dimensions
        for dec_type in ["SC", "SCL", "hybSCL", "BP"]:
            for ch_type in ["uplink", "downlink"]:

                enc = Polar5GEncoder(k, n, channel_type=ch_type)
                dec = Polar5GDecoder(enc, dec_type=dec_type)

                b = source([100, n])
                b_res = tf.reshape(b, [4, 5, 5, n])

                # encode 2D Tensor
                c = dec(b).numpy()
                # encode 4D Tensor
                c_res = dec(b_res).numpy()
                # and reshape to 2D shape
                c_res = tf.reshape(c_res, [100, k])
                # both version should yield same result
                self.assertTrue(np.array_equal(c, c_res))

    # Filter warnings related to large ressource allocation
    @pytest.mark.filterwarnings("ignore: Required ressource allocation")
    def test_batch(self):
        """Test that all samples in batch yield same output (for same input).
        """
        bs = 100
        k = 95
        n = 145

        enc = Polar5GEncoder(k, n)
        source = GaussianPriorSource()

        for dec_type in ["SC", "SCL", "hybSCL", "BP"]:
            dec = Polar5GDecoder(enc, dec_type=dec_type)

            llr = source([[1,4,n], 0.5])
            llr_rep = tf.tile(llr, [bs, 1, 1])

            # and run tf version (to be tested)
            c = dec(llr_rep).numpy()

            for i in range(bs):
                self.assertTrue(np.array_equal(c[0,:,:], c[i,:,:]))

    @pytest.mark.usefixtures("only_gpu")
    def test_tf_fun(self):
        """Test that tf.function decorator works
        include xla compiler test."""

        bs = 10
        k = 45
        n = 67
        enc = Polar5GEncoder(k, n)
        source = GaussianPriorSource()

        # hybSCL does not support graph mode!
        for dec_type in ["SC", "SCL", "BP"]:
            dec = Polar5GDecoder(enc, dec_type=dec_type)

            @tf.function
            def run_graph(u):
                return dec(u)

            @tf.function(jit_compile=True)
            def run_graph_xla(u):
                return dec(u)

            # test that for arbitrary input only binary values are returned
            u = source([[bs, n], 0.5])
            x = run_graph(u).numpy()

            # execute the graph twice
            x = run_graph(u).numpy()

            # and change batch_size
            u = source([[bs+1, n], 0.5])
            x = run_graph(u).numpy()

            # run same test for XLA (jit_compile=True)
            # BP does currently not support XLA
            if dec_type != "BP":
                u = source([[bs, n], 0.5])
                x = run_graph_xla(u).numpy()
                x = run_graph_xla(u).numpy()
                u = source([[bs+1, n], 0.5])
                x = run_graph_xla(u).numpy()

    def test_dtype_flexible(self):
        """Test that output dtype can be variable."""

        batch_size = 100
        k = 30
        n = 64
        source = GaussianPriorSource()
        enc = Polar5GEncoder(k, n)
        dtypes_supported = (tf.float16, tf.float32, tf.float64)

        for dt_in in dtypes_supported:
            for dt_out in dtypes_supported:
                llr = source([[batch_size, n], 0.5])
                llr = tf.cast(llr, dt_in)

                dec = Polar5GDecoder(enc, output_dtype=dt_out)

                x = dec(llr)

                self.assertTrue(x.dtype==dt_out)

        # test that complex inputs raise error
        llr = source([[batch_size, n], 0.5])
        llr_c = tf.complex(llr, tf.zeros_like(llr))
        dec = Polar5GDecoder(enc, output_dtype=tf.float32)

        with self.assertRaises(TypeError):
            x = dec(llr_c)

    def test_return_crc(self):
        """Test that correct CRC status is returned."""

        k = 32
        n = 64
        bs = 100
        no = 1.
        num_mc_iter = 10

        channel = AWGN()
        source = BinarySource()
        mapper = Mapper("qam", 2)
        demapper = Demapper("app", "qam", 2)

        for channel_type in ("downlink", "uplink"):
            enc = Polar5GEncoder(k, n, channel_type=channel_type)
            dec = Polar5GDecoder(enc, "SCL", return_crc_status=True)
            for it in range(num_mc_iter):
                u = source((bs, 3, k))
                c = enc(u)
                x = mapper(c)
                y = channel((x, no))
                llr_ch = demapper((y, no))
                u_hat, crc_status = dec(llr_ch)

                # test for individual error patterns
                err_patt = tf.reduce_any(tf.not_equal(u_hat, u), axis=-1)
                diffs = tf.cast(err_patt, tf.float32) - (1. - crc_status)
                num_diffs = tf.reduce_sum(tf.abs(diffs)).numpy()
                self.assertTrue(num_diffs < 5) # allow a few CRC mis-detections


