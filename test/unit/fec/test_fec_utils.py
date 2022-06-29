#
# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
try:
    import sionna
except ImportError as e:
    import sys
    sys.path.append("../")
from numpy.lib.npyio import load

import unittest
import pytest  # required for filter warnings
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
from sionna.fec.utils import bin2int_tf, j_fun, j_fun_inv, j_fun_tf, j_fun_inv_tf,GaussianPriorSource, llr2mi, bin2int, int2bin, int2bin_tf, alist2mat, load_alist, gm2pcm, pcm2gm, verify_gm_pcm, make_systematic, load_parity_check_examples, LinearEncoder
from sionna.utils import log2, log10, BinarySource
from sionna.fec.polar.utils import generate_dense_polar, generate_5g_ranking
from sionna.fec.polar import PolarEncoder

class TestFECUtils(unittest.TestCase):
    """Test FEC utilities."""

    def test_log2(self):
        """Test log2. against numpy implementation."""

        x = np.array([1., 2., 3., 100., 123., 1337])
        y_np = np.log2(x)
        y_tf = log2(tf.cast(x, dtype=tf.float32))
        self.assertTrue(np.allclose(y_np, y_tf.numpy()))

    def test_log10(self):
        """Test log10 against numpy implementation."""

        x = np.array([1., 2., 3., 100., 123., 1337])
        y_np = np.log10(x)
        y_tf = log10(tf.cast(x, dtype=tf.float32))
        self.assertTrue(np.allclose(y_np, y_tf.numpy()))

    def test_j_fun_np(self):
        """Test that J(inv_j(x))==x for numpy implementation."""

        x = np.arange(0.01,20,0.1)
        y = j_fun(x)
        z = j_fun_inv(y)
        self.assertTrue(np.allclose(x, z, rtol=0.001))

    def test_j_fun_tf(self):
        """Test that J(inv_j(x))==x for Tensorflow implementation."""

        x = np.arange(0.01,20,0.1)
        x_tf = tf.constant(x, dtype=tf.float32)
        y = j_fun_tf(x_tf)
        z = j_fun_inv_tf(y)
        self.assertTrue(np.allclose(x, z.numpy(), rtol=0.001))

    def test_gaussian_prior(self):
        """Test that Gaussian priors have the correct mutual information.

        Indirectly, also validates llr2mi function."""

        num_samples =  [100000]
        s = GaussianPriorSource(specified_by_mi=True)
        mi = np.arange(0.01, 0.99, 0.01)
        ia_hat = np.zeros_like(mi)
        for i, mii in enumerate(mi):
            x = s([num_samples, mii])
            ia_hat[i] = llr2mi(x).numpy()
        self.assertTrue(np.allclose(mi, ia_hat, atol=1e-2)) # this is a
        # montecarlo sim and Gaussian approx; we can set the tolerance
        # relatively high

    def test_gaussian_prior_sigma(self):
        """Test that Gaussian priors have correct sigma_ch.

        The standard_dev of the generated LLRs must be:
        sigma_llr^2 = 4 / sigma_ch^2
        and
        mu_llr = sigma_llr^2 / 2
        """

        num_samples =  [100000]
        s = GaussianPriorSource(specified_by_mi=False)
        sigma_ch= np.arange(0.3, 5, 0.1)

        sigma_target = np.sqrt(4 * sigma_ch**(-2))
        mu_target = sigma_target**(2) / 2

        sigma_hat = np.zeros_like(sigma_ch)
        mu_hat = np.zeros_like(sigma_ch)

        for i, no in enumerate((sigma_ch)):
            x = s([num_samples, no**2])
            sigma_hat[i] = tf.sqrt(tf.reduce_mean(
                                        tf.pow(x - tf.reduce_mean(x),
                                               2))).numpy()
            mu_hat[i] = tf.reduce_mean(x).numpy()

        # this is a montecarlo sim and approximated; we can set the tolerance
        # relatively high
        self.assertTrue(np.allclose(sigma_target, sigma_hat, atol=1e-1))
        # -1.* due to logits vs llrs
        self.assertTrue(np.allclose(mu_target, -1. * mu_hat, atol=1e-1))

    def test_gaussian_prior_tf_fun(self):
        """Test that Gaussian source works in graph mode and supports XLA.
        """

        def run_graph(num_samples):
            x = s([num_samples, 1])
            return x

        def run_graph_xla(num_samples):
            x = s([num_samples, 1.])
            return x

        num_samples = [100000]
        s = GaussianPriorSource()
        x = run_graph(num_samples)
        x = run_graph_xla(num_samples)

    def test_keras(self):
        """Test that Keras model of GaussianPriorSource can be compiled."""

        inputs = tf.keras.Input(shape=(1), dtype=tf.float32)
        x = GaussianPriorSource()([tf.shape(inputs), 0.])
        model = tf.keras.Model(inputs=inputs, outputs=x)

        # test that output batch dim is none
        self.assertTrue(model.output_shape[0] is None)

        # test that model can be called
        model(100)
        # call twice to see that bs can change
        model(101)
        model.summary()

    def test_bin2int(self):
        """Test bin2int function against pre-defined cases."""

        # each entry defines [#testcase, #groundtruth]
        patterns = [[[1, 0, 1], 5],
                    [[1], 1],
                    [[0], 0],
                    [[1, 1, 1, 1], 15],
                    [[0, 1, 0, 1, 1, 1 , 0], 46]]

        for p in patterns:
            # p[0] is input
            # p[1] is desired output
            x = bin2int(p[0])
            self.assertEqual(x, p[1])

    def test_int2bin(self):
        """Test int2bin function against pre-defined cases."""

        # each entry defines [#testcase, #len, #groundtruth]
        patterns = [[5, 3, [1, 0, 1]],
                    [1, 1, [1]],
                    [1, 2, [0, 1]],
                    [15, 4, [1, 1, 1, 1]],
                    [46, 7, [0, 1, 0, 1, 1, 1, 0]]]

        for p in patterns:
            # p[0] is input
            # p[1] is len_
            # p[2] is desired output
            x = int2bin(p[0], p[1])
            self.assertEqual(x, p[2])

    def test_bin2int_tf(self):
        """Test bin2int function against pre-defined cases."""

        # each entry defines [#testcase, #groundtruth]
        patterns = [[[1, 0, 1], 5],
                    [[1], 1],
                    [[0], 0],
                    [[1, 1, 1, 1], 15],
                    [[0, 1, 0, 1, 1, 1, 0], 46]]

        for p in patterns:
            # p[0] is input
            # p[1] is desired output
            x = bin2int_tf(p[0]).numpy()
            self.assertEqual(x, p[1])

    def test_int2bin_tf(self):
        """Test int2bin function against pre-defined cases."""

        # each entry defines [#testcase, #len, #groundtruth]
        patterns = [[5, 3, [1, 0, 1]],
                    [1, 1, [1]],
                    [1, 2, [0, 1]],
                    [15, 4, [1, 1, 1, 1]],
                    [46, 7, [0, 1, 0, 1, 1, 1, 0]],
                    [13, 3, [1, 0, 1]],
                    [6, 0, []],
                    [[6, 12], 4, [[0, 1, 1, 0], [1, 1, 0, 0]]]]

        for p in patterns:
            # p[0] is input
            # p[1] is len_
            # p[2] is desired output
            x = int2bin_tf(p[0], p[1]).numpy()
            self.assertTrue(np.array_equal(x, p[2]))

    def test_alist(self):
        """Test alist2mat with explicit example."""

        # (7,4) Hamming code
        alist = [[7, 3],
                 [3, 4],
                 [1, 1, 1, 2, 2, 2, 3],
                 [4, 4, 4],
                 [1, 0, 0],
                 [2, 0, 0],
                 [3, 0, 0],
                 [1, 2, 0],
                 [1, 3, 0],
                 [2, 3, 0],
                 [1, 2, 3],
                 [1, 4, 5, 7],
                 [2, 4, 6, 7],
                 [3, 5, 6, 7]]

        pcm,k,n,r = alist2mat(alist, verbose=False)

        # test for valid code parameters
        self.assertTrue(k==4)
        self.assertTrue(n==7)
        self.assertTrue(r==4/7)
        self.assertTrue(len(pcm)==3)
        self.assertTrue(len(pcm[0])==7)

        pcm_true = [[1, 0, 0, 1, 1, 0, 1],
                    [0, 1, 0, 1, 0, 1, 1],
                    [0, 0, 1, 0, 1, 1, 1]]

        self.assertTrue(np.array_equal(pcm_true, pcm))

    def test_alist2(self):
        """Test to load example alist files.
        """

        path = "codes/ldpc/wimax_576_0.5.alist"

        # load file
        alist = load_alist(path)

        # convert to full pcm
        pcm, k, n, r = alist2mat(alist, verbose=False)

        # check parameters for consistency
        self.assertTrue(k==288)
        self.assertTrue(n==576)
        self.assertTrue(r==0.5)
        self.assertTrue(len(pcm)==n-k)
        self.assertTrue(len(pcm[0])==n)

    def test_verify_gm_pcm(self):
        """Test that verify_gm_pcm identifies invalid pairs of gm/pcm."""

        n = 20
        k = 12

        # invalid shapes pcm
        gm = np.zeros((n,k))
        pcm = np.zeros((n,k))
        with self.assertRaises(AssertionError):
            verify_gm_pcm(gm, pcm)

        # invalid shapes gm
        gm = np.zeros((n,n-k))
        pcm = np.zeros((n,k))
        with self.assertRaises(AssertionError):
            verify_gm_pcm(gm, pcm)

        id = 0
        pcm, k, n, _ = load_parity_check_examples(pcm_id=id)
        gm = pcm2gm(pcm)

        # verify correct pair passes test
        self.assertTrue(verify_gm_pcm(gm, pcm))

        # incorrect pair should not pass the test
        id = 3 # use longer matrix (as it requires column swaps)
        pcm, k, n, _ = load_parity_check_examples(pcm_id=id)
        gm = pcm2gm(pcm)
        self.assertTrue(verify_gm_pcm(gm, pcm))
        gm_sys,_ = make_systematic(gm)
        self.assertFalse(verify_gm_pcm(gm_sys, pcm))

        # test nonbinary input
        id = 0
        pcm, k, n, _ = load_parity_check_examples(pcm_id=id)
        gm = pcm2gm(pcm)
        gm_nonbin = np.copy(gm)
        pcm_nonbin = np.copy(pcm)
        gm_nonbin[0, 0] = 2 # make elements non-binary
        pcm_nonbin[0, 0] = 2 # make elements non-binary
        with self.assertRaises(AssertionError):
            verify_gm_pcm(gm_nonbin, pcm)
        with self.assertRaises(AssertionError):
            verify_gm_pcm(gm, pcm_nonbin)

    def test_pcm2gm(self):
        """test pcm2gm function for consistency.

        Note: pcm2gm relies on the function make_systematic which is
        tested separately."""

        for id in [0, 1, 2, 3, 4]:
            pcm, _, _, _ = load_parity_check_examples(pcm_id=id)

            gm = pcm2gm(pcm)
            # verify that gm and pcm are (binary) orthogonal
            self.assertTrue(verify_gm_pcm(gm, pcm))

    def test_gm2pcm(self):
        """test gm2pcm function for consistency.

        Note: gm2pcm relies on the core function make_systematic which is
        tested separately."""

        # for the tests we interpret the built-in pcms as gm and try to find an
        # orthogonal pcm
        for id in [0, 1, 2, 3, 4]:
            # pcm is interpreted as gm
            gm, _, _, _ = load_parity_check_examples(pcm_id=id)

            pcm = gm2pcm(gm)
            # verify that gm and pcm are (binary) orthogonal
            self.assertTrue(verify_gm_pcm(gm, pcm))

    def test_load_parity_check(self):
        """Test that code parameters are correct."""

        # test for all integrated examples
        ids = (0, 1, 2, 3, 4)
        for id in ids:
            pcm, k, n, r = load_parity_check_examples(id, verbose=False)

            n_pcm = pcm.shape[1]
            k_pcm = n_pcm - pcm.shape[0]
            self.assertTrue(k==k_pcm)
            self.assertTrue(n==n_pcm)
            self.assertTrue(r==k_pcm/n_pcm)

            # verify that pcm is binary
            self.assertTrue(((pcm==0) | (pcm==1)).all())

    # filter a user warning rearding all-zero column test
    # as we use this specific structure intentionally
    @pytest.mark.filterwarnings("ignore: All-zero column")
    def test_make_systematic(self):
        """Test that shapes do not change and that identity matrix is found."""

        # test for consistency
        for id in [0, 1, 2, 3, 4]:
            for is_pcm in [False, True]:
                pcm, k, n, _ = load_parity_check_examples(pcm_id=id)
                m = n - k
                pcm_sys,_ = make_systematic(np.array(pcm), is_pcm=is_pcm)

                # test that shapes do not change
                self.assertEqual(pcm.shape[0], pcm_sys.shape[0])
                self.assertEqual(pcm.shape[1], pcm_sys.shape[1])

                # test that identity is part of matrix
                if is_pcm:
                    self.assertTrue(np.array_equal(np.eye(m), pcm_sys[:,-m:]))
                else:
                    self.assertTrue(np.array_equal(np.eye(m), pcm_sys[:,:m]))

        # Test that non full rank raises ValueError
        id = 0
        pcm, _, _, _ = load_parity_check_examples(pcm_id=id)
        pcm[1,:] = pcm[0,:] # overwrite one row (non-full rank)

        with self.assertRaises(ValueError):
            make_systematic(pcm, is_pcm=True)

        # test with all-zero and all-one inputs
        k = 13
        n = 20
        for is_pcm in (False, True):
            mat = np.zeros((k,n))
            with self.assertRaises(ValueError):
                make_systematic(mat, is_pcm=is_pcm)
            mat = np.ones((k,n))
            with self.assertRaises(ValueError):
                make_systematic(mat, is_pcm=is_pcm)

class TestGenericLinearEncoder(unittest.TestCase):
    """Test Generic Linear Encoder."""

    def test_dim_mismatch(self):
        """Test against inconsistent inputs. """
        id = 2
        pcm, k, _, _ = load_parity_check_examples(id)
        bs = 20
        enc = LinearEncoder(pcm, is_pcm=True)

        # test for non-invalid input shape
        with self.assertRaises(BaseException):
            x = enc(tf.zeros([bs, k+1]))

        # test for non-binary matrix
        with self.assertRaises(BaseException):
            pcm[0,0]=2
            enc = LinearEncoder(pcm) # we interpret the pcm as gm for this test

        # test for non-binary matrix
        with self.assertRaises(BaseException):
            pcm[0,0]=2
            enc = LinearEncoder(pcm, is_pcm=True)

    def test_tf_fun(self):
        """Test that tf.function works as expected and XLA is supported."""

        @tf.function
        def run_graph(u):
            c = enc(u)
            return c

        @tf.function(jit_compile=True)
        def run_graph_xla(u):
            c = enc(u)
            return c

        id = 2
        pcm, k, _, _ = load_parity_check_examples(id)
        bs = 20
        enc = LinearEncoder(pcm, is_pcm=True)
        source = BinarySource()

        u = source([bs,k])
        run_graph(u)
        run_graph_xla(u)

    def test_dtypes_flexible(self):
        """Test that encoder supports variable dtypes and
        yields same result."""

        dt_supported = (tf.float16, tf.float32, tf.float64, tf.int32, tf.int64)

        id = 2
        pcm, k, _, _ = load_parity_check_examples(id)
        bs = 20
        enc_ref = LinearEncoder(pcm, is_pcm=True, dtype=tf.float32)
        source = BinarySource()

        u = source([bs, k])
        c_ref = enc_ref(u)

        for dt in dt_supported:
            enc = LinearEncoder(pcm, is_pcm=True, dtype=dt)
            u_dt = tf.cast(u, dt)
            c = enc(u_dt)

            c_32 = tf.cast(c, tf.float32)

            self.assertTrue(np.array_equal(c_ref.numpy(), c_32.numpy()))

    def test_multi_dimensional(self):
        """Test against arbitrary input shapes.

        The encoder should only operate on axis=-1.
        """
        id = 3
        pcm, k, n, _ = load_parity_check_examples(id)
        shapes =[[10, 20, 30, k], [1, 40, k], [10, 2, 3, 4, 3, k]]
        enc = LinearEncoder(pcm, is_pcm=True)
        source = BinarySource()

        for s in shapes:
            u = source(s)
            u_ref = tf.reshape(u, [-1, k])

            c = enc(u) # encode with shape s
            c_ref = enc(u_ref) # encode as 2-D array
            s[-1] = n
            c_ref = tf.reshape(c_ref, s)
            self.assertTrue(np.array_equal(c.numpy(), c_ref.numpy()))

        # and verify that wrong last dimension raises an error
        with self.assertRaises(tf.errors.InvalidArgumentError):
            s = [10, 2, k-1]
            u = source(s)
            x = enc(u)

    def test_against_baseline(self):
        """Test that PolarEncoder leads to same result.
        """
        bs = 1000
        k = 57
        n = 128

        # generate polar frozen positions
        f,_ = generate_5g_ranking(k, n)

        enc_ref = PolarEncoder(f, n) # reference encoder

        pcm, gm = generate_dense_polar(f, n) # get polar encoding matrix
        enc = LinearEncoder(gm)

        # draw random info bits
        source = BinarySource()
        u = source([bs, k])

        # encode u with both encoders
        c = enc(u)
        c_ref = enc_ref(u)

        # and compare results
        self.assertTrue(np.array_equal(c.numpy(), c_ref.numpy()))

    def test_keras(self):
        """Test that Keras model can be compiled (supports dynamic shapes)."""
        bs = 10
        id = 2
        pcm, k, _, _ = load_parity_check_examples(id)

        source = BinarySource()

        inputs = tf.keras.Input(shape=(k), dtype=tf.float32)
        x = LinearEncoder(pcm, is_pcm=True)(inputs)
        model = tf.keras.Model(inputs=inputs, outputs=x)

        b = source([bs,k])
        model(b)
        # call twice to see that bs can change
        b2 = source([bs+1,k])
        model(b2)
        model.summary()

    def test_random_matrices(self):
        """Test against random parity-check matrices."""

        n_trials = 100 # test against multiple random pcm realizations
        bs = 100
        k = 89
        n = 123
        source = BinarySource()

        for _ in range(n_trials):
            # sample a random matrix
            pcm = np.random.uniform(low=0, high=2, size=(n-k, n)).astype(int)

            # catch internal errors due to non-full rank of pcm (randomly
            # sampled!)
            # in this test we only test that if the encoder initalization
            # succeeds and the resulting encoder object produces valid codewords
            try:
                enc = LinearEncoder(pcm, is_pcm=True)
            except:
                pass # ignore this pcm realization

            u = source([bs, k])
            c = enc(u)
            # verify that all codewords fullfil all parity-checks
            c = tf.expand_dims(c, axis=2)
            pcm = tf.expand_dims(tf.cast(pcm, tf.float32),axis=0)
            s = tf.matmul(pcm,c).numpy()
            s = np.mod(s, 2)
            self.assertTrue(np.sum(np.abs(s))==0)

