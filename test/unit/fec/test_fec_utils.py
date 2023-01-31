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
from sionna.fec.utils import bin2int_tf, j_fun, j_fun_inv, j_fun_tf, j_fun_inv_tf, GaussianPriorSource, llr2mi, bin2int, int2bin, int2bin_tf, alist2mat, load_alist, gm2pcm, pcm2gm, verify_gm_pcm, make_systematic, load_parity_check_examples, LinearEncoder, generate_reg_ldpc, generate_prng_seq, int_mod_2
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


    def test_generate_reg_ldpc(self):
        """Test LDPC generator function."""

        # fix seed of rng
        np.random.seed(1337)
        # v,c,n
        params = [[3,6,100],
                  [1,10,1000],
                  [3,6,10000],
                  [2,7,703]]
        for v,c,n_des in params:
            pcm, k, n, r = generate_reg_ldpc(v, c, n_des, verbose=False)

            # test that rate is correct
            self.assertTrue(r==k/n)
            # test that VN degree equals v
            self.assertTrue((np.sum(pcm,axis=0)==v).all())
            # test that CN degree equals v
            self.assertTrue((np.sum(pcm,axis=1)==c).all())
            # test that resulting length never decreases
            self.assertTrue(n>=n_des)
            # test that pcm has expected shape
            self.assertTrue(pcm.shape[0]==n-k)
            self.assertTrue(pcm.shape[1]==n)

    def test_gen_rand_seq(self):
        """Test random sequence generator."""

        l = 100
        # check valid inputs
        n_rs = [0, 10, 65535]
        n_ids = [0, 10, 65535]
        s_old = None
        for n_r in n_rs:
            for n_id  in n_ids:
                s = generate_prng_seq(l, n_r, n_id)
                # verify that new sequence is unique
                if s_old is not None:
                    self.assertFalse(np.array_equal(s,s_old))
                s_old = s

        # test against invalid inputs
        l = -1
        with self.assertRaises(AssertionError):
            generate_prng_seq(l, n_r, n_id)

        n_rs = [-0, 1.2, 65536] # invalid
        n_ids = [0, 10, 65535] # valid
        for n_r in n_rs:
            for n_id  in n_ids:
                with self.assertRaises(AssertionError):
                    generate_prng_seq(l, n_r, n_id)

        n_rs = [0, 10, 65535] # valid
        n_ids = [-0, 1.2, 65536] # invalid
        for n_r in n_rs:
            for n_id  in n_ids:
                with self.assertRaises(AssertionError):
                    generate_prng_seq(l, n_r, n_id)

        # test against reference example
        n_rnti = 20001
        n_id = 41
        l = 100
        s_ref = np.array([0., 1., 1., 1., 1., 0., 0., 0., 0., 1., 0., 1., 0.,
                          1., 1., 1., 0., 0., 0., 1., 1., 1., 0., 0., 0., 1.,
                          1., 0., 0., 1., 1., 1., 0., 1., 0., 0., 1., 1., 1.,
                          0., 1., 0., 0., 0., 0., 0., 1., 1., 1., 0., 1., 1.,
                          0., 1., 1., 0., 0., 0., 1., 0., 0., 1., 0., 0., 1.,
                          0., 0., 0., 0., 0., 0., 1., 1., 1., 0., 1., 0., 0.,
                          1., 1., 0., 1., 1., 1., 0., 0., 0., 0., 0., 1., 0.,
                          1., 1., 1., 1., 1., 1., 1., 0., 0.])
        s = generate_prng_seq(l, n_rnti, n_id)
        self.assertTrue(np.array_equal(s, s_ref))

        # and test against wrong parameters
        s = generate_prng_seq(l, n_rnti, n_id+1)
        self.assertFalse(np.array_equal(s, s_ref))
        s = generate_prng_seq(l, n_rnti+1, n_id)
        self.assertFalse(np.array_equal(s, s_ref))

        # test that explicit value of c_init overwrites other parameters
        c_init = 1337
        s_ref = generate_prng_seq(l, c_init=c_init)
        # no effect expected (c_init overwrites other inputs)
        s = generate_prng_seq(l, n_rnti, n_id, c_init)
        self.assertTrue(np.array_equal(s, s_ref))
        s = generate_prng_seq(l, n_rnti+1, n_id, c_init)
        self.assertTrue(np.array_equal(s, s_ref))
        s = generate_prng_seq(l, n_rnti, n_id+2, c_init)
        self.assertTrue(np.array_equal(s, s_ref))
        # different sequence expected as c_init changes
        s = generate_prng_seq(l, n_rnti, n_id, c_init+1)
        self.assertFalse(np.array_equal(s, s_ref))

    def test_mod2(self):
        """Test modulo 2 operation."""

        s = [10, 20, 30]

        # int inputs
        x = tf.random.uniform(s, minval=-2**30, maxval=2**30, dtype=tf.int32)

        y = int_mod_2(x)
        y_ref = tf.math.mod(tf.cast(x, tf.float64), 2.)
        self.assertTrue(np.array_equal(y.numpy(), y_ref.numpy()))

        # float inputs
        x = tf.random.uniform(s, minval=-1000, maxval=1000, dtype=tf.float32)

        y = int_mod_2(x)
        # model implicit cast
        x_f = tf.sign(x) * tf.math.floor(tf.abs(x))
        y_ref = tf.math.mod(tf.math.ceil(x_f), 2.)
        self.assertTrue(np.array_equal(y.numpy(), y_ref.numpy()))
