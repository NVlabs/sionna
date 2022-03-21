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
from sionna.fec.utils import bin2int_tf, j_fun, j_fun_inv, j_fun_tf, j_fun_inv_tf,GaussianPriorSource, llr2mi, bin2int, int2bin, int2bin_tf, alist2mat, load_alist
from sionna.utils import log2, log10


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

