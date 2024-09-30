#
# SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
import pytest
import unittest
import numpy as np
import tensorflow as tf
import sionna
from sionna import config
from sionna.mimo import MaximumLikelihoodDetector
from sionna.mimo import MaximumLikelihoodDetectorWithPrior
from sionna.mapping import Constellation
from scipy.special import logsumexp
from scipy.stats import unitary_group

class TestSymbolMaximumLikelihoodDetector(unittest.TestCase):

    def test_vecs(self):
        """
        Test the list of all possible vectors of symbols build by the baseclass
        at init
        """
        def build_vecs(num_bits_per_symbol, num_streams):
            C = Constellation("qam", num_bits_per_symbol)
            num_points = 2**num_bits_per_symbol
            L = np.zeros([num_points**num_streams, num_streams], complex)
            for k in range(num_streams):
                tile_point = num_points**(num_streams-k-1)
                tile_const = num_points**k
                for j in range(tile_const):
                    for i,p in enumerate(C.points):
                        min_index = j*num_points*tile_point + ( i*tile_point )
                        max_index = j*num_points*tile_point + ( (i+1)*tile_point )
                        L[min_index:max_index, k] = p
            return L

        for num_bits_per_symbol in (2,4):
            for num_streams in (1,2,3,4):
                ref_vecs = build_vecs(num_bits_per_symbol, num_streams)
                ml = MaximumLikelihoodDetector("symbol", "app", num_streams, "qam", num_bits_per_symbol)
                test_vecs = ml._vecs
                max_dist = np.abs(test_vecs-ref_vecs)
                self.assertTrue(np.allclose(max_dist, 0.0, atol=1e-5))

    def test_output_dimensions(self):
        for num_bits_per_symbol in (2,4):
            num_points = 2**num_bits_per_symbol
            for num_streams in (1,2,3,4):
                for num_rx_ant in (4, 16, 32):
                    ml = MaximumLikelihoodDetector("symbol", "app", num_streams, "qam", num_bits_per_symbol)
                    batch_size = 8
                    dim1 = 3
                    dim2 = 5
                    y = tf.complex( config.tf_rng.normal([batch_size, dim1, dim2, num_rx_ant]),
                                    config.tf_rng.normal([batch_size, dim1, dim2, num_rx_ant]))
                    h = tf.complex( config.tf_rng.normal([batch_size, dim1, dim2, num_rx_ant, num_streams]),
                                    config.tf_rng.normal([batch_size, dim1, dim2, num_rx_ant, num_streams]))

                    s = tf.eye(num_rx_ant, dtype=tf.complex64)
                    logits = ml((y,h,s))
                    self.assertEqual(logits.shape, [batch_size, dim1, dim2, num_streams, num_points])

                    s = tf.eye(num_rx_ant, dtype=tf.complex64, batch_shape=[batch_size, dim1, dim2])
                    logits = ml((y,h,s))
                    self.assertEqual(logits.shape, [batch_size, dim1, dim2, num_streams, num_points])

    def test_logits_calc_eager(self):
        "Test exponents calculation"

        sionna.config.xla_compat = False

        def build_vecs(num_bits_per_symbol, num_streams):
            C = Constellation("qam", num_bits_per_symbol)
            points = C.points.numpy()
            num_points = 2**num_bits_per_symbol
            L = np.zeros([num_points**num_streams, num_streams], complex)
            for k in range(num_streams):
                tile_point = num_points**(num_streams-k-1)
                tile_const = num_points**k
                for j in range(tile_const):
                    for i,p in enumerate(points):
                        min_index = j*num_points*tile_point + ( i*tile_point )
                        max_index = j*num_points*tile_point + ( (i+1)*tile_point )
                        L[min_index:max_index, k] = p

            c = []
            for p in points:
                c_ = []
                for j in range(num_streams):
                    c_.append(np.where(np.isclose(L[:,j],p))[0])
                c_ = np.stack(c_, axis=-1)
                c.append(c_)
            c = np.stack(c, axis=-1)
            return L, c

        batch_size = 16
        for num_bits_per_symbol in (2,4):
            for num_streams in (1,2,3,4):
                for num_rx_ant in (2, 16, 32):
                    # Prepare for reference computation
                    ref_vecs, ref_c = build_vecs(num_bits_per_symbol, num_streams)
                    # Generate random channel outputs and channels
                    y = config.np_rng.normal(size=[batch_size, num_rx_ant]) + 1j*config.np_rng.normal(size=[batch_size, num_rx_ant])
                    h = config.np_rng.normal(size=[batch_size, num_rx_ant, num_streams]) + 1j*config.np_rng.normal(size=[batch_size, num_rx_ant, num_streams])
                    # Generate well conditioned covariance matrices
                    e = config.np_rng.uniform(low=0.5, high=2.0, size=[batch_size, num_rx_ant])
                    e = np.expand_dims(np.eye(num_rx_ant), axis=0)*np.expand_dims(e, -2)
                    u = unitary_group.rvs(dim=num_rx_ant)
                    u = np.expand_dims(u, axis=0)
                    s = np.matmul(u, np.matmul(e, np.conjugate(np.transpose(u, [0, 2, 1]))))
                    # Compute reference exponents
                    diff = np.transpose(np.matmul(h, ref_vecs.T), [0, 2, 1])
                    diff = np.expand_dims(y, axis=1) - diff
                    s_inv = np.linalg.inv(s)
                    s_inv = np.expand_dims(s_inv, axis=-3)
                    diff_ = np.expand_dims(diff, axis=-1)
                    diffT = np.conjugate(np.expand_dims(diff, axis=-2))
                    ref_exp = -np.matmul(diffT, np.matmul(s_inv, diff_))
                    ref_exp = np.squeeze(ref_exp, axis=(-1,-2))
                    ref_exp = ref_exp.real
                    ref_exp = np.take(ref_exp, ref_c, axis=-1)
                    # Compute reference logits with "app"
                    ref_app = logsumexp(ref_exp, axis=-3)
                    # Compute reference logits with "maxlog"
                    ref_maxlog = np.max(ref_exp, axis=-3)

                    ## Test for "app"

                    ml = MaximumLikelihoodDetector("symbol", "app", num_streams, "qam", num_bits_per_symbol)

                    def call_sys_app(y, h, s):
                        test_logits = ml([y, h, s])
                        return test_logits
                    test_logits = call_sys_app( tf.cast(y, tf.complex64),
                                                tf.cast(h, tf.complex64),
                                                tf.cast(s, tf.complex64)).numpy()
                    self.assertTrue(np.allclose(ref_app, test_logits, atol=1e-5))

                    ## Test for "maxlog"

                    ml = MaximumLikelihoodDetector("symbol", "maxlog", num_streams, "qam", num_bits_per_symbol)

                    def call_sys_maxlog(y, h, s):
                        test_logits = ml([y, h, s])
                        return test_logits
                    test_logits = call_sys_maxlog(  tf.cast(y, tf.complex64),
                                                    tf.cast(h, tf.complex64),
                                                    tf.cast(s, tf.complex64)).numpy()
                    self.assertTrue(np.allclose(test_logits, ref_maxlog, atol=1e-5))

    def test_logits_calc_graph(self):
        "Test exponents calculation"

        sionna.config.xla_compat = False

        def build_vecs(num_bits_per_symbol, num_streams):
            C = Constellation("qam", num_bits_per_symbol)
            points = C.points.numpy()
            num_points = 2**num_bits_per_symbol
            L = np.zeros([num_points**num_streams, num_streams], complex)
            for k in range(num_streams):
                tile_point = num_points**(num_streams-k-1)
                tile_const = num_points**k
                for j in range(tile_const):
                    for i,p in enumerate(points):
                        min_index = j*num_points*tile_point + ( i*tile_point )
                        max_index = j*num_points*tile_point + ( (i+1)*tile_point )
                        L[min_index:max_index, k] = p

            c = []
            for p in points:
                c_ = []
                for j in range(num_streams):
                    c_.append(np.where(np.isclose(L[:,j],p))[0])
                c_ = np.stack(c_, axis=-1)
                c.append(c_)
            c = np.stack(c, axis=-1)
            return L, c

        batch_size = 16
        for num_bits_per_symbol in (2,4):
            for num_streams in (1,2,3,4):
                for num_rx_ant in (2, 16, 32):
                    # Prepare for reference computation
                    ref_vecs, ref_c = build_vecs(num_bits_per_symbol, num_streams)
                    # Generate random channel outputs and channels
                    y = config.np_rng.normal(size=[batch_size, num_rx_ant]) + 1j*config.np_rng.normal(size=[batch_size, num_rx_ant])
                    h = config.np_rng.normal(size=[batch_size, num_rx_ant, num_streams]) + 1j*config.np_rng.normal(size=[batch_size, num_rx_ant, num_streams])
                    # Generate well conditioned covariance matrices
                    e = config.np_rng.uniform(low=0.5, high=2.0, size=[batch_size, num_rx_ant])
                    e = np.expand_dims(np.eye(num_rx_ant), axis=0)*np.expand_dims(e, -2)
                    u = unitary_group.rvs(dim=num_rx_ant)
                    u = np.expand_dims(u, axis=0)
                    s = np.matmul(u, np.matmul(e, np.conjugate(np.transpose(u, [0, 2, 1]))))
                    # Compute reference exponents
                    diff = np.transpose(np.matmul(h, ref_vecs.T), [0, 2, 1])
                    diff = np.expand_dims(y, axis=1) - diff
                    s_inv = np.linalg.inv(s)
                    s_inv = np.expand_dims(s_inv, axis=-3)
                    diff_ = np.expand_dims(diff, axis=-1)
                    diffT = np.conjugate(np.expand_dims(diff, axis=-2))
                    ref_exp = -np.matmul(diffT, np.matmul(s_inv, diff_))
                    ref_exp = np.squeeze(ref_exp, axis=(-1,-2))
                    ref_exp = ref_exp.real
                    ref_exp = np.take(ref_exp, ref_c, axis=-1)
                    # Compute reference logits with "app"
                    ref_app = logsumexp(ref_exp, axis=-3)
                    # Compute reference logits with "maxlog"
                    ref_maxlog = np.max(ref_exp, axis=-3)

                    ## Test for "app"

                    ml = MaximumLikelihoodDetector("symbol", "app", num_streams, "qam", num_bits_per_symbol)

                    @tf.function
                    def call_sys_app(y, h, s):
                        test_logits = ml([y, h, s])
                        return test_logits
                    test_logits = call_sys_app( tf.cast(y, tf.complex64),
                                                tf.cast(h, tf.complex64),
                                                tf.cast(s, tf.complex64)).numpy()
                    self.assertTrue(np.allclose(ref_app, test_logits, atol=1e-5))

                    ## Test for "maxlog"

                    ml = MaximumLikelihoodDetector("symbol", "maxlog", num_streams, "qam", num_bits_per_symbol)

                    @tf.function
                    def call_sys_maxlog(y, h, s):
                        test_logits = ml([y, h, s])
                        return test_logits
                    test_logits = call_sys_maxlog(  tf.cast(y, tf.complex64),
                                                    tf.cast(h, tf.complex64),
                                                    tf.cast(s, tf.complex64)).numpy()
                    self.assertTrue(np.allclose(test_logits, ref_maxlog, atol=1e-5))

    @pytest.mark.usefixtures("only_gpu")
    def test_logits_calc_jit(self):
        "Test exponents calculation"

        sionna.config.xla_compat = True

        def build_vecs(num_bits_per_symbol, num_streams):
            C = Constellation("qam", num_bits_per_symbol)
            points = C.points.numpy()
            num_points = 2**num_bits_per_symbol
            L = np.zeros([num_points**num_streams, num_streams], complex)
            for k in range(num_streams):
                tile_point = num_points**(num_streams-k-1)
                tile_const = num_points**k
                for j in range(tile_const):
                    for i,p in enumerate(points):
                        min_index = j*num_points*tile_point + ( i*tile_point )
                        max_index = j*num_points*tile_point + ( (i+1)*tile_point )
                        L[min_index:max_index, k] = p

            c = []
            for p in points:
                c_ = []
                for j in range(num_streams):
                    c_.append(np.where(np.isclose(L[:,j],p))[0])
                c_ = np.stack(c_, axis=-1)
                c.append(c_)
            c = np.stack(c, axis=-1)
            return L, c

        batch_size = 16
        for num_bits_per_symbol in (2,4):
            for num_streams in (1,2,3,4):
                for num_rx_ant in (2, 16, 32):
                    # Prepare for reference computation
                    ref_vecs, ref_c = build_vecs(num_bits_per_symbol, num_streams)
                    # Generate random channel outputs and channels
                    y = config.np_rng.normal(size=[batch_size, num_rx_ant]) + 1j*config.np_rng.normal(size=[batch_size, num_rx_ant])
                    h = config.np_rng.normal(size=[batch_size, num_rx_ant, num_streams]) + 1j*config.np_rng.normal(size=[batch_size, num_rx_ant, num_streams])
                    # Generate well conditioned covariance matrices
                    e = config.np_rng.uniform(low=0.5, high=2.0, size=[batch_size, num_rx_ant])
                    e = np.expand_dims(np.eye(num_rx_ant), axis=0)*np.expand_dims(e, -2)
                    u = unitary_group.rvs(dim=num_rx_ant)
                    u = np.expand_dims(u, axis=0)
                    s = np.matmul(u, np.matmul(e, np.conjugate(np.transpose(u, [0, 2, 1]))))
                    # Compute reference exponents
                    diff = np.transpose(np.matmul(h, ref_vecs.T), [0, 2, 1])
                    diff = np.expand_dims(y, axis=1) - diff
                    s_inv = np.linalg.inv(s)
                    s_inv = np.expand_dims(s_inv, axis=-3)
                    diff_ = np.expand_dims(diff, axis=-1)
                    diffT = np.conjugate(np.expand_dims(diff, axis=-2))
                    ref_exp = -np.matmul(diffT, np.matmul(s_inv, diff_))
                    ref_exp = np.squeeze(ref_exp, axis=(-1,-2))
                    ref_exp = ref_exp.real
                    ref_exp = np.take(ref_exp, ref_c, axis=-1)
                    # Compute reference logits with "app"
                    ref_app = logsumexp(ref_exp, axis=-3)
                    # Compute reference logits with "maxlog"
                    ref_maxlog = np.max(ref_exp, axis=-3)

                    ## Test for "app"

                    ml = MaximumLikelihoodDetector("symbol", "app", num_streams, "qam", num_bits_per_symbol, dtype=tf.complex128)

                    @tf.function(jit_compile=True)
                    def call_sys_app(y, h, s):
                        test_logits = ml([y, h, s])
                        return test_logits
                    test_logits = call_sys_app( tf.cast(y, tf.complex128),
                                                tf.cast(h, tf.complex128),
                                                tf.cast(s, tf.complex128)).numpy()
                    self.assertTrue(np.allclose(test_logits, ref_app, atol=1e-5))

                    ## Test for "maxlog"

                    ml = MaximumLikelihoodDetector("symbol", "maxlog", num_streams, "qam", num_bits_per_symbol, dtype=tf.complex128)

                    @tf.function(jit_compile=True)
                    def call_sys_maxlog(y, h, s):
                        test_logits = ml([y, h, s])
                        return test_logits
                    test_logits = call_sys_maxlog(  tf.cast(y, tf.complex128),
                                                    tf.cast(h, tf.complex128),
                                                    tf.cast(s, tf.complex128)).numpy()
                    self.assertTrue(np.allclose(test_logits, ref_maxlog, atol=1e-5))

class TestMaximumLikelihoodDetectorWithPrior(unittest.TestCase):

    def test_vecs_ind(self):
        """
        Test the list of all possible vectors of symbol indices build by the
        baseclass at init
        """
        def build_vecs_ind(num_bits_per_symbol, num_streams):
            num_points = 2**num_bits_per_symbol
            L = np.zeros([num_points**num_streams, num_streams, 2], int)
            for k in range(num_streams):
                tile_point = num_points**(num_streams-k-1)
                tile_const = num_points**k
                for j in range(tile_const):
                    for i in range(num_points):
                        min_index = j*num_points*tile_point + ( i*tile_point )
                        max_index = j*num_points*tile_point + ( (i+1)*tile_point )
                        L[min_index:max_index, k] = [k,i]
            return L

        for num_bits_per_symbol in (2,4):
            for num_streams in (1,2,3,4):
                ref_vecs = build_vecs_ind(num_bits_per_symbol, num_streams)
                ml = MaximumLikelihoodDetectorWithPrior("symbol", "app", num_streams, "qam", num_bits_per_symbol)
                test_vecs = ml._vecs_ind
                max_dist = np.abs(test_vecs-ref_vecs)
                self.assertTrue(np.allclose(max_dist, 0.0, atol=1e-5))

    def test_output_dimensions(self):
        for num_bits_per_symbol in (2,4):
            num_points = 2**num_bits_per_symbol
            for num_streams in (1,2,3,4):
                for num_rx_ant in (4, 16, 32):
                    ml = MaximumLikelihoodDetectorWithPrior("symbol", "app", num_streams, "qam", num_bits_per_symbol)
                    batch_size = 8
                    dim1 = 3
                    dim2 = 5
                    y = tf.complex( config.tf_rng.normal([batch_size, dim1, dim2, num_rx_ant]),
                                    config.tf_rng.normal([batch_size, dim1, dim2, num_rx_ant]))
                    h = tf.complex( config.tf_rng.normal([batch_size, dim1, dim2, num_rx_ant, num_streams]),
                                    config.tf_rng.normal([batch_size, dim1, dim2, num_rx_ant, num_streams]))
                    prior = config.tf_rng.normal([batch_size, dim1, dim2, num_streams, num_points])

                    s = tf.eye(num_rx_ant, dtype=tf.complex64)
                    logits = ml((y,h,prior,s))
                    self.assertEqual(logits.shape, [batch_size, dim1, dim2, num_streams, num_points])

                    s = tf.eye(num_rx_ant, dtype=tf.complex64, batch_shape=[batch_size, dim1, dim2])
                    logits = ml((y,h,prior,s))
                    self.assertEqual(logits.shape, [batch_size, dim1, dim2, num_streams, num_points])

    def test_logits_calc_eager(self):
        "Test exponents calculation"

        sionna.config.xla_compat = False

        def build_vecs(num_bits_per_symbol, num_streams):
            C = Constellation("qam", num_bits_per_symbol)
            points = C.points.numpy()
            num_points = 2**num_bits_per_symbol
            L = np.zeros([num_points**num_streams, num_streams], complex)
            L_ind = np.zeros([num_points**num_streams, num_streams, 2], int)
            for k in range(num_streams):
                tile_point = num_points**(num_streams-k-1)
                tile_const = num_points**k
                for j in range(tile_const):
                    for i,p in enumerate(points):
                        min_index = j*num_points*tile_point + ( i*tile_point )
                        max_index = j*num_points*tile_point + ( (i+1)*tile_point )
                        L[min_index:max_index, k] = p
                        L_ind[min_index:max_index, k] = [k,i]

            c = []
            for p in points:
                c_ = []
                for j in range(num_streams):
                    c_.append(np.where(np.isclose(L[:,j],p))[0])
                c_ = np.stack(c_, axis=-1)
                c.append(c_)
            c = np.stack(c, axis=-1)
            return L, L_ind, c

        batch_size = 16
        for num_bits_per_symbol in (2,4):
            for num_streams in (1,2,3,4):
                for num_rx_ant in (2, 16, 32):
                    num_points = 2**num_bits_per_symbol
                    # Prepare for reference computation
                    ref_vecs, ref_vecs_ind, ref_c = build_vecs(num_bits_per_symbol, num_streams)
                    num_vecs = ref_vecs.shape[0]
                    # Generate random channel outputs and channels
                    y = config.np_rng.normal(size=[batch_size, num_rx_ant]) + 1j*config.np_rng.normal(size=[batch_size, num_rx_ant])
                    h = config.np_rng.normal(size=[batch_size, num_rx_ant, num_streams]) + 1j*config.np_rng.normal(size=[batch_size, num_rx_ant, num_streams])
                    # Generate priors on symbols
                    prior = config.np_rng.normal(size=[batch_size, num_streams, num_points])
                    # Generate well conditioned covariance matrices
                    e = config.np_rng.uniform(low=0.5, high=2.0, size=[batch_size, num_rx_ant])
                    e = np.expand_dims(np.eye(num_rx_ant), axis=0)*np.expand_dims(e, -2)
                    u = unitary_group.rvs(dim=num_rx_ant)
                    u = np.expand_dims(u, axis=0)
                    s = np.matmul(u, np.matmul(e, np.conjugate(np.transpose(u, [0, 2, 1]))))
                    # Compute reference exponents
                    diff = np.transpose(np.matmul(h, ref_vecs.T), [0, 2, 1])
                    diff = np.expand_dims(y, axis=1) - diff
                    s_inv = np.linalg.inv(s)
                    s_inv = np.expand_dims(s_inv, axis=-3)
                    diff_ = np.expand_dims(diff, axis=-1)
                    diffT = np.conjugate(np.expand_dims(diff, axis=-2))
                    ref_exp = -np.matmul(diffT, np.matmul(s_inv, diff_))
                    ref_exp = np.squeeze(ref_exp, axis=(-1,-2))
                    ref_exp = ref_exp.real
                    # prior_ = np.take(prior, ref_vecs_ind, axis=1)
                    prior_ = []
                    for i in range(batch_size):
                        prior_.append([])
                        for j in range(num_vecs):
                            prior_[-1].append([])
                            for k in range(num_streams):
                                prior_[-1][-1].append(prior[i,ref_vecs_ind[j,k][0],ref_vecs_ind[j,k][1]])
                    prior_ = np.array(prior_)
                    prior_ = np.sum(prior_, axis=-1)
                    ref_exp = ref_exp + prior_
                    ref_exp = np.take(ref_exp, ref_c, axis=-1)
                    # Compute reference logits with "app"
                    ref_app = logsumexp(ref_exp, axis=-3)
                    # Compute reference logits with "maxlog"
                    ref_maxlog = np.max(ref_exp, axis=-3)

                    ## Test for "app"

                    ml = MaximumLikelihoodDetectorWithPrior("symbol", "app", num_streams, "qam", num_bits_per_symbol)

                    def call_sys_app(y, h, prior, s):
                        test_logits = ml([y, h, prior, s])
                        return test_logits
                    test_logits = call_sys_app( tf.cast(y, tf.complex64),
                                                tf.cast(h, tf.complex64),
                                                tf.cast(prior, tf.float32),
                                                tf.cast(s, tf.complex64)).numpy()
                    self.assertTrue(np.allclose(ref_app, test_logits, atol=1e-5))

                    ## Test for "maxlog"

                    ml = MaximumLikelihoodDetectorWithPrior("symbol", "maxlog", num_streams, "qam", num_bits_per_symbol)

                    def call_sys_maxlog(y, h, prior, s):
                        test_logits = ml([y, h, prior, s])
                        return test_logits
                    test_logits = call_sys_maxlog(  tf.cast(y, tf.complex64),
                                                    tf.cast(h, tf.complex64),
                                                    tf.cast(prior, tf.float32),
                                                    tf.cast(s, tf.complex64)).numpy()
                    self.assertTrue(np.allclose(test_logits, ref_maxlog, atol=1e-5))

    def test_logits_calc_graph(self):
        "Test exponents calculation"

        sionna.config.xla_compat = False

        def build_vecs(num_bits_per_symbol, num_streams):
            C = Constellation("qam", num_bits_per_symbol)
            points = C.points.numpy()
            num_points = 2**num_bits_per_symbol
            L = np.zeros([num_points**num_streams, num_streams], complex)
            L_ind = np.zeros([num_points**num_streams, num_streams, 2], int)
            for k in range(num_streams):
                tile_point = num_points**(num_streams-k-1)
                tile_const = num_points**k
                for j in range(tile_const):
                    for i,p in enumerate(points):
                        min_index = j*num_points*tile_point + ( i*tile_point )
                        max_index = j*num_points*tile_point + ( (i+1)*tile_point )
                        L[min_index:max_index, k] = p
                        L_ind[min_index:max_index, k] = [k,i]

            c = []
            for p in points:
                c_ = []
                for j in range(num_streams):
                    c_.append(np.where(np.isclose(L[:,j],p))[0])
                c_ = np.stack(c_, axis=-1)
                c.append(c_)
            c = np.stack(c, axis=-1)
            return L, L_ind, c

        batch_size = 16
        for num_bits_per_symbol in (2,4):
            for num_streams in (1,2,3,4):
                for num_rx_ant in (2, 16, 32):
                    num_points = 2**num_bits_per_symbol
                    # Prepare for reference computation
                    ref_vecs, ref_vecs_ind, ref_c = build_vecs(num_bits_per_symbol, num_streams)
                    num_vecs = ref_vecs.shape[0]
                    # Generate random channel outputs and channels
                    y = config.np_rng.normal(size=[batch_size, num_rx_ant]) + 1j*config.np_rng.normal(size=[batch_size, num_rx_ant])
                    h = config.np_rng.normal(size=[batch_size, num_rx_ant, num_streams]) + 1j*config.np_rng.normal(size=[batch_size, num_rx_ant, num_streams])
                    # Generate priors on symbols
                    prior = config.np_rng.normal(size=[batch_size, num_streams, num_points])
                    # Generate well conditioned covariance matrices
                    e = config.np_rng.uniform(low=0.5, high=2.0, size=[batch_size, num_rx_ant])
                    e = np.expand_dims(np.eye(num_rx_ant), axis=0)*np.expand_dims(e, -2)
                    u = unitary_group.rvs(dim=num_rx_ant)
                    u = np.expand_dims(u, axis=0)
                    s = np.matmul(u, np.matmul(e, np.conjugate(np.transpose(u, [0, 2, 1]))))
                    # Compute reference exponents
                    diff = np.transpose(np.matmul(h, ref_vecs.T), [0, 2, 1])
                    diff = np.expand_dims(y, axis=1) - diff
                    s_inv = np.linalg.inv(s)
                    s_inv = np.expand_dims(s_inv, axis=-3)
                    diff_ = np.expand_dims(diff, axis=-1)
                    diffT = np.conjugate(np.expand_dims(diff, axis=-2))
                    ref_exp = -np.matmul(diffT, np.matmul(s_inv, diff_))
                    ref_exp = np.squeeze(ref_exp, axis=(-1,-2))
                    ref_exp = ref_exp.real
                    # prior_ = np.take(prior, ref_vecs_ind, axis=1)
                    prior_ = []
                    for i in range(batch_size):
                        prior_.append([])
                        for j in range(num_vecs):
                            prior_[-1].append([])
                            for k in range(num_streams):
                                prior_[-1][-1].append(prior[i,ref_vecs_ind[j,k][0],ref_vecs_ind[j,k][1]])
                    prior_ = np.array(prior_)
                    prior_ = np.sum(prior_, axis=-1)
                    ref_exp = ref_exp + prior_
                    ref_exp = np.take(ref_exp, ref_c, axis=-1)
                    # Compute reference logits with "app"
                    ref_app = logsumexp(ref_exp, axis=-3)
                    # Compute reference logits with "maxlog"
                    ref_maxlog = np.max(ref_exp, axis=-3)

                    ## Test for "app"

                    ml = MaximumLikelihoodDetectorWithPrior("symbol", "app", num_streams, "qam", num_bits_per_symbol)

                    @tf.function
                    def call_sys_app(y, h, prior, s):
                        test_logits = ml([y, h, prior, s])
                        return test_logits
                    test_logits = call_sys_app( tf.cast(y, tf.complex64),
                                                tf.cast(h, tf.complex64),
                                                tf.cast(prior, tf.float32),
                                                tf.cast(s, tf.complex64)).numpy()
                    self.assertTrue(np.allclose(ref_app, test_logits, atol=1e-5))

                    ## Test for "maxlog"

                    ml = MaximumLikelihoodDetectorWithPrior("symbol", "maxlog", num_streams, "qam", num_bits_per_symbol)

                    @tf.function
                    def call_sys_maxlog(y, h, prior, s):
                        test_logits = ml([y, h, prior, s])
                        return test_logits
                    test_logits = call_sys_maxlog(  tf.cast(y, tf.complex64),
                                                    tf.cast(h, tf.complex64),
                                                    tf.cast(prior, tf.float32),
                                                    tf.cast(s, tf.complex64)).numpy()
                    self.assertTrue(np.allclose(test_logits, ref_maxlog, atol=1e-5))

    @pytest.mark.usefixtures("only_gpu")
    def test_logits_calc_jit(self):
        "Test exponents calculation"

        sionna.config.xla_compat = True

        def build_vecs(num_bits_per_symbol, num_streams):
            C = Constellation("qam", num_bits_per_symbol)
            points = C.points.numpy()
            num_points = 2**num_bits_per_symbol
            L = np.zeros([num_points**num_streams, num_streams], complex)
            L_ind = np.zeros([num_points**num_streams, num_streams, 2], int)
            for k in range(num_streams):
                tile_point = num_points**(num_streams-k-1)
                tile_const = num_points**k
                for j in range(tile_const):
                    for i,p in enumerate(points):
                        min_index = j*num_points*tile_point + ( i*tile_point )
                        max_index = j*num_points*tile_point + ( (i+1)*tile_point )
                        L[min_index:max_index, k] = p
                        L_ind[min_index:max_index, k] = [k,i]

            c = []
            for p in points:
                c_ = []
                for j in range(num_streams):
                    c_.append(np.where(np.isclose(L[:,j],p))[0])
                c_ = np.stack(c_, axis=-1)
                c.append(c_)
            c = np.stack(c, axis=-1)
            return L, L_ind, c

        batch_size = 16
        for num_bits_per_symbol in (2,4):
            for num_streams in (1,2,3,4):
                for num_rx_ant in (2, 16, 32):
                    num_points = 2**num_bits_per_symbol
                    # Prepare for reference computation
                    ref_vecs, ref_vecs_ind, ref_c = build_vecs(num_bits_per_symbol, num_streams)
                    num_vecs = ref_vecs.shape[0]
                    # Generate random channel outputs and channels
                    y = config.np_rng.normal(size=[batch_size, num_rx_ant]) + 1j*config.np_rng.normal(size=[batch_size, num_rx_ant])
                    h = config.np_rng.normal(size=[batch_size, num_rx_ant, num_streams]) + 1j*config.np_rng.normal(size=[batch_size, num_rx_ant, num_streams])
                    # Generate priors on symbols
                    prior = config.np_rng.normal(size=[batch_size, num_streams, num_points])
                    # Generate well conditioned covariance matrices
                    e = config.np_rng.uniform(low=0.5, high=2.0, size=[batch_size, num_rx_ant])
                    e = np.expand_dims(np.eye(num_rx_ant), axis=0)*np.expand_dims(e, -2)
                    u = unitary_group.rvs(dim=num_rx_ant)
                    u = np.expand_dims(u, axis=0)
                    s = np.matmul(u, np.matmul(e, np.conjugate(np.transpose(u, [0, 2, 1]))))
                    # Compute reference exponents
                    diff = np.transpose(np.matmul(h, ref_vecs.T), [0, 2, 1])
                    diff = np.expand_dims(y, axis=1) - diff
                    s_inv = np.linalg.inv(s)
                    s_inv = np.expand_dims(s_inv, axis=-3)
                    diff_ = np.expand_dims(diff, axis=-1)
                    diffT = np.conjugate(np.expand_dims(diff, axis=-2))
                    ref_exp = -np.matmul(diffT, np.matmul(s_inv, diff_))
                    ref_exp = np.squeeze(ref_exp, axis=(-1,-2))
                    ref_exp = ref_exp.real
                    # prior_ = np.take(prior, ref_vecs_ind, axis=1)
                    prior_ = []
                    for i in range(batch_size):
                        prior_.append([])
                        for j in range(num_vecs):
                            prior_[-1].append([])
                            for k in range(num_streams):
                                prior_[-1][-1].append(prior[i,ref_vecs_ind[j,k][0],ref_vecs_ind[j,k][1]])
                    prior_ = np.array(prior_)
                    prior_ = np.sum(prior_, axis=-1)
                    ref_exp = ref_exp + prior_
                    ref_exp = np.take(ref_exp, ref_c, axis=-1)
                    # Compute reference logits with "app"
                    ref_app = logsumexp(ref_exp, axis=-3)
                    # Compute reference logits with "maxlog"
                    ref_maxlog = np.max(ref_exp, axis=-3)

                    ## Test for "app"

                    ml = MaximumLikelihoodDetectorWithPrior("symbol", "app", num_streams, "qam", num_bits_per_symbol, dtype=tf.complex128)

                    @tf.function(jit_compile=True)
                    def call_sys_app(y, h, prior, s):
                        test_logits = ml([y, h, prior, s])
                        return test_logits
                    test_logits = call_sys_app( tf.cast(y, tf.complex128),
                                                tf.cast(h, tf.complex128),
                                                tf.cast(prior, tf.float64),
                                                tf.cast(s, tf.complex128)).numpy()
                    self.assertTrue(np.allclose(test_logits, ref_app, atol=1e-5))

                    ## Test for "maxlog"

                    ml = MaximumLikelihoodDetectorWithPrior("symbol", "maxlog", num_streams, "qam", num_bits_per_symbol, dtype=tf.complex128)

                    @tf.function(jit_compile=True)
                    def call_sys_maxlog(y, h, prior, s):
                        test_logits = ml([y, h, prior, s])
                        return test_logits
                    test_logits = call_sys_maxlog(  tf.cast(y, tf.complex128),
                                                    tf.cast(h, tf.complex128),
                                                    tf.cast(prior, tf.float64),
                                                    tf.cast(s, tf.complex128)).numpy()
                    self.assertTrue(np.allclose(test_logits, ref_maxlog, atol=1e-5))
