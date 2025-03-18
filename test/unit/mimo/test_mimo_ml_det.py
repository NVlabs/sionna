#
# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0#
import pytest
import numpy as np
import tensorflow as tf
from scipy.special import logsumexp
from scipy.stats import unitary_group

from sionna.phy import config
from sionna.phy.mimo import MaximumLikelihoodDetector
from sionna.phy.mapping import Constellation

@pytest.mark.parametrize("num_bits_per_symbol", [2,4])
@pytest.mark.parametrize("num_streams", [1,2,3,4])
@pytest.mark.parametrize("with_prior", [False, True])
def test_vecs(num_bits_per_symbol, num_streams, with_prior):
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

    ref_vecs = build_vecs(num_bits_per_symbol, num_streams)
    ml = MaximumLikelihoodDetector("symbol", "app", num_streams, "qam", num_bits_per_symbol, with_prior=with_prior)
    test_vecs = ml._vecs
    max_dist = np.abs(test_vecs-ref_vecs)
    assert np.allclose(max_dist, 0.0, atol=1e-5)

@pytest.mark.parametrize("num_bits_per_symbol", [2,4])
@pytest.mark.parametrize("num_streams", [1,2,3,4])
@pytest.mark.parametrize("num_rx_ant", [4, 16, 32])
@pytest.mark.parametrize("with_prior", [False, True])
def test_output_dimensions(num_bits_per_symbol, num_streams, num_rx_ant, with_prior):
    num_points = 2**num_bits_per_symbol
    ml = MaximumLikelihoodDetector("symbol", "app", num_streams, "qam", num_bits_per_symbol, with_prior=with_prior)
    batch_size = 8
    dim1 = 3
    dim2 = 5
    y = tf.complex( config.tf_rng.normal([batch_size, dim1, dim2, num_rx_ant]),
                    config.tf_rng.normal([batch_size, dim1, dim2, num_rx_ant]))
    h = tf.complex( config.tf_rng.normal([batch_size, dim1, dim2, num_rx_ant, num_streams]),
                    config.tf_rng.normal([batch_size, dim1, dim2, num_rx_ant, num_streams]))
    s = tf.eye(num_rx_ant, dtype=tf.complex64)
    prior = config.tf_rng.normal([batch_size, dim1, dim2, num_streams, num_points])
    if with_prior: 
        logits = ml(y,h,s,prior)
    else:
        logits = ml(y,h,s)
    assert logits.shape == [batch_size, dim1, dim2, num_streams, num_points]

    s = tf.eye(num_rx_ant, dtype=tf.complex64, batch_shape=[batch_size, dim1, dim2])
    if with_prior: 
        logits = ml(y,h,s,prior)
    else:
        logits = ml(y,h,s)
    assert logits.shape == [batch_size, dim1, dim2, num_streams, num_points]

@pytest.mark.parametrize("num_bits_per_symbol", [2,4])
@pytest.mark.parametrize("num_streams", [1,2,3,4])
@pytest.mark.parametrize("num_rx_ant", [2, 16, 32])
@pytest.mark.parametrize("with_prior", [False, True])
def test_logits_calc_eager(num_bits_per_symbol, num_streams, num_rx_ant, with_prior):
    "Test exponents calculation"

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
    num_points = 2**num_bits_per_symbol
    # Prepare for reference computation
    ref_vecs, ref_vecs_ind, ref_c = build_vecs(num_bits_per_symbol, num_streams)
    num_vecs = ref_vecs.shape[0]
    # Generate random channel outputs and channels
    y = config.np_rng.normal(size=[batch_size, num_rx_ant]) + 1j*config.np_rng.normal(size=[batch_size, num_rx_ant])
    h = config.np_rng.normal(size=[batch_size, num_rx_ant, num_streams]) + 1j*config.np_rng.normal(size=[batch_size, num_rx_ant, num_streams])
    if with_prior:
        prior = config.np_rng.normal(size=[batch_size, num_streams, num_points])
    else:
        prior = None
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
    if with_prior:
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

    ml = MaximumLikelihoodDetector("symbol", "app", num_streams, "qam", num_bits_per_symbol, with_prior=with_prior)

    def call_sys_app(y, h, s, prior=None):
        if prior is None:
            test_logits = ml(y, h, s)
        else:
            test_logits = ml(y, h, s, prior)
        return test_logits
    test_logits = call_sys_app( tf.cast(y, tf.complex64),
                                tf.cast(h, tf.complex64),
                                tf.cast(s, tf.complex64),
                                tf.cast(prior, tf.float32) if with_prior else None).numpy()
    assert np.allclose(ref_app, test_logits, atol=1e-5)

    ## Test for "maxlog"

    ml = MaximumLikelihoodDetector("symbol", "maxlog", num_streams, "qam", num_bits_per_symbol, with_prior=with_prior)

    def call_sys_maxlog(y, h, s, prior=None):
        if prior is None:
            test_logits = ml(y, h, s)
        else:
            test_logits = ml(y, h, s, prior)
        return test_logits
    test_logits = call_sys_maxlog( tf.cast(y, tf.complex64),
                                tf.cast(h, tf.complex64),
                                tf.cast(s, tf.complex64),
                                tf.cast(prior, tf.float32) if with_prior else None).numpy()
    assert np.allclose(test_logits, ref_maxlog, atol=1e-5)

@pytest.mark.parametrize("num_bits_per_symbol", [2,4])
@pytest.mark.parametrize("num_streams", [1,2,3,4])
@pytest.mark.parametrize("num_rx_ant", [2, 16, 32])
@pytest.mark.parametrize("with_prior", [False, True])
def test_logits_calc_graph(num_bits_per_symbol, num_streams, num_rx_ant, with_prior):
    "Test exponents calculation"

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
    num_points = 2**num_bits_per_symbol
    # Prepare for reference computation
    ref_vecs, ref_vecs_ind, ref_c = build_vecs(num_bits_per_symbol, num_streams)
    num_vecs = ref_vecs.shape[0]
    # Generate random channel outputs and channels
    y = config.np_rng.normal(size=[batch_size, num_rx_ant]) + 1j*config.np_rng.normal(size=[batch_size, num_rx_ant])
    h = config.np_rng.normal(size=[batch_size, num_rx_ant, num_streams]) + 1j*config.np_rng.normal(size=[batch_size, num_rx_ant, num_streams])
    if with_prior:
        prior = config.np_rng.normal(size=[batch_size, num_streams, num_points])
    else:
        prior = None
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
    if with_prior:
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

    ml = MaximumLikelihoodDetector("symbol", "app", num_streams, "qam", num_bits_per_symbol, with_prior=with_prior)

    @tf.function
    def call_sys_app(y, h, s, prior=None):
        if prior is None:
            test_logits = ml(y, h, s)
        else:
            test_logits = ml(y, h, s, prior)
        return test_logits
    test_logits = call_sys_app( tf.cast(y, tf.complex64),
                                tf.cast(h, tf.complex64),
                                tf.cast(s, tf.complex64),
                                tf.cast(prior, tf.float32) if with_prior else None).numpy()
    assert np.allclose(ref_app, test_logits, atol=1e-5)

    ## Test for "maxlog"

    ml = MaximumLikelihoodDetector("symbol", "maxlog", num_streams, "qam", num_bits_per_symbol, with_prior=with_prior)

    @tf.function
    def call_sys_maxlog(y, h, s, prior=None):
        if prior is None:
            test_logits = ml(y, h, s)
        else:
            test_logits = ml(y, h, s, prior)
        return test_logits
    test_logits = call_sys_maxlog( tf.cast(y, tf.complex64),
                                tf.cast(h, tf.complex64),
                                tf.cast(s, tf.complex64),
                                tf.cast(prior, tf.float32) if with_prior else None).numpy()
    assert np.allclose(test_logits, ref_maxlog, atol=1e-5)

@pytest.mark.usefixtures("only_gpu")
@pytest.mark.parametrize("num_bits_per_symbol", [2,4])
@pytest.mark.parametrize("num_streams", [1,2,3,4])
@pytest.mark.parametrize("num_rx_ant", [2, 16, 32])
@pytest.mark.parametrize("with_prior", [False, True])
def test_logits_calc_jit(num_bits_per_symbol, num_streams, num_rx_ant, with_prior):
    "Test exponents calculation"

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
    if with_prior:
        ref_exp = ref_exp + prior_
    ref_exp = np.take(ref_exp, ref_c, axis=-1)
    # Compute reference logits with "app"
    ref_app = logsumexp(ref_exp, axis=-3)
    # Compute reference logits with "maxlog"
    ref_maxlog = np.max(ref_exp, axis=-3)

    ml = MaximumLikelihoodDetector("symbol", "app", num_streams, "qam", num_bits_per_symbol, with_prior=with_prior, precision="double")

    if with_prior:
        @tf.function(jit_compile=True)
        def call_sys_app(y, h, prior, s):
            test_logits = ml(y, h, s, prior)
            return test_logits
        test_logits = call_sys_app( tf.cast(y, tf.complex128),
                                    tf.cast(h, tf.complex128),
                                    tf.cast(prior, tf.float64),
                                    tf.cast(s, tf.complex128)).numpy()
        assert np.allclose(test_logits, ref_app, atol=1e-5)

        ## Test for "maxlog"
        ml = MaximumLikelihoodDetector("symbol", "maxlog", num_streams, "qam", num_bits_per_symbol, with_prior=with_prior, precision="double")

        @tf.function(jit_compile=True)
        def call_sys_maxlog(y, h, prior, s):
            test_logits = ml(y, h, s, prior)
            return test_logits
        test_logits = call_sys_maxlog(  tf.cast(y, tf.complex128),
                                        tf.cast(h, tf.complex128),
                                        tf.cast(prior, tf.float64),
                                        tf.cast(s, tf.complex128)).numpy()
        assert np.allclose(test_logits, ref_maxlog, atol=1e-5)
    else:
        @tf.function(jit_compile=True)
        def call_sys_app(y, h, s):
            test_logits = ml(y, h, s)
            return test_logits
        test_logits = call_sys_app( tf.cast(y, tf.complex128),
                                    tf.cast(h, tf.complex128),
                                    tf.cast(s, tf.complex128)).numpy()
        assert np.allclose(test_logits, ref_app, atol=1e-5)

        ## Test for "maxlog"
        ml = MaximumLikelihoodDetector("symbol", "maxlog", num_streams, "qam", num_bits_per_symbol, with_prior=with_prior, precision="double")

        @tf.function(jit_compile=True)
        def call_sys_maxlog(y, h, s):
            test_logits = ml(y, h, s)
            return test_logits
        test_logits = call_sys_maxlog(  tf.cast(y, tf.complex128),
                                        tf.cast(h, tf.complex128),
                                        tf.cast(s, tf.complex128)).numpy()
        assert np.allclose(test_logits, ref_maxlog, atol=1e-5)