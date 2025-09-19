#
# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0#
import tensorflow as tf
from tensorflow.python.ops.gen_batch_ops import batch
import pytest
import numpy as np
import scipy as sp
import random

from sionna.phy import config
from sionna.phy.fec.ldpc.decoding import LDPCBPDecoder, LDPC5GDecoder
from sionna.phy.fec.ldpc.encoding import LDPC5GEncoder
from sionna.phy.fec.ldpc.utils import DecoderStatisticsCallback, EXITCallback, WeightedBPCallback
from sionna.phy.fec.utils import GaussianPriorSource, load_parity_check_examples

def utils_gen_ragged_tensor(node_degrees, batch_size):
    """Generate a ragged tensor for testing of the node update functions."""

    row_ids = []
    for c,i in enumerate(node_degrees):
        for j in range(i):
            row_ids.append(c)

    row_ids = tf.constant(row_ids, tf.int32)
    # last dim used as batch dim
    v = config.tf_rng.normal((row_ids.shape[0], batch_size))
    msg = tf.RaggedTensor.from_value_rowids(values=v, value_rowids=row_ids)
    msg = tf.cast(msg, tf.float32)
    return msg

#############################
# Testcases for EXITCallback
#############################

def test_exit_callback(num_iter=20, node_degrees=[5,3,4]*100, batch_size=10000):
    """
    """

    #####################################
    # test that MI is correctly estimated
    #####################################
    exit_cb = EXITCallback(num_iter)

    source = GaussianPriorSource()
    mi_ref = np.linspace(0.02, 0.98, num_iter)

    row_ids = []
    for c,i in enumerate(node_degrees):
        for j in range(i):
            row_ids.append(c)
    row_ids = tf.constant(row_ids, tf.int32)

    num_trials = 10
    for _ in range(num_trials):
        for it in range(num_iter):
            # last dim used as batch dim
            v = source((row_ids.shape[0], batch_size), mi=mi_ref[it])
            msg = tf.RaggedTensor.from_value_rowids(values=v, value_rowids=row_ids)
            exit_cb(-msg, it, 0)

    assert np.allclose(exit_cb.mi[:num_iter], mi_ref, atol=0.01)

    ################################
    # test decoder integration
    ################################
    # parameters

    pcm_id = 2 # decide which parity check matrix should be used (0-2: BCH; 3: (3,6)-LDPC 4: LDPC 802.11n
    pcm, k, n , coderate = load_parity_check_examples(pcm_id, verbose=True)

    # init callbacks for tracking of EXIT charts
    cb_exit_vn = EXITCallback(num_iter)
    cb_exit_cn = EXITCallback(num_iter)

    # init components
    decoder = LDPCBPDecoder(pcm,
                            hard_out=False,
                            cn_update="boxplus",
                            num_iter=num_iter,
                            v2c_callbacks=[cb_exit_vn,], # register callbacks
                            c2v_callbacks=[cb_exit_cn,],) # register callbacks

    llr_ch = tf.ones((batch_size, n))

    # and run decoder
    decoder(llr_ch)

#################################
# Testcases for DecoderStatistics
#################################

def test_stats_callback(node_degrees = [3,4,5,6,7], batch_size=100):
    """
    """
    msg = utils_gen_ragged_tensor(node_degrees, batch_size)

    num_iter = 20
    dec_stats = DecoderStatisticsCallback(num_iter)

    ###############################
    # test all codewords are correct
    ###############################
    num_samples_ref = np.zeros((num_iter,))
    num_decoded_cws_ref = np.zeros((num_iter,))
    for it in range(num_iter):
        dec_stats(tf.abs(msg), it) # all checks are ok
        num_samples_ref[it] += batch_size
        num_decoded_cws_ref[it] += batch_size

        assert np.allclose(dec_stats.num_samples, num_samples_ref)
        assert np.allclose(dec_stats.num_decoded_cws, num_decoded_cws_ref)
    assert dec_stats.avg_number_iterations==0
    assert np.allclose(dec_stats.success_rate,
                    dec_stats.num_decoded_cws/dec_stats.num_samples)

    ##############################
    # test no codewords is correct
    ##############################
    dec_stats.reset_stats() # reset decoder

    num_samples_ref = np.zeros((num_iter,))
    num_decoded_cws_ref = np.zeros((num_iter,))
    for it in range(num_iter):
        dec_stats(-tf.abs(msg), it) # no checks is ok
        num_samples_ref[it] += batch_size
        num_decoded_cws_ref[it] += 0

        assert np.allclose(dec_stats.num_samples, num_samples_ref)
        assert np.allclose(dec_stats.num_decoded_cws, num_decoded_cws_ref)
    assert dec_stats.avg_number_iterations==num_iter
    assert np.allclose(dec_stats.success_rate,
                    dec_stats.num_decoded_cws/dec_stats.num_samples)

    ##################################
    # test 20% of codewords are correct
    ###################################

    dec_stats.reset_stats() # reset decoder

    split = 0.2
    msg1 = utils_gen_ragged_tensor(node_degrees, int(split*batch_size))
    msg2 = utils_gen_ragged_tensor(node_degrees, int((1-split)*batch_size))

    num_samples_ref = np.zeros((num_iter,))
    num_decoded_cws_ref = np.zeros((num_iter,))
    for it in range(num_iter):
        dec_stats(tf.abs(msg1), it) # all checks ok
        num_samples_ref[it] += msg1.shape[-1]
        num_decoded_cws_ref[it] += msg1.shape[-1]
    for it in range(num_iter):
        dec_stats(-tf.abs(msg2), it) # no checks is ok
        num_samples_ref[it] += msg2.shape[-1]
        num_decoded_cws_ref[it] += 0

        assert np.allclose(dec_stats.num_samples, num_samples_ref)
        assert np.allclose(dec_stats.num_decoded_cws, num_decoded_cws_ref)
    assert dec_stats.avg_number_iterations==(1-split)*num_iter
    assert np.allclose(dec_stats.success_rate,
                    dec_stats.num_decoded_cws/dec_stats.num_samples)

    #############################################
    # test that it can be deplyoed in the decoder
    #############################################
    num_runs = 10
    dec_stats.reset_stats() # reset decoder
    encoder = LDPC5GEncoder(100,200)
    decoder = LDPC5GDecoder(encoder=encoder,
                            num_iter=num_iter, # number of BP iterations
                            c2v_callbacks= [dec_stats,])

    llr_ch = tf.ones([batch_size, encoder.n])
    for i in range(num_runs):
        decoder(llr_ch)
    # test that number of samples is correctly tracked
    num_samples_ref = np.ones((num_iter,))*num_runs*batch_size
    assert np.allclose(dec_stats.num_samples, num_samples_ref)

####################################
# Testcases for weighted BP Callback
####################################
@pytest.mark.parametrize("mode", ["eager", "graph"])
def test_wbp_callback(mode, pcm_id=1, num_iter=5, batch_size=100):
    """ Test weighted BP Callbacks
    """

    #####################
    # Decoder integration
    #####################

    pcm, k, n , coderate = load_parity_check_examples(pcm_id, verbose=True)

    # VN weights
    wbp_cb_vn = WeightedBPCallback(np.sum(pcm))
    # CN weights
    wbp_cb_cn = WeightedBPCallback(np.sum(pcm))

    # init components
    decoder = LDPCBPDecoder(pcm,
                            hard_out=False,
                            cn_update="boxplus",
                            num_iter=num_iter,
                            v2c_callbacks=[wbp_cb_vn,], # register callbacks
                            c2v_callbacks=[wbp_cb_cn,],) # register callbacks

    llr_ch = tf.ones((batch_size, n))

    # and run decoder
    decoder(llr_ch)

    ################
    # Test gradients
    ################

    @tf.function(jit_compile=True)
    def run_graph_xla(x):
        return decoder(x)

    @tf.function(jit_compile=False)
    def run_graph(x):
        return decoder(x)

    # gradients seems 0 for all-one input, however that's
    # a very artifical scenario
    llr_ch = 2*tf.ones((batch_size, n))
    with tf.GradientTape(persistent=True) as tape:
            if mode=="eager":
                y = decoder(llr_ch)
            elif mode=="graph":
                y = run_graph(llr_ch)
            else:
                y = run_graph_xla(llr_ch)

    # CN callback weights
    grads_cn = tape.gradient(y, wbp_cb_cn._edge_weights)
    assert grads_cn is not None
    assert not np.allclose(grads_cn.numpy(), np.zeros_like(grads_cn.numpy()))

    # VN callbackweights
    grads_vn = tape.gradient(y, wbp_cb_vn._edge_weights)
    assert grads_vn is not None
    assert not np.allclose(grads_vn.numpy(), np.zeros_like(grads_vn.numpy()))


