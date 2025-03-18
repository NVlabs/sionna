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
from sionna.phy.fec.ldpc.decoding import LDPCBPDecoder, LDPC5GDecoder, cn_update_minsum, cn_update_phi, cn_update_tanh, vn_update_sum, cn_update_offset_minsum
from sionna.phy.fec.ldpc.encoding import LDPC5GEncoder
from sionna.phy.fec.utils import GaussianPriorSource, load_parity_check_examples
from sionna.phy.utils import hard_decisions, sim_ber, ebnodb2no
from sionna.phy.mapping import BinarySource
from sionna.phy.fec.linear import LinearEncoder
from sionna.phy.channel import AWGN


CN_UPDATES = ["min", "boxplus", "boxplus-phi", "offset-minsum"]

#############################
# Testcases for LDPCBPDecoder
#############################

@pytest.mark.parametrize("r", [0.5, 0.75])
@pytest.mark.parametrize("n", [64, 100])
@pytest.mark.parametrize("batch_size", [10])
def test_pcm_consistency(r, n, batch_size):
    """Test against correct pcm formats.
    Parity-check matrix is only allowed to contain binary values
    """
    k = int(n*r)

    # Raise error if PCM contains other elements than 0,1
    pcm = config.np_rng.uniform(0, 2, [n-k, n]).astype(int)
    # set a random position to 2 (invalid)
    idx = config.np_rng.uniform(0, n-k, [2,]).astype(int)
    pcm[idx[0], idx[1]] = 2
    with pytest.raises(BaseException):
        dec = LDPCBPDecoder(pcm)

    # raise error if input shape does not match PCM dim
    pcm = config.np_rng.uniform(0,2,[n-k, n]).astype(int)
    dec = LDPCBPDecoder(pcm)
    llr = config.tf_rng.uniform([tf.cast(batch_size, dtype=tf.int32),
                                    tf.cast(n+1, dtype=tf.int32)],
                                    dtype=tf.float32)
    with pytest.raises(BaseException):
        dec(llr)

@pytest.mark.parametrize("pcm_id", [0,1,2])
@pytest.mark.parametrize("num_iter", [0,1,10,100])
def test_message_passing(pcm_id, num_iter):
    """Test that message passing works correctly; this tests uses no CN/VN functions (i.e. identity functions.)"""

    pcm, k,n,r = load_parity_check_examples(pcm_id=pcm_id)
    dec = LDPCBPDecoder(pcm,
                        cn_update="identity",
                        vn_update="identity",
                        hard_out=False,
                        num_iter=num_iter,
                        llr_max=100000,
                        return_state=True)# avoid clipping

    # feed node indices as inputs (to test correct message passing)
    llr_ch = tf.cast(np.arange(n), tf.float32)
    # add batch dim
    llr_ch = tf.expand_dims(llr_ch,axis=0)

    y, msg_v2c = dec(llr_ch) # invert sign due to different LLR definition

    # normalize y by node degree (as VN nodes marginalize)
    vn_degree = np.sum(pcm, axis=0)
    if num_iter>0:
        y_ = y/(vn_degree+1) # +1 as we also marginalize over llr_ch
    else: # for 0 iterations the decoder directly returns llr_ch
        y_ = y
    assert np.array_equal(llr_ch.numpy(), (y_).numpy())

    # also test that msg_v2c has same values per VN node
    msg_v2c = tf.RaggedTensor.from_value_rowids(values=msg_v2c,
                                                value_rowids=dec._vn_idx)
    # loop over ragged tensor
    for i in range(n):
        for j in range(msg_v2c[i,:].shape[0]):
            assert msg_v2c[i,j]==i

@pytest.mark.parametrize("cn_update", CN_UPDATES)
@pytest.mark.parametrize("use_xla", [False, True])
@pytest.mark.parametrize("num_iter", [0,1,10])
def test_graph_mode(cn_update, use_xla, num_iter, pcm_id=0, batch_size=10):
    """Test that decoder supports graph / XLA mode."""

    # init decoder
    pcm, k,n,r = load_parity_check_examples(pcm_id=pcm_id)
    dec = LDPCBPDecoder(pcm,
                        cn_update=cn_update,
                        vn_update="sum",
                        hard_out=False,
                        num_iter=num_iter)
    source = GaussianPriorSource()

    @tf.function(jit_compile=use_xla)
    def run_graph(batch_size):
        llr_ch = source([batch_size, n], 0.1)
        return dec(llr_ch)

    # run with batch_size as python integer
    run_graph(batch_size)
    # run again with increased batch_size
    run_graph(batch_size+1)

    # and test with tf.constant
    batch_size = tf.constant(batch_size, tf.int32)
    run_graph(batch_size)
    # and run again
    run_graph(batch_size+1)

# XLA is currently not supported for layered decoding
@pytest.mark.parametrize("mode", ["eager", "graph"])
def test_scheduling(mode):
    """Test different schedulings."""

    # checks are indpendent
    pcm = np.array([[1, 1, 1, 0, 0, 0], [0, 0, 0, 1, 1, 1]])
    n = pcm.shape[1]

    cns1 = "flooding"
    cns2 = np.stack([[0],[1]], axis=0) # layered
    cns3 = np.stack([[0],[0]], axis=0) # update only CN0

    y_out = []
    for cns in [cns1, cns2, cns3]:

        dec = LDPCBPDecoder(pcm,
                            num_iter=10,
                            hard_out=False,
                            vn_update="sum",
                            cn_update="min",
                            cn_schedule=cns,
                            llr_max=100000,)
        @tf.function(jit_compile=False)
        def run_graph(x):
            return dec(x)

        x = tf.cast(tf.range(n), tf.float32)
        if mode=="graph":
            y = run_graph(x)
        else:
            y = dec(x)
        y_out.append(y)

    # layered and flooding should be the same output (nodes are independent)
    assert np.array_equal(y_out[0].numpy(), y_out[1].numpy())

    # output shuold be different if only first CN is updated
    # check that the 2nd node is not updated
    assert not np.array_equal(y_out[0].numpy(), y_out[2].numpy())

@pytest.mark.parametrize("shape", [[], [2,3], [2,3,4,5]])
@pytest.mark.parametrize("pcm_id", [0,1])
def test_batch_and_multidimension(shape, pcm_id, num_iter=100):
    """Test that batches are properly handled.
    Further, test multi-dimensional shapes."""

    # init decoder
    pcm, k,n,r = load_parity_check_examples(pcm_id=pcm_id)
    dec = LDPCBPDecoder(pcm,
                        num_iter=num_iter,
                        hard_out=False)
    source = GaussianPriorSource()
    shape.append(n)
    llr_ch = source(shape, 0.1)
    y = dec(llr_ch)

    #reshape before decoding
    y_ref_ = dec(tf.reshape(llr_ch, (-1,n)))
    # restore shape after decoding
    y_ref = tf.reshape(y_ref_, shape)

    assert np.allclose(y.numpy(), y_ref.numpy(), rtol=0.001, atol=0.001)

@pytest.mark.parametrize("dt_in", [tf.float32, tf.float64])
@pytest.mark.parametrize("prec", ["single", "double"])
def test_dtypes(prec, dt_in, pcm_id=0, batch_size=10):
    """Test different precisions."""
        # init decoder
    pcm, k,n,r = load_parity_check_examples(pcm_id=pcm_id)
    dec = LDPCBPDecoder(pcm,
                        hard_out=False,
                        precision=prec,
                        return_state=True)

    # test that input yields no error
    llr_ch = tf.zeros([batch_size, n], dt_in)
    y, v2c_msg = dec(llr_ch)

    # test that output has correct format
    if prec=="single":
        assert y.dtype==tf.float32
        assert v2c_msg.dtype==tf.float32
    else:
        assert y.dtype==tf.float64
        assert v2c_msg.dtype==tf.float64

@pytest.mark.parametrize("pcm_id", [0,1])
@pytest.mark.parametrize("num_iter", [1,10,100])
def test_internal_state(pcm_id, num_iter, batch_size=10):
    """test that internal state is correctly returned.
    For this test, we run the decoder 1 x num_iter and num_iter x 1
    and compare both results.
    """

    pcm, k,n,r = load_parity_check_examples(pcm_id=pcm_id)
    source = GaussianPriorSource()
    dec_ref = LDPCBPDecoder(pcm,
                            hard_out=False,
                            num_iter=num_iter)
    dec = LDPCBPDecoder(pcm,
                        hard_out=False,
                        return_state=True,
                        num_iter=1)

    # test that input yields no error
    llr_ch = source([batch_size, n], 0.1)

    # run reference decoder with num_iter iterations
    y_ref = dec_ref(llr_ch)

    # run decoder num_iter times with 1 iteration
    # always feed state of last iteration
    msg_v2c = None
    for i in range(num_iter):
        y, msg_v2c = dec(llr_ch, msg_v2c=msg_v2c)

    assert np.allclose(y.numpy(), y_ref.numpy())

    # also test that number iter can be feed during call
    y, _ = dec(llr_ch, num_iter=num_iter)

    assert np.allclose(y.numpy(), y_ref.numpy())

# XLA currently not supported for training
@pytest.mark.parametrize("mode", ["eager", "graph"])
def test_gradient(mode, pcm_id=1, batch_size=10):
    """Test that gradients are accessible and not None."""
    pcm, k,n,r = load_parity_check_examples(pcm_id)

    dec = LDPCBPDecoder(pcm, num_iter=2, hard_out=False)
    # calculate gradients w.r.t x
    x = tf.Variable(tf.ones((batch_size, n)), tf.float32)

    @tf.function(jit_compile=True)
    def run_graph_xla(x):
        with tf.GradientTape() as tape:
            y = dec(x)
        return tape.gradient(y, x)

    @tf.function(jit_compile=False)
    def run_graph(x):
        with tf.GradientTape() as tape:
            y = dec(x)
        return tape.gradient(y, x)

    if mode=="eager":
        with tf.GradientTape() as tape:
            y = dec(x)
        grads = tape.gradient(y, x)
    elif mode=="graph":
        grads = run_graph(x)
    else:
        grads = run_graph_xla(x)
    assert grads is not None

@pytest.mark.parametrize("cn_update", CN_UPDATES)
@pytest.mark.parametrize("num_iter", [0, 1, 10])
def test_all_erasure(cn_update, num_iter, pcm_id=2, batch_size=10):
    """test that an all-erasure (llr_ch=0) yields exact 0 outputs.
    This tests against biases in the decoder."""
    pcm, k, n, _ = load_parity_check_examples(2)
    dec = LDPCBPDecoder(pcm,
                        cn_update=cn_update,
                        hard_out=False,
                        num_iter=num_iter)
    llr_ch = tf.zeros((batch_size, n), tf.float32)

    y = dec(llr_ch)
    assert np.array_equal(llr_ch.numpy(), y.numpy())

@pytest.mark.parametrize("num_iter", [0, 10])
def test_hard_output(num_iter, pcm_id=2, batch_size=10):
    """Test hard-out flag yields hard-decided output."""
    pcm, k, n, _ = load_parity_check_examples(2)
    source = GaussianPriorSource()
    dec = LDPCBPDecoder(pcm,
                        hard_out=True,
                        num_iter=num_iter)
    llr_ch = source([batch_size, n], 0.1)
    y = dec(llr_ch)
    y_np = y.numpy()
    # only binary values are allowed
    assert np.array_equal(y_np, y_np.astype(bool))

def test_sparse(num_iter=10, batch_size=10, pcm_id=2):
    """Test that parity-check matrix can be also scipy.sparse mat."""

    pcm, k, n, _ = load_parity_check_examples(pcm_id)
    source = GaussianPriorSource()

    # generate sparse parity-check matrices
    pcm_csc = sp.sparse.csc_matrix(pcm)
    pcm_csr = sp.sparse.csr_matrix(pcm)

    # instantiate decoders with different pcm datatypes
    dec = LDPCBPDecoder(pcm, num_iter=num_iter)
    dec_csc = LDPCBPDecoder(pcm_csc, num_iter=num_iter)
    dec_csr = LDPCBPDecoder(pcm_csr, num_iter=num_iter)

    llr = source([batch_size, n], 0.9)

    # and decode the same llrs with each decoder
    res = dec(llr)
    res_csc = dec_csc(llr)
    res_csr = dec_csr(llr)

    # results must be the same
    assert np.allclose(res, res_csc)
    assert np.allclose(res, res_csr)

@pytest.mark.parametrize("cn_update", CN_UPDATES)
@pytest.mark.parametrize("pcm_id", [1,2,3])
def test_e2e_ldpc(pcm_id, cn_update, no=0.3, num_iter=10, batch_size=10):
    """End-to-end test of LDPC coding scheme using a linear encoder."""

    pcm, k, n, _ = load_parity_check_examples(pcm_id)
    source = BinarySource()
    channel = AWGN()

    encoder = LinearEncoder(pcm, is_pcm=True)
    dec = LDPCBPDecoder(pcm, num_iter=num_iter, cn_update=cn_update)

    bits = source([batch_size, k])
    c = encoder(bits)
    x_bpsk = 2*c-1 # logit definition!
    x_bpsk = tf.cast(x_bpsk, tf.complex64)
    y = channel(x_bpsk, no)
    # real-valued domain is fine
    llr_ch = tf.math.real(2/no**2 * y)
    c_hat = dec(llr_ch)

    # test that transmitted codeword could is correctly recovered
    assert np.array_equal(c.numpy(), c_hat.numpy())

    # check that there was at least one transmission error
    # otherwise the test is useless
    c_hat_no_coding = hard_decisions(llr_ch)
    assert not np.array_equal(c.numpy(), c_hat_no_coding.numpy())

@pytest.mark.parametrize("num_iter", [0,1,10])
@pytest.mark.parametrize("llr_max", [0, 5, 100])
def test_llr_max(llr_max, num_iter, pcm_id=1, batch_size=10):
    """Test that llr_max is correctly set and can be None."""

    pcm, k, n, _ = load_parity_check_examples(pcm_id)
    dec = LDPCBPDecoder(pcm, num_iter=num_iter, hard_out=False,
                        llr_max=llr_max, return_state=True)

    # generate large random inputs
    llr_ch = 2*llr_max * config.tf_rng.normal((batch_size, n))
    y, msg = dec(llr_ch)

    #check that no larger value than llr_max exists
    assert np.max(np.abs(y))<=llr_max
    assert np.max(np.abs(msg))<=llr_max

#####################################
# Testcases for node update functions
#####################################

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

@pytest.mark.parametrize("mode", ["eager", "graph", "xla"])
@pytest.mark.parametrize("no", [0, 0.1, 1.])
@pytest.mark.parametrize("llr_clipping", [5, 20, 100, None])
def test_vn_update_sum(no, llr_clipping, mode, batch_size=100, node_degrees=[3,4,5,6,7]):
    """Test VN update against reference implementation."""

    @tf.function(jit_compile=True)
    def run_graph_xla(msg_c2v, llr_ch):
        return vn_update_sum(msg_c2v, llr_ch, llr_clipping)

    @tf.function(jit_compile=False)
    def run_graph(msg_c2v, llr_ch):
        return vn_update_sum(msg_c2v, llr_ch, llr_clipping)

    # test with random input
    msg_c2v = utils_gen_ragged_tensor(node_degrees, batch_size)
    num_nodes = len(node_degrees)
    llr_ch = no * config.tf_rng.normal((num_nodes, batch_size))

    if mode=="eager":
        msg_v2c, x_tot = vn_update_sum(msg_c2v, llr_ch, llr_clipping)
    elif mode=="graph":
        msg_v2c, x_tot = run_graph(msg_c2v, llr_ch)
    else: # xla
        msg_v2c, x_tot = run_graph_xla(msg_c2v, llr_ch)

    # numpy reference implementation
    msg_np = msg_c2v.numpy()
    x_e_ref = [] # use list to mimic ragged array
    x_tot_ref = np.zeros((num_nodes, batch_size))
    # loop over each node
    for node_idx in range(msg_np.shape[0]):
        x_in = msg_np[node_idx]
        # marginalize
        x_tot_ref[node_idx,:] = np.sum(x_in, axis=0, keepdims=True) \
                                + llr_ch.numpy()[node_idx]
        # extrinsic node output
        x_e_ref.append(x_tot_ref[node_idx,:] - x_in)

    # check results per node
    for idx,v in enumerate(x_e_ref):
        # apply clipping if required
        if llr_clipping is not None:
            v = np.minimum(v, llr_clipping)
            v = np.maximum(v, -llr_clipping)
        assert np.allclose(msg_v2c[idx,:], v, rtol=0.001, atol=0.001)

    for idx,v in enumerate(x_tot_ref):
         # apply clipping if required
        if llr_clipping is not None:
            v = np.minimum(v, llr_clipping)
            v = np.maximum(v, -llr_clipping)
        assert np.allclose(x_tot[idx,:], v, rtol=0.001, atol=0.001)

@pytest.mark.parametrize("mode", ["eager", "graph", "xla"])
@pytest.mark.parametrize("llr_clipping", [5, 20, 100, None])
def test_cn_update_minsum(llr_clipping, mode, batch_size=100, node_degrees=[3,4,5,6,7]):
    """Test minsum CN update against reference implementation."""

    @tf.function(jit_compile=True)
    def run_graph_xla(msg_v2c):
        return cn_update_minsum(msg_v2c, llr_clipping)

    @tf.function(jit_compile=False)
    def run_graph(msg_v2c):
        return cn_update_minsum(msg_v2c, llr_clipping)

    msg_v2c = utils_gen_ragged_tensor(node_degrees, batch_size)
    if mode=="eager":
        msg_c2v = cn_update_minsum(msg_v2c, llr_clipping)
    elif mode=="graph":
        msg_c2v = run_graph(msg_v2c)
    else: # xla
        msg_c2v = run_graph_xla(msg_v2c)

    # numpy reference implementation
    msg_np = msg_v2c.numpy()

    # loop over each node
    msg_c2v_ref = [] # use list to mimic ragged array
    # find extrinsic min for each node individually
    for node_idx in range(msg_np.shape[0]):
        # all incoming message of specific CN
        sign_out = np.prod(np.sign(msg_np[node_idx]),axis=0, keepdims=True) \
            * np.sign(msg_np[node_idx])
        x_in = np.abs(msg_np[node_idx])
        # init array of outgoing message for this node
        x_out = np.zeros((x_in.shape))
        for i in range(x_in.shape[0]):
            # loop over batch
            for bs_idx in range(x_in.shape[1]):
                cur_min = np.inf
                for j in range(x_in.shape[0]):
                    if i!=j:
                        cur_min = np.minimum(cur_min, x_in[j,bs_idx])
                x_out[i,bs_idx] = cur_min * sign_out[i,bs_idx]
        msg_c2v_ref.append(x_out)

    # check results per node
    for idx,v in enumerate(msg_c2v_ref):
        # apply clipping if required
        if llr_clipping is not None:
            v = np.minimum(v, llr_clipping)
            v = np.maximum(v, -llr_clipping)
        assert np.allclose(msg_c2v[idx,:].numpy(), v,rtol=0.001, atol=0.001)

    # Test also for the corner case of a double minimum
    # this results in the same messages to all nodes
    v = [2.1, 2.1, 3, 4]
    msg_c2v_ref = np.array([2.1, 2.1, 2.1, 2.1])
    row_ids = [0,0,0,0]
    msg = tf.RaggedTensor.from_value_rowids(values=v, value_rowids=row_ids)
    msg_c2v = cn_update_minsum(msg, llr_clipping=None)
    assert np.allclose(msg_c2v.flat_values.numpy(), msg_c2v_ref)

@pytest.mark.parametrize("mode", ["eager", "graph", "xla"])
@pytest.mark.parametrize("llr_clipping", [5, 20, 100, None])
@pytest.mark.parametrize("offset", [0, 0.5, 1.])
def test_cn_update_offset_minsum(llr_clipping, mode, offset, batch_size=100, node_degrees=[3,4,5,6,7]):
    """Test offset minsum CN update against reference implementation."""
    @tf.function(jit_compile=True)
    def run_graph_xla(msg_v2c):
        return cn_update_offset_minsum(msg_v2c, llr_clipping, offset=offset)

    @tf.function(jit_compile=False)
    def run_graph(msg_v2c):
        return cn_update_offset_minsum(msg_v2c, llr_clipping, offset=offset)

    msg_v2c = utils_gen_ragged_tensor(node_degrees, batch_size)
    if mode=="eager":
        msg_c2v = cn_update_offset_minsum(msg_v2c, llr_clipping, offset=offset)
    elif mode=="graph":
        msg_c2v = run_graph(msg_v2c)
    else: # xla
        msg_c2v = run_graph_xla(msg_v2c)

    # numpy reference implementation
    msg_np = msg_v2c.numpy()

    # loop over each node
    msg_c2v_ref = [] # use list to mimic ragged array
    # find extrinsic min for each node individually
    for node_idx in range(msg_np.shape[0]):
        # all incoming message of specific CN
        sign_out = np.prod(np.sign(msg_np[node_idx]),axis=0, keepdims=True) \
            * np.sign(msg_np[node_idx])
        x_in = np.abs(msg_np[node_idx])
        # init array of outgoing message for this node
        x_out = np.zeros((x_in.shape))
        for i in range(x_in.shape[0]):
            # loop over batch
            for bs_idx in range(x_in.shape[1]):
                cur_min = np.inf
                for j in range(x_in.shape[0]):
                    if i!=j:
                        cur_min = np.minimum(cur_min, x_in[j,bs_idx])

                # offset correction
                x_out[i,bs_idx] = np.maximum(cur_min - offset,0)
                # and apply sign
                x_out[i,bs_idx] *= sign_out[i,bs_idx]
        msg_c2v_ref.append(x_out)

    # check results per node
    for idx,v in enumerate(msg_c2v_ref):
        # apply clipping if required
        if llr_clipping is not None:
            v = np.minimum(v, llr_clipping)
            v = np.maximum(v, -llr_clipping)
        assert np.allclose(msg_c2v[idx,:].numpy(), v,rtol=0.001, atol=0.001)

@pytest.mark.parametrize("mode", ["eager", "graph", "xla"])
@pytest.mark.parametrize("llr_clipping", [5, 20, 100, None])
def test_cn_update_boxplus(llr_clipping, mode, batch_size=100, node_degrees=[3,4,5,6,7]):
    """Test boxplus CN update against reference implementation."""

    @tf.function(jit_compile=True)
    def run_graph_xla(msg_v2c):
        return cn_update_tanh(msg_v2c, llr_clipping)

    @tf.function(jit_compile=False)
    def run_graph(msg_v2c):
        return cn_update_tanh(msg_v2c, llr_clipping)

    msg_v2c = utils_gen_ragged_tensor(node_degrees, batch_size)
    if mode=="eager":
        msg_c2v = cn_update_tanh(msg_v2c, llr_clipping)
    elif mode=="graph":
        msg_c2v = run_graph(msg_v2c)
    else: # xla
        msg_c2v = run_graph_xla(msg_v2c)

    # numpy reference implementation
    msg_v2c_np = msg_v2c.numpy()
    msg_c2v_ref = []
    for cn in msg_v2c_np:
        d = len(cn)
        msg_out = np.zeros((d,batch_size))
        for i in range(d):
            v = 1
            for j in range(d):
                if i!=j: # exclude extrinsic msg
                    v *= np.tanh(cn[j]/2)
            msg_out[i,:] = 2* np.arctanh(v)
        msg_c2v_ref.append(msg_out)

    # check results per node
    for idx,v in enumerate(msg_c2v_ref):
        # apply clipping if required
        if llr_clipping is not None:
            v = np.minimum(v, llr_clipping)
            v = np.maximum(v, -llr_clipping)
        assert np.allclose(msg_c2v[idx,:].numpy(), v, rtol=0.001, atol=0.001)

@pytest.mark.parametrize("mode", ["eager", "graph", "xla"])
@pytest.mark.parametrize("llr_clipping", [5, 20, 100, None])
def test_cn_update_boxplus_phi(llr_clipping, mode, batch_size=100, node_degrees=[3,4,5,6,7]):
    """Test boxplus-phi CN update against reference implementation."""

    @tf.function(jit_compile=True)
    def run_graph_xla(msg_v2c):
        return cn_update_phi(msg_v2c, llr_clipping)

    @tf.function(jit_compile=False)
    def run_graph(msg_v2c):
        return cn_update_phi(msg_v2c, llr_clipping)

    msg_v2c = utils_gen_ragged_tensor(node_degrees, batch_size)
    if mode=="eager":
        msg_c2v = cn_update_phi(msg_v2c, llr_clipping)
    elif mode=="graph":
        msg_c2v = run_graph(msg_v2c)
    else: # xla
        msg_c2v = run_graph_xla(msg_v2c)

    # numpy reference implementation
    msg_v2c_np = msg_v2c.numpy()
    msg_c2v_ref = []
    for cn in msg_v2c_np:
        d = len(cn)
        msg_out = np.zeros((d,batch_size))
        for i in range(d):
            v = 0
            s = 1
            for j in range(d):
                if i!=j: # exclude extrinsic msg
                    s *= np.sign(cn[j])
                    v += -np.log(np.tanh(np.abs(cn[j])/2))
            msg_out[i,:] = s * (-1.*np.log(np.tanh(v/2)))
        msg_c2v_ref.append(msg_out)

    # check results per node
    for idx,v in enumerate(msg_c2v_ref):
        # apply clipping if required
        if llr_clipping is not None:
            v = np.minimum(v, llr_clipping)
            v = np.maximum(v, -llr_clipping)
        assert np.allclose(msg_c2v[idx,:].numpy(), v, rtol=0.001, atol=0.001)

#############################
# Testcases for LDPC5GDecoder
#############################

@pytest.mark.parametrize("cn_update", CN_UPDATES)
@pytest.mark.parametrize("use_xla", [False, True])
@pytest.mark.parametrize("return_info_bits", [False, True])
@pytest.mark.parametrize("num_iter", [0, 1, 10])
def test_graph_mode_5g(cn_update, use_xla, num_iter, return_info_bits, k=34, n=89, batch_size=10):
    """Test that decoder supports graph / XLA mode."""

    # init decoder
    encoder = LDPC5GEncoder(k, n)
    decoder = LDPC5GDecoder(encoder, num_iter=num_iter,
                            cn_update=cn_update,return_info_bits=return_info_bits)
    source = GaussianPriorSource()

    @tf.function(jit_compile=use_xla)
    def run_graph(batch_size):
        llr_ch = source([batch_size, n], 0.1)
        return decoder(llr_ch)

    # run with batch_size as python integer
    run_graph(batch_size)
    # run again with increased batch_size
    run_graph(batch_size+1)

    # and test with tf.constant
    batch_size = tf.constant(batch_size, tf.int32)
    run_graph(batch_size)
    # and run again
    run_graph(batch_size+1)

def test_scheduling_5g():
    """Test layered scheduling for 5G code.
    We test against the rule of thumb that layered decoding requires approx 50%
    of the iterations for similar results.
    Note that correct scheduling was tested already in the BP decoder."""

    ebno_db = np.arange(0, 3, 0.5) # sim SNR range
    k = 200
    n = 400
    source = BinarySource()
    enc = LDPC5GEncoder(k=k, n=n)
    cn_update = "boxplus"

    bler = []
    for cns, num_iter in zip(["layered", "flooding"],[8, 16]):

        dec = LDPC5GDecoder(enc,
                            num_iter=num_iter,
                            cn_update=cn_update,
                            cn_schedule=cns)
        # run decoding graph
        @tf.function(jit_compile=False) # no XLA for layered
        def run_graph(batch_size, ebno_db):
            no = ebnodb2no(ebno_db, 2, k/n)
            b = source((batch_size, k))
            c = enc(b)
            x = (2*c-1) # bpsk
            y = x + tf.math.sqrt(no) * config.tf_rng.normal((batch_size, n))
            llr_ch = 2 * y / no
            return b, dec(llr_ch)

        _, bler_ = sim_ber(run_graph,
                            ebno_dbs=ebno_db,
                            max_mc_iter=10,
                            num_target_block_errors=100,
                            target_bler=1e-3,
                            batch_size=10000,
                            soft_estimates=False,
                            early_stop=True,
                            verbose=False)
        bler.append(bler_)

    # verify that blers are similar; allow rtol as this is only a rule of thumb
    # and blers are in the log-domain, i.e. a factor 2x is still ok
    assert np.allclose(bler[0].numpy(), bler[1].numpy(), rtol=0.7)

@pytest.mark.parametrize("parameters", [[12, 25],[20, 65], [45, 63], [12, 59], [500,1000]]) # k,n
def test_scheduling_pruning_5g(parameters):
    """Test layered scheduling for 5G code.
    Test that pruning of the pcm does not mess up the CN update schedule.
    This needs to be tested for various code configurations..
    """
    k, n = parameters
    enc = LDPC5GEncoder(k, n)

    retval=[]
    for p in [False, True]:

        dec = LDPC5GDecoder(enc, cn_schedule="layered", num_iter=5,
                            return_infobits=False, hard_out=False, llr_max=10000,cn_update="minsum", prune_pcm=p)

        x = tf.cast(tf.range(n), tf.float32)
        y = -dec(-x)
        retval.append(y)

    assert np.allclose(retval[0].numpy(), retval[1].numpy())

@pytest.mark.parametrize("k", [100, 400, 800, 2000, 4000, 8000])
@pytest.mark.parametrize("r", [0.34, 0.5, 0.75, 0.9])
def test_pruning_5g(k, r, batch_size=100):
    """Test degree-1 VN pruning"""

    source = GaussianPriorSource()
    n = int(k/r)

    enc = LDPC5GEncoder(k, n)
    dec = LDPC5GDecoder(enc,
                        prune_pcm=True,
                        hard_out=False,
                        num_iter=10)

    dec_ref = LDPC5GDecoder(enc,
                            prune_pcm=False,
                            hard_out=False,
                            num_iter=10)

    llr = source([batch_size, n], 0.5)
    x = dec(llr)
    x_ref = dec_ref(llr)

    # allow small difference as iterative error can accumulate after
    # multiple iterations
    diff = tf.reduce_mean(tf.math.abs(x-x_ref)).numpy()
    assert diff < 5e-2

@pytest.mark.parametrize("parameters", [[12, 20, 1], [200, 250, 2], [345, 544, 4], [231, 808, 8]])
def test_output_interleaver_5g(parameters, batch_size=10):
    """Test output interleaver. Parameters are k,n,m
    """
    k, n, m =  parameters

    source = BinarySource()
    enc_ref = LDPC5GEncoder(k, n) # no mapper
    enc = LDPC5GEncoder(k, n, m)
    dec_ref = LDPC5GDecoder(enc_ref, cn_update="minsum")
    dec = LDPC5GDecoder(enc, cn_update="minsum")
    dec_cw = LDPC5GDecoder(enc, cn_update="minsum", return_infobits=False)

    u = source([batch_size, k])
    c = enc(u)
    c_ref = enc_ref(u)
    # emulate tx (no noise/scaling due to minsum required)
    y = 2*c-1
    y_ref = 2*c_ref-1

    u_hat = dec(y)
    c_hat = dec_cw(y)
    u_hat_ref = dec_ref(y_ref)

    assert np.array_equal(u_hat.numpy(), u_hat_ref.numpy())

    # also verify that codeword is correctly returned
    assert np.array_equal(c_hat.numpy(), c.numpy())

    # and verify that c and c_ref are different for m>1
    if m>1:
        assert not np.array_equal(c.numpy(), c_ref.numpy())

@pytest.mark.parametrize("cn_update", CN_UPDATES)
@pytest.mark.parametrize("n", [100, 1234, 9000])
@pytest.mark.parametrize("r", [0.3, 0.5, 0.9])
def test_e2e_ldpc_5g(r, n, cn_update, no=0.3, num_iter=20, batch_size=10):
    """Test end-to-end LDPC coding scheme with 5G NR Encoder."""

    # number of info bits
    k = int(r*n)

    source = BinarySource()
    channel = AWGN()
    encoder = LDPC5GEncoder(k, n)
    decoder = LDPC5GDecoder(encoder, num_iter=num_iter, cn_update=cn_update)

    bits = source([batch_size, k])
    c = encoder(bits)
    x_bpsk = 2*c-1 # logit definition!
    x_bpsk = tf.cast(x_bpsk, tf.complex64)
    y = channel(x_bpsk, no)
    # real-valued domain is fine
    llr_ch = tf.math.real(2/no**2 * y)
    b_hat = decoder(llr_ch)

    # test that transmitted codeword could is correctly recovered
    assert np.array_equal(bits.numpy(), b_hat.numpy())

    # check that there was at least one transmission error
    # otherwise the test is useless
    c_hat_no_coding = hard_decisions(llr_ch)
    assert not np.array_equal(c.numpy(), c_hat_no_coding.numpy())

@pytest.mark.parametrize("cn_update", CN_UPDATES)
@pytest.mark.parametrize("dt_in", [tf.float32, tf.float64])
@pytest.mark.parametrize("prec", ["single", "double"])
@pytest.mark.parametrize("return_infobits", [False, True])
def test_dtypes_5g(dt_in, prec, cn_update, return_infobits, k=50, n=100, batch_size=10):
    """Test different precisions."""
        # init decoder

    encoder = LDPC5GEncoder(k, n)
    decoder = LDPC5GDecoder(encoder,
                            cn_update=cn_update,
                            precision=prec,
                            return_infobits=return_infobits,
                            return_state=True)

    # test that input yields no error
    llr_ch = tf.zeros([batch_size, n], dt_in)
    y, v2c_msg = decoder(llr_ch)

    # test that output has correct format
    if prec=="single":
        assert y.dtype==tf.float32
        assert v2c_msg.dtype==tf.float32
    else:
        assert y.dtype==tf.float64
        assert v2c_msg.dtype==tf.float64

@pytest.mark.parametrize("num_iter", [1,10,100])
def test_internal_state_5g(num_iter, k=50, n=100, batch_size=10):
    """test that internal state is correctly returned.
    For this test, we run the decoder 1 x num_iter and num_iter x 1
    and compare both results.
    """
    source = GaussianPriorSource()
    encoder = LDPC5GEncoder(k, n)
    decoder_ref = LDPC5GDecoder(encoder,
                                return_infobits=False,
                                hard_out=False,
                                return_state=False,
                                num_iter=num_iter)
    decoder = LDPC5GDecoder(encoder,
                            return_infobits=False,
                            hard_out=False,
                            return_state=True,
                            num_iter=1)

    # test that input yields no error
    llr_ch = source([batch_size, n], 0.1)

    # run reference decoder with num_iter iterations
    y_ref = decoder_ref(llr_ch)

    # run decoder num_iter times with 1 iteration
    # always feed state of last iteration
    msg_v2c = None
    for i in range(num_iter):
        y, msg_v2c = decoder(llr_ch, msg_v2c=msg_v2c)

    assert np.array_equal(y.numpy(), y_ref. numpy())

    # also test that number iter can be feed during call
    y, _ = decoder(llr_ch, num_iter=num_iter)

    assert np.array_equal(y.numpy(), y_ref. numpy())

# XLA currently not supported for training
@pytest.mark.parametrize("mode", ["eager", "graph"])
def test_gradient_5g(mode, k=20, n=50, batch_size=10):
    """Test that gradients are accessible and not None."""

    enc = LDPC5GEncoder(k,n)
    dec = LDPC5GDecoder(enc, num_iter=2, hard_out=False)
    x = tf.Variable(tf.ones((batch_size, n)), tf.float32)

    @tf.function(jit_compile=True)
    def run_graph_xla(x):
        with tf.GradientTape() as tape:
            y = dec(x)
        return tape.gradient(y, x)

    @tf.function(jit_compile=False)
    def run_graph(x):
        with tf.GradientTape() as tape:
            y = dec(x)
        return tape.gradient(y, x)

    if mode=="eager":
        with tf.GradientTape() as tape:
            y = dec(x)
        grads = tape.gradient(y, x)
    elif mode=="graph":
        grads = run_graph(x)
    else:
        grads = run_graph_xla(x)
    assert grads is not None

@pytest.mark.parametrize("cn_update", CN_UPDATES)
@pytest.mark.parametrize("num_iter", [0, 1, 10])
def test_all_erasure_5g(cn_update, num_iter, k=75, n=150, batch_size=10):
    """test that an all-erasure (llr_ch=0) yields exact 0 outputs.
    This tests against biases in the decoder."""

    encoder = LDPC5GEncoder(k, n)
    decoder = LDPC5GDecoder(encoder,
                            cn_update=cn_update,
                            return_infobits=False,
                            hard_out=False,
                            num_iter=num_iter)
    llr_ch = tf.zeros((batch_size, n), tf.float32)

    y = decoder(llr_ch)
    assert np.array_equal(llr_ch.numpy(), y.numpy())

@pytest.mark.parametrize("num_iter", [0, 10])
def test_hard_output_5g(num_iter, k=75, n=150, batch_size=10):
    """Test hard-out flag yields hard-decided output."""

    source = GaussianPriorSource()
    encoder = LDPC5GEncoder(k, n)
    decoder = LDPC5GDecoder(encoder,
                            hard_out=True,
                            num_iter=num_iter,
                            return_infobits=False)

    llr_ch = source([batch_size, n], 0.1)
    y = decoder(llr_ch)
    y_np = y.numpy()
    # only binary values are allowed
    assert np.array_equal(y_np, y_np.astype(bool))

@pytest.mark.parametrize("num_iter", [0,1,10])
@pytest.mark.parametrize("llr_max", [0, 5, 100])
def test_llr_max_5g(llr_max, num_iter, k=12, n=20, batch_size=10):
    """Test that llr_max is correctly set and can be None."""

    encoder = LDPC5GEncoder(k, n)
    decoder = LDPC5GDecoder(encoder,
                            hard_out=False,
                            num_iter=num_iter,
                            return_infobits=False,
                            return_state=True,
                            llr_max=llr_max)

    # generate large random inputs
    llr_ch = 2*llr_max * config.tf_rng.normal((batch_size, n))
    y, msg = decoder(llr_ch)

    #check that no larger value than llr_max exists
    assert np.max(np.abs(y))<=llr_max
    assert np.max(np.abs(msg))<=llr_max

@pytest.mark.parametrize("shape", [[], [2,3], [2,3,4,5]])
def test_batch_and_multidimension_5g(shape, k=200, n=300, num_iter=10):
    """Test that batches are properly handled.
    Further, test multi-dimensional shapes."""

    # init decoder
    encoder = LDPC5GEncoder(k, n)
    decoder = LDPC5GDecoder(encoder,
                            hard_out=False,
                            num_iter=num_iter,
                            return_infobits=False)

    source = GaussianPriorSource()
    shape.append(n)
    llr_ch = source(shape, 0.1)
    y = decoder(llr_ch)

    #reshape before decoding
    y_ref_ = decoder(tf.reshape(llr_ch, (-1,n)))
    # restore shape after decoding
    y_ref = tf.reshape(y_ref_, shape)

    assert np.allclose(y.numpy(), y_ref.numpy(), rtol=0.001, atol=0.001)

@pytest.mark.parametrize("parameters", [[64,128], [64, 180], [167, 201], [439, 800], [948, 1024],[3893, 7940], [6530, 10023], [8448, 23000]])
def test_rate_matching_5g(parameters, batch_size=100):
    """Test that if return_infobit==False, the full codeword is returned.

    We test this for zero iterations, to see if all internal reshapes are correctly recovered before returning the estimate.
    """

    k = parameters[0]
    n = parameters[1]
    enc = LDPC5GEncoder(k, n)
    dec = LDPC5GDecoder(enc,
                        hard_out=False,
                        return_infobits=False,
                        num_iter=0)
    llr = config.tf_rng.normal([batch_size, n], mean=4.2, stddev=1)
    # check if return after 0 iterations equals input
    c_hat = dec(llr)
    assert np.array_equal(c_hat.numpy(), llr.numpy())
