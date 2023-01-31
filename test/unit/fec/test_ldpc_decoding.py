#
# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
try:
    import sionna
except ImportError as e:
    import sys
    sys.path.append("../")

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
from tensorflow.python.ops.gen_batch_ops import batch

import unittest
import numpy as np
import scipy as sp

from sionna.fec.ldpc.decoding import LDPCBPDecoder, LDPC5GDecoder
from sionna.fec.ldpc.encoding import LDPC5GEncoder
from sionna.fec.utils import GaussianPriorSource, load_parity_check_examples
from sionna.utils import BinarySource

class TestBPDecoding(unittest.TestCase):
    "Testcases for LDPCBPDecoder class."

    def test_dtypes(self):
        """Test against correct dtypes:
        - input parameters (must be int etc.)
        - parity-check matrix is only allowed to contain binary values
        """

        # Raise error if PCM contains other elements than 0,1
        pcm = np.random.uniform(0,2,[100,150]).astype(int)
        pcm[10,20] = 2
        with self.assertRaises(AssertionError):
            dec = LDPCBPDecoder(pcm)

        # raise error if llrs are not tf.float32
        batch_size = 100
        n = 64
        k = 32
        pcm = np.random.uniform(0,2,[n-k, n]).astype(int)
        dec = LDPCBPDecoder(pcm)
        llr = tf.random.uniform([tf.cast(batch_size, dtype=tf.int32),
                                 tf.cast(n, dtype=tf.int32)],
                                 maxval=100,
                                 dtype=tf.int32)
        with self.assertRaises(TypeError):
            dec(llr)

        # raise error if input shape does not match PCM dim
        batch_size = 100
        n = 64
        k = 32
        pcm = np.random.uniform(0,2,[n-k, n]).astype(int)
        dec = LDPCBPDecoder(pcm)
        llr = tf.random.uniform([tf.cast(batch_size, dtype=tf.int32),
                                 tf.cast(n+1, dtype=tf.int32)],
                                 dtype=tf.float32)
        with self.assertRaises(AssertionError):
            dec(llr)


    def test_CN(self):
        """Test that CN function works correctly (i.e., extrinsic and sign preserving). Must be done for all node types.

        Test CN-degree 2 as well for all types. Must be a forwarding node
        """
        Ntrials = 100 # nb trials
        k = 12
        n = 24
        enc = LDPC5GEncoder(k, n)
        dec = LDPC5GDecoder(enc)

        # test cn_update_tanh
        for _ in range(Ntrials):
            msg = np.random.normal(size=[10]) #generate random inputs
            x = tf.RaggedTensor.from_row_splits(
                                    values=tf.constant(msg, dtype=tf.float32),
                                    row_splits=[0, len(msg)])
            y1 = dec._cn_update_tanh(x)
            y2 = dec._cn_update_phi(x)

            # minsum needs batch dim
            y3 = dec._cn_update_minsum(tf.expand_dims(x, axis=2))
            y3 = tf.squeeze(y3, axis=2)

            # both CN functions should yield same results (minsum does NOT!)
            self.assertTrue(np.allclose(y1.numpy(),y2.numpy(), atol=1e-4))

            # check that sign is correct (treat 0 as positive)
            s = 2*(msg >= 0).astype(int) - 1
            s = s*np.prod(s)
            y1_s = 2*(y1.numpy() >= 0).astype(int) - 1
            y2_s = 2*(y2.numpy() >= 0).astype(int) - 1
            y3_s = 2*(y3.numpy() >= 0).astype(int) - 1

            # ignore cases where all CN messages are small; otherwise the sign
            # becomes random
            if np.sum(np.abs(y1.numpy()))>1e-3:
                self.assertTrue(np.allclose(s, y1_s)), "sign tanh"
                self.assertTrue(np.allclose(s, y2_s)), "sign phi"
                self.assertTrue(np.allclose(s, y3_s)), "sign minsum"


            # test that exact zero input leads to exact zero output
            msg[-1] = 0.
            x = tf.RaggedTensor.from_row_splits(
                                        values=tf.constant(msg,
                                                           dtype=tf.float32),
                                        row_splits=[0, len(msg)])
            y1 = dec._cn_update_tanh(x).numpy()
            y2 = dec._cn_update_phi(x).numpy()

            # minsum needs batch dim
            y3 = dec._cn_update_minsum(tf.expand_dims(x, axis=2))
            y3 = tf.squeeze(y3, axis=2).numpy()
            # the tanh-implementation is numerically not exact for exact 0
            # inputs
            self.assertTrue(np.array_equal(y1[:,:-1], np.zeros_like(y1[:,:-1])))
            self.assertTrue(np.array_equal(y2[:,:-1], np.zeros_like(y2[:,:-1])))
            self.assertTrue(np.array_equal(y3[:,:-1], np.zeros_like(y3[:,:-1])))

    def test_int_state(self):
        """Test internal state functionality of decoder.
        This implies that Nx1 iterations yield exact same result as N
        iterations."""
        batch_size = 1
        Niter = 5
        pcm, k, n, _ = load_parity_check_examples(2)

        dec = LDPCBPDecoder(pcm, num_iter=Niter)
        dec2 = LDPCBPDecoder(pcm, num_iter=1, stateful=True)

        llr = tf.random.normal([batch_size, n], mean=4.2, stddev=1)

        res1 = dec(llr)

        res2, msg_vn = dec2([llr, None]) # iter 0 to init msg_vn

        for i in range(Niter-1): # remaining iterations
            res2,_ = dec2([llr, msg_vn])
        # results must be the same, otherwise the internal state is not
        # correctly recovered
        self.assertTrue(np.allclose(res1,res2))

    def test_phi(self):
        """Test that phi is self-inverse."""
        x = np.arange(0.01, 16.6, 0.01)
        y = LDPCBPDecoder._phi(None, x)
        z = LDPCBPDecoder._phi(None, y)
        self.assertTrue(np.allclose(x, z))

    def test_VN(self):
        """Test that VN function works correctly (i.e., extrinsic).
        """
        Ntrials = 1000 # nb trials
        k = 12
        n = 24
        enc = LDPC5GEncoder(k, n)
        dec = LDPC5GDecoder(enc)

        # test vn updates
        for _ in range(Ntrials):
            msg = np.random.normal(size=[10]) #generate random inputs
            msg_ch = np.random.normal(size=[1]) #generate random inputs

            x = tf.RaggedTensor.from_row_splits(
                                        values=tf.constant(msg, dtype=tf.float32),
                                        row_splits=[0, len(msg)])

            y = dec._vn_update(x, msg_ch).numpy()

            y_ref = np.sum(msg) - msg + msg_ch
            self.assertTrue(np.allclose(y_ref, y, atol=1e-5))

    def test_batch(self):
        """Test that batch of codewords yields the same results for each batch
        sample."""

        batch_size = 100
        Niter = 10
        pcm, k, n, _ = load_parity_check_examples(2)

        dec = LDPCBPDecoder(pcm)
        llr = tf.random.normal([1, n], mean=4.2, stddev=1)
        llr = tf.tile(llr, [batch_size,1])
        x = dec(llr).numpy()
        for i in range(batch_size):
            # if decoder runs on GPU, the reduce_prod/reduce_sum in the GPU
            # yields slightly different result (probably due to scheduling).
            # This leads to slightly different results within one batch
            # which is further amplified with more iterations.
            self.assertTrue(np.allclose(x[0,:],x[i,:],atol=1e-4))


    def test_gradient(self):
        """Test that gradient is accessible and not None."""

        batch_size = 100
        pcm, k, n, _ = load_parity_check_examples(2)

        # check that trainable parameter works as expected
        dec = LDPCBPDecoder(pcm, trainable=True)
        self.assertFalse(len(dec.trainable_variables)==0) # trainable variable
        dec = LDPCBPDecoder(pcm, trainable=False)
        self.assertTrue(len(dec.trainable_variables)==0) # no trainable variable

        cns = ['boxplus', 'boxplus-phi','minsum']
        trainable = [True, False]
        for cn in cns:
            for t in trainable:
                dec = LDPCBPDecoder(pcm,
                                    trainable=t,
                                    cn_type=cn,
                                    hard_out=False)
                llr = tf.random.normal([batch_size, n], mean=4.2, stddev=1)

                with tf.GradientTape() as tape:
                    x = dec(llr)
                    grads = tape.gradient(x, dec.trainable_variables)

                    # check that gradients exist
                    self.assertIsNotNone(grads)

                    # check that gradients are provided
                    if t: # if trainable we should get gradients
                        self.assertTrue(len(grads)>0), "no gradient found"

                        # and check that array is not None
                        for g in grads:
                            self.assertTrue(not g is None), "grad is None"
                    else:
                        self.assertTrue(len(grads)==0), \
                                     "gradient should not exist"

    def test_all_erasure(self):
        """Test that all-erasure (llr=0) cw yields constant all-zero output."""

        batch_size = 100
        pcm, k, n, _ = load_parity_check_examples(2)

        cns = ['boxplus', 'boxplus-phi', 'minsum']
        trainable = [True, False]
        for cn in cns:
            for t in trainable:
                dec = LDPCBPDecoder(pcm, trainable=t, cn_type=cn)
                llr = tf.zeros([batch_size, n])
                x = dec(llr)
                self.assertTrue(np.array_equal(x.numpy(), llr.numpy()))

    def test_hard_out(self):
        """Test hard-out flag yields hard-decided output."""

        batch_size = 100
        pcm, k, n, _ = load_parity_check_examples(2)

        cns = ['boxplus', 'boxplus-phi','minsum']
        trainable = [True, False]
        for cn in cns:
            for t in trainable:
                dec = LDPCBPDecoder(pcm, trainable=t, cn_type=cn, hard_out=True)

                # test that all zero CW yields hard-decided all-zero cw
                llr = -10.*tf.ones([batch_size, n]) # all-zero input
                x = dec(llr).numpy()
                self.assertTrue(np.array_equal(x, np.zeros_like(x)))

                # test that for arbitrary input only 0,1 values are returned
                llr = tf.random.normal([batch_size, n], mean=4.2, stddev=1)
                x = dec(llr).numpy()
                #x contains only {0,1}
                self.assertTrue(np.array_equal(x, x.astype(bool)))

    def test_tf_fun(self):
        """Test tf.function"""

        @tf.function
        def run_graph(llr):
            return dec(llr)

        @tf.function(jit_compile=True)
        def run_graph_xla(llr):
            return dec(llr)

        batch_size = 100
        pcm, k, n, _ = load_parity_check_examples(2)

        cns = ['boxplus', 'boxplus-phi','minsum']
        trainable = [True, False]
        for cn in cns:
            for t in trainable:
                dec = LDPCBPDecoder(pcm, trainable=t, cn_type=cn, hard_out=True)

                # test that all zero CW yields hard-decided all-zero cw
                llr = -10.*tf.ones([batch_size, n]) # all-zero input
                x = dec(llr).numpy()
                self.assertTrue(np.array_equal(x, np.zeros_like(x)))

                # test that for arbitrary inputs
                llr = tf.random.normal([batch_size, n], mean=4.2, stddev=1)
                x = run_graph(llr).numpy()

                # execute the graph twice with same input shape
                x = run_graph(llr).numpy()

                # and change batch_size
                llr = -10.*tf.ones([2*batch_size, n]) # all-zero input
                x = run_graph(llr).numpy()

                # and again with jit_compile=True
                x = run_graph_xla(llr).numpy()

                # execute the graph twice
                x = run_graph_xla(llr).numpy()

                # and change batch_size
                llr = -10.*tf.ones([2*batch_size, n]) # all-zero input
                x = run_graph_xla(llr).numpy()

    def test_output_dim(self):
        """Test that output dim is n."""
        batch_size = 100
        Niter = 10
        pcm, k, n, _ = load_parity_check_examples(2)

        dec = LDPCBPDecoder(pcm)
        llr = tf.random.normal([batch_size, n], mean=1., stddev=1)
        dec = LDPCBPDecoder(pcm, track_exit=False)
        x = dec(llr)
        self.assertTrue(np.shape(x)[1]==n)

    def test_multi_dim(self):
        """Test that 2+D Tensors are correctly handled."""

        pcm, k, n, _ = load_parity_check_examples(2)
        dec = LDPCBPDecoder(pcm)
        shapes =[[10, 2, 3, n], [1, 4, n], [10, 2, 3, 3, n]]

        for s in shapes:
            llr = tf.random.normal(s, mean=0, stddev=1)
            llr_ref = tf.reshape(llr, [-1, n])

            c = dec(llr)
            c_ref = dec(llr_ref)
            s[-1] = n
            c_ref = tf.reshape(c_ref, s)
            self.assertTrue(np.allclose(c.numpy(), c_ref.numpy(), atol=0.001))

        # and verify that wrong last dimension raises an error
        with self.assertRaises(tf.errors.InvalidArgumentError):
            s = [10, 2, n-1]
            llr = tf.random.normal(s, mean=0, stddev=1)
            c = dec(llr)

    def test_all_zero(self):
        """Test all-zero cw without noise yields all-zero info bits."""
        batch_size = 100
        pcm, k, n, _ = load_parity_check_examples(2)

        cns = ['boxplus', 'boxplus-phi','minsum']
        trainable = [True, False]
        for cn in cns:
            for t in trainable:
                dec = LDPCBPDecoder(pcm, trainable=t, cn_type=cn, hard_out=True)
                # init with all-zero and large LLRs/logits (=high SNR)
                llr = -10.* tf.ones([batch_size, n])
                x = np.zeros_like(llr)
                x_hat = dec(llr)
                self.assertTrue(np.array_equal(x, x_hat.numpy()))

    def test_keras(self):
        """Test that Keras model can be compiled (supports dynamic shapes)"""
        bs = 10
        source = BinarySource()
        pcm, k, n, _ = load_parity_check_examples(2)

        inputs = tf.keras.Input(shape=(n), dtype=tf.float32)
        x = LDPCBPDecoder(pcm)(inputs)
        model = tf.keras.Model(inputs=inputs, outputs=x)
        # test that output batch dim is none
        self.assertTrue(model.output_shape[0] is None)
        llr = tf.ones([bs, n])
        model(llr)
        # call twice to see that bs can change
        llr2 = tf.ones([bs, n])
        model(llr2)
        model.summary()

    def test_dtype2(self):
        """Test that output dtype can be flexible"""
        batch_size = 100
        pcm, k, n, _ = load_parity_check_examples(2)
        dec_32 = LDPCBPDecoder(pcm, output_dtype=tf.float32)
        dec_64 = LDPCBPDecoder(pcm, output_dtype=tf.float64)
        llr_32 = tf.random.uniform([tf.cast(batch_size, dtype=tf.int32),
                                    tf.cast(n, dtype=tf.int32)],
                                    dtype=tf.float32)
        llr_64 = tf.random.uniform([tf.cast(batch_size, dtype=tf.int32),
                                    tf.cast(n, dtype=tf.int32)],
                                    dtype=tf.float64)

        # output for both inputs is tf.float32
        u_32 = dec_32(llr_32)
        u_64 = dec_32(llr_64)
        self.assertTrue(u_32.dtype is tf.float32)
        self.assertTrue(u_64.dtype is tf.float32)

        # output for both inputs is tf.float64
        u_32 = dec_64(llr_32)
        u_64 = dec_64(llr_64)
        self.assertTrue(u_32.dtype is tf.float64)
        self.assertTrue(u_64.dtype is tf.float64)

    def test_sparse(self):
        """Test that parity-check matrix can be also scipy.sparse mat."""
        batch_size = 10
        Niter = 10
        pcm, k, n, _ = load_parity_check_examples(3)
        source = GaussianPriorSource()

        # generate sparse parity-check matrices
        pcm_csc = sp.sparse.csc_matrix(pcm)
        pcm_csr = sp.sparse.csr_matrix(pcm)

        # instantiate decoders with different pcm datatypes
        dec = LDPCBPDecoder(pcm, num_iter=Niter)
        dec_csc = LDPCBPDecoder(pcm_csc, num_iter=Niter)
        dec_csr = LDPCBPDecoder(pcm_csr, num_iter=Niter)

        llr = source([[batch_size, n], 0.9])

        # and decode the same llrs with each decoder
        res = dec(llr)
        res_csc = dec_csc(llr)
        res_csr = dec_csr(llr)

        # results must be the same
        self.assertTrue(np.allclose(res, res_csc))
        self.assertTrue(np.allclose(res, res_csr))

    def test_llrmax(self):
        """Test that llr_max can be set."""
        pcm, _, n, _ = load_parity_check_examples(0)
        # no iteration: decoder returns clipped llrs
        dec = LDPCBPDecoder(pcm, num_iter=0, hard_out=False)

        # test default value
        llr_max_def = dec.llr_max.numpy() # get default value
        x = tf.ones((1,n))*100
        y = dec(x).numpy() # run 0 iterations
        np.max(y)==llr_max_def

        # set new llr_max
        llr_maxs = [17., 45.3, 78]
        for l in llr_maxs:
            dec.llr_max = l
            y = dec(x).numpy() # run 0 iterations
            print(np.abs(np.max(y)-l)<1e-6)


class TestBPDecoding5G(unittest.TestCase):
    """Checks LDPC5GDecoding layer.
    Remark: As this layer inherits from BPDecoding many cases are covered by
    previous tests."""

    def test_encoding(self):
        """Test that encoded info bits can be reconstruced after decoding
        (assuming no/little noise)."""

        batch_size = 100

        # k, n
        params =[[64, 128], [64, 180], [167, 201], [439, 800], [3893, 7940],
                 [6530, 10023], [8448, 23000]]

        # generate random bits
        for ret_info in [True, False]:
            src = BinarySource()
            for p in params:
                k = p[0]
                n = p[1]
                enc = LDPC5GEncoder(k, n)
                dec = LDPC5GDecoder(enc, hard_out=True, return_infobits=ret_info)
                b = src([batch_size, k])
                c = enc(b)
                x = 2*c -1 # BPSK (neg. sign due to  sionna llr definition)
                llr = 5 * x # scale as we have no noise -> larger LLRs
                b_hat = dec(llr)
                if ret_info:
                    self.assertTrue(np.array_equal(b.numpy(), b_hat.numpy()))
                else:
                    self.assertTrue(np.array_equal(c.numpy(), b_hat.numpy()))

    def test_dimensions(self):
        """Test for dimension mismatched between input_shape and k, n."""

        batch_size = 100
        n = 128
        k = 64
        enc = LDPC5GEncoder(k, n)
        dec = LDPC5GDecoder(enc)
        llr = tf.random.uniform([tf.cast(batch_size, dtype=tf.int32),
                                 tf.cast(n+1, dtype=tf.int32)],
                                 dtype=tf.float32)

        with self.assertRaises(AssertionError):
            dec(llr)

        # varying batch-sizes should be supported
        llr = tf.random.uniform([tf.cast(batch_size+1, dtype=tf.int32),
                                 tf.cast(n, dtype=tf.int32)],
                                 dtype=tf.float32)
        dec(llr)

    def test_multi_dim(self):
        """Test that 2+D Tensors are correctly handled."""

        k = 100
        n = 200
        shapes =[[10, 20, 30, n], [1, 40, n], [10, 2, 3, 4, 3, n]]
        enc = LDPC5GEncoder(k, n)
        dec = LDPC5GDecoder(enc, num_iter=10)
        source = GaussianPriorSource()

        for s in shapes:
            llr = source([s, 1])
            llr_ref = tf.reshape(llr, [-1, n])

            c = dec(llr)
            c_ref = dec(llr_ref)
            s[-1] = k
            c_ref = tf.reshape(c_ref, s)
            self.assertTrue(np.allclose(c.numpy(), c_ref.numpy(), atol=0.01))

        # and verify that wrong last dimension raises an error
        with self.assertRaises(BaseException):
            s = [10, 2, k-1]
            llr = tf.random.normal(s, mean=0, stddev=1)
            c = dec(llr)

    def test_gradient(self):
        """Test that gradient is accessible and not None."""

        batch_size = 100
        n = 128
        k = 64
        enc = LDPC5GEncoder(k, n)

        cns = ['boxplus', 'boxplus-phi', 'minsum']
        trainable = [True, False]
        for cn in cns:
            for t in trainable:
                dec = LDPC5GDecoder(enc,
                                    trainable=t,
                                    cn_type=cn,
                                    hard_out=False)
                llr = tf.random.normal([batch_size, n], mean=4.2, stddev=1)

                with tf.GradientTape() as tape:
                    x = dec(llr)
                    grads = tape.gradient(x, dec.trainable_variables)

                    # check that gradients exist
                    self.assertIsNotNone(grads)

                    # check that gradients are provided
                    if t: # if trainable we should get gradients
                        self.assertTrue(len(grads)>0), "no gradient found"

                        # and check that array is not None
                        for g in grads:
                            self.assertTrue(not g is None), "grad is None"
                    else:
                        self.assertTrue(len(grads)==0), \
                                                "gradient should not exist"

    def test_dtype(self):
        """Test that output dtype can be flexible."""
        batch_size = 100
        n = 128
        k = 64
        enc = LDPC5GEncoder(k, n)
        dec_32 = LDPC5GDecoder(enc, output_dtype=tf.float32)
        dec_64 = LDPC5GDecoder(enc, output_dtype=tf.float64)
        llr_32 = tf.random.uniform([tf.cast(batch_size, dtype=tf.int32),
                                 tf.cast(n, dtype=tf.int32)],
                                 dtype=tf.float32)
        llr_64 = tf.random.uniform([tf.cast(batch_size, dtype=tf.int32),
                                 tf.cast(n, dtype=tf.int32)],
                                 dtype=tf.float64)

        # output for both inputs is tf.float32
        u_32 = dec_32(llr_32)
        u_64 = dec_32(llr_64)
        self.assertTrue(u_32.dtype is tf.float32)
        self.assertTrue(u_64.dtype is tf.float32)

        # output for both inputs is tf.float64
        u_32 = dec_64(llr_32)
        u_64 = dec_64(llr_64)
        self.assertTrue(u_32.dtype is tf.float64)
        self.assertTrue(u_64.dtype is tf.float64)

    def test_full_cw_ratematching(self):
        """Test that if return_infobit==False, the full codeword is returned.

        We test this for zero iterations, to see if all internal reshapes are correctly recovered before returning the estimate.
        """
        batch_size = 100
        params =[[64,128], [64, 180], [167, 201], [439, 800], [3893, 7940],
                 [6530, 10023], [8448, 23000]]

        for p in params:
            k = p[0]
            n = p[1]
            enc = LDPC5GEncoder(k, n)
            dec = LDPC5GDecoder(enc,
                                hard_out=False,
                                return_infobits=False,
                                num_iter=0)
            llr = tf.random.normal([batch_size, n], mean=4.2, stddev=1)
            # check if return after 0 iterations equals input
            c_hat = dec(llr)
            self.assertTrue(np.array_equal(c_hat.numpy(), llr.numpy()))

    def test_keras(self):
        """Test that Keras model can be compiled (supports dynamic shapes)."""
        bs = 10
        n = 200
        k = 100
        enc = LDPC5GEncoder(k, n)
        for return_info in [True, False]:
            inputs = tf.keras.Input(shape=(n), dtype=tf.float32)
            x = LDPC5GDecoder(enc, return_infobits=return_info)(inputs)
            model = tf.keras.Model(inputs=inputs, outputs=x)
            # test that output batch dim is none
            self.assertTrue(model.output_shape[0] is None)

            llr = -10.* tf.ones([bs, n])
            model(llr)
            # call twice to see that bs can change
            llr2 = -10.* tf.ones([bs, n])
            model(llr2)
            model.summary()

    def test_tf_fun(self):
        """Test graph mode and support for XLA."""

        @tf.function
        def run_graph(llr):
            return dec(llr)

        @tf.function(jit_compile=True)
        def run_graph_xla(llr):
            return dec(llr)

        batch_size = 100
        n = 100
        k = 50
        enc = LDPC5GEncoder(k, n)

        for cn_type in ["minsum", "boxplus", "boxplus-phi"]:
            for return_info_bits in [True, False]:
                for hard_out in [True, False]:
                    dec = LDPC5GDecoder(enc,
                                    hard_out=hard_out,
                                    cn_type=cn_type,
                                    return_infobits=return_info_bits)

                    # test that all zero CW yields hard-decided all-zero cw
                    llr = -10.*tf.ones([batch_size, n]) # all-zero input
                    x = dec(llr).numpy()
                    if hard_out:
                        self.assertTrue(np.array_equal(x, np.zeros_like(x)))

                    # test that for arbitrary input only 0,1 values are returned
                    llr = tf.random.normal([batch_size, n], mean=4.2, stddev=1)
                    x = run_graph(llr).numpy()

                    # execute the graph twice
                    x = run_graph(llr).numpy()

                    # and change batch_size
                    llr = -10.*tf.ones([2*batch_size, n]) # all-zero input
                    x = run_graph(llr).numpy()

                    # and again with jit_compile=True
                    x = run_graph_xla(llr).numpy()

                    # execute the graph twice
                    x = run_graph_xla(llr).numpy()

                    # and change batch_size
                    llr = -10.*tf.ones([2*batch_size, n]) # all-zero input
                    x = run_graph_xla(llr).numpy()

    def test_dtype_flexible(self):
        """Test that output_dtype can be flexible and
        only floats are supported."""
        batch_size = 100
        k = 100
        n = 200
        source = GaussianPriorSource()

        enc = LDPC5GEncoder(k,n)

        dtypes_supported = (tf.float16, tf.float32, tf.float64)

        for dt_in in dtypes_supported:
            for dt_out in dtypes_supported:
                llr = source([[batch_size, n], 0.5])
                llr = tf.cast(llr, dt_in)

                dec = LDPC5GDecoder(enc, output_dtype=dt_out)

                x = dec(llr)

                self.assertTrue(x.dtype==dt_out)

        # test that complex inputs raise error
        llr = source([[batch_size, n], 0.5])
        llr_c = tf.complex(llr, tf.zeros_like(llr))
        dec = LDPC5GDecoder(enc, output_dtype=tf.float32)

        with self.assertRaises(TypeError):
            x = dec(llr_c)

    def test_pruning(self):
        """Test degree-1 VN pruning"""

        batch_size = 100
        ks = [100, 400, 800, 2000, 4000, 8000]
        rs = [ 0.34, 0.5, 0.75, 0.9]
        source = GaussianPriorSource()

        for k in ks:
            for r in rs:

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

                llr = source([[batch_size, n], 0.5])
                x = dec(llr)
                x_ref = dec_ref(llr)

                # allow small difference as iterative error can accumulate after
                # multiple iterations
                diff = tf.reduce_mean(tf.math.abs(x-x_ref)).numpy()
                self.assertTrue(diff < 5e-2)

    def test_pruning(self):
        """Test output interleaver."""

        bs = 10
        source = BinarySource()

        # k, n, m
        params = [[12, 20, 1], [200, 250, 2], [345, 544, 4], [231, 808, 8]]

        for (k,n,m) in params:
            enc_ref = LDPC5GEncoder(k, n) # no mapper
            enc = LDPC5GEncoder(k, n, m)
            dec_ref = LDPC5GDecoder(enc_ref, cn_type="minsum")
            dec = LDPC5GDecoder(enc, cn_type="minsum")
            dec_cw = LDPC5GDecoder(enc, cn_type="minsum", return_infobits=False)

            u = source([bs, k])
            c = enc(u)
            c_ref = enc_ref(u)
            # emulate tx (no noise/scaling due to minsum required)
            y = 2*c-1
            y_ref = 2*c_ref-1

            u_hat = dec(y)
            c_hat = dec_cw(y)
            u_hat_ref = dec_ref(y_ref)

            self.assertTrue(np.array_equal(u_hat.numpy(),
                                           u_hat_ref.numpy()))

            # also verify that codeword is correctly returned
            self.assertTrue(np.array_equal(c_hat.numpy(),
                                           c.numpy()))

            # and verify that c and c_ref are different for m>1
            if m>1:
                self.assertFalse(np.array_equal(c.numpy(),
                                                c_ref.numpy()))

    def test_int_state(self):
        """Test internal state functionality of decoder.
        This implies that Nx1 iterations yields exact same result as N
        iterations."""
        batch_size = 1
        Niter = 5
        k = 100
        n = 200

        enc = LDPC5GEncoder(k,n)
        dec = LDPC5GDecoder(enc, num_iter=Niter)
        dec2 = LDPC5GDecoder(enc, num_iter=1, stateful=True)

        llr = tf.random.normal([batch_size, n], mean=4.2, stddev=1)

        res1 = dec(llr)

        res2, msg_vn = dec2([llr, None]) # iter 0 to init msg_vn

        for i in range(Niter-1): # remaining iterations
            res2, msg_vn = dec2([llr, msg_vn])

        # results must be the same, otherwise the internal state is not
        # correctly recovered
        self.assertTrue(np.allclose(res1,res2))

    def test_llrmax(self):
        """Test that llr_max can be set."""
        k = 12
        n = 20
        enc = LDPC5GEncoder(k,n)
        dec = LDPC5GDecoder(enc, hard_out=False, num_iter=0)

        # test default value
        llr_max_def = dec.llr_max.numpy() # get default value
        x = tf.ones((1,n))*100
        y = dec(x).numpy() # run 0 iterations
        np.max(y)==llr_max_def

        # set new llr_max
        llr_maxs = [17., 45.3, 78]
        for l in llr_maxs:
            dec.llr_max = l
            y = dec(x).numpy() # run 0 iterations
            print(np.abs(np.max(y)-l)<1e-6)
