#
# SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
import unittest
import numpy as np
import tensorflow as tf
from sionna.channel import BinarySymmetricChannel, BinaryZChannel, BinaryMemorylessChannel, BinaryErasureChannel
from sionna.utils import BinarySource

class TestDMCs(unittest.TestCase):
    """General tests for all discrete channels."""

    def test_dtypes(self):
        """Test that different input dtypes are supported"""
        dtypes = (tf.float16, tf.float32, tf.float64,
                  tf.uint8, tf.uint16, tf.uint32, tf.uint64,
                  tf.int8, tf.int16, tf.int32, tf.int64)

        channels = (BinarySymmetricChannel, BinaryZChannel,
                    BinaryErasureChannel, BinaryMemorylessChannel)

        source = BinarySource()
        for channel in channels:
            for is_binary in (True, False):
                for return_llrs in (True, False):

                    # select valid dtypes for each mode
                    if is_binary:
                        if return_llrs:
                            dt_valid = (tf.float16, tf.float32, tf.float64)
                        else:
                            if channel is BinaryErasureChannel:
                                # no unints for BEC
                                dt_valid = (tf.float16, tf.float32, tf.float64,
                                            tf.int8, tf.int16, tf.int32, tf.
                                            int64)
                            else:
                                dt_valid = (tf.float16, tf.float32, tf.float64,
                                            tf.uint8, tf.uint16, tf.uint32,
                                            tf.uint64, tf.int8, tf.int16,
                                            tf.int32, tf.int64, tf.bool)
                    else:
                        if return_llrs:
                            dt_valid = (tf.float16, tf.float32, tf.float64)
                        else:
                            # no unsigned dtype
                            dt_valid = (tf.float16, tf.float32, tf.float64,
                                        tf.int8, tf.int16, tf.int32, tf.int64)
                    for dt in dtypes:
                        if dt in dt_valid:
                            ch = channel(bipolar_input=(not is_binary),
                                         return_llrs=return_llrs,
                                         dtype=dt)
                        else:
                            # most throw an error
                            with self.assertRaises(AssertionError):
                               ch = channel(bipolar_input=(not is_binary),
                                            return_llrs=return_llrs,
                                            dtype=dt)
                            continue # next dtype

                        if channel is BinaryMemorylessChannel:
                            pb = (0.1, 0.1)
                        else:
                            pb = 0.1
                        x = source((10, 11))
                        if not is_binary:
                            x = 2 * x - 1
                        x = tf.cast(x, dtype=dt)
                        pb = tf.cast(pb, dtype=dt)
                        y = ch((x, pb))
                        self.assertTrue(y.dtype==dt)

                        # also test graph / XLA mode
                        for jc in (False, True):
                            @tf.function(jit_compile=jc)
                            def run_graph(x, pb):
                                return ch((x, pb))
                            y = run_graph(x, pb)


    def test_llrs(self):
        """Test llr output against Monte Carlo based estimation."""
        num_samples = int(1e6)
        source = BinarySource()
        channel = BinaryMemorylessChannel(return_llrs=False,
                                          bipolar_input=False,
                                          llr_max=20.)

        channel_ref = BinaryMemorylessChannel(return_llrs=True,
                                              bipolar_input=False,
                                              llr_max=20.)

        # test different error probabilities
        pbs = [(0., 0.,), (0.1, 0.1), (0.5, 0.5),(0.99, 0.99),
                (0.1, 0.4), (0., 0.5), (0.01, 0.99)]
        for pb in pbs:

            x_tf = source((num_samples,))
            y_tf = channel((x_tf, pb))
            y_ref = channel_ref((x_tf, pb))
            x = x_tf.numpy()
            y = y_tf.numpy()

            trans_mat = np.zeros((2,2))

            for i in range(num_samples):
                trans_mat[int(x[i]), int(y[i])] +=1

            trans_mat /= num_samples
            # calculate LLRs based on simulated probabilities
            eps = 1e-20 # for stability
            l_0 = - (np.log(trans_mat[0,0]+eps)- np.log(trans_mat[1,0]+eps))
            l_1 = - (np.log(trans_mat[0,1]+eps) - np.log(trans_mat[1,1]+eps))

            # remove nans from div-by-zeros
            l_0 = np.nan_to_num(l_0)
            l_1 = np.nan_to_num(l_1)

            #clipping
            l_0 = np.clip(l_0, -20., 20)
            l_1 = np.clip(l_1, -20., 20)

            # allow certain tolerance as the values are based on Monte Carlo
            c1 = np.isclose(np.minimum(l_0, l_1),
                            tf.reduce_min(y_ref).numpy(),
                            rtol=0.01, atol=0.1)
            c2 = np.isclose(np.maximum(l_0, l_1),
                            tf.reduce_max(y_ref).numpy(),
                            rtol=0.01, atol=0.1)

            self.assertTrue(c1)
            self.assertTrue(c2)

    def test_gradient(self):
        """Test that channel is differentiable w.r.t pb."""
        bs = 10000

        optimizer = tf.keras.optimizers.Adam(learning_rate = 0.01)
        channel = BinarySymmetricChannel()
        x = tf.zeros((bs,))

        # randomly initialized variable
        pb = tf.Variable(0.1, trainable=True)

        # we approximate a target error rate via SGD
        target_ber = 0.4

        for _ in range(100):
            with tf.GradientTape() as tape:
                y = channel((x, pb))
                loss = tf.reduce_mean((tf.reduce_mean(y)-target_ber)**2)
            weights = [pb]
            grads = tape.gradient(loss, weights)
            optimizer.apply_gradients(zip(grads, weights))

        self.assertTrue(np.isclose(pb.numpy(), target_ber, rtol=0.1, atol=0.01))

    def test_pb(self):
        "Tests for correct error statistics and broadcasting"

        # large number of samples for sufficient statistics
        bs = 1e5

        # test different broadcastable shapes
        pbs = [(0.2, 0.4),
               (0.95, 0.2),
               (1.0, 0.0),
               ((0.1, 0.1, 0.1, 0.8),(0.4, 0.2, 0.3, 0.8)),
               (((0.2, 0.7, 0.0, 0.5), (0.02, 0.4, 0.9, 0.3)),
                ((0.2, 0.3, 0.1, 0.0), (0.2, 0.7, 0.2, 0.3)))]

        source = BinarySource()
        for pb in pbs:
            for is_binary in (True, False):
                channel = BinaryMemorylessChannel(bipolar_input=(not is_binary))

                # random bits
                x = source((int(bs), 2, 4))
                if not is_binary:
                    x = 2*x-1
                y = channel((x, pb))

                # count errors
                e = tf.where(x!=y, 1., 0.)

                if is_binary:
                    neutral_element = 0.
                else:
                    neutral_element = -1.
                # evaluate x=0 and x=1 separately
                e0 = tf.where(x==neutral_element, e, 0.)
                e1 = tf.where(x!=neutral_element, e, 0.)

                # get per bit position ber
                ber0 = tf.reduce_sum(e0, axis=0) \
                    / tf.reduce_sum(tf.cast(x==neutral_element,tf.float32), axis=0)
                ber1 = tf.reduce_sum(e1, axis=0) \
                    / tf.reduce_sum(tf.cast(x!=neutral_element, tf.float32), axis=0)

                # allow certain mismatch due to Monte-Carlo sampling
                self.assertTrue(np.all((ber0 - pb[0]) < 0.01))
                self.assertTrue(np.all((ber1 - pb[1]) < 0.01))


class TestBEC(unittest.TestCase):
    """Tests for Binary Erasure Channel."""

    def test_pb(self):
        "Tests for correct error statistics and broadcasting"

        # large number of samples for sufficient statistics
        bs = 1e5

        # test also different broadcastable shapes
        pbs = [0.2,
               0.95,
               (0.1, 0.1, 0.1, 0.8),
               ((0.2, 0.3, 0.1, 0.0), (0.2, 0.7, 0.2, 0.3))]

        source = BinarySource()
        for pb in pbs:
            for is_binary in (True, False):
                channel = BinaryErasureChannel(bipolar_input=(not is_binary))

                # random bits
                x = source((int(bs), 2, 4))
                if not is_binary:
                    x = 2*x-1
                y = channel((x, pb))

                if is_binary:
                    erased_element = -1.
                else:
                    erased_element = 0.

                # get per bit position ber
                e = tf.where(y==erased_element, 1., 0.)
                ber = tf.reduce_mean(e,axis=0)

                # allow certain mismatch due to Monte-Carlo sampling
                self.assertTrue(np.all((ber - pb) < 0.01))


