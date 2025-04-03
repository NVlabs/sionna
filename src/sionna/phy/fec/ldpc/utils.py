#
# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0#
"""Utility functions for LDPC decoding."""

import tensorflow as tf
import matplotlib.pyplot as plt
from sionna.phy.fec.utils import llr2mi
from sionna.phy.block import Object


class EXITCallback():
    # pylint: disable=line-too-long
    """Callback for the LDPCBPDecoder to track EXIT statistics.

    Can be registered as ``c2v_callbacks`` ``v2c_callbacks`` in the
    :class:`~sionna.phy.fec.ldpc.decoding.LDPCBPDecoder` and the
    :class:`~sionna.phy.fec.ldpc.decoding.LDPC5GDecoder`.

    This callback requires all-zero codeword simulations.

    Parameters
    ----------
    num_iter: int
        Maximum number of decoding iterations.

    Input
    -----
    msg: [num_vns, None, batch_size], tf.ragged, tf.float
        The v2c or c2v messages as ragged tensor

    it: int
        Current number of decoding iterations

    Output
    ------
    : tf.ragged, tf.float
        Same as ``msg``

    """
    def __init__(self, num_iter):
        self._mi = tf.Variable(tf.zeros(num_iter+1), dtype=tf.float32)
        self._num_samples = tf.Variable(tf.zeros(num_iter+1), dtype=tf.float32)

    @property
    def mi(self):
        """Mutual information after each iteration"""
        return self._mi / self._num_samples

    def __call__(self, msg, it, *args, **kwargs):
        self._mi[it].assign(llr2mi(-1*msg.flat_values)+self._mi[it])
        self._num_samples[it].assign(1.+self._num_samples[it])
        return msg


class DecoderStatisticsCallback():
    """Callback for the LDPCBPDecoder to track decoder statistics.

    Can be registered as ``c2v_callbacks`` in the
    :class:`~sionna.phy.fec.ldpc.decoding.LDPCBPDecoder` and the
    :class:`~sionna.phy.fec.ldpc.decoding.LDPC5GDecoder`.

    Remark: the decoding statistics are based on CN convergence, i.e.,
    successful decoding is assumed if all check nodes are fulfilled.
    This overestimates the success-rate as it includes cases where the decoder
    converges to the wrong codeword.

    Parameters
    ----------
        num_iter: int
            Maximum number of decoding iterations.

    Input
    -----
    msg: [num_vns, None, batch_size], tf.ragged, tf.float
        v2c messages as ragged tensor

    it: int
        Current number of decoding iterations

    Output
    ------
    : tf.ragged, tf.float
        Same as ``msg``
    """
    def __init__(self, num_iter):
        self._num_samples = tf.Variable(tf.zeros((num_iter,), tf.int64),
                                        dtype=tf.int64)
        self._decoded_samples = tf.Variable(tf.zeros((num_iter,), tf.int64),
                                            dtype=tf.int64)
        self._num_iter = num_iter

    @property
    def num_samples(self):
        """Total number of processed codewords"""
        return self._num_samples

    @property
    def num_decoded_cws(self):
        """Number of decoded codewords after each iteration"""
        return self._decoded_samples

    @property
    def success_rate(self):
        """Success rate after each iteration"""
        succ = tf.cast(self._decoded_samples, tf.float64)
        num_samples = tf.cast(self._num_samples, tf.float64)
        return succ/num_samples

    @property
    def avg_number_iterations(self):
        """Average number of decoding iterations"""
        num_decoded = tf.cast(self._decoded_samples, tf.float64)
        num_samples = tf.cast(self._num_samples, tf.float64)

        num_active = num_samples - num_decoded

        total_iters = tf.reduce_sum(num_active)
        avg_iter = total_iters / num_samples[0]
        return avg_iter

    def reset_stats(self):
        """Reset internal statistics"""
        self._num_samples.assign(tf.zeros((self._num_iter,), tf.int64))
        self._decoded_samples.assign(tf.zeros((self._num_iter,), tf.int64))

    def __call__(self, msg, it, *args, **kwargs):
        # check sign of each CN
        sign_val = tf.ragged.map_flat_values(lambda x :
                                         tf.where(tf.equal(x, 0),
                                         tf.ones_like(x), x),
                                         tf.sign(msg))
        # calculate sign of entire node
        sign_node = tf.reduce_prod(sign_val, axis=1)
        node_success = tf.where(sign_node>0, True, False)
        cw_success = tf.reduce_all(node_success, axis=0)
        x = tf.where(cw_success, 1., 0)
        x = tf.reduce_sum(x)

        # and update internal variables
        num_samples = tf.cast(msg.shape[-1], tf.int64)
        num_decoded = tf.cast(x, tf.int64)
        updates = tf.tensor_scatter_nd_update(
                                    tf.zeros([self._num_iter,], tf.int64),
                                    [[it]], [num_samples])
        self._num_samples.assign_add(updates)

        updates = tf.tensor_scatter_nd_update(
                                    tf.zeros([self._num_iter,], tf.int64),
                                    [[it]], [num_decoded])
        self._decoded_samples.assign_add(updates)
        # return original message for compatibility with callbacks
        return msg

class WeightedBPCallback(Object):
    # pylint: disable=line-too-long
    r"""Callback for the LDPCBPDecoder to enable weighted BP [Nachmani]_.

    The BP decoder is fully differentiable and can be made trainable
    by following the concept of `weighted BP` [Nachmani]_ as shown in Fig. 1
    leading to

    .. math::
        y_{j \to i} = 2 \operatorname{tanh}^{-1} \left( \prod_{i' \in \mathcal{N}(j) \setminus i} \operatorname{tanh} \left( \frac{\textcolor{red}{w_{i' \to j}} \cdot x_{i' \to j}}{2} \right) \right)

    where :math:`w_{i \to j}` denotes the trainable weight of message
    :math:`x_{i \to j}`.
    Please note that the training of some check node types may be not supported.

    ..  figure:: ../figures/weighted_bp.png

        Fig. 1: Weighted BP as proposed in [Nachmani]_.

    Can be registered as ``c2v_callbacks`` and ``v2c_callbacks`` in the
    :class:`~sionna.phy.fec.ldpc.decoding.LDPCBPDecoder` and the
    :class:`~sionna.phy.fec.ldpc.decoding.LDPC5GDecoder`.

    Parameters
    ----------
    num_edges: int
        Number of edges in the decoding graph

    precision : `None` (default) | 'single' | 'double'
        Precision used for internal calculations and outputs.
        If set to `None`, :py:attr:`~sionna.config.precision` is used.

    Input
    -----
    msg: [num_vns, None, batch_size], tf.ragged, tf.float
        v2c messages as ragged tensor

    Output
    ------
    : tf.ragged, tf.float
        Same as ``msg``
    """
    def __init__(self, num_edges, precision=None, **kwargs):
        """Nothing to init"""
        super().__init__(precision=precision, **kwargs)

        self._edge_weights = tf.Variable(tf.ones((num_edges,)),
                                         trainable=True,
                                         dtype=self.rdtype)

    @property
    def weights(self):
        return self._edge_weights

    def show_weights(self, size=7):
        """Show histogram of trainable weights.

        Input
        -----
            size: float
                Figure size of the matplotlib figure.
        """
        plt.figure(figsize=(size,size))
        plt.hist(self._edge_weights.numpy(), density=True, bins=20, align='mid')
        plt.xlabel('weight value')
        plt.ylabel('density')
        plt.grid(True, which='both', axis='both')
        plt.title('Weight Distribution')

    def _mult_weights(self, x):
        """Multiply messages with trainable weights for weighted BP."""
        # transpose for simpler broadcasting of training variables
        x = tf.transpose(x, (1, 0))
        x = tf.math.multiply(x, self._edge_weights)
        x = tf.transpose(x, (1, 0))
        return x

    def __call__(self, msg, *args):
        # transpose for simpler broadcasting of training variables
        # Remark: BP uses internal shape [num_edges, batchsize]
        msg = tf.ragged.map_flat_values(self._mult_weights, msg)
        return msg
