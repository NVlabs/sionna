#
# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0#

"""PUSCH Precoding Block for the the NR (5G) module of Sionna PHY"""

import tensorflow as tf

from sionna.phy import Block

class PUSCHPrecoder(Block):
    # pylint: disable=line-too-long
    r"""
    Precodes a batch of modulated symbols mapped onto a resource grid
    for PUSCH transmissions.

    Each transmitter is assumed to have its
    own precoding matrix.

    Parameters
    ----------
    precoding_matrices : `list`, [num_tx, num_antenna_ports, num_layers]. `tf.complex`
        List of precoding matrices, one for each transmitter.
        All precoding matrices must have the same shape.

    precision : `None` (default) | "single" | "double"
        Precision used for internal calculations and outputs.
        If set to `None`,
        :attr:`~sionna.phy.config.Config.precision` is used.

    Input
    -----
        : [batch_size, num_tx, num_layers, num_symbols_per_slot,num_subcarriers], `tf.complex`
            Batch of resource grids to be precoded

    Output
    ------
        : [batch_size, num_tx, num_antenna_ports, num_symbols_per_slot, num_subcarriers], `tf.complex`
            Batch of precoded resource grids
    """
    def __init__(self,
                 precoding_matrices,
                 precision=None,
                 **kwargs):
        super().__init__(precision=precision, **kwargs)

        self._num_tx = len(precoding_matrices)

        # Check that all precoding matrices have the same shape
        shape = precoding_matrices[0].shape
        w_list = []
        for w in precoding_matrices:
            assert w.shape[0]==shape[0] and w.shape[1]==shape[1], \
                "All precoding matrices must have the same shape"
            w_list.append(w)

        # w has shape:
        #[num_tx, num_antenna_ports, num_layers]
        self._w = tf.constant(w_list, self.cdtype)

    def build(self, input_shape):
        _, num_tx, num_layers, _, _ = input_shape
        assert num_tx==len(self._w), \
            f"""The input shape is for {num_tx} transmitters, but you have
                configured precoding matrices for {len(self._w)}."""
        assert num_layers==self._w[0].shape[1], \
            f"""You have configured precoding matrices for
                {self._w[0].shape[1]} layers, but the input
                provides {num_layers} layers."""

    def call(self, inputs):

        # inputs has shape:
        # [batch_size, num_tx, num_layers, num_symbols_per_slot,...
        #  ..., num_subcarriers]

        # Change ordering of dimensions:
        # [batch_size, num_symbols_per_slot, num_subcarriers, num_tx,...
        #  ..., num_layers]
        inputs = tf.transpose(inputs, [0, 3, 4, 1, 2])

        # Add dimension for matrix multiplication:
        inputs = tf.expand_dims(inputs, -1)

        # Precode:
        # [batch_size, num_symbols_per_slot, num_subcarriers,...
        #  ..., num_tx, num_antenna_ports]
        z = tf.squeeze(tf.matmul(self._w, inputs), -1)

        # Re-order:
        # [batch_size, num_tx, num_antenna_ports, num_symbols_per_slot,...
        #  ..., num_subcarriers]
        z = tf.transpose(z, [0, 3, 4, 1, 2])

        return z
