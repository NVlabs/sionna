#
# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0#
"""
Block for applying OFDM channel: single-tap channel response in the frequency
domain
"""

import tensorflow as tf
from sionna.phy import Block
from sionna.phy.utils import expand_to_rank
from .awgn import AWGN

class ApplyOFDMChannel(Block):
    # pylint: disable=line-too-long
    r"""
    Apply single-tap channel frequency responses to channel inputs

    For each OFDM symbol :math:`s` and subcarrier :math:`n`, the single-tap channel
    is applied as follows:

    .. math::
        y_{s,n} = \widehat{h}_{s, n} x_{s,n} + w_{s,n}

    where :math:`y_{s,n}` is the channel output computed by this layer,
    :math:`\widehat{h}_{s, n}` the frequency channel response (``h_freq``),
    :math:`x_{s,n}` the channel input ``x``, and :math:`w_{s,n}` the additive noise.

    For multiple-input multiple-output (MIMO) links, the channel output is computed for each antenna
    of each receiver and by summing over all the antennas of all transmitters.

    Parameters
    ----------
    precision : `None` (default) | "single" | "double"
        Precision used for internal calculations and outputs.
        If set to `None`,
        :attr:`~sionna.phy.config.Config.precision` is used.

    Input
    -----
    x :  [batch size, num_tx, num_tx_ant, num_ofdm_symbols, fft_size], `tf.complex`
        Channel inputs

    h_freq : [batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_ofdm_symbols, fft_size], `tf.complex`
        Channel frequency responses

    no : `None` (default) | tensor, `tf.float`
        Tensor whose shape can be broadcast to the shape of the
        channel outputs:
        [batch size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size].
        The (optional) noise power ``no`` is per complex dimension. If ``no`` is a
        scalar, noise of the same variance will be added to the outputs.
        If ``no`` is a tensor, it must have a shape that can be broadcast to
        the shape of the channel outputs. This allows, e.g., adding noise of
        different variance to each example in a batch. If ``no`` has a lower
        rank than the channel outputs, then ``no`` will be broadcast to the
        shape of the channel outputs by adding dummy dimensions after the
        last axis.

    Output
    -------
    y : [batch size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size], `tf.complex`
        Channel outputs
    """

    def __init__(self, precision=None, **kwargs):
        super().__init__(precision=precision, **kwargs)
        self._awgn = AWGN(precision=self.precision)

    def call(self, x, h_freq, no=None):

        # Apply the channel response
        x = expand_to_rank(x, h_freq.shape.rank, axis=1)
        y = tf.reduce_sum(tf.reduce_sum(h_freq*x, axis=4), axis=3)

        # Add AWGN if requested
        if no is not None:
            y = self._awgn(y, no)

        return y
