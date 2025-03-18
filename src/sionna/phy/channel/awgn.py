#
# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0#
"""Block for simulating an AWGN channel"""

import tensorflow as tf
from sionna.phy.block import Block
from sionna.phy.utils import expand_to_rank, complex_normal

class AWGN(Block):
    r"""
    Add complex AWGN to the inputs with a certain variance

    This layer blocks complex AWGN noise with variance ``no`` to the input.
    The noise has variance ``no/2`` per real dimension.
    It can be either a scalar or a tensor which can be broadcast to the shape
    of the input.

    Example
    --------

    Setting-up:

    >>> awgn_channel = AWGN()

    Running:

    >>> # x is the channel input
    >>> # no is the noise variance
    >>> y = awgn_channel(x, no)

    Parameters
    ----------
    precision : `None` (default) | "single" | "double"
        Precision used for internal calculations and outputs.
        If set to `None`,
        :attr:`~sionna.phy.config.Config.precision` is used.

    Input
    -----
    x :  Tensor, tf.complex
        Channel input

    no : Scalar or Tensor, `tf.float`
        Scalar or tensor whose shape can be broadcast to the shape of ``x``.
        The noise power ``no`` is per complex dimension. If ``no`` is a
        scalar, noise of the same variance will be added to the input.
        If ``no`` is a tensor, it must have a shape that can be broadcast to
        the shape of ``x``. This allows, e.g., adding noise of different
        variance to each example in a batch. If ``no`` has a lower rank than
        ``x``, then ``no`` will be broadcast to the shape of ``x`` by adding
        dummy dimensions after the last axis.

    Output
    -------
        y : Tensor with same shape as ``x``, `tf.complex`
            Channel output
    """

    def __init__(self, *, precision=None, **kwargs):
        super().__init__(precision=precision, **kwargs)

    def call(self, x, no):

        # Create tensors of real-valued Gaussian noise for each complex dim.
        noise = complex_normal(tf.shape(x), precision=self.precision)

        # Add extra dimensions for broadcasting
        no = expand_to_rank(no, tf.rank(x), axis=-1)

        # Apply variance scaling
        no = tf.cast(no, self.rdtype)
        noise *= tf.cast(tf.sqrt(no), noise.dtype)

        # Add noise to input
        y = x + noise

        return y
