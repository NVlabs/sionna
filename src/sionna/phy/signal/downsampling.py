#
# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0#
"""Block implementing downsampling"""

from tensorflow.experimental.numpy import swapaxes
from sionna.phy import Block

class Downsampling(Block):
    # pylint: disable=line-too-long
    """
    Downsamples a tensor along a specified axis by retaining one out of
    ``samples_per_symbol`` elements

    Parameters
    ----------
    samples_per_symbol: `int`
        Downsampling factor. If ``samples_per_symbol`` is equal to `n`, then the
        downsampled axis will be `n`-times shorter.

    offset: `int`, (default 0)
        Index of the first element to be retained

    num_symbols: `None` (default) | `int`
        Total number of symbols to be retained after downsampling

    axis: `int`, (default -1)
        Dimension to be downsampled. Must not be the first dimension.

    precision : `None` (default) | "single" | "double"
        Precision used for internal calculations and outputs.
        If set to `None`,
        :attr:`~sionna.phy.config.Config.precision` is used.

    Input
    -----
    x : [...,n,...], `tf.float` or `tf.complex`
        Tensor to be downsampled. `n` is the size of the `axis` dimension.

    Output
    ------
    y : [...,k,...], `tf.float` or `tf.complex`
        Downsampled tensor, where ``k``
        is min((``n``-``offset``)//``samples_per_symbol``, ``num_symbols``).
    """
    def __init__(self,
                 samples_per_symbol,
                 offset=0,
                 num_symbols=None,
                 axis=-1,
                 precision=None,
                 **kwargs):
        super().__init__(precision=precision, **kwargs)
        self._samples_per_symbol = samples_per_symbol
        self._offset = offset
        self._num_symbols = num_symbols
        self._axis = axis

    def call(self, inputs):
        # Put selected axis last
        x = swapaxes(inputs, self._axis, -1)

        # Downsample
        x = x[...,self._offset::self._samples_per_symbol]

        if self._num_symbols is not None:
            x = x[...,:self._num_symbols]

        # Put last axis to original position
        x = swapaxes(x, -1, self._axis)

        return x
