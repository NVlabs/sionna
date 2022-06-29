#
# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Layers implementing downsampling"""

from tensorflow.keras.layers import Layer
from tensorflow.experimental.numpy import swapaxes

class Downsampling(Layer):
    # pylint: disable=line-too-long
    """Downsampling(samples_per_symbol, offset=0, num_symbols=None, axis=-1, **kwargs)

    Downsamples a tensor along a specified axis by retaining one out of
    ``samples_per_symbol`` elements.

    Parameters
    ----------
    samples_per_symbol: int
        The downsampling factor. If ``samples_per_symbol`` is equal to `n`, then the
        downsampled axis will be `n`-times shorter.

    offset: int
        Defines the index of the first element to be retained.
        Defaults to zero.

    num_symbols: int
        Defines the total number of symbols to be retained after
        downsampling.
        Defaults to None (i.e., the maximum possible number).

    axis: int
        The dimension to be downsampled. Must not be the first dimension.

    Input
    -----
    x : [...,n,...], tf.DType
        The tensor to be downsampled. `n` is the size of the `axis` dimension.

    Output
    ------
    y : [...,k,...], same dtype as ``x``
        The downsampled tensor, where ``k``
        is min((``n``-``offset``)//``samples_per_symbol``, ``num_symbols``).
    """
    def __init__(self,
                 samples_per_symbol,
                 offset=0,
                 num_symbols=None,
                 axis=-1, **kwargs):
        super().__init__(**kwargs)
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
