#
# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0#
"""Block implementing upsampling"""

import tensorflow as tf
from tensorflow.experimental.numpy import swapaxes
from sionna.phy import Block
from sionna.phy.utils import flatten_last_dims

class Upsampling(Block):
    """Upsampling(samples_per_symbol, axis=-1, precision=None, **kwargs)

    Upsamples a tensor along a specified axis by inserting zeros
    between samples

    Parameters
    ----------
    samples_per_symbol: `int`
        Upsampling factor. If ``samples_per_symbol`` is equal to `n`,
        then the upsampled axis will be `n`-times longer.

    axis: `int`, (default -1)
        Dimension to be up-sampled. Must not be the first dimension.

    precision : `None` (default) | "single" | "double"
        Precision used for internal calculations and outputs.
        If set to `None`,
        :attr:`~sionna.phy.config.Config.precision` is used.

    Input
    -----
    x : [...,n,...], `tf.float` or `tf.complex`
        Tensor to be upsampled. `n` is the size of the `axis` dimension.

    Output
    ------
    y : [...,n*samples_per_symbol,...], `tf.float` or `tf.complex`
        Upsampled tensor
    """
    def __init__(self,
                 samples_per_symbol,
                 axis=-1,
                 precision=None,
                **kwargs):
        super().__init__(precision=precision, **kwargs)
        self._samples_per_symbol = samples_per_symbol
        self._axis = axis

    def build(self, input_shape):
        paddings = []
        for _ in range(len(input_shape)):
            paddings.append([0, 0])
        paddings.append([0, self._samples_per_symbol-1])
        self._paddings = paddings

    def call(self, inputs):
        x = swapaxes(inputs, self._axis, -1)
        x = tf.expand_dims(x, -1)
        x = tf.pad(x,
                   self._paddings,
                   constant_values=tf.cast(0, dtype=x.dtype))
        x = flatten_last_dims(x, 2)
        x = swapaxes(x, -1, self._axis)
        return x
