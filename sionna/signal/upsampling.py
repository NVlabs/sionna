#
# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Layers implementing upsampling"""

import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.experimental.numpy import swapaxes
from sionna.utils.tensors import flatten_last_dims

class Upsampling(Layer):
    """Upsampling(samples_per_symbol, axis=-1, **kwargs)

    Upsamples a tensor along a specified axis by inserting zeros
    between samples.

    Parameters
    ----------
    samples_per_symbol: int
        The upsampling factor. If ``samples_per_symbol`` is equal to `n`,
        then the upsampled axis will be `n`-times longer.

    axis: int
        The dimension to be up-sampled. Must not be the first dimension.

    Input
    -----
    x : [...,n,...], tf.DType
        The tensor to be upsampled. `n` is the size of the `axis` dimension.

    Output
    ------
    y : [...,n*samples_per_symbol,...], same dtype as ``x``
        The upsampled tensor.
    """
    def __init__(self, samples_per_symbol, axis=-1, **kwargs):
        super().__init__(**kwargs)
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
