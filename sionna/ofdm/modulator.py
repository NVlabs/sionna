#
# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Class definition for the OFDM Modulator"""

import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.signal import ifftshift
from sionna.utils import flatten_last_dims
from sionna.signal import ifft


class OFDMModulator(Layer):
    """
    OFDMModulator(cyclic_prefix_length, **kwargs)

    Computes the time-domain representation of an OFDM resource grid
    with (optional) cyclic prefix.

    Parameters
    ----------
    cyclic_prefix_length : int
        Integer indicating the length of the
        cyclic prefix that it prepended to each OFDM symbol. It cannot
        be longer than the FFT size.

    Input
    -----
    : [...,num_ofdm_symbols,fft_size], tf.complex
        A resource grid in the frequency domain.

    Output
    ------
    : [...,num_ofdm_symbols*(fft_size+cyclic_prefix_length)], tf.complex
        Time-domain OFDM signal.
    """

    def __init__(self, cyclic_prefix_length=0, **kwargs):
        super().__init__(**kwargs)
        self.cyclic_prefix_length = cyclic_prefix_length

    @property
    def cyclic_prefix_length(self):
        return self._cyclic_prefix_length

    @cyclic_prefix_length.setter
    def cyclic_prefix_length(self, value):
        assert value >=0, "`cyclic_prefix_length` must be nonnegative."
        self._cyclic_prefix_length = value

    def build(self, input_shape):
        # Verify that cyclic prefix is not longer than the FFT size.
        fft_size = input_shape[-1]
        assert self.cyclic_prefix_length<=fft_size, \
            "shape(inputs)[-1] must not be smaller than `cylic_prefix_length`"

    def call(self, inputs):
        # Shift DC subcarrier to first position
        inputs = ifftshift(inputs, axes=-1)

        # Compute IFFT along the last dimension
        x = ifft(inputs)

        # Obtain cyclic prefix
        cp = x[...,tf.shape(inputs)[-1]-self._cyclic_prefix_length:]

        # Prepend cyclic prefix
        x = tf.concat([cp, x], -1)

        # Serialize last two dimensions
        x = flatten_last_dims(x, 2)

        return x
