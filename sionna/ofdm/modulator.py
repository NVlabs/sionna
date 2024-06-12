#
# SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Class definition for the OFDM Modulator"""

import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.signal import ifftshift
from sionna.utils import flatten_last_dims
from sionna.signal import ifft


class OFDMModulator(Layer):
    # pylint: disable=line-too-long
    """
    OFDMModulator(cyclic_prefix_length=0, cyclic_prefix_length_first_symbol=None, symbols_per_block=1, **kwargs)

    Computes the time-domain representation of an OFDM resource grid
    with (optional) cyclic prefix.

    When only `cyclic_prefix_length` is given then a cyclic prefix of length
    `cyclic_prefix_length` is prepended for each symbol. When additionally
    `cyclic_prefix_length_first_symbol` and `symbols_per_block` are given then
    the length of the cyclic prefix is `cyclic_prefix_length_first_symbol` for
    the first symbol of each block and `cyclic_prefix_length` for the
    remaining symbols. For LTE one block corresponds to one slot (i.e., 7
    symbols). For 5G NR one block corresponds to one half subframe and the
    number of symbols depends on the numerology.

    Parameters
    ----------
    cyclic_prefix_length : int
        Integer indicating the length of the cyclic prefix that it prepended
        to each OFDM symbol (except for the first symbol of each block if
        `cyclic_prefix_length_first_symbol` and `symbols per block` is given).
        It cannot be longer than the FFT size.

    cyclic_prefix_length_first_symbol : int
        Integer indicating the length of the cyclic prefix that it prepended
        to the first OFDM symbol of each block. It cannot be longer than the
        FFT size.

    symbols_per_block : int
        Integer indicating the number of symbols per block.

    Input
    -----
    : [...,num_ofdm_symbols,fft_size], tf.complex
        A resource grid in the frequency domain.

    Output
    ------
    : [...,num_ofdm_symbols*(fft_size+cyclic_prefix_length)], tf.complex
        Time-domain OFDM signal.
    """

    def __init__(self, cyclic_prefix_length=0,
                 cyclic_prefix_length_first_symbol=None, symbols_per_block=1,
                 **kwargs):
        super().__init__(**kwargs)
        self.cyclic_prefix_length = cyclic_prefix_length
        self.cyclic_prefix_length_first_symbol = (
            cyclic_prefix_length_first_symbol)
        self.symbols_per_block = symbols_per_block

    @property
    def cyclic_prefix_length(self):
        return self._cyclic_prefix_length

    @cyclic_prefix_length.setter
    def cyclic_prefix_length(self, value):
        assert isinstance(value, int) and value >=0,\
            "`cyclic_prefix_length` must be a nonnegative integer."
        self._cyclic_prefix_length = value

    @property
    def cyclic_prefix_length_first_symbol(self):
        if self._cyclic_prefix_length_first_symbol is None:
            return self._cyclic_prefix_length
        else:
            return self._cyclic_prefix_length_first_symbol

    @cyclic_prefix_length_first_symbol.setter
    def cyclic_prefix_length_first_symbol(self, value):
        assert (value is None or isinstance(value, int) and
                value >= self._cyclic_prefix_length),\
            ("`cyclic_prefix_length_first_symbol` must be larger or equal " +
             "to `cyclic_prefix_length`.")
        self._cyclic_prefix_length_first_symbol = value

    @property
    def symbols_per_block(self):
        return self._symbols_per_block

    @symbols_per_block.setter
    def symbols_per_block(self, value):
        assert isinstance(value, int) and value>=1,\
            "`symbols_per_block` must be a positive integer."
        self._symbols_per_block = value

    def build(self, input_shape):
        fft_size = input_shape[-1]
        num_ofdm_symbols = input_shape[-2]

        # Verify that cyclic prefix is not longer than the FFT size.
        assert self.cyclic_prefix_length<=fft_size, \
            "shape(inputs)[-1] must not be smaller than `cylic_prefix_length`"
        assert self.cyclic_prefix_length_first_symbol <= fft_size, \
            ("shape(inputs)[-1] must not be smaller than " +
             " `cylic_prefix_length_first_symbol`")

        # Compute padding size to fill the last block
        self._num_pad_symbols = -num_ofdm_symbols % self.symbols_per_block

    def call(self, inputs):
        fft_size = tf.shape(inputs)[-1]
        num_ofdm_symbols = tf.shape(inputs)[-2]
        batch_dims = tf.shape(inputs)[:-2]

        # Shift DC subcarrier to first position
        inputs = ifftshift(inputs, axes=-1)

        # Compute IFFT along the last dimension
        x = ifft(inputs)

        # Add padding to fill up last block
        if self._num_pad_symbols != 0:
            padding_shape = tf.concat([batch_dims,
                                   [self._num_pad_symbols, fft_size]], axis=0)
            padding = tf.zeros(padding_shape, dtype=x.dtype)
            x = tf.concat([x, padding], axis=-2)

        # Obtain cyclic prefix
        cp = x[...,fft_size-self.cyclic_prefix_length:]

        # Prepend cyclic prefix
        x = tf.concat([cp, x], -1)

        # Reshape to blocks
        num_blocks = tf.math.ceil(num_ofdm_symbols / self.symbols_per_block)
        samples_per_block = (self.symbols_per_block *
                             (self.cyclic_prefix_length + fft_size))
        shape = tf.concat([batch_dims,
                           [num_blocks, samples_per_block]], axis=0)
        x = tf.reshape(x, shape)

        # Obtain additional cyclic prefix for first symbol in block
        cp = x[...,fft_size+self.cyclic_prefix_length-
                   self.cyclic_prefix_length_first_symbol:fft_size]

        # Prepend additional cyclic prefix
        x = tf.concat([cp, x], -1)

        # Serialize last two dimensions
        x = flatten_last_dims(x, 2)

        # Remove padding
        if self._num_pad_symbols != 0:
            x = x[..., :-self._num_pad_symbols *
                         (self.cyclic_prefix_length + fft_size)]

        return x
