#
# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0#
"""Class definition for the OFDM Modulator"""

import tensorflow as tf
from tensorflow.signal import ifftshift

from sionna.phy import Block
from sionna.phy.utils import flatten_last_dims
from sionna.phy.signal import ifft


class OFDMModulator(Block):
    # pylint: disable=line-too-long
    """
    Computes the time-domain representation of an OFDM resource grid
    with (optional) cyclic prefix

    Parameters
    ----------
    cyclic_prefix_length : `int` (default 0) | [num_ofdm_symbols], `int`
        Integer or vector of integers indicating the length of the
        cyclic prefix that is prepended to each OFDM symbol. None of its
        elements can be larger than the FFT size.

    precision : `None` (default) | "single" | "double"
        Precision used for internal calculations and outputs.
        If set to `None`,
        :attr:`~sionna.phy.config.Config.precision` is used.

    Input
    -----
    : [...,num_ofdm_symbols,fft_size], `tf.complex`
        Resource grid in the frequency domain

    Output
    ------
    : [...,num_ofdm_symbols*(fft_size+cyclic_prefix_length)] or [...,num_ofdm_symbols*fft_size+sum(cyclic_prefix_length)], `tf.complex`
        Time-domain OFDM signal
    """
    def __init__(self, cyclic_prefix_length=0, precision=None, **kwargs):
        super().__init__(precision=precision, **kwargs)
        self._cyclic_prefix_length = None
        self.cyclic_prefix_length = cyclic_prefix_length

    @property
    def cyclic_prefix_length(self):
        """
        scalar or [num_ofdm_symbols], int : Get/set the cyclic prefix length
        """
        return self._cyclic_prefix_length

    @cyclic_prefix_length.setter
    def cyclic_prefix_length(self, value):
        value = tf.cast(value, tf.int32)
        if not tf.reduce_all(value>=0):
            msg = "`cyclic_prefix_length` must be nonnegative."
            raise ValueError(msg)
        if not 0<= tf.rank(value)<=1:
            msg = "`cyclic_prefix_length` must be of rank 0 or 1"
            raise ValueError(msg)
        self._cyclic_prefix_length = value

    def build(self, input_shape):
        num_ofdm_symbols, fft_size = input_shape[-2:]
        if not tf.reduce_all(self.cyclic_prefix_length<=fft_size):
            msg = "`cyclic_prefix_length` cannot be larger than `fft_size`."
            raise ValueError(msg)
        if len(self.cyclic_prefix_length.shape)==1:
            if not self.cyclic_prefix_length.shape[0]==num_ofdm_symbols:
                msg = "`cyclic_prefix_length` must be of size [num_ofdm_symbols]"
                raise ValueError(msg)

            # Compute indices of CP symbols
            # These are offset by the number of the OFDM symbol
            # [num_ofdm_symbols, 1]
            offsets = tf.expand_dims(tf.range(1, num_ofdm_symbols+1)*fft_size,
                                     1)
            # [num_ofdm_symbols, None] (ragged tensor)
            cp_ind = tf.ragged.range(starts=-self.cyclic_prefix_length,
                                     limits=0) + offsets

            # Compute indices of symbols containing the actual sequence
            # [num_ofdm_symbols, fft_size]
            data_ind = tf.repeat(tf.expand_dims(tf.range(0, fft_size), 0),
                                 num_ofdm_symbols, 0) + offsets - fft_size

            # Concat CP and sequence indices
            # [num_ofdm_symbols, None]
            ind = tf.concat([cp_ind, data_ind], axis=-1)

            # Flatten in time domain
            # [num_ofdm_symbols *fft_size + sum(cyclic_prefix_length)]
            self._ind = ind.flat_values

    def call(self, inputs):

        # Shift DC subcarrier to first position
        x_freq = ifftshift(inputs, axes=-1)

        # Compute IFFT along the last dimension
        x_time = ifft(x_freq)

        if len(self.cyclic_prefix_length.shape)==1:
            # Individual CP length per OFDM symbol

            # Flatten last two dimensions
            x_time = flatten_last_dims(x_time, 2)

            # Gather full time-domain signal
            return tf.gather(x_time, self._ind, axis=-1)

        else:
            # Same CP length for all OFDM symbols

            # Obtain cyclic prefix
            cp = x_time[...,tf.shape(x_time)[-1]-self._cyclic_prefix_length:]

            # Prepend cyclic prefix
            x_time = tf.concat([cp, x_time], -1)

            # Serialize last two dimensions
            return  flatten_last_dims(x_time, 2)
