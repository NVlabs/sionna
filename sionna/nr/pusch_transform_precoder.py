#
# SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Class definitions for PUSCH transform precoder and deprecoder"""

import tensorflow as tf
from tensorflow.keras.layers import Layer


def _check_largest_prime_factor_not_larger_then_5(n):
    for p in [2, 3, 5]:
        while n % p == 0:
            n /= p
    if n > 1:
        raise ValueError(
            "Number of subcarriers shouldn't have a prime factor > 5")


class PUSCHTransformPrecoder(Layer):
    r"""PUSCHTransformPrecoder(num_subcarriers, dtype=tf.complex64, **kwargs)
    Performs transform precoding of layer mapped symbols as defined in
    [3GPP38211]_ Sec. 6.3.1.4.

    The input will be reshaped into blocks of size ``num_subcarriers`` to which
    the FFT will be applied individually.

    The class inherits from the Keras layer class and can be used as layer in a
    Keras model.

    Parameters
    ----------
        num_subcarriers: int
            Number of subcarriers. The largest prime factor must not be larger
            than 5.

        dtype : One of [tf.complex64, tf.complex128]
            Dtype of inputs and outputs. Defaults to tf.complex64.

    Input
    -----
        inputs: [...,n], tf.complex
            Tensor containing the sequence of symbols to be transform precoded.

    Output
    ------
        : [...,n], tf.complex
            Tensor containing the sequence of symbols that have been transform
            precoded.
    """

    def __init__(self,
                 num_subcarriers,
                 dtype=tf.complex64,
                 **kwargs):
        super().__init__(dtype=dtype, **kwargs)
        _check_largest_prime_factor_not_larger_then_5(num_subcarriers)
        self._num_subcarriers = num_subcarriers

    def call(self, y):
        orig_shape = tf.shape(y)
        y_reshaped = tf.reshape(y, [-1, self._num_subcarriers])
        y_transformed = tf.cast(tf.sqrt(1 / self._num_subcarriers),
                                self._dtype) * tf.signal.fft(y_reshaped)
        y_result = tf.reshape(y_transformed, orig_shape)
        return y_result


class PUSCHTransformDeprecoder(Layer):
    r"""PUSCHTransformDeprecoder(num_subcarriers, dtype=tf.complex64, **kwargs)
    Performs transform deprecoding of layer mapped symbols as defined in
    [3GPP38211]_ Sec. 6.3.1.4.

    The input will be reshaped into blocks of size ``num_subcarriers`` to which
    the IFFT will be applied individually.

    The class inherits from the Keras layer class and can be used as layer in a
    Keras model.

    Parameters
    ----------
        num_subcarriers: int
            Number of subcarriers. The largest prime factor must not be larger
            than 5.

        dtype : One of [tf.complex64, tf.complex128]
            Dtype of inputs and outputs. Defaults to tf.complex64.

    Input
    -----
        inputs: [...,n], tf.complex
            Tensor containing the sequence of symbols after transform precoding.

    Output
    ------
        : [...,n], tf.complex
            Tensor containing the sequence of symbols before transform
            precoding.
    """

    def __init__(self,
                 num_subcarriers,
                 dtype=tf.complex64,
                 **kwargs):
        super().__init__(dtype=dtype, **kwargs)
        _check_largest_prime_factor_not_larger_then_5(num_subcarriers)
        self._num_subcarriers = num_subcarriers

    def call(self, y):
        orig_shape = tf.shape(y)
        y_reshaped = tf.reshape(y, [-1, self._num_subcarriers])
        y_transformed = tf.cast(tf.sqrt(float(self._num_subcarriers)),
                                self._dtype) * tf.signal.ifft(y_reshaped)
        y_result = tf.reshape(y_transformed, orig_shape)
        return y_result
