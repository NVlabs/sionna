#
# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Userful metrics for the Sionna library."""

import tensorflow as tf
from tensorflow.keras.metrics import Metric
from tensorflow.keras.losses import BinaryCrossentropy


class BitwiseMutualInformation(Metric):
    """BitwiseMutualInformation(name="bitwise_mutual_information", **kwargs)

    Computes the bitwise mutual information between bits and LLRs.

    This class implements a Keras metric for the bitwise mutual information
    between a tensor of bits and LLR (logits).

    Input
    -----
        bits : tf.float32
            A tensor of arbitrary shape filled with ones and zeros.

        llr : tf.float32
            A tensor of the same shape as ``bits`` containing logits.

    Output
    ------
        : tf.float32
            A scalar, the bit-wise mutual information.

    """
    def __init__(self, name="bitwise_mutual_information", **kwargs):
        super().__init__(name, **kwargs)
        self.bmi = self.add_weight(name="bmi", initializer="zeros",
                                   dtype=tf.float32)
        self.counter = self.add_weight(name="counter", initializer="zeros")
        self.bce = BinaryCrossentropy(from_logits=True)

    def update_state(self, bits, llr):
        self.counter.assign_add(1)
        self.bmi.assign_add(1-self.bce(bits, llr)/tf.math.log(2.))

    def result(self):
        return tf.cast(tf.math.divide_no_nan(self.bmi, self.counter),
                       dtype=tf.float32)

    def reset_state(self):
        self.bmi.assign(0.0)
        self.counter.assign(0.0)

class BitErrorRate(Metric):
    """BitErrorRate(name="bit_error_rate", **kwargs)

    Computes the average bit error rate (BER) between two binary tensors.

    This class implements a Keras metric for the bit error rate
    between two tensors of bits.

    Input
    -----
        b : tf.float32
            A tensor of arbitrary shape filled with ones and
            zeros.

        b_hat : tf.float32
            A tensor of the same shape as ``b`` filled with
            ones and zeros.

    Output
    ------
        : tf.float32
            A scalar, the BER.
    """
    def __init__(self, name="bit_error_rate", **kwargs):
        super().__init__(name, **kwargs)
        self.ber = self.add_weight(name="ber",
                                   initializer="zeros",
                                   dtype=tf.float64)
        self.counter = self.add_weight(name="counter",
                                       initializer="zeros",
                                       dtype=tf.float64)

    def update_state(self, b, b_hat):
        self.counter.assign_add(1)
        self.ber.assign_add(compute_ber(b, b_hat))

    def result(self):
        #cast results of computer_ber for compatibility with tf.float32
        return tf.cast(tf.math.divide_no_nan(self.ber, self.counter),
                       dtype=tf.float32)

    def reset_state(self):
        self.ber.assign(0.0)
        self.counter.assign(0.0)

def compute_ber(b, b_hat):
    """Computes the bit error rate (BER) between two binary tensors.

    Input
    -----
        b : tf.float32
            A tensor of arbitrary shape filled with ones and
            zeros.

        b_hat : tf.float32
            A tensor of the same shape as ``b`` filled with
            ones and zeros.

    Output
    ------
        : tf.float64
            A scalar, the BER.
    """
    ber = tf.not_equal(b, b_hat)
    ber = tf.cast(ber, tf.float64) # tf.float64 to suport large batch-sizes
    return tf.reduce_mean(ber)

def compute_ser(s, s_hat):
    """Computes the symbol error rate (SER) between two integer tensors.

    Input
    -----
        s : tf.int
            A tensor of arbitrary shape filled with integers indicating
            the symbol indices.

        s_hat : tf.int
            A tensor of the same shape as ``s`` filled with integers indicating
            the estimated symbol indices.

    Output
    ------
        : tf.float64
            A scalar, the SER.
    """
    ser = tf.not_equal(s, s_hat)
    ser = tf.cast(ser, tf.float64) # tf.float64 to suport large batch-sizes
    return tf.reduce_mean(ser)

def compute_bler(b, b_hat):
    """Computes the block error rate (BLER) between two binary tensors.

    A block error happens if at least one element of ``b`` and ``b_hat``
    differ in one block. The BLER is evaluated over the last dimension of
    the input, i. e., all elements of the last dimension are considered to
    define a block.

    This is also sometimes referred to as `word error rate` or `frame error
    rate`.

    Input
    -----
        b : tf.float32
            A tensor of arbitrary shape filled with ones and
            zeros.

        b_hat : tf.float32
            A tensor of the same shape as ``b`` filled with
            ones and zeros.

    Output
    ------
        : tf.float64
            A scalar, the BLER.
    """
    bler = tf.reduce_any(tf.not_equal(b, b_hat), axis=-1)
    bler = tf.cast(bler, tf.float64) # tf.float64 to suport large batch-sizes
    return tf.reduce_mean(bler)

def count_errors(b, b_hat):
    """Counts the number of bit errors between two binary tensors.

    Input
    -----
        b : tf.float32
            A tensor of arbitrary shape filled with ones and
            zeros.

        b_hat : tf.float32
            A tensor of the same shape as ``b`` filled with
            ones and zeros.

    Output
    ------
        : tf.int64
            A scalar, the number of bit errors.
    """
    errors = tf.not_equal(b,b_hat)
    errors = tf.cast(errors, tf.int64)
    return tf.reduce_sum(errors)

def count_block_errors(b, b_hat):
    """Counts the number of block errors between two binary tensors.

    A block error happens if at least one element of ``b`` and ``b_hat``
    differ in one block. The BLER is evaluated over the last dimension of
    the input, i. e., all elements of the last dimension are considered to
    define a block.

    This is also sometimes referred to as `word error rate` or `frame error
    rate`.

    Input
    -----
        b : tf.float32
            A tensor of arbitrary shape filled with ones and
            zeros.

        b_hat : tf.float32
            A tensor of the same shape as ``b`` filled with
            ones and zeros.

    Output
    ------
        : tf.int64
            A scalar, the number of block errors.
    """
    errors = tf.reduce_any(tf.not_equal(b,b_hat), axis=-1)
    errors = tf.cast(errors, tf.int64)
    return tf.reduce_sum(errors)

