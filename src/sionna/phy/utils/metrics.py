#
# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0#
"""Functions to compute frequently used metrics in Sionna PHY"""

import tensorflow as tf
from sionna.phy import dtypes

def compute_ber(b, b_hat, precision="double"):
    """Computes the bit error rate (BER) between two binary tensors

    Input
    -----
    b : `tf.float` or `tf.int`
        A tensor of arbitrary shape filled with ones and
        zeros

    b_hat : `tf.float` or `tf.int`
        A tensor like ``b``

    precision : `str`, "single" | "double" (default)
        Precision used for internal calculations and outputs

    Output
    ------
    : `tf.float`
        BER
    """
    b_hat = tf.cast(b_hat, b.dtype)
    rdtype = dtypes[precision]["tf"]["rdtype"]
    ber = tf.not_equal(b, b_hat)
    ber = tf.cast(ber, rdtype)
    return tf.reduce_mean(ber)

def compute_ser(s, s_hat, precision="double"):
    """Computes the symbol error rate (SER) between two integer tensors

    Input
    -----
    s : `tf.float` or `tf.int`
        A tensor of arbitrary shape filled with integers

    s_hat : `tf.float` or `tf.int`
        A tensor like ``s``

    precision : `str`, "single" | "double" (default)
        Precision used for internal calculations and outputs

    Output
    ------
    : `tf.float`
        SER
    """
    s_hat = tf.cast(s_hat, s.dtype)
    rdtype = dtypes[precision]["tf"]["rdtype"]
    ser = tf.not_equal(s, s_hat)
    ser = tf.cast(ser, rdtype)
    return tf.reduce_mean(ser)

def compute_bler(b, b_hat, precision="double"):
    """Computes the block error rate (BLER) between two binary tensors

    A block error happens if at least one element of ``b`` and ``b_hat``
    differ in one block. The BLER is evaluated over the last dimension of
    the input, i. e., all elements of the last dimension are considered to
    define a block.

    This is also sometimes referred to as `word error rate` or `frame error
    rate`.

    Input
    -----
    b : `tf.float` or `tf.int`
        A tensor of arbitrary shape filled with ones and
        zeros

    b_hat : `tf.float` or `tf.int`
        A tensor like ``b``

    precision : `str`, "single" | "double" (default)
        Precision used for internal calculations and outputs

    Output
    ------
    : `tf.float`
        BLER
    """
    b_hat = tf.cast(b_hat, b.dtype)
    rdtype = dtypes[precision]["tf"]["rdtype"]
    bler = tf.reduce_any(tf.not_equal(b, b_hat), axis=-1)
    bler = tf.cast(bler, rdtype)
    return tf.reduce_mean(bler)

def count_errors(b, b_hat):
    """Counts the number of bit errors between two binary tensors

    Input
    -----
    b : `tf.float` or `tf.int`
        A tensor of arbitrary shape filled with ones and
        zeros

    b_hat : `tf.float` or `tf.int`
        A tensor like ``b``

    Output
    ------
    : `tf.int64`
        Number of bit errors
    """
    b_hat = tf.cast(b_hat, b.dtype)
    errors = tf.not_equal(b, b_hat)
    errors = tf.cast(errors, tf.int64)
    return tf.reduce_sum(errors)

def count_block_errors(b, b_hat):
    """Counts the number of block errors between two binary tensors

    A block error happens if at least one element of ``b`` and ``b_hat``
    differ in one block. The BLER is evaluated over the last dimension of
    the input, i. e., all elements of the last dimension are considered to
    define a block.

    This is also sometimes referred to as `word error rate` or `frame error
    rate`.

    Input
    -----
    b : `tf.float` or `tf.int`
        A tensor of arbitrary shape filled with ones and
        zeros

    b_hat : `tf.float` or `tf.int`
        A tensor like ``b``

    Output
    ------
    : `tf.int64`
        Number of block errors
    """
    b_hat = tf.cast(b_hat, b.dtype)
    errors = tf.reduce_any(tf.not_equal(b,b_hat), axis=-1)
    errors = tf.cast(errors, tf.int64)
    return tf.reduce_sum(errors)

