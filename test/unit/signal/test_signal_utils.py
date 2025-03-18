#
# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0#

import pytest
import numpy as np
import tensorflow as tf
from sionna.phy import config, dtypes
from sionna.phy.signal import convolve

@pytest.mark.parametrize("inp_dtype", [tf.float32, tf.float64, tf.complex64, tf.complex128])
@pytest.mark.parametrize("ker_dtype", [tf.float32, tf.float64, tf.complex64, tf.complex128])
@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.parametrize("padding", ['valid', 'same', 'full'])
def test_dtype(inp_dtype, ker_dtype, padding, precision):
    """Test the output dtype for all possible combinations of
    input dypes"""
    if inp_dtype.is_complex:
        inp = tf.complex(config.tf_rng.uniform([64, 100],
                            dtype=inp_dtype.real_dtype),
                            config.tf_rng.uniform([64, 100],
                            dtype=inp_dtype.real_dtype))
    else:
        inp = config.tf_rng.uniform([64, 100], dtype=inp_dtype)
    if ker_dtype.is_complex:
        ker = tf.complex(config.tf_rng.uniform([10],
                            dtype=ker_dtype.real_dtype),
                            config.tf_rng.uniform([10],
                            dtype=ker_dtype.real_dtype))
    else:
        ker = config.tf_rng.uniform([10], dtype=ker_dtype)

    if inp_dtype.is_complex or ker_dtype.is_complex:
        out_dtype = dtypes[precision]["tf"]["cdtype"]
    else:
        out_dtype = dtypes[precision]["tf"]["rdtype"]

    out = convolve(inp, ker, padding=padding, precision=precision)
    assert out.dtype == out_dtype

def test_shape():
    """Test the output shape for all padding models and for even and odd
    kernel lengths"""
    #######################################
    # Even kernel length
    #######################################
    input_shape = [64, 16, 24, 100]
    kernel_length = 8
    inp = config.tf_rng.uniform(input_shape, dtype=tf.float32)
    ker = config.tf_rng.uniform([kernel_length], dtype=tf.float32)
    #########################
    # 'valid' padding
    #########################
    out = convolve(inp, ker, 'valid')
    out_shape = input_shape[:-1] + [input_shape[-1] - kernel_length + 1]
    assert out.shape == out_shape
    #########################
    # 'same' padding
    #########################
    out = convolve(inp, ker, 'same')
    out_shape = input_shape
    assert out.shape == out_shape
    #########################
    # 'full' padding
    #########################
    out = convolve(inp, ker, 'full')
    out_shape = input_shape[:-1] + [input_shape[-1] + kernel_length - 1]
    assert out.shape == out_shape
    #######################################
    # Odd kernel length
    #######################################
    input_shape = [64, 16, 24, 100]
    kernel_length = 5
    inp = config.tf_rng.uniform(input_shape, dtype=tf.float32)
    ker = config.tf_rng.uniform([kernel_length], dtype=tf.float32)
    #########################
    # 'valid' padding
    #########################
    out = convolve(inp, ker, 'valid')
    out_shape = input_shape[:-1] + [input_shape[-1] - kernel_length + 1]
    assert out.shape == out_shape
    #########################
    # 'same' padding
    #########################
    out = convolve(inp, ker, 'same')
    out_shape = input_shape
    assert out.shape == out_shape
    #########################
    # 'full' padding
    #########################
    out = convolve(inp, ker, 'full')
    out_shape = input_shape[:-1] + [input_shape[-1] + kernel_length - 1]
    assert out.shape == out_shape

@pytest.mark.parametrize("inp_dtype", [tf.float64, tf.complex128])
@pytest.mark.parametrize("ker_dtype", [tf.float64, tf.complex128])
@pytest.mark.parametrize("padding", ['valid', 'same', 'full'])
@pytest.mark.parametrize("kernel_size", [1, 2, 5, 8, 100])
def test_computation(inp_dtype, ker_dtype, padding, kernel_size):
    "Test the convolution calculation against the one of np.convolve()"
    input_length = 100
    if inp_dtype.is_complex:
        inp = tf.complex(config.tf_rng.uniform([1, input_length],
                            dtype=inp_dtype.real_dtype),
                            config.tf_rng.uniform([1, input_length],
                            dtype=inp_dtype.real_dtype))
    else:
        inp = config.tf_rng.uniform([1, input_length],
                                dtype=inp_dtype)
    if ker_dtype.is_complex:
        ker = tf.complex(config.tf_rng.uniform([kernel_size],
                            dtype=ker_dtype.real_dtype),
                            config.tf_rng.uniform([kernel_size],
                            dtype=ker_dtype.real_dtype))
    else:
        ker = config.tf_rng.uniform([kernel_size], dtype=ker_dtype)
    #
    out = convolve(inp, ker, padding, precision="double")
    out_ref = np.convolve(inp.numpy()[0], ker.numpy(),
                            mode=padding)
    max_err = np.max(np.abs(out.numpy()[0] - out_ref))
    assert max_err <= 1e-10
