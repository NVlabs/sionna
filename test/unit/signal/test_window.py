#
# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0#
import pytest
import numpy as np
import tensorflow as tf
from sionna.phy import dtypes, config
from sionna.phy.signal import CustomWindow

@pytest.mark.parametrize("inp_dtype", [tf.float32, tf.complex64])
@pytest.mark.parametrize("win_dtype", [tf.float32, tf.float64])
@pytest.mark.parametrize("precision", ["single", "double"])
def test_dtype(inp_dtype, win_dtype, precision):
    """Test the output dtype for all possible combinations of
    input and window dtypes"""
    if inp_dtype.is_complex:
        inp = tf.complex(config.tf_rng.uniform([64, 100],
                            dtype=inp_dtype.real_dtype),
                            config.tf_rng.uniform([64, 100],
                            dtype=inp_dtype.real_dtype))
    else:
        inp = config.tf_rng.uniform([64, 100], dtype=inp_dtype)
    win_coeff = config.tf_rng.uniform([100], dtype=win_dtype)

    window = CustomWindow(coefficients=win_coeff, precision=precision)
    out = window(inp)

    if inp_dtype.is_complex:
        out_dtype = dtypes[precision]["tf"]["cdtype"]
    else:
        out_dtype = dtypes[precision]["tf"]["rdtype"]
    assert out.dtype == out_dtype

def test_shape():
    """Test the output shape"""
    input_shape = [64, 16, 24, 100]
    window_length = input_shape[-1]
    inp = config.tf_rng.uniform(input_shape)
    win_coeff = config.tf_rng.uniform([window_length], dtype=tf.float32)
    window = CustomWindow(win_coeff)
    out = window(inp)
    assert out.shape == input_shape

@pytest.mark.parametrize("inp_dtype", [tf.float64, tf.complex128])
@pytest.mark.parametrize("ker_dtype", [tf.float64, tf.float64])
def test_computation(inp_dtype, ker_dtype):
    "Test the calculation"
    precision = "double"
    batch_size = 64
    input_length = 100
    if inp_dtype.is_complex:
        inp = tf.complex(config.tf_rng.uniform([batch_size, input_length],
                            dtype=inp_dtype.real_dtype),
                            config.tf_rng.uniform([batch_size, input_length],
                            dtype=inp_dtype.real_dtype))
    else:
        inp = config.tf_rng.uniform([1, input_length],
                                dtype=inp_dtype)
    coefficients = config.tf_rng.uniform([input_length], dtype=ker_dtype)
    window = CustomWindow(coefficients=coefficients, precision=precision)
    out = window(inp)
    out_ref = np.expand_dims(coefficients, 0)*inp
    max_err = np.max(np.abs(out.numpy() - out_ref))
    assert max_err <= 1e-10

def test_normalization():
    "Test the normalization"
    win_length = 128
    coeff = config.tf_rng.uniform([win_length], dtype=tf.float64)
    window = CustomWindow(coeff, normalize=True, dtype=tf.float64)
    inp = tf.ones_like(coeff)
    out = window(inp)
    assert np.abs(np.mean(np.abs(out)**2)-1) <= 1e-6

def test_trainable():
    "Test gradient computation"
    config.precision = "single"
    batch_size = 64
    win_length = 128
    inp = tf.complex(config.tf_rng.uniform([batch_size, win_length],
                        dtype=tf.float32),
                        config.tf_rng.uniform([batch_size, win_length],
                        dtype=tf.float32))
    coeff = config.tf_rng.uniform([win_length], dtype=tf.float32)
    # Trainable
    window = CustomWindow(tf.Variable(coeff, trainable=True))
    with tf.GradientTape() as tape:
        out = window(inp)
        loss = tf.reduce_mean(tf.square(tf.abs(out))) # Meaningless loss
    grad = tape.gradient(loss, tape.watched_variables())
    assert len(grad) == 1
    assert grad[0].shape == [win_length]

    # Trainable off
    window = CustomWindow(coeff)
    with tf.GradientTape() as tape:
        out = window(inp)
        loss = tf.reduce_mean(tf.square(tf.abs(out))) # Meaningless loss
    grad = tape.gradient(loss, tape.watched_variables())
    assert len(grad) == 0
