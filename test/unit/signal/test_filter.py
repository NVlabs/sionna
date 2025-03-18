#
# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0#

import unittest
import pytest
import numpy as np
import tensorflow as tf
from sionna.phy import config, dtypes
from sionna.phy.signal import CustomFilter, CustomWindow

@pytest.mark.parametrize("inp_dtype", [tf.float32, tf.complex64])
@pytest.mark.parametrize("filt_dtype", [tf.float32, tf.complex64])
@pytest.mark.parametrize("padding", ["valid"])
@pytest.mark.parametrize("precision", ["single", "double"])
def test_dtype(inp_dtype, filt_dtype, padding, precision):
    """Test the output dtype for all possible combinations of
    input and window dtypes"""
    batch_size = 64
    inp_length = 1000
    span_in_symbols = 8
    samples_per_symbol = 4
    filter_length = samples_per_symbol*span_in_symbols+1

    # Generate inputs
    if inp_dtype.is_complex:
        inp = tf.complex(config.tf_rng.uniform([batch_size, inp_length],
                                                dtype=inp_dtype.real_dtype),
                        config.tf_rng.uniform([batch_size, inp_length],
                                                dtype=inp_dtype.real_dtype))
    else:
        inp = config.tf_rng.uniform([batch_size, inp_length], dtype=inp_dtype)

    # Generate filter coefficiens
    if filt_dtype.is_complex:
        coefficients = tf.complex(config.tf_rng.uniform([filter_length],
                                            dtype=filt_dtype.real_dtype),
                        config.tf_rng.uniform([filter_length],
                                                dtype=filt_dtype.real_dtype))
    else:
        coefficients = config.tf_rng.uniform([filter_length], dtype=filt_dtype)

    if inp_dtype.is_complex or filt_dtype.is_complex:
        out_dtype = dtypes[precision]["tf"]["cdtype"]
    else:
        out_dtype = dtypes[precision]["tf"]["rdtype"]

    # No windowing
    filt = CustomFilter(samples_per_symbol, coefficients, precision=precision)
    out = filt(inp, padding=padding)
    assert out.dtype == out_dtype

    # Windowing
    win_coeff = config.tf_rng.normal([filter_length], dtype=tf.float64)
    window = CustomWindow(coefficients=win_coeff, precision=precision)
    filt = CustomFilter(samples_per_symbol, coefficients, window=window, precision=precision)
    out = filt(inp, padding=padding)
    assert out.dtype == out_dtype

def test_shape():
    """Test the output shape"""
    input_shape = [16, 8, 24, 1000]
    inp = config.tf_rng.uniform(input_shape, dtype=tf.float32)

    for span_in_symbols in (7, 8):
        for samples_per_symbol in (1, 3, 4):
            filter_length = span_in_symbols*samples_per_symbol
            if (filter_length % 2) == 0:
                filter_length = filter_length + 1
            win_coeff = config.tf_rng.normal([filter_length], dtype=tf.float64)
            window = CustomWindow(coefficients=win_coeff)
            for win in (None, window):
                coefficients = config.tf_rng.uniform([filter_length], dtype=tf.float32)
                filt = CustomFilter(samples_per_symbol, coefficients=coefficients, window=win)
                #########################
                # 'valid' padding
                #########################
                out = filt(inp, 'valid')
                out_shape = input_shape[:-1] + [input_shape[-1] - filter_length + 1]
                assert out.shape == out_shape
                #########################
                # 'same' padding
                #########################
                out = filt(inp, 'same')
                out_shape = input_shape
                assert out.shape == out_shape

                #########################
                # 'full' padding
                #########################
                out = filt(inp, 'full')
                out_shape = input_shape[:-1] + [input_shape[-1] + filter_length - 1]
                assert out.shape == out_shape

@pytest.mark.parametrize("inp_dtype", [tf.float64, tf.complex128])
@pytest.mark.parametrize("fil_dtype", [tf.float64, tf.complex128])
@pytest.mark.parametrize("padding", ["valid", "same", "full"])
@pytest.mark.parametrize("span_in_symbols", [7, 8])
@pytest.mark.parametrize("samples_per_symbol", [1, 3, 4])
def test_computation(inp_dtype, fil_dtype, padding, span_in_symbols, samples_per_symbol):
    "Test the calculation"
    precision="double"
    input_length = 1000
    filter_length = span_in_symbols*samples_per_symbol
    if (filter_length % 2) == 0:
        filter_length = filter_length + 1
    win_coeff = config.tf_rng.normal([filter_length], dtype=tf.float64)
    window = CustomWindow(coefficients=win_coeff, precision=precision)
    for win in (None, window):
        if inp_dtype.is_complex:
            inp = tf.complex(config.tf_rng.uniform([1, input_length],
                                dtype=inp_dtype.real_dtype),
                            config.tf_rng.uniform([1, input_length],
                                dtype=inp_dtype.real_dtype))
        else:
            inp = config.tf_rng.uniform([1, input_length],
                                    dtype=inp_dtype)
        if fil_dtype.is_complex:
            fil_coeff = tf.complex(  config.tf_rng.uniform([filter_length],
                                        dtype=fil_dtype.real_dtype),
                                    config.tf_rng.uniform([filter_length],
                                        dtype=fil_dtype.real_dtype))
        else:
            fil_coeff = config.tf_rng.uniform([filter_length], dtype=fil_dtype)
        filt = CustomFilter(samples_per_symbol, coefficients=fil_coeff, window=win, normalize=False, precision=precision)

        # No conjugate
        out = filt(inp, padding, conjugate=False)
        ker_ref = fil_coeff.numpy()*win_coeff.numpy() if win else fil_coeff.numpy()
        out_ref = np.convolve(inp.numpy()[0], ker_ref, mode=padding)
        max_err = np.max(np.abs(out.numpy()[0] - out_ref))
        assert max_err <= 1e-10

        # Conjugate
        out = filt(inp, padding, conjugate=True)
        ker_ref = np.conj(ker_ref)
        out_ref = np.convolve(inp.numpy()[0], ker_ref, mode=padding)
        max_err = np.max(np.abs(out.numpy()[0] - out_ref))
        assert max_err <= 1e-10

@pytest.mark.parametrize("fil_dtype", [tf.float64, tf.complex128])
def test_normalization(fil_dtype):
    "Test the normalization"
    span_in_symbols = 8
    samples_per_symbol = 4
    filter_length = samples_per_symbol*span_in_symbols+1
    if fil_dtype == tf.float64:
        precision = "single"
    else:
        precision = "double"
    if fil_dtype.is_complex:
        fil_coeff = tf.complex(  config.tf_rng.uniform([filter_length],
                                    dtype=fil_dtype.real_dtype),
                                    config.tf_rng.uniform([filter_length],
                                    dtype=fil_dtype.real_dtype))
    else:
        fil_coeff = config.tf_rng.uniform([filter_length], dtype=fil_dtype)
    filt = CustomFilter(samples_per_symbol, coefficients=fil_coeff, normalize=True, precision=precision)

    x = tf.cast([1], tf.float64)

    assert np.abs(tf.reduce_sum(tf.abs(filt(x))**2)-1) < 1e-5

@pytest.mark.parametrize("fil_dtype", [tf.float64, tf.complex128])
def test_conjugate_filter(fil_dtype):
    span_in_symbols = 8
    samples_per_symbol = 4
    filter_length = samples_per_symbol*span_in_symbols+1
    if fil_dtype == tf.float64:
        precision = "single"
    else:
        precision = "double"
    if fil_dtype.is_complex:
        fil_coeff = tf.complex(  config.tf_rng.uniform([filter_length],
                                    dtype=fil_dtype.real_dtype),
                                    config.tf_rng.uniform([filter_length],
                                    dtype=fil_dtype.real_dtype))
    else:
        fil_coeff = config.tf_rng.uniform([filter_length], dtype=fil_dtype)
    filt = CustomFilter(samples_per_symbol, coefficients=fil_coeff, normalize=False, precision=precision)

    x = tf.cast([1], tf.float64)
    f = filt(x, conjugate=True)
    assert np.min(np.abs(np.conj(filt.coefficients) - f))< 1e-10

    f = filt(x, conjugate=False)
    assert np.min(np.abs(filt.coefficients - f))< 1e-10

@pytest.mark.parametrize("fil_dtype", [tf.float64, tf.complex128])
def test_variable_coefficients(fil_dtype):
    "Test gradient computation if coefficients are variables"
    batch_size = 64
    span_in_symbols = 8
    samples_per_symbol = 4
    filter_length = samples_per_symbol*span_in_symbols+1
    input_length = 1024
    inp = tf.complex(config.tf_rng.uniform([batch_size, input_length],
                        dtype=fil_dtype.real_dtype),
                        config.tf_rng.uniform([batch_size, input_length],
                        dtype=fil_dtype.real_dtype))
    if fil_dtype.is_complex:
        fil_coeff = tf.complex(  config.tf_rng.uniform([filter_length],
                                    dtype=fil_dtype.real_dtype),
                                    config.tf_rng.uniform([filter_length],
                                    dtype=fil_dtype.real_dtype))
    else:
        fil_coeff = config.tf_rng.uniform([filter_length], dtype=fil_dtype)

    # Trainable
    filt = CustomFilter(samples_per_symbol, coefficients=tf.Variable(fil_coeff, trainable=True), precision="double")
    with tf.GradientTape() as tape:
        out = filt(inp)
        loss = tf.reduce_mean(tf.square(tf.abs(out))) # Meaningless loss
    grad = tape.gradient(loss, tape.watched_variables())
    assert len(grad) == 1
    assert grad[0].shape == [filter_length]
    if fil_dtype.is_floating:
        assert np.sum(grad[0].numpy().real) != 0
        assert np.sum(grad[0].numpy().imag) == 0
    if fil_dtype.is_complex:
        assert np.sum(grad[0].numpy().real) != 0
        assert np.sum(grad[0].numpy().imag) != 0

    # Not trainable 
    filt = CustomFilter(samples_per_symbol, coefficients=fil_coeff, precision="double")
    with tf.GradientTape() as tape:
        out = filt(inp)
        loss = tf.reduce_mean(tf.square(tf.abs(out))) # Meaningless loss
    grad = tape.gradient(loss, tape.watched_variables())
    assert len(grad) == 0

@pytest.mark.parametrize("fil_dtype", [tf.float64, tf.complex128])
def test_aclr_diff(fil_dtype):
    "Test if ACLR computation is differentiable"
    span_in_symbols = 8
    samples_per_symbol = 4
    filter_length = samples_per_symbol*span_in_symbols+1
    if fil_dtype.is_complex:
        fil_coeff = tf.complex(config.tf_rng.uniform([filter_length],
                                    dtype=fil_dtype.real_dtype),
                               config.tf_rng.uniform([filter_length],
                                    dtype=fil_dtype.real_dtype))
    else:
        fil_coeff = config.tf_rng.uniform([filter_length], dtype=fil_dtype)

    fil_coeff = tf.Variable(fil_coeff, trainable=True)

    filt = CustomFilter(samples_per_symbol, coefficients=fil_coeff, precision="double")

    with tf.GradientTape() as tape:
        out = filt.aclr
        loss = tf.reduce_mean(tf.square(tf.abs(out)))
    grad = tape.gradient(loss, tape.watched_variables())
    assert len(grad) == 1
    assert grad[0].shape == [filter_length]
    if fil_dtype.is_floating:
        assert np.sum(grad[0].numpy().real) != 0
        assert np.sum(grad[0].numpy().imag) == 0
    if fil_dtype.is_complex:
        assert np.sum(grad[0].numpy().real) != 0
        assert np.sum(grad[0].numpy().imag) != 0