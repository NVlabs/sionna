#
# SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import unittest
import numpy as np
import tensorflow as tf
from sionna import config
from sionna.signal import CustomFilter, CustomWindow


class TestFilter(unittest.TestCase):

    def test_dtype(self):
        """Test the output dtype for all possible combinations of
        input and window dtypes"""
        # Map all possible combinations of (input, filter) dtypes to
        # the expected output dtype
        expected_type = {(tf.float32, tf.float32) : tf.float32,
                         (tf.float32, tf.complex64) : tf.complex64,
                         (tf.complex64, tf.float32) : tf.complex64,
                         (tf.complex64, tf.complex64) : tf.complex64,
                         (tf.float64, tf.float64) : tf.float64,
                         (tf.float64, tf.complex128) : tf.complex128,
                         (tf.complex128, tf.float64) : tf.complex128,
                         (tf.complex128, tf.complex128) : tf.complex128}
        batch_size = 64
        inp_length = 1000
        span_in_symbols = 8
        samples_per_symbol = 4
        filter_length = samples_per_symbol*span_in_symbols+1
        for padding in ('valid', 'same', 'full'):
            for dtypes in expected_type:
                inp_dtype = dtypes[0]
                fil_dtype = dtypes[1]
                if inp_dtype.is_complex:
                    inp = tf.complex(config.tf_rng.uniform([batch_size, inp_length],
                                        dtype=inp_dtype.real_dtype),
                                     config.tf_rng.uniform([batch_size, inp_length],
                                        dtype=inp_dtype.real_dtype))
                else:
                    inp = config.tf_rng.uniform([batch_size, inp_length], dtype=inp_dtype)
                #########################
                # No windowing
                filt = CustomFilter(span_in_symbols, samples_per_symbol, dtype=fil_dtype)
                out = filt(inp, padding=padding)
                self.assertEqual(out.dtype, expected_type[dtypes])
                #########################
                # Windowing
                win_dtype = fil_dtype if fil_dtype.is_floating else fil_dtype.real_dtype
                window = CustomWindow(length=filter_length, dtype=win_dtype)
                filt = CustomFilter(span_in_symbols, samples_per_symbol, window=window, dtype=fil_dtype)
                out = filt(inp, padding=padding)
                self.assertEqual(out.dtype, expected_type[dtypes])

    def test_shape(self):
        """Test the output shape"""
        input_shape = [16, 8, 24, 1000]
        inp = config.tf_rng.uniform(input_shape, dtype=tf.float32)

        for span_in_symbols in (7, 8):
            for samples_per_symbol in (1, 3, 4):
                filter_length = span_in_symbols*samples_per_symbol
                if (filter_length % 2) == 0:
                    filter_length = filter_length + 1
                window = CustomWindow(length=filter_length, dtype=tf.float32)
                for win in (None, window):
                    filt = CustomFilter(span_in_symbols, samples_per_symbol, window=win, dtype=tf.float32)
                    #########################
                    # 'valid' padding
                    #########################
                    out = filt(inp, 'valid')
                    out_shape = input_shape[:-1] + [input_shape[-1] - filter_length + 1]
                    self.assertEqual(out.shape, out_shape)
                    #########################
                    # 'same' padding
                    #########################
                    out = filt(inp, 'same')
                    out_shape = input_shape
                    self.assertEqual(out.shape, out_shape)
                    #########################
                    # 'full' padding
                    #########################
                    out = filt(inp, 'full')
                    out_shape = input_shape[:-1] + [input_shape[-1] + filter_length - 1]
                    self.assertEqual(out.shape, out_shape)

    def test_computation(self):
        "Test the calculation"
        # Focus on high precision as this is what numpy uses
        tested_dtypes = ((tf.float64, tf.float64),
                         (tf.float64, tf.complex128),
                         (tf.complex128, tf.float64),
                         (tf.complex128, tf.complex128))
        input_length = 1000
        for dtypes in tested_dtypes:
            inp_dtype = dtypes[0]
            fil_dtype = dtypes[1]
            for span_in_symbols in (7, 8):
                for samples_per_symbol in (1, 3, 4):
                    filter_length = span_in_symbols*samples_per_symbol
                    if (filter_length % 2) == 0:
                        filter_length = filter_length + 1
                    win_coeff = config.tf_rng.normal([filter_length], dtype=tf.float64)
                    window = CustomWindow(length=filter_length, coefficients=win_coeff, dtype=tf.float64)
                    for win in (None, window):
                        for padding in ('valid', 'same', 'full'):
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
                            filt = CustomFilter(span_in_symbols, samples_per_symbol, coefficients=fil_coeff, window=win, normalize=False, dtype=fil_dtype)
                            # No conjugate
                            out = filt(inp, padding, conjugate=False)
                            ker_ref = fil_coeff.numpy()*win_coeff.numpy() if win else fil_coeff.numpy()
                            out_ref = np.convolve(inp.numpy()[0], ker_ref, mode=padding)
                            max_err = np.max(np.abs(out.numpy()[0] - out_ref))
                            self.assertLessEqual(max_err, 1e-10)
                            # Conjugate
                            out = filt(inp, padding, conjugate=True)
                            ker_ref = np.conj(ker_ref)
                            out_ref = np.convolve(inp.numpy()[0], ker_ref, mode=padding)
                            max_err = np.max(np.abs(out.numpy()[0] - out_ref))
                            self.assertLessEqual(max_err, 1e-10)

    def test_normalization(self):
        "Test the normalization"
        span_in_symbols = 8
        samples_per_symbol = 4
        filter_length = samples_per_symbol*span_in_symbols+1
        for fil_dtype in (tf.float64, tf.complex128):
            if fil_dtype.is_complex:
                fil_coeff = tf.complex(  config.tf_rng.uniform([filter_length],
                                            dtype=fil_dtype.real_dtype),
                                         config.tf_rng.uniform([filter_length],
                                            dtype=fil_dtype.real_dtype))
            else:
                fil_coeff = config.tf_rng.uniform([filter_length], dtype=fil_dtype)
            filt = CustomFilter(span_in_symbols, samples_per_symbol, coefficients=fil_coeff, window=None, normalize=True, dtype=fil_dtype)
            norm_fil_coeff_ref = fil_coeff/tf.cast(tf.sqrt(tf.reduce_sum(tf.square(tf.abs(fil_coeff)))), fil_dtype)
            max_err = np.max(np.abs(norm_fil_coeff_ref.numpy() - filt.coefficients.numpy()))
            self.assertLessEqual(max_err, 1e-10)

    def test_trainable(self):
        "Test gradient computation"
        batch_size = 64
        span_in_symbols = 8
        samples_per_symbol = 4
        filter_length = samples_per_symbol*span_in_symbols+1
        input_length = 1024
        for fil_dtype in (tf.float64, tf.complex128):
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
            #########################
            # Trainable on
            filt = CustomFilter(span_in_symbols, samples_per_symbol, coefficients=fil_coeff, window=None, trainable=True, dtype=fil_dtype)
            with tf.GradientTape() as tape:
                out = filt(inp)
                loss = tf.reduce_mean(tf.square(tf.abs(out))) # Meaningless loss
            grad = tape.gradient(loss, tape.watched_variables())
            if fil_dtype.is_floating:
                self.assertTrue(len(grad) == 1)
                self.assertTrue(grad[0].shape == [filter_length])
            elif fil_dtype.is_complex:
                self.assertTrue(len(grad) == 2)
                self.assertTrue(grad[0].shape == [filter_length])
                self.assertTrue(grad[1].shape == [filter_length])
            #########################
            # Trainable off
            filt = CustomFilter(span_in_symbols, samples_per_symbol, coefficients=fil_coeff, window=None, trainable=False, dtype=fil_dtype)
            with tf.GradientTape() as tape:
                out = filt(inp)
                loss = tf.reduce_mean(tf.square(tf.abs(out))) # Meaningless loss
            grad = tape.gradient(loss, tape.watched_variables())
            self.assertTrue(len(grad) == 0)

    def test_aclr_diff(self):
        "Test if ACLR computation is differentiable"
        batch_size = 64
        span_in_symbols = 8
        samples_per_symbol = 4
        filter_length = samples_per_symbol*span_in_symbols+1
        input_length = 1024
        for fil_dtype in (tf.float64, tf.complex128):
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
            #########################
            # Trainable on
            filt = CustomFilter(span_in_symbols, samples_per_symbol, coefficients=fil_coeff, window=None, trainable=True, dtype=fil_dtype)
            with tf.GradientTape() as tape:
                out = filt.aclr
                loss = tf.reduce_mean(tf.square(tf.abs(out))) # Meaningless loss
            grad = tape.gradient(loss, tape.watched_variables())
            if fil_dtype.is_floating:
                self.assertTrue(len(grad) == 1)
                self.assertTrue(grad[0].shape == [filter_length])
            elif fil_dtype.is_complex:
                self.assertTrue(len(grad) == 2)
                self.assertTrue(grad[0].shape == [filter_length])
                self.assertTrue(grad[1].shape == [filter_length])
