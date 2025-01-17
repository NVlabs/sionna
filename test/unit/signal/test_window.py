#
# SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import unittest
import numpy as np
import tensorflow as tf
from sionna import config
from sionna.signal import CustomWindow


class TestWindow(unittest.TestCase):

    def test_dtype(self):
        """Test the output dtype for all possible combinations of
        input and window dtypes"""
        # Map all possible combinations of (input, window) dtypes to
        # the expected output dtype
        expected_type = {(tf.float32, tf.float32) : tf.float32,
                         (tf.complex64, tf.float32) : tf.complex64,
                         (tf.float64, tf.float64) : tf.float64,
                         (tf.complex128, tf.float64) : tf.complex128}
        for dtypes in expected_type:
            inp_dtype = dtypes[0]
            win_dtype = dtypes[1]
            if inp_dtype.is_complex:
                inp = tf.complex(config.tf_rng.uniform([64, 100],
                                    dtype=inp_dtype.real_dtype),
                                 config.tf_rng.uniform([64, 100],
                                    dtype=inp_dtype.real_dtype))
            else:
                inp = config.tf_rng.uniform([64, 100], dtype=inp_dtype)

            window = CustomWindow(length=100, dtype=win_dtype)
            out = window(inp)
            self.assertEqual(out.dtype, expected_type[dtypes])

    def test_shape(self):
        """Test the output shape"""
        input_shape = [64, 16, 24, 100]
        window_length = input_shape[-1]
        inp = config.tf_rng.uniform(input_shape)
        window = CustomWindow(length=window_length)
        out = window(inp)
        self.assertEqual(out.shape, input_shape)

    def test_computation(self):
        "Test the calculation"
        # Focus on high precision as this is what numpy uses
        tested_dtypes = ((tf.float64, tf.float64),
                         (tf.complex128, tf.float64))
        batch_size = 64
        input_length = 100
        for dtypes in tested_dtypes:
            inp_dtype = dtypes[0]
            ker_dtype = dtypes[1]
            if inp_dtype.is_complex:
                inp = tf.complex(config.tf_rng.uniform([batch_size, input_length],
                                    dtype=inp_dtype.real_dtype),
                                 config.tf_rng.uniform([batch_size, input_length],
                                    dtype=inp_dtype.real_dtype))
            else:
                inp = config.tf_rng.uniform([1, input_length],
                                        dtype=inp_dtype)
            coefficients = config.tf_rng.uniform([input_length], dtype=ker_dtype)
            window = CustomWindow(input_length, coefficients=coefficients, dtype=ker_dtype)
            out = window(inp)
            out_ref = np.expand_dims(coefficients, 0)*inp
            max_err = np.max(np.abs(out.numpy() - out_ref))
            self.assertLessEqual(max_err, 1e-10)

    def test_normalization(self):
        "Test the normalization"
        win_length = 128
        coeff = config.tf_rng.uniform([win_length], dtype=tf.float64)
        window = CustomWindow(win_length, coeff, normalize=True, dtype=tf.float64)
        norm_coeff = coeff/tf.cast(tf.sqrt(tf.reduce_mean(tf.square(tf.abs(coeff)))), tf.float64)
        max_err = np.max(np.abs(norm_coeff.numpy() - window.coefficients.numpy()))
        self.assertLessEqual(max_err, 1e-10)

    def test_trainable(self):
        "Test gradient computation"
        batch_size = 64
        win_length = 128
        inp = tf.complex(config.tf_rng.uniform([batch_size, win_length],
                            dtype=tf.float32),
                            config.tf_rng.uniform([batch_size, win_length],
                            dtype=tf.float32))
        coeff = config.tf_rng.uniform([win_length], dtype=tf.float32)
        #########################
        # Trainable on
        window = CustomWindow(win_length, trainable=True, dtype=tf.float32)
        with tf.GradientTape() as tape:
            out = window(inp)
            loss = tf.reduce_mean(tf.square(tf.abs(out))) # Meaningless loss
        grad = tape.gradient(loss, tape.watched_variables())
        self.assertTrue(len(grad) == 1)
        self.assertTrue(grad[0].shape == [win_length])
        #########################
        # Trainable off
        window = CustomWindow(win_length, trainable=False, dtype=tf.float32)
        with tf.GradientTape() as tape:
            out = window(inp)
            loss = tf.reduce_mean(tf.square(tf.abs(out))) # Meaningless loss
        grad = tape.gradient(loss, tape.watched_variables())
        self.assertTrue(len(grad) == 0)
