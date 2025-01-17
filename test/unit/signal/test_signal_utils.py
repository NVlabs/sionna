#
# SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import unittest
import numpy as np
import tensorflow as tf
from sionna import config
from sionna.signal import convolve

class TestConvolve(unittest.TestCase):

    def test_dtype(self):
        """Test the output dtype for all possible combinations of
        input dypes"""
        # Map all possible combinations of input and kernel dtypes to
        # the expected output dtype
        expected_type = {(tf.float32, tf.float32) : tf.float32,
                         (tf.float32, tf.complex64) : tf.complex64,
                         (tf.complex64, tf.float32) : tf.complex64,
                         (tf.complex64, tf.complex64) : tf.complex64,
                         (tf.float64, tf.float64) : tf.float64,
                         (tf.float64, tf.complex128) : tf.complex128,
                         (tf.complex128, tf.float64) : tf.complex128,
                         (tf.complex128, tf.complex128) : tf.complex128}
        for padding in ('valid', 'same', 'full'):
            for dtypes in expected_type:
                inp_dtype = dtypes[0]
                ker_dtype = dtypes[1]
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
                out = convolve(inp, ker, padding)
                self.assertEqual(out.dtype, expected_type[dtypes])

    def test_shape(self):
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
        self.assertEqual(out.shape, out_shape)
        #########################
        # 'same' padding
        #########################
        out = convolve(inp, ker, 'same')
        out_shape = input_shape
        self.assertEqual(out.shape, out_shape)
        #########################
        # 'full' padding
        #########################
        out = convolve(inp, ker, 'full')
        out_shape = input_shape[:-1] + [input_shape[-1] + kernel_length - 1]
        self.assertEqual(out.shape, out_shape)
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
        self.assertEqual(out.shape, out_shape)
        #########################
        # 'same' padding
        #########################
        out = convolve(inp, ker, 'same')
        out_shape = input_shape
        self.assertEqual(out.shape, out_shape)
        #########################
        # 'full' padding
        #########################
        out = convolve(inp, ker, 'full')
        out_shape = input_shape[:-1] + [input_shape[-1] + kernel_length - 1]
        self.assertEqual(out.shape, out_shape)

    def test_computation(self):
        "Test the convolution calculation against the one of np.convolve()"
        # Focus on high precision as this is what numpy uses
        tested_dtypes = ((tf.float64, tf.float64),
                         (tf.float64, tf.complex128),
                         (tf.complex128, tf.float64),
                         (tf.complex128, tf.complex128))
        input_length = 100
        for kernel_size in (1, 2, 5, 8, input_length):
            for padding in ('valid', 'same', 'full'):
                for dtypes in tested_dtypes:
                    inp_dtype = dtypes[0]
                    ker_dtype = dtypes[1]
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
                    out = convolve(inp, ker, padding)
                    out_ref = np.convolve(inp.numpy()[0], ker.numpy(),
                                            mode=padding)
                    max_err = np.max(np.abs(out.numpy()[0] - out_ref))
                    self.assertLessEqual(max_err, 1e-10)
