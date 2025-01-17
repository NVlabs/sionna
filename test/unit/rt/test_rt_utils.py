#
# SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import unittest
import numpy as np
import tensorflow as tf
from sionna.rt import sample_points_on_hemisphere, dot

class TestSampleHemisphere(unittest.TestCase):

    def test_01(self):
        """Test that the points on the hemispheres have
        a positive dot product with the normal vectors"""
        dtype = tf.float32
        num_samples = 10000
        normals = tf.constant([
                       [0,0,1],
                       [0,0,-1],
                       [1,0,0],
                       [-1,0,0],
                       [0,1,0],
                       [0,-1,0],
                       [1/np.sqrt(2), 0, 1/np.sqrt(2)],
                       [1/np.sqrt(2), -1/np.sqrt(2), 0]
                      ], dtype=dtype)
        points = sample_points_on_hemisphere(normals, num_samples)
        self.assertFalse(tf.reduce_any(dot(tf.expand_dims(normals, axis=1),
                                           points)<0))

    def test_02(self):
        """Test output shape for num_samples=1"""
        dtype = tf.float32
        num_samples = 1
        normals = tf.constant([
                       [0,0,1],
                       [0,0,-1]
                      ], dtype=dtype)
        points = sample_points_on_hemisphere(normals, num_samples)
        self.assertTrue(np.array_equal(points.shape, [2, 3]))

        normals = tf.constant([
                       [0,0,1]
                      ], dtype=dtype)
        points = sample_points_on_hemisphere(normals, num_samples)
        self.assertTrue(np.array_equal(points.shape, [1, 3]))
