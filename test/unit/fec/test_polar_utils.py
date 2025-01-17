#
# SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import unittest
import numpy as np
import tensorflow as tf
from sionna.fec.polar.utils import generate_5g_ranking, generate_rm_code, generate_dense_polar
from sionna.fec.polar import PolarEncoder
from sionna.utils import BinarySource

class TestPolarUtils(unittest.TestCase):
    """Test polar utils.

    Remark: several 5G Polar code related tests can be found in
    test_polar_encoding.py"""

    def test_invalid_inputs(self):
        """Test against invalid values of n and k"""
        param_invalid = [[-1, 32],[10,-3],[1.0, 32],[3, 32.],[33,32], [10, 31],
                         [1025, 2048], [16, 33], [7, 16], [1000, 2048]] # (k,n)
        for p in param_invalid:
            with self.assertRaises(AssertionError):
                generate_5g_ranking(p[0],p[1])

        param_valid = [[1, 512],[10,32],[1000, 1024],[3, 256], [10,64], [0,32],
                       [1024,1024]] # (k,n)
        for p in param_valid:
            generate_5g_ranking(p[0],p[1])

    def test_generate_rm(self):
        """Test that Reed-Muller Code design yields valid constructions.

        We test against the parameters from
        https://en.wikipedia.org/wiki/Reed%E2%80%93Muller_code
        """

        # r, m, n, k, d_min
        param = [[0,0,1,1,1], [1,1,2,2,1], [2,2,4,4,1], [3,3,8,8,1],
                 [4,4,16,16,1], [5,5,32,32,1], [0,1,2,1,2], [1,2,4,3,2],
                 [2,3,8,7,2], [3,4,16,15,2], [4,5,32,31,2], [0,2,4,1,4],
                 [1,3,8,4,4], [2,4,16,11,4], [3,5,32,26,4], [0,3,8,1,8],
                 [1,4,16,5,8], [2,5,32,16,8], [0,4,16,1,16], [1,5,32,6,16],
                 [0,5,32,1,32]]

        for p in param:
            frozen_pos, info_pos, n, k , d_min = generate_rm_code(p[0], p[1])

            # check against correct parameters
            self.assertEqual(n, p[2])
            self.assertEqual(k, p[3])
            self.assertEqual(d_min, p[4])
            self.assertEqual(len(frozen_pos), n-k)
            self.assertEqual(len(info_pos), k)

    def test_generate_dense_polar(self):
        """test naive (dense) polar code construction method."""

        # sweep over many code parameters
        ns = [32, 64, 128, 256, 512, 1024]
        rates = [0.1, 0.5, 0.9]
        batch_size = 100
        source = BinarySource()

        for n in ns:
            for r in rates:
                k = int(n*r)

                frozen_pos, _ = generate_5g_ranking(k, n)

                # run method under test
                pcm, gm = generate_dense_polar(frozen_pos, n, verbose=False)

                # cast dtype to tf.float32
                gm = tf.cast(gm, dtype=tf.float32)
                pcm = tf.cast(pcm, dtype=tf.float32)

                # verify shapes
                self.assertTrue(pcm.shape[0]==n-k)
                self.assertTrue(pcm.shape[1]==n)
                self.assertTrue(gm.shape[0]==k)
                self.assertTrue(gm.shape[1]==n)

                # Verify that H*G has an all-zero syndrome.
                s = np.mod(np.matmul(pcm, np.transpose(gm)),2)
                self.assertTrue(np.sum(s)==0) # Non-zero syndrom for H*G'

                # compare against Sionna encoder
                encoder = PolarEncoder(frozen_pos, n)

                # draw random info bits
                u = source([batch_size, k])
                # and encode as reference
                c = encoder(u)

                # encode via generator matrix
                c_new = tf.matmul(tf.expand_dims(u, axis=1), gm)
                # account for GF(2) operations
                c_new = tf.math.mod(tf.squeeze(c_new, axis=1), 2)

                # both encodings must lead to same result
                self.assertTrue(np.array_equal(c, c_new.numpy()))

                # verify H, i.e., must lead to zero syndrome: Hc^t=0
                s = tf.matmul(pcm, tf.expand_dims(c, axis=2))
                # account for GF(2) operations
                s = tf.math.mod(tf.squeeze(s, axis=2), 2)
                self.assertTrue(np.array_equal(s.numpy(),
                                               np.zeros_like(s.numpy())))
