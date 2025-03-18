
#
# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0#

import unittest
import numpy as np
import tensorflow as tf

from sionna.phy.mimo.precoding import grid_of_beams_dft, \
    rzf_precoder
from sionna.phy import config, Block


class TestPrecoders(unittest.TestCase):

    def test_rzf_precoder(self):
        """
        Verify that the RZF precoder diagonalizes the channel without regularization
        """
        class RZF(Block):
            def __init__(self, K, M):
                super().__init__()
                dtype = tf.float64

                # generate channel matrix
                h_r = config.tf_rng.uniform(shape=[K, M], dtype=dtype)
                h_i = config.tf_rng.uniform(shape=[K, M], dtype=dtype)
                self.h = tf.complex(h_r, h_i)

                # generate signal
                x_r = config.tf_rng.uniform(shape=[K], dtype=dtype)
                x_i = config.tf_rng.uniform(shape=[K], dtype=dtype)
                self.x = tf.complex(x_r, x_i)

            def precode(self):
                # precode signal and retrieve the precoder matrix
                _, precoder = rzf_precoder(self.x,
                                           self.h,
                                           return_precoding_matrix=True,
                                           precision="double")
                return precoder

            @tf.function(jit_compile=False)
            def precode_graph(self):
                return self.precode()

            @tf.function(jit_compile=True)
            def precode_xla(self):
                return self.precode()

            def call(self, mod):
                if mod=='eager':
                    return self.precode()
                elif mod=='graph':
                    return self.precode_graph()
                elif mod=='xla':
                    return self.precode_xla()
                else:
                    raise ValueError("input 'mod' not recognized")

        # Verify that the composite channel H*precoder is diagonal
        K, M = 10, 15
        rzf = RZF(K, M)
        for mod in ['eager', 'graph', 'xla']:
            precoder = rzf(mod)
            h_precoder = tf.matmul(rzf.h, precoder)
            h_precoder_minus_diag = h_precoder - tf.linalg.diag(tf.linalg.diag_part(h_precoder))
            self.assertAlmostEqual(tf.abs(tf.norm(h_precoder_minus_diag)).numpy(), 0)

    def test_gob_orthogonality(self):
        """test that beams in the DFT GoB are orthogonal."""

        num_rows = 2
        num_cols = 6
        
        # compute a Grid of Beams (GoB) of Discrete Fourier Transform (DFT) beams
        gob = grid_of_beams_dft(num_rows, num_cols)

        # flatten the first two dimensions, accounting for the beam index pair
        gob1 = tf.reshape(gob, [num_rows*num_cols, num_rows*num_cols])
        
        # conjugate transpose
        gob1_h = tf.transpose(gob1, conjugate=True)

        # project rows (= beams) in pairwise fashion
        prod = abs(tf.linalg.matmul(gob1, gob1_h))
        
        # check that different rows (=beams) are orthogoanl
        success = ((prod.numpy() - np.eye(num_rows*num_cols)).sum() < 1e-3)
        self.assertTrue(success)

