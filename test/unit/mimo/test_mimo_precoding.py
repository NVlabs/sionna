
#
# SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

try:
    import sionna
except ImportError as e:
    import sys
    sys.path.append("..")  # sys.path.append(".")
    import sionna

import unittest
import numpy as np
import tensorflow as tf

from sionna.mimo.precoding import normalize_precoding_power, grid_of_beams_dft


gpus = tf.config.list_physical_devices('GPU')
if gpus:
    gpu_num = 0
    try:
        tf.config.set_visible_devices(gpus[gpu_num], 'GPU')
        print('Only GPU number', gpu_num, 'used.')
        tf.config.experimental.set_memory_growth(gpus[gpu_num], True)
    except RuntimeError as e:
        print(e)


class TestDFT(unittest.TestCase):

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

# if __name__ == '__main__':
#     unittest.main()
