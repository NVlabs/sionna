#
# SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
try:
    import sionna
except ImportError as e:
    import sys

    sys.path.append("../")

import unittest
import numpy as np
import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
print('Number of GPUs available :', len(gpus))
if gpus:
    gpu_num = 0  # Number of the GPU to be used
    try:
        tf.config.set_visible_devices(gpus[gpu_num], 'GPU')
        print('Only GPU number', gpu_num, 'used.')
        tf.config.experimental.set_memory_growth(gpus[gpu_num], True)
    except RuntimeError as e:
        print(e)

from sionna.nr import PUSCHTransformPrecoder, PUSCHTransformDeprecoder


class TestPUSCHTransformPrecoder(unittest.TestCase):
    """Test PUSCHTransformPrecoder and PUSCHTransformDeprecoder"""

    def test_precoder_against_reference(self):
        for prbs in [2, 270]:
            ref_data = np.load(f"unit/nr/pusch_transform_precoding_{prbs}_prbs.npz")
            tp = PUSCHTransformPrecoder(num_subcarriers=12 * prbs)
            x_transform_precoded = tp(ref_data["x_layer_mapped"])
            np.testing.assert_array_almost_equal(x_transform_precoded,
                                                 ref_data["x_transform_precoded"])

    def test_deprecoder_against_reference(self):
        for prbs in [2, 270]:
            ref_data = np.load(f"unit/nr/pusch_transform_precoding_{prbs}_prbs.npz")
            tp = PUSCHTransformDeprecoder(num_subcarriers=12 * prbs)
            x_layer_mapped = tp(ref_data["x_transform_precoded"])
            np.testing.assert_array_almost_equal(x_layer_mapped,
                                                 ref_data["x_layer_mapped"])

    def test_invalid_subcarrier_count(self):
        with self.assertRaises(ValueError):
            PUSCHTransformPrecoder(num_subcarriers=273 * 12)
        with self.assertRaises(ValueError):
            PUSCHTransformDeprecoder(num_subcarriers=273 * 12)
