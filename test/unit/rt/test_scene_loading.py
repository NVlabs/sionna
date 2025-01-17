#
# SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

try:
    import sionna
except ImportError as e:
    import sys
    sys.path.append("..")
    import sionna

import unittest
import numpy as np
import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
print('Number of GPUs available :', len(gpus))
if gpus:
    gpu_num = 0
    try:
        tf.config.set_visible_devices(gpus[gpu_num], 'GPU')
        print('Only GPU number', gpu_num, 'used.')
        tf.config.experimental.set_memory_growth(gpus[gpu_num], True)
    except RuntimeError as e:
        print(e)

from sionna.rt import load_scene

class TestSingleReflectionWithoutLoS(unittest.TestCase):
    def test_scene_objects_id(self):
        """Test that loading the same scene multiple times always results in
        the scene objects having consistent IDs across loadings.
        """

        scene = load_scene(sionna.rt.scene.munich)
        name_2_id_1 = {name : obj.object_id for name, obj in scene.objects.items()}
        all_ids = list(name_2_id_1.values())

        # Check that the lowest id is 0
        self.assertTrue(min(all_ids) == 0)

        # Check that the highest id equal the number of objects minus one
        self.assertTrue(max(all_ids) == (len(all_ids) - 1))

        # Check that their are no duplicated ids
        self.assertTrue(len(all_ids) == len(set(all_ids)))

        for _ in range(3):
            # Load integrated scene
            scene = load_scene(sionna.rt.scene.munich)
            name_2_id = {name : obj.object_id for name, obj in scene.objects.items()}

            self.assertTrue(name_2_id_1 == name_2_id)
