#
# SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#



import unittest
import numpy as np
import tensorflow as tf
import sionna
from sionna.rt import *
from sionna.constants import SPEED_OF_LIGHT, PI


class TestObjectPosition(unittest.TestCase):
    """Tests related to the change of an object's position"""

    def test_change_position_with_dtype(self):
        """Changing the position works in all dtypes"""
        for dtype in [tf.complex64, tf.complex128]:
            scene = load_scene(sionna.rt.scene.simple_street_canyon, dtype=dtype)
            obj = scene.get("building_3")
            target_position = tf.cast([12., 0.5, -3.], dtype.real_dtype)
            obj.position = target_position
            self.assertEqual(obj.position.dtype, dtype.real_dtype)
            self.assertTrue(np.array_equal(obj.position, target_position))
    
    def test_change_position_via_ray(self):
        """Modifying a position leads to the desired result"""
        scene = load_scene(sionna.rt.scene.simple_reflector)
        scene.tx_array = PlanarArray(1,1,0.5,0.5,"iso","V")
        scene.rx_array = PlanarArray(1,1,0.5,0.5,"iso","V")

        d0 = 100
        scene.add(Transmitter("tx", position=[0.1,0,d0]))
        scene.add(Receiver("rx", position=[0.1,0,d0]))
        paths = scene.compute_paths(los=False, diffraction=False, max_depth=1)
        paths.normalize_delays = False
        tau = tf.squeeze(paths.cir()[1])
        d1 = SPEED_OF_LIGHT*tau/2
        self.assertEqual(d1,d0)

        d2 = 30
        scene.get("reflector").position += [0,0,d0-d2]
        paths = scene.compute_paths(los=False, diffraction=False, max_depth=1)
        paths.normalize_delays = False
        tau = tf.squeeze(paths.cir()[1])
        d3 = SPEED_OF_LIGHT*tau/2
        self.assertEqual(d2,d3)

class TestObjectOrientation(unittest.TestCase):
    """Tests related to the change of an object's orientation"""

    def test_change_orientation_with_dtype(self):
        """Changing the orientation works in all dtypes"""
        for dtype in [tf.complex64, tf.complex128]:
            scene = load_scene(sionna.rt.scene.simple_street_canyon, dtype=dtype)
            obj = scene.get("building_3")
            target_orientation = tf.cast([-PI/3, 0.1, PI/2], dtype.real_dtype)
            obj.orientation = target_orientation
            self.assertEqual(obj.orientation.dtype, dtype.real_dtype)
            self.assertTrue(np.array_equal(obj.orientation, target_orientation))
    
    def test_no_position_change(self):
        """Changing orientation should not change the position"""
        for dtype in [tf.complex64, tf.complex128]:
            scene = load_scene(sionna.rt.scene.simple_street_canyon, dtype=dtype)
            obj = scene.get("building_3")
            pos_org = obj.position
            obj.orientation = [0.2,0.3,-0.4]
            self.assertAlmostEqual(np.sum(obj.position-pos_org), 0, 5)

    def test_orientation_impacts_paths(self):
        """Test showing that rotating a simple reflector can make a paths dissappear"""
        scene = load_scene(sionna.rt.scene.simple_reflector)
        scene.tx_array = PlanarArray(1,1,0.5,0.5,"iso","V")
        scene.rx_array = PlanarArray(1,1,0.5,0.5,"iso","V")
        scene.add(Transmitter("tx", position=[0.1,0,100]))
        scene.add(Receiver("rx", position=[0.1,0,100]))
        
        # There should be a single reflected path
        paths = scene.compute_paths(los=False, diffraction=False, max_depth=1)
        self.assertEqual(tf.squeeze(paths.tau).shape, [])
        
        # Rotating the reflector by PI/4 should make the path dissappear
        scene.get("reflector").orientation = [0,0,PI/4]
        paths = scene.compute_paths(los=False, diffraction=False, max_depth=1)
        self.assertEqual(tf.squeeze(paths.tau).shape, [0])