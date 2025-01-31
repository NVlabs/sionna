# SPDX-FileCopyrightText: Copyright (c) 2024 ORANGE - Author: Guillaume Larue <guillaume.larue@orange.com>. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

try:
    import sionna
except ImportError as e:
    import sys
    sys.path.append("..")
    import sionna

import pytest
import unittest
import numpy as np
import tensorflow as tf
import xml.etree.ElementTree as ET
import sys

from sionna.rt import *
from sionna.constants import SPEED_OF_LIGHT, PI

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

class TestAssetVelocity(unittest.TestCase):
    """Tests related to the change of an asset's velocity"""

    def test_change_velocity_with_dtype(self):
        """Changing the velocity works in all dtypes"""
        for dtype in [tf.complex64, tf.complex128]:
            scene = load_scene(sionna.rt.scene.floor_wall, dtype=dtype)
            asset = AssetObject(name="asset_0", filename=sionna.rt.asset_object.test_asset_3, radio_material='itu_metal')
            scene.add(asset)
            target_velocity = tf.cast([12., 0.5, -3.], dtype.real_dtype)
            asset.velocity = target_velocity
            self.assertEqual(asset.velocity.dtype, dtype.real_dtype)
            self.assertTrue(np.array_equal(asset.velocity, target_velocity))

    def test_change_velocity_via_ray(self):
        """Test that moving reflector has the right Doppler shift"""
        scene = load_scene()
        asset = AssetObject(name="asset_0", filename=sionna.rt.asset_object.test_asset_3, radio_material='itu_metal')
        scene.add(asset)
        asset.velocity = [0, 0, -20]
        scene.tx_array = PlanarArray(1,1,0.5,0.5,"iso","V")
        scene.rx_array = scene.tx_array
        scene.add(Transmitter("tx", [-25,0.1,50]))
        scene.add(Receiver("rx",    [ 25,0.1,50]))      
        
        # Compute the reflected path
        paths = scene.compute_paths(max_depth=1, los=False)

        # Compute theoretical Doppler shift for this path
        theta_t = tf.squeeze(paths.theta_t)
        phi_t = tf.squeeze(paths.phi_t)
        k_0 = r_hat(theta_t, phi_t)
        theta_r = tf.squeeze(paths.theta_r)
        phi_r = tf.squeeze(paths.phi_r)
        k_1 = -r_hat(theta_r, phi_r)
        doppler_theo = np.sum((k_1-k_0)*asset.velocity)/scene.wavelength
        
        a = tf.squeeze(paths.doppler).numpy()
        b = doppler_theo.numpy()

        self.assertAlmostEqual(a,b,places=3)

    def test_asset_vs_object_velocities_consistency(self):
        """Check if the position of the asset is consistent with that of its shape constituents. Here we consider a composite asset made of 
        two 1x1x1m cubes, centered in (0,0,0), spaced by 1 m along the y-axis. Hence, the centers of the cubes are (0,-1,0) and (0,+1,0). """
        scene = load_scene(sionna.rt.scene.floor_wall)

        
        asset = AssetObject(name="asset_0", filename=sionna.rt.asset_object.test_asset_1) 
        scene.add(asset)

        asset = scene.get("asset_0")
        cube_0_object = scene.get("asset_0_cube_0")
        cube_1_object = scene.get("asset_0_cube_1")

        self.assertTrue(np.array_equal(asset.velocity,np.array([0,0,0]) ))
        self.assertTrue(np.array_equal(cube_0_object.velocity,np.array([0,0,0])))
        self.assertTrue(np.array_equal(cube_1_object.velocity,np.array([0,0,0])))

        # Change asset velocity
        random_velocity = np.random.random(3)
        asset.velocity = random_velocity

        epsilon = 1e-5 
        self.assertTrue(tf.reduce_all(tf.math.less_equal(tf.math.abs(asset.velocity-random_velocity),epsilon)))
        self.assertTrue(tf.reduce_all(tf.math.less_equal(tf.math.abs(cube_0_object.velocity-random_velocity),epsilon)))
        self.assertTrue(tf.reduce_all(tf.math.less_equal(tf.math.abs(cube_1_object.velocity-random_velocity),epsilon)))


        # Change asset's scene object velocity
        random_velocity_2 = np.random.random(3)
        cube_0_object.velocity = random_velocity_2

        self.assertEqual(asset.velocity,None)
        epsilon = 1e-5 
        self.assertTrue(tf.reduce_all(tf.math.less_equal(tf.math.abs(cube_0_object.velocity-random_velocity_2),epsilon)))
        self.assertTrue(tf.reduce_all(tf.math.less_equal(tf.math.abs(cube_1_object.velocity-random_velocity),epsilon)))

    def test_velocity_without_scene(self):
        """Check that velocity is properly set even if the scene is not set yet"""
        epsilon = 1e-5
        asset = AssetObject(name="asset_0", filename=sionna.rt.asset_object.test_asset_1)
        self.assertTrue(tf.reduce_all(tf.math.less_equal(tf.math.abs(asset.velocity - [0,0,0]),epsilon)))

        asset.velocity = [4,3,2]
        self.assertTrue(tf.reduce_all(tf.math.less_equal(tf.math.abs(asset.velocity - [4,3,2]),epsilon)))

        scene = load_scene()
        scene.add(asset)
        scene_asset = scene.get("asset_0")
        cube_0_object = scene.get("asset_0_cube_0")
        cube_1_object = scene.get("asset_0_cube_1")
        self.assertTrue(tf.reduce_all(tf.math.less_equal(tf.math.abs(asset.velocity - [4,3,2]),epsilon)))
        self.assertTrue(tf.reduce_all(tf.math.less_equal(tf.math.abs(scene_asset.velocity -[4,3,2]),epsilon)))
        self.assertTrue(tf.reduce_all(tf.math.less_equal(tf.math.abs(cube_0_object.velocity - [4,3,2]),epsilon)))
        self.assertTrue(tf.reduce_all(tf.math.less_equal(tf.math.abs(cube_1_object.velocity - [4,3,2]),epsilon)))
