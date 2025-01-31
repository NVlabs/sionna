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
import sys

from sionna.rt import *
from sionna.constants import SPEED_OF_LIGHT

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
        
        
class TestAssetPosition(unittest.TestCase):
    """Tests related to the change of an asset's position"""

    def test_change_position_with_dtype(self):
        """Changing the position works in all dtypes"""
        for dtype in [tf.complex64, tf.complex128]:
            scene = load_scene(sionna.rt.scene.floor_wall, dtype=dtype)
            asset = AssetObject(name="asset_0", filename=sionna.rt.asset_object.test_asset_1)
            scene.add(asset)
            target_position = tf.cast([12., 0.5, -3.], dtype.real_dtype)
            asset.position = target_position
            self.assertEqual(asset.position.dtype, dtype.real_dtype)
            self.assertTrue(np.array_equal(asset.position, target_position))
    
    def test_change_position_via_ray(self):
        """Modifying a position leads to the desired result (in terms of propagation)"""
        scene = load_scene() # Load empty scene
        scene.tx_array = PlanarArray(1,1,0.5,0.5,"iso","V")
        scene.rx_array = PlanarArray(1,1,0.5,0.5,"iso","V")

        ## The test cube(s) are two 1x1x1m cubes, centered in (0,0,0), spaced by 1 m along the y-axis
        cube_edge_length = 1 
        cubes_separation = 1
        asset = AssetObject(name="reflector", filename=sionna.rt.asset_object.test_asset_1, position=[0,-cubes_separation,-cube_edge_length/2]) # we shift the asset so that the face of the metal cube is aligned with the xy-plane and center in (0,0)
        scene.add(asset)

        d0 = 100
        d1 = np.sqrt(d0*d0 + 1) #Hypothenuse of the TX (or RX) to reflector asset square triangle  
        scene.add(Transmitter("tx", position=[0,+1,d0]))
        scene.add(Receiver("rx", position=[0,-1,d0]))
        paths = scene.compute_paths(los=False, diffraction=False, max_depth=1, method="exhaustive")
        paths.normalize_delays = False
        cir = paths.cir()
        tau = tf.squeeze(cir[1])[1]
        d2 = SPEED_OF_LIGHT*tau/2

        epsilon = 1e-5 
        self.assertTrue(tf.reduce_all(tf.math.less_equal(tf.math.abs(d1 - d2),epsilon)))
        #self.assertEqual(d1,d2)

        d3 = 80 
        d4 = np.sqrt(d3*d3 + 1)
        asset.position += [0,0,(d0-d3)]
        paths = scene.compute_paths(los=False, diffraction=False, max_depth=1, method="exhaustive")
        paths.normalize_delays = False
        cir = paths.cir()
        tau = tf.squeeze(cir[1])[1]
        d5 = SPEED_OF_LIGHT*tau/2

        epsilon = 1e-5 
        self.assertTrue(tf.reduce_all(tf.math.less_equal(tf.math.abs(d4 - d5),epsilon)))
        #self.assertEqual(d4,d5)

    def test_asset_vs_object_position_consistency(self):
        """Check if the position of the asset is consistent with that of its shape constituents. Here we consider a composite asset made of 
        two 1x1x1m cubes, centered in (0,0,0), spaced by 1 m along the y-axis. Hence, the centers of the cubes are (0,-1,0) and (0,+1,0). """
        scene = load_scene(sionna.rt.scene.floor_wall)

        random_position = np.random.random(3)
        asset = AssetObject(name="asset_0", filename=sionna.rt.asset_object.test_asset_1, position=random_position) 
        scene.add(asset)

        asset = scene.get("asset_0")
        cube_0_object = scene.get("asset_0_cube_0")
        cube_1_object = scene.get("asset_0_cube_1")
        cube_0_position = np.array([0,-1,0]) + random_position 
        cube_1_position = np.array([0,+1,0]) + random_position

        self.assertTrue("asset_0" in scene.asset_objects)
        self.assertTrue(isinstance(asset,AssetObject)) 
        self.assertTrue(isinstance(cube_0_object,SceneObject)) 
        self.assertTrue(isinstance(cube_1_object,SceneObject)) 
        self.assertTrue(tf.reduce_all(asset.position==random_position))
        epsilon = 1e-5 # The Bounding-boxes computed by sionna to estimate the position of an object are not entirely accurate
        self.assertTrue(tf.reduce_all(tf.math.less_equal(tf.math.abs(cube_0_object.position-cube_0_position),epsilon)))
        self.assertTrue(tf.reduce_all(tf.math.less_equal(tf.math.abs(cube_1_object.position-cube_1_position),epsilon)))

        
    def test_init_bias_when_reloading_scene(self):
        """Check if the bias introduced at asset object init to avoid Mitsuba mixing up shapes at the same position, 
        is not added several time when reloading a scene."""
        
        scene = load_scene(sionna.rt.scene.floor_wall)

        asset_0 = AssetObject(name="asset_0", filename=sionna.rt.asset_object.test_asset_1) 
        scene.add(asset_0)

        cube_0_object = scene.get("asset_0_cube_0")
        cube_1_object = scene.get("asset_0_cube_1")

        asset_0_position_0 = tf.Variable(asset_0.position)
        cube_0_position_0 = tf.Variable(cube_0_object.position)
        cube_1_position_0 = tf.Variable(cube_1_object.position)

        # Adding a secondary asset to reload the scene
        asset_1 = AssetObject(name="asset_1", filename=sionna.rt.asset_object.test_asset_1) 
        scene.add(asset_1)

        asset_0_position_1 = tf.Variable(asset_0.position)
        cube_0_position_1 = tf.Variable(cube_0_object.position)
        cube_1_position_1 = tf.Variable(cube_1_object.position)

        # Or manually reload the scene
        scene.reload()

        asset_0_position_2 = tf.Variable(asset_0.position)
        cube_0_position_2 = tf.Variable(cube_0_object.position)
        cube_1_position_2 = tf.Variable(cube_1_object.position)

        self.assertTrue(tf.reduce_all((asset_0_position_0==asset_0_position_1) == (asset_0_position_0==asset_0_position_2)))
        self.assertTrue(tf.reduce_all((cube_0_position_0==cube_0_position_1) == (cube_0_position_0==cube_0_position_2)))
        self.assertTrue(tf.reduce_all((cube_1_position_0==cube_1_position_1) == (cube_1_position_0==cube_1_position_2)))


    def test_position_add_vs_set(self):
        """Check that position accumulation lead to the same result as setting the complete position at once for asset objects"""
        scene = load_scene()
        asset = AssetObject(name="asset_0", filename=sionna.rt.asset_object.test_asset_1)
        scene.add(asset)
        
        cube_0_object = scene.get("asset_0_cube_0")
        cube_1_object = scene.get("asset_0_cube_1")

        epsilon = 1e-5 # The Boudning-boxes computed by sionna to estimate the position of an object are not entirely accurate
        self.assertTrue(tf.reduce_all(tf.math.less_equal(tf.math.abs(cube_0_object.position-[0,-1,0]),epsilon)))
        self.assertTrue(tf.reduce_all(tf.math.less_equal(tf.math.abs(cube_1_object.position-[0,+1,0]),epsilon)))

        # Add position
        asset.position += [+2, -3, +5]
        self.assertTrue(tf.reduce_all(tf.math.less_equal(tf.math.abs(cube_0_object.position-[+2, -4, +5]),epsilon)))
        self.assertTrue(tf.reduce_all(tf.math.less_equal(tf.math.abs(cube_1_object.position-[+2, -2, +5]),epsilon)))

        # Add position
        asset.position += [-1, +1, -2]
        self.assertTrue(tf.reduce_all(tf.math.less_equal(tf.math.abs(cube_0_object.position-[+1, -3, +3]),epsilon)))
        self.assertTrue(tf.reduce_all(tf.math.less_equal(tf.math.abs(cube_1_object.position-[+1, -1, +3]),epsilon)))

        # Set position
        asset.position = [0, 0, 0]
        self.assertTrue(tf.reduce_all(tf.math.less_equal(tf.math.abs(cube_0_object.position-[0,-1,0]),epsilon)))
        self.assertTrue(tf.reduce_all(tf.math.less_equal(tf.math.abs(cube_1_object.position-[0,+1,0]),epsilon)))

        # Set position
        asset.position = [+1,-2,+3]
        self.assertTrue(tf.reduce_all(tf.math.less_equal(tf.math.abs(cube_0_object.position-[+1,-3,+3]),epsilon)))
        self.assertTrue(tf.reduce_all(tf.math.less_equal(tf.math.abs(cube_1_object.position-[+1,-1,+3]),epsilon)))

    def test_position_without_scene(self):
        """Check that position is properly set even if the scene is not set yet"""
        epsilon = 1e-5
        asset = AssetObject(name="asset_0", filename=sionna.rt.asset_object.test_asset_1)
        self.assertTrue(tf.reduce_all(tf.math.less_equal(tf.math.abs(asset.position - [0,0,0]),epsilon)))

        asset.position = [4,3,2]
        self.assertTrue(tf.reduce_all(tf.math.less_equal(tf.math.abs(asset.position - [4,3,2]),epsilon)))

        scene = load_scene()
        scene.add(asset)
        scene_asset = scene.get("asset_0")
        cube_0_object = scene.get("asset_0_cube_0")
        cube_1_object = scene.get("asset_0_cube_1")
        self.assertTrue(tf.reduce_all(tf.math.less_equal(tf.math.abs(asset.position - [4,3,2]),epsilon)))
        self.assertTrue(tf.reduce_all(tf.math.less_equal(tf.math.abs(scene_asset.position -[4,3,2]),epsilon)))
        self.assertTrue(tf.reduce_all(tf.math.less_equal(tf.math.abs(cube_0_object.position - [4,2,2]),epsilon)))
        self.assertTrue(tf.reduce_all(tf.math.less_equal(tf.math.abs(cube_1_object.position - [4,4,2]),epsilon)))
