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
from sionna.constants import PI

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
        
        
class TestAssetOrientation(unittest.TestCase):
    """Tests related to the change of an asset's orientation"""

    def test_change_orientation_with_dtype(self):
        """Changing the orientation works in all dtypes"""
        for dtype in [tf.complex64, tf.complex128]:
            scene = load_scene(sionna.rt.scene.floor_wall, dtype=dtype)
            asset = AssetObject(name="asset_0", filename=sionna.rt.asset_object.test_asset_1)
            scene.add(asset)
            target_orientation = tf.cast([-PI/3, 0.1, PI/2], dtype.real_dtype)
            asset.orientation = target_orientation
            self.assertEqual(asset.orientation.dtype, dtype.real_dtype)
            self.assertTrue(np.array_equal(asset.orientation, target_orientation))
    
    def test_no_position_change(self):
        """Changing orientation should not change the position of the asset"""
        scene = load_scene(sionna.rt.scene.floor_wall)
        asset = AssetObject(name="asset_0", filename=sionna.rt.asset_object.test_asset_1)
        scene.add(asset)
        pos_org = asset.position
        asset.orientation = [0.2,0.3,-0.4]
        self.assertTrue(tf.reduce_all(asset.position==pos_org))

    def test_orientation_axis_convention(self):
        """Check axis convention when rotating asset"""
        scene = load_scene()
        asset = AssetObject(name="asset_0", filename=sionna.rt.asset_object.test_asset_1)
        scene.add(asset)
        
        cube_0_object = scene.get("asset_0_cube_0")
        cube_1_object = scene.get("asset_0_cube_1")

        epsilon = 1e-5 # The Boudning-boxes computed by sionna to estimate the position of an object are not entirely accurate
        self.assertTrue(tf.reduce_all(tf.math.less_equal(tf.math.abs(cube_0_object.position-[0,-1,0]),epsilon)))
        self.assertTrue(tf.reduce_all(tf.math.less_equal(tf.math.abs(cube_1_object.position-[0,+1,0]),epsilon)))

        # Rotation around x-axis
        asset.orientation += [0, 0, PI / 2]
        self.assertTrue(tf.reduce_all(tf.math.less_equal(tf.math.abs(cube_0_object.position-[0,0,-1]),epsilon)))
        self.assertTrue(tf.reduce_all(tf.math.less_equal(tf.math.abs(cube_1_object.position-[0,0,+1]),epsilon)))

        # Rotation around y-axis
        asset.orientation += [0, PI / 2, 0]
        self.assertTrue(tf.reduce_all(tf.math.less_equal(tf.math.abs(cube_0_object.position-[-1,0,0]),epsilon)))
        self.assertTrue(tf.reduce_all(tf.math.less_equal(tf.math.abs(cube_1_object.position-[+1,0,0]),epsilon)))

        # Rotation around z-axis
        asset.orientation += [PI / 2, 0, 0]
        self.assertTrue(tf.reduce_all(tf.math.less_equal(tf.math.abs(cube_0_object.position-[0,-1,0]),epsilon)))
        self.assertTrue(tf.reduce_all(tf.math.less_equal(tf.math.abs(cube_1_object.position-[0,+1,0]),epsilon)))

    def test_set_orientation_before_adding_asset_to_scene(self):
        """Check orientation parameter works before adding asset to scene"""
        scene = load_scene()
        asset = AssetObject(name="asset_0", filename=sionna.rt.asset_object.test_asset_1)
        scene.add(asset)
        
        cube_0_object = scene.get("asset_0_cube_0")
        cube_1_object = scene.get("asset_0_cube_1")

        epsilon = 1e-5 # The Boudning-boxes computed by sionna to estimate the position of an object are not entirely accurate
        self.assertTrue(tf.reduce_all(tf.math.less_equal(tf.math.abs(cube_0_object.position-[0,-1,0]),epsilon)))
        self.assertTrue(tf.reduce_all(tf.math.less_equal(tf.math.abs(cube_1_object.position-[0,+1,0]),epsilon)))

        # Rotation around x-axis
        scene = load_scene()
        asset = AssetObject(name="asset_0", filename=sionna.rt.asset_object.test_asset_1,orientation=[0, 0, PI / 2])
        scene.add(asset)
        cube_0_object = scene.get("asset_0_cube_0")
        cube_1_object = scene.get("asset_0_cube_1")
        self.assertTrue(tf.reduce_all(tf.math.less_equal(tf.math.abs(cube_0_object.position-[0,0,-1]),epsilon)))
        self.assertTrue(tf.reduce_all(tf.math.less_equal(tf.math.abs(cube_1_object.position-[0,0,+1]),epsilon)))

        # Rotation around y-axis
        scene = load_scene()
        asset = AssetObject(name="asset_0", filename=sionna.rt.asset_object.test_asset_1,orientation=[0, PI / 2, PI / 2])
        scene.add(asset)
        cube_0_object = scene.get("asset_0_cube_0")
        cube_1_object = scene.get("asset_0_cube_1")
        self.assertTrue(tf.reduce_all(tf.math.less_equal(tf.math.abs(cube_0_object.position-[-1,0,0]),epsilon)))
        self.assertTrue(tf.reduce_all(tf.math.less_equal(tf.math.abs(cube_1_object.position-[+1,0,0]),epsilon)))

        # Rotation around z-axis
        scene = load_scene()
        asset = AssetObject(name="asset_0", filename=sionna.rt.asset_object.test_asset_1,orientation=[PI / 2, PI / 2, PI / 2])
        scene.add(asset)
        cube_0_object = scene.get("asset_0_cube_0")
        cube_1_object = scene.get("asset_0_cube_1")
        self.assertTrue(tf.reduce_all(tf.math.less_equal(tf.math.abs(cube_0_object.position-[0,-1,0]),epsilon)))
        self.assertTrue(tf.reduce_all(tf.math.less_equal(tf.math.abs(cube_1_object.position-[0,+1,0]),epsilon)))

        
    def test_orientation_add_vs_set(self):
        """Check that rotation accumulation lead to the same result as setting the complete rotation at once for asset objects"""
        scene = load_scene()
        asset = AssetObject(name="asset_0", filename=sionna.rt.asset_object.test_asset_1)
        scene.add(asset)
        
        cube_0_object = scene.get("asset_0_cube_0")
        cube_1_object = scene.get("asset_0_cube_1")

        epsilon = 1e-5 # The Boudning-boxes computed by sionna to estimate the position of an object are not entirely accurate
        self.assertTrue(tf.reduce_all(tf.math.less_equal(tf.math.abs(cube_0_object.position-[0,-1,0]),epsilon)))
        self.assertTrue(tf.reduce_all(tf.math.less_equal(tf.math.abs(cube_1_object.position-[0,+1,0]),epsilon)))

        # Add rotation around z-axis
        asset.orientation += [PI / 2, 0, 0]
        self.assertTrue(tf.reduce_all(tf.math.less_equal(tf.math.abs(cube_0_object.position-[+1,0,0]),epsilon)))
        self.assertTrue(tf.reduce_all(tf.math.less_equal(tf.math.abs(cube_1_object.position-[-1,0,0]),epsilon)))

        # Add rotation around z-axis
        asset.orientation += [3 * PI / 2, 0, 0]
        self.assertTrue(tf.reduce_all(tf.math.less_equal(tf.math.abs(cube_0_object.position-[0,-1,0]),epsilon)))
        self.assertTrue(tf.reduce_all(tf.math.less_equal(tf.math.abs(cube_1_object.position-[0,+1,0]),epsilon)))

        # Set rotation around z-axis
        asset.orientation = [0, 0, 0]
        self.assertTrue(tf.reduce_all(tf.math.less_equal(tf.math.abs(cube_0_object.position-[0,-1,0]),epsilon)))
        self.assertTrue(tf.reduce_all(tf.math.less_equal(tf.math.abs(cube_1_object.position-[0,+1,0]),epsilon)))

        # Set rotation around z-axis
        asset.orientation = [2 * PI, 0, 0]
        self.assertTrue(tf.reduce_all(tf.math.less_equal(tf.math.abs(cube_0_object.position-[0,-1,0]),epsilon)))
        self.assertTrue(tf.reduce_all(tf.math.less_equal(tf.math.abs(cube_1_object.position-[0,+1,0]),epsilon)))

    def test_orientation_impacts_paths(self):
        """Test showing that rotating a simple reflector asset can make a paths disappear"""
        scene = load_scene() # Empty scene
        asset = AssetObject(name="reflector", filename=sionna.rt.scene.simple_reflector) # N.B. a scene file can be added as an asset into a scene
        scene.add(asset)

        scene.tx_array = PlanarArray(1,1,0.5,0.5,"iso","V")
        scene.rx_array = PlanarArray(1,1,0.5,0.5,"iso","V")
        scene.add(Transmitter("tx", position=[0.1,0,100]))
        scene.add(Receiver("rx", position=[0.1,0,100]))
        
        # There should be a single reflected path
        paths = scene.compute_paths(los=False, diffraction=False, max_depth=1)
        self.assertEqual(tf.squeeze(paths.tau).shape, [])
        
        # Rotating the reflector by PI/4 should make the path disappear
        asset.orientation = [0,0,PI/4]

        paths = scene.compute_paths(los=False, diffraction=False, max_depth=1)
        self.assertEqual(tf.squeeze(paths.tau).shape, [0])

    def test_reverse_orientation_transform(self):
        """Test showing that reversing the transform get the object to its initial coordinates"""
        scene = load_scene() # Empty scene
        asset = AssetObject(name="asset_0", filename=sionna.rt.asset_object.test_asset_1)
        scene.add(asset)

        init_pos_cube_0 = scene.get('asset_0_cube_0').position
        init_pos_cube_1 = scene.get('asset_0_cube_1').position

        random_rotation = np.random.random(3)
        asset.orientation += +random_rotation
        asset.orientation += -random_rotation

        epsilon = 1e-5
        self.assertTrue(tf.reduce_all(tf.math.less_equal(tf.math.abs(init_pos_cube_0 - scene.get('asset_0_cube_0').position),epsilon)))
        self.assertTrue(tf.reduce_all(tf.math.less_equal(tf.math.abs(init_pos_cube_1 - scene.get('asset_0_cube_1').position),epsilon)))


    def test_look_at(self):
        """Check that the look_at argument works to set an asset orientation"""
        scene = load_scene() # Empty scene
        asset_0 = AssetObject(name="asset_0", filename=sionna.rt.asset_object.test_asset_1, position=[0,0,0])
        scene.add(asset_0)
        asset_1 = AssetObject(name="asset_1", filename=sionna.rt.asset_object.test_asset_1, position=[2,2,0], look_at=asset_0)
        scene.add(asset_1) 

        print(asset_0.position,asset_1.orientation)
        epsilon = 1e-5
        self.assertTrue(tf.reduce_all(tf.math.less_equal(tf.math.abs(np.array([2-np.sqrt(2)/2,2+np.sqrt(2)/2,0]) - scene.get('asset_1_cube_0').position),epsilon)))
        self.assertTrue(tf.reduce_all(tf.math.less_equal(tf.math.abs(np.array([2+np.sqrt(2)/2,2-np.sqrt(2)/2,0]) - scene.get('asset_1_cube_1').position),epsilon)))

    def test_look_at_update(self):
        """Check that the look_at argument works to update an asset orientation"""
        scene = load_scene() # Empty scene
        asset_0 = AssetObject(name="asset_0", filename=sionna.rt.asset_object.test_asset_1, position=[0,0,0])
        asset_1 = AssetObject(name="asset_1", filename=sionna.rt.asset_object.test_asset_1, position=[2,2,0])
        scene.add([asset_0,asset_1]) 

        epsilon = 1e-5
        self.assertTrue(tf.reduce_all(tf.math.less_equal(tf.math.abs(np.array([0,-1,0]) - scene.get('asset_0_cube_0').position),epsilon)))
        self.assertTrue(tf.reduce_all(tf.math.less_equal(tf.math.abs(np.array([0,+1,0]) - scene.get('asset_0_cube_1').position),epsilon)))
        self.assertTrue(tf.reduce_all(tf.math.less_equal(tf.math.abs(np.array([2,+1,0]) - scene.get('asset_1_cube_0').position),epsilon)))
        self.assertTrue(tf.reduce_all(tf.math.less_equal(tf.math.abs(np.array([2,+3,0]) - scene.get('asset_1_cube_1').position),epsilon)))

        asset_0.look_at(asset_1)
        epsilon = 1e-5
        self.assertTrue(tf.reduce_all(tf.math.less_equal(tf.math.abs(np.array([+np.sqrt(2)/2,-np.sqrt(2)/2,0]) - scene.get('asset_0_cube_0').position),epsilon)))
        self.assertTrue(tf.reduce_all(tf.math.less_equal(tf.math.abs(np.array([-np.sqrt(2)/2,+np.sqrt(2)/2,0]) - scene.get('asset_0_cube_1').position),epsilon)))

        asset_1.look_at([0,0,0])
        epsilon = 1e-5
        self.assertTrue(tf.reduce_all(tf.math.less_equal(tf.math.abs(np.array([2-np.sqrt(2)/2,2+np.sqrt(2)/2,0]) - scene.get('asset_1_cube_0').position),epsilon)))
        self.assertTrue(tf.reduce_all(tf.math.less_equal(tf.math.abs(np.array([2+np.sqrt(2)/2,2-np.sqrt(2)/2,0]) - scene.get('asset_1_cube_1').position),epsilon)))


        
