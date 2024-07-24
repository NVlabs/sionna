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
import tensorflow as tf
import sys

from sionna.rt import *

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

        
class TestAddAssetObject(unittest.TestCase):
    """Tests related to the AssetObject class"""
    
    def test_add_asset(self):
        """Adding asset to scene"""
        scene = load_scene(sionna.rt.scene.floor_wall)
        asset = AssetObject(name="asset_0", filename=sionna.rt.asset_object.test_asset_1)
        ref_obj = scene.get("floor")
        scene.add(asset)
        self.assertTrue("asset_0" in scene.asset_objects)
        self.assertTrue(isinstance(scene.get("asset_0"),AssetObject)) 
        self.assertTrue(isinstance(scene.get("asset_0_cube_0"),SceneObject)) 
        self.assertTrue(isinstance(scene.get("asset_0_cube_1"),SceneObject)) 

        # Check that the scene is automatically reloaded
        self.assertTrue(ref_obj != scene.get("floor"))

    def test_add_asset_list(self):
        """Adding list of asset to scene"""
        scene = load_scene(sionna.rt.scene.floor_wall)
        asset_0 = AssetObject(name="asset_0", filename=sionna.rt.asset_object.test_asset_1)
        asset_1 = AssetObject(name="asset_1", filename=sionna.rt.asset_object.test_asset_1)
        asset_list = [asset_0, asset_1]
        scene.add(asset_list)

        self.assertTrue("asset_0" in scene.asset_objects)
        self.assertTrue(isinstance(scene.get("asset_0"),AssetObject)) 
        self.assertTrue(isinstance(scene.get("asset_0_cube_0"),SceneObject)) 
        self.assertTrue(isinstance(scene.get("asset_0_cube_1"),SceneObject)) 

        self.assertTrue("asset_1" in scene.asset_objects)
        self.assertTrue(isinstance(scene.get("asset_1"),AssetObject)) 
        self.assertTrue(isinstance(scene.get("asset_1_cube_0"),SceneObject)) 
        self.assertTrue(isinstance(scene.get("asset_1_cube_1"),SceneObject)) 

    def test_add_asset_adjust_to_scene_dtype(self):
        """When adding an asset to a scene, the asset should adapt to the scene dtype"""
        scene = load_scene(sionna.rt.scene.floor_wall, dtype=tf.complex64)
        asset = AssetObject(name="asset_0", filename=sionna.rt.asset_object.test_asset_1, dtype=tf.complex128)
        scene.add(asset)
        self.assertEqual(scene.dtype, asset.dtype)
        self.assertEqual(scene.dtype.real_dtype, asset.position.dtype)

    def test_add_asset_overwrite(self):
        """When adding an asset to a scene, the asset should overwrite any asset with the same name"""
        scene = load_scene(sionna.rt.scene.floor_wall)
        asset_0 = AssetObject(name="asset_0", filename=sionna.rt.asset_object.test_asset_1)
        scene.add(asset_0)

        self.assertTrue("asset_0" in scene.asset_objects)
        self.assertTrue(scene.get("asset_0") == asset_0)  

        asset_0_bis = AssetObject(name="asset_0", filename=sionna.rt.asset_object.test_asset_1)
        scene.add(asset_0_bis)

        self.assertTrue("asset_0" in scene.asset_objects)
        self.assertTrue(scene.get("asset_0") != asset_0) 

    def test_remove_asset(self):
        """Removing an asset from scene"""
        scene = load_scene(sionna.rt.scene.floor_wall)
        asset = AssetObject(name="asset_0", filename=sionna.rt.asset_object.test_asset_1)
        scene.add(asset)
        self.assertTrue("asset_0" in scene.asset_objects)
        self.assertTrue(isinstance(scene.get("asset_0"),AssetObject)) 
        self.assertTrue(isinstance(scene.get("asset_0_cube_0"),SceneObject)) 
        self.assertTrue(isinstance(scene.get("asset_0_cube_1"),SceneObject)) 

        scene.remove("asset_0")
        self.assertTrue("asset_0" not in scene.asset_objects)
        self.assertTrue(scene.get("asset_0") == None) 
        self.assertTrue(scene.get("asset_0_cube_0") == None) 
        self.assertTrue(scene.get("asset_0_cube_1") == None) 

    def test_asset_shape_dictionary(self):
        """Instanciation of the asset's shapes dict is correct when adding asset"""
        scene = load_scene(sionna.rt.scene.floor_wall)
        asset = AssetObject(name="asset_0", filename=sionna.rt.asset_object.test_asset_1)

        self.assertFalse("asset_0" in scene.asset_objects)
        self.assertTrue(asset.shapes == {})

        scene.add(asset)
        self.assertTrue("asset_0" in scene.asset_objects)
        self.assertTrue(isinstance(scene.get("asset_0"),AssetObject)) 
        self.assertTrue(isinstance(scene.get("asset_0_cube_0"),SceneObject)) 
        self.assertTrue(isinstance(scene.get("asset_0_cube_1"),SceneObject)) 
        print("SceneObject: ",asset.shapes["asset_0_cube_0"])
        self.assertTrue(isinstance(asset.shapes["asset_0_cube_0"],SceneObject))
        self.assertTrue(isinstance(asset.shapes["asset_0_cube_1"],SceneObject))   