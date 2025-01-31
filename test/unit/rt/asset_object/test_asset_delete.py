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


class TestAssetDelete(unittest.TestCase):
    """Tests related to the deletion an asset object"""

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
        

    def test_weak_references(self):
        """Check that scene objects are weakly referenced outside the scene class"""
        scene = load_scene(sionna.rt.scene.floor_wall)
        asset = AssetObject(name="asset_0", filename=sionna.rt.asset_object.test_asset_1)
        scene.add(asset) 
        
        # What happens when removing the asset from the scene (N.B. the asset object is not removed but the link to the scene and the corresponding SceneObjects should be)
        asset_shapes = asset.shapes
        asset_object = asset_shapes["asset_0_cube_0"]
        scene_object = scene.get("asset_0_cube_0")
        self.assertTrue(scene_object is asset_object) 

        scene.remove("asset_0")

        self.assertTrue(scene.get("asset_0") == None) 
        self.assertTrue(scene.get("asset_0_cube_0") == None)
        self.assertTrue(asset_shapes == {})

        with self.assertRaises(ReferenceError) as context:
            print(asset_object)
        self.assertEqual(str(context.exception), "weakly-referenced object no longer exists")

    def test_asset_scene_objets_are_scene_scene_object(self):
        """Check that AssetObjects reference the same SceneObjects as Scene"""
        scene = load_scene(sionna.rt.scene.floor_wall)
        asset = AssetObject(name="asset_0", filename=sionna.rt.asset_object.test_asset_1)
        scene.add(asset) 
        
        # What happens when removing the asset from the scene (N.B. the asset object is not removed but the link to the scene and the corresponding SceneObjects should be)
        asset_shapes = asset.shapes
        asset_object = asset_shapes["asset_0_cube_0"]
        scene_object = scene.get("asset_0_cube_0")
        self.assertTrue(asset_object is scene_object)

    def test_shape_name_is_consistent_with_scene_object_name(self):
        """Check that the name of shape in the xml is consistent with scene object names, i.e. shape name is f'mesh-{scene_object.name}'"""
        scene = load_scene(sionna.rt.scene.floor_wall)
        asset_1 = AssetObject(name="asset_1", filename=sionna.rt.asset_object.test_asset_1) 
        asset_2 = AssetObject(name="mesh-asset_2", filename=sionna.rt.asset_object.test_asset_2) 
        scene.add([asset_1,asset_2])

        root = scene._xml_tree.getroot()
        shapes_in_root = root.findall('shape')
        shapes_in_root = [shape.get('id') for shape in shapes_in_root]

        for obj_name in scene.objects:
            self.assertTrue(f"mesh-{obj_name}" in shapes_in_root)

    def test_remove_scene_object(self):
        scene = load_scene(sionna.rt.scene.floor_wall)
        asset = AssetObject(name="asset_0", filename=sionna.rt.asset_object.test_asset_1) 
        scene.add(asset)
        ref_obj = scene.get('wall')
        ref_obj_mi_shape = ref_obj.mi_shape

        # Remove object >>> Object does not exist anymore
        self.assertTrue(isinstance(scene.get('floor'),SceneObject))
        scene.remove("floor")
        self.assertEqual(scene.get('floor'),None)
                    
        # Scene is reloaded
        self.assertTrue(ref_obj is scene.get('wall'))
        self.assertFalse(ref_obj_mi_shape is scene.get('wall').mi_shape)

        # Can't remove an object from an asset
        with self.assertRaises(ValueError) as context:
            scene.remove("asset_0_cube_0")
        self.assertEqual(str(context.exception), "Can't remove a SceneObject part of an AssetObject. Try removing the complete AssetObject instead.")


