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
        
        
class TestAssetMaterialUpdate(unittest.TestCase):
    """Tests related to the update of asset's materials (i.e. after adding the asset to a scene)"""    
    def test_str_asset_material_update(self):
        """Test showing that changing asset material as a `str` after adding the asset works. Here the material name point to an existing scene material"""

        scene = load_scene(sionna.rt.scene.floor_wall)
        asset = AssetObject(name="asset_0", filename=sionna.rt.asset_object.test_asset_1, radio_material="itu_metal") # create a metal asset

        prev_glass = scene.get("itu_glass")
        prev_wood = scene.get("itu_wood")
        prev_metal =scene.get("itu_metal")

        # Before adding asset none of these materials are used and the bsdf are all placeholders
        self.assertFalse(prev_glass.is_used)
        self.assertFalse(prev_wood.is_used)
        self.assertFalse(prev_metal.is_used)

        # Before adding the asset, the asset radio material is the "itu_metal" str.
        self.assertTrue(asset.radio_material == "itu_metal")

        # Add the asset
        scene.add(asset)
        new_glass = scene.get("itu_glass")
        new_wood = scene.get("itu_wood")
        new_metal =scene.get("itu_metal")

        # After adding the asset, the asset radio material is the "itu_metal" RadioMaterial from scene
        metal_radio_material = scene.radio_materials["itu_metal"] 
        self.assertTrue(asset.radio_material == metal_radio_material)
        self.assertTrue(scene.get("asset_0").radio_material == metal_radio_material)

        # all shapes are metal
        asset_shapes = [asset.shapes[shape_name] for shape_name in asset.shapes]
        for shape in asset_shapes:
            self.assertTrue(shape.radio_material == metal_radio_material)
            self.assertTrue(shape.object_id in scene.get("itu_metal").using_objects)

        # After adding the asset, the material "itu_metal" is now used but the bsdf are still placeholders (since we use the material from the scene not from the asset xml)
        self.assertFalse(new_glass.is_used)
        self.assertFalse(new_wood.is_used)
        self.assertTrue(new_metal.is_used)

        # Now we change the asset material using a string pointing to an existing material
        ref_obj = scene.get("floor")
        asset.radio_material = "itu_glass"

        # After changing the asset material, the asset radio material is the "itu_glass" RadioMaterial from scene
        glass_radio_material = scene.radio_materials["itu_glass"] 
        self.assertTrue(asset.radio_material == glass_radio_material)
        self.assertTrue(scene.get("asset_0").radio_material == glass_radio_material)

        # all shapes are metal
        asset_shapes = [asset.shapes[shape_name] for shape_name in asset.shapes]
        for shape in asset_shapes:
            self.assertTrue(shape.radio_material == glass_radio_material)
            self.assertTrue(shape.object_id in scene.get("itu_glass").using_objects)

        # After changing the asset material, the material "itu_metal" is not used anymore and "itu _glass" is now used
        self.assertTrue(new_glass.is_used)
        self.assertFalse(new_wood.is_used)
        self.assertFalse(new_metal.is_used)

        # The material change should not trigger an auto reload of the scene. To update the view (i.e. bsdf related), the user must trigger scene.reload()
        # which might break differentiability...
        self.assertTrue(ref_obj == scene.get("floor"))


    def test_str_asset_material_unknown_update(self):
        """Check that changing asset material (after adding asset to scene) as a `str` refering to an unknown scene material, leads to the creation of a placeholder material (and bsdf)"""
        scene = load_scene(sionna.rt.scene.floor_wall)
        asset = AssetObject(name="asset_0", filename=sionna.rt.asset_object.test_asset_1, radio_material="itu_metal") # create a metal asset

        # Add the asset
        scene.add(asset)

        # Now we change the asset material using a string pointing to an unknown material >>> Error
        with self.assertRaises(ValueError) as context:
            asset.radio_material = "custom_rm"
        self.assertEqual(str(context.exception), "Unknown radio material 'custom_rm'")
 
        
       
    def test_radio_material_asset_material_update(self):
        """Test showing that changing asset material using a RadioMaterial object after adding the asset works"""
        scene = load_scene(sionna.rt.scene.floor_wall)
        asset = AssetObject(name="asset_0", filename=sionna.rt.asset_object.test_asset_1, radio_material="itu_metal") # create a metal asset

        prev_glass = scene.get("itu_glass")
        prev_wood = scene.get("itu_wood")
        prev_metal =scene.get("itu_metal")

        # Before adding asset none of these materials are used and the bsdf are all placeholders
        self.assertFalse(prev_glass.is_used)
        self.assertFalse(prev_wood.is_used)
        self.assertFalse(prev_metal.is_used)

        # Before adding the asset, the asset radio material is the "itu_metal" str.
        self.assertTrue(asset.radio_material == "itu_metal")

        # Add the asset
        scene.add(asset)
        new_glass = scene.get("itu_glass")
        new_wood = scene.get("itu_wood")
        new_metal =scene.get("itu_metal")

        # After adding the asset, the asset radio material is the "itu_metal" RadioMaterial from scene
        metal_radio_material = scene.radio_materials["itu_metal"] 
        self.assertTrue(asset.radio_material == metal_radio_material)
        self.assertTrue(scene.get("asset_0").radio_material == metal_radio_material)

        # all shapes are metal
        asset_shapes = [asset.shapes[shape_name] for shape_name in asset.shapes]
        for shape in asset_shapes:
            self.assertTrue(shape.radio_material == metal_radio_material)
            self.assertTrue(shape.object_id in scene.get("itu_metal").using_objects)

        # After adding the asset, the material "itu_metal" is now used but the bsdf are still placeholders (since we use the material from the scene not from the asset xml)
        self.assertFalse(new_glass.is_used)
        self.assertFalse(new_wood.is_used)
        self.assertTrue(new_metal.is_used)

        # Now we change the asset material using a new RadioMaterial object 
        ref_obj = scene.get("floor")
        asset.radio_material = RadioMaterial(name="custom_rm")

        # After changing the asset material, the asset radio material is the "itu_glass" RadioMaterial from scene
        custom_rm = scene.radio_materials["custom_rm"] 
        self.assertTrue(asset.radio_material == custom_rm)
        self.assertTrue(scene.get("asset_0").radio_material == custom_rm)

        # all shapes are metal
        asset_shapes = [asset.shapes[shape_name] for shape_name in asset.shapes]
        for shape in asset_shapes:
            self.assertTrue(shape.object_id in scene.get("custom_rm").using_objects)
            self.assertTrue(shape.radio_material == custom_rm)

        # After changing the asset material, the material "itu_metal" is not used anymore and "itu _glass" is now used
        self.assertFalse(new_glass.is_used)
        self.assertFalse(new_wood.is_used)
        self.assertFalse(new_metal.is_used)
        self.assertTrue(custom_rm.is_used)

        # The material change should not trigger an auto reload of the scene (default behaviour). To update the view (i.e. bsdf related), the user must trigger scene.reload()
        # which might break differentiability...
        self.assertTrue(ref_obj == scene.get("floor"))

     
    def test_radio_material_from_scene_asset_material_update(self):
        """Test showing that changing asset material using a RadioMaterial object from the scene after adding the asset works"""
        scene = load_scene(sionna.rt.scene.floor_wall)
        asset = AssetObject(name="asset_0", filename=sionna.rt.asset_object.test_asset_1, radio_material="itu_metal") # create a metal asset

        prev_glass = scene.get("itu_glass")
        prev_wood = scene.get("itu_wood")
        prev_metal =scene.get("itu_metal")

        # Before adding asset none of these materials are used and the bsdf are all placeholders
        self.assertFalse(prev_glass.is_used)
        self.assertFalse(prev_wood.is_used)
        self.assertFalse(prev_metal.is_used)

        # Before adding the asset, the asset radio material is the "itu_metal" str.
        self.assertTrue(asset.radio_material == "itu_metal")

        # Add the asset
        scene.add(asset)
        new_glass = scene.get("itu_glass")
        new_wood = scene.get("itu_wood")
        new_metal =scene.get("itu_metal")

        # After adding the asset, the asset radio material is the "itu_metal" RadioMaterial from scene
        metal_radio_material = scene.radio_materials["itu_metal"] 
        self.assertTrue(asset.radio_material == metal_radio_material)
        self.assertTrue(scene.get("asset_0").radio_material == metal_radio_material)

        # all shapes are metal
        asset_shapes = [asset.shapes[shape_name] for shape_name in asset.shapes]
        for shape in asset_shapes:
            self.assertTrue(shape.radio_material == metal_radio_material)
            self.assertTrue(shape.object_id in scene.get("itu_metal").using_objects)

        # After adding the asset, the material "itu_metal" is now used but the bsdf are still placeholders (since we use the material from the scene not from the asset xml)
        self.assertFalse(new_glass.is_used)
        self.assertFalse(new_wood.is_used)
        self.assertTrue(new_metal.is_used)

        # Now we change the asset material using a RadioMaterial object from the scene
        ref_obj = scene.get("floor")
        asset.radio_material = scene.get("itu_glass")

        # After changing the asset material, the asset radio material is the "itu_glass" RadioMaterial from scene
        glass_radio_material = scene.radio_materials["itu_glass"] 
        self.assertTrue(asset.radio_material == glass_radio_material)
        self.assertTrue(scene.get("asset_0").radio_material == glass_radio_material)

        # all shapes are metal
        asset_shapes = [asset.shapes[shape_name] for shape_name in asset.shapes]
        for shape in asset_shapes:
            self.assertTrue(shape.radio_material == glass_radio_material)
            self.assertTrue(shape.object_id in scene.get("itu_glass").using_objects)

        # After changing the asset material, the material "itu_metal" is not used anymore and "itu _glass" is now used
        self.assertTrue(new_glass.is_used)
        self.assertFalse(new_wood.is_used)
        self.assertFalse(new_metal.is_used)

        # The material change should not trigger an auto reload of the scene. To update the view (i.e. bsdf related), the user must trigger scene.reload()
        # which might break differentiability...
        self.assertTrue(ref_obj == scene.get("floor"))

    
    def test_already_used_name_asset_material_update(self):
        """Test showing that chaning asset material (after adding asset to scene) with a name already in use(item or other RadioMaterial) 
        already present in the scene doesn't work."""

        # create a floor material asset, where floor is str already used in the scene by another SceneObject
        scene = load_scene(sionna.rt.scene.floor_wall)       
        asset = AssetObject(name="asset_0", filename=sionna.rt.asset_object.test_asset_1) 
        scene.add(asset)

        with self.assertRaises(ValueError) as context:
            asset.radio_material = 'floor'
        self.assertEqual(str(context.exception), "Unknown radio material 'floor'")

        # create a itu_wood material asset, where itu_wood is a new RadioMaterial whose name is already used in the scene by another RadioMaterial
        scene = load_scene(sionna.rt.scene.floor_wall)   
        custom_rm = RadioMaterial("itu_wood")    
        asset = AssetObject(name="asset_0", filename=sionna.rt.asset_object.test_asset_1) 
        scene.add(asset)

        with self.assertRaises(ValueError) as context:
            asset.radio_material = custom_rm
        self.assertEqual(str(context.exception), "Name 'itu_wood' is already used by another item of the scene")

        # create a floor material asset, where floor is a new RadioMaterial whose name is already used in the scene by another SceneObject
        scene = load_scene(sionna.rt.scene.floor_wall)   
        custom_rm = RadioMaterial("floor")         
        asset = AssetObject(name="asset_0", filename=sionna.rt.asset_object.test_asset_1) 
        scene.add(asset)

        with self.assertRaises(ValueError) as context:
            asset.radio_material = custom_rm
        self.assertEqual(str(context.exception), "Name 'floor' is already used by another item of the scene")
       
       
    def test_wrong_type_asset_material_update(self):
        """Test showing that changing asset material (after adding asset to scene) using an invalid type raises an error"""
        scene = load_scene(sionna.rt.scene.floor_wall)
        asset = AssetObject(name="asset_0", filename=sionna.rt.asset_object.test_asset_1, radio_material='itu_wood')
        scene.add(asset)
        with self.assertRaises(TypeError) as context:
            asset.radio_material=5 # replace the asset material using an int as a radio_material
        self.assertEqual(str(context.exception), "Radio material must be of type 'str' or 'sionna.rt.RadioMaterial")

    def test_none_asset_material_update(self):
        """Test showing that changing asset material to None (after adding asset to scene) does not work"""
        scene = load_scene(sionna.rt.scene.floor_wall)
        asset = AssetObject(name="asset_0", filename=sionna.rt.asset_object.test_asset_1, radio_material='itu_wood')
        scene.add(asset)
        with self.assertRaises(TypeError) as context:
            asset.radio_material=None # replace the asset material to None
        self.assertEqual(str(context.exception), "Radio material must be of type 'str' or 'sionna.rt.RadioMaterial")

    def test_asset_material_update_via_ray(self):
        """Check that an asset material update cause difference in the ray propagation"""

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
        
        # Since the material is not vacuum, the path has energy
        self.assertTrue(tf.reduce_sum(tf.squeeze(cir[0])) != 0)

        # Change asset radio material
        ref_obj = scene.get('floor')
        asset.radio_material = 'vacuum'#fully_absorbant_radio_material
        
        # Check that the scene have not been reloaded (no need if just changing material properties, and not visuals)
        self.assertTrue(ref_obj == scene.get("floor"))

        # Measure new pathes with vacuum material (the path still exist but it has 0 energy)
        paths = scene.compute_paths(los=False, diffraction=False, max_depth=1, method="exhaustive")
        paths.normalize_delays = False
        cir = paths.cir()
        self.assertTrue(tf.reduce_sum(tf.squeeze(cir[0])) == 0) 