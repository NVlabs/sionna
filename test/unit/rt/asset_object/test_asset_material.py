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
        
        
class TestSetAssetMaterial(unittest.TestCase):
    """Tests related to the asset's materials definition before adding the asset to a scene"""

    def test_xml_asset_material(self):
        """Test showing that specifying no asset material before adding the asset works when the asset is added to scene.
        When no asset material is specified, the material from the asset xml file are used. If this materials have the same 
        name as existing scene material, then the scene material are used. Yet, if the scene material have placeholder bsdfs,
        the later are replace by the asset's xml bsdf description."""

        scene = load_scene(sionna.rt.scene.floor_wall)
        asset = AssetObject(name="asset_0", filename=sionna.rt.asset_object.test_asset_1) # since no material are specified, the material from the asset xml wiil be used: test asset is made of itu_metal and itu_wood

        # The materials described in the asset xml file are 'itu_wood'and 'itu_metal' which are existing scene material, although with placeholder bsdfs
        prev_glass = scene.get("itu_glass")
        prev_wood = scene.get("itu_wood")
        prev_metal =scene.get("itu_metal")
        prev_glass_bsdf = prev_glass.bsdf
        prev_wood_bsdf = prev_wood.bsdf
        prev_metal_bsdf = prev_metal.bsdf
        prev_glass_bsdf_xml = prev_glass.bsdf.xml_element
        prev_wood_bsdf_xml = prev_wood.bsdf.xml_element
        prev_metal_bsdf_xml = prev_metal.bsdf.xml_element

        # Before adding asset none of these materials are used and the bsdf are all placeholders
        self.assertFalse(prev_glass.is_used)
        self.assertFalse(prev_wood.is_used)
        self.assertFalse(prev_metal.is_used)
        self.assertFalse(prev_glass.is_placeholder)
        self.assertFalse(prev_wood.is_placeholder)
        self.assertFalse(prev_metal.is_placeholder)
        self.assertTrue(prev_glass_bsdf.is_placeholder)
        self.assertTrue(prev_wood_bsdf.is_placeholder)
        self.assertTrue(prev_metal_bsdf.is_placeholder)

        # Add the asset
        ref_obj = scene.get("floor")
        scene.add(asset)

        # After adding the asset, the asset radio material is a dictionary where each key is a material name, and the value are the list of scene object using these material within the asset
        # When adding asset to scene, the material are specified on a per shape basis (using the asset xml) since no material have been specified.
        #self.assertTrue(asset.radio_material == {prev_wood.name: [scene.get("asset_0_cube_0")],prev_metal.name:[scene.get("asset_0_cube_1")]})
        self.assertTrue(asset.radio_material == None)
        self.assertTrue(scene.get("asset_0").radio_material == asset.radio_material)

        # After adding the asset, the material described in the asset XML (itu_wood and itu_metal)  are now used and their bsdf are not placeholders anymore
        new_glass = scene.get("itu_glass")
        new_wood = scene.get("itu_wood")
        new_metal =scene.get("itu_metal")
        new_glass_bsdf = new_glass.bsdf
        new_wood_bsdf = new_wood.bsdf
        new_metal_bsdf = new_metal.bsdf
        self.assertFalse(new_glass.is_used)
        self.assertTrue(new_wood.is_used)
        self.assertTrue(new_metal.is_used)
        self.assertFalse(prev_glass.is_placeholder)
        self.assertFalse(prev_wood.is_placeholder)
        self.assertFalse(prev_metal.is_placeholder)
        self.assertTrue(new_glass_bsdf.is_placeholder)
        self.assertFalse(new_wood_bsdf.is_placeholder)
        self.assertFalse(new_metal_bsdf.is_placeholder)
        self.assertTrue(asset.shapes['asset_0_cube_0'].object_id in scene.get("itu_wood").using_objects)
        self.assertTrue(asset.shapes['asset_0_cube_1'].object_id in scene.get("itu_metal").using_objects)

        # After adding asset, the material object instance and bsdf should not change
        new_glass = scene.get("itu_glass")
        new_wood = scene.get("itu_wood")
        new_metal =scene.get("itu_metal")
        new_glass_bsdf = new_glass.bsdf
        new_wood_bsdf = new_wood.bsdf
        new_metal_bsdf = new_metal.bsdf
        new_glass_bsdf_xml = new_glass_bsdf.xml_element
        new_wood_bsdf_xml = new_wood_bsdf.xml_element
        new_metal_bsdf_xml = new_metal_bsdf.xml_element

        self.assertTrue(new_glass is prev_glass)
        self.assertTrue(new_wood is prev_wood)
        self.assertTrue(new_metal is prev_metal)
        self.assertTrue(new_glass_bsdf is prev_glass_bsdf)
        self.assertTrue(new_wood_bsdf is prev_wood_bsdf)
        self.assertTrue(new_metal_bsdf is prev_metal_bsdf)

        # Although the bsdf and material object are the same, the bsdf xml_element shoud have been updated
        self.assertTrue(prev_glass_bsdf_xml == new_glass_bsdf_xml)
        self.assertFalse(prev_wood_bsdf_xml == new_wood_bsdf_xml)
        self.assertFalse(prev_metal_bsdf_xml == new_metal_bsdf_xml)

        # Check that the scene is automatically reloaded
        self.assertTrue(ref_obj != scene.get("floor"))

    def test_xml_asset_material_unknown(self):
        """Check that specifying asset material as None works even when the asset_xml file refering to unknown scene materials,
        thus leading to the creation of placeholder materials (and bsdf)"""
        scene = load_scene(sionna.rt.scene.floor_wall)
        asset = AssetObject(name="asset_0", filename=sionna.rt.asset_object.test_asset_2) # radio material = None >>> use material name from the XML file 
        
        self.assertTrue(asset.radio_material == None)
        self.assertTrue(scene.get("custom_rm_1") == None)
        self.assertTrue(scene.get("custom_rm_2") == None)

        # Add the asset
        ref_obj = scene.get("floor")
        scene.add(asset)

        # After adding the asset, the asset radio material are placeholder RadioMaterials
        scene_custom_rm_1 = scene.radio_materials["custom_rm_1"] 
        scene_custom_rm_2 = scene.radio_materials["custom_rm_2"] 
        self.assertTrue(asset.radio_material == None)
        self.assertTrue(asset.shapes['asset_0_cube_1'].radio_material == scene_custom_rm_1)
        self.assertTrue(asset.shapes['asset_0_cube_0'].radio_material == scene_custom_rm_2)
        self.assertTrue(scene_custom_rm_1.is_placeholder)
        self.assertFalse(scene_custom_rm_1.bsdf.is_placeholder)
        self.assertTrue(scene_custom_rm_1.is_used)
        self.assertTrue(scene_custom_rm_2.is_placeholder)
        self.assertFalse(scene_custom_rm_2.bsdf.is_placeholder)
        self.assertTrue(scene_custom_rm_2.is_used)

        # After adding the asset, the asset radio material is a dictionary where each key is a material name, and the value are the list of scene object using these material within the asset
        # self.assertTrue(asset.radio_material == {scene_custom_rm_2.name: [scene.get("asset_0_cube_0")],scene_custom_rm_1.name:[scene.get("asset_0_cube_1")]})
        
        # Check that the scene is automatically reloaded
        self.assertTrue(ref_obj != scene.get("floor"))


    def test_str_asset_material(self):
        """Test showing that specifying asset material as a `str` before adding the asset work when the asset is added to scene. Here the material name point to an existing scene material"""

        scene = load_scene(sionna.rt.scene.floor_wall)
        asset = AssetObject(name="asset_0", filename=sionna.rt.asset_object.test_asset_1, radio_material="itu_metal") # create a metal asset

        prev_glass = scene.get("itu_glass")
        prev_wood = scene.get("itu_wood")
        prev_metal =scene.get("itu_metal")
        prev_glass_bsdf = prev_glass.bsdf
        prev_wood_bsdf = prev_wood.bsdf
        prev_metal_bsdf = prev_metal.bsdf
        prev_glass_bsdf_xml = prev_glass.bsdf.xml_element
        prev_wood_bsdf_xml = prev_wood.bsdf.xml_element
        prev_metal_bsdf_xml = prev_metal.bsdf.xml_element

        # Before adding asset none of these materials are used and the bsdf are all placeholders
        self.assertFalse(prev_glass.is_used)
        self.assertFalse(prev_wood.is_used)
        self.assertFalse(prev_metal.is_used)
        self.assertFalse(prev_glass.is_placeholder)
        self.assertFalse(prev_wood.is_placeholder)
        self.assertFalse(prev_metal.is_placeholder)
        self.assertTrue(prev_glass_bsdf.is_placeholder)
        self.assertTrue(prev_wood_bsdf.is_placeholder)
        self.assertTrue(prev_metal_bsdf.is_placeholder)

        # Before adding the asset, the asset radio material is the "itu_metal" str.
        self.assertTrue(asset.radio_material == "itu_metal")

        # Add the asset
        ref_obj = scene.get("floor")
        scene.add(asset)

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
        new_glass = scene.get("itu_glass")
        new_wood = scene.get("itu_wood")
        new_metal =scene.get("itu_metal")
        new_glass_bsdf = new_glass.bsdf
        new_wood_bsdf = new_wood.bsdf
        new_metal_bsdf = new_metal.bsdf
        self.assertFalse(new_glass.is_used)
        self.assertFalse(new_wood.is_used)
        self.assertTrue(new_metal.is_used)
        self.assertFalse(prev_glass.is_placeholder)
        self.assertFalse(prev_wood.is_placeholder)
        self.assertFalse(prev_metal.is_placeholder)
        self.assertTrue(new_glass_bsdf.is_placeholder)
        self.assertTrue(new_wood_bsdf.is_placeholder)
        self.assertTrue(new_metal_bsdf.is_placeholder)
        

        # After adding asset, the material object instance and bsdf should not change
        new_glass = scene.get("itu_glass")
        new_wood = scene.get("itu_wood")
        new_metal =scene.get("itu_metal")
        new_glass_bsdf = new_glass.bsdf
        new_wood_bsdf = new_wood.bsdf
        new_metal_bsdf = new_metal.bsdf
        new_glass_bsdf_xml = new_glass_bsdf.xml_element
        new_wood_bsdf_xml = new_wood_bsdf.xml_element
        new_metal_bsdf_xml = new_metal_bsdf.xml_element

        self.assertTrue(new_glass is prev_glass)
        self.assertTrue(new_wood is prev_wood)
        self.assertTrue(new_metal is prev_metal)
        self.assertTrue(new_glass_bsdf is prev_glass_bsdf)
        self.assertTrue(new_wood_bsdf is prev_wood_bsdf)
        self.assertTrue(new_metal_bsdf is prev_metal_bsdf)

        # Since the used material for the asset existed in the scene the bsdf xml_element shoudn't have been updated
        self.assertTrue(prev_glass_bsdf_xml == new_glass_bsdf_xml)
        self.assertTrue(prev_wood_bsdf_xml == new_wood_bsdf_xml)
        self.assertTrue(prev_metal_bsdf_xml == new_metal_bsdf_xml)

        # Check that the scene is automatically reloaded
        self.assertTrue(ref_obj != scene.get("floor"))

    def test_str_asset_material_unknown(self):
        """Check that specifying asset material as a `str` refering to an unknown scene material, leads to the creation of a placeholder material (and bsdf)"""

        scene = load_scene(sionna.rt.scene.floor_wall)
        asset = AssetObject(name="asset_0", filename=sionna.rt.asset_object.test_asset_1, radio_material="custom_rm") 
        
        self.assertTrue(asset.radio_material == "custom_rm")
        self.assertTrue(scene.get("custom_rm") == None)

        # Add the asset
        ref_obj = scene.get("floor")
        scene.add(asset)

        # After adding the asset, the asset radio material is the "custom_rm" placeholder RadioMaterial
        scene_custom_rm = scene.radio_materials["custom_rm"] 
        self.assertTrue(asset.radio_material == scene_custom_rm)
        self.assertTrue(scene.get("asset_0").radio_material == scene_custom_rm)
        self.assertTrue(scene_custom_rm.is_placeholder)
        self.assertTrue(scene_custom_rm.bsdf.is_placeholder)
        self.assertTrue(scene_custom_rm.is_used)

        # Check that the scene is automatically reloaded
        self.assertTrue(ref_obj != scene.get("floor"))

       
    def test_radio_material_asset_material(self):
        """Test showing that specifying asset material as a RadioMaterial before adding the asset work when the asset is added to scene"""
        scene = load_scene(sionna.rt.scene.floor_wall)
        custom_rm = RadioMaterial("custom_rm", bsdf=BSDF("custom_bsdf"))
        asset = AssetObject(name="asset_0", filename=sionna.rt.asset_object.test_asset_1, radio_material=custom_rm) # create a custom_rm asset

        prev_custom_rm = scene.get("custom_rm")
        prev_glass = scene.get("itu_glass")
        prev_wood = scene.get("itu_wood")
        prev_metal =scene.get("itu_metal")
        prev_glass_bsdf = prev_glass.bsdf
        prev_wood_bsdf = prev_wood.bsdf
        prev_metal_bsdf = prev_metal.bsdf
        prev_glass_bsdf_xml = prev_glass.bsdf.xml_element
        prev_wood_bsdf_xml = prev_wood.bsdf.xml_element
        prev_metal_bsdf_xml = prev_metal.bsdf.xml_element

        # Before adding asset none of these materials are used and the bsdf are all placeholders
        # The custom rm is not in scene radio_materials
        self.assertFalse(prev_glass.is_used)
        self.assertFalse(prev_wood.is_used)
        self.assertFalse(prev_metal.is_used)
        self.assertFalse(prev_glass.is_placeholder)
        self.assertFalse(prev_wood.is_placeholder)
        self.assertFalse(prev_metal.is_placeholder)
        self.assertTrue(prev_glass_bsdf.is_placeholder)
        self.assertTrue(prev_wood_bsdf.is_placeholder)
        self.assertTrue(prev_metal_bsdf.is_placeholder)
        self.assertTrue(prev_custom_rm == None)

        # Before adding the asset, the asset radio material is the custom_rm RadioMateral.
        self.assertTrue(asset.radio_material == custom_rm)

        # Add the asset
        ref_obj = scene.get("floor")
        scene.add(asset)

        # After adding the asset, the asset radio material is the new custom_rm
        scene_custom_rm = scene.get("custom_rm")
        self.assertTrue(custom_rm == scene_custom_rm)
        self.assertTrue(asset.radio_material == scene_custom_rm)
        self.assertTrue(scene.get("asset_0").radio_material == scene_custom_rm)

        # all shapes are using this material
        asset_shapes = [asset.shapes[shape_name] for shape_name in asset.shapes]
        for shape in asset_shapes:
            self.assertTrue(shape.radio_material == scene_custom_rm)
            self.assertTrue(shape.object_id in scene_custom_rm.using_objects)

        # After adding the asset, the material are still not used
        new_glass = scene.get("itu_glass")
        new_wood = scene.get("itu_wood")
        new_metal =scene.get("itu_metal")
        new_glass_bsdf = new_glass.bsdf
        new_wood_bsdf = new_wood.bsdf
        new_metal_bsdf = new_metal.bsdf
        self.assertFalse(new_glass.is_used)
        self.assertFalse(new_wood.is_used)
        self.assertFalse(new_metal.is_used)
        self.assertFalse(prev_glass.is_placeholder)
        self.assertFalse(prev_wood.is_placeholder)
        self.assertFalse(prev_metal.is_placeholder)
        self.assertTrue(new_glass_bsdf.is_placeholder)
        self.assertTrue(new_wood_bsdf.is_placeholder)
        self.assertTrue(new_metal_bsdf.is_placeholder)
        
        # Check that the added rm is not placeholder, nor its bsdf
        self.assertTrue(scene_custom_rm.is_used)
        self.assertFalse(scene_custom_rm.is_placeholder)
        self.assertFalse(scene_custom_rm.bsdf.is_placeholder)

        # After adding asset, the material object instance and bsdf should not change
        new_glass = scene.get("itu_glass")
        new_wood = scene.get("itu_wood")
        new_metal =scene.get("itu_metal")
        new_glass_bsdf = new_glass.bsdf
        new_wood_bsdf = new_wood.bsdf
        new_metal_bsdf = new_metal.bsdf
        new_glass_bsdf_xml = new_glass_bsdf.xml_element
        new_wood_bsdf_xml = new_wood_bsdf.xml_element
        new_metal_bsdf_xml = new_metal_bsdf.xml_element

        self.assertTrue(new_glass is prev_glass)
        self.assertTrue(new_wood is prev_wood)
        self.assertTrue(new_metal is prev_metal)
        self.assertTrue(new_glass_bsdf is prev_glass_bsdf)
        self.assertTrue(new_wood_bsdf is prev_wood_bsdf)
        self.assertTrue(new_metal_bsdf is prev_metal_bsdf)

        # Since the the itu material are not used by the asset, the bsdf xml_element shoudn't have been updated
        self.assertTrue(prev_glass_bsdf_xml == new_glass_bsdf_xml)
        self.assertTrue(prev_wood_bsdf_xml == new_wood_bsdf_xml)
        self.assertTrue(prev_metal_bsdf_xml == new_metal_bsdf_xml)

        # Check that the scene is automatically reloaded
        self.assertTrue(ref_obj != scene.get("floor"))
     
    def test_radio_material_from_scene_asset_material(self):
        """Test showing that specifying asset material as a RadioMaterial from the scene before adding the asset work when the asset is added to scene"""
        scene = load_scene(sionna.rt.scene.floor_wall)
        itu_wood = scene.get("itu_wood")
        asset = AssetObject(name="asset_0", filename=sionna.rt.asset_object.test_asset_1, radio_material=itu_wood) # create a itu_wood asset, where itu_wood is a RadioMaterial extracted from scene

        prev_glass = scene.get("itu_glass")
        prev_wood = scene.get("itu_wood")
        prev_metal =scene.get("itu_metal")
        prev_glass_bsdf = prev_glass.bsdf
        prev_wood_bsdf = prev_wood.bsdf
        prev_metal_bsdf = prev_metal.bsdf
        prev_glass_bsdf_xml = prev_glass.bsdf.xml_element
        prev_wood_bsdf_xml = prev_wood.bsdf.xml_element
        prev_metal_bsdf_xml = prev_metal.bsdf.xml_element

        # Before adding asset none of these materials are used and the bsdf are all placeholders
        self.assertFalse(prev_glass.is_used)
        self.assertFalse(prev_wood.is_used)
        self.assertFalse(prev_metal.is_used)
        self.assertFalse(prev_glass.is_placeholder)
        self.assertFalse(prev_wood.is_placeholder)
        self.assertFalse(prev_metal.is_placeholder)
        self.assertTrue(prev_glass_bsdf.is_placeholder)
        self.assertTrue(prev_wood_bsdf.is_placeholder)
        self.assertTrue(prev_metal_bsdf.is_placeholder)

        # Before adding the asset, the asset radio material is the itu_wood RadioMaterial.
        self.assertTrue(asset.radio_material == itu_wood)
        self.assertTrue(asset.radio_material == scene.get("itu_wood"))

        # Add the asset
        ref_obj = scene.get("floor")
        scene.add(asset)

        # After adding the asset, the asset radio material is the itu_wood RadioMaterial.
        self.assertTrue(itu_wood == scene.get("itu_wood"))
        self.assertTrue(asset.radio_material == itu_wood)
        self.assertTrue(scene.get("asset_0").radio_material == itu_wood)

        # all shapes are using this material
        asset_shapes = [asset.shapes[shape_name] for shape_name in asset.shapes]
        for shape in asset_shapes:
            self.assertTrue(shape.radio_material == itu_wood)
            self.assertTrue(shape.object_id in itu_wood.using_objects)

        # After adding the asset, the material are still not used except for the itu_wood 
        new_glass = scene.get("itu_glass")
        new_wood = scene.get("itu_wood")
        new_metal = scene.get("itu_metal")
        new_glass_bsdf = new_glass.bsdf
        new_wood_bsdf = new_wood.bsdf
        new_metal_bsdf = new_metal.bsdf
        self.assertFalse(new_glass.is_used)
        self.assertTrue(new_wood.is_used)
        self.assertFalse(new_metal.is_used)
        self.assertFalse(prev_glass.is_placeholder)
        self.assertFalse(prev_wood.is_placeholder)
        self.assertFalse(prev_metal.is_placeholder)
        self.assertTrue(new_glass_bsdf.is_placeholder)
        self.assertTrue(new_wood_bsdf.is_placeholder)
        self.assertTrue(new_metal_bsdf.is_placeholder)
        
        # After adding asset, the material object instance and bsdf should not change
        new_glass = scene.get("itu_glass")
        new_wood = scene.get("itu_wood")
        new_metal =scene.get("itu_metal")
        new_glass_bsdf = new_glass.bsdf
        new_wood_bsdf = new_wood.bsdf
        new_metal_bsdf = new_metal.bsdf
        new_glass_bsdf_xml = new_glass_bsdf.xml_element
        new_wood_bsdf_xml = new_wood_bsdf.xml_element
        new_metal_bsdf_xml = new_metal_bsdf.xml_element

        self.assertTrue(new_glass is prev_glass)
        self.assertTrue(new_wood is prev_wood)
        self.assertTrue(new_metal is prev_metal)
        self.assertTrue(new_glass_bsdf is prev_glass_bsdf)
        self.assertTrue(new_wood_bsdf is prev_wood_bsdf)
        self.assertTrue(new_metal_bsdf is prev_metal_bsdf)

        # Since the asset uses the scene material as is, the bsdf xml_element shoudn't have been updated
        self.assertTrue(prev_glass_bsdf_xml == new_glass_bsdf_xml)
        self.assertTrue(prev_wood_bsdf_xml == new_wood_bsdf_xml)
        self.assertTrue(prev_metal_bsdf_xml == new_metal_bsdf_xml)

        # Check that the scene is automatically reloaded
        self.assertTrue(ref_obj != scene.get("floor"))
  
    def test_already_used_name_asset_material(self):
        """Test showing that specifying asset material with a name already in use(item or other RadioMaterial) already present in the scene doesn't work when the asset is added to scene"""

        # create a floor material asset, where floor is str already used in the scene by another SceneObject
        scene = load_scene(sionna.rt.scene.floor_wall)       
        asset = AssetObject(name="asset_0", filename=sionna.rt.asset_object.test_asset_1, radio_material="floor") 

        with self.assertRaises(ValueError) as context:
            scene.add(asset)
        self.assertEqual(str(context.exception), "Name 'floor' is already used by another item of the scene")

        # create a itu_wood material asset, where itu_wood is a new RadioMaterial whose name is already used in the scene by another RadioMaterial
        scene = load_scene(sionna.rt.scene.floor_wall)   
        custom_rm = RadioMaterial("itu_wood")    
        asset = AssetObject(name="asset_0", filename=sionna.rt.asset_object.test_asset_1, radio_material=custom_rm) 

        with self.assertRaises(ValueError) as context:
            scene.add(asset)
        self.assertEqual(str(context.exception), "Name 'itu_wood' is already used by another item of the scene")

        # create a floor material asset, where floor is a new RadioMaterial whose name is already used in the scene by another SceneObject
        scene = load_scene(sionna.rt.scene.floor_wall)   
        custom_rm = RadioMaterial("floor")         
        asset = AssetObject(name="asset_0", filename=sionna.rt.asset_object.test_asset_1, radio_material=custom_rm) 

        with self.assertRaises(ValueError) as context:
            scene.add(asset)
        self.assertEqual(str(context.exception), "Name 'floor' is already used by another item of the scene")
       
        
    def test_wrong_type_asset_material(self):
        """Test showing that specifying asset material using an invalid type raises an error"""
        scene = load_scene(sionna.rt.scene.floor_wall)
        
        with self.assertRaises(TypeError) as context:
            asset = AssetObject(name="asset_0", filename=sionna.rt.asset_object.test_asset_1, radio_material=5) # create an asset using an int as a radio_material
        self.assertEqual(str(context.exception), "`radio_material` must be `str` or `RadioMaterial` (or None)")


    def test_asset_material_via_ray(self):
        """Check that adding asset material with different material cause difference in the ray propagation"""

        scene = load_scene() # Load empty scene
        scene.tx_array = PlanarArray(1,1,0.5,0.5,"iso","V")
        scene.rx_array = PlanarArray(1,1,0.5,0.5,"iso","V")

        ## The test cube(s) are two 1x1x1m cubes, centered in (0,0,0), spaced by 1 m along the y-axis
        cube_edge_length = 1 
        cubes_separation = 1
        asset = AssetObject(name="reflector", filename=sionna.rt.asset_object.test_asset_1, position=[0,-cubes_separation,-cube_edge_length/2]) 
        scene.add(asset)

        d0 = 100
        scene.add(Transmitter("tx", position=[0,+1,d0]))
        scene.add(Receiver("rx", position=[0,-1,d0]))
        paths = scene.compute_paths(los=False, diffraction=False, max_depth=1, method="exhaustive")
        paths.normalize_delays = False
        cir = paths.cir()
        
        # Since the material is not vacuum, the path has energy
        self.assertTrue(tf.reduce_sum(tf.squeeze(cir[0])) != 0)

        # Change asset radio material to vacuum
        scene = load_scene() # Load empty scene
        scene.tx_array = PlanarArray(1,1,0.5,0.5,"iso","V")
        scene.rx_array = PlanarArray(1,1,0.5,0.5,"iso","V")

        ## The test cube(s) are two 1x1x1m cubes, centered in (0,0,0), spaced by 1 m along the y-axis
        cube_edge_length = 1 
        cubes_separation = 1
        asset = AssetObject(name="reflector", filename=sionna.rt.asset_object.test_asset_1, position=[0,-cubes_separation,-cube_edge_length/2], radio_material='vacuum') 
        scene.add(asset)

        d0 = 100
        scene.add(Transmitter("tx", position=[0,+1,d0]))
        scene.add(Receiver("rx", position=[0,-1,d0]))
        paths = scene.compute_paths(los=False, diffraction=False, max_depth=1, method="exhaustive")
        paths.normalize_delays = False
        cir = paths.cir()
        
        # Since the material is now vacuum, the path has no energy
        self.assertTrue(tf.reduce_sum(tf.squeeze(cir[0])) == 0)