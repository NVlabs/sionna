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
import xml.etree.ElementTree as ET
import sys
import numpy as np

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
        

class TestAssetMisc(unittest.TestCase):
    """Miscellaneous tests related to asset's materials data structures and methods"""  
    def test_placeholder_material_replaced_by_asset_material(self):
        """Check that material placeholder replaced by asset material if asset material is defined (ie non placeholder).
        Check that replacing a placeholder material does not add a new material but rather transfer (i.e. assign) the properties 
        of the new material to the old, placeholder, one"""
        
        def remove_all_whitespace(s):
            return s.replace(" ", "").replace("\t", "").replace("\n", "").replace("\r", "")
        scene = load_scene(sionna.rt.scene.floor_wall)
        asset = AssetObject(name="asset_0", filename=sionna.rt.asset_object.test_asset_1) 
        asset_xml = asset.xml_tree.getroot()
        asset_bsdf_elts = asset_xml.findall('bsdf')
        for elt in asset_bsdf_elts:
            if elt.get('id') == 'mat-itu_metal':
                itu_metal_asset_xml = elt
                break

        # set the 'itu_metal' RadioMaterial from scene as a placeholder
        itu_metal_scene = scene.get('itu_metal')
        itu_metal_scene.is_placeholder = True
        itu_metal_scene_xml = itu_metal_scene.bsdf.xml_element
        self.assertTrue(itu_metal_scene.is_placeholder)
        scene.add(asset)
        

        # Since no material have been defined in the asset, the asset material should be the scene material, and the scene material 
        # shouldn't have changed (except the bsdf of the scene material which is updated by the non placeholder bsdf of the asset)
        self.assertTrue(scene.get('itu_metal')==itu_metal_scene)
        self.assertTrue(itu_metal_scene.relative_permittivity == 1.0)
        self.assertTrue(itu_metal_scene.bsdf.xml_element==itu_metal_asset_xml and itu_metal_scene.bsdf.xml_element != itu_metal_scene_xml)
        self.assertTrue(scene.get('asset_0_cube_1').radio_material==itu_metal_scene)

        # Specifying asset radio material
        scene = load_scene(sionna.rt.scene.floor_wall)
        itu_metal_asset = RadioMaterial('itu_metal',relative_permittivity=0.0)
        itu_metal_asset_xml = itu_metal_asset.bsdf.xml_element
        itu_metal_asset_xml_str = ET.tostring(itu_metal_asset_xml, encoding='unicode', method='xml')
        asset = AssetObject(name="asset_0", filename=sionna.rt.asset_object.test_asset_1,radio_material=itu_metal_asset) 

        # set the 'itu_metal' RadioMaterial from scene as a placeholder
        itu_metal_scene = scene.get('itu_metal')
        itu_metal_scene.is_placeholder = True
        itu_metal_scene_xml = itu_metal_scene.bsdf.xml_element
        itu_metal_scene_xml_str = ET.tostring(itu_metal_scene_xml, encoding='unicode')
        self.assertTrue(itu_metal_scene.is_placeholder)

        ref_obj = scene.get("floor")
        ref_obj_mi_shape = ref_obj.mi_shape
        scene.add(asset)

        new_itu_metal_scene = scene.get('itu_metal')
        new_itu_metal_scene_xml = new_itu_metal_scene.bsdf.xml_element
        new_itu_metal_scene_xml_str = ET.tostring(new_itu_metal_scene_xml, encoding='unicode')

        # Since a material have been defined in the asset, the scene material, whichwas a placeholder, should have been replace by the asset's material
        # Yet, the scene material object should be the same although its properties should have changed (the asset material have been assigned to the scene material)
        self.assertTrue(new_itu_metal_scene == itu_metal_scene) # the material object is the same it has just beeen updated
        self.assertFalse(new_itu_metal_scene.is_placeholder)
        self.assertTrue(new_itu_metal_scene.relative_permittivity==0.0) # The material properties have been updated according to asset RadioMaterial
        self.assertTrue(remove_all_whitespace(itu_metal_scene_xml_str) != remove_all_whitespace(new_itu_metal_scene_xml_str)) # The scene bsdf have been updated compared to before adding the asset
        self.assertTrue(remove_all_whitespace(new_itu_metal_scene_xml_str) == remove_all_whitespace(itu_metal_asset_xml_str))
        self.assertTrue(new_itu_metal_scene_xml != itu_metal_asset_xml)
        self.assertTrue(scene.get('asset_0_cube_1').radio_material==itu_metal_scene)

        # Check that the scene is automatically reloaded
        self.assertTrue(ref_obj == scene.get("floor"))
        self.assertTrue(ref_obj_mi_shape != scene.get("floor").mi_shape)


    def test_placeholder_bsdf_replaced_by_asset_bsdf(self):
        """Check that placeholder bsdf replaced by asset bsdf if asset bsdf is defined (ie non placeholder)"""
        def remove_all_whitespace(s):
            return s.replace(" ", "").replace("\t", "").replace("\n", "").replace("\r", "")
        scene = load_scene(sionna.rt.scene.floor_wall)
        asset = AssetObject(name="asset_0", filename=sionna.rt.asset_object.test_asset_1) 
        asset_xml = asset.xml_tree.getroot()
        asset_bsdf_elts = asset_xml.findall('bsdf')
        for elt in asset_bsdf_elts:
            if elt.get('id') == 'mat-itu_metal':
                itu_metal_asset_xml = elt
                itu_metal_asset_xml_str = ET.tostring(itu_metal_asset_xml, encoding='unicode')
                break

        # set the 'itu_metal' RadioMaterial BSDF from scene not as a placeholder
        itu_metal_scene = scene.get('itu_metal')
        itu_metal_scene.bsdf.is_placeholder = False
        itu_metal_scene_xml = itu_metal_scene.bsdf.xml_element
        itu_metal_scene_xml_str = ET.tostring(itu_metal_scene_xml, encoding='unicode')
        self.assertFalse(itu_metal_scene.is_placeholder)
        self.assertFalse(itu_metal_scene.bsdf.is_placeholder)
        self.assertTrue(itu_metal_scene.relative_permittivity==1.0)
        scene.add(asset)

        new_itu_metal_scene = scene.get('itu_metal')
        new_itu_metal_scene_xml = new_itu_metal_scene.bsdf.xml_element
        new_itu_metal_scene_xml_str = ET.tostring(new_itu_metal_scene_xml, encoding='unicode')

        # Since, nor the bsdf neither the material are placeholder, the asset should not remove the material and bsdf properties 
        self.assertTrue(new_itu_metal_scene == itu_metal_scene) # the material object is the same
        self.assertFalse(new_itu_metal_scene.is_placeholder)
        self.assertFalse(new_itu_metal_scene.bsdf.is_placeholder)
        self.assertTrue(new_itu_metal_scene.relative_permittivity==1.0) # The material properties are unchanged
        
        self.assertTrue(remove_all_whitespace(itu_metal_scene_xml_str) == remove_all_whitespace(new_itu_metal_scene_xml_str)) # The scene bsdf are unchanged
        self.assertTrue(remove_all_whitespace(new_itu_metal_scene_xml_str) != remove_all_whitespace(itu_metal_asset_xml_str))
        self.assertTrue(new_itu_metal_scene_xml != itu_metal_asset_xml)
        self.assertTrue(new_itu_metal_scene_xml == itu_metal_scene_xml)
        self.assertTrue(scene.get('asset_0_cube_1').radio_material==itu_metal_scene)   

        # set the 'itu_metal' RadioMaterial BSDF from scene as a placeholder
        scene = load_scene(sionna.rt.scene.floor_wall)
        asset = AssetObject(name="asset_0", filename=sionna.rt.asset_object.test_asset_1) 
        asset_xml = asset.xml_tree.getroot()
        asset_bsdf_elts = asset_xml.findall('bsdf')
        for elt in asset_bsdf_elts:
            if elt.get('id') == 'mat-itu_metal':
                itu_metal_asset_xml = elt
                itu_metal_asset_xml_str = ET.tostring(itu_metal_asset_xml, encoding='unicode')
                break
        itu_metal_scene = scene.get('itu_metal')
        #itu_metal_scene.bsdf.is_placeholder = True
        itu_metal_scene_xml = itu_metal_scene.bsdf.xml_element
        itu_metal_scene_xml_str = ET.tostring(itu_metal_scene_xml, encoding='unicode')
        self.assertFalse(itu_metal_scene.is_placeholder)
        self.assertTrue(itu_metal_scene.bsdf.is_placeholder) # by default the bsdf is placeholder 
        self.assertTrue(itu_metal_scene.relative_permittivity==1.0)

        ref_obj = scene.get("floor")
        ref_obj_mi_shape = ref_obj.mi_shape
        scene.add(asset)

        new_itu_metal_scene = scene.get('itu_metal')
        new_itu_metal_scene_xml = new_itu_metal_scene.bsdf.xml_element
        new_itu_metal_scene_xml_str = ET.tostring(new_itu_metal_scene_xml, encoding='unicode')

        # Since, the bsdf of the scene material is now a  placeholder, the asset should update bsdf properties 
        self.assertTrue(new_itu_metal_scene == itu_metal_scene) # the material object is the same
        self.assertFalse(new_itu_metal_scene.is_placeholder)
        # self.assertFalse(new_itu_metal_scene.bsdf.is_placeholder) # The BSDF is not placeholder now that the asset has been added 
        self.assertTrue(new_itu_metal_scene.relative_permittivity==1.0) # The material properties are unchanged
        
        self.assertTrue(remove_all_whitespace(itu_metal_scene_xml_str) != remove_all_whitespace(new_itu_metal_scene_xml_str)) # The scene bsdf has changed
        self.assertTrue(remove_all_whitespace(new_itu_metal_scene_xml_str) == remove_all_whitespace(itu_metal_asset_xml_str))
        self.assertTrue(new_itu_metal_scene_xml == itu_metal_asset_xml)
        self.assertTrue(new_itu_metal_scene_xml != itu_metal_scene_xml)
        self.assertTrue(scene.get('asset_0_cube_1').radio_material==itu_metal_scene) 

        # Check that the scene is automatically reloaded
        self.assertTrue(ref_obj == scene.get("floor"))
        self.assertTrue(ref_obj_mi_shape != scene.get("floor").mi_shape)

    def test_non_placeholder_material_replaced_by_asset_material_if_overwrite(self):
        """Check that material non-placeholder replaced by asset material when specified"""
        def remove_all_whitespace(s):
            return s.replace(" ", "").replace("\t", "").replace("\n", "").replace("\r", "")

        # Specifying asset radio material - No overwrite
        scene = load_scene(sionna.rt.scene.floor_wall)
        itu_metal_asset = RadioMaterial('itu_metal',relative_permittivity=0.0)
        itu_metal_asset_xml = itu_metal_asset.bsdf.xml_element
        itu_metal_asset_xml_str = ET.tostring(itu_metal_asset_xml, encoding='unicode', method='xml')
        asset = AssetObject(name="asset_0", filename=sionna.rt.asset_object.test_asset_1,radio_material=itu_metal_asset) 
        self.assertFalse(itu_metal_asset.is_placeholder)

        # The 'itu_metal' RadioMaterial from scene is not placeholder
        itu_metal_scene = scene.get('itu_metal')
        itu_metal_scene_xml = itu_metal_scene.bsdf.xml_element
        itu_metal_scene_xml_str = ET.tostring(itu_metal_scene_xml, encoding='unicode')
        self.assertFalse(itu_metal_scene.is_placeholder)

        # Since both the asset  and the scene material are non-placeholders adding asset to scene should raise an error.
        
        with self.assertRaises(ValueError) as context:
            scene.add(asset)
        self.assertEqual(str(context.exception), "Name 'itu_metal' is already used by another item of the scene")
        
        # Specifying asset radio material - Set asset overwrite_scene_radio_materials = True
        scene = load_scene(sionna.rt.scene.floor_wall)
        itu_metal_asset = RadioMaterial('itu_metal',relative_permittivity=0.0)
        itu_metal_asset_xml = itu_metal_asset.bsdf.xml_element
        itu_metal_asset_xml_str = ET.tostring(itu_metal_asset_xml, encoding='unicode', method='xml')
        asset = AssetObject(name="asset_0", filename=sionna.rt.asset_object.test_asset_1,radio_material=itu_metal_asset, overwrite_scene_radio_materials=True) 
        self.assertFalse(itu_metal_asset.is_placeholder)

        # The 'itu_metal' RadioMaterial from scene is not placeholder
        itu_metal_scene = scene.get('itu_metal')
        itu_metal_scene_xml = itu_metal_scene.bsdf.xml_element
        itu_metal_scene_xml_str = ET.tostring(itu_metal_scene_xml, encoding='unicode')
        self.assertFalse(itu_metal_scene.is_placeholder)

        # Both the asset and the scene material are non-placeholders, but the overwrite arg. from the asset is set to True
        # Thus, adding asset to scene should not raise an error.
        ref_obj = scene.get("floor")
        ref_obj_mi_shape = ref_obj.mi_shape
        scene.add(asset)
        
        new_itu_metal_scene = scene.get('itu_metal')
        new_itu_metal_scene_xml = new_itu_metal_scene.bsdf.xml_element
        new_itu_metal_scene_xml_str = ET.tostring(new_itu_metal_scene_xml, encoding='unicode')

        # Since the asset was set to overwrite existing scene material, the scene material should have been replaced by the asset's material
        # Yet, the scene material object should be the same although its properties should have changed (the asset material have been assigned to the scene material)
        self.assertTrue(new_itu_metal_scene == itu_metal_scene) # the material object is the same it has just beeen updated
        self.assertFalse(new_itu_metal_scene.is_placeholder)
        self.assertTrue(new_itu_metal_scene.relative_permittivity==0.0) # The material properties have been updated according to asset RadioMaterial
        self.assertTrue(remove_all_whitespace(itu_metal_scene_xml_str) != remove_all_whitespace(new_itu_metal_scene_xml_str)) # The scene bsdf have been updated compared to before adding the asset
        self.assertTrue(remove_all_whitespace(new_itu_metal_scene_xml_str) == remove_all_whitespace(itu_metal_asset_xml_str))
        self.assertTrue(new_itu_metal_scene_xml != itu_metal_asset_xml)
        self.assertTrue(scene.get('asset_0_cube_1').radio_material==itu_metal_scene)

        # Check that the scene is automatically reloaded
        self.assertTrue(ref_obj == scene.get("floor"))
        self.assertTrue(ref_obj_mi_shape != scene.get("floor").mi_shape)


    def test_non_placeholder_bsdf_replaced_by_asset_bsdf_if_overwrite(self):
        """Check that bsdf non-placeholder replaced by asset bsdf when specified"""
        def remove_all_whitespace(s):
            return s.replace(" ", "").replace("\t", "").replace("\n", "").replace("\r", "")
        scene = load_scene(sionna.rt.scene.floor_wall)
        asset = AssetObject(name="asset_0", filename=sionna.rt.asset_object.test_asset_1) 
        asset_xml = asset.xml_tree.getroot()
        asset_bsdf_elts = asset_xml.findall('bsdf')
        for elt in asset_bsdf_elts:
            if elt.get('id') == 'mat-itu_metal':
                itu_metal_asset_xml = elt
                itu_metal_asset_xml_str = ET.tostring(itu_metal_asset_xml, encoding='unicode')
                break

        # set the 'itu_metal' RadioMaterial BSDF from scene not as a placeholder
        itu_metal_scene = scene.get('itu_metal')
        itu_metal_scene.bsdf.is_placeholder = False
        itu_metal_scene_xml = itu_metal_scene.bsdf.xml_element
        itu_metal_scene_xml_str = ET.tostring(itu_metal_scene_xml, encoding='unicode')
        self.assertFalse(itu_metal_scene.is_placeholder)
        self.assertFalse(itu_metal_scene.bsdf.is_placeholder)
        self.assertTrue(itu_metal_scene.relative_permittivity==1.0)
        scene.add(asset)

        new_itu_metal_scene = scene.get('itu_metal')
        new_itu_metal_scene_xml = new_itu_metal_scene.bsdf.xml_element
        new_itu_metal_scene_xml_str = ET.tostring(new_itu_metal_scene_xml, encoding='unicode')

        # Since, nor the bsdf neither the material are placeholder, the asset should not remove the material and bsdf properties 
        self.assertTrue(new_itu_metal_scene == itu_metal_scene) # the material object is the same
        self.assertFalse(new_itu_metal_scene.is_placeholder)
        self.assertFalse(new_itu_metal_scene.bsdf.is_placeholder)
        self.assertTrue(new_itu_metal_scene.relative_permittivity==1.0) # The material properties are unchanged
        
        self.assertTrue(remove_all_whitespace(itu_metal_scene_xml_str) == remove_all_whitespace(new_itu_metal_scene_xml_str)) # The scene bsdf are unchanged
        self.assertTrue(remove_all_whitespace(new_itu_metal_scene_xml_str) != remove_all_whitespace(itu_metal_asset_xml_str))
        self.assertTrue(new_itu_metal_scene_xml != itu_metal_asset_xml)
        self.assertTrue(new_itu_metal_scene_xml == itu_metal_scene_xml)
        self.assertTrue(scene.get('asset_0_cube_1').radio_material==itu_metal_scene)   

        # Again, set the 'itu_metal' RadioMaterial BSDF from scene not as a placeholder, but set the asset to overwrite existing bsdf (even if placeholders)
        scene = load_scene(sionna.rt.scene.floor_wall)
        asset = AssetObject(name="asset_0", filename=sionna.rt.asset_object.test_asset_1,overwrite_scene_bsdfs=True) 
        asset_xml = asset.xml_tree.getroot()
        asset_bsdf_elts = asset_xml.findall('bsdf')
        for elt in asset_bsdf_elts:
            if elt.get('id') == 'mat-itu_metal':
                itu_metal_asset_xml = elt
                itu_metal_asset_xml_str = ET.tostring(itu_metal_asset_xml, encoding='unicode')
                break
        itu_metal_scene = scene.get('itu_metal')
        itu_metal_scene.bsdf.is_placeholder = False
        itu_metal_scene_bsdf = itu_metal_scene.bsdf
        itu_metal_scene_xml = itu_metal_scene.bsdf.xml_element
        itu_metal_scene_xml_str = ET.tostring(itu_metal_scene_xml, encoding='unicode')
        self.assertFalse(itu_metal_scene.is_placeholder)
        self.assertFalse(itu_metal_scene.bsdf.is_placeholder) 
        self.assertTrue(itu_metal_scene.relative_permittivity==1.0)

        ref_obj = scene.get("floor")
        ref_obj_mi_shape = ref_obj.mi_shape
        scene.add(asset)

        new_itu_metal_scene = scene.get('itu_metal')
        new_itu_metal_scene_xml = new_itu_metal_scene.bsdf.xml_element
        new_itu_metal_scene_xml_str = ET.tostring(new_itu_metal_scene_xml, encoding='unicode')

        # Since, the asset is configured to overwrite existing bsdf, the asset should update existing scene bsdf properties 
        self.assertTrue(new_itu_metal_scene == itu_metal_scene) # the material object is the same
        self.assertFalse(new_itu_metal_scene.is_placeholder)
        self.assertTrue(new_itu_metal_scene.bsdf == itu_metal_scene_bsdf) # The BSDF object is not changed simply updated
        self.assertTrue(new_itu_metal_scene.relative_permittivity==1.0) # The material properties are unchanged
        
        self.assertTrue(remove_all_whitespace(itu_metal_scene_xml_str) != remove_all_whitespace(new_itu_metal_scene_xml_str)) # The scene bsdf has changed
        self.assertTrue(remove_all_whitespace(new_itu_metal_scene_xml_str) == remove_all_whitespace(itu_metal_asset_xml_str))
        self.assertTrue(new_itu_metal_scene_xml == itu_metal_asset_xml)
        self.assertTrue(new_itu_metal_scene_xml != itu_metal_scene_xml)
        self.assertTrue(scene.get('asset_0_cube_1').radio_material==itu_metal_scene)  

        # Check that the scene is automatically reloaded
        self.assertTrue(ref_obj == scene.get("floor"))
        self.assertTrue(ref_obj_mi_shape != scene.get("floor").mi_shape)


    


    def test_asset_removal_keep_materials(self):
        """Check that asset removal does not delete material even when not used anymore"""
        scene = load_scene(sionna.rt.scene.floor_wall)
        asset_1 = AssetObject(name="asset_1", filename=sionna.rt.asset_object.test_asset_1) #"itu_metal", "itu_wood"
        asset_2 = AssetObject(name="asset_2", filename=sionna.rt.asset_object.test_asset_2) #"custom_rm_1" and "mat-custom_rm_2"
        custom_rm_3 = RadioMaterial('mat-custom_rm_3')
        asset_3 = AssetObject(name="asset_3", filename=sionna.rt.asset_object.test_asset_2, radio_material=custom_rm_3) 
        custom_rm_4 = RadioMaterial('custom_rm_4')
        asset_4 = AssetObject(name="asset_4", filename=sionna.rt.asset_object.test_asset_2, radio_material=custom_rm_4) 
        scene.add([asset_1,asset_2,asset_3,asset_4])
        for mat_name in scene.radio_materials:
            self.assertTrue(scene.get(mat_name).bsdf.name == f"mat-{mat_name}")
        self.assertTrue("custom_rm_1" in scene.radio_materials)
        self.assertTrue("custom_rm_2" in scene.radio_materials)
        self.assertTrue("mat-custom_rm_3" in scene.radio_materials)
        self.assertTrue("custom_rm_4" in scene.radio_materials)

        scene.remove([asset_1.name,asset_2.name,asset_3.name,asset_4.name])
        for mat_name in scene.radio_materials:
            self.assertTrue(scene.get(mat_name).bsdf.name == f"mat-{mat_name}")
        self.assertTrue("custom_rm_1" in scene.radio_materials)
        self.assertTrue("custom_rm_2" in scene.radio_materials)
        self.assertTrue("mat-custom_rm_3" in scene.radio_materials)
        self.assertTrue("custom_rm_4" in scene.radio_materials)

    

    
    def test_add_remove_asset_update_using_objects_mat_bsdf(self):
        """Chek that the using_objects list are updated as expected when adding or removing assets."""
        scene = load_scene(sionna.rt.scene.floor_wall)
        asset_0 = AssetObject("asset_0", sionna.rt.asset_object.test_asset_1)
        asset_1 = AssetObject("asset_1", sionna.rt.asset_object.test_asset_1)

        itu_wood = scene.get('itu_wood')
        itu_metal = scene.get('itu_metal')
        self.assertFalse(itu_wood.is_used)
        self.assertFalse(itu_metal.is_used)
        self.assertTrue(itu_wood.using_objects.numpy().tolist() == [])
        self.assertTrue(itu_metal.using_objects.numpy().tolist() == [])

        scene.add(asset_0)
        self.assertTrue(itu_wood.is_used)
        self.assertTrue(itu_metal.is_used)
        self.assertTrue(itu_wood.use_counter == 1)
        self.assertTrue(itu_metal.use_counter == 1)
        self.assertTrue(itu_wood.using_objects.numpy().tolist() == [asset_0.shapes["asset_0_cube_0"].object_id])
        self.assertTrue(itu_metal.using_objects.numpy().tolist() == [asset_0.shapes["asset_0_cube_1"].object_id])

        scene.add(asset_1)
        self.assertTrue(itu_wood.is_used)
        self.assertTrue(itu_metal.is_used)
        self.assertTrue(itu_wood.use_counter == 2)
        self.assertTrue(itu_metal.use_counter == 2)
        self.assertTrue(itu_wood.using_objects.numpy().tolist().sort() == [asset_0.shapes["asset_0_cube_0"].object_id,asset_1.shapes["asset_1_cube_0"].object_id].sort())
        self.assertTrue(itu_metal.using_objects.numpy().tolist().sort() == [asset_0.shapes["asset_0_cube_1"].object_id,asset_1.shapes["asset_1_cube_1"].object_id].sort())

        scene.remove("asset_1")
        self.assertTrue(itu_wood.is_used)
        self.assertTrue(itu_metal.is_used)
        self.assertTrue(itu_wood.use_counter == 1)
        self.assertTrue(itu_metal.use_counter == 1)
        self.assertTrue(itu_wood.using_objects.numpy().tolist() == [asset_0.shapes["asset_0_cube_0"].object_id])
        self.assertTrue(itu_metal.using_objects.numpy().tolist() == [asset_0.shapes["asset_0_cube_1"].object_id])

        scene.remove("asset_0")
        self.assertFalse(itu_wood.is_used)
        self.assertFalse(itu_metal.is_used)
        self.assertTrue(itu_wood.using_objects.numpy().tolist() == [])
        self.assertTrue(itu_metal.using_objects.numpy().tolist() == [])
    
    def test_asset_material_updated_on_scene_object_material_update(self):
        """ Check that when changing material properties of a SceneObject belonging to an asset update the asset_material property"""  
        # Single shape asset >>> replace old material by new one
        scene = load_scene(sionna.rt.scene.floor_wall)
        asset = AssetObject("asset_0", sionna.rt.asset_object.test_asset_3)
        self.assertTrue(asset.radio_material == None)

        scene.add(asset)
        
        itu_marble = scene.get('itu_marble')
        self.assertTrue(asset.radio_material == itu_marble)
        scene_obj = scene.get("asset_0_cube")
        self.assertTrue(scene_obj.radio_material == itu_marble)

        itu_metal = scene.get('itu_metal')
        scene_obj.radio_material = itu_metal
        self.assertTrue(asset.radio_material == itu_metal)
        scene_obj = scene.get("asset_0_cube")
        self.assertTrue(scene_obj.radio_material == itu_metal)

        # Multi-shape asset with single material to multiple material 
        # >>> replace the single material by None, since one all shapes does not have the same material.
        scene = load_scene(sionna.rt.scene.floor_wall)
        asset = AssetObject("asset_0", sionna.rt.asset_object.test_asset_1, radio_material="itu_metal")
        self.assertTrue(asset.radio_material == "itu_metal")

        scene.add(asset)
        
        itu_metal = scene.get('itu_metal')
        self.assertTrue(asset.radio_material == itu_metal)
        scene_obj_0 = scene.get("asset_0_cube_0")
        self.assertTrue(scene_obj_0.radio_material == itu_metal)
        scene_obj_1 = scene.get("asset_0_cube_1")
        self.assertTrue(scene_obj_1.radio_material == itu_metal)

        itu_wood = scene.get('itu_wood')
        scene_obj_0.radio_material = itu_wood
        self.assertTrue(asset.radio_material == None)
        scene_obj_0 = scene.get("asset_0_cube_0")
        self.assertTrue(scene_obj_0.radio_material == itu_wood)
        scene_obj_1 = scene.get("asset_0_cube_1")
        self.assertTrue(scene_obj_1.radio_material == itu_metal)

        # Multi-shape asset with multiple materials to single material>>> Replace None by the single material
        scene = load_scene(sionna.rt.scene.floor_wall)
        asset = AssetObject("asset_0", sionna.rt.asset_object.test_asset_1)
        self.assertTrue(asset.radio_material == None)

        scene.add(asset)
        
        itu_metal = scene.get('itu_metal')
        itu_wood = scene.get('itu_wood')
        self.assertTrue(asset.radio_material == None)
        scene_obj_0 = scene.get("asset_0_cube_0")
        self.assertTrue(scene_obj_0.radio_material == itu_wood)
        scene_obj_1 = scene.get("asset_0_cube_1")
        self.assertTrue(scene_obj_1.radio_material == itu_metal)

        scene_obj_1.radio_material = itu_wood
        self.assertTrue(asset.radio_material == itu_wood)
        scene_obj_0 = scene.get("asset_0_cube_0")
        self.assertTrue(scene_obj_0.radio_material == itu_wood)
        scene_obj_1 = scene.get("asset_0_cube_1")
        self.assertTrue(scene_obj_1.radio_material == itu_wood)

    def test_reload_scene_keep_asset_properties(self):
        """Check that when the scene is reloaded, the properties of the asset object are kept, 
        even if they were previously changed by user (e.g. position, radio material.radio_material), 
        aswell as the asset constituent"""
        scene = load_scene(sionna.rt.scene.floor_wall)
        asset = AssetObject(name="asset_0", filename=sionna.rt.asset_object.test_asset_1, radio_material="itu_metal")
        
        self.assertTrue(np.equal(asset.position, [0,0,0]).all())
        self.assertTrue(np.equal(asset.orientation, [0,0,0]).all())
        self.assertEqual(asset.radio_material, 'itu_metal')

        scene.add(asset)

        epsilon = 1e-5

        self.assertTrue(tf.reduce_all(tf.math.less_equal(tf.math.abs(asset.position - [0,0,0]), epsilon)))
        self.assertTrue(tf.reduce_all(tf.math.less_equal(tf.math.abs(asset.shapes['asset_0_cube_0'].position - [0,-1,0]),epsilon)))
        self.assertTrue(tf.reduce_all(tf.math.less_equal(tf.math.abs(asset.shapes['asset_0_cube_1'].position - [0,+1,0]),epsilon)))

        self.assertTrue(tf.reduce_all(tf.math.less_equal(tf.math.abs(asset.orientation - [0,0,0]), epsilon)))
        self.assertTrue(tf.reduce_all(tf.math.less_equal(tf.math.abs(asset.shapes['asset_0_cube_0'].orientation - [0,0,0]),epsilon)))
        self.assertTrue(tf.reduce_all(tf.math.less_equal(tf.math.abs(asset.shapes['asset_0_cube_1'].orientation - [0,0,0]),epsilon)))
        
        itu_metal = scene.get('itu_metal')
        self.assertEqual(asset.radio_material, itu_metal)
        self.assertEqual(asset.shapes['asset_0_cube_0'].radio_material, itu_metal)
        self.assertEqual(asset.shapes['asset_0_cube_1'].radio_material, itu_metal)

        # Changing whole asset properties
        asset.position += [5,2,1]
        asset.orientation += [PI / 2, 0, 0]
        asset.radio_material = 'itu_wood'

        self.assertTrue(tf.reduce_all(tf.math.less_equal(tf.math.abs(asset.position - [5,2,1]), epsilon)))
        self.assertTrue(tf.reduce_all(tf.math.less_equal(tf.math.abs(asset.shapes['asset_0_cube_0'].position - [6,+2,1]),epsilon)))
        self.assertTrue(tf.reduce_all(tf.math.less_equal(tf.math.abs(asset.shapes['asset_0_cube_1'].position - [4,+2,1]),epsilon)))

        self.assertTrue(tf.reduce_all(tf.math.less_equal(tf.math.abs(asset.orientation - [PI/2,0,0]), epsilon)))
        self.assertTrue(tf.reduce_all(tf.math.less_equal(tf.math.abs(asset.shapes['asset_0_cube_0'].orientation - [PI/2,0,0]),epsilon)))
        self.assertTrue(tf.reduce_all(tf.math.less_equal(tf.math.abs(asset.shapes['asset_0_cube_1'].orientation - [PI/2,0,0]),epsilon)))
        
        itu_wood = scene.get('itu_wood')
        self.assertEqual(asset.radio_material, itu_wood)
        self.assertEqual(asset.shapes['asset_0_cube_0'].radio_material, itu_wood)
        self.assertEqual(asset.shapes['asset_0_cube_1'].radio_material, itu_wood)

        # Reload scene
        scene.reload()

        self.assertTrue(tf.reduce_all(tf.math.less_equal(tf.math.abs(asset.position - [5,2,1]), epsilon)))
        self.assertTrue(tf.reduce_all(tf.math.less_equal(tf.math.abs(asset.shapes['asset_0_cube_0'].position - [6,+2,1]),epsilon)))
        self.assertTrue(tf.reduce_all(tf.math.less_equal(tf.math.abs(asset.shapes['asset_0_cube_1'].position - [4,+2,1]),epsilon)))

        self.assertTrue(tf.reduce_all(tf.math.less_equal(tf.math.abs(asset.orientation - [PI/2,0,0]), epsilon)))
        self.assertTrue(tf.reduce_all(tf.math.less_equal(tf.math.abs(asset.shapes['asset_0_cube_0'].orientation - [PI/2,0,0]),epsilon)))
        self.assertTrue(tf.reduce_all(tf.math.less_equal(tf.math.abs(asset.shapes['asset_0_cube_1'].orientation - [PI/2,0,0]),epsilon)))
        
        itu_wood = scene.get('itu_wood')
        self.assertEqual(asset.radio_material, itu_wood)
        self.assertEqual(asset.shapes['asset_0_cube_0'].radio_material, itu_wood)
        self.assertEqual(asset.shapes['asset_0_cube_1'].radio_material, itu_wood)

    def test_reload_scene_keep_asset_scene_object_properties(self):
        """Check that when the scene is reloaded, the properties of the asset object are kept, 
        even if the scene object of the asset were previously changed by user individually (e.g. position, radio material.radio_material)"""
        scene = load_scene(sionna.rt.scene.floor_wall)
        asset = AssetObject(name="asset_0", filename=sionna.rt.asset_object.test_asset_1, radio_material="itu_metal")
        
        self.assertTrue(np.equal(asset.position, [0,0,0]).all())
        self.assertTrue(np.equal(asset.orientation, [0,0,0]).all())
        self.assertEqual(asset.radio_material, 'itu_metal')

        scene.add(asset)

        epsilon = 1e-5

        self.assertTrue(tf.reduce_all(tf.math.less_equal(tf.math.abs(asset.position - [0,0,0]), epsilon)))
        self.assertTrue(tf.reduce_all(tf.math.less_equal(tf.math.abs(asset.shapes['asset_0_cube_0'].position - [0,-1,0]),epsilon)))
        self.assertTrue(tf.reduce_all(tf.math.less_equal(tf.math.abs(asset.shapes['asset_0_cube_1'].position - [0,+1,0]),epsilon)))

        self.assertTrue(tf.reduce_all(tf.math.less_equal(tf.math.abs(asset.orientation - [0,0,0]), epsilon)))
        self.assertTrue(tf.reduce_all(tf.math.less_equal(tf.math.abs(asset.shapes['asset_0_cube_0'].orientation - [0,0,0]),epsilon)))
        self.assertTrue(tf.reduce_all(tf.math.less_equal(tf.math.abs(asset.shapes['asset_0_cube_1'].orientation - [0,0,0]),epsilon)))
        
        itu_metal = scene.get('itu_metal')
        self.assertEqual(asset.radio_material, itu_metal)
        self.assertEqual(asset.shapes['asset_0_cube_0'].radio_material, itu_metal)
        self.assertEqual(asset.shapes['asset_0_cube_1'].radio_material, itu_metal)

        # Changing whole asset properties
        asset.position += [5,2,1]
        asset.orientation += [PI / 2, 0, 0]
        asset.radio_material = 'itu_wood'

        self.assertTrue(tf.reduce_all(tf.math.less_equal(tf.math.abs(asset.position - [5,2,1]), epsilon)))
        self.assertTrue(tf.reduce_all(tf.math.less_equal(tf.math.abs(asset.shapes['asset_0_cube_0'].position - [6,+2,1]),epsilon)))
        self.assertTrue(tf.reduce_all(tf.math.less_equal(tf.math.abs(asset.shapes['asset_0_cube_1'].position - [4,+2,1]),epsilon)))

        self.assertTrue(tf.reduce_all(tf.math.less_equal(tf.math.abs(asset.orientation - [PI/2,0,0]), epsilon)))
        self.assertTrue(tf.reduce_all(tf.math.less_equal(tf.math.abs(asset.shapes['asset_0_cube_0'].orientation - [PI/2,0,0]),epsilon)))
        self.assertTrue(tf.reduce_all(tf.math.less_equal(tf.math.abs(asset.shapes['asset_0_cube_1'].orientation - [PI/2,0,0]),epsilon)))
        
        itu_wood = scene.get('itu_wood')
        self.assertEqual(asset.radio_material, itu_wood)
        self.assertEqual(asset.shapes['asset_0_cube_0'].radio_material, itu_wood)
        self.assertEqual(asset.shapes['asset_0_cube_1'].radio_material, itu_wood)

        # Changing individual asset's scene object properties
        asset.shapes['asset_0_cube_0'].position += [-2,+3,-5]
        asset.shapes['asset_0_cube_0'].orientation += [0, PI/2, 0]
        asset.shapes['asset_0_cube_0'].radio_material = 'itu_glass'

        self.assertTrue(tf.reduce_all(tf.math.less_equal(tf.math.abs(asset.position - [5,2,1]), epsilon)))
        self.assertTrue(tf.reduce_all(tf.math.less_equal(tf.math.abs(asset.shapes['asset_0_cube_0'].position - [4,+5,-4]),epsilon)))

        self.assertTrue(tf.reduce_all(tf.math.less_equal(tf.math.abs(asset.orientation - [PI/2,0,0]), epsilon)))
        self.assertTrue(tf.reduce_all(tf.math.less_equal(tf.math.abs(asset.shapes['asset_0_cube_0'].orientation - [PI/2,PI/2,0]),epsilon)))
        
        itu_glass = scene.get('itu_glass')
        self.assertEqual(asset.radio_material, None)
        self.assertEqual(asset.shapes['asset_0_cube_0'].radio_material, itu_glass)

        # Reload scene
        scene.reload()

        self.assertTrue(tf.reduce_all(tf.math.less_equal(tf.math.abs(asset.position - [5,2,1]), epsilon)))
        self.assertTrue(tf.reduce_all(tf.math.less_equal(tf.math.abs(asset.shapes['asset_0_cube_0'].position - [4,+5,-4]),epsilon)))
        self.assertTrue(tf.reduce_all(tf.math.less_equal(tf.math.abs(asset.shapes['asset_0_cube_1'].position - [4,+2,1]),epsilon)))

        self.assertTrue(tf.reduce_all(tf.math.less_equal(tf.math.abs(asset.orientation - [PI/2,0,0]), epsilon)))
        self.assertTrue(tf.reduce_all(tf.math.less_equal(tf.math.abs(asset.shapes['asset_0_cube_0'].orientation - [PI/2,PI/2,0]),epsilon)))
        self.assertTrue(tf.reduce_all(tf.math.less_equal(tf.math.abs(asset.shapes['asset_0_cube_1'].orientation - [PI/2,0,0]),epsilon)))
        
        itu_wood = scene.get('itu_wood')
        itu_glass = scene.get('itu_glass')
        self.assertEqual(asset.radio_material, None)
        self.assertEqual(asset.shapes['asset_0_cube_0'].radio_material, itu_glass)
        self.assertEqual(asset.shapes['asset_0_cube_1'].radio_material, itu_wood)
        
    def test_update_asset_material_object_using(self):
        """Check that changing the material of an AssetObject or a SceneObject, update the object_using sets of the material and its bsdf accordingly"""
        scene = load_scene(sionna.rt.scene.floor_wall)
        asset_0 = AssetObject(name="asset_0", filename=sionna.rt.asset_object.test_asset_1, radio_material="itu_metal")
        asset_1 = AssetObject(name="asset_1", filename=sionna.rt.asset_object.test_asset_1, radio_material="itu_concrete")

        scene.add([asset_0, asset_1])

        concrete = scene.get('itu_concrete')
        brick = scene.get('itu_brick')
        metal = scene.get('itu_metal')

        self.assertTrue(concrete.is_used)
        self.assertTrue(brick.is_used)
        self.assertTrue(metal.is_used)

        concrete_obj = concrete.using_objects.numpy().tolist() 
        brick_obj = brick.using_objects.numpy().tolist() 
        metal_obj = metal.using_objects.numpy().tolist() 

        self.assertEqual(len(concrete_obj), 3)
        self.assertEqual(len(brick_obj), 1)
        self.assertEqual(len(metal_obj), 2)

        self.assertTrue([scene.get("floor").object_id, scene.get("asset_1_cube_0").object_id, scene.get("asset_1_cube_1").object_id].sort() == concrete_obj.sort())
        self.assertTrue([scene.get("wall").object_id] == brick_obj)
        self.assertTrue([scene.get("asset_0_cube_0").object_id, scene.get("asset_0_cube_1").object_id].sort() == metal_obj.sort())

        asset_0.radio_material = "itu_glass"
        asset_1.radio_material = "itu_wood"
        scene.get("floor").radio_material = "itu_brick"
        glass = scene.get('itu_glass')
        wood = scene.get('itu_wood')

        self.assertFalse(concrete.is_used)
        self.assertTrue(brick.is_used)
        self.assertFalse(metal.is_used)
        self.assertTrue(glass.is_used)
        self.assertTrue(wood.is_used)
        

        concrete_obj = concrete.using_objects.numpy().tolist() 
        brick_obj = brick.using_objects.numpy().tolist() 
        metal_obj = metal.using_objects.numpy().tolist() 
        glass_obj = glass.using_objects.numpy().tolist() 
        wood_obj = wood.using_objects.numpy().tolist() 

        self.assertEqual(len(concrete_obj), 0)
        self.assertEqual(len(brick_obj), 2)
        self.assertEqual(len(metal_obj), 0)
        self.assertEqual(len(glass_obj), 2)
        self.assertEqual(len(wood_obj), 2)

        self.assertTrue([scene.get("asset_1_cube_0").object_id, scene.get("asset_1_cube_1").object_id].sort() == glass_obj.sort())
        self.assertTrue([scene.get("floor").object_id, scene.get("wall").object_id].sort() == brick_obj.sort())
        self.assertTrue([scene.get("asset_0_cube_0").object_id, scene.get("asset_0_cube_1").object_id].sort() == wood_obj.sort())    


    def test_original_asset_bsdf_are_stored(self):
        """Test showing that the asset's original bsdfs are propely stored in the asset data-structure"""
        # single-material asset
        asset = AssetObject(name="asset_0", filename=sionna.rt.asset_object.test_asset_3) 
        asset_xml_root = asset._xml_tree.getroot()
        bsdfs_in_asset_xml = asset_xml_root.findall('bsdf')
        bsdf_ids = [bsdf.get('id') for bsdf in bsdfs_in_asset_xml]

        for id in bsdf_ids:
            self.assertTrue(id in asset.original_bsdfs)
            for bsdf in bsdfs_in_asset_xml:
                if bsdf.get('id') == id:
                    self.assertTrue(bsdf,asset.original_bsdfs[id].xml_element) 
        self.assertEqual([b for b in asset.original_bsdfs].sort(),['mat-itu_marble'].sort()) 
                    

        # Multi-material asset
        asset = AssetObject(name="asset_0", filename=sionna.rt.asset_object.test_asset_1) 
        asset_xml_root = asset._xml_tree.getroot()
        bsdfs_in_asset_xml = asset_xml_root.findall('bsdf')
        bsdf_ids = [bsdf.get('id') for bsdf in bsdfs_in_asset_xml]

        for id in bsdf_ids:
            self.assertTrue(id in asset.original_bsdfs)
            for bsdf in bsdfs_in_asset_xml:
                if bsdf.get('id') == id:
                    self.assertTrue(bsdf,asset.original_bsdfs[id].xml_element) 
        self.assertEqual([b for b in asset.original_bsdfs].sort(),['mat-itu_metal','mat-itu_wood'].sort()) 