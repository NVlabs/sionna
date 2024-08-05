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

class TestBSDFUpdate(unittest.TestCase):
    """Tests related to updating the BSDF of a material"""

    def test_set_bsdf_when_placeholder(self):
        """Test setting the RGB property of a placeholder BSDF"""
        placeholder_bsdf = BSDF(name="placeholder_bsdf")
        placeholder_bsdf.is_placeholder = True
        self.assertTrue(placeholder_bsdf.is_placeholder)

        placeholder_bsdf_rgb = placeholder_bsdf.rgb
        placeholder_bsdf.rgb = (0.5, 0.5, 0.5)
        self.assertFalse(placeholder_bsdf.is_placeholder)
        self.assertTrue(np.equal(placeholder_bsdf.rgb, (0.5, 0.5, 0.5)).all())
        self.assertFalse(np.equal(placeholder_bsdf.rgb, placeholder_bsdf_rgb).all())

    def test_set_bsdf_when_not_placeholder(self):
        """Test setting the RGB property of a non-placeholder BSDF"""
        non_placeholder_bsdf = BSDF(name="non_placeholder_bsdf")
        self.assertFalse(non_placeholder_bsdf.is_placeholder)

        non_placeholder_bsdf_rgb = non_placeholder_bsdf.rgb
        non_placeholder_bsdf.rgb = (0.5, 0.5, 0.5)
        self.assertFalse(non_placeholder_bsdf.is_placeholder)
        self.assertTrue(np.equal(non_placeholder_bsdf.rgb, (0.5, 0.5, 0.5)).all())
        self.assertFalse(np.equal(non_placeholder_bsdf.rgb, non_placeholder_bsdf_rgb).all())

    def test_update_bsdf_in_scene_when_placeholder(self):
        """Test updating the BSDF of a material in the scene when the BSDF is a placeholder"""
        scene = load_scene(sionna.rt.scene.floor_wall)
        placeholder_bsdf = scene.get("itu_glass").bsdf
        self.assertTrue(placeholder_bsdf.is_placeholder)

        bsdfs_in_root = scene._xml_tree.getroot().findall('bsdf')
        for bsdf in bsdfs_in_root:
            if bsdf.get('id') == 'mat-itu_glass':
                break
        rgb = [float(x) for x in bsdf.find('rgb').get('value').split()]
        self.assertFalse(np.equal(rgb, (0.5, 0.5, 0.5)).all())

        placeholder_bsdf_rgb = placeholder_bsdf.rgb
        placeholder_bsdf.rgb = (0.5, 0.5, 0.5)
        self.assertFalse(placeholder_bsdf.is_placeholder)
        self.assertTrue(np.equal(placeholder_bsdf.rgb, (0.5, 0.5, 0.5)).all())
        self.assertFalse(np.equal(placeholder_bsdf.rgb, placeholder_bsdf_rgb).all())

        bsdfs_in_root = scene._xml_tree.getroot().findall('bsdf')
        for bsdf in bsdfs_in_root:
            if bsdf.get('id') == 'mat-itu_glass':
                break
        rgb = [float(x) for x in bsdf.find('rgb').get('value').split()]
        self.assertTrue(np.equal(rgb, (0.5, 0.5, 0.5)).all())


    def test_update_bsdf_in_scene_when_not_placeholder(self):
        """Test updating the BSDF of a material in the scene when the BSDF is not a placeholder"""
        scene = load_scene(sionna.rt.scene.floor_wall)
        non_placeholder_bsdf = scene.get("itu_concrete").bsdf
        self.assertFalse(non_placeholder_bsdf.is_placeholder)

        bsdfs_in_root = scene._xml_tree.getroot().findall('bsdf')
        for bsdf in bsdfs_in_root:
            if bsdf.get('id') == 'mat-itu_concrete':
                break
        rgb = [float(x) for x in bsdf.find('bsdf').find('rgb').get('value').split()]
        self.assertFalse(np.equal(rgb, (0.5, 0.5, 0.5)).all())


        non_placeholder_bsdf_rgb = non_placeholder_bsdf.rgb
        non_placeholder_bsdf.rgb = (0.5, 0.5, 0.5)
        self.assertFalse(non_placeholder_bsdf.is_placeholder)
        self.assertTrue(np.equal(non_placeholder_bsdf.rgb, (0.5, 0.5, 0.5)).all())
        self.assertFalse(np.equal(non_placeholder_bsdf.rgb, non_placeholder_bsdf_rgb).all())

        bsdfs_in_root = scene._xml_tree.getroot().findall('bsdf')
        for bsdf in bsdfs_in_root:
            if bsdf.get('id') == 'mat-itu_concrete':
                break
        rgb = [float(x) for x in bsdf.find('rgb').get('value').split()]
        self.assertTrue(np.equal(rgb, (0.5, 0.5, 0.5)).all())

    def test_scene_reload_on_placeholder_bsdf_update(self):
        """Test that the scene is reloaded when the BSDF is updated"""
        scene = load_scene(sionna.rt.scene.floor_wall)
        placeholder_bsdf = scene.get("itu_glass").bsdf
        self.assertTrue(placeholder_bsdf.is_placeholder)
        ref_obj = scene.get("floor").mi_shape
        placeholder_bsdf.rgb = (0.5, 0.5, 0.5)
        self.assertNotEqual(ref_obj, scene.get("floor").mi_shape) 

    def test_scene_reload_on_non_placeholder_bsdf_update(self):
        """Test that the scene is reloaded when the BSDF is updated"""
        scene = load_scene(sionna.rt.scene.floor_wall)
        non_placeholder_bsdf = scene.get("itu_concrete").bsdf
        self.assertFalse(non_placeholder_bsdf.is_placeholder)
        ref_obj = scene.get("floor").mi_shape
        non_placeholder_bsdf.rgb = (0.5, 0.5, 0.5)
        self.assertNotEqual(ref_obj, scene.get("floor").mi_shape) 



class TestBSDF(unittest.TestCase):
    """Tests related to the BSDF class"""

    def test_bsdf_modification_update_shape_bsdf(self):
        """Check that the modification/setup of a material and/or bsdf leads to the update of the corresponding shape's bsdf in the XML file."""
        scene = load_scene(sionna.rt.scene.floor_wall)

        # Add asset with mat
        asset = AssetObject("asset_0", sionna.rt.asset_object.test_asset_1, radio_material='itu_metal')
        scene.add(asset)

        root = scene._xml_tree.getroot()
        elements_in_root = root.findall('shape')  
        for elt in elements_in_root:
            if id == elt.get('id') == "mesh-asset_0_cube_0":
                break

        bsdf = elt.find('ref').get('id')
        self.assertEqual(bsdf, "mat-itu_metal")

        for elt in elements_in_root:
            if id == elt.get('id') == "mesh-asset_0_cube_1":
                break

        bsdf = elt.find('ref').get('id')
        self.assertEqual(bsdf, "mat-itu_metal")

        # Change scene object material
        scene.get("asset_0_cube_0").radio_material = 'itu_wood'
        for elt in elements_in_root:
            if elt.get('id') == "mesh-asset_0_cube_0":
                break

        bsdf = elt.find('ref').get('id')
        self.assertEqual(bsdf, "mat-itu_wood")

        scene.get("asset_0_cube_1").radio_material = 'itu_glass'
        for elt in elements_in_root:
            if elt.get('id') == "mesh-asset_0_cube_1":
                break

        bsdf = elt.find('ref').get('id')
        self.assertEqual(bsdf, "mat-itu_glass")

        

    def test_itu_materials_bsdfs_are_placeholders(self):
        """Test showing that Sionna base materials'bsdfs are placeholders, except when they are defined in the scene XML file"""
        scene = load_scene() # no material defined in the XML file
        for mat in scene.radio_materials:
            mat = scene.get(mat)
            if mat.name[:3] == 'itu':
                if mat.is_used:
                    self.assertTrue(not mat.bsdf.is_placeholder)
                else:
                    self.assertTrue(mat.bsdf.is_placeholder)

        scene = load_scene(sionna.rt.scene.floor_wall) # material 'itu_concrete' and 'itu_brick' are defined in the XML file
        for mat in scene.radio_materials:
            mat = scene.get(mat)
            if mat.name == 'itu_concrete' or mat.name == 'itu_brick':
                self.assertTrue(mat.is_used)
            if mat.name[:3] == 'itu':
                if mat.is_used:
                    self.assertTrue(not mat.bsdf.is_placeholder)
                else:
                    self.assertTrue(mat.bsdf.is_placeholder)

    def test_rgb_setter_bsdf(self):
        """Check that the bsdf RGB setter works as expected"""
        bsdf = BSDF("bsdf")

        # Valid RGB triplet
        prev_rgb = bsdf.rgb
        new_rgb = (0.5,0.2,0.6)
        bsdf.rgb = new_rgb
        self.assertTrue(np.equal(bsdf.rgb, new_rgb).all())
        self.assertTrue(not np.equal(bsdf.rgb, prev_rgb).all())

        # Invalid format or values
        prev_rgb = bsdf.rgb
        new_rgb = (0.5,0.2,0.6,0.9)
        with self.assertRaises(TypeError) as context:
            bsdf.rgb = new_rgb
        self.assertEqual(str(context.exception), "`rgb` must be a list of 3 floats comprised between 0 and 1")

        new_rgb = (256,45,18)
        with self.assertRaises(TypeError) as context:
            bsdf.rgb = new_rgb
        self.assertEqual(str(context.exception), "`rgb` must be a list of 3 floats comprised between 0 and 1")

        new_rgb = "(0.5,0.2,0.6)"
        with self.assertRaises(TypeError) as context:
            bsdf.rgb = new_rgb
        self.assertEqual(str(context.exception), "`rgb` must be a list of 3 floats comprised between 0 and 1")

    def test_color_setter_bsdf(self):
        """Check that the bsdf color setter works as expected"""
        bsdf = BSDF("bsdf")

        # Valid RGB triplet
        prev_rgb = bsdf.rgb
        new_rgb = (0.5,0.2,0.6)
        bsdf.color = new_rgb
        self.assertEqual(bsdf.color, None)
        self.assertTrue(np.equal(bsdf.rgb, new_rgb).all())
        self.assertTrue(not np.equal(bsdf.rgb, prev_rgb).all())

        # Valid color name
        prev_rgb = bsdf.rgb
        new_color = 'Red'
        bsdf.color = new_color
        self.assertEqual(bsdf.color, 'red')
        self.assertTrue(np.equal(bsdf.rgb, (1,0,0)).all())
        self.assertTrue(not np.equal(bsdf.rgb, prev_rgb).all())

        # Invalid format or values
        prev_rgb = bsdf.rgb
        new_rgb = (0.5,0.2,0.6,0.9)
        with self.assertRaises(TypeError) as context:
            bsdf.color = new_rgb
        self.assertEqual(str(context.exception), "`color` must be a list of 3 `float` between 0 and 1, or a valid `str` color name.")

        new_rgb = (256,45,18)
        with self.assertRaises(TypeError) as context:
            bsdf.color = new_rgb
        self.assertEqual(str(context.exception), "`color` must be a list of 3 `float` between 0 and 1, or a valid `str` color name.")

        new_rgb = "(0.5,0.2,0.6)"
        with self.assertRaises(TypeError) as context:
            bsdf.color = new_rgb
        self.assertEqual(str(context.exception), "Unknown color name '(0.5,0.2,0.6)'.")

    def test_xml_element_setter_bsdf(self):
        """Check that the bsdf xml_element setter works as expected"""
        bsdf = BSDF("bsdf")

        # Valid BSDF Element tree
        xml_str = '<bsdf type="diffuse"><rgb value="0.9115233420769805 0.3130895838503488 0.269888101787389" name="reflectance" /></bsdf>'
        xml_element = ET.fromstring(xml_str)
        bsdf.xml_element = xml_element
        self.assertEqual(bsdf.xml_element, xml_element)

        # Invalid Element type
        with self.assertRaises(TypeError) as context:
            bsdf.xml_element = "not_an_element"
        self.assertEqual(str(context.exception), "`element` must be an ET.Element descriptor of a BSDF.")

        # Invalid root element
        xml_str = '<material><bsdf type="diffuse"><rgb value="0.9115233420769805 0.3130895838503488 0.269888101787389" name="reflectance" /></bsdf></material>'
        xml_element = ET.fromstring(xml_str)
        with self.assertRaises(ValueError) as context:
            bsdf.xml_element = xml_element
        self.assertEqual(str(context.exception), "The root element must be <bsdf>.")
    
    def test_set_bsdf(self):
        """Check that the bsdf init method works as expected"""
        
        # Name is set
        bsdf = BSDF("bsdf")
        self.assertEqual(bsdf.name, "bsdf")
        self.assertEqual(bsdf.color, None)
        self.assertEqual(len(bsdf.rgb), 3)

        # If xml_element is defined it will be used no matter the color argument
        xml_str = '<bsdf type="diffuse"><rgb value="0.9 0.3 0.2" name="reflectance" /></bsdf>'
        xml_element = ET.fromstring(xml_str)
        bsdf = BSDF("bsdf",xml_element=xml_element)

        self.assertEqual(bsdf.color, None)
        self.assertEqual(bsdf.rgb, None)
        self.assertEqual(bsdf.xml_element, xml_element)

        color = 'red'
        bsdf = BSDF("bsdf",xml_element=xml_element,color=color)

        self.assertEqual(bsdf.color, None)
        self.assertEqual(bsdf.rgb, None)
        self.assertEqual(bsdf.xml_element, xml_element)

        color = (1,1,1)
        bsdf = BSDF("bsdf",xml_element=xml_element,color=color)

        self.assertEqual(bsdf.color, None)
        self.assertEqual(bsdf.rgb, None)
        self.assertEqual(bsdf.xml_element, xml_element)
        
        # Valid RGB triplet
        color = (1,1,1)
        bsdf = BSDF("bsdf",color=color)
        self.assertEqual(bsdf.color, None)
        self.assertTrue(np.equal(bsdf.rgb, color).all())

        # Valid color name
        color = 'Red'
        bsdf = BSDF("bsdf",color=color)
        self.assertEqual(bsdf.color, 'red')
        self.assertTrue(np.equal(bsdf.rgb, (1,0,0)).all())

        # Invalid color or XML element
        color = (0.5,0.2,0.6,0.9)
        with self.assertRaises(TypeError) as context: 
            bsdf = BSDF("bsdf",color=color)
        self.assertEqual(str(context.exception), "`color` must be a list of 3 `float` between 0 and 1, or a valid `str` color name.")

        color = (256,45,18)
        with self.assertRaises(TypeError) as context:
            bsdf = BSDF("bsdf",color=color)
        self.assertEqual(str(context.exception), "`color` must be a list of 3 `float` between 0 and 1, or a valid `str` color name.")

        color = "(0.5,0.2,0.6)"
        with self.assertRaises(TypeError) as context:
            bsdf = BSDF("bsdf",color=color)
        self.assertEqual(str(context.exception), "Unknown color name '(0.5,0.2,0.6)'.")

class TestRadioMaterial(unittest.TestCase):
    """Tests related to the RadioMaterial class"""

    # def setUp(self):
    #     self.scene = load_scene(sionna.rt.scene.floor_wall)
    
    def test_itu_materials_are_not_placeholders(self):
        """Test showing that Sionna base materials are not placeholders"""
        scene = load_scene(sionna.rt.scene.floor_wall)
        for mat in scene.radio_materials:
            mat = scene.get(mat)
            if mat.name[:3] == 'itu':
                self.assertTrue(not mat.is_placeholder)

    def test_bsdf_name_is_consistent_with_mat_name(self):
        """Check that the name of bsdf is set according to the radio_material name, i.e. bsdf name is f'mat-{radio_material.name}'"""
        scene = load_scene(sionna.rt.scene.floor_wall)
        # The asset object uses bsdf with badly formated material "custom_rm_1" and "mat-custom_rm_2" 
        # >>> Check that when adding asset, name convention are respected both for bsdf and material
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

        root = scene._xml_tree.getroot()
        bsdfs_in_root = root.findall('bsdf')
        bsdfs_in_root = [bsdf.get('id') for bsdf in bsdfs_in_root]
        self.assertTrue("mat-custom_rm_1" in bsdfs_in_root)
        self.assertTrue("mat-custom_rm_2" in bsdfs_in_root)
        self.assertTrue("mat-mat-custom_rm_3" in bsdfs_in_root)
        self.assertTrue("mat-custom_rm_4" in bsdfs_in_root)

    def test_new_radio_material_assignation_to_scene_object(self):
        """Check that the assignation of a new radio_material (i.e. not present at scene init) to a scene object is working."""
        scene = load_scene(sionna.rt.scene.floor_wall)
        rm = RadioMaterial('new')
        scene.add(rm)
        scene_obj = scene.get('floor')
        scene_obj_mi_shape = scene_obj.mi_shape
        scene_obj.radio_material = rm

        concrete = scene.get('itu_concrete')
        self.assertTrue(scene_obj.radio_material == rm)
        self.assertTrue(scene_obj.object_id not in concrete.using_objects)
        self.assertTrue(scene_obj.object_id in rm.using_objects)

        scene.reload()
        new_scene_obj = scene.get('floor')
        self.assertTrue(new_scene_obj.radio_material == rm)
        self.assertFalse(new_scene_obj.object_id in concrete.using_objects)
        self.assertTrue(new_scene_obj.object_id in rm.using_objects)
        self.assertTrue(new_scene_obj is scene_obj)
        self.assertFalse(new_scene_obj.mi_shape is scene_obj_mi_shape)

class TestObjectUsingMatBSDFSync(unittest.TestCase):
    """Tests related to synchronization of objects_using between material and BSDF"""

    def setUp(self):
        self.scene = load_scene(sionna.rt.scene.floor_wall)

    def test_initial_state_of_materials_and_bsdfs(self):
        """Test initial state of materials and their BSDFs"""
        itu_concrete = self.scene.get('itu_concrete')
        itu_brick = self.scene.get('itu_brick')

        self.assertTrue(itu_concrete.is_used)
        self.assertTrue(itu_concrete.bsdf.is_used)
        self.assertTrue(itu_brick.is_used)
        self.assertTrue(itu_brick.bsdf.is_used)
        self.assertEqual(itu_concrete.using_objects.numpy().tolist(), itu_concrete.bsdf.using_objects.numpy().tolist())
        self.assertEqual(itu_brick.using_objects.numpy().tolist(), itu_brick.bsdf.using_objects.numpy().tolist())

    def test_assigning_radio_material(self):
        """Test assigning one RadioMaterial to another. The assign method should not copy the object_using from one mat to the other since they remains distinct materials."""
        itu_concrete = self.scene.get('itu_concrete')
        itu_brick = self.scene.get('itu_brick')

        itu_brick.assign(itu_concrete)

        new_itu_concrete = self.scene.get('itu_concrete')
        new_itu_brick = self.scene.get('itu_brick')

        self.assertTrue(new_itu_concrete.is_used)
        self.assertTrue(new_itu_brick.is_used)
        self.assertEqual(new_itu_concrete.using_objects.numpy().tolist(), new_itu_concrete.bsdf.using_objects.numpy().tolist())
        self.assertEqual(new_itu_brick.using_objects.numpy().tolist(), new_itu_brick.bsdf.using_objects.numpy().tolist())

        self.assertEqual(new_itu_concrete.using_objects.numpy().tolist(), itu_concrete.using_objects.numpy().tolist())
        self.assertEqual(new_itu_brick.using_objects.numpy().tolist(), itu_brick.using_objects.numpy().tolist())
        self.assertEqual(new_itu_concrete.bsdf.using_objects.numpy().tolist(), itu_concrete.bsdf.using_objects.numpy().tolist())
        self.assertEqual(new_itu_brick.bsdf.using_objects.numpy().tolist(), itu_brick.bsdf.using_objects.numpy().tolist())

    def test_assigning_bsdf_to_material_bsdf(self):
        """Test assigning a non-used BSDF to a material's bsdf. This should not copy the object_using from one BSDF to the other since they remains distinct BSDF when using assign method.
        The assign method from BSDF class, contrarily to the RadioMaterial class, triggers the reload of the scene by default."""
        bsdf = BSDF("bsdf")
        itu_concrete = self.scene.get('itu_concrete')
        itu_concrete_obj_using = itu_concrete.using_objects.numpy().tolist() 

        self.assertFalse(bsdf.is_used)
        self.assertEqual(bsdf.using_objects.numpy().tolist(),[])

        itu_concrete.bsdf.assign(bsdf)

        new_itu_concrete = self.scene.get('itu_concrete')

        self.assertTrue(new_itu_concrete is itu_concrete)
        self.assertTrue(new_itu_concrete.is_used)
        self.assertTrue(new_itu_concrete.bsdf.is_used)
        self.assertFalse(bsdf.is_used)
        self.assertEqual(bsdf.using_objects.numpy().tolist(),[])
        self.assertEqual(new_itu_concrete.using_objects.numpy().tolist(), new_itu_concrete.bsdf.using_objects.numpy().tolist())
        self.assertNotEqual(new_itu_concrete.using_objects.numpy().tolist(),bsdf.using_objects.numpy().tolist())
        self.assertNotEqual(new_itu_concrete.using_objects.numpy().tolist(),itu_concrete_obj_using)

    def test_assigning_used_bsdf_to_material_bsdf(self):
        """Test assigning a used BSDF to a material's bsdf"""
        itu_concrete = self.scene.get('itu_concrete')
        itu_concrete_bsdf = itu_concrete.bsdf
        itu_brick = self.scene.get('itu_brick')
        itu_brick_bsdf = itu_brick.bsdf

        itu_concrete = self.scene.get('itu_concrete')   
        itu_concrete_bsdf = itu_concrete.bsdf   
        itu_concrete_obj_using = itu_concrete.using_objects.numpy().tolist() 
        itu_brick = self.scene.get('itu_brick') 
        itu_brick_bsdf = itu_brick.bsdf  
        itu_brick_obj_using = itu_brick.using_objects.numpy().tolist() 

        itu_brick.bsdf.assign(itu_concrete.bsdf)

        new_itu_concrete = self.scene.get('itu_concrete')
        new_itu_brick = self.scene.get('itu_brick')

        self.assertTrue(new_itu_concrete is itu_concrete)
        self.assertTrue(new_itu_brick is itu_brick)
        self.assertTrue(new_itu_concrete.bsdf is itu_concrete_bsdf)
        self.assertTrue(new_itu_brick.bsdf is itu_brick_bsdf)
        self.assertFalse(new_itu_brick.bsdf is new_itu_concrete.bsdf)
        self.assertTrue(new_itu_concrete.is_used)
        self.assertTrue(new_itu_brick.is_used)
        self.assertTrue(new_itu_concrete.bsdf.is_used)
        self.assertTrue(new_itu_brick.bsdf.is_used)
        self.assertEqual(new_itu_concrete.using_objects.numpy().tolist(), new_itu_concrete.bsdf.using_objects.numpy().tolist())
        self.assertEqual(new_itu_brick.using_objects.numpy().tolist(), new_itu_brick.bsdf.using_objects.numpy().tolist())
        self.assertNotEqual(new_itu_brick.using_objects.numpy().tolist(), new_itu_concrete.using_objects.numpy().tolist())
        self.assertNotEqual(new_itu_concrete.using_objects.numpy().tolist(), itu_concrete_obj_using) #Because of the scene reload triggered by BSDF assign()
        self.assertNotEqual(new_itu_brick.using_objects.numpy().tolist(), itu_brick_obj_using)

    def test_setting_new_bsdf_to_material_bsdf(self):
        """Test setting a new BSDF to a material's BSDF"""
        bsdf = BSDF("bsdf")
        itu_concrete = self.scene.get('itu_concrete')
        itu_concrete_bsdf = itu_concrete.bsdf 

        self.assertTrue(not bsdf.is_used)
        self.assertTrue(bsdf.using_objects.numpy().tolist() == [])

        itu_concrete.bsdf = bsdf

        new_itu_concrete = self.scene.get('itu_concrete')

        self.assertFalse(itu_concrete_bsdf.is_used)
        self.assertTrue(new_itu_concrete.is_used)
        self.assertTrue(new_itu_concrete.bsdf.is_used)
        self.assertTrue(bsdf.is_used)
        self.assertEqual(new_itu_concrete.using_objects.numpy().tolist(), bsdf.using_objects.numpy().tolist())
        self.assertEqual(new_itu_concrete.using_objects.numpy().tolist(), new_itu_concrete.bsdf.using_objects.numpy().tolist())

    def test_setting_used_bsdf_to_material_bsdf(self):
        """Check that assigning a used bsdf (i.e. BSDF used by another material too) to another radiomaterial triggers an error"""
        itu_concrete = self.scene.get('itu_concrete')
        itu_brick= self.scene.get('itu_brick')

        self.assertTrue(itu_concrete.bsdf.is_used)

        with self.assertRaises(ValueError) as context:
            itu_brick.bsdf = itu_concrete.bsdf
        self.assertEqual(str(context.exception), "Can't set an already used BSDF to another material. Prefer the assign method.")

        


class TestSceneReload(unittest.TestCase):
    """Tests related to the reloading of a scene"""

    def test_reload_scene_reset_material_object_using(self):
        """Check that the scene reload reset the object_using counters of the bsdfs and material, before instantianting new SceneObjects"""
        scene = load_scene(sionna.rt.scene.floor_wall)
        mat = scene.get("itu_concrete")
        bsdf = mat.bsdf
        mat_obj_using = mat.using_objects.numpy().tolist() 
        bsdf_obj_using = bsdf.using_objects.numpy().tolist() 

        self.assertTrue(mat_obj_using == bsdf_obj_using)
        self.assertTrue(mat.is_used)
        self.assertTrue(bsdf.is_used)

        scene.reload()

        new_mat = scene.get("itu_concrete")
        new_bsdf = new_mat.bsdf
        new_mat_obj_using = new_mat.using_objects.numpy().tolist() 
        new_bsdf_obj_using = new_bsdf.using_objects.numpy().tolist() 

        self.assertTrue(new_mat_obj_using == new_bsdf_obj_using)
        self.assertTrue(new_mat.is_used)
        self.assertTrue(new_bsdf.is_used)

        self.assertTrue(new_mat is mat)
        self.assertTrue(new_bsdf is bsdf)
        self.assertTrue(new_mat_obj_using != mat_obj_using)
        self.assertTrue(len(new_mat_obj_using) == len(mat_obj_using))

    def test_reload_scene_keep_scene_object_properties(self):
        """Check that when the scene is reloaded, the properties of the scene object, not related to any assets, are kept, 
        even if they were previously changed by user (e.g. position)"""
        
        # Manual reload
        scene = load_scene(sionna.rt.scene.floor_wall)

        # change scene SceneObject property:
        scene_obj = scene.get('floor')
        scene_obj_mi_shape = scene_obj.mi_shape
        original_pos = scene_obj.position
        new_pos = [7,9,1]

        # Position
        self.assertTrue(np.equal(scene_obj.position,original_pos).all) 
        scene_obj.position = new_pos
        self.assertTrue(np.equal(scene_obj.position,new_pos).all) 

        # Material
        self.assertTrue(scene_obj.radio_material == scene.get('itu_concrete')) 
        scene_obj.radio_material = 'itu_glass'
        self.assertTrue(scene_obj.radio_material == scene.get('itu_glass')) 

        # Reload
        self.assertTrue(scene._bypass_reload != True)
        scene.reload()
        new_scene_obj = scene.get('floor')

        # Check position after reload
        self.assertTrue(new_scene_obj is scene_obj)
        self.assertFalse(new_scene_obj.mi_shape is scene_obj_mi_shape)
        self.assertTrue(np.equal(scene_obj.position,new_pos).all) 
        self.assertTrue(np.equal(new_scene_obj.position,new_pos).all) 

        # Check material after reload
        self.assertTrue(scene_obj.radio_material == scene.get('itu_glass')) 
        self.assertTrue(new_scene_obj.radio_material == scene.get('itu_glass')) 

        # Auto reload (e.g. when adding an asset)
        scene = load_scene(sionna.rt.scene.floor_wall)
        asset = AssetObject(name="asset_0", filename=sionna.rt.asset_object.test_asset_1) 

        # change scene SceneObject property:
        scene_obj = scene.get('floor')
        scene_obj_mi_shape = scene_obj.mi_shape
        original_pos = scene_obj.position
        new_pos = [7,9,1]

        # Position
        self.assertTrue(np.equal(scene_obj.position,original_pos).all) 
        scene_obj.position = new_pos
        self.assertTrue(np.equal(scene_obj.position,new_pos).all) 

        # Material
        self.assertTrue(scene_obj.radio_material == scene.get('itu_concrete')) 
        scene_obj.radio_material = 'itu_glass'
        self.assertTrue(scene_obj.radio_material == scene.get('itu_glass')) 

        # Reload by adding an asset
        self.assertTrue(scene._bypass_reload != True)
        scene.add(asset)
        new_scene_obj = scene.get('floor')

        # Check position after reload
        self.assertTrue(new_scene_obj is scene_obj)
        self.assertFalse(new_scene_obj.mi_shape is scene_obj_mi_shape)
        self.assertTrue(np.equal(scene_obj.position,new_pos).all) 
        self.assertTrue(np.equal(new_scene_obj.position,new_pos).all) 

        # Check material after reload
        self.assertTrue(scene_obj.radio_material == scene.get('itu_glass')) 
        self.assertTrue(new_scene_obj.radio_material == scene.get('itu_glass')) 


    def test_scene_reload(self):
        """Check that the scene is properly reloaded when necessary and only when necessary"""
        # Reload when:
        #   - After adding assets
        scene = load_scene(sionna.rt.scene.floor_wall)
        asset = AssetObject(name="asset_0", filename=sionna.rt.asset_object.test_asset_1) # create a metal asset
        ref_obj = scene.get("floor")
        ref_obj_mi_shape = ref_obj.mi_shape
        scene.add(asset)
        self.assertTrue(ref_obj == scene.get("floor"))
        self.assertTrue(ref_obj_mi_shape != scene.get("floor").mi_shape)

        #   - After removing assets
        ref_obj = scene.get("floor")
        ref_obj_mi_shape = ref_obj.mi_shape
        scene.remove("asset_0")
        self.assertTrue(ref_obj == scene.get("floor"))
        self.assertTrue(ref_obj_mi_shape != scene.get("floor").mi_shape)

        #   - After changing/setting bsdf
        ref_obj = scene.get("floor")
        ref_obj_mi_shape = ref_obj.mi_shape
        metal = scene.get("itu_metal")
        metal.bsdf.rgb = (0.5,0.2,0.9)
        self.assertTrue(ref_obj == scene.get("floor"))
        self.assertTrue(ref_obj_mi_shape != scene.get("floor").mi_shape)

        # - After manually calling scene.reload()
        ref_obj = scene.get("floor")
        ref_obj_mi_shape = ref_obj.mi_shape
        scene.reload()
        self.assertTrue(ref_obj == scene.get("floor"))
        self.assertTrue(ref_obj_mi_shape != scene.get("floor").mi_shape)

        # - Not when changing material of an object (user has to manually trigger the reload):
        ref_obj = scene.get("floor")
        ref_obj_mi_shape = ref_obj.mi_shape
        ref_obj.radio_material = "itu_glass"   
        self.assertTrue(ref_obj == scene.get("floor"))
        self.assertTrue(ref_obj_mi_shape == scene.get("floor").mi_shape)
                 
    def test_scene_reload_via_ray(self):
        """Check that the scene is properly reloaded and the propagation properties are thus changed"""
        scene = load_scene() # Load empty scene
        scene.tx_array = PlanarArray(1,1,0.5,0.5,"iso","V")
        scene.rx_array = PlanarArray(1,1,0.5,0.5,"iso","V")

        ## The test cube(s) are two 1x1x1m cubes, centered in (0,0,0), spaced by 1 m along the y-axis
        cube_edge_length = 1 
        cubes_separation = 1
        asset = AssetObject(name="reflector", filename=sionna.rt.asset_object.test_asset_1, position=[0,-cubes_separation,-cube_edge_length/2]) # we shift the asset so that the face of the metal cube is aligned with the xy-plane and center in (0,0)
        
        d0 = 100
        scene.add(Transmitter("tx", position=[0,+1,d0]))
        scene.add(Receiver("rx", position=[0,-1,d0]))
        paths = scene.compute_paths(los=False, diffraction=False, max_depth=1, method="exhaustive")
        paths.normalize_delays = False
        cir = paths.cir()
        self.assertTrue(len(tf.squeeze(cir[0])) == 0)
        
        scene.add(asset)
        
        paths = scene.compute_paths(los=False, diffraction=False, max_depth=1, method="exhaustive")
        paths.normalize_delays = False
        cir = paths.cir()
        self.assertTrue(len(tf.squeeze(cir[0])) == 2 )
        
        scene.remove("reflector")
        
        paths = scene.compute_paths(los=False, diffraction=False, max_depth=1, method="exhaustive")
        paths.normalize_delays = False
        cir = paths.cir()
        self.assertTrue(len(tf.squeeze(cir[0])) == 0)