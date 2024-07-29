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


class TestXMLChange(unittest.TestCase):
    """Tests related to the change of the XML file"""

    def test_append_new_shape_to_xml(self):
        """Check that the scene.append_to_xml() method works when adding a new shape"""
        shape = ET.Element('shape', attrib={'type': 'ply', 'id': 'mesh-new'})
        ET.SubElement(shape, 'string', attrib={'name': 'filename', 'value': 'meshes/wall.ply'})
        ET.SubElement(shape, 'boolean', attrib={'name': 'face_normals', 'value': 'true'})
        ET.SubElement(shape, 'ref', attrib={'id': 'mat-itu_glass', 'name': 'bsdf'})

        scene = load_scene(sionna.rt.scene.floor_wall)
        root = scene._xml_tree.getroot()
        elements_in_root = root.findall('shape')  
        ids_in_root = [elt.get('id') for elt in elements_in_root]
        self.assertFalse('mesh-new' in ids_in_root)

        out = scene.append_to_xml(shape, overwrite=False)
        elements_in_root = root.findall('shape')  
        ids_in_root = [elt.get('id') for elt in elements_in_root]
        self.assertTrue('mesh-new' in ids_in_root)
        self.assertTrue(out is None)

        scene = load_scene(sionna.rt.scene.floor_wall)
        root = scene._xml_tree.getroot()
        elements_in_root = root.findall('shape')  
        ids_in_root = [elt.get('id') for elt in elements_in_root]
        self.assertFalse('mesh-new' in ids_in_root)

        out = scene.append_to_xml(shape, overwrite=True)
        elements_in_root = root.findall('shape')  
        ids_in_root = [elt.get('id') for elt in elements_in_root]
        self.assertTrue('mesh-new' in ids_in_root)
        self.assertTrue(out is None)


    def test_append_new_bsdf_to_xml(self):
        """Check that the scene.append_to_xml() method works when adding a new bsdf"""
        bsdf = ET.Element('bsdf', attrib={'type': 'diffuse', 'id': 'bsdf-new', 'name': 'bsdf-new'})
        ET.SubElement(bsdf, 'rgb', attrib={'value': '1.000000 0.000000 0.300000', 'name': 'reflectance'})
        
        scene = load_scene(sionna.rt.scene.floor_wall)
        root = scene._xml_tree.getroot()
        elements_in_root = root.findall('bsdf')  
        ids_in_root = [elt.get('id') for elt in elements_in_root]
        self.assertFalse('bsdf-new' in ids_in_root)

        out = scene.append_to_xml(bsdf, overwrite=False)
        elements_in_root = root.findall('bsdf')  
        ids_in_root = [elt.get('id') for elt in elements_in_root]
        self.assertTrue('bsdf-new' in ids_in_root)
        self.assertTrue(out is None)

        scene = load_scene(sionna.rt.scene.floor_wall)
        root = scene._xml_tree.getroot()
        elements_in_root = root.findall('bsdf')  
        ids_in_root = [elt.get('id') for elt in elements_in_root]
        self.assertFalse('bsdf-new' in ids_in_root)

        out = scene.append_to_xml(bsdf, overwrite=True)
        elements_in_root = root.findall('bsdf')  
        ids_in_root = [elt.get('id') for elt in elements_in_root]
        self.assertTrue('bsdf-new' in ids_in_root)
        self.assertTrue(out is None)

    def test_append_other_to_xml(self):
        """Check that the scene.append_to_xml() method does not accept non bsdf or shape elements"""
        other_elt = ET.Element('other', attrib={'type': 'diffuse', 'id': 'other_elt', 'name': 'other_elt'})
        ET.SubElement(other_elt, 'boolean', attrib={'name': 'other', 'value': 'false'})

        scene = load_scene(sionna.rt.scene.floor_wall)
        root = scene._xml_tree.getroot()
        elements_in_root = root.findall('other')  
        ids_in_root = [elt.get('id') for elt in elements_in_root]
        self.assertFalse('other_elt' in ids_in_root)

        with self.assertRaises(ValueError) as context:
            out = scene.append_to_xml(other_elt, overwrite=False)
        self.assertEqual(str(context.exception), "`element` must be an instance of ``ET.Element`` of type <shape> or <bsdf>")
                             
        elements_in_root = root.findall('other')  
        ids_in_root = [elt.get('id') for elt in elements_in_root]
        self.assertFalse('other_elt' in ids_in_root)
        

        scene = load_scene(sionna.rt.scene.floor_wall)
        root = scene._xml_tree.getroot()
        elements_in_root = root.findall('other')  
        ids_in_root = [elt.get('id') for elt in elements_in_root]
        self.assertFalse('other_elt' in ids_in_root)

        with self.assertRaises(ValueError) as context:
            out = scene.append_to_xml(other_elt, overwrite=True)
        self.assertEqual(str(context.exception), "`element` must be an instance of ``ET.Element`` of type <shape> or <bsdf>")

        elements_in_root = root.findall('other')  
        ids_in_root = [elt.get('id') for elt in elements_in_root]
        self.assertFalse('other_elt' in ids_in_root)
        

    def test_append_existing_shape_to_xml(self):
        """Check that the scene.append_to_xml() method does not change the XML when adding an existing shape with 'overwrite' argumnt set to False"""
        shape = ET.Element('shape', attrib={'type': 'ply', 'id': 'mesh-floor'})
        ET.SubElement(shape, 'string', attrib={'name': 'filename', 'value': 'meshes/wall.ply'})
        ET.SubElement(shape, 'boolean', attrib={'name': 'face_normals', 'value': 'true'})
        ET.SubElement(shape, 'ref', attrib={'id': 'mat-itu_glass', 'name': 'bsdf'})

        scene = load_scene(sionna.rt.scene.floor_wall)
        root = scene._xml_tree.getroot()
        elements_in_root = root.findall('shape')  
        ids_in_root = [elt.get('id') for elt in elements_in_root]
        self.assertTrue('mesh-floor' in ids_in_root)

        for elt in elements_in_root:
            if elt.get('id') == 'mesh-floor':
                break

        with self.assertWarns(UserWarning) as context:
            out = scene.append_to_xml(shape, overwrite=False)
        self.assertTrue(ET.tostring(elt, encoding='unicode') == ET.tostring(out, encoding='unicode'))
        self.assertEqual(str(context.warning), "Element of type shape with id: mesh-floor is already present in xml file. Set 'overwrite=True' to overwrite.")
             
        elements_in_root = root.findall('shape')  
        ids_in_root = [elt.get('id') for elt in elements_in_root]
        for new_elt in elements_in_root:
            if new_elt.get('id') == 'mesh-floor':
                break
        self.assertTrue('mesh-floor' in ids_in_root)
        self.assertTrue(ET.tostring(elt, encoding='unicode') == ET.tostring(new_elt, encoding='unicode'))


    def test_append_existing_shape_to_xml_overwrite(self):
        """Check that the scene.append_to_xml() method does change the XML when adding an existing shape with 'overwrite' argumnt set to True"""
        shape = ET.Element('shape', attrib={'type': 'ply', 'id': 'mesh-floor'})
        ET.SubElement(shape, 'string', attrib={'name': 'filename', 'value': 'meshes/wall.ply'})
        ET.SubElement(shape, 'boolean', attrib={'name': 'face_normals', 'value': 'true'})
        ET.SubElement(shape, 'ref', attrib={'id': 'mat-itu_glass', 'name': 'bsdf'})

        scene = load_scene(sionna.rt.scene.floor_wall)
        root = scene._xml_tree.getroot()
        elements_in_root = root.findall('shape')  
        ids_in_root = [elt.get('id') for elt in elements_in_root]
        self.assertTrue('mesh-floor' in ids_in_root)
        
        for elt in elements_in_root:
            if elt.get('id') == 'mesh-floor':
                break

        with self.assertWarns(UserWarning) as context:
            out = scene.append_to_xml(shape, overwrite=True)
        self.assertTrue(ET.tostring(elt, encoding='unicode')== ET.tostring(out, encoding='unicode'))
        self.assertEqual(str(context.warning), "Element of type shape with id: mesh-floor is already present in xml file. Overwriting with new element.")
             
        elements_in_root = root.findall('shape')  
        ids_in_root = [elt.get('id') for elt in elements_in_root]
        for new_elt in elements_in_root:
            if new_elt.get('id') == 'mesh-floor':
                break
        self.assertTrue('mesh-floor' in ids_in_root)
        self.assertTrue(ET.tostring(elt, encoding='unicode') != ET.tostring(new_elt, encoding='unicode'))


    def test_append_existing_bsdf_to_xml(self):
        """Check that the scene.append_to_xml() method does not change the XML when adding an existing bsdf with 'overwrite' argumnt set to False"""
        bsdf = ET.Element('bsdf', attrib={'type': 'diffuse', 'id': 'mat-itu_concrete', 'name': 'mat-itu_concrete'})
        ET.SubElement(bsdf, 'rgb', attrib={'value': '1.000000 0.000000 0.300000', 'name': 'reflectance'})

        scene = load_scene(sionna.rt.scene.floor_wall)
        root = scene._xml_tree.getroot()
        elements_in_root = root.findall('bsdf')  
        ids_in_root = [elt.get('id') for elt in elements_in_root]
        self.assertTrue('mat-itu_concrete' in ids_in_root)

        for elt in elements_in_root:
            if elt.get('id') == 'mat-itu_concrete':
                break

        with self.assertWarns(UserWarning) as context:
            out = scene.append_to_xml(bsdf, overwrite=False)
        self.assertTrue(ET.tostring(elt, encoding='unicode') == ET.tostring(out, encoding='unicode'))
        self.assertEqual(str(context.warning), "Element of type bsdf with id: mat-itu_concrete is already present in xml file. Set 'overwrite=True' to overwrite.")
             
        elements_in_root = root.findall('bsdf')  
        ids_in_root = [elt.get('id') for elt in elements_in_root]
        for new_elt in elements_in_root:
            if new_elt.get('id') == 'mat-itu_concrete':
                break
        self.assertTrue('mat-itu_concrete' in ids_in_root)
        self.assertTrue(ET.tostring(elt, encoding='unicode') == ET.tostring(new_elt, encoding='unicode'))


    def test_append_existing_bsdf_to_xml_overwrite(self):
        """Check that the scene.append_to_xml() method does change the XML when adding an existing bsdf with 'overwrite' argumnt set to True"""
        bsdf = ET.Element('bsdf', attrib={'type': 'diffuse', 'id': 'mat-itu_concrete', 'name': 'mat-itu_concrete'})
        ET.SubElement(bsdf, 'rgb', attrib={'value': '1.000000 0.000000 0.300000', 'name': 'reflectance'})

        scene = load_scene(sionna.rt.scene.floor_wall)
        root = scene._xml_tree.getroot()
        elements_in_root = root.findall('bsdf')  
        ids_in_root = [elt.get('id') for elt in elements_in_root]
        self.assertTrue('mat-itu_concrete' in ids_in_root)
        
        for elt in elements_in_root:
            if elt.get('id') == 'mat-itu_concrete':
                break

        with self.assertWarns(UserWarning) as context:
            out = scene.append_to_xml(bsdf, overwrite=True)
        self.assertTrue(ET.tostring(elt, encoding='unicode')== ET.tostring(out, encoding='unicode'))
        self.assertEqual(str(context.warning), "Element of type bsdf with id: mat-itu_concrete is already present in xml file. Overwriting with new element.")
             
        elements_in_root = root.findall('bsdf')  
        ids_in_root = [elt.get('id') for elt in elements_in_root]
        for new_elt in elements_in_root:
            if new_elt.get('id') == 'mat-itu_concrete':
                break
        self.assertTrue('mat-itu_concrete' in ids_in_root)
        self.assertTrue(ET.tostring(elt, encoding='unicode') != ET.tostring(new_elt, encoding='unicode'))



    def test_remove_existing_shape_from_xml(self):
        """Check that the scene.remove_from_xml() method does change the XML when removing an existing shape without specyfing 'element_type' argument"""
        scene = load_scene(sionna.rt.scene.floor_wall)
        root = scene._xml_tree.getroot()
        elements_in_root = root.findall('shape')  
        ids_in_root = [elt.get('id') for elt in elements_in_root]
        self.assertTrue('mesh-floor' in ids_in_root)
        
        scene.remove_from_xml('mesh-floor')
        elements_in_root = root.findall('shape')  
        ids_in_root = [elt.get('id') for elt in elements_in_root]
        self.assertFalse('mesh-floor' in ids_in_root)

    def test_remove_existing_shape_from_xml_correct_type(self):
        """Check that the scene.remove_from_xml() method does change the XML when removing an existing shape while specyfing the correct 'element_type' argument"""
        scene = load_scene(sionna.rt.scene.floor_wall)
        root = scene._xml_tree.getroot()
        elements_in_root = root.findall('shape')  
        ids_in_root = [elt.get('id') for elt in elements_in_root]
        self.assertTrue('mesh-floor' in ids_in_root)
        
        scene.remove_from_xml('mesh-floor', element_type='shape')
        elements_in_root = root.findall('shape')  
        ids_in_root = [elt.get('id') for elt in elements_in_root]
        self.assertFalse('mesh-floor' in ids_in_root)

    def test_remove_existing_shape_from_xml_wrong_type(self):
        """Check that the scene.remove_from_xml() method does not change the XML when removing an existing shape while specyfing the wrong 'element_type' argument"""
        scene = load_scene(sionna.rt.scene.floor_wall)
        root = scene._xml_tree.getroot()
        elements_in_root = root.findall('shape')  
        ids_in_root = [elt.get('id') for elt in elements_in_root]
        self.assertTrue('mesh-floor' in ids_in_root)
        
        with self.assertWarns(UserWarning) as context:
            scene.remove_from_xml('mesh-floor', element_type='bsdf')
        self.assertEqual(str(context.warning), "No bsdf element with name mesh-floor in root to remove.")
        elements_in_root = root.findall('shape')  
        ids_in_root = [elt.get('id') for elt in elements_in_root]
        self.assertTrue('mesh-floor' in ids_in_root)
        
    def test_remove_existing_bsdf_from_xml(self):
        """Check that the scene.remove_from_xml() method does change the XML when removing an existing bsdf without specyfing 'element_type' argument"""
        scene = load_scene(sionna.rt.scene.floor_wall)
        root = scene._xml_tree.getroot()
        elements_in_root = root.findall('bsdf')  
        ids_in_root = [elt.get('id') for elt in elements_in_root]
        self.assertTrue('mat-itu_concrete' in ids_in_root)
        
        scene.remove_from_xml('mat-itu_concrete')
        elements_in_root = root.findall('bsdf')  
        ids_in_root = [elt.get('id') for elt in elements_in_root]
        self.assertFalse('mat-itu_concrete' in ids_in_root)

    def test_remove_existing_bsdf_from_xml_correct_type(self):
        """Check that the scene.remove_from_xml() method does change the XML when removing an existing bsdf while specyfing the correct 'element_type' argument"""
        scene = load_scene(sionna.rt.scene.floor_wall)
        root = scene._xml_tree.getroot()
        elements_in_root = root.findall('bsdf')  
        ids_in_root = [elt.get('id') for elt in elements_in_root]
        self.assertTrue('mat-itu_concrete' in ids_in_root)
        
        scene.remove_from_xml('mat-itu_concrete', element_type='bsdf')
        elements_in_root = root.findall('bsdf')  
        ids_in_root = [elt.get('id') for elt in elements_in_root]
        self.assertFalse('mat-itu_concrete' in ids_in_root)

    def test_remove_existing_bsdf_from_xml_wrong_type(self):
        """Check that the scene.remove_from_xml() method does not change the XML when removing an existing bsdf while specyfing the wrong 'element_type' argument"""
        scene = load_scene(sionna.rt.scene.floor_wall)
        root = scene._xml_tree.getroot()
        elements_in_root = root.findall('bsdf')  
        ids_in_root = [elt.get('id') for elt in elements_in_root]
        self.assertTrue('mat-itu_concrete' in ids_in_root)
        
        with self.assertWarns(UserWarning) as context:
            scene.remove_from_xml('mat-itu_concrete', element_type='shape')
        self.assertEqual(str(context.warning), "No shape element with name mat-itu_concrete in root to remove.")
        elements_in_root = root.findall('bsdf')  
        ids_in_root = [elt.get('id') for elt in elements_in_root]
        self.assertTrue('mat-itu_concrete' in ids_in_root)

    def test_remove_existing_invalid_from_xml(self):
        """Check that the scene.remove_from_xml() method does not change the XML when removing an existing invalid element without specyfing 'element_type' argument"""
        # Remove existing other invalid type (without specyfing type)
        scene = load_scene(sionna.rt.scene.floor_wall)
        root = scene._xml_tree.getroot()
        elements_in_root = root.findall('sensor')  
        ids_in_root = [elt.get('id') for elt in elements_in_root]
        self.assertTrue('Camera' in ids_in_root)
        
        with self.assertWarns(UserWarning) as context:
            scene.remove_from_xml('Camera')
        self.assertEqual(str(context.warning), "No ['bsdf', 'shape'] element with name Camera in root to remove.")
        elements_in_root = root.findall('sensor')  
        ids_in_root = [elt.get('id') for elt in elements_in_root]
        self.assertTrue('Camera' in ids_in_root)

    def test_remove_existing_invalid_from_xml_correct_type(self):
        """Check that the scene.remove_from_xml() method does not change the XML when removing an existing invalid element while specyfing correct (but invalid)'element_type' argument"""
        scene = load_scene(sionna.rt.scene.floor_wall)
        root = scene._xml_tree.getroot()
        elements_in_root = root.findall('sensor')  
        ids_in_root = [elt.get('id') for elt in elements_in_root]
        self.assertTrue('Camera' in ids_in_root)
        
        with self.assertRaises(ValueError) as context:
            scene.remove_from_xml('Camera', element_type='sensor')
        self.assertEqual(str(context.exception), "`element_type` must be string. Valid types are ['bsdf', 'shape'].")
        elements_in_root = root.findall('sensor')  
        ids_in_root = [elt.get('id') for elt in elements_in_root]
        self.assertTrue('Camera' in ids_in_root)

    def test_remove_existing_invalid_from_xml_correct_type(self):
        """Check that the scene.remove_from_xml() method does not change the XML when removing an existing invalid element while specyfing incorrect (but valid)'element_type' argument"""
        scene = load_scene(sionna.rt.scene.floor_wall)
        root = scene._xml_tree.getroot()
        elements_in_root = root.findall('sensor')  
        ids_in_root = [elt.get('id') for elt in elements_in_root]
        self.assertTrue('Camera' in ids_in_root)
        
        with self.assertWarns(UserWarning) as context:
            scene.remove_from_xml('Camera', element_type='shape')
        self.assertEqual(str(context.warning), "No shape element with name Camera in root to remove.")
        elements_in_root = root.findall('sensor')  
        ids_in_root = [elt.get('id') for elt in elements_in_root]
        self.assertTrue('Camera' in ids_in_root)

    
    def test_remove_non_existing_shape_from_xml(self):
        """Check that the scene.remove_from_xml() method does not change the XML when removing a non-existing shape without specyfing 'element_type' argument"""
        scene = load_scene(sionna.rt.scene.floor_wall)
        root = scene._xml_tree.getroot()
        elements_in_root = root.findall('shape')  
        ids_in_root = [elt.get('id') for elt in elements_in_root]
        self.assertFalse('mesh-wrong' in ids_in_root)
        
        with self.assertWarns(UserWarning) as context:
            scene.remove_from_xml('mesh-wrong')
        self.assertEqual(str(context.warning), "No ['bsdf', 'shape'] element with name mesh-wrong in root to remove.")
        elements_in_root = root.findall('shape')  
        ids_in_root = [elt.get('id') for elt in elements_in_root]
        self.assertFalse('mesh-wrong' in ids_in_root)

    def test_remove_non_existing_shape_from_xml_correct_type(self):
        """Check that the scene.remove_from_xml() method does not change the XML when removing a non-existing shape while specyfing the correct 'element_type' argument"""
        scene = load_scene(sionna.rt.scene.floor_wall)
        root = scene._xml_tree.getroot()
        elements_in_root = root.findall('shape')  
        ids_in_root = [elt.get('id') for elt in elements_in_root]
        self.assertFalse('mesh-wrong' in ids_in_root)
        
        with self.assertWarns(UserWarning) as context:
            scene.remove_from_xml('mesh-wrong', element_type='shape')
        self.assertEqual(str(context.warning), "No shape element with name mesh-wrong in root to remove.")
        elements_in_root = root.findall('shape')  
        ids_in_root = [elt.get('id') for elt in elements_in_root]
        self.assertFalse('mesh-wrong' in ids_in_root)
    
    def test_remove_non_existing_shape_from_xml_wrong_type(self):
        """Check that the scene.remove_from_xml() method does not change the XML when removing a non-existing shape while specyfing the wrong 'element_type' argument"""
        scene = load_scene(sionna.rt.scene.floor_wall)
        scene = load_scene(sionna.rt.scene.floor_wall)
        root = scene._xml_tree.getroot()
        elements_in_root = root.findall('shape')  
        ids_in_root = [elt.get('id') for elt in elements_in_root]
        self.assertFalse('mesh-wrong' in ids_in_root)
        
        with self.assertWarns(UserWarning) as context:
            scene.remove_from_xml('mesh-wrong', element_type='bsdf')
        self.assertEqual(str(context.warning), "No bsdf element with name mesh-wrong in root to remove.")
        elements_in_root = root.findall('shape')  
        ids_in_root = [elt.get('id') for elt in elements_in_root]
        self.assertFalse('mesh-wrong' in ids_in_root)
     