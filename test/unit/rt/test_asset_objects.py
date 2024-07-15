#
# SPDX-FileCopyrightText: Copyright (c) 2024 ORANGE - Author: Guillaume Larue <guillaume.larue@orange.com>. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


#
# SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

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
import itertools

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


class TestAddAssetObject(unittest.TestCase):
    """Tests related to the AssetObject class"""
    
    def test_add_asset(self):
        """Adding asset to scene"""
        scene = load_scene(sionna.rt.scene.floor_wall)
        asset = AssetObject(name="asset_0", filename=sionna.rt.asset_object.test_asset)
        scene.add(asset)
        self.assertTrue("asset_0" in scene.asset_objects)
        self.assertTrue(isinstance(scene.get("asset_0"),AssetObject)) 
        self.assertTrue(isinstance(scene.get("asset_0_cube_0"),SceneObject)) 
        self.assertTrue(isinstance(scene.get("asset_0_cube_1"),SceneObject)) 

    def test_add_asset_list(self):
        """Adding list of asset to scene"""
        scene = load_scene(sionna.rt.scene.floor_wall)
        asset_0 = AssetObject(name="asset_0", filename=sionna.rt.asset_object.test_asset)
        asset_1 = AssetObject(name="asset_1", filename=sionna.rt.asset_object.test_asset)
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
        asset = AssetObject(name="asset_0", filename=sionna.rt.asset_object.test_asset, dtype=tf.complex128)
        scene.add(asset)
        self.assertEqual(scene.dtype, asset.dtype)
        self.assertEqual(scene.dtype.real_dtype, asset.position.dtype)

    def test_add_asset_overwrite(self):
        """When adding an asset to a scene, the asset should overwrite any asset with the same name"""
        scene = load_scene(sionna.rt.scene.floor_wall)
        asset_0 = AssetObject(name="asset_0", filename=sionna.rt.asset_object.test_asset)
        scene.add(asset_0)

        self.assertTrue("asset_0" in scene.asset_objects)
        self.assertTrue(scene.get("asset_0") == asset_0)  

        asset_0_bis = AssetObject(name="asset_0", filename=sionna.rt.asset_object.test_asset)
        scene.add(asset_0_bis)

        self.assertTrue("asset_0" in scene.asset_objects)
        self.assertTrue(scene.get("asset_0") != asset_0) 

    def test_remove_asset(self):
        """Removing an asset from scene"""
        scene = load_scene(sionna.rt.scene.floor_wall)
        asset = AssetObject(name="asset_0", filename=sionna.rt.asset_object.test_asset)
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
        # Instanciation of the dict is correct when adding asset
        # What appens we deleteing the asset
        # What if loading a second asset / reload scene
        # What if overwrting the asset
        # What if deleting one SceneObject of a composite asset? Does it still work for the remaining object?
        self.assertTrue(False)

    def test_add_xml_remove_xml(self):
        # oerwrite?
        self.assertTrue(False)


class TestAssetPosition(unittest.TestCase):
    """Tests related to the change of an asset's position"""

    def test_change_position_with_dtype(self):
        """Changing the position works in all dtypes"""
        for dtype in [tf.complex64, tf.complex128]:
            scene = load_scene(sionna.rt.scene.floor_wall, dtype=dtype)
            asset = AssetObject(name="asset_0", filename=sionna.rt.asset_object.test_asset)
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
        asset = AssetObject(name="reflector", filename=sionna.rt.asset_object.test_asset, position=[0,-cubes_separation,-cube_edge_length/2]) # we shift the asset so that the face of the metal cube is aligned with the xy-plane and center in (0,0)
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
        two 1x1x1m cubes, centered in (0,0,0), spaced by 1 m along the y-axis. Hence, the barycenters of the cubes are (0,-1,0) and (0,+1,0). """
        scene = load_scene(sionna.rt.scene.floor_wall)

        cube_edge_length = 1 
        cubes_separation = 1
        random_position = np.random.random(3)
        asset = AssetObject(name="asset_0", filename=sionna.rt.asset_object.test_asset, position=random_position) 
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
        epsilon = 1e-5 # The Boudning-boxes computed by sionna to estimate the position of an object are not entirely accurate
        self.assertTrue(tf.reduce_all(tf.math.less_equal(tf.math.abs(cube_0_object.position-cube_0_position),epsilon)))
        self.assertTrue(tf.reduce_all(tf.math.less_equal(tf.math.abs(cube_1_object.position-cube_1_position),epsilon)))

        
    def test_init_bias_when_reloading_scene(self):
        """Check if the bias introduced at asset object init to avoid Mitsuba mixing up shapes at the same position, is not added several time when reloading a scene."""
        
        scene = load_scene(sionna.rt.scene.floor_wall)

        asset_0 = AssetObject(name="asset_0", filename=sionna.rt.asset_object.test_asset) 
        scene.add(asset_0)

        cube_0_object = scene.get("asset_0_cube_0")
        cube_1_object = scene.get("asset_0_cube_1")

        asset_0_position_0 = tf.Variable(asset_0.position)
        cube_0_position_0 = tf.Variable(cube_0_object.position)
        cube_1_position_0 = tf.Variable(cube_1_object.position)

        # Adding a secondary asset to reload the scene
        asset_1 = AssetObject(name="asset_1", filename=sionna.rt.asset_object.test_asset) 
        scene.add(asset_1)

        asset_0_position_1 = tf.Variable(asset_0.position)
        cube_0_position_1 = tf.Variable(cube_0_object.position)
        cube_1_position_1 = tf.Variable(cube_1_object.position)

        # Or manually reload the scene
        scene.reload_scene()

        asset_0_position_2 = tf.Variable(asset_0.position)
        cube_0_position_2 = tf.Variable(cube_0_object.position)
        cube_1_position_2 = tf.Variable(cube_1_object.position)

        self.assertTrue(tf.reduce_all((asset_0_position_0==asset_0_position_1) == (asset_0_position_0==asset_0_position_2)))
        self.assertTrue(tf.reduce_all((cube_0_position_0==cube_0_position_1) == (cube_0_position_0==cube_0_position_2)))
        self.assertTrue(tf.reduce_all((cube_1_position_0==cube_1_position_1) == (cube_1_position_0==cube_1_position_2)))


    def test_position_add_vs_set(self):
        """Check that position accumulation lead to the same result as setting the complete position at once for asset objects"""
        scene = load_scene()
        asset = AssetObject(name="asset_0", filename=sionna.rt.asset_object.test_asset)
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
   
    

class TestAssetOrientation(unittest.TestCase):
    """Tests related to the change of an asset's orientation"""

    def test_change_orientation_with_dtype(self):
        """Changing the orientation works in all dtypes"""
        for dtype in [tf.complex64, tf.complex128]:
            scene = load_scene(sionna.rt.scene.floor_wall, dtype=dtype)
            asset = AssetObject(name="asset_0", filename=sionna.rt.asset_object.test_asset)
            scene.add(asset)
            target_orientation = tf.cast([-PI/3, 0.1, PI/2], dtype.real_dtype)
            asset.orientation = target_orientation
            self.assertEqual(asset.orientation.dtype, dtype.real_dtype)
            self.assertTrue(np.array_equal(asset.orientation, target_orientation))
    
    def test_no_position_change(self):
        """Changing orientation should not change the position of the asset"""
        scene = load_scene(sionna.rt.scene.floor_wall)
        asset = AssetObject(name="asset_0", filename=sionna.rt.asset_object.test_asset)
        scene.add(asset)
        pos_org = asset.position
        asset.orientation = [0.2,0.3,-0.4]
        self.assertTrue(tf.reduce_all(asset.position==pos_org))

    def test_orientation_axis_convention(self):
        """Check axis convention when rotating asset"""
        scene = load_scene()
        asset = AssetObject(name="asset_0", filename=sionna.rt.asset_object.test_asset)
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

        
    def test_orientation_add_vs_set(self):
        """Check that rotation accumulation lead to the same result as setting the complete rotation at once for asset objects"""
        scene = load_scene()
        asset = AssetObject(name="asset_0", filename=sionna.rt.asset_object.test_asset)
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
        """Test showing that rotating a simple reflector asset can make a paths dissappear"""
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
        
        # Rotating the reflector by PI/4 should make the path dissappear
        asset.orientation = [0,0,PI/4]

        paths = scene.compute_paths(los=False, diffraction=False, max_depth=1)
        self.assertEqual(tf.squeeze(paths.tau).shape, [0])

class TestAssetMaterial(unittest.TestCase):
    """Tests related to the asset's materials"""

    def test_no_asset_material_set_before_adding_asset(self):
        """Test showing that specifying no asset material before adding the asset work when the asset is added to scene"""
        scene = load_scene(sionna.rt.scene.floor_wall)
        asset = AssetObject(name="asset_0", filename=sionna.rt.asset_object.test_asset) # test asset is made of metal and wood

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
        self.assertTrue(not prev_glass.is_used)
        self.assertTrue(not prev_wood.is_used)
        self.assertTrue(not prev_metal.is_used)
        self.assertTrue(prev_glass_bsdf.is_placeholder)
        self.assertTrue(prev_wood_bsdf.is_placeholder)
        self.assertTrue(prev_metal_bsdf.is_placeholder)

        # Add the asset
        scene.add(asset)

        # After adding the asset, the asset radio material is still None (since it has never been specified)
        # When adding asset to scene, the material are specified on a per shape basis (using the asset xml) since no material have been specified.
        self.assertTrue(asset.radio_material == None)
        self.assertTrue(scene.get("asset_0").radio_material == None)

        # After adding the asset, the material the asset use are now used and their bsdf are not placeholders anymore
        new_glass = scene.get("itu_glass")
        new_wood = scene.get("itu_wood")
        new_metal =scene.get("itu_metal")
        new_glass_bsdf = new_glass.bsdf
        new_wood_bsdf = new_wood.bsdf
        new_metal_bsdf = new_metal.bsdf
        self.assertTrue(not new_glass.is_used)
        self.assertTrue(new_wood.is_used)
        self.assertTrue(new_metal.is_used)
        self.assertTrue(new_glass_bsdf.is_placeholder)
        self.assertTrue(not new_wood_bsdf.is_placeholder)
        self.assertTrue(not new_metal_bsdf.is_placeholder)
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
        self.assertTrue(not prev_wood_bsdf_xml == new_wood_bsdf_xml)
        self.assertTrue(not prev_metal_bsdf_xml == new_metal_bsdf_xml)
        
    def test_str_asset_material_set_before_adding_asset(self):
        """Test showing that specifying asset material as a str before adding the asset work when the asset is added to scene"""
        # str existing material

        # str non existing material
        self.assertTrue(False)
        
    # set material after adding assset to scene

    # are itu init material placeholders

    # are itu init material bsdf placeholders

    # are material placeholder replaced by asset material when specified

    # are bsdf placeholder replaced by asset's bsdf when specified

    # what if adding an asset with material and bsdf non placeholder to the scene with a material non placeholder with placeholder bsdf? does the material stay unchanged while the bsdf updated?

    # scene init replace placeholder bsdf?

    # creation of asset:
        # Set no material use shape dependant material
            # Create placeholder when necessary
            # Use scene material otherwise
            # Replace the bsdf when and only when they  are placeholder
        # Set material with str with name of an alreeady existing material
        # Set material with str with name of a non existing material
        # Set material with RadioMaterial
        # Set material with RadioMaterial from the scene
        # Set material with RadioMaterial that has an already existing name
        # Set material with wrong type
        
    # Asset removal delete material when not in use anymore

    # Previous asset related material are not impact new material

    # trying to replace a non placeholder material (e..g from a previous asset add) raise a warning and does not change the non placeholder material

    # Asset add/remove update well object_id in material and corresponding bsdf

    # changing bsdf of a material

    # changing material of an asset 
        # Using str
        # Using RM
        # Using wrong type
        # Need to reload scene?

    # Asset material update change propagation properties

    # Reload scene

    # change (material) properties of an object belonging to an asset where self._material is not None 
    # should update asset material to None (ie not all scene object have the same material anymore)

    # change asset material should update shape xmL?  

    # Who is in charge of reloading scene? automatic when changing bsdf? (maybe add a scene.bypass_reload scene to avoid intempestive reload)

    # Check if material is not str or RadioMaterial

    # Check if object_id are removed from material and bsdf object_using sets when scene_objects are cleared

    # Check objects_using syncrho between material and bsdf

    # Overwrite bsdf and or material when placeholder or when asked by user. For material check that assign works well (including for object_ids)

    # Problem? change a property (material, position, speed, etc.) of a non-asset scene object, then add an asset >>> trigger scene reload, this should revert the scene object properties... > non wanted

    # Check availability of origin_bsdfs in corresponding data structure

    # Test assign method radio_material assign also the object_id and bsdf

# BSDF/Material + secure API when adding an asset with an existing material to the scene? modyfing material properties 
# => At least add a warning stating that the asset material has not been taken into account since there is already a material with that name
#
#
#
#
# Shape name / asset name / -mesh
#
# Append asset/ remove asset impact on the scene could be managed inside the asset class or even scene object class?

# # Material & BSDF of the Shape (if defined the radio material object contain both the radio propagation properties and the rendering (i.e. BSDF) properties)
# if asset.material is not None:
# # If the asset as a radio material defined, it will be used for all shapes within the asset
# material_name = asset.material.name
# bsdf_name = asset.material.bsdf.name
# else:
# # If not, for each shape we parse the corresponding bsdf within the xml (as exported from Blender).
# # In this case several material/bsdf can be defined for the different constituent shapes of the asset
# bsdf_name = shape.findall('.//ref')[0].get('id')
# material_name = bsdf_name
# if bsdf_name.startswith("mat-"):
# material_name = material_name[4:]

# # The material name associated to each shape is store within the asset's shapes properties dict.
# shapes[new_shape_id]['material'] = material_name

# # We then check if the material is already in use within the current Sionna scene (if so the BSDF is already described within 
# # the scene xml file, since the materials are instantiated based on the xml BSDFs in the first place).
# if material_name not in self._radio_materials: # 
# # If the material does not exist in Sionna scene data structure, check if the BSDF is not already present in the XML scene descriptor 
# # (if so the material should be instantiated during the next load_scene_objects() call)

# bsdfs_in_root = [bsdf.get('id') for bsdf in root.findall('./bsdf')]
# if bsdf_name not in bsdfs_in_root:
# # If not add the asset's bsdf to the xml file 
# if asset.material is None: 
#     # Find the bsdf element with the specific bsdf name from the asset xml file and append it to the scene xml file
#     bsdf = root_to_append.find(f".//bsdf[@id='{bsdf_name}']")
# else:
#     # Or, if defined, use the asset's material's BSDF, and add the material to the scene's materials
#     bsdf = asset.material.bsdf.xml_tree
#     self.add(asset.material) 
# root.append(bsdf)

# @property
#     def radio_material(self):
#     

#     @radio_material.setter
#     def radio_material(self, mat):
#     

#     @property
#     def velocity(self):
#         

#     @velocity.setter
#     def velocity(self, v):

#     def look_at(self, target):
#         

# def test_replace_asset_different_material(self):
#     #     586 # Check if the material is still in use. If not:
#     #     587 # - (1) remove the material from the scene's material
#     #     588 # - (2) remove the bsdf from the scene's xml file
#     #     589 if not radio_material.is_used:
    
#     #     357         f"Object with id {object_id} is not in the set of {self.name}"
#     #     358     self._objects_using.discard(object_id)
#     #     359     if self._bsdf is not None:
        # AssertionError: Object with id 7 is not in the set of itu_metal

#     #     asset = AssetObject(f"asset_{i}", filename=sionna.rt.scene.simple_reflector, position=position, orientation=orientation)#, radio_material=custom_rm)#sionna.rt.asset_object.test_asset #"./sionna/rt/assets/two_persons/two_persons.xml"
#     #     asset_list.append(asset)
#     #     asset = AssetObject(f"asset_{i}", filename=sionna.rt.asset_object.test_asset, position=position, orientation=orientation)#, radio_material=custom_rm)#sionna.rt.asset_object.test_asset #"./sionna/rt/assets/two_persons/two_persons.xml"
#     #     asset_list.append(asset)
            
#     #     scene.add(asset_list)
    

#TODO:  
#       - change folder and name not strictly xml related anymore
#       - rm tmp folder at init of scene?
#       - Clean using scene.shapes() instead of xml manipulation?
#       - Check coordinate and rotation conventions
#       - Implement Logger functionalities
#       - Check if adding group asset (multiple shape) works.
#       - Behaviour when BSDF/material already exist (but with different properties) 
#       - XML to dict?
#       - Check if that is okay: material_name = shape.findall('.//ref')[0].get('id') ref and not bsdf
#       - When using nested asset --> How the position/rotation transform are handled?
#       - Maybe should not use XML transform but rather position the object and then use SceneObject API to move/rotate object to keep consistency between object coordinate in the object properties and scene display
#               |---> currently all move/orientation are relative to the initial transform.
#       - Shapes properties dict to be defined within asset class, not in append asset function
#       - The meshes of an asset should be added (removed) to the tmp folder of the scene when adding to the scene not on asset init.

        