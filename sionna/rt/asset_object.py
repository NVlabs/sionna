#
# SPDX-FileCopyrightText: Copyright (c) 2024 ORANGE - Author: Guillaume Larue <guillaume.larue@orange.com>. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""
Base class for asset objects.
"""


import os
import tensorflow as tf


import numpy as np
import xml.etree.ElementTree as ET
import warnings
from importlib_resources import files

from .radio_material import RadioMaterial
from .bsdf import BSDF
from . import assets

from ..utils.misc import copy_and_rename_files
# from .utils import normalize, theta_phi_from_unit_vec
from sionna.constants import PI


class AssetObject():
    # pylint: disable=line-too-long
    r"""AssetObject(name, filename, position=(0.,0.,0.), orientation=(0.,0.,0.), radio_material=None, dtype=tf.complex64)

    A class for managing asset objects. An asset is an object that can be added or removed from a scene and which can consists of multiple shapes. 
    When added to a scene, the asset creates SceneObject instances corresponding to each of its mesh shapes. These scene objects can be moved, rotated, 
    and manipulated individually or collectively through the AssetObject.

    The AssetObject keeps track of the corresponding scene objects, allowing for higher-level operations such as moving all shapes of an asset together 
    while maintaining their relative positions and orientations. The asset is associated with an XML file descriptor pointing to one or multiple mesh 
    (.ply) files. A RadioMaterial can be assigned to the asset, which will be applied to all its shapes. If no material is provided, materials will be 
    inferred from the BSDF descriptors in the asset XML file.

    Note: When exporting an asset in Blender with the Mitsuba (Sionna) file format, it's important to set Z-axis as "up" and Y-axis as "forward". 
    This should align the coordinates correctly.

    Parameters
    ----------
    name : str
        Name of the asset object.

    filename : str
        Path to the asset XML file.

    position : tuple of 3x float, optional,  (default is (0., 0., 0.))
        Initial position of the asset.

    orientation : tuple of 3x float, optional,  (default is (0., 0., 0.))
        Initial orientation of the asset in radians.

    radio_material : :class:`~sionna.rt.RadioMaterial`:, optional (default is None)
        If set, the radio material is to be associated with all the asset's shapes. If not specified, materials will be inferred from the BSDF descriptors in the XML file.

    overwrite_scene_bsdfs : :boolean:, optional, (default is False)
    If True replace all existing bsdf from the scene by the ones specified in the asset files. Otherwise, replace only placeholder scene's bsdfs.

    overwrite_scene_radio_materials : :boolean:, optional, (default is False)
    If True update all existing radio_materials from the scene by the ones specified in the asset files. Otherwise, replace only placeholder scene's radio_materials.
        
    dtype : tf.DType, optional  (default is `tf.complex64`)
        Datatype for all computations, inputs, and outputs.
    """


    def __init__(self, 
                 name, 
                 filename, 
                 position=(0.,0.,0.), 
                 orientation=(0.,0.,0.), 
                 radio_material=None,
                 overwrite_scene_bsdfs = False,
                 overwrite_scene_radio_materials = False,
                 dtype=tf.complex64
                ):

        #TODO: 
        # - Set scale parameter transform?
        # - Implem material argument as either a RadioMaterial class instance or a string e.g. ("itu_marble") which must point to an existing scene material to be used? 
        # - Modify radio material properties of asset already added to a scene?
        # - Define the look_at method?
        # - Define asset library and method to automaticaly load asset (based on string e.g. asset = load_asset("cube"))
        # - Random init transform on multishaped asset can move the shape relative to each other...
        # - Management/update of the asset when altering properties of scene object (e.g. delete one of the scene object? change the material? etc.)
        # - Shapes = list not dict
        # - Add assets example path (+ redesign the two persons asset so that there is a perfect symetrie)
        
        if dtype not in (tf.complex64, tf.complex128):
            raise ValueError("`dtype` must be tf.complex64 or tf.complex128`")
        self._dtype = dtype
        self._rdtype = dtype.real_dtype

        # Asset name
        self._name = name

        # Initialize shapes associated with asset
        self._shapes = {} #Attributed when added to a scene 
        
        # Asset's XML and meshes sources directory
        self._filename = filename
        self._xml_tree = ET.parse(filename)

        # Change asset's mesh directory and asset xml file to a dedicated directory:
        self._meshes_folder_path = f"meshes/{self._name}/"
        for shape in self._xml_tree.findall(".//shape"):
        # Find <string> elements with name="filename" within each asset <shape> to modify the shape pathes to the new mesh folder
            string_element = shape.find(".//string[@name='filename']")
            mesh_path = string_element.get('value')
            filename = mesh_path.split('/')[-1]
            new_mesh_path = self._meshes_folder_path + filename
            string_element.set('value', new_mesh_path)

        # Position & Orientation properties
        # Init boolean flag: Used for inital transforms applied when adding the asset to a scene
        self._position_init = True
        self._orientation_init = True
        self._random_position_init_bias = np.random.random(3) # Initial (temporary) position transform to avoid ovelapping asset which mess with mitsuba load_scene method.

        # (Initial) position and orientation of the asset
        self._position = tf.cast(position, dtype=self._rdtype)
        if np.max(orientation) > 2 * PI:
            warnings.warn("Orientation angle exceeds 2π. Angles should be in radians. If already in radians, you can ignore this warning; otherwise, convert to radians.")
        self._orientation = tf.cast(orientation, dtype=self._rdtype) #in radians

        # if look_at is None:
        #     self.orientation = orientation
        # else:
        #     self.look_at(look_at)

        # Material (If multiple shapes within the asset >> Associate the same material to all shapes)
        self._radio_material = radio_material
        self._overwrite_scene_bsdfs = overwrite_scene_bsdfs # If true, replace scene's bsdfs when adding asset even when they are not placeholder bsdf
        self._overwrite_scene_radio_materials = overwrite_scene_radio_materials # If true, update scene's materials when adding asset even when they are not placeholder material
        
        # Init scene propertie
        self._scene = None

        # Structure to store original asset bsdfs as specified in the asset XML file
        self._original_assets_bsdfs = {}





        

    @property
    def scene(self):
        r"""
        :class:`~sionna.rt.Scene` : Get the scene
        """
        return self._scene

    @scene.setter
    def scene(self, scene):
        r"""
        :class:`~sionna.rt.Scene` : Set the scene to current asset
        """ 
        if self._scene != None and self._scene != self: 
            msg = f"The asset '{self}' is already assigned to another Scene '{self._scene}'. " \
            "Assets object instance can only be assigned to a single scene."
            raise RuntimeError(msg)
        
        # Affect the scene to the current asset
        self._scene = scene

        # Affect scene's dtype to the current asset
        self.dtype = scene.dtype   
        
        # Reset position/orientation init boolean flag (if adding  an already used asset to a different scene?)
        # >>>Probably not necessary since the init flag are reset in the load_scene_object meth...
        self._position_init = True
        self._orientation_init = True  
        
        # Move asset's meshes to the scene's meshes directory
        asset_path = os.path.dirname(self.filename)
        meshes_source_dir = os.path.join(asset_path, 'meshes')
        destination_dir = os.path.join(self._scene.tmp_directory_path, self._meshes_folder_path)
        copy_and_rename_files(meshes_source_dir,destination_dir,prefix='')
        
        # Get the root of the asset's XML
        root_to_append = self._xml_tree.getroot()
        
        # 1 Update Materials
        # Adding material to the  scene will trigger radio_material.scene setter/bsdf.scene setter >>> Update/set bsdf in scene xml 
        if self._radio_material is not None:
            if isinstance(self._radio_material, str):
                mat = self._scene.get(self.radio_material)
                if (mat is not None) and (not isinstance(mat, RadioMaterial)):
                    # If the radio material does not exist, but an item of the scene already uses that name.
                    raise ValueError(f"RadioMaterial name'{self._radio_material}' already used by another item, can't create placeholder material")
                elif mat is None:
                    # If the radio material does not exist and the name is available, then a placeholder is used. 
                    # In this case, the user is in charge of defining the radio material properties afterward.
                    mat = RadioMaterial(self._radio_material)
                    mat.is_placeholder = True
                    self._scene.add(mat)
                self._radio_material = mat
            elif isinstance(self._radio_material, RadioMaterial):
                mat = self._scene.get(self._radio_material.name)
                if isinstance(mat, RadioMaterial) and self._overwrite_scene_radio_materials:
                    # There exist a different instance of RadioMaterial with the same name but the user explicitely specified that 
                    # he wants to replace existing scene materials by the one from the asset, even if the existing material is not a placeholder.
                    mat.assign(self._radio_material)
                    self._radio_material = mat
                else:
                    # The scene.add() method will trigger an error if there exist an item with the same name or non-placeholder radio material which is not the same object.
                    # If there exist a placeholder radio material with same name, it will be updated with the asset radio material properties
                    # If the asset's radio material is a radio material from the scene (same object), then the asset radio material will not be (re)added to the scene.
                    # See scene.add() method for more details.
                    self._scene.add(self._radio_material) 
            else:
                raise TypeError(f"Asset 'radio_material' must be of type 'str', or 'RadioMaterial'")

        else:
            # When no radio_material are specified, add all the bsdf of the asset to the xml file. Overwrite existing bsdfs if the corresponding material's bsdfs are placeholder
            bsdfs_to_append = root_to_append.findall('bsdf')  
            for bsdf in bsdfs_to_append:  
                bsdf_name = bsdf.get('id')
                       
                # Store original bsdfs, if needed
                self._original_assets_bsdfs[f"origin_{bsdf_name}"] = BSDF(name=f"origin_{bsdf_name}", xml_element=bsdf)

                # Change naming to adhere to Sionna conventions 
                if bsdf_name.startswith('mat-'):
                    mat_name = bsdf_name[4:]
                else:
                    mat_name = bsdf_name
                bsdf.set('id',f"mat-{mat_name}")
                mat = self._scene.get(mat_name)
                if (mat is not None):
                    if not isinstance(mat, RadioMaterial): #>>> NECESSARY? This will be handled at add() of the placeholder material?
                        # If the radio material does not exist, but an item of the scene already uses that name.
                        raise ValueError(f"RadioMaterial name'{mat_name}' already used by another item, can't create placeholder material")
                    else:
                        # A radio material exists in the scene with that name.
                        if mat.bsdf.is_placeholder or self._overwrite_scene_bsdfs:
                            # If the material bsdf is a placeholder (or the user explicitely want to overwrite existing bsdfs) then the existing bsdf  
                            # is updated with the bsdf of the asset.
                            mat.bsdf.xml_element = bsdf
                        else:
                            # If the material bsdf is not a placeholder then we keep the original scene bsdf
                            pass
                else:
                    # If the radio material does not exist and the name is available, then a placeholder is used. 
                    # In this case, the user is in charge of defining the radio material properties afterward.
                    # The asset's bsdf is affected to the newly material
                    material_bsdf = BSDF(name=f"mat-{mat_name}",xml_element=bsdf)
                    mat = RadioMaterial(mat_name, bsdf=material_bsdf)
                    mat.is_placeholder = True
                    self._scene.add(mat)

        
        # 2 Update geometry
        # Find all shapes elements in the asset 
        shapes_to_append = root_to_append.findall('shape')  
           
        # Append each shape to the parent element in the original scene while adapting their ids
        for shape in shapes_to_append:
            # Define the shape name
            shape_id = shape.get('id')
            if shape_id.startswith('mesh-'):
                shape_name = shape_id[5:]
            else:
                shape_name = shape_id
            new_shape_id = f"{self._name}_{shape_name}"
            self._shapes[new_shape_id] = None
            shape.set('id',f"mesh-{new_shape_id}")
            shape.set('name',f"mesh-{new_shape_id}")  
            
            # Define shape transforms - Add (temporary) position bias
            # IT SEEMS THAT MITSUBA AUTOMATICALLY MERGE VERTEX AT THE SAME POSITION, WHEN LOADING THE SCENE (i.e it is not possible to add the same asset twice at the same position)
            # Hence, as a quick fix, we apply a small random transforms within the xml scene descriptor to ensure that two asset never share the same position before calling mi.load() function. 
            # The correct, position and orientation are then applied when loading the scene objects (self._scene._load_scene_objects() method) after calling mi.load() function.
            transform = ET.SubElement(shape, 'transform', name="to_world")
            position = self._random_position_init_bias
            translate_value = f"{position[0]} {position[1]} {position[2]}"
            ET.SubElement(transform, 'translate', value=translate_value)
            
            ref = shape.find('ref')
            if self._radio_material is not None:
                # If the specified material has been get/set from/to the scene: specify the correct bsdf name in the shape descriptor (for all shapes of the asset)
                ref.set('id',f"mat-{self._radio_material.name}")
                self._scene.add(self._radio_material)

            # When no radio material is specified for the asset we consider per shape radio materials definition based on the xml descriptor of the asset
            else:
                bsdf_name = ref.get('id')
                if bsdf_name.startswith('mat-'):
                    mat_name = bsdf_name[4:]
                else:
                    mat_name = bsdf_name
                ref.set('id',f"mat-{mat_name}")
                

            # Add shape to xml
            self._scene.append_to_xml(shape, overwrite=False)


    @property
    def shapes(self):
        return self._shapes

    # To be removed?
    @shapes.setter
    def shapes(self, shapes):
        self._shapes = shapes

    @property
    def filename(self):
        return self._filename
    
    @property
    def name(self):
        return self._name

    @property
    def xml_tree(self):
        return self._xml_tree

    @property
    def dtype(self):
        r"""
        `tf.complex64 | tf.complex128` : Datatype used in tensors
        """
        return self._dtype
    
    @dtype.setter
    def dtype(self, new_dtype):
        if new_dtype not in (tf.complex64, tf.complex128):
            raise ValueError("`dtype` must be tf.complex64 or tf.complex128`")
        self._dtype = new_dtype
        self._rdtype = new_dtype.real_dtype

        self._position = tf.cast(self._position, dtype=self._rdtype)
        self._orientation = tf.cast(self._orientation, dtype=self._rdtype)

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, new_position):
        # Move all shapes associated to assed while keeping their relative positions
        position = tf.cast(new_position, dtype=self._rdtype)
        if self._position_init and self._scene is not None:
            diff = position - self._random_position_init_bias # Correct the initial position bias initally added to avoid mitsuba to merge edges at the same position
            self._position_init = False
        else:
            diff = position - self._position
        self._position = position

        for shape_id in self.shapes:
            
            scene_object = self.shapes[shape_id] 
            print(shape_id)
            print(self._scene._scene_objects)
            print(scene_object,scene_object.position)
            if scene_object is not None:
                scene_object.position += diff
        
    @property
    def orientation(self):
        return self._orientation

    @orientation.setter
    def orientation(self, new_orientation):
        # Rotate all shapes associated to asset while keeping their relative positions (i.e. rotate arround the asset position)
        orientation = tf.cast(new_orientation, dtype=self._rdtype)
        self._orientation = orientation

        for shape_id in self.shapes:
            scene_object = self.shapes[shape_id] 
            if scene_object is not None:
                scene_object = self._scene.get(shape_id)
                old_center_of_rotation = scene_object.center_of_rotation
                new_center_of_rotation = (self._position - scene_object.position)
                scene_object.center_of_rotation = new_center_of_rotation
                scene_object.orientation = orientation
                scene_object.center_of_rotation = old_center_of_rotation
                
    @property
    def random_position_init_bias(self):
        return self._random_position_init_bias

    @property
    def position_init(self):
        return self._position_init
    
    @position_init.setter
    def position_init(self, position_init):
        # Set the asset in its init state (for initial position and rotation definition)
        self._position_init = position_init

    @property
    def orientation_init(self):
        return self._orientation_init
    
    @orientation_init.setter
    def orientation_init(self, orientation_init):
        # Set the asset in its init state (for initial position and rotation definition)
        self._orientation_init = orientation_init


    @property
    def radio_material(self):
        r"""
        :class:`~sionna.rt.RadioMaterial` : Get/set the radio material of the
        object. Setting can be done by using either an instance of
        :class:`~sionna.rt.RadioMaterial` or the material name (`str`).
        If the radio material is not part of the scene, it will be added. This
        can raise an error if a different radio material with the same name was
        already added to the scene.
        """
        return self._radio_material

    @radio_material.setter
    def radio_material(self, mat):
        if not isinstance(mat, RadioMaterial) or not isinstance(mat, str):
            err_msg = f"Radio material must be of type 'str' or 'sionna.rt.RadioMaterial"
            raise TypeError(err_msg)
        
        mat_obj = mat

        # If the asset as been added to a scene
        if self._scene is not None:
            if isinstance(mat, str):
                mat_obj = self.scene.get(mat)
                if (mat_obj is None) or (not isinstance(mat_obj, RadioMaterial)):
                    err_msg = f"Unknown radio material '{mat}'"
                    raise TypeError(err_msg)
                
            for shape_name in self._shapes:
                # Get the current radio material of the scene object corresponding to that shape.
                scene_object = self.get(shape_name)
                # prev_material = scene_object.radio_material
                
                # Update the material of the scene object
                scene_object.radio_material = mat_obj

                # DEPRECATED: (Problematic if deleting base sionna material e.g. itu_wood, the user should rather use force_material_update when adding an asset to remove existing material properties)
                # Check if the previous material is still in use. If not:
                # - (1) remove the material from the scene's material
                # - (2) remove the bsdf from the scene's xml file
                # if not prev_material.is_used:
                #     warnings.warn(f"RadioMaterial {prev_material.name} is not used anymore, it will be deleted from the scene")
                #     bsdf_name = prev_material.bsdf.name
                #     self._scene.remove(prev_material.name)
                #     self._scene.remove_from_xml(bsdf_name,"bsdf") 
          

        # Store the new asset material (which is now the same for all asset's shape)
        self._radio_material = mat_obj

            
    ############################## SceneObject properties to be extended to AssetObject ###################
    


    # @property
    # def velocity(self):
    #     """
    #     [3], tf.float : Get/set the velocity vector [m/s]
    #     """
    #     return self._velocity

    # @velocity.setter
    # def velocity(self, v):
    #     if not tf.shape(v)==3:
    #         raise ValueError("`velocity` must have shape [3]")
    #     self._velocity = tf.cast(v, self._rdtype)

    #     for shape_id in self.shapes:
    #         scene_object = self._scene.get(shape_id)
    #         scene_object.velocity = self._velocity

    
    # def look_at(self, target):
    #     # pylint: disable=line-too-long
    #     r"""
    #     Sets the orientation so that the x-axis points toward an
    #     ``Object``.

    #     Input
    #     -----
    #     target : [3], float | :class:`sionna.rt.Object` | str
    #         A position or the name or instance of an
    #         :class:`sionna.rt.Object` in the scene to point toward to
    #     """
    #     # Get position to look at
    #     if isinstance(target, str):
    #         if self._scene == None:
    #             err_msg = f"No scene have been affected to current AssetObject"
    #             raise TypeError(err_msg)
            
    #         obj = self.scene.get(target)
    #         if not isinstance(obj, Object) or not isinstance(obj, AssetObject):
    #             raise ValueError(f"No camera, device, or object named '{target}' found.")
    #         else:
    #             target = obj.position
    #     elif isinstance(target, Object) or isinstance(target, AssetObject):
    #         target = target.position
    #     else:
    #         target = tf.cast(target, dtype=self._rdtype)
    #         if not target.shape[0]==3:
    #             raise ValueError("`target` must be a three-element vector)")

    #     # Compute angles relative to LCS
    #     x = target - self.position
    #     x, _ = normalize(x)
    #     theta, phi = theta_phi_from_unit_vec(x)
    #     alpha = phi # Rotation around z-axis
    #     beta = theta-PI/2 # Rotation around y-axis
    #     gamma = 0.0 # Rotation around x-axis
    #     self.orientation = (alpha, beta, gamma)


#
# Module variables for example asset files
#
test_asset = str(files(assets).joinpath("test_asset/test_asset.xml"))
# pylint: disable=C0301
"""
Example asset containing two 1x1x1m cubes spaced by 1m along the y-axis.
"""

body = str(files(assets).joinpath("body/body.xml"))
# pylint: disable=C0301
r"""
Example asset containing one human body
"""

monkey = str(files(assets).joinpath("monkey/monkey.xml"))
# pylint: disable=C0301
r"""
Example asset containing one monkey head
"""

two_persons = str(files(assets).joinpath("two_persons/two_persons.xml"))
# pylint: disable=C0301
r"""
Example asset containing two human bodies
"""