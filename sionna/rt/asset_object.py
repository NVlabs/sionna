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
import mitsuba as mi
import xml.etree.ElementTree as ET
import warnings

from .radio_material import RadioMaterial
from .object import Object

from ..utils.misc import copy_and_rename_files
from .utils import normalize, theta_phi_from_unit_vec
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

    Parameters
    ----------
    name : str
        Name of the asset object.

    filename : str
        Path to the asset XML file.

    position : tuple of 3x float, optional
        Initial position of the asset (default is (0., 0., 0.)).

    orientation : tuple of 3x float, optional
        Initial orientation of the asset in radians (default is (0., 0., 0.)).

    radio_material : :class:`~sionna.rt.RadioMaterial`:, optional
        If set, the radio material is to be associated with all the asset's shapes. If not specified, materials will be inferred from the BSDF descriptors in the XML file.

    dtype : tf.DType, optional
        Datatype for all computations, inputs, and outputs. Defaults to `tf.complex64`.
    """


    def __init__(self, 
                 name, 
                 filename, 
                 position=(0.,0.,0.), 
                 orientation=(0.,0.,0.), 
                 radio_material=None,
                 dtype=tf.complex64
                ):

        #TODO: 
        # - Set scale parameter transform?
        # - Implem material argument as either a RadioMaterial class instance or a string e.g. ("itu_marble") which must point to an existing scene material to be used? 
        # - Modify radio material properties of asset already added to a scene?
        # - Define the look_at method?
        # - Define asset library and method to automaticaly load asset (based on string e.g. asset = load_asset("cube"))
        
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

        # Init scene propertie
        self._scene = None

    @property
    def scene(self):
        """
        :class:`~sionna.rt.Scene` : Get/set the scene
        """
        return self._scene

    @scene.setter
    def scene(self, scene):
        # Affect the scene to the current asset
        self._scene = scene
        
        # Reset position/orientation init boolean flag (if adding the an already used asset to a different scene?)
        self._position_init = True
        self._orientation_init = True  

        # Move asset's meshes to the scene's meshes directory
        asset_path = os.path.dirname(self.filename)
        meshes_source_dir = os.path.join(asset_path, 'meshes')
        destination_dir = os.path.join(self._scene.tmp_directory_path, self._meshes_folder_path)
        copy_and_rename_files(meshes_source_dir,destination_dir,prefix='')

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
    def position(self):
        return self._position

    @position.setter
    def position(self, new_position):
        # Move all shapes associated to assed while keeping their relative positions
        position = tf.cast(new_position, dtype=self._rdtype)
        if self._position_init:
            diff = position
            self._position_init = False
        else:
            diff = position - self._position
        self._position = position

        for shape_id in self.shapes:
            scene_object = self._scene.get(shape_id)
            old_position = scene_object.position
            new_position = old_position + diff
            scene_object.position = new_position
        
    @property
    def orientation(self):
        return self._orientation

    @orientation.setter
    def orientation(self, new_orientation):
        # Rotate all shapes associated to asset while keeping their relative positions (i.e. rotate arround the asset position)
        orientation = tf.cast(new_orientation, dtype=self._rdtype)
        self._orientation = orientation
        for shape_id in self.shapes:
            scene_object = self._scene.get(shape_id)
            old_center_of_rotation = scene_object.center_of_rotation
            new_center_of_rotation = (self._position - scene_object.position)
            scene_object.center_of_rotation = new_center_of_rotation
            scene_object.orientation = orientation
            scene_object.center_of_rotation = old_center_of_rotation
                
    
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


    ############################## SceneObject properties to be extended to AssetObject ###################
    # @radio_material.setter
    # def radio_material(self, mat):
    #     # Note: _radio_material is set at __init__, but pylint doesn't see it.
    #     if mat is None:
    #         mat_obj = None

    #     elif isinstance(mat, str):
    #         if self._scene == None:
    #             err_msg = f"No scene have been affected to current AssetObject"
    #             raise TypeError(err_msg)
            
    #         mat_obj = self.scene.get(mat)
    #         if (mat_obj is None) or (not isinstance(mat_obj, RadioMaterial)):
    #             err_msg = f"Unknown radio material '{mat}'"
    #             raise TypeError(err_msg)

    #     elif not isinstance(mat, RadioMaterial):
    #         err_msg = ("The material must be a material name (str) or an "
    #                     "instance of RadioMaterial")
    #         raise TypeError(err_msg)

    #     else:
    #         mat_obj = mat

    #     self._radio_material = mat_obj

    #     for shape_id in self.shapes:
    #         scene_object = self._scene.get(shape_id)
    #         previous_radio_material = scene_object.radio_material
    #         scene_object.radio_material = self._radio_material

    #         if not previous_radio_material.is_used:
    #             warnings.warn(f"RadioMaterial {previous_radio_material.name} is not used anymore, it will be deleted from the scene")
    #             bsdf_name = previous_radio_material.bsdf.name
    #             self._scene.remove(previous_radio_material.name)
    #             self._scene.remove_bsdf(bsdf_name)
        
    #     self._scene.reload_scene_after_modifying_assets()


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