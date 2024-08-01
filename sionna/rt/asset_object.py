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
import copy

from importlib_resources import files


from .radio_material import RadioMaterial
from .bsdf import BSDF
from .object import Object
from .object import Object

from ..utils.misc import copy_and_rename_files
from .utils import normalize, theta_phi_from_unit_vec
from .utils import normalize, theta_phi_from_unit_vec
from sionna.constants import PI

from . import assets

from . import assets


class AssetObject():
    # pylint: disable=line-too-long
    r"""AssetObject(name, filename, position=(0.,0.,0.), orientation=(0.,0.,0.), radio_material=None, dtype=tf.complex64)

    A class for managing asset objects. An asset is an object that can be added or removed from a scene and which can consist of multiple shapes. 
    When added to a scene, the asset creates SceneObject instances corresponding to each of its mesh shapes. These scene objects can be moved, rotated, 
    and manipulated individually or collectively through the AssetObject.

    The AssetObject keeps track of the corresponding scene objects, allowing for higher-level operations such as moving all shapes of an asset together 
    while maintaining their relative positions and orientations. The asset is associated with an XML file descriptor pointing to one or multiple mesh 
    (.ply) files. A RadioMaterial can be assigned to the asset, which will be applied to all its shapes. If no material is provided, materials will be 
    inferred from the BSDF descriptors in the asset XML file.

    Note: When exporting an asset in Blender with the Mitsuba (Sionna) file format, it's important to set Z-axis as "up" and Y-axis as "forward". 
    This should align the coordinates correctly.

    Example asset can be loaded as follows:

    .. code-block:: Python
        scene = load_scene()
        asset = AssetObject(name='asset_name', filename=sionna.rt.asset_object.test_asset_1)
        scene.add(asset)

    Parameters
    ----------
    name : str
        Name of the asset object.

    filename : str
        Path to the asset XML file.

    position : [3], float
        Position :math:`(x,y,z)` as three-dimensional vector.
        Defaults to `None`, in which case the position is not set at init.

    orientation : [3], float | None
        Orientation :math:`(\alpha, \beta, \gamma)` specified
        through three angles corresponding to a 3D rotation
        as defined in :eq:`rotation`.
        This parameter is ignored if ``look_at`` is not `None`.

    look_at : [3], float | :class:`sionna.rt.Object` | :class:`sionna.rt.AssetObject` | None
        A position or the instance of an :class:`~sionna.rt.Object` to
        point toward to.

    radio_material : :class:`~sionna.rt.RadioMaterial`:, optional (default is None)
        If set, the radio material is to be associated with all the asset's shapes. If not specified, materials will be inferred from the BSDF descriptors in the XML file.

    overwrite_scene_bsdfs : :boolean:, optional, (default is False)
        If True replace all existing bsdf from the scene by the ones specified in the asset files. Otherwise, replace only placeholder scene's bsdfs. 
        This argument can be only defined at asset initialisation and has an effect only upon adding the asset to as scene.

    overwrite_scene_radio_materials : :boolean:, optional, (default is False)
        If True update existing radio_material from the scene by the ones specified in the asset files. Otherwise, replace only placeholder scene's radio_materials. 
        This argument can be only defined at asset initialisation and has an effect only upon adding the asset to as scene.
        
    dtype : tf.DType, optional  (default is `tf.complex64`)
        Datatype for all computations, inputs, and outputs.
    """

    def __init__(self, 
                 name, 
                 filename, 
                 position=(0.,0.,0.), 
                 orientation=(0.,0.,0.), 
                 look_at=None,
                 radio_material=None,
                 overwrite_scene_bsdfs = False,
                 overwrite_scene_radio_materials = False,
                 dtype=tf.complex64
                ):

      
        if dtype not in (tf.complex64, tf.complex128):
            raise TypeError("`dtype` must be tf.complex64 or tf.complex128`")
        
        self._dtype = dtype
        self._rdtype = dtype.real_dtype

        # Asset name
        self._name = name

        # Initialize shapes associated with asset
        self._shapes = {} #Attributed when added to a scene 
        
        # Asset's XML and meshes sources directory
        self._filename = filename
        self._xml_tree = ET.parse(filename)

        # Init scene propertie
        self._scene = None

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
        self._init = True
        self._random_position_init_bias = np.random.random(3) # Initial (temporary) position transform to avoid ovelapping asset which mess with mitsuba load_scene method.

        # (Initial) position and orientation of the asset
        self._position = tf.cast(position, dtype=self._rdtype)
        if np.max(orientation) > 2 * PI:
            warnings.warn("Orientation angle exceeds 2π. Angles should be in radians. If already in radians, you can ignore this warning; otherwise, convert to radians.")
        

        if look_at is None:
            self._orientation = tf.cast(orientation, dtype=self._rdtype) #in radians
        else:
            self._orientation = tf.cast([0,0,0], dtype=self._rdtype)
            self.look_at(look_at)

        # Velocity
        self._velocity = tf.cast([0,0,0], dtype=self._rdtype)

        # Material (If multiple shapes within the asset >>> Associate the same material to all shapes)
        if radio_material != None:
            if not isinstance(radio_material,str) and not isinstance(radio_material,RadioMaterial):
                raise TypeError("`radio_material` must be `str` or `RadioMaterial` (or None)")
            
        self._radio_material = radio_material
        self._overwrite_scene_bsdfs = overwrite_scene_bsdfs # If True, replace scene's bsdfs when adding asset even when they are not placeholder bsdf
        self._overwrite_scene_radio_materials = overwrite_scene_radio_materials # If True, update scene's materials when adding asset even when they are not placeholder material
        
        
        # Structure to store original asset bsdfs as specified in the asset XML file, if needed
        self._original_bsdfs = {}
        root_to_append = self._xml_tree.getroot()
        bsdfs_to_append = root_to_append.findall('bsdf')  
        for bsdf in bsdfs_to_append:  
            bsdf = copy.deepcopy(bsdf)
            bsdf_name = bsdf.get('id')
            self._original_bsdfs[f"{bsdf_name}"] = BSDF(name=f"{bsdf_name}", xml_element=bsdf)

        # Bypass update flag - internal flag used to avoid intempestiv update trigger by shape modification called within an asset method
        self._bypass_update = False
        
        
    @property
    def original_bsdfs(self):
        r"""
        `dict` : { "name", :class:`~sionna.rt.BSDF`}:  Get the original asset's bsdfs
        """
        return self._original_bsdfs

    @property
    def overwrite_scene_bsdfs(self):
        r"""
        bool : Get the flag overwrite scene BSDFs
        """
        return self._overwrite_scene_bsdfs 

    # @overwrite_scene_bsdfs.setter
    # def overwrite_scene_bsdfs(self, b):
    #     self._overwrite_scene_bsdfs = b 

    @property
    def overwrite_scene_radio_materials(self):
        r"""
        bool : Get the flag overwrite scene radio materials
        """
        return self._overwrite_scene_radio_materials 

    # @overwrite_scene_radio_materials.setter
    # def overwrite_scene_radio_materials(self, b):
    #     self._overwrite_scene_radio_materials = b   

    @property
    def filename(self):
        r"""
        str : Get the filename of the asset
        """
        return self._filename
    
    @property
    def name(self):
        r"""
        str : Get the name of the asset
        """
        return self._name

    @property
    def xml_tree(self):
        r"""
        xml.etree.ElementTree.ElementTree : Get the XML tree of the asset
        """
        return self._xml_tree

    @property
    def dtype(self):
        r"""
        tf.complex64 | tf.complex128 : Get/set the datatype used in tensors
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
        r"""
        [3], tf.float : Get the position of the asset
        """
        return self._position

    @position.setter
    def position(self, new_position):
        r"""
        Set the position of the asset.

        This method updates the position of the asset and moves all associated shapes
        while keeping their relative positions. If the asset is being initialized and
        is part of a scene, it corrects the initial position bias that was added to
        avoid overlapping shapes.

        Parameters
        ----------
        new_position : [3], float
            The new position of the asset.

        Notes
        -----
        - If the asset is being initialized (`self._init` is `True`) and is part of a scene,
        the initial position bias is corrected.
        - The method calculates the difference between the new position and the current
        position (or the initial position bias) and applies this difference to all
        associated shapes to move them accordingly.
        """
        # Move all shapes associated to assed while keeping their relative positions
        position = tf.cast(new_position, dtype=self._rdtype)
        if self._init and self._scene is not None:
            diff = position - self._random_position_init_bias # Correct the initial position bias initally added to avoid mitsuba to merge edges at the same position
        else:
            diff = position - self._position
        self._position = position

        for shape_id in self.shapes:
            
            scene_object = self.shapes[shape_id] 
            if scene_object is not None:
                scene_object.position += diff
        
    @property
    def orientation(self):
        r"""
        [3], tf.float : Get/set the orientation of the asset
        """
        return self._orientation

    @orientation.setter
    def orientation(self, new_orientation):
        r"""
        Set the orientation of the asset.

        This method updates the orientation of the asset and rotates all associated shapes
        while keeping their relative positions. If the asset is being initialized and
        is part of a scene, it applies the new orientation directly.

        Parameters
        ----------
        new_orientation : [3], float
            The new orientation of the asset in radians.

        Notes
        -----
        - If the asset is being initialized (`self._init` is `True`) and is part of a scene,
        the new orientation is applied directly.
        - The method calculates the difference between the new orientation and the current
        orientation (or applies the new orientation directly if initializing) and applies
        this difference to all associated shapes to rotate them accordingly.
        - The rotation is performed around a center of rotation shifted by the asset's position.
        """
        # Rotate all shapes associated to asset while keeping their relative positions (i.e. rotate arround the asset position)
        orientation = tf.cast(new_orientation, dtype=self._rdtype)
        if self._init and self._scene is not None:
            diff = orientation
        else:
            diff = orientation - self._orientation

        self._orientation = orientation

        for shape_id in self.shapes:
            scene_object = self.shapes[shape_id] 
            if scene_object is not None:
                scene_object = self._scene.get(shape_id)
                old_center_of_rotation = scene_object.center_of_rotation
                new_center_of_rotation = self._position - scene_object.position#self._position#(
                scene_object.center_of_rotation = new_center_of_rotation
                scene_object.orientation += diff
                scene_object.center_of_rotation = old_center_of_rotation
                
    @property
    def random_position_init_bias(self):
        r"""
        [3], float : Get the random initial position bias
        """
        return self._random_position_init_bias

    @property
    def init(self):
        r"""
        bool : Get/set the asset initialization flag used for initial position and rotation definition.
        """
        return self._init
    
    @init.setter
    def init(self, init):
        self._init = init

    @property
    def radio_material(self):
        r"""
        :class:`~sionna.rt.RadioMaterial` : Get the radio material of the object. 
        """
        return self._radio_material

    @radio_material.setter
    def radio_material(self, mat):
        r"""
        Set the radio material of the asset.

        This method updates the radio material of the asset and ensures that all
        associated shapes use the same material. If the asset is part of a scene,
        it checks if the material already exists in the scene and handles any
        conflicts appropriately.

        Parameters
        ----------
        mat : str or :class:`~sionna.rt.RadioMaterial`
            The new radio material for the asset. It can be specified as a string
            (the name of the material) or as an instance of `RadioMaterial`.

        Notes
        -----
        - If the asset is part of a scene and the material is specified as a string,
        it checks if a material with that name already exists in the scene. If it
        does not exist, an error is raised.
        - If the material is specified as a `RadioMaterial` instance, it is added to
        the scene if it does not already exist.
        - The method temporarily disables scene auto-reload to avoid intempestive reloading
        at each asset's shape radio material (more precisely BSDF) assignation, or if multiple 
        asset are added simultaneously
        - The method sets an internal bypass flag to avoid re-updating the asset
        material for each shape modification.

        Raises
        ------
        TypeError
            If the material is not of type `str` or `RadioMaterial`.

        ValueError
            If the material specified as a string does not exist in the scene.

        ValueError
            If the material specified as `RadioMaterial` has a name already in 
            use by an item of the scene
        """
        if not isinstance(mat, RadioMaterial) and not isinstance(mat, str):
            err_msg = f"Radio material must be of type 'str' or 'sionna.rt.RadioMaterial"
            raise TypeError(err_msg)

        mat_obj = mat

        # If the asset as been added to a scene
        if self._scene is not None:
            if isinstance(mat, str):
                mat_obj = self.scene.get(mat)
                if (mat_obj is None) or (not isinstance(mat_obj, RadioMaterial)):
                    err_msg = f"Unknown radio material '{mat}'"
                    raise ValueError(err_msg)

            
            # Turn scene auto reload off, if a new material was to be added
            b_tmp = self._scene.bypass_reload
            self._scene.bypass_reload = True
            # Add radio material before assigning it to SceneObject ensures that if auto-reload 
            # is enabled the asset has the correct material upon reload
            if mat_obj.name not in self._scene.radio_materials:
                self._radio_material = mat_obj 
                self._scene.add(mat_obj)

            # set asset update bypass to True so that the asset material is not re-updated at each asset shape modification
            self._bypass_update = True
            for shape_name in self._shapes:
                
                # Get the current radio material of the scene object corresponding to that shape.
                scene_object = self._scene.get(shape_name)
                
                # Update the material of the scene object
                scene_object.radio_material = mat_obj

            self._scene.bypass_reload = b_tmp

            self._bypass_update = False
          
        # Store the new asset material (which is now the same for all asset's shape)
        self._radio_material = mat_obj

    @property
    def velocity(self):
        """
        [3], tf.float : Get/set the velocity vector to all shapees of the asset[m/s]
        """
        return self._velocity

    @velocity.setter
    def velocity(self, v):
        if not tf.shape(v)==3:
            raise ValueError("`velocity` must have shape [3]")
        self._velocity = tf.cast(v, self._rdtype)

        # set asset update bypass to True so that the asset material is not re-updated at each asset shape modification
        self._bypass_update = True
        
        for shape_id in self.shapes:
            scene_object = self._scene.get(shape_id)
            scene_object.velocity = self._velocity
        
        self._bypass_update = False 
    
    @property
    def shapes(self):
        """
        `dict`, { "name", :class:`~sionna.rt.SceneObject`} : dictionary
            of asset's scene objects
        """
        return self._shapes

    @shapes.setter
    def shapes(self, d):
        if not isinstance(d, dict):
            raise ValueError("Expected a dictionary")
        self._shapes = d
    
    @property
    def scene(self):
        r"""
        :class:`~sionna.rt.Scene` | None : Get the asset's scene
        """
        return self._scene

    @scene.setter
    def scene(self, scene):
        r"""
        :class:`~sionna.rt.Scene` | None: Set a scene to current asset. 
        If scene is None, the asset is properly reset and can be added again to another scene
        """ 
        if self._scene is not None and scene is None:
            self._remove_from_scene()

        elif scene is not self._scene:
            self._assign_to_scene(scene)
            
        # Otherwise, no action needed since either:
        # - scene is not None but self._scene is already set to the same scene.
        # - both scene and self._scene are None.

    def _remove_from_scene(self):
        r"""
        Remove the asset from its current scene.
        """
        # Clear the asset's shapes dictionary
        self._shapes.clear()
        # Set asset's scene to None
        self._scene = None
        # Reset asset init boolean flag (in case the asset will be added again to a scene).
        self._init = True 
    
    def _assign_to_scene(self, scene):
        r"""
        Assign the asset to a new scene.
        
        Input
        -----
        scene : :class:`~sionna.rt.Scene`
            The scene to assign the asset to.
        """
        if self._scene is not None: 
            msg = f"The asset '{self}' is already assigned to another Scene '{self._scene}'. " \
            "Assets object instance can only be assigned to a single scene. Please first remove " \
            "asset from current scene by using asset.scene.remove(asset.name)"
            raise RuntimeError(msg)

        # Affect the scene to the current asset
        self._scene = scene

        # Affect scene's dtype to the current asset
        self.dtype = scene.dtype   
        
        # Reset asset init boolean flag (in case the asset was previously added to a scene).
        self._init = True 
        
        # Move asset's meshes to the scene's meshes directory
        self._move_meshes_to_scene()

        # Update materials
        self._update_materials()
            
        # Update geometry
        self._update_geometry()

    def _move_meshes_to_scene(self):
        r"""
        Move asset's meshes to the scene's meshes directory.
        """
        asset_path = os.path.dirname(self.filename)
        meshes_source_dir = os.path.join(asset_path, 'meshes')
        destination_dir = os.path.join(self._scene.tmp_directory_path, self._meshes_folder_path)
        copy_and_rename_files(meshes_source_dir, destination_dir, prefix='')

    def _update_materials(self):
        r"""
        Update materials in the scene.
        """    
        if self._radio_material is not None:
            self._handle_radio_material()
        else:
            self._append_bsdfs()

    def _update_geometry(self):
        r"""
        Update the geometry of the scene.

        This method updates the geometry of the scene by appending each shape
        defined in the asset's XML to the scene's XML. It also adapts the shape
        IDs to ensure they are unique within the scene and applies a temporary
        position bias to avoid overlapping shapes when loading the scene.

        Notes
        -----
        - Mitsuba automatically merges vertices at the same position when loading
        the scene. To avoid this, a small random transform is applied to ensure
        that two assets never share the same position before calling the
        `mi.load()` function. The correct position and orientation are then
        applied during :meth:`~sionna.rt.Scene._load_scene_objects()`call, after calling `:meth:mi.load()`.
        """
        # Get the root of the asset's XML 
        root_to_append = self._xml_tree.getroot() 

        # Find all shapes elements in the asset 
        shapes_to_append = root_to_append.findall('shape')  
        
        # Append each shape to the parent element in the original scene while adapting their ids
        for shape in shapes_to_append:
            # Define the shape name
            shape_id = shape.get('id')
            shape_name = shape_id[5:] if shape_id.startswith('mesh-') else shape_id
            new_shape_id = f"{self._name}_{shape_name}"
            self._shapes[new_shape_id] = None
            shape.set('id',f"mesh-{new_shape_id}")
            shape.set('name',f"mesh-{new_shape_id}")  
            
            # Define shape transforms - Add (temporary) position bias
            # Mitsuba automatically merges vertices at the same position when loading the scene.
            # To avoid this, apply a small random transform within the XML scene descriptor.
            # The correct position and orientation are then applied when loading the scene objects.
            transform = ET.SubElement(shape, 'transform', name="to_world")
            position = self._random_position_init_bias
            translate_value = f"{position[0]} {position[1]} {position[2]}"
            ET.SubElement(transform, 'translate', value=translate_value)
            
            # Set the correct BSDF name in the shape descriptor
            ref = shape.find('ref')
            if self._radio_material is not None:
                # If a radio material is specified, use its name for all shapes of the asset
                ref.set('id',f"mat-{self._radio_material.name}")
                self._scene.add(self._radio_material)

            # If no radio material is specified, use the BSDF name from the XML descriptor
            else:
                bsdf_name = ref.get('id')
                mat_name = bsdf_name[4:] if bsdf_name.startswith('mat-') else bsdf_name
                ref.set('id',f"mat-{mat_name}")

            # Add shape to xml
            self._scene.append_to_xml(shape, overwrite=False)

    def _handle_radio_material(self):
        r"""
        Handle the assignment of the radio material to the asset.

        This method manages the assignment of the radio material of the asset,
        ensuring that the material is properly gotten from or added to the scene and that any
        conflicts with existing materials are handled appropriately.

        If the radio material is specified as a string, it checks if a material
        with that name already exists in the scene. If it does not exist, a
        placeholder material is created and added to the scene. If a non-material item
        with the same name but exists, an error is raised.

        If the radio material is specified as a `RadioMaterial` instance, it
        checks if a material with the same name already exists in the scene. If
        the `overwrite_scene_radio_materials` flag is set to `True`, the existing
        material is replaced with the new one. Otherwise, the new material is
        added to the scene only if it does not conflict with existing materials.

        Raises
        ------
        ValueError
            If an item with the same name but of a different type already
            exists in the scene.

        TypeError
            If the radio material is not of type `str` or `RadioMaterial`.
        """
        if isinstance(self._radio_material, str):
            mat = self._scene.get(self.radio_material)
            if mat is None:
                # If the radio material does not exist and the name is available, then a placeholder is used. 
                # In this case, the user is in charge of defining the radio material properties afterward.
                mat = RadioMaterial(self._radio_material)
                mat.is_placeholder = True
                self._scene.add(mat)
            elif not isinstance(mat, RadioMaterial):
                # If the radio material does not exist, but an item of the scene already uses that name.
                raise ValueError(f"Name '{self._radio_material}' is already used by another item of the scene")
            self._radio_material = mat 
        elif isinstance(self._radio_material, RadioMaterial):
            mat = self._scene.get(self._radio_material.name)
            if isinstance(mat, RadioMaterial) and self._overwrite_scene_radio_materials:
                # There exist a different instance of RadioMaterial with the same name but the user explicitely specified that 
                # he wants to replace existing scene materials by the one from the asset (even if the existing material is not a placeholder).
                mat.assign(self._radio_material)
                self._radio_material = mat 
            else:
                # The scene.add() method will trigger an error if there exist an item with the same name or non-placeholder radio material which is not the same object.
                # If there exist a placeholder radio material with same name, it will be updated with the asset radio material properties
                # If the asset's radio material is a radio material from the scene (same object), then the asset radio material will not be (re)added to the scene.
                # See scene.add() method for more details.
                self._scene.add(self._radio_material) 
                self._radio_material = self._scene.get(self._radio_material.name)

        else:
            raise TypeError(f"Asset 'radio_material' must be of type 'str', or 'RadioMaterial'")
        
    def _append_bsdfs(self):
        r"""
        Append BSDFs to the scene.

        This method appends/get radio material from/to the scene based on the BSDFs (Bidirectional Scattering Distribution Functions)
        defined in the asset's XML file. If no radio material is specified for the asset, it ensures that all BSDFs are added to the scene. 
        Existing BSDFs in the scene are overwritten if they are placeholders or if the `overwrite_scene_bsdfs` flag is set to `True`.

        Notes
        -----
        - If a BSDF with the same name already exists in the scene and is a placeholder,
        it is updated with the BSDF from the asset.
        - If a BSDF with the same name already exists in the scene and is not a placeholder,
        it is not overwritten unless the `overwrite_scene_bsdfs` flag is set to `True`.
        - If a radio material with the same name does not exist in the scene, a new
        placeholder radio material is created and added to the scene.

        Raises
        ------
        ValueError
            If a scene item with the same name but of a different type already
            exists in the scene.
        """
       
        # Get the root of the asset's XML 
        root_to_append = self._xml_tree.getroot() 

        bsdfs_to_append = root_to_append.findall('bsdf')  
        for bsdf in bsdfs_to_append:  
            bsdf_name = bsdf.get('id')
                
            # Change naming to adhere to Sionna conventions 
            mat_name = bsdf_name[4:] if bsdf_name.startswith('mat-') else bsdf_name

            bsdf.set('id',f"mat-{mat_name}")
            mat = self._scene.get(mat_name)

            if isinstance(mat, RadioMaterial):  # >>> A radio material exists in the scene with that name.
                if mat.bsdf.is_placeholder or self._overwrite_scene_bsdfs:
                    # If the material bsdf is a placeholder (or the user explicitely want to overwrite existing bsdfs) then the existing bsdf  
                    # is updated with the bsdf of the asset.
                    mat.bsdf.xml_element = bsdf
                # Else, if the material bsdf is not a placeholder then we keep the original scene bsdf
                
            else: # >>> mat is None or another class 
                # If the radio material does not exist and the name is available, then a placeholder is used. 
                # In this case, the user is in charge of defining the radio material properties afterward.
                # The asset's bsdf is affected to the newly material
                # If the radio material does not exist, but an item of the scene already uses that name, then scene.add() will raise an error.
                material_bsdf = BSDF(name=f"mat-{mat_name}",xml_element=bsdf)
                mat = RadioMaterial(mat_name, bsdf=material_bsdf)
                mat.is_placeholder = True
                self._scene.add(mat)

    def update_shape(self, key, value):
        self._shapes[key] = value

    def get_shape(self, key):
        return self._shapes.get(key)

    def remove_shape(self, key):
        if key in self._shapes:
            del self._shapes[key]

    def update_radio_material(self):
        r"""
        Ensure all asset's shapes share the same radio material.

        This method checks if all shapes associated with the asset share the same
        radio material. If they do, the asset's `radio_material` property is set
        to that material. If they do not, the `radio_material` property is set to `None`.

        Notes
        -----
        - This method is bypassed if the `_bypass_update` flag is set to `True`.
        - The method iterates over all shapes of the asset and collects their radio
        materials. If there is only one unique material, it is assigned to the
        asset's `radio_material` property. Otherwise, the property is set to `None`.
        """
        if not self._bypass_update:
            asset_mats = []
            for shape_name in self._shapes:
                shape = self._shapes[shape_name]
                shape_radio_material = shape.radio_material

                # Store the asset material(s)
                if shape_radio_material not in asset_mats:
                    asset_mats.append(shape_radio_material)
                
            if len(asset_mats) == 1:
                # If there is a single material used by all shapes of an asset, the general asset material property is set to that material
                self._radio_material = asset_mats[0]
            else:
                # Otherwise, it is set to None
                self._radio_material = None

    def update_velocity(self):
        r"""
        Ensure all asset's shapes share the same velocity.

        This method checks if all shapes associated with the asset share the same
        velocity. If they do, the asset's `velocity` property is set to that velocity.
        If they do not, the `velocity` property is set to `None`.

        Notes
        -----
        - This method is bypassed if the `_bypass_update` flag is set to `True`.
        - The method iterates over all shapes of the asset and collects their velocities.
        If all shapes have the same velocity, it is assigned to the asset's `velocity`
        property. Otherwise, the property is set to `None`.
        """
        if not self._bypass_update:

            shape_names = list(self._shapes.keys())

            first_shape_velocity = self._shapes[shape_names[0]].velocity

            for shape_name in shape_names[1:]:
                shape = self._shapes[shape_name]
                shape_velocity = shape.velocity

                if not np.array_equal(shape_velocity, first_shape_velocity):
                    # Not all shapes share the same velocity vector
                    self._velocity = None
                    return
                
                first_shape_velocity = shape_velocity
            # All shapes share the same velocity vector
            self._velocity = shape_velocity

    def look_at(self, target):
        # pylint: disable=line-too-long
        r"""
        Sets the orientation of the asset so that the x-axis points toward an
        ``Object``.

        Input
        -----
        target : [3], float | :class:`sionna.rt.Object` | str
            A position or the name or instance of an
            :class:`sionna.rt.Object` in the scene to point toward to
        """
        # Get position to look at
        if isinstance(target, str):
            if self._scene == None:
                err_msg = f"No scene have been affected to current AssetObject"
                raise TypeError(err_msg)
            
            obj = self.scene.get(target)
            if not isinstance(obj, Object) or not isinstance(obj, AssetObject):
                raise ValueError(f"No camera, device, asset or object named '{target}' found.")
            else:
                target = obj.position
        elif isinstance(target, Object) or isinstance(target, AssetObject):
            target = target.position
        else:
            target = tf.cast(target, dtype=self._rdtype)
            if not target.shape[0]==3:
                raise ValueError("`target` must be a three-element vector)")

        # Compute angles relative to LCS
        x = target - self.position
        x, _ = normalize(x)
        theta, phi = theta_phi_from_unit_vec(x)
        alpha = phi # Rotation around z-axis
        beta = theta-PI/2 # Rotation around y-axis
        gamma = 0.0 # Rotation around x-axis
        self.orientation = (alpha, beta, gamma)


           
    
    
# Module variables for example asset files
#
test_asset_1 = str(files(assets).joinpath("test/test_asset_1/test_asset_1.xml"))
# pylint: disable=C0301
"""
Example asset containing two 1x1x1m cubes spaced by 1m along the y-axis, with mat-itu_wood and mat-itu_metal materials.
"""

test_asset_2 = str(files(assets).joinpath("test/test_asset_2/test_asset_2.xml"))
# pylint: disable=C0301
"""
Example asset containing two 1x1x1m cubes spaced by 1m along the y-axis, with mat-custom_rm_1 and custom_rm_2 materials.
"""

test_asset_3 = str(files(assets).joinpath("test/test_asset_3/test_asset_3.xml"))
# pylint: disable=C0301
"""
Example asset containing a single 1x1x1m cubes with mat-itu_marble material.
"""

test_asset_4 = str(files(assets).joinpath("test/test_asset_4/test_asset_4.xml"))
# pylint: disable=C0301
"""
Example asset containing a single 1x1x1m cubes with mat-floor material.
"""

monkey = str(files(assets).joinpath("monkey/monkey.xml"))
# pylint: disable=C0301
"""
Example asset containing the famous "Suzanne" monkey head from Blender with mat-itu_marble material.
"""

body = str(files(assets).joinpath("body/body.xml"))
# pylint: disable=C0301
"""
Example asset containing a single body with mat-itu_marble material.
"""

two_persons = str(files(assets).joinpath("two_persons/two_persons.xml"))
# pylint: disable=C0301
"""
Example asset containing two persons with mat-itu_marble material.
"""