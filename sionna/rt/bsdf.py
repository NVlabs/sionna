#
# SPDX-FileCopyrightText: Copyright (c) 2024 ORANGE - Author: Guillaume Larue <guillaume.larue@orange.com>. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import tensorflow as tf
import xml.etree.ElementTree as ET

from . import scene


class BSDF:
    # pylint: disable=line-too-long
    r"""
    Class implementing a BSDF descriptor

    BSDF stands for bidirectional scattering distribution function. Essentially, it's a mathematical function that determines the probability that a 
    specific ray of light will be reflected (scattered) at a given angle. Such functions is used within graphical engine for rendering. Here, we do
    not properly define a bsdf and bsdf related functions, but rather use this class as a data structure to store bsdf related information, such as the objects
    within a scene that use a given bsdf.
   
    

    Parameters
    -----------
    name : str
        Unique name of the bsdf
 
    xml_tree : `:class:~Element`, optional 
        XML Element tree instance from xml.etree.ElementTree library (default is None).

    rgb: tuple of 3x float, optional
        RGB color of the asset (default is None).
    """

    # TODO: - Update scene after change of BSDF?
    def __init__(self,
                name,
                xml_tree = None,
                rgb = None,
                ):

        if not isinstance(name, str):
            raise TypeError("`name` must be a string")
        self._name = name

        # XML Tree or RGB color of the BSDF
        if xml_tree is None and rgb is None:
            raise TypeError("Either `xml_tree` or `rgb` must be set")
        
        if xml_tree is not None:
            # Store the specified XML descriptor
            self._xml_tree = xml_tree
            self._xml_tree.set('id',f"{self._name}")
            self._xml_tree.set('name',f"{self._name}")  
       
        else:
            if len(rgb) != 3 and max(rgb) > 1 and min(rgb) < 0:
                raise TypeError("`rgb` must be a list of 3 floats comprised between 0 and 1")
            self._rgb = rgb

            # Create a default bsdf element
            self._xml_tree = ET.Element('bsdf', attrib={'type': 'diffuse', 'id': self._name, 'name': self._name})
            rgb_value = f"{self._rgb[0]} {self._rgb[1]} {self._rgb[2]}"
            ET.SubElement(self._xml_tree, 'rgb', attrib={'value': rgb_value, 'name': 'reflectance'})

        # By default we assume the bsdf not to be a placeholder (i.e. it is defined by the user). 
        # When the bsdf is automatically instantiated during the creation of a material, then _is_placeholder argument is set to True    
        self._is_placeholder = False
            

        # Set of objects identifiers that use this bsdf
        self._objects_using = set()

        # Scene 
        self._scene = None




    @property
    def name(self):
        """
        str (read-only) : Name of the BSDF
        """
        return self._name
    
    @name.setter
    def name(self, name):
        """
        str (read-only) : Name of the BSDF
        """
        self._name = name
        self._xml_tree.set('id',f"{self._name}")
        self._xml_tree.set('name',f"{self._name}") 

    @property
    def xml_tree(self):
        """
        str (read-only) : XML tree description of the BSDF
        """
        return self._xml_tree
    
    @property
    def rgb(self):
        """
        str (read-only) : Color of the BSDF
        """
        return self._rgb
    
    @xml_tree.setter
    def xml_tree(self, xml_tree):
        """
        `:~class:Element:`  : XML tree (see ElementTree library)
        """
        # Store the specified XML descriptor and change the name to match that of the bsdf/material
        self._xml_tree = xml_tree
        self._xml_tree.set('id',f"{self._name}")
        self._xml_tree.set('name',f"{self._name}")  
        
        self._is_placeholder = False

        if self._scene is not None:
            self.update_scene_bsdf()

    # Setter for the xml_tree argument with an additional parameter, allowing to control the update of the scene
    def set_xml_tree(self, xml_tree, update_scene): 
        """
        `:~class:Element:`  : XML tree (see ElementTree library)
        """
        self._xml_tree = xml_tree
        self._xml_tree.set('id',f"{self._name}")
        self._xml_tree.set('name',f"{self._name}")  
        
        self._is_placeholder = False

        if self._scene is not None and update_scene:
            self.update_scene_bsdf()

    @rgb.setter
    def rgb(self, rgb):
        """
        `:~class:Element:`  : XML tree (see ElementTree library)
        """
        if len(rgb) != 3 and max(rgb) > 1 and min(rgb) < 0:
            raise TypeError("`rgb` must be a list of 3 floats comprised between 0 and 1")
        self._rgb = rgb

        # Create a default bsdf element
        self._xml_tree = ET.Element('bsdf', attrib={'type': 'diffuse', 'id': self._name, 'name': self._name})
        rgb_value = f"{self._rgb[0]} {self._rgb[1]} {self._rgb[2]}"
        ET.SubElement(self._xml_tree, 'rgb', attrib={'value': rgb_value, 'name': 'reflectance'})
        
        self._is_placeholder = False

        if self._scene is not None:
            self.update_scene_bsdf()

    def update_scene_bsdf(self):
        self._scene.update_bsdf(self)

    @property
    def use_counter(self):
        """
        int : Number of scene objects using this BSDF
        """
        return len(self._objects_using)

    @property
    def is_used(self):
        """bool : Indicator if the BSDF is used by at least one object of
        the scene"""
        return self.use_counter > 0

    @property
    def using_objects(self):
        """
        [num_using_objects], tf.int : Identifiers of the objects using this
        BSDF
        """
        tf_objects_using = tf.cast(tuple(self._objects_using), tf.int32)
        return tf_objects_using
        #return self._objects_using#tf_objects_using

    def add_object_using(self, object_id):
        """
        Add an object to the set of objects using this BSDF
        """
        self._objects_using.add(object_id)
        #self._objects_using.append(object_id)

    def discard_object_using(self, object_id):
        """
        Remove an object from the set of objects using this BSDF
        """
        assert object_id in self._objects_using,\
            f"Object with id {object_id} is not in the set of {self.name}"
        self._objects_using.discard(object_id)
        #self._objects_using.pop(object_id)

    def reset_objects_using(self):
        self._objects_using = set()

    @property
    def scene(self):
        """
        :class:`~sionna.rt.Scene` : Get/set the scene
        """
        return self._scene

    @scene.setter
    def scene(self, scene):
        self._scene = scene

    @property
    def is_placeholder(self):
        """
        bool : Get/set if this bsdf is a placeholder. A bsdf is considered a placeholder when it has been randomly defined upon instantiation of a RadioMaterial(random rgb tuple)
        """
        return self._is_placeholder

    @is_placeholder.setter
    def is_placeholder(self, v):
        self._is_placeholder = v

    
