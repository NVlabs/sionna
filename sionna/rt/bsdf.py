#
# SPDX-FileCopyrightText: Copyright (c) 2024 ORANGE - Author: Guillaume Larue <guillaume.larue@orange.com>. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import numpy as np
import tensorflow as tf
import xml.etree.ElementTree as ET
import copy

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
 
    xml_element : `:class:~Element`, optional 
        XML Element  instance from xml.etree.ElementTree.Element library (default is None).

    rgb: tuple of 3x float, optional
        RGB color of the asset (default is None).
    """

    def __init__(self,
                name,
                xml_element = None,
                rgb = None,
                ):

        if not isinstance(name, str):
            raise TypeError("`name` must be a string")
        self._name = name


        if xml_element is not None:
            # Store the specified XML descriptor
            self._xml_element = xml_element
            self._xml_element.set('id',f"{self._name}")
            self._xml_element.set('name',f"{self._name}")  
       
        else:
            if rgb is None:
                # If neither RGB or XML element are specified a random color is chosen
                rgb = np.random.rand(3)
            elif len(rgb) != 3 and max(rgb) > 1 and min(rgb) < 0:
                raise TypeError("`rgb` must be a list of 3 floats comprised between 0 and 1")
            self._rgb = rgb

            # Create a default RGB bsdf element
            self._xml_element = ET.Element('bsdf', attrib={'type': 'diffuse', 'id': self._name, 'name': self._name})
            rgb_value = f"{self._rgb[0]} {self._rgb[1]} {self._rgb[2]}"
            ET.SubElement(self._xml_element, 'rgb', attrib={'value': rgb_value, 'name': 'reflectance'})
        

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
        self._xml_element.set('id',f"{self._name}")
        self._xml_element.set('name',f"{self._name}") 

        # if self._scene is not None:
        #     # When the name of the bsdf is updated, we must also update the XML file:
        #     # - The bsdf name must be changed
        #     # - The corresponding shapes's bsdf references must be updated
        #     self._scene.append_to_xml(self._xml_element, overwrite=True)
            
        #     for object_id in self.using_objects:
        #         for shape_name in self._scene._scene_objects:
        #             shape = self._scene._scene_objects[shape_name]
        #             shape_id = shape.object_id
        #             if shape_id == object_id:
        #                 self._scene.update_shape_bsdf_xml(shape_name, self._name)
        #                 break 
            
        #     self._scene.reload_scene()

    @property
    def xml_element(self):
        """
        str (read-only) : XML tree description of the BSDF
        """
        return self._xml_element
    
    @property
    def rgb(self):
        """
        str (read-only) : Color of the BSDF
        """
        return self._rgb
    
    @xml_element.setter
    def xml_element(self, xml_element):
        """
        `:~class:Element:`  : XML tree (see ElementTree library)
        """
        if not isinstance(xml_element, ET.Element):
            raise TypeError("`element` must be an ET.Element descriptor of a BSDF.")

        # Check if the root element is <bsdf>
        if xml_element.tag != 'bsdf':
            raise ValueError("The root element must be <bsdf>.")

        # Store the specified XML descriptor and change the name to match that of the bsdf/material
        self._xml_element = xml_element
        self._xml_element.set('id',f"{self._name}")
        self._xml_element.set('name',f"{self._name}")  
        
        self._is_placeholder = False

        if self._scene is not None:
            self._scene.append_to_xml(self._xml_element, overwrite=True)
            self._scene.reload_scene()

    # # Setter for the xml_element argument with an additional parameter, allowing to control the update of the scene
    # def set_xml_element(self, xml_element, update_scene): 
    #     """
    #     `:~class:Element:`  : XML tree (see ElementTree library)
    #     """
    #     self._xml_element = xml_element
    #     self._xml_element.set('id',f"{self._name}")
    #     self._xml_element.set('name',f"{self._name}")  
        
    #     self._is_placeholder = False

    #     if self._scene is not None and update_scene:
    #         self.update_scene_bsdf()

    @rgb.setter
    def rgb(self, rgb):
        """
        `:list[float, float, float]:`  : RGB float triplet with values comprised between 0 and 1
        """
        if len(rgb) != 3 or max(rgb) > 1 or min(rgb) < 0:
            raise TypeError("`rgb` must be a list of 3 floats comprised between 0 and 1")
        self._rgb = rgb

        # Create a default bsdf element
        self._xml_element = ET.Element('bsdf', attrib={'type': 'diffuse', 'id': self._name, 'name': self._name})
        rgb_value = f"{self._rgb[0]} {self._rgb[1]} {self._rgb[2]}"
        ET.SubElement(self._xml_element, 'rgb', attrib={'value': rgb_value, 'name': 'reflectance'})
        
        self._is_placeholder = False

        if self._scene is not None:
            self._scene.append_to_xml(self._xml_element, overwrite=True)
            self._scene.reload_scene()

    # def update_scene_bsdf(self):
    #     self._scene.update_bsdf(self)

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
        
        # # Update the corresponding shapes's XML to reference the correct bsdf
        # for shape_name in self._scene._scene_objects:
        #     shape = self._scene._scene_objects[shape_name]
        #     shape_id = shape.object_id
        #     if shape_id == object_id:
        #         self._scene.update_shape_bsdf_xml(shape_name, self._name)
        #         break 

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
        if scene is not None:
            self._scene = scene

            existing_bsdf = self._scene.append_to_xml(self._xml_element, overwrite=False)
            
            if existing_bsdf is not None:
                self._xml_element = existing_bsdf
                self._is_placeholder = False
            self._scene.reload_scene()

    def assign(self, b):
        """
        Assign new values to the BSDF properties from another
        BSDF ``b``.

        Input
        ------
        b : :class:`~sionna.rt.BSDF
            BSDF from which to assign the new values
        """
        if not isinstance(b, BSDF):
            raise TypeError("`b` is not a BSDF")

        # When assigning a the bsdf attributes we should not use the same objects:
        self._rgb = b.rgb.copy()
        self._xml_element = copy.deepcopy(b.xml_element)

        # Since assign method does not replace the object itself, the we should update the name of the assigned bsdf before updating xml file
        self._xml_element.set('id',f"{self._name}")
        self._xml_element.set('name',f"{self._name}")  

        existing_bsdf = self._scene.append_to_xml(self._xml_element, overwrite=True)

        # for obj_id in b.using_objects:
        #     self.add_object_using(obj_id)

        self._scene.reload_scene()

    @property
    def is_placeholder(self):
        """
        bool : Get/set if this bsdf is a placeholder. A bsdf is considered a placeholder when it has been randomly defined upon instantiation of a RadioMaterial(random rgb tuple)
        """
        return self._is_placeholder

    @is_placeholder.setter
    def is_placeholder(self, v):
        self._is_placeholder = v

    
