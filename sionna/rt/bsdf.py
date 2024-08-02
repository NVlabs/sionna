#
# SPDX-FileCopyrightText: Copyright (c) 2024 ORANGE - Author: Guillaume Larue <guillaume.larue@orange.com>. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import numpy as np
import tensorflow as tf
import xml.etree.ElementTree as ET
import copy
import matplotlib.colors as mcolors



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
 
    xml_element : :class:`ET.Element`, optional 
        XML Element  instance from xml.etree.ElementTree.Element library (default is None).

    color: [3], float, or `str`, optional
        RGB color triplet or color name `str` of the asset (default is None).
    """

    def __init__(self,
                name,
                xml_element = None,
                color = None,
                ):

        if not isinstance(name, str):
            raise TypeError("`name` must be a string")
        self._name = name


        if xml_element is not None:
            # Store the specified XML descriptor
            self._xml_element = xml_element
            self._xml_element.set('id',f"{self._name}")
            self._xml_element.set('name',f"{self._name}")  

            self._rgb = None
            self._color = None
       
        else:
            if color is None:
                # If neither RGB or XML element are specified a random color is chosen
                rgb = np.random.rand(3)
            elif isinstance(color,str):
                (color,rgb) = self._color_name_to_rgb(color)
            elif len(color) == 3 and max(color) <= 1 and min(color) >= 0:
                rgb = color
                color = None
            else:
                raise TypeError("`color` must be a list of 3 `float` between 0 and 1, or a valid `str` color name.")                

            self._rgb = rgb
            self._color = color

            # Create a default RGB bsdf element
            self._xml_element = ET.Element('bsdf', attrib={'type': 'diffuse', 'id': self._name, 'name': self._name})
            rgb_value = f"{self._rgb[0]} {self._rgb[1]} {self._rgb[2]}"
            ET.SubElement(self._xml_element, 'rgb', attrib={'value': rgb_value, 'name': 'reflectance'})
        

        # By default we assume the bsdf not to be a placeholder (i.e. it is defined by the user). 
        # When the bsdf is automatically instantiated, e.g. during the creation of a material, then _is_placeholder argument is set to True by the parent object.    
        self._is_placeholder = False

        # Set of objects identifiers that use this bsdf
        self._objects_using = set()

        # Scene 
        self._scene = None

    @property
    def name(self):
        """
        str: Get the name of the BSDF.
        """
        return self._name
    
    @name.setter
    def name(self, name):
        """
        Set the name of the BSDF.

        Parameters
        ----------
        name : str
            The new name of the BSDF.
        """
        self._name = name
        self._xml_element.set('id',f"{self._name}")
        self._xml_element.set('name',f"{self._name}") 

    @property
    def xml_element(self):
        """
        :class:`ET.Element` : XML tree description of the BSDF.
        """
        return self._xml_element
        
    @xml_element.setter
    def xml_element(self, xml_element):
        """
        Set the XML element of the BSDF.

        Parameters
        ----------
        :class:`ET.Element` : ET.Element
            new XML element descriptor of the bsdf(see ElementTree library).

        Raises
        ------
        TypeError
            If `xml_element` is not an instance of ET.Element.
        ValueError
            If the root element is not <bsdf>.
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

        # Set color and rgb to `None` since rgb or color are not always present in a complex bsdf xml descriptor
        self._rgb = None
        self._color = None

        if self._scene is not None:
            self._scene.append_to_xml(self._xml_element, overwrite=True)
            self._scene.reload()

    @property
    def rgb(self):
        """
        [3], float : RGB color of the BSDF.
        """
        return self._rgb
    
    @rgb.setter
    def rgb(self, rgb):
        """
        Set the RGB color of the BSDF.

        Parameters
        ----------
        rgb : [3], float
            RGB float triplet with values comprised between 0 and 1.

        Raises
        ------
        TypeError
            If `rgb` is not a list of 3 floats between 0 and 1.
        """
        if len(rgb) != 3 or max(rgb) > 1 or min(rgb) < 0:
            raise TypeError("`rgb` must be a list of 3 floats comprised between 0 and 1")
        self._rgb = rgb
        self._color = None

        # Create a default bsdf element
        self._xml_element = ET.Element('bsdf', attrib={'type': 'diffuse', 'id': self._name, 'name': self._name})
        rgb_value = f"{self._rgb[0]} {self._rgb[1]} {self._rgb[2]}"
        ET.SubElement(self._xml_element, 'rgb', attrib={'value': rgb_value, 'name': 'reflectance'})
        
        self._is_placeholder = False

        if self._scene is not None:
            self._scene.append_to_xml(self._xml_element, overwrite=True)
            self._scene.reload()

    @property
    def color(self):
        """
        str or None : Color name of the BSDF or None.
        """
        return self._color

    @color.setter
    def color(self, color):
        """
        Set the color of the BSDF.

        Parameters
        ----------
        color : [3], float or `str`
            RGB float triplet with values comprised between 0 and 1, or a color name.

        Raises
        ------
        TypeError
            If the input is not a list of 3 floats between 0 and 1, or a valid color name.
        """
        if isinstance(color, str):
            color,rgb = self._color_name_to_rgb(color)
        elif len(color) == 3 and max(color) <= 1 and min(color) >= 0:
            rgb = color
            color = None
        else:
            raise TypeError("`color` must be a list of 3 `float` between 0 and 1, or a valid `str` color name.")

        self._rgb = rgb
        self._color = color

        # Create a default bsdf element
        self._xml_element = ET.Element('bsdf', attrib={'type': 'diffuse', 'id': self._name, 'name': self._name})
        rgb_value = f"{self._rgb[0]} {self._rgb[1]} {self._rgb[2]}"
        ET.SubElement(self._xml_element, 'rgb', attrib={'value': rgb_value, 'name': 'reflectance'})
        
        self._is_placeholder = False

        if self._scene is not None:
            self._scene.append_to_xml(self._xml_element, overwrite=True)
            self._scene.reload()

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
        self._rgb = copy.copy(b.rgb)
        self._color = copy.copy(b.color)
        self._xml_element = copy.deepcopy(b.xml_element)

        # Since assign method does not replace the object itself, the we should update the name of the assigned bsdf before updating xml file
        self._xml_element.set('id',f"{self._name}")
        self._xml_element.set('name',f"{self._name}")
        
        # No that a BSDF has been assigned, the BSDF is not a placeholder anymore
        self.is_placeholder = False

        if self._scene is not None:
            self._scene.append_to_xml(self._xml_element, overwrite=True)
            self._scene.reload()

    ##############################################
    # Internal methods.
    # Should not be documented.
    ##############################################

    def _color_name_to_rgb(self,color):
        """
        Convert a color name to an RGB triplet.

        Parameters
        ----------
        color : str
            The name of the color.

        Returns
        -------
        tuple
            A tuple containing the color name and the corresponding RGB triplet.

        Raises
        ------
        TypeError
            If `color` is not a string or if the color name is unknown.
        """
        if not isinstance(color,str):
            raise TypeError("`color` must be a `str`.")
        
        # Dictionary mapping color names to RGB values using matplotlib
        color_names = {name: mcolors.to_rgb(color) for name, color in mcolors.CSS4_COLORS.items()}

        # If a color name is provided, get the corresponding RGB value
        color = color.lower()
        if color in color_names:
            rgb = color_names[color]
            return (color,rgb)
        else:
            raise TypeError(f"Unknown color name '{color}'.")

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

    def add_object_using(self, object_id):
        """
        Add an object to the set of objects using this BSDF
        """
        self._objects_using.add(object_id)
        
    def discard_object_using(self, object_id):
        """
        Remove an object from the set of objects using this BSDF
        """
        assert object_id in self._objects_using,\
            f"Object with id {object_id} is not in the set of {self.name}"
        self._objects_using.discard(object_id)

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
            self._scene.reload()

    @property
    def is_placeholder(self):
        """
        bool : Get/set if this bsdf is a placeholder. A bsdf is considered a placeholder when it has been randomly defined upon instantiation of a RadioMaterial(random rgb tuple)
        """
        return self._is_placeholder

    @is_placeholder.setter
    def is_placeholder(self, v):
        self._is_placeholder = v


    
