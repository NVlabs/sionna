#
# SPDX-FileCopyrightText: Copyright (c) 2024 ORANGE - Author: Guillaume Larue <guillaume.larue@orange.com>. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import numpy as np
import tensorflow as tf
import xml.etree.ElementTree as ET
import copy
import matplotlib.colors as mcolors
import warnings



class BSDF:
    # pylint: disable=line-too-long
    r"""BSDF(name, xml_element=None, color=None)

    Class implementing a BSDF descriptor

    BSDF stands for bidirectional scattering distribution function. Essentially, it's a mathematical function that determines the probability that a 
    specific ray of light will be reflected (scattered) at a given angle. Such functions is used within graphical engine for rendering. Here, we do
    not properly define a bsdf and bsdf related functions, but rather use this class as a data structure to store bsdf related information, such as the objects
    within a scene that use a given bsdf.

    Parameters
    ----------
    name : str
        Unique name of the bsdf
 
    xml_element : :class:`ET.Element`, optional 
        XML Element  instance from xml.etree.ElementTree.Element library (default is None).

    color: [3], `float`, or `str`, optional
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

        # Radio material
        self._radio_material = None

    @property
    def name(self):
        r"""
        str (read-only): Get the name of the BSDF.
        """
        return self._name        

    @property
    def xml_element(self):
        r"""
        Get/set the XML element description of the BSDF.

        Returns
        -------
        xml_element : :class:`ET.Element`
            The XML element descriptor of the BSDF.

        Parameters
        ----------
        xml_element : :class:`ET.Element`
            The new XML element descriptor of the BSDF (see ElementTree library).

        Raises
        ------
        TypeError
            If `xml_element` is not an instance of :class:`ET.Element`.
        ValueError
            If the root element is not <bsdf>.
        """
        return self._xml_element
       
    @xml_element.setter
    def xml_element(self, xml_element):
        
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

        if self.scene is not None:
            self.scene.append_to_xml(self._xml_element, overwrite=True)
            self.scene.reload()

    @property
    def rgb(self):
        r"""
        Get/set the RGB color of the BSDF.

        Returns
        -------
        rgb : list of float
            A list of 3 floats representing the RGB color of the BSDF, with values between 0 and 1.

        Parameters
        ----------
        rgb : list of float
            A list of 3 floats representing the RGB color, with values between 0 and 1.

        Raises
        ------
        TypeError
            If `rgb` is not a list of 3 floats between 0 and 1.
        """
        return self._rgb
    
    @rgb.setter
    def rgb(self, rgb):
        
        if len(rgb) != 3 or max(rgb) > 1 or min(rgb) < 0:
            raise TypeError("`rgb` must be a list of 3 floats comprised between 0 and 1")
        self._rgb = rgb
        self._color = None

        # Create a default bsdf element
        self._xml_element = ET.Element('bsdf', attrib={'type': 'diffuse', 'id': self._name, 'name': self._name})
        rgb_value = f"{self._rgb[0]} {self._rgb[1]} {self._rgb[2]}"
        ET.SubElement(self._xml_element, 'rgb', attrib={'value': rgb_value, 'name': 'reflectance'})
        
        self._is_placeholder = False

        if self.scene is not None:
            self.scene.append_to_xml(self._xml_element, overwrite=True)
            self.scene.reload()
        
    @property
    def color(self):
        r"""
        Get/set the color of the BSDF.

        Returns
        -------
        str or None
            Color name of the BSDF or None.

        Parameters
        ----------
        color : list of float or str
            RGB float triplet with values comprised between 0 and 1, or a color name.

        Raises
        ------
        TypeError
            If the input is not a list of 3 floats between 0 and 1, or a valid color name.
        """
        return self._color

    @color.setter
    def color(self, color):
        
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

        if self.scene is not None:
            self.scene.append_to_xml(self._xml_element, overwrite=True)
            self.scene.reload()

    def assign(self, b):
        r"""
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

        if self.scene is not None:
            self.scene.append_to_xml(self._xml_element, overwrite=True)
            self.scene.reload()

    ##############################################
    # Internal methods.
    # Should not be documented.
    ##############################################

    @property
    def use_counter(self):
        r"""
        int : Number of scene objects using this BSDF
        """
        if self.radio_material is not None:
            return self.radio_material.use_counter
        else:
            return set()

    @property
    def is_used(self):
        r"""
        bool : Indicator if the BSDF is used by at least one object of the scene
        """
        if self.radio_material is not None:
            return self.radio_material.is_used
        else:
            return False

    @property
    def using_objects(self):
        r"""
        [num_using_objects], tf.int : Identifiers of the objects using this BSDF
        """
        if self.radio_material is not None:
            return self.radio_material.using_objects
        else:
            return tf.cast(tuple(set()), tf.int32)

    @property
    def scene(self):
        r"""
        :class:`~sionna.rt.Scene` : Get/set the scene
        """
        if self._radio_material is not None:
            return self._radio_material.scene 
        else:
            return None

    @property
    def radio_material(self):
        r"""
        :class:`~sionna.rt.RadioMaterial` : Get/set the BSDF's RadioMaterial
        """
        return self._radio_material

    @radio_material.setter
    def radio_material(self, radio_material):
        if radio_material is not None:
            self._radio_material = radio_material

            self._name = f"mat-{self._radio_material.name}"
            self._xml_element.set('id',f"{self._name}")
            self._xml_element.set('name',f"{self._name}")
        else:
            self._radio_material = None

    @property
    def has_radio_material(self):
        r"""
        bool : Return True if the radio_material of the BSDF is set, False otherwise.
        """
        if self.radio_material is not None:
            return True
        else:
            return False

    @property
    def is_placeholder(self):
        r"""
        bool : Get/set if this bsdf is a placeholder. A bsdf is considered a placeholder when it has been randomly defined upon instantiation of a RadioMaterial(random rgb tuple)
        """
        return self._is_placeholder

    @is_placeholder.setter
    def is_placeholder(self, v):
        self._is_placeholder = v

    def set_scene(self, overwrite=False):
        r"""
        Set the BSDF XML to the scene.
        """
        if self.scene is not None:
            existing_bsdf = self.scene.append_to_xml(self._xml_element, overwrite=overwrite)
            
            if existing_bsdf is not None:
                if not overwrite:
                    self._xml_element = existing_bsdf
                self._is_placeholder = False
            self.scene.reload()

    def _color_name_to_rgb(self,color):
        r"""
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
        