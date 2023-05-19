#
# SPDX-FileCopyrightText: Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""
Class representing objects in the scene
"""

from .object import Object
from .radio_material import RadioMaterial


class SceneObject(Object):
    # pylint: disable=line-too-long
    r"""
    SceneObject()

    Every object in the scene is implemented by an instance of this class
    """

    def __init__(self,
                 name,
                 radio_material=None):

        # Initialize the base class Object
        super().__init__(name)

        # Set the radio material
        self.radio_material = radio_material

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
        # Note: _radio_material is set at __init__, but pylint doesn't see it.
        if mat is None:
            mat_obj = None

        elif isinstance(mat, str):
            mat_obj = self.scene.get(mat)
            if (mat_obj is None) or (not isinstance(mat_obj, RadioMaterial)):
                err_msg = f"Unknown radio material '{mat}'"
                raise TypeError(err_msg)

        elif not isinstance(mat, RadioMaterial):
            err_msg = ("The material must be a material name (str) or an "
                        "instance of RadioMaterial")
            raise TypeError(err_msg)

        else:
            mat_obj = mat

        # Decrease the use counter of the current radio material
        # pylint: disable=access-member-before-definition
        if hasattr(self, '_radio_material') and self._radio_material:
            self._radio_material.decrease_use()
        # Assign the new material
        # pylint: disable=access-member-before-definition
        self._radio_material = mat_obj

        # If the radio material is set to None, we can stop here
        # pylint: disable=access-member-before-definition
        if not self._radio_material:
            return

        # Increase the use counter of the newly used material
        # pylint: disable=access-member-before-definition
        self._radio_material.increase_use()

        # Add the RadioMaterial to the scene if not already done
        self.scene.add(self._radio_material)
