#
# SPDX-FileCopyrightText: Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""
Abstract Base class for cameras, radios, and scene objects.
"""

from abc import abstractmethod
from .object import Object


class OrientedObject(Object):
    # pylint: disable=line-too-long
    r"""OrientedObject(name, position, orientation=[0.,0.,0.], look_at=None)

    Baseclass for implementing objects that are part of the scene and with
    editable localization and orientation.

    Input
    -----
    name : str
        Name of the object

    position : [3], float
        Position :math:`(x,y,z)` as three-dimensional vector

    orientation : [3], float
        Orientation :math:`(\alpha, \beta, \gamma)` specified
        through three angles corresponding to a 3D rotation
        as defined in :eq:`rotation`.
        This parameter is ignored if ``look_at`` is not `None`.
        Defaults to [0,0,0].

    look_at : [3], float | :class:`sionna.rt.OrientedObject` | None
        A position or the instance of an :class:`~sionna.rt.OrientedObject` to
        point toward to.
        If set to `None`, then ``orientation`` is used to orientate the object.
    """

    def __init__(self, name, position, orientation=(0.,0.,0.), look_at=None):

        self.position = position
        if look_at is None:
            self.orientation = orientation
        else:
            self.look_at(look_at)

        super().__init__(name)

    @property
    @abstractmethod
    def position(self):
        """
        [3], tf.float : Get/set the position
        """
        pass

    @position.setter
    @abstractmethod
    def position(self, new_position):
        pass

    @property
    @abstractmethod
    def orientation(self):
        """
        [3], tf.float : Get/set the orientation
        """
        pass

    @orientation.setter
    @abstractmethod
    def orientation(self, new_orient):
        pass

    @abstractmethod
    def look_at(self, target):
        # pylint: disable=line-too-long
        r"""
        Sets the orientation so that the x-axis points toward an
        ``OrientedObject``.

        Input
        -----
        target : [3], float | :class:`sionna.rt.OrientedObject` | str
            A position or the name or instance of an
            :class:`sionna.rt.OrientedObject` in the scene to point toward to
        """
        pass
