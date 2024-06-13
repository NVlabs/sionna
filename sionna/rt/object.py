#
# SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""
Abstract Base class for cameras, radios, and scene objects.
"""

from abc import ABC
from abc import abstractmethod


class Object(ABC):
    # pylint: disable=line-too-long
    r"""Object(name)

    Baseclass for implementing items that are part of the scene.

    Input
    -----
    name : str
        Name of the object

    position : [3], float
        Position :math:`(x,y,z)` as three-dimensional vector.
        Defaults to `None`, in which case the position is not set at init.

    orientation : [3], float | None
        Orientation :math:`(\alpha, \beta, \gamma)` specified
        through three angles corresponding to a 3D rotation
        as defined in :eq:`rotation`.
        This parameter is ignored if ``look_at`` is not `None`.

    look_at : [3], float | :class:`sionna.rt.Object` | None
        A position or the instance of an :class:`~sionna.rt.Object` to
        point toward to.
    """

    # The following names are reserved and therefore cannot be used by objects
    RESERVED_NAMES = ('preview',)

    def __init__(self, name, position=None, orientation=None,
                 look_at=None, **kwargs):

        # Set the name
        if not isinstance(name, str):
            raise TypeError("`name` must be a string")
        if name in Object.RESERVED_NAMES:
            msg = f"Cannot use name '{name}' as it is reserved."
            raise ValueError(msg)
        self._name = name

        # Scene
        self._scene = None

        if position is not None:
            self.position = position

        if look_at is None:
            if orientation is not None:
                self.orientation = orientation
        else:
            self.look_at(look_at)

        super().__init__(**kwargs)

    @property
    def name(self):
        """
        str (read-only) : Name
        """
        return self._name

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
        ``Object``.

        Input
        -----
        target : [3], float | :class:`sionna.rt.Object` | str
            A position or the name or instance of an
            :class:`sionna.rt.Object` in the scene to point toward to
        """
        pass

    ##############################################
    # Internal methods.
    # Should not be appear in the end user
    # documentation
    ##############################################

    @property
    def scene(self):
        """
        :class:`~sionna.rt.Scene` : Get/set the scene
        """
        return self._scene

    @scene.setter
    def scene(self, scene):
        self._scene = scene
