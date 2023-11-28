#
# SPDX-FileCopyrightText: Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""
Class implementing a radio device, which can be either a transmitter or a
receiver.
"""

import tensorflow as tf

from .oriented_object import OrientedObject
from .utils import normalize, theta_phi_from_unit_vec
from sionna.constants import PI


class RadioDevice(OrientedObject):
    # pylint: disable=line-too-long
    r"""RadioDevice(name, position, orientation=[0.,0.,0.], look_at=None, dtype=tf.complex64)

    Class defining a generic radio device.

    :class:`~sionna.rt.Transmitter` and :class:`~sionna.rt.Receiver`
    inherit from this class and should be used.

    Parameters
    ----------
    name : str
        Name

    position : [3], float
        Position :math:`(x,y,z)` as three-dimensional vector

    orientation : [3], float
        Orientation :math:`(\alpha, \beta, \gamma)` specified
        through three angles corresponding to a 3D rotation
        as defined in :eq:`rotation`.
        This parameter is ignored if ``look_at`` is not `None`.
        Defaults to [0,0,0].

    look_at : [3], float | :class:`~sionna.rt.Transmitter` | :class:`~sionna.rt.Receiver` | :class:`~sionna.rt.Camera` | None
        A position or the instance of a :class:`~sionna.rt.Transmitter`,
        :class:`~sionna.rt.Receiver`, or :class:`~sionna.rt.Camera` to look at.
        If set to `None`, then ``orientation`` is used to orientate the device.

    color : [3], float
        Defines the RGB (red, green, blue) ``color`` parameter for the device as displayed in the previewer and renderer.
        Each RGB component must have a value within the range :math:`\in [0,1]`.

    dtype : tf.complex
        Datatype to be used in internal calculations.
        Defaults to `tf.complex64`.
    """

    def __init__(self,
                 name,
                 position,
                 orientation=(0.,0.,0.),
                 look_at=None,
                 color=(0,0,0),
                 dtype=tf.complex64):

        if dtype not in (tf.complex64, tf.complex128):
            raise ValueError("`dtype` must be tf.complex64 or tf.complex128`")
        self._dtype = dtype
        self._rdtype = dtype.real_dtype
        self.position = position
        self.orientation = orientation
        self.color = color

        # Initialize the base class OrientedObject
        # Position and orientation are set through this call
        super().__init__(name, position, orientation, look_at)

    @property
    def position(self):
        """
        [3], tf.float : Get/set the position
        """
        return self._position

    @position.setter
    def position(self, v):
        if isinstance(v, tf.Variable):
            if v.dtype != self._rdtype:
                msg = f"`position` must have dtype={self._rdtype}"
                raise TypeError(msg)
            else:
                self._position = v
        else:
            self._position = tf.cast(v, dtype=self._rdtype)

    @property
    def orientation(self):
        """
        [3], tf.float : Get/set the orientation
        """
        return self._orientation

    @orientation.setter
    def orientation(self, v):
        if isinstance(v, tf.Variable):
            if v.dtype != self._rdtype:
                msg = f"`orientation` must have dtype={self._rdtype}"
                raise TypeError(msg)
            else:
                self._orientation = v
        else:
            self._orientation = tf.cast(v, dtype=self._rdtype)

    def look_at(self, target):
        # pylint: disable=line-too-long
        r"""
        Sets the orientation so that the x-axis points toward a
        position, radio device, or camera.

        Given a point :math:`\mathbf{x}\in\mathbb{R}^3` with spherical angles
        :math:`\theta` and :math:`\varphi`, the orientation of the radio device
        will be set equal to :math:`(\varphi, \frac{\pi}{2}-\theta, 0.0)`.

        Input
        -----
        target : [3], float | :class:`~sionna.rt.Transmitter` | :class:`~sionna.rt.Receiver` | :class:`~sionna.rt.Camera` | str
            A position or the name or instance of a
            :class:`~sionna.rt.Transmitter`, :class:`~sionna.rt.Receiver`, or
            :class:`~sionna.rt.Camera` in the scene to look at.
        """
        # Get position to look at
        if isinstance(target, str):
            obj = self.scene.get(target)
            if not isinstance(obj, OrientedObject):
                raise ValueError(f"No camera or device named '{target}' found.")
            else:
                target = obj.position
        elif isinstance(target, OrientedObject):
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

    @property
    def color(self):
        r"""
        [3], float : Get/set the the RGB (red, green, blue) color for the device as displayed in the previewer and renderer.
        Each RGB component must have a value within the range :math:`\in [0,1]`.
        """
        return self._color

    @color.setter
    def color(self, new_color):
        new_color = tf.cast(new_color, dtype=self._rdtype)
        if not (tf.rank(new_color) == 1 and new_color.shape[0] == 3):
            msg = "Color must be shaped as [r,g,b] (rank=1 and shape=[3])"
            raise ValueError(msg)
        if tf.reduce_any(new_color < 0.) or tf.reduce_any(new_color > 1.):
            msg = "Color components must be in the range (0,1)"
            raise ValueError(msg)
        self._color = new_color
