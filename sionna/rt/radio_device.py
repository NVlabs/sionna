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
    r"""RadioDevice(name, position, orientation=[0.,0.,0.], look_at=None, trainable_position=False, trainable_orientation=False, dtype=tf.complex64)

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

    trainable_position : bool
        Determines if the ``position`` is a trainable variable or not.
        Defaults to `False`.

    trainable_orientation : bool
        Determines if the ``orientation`` is a trainable variable or not.
        Defaults to `False`.

    dtype : tf.complex
        Datatype to be used in internal calculations.
        Defaults to `tf.complex64`.
    """

    def __init__(self,
                 name,
                 position,
                 orientation=(0.,0.,0.),
                 look_at=None,
                 trainable_position=False,
                 trainable_orientation=False,
                 dtype=tf.complex64):

        if dtype not in (tf.complex64, tf.complex128):
            raise ValueError("`dtype` must be tf.complex64 or tf.complex128`")
        self._dtype = dtype
        self._rdtype = dtype.real_dtype

        self._position = tf.Variable(tf.zeros([3], self._rdtype))
        self._orientation = tf.Variable(tf.zeros([3], self._rdtype))

        self.trainable_position = trainable_position
        self.trainable_orientation = trainable_orientation

        # Initialize the base class OrientedObject
        # Position and orientation are set through this call
        super().__init__(name, position, orientation, look_at)

    @property
    def trainable_position(self):
        """
        bool : Get/set if the position is a trainable variable
            or not.
            Defaults to `False`.
        """
        return self._trainable_position

    @trainable_position.setter
    def trainable_position(self, value):
        if not isinstance(value, bool):
            raise TypeError("`trainable_position` must be bool")
        # pylint: disable=protected-access
        self._position._trainable = value
        self._trainable_position = value

    @property
    def trainable_orientation(self):
        """
        bool : Get/set if the orientation is a trainable variable
            or not.
            Defaults to `False`.
        """
        return self._trainable_orientation

    @trainable_orientation.setter
    def trainable_orientation(self, value):
        if not isinstance(value, bool):
            raise TypeError("`trainable_orientation` must be bool")
        # pylint: disable=protected-access
        self._orientation._trainable = value
        self._trainable_orientationn = value

    @property
    def position(self):
        """
        [3], tf.float : Get/set the position
        """
        return self._position.value()

    @position.setter
    def position(self, new_position):
        new_position = tf.cast(new_position, dtype=self._rdtype)
        if not (tf.rank(new_position) == 1 and new_position.shape[0] == 3):
            msg = "Position must be shaped as [x,y,z] (rank=1 and shape=[3])"
            raise ValueError(msg)
        self._position.assign(new_position)

    @property
    def orientation(self):
        """
        [3], tf.float : Get/set the orientation
        """
        return self._orientation.value()

    @orientation.setter
    def orientation(self, new_orient):
        new_orient = tf.cast(new_orient, dtype=self._rdtype)
        if not (tf.rank(new_orient) == 1 and new_orient.shape[0] == 3):
            msg = "Orientation must be shaped as [a,b,c] (rank=1 and shape=[3])"
            raise ValueError(msg)
        self._orientation.assign(new_orient)

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
