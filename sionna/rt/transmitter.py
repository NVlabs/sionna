#
# SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""
Class implementing a transmitter
"""

import tensorflow as tf
from .radio_device import RadioDevice
from .utils import dbm_to_watt

class Transmitter(RadioDevice):
    # pylint: disable=line-too-long
    r"""
    Class defining a transmitter

    The ``position``, ``orientation``, and ``power_dbm`` properties can be assigned to a TensorFlow
    variable or tensor. In the latter case, the tensor can be the output of a callable,
    such as a Keras layer implementing a neural network. In the former case, it
    can be set to a trainable variable:

    .. code-block:: Python

        tx = Transmitter(name="my_tx",
                         position=tf.Variable([0, 0, 0], dtype=tf.float32),
                         orientation=tf.Variable([0, 0, 0], dtype=tf.float32),
                         power_dbm=tf.Variable(44, dtype=tf.float32))

    Parameters
    ----------
    name : str
        Name

    position : [3], float
        Position :math:`(x,y,z)` [m] as three-dimensional vector

    power_dbm: float
        Transmit power [dBm]

    orientation : [3], float
        Orientation :math:`(\alpha, \beta, \gamma)` [rad] specified
        through three angles corresponding to a 3D rotation
        as defined in :eq:`rotation`.
        This parameter is ignored if ``look_at`` is not `None`.
        Defaults to [0,0,0].

    look_at : [3], float | :class:`~sionna.rt.Transmitter` | :class:`~sionna.rt.Receiver` | :class:`~sionna.rt.RIS` | :class:`~sionna.rt.Camera` | None
        A position or the instance of a :class:`~sionna.rt.Transmitter`,
        :class:`~sionna.rt.Receiver`, :class:`~sionna.rt.RIS`, or :class:`~sionna.rt.Camera` to look at.
        If set to `None`, then ``orientation`` is used to orientate the device.

    color : [3], float
        Defines the RGB (red, green, blue) ``color`` parameter for the device as displayed in the previewer and renderer.
        Each RGB component must have a value within the range :math:`\in [0,1]`.
        Defaults to `[0.160, 0.502, 0.725]`.

    dtype : tf.complex
        Datatype to be used in internal calculations.
        Defaults to `tf.complex64`.
    """

    def __init__(self,
                 name,
                 position,
                 orientation=(0.,0.,0.),
                 look_at=None,
                 power_dbm=44,
                 color=(0.160, 0.502, 0.725),
                 dtype=tf.complex64):

        # Initialize the base class Object
        super().__init__(name=name,
                         position=position,
                         orientation=orientation,
                         look_at=look_at,
                         color=color,
                         dtype=dtype)

        self.power_dbm = power_dbm

    @property
    def power_dbm(self):
        """ tf.float : Get/set transmit power [dBm] """
        return self._power_dbm

    @power_dbm.setter
    def power_dbm(self, value):
        if isinstance(value, tf.Variable):
            if value.dtype != self._rdtype:
                msg = f"`power_dbm` must have dtype={self._rdtype}"
                raise TypeError(msg)
            else:
                self._power_dbm = value
        else:
            self._power_dbm = tf.cast(value, dtype=self._rdtype)

    @property
    def power(self):
        """ tf.float : Get the transmit power [W] """
        return dbm_to_watt(self._power_dbm)
