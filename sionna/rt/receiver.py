#
# SPDX-FileCopyrightText: Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""
Class impelenting a receiver
"""

import tensorflow as tf
from .radio_device import RadioDevice

class Receiver(RadioDevice):
    # pylint: disable=line-too-long
    r"""
    Class defining a receiver

    Parameters
    ----------
    name : str
        Name

    position : [3], float
        Position :math:`(x,y,z)` as three-dimensional vector

    orientation : [3], float
        Orientation :math:`(\alpha, \beta, \gamma)` [rad] specified
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

        # Initialize the base class Object
        super().__init__(name=name,
                         position=position,
                         orientation=orientation,
                         look_at=look_at,
                         trainable_position=trainable_position,
                         trainable_orientation=trainable_orientation,
                         dtype=dtype)
