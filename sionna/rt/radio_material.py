#
# SPDX-FileCopyrightText: Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""
Implements a radio material.
A radio material provides the EM radio properties for a specific material.
"""

import tensorflow as tf

from . import scene
from sionna.constants import DIELECTRIC_PERMITTIVITY_VACUUM, PI


class RadioMaterial:
    # pylint: disable=line-too-long
    r"""
    RadioMaterial(name, relative_permittivity=1.0, conductivity=0.0, frequency_update_callback=None, trainable=False)

    Class implementing a radio material

    A radio material is defined by its relative permittivity
    :math:`\varepsilon_r` and conductivity :math:`\sigma` (see :eq:`eta`).

    We assume non-ionized and non-magnetic materials, and therefore the
    permeability :math:`\mu` of the material is assumed to be equal
    to the permeability of vacuum i.e., :math:`\mu_r=1.0`.

    For frequency-dependent materials, it is possible to
    specify a callback function ``frequency_update_callback`` that computes
    the material properties :math:`(\varepsilon_r, \sigma)` from the
    frequency. If a callback function is specified, the material properties
    cannot be set and the values specified at instantiation are ignored.
    The callback should return `-1` for both the relative permittivity and
    the conductivity if these are not defined for the given carrier frequency.

    The material properties are TensorFlow variables that can be made
    trainable.

    Parameters
    -----------
    name : str
        Unique name of the material

    relative_permittivity : float | `None`
        The relative permittivity of the material.
        Must be larger or equal to 1.
        Defaults to 1. Ignored if ``frequency_update_callback``
        is provided.

    conductivity : float | `None`
        Conductivity of the material [S/m].
        Must be non-negative.
        Defaults to 0.
        Ignored if ``frequency_update_callback``
        is provided.

    frequency_update_callback : callable | `None`
        An optional callable object used to obtain the material parameters
        from the scene's :attr:`~sionna.rt.Scene.frequency`.
        This callable must take as input the frequency [Hz] and
        must return the material properties as a tuple:

        ``(relative_permittivity, conductivity)``.

        If set to `None`, the material properties are constant and equal
        to ``relative_permittivity`` and ``conductivity``.
        Defaults to `None`.

    trainable : bool
        Determines if the material properties are trainable.
        Radio materials with a ``frequency_update_callback``
        function cannot be made trainable.
        Defaults to `False`.

    dtype : tf.complex64 or tf.complex128
        Datatype.
        Defaults to `tf.complex64`.
    """

    def __init__(self,
                 name,
                 relative_permittivity=1.0,
                 conductivity=0.0,
                 frequency_update_callback=None,
                 trainable=False,
                 dtype=tf.complex64):

        if not isinstance(name, str):
            raise TypeError("`name` must be a string")
        self._name = name

        if dtype not in (tf.complex64, tf.complex128):
            msg = "`dtype` must be `tf.complex64` or `tf.complex128`"
            raise ValueError(msg)
        self._dtype = dtype
        self._rdtype = dtype.real_dtype

        self._relative_permittivity = tf.Variable(tf.ones((), self._rdtype))
        self._conductivity = tf.Variable(tf.zeros((), self._rdtype))

        if frequency_update_callback is None:
            self.relative_permittivity = relative_permittivity
            self.conductivity = conductivity

        # Save the callback for when the frequency is updated
        self._frequency_update_callback = frequency_update_callback

        # Configure trainability if possible
        self.trainable = trainable

        # Run frequency_update_callback to set the properties
        self.frequency_update(scene.Scene().frequency)

        # To keep track of the use of this material, we use a reference counter
        # which counts how many objects in the scene use this material
        self._use_counter = 0

        # When loading a scene, the custom materials (i.e., the materials not
        # baked-in Sionna but defined by the user) are not defined yet.
        # If when loading a scene a non-defined material is encountered,
        # then a "placeholder" material is created which is used until the
        # material is defined by the user.
        # Note that propagation simulation cannot be done if placeholders are
        # used.
        self._is_placeholder = False # Is this material a placeholder

    @property
    def name(self):
        """
        str (read-only) : Name of the radio material
        """
        return self._name

    @property
    def relative_permittivity(self):
        r"""
        tf.float : Get/set the relative permittivity
            :math:`\varepsilon_r` :eq:`eta`
        """
        return self._relative_permittivity.value()

    @relative_permittivity.setter
    def relative_permittivity(self, v):
        v = tf.cast(v, self._rdtype)
        self._relative_permittivity.assign(v)

    @property
    def relative_permeability(self):
        r"""
        tf.float (read-only) : Relative permeability
            :math:`\mu_r` :eq:`mu`.
            Defaults to 1.
        """
        return tf.cast(1., self._rdtype)

    @property
    def conductivity(self):
        r"""
        tf.float: Get/set the conductivity
            :math:`\sigma` [S/m] :eq:`eta`
        """
        return self._conductivity.value()

    @conductivity.setter
    def conductivity(self, v):
        v = tf.cast(v, self._rdtype)
        self._conductivity.assign(v)

    @property
    def complex_relative_permittivity(self):
        r"""
        tf.complex (read-only) : Complex relative permittivity
            :math:`\eta` :eq:`eta`
        """
        epsilon_0 = DIELECTRIC_PERMITTIVITY_VACUUM
        eta_prime = self.relative_permittivity
        sigma = self.conductivity
        frequency = scene.Scene().frequency
        omega = tf.cast(2.*PI*frequency, self._rdtype)
        return tf.complex(eta_prime,
                          -tf.math.divide_no_nan(sigma, epsilon_0*omega))

    @property
    def trainable(self):
        """
        bool : Get/set if the conductivity
            and relative permittivity are trainable variables
            or not.
            Defaults to `False`.
        """
        return self._trainable

    @trainable.setter
    def trainable(self, value):
        if not isinstance(value, bool):
            raise TypeError("`trainable` must be bool")
        if value and self._frequency_update_callback is not None:
            err_msg = "Radio materials with frequency_update_callback" + \
                      " cannot be made trainable."
            raise ValueError(err_msg)
        # pylint: disable=protected-access
        self._relative_permittivity._trainable = value
        self._conductivity._trainable = value
        self._trainable = value

    @property
    def frequency_update_callback(self):
        """
        callable : Get/set frequency update callback function
        """
        return self._frequency_update_callback

    @frequency_update_callback.setter
    def frequency_update_callback(self, value):
        self._frequency_update_callback = value
        self.frequency_update(scene.Scene().frequency)

    @property
    def well_defined(self):
        """bool : Get if the material is well-defined"""
        return ( (self._conductivity >= 0.)
             and (self._relative_permittivity >= 1.) )

    @property
    def is_used(self):
        """bool : Get if the material is used by at least one object of
        the scene"""
        return self.use_counter > 0

    @property
    def use_counter(self):
        """
        int : Number of scene objects using this material
        """
        return self._use_counter

    ##############################################
    # Internal methods.
    # Should not be documented.
    ##############################################

    def frequency_update(self, fc):
        # pylint: disable=line-too-long
        r"""
        frequency_update(fc)

        Callback for when the frequency is updated

        Input
        ------
        fc : float
            The new value for the frequency [Hz]
        """
        if self._frequency_update_callback is None:
            return

        parameters = self._frequency_update_callback(fc)
        relative_permittivity, conductivity = parameters
        self.relative_permittivity = relative_permittivity
        self.conductivity = conductivity

    def increase_use(self):
        """
        Must be called when this material is used by a scene object
        """
        self._use_counter += 1

    def decrease_use(self):
        """
        Must be called when a scene object stops using this material
        """
        self._use_counter -= 1
        assert self._use_counter >= 0,\
            f"Reference counter of material '{self.name}' is negative!"

    @property
    def is_placeholder(self):
        return self._is_placeholder

    @is_placeholder.setter
    def is_placeholder(self, v):
        self._is_placeholder = v
