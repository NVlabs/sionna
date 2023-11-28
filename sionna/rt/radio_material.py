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
from .scattering_pattern import ScatteringPattern, LambertianPattern

class RadioMaterial:
    # pylint: disable=line-too-long
    r"""
    Class implementing a radio material

    A radio material is defined by its relative permittivity
    :math:`\varepsilon_r` and conductivity :math:`\sigma` (see :eq:`eta`),
    as well as optional parameters related to diffuse scattering, such as the
    scattering coefficient :math:`S`, cross-polarization discrimination
    coefficient :math:`K_x`, and scattering pattern :math:`f_\text{s}(\hat{\mathbf{k}}_\text{i}, \hat{\mathbf{k}}_\text{s})`.

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

    The material properties can be assigned to a TensorFlow variable or
    tensor. In the latter case, the tensor could be the output of a callable,
    such as a Keras layer implementing a neural network. In the former case, it
    could be set to a trainable variable:

    .. code-block:: Python

        mat = RadioMaterial("my_mat")
        mat.conductivity = tf.Variable(0.0, dtype=tf.float32)

    Parameters
    -----------
    name : str
        Unique name of the material

    relative_permittivity : float | `None`
        Relative permittivity of the material.
        Must be larger or equal to 1.
        Defaults to 1. Ignored if ``frequency_update_callback``
        is provided.

    conductivity : float | `None`
        Conductivity of the material [S/m].
        Must be non-negative.
        Defaults to 0.
        Ignored if ``frequency_update_callback``
        is provided.

    scattering_coefficient : float
        Scattering coefficient :math:`S\in[0,1]` as defined in
        :eq:`scattering_coefficient`.
        Defaults to 0.

    xpd_coefficient : float
        Cross-polarization discrimination coefficient :math:`K_x\in[0,1]` as
        defined in :eq:`xpd`.
        Only relevant if ``scattering_coefficient``>0.
        Defaults to 0.

    scattering_pattern : ScatteringPattern
        :class:`~sionna.rt.ScatteringPattern` to be applied.
        Only relevant if ``scattering_coefficient``>0.
        Defaults to `None`, which implies a :class:`~sionna.rt.LambertianPattern`.

    frequency_update_callback : callable | `None`
        An optional callable object used to obtain the material parameters
        from the scene's :attr:`~sionna.rt.Scene.frequency`.
        This callable must take as input the frequency [Hz] and
        must return the material properties as a tuple:

        ``(relative_permittivity, conductivity)``.

        If set to `None`, the material properties are constant and equal
        to ``relative_permittivity`` and ``conductivity``.
        Defaults to `None`.

    dtype : tf.complex64 or tf.complex128
        Datatype.
        Defaults to `tf.complex64`.
    """

    def __init__(self,
                 name,
                 relative_permittivity=1.0,
                 conductivity=0.0,
                 scattering_coefficient=0.0,
                 xpd_coefficient=0.0,
                 scattering_pattern=None,
                 frequency_update_callback=None,
                 dtype=tf.complex64):

        if not isinstance(name, str):
            raise TypeError("`name` must be a string")
        self._name = name

        if dtype not in (tf.complex64, tf.complex128):
            msg = "`dtype` must be `tf.complex64` or `tf.complex128`"
            raise ValueError(msg)
        self._dtype = dtype
        self._rdtype = dtype.real_dtype

        if scattering_pattern is None:
            scattering_pattern = LambertianPattern(dtype=dtype)

        self.scattering_pattern = scattering_pattern
        self.scattering_coefficient = scattering_coefficient
        self.xpd_coefficient = xpd_coefficient

        if frequency_update_callback is None:
            self.relative_permittivity = relative_permittivity
            self.conductivity = conductivity

        # Save the callback for when the frequency is updated
        # or if the RadioMaterial is added to a scene
        self._frequency_update_callback = frequency_update_callback

        # When loading a scene, the custom materials (i.e., the materials not
        # baked-in Sionna but defined by the user) are not defined yet.
        # If when loading a scene a non-defined material is encountered,
        # then a "placeholder" material is created which is used until the
        # material is defined by the user.
        # Note that propagation simulation cannot be done if placeholders are
        # used.
        self._is_placeholder = False # Is this material a placeholder

        # Set of objects identifiers that use this material
        self._objects_using = set()


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
        return self._relative_permittivity

    @relative_permittivity.setter
    def relative_permittivity(self, v):
        if isinstance(v, tf.Variable):
            if v.dtype != self._rdtype:
                msg = f"`relative_permittivity` must have dtype={self._rdtype}"
                raise TypeError(msg)
            else:
                self._relative_permittivity = v
        else:
            self._relative_permittivity = tf.cast(v, self._rdtype)

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
        return self._conductivity

    @conductivity.setter
    def conductivity(self, v):
        if isinstance(v, tf.Variable):
            if v.dtype != self._rdtype:
                msg = f"`conductivity` must have dtype={self._rdtype}"
                raise TypeError(msg)
            else:
                self._conductivity = v
        else:
            self._conductivity = tf.cast(v, self._rdtype)

    @property
    def scattering_coefficient(self):
        r"""
        tf.float: Get/set the scattering coefficient
            :math:`S\in[0,1]` :eq:`scattering_coefficient`.
        """
        return self._scattering_coefficient

    @scattering_coefficient.setter
    def scattering_coefficient(self, v):
        if isinstance(v, tf.Variable):
            if v.dtype != self._rdtype:
                msg=f"`scattering_coefficient` must have dtype={self._rdtype}"
                raise TypeError(msg)
            else:
                self._scattering_coefficient = v
        else:
            self._scattering_coefficient = tf.cast(v, self._rdtype)

    @property
    def xpd_coefficient(self):
        r"""
        tf.float: Get/set the cross-polarization discrimination coefficient
            :math:`K_x\in[0,1]` :eq:`xpd`.
        """
        return self._xpd_coefficient

    @xpd_coefficient.setter
    def xpd_coefficient(self, v):
        if isinstance(v, tf.Variable):
            if v.dtype != self._rdtype:
                msg=f"`xpd_coefficient` must have dtype={self._rdtype}"
                raise TypeError(msg)
            else:
                self._xpd_coefficient = v
        else:
            self._xpd_coefficient = tf.cast(v, self._rdtype)

    @property
    def scattering_pattern(self):
        r"""
        ScatteringPattern: Get/set the ScatteringPattern.
        """
        return self._scattering_pattern

    @scattering_pattern.setter
    def scattering_pattern(self, v):
        if not isinstance(v, ScatteringPattern) and v is not None:
            raise ValueError("Not a valid instanc of ScatteringPattern")
        self._scattering_pattern = v

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
    def frequency_update_callback(self):
        """
        callable : Get/set frequency update callback function
        """
        return self._frequency_update_callback

    @frequency_update_callback.setter
    def frequency_update_callback(self, value):
        self._frequency_update_callback = value
        self.frequency_update()

    @property
    def well_defined(self):
        """bool : Get if the material is well-defined"""
        # pylint: disable=chained-comparison
        return ((self._conductivity >= 0.)
             and (self.relative_permittivity >= 1.)
             and (0. <= self.scattering_coefficient <= 1.)
             and (0. <= self.xpd_coefficient <= 1.)
             and (0. <= self.scattering_pattern.lambda_ <= 1.))

    @property
    def use_counter(self):
        """
        int : Number of scene objects using this material
        """
        return len(self._objects_using)

    @property
    def is_used(self):
        """bool : Indicator if the material is used by at least one object of
        the scene"""
        return self.use_counter > 0

    @property
    def using_objects(self):
        """
        [num_using_objects], tf.int : Identifiers of the objects using this
        material
        """
        tf_objects_using = tf.cast(tuple(self._objects_using), tf.int32)
        return tf_objects_using

    ##############################################
    # Internal methods.
    # Should not be documented.
    ##############################################

    def frequency_update(self):
        # pylint: disable=line-too-long
        r"""Callback for when the frequency is updated
        """
        if self._frequency_update_callback is None:
            return

        parameters = self._frequency_update_callback(scene.Scene().frequency)
        relative_permittivity, conductivity = parameters
        self.relative_permittivity = relative_permittivity
        self.conductivity = conductivity

    def add_object_using(self, object_id):
        """
        Add an object to the set of objects using this material
        """
        self._objects_using.add(object_id)

    def discard_object_using(self, object_id):
        """
        Remove an object from the set of objects using this material
        """
        assert object_id in self._objects_using,\
            f"Object with id {object_id} is not in the set of {self.name}"
        self._objects_using.discard(object_id)

    @property
    def is_placeholder(self):
        """
        bool : Get/set if this radio material is a placeholder
        """
        return self._is_placeholder

    @is_placeholder.setter
    def is_placeholder(self, v):
        self._is_placeholder = v

    def assign(self, rm):
        """
        Assign new values to the radio material properties from another
        radio material ``rm``

        Input
        ------
        rm : :class:`~sionna.rt.RadioMaterial
            Radio material from which to assign the new values
        """
        if not isinstance(rm, RadioMaterial):
            raise TypeError("`rm` is not a RadioMaterial")
        self.relative_permittivity = rm.relative_permittivity
        self.conductivity = rm.conductivity
        self.scattering_coefficient = rm.scattering_coefficient
        self.xpd_coefficient = rm.xpd_coefficient
        self.scattering_pattern = rm.scattering_pattern
        self.frequency_update_callback = rm.frequency_update_callback
