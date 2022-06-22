# $Id: edfa.py 0 13.06.2022 13:53$
# Author: Tim Alexander Uhlemann <uhlemann@ieee.org>
# Copyright:

"""
This module defines the following classes:

- `EDFA`, Optical amplifier

Exception classes:

Functions:


How To Use This Module
======================
(See the individual classes, methods, and attributes for details.)
"""

__docformat__ = 'restructuredtext'

# Standard library imports

# Third party imports
import tensorflow as tf
from tensorflow.keras.layers import Layer

# Local application imports
import sionna
from sionna.channel.optical import utils


class EDFA(Layer):
    r"""EDFA(G, F, f_c, dt, dtype=tf.complex64, **kwargs)

    Erbium-Doped Fiber Amplifier

    Amplifies the input optical signal ``x`` by factor ``sqrt(G)`` and adds
    amplified spontaneous emission (ASE) noise.

    This class inherits from the Keras `Layer` class and can be used as layer in
    a Keras model.

    Example
    --------

    Setting-up:

    >>> edfa = EDFA(4.0, 2.0, 193.55e12, 1.0)

    Running:

    >>> # x is the optical input signal
    >>> y = EDFA(([1.0+1.0j, 1.0+1.0j, 1.0+1.0j]))

    Parameters
    ----------
        g : Tensor, tf.float
            Amplifier gain in (1)

        n_sp : Tensor, tf.float
            Spontaneous emission factor in (1)

        f_c : Tensor, tf.float
            Carrier frequency in (Hz)

        dt : Tensor, tf.float
            Time step in (s)

        dtype : Complex tf.DType
            Defines the datatype for internal calculations and the output
            dtype. Defaults to `tf.complex64`.

    Input
    -----
        (x) : Tuple:

        x : Tensor, tf.complex
            Optical input signal

    Output
    -------
        y : Tensor with same shape as ``x``, tf.complex
            Amplifier output
    """
    def __init__(self, g, f, f_c, dt, dtype=tf.complex64, **kwargs):
        super().__init__(dtype=dtype, **kwargs)

        self._complex_dtype = dtype  # Complex datatype
        self._real_dtype = dtype.real_dtype  # Complex datatype

        self._g = tf.cast(g, self._real_dtype)  # Gain in (1)
        self._f = tf.cast(f, self._real_dtype)  # Gain in (1)
        self._f_c = tf.cast(f_c, self._real_dtype)  # Carrier frequency in (Hz)
        self._dt = tf.cast(dt, self._real_dtype)  # Sampling duration in (s)
        # in (Ws^2)

        # Spontaneous emission factor in (1)
        if self._g == 1.0:
            self._n_sp = tf.cast(0.0, self._real_dtype)
        else:
            self._n_sp = self._f / tf.cast(
                2.0, self._real_dtype) * self._g / (
                                 self._g - tf.cast(1.0, self._real_dtype))

        self._rho_n_ase = tf.cast(
            self._n_sp * (self._g - tf.cast(1.0, self._real_dtype)) *
            sionna.constants.H * self._f_c,
            self._real_dtype)  # Noise density in (W/Hz)
        self._p_n_ase = tf.cast(
            2.0, self._real_dtype) * self._rho_n_ase * tf.cast(
            1.0, self._real_dtype) / (self._dt)  # Noise power in (W)

    def call(self, inputs, **kwargs):
        x = tf.cast(inputs, self._complex_dtype)

        # Calculate noise signal with given noise power
        n = tf.complex(
            tf.random.normal(
                tf.shape(x),
                tf.cast(0.0, self._real_dtype),
                tf.sqrt(self._p_n_ase / tf.cast(2.0, self._real_dtype)),
                self._real_dtype),
            tf.random.normal(
                tf.shape(x),
                tf.cast(0.0, self._real_dtype),
                tf.sqrt(self._p_n_ase / tf.cast(2.0, self._real_dtype)),
                self._real_dtype)
        )

        # Amplify signal
        x = x * tf.cast(tf.sqrt(self._g), self._complex_dtype)

        # Add noise signal
        y = x + n

        return y
