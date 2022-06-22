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


class EDFA(Layer):
    r"""EDFA(g, f, f_c, dt, dtype=tf.complex64, **kwargs)

    **Erbium-Doped Fiber Amplifier**

    Amplifies the input optical signal ``x`` by factor ``sqrt(G)`` and adds
    amplified spontaneous emission (ASE) noise.

    The noise figure including the noise due to beating of signal and
    spontaneous emission only is :math:`F_\mathrm{ASE,shot} =\frac{\mathrm{SNR}_\mathrm{in}}{\mathrm{SNR}_\mathrm{out}}`
    where ideally the detector is limited by shot noise only. Shot noise is
    neglected here but only helps to derive the noise power of the amplifier, as
    otherwise the input SNR was infinitely large. Hence, for the input SNR
    following [A2012]_ that :math:`\mathrm{SNR}_\mathrm{in}=\frac{P}{2hf_0B_\mathrm{w}}`
    The output SNR is decreased by ASE noise induced by the amplification. Note
    that, shot noise is applied after the amplifier and is hence not amplified.
    It results that :math:`\mathrm{SNR}_\mathrm{out}=\frac{GP}{\left(4\rho_\mathrm{ASE}+2hf_0\right)B_\mathrm{w}}`
    Hence, one can write the former equation as :math:`F_\mathrm{ASE,shot} = 2 n_\mathrm{sp} \left(1-G^{-1}\right) + G^{-1}`.
    Dropping shot noise again results in :math:`F = 2 n_\mathrm{sp} \left(1-G^{-1}\right)=2 n_\mathrm{sp} \frac{G-1}{G}`.

    For, e.g., a transparent link, the required gain per span is :math:`G = \exp\left(\alpha \Delta z\right)`.
    The spontaneous emission factor calculates as :math:`n_\mathrm{sp}=\frac{F}{2}\frac{G}{G-1}`.
    According to [A2012]_, [EKWFG2010]_ combined with [BGT2000]_, and [GD1991]_,
    for the noise power spectral density of the EDFA per state of
    polarization one obtains :math:`\rho_\mathrm{ASE}^{(1)} = n_\mathrm{sp}\left(G-1\right) h f_0=\frac{1}{2}G F h f_0`.
    At simulation frequency :math:`f_\mathrm{sim}` the noise has a power of
    :math:`P_\mathrm{ASE}^{(1)}=\sigma_\mathrm{n,ASE}^2=2\rho_\mathrm{ASE}^{(1)}\cdot f_\mathrm{sim}`
    where the factor :math:`2` accounts for the unpolarized noise.
    Here the :math:`()^{(1)}` means that this is the noise introduced by a
    single EDFA.

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
        g : float
            Amplifier gain :math:`G` in :math:`(1)`

        f : float
            Noise figure :math:`F` in :math:`(1)`

        f_c : float
            Carrier frequency :math:`f_\mathrm{c}` in :math:`(Hz)`

        dt : float
            Time step :math:`\Delta_t` in :math:`(s)`

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
