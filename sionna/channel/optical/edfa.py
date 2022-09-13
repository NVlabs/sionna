#
# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Author: Tim Alexander Uhlemann <uhlemann@ieee.org>

"""
This module defines a model for an Erbium-Doped Fiber Amplifier.
"""
import tensorflow as tf
from tensorflow.keras.layers import Layer
import sionna


class EDFA(Layer):
    # pylint: disable=line-too-long
    r"""EDFA(g, f, f_c, dt, dtype=tf.complex64, **kwargs)

    Layer implementing a model of an Erbium-Doped Fiber Amplifier

    Amplifies the optical input signal by a given gain and adds
    amplified spontaneous emission (ASE) noise.

    The noise figure including the noise due to beating of signal and
    spontaneous emission only is :math:`F_\mathrm{ASE,shot} =\frac{\mathrm{SNR}
    _\mathrm{in}}{\mathrm{SNR}_\mathrm{out}}`,
    where ideally the detector is limited by shot noise only, and
    :math:`\text{SNR}` is the signal to noise ratio. Shot noise is
    neglected here but is required to derive the noise power of the amplifier,
    as otherwise the input SNR is infinitely large. Hence, for the input SNR
    it follows [A2012]_ that
    :math:`\mathrm{SNR}_\mathrm{in}=\frac{P}{2hf_cW}` where :math:`h` denotes
    Planck's constant, :math:`P` is the signal power and :math:`W` the
    considered bandwidth.
    The output SNR is decreased by ASE noise induced by the amplification.
    Note that, shot noise is applied after the amplifier and is hence not
    amplified. It results that :math:`\mathrm{SNR}_\mathrm{out}=\frac{GP}{\left
    (4\rho_\mathrm{ASE}+2hf_c\right)W}`, where :math:`G` is the
    parametrized gain.
    Hence, one can write the former equation as :math:`F_\mathrm{ASE,shot} = 2
    n_\mathrm{sp} \left(1-G^{-1}\right) + G^{-1}`.
    Dropping shot noise again results in :math:`F = 2 n_\mathrm{sp} \left(1-G^
    {-1}\right)=2 n_\mathrm{sp} \frac{G-1}{G}`.

    For, e.g., a transparent link, the required gain per span is :math:`G =
    \exp\left(\alpha \ell \right)`.
    The spontaneous emission factor calculates as :math:`n_\mathrm{sp}=\frac{F}
    {2}\frac{G}{G-1}`.
    According to [A2012]_, [EKWFG2010]_ combined with [BGT2000]_, and [GD1991]_,
    for the noise power spectral density of the EDFA per state of
    polarization one obtains :math:`\rho_\mathrm{ASE}^{(1)} = n_\mathrm{sp}\left
    (G-1\right) h f_c=\frac{1}{2}G F h f_c`.
    At simulation frequency :math:`f_\mathrm{sim}` the noise has a power of
    :math:`P_\mathrm{ASE}^{(1)}=\sigma_\mathrm{n,ASE}^2=2\rho_\mathrm{ASE}^{(1)}
    \cdot f_\mathrm{sim}`
    where the factor :math:`2` accounts for the unpolarized noise.
    Here the :math:`()^{(1)}` means that this is the noise introduced by a
    single EDFA.

    This class inherits from the Keras `Layer` class and can be used as layer in
    a Keras model.

    Example
    --------

    Setting-up:

    >>> edfa = EDFA(g=4.0,
    >>>             f=2.0,
    >>>             f_c=193.55e12,
    >>>             dt=1.0e-12)

    Running:

    >>> # x is the optical input signal
    >>> y = EDFA(x)

    Parameters
    ----------
        g : float
            Amplifier gain (linear domain)

        f : float
            Noise figure (linear domain)

        f_c : float
            Carrier frequency :math:`f_\mathrm{c}` in :math:`(\text{Hz})`

        dt : float
            Time step :math:`\Delta_t` in :math:`(\text{s})`

        dtype : Complex tf.DType
            Defines the datatype for internal calculations and the output
            dtype. Defaults to `tf.complex64`.

    Input
    -----
        x : Tensor, tf.complex
            Optical input signal

    Output
    -------
        y : Tensor with same shape as ``x``, ``dtype``
            Amplifier output
    """
    def __init__(self, g, f, f_c, dt, dtype=tf.complex64, **kwargs):
        super().__init__(dtype=dtype, **kwargs)

        self._dtype = dtype
        self._cdtype = tf.as_dtype(dtype)
        self._rdtype = tf.as_dtype(dtype).real_dtype

        self._g = tf.cast(g, self._rdtype)
        self._f = tf.cast(f, self._rdtype)
        self._f_c = tf.cast(f_c, self._rdtype)
        self._dt = tf.cast(dt, self._rdtype)

        # Spontaneous emission factor
        if self._g == 1.0:
            self._n_sp = tf.cast(0.0, self._rdtype)
        else:
            self._n_sp = self._f / tf.cast(
                2.0, self._rdtype) * self._g / (
                                 self._g - tf.cast(1.0, self._rdtype))

        self._rho_n_ase = tf.cast(
            self._n_sp * (self._g - tf.cast(1.0, self._rdtype)) *
            sionna.constants.H * self._f_c,
            self._rdtype)  # Noise density in (W/Hz)
        self._p_n_ase = tf.cast(
            2.0, self._rdtype) * self._rho_n_ase * tf.cast(
            1.0, self._rdtype) / (self._dt)  # Noise power in (W)

    def call(self, inputs):
        x = tf.cast(inputs, self._cdtype)

        # Calculate noise signal with given noise power
        n = tf.complex(
            tf.random.normal(
                tf.shape(x),
                tf.cast(0.0, self._rdtype),
                tf.sqrt(self._p_n_ase / tf.cast(2.0, self._rdtype)),
                self._rdtype),
            tf.random.normal(
                tf.shape(x),
                tf.cast(0.0, self._rdtype),
                tf.sqrt(self._p_n_ase / tf.cast(2.0, self._rdtype)),
                self._rdtype))

        # Amplify signal
        x = x * tf.cast(tf.sqrt(self._g), self._cdtype)

        # Add noise signal
        y = x + n

        return y
