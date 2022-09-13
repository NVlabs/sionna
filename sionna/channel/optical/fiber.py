#
# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Author: Tim Alexander Uhlemann <uhlemann@ieee.org>

"""
This module defines the split-step Fourier method to approximate the solution of
the nonlinear Schroedinger equation.
"""

import tensorflow as tf
from tensorflow.keras.layers import Layer
import sionna
from sionna.channel import utils


class SSFM(Layer):
    # pylint: disable=line-too-long
    r"""SSFM(alpha, beta_2, f_c, gamma, half_window_length, length, n_ssfm, n_sp=1.0,  dt=1.0, t_norm=1e-12, with_amplification=False, with_attenuation=True, with_dispersion=True, with_nonlinearity=True, swap_memory=True, dtype=tf.complex64, **kwargs)

    Layer implementing the split-step Fourier method (SSFM).

    The SSFM (first mentioned in [SSFM]_) numerically solves the nonlinear
    Schrödinger equation (NLSE)

    .. math::

        \frac{\partial E(t,z)}{\partial z}= -\frac{\alpha}{2}E(t,z) + j\frac{\beta_2}{2}\frac{\partial^2 E(t,z)}{\partial t^2}-j\gamma |E(t,z)|^2 E(t,z) + n(n_{\text{sp}};\,t,\,z)

    with attenuation coefficient :math:`\alpha`, group velocity dispersion
    parameters :math:`\beta_2`, and nonlinearity coefficient :math:`\gamma`.
    Further, :math:`n(n_{\text{sp}};\,t,\,z)` denotes the noise due to an
    optional ideally distributed Raman amplification with spontaneous emission
    factor :math:`n_\text{sp}`.
    The optical signal :math:`E(t,\,z)` has the unit :math:`\sqrt{\text{W}}`.

    By now, the SSFM is implemented for a single polarization only and as a
    symmetrized SSFM according to Eq. (7) of [SymSSFM]_ which can be written as

    .. math::

        E(z+\Delta_z,t) \approx \exp\left(\frac{\Delta_z}{2}\hat{D}\right)\exp\left(\int^{z+\Delta_z}_z \hat{N}(z')dz'\right)\exp\left(\frac{\Delta_z}{2}\hat{D}\right)E(z,\,t)

    The integral is approximated by :math:`\Delta_z\hat{N}` with
    :math:`\Delta_z` denoting the (small) simulation step size. Further,
    :math:`\hat{D}` and :math:`\hat{N}` denote the linear and nonlinear SSFM
    operator, respectively [A2012]_.

    Additionally, ideally distributed Raman amplification can be applied, which
    is implemented as in [RamanASE]_. Note that, currently, the implemented
    Raman amplification will always result in a transparent fiber link. Hence,
    the introduced gain cannot be parametrized.

    The SSFM operates on normalized time :math:`T_\text{norm}`
    (e.g., :math:`T_\text{norm}=1\,\text{ps}=1\cdot 10^{-12}\,\text{s}`) and
    distance units :math:`L_\text{norm}`
    (e.g., :math:`L_\text{norm}=1\,\text{km}=1\cdot 10^{3}\,\text{m}`).
    Hence, all parameters as well as the signal itself have to be given with the
    same unit prefix for the
    same unit (e.g., always pico for time, or kilo for distance). Note
    that, despite the normalization, the SSFM is implemented with physical units
    different from the normalization, e.g., used for the nonlinear
    Fourier transform. For simulations only :math:`T_\text{norm}` has to be
    provided.

    To avoid reflections at the signal boundaries during simulation, a Hamming
    window can be applied in each SSFM-step where the length can be defined by
    ``half_window_length``.

    This class inherits from the Keras `Layer` class and can be used as layer in
    a Keras model.

    Example
    --------

    Setting-up:

    >>> ssfm = SSFM(alpha=0.046,
    >>>             beta_2=-21.67,
    >>>             f_c=193.55e12,
    >>>             gamma=1.27,
    >>>             half_window_length=100,
    >>>             length=80,
    >>>             n_ssfm=200,
    >>>             n_sp=1.0,
    >>>             t_norm=1e-12,
    >>>             with_amplification=False,
    >>>             with_attenuation=True,
    >>>             with_dispersion=True,
    >>>             with_nonlinearity=True)

    Running:

    >>> # x is the optical input signal
    >>> y = ssfm(x)

    Parameters
    ----------
        alpha : float
            Attenuation coefficient :math:`\alpha` in :math:`(1/L_\text{norm})`.

        beta_2 : float
            Group velocity dispersion coefficient :math:`\beta_2` in :math:`(T_\text{norm}^2/L_\text{norm})`.

        f_c : float
            Carrier frequency :math:`f_\mathrm{c}` in :math:`(\text{Hz})`.

        gamma : float
            Nonlinearity coefficient :math:`\gamma` in :math:`(1/L_\text{norm}/\text{W})`.

        half_window_length : int
            Half of the Hamming window length.

        length : float
            Fiber length :math:`\ell` in :math:`(L_\text{norm})`.

        n_ssfm : int
            Number of steps :math:`N_\mathrm{SSFM}`.

        n_sp : float
            Spontaneous emission factor :math:`n_\mathrm{sp}` of Raman amplification.

        sample_duration : float
            Normalized time step :math:`\Delta_t` in :math:`(T_\text{norm})`.

        t_norm : float
            Time normalization :math:`T_\text{norm}` in :math:`(\text{s})`.

        with_attenuation : bool
            Enables application of attenuation. Defaults to True.

        with_amplification : bool
            Enables application of ideal inline amplification and corresponding
            noise. Defaults to False.

        with_dispersion : bool
            Enables application of chromatic dispersion. Defaults to True.

        with_nonlinearity : bool
            Enables application of Kerr nonlinearity. Defaults to True.

        swap_memory : bool
            Use CPU memory for while loop. Defaults to True.

        dtype : Complex tf.DType
            Defines the datatype for internal calculations and the output
            dtype. Defaults to `tf.complex64`.

    Input
    -----
        x : Tensor, tf.complex
            Input signal in :math:`(\sqrt{\text{W}})`

    Output
    ------
        y : Tensor with same shape as ``x``, `tf.complex`
            Channel output
    """
    def __init__(self, alpha=0.046, beta_2=-21.67, f_c=193.55e12,
                 gamma=1.27, half_window_length=0, length=80,
                 n_ssfm=1, n_sp=1.0, sample_duration=1.0, t_norm=1e-12,
                 with_amplification=False, with_attenuation=True,
                 with_dispersion=True, with_nonlinearity=True,
                 swap_memory=True, dtype=tf.complex64, **kwargs):
        super().__init__(dtype=dtype, **kwargs)

        self._dtype = dtype
        self._cdtype = tf.as_dtype(dtype)
        self._rdtype = tf.as_dtype(dtype).real_dtype

        self._alpha = tf.cast(alpha, dtype=self._rdtype)
        self._beta_2 = tf.cast(beta_2, dtype=self._rdtype)
        self._f_c = tf.cast(f_c, dtype=self._rdtype)
        self._gamma = tf.cast(gamma, dtype=self._rdtype)
        self._half_window_length = half_window_length
        self._length = tf.cast(length, dtype=self._rdtype)
        self._n_ssfm = tf.cast(n_ssfm, dtype=tf.int32)
        self._n_sp = tf.cast(n_sp, dtype=self._rdtype)
        self._swap_memory = swap_memory
        self._t_norm = tf.cast(t_norm, dtype=self._rdtype)
        self._with_amplification = tf.cast(with_amplification, dtype=tf.bool)
        self._with_attenuation = tf.cast(with_attenuation, dtype=tf.bool)
        self._with_dispersion = tf.cast(with_dispersion, dtype=tf.bool)
        self._with_nonlinearity = tf.cast(with_nonlinearity, dtype=tf.bool)
        self._sample_duration = tf.cast(sample_duration, dtype=self._rdtype)

        self._rho_n = \
            sionna.constants.H * self._f_c * self._alpha * self._length * \
            self._n_sp  # in (W/Hz)

        # Calculate noise power depending on simulation bandwidth
        self._p_n_ase = self._rho_n / self._sample_duration / self._t_norm
        # in (Ws)

        # Precalculate uniform step size
        self._dz = self._length / tf.cast(self._n_ssfm, dtype=self._rdtype)

        self._window = tf.complex(
            tf.signal.hamming_window(
                window_length=2*self._half_window_length,
                dtype=self._rdtype
            ),
            tf.zeros(
                2*self._half_window_length,
                dtype=self._rdtype
            )
        )

    def _apply_linear_operator(self, q, dz, zeros, frequency_vector):
        # Chromatic dispersion
        if self._with_dispersion:
            dispersion = tf.exp(
                tf.complex(
                    zeros,
                    -self._beta_2 / tf.cast(2.0, self._rdtype) * dz *
                    (
                            tf.cast(2.0, self._rdtype) *
                            tf.cast(sionna.constants.PI, self._rdtype) *
                            frequency_vector
                    ) ** tf.cast(2.0, self._rdtype)
                )
            )
            dispersion = tf.signal.fftshift(dispersion, axes=-1)
            q = tf.signal.ifft(tf.signal.fft(q) * dispersion)

        # Attenuation
        if self._with_attenuation:
            q = q * tf.cast(tf.exp(-self._alpha / 2.0 * dz), self._cdtype)

        # Amplification (Raman)
        if self._with_amplification:
            q = q * tf.cast(tf.exp(self._alpha / 2.0 * dz), self._cdtype)

        return q

    def _apply_noise(self, q, dz):
        # Noise due to Amplification (Raman)
        if self._with_amplification:
            q_n = tf.complex(
                tf.random.normal(
                    q.shape,
                    tf.cast(0.0, self._rdtype),
                    tf.sqrt(
                        self._p_n_ase *
                        tf.cast(dz, self._rdtype) /
                        tf.cast(self._length, self._rdtype) /
                        tf.cast(2.0, self._rdtype)
                    ),
                    self._rdtype),
                tf.random.normal(
                    q.shape,
                    tf.cast(0.0, self._rdtype),
                    tf.sqrt(
                        self._p_n_ase /
                        tf.cast(dz, self._rdtype) /
                        tf.cast(self._length, self._rdtype) /
                        tf.cast(2.0, self._rdtype)
                    ),
                    self._rdtype
                )
            )
            q = q + q_n

        return q

    def _apply_nonlinear_operator(self, q, dz, zeros):
        if self._with_nonlinearity:
            q = q * tf.exp(
                tf.complex(
                    zeros,
                    tf.abs(q) ** tf.cast(2.0, self._rdtype) * self._gamma *
                    tf.negative(tf.math.real(dz)))
            )

        return q

    def _apply_window(self, q, window):
        return q * window

    def _step(self, q, precalculations, n_steps, step_counter):
        (window, dz, zeros, f) = precalculations

        # Apply window-function
        q = self._apply_window(q, window)
        q = self._apply_nonlinear_operator(q, dz, zeros)  # N
        q = self._apply_noise(q, dz)
        q = self._apply_linear_operator(q, dz, zeros, f)  # D

        step_counter = step_counter + 1

        return q, precalculations, n_steps, step_counter

    def _cond(self, q, precalculations, n_steps, step_counter):
        # pylint: disable=unused-argument
        return tf.less(step_counter, n_steps)

    def call(self, inputs):
        x = inputs

        # Calculate support parameters
        input_shape = x.shape

        # Generate frequency vectors
        _, f = utils.time_frequency_vector(
            input_shape[-1], self._sample_duration, dtype=self._rdtype)

        # Window function calculation (depends on length of the signal)
        window = tf.concat(
            [
                self._window[0:self._half_window_length],
                tf.complex(
                    tf.ones(
                        [input_shape[-1] - 2*self._half_window_length],
                        dtype=self._rdtype
                    ),
                    tf.zeros(
                        [input_shape[-1] - 2*self._half_window_length],
                        dtype=self._rdtype
                    )
                ),
                self._window[self._half_window_length::]
            ],
            axis=0
        )

        # All-zero vector
        zeros = tf.zeros(input_shape, dtype=self._rdtype)

        # Spatial step size
        dz = tf.cast(self._dz, dtype=self._rdtype)

        dz_half = dz/tf.cast(2.0, self._rdtype)

        # SSFM step counter
        iterator = tf.constant(0, dtype=tf.int32, name="step_counter")

        # Symmetric SSFM
        # Start with half linear propagation
        x = self._apply_linear_operator(x, dz_half, zeros, f)
        # Proceed with N_SSFM-1 steps applying nonlinear and linear operator
        x, _, _, _ = tf.while_loop(
            self._cond,
            self._step,
            (x, (window, dz, zeros, f), self._n_ssfm-1, iterator),
            swap_memory=self._swap_memory,
            parallel_iterations=1
        )
        # Final nonlinear operator
        x = self._apply_nonlinear_operator(x, dz, zeros)
        # Final noise application
        x = self._apply_noise(x, dz)
        # End with half linear propagation
        x = self._apply_linear_operator(x, dz_half, zeros, f)

        return x
