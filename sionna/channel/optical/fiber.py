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
    r"""SSFM(alpha, beta_2, f_c, gamma, half_window_length, ell, n_ssfm, n_sp, dt, t_norm, with_amplification, with_attenuation, with_dispersion, with_nonlinearity, swap_memory=True, dtype=tf.complex64, **kwargs)

    Layer implementing the split-step Fourier method (SSFM).

    The SSFM (first mentioned in [SSFM]_) is implemented as
    symmetrized SSFM according to Eq. (7) of [SymSSFM]_ and can be written as

    .. math::

        E(z+\Delta_z,t) \approx \exp\left(\frac{\Delta_z}{2}\hat{D}\right)\exp\left(\int^{z+\Delta_z}_z \hat{N}(z')dz'\right)\exp\left(\frac{\Delta_z}{2}\hat{D}\right)E(z,\,t)

    The integral is approximated by :math:`\Delta_z\hat{N}`.

    Additionally, ideally distributed Raman amplification may be applied, which
    is implemented as in [RamanASE]_.

    The SSFM operates on normalized time :math:`T_\text{norm}` (e.g., 1 ps) and
    distance units :math:`L_\text{norm}` (e.g., 1 km). Hence, parameters as well
    as the signal itself have to be given in the same normalized units. Note
    that, despite the normalization, the SSFM is implemented with physical units
    different from the normalization, e.g., used for the nonlinear
    Fourier transform.

    This class inherits from the Keras `Layer` class and can be used as layer in
    a Keras model.

    Example
    --------

    Setting-up:

    >>> ssfm_channel = SSFM(
    >>>     0.046, -21.67, 193.55e12, 1.27, 100, 80, 200, 1e-17, 1.0,
    >>>     True, True, True, True)

    Running:

    >>> y = ssfm_channel([1.0+1.0j, 1.0+1.0j, 1.0+1.0j])

    Parameters
    ----------
        alpha : float
            Attenuation coefficient :math:`\alpha` in :math:`(1/L_\text{norm})`

        beta_2 : float
            Chromatic dispersion coefficient :math:`\beta_2` in :math:`(T_\text{norm}^2/L_\text{norm})`

        f_c : float
            Carrier frequency :math:`f_\mathrm{c}` in :math:`(1/T_\text{norm})`

        gamma : float
            Kerr-nonlinearity :math:`\gamma` in :math:`(1/L_\text{norm}/W)`

        half_window_length : int
            Half of the Hamming window length

        ell : float
            Fiber length :math:`\ell` in :math:`(L_\text{norm})`

        n_ssfm : int
            Number of steps :math:`N_\mathrm{SSFM}`

        n_sp : float
            Spontaneous emission factor :math:`n_\mathrm{sp}` of Raman amplification

        dt : float
            Normalized time step :math:`\Delta_t` in :math:`(T_\text{norm})`

        t_norm : float
            Time normalization :math:`T_\text{norm}` in :math:`(s)`.

        with_attenuation : bool
            Enables application of attenuation

        with_amplification : bool
            Enables application of ideal inline amplification and corresponding
            noise

        with_dispersion : bool
            Enables application of chromatic dispersion

        with_nonlinearity : bool
            Enables application of Kerr nonlinearity

        swap_memory : bool
            Use CPU memory for while loop. Defaults to True.

        dtype : Complex tf.DType
            Defines the datatype for internal calculations and the output
            dtype. Defaults to `tf.complex64`.

    Input
    -----
        x : Tensor, tf.complex
            Input signal in :math:`(W^{\frac12})`

    Output
    -------
        y : Tensor with same shape as ``x``, tf.complex
            Channel output
    """
    def __init__(self, alpha, beta_2, f_c, gamma, half_window_length, ell,
                 n_ssfm, n_sp, dt, t_norm,
                 with_amplification, with_attenuation, with_dispersion,
                 with_nonlinearity,
                 swap_memory=True, dtype=tf.complex64, **kwargs):
        super().__init__(dtype=dtype, **kwargs)

        self._alpha = tf.cast(alpha, dtype=dtype.real_dtype)
        self._beta_2 = tf.cast(beta_2, dtype=dtype.real_dtype)
        self._f_c = tf.cast(f_c, dtype=dtype.real_dtype)
        self._gamma = tf.cast(gamma, dtype=dtype.real_dtype)
        self._half_window_length = half_window_length
        self._ell = tf.cast(ell, dtype=dtype.real_dtype)
        self._n_ssfm = tf.cast(n_ssfm, dtype=tf.int32)
        self._n_sp = tf.cast(n_sp, dtype=dtype.real_dtype)
        self._swap_memory = swap_memory
        self._t_norm = tf.cast(t_norm, dtype=dtype.real_dtype)
        self._with_amplification = tf.cast(with_amplification, dtype=tf.bool)
        self._with_attenuation = tf.cast(with_attenuation, dtype=tf.bool)
        self._with_dispersion = tf.cast(with_dispersion, dtype=tf.bool)
        self._with_nonlinearity = tf.cast(with_nonlinearity, dtype=tf.bool)
        self._cdtype = dtype
        self._rdtype = dtype.real_dtype
        self._dt = tf.cast(dt, dtype=dtype.real_dtype)

        self._rho_n = \
            sionna.constants.H / (self._t_norm ** 2.0) * 2.0 * \
            sionna.constants.PI * self._f_c * self._alpha * self._ell * \
            self._n_sp

        # Calculate noise power depending on simulation bandwidth
        self._p_n_ase = self._rho_n / self._dt

        # Precalculate dispersion
        self._dz = self._ell / tf.cast(self._n_ssfm, dtype=self._rdtype)

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

    def apply_linear_operator(self, q, dz, zeros, f):
        # Chromatic dispersion
        if self._with_dispersion:
            dispersion = tf.exp(
                tf.complex(
                    zeros,
                    -self._beta_2 / tf.cast(2.0, self._rdtype) * dz *
                    (
                            tf.cast(2.0, self._rdtype) *
                            tf.cast(sionna.constants.PI, self._rdtype) * f
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

    def apply_noise(self, q, dz):
        # Noise due to Amplification (Raman)
        if self._with_amplification:
            q_n = tf.complex(
                tf.random.normal(
                    q.shape,
                    tf.cast(0.0, self._rdtype),
                    tf.sqrt(
                        self._p_n_ase /
                        tf.cast(self._n_ssfm, self._rdtype) /
                        tf.cast(2.0, self._rdtype)
                    ),
                    self._rdtype),
                tf.random.normal(
                    q.shape,
                    tf.cast(0.0, self._rdtype),
                    tf.sqrt(
                        self._p_n_ase /
                        tf.cast(self._n_ssfm, self._rdtype) /
                        tf.cast(2.0, self._rdtype)
                    ),
                    self._rdtype
                )
            )
            q = q + q_n

        return q

    def apply_nonlinear_operator(self, q, dz, zeros):
        if self._with_nonlinearity:
            nonlinearity = tf.complex(
                zeros,
                tf.abs(q) ** tf.cast(2.0, self._rdtype) * self._gamma *
                tf.negative(tf.math.real(dz)))
            q = q * tf.exp(nonlinearity)

        return q

    def apply_window(self, q, window):
        return q * window

    def _step(self, q, precalculations, n_steps, step_counter):
        (window, dz, zeros, f) = precalculations

        # Apply window-function
        q = self.apply_window(q, window)
        q = self.apply_nonlinear_operator(q, dz, zeros)  # N
        q = self.apply_noise(q, dz)
        q = self.apply_linear_operator(q, dz, zeros, f)  # D

        step_counter = step_counter + 1

        return q, precalculations, n_steps, step_counter

    def _cond(self, q, precalculations, n_steps, step_counter):
        return tf.less(step_counter, n_steps)

    def call(self, inputs):
        x = inputs

        # Calculate support parameters
        input_shape = x.shape

        # Generate frequency vectors
        _, f = utils.time_frequency_vector(
            input_shape[-1], self._dt, dtype=self._rdtype)

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
        x = self.apply_linear_operator(x, dz_half, zeros, f)
        # Proceed with N_SSFM-1 steps applying nonlinear and linear operator
        x, _, _, _ = tf.while_loop(
            self._cond,
            self._step,
            (x, (window, dz, zeros, f), self._n_ssfm-1, iterator),
            swap_memory=self._swap_memory,
            parallel_iterations=1
        )
        # Final nonlinear operator
        x = self.apply_nonlinear_operator(x, dz, zeros)
        # Final noise application
        x = self.apply_noise(x, dz)
        # End with half linear propagation
        x = self.apply_linear_operator(x, dz_half, zeros, f)

        return x
