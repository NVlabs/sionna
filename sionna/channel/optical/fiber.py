#
# SPDX-FileCopyrightText: Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

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
    r"""SSFM(alpha=0.046,beta_2=-21.67,f_c=193.55e12,gamma=1.27,half_window_length=0,length=80,n_ssfm=1,n_sp=1.0,sample_duration=1.0,t_norm=1e-12,with_amplification=False,with_attenuation=True,with_dispersion=True,with_manakov=False,with_nonlinearity=True,swap_memory=True,dtype=tf.complex64,**kwargs)

    Layer implementing the split-step Fourier method (SSFM)

    The SSFM (first mentioned in [HT1973]_) numerically solves the generalized
    nonlinear Schrödinger equation (NLSE)

    .. math::

        \frac{\partial E(t,z)}{\partial z}=-\frac{\alpha}{2} E(t,z)+j\frac{\beta_2}{2}\frac{\partial^2 E(t,z)}{\partial t^2}-j\gamma |E(t,z)|^2 E(t,z) + n(n_{\text{sp}};\,t,\,z)

    for an unpolarized (or single polarized) optical signal;
    or the Manakov equation (according to [WMC1991]_)

    .. math::

        \frac{\partial \mathbf{E}(t,z)}{\partial z}=-\frac{\alpha}{2} \mathbf{E}(t,z)+j\frac{\beta_2}{2}\frac{\partial^2 \mathbf{E}(t,z)}{\partial t^2}-j\gamma \frac{8}{9}||\mathbf{E}(t,z)||_2^2 \mathbf{E}(t,z) + \mathbf{n}(n_{\text{sp}};\,t,\,z)

    for dual polarization, with attenuation coefficient :math:`\alpha`, group
    velocity dispersion parameters :math:`\beta_2`, and nonlinearity
    coefficient :math:`\gamma`. The noise terms :math:`n(n_{\text{sp}};\,t,\,z)`
    and :math:`\mathbf{n}(n_{\text{sp}};\,t,\,z)`, respectively, stem from
    an (optional) ideally distributed Raman amplification with
    spontaneous emission factor :math:`n_\text{sp}`. The optical signal
    :math:`E(t,\,z)` has the unit :math:`\sqrt{\text{W}}`. For the dual
    polarized case, :math:`\mathbf{E}(t,\,z)=(E_x(t,\,z), E_y(t,\,z))`
    is a vector consisting of the signal components of both polarizations.

    The symmetrized SSFM is applied according to Eq. (7) of [FMF1976]_ that
    can be written as

    .. math::

        E(z+\Delta_z,t) \approx \exp\left(\frac{\Delta_z}{2}\hat{D}\right)\exp\left(\int^{z+\Delta_z}_z \hat{N}(z')dz'\right)\exp\left(\frac{\Delta_z}{2}\hat{D}\right)E(z,\,t)

    where only the single-polarized case is shown. The integral is
    approximated by :math:`\Delta_z\hat{N}` with :math:`\hat{D}` and
    :math:`\hat{N}` denoting the linear and nonlinear SSFM operator,
    respectively [A2012]_.

    Additionally, ideally distributed Raman amplification may be applied, which
    is implemented as in [MFFP2009]_. Please note that the implemented
    Raman amplification currently results in a transparent fiber link. Hence,
    the introduced gain cannot be parametrized.

    The SSFM operates on normalized time :math:`T_\text{norm}`
    (e.g., :math:`T_\text{norm}=1\,\text{ps}=1\cdot 10^{-12}\,\text{s}`) and
    distance units :math:`L_\text{norm}`
    (e.g., :math:`L_\text{norm}=1\,\text{km}=1\cdot 10^{3}\,\text{m}`).
    Hence, all parameters as well as the signal itself have to be given with the
    same unit prefix for the
    same unit (e.g., always pico for time, or kilo for distance). Despite the normalization,
    the SSFM is implemented with physical
    units, which is different from the normalization, e.g., used for the
    nonlinear Fourier transform. For simulations, only :math:`T_\text{norm}` has to be
    provided.

    To avoid reflections at the signal boundaries during simulation, a Hamming
    window can be applied in each SSFM-step, whose length can be
    defined by ``half_window_length``.

    Example
    --------

    Setting-up:

    >>> ssfm = SSFM(
    >>>     alpha=0.046,
    >>>     beta_2=-21.67,
    >>>     f_c=193.55e12,
    >>>     gamma=1.27,
    >>>     half_window_length=100,
    >>>     length=80,
    >>>     n_ssfm=200,
    >>>     n_sp=1.0,
    >>>     t_norm=1e-12,
    >>>     with_amplification=False,
    >>>     with_attenuation=True,
    >>>     with_dispersion=True,
    >>>     with_manakov=False,
    >>>     with_nonlinearity=True)

    Running:

    >>> # x is the optical input signal
    >>> y = ssfm(x)

    Parameters
    ----------
        alpha : float
            Attenuation coefficient :math:`\alpha` in :math:`(1/L_\text{norm})`.
            Defaults to 0.046.

        beta_2 : float
            Group velocity dispersion coefficient :math:`\beta_2` in :math:`(T_\text{norm}^2/L_\text{norm})`.
            Defaults to -21.67

        f_c : float
            Carrier frequency :math:`f_\mathrm{c}` in :math:`(\text{Hz})`.
            Defaults to 193.55e12.

        gamma : float
            Nonlinearity coefficient :math:`\gamma` in :math:`(1/L_\text{norm}/\text{W})`.
            Defaults to 1.27.

        half_window_length : int
            Half of the Hamming window length. Defaults to 0.

        length : float
            Fiber length :math:`\ell` in :math:`(L_\text{norm})`.
            Defaults to 80.0.

        n_ssfm : int | "adaptive"
            Number of steps :math:`N_\mathrm{SSFM}`.
            Set to "adaptive" to use nonlinear-phase rotation to calculate
            the step widths adaptively (maxmimum rotation can be set in phase_inc).
            Defaults to 1.

        n_sp : float
            Spontaneous emission factor :math:`n_\mathrm{sp}` of Raman amplification.
            Defaults to 1.0.

        sample_duration : float
            Normalized time step :math:`\Delta_t` in :math:`(T_\text{norm})`.
            Defaults to 1.0.

        t_norm : float
            Time normalization :math:`T_\text{norm}` in :math:`(\text{s})`.
            Defaults to 1e-12.

        with_amplification : bool
            Enable ideal inline amplification and corresponding
            noise. Defaults to `False`.

        with_attenuation : bool
            Enable attenuation. Defaults to `True`.

        with_dispersion : bool
            Apply chromatic dispersion. Defaults to `True`.

        with_manakov : bool
            Considers axis [-2] as x- and y-polarization and calculates the
            nonlinear step as given by the Manakov equation. Defaults to `False.`

        with_nonlinearity : bool
            Apply Kerr nonlinearity. Defaults to `True`.

        phase_inc: float
            Maximum nonlinear-phase rotation in rad allowed during simulation.
            To be used with ``n_ssfm`` = "adaptive".
            Defaults to 1e-4.

        swap_memory : bool
            Use CPU memory for while loop. Defaults to `True`.

        dtype : tf.complex
            Defines the datatype for internal calculations and the output
            dtype. Defaults to `tf.complex64`.

    Input
    -----
        x : [...,n] or [...,2,n], tf.complex
            Input signal in :math:`(\sqrt{\text{W}})`. If ``with_manakov`` is `True`,
            the second last dimension is interpreted as x- and y-polarization,
            respectively.

    Output
    ------
        y : Tensor with same shape as ``x``, `tf.complex`
            Channel output
    """
    def __init__(self,
                 alpha=0.046,
                 beta_2=-21.67,
                 f_c=193.55e12,
                 gamma=1.27,
                 half_window_length=0,
                 length=80,
                 n_ssfm=1,
                 n_sp=1.0,
                 sample_duration=1.0,
                 t_norm=1e-12,
                 with_amplification=False,
                 with_attenuation=True,
                 with_dispersion=True,
                 with_manakov=False,
                 with_nonlinearity=True,
                 phase_inc=1e-4,
                 swap_memory=True,
                 dtype=tf.complex64,
                 **kwargs):

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
        self._phase_inc = tf.cast(phase_inc, dtype=self._rdtype)

        if n_ssfm == "adaptive":
            self._n_ssfm = tf.cast(-1, dtype=tf.int32) # adaptive == -1
        elif isinstance(n_ssfm, int):
            self._n_ssfm = tf.cast(n_ssfm, dtype=tf.int32)
            # Precalculate uniform step size
            tf.assert_greater(self._n_ssfm, 0)
        else:
            raise ValueError("Unsupported parameter for n_ssfm. \
                              Either an integer or 'adaptive'.")

        # only used for constant step width -> negative value calculated
        # with adaptive step widths can be ignored
        self._dz = self._length / tf.cast(self._n_ssfm, dtype=self._rdtype)
        self._n_sp = tf.cast(n_sp, dtype=self._rdtype)
        self._swap_memory = swap_memory
        self._t_norm = tf.cast(t_norm, dtype=self._rdtype)
        self._sample_duration = tf.cast(sample_duration, dtype=self._rdtype)

        # Booleans are not casted to avoid branches in the graph
        self._with_amplification = with_amplification
        self._with_attenuation = with_attenuation
        self._with_dispersion = with_dispersion
        self._with_manakov = with_manakov
        self._with_nonlinearity = with_nonlinearity

        self._rho_n = \
            sionna.constants.H * self._f_c * self._alpha * self._length * \
            self._n_sp  # in (W/Hz)

        # Calculate noise power depending on simulation bandwidth
        self._p_n_ase = self._rho_n / self._sample_duration / self._t_norm
        # in (Ws)
        if self._with_manakov:
            self._p_n_ase = self._p_n_ase / 2.0

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
            step_noise = self._p_n_ase * tf.cast(dz, self._rdtype) \
                        / tf.cast(self._length, self._rdtype) \
                        / tf.cast(2.0, self._rdtype)
            q_n = tf.complex(
                tf.random.normal(
                    q.shape,
                    tf.cast(0.0, self._rdtype),
                    tf.sqrt(step_noise),
                    self._rdtype),
                tf.random.normal(
                    q.shape,
                    tf.cast(0.0, self._rdtype),
                    tf.sqrt(step_noise),
                    self._rdtype)
            )
            q = q + q_n

        return q

    def _apply_nonlinear_operator(self, q, dz, zeros):
        if self._with_nonlinearity:
            if self._with_manakov:
                q = q * tf.exp(
                    tf.complex(
                        zeros,
                        tf.cast(8.0/9.0, self._rdtype) * tf.reduce_sum(
                            tf.abs(q) ** tf.cast(2.0, self._rdtype),
                            axis=-2,
                            keepdims=True
                        ) * self._gamma * tf.negative(tf.math.real(dz)))
                )
            else:
                q = q * tf.exp(
                    tf.complex(
                        zeros,
                        tf.abs(q) ** tf.cast(2.0, self._rdtype) * self._gamma *
                        tf.negative(tf.math.real(dz)))
                )

        return q


    def _calculate_step_width(self, q, remaining_length):
        max_power = tf.math.reduce_max(tf.math.pow(tf.math.abs(q),2.0),axis=None)
        # ensure that the exact length is reached in the end
        dz = tf.math.minimum(self._phase_inc / self._gamma / max_power,remaining_length)
        return dz

    def _adaptive_step(self,q, precalculations, remaining_length, step_counter):

        (window, _, zeros, f) = precalculations

        dz = self._calculate_step_width(q,remaining_length)

        # Apply window-function
        q = self._apply_window(q, window)
        q = self._apply_linear_operator(q, dz, zeros, f)  # D
        q = self._apply_nonlinear_operator(q, dz, zeros)  # N
        q = self._apply_noise(q, dz)
        remaining_length = remaining_length - dz

        precalculations = (window, dz, zeros, f)
        step_counter = step_counter + 1
        return q, precalculations, remaining_length, step_counter

    def _cond_adaptive(self, q, precalculations,remaining_length,step_counter):
        # pylint: disable=unused-argument
        return tf.greater_equal(remaining_length, 1e-3) # avoid numerical issues for 0


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
        if self._with_manakov:
            tf.assert_equal(tf.shape(inputs)[-2], 2)

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
        # SSFM step counter
        iterator = tf.constant(0, dtype=tf.int32, name="step_counter")

        if self._n_ssfm == -1: # adaptive step width

            x, _, _, _ = tf.while_loop(
                self._cond_adaptive,
                self._adaptive_step,
                (x, (window, tf.cast(0.,self._rdtype), zeros, f), self._length, iterator),
                swap_memory=self._swap_memory,
                parallel_iterations=1
            )

        # constant step size
        else:
            # Spatial step size
            dz = tf.cast(self._dz, dtype=self._rdtype)

            dz_half = dz/tf.cast(2.0, self._rdtype)

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
