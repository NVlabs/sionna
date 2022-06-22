# $Id: fiber.py 0 13.06.2022 13:53$
# Author: Tim Alexander Uhlemann <uhlemann@ieee.org>
# Copyright:

"""
This module defines the following classes:

- `SSFM`, the split-step Fourier method to approximate the solution of the NLSE

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

class SSFM(Layer):
    r"""SSFM(alpha, beta_2, f_c, gamma, half_window_length, ell, n_ssfm, n_sp, dt, t_norm, with_amplification, with_attenuation, with_dispersion, with_nonlinearity, swap_memory=True, dtype=tf.complex64, **kwargs)

    Perform SSFM in normalized units on optical input signal.

    The normalized time unit is :math:`T_\text{norm}` (e.g., 1 ps) in the
    following. The
    same scheme is used for normalized length :math:`L_\text{norm}` (e.g.,
    1 km). Note that, all parameters have to be given in the same normalized
    units.

    This class inherits from the Keras `Layer` class and can be used as layer in
    a Keras model.

    Example
    --------

    Setting-up:

    >>> ssfm_channel = SSFM(
    >>>     0.046, -21.67, 193.55e12, 1.27, 100, 80, 200, 1e-17, 1.0,
    >>>     True, True, True, True)

    Running:

    >>> y = ssfm_channel(
    >>>     ([1.0+1.0j, 1.0+1.0j, 1.0+1.0j])
    >>> )

    Parameters
    ----------
        alpha : Tensor, tf.float
            Attenuation coefficient in :math:`(1/L_\text{norm})`.

        beta_2 : Tensor, tf.float
            Chromatic dispersion coefficient in :math:`(T_\text{norm}^2/L_\text{norm})`.

        f_c : Tensor, tf.float
            Carrier frequency in :math:`(1/T_\text{norm})`.

        gamma : Tensor, tf.float
            Kerr-nonlinearity in :math:`(1/L_\text{norm}/W)`.

        half_window_length : Tensor, tf.int
            Half of the Hammwing window length in :math:`(1)`.

        ell : Tensor, tf.float
            Fiber length in :math:`(L_\text{norm})`.

        n_ssfm : Tensor, tf.int
            Number of steps in :math:`(1)`.

        n_sp : Tensor, tf.float
            Spontaneous emission factor of Raman amplification in :math:`(1)`.

        dt : Tensor, tf.float
            Time step in :math:`(1)`

        t_norm : Tensor, tf.float
            Time normalization in :math:`(s/T_\text{norm})`.

        with_attenuation : Tensor, tf.bool
            Enables application of attenuation (True/False).

        with_amplification : Tensor, tf.bool
            Enables application of ideal inline amplification and corresponding
            noise (True/False).

        with_dispersion : Tensor, tf.bool
            Enables application of chromatic dispersion (True/False).

        with_nonlinearity : Tensor, tf.bool
            Enables application of Kerr nonlinearity (True/False).

        swap_memory : Tensor, tf.float
            Use CPU memory for while loop (True/False). Defaults to True.

        dtype : Complex tf.DType
            Defines the datatype for internal calculations and the output
            dtype. Defaults to `tf.complex64`.

    Input
    -----
        (x, dt, with_attenuation, with_amplification, with_dispersion, with_nonlinearity) : Tuple:

        x : Tensor, tf.complex
            Input signal in :math:'(W^(1/2))'


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

        self._alpha = tf.cast(alpha, dtype=dtype)
        self._beta_2 = tf.cast(beta_2, dtype=dtype.real_dtype)
        self._f_c = tf.cast(f_c, dtype=dtype.real_dtype)
        self._gamma = tf.cast(gamma, dtype=dtype.real_dtype)
        self._half_window_length = half_window_length
        self._ell = tf.cast(ell, dtype=dtype.real_dtype)
        self._n_ssfm = tf.cast(n_ssfm, dtype=tf.int32)
        self._n_sp = tf.cast(n_sp, dtype=dtype.real_dtype)
        self._swap_memory = swap_memory
        self._t_norm = t_norm
        self._with_amplification = tf.cast(with_amplification, dtype=tf.bool)
        # self._with_attenuation = tf.cast(with_attenuation, dtype=tf.bool)
        self._with_attenuation = with_attenuation
        self._with_dispersion = tf.cast(with_dispersion, dtype=tf.bool)
        self._with_nonlinearity = tf.cast(with_nonlinearity, dtype=tf.bool)
        self._complex_dtype = dtype
        self._dt = tf.cast(dt, dtype=dtype.real_dtype)

        self._rho_n = \
            sionna.constants.H / (self._t_norm ** 2.0) * self._f_c * tf.cast(
                self._alpha, dtype.real_dtype) * self._ell * self._n_sp

        # Calculate noise power depending on simulation bandwidth
        self._p_n_ase = self._rho_n / self._dt

        # Precalculate dispersion
        self._dz = self._ell / tf.cast(self._n_ssfm, dtype=dtype.real_dtype)

        self._window = tf.complex(
            tf.signal.hamming_window(
                window_length=2*self._half_window_length,
                dtype=self._complex_dtype.real_dtype
            ),
            tf.zeros(
                2*self._half_window_length,
                dtype=self._complex_dtype.real_dtype
            )
        )

    def call(self, inputs):
        (x) = inputs

        cdtype = self._complex_dtype
        rdtype = self._complex_dtype.real_dtype

        # Calculate support parameters
        input_shape = x.shape

        # Generate frequency vectors
        t, f = utils.time_frequency_vector(
            x.shape[-1], self._dt, dtype=rdtype)

        # Window function calculation
        window = tf.concat(
            [
                self._window[0:self._half_window_length],
                tf.complex(
                    tf.ones(
                        [input_shape[-1] - 2*self._half_window_length],
                        dtype=rdtype
                    ),
                    tf.zeros(
                        [input_shape[-1] - 2*self._half_window_length],
                        dtype=rdtype
                    )
                ),
                self._window[self._half_window_length::]
            ],
            axis=0
        )

        # SSFM
        dispersion = tf.exp(
            tf.complex(
                tf.zeros(input_shape[-1], rdtype),
                -self._beta_2 / tf.cast(2.0, rdtype) * self._dz *
                (tf.cast(2.0, rdtype) * tf.cast(sionna.constants.PI, rdtype) *
                 f) ** tf.cast(2.0, rdtype)
            )
        )
        dispersion = tf.signal.fftshift(dispersion, axes=-1)

        zeros = tf.zeros(tf.shape(x), dtype=rdtype)

        dz = tf.complex(
            self._dz,
            tf.zeros(tf.shape(self._dz), dtype=rdtype))

        # SSFM step counter
        iterator = tf.constant(0, dtype=tf.int32, name="step_counter")

        def body(q, step_counter):
            # Apply window-function
            q = q * window

            # Nonlinearity
            # Precalculate nonlinearity (outside if for graph execution)
            nonlinearity = tf.complex(
                zeros,
                tf.abs(q) ** tf.cast(2.0, rdtype) * self._gamma *
                tf.negative(tf.math.real(dz)))
            if self._with_nonlinearity:
                # First half step?
                if step_counter == 0:
                    scaling = tf.cast(0.5, rdtype)
                else:
                    scaling = tf.cast(1.0, rdtype)
                q = q * tf.exp(nonlinearity * tf.cast(scaling, cdtype))

            # Chromatic dispersion
            if self._with_dispersion:
                q = tf.signal.ifft(tf.signal.fft(q) * dispersion)

            # Attenuation
            if self._with_attenuation:
                q = q * tf.exp(-self._alpha / 2.0 * dz)

            # Amplification (Raman)
            if self._with_amplification:
                q_n = tf.complex(
                    tf.random.normal(
                        q.shape,
                        tf.cast(0.0, rdtype),
                        tf.sqrt(
                            self._p_n_ase /
                            tf.cast(self._n_ssfm, rdtype) /
                            tf.cast(2.0, rdtype)
                        ),
                        rdtype),
                    tf.random.normal(
                        q.shape,
                        tf.cast(0.0, rdtype),
                        tf.sqrt(
                            self._p_n_ase /
                            tf.cast(self._n_ssfm, rdtype) /
                            tf.cast(2.0, rdtype)
                        ),
                        rdtype
                    )
                )
                q = q * tf.exp(self._alpha/2.0 * dz) + q_n

            # Nonlinearity
            if self._with_nonlinearity and step_counter == (self._n_ssfm - 1):
                q = q * tf.exp(
                    nonlinearity * tf.cast(
                        tf.cast(0.5, rdtype), cdtype))

            step_counter = step_counter + 1

            return q, step_counter

        def cond(q, step_counter):
            return tf.less(step_counter, self._n_ssfm)

        y, _ = tf.while_loop(
            cond,
            body,
            (x, iterator),
            swap_memory=self._swap_memory,
            parallel_iterations=1
        )

        return y
