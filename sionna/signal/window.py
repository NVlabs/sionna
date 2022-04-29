#
# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Layers implementing windowing functions"""

import tensorflow as tf
from tensorflow.keras.layers import Layer
from abc import ABC, abstractmethod
from sionna.utils.tensors import expand_to_rank
import matplotlib.pyplot as plt
import numpy as np

class Window(ABC, Layer):
    # pylint: disable=line-too-long
    r"""Window(length, trainable=False, normalize=False, dtype=tf.float32, **kwargs)

    This is an abtract class for defining and applying a window function of length ``length`` to an input ``x`` of the same length.

    The window function is applied through element-wise multiplication.

    The window function is real-valued, i.e., has `tf.float` as `dtype`.
    The `dtype` of the output is the same as the `dtype` of the input ``x`` to which the window function is applied.
    The window function and the input must have the same precision.

    Parameters
    ----------
    length: int
        Window length (number of samples).

    trainable: bool
        If `True`, the window coefficients are trainable variables.
        Defaults to `False`.

    normalize: bool
        If `True`, the window is normalized to have unit average power
        per coefficient.
        Defaults to `False`.

    dtype: tf.DType
        The `dtype` of the filter coefficients.
        Must be either `tf.float32` or `tf.float64`.
        Defaults to `tf.float32`.

    Input
    -----
    x : [..., N], tf.complex or tf.float
        The input to which the window function is applied.
        The window function is applied along the last dimension.
        The length of the last dimension ``N`` must be the same as the ``length`` of the window function.

    Output
    ------
    y : [...,N], tf.complex or tf.float
        Output of the windowing operation.
        The output has the same shape and `dtype` as the input ``x``.
    """

    def __init__(self,
                 length,
                 trainable=False,
                 normalize=False,
                 dtype=tf.float32,
                 **kwargs):
        super().__init__(dtype=dtype, **kwargs)

        assert length>0, "Length must be positive"
        self._length = length

        assert isinstance(trainable, bool), "trainable must be bool"
        self._trainable = trainable

        assert isinstance(normalize, bool), "normalize must be bool"
        self._normalize = normalize

        assert dtype.is_floating,\
                    "`dtype` must be either `tf.float32` or `tf.float64`"

        self._coefficients = tf.Variable(self._coefficients_source,
                                            trainable=self.trainable,
                                            dtype=tf.as_dtype(self.dtype))

    @property
    @abstractmethod
    def _coefficients_source(self):
        """Internal property that returns the (unormalized) window coefficients.
        Concrete classes that inherits from this one must implement this
        property."""
        pass

    @property
    def coefficients(self):
        """The window coefficients (after normalization)"""
        w = self._coefficients

        # Normalize if requested
        if self.normalize:
            energy = tf.reduce_mean(tf.square(w))
            w = w / tf.cast(tf.sqrt(energy), w.dtype)

        return w

    @property
    def length(self):
        "Window length in number of samples"
        return self._length

    @property
    def trainable(self):
        "`True` if the window coefficients are trainable. `False` otherwise."
        return self._trainable

    @property
    def normalize(self):
        """`True` if the window is normalized to have unit average power per coefficient. `False`
        otherwise."""
        return self._normalize

    def show(self, samples_per_symbol, domain="time", scale="lin"):
        r"""Plot the window in time or frequency domain

        For the computation of the Fourier transform, a minimum DFT size
        of 1024 is assumed which is obtained through zero padding of
        the window coefficients in the time domain.

        Input
        -----
        samples_per_symbol: int
            Number of samples per symbol, i.e., the oversampling factor.

        domain: str, one of ["time", "frequency"]
            The desired domain.
            Defaults to "time"

        scale: str, one of ["lin", "db"]
            The y-scale of the magnitude in the frequency domain.
            Can be "lin" (i.e., linear) or "db" (, i.e., Decibel).
            Defaults to "lin".
        """
        assert domain in ["time", "frequency"], "Invalid domain"
        # Sampling times
        n_min = -(self.length//2)
        n_max = n_min + self.length
        sampling_times = np.arange(n_min, n_max, dtype=np.float32)
        sampling_times /= samples_per_symbol
        #
        if domain=="time":
            plt.figure(figsize=(12,6))
            plt.plot(sampling_times, np.real(self.coefficients.numpy()))
            plt.title("Time domain")
            plt.grid()
            plt.xlabel(r"Normalized time $(t/T)$")
            plt.ylabel(r"$w(t)$")
            plt.xlim(sampling_times[0], sampling_times[-1])
        else:
            assert scale in ["lin", "db"], "Invalid scale"
            fft_size = max(1024, self.coefficients.shape[-1])
            h = np.fft.fft(self.coefficients.numpy(), fft_size)
            h = np.fft.fftshift(h)
            h = np.abs(h)
            plt.figure(figsize=(12,6))
            if scale=="db":
                h = np.maximum(h, 1e-10)
                h = 10*np.log10(h)
                plt.ylabel(r"$|W(f)|$ (dB)")
            else:
                plt.ylabel(r"$|W(f)|$")
            f = np.linspace(-samples_per_symbol/2,
                            samples_per_symbol/2, fft_size)
            plt.plot(f, h)
            plt.title("Frequency domain")
            plt.grid()
            plt.xlabel(r"Normalized frequency $(f/W)$")
            plt.xlim(f[0], f[-1])

    def call(self, x):
        x_dtype = tf.as_dtype(x.dtype)

        # Expand to the same rank as the input for broadcasting
        w = self.coefficients
        w = expand_to_rank(w, tf.rank(x), 0)

        if x_dtype.is_floating:
            y = x*w
        elif x_dtype.is_complex:
            w = tf.complex(w, tf.zeros_like(w))
            y = w*x

        return y


class CustomWindow(Window):
    # pylint: disable=line-too-long
    r"""CustomWindow(length, coefficients=None, trainable=False, normalize=False, dtype=tf.float32, **kwargs)

    Layer for defining and applying a custom window function of length ``length`` to an input ``x`` of the same length.

    The window function is applied through element-wise multiplication.

    The window function is real-valued, i.e., has `tf.float` as `dtype`.
    The `dtype` of the output is the same as the `dtype` of the input ``x`` to which the window function is applied.
    The window function and the input must have the same precision.

    The window coefficients can be set through the ``coefficients`` parameter.
    If not provided, random window coefficients are generated by sampling a Gaussian distribution.

    Parameters
    ----------
    length: int
        Window length (number of samples).

    coefficients: [N], tf.float
        Optional window coefficients.
        If set to `None`, then a random window of length ``length`` is generated by sampling a Gaussian distribution.
        Defaults to `None`.

    trainable: bool
        If `True`, the window coefficients are trainable variables.
        Defaults to `False`.

    normalize: bool
        If `True`, the window is normalized to have unit average power
        per coefficient.
        Defaults to `False`.

    dtype: tf.DType
        The `dtype` of the filter coefficients.
        Must be either `tf.float32` or `tf.float64`.
        Defaults to `tf.float32`.

    Input
    -----
    x : [..., N], tf.complex or tf.float
        The input to which the window function is applied.
        The window function is applied along the last dimension.
        The length of the last dimension ``N`` must be the same as the ``length`` of the window function.

    Output
    ------
    y : [...,N], tf.complex or tf.float
        Output of the windowing operation.
        The output has the same shape and `dtype` as the input ``x``.
    """

    def __init__(self,
                 length,
                 coefficients=None,
                 trainable=False,
                 normalize=False,
                 dtype=tf.float32,
                 **kwargs):

        if coefficients is not None:
            assert len(coefficients) == length,\
                "specified `length` does not match the one of `coefficients`"
            self._c = tf.constant(coefficients, dtype=dtype)
        else:
            self._c = tf.keras.initializers.RandomNormal()([length], dtype)

        super().__init__(length,
                         trainable,
                         normalize,
                         dtype,
                         **kwargs)

    @property
    def _coefficients_source(self):
        return self._c


class HannWindow(Window):
    # pylint: disable=line-too-long
    r"""HannWindow(length, trainable=False, normalize=False, dtype=tf.float32, **kwargs)

    Layer for applying a Hann window function of length ``length`` to an input ``x`` of the same length.

    The window function is applied through element-wise multiplication.

    The window function is real-valued, i.e., has `tf.float` as `dtype`.
    The `dtype` of the output is the same as the `dtype` of the input ``x`` to which the window function is applied.
    The window function and the input must have the same precision.

    The Hann window is defined by

    .. math::
        w_n = \sin^2 \left( \frac{\pi n}{N} \right), 0 \leq n \leq N-1

    where :math:`N` is the window length.

    Parameters
    ----------
    length: int
        Window length (number of samples).

    trainable: bool
        If `True`, the window coefficients are trainable variables.
        Defaults to `False`.

    normalize: bool
        If `True`, the window is normalized to have unit average power
        per coefficient.
        Defaults to `False`.

    dtype: tf.DType
        The `dtype` of the filter coefficients.
        Must be either `tf.float32` or `tf.float64`.
        Defaults to `tf.float32`.

    Input
    -----
    x : [..., N], tf.complex or tf.float
        The input to which the window function is applied.
        The window function is applied along the last dimension.
        The length of the last dimension ``N`` must be the same as the ``length`` of the window function.

    Output
    ------
    y : [...,N], tf.complex or tf.float
        Output of the windowing operation.
        The output has the same shape and `dtype` as the input ``x``.
    """

    @property
    def _coefficients_source(self):
        n = np.arange(self.length)
        coefficients = np.square(np.sin(np.pi*n/self.length))
        return tf.constant(coefficients, self.dtype)


class HammingWindow(Window):
    # pylint: disable=line-too-long
    r"""HammingWindow(length, trainable=False, normalize=False, dtype=tf.float32, **kwargs)

    Layer for applying a Hamming window function of length ``length`` to an input ``x`` of the same length.

    The window function is applied through element-wise multiplication.

    The window function is real-valued, i.e., has `tf.float` as `dtype`.
    The `dtype` of the output is the same as the `dtype` of the input ``x`` to which the window function is applied.
    The window function and the input must have the same precision.

    The Hamming window is defined by

    .. math::
        w_n = a_0 - (1-a_0) \cos \left( \frac{2 \pi n}{N} \right), 0 \leq n \leq N-1

    where :math:`N` is the window length and :math:`a_0 = \frac{25}{46}`.

    Parameters
    ----------
    length: int
        Window length (number of samples).

    trainable: bool
        If `True`, the window coefficients are trainable variables.
        Defaults to `False`.

    normalize: bool
        If `True`, the window is normalized to have unit average power
        per coefficient.
        Defaults to `False`.

    dtype: tf.DType
        The `dtype` of the filter coefficients.
        Must be either `tf.float32` or `tf.float64`.
        Defaults to `tf.float32`.

    Input
    -----
    x : [..., N], tf.complex or tf.float
        The input to which the window function is applied.
        The window function is applied along the last dimension.
        The length of the last dimension ``N`` must be the same as the ``length`` of the window function.

    Output
    ------
    y : [...,N], tf.complex or tf.float
        Output of the windowing operation.
        The output has the same shape and `dtype` as the input ``x``.
    """

    @property
    def _coefficients_source(self):
        n = self.length
        nn = np.arange(n)
        a0 = 25./46.
        a1 = 1. - a0
        coefficients = a0 - a1*np.cos(2.*np.pi*nn/n)
        return tf.constant(coefficients, self.dtype)


class BlackmanWindow(Window):
    # pylint: disable=line-too-long
    r"""BlackmanWindow(length, trainable=False, normalize=False, dtype=tf.float32, **kwargs)

    Layer for applying a Blackman window function of length ``length`` to an input ``x`` of the same length.

    The window function is applied through element-wise multiplication.

    The window function is real-valued, i.e., has `tf.float` as `dtype`.
    The `dtype` of the output is the same as the `dtype` of the input ``x`` to which the window function is applied.
    The window function and the input must have the same precision.

    The Blackman window is defined by

    .. math::
        w_n = a_0 - a_1 \cos \left( \frac{2 \pi n}{N} \right) + a_2 \cos \left( \frac{4 \pi n}{N} \right), 0 \leq n \leq N-1

    where :math:`N` is the window length, :math:`a_0 = \frac{7938}{18608}`, :math:`a_1 = \frac{9240}{18608}`, and :math:`a_2 = \frac{1430}{18608}`.

    Parameters
    ----------
    length: int
        Window length (number of samples).

    trainable: bool
        If `True`, the window coefficients are trainable variables.
        Defaults to `False`.

    normalize: bool
        If `True`, the window is normalized to have unit average power
        per coefficient.
        Defaults to `False`.

    dtype: tf.DType
        The `dtype` of the filter coefficients.
        Must be either `tf.float32` or `tf.float64`.
        Defaults to `tf.float32`.

    Input
    -----
    x : [..., N], tf.complex or tf.float
        The input to which the window function is applied.
        The window function is applied along the last dimension.
        The length of the last dimension ``N`` must be the same as the ``length`` of the window function.

    Output
    ------
    y : [...,N], tf.complex or tf.float
        Output of the windowing operation.
        The output has the same shape and `dtype` as the input ``x``.
    """

    @property
    def _coefficients_source(self):
        n = self.length
        nn = np.arange(n)
        a0 = 7938./18608.
        a1 = 9240./18608.
        a2 = 1430./18608.
        coefficients = a0 - a1*np.cos(2.*np.pi*nn/n) + a2*np.cos(4.*np.pi*nn/n)
        return tf.constant(coefficients, self.dtype)
