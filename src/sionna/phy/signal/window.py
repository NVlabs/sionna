#
# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0#
"""Blocks implementing windowing functions"""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sionna.phy import Block
from sionna.phy.utils import expand_to_rank

class Window(Block):
    # pylint: disable=line-too-long
    r"""
    Abtract class defining a window function

    The window function is applied through element-wise multiplication.

    The window function is real-valued, i.e., has `tf.float` as `dtype`.
    The `dtype` of the output is the same as the `dtype` of the input ``x`` to which the window function is applied.
    The window function and the input must have the same precision.

    Parameters
    ----------
    normalize: `bool`, (default `False`)
        If `True`, the window is normalized to have unit average power
        per coefficient.

    precision : `None` (default) | "single" | "double"
        Precision used for internal calculations and outputs.
        If set to `None`,
        :attr:`~sionna.phy.config.Config.precision` is used.

    Input
    -----
    x : [..., N], `tf.complex` or `tf.float`
        The input to which the window function is applied.
        The window function is applied along the last dimension.
        The length of the last dimension ``N`` must be the same
        as the ``length`` of the window function.

    Output
    ------
    y : [...,N], `tf.complex` or `tf.float`
        Output of the windowing operation
    """

    def __init__(self,
                 normalize=False,
                 precision=None,
                 **kwargs):
        super().__init__(precision=precision, **kwargs)

        assert isinstance(normalize, bool), "normalize must be bool"
        self._normalize = normalize

    @property
    def coefficients(self):
        """
        [N], `tf.float` : Set/get raw window coefficients
            (before normalization)
        """
        return self._coefficients

    @coefficients.setter
    def coefficients(self, v):
        self._coefficients = self._cast_or_check_precision(v)

    @property
    def length(self):
        """
        `int` : Window length in number of samples
        """
        return self.coefficients.shape[0]

    @property
    def normalize(self):
        """
        `bool` : If `True`, the window is normalized to have unit average
        power per coefficient. 
        """
        return self._normalize

    def show(self, samples_per_symbol, domain="time", scale="lin"):
        r"""Plot the window in time or frequency domain

        For the computation of the Fourier transform, a minimum DFT size
        of 1024 is assumed which is obtained through zero padding of
        the window coefficients in the time domain.

        Input
        -----
        samples_per_symbol: `int`
            Number of samples per symbol, i.e., the oversampling factor

        domain: "time" (default) | "frequency"
            Desired domain

        scale: "lin" (default) | "db"
            y-scale of the magnitude in the frequency domain.
            Can be "lin" (i.e., linear) or "db" (, i.e., Decibel).
        """
        assert domain in ["time", "frequency"], "Invalid domain"
        # Normalize if requested
        w = self.coefficients
        if self.normalize:
            energy = tf.reduce_mean(tf.square(w))
            w = w / tf.cast(tf.sqrt(energy), w.dtype)

        # Sampling times
        n_min = -(self.length//2)
        n_max = n_min + self.length
        sampling_times = np.arange(n_min, n_max, dtype=np.float32)
        sampling_times /= samples_per_symbol
        #
        if domain=="time":
            plt.figure(figsize=(12,6))
            plt.plot(sampling_times, np.real(w.numpy()))
            plt.title("Time domain")
            plt.grid()
            plt.xlabel(r"Normalized time $(t/T)$")
            plt.ylabel(r"$w(t)$")
            plt.xlim(sampling_times[0], sampling_times[-1])
        else:
            assert scale in ["lin", "db"], "Invalid scale"
            fft_size = max(1024, w.shape[-1])
            h = np.fft.fft(w.numpy(), fft_size)
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
        w = self.coefficients

        # Normalize if requested
        if self.normalize:
            energy = tf.reduce_mean(tf.square(w))
            w = w / tf.cast(tf.sqrt(energy), w.dtype)

        # Expand to the same rank as the input for broadcasting
        w = expand_to_rank(w, tf.rank(x), 0)

        # Cast to correct dtype if necessary
        if x.dtype.is_complex:
            w = tf.complex(w, tf.zeros_like(w))

        # Apply window
        y = w*x

        return y

class CustomWindow(Window):
    # pylint: disable=line-too-long
    r"""
    Block for defining custom window function

    The window function is applied through element-wise multiplication.

    Parameters
    ----------
    coefficients: [N], `tf.float`
        Window coefficients

    normalize: `bool`, (default `False`)
        If `True`, the window is normalized to have unit average power
        per coefficient.

    precision : `None` (default) | "single" | "double"
        Precision used for internal calculations and outputs.
        If set to `None`,
        :attr:`~sionna.phy.config.Config.precision` is used.

    Input
    -----
    x : [..., N], `tf.complex` or `tf.float`
        Input to which the window function is applied.
        The window function is applied along the last dimension.
        The length of the last dimension ``N`` must be the same as the ``length`` of the window function.

    Output
    ------
    y : [...,N], `tf.complex` or `tf.float`
        Output of the windowing operation
    """

    def __init__(self,
                 coefficients,
                 normalize=False,
                 precision=None,
                 **kwargs):

        super().__init__(normalize=normalize,
                         precision=precision,
                         **kwargs)

        self.coefficients = coefficients

class HannWindow(Window):
    # pylint: disable=line-too-long
    r"""
    Block for defining a Hann window function

    The window function is applied through element-wise multiplication.

    The Hann window is defined by

    .. math::
        w_n = \sin^2 \left( \frac{\pi n}{N} \right), 0 \leq n \leq N-1

    where :math:`N` is the window length.

    Parameters
    ----------
    normalize: `bool`, (default `False`)
        If `True`, the window is normalized to have unit average power
        per coefficient.

    precision : `None` (default) | "single" | "double"
        Precision used for internal calculations and outputs.
        If set to `None`,
        :attr:`~sionna.phy.config.Config.precision` is used.

    Input
    -----
    x : [..., N], `tf.complex` or `tf.float`
        The input to which the window function is applied.
        The window function is applied along the last dimension.
        The length of the last dimension ``N`` must be the same
        as the ``length`` of the window function.

    Output
    ------
    y : [...,N], `tf.complex` or `tf.float`
        Output of the windowing operation
    """
    def __init__(self,
                 normalize=False,
                 precision=None,
                 **kwargs):

        super().__init__(normalize=normalize,
                         precision=precision,
                         **kwargs)

    def build(self, input_shape):
        length = input_shape[-1]
        n = np.arange(length)
        coefficients = np.square(np.sin(np.pi*n/length))
        self.coefficients = coefficients

class HammingWindow(Window):
    # pylint: disable=line-too-long
    r"""
    Block for defining a Hamming window function

    The window function is applied through element-wise multiplication.

    The Hamming window is defined by

    .. math::
        w_n = a_0 - (1-a_0) \cos \left( \frac{2 \pi n}{N} \right), 0 \leq n \leq N-1

    where :math:`N` is the window length and :math:`a_0 = \frac{25}{46}`.

    Parameters
    ----------
    normalize: `bool`, (default `False`)
        If `True`, the window is normalized to have unit average power
        per coefficient.

    precision : `None` (default) | "single" | "double"
        Precision used for internal calculations and outputs.
        If set to `None`,
        :attr:`~sionna.phy.config.Config.precision` is used.

    Input
    -----
    x : [..., N], `tf.complex` or `tf.float`
        The input to which the window function is applied.
        The window function is applied along the last dimension.
        The length of the last dimension ``N`` must be the same
        as the ``length`` of the window function.

    Output
    ------
    y : [...,N], `tf.complex` or `tf.float`
        Output of the windowing operation
    """
    def __init__(self,
                normalize=False,
                precision=None,
                **kwargs):

        super().__init__(normalize=normalize,
                         precision=precision,
                         **kwargs)

    def build(self, input_shape):
        n = input_shape[-1]
        nn = np.arange(n)
        a0 = 25./46.
        a1 = 1. - a0
        coefficients = a0 - a1*np.cos(2.*np.pi*nn/n)
        self.coefficients = coefficients

class BlackmanWindow(Window):
    # pylint: disable=line-too-long
    r"""
    Block for defining a Blackman window function

    The window function is applied through element-wise multiplication.

    The Blackman window is defined by

    .. math::
        w_n = a_0 - a_1 \cos \left( \frac{2 \pi n}{N} \right) + a_2 \cos \left( \frac{4 \pi n}{N} \right), 0 \leq n \leq N-1

    where :math:`N` is the window length, :math:`a_0 = \frac{7938}{18608}`, :math:`a_1 = \frac{9240}{18608}`, and :math:`a_2 = \frac{1430}{18608}`.

    Parameters
    ----------
    normalize: `bool`, (default `False`)
        If `True`, the window is normalized to have unit average power
        per coefficient.

    precision : `None` (default) | "single" | "double"
        Precision used for internal calculations and outputs.
        If set to `None`,
        :attr:`~sionna.phy.config.Config.precision` is used.

    Input
    -----
    x : [..., N], `tf.complex` or `tf.float`
        The input to which the window function is applied.
        The window function is applied along the last dimension.
        The length of the last dimension ``N`` must be the same
        as the ``length`` of the window function.

    Output
    ------
    y : [...,N], `tf.complex` or `tf.float`
        Output of the windowing operation
    """
    def __init__(self,
                normalize=False,
                precision=None,
                **kwargs):

        super().__init__(normalize=normalize,
                         precision=precision,
                         **kwargs)

    def build(self, input_shape):
        n = input_shape[-1]
        nn = np.arange(n)
        a0 = 7938./18608.
        a1 = 9240./18608.
        a2 = 1430./18608.
        coefficients = a0 - a1*np.cos(2.*np.pi*nn/n) + a2*np.cos(4.*np.pi*nn/n)
        self.coefficients = coefficients
