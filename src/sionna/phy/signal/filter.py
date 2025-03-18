#
# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0#
"""Blocks implementing filters"""

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sionna.phy import Block
from . import convolve, Window, HannWindow, HammingWindow, BlackmanWindow, empirical_aclr

class Filter(Block):
    # pylint: disable=line-too-long
    r"""
    Abtract class defining a filter of ``length`` K which can be
    applied to an input ``x`` of length N

    The filter length K is equal to the filter span in symbols (``span_in_symbols``)
    multiplied by the oversampling factor (``samples_per_symbol``).
    If this product is even, a value of one will be added.

    The filter is applied through discrete convolution.

    An optional windowing function ``window`` can be applied to the filter.

    Three padding modes are available for applying the filter:

    *   "full" (default): Returns the convolution at each point of overlap between ``x`` and the filter.
        The length of the output is N + K - 1. Zero-padding of the input ``x`` is performed to
        compute the convolution at the borders.
    *   "same": Returns an output of the same length as the input ``x``. The convolution is computed such
        that the coefficients of the input ``x`` are centered on the coefficient of the filter with index
        (K-1)/2. Zero-padding of the input signal is performed to compute the convolution at the borders.
    *   "valid": Returns the convolution only at points where ``x`` and the filter completely overlap.
        The length of the output is N - K + 1.

    Parameters
    ----------
    span_in_symbols: `int`
        Filter span as measured by the number of symbols

    samples_per_symbol: `int`
        Number of samples per symbol, i.e., the oversampling factor

    window: `None` (default) | :class:`~sionna.phy.signal.Window` | "hann" | "hamming" | "blackman"
        Window that is applied to the filter coefficients

    normalize: `bool`, (default `True`)
        If `True`, the filter is normalized to have unit power.

    precision : `None` (default) | "single" | "double"
        Precision used for internal calculations and outputs.
        If set to `None`,
        :attr:`~sionna.phy.config.Config.precision` is used.

    Input
    -----
    x : [...,N], `tf.complex` or `tf.float`
        Input to which the filter is applied along the last dimension

    padding : "full" (default) | "valid" | "same"
        Padding mode for convolving ``x`` and the filter

    conjugate : `bool`, (default `False`)
        If `True`, the complex conjugate of the filter is applied.

    Output
    ------
    y : [...,M], `tf.complex` or `tf.float`
        Filtered input. The length M depends on the ``padding``.
    """
    def __init__(self,
                 span_in_symbols,
                 samples_per_symbol,
                 window=None,
                 normalize=True,
                 precision=None,
                 **kwargs):
        super().__init__(precision=precision, **kwargs)

        assert span_in_symbols>0, "span_in_symbols must be positive"
        self._span_in_symbols = span_in_symbols

        assert samples_per_symbol>0, "samples_per_symbol must be positive"
        self._samples_per_symbol = samples_per_symbol

        self.window = window

        assert isinstance(normalize, bool), "normalize must be bool"
        self._normalize = normalize

    @property
    def span_in_symbols(self):
        """
        `int` : Filter span in symbols
        """
        return self._span_in_symbols

    @property
    def samples_per_symbol(self):
        """
        `int` : Number of samples per symbol, i.e., the oversampling factor
        """
        return self._samples_per_symbol

    @property
    def length(self):
        """
        `int` : Filter length in samples
        """
        l = self._span_in_symbols*self._samples_per_symbol
        l = 2*(l//2)+1 # Force length to be the next odd number
        return l

    @property
    def window(self):
        """
        :class:`~sionna.phy.signal.Window` : Get/set window function that
            is applied to the filter coefficients
        """
        return self._window

    @window.setter
    def window(self, value):
        if isinstance(value, str):
            if value=="hann":
                self._window = HannWindow()
            elif value=="hamming":
                self._window = HammingWindow()
            elif value=="blackman":
                self._window = BlackmanWindow()
            else:
                raise AssertionError("Invalid window type")
        elif isinstance(value, Window) or value is None:
            self._window = value
        else:
            raise AssertionError("Invalid window type")
        if value is not None:
            assert self._window.precision == self._precision, \
                "Window and Filter must have the same precision."
            # Run window once to initialize coefficients
            self._window(tf.ones([self.length], self.cdtype))
    @property
    def normalize(self):
        """
        `bool` : If `True` the filter is normalized to have unit power.
        """
        return self._normalize

    @property
    def coefficients(self):
        """
        [K], `tf.float` of `tf.complex` : Set/get raw filter coefficients
        """
        return self._coefficients

    @coefficients.setter
    def coefficients(self, v):
        self._coefficients = self._cast_or_check_precision(v)

    @property
    def sampling_times(self):
        """
        [K], `numpy.float32` : Sampling times in multiples of
            the symbol duration
        """
        n_min = -(self.length//2)
        n_max = n_min + self.length
        t = np.arange(n_min, n_max, dtype=np.float32)
        t /= self.samples_per_symbol
        return t

    def show(self, response="impulse", scale="lin"):
        r"""Plot the impulse or magnitude response

        Plots the impulse response (time domain) or magnitude response
        (frequency domain) of the filter.

        For the computation of the magnitude response, a minimum DFT size
        of 1024 is assumed which is obtained through zero padding of
        the filter coefficients in the time domain.

        Input
        -----
        response: "impulse" (default) | "magnitude"
            Desired response type

        scale: "lin" (default) | "db"
            y-scale of the magnitude response.
            Can be "lin" (i.e., linear) or "db" (, i.e., Decibel).
        """
        assert response in ["impulse", "magnitude"], "Invalid response"

        h = self.coefficients

        # Apply window
        if self.window is not None:
            h = self._window(h)

        # Ensure unit L2-norm of the coefficients
        if self.normalize:
            energy = tf.reduce_sum(tf.square(tf.abs(h)))
            h = h / tf.cast(tf.sqrt(energy), h.dtype)

        if response=="impulse":
            plt.figure(figsize=(12,6))
            plt.plot(self.sampling_times, np.real(h))
            if self.coefficients.dtype.is_complex:
                plt.plot(self.sampling_times, np.imag(h))
                plt.legend(["Real part", "Imaginary part"])
            plt.title("Impulse response")
            plt.grid()
            plt.xlabel(r"Normalized time $(t/T)$")
            plt.ylabel(r"$h(t)$")
            plt.xlim(self.sampling_times[0], self.sampling_times[-1])

        else:
            assert scale in ["lin", "db"], "Invalid scale"
            fft_size = max(1024, h.shape[-1])
            h = np.fft.fft(h, fft_size)
            h = np.fft.fftshift(h)
            h = np.abs(h)
            plt.figure(figsize=(12,6))
            if scale=="db":
                h = np.maximum(h, 1e-10)
                h = 10*np.log10(h)
                plt.ylabel(r"$|H(f)|$ (dB)")
            else:
                plt.ylabel(r"$|H(f)|$")
            f = np.linspace(-self._samples_per_symbol/2,
                            self._samples_per_symbol/2, fft_size)
            plt.plot(f, h)
            plt.title("Magnitude response")
            plt.grid()
            plt.xlabel(r"Normalized frequency $(f/W)$")
            plt.xlim(f[0], f[-1])

    @property
    def aclr(self):
        """ACLR of the filter

        This ACLR corresponds to what one would obtain from using
        this filter as pulse shaping filter on an i.i.d. sequence of symbols.
        The in-band is assumed to range from [-0.5, 0.5] in normalized
        frequency.

        `tf.float` : ACLR in linear scale
        """
        h = self.coefficients

        # Apply window
        if self.window is not None:
            h = self._window(h)

        # Ensure unit L2-norm of the coefficients
        if self.normalize:
            energy = tf.reduce_sum(tf.square(tf.abs(h)))
            h = h / tf.cast(tf.sqrt(energy), h.dtype)

        fft_size = 1024
        n = fft_size - tf.shape(h)[-1]
        z = tf.zeros([n], h.dtype)
        c = tf.cast(tf.concat([h, z], -1), tf.complex64)
        return empirical_aclr(c,
                              oversampling=self._samples_per_symbol,
                              precision=self.precision)

    def call(self, x, padding='full', conjugate=False):
        h = self.coefficients

        # Apply window
        if self.window is not None:
            h = self._window(h)

        # Ensure unit L2-norm of the coefficients
        if self.normalize:
            energy = tf.reduce_sum(tf.square(tf.abs(h)))
            h = h / tf.cast(tf.sqrt(energy), h.dtype)

        # (Optionally) compute the complex conjugate
        if conjugate and h.dtype.is_complex:
            h = tf.math.conj(h)

        y = convolve(x, h, padding=padding, precision=self.precision)
        return y

class RaisedCosineFilter(Filter):
    # pylint: disable=line-too-long
    r"""
    Block for applying a raised-cosine filter of ``length`` K
    to an input ``x`` of length N

    The raised-cosine filter is defined by

    .. math::
        h(t) =
        \begin{cases}
        \frac{\pi}{4T} \text{sinc}\left(\frac{1}{2\beta}\right), & \text { if }t = \pm \frac{T}{2\beta}\\
        \frac{1}{T}\text{sinc}\left(\frac{t}{T}\right)\frac{\cos\left(\frac{\pi\beta t}{T}\right)}{1-\left(\frac{2\beta t}{T}\right)^2}, & \text{otherwise}
        \end{cases}

    where :math:`\beta` is the roll-off factor and :math:`T` the symbol duration.

    The filter length K is equal to the filter span in symbols (``span_in_symbols``)
    multiplied by the oversampling factor (``samples_per_symbol``).
    If this product is even, a value of one will be added.

    The filter is applied through discrete convolution.

    An optional windowing function ``window`` can be applied to the filter.

    The `dtype` of the output is `tf.float` if both ``x`` and the filter coefficients have dtype `tf.float`.
    Otherwise, the dtype of the output is `tf.complex`.

    Three padding modes are available for applying the filter:

    *   "full" (default): Returns the convolution at each point of overlap between ``x`` and the filter.
        The length of the output is N + K - 1. Zero-padding of the input ``x`` is performed to
        compute the convolution at the borders.
    *   "same": Returns an output of the same length as the input ``x``. The convolution is computed such
        that the coefficients of the input ``x`` are centered on the coefficient of the filter with index
        (K-1)/2. Zero-padding of the input signal is performed to compute the convolution at the borders.
    *   "valid": Returns the convolution only at points where ``x`` and the filter completely overlap.
        The length of the output is N - K + 1.

    Parameters
    ----------
    span_in_symbols: `int`
        Filter span as measured by the number of symbols

    samples_per_symbol: `int`
        Number of samples per symbol, i.e., the oversampling factor

    beta : `float`
        Roll-off factor.
        Must be in the range :math:`[0,1]`.

    window: `None` (default) | :class:`~sionna.phy.signal.Window` | "hann" | "hamming" | "blackman"
        Window that is applied to the filter coefficients

    normalize: `bool`, (default `True`)
        If `True`, the filter is normalized to have unit power.

    precision : `None` (default) | "single" | "double"
        Precision used for internal calculations and outputs.
        If set to `None`,
        :attr:`~sionna.phy.config.Config.precision` is used.

    Input
    -----
    x : [...,N], `tf.complex` or `tf.float`
        Input to which the filter is applied along the last dimension

    padding : "full" (default) | "valid" | "same"
        Padding mode for convolving ``x`` and the filter

    conjugate : `bool`, (default `False`)
        If `True`, the complex conjugate of the filter is applied.

    Output
    ------
    y : [...,M], `tf.complex` or `tf.float`
        Filtered input. The length M depends on the ``padding``.
    """
    def __init__(self,
                 span_in_symbols,
                 samples_per_symbol,
                 beta,
                 window=None,
                 normalize=True,
                 precision=None,
                 **kwargs):

        super().__init__(span_in_symbols,
                         samples_per_symbol,
                         window=window,
                         normalize=normalize,
                         precsion=precision,
                         **kwargs)

        assert 0 <= beta <= 1, "beta must be from the intervall [0,1]"
        self._beta = beta
        self.coefficients = self._raised_cosine(self.sampling_times,
                                                1.0,
                                                self.beta)

    @property
    def beta(self):
        """
        `float` : Roll-off factor
        """
        return self._beta

    def _raised_cosine(self, t, symbol_duration, beta):
        """Raised-cosine filter from Wikipedia
        https://en.wikipedia.org/wiki/Raised-cosine_filter"""
        h = np.zeros([len(t)], np.float32)
        for i, tt in enumerate(t):
            tt = np.abs(tt)
            if beta>0 and (tt-np.abs(symbol_duration/2/beta)==0):
                h[i] = np.pi/4/symbol_duration*np.sinc(1/2/beta)
            else:
                h[i] = 1./symbol_duration*np.sinc(tt/symbol_duration)\
                    * np.cos(np.pi*beta*tt/symbol_duration)\
                    /(1-(2*beta*tt/symbol_duration)**2)
        return h

class RootRaisedCosineFilter(Filter):
    # pylint: disable=line-too-long
    r"""
    Block for applying a root-raised-cosine filter of ``length`` K
    to an input ``x`` of length N

    The root-raised-cosine filter is defined by

    .. math::
        h(t) =
        \begin{cases}
        \frac{1}{T} \left(1 + \beta\left(\frac{4}{\pi}-1\right) \right), & \text { if }t = 0\\
        \frac{\beta}{T\sqrt{2}} \left[ \left(1+\frac{2}{\pi}\right)\sin\left(\frac{\pi}{4\beta}\right) + \left(1-\frac{2}{\pi}\right)\cos\left(\frac{\pi}{4\beta}\right) \right], & \text { if }t = \pm\frac{T}{4\beta} \\
        \frac{1}{T} \frac{\sin\left(\pi\frac{t}{T}(1-\beta)\right) + 4\beta\frac{t}{T}\cos\left(\pi\frac{t}{T}(1+\beta)\right)}{\pi\frac{t}{T}\left(1-\left(4\beta\frac{t}{T}\right)^2\right)}, & \text { otherwise}
        \end{cases}

    where :math:`\beta` is the roll-off factor and :math:`T` the symbol duration.

    The filter length K is equal to the filter span in symbols (``span_in_symbols``)
    multiplied by the oversampling factor (``samples_per_symbol``).
    If this product is even, a value of one will be added.

    The filter is applied through discrete convolution.

    An optional windowing function ``window`` can be applied to the filter.

    The `dtype` of the output is `tf.float` if both ``x`` and the filter coefficients have dtype `tf.float`.
    Otherwise, the dtype of the output is `tf.complex`.

    Three padding modes are available for applying the filter:

    *   "full" (default): Returns the convolution at each point of overlap between ``x`` and the filter.
        The length of the output is N + K - 1. Zero-padding of the input ``x`` is performed to
        compute the convolution at the borders.
    *   "same": Returns an output of the same length as the input ``x``. The convolution is computed such
        that the coefficients of the input ``x`` are centered on the coefficient of the filter with index
        (K-1)/2. Zero-padding of the input signal is performed to compute the convolution at the borders.
    *   "valid": Returns the convolution only at points where ``x`` and the filter completely overlap.
        The length of the output is N - K + 1.

    Parameters
    ----------
    span_in_symbols: `int`
        Filter span as measured by the number of symbols

    samples_per_symbol: `int`
        Number of samples per symbol, i.e., the oversampling factor

    beta : `float`
        Roll-off factor.
        Must be in the range :math:`[0,1]`.

    window: `None` (default) | :class:`~sionna.phy.signal.Window` | "hann" | "hamming" | "blackman"
        Window that is applied to the filter coefficients

    normalize: `bool`, (default `True`)
        If `True`, the filter is normalized to have unit power.

    precision : `None` (default) | "single" | "double"
        Precision used for internal calculations and outputs.
        If set to `None`,
        :attr:`~sionna.phy.config.Config.precision` is used.

    Input
    -----
    x : [...,N], `tf.complex` or `tf.float`
        Input to which the filter is applied along the last dimension

    padding : "full" (default) | "valid" | "same"
        Padding mode for convolving ``x`` and the filter

    conjugate : `bool`, (default `False`)
        If `True`, the complex conjugate of the filter is applied.

    Output
    ------
    y : [...,M], `tf.complex` or `tf.float`
        Filtered input. The length M depends on the ``padding``.
    """
    def __init__(self,
                 span_in_symbols,
                 samples_per_symbol,
                 beta,
                 window=None,
                 normalize=True,
                 precision=None,
                 **kwargs):

        super().__init__(span_in_symbols,
                         samples_per_symbol,
                         window=window,
                         normalize=normalize,
                         precision=precision,
                         **kwargs)

        assert 0 <= beta <= 1, "beta must be from the intervall [0,1]"
        self._beta = beta
        self.coefficients = self._root_raised_cosine(self.sampling_times,
                                                     1.0,
                                                     self.beta)

    @property
    def beta(self):
        """
        `float`: Roll-off factor
        """
        return self._beta

    def _root_raised_cosine(self, t, symbol_duration, beta):
        """Root-raised-cosine filter from Wikipedia
            https://en.wikipedia.org/wiki/Root-raised-cosine_filter"""
        h = np.zeros([len(t)], np.float32)
        for i, tt in enumerate(t):
            tt = np.abs(tt)
            if tt==0:
                h[i] = 1/symbol_duration*(1+beta*(4/np.pi-1))
            elif beta>0 and (tt-np.abs(symbol_duration/4/beta)==0):
                h[i] = beta/symbol_duration/np.sqrt(2)\
                    * ((1+2/np.pi)*np.sin(np.pi/4/beta) + \
                                            (1-2/np.pi)*np.cos(np.pi/4/beta))
            else:
                h[i] = 1/symbol_duration\
                / (np.pi*tt/symbol_duration*(1-(4*beta*tt/symbol_duration)**2))\
                * (np.sin(np.pi*tt/symbol_duration*(1-beta)) + \
                4*beta*tt/symbol_duration\
                *np.cos(np.pi*tt/symbol_duration*(1+beta)))
        return h

class SincFilter(Filter):
    # pylint: disable=line-too-long
    r"""
    Block for applying a sinc filter of ``length`` K
    to an input ``x`` of length N

    The sinc filter is defined by

    .. math::
        h(t) = \frac{1}{T}\text{sinc}\left(\frac{t}{T}\right)

    where :math:`T` the symbol duration.

    The filter length K is equal to the filter span in symbols (``span_in_symbols``)
    multiplied by the oversampling factor (``samples_per_symbol``).
    If this product is even, a value of one will be added.

    The filter is applied through discrete convolution.

    An optional windowing function ``window`` can be applied to the filter.

    The `dtype` of the output is `tf.float` if both ``x`` and the filter coefficients have dtype `tf.float`.
    Otherwise, the dtype of the output is `tf.complex`.

    Three padding modes are available for applying the filter:

    *   "full" (default): Returns the convolution at each point of overlap between ``x`` and the filter.
        The length of the output is N + K - 1. Zero-padding of the input ``x`` is performed to
        compute the convolution at the borders.
    *   "same": Returns an output of the same length as the input ``x``. The convolution is computed such
        that the coefficients of the input ``x`` are centered on the coefficient of the filter with index
        (K-1)/2. Zero-padding of the input signal is performed to compute the convolution at the borders.
    *   "valid": Returns the convolution only at points where ``x`` and the filter completely overlap.
        The length of the output is N - K + 1.

    Parameters
    ----------
    span_in_symbols: `int`
        Filter span as measured by the number of symbols

    samples_per_symbol: `int`
        Number of samples per symbol, i.e., the oversampling factor

    window: `None` (default) | :class:`~sionna.phy.signal.Window` | "hann" | "hamming" | "blackman"
        Window that is applied to the filter coefficients

    normalize: `bool`, (default `True`)
        If `True`, the filter is normalized to have unit power

    precision : `None` (default) | "single" | "double"
        Precision used for internal calculations and outputs.
        If set to `None`,
        :attr:`~sionna.phy.config.Config.precision` is used.

    Input
    -----
    x : [...,N], `tf.complex` or `tf.float`
        Input to which the filter is applied along the last dimension

    padding : "full" (default) | "valid" | "same"
        Padding mode for convolving ``x`` and the filter

    conjugate : `bool`, (default `False`)
        If `True`, the complex conjugate of the filter is applied.

    Output
    ------
    y : [...,M], `tf.complex` or `tf.float`
        Filtered input. The length M depends on the ``padding``.
    """
    def __init__(self,
                 span_in_symbols,
                 samples_per_symbol,
                 window=None,
                 normalize=True,
                 precision=None,
                 **kwargs):

        super().__init__(span_in_symbols,
                         samples_per_symbol,
                         window=window,
                         normalize=normalize,
                         precision=precision,
                         **kwargs)

        self.coefficients = self._sinc(self.sampling_times, 1.0)

    def _sinc(self, t, symbol_duration):
        """Sinc filter"""
        return 1/symbol_duration*np.sinc(t/symbol_duration)

class CustomFilter(Filter):
    # pylint: disable=line-too-long
    r"""
    Block for applying a custom filter of ``length`` K
    to an input ``x`` of length N

    The filter length K is equal to the filter span in symbols (``span_in_symbols``)
    multiplied by the oversampling factor (``samples_per_symbol``).
    If this product is even, a value of one will be added.

    The filter is applied through discrete convolution.

    An optional windowing function ``window`` can be applied to the filter.

    The `dtype` of the output is `tf.float` if both ``x`` and the filter coefficients have dtype `tf.float`.
    Otherwise, the dtype of the output is `tf.complex`.

    Three padding modes are available for applying the filter:

    *   "full" (default): Returns the convolution at each point of overlap between ``x`` and the filter.
        The length of the output is N + K - 1. Zero-padding of the input ``x`` is performed to
        compute the convolution at the borders.
    *   "same": Returns an output of the same length as the input ``x``. The convolution is computed such
        that the coefficients of the input ``x`` are centered on the coefficient of the filter with index
        (K-1)/2. Zero-padding of the input signal is performed to compute the convolution at the borders.
    *   "valid": Returns the convolution only at points where ``x`` and the filter completely overlap.
        The length of the output is N - K + 1.

    Parameters
    ----------
    samples_per_symbol: `int`
        Number of samples per symbol, i.e., the oversampling factor

    coefficients: [K], `tf.float` or `tf.complex`
        Filter coefficients

    window: `None` (default) | :class:`~sionna.phy.signal.Window` | "hann" | "hamming" | "blackman"
        Window that is applied to the filter coefficients

    normalize: `bool`, (default `True`)
        If `True`, the filter is normalized to have unit power.

    precision : `None` (default) | "single" | "double"
        Precision used for internal calculations and outputs.
        If set to `None`,
        :attr:`~sionna.phy.config.Config.precision` is used.

    Input
    -----
    x : [...,N], `tf.complex` or `tf.float`
        Input to which the filter is applied along the last dimension

    padding : "full" (default) | "valid" | "same"
        Padding mode for convolving ``x`` and the filter

    conjugate : `bool`, (default `False`)
        If `True`, the complex conjugate of the filter is applied.

    Output
    ------
    y : [...,M], `tf.complex` or `tf.float`
        Filtered input. The length M depends on the ``padding``.
    """
    def __init__(self,
                 samples_per_symbol,
                 coefficients,
                 window=None,
                 normalize=True,
                 precision=None,
                 **kwargs):

        assert samples_per_symbol>0, "samples_per_symbol must be positive"

        l = coefficients.shape[-1]
        assert l%2==1, "The number of coefficients must be odd"
        span_in_symbols = l//samples_per_symbol

        super().__init__(span_in_symbols,
                         samples_per_symbol,
                         window=window,
                         normalize=normalize,
                         precision=precision,
                         **kwargs)

        self.coefficients = coefficients
        assert self.length == l, \
            f"""`coefficients` must have length {self.length}"""
