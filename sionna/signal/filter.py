#
# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Layers implementing filters"""

import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.initializers import RandomNormal
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import numpy as np
from . import convolve, Window, HannWindow, HammingWindow, BlackmanWindow, empirical_aclr


class Filter(ABC, Layer):
    # pylint: disable=line-too-long
    r"""Filter(span_in_symbols, samples_per_symbol, window=None, normalize=True, trainable=False, dtype=tf.float32, **kwargs)

    This is an abtract class for defining a filter of ``length`` K which can be
    applied to an input ``x`` of length N.

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
    span_in_symbols: int
        Filter span as measured by the number of symbols.

    samples_per_symbol: int
        Number of samples per symbol, i.e., the oversampling factor.

    window: Window or string (["hann", "hamming", "blackman"])
        Instance of :class:`~sionna.signal.Window` that is applied to the filter coefficients.
        Alternatively, a string indicating the window name can be provided. In this case,
        the chosen window will be instantiated with the default parameters. Custom windows
        must be provided as instance.

    normalize: bool
        If `True`, the filter is normalized to have unit power.
        Defaults to `True`.

    trainable: bool
        If `True`, the filter coefficients are trainable.
        Defaults to `False`.

    dtype: tf.DType
        The `dtype` of the filter coefficients.
        Defaults to `tf.float32`.

    Input
    -----
    x : [..., N], tf.complex or tf.float
        The input to which the filter is applied.
        The filter is applied along the last dimension.

    padding : string (["full", "valid", "same"])
        Padding mode for convolving ``x`` and the filter.
        Must be one of "full", "valid", or "same". Case insensitive.
        Defaults to "full".

    conjugate : bool
        If `True`, the complex conjugate of the filter is applied.
        Defaults to `False`.

    Output
    ------
    y : [...,M], tf.complex or tf.float
        Filtered input.
        It is `tf.float` only if both ``x`` and the filter are `tf.float`.
        It is `tf.complex` otherwise.
        The length M depends on the ``padding``.
    """
    def __init__(self,
                 span_in_symbols,
                 samples_per_symbol,
                 window=None,
                 normalize=True,
                 trainable=False,
                 dtype=tf.float32,
                 **kwargs):
        super().__init__(dtype=dtype, **kwargs)

        assert span_in_symbols>0, "span_in_symbols must be positive"
        self._span_in_symbols = span_in_symbols

        assert samples_per_symbol>0, "samples_per_symbol must be positive"
        self._samples_per_symbol = samples_per_symbol

        self.window = window

        assert isinstance(normalize, bool), "normalize must be bool"
        self._normalize = normalize

        assert isinstance(trainable, bool), "trainable must be bool"
        self._trainable = trainable

        assert self.length==self._coefficients_source.shape[-1], \
            "The number of coefficients must match the filter length."

        dtype = tf.as_dtype(self._dtype)
        if dtype.is_floating:
            self._coefficients = tf.Variable(self._coefficients_source,
                                                trainable=self.trainable)
        elif dtype.is_complex:
            c = self._coefficients_source
            self._coefficients = [  tf.Variable(tf.math.real(c),
                                                trainable=self.trainable),
                                    tf.Variable(tf.math.imag(c),
                                                trainable=self.trainable)]

    @property
    def length(self):
        """The filter length in samples"""
        l = self._span_in_symbols*self._samples_per_symbol
        l = 2*(l//2)+1 # Force length to be the next odd number
        return l

    @property
    def window(self):
        """The window function that is applied to the filter coefficients. `None` if no window is applied."""
        return self._window

    @window.setter
    def window(self, value):
        if isinstance(value, str):
            if value=="hann":
                self._window = HannWindow(self.length)
            elif value=="hamming":
                self._window = HammingWindow(self.length)
            elif value=="blackman":
                self._window = BlackmanWindow(self.length)
            else:
                raise AssertionError("Invalid window type")
        elif isinstance(value, Window) or value is None:
            self._window = value
        else:
            raise AssertionError("Invalid window type")

    @property
    def normalize(self):
        """`True` if the filter is normalized to have unit power. `False` otherwise."""
        return self._normalize

    @property
    def trainable(self):
        """`True` if the filter coefficients are trainable. `False` otherwise."""
        return self._trainable

    @property
    @abstractmethod
    def _coefficients_source(self):
        """Internal property that returns the (unormalized) filter coefficients.
        Concrete classes that inherits from this one must implement this
        property."""
        pass

    @property
    def coefficients(self):
        """The filter coefficients (after normalization)"""
        h = self._coefficients
        dtype = tf.as_dtype(self.dtype)

        # Combine both real dimensions to complex if necessary
        if dtype.is_complex:
            h = tf.complex(h[0], h[1])

        # Apply window
        if self.window is not None:
            h = self._window(h)

        # Ensure unit L2-norm of the coefficients
        if self.normalize:
            energy = tf.reduce_sum(tf.square(tf.abs(h)))
            h = h / tf.cast(tf.sqrt(energy), h.dtype)

        return h

    @property
    def sampling_times(self):
        """Sampling times in multiples of the symbol duration"""
        n_min = -(self.length//2)
        n_max = n_min + self.length
        t = np.arange(n_min, n_max, dtype=np.float32)
        t /= self._samples_per_symbol
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
        response: str, one of ["impulse", "magnitude"]
            The desired response type.
            Defaults to "impulse"

        scale: str, one of ["lin", "db"]
            The y-scale of the magnitude response.
            Can be "lin" (i.e., linear) or "db" (, i.e., Decibel).
            Defaults to "lin".
        """
        assert response in ["impulse", "magnitude"], "Invalid response"
        if response=="impulse":
            dtype = tf.as_dtype(self.dtype)
            plt.figure(figsize=(12,6))
            plt.plot(self.sampling_times, np.real(self.coefficients))
            if dtype.is_complex:
                plt.plot(self.sampling_times, np.imag(self.coefficients))
                plt.legend(["Real part", "Imaginary part"])
            plt.title("Impulse response")
            plt.grid()
            plt.xlabel(r"Normalized time $(t/T)$")
            plt.ylabel(r"$h(t)$")
            plt.xlim(self.sampling_times[0], self.sampling_times[-1])

        else:
            assert scale in ["lin", "db"], "Invalid scale"
            fft_size = max(1024, self.coefficients.shape[-1])
            h = np.fft.fft(self.coefficients, fft_size)
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
        """
        fft_size = 1024
        n = fft_size - tf.shape(self.coefficients)[-1]
        z = tf.zeros([n], self.coefficients.dtype)
        c = tf.cast(tf.concat([self.coefficients, z], -1), tf.complex64)
        return empirical_aclr(c, self._samples_per_symbol)

    def call(self, x, padding='full', conjugate=False):
        h = self.coefficients
        dtype = tf.as_dtype(self.dtype)
        if conjugate and dtype.is_complex:
            h = tf.math.conj(h)
        y = convolve(x,h,padding)
        return y


class RaisedCosineFilter(Filter):
    # pylint: disable=line-too-long
    r"""RaisedCosineFilter(span_in_symbols, samples_per_symbol, beta, window=None, normalize=True, trainable=False, dtype=tf.float32, **kwargs)

    Layer for applying a raised-cosine filter of ``length`` K
    to an input ``x`` of length N.

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
    span_in_symbols: int
        Filter span as measured by the number of symbols.

    samples_per_symbol: int
        Number of samples per symbol, i.e., the oversampling factor.

    beta : float
        Roll-off factor.
        Must be in the range :math:`[0,1]`.

    window: Window or string (["hann", "hamming", "blackman"])
        Instance of :class:`~sionna.signal.Window` that is applied to the filter coefficients.
        Alternatively, a string indicating the window name can be provided. In this case,
        the chosen window will be instantiated with the default parameters. Custom windows
        must be provided as instance.

    normalize: bool
        If `True`, the filter is normalized to have unit power.
        Defaults to `True`.

    trainable: bool
        If `True`, the filter coefficients are trainable variables.
        Defaults to `False`.

    dtype: tf.DType
        The `dtype` of the filter coefficients.
        Defaults to `tf.float32`.

    Input
    -----
    x : [..., N], tf.complex or tf.float
        The input to which the filter is applied.
        The filter is applied along the last dimension.

    padding : string (["full", "valid", "same"])
        Padding mode for convolving ``x`` and the filter.
        Must be one of "full", "valid", or "same".
        Defaults to "full".

    conjugate : bool
        If `True`, the complex conjugate of the filter is applied.
        Defaults to `False`.

    Output
    ------
    y : [...,M], tf.complex or tf.float
        Filtered input.
        It is `tf.float` only if both ``x`` and the filter are `tf.float`.
        It is `tf.complex` otherwise.
        The length M depends on the ``padding``.
    """
    def __init__(self,
                 span_in_symbols,
                 samples_per_symbol,
                 beta,
                 window=None,
                 normalize=True,
                 trainable=False,
                 dtype=tf.float32,
                 **kwargs):

        assert 0 <= beta <= 1, "beta must be from the intervall [0,1]"
        self._beta = beta

        super().__init__(span_in_symbols,
                         samples_per_symbol,
                         window=window,
                         normalize=normalize,
                         trainable=trainable,
                         dtype=dtype,
                         **kwargs)

    @property
    def beta(self):
        """Roll-off factor"""
        return self._beta

    @property
    def _coefficients_source(self):
        h = self._raised_cosine(self.sampling_times,
                                1.0,
                                self.beta)
        h = tf.constant(h, self.dtype)
        return h

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
    r"""RootRaisedCosineFilter(span_in_symbols, samples_per_symbol, beta, window=None, normalize=True, trainable=False, dtype=tf.float32, **kwargs)

    Layer for applying a root-raised-cosine filter of ``length`` K
    to an input ``x`` of length N.

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
    span_in_symbols: int
        Filter span as measured by the number of symbols.

    samples_per_symbol: int
        Number of samples per symbol, i.e., the oversampling factor.

    beta : float
        Roll-off factor.
        Must be in the range :math:`[0,1]`.

    window: Window or string (["hann", "hamming", "blackman"])
        Instance of :class:`~sionna.signal.Window` that is applied to the filter coefficients.
        Alternatively, a string indicating the window name can be provided. In this case,
        the chosen window will be instantiated with the default parameters. Custom windows
        must be provided as instance.

    normalize: bool
        If `True`, the filter is normalized to have unit power.
        Defaults to `True`.

    trainable: bool
        If `True`, the filter coefficients are trainable variables.
        Defaults to `False`.

    dtype: tf.DType
        The `dtype` of the filter coefficients.
        Defaults to `tf.float32`.

    Input
    -----
    x : [..., N], tf.complex or tf.float
        The input to which the filter is applied.
        The filter is applied along the last dimension.

    padding : string (["full", "valid", "same"])
        Padding mode for convolving ``x`` and the filter.
        Must be one of "full", "valid", or "same". Case insensitive.
        Defaults to "full".

    conjugate : bool
        If `True`, the complex conjugate of the filter is applied.
        Defaults to `False`.

    Output
    ------
    y : [...,M], tf.complex or tf.float
        Filtered input.
        It is `tf.float` only if both ``x`` and the filter are `tf.float`.
        It is `tf.complex` otherwise.
        The length M depends on the ``padding``.
    """
    def __init__(self,
                 span_in_symbols,
                 samples_per_symbol,
                 beta,
                 window=None,
                 normalize=True,
                 trainable=False,
                 dtype=tf.float32,
                 **kwargs):

        assert 0 <= beta <= 1, "beta must be from the intervall [0,1]"
        self._beta = beta

        super().__init__(span_in_symbols,
                         samples_per_symbol,
                         window=window,
                         normalize=normalize,
                         trainable=trainable,
                         dtype=dtype,
                         **kwargs)

    @property
    def beta(self):
        """Roll-off factor"""
        return self._beta

    @property
    def _coefficients_source(self):
        h = self._root_raised_cosine(self.sampling_times,
                                     1.0,
                                     self.beta)
        h = tf.constant(h, self.dtype)
        return h

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
    r"""SincFilter(span_in_symbols, samples_per_symbol, window=None, normalize=True, trainable=False, dtype=tf.float32, **kwargs)

    Layer for applying a sinc filter of ``length`` K
    to an input ``x`` of length N.

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
    span_in_symbols: int
        Filter span as measured by the number of symbols.

    samples_per_symbol: int
        Number of samples per symbol, i.e., the oversampling factor.

    window: Window or string (["hann", "hamming", "blackman"])
        Instance of :class:`~sionna.signal.Window` that is applied to the filter coefficients.
        Alternatively, a string indicating the window name can be provided. In this case,
        the chosen window will be instantiated with the default parameters. Custom windows
        must be provided as instance.

    normalize: bool
        If `True`, the filter is normalized to have unit power.
        Defaults to `True`.

    trainable: bool
        If `True`, the filter coefficients are trainable variables.
        Defaults to `False`.

    dtype: tf.DType
        The `dtype` of the filter coefficients.
        Defaults to `tf.float32`.

    Input
    -----
    x : [..., N], tf.complex or tf.float
        The input to which the filter is applied.
        The filter is applied along the last dimension.

    padding : string (["full", "valid", "same"])
        Padding mode for convolving ``x`` and the filter.
        Must be one of "full", "valid", or "same". Case insensitive.
        Defaults to "full".

    conjugate : bool
        If `True`, the complex conjugate of the filter is applied.
        Defaults to `False`.

    Output
    ------
    y : [...,M], tf.complex or tf.float
        Filtered input.
        It is `tf.float` only if both ``x`` and the filter are `tf.float`.
        It is `tf.complex` otherwise.
        The length M depends on the ``padding``.
    """
    def __init__(self,
                 span_in_symbols,
                 samples_per_symbol,
                 window=None,
                 normalize=True,
                 trainable=False,
                 dtype=tf.float32,
                 **kwargs):
        super().__init__(span_in_symbols,
                         samples_per_symbol,
                         window=window,
                         normalize=normalize,
                         trainable=trainable,
                         dtype=dtype,
                         **kwargs)

    @property
    def _coefficients_source(self):
        h = self._sinc(self.sampling_times,
                       1.0)
        h = tf.constant(h, self.dtype)
        return h

    def _sinc(self, t, symbol_duration):
        """Sinc filter"""
        return 1/symbol_duration*np.sinc(t/symbol_duration)


class CustomFilter(Filter):
    # pylint: disable=line-too-long
    r"""CustomFilter(span_in_symbols=None, samples_per_symbol=None, coefficients=None, window=None, normalize=True, trainable=False, dtype=tf.float32, **kwargs)

    Layer for applying a custom filter of ``length`` K
    to an input ``x`` of length N.

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
    span_in_symbols: int
        Filter span as measured by the number of symbols.
        Only needs to be provided if ``coefficients`` is None.

    samples_per_symbol: int
        Number of samples per symbol, i.e., the oversampling factor.
        Must always be provided.

    coefficients: [K], tf.float or tf.complex
        Optional filter coefficients.
        If set to `None`, then a random filter of K is generated
        by sampling a Gaussian distribution. Defaults to `None`.

    window: Window or string (["hann", "hamming", "blackman"])
        Instance of :class:`~sionna.signal.Window` that is applied to the filter coefficients.
        Alternatively, a string indicating the window name can be provided. In this case,
        the chosen window will be instantiated with the default parameters. Custom windows
        must be provided as instance.

    normalize: bool
        If `True`, the filter is normalized to have unit power.
        Defaults to `True`.

    trainable: bool
        If `True`, the filter coefficients are trainable variables.
        Defaults to `False`.

    dtype: tf.DType
        The `dtype` of the filter coefficients.
        Defaults to `tf.float32`.

    Input
    -----
    x : [..., N], tf.complex or tf.float
        The input to which the filter is applied.
        The filter is applied along the last dimension.

    padding : string (["full", "valid", "same"])
        Padding mode for convolving ``x`` and the filter.
        Must be one of "full", "valid", or "same". Case insensitive.
        Defaults to "full".

    conjugate : bool
        If `True`, the complex conjugate of the filter is applied.
        Defaults to `False`.

    Output
    ------
    y : [...,M], tf.complex or tf.float
        Filtered input.
        It is `tf.float` only if both ``x`` and the filter are `tf.float`.
        It is `tf.complex` otherwise.
        The length M depends on the ``padding``.
    """
    def __init__(self,
                 span_in_symbols=None,
                 samples_per_symbol=None,
                 coefficients=None,
                 window=None,
                 normalize=True,
                 trainable=False,
                 dtype=tf.float32,
                 **kwargs):

        assert samples_per_symbol is not None and samples_per_symbol>0, \
        "samples_per_symbol must be positive"
        self._samples_per_symbol = samples_per_symbol

        if coefficients is None:
            assert span_in_symbols is not None and span_in_symbols>0, \
                "span_in_symbols must be positive"
            self._span_in_symbols = span_in_symbols

        if coefficients is not None:
            l = coefficients.shape[-1]
            assert l%2==1, \
                "The number of coefficients must be odd"
            self._span_in_symbols = l//self._samples_per_symbol
        else:
            if dtype.is_complex:
                h = RandomNormal()([2, self.length], dtype.real_dtype)
                coefficients = tf.complex(h[0], h[1])
            else:
                coefficients = RandomNormal()([self.length], dtype)

        # Coefficients setter initialises coefficients properly
        self._h = tf.constant(coefficients, dtype)

        super().__init__(self._span_in_symbols,
                         self._samples_per_symbol,
                         window=window,
                         normalize=normalize,
                         trainable=trainable,
                         dtype=dtype,
                         **kwargs)

    @property
    def _coefficients_source(self):
        return self._h
