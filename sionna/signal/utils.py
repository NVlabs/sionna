#
# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Utility functions for the filter module"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sionna.utils.tensors import expand_to_rank
from tensorflow.experimental.numpy import swapaxes


def convolve(inp, ker, padding='full', axis=-1):
    # pylint: disable=line-too-long
    r"""
    Filters an input ``inp`` of length `N` by convolving it with a kernel ``ker`` of length `K`.

    The length of the kernel ``ker`` must not be greater than the one of the input sequence ``inp``.

    The `dtype` of the output is `tf.float` only if both ``inp`` and ``ker`` are `tf.float`. It is `tf.complex` otherwise.
    ``inp`` and ``ker`` must have the same precision.

    Three padding modes are available:

    *   "full" (default): Returns the convolution at each point of overlap between ``ker`` and ``inp``.
        The length of the output is `N + K - 1`. Zero-padding of the input ``inp`` is performed to
        compute the convolution at the border points.
    *   "same": Returns an output of the same length as the input ``inp``. The convolution is computed such
        that the coefficients of the input ``inp`` are centered on the coefficient of the kernel ``ker`` with index
        ``(K-1)/2`` for kernels of odd length, and ``K/2 - 1`` for kernels of even length.
        Zero-padding of the input signal is performed to compute the convolution at the border points.
    *   "valid": Returns the convolution only at points where ``inp`` and ``ker`` completely overlap.
        The length of the output is `N - K + 1`.

    Input
    ------
    inp : [...,N], tf.complex or tf.real
        Input to filter.

    ker : [K], tf.complex or tf.real
        Kernel of the convolution.

    padding : string
        Padding mode. Must be one of "full", "valid", or "same". Case insensitive.
        Defaults to "full".

    axis : int
        Axis along which to perform the convolution.
        Defaults to `-1`.

    Output
    -------
    out : [...,M], tf.complex or tf.float
        Convolution output.
        It is `tf.float` only if both ``inp`` and ``ker`` are `tf.float`. It is `tf.complex` otherwise.
        The length `M` of the output depends on the ``padding``.
    """

    # We don't want to be sensitive to case
    padding = padding.lower()
    assert padding in ('valid', 'same', 'full'), "Invalid padding method"

    # Ensure we process along the axis requested by the user
    inp = tf.experimental.numpy.swapaxes(inp, axis, -1)

    # Reshape the input to a 2D tensor
    batch_shape = tf.shape(inp)[:-1]
    inp_len = tf.shape(inp)[-1]
    inp_dtype = inp.dtype
    ker_dtype = ker.dtype
    inp = tf.reshape(inp, [-1, inp_len])

    # Using Tensorflow convolution implementation, we need to manually flip
    # the kernel
    ker = tf.reverse(ker, axis=(0,))
    # Tensorflow convolution expects convolution kernels with input and
    # output dims
    ker = expand_to_rank(ker, 3, 1)
    # Tensorflow convolution expects a channel dim for the convolution
    inp = tf.expand_dims(inp, axis=-1)

    # Pad the kernel or input if required depending on the convolution type.
    # Also, set the padding-mode for TF convolution
    if padding == 'valid':
        # No padding required in this case
        tf_conv_mode = 'VALID'
    elif padding == 'same':
        ker = tf.pad(ker, [[0,1],[0,0],[0,0]])
        tf_conv_mode = 'SAME'
    elif padding == 'full':
        ker_len = ker.shape[0] #tf.shape(ker)[0]
        if (ker_len % 2) == 0:
            extra_padding_left = ker_len // 2
            extra_padding_right = extra_padding_left-1
        else:
            extra_padding_left = (ker_len-1) // 2
            extra_padding_right = extra_padding_left
        inp = tf.pad(inp, [[0,0],
                        [extra_padding_left,extra_padding_right],
                        [0,0]])
        tf_conv_mode = 'SAME'

    # Extract the real and imaginary components of the input and kernel
    inp_real = tf.math.real(inp)
    ker_real = tf.math.real(ker)
    inp_imag = tf.math.imag(inp)
    ker_imag = tf.math.imag(ker)

    # Compute convolution
    # The output is complex-valued if the input or the kernel is.
    # Defaults to False, and set to True if required later
    complex_output = False
    out_1 = tf.nn.convolution(inp_real, ker_real, padding=tf_conv_mode)
    if inp_dtype.is_complex:
        out_4 = tf.nn.convolution(inp_imag, ker_real, padding=tf_conv_mode)
        complex_output = True
    else:
        out_4 = tf.zeros_like(out_1)
    if ker_dtype.is_complex:
        out_3 = tf.nn.convolution(inp_real, ker_imag, padding=tf_conv_mode)
        complex_output = True
    else:
        out_3 = tf.zeros_like(out_1)
    if inp_dtype.is_complex and ker.dtype.is_complex:
        out_2 = tf.nn.convolution(inp_imag, ker_imag, padding=tf_conv_mode)
    else:
        out_2 = tf.zeros_like(out_1)
    if complex_output:
        out = tf.complex(out_1 - out_2,
                        out_3 + out_4)
    else:
        out = out_1

    # Reshape the output to the expected shape
    out = tf.squeeze(out, axis=-1)
    out_len = tf.shape(out)[-1]
    out = tf.reshape(out, tf.concat([batch_shape, [out_len]], axis=-1))
    out = tf.experimental.numpy.swapaxes(out, axis, -1)

    return out


def fft(tensor, axis=-1):
    r"""Computes the normalized DFT along a specified axis.

    This operation computes the normalized one-dimensional discrete Fourier
    transform (DFT) along the ``axis`` dimension of a ``tensor``.
    For a vector :math:`\mathbf{x}\in\mathbb{C}^N`, the DFT
    :math:`\mathbf{X}\in\mathbb{C}^N` is computed as

    .. math::
        X_m = \frac{1}{\sqrt{N}}\sum_{n=0}^{N-1} x_n \exp \left\{
            -j2\pi\frac{mn}{N}\right\},\quad m=0,\dots,N-1.

    Input
    -----
    tensor : tf.complex
        Tensor of arbitrary shape.
    axis : int
        Indicates the dimension along which the DFT is taken.

    Output
    ------
    : tf.complex
        Tensor of the same dtype and shape as ``tensor``.
    """
    fft_size = tf.cast(tf.shape(tensor)[axis], tensor.dtype)
    scale = 1/tf.sqrt(fft_size)

    if axis not in [-1, tensor.shape.rank]:
        output =  tf.signal.fft(swapaxes(tensor, axis, -1))
        output = swapaxes(output, axis, -1)
    else:
        output = tf.signal.fft(tensor)

    return scale * output


def ifft(tensor, axis=-1):
    r"""Computes the normalized IDFT along a specified axis.

    This operation computes the normalized one-dimensional discrete inverse
    Fourier transform (IDFT) along the ``axis`` dimension of a ``tensor``.
    For a vector :math:`\mathbf{X}\in\mathbb{C}^N`, the IDFT
    :math:`\mathbf{x}\in\mathbb{C}^N` is computed as

    .. math::
        x_n = \frac{1}{\sqrt{N}}\sum_{m=0}^{N-1} X_m \exp \left\{
            j2\pi\frac{mn}{N}\right\},\quad n=0,\dots,N-1.

    Input
    -----
    tensor : tf.complex
        Tensor of arbitrary shape.

    axis : int
        Indicates the dimension along which the IDFT is taken.

    Output
    ------
    : tf.complex
        Tensor of the same dtype and shape as ``tensor``.
    """
    fft_size = tf.cast(tf.shape(tensor)[axis], tensor.dtype)
    scale = tf.sqrt(fft_size)

    if axis not in [-1, tensor.shape.rank]:
        output =  tf.signal.ifft(swapaxes(tensor, axis, -1))
        output = swapaxes(output, axis, -1)
    else:
        output = tf.signal.ifft(tensor)

    return scale * output


def empirical_psd(x, show=True, oversampling=1.0, ylim=(-30,3)):
    r"""Computes the empirical power spectral density.

    Computes the empirical power spectral density (PSD) of tensor ``x``
    along the last dimension by averaging over all other dimensions.
    Note that this function
    simply returns the averaged absolute squared discrete Fourier
    spectrum of ``x``.

    Input
    -----
    x : [...,N], tf.complex
        The signal of which to compute the PSD.

    show : bool
        Indicates if a plot of the PSD should be generated.
        Defaults to True,

    oversampling : float
        The oversampling factor. Defaults to 1.

    ylim : tuple of floats
        The limits of the y axis. Defaults to [-30, 3].
        Only relevant if ``show`` is True.

    Output
    ------
    freqs : [N], float
        The normalized frequencies at which the PSD was evaluated.

    psd : [N], float
        The PSD.
    """
    psd = tf.pow(tf.abs(fft(x)), 2)
    psd = tf.reduce_mean(psd, tf.range(0, tf.rank(psd)-1))
    psd = tf.signal.fftshift(psd)
    f_min = -0.5*oversampling
    f_max = -f_min
    freqs = tf.linspace(f_min, f_max, tf.shape(psd)[0])
    if show:
        plt.figure()
        plt.plot(freqs, 10*np.log10(psd))
        plt.title("Power Spectral Density")
        plt.xlabel("Normalized Frequency")
        plt.xlim([freqs[0], freqs[-1]])
        plt.ylabel(r"$\mathbb{E}\left[|X(f)|^2\right]$ (dB)")
        plt.ylim(ylim)
        plt.grid(True, which="both")

    return (freqs, psd)


def empirical_aclr(x, oversampling=1.0, f_min=-0.5, f_max=0.5):
    r"""Computes the empirical ACLR.

    Computes the empirical adjacent channel leakgae ration (ACLR)
    of tensor ``x`` based on its empirical power spectral density (PSD)
    which is computed along the last dimension by averaging over
    all other dimensions.

    It is assumed that the in-band ranges from [``f_min``, ``f_max``] in
    normalized frequency. The ACLR is then defined as

    .. math::

        \text{ACLR} = \frac{P_\text{out}}{P_\text{in}}

    where :math:`P_\text{in}` and :math:`P_\text{out}` are the in-band
    and out-of-band power, respectively.

    Input
    -----
    x : [...,N],  complex
        The signal for which to compute the ACLR.

    oversampling : float
        The oversampling factor. Defaults to 1.

    f_min : float
        The lower border of the in-band in normalized frequency.
        Defaults to -0.5.

    f_max : float
        The upper border of the in-band in normalized frequency.
        Defaults to 0.5.

    Output
    ------
    aclr : float
        The ACLR in linear scale.
    """
    freqs, psd = empirical_psd(x, oversampling=oversampling, show=False)
    ind_out = tf.where(tf.logical_or(tf.less(freqs, f_min),
                                     tf.greater(freqs, f_max)))
    ind_in = tf.where(tf.logical_and(tf.greater(freqs, f_min),
                                     tf.less(freqs, f_max)))
    p_out = tf.reduce_sum(tf.gather(psd, ind_out))
    p_in = tf.reduce_sum(tf.gather(psd, ind_in))
    aclr = p_out/p_in
    return aclr
