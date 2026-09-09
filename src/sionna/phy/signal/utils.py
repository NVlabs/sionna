#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Utility functions for the signal module"""

from typing import Literal, Optional, Tuple

import math
import matplotlib.pyplot as plt
import numpy as np
import torch

from sionna.phy import config, dtypes
from sionna.phy.config import Precision
from sionna.phy.utils import expand_to_rank

__all__ = ["convolve", "fft", "ifft", "empirical_psd", "empirical_aclr"]


def convolve(
    inp: torch.Tensor,
    ker: torch.Tensor,
    padding: Literal["full", "same", "valid"] = "full",
    axis: int = -1,
    precision: Optional[Precision] = None,
) -> torch.Tensor:
    r"""Filters an input ``inp`` of length `N` by convolving it with a kernel ``ker`` of length `K`.

    The length of the kernel ``ker`` must not be greater than the one of the input sequence ``inp``.

    The dtype of the output is `torch.float` only if both ``inp`` and ``ker`` are `torch.float`.
    It is `torch.complex` otherwise. ``inp`` and ``ker`` must have the same precision.

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

    :param inp: Input to filter with shape [..., N] (`torch.complex` or `torch.float`)
    :param ker: Kernel of the convolution with shape [K] (`torch.complex` or `torch.float`)
    :param padding: Padding mode. One of "full" (default), "valid", or "same".
    :param axis: Axis along which to perform the convolution
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`, :attr:`~sionna.phy.config.Config.precision` is used.

    :output out: [..., M], `torch.complex` or `torch.float`.
        Convolution output.
        The length `M` of the output depends on the ``padding``.

    .. rubric:: Examples

    .. code-block:: python

        import torch
        from sionna.phy.signal import convolve

        inp = torch.randn(64, 100)
        ker = torch.randn(10)
        out = convolve(inp, ker, padding="same")
        print(out.shape)
        # torch.Size([64, 100])
    """
    # We don't want to be sensitive to case
    padding = padding.lower()
    if padding not in ("valid", "same", "full"):
        raise ValueError(
            "padding must be one of 'valid', 'same' or 'full', "
            f"got {padding!r}"
        )

    # Ensure we process along the axis requested by the user
    inp = torch.swapaxes(inp, axis, -1)

    # Cast inputs
    if precision is None:
        rdtype = config.dtype
        cdtype = config.cdtype
    else:
        rdtype = dtypes[precision]["torch"]["dtype"]
        cdtype = dtypes[precision]["torch"]["cdtype"]

    if inp.is_complex():
        inp = inp.to(cdtype)
    elif inp.is_floating_point():
        inp = inp.to(rdtype)
    if ker.is_complex():
        ker = ker.to(cdtype)
    elif ker.is_floating_point():
        ker = ker.to(rdtype)

    # Reshape the input to a 2D tensor
    batch_shape = inp.shape[:-1]
    inp_len = inp.shape[-1]
    inp_dtype = inp.dtype
    ker_dtype = ker.dtype
    inp = inp.reshape(-1, inp_len)

    # Flip the kernel for convolution (conv1d does correlation)
    ker = torch.flip(ker, dims=(0,))
    # conv1d expects kernel shape: [out_channels, in_channels, kernel_size]
    ker = ker.unsqueeze(0).unsqueeze(0)
    # Add channel dim to input: [batch, 1, length]
    inp = inp.unsqueeze(1)

    # Determine padding amounts
    # We always compute full convolution and then extract the relevant portion
    ker_len = ker.shape[-1]
    pad_left = ker_len - 1
    pad_right = ker_len - 1

    # Extract the real and imaginary components of the input and kernel
    inp_real = inp.real if inp.is_complex() else inp
    ker_real = ker.real if ker.is_complex() else ker
    inp_imag = inp.imag if inp.is_complex() else torch.zeros_like(inp_real)
    ker_imag = ker.imag if ker.is_complex() else torch.zeros_like(ker_real)

    # Pad inputs for full convolution
    inp_real = torch.nn.functional.pad(inp_real, (pad_left, pad_right))
    inp_imag = torch.nn.functional.pad(inp_imag, (pad_left, pad_right))

    # Compute convolution
    # The output is complex-valued if the input or the kernel is
    complex_output = inp_dtype.is_complex or ker_dtype.is_complex

    out_1 = torch.nn.functional.conv1d(inp_real, ker_real)
    if inp_dtype.is_complex:
        out_4 = torch.nn.functional.conv1d(inp_imag, ker_real)
    else:
        out_4 = torch.zeros_like(out_1)
    if ker_dtype.is_complex:
        out_3 = torch.nn.functional.conv1d(inp_real, ker_imag)
    else:
        out_3 = torch.zeros_like(out_1)
    if inp_dtype.is_complex and ker_dtype.is_complex:
        out_2 = torch.nn.functional.conv1d(inp_imag, ker_imag)
    else:
        out_2 = torch.zeros_like(out_1)

    if complex_output:
        out = torch.complex(out_1 - out_2, out_3 + out_4)
    else:
        out = out_1

    # Extract the relevant portion based on padding mode
    # Full convolution length is inp_len + ker_len - 1
    if padding == "valid":
        # Valid: only where input and kernel fully overlap
        # Length is inp_len - ker_len + 1
        start = ker_len - 1
        end = start + inp_len - ker_len + 1
        out = out[..., start:end]
    elif padding == "same":
        # Same: output has same length as input
        # Center the output on the input
        start = (ker_len - 1) // 2
        end = start + inp_len
        out = out[..., start:end]
    # else: 'full' - keep the entire output

    # Reshape the output to the expected shape
    out = out.squeeze(1)  # Remove channel dim
    out_len = out.shape[-1]
    out = out.reshape(*batch_shape, out_len)
    out = torch.swapaxes(out, axis, -1)

    return out


def fft(
    tensor: torch.Tensor,
    axis: int = -1,
    precision: Optional[Precision] = None,
) -> torch.Tensor:
    r"""Computes the normalized DFT along a specified axis.

    This operation computes the normalized one-dimensional discrete Fourier
    transform (DFT) along the ``axis`` dimension of a ``tensor``.
    For a vector :math:`\mathbf{x}\in\mathbb{C}^N`, the DFT
    :math:`\mathbf{X}\in\mathbb{C}^N` is computed as

    .. math::
        X_m = \frac{1}{\sqrt{N}}\sum_{n=0}^{N-1} x_n \exp \left\{
            -j2\pi\frac{mn}{N}\right\},\quad m=0,\dots,N-1.

    :param tensor: Tensor of arbitrary shape (`torch.complex`)
    :param axis: Dimension along which the DFT is taken
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`, :attr:`~sionna.phy.config.Config.precision` is used.

    :output x_f: `torch.complex`.
        Tensor of the same shape as ``tensor``.

    .. rubric:: Examples

    .. code-block:: python

        import torch
        from sionna.phy.signal import fft

        x = torch.randn(32, 64, dtype=torch.complex64)
        X = fft(x)
        print(X.shape)
        # torch.Size([32, 64])
    """
    if precision is None:
        cdtype = config.cdtype
    else:
        cdtype = dtypes[precision]["torch"]["cdtype"]

    fft_size = tensor.shape[axis]
    scale = 1.0 / (fft_size**0.5)
    tensor = tensor.to(cdtype)

    output = torch.fft.fft(tensor, dim=axis)
    return scale * output


def ifft(
    tensor: torch.Tensor,
    axis: int = -1,
    precision: Optional[Precision] = None,
) -> torch.Tensor:
    r"""Computes the normalized IDFT along a specified axis.

    This operation computes the normalized one-dimensional discrete inverse
    Fourier transform (IDFT) along the ``axis`` dimension of a ``tensor``.
    For a vector :math:`\mathbf{X}\in\mathbb{C}^N`, the IDFT
    :math:`\mathbf{x}\in\mathbb{C}^N` is computed as

    .. math::
        x_n = \frac{1}{\sqrt{N}}\sum_{m=0}^{N-1} X_m \exp \left\{
            j2\pi\frac{mn}{N}\right\},\quad n=0,\dots,N-1.

    :param tensor: Tensor of arbitrary shape (`torch.complex`)
    :param axis: Dimension along which the IDFT is taken
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`, :attr:`~sionna.phy.config.Config.precision` is used.

    :output x: `torch.complex`.
        Tensor of the same shape as ``tensor``.

    .. rubric:: Examples

    .. code-block:: python

        import torch
        from sionna.phy.signal import ifft

        X = torch.randn(32, 64, dtype=torch.complex64)
        x = ifft(X)
        print(x.shape)
        # torch.Size([32, 64])
    """
    if precision is None:
        cdtype = config.cdtype
    else:
        cdtype = dtypes[precision]["torch"]["cdtype"]

    fft_size = tensor.shape[axis]
    scale = fft_size**0.5
    tensor = tensor.to(cdtype)

    output = torch.fft.ifft(tensor, dim=axis)
    return scale * output


def empirical_psd(
    x: torch.Tensor,
    show: bool = True,
    oversampling: float = 1.0,
    ylim: Tuple[float, float] = (-30, 3),
    precision: Optional[Precision] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Computes the empirical power spectral density.

    Computes the empirical power spectral density (PSD) of tensor ``x``
    along the last dimension by averaging over all other dimensions.
    Note that this function simply returns the averaged absolute squared
    discrete Fourier spectrum of ``x``.

    :param x: Signal of which to compute the PSD with shape [..., N] (`torch.complex`)
    :param show: Indicates if a plot of the PSD should be generated
    :param oversampling: Oversampling factor
    :param ylim: Limits of the y axis. Only relevant if ``show`` is `True`.
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`, :attr:`~sionna.phy.config.Config.precision` is used.

    :output freqs: [N], `torch.float`.
        Normalized frequencies.

    :output psd: [N], `torch.float`.
        PSD.

    .. rubric:: Examples

    .. code-block:: python

        import torch
        from sionna.phy.signal import empirical_psd

        x = torch.randn(100, 256, dtype=torch.complex64)
        freqs, psd = empirical_psd(x, show=False)
        print(freqs.shape, psd.shape)
        # torch.Size([256]) torch.Size([256])
    """
    if precision is None:
        rdtype = config.dtype
        cdtype = config.cdtype
    else:
        rdtype = dtypes[precision]["torch"]["dtype"]
        cdtype = dtypes[precision]["torch"]["cdtype"]

    x = x.to(dtype=cdtype)
    psd = torch.abs(fft(x, precision=precision)) ** 2
    # Average over all dimensions except the last (if there are multiple dimensions)
    if psd.dim() > 1:
        psd = psd.mean(dim=tuple(range(psd.dim() - 1)))
    psd = torch.fft.fftshift(psd, dim=-1)

    f_min = -0.5 * oversampling
    f_max = -f_min
    freqs = torch.linspace(f_min, f_max, psd.shape[0], dtype=rdtype, device=x.device)

    if show:
        plt.figure()
        psd_np = psd.detach().cpu().numpy()
        freqs_np = freqs.detach().cpu().numpy()
        plt.plot(freqs_np, 10 * np.log10(psd_np))
        plt.title("Power Spectral Density")
        plt.xlabel("Normalized Frequency")
        plt.xlim([freqs_np[0], freqs_np[-1]])
        plt.ylabel(r"$\mathbb{E}\left[|X(f)|^2\right]$ (dB)")
        plt.ylim(ylim)
        plt.grid(True, which="both")

    return freqs, psd


def empirical_aclr(
    x: torch.Tensor,
    oversampling: float = 1.0,
    f_min: float = -0.5,
    f_max: float = 0.5,
    precision: Optional[Precision] = None,
) -> torch.Tensor:
    r"""Computes the empirical ACLR.

    Computes the empirical adjacent channel leakage ratio (ACLR)
    of tensor ``x`` based on its empirical power spectral density (PSD)
    which is computed along the last dimension by averaging over
    all other dimensions.

    It is assumed that the in-band ranges from [``f_min``, ``f_max``] in
    normalized frequency. The ACLR is then defined as

    .. math::

        \text{ACLR} = \frac{P_\text{out}}{P_\text{in}}

    where :math:`P_\text{in}` and :math:`P_\text{out}` are the in-band
    and out-of-band power, respectively.

    :param x: Signal for which to compute the ACLR with shape [..., N] (`torch.complex`)
    :param oversampling: Oversampling factor
    :param f_min: Lower border of the in-band in normalized frequency
    :param f_max: Upper border of the in-band in normalized frequency
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`, :attr:`~sionna.phy.config.Config.precision` is used.

    :output aclr: `float`.
        ACLR in linear scale.

    .. rubric:: Examples

    .. code-block:: python

        import torch
        from sionna.phy.signal import empirical_aclr

        x = torch.randn(100, 256, dtype=torch.complex64)
        aclr = empirical_aclr(x, oversampling=2.0)
        print(aclr.shape)
        # torch.Size([])
    """
    if oversampling <= 0:
        raise ValueError(f"oversampling must be positive, got {oversampling}")

    if f_min >= f_max:
        raise ValueError(
            f"f_min ({f_min}) must be smaller than f_max ({f_max})"
        )

    # The spectrum returned by `empirical_psd()` spans this range.
    spectrum_min = -0.5 * oversampling
    spectrum_max = -spectrum_min
    if f_min >= spectrum_max or f_max <= spectrum_min:
        raise ValueError(
            f"The in-band ({f_min}, {f_max}) lies outside the spectrum "
            f"[{spectrum_min}, {spectrum_max}] produced with "
            f"oversampling={oversampling}"
        )

    # Determine from the scalar grid parameters whether the open interval
    # (f_min, f_max) contains a frequency bin. Doing this before creating the
    # tensor mask avoids synchronizing a CUDA result back to Python and keeps
    # the valid path compatible with `torch.compile(fullgraph=True)`.
    num_bins = x.shape[-1]
    if num_bins == 1:
        has_in_band_bin = f_min < spectrum_min < f_max
    else:
        if f_min < spectrum_min:
            first_in_band_bin = 0
        else:
            frequency_spacing = oversampling / (num_bins - 1)
            first_in_band_bin = (
                math.floor((f_min - spectrum_min) / frequency_spacing) + 1
            )
        has_in_band_bin = (
            first_in_band_bin < num_bins
            and spectrum_min
            + first_in_band_bin * oversampling / (num_bins - 1)
            < f_max
        )

    if not has_in_band_bin:
        raise ValueError(
            f"The in-band ({f_min}, {f_max}) contains no frequency bin of "
            f"the {num_bins}-point spectrum; it is narrower than the "
            "resolution of the frequency grid"
        )

    freqs, psd = empirical_psd(
        x, oversampling=oversampling, precision=precision, show=False
    )
    ind_out = (freqs < f_min) | (freqs > f_max)
    ind_in = (freqs > f_min) & (freqs < f_max)

    p_out = psd[ind_out].sum()
    p_in = psd[ind_in].sum()
    aclr = p_out / p_in
    return aclr
