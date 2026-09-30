#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""FIR interpolation and decimation without computing discarded samples."""

from __future__ import annotations

from typing import Optional

import torch
from sionna.phy import Block
from sionna.phy.config import Precision

__all__ = ["InterpolatingFIR", "DecimatingFIR", "UpFirDn"]


def _validate_coefficients(coefficients: torch.Tensor) -> torch.Tensor:
    h = torch.as_tensor(coefficients)
    if h.ndim != 1 or h.numel() == 0:
        raise ValueError("coefficients must be a nonempty one-dimensional tensor")
    return h


def _move_axis_to_last(x: torch.Tensor, axis: int) -> tuple[torch.Tensor, int]:
    """Move the user-selected filtering axis to the last tensor dimension.

    The local convolution helpers always filter along the last dimension.  This
    helper normalizes negative axes, validates the axis value, and returns the
    normalized axis so the caller can restore the original layout afterward.
    """
    # The convolution primitives below operate on the last dimension.  Keep the
    # public API axis-aware by moving the requested dimension to the end and
    # restoring it after filtering.
    ndim = x.ndim
    if axis < 0:
        axis += ndim
    if axis < 0 or axis >= ndim:
        raise ValueError(f"axis {axis} out of range for tensor with rank {ndim}")
    return torch.swapaxes(x, axis, -1), axis


def _restore_axis_from_last(x: torch.Tensor, axis: int) -> torch.Tensor:
    """Undo :func:`_move_axis_to_last` for a previously normalized axis."""
    return torch.swapaxes(x, -1, axis)


def _as_batched_1d(x: torch.Tensor) -> tuple[torch.Tensor, torch.Size]:
    """Reshape an arbitrary-rank last-axis signal into conv1d layout.

    Returns a tensor with shape ``[flat_batch, 1, time]`` plus the original
    leading batch shape. The single channel dimension is intentional: these
    blocks implement one FIR applied independently to every leading batch item.
    """
    # torch.nn.functional.{conv1d,conv_transpose1d} expects [batch, channel,
    # time].  Collapse all leading dimensions into one batch dimension and use a
    # single channel; restore the original batch shape after filtering.
    batch_shape = x.shape[:-1]
    return x.reshape(-1, 1, x.shape[-1]), batch_shape


def _coefficients_for_input(h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Move FIR coefficients to the input device and compatible dtype.

    Real-valued filters stay real when the input is real. If either operand is
    complex, the coefficients are promoted to the corresponding complex dtype
    so the real/imaginary convolution dispatch sees consistent operands.
    """
    # Match the filter buffer to the input device and to the precision required
    # by the input/filter combination.  A real filter remains real for real
    # inputs, but any complex operand promotes the kernel to complex.
    if h.is_complex() or x.is_complex():
        dtype = torch.complex64 if x.real.dtype == torch.float32 else torch.complex128
    else:
        dtype = x.dtype
    return h.to(dtype=dtype, device=x.device)


def _conv1d_valid(x: torch.Tensor, h: torch.Tensor, *, stride: int = 1) -> torch.Tensor:
    """Valid convolution with optional stride and explicit complex support.

    ``x`` must have shape ``[batch, 1, time]`` and ``h`` must have shape
    ``[taps]``. The returned tensor has shape ``[batch, 1, output_time]``.
    PyTorch ``conv1d`` computes correlation, so this helper flips ``h`` to
    implement FIR convolution.
    """
    # PyTorch's conv1d is a real-valued correlation primitive.  We implement
    # convolution by flipping the FIR taps, and handle complex arithmetic by
    # dispatching only the real convolutions required by the input/filter dtype:
    #
    #   real x,    real h    -> 1 convolution
    #   complex x, real h    -> 2 convolutions
    #   real x,    complex h -> 2 convolutions
    #   complex x, complex h -> 4 convolutions
    #
    # For complex x=a+jb and h=c+jd, the output is
    #   (a*c - b*d) + j(a*d + b*c).
    hr = h.real if h.is_complex() else h
    hi = h.imag if h.is_complex() else None
    xr = x.real if x.is_complex() else x
    xi = x.imag if x.is_complex() else None

    wr = torch.flip(hr, dims=[0]).reshape(1, 1, -1)
    yr = torch.nn.functional.conv1d(xr, wr, stride=stride)
    yi = None
    if xi is not None:
        yi = torch.nn.functional.conv1d(xi, wr, stride=stride)
    if hi is not None:
        wi = torch.flip(hi, dims=[0]).reshape(1, 1, -1)
        yri = torch.nn.functional.conv1d(xr, wi, stride=stride)
        if xi is not None:
            yii = torch.nn.functional.conv1d(xi, wi, stride=stride)
            real = yr - yii
            imag = yri + yi
        else:
            real = yr
            imag = yri
        return torch.complex(real, imag)
    if yi is not None:
        return torch.complex(yr, yi)
    return yr


def _conv_transpose1d(x: torch.Tensor, h: torch.Tensor, *, stride: int) -> torch.Tensor:
    """Transposed-convolution FIR interpolation with explicit complex support.

    ``x`` must have shape ``[batch, 1, time]`` and ``h`` must have shape
    ``[taps]``. With stride ``L``, this computes the nonzero work of
    zero-stuff-by-``L`` followed by full FIR convolution.
    """
    # A transposed convolution with stride=L is exactly FIR interpolation by L:
    # each input symbol places a shifted copy of h into the high-rate output,
    # y[n*L + k] += x[n] h[k]. This avoids materializing the zero-stuffed
    # sequence and avoids multiplying filter taps by inserted zeros.
    #
    # conv_transpose1d is also real-valued, so use the same minimal
    # real/complex dispatch as _conv1d_valid().
    hr = h.real if h.is_complex() else h
    hi = h.imag if h.is_complex() else None
    xr = x.real if x.is_complex() else x
    xi = x.imag if x.is_complex() else None

    wr = hr.reshape(1, 1, -1)
    yr = torch.nn.functional.conv_transpose1d(xr, wr, stride=stride)
    yi = None
    if xi is not None:
        yi = torch.nn.functional.conv_transpose1d(xi, wr, stride=stride)
    if hi is not None:
        wi = hi.reshape(1, 1, -1)
        yri = torch.nn.functional.conv_transpose1d(xr, wi, stride=stride)
        if xi is not None:
            yii = torch.nn.functional.conv_transpose1d(xi, wi, stride=stride)
            real = yr - yii
            imag = yri + yi
        else:
            real = yr
            imag = yri
        return torch.complex(real, imag)
    if yi is not None:
        return torch.complex(yr, yi)
    return yr


class InterpolatingFIR(Block):
    """FIR interpolation without explicitly inserting zero samples.

    The input is filtered along ``axis``. For an input length ``N``, FIR length
    ``K``, and interpolation factor ``L``, the output length is
    ``(N-1)*L + K``. This is the useful nonzero part of explicit upsampling
    followed by full convolution.
    """

    def __init__(
        self,
        samples_per_symbol: int,
        coefficients: torch.Tensor,
        *,
        axis: int = -1,
        precision: Optional[Precision] = None,
        device: Optional[str] = None,
    ) -> None:
        """Create an interpolating FIR.

        :param samples_per_symbol: Interpolation factor ``L``.
        :param coefficients: FIR taps ordered in ordinary convolution order.
        :param axis: Tensor dimension to filter and interpolate.
        """
        super().__init__(precision=precision, device=device)
        if samples_per_symbol <= 0:
            raise ValueError("samples_per_symbol must be positive")
        self.samples_per_symbol = int(samples_per_symbol)
        self.axis = int(axis)
        self.register_buffer(
            "coefficients",
            self._convert(_validate_coefficients(coefficients).detach().clone()),
        )

    @property
    def length(self) -> int:
        """Number of FIR taps."""
        return int(self.coefficients.shape[-1])

    def call(self, x: torch.Tensor) -> torch.Tensor:
        """Apply FIR interpolation to ``x`` along the configured axis."""
        x, axis = _move_axis_to_last(x, self.axis)
        x3, batch_shape = _as_batched_1d(x)
        h = _coefficients_for_input(self.coefficients, x)
        # Output length is (N-1)*L + K. Compared to explicit upsampling to N*L
        # followed by full convolution, this omits the final L-1 samples that
        # can only come from trailing inserted zeros. Those samples are exactly
        # zero for the explicit path and are not useful in rate-changing chains.
        y = _conv_transpose1d(x3, h, stride=self.samples_per_symbol)
        y = y.reshape(*batch_shape, y.shape[-1])
        return _restore_axis_from_last(y, axis)


class DecimatingFIR(Block):
    """Full FIR filtering followed by decimation, computed only at kept phases.

    This block is equivalent to full FIR convolution followed by
    ``Downsampling(samples_per_symbol, offset, num_symbols)``. It avoids
    producing the discarded phases by using strided convolution.
    """

    def __init__(
        self,
        samples_per_symbol: int,
        coefficients: torch.Tensor,
        *,
        offset: int = 0,
        num_symbols: Optional[int] = None,
        axis: int = -1,
        precision: Optional[Precision] = None,
        device: Optional[str] = None,
    ) -> None:
        """Create a decimating FIR.

        :param samples_per_symbol: Decimation factor ``L``.
        :param coefficients: FIR taps ordered in ordinary convolution order.
        :param offset: First full-convolution output sample to retain.
        :param num_symbols: Optional maximum number of output samples.
        :param axis: Tensor dimension to filter and decimate.
        """
        super().__init__(precision=precision, device=device)
        if samples_per_symbol <= 0:
            raise ValueError("samples_per_symbol must be positive")
        if offset < 0:
            raise ValueError("offset must be non-negative")
        if num_symbols is not None and num_symbols < 0:
            raise ValueError("num_symbols must be non-negative")
        self.samples_per_symbol = int(samples_per_symbol)
        self.offset = int(offset)
        self.num_symbols = None if num_symbols is None else int(num_symbols)
        self.axis = int(axis)
        self.register_buffer(
            "coefficients",
            self._convert(_validate_coefficients(coefficients).detach().clone()),
        )

    @property
    def length(self) -> int:
        """Number of FIR taps."""
        return int(self.coefficients.shape[-1])

    def call(self, x: torch.Tensor) -> torch.Tensor:
        """Apply full-convolution FIR filtering and retain one phase."""
        x, axis = _move_axis_to_last(x, self.axis)
        x3, batch_shape = _as_batched_1d(x)
        h = _coefficients_for_input(self.coefficients, x)
        pad = self.length - 1
        # Full convolution can be written as valid convolution after padding the
        # input with K-1 zeros on both sides.  The downsampling phase is just an
        # offset into this full-convolution output.  By slicing the padded input
        # before the strided convolution, conv1d computes only samples
        # offset, offset+L, offset+2L, ... rather than computing every output
        # sample and discarding most of them.
        xr = torch.nn.functional.pad(x3.real if x3.is_complex() else x3, (pad, pad))
        if x3.is_complex():
            xi = torch.nn.functional.pad(x3.imag, (pad, pad))
            padded = torch.complex(xr, xi)
        else:
            padded = xr
        if self.offset:
            padded = padded[..., self.offset :]
        if self.offset >= x.shape[-1] + self.length - 1 or self.num_symbols == 0:
            y = x3[..., :0]
            if h.is_complex() and not y.is_complex():
                y = y.to(h.dtype)
        else:
            y = _conv1d_valid(padded, h, stride=self.samples_per_symbol)
        if self.num_symbols is not None:
            y = y[..., : self.num_symbols]
        y = y.reshape(*batch_shape, y.shape[-1])
        return _restore_axis_from_last(y, axis)


class UpFirDn(Block):
    """Convenience up/filter/down block for explicit-reference equivalence.

    This composes :class:`InterpolatingFIR` with a phase-selecting decimator,
    matching ``Upsampling -> FIR full convolution -> Downsampling`` for the
    same FIR taps. The common pure-interpolation and pure-decimation cases use
    the efficient specialized kernels. The general rational case is correct for
    this reference ordering, but it is not a fully optimized polyphase
    arbitrary-rational resampler.
    """

    def __init__(
        self,
        coefficients: torch.Tensor,
        *,
        up: int = 1,
        down: int = 1,
        offset: int = 0,
        num_symbols: Optional[int] = None,
        axis: int = -1,
        precision: Optional[Precision] = None,
        device: Optional[str] = None,
    ) -> None:
        """Create a convenience up/filter/down block.

        :param coefficients: FIR taps applied immediately after interpolation.
        :param up: Integer interpolation factor.
        :param down: Integer decimation factor.
        :param offset: Decimation phase after interpolation.
        :param num_symbols: Optional maximum number of output samples.
        :param axis: Tensor dimension to resample.
        """
        super().__init__(precision=precision, device=device)
        if up <= 0 or down <= 0:
            raise ValueError("up and down must be positive")
        if offset < 0:
            raise ValueError("offset must be non-negative")
        if num_symbols is not None and num_symbols < 0:
            raise ValueError("num_symbols must be non-negative")
        self.up = int(up)
        self.down = int(down)
        self.offset = int(offset)
        self.num_symbols = None if num_symbols is None else int(num_symbols)
        self.axis = int(axis)
        self.interpolator = InterpolatingFIR(
            self.up,
            coefficients,
            axis=axis,
            precision=self.precision,
            device=self.device,
        )
        self.decimator = DecimatingFIR(
            self.down,
            torch.tensor([1.0], dtype=torch.as_tensor(coefficients).real.dtype),
            offset=offset,
            num_symbols=num_symbols,
            axis=axis,
            precision=self.precision,
            device=self.device,
        )

    def call(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the configured interpolation and optional decimation."""
        # This wrapper is intentionally simple. It filters during interpolation
        # and then keeps one phase of that filtered high-rate sequence. That
        # matches the explicit reference chain but is not the minimum-arithmetic
        # implementation for all rational up/down combinations.
        y = self.interpolator(x)
        # Explicit upsampling includes up-1 trailing zeros. They matter when
        # the selected decimation phase reaches the end of the full output.
        if self.up > 1:
            y, axis = _move_axis_to_last(y, self.axis)
            y = torch.nn.functional.pad(y, (0, self.up - 1))
            y = _restore_axis_from_last(y, axis)
        return self.decimator(y)
