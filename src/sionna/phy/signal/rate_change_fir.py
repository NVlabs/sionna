#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""FIR interpolation and decimation without computing discarded samples."""

from __future__ import annotations

from math import prod
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
    return x.reshape(prod(batch_shape), 1, x.shape[-1]), batch_shape


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

    Each call emits ``N*L`` samples for a chunk of ``N`` input symbols and
    retains the FIR overlap for the next call. Concatenating chunk outputs
    therefore matches one call on the concatenated input. :meth:`flush`
    returns the final ``K-1`` samples and resets the block. The combined
    output matches explicit upsampling followed by full FIR convolution.
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
        self.register_buffer("_overlap", None, persistent=False)

    @property
    def length(self) -> int:
        """Number of FIR taps."""
        return int(self.coefficients.shape[-1])

    def call(self, x: torch.Tensor) -> torch.Tensor:
        """Process the next input chunk along the configured axis."""
        x, axis = _move_axis_to_last(x, self.axis)
        batch_shape = x.shape[:-1]
        h = _coefficients_for_input(self.coefficients, x)
        if x.shape[-1] == 0:
            dtype = h.dtype if self._overlap is None else self._overlap.dtype
            return _restore_axis_from_last(
                x.new_empty((*batch_shape, 0), dtype=dtype), axis
            )

        if self._overlap is not None and (batch_shape != self._overlap.shape[:-1]):
            raise ValueError("chunk batch shape must remain fixed until flush()")

        x3, _ = _as_batched_1d(x)
        part = _conv_transpose1d(x3, h, stride=self.samples_per_symbol)
        part = part.reshape(*batch_shape, part.shape[-1])
        # A transposed convolution omits the L-1 trailing inserted zeros.
        # Restore them to produce exactly N*L stable samples on each call.
        part = torch.nn.functional.pad(part, (0, self.samples_per_symbol - 1))
        # Add the K-1 output samples retained from the previous chunk.
        if self._overlap is not None:
            part = part + torch.nn.functional.pad(
                self._overlap, (0, part.shape[-1] - self._overlap.shape[-1])
            )
        split = x.shape[-1] * self.samples_per_symbol
        y = part[..., :split]
        self._overlap = part[..., split:].clone()
        return _restore_axis_from_last(y, axis)

    def flush(self) -> torch.Tensor:
        """Return the remaining FIR output and reset the stream state."""
        if self._overlap is None:
            dtype = self.cdtype if self.coefficients.is_complex() else self.dtype
            return torch.empty(0, dtype=dtype, device=self.device)
        y = _restore_axis_from_last(self._overlap, self.axis)
        self._overlap = None
        return y


class DecimatingFIR(Block):
    """Full FIR filtering followed by decimation, computed only at kept phases.

    This block is equivalent to full FIR convolution followed by
    ``Downsampling(samples_per_symbol, offset, num_symbols)``. It avoids
    producing the discarded phases by using strided convolution. Calls on
    consecutive chunks share FIR history and decimation phase. :meth:`flush`
    returns the remaining samples and resets the block.
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
        self.register_buffer("_history", None, persistent=False)
        self._skip = self.offset
        self._remaining = self.num_symbols

    @property
    def length(self) -> int:
        """Number of FIR taps."""
        return int(self.coefficients.shape[-1])

    def call(self, x: torch.Tensor) -> torch.Tensor:
        """Process the next input chunk and retain the configured phase."""
        x, axis = _move_axis_to_last(x, self.axis)
        batch_shape = x.shape[:-1]
        h = _coefficients_for_input(self.coefficients, x)
        if self._history is None:
            self._history = x.new_zeros((*batch_shape, self.length - 1))
        elif batch_shape != self._history.shape[:-1]:
            raise ValueError("chunk batch shape must remain fixed until flush()")

        joined = torch.cat((self._history, x), dim=-1)
        # _skip is the next retained sample's position within this chunk.
        if self._skip >= x.shape[-1] or self._remaining == 0:
            y = x[..., :0].to(h.dtype)
        else:
            windows, _ = _as_batched_1d(joined[..., self._skip :])
            y = _conv1d_valid(windows, h, stride=self.samples_per_symbol)
            y = y.reshape(*batch_shape, y.shape[-1])
            if self._remaining is not None:
                y = y[..., : self._remaining]
                self._remaining -= y.shape[-1]

        self._skip -= x.shape[-1]
        if self._skip < 0:
            self._skip %= self.samples_per_symbol
        # Copy only the history so a short state does not retain a whole chunk.
        self._history = joined[..., x.shape[-1] :].clone()
        return _restore_axis_from_last(y, axis)

    def flush(self) -> torch.Tensor:
        """Emit the full-convolution tail and reset FIR and phase state."""
        if self._history is None:
            dtype = self.cdtype if self.coefficients.is_complex() else self.dtype
            return torch.empty(0, dtype=dtype, device=self.device)
        tail = torch.zeros_like(self._history)
        y = self.call(_restore_axis_from_last(tail, self.axis))
        self._history = None
        self._skip = self.offset
        self._remaining = self.num_symbols
        return y


class UpFirDn(Block):
    """Convenience up/filter/down block for explicit-reference equivalence.

    This composes :class:`InterpolatingFIR` with a phase-selecting decimator,
    matching ``Upsampling -> FIR full convolution -> Downsampling`` for the
    same FIR taps. The general rational case is correct for this reference
    ordering, but it is not a fully optimized polyphase arbitrary-rational
    resampler. Calls on consecutive chunks share state;
    :meth:`flush` emits the remaining samples and resets both stages.
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
        """Process the next input chunk through both stages."""
        y = self.interpolator(x)
        return self.decimator(y)

    def flush(self) -> torch.Tensor:
        """Emit the remaining samples and reset both stages."""
        if self.interpolator._overlap is None:
            return self.decimator.flush()
        y = self.interpolator.flush()
        final = self.decimator(y)
        tail = self.decimator.flush()
        return torch.cat((final, tail), dim=self.axis)
