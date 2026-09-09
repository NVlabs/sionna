#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
from typing import Optional, Union

import numpy as np
import torch

from sionna._validation import check_tensor_all
from sionna.phy import Block
from sionna.phy.config import Precision
from sionna.phy.signal import ifft
from sionna.phy.utils import flatten_last_dims

__all__ = ["OFDMModulator"]


class OFDMModulator(Block):
    r"""Computes the time-domain representation of an OFDM resource grid
    with (optional) cyclic prefix.

    :param cyclic_prefix_length: Integer or vector of integers indicating the
        length of the cyclic prefix that is prepended to each OFDM symbol.
        None of its elements can be larger than the FFT size. Defaults to `0`.
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`, :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for tensor operations. If `None`,
        :attr:`~sionna.phy.config.Config.device` is used.

    :input inputs: [..., num_ofdm_symbols, fft_size], `torch.complex`.
        Resource grid in the frequency domain.

    :output x_time: [..., num_ofdm_symbols*(fft_size+cyclic_prefix_length)] or [..., num_ofdm_symbols*fft_size+sum(cyclic_prefix_length)], `torch.complex`.
        Time-domain OFDM signal.

    .. rubric:: Examples

    .. code-block:: python

        import torch
        from sionna.phy.ofdm import OFDMModulator

        modulator = OFDMModulator(cyclic_prefix_length=16)
        # Resource grid: [batch, num_ofdm_symbols, fft_size]
        x_freq = torch.randn(64, 14, 72, dtype=torch.complex64)
        x_time = modulator(x_freq)
        print(x_time.shape)
        # torch.Size([64, 1232])  # 14 * (72 + 16) = 1232
    """

    def __init__(
        self,
        cyclic_prefix_length: Union[int, np.ndarray, torch.Tensor] = 0,
        precision: Optional[Precision] = None,
        device: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(precision=precision, device=device, **kwargs)
        self._cp_length_scalar: Optional[int] = None  # Cached scalar for call()
        # Register tensors as buffers for CUDA graph compatibility
        self.register_buffer("_cyclic_prefix_length", None)
        self.register_buffer("_ind", None)
        self.cyclic_prefix_length = cyclic_prefix_length

    @property
    def cyclic_prefix_length(self) -> torch.Tensor:
        """Get/set the cyclic prefix length (scalar or per-symbol)"""
        return self._cyclic_prefix_length

    @cyclic_prefix_length.setter
    def cyclic_prefix_length(self, value: Union[int, np.ndarray, torch.Tensor]) -> None:
        if isinstance(value, (int, float)):
            value = torch.tensor([value], dtype=torch.int32, device=self.device)
        elif isinstance(value, np.ndarray):
            value = torch.tensor(value, dtype=torch.int32, device=self.device)
        else:
            value = value.to(dtype=torch.int32, device=self.device)

        check_tensor_all(
            value >= 0,
            name="cyclic_prefix_length",
            message="`cyclic_prefix_length` must be nonnegative.",
        )
        if not 0 <= value.dim() <= 1:
            raise ValueError("`cyclic_prefix_length` must be of rank 0 or 1.")

        # Store as 0D if scalar, 1D otherwise
        if value.numel() == 1:
            # Register as buffer for CUDA graph compatibility
            self.register_buffer("_cyclic_prefix_length", value.squeeze())
            # Cache scalar value to avoid .item() during tracing
            self._cp_length_scalar = int(value.item())
        else:
            self.register_buffer("_cyclic_prefix_length", value)
            self._cp_length_scalar = None

    def build(self, input_shape: tuple) -> None:
        """Build the modulator based on input shape.

        :param input_shape: Shape of the input tensor
            `[..., num_ofdm_symbols, fft_size]`
        """
        num_ofdm_symbols, fft_size = input_shape[-2:]
        cp_len = self._cyclic_prefix_length

        check_tensor_all(
            cp_len <= fft_size,
            name="cyclic_prefix_length",
            message="`cyclic_prefix_length` cannot be larger than `fft_size`.",
        )

        if cp_len.dim() == 1:
            if cp_len.shape[0] != num_ofdm_symbols:
                raise ValueError(
                    "`cyclic_prefix_length` must be of size [num_ofdm_symbols]"
                )

            # Compute indices of CP symbols and data symbols
            # Build the gather indices for variable CP lengths
            # Convert to list once to avoid .item() calls during tracing
            cp_lengths = cp_len.tolist()
            indices_list = []
            offset = 0
            for i in range(num_ofdm_symbols):
                cp_length_i = cp_lengths[i]
                # CP indices (last cp_length_i samples of the OFDM symbol)
                cp_start = (i + 1) * fft_size - cp_length_i
                cp_indices = torch.arange(
                    cp_start, (i + 1) * fft_size, dtype=torch.int64, device=self.device
                )
                # Data indices
                data_indices = torch.arange(
                    i * fft_size,
                    (i + 1) * fft_size,
                    dtype=torch.int64,
                    device=self.device,
                )
                indices_list.append(cp_indices)
                indices_list.append(data_indices)
                offset += cp_length_i + fft_size

            # Concatenate all indices
            self.register_buffer("_ind", torch.cat(indices_list))

    def call(self, inputs: torch.Tensor) -> torch.Tensor:
        """Modulate OFDM resource grid to time-domain signal.

        :param inputs: Resource grid in frequency domain with shape
            `[..., num_ofdm_symbols, fft_size]`
        :output x_time: Time-domain OFDM signal with shape
            `[..., num_ofdm_symbols*(fft_size+cyclic_prefix_length)]` or
            `[..., num_ofdm_symbols*fft_size+sum(cyclic_prefix_length)]`
        """
        # Shift DC subcarrier to first position
        x_freq = torch.fft.ifftshift(inputs, dim=-1)

        # Compute IFFT along the last dimension
        x_time = ifft(x_freq, precision=self.precision)

        cp_len = self._cyclic_prefix_length

        if cp_len.dim() == 1:
            # Individual CP length per OFDM symbol
            # Flatten last two dimensions
            x_time = flatten_last_dims(x_time, 2)

            # Gather full time-domain signal
            return x_time[..., self._ind]
        else:
            # Same CP length for all OFDM symbols
            # Use cached scalar to avoid .item() during tracing
            cp_length = self._cp_length_scalar

            if cp_length > 0:
                # Obtain cyclic prefix
                cp = x_time[..., -cp_length:]

                # Prepend cyclic prefix
                x_time = torch.cat([cp, x_time], dim=-1)

            # Serialize last two dimensions
            return flatten_last_dims(x_time, 2)
