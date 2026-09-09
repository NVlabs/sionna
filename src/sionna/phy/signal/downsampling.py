#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Block implementing downsampling"""

from typing import Optional

import torch

from sionna.phy import Block
from sionna.phy.config import Precision

__all__ = ["Downsampling"]


class Downsampling(Block):
    """Downsamples a tensor along a specified axis by retaining one out of
    ``samples_per_symbol`` elements.

    :param samples_per_symbol: Downsampling factor. If ``samples_per_symbol``
        is equal to `n`, then the downsampled axis will be `n`-times shorter.
    :param offset: Index of the first element to be retained. Defaults to 0.
    :param num_symbols: Total number of symbols to be retained after
        downsampling. If `None`, all available symbols are retained.
        Defaults to `None`.
    :param axis: Dimension to be downsampled. Defaults to -1.
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`, :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for computation. If `None`,
        :attr:`~sionna.phy.config.Config.device` is used.

    :input x: [..., n, ...], `torch.float` or `torch.complex`.
        Tensor to be downsampled. `n` is the size of the `axis` dimension.

    :output y: [..., k, ...], `torch.float` or `torch.complex`.
        Downsampled tensor. The number of available output samples is
        ``k_available = max(0, ceil((n-offset)/samples_per_symbol))``.
        If ``num_symbols`` is `None`, then ``k = k_available``. Otherwise,
        ``k = min(k_available, num_symbols)``.

    .. rubric:: Examples

    .. code-block:: python

        import torch
        from sionna.phy.signal import Downsampling

        downsampler = Downsampling(samples_per_symbol=4, offset=2)
        x = torch.randn(32, 400)
        y = downsampler(x)
        print(y.shape)
        # torch.Size([32, 100])
    """

    def __init__(
        self,
        samples_per_symbol: int,
        offset: int = 0,
        num_symbols: Optional[int] = None,
        axis: int = -1,
        precision: Optional[Precision] = None,
        device: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(precision=precision, device=device, **kwargs)

        if samples_per_symbol < 1:
            raise ValueError(
                "samples_per_symbol must be a positive integer, "
                f"got {samples_per_symbol}"
            )

        # A negative offset or num_symbols would be reinterpreted by Python
        # slice semantics as counting back from the end of the signal, which
        # silently returns the wrong samples instead of failing.
        if offset < 0:
            raise ValueError(
                f"offset must be a non-negative integer, got {offset}"
            )

        if num_symbols is not None and num_symbols < 0:
            raise ValueError(
                "num_symbols must be a non-negative integer or None, "
                f"got {num_symbols}"
            )

        self._samples_per_symbol = samples_per_symbol
        self._offset = offset
        self._num_symbols = num_symbols
        self._axis = axis

    @property
    def samples_per_symbol(self) -> int:
        """Downsampling factor"""
        return self._samples_per_symbol

    @property
    def offset(self) -> int:
        """Index of the first element to be retained"""
        return self._offset

    @property
    def num_symbols(self) -> Optional[int]:
        """Total number of symbols to be retained after downsampling"""
        return self._num_symbols

    @property
    def axis(self) -> int:
        """Dimension to be downsampled"""
        return self._axis

    def call(self, x: torch.Tensor) -> torch.Tensor:
        # Put selected axis last
        x = torch.swapaxes(x, self._axis, -1)

        # Downsample
        x = x[..., self._offset :: self._samples_per_symbol]

        if self._num_symbols is not None:
            x = x[..., : self._num_symbols]

        # Put last axis to original position
        x = torch.swapaxes(x, -1, self._axis)

        return x
