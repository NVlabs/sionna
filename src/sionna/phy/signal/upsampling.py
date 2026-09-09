#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Block implementing upsampling"""

from typing import Optional

import torch

from sionna.phy import Block
from sionna.phy.config import Precision
from sionna.phy.utils import flatten_last_dims

__all__ = ["Upsampling"]


class Upsampling(Block):
    """Upsamples a tensor along a specified axis by inserting zeros
    between samples.

    :param samples_per_symbol: Upsampling factor. If ``samples_per_symbol``
        is equal to `n`, then the upsampled axis will be `n`-times longer.
    :param axis: Dimension to be up-sampled. Defaults to -1.
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`, :attr:`~sionna.phy.config.Config.precision` is
        used.
    :param device: Device for computation. If `None`,
        :attr:`~sionna.phy.config.Config.device` is used.

    :input x: [..., n, ...], `torch.float` or `torch.complex`.
        Tensor to be upsampled. `n` is the size of the `axis` dimension.

    :output y: [..., n*samples_per_symbol, ...], `torch.float` or `torch.complex`.
        Upsampled tensor.

    .. rubric:: Examples

    .. code-block:: python

        import torch
        from sionna.phy.signal import Upsampling

        upsampler = Upsampling(samples_per_symbol=4)
        x = torch.randn(32, 100)
        y = upsampler(x)
        print(y.shape)
        # torch.Size([32, 400])
    """

    def __init__(
        self,
        samples_per_symbol: int,
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

        self._samples_per_symbol = samples_per_symbol
        self._axis = axis

    @property
    def samples_per_symbol(self) -> int:
        """Upsampling factor"""
        return self._samples_per_symbol

    @property
    def axis(self) -> int:
        """Dimension to be upsampled"""
        return self._axis

    def call(self, x: torch.Tensor) -> torch.Tensor:
        # Move target axis to last position
        x = torch.swapaxes(x, self._axis, -1)

        # Add dimension for zero-padding
        x = x.unsqueeze(-1)

        # Pad with zeros: [samples_per_symbol - 1] zeros after each sample
        pad = (0, self._samples_per_symbol - 1)  # (left, right) for last dim
        x = torch.nn.functional.pad(x, pad, value=0)

        # Flatten last two dimensions
        x = flatten_last_dims(x, 2)

        # Move last axis back to original position
        x = torch.swapaxes(x, -1, self._axis)

        return x
