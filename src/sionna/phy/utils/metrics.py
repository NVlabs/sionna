#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Functions to compute frequently used metrics in Sionna PHY"""

import torch
from sionna.phy.config import dtypes, Precision

__all__ = [
    "compute_ber",
    "compute_ser",
    "compute_bler",
    "count_errors",
    "count_block_errors",
]


def _validate_metric_inputs(
    values: torch.Tensor, estimates: torch.Tensor
) -> None:
    """Validate the shared shape contract of error metrics."""
    if values.shape != estimates.shape:
        raise ValueError("Inputs must have the same shape.")
    if values.numel() == 0:
        raise ValueError("Inputs must not be empty.")


def compute_ber(
    b: torch.Tensor, b_hat: torch.Tensor, precision: Precision = "double"
) -> torch.Tensor:
    """Computes the bit error rate (BER) between two binary tensors.

    :param b: A tensor of arbitrary shape filled with ones and zeros.
    :param b_hat: A tensor like ``b``.
    :param precision: Precision used for internal calculations and outputs.
        Defaults to ``"double"``.

    :output ber: `torch.float`.
        BER.

    .. rubric:: Examples

    .. code-block:: python

        import torch
        from sionna.phy.utils import compute_ber

        b = torch.tensor([0, 1, 0, 1])
        b_hat = torch.tensor([0, 1, 1, 0])
        print(compute_ber(b, b_hat).item())
        # 0.5
    """
    _validate_metric_inputs(b, b_hat)
    b_hat = b_hat.to(b.dtype)
    rdtype = dtypes[precision]["torch"]["dtype"]
    ber = torch.ne(b, b_hat)
    ber = ber.to(rdtype)
    return torch.mean(ber)


def compute_ser(
    s: torch.Tensor, s_hat: torch.Tensor, precision: Precision = "double"
) -> torch.Tensor:
    """Computes the symbol error rate (SER) between two integer tensors.

    :param s: A tensor of arbitrary shape filled with integers.
    :param s_hat: A tensor like ``s``.
    :param precision: Precision used for internal calculations and outputs.
        Defaults to ``"double"``.

    :output ser: `torch.float`.
        SER.

    .. rubric:: Examples

    .. code-block:: python

        import torch
        from sionna.phy.utils import compute_ser

        s = torch.tensor([0, 1, 2, 3])
        s_hat = torch.tensor([0, 1, 3, 2])
        print(compute_ser(s, s_hat).item())
        # 0.5
    """
    return compute_ber(s, s_hat, precision)


def compute_bler(
    b: torch.Tensor, b_hat: torch.Tensor, precision: Precision = "double"
) -> torch.Tensor:
    """Computes the block error rate (BLER) between two binary tensors.

    A block error happens if at least one element of ``b`` and ``b_hat``
    differ in one block. The BLER is evaluated over the last dimension of
    the input, i.e., all elements of the last dimension are considered to
    define a block.

    This is also sometimes referred to as `word error rate` or `frame error
    rate`.

    :param b: A tensor of arbitrary shape filled with ones and zeros.
    :param b_hat: A tensor like ``b``.
    :param precision: Precision used for internal calculations and outputs.
        Defaults to ``"double"``.

    :output bler: `torch.float`.
        BLER.

    .. rubric:: Examples

    .. code-block:: python

        import torch
        from sionna.phy.utils import compute_bler

        b = torch.tensor([[0, 1], [1, 0]])
        b_hat = torch.tensor([[0, 1], [1, 1]])
        # The first block is correct, the second block is incorrect
        print(compute_bler(b, b_hat).item())
        # 0.5
    """
    _validate_metric_inputs(b, b_hat)
    b_hat = b_hat.to(b.dtype)
    rdtype = dtypes[precision]["torch"]["dtype"]
    bler = torch.any(torch.ne(b, b_hat), dim=-1)
    bler = bler.to(rdtype)
    return torch.mean(bler)


def count_errors(b: torch.Tensor, b_hat: torch.Tensor) -> torch.Tensor:
    """Counts the number of bit errors between two binary tensors.

    :param b: A tensor of arbitrary shape filled with ones and zeros.
    :param b_hat: A tensor like ``b``.

    :output num_errors: `torch.int64`.
        Number of bit errors.

    .. rubric:: Examples

    .. code-block:: python

        import torch
        from sionna.phy.utils import count_errors

        b = torch.tensor([0, 1, 0, 1])
        b_hat = torch.tensor([0, 1, 1, 0])
        print(count_errors(b, b_hat).item())
        # 2
    """
    _validate_metric_inputs(b, b_hat)
    b_hat = b_hat.to(b.dtype)
    errors = torch.ne(b, b_hat)
    errors = errors.to(torch.int64)
    return torch.sum(errors)


def count_block_errors(b: torch.Tensor, b_hat: torch.Tensor) -> torch.Tensor:
    """Counts the number of block errors between two binary tensors.

    A block error happens if at least one element of ``b`` and ``b_hat``
    differ in one block. The BLER is evaluated over the last dimension of
    the input, i.e., all elements of the last dimension are considered to
    define a block.

    This is also sometimes referred to as `word error rate` or `frame error
    rate`.

    :param b: A tensor of arbitrary shape filled with ones and zeros.
    :param b_hat: A tensor like ``b``.

    :output num_errors: `torch.int64`.
        Number of block errors.

    .. rubric:: Examples

    .. code-block:: python

        import torch
        from sionna.phy.utils import count_block_errors

        b = torch.tensor([[0, 1], [1, 0]])
        b_hat = torch.tensor([[0, 1], [1, 1]])
        print(count_block_errors(b, b_hat).item())
        # 1
    """
    _validate_metric_inputs(b, b_hat)
    b_hat = b_hat.to(b.dtype)
    errors = torch.any(torch.ne(b, b_hat), dim=-1)
    errors = errors.to(torch.int64)
    return torch.sum(errors)
