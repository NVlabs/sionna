#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Linear algebra utilities."""

from __future__ import annotations

import torch

__all__ = ["inv_cholesky", "matrix_pinv"]


def inv_cholesky(tensor: torch.Tensor) -> torch.Tensor:
    r"""Inverse of the Cholesky decomposition of a matrix

    Given a batch of :math:`M \times M` Hermitian positive definite
    matrices :math:`\mathbf{A}`, this function computes
    :math:`\mathbf{L}^{-1}`, where :math:`\mathbf{L}` is
    the Cholesky decomposition, such that
    :math:`\mathbf{A}=\mathbf{L}\mathbf{L}^{\textsf{H}}`.

    .. note::
        This function assumes that every input matrix is Hermitian positive
        definite. Factorization errors are deliberately not checked because
        doing so synchronizes CUDA execution and prevents CUDA graph capture.
        Results are undefined when the precondition is violated.

    :param tensor: [..., M, M], `torch.float` | `torch.complex`.
        Input tensor of rank greater than one.

    :output inv_chol: [..., M, M], `torch.float` | `torch.complex`.
        A tensor of the same shape and type as ``tensor`` containing
        the inverse of the Cholesky decomposition of its last two dimensions.

    .. rubric:: Examples

    >>> import torch
    >>> from sionna.phy.utils.linalg import inv_cholesky
    >>> a = torch.eye(2)
    >>> inv_cholesky(a)
    tensor([[1., 0.],
            [0., 1.]])
    """
    chol_factor = torch.linalg.cholesky_ex(tensor, check_errors=False)[0]
    dim = chol_factor.shape[-1]
    identity = torch.eye(dim, dtype=chol_factor.dtype, device=chol_factor.device)
    identity = identity.expand(*chol_factor.shape[:-2], dim, dim)
    return torch.linalg.solve_triangular(chol_factor, identity, upper=False)


def matrix_pinv(tensor: torch.Tensor) -> torch.Tensor:
    r"""Computes the Moore–Penrose (or pseudo) inverse of a matrix

    Given a batch of :math:`M \times K` matrices :math:`\mathbf{A}` with rank
    :math:`K` (i.e., linearly independent columns), the function returns
    :math:`\mathbf{A}^+`, such that
    :math:`\mathbf{A}^{+}\mathbf{A}=\mathbf{I}_K`.

    The two inner dimensions are assumed to correspond to the matrix rows
    and columns, respectively.

    .. note::
        This is a Cholesky-based fast path for tall or square matrices with
        full column rank, not a rank-revealing general pseudoinverse.
        Factorization errors are deliberately not checked because doing so
        synchronizes CUDA execution and prevents CUDA graph capture. Results
        are undefined for rank-deficient or otherwise invalid inputs.

    :param tensor: [..., M, K], `torch.float` | `torch.complex`.
        Input tensor of rank greater than or equal to two.

    :output pinv: [..., K, M], `torch.float` | `torch.complex`.
        A tensor containing the matrix pseudo inverse of the last two
        dimensions of ``tensor``.

    .. rubric:: Examples

    >>> import torch
    >>> from sionna.phy.utils.linalg import matrix_pinv
    >>> a = torch.randn(4, 2)
    >>> matrix_pinv(a).shape
    torch.Size([2, 4])
    """
    tensor_h = tensor.mH  # conjugate transpose
    gram = tensor_h @ tensor
    chol = torch.linalg.cholesky_ex(gram, check_errors=False)[0]
    # Solve (L L^H) X^T = A^H via two triangular solves
    y = torch.linalg.solve_triangular(chol, tensor_h, upper=False)
    return torch.linalg.solve_triangular(chol.mH, y, upper=True)
