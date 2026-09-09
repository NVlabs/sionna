#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Spatial-consistency utilities for 3GPP TR 38.901."""

from typing import Literal, Optional

import torch

from sionna.phy import config, dtypes
from sionna.phy.config import Precision

__all__ = [
    "spatial_consistency_correlation_matrix",
    "spatial_consistency_matrix_sqrt",
]


def _dtype(precision: Optional[Precision]) -> torch.dtype:
    if precision is None:
        return config.dtype
    return dtypes[precision]["torch"]["dtype"]


def _to_real_tensor(
    value,
    precision: Optional[Precision] = None,
    device: Optional[str] = None,
) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        if value.is_complex():
            raise TypeError("Expected a real-valued tensor")
        dtype = (
            value.dtype
            if precision is None and value.is_floating_point()
            else _dtype(precision)
        )
        if device is None:
            device = value.device
        return value.to(dtype=dtype, device=device)
    if device is None:
        device = config.device
    return torch.as_tensor(value, dtype=_dtype(precision), device=device)


def spatial_consistency_correlation_matrix(
    distance_2d,
    correlation_distance,
    states: Optional[torch.Tensor] = None,
    correlation_distance_layout: Literal["broadcast", "per_terminal"] = "broadcast",
    precision: Optional[Precision] = None,
    device: Optional[str] = None,
) -> torch.Tensor:
    r"""Computes a TR 38.901 spatial-consistency correlation matrix.

    For two terminals :math:`i` and :math:`j` separated by horizontal distance
    :math:`d_{i,j}`, the correlation coefficient is

    .. math::

        C_{i,j} = \exp\left(-\frac{d_{i,j}}{D}\right)

    where :math:`D` is the parameter-specific correlation distance. The matrix
    :math:`\mathbf{C}` is the finite-dimensional correlation matrix obtained by
    applying the exponential normalized autocorrelation function from
    Eq. (7.4-5), Section 7.4.4 of :cite:p:`TR38901V1920`, to all pairs of terminal
    positions. The spatial-consistency procedure in Section 7.6.3.1 of
    :cite:p:`TR38901V1920` uses this correlation law with the parameter-specific
    correlation distances from Table 7.6.3.1-2.

    If ``states`` is provided, entries for terminals with different states are
    set to zero, while the diagonal remains one. This implements the
    applicability rule from Section 7.6.3.4 of :cite:p:`TR38901V1920` that spatial
    consistency is not modelled across different link types such as outdoor
    LoS, outdoor NLoS, and O2I.

    :param distance_2d: Pairwise horizontal distances [m], with shape
        ``[..., num_points, num_points]``.
    :param correlation_distance: Correlation distance :math:`D` [m]. For
        ``correlation_distance_layout="broadcast"``, this can be a scalar or a
        tensor broadcastable to ``distance_2d``. For
        ``correlation_distance_layout="per_terminal"``, its last dimension
        must have length ``num_points`` and contain one distance per terminal.
        The corresponding pairwise distance is the geometric mean of the two
        terminal distances, which keeps the result symmetric.
    :param states: Optional integer or Boolean state labels with shape
        ``[..., num_points]``. Points with unequal state labels are
        uncorrelated.
    :param correlation_distance_layout: Interpretation of
        ``correlation_distance``. Must be ``"broadcast"`` or
        ``"per_terminal"``. Defaults to ``"broadcast"``.
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`, :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for computation. If `None`,
        :attr:`~sionna.phy.config.Config.device` is used.

    :output C: Spatial-consistency correlation matrix with shape broadcast
        from ``distance_2d`` and ``correlation_distance``.
    """
    distance_2d = _to_real_tensor(distance_2d, precision=precision, device=device)
    correlation_distance = _to_real_tensor(
        correlation_distance, precision=precision, device=distance_2d.device
    )
    correlation_distance = torch.clamp(
        correlation_distance, min=torch.finfo(distance_2d.dtype).tiny
    )
    if correlation_distance_layout not in ("broadcast", "per_terminal"):
        raise ValueError(
            "correlation_distance_layout must be 'broadcast' or "
            "'per_terminal'"
        )
    if correlation_distance_layout == "per_terminal":
        if (
            correlation_distance.dim() == 0
            or correlation_distance.shape[-1] != distance_2d.shape[-1]
        ):
            raise ValueError(
                "A per-terminal correlation distance must have last dimension "
                "equal to the number of points"
            )
        correlation_distance = torch.sqrt(
            correlation_distance.unsqueeze(-1)
            * correlation_distance.unsqueeze(-2)
        )

    correlation = torch.exp(-distance_2d / correlation_distance)

    if states is not None:
        states = torch.as_tensor(states, device=distance_2d.device)
        same_state = states.unsqueeze(-1) == states.unsqueeze(-2)
        correlation = torch.where(same_state, correlation, torch.zeros_like(correlation))

    eye = torch.eye(
        distance_2d.shape[-1],
        dtype=torch.bool,
        device=distance_2d.device,
    )
    return torch.where(eye, torch.ones_like(correlation), correlation)


def _compiled_psd_cholesky(matrix: torch.Tensor) -> torch.Tensor:
    """Return a graph-safe Cholesky-like factor for a PSD matrix."""
    n = matrix.shape[-1]
    factor = torch.zeros_like(matrix)
    for column in range(n):
        residual = matrix[..., column:, column] - (
            factor[..., column:, :column]
            * factor[..., column, :column].unsqueeze(-2)
        ).sum(dim=-1)
        diagonal = residual[..., 0]
        if column == 0:
            duplicate = torch.zeros_like(diagonal, dtype=torch.bool)
        else:
            duplicate = torch.any(
                torch.all(
                    matrix[..., column, :].unsqueeze(-2)
                    == matrix[..., :column, :],
                    dim=-1,
                ),
                dim=-1,
            )
        positive = (diagonal > 0.0) & (~duplicate)
        divisor = torch.sqrt(
            torch.where(positive, diagonal, torch.ones_like(diagonal))
        )
        values = residual / divisor.unsqueeze(-1)
        values = torch.where(
            positive.unsqueeze(-1), values, torch.zeros_like(values)
        )
        factor[..., column:, column] = values
    return factor


def spatial_consistency_matrix_sqrt(
    correlation_matrix,
    jitter: Optional[float] = None,
    precision: Optional[Precision] = None,
    device: Optional[str] = None,
) -> torch.Tensor:
    r"""Computes a square root of a spatial-consistency matrix.

    The returned matrix :math:`\mathbf{L}` satisfies approximately

    .. math::

        \mathbf{C} = \mathbf{L}\mathbf{L}^{\mathsf{T}}

    where :math:`\mathbf{C}` is the input correlation matrix. For
    positive-definite matrices a lower-triangular Cholesky factor is returned.
    Singular positive-semidefinite matrices, such as those caused by co-located
    terminals, are supported both eagerly and under ``torch.compile`` without
    adding jitter. This keeps co-located terminals exactly tied to the same
    random-field value. The eager fallback uses a symmetric eigendecomposition;
    the compiled path uses a graph-safe PSD Cholesky recurrence. A small
    diagonal jitter can optionally be added before factorization.

    :param correlation_matrix: Correlation matrix with shape
        ``[..., num_points, num_points]``.
    :param jitter: Optional diagonal jitter. If `None`, no jitter is added.
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`, the dtype of ``correlation_matrix`` is preserved for
        tensor inputs, otherwise :attr:`~sionna.phy.config.Config.precision` is
        used.
    :param device: Device for computation. If `None`,
        :attr:`~sionna.phy.config.Config.device` is used.

    :output L: Matrix square root with the same shape as
        ``correlation_matrix``.
    """
    correlation_matrix = _to_real_tensor(
        correlation_matrix, precision=precision, device=device
    )
    correlation_matrix = 0.5 * (
        correlation_matrix + correlation_matrix.transpose(-1, -2)
    )
    eye = torch.eye(
        correlation_matrix.shape[-1],
        dtype=correlation_matrix.dtype,
        device=correlation_matrix.device,
    )
    added_jitter = jitter is not None and jitter != 0.0
    if added_jitter:
        correlation_matrix = correlation_matrix + jitter * eye
    if torch.compiler.is_compiling():
        # cuSOLVER's eigendecomposition cannot be captured in a CUDA graph.
        # This recurrence also handles exact zero pivots without perturbing
        # perfectly correlated terminals.
        return _compiled_psd_cholesky(correlation_matrix)

    chol, info = torch.linalg.cholesky_ex(correlation_matrix, check_errors=False)
    eye_bool = torch.eye(
        correlation_matrix.shape[-1],
        dtype=torch.bool,
        device=correlation_matrix.device,
    )
    has_perfect_off_diagonal_correlation = torch.any(
        (correlation_matrix >= 1.0 - 10.0 * torch.finfo(correlation_matrix.dtype).eps)
        & (~eye_bool)
    )
    if bool(torch.all(info == 0)) and not (
        (not added_jitter) and bool(has_perfect_off_diagonal_correlation)
    ):
        return chol

    # Cholesky failed for at least one matrix. The following eager checks use
    # Python control flow and may synchronize CUDA tensors, but this path is
    # only needed for singular or nearly singular matrices, e.g., duplicate UT
    # positions.
    tolerance = max(1e-10, 1000.0 * torch.finfo(correlation_matrix.dtype).eps)
    n = correlation_matrix.shape[-1]
    flat = correlation_matrix.reshape(-1, n, n)
    output = torch.empty_like(flat)
    # Batched eigh can request very large workspaces. Chunking keeps the
    # fallback near a 2M-element working-set budget for large calibration
    # batches with duplicate positions.
    chunk_size = max(1, min(flat.shape[0], max(16, 2_000_000 // (n * n))))
    for start in range(0, flat.shape[0], chunk_size):
        stop = min(start + chunk_size, flat.shape[0])
        chunk = flat[start:stop]
        factor_dtype = torch.float64 if chunk.dtype == torch.float32 else chunk.dtype
        chunk = chunk.to(dtype=factor_dtype)
        eigenvalues, eigenvectors = torch.linalg.eigh(chunk)
        if bool(torch.any(eigenvalues < -tolerance)):
            raise RuntimeError(
                "Spatial-consistency correlation matrix is not positive "
                "semidefinite"
            )
        eigenvalues = torch.clamp(eigenvalues, min=0.0)
        scaled_eigenvectors = eigenvectors * torch.sqrt(eigenvalues).unsqueeze(-2)
        output[start:stop] = (
            scaled_eigenvectors @ eigenvectors.transpose(-1, -2)
        ).to(dtype=output.dtype)
    return output.reshape_as(correlation_matrix)
