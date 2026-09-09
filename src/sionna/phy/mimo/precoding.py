#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Classes and functions related to MIMO transmit precoding."""

import math
from typing import Optional, Tuple, Union, List
import torch

from sionna._validation import check_tensor_all
from sionna.phy.config import config, dtypes, Precision
from sionna.phy.constants import PI
from sionna.phy.utils import expand_to_rank

__all__ = [
    "rzf_precoding_matrix",
    "cbf_precoding_matrix",
    "rzf_precoder",
    "grid_of_beams_dft_ula",
    "grid_of_beams_dft",
    "flatten_precoding_mat",
    "normalize_precoding_power",
]


def rzf_precoding_matrix(
    h: torch.Tensor,
    alpha: Union[float, torch.Tensor] = 0.0,
    precision: Optional[Precision] = None,
) -> torch.Tensor:
    r"""Computes the Regularized Zero-Forcing (RZF) Precoder.

    This function computes the RZF precoding matrix for a MIMO link, assuming the
    following model:

    .. math::

        \mathbf{y} = \mathbf{H}\mathbf{G}\mathbf{x} + \mathbf{n}

    where :math:`\mathbf{y}\in\mathbb{C}^K` is the received signal vector,
    :math:`\mathbf{H}\in\mathbb{C}^{K\times M}` is the known channel matrix,
    :math:`\mathbf{G}\in\mathbb{C}^{M\times K}` is the precoding matrix,
    :math:`\mathbf{x}\in\mathbb{C}^K` is the symbol vector to be precoded,
    and :math:`\mathbf{n}\in\mathbb{C}^K` is a noise vector.

    The precoding matrix :math:`\mathbf{G}` is defined as:

    .. math::

        \mathbf{G} = \mathbf{V}\mathbf{D}

    where

    .. math::

        \mathbf{V} &= \mathbf{H}^{\mathsf{H}}\left(\mathbf{H}
        \mathbf{H}^{\mathsf{H}} + \alpha \mathbf{I} \right)^{-1}\\
        \mathbf{D} &= \mathop{\text{diag}}\left( \lVert \mathbf{v}_{k} \rVert_2^{-1}, k=0,\dots,K-1 \right)

    where :math:`\alpha>0` is the regularization parameter. The matrix :math:`\mathbf{D}`
    ensures that each stream is precoded with a unit-norm vector,
    i.e., :math:`\mathop{\text{tr}}\left(\mathbf{G}\mathbf{G}^{\mathsf{H}}\right)=K`.
    The function returns the matrix :math:`\mathbf{G}`.

    :param h: Channel matrices with shape [..., K, M]
    :param alpha: Regularization parameter with shape [...] or scalar
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`, :attr:`~sionna.phy.config.Config.precision` is used.

    :output g: [..., M, K], `torch.complex`. Precoding matrices.

    .. rubric:: Examples

    .. code-block:: python

        h = torch.complex(torch.randn(4, 8), torch.randn(4, 8))
        g = rzf_precoding_matrix(h, alpha=0.1)
        # g.shape = torch.Size([8, 4])
    """
    # Determine dtype
    if precision is None:
        cdtype = config.cdtype
    else:
        cdtype = dtypes[precision]["torch"]["cdtype"]

    h = h.to(dtype=cdtype)
    alpha = torch.as_tensor(alpha, dtype=cdtype, device=h.device)

    # Compute pseudo inverse for precoding
    g = h @ h.mH
    alpha = expand_to_rank(alpha, g.dim(), axis=-1)
    k = g.shape[-1]
    eye = torch.eye(k, dtype=cdtype, device=g.device)
    eye = expand_to_rank(eye, g.dim(), 0)
    g = g + alpha * eye

    # Cholesky decomposition and solve
    # Use cholesky_ex with check_errors=False for CUDA graph compatibility
    l, _ = torch.linalg.cholesky_ex(g, check_errors=False)
    # Solve L @ L^H @ X = h for X
    y = torch.linalg.solve_triangular(l, h, upper=False)
    g = torch.linalg.solve_triangular(l.mH, y, upper=True)
    g = g.mH

    # Normalize each column to unit power
    norm = torch.sqrt((g.abs() ** 2).sum(dim=-2, keepdim=True))
    g = torch.where(norm > 0, g / norm, g)

    return g


def cbf_precoding_matrix(
    h: torch.Tensor,
    precision: Optional[Precision] = None,
) -> torch.Tensor:
    r"""Computes the conjugate beamforming (CBF) Precoder.

    This function computes the CBF precoding matrix for a MIMO link, assuming the
    following model:

    .. math::

        \mathbf{y} = \mathbf{H}\mathbf{G}\mathbf{x} + \mathbf{n}

    where :math:`\mathbf{y}\in\mathbb{C}^K` is the received signal vector,
    :math:`\mathbf{H}\in\mathbb{C}^{K\times M}` is the known channel matrix,
    :math:`\mathbf{G}\in\mathbb{C}^{M\times K}` is the precoding matrix,
    :math:`\mathbf{x}\in\mathbb{C}^K` is the symbol vector to be precoded,
    and :math:`\mathbf{n}\in\mathbb{C}^K` is a noise vector.

    The precoding matrix :math:`\mathbf{G}` is defined as:

    .. math::

        \mathbf{G} = \mathbf{V}\mathbf{D}

    where

    .. math::

        \mathbf{V} &= \mathbf{H}^{\mathsf{H}} \\
        \mathbf{D} &= \mathop{\text{diag}}\left( \lVert \mathbf{v}_{k} \rVert_2^{-1}, k=0,\dots,K-1 \right).

    The matrix :math:`\mathbf{D}`
    ensures that each stream is precoded with a unit-norm vector,
    i.e., :math:`\mathop{\text{tr}}\left(\mathbf{G}\mathbf{G}^{\mathsf{H}}\right)=K`.
    The function returns the matrix :math:`\mathbf{G}`.

    :param h: Channel matrices with shape [..., K, M]
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`, :attr:`~sionna.phy.config.Config.precision` is used.

    :output g: [..., M, K], `torch.complex`. Precoding matrices.

    .. rubric:: Examples

    .. code-block:: python

        h = torch.complex(torch.randn(4, 8), torch.randn(4, 8))
        g = cbf_precoding_matrix(h)
        # g.shape = torch.Size([8, 4])
    """
    # Determine dtype
    if precision is None:
        cdtype = config.cdtype
    else:
        cdtype = dtypes[precision]["torch"]["cdtype"]

    h = h.to(dtype=cdtype)

    # Compute conjugate transpose of channel matrix
    g = h.mH

    # Normalize each column to unit power
    norm = torch.sqrt((g.abs() ** 2).sum(dim=-2, keepdim=True))
    g = torch.where(norm > 0, g / norm, g)

    return g


def rzf_precoder(
    x: torch.Tensor,
    h: torch.Tensor,
    alpha: Union[float, torch.Tensor] = 0.0,
    return_precoding_matrix: bool = False,
    precision: Optional[Precision] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    r"""Regularized Zero-Forcing (RZF) Precoder.

    This function implements RZF precoding for a MIMO link, assuming the
    following model:

    .. math::

        \mathbf{y} = \mathbf{H}\mathbf{G}\mathbf{x} + \mathbf{n}

    where :math:`\mathbf{y}\in\mathbb{C}^K` is the received signal vector,
    :math:`\mathbf{H}\in\mathbb{C}^{K\times M}` is the known channel matrix,
    :math:`\mathbf{G}\in\mathbb{C}^{M\times K}` is the precoding matrix,
    :math:`\mathbf{x}\in\mathbb{C}^K` is the symbol vector to be precoded,
    and :math:`\mathbf{n}\in\mathbb{C}^K` is a noise vector.

    The precoding matrix :math:`\mathbf{G}` is defined as (Eq. 4.37) :cite:p:`BHS2017`:

    .. math::

        \mathbf{G} = \mathbf{V}\mathbf{D}

    where

    .. math::

        \mathbf{V} &= \mathbf{H}^{\mathsf{H}}\left(\mathbf{H} \mathbf{H}^{\mathsf{H}} + \alpha \mathbf{I} \right)^{-1}\\
        \mathbf{D} &= \mathop{\text{diag}}\left( \lVert \mathbf{v}_{k} \rVert_2^{-1}, k=0,\dots,K-1 \right)

    where :math:`\alpha>0` is the regularization parameter.

    This ensures that each stream is precoded with a unit-norm vector,
    i.e., :math:`\mathop{\text{tr}}\left(\mathbf{G}\mathbf{G}^{\mathsf{H}}\right)=K`.
    The function returns the precoded vector :math:`\mathbf{G}\mathbf{x}`.

    :param x: Symbol vectors to be precoded with shape [..., K]
    :param h: Channel matrices with shape [..., K, M]
    :param alpha: Regularization parameter with shape [...] or scalar
    :param return_precoding_matrix: If `True`, the precoding matrices are
        also returned
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`, :attr:`~sionna.phy.config.Config.precision` is used.

    :output x_precoded: [..., M], `torch.complex`. Precoded symbol vectors.
    :output g: [..., M, K], `torch.complex`. Precoding matrices. Only returned
        if ``return_precoding_matrix=True``.

    .. rubric:: Examples

    .. code-block:: python

        x = torch.complex(torch.randn(4), torch.randn(4))
        h = torch.complex(torch.randn(4, 8), torch.randn(4, 8))
        x_precoded = rzf_precoder(x, h, alpha=0.1)
        # x_precoded.shape = torch.Size([8])
    """
    # Determine dtype
    if precision is None:
        cdtype = config.cdtype
    else:
        cdtype = dtypes[precision]["torch"]["cdtype"]

    x = x.to(dtype=cdtype)
    h = h.to(dtype=cdtype)

    # Compute the precoding matrix
    g = rzf_precoding_matrix(h, alpha=alpha, precision=precision)

    # Precode
    x_precoded = (g @ x.unsqueeze(-1)).squeeze(-1)

    if return_precoding_matrix:
        return x_precoded, g
    else:
        return x_precoded


def grid_of_beams_dft_ula(
    num_ant: int,
    oversmpl: int = 1,
    precision: Optional[Precision] = None,
    device: Optional[str] = None,
) -> torch.Tensor:
    r"""Computes the Discrete Fourier Transform (DFT) Grid of Beam (GoB)
    coefficients for a uniform linear array (ULA).

    The coefficient applied to antenna :math:`n` for beam :math:`m` is expressed
    as:

    .. math::
        c_n^m = \frac{1}{\sqrt{N}}e^{j\frac{2\pi n m}{N O}},
        \quad n=0,\dots,N-1,\quad m=0,\dots,NO-1

    where :math:`N` is the number of antennas ``num_ant`` and :math:`O` is the oversampling
    factor ``oversmpl``.

    For a half-wavelength-spaced ULA, define the wrapped spatial frequency
    :math:`\nu_m=m/(NO)` for :math:`m\le\lfloor NO/2\rfloor` and
    :math:`\nu_m=(m-NO)/(NO)` otherwise. With positive phase progression
    corresponding to positive azimuth, the main lobe points towards
    :math:`\theta_m=\arcsin(2\nu_m)`, where :math:`\theta_m=0` is perpendicular
    to the antenna array.

    :param num_ant: Number of antennas
    :param oversmpl: Oversampling factor
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`, :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for computation. If `None`,
        :attr:`~sionna.phy.config.Config.device` is used.

    :output gob: [num_ant x oversmpl, num_ant], `torch.complex`.
        The :math:`m`-th row contains the `num_ant` antenna coefficients for
        the :math:`m`-th DFT beam.

    .. rubric:: Examples

    .. code-block:: python

        gob = grid_of_beams_dft_ula(num_ant=8, oversmpl=2)
        # gob.shape = torch.Size([16, 8])
    """
    if precision is None:
        rdtype = config.dtype
    else:
        rdtype = dtypes[precision]["torch"]["dtype"]

    if device is None:
        device = config.device

    oversmpl = int(oversmpl)

    # Beam indices: [0, .., num_ant * oversmpl - 1]
    beam_ind = torch.arange(num_ant * oversmpl, dtype=rdtype, device=device).unsqueeze(-1)

    # Antenna indices: [0, .., num_ant - 1]
    antenna_ind = torch.arange(num_ant, dtype=rdtype, device=device).unsqueeze(0)

    # Compute phases and combine to complex coefficients
    phases = 2 * PI * beam_ind * antenna_ind / (num_ant * oversmpl)
    gob = torch.complex(torch.cos(phases), torch.sin(phases)) / math.sqrt(num_ant)

    return gob


def grid_of_beams_dft(
    num_ant_v: int,
    num_ant_h: int,
    oversmpl_v: int = 1,
    oversmpl_h: int = 1,
    precision: Optional[Precision] = None,
    device: Optional[str] = None,
) -> torch.Tensor:
    r"""Computes the Discrete Fourier Transform (DFT) Grid of Beam (GoB)
    coefficients for a uniform rectangular array (URA).

    GoB indices are arranged over a 2D grid indexed by :math:`(m_v,m_h)`.
    The coefficient of the beam with index :math:`(m_v,m_h)` applied to the
    antenna located at row :math:`n_v` and column :math:`n_h` of the rectangular
    array is expressed as:

    .. math::
        c_{n_v,n_h}^{m_v,m_h}
        = \frac{1}{\sqrt{N_vN_h}}
          e^{j\frac{2\pi n_v m_v}{N_v O_v}}
          e^{j\frac{2\pi n_h m_h}{N_h O_h}}

    where :math:`n_v=0,\dots,N_v-1`, :math:`n_h=0,\dots,N_h-1`,
    :math:`m_v=0,\dots,N_v O_v-1`, :math:`m_h=0,\dots,N_h O_h-1`,
    :math:`N_v,N_h` are the numbers of antennas ``num_ant_v``, ``num_ant_h``,
    and :math:`O_v,O_h` are the oversampling factors ``oversmpl_v``,
    ``oversmpl_h`` in the vertical and
    horizontal direction, respectively.

    We can rewrite more concisely the matrix coefficients
    :math:`c^{m_v,m_h}` as follows:

    .. math::
        \mathbf{c}^{m_v,m_h}
        = \mathbf{c}_h^{m_h} \otimes \mathbf{c}_v^{m_v}

    where :math:`\otimes` denotes the Kronecker product,
    :math:`\mathbf{c}_v^{m_v}` and :math:`\mathbf{c}_h^{m_h}` are the vertical
    and horizontal ULA DFT beams, respectively, and the ordering follows
    column-wise flattening of the rectangular array.

    Such a DFT GoB is, e.g., defined in Section 5.2.2.2.1 :cite:p:`3GPPTS38214`.

    :param num_ant_v: Number of antenna rows (i.e., in vertical direction)
    :param num_ant_h: Number of antenna columns (i.e., in horizontal direction)
    :param oversmpl_v: Oversampling factor in vertical direction
    :param oversmpl_h: Oversampling factor in horizontal direction
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`, :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for computation. If `None`,
        :attr:`~sionna.phy.config.Config.device` is used.

    :output gob: [num_ant_v x oversmpl_v, num_ant_h x oversmpl_h, num_ant_v x num_ant_h], `torch.complex`.
        The elements :math:`[m_v,m_h,:]` contain the antenna coefficients of the
        DFT beam with index pair :math:`(m_v,m_h)`.

    .. rubric:: Examples

    .. code-block:: python

        gob = grid_of_beams_dft(num_ant_v=4, num_ant_h=8)
        # gob.shape = torch.Size([4, 8, 32])
    """
    # Compute the DFT coefficients for vertical and horizontal directions
    gob_v = grid_of_beams_dft_ula(
        num_ant_v, oversmpl=oversmpl_v, precision=precision, device=device
    )
    gob_v = gob_v[:, None, :, None]

    gob_h = grid_of_beams_dft_ula(
        num_ant_h, oversmpl=oversmpl_h, precision=precision, device=device
    )
    gob_h = gob_h[None, :, None, :]

    # Kronecker product
    # [num_ant_v * oversmpl_v, num_ant_h * oversmpl_h, num_ant_v, num_ant_h]
    coef_vh = gob_h * gob_v

    # Flatten the last two dimensions
    coef_vh = flatten_precoding_mat(coef_vh)

    return coef_vh


def flatten_precoding_mat(
    precoding_mat: torch.Tensor,
    by_column: bool = True,
) -> torch.Tensor:
    r"""Flattens a [..., num_ant_v, num_ant_h] precoding matrix associated with
    a rectangular array by producing a [..., num_ant_v x num_ant_h] precoding vector.

    :param precoding_mat: Precoding matrix with shape
        [..., num_antennas_vertical, num_antennas_horizontal].
        The element :math:`(i,j)` contains the precoding coefficient of the
        antenna element located at row :math:`i` and column :math:`j`
        of a rectangular antenna array.
    :param by_column: If `True`, flattening occurs on a per-column basis,
        i.e., the first column is appended to the second, and so on.
        Else, flattening is performed on a per-row basis.

    :output precoding_vec: [..., num_antennas_vertical x num_antennas_horizontal], `torch.complex`.
        Flattened precoding matrix.

    .. rubric:: Examples

    .. code-block:: python

        mat = torch.randn(4, 8, dtype=torch.complex64)
        vec = flatten_precoding_mat(mat)
        # vec.shape = torch.Size([32])
    """
    # Transpose the last two dimensions if flattening by column
    if by_column:
        precoding_mat = precoding_mat.mT

    # Flatten the last two dimensions
    shape = list(precoding_mat.shape[:-2]) + [-1]
    precoding_vec = precoding_mat.reshape(shape)

    return precoding_vec


def normalize_precoding_power(
    precoding_vec: torch.Tensor,
    tx_power_list: Optional[List[float]] = None,
    precision: Optional[Precision] = None,
) -> torch.Tensor:
    r"""Normalizes the beam coefficient power to 1 by default, or to
    ``tx_power_list`` if provided as input.

    :param precoding_vec: Precoding vectors with shape [N, M].
        Each row contains a set of antenna coefficients whose power is to be normalized.
    :param tx_power_list: The :math:`i`-th element defines the power of the
        :math:`i`-th precoding vector. If `None`, power is normalized to 1.
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`, :attr:`~sionna.phy.config.Config.precision` is used.

    :output precoding_vec: [N, M], `torch.complex`. Normalized antenna coefficients.

    .. rubric:: Examples

    .. code-block:: python

        vec = torch.complex(torch.randn(4, 8), torch.randn(4, 8))
        vec_norm = normalize_precoding_power(vec)
        # Each row now has unit power
    """
    if precision is None:
        cdtype = config.cdtype
        rdtype = config.dtype
    else:
        cdtype = dtypes[precision]["torch"]["cdtype"]
        rdtype = dtypes[precision]["torch"]["dtype"]

    precoding_vec = precoding_vec.to(dtype=cdtype)

    if precoding_vec.dim() == 1:
        precoding_vec = precoding_vec.unsqueeze(0)

    if tx_power_list is None:
        tx_power_list = [1.0] * precoding_vec.shape[0]
    if any(power < 0 for power in tx_power_list):
        raise ValueError("Transmit powers must be nonnegative.")

    precoding_vec_norm = torch.norm(precoding_vec, dim=1, keepdim=True)

    check_tensor_all(
        precoding_vec_norm != 0,
        name="precoding_vec",
        message=(
            "Precoding vectors with zero norm cannot be normalized to a given "
            "power."
        ),
    )

    tx_power = torch.tensor(tx_power_list, dtype=rdtype, device=precoding_vec.device).unsqueeze(-1)

    # Normalize the power of each row
    precoding_vec = (precoding_vec / precoding_vec_norm) * torch.sqrt(tx_power)

    return precoding_vec

