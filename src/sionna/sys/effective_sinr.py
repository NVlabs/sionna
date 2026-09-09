#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Effective SINR computation for Sionna SYS"""

import json
import os
import warnings
from abc import abstractmethod
from numbers import Real
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch

from sionna._validation import check_tensor_values_in
from sionna.phy import Block, config, dtypes
from sionna.phy.config import Precision
from sionna.phy.utils import (
    DeepUpdateDict,
    db_to_lin,
    dict_keys_to_int,
    expand_to_rank,
    gather_from_batched_indices,
    scalar_to_shaped_tensor,
    to_list,
)

__all__ = ["EffectiveSINR", "EESM"]


class EffectiveSINR(Block):
    r"""Class template for computing the effective SINR from input SINR values
    across multiple subcarriers and streams.

    :input sinr: [..., num_ofdm_symbols, num_subcarriers, num_ut, num_streams_per_ut], `torch.float`.
        Post-equalization SINR in linear scale for different OFDM symbols,
        subcarriers, users and streams.
        If one entry is zero, the corresponding stream is considered as not
        utilized.
    :input mcs_index: [..., num_ut], `torch.int32` (default: `None`).
        Modulation and coding scheme (MCS) index for each user.
    :input mcs_table_index: [..., num_ut], `torch.int32` (default: `None`).
        MCS table index for each user.
    :input mcs_category: [..., num_ut], `torch.int32` (default: `None`).
        MCS table category for each user.
    :input per_stream: `bool` (default: `False`).
        If `True`, the effective SINR is computed on a per-user and
        per-stream basis and is aggregated across different subcarriers.
        If `False`, the effective SINR is computed on a per-user basis and
        is aggregated across streams and subcarriers.
    :input kwargs: `dict`.
        Additional input parameters.

    :output sinr_eff: ([..., num_ut, num_streams_per_ut] | [..., num_ut]), `torch.float`.
        Effective SINR in linear scale for each user and associated stream.
        If ``per_stream`` is `True`, then ``sinr_eff`` has shape
        ``[..., num_ut, num_streams_per_ut]``, and ``sinr_eff[..., u, s]`` is
        the effective SINR for stream ``s`` of user ``u`` across all
        subcarriers.
        If ``per_stream`` is `False`, then ``sinr_eff`` has shape
        ``[..., num_ut]``, and ``sinr_eff[..., u]`` is the effective SINR for
        user ``u`` across all streams and subcarriers.
    """

    def calibrate(self) -> None:
        """Optional calibration hook for subclasses.

        The base implementation is a no-op. Concrete models such as
        :class:`~sionna.sys.EESM` ship fixed parameters and do not recalibrate
        when this method is called.
        """
        pass

    @abstractmethod
    def call(
        self,
        sinr: torch.Tensor,
        mcs_index: Optional[torch.Tensor] = None,
        mcs_table_index: Optional[torch.Tensor] = None,
        mcs_category: Optional[torch.Tensor] = None,
        per_stream: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute effective SINR. Must be implemented by subclasses."""
        pass


class EESM(EffectiveSINR):
    r"""Computes the effective SINR from input SINR values across multiple
    subcarriers and streams via the exponential effective SINR mapping (EESM)
    method.

    Let :math:`\mathrm{SINR}_{u,c,s}>0` be the SINR experienced by user
    :math:`u` on subcarrier :math:`c=1,\dots,C`, and stream
    :math:`s=1,\dots,S_c`.
    If ``per_stream`` is `False`, it computes the effective SINR aggregated
    across all utilized streams and subcarriers for each user :math:`u`:

    .. math::
        \mathrm{SINR}^{\mathrm{eff}}_u = -\beta_u \log \left(
        \frac{1}{|\mathcal{R}_u|}
        \sum_{(c,s)\in\mathcal{R}_u}
        e^{-\frac{\mathrm{SINR}_{u,c,s}}{\beta_u}}
        \right),
        \quad \forall\, u

    where :math:`\mathcal{R}_u` is the set of used (positive-SINR) resource
    elements for user :math:`u` across subcarriers and streams, and
    :math:`\beta>0` is a parameter depending on the Modulation and Coding
    Scheme (MCS) of user :math:`u`.

    If ``per_stream`` is `True`, it computes the effective SINR aggregated
    across subcarriers, for each user :math:`u` and associated stream
    :math:`s`:

    .. math::
        \mathrm{SINR}^{\mathrm{eff}}_{u,s} = -\beta_u \log \left(
        \frac{1}{|\mathcal{R}_{u,s}|}
        \sum_{c\in\mathcal{R}_{u,s}}
        e^{-\frac{\mathrm{SINR}_{u,c,s}}{\beta_u}}
        \right),
        \quad \forall\, u,s.

    :param load_beta_table_from: File name from which the tables containing the
        values of :math:`\beta` parameters are loaded.
        If ``'default'``, uses the built-in table.
    :param sinr_eff_min_db: Minimum effective SINR value [dB]. Useful to avoid
        numerical errors. Defaults to -30.
    :param sinr_eff_max_db: Maximum effective SINR value [dB]. Useful to avoid
        numerical errors. Defaults to 30.
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`,
        :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for computation. If `None`,
        :attr:`~sionna.phy.config.Config.device` is used.

    :input sinr: [..., num_ofdm_symbols, num_subcarriers, num_ut, num_streams_per_ut], `torch.float`.
        Post-equalization SINR in linear scale for different OFDM symbols,
        subcarriers, users and streams.
        If one entry is zero, the corresponding stream is considered as not
        utilized.
    :input mcs_index: [..., num_ut], `torch.int32`.
        Modulation and coding scheme (MCS) index for each user.
    :input mcs_table_index: [..., num_ut], `torch.int32` (default: 1).
        MCS table index for each user. The default beta table covers indices
        ``{1, 2}`` only; unsupported indices raise ``ValueError``. Supply a
        custom beta table, or bypass EESM by passing ``sinr_eff`` and
        ``num_allocated_re`` to :class:`~sionna.sys.PHYAbstraction`.
    :input mcs_category: [..., num_ut], `torch.int32` (default: `None`).
        Accepted for API symmetry with :class:`~sionna.sys.PHYAbstraction`.
        The default beta parameters are category-independent (shared across
        PUSCH/PDSCH for the same table index) and this argument is unused.
    :input per_stream: `bool` (default: `False`).
        If `True`, then the effective SINR is computed on a per-user and
        per-stream basis and is aggregated across different subcarriers.
        If `False`, then the effective SINR is computed on a per-user basis and
        is aggregated across streams and subcarriers.

    :output sinr_eff: ([..., num_ut, num_streams_per_ut] | [..., num_ut]), `torch.float`.
        Effective SINR in linear scale for each user and associated stream.
        If ``per_stream`` is `True`, then ``sinr_eff`` has shape
        ``[..., num_ut, num_streams_per_ut]``, and ``sinr_eff[..., u, s]`` is
        the effective SINR for stream ``s`` of user ``u`` across all
        subcarriers.
        If ``per_stream`` is `False`, then ``sinr_eff`` has shape
        ``[..., num_ut]``, and ``sinr_eff[..., u]`` is the effective SINR for
        user ``u`` across all streams and subcarriers.
        Users, or streams, without any used resource are assigned a null
        effective SINR of exactly 0.

    .. rubric:: Notes

    If the input SINR is zero for a specific stream, the stream is
    considered unused and does not contribute to the effective SINR
    computation. The averages above are therefore over used resources only,
    not over the full ``C`` / ``CS`` grid.

    A user, or stream, whose resources are all unused has no defined effective
    SINR and is assigned exactly 0, which lies outside the range spanned by
    ``sinr_eff_min_db`` and ``sinr_eff_max_db``. Consumers that treat
    non-positive values as missing, such as
    :class:`~sionna.sys.OuterLoopLinkAdaptation`, depend on this convention.

    .. rubric:: Examples

    .. code-block:: python

        import torch
        from sionna.phy import config
        from sionna.sys import EESM
        from sionna.phy.utils import db_to_lin

        batch_size = 10
        num_ofdm_symbols = 12
        num_subcarriers = 32
        num_ut = 15
        num_streams_per_ut = 2

        # Generate random MCS indices
        mcs_index = torch.randint(0, 27, (batch_size, num_ut))

        # Instantiate the EESM object
        eesm = EESM()

        # Generate random SINR values
        sinr_db = torch.rand(batch_size, num_ofdm_symbols, num_subcarriers,
                             num_ut, num_streams_per_ut) * 35 - 5
        sinr = db_to_lin(sinr_db)

        # Compute the effective SINR for each receiver
        sinr_eff = eesm(sinr, mcs_index, mcs_table_index=1, per_stream=False)
        print(sinr_eff.shape)
        # torch.Size([10, 15])

        # Compute the per-stream effective SINR for each receiver
        sinr_eff_per_stream = eesm(sinr, mcs_index, mcs_table_index=2,
                                   per_stream=True)
        print(sinr_eff_per_stream.shape)
        # torch.Size([10, 15, 2])
    """

    def __init__(
        self,
        load_beta_table_from: str = "default",
        sinr_eff_min_db: float = -30.0,
        sinr_eff_max_db: float = 30.0,
        precision: Optional[Precision] = None,
        device: Optional[str] = None,
    ) -> None:
        super().__init__(precision=precision, device=device)

        self._sinr_eff_min = db_to_lin(sinr_eff_min_db, precision=precision).to(self.device)
        self._sinr_eff_max = db_to_lin(sinr_eff_max_db, precision=precision).to(self.device)
        self._beta_table: Optional[Dict] = None
        self._beta_tensor: Optional[torch.Tensor] = None

        if load_beta_table_from == "default":
            self.beta_table_filenames = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "esm_params/eesm_beta_table.json",
            )
        else:
            self.beta_table_filenames = load_beta_table_from

    @property
    def beta_table(self) -> Dict:
        r"""`dict` (read-only): Maps MCS indices to the corresponding
        parameters, commonly called :math:`\beta`, calibrating the Exponential
        Effective SINR Map (EESM) method. It has the form
        ``beta_table['index'][mcs_table_index][mcs]``.
        """
        return self._beta_table

    @property
    def beta_tensor(self) -> torch.Tensor:
        r"""[n_tables, n_mcs] (read-only): Tensor corresponding to
        ``self.beta_table``.
        """
        return self._beta_tensor

    @property
    def beta_table_filenames(self) -> List[str]:
        r"""`str` | list of `str`: Get/set the absolute path name of the JSON
        file containing the mapping between MCS and EESM beta parameters,
        stored in ``beta``.
        """
        return self._beta_table_filenames

    @beta_table_filenames.setter
    def beta_table_filenames(self, value: Union[str, List[str]]) -> None:
        self._beta_table_filenames = to_list(value)
        # Load the table
        self._beta_table = DeepUpdateDict({})
        for f in self._beta_table_filenames:
            try:
                with open(f, "r", encoding="utf-8") as file:
                    subtable = json.load(file, object_hook=dict_keys_to_int)
                    # Merge with the existing one
                    self._beta_table.deep_update(subtable)
            except FileNotFoundError:
                warnings.warn(
                    f"EESM beta parameters file '{f}' does not exist. Skipping...",
                    UserWarning,
                    stacklevel=2,
                )

        if self._beta_table == {}:
            raise ValueError("No EESM beta parameter table found.")

        # Check table validity
        self.validate_beta_table()

        # Build the corresponding tensor
        table_idx_vec = list(self._beta_table["index"].keys())
        n_mcs_vec = []
        for table_idx in table_idx_vec:
            n_mcs_vec.append(len(self._beta_table["index"][table_idx]))
        beta_tensor = np.zeros([max(table_idx_vec), max(n_mcs_vec)])
        for table_idx in table_idx_vec:
            mcs_vec = self._beta_table["index"][table_idx]
            beta_tensor[table_idx - 1, : len(mcs_vec)] = mcs_vec
        self._beta_tensor = torch.tensor(
            beta_tensor, dtype=self.dtype, device=self.device
        )

    def validate_beta_table(self) -> bool:
        r"""Validates the EESM beta parameter dictionary ``self.beta_table``.

        :output is_valid: `True` if ``self.beta_table`` has a valid structure.

        :raises ValueError: If the structure is invalid.
        """
        if not isinstance(self._beta_table, dict):
            raise ValueError("Must be a dictionary")
        if not set(self._beta_table.keys()) >= {"index"}:
            raise ValueError("Key must be 'index'")
        for table_index in self._beta_table["index"]:
            if not isinstance(table_index, int) or table_index < 1:
                raise ValueError("EESM table indices must be positive integers")
            betas = self._beta_table["index"][table_index]
            if not isinstance(betas, list):
                raise ValueError(
                    f"self.beta_table['index'][{table_index}] must be a list"
                )
            if len(betas) == 0:
                raise ValueError(
                    f"self.beta_table['index'][{table_index}] must be non-empty"
                )
            if any(
                isinstance(beta, bool)
                or not isinstance(beta, Real)
                or not np.isfinite(beta)
                or beta <= 0
                for beta in betas
            ):
                raise ValueError(
                    f"self.beta_table['index'][{table_index}] must contain "
                    "finite positive beta values"
                )
        return True

    def call(
        self,
        sinr: torch.Tensor,
        mcs_index: torch.Tensor,
        mcs_table_index: Union[int, torch.Tensor] = 1,
        mcs_category: Optional[torch.Tensor] = None,
        per_stream: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute effective SINR using EESM method."""
        # Cast and reshape inputs
        sinr = sinr.to(self.dtype)
        num_ut = sinr.shape[-2]
        num_batch_dim = sinr.dim() - 4
        batch_dim = list(sinr.shape[:num_batch_dim])

        mcs_index = scalar_to_shaped_tensor(
            mcs_index, torch.int32, batch_dim + [num_ut], device=self.device
        )
        mcs_table_index = scalar_to_shaped_tensor(
            mcs_table_index, torch.int32, batch_dim + [num_ut], device=self.device
        )
        supported_table_indices = sorted(self._beta_table["index"].keys())
        check_tensor_values_in(
            mcs_table_index,
            supported_table_indices,
            name="mcs_table_index",
            message=(
                "mcs_table_index must be in "
                f"{supported_table_indices} for the loaded EESM beta table. "
                "Pass sinr_eff and num_allocated_re to PHYAbstraction to bypass "
                "EESM, or supply a custom beta table covering the requested "
                "indices."
            ),
        )
        # The default beta parameters are category-independent.
        _ = mcs_category

        # Transpose SINR from / to:
        # [..., num_ofdm_symbols, num_subcarriers, num_ut, num_streams_per_ut]
        # [..., num_ut, num_streams_per_ut, num_ofdm_symbols, num_subcarriers]
        perm = list(range(num_batch_dim)) + [
            num_batch_dim + 2,
            num_batch_dim + 3,
            num_batch_dim,
            num_batch_dim + 1,
        ]
        sinr = sinr.permute(*perm)

        # Axis over which SINR is aggregated
        if per_stream:
            axis = (-2, -1)
        else:
            axis = (-3, -2, -1)

        # If per_stream is True: n. used subcarriers per stream and rx
        # [..., num_ut, num_streams_per_ut]
        # If per_stream is False: n. used subcarriers and layers per rx
        # [..., num_ut]
        num_used_res = (sinr > 0).to(self.dtype).sum(dim=axis)

        # Ensure MCS is non-negative (for non-scheduled UTs)
        mcs_index = torch.maximum(mcs_index, torch.tensor(0, device=self.device))

        # Gather beta
        # Stack indices to extract BLER from interpolated table
        idx = torch.stack([mcs_table_index - 1, mcs_index], dim=-1)
        # [..., num_ut]
        beta = gather_from_batched_indices(self._beta_tensor, idx)
        beta = beta.to(sinr.dtype)

        # [..., num_ut, 1, 1, 1]
        beta_expand1 = expand_to_rank(beta, sinr.dim(), axis=-1)

        # Exponentiate SINR
        # [..., num_ut, num_streams_per_ut, num_ofdm_symbols, num_subcarriers]
        sinr_exp = torch.where(sinr > 0, torch.exp(-sinr / beta_expand1), torch.tensor(0.0, device=self.device))

        # Log + average across resources
        sinr_eff = torch.log(sinr_exp.sum(dim=axis) / num_used_res)
        beta_expand2 = expand_to_rank(beta, sinr_eff.dim(), axis=-1)
        # If per_stream is True: [..., num_ut, num_streams_per_ut]
        # If per_stream is False: [..., num_ut]
        sinr_eff = -beta_expand2 * sinr_eff

        # Project sinr_eff within [self._sinr_eff_min, self._sinr_eff_max]
        sinr_eff = torch.clamp(sinr_eff, self._sinr_eff_min, self._sinr_eff_max)

        # Assign a null SINR to users with no assigned resources
        sinr_eff = torch.where(
            num_used_res > 0, sinr_eff, torch.zeros_like(sinr_eff)
        )

        return sinr_eff
