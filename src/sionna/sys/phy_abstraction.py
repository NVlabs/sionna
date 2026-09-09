#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Physical layer abstraction for Sionna SYS"""

import datetime
import json
import logging
import os
import time
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch

from sionna.phy import Block, config, dtypes
from sionna.phy.config import Precision
from sionna.phy.nr.utils import CodedAWGNChannelNR, MCSDecoderNR, TransportBlockNR
from sionna.phy.utils import (
    DeepUpdateDict,
    Interpolate,
    MCSDecoder,
    SingleLinkChannel,
    SplineGriddataInterpolation,
    TransportBlock,
    dict_keys_to_int,
    gather_from_batched_indices,
    lin_to_db,
    scalar_to_shaped_tensor,
    sim_ber,
    to_list,
)
from sionna.sys.effective_sinr import EESM, EffectiveSINR

__all__ = ["PHYAbstraction"]


class PHYAbstraction(Block):
    r"""Class for physical layer abstraction.

    For a given signal-to-interference-plus-noise-ratio (SINR) provided on a
    per-stream basis, and for a given modulation order, coderate, and number of
    coded bits specified for each user, it produces the corresponding number of
    successfully decoded bits, HARQ feedback, effective SINR, block error rate
    (BLER), and transport BLER (TBLER).

    At object instantiation, precomputed BLER tables are loaded and interpolated
    on a fine (SINR, code block size) grid for each modulation and coding scheme
    (MCS) index.

    When the object is called, the post-equalization SINR is first converted to
    an effective SINR. Then, the effective SINR is used to retrieve the BLER from
    pre-computed and interpolated tables. Finally, the BLER determines the TBLER,
    which represents the probability that at least one code block is incorrectly
    received.

    :param interp_fun: Function for interpolating data defined on rectangular or
        unstructured grids, used for BLER and SINR interpolation.
        If `None`, it is set to an instance of
        :class:`~sionna.phy.utils.SplineGriddataInterpolation`.
    :param mcs_decoder_fun: Function mapping MCS indices to modulation order and
        coderate. If `None`, it is set to an instance of
        :class:`~sionna.phy.nr.utils.MCSDecoderNR`.
    :param transport_block_fun: Function computing the number and size (measured
        in bits) of code blocks within a transport block.
        If `None`, it is set to an instance of
        :class:`~sionna.phy.nr.utils.TransportBlockNR`. Custom implementations
        retain the legacy ``(cb_size, num_cb)`` contract; because they do not
        expose ``tb_size``, decoded-bit accounting falls back to
        ``cb_size * num_cb`` for those hooks.
    :param sinr_effective_fun: Function computing the effective SINR.
        If `None`, it is set to an instance of
        :class:`~sionna.sys.EESM`.
    :param load_bler_tables_from: Name of file(s) containing pre-computed
        SINR-to-BLER tables for different categories, table indices, MCS
        indices, SINR and code block sizes. If "default", then the pre-computed
        tables stored in "bler_tables/" folder are loaded.
    :param snr_db_interp_min_max_delta: Tuple of (`min`, `max`, `delta`) values
        [dB] defining the list of SINR [dB] values at which the BLER is
        interpolated, as `min, min+delta, min+2*delta,...,` up until `max`.
    :param cbs_interp_min_max_delta: Tuple of (`min`, `max`, `delta`) values
        defining the list of code block size values at which the BLER and
        SINR are interpolated, as `min, min+delta, min+2*delta,...,max`.
        The default grid extends to the full 5G-NR CBS range, but the built-in
        BLER tables are only simulated up to CBS 2000. With the default
        interpolator, requests above the largest simulated CBS use that
        boundary curve (clamp-to-edge); custom interpolators may differ.
    :param bler_interp_delta: Spacing of the BLER grid at which SINR is
        interpolated.
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`, :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for computation. If `None`,
        :attr:`~sionna.phy.config.Config.device` is used.

    :input mcs_index: [..., num_ut], `torch.int32`.
        MCS index for each user.
    :input sinr: [..., num_ofdm_symbols, num_subcarriers, num_ut, num_streams_per_ut], `torch.float` | `None`.
        Post-equalization SINR in linear scale for each OFDM symbol, subcarrier,
        user and stream.
        If `None`, then ``sinr_eff`` and ``num_allocated_re`` are both required.
    :input sinr_eff: [..., num_ut], `torch.float` | `None`.
        Effective SINR in linear scale for each user.
        If `None`, then ``sinr`` is required.
    :input num_allocated_re: [..., num_ut], `torch.int32` | `None`.
        Number of allocated resources in a slot, computed across OFDM symbols,
        subcarriers and streams, for each user.
        If `None`, then ``sinr`` is required.
    :input mcs_table_index: [..., num_ut], `torch.int32` | `int`.
        MCS table index. Defaults to 1. For further details, refer to the
        :ref:`mcs_table_cat_note`.
    :input mcs_category: [..., num_ut], `torch.int32` | `int`.
        MCS table category. Defaults to 0. For further details, refer to the
        :ref:`mcs_table_cat_note`.
    :input check_mcs_index_validity: `bool`.
        If `True`, a ValueError is thrown if the input MCS indices are not
        valid for the given configuration. Defaults to `True`.

    :output num_decoded_bits: [..., num_ut], `torch.int32`.
        Number of successfully delivered transport-block information bits for
        each user (``tb_size`` on ACK, else 0). This excludes TB/CB CRC bits:
        those CRCs are part of the FEC code-block payload but are not counted
        here, because this output is intended for system-level throughput
        accounting. For a custom ``transport_block_fun`` that only implements
        the legacy two-value return contract, it falls back to
        ``cb_size * num_cb``.
    :output harq_feedback: [..., num_ut], -1 | 0 | 1.
        If 0 (1, resp.), then a NACK (ACK, resp.) is received. If -1, feedback
        is missing since the user is not scheduled for transmission.
    :output sinr_eff: [..., num_ut], `torch.float`.
        Effective SINR in linear scale for each user.
    :output tbler: [..., num_ut], `torch.float`.
        Transport block error rate (BLER) for each user.
    :output bler: [..., num_ut], `torch.float`.
        Block error rate (BLER) for each user.

    .. rubric:: Notes

    In this class, the terms SNR (signal-to-noise ratio) and SINR
    (signal-to-interference-plus-noise ratio) can be used interchangeably.
    This is because the equivalent AWGN model used for BLER mapping does not
    explicitly account for interference.

    When a requested (category, table index, MCS, CBS, SINR) combination has no
    calibrated BLER entry, :meth:`get_bler` returns ``inf``. That value
    propagates to TBLER and yields a deterministic NACK. Link adaptation relies
    on the same convention to skip unavailable MCS candidates.

    .. rubric:: Examples

    .. code-block:: python

        import numpy as np
        from sionna.sys import PHYAbstraction, EESM
        from sionna.phy.nr.utils import MCSDecoderNR, TransportBlockNR
        from sionna.phy.utils import SplineGriddataInterpolation

        # Instantiate the class for BLER and SINR interpolation
        bler_snr_interp_fun = SplineGriddataInterpolation()
        # Instantiate the class for mapping MCS to modulation order and coderate
        # in 5G NR
        mcs_decoder_fun = MCSDecoderNR()
        # Instantiate the class for computing the number and size of code blocks
        # within a transport block in 5G NR
        transport_block_fun = TransportBlockNR()
        # Instantiate the class for computing the effective SINR
        sinr_effective_fun = EESM()

        # By instantiating a PHYAbstraction object, precomputed BLER tables are
        # loaded and interpolated on a fine (SINR, code block size) grid for each MCS
        phy_abs = PHYAbstraction(
            interp_fun=bler_snr_interp_fun,
            mcs_decoder_fun=mcs_decoder_fun,
            transport_block_fun=transport_block_fun,
            sinr_effective_fun=sinr_effective_fun)

        # Plot a BLER table
        phy_abs.plot(plot_subset={'category': {0: {'index': {1: {'MCS': 14}}}}},
                        show=True);

    .. figure:: ../figures/category0_table1_mcs14.png
        :align: center
        :width: 70%

    .. code-block:: python

        # One can also compute new BLER tables
        # SINR values and code block sizes @ new simulations are performed
        snr_dbs = np.linspace(-5, 25, 5)
        cb_sizes = np.arange(24, 8448, 1000)
        # MCS values @ new simulations are performed
        sim_set = {'category': {
            0:
            {'index': {
                1: {'MCS': [15]}
            }}}}

        # Compute new tables
        new_table = phy_abs.new_bler_table(
            snr_dbs,
            cb_sizes,
            sim_set,
            max_mc_iter=15,
            batch_size=10,
            verbose=True)
    """

    def __init__(
        self,
        interp_fun: Optional[Interpolate] = None,
        mcs_decoder_fun: Optional[MCSDecoder] = None,
        transport_block_fun: Optional[TransportBlock] = None,
        sinr_effective_fun: Optional[EffectiveSINR] = None,
        load_bler_tables_from: Union[str, List[str]] = "default",
        snr_db_interp_min_max_delta: Tuple[float, float, float] = (-5, 30.01, 0.1),
        cbs_interp_min_max_delta: Tuple[int, int, int] = (24, 8448, 100),
        bler_interp_delta: float = 0.01,
        precision: Optional[Precision] = None,
        device: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(precision=precision, device=device)

        # Set default values
        if interp_fun is None:
            interp_fun = SplineGriddataInterpolation()
        if mcs_decoder_fun is None:
            mcs_decoder_fun = MCSDecoderNR(precision=precision, device=device)
        if transport_block_fun is None:
            transport_block_fun = TransportBlockNR(precision=precision, device=device)
        if sinr_effective_fun is None:
            sinr_effective_fun = EESM(precision=precision, device=device)

        # Check inputs
        if not isinstance(interp_fun, Interpolate):
            raise ValueError(
                "interp_fun must be an instance of sionna.phy.utils.Interpolate"
            )
        if not isinstance(mcs_decoder_fun, MCSDecoder):
            raise ValueError(
                "mcs_decoder_fun must be an instance of sionna.phy.utils.MCSDecoder"
            )
        if not isinstance(transport_block_fun, TransportBlock):
            raise ValueError(
                "transport_block_fun must be a subclass of sionna.phy.utils.TransportBlock"
            )
        if not isinstance(sinr_effective_fun, EffectiveSINR):
            raise ValueError(
                "sinr_effective_fun must be a subclass of sionna.sys.EffectiveSINR"
            )

        # -------------- #
        # Initialization #
        # -------------- #
        self._kwargs = kwargs
        self._bler_table: Optional[DeepUpdateDict] = None
        self._bler_table_interp: Optional[torch.Tensor] = None
        self._snr_table_interp: Optional[torch.Tensor] = None

        # ------------- #
        # Instantiation #
        # ------------- #
        # Function interpolating (CBS, SNR) -> BLER
        self._bler_interp_fun = interp_fun.struct
        # Function interpolating (CBS, BLER) -> SNR
        self._snr_interp_fun = interp_fun.unstruct
        # Function mapping MCS index to modulation order and coderate
        self._mcs_decoder_fun = mcs_decoder_fun
        # Function computing number and size of code blocks
        self._transport_block_fun = transport_block_fun
        # Resolve TB information-bit accounting once so the compiled call path
        # does not need an isinstance guard.
        if isinstance(transport_block_fun, TransportBlockNR):
            self._transport_block_info = transport_block_fun.transport_block_size
        else:

            def _transport_block_info(
                modulation_order, target_coderate, num_coded_bits, **tb_kwargs
            ):
                cb_size, num_cb = transport_block_fun(
                    modulation_order, target_coderate, num_coded_bits, **tb_kwargs
                )
                return num_cb * cb_size, cb_size, num_cb

            self._transport_block_info = _transport_block_info
        # Function computing the effective SINR
        self._sinr_effective_fun = sinr_effective_fun

        # Interpolation grid
        self._cbs_interp: Optional[np.ndarray] = None
        self._snr_dbs_interp: Optional[np.ndarray] = None
        self._blers_interp: Optional[np.ndarray] = None

        if load_bler_tables_from == "default":
            filenames = [
                "bler_tables/PUSCH_table1.json",
                "bler_tables/PUSCH_table2.json",
                "bler_tables/PDSCH_table1.json",
                "bler_tables/PDSCH_table2.json",
                "bler_tables/PDSCH_table3.json",
                "bler_tables/PDSCH_table4.json",
            ]
            self.bler_table_filenames = [
                os.path.join(os.path.dirname(os.path.abspath(__file__)), f)
                for f in filenames
            ]
        else:
            self.bler_table_filenames = load_bler_tables_from

        # SNR/BLER interpolation grid
        self.snr_db_interp_min_max_delta = snr_db_interp_min_max_delta
        self.cbs_interp_min_max_delta = cbs_interp_min_max_delta
        self.bler_interp_delta = bler_interp_delta

    @staticmethod
    def load_table(filename: str) -> Dict:
        r"""Loads a table stored in JSON file.

        :param filename: Name of the JSON file containing the table

        :output table: Table loaded from file
        """
        with open(filename, "r", encoding="utf-8") as f:
            table = json.load(f, object_hook=dict_keys_to_int)
        return table

    # ----------- #
    # BLER tables #
    # ----------- #
    @property
    def bler_table_filenames(self) -> List[str]:
        r"""`str` | list of `str`: Get/set the absolute path name of the files
        containing BLER tables.
        """
        return self._bler_table_filenames

    @bler_table_filenames.setter
    def bler_table_filenames(self, value: Union[str, List[str]]) -> None:
        self._bler_table_filenames = to_list(value)
        # Load the table
        self._bler_table = DeepUpdateDict({"category": {}})
        for f in self._bler_table_filenames:
            try:
                with open(f, "r", encoding="utf-8") as file:
                    bler_subtable = json.load(file, object_hook=dict_keys_to_int)
                    # Merge with the existing one
                    self._bler_table.deep_update(
                        bler_subtable, stop_at_keys=("CBS", "SNR_db")
                    )
            except FileNotFoundError:
                warnings.warn(
                    f"BLER table file '{f}' does not exist. Skipping...",
                    UserWarning,
                    stacklevel=2,
                )
        if self._bler_table == {}:
            warnings.warn(
                "No BLER table found. You can generate them via "
                "PHYAbstraction.new_bler_table method.",
                UserWarning,
                stacklevel=2,
            )
        # Check table validity
        self.validate_bler_table()

    def _get_batch_size_interp_mat(self) -> List[int]:
        """Compute the batch size of interpolation tensors."""
        categories = list(self._bler_table["category"].keys())
        max_table_idx_list, max_mcs_list = [], []
        for ch in categories:
            table_idx_list = list(self._bler_table["category"][ch]["index"].keys())
            max_table_idx_list.append(max(table_idx_list))
            for table_idx in table_idx_list:
                mcs_list = list(
                    self._bler_table["category"][ch]["index"][table_idx]["MCS"].keys()
                )
                max_mcs_list.append(max(mcs_list))
        if (
            len(categories) > 0
            and len(max_table_idx_list) > 0
            and len(max_mcs_list) > 0
        ):
            return [max(categories) + 1, max(max_table_idx_list), max(max_mcs_list) + 1]
        return [0, 0, 0]

    @property
    def bler_table(self) -> Dict:
        r"""`dict` (read-only): Collection of tables containing BLER values for
        different values of SNR, MCS table, MCS index and CB size.
        ``bler_table['category'][cat]['index'][mcs_table_index]['MCS'][mcs]['CBS'][cb_size]``
        contains the lists of BLER values.
        ``bler_table['category'][cat]['index'][mcs_table_index]['MCS'][mcs]['SNR_db']``
        contains the list of SNR values.
        ``bler_table['category'][cat]['index'][mcs_table_index]['MCS'][mcs]['EbN0_db']``
        contains the list of :math:`E_b/N_0` values.
        """
        return self._bler_table

    # ------------------- #
    # Interpolated tables #
    # ------------------- #
    @property
    def bler_table_interp(self) -> torch.Tensor:
        r"""[n_categories, n_tables, n_mcs, n_cbs_index, n_snr], `torch.float`
        (read-only): Tensor containing BLER values interpolated across SINR and
        CBS values, for different categories and MCS table indices. The first
        axis accounts for the category, e.g., 'PDSCH' or 'PUSCH' in 5G-NR, the
        second axis corresponds to the 38.214 MCS table index while the third
        axis carries the MCS index.
        """
        return self._bler_table_interp

    @property
    def snr_table_interp(self) -> torch.Tensor:
        r"""[n_categories, n_tables, n_mcs, n_cbs_index, n_bler], `torch.float`
        (read-only): Tensor containing SINR values interpolated across BLER and
        CBS values, for different categories and MCS table indices. The first
        axis accounts for the category, e.g., 'PDSCH' or 'PUSCH' in 5G-NR, the
        second axis corresponds to the 38.214 MCS table index and the third axis
        accounts for the MCS index.
        """
        return self._snr_table_interp

    # ------------------ #
    # Interpolation grid #
    # ------------------ #
    @property
    def snr_db_interp_min_max_delta(self) -> Tuple[float, float, float]:
        r"""[3], `tuple`: Get/set the tuple of (`min`, `max`, `delta`) values
        [dB] defining the list of SINR values at which the BLER is
        interpolated, as `min, min+delta, min+2*delta,...,` up until `max`.
        """
        return self._snr_db_interp_min_max_delta

    @snr_db_interp_min_max_delta.setter
    def snr_db_interp_min_max_delta(self, value: Tuple[float, float, float]) -> None:
        if hasattr(value, "__len__") and len(value) == 3:
            self._snr_db_interp_min_max_delta = value
        else:
            raise ValueError("snr_db_interp_min_max_delta must have length 3")
        self._snr_dbs_interp = np.arange(
            self._snr_db_interp_min_max_delta[0],
            self._snr_db_interp_min_max_delta[1],
            self._snr_db_interp_min_max_delta[2],
        )

        if (self._bler_table is not None) and (self._cbs_interp is not None):
            # Interpolate BLER
            self._interpolate_bler()

    @property
    def cbs_interp_min_max_delta(self) -> Tuple[int, int, int]:
        r"""[3], `tuple`: Get/set the tuple of (`min`, `max`, `delta`) values
        defining the list of code block size values at which the BLER and SINR
        are interpolated, as `min, min+delta, min+2*delta,...,` up until `max`.
        """
        return self._cbs_interp_min_max_delta

    @cbs_interp_min_max_delta.setter
    def cbs_interp_min_max_delta(self, value: Tuple[int, int, int]) -> None:
        if hasattr(value, "__len__") and len(value) == 3:
            self._cbs_interp_min_max_delta = value
        else:
            raise ValueError("cbs_interp_min_max_delta must have length 3")
        self._cbs_interp = np.arange(
            self._cbs_interp_min_max_delta[0],
            self._cbs_interp_min_max_delta[1],
            self._cbs_interp_min_max_delta[2],
        )
        if self._bler_table is not None:
            if self._blers_interp is not None:
                # Interpolate SNR
                self._interpolate_snr()
            if self._snr_dbs_interp is not None:
                # Interpolate BLER
                self._interpolate_bler()

    @property
    def bler_interp_delta(self) -> float:
        r"""`float`: Get/set the spacing of the BLER grid at which SINR is
        interpolated."""
        return self._bler_interp_delta

    @bler_interp_delta.setter
    def bler_interp_delta(self, value: float) -> None:
        self._bler_interp_delta = value
        self._blers_interp = np.arange(0, 1, self._bler_interp_delta)
        if (self._bler_table is not None) and (self._cbs_interp is not None):
            # Interpolate SNR
            self._interpolate_snr()

    def get_idx_from_grid(self, val: torch.Tensor, which: str) -> torch.Tensor:
        r"""Retrieves the index of a SINR or CBS value in the interpolation grid.

        :param val: Values to be quantized
        :param which: Whether the values are SNR (equivalent to SINR) or CBS.
            Must be "snr" or "cbs".

        :output idx: Index of the values in the interpolation grid
        """
        if which not in ("snr", "cbs"):
            raise ValueError("which must be 'snr' or 'cbs'")
        if which == "snr":
            len_grid = len(self._snr_dbs_interp)
            min_max_delta = self._snr_db_interp_min_max_delta
        else:
            len_grid = len(self._cbs_interp)
            min_max_delta = self._cbs_interp_min_max_delta
        min_grid = min_max_delta[0]
        delta_grid = min_max_delta[2]
        val = torch.as_tensor(val, dtype=self.dtype, device=self.device)
        idx = torch.round((val - min_grid) / delta_grid).to(torch.int32)
        idx = torch.clamp(idx, 0, len_grid - 1)
        return idx

    # ------------- #
    # Retrieve BLER #
    # ------------- #
    def get_bler(
        self,
        mcs_index: Union[int, torch.Tensor],
        mcs_table_index: Union[int, torch.Tensor],
        mcs_category: Union[int, torch.Tensor],
        cb_size: Union[int, torch.Tensor],
        snr_eff: torch.Tensor,
    ) -> torch.Tensor:
        r"""Retrieves from interpolated tables the BLER corresponding to a certain
        table index, MCS, CB size, and SINR values provided as input.
        If the corresponding interpolated table is not available, it returns
        `Inf`.

        :param mcs_index: MCS index for each user
        :param mcs_table_index: MCS table index for each user.
            For further details, refer to the :ref:`mcs_table_cat_note`.
        :param mcs_category: MCS table category for each user.
            For further details, refer to the :ref:`mcs_table_cat_note`.
        :param cb_size: Code block size for each user
        :param snr_eff: Effective SINR for each user

        :output bler: BLER corresponding to the input channel type, table index,
            MCS, CB size and SINR, retrieved from internal interpolation tables.
            Missing table entries are returned as ``inf``.
        """
        # Cast inputs to appropriate type and shape
        if not isinstance(snr_eff, torch.Tensor):
            snr_eff = torch.tensor(snr_eff, dtype=self.dtype, device=self.device)
        snr_eff = snr_eff.to(self.dtype)
        shape = list(snr_eff.shape)
        mcs_category = scalar_to_shaped_tensor(
            mcs_category, torch.int32, shape, device=self.device
        )
        mcs_index = scalar_to_shaped_tensor(
            mcs_index, torch.int32, shape, device=self.device
        )
        mcs_table_index = scalar_to_shaped_tensor(
            mcs_table_index, torch.int32, shape, device=self.device
        )
        cb_size = scalar_to_shaped_tensor(cb_size, torch.int32, shape, device=self.device)

        # Convert SNR to dB
        snr_eff_db = lin_to_db(snr_eff, precision=self.precision)

        # Quantize the SNR [dB] and CBS to the corresponding interpolation index
        snr_db_idx = self.get_idx_from_grid(snr_eff_db, "snr")
        cbs_idx = self.get_idx_from_grid(cb_size.to(self.dtype), "cbs")

        # Stack indices to extract BLER from interpolated table
        idx = torch.stack(
            [mcs_category, mcs_table_index - 1, mcs_index, cbs_idx, snr_db_idx], dim=-1
        )

        # Compute BLER
        bler = gather_from_batched_indices(self._bler_table_interp, idx)
        bler = bler.to(self.dtype)

        return bler

    def call(
        self,
        mcs_index: torch.Tensor,
        sinr: Optional[torch.Tensor] = None,
        sinr_eff: Optional[torch.Tensor] = None,
        num_allocated_re: Optional[torch.Tensor] = None,
        mcs_table_index: Union[int, torch.Tensor] = 1,
        mcs_category: Union[int, torch.Tensor] = 0,
        check_mcs_index_validity: bool = True,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute PHY abstraction outputs.

        Refer to the class docstring for the Input/Output specification.
        """
        if not (
            (sinr is not None)
            ^ ((sinr_eff is not None) and (num_allocated_re is not None))
        ):
            raise ValueError(
                "Exactly one of 'sinr' or the pair "
                "('sinr_eff', 'num_allocated_re') is required as input"
            )

        if sinr is not None:
            # Total number of allocated streams across all resource elements
            # [..., num_ut]
            num_allocated_re = (sinr > 0).to(torch.int32).sum(dim=(-4, -3, -1))

            # Effective SINR
            # [..., num_ut]
            sinr_eff = self._sinr_effective_fun(
                sinr,
                mcs_index=mcs_index,
                mcs_table_index=mcs_table_index,
                mcs_category=mcs_category,
                per_stream=False,
                **kwargs,
            )
        else:
            sinr_eff = sinr_eff.to(self.dtype)
            num_allocated_re = num_allocated_re.to(torch.int32)

        # Whether a user is scheduled
        # [..., num_ut]
        ut_is_scheduled = num_allocated_re > 0

        # Convert MCS index to modulation order and coderate
        # [..., num_ut]
        modulation_order, target_coderate = self._mcs_decoder_fun(
            mcs_index,
            mcs_table_index,
            mcs_category,
            check_index_validity=check_mcs_index_validity,
            **kwargs,
        )

        # Compute the number of coded bits
        num_coded_bits = modulation_order * num_allocated_re

        # Compute the transport-block information bits and its CB segmentation
        # [..., num_ut]
        # Custom TransportBlock hooks retain the abstract (cb_size, num_cb)
        # contract and fall back to CRC-inclusive accounting (see __init__).
        tb_size, cb_size, num_cb = self._transport_block_info(
            modulation_order, target_coderate, num_coded_bits, **kwargs
        )

        # Retrieve the BLER from the stored tables
        # [..., num_ut]
        # Missing rows are +inf. This propagates to TBLER=+inf and therefore
        # yields a deterministic NACK below.
        bler = self.get_bler(
            mcs_index, mcs_table_index, mcs_category, cb_size, sinr_eff
        )

        # Compute TBLER = Pr(at least a CB is incorrectly received)
        # [..., num_ut]
        one = torch.tensor(1.0, dtype=bler.dtype, device=self.device)
        tbler_finite = one - torch.pow(one - bler, num_cb.to(bler.dtype))
        # Preserve the missing-row +inf sentinel for every num_cb parity.
        tbler = torch.where(torch.isinf(bler), bler, tbler_finite)

        # Set BLER=-1 and TBLER=-1 for non-scheduled UTs
        minus_one = torch.tensor(-1.0, dtype=bler.dtype, device=self.device)
        bler = torch.where(ut_is_scheduled, bler, minus_one)
        tbler = torch.where(ut_is_scheduled, tbler, minus_one)

        # HARQ feedback
        # Prefer rand_like under torch.compile: torch.rand(symbolic_size) fails
        # fake-tensor propagation (SymIntArrayRef) after Dynamo graph breaks.
        if torch.compiler.is_compiling():
            rnd = torch.rand_like(tbler)
        else:
            rnd = torch.rand(
                tbler.shape,
                dtype=self.dtype,
                device=self.device,
                generator=self.torch_rng,
            )
        harq_feedback = torch.where(
            rnd < tbler,
            torch.tensor(0, dtype=torch.int32, device=self.device),
            torch.tensor(1, dtype=torch.int32, device=self.device),
        )

        # Successfully delivered TB information bits for each user
        # [..., num_ut]
        num_decoded_bits = harq_feedback * tb_size
        num_decoded_bits = torch.where(
            ut_is_scheduled,
            num_decoded_bits,
            torch.tensor(0, dtype=torch.int32, device=self.device),
        )

        # Assign HARQ=-1 for the non-scheduled UTs
        harq_feedback = torch.where(
            ut_is_scheduled,
            harq_feedback,
            torch.tensor(-1, dtype=torch.int32, device=self.device),
        )

        return num_decoded_bits, harq_feedback, sinr_eff, tbler, bler

    # -------------------- #
    # Interpolation method #
    # -------------------- #
    def _interpolate_bler(self) -> None:
        """Interpolates the BLER over a fine (CBS, SINR) grid."""
        if self._bler_table is None:
            raise ValueError(
                "BLER table is not provided; Interpolation cannot be performed"
            )
        if self._cbs_interp is None:
            raise ValueError(
                "CBS interpolation grid is not provided; Interpolation cannot be performed"
            )

        interp_batch_size = self._get_batch_size_interp_mat()

        # [num_category, num_table_idx, num_mcs_index, num_cbs_interp, num_snr_interp]
        self._bler_table_interp_np = np.full(
            interp_batch_size + [len(self._cbs_interp), len(self._snr_dbs_interp)],
            np.inf,
        )

        for category in self._bler_table["category"]:
            for table_idx in self._bler_table["category"][category]["index"]:
                table_mcs = self._bler_table["category"][category]["index"][table_idx][
                    "MCS"
                ]

                for mcs in table_mcs:
                    cbs_vec = list(table_mcs[mcs]["CBS"].keys())
                    snr_vec = table_mcs[mcs]["SNR_db"]

                    # Collect BLER values
                    bler_val = np.zeros((len(cbs_vec), len(snr_vec)))
                    for idx_cbs, cbs in enumerate(cbs_vec):
                        bler_val[idx_cbs, :] = table_mcs[mcs]["CBS"][cbs]["BLER"]

                    # Interpolate BLER
                    try:
                        bler_interp = self._bler_interp_fun(
                            bler_val,
                            cbs_vec,
                            snr_vec,
                            self._cbs_interp,
                            self._snr_dbs_interp,
                            **self._kwargs,
                        )
                    except ValueError as e:
                        warnings.warn(
                            f"SINR-to-BLER interpolation failed for "
                            f"category {category}, "
                            f"index {table_idx}, MCS {mcs}.\nError: {e}",
                            RuntimeWarning,
                            stacklevel=2,
                        )
                        continue

                    # Ensure BLER is within 0 and 1
                    bler_interp = np.clip(bler_interp, 0, 1)

                    # Store it
                    self._bler_table_interp_np[
                        category, table_idx - 1, mcs, :, :
                    ] = bler_interp

        # Convert to tensor
        self._bler_table_interp = torch.tensor(
            self._bler_table_interp_np, dtype=self.dtype, device=self.device
        )

    def _interpolate_snr(self) -> None:
        """Interpolates the SINR table over a fine (CBS, BLER) grid."""
        if self._bler_table is None:
            raise ValueError(
                "BLER table is not provided; Interpolation cannot be performed"
            )
        if self._blers_interp is None:
            raise ValueError(
                "BLER interpolation grid is not provided; Interpolation cannot be performed"
            )

        interp_batch_size = self._get_batch_size_interp_mat()
        self._snr_table_interp_np = np.full(
            interp_batch_size + [len(self._cbs_interp), len(self._blers_interp)], np.inf
        )

        for category in self._bler_table["category"]:
            for table_index in self._bler_table["category"][category]["index"]:
                table_mcs = self._bler_table["category"][category]["index"][table_index][
                    "MCS"
                ]

                for mcs in table_mcs:
                    # Collect values of SNR, CBS and BLER in arrays
                    snr_vec = table_mcs[mcs]["SNR_db"]
                    cbs_vec = list(table_mcs[mcs]["CBS"].keys())
                    snr_vec_tile = np.tile(snr_vec, len(cbs_vec))
                    cbs_vec_rep = np.repeat(cbs_vec, len(snr_vec))
                    bler_vec = [
                        bler
                        for cbs in cbs_vec
                        for bler in table_mcs[mcs]["CBS"][cbs]["BLER"]
                    ]
                    try:
                        # Interpolate the SNR as a function of CBS and BLER
                        snr_interp = self._snr_interp_fun(
                            snr_vec_tile,
                            cbs_vec_rep,
                            bler_vec,
                            self._cbs_interp,
                            self._blers_interp,
                            **self._kwargs,
                        )
                    except ValueError as e:
                        warnings.warn(
                            f"BLER-to-SINR interpolation failed for "
                            f"category {category}, "
                            f"index {table_index}, MCS {mcs}.\n"
                            f"Error message: {e}",
                            RuntimeWarning,
                            stacklevel=2,
                        )
                        continue
                    self._snr_table_interp_np[
                        category, table_index - 1, mcs, :, :
                    ] = snr_interp

        # Convert to tensor
        self._snr_table_interp = torch.tensor(
            self._snr_table_interp_np, dtype=self.dtype, device=self.device
        )

    def validate_bler_table(self) -> bool:
        r"""Validates the dictionary structure of ``self.bler_table``.

        :output is_valid: `True` if ``self.bler_table`` has a valid structure

        :raises ValueError: If the structure is invalid
        """
        if not isinstance(self._bler_table, dict):
            raise ValueError("Must be a dictionary")

        if np.any(np.array(list(self._bler_table["category"].keys())) < 0):
            raise ValueError("Categories must be nonnegative integers")

        for _, bler_table_tmp in self._bler_table["category"].items():
            if set(bler_table_tmp.keys()) != {"index"}:
                raise ValueError("Key must be 'index'")

            if np.any(np.array(list(bler_table_tmp["index"].keys())) < 1):
                raise ValueError("Table indices must be positive integers")

            for table_index in bler_table_tmp["index"]:
                if set(bler_table_tmp["index"][table_index].keys()) != {"MCS"}:
                    raise ValueError("Key must be 'MCS'")

                if np.any(
                    np.array(
                        list(bler_table_tmp["index"][table_index]["MCS"].keys())
                    )
                    < 0
                ):
                    raise ValueError("MCS indices must be nonnegative integers")

                for mcs in bler_table_tmp["index"][table_index]["MCS"]:
                    if set(
                        bler_table_tmp["index"][table_index]["MCS"][mcs].keys()
                    ) != {"CBS", "SNR_db"}:
                        raise ValueError("Keys must be ['CBS', 'SNR_db']")

                    for cbs in bler_table_tmp["index"][table_index]["MCS"][mcs]["CBS"]:
                        if (
                            set(
                                bler_table_tmp["index"][table_index]["MCS"][mcs]["CBS"][
                                    cbs
                                ].keys()
                            )
                            != {"BLER"}
                        ):
                            raise ValueError("Keys must be 'BLER'")
        return True

    def plot(
        self,
        plot_subset: Union[str, Dict] = "all",
        show: bool = True,
        save_path: Optional[str] = None,
    ) -> List[str]:
        r"""Visualizes and/or saves to file the SINR-to-BLER tables.

        :param plot_subset: Dictionary containing the list of MCS indices to
            consider, stored at
            ``plot_subset['category'][category]['index'][mcs_table_index]['MCS']``.
            If "all", then plots are produced for all available BLER tables.
        :param show: If `True`, then plots are visualized. Defaults to `True`.
        :param save_path: Folder path where BLER plots are saved. If `None`,
            then plots are not saved.

        :output fignames: List of names of files containing BLER plots
        """
        if self._bler_table is None:
            raise ValueError(
                "Plots cannot be produced as self.bler_table has not been loaded or computed"
            )

        # Create folder if it does not exist
        if save_path is not None:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
                logging.info("\nCreated folder %s\n", save_path)

        fignames = []
        if plot_subset == "all":
            plot_subset = self._bler_table

        for category, plot_subset_cat in plot_subset["category"].items():
            for table_index, plot_subset_cat_tab in plot_subset_cat["index"].items():
                for mcs in to_list(plot_subset_cat_tab["MCS"]):
                    try:
                        num_bits_per_symbol, coderate = self._mcs_decoder_fun(
                            mcs, table_index, category, **self._kwargs
                        )
                        if isinstance(num_bits_per_symbol, torch.Tensor):
                            num_bits_per_symbol = num_bits_per_symbol.item()
                        if isinstance(coderate, torch.Tensor):
                            coderate = coderate.item()
                    except (ValueError, AssertionError) as e:
                        print(
                            f"Invalid (category={category}, "
                            f"index={table_index}, "
                            f"MCS={mcs}) combination. \n"
                            f"Error message: {e}\n"
                            f"Skipping...\n"
                        )
                        continue

                    # SNR value at the channel capacity for the spectral
                    # efficiency associated to the current MCS
                    snr_shannon = 2 ** (num_bits_per_symbol * coderate) - 1
                    snr_shannon_db = 10 * np.log10(snr_shannon)

                    try:
                        snr_dbs = self._bler_table["category"][category]["index"][
                            table_index
                        ]["MCS"][mcs]["SNR_db"]

                        fig, ax = plt.subplots()
                        for cbs in self._bler_table["category"][category]["index"][
                            table_index
                        ]["MCS"][mcs]["CBS"]:
                            bler = self._bler_table["category"][category]["index"][
                                table_index
                            ]["MCS"][mcs]["CBS"][cbs]["BLER"]
                            ax.semilogy(snr_dbs, bler, label=f"code block size={cbs}")

                        ax.plot(
                            [snr_shannon_db] * 2,
                            ax.get_ylim(),
                            "--k",
                            label="SNR @capacity",
                        )
                        ax.set_title(
                            f"MCS index {mcs} (table category {category}, "
                            f"index {table_index})"
                        )
                        ax.legend()
                        ax.grid(True)
                        ax.set_xlabel("SNR [dB]")
                        ax.set_ylabel("BLER")

                        # Save to file
                        if save_path is not None:
                            figname = (
                                f"category{category}_table"
                                f"{table_index}_mcs{mcs}.png"
                            )
                            figname = os.path.join(save_path, figname)
                            fig.savefig(figname)
                            fignames.append(figname)

                        if show:
                            plt.show()
                        plt.close(fig)
                    except KeyError as e:
                        print(
                            f"\nBLER for (category={category}, index="
                            f"{table_index}, MCS={mcs}) not available "
                            f"for plotting. Error message: {e}."
                            "\nSkipping..."
                        )

        return fignames

    def new_bler_table(
        self,
        snr_dbs: Union[List[float], np.ndarray],
        cb_sizes: Union[List[int], np.ndarray],
        sim_set: Dict,
        channel: Optional[SingleLinkChannel] = None,
        filename: Optional[str] = None,
        write_mode: str = "w",
        batch_size: int = 1000,
        max_mc_iter: int = 100,
        target_bler: Optional[float] = None,
        compile_mode: Optional[str] = None,
        early_stop: bool = True,
        filename_log: Optional[str] = None,
        verbose: bool = True,
    ) -> Dict:
        r"""Computes static tables mapping SNR values of an AWGN channel to the
        corresponding block error rate (BLER) via Monte-Carlo simulations for
        different MCS indices, code block sizes and channel types.
        Note that the newly computed table is merged with the internal
        ``self.bler_table``.

        The simulation continues with the next SNR point after
        ``max_mc_iter`` batches of size ``batch_size`` have been simulated.
        Early stopping allows to stop the simulation after the first error-free
        SNR point or after reaching a certain ``target_bler``.
        For more details, please see :func:`~sionna.phy.utils.misc.sim_ber`.

        :param snr_dbs: List of SNR [dB] value(s) at which the BLER is computed
        :param cb_sizes: List of code block (CB) size(s) at which the BLER is
            computed
        :param sim_set: Dictionary containing the list of MCS indices at which
            the BLER is computed via simulation. The dictionary structure is of
            the kind:
            ``sim_set['category'][category]['index'][mcs_table_index]['MCS'][mcs_list]``.
        :param channel: Object for simulating single-link, i.e., single-carrier
            and single-stream, channels. If `None`, it is set to an instance of
            :class:`~sionna.phy.nr.utils.CodedAWGNChannelNR`.
        :param filename: Name of JSON file where the BLER tables are saved.
            If `None`, results are not saved.
        :param write_mode: If 'w', then ``filename`` is rewritten.
            If 'a', then the produced results are appended to ``filename``.
        :param batch_size: Batch size for Monte-Carlo BLER simulations
        :param max_mc_iter: Maximum number of Monte-Carlo iterations per SNR
            point
        :param target_bler: The simulation stops after the first SNR point
            which achieves a lower block error rate as specified by
            ``target_bler``. This requires ``early_stop`` to be `True`.
        :param compile_mode: Execution mode of channel call method.
            If `None`, channel is executed as is.
        :param early_stop: If `True`, the simulation stops after the first
            error-free SNR point.
        :param filename_log: Name of logging file. If `None`, logs are not
            produced.
        :param verbose: If `True`, the simulation progress is visualized, as
            well as the names of files of results and figures.

        :output new_table: Newly computed BLER table
        """

        def _log(msg: str, level: str = "info") -> None:
            """Logging and printing simulation progress."""
            if verbose:
                print(msg)
            if filename_log is not None:
                if level == "info":
                    logging.info(msg)
                elif level == "warning":
                    logging.warning(msg)
                elif level == "error":
                    logging.error(msg)
                else:
                    raise ValueError("unrecognized 'level' input")

        if channel is None:
            channel = CodedAWGNChannelNR(precision=self.precision, device=self.device)

        # Check input validity
        if not isinstance(channel, SingleLinkChannel):
            raise ValueError(
                "'channel' must be an instance of sionna.phy.utils.SingleLinkChannel"
            )

        if filename_log is not None:
            # Logging settings
            logging.basicConfig(
                filename=filename_log,
                filemode="w",
                level=logging.INFO,
                format="%(asctime)s - %(levelname)s - %(message)s",
            )

        if write_mode not in ["a", "w"]:
            raise ValueError(
                "'write_mode' must be either 'a' for appending to file or 'w' for rewriting"
            )

        snr_dbs = to_list(snr_dbs)
        cb_sizes = to_list(cb_sizes)

        if (
            (filename is not None)
            and os.path.isfile(filename)
            and (write_mode == "a")
        ):
            # Append to file
            new_table = self.load_table(filename)
        else:
            # (Re)write to file
            new_table = {"category": {}}

        # Total number of points to simulate
        n_sims_tot = len(cb_sizes) * sum(
            len(sim_set["category"][cat]["index"][ti]["MCS"])
            for cat in sim_set["category"]
            for ti in sim_set["category"][cat]["index"]
        )

        _log(
            "\nBLER simulations started. "
            "\nTotal # (category, index, MCS, SINR) "
            f"points to simulate: {n_sims_tot}\n"
        )

        # ---------------- #
        # BLER simulations #
        # ---------------- #
        n_sims_done = 0
        time_start = time.time()

        for category, sim_set_cat in sim_set["category"].items():
            if category not in new_table["category"]:
                new_table["category"][category] = {"index": {}}

            for table_index, sim_set_tab in sim_set_cat["index"].items():
                if table_index not in new_table["category"][category]["index"].keys():
                    new_table["category"][category]["index"][table_index] = {"MCS": {}}

                for mcs in sim_set_tab["MCS"]:
                    # Compute modulation order and code-rate associated with the
                    # MCS index, table index and channel type
                    try:
                        num_bits_per_symbol, coderate = self._mcs_decoder_fun(
                            mcs, table_index, category, **self._kwargs
                        )
                        if isinstance(num_bits_per_symbol, torch.Tensor):
                            num_bits_per_symbol = num_bits_per_symbol.item()
                        if isinstance(coderate, torch.Tensor):
                            coderate = coderate.item()
                    except (ValueError, AssertionError) as e:
                        _log(
                            f"Invalid (category={category}, "
                            f"index={table_index}, "
                            f"MCS={mcs}) combination. \n"
                            f"Error message: {e}\n"
                            f"Skipping...\n",
                            level="warning",
                        )
                        continue

                    # Eb/N0, where Eb=energy per information (uncoded) bit
                    ebno_dbs = [
                        x - 10 * np.log10(num_bits_per_symbol * coderate)
                        for x in snr_dbs
                    ]

                    # N. successful simulations for the MCS
                    n_sims_mcs = 0

                    for cbs in cb_sizes:
                        _log(
                            f"\nSimulating category={category}, "
                            f"index={table_index}, "
                            f"CBS={cbs}, MCS={mcs}...\n"
                        )

                        try:
                            # Instantiate the AWGN coded channel
                            channel.num_bits_per_symbol = num_bits_per_symbol
                            channel.num_info_bits = cbs
                            channel.target_coderate = coderate

                            # Compute BLER via Monte-Carlo simulations
                            _, bler = sim_ber(
                                channel,
                                ebno_dbs,
                                batch_size,
                                max_mc_iter,
                                soft_estimates=False,
                                early_stop=early_stop,
                                target_bler=target_bler,
                                compile_mode=compile_mode,
                                forward_keyboard_interrupt=True,
                                verbose=verbose,
                                precision=self.precision,
                                device=self.device,
                            )

                            n_sims_mcs += 1
                            if n_sims_mcs == 1:
                                # Initialize dictionary
                                new_table["category"][category]["index"][table_index][
                                    "MCS"
                                ][mcs] = {"CBS": {}, "SNR_db": snr_dbs}

                            # Record results in bler_table
                            new_table["category"][category]["index"][table_index][
                                "MCS"
                            ][mcs]["CBS"][cbs] = {"BLER": bler.cpu().numpy().tolist()}

                            # Write to JSON file
                            if filename is not None:
                                with open(filename, "w", encoding="utf-8") as file:
                                    json.dump(new_table, file, indent=6)
                                    _log(
                                        f"\nResults written in file "
                                        f"{os.path.abspath(filename)}\n"
                                    )

                        except (ValueError, RuntimeError) as e:
                            _log(
                                f"\nBER/BLER simulations failed for "
                                f"(category={category}, "
                                f"index={table_index}, "
                                f"CBS={cbs}, MCS={mcs})\n"
                                f"Error message: {e}\n"
                                f"Skipping...\n",
                                level="error",
                            )

                        n_sims_done += 1

                        # Compute simulation progress and remaining execution time
                        progress = n_sims_done / n_sims_tot * 100
                        time_s = (time.time() - time_start) * (100 - progress) / progress
                        time_s = datetime.timedelta(seconds=time_s)
                        time_hms = str(time_s).split(".", maxsplit=1)[0]
                        _log(
                            f"\nPROGRESS: {progress:.2f}%\n"
                            f"Estimated remaining time: "
                            f"{time_hms} [h:m:s]\n"
                        )

        _log("\nSimulations completed!\n")

        # Merge the newly computed table with self.bler_table
        self._bler_table.deep_update(new_table, stop_at_keys=("CBS", "SNR_db"))
        self.validate_bler_table()

        # Append file name to the internal list of loaded BLER files
        if filename is not None:
            self._bler_table_filenames.append(filename)

        return new_table
