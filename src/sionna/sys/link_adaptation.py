#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Link adaptation for Sionna SYS"""

from typing import Any, Optional, Tuple, Union

import torch

from sionna.phy import Block, config, dtypes
from sionna.phy.config import Precision
from sionna.phy.utils import (
    db_to_lin,
    expand_to_rank,
    find_true_position,
    gather_from_batched_indices,
    insert_dims,
    lin_to_db,
    scalar_to_shaped_tensor,
)
from sionna.sys.phy_abstraction import PHYAbstraction
from sionna.sys.utils import is_scheduled_in_slot

__all__ = ["InnerLoopLinkAdaptation", "OuterLoopLinkAdaptation"]


def _to_python_float(value: Union[float, int, torch.Tensor]) -> float:
    """Convert a value to a Python float in a torch.compile friendly way.

    Note: If a tensor is passed, this will cause a graph break in torch.compile
    due to the .item() call. To avoid graph breaks, ensure Python floats are
    passed directly when setting properties during compiled execution.

    :param value: Value to convert (float, int, or scalar tensor)

    :output value: Python float value
    """
    if isinstance(value, (int, float)):
        return float(value)
    elif isinstance(value, torch.Tensor):
        # Detach and move to CPU to ensure we can extract the value
        return float(value.detach().cpu().item())
    else:
        # Fallback for other numeric types (numpy scalars, etc.)
        return float(value)


def _validate_bler_target(value: float) -> float:
    """Require a BLER target strictly inside (0, 1)."""
    value = _to_python_float(value)
    if not 0.0 < value < 1.0:
        raise ValueError("'bler_target' must satisfy 0 < bler_target < 1")
    return value


def _validate_offset_bounds(offset_min: float, offset_max: float) -> None:
    """Require offset_min <= offset_max."""
    if offset_min > offset_max:
        raise ValueError("'offset_min' must not exceed 'offset_max'")


class InnerLoopLinkAdaptation(Block):
    r"""Inner loop link adaptation (ILLA).

    Computes the highest available modulation and coding scheme (MCS) whose
    associated transport block error rate (TBLER) does not exceed the specified
    ``bler_target``:

    .. math::

        \max \left\{ \text{MCS}: \ \text{TBLER}(\text{MCS}, \text{SINR}_{\text{eff}}) \le \text{BLER}_{\text{target}} \right\}

    where :math:`\text{SINR}_{\text{eff}}` is the effective SINR value provided
    as input.
    If no such MCS exists, the lowest available MCS index is returned. If a user
    is not scheduled, ``fill_mcs_value`` is returned.

    :param phy_abstraction: An instance of :class:`~sionna.sys.PHYAbstraction`.
        If `None`, a default instance is created.
    :param bler_target: BLER target. Defaults to 0.1.
    :param fill_mcs_value: MCS value assigned to non-scheduled users.
        Defaults to 0.
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`, :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for computation. If `None`,
        :attr:`~sionna.phy.config.Config.device` is used.

    :input sinr: [..., num_ofdm_symbols, num_subcarriers, num_ut, num_streams_per_ut], `torch.float` | `None` (default).
        SINR for each OFDM symbol, subcarrier, user and stream.
        If `None`, then ``sinr_eff`` and ``num_allocated_re`` are both
        required.
    :input sinr_eff: [..., num_ut], `torch.float` | `None` (default).
        Estimated effective SINR for each user.
        If `None`, then ``sinr`` is required.
    :input num_allocated_re: [..., num_ut], `torch.int32` | `None` (default).
        Number of allocated resources in a slot, computed across OFDM symbols,
        subcarriers and streams, for each user.
        If `None`, then ``sinr`` is required.
    :input mcs_table_index: [..., num_ut], `torch.int32` | `int` (default: 1).
        MCS table index for each user. For further details, refer to the
        :ref:`mcs_table_cat_note`.
    :input mcs_category: [..., num_ut], `torch.int32` | `int` (default: 0).
        MCS table category for each user. For further details, refer to the
        :ref:`mcs_table_cat_note`.
    :input return_lowest_available_mcs: `bool` (default: `False`).
        If `True`, the lowest MCS available in ``phy_abstraction`` BLER tables
        is returned for each user. Only used for internal purposes.

    :output mcs_index: [..., num_ut].
        Highest available MCS whose BLER does not exceed the target, or the
        lowest available MCS if no such MCS exists, for each user.

    .. rubric:: Examples

    .. code-block:: python

        import torch
        from sionna.sys import PHYAbstraction, InnerLoopLinkAdaptation

        bler_target = 0.1

        # Initialize the PHY abstraction object
        phy_abs = PHYAbstraction()

        # Initialize the ILLA object
        illa = InnerLoopLinkAdaptation(phy_abs, bler_target=0.1)

        # Effective SINR for each user
        sinr_eff = torch.tensor([0.1, 10, 100])
        # N. allocated resource elements for each user
        num_allocated_re = torch.tensor([20, 30, 30])

        # Compute the MCS index for each user
        mcs_index = illa(sinr_eff=sinr_eff,
                        num_allocated_re=num_allocated_re,
                        mcs_table_index=1,
                        mcs_category=0)
        print("Selected MCS index =", mcs_index)
    """

    def __init__(
        self,
        phy_abstraction: Optional[PHYAbstraction] = None,
        bler_target: float = 0.1,
        fill_mcs_value: int = 0,
        precision: Optional[Precision] = None,
        device: Optional[str] = None,
    ) -> None:
        super().__init__(precision=precision, device=device)

        if phy_abstraction is None:
            phy_abstraction = PHYAbstraction(precision=precision, device=device)

        self._phy_abstraction = phy_abstraction
        self._fill_mcs_value = torch.tensor(
            fill_mcs_value, dtype=torch.int32, device=self.device
        )
        self._bler_target = torch.tensor(
            _validate_bler_target(bler_target),
            dtype=self.dtype,
            device=self.device,
        )

    @property
    def phy_abstraction(self) -> PHYAbstraction:
        """PHYAbstraction object used to compute TBLER (read-only)."""
        return self._phy_abstraction

    @property
    def bler_target(self) -> torch.Tensor:
        """Get/set the BLER target for each user."""
        return self._bler_target

    @bler_target.setter
    def bler_target(self, value: Union[float, torch.Tensor]) -> None:
        self._bler_target = torch.tensor(
            _validate_bler_target(value), dtype=self.dtype, device=self.device
        )

    def call(
        self,
        sinr: Optional[torch.Tensor] = None,
        sinr_eff: Optional[torch.Tensor] = None,
        num_allocated_re: Optional[torch.Tensor] = None,
        mcs_table_index: Union[int, torch.Tensor] = 1,
        mcs_category: Union[int, torch.Tensor] = 0,
        return_lowest_available_mcs: bool = False,
        **kwargs: Any,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Select optimal MCS index for each user."""
        # Validate inputs
        if not (
            (sinr is not None)
            ^ ((sinr_eff is not None) and (num_allocated_re is not None))
        ):
            raise ValueError(
                "Exactly one of 'sinr' or the pair "
                "('sinr_eff', 'num_allocated_re') is required as input"
            )

        # Number of available MCS indices
        num_mcs = self._phy_abstraction.bler_table_interp.shape[2]

        # Check which UTs are scheduled
        ut_is_scheduled = is_scheduled_in_slot(
            sinr=sinr, num_allocated_re=num_allocated_re
        )

        # Determine batch dimensions and num_ut
        if sinr is not None:
            sinr = sinr.to(self.dtype)
            batch_dims = list(sinr.shape[:-4])
            num_ut = sinr.shape[-2]
        else:
            sinr_eff = sinr_eff.to(self.dtype)
            batch_dims = list(sinr_eff.shape[:-1])
            num_ut = sinr_eff.shape[-1]

        # ----------------------- #
        # Tile across MCS indices #
        # ----------------------- #
        # [..., num_mcs, num_ut]
        mcs_index_all = torch.arange(num_mcs, dtype=torch.int32, device=self.device)
        mcs_index_all = insert_dims(mcs_index_all, len(batch_dims), axis=0).unsqueeze(-1)
        mcs_index_all = mcs_index_all.expand(*batch_dims, num_mcs, num_ut)

        # [..., num_mcs, num_ut]
        mcs_table_index_tiled = scalar_to_shaped_tensor(
            mcs_table_index, torch.int32, batch_dims + [num_ut], device=self.device
        )
        mcs_table_index_tiled = mcs_table_index_tiled.unsqueeze(-2).expand(
            *batch_dims, num_mcs, num_ut
        )

        # [..., num_mcs, num_ut]
        mcs_category_tiled = scalar_to_shaped_tensor(
            mcs_category, torch.int32, batch_dims + [num_ut], device=self.device
        )
        mcs_category_tiled = mcs_category_tiled.unsqueeze(-2).expand(
            *batch_dims, num_mcs, num_ut
        )

        if num_allocated_re is not None:
            # [..., num_mcs, num_ut]
            num_allocated_re = num_allocated_re.to(torch.int32)
            num_allocated_re_tiled = num_allocated_re.unsqueeze(-2).expand(
                *batch_dims, num_mcs, num_ut
            )
        else:
            num_allocated_re_tiled = None

        # -------------- #
        # Effective SINR #
        # -------------- #
        # Expand across all possible MCS indices
        if sinr is not None:
            # [..., num_mcs, num_ofdm_symbols, num_subcarriers, num_ut, num_streams_per_ut]
            sinr_tiled = sinr.unsqueeze(-5).expand(
                *batch_dims, num_mcs, *sinr.shape[-4:]
            )
            sinr_eff_tiled = None
        else:
            # [..., num_mcs, num_ut]
            sinr_tiled = None
            sinr_eff_tiled = sinr_eff.unsqueeze(-2).expand(*batch_dims, num_mcs, num_ut)

        # ----- #
        # TBLER #
        # ----- #
        # [..., num_mcs, num_ut]
        *_, tbler_per_mcs, _ = self._phy_abstraction(
            mcs_index_all,
            sinr=sinr_tiled,
            sinr_eff=sinr_eff_tiled,
            num_allocated_re=num_allocated_re_tiled,
            mcs_table_index=mcs_table_index_tiled,
            mcs_category=mcs_category_tiled,
            check_mcs_index_validity=False,
            **kwargs,
        )

        # ---------- #
        # Select MCS #
        # ---------- #
        # Find the highest MCS with TBLER <= bler_target
        # If no such MCS is found, returns -1
        # TBLER is +inf for MCS indices without BLER data, so unavailable
        # candidates do not satisfy the target.
        # [..., num_ut]
        mcs_index = find_true_position(
            tbler_per_mcs <= self._bler_target,
            side="last",
            axis=-2,
        )

        # Lowest available MCS (TBLER in valid range [0, 1])
        # [..., num_ut]
        lowest_available_mcs = find_true_position(
            (tbler_per_mcs >= 0) & (tbler_per_mcs <= 1), side="first", axis=-2
        )

        # If all MCS have TBLER > bler_target, select lowest available MCS
        mcs_index = torch.where(mcs_index != -1, mcs_index, lowest_available_mcs)

        # A non-scheduled user receives MCS=fill_mcs_value
        mcs_index = torch.where(ut_is_scheduled, mcs_index, self._fill_mcs_value)

        if return_lowest_available_mcs:
            return mcs_index, lowest_available_mcs
        return mcs_index


class OuterLoopLinkAdaptation(Block):
    r"""Outer-loop link adaptation (OLLA).

    The modulation and coding scheme (MCS) index for a user is determined as the
    highest index whose corresponding transport block error rate (TBLER) remains
    below the specified ``bler_target``.
    The SINR value used for TBLER computation is given by the last effective
    SINR feedback, :math:`\text{SINR}_{\text{eff}}` [dB], reduced by an offset
    value, :math:`\Delta_{\mathrm{offset}}`:

    .. math::

        \max \left\{ \text{MCS}: \ \text{TBLER}(\text{MCS}, \text{SINR}_{\text{eff}}-\Delta_{\text{offset}}) \le \text{BLER}_{\text{target}} \right\}

    The value of :math:`\Delta_{\text{offset}}` is adjusted depending on the
    HARQ feedback :cite:p:`Pedersen05`:

    .. math::

        \Delta_{\mathrm{offset}} = \left\{
        \begin{array}{l}
            \Delta_{\mathrm{offset}} - \Delta_{\mathrm{down}} \quad \text{if HARQ=ACK} \\
            \Delta_{\mathrm{offset}} + \Delta_{\mathrm{up}} \quad \text{if HARQ=NACK}
        \end{array}
        \right.

    where the relationship between
    :math:`\Delta_{\mathrm{up}}` and :math:`\Delta_{\mathrm{down}}` is given by
    :cite:p:`Sampath97`:

    .. math::
        \frac{\Delta_{\mathrm{up}}}{\Delta_{\mathrm{down}}} = \frac{1 - \mathrm{BLER}_{\mathrm{target}}}{\mathrm{BLER}_{\mathrm{target}}}.

    :param phy_abstraction: An instance of :class:`~sionna.sys.PHYAbstraction`
    :param num_ut: Number of user terminals
    :param bler_target: BLER target value, within 0 and 1. Defaults to 0.1.
    :param delta_up: Increment applied to the SINR offset [dB] when a NACK is
        received for a user. Defaults to 1.0.
    :param batch_size: Batch size or shape. It accounts for multiple users for
        which link adaptation is performed simultaneously. If `None`, the batch
        size is set to [].
    :param sinr_eff_init: Initial value of effective SINR for each user.
        Non-positive values are treated as missing and replaced by
        ``sinr_eff_init_fill``. If `float`, the same value is assigned to all
        users. Defaults to 1.0.
    :param sinr_eff_init_fill: Value replacing non-positive ``sinr_eff_init``
        values. Defaults to 1.0.
    :param offset_min: Minimum SINR [dB] offset value. Defaults to -20.0.
    :param offset_max: Maximum SINR [dB] offset value. Defaults to 20.0.
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`, :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for computation. If `None`,
        :attr:`~sionna.phy.config.Config.device` is used.

    :input num_allocated_re: [..., num_ut], `torch.int32`.
        Number of allocated resources in the upcoming slot, computed across OFDM
        symbols, subcarriers and streams, for each user.
    :input harq_feedback: [..., num_ut], -1 | 0 | 1.
        If 0 (1, resp.), then a NACK (ACK, resp.) is received. If -1, feedback
        is missing.
    :input sinr_eff: [..., num_ut], `torch.float` | `None` (default).
        Estimated effective SINR for each user. Non-positive values are treated
        as missing.
    :input mcs_table_index: [..., num_ut], `torch.int32` | `int` (default: 1).
        MCS table index for each user. For further details, refer to the
        :ref:`mcs_table_cat_note`.
    :input mcs_category: [..., num_ut], `torch.int32` | `int` (default: 0).
        MCS table category for each user. For further details, refer to the
        :ref:`mcs_table_cat_note`.

    :output mcs_index: [..., num_ut].
        Selected MCS index for each user.

    .. rubric:: Examples

    .. code-block:: python

        import torch
        from sionna.sys import PHYAbstraction, OuterLoopLinkAdaptation

        num_ut = 4
        bler_target = 0.1

        # Initialize the PHY abstraction object
        phy_abs = PHYAbstraction()

        # Initialize the OLLA object
        olla = OuterLoopLinkAdaptation(phy_abs, num_ut=num_ut,
                                       bler_target=bler_target)

        # Number of allocated REs for each user
        num_allocated_re = torch.tensor([100, 200, 150, 50])

        # HARQ feedback for each user (-1: N/A, 0: NACK, 1: ACK)
        harq_feedback = torch.tensor([1, 0, 1, -1])

        # Effective SINR feedback for each user
        sinr_eff = torch.tensor([10.0, 5.0, 8.0, 0.0])

        # Compute the MCS index for each user
        mcs_index = olla(num_allocated_re, harq_feedback, sinr_eff)
    """

    def __init__(
        self,
        phy_abstraction: PHYAbstraction,
        num_ut: int,
        bler_target: float = 0.1,
        delta_up: float = 1.0,
        batch_size: Optional[Union[int, list]] = None,
        sinr_eff_init: float = 1.0,
        sinr_eff_init_fill: float = 1.0,
        offset_min: float = -20.0,
        offset_max: float = 20.0,
        precision: Optional[Precision] = None,
        device: Optional[str] = None,
    ) -> None:
        super().__init__(precision=precision, device=device)

        if sinr_eff_init_fill <= 0:
            raise ValueError("'sinr_eff_init_fill' must be positive")

        if batch_size is None:
            batch_size = []
        elif not isinstance(batch_size, list):
            batch_size = [batch_size]

        bler_target = _validate_bler_target(bler_target)
        delta_up = _to_python_float(delta_up)
        if delta_up <= 0:
            raise ValueError("'delta_up' must be positive")
        offset_min = _to_python_float(offset_min)
        offset_max = _to_python_float(offset_max)
        _validate_offset_bounds(offset_min, offset_max)

        self._batch_size = batch_size
        self._num_ut = num_ut
        self._phy_abstraction = phy_abstraction
        self._illa = InnerLoopLinkAdaptation(phy_abstraction, bler_target=bler_target,
                                             precision=precision, device=device)

        # Store scalar parameters as Python floats to avoid device issues with torch.compile
        self._bler_target_value = bler_target
        self._delta_up_value = delta_up
        self._delta_down_value = self._get_delta_down_value()
        self._offset_min_value = offset_min
        self._offset_max_value = offset_max

        # Initialize effective SINR [dB]
        sinr_eff_init_tensor = scalar_to_shaped_tensor(
            sinr_eff_init, self.dtype, self._batch_size + [self._num_ut], device=self.device
        )
        sinr_eff_init_fill_db = lin_to_db(
            torch.tensor(sinr_eff_init_fill, dtype=self.dtype, device=self.device),
            precision=self.precision
        )
        # Convert effective SINR to dB and fill N/A values (<=0)
        # State tensors as regular attributes (not buffers) for simpler torch.compile handling
        self._sinr_eff_db_last = torch.where(
            sinr_eff_init_tensor > 0,
            lin_to_db(sinr_eff_init_tensor, precision=self.precision),
            sinr_eff_init_fill_db
        )

        # Reset SINR offset to 0
        self._offset = torch.zeros(
            self._batch_size + [self._num_ut], dtype=self.dtype, device=self.device
        )

    def _get_delta_down_value(self) -> float:
        return self._delta_up_value * self._bler_target_value / (1 - self._bler_target_value)

    def reset(
        self,
        sinr_eff_init: float = 1.0,
        sinr_eff_init_fill: float = 0.1,
    ) -> None:
        """Resets the values of ``sinr_eff_db_last`` and ``offset``.

        :param sinr_eff_init: Initial effective SINR value (linear scale).
        :param sinr_eff_init_fill: Fill value for N/A SINR entries (linear
            scale).
        """
        device = self._sinr_eff_db_last.device
        sinr_eff_init_tensor = scalar_to_shaped_tensor(
            sinr_eff_init, self.dtype, self._batch_size + [self._num_ut], device=device
        )
        sinr_eff_init_fill_db = lin_to_db(
            torch.tensor(sinr_eff_init_fill, dtype=self.dtype, device=device),
            precision=self.precision
        )
        # Convert effective SINR to dB and fill N/A values (<=0)
        self._sinr_eff_db_last = torch.where(
            sinr_eff_init_tensor > 0,
            lin_to_db(sinr_eff_init_tensor, precision=self.precision),
            sinr_eff_init_fill_db
        )
        # Reset SINR offset to 0
        self._offset = torch.zeros_like(self._offset)

    @property
    def offset(self) -> torch.Tensor:
        """Effective SINR [dB] offset for each user (read-only)."""
        return self._offset

    @property
    def offset_max(self) -> float:
        """Get/set the maximum ``offset`` value."""
        return self._offset_max_value

    @offset_max.setter
    def offset_max(self, value: float) -> None:
        value = _to_python_float(value)
        _validate_offset_bounds(self._offset_min_value, value)
        self._offset_max_value = value

    @property
    def offset_min(self) -> float:
        """Get/set the minimum ``offset`` value."""
        return self._offset_min_value

    @offset_min.setter
    def offset_min(self, value: float) -> None:
        value = _to_python_float(value)
        _validate_offset_bounds(value, self._offset_max_value)
        self._offset_min_value = value

    @property
    def bler_target(self) -> float:
        """Get/set the BLER target for each user."""
        return self._bler_target_value

    @bler_target.setter
    def bler_target(self, value: float) -> None:
        self._bler_target_value = _validate_bler_target(value)
        self._delta_down_value = self._get_delta_down_value()
        self._illa.bler_target = self._bler_target_value

    @property
    def sinr_eff_db_last(self) -> torch.Tensor:
        """Get/set the last observed effective SINR [dB] value for each user."""
        return self._sinr_eff_db_last

    @sinr_eff_db_last.setter
    def sinr_eff_db_last(self, value: torch.Tensor) -> None:
        self._sinr_eff_db_last = value.to(dtype=self.dtype)

    @property
    def delta_down(self) -> float:
        r"""Decrement applied to the SINR offset when an ACK is received (read-only).

        Computed as ``delta_up * bler_target / (1 - bler_target)``.
        """
        return self._delta_down_value

    @property
    def delta_up(self) -> float:
        """Get/set the increment applied to the SINR offset when a NACK is received."""
        return self._delta_up_value

    @delta_up.setter
    def delta_up(self, value: float) -> None:
        value = _to_python_float(value)
        if value <= 0:
            raise ValueError("'delta_up' must be positive")
        self._delta_up_value = value
        self._delta_down_value = self._get_delta_down_value()

    def call(
        self,
        num_allocated_re: torch.Tensor,
        harq_feedback: Optional[torch.Tensor] = None,
        sinr_eff: Optional[torch.Tensor] = None,
        mcs_table_index: int = 1,
        mcs_category: int = 0,
    ) -> torch.Tensor:
        """Run outer loop link adaptation."""
        shape = num_allocated_re.shape

        # Handle defaults - use module's device (which should match input device)
        if harq_feedback is None:
            harq_feedback = torch.full(shape, -1, dtype=torch.int32, device=self.device)
        else:
            harq_feedback = harq_feedback.to(dtype=torch.int32)

        if sinr_eff is None:
            sinr_eff = torch.zeros(shape, dtype=self.dtype, device=self.device)
        else:
            sinr_eff = sinr_eff.to(dtype=self.dtype)

        num_allocated_re = num_allocated_re.to(dtype=torch.int32)
        mcs_table_index = scalar_to_shaped_tensor(
            mcs_table_index, torch.int32, list(shape), device=self.device
        )
        mcs_category = scalar_to_shaped_tensor(
            mcs_category, torch.int32, list(shape), device=self.device
        )

        # ---------------------------- #
        # Update effective SINR offset #
        # ---------------------------- #
        # Use Python floats for scalar parameters to avoid device issues with torch.compile
        delta_down = self._delta_down_value
        delta_up = self._delta_up_value
        offset_min = self._offset_min_value
        offset_max = self._offset_max_value

        new_offset = torch.where(
            harq_feedback == 1,
            self._offset - delta_down,
            torch.where(
                harq_feedback == 0,
                self._offset + delta_up,
                self._offset
            )
        )

        # Project offset to [offset_min; offset_max]
        new_offset = torch.clamp(new_offset, offset_min, offset_max)
        self._offset = new_offset

        # ----------------------------------- #
        # Update last observed effective SINR #
        # ----------------------------------- #
        new_sinr_eff_db = torch.where(
            sinr_eff > 0,
            lin_to_db(sinr_eff, precision=self.precision),
            self._sinr_eff_db_last
        )
        self._sinr_eff_db_last = new_sinr_eff_db

        # -------------------------- #
        # Offset SINR and apply ILLA #
        # -------------------------- #
        sinr_eff_offset = db_to_lin(
            self._sinr_eff_db_last - self._offset,
            precision=self.precision
        )
        mcs_index = self._illa(
            sinr_eff=sinr_eff_offset,
            num_allocated_re=num_allocated_re,
            mcs_table_index=mcs_table_index,
            mcs_category=mcs_category,
            return_lowest_available_mcs=False
        )

        return mcs_index
