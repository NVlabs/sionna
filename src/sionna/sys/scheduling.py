#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Scheduling algorithms for Sionna SYS"""

from typing import Any, List, Optional, Tuple, Union

import torch

from sionna.phy import Block, config, dtypes
from sionna.phy.config import Precision
from sionna.phy.utils import insert_dims

__all__ = ["PFSchedulerSUMIMO"]


class PFSchedulerSUMIMO(Block):
    r"""Proportional fairness (PF) scheduler for single-user MIMO (SU-MIMO) systems.

    Schedules users according to a proportional fairness (PF) metric in a
    single-user (SU) multiple-input multiple-output (MIMO) system, i.e., at most
    one user is scheduled per time-frequency resource.

    Fixing the time slot :math:`t`, :math:`\tilde{R}_t(u,i)` is
    the :emphasis:`achievable` rate for user :math:`u` on the time-frequency
    resource :math:`i` during the current slot.
    Let :math:`T_{t-1}(u)` denote the throughput :emphasis:`achieved` by user
    :math:`u` up to and including slot :math:`t-1`.
    Resource :math:`i` is assigned to the user with the highest PF metric,
    as defined in :cite:p:`Jalali00`:

    .. math::
        \operatorname{argmax}_{u} \frac{\tilde{R}_{t}(u,i)}{T_{t-1}(u)}.

    All streams within a scheduled resource element are assigned to the selected user.

    Let :math:`R_t(u)` be the rate achieved by user :math:`u` in slot :math:`t`.
    The throughput :math:`T` by each user :math:`u` is updated via
    geometric discounting:

    .. math::
        T_t(u) = \beta \, T_{t-1}(u) + (1-\beta) \, R_t(u)

    where :math:`\beta\in(0,1)` is the discount factor.

    :param num_ut: Number of user terminals
    :param num_freq_res: Number of available frequency resources.
        A frequency resource is the smallest frequency unit that can be
        allocated to a user, typically a physical resource block (PRB).
    :param num_ofdm_sym: Number of OFDM symbols in a slot
    :param batch_size: Batch size or shape. It can account for multiple sectors in
        which scheduling is performed simultaneously. If `None`, the batch size
        is set to [].
    :param num_streams_per_ut: Number of streams per user. Defaults to 1.
    :param beta: Discount factor for computing the time-averaged achieved rate.
        Must be within (0,1). Defaults to 0.98.
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`, :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for computation. If `None`,
        :attr:`~sionna.phy.config.Config.device` is used.

    :input rate_last_slot: [batch_size, num_ut], `torch.float`.
        Rate achieved by each user in the last slot.
    :input rate_achievable_curr_slot: [batch_size, num_ofdm_sym, num_freq_res, num_ut], `torch.float`.
        Achievable rate for each user across the OFDM grid in the
        current slot.

    :output is_scheduled: [batch_size, num_ofdm_sym, num_freq_res, num_ut, num_streams_per_ut], `torch.bool`.
        Whether a user is scheduled for transmission for each available resource.

    .. rubric:: Examples

    .. code-block:: python

        import torch
        from sionna.sys import PFSchedulerSUMIMO

        num_ut = 4
        num_freq_res = 52
        num_ofdm_sym = 14
        batch_size = 10

        # Create PF scheduler
        scheduler = PFSchedulerSUMIMO(
            num_ut,
            num_freq_res,
            num_ofdm_sym,
            batch_size=batch_size
        )

        # Generate random achievable rates and last slot rates
        rate_last_slot = torch.rand(batch_size, num_ut) * 100
        rate_achievable_curr_slot = torch.rand(
            batch_size, num_ofdm_sym, num_freq_res, num_ut
        ) * 100

        # Get scheduling decisions
        is_scheduled = scheduler(rate_last_slot, rate_achievable_curr_slot)
        print(is_scheduled.shape)
        # torch.Size([10, 14, 52, 4, 1])
    """

    def __init__(
        self,
        num_ut: int,
        num_freq_res: int,
        num_ofdm_sym: int,
        batch_size: Optional[Union[List[int], int]] = None,
        num_streams_per_ut: int = 1,
        beta: float = 0.98,
        precision: Optional[Precision] = None,
        device: Optional[str] = None,
    ) -> None:
        super().__init__(precision=precision, device=device)

        if batch_size is None:
            batch_size = []
        elif not isinstance(batch_size, list):
            if isinstance(batch_size, int) or len(batch_size) == 0:
                batch_size = [batch_size]

        self._batch_size = batch_size
        self._num_ut = int(num_ut)
        self._num_freq_res = int(num_freq_res)
        self._num_ofdm_sym = int(num_ofdm_sym)
        self._num_streams_per_ut = int(num_streams_per_ut)

        # Validate and store beta as Python float to avoid device issues with torch.compile
        if not 0.0 < beta < 1.0:
            raise ValueError("Discount factor 'beta' must be within (0, 1)")
        self._beta_value = float(beta)

        # Register state tensors as buffers for proper device tracking
        # Average achieved rate (internal state)
        self.register_buffer(
            "_rate_achieved_past",
            torch.ones(
                list(batch_size) + [num_ut], dtype=self.dtype, device=self.device
            ),
            persistent=False,
        )

        # PF metric (internal state for debugging)
        self.register_buffer(
            "_pf_metric",
            torch.zeros(
                list(batch_size) + [num_ofdm_sym, num_freq_res, num_ut],
                dtype=self.dtype,
                device=self.device,
            ),
            persistent=False,
        )

    @property
    def rate_achieved_past(self) -> torch.Tensor:
        r"""[batch_size, num_ut] (read-only) : :math:`\beta`-discounted
        time-averaged achieved rate for each user"""
        return self._rate_achieved_past

    @property
    def pf_metric(self) -> torch.Tensor:
        """[batch_size, num_ofdm_sym, num_freq_res, num_ut] (read-only) :
        Proportional fairness (PF) metric in the last slot"""
        return self._pf_metric

    @property
    def beta(self) -> float:
        """Get/set the discount factor for computing the time-averaged
        achieved rate. Must be within (0,1)."""
        return self._beta_value

    @beta.setter
    def beta(self, value: float) -> None:
        if not 0.0 < value < 1.0:
            raise ValueError("Discount factor 'beta' must be within (0, 1)")
        self._beta_value = float(value)

    def call(
        self,
        rate_last_slot: torch.Tensor,
        rate_achievable_curr_slot: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute scheduling decisions based on proportional fairness."""
        # ------------------------ #
        # Validate and cast inputs #
        # ------------------------ #
        expected_rate_last_slot_shape = self._batch_size + [self._num_ut]
        if list(rate_last_slot.shape) != expected_rate_last_slot_shape:
            raise ValueError(
                f"Inconsistent 'rate_last_slot' shape: expected "
                f"{expected_rate_last_slot_shape}, got {list(rate_last_slot.shape)}"
            )

        expected_rate_achievable_shape = self._batch_size + [
            self._num_ofdm_sym,
            self._num_freq_res,
            self._num_ut,
        ]
        if list(rate_achievable_curr_slot.shape) != expected_rate_achievable_shape:
            raise ValueError(
                f"Inconsistent 'rate_achievable_curr_slot' shape: expected "
                f"{expected_rate_achievable_shape}, "
                f"got {list(rate_achievable_curr_slot.shape)}"
            )

        # [batch_size, num_ut]
        rate_last_slot = rate_last_slot.to(self.dtype)
        # [batch_size, num_ofdm_sym, num_freq_res, num_ut]
        rate_achievable_curr_slot = rate_achievable_curr_slot.to(self.dtype)

        # ---------------------------- #
        # Update average achieved rate #
        # ---------------------------- #
        # [batch_size, num_ut]
        # Use Python float for beta to avoid device mismatch issues with torch.compile
        beta = self._beta_value
        rate_achieved_past_new = beta * self._rate_achieved_past + (1 - beta) * rate_last_slot

        # Store updated state using in-place copy for torch.compile compatibility
        self._rate_achieved_past.copy_(rate_achieved_past_new)

        # [batch_size, 1, 1, num_ut]
        rate_achieved_past = insert_dims(rate_achieved_past_new, 2, axis=-2)

        # ----------------- #
        # Compute PF metric #
        # ----------------- #
        # [batch_size, num_ofdm_sym, num_freq_res, num_ut]
        pf_metric = rate_achievable_curr_slot / rate_achieved_past

        # Store for debugging access using in-place copy for torch.compile compatibility
        self._pf_metric.copy_(pf_metric)

        # ------------ #
        # Schedule UTs #
        # ------------ #
        # Assign each time/frequency resource to the user with highest PF metric
        # [batch_size, num_ofdm_sym, num_freq_res]
        scheduled_ut = torch.argmax(self._pf_metric, dim=-1)
        # [batch_size, num_ofdm_sym, num_freq_res, num_ut]
        is_scheduled = torch.nn.functional.one_hot(
            scheduled_ut, num_classes=self._num_ut
        )
        # [batch_size, num_ofdm_sym, num_freq_res, num_ut, 1]
        is_scheduled = is_scheduled.unsqueeze(-1)
        # [batch_size, num_ofdm_sym, num_freq_res, num_ut, num_streams]
        is_scheduled = is_scheduled.expand(
            *is_scheduled.shape[:-1], self._num_streams_per_ut
        )

        return is_scheduled.to(torch.bool)
