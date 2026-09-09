#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Utility functions for Sionna SYS"""

from typing import Optional, Tuple

import torch

from sionna._validation import check_binary, check_tensor_all
from sionna.phy import config, dtypes
from sionna.phy.config import Precision
from sionna.phy.utils import insert_dims

__all__ = [
    "is_scheduled_in_slot",
    "get_pathloss",
    "spread_across_subcarriers",
]


def is_scheduled_in_slot(
    sinr: Optional[torch.Tensor] = None,
    num_allocated_re: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    r"""Determines whether a user is scheduled in a slot.

    :param sinr: SINR for each OFDM symbol, subcarrier, user, and stream.
        If `None`, then ``num_allocated_re`` is required.
    :param num_allocated_re: Number of allocated resources (streams/REs/PRBs etc.)
        per user.
        If `None`, then ``sinr`` is required.

    :output is_scheduled: [..., num_ut], `torch.bool`.
        Whether a user is scheduled in a slot.

    .. rubric:: Examples

    .. code-block:: python

        import torch
        from sionna.sys.utils import is_scheduled_in_slot

        # Using SINR input
        sinr = torch.rand(2, 14, 52, 4, 2)  # [batch, symbols, subcarriers, users, streams]
        is_sched = is_scheduled_in_slot(sinr=sinr)
        print(is_sched.shape)
        # torch.Size([2, 4])

        # Using num_allocated_re input
        num_re = torch.tensor([10, 0, 5, 8])  # 4 users, 2nd not scheduled
        is_sched = is_scheduled_in_slot(num_allocated_re=num_re)
        print(is_sched)
        # tensor([ True, False,  True,  True])
    """
    if not ((sinr is not None) ^ (num_allocated_re is not None)):
        raise ValueError(
            "Exactly one of 'sinr' or 'num_allocated_re' is required as input"
        )

    if sinr is not None:
        return sinr.sum(dim=(-4, -3, -1)) > 0
    else:
        return num_allocated_re > 0


def get_pathloss(
    h_freq: torch.Tensor,
    rx_tx_association: Optional[torch.Tensor] = None,
    precision: Optional[Precision] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    r"""Computes the pathloss for each receiver-transmitter pair and, if the
    receiver-transmitter association is provided, the pathloss between each
    user and the associated base station.

    :param h_freq: OFDM channel matrix.
    :param rx_tx_association: Its :math:`(i,j)` element is 1 if receiver
        :math:`i` is attached to transmitter :math:`j`, 0 otherwise. It must
        define exactly one serving link per user.
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`, :attr:`~sionna.phy.config.Config.precision` is used.

    :output pathloss_all_pairs: [..., num_rx, num_tx, num_ofdm_symbols], `torch.float`.
        Pathloss for each RX-TX pair and across OFDM symbols.
    :output pathloss_serving_tx: [..., num_ut, num_ofdm_symbols], `torch.float`.
        Pathloss between each user and the associated base station. Only computed
        if ``rx_tx_association`` is provided as input.

    .. rubric:: Examples

    .. code-block:: python

        import torch
        from sionna.sys.utils import get_pathloss

        # Create a simple channel matrix
        h = torch.randn(2, 4, 2, 3, 2, 14, 52, dtype=torch.complex64)
        pathloss_all, _ = get_pathloss(h)
        print(pathloss_all.shape)
        # torch.Size([2, 4, 3, 14])
    """
    if precision is None:
        dtype = config.dtype
    else:
        dtype = dtypes[precision]["torch"]["dtype"]

    # Compute RX power
    # [..., num_rx, num_rx_ant, num_tx, num_tx_ant, num_ofdm_symbols, num_subcarriers]
    rx_power = torch.abs(h_freq).pow(2).to(dtype)

    # Average across TX/RX antennas and subcarriers
    # [..., num_rx, num_tx, num_ofdm_symbols]
    rx_power = rx_power.mean(dim=(-1, -3, -5))

    # Get pathloss
    # [..., num_rx, num_tx, num_ofdm_symbols]
    pathloss_all_pairs = 1.0 / rx_power

    if rx_tx_association is None:
        pathloss_serving_tx = None
    else:
        rx_tx_association = rx_tx_association.to(
            device=pathloss_all_pairs.device
        )
        check_binary(
            rx_tx_association,
            name="rx_tx_association",
            message="rx_tx_association must contain binary values",
        )

        # Each user has exactly one serving link. Depending on the direction,
        # users are represented by either the RX or TX dimension.
        num_ut = max(rx_tx_association.shape)
        check_tensor_all(
            rx_tx_association.sum() == num_ut,
            name="rx_tx_association",
            message=(
                "rx_tx_association must define exactly one serving link "
                "per user"
            ),
        )

        # Extract all marked links in the same receiver-major order as Boolean
        # indexing, without introducing a data-dependent output shape.
        association_flat = rx_tx_association.reshape(-1).to(torch.bool)
        flat_indices = torch.arange(
            association_flat.numel(), device=pathloss_all_pairs.device
        )
        sort_keys = torch.where(
            association_flat,
            flat_indices,
            flat_indices + association_flat.numel(),
        )
        serving_indices = torch.argsort(sort_keys)[:num_ut]

        pathloss_flat = pathloss_all_pairs.flatten(start_dim=-3, end_dim=-2)
        serving_indices = insert_dims(
            serving_indices, num_dims=pathloss_flat.dim() - 2, axis=0
        )
        serving_indices = serving_indices.unsqueeze(-1).expand(
            *pathloss_flat.shape[:-2], num_ut, pathloss_flat.shape[-1]
        )
        pathloss_serving_tx = torch.gather(
            pathloss_flat, dim=-2, index=serving_indices
        )

    return pathloss_all_pairs, pathloss_serving_tx


def spread_across_subcarriers(
    tx_power_per_ut: torch.Tensor,
    is_scheduled: torch.Tensor,
    num_tx: Optional[int] = None,
    precision: Optional[Precision] = None,
) -> torch.Tensor:
    r"""Distributes the power uniformly across all allocated subcarriers
    and streams for each user.

    :param tx_power_per_ut: Transmit power [W] for each user across OFDM
        symbols. Expected shape ``[..., num_ofdm_sym, num_ut]``. The power of
        each OFDM symbol is distributed uniformly across that user's allocated
        subcarriers and streams.
    :param is_scheduled: Whether a user is scheduled on a given subcarrier
        and stream.
    :param num_tx: Number of transmitters. If `None`, it is set to
        ``num_ut``, as in uplink.
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`, :attr:`~sionna.phy.config.Config.precision` is used.

    :output tx_power: [..., num_tx, num_streams_per_tx, num_ofdm_sym, num_subcarriers], `torch.float`.
        Transmit power [W] for each user, across subcarriers, streams,
        and OFDM symbols.

    .. rubric:: Examples

    .. code-block:: python

        import torch
        from sionna.sys.utils import spread_across_subcarriers

        batch_size = 2
        num_ofdm_sym = 14
        num_ut = 4
        num_subcarriers = 52
        num_streams = 2

        tx_power_per_ut = torch.ones(batch_size, num_ofdm_sym, num_ut)
        is_scheduled = torch.ones(batch_size, num_ofdm_sym, num_subcarriers,
                                  num_ut, num_streams, dtype=torch.bool)

        tx_power = spread_across_subcarriers(tx_power_per_ut, is_scheduled)
        print(tx_power.shape)
        # torch.Size([2, 4, 2, 14, 52])
    """
    if precision is None:
        dtype = config.dtype
    else:
        dtype = dtypes[precision]["torch"]["dtype"]

    tx_power_per_ut = tx_power_per_ut.to(dtype)
    # Ensure is_scheduled is on the same device as tx_power_per_ut
    is_scheduled = is_scheduled.to(device=tx_power_per_ut.device)
    num_ofdm_sym = is_scheduled.shape[-4]
    num_subcarriers = is_scheduled.shape[-3]
    num_ut = is_scheduled.shape[-2]
    num_streams_per_ut = is_scheduled.shape[-1]
    lbs = is_scheduled.dim() - 4

    if num_tx is None:
        num_tx = num_ut

    # [..., num_ofdm_sym, num_ut, num_subcarriers, num_streams_per_ut]
    is_scheduled = is_scheduled.permute(
        *range(lbs), lbs, lbs + 2, lbs + 1, lbs + 3
    )

    # [..., num_ofdm_sym, num_ut, 1, 1]
    tx_power = insert_dims(tx_power_per_ut, 2, axis=-1)
    # Tile to [..., num_ofdm_sym, num_ut, num_subcarriers, num_streams_per_ut]
    tx_power = tx_power.expand(*[-1] * (lbs + 2), num_subcarriers, num_streams_per_ut)
    # Mask according to scheduling decisions
    # [..., num_ofdm_sym, num_ut, num_subcarriers, num_streams_per_ut]
    tx_power = torch.where(
        is_scheduled,
        tx_power,
        torch.zeros(1, dtype=dtype, device=tx_power.device),
    )

    # N. allocated resources per user
    # [..., num_ofdm_sym, num_ut]
    num_allocated_re = is_scheduled.to(torch.int32).sum(dim=(-2, -1))
    # [..., num_ofdm_sym, num_ut, 1, 1]
    num_allocated_re = insert_dims(num_allocated_re, 2, axis=-1)

    # Spread power equally across streams for each user
    # [..., num_ofdm_sym, num_ut, num_subcarriers, num_streams_per_ut]
    tx_power = torch.where(
        num_allocated_re > 0,
        tx_power / num_allocated_re.to(dtype),
        torch.zeros(1, dtype=dtype, device=tx_power.device),
    )

    # [..., num_ut, num_streams_per_ut, num_ofdm_sym, num_subcarriers]
    tx_power = tx_power.permute(*range(lbs), lbs + 1, lbs + 3, lbs, lbs + 2)

    # [..., num_tx, num_streams_per_tx, num_ofdm_sym, num_subcarriers]
    tx_power = tx_power.reshape(
        *tx_power.shape[:-4], num_tx, -1, num_ofdm_sym, num_subcarriers
    )
    return tx_power
