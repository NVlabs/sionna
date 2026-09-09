#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Sionna System-Level (SYS) Package"""

from sionna.sys.effective_sinr import EESM, EffectiveSINR
from sionna.sys.link_adaptation import (
    InnerLoopLinkAdaptation,
    OuterLoopLinkAdaptation,
)
from sionna.sys.metrics import (
    coupling_loss_db,
    geometry_sinr_db,
    geometry_sir_db,
    received_power_dbm,
    serving_indices,
    wideband_sir_db,
)
from sionna.sys.phy_abstraction import PHYAbstraction
from sionna.sys.power_control import (
    downlink_fair_power_control,
    open_loop_uplink_power_control,
)
from sionna.sys.scheduling import PFSchedulerSUMIMO
from sionna.sys.topology import (
    HexGrid,
    Hexagon,
    IndoorFactoryTopology,
    convert_hex_coord,
    gen_hexgrid_topology,
    gen_tr38901_indoor_factory_topology,
    gen_tr38901_indoor_office_topology,
    gen_tr38901_multicell_topology,
    get_num_hex_in_grid,
)
from sionna.sys.utils import (
    get_pathloss,
    is_scheduled_in_slot,
    spread_across_subcarriers,
)

__all__ = [
    # effective_sinr
    "EffectiveSINR",
    "EESM",
    # link_adaptation
    "InnerLoopLinkAdaptation",
    "OuterLoopLinkAdaptation",
    # metrics
    "coupling_loss_db",
    "received_power_dbm",
    "serving_indices",
    "geometry_sir_db",
    "geometry_sinr_db",
    "wideband_sir_db",
    # phy_abstraction
    "PHYAbstraction",
    # power_control
    "open_loop_uplink_power_control",
    "downlink_fair_power_control",
    # scheduling
    "PFSchedulerSUMIMO",
    # topology
    "get_num_hex_in_grid",
    "convert_hex_coord",
    "Hexagon",
    "HexGrid",
    "IndoorFactoryTopology",
    "gen_hexgrid_topology",
    "gen_tr38901_multicell_topology",
    "gen_tr38901_indoor_office_topology",
    "gen_tr38901_indoor_factory_topology",
    # utils
    "is_scheduled_in_slot",
    "get_pathloss",
    "spread_across_subcarriers",
]
