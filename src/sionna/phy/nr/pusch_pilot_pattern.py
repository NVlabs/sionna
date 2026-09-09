#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""PUSCH pilot pattern for the 5G NR module of Sionna PHY."""

from collections.abc import Sequence
from typing import List, Optional, Union
import warnings
import numpy as np

from sionna.phy.ofdm import PilotPattern
from .pusch_config import PUSCHConfig


__all__ = ["PUSCHPilotPattern"]


class PUSCHPilotPattern(PilotPattern):
    r"""Pilot pattern for NR PUSCH.

    This class defines a :class:`~sionna.phy.ofdm.PilotPattern`
    that is used to configure an OFDM :class:`~sionna.phy.ofdm.ResourceGrid`.

    For every transmitter, a separate :class:`~sionna.phy.nr.PUSCHConfig`
    needs to be provided from which the pilot pattern will be created.

    :param pusch_configs: PUSCH configurations according to which the pilot
        pattern will be created. One configuration is needed for each
        transmitter.
    :param precision: `None` (default) | ``"single"`` | ``"double"``.
        Precision used for internal calculations and outputs.
        If set to `None`,
        :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for tensor operations. If `None`,
        :attr:`~sionna.phy.config.Config.device` is used.

    .. rubric:: Examples

    .. code-block:: python

        from sionna.phy.nr import PUSCHConfig, PUSCHPilotPattern

        pusch_config = PUSCHConfig()
        pilot_pattern = PUSCHPilotPattern(pusch_config)
        print(pilot_pattern.mask.shape)
    """

    def __init__(
        self,
        pusch_configs: Union[PUSCHConfig, List[PUSCHConfig]],
        precision: Optional[str] = None,
        device: Optional[str] = None,
    ):
        # Check correct type of pusch_configs
        if isinstance(pusch_configs, PUSCHConfig):
            pusch_configs = [pusch_configs]
        elif isinstance(pusch_configs, Sequence):
            for c in pusch_configs:
                if not isinstance(c, PUSCHConfig):
                    raise TypeError(
                        "Each element of pusch_configs must be a valid PUSCHConfig")
        else:
            raise ValueError("Invalid value for pusch_configs")

        # Check validity of provided pusch_configs
        num_tx = len(pusch_configs)
        num_streams_per_tx = pusch_configs[0].num_layers
        dmrs_grid = pusch_configs[0].dmrs_grid
        num_subcarriers = dmrs_grid[0].shape[0]
        num_ofdm_symbols = pusch_configs[0].l_d
        precoding = pusch_configs[0].precoding
        dmrs_ports = []
        num_pilots = np.sum(pusch_configs[0].dmrs_mask)

        for pusch_config in pusch_configs:
            if pusch_config.num_layers != num_streams_per_tx:
                raise ValueError(
                    "All pusch_configs must have the same number of layers")
            if pusch_config.dmrs_grid[0].shape[0] != num_subcarriers:
                raise ValueError(
                    "All pusch_configs must have the same number of subcarriers")
            if pusch_config.l_d != num_ofdm_symbols:
                raise ValueError(
                    "All pusch_configs must have the same number of OFDM symbols")
            if pusch_config.precoding != precoding:
                raise ValueError(
                    "All pusch_configs must have the same precoding method")
            if np.sum(pusch_config.dmrs_mask) != num_pilots:
                raise ValueError(
                    "All pusch_configs must have the same number of masked REs")

            with warnings.catch_warnings():
                warnings.simplefilter('always')
                for port in pusch_config.dmrs.dmrs_port_set:
                    if port in dmrs_ports:
                        msg = f"DMRS port {port} used by multiple transmitters"
                        warnings.warn(msg, UserWarning, stacklevel=2)
            dmrs_ports += pusch_config.dmrs.dmrs_port_set

        # Create mask and pilots tensors
        mask = np.zeros([num_tx, num_streams_per_tx, num_ofdm_symbols,
                         num_subcarriers], bool)
        num_pilots = np.sum(pusch_configs[0].dmrs_mask)
        pilots = np.zeros([num_tx, num_streams_per_tx, num_pilots], complex)

        for i, pusch_config in enumerate(pusch_configs):
            for j in range(num_streams_per_tx):
                ind0, ind1 = pusch_config.symbol_allocation
                mask[i, j] = np.transpose(
                    pusch_config.dmrs_mask[:, ind0:ind0 + ind1])
                dmrs_grid = np.transpose(
                    pusch_config.dmrs_grid[j, :, ind0:ind0 + ind1])
                pilots[i, j] = dmrs_grid[np.where(mask[i, j])]

        # Init PilotPattern class
        super().__init__(
            mask, pilots, normalize=False, precision=precision, device=device
        )

