#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""3GPP TR 38.901 indoor hotspot (InH) channel scenario"""

from typing import Optional

import torch

from .system_level_scenario import SystemLevelScenario
from .antenna import HandheldUTArray, PanelArray

__all__ = ["InHScenario"]


class InHScenario(SystemLevelScenario):
    r"""
    3GPP TR 38.901 indoor hotspot (InH) channel model scenario.

    This scenario implements the indoor-office model from Section 7 of
    :cite:p:`TR38901V1920`. The ``indoor_scenario`` parameter
    selects the open-office or mixed-office line-of-sight probability from
    Table 7.4.2-1. All other pathloss and fast-fading parameters are identical
    for the two variants.

    :param carrier_frequency: Carrier frequency [Hz]
    :param indoor_scenario: Indoor-office line-of-sight probability model.
        Must be ``"open"`` or ``"mixed"``.
    :param ut_array: Antenna array used by UTs. This can be a
        :class:`~sionna.phy.channel.tr38901.PanelArray` or
        :class:`~sionna.phy.channel.tr38901.HandheldUTArray`.
    :param bs_array: Antenna array used by base stations. This can be a
        :class:`~sionna.phy.channel.tr38901.PanelArray` or
        :class:`~sionna.phy.channel.tr38901.HandheldUTArray`.
    :param direction: Link direction. Must be ``"uplink"`` or ``"downlink"``.
    :param enable_pathloss: If `True`, apply pathloss. Otherwise doesn't.
        Defaults to `True`.
    :param enable_shadow_fading: If `True`, apply shadow fading.
        Otherwise doesn't. Defaults to `True`.
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`,
        :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for computation (e.g., ``"cpu"``, ``"cuda:0"``).
        If `None`, :attr:`~sionna.phy.config.Config.device` is used.
    :param spec_version: Version of the TR 38.901 parameter tables to use.
        Supported values are ``"16.1"`` and ``"19.2"``. Defaults to
        ``"19.2"``.
    """

    def __init__(
        self,
        carrier_frequency: float,
        indoor_scenario: str,
        ut_array: PanelArray | HandheldUTArray,
        bs_array: PanelArray | HandheldUTArray,
        direction: str,
        enable_pathloss: bool = True,
        enable_shadow_fading: bool = True,
        precision: Optional[str] = None,
        device: Optional[str] = None,
        spec_version: str = "19.2",
    ) -> None:
        if indoor_scenario not in ("open", "mixed"):
            raise ValueError("indoor_scenario must be 'open' or 'mixed'")
        self._indoor_scenario = indoor_scenario

        super().__init__(
            carrier_frequency,
            "low",
            ut_array,
            bs_array,
            direction,
            enable_pathloss,
            enable_shadow_fading,
            spec_version=spec_version,
            precision=precision,
            device=device,
        )

    #########################################
    # Public methods and properties
    #########################################

    @property
    def indoor_scenario(self) -> str:
        """Indoor-office line-of-sight probability model."""
        return self._indoor_scenario

    @property
    def use_indoor_lsp_params(self) -> bool:
        """Do not switch indoor InH links to O2I LSP parameters."""
        return False

    @property
    def indoor_links_can_be_los(self) -> bool:
        """Allow indoor-office links to be LoS or NLoS."""
        return True

    @property
    def indoor_links_use_o2i_zenith_model(self) -> bool:
        """Use ordinary InH LoS/NLoS zenith-angle generation for indoor links."""
        return False

    @property
    def o2i_pathloss_enabled(self) -> bool:
        """Disable outdoor-to-indoor penetration loss for InH links."""
        return False

    def clip_carrier_frequency_lsp(self, fc: torch.Tensor) -> torch.Tensor:
        r"""Clip the carrier frequency ``fc`` in GHz for LSP calculation.

        For InH, Table 7.5-6 of :cite:p:`TR38901V1920`
        specifies that frequencies below 6 GHz use 6 GHz for
        frequency-dependent LSP values.

        :param fc: Carrier frequency [GHz]

        :output fc_clipped: Clipped carrier frequency used for LSP computation
        """
        min_fc = torch.tensor(6.0, dtype=self.dtype, device=self.device)
        return torch.maximum(fc, min_fc)

    @property
    def min_2d_in(self) -> torch.Tensor:
        """Minimum indoor 2D distance for indoor UTs [m]"""
        return torch.tensor(0.0, dtype=self.dtype, device=self.device)

    @property
    def max_2d_in(self) -> torch.Tensor:
        """Maximum indoor 2D distance for indoor UTs [m]"""
        return torch.tensor(0.0, dtype=self.dtype, device=self.device)

    @property
    def los_probability(self) -> torch.Tensor:
        r"""Probability of each BS-UT link to be LoS.

        Computed from the indoor open-office or mixed-office formulas in
        Table 7.4.2-1 of :cite:p:`TR38901V1920`.

        Shape [batch size, num_bs, num_ut]
        """
        distance_2d = self.distance_2d
        if self.indoor_scenario == "open":
            los_probability = torch.where(
                distance_2d <= 5.0,
                torch.tensor(1.0, dtype=self.dtype, device=self.device),
                torch.where(
                    distance_2d <= 49.0,
                    torch.exp(-(distance_2d - 5.0) / 70.8),
                    torch.exp(-(distance_2d - 49.0) / 211.7) * 0.54,
                ),
            )
        else:
            los_probability = torch.where(
                distance_2d <= 1.2,
                torch.tensor(1.0, dtype=self.dtype, device=self.device),
                torch.where(
                    distance_2d < 6.5,
                    torch.exp(-(distance_2d - 1.2) / 4.7),
                    torch.exp(-(distance_2d - 6.5) / 32.6) * 0.32,
                ),
            )
        return los_probability

    @property
    def rays_per_cluster(self) -> int:
        """Number of rays per cluster"""
        return 20

    @property
    def los_parameter_filepath(self) -> str:
        """Path of the configuration file for LoS scenario"""
        return "InH_LoS.json"

    @property
    def nlos_parameter_filepath(self) -> str:
        """Path of the configuration file for NLoS scenario"""
        return "InH_NLoS.json"

    @property
    def o2i_parameter_filepath(self) -> str:
        """Path of the unused O2I configuration file"""
        return "InH_O2I.json"

    #########################
    # Utility methods
    #########################

    def _compute_lsp_log_mean_std(self) -> None:
        r"""Computes the mean and standard deviations of LSPs in log-domain."""

        batch_size = self.batch_size
        num_bs = self.num_bs
        num_ut = self.num_ut
        fc = self.clip_carrier_frequency_lsp(self.carrier_frequency / 1e9)
        log_fc = torch.log10(1.0 + fc)

        ## Mean
        # DS
        log_mean_ds = self.get_param("muDS")
        # ASD
        log_mean_asd = self.get_param("muASD")
        # ASA
        log_mean_asa = self.get_param("muASA")
        # SF. Has zero-mean.
        log_mean_sf = torch.zeros(
            batch_size, num_bs, num_ut, dtype=self.dtype, device=self.device
        )
        # K. Given in dB in the 3GPP tables, hence the division by 10
        log_mean_k = self.get_param("muK") / 10.0
        # ZSA
        log_mean_zsa = self.get_param("muZSA")
        # ZSD from TR 38.901 Table 7.5-10.
        log_mean_zsd_los = -1.43 * log_fc + 2.228
        log_mean_zsd_nlos = torch.tensor(
            1.08, dtype=self.dtype, device=self.device
        )
        log_mean_zsd = torch.where(self.los, log_mean_zsd_los, log_mean_zsd_nlos)

        lsp_log_mean = torch.stack(
            [
                log_mean_ds,
                log_mean_asd,
                log_mean_asa,
                log_mean_sf,
                log_mean_k,
                log_mean_zsa,
                log_mean_zsd,
            ],
            dim=3,
        )

        ## STD
        # DS
        log_std_ds = self.get_param("sigmaDS")
        # ASD
        log_std_asd = self.get_param("sigmaASD")
        # ASA
        log_std_asa = self.get_param("sigmaASA")
        # SF. Given in dB in the 3GPP tables, hence the division by 10
        log_std_sf = self.get_param("sigmaSF") / 10.0
        # K. Given in dB in the 3GPP tables, hence the division by 10
        log_std_k = self.get_param("sigmaK") / 10.0
        # ZSA
        log_std_zsa = self.get_param("sigmaZSA")
        # ZSD from TR 38.901 Table 7.5-10.
        log_std_zsd_los = 0.13 * log_fc + 0.30
        log_std_zsd_nlos = torch.tensor(
            0.36, dtype=self.dtype, device=self.device
        )
        log_std_zsd = torch.where(self.los, log_std_zsd_los, log_std_zsd_nlos)

        lsp_log_std = torch.stack(
            [
                log_std_ds,
                log_std_asd,
                log_std_asa,
                log_std_sf,
                log_std_k,
                log_std_zsa,
                log_std_zsd,
            ],
            dim=3,
        )

        self._update_attr("_lsp_log_mean", lsp_log_mean)
        self._update_attr("_lsp_log_std", lsp_log_std)

        zod_offset = torch.zeros(
            batch_size, num_bs, num_ut, dtype=self.dtype, device=self.device
        )
        self._update_attr("_zod_offset", zod_offset)

    def _compute_pathloss_basic(self) -> None:
        r"""Computes the basic component of the pathloss [dB]."""

        distance_3d = self.distance_3d
        fc = self.carrier_frequency / 1e9

        pl_los = 32.4 + 17.3 * torch.log10(distance_3d) + 20.0 * torch.log10(fc)
        pl_nlos_prime = (
            38.3 * torch.log10(distance_3d)
            + 17.30
            + 24.9 * torch.log10(fc)
        )
        pl_nlos = torch.maximum(pl_los, pl_nlos_prime)
        pl_b = torch.where(self.los, pl_los, pl_nlos)

        self._update_attr("_pl_b", pl_b)
