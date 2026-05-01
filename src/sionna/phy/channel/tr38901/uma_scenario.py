#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""3GPP TR38.901 urban macrocell (UMa) channel scenario"""

from typing import Optional

import torch

from sionna.phy import SPEED_OF_LIGHT
from sionna.phy.utils import rand
from .system_level_scenario import SystemLevelScenario
from .antenna import PanelArray

__all__ = ["UMaScenario"]


class UMaScenario(SystemLevelScenario):
    r"""
    3GPP TR 38.901 urban macrocell (UMa) channel model scenario.

    :param carrier_frequency: Carrier frequency [Hz]
    :param o2i_model: Outdoor to indoor (O2I) pathloss model, used for indoor UTs.
        Must be 'low' or 'high'. See section 7.4.3 from 38.901 specification.
    :param ut_array: Panel array used by the UTs. All UTs share the same
        antenna array configuration.
    :param bs_array: Panel array used by the BSs. All BSs share the same
        antenna array configuration.
    :param direction: Link direction. Either 'uplink' or 'downlink'.
    :param enable_pathloss: If `True`, apply pathloss. Defaults to `True`.
    :param enable_shadow_fading: If `True`, apply shadow fading. Defaults to `True`.
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`, :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for computation (e.g., 'cpu', 'cuda:0').
        If `None`, :attr:`~sionna.phy.config.Config.device` is used.
    """

    def __init__(
        self,
        carrier_frequency: float,
        o2i_model: str,
        ut_array: PanelArray,
        bs_array: PanelArray,
        direction: str,
        enable_pathloss: bool = True,
        enable_shadow_fading: bool = True,
        precision: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        super().__init__(
            carrier_frequency,
            o2i_model,
            ut_array,
            bs_array,
            direction,
            enable_pathloss,
            enable_shadow_fading,
            precision=precision,
            device=device,
        )

    #########################################
    # Public methods and properties
    #########################################

    def clip_carrier_frequency_lsp(self, fc: torch.Tensor) -> torch.Tensor:
        r"""Clip the carrier frequency ``fc`` in GHz for LSP calculation

        :param fc: Carrier frequency [GHz]
        """
        # Use torch.maximum to avoid Python boolean comparison which causes graph breaks
        min_fc = torch.tensor(6.0, dtype=self.dtype, device=self.device)
        return torch.maximum(fc, min_fc)

    @property
    def min_2d_in(self) -> torch.Tensor:
        """Minimum indoor 2D distance for indoor UTs [m]"""
        return torch.tensor(0.0, dtype=self.dtype, device=self.device)

    @property
    def max_2d_in(self) -> torch.Tensor:
        """Maximum indoor 2D distance for indoor UTs [m]"""
        return torch.tensor(25.0, dtype=self.dtype, device=self.device)

    @property
    def los_probability(self) -> torch.Tensor:
        r"""Probability of each UT to be LoS. Used to randomly generate LoS
        status of outdoor UTs.

        Computed following section 7.4.2 of TR 38.901.

        Shape [batch size, num_bs, num_ut]
        """
        h_ut = self.h_ut
        c = torch.pow((h_ut - 13.0) / 10.0, 1.5)
        c = torch.where(
            h_ut < 13.0,
            torch.tensor(0.0, dtype=self.dtype, device=self.device),
            c,
        )
        c = c.unsqueeze(1)

        distance_2d_out = self._distance_2d_out
        los_probability = (
            18.0 / distance_2d_out
            + torch.exp(-distance_2d_out / 63.0) * (1.0 - 18.0 / distance_2d_out)
        ) * (
            1.0
            + c
            * 5.0
            / 4.0
            * torch.pow(distance_2d_out / 100.0, 3)
            * torch.exp(-distance_2d_out / 150.0)
        )

        los_probability = torch.where(
            distance_2d_out < 18.0,
            torch.tensor(1.0, dtype=self.dtype, device=self.device),
            los_probability,
        )
        return los_probability

    @property
    def rays_per_cluster(self) -> int:
        """Number of rays per cluster"""
        return 20

    @property
    def los_parameter_filepath(self) -> str:
        """Path of the configuration file for LoS scenario"""
        return "UMa_LoS.json"

    @property
    def nlos_parameter_filepath(self) -> str:
        """Path of the configuration file for NLoS scenario"""
        return "UMa_NLoS.json"

    @property
    def o2i_parameter_filepath(self) -> str:
        """Path of the configuration file for indoor scenario"""
        return "UMa_O2I.json"

    #########################
    # Utility methods
    #########################

    def _compute_lsp_log_mean_std(self) -> None:
        r"""Computes the mean and standard deviations of LSPs in log-domain"""

        batch_size = self.batch_size
        num_bs = self.num_bs
        num_ut = self.num_ut
        distance_2d = self.distance_2d
        h_bs = self.h_bs
        h_bs = h_bs.unsqueeze(2)  # For broadcasting
        h_ut = self.h_ut
        h_ut = h_ut.unsqueeze(1)  # For broadcasting

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
        # ZSD
        log_mean_zsd_los = torch.maximum(
            torch.tensor(-0.5, dtype=self.dtype, device=self.device),
            -2.1 * (distance_2d / 1000.0) - 0.01 * torch.abs(h_ut - 1.5) + 0.75,
        )
        log_mean_zsd_nlos = torch.maximum(
            torch.tensor(-0.5, dtype=self.dtype, device=self.device),
            -2.1 * (distance_2d / 1000.0) - 0.01 * torch.abs(h_ut - 1.5) + 0.9,
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
        # K. Given in dB in the 3GPP tables, hence the division by 10.
        log_std_k = self.get_param("sigmaK") / 10.0
        # ZSA
        log_std_zsa = self.get_param("sigmaZSA")
        # ZSD
        log_std_zsd = self.get_param("sigmaZSD")

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

        # ZOD offset
        fc = self.carrier_frequency / 1e9
        if fc < 6.0:
            fc = torch.tensor(6.0, dtype=self.dtype, device=self.device)
        a = 0.208 * torch.log10(fc) - 0.782
        b = torch.tensor(25.0, dtype=self.dtype, device=self.device)
        c = -0.13 * torch.log10(fc) + 2.03
        e = 7.66 * torch.log10(fc) - 5.96
        zod_offset = e - torch.pow(
            torch.tensor(10.0, dtype=self.dtype, device=self.device),
            a * torch.log10(torch.maximum(b, distance_2d)) + c - 0.07 * (h_ut - 1.5),
        )
        zod_offset = torch.where(
            self.los,
            torch.tensor(0.0, dtype=self.dtype, device=self.device),
            zod_offset,
        )
        self._update_attr("_zod_offset", zod_offset)

    def _compute_pathloss_basic(self) -> None:
        r"""Computes the basic component of the pathloss [dB]"""

        batch_size = self.batch_size
        num_bs = self.num_bs
        num_ut = self.num_ut
        distance_2d = self.distance_2d
        distance_3d = self.distance_3d
        fc = self.carrier_frequency  # Carrier frequency (Hz)
        h_bs = self.h_bs
        h_bs = h_bs.unsqueeze(2)  # For broadcasting
        h_ut = self.h_ut
        h_ut = h_ut.unsqueeze(1)  # For broadcasting

        # Break point distance
        g = (
            (5.0 / 4.0)
            * torch.pow(distance_2d / 100.0, 3.0)
            * torch.exp(-distance_2d / 150.0)
        )
        g = torch.where(
            distance_2d < 18.0,
            torch.tensor(0.0, dtype=self.dtype, device=self.device),
            g,
        )
        c = g * torch.pow((h_ut - 13.0) / 10.0, 1.5)
        c = torch.where(
            h_ut < 13.0,
            torch.tensor(0.0, dtype=self.dtype, device=self.device),
            c,
        )
        p = 1.0 / (1.0 + c)
        r = rand(
            (batch_size, num_bs, num_ut),
            dtype=self.dtype,
            device=self.device,
            generator=self.torch_rng,
        )
        r = torch.where(
            r < p,
            torch.tensor(1.0, dtype=self.dtype, device=self.device),
            torch.tensor(0.0, dtype=self.dtype, device=self.device),
        )

        max_value = h_ut - 1.5
        # Per TR 38.901 Table 7.4.1-1 Note 1, h_E must be sampled from a
        # discrete uniform distribution: {12, 15, 18, ..., h_UT - 1.5}.
        # torch.randint does not support non-scalar bounds, so we sample
        # U(0, 1) with scalar bounds and map to a discrete index instead.
        n_values = torch.floor(
            (max_value - torch.tensor(12.0, dtype=self.dtype, device=self.device))
            / torch.tensor(3.0, dtype=self.dtype, device=self.device)
        ) + torch.tensor(1.0, dtype=self.dtype, device=self.device)
        n_values = torch.maximum(
            n_values,
            torch.tensor(1.0, dtype=self.dtype, device=self.device),
        )
        u = rand(
            (batch_size, num_bs, num_ut),
            dtype=self.dtype,
            device=self.device,
            generator=self.torch_rng,
        )
        s = (
            torch.tensor(12.0, dtype=self.dtype, device=self.device)
            + torch.tensor(3.0, dtype=self.dtype, device=self.device)
            * torch.floor(u * n_values)
        )
        # Edge case: when h_ut <= 13.5m, max_value <= 12m, only valid value is 12
        s = torch.where(
            max_value < torch.tensor(12.0, dtype=self.dtype, device=self.device),
            torch.tensor(12.0, dtype=self.dtype, device=self.device),
            s,
        )

        h_e = r + (1.0 - r) * s
        self._update_attr("_h_e", h_e)
        h_bs_prime = h_bs - h_e
        h_ut_prime = h_ut - h_e
        distance_breakpoint = 4 * h_bs_prime * h_ut_prime * fc / SPEED_OF_LIGHT

        ## Basic path loss for LoS

        pl_1 = 28.0 + 22.0 * torch.log10(distance_3d) + 20.0 * torch.log10(fc / 1e9)
        pl_2 = (
            28.0
            + 40.0 * torch.log10(distance_3d)
            + 20.0 * torch.log10(fc / 1e9)
            - 9.0
            * torch.log10(torch.square(distance_breakpoint) + torch.square(h_bs - h_ut))
        )
        pl_los = torch.where(distance_2d < distance_breakpoint, pl_1, pl_2)

        ## Basic pathloss for NLoS and O2I

        pl_3 = (
            13.54
            + 39.08 * torch.log10(distance_3d)
            + 20.0 * torch.log10(fc / 1e9)
            - 0.6 * (h_ut - 1.5)
        )
        pl_nlos = torch.maximum(pl_los, pl_3)

        ## Set the basic pathloss according to UT state

        # LoS
        pl_b = torch.where(self.los, pl_los, pl_nlos)

        self._update_attr("_pl_b", pl_b)
