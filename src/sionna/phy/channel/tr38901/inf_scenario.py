#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""3GPP TR 38.901 indoor factory (InF) channel scenario"""

from typing import Optional, Union

import torch

from sionna._validation import check_tensor_all
from .system_level_scenario import SystemLevelScenario
from .antenna import HandheldUTArray, PanelArray

__all__ = ["InFScenario"]


_FACTORY_SCENARIO_ALIASES = {
    "sl": "sl",
    "inf-sl": "sl",
    "sparse-low": "sl",
    "sparse-clutter-low-bs": "sl",
    "dl": "dl",
    "inf-dl": "dl",
    "dense-low": "dl",
    "dense-clutter-low-bs": "dl",
    "sh": "sh",
    "inf-sh": "sh",
    "sparse-high": "sh",
    "sparse-clutter-high-bs": "sh",
    "dh": "dh",
    "inf-dh": "dh",
    "dense-high": "dh",
    "dense-clutter-high-bs": "dh",
    "hh": "hh",
    "inf-hh": "hh",
    "high-high": "hh",
    "high-tx-high-rx": "hh",
}

_DEFAULT_HALL_DIMENSIONS = {
    "sl": (120.0, 60.0, 10.0),
    "dl": (300.0, 150.0, 10.0),
    "sh": (300.0, 150.0, 10.0),
    "dh": (120.0, 60.0, 10.0),
    "hh": (300.0, 150.0, 10.0),
}

_DEFAULT_CLUTTER = {
    "sl": (0.20, 10.0, 2.0),
    "dl": (0.60, 2.0, 6.0),
    "sh": (0.20, 10.0, 2.0),
    "dh": (0.60, 2.0, 6.0),
    "hh": (0.0, 1.0, 0.0),
}

_NLOS_SHADOW_FADING_STD = {
    "sl": 5.7,
    "dl": 7.2,
    "sh": 5.9,
    "dh": 4.0,
    "hh": 4.3,
}


def _factory_scenario(value: str) -> str:
    key = value.strip().lower().replace("_", "-").replace(" ", "-")
    if key not in _FACTORY_SCENARIO_ALIASES:
        raise ValueError(
            "factory_scenario must be one of 'SL', 'DL', 'SH', 'DH', or 'HH'"
        )
    return _FACTORY_SCENARIO_ALIASES[key]


class InFScenario(SystemLevelScenario):
    r"""
    3GPP TR 38.901 indoor factory (InF) channel model scenario.

    The model implements the InF scenarios from Section 7 of
    :cite:p:`TR38901V1920`. The ``factory_scenario`` parameter
    selects the InF sub-scenario from Table 7.2-4:

    * ``"SL"``: sparse clutter, low BS;
    * ``"DL"``: dense clutter, low BS;
    * ``"SH"``: sparse clutter, high BS;
    * ``"DH"``: dense clutter, high BS;
    * ``"HH"``: high Tx and high Rx, which is LOS-only.

    For InF-SL, InF-DL, InF-SH, and InF-DH, the default hall dimensions and
    clutter parameters follow the calibration assumptions from Table 7.8-7 of
    :cite:p:`TR38901V1920`. That table does not include InF-HH.
    For InF-HH only, the defaults ``(300, 150, 10)`` m, ``0.0``, ``1.0`` m,
    and ``0.0`` m for hall dimensions, clutter density, clutter size, and
    clutter height, respectively, are implementation assumptions rather than
    standardized calibration values. Pass explicit values for a particular
    InF-HH layout.

    :param carrier_frequency: Carrier frequency [Hz].
    :param factory_scenario: Indoor-factory sub-scenario. Must be ``"SL"``,
        ``"DL"``, ``"SH"``, ``"DH"``, or ``"HH"``. Defaults to ``"SH"``.
    :param ut_array: Antenna array used by UTs. This can be a
        :class:`~sionna.phy.channel.tr38901.PanelArray` or
        :class:`~sionna.phy.channel.tr38901.HandheldUTArray`.
    :param bs_array: Antenna array used by base stations. This can be a
        :class:`~sionna.phy.channel.tr38901.PanelArray` or
        :class:`~sionna.phy.channel.tr38901.HandheldUTArray`.
    :param direction: Link direction. Must be ``"uplink"`` or ``"downlink"``.
    :param hall_dimensions: Optional hall dimensions ``(length, width,
        height)`` [m]. If `None`, the sub-scenario default described above is
        used.
    :param clutter_density: Surface fraction occupied by clutter. If `None`,
        the sub-scenario default described above is used.
    :param clutter_size: Typical clutter size :math:`d_\mathrm{clutter}` [m].
        If `None`, the sub-scenario default described above is used.
    :param clutter_height: Effective clutter height :math:`h_c` [m]. If
        `None`, the sub-scenario default described above is used.
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
        InF is implemented for ``"16.1"`` and ``"19.2"``. Defaults to
        ``"19.2"``.
    """

    def __init__(
        self,
        carrier_frequency: float,
        factory_scenario: str,
        ut_array: PanelArray | HandheldUTArray,
        bs_array: PanelArray | HandheldUTArray,
        direction: str,
        hall_dimensions: Optional[tuple[float, float, float]] = None,
        clutter_density: Optional[float] = None,
        clutter_size: Optional[float] = None,
        clutter_height: Optional[float] = None,
        enable_pathloss: bool = True,
        enable_shadow_fading: bool = True,
        precision: Optional[str] = None,
        device: Optional[str] = None,
        spec_version: str = "19.2",
    ) -> None:
        self._factory_scenario = _factory_scenario(factory_scenario)
        if hall_dimensions is None:
            hall_dimensions = _DEFAULT_HALL_DIMENSIONS[self._factory_scenario]
        default_density, default_size, default_height = (
            _DEFAULT_CLUTTER[self._factory_scenario]
        )
        if clutter_density is None:
            clutter_density = default_density
        if clutter_size is None:
            clutter_size = default_size
        if clutter_height is None:
            clutter_height = default_height

        super().__init__(
            carrier_frequency,
            "high",
            ut_array,
            bs_array,
            direction,
            enable_pathloss,
            enable_shadow_fading,
            precision=precision,
            device=device,
            spec_version=spec_version,
        )

        self.register_buffer(
            "_hall_dimensions",
            torch.tensor(hall_dimensions, dtype=self.dtype, device=self.device),
        )
        self.register_buffer(
            "_clutter_density",
            torch.tensor(clutter_density, dtype=self.dtype, device=self.device),
        )
        self.register_buffer(
            "_clutter_size",
            torch.tensor(clutter_size, dtype=self.dtype, device=self.device),
        )
        self.register_buffer(
            "_clutter_height",
            torch.tensor(clutter_height, dtype=self.dtype, device=self.device),
        )

    #########################################
    # Public methods and properties
    #########################################

    @property
    def factory_scenario(self) -> str:
        """Canonical InF sub-scenario."""
        return self._factory_scenario

    @property
    def hall_dimensions(self) -> torch.Tensor:
        """Hall dimensions ``(length, width, height)`` [m]."""
        return self._hall_dimensions

    @property
    def clutter_density(self) -> torch.Tensor:
        """Surface fraction occupied by clutter."""
        return self._clutter_density

    @property
    def clutter_size(self) -> torch.Tensor:
        r"""Typical clutter size :math:`d_\mathrm{clutter}` [m]."""
        return self._clutter_size

    @property
    def clutter_height(self) -> torch.Tensor:
        """Effective clutter height :math:`h_c` [m]."""
        return self._clutter_height

    @property
    def use_indoor_lsp_params(self) -> bool:
        """Do not switch indoor InF links to O2I LSP parameters."""
        return False

    @property
    def indoor_links_can_be_los(self) -> bool:
        """Allow indoor-factory links to be LoS or NLoS."""
        return True

    @property
    def indoor_links_use_o2i_zenith_model(self) -> bool:
        """Use ordinary InF LoS/NLoS zenith-angle generation."""
        return False

    @property
    def o2i_pathloss_enabled(self) -> bool:
        """Disable outdoor-to-indoor penetration loss for InF links."""
        return False

    def clip_carrier_frequency_lsp(self, fc: torch.Tensor) -> torch.Tensor:
        r"""Return ``fc`` unchanged for InF LSP calculation.

        :param fc: Carrier frequency [GHz].

        :output fc: Unmodified carrier frequency [GHz].
        """
        return fc

    @property
    def min_2d_in(self) -> torch.Tensor:
        """Minimum indoor 2D distance for indoor UTs [m]."""
        return torch.tensor(0.0, dtype=self.dtype, device=self.device)

    @property
    def max_2d_in(self) -> torch.Tensor:
        """Maximum indoor 2D distance for indoor UTs [m]."""
        return torch.tensor(0.0, dtype=self.dtype, device=self.device)

    @property
    def los_probability(self) -> torch.Tensor:
        r"""Probability of each BS-UT link to be LoS.

        Computed from the InF formulas in Table 7.4.2-1 of
        :cite:p:`TR38901V1920`. For ``"HH"``, the LOS probability is one.
        If the clutter density is zero, the formula is evaluated with the
        limiting behavior of an unobstructed hall, i.e., LOS probability close
        to one.

        Shape [batch size, num_bs, num_ut].
        """
        if self.factory_scenario == "hh":
            return torch.ones_like(self.distance_2d)

        density = torch.clamp(
            self.clutter_density,
            min=torch.tensor(0.0, dtype=self.dtype, device=self.device),
            max=torch.tensor(1.0 - torch.finfo(self.dtype).eps,
                             dtype=self.dtype, device=self.device),
        )
        k = -self.clutter_size/torch.log1p(-density)

        if self.factory_scenario in ("sh", "dh"):
            h_bs = self.h_bs.unsqueeze(2)
            h_ut = self.h_ut.unsqueeze(1)
            denominator = torch.clamp(
                self.clutter_height - h_ut,
                min=torch.finfo(self.dtype).eps,
            )
            k = k*(h_bs - h_ut)/denominator

        k = torch.clamp(k, min=torch.finfo(self.dtype).eps)
        p = torch.exp(-self.distance_2d/k)
        return torch.clamp(p, min=0.0, max=1.0)

    @property
    def rays_per_cluster(self) -> int:
        """Number of rays per cluster."""
        return 20

    @property
    def los_parameter_filepath(self) -> str:
        """Path of the configuration file for LoS scenario."""
        return "InF_LoS.json"

    @property
    def nlos_parameter_filepath(self) -> str:
        """Path of the configuration file for NLoS scenario."""
        return "InF_NLoS.json"

    @property
    def o2i_parameter_filepath(self) -> str:
        """Path of the unused O2I configuration file."""
        return "InF_O2I.json"

    def set_topology(
        self,
        ut_loc: Optional[torch.Tensor] = None,
        bs_loc: Optional[torch.Tensor] = None,
        ut_orientations: Optional[torch.Tensor] = None,
        bs_orientations: Optional[torch.Tensor] = None,
        ut_velocities: Optional[torch.Tensor] = None,
        in_state: Optional[torch.Tensor] = None,
        los: Optional[Union[bool, str, torch.Tensor]] = None,
        bs_virtual_loc: Optional[torch.Tensor] = None,
        bs_site_ids: Optional[torch.Tensor] = None,
        spatial_consistency_track_ids: Optional[torch.Tensor] = None,
        distance_2d_in: Optional[torch.Tensor] = None,
        ut_spatial_region_ids: Optional[torch.Tensor] = None,
    ) -> bool:
        r"""Set the network topology.

        The arguments are identical to
        :meth:`~sionna.phy.channel.tr38901.SystemLevelScenario.set_topology`.
        The ``"HH"`` sub-scenario is LOS-only and rejects forced NLOS links.
        """
        if self.factory_scenario == "hh":
            message = "InF-HH is LOS-only and does not support forced NLOS"
            if isinstance(los, torch.Tensor):
                check_tensor_all(los, name="los", message=message)
            elif los is False:
                raise ValueError(message)
        return super().set_topology(
            ut_loc,
            bs_loc,
            ut_orientations,
            bs_orientations,
            ut_velocities,
            in_state,
            los,
            bs_virtual_loc,
            bs_site_ids,
            spatial_consistency_track_ids,
            distance_2d_in,
            ut_spatial_region_ids,
        )

    #########################
    # Utility methods
    #########################

    def _broadcast_state_value(
        self, los_value: torch.Tensor, nlos_value: torch.Tensor
    ) -> torch.Tensor:
        los_value = torch.as_tensor(los_value, dtype=self.dtype,
                                    device=self.device)
        nlos_value = torch.as_tensor(nlos_value, dtype=self.dtype,
                                     device=self.device)
        shape = (self.batch_size, self.num_bs, self.num_ut)
        los_value = los_value*torch.ones(shape, dtype=self.dtype,
                                         device=self.device)
        nlos_value = nlos_value*torch.ones(shape, dtype=self.dtype,
                                           device=self.device)
        return torch.where(self.los, los_value, nlos_value)

    def _hall_volume_surface_ratio(self) -> torch.Tensor:
        length, width, height = self.hall_dimensions
        volume = length*width*height
        surface = 2.0*(length*width + length*height + width*height)
        return volume/surface

    def _compute_lsp_log_mean_std(self) -> None:
        r"""Computes the mean and standard deviations of LSPs in log-domain."""

        batch_size = self.batch_size
        num_bs = self.num_bs
        num_ut = self.num_ut
        fc = self.carrier_frequency/1e9
        log_fc = torch.log10(1.0 + fc)
        volume_surface_ratio = self._hall_volume_surface_ratio()

        log_mean_ds_los = torch.log10(26.0*volume_surface_ratio + 14.0) - 9.35
        log_mean_ds_nlos = torch.log10(30.0*volume_surface_ratio + 32.0) - 9.44
        log_mean_ds = self._broadcast_state_value(
            log_mean_ds_los, log_mean_ds_nlos
        )
        log_mean_asd = self._broadcast_state_value(
            torch.tensor(1.56, dtype=self.dtype, device=self.device),
            torch.tensor(1.57, dtype=self.dtype, device=self.device),
        )
        log_mean_asa = self._broadcast_state_value(
            -0.18*log_fc + 1.78,
            torch.tensor(1.72, dtype=self.dtype, device=self.device),
        )
        log_mean_sf = torch.zeros(
            batch_size, num_bs, num_ut, dtype=self.dtype, device=self.device
        )
        log_mean_k = self._broadcast_state_value(
            torch.tensor(7.0/10.0, dtype=self.dtype, device=self.device),
            torch.tensor(0.0, dtype=self.dtype, device=self.device),
        )
        log_mean_zsa = self._broadcast_state_value(
            -0.2*log_fc + 1.5,
            -0.13*log_fc + 1.45,
        )
        log_mean_zsd = self._broadcast_state_value(
            torch.tensor(1.35, dtype=self.dtype, device=self.device),
            torch.tensor(1.2, dtype=self.dtype, device=self.device),
        )

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

        log_std_ds = self._broadcast_state_value(
            torch.tensor(0.15, dtype=self.dtype, device=self.device),
            torch.tensor(0.19, dtype=self.dtype, device=self.device),
        )
        log_std_asd = self._broadcast_state_value(
            torch.tensor(0.25, dtype=self.dtype, device=self.device),
            torch.tensor(0.20, dtype=self.dtype, device=self.device),
        )
        log_std_asa = self._broadcast_state_value(
            0.12*log_fc + 0.2,
            torch.tensor(0.3, dtype=self.dtype, device=self.device),
        )
        nlos_sigma_sf = _NLOS_SHADOW_FADING_STD[self.factory_scenario]
        log_std_sf = self._broadcast_state_value(
            torch.tensor(4.3/10.0, dtype=self.dtype, device=self.device),
            torch.tensor(nlos_sigma_sf/10.0, dtype=self.dtype,
                         device=self.device),
        )
        log_std_k = self._broadcast_state_value(
            torch.tensor(8.0/10.0, dtype=self.dtype, device=self.device),
            torch.tensor(0.0, dtype=self.dtype, device=self.device),
        )
        log_std_zsa = self._broadcast_state_value(
            torch.tensor(0.35, dtype=self.dtype, device=self.device),
            torch.tensor(0.45, dtype=self.dtype, device=self.device),
        )
        log_std_zsd = self._broadcast_state_value(
            torch.tensor(0.35, dtype=self.dtype, device=self.device),
            torch.tensor(0.55, dtype=self.dtype, device=self.device),
        )

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

        distance_3d = torch.clamp(self.distance_3d,
                                  min=torch.finfo(self.dtype).eps)
        fc = torch.clamp(self.carrier_frequency/1e9,
                         min=torch.finfo(self.dtype).eps)

        pl_los = 31.84 + 21.50*torch.log10(distance_3d) \
            + 19.00*torch.log10(fc)

        pl_sl = 33.0 + 25.5*torch.log10(distance_3d) + 20.0*torch.log10(fc)
        pl_dl = 18.6 + 35.7*torch.log10(distance_3d) + 20.0*torch.log10(fc)
        pl_sh = 32.4 + 23.0*torch.log10(distance_3d) + 20.0*torch.log10(fc)
        pl_dh = 33.63 + 21.9*torch.log10(distance_3d) + 20.0*torch.log10(fc)

        if self.factory_scenario == "sl":
            pl_nlos = torch.maximum(pl_los, pl_sl)
        elif self.factory_scenario == "dl":
            pl_nlos = torch.maximum(torch.maximum(pl_los, pl_sl), pl_dl)
        elif self.factory_scenario == "sh":
            pl_nlos = torch.maximum(pl_los, pl_sh)
        elif self.factory_scenario == "dh":
            pl_nlos = torch.maximum(pl_los, pl_dh)
        else:
            pl_nlos = pl_los

        pl_b = torch.where(self.los, pl_los, pl_nlos)
        self._update_attr("_pl_b", pl_b)
