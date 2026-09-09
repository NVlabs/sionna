#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""3GPP TR38.901 rural macrocell (RMa) channel scenario"""

from typing import Optional, Union

import torch

from sionna._validation import check_tensor_all
from sionna.phy import SPEED_OF_LIGHT, PI
from sionna.phy.channel.utils import rad_2_deg
from .system_level_scenario import SystemLevelScenario
from .antenna import HandheldUTArray, PanelArray

__all__ = ["RMaScenario"]


class RMaScenario(SystemLevelScenario):
    r"""
    3GPP TR 38.901 rural macrocell (RMa) channel model scenario.

    :param carrier_frequency: Carrier frequency [Hz]
    :param ut_array: Antenna array used by UTs. This can be a
        :class:`~sionna.phy.channel.tr38901.PanelArray` or
        :class:`~sionna.phy.channel.tr38901.HandheldUTArray`.
    :param bs_array: Antenna array used by base stations. This can be a
        :class:`~sionna.phy.channel.tr38901.PanelArray` or
        :class:`~sionna.phy.channel.tr38901.HandheldUTArray`.
    :param direction: Link direction. Either ``"uplink"`` or ``"downlink"``.
    :param enable_pathloss: If `True`, apply pathloss. Otherwise don't.
        Defaults to `True`.
    :param enable_shadow_fading: If `True`, apply shadow fading. Otherwise
        don't. Defaults to `True`.
    :param average_street_width: Average street width [m]. Defaults to 20.0.
    :param average_building_height: Average building height [m]. Defaults to
        5.0.
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`, :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for computation (e.g., 'cpu', 'cuda:0').
        If `None`, :attr:`~sionna.phy.config.Config.device` is used.
    :param spec_version: Version of the TR 38.901 parameter tables to use.
        Supported values are ``"16.1"`` and ``"19.2"``. Defaults to
        ``"19.2"``.
    :param car_window_type: Car-window type for the car penetration model of
        Section 7.4.3.2. Must be ``"ordinary"`` (9 dB mean) or
        ``"metallized"`` (20 dB mean). Defaults to ``"ordinary"``. The car
        penetration loss is sampled once per in-car UT and shared by all of its
        BS links.

    .. rubric:: Examples

    >>> from sionna.phy.channel.tr38901 import PanelArray, RMaScenario
    >>> # Configure antenna arrays
    >>> ut_array = PanelArray(num_rows_per_panel=1,
    ...                       num_cols_per_panel=1,
    ...                       polarization="single",
    ...                       polarization_type="V",
    ...                       antenna_pattern="omni",
    ...                       carrier_frequency=3.5e9)
    >>> bs_array = PanelArray(num_rows_per_panel=4,
    ...                       num_cols_per_panel=4,
    ...                       polarization="dual",
    ...                       polarization_type="cross",
    ...                       antenna_pattern="38.901",
    ...                       carrier_frequency=3.5e9)
    >>> scenario = RMaScenario(carrier_frequency=3.5e9,
    ...                        ut_array=ut_array,
    ...                        bs_array=bs_array,
    ...                        direction="downlink")
    """

    def __init__(
        self,
        carrier_frequency: float,
        ut_array: PanelArray | HandheldUTArray,
        bs_array: PanelArray | HandheldUTArray,
        direction: str,
        enable_pathloss: bool = True,
        enable_shadow_fading: bool = True,
        average_street_width: float = 20.0,
        average_building_height: float = 5.0,
        precision: Optional[str] = None,
        device: Optional[str] = None,
        spec_version: str = "19.2",
        car_window_type: str = "ordinary",
    ) -> None:
        # Only the low-loss O2I model is available for RMa.
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

        if not isinstance(car_window_type, str):
            raise TypeError("car_window_type must be a string")
        car_window_type = car_window_type.lower()
        if car_window_type not in ("ordinary", "metallized"):
            raise ValueError(
                "car_window_type must be 'ordinary' or 'metallized'"
            )
        self._car_window_type = car_window_type
        car_loss_mean = 9.0 if car_window_type == "ordinary" else 20.0
        self.register_buffer(
            "_car_penetration_loss_mean",
            torch.tensor(car_loss_mean, dtype=self.dtype, device=self.device),
        )
        self._in_car: Optional[torch.Tensor] = None
        self._in_car_initialized = False
        self._in_car_explicit = False

        # Average street width [m]
        # Register as buffers for CUDAGraph compatibility
        self.register_buffer(
            "_average_street_width",
            torch.tensor(average_street_width, dtype=self.dtype, device=self.device),
        )

        # Average building height [m]
        self.register_buffer(
            "_average_building_height",
            torch.tensor(average_building_height, dtype=self.dtype, device=self.device),
        )

    #########################################
    # Public methods and properties
    #########################################

    def clip_carrier_frequency_lsp(self, fc: torch.Tensor) -> torch.Tensor:
        r"""Clip the carrier frequency ``fc`` in GHz for LSP calculation.

        :param fc: Carrier frequency [GHz]

        :output fc_clipped: `float`.
            Clipped carrier frequency, that should be used for LSP computation.
        """
        return fc

    @property
    def min_2d_in(self) -> torch.Tensor:
        """Minimum indoor 2D distance for indoor UTs [m]"""
        return torch.tensor(0.0, dtype=self.dtype, device=self.device)

    @property
    def max_2d_in(self) -> torch.Tensor:
        """Maximum indoor 2D distance for indoor UTs [m]"""
        return torch.tensor(10.0, dtype=self.dtype, device=self.device)

    @property
    def average_street_width(self) -> torch.Tensor:
        """Average street width [m]"""
        return self._average_street_width

    @property
    def average_building_height(self) -> torch.Tensor:
        """Average building height [m]"""
        return self._average_building_height

    @property
    def car_window_type(self) -> str:
        """Car-window type used by the car penetration model."""
        return self._car_window_type

    @property
    def car_penetration_loss_mean(self) -> torch.Tensor:
        """Mean car penetration loss [dB]."""
        return self._car_penetration_loss_mean

    @property
    def in_car(self) -> torch.Tensor:
        """In-car state of UTs. Shape [batch size, number of UTs]."""
        return self._in_car

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
        in_car: Optional[torch.Tensor] = None,
    ) -> bool:
        r"""Set the RMa topology and optional UT-specific in-car state.

        Unspecified parameters reuse their value from the previous call.
        Parameters that have never been set must be provided on the first
        call.

        If ``in_car`` is omitted on the first call, every non-indoor UT is
        treated as in-car, matching the default population in Table 7.2-3 of
        :cite:p:`TR38901V1920`. This inferred mask follows later
        ``in_state`` updates. Once ``in_car`` is supplied explicitly, omission
        on later calls reuses that explicit mask. Set ``in_car=False`` for
        pedestrian or otherwise unprotected outdoor UTs.

        :param ut_loc: Locations of the UTs [m].
            Shape [batch size, number of UTs, 3].
        :param bs_loc: Locations of the base stations [m].
            Shape [batch size, number of base stations, 3].
        :param ut_orientations: Orientations of the UT arrays [radian].
            Shape [batch size, number of UTs, 3].
        :param bs_orientations: Orientations of the BS arrays [radian].
            Shape [batch size, number of base stations, 3].
        :param ut_velocities: Velocity vectors of the UTs [m/s].
            Shape [batch size, number of UTs, 3].
        :param in_state: Indoor state of every UT. `True` means indoor and
            `False` means non-indoor. Shape [batch size, number of UTs].
        :param los: LoS/NLoS state control. A scalar boolean forces that state
            for every outdoor link. A boolean tensor specifies each link with
            shape [batch size, number of base stations, number of UTs] or
            [number of base stations, number of UTs]. ``"random"`` draws fresh states
            following Section 7.4.2; `None` reuses the previous setting and is
            equivalent to ``"random"`` on the first call.
        :param bs_virtual_loc: Virtual BS locations for each UT [m], used for
            wraparound distances and angles. If omitted while ``bs_loc`` is
            supplied, the physical BS locations are used.
            Shape [batch size, number of base stations, number of UTs, 3].
        :param bs_site_ids: Site identifier of each BS. Co-sited base stations share
            site-level random quantities. If omitted, exact duplicate BS
            locations are treated as co-sited. Shape [number of base stations] or
            [batch size, number of base stations].
        :param spatial_consistency_track_ids: Optional grouping identifiers
            for UT entries representing positions on the same track in the
            current topology snapshot. Equal identifiers share
            cluster-specific angle signs and random ray-coupling permutations.
            Shape [number of UTs] or [batch size, number of UTs].
        :param distance_2d_in: Optional pre-sampled indoor 2D distance [m] for
            every UT. Values for non-indoor UTs are ignored.
            Shape [batch size, number of UTs].
        :param ut_spatial_region_ids: Optional correlation-region identifier
            for every UT. Unequal identifiers decorrelate supported spatial
            random fields without changing pathloss or geometry.
            Shape [number of UTs] or [batch size, number of UTs].
        :param in_car: In-car state of every UT. In-car and indoor states are
            mutually exclusive. Shape [batch size, number of UTs].

        :output updated: `True` if the topology was updated, `False` otherwise.
        """
        in_car_tensor = None
        if in_car is not None:
            in_car_tensor = torch.as_tensor(in_car, device=self.device)
            if in_car_tensor.dtype != torch.bool:
                raise TypeError("`in_car` must have dtype torch.bool")
            prospective_ut_loc = ut_loc if ut_loc is not None else self._ut_loc
            if (
                prospective_ut_loc is not None
                and in_car_tensor.shape != prospective_ut_loc.shape[:2]
            ):
                raise ValueError(
                    "`in_car` must have shape [batch size, number of UTs]"
                )

        prospective_indoor = (
            torch.as_tensor(in_state, device=self.device)
            if in_state is not None
            else self._in_state
        )
        if (
            prospective_indoor is not None
            and prospective_indoor.dtype == torch.bool
        ):
            if in_car_tensor is not None:
                prospective_in_car = in_car_tensor
            elif self._in_car_explicit:
                prospective_in_car = self._in_car
            elif in_state is not None or not self._in_car_initialized:
                prospective_in_car = ~prospective_indoor
            else:
                prospective_in_car = self._in_car

            if prospective_in_car.shape == prospective_indoor.shape:
                check_tensor_all(
                    ~(prospective_in_car & prospective_indoor),
                    name="in_car",
                    message="A UT cannot be both indoor and in-car",
                )

        updated = super().set_topology(
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

        car_state_updated = False
        if in_car_tensor is not None:
            self._update_attr("_in_car", in_car_tensor)
            self._in_car_initialized = True
            self._in_car_explicit = True
            car_state_updated = True
        elif (
            not self._in_car_initialized
            or (in_state is not None and not self._in_car_explicit)
        ):
            self._update_attr("_in_car", ~self.indoor)
            self._in_car_initialized = True
            car_state_updated = True

        return updated or car_state_updated

    def reset_topology(self) -> None:
        """Reset topology-dependent RMa state."""
        super().reset_topology()
        if hasattr(self, "_in_car"):
            delattr(self, "_in_car")
        self._in_car = None
        self._in_car_initialized = False
        self._in_car_explicit = False

    def allocate_topology_tensors(
        self, batch_size: int, num_bs: int, num_ut: int
    ) -> None:
        r"""Pre-allocate topology-dependent RMa tensors.

        This is required before the first topology update inside a
        :func:`torch.compile`-decorated function. Calling it again resets the
        current topology and allocates tensors with the requested shapes.

        :param batch_size: Batch size.
        :param num_bs: Number of base stations.
        :param num_ut: Number of user terminals.
        """
        super().allocate_topology_tensors(batch_size, num_bs, num_ut)
        self._register_buffer_safe(
            "_in_car",
            torch.zeros(
                batch_size, num_ut, dtype=torch.bool, device=self.device
            ),
        )
        self._in_car_initialized = False
        self._in_car_explicit = False

    @property
    def los_probability(self) -> torch.Tensor:
        r"""Probability of each UT to be LoS. Used to randomly generate LoS
        status of outdoor UTs.

        Computed following section 7.4.2 of TR 38.901.

        Shape [batch size, num_bs, num_ut]
        """
        distance_2d_out = self._distance_2d_out
        los_probability = torch.exp(-(distance_2d_out - 10.0) / 1000.0)
        los_probability = torch.where(
            distance_2d_out < 10.0,
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
        return "RMa_LoS.json"

    @property
    def nlos_parameter_filepath(self) -> str:
        """Path of the configuration file for NLoS scenario"""
        return "RMa_NLoS.json"

    @property
    def o2i_parameter_filepath(self) -> str:
        """Path of the configuration file for indoor scenario"""
        return "RMa_O2I.json"

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
        # ZSD mean from TR 38.901 Table 7.5-9.
        log_mean_zsd_los = torch.maximum(
            torch.tensor(-1.0, dtype=self.dtype, device=self.device),
            -0.17*(distance_2d/1000.0) - 0.01*(h_ut - 1.5) + 0.22,
        )
        log_mean_zsd_nlos = torch.maximum(
            torch.tensor(-1.0, dtype=self.dtype, device=self.device),
            -0.19*(distance_2d/1000.0) - 0.01*(h_ut - 1.5) + 0.28,
        )
        log_mean_zsd = torch.where(
            self.los, log_mean_zsd_los, log_mean_zsd_nlos
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

        ## STD
        # DS
        log_std_ds = self.get_param("sigmaDS")
        # ASD
        log_std_asd = self.get_param("sigmaASD")
        # ASA
        log_std_asa = self.get_param("sigmaASA")
        # SF. Given in dB in the 3GPP tables, hence the division by 10
        # O2I and NLoS cases just require the use of a predefined value
        log_std_sf_o2i_nlos = self.get_param("sigmaSF") / 10.0
        # For LoS, two possible scenarios depending on the 2D location of the user
        distance_breakpoint = (
            2.0 * PI * h_bs * h_ut * self.carrier_frequency / SPEED_OF_LIGHT
        )
        log_std_sf_los = torch.where(
            distance_2d < distance_breakpoint,
            self.get_param("sigmaSF1") / 10.0,
            self.get_param("sigmaSF2") / 10.0,
        )
        # Use the correct SF STD according to the user scenario: NLoS/O2I, or LoS
        log_std_sf = torch.where(self.los, log_std_sf_los, log_std_sf_o2i_nlos)
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
        zod_offset = rad_2_deg(
            torch.atan(
                torch.tensor(31.5, dtype=self.dtype, device=self.device)
                / distance_2d
            )
            - torch.atan(
                torch.tensor(33.5, dtype=self.dtype, device=self.device)
                / distance_2d
            )
        )
        zod_offset = torch.where(
            self.los,
            torch.tensor(0.0, dtype=self.dtype, device=self.device),
            zod_offset,
        )
        self._update_attr("_zod_offset", zod_offset)

    def _compute_pathloss_basic(self) -> None:
        r"""Computes the basic component of the pathloss [dB]"""

        distance_2d = self.distance_2d
        distance_3d = self.distance_3d
        fc = self.carrier_frequency / 1e9  # Carrier frequency (GHz)
        h_bs = self.h_bs
        h_bs = h_bs.unsqueeze(2)  # For broadcasting
        h_ut = self.h_ut
        h_ut = h_ut.unsqueeze(1)  # For broadcasting
        average_building_height = self.average_building_height

        # Break point distance
        # For this computation, the carrier frequency needs to be in Hz
        distance_breakpoint = (
            2.0 * PI * h_bs * h_ut * self.carrier_frequency / SPEED_OF_LIGHT
        )

        ## Basic path loss for LoS

        pl_1 = (
            20.0 * torch.log10(40.0 * PI * distance_3d * fc / 3.0)
            + torch.minimum(
                0.03 * torch.pow(average_building_height, 1.72),
                torch.tensor(10.0, dtype=self.dtype, device=self.device),
            )
            * torch.log10(distance_3d)
            - torch.minimum(
                0.044 * torch.pow(average_building_height, 1.72),
                torch.tensor(14.77, dtype=self.dtype, device=self.device),
            )
            + 0.002 * torch.log10(average_building_height) * distance_3d
        )
        pl_2 = (
            20.0 * torch.log10(40.0 * PI * distance_breakpoint * fc / 3.0)
            + torch.minimum(
                0.03 * torch.pow(average_building_height, 1.72),
                torch.tensor(10.0, dtype=self.dtype, device=self.device),
            )
            * torch.log10(distance_breakpoint)
            - torch.minimum(
                0.044 * torch.pow(average_building_height, 1.72),
                torch.tensor(14.77, dtype=self.dtype, device=self.device),
            )
            + 0.002 * torch.log10(average_building_height) * distance_breakpoint
            + 40.0 * torch.log10(distance_3d / distance_breakpoint)
        )
        pl_los = torch.where(distance_2d < distance_breakpoint, pl_1, pl_2)

        ## Basic pathloss for NLoS and O2I

        pl_3 = (
            161.04
            - 7.1 * torch.log10(self.average_street_width)
            + 7.5 * torch.log10(average_building_height)
            - (24.37 - 3.7 * torch.square(average_building_height / h_bs))
            * torch.log10(h_bs)
            + (43.42 - 3.1 * torch.log10(h_bs)) * (torch.log10(distance_3d) - 3.0)
            + 20.0 * torch.log10(fc)
            - (3.2 * torch.square(torch.log10(11.75 * h_ut)) - 4.97)
        )
        pl_nlos = torch.maximum(pl_los, pl_3)

        ## Set the basic pathloss according to UT state

        # LoS
        pl_b = torch.where(self.outdoor_los, pl_los, pl_nlos)

        self._update_attr("_pl_b", pl_b)
