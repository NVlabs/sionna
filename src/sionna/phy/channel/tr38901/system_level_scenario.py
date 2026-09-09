#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Class used to define a system level 3GPP channel simulation scenario"""

import warnings
from abc import abstractmethod
from typing import Optional, Union

import torch

from sionna._validation import check_tensor_all
from sionna.phy import PI, SPEED_OF_LIGHT
from sionna.phy.object import Object
from sionna.phy.utils import insert_dims, normal, sample_bernoulli, rand
from sionna.phy.channel.utils import rad_2_deg, wrap_angle_0_360

from . import models
from .antenna import HandheldUTArray, PanelArray
from .spatial_consistency import (
    spatial_consistency_correlation_matrix,
    spatial_consistency_matrix_sqrt,
)

__all__ = ["SystemLevelScenario"]

AntennaArrayLike = Union[PanelArray, HandheldUTArray]


class SystemLevelScenario(Object):
    r"""
    Base class for setting up the scenario for system level 3GPP channel
    simulation.

    Scenarios for system level channel simulation, such as UMi, UMa, RMa, or InH,
    are defined by implementing this base class.

    :param carrier_frequency: Carrier frequency [Hz]
    :param o2i_model: Outdoor to indoor (O2I) pathloss model, used for
        indoor UTs. Must be ``"low"`` or ``"high"``.
        See Section 7.4.3 of TR 38.901. For UMi and UMa below 6 GHz, both
        choices use the backward-compatible model from Table 7.4.3-3 and
        therefore produce the same penetration loss.
    :param ut_array: Antenna array configuration used by UTs. This can be a
        :class:`~sionna.phy.channel.tr38901.PanelArray` or
        :class:`~sionna.phy.channel.tr38901.HandheldUTArray`.
    :param bs_array: Antenna array configuration used by base stations. This can be a
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
        o2i_model: str,
        ut_array: AntennaArrayLike,
        bs_array: AntennaArrayLike,
        direction: str,
        enable_pathloss: bool = True,
        enable_shadow_fading: bool = True,
        precision: Optional[str] = None,
        device: Optional[str] = None,
        spec_version: str = "19.2",
    ) -> None:
        super().__init__(precision=precision, device=device)

        self._spec_version = models._validate_spec_version(spec_version)
        self._legacy_o2i_indoor_distance = (
            self.scenario_kind in ("umi", "uma")
            and float(carrier_frequency) < 6e9
        )

        # Carrier frequency (Hz)
        # Register as buffers for CUDAGraph compatibility
        self.register_buffer(
            "_carrier_frequency",
            torch.tensor(carrier_frequency, dtype=self.dtype, device=self.device),
        )

        # Wavelength (m)
        self.register_buffer(
            "_lambda_0",
            torch.tensor(
                SPEED_OF_LIGHT / carrier_frequency, dtype=self.dtype, device=self.device
            ),
        )

        # O2I model
        if o2i_model not in ("low", "high"):
            raise ValueError("o2i_model must be 'low' or 'high'")
        self._o2i_model = o2i_model

        # UT and base-station arrays
        if not isinstance(ut_array, (PanelArray, HandheldUTArray)):
            raise TypeError(
                "'ut_array' must be an instance of PanelArray or "
                "HandheldUTArray"
            )
        if not isinstance(bs_array, (PanelArray, HandheldUTArray)):
            raise TypeError(
                "'bs_array' must be an instance of PanelArray or "
                "HandheldUTArray"
            )
        self._ut_array = ut_array
        self._bs_array = bs_array

        # Direction
        if direction not in ("uplink", "downlink"):
            raise ValueError("'direction' must be 'uplink' or 'downlink'")
        self._direction = direction

        # Pathloss and shadow fading
        self._enable_pathloss = enable_pathloss
        self._enable_shadow_fading = enable_shadow_fading

        # Scenario
        self._ut_loc: Optional[torch.Tensor] = None
        self._bs_loc: Optional[torch.Tensor] = None
        self._bs_virtual_loc: Optional[torch.Tensor] = None
        self._bs_site_ids: Optional[torch.Tensor] = None
        self._bs_site_representatives: Optional[torch.Tensor] = None
        self._ut_orientations: Optional[torch.Tensor] = None
        self._bs_orientations: Optional[torch.Tensor] = None
        self._ut_velocities: Optional[torch.Tensor] = None
        self._in_state: Optional[torch.Tensor] = None
        self._ut_spatial_region_ids: Optional[torch.Tensor] = None
        self._spatial_consistency_track_ids: Optional[torch.Tensor] = None
        self._requested_los: Optional[Union[bool, torch.Tensor]] = None
        self._explicit_bs_site_ids = False
        self._enable_spatial_consistency = False

        # Internal state
        self._distance_2d: Optional[torch.Tensor] = None
        self._distance_3d: Optional[torch.Tensor] = None
        self._raw_distance_2d_in: Optional[torch.Tensor] = None
        self._distance_2d_in: Optional[torch.Tensor] = None
        self._distance_2d_out: Optional[torch.Tensor] = None
        self._distance_3d_in: Optional[torch.Tensor] = None
        self._distance_3d_out: Optional[torch.Tensor] = None
        self._matrix_ut_distance_2d: Optional[torch.Tensor] = None
        self._los_aod: Optional[torch.Tensor] = None
        self._los_aoa: Optional[torch.Tensor] = None
        self._los_zod: Optional[torch.Tensor] = None
        self._los_zoa: Optional[torch.Tensor] = None
        self._outdoor_los: Optional[torch.Tensor] = None
        self._los: Optional[torch.Tensor] = None
        self._lsp_log_mean: Optional[torch.Tensor] = None
        self._lsp_log_std: Optional[torch.Tensor] = None
        self._zod_offset: Optional[torch.Tensor] = None
        self._pl_b: Optional[torch.Tensor] = None
        self._spatial_region_ids_initialized = False
        self._indoor_distance_initialized = False

        # Flag to track if topology has been initialized (frozen after first set_topology)
        self._topology_frozen = False

        # Load parameters for this scenario
        self._load_params()

    @property
    def carrier_frequency(self) -> torch.Tensor:
        """Carrier frequency [Hz]"""
        return self._carrier_frequency

    @property
    def direction(self) -> str:
        """Direction of communication. Either ``"uplink"`` or ``"downlink"``."""
        return self._direction

    @property
    def scenario_kind(self) -> str:
        """Scenario family identifier such as ``"umi"``, ``"uma"``, or ``"rma"``."""
        name = type(self).__name__.lower()
        for kind in ("rma", "umi", "uma", "inh", "inf"):
            if name.startswith(kind):
                return kind
        raise ValueError(f"Unsupported TR 38.901 scenario class {type(self).__name__}")

    @property
    def pathloss_enabled(self) -> bool:
        """`True` if pathloss is enabled. `False` otherwise."""
        return self._enable_pathloss

    @property
    def shadow_fading_enabled(self) -> bool:
        """`True` if shadow fading is enabled. `False` otherwise."""
        return self._enable_shadow_fading

    @property
    def spec_version(self) -> str:
        """TR 38.901 parameter-table version."""
        return self._spec_version

    @property
    def lambda_0(self) -> torch.Tensor:
        """Wavelength [m]"""
        return self._lambda_0

    @property
    def batch_size(self) -> int:
        """Batch size"""
        return self._ut_loc.shape[0]

    @property
    def num_ut(self) -> int:
        """Number of UTs"""
        return self._ut_loc.shape[1]

    @property
    def num_bs(self) -> int:
        """Number of base stations"""
        return self._bs_loc.shape[1]

    @property
    def h_ut(self) -> torch.Tensor:
        """Height of UTs [m]. Shape [batch size, number of UTs]"""
        return self._ut_loc[:, :, 2]

    @property
    def h_bs(self) -> torch.Tensor:
        """Height of base stations [m]. Shape [batch size, number of base stations]"""
        return self._bs_loc[:, :, 2]

    @property
    def ut_loc(self) -> torch.Tensor:
        """Locations of UTs [m]. Shape [batch size, number of UTs, 3]"""
        return self._ut_loc

    @property
    def bs_loc(self) -> torch.Tensor:
        """Locations of base stations [m]. Shape [batch size, number of base stations, 3]"""
        return self._bs_loc

    @property
    def bs_virtual_loc(self) -> torch.Tensor:
        """Virtual location of base stations, relative to each UT position [m].
        Useful in case of wraparound.
        Broadcastable to [batch size, number of UTs, number of base stations, 3]"""
        return self._bs_virtual_loc

    @property
    def bs_site_ids(self) -> torch.Tensor:
        """Site identifier of each BS.
        Shape [batch size, number of base stations]"""
        return self._bs_site_ids

    @property
    def bs_site_representatives(self) -> torch.Tensor:
        """Representative BS index for each BS site.
        Shape [batch size, number of base stations]"""
        return self._bs_site_representatives

    @property
    def ut_orientations(self) -> torch.Tensor:
        """Orientations of UTs [radian]. Shape [batch size, number of UTs, 3]"""
        return self._ut_orientations

    @property
    def bs_orientations(self) -> torch.Tensor:
        """Orientations of base stations [radian]. Shape [batch size, number of base stations, 3]"""
        return self._bs_orientations

    @property
    def ut_velocities(self) -> torch.Tensor:
        """UT velocities [m/s]. Shape [batch size, number of UTs, 3]"""
        return self._ut_velocities

    @property
    def ut_array(self) -> AntennaArrayLike:
        """Antenna array used by UTs."""
        return self._ut_array

    @property
    def bs_array(self) -> AntennaArrayLike:
        """Antenna array used by base stations."""
        return self._bs_array

    @property
    def indoor(self) -> torch.Tensor:
        """Indoor state of UTs. `True` is indoor, `False` otherwise.
        Shape [batch size, number of UTs]"""
        return self._in_state

    @property
    def ut_spatial_region_ids(self) -> torch.Tensor:
        """Correlation-region identifier of each UT.

        Unequal identifiers decorrelate supported spatial random fields. The
        labels do not alter pathloss or geometry; blockage model A uses the
        same partition when enabled. Identifiers should be unique across
        buildings. Shape [batch size, number of UTs].
        """
        return self._ut_spatial_region_ids

    @property
    def spatial_consistency_track_ids(self) -> Optional[torch.Tensor]:
        """Optional grouping identifiers for positions on the same UT track.

        When spatial consistency is enabled, UT entries with equal identifiers
        in one topology snapshot share the cluster-specific angle signs and
        random ray coupling. The identifiers do not preserve random variables
        across topology updates. `None` means that all UT entries are treated
        as distinct simultaneous UTs.
        Shape [batch size, number of UTs].
        """
        return self._spatial_consistency_track_ids

    def set_spatial_consistency_enabled(self, enabled: bool) -> None:
        """Enable or disable additional static spatial random fields."""
        self._enable_spatial_consistency = bool(enabled)

    @property
    def los(self) -> torch.Tensor:
        """LoS state of BS-UT links. `True` if LoS, `False` otherwise.
        Shape [batch size, number of base stations, number of UTs]"""
        return self._los

    @property
    def outdoor_los(self) -> torch.Tensor:
        """LoS condition of the outdoor part of each BS-UT link.

        For outdoor UTs, this is identical to :attr:`los`. For O2I links, it
        retains the sampled outdoor propagation condition while :attr:`los` is
        `False`, because an O2I link has no deterministic LoS component.
        Shape [batch size, number of base stations, number of UTs].
        """
        return self._outdoor_los

    @property
    def distance_2d(self) -> torch.Tensor:
        """Distance between each UT and each BS in the X-Y plane [m].
        Shape [batch size, number of base stations, number of UTs]"""
        return self._distance_2d

    @property
    def distance_2d_in(self) -> torch.Tensor:
        """Indoor distance between each UT and BS in the X-Y plane [m],
        i.e., part of the total distance that corresponds to indoor
        propagation in the X-Y plane.
        Set to 0 for UTs located outdoor.
        Shape [batch size, number of base stations, number of UTs]"""
        return self._distance_2d_in

    @property
    def distance_2d_out(self) -> torch.Tensor:
        """Outdoor distance between each UT and BS in the X-Y plane [m],
        i.e., part of the total distance that corresponds to outdoor
        propagation in the X-Y plane.
        Equals ``distance_2d`` for UTs located outdoor.
        Shape [batch size, number of base stations, number of UTs]"""
        return self._distance_2d_out

    @property
    def distance_3d(self) -> torch.Tensor:
        """Distance between each UT and each BS [m].
        Shape [batch size, number of base stations, number of UTs]"""
        return self._distance_3d

    @property
    def distance_3d_in(self) -> torch.Tensor:
        """Indoor distance between each UT and BS [m],
        i.e., part of the total distance that corresponds to indoor
        propagation. Set to 0 for UTs located outdoor.
        Shape [batch size, number of base stations, number of UTs]"""
        return self._distance_3d_in

    @property
    def distance_3d_out(self) -> torch.Tensor:
        """Outdoor distance between each UT and BS [m],
        i.e., part of the total distance that corresponds to outdoor
        propagation. Equals ``distance_3d`` for UTs located outdoor.
        Shape [batch size, number of base stations, number of UTs]"""
        return self._distance_3d_out

    @property
    def matrix_ut_distance_2d(self) -> torch.Tensor:
        """Distance between all pairs of UTs in the X-Y plane [m].
        Shape [batch size, number of UTs, number of UTs]"""
        return self._matrix_ut_distance_2d

    @property
    def los_aod(self) -> torch.Tensor:
        """LoS azimuth angle of departure of each BS-UT link [deg].
        Shape [batch size, number of base stations, number of UTs]"""
        return self._los_aod

    @property
    def los_aoa(self) -> torch.Tensor:
        """LoS azimuth angle of arrival of each BS-UT link [deg].
        Shape [batch size, number of base stations, number of UTs]"""
        return self._los_aoa

    @property
    def los_zod(self) -> torch.Tensor:
        """LoS zenith angle of departure of each BS-UT link [deg].
        Shape [batch size, number of base stations, number of UTs]"""
        return self._los_zod

    @property
    def los_zoa(self) -> torch.Tensor:
        """LoS zenith angle of arrival of each BS-UT link [deg].
        Shape [batch size, number of base stations, number of UTs]"""
        return self._los_zoa

    @property
    @abstractmethod
    def los_probability(self) -> torch.Tensor:
        """Probability of each UT to be LoS. Used to randomly generate LoS
        status of outdoor UTs. Shape [batch size, number of UTs]"""
        pass

    @property
    @abstractmethod
    def min_2d_in(self) -> torch.Tensor:
        """Minimum indoor 2D distance for indoor UTs [m]"""
        pass

    @property
    @abstractmethod
    def max_2d_in(self) -> torch.Tensor:
        """Maximum indoor 2D distance for indoor UTs [m]"""
        pass

    @property
    def lsp_log_mean(self) -> torch.Tensor:
        """Mean of LSPs in the log domain.
        Shape [batch size, number of base stations, number of UTs, 7].
        The last dimension corresponds to the LSPs, in the following order:
        DS - ASD - ASA - SF - K - ZSA - ZSD"""
        return self._lsp_log_mean

    @property
    def lsp_log_std(self) -> torch.Tensor:
        """STD of LSPs in the log domain.
        Shape [batch size, number of base stations, number of UTs, 7].
        The last dimension corresponds to the LSPs, in the following order:
        DS - ASD - ASA - SF - K - ZSA - ZSD"""
        return self._lsp_log_std

    @property
    @abstractmethod
    def rays_per_cluster(self) -> int:
        """Number of rays per cluster"""
        pass

    @property
    def zod_offset(self) -> torch.Tensor:
        """Zenith angle of departure offset"""
        return self._zod_offset

    @property
    def num_clusters_los(self) -> int:
        """Number of clusters for LoS scenario"""
        return self._params_los["numClusters"]

    @property
    def num_clusters_nlos(self) -> int:
        """Number of clusters for NLoS scenario"""
        return self._params_nlos["numClusters"]

    @property
    def num_clusters_indoor(self) -> int:
        """Number of clusters for indoor scenario"""
        return self._params_o2i["numClusters"]

    @property
    def num_clusters_max(self) -> int:
        """Maximum number of clusters over indoor, LoS, and NLoS scenarios"""
        num_clusters_los = self._params_los["numClusters"]
        num_clusters_nlos = self._params_nlos["numClusters"]
        num_clusters_o2i = self._params_o2i["numClusters"]
        return max(num_clusters_los, num_clusters_nlos, num_clusters_o2i)

    @property
    def basic_pathloss(self) -> torch.Tensor:
        """Basic pathloss component [dB].
        See section 7.4.1 of 38.901 specification.
        Shape [batch size, num BS, num UT]"""
        return self._pl_b

    def _update_attr(self, name: str, value: torch.Tensor) -> None:
        """Update attribute, using in-place copy if tensor already exists.

        On first call for each attribute, registers the tensor as a buffer
        for torch.compile/CUDAGraph compatibility. On subsequent calls,
        updates the buffer in-place using copy_().

        After the first `set_topology` call, shapes are frozen. To change
        shapes, call :meth:`reset_topology` first.

        :param name: Attribute name (e.g., "_ut_loc")
        :param value: New tensor value
        :raises RuntimeError: If shapes don't match after topology is frozen,
            or if trying to register new buffers during torch.compile tracing
        """
        existing = getattr(self, name, None)
        if existing is not None and name in self._buffers:
            # Buffer already registered - update in-place
            if existing.shape != value.shape:
                raise RuntimeError(
                    f"Cannot change shape of '{name}'. "
                    f"Expected {existing.shape}, got {value.shape}. "
                    f"Call reset_topology() before changing batch_size/num_ut/num_bs."
                )
            existing.copy_(value)
        else:
            # First time setting this attribute - need to register as buffer
            # But buffer registration doesn't work inside torch.compile tracing
            if torch.compiler.is_compiling():
                raise RuntimeError(
                    f"Cannot initialize topology buffer '{name}' inside torch.compile. "
                    f"Call the model once (eager) before compiling to initialize buffers, "
                    f"or use allocate_topology_tensors() to pre-allocate."
                )
            self._register_buffer_safe(name, value)

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
        r"""
        Set the network topology.

        It is possible to set up a different network topology for each batch
        example.

        When calling this function, not specifying a parameter leads to the
        reuse of the previously given value. Not specifying a value that was not
        set at a former call raises an error.

        :param ut_loc: Locations of the UTs [m].
            Shape [batch size, number of UTs, 3]
        :param bs_loc: Locations of base stations [m].
            Shape [batch size, number of base stations, 3]
        :param ut_orientations: Orientations of the UTs arrays [radian].
            Shape [batch size, number of UTs, 3]
        :param bs_orientations: Orientations of the base-station arrays [radian].
            Shape [batch size, number of base stations, 3]
        :param ut_velocities: Velocity vectors of UTs [m/s].
            Shape [batch size, number of UTs, 3]
        :param in_state: Indoor/outdoor state of UTs. `True` means indoor and
            `False` means outdoor.
            Shape [batch size, number of UTs]
        :param los: LoS/NLoS state control. If set to `True`, all outdoor UTs
            are forced to be in LoS. If set to `False`, all outdoor UTs are
            forced to be in NLoS. If a boolean tensor is provided, it specifies
            the requested LoS/NLoS state for each BS-UT link with shape
            [batch size, number of base stations, number of UTs] or
            [number of base stations, number of UTs]. If set to ``"random"``, fresh
            stochastic LoS/NLoS states are sampled following Section 7.4.2 of
            :cite:p:`TR38901V1920`. If set to `None`, the previous setting is reused;
            on the first call this is equivalent to ``"random"``.
        :param bs_virtual_loc: Virtual locations of base stations for each UT [m].
            Used to compute BS-UT relative distance and angles.
            If `None` while ``bs_loc`` is specified, then it is set to
            ``bs_loc`` upon reshaping. If neither ``bs_virtual_loc`` nor
            ``bs_loc`` are specified, then the previous value is used.
            Shape [batch size, number of base stations, number of UTs, 3]
        :param bs_site_ids: Site identifier of each BS. Co-sited base stations share the
            same site identifier and use common site-level random quantities,
            such as co-sited LSPs. If `None`, exact duplicate BS locations are
            treated as co-sited; near duplicates remain separate and emit a
            warning. Shape [number of base stations] or [batch size, number of base stations].
        :param spatial_consistency_track_ids: Optional grouping identifiers for
            UT entries representing positions on the same track in the current
            topology snapshot. When spatial consistency is enabled, equal IDs
            share cluster-angle signs and random ray-coupling permutations,
            but do not retain random variables across topology updates. Shape
            [number of UTs] or
            [batch size, number of UTs].
        :param distance_2d_in: Optional pre-sampled indoor 2D distance [m] for
            every UT or link. Below 6 GHz, UMi and UMa use the legacy
            link-specific model and accept shape
            ``[batch size, number of base stations, number of UTs]``. Other O2I models
            use a UT-specific value with shape
            ``[batch size, number of UTs]``. Values for outdoor UTs are
            ignored.
        :param ut_spatial_region_ids: Optional correlation-region identifier
            for every UT. Unequal IDs decorrelate LSP fields and the additional
            fields controlled by spatial consistency; correlation within a
            region uses 2D distance. IDs do not add floor penetration loss or
            change pathloss or geometry; blockage model A uses the same
            partition when enabled. If omitted on the first call, indoor UTs
            in outdoor scenarios are assigned from a 3 m floor grid, while all
            InH/InF UTs use one region. The fallback does not identify separate
            buildings. Shape [number of UTs] or
            [batch size, number of UTs].

        :output updated: `True` if the topology was updated, `False` otherwise
        """

        if ut_loc is None and self._ut_loc is None:
            raise RuntimeError("`ut_loc` is None and was not previously set")

        if bs_loc is None and self._bs_loc is None:
            raise RuntimeError("`bs_loc` is None and was not previously set")

        if (
            bs_virtual_loc is None
            and bs_loc is None
            and self._bs_virtual_loc is None
        ):
            raise RuntimeError(
                "`bs_virtual_loc` is None and was not previously set"
            )

        if in_state is None and self._in_state is None:
            raise RuntimeError("`in_state` is None and was not previously set")

        if ut_orientations is None and self._ut_orientations is None:
            raise RuntimeError(
                "`ut_orientations` is None and was not previously set"
            )

        if bs_orientations is None and self._bs_orientations is None:
            raise RuntimeError(
                "`bs_orientations` is None and was not previously set"
            )

        if ut_velocities is None and self._ut_velocities is None:
            raise RuntimeError(
                "`ut_velocities` is None and was not previously set"
            )

        # Boolean used to keep track of whether or not we need to (re-)compute
        # the distances between users, correlation matrices...
        # This is required if the UT locations, BS locations, indoor/outdoor
        # state of UTs, or LoS/NLoS states of outdoor UTs are updated.
        need_for_update = False

        # Update topology tensors using _update_attr which automatically uses
        # in-place operations when tensors are pre-allocated (for CUDAGraph
        # compatibility) or standard assignment otherwise.
        if ut_loc is not None:
            self._update_attr("_ut_loc", self._convert(ut_loc))
            need_for_update = True

        if bs_loc is not None:
            self._update_attr("_bs_loc", self._convert(bs_loc))
            need_for_update = True

        if bs_virtual_loc is not None:
            self._update_attr("_bs_virtual_loc", self._convert(bs_virtual_loc))
            need_for_update = True
        elif bs_loc is not None:
            # Set virtual BS locations to the effective ones
            # [batch size, number of base stations, number of UTs, 3]
            bs_virtual_loc = insert_dims(self._bs_loc, num_dims=1, axis=2)
            bs_virtual_loc = bs_virtual_loc.expand(-1, -1, self.num_ut, -1).clone()
            self._update_attr(
                "_bs_virtual_loc", bs_virtual_loc
            )

        if bs_site_ids is not None:
            bs_site_ids = torch.as_tensor(
                bs_site_ids, dtype=torch.int64, device=self.device
            )
            if bs_site_ids.dim() == 1:
                bs_site_ids = (
                    bs_site_ids.unsqueeze(0).expand(self._bs_loc.shape[0], -1).clone()
                )
            if bs_site_ids.shape != self._bs_loc.shape[:2]:
                raise ValueError(
                    "`bs_site_ids` must have shape [number of base stations] "
                    "or [batch size, number of base stations]"
                )
            self._update_attr("_bs_site_ids", bs_site_ids)
            self._explicit_bs_site_ids = True
            need_for_update = True

        if bs_orientations is not None:
            self._update_attr("_bs_orientations", self._convert(bs_orientations))

        if ut_orientations is not None:
            self._update_attr("_ut_orientations", self._convert(ut_orientations))

        if ut_velocities is not None:
            self._update_attr("_ut_velocities", self._convert(ut_velocities))

        if in_state is not None:
            in_state_tensor = torch.as_tensor(in_state, device=self.device)
            if in_state_tensor.dtype != torch.bool:
                raise TypeError("`in_state` must have dtype torch.bool")
            if in_state_tensor.shape != self._ut_loc.shape[:2]:
                raise ValueError(
                    "`in_state` must have shape "
                    "[batch size, number of UTs]"
                )
            if self.scenario_kind in ("inh", "inf"):
                check_tensor_all(
                    in_state_tensor,
                    name="in_state",
                    message="InH and InF require every UT to be indoor",
                )
            self._update_attr("_in_state", in_state_tensor)
            need_for_update = True

        if ut_spatial_region_ids is not None:
            region_ids = torch.as_tensor(
                ut_spatial_region_ids, dtype=torch.int64, device=self.device
            )
            if region_ids.dim() == 1:
                region_ids = region_ids.unsqueeze(0).expand(
                    self._ut_loc.shape[0], -1
                ).clone()
            if region_ids.shape != self._ut_loc.shape[:2]:
                raise ValueError(
                    "`ut_spatial_region_ids` must have shape [number of UTs] "
                    "or [batch size, number of UTs]"
                )
            self._update_attr("_ut_spatial_region_ids", region_ids)
            self._spatial_region_ids_initialized = True
            need_for_update = True
        elif not self._spatial_region_ids_initialized:
            if self.scenario_kind in ("inh", "inf"):
                region_ids = torch.zeros_like(self._in_state, dtype=torch.int64)
            else:
                floor_ids = torch.round((self.h_ut - 1.5) / 3.0).to(torch.int64)
                region_ids = torch.where(
                    self._in_state,
                    floor_ids,
                    torch.zeros_like(floor_ids),
                )
            self._update_attr("_ut_spatial_region_ids", region_ids)
            self._spatial_region_ids_initialized = True

        if spatial_consistency_track_ids is not None:
            track_ids = torch.as_tensor(
                spatial_consistency_track_ids,
                dtype=torch.int64,
                device=self.device,
            )
            if track_ids.dim() == 1:
                track_ids = (
                    track_ids.unsqueeze(0).expand(self._ut_loc.shape[0], -1).clone()
                )
            if track_ids.shape != self._ut_loc.shape[:2]:
                raise ValueError(
                    "`spatial_consistency_track_ids` must have shape "
                    "[number of UTs] or [batch size, number of UTs]"
                )
            self._update_attr("_spatial_consistency_track_ids", track_ids)
            need_for_update = True

        if distance_2d_in is not None:
            if self.scenario_kind in ("inh", "inf"):
                raise ValueError(
                    "`distance_2d_in` is not applicable to native indoor "
                    "InH and InF scenarios"
                )
            need_for_update = True

        if isinstance(los, str) and los == "random":
            self._requested_los = None
            need_for_update = True
        elif los is not None:
            if isinstance(los, torch.Tensor):
                los_tensor = los.to(dtype=torch.bool, device=self.device)
                if los_tensor.dim() == 2:
                    los_tensor = los_tensor.unsqueeze(0).expand(
                        self.batch_size, -1, -1
                    ).clone()
                if los_tensor.shape != (self.batch_size, self.num_bs, self.num_ut):
                    raise ValueError(
                        "`los` tensor must have shape "
                        "[batch size, number of base stations, number of UTs] or "
                        "[number of base stations, number of UTs]"
                    )
                self._requested_los = los_tensor
            elif isinstance(los, bool):
                self._requested_los = los
            else:
                raise ValueError(
                    "`los` must be True, False, None, 'random', or a bool tensor"
                )
            need_for_update = True

        if need_for_update:
            # Update topology-related quantities
            self._compute_distance_2d_3d_and_angles()
            self._compute_bs_site_representatives()
            self._sample_indoor_distance(distance_2d_in)
            self._sample_los()

            # Compute the LSPs means and stds
            self._compute_lsp_log_mean_std()

            # Compute the basic path-loss
            self._compute_pathloss_basic()

            # Freeze topology after first complete set_topology call
            # This enables optimal performance with torch.compile
            # Call reset_topology() to allow shape changes
            if not self._topology_frozen:
                self._topology_frozen = True

        return need_for_update

    def _register_buffer_safe(self, name: str, tensor: torch.Tensor) -> None:
        """Register a buffer, replacing any existing attribute with the same name.

        This is needed because attributes are initialized to None in __init__,
        and register_buffer will fail if the attribute already exists as a
        non-buffer.

        :param name: Buffer name
        :param tensor: Tensor to register
        """
        if hasattr(self, name) and name not in self._buffers:
            delattr(self, name)
        self.register_buffer(name, tensor)

    def reset_topology(self) -> None:
        """Reset the topology to allow different batch_size/num_ut/num_bs.

        This method clears all topology buffers and returns the scenario
        to its initial state. The next `set_topology` call will re-initialize
        the buffers with the new shapes and freeze again.

        Use this when you need to change the batch_size, num_ut, or num_bs,
        for example when switching between training and evaluation with
        different batch sizes.

        Note: After reset, the next `set_topology` call will re-freeze.
        If using torch.compile, this will trigger recompilation.
        """
        # List of all topology buffer names
        topology_buffers = [
            "_ut_loc",
            "_bs_loc",
            "_bs_virtual_loc",
            "_bs_site_ids",
            "_bs_site_representatives",
            "_ut_orientations",
            "_bs_orientations",
            "_ut_velocities",
            "_in_state",
            "_ut_spatial_region_ids",
            "_spatial_consistency_track_ids",
            "_distance_2d",
            "_distance_3d",
            "_raw_distance_2d_in",
            "_distance_2d_in",
            "_distance_2d_out",
            "_distance_3d_in",
            "_distance_3d_out",
            "_matrix_ut_distance_2d",
            "_los_aod",
            "_los_aoa",
            "_los_zod",
            "_los_zoa",
            "_outdoor_los",
            "_los",
            "_lsp_log_mean",
            "_lsp_log_std",
            "_zod_offset",
            "_pl_b",
        ]

        # Remove all topology buffers - delattr removes from _buffers dict
        for name in topology_buffers:
            if hasattr(self, name):
                delattr(self, name)

        # Unfreeze to allow new shapes
        self._topology_frozen = False
        self._explicit_bs_site_ids = False
        self._spatial_region_ids_initialized = False
        self._indoor_distance_initialized = False
        self._spatial_consistency_track_ids = None
        self._requested_los = None

    def allocate_topology_tensors(
        self,
        batch_size: int,
        num_bs: int,
        num_ut: int,
    ) -> None:
        r"""
        Pre-allocate all tensors used for topology updates.

        This method must be called before using `set_topology` for the first
        time inside a `torch.compile`-decorated function. Outside a compiled
        function, it is optional because the first eager call registers the
        buffers automatically.

        Calling this method again reinitializes all topology buffers to the
        new shapes (equivalent to :meth:`reset_topology` + allocate).

        After the first `set_topology` call (or after calling this method),
        the shapes of topology tensors are frozen and cannot be changed.
        This is required for `torch.compile` with `mode="reduce-overhead"`.

        :param batch_size: Batch size
        :param num_bs: Number of base stations
        :param num_ut: Number of user terminals
        """
        # Reset any existing buffers to allow reinitialization
        self.reset_topology()

        # Pre-allocate scenario tensors - register as buffers for CUDAGraph compatibility
        self._register_buffer_safe(
            "_ut_loc",
            torch.zeros(batch_size, num_ut, 3, dtype=self.dtype, device=self.device),
        )
        self._register_buffer_safe(
            "_bs_loc",
            torch.zeros(batch_size, num_bs, 3, dtype=self.dtype, device=self.device),
        )
        self._register_buffer_safe(
            "_bs_virtual_loc",
            torch.zeros(
                batch_size, num_bs, num_ut, 3, dtype=self.dtype, device=self.device
            ),
        )
        bs_indices = torch.arange(num_bs, dtype=torch.int64, device=self.device)
        bs_indices = bs_indices.reshape(1, num_bs).expand(batch_size, -1).clone()
        self._register_buffer_safe("_bs_site_ids", bs_indices.clone())
        self._register_buffer_safe("_bs_site_representatives", bs_indices.clone())
        self._register_buffer_safe(
            "_ut_orientations",
            torch.zeros(batch_size, num_ut, 3, dtype=self.dtype, device=self.device),
        )
        self._register_buffer_safe(
            "_bs_orientations",
            torch.zeros(batch_size, num_bs, 3, dtype=self.dtype, device=self.device),
        )
        self._register_buffer_safe(
            "_ut_velocities",
            torch.zeros(batch_size, num_ut, 3, dtype=self.dtype, device=self.device),
        )
        self._register_buffer_safe(
            "_in_state",
            torch.zeros(batch_size, num_ut, dtype=torch.bool, device=self.device),
        )
        self._register_buffer_safe(
            "_ut_spatial_region_ids",
            torch.zeros(batch_size, num_ut, dtype=torch.int64, device=self.device),
        )
        track_ids = torch.arange(num_ut, dtype=torch.int64, device=self.device)
        track_ids = track_ids.reshape(1, num_ut).expand(batch_size, -1).clone()
        self._register_buffer_safe("_spatial_consistency_track_ids", track_ids)

        # Pre-allocate internal state tensors
        self._register_buffer_safe(
            "_distance_2d",
            torch.zeros(
                batch_size, num_bs, num_ut, dtype=self.dtype, device=self.device
            ),
        )
        self._register_buffer_safe(
            "_distance_3d",
            torch.zeros(
                batch_size, num_bs, num_ut, dtype=self.dtype, device=self.device
            ),
        )
        self._register_buffer_safe(
            "_raw_distance_2d_in",
            torch.zeros(
                batch_size, num_bs, num_ut, dtype=self.dtype, device=self.device
            ),
        )
        self._register_buffer_safe(
            "_distance_2d_in",
            torch.zeros(
                batch_size, num_bs, num_ut, dtype=self.dtype, device=self.device
            ),
        )
        self._register_buffer_safe(
            "_distance_2d_out",
            torch.zeros(
                batch_size, num_bs, num_ut, dtype=self.dtype, device=self.device
            ),
        )
        self._register_buffer_safe(
            "_distance_3d_in",
            torch.zeros(
                batch_size, num_bs, num_ut, dtype=self.dtype, device=self.device
            ),
        )
        self._register_buffer_safe(
            "_distance_3d_out",
            torch.zeros(
                batch_size, num_bs, num_ut, dtype=self.dtype, device=self.device
            ),
        )
        self._register_buffer_safe(
            "_matrix_ut_distance_2d",
            torch.zeros(
                batch_size, num_ut, num_ut, dtype=self.dtype, device=self.device
            ),
        )

        # Angle tensors
        self._register_buffer_safe(
            "_los_aod",
            torch.zeros(
                batch_size, num_bs, num_ut, dtype=self.dtype, device=self.device
            ),
        )
        self._register_buffer_safe(
            "_los_aoa",
            torch.zeros(
                batch_size, num_bs, num_ut, dtype=self.dtype, device=self.device
            ),
        )
        self._register_buffer_safe(
            "_los_zod",
            torch.zeros(
                batch_size, num_bs, num_ut, dtype=self.dtype, device=self.device
            ),
        )
        self._register_buffer_safe(
            "_los_zoa",
            torch.zeros(
                batch_size, num_bs, num_ut, dtype=self.dtype, device=self.device
            ),
        )

        # LoS state tensors
        self._register_buffer_safe(
            "_outdoor_los",
            torch.zeros(
                batch_size, num_bs, num_ut, dtype=torch.bool, device=self.device
            ),
        )
        self._register_buffer_safe(
            "_los",
            torch.zeros(
                batch_size, num_bs, num_ut, dtype=torch.bool, device=self.device
            ),
        )

        # LSP tensors (7 LSPs: DS, ASD, ASA, SF, K, ZSA, ZSD)
        self._register_buffer_safe(
            "_lsp_log_mean",
            torch.zeros(
                batch_size, num_bs, num_ut, 7, dtype=self.dtype, device=self.device
            ),
        )
        self._register_buffer_safe(
            "_lsp_log_std",
            torch.zeros(
                batch_size, num_bs, num_ut, 7, dtype=self.dtype, device=self.device
            ),
        )
        self._register_buffer_safe(
            "_zod_offset",
            torch.zeros(
                batch_size, num_bs, num_ut, dtype=self.dtype, device=self.device
            ),
        )

        # Pathloss tensor
        self._register_buffer_safe(
            "_pl_b",
            torch.zeros(
                batch_size, num_bs, num_ut, dtype=self.dtype, device=self.device
            ),
        )

        # Freeze topology shapes for torch.compile compatibility
        self._topology_frozen = True

    def spatial_correlation_matrix(self, correlation_distance: float) -> torch.Tensor:
        r"""Computes and returns a 2D spatial exponential correlation matrix
        :math:`C` over the UTs, such that :math:`C` has shape
        (number of UTs) x (number of UTs), and

        .. math::
            C_{n,m} = \exp\left(-\frac{d_{n,m}}{D}\right)

        where :math:`d_{n,m}` is the distance between UT :math:`n` and UT
        :math:`m` in the X-Y plane, and :math:`D` the correlation distance.

        :param correlation_distance: Correlation distance, i.e., distance
            such that the correlation is :math:`e^{-1} \approx 0.37`

        :output C: Spatial correlation :math:`C`,
            shape [batch size, number of UTs, number of UTs]
        """
        return torch.exp(-self.matrix_ut_distance_2d / correlation_distance)

    @property
    @abstractmethod
    def los_parameter_filepath(self) -> str:
        """Path of the configuration file for LoS scenario"""
        pass

    @property
    @abstractmethod
    def nlos_parameter_filepath(self) -> str:
        """Path of the configuration file for NLoS scenario"""
        pass

    @property
    @abstractmethod
    def o2i_parameter_filepath(self) -> str:
        """Path of the configuration file for indoor scenario"""
        pass

    @property
    def o2i_model(self) -> str:
        """O2I model used for pathloss computation of indoor UTs.
        Either ``"low"`` or ``"high"``. See section 7.4.3 of TR 38.901."""
        return self._o2i_model

    @property
    def use_indoor_lsp_params(self) -> bool:
        """Use the scenario's dedicated indoor/O2I LSP and ray parameters."""
        return True

    @property
    def indoor_links_can_be_los(self) -> bool:
        """Allow indoor links to keep their sampled outdoor LoS condition."""
        return False

    @property
    def indoor_links_use_o2i_zenith_model(self) -> bool:
        """Use O2I zenith-angle generation rules for indoor UTs."""
        return True

    @property
    def o2i_pathloss_enabled(self) -> bool:
        """Apply outdoor-to-indoor penetration loss to indoor UTs."""
        return True

    @abstractmethod
    def clip_carrier_frequency_lsp(self, fc: torch.Tensor) -> torch.Tensor:
        r"""Clip the carrier frequency ``fc`` in GHz for LSP calculation.

        :param fc: Carrier frequency [GHz]

        :output fc_clipped: Clipped carrier frequency, that should be used for LSP
            computation
        """
        pass

    _LOG_LINEAR_PARAMS = {
        "muDS",
        "sigmaDS",
        "muASD",
        "sigmaASD",
        "muASA",
        "sigmaASA",
        "muZSA",
        "sigmaZSA",
    }

    def _get_log_linear_param(
        self, parameter_name: str, fc: torch.Tensor
    ) -> torch.Tensor:
        pa_los = self._params_los[parameter_name + "a"]
        pb_los = self._params_los[parameter_name + "b"]
        pc_los = self._params_los[parameter_name + "c"]

        pa_nlos = self._params_nlos[parameter_name + "a"]
        pb_nlos = self._params_nlos[parameter_name + "b"]
        pc_nlos = self._params_nlos[parameter_name + "c"]

        pa_o2i = self._params_o2i[parameter_name + "a"]
        pb_o2i = self._params_o2i[parameter_name + "b"]
        pc_o2i = self._params_o2i[parameter_name + "c"]

        parameter_value_los = pa_los * torch.log10(pb_los + fc) + pc_los
        parameter_value_nlos = pa_nlos * torch.log10(pb_nlos + fc) + pc_nlos
        parameter_value_o2i = pa_o2i * torch.log10(pb_o2i + fc) + pc_o2i

        return self.broadcast_params(
            parameter_value_los, parameter_value_nlos, parameter_value_o2i
        )

    def _get_cds_param(self, parameter_name: str, fc: torch.Tensor) -> torch.Tensor:
        pa_los = self._params_los[parameter_name + "a"]
        pb_los = self._params_los[parameter_name + "b"]
        pc_los = self._params_los[parameter_name + "c"]

        pa_nlos = self._params_nlos[parameter_name + "a"]
        pb_nlos = self._params_nlos[parameter_name + "b"]
        pc_nlos = self._params_nlos[parameter_name + "c"]

        pa_o2i = self._params_o2i[parameter_name + "a"]
        pb_o2i = self._params_o2i[parameter_name + "b"]
        pc_o2i = self._params_o2i[parameter_name + "c"]

        parameter_value_los = torch.maximum(pa_los, pb_los - pc_los * torch.log10(fc))
        parameter_value_nlos = torch.maximum(
            pa_nlos, pb_nlos - pc_nlos * torch.log10(fc)
        )
        parameter_value_o2i = torch.maximum(pa_o2i, pb_o2i - pc_o2i * torch.log10(fc))

        return self.broadcast_params(
            parameter_value_los, parameter_value_nlos, parameter_value_o2i
        )

    def _get_generic_param(self, parameter_name: str) -> torch.Tensor:
        parameter_value_los = self._params_los[parameter_name]
        parameter_value_nlos = self._params_nlos[parameter_name]
        parameter_value_o2i = self._params_o2i[parameter_name]

        return self.broadcast_params(
            parameter_value_los, parameter_value_nlos, parameter_value_o2i
        )

    def get_param(self, parameter_name: str) -> torch.Tensor:
        r"""Given a ``parameter_name`` used in the configuration file, returns
        a tensor with shape [batch size, number of base stations, number of UTs] of the
        parameter value according to each BS-UT link state (LoS, NLoS,
        indoor).

        :param parameter_name: Name of the parameter used in the
            configuration file

        :output value: Parameter value for each BS-UT link,
            shape [batch size, number of base stations, number of UTs]
        """
        if parameter_name in self._LOG_LINEAR_PARAMS:
            fc = self._carrier_frequency / 1e9
            fc = self.clip_carrier_frequency_lsp(fc)
            return self._get_log_linear_param(parameter_name, fc)
        elif parameter_name == "cDS":
            fc = self._carrier_frequency / 1e9
            fc = self.clip_carrier_frequency_lsp(fc)
            return self._get_cds_param(parameter_name, fc)
        else:
            return self._get_generic_param(parameter_name)

    def broadcast_params(
        self, parameter_value_los, parameter_value_nlos, parameter_value_o2i
    ) -> torch.Tensor:
        r"""Broadcast parameters to the shape
        [batch size, number of base stations, number of UTs] based on the link state
        (LoS, NLoS, indoor).
        """
        parameter_tensor = torch.zeros(
            self.batch_size,
            self.num_bs,
            self.num_ut,
            dtype=self.dtype,
            device=self.device,
        )

        # Expand to allow broadcasting with the BS dimension
        indoor = self.indoor.unsqueeze(1)

        # LoS
        if isinstance(parameter_value_los, torch.Tensor):
            parameter_value_los = parameter_value_los.to(
                dtype=self.dtype, device=self.device
            )
        else:
            parameter_value_los = torch.tensor(
                parameter_value_los, dtype=self.dtype, device=self.device
            )
        parameter_tensor = torch.where(self.los, parameter_value_los, parameter_tensor)

        # NLoS
        if isinstance(parameter_value_nlos, torch.Tensor):
            parameter_value_nlos = parameter_value_nlos.to(
                dtype=self.dtype, device=self.device
            )
        else:
            parameter_value_nlos = torch.tensor(
                parameter_value_nlos, dtype=self.dtype, device=self.device
            )
        if self.use_indoor_lsp_params:
            nlos_mask = (~self.los) & (~indoor)
        else:
            nlos_mask = ~self.los
        parameter_tensor = torch.where(nlos_mask, parameter_value_nlos, parameter_tensor)

        # O2I
        if isinstance(parameter_value_o2i, torch.Tensor):
            parameter_value_o2i = parameter_value_o2i.to(
                dtype=self.dtype, device=self.device
            )
        else:
            parameter_value_o2i = torch.tensor(
                parameter_value_o2i, dtype=self.dtype, device=self.device
            )
        if self.use_indoor_lsp_params:
            parameter_tensor = torch.where(indoor, parameter_value_o2i, parameter_tensor)

        return parameter_tensor

    #####################################################
    # Internal utility methods
    #####################################################

    def _compute_distance_2d_3d_and_angles(self) -> None:
        r"""
        Computes the following internal values:
        * 2D distances for all BS-UT pairs in the X-Y plane
        * 3D distances for all BS-UT pairs
        * 2D distances for all pairs of UTs in the X-Y plane
        * LoS AoA, AoD, ZoA, ZoD for all BS-UT pairs

        This function is called at every update of the topology.
        """
        ut_loc = self._ut_loc
        # [batch_size, 1, num_ut, 3]
        ut_loc_exp = ut_loc.unsqueeze(1)
        # [batch_size, num_bs, num_ut, 3]
        bs_virtual_loc = self._bs_virtual_loc

        delta_loc_xy = ut_loc_exp[:, :, :, :2] - bs_virtual_loc[:, :, :, :2]
        delta_loc = ut_loc_exp - bs_virtual_loc

        # 2D distances for all BS-UT pairs in the (x-y) plane
        distance_2d = torch.sqrt((delta_loc_xy**2).sum(dim=3))
        self._update_attr("_distance_2d", distance_2d)

        # 3D distances for all BS-UT pairs
        distance_3d = torch.sqrt((delta_loc**2).sum(dim=3))
        self._update_attr("_distance_3d", distance_3d)

        # LoS AoA, AoD, ZoA, ZoD
        los_aod = torch.atan2(delta_loc[:, :, :, 1], delta_loc[:, :, :, 0])
        los_aoa = los_aod + PI
        los_zod = torch.atan2(distance_2d, delta_loc[:, :, :, 2])
        los_zoa = PI - los_zod

        # Angles are converted to degrees and wrapped to (0,360)
        self._update_attr("_los_aod", wrap_angle_0_360(rad_2_deg(los_aod)))
        self._update_attr("_los_aoa", wrap_angle_0_360(rad_2_deg(los_aoa)))
        self._update_attr("_los_zod", wrap_angle_0_360(rad_2_deg(los_zod)))
        self._update_attr("_los_zoa", wrap_angle_0_360(rad_2_deg(los_zoa)))

        # 2D distances for all pairs of UTs in the (x-y) plane
        ut_loc_xy = self._ut_loc[:, :, :2]
        ut_loc_xy_expanded_1 = ut_loc_xy.unsqueeze(1)
        ut_loc_xy_expanded_2 = ut_loc_xy.unsqueeze(2)
        delta_loc_xy_ut = ut_loc_xy_expanded_1 - ut_loc_xy_expanded_2
        matrix_ut_distance_2d = torch.sqrt((delta_loc_xy_ut**2).sum(dim=3))
        self._update_attr("_matrix_ut_distance_2d", matrix_ut_distance_2d)

    def _compute_bs_site_representatives(self) -> None:
        """Compute representative BS indices for co-sited sectors."""
        batch_size = self.batch_size
        num_bs = self.num_bs
        bs_indices = torch.arange(num_bs, dtype=torch.int64, device=self.device)

        if self._explicit_bs_site_ids and self._bs_site_ids is not None:
            site_ids = self._bs_site_ids
            same_site = site_ids.unsqueeze(2) == site_ids.unsqueeze(1)
        else:
            bs_loc = self._bs_loc
            same_site = (bs_loc.unsqueeze(2) == bs_loc.unsqueeze(1)).all(dim=-1)
            self._warn_if_near_duplicate_bs_locations(same_site)

        candidates = torch.where(
            same_site,
            bs_indices.reshape(1, 1, num_bs),
            torch.full(
                (batch_size, num_bs, num_bs),
                num_bs,
                dtype=torch.int64,
                device=self.device,
            ),
        )
        representatives = candidates.min(dim=2).values

        if not self._explicit_bs_site_ids:
            self._update_attr("_bs_site_ids", representatives)
        self._update_attr("_bs_site_representatives", representatives)

    def _warn_if_near_duplicate_bs_locations(
        self, same_site: torch.Tensor
    ) -> None:
        """Warn when implicit site inference narrowly misses a duplicate."""
        if torch.compiler.is_compiling() or self.num_bs < 2:
            return

        eps = torch.finfo(self._bs_loc.dtype).eps
        near_same_site = torch.isclose(
            self._bs_loc.unsqueeze(2),
            self._bs_loc.unsqueeze(1),
            rtol=8.0 * eps,
            atol=8.0 * eps,
        ).all(dim=-1)
        if torch.any(near_same_site & ~same_site).item():
            warnings.warn(
                "Some BS locations differ only within floating-point tolerance "
                "and are treated as separate sites. Pass `bs_site_ids` "
                "explicitly if these base stations are co-sited.",
                category=UserWarning,
                stacklevel=4,
            )

    def share_by_bs_site(self, value: torch.Tensor) -> torch.Tensor:
        """Share BS-link values among co-sited base stations using representative links."""
        representatives = self._bs_site_representatives
        if representatives is None:
            return value

        index = representatives
        while index.dim() < value.dim():
            index = index.unsqueeze(-1)
        index = index.expand(value.shape)
        return torch.gather(value, dim=1, index=index)

    def _sample_los(self) -> None:
        r"""Set the LoS state of each UT randomly, following the procedure
        described in section 7.4.2 of TR 38.901.
        LoS state of each UT is randomly assigned according to a Bernoulli
        distribution, which probability depends on the channel model.
        """
        if self._requested_los is None:
            los_probability = self.los_probability
            if self._enable_spatial_consistency:
                los = self._sample_spatially_consistent_los(los_probability)
            else:
                los = sample_bernoulli(
                    [self.batch_size, self.num_bs, self.num_ut],
                    los_probability,
                    precision=self.precision,
                    device=self.device,
                )
        elif isinstance(self._requested_los, torch.Tensor):
            los = self._requested_los
        else:
            los = torch.full(
                (self.batch_size, self.num_bs, self.num_ut),
                self._requested_los,
                dtype=torch.bool,
                device=self.device,
            )

        los = self.share_by_bs_site(los)
        self._update_attr("_outdoor_los", los)
        if self.indoor_links_can_be_los:
            self._update_attr("_los", los)
        else:
            self._update_attr("_los", los & (~self._in_state.unsqueeze(1)))

    def _los_state_correlation_distance(self) -> torch.Tensor:
        """Return the scenario-specific LoS-state correlation distance [m]."""
        kind = self.scenario_kind
        if kind in ("umi", "uma"):
            value = 50.0
        elif kind == "rma":
            value = 60.0
        elif kind == "inh":
            value = 10.0
        elif kind == "inf":
            value = 0.5 * self.clutter_size
            return value.to(dtype=self.dtype, device=self.device)
        else:
            raise ValueError(
                "LoS-state spatial consistency is not configured for scenario "
                f"{type(self).__name__}"
            )
        return torch.tensor(value, dtype=self.dtype, device=self.device)

    def _sample_spatially_consistent_los(
        self, los_probability: torch.Tensor
    ) -> torch.Tensor:
        """Sample LoS states from a spatially correlated Gaussian copula."""
        correlation = spatial_consistency_correlation_matrix(
            self.matrix_ut_distance_2d.unsqueeze(1),
            self._los_state_correlation_distance(),
            states=self.ut_spatial_region_ids.unsqueeze(1),
            precision=self.precision,
            device=self.device,
        )
        matrix_sqrt = spatial_consistency_matrix_sqrt(
            correlation,
            precision=self.precision,
            device=self.device,
        )
        white = normal(
            (self.batch_size, self.num_bs, self.num_ut, 1),
            dtype=self.dtype,
            device=self.device,
            generator=self.torch_rng,
        )
        correlated = torch.matmul(matrix_sqrt, white).squeeze(-1)
        sqrt_two = torch.sqrt(
            torch.tensor(2.0, dtype=self.dtype, device=self.device)
        )
        uniform = 0.5 * torch.erfc(-correlated / sqrt_two)
        return uniform < los_probability

    def _sample_indoor_distance(
        self, distance_2d_in: Optional[torch.Tensor] = None
    ) -> None:
        r"""Set 2D indoor distances according to Section 7.4.3.1 of TR 38.901.

        If ``distance_2d_in`` is not provided, distances are sampled by the
        scenario. Otherwise, the provided UT-specific distances are reused.
        """
        if self.scenario_kind in ("inh", "inf"):
            self._update_attr("_raw_distance_2d_in", self.distance_2d)
            self._update_attr("_distance_2d_in", self.distance_2d)
            self._update_attr("_distance_2d_out", torch.zeros_like(self.distance_2d))
            self._update_attr("_distance_3d_in", self.distance_3d)
            self._update_attr("_distance_3d_out", torch.zeros_like(self.distance_3d))
            self._indoor_distance_initialized = True
            return

        legacy_link_specific = self._legacy_o2i_indoor_distance
        if distance_2d_in is not None:
            raw_distance = torch.as_tensor(
                distance_2d_in, dtype=self.dtype, device=self.device
            )
            ut_shape = (self.batch_size, self.num_ut)
            link_shape = (self.batch_size, self.num_bs, self.num_ut)
            if legacy_link_specific:
                if raw_distance.shape == ut_shape:
                    raw_distance = raw_distance.unsqueeze(1).expand(
                        -1, self.num_bs, -1
                    ).clone()
                elif raw_distance.shape != link_shape:
                    raise ValueError(
                        "Legacy UMi/UMa `distance_2d_in` must have shape "
                        "[batch size, number of UTs] or [batch size, number "
                        "of base stations, number of UTs]"
                    )
                raw_distance = self.share_by_bs_site(raw_distance)
            else:
                if raw_distance.shape != ut_shape:
                    raise ValueError(
                        "`distance_2d_in` must have shape "
                        "[batch size, number of UTs] for the selected model"
                    )
                raw_distance = raw_distance.unsqueeze(1).expand(
                    -1, self.num_bs, -1
                ).clone()
            self._update_attr("_raw_distance_2d_in", raw_distance)
            self._indoor_distance_initialized = True
        elif not self._indoor_distance_initialized:
            if legacy_link_specific:
                normalized_distance = rand(
                    (self.batch_size, self.num_bs, self.num_ut),
                    dtype=self.dtype,
                    device=self.device,
                    generator=self.torch_rng,
                )
                normalized_distance = self.share_by_bs_site(normalized_distance)
            else:
                if self._enable_spatial_consistency:
                    target_correlation = spatial_consistency_correlation_matrix(
                        self.matrix_ut_distance_2d,
                        25.0,
                        states=self.ut_spatial_region_ids,
                        precision=self.precision,
                        device=self.device,
                    )
                    latent_correlation = 2.0 * torch.sin(
                        PI * target_correlation / 6.0
                    )
                    matrix_sqrt = spatial_consistency_matrix_sqrt(
                        latent_correlation,
                        precision=self.precision,
                        device=self.device,
                    )
                    white = normal(
                        (self.batch_size, self.num_ut, 2),
                        dtype=self.dtype,
                        device=self.device,
                        generator=self.torch_rng,
                    )
                    gaussian = torch.matmul(matrix_sqrt, white)
                    sqrt_two = torch.sqrt(
                        torch.tensor(2.0, dtype=self.dtype, device=self.device)
                    )
                    uniform = 0.5 * torch.erfc(-gaussian / sqrt_two)
                    normalized_distance = uniform.min(dim=-1).values
                else:
                    uniform = rand(
                        (self.batch_size, self.num_ut, 2),
                        dtype=self.dtype,
                        device=self.device,
                        generator=self.torch_rng,
                    )
                    normalized_distance = uniform.min(dim=-1).values
                normalized_distance = normalized_distance.unsqueeze(1).expand(
                    -1, self.num_bs, -1
                ).clone()
            raw_distance = (
                normalized_distance * (self.max_2d_in - self.min_2d_in)
                + self.min_2d_in
            )
            self._update_attr("_raw_distance_2d_in", raw_distance)
            self._indoor_distance_initialized = True

        indoor_mask = self.indoor.unsqueeze(1).to(self.dtype)
        effective_distance = self._raw_distance_2d_in * indoor_mask
        self._update_attr("_distance_2d_in", effective_distance)
        self._update_attr(
            "_distance_2d_out", self.distance_2d - effective_distance
        )
        ratio = torch.where(
            self.distance_2d > 0.0,
            effective_distance / self.distance_2d.clamp_min(
                torch.finfo(self.dtype).tiny
            ),
            torch.zeros_like(effective_distance),
        )
        self._update_attr("_distance_3d_in", ratio * self.distance_3d)
        self._update_attr(
            "_distance_3d_out", self.distance_3d - self._distance_3d_in
        )

    def _load_params(self) -> None:
        r"""Load the configuration files corresponding to the 3 possible states
        of UTs: LoS, NLoS, and O2I"""

        source = self._parameter_file(self.o2i_parameter_filepath)
        self._params_o2i = models.load_json(source)

        for param_name in self._params_o2i:
            v = self._params_o2i[param_name]
            if isinstance(v, float):
                # Register as buffer for CUDAGraph compatibility
                tensor = torch.tensor(v, dtype=self.dtype, device=self.device)
                self.register_buffer(f"_params_o2i_{param_name}", tensor)
                self._params_o2i[param_name] = tensor
            elif isinstance(v, int):
                # Keep integers as Python int for num_clusters, etc.
                pass

        source = self._parameter_file(self.los_parameter_filepath)
        self._params_los = models.load_json(source)

        for param_name in self._params_los:
            v = self._params_los[param_name]
            if isinstance(v, float):
                # Register as buffer for CUDAGraph compatibility
                tensor = torch.tensor(v, dtype=self.dtype, device=self.device)
                self.register_buffer(f"_params_los_{param_name}", tensor)
                self._params_los[param_name] = tensor
            elif isinstance(v, int):
                # Keep integers as Python int for num_clusters, etc.
                pass

        source = self._parameter_file(self.nlos_parameter_filepath)
        self._params_nlos = models.load_json(source)

        for param_name in self._params_nlos:
            v = self._params_nlos[param_name]
            if isinstance(v, float):
                # Register as buffer for CUDAGraph compatibility
                tensor = torch.tensor(v, dtype=self.dtype, device=self.device)
                self.register_buffer(f"_params_nlos_{param_name}", tensor)
                self._params_nlos[param_name] = tensor
            elif isinstance(v, int):
                # Keep integers as Python int for num_clusters, etc.
                pass

    def _parameter_file(self, filename: str):
        """Return the package resource for a versioned parameter file."""

        return models.parameter_file(filename, self.spec_version)

    @abstractmethod
    def _compute_lsp_log_mean_std(self) -> None:
        r"""Computes the mean and standard deviations of LSPs in log-domain"""
        pass

    @abstractmethod
    def _compute_pathloss_basic(self) -> None:
        r"""Computes the basic component of the pathloss [dB]"""
        pass
