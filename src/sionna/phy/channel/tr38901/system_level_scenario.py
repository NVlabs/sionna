#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Class used to define a system level 3GPP channel simulation scenario"""

import json
from abc import abstractmethod
from typing import Optional, Union

import torch
from importlib_resources import files

from sionna.phy import PI, SPEED_OF_LIGHT
from sionna.phy.object import Object
from sionna.phy.utils import insert_dims, sample_bernoulli, rand
from sionna.phy.channel.utils import rad_2_deg, wrap_angle_0_360

from . import models
from .antenna import PanelArray

__all__ = ["SystemLevelScenario"]


class SystemLevelScenario(Object):
    r"""
    Base class for setting up the scenario for system level 3GPP channel
    simulation.

    Scenarios for system level channel simulation, such as UMi, UMa, or RMa,
    are defined by implementing this base class.

    :param carrier_frequency: Carrier frequency [Hz]
    :param o2i_model: Outdoor to indoor (O2I) pathloss model, used for
        indoor UTs. Must be ``"low"`` or ``"high"``.
        See section 7.4.3 from 38.901 specification.
    :param ut_array: Panel array configuration used by UTs
    :param bs_array: Panel array configuration used by BSs
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
        super().__init__(precision=precision, device=device)

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
        assert o2i_model in ("low", "high"), "o2i_model must be 'low' or 'high'"
        self._o2i_model = o2i_model

        # UTs and BSs arrays
        assert isinstance(
            ut_array, PanelArray
        ), "'ut_array' must be an instance of PanelArray"
        assert isinstance(
            bs_array, PanelArray
        ), "'bs_array' must be an instance of PanelArray"
        self._ut_array = ut_array
        self._bs_array = bs_array

        # Direction
        assert direction in (
            "uplink",
            "downlink",
        ), "'direction' must be 'uplink' or 'downlink'"
        self._direction = direction

        # Pathloss and shadow fading
        self._enable_pathloss = enable_pathloss
        self._enable_shadow_fading = enable_shadow_fading

        # Scenario
        self._ut_loc: Optional[torch.Tensor] = None
        self._bs_loc: Optional[torch.Tensor] = None
        self._bs_virtual_loc: Optional[torch.Tensor] = None
        self._ut_orientations: Optional[torch.Tensor] = None
        self._bs_orientations: Optional[torch.Tensor] = None
        self._ut_velocities: Optional[torch.Tensor] = None
        self._in_state: Optional[torch.Tensor] = None
        self._requested_los: Optional[bool] = None

        # Internal state
        self._distance_2d: Optional[torch.Tensor] = None
        self._distance_3d: Optional[torch.Tensor] = None
        self._distance_2d_in: Optional[torch.Tensor] = None
        self._distance_2d_out: Optional[torch.Tensor] = None
        self._distance_3d_in: Optional[torch.Tensor] = None
        self._distance_3d_out: Optional[torch.Tensor] = None
        self._matrix_ut_distance_2d: Optional[torch.Tensor] = None
        self._los_aod: Optional[torch.Tensor] = None
        self._los_aoa: Optional[torch.Tensor] = None
        self._los_zod: Optional[torch.Tensor] = None
        self._los_zoa: Optional[torch.Tensor] = None
        self._los: Optional[torch.Tensor] = None
        self._lsp_log_mean: Optional[torch.Tensor] = None
        self._lsp_log_std: Optional[torch.Tensor] = None
        self._zod_offset: Optional[torch.Tensor] = None
        self._pl_b: Optional[torch.Tensor] = None

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
    def pathloss_enabled(self) -> bool:
        """`True` if pathloss is enabled. `False` otherwise."""
        return self._enable_pathloss

    @property
    def shadow_fading_enabled(self) -> bool:
        """`True` if shadow fading is enabled. `False` otherwise."""
        return self._enable_shadow_fading

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
        """Number of BSs"""
        return self._bs_loc.shape[1]

    @property
    def h_ut(self) -> torch.Tensor:
        """Height of UTs [m]. Shape [batch size, number of UTs]"""
        return self._ut_loc[:, :, 2]

    @property
    def h_bs(self) -> torch.Tensor:
        """Height of BSs [m]. Shape [batch size, number of BSs]"""
        return self._bs_loc[:, :, 2]

    @property
    def ut_loc(self) -> torch.Tensor:
        """Locations of UTs [m]. Shape [batch size, number of UTs, 3]"""
        return self._ut_loc

    @property
    def bs_loc(self) -> torch.Tensor:
        """Locations of BSs [m]. Shape [batch size, number of BSs, 3]"""
        return self._bs_loc

    @property
    def bs_virtual_loc(self) -> torch.Tensor:
        """Virtual location of BSs, relative to each UT position [m].
        Useful in case of wraparound.
        Broadcastable to [batch size, number of UTs, number of BSs, 3]"""
        return self._bs_virtual_loc

    @property
    def ut_orientations(self) -> torch.Tensor:
        """Orientations of UTs [radian]. Shape [batch size, number of UTs, 3]"""
        return self._ut_orientations

    @property
    def bs_orientations(self) -> torch.Tensor:
        """Orientations of BSs [radian]. Shape [batch size, number of BSs, 3]"""
        return self._bs_orientations

    @property
    def ut_velocities(self) -> torch.Tensor:
        """UTs velocities [m/s]. Shape [batch size, number of UTs, 3]"""
        return self._ut_velocities

    @property
    def ut_array(self) -> PanelArray:
        """PanelArray used by UTs"""
        return self._ut_array

    @property
    def bs_array(self) -> PanelArray:
        """PanelArray used by BSs"""
        return self._bs_array

    @property
    def indoor(self) -> torch.Tensor:
        """Indoor state of UTs. `True` is indoor, `False` otherwise.
        Shape [batch size, number of UTs]"""
        return self._in_state

    @property
    def los(self) -> torch.Tensor:
        """LoS state of BS-UT links. `True` if LoS, `False` otherwise.
        Shape [batch size, number of BSs, number of UTs]"""
        return self._los

    @property
    def distance_2d(self) -> torch.Tensor:
        """Distance between each UT and each BS in the X-Y plane [m].
        Shape [batch size, number of BSs, number of UTs]"""
        return self._distance_2d

    @property
    def distance_2d_in(self) -> torch.Tensor:
        """Indoor distance between each UT and BS in the X-Y plane [m],
        i.e., part of the total distance that corresponds to indoor
        propagation in the X-Y plane.
        Set to 0 for UTs located outdoor.
        Shape [batch size, number of BSs, number of UTs]"""
        return self._distance_2d_in

    @property
    def distance_2d_out(self) -> torch.Tensor:
        """Outdoor distance between each UT and BS in the X-Y plane [m],
        i.e., part of the total distance that corresponds to outdoor
        propagation in the X-Y plane.
        Equals ``distance_2d`` for UTs located outdoor.
        Shape [batch size, number of BSs, number of UTs]"""
        return self._distance_2d_out

    @property
    def distance_3d(self) -> torch.Tensor:
        """Distance between each UT and each BS [m].
        Shape [batch size, number of BSs, number of UTs]"""
        return self._distance_3d

    @property
    def distance_3d_in(self) -> torch.Tensor:
        """Indoor distance between each UT and BS [m],
        i.e., part of the total distance that corresponds to indoor
        propagation. Set to 0 for UTs located outdoor.
        Shape [batch size, number of BSs, number of UTs]"""
        return self._distance_3d_in

    @property
    def distance_3d_out(self) -> torch.Tensor:
        """Outdoor distance between each UT and BS [m],
        i.e., part of the total distance that corresponds to outdoor
        propagation. Equals ``distance_3d`` for UTs located outdoor.
        Shape [batch size, number of BSs, number of UTs]"""
        return self._distance_3d_out

    @property
    def matrix_ut_distance_2d(self) -> torch.Tensor:
        """Distance between all pairs of UTs in the X-Y plane [m].
        Shape [batch size, number of UTs, number of UTs]"""
        return self._matrix_ut_distance_2d

    @property
    def los_aod(self) -> torch.Tensor:
        """LoS azimuth angle of departure of each BS-UT link [deg].
        Shape [batch size, number of BSs, number of UTs]"""
        return self._los_aod

    @property
    def los_aoa(self) -> torch.Tensor:
        """LoS azimuth angle of arrival of each BS-UT link [deg].
        Shape [batch size, number of BSs, number of UTs]"""
        return self._los_aoa

    @property
    def los_zod(self) -> torch.Tensor:
        """LoS zenith angle of departure of each BS-UT link [deg].
        Shape [batch size, number of BSs, number of UTs]"""
        return self._los_zod

    @property
    def los_zoa(self) -> torch.Tensor:
        """LoS zenith angle of arrival of each BS-UT link [deg].
        Shape [batch size, number of BSs, number of UTs]"""
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
        Shape [batch size, number of BSs, number of UTs, 7].
        The last dimension corresponds to the LSPs, in the following order:
        DS - ASD - ASA - SF - K - ZSA - ZSD"""
        return self._lsp_log_mean

    @property
    def lsp_log_std(self) -> torch.Tensor:
        """STD of LSPs in the log domain.
        Shape [batch size, number of BSs, number of UTs, 7].
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
        los: Optional[bool] = None,
        bs_virtual_loc: Optional[torch.Tensor] = None,
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
        :param bs_loc: Locations of BSs [m].
            Shape [batch size, number of BSs, 3]
        :param ut_orientations: Orientations of the UTs arrays [radian].
            Shape [batch size, number of UTs, 3]
        :param bs_orientations: Orientations of the BSs arrays [radian].
            Shape [batch size, number of BSs, 3]
        :param ut_velocities: Velocity vectors of UTs [m/s].
            Shape [batch size, number of UTs, 3]
        :param in_state: Indoor/outdoor state of UTs. `True` means indoor and
            `False` means outdoor.
            Shape [batch size, number of UTs]
        :param los: If not `None`, all UTs located outdoor are forced to be in
            LoS if ``los`` is set to `True`, or in NLoS if it is set to
            `False`. If set to `None`, the LoS/NLoS states of UTs is set
            following 3GPP specification (Section 7.4.2 of TR 38.901).
        :param bs_virtual_loc: Virtual locations of BSs for each UT [m].
            Used to compute BS-UT relative distance and angles.
            If `None` while ``bs_loc`` is specified, then it is set to
            ``bs_loc`` upon reshaping. If neither ``bs_virtual_loc`` nor
            ``bs_loc`` are specified, then the previous value is used.
            Shape [batch size, number of BSs, number of UTs, 3]

        :output updated: `True` if the topology was updated, `False` otherwise
        """

        assert (ut_loc is not None) or (
            self._ut_loc is not None
        ), "`ut_loc` is None and was not previously set"

        assert (bs_loc is not None) or (
            self._bs_loc is not None
        ), "`bs_loc` is None and was not previously set"

        assert (
            (bs_virtual_loc is not None)
            or (bs_loc is not None)
            or (self._bs_virtual_loc is not None)
        ), "`bs_virtual_loc` is None and was not previously set"

        assert (in_state is not None) or (
            self._in_state is not None
        ), "`in_state` is None and was not previously set"

        assert (ut_orientations is not None) or (
            self._ut_orientations is not None
        ), "`ut_orientations` is None and was not previously set"

        assert (bs_orientations is not None) or (
            self._bs_orientations is not None
        ), "`bs_orientations` is None and was not previously set"

        assert (ut_velocities is not None) or (
            self._ut_velocities is not None
        ), "`ut_velocities` is None and was not previously set"

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
            # [batch size, number of BSs, 1, 3]
            self._update_attr(
                "_bs_virtual_loc", insert_dims(self._bs_loc, num_dims=1, axis=2)
            )

        if bs_orientations is not None:
            self._update_attr("_bs_orientations", self._convert(bs_orientations))

        if ut_orientations is not None:
            self._update_attr("_ut_orientations", self._convert(ut_orientations))

        if ut_velocities is not None:
            self._update_attr("_ut_velocities", self._convert(ut_velocities))

        if in_state is not None:
            if isinstance(in_state, torch.Tensor):
                in_state_tensor = in_state.to(device=self.device)
            else:
                in_state_tensor = torch.as_tensor(in_state, device=self.device)
            self._update_attr("_in_state", in_state_tensor)
            need_for_update = True

        if los is not None:
            self._requested_los = los
            need_for_update = True

        if need_for_update:
            # Update topology-related quantities
            self._compute_distance_2d_3d_and_angles()
            self._sample_indoor_distance()
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
            "_ut_orientations",
            "_bs_orientations",
            "_ut_velocities",
            "_in_state",
            "_distance_2d",
            "_distance_3d",
            "_distance_2d_in",
            "_distance_2d_out",
            "_distance_3d_in",
            "_distance_3d_out",
            "_matrix_ut_distance_2d",
            "_los_aod",
            "_los_aoa",
            "_los_zod",
            "_los_zoa",
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

        # Clear any cached compiled graphs that might reference old buffers
        import torch._dynamo

        torch._dynamo.reset()

    def allocate_topology_tensors(
        self,
        batch_size: int,
        num_bs: int,
        num_ut: int,
    ) -> None:
        r"""
        Pre-allocate all tensors used for topology updates.

        This method is optional. When using `set_topology` inside a
        `torch.compile`-decorated function, tensors are automatically allocated
        as buffers on the first call and updated in-place on subsequent calls.

        Calling this method explicitly can be useful if you want to ensure
        all tensors are allocated before the first forward pass, or if you
        want to pre-allocate with specific shapes before calling `set_topology`.

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
                batch_size, num_bs, 1, 3, dtype=self.dtype, device=self.device
            ),
        )
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

        # LoS state tensor
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
        a tensor with shape [batch size, number of BSs, number of UTs] of the
        parameter value according to each BS-UT link state (LoS, NLoS,
        indoor).

        :param parameter_name: Name of the parameter used in the
            configuration file

        :output value: Parameter value for each BS-UT link,
            shape [batch size, number of BSs, number of UTs]
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
        [batch size, number of BSs, number of UTs] based on the link state
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
        parameter_tensor = torch.where(
            (~self.los) & (~indoor), parameter_value_nlos, parameter_tensor
        )

        # O2I
        if isinstance(parameter_value_o2i, torch.Tensor):
            parameter_value_o2i = parameter_value_o2i.to(
                dtype=self.dtype, device=self.device
            )
        else:
            parameter_value_o2i = torch.tensor(
                parameter_value_o2i, dtype=self.dtype, device=self.device
            )
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
        los_zoa = los_zod - PI

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

    def _sample_los(self) -> None:
        r"""Set the LoS state of each UT randomly, following the procedure
        described in section 7.4.2 of TR 38.901.
        LoS state of each UT is randomly assigned according to a Bernoulli
        distribution, which probability depends on the channel model.
        """
        if self._requested_los is None:
            los_probability = self.los_probability
            los = sample_bernoulli(
                [self.batch_size, self.num_bs, self.num_ut],
                los_probability,
                precision=self.precision,
                device=self.device,
            )
        else:
            los = torch.full(
                (self.batch_size, self.num_bs, self.num_ut),
                self._requested_los,
                dtype=torch.bool,
                device=self.device,
            )

        self._update_attr("_los", los & (~self._in_state.unsqueeze(1)))

    def _sample_indoor_distance(self) -> None:
        r"""Sample 2D indoor distances for indoor devices, according to section
        7.4.3.1 of TR 38.901.
        """
        indoor = self.indoor
        indoor = indoor.unsqueeze(1)  # For broadcasting with BS dim
        indoor_mask = torch.where(
            indoor,
            torch.tensor(1.0, dtype=self.dtype, device=self.device),
            torch.tensor(0.0, dtype=self.dtype, device=self.device),
        )

        # Sample the indoor 2D distances for each BS-UT link.
        # Per TR 38.901 §7.4.3.1, d_2D-in = min(U1, U2) where U1 and U2
        # are independent uniform samples over [min_2d_in, max_2d_in].
        # A single uniform sample was used previously, which is incorrect.
        u1 = (
            rand(
                (self.batch_size, self.num_bs, self.num_ut),
                dtype=self.dtype,
                device=self.device,
                generator=self.torch_rng,
            )
            * (self.max_2d_in - self.min_2d_in)
            + self.min_2d_in
        )
        u2 = (
            rand(
                (self.batch_size, self.num_bs, self.num_ut),
                dtype=self.dtype,
                device=self.device,
                generator=self.torch_rng,
            )
            * (self.max_2d_in - self.min_2d_in)
            + self.min_2d_in
        )
        distance_2d_in = torch.minimum(u1, u2) * indoor_mask
        self._update_attr("_distance_2d_in", distance_2d_in)

        # Compute the outdoor 2D distances
        self._update_attr("_distance_2d_out", self.distance_2d - self._distance_2d_in)

        # Compute the indoor 3D distances
        self._update_attr(
            "_distance_3d_in",
            (self._distance_2d_in / self.distance_2d) * self.distance_3d,
        )

        # Compute the outdoor 3D distances
        self._update_attr("_distance_3d_out", self.distance_3d - self._distance_3d_in)

    def _load_params(self) -> None:
        r"""Load the configuration files corresponding to the 3 possible states
        of UTs: LoS, NLoS, and O2I"""

        source = files(models).joinpath(self.o2i_parameter_filepath)
        with open(source) as f:
            self._params_o2i = json.load(f)

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

        source = files(models).joinpath(self.los_parameter_filepath)
        with open(source) as f:
            self._params_los = json.load(f)

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

        source = files(models).joinpath(self.nlos_parameter_filepath)
        with open(source) as f:
            self._params_nlos = json.load(f)

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

    @abstractmethod
    def _compute_lsp_log_mean_std(self) -> None:
        r"""Computes the mean and standard deviations of LSPs in log-domain"""
        pass

    @abstractmethod
    def _compute_pathloss_basic(self) -> None:
        r"""Computes the basic component of the pathloss [dB]"""
        pass
