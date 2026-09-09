#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Base class for implementing system level channel models from 3GPP TR38.901
specification"""

from typing import Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch

from sionna.phy.channel import ChannelModel
from sionna.phy.channel.utils import deg_2_rad

from .channel_coefficients import ChannelCoefficientsGenerator, Topology
from .blockage import BlockageModelA, BlockageModelB
from .lsp import LSP, LSPGenerator
from .rays import Rays, RaysGenerator
from .system_level_scenario import SystemLevelScenario

__all__ = ["SystemLevelChannel"]


class SystemLevelChannel(ChannelModel):
    r"""
    Base class for implementing 3GPP system level channel models, such as UMi,
    UMa, RMa, InH, and InF.

    :param scenario: Scenario for the channel simulation
    :param always_generate_lsp: If `True`, new large scale parameters (LSPs)
        are generated for every new generation of channel impulse responses.
        Otherwise, always reuse the same LSPs, except if the topology is
        changed. Defaults to `False`.
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`, :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for computation (e.g., 'cpu', 'cuda:0').
        If `None`, :attr:`~sionna.phy.config.Config.device` is used.
    :param enable_spatial_consistency: If `True`, additionally generate
        LoS/NLoS states, applicable O2I terms, and small-scale random variables
        from spatially correlated fields for the current topology snapshot.
        LSP spatial correlation is applied independently of this flag. Random
        fields are not retained across topology updates; the stateful mobility
        procedure of Section 7.6.3.2 is not implemented. See
        :ref:`tr38901-spatial-consistency`. Defaults to `False`.
    :param enable_blockage: If `True`, apply the selected blockage model
        according to Section 7.6.4 of :cite:p:`TR38901V1920`.
        Blockage is disabled by default for backwards compatibility. The
        optional, on-demand temporal variability of blockage is currently not
        supported.
    :param blockage_self_blocking: Self-blocking mode for blockage model A.
        Must explicitly be ``"portrait"`` or ``"landscape"`` when model A is
        enabled. The explicit value ``"none"`` disables self-blocking as a
        non-standard extension. `None` is valid only when blockage is disabled
        or model B is selected.
    :param blockage_num_non_self_blockers: Number of non-self-blocking regions
        for blockage model A. Defaults to 4 as specified by
        :cite:p:`TR38901V1920`.
    :param blockage_model: Blockage model variant. Must be ``"A"`` or ``"B"``.
        Defaults to ``"A"``.
    :param blockage_screen_centers: Blockage model B screen centres [m].
        Required if ``blockage_model`` is ``"B"``. Shape ``[num_blockers, 3]``
        or ``[batch size, num_blockers, 3]``.
    :param blockage_screen_widths: Blockage model B screen widths [m]. Shape
        ``[num_blockers]`` or ``[batch size, num_blockers]``.
    :param blockage_screen_heights: Blockage model B screen heights [m]. Shape
        ``[num_blockers]`` or ``[batch size, num_blockers]``.

    :input num_time_samples: `int`.
        Number of time samples.

    :input sampling_frequency: `float`.
        Sampling frequency [Hz].

    :output a: [batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_samples], `torch.complex`.
        Path coefficients.

    :output tau: [batch size, num_rx, num_tx, num_paths], `torch.float`.
        Path delays [s].

    :output rays: :class:`~sionna.phy.channel.tr38901.Rays`.
        Sampled rays. Only returned if ``self.return_rays`` is `True`.

    .. rubric:: Examples

    .. code-block:: python

        import torch
        from sionna.phy.channel.tr38901 import SystemLevelScenario, SystemLevelChannel

        # Assuming a concrete scenario implementation exists
        scenario = MyScenario(carrier_frequency=3.5e9, ...)
        channel = SystemLevelChannel(scenario)

        # Set topology
        channel.set_topology(ut_loc, bs_loc, ut_orientations, bs_orientations,
                             ut_velocities, in_state)

        # Generate channel
        h, delays = channel(num_time_samples=100, sampling_frequency=1e6)
    """

    def __init__(
        self,
        scenario: SystemLevelScenario,
        always_generate_lsp: bool = False,
        precision: Optional[str] = None,
        device: Optional[str] = None,
        enable_spatial_consistency: bool = False,
        enable_blockage: bool = False,
        blockage_self_blocking: Optional[str] = None,
        blockage_num_non_self_blockers: int = 4,
        blockage_model: str = "A",
        blockage_screen_centers: Optional[torch.Tensor] = None,
        blockage_screen_widths: Optional[torch.Tensor] = None,
        blockage_screen_heights: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__(precision=scenario.precision, device=scenario.device)

        self._scenario = scenario
        self._scenario.set_spatial_consistency_enabled(enable_spatial_consistency)
        self._lsp_sampler = LSPGenerator(scenario)
        self._blockage_model = None
        if enable_blockage:
            blockage_model = blockage_model.upper()
            if blockage_model == "A":
                if blockage_self_blocking is None:
                    raise ValueError(
                        "blockage_self_blocking must explicitly select "
                        "'portrait' or 'landscape' when blockage model A is "
                        "enabled; use 'none' only for the non-standard no-self "
                        "variant"
                    )
                self._blockage_model = BlockageModelA(
                    scenario,
                    self_blocking=blockage_self_blocking,
                    num_non_self_blockers=blockage_num_non_self_blockers,
                    precision=self.precision,
                    device=self.device,
                )
            elif blockage_model == "B":
                if (
                    blockage_screen_centers is None
                    or blockage_screen_widths is None
                    or blockage_screen_heights is None
                ):
                    raise ValueError(
                        "blockage_screen_centers, blockage_screen_widths, "
                        "and blockage_screen_heights are required for "
                        "blockage model B"
                    )
                self._blockage_model = BlockageModelB(
                    scenario,
                    blockage_screen_centers,
                    blockage_screen_widths,
                    blockage_screen_heights,
                    precision=self.precision,
                    device=self.device,
                )
            else:
                raise ValueError("blockage_model must be 'A' or 'B'")
        self._ray_sampler = RaysGenerator(
            scenario,
            enable_spatial_consistency=enable_spatial_consistency,
            blockage_model=self._blockage_model,
        )
        self._set_topology_called = False
        self._return_rays = False
        self._lsp: Optional[LSP] = None

        if scenario.direction == "uplink":
            tx_array = scenario.ut_array
            rx_array = scenario.bs_array
        else:  # "downlink"
            tx_array = scenario.bs_array
            rx_array = scenario.ut_array

        self._cir_sampler = ChannelCoefficientsGenerator(
            scenario.carrier_frequency,
            tx_array,
            rx_array,
            subclustering=True,
            precision=self.precision,
            device=self.device,
        )

        # Are new LSPs needed
        self._always_generate_lsp = always_generate_lsp

    @property
    def return_rays(self) -> bool:
        """Indicates whether the call method returns the generated rays."""
        return self._return_rays

    @return_rays.setter
    def return_rays(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise TypeError("return_rays must be bool")
        self._return_rays = value

    def sample_lsp(self) -> LSP:
        r"""Sample large-scale parameters for the current topology.

        This method returns a fresh realization of the TR 38.901 large-scale
        parameters (LSPs) for the topology configured with
        :meth:`set_topology`. It is useful for diagnostics and calibration
        plots, e.g., to inspect the spatial correlation of shadow fading,
        delay spread, or angular spreads. The returned sample is not cached by
        the channel.

        :output lsp: :class:`~sionna.phy.channel.tr38901.LSP`.
            Fresh LSP realization.
        """
        if not self._set_topology_called:
            raise RuntimeError("set_topology must be called before sample_lsp()")
        return self._lsp_sampler()

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
    ) -> None:
        r"""
        Set the network topology.

        It is possible to set up a different network topology for each batch
        example. The batch size used when setting up the network topology
        is used for the link simulations.

        When calling this function, not specifying a parameter leads to the
        reuse of the previously given value. Not specifying a value that was not
        set at a former call raises an error.

        :param ut_loc: Locations of the UTs [m].
            Shape [batch size, num_ut, 3].
        :param bs_loc: Locations of the base stations [m].
            Shape [batch size, num_bs, 3].
        :param ut_orientations: Orientations of the UT arrays [radian].
            Shape [batch size, num_ut, 3].
        :param bs_orientations: Orientations of the BS arrays [radian].
            Shape [batch size, num_bs, 3].
        :param ut_velocities: Velocity vectors of the UTs [m/s].
            Shape [batch size, num_ut, 3].
        :param in_state: Indoor/outdoor state of the UTs. `True` means indoor
            and `False` means outdoor. Shape [batch size, num_ut].
        :param los: LoS/NLoS state control. If set to `True`, all outdoor UTs
            are forced to be in LoS. If set to `False`, all outdoor UTs are
            forced to be in NLoS. If a boolean tensor is provided, it specifies
            the requested LoS/NLoS state for each BS-UT link with shape
            [batch size, num_bs, num_ut] or [num_bs, num_ut]. If set to
            ``"random"``, fresh
            stochastic LoS/NLoS states are sampled following Section 7.4.2 of
            :cite:p:`TR38901V1920`. If set to `None`, the
            previous setting is reused; on the first call this is equivalent
            to ``"random"``.
        :param bs_virtual_loc: Virtual locations of the base stations for each UT [m].
            Used to compute BS-UT relative distance and angles.
            If `None` while ``bs_loc`` is specified, then it is set to
            ``bs_loc`` upon reshaping.
            Shape [batch size, num_bs, num_ut, 3].
        :param bs_site_ids: Site identifier of each BS. Co-sited base stations share the
            same site identifier and use common site-level random quantities,
            such as co-sited LSPs. If `None`, exact duplicate BS locations are
            treated as co-sited; near duplicates remain separate and emit a
            warning. Shape [num_bs] or [batch size, num_bs].
        :param spatial_consistency_track_ids: Optional grouping identifiers for
            UT entries representing positions on the same track in the current
            topology snapshot. When spatial consistency is enabled, equal IDs
            share cluster-angle signs and random ray-coupling permutations.
            They do not retain random variables across topology updates. Shape
            [num_ut] or [batch size, num_ut].
        :param distance_2d_in: Optional pre-sampled indoor 2D distance [m] for
            every UT. Values for outdoor UTs are ignored.
            Shape [batch size, num_ut] for the current O2I model, or
            [batch size, num_bs, num_ut] for the legacy below-6-GHz UMi/UMa
            model.
        :param ut_spatial_region_ids: Optional integer correlation-region
            identifier for every UT. Unequal IDs decorrelate LSP fields and
            the additional fields controlled by ``enable_spatial_consistency``;
            correlation within a region uses 2D distance. IDs do not add floor
            penetration loss or change pathloss or geometry. Blockage model A
            uses the same partition when enabled. If omitted initially,
            UMi/UMa/RMa infer a 3 m floor grid for indoor UTs, while InH/InF
            use one common region. The inferred IDs do not distinguish
            buildings. Shape [num_ut] or
            [batch size, num_ut].
        :param in_car: RMa-only in-car state of every UT. If omitted for the
            first RMa topology, all non-indoor UTs are treated as in-car per
            Table 7.2-3. Shape [batch size, num_ut].
        """

        # Update the scenario topology
        topology_args = (
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
        if self._scenario.scenario_kind == "rma":
            need_for_update = self._scenario.set_topology(
                *topology_args, in_car=in_car
            )
        else:
            if in_car is not None:
                raise ValueError("`in_car` is only supported by RMa")
            need_for_update = self._scenario.set_topology(*topology_args)

        if need_for_update:
            # Update the LSP sampler
            self._lsp_sampler.topology_updated_callback()

            # Update the ray sampler
            self._ray_sampler.topology_updated_callback()

            # Sample LSPs if no need to generate them every time
            if not self._always_generate_lsp:
                self._lsp = self._lsp_sampler()

        if not self._set_topology_called:
            self._set_topology_called = True

    def allocate_topology_tensors(
        self,
        batch_size: int,
        num_bs: int,
        num_ut: int,
    ) -> None:
        r"""
        Pre-allocate all tensors used for topology updates.

        This method should be called before using `set_topology` within a
        `torch.compile`-decorated function with CUDAGraphs enabled. It ensures
        that all internal tensors are pre-allocated with fixed shapes, allowing
        subsequent `set_topology` calls to use in-place operations that are
        compatible with CUDAGraph capture.

        Calling this method again reinitializes all topology buffers to the
        new shapes (equivalent to :meth:`reset_topology` + allocate).

        After calling this method, the shapes of the topology cannot change.

        :param batch_size: Batch size
        :param num_bs: Number of base stations
        :param num_ut: Number of user terminals
        """
        self._scenario.allocate_topology_tensors(batch_size, num_bs, num_ut)
        self._lsp_sampler.allocate_topology_tensors(batch_size, num_bs, num_ut)
        self._ray_sampler.allocate_topology_tensors(batch_size, num_bs, num_ut)

    def reset_topology(self) -> None:
        """Reset the topology to allow different batch_size/num_ut/num_bs.

        This method clears all topology buffers and returns the channel
        to its initial state. The next `set_topology` call will re-initialize
        the buffers with the new shapes.

        Use this when you need to change the batch_size, num_ut, or num_bs,
        for example when switching between training and evaluation with
        different batch sizes.

        Note: If using torch.compile, this will trigger recompilation.
        """
        self._scenario.reset_topology()
        self._lsp_sampler.reset_topology()
        self._ray_sampler.reset_topology()
        self._set_topology_called = False
        self._lsp = None

    def __call__(
        self,
        num_time_samples: int,
        sampling_frequency: float,
        batch_size_compat: Optional[float] = None,
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, Rays],
    ]:
        """Generate channel impulse responses.

        :param num_time_samples: Number of time samples
        :param sampling_frequency: Sampling frequency [Hz]
        :param batch_size_compat: Optional third argument passed by generic
            channel layers. For a compatibility call of the form
            ``(batch_size, num_time_samples, sampling_frequency)``, the first
            argument is ignored, the second is used as ``num_time_samples``,
            and this argument is used as ``sampling_frequency``.
        """
        # Some channel layers (GenerateOFDMChannel and GenerateTimeChannel)
        # give as input (batch_size, num_time_samples, sampling_frequency)
        # instead of (num_time_samples, sampling_frequency), as specified
        # in the ChannelModel interface.
        # With this model, the batch size is ignored, and only the required
        # parameters are kept.
        if batch_size_compat is not None:
            # batch_size = num_time_samples
            num_time_samples = int(sampling_frequency)
            sampling_frequency = batch_size_compat

        # Sample LSPs if required
        if self._always_generate_lsp:
            lsp = self._lsp_sampler()
        else:
            lsp = self._lsp

        # Sample rays
        rays = self._ray_sampler(lsp)

        # Sample channel responses
        # First we need to create a topology
        # Indicates which end of the channel is moving: TX or RX
        if self._scenario.direction == "downlink":
            moving_end = "rx"
            tx_orientations = self._scenario.bs_orientations
            rx_orientations = self._scenario.ut_orientations
        else:  # 'uplink'
            moving_end = "tx"
            tx_orientations = self._scenario.ut_orientations
            rx_orientations = self._scenario.bs_orientations

        topology = Topology(
            velocities=self._scenario.ut_velocities,
            moving_end=moving_end,
            los_aoa=deg_2_rad(self._scenario.los_aoa),
            los_aod=deg_2_rad(self._scenario.los_aod),
            los_zoa=deg_2_rad(self._scenario.los_zoa),
            los_zod=deg_2_rad(self._scenario.los_zod),
            los=self._scenario.los,
            distance_3d=self._scenario.distance_3d,
            tx_orientations=tx_orientations,
            rx_orientations=rx_orientations,
        )

        # The channel coefficient needs the cluster delay spread parameter in ns
        c_ds = self._scenario.get_param("cDS") * 1e-9

        # According to the link direction, we need to specify which from BS
        # and UT is uplink, and which is downlink.
        # Default is downlink, so for uplink we create the TX/RX view expected
        # by the coefficient generator by transposing BS and UT dimensions and
        # swapping angles of arrival/departure. The returned rays follow this
        # TX/RX view when ``return_rays`` is enabled.
        if self._scenario.direction == "uplink":
            aoa = rays.aoa
            zoa = rays.zoa
            aod = rays.aod
            zod = rays.zod
            rays.aod = aoa.permute(0, 2, 1, 3, 4)
            rays.zod = zoa.permute(0, 2, 1, 3, 4)
            rays.aoa = aod.permute(0, 2, 1, 3, 4)
            rays.zoa = zod.permute(0, 2, 1, 3, 4)
            rays.powers = rays.powers.permute(0, 2, 1, 3)
            rays.delays = rays.delays.permute(0, 2, 1, 3)
            if rays.cluster_sort_indices is not None:
                rays.cluster_sort_indices = rays.cluster_sort_indices.permute(
                    0, 2, 1, 3
                )
            if rays.strongest_cluster_indices is not None:
                rays.strongest_cluster_indices = (
                    rays.strongest_cluster_indices.permute(0, 2, 1, 3)
                )
            rays.xpr = rays.xpr.permute(0, 2, 1, 3, 4)
            if rays.phases is not None:
                rays.phases = rays.phases.permute(0, 2, 1, 3, 4, 5)
            if rays.blockage_loss_db is not None:
                rays.blockage_loss_db = rays.blockage_loss_db.permute(
                    0, 2, 1, 3, 4
                )
            if rays.los_blockage_loss_db is not None:
                rays.los_blockage_loss_db = rays.los_blockage_loss_db.permute(
                    0, 2, 1
                )

            los_aod = topology.los_aod
            los_aoa = topology.los_aoa
            los_zod = topology.los_zod
            los_zoa = topology.los_zoa
            topology.los_aoa = los_aod.permute(0, 2, 1)
            topology.los_aod = los_aoa.permute(0, 2, 1)
            topology.los_zoa = los_zod.permute(0, 2, 1)
            topology.los_zod = los_zoa.permute(0, 2, 1)
            topology.los = topology.los.permute(0, 2, 1)
            c_ds = c_ds.permute(0, 2, 1)
            topology.distance_3d = topology.distance_3d.permute(0, 2, 1)

            # Concerning LSPs, only these two are used.
            # We do not transpose the others to reduce complexity
            k_factor = lsp.k_factor.permute(0, 2, 1)
            sf = lsp.sf.permute(0, 2, 1)
        else:
            k_factor = lsp.k_factor
            sf = lsp.sf

        h, delays = self._cir_sampler(
            num_time_samples, sampling_frequency, k_factor, rays, topology, c_ds
        )

        # Step 12
        h = self._step_12(h, sf, lsp.pathloss)

        # Reshaping to match the expected output
        h = h.permute(0, 2, 4, 1, 5, 3, 6)
        delays = delays.permute(0, 2, 1, 3)

        # Stop gradients to avoid useless backpropagation
        h = h.detach()
        delays = delays.detach()

        if self.return_rays:
            return h, delays, rays
        else:
            return h, delays

    def show_topology(self, bs_index: int = 0, batch_index: int = 0) -> None:
        r"""
        Show the network topology of the batch example with index
        ``batch_index``.

        The ``bs_index`` parameter specifies with respect to which BS the
        LoS/NLoS state of UTs is indicated.

        :param bs_index: BS index with respect to which the LoS/NLoS state of
            UTs is indicated. Defaults to 0.
        :param batch_index: Batch example for which the topology is shown.
            Defaults to 0.
        """

        def draw_coordinate_system(ax, loc, ort, delta):
            # This function draws the coordinate system x-y-z, represented by
            # three lines with colors red-green-blue (rgb), to show the
            # orientation of the array (LCS) in the GCS.
            # To always draw a visible and not too big axes, we scale them
            # according to the spread of the network in each direction.

            a = ort[0]
            b = ort[1]
            c = ort[2]

            arrow_ratio_size = 0.1

            x_ = np.array([np.cos(a) * np.cos(b), np.sin(a) * np.cos(b), -np.sin(b)])
            scale_x = arrow_ratio_size / np.sqrt(np.sum(np.square(x_ / delta)))
            x_ = x_ * scale_x

            y_ = np.array(
                [
                    np.cos(a) * np.sin(b) * np.sin(c) - np.sin(a) * np.cos(c),
                    np.sin(a) * np.sin(b) * np.sin(c) + np.cos(a) * np.cos(c),
                    np.cos(b) * np.sin(c),
                ]
            )
            scale_y = arrow_ratio_size / np.sqrt(np.sum(np.square(y_ / delta)))
            y_ = y_ * scale_y

            z_ = np.array(
                [
                    np.cos(a) * np.sin(b) * np.cos(c) + np.sin(a) * np.sin(c),
                    np.sin(a) * np.sin(b) * np.cos(c) - np.cos(a) * np.sin(c),
                    np.cos(b) * np.cos(c),
                ]
            )
            scale_z = arrow_ratio_size / np.sqrt(np.sum(np.square(z_ / delta)))
            z_ = z_ * scale_z

            ax.plot(
                [loc[0], loc[0] + x_[0]],
                [loc[1], loc[1] + x_[1]],
                [loc[2], loc[2] + x_[2]],
                c="r",
            )
            ax.plot(
                [loc[0], loc[0] + y_[0]],
                [loc[1], loc[1] + y_[1]],
                [loc[2], loc[2] + y_[2]],
                c="g",
            )
            ax.plot(
                [loc[0], loc[0] + z_[0]],
                [loc[1], loc[1] + z_[1]],
                [loc[2], loc[2] + z_[2]],
                c="b",
            )

        indoor = self._scenario.indoor.cpu().numpy()[batch_index]
        los = self._scenario.los.cpu().numpy()[batch_index, bs_index]

        indoor_indices = np.where(indoor)
        los_indices = np.where(los)
        nlos_indices = np.where(
            np.logical_and(np.logical_not(indoor), np.logical_not(los))
        )

        ut_loc = self._scenario.ut_loc.cpu().numpy()[batch_index]
        bs_loc = self._scenario.bs_loc.cpu().numpy()[batch_index]
        ut_orientations = self._scenario.ut_orientations.cpu().numpy()[batch_index]
        bs_orientations = self._scenario.bs_orientations.cpu().numpy()[batch_index]

        delta_x = np.max(np.concatenate([ut_loc[:, 0], bs_loc[:, 0]])) - np.min(
            np.concatenate([ut_loc[:, 0], bs_loc[:, 0]])
        )
        delta_y = np.max(np.concatenate([ut_loc[:, 1], bs_loc[:, 1]])) - np.min(
            np.concatenate([ut_loc[:, 1], bs_loc[:, 1]])
        )
        delta_z = np.max(np.concatenate([ut_loc[:, 2], bs_loc[:, 2]])) - np.min(
            np.concatenate([ut_loc[:, 2], bs_loc[:, 2]])
        )
        delta = np.array([delta_x, delta_y, delta_z])

        indoor_ut_loc = ut_loc[indoor_indices]
        los_ut_loc = ut_loc[los_indices]
        nlos_ut_loc = ut_loc[nlos_indices]

        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        # Showing BS
        ax.scatter(
            bs_loc[:, 0],
            bs_loc[:, 1],
            bs_loc[:, 2],
            c="k",
            label="BS",
            depthshade=False,
        )
        # Showing BS indices and orientations
        for u, loc in enumerate(bs_loc):
            ax.text(loc[0], loc[1], loc[2], f"{u}")
            draw_coordinate_system(ax, loc, bs_orientations[u], delta)
        # Showing UTs
        ax.scatter(
            indoor_ut_loc[:, 0],
            indoor_ut_loc[:, 1],
            indoor_ut_loc[:, 2],
            c="b",
            label="UT Indoor",
            depthshade=False,
        )
        ax.scatter(
            los_ut_loc[:, 0],
            los_ut_loc[:, 1],
            los_ut_loc[:, 2],
            c="r",
            label="UT LoS",
            depthshade=False,
        )
        ax.scatter(
            nlos_ut_loc[:, 0],
            nlos_ut_loc[:, 1],
            nlos_ut_loc[:, 2],
            c="y",
            label="UT NLoS",
            depthshade=False,
        )
        # Showing UT indices and orientations
        for u, loc in enumerate(ut_loc):
            ax.text(loc[0], loc[1], loc[2], f"{u}")
            draw_coordinate_system(ax, loc, ut_orientations[u], delta)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_zlabel("z [m]")
        plt.legend()
        plt.tight_layout()

    #####################################################
    # Internal utility methods
    #####################################################

    def _step_12(
        self,
        h: torch.Tensor,
        sf: torch.Tensor,
        pathloss: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply path loss and shadow fading ``sf`` to paths coefficients ``h``.

        :param h: Paths coefficients.
            Shape [batch size, num_tx, num_rx, num_paths, num_rx_ant, num_tx_ant, num_time_samples].
        :param sf: Shadow fading.
            Shape [batch size, num_tx, num_rx].
        :param pathloss: Path loss [dB] sampled with ``sf`` and the other LSPs.
            Shape [batch size, num_bs, num_ut].
        """
        if self._scenario.pathloss_enabled:
            if pathloss is None:
                raise RuntimeError("Path loss was not sampled with the LSPs")
            pl_db = pathloss
            if self._scenario.direction == "uplink":
                pl_db = pl_db.permute(0, 2, 1)
        else:
            pl_db = torch.tensor(0.0, dtype=self.dtype, device=self.device)

        if not self._scenario.shadow_fading_enabled:
            sf = torch.ones_like(sf)

        gain = torch.pow(
            torch.tensor(10.0, dtype=self.dtype, device=self.device), -pl_db / 20.0
        ) * torch.sqrt(sf)

        # Expand gain to match h dimensions
        num_extra_dims = h.dim() - gain.dim()
        for _ in range(num_extra_dims):
            gain = gain.unsqueeze(-1)

        h = h * gain.to(dtype=h.dtype)

        return h
