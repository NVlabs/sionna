#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Urban microcell (UMi) channel model from 3GPP TR38.901 specification"""

from typing import Optional, Union

import torch

from .system_level_channel import SystemLevelChannel
from .umi_scenario import UMiScenario
from .antenna import HandheldUTArray, PanelArray

__all__ = ["UMi"]


class UMi(SystemLevelChannel):
    r"""
    Urban microcell (UMi) channel model from 3GPP
    :cite:p:`TR38901V1920` specification.

    Setting up a UMi model requires configuring the network topology, i.e., the
    UT and base-station locations, UT velocities, etc. This is achieved using the
    :meth:`~sionna.phy.channel.tr38901.UMi.set_topology` method. Setting a
    different topology for each batch example is possible. The batch size used
    when setting up the network topology is used for the link simulations.
    TR 38.901 calibration-style multi-cell drops can be generated with
    :func:`~sionna.sys.gen_tr38901_multicell_topology`; more general hexagonal
    grid drops can be generated with
    :func:`~sionna.sys.gen_hexgrid_topology`.

    Spatial consistency and blockage are optional add-on features, both
    disabled by default. See :ref:`tr38901-spatial-consistency` and
    :ref:`tr38901-blockage` for details.

    :param carrier_frequency: Carrier frequency [Hz]
    :param o2i_model: Outdoor-to-indoor loss model for UTs located indoor.
        Set to ``"low"`` to use the low-loss model, or to ``"high"`` to use
        the high-loss model. See section 7.4.3 of
        :cite:p:`TR38901V1920` for details.
    :param ut_array: Antenna array used by the UTs. This can be a
        :class:`~sionna.phy.channel.tr38901.PanelArray` or
        :class:`~sionna.phy.channel.tr38901.HandheldUTArray`.
    :param bs_array: Antenna array used by the base stations. This can be a
        :class:`~sionna.phy.channel.tr38901.PanelArray` or
        :class:`~sionna.phy.channel.tr38901.HandheldUTArray`.
    :param direction: Link direction. Either ``"uplink"`` or ``"downlink"``.
    :param enable_pathloss: If `True`, apply pathloss. Otherwise don't.
        Defaults to `True`.
    :param enable_shadow_fading: If `True`, apply shadow fading. Otherwise
        don't. Defaults to `True`.
    :param always_generate_lsp: If `True`, new large scale parameters (LSPs)
        are generated for every new generation of channel impulse responses.
        Otherwise, always reuse the same LSPs, except if the topology is
        changed. Defaults to `False`.
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`, :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for computation (e.g., ``"cpu"``, ``"cuda:0"``).
        If `None`, :attr:`~sionna.phy.config.Config.device` is used.
    :param spec_version: Version of the TR 38.901 parameter tables to use.
        Supported values are ``"16.1"`` and ``"19.2"``. Defaults to
        ``"19.2"``.
    :param enable_spatial_consistency: If `True`, additionally generate
        LoS/NLoS states, applicable O2I terms, and small-scale random variables
        from spatially correlated fields for the current topology snapshot.
        LSP spatial correlation is always applied. Random fields are not
        retained across topology updates. See
        :ref:`tr38901-spatial-consistency`. Defaults to `False`.
    :param enable_blockage: If `True`, apply the selected blockage model
        according to Section 7.6.4 of :cite:p:`TR38901V1920`.
        The optional, on-demand temporal variability of blockage is currently
        not supported. See :ref:`tr38901-blockage`. Defaults to `False`.
    :param blockage_self_blocking: Self-blocking mode for blockage model A.
        Must explicitly be ``"portrait"`` or ``"landscape"`` when model A is
        enabled. The explicit value ``"none"`` disables self-blocking as a
        non-standard extension. `None` is valid only when blockage is disabled
        or model B is selected.
    :param blockage_num_non_self_blockers: Number of non-self-blocking regions
        for blockage model A. Defaults to 4.
    :param blockage_model: Blockage model variant. Must be ``"A"`` or ``"B"``.
        Defaults to ``"A"``.
    :param blockage_screen_centers: Blockage model B screen centres [m].
    :param blockage_screen_widths: Blockage model B screen widths [m].
    :param blockage_screen_heights: Blockage model B screen heights [m].

    :input num_time_samples: `int`.
        Number of time samples.

    :input sampling_frequency: `float`.
        Sampling frequency [Hz].

    :output a: [batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_samples], `torch.complex`.
        Path coefficients.

    :output tau: [batch size, num_rx, num_tx, num_paths], `torch.float`.
        Path delays [s].

    .. rubric:: Examples

    .. code-block:: python

        import torch
        from sionna.phy.channel.tr38901 import PanelArray, UMi
        from sionna.sys import gen_tr38901_multicell_topology

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        carrier_frequency = 3.5e9

        bs_array = PanelArray(num_rows_per_panel=1, num_cols_per_panel=1,
                              polarization='dual', polarization_type='cross',
                              antenna_pattern='38.901',
                              carrier_frequency=carrier_frequency,
                              device=device)
        ut_array = PanelArray(num_rows_per_panel=1, num_cols_per_panel=1,
                              polarization='single', polarization_type='V',
                              antenna_pattern='omni',
                              carrier_frequency=carrier_frequency,
                              device=device)

        channel_model = UMi(carrier_frequency=carrier_frequency,
                            o2i_model='low',
                            ut_array=ut_array,
                            bs_array=bs_array,
                            direction='downlink',
                            device=device)

        topology = gen_tr38901_multicell_topology("umi",
                                                  batch_size=1,
                                                  num_ut_per_sector=1,
                                                  carrier_frequency=carrier_frequency,
                                                  num_rings=1,
                                                  device=device)

        # The helper output can be replaced by explicit tensors for arbitrary
        # geometries.
        channel_model.set_topology(*topology)

        h, tau = channel_model(num_time_samples=1, sampling_frequency=1e6)
    """

    def __init__(
        self,
        carrier_frequency: float,
        o2i_model: str,
        ut_array: Union[PanelArray, HandheldUTArray],
        bs_array: Union[PanelArray, HandheldUTArray],
        direction: str,
        enable_pathloss: bool = True,
        enable_shadow_fading: bool = True,
        always_generate_lsp: bool = False,
        precision: Optional[str] = None,
        device: Optional[str] = None,
        spec_version: str = "19.2",
        enable_spatial_consistency: bool = False,
        enable_blockage: bool = False,
        blockage_self_blocking: Optional[str] = None,
        blockage_num_non_self_blockers: int = 4,
        blockage_model: str = "A",
        blockage_screen_centers=None,
        blockage_screen_widths=None,
        blockage_screen_heights=None,
    ) -> None:
        # UMi scenario
        scenario = UMiScenario(
            carrier_frequency,
            o2i_model,
            ut_array,
            bs_array,
            direction,
            enable_pathloss,
            enable_shadow_fading,
            spec_version=spec_version,
            precision=precision,
            device=device,
        )

        super().__init__(
            scenario,
            always_generate_lsp,
            enable_spatial_consistency=enable_spatial_consistency,
            enable_blockage=enable_blockage,
            blockage_self_blocking=blockage_self_blocking,
            blockage_num_non_self_blockers=blockage_num_non_self_blockers,
            blockage_model=blockage_model,
            blockage_screen_centers=blockage_screen_centers,
            blockage_screen_widths=blockage_screen_widths,
            blockage_screen_heights=blockage_screen_heights,
            precision=precision,
            device=device,
        )

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
    ) -> None:
        r"""Set the network topology.

        This method forwards to
        :meth:`~sionna.phy.channel.tr38901.SystemLevelChannel.set_topology`.
        UMi calibration-style multi-cell topologies can be generated with
        :func:`~sionna.sys.gen_tr38901_multicell_topology`; more general
        hexagonal-grid topologies can be generated with
        :func:`~sionna.sys.gen_hexgrid_topology`.

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
            ``"random"``, fresh stochastic LoS/NLoS states are sampled
            following Section 7.4.2 of
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
            UT entries representing positions on one track in the current
            topology snapshot. When spatial consistency is enabled, equal IDs
            share cluster-angle signs and random ray-coupling permutations,
            but do not retain random variables across topology updates. Shape
            [num_ut] or [batch size, num_ut].
        :param distance_2d_in: Optional pre-sampled indoor 2D distance [m] for
            every UT. Values for outdoor UTs are ignored.
            Shape [batch size, num_ut] for the current O2I model, or
            [batch size, num_bs, num_ut] below 6 GHz.
        :param ut_spatial_region_ids: Optional integer correlation-region
            identifier for every UT. Unequal IDs decorrelate supported random
            fields; correlation within a region uses 2D distance. IDs do not
            add floor loss or change pathloss or geometry. Blockage model A
            uses the same partition when enabled. Shape [num_ut] or
            [batch size, num_ut].
        """
        super().set_topology(
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
