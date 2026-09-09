#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Unit tests for sionna.sys.topology"""

import math

import numpy as np
import pytest
import torch

from sionna.phy import config
from sionna.phy.utils import flatten_dims
from sionna.phy.channel import tr38901
from sionna.sys.topology import (
    HexGrid,
    IndoorFactoryTopology,
    gen_tr38901_indoor_factory_topology,
    gen_tr38901_indoor_office_topology,
    gen_tr38901_multicell_topology,
)
from sys_utils import wraparound_dist_np


class TestHexagonalGrid:
    """Tests for the HexGrid class."""

    def test_hexagonal_grid(self, device, precision):
        """Checks that the centers are aligned with pre-computed ones."""
        grid = HexGrid(
            cell_radius=4,
            num_rings=3,
            center_loc=(-2, 3),
            precision=precision,
            device=device,
        )
        grid.cell_radius = 1
        grid.num_rings = 2
        grid.center_loc = (0, 0)

        centers_precomputed = np.array(
            [
                (0.0, 0.0),
                (-1.5, 0.8660254037844386),
                (0.0, 1.7320508075688772),
                (1.5, 0.8660254037844386),
                (1.5, -0.8660254037844386),
                (0.0, -1.7320508075688772),
                (-1.5, -0.8660254037844386),
                (-3.0, 1.7320508075688772),
                (-1.5, 2.598076211353316),
                (0.0, 3.4641016151377544),
                (1.5, 2.598076211353316),
                (3.0, 1.7320508075688772),
                (3.0, 0.0),
                (3.0, -1.7320508075688772),
                (1.5, -2.598076211353316),
                (0.0, -3.4641016151377544),
                (-1.5, -2.598076211353316),
                (-3.0, -1.7320508075688772),
                (-3.0, 0.0),
            ]
        )

        centers = grid.cell_loc.cpu().numpy()[:, :2]
        is_found = np.zeros(len(centers))

        for c in centers:
            dist_c_centers = np.linalg.norm(np.array([c]) - centers_precomputed, axis=1)
            closest_center = np.argmin(dist_c_centers)
            assert dist_c_centers[closest_center] < 1e-5, "Center not found among pre-computed ones"
            is_found[closest_center] = 1

        assert np.sum(is_found) == len(is_found), "Not all centers were found"

    @pytest.mark.parametrize(
        ("property_name", "value"),
        [
            ("center_loc", (1, 0)),
            ("cell_height", 8.0),
            ("isd", 24.0),
            ("cell_radius", 12.0),
        ],
    )
    def test_property_updates_refresh_mirror_cells(
        self, device, precision, property_name, value
    ):
        """Property mutations match an equivalent freshly constructed grid."""
        grid = HexGrid(
            num_rings=2,
            cell_radius=10.0,
            cell_height=2.0,
            center_loc=(0, 0),
            precision=precision,
            device=device,
        )
        setattr(grid, property_name, value)

        fresh_grid = HexGrid(
            num_rings=grid.num_rings,
            cell_radius=grid.cell_radius.item(),
            cell_height=grid.cell_height.item(),
            center_loc=grid.center_loc.tolist(),
            precision=precision,
            device=device,
        )

        torch.testing.assert_close(grid.cell_loc, fresh_grid.cell_loc)
        torch.testing.assert_close(
            grid.mirror_cell_loc, fresh_grid.mirror_cell_loc
        )

        point = torch.tensor(
            [7.0, -4.0, 1.5], dtype=grid.dtype, device=device
        )
        wraparound_dist = torch.linalg.vector_norm(
            grid.mirror_cell_loc - point, dim=-1
        ).amin(dim=-1)
        fresh_wraparound_dist = torch.linalg.vector_norm(
            fresh_grid.mirror_cell_loc - point, dim=-1
        ).amin(dim=-1)
        torch.testing.assert_close(wraparound_dist, fresh_wraparound_dist)

    def test_drop_uts(self, device, precision):
        """Validate UT locations from call method."""
        isd = 50
        bs_height = 10
        num_rings = 1

        grid = HexGrid(
            isd=isd,
            num_rings=num_rings,
            cell_height=bs_height,
            precision=precision,
            device=device,
        )

        num_ut_per_sector = 100

        min_bs_ut_dist_vec = [20, 20, 20]
        max_bs_ut_dist_vec = [30, 35, 40]
        min_ut_height_vec = [1, 9, 12]
        max_ut_height_vec = [2, 11, 15]

        assert len(np.unique([
            len(min_bs_ut_dist_vec),
            len(min_ut_height_vec),
            len(max_ut_height_vec),
        ])) == 1

        for ii in range(len(min_bs_ut_dist_vec)):
            # [batch_size, num_cells, 3, num_ut_per_sector, 3]
            ut_loc, *_ = grid(
                1,
                num_ut_per_sector,
                min_bs_ut_dist_vec[ii],
                max_bs_ut_dist=max_bs_ut_dist_vec[ii],
                min_ut_height=min_ut_height_vec[ii],
                max_ut_height=max_ut_height_vec[ii],
            )
            # [num_cells, num_ut_per_cell, 3]
            ut_loc = flatten_dims(ut_loc, num_dims=2, axis=2)[0, ::].cpu().numpy()

            cell_loc = grid.cell_loc.cpu().numpy()

            for cell in range(grid.num_cells):
                for ut in range(ut_loc.shape[1]):
                    ut_cell_dist_3d = np.linalg.norm(
                        cell_loc[cell, :] - ut_loc[cell, ut, :]
                    )
                    ut_cell_dist_2d = np.linalg.norm(
                        cell_loc[cell, :2] - ut_loc[cell, ut, :2]
                    )

                    # 2D UT-cell center distance must be at most ISD / sqrt(3)
                    assert ut_cell_dist_2d <= grid.isd.item() / np.sqrt(3), (
                        "2D distance exceeds ISD / sqrt(3)"
                    )

                    # 3D UT-cell center distance must be >= min_bs_ut_dist
                    assert ut_cell_dist_3d >= min_bs_ut_dist_vec[ii], (
                        "3D distance is less than min_bs_ut_dist"
                    )

                    # 3D UT-cell center distance must be <= max_bs_ut_dist
                    assert ut_cell_dist_3d <= max_bs_ut_dist_vec[ii], (
                        "3D distance exceeds max_bs_ut_dist"
                    )

    def test_wraparound(self, device, precision):
        """Validate wraparound method against its non-PyTorch version."""

        def drop_uts(isd, num_rings, batch_size, num_ut_per_sector,
                     min_bs_ut_dist, min_ut_height, max_ut_height):
            grid = HexGrid(isd=isd, num_rings=num_rings, precision=precision, device=device)
            ut_loc, cell_mirror_coord, wrap_dist_pt = grid(
                batch_size,
                num_ut_per_sector,
                min_bs_ut_dist,
                min_ut_height=min_ut_height,
                max_ut_height=max_ut_height,
            )
            return ut_loc, cell_mirror_coord, wrap_dist_pt, grid

        batch_size = 1
        num_ut_per_sector = 5
        min_bs_ut_dist = 20
        isd = 50
        min_ut_height = 1
        max_ut_height = 2
        num_rings = 1

        # Run drop_uts
        ut_loc, cell_mirror_coord, wrap_dist_pt, grid = drop_uts(
            isd, num_rings,
            batch_size, num_ut_per_sector,
            min_bs_ut_dist, min_ut_height, max_ut_height
        )

        # Flatten to [..., 3] for UT locations
        ut_loc = flatten_dims(ut_loc, num_dims=4, axis=0).cpu().numpy()
        # [..., num_cells, 3]
        cell_mirror_coord = flatten_dims(cell_mirror_coord, num_dims=4, axis=0).cpu().numpy()
        # [..., num_cells]
        wrap_dist_pt = flatten_dims(wrap_dist_pt, num_dims=4, axis=0).cpu().numpy()

        # Compare wraparound distance against the Numpy version
        for ut in range(ut_loc.shape[0]):
            wrap_dist_np_vec = wraparound_dist_np(grid, ut_loc[ut, :])
            for cell in range(grid.num_cells):
                wrap_dist_np1 = np.linalg.norm(
                    cell_mirror_coord[ut, cell, :] - ut_loc[ut, :]
                )
                assert abs(wrap_dist_np_vec[cell] - wrap_dist_np1) < 1e-5, (
                    f"Mismatch between numpy wraparound methods at cell {cell}"
                )
                assert abs(wrap_dist_np_vec[cell] - wrap_dist_pt[ut, cell]) < 1e-5, (
                    f"Mismatch between numpy and PyTorch at cell {cell}"
                )

    @pytest.mark.parametrize("mode", ["default", "reduce-overhead"])
    def test_compiled(self, device, precision, mode):
        """Test that HexGrid works with torch.compile."""
        if device == "cpu" and mode == "reduce-overhead":
            pytest.skip("reduce-overhead mode not well supported on CPU")

        isd = 50
        num_rings = 1
        min_bs_ut_dist = 20

        grid = HexGrid(isd=isd, num_rings=num_rings, precision=precision, device=device)

        @torch.compile(mode=mode, fullgraph=True, dynamic=True)
        def compiled_call(reference):
            return grid.call(
                reference.shape[0],
                reference.shape[1],
                min_bs_ut_dist,
            )

        for batch_size, num_ut_per_sector in ((2, 3), (3, 5)):
            reference = torch.empty(
                batch_size, num_ut_per_sector, device=device
            )
            ut_loc, cell_mirror_coord, wrap_dist = compiled_call(reference)

            assert ut_loc.shape == (
                batch_size, grid.num_cells, 3, num_ut_per_sector, 3
            )
            assert cell_mirror_coord.shape == (
                batch_size,
                grid.num_cells,
                3,
                num_ut_per_sector,
                grid.num_cells,
                3,
            )
            assert wrap_dist.shape == (
                batch_size,
                grid.num_cells,
                3,
                num_ut_per_sector,
                grid.num_cells,
            )


class TestTR38901MulticellTopology:
    """Tests for TR 38.901 multi-cell topology generation."""

    def test_shapes_and_site_ids(self, device, precision):
        topology = gen_tr38901_multicell_topology(
            "umi", batch_size=2, num_ut_per_sector=1,
            carrier_frequency=3.5e9,
            precision=precision, device=device
        )
        (
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
        ) = topology

        assert ut_loc.shape == (2, 57, 3)
        assert bs_loc.shape == (2, 57, 3)
        assert ut_orientations.shape == (2, 57, 3)
        assert bs_orientations.shape == (2, 57, 3)
        assert ut_velocities.shape == (2, 57, 3)
        assert in_state.shape == (2, 57)
        assert los is None
        assert bs_virtual_loc.shape == (2, 57, 57, 3)
        assert bs_site_ids.shape == (57,)
        assert bs_site_ids.dtype == torch.int64
        assert spatial_consistency_track_ids is None
        assert distance_2d_in.shape == (2, 57, 57)
        distance_2d_in_by_site = distance_2d_in.reshape(2, 19, 3, 57)
        torch.testing.assert_close(
            distance_2d_in_by_site[:, :, 0],
            distance_2d_in_by_site[:, :, 1],
        )
        torch.testing.assert_close(
            distance_2d_in_by_site[:, :, 0],
            distance_2d_in_by_site[:, :, 2],
        )
        outdoor_mask = (~in_state).unsqueeze(1).expand_as(distance_2d_in)
        assert torch.all(distance_2d_in[outdoor_mask] == 0.0)

        expected = torch.arange(19, dtype=torch.int64, device=device)
        expected = expected.repeat_interleave(3)
        torch.testing.assert_close(bs_site_ids, expected)

    def test_seed_reproducibility(self, device, precision):
        arguments = {
            "scenario": "umi",
            "batch_size": 2,
            "num_ut_per_sector": 3,
            "carrier_frequency": 6e9,
            "num_rings": 0,
            "precision": precision,
            "device": device,
        }
        config.seed = 2026
        first = gen_tr38901_multicell_topology(**arguments)
        config.seed = 2026
        second = gen_tr38901_multicell_topology(**arguments)

        for first_value, second_value in zip(first, second):
            if first_value is None:
                assert second_value is None
            else:
                assert torch.equal(first_value, second_value)

    def test_sector_drop_stays_in_voronoi_intersections(
        self, device, precision
    ):
        """Keep every sector drop inside its complete geometric support."""
        isd = 200.0
        num_ut_per_sector = 2048
        topology, site_positions = gen_tr38901_multicell_topology(
            "umi",
            batch_size=1,
            num_ut_per_sector=num_ut_per_sector,
            carrier_frequency=6e9,
            num_rings=1,
            isd=isd,
            min_bs_ut_dist=0.0,
            indoor_probability=0.0,
            return_site_positions=True,
            precision=precision,
            device=device,
        )
        num_sites = site_positions.shape[0]
        ut_xy = topology[0][0, :, :2].reshape(
            num_sites, 3, num_ut_per_sector, 2
        )
        relative_xy = ut_xy - site_positions[:, None, None, :]

        sector_yaws = torch.deg2rad(
            torch.tensor(
                [30.0, 150.0, 270.0],
                dtype=ut_xy.dtype,
                device=device,
            )
        )
        angle = torch.atan2(relative_xy[..., 1], relative_xy[..., 0])
        angle_delta = torch.atan2(
            torch.sin(angle - sector_yaws[None, :, None]),
            torch.cos(angle - sector_yaws[None, :, None]),
        )
        angle_tol = 2e-5
        assert torch.all(torch.abs(angle_delta) <= math.pi / 3.0 + angle_tol)

        normal_angles = torch.arange(6, dtype=ut_xy.dtype, device=device)
        normal_angles = normal_angles * math.pi / 3.0
        hex_normals = torch.stack(
            [torch.cos(normal_angles), torch.sin(normal_angles)], dim=-1
        )
        projections = torch.einsum(
            "...d,kd->...k", relative_xy, hex_normals
        )
        distance_tol = 1e-4 * isd
        assert torch.all(projections <= 0.5 * isd + distance_tol)

        # The sampled support must extend to both radial sector boundaries.
        assert torch.all(
            torch.amin(angle_delta, dim=-1) < -math.pi / 3.0 + 0.05
        )
        assert torch.all(
            torch.amax(angle_delta, dim=-1) > math.pi / 3.0 - 0.05
        )

        # It must also extend to both outer Voronoi edges in every sector.
        edge_angles = sector_yaws[:, None] + torch.tensor(
            [-math.pi / 6.0, math.pi / 6.0],
            dtype=ut_xy.dtype,
            device=device,
        )
        edge_normals = torch.stack(
            [torch.cos(edge_angles), torch.sin(edge_angles)], dim=-1
        )
        edge_projections = torch.sum(
            relative_xy.unsqueeze(-2) * edge_normals[None, :, None, :, :],
            dim=-1,
        )
        assert torch.all(
            torch.amax(edge_projections, dim=-2) > 0.5 * isd - 0.02 * isd
        )

    def test_sector_drop_is_uniform_by_area(self, device, precision):
        """Check triangle weights and area moments without distance rejection."""
        isd = 200.0
        num_ut_per_sector = 12000
        topology, site_positions = gen_tr38901_multicell_topology(
            "umi",
            batch_size=1,
            num_ut_per_sector=num_ut_per_sector,
            carrier_frequency=6e9,
            num_rings=0,
            isd=isd,
            min_bs_ut_dist=0.0,
            indoor_probability=0.0,
            return_site_positions=True,
            precision=precision,
            device=device,
        )
        relative_xy = topology[0][0, :, :2].reshape(
            3, num_ut_per_sector, 2
        ) - site_positions[0]
        sector_yaws = torch.deg2rad(
            torch.tensor(
                [30.0, 150.0, 270.0],
                dtype=relative_xy.dtype,
                device=device,
            )
        )
        forward_direction = torch.stack(
            [torch.cos(sector_yaws), torch.sin(sector_yaws)], dim=-1
        )
        lateral_direction = torch.stack(
            [-torch.sin(sector_yaws), torch.cos(sector_yaws)], dim=-1
        )
        forward = torch.sum(
            relative_xy * forward_direction[:, None, :], dim=-1
        )
        lateral = torch.sum(
            relative_xy * lateral_direction[:, None, :], dim=-1
        )

        # The two component triangles have equal area.
        triangle_fraction = torch.mean(
            (lateral >= 0.0).to(relative_xy.dtype), dim=-1
        )
        torch.testing.assert_close(
            triangle_fraction,
            torch.full_like(triangle_fraction, 0.5),
            rtol=0.0,
            atol=0.02,
        )

        cell_radius = isd / math.sqrt(3.0)
        torch.testing.assert_close(
            torch.mean(forward, dim=-1),
            torch.full(
                (3,),
                0.5 * cell_radius,
                dtype=relative_xy.dtype,
                device=device,
            ),
            rtol=0.0,
            atol=0.01 * cell_radius,
        )
        torch.testing.assert_close(
            torch.mean(lateral, dim=-1),
            torch.zeros(3, dtype=relative_xy.dtype, device=device),
            rtol=0.0,
            atol=0.01 * cell_radius,
        )
        mean_radius_squared = torch.mean(
            torch.sum(relative_xy**2, dim=-1), dim=-1
        )
        torch.testing.assert_close(
            mean_radius_squared,
            torch.full_like(
                mean_radius_squared, 5.0 * cell_radius**2 / 12.0
            ),
            rtol=0.02,
            atol=0.0,
        )

    @pytest.mark.parametrize(
        "scenario,minimum_distance", [("umi", 10.0), ("uma", 35.0)]
    )
    def test_standard_outdoor_d2d_boundary(
        self, device, precision, scenario, minimum_distance
    ):
        """Use the public UMi/UMa minimum for outdoor d2D."""
        topology, site_positions = gen_tr38901_multicell_topology(
            scenario,
            batch_size=1,
            num_ut_per_sector=4096,
            carrier_frequency=6e9,
            num_rings=0,
            indoor_probability=0.0,
            return_site_positions=True,
            precision=precision,
            device=device,
        )
        distance_2d = torch.linalg.norm(
            topology[0][0, :, :2] - site_positions[0], dim=-1
        )
        assert torch.all(distance_2d >= minimum_distance - 1e-4)
        assert torch.any(distance_2d < minimum_distance + 2.0)

    @pytest.mark.parametrize(
        "scenario,minimum_distance", [("umi", 10.0), ("uma", 35.0)]
    )
    def test_standard_indoor_d2d_out_boundary(
        self, device, precision, scenario, minimum_distance
    ):
        """Use the public UMi/UMa minimum for indoor outdoor-link d2D."""
        topology, site_positions = gen_tr38901_multicell_topology(
            scenario,
            batch_size=1,
            num_ut_per_sector=4096,
            carrier_frequency=6e9,
            num_rings=0,
            indoor_probability=1.0,
            return_site_positions=True,
            precision=precision,
            device=device,
        )
        distance_2d = torch.linalg.norm(
            topology[0][0, :, :2] - site_positions[0], dim=-1
        )
        distance_2d_out = distance_2d - topology[-1][0]
        assert torch.all(distance_2d_out >= minimum_distance - 1e-4)
        assert torch.any(distance_2d_out < minimum_distance + 2.0)

    def test_carrier_frequency_selects_indoor_distance_model(
        self, device, precision
    ):
        """Use the below-6-GHz compatibility model only where specified."""

        config.seed = 1234
        low_frequency = gen_tr38901_multicell_topology(
            "umi",
            batch_size=1,
            num_ut_per_sector=1,
            carrier_frequency=0.5e9,
            num_rings=0,
            indoor_probability=1.0,
            precision=precision,
            device=device,
        )
        config.seed = 1234
        high_frequency = gen_tr38901_multicell_topology(
            "umi",
            batch_size=1,
            num_ut_per_sector=1,
            carrier_frequency=100e9,
            num_rings=0,
            indoor_probability=1.0,
            precision=precision,
            device=device,
        )

        low_distance = low_frequency[-1][:, 0]
        high_distance = high_frequency[-1]
        assert torch.all(high_distance <= low_distance)
        assert torch.any(high_distance < low_distance)

    def test_sector_yaws_follow_table_78(self, device, precision):
        topology = gen_tr38901_multicell_topology(
            "uma", batch_size=1, num_ut_per_sector=1,
            carrier_frequency=3.5e9,
            precision=precision, device=device
        )
        bs_orientations = topology[3]
        expected = torch.deg2rad(
            torch.tensor([30.0, 150.0, 270.0],
                         dtype=bs_orientations.dtype, device=device)
        )
        torch.testing.assert_close(bs_orientations[0, :3, 0], expected)

    def test_indoor_height_model(self, device, precision):
        topology = gen_tr38901_multicell_topology(
            "umi",
            batch_size=1,
            num_ut_per_sector=1,
            carrier_frequency=3.5e9,
            indoor_probability=1.0,
            precision=precision,
            device=device,
        )
        ut_loc = topology[0]
        in_state = topology[5]
        assert torch.all(in_state)
        assert torch.all(ut_loc[..., 2] >= 1.5)
        assert torch.all(ut_loc[..., 2] <= 22.5)

    @pytest.mark.parametrize(
        "scenario,max_indoor_distance", [("umi", 25.0), ("rma", 10.0)]
    )
    def test_returns_indoor_distance_used_for_placement(
        self, device, precision, scenario, max_indoor_distance
    ):
        """Return the same indoor distance included in drop rejection."""
        min_distance = 10.0
        topology, site_positions = gen_tr38901_multicell_topology(
            scenario,
            batch_size=2,
            num_ut_per_sector=2,
            carrier_frequency=3.5e9,
            num_rings=0,
            min_bs_ut_dist=min_distance,
            indoor_probability=1.0,
            return_site_positions=True,
            precision=precision,
            device=device,
        )
        ut_loc, bs_loc = topology[:2]
        indoor_distance = topology[-1]

        assert site_positions.shape == (1, 2)
        expected_shape = (2, 3, 6) if scenario == "umi" else (2, 6)
        assert indoor_distance.shape == expected_shape
        assert torch.all(indoor_distance >= 0.0)
        assert torch.all(indoor_distance <= max_indoor_distance)
        home_indoor_distance = (
            indoor_distance[:, 0] if indoor_distance.dim() == 3
            else indoor_distance
        )
        serving_distance = torch.linalg.norm(
            ut_loc[..., :2] - bs_loc[:, :1, :2], dim=-1
        )
        assert torch.all(
            serving_distance >= min_distance + home_indoor_distance - 1e-5
        )

    @pytest.mark.parametrize("scenario", ["umi", "uma"])
    def test_below_6ghz_indoor_distance_is_link_specific_by_site(
        self, device, precision, scenario
    ):
        """Return one legacy indoor distance for every site-UT link."""
        num_ut_per_sector = 2
        min_distance = 10.0
        topology, site_positions = gen_tr38901_multicell_topology(
            scenario,
            batch_size=1,
            num_ut_per_sector=num_ut_per_sector,
            carrier_frequency=3.5e9,
            num_rings=1,
            min_bs_ut_dist=min_distance,
            indoor_probability=1.0,
            return_site_positions=True,
            precision=precision,
            device=device,
        )
        ut_loc = topology[0]
        distance_2d_in = topology[-1]
        num_sites = site_positions.shape[0]
        num_ut = ut_loc.shape[1]
        assert distance_2d_in.shape == (1, 3*num_sites, num_ut)

        distance_2d_in_by_site = distance_2d_in.reshape(
            1, num_sites, 3, num_ut
        )
        torch.testing.assert_close(
            distance_2d_in_by_site[:, :, 0],
            distance_2d_in_by_site[:, :, 1],
        )
        torch.testing.assert_close(
            distance_2d_in_by_site[:, :, 0],
            distance_2d_in_by_site[:, :, 2],
        )
        assert torch.any(
            distance_2d_in_by_site[:, 0, 0]
            != distance_2d_in_by_site[:, 1, 0]
        )

        home_site_ids = torch.arange(
            num_sites, dtype=torch.int64, device=device
        ).repeat_interleave(3*num_ut_per_sector)
        ut_indices = torch.arange(num_ut, dtype=torch.int64, device=device)
        home_indoor_distance = distance_2d_in[
            0, 3*home_site_ids, ut_indices
        ]
        home_link_distance = torch.linalg.norm(
            ut_loc[0, :, :2] - site_positions[home_site_ids], dim=-1
        )
        assert torch.all(
            home_link_distance
            >= min_distance + home_indoor_distance - 1e-5
        )

    def test_topology_tuple_preserves_indoor_distance_in_channel(
        self, device, precision
    ):
        """Pass the helper output directly without resampling its distance."""
        carrier_frequency = 3.5e9
        topology = gen_tr38901_multicell_topology(
            "umi",
            batch_size=1,
            num_ut_per_sector=1,
            carrier_frequency=carrier_frequency,
            num_rings=1,
            indoor_probability=1.0,
            precision=precision,
            device=device,
        )
        ut_array = tr38901.PanelArray(
            num_rows_per_panel=1,
            num_cols_per_panel=1,
            polarization="single",
            polarization_type="V",
            antenna_pattern="omni",
            carrier_frequency=carrier_frequency,
            precision=precision,
            device=device,
        )
        bs_array = tr38901.PanelArray(
            num_rows_per_panel=1,
            num_cols_per_panel=1,
            polarization="single",
            polarization_type="V",
            antenna_pattern="omni",
            carrier_frequency=carrier_frequency,
            precision=precision,
            device=device,
        )
        channel = tr38901.UMi(
            carrier_frequency,
            "low",
            ut_array,
            bs_array,
            "downlink",
            precision=precision,
            device=device,
        )

        channel.set_topology(*topology)

        torch.testing.assert_close(
            channel._scenario.distance_2d_in, topology[-1]
        )

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"scenario": "invalid"},
            {"batch_size": 0},
            {"num_ut_per_sector": 0},
            {"num_rings": 3},
            {"carrier_frequency": 0.0},
            {"isd": 0.0},
            {"bs_height": 0.0},
            {"min_bs_ut_dist": -1.0},
            {"indoor_probability": 1.1},
        ],
    )
    def test_rejects_invalid_arguments(self, kwargs):
        arguments = {
            "scenario": "umi",
            "batch_size": 1,
            "num_ut_per_sector": 1,
            "carrier_frequency": 3.5e9,
        }
        arguments.update(kwargs)
        with pytest.raises(ValueError):
            gen_tr38901_multicell_topology(**arguments)

    def test_rma_defaults_and_zero_mobility_outputs(self, device, precision):
        topology = gen_tr38901_multicell_topology(
            "rma",
            batch_size=1,
            num_ut_per_sector=1,
            carrier_frequency=3.5e9,
            precision=precision,
            device=device,
        )
        ut_loc, bs_loc, ut_orientations, _, ut_velocities = topology[:5]

        assert ut_loc.shape == (1, 57, 3)
        torch.testing.assert_close(
            bs_loc[..., 2],
            torch.full_like(bs_loc[..., 2], 35.0),
        )
        torch.testing.assert_close(
            ut_orientations,
            torch.zeros_like(ut_orientations),
        )
        torch.testing.assert_close(ut_velocities, torch.zeros_like(ut_velocities))
        torch.testing.assert_close(
            ut_loc[..., 2], torch.full_like(ut_loc[..., 2], 1.5)
        )

        with pytest.raises(ValueError, match="fixed to 1.5 m"):
            gen_tr38901_multicell_topology(
                "rma",
                batch_size=1,
                num_ut_per_sector=1,
                carrier_frequency=3.5e9,
                apply_tr36873_indoor_heights=True,
                precision=precision,
                device=device,
            )


class TestTR38901IndoorTopologies:
    """Tests for TR 38.901 indoor topology helpers."""

    def test_indoor_office_defaults(self, device, precision):
        topology, site_positions = gen_tr38901_indoor_office_topology(
            batch_size=2,
            num_ut_per_sector=1,
            return_site_positions=True,
            precision=precision,
            device=device,
        )
        (
            ut_loc,
            bs_loc,
            ut_orientations,
            bs_orientations,
            ut_velocities,
            in_state,
            los,
            bs_virtual_loc,
            bs_site_ids,
        ) = topology

        assert site_positions.shape == (12, 2)
        assert ut_loc.shape == (2, 36, 3)
        assert bs_loc.shape == (2, 36, 3)
        assert ut_orientations.shape == (2, 36, 3)
        assert bs_orientations.shape == (2, 36, 3)
        assert ut_velocities.shape == (2, 36, 3)
        assert in_state.shape == (2, 36)
        assert los is None
        assert bs_virtual_loc.shape == (2, 36, 36, 3)
        assert bs_site_ids.shape == (36,)
        assert torch.all(in_state)
        torch.testing.assert_close(ut_loc[..., 2], torch.ones_like(ut_loc[..., 2]))
        assert torch.all((ut_loc[..., 0] >= 0.0) & (ut_loc[..., 0] <= 120.0))
        assert torch.all((ut_loc[..., 1] >= 0.0) & (ut_loc[..., 1] <= 50.0))

        expected_sites = torch.tensor(
            [[10.0, 15.0], [30.0, 15.0], [50.0, 15.0]],
            dtype=site_positions.dtype,
            device=device,
        )
        torch.testing.assert_close(site_positions[:3], expected_sites)
        expected_yaws = torch.deg2rad(
            torch.tensor([30.0, 150.0, 270.0],
                         dtype=bs_orientations.dtype, device=device)
        )
        torch.testing.assert_close(bs_orientations[0, :3, 0], expected_yaws)
        expected_ids = torch.arange(12, dtype=torch.int64, device=device)
        expected_ids = expected_ids.repeat_interleave(3)
        torch.testing.assert_close(bs_site_ids, expected_ids)
        torch.testing.assert_close(bs_virtual_loc[:, :, 0, :], bs_loc)

    @pytest.mark.parametrize(
        "factory_scenario,hall_length,hall_width,bs_spacing,bs_height",
        [
            ("SL", 120.0, 60.0, 20.0, 1.5),
            ("DL", 300.0, 150.0, 50.0, 1.5),
            ("SH", 300.0, 150.0, 50.0, 8.0),
            ("DH", 120.0, 60.0, 20.0, 8.0),
        ],
    )
    def test_indoor_factory_defaults(
        self, device, precision, factory_scenario, hall_length, hall_width,
        bs_spacing, bs_height
    ):
        topology, site_positions = gen_tr38901_indoor_factory_topology(
            factory_scenario,
            batch_size=2,
            num_ut=30,
            return_site_positions=True,
            precision=precision,
            device=device,
        )
        (
            ut_loc,
            bs_loc,
            ut_orientations,
            bs_orientations,
            ut_velocities,
            in_state,
            los,
            bs_virtual_loc,
            bs_site_ids,
        ) = topology

        assert site_positions.shape == (18, 2)
        assert ut_loc.shape == (2, 30, 3)
        assert bs_loc.shape == (2, 18, 3)
        assert ut_orientations.shape == (2, 30, 3)
        assert bs_orientations.shape == (2, 18, 3)
        assert ut_velocities.shape == (2, 30, 3)
        assert in_state.shape == (2, 30)
        assert los is None
        assert bs_virtual_loc.shape == (2, 18, 30, 3)
        assert bs_site_ids.shape == (18,)
        assert torch.all(in_state)
        torch.testing.assert_close(ut_loc[..., 2], 1.5*torch.ones_like(ut_loc[..., 2]))
        torch.testing.assert_close(bs_loc[..., 2], bs_height*torch.ones_like(bs_loc[..., 2]))
        assert torch.all((ut_loc[..., 0] >= 0.0) & (ut_loc[..., 0] <= hall_length))
        assert torch.all((ut_loc[..., 1] >= 0.0) & (ut_loc[..., 1] <= hall_width))

        expected_first = torch.tensor(
            [0.5*bs_spacing, 0.5*bs_spacing],
            dtype=site_positions.dtype,
            device=device,
        )
        expected_last = torch.tensor(
            [hall_length - 0.5*bs_spacing, hall_width - 0.5*bs_spacing],
            dtype=site_positions.dtype,
            device=device,
        )
        torch.testing.assert_close(site_positions[0], expected_first)
        torch.testing.assert_close(site_positions[-1], expected_last)
        torch.testing.assert_close(
            bs_site_ids, torch.arange(18, dtype=torch.int64, device=device)
        )
        torch.testing.assert_close(bs_virtual_loc[:, :, 0, :], bs_loc)

        distances = torch.linalg.norm(
            ut_loc[:, :, :2].unsqueeze(2) - site_positions.reshape(1, 1, 18, 2),
            dim=-1,
        )
        assert torch.all(distances.min(dim=-1).values >= 1.0)

    def test_indoor_factory_aliases(self, device, precision):
        _topology, site_positions = gen_tr38901_indoor_factory_topology(
            "InF-SH",
            batch_size=1,
            num_ut=2,
            return_site_positions=True,
            precision=precision,
            device=device,
        )
        assert site_positions.shape == (18, 2)

    def test_indoor_factory_result_metadata(self, device, precision):
        topology = gen_tr38901_indoor_factory_topology(
            "InF-SH",
            batch_size=1,
            num_ut=2,
            hall_length=180.0,
            hall_width=90.0,
            hall_height=12.0,
            bs_spacing=30.0,
            precision=precision,
            device=device,
        )

        assert isinstance(topology, tuple)
        assert isinstance(topology, IndoorFactoryTopology)
        assert topology.factory_scenario == "SH"
        assert topology.hall_dimensions == (180.0, 90.0, 12.0)

        with pytest.warns(UserWarning, match="hall-dimension validation"):
            unpacked = tuple(topology)
        assert len(unpacked) == 9
        assert unpacked[0] is topology[0]

    def test_indoor_topology_validation(self, device, precision):
        with pytest.raises(ValueError, match="num_ut_per_sector|num_ut"):
            gen_tr38901_indoor_office_topology(
                batch_size=1,
                num_ut_per_sector=0,
                precision=precision,
                device=device,
            )

        with pytest.raises(ValueError, match="spacing"):
            gen_tr38901_indoor_factory_topology(
                "SH",
                batch_size=1,
                num_ut=1,
                bs_spacing=0.0,
                precision=precision,
                device=device,
            )

        with pytest.raises(ValueError, match="InF-HH"):
            gen_tr38901_indoor_factory_topology(
                "HH",
                batch_size=1,
                num_ut=1,
                precision=precision,
                device=device,
            )

        with pytest.raises(ValueError, match="hall_height"):
            gen_tr38901_indoor_factory_topology(
                "SH",
                batch_size=1,
                num_ut=1,
                hall_height=0.0,
                precision=precision,
                device=device,
            )

    def test_indoor_helpers_work_with_channels(self, device, precision):
        carrier_frequency = 3.5e9
        ut_array = tr38901.PanelArray(
            num_rows_per_panel=1,
            num_cols_per_panel=1,
            polarization="single",
            polarization_type="V",
            antenna_pattern="omni",
            carrier_frequency=carrier_frequency,
            precision=precision,
            device=device,
        )
        bs_array = tr38901.PanelArray(
            num_rows_per_panel=1,
            num_cols_per_panel=1,
            polarization="single",
            polarization_type="V",
            antenna_pattern="omni",
            carrier_frequency=carrier_frequency,
            precision=precision,
            device=device,
        )

        inh = tr38901.InH(
            carrier_frequency,
            ut_array,
            bs_array,
            "downlink",
            precision=precision,
            device=device,
        )
        inh_topology = gen_tr38901_indoor_office_topology(
            batch_size=1,
            num_ut_per_sector=1,
            precision=precision,
            device=device,
        )
        inh.set_topology(*inh_topology)
        assert inh._scenario.num_bs == 36

        inf = tr38901.InF(
            carrier_frequency,
            ut_array,
            bs_array,
            "downlink",
            factory_scenario="SH",
            precision=precision,
            device=device,
        )
        inf_topology = gen_tr38901_indoor_factory_topology(
            "SH",
            batch_size=1,
            num_ut=4,
            precision=precision,
            device=device,
        )
        inf.set_topology(*inf_topology)
        assert inf._scenario.num_bs == 18

        dl_topology = gen_tr38901_indoor_factory_topology(
            "DL",
            batch_size=1,
            num_ut=4,
            precision=precision,
            device=device,
        )
        with pytest.raises(ValueError, match="factory scenario"):
            dl_topology.set_topology(inf)

        custom_inf_topology = gen_tr38901_indoor_factory_topology(
            "SH",
            batch_size=1,
            num_ut=4,
            hall_length=180.0,
            hall_width=90.0,
            hall_height=12.0,
            bs_spacing=30.0,
            precision=precision,
            device=device,
        )
        with pytest.raises(ValueError, match="hall_dimensions"):
            custom_inf_topology.set_topology(inf)

        custom_inf = tr38901.InF(
            carrier_frequency,
            ut_array,
            bs_array,
            "downlink",
            factory_scenario=custom_inf_topology.factory_scenario,
            hall_dimensions=custom_inf_topology.hall_dimensions,
            precision=precision,
            device=device,
        )
        custom_inf_topology.set_topology(custom_inf)
        assert custom_inf._scenario.num_bs == 18
