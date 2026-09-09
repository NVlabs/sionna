#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Tests for 3GPP TR 38.901 spatial consistency."""

import pytest
import torch

from sionna.phy.channel import gen_single_sector_topology, tr38901
from sionna.phy.channel.tr38901 import (
    ChannelCoefficientsGenerator,
    LSP,
    LSPGenerator,
    Rays,
    RaysGenerator,
    Topology,
    spatial_consistency_correlation_matrix,
    spatial_consistency_matrix_sqrt,
)


def _dtype(precision):
    return torch.float32 if precision == "single" else torch.float64


def _arrays(carrier_frequency, precision, device):
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
    return ut_array, bs_array


def _umi_scenario(batch_size, num_bs, num_ut, precision, device, los=True):
    carrier_frequency = 3.5e9
    dtype = _dtype(precision)
    ut_array, bs_array = _arrays(carrier_frequency, precision, device)
    scenario = tr38901.UMiScenario(
        carrier_frequency,
        "low",
        ut_array,
        bs_array,
        "downlink",
        precision=precision,
        device=device,
    )
    ut_x = 100.0 + 6.0 * torch.arange(num_ut, dtype=dtype, device=device)
    ut_loc = torch.stack(
        [
            ut_x,
            torch.zeros_like(ut_x),
            torch.full_like(ut_x, 1.5),
        ],
        dim=-1,
    ).reshape(1, num_ut, 3).expand(batch_size, -1, -1).clone()
    bs_loc_one = torch.tensor(
        [[0.0, 0.0, 10.0]], dtype=dtype, device=device
    ).expand(num_bs, -1)
    bs_loc = bs_loc_one.reshape(1, num_bs, 3).expand(batch_size, -1, -1).clone()
    scenario.set_topology(
        ut_loc=ut_loc,
        bs_loc=bs_loc,
        ut_orientations=torch.zeros(batch_size, num_ut, 3, dtype=dtype, device=device),
        bs_orientations=torch.zeros(batch_size, num_bs, 3, dtype=dtype, device=device),
        ut_velocities=torch.zeros(batch_size, num_ut, 3, dtype=dtype, device=device),
        in_state=torch.zeros(batch_size, num_ut, dtype=torch.bool, device=device),
        los=los,
    )
    return scenario


def _constant_lsp(batch_size, num_bs, num_ut, dtype, device):
    shape = (batch_size, num_bs, num_ut)
    return LSP(
        ds=torch.full(shape, 1e-7, dtype=dtype, device=device),
        asd=torch.full(shape, 15.0, dtype=dtype, device=device),
        asa=torch.full(shape, 20.0, dtype=dtype, device=device),
        sf=torch.ones(shape, dtype=dtype, device=device),
        k_factor=torch.full(shape, 10.0, dtype=dtype, device=device),
        zsa=torch.full(shape, 8.0, dtype=dtype, device=device),
        zsd=torch.full(shape, 5.0, dtype=dtype, device=device),
    )


def _corrcoef(samples):
    samples = samples - samples.mean(dim=0, keepdim=True)
    cov = samples.transpose(0, 1) @ samples / (samples.shape[0] - 1)
    std = torch.sqrt(torch.diagonal(cov).clamp_min(1e-30))
    return cov / (std[:, None] * std[None, :])


class TestSpatialConsistencyUtilities:
    """Tests for standalone spatial-consistency helpers."""

    def test_correlation_matrix_uses_exponential_law_and_state_mask(
        self, device, precision
    ):
        """Check TR 38.901 exponential correlation and state masking."""
        dtype = _dtype(precision)
        distance = torch.tensor(
            [[[0.0, 5.0, 10.0], [5.0, 0.0, 5.0], [10.0, 5.0, 0.0]]],
            dtype=dtype,
            device=device,
        )
        states = torch.tensor([[0, 0, 1]], device=device)
        corr = spatial_consistency_correlation_matrix(
            distance,
            10.0,
            states=states,
            precision=precision,
            device=device,
        )
        expected = torch.exp(-distance / 10.0)
        expected[:, 0, 2] = 0.0
        expected[:, 2, 0] = 0.0
        expected[:, 1, 2] = 0.0
        expected[:, 2, 1] = 0.0
        expected[:, 2, 2] = 1.0
        torch.testing.assert_close(corr, expected)

    def test_per_terminal_correlation_distance_is_symmetric(
        self, device, precision
    ):
        """Check pairwise decorrelation distances for vector inputs."""
        dtype = _dtype(precision)
        distance = torch.tensor(
            [[[0.0, 5.0], [5.0, 0.0]]],
            dtype=dtype,
            device=device,
        )
        expected = torch.exp(
            -distance / torch.sqrt(torch.tensor(400.0, dtype=dtype, device=device))
        )
        for values in ([10.0, 40.0], [[10.0, 40.0]]):
            decorrelation_distance = torch.tensor(
                values, dtype=dtype, device=device
            )
            corr = spatial_consistency_correlation_matrix(
                distance,
                decorrelation_distance,
                correlation_distance_layout="per_terminal",
                precision=precision,
                device=device,
            )
            torch.testing.assert_close(corr, expected)
            torch.testing.assert_close(corr, corr.transpose(-1, -2))

    def test_matrix_sqrt_reconstructs_correlation(self, device, precision):
        """Check matrix square root for a well-conditioned matrix."""
        dtype = _dtype(precision)
        distance = torch.tensor(
            [[[0.0, 3.0, 9.0], [3.0, 0.0, 6.0], [9.0, 6.0, 0.0]]],
            dtype=dtype,
            device=device,
        )
        corr = spatial_consistency_correlation_matrix(
            distance, 12.0, precision=precision, device=device
        )
        sqrt = spatial_consistency_matrix_sqrt(
            corr, jitter=0.0, precision=precision, device=device
        )
        torch.testing.assert_close(sqrt @ sqrt.transpose(-1, -2), corr)

    def test_matrix_sqrt_preserves_duplicate_points(self, device, precision):
        """Check co-located terminals get identical random-field rows."""
        dtype = _dtype(precision)
        distance = torch.tensor(
            [[[0.0, 0.0, 5.0], [0.0, 0.0, 5.0], [5.0, 5.0, 0.0]]],
            dtype=dtype,
            device=device,
        )
        corr = spatial_consistency_correlation_matrix(
            distance, 10.0, precision=precision, device=device
        )
        sqrt = spatial_consistency_matrix_sqrt(
            corr, precision=precision, device=device
        )
        torch.testing.assert_close(sqrt[..., 0, :], sqrt[..., 1, :])
        torch.testing.assert_close(sqrt @ sqrt.transpose(-1, -2), corr)

    def test_pairwise_correlation_distance_is_not_misread_as_per_terminal(
        self, device, precision
    ):
        """Check an ``N x N`` distance tensor retains broadcast semantics."""
        dtype = _dtype(precision)
        distance = torch.tensor(
            [[0.0, 2.0], [2.0, 0.0]], dtype=dtype, device=device
        )
        correlation_distance = torch.tensor(
            [[10.0, 20.0], [20.0, 10.0]], dtype=dtype, device=device
        )
        correlation = spatial_consistency_correlation_matrix(
            distance,
            correlation_distance,
            precision=precision,
            device=device,
        )
        expected = torch.exp(-distance / correlation_distance)
        torch.testing.assert_close(correlation, expected)

    def test_integer_inputs_are_promoted(self, device, precision):
        """Check integer distances use the configured real precision."""
        distance = torch.tensor([[0, 2], [2, 0]], device=device)
        correlation = spatial_consistency_correlation_matrix(
            distance, 10, precision=precision, device=device
        )
        assert correlation.dtype == _dtype(precision)
        assert torch.all(torch.isfinite(correlation))

    def test_compiled_singular_matrix_without_jitter(
        self, device, precision
    ):
        """Check compiled factorization preserves exact duplicate points."""
        dtype = _dtype(precision)
        positions = torch.tensor(
            [0.0, 5.0, 5.0, 10.0], dtype=dtype, device=device
        )
        distance = torch.abs(positions.unsqueeze(-1) - positions.unsqueeze(-2))
        corr = torch.exp(-distance / 10.0).unsqueeze(0)

        def factorize(matrix):
            return spatial_consistency_matrix_sqrt(
                matrix,
                precision=precision,
                device=device,
            )

        factor = torch.compile(
            factorize, fullgraph=True, mode="reduce-overhead"
        )(corr)

        assert torch.all(torch.isfinite(factor))
        torch.testing.assert_close(factor[..., 1, :], factor[..., 2, :])
        torch.testing.assert_close(factor @ factor.transpose(-1, -2), corr)

    def test_compiled_matrix_sqrt_preserves_small_positive_mode(
        self, device, precision
    ):
        """Check compilation does not discard a representable PSD mode."""
        dtype = _dtype(precision)
        one = torch.tensor(1.0, dtype=dtype, device=device)
        rho = one - 2.0 * torch.finfo(dtype).eps
        corr = torch.stack(
            [torch.stack([one, rho]), torch.stack([rho, one])]
        ).unsqueeze(0)

        def factorize(matrix):
            return spatial_consistency_matrix_sqrt(
                matrix,
                precision=precision,
                device=device,
            )

        eager_factor = factorize(corr)
        compiled_factor = torch.compile(
            factorize, fullgraph=True, mode="reduce-overhead"
        )(corr)

        assert compiled_factor[..., 1, 1].item() > 0.0
        torch.testing.assert_close(
            eager_factor @ eager_factor.transpose(-1, -2), corr
        )
        torch.testing.assert_close(
            compiled_factor @ compiled_factor.transpose(-1, -2), corr
        )

    def test_matrix_sqrt_clamps_roundoff_sized_negative_modes(
        self, device, precision
    ):
        """Check numerical PSD roundoff is tolerated by the eigensolver path."""
        dtype = _dtype(precision)
        negative_mode = 2e-5 if precision == "single" else 2e-12
        diagonal = torch.tensor(
            [1.0, 0.5, 0.25, -negative_mode],
            dtype=dtype,
            device=device,
        )
        corr = torch.diag(diagonal)
        sqrt = spatial_consistency_matrix_sqrt(
            corr, precision=precision, device=device
        )
        assert torch.all(torch.isfinite(sqrt))
        expected = torch.diag(torch.clamp(diagonal, min=0.0))
        torch.testing.assert_close(sqrt @ sqrt.transpose(-1, -2), expected)


class TestRaysSpatialConsistency:
    """Tests for spatial consistency in the 3GPP ray generator."""

    def test_cluster_random_fields_follow_delay_sort(self, device, precision):
        """Check cluster-indexed spatial fields preserve raw-cluster identity."""
        batch_size = 2
        num_bs = 1
        num_ut = 3
        scenario = _umi_scenario(batch_size, num_bs, num_ut, precision, device)
        ray_sampler = RaysGenerator(scenario, enable_spatial_consistency=True)
        num_clusters = scenario.num_clusters_max
        dtype = _dtype(precision)

        order = torch.arange(num_clusters - 1, -1, -1, device=device)
        order = order.reshape(1, 1, 1, num_clusters)
        cluster_sort_indices = order.expand(
            batch_size, num_bs, num_ut, num_clusters
        )
        samples = torch.arange(
            batch_size*num_bs*num_ut*num_clusters*2,
            dtype=dtype,
            device=device,
        ).reshape(batch_size, num_bs, num_ut, num_clusters, 2)

        sorted_samples = ray_sampler._sort_clusters_by_delay(
            samples, cluster_sort_indices
        )
        expected = torch.gather(
            samples,
            dim=3,
            index=cluster_sort_indices.unsqueeze(-1).expand_as(samples),
        )
        torch.testing.assert_close(sorted_samples, expected)

    def test_all_downstream_cluster_randomness_follows_delay_sort(
        self, device, precision, monkeypatch
    ):
        """Check every cluster-indexed random field uses the delay ordering."""

        batch_size = 1
        num_bs = 1
        num_ut = 3
        scenario = _umi_scenario(
            batch_size, num_bs, num_ut, precision, device, los=True
        )
        ray_sampler = RaysGenerator(scenario, enable_spatial_consistency=True)
        ray_sampler.topology_updated_callback()
        original_sort = ray_sampler._sort_clusters_by_delay
        sorted_ranks = []

        def tracked_sort(samples, cluster_sort_indices=None):
            sorted_samples = original_sort(samples, cluster_sort_indices)
            assert cluster_sort_indices is not None
            indices = cluster_sort_indices
            while indices.dim() < samples.dim():
                indices = indices.unsqueeze(-1)
            expected = torch.gather(
                samples,
                dim=3,
                index=indices.expand(*samples.shape),
            )
            torch.testing.assert_close(sorted_samples, expected)
            sorted_ranks.append(samples.dim())
            return sorted_samples

        monkeypatch.setattr(ray_sampler, "_sort_clusters_by_delay", tracked_sort)
        lsp = _constant_lsp(
            batch_size, num_bs, num_ut, _dtype(precision), device
        )
        rays = ray_sampler(lsp)

        identity = torch.arange(
            scenario.num_clusters_max, device=device
        ).reshape(1, 1, 1, -1)
        assert rays.cluster_sort_indices is not None
        assert not torch.equal(
            rays.cluster_sort_indices,
            identity.expand_as(rays.cluster_sort_indices),
        )
        assert sorted_ranks.count(4) == 9
        assert sorted_ranks.count(5) == 5
        assert sorted_ranks.count(6) == 1
        assert len(sorted_ranks) == 15
        assert rays.phases is not None

    def test_spatial_normal_matches_configured_correlation(self, device, precision):
        """Check empirical correlation of generated normal random fields."""
        batch_size = 12000
        num_ut = 4
        scenario = _umi_scenario(batch_size, 1, num_ut, precision, device, los=True)
        ray_sampler = RaysGenerator(scenario, enable_spatial_consistency=True)
        ray_sampler.topology_updated_callback()

        samples = ray_sampler._spatial_normal((batch_size, 1, num_ut, 1))
        samples = samples[:, 0, :, 0]
        measured = _corrcoef(samples)
        distance = scenario.matrix_ut_distance_2d[0]
        expected = torch.exp(-distance / 12.0)
        atol = 5e-2 if precision == "single" else 4e-2
        torch.testing.assert_close(measured, expected, atol=atol, rtol=0.0)

    def test_spatial_uniform_matches_gaussian_copula_correlation(
        self, device, precision
    ):
        """Check the correlation induced by the Gaussian-copula transform."""
        batch_size = 12000
        num_ut = 4
        scenario = _umi_scenario(batch_size, 1, num_ut, precision, device, los=True)
        ray_sampler = RaysGenerator(scenario, enable_spatial_consistency=True)
        ray_sampler.topology_updated_callback()

        samples = ray_sampler._spatial_uniform((batch_size, 1, num_ut, 1))
        samples = samples[:, 0, :, 0]
        measured = _corrcoef(samples)
        normal_corr = torch.exp(-scenario.matrix_ut_distance_2d[0] / 12.0)
        expected = (6.0 / torch.pi) * torch.asin(normal_corr / 2.0)
        atol = 6e-2 if precision == "single" else 5e-2
        torch.testing.assert_close(measured, expected, atol=atol, rtol=0.0)

    def test_track_ids_freeze_mobility_discrete_fields(self, device, precision):
        """Check same-track angle signs and ray coupling are fixed."""
        batch_size = 128
        num_ut = 2
        scenario = _umi_scenario(batch_size, 1, num_ut, precision, device, los=True)
        scenario.set_topology(
            spatial_consistency_track_ids=torch.zeros(
                batch_size, num_ut, dtype=torch.int64, device=device
            )
        )
        ray_sampler = RaysGenerator(scenario, enable_spatial_consistency=True)
        ray_sampler.topology_updated_callback()

        signs = ray_sampler._spatial_binary_sign((batch_size, 1, num_ut, 7))
        torch.testing.assert_close(signs[:, :, 0], signs[:, :, 1])

        angles = torch.arange(
            scenario.num_clusters_max * scenario.rays_per_cluster,
            dtype=_dtype(precision),
            device=device,
        ).reshape(1, 1, 1, scenario.num_clusters_max, scenario.rays_per_cluster)
        angles = angles.expand(batch_size, 1, num_ut, -1, -1).clone()
        shuffled = ray_sampler._shuffle_angles(angles)
        torch.testing.assert_close(shuffled[:, :, 0], shuffled[:, :, 1])

    def test_track_ids_with_moved_ut_keep_valid_sign_covariance(
        self, device, precision
    ):
        """Check moved tracks do not make the sign covariance indefinite."""
        batch_size = 64
        num_ut = 3
        scenario = _umi_scenario(batch_size, 1, num_ut, precision, device, los=True)
        scenario.set_topology(
            spatial_consistency_track_ids=torch.tensor(
                [0, 0, 1], dtype=torch.int64, device=device
            )
        )
        ray_sampler = RaysGenerator(scenario, enable_spatial_consistency=True)
        ray_sampler.topology_updated_callback()

        signs = ray_sampler._spatial_binary_sign((batch_size, 1, num_ut, 4))
        torch.testing.assert_close(signs[:, :, 0], signs[:, :, 1])

    def test_public_channel_spatial_los_matches_for_colocated_uts(
        self, device, precision
    ):
        """Check spatially consistent random LoS sampling for co-located UTs."""
        carrier_frequency = 3.5e9
        dtype = _dtype(precision)
        ut_array, bs_array = _arrays(carrier_frequency, precision, device)
        batch_size = 256
        channel = tr38901.UMi(
            carrier_frequency=carrier_frequency,
            o2i_model="low",
            ut_array=ut_array,
            bs_array=bs_array,
            direction="downlink",
            enable_pathloss=False,
            enable_shadow_fading=False,
            always_generate_lsp=True,
            precision=precision,
            device=device,
            enable_spatial_consistency=True,
        )
        channel.set_topology(
            ut_loc=torch.tensor(
                [[[100.0, 0.0, 1.5], [100.0, 0.0, 1.5]]],
                dtype=dtype,
                device=device,
            ).expand(batch_size, -1, -1).clone(),
            bs_loc=torch.tensor(
                [[[0.0, 0.0, 10.0]]], dtype=dtype, device=device
            ).expand(batch_size, -1, -1).clone(),
            ut_orientations=torch.zeros(batch_size, 2, 3, dtype=dtype, device=device),
            bs_orientations=torch.zeros(batch_size, 1, 3, dtype=dtype, device=device),
            ut_velocities=torch.zeros(batch_size, 2, 3, dtype=dtype, device=device),
            in_state=torch.zeros(batch_size, 2, dtype=torch.bool, device=device),
            los="random",
        )

        torch.testing.assert_close(
            channel._scenario.los[:, :, 0],
            channel._scenario.los[:, :, 1],
        )

    def test_lsp_sampler_handles_colocated_uts(self, device, precision):
        """Check singular LSP spatial-correlation matrices are supported."""
        carrier_frequency = 3.5e9
        dtype = _dtype(precision)
        ut_array, bs_array = _arrays(carrier_frequency, precision, device)
        channel = tr38901.UMi(
            carrier_frequency=carrier_frequency,
            o2i_model="low",
            ut_array=ut_array,
            bs_array=bs_array,
            direction="downlink",
            enable_pathloss=False,
            enable_shadow_fading=False,
            always_generate_lsp=True,
            precision=precision,
            device=device,
        )
        channel.set_topology(
            ut_loc=torch.tensor(
                [[[100.0, 0.0, 1.5], [100.0, 0.0, 1.5]]],
                dtype=dtype,
                device=device,
            ),
            bs_loc=torch.tensor([[[0.0, 0.0, 10.0]]], dtype=dtype, device=device),
            ut_orientations=torch.zeros(1, 2, 3, dtype=dtype, device=device),
            bs_orientations=torch.zeros(1, 1, 3, dtype=dtype, device=device),
            ut_velocities=torch.zeros(1, 2, 3, dtype=dtype, device=device),
            in_state=torch.zeros(1, 2, dtype=torch.bool, device=device),
            los=True,
        )

        lsp = channel.sample_lsp()
        atol = 1e-2 if precision == "single" else 1e-5
        torch.testing.assert_close(
            lsp.ds[:, :, 0], lsp.ds[:, :, 1], atol=atol, rtol=0.0
        )
        torch.testing.assert_close(
            lsp.asa[:, :, 0], lsp.asa[:, :, 1], atol=atol, rtol=0.0
        )

    def test_spatial_regions_decorrelate_colocated_uts(self, device, precision):
        """Check floor/region boundaries mask LSP and ray random fields."""
        scenario = _umi_scenario(1, 1, 2, precision, device, los=True)
        scenario.set_topology(
            ut_loc=scenario.ut_loc[:, :1].expand(-1, 2, -1).clone(),
            ut_spatial_region_ids=torch.tensor([0, 1], device=device),
        )

        lsp_sampler = LSPGenerator(scenario)
        lsp_sampler.topology_updated_callback()
        lsp_covariance = (
            lsp_sampler._spatial_lsp_correlation_matrix_sqrt
            @ lsp_sampler._spatial_lsp_correlation_matrix_sqrt.transpose(-1, -2)
        )
        expected = torch.eye(
            2, dtype=_dtype(precision), device=device
        ).reshape(1, 1, 1, 2, 2).expand_as(lsp_covariance)
        torch.testing.assert_close(lsp_covariance, expected)

        ray_sampler = RaysGenerator(scenario, enable_spatial_consistency=True)
        ray_sampler.topology_updated_callback()
        ray_factor = ray_sampler._spatial_consistency_matrix_sqrt
        ray_covariance = ray_factor @ ray_factor.transpose(-1, -2)
        torch.testing.assert_close(
            ray_covariance,
            expected[:, :, 0],
        )

    def test_co_sited_sectors_share_spatial_random_fields(self, device, precision):
        """Check site-specific sharing for co-located BS sectors."""
        batch_size = 128
        num_ut = 3
        scenario = _umi_scenario(batch_size, 2, num_ut, precision, device, los=True)
        ray_sampler = RaysGenerator(scenario, enable_spatial_consistency=True)
        ray_sampler.topology_updated_callback()

        samples = ray_sampler._spatial_normal((batch_size, 2, num_ut, 5))
        torch.testing.assert_close(samples[:, 0], samples[:, 1])

    def test_default_ray_fields_are_site_specific(self, device, precision):
        """Check default Step 5-9 draws share only across co-sited sectors."""
        batch_size = 64
        num_ut = 2
        scenario = _umi_scenario(
            batch_size, 2, num_ut, precision, device, los=True
        )
        ray_sampler = RaysGenerator(scenario, enable_spatial_consistency=False)

        signs = ray_sampler._spatial_binary_sign(
            (batch_size, 2, num_ut, 16)
        )
        torch.testing.assert_close(signs[:, 0], signs[:, 1])
        assert torch.any(signs[:, 0, 0] != signs[:, 0, 1])

        angles = torch.arange(
            scenario.num_clusters_max * scenario.rays_per_cluster,
            dtype=_dtype(precision),
            device=device,
        ).reshape(
            1,
            1,
            1,
            scenario.num_clusters_max,
            scenario.rays_per_cluster,
        )
        angles = angles.expand(batch_size, 2, num_ut, -1, -1).clone()
        shuffled = ray_sampler._shuffle_angles(angles)
        torch.testing.assert_close(shuffled[:, 0], shuffled[:, 1])
        assert torch.any(shuffled[:, 0, 0] != shuffled[:, 0, 1])

    def test_strong_cluster_coupling_stays_within_subclusters(
        self, device, precision
    ):
        """Check Step 8 preserves Table 7.5-5 groups for strong clusters."""
        scenario = _umi_scenario(4, 1, 2, precision, device, los=True)
        ray_sampler = RaysGenerator(scenario)
        num_clusters = scenario.num_clusters_max
        num_rays = scenario.rays_per_cluster
        angles = torch.arange(
            num_rays, dtype=_dtype(precision), device=device
        ).reshape(1, 1, 1, 1, num_rays)
        angles = angles.expand(4, 1, 2, num_clusters, -1).clone()
        strongest = torch.tensor(
            [1, 3], dtype=torch.int64, device=device
        ).reshape(1, 1, 1, 2).expand(4, 1, 2, 2)

        shuffled = ray_sampler._shuffle_angles(angles, strongest)
        group = torch.empty(num_rays, dtype=torch.int64, device=device)
        group[ray_sampler._subcluster_1_indices] = 0
        group[ray_sampler._subcluster_2_indices] = 1
        group[ray_sampler._subcluster_3_indices] = 2
        source_groups = group[shuffled.to(torch.int64)]
        output_groups = group.reshape(1, 1, 1, 1, num_rays)
        for cluster in (1, 3):
            torch.testing.assert_close(
                source_groups[..., cluster, :],
                output_groups.expand_as(source_groups)[..., cluster, :],
            )

    def test_rays_include_spatially_consistent_phases_when_enabled(
        self, device, precision
    ):
        """Check that Step 10 phases are generated by the ray sampler."""
        batch_size = 16
        num_ut = 2
        dtype = _dtype(precision)
        scenario = _umi_scenario(batch_size, 1, num_ut, precision, device, los=True)
        lsp = _constant_lsp(batch_size, 1, num_ut, dtype, device)

        default_sampler = RaysGenerator(scenario)
        default_sampler.topology_updated_callback()
        assert default_sampler(lsp).phases is None

        spatial_sampler = RaysGenerator(scenario, enable_spatial_consistency=True)
        spatial_sampler.topology_updated_callback()
        rays = spatial_sampler(lsp)
        expected_shape = (
            batch_size,
            1,
            num_ut,
            scenario.num_clusters_max,
            scenario.rays_per_cluster,
            4,
        )
        assert rays.phases.shape == expected_shape
        assert torch.all(rays.phases >= -torch.pi)
        assert torch.all(rays.phases <= torch.pi)

    def test_rma_indoor_links_use_o2i_spatial_state(self, device, precision):
        """Check that RMa indoor links use the dedicated O2I state."""
        carrier_frequency = 3.5e9
        dtype = _dtype(precision)
        ut_array, bs_array = _arrays(carrier_frequency, precision, device)
        scenario = tr38901.RMaScenario(
            carrier_frequency,
            ut_array,
            bs_array,
            "downlink",
            precision=precision,
            device=device,
        )
        common_topology = dict(
            ut_loc=torch.tensor(
                [[[100.0, 0.0, 1.5], [110.0, 0.0, 1.5]]],
                dtype=dtype,
                device=device,
            ),
            bs_loc=torch.tensor([[[0.0, 0.0, 35.0]]], dtype=dtype, device=device),
            ut_orientations=torch.zeros(1, 2, 3, dtype=dtype, device=device),
            bs_orientations=torch.zeros(1, 1, 3, dtype=dtype, device=device),
            ut_velocities=torch.zeros(1, 2, 3, dtype=dtype, device=device),
            in_state=torch.ones(1, 2, dtype=torch.bool, device=device),
        )

        scenario.set_topology(**common_topology, los=True)
        ray_sampler = RaysGenerator(scenario, enable_spatial_consistency=True)
        states = ray_sampler._spatial_consistency_states()
        assert torch.all(states == 2)

        scenario.set_topology(los=False)
        ray_sampler.topology_updated_callback()
        states = ray_sampler._spatial_consistency_states()
        assert torch.all(states == 2)

    def test_channel_coefficients_use_phases_from_rays(self, device, precision):
        """Check that provided spatially consistent phases are not resampled."""
        carrier_frequency = 3.5e9
        dtype = _dtype(precision)
        ut_array, bs_array = _arrays(carrier_frequency, precision, device)
        generator = ChannelCoefficientsGenerator(
            carrier_frequency,
            bs_array,
            ut_array,
            subclustering=False,
            precision=precision,
            device=device,
        )
        phases = torch.zeros(1, 1, 1, 1, 1, 4, dtype=dtype, device=device)
        rays = Rays(
            delays=torch.zeros(1, 1, 1, 1, dtype=dtype, device=device),
            powers=torch.ones(1, 1, 1, 1, dtype=dtype, device=device),
            aoa=torch.zeros(1, 1, 1, 1, 1, dtype=dtype, device=device),
            aod=torch.zeros(1, 1, 1, 1, 1, dtype=dtype, device=device),
            zoa=torch.full((1, 1, 1, 1, 1), torch.pi / 2, dtype=dtype, device=device),
            zod=torch.full((1, 1, 1, 1, 1), torch.pi / 2, dtype=dtype, device=device),
            xpr=torch.ones(1, 1, 1, 1, 1, dtype=dtype, device=device),
            phases=phases,
        )
        topology = Topology(
            velocities=torch.zeros(1, 1, 3, dtype=dtype, device=device),
            moving_end="rx",
            los_aoa=torch.zeros(1, 1, 1, dtype=dtype, device=device),
            los_aod=torch.zeros(1, 1, 1, dtype=dtype, device=device),
            los_zoa=torch.full((1, 1, 1), torch.pi / 2, dtype=dtype, device=device),
            los_zod=torch.full((1, 1, 1), torch.pi / 2, dtype=dtype, device=device),
            los=torch.zeros(1, 1, 1, dtype=torch.bool, device=device),
            distance_3d=torch.ones(1, 1, 1, dtype=dtype, device=device),
            tx_orientations=torch.zeros(1, 1, 3, dtype=dtype, device=device),
            rx_orientations=torch.zeros(1, 1, 3, dtype=dtype, device=device),
        )
        captured = {}

        def fake_step_11(phi, topology, k_factor, rays, sample_times, c_ds):
            del topology, k_factor, rays, sample_times, c_ds
            captured["phi"] = phi
            h = torch.zeros(1, 1, 1, 1, 1, 1, 1, dtype=generator.cdtype, device=device)
            delays = torch.zeros(1, 1, 1, 1, dtype=dtype, device=device)
            return h, delays

        generator._step_11 = fake_step_11
        generator(
            1,
            1.0,
            torch.ones(1, 1, 1, dtype=dtype, device=device),
            rays,
            topology,
        )
        torch.testing.assert_close(captured["phi"], phases)

    def test_public_channel_opt_in_smoke(self, device, precision):
        """Check the public UMi opt-in path produces CIRs and rays."""
        carrier_frequency = 3.5e9
        dtype = _dtype(precision)
        ut_array, bs_array = _arrays(carrier_frequency, precision, device)
        batch_size = 2
        num_ut = 2
        for direction in ("downlink", "uplink"):
            channel = tr38901.UMi(
                carrier_frequency=carrier_frequency,
                o2i_model="low",
                ut_array=ut_array,
                bs_array=bs_array,
                direction=direction,
                enable_pathloss=False,
                enable_shadow_fading=False,
                always_generate_lsp=True,
                precision=precision,
                device=device,
                enable_spatial_consistency=True,
            )
            channel.set_topology(
                ut_loc=torch.tensor(
                    [[[100.0, 0.0, 1.5], [105.0, 0.0, 1.5]]],
                    dtype=dtype,
                    device=device,
                ).expand(batch_size, -1, -1).clone(),
                bs_loc=torch.tensor(
                    [[[0.0, 0.0, 10.0]]],
                    dtype=dtype,
                    device=device,
                ).expand(batch_size, -1, -1).clone(),
                ut_orientations=torch.zeros(
                    batch_size, num_ut, 3, dtype=dtype, device=device
                ),
                bs_orientations=torch.zeros(
                    batch_size, 1, 3, dtype=dtype, device=device
                ),
                ut_velocities=torch.zeros(
                    batch_size, num_ut, 3, dtype=dtype, device=device
                ),
                in_state=torch.zeros(
                    batch_size, num_ut, dtype=torch.bool, device=device
                ),
                los=True,
            )
            channel.return_rays = True
            h, tau, rays = channel(num_time_samples=1, sampling_frequency=1.0)
            assert h.shape[0] == batch_size
            assert tau.shape[0] == batch_size
            assert rays.phases is not None
            assert rays.cluster_sort_indices is not None
            expected_link_shape = (
                (batch_size, 1, num_ut)
                if direction == "downlink"
                else (batch_size, num_ut, 1)
            )
            assert rays.cluster_sort_indices.shape[:3] == expected_link_shape

    def test_preallocated_topology_reduce_overhead_compile_has_no_graph_breaks(
        self, device
    ):
        """Check dynamic topology updates compile without CUDA-graph breaks."""
        if device == "cpu":
            pytest.skip("CUDA graph compatibility is only relevant on GPU")
        if not hasattr(torch, "compile"):
            pytest.skip("torch.compile is not available")

        import torch._dynamo as dynamo
        from torch._dynamo.utils import counters

        carrier_frequency = 3.5e9
        ut_array, bs_array = _arrays(carrier_frequency, "single", device)
        channel = tr38901.UMi(
            carrier_frequency=carrier_frequency,
            o2i_model="low",
            ut_array=ut_array,
            bs_array=bs_array,
            direction="uplink",
            enable_pathloss=False,
            enable_shadow_fading=False,
            always_generate_lsp=True,
            enable_spatial_consistency=True,
            enable_blockage=True,
            blockage_self_blocking="portrait",
            precision="single",
            device=device,
        )
        channel.allocate_topology_tensors(batch_size=2, num_bs=1, num_ut=2)

        def generate_rays():
            topology = gen_single_sector_topology(
                batch_size=2,
                num_ut=2,
                scenario="umi",
                min_ut_velocity=0.0,
                max_ut_velocity=0.0,
                device=device,
            )
            bs_virtual_loc = topology[1].unsqueeze(2).expand(-1, -1, 2, -1)
            track_ids = torch.tensor([[0, 0], [1, 1]], device=device)
            channel.set_topology(
                *topology,
                bs_virtual_loc=bs_virtual_loc,
                spatial_consistency_track_ids=track_ids,
            )
            lsp = channel._lsp_sampler()
            rays = channel._ray_sampler(lsp)
            return (
                rays.delays.sum()
                + rays.aoa.sum()
                + rays.phases.sum()
                + rays.blockage_loss_db.sum()
            )

        dynamo.reset()
        counters.clear()
        compiled_generate_rays = torch.compile(generate_rays, mode="reduce-overhead")
        output = compiled_generate_rays()
        torch.cuda.synchronize(torch.device(device))

        assert torch.isfinite(output)
        assert not dict(counters.get("graph_break", {}))
