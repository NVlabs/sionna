#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Tests for 3GPP TR 38.901 blockage."""

import math

import pytest
import torch

from sionna.phy.channel import tr38901
from sionna.phy.channel.tr38901 import (
    BlockageModelA,
    BlockageModelB,
    ChannelCoefficientsGenerator,
    Rays,
    Topology,
)
from sionna.phy.channel.utils import deg_2_rad


def _dtype(precision):
    return torch.float32 if precision == "single" else torch.float64


def _corrcoef(samples):
    samples = samples - samples.mean(dim=0, keepdim=True)
    covariance = samples.mT @ samples / (samples.shape[0] - 1)
    standard_deviation = torch.sqrt(torch.diagonal(covariance).clamp_min(1e-30))
    return covariance / (
        standard_deviation.unsqueeze(0) * standard_deviation.unsqueeze(1)
    )


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


def _umi_scenario(precision, device, num_bs=1, ut_x=(0.0,)):
    carrier_frequency = 30e9
    dtype = _dtype(precision)
    num_ut = len(ut_x)
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
    scenario.set_topology(
        ut_loc=torch.tensor(
            [[[x, 0.0, 1.5] for x in ut_x]], dtype=dtype, device=device
        ),
        bs_loc=torch.tensor(
            [[[100.0, 0.0, 10.0]]], dtype=dtype, device=device
        ).expand(1, num_bs, -1).clone(),
        ut_orientations=torch.zeros(1, num_ut, 3, dtype=dtype, device=device),
        bs_orientations=torch.zeros(1, num_bs, 3, dtype=dtype, device=device),
        ut_velocities=torch.zeros(1, num_ut, 3, dtype=dtype, device=device),
        in_state=torch.zeros(1, num_ut, dtype=torch.bool, device=device),
        los=True,
    )
    return scenario


def _horizontal_los_scenario(precision, device):
    carrier_frequency = 30e9
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
    scenario.set_topology(
        ut_loc=torch.tensor([[[0.0, 0.0, 1.5]]], dtype=dtype, device=device),
        bs_loc=torch.tensor([[[10.0, 0.0, 1.5]]], dtype=dtype, device=device),
        ut_orientations=torch.zeros(1, 1, 3, dtype=dtype, device=device),
        bs_orientations=torch.zeros(1, 1, 3, dtype=dtype, device=device),
        ut_velocities=torch.zeros(1, 1, 3, dtype=dtype, device=device),
        in_state=torch.zeros(1, 1, dtype=torch.bool, device=device),
        los=True,
    )
    return scenario


class TestBlockageModelA:
    """Tests for TR 38.901 blockage model A."""

    def test_self_blocking_selection_is_required(self, device, precision):
        """Check that omitting the standards-required region is not implicit."""
        scenario = _umi_scenario(precision, device)
        with pytest.raises(TypeError, match="self_blocking"):
            BlockageModelA(
                scenario,
                num_non_self_blockers=0,
                precision=precision,
                device=device,
            )
        with pytest.raises(ValueError, match="non-standard no-self"):
            BlockageModelA(
                scenario,
                self_blocking=None,
                num_non_self_blockers=0,
                precision=precision,
                device=device,
            )

    def test_uniform_correlation_uses_gaussian_copula_prewarp(
        self, device, precision
    ):
        """Check the analytic inverse for correlated uniform variables."""
        scenario = _umi_scenario(
            precision,
            device,
            ut_x=(0.0, 10.0 * math.log(2.0)),
        )
        model = BlockageModelA(
            scenario,
            self_blocking="none",
            num_non_self_blockers=0,
            precision=precision,
            device=device,
        )
        model.topology_updated_callback()

        uniform_correlation = torch.exp(-scenario.matrix_ut_distance_2d / 10.0)
        expected_gaussian_correlation = 2.0 * torch.sin(
            (torch.pi / 6.0) * uniform_correlation
        )
        actual_gaussian_correlation = model._matrix_sqrt @ model._matrix_sqrt.mT
        torch.testing.assert_close(
            actual_gaussian_correlation,
            expected_gaussian_correlation,
        )

        induced_uniform_correlation = (6.0 / torch.pi) * torch.asin(
            actual_gaussian_correlation / 2.0
        )
        torch.testing.assert_close(
            induced_uniform_correlation,
            uniform_correlation,
        )

    def test_spatial_regions_decorrelate_blockage(self, device, precision):
        """Check co-located UTs on different floors are uncorrelated."""
        scenario = _umi_scenario(precision, device, ut_x=(0.0, 0.0))
        scenario.set_topology(
            ut_spatial_region_ids=torch.tensor([0, 1], device=device)
        )
        model = BlockageModelA(
            scenario,
            self_blocking="none",
            num_non_self_blockers=0,
            precision=precision,
            device=device,
        )
        model.topology_updated_callback()

        covariance = model._matrix_sqrt @ model._matrix_sqrt.mT
        # Different spatial-region IDs remove the off-diagonal covariance.
        expected = torch.eye(
            2, dtype=_dtype(precision), device=device
        ).unsqueeze(0)
        torch.testing.assert_close(covariance, expected)

    def test_blocker_centre_uniforms_have_exponential_correlation(
        self, device, precision
    ):
        """Check the specified exponential correlation empirically."""
        scenario = _umi_scenario(
            precision,
            device,
            ut_x=(0.0, 10.0 * math.log(2.0)),
        )
        model = BlockageModelA(
            scenario,
            self_blocking="none",
            num_non_self_blockers=100_000,
            precision=precision,
            device=device,
        )
        generator_state = model.torch_rng.get_state()
        try:
            model.torch_rng.manual_seed(1234)
            model.topology_updated_callback()
        finally:
            model.torch_rng.set_state(generator_state)

        measured = _corrcoef(model._blocker_phi[0].mT)
        expected = torch.exp(-scenario.matrix_ut_distance_2d[0] / 10.0)
        torch.testing.assert_close(measured, expected, rtol=0.0, atol=1.5e-2)

    def test_self_blocking_region_adds_30_db(self, device, precision):
        """Check Table 7.6.4.1-1 self-blocking regions."""
        dtype = _dtype(precision)
        scenario = _umi_scenario(precision, device)
        model = BlockageModelA(
            scenario,
            self_blocking="portrait",
            num_non_self_blockers=0,
            precision=precision,
            device=device,
        )

        aoa = torch.tensor([[[[260.0]]]], dtype=dtype, device=device)
        zoa = torch.tensor([[[[100.0]]]], dtype=dtype, device=device)
        loss, los_loss = model(aoa, zoa, aoa.squeeze(-1), zoa.squeeze(-1))
        torch.testing.assert_close(loss, torch.full_like(loss, 30.0))
        torch.testing.assert_close(los_loss, torch.full_like(los_loss, 30.0))

        # A ray outside the selected angular region is not attenuated.
        aoa = torch.tensor([[[[10.0]]]], dtype=dtype, device=device)
        zoa = torch.tensor([[[[100.0]]]], dtype=dtype, device=device)
        loss, _ = model(aoa, zoa)
        torch.testing.assert_close(loss, torch.zeros_like(loss))

    def test_non_self_blocking_matches_equation_7_6_22(self, device, precision):
        """Check the non-self-blocking knife-edge formula."""
        dtype = _dtype(precision)
        scenario = _umi_scenario(precision, device)
        model = BlockageModelA(
            scenario,
            self_blocking="none",
            num_non_self_blockers=1,
            precision=precision,
            device=device,
        )
        model.topology_updated_callback()
        model._blocker_phi.copy_(
            torch.tensor([[[90.0]]], dtype=dtype, device=device)
        )
        model._blocker_x.copy_(
            torch.tensor([[[10.0]]], dtype=dtype, device=device)
        )
        model._blocker_y.copy_(
            torch.tensor([[[5.0]]], dtype=dtype, device=device)
        )

        aoa = torch.tensor([[[[90.0]]]], dtype=dtype, device=device)
        zoa = torch.tensor([[[[90.0]]]], dtype=dtype, device=device)
        loss, _ = model(aoa, zoa)

        wavelength = scenario.lambda_0
        r = torch.tensor(10.0, dtype=dtype, device=device)

        def f(angle_deg):
            angle = deg_2_rad(torch.tensor(angle_deg, dtype=dtype, device=device))
            radicand = (torch.pi / wavelength) * r * (1.0 / torch.cos(angle) - 1.0)
            return torch.atan(0.5 * torch.pi * torch.sqrt(radicand)) / torch.pi

        expected = -20.0 * torch.log10(
            1.0 - (2.0 * f(5.0)) * (2.0 * f(2.5))
        )
        torch.testing.assert_close(loss.squeeze(), expected)

        # The blocker must not attenuate rays outside its angular extent.
        outside, _ = model(
            torch.tensor([[[[120.0]]]], dtype=dtype, device=device),
            zoa,
        )
        torch.testing.assert_close(outside, torch.zeros_like(outside))

    def test_non_self_blockers_are_all_correlated_over_bs_links(
        self, device, precision
    ):
        """Check Section 7.6.3.4 all-correlated BS-link behavior."""
        dtype = _dtype(precision)
        scenario = _umi_scenario(precision, device, num_bs=2)
        model = BlockageModelA(
            scenario,
            self_blocking="none",
            num_non_self_blockers=1,
            precision=precision,
            device=device,
        )
        model.topology_updated_callback()
        model._blocker_phi.copy_(
            torch.tensor([[[45.0]]], dtype=dtype, device=device)
        )
        model._blocker_x.copy_(
            torch.tensor([[[10.0]]], dtype=dtype, device=device)
        )
        model._blocker_y.copy_(
            torch.tensor([[[5.0]]], dtype=dtype, device=device)
        )
        aoa = torch.full((1, 2, 1, 1), 45.0, dtype=dtype, device=device)
        zoa = torch.full((1, 2, 1, 1), 90.0, dtype=dtype, device=device)
        loss, _ = model(aoa, zoa)
        torch.testing.assert_close(loss[:, 0], loss[:, 1])


class TestBlockageModelB:
    """Tests for TR 38.901 blockage model B."""

    def test_geometric_screen_loss_matches_equation_7_6_29(
        self, device, precision
    ):
        """Check screen diffraction loss for a centred horizontal LOS link."""
        dtype = _dtype(precision)
        scenario = _horizontal_los_scenario(precision, device)
        model = BlockageModelB(
            scenario,
            blocker_centers=torch.tensor([[5.0, 0.0, 1.5]], dtype=dtype, device=device),
            blocker_widths=torch.tensor([2.0], dtype=dtype, device=device),
            blocker_heights=torch.tensor([2.0], dtype=dtype, device=device),
            precision=precision,
            device=device,
        )

        aoa = torch.zeros(1, 1, 1, 1, 1, dtype=dtype, device=device)
        zoa = torch.full_like(aoa, 90.0)
        ray_loss, los_loss = model(
            aoa,
            zoa,
            scenario.los_aoa,
            scenario.los_zoa,
        )

        wavelength = scenario.lambda_0
        edge_distance = torch.sqrt(
            torch.tensor(26.0, dtype=dtype, device=device)
        )

        def loss_from_excess(excess):
            f = torch.atan(
                0.5 * torch.pi
                * torch.sqrt((torch.pi / wavelength) * excess)
            ) / torch.pi
            return -20.0 * torch.log10(1.0 - (2.0 * f) * (2.0 * f))

        expected_ray = loss_from_excess(edge_distance - 5.0)
        expected_los = loss_from_excess(2.0 * edge_distance - 10.0)
        torch.testing.assert_close(ray_loss.squeeze(), expected_ray)
        torch.testing.assert_close(los_loss.squeeze(), expected_los)

    def test_multiple_screens_sum_losses_in_db(self, device, precision):
        """Check that multiple Model B screens add losses in dB."""
        dtype = _dtype(precision)
        scenario = _horizontal_los_scenario(precision, device)
        first = BlockageModelB(
            scenario,
            blocker_centers=torch.tensor([[4.0, 0.0, 1.5]], dtype=dtype, device=device),
            blocker_widths=torch.tensor([2.0], dtype=dtype, device=device),
            blocker_heights=torch.tensor([2.0], dtype=dtype, device=device),
            precision=precision,
            device=device,
        )
        second = BlockageModelB(
            scenario,
            blocker_centers=torch.tensor([[6.0, 0.0, 1.5]], dtype=dtype, device=device),
            blocker_widths=torch.tensor([2.0], dtype=dtype, device=device),
            blocker_heights=torch.tensor([2.0], dtype=dtype, device=device),
            precision=precision,
            device=device,
        )
        combined = BlockageModelB(
            scenario,
            blocker_centers=torch.tensor(
                [[4.0, 0.0, 1.5], [6.0, 0.0, 1.5]],
                dtype=dtype,
                device=device,
            ),
            blocker_widths=torch.tensor([2.0, 2.0], dtype=dtype, device=device),
            blocker_heights=torch.tensor([2.0, 2.0], dtype=dtype, device=device),
            precision=precision,
            device=device,
        )
        aoa = torch.zeros(1, 1, 1, 1, 1, dtype=dtype, device=device)
        zoa = torch.full_like(aoa, 90.0)
        first_ray_loss, first_los_loss = first(
            aoa, zoa, scenario.los_aoa, scenario.los_zoa
        )
        second_ray_loss, second_los_loss = second(
            aoa, zoa, scenario.los_aoa, scenario.los_zoa
        )
        combined_ray_loss, combined_los_loss = combined(
            aoa, zoa, scenario.los_aoa, scenario.los_zoa
        )

        torch.testing.assert_close(
            combined_ray_loss,
            first_ray_loss + second_ray_loss,
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            combined_los_loss,
            first_los_loss + second_los_loss,
            rtol=0.0,
            atol=0.0,
        )

    def test_public_umi_model_b_opt_in_smoke(self, device, precision):
        """Check the public Model B path with explicit blocker screens."""
        carrier_frequency = 30e9
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
            enable_blockage=True,
            blockage_model="B",
            blockage_screen_centers=torch.tensor(
                [[80.0, 10.0, 1.5]], dtype=dtype, device=device
            ),
            blockage_screen_widths=torch.tensor([2.0], dtype=dtype, device=device),
            blockage_screen_heights=torch.tensor([10.0], dtype=dtype, device=device),
        )
        channel.set_topology(
            ut_loc=torch.tensor(
                [[[100.0, 10.0, 1.5]]], dtype=dtype, device=device
            ),
            bs_loc=torch.tensor(
                [[[0.0, 0.0, 30.0]]], dtype=dtype, device=device
            ),
            ut_orientations=torch.zeros(1, 1, 3, dtype=dtype, device=device),
            bs_orientations=torch.zeros(1, 1, 3, dtype=dtype, device=device),
            ut_velocities=torch.zeros(1, 1, 3, dtype=dtype, device=device),
            in_state=torch.zeros(1, 1, dtype=torch.bool, device=device),
            los=True,
        )
        channel.return_rays = True
        h, tau, rays = channel(1, 1.0)
        assert torch.isfinite(h).all()
        assert torch.isfinite(tau).all()
        assert rays.blockage_loss_db is not None
        assert rays.los_blockage_loss_db is not None
        assert not rays.blockage_loss_applied_to_powers

    def test_public_inf_model_b_opt_in_smoke(self, device, precision):
        """Check that InF exposes the explicit-screen Model B path."""
        carrier_frequency = 30e9
        dtype = _dtype(precision)
        ut_array, bs_array = _arrays(carrier_frequency, precision, device)
        channel = tr38901.InF(
            carrier_frequency=carrier_frequency,
            ut_array=ut_array,
            bs_array=bs_array,
            direction="downlink",
            factory_scenario="SH",
            enable_pathloss=False,
            enable_shadow_fading=False,
            always_generate_lsp=True,
            precision=precision,
            device=device,
            enable_blockage=True,
            blockage_screen_centers=torch.tensor(
                [[8.0, 0.0, 1.5]], dtype=dtype, device=device
            ),
            blockage_screen_widths=torch.tensor([2.0], dtype=dtype, device=device),
            blockage_screen_heights=torch.tensor([3.0], dtype=dtype, device=device),
        )
        channel.set_topology(
            ut_loc=torch.tensor([[[10.0, 0.0, 1.5]]], dtype=dtype, device=device),
            bs_loc=torch.tensor([[[0.0, 0.0, 8.0]]], dtype=dtype, device=device),
            ut_orientations=torch.zeros(1, 1, 3, dtype=dtype, device=device),
            bs_orientations=torch.zeros(1, 1, 3, dtype=dtype, device=device),
            ut_velocities=torch.zeros(1, 1, 3, dtype=dtype, device=device),
            los=True,
        )
        channel.return_rays = True
        h, tau, rays = channel(1, 1.0)
        assert torch.isfinite(h).all()
        assert torch.isfinite(tau).all()
        assert rays.blockage_loss_db is not None
        assert rays.los_blockage_loss_db is not None
        assert not rays.blockage_loss_applied_to_powers


class TestBlockageIntegration:
    """Integration tests for blockage in channel generation."""

    def test_public_model_a_requires_explicit_self_blocking(
        self, device, precision
    ):
        """Check that public Model A use cannot omit the self-blocker choice."""
        carrier_frequency = 30e9
        ut_array, bs_array = _arrays(carrier_frequency, precision, device)
        kwargs = dict(
            carrier_frequency=carrier_frequency,
            o2i_model="low",
            ut_array=ut_array,
            bs_array=bs_array,
            direction="downlink",
            enable_blockage=True,
            precision=precision,
            device=device,
        )

        with pytest.raises(ValueError, match="must explicitly select"):
            tr38901.UMi(**kwargs)

        channel = tr38901.UMi(
            **kwargs,
            blockage_self_blocking="none",
        )
        assert channel._blockage_model.self_blocking == "none"

    def test_los_blockage_attenuates_deterministic_los_component(
        self, device, precision
    ):
        """Check that LOS blockage is applied to the deterministic component."""
        carrier_frequency = 30e9
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
        common_rays = dict(
            delays=torch.zeros(1, 1, 1, 1, dtype=dtype, device=device),
            powers=torch.zeros(1, 1, 1, 1, dtype=dtype, device=device),
            aoa=torch.zeros(1, 1, 1, 1, 1, dtype=dtype, device=device),
            aod=torch.zeros(1, 1, 1, 1, 1, dtype=dtype, device=device),
            zoa=torch.full(
                (1, 1, 1, 1, 1), torch.pi / 2, dtype=dtype, device=device
            ),
            zod=torch.full(
                (1, 1, 1, 1, 1), torch.pi / 2, dtype=dtype, device=device
            ),
            xpr=torch.ones(1, 1, 1, 1, 1, dtype=dtype, device=device),
            phases=torch.zeros(1, 1, 1, 1, 1, 4, dtype=dtype, device=device),
        )
        topology = Topology(
            velocities=torch.zeros(1, 1, 3, dtype=dtype, device=device),
            moving_end="rx",
            los_aoa=torch.zeros(1, 1, 1, dtype=dtype, device=device),
            los_aod=torch.zeros(1, 1, 1, dtype=dtype, device=device),
            los_zoa=torch.full((1, 1, 1), torch.pi / 2, dtype=dtype, device=device),
            los_zod=torch.full((1, 1, 1), torch.pi / 2, dtype=dtype, device=device),
            los=torch.ones(1, 1, 1, dtype=torch.bool, device=device),
            distance_3d=torch.ones(1, 1, 1, dtype=dtype, device=device),
            tx_orientations=torch.zeros(1, 1, 3, dtype=dtype, device=device),
            rx_orientations=torch.zeros(1, 1, 3, dtype=dtype, device=device),
        )
        k_factor = torch.full((1, 1, 1), 1e12, dtype=dtype, device=device)

        rays = Rays(**common_rays)
        h_ref, _ = generator(1, 1.0, k_factor, rays, topology)
        rays_blocked = Rays(
            **common_rays,
            los_blockage_loss_db=torch.full(
                (1, 1, 1), 20.0, dtype=dtype, device=device
            ),
        )
        h_blocked, _ = generator(1, 1.0, k_factor, rays_blocked, topology)

        torch.testing.assert_close(
            torch.abs(h_blocked),
            0.1 * torch.abs(h_ref),
            rtol=2e-5,
            atol=2e-6,
        )

    def test_ray_blockage_attenuates_nlos_subpaths(self, device, precision):
        """Check per-ray blockage attenuation for NLOS components."""
        carrier_frequency = 30e9
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
        common_rays = dict(
            delays=torch.zeros(1, 1, 1, 1, dtype=dtype, device=device),
            powers=torch.ones(1, 1, 1, 1, dtype=dtype, device=device),
            aoa=torch.zeros(1, 1, 1, 1, 1, dtype=dtype, device=device),
            aod=torch.zeros(1, 1, 1, 1, 1, dtype=dtype, device=device),
            zoa=torch.full(
                (1, 1, 1, 1, 1), torch.pi / 2, dtype=dtype, device=device
            ),
            zod=torch.full(
                (1, 1, 1, 1, 1), torch.pi / 2, dtype=dtype, device=device
            ),
            xpr=torch.ones(1, 1, 1, 1, 1, dtype=dtype, device=device),
            phases=torch.zeros(1, 1, 1, 1, 1, 4, dtype=dtype, device=device),
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
        k_factor = torch.ones((1, 1, 1), dtype=dtype, device=device)

        h_ref, _ = generator(1, 1.0, k_factor, Rays(**common_rays), topology)
        h_blocked, _ = generator(
            1,
            1.0,
            k_factor,
            Rays(
                **common_rays,
                blockage_loss_db=torch.full(
                    (1, 1, 1, 1, 1), 20.0, dtype=dtype, device=device
                ),
            ),
            topology,
        )
        torch.testing.assert_close(
            torch.abs(h_blocked),
            0.1 * torch.abs(h_ref),
            rtol=2e-5,
            atol=2e-6,
        )

    def test_public_umi_blockage_opt_in_smoke(self, device, precision):
        """Check the public opt-in path returns blockage losses."""
        carrier_frequency = 30e9
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
            enable_blockage=True,
            blockage_self_blocking="landscape",
        )
        channel.set_topology(
            ut_loc=torch.tensor(
                [[[100.0, 0.0, 1.5]]], dtype=dtype, device=device
            ),
            bs_loc=torch.tensor(
                [[[0.0, 0.0, 10.0]]], dtype=dtype, device=device
            ),
            ut_orientations=torch.zeros(1, 1, 3, dtype=dtype, device=device),
            bs_orientations=torch.zeros(1, 1, 3, dtype=dtype, device=device),
            ut_velocities=torch.zeros(1, 1, 3, dtype=dtype, device=device),
            in_state=torch.zeros(1, 1, dtype=torch.bool, device=device),
            los=True,
        )
        channel.return_rays = True
        h, tau, rays = channel(1, 1.0)
        assert torch.isfinite(h).all()
        assert torch.isfinite(tau).all()
        assert rays.blockage_loss_db is not None
        assert rays.los_blockage_loss_db is not None
        assert rays.blockage_loss_db.shape[-1] == channel._scenario.rays_per_cluster
