#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Tests for 3GPP TR 38.901 Rays classes"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from scipy.stats import kstest

from sionna.phy import PI, config
from sionna.phy.channel import tr38901
from sionna.phy.channel.tr38901 import LSP, Rays, RaysGenerator

from channel_test_utils import (
    delays_ref,
    generate_random_bool,
    generate_random_loc,
    powers_ref,
    xpr_ref,
)


class TestRays:
    """Tests for the Rays data class"""

    def test_rays_instantiation(self, device, precision):
        """Test that Rays can be instantiated with the expected shapes"""
        batch_size = 10
        num_bs = 3
        num_ut = 5
        num_clusters = 20
        num_rays = 20

        # Create random tensors with expected shapes
        delays = torch.rand(batch_size, num_bs, num_ut, num_clusters, device=device)
        powers = torch.rand(batch_size, num_bs, num_ut, num_clusters, device=device)
        aoa = torch.rand(batch_size, num_bs, num_ut, num_clusters, num_rays, device=device)
        aod = torch.rand(batch_size, num_bs, num_ut, num_clusters, num_rays, device=device)
        zoa = torch.rand(batch_size, num_bs, num_ut, num_clusters, num_rays, device=device)
        zod = torch.rand(batch_size, num_bs, num_ut, num_clusters, num_rays, device=device)
        xpr = torch.rand(batch_size, num_bs, num_ut, num_clusters, num_rays, device=device)

        rays = Rays(
            delays=delays,
            powers=powers,
            aoa=aoa,
            aod=aod,
            zoa=zoa,
            zod=zod,
            xpr=xpr,
        )

        # Verify all attributes are correctly assigned
        assert torch.equal(rays.delays, delays)
        assert torch.equal(rays.powers, powers)
        assert torch.equal(rays.aoa, aoa)
        assert torch.equal(rays.aod, aod)
        assert torch.equal(rays.zoa, zoa)
        assert torch.equal(rays.zod, zod)
        assert torch.equal(rays.xpr, xpr)

    def test_rays_delays_shape(self, device, precision):
        """Test that delays have correct shape"""
        batch_size = 5
        num_bs = 2
        num_ut = 3
        num_clusters = 15
        num_rays = 10

        delays = torch.rand(batch_size, num_bs, num_ut, num_clusters, device=device)
        powers = torch.rand(batch_size, num_bs, num_ut, num_clusters, device=device)
        aoa = torch.rand(batch_size, num_bs, num_ut, num_clusters, num_rays, device=device)
        aod = torch.rand(batch_size, num_bs, num_ut, num_clusters, num_rays, device=device)
        zoa = torch.rand(batch_size, num_bs, num_ut, num_clusters, num_rays, device=device)
        zod = torch.rand(batch_size, num_bs, num_ut, num_clusters, num_rays, device=device)
        xpr = torch.rand(batch_size, num_bs, num_ut, num_clusters, num_rays, device=device)

        rays = Rays(
            delays=delays,
            powers=powers,
            aoa=aoa,
            aod=aod,
            zoa=zoa,
            zod=zod,
            xpr=xpr,
        )

        assert rays.delays.shape == (batch_size, num_bs, num_ut, num_clusters)

    def test_rays_powers_normalized(self, device, precision):
        """Test that powers can be normalized"""
        batch_size = 5
        num_bs = 2
        num_ut = 3
        num_clusters = 10
        num_rays = 5

        # Create unnormalized powers
        powers_unnorm = torch.rand(batch_size, num_bs, num_ut, num_clusters, device=device)
        # Normalize
        powers = powers_unnorm / powers_unnorm.sum(dim=-1, keepdim=True)

        delays = torch.rand(batch_size, num_bs, num_ut, num_clusters, device=device)
        aoa = torch.rand(batch_size, num_bs, num_ut, num_clusters, num_rays, device=device)
        aod = torch.rand(batch_size, num_bs, num_ut, num_clusters, num_rays, device=device)
        zoa = torch.rand(batch_size, num_bs, num_ut, num_clusters, num_rays, device=device)
        zod = torch.rand(batch_size, num_bs, num_ut, num_clusters, num_rays, device=device)
        xpr = torch.rand(batch_size, num_bs, num_ut, num_clusters, num_rays, device=device)

        rays = Rays(
            delays=delays,
            powers=powers,
            aoa=aoa,
            aod=aod,
            zoa=zoa,
            zod=zod,
            xpr=xpr,
        )

        # Check that powers sum to 1 for each BS-UT link
        power_sum = rays.powers.sum(dim=-1)
        assert torch.allclose(power_sum, torch.ones_like(power_sum), atol=1e-5)

    def test_rays_angles_in_radians(self, device, precision):
        """Test that angles can be stored in radians"""
        batch_size = 3
        num_bs = 1
        num_ut = 2
        num_clusters = 5
        num_rays = 4

        # Create angles in radians (between -pi and pi for azimuth, 0 to pi for zenith)
        aoa = (torch.rand(batch_size, num_bs, num_ut, num_clusters, num_rays, device=device) * 2 - 1) * PI
        aod = (torch.rand(batch_size, num_bs, num_ut, num_clusters, num_rays, device=device) * 2 - 1) * PI
        zoa = torch.rand(batch_size, num_bs, num_ut, num_clusters, num_rays, device=device) * PI
        zod = torch.rand(batch_size, num_bs, num_ut, num_clusters, num_rays, device=device) * PI

        delays = torch.rand(batch_size, num_bs, num_ut, num_clusters, device=device)
        powers = torch.rand(batch_size, num_bs, num_ut, num_clusters, device=device)
        xpr = torch.rand(batch_size, num_bs, num_ut, num_clusters, num_rays, device=device)

        rays = Rays(
            delays=delays,
            powers=powers,
            aoa=aoa,
            aod=aod,
            zoa=zoa,
            zod=zod,
            xpr=xpr,
        )

        # Check azimuth angles are in valid range
        assert torch.all(rays.aoa >= -PI)
        assert torch.all(rays.aoa <= PI)
        assert torch.all(rays.aod >= -PI)
        assert torch.all(rays.aod <= PI)

        # Check zenith angles are in valid range
        assert torch.all(rays.zoa >= 0)
        assert torch.all(rays.zoa <= PI)
        assert torch.all(rays.zod >= 0)
        assert torch.all(rays.zod <= PI)

    def test_rays_xpr_positive(self, device, precision):
        """Test that XPR values are positive"""
        batch_size = 3
        num_bs = 1
        num_ut = 2
        num_clusters = 5
        num_rays = 4

        delays = torch.rand(batch_size, num_bs, num_ut, num_clusters, device=device)
        powers = torch.rand(batch_size, num_bs, num_ut, num_clusters, device=device)
        aoa = torch.rand(batch_size, num_bs, num_ut, num_clusters, num_rays, device=device)
        aod = torch.rand(batch_size, num_bs, num_ut, num_clusters, num_rays, device=device)
        zoa = torch.rand(batch_size, num_bs, num_ut, num_clusters, num_rays, device=device)
        zod = torch.rand(batch_size, num_bs, num_ut, num_clusters, num_rays, device=device)
        # XPR should be positive (linear scale)
        xpr = torch.abs(torch.randn(batch_size, num_bs, num_ut, num_clusters, num_rays, device=device)) + 0.1

        rays = Rays(
            delays=delays,
            powers=powers,
            aoa=aoa,
            aod=aod,
            zoa=zoa,
            zod=zod,
            xpr=xpr,
        )

        assert torch.all(rays.xpr > 0)


class TestRaysGenerator:
    """Tests for RaysGenerator matching TensorFlow implementation tests."""

    # Test configuration
    BATCH_SIZE = 100000  # Large batch for statistical tests
    CARRIER_FREQUENCY = 3.5e9  # Hz
    H_UT = 1.5  # Height of UTs
    H_BS = 35.0  # Height of BSs

    # Test thresholds
    MAX_ERR = 3e-2  # Maximum KS error

    # Number of clusters per model/submodel
    NUM_CLUSTERS = {
        "rma": {"los": 11, "nlos": 10, "o2i": 10},
        "umi": {"los": 12, "nlos": 19, "o2i": 12},
        "uma": {"los": 12, "nlos": 20, "o2i": 12},
    }

    def test_ray_offset_negative_1_1481_is_exact(self, device, precision):
        """Pin the negative Table 7.5-3 offset at its production index."""

        scenario = SimpleNamespace(precision=precision, device=device)
        ray_sampler = RaysGenerator(scenario)
        expected = torch.tensor(-1.1481, dtype=ray_sampler.dtype, device=device)

        assert ray_sampler._ray_offsets.shape == (20,)
        assert torch.equal(ray_sampler._ray_offsets[15], expected)

    def test_cluster_power_pruning_keeps_fixed_shape(self, device, precision, monkeypatch):
        """Test Step 6 pruning zeros weak clusters without changing shape."""
        import sionna.phy.channel.tr38901.rays as rays_module

        fc = self.CARRIER_FREQUENCY
        dtype = torch.float32 if precision == "single" else torch.float64
        bs_array = tr38901.PanelArray(
            num_rows_per_panel=1,
            num_cols_per_panel=1,
            polarization="single",
            polarization_type="V",
            antenna_pattern="omni",
            carrier_frequency=fc,
            precision=precision,
            device=device,
        )
        ut_array = tr38901.PanelArray(
            num_rows_per_panel=1,
            num_cols_per_panel=1,
            polarization="single",
            polarization_type="V",
            antenna_pattern="omni",
            carrier_frequency=fc,
            precision=precision,
            device=device,
        )
        scenario = tr38901.UMiScenario(
            fc, "low", ut_array, bs_array, "downlink", precision=precision, device=device
        )
        scenario.set_topology(
            ut_loc=torch.tensor([[[100.0, 0.0, 1.5]]], dtype=dtype, device=device),
            bs_loc=torch.tensor([[[0.0, 0.0, 10.0]]], dtype=dtype, device=device),
            ut_orientations=torch.zeros(1, 1, 3, dtype=dtype, device=device),
            bs_orientations=torch.zeros(1, 1, 3, dtype=dtype, device=device),
            ut_velocities=torch.zeros(1, 1, 3, dtype=dtype, device=device),
            in_state=torch.tensor([[False]], device=device),
            los=False,
        )

        def fake_normal(shape, dtype, device, generator=None):
            del generator
            return torch.zeros(shape, dtype=dtype, device=device)

        monkeypatch.setattr(rays_module, "normal", fake_normal)

        ray_sampler = RaysGenerator(scenario)
        ray_sampler.topology_updated_callback()
        weights = torch.full(
            (scenario.num_clusters_max,), 1e-9, dtype=dtype, device=device
        )
        weights[:3] = torch.tensor(
            [1.0, 0.5, 0.003], dtype=dtype, device=device
        )
        r_tau = scenario.get_param("rTau")[0, 0, 0]
        delay_spread = torch.tensor(1e-7, dtype=dtype, device=device)
        unscaled_delays = (
            -torch.log(weights)
            * r_tau
            * delay_spread
            / (r_tau - 1.0)
        ).reshape(1, 1, 1, -1)
        powers, powers_for_angles = ray_sampler._cluster_powers(
            delay_spread.reshape(1, 1, 1),
            torch.ones(1, 1, 1, dtype=dtype, device=device),
            unscaled_delays,
        )

        assert powers.shape == (1, 1, 1, scenario.num_clusters_max)
        normalized = weights / weights.sum()
        expected = torch.where(
            normalized >= normalized.max() * 10.0**-2.5,
            normalized,
            torch.zeros_like(normalized),
        )
        torch.testing.assert_close(powers[0, 0, 0], expected)
        assert powers.sum() < 1.0
        assert torch.count_nonzero(powers) == 2
        torch.testing.assert_close(powers, powers_for_angles)

    def test_los_cluster_pruning_uses_pre_k_powers(self, device, precision, monkeypatch):
        """Test LoS Step 6 pruning follows Eq. 7.5-6, before K-factor power."""
        import sionna.phy.channel.tr38901.rays as rays_module

        fc = self.CARRIER_FREQUENCY
        dtype = torch.float32 if precision == "single" else torch.float64
        bs_array = tr38901.PanelArray(
            num_rows_per_panel=1,
            num_cols_per_panel=1,
            polarization="single",
            polarization_type="V",
            antenna_pattern="omni",
            carrier_frequency=fc,
            precision=precision,
            device=device,
        )
        ut_array = tr38901.PanelArray(
            num_rows_per_panel=1,
            num_cols_per_panel=1,
            polarization="single",
            polarization_type="V",
            antenna_pattern="omni",
            carrier_frequency=fc,
            precision=precision,
            device=device,
        )
        scenario = tr38901.UMiScenario(
            fc, "low", ut_array, bs_array, "downlink", precision=precision, device=device
        )
        scenario.set_topology(
            ut_loc=torch.tensor([[[100.0, 0.0, 1.5]]], dtype=dtype, device=device),
            bs_loc=torch.tensor([[[0.0, 0.0, 10.0]]], dtype=dtype, device=device),
            ut_orientations=torch.zeros(1, 1, 3, dtype=dtype, device=device),
            bs_orientations=torch.zeros(1, 1, 3, dtype=dtype, device=device),
            ut_velocities=torch.zeros(1, 1, 3, dtype=dtype, device=device),
            in_state=torch.tensor([[False]], device=device),
            los=True,
        )

        def fake_normal(shape, dtype, device, generator=None):
            del generator
            return torch.zeros(shape, dtype=dtype, device=device)

        monkeypatch.setattr(rays_module, "normal", fake_normal)

        ray_sampler = RaysGenerator(scenario)
        ray_sampler.topology_updated_callback()
        unscaled_delays = torch.full(
            (1, 1, 1, scenario.num_clusters_max),
            3e-6,
            dtype=dtype,
            device=device,
        )
        unscaled_delays[:, :, :, :2] = 0.0
        powers, powers_for_angles = ray_sampler._cluster_powers(
            torch.full((1, 1, 1), 1e-7, dtype=dtype, device=device),
            torch.full((1, 1, 1), 1000.0, dtype=dtype, device=device),
            unscaled_delays,
        )

        assert torch.count_nonzero(powers) == 2
        assert powers[0, 0, 0, 0].item() == pytest.approx(0.5)
        assert powers[0, 0, 0, 1].item() == pytest.approx(0.5)
        assert powers_for_angles[0, 0, 0, 1] > 0.0

    @pytest.mark.parametrize("spec_version", ["16.1", "19.2"])
    def test_rma_o2i_cluster_mask_uses_indoor_parameters(
        self, device, precision, spec_version
    ):
        """Test supported RMa releases use the dedicated O2I cluster count."""
        fc = self.CARRIER_FREQUENCY
        dtype = torch.float32 if precision == "single" else torch.float64
        bs_array = tr38901.PanelArray(
            num_rows_per_panel=1,
            num_cols_per_panel=1,
            polarization="single",
            polarization_type="V",
            antenna_pattern="omni",
            carrier_frequency=fc,
            precision=precision,
            device=device,
        )
        ut_array = tr38901.PanelArray(
            num_rows_per_panel=1,
            num_cols_per_panel=1,
            polarization="single",
            polarization_type="V",
            antenna_pattern="omni",
            carrier_frequency=fc,
            precision=precision,
            device=device,
        )
        scenario = tr38901.RMaScenario(
            fc,
            ut_array,
            bs_array,
            "downlink",
            precision=precision,
            device=device,
            spec_version=spec_version,
        )
        scenario.set_topology(
            ut_loc=torch.tensor([[[100.0, 0.0, 1.5]]], dtype=dtype, device=device),
            bs_loc=torch.tensor([[[0.0, 0.0, 35.0]]], dtype=dtype, device=device),
            ut_orientations=torch.zeros(1, 1, 3, dtype=dtype, device=device),
            bs_orientations=torch.zeros(1, 1, 3, dtype=dtype, device=device),
            ut_velocities=torch.zeros(1, 1, 3, dtype=dtype, device=device),
            in_state=torch.tensor([[True]], device=device),
            los=True,
        )
        ray_sampler = RaysGenerator(scenario)
        ray_sampler.topology_updated_callback()

        expected = torch.cat(
            [
                torch.zeros(
                    scenario.num_clusters_indoor, dtype=dtype, device=device
                ),
                torch.ones(
                    scenario.num_clusters_max - scenario.num_clusters_indoor,
                    dtype=dtype,
                    device=device,
                ),
            ]
        ).reshape(1, 1, 1, scenario.num_clusters_max)
        torch.testing.assert_close(ray_sampler._cluster_mask, expected)

    @pytest.fixture(scope="class")
    @classmethod
    def ray_samples(cls, request):
        """Sample rays from all channel models for testing.

        Uses the device specified by --device flag.
        """
        device_option = request.config.getoption("--device", default="gpu")
        if device_option == "cpu":
            device = "cpu"
        elif device_option == "gpu" and torch.cuda.is_available():
            device = "cuda:0"
        elif device_option == "all" and torch.cuda.is_available():
            device = "cuda:0"  # Use GPU for class-scoped fixture when "all"
        else:
            device = "cpu"
        batch_size = cls.BATCH_SIZE
        fc = cls.CARRIER_FREQUENCY
        dtype = torch.float64

        # Create antenna arrays
        bs_array = tr38901.PanelArray(
            num_rows_per_panel=1,
            num_cols_per_panel=1,
            polarization="single",
            polarization_type="V",
            antenna_pattern="38.901",
            carrier_frequency=fc,
            precision="double",
            device=device,
        )
        ut_array = tr38901.PanelArray(
            num_rows_per_panel=1,
            num_cols_per_panel=1,
            polarization="single",
            polarization_type="V",
            antenna_pattern="38.901",
            carrier_frequency=fc,
            precision="double",
            device=device,
        )

        # Generate shared topology
        ut_orientations = torch.empty([batch_size, 1, 3], dtype=dtype, device=device).uniform_(
            -PI, PI, generator=config.torch_rng(device)
        )
        bs_orientations = torch.empty([batch_size, 1, 3], dtype=dtype, device=device).uniform_(
            -PI, PI, generator=config.torch_rng(device)
        )
        ut_velocities = torch.empty([batch_size, 1, 3], dtype=dtype, device=device).uniform_(
            -1.0, 1.0, generator=config.torch_rng(device)
        )

        ut_loc = generate_random_loc(
            batch_size, 1, (100, 2000), (100, 2000), (1.5, 1.5),
            share_loc=True, dtype=dtype, device=device
        )
        bs_loc = generate_random_loc(
            batch_size, 1, (0, 100), (0, 100), (35.0, 35.0),
            share_loc=True, dtype=dtype, device=device
        )

        # Force the LSPs to specific values
        ds = np.power(10.0, -7.49)
        asd = np.power(10.0, 0.90)
        asa = np.power(10.0, 1.52)
        zsa = np.power(10.0, 0.47)
        zsd = np.power(10.0, -0.29)
        k = np.power(10.0, 7.0 / 10.0)

        ds_ = torch.full([batch_size, 1, 1], ds, dtype=dtype, device=device)
        asd_ = torch.full([batch_size, 1, 1], asd, dtype=dtype, device=device)
        asa_ = torch.full([batch_size, 1, 1], asa, dtype=dtype, device=device)
        zsa_ = torch.full([batch_size, 1, 1], zsa, dtype=dtype, device=device)
        zsd_ = torch.full([batch_size, 1, 1], zsd, dtype=dtype, device=device)
        k_ = torch.full([batch_size, 1, 1], k, dtype=dtype, device=device)
        sf_ = torch.zeros([batch_size, 1, 1], dtype=dtype, device=device)

        lsp = LSP(ds_, asd_, asa_, sf_, k_, zsa_, zsd_)

        samples = {
            "ds": ds,
            "asd": asd,
            "asa": asa,
            "zsa": zsa,
            "zsd": zsd,
            "k": k,
        }

        # Test all scenarios
        for model_name, scenario_cls in [
            ("rma", tr38901.RMaScenario),
            ("umi", tr38901.UMiScenario),
            ("uma", tr38901.UMaScenario),
        ]:
            samples[model_name] = {}

            # Create scenario
            if model_name == "rma":
                scenario = scenario_cls(
                    fc, ut_array, bs_array, "downlink", precision="double", device=device
                )
            else:
                scenario = scenario_cls(
                    fc, "low", ut_array, bs_array, "downlink", precision="double", device=device
                )

            ray_sampler = RaysGenerator(scenario)

            for submodel, los_val, in_state_p in [
                ("los", True, 0.0),
                ("nlos", False, 0.0),
                ("o2i", False, 1.0),
            ]:
                in_state = generate_random_bool(batch_size, 1, in_state_p, device=device)
                if los_val is not None:
                    scenario.set_topology(
                        ut_loc, bs_loc, ut_orientations, bs_orientations,
                        ut_velocities, in_state, los=los_val
                    )
                else:
                    scenario.set_topology(
                        ut_loc, bs_loc, ut_orientations, bs_orientations,
                        ut_velocities, in_state
                    )
                ray_sampler.topology_updated_callback()
                rays = ray_sampler(lsp)

                samples[model_name][submodel] = {
                    "delays": rays.delays.squeeze().cpu().numpy(),
                    "powers": rays.powers.squeeze().cpu().numpy(),
                    "aoa": rays.aoa.squeeze().cpu().numpy(),
                    "aod": rays.aod.squeeze().cpu().numpy(),
                    "zoa": rays.zoa.squeeze().cpu().numpy(),
                    "zod": rays.zod.squeeze().cpu().numpy(),
                    "xpr": rays.xpr.squeeze().cpu().numpy(),
                    "los_aoa": scenario.los_aoa.cpu().numpy(),
                    "los_aod": scenario.los_aod.cpu().numpy(),
                    "los_zoa": scenario.los_zoa.cpu().numpy(),
                    "los_zod": scenario.los_zod.cpu().numpy(),
                    "mu_log_zsd": scenario.lsp_log_mean[:, 0, 0, 6].cpu().numpy(),
                }

            samples[model_name]["d_2d"] = scenario.distance_2d[0, 0, 0].cpu().numpy()

        return samples

    @pytest.mark.parametrize("model", ["rma", "umi", "uma"])
    @pytest.mark.parametrize("submodel", ["los", "nlos", "o2i"])
    def test_delays_distribution(self, ray_samples, model, submodel):
        """Test ray generation: Delays distribution."""
        num_clusters = self.NUM_CLUSTERS[model][submodel]
        batch_size = self.BATCH_SIZE

        tau = ray_samples[model][submodel]["delays"][:, :num_clusters].flatten()
        _, ref_tau = delays_ref(
            model, submodel, batch_size, num_clusters,
            ray_samples["ds"], ray_samples["k"]
        )
        ref_tau = ref_tau[:, :num_clusters].flatten()

        D, _ = kstest(tau, ref_tau)
        assert D <= self.MAX_ERR, f"{model}:{submodel} delays distribution failed"

    @pytest.mark.parametrize("model", ["rma", "umi", "uma"])
    @pytest.mark.parametrize("submodel", ["los", "nlos", "o2i"])
    def test_powers_distribution(self, ray_samples, model, submodel):
        """Test ray generation: Powers distribution."""
        num_clusters = self.NUM_CLUSTERS[model][submodel]
        batch_size = self.BATCH_SIZE

        p = ray_samples[model][submodel]["powers"][:, :num_clusters].flatten()
        unscaled_tau, _ = delays_ref(
            model, submodel, batch_size, num_clusters,
            ray_samples["ds"], ray_samples["k"]
        )
        ref_p, _ = powers_ref(
            model, submodel, batch_size, num_clusters,
            unscaled_tau, ray_samples["ds"], ray_samples["k"]
        )
        ref_p = ref_p[:, :num_clusters].flatten()

        D, _ = kstest(ref_p, p)
        assert D <= self.MAX_ERR, f"{model}:{submodel} powers distribution failed"

    @pytest.mark.parametrize("model", ["rma", "umi", "uma"])
    @pytest.mark.parametrize("submodel", ["los", "nlos", "o2i"])
    def test_xpr_distribution(self, ray_samples, model, submodel):
        """Test ray generation: XPR distribution."""
        num_clusters = self.NUM_CLUSTERS[model][submodel]
        batch_size = self.BATCH_SIZE

        samples = ray_samples[model][submodel]["xpr"][:, :num_clusters].flatten()
        ref_samples = xpr_ref(model, submodel, batch_size, num_clusters)
        ref_samples = ref_samples[:, :num_clusters].flatten()

        D, _ = kstest(ref_samples, samples)
        assert D <= self.MAX_ERR, f"{model}:{submodel} XPR distribution failed"

    @pytest.mark.parametrize("model", ["rma", "umi", "uma"])
    @pytest.mark.parametrize("submodel", ["los", "nlos", "o2i"])
    def test_delays_positive(self, ray_samples, model, submodel):
        """Test that all delays are non-negative."""
        delays = ray_samples[model][submodel]["delays"]
        assert np.all(delays >= 0), f"{model}:{submodel} delays should be non-negative"

    @pytest.mark.parametrize("model", ["rma", "umi", "uma"])
    @pytest.mark.parametrize("submodel", ["los", "nlos", "o2i"])
    def test_power_removed_by_pruning_is_not_renormalized(
        self, ray_samples, model, submodel
    ):
        """Check Step 6 retains the power loss from eliminated clusters."""
        powers = ray_samples[model][submodel]["powers"]
        num_clusters = self.NUM_CLUSTERS[model][submodel]
        power_sum = powers[:, :num_clusters].sum(axis=-1)
        max_removed_power = (num_clusters - 1) * 10.0**-2.5
        assert np.all(power_sum <= 1.0 + 1e-6)
        assert np.all(power_sum >= 1.0 - max_removed_power - 1e-6)

    @pytest.mark.parametrize("model", ["rma", "umi", "uma"])
    @pytest.mark.parametrize("submodel", ["los", "nlos", "o2i"])
    def test_xpr_positive(self, ray_samples, model, submodel):
        """Test that XPR values are positive."""
        xpr = ray_samples[model][submodel]["xpr"]
        assert np.all(xpr > 0), f"{model}:{submodel} XPR should be positive"

    @pytest.mark.parametrize("model", ["rma", "umi", "uma"])
    @pytest.mark.parametrize("submodel", ["los", "nlos", "o2i"])
    def test_zoa_in_range(self, ray_samples, model, submodel):
        """Test that ZoA values are in valid range (0 to pi)."""
        zoa = ray_samples[model][submodel]["zoa"]
        assert np.all(zoa >= 0), f"{model}:{submodel} ZoA should be >= 0"
        assert np.all(zoa <= np.pi), f"{model}:{submodel} ZoA should be <= pi"

    @pytest.mark.parametrize("model", ["rma", "umi", "uma"])
    @pytest.mark.parametrize("submodel", ["los", "nlos", "o2i"])
    def test_zod_in_range(self, ray_samples, model, submodel):
        """Test that ZoD values are in valid range (0 to pi)."""
        zod = ray_samples[model][submodel]["zod"]
        assert np.all(zod >= 0), f"{model}:{submodel} ZoD should be >= 0"
        assert np.all(zod <= np.pi), f"{model}:{submodel} ZoD should be <= pi"

    @pytest.mark.parametrize("model", ["rma", "umi", "uma"])
    @pytest.mark.parametrize("submodel", ["los", "nlos", "o2i"])
    def test_aoa_in_range(self, ray_samples, model, submodel):
        """Test that AoA values are in valid range (-pi to pi)."""
        aoa = ray_samples[model][submodel]["aoa"]
        assert np.all(aoa >= -np.pi), f"{model}:{submodel} AoA should be >= -pi"
        assert np.all(aoa <= np.pi), f"{model}:{submodel} AoA should be <= pi"

    @pytest.mark.parametrize("model", ["rma", "umi", "uma"])
    @pytest.mark.parametrize("submodel", ["los", "nlos", "o2i"])
    def test_aod_in_range(self, ray_samples, model, submodel):
        """Test that AoD values are in valid range (-pi to pi)."""
        aod = ray_samples[model][submodel]["aod"]
        assert np.all(aod >= -np.pi), f"{model}:{submodel} AoD should be >= -pi"
        assert np.all(aod <= np.pi), f"{model}:{submodel} AoD should be <= pi"

    def test_rays_output_shapes(self, device, precision):
        """Test that RaysGenerator produces correct output shapes."""
        batch_size = 10
        fc = 3.5e9
        dtype = torch.float64

        bs_array = tr38901.PanelArray(
            num_rows_per_panel=1,
            num_cols_per_panel=1,
            polarization="single",
            polarization_type="V",
            antenna_pattern="38.901",
            carrier_frequency=fc,
            precision="double",
            device=device,
        )
        ut_array = tr38901.PanelArray(
            num_rows_per_panel=1,
            num_cols_per_panel=1,
            polarization="single",
            polarization_type="V",
            antenna_pattern="38.901",
            carrier_frequency=fc,
            precision="double",
            device=device,
        )

        scenario = tr38901.UMiScenario(
            fc, "low", ut_array, bs_array, "downlink", precision="double", device=device
        )

        ut_loc = generate_random_loc(
            batch_size, 1, (100, 500), (100, 500), (1.5, 1.5),
            dtype=dtype, device=device
        )
        bs_loc = generate_random_loc(
            batch_size, 1, (0, 50), (0, 50), (10.0, 10.0),
            dtype=dtype, device=device
        )
        ut_orientations = torch.zeros(batch_size, 1, dtype=dtype, device=device)
        bs_orientations = torch.zeros(batch_size, 1, dtype=dtype, device=device)
        ut_velocities = torch.zeros(batch_size, 1, dtype=dtype, device=device)
        in_state = generate_random_bool(batch_size, 1, 0.0, device=device)

        scenario.set_topology(
            ut_loc, bs_loc, ut_orientations, bs_orientations,
            ut_velocities, in_state, los=True
        )

        ray_sampler = RaysGenerator(scenario)
        ray_sampler.topology_updated_callback()

        # Create LSP
        ds_ = torch.full([batch_size, 1, 1], 1e-7, dtype=dtype, device=device)
        asd_ = torch.full([batch_size, 1, 1], 10.0, dtype=dtype, device=device)
        asa_ = torch.full([batch_size, 1, 1], 10.0, dtype=dtype, device=device)
        zsa_ = torch.full([batch_size, 1, 1], 5.0, dtype=dtype, device=device)
        zsd_ = torch.full([batch_size, 1, 1], 5.0, dtype=dtype, device=device)
        k_ = torch.full([batch_size, 1, 1], 2.0, dtype=dtype, device=device)
        sf_ = torch.ones([batch_size, 1, 1], dtype=dtype, device=device)

        lsp = LSP(ds_, asd_, asa_, sf_, k_, zsa_, zsd_)
        rays = ray_sampler(lsp)

        # RaysGenerator outputs num_clusters_max for efficient batching
        # UMi has: LoS=12, NLoS=19, O2I=12, so max=19
        num_clusters = scenario.num_clusters_max
        num_rays = 20

        assert rays.delays.shape == (batch_size, 1, 1, num_clusters)
        assert rays.powers.shape == (batch_size, 1, 1, num_clusters)
        assert rays.aoa.shape == (batch_size, 1, 1, num_clusters, num_rays)
        assert rays.aod.shape == (batch_size, 1, 1, num_clusters, num_rays)
        assert rays.zoa.shape == (batch_size, 1, 1, num_clusters, num_rays)
        assert rays.zod.shape == (batch_size, 1, 1, num_clusters, num_rays)
        assert rays.xpr.shape == (batch_size, 1, 1, num_clusters, num_rays)
