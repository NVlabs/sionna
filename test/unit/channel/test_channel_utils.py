#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Unit tests for channel utility functions"""

import math
import numpy as np
import pytest
import torch

from sionna.phy import config, dtypes, PI
from sionna.phy.channel import (
    subcarrier_frequencies,
    time_frequency_vector,
    time_lag_discrete_time_channel,
    time_to_ofdm_channel,
    cir_to_ofdm_channel,
    cir_to_time_channel,
    deg_2_rad,
    rad_2_deg,
    wrap_angle_0_360,
    drop_uts_in_sector,
    set_3gpp_scenario_parameters,
    relocate_uts,
    gen_single_sector_topology,
    gen_single_sector_topology_interferers,
    exp_corr_mat,
    one_ring_corr_mat,
)
from sionna.phy.ofdm import ResourceGrid

# Import test utilities
import sys

sys.path.insert(0, str(__file__).rsplit("/", 1)[0])
from channel_test_utils import exp_corr_mat_numpy, one_ring_corr_numpy


class TestSubcarrierFrequencies:
    """Tests for subcarrier_frequencies function"""

    def test_even_subcarriers(self, precision, device):
        """Test with even number of subcarriers"""
        num_subcarriers = 64
        spacing = 15e3

        freqs = subcarrier_frequencies(
            num_subcarriers, spacing, precision=precision, device=device
        )

        assert freqs.shape == (num_subcarriers,)
        assert freqs.device.type == device.split(":")[0]
        # Check symmetry around zero
        assert torch.isclose(
            freqs[0], torch.tensor(-32 * spacing, device=device, dtype=freqs.dtype)
        )
        assert torch.isclose(
            freqs[-1], torch.tensor(31 * spacing, device=device, dtype=freqs.dtype)
        )
        # Zero frequency should be at index 32
        assert torch.isclose(
            freqs[32], torch.tensor(0.0, device=device, dtype=freqs.dtype)
        )

    def test_odd_subcarriers(self, precision, device):
        """Test with odd number of subcarriers"""
        num_subcarriers = 63
        spacing = 15e3

        freqs = subcarrier_frequencies(
            num_subcarriers, spacing, precision=precision, device=device
        )

        assert freqs.shape == (num_subcarriers,)
        # Check symmetry around zero
        assert torch.isclose(
            freqs[0], torch.tensor(-31 * spacing, device=device, dtype=freqs.dtype)
        )
        assert torch.isclose(
            freqs[-1], torch.tensor(31 * spacing, device=device, dtype=freqs.dtype)
        )
        # Zero frequency should be at the center
        assert torch.isclose(
            freqs[31], torch.tensor(0.0, device=device, dtype=freqs.dtype)
        )

    def test_example_docstring(self, device):
        """Verify the docstring example works correctly"""
        freqs = subcarrier_frequencies(64, 15e3, device=device)
        assert freqs.shape == torch.Size([64])


class TestTimeFrequencyVector:
    """Tests for time_frequency_vector function"""

    def test_even_samples(self, precision, device):
        """Test with even number of samples"""
        num_samples = 128
        sample_duration = 1e-6

        t, f = time_frequency_vector(
            num_samples, sample_duration, precision=precision, device=device
        )

        assert t.shape == (num_samples,)
        assert f.shape == (num_samples,)
        # Check that t and f are symmetric around zero
        assert torch.isclose(
            t[64], torch.tensor(0.0, device=device, dtype=t.dtype), atol=1e-12
        )

    def test_odd_samples(self, precision, device):
        """Test with odd number of samples"""
        num_samples = 127
        sample_duration = 1e-6

        t, f = time_frequency_vector(
            num_samples, sample_duration, precision=precision, device=device
        )

        assert t.shape == (num_samples,)
        assert f.shape == (num_samples,)
        # Zero should be at the center
        center_idx = num_samples // 2
        assert torch.isclose(
            t[center_idx], torch.tensor(0.0, device=device, dtype=t.dtype), atol=1e-12
        )

    def test_example_docstring(self, device):
        """Verify the docstring example works correctly"""
        t, f = time_frequency_vector(128, 1e-6, device=device)
        assert t.shape == torch.Size([128])
        assert f.shape == torch.Size([128])


class TestTimeLagDiscreteTimeChannel:
    """Tests for time_lag_discrete_time_channel function"""

    def test_default_values(self):
        """Test with default maximum delay spread"""
        l_min, l_max = time_lag_discrete_time_channel(20e6)

        assert l_min == -6
        # l_max = ceil(3e-6 * 20e6) + 6 = ceil(60) + 6 = 66
        assert l_max == 66

    def test_custom_delay_spread(self):
        """Test with custom maximum delay spread"""
        l_min, l_max = time_lag_discrete_time_channel(10e6, maximum_delay_spread=1e-6)

        assert l_min == -6
        # l_max = ceil(1e-6 * 10e6) + 6 = ceil(10) + 6 = 16
        assert l_max == 16

    def test_example_docstring(self):
        """Verify the docstring example works correctly"""
        l_min, l_max = time_lag_discrete_time_channel(20e6)
        assert l_min == -6
        assert l_max == 66


class TestDegRadConversions:
    """Tests for degree/radian conversion functions"""

    def test_deg_2_rad(self, device):
        """Test degree to radian conversion"""
        angle_deg = torch.tensor([0.0, 90.0, 180.0, 360.0], device=device)
        angle_rad = deg_2_rad(angle_deg)

        expected = torch.tensor([0.0, PI / 2, PI, 2 * PI], device=device)
        assert torch.allclose(angle_rad, expected, atol=1e-6)

    def test_rad_2_deg(self, device):
        """Test radian to degree conversion"""
        angle_rad = torch.tensor([0.0, PI / 2, PI, 2 * PI], device=device)
        angle_deg = rad_2_deg(angle_rad)

        expected = torch.tensor([0.0, 90.0, 180.0, 360.0], device=device)
        assert torch.allclose(angle_deg, expected, atol=1e-5)

    def test_roundtrip(self, device):
        """Test that deg->rad->deg is identity"""
        original = torch.tensor(
            [0.0, 45.0, 90.0, 135.0, 180.0, 270.0, 360.0], device=device
        )
        result = rad_2_deg(deg_2_rad(original))
        assert torch.allclose(result, original, atol=1e-5)


class TestWrapAngle:
    """Tests for wrap_angle_0_360 function"""

    def test_basic_cases(self, device):
        """Test basic wrapping cases"""
        angles = torch.tensor([-90.0, 0.0, 450.0, 720.0], device=device)
        wrapped = wrap_angle_0_360(angles)

        expected = torch.tensor([270.0, 0.0, 90.0, 0.0], device=device)
        assert torch.allclose(wrapped, expected, atol=1e-5)

    def test_negative_angles(self, device):
        """Test wrapping of negative angles"""
        angles = torch.tensor([-360.0, -180.0, -90.0, -45.0], device=device)
        wrapped = wrap_angle_0_360(angles)

        expected = torch.tensor([0.0, 180.0, 270.0, 315.0], device=device)
        assert torch.allclose(wrapped, expected, atol=1e-5)


class TestCirToOfdmChannel:
    """Tests for cir_to_ofdm_channel function"""

    def test_output_shape(self, device):
        """Test that output has correct shape"""
        batch_size = 2
        num_paths = 4
        num_time_steps = 10
        fft_size = 64

        a = torch.randn(
            batch_size,
            1,
            1,
            1,
            1,
            num_paths,
            num_time_steps,
            dtype=torch.complex64,
            device=device,
        )
        tau = torch.rand(batch_size, 1, 1, num_paths, device=device) * 1e-6
        frequencies = subcarrier_frequencies(fft_size, 15e3, device=device)

        h_f = cir_to_ofdm_channel(frequencies, a, tau)

        assert h_f.shape == (batch_size, 1, 1, 1, 1, num_time_steps, fft_size)

    def test_normalized_channel(self, device):
        """Test that normalized channel has unit average energy"""
        batch_size = 4
        num_paths = 8
        num_time_steps = 5
        fft_size = 32

        a = torch.randn(
            batch_size,
            1,
            2,
            1,
            2,
            num_paths,
            num_time_steps,
            dtype=torch.complex64,
            device=device,
        )
        tau = torch.rand(batch_size, 1, 1, num_paths, device=device) * 1e-6
        frequencies = subcarrier_frequencies(fft_size, 15e3, device=device)

        h_f = cir_to_ofdm_channel(frequencies, a, tau, normalize=True)

        # Check that average power per resource grid is approximately 1
        avg_power = h_f.abs().square().mean(dim=(2, 4, 5, 6))
        assert torch.allclose(avg_power, torch.ones_like(avg_power), atol=0.1)


class TestCirToTimeChannel:
    """Tests for cir_to_time_channel function"""

    def test_output_shape(self, device):
        """Test that output has correct shape"""
        batch_size = 2
        num_paths = 4
        num_time_steps = 10
        l_min, l_max = -6, 20

        a = torch.randn(
            batch_size,
            1,
            1,
            1,
            1,
            num_paths,
            num_time_steps,
            dtype=torch.complex64,
            device=device,
        )
        tau = torch.rand(batch_size, 1, 1, num_paths, device=device) * 1e-6

        h_t = cir_to_time_channel(20e6, a, tau, l_min=l_min, l_max=l_max)

        expected_taps = l_max - l_min + 1
        assert h_t.shape == (batch_size, 1, 1, 1, 1, num_time_steps, expected_taps)


class TestDropUtsInSector:
    """Tests for drop_uts_in_sector function"""

    def test_output_shape(self, precision, device):
        """Test that output has correct shape"""
        batch_size = 8
        num_ut = 4
        min_bs_ut_dist = 35.0
        isd = 500.0

        ut_loc = drop_uts_in_sector(
            batch_size, num_ut, min_bs_ut_dist, isd, precision=precision, device=device
        )

        assert ut_loc.shape == (batch_size, num_ut, 2)
        assert ut_loc.device.type == device.split(":")[0]

    def test_distance_constraint(self, device):
        """Test that UTs are at least min_bs_ut_dist away from BS"""
        batch_size = 100
        num_ut = 10
        min_bs_ut_dist = 35.0
        isd = 500.0

        ut_loc = drop_uts_in_sector(
            batch_size, num_ut, min_bs_ut_dist, isd, device=device
        )

        # Compute distances from origin
        distances = torch.sqrt(ut_loc[..., 0] ** 2 + ut_loc[..., 1] ** 2)

        # All distances should be >= min_bs_ut_dist
        assert torch.all(
            distances >= min_bs_ut_dist - 0.1
        )  # Small tolerance for numerical errors


class TestSet3gppScenarioParameters:
    """Tests for set_3gpp_scenario_parameters function"""

    @pytest.mark.parametrize(
        ("scenario", "expected_min_bs_ut_dist"),
        [
            ("umi", 10.0),
            ("umi-calibration", 10.0),
            ("uma", 35.0),
            ("uma-calibration", 35.0),
        ],
    )
    def test_umi_uma_default_min_bs_ut_dist(
        self, scenario, expected_min_bs_ut_dist, device
    ):
        """Test Table 7.8-1 minimum BS-UT distance defaults."""
        min_bs_ut_dist, *_ = set_3gpp_scenario_parameters(scenario, device=device)

        assert min_bs_ut_dist.item() == expected_min_bs_ut_dist

    @pytest.mark.parametrize(
        "scenario", ["umi", "uma", "rma", "umi-calibration", "uma-calibration"]
    )
    def test_all_scenarios(self, scenario, device):
        """Test that all scenarios return valid parameters"""
        params = set_3gpp_scenario_parameters(scenario, device=device)

        assert len(params) == 8
        for p in params:
            assert isinstance(p, torch.Tensor)

    def test_custom_parameters(self, device):
        """Test that custom parameters override defaults"""
        custom_isd = 1000.0
        custom_bs_height = 50.0

        params = set_3gpp_scenario_parameters(
            "umi", isd=custom_isd, bs_height=custom_bs_height, device=device
        )

        _, isd, bs_height, _, _, _, _, _ = params
        assert torch.isclose(isd, torch.tensor(custom_isd, device=device))
        assert torch.isclose(bs_height, torch.tensor(custom_bs_height, device=device))


class TestRelocateUts:
    """Tests for relocate_uts function"""

    def test_no_rotation_no_translation(self, device):
        """Test that sector_id=0 and cell_loc=0 returns same locations"""
        batch_size = 4
        num_ut = 3

        ut_loc = torch.rand(batch_size, num_ut, 2, device=device)
        sector_id = torch.tensor(0, device=device)
        cell_loc = torch.zeros(2, device=device)

        relocated = relocate_uts(ut_loc, sector_id, cell_loc)

        assert torch.allclose(relocated, ut_loc, atol=1e-5)

    def test_translation_only(self, device):
        """Test that translation works correctly"""
        batch_size = 2
        num_ut = 2

        ut_loc = torch.tensor(
            [[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]], device=device
        )
        sector_id = torch.tensor(0, device=device)
        cell_loc = torch.tensor([10.0, 20.0], device=device)

        relocated = relocate_uts(ut_loc, sector_id, cell_loc)

        expected = ut_loc + cell_loc
        assert torch.allclose(relocated, expected, atol=1e-5)


class TestGenSingleSectorTopology:
    """Tests for gen_single_sector_topology function"""

    def test_output_shapes(self, device):
        """Test that outputs have correct shapes"""
        batch_size = 4
        num_ut = 3

        result = gen_single_sector_topology(batch_size, num_ut, "umi", device=device)
        ut_loc, bs_loc, ut_orientations, bs_orientations, ut_velocities, in_state = (
            result
        )

        assert ut_loc.shape == (batch_size, num_ut, 3)
        assert bs_loc.shape == (batch_size, 1, 3)
        assert ut_orientations.shape == (batch_size, num_ut, 3)
        assert bs_orientations.shape == (batch_size, 1, 3)
        assert ut_velocities.shape == (batch_size, num_ut, 3)
        assert in_state.shape == (batch_size, num_ut)

    def test_bs_at_origin(self, device):
        """Test that BS is located at origin (x=0, y=0)"""
        result = gen_single_sector_topology(4, 3, "umi", device=device)
        bs_loc = result[1]

        assert torch.allclose(bs_loc[..., 0], torch.zeros_like(bs_loc[..., 0]))
        assert torch.allclose(bs_loc[..., 1], torch.zeros_like(bs_loc[..., 1]))


class TestGenSingleSectorTopologyInterferers:
    """Tests for gen_single_sector_topology_interferers function"""

    def test_output_shapes(self, device):
        """Test that outputs have correct shapes"""
        batch_size = 4
        num_ut = 3
        num_interferer = 2

        result = gen_single_sector_topology_interferers(
            batch_size, num_ut, num_interferer, "umi", device=device
        )
        ut_loc, bs_loc, ut_orientations, bs_orientations, ut_velocities, in_state = (
            result
        )

        total_uts = num_ut + num_interferer
        assert ut_loc.shape == (batch_size, total_uts, 3)
        assert bs_loc.shape == (batch_size, 1, 3)
        assert ut_orientations.shape == (batch_size, total_uts, 3)
        assert bs_orientations.shape == (batch_size, 1, 3)
        assert ut_velocities.shape == (batch_size, total_uts, 3)
        assert in_state.shape == (batch_size, total_uts)


class TestExpCorrMat:
    """Tests for exp_corr_mat function (ported from TensorFlow tests)"""

    def test_single_dim(self, precision, device):
        """Tests for scalar inputs"""
        values = [0.0, 0.9999, 0.5 + 0.3j]
        dims = [1, 2, 4, 7, 64]

        for a in values:
            for n in dims:
                R1 = exp_corr_mat_numpy(a, n, precision)
                R2 = exp_corr_mat(torch.tensor(a), n, precision, device)
                R1 = R1.to(device)
                err = torch.max(torch.abs(R1 - R2))
                assert err < 1e-5, f"Failed for a={a}, n={n}, err={err}"

    def test_catch_abs_val_error(self, device):
        """Absolute value of a greater than 1 should raise error"""
        with pytest.raises(ValueError):
            exp_corr_mat(torch.tensor(1.1 + 0.3j), 12, device=device)

    def test_multiple_dims(self, device):
        """Test with batched input"""
        values = torch.rand(2, 4, 3, device=device) * 0.9
        n = 11
        precision = "double"

        R2 = exp_corr_mat(values, n, precision, device)
        R2_flat = R2.reshape(-1, n, n)

        for i, a in enumerate(values.flatten().tolist()):
            R1 = exp_corr_mat_numpy(a, n, precision)
            R1 = R1.to(device)
            err = torch.max(torch.abs(R1 - R2_flat[i]))
            assert err < 1e-10, f"Failed for a={a}, err={err}"

    def test_example_docstring(self, device):
        """Verify the docstring example works correctly"""
        R = exp_corr_mat(torch.tensor(0.9 + 0.1j), 4, device=device)
        assert R.shape == torch.Size([4, 4])

        R = exp_corr_mat(torch.rand(2, 3) * 0.9, 4, device=device)
        assert R.shape == torch.Size([2, 3, 4, 4])


class TestCompileCompatibility:
    """Tests to verify functions are compatible with torch.compile"""

    @pytest.mark.parametrize("backend", ["inductor", "eager"])
    def test_subcarrier_frequencies_compile(self, device, backend):
        """Test that subcarrier_frequencies can be compiled"""
        compiled_fn = torch.compile(subcarrier_frequencies, backend=backend)
        result = compiled_fn(64, 15e3, device=device)
        expected = subcarrier_frequencies(64, 15e3, device=device)
        assert torch.allclose(result, expected)

    @pytest.mark.parametrize("backend", ["inductor", "eager"])
    def test_time_frequency_vector_compile(self, device, backend):
        """Test that time_frequency_vector can be compiled"""
        compiled_fn = torch.compile(time_frequency_vector, backend=backend)
        t, f = compiled_fn(128, 1e-6, device=device)
        t_exp, f_exp = time_frequency_vector(128, 1e-6, device=device)
        assert torch.allclose(t, t_exp)
        assert torch.allclose(f, f_exp)

    @pytest.mark.parametrize("backend", ["inductor", "eager"])
    def test_deg_2_rad_compile(self, device, backend):
        """Test that deg_2_rad can be compiled"""
        compiled_fn = torch.compile(deg_2_rad, backend=backend)
        x = torch.tensor([0.0, 90.0, 180.0], device=device)
        result = compiled_fn(x)
        expected = deg_2_rad(x)
        assert torch.allclose(result, expected)

    @pytest.mark.parametrize("backend", ["inductor", "eager"])
    def test_rad_2_deg_compile(self, device, backend):
        """Test that rad_2_deg can be compiled"""
        compiled_fn = torch.compile(rad_2_deg, backend=backend)
        x = torch.tensor([0.0, PI / 2, PI], device=device)
        result = compiled_fn(x)
        expected = rad_2_deg(x)
        assert torch.allclose(result, expected)

    @pytest.mark.parametrize("backend", ["inductor", "eager"])
    def test_wrap_angle_compile(self, device, backend):
        """Test that wrap_angle_0_360 can be compiled"""
        compiled_fn = torch.compile(wrap_angle_0_360, backend=backend)
        x = torch.tensor([-90.0, 450.0, 720.0], device=device)
        result = compiled_fn(x)
        expected = wrap_angle_0_360(x)
        assert torch.allclose(result, expected)

    @pytest.mark.parametrize("backend", ["inductor", "eager"])
    def test_cir_to_ofdm_channel_compile(self, device, backend):
        """Test that cir_to_ofdm_channel can be compiled"""
        compiled_fn = torch.compile(cir_to_ofdm_channel, backend=backend)

        batch_size, num_paths, num_time_steps, fft_size = 2, 4, 5, 32
        a = torch.randn(
            batch_size,
            1,
            1,
            1,
            1,
            num_paths,
            num_time_steps,
            dtype=torch.complex64,
            device=device,
        )
        tau = torch.rand(batch_size, 1, 1, num_paths, device=device) * 1e-6
        frequencies = subcarrier_frequencies(fft_size, 15e3, device=device)

        result = compiled_fn(frequencies, a, tau)
        expected = cir_to_ofdm_channel(frequencies, a, tau)
        assert torch.allclose(result, expected, rtol=1e-4, atol=1e-6)

    @pytest.mark.parametrize("backend", ["inductor", "eager"])
    def test_cir_to_time_channel_compile(self, device, backend):
        """Test that cir_to_time_channel can be compiled"""
        compiled_fn = torch.compile(cir_to_time_channel, backend=backend)

        batch_size, num_paths, num_time_steps = 2, 4, 5
        a = torch.randn(
            batch_size,
            1,
            1,
            1,
            1,
            num_paths,
            num_time_steps,
            dtype=torch.complex64,
            device=device,
        )
        tau = torch.rand(batch_size, 1, 1, num_paths, device=device) * 1e-6

        result = compiled_fn(20e6, a, tau, -6, 20)
        expected = cir_to_time_channel(20e6, a, tau, -6, 20)
        assert torch.allclose(result, expected, rtol=1e-4, atol=1e-6)

    @pytest.mark.parametrize("backend", ["inductor", "eager"])
    def test_exp_corr_mat_compile(self, device, backend):
        """Test that exp_corr_mat can be compiled"""
        # Create a wrapper since exp_corr_mat has non-tensor args
        def wrapper(a):
            return exp_corr_mat(a, 4, device=device)

        compiled_fn = torch.compile(wrapper, backend=backend)
        a = torch.tensor(0.5 + 0.2j, device=device)
        result = compiled_fn(a)
        expected = wrapper(a)
        assert torch.allclose(result, expected)

    @pytest.mark.parametrize("backend", ["inductor", "eager"])
    def test_one_ring_corr_mat_compile(self, device, backend):
        """Test that one_ring_corr_mat can be compiled"""
        # Create a wrapper since one_ring_corr_mat has non-tensor args
        def wrapper(phi_deg):
            return one_ring_corr_mat(phi_deg, 4, device=device)

        compiled_fn = torch.compile(wrapper, backend=backend)
        phi_deg = torch.tensor(45.0, device=device)
        result = compiled_fn(phi_deg)
        expected = wrapper(phi_deg)
        assert torch.allclose(result, expected, rtol=1e-4, atol=1e-6)

    @pytest.mark.parametrize("backend", ["inductor", "eager"])
    def test_relocate_uts_compile(self, device, backend):
        """Test that relocate_uts can be compiled"""
        compiled_fn = torch.compile(relocate_uts, backend=backend)

        ut_loc = torch.rand(4, 3, 2, device=device)
        sector_id = torch.tensor(1, device=device)
        cell_loc = torch.tensor([10.0, 20.0], device=device)

        result = compiled_fn(ut_loc, sector_id, cell_loc)
        expected = relocate_uts(ut_loc, sector_id, cell_loc)
        assert torch.allclose(result, expected, rtol=1e-4, atol=1e-6)

    def test_gen_single_sector_topology_compile(self, device, mode):
        """Test that gen_single_sector_topology can be compiled"""
        compiled_fn = torch.compile(gen_single_sector_topology, mode=mode)
        result = compiled_fn(batch_size=4, num_ut=2, scenario="umi", device=device)
        # Should return 6 tensors: ut_loc, bs_loc, ut_orientations, bs_orientations, ut_velocities, in_state
        assert len(result) == 6
        ut_loc, bs_loc, ut_orientations, bs_orientations, ut_velocities, in_state = result
        assert ut_loc.shape == (4, 2, 3)
        assert bs_loc.shape == (4, 1, 3)

    def test_gen_single_sector_topology_interferers_compile(self, device, mode):
        """Test that gen_single_sector_topology_interferers can be compiled"""
        compiled_fn = torch.compile(gen_single_sector_topology_interferers, mode=mode)
        result = compiled_fn(batch_size=4, num_ut=2, num_interferer=3, scenario="umi", device=device)
        # Should return 6 tensors
        assert len(result) == 6
        ut_loc, bs_loc, ut_orientations, bs_orientations, ut_velocities, in_state = result
        # Total UTs = num_ut + num_interferer = 5
        assert ut_loc.shape == (4, 5, 3)
        assert bs_loc.shape == (4, 1, 3)


class TestOneRingCorrMat:
    """Tests for one_ring_corr_mat function (ported from TensorFlow tests)"""

    def test_single_dim(self, device):
        """Tests for scalar inputs"""
        phi_degs = [-180, -90, 0, 45, 90, 180]
        num_ants = [1, 4, 16]
        d_hs = [0.2, 0.5, 1.0]
        sigma_phi_degs = [2, 5, 15]

        for phi_deg in phi_degs:
            for num_ant in num_ants:
                for d_h in d_hs:
                    for sigma_phi_deg in sigma_phi_degs:
                        R1 = one_ring_corr_numpy(phi_deg, num_ant, d_h, sigma_phi_deg)
                        R2 = one_ring_corr_mat(
                            torch.tensor(float(phi_deg)),
                            num_ant,
                            d_h,
                            sigma_phi_deg,
                            precision="double",
                            device=device,
                        )
                        R1_tensor = torch.tensor(R1, device=device)
                        assert torch.allclose(
                            R1_tensor, R2, rtol=1e-4, atol=1e-10
                        ), f"Failed for phi_deg={phi_deg}, num_ant={num_ant}, d_h={d_h}, sigma={sigma_phi_deg}"

    def test_multiple_dims(self, device):
        """Test with batched input"""
        phi_degs = torch.rand(2, 4, 3, device=device) * 360 - 180
        num_ant = 8
        d_h = 0.7
        sigma_phi_deg = 10

        R2 = one_ring_corr_mat(
            phi_degs, num_ant, d_h, sigma_phi_deg, precision="double", device=device
        )
        R2_flat = R2.reshape(-1, num_ant, num_ant)

        for i, phi_deg in enumerate(phi_degs.flatten().tolist()):
            R1 = one_ring_corr_numpy(phi_deg, num_ant, d_h, sigma_phi_deg)
            R1_tensor = torch.tensor(R1, device=device)
            assert torch.allclose(R1_tensor, R2_flat[i], rtol=1e-4, atol=1e-10)

    def test_warning_large_asd(self, device):
        """Should warn when sigma_phi_deg > 15"""
        with pytest.warns(UserWarning):
            one_ring_corr_mat(torch.tensor(35.0), 32, 0.7, 16, device=device)

    def test_example_docstring(self, device):
        """Verify the docstring example works correctly"""
        R = one_ring_corr_mat(torch.tensor(45.0), 4, device=device)
        assert R.shape == torch.Size([4, 4])

        R = one_ring_corr_mat(torch.rand(2, 3) * 180 - 90, 4, device=device)
        assert R.shape == torch.Size([2, 3, 4, 4])


class TestTimeToOfdmChannel:
    """Regression tests for time_to_ofdm_channel output contract."""

    def test_output_fft_size_when_padded(self, device):
        """Shorter impulse responses are zero-padded to fft_size."""
        rg = ResourceGrid(
            num_ofdm_symbols=1, fft_size=8, subcarrier_spacing=15e3, device=device
        )
        h_t = torch.randn(
            1, rg.num_time_samples, 5, dtype=torch.complex64, device=device
        )
        h_f = time_to_ofdm_channel(h_t, rg, l_min=0)
        assert h_f.shape[-1] == rg.fft_size

    def test_rejects_impulse_longer_than_fft(self, device):
        """Longer impulse responses must raise instead of returning wrong shape."""
        rg = ResourceGrid(
            num_ofdm_symbols=1, fft_size=4, subcarrier_spacing=15e3, device=device
        )
        h_t = torch.randn(
            1, rg.num_time_samples, 6, dtype=torch.complex64, device=device
        )
        with pytest.raises(ValueError, match="must not exceed rg.fft_size"):
            time_to_ofdm_channel(h_t, rg, l_min=0)
