#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Tests for 3GPP TR 38.901 system level channel scenarios (RMa, UMa, UMi)"""

import numpy as np
import pytest
import torch

from sionna.phy.channel.tr38901 import (
    PanelArray,
    RMaScenario,
    RMa,
    UMaScenario,
    UMa,
    UMiScenario,
    UMi,
)
from channel_test_utils import generate_random_loc, generate_random_bool


# Test configuration
BATCH_SIZE = 16
CARRIER_FREQUENCY = 3.5e9  # Hz
MAX_ERR = 1e-2
H_UT = 1.5
H_BS = 10.0
NB_BS = 3
NB_UT = 10


def create_arrays(fc, device, precision="single"):
    """Create UT and BS panel arrays for testing."""
    bs_array = PanelArray(
        num_rows_per_panel=2,
        num_cols_per_panel=2,
        polarization="dual",
        polarization_type="VH",
        antenna_pattern="38.901",
        carrier_frequency=fc,
        precision=precision,
        device=device,
    )
    ut_array = PanelArray(
        num_rows_per_panel=1,
        num_cols_per_panel=1,
        polarization="dual",
        polarization_type="VH",
        antenna_pattern="38.901",
        carrier_frequency=fc,
        precision=precision,
        device=device,
    )
    return ut_array, bs_array


class TestRMaScenario:
    """Tests for RMaScenario"""

    def test_instantiation(self, device, precision):
        """Test RMaScenario can be instantiated"""
        ut_array, bs_array = create_arrays(CARRIER_FREQUENCY, device, precision)
        scenario = RMaScenario(
            carrier_frequency=CARRIER_FREQUENCY,
            ut_array=ut_array,
            bs_array=bs_array,
            direction="uplink",
            precision=precision,
            device=device,
        )
        assert scenario is not None
        assert scenario.carrier_frequency.item() == pytest.approx(CARRIER_FREQUENCY)

    def test_distance_calculation(self, device, precision):
        """Test distance calculations (2D and 3D)"""
        ut_array, bs_array = create_arrays(CARRIER_FREQUENCY, device, precision)
        scenario = RMaScenario(
            carrier_frequency=CARRIER_FREQUENCY,
            ut_array=ut_array,
            bs_array=bs_array,
            direction="uplink",
            precision=precision,
            device=device,
        )

        dtype = torch.float32 if precision == "single" else torch.float64

        # Generate random locations
        ut_loc = generate_random_loc(
            BATCH_SIZE, NB_UT, (100, 2000), (100, 2000), (H_UT, H_UT),
            dtype=dtype, device=device
        )
        bs_loc = generate_random_loc(
            BATCH_SIZE, NB_BS, (0, 100), (0, 100), (H_BS, H_BS),
            dtype=dtype, device=device
        )
        ut_orientations = torch.zeros(BATCH_SIZE, NB_UT, 3, dtype=dtype, device=device)
        bs_orientations = torch.zeros(BATCH_SIZE, NB_BS, 3, dtype=dtype, device=device)
        ut_velocities = torch.zeros(BATCH_SIZE, NB_UT, 3, dtype=dtype, device=device)
        in_state = generate_random_bool(BATCH_SIZE, NB_UT, 0.5, device=device)

        scenario.set_topology(
            ut_loc, bs_loc, ut_orientations, bs_orientations, ut_velocities, in_state
        )

        # Get distances
        d_3d = scenario.distance_3d
        d_3d_in = scenario.distance_3d_in
        d_3d_out = scenario.distance_3d_out
        d_2d = scenario.distance_2d
        d_2d_in = scenario.distance_2d_in
        d_2d_out = scenario.distance_2d_out

        # Verify total 3D distances
        ut_loc_expanded = ut_loc.unsqueeze(1)
        bs_loc_expanded = bs_loc.unsqueeze(2)
        d_3d_ref = torch.sqrt(((ut_loc_expanded - bs_loc_expanded) ** 2).sum(dim=3))
        max_err = torch.max(torch.abs(d_3d - d_3d_ref) / d_3d_ref)
        assert max_err <= MAX_ERR

        # Verify 3D indoor + outdoor = total
        max_err = torch.max(torch.abs(d_3d - d_3d_in - d_3d_out) / d_3d)
        assert max_err <= MAX_ERR

        # Verify total 2D distances
        d_2d_ref = torch.sqrt(
            ((ut_loc_expanded[:, :, :, :2] - bs_loc_expanded[:, :, :, :2]) ** 2).sum(dim=3)
        )
        max_err = torch.max(torch.abs(d_2d - d_2d_ref) / d_2d_ref)
        assert max_err <= MAX_ERR

        # Verify 2D indoor + outdoor = total
        max_err = torch.max(torch.abs(d_2d - d_2d_in - d_2d_out) / d_2d)
        assert max_err <= MAX_ERR

    def test_get_param(self, device, precision):
        """Test the get_param() function retrieves correct values"""
        ut_array, bs_array = create_arrays(CARRIER_FREQUENCY, device, precision)
        scenario = RMaScenario(
            carrier_frequency=CARRIER_FREQUENCY,
            ut_array=ut_array,
            bs_array=bs_array,
            direction="uplink",
            precision=precision,
            device=device,
        )

        dtype = torch.float32 if precision == "single" else torch.float64

        ut_loc = generate_random_loc(
            BATCH_SIZE, NB_UT, (100, 2000), (100, 2000), (H_UT, H_UT),
            dtype=dtype, device=device
        )
        bs_loc = generate_random_loc(
            BATCH_SIZE, NB_BS, (0, 100), (0, 100), (H_BS, H_BS),
            dtype=dtype, device=device
        )
        ut_orientations = torch.zeros(BATCH_SIZE, NB_UT, 3, dtype=dtype, device=device)
        bs_orientations = torch.zeros(BATCH_SIZE, NB_BS, 3, dtype=dtype, device=device)
        ut_velocities = torch.zeros(BATCH_SIZE, NB_UT, 3, dtype=dtype, device=device)
        in_state = generate_random_bool(BATCH_SIZE, NB_UT, 0.5, device=device)

        scenario.set_topology(
            ut_loc, bs_loc, ut_orientations, bs_orientations, ut_velocities, in_state
        )

        # Test that muDSc is correctly extracted (RMa-specific values)
        param_tensor = scenario.get_param("muDSc")

        # Build reference tensor
        indoor = scenario.indoor.unsqueeze(1).cpu().numpy()
        indoor = np.tile(indoor, [1, NB_BS, 1])
        los = scenario.los.cpu().numpy()

        param_tensor_ref = np.zeros([BATCH_SIZE, NB_BS, NB_UT])
        param_tensor_ref[np.where(los)] = -7.49  # LoS value
        param_tensor_ref[np.where(np.logical_and(~los, ~indoor))] = -7.43  # NLoS value
        param_tensor_ref[np.where(indoor)] = -7.47  # O2I value

        max_err = np.max(np.abs(param_tensor.cpu().numpy() - param_tensor_ref))
        assert max_err <= 1e-5

    def test_los_probability_bounds(self, device, precision):
        """Test that LoS probability is between 0 and 1"""
        ut_array, bs_array = create_arrays(CARRIER_FREQUENCY, device, precision)
        scenario = RMaScenario(
            carrier_frequency=CARRIER_FREQUENCY,
            ut_array=ut_array,
            bs_array=bs_array,
            direction="uplink",
            precision=precision,
            device=device,
        )

        dtype = torch.float32 if precision == "single" else torch.float64

        ut_loc = generate_random_loc(
            BATCH_SIZE, NB_UT, (100, 2000), (100, 2000), (H_UT, H_UT),
            dtype=dtype, device=device
        )
        bs_loc = generate_random_loc(
            BATCH_SIZE, NB_BS, (0, 100), (0, 100), (H_BS, H_BS),
            dtype=dtype, device=device
        )
        ut_orientations = torch.zeros(BATCH_SIZE, NB_UT, 3, dtype=dtype, device=device)
        bs_orientations = torch.zeros(BATCH_SIZE, NB_BS, 3, dtype=dtype, device=device)
        ut_velocities = torch.zeros(BATCH_SIZE, NB_UT, 3, dtype=dtype, device=device)
        in_state = generate_random_bool(BATCH_SIZE, NB_UT, 0.5, device=device)

        scenario.set_topology(
            ut_loc, bs_loc, ut_orientations, bs_orientations, ut_velocities, in_state
        )

        los_prob = scenario.los_probability
        assert torch.all(los_prob >= 0.0)
        assert torch.all(los_prob <= 1.0)


class TestUMaScenario:
    """Tests for UMaScenario"""

    def test_instantiation(self, device, precision):
        """Test UMaScenario can be instantiated"""
        ut_array, bs_array = create_arrays(CARRIER_FREQUENCY, device, precision)
        scenario = UMaScenario(
            carrier_frequency=CARRIER_FREQUENCY,
            o2i_model="low",
            ut_array=ut_array,
            bs_array=bs_array,
            direction="uplink",
            precision=precision,
            device=device,
        )
        assert scenario is not None
        assert scenario.carrier_frequency.item() == pytest.approx(CARRIER_FREQUENCY)

    def test_o2i_model_validation(self, device, precision):
        """Test that o2i_model must be 'low' or 'high'"""
        ut_array, bs_array = create_arrays(CARRIER_FREQUENCY, device, precision)

        with pytest.raises(AssertionError):
            UMaScenario(
                carrier_frequency=CARRIER_FREQUENCY,
                o2i_model="invalid",
                ut_array=ut_array,
                bs_array=bs_array,
                direction="uplink",
                precision=precision,
                device=device,
            )

    def test_los_probability_bounds(self, device, precision):
        """Test that LoS probability is between 0 and 1"""
        ut_array, bs_array = create_arrays(CARRIER_FREQUENCY, device, precision)
        scenario = UMaScenario(
            carrier_frequency=CARRIER_FREQUENCY,
            o2i_model="low",
            ut_array=ut_array,
            bs_array=bs_array,
            direction="uplink",
            precision=precision,
            device=device,
        )

        dtype = torch.float32 if precision == "single" else torch.float64

        ut_loc = generate_random_loc(
            BATCH_SIZE, NB_UT, (100, 2000), (100, 2000), (H_UT, H_UT),
            dtype=dtype, device=device
        )
        bs_loc = generate_random_loc(
            BATCH_SIZE, NB_BS, (0, 100), (0, 100), (H_BS, H_BS),
            dtype=dtype, device=device
        )
        ut_orientations = torch.zeros(BATCH_SIZE, NB_UT, 3, dtype=dtype, device=device)
        bs_orientations = torch.zeros(BATCH_SIZE, NB_BS, 3, dtype=dtype, device=device)
        ut_velocities = torch.zeros(BATCH_SIZE, NB_UT, 3, dtype=dtype, device=device)
        in_state = generate_random_bool(BATCH_SIZE, NB_UT, 0.5, device=device)

        scenario.set_topology(
            ut_loc, bs_loc, ut_orientations, bs_orientations, ut_velocities, in_state
        )

        los_prob = scenario.los_probability
        assert torch.all(los_prob >= 0.0)
        assert torch.all(los_prob <= 1.0)

    def test_h_e_discrete_values(self, device, precision):
        """Test that h_E effective environment height is sampled from the correct
        discrete set {12, 15, 18, ...} per TR 38.901 Table 7.4.1-1 Note 1,
        not from a continuous distribution."""
        ut_array, bs_array = create_arrays(CARRIER_FREQUENCY, device, precision)
        scenario = UMaScenario(
            carrier_frequency=CARRIER_FREQUENCY,
            o2i_model="low",
            ut_array=ut_array,
            bs_array=bs_array,
            direction="uplink",
            precision=precision,
            device=device,
        )
        dtype = torch.float32 if precision == "single" else torch.float64

        # Use h_ut well above 13.5m so discrete set has multiple valid values
        # Valid set for h_ut=22.5m: {12, 15, 18, 21} — 4 values
        ut_loc = generate_random_loc(
            BATCH_SIZE, NB_UT, (100, 2000), (100, 2000), (22.5, 22.5),
            dtype=dtype, device=device
        )
        bs_loc = generate_random_loc(
            BATCH_SIZE, NB_BS, (0, 100), (0, 100), (H_BS, H_BS),
            dtype=dtype, device=device
        )
        ut_orientations = torch.zeros(BATCH_SIZE, NB_UT, 3, dtype=dtype, device=device)
        bs_orientations = torch.zeros(BATCH_SIZE, NB_BS, 3, dtype=dtype, device=device)
        ut_velocities   = torch.zeros(BATCH_SIZE, NB_UT, 3, dtype=dtype, device=device)
        in_state = generate_random_bool(BATCH_SIZE, NB_UT, 0.5, device=device)

        scenario.set_topology(
            ut_loc, bs_loc, ut_orientations, bs_orientations, ut_velocities, in_state
        )

        # h_e = r + (1 - r) * s, where r is Bernoulli (0 or 1).
        # When r=1, h_e=1.0 (fixed). When r=0, h_e=s from discrete set.
        # We extract s by checking only the r=0 cases via h_e values > 1.0
        h_e = scenario._h_e  # shape: [batch_size, num_bs, num_ut]
        s_samples = h_e[h_e > 1.0]

        if s_samples.numel() > 0:
            # Every sampled value must be exactly 12, 15, 18, or 21
            # Check: (s - 12) must be divisible by 3 with no remainder
            remainders = torch.remainder(
                s_samples - torch.tensor(12.0, dtype=dtype, device=device),
                torch.tensor(3.0, dtype=dtype, device=device),
            )
            assert torch.all(remainders < 1e-4), (
                "h_E samples are not from the discrete set {12, 15, 18, ...} "
                "as required by TR 38.901 Table 7.4.1-1 Note 1"
            )
            # All values must be >= 12
            assert torch.all(s_samples >= torch.tensor(12.0, dtype=dtype, device=device))

    def test_h_e_edge_case_small_h_ut(self, device, precision):
        """Test that when h_ut <= 13.5m, h_E defaults to 12m (only valid value)
        per TR 38.901 Table 7.4.1-1 Note 1 edge case."""
        ut_array, bs_array = create_arrays(CARRIER_FREQUENCY, device, precision)
        scenario = UMaScenario(
            carrier_frequency=CARRIER_FREQUENCY,
            o2i_model="low",
            ut_array=ut_array,
            bs_array=bs_array,
            direction="uplink",
            precision=precision,
            device=device,
        )
        dtype = torch.float32 if precision == "single" else torch.float64

        # h_ut = 13.0m → max_value = 11.5m < 12m → only valid s is 12.0
        ut_loc = generate_random_loc(
            BATCH_SIZE, NB_UT, (100, 2000), (100, 2000), (13.0, 13.0),
            dtype=dtype, device=device
        )
        bs_loc = generate_random_loc(
            BATCH_SIZE, NB_BS, (0, 100), (0, 100), (H_BS, H_BS),
            dtype=dtype, device=device
        )
        ut_orientations = torch.zeros(BATCH_SIZE, NB_UT, 3, dtype=dtype, device=device)
        bs_orientations = torch.zeros(BATCH_SIZE, NB_BS, 3, dtype=dtype, device=device)
        ut_velocities   = torch.zeros(BATCH_SIZE, NB_UT, 3, dtype=dtype, device=device)
        in_state = generate_random_bool(BATCH_SIZE, NB_UT, 0.5, device=device)

        scenario.set_topology(
            ut_loc, bs_loc, ut_orientations, bs_orientations, ut_velocities, in_state
        )

        h_e = scenario._h_e
        s_samples = h_e[h_e > 1.0]

        if s_samples.numel() > 0:
            # All s values must be exactly 12.0 when h_ut = 13.0m
            assert torch.all(
                torch.abs(s_samples - torch.tensor(12.0, dtype=dtype, device=device)) < 1e-4
            ), "h_E must equal 12.0m when h_ut <= 13.5m"


class TestUMiScenario:
    """Tests for UMiScenario"""

    def test_instantiation(self, device, precision):
        """Test UMiScenario can be instantiated"""
        ut_array, bs_array = create_arrays(CARRIER_FREQUENCY, device, precision)
        scenario = UMiScenario(
            carrier_frequency=CARRIER_FREQUENCY,
            o2i_model="low",
            ut_array=ut_array,
            bs_array=bs_array,
            direction="uplink",
            precision=precision,
            device=device,
        )
        assert scenario is not None
        assert scenario.carrier_frequency.item() == pytest.approx(CARRIER_FREQUENCY)

    def test_los_probability_bounds(self, device, precision):
        """Test that LoS probability is between 0 and 1"""
        ut_array, bs_array = create_arrays(CARRIER_FREQUENCY, device, precision)
        scenario = UMiScenario(
            carrier_frequency=CARRIER_FREQUENCY,
            o2i_model="high",
            ut_array=ut_array,
            bs_array=bs_array,
            direction="uplink",
            precision=precision,
            device=device,
        )

        dtype = torch.float32 if precision == "single" else torch.float64

        ut_loc = generate_random_loc(
            BATCH_SIZE, NB_UT, (100, 2000), (100, 2000), (H_UT, H_UT),
            dtype=dtype, device=device
        )
        bs_loc = generate_random_loc(
            BATCH_SIZE, NB_BS, (0, 100), (0, 100), (H_BS, H_BS),
            dtype=dtype, device=device
        )
        ut_orientations = torch.zeros(BATCH_SIZE, NB_UT, 3, dtype=dtype, device=device)
        bs_orientations = torch.zeros(BATCH_SIZE, NB_BS, 3, dtype=dtype, device=device)
        ut_velocities = torch.zeros(BATCH_SIZE, NB_UT, 3, dtype=dtype, device=device)
        in_state = generate_random_bool(BATCH_SIZE, NB_UT, 0.5, device=device)

        scenario.set_topology(
            ut_loc, bs_loc, ut_orientations, bs_orientations, ut_velocities, in_state
        )

        los_prob = scenario.los_probability
        assert torch.all(los_prob >= 0.0)
        assert torch.all(los_prob <= 1.0)


class TestRMaChannel:
    """Tests for RMa channel model"""

    def test_instantiation(self, device, precision):
        """Test RMa channel can be instantiated"""
        ut_array, bs_array = create_arrays(CARRIER_FREQUENCY, device, precision)
        channel = RMa(
            carrier_frequency=CARRIER_FREQUENCY,
            ut_array=ut_array,
            bs_array=bs_array,
            direction="uplink",
            precision=precision,
            device=device,
        )
        assert channel is not None

    def test_channel_generation(self, device, precision):
        """Test channel impulse response generation"""
        dtype = torch.float32 if precision == "single" else torch.float64

        ut_array, bs_array = create_arrays(CARRIER_FREQUENCY, device, precision)
        channel = RMa(
            carrier_frequency=CARRIER_FREQUENCY,
            ut_array=ut_array,
            bs_array=bs_array,
            direction="uplink",
            precision=precision,
            device=device,
        )

        # Set up topology
        ut_loc = generate_random_loc(
            BATCH_SIZE, NB_UT, (100, 2000), (100, 2000), (H_UT, H_UT),
            dtype=dtype, device=device
        )
        bs_loc = generate_random_loc(
            BATCH_SIZE, NB_BS, (0, 100), (0, 100), (H_BS, H_BS),
            dtype=dtype, device=device
        )
        ut_orientations = torch.zeros(BATCH_SIZE, NB_UT, 3, dtype=dtype, device=device)
        bs_orientations = torch.zeros(BATCH_SIZE, NB_BS, 3, dtype=dtype, device=device)
        ut_velocities = torch.zeros(BATCH_SIZE, NB_UT, 3, dtype=dtype, device=device)
        in_state = generate_random_bool(BATCH_SIZE, NB_UT, 0.5, device=device)

        channel.set_topology(
            ut_loc, bs_loc, ut_orientations, bs_orientations, ut_velocities, in_state
        )

        # Generate channel
        num_time_samples = 10
        sampling_frequency = 1e6
        result = channel(num_time_samples, sampling_frequency)
        h, delays = result[0], result[1]

        # Check output shapes
        assert h.dim() == 7  # [batch, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time]
        assert h.shape[0] == BATCH_SIZE
        assert delays.dim() == 4  # [batch, num_rx, num_tx, num_paths]
        assert delays.shape[0] == BATCH_SIZE


class TestUMaChannel:
    """Tests for UMa channel model"""

    def test_instantiation(self, device, precision):
        """Test UMa channel can be instantiated"""
        ut_array, bs_array = create_arrays(CARRIER_FREQUENCY, device, precision)
        channel = UMa(
            carrier_frequency=CARRIER_FREQUENCY,
            o2i_model="low",
            ut_array=ut_array,
            bs_array=bs_array,
            direction="uplink",
            precision=precision,
            device=device,
        )
        assert channel is not None

    def test_channel_generation(self, device, precision):
        """Test channel impulse response generation"""
        dtype = torch.float32 if precision == "single" else torch.float64

        ut_array, bs_array = create_arrays(CARRIER_FREQUENCY, device, precision)
        channel = UMa(
            carrier_frequency=CARRIER_FREQUENCY,
            o2i_model="low",
            ut_array=ut_array,
            bs_array=bs_array,
            direction="downlink",
            precision=precision,
            device=device,
        )

        # Set up topology
        ut_loc = generate_random_loc(
            BATCH_SIZE, NB_UT, (100, 2000), (100, 2000), (H_UT, H_UT),
            dtype=dtype, device=device
        )
        bs_loc = generate_random_loc(
            BATCH_SIZE, NB_BS, (0, 100), (0, 100), (H_BS, H_BS),
            dtype=dtype, device=device
        )
        ut_orientations = torch.zeros(BATCH_SIZE, NB_UT, 3, dtype=dtype, device=device)
        bs_orientations = torch.zeros(BATCH_SIZE, NB_BS, 3, dtype=dtype, device=device)
        ut_velocities = torch.zeros(BATCH_SIZE, NB_UT, 3, dtype=dtype, device=device)
        in_state = generate_random_bool(BATCH_SIZE, NB_UT, 0.5, device=device)

        channel.set_topology(
            ut_loc, bs_loc, ut_orientations, bs_orientations, ut_velocities, in_state
        )

        # Generate channel
        num_time_samples = 10
        sampling_frequency = 1e6
        result = channel(num_time_samples, sampling_frequency)
        h, delays = result[0], result[1]

        # Check output shapes
        assert h.dim() == 7
        assert h.shape[0] == BATCH_SIZE
        assert delays.dim() == 4
        assert delays.shape[0] == BATCH_SIZE


class TestUMiChannel:
    """Tests for UMi channel model"""

    def test_instantiation(self, device, precision):
        """Test UMi channel can be instantiated"""
        ut_array, bs_array = create_arrays(CARRIER_FREQUENCY, device, precision)
        channel = UMi(
            carrier_frequency=CARRIER_FREQUENCY,
            o2i_model="high",
            ut_array=ut_array,
            bs_array=bs_array,
            direction="uplink",
            precision=precision,
            device=device,
        )
        assert channel is not None

    def test_channel_generation(self, device, precision):
        """Test channel impulse response generation"""
        ut_array, bs_array = create_arrays(CARRIER_FREQUENCY, device, precision)
        channel = UMi(
            carrier_frequency=CARRIER_FREQUENCY,
            o2i_model="low",
            ut_array=ut_array,
            bs_array=bs_array,
            direction="uplink",
            precision=precision,
            device=device,
        )

        dtype = torch.float32 if precision == "single" else torch.float64

        # Set up topology
        ut_loc = generate_random_loc(
            BATCH_SIZE, NB_UT, (100, 2000), (100, 2000), (H_UT, H_UT),
            dtype=dtype, device=device
        )
        bs_loc = generate_random_loc(
            BATCH_SIZE, NB_BS, (0, 100), (0, 100), (H_BS, H_BS),
            dtype=dtype, device=device
        )
        ut_orientations = torch.zeros(BATCH_SIZE, NB_UT, 3, dtype=dtype, device=device)
        bs_orientations = torch.zeros(BATCH_SIZE, NB_BS, 3, dtype=dtype, device=device)
        ut_velocities = torch.zeros(BATCH_SIZE, NB_UT, 3, dtype=dtype, device=device)
        in_state = generate_random_bool(BATCH_SIZE, NB_UT, 0.5, device=device)

        channel.set_topology(
            ut_loc, bs_loc, ut_orientations, bs_orientations, ut_velocities, in_state
        )

        # Generate channel
        num_time_samples = 10
        sampling_frequency = 1e6
        result = channel(num_time_samples, sampling_frequency)
        h, delays = result[0], result[1]

        # Check output shapes
        assert h.dim() == 7
        assert h.shape[0] == BATCH_SIZE
        assert delays.dim() == 4
        assert delays.shape[0] == BATCH_SIZE


class TestAllScenarios:
    """Cross-scenario tests"""

    @pytest.mark.parametrize("scenario_class,kwargs", [
        (RMaScenario, {}),
        (UMaScenario, {"o2i_model": "low"}),
        (UMiScenario, {"o2i_model": "low"}),
    ])
    def test_lsp_shapes(self, device, precision, scenario_class, kwargs):
        """Test LSP shapes are correct for all scenarios"""
        ut_array, bs_array = create_arrays(CARRIER_FREQUENCY, device, precision)
        scenario = scenario_class(
            carrier_frequency=CARRIER_FREQUENCY,
            ut_array=ut_array,
            bs_array=bs_array,
            direction="uplink",
            precision=precision,
            device=device,
            **kwargs,
        )

        dtype = torch.float32 if precision == "single" else torch.float64

        ut_loc = generate_random_loc(
            BATCH_SIZE, NB_UT, (100, 2000), (100, 2000), (H_UT, H_UT),
            dtype=dtype, device=device
        )
        bs_loc = generate_random_loc(
            BATCH_SIZE, NB_BS, (0, 100), (0, 100), (H_BS, H_BS),
            dtype=dtype, device=device
        )
        ut_orientations = torch.zeros(BATCH_SIZE, NB_UT, 3, dtype=dtype, device=device)
        bs_orientations = torch.zeros(BATCH_SIZE, NB_BS, 3, dtype=dtype, device=device)
        ut_velocities = torch.zeros(BATCH_SIZE, NB_UT, 3, dtype=dtype, device=device)
        in_state = generate_random_bool(BATCH_SIZE, NB_UT, 0.5, device=device)

        scenario.set_topology(
            ut_loc, bs_loc, ut_orientations, bs_orientations, ut_velocities, in_state
        )

        # Check LSP mean and std shapes
        assert scenario.lsp_log_mean.shape == (BATCH_SIZE, NB_BS, NB_UT, 7)
        assert scenario.lsp_log_std.shape == (BATCH_SIZE, NB_BS, NB_UT, 7)

    @pytest.mark.parametrize("direction", ["uplink", "downlink"])
    def test_direction_validation(self, device, precision, direction):
        """Test that direction parameter works for both values"""
        ut_array, bs_array = create_arrays(CARRIER_FREQUENCY, device, precision)
        scenario = RMaScenario(
            carrier_frequency=CARRIER_FREQUENCY,
            ut_array=ut_array,
            bs_array=bs_array,
            direction=direction,
            precision=precision,
            device=device,
        )
        assert scenario.direction == direction

    def test_invalid_direction(self, device, precision):
        """Test that invalid direction raises error"""
        ut_array, bs_array = create_arrays(CARRIER_FREQUENCY, device, precision)
        with pytest.raises(AssertionError):
            RMaScenario(
                carrier_frequency=CARRIER_FREQUENCY,
                ut_array=ut_array,
                bs_array=bs_array,
                direction="invalid",
                precision=precision,
                device=device,
            )

