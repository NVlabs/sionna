#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Tests for 3GPP TR 38.901 CDL channel model"""

import numpy as np
import pytest
import torch

from sionna.phy import PI, config
from sionna.phy.channel.tr38901 import CDL, PanelArray
from channel_test_utils import (
    CDL_POWERS, CDL_DELAYS, CDL_AOD, CDL_AOA, CDL_ZOD, CDL_ZOA, CDL_XPR,
    cdl_aod, cdl_aoa, cdl_zod, cdl_zoa,
)


class TestCDL:
    """Tests for the CDL channel model"""

    # Carrier frequency
    CARRIER_FREQUENCY = 3.5e9  # Hz

    # Delay spread
    DELAY_SPREAD = 100e-9  # s

    # Maximum allowed deviation (absolute error)
    MAX_ERR = 1e-4

    # Maximum allowed deviation (relative error)
    MAX_ERR_REL = 1e-2

    def _cdl_aod(self, model: str) -> np.ndarray:
        """Return the AoD in radians with per-ray angle spreads applied.
        
        Note: For LoS models (D, E), the test_utils values already have
        the LoS component removed, matching what CDL outputs.
        
        Returns shape [num_clusters, 20] with ray AoDs in radians.
        """
        return cdl_aod(model)

    def _cdl_aoa(self, model: str) -> np.ndarray:
        """Return the AoA in radians with per-ray angle spreads applied.
        
        Note: For LoS models (D, E), the test_utils values already have
        the LoS component removed, matching what CDL outputs.
        
        Returns shape [num_clusters, 20] with ray AoAs in radians.
        """
        return cdl_aoa(model)

    def _cdl_zod(self, model: str) -> np.ndarray:
        """Return the ZoD in radians with per-ray angle spreads applied.
        
        Note: For LoS models (D, E), the test_utils values already have
        the LoS component removed, matching what CDL outputs.
        
        Returns shape [num_clusters, 20] with ray ZoDs in radians.
        """
        return cdl_zod(model)

    def _cdl_zoa(self, model: str) -> np.ndarray:
        """Return the ZoA in radians with per-ray angle spreads applied.
        
        Note: For LoS models (D, E), the test_utils values already have
        the LoS component removed, matching what CDL outputs.
        
        Returns shape [num_clusters, 20] with ray ZoAs in radians.
        """
        return cdl_zoa(model)

    def test_cdl_instantiation(self, device, precision):
        """Test that CDL models can be instantiated"""
        tx_array = PanelArray(
            num_rows_per_panel=1,
            num_cols_per_panel=1,
            polarization="single",
            polarization_type="V",
            antenna_pattern="omni",
            carrier_frequency=self.CARRIER_FREQUENCY,
            precision=precision,
            device=device,
        )
        rx_array = PanelArray(
            num_rows_per_panel=1,
            num_cols_per_panel=1,
            polarization="single",
            polarization_type="V",
            antenna_pattern="omni",
            carrier_frequency=self.CARRIER_FREQUENCY,
            precision=precision,
            device=device,
        )

        for model in ("A", "B", "C", "D", "E"):
            cdl = CDL(
                model=model,
                delay_spread=self.DELAY_SPREAD,
                carrier_frequency=self.CARRIER_FREQUENCY,
                ut_array=rx_array,
                bs_array=tx_array,
                direction="downlink",
                precision=precision,
                device=device,
            )
            assert cdl._model == model
            assert cdl.spec_version == "19.2"

        cdl = CDL(
            model="A",
            delay_spread=self.DELAY_SPREAD,
            carrier_frequency=self.CARRIER_FREQUENCY,
            ut_array=rx_array,
            bs_array=tx_array,
            direction="downlink",
            precision=precision,
            device=device,
            spec_version="16.1",
        )
        assert cdl.spec_version == "16.1"

    def test_cdl_output_shape(self, device, precision):
        """Test that CDL output has correct shape"""
        tx_array = PanelArray(
            num_rows_per_panel=2,
            num_cols_per_panel=2,
            polarization="single",
            polarization_type="V",
            antenna_pattern="omni",
            carrier_frequency=self.CARRIER_FREQUENCY,
            precision=precision,
            device=device,
        )
        rx_array = PanelArray(
            num_rows_per_panel=1,
            num_cols_per_panel=2,
            polarization="single",
            polarization_type="V",
            antenna_pattern="omni",
            carrier_frequency=self.CARRIER_FREQUENCY,
            precision=precision,
            device=device,
        )

        batch_size = 10
        num_time_steps = 5

        cdl = CDL(
            model="A",
            delay_spread=self.DELAY_SPREAD,
            carrier_frequency=self.CARRIER_FREQUENCY,
            ut_array=rx_array,
            bs_array=tx_array,
            direction="downlink",
            precision=precision,
            device=device,
        )

        a, tau = cdl(batch_size=batch_size, num_time_steps=num_time_steps, sampling_frequency=1e6)

        # Check shapes
        # a: [batch, num_rx=1, num_rx_ant, num_tx=1, num_tx_ant, num_paths, num_time_steps]
        assert a.shape[0] == batch_size
        assert a.shape[1] == 1  # num_rx
        assert a.shape[2] == rx_array.num_ant  # num_rx_ant
        assert a.shape[3] == 1  # num_tx
        assert a.shape[4] == tx_array.num_ant  # num_tx_ant
        assert a.shape[5] == 23  # num_paths for CDL-A
        assert a.shape[6] == num_time_steps

        # tau: [batch, num_rx=1, num_tx=1, num_paths]
        assert tau.shape[0] == batch_size
        assert tau.shape[1] == 1
        assert tau.shape[2] == 1
        assert tau.shape[3] == 23  # num_paths for CDL-A

    @pytest.mark.parametrize("model", ["A", "B", "C", "D", "E"])
    def test_cdl_delays(self, device, precision, model):
        """Test that CDL delays match the specification"""
        tx_array = PanelArray(
            num_rows_per_panel=1,
            num_cols_per_panel=1,
            polarization="single",
            polarization_type="V",
            antenna_pattern="omni",
            carrier_frequency=self.CARRIER_FREQUENCY,
            precision=precision,
            device=device,
        )
        rx_array = PanelArray(
            num_rows_per_panel=1,
            num_cols_per_panel=1,
            polarization="single",
            polarization_type="V",
            antenna_pattern="omni",
            carrier_frequency=self.CARRIER_FREQUENCY,
            precision=precision,
            device=device,
        )

        cdl = CDL(
            model=model,
            delay_spread=self.DELAY_SPREAD,
            carrier_frequency=self.CARRIER_FREQUENCY,
            ut_array=rx_array,
            bs_array=tx_array,
            direction="downlink",
            precision=precision,
            device=device,
        )

        _, tau = cdl(batch_size=1, num_time_steps=1, sampling_frequency=1e6)

        # Normalize delays by delay spread and compare to reference
        delays = tau[0, 0, 0].cpu().numpy() / self.DELAY_SPREAD
        ref_delays = np.sort(CDL_DELAYS[model])

        max_err = np.max(np.abs(ref_delays - delays))
        assert max_err <= self.MAX_ERR, f"CDL-{model}: delay error {max_err} > {self.MAX_ERR}"

    @pytest.mark.parametrize("model", ["A", "B", "C", "D", "E"])
    def test_cdl_powers(self, device, precision, model):
        """Test that CDL powers match the specification"""
        tx_array = PanelArray(
            num_rows_per_panel=1,
            num_cols_per_panel=1,
            polarization="single",
            polarization_type="V",
            antenna_pattern="omni",
            carrier_frequency=self.CARRIER_FREQUENCY,
            precision=precision,
            device=device,
        )
        rx_array = PanelArray(
            num_rows_per_panel=1,
            num_cols_per_panel=1,
            polarization="single",
            polarization_type="V",
            antenna_pattern="omni",
            carrier_frequency=self.CARRIER_FREQUENCY,
            precision=precision,
            device=device,
        )

        cdl = CDL(
            model=model,
            delay_spread=self.DELAY_SPREAD,
            carrier_frequency=self.CARRIER_FREQUENCY,
            ut_array=rx_array,
            bs_array=tx_array,
            direction="downlink",
            precision=precision,
            device=device,
        )

        # Generate large batch to get good power estimate
        a, tau = cdl(batch_size=100000, num_time_steps=1, sampling_frequency=1e6)

        # Compute average power per path
        powers = torch.mean(torch.abs(a[:, 0, 0, 0, 0, :, 0]) ** 2, dim=0).cpu().numpy()
        
        # Get actual delays from model (normalized by delay spread)
        actual_delays = tau[0, 0, 0].cpu().numpy() / self.DELAY_SPREAD

        # Reference powers (normalized)
        ref_powers = np.power(10.0, CDL_POWERS[model] / 10.0)
        ref_powers = ref_powers / np.sum(ref_powers)
        ref_delays = CDL_DELAYS[model]

        # Sort both by (delay, power) to handle tied delays consistently
        # This ensures deterministic ordering even when delays are identical
        actual_order = np.lexsort((powers, actual_delays))
        ref_order = np.lexsort((ref_powers, ref_delays))
        
        powers_sorted = powers[actual_order]
        ref_powers_sorted = ref_powers[ref_order]

        # Check relative error
        max_err = np.max(np.abs(ref_powers_sorted - powers_sorted) / ref_powers_sorted)
        assert max_err <= self.MAX_ERR_REL, f"CDL-{model}: power error {max_err} > {self.MAX_ERR_REL}"

    @pytest.mark.parametrize("model", ["A", "B", "C", "D", "E"])
    def test_cdl_aod(self, device, precision, model):
        """Test that CDL AoD angles match the specification"""
        tx_array = PanelArray(
            num_rows_per_panel=1,
            num_cols_per_panel=1,
            polarization="single",
            polarization_type="V",
            antenna_pattern="omni",
            carrier_frequency=self.CARRIER_FREQUENCY,
            precision=precision,
            device=device,
        )
        rx_array = PanelArray(
            num_rows_per_panel=1,
            num_cols_per_panel=1,
            polarization="single",
            polarization_type="V",
            antenna_pattern="omni",
            carrier_frequency=self.CARRIER_FREQUENCY,
            precision=precision,
            device=device,
        )

        cdl = CDL(
            model=model,
            delay_spread=self.DELAY_SPREAD,
            carrier_frequency=self.CARRIER_FREQUENCY,
            ut_array=rx_array,
            bs_array=tx_array,
            direction="downlink",
            precision=precision,
            device=device,
        )

        # Access internal AoD values
        # Shape is [num_clusters, 20 rays] after angle spread is applied
        aod = cdl._aod.cpu().numpy()
        ref_aod = self._cdl_aod(model)

        max_err = np.max(np.abs(ref_aod - aod))
        assert max_err <= self.MAX_ERR, f"CDL-{model}: AoD error {max_err} > {self.MAX_ERR}"

    @pytest.mark.parametrize("model", ["A", "B", "C", "D", "E"])
    def test_cdl_aoa(self, device, precision, model):
        """Test that CDL AoA angles match the specification"""
        tx_array = PanelArray(
            num_rows_per_panel=1,
            num_cols_per_panel=1,
            polarization="single",
            polarization_type="V",
            antenna_pattern="omni",
            carrier_frequency=self.CARRIER_FREQUENCY,
            precision=precision,
            device=device,
        )
        rx_array = PanelArray(
            num_rows_per_panel=1,
            num_cols_per_panel=1,
            polarization="single",
            polarization_type="V",
            antenna_pattern="omni",
            carrier_frequency=self.CARRIER_FREQUENCY,
            precision=precision,
            device=device,
        )

        cdl = CDL(
            model=model,
            delay_spread=self.DELAY_SPREAD,
            carrier_frequency=self.CARRIER_FREQUENCY,
            ut_array=rx_array,
            bs_array=tx_array,
            direction="downlink",
            precision=precision,
            device=device,
        )

        # Access internal AoA values
        aoa = cdl._aoa.cpu().numpy()
        ref_aoa = self._cdl_aoa(model)

        max_err = np.max(np.abs(ref_aoa - aoa))
        assert max_err <= self.MAX_ERR, f"CDL-{model}: AoA error {max_err} > {self.MAX_ERR}"

    @pytest.mark.parametrize("model", ["A", "B", "C", "D", "E"])
    def test_cdl_zod(self, device, precision, model):
        """Test that CDL ZoD angles match the specification"""
        tx_array = PanelArray(
            num_rows_per_panel=1,
            num_cols_per_panel=1,
            polarization="single",
            polarization_type="V",
            antenna_pattern="omni",
            carrier_frequency=self.CARRIER_FREQUENCY,
            precision=precision,
            device=device,
        )
        rx_array = PanelArray(
            num_rows_per_panel=1,
            num_cols_per_panel=1,
            polarization="single",
            polarization_type="V",
            antenna_pattern="omni",
            carrier_frequency=self.CARRIER_FREQUENCY,
            precision=precision,
            device=device,
        )

        cdl = CDL(
            model=model,
            delay_spread=self.DELAY_SPREAD,
            carrier_frequency=self.CARRIER_FREQUENCY,
            ut_array=rx_array,
            bs_array=tx_array,
            direction="downlink",
            precision=precision,
            device=device,
        )

        # Access internal ZoD values
        zod = cdl._zod.cpu().numpy()
        ref_zod = self._cdl_zod(model)

        max_err = np.max(np.abs(ref_zod - zod))
        assert max_err <= self.MAX_ERR, f"CDL-{model}: ZoD error {max_err} > {self.MAX_ERR}"

    @pytest.mark.parametrize("model", ["A", "B", "C", "D", "E"])
    def test_cdl_zoa(self, device, precision, model):
        """Test that CDL ZoA angles match the specification"""
        tx_array = PanelArray(
            num_rows_per_panel=1,
            num_cols_per_panel=1,
            polarization="single",
            polarization_type="V",
            antenna_pattern="omni",
            carrier_frequency=self.CARRIER_FREQUENCY,
            precision=precision,
            device=device,
        )
        rx_array = PanelArray(
            num_rows_per_panel=1,
            num_cols_per_panel=1,
            polarization="single",
            polarization_type="V",
            antenna_pattern="omni",
            carrier_frequency=self.CARRIER_FREQUENCY,
            precision=precision,
            device=device,
        )

        cdl = CDL(
            model=model,
            delay_spread=self.DELAY_SPREAD,
            carrier_frequency=self.CARRIER_FREQUENCY,
            ut_array=rx_array,
            bs_array=tx_array,
            direction="downlink",
            precision=precision,
            device=device,
        )

        # Access internal ZoA values
        zoa = cdl._zoa.cpu().numpy()
        ref_zoa = self._cdl_zoa(model)

        max_err = np.max(np.abs(ref_zoa - zoa))
        assert max_err <= self.MAX_ERR, f"CDL-{model}: ZoA error {max_err} > {self.MAX_ERR}"

    @pytest.mark.parametrize("model", ["A", "B", "C", "D", "E"])
    def test_cdl_xpr(self, device, precision, model):
        """Test that CDL XPR matches the specification"""
        tx_array = PanelArray(
            num_rows_per_panel=1,
            num_cols_per_panel=1,
            polarization="single",
            polarization_type="V",
            antenna_pattern="omni",
            carrier_frequency=self.CARRIER_FREQUENCY,
            precision=precision,
            device=device,
        )
        rx_array = PanelArray(
            num_rows_per_panel=1,
            num_cols_per_panel=1,
            polarization="single",
            polarization_type="V",
            antenna_pattern="omni",
            carrier_frequency=self.CARRIER_FREQUENCY,
            precision=precision,
            device=device,
        )

        cdl = CDL(
            model=model,
            delay_spread=self.DELAY_SPREAD,
            carrier_frequency=self.CARRIER_FREQUENCY,
            ut_array=rx_array,
            bs_array=tx_array,
            direction="downlink",
            precision=precision,
            device=device,
        )

        # Access internal XPR value (linear scale)
        xpr = cdl._xpr.cpu().numpy()
        ref_xpr = CDL_XPR[model]

        max_err = np.abs(ref_xpr - xpr)
        assert max_err <= self.MAX_ERR, f"CDL-{model}: XPR error {max_err} > {self.MAX_ERR}"

    def test_cdl_uplink_direction(self, device, precision):
        """Test that CDL works in uplink direction"""
        tx_array = PanelArray(
            num_rows_per_panel=1,
            num_cols_per_panel=2,
            polarization="single",
            polarization_type="V",
            antenna_pattern="omni",
            carrier_frequency=self.CARRIER_FREQUENCY,
            precision=precision,
            device=device,
        )
        rx_array = PanelArray(
            num_rows_per_panel=2,
            num_cols_per_panel=2,
            polarization="single",
            polarization_type="V",
            antenna_pattern="omni",
            carrier_frequency=self.CARRIER_FREQUENCY,
            precision=precision,
            device=device,
        )

        cdl = CDL(
            model="A",
            delay_spread=self.DELAY_SPREAD,
            carrier_frequency=self.CARRIER_FREQUENCY,
            ut_array=tx_array,  # UT is now TX
            bs_array=rx_array,  # BS is now RX
            direction="uplink",
            precision=precision,
            device=device,
        )

        a, tau = cdl(batch_size=5, num_time_steps=3, sampling_frequency=1e6)

        # Check that output is generated without errors
        assert a.shape[0] == 5
        assert a.shape[6] == 3

    def test_cdl_with_velocity(self, device, precision):
        """Test that CDL works with specified velocity"""
        tx_array = PanelArray(
            num_rows_per_panel=1,
            num_cols_per_panel=1,
            polarization="single",
            polarization_type="V",
            antenna_pattern="omni",
            carrier_frequency=self.CARRIER_FREQUENCY,
            precision=precision,
            device=device,
        )
        rx_array = PanelArray(
            num_rows_per_panel=1,
            num_cols_per_panel=1,
            polarization="single",
            polarization_type="V",
            antenna_pattern="omni",
            carrier_frequency=self.CARRIER_FREQUENCY,
            precision=precision,
            device=device,
        )

        cdl = CDL(
            model="A",
            delay_spread=self.DELAY_SPREAD,
            carrier_frequency=self.CARRIER_FREQUENCY,
            ut_array=rx_array,
            bs_array=tx_array,
            direction="downlink",
            ut_velocity=[10.0, 0.0, 0.0],  # 10 m/s in x direction
            precision=precision,
            device=device,
        )

        a, tau = cdl(batch_size=5, num_time_steps=10, sampling_frequency=1e6)

        # Check that channels vary over time (Doppler effect)
        assert a.shape[6] == 10
        # The channel should not be constant over time
        variation = torch.std(torch.abs(a), dim=-1).mean()
        assert variation > 0

    def test_cdl_random_velocity(self, device, precision):
        """Test that CDL works with random velocity generation"""
        tx_array = PanelArray(
            num_rows_per_panel=1,
            num_cols_per_panel=1,
            polarization="single",
            polarization_type="V",
            antenna_pattern="omni",
            carrier_frequency=self.CARRIER_FREQUENCY,
            precision=precision,
            device=device,
        )
        rx_array = PanelArray(
            num_rows_per_panel=1,
            num_cols_per_panel=1,
            polarization="single",
            polarization_type="V",
            antenna_pattern="omni",
            carrier_frequency=self.CARRIER_FREQUENCY,
            precision=precision,
            device=device,
        )

        cdl = CDL(
            model="A",
            delay_spread=self.DELAY_SPREAD,
            carrier_frequency=self.CARRIER_FREQUENCY,
            ut_array=rx_array,
            bs_array=tx_array,
            direction="downlink",
            min_speed=5.0,
            max_speed=10.0,
            precision=precision,
            device=device,
        )

        a, tau = cdl(batch_size=5, num_time_steps=10, sampling_frequency=1e6)

        # Check that output is generated
        assert a.shape[0] == 5
        assert a.shape[6] == 10

    def test_cdl_dual_polarization(self, device, precision):
        """Test that CDL works with dual polarization antennas"""
        tx_array = PanelArray(
            num_rows_per_panel=2,
            num_cols_per_panel=2,
            polarization="dual",
            polarization_type="cross",
            antenna_pattern="38.901",
            carrier_frequency=self.CARRIER_FREQUENCY,
            precision=precision,
            device=device,
        )
        rx_array = PanelArray(
            num_rows_per_panel=1,
            num_cols_per_panel=1,
            polarization="dual",
            polarization_type="cross",
            antenna_pattern="38.901",
            carrier_frequency=self.CARRIER_FREQUENCY,
            precision=precision,
            device=device,
        )

        cdl = CDL(
            model="A",
            delay_spread=self.DELAY_SPREAD,
            carrier_frequency=self.CARRIER_FREQUENCY,
            ut_array=rx_array,
            bs_array=tx_array,
            direction="downlink",
            precision=precision,
            device=device,
        )

        a, tau = cdl(batch_size=5, num_time_steps=3, sampling_frequency=1e6)

        # Check shapes with dual polarization
        assert a.shape[2] == rx_array.num_ant  # 2 for dual pol single element
        assert a.shape[4] == tx_array.num_ant  # 8 for 2x2 dual pol

    def test_cdl_los_models(self, device, precision):
        """Test that CDL-D and CDL-E have LoS component"""
        tx_array = PanelArray(
            num_rows_per_panel=1,
            num_cols_per_panel=1,
            polarization="single",
            polarization_type="V",
            antenna_pattern="omni",
            carrier_frequency=self.CARRIER_FREQUENCY,
            precision=precision,
            device=device,
        )
        rx_array = PanelArray(
            num_rows_per_panel=1,
            num_cols_per_panel=1,
            polarization="single",
            polarization_type="V",
            antenna_pattern="omni",
            carrier_frequency=self.CARRIER_FREQUENCY,
            precision=precision,
            device=device,
        )

        # CDL-D should have LoS
        cdl_d = CDL(
            model="D",
            delay_spread=self.DELAY_SPREAD,
            carrier_frequency=self.CARRIER_FREQUENCY,
            ut_array=rx_array,
            bs_array=tx_array,
            direction="downlink",
            precision=precision,
            device=device,
        )
        assert cdl_d._has_los is True

        # CDL-E should have LoS
        cdl_e = CDL(
            model="E",
            delay_spread=self.DELAY_SPREAD,
            carrier_frequency=self.CARRIER_FREQUENCY,
            ut_array=rx_array,
            bs_array=tx_array,
            direction="downlink",
            precision=precision,
            device=device,
        )
        assert cdl_e._has_los is True

        # CDL-A should not have LoS
        cdl_a = CDL(
            model="A",
            delay_spread=self.DELAY_SPREAD,
            carrier_frequency=self.CARRIER_FREQUENCY,
            ut_array=rx_array,
            bs_array=tx_array,
            direction="downlink",
            precision=precision,
            device=device,
        )
        assert cdl_a._has_los is False
