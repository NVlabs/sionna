#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Unit tests for sionna.sys.power_control"""

import numpy as np
import pytest
import torch

from sionna.phy import config
from sionna.phy.utils import dbm_to_watt, lin_to_db, db_to_lin
from sionna.sys import downlink_fair_power_control, open_loop_uplink_power_control


class TestPowerControl:
    """Tests for power control functions."""

    def test_open_loop_ul_power_control(self, device, precision):
        """Test open loop uplink power control."""
        batch_size = 2
        num_ut = 4

        # Generate random pathloss (linear scale)
        rdtype = torch.float32 if precision == "single" else torch.float64
        pathloss_db = torch.rand(batch_size, num_ut, dtype=rdtype, device=device) * 40 + 80
        pathloss = db_to_lin(pathloss_db, precision=precision)

        # Number of allocated subcarriers per user
        num_allocated_subcarriers = torch.randint(12, 52, (batch_size, num_ut), device=device)

        # Power control parameters
        p0_dbm = -90.0
        alpha = 0.8
        ut_max_power_dbm = 26.0

        # Compute UL power
        tx_power = open_loop_uplink_power_control(
            pathloss,
            num_allocated_subcarriers,
            alpha=alpha,
            p0_dbm=p0_dbm,
            ut_max_power_dbm=ut_max_power_dbm,
            precision=precision,
        )

        # Check output shape
        assert tx_power.shape == (batch_size, num_ut)

        # Check that power is positive
        assert (tx_power >= 0).all()

        # Check that power does not exceed max
        ut_max_power_watt = dbm_to_watt(ut_max_power_dbm).item()
        assert (tx_power <= ut_max_power_watt * 1.01).all()  # small tolerance

    def test_downlink_fair_power_control(self, device, precision):
        """Test downlink fair power control."""
        batch_size = 2
        num_ut = 4

        # Generate random pathloss (linear scale)
        rdtype = torch.float32 if precision == "single" else torch.float64
        pathloss_db = torch.rand(batch_size, num_ut, dtype=rdtype, device=device) * 40 + 80
        pathloss = db_to_lin(pathloss_db, precision=precision)

        # Interference plus noise
        interference_plus_noise = 1e-10  # [W]

        # Number of allocated resources per user
        num_allocated_re = torch.randint(10, 100, (batch_size, num_ut), device=device)

        # Power control parameters
        bs_max_power_dbm = 56.0

        # Compute DL power
        tx_power, utility = downlink_fair_power_control(
            pathloss,
            interference_plus_noise=interference_plus_noise,
            num_allocated_re=num_allocated_re,
            bs_max_power_dbm=bs_max_power_dbm,
            precision=precision,
        )

        # Check output shapes
        assert tx_power.shape == (batch_size, num_ut)
        assert utility.shape == (batch_size, num_ut)

        # Check that power is non-negative
        assert (tx_power >= 0).all()

    def test_downlink_fair_power_control_reference(self, device, precision):
        """Check total power, lower bounds, KKT conditions, and utility."""
        rdtype = torch.float32 if precision == "single" else torch.float64
        pathloss = torch.tensor(
            [[1.0, 1.2, 1.5]], dtype=rdtype, device=device
        )
        num_allocated_re = torch.tensor(
            [[1.0, 2.0, 4.0]], dtype=rdtype, device=device
        )
        guaranteed_power_ratio = 0.15
        bs_max_power_dbm = 30.0
        max_power = dbm_to_watt(
            torch.tensor(bs_max_power_dbm, dtype=rdtype, device=device),
            precision=precision,
        )

        tx_power, utility, mu_inv = downlink_fair_power_control(
            pathloss,
            interference_plus_noise=1.0,
            num_allocated_re=num_allocated_re,
            bs_max_power_dbm=bs_max_power_dbm,
            guaranteed_power_ratio=guaranteed_power_ratio,
            fairness=0.0,
            return_lagrangian=True,
            precision=precision,
        )

        # Waterfilling with cq=[1, 5/6, 2/3] leaves the third user below its
        # guaranteed floor, so it sits at the floor and the remaining 0.95 W is
        # shared at water level mu^-1 = 1.45 across the first two users.
        expected_mu_inv = torch.tensor([1.45], dtype=rdtype, device=device)
        expected_tx_power = torch.tensor(
            [[0.45, 0.50, 0.05]], dtype=rdtype, device=device
        )
        torch.testing.assert_close(
            tx_power, expected_tx_power, rtol=1e-3, atol=1e-4
        )
        torch.testing.assert_close(
            mu_inv, expected_mu_inv, rtol=1e-3, atol=1e-4
        )
        torch.testing.assert_close(
            tx_power.sum(dim=-1),
            max_power.reshape(1),
            rtol=1e-3,
            atol=1e-4,
        )

        guaranteed_power = (
            guaranteed_power_ratio * max_power / pathloss.shape[-1]
        )
        assert torch.all(tx_power >= guaranteed_power - 1e-4)

        # The utility must use the per-resource power inside the logarithm.
        cq = 1.0 / pathloss
        expected_utility = num_allocated_re * torch.log1p(
            expected_tx_power / num_allocated_re * cq
        )
        torch.testing.assert_close(
            utility, expected_utility, rtol=1e-3, atol=1e-4
        )

        is_above_floor = tx_power > guaranteed_power + 1e-4
        kkt_residual = (
            cq * mu_inv.unsqueeze(-1)
            - (1.0 + tx_power / num_allocated_re * cq)
        )
        torch.testing.assert_close(
            kkt_residual[is_above_floor],
            torch.zeros_like(kkt_residual[is_above_floor]),
            rtol=0.0,
            atol=1e-3,
        )

    @pytest.mark.parametrize("mode", ["default", "reduce-overhead"])
    def test_ul_power_control_compiled(self, device, mode):
        """Test that open_loop_uplink_power_control works with torch.compile."""
        if device == "cpu" and mode == "reduce-overhead":
            pytest.skip("reduce-overhead mode not well supported on CPU")

        batch_size = 2
        num_ut = 4

        pathloss_db = torch.rand(batch_size, num_ut, device=device) * 40 + 80
        pathloss = db_to_lin(pathloss_db)

        num_allocated_subcarriers = torch.randint(12, 52, (batch_size, num_ut), device=device)

        compiled_fn = torch.compile(open_loop_uplink_power_control, mode=mode)

        tx_power = compiled_fn(
            pathloss,
            num_allocated_subcarriers,
            alpha=0.8,
            p0_dbm=-90.0,
            ut_max_power_dbm=26.0,
        )

        assert tx_power.shape == (batch_size, num_ut)
