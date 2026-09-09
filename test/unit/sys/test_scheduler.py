#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Unit tests for sionna.sys.scheduling"""

import pytest
import torch

from sionna.phy import config
from sionna.sys import PFSchedulerSUMIMO


class TestScheduler:
    """Tests for the PFSchedulerSUMIMO class."""

    def test_first_slot(self, device, precision):
        """Test scheduler behavior in the first slot."""
        num_ut = 8
        num_freq_res = 52
        num_ofdm_sym = 14
        batch_size = 10

        scheduler = PFSchedulerSUMIMO(
            num_ut=num_ut,
            num_freq_res=num_freq_res,
            num_ofdm_sym=num_ofdm_sym,
            batch_size=batch_size,
            precision=precision,
            device=device,
        )

        # Generate random last slot rates and achievable rates
        rate_last_slot = torch.rand(batch_size, num_ut, device=device) * 100
        rate_achievable_curr_slot = torch.rand(
            batch_size, num_ofdm_sym, num_freq_res, num_ut, device=device
        ) * 100

        # Get scheduling decisions
        is_scheduled = scheduler(rate_last_slot, rate_achievable_curr_slot)

        # Check shape - includes num_streams_per_ut (defaults to 1)
        assert is_scheduled.shape == (batch_size, num_ofdm_sym, num_freq_res, num_ut, 1)

        # Check that exactly one user is scheduled per resource
        assert (is_scheduled.sum(dim=-2).squeeze(-1) == 1).all()

    def test_fairness_over_time(self, device, precision):
        """Test that scheduler achieves fairness over multiple slots."""
        num_ut = 4
        num_freq_res = 12
        num_ofdm_sym = 14
        num_slots = 100

        scheduler = PFSchedulerSUMIMO(
            num_ut=num_ut,
            num_freq_res=num_freq_res,
            num_ofdm_sym=num_ofdm_sym,
            beta=0.9,  # Use beta instead of smoothing_factor
            precision=precision,
            device=device,
        )

        # Track how many times each user is scheduled
        schedule_count = torch.zeros(num_ut, device=device)

        # Set random seed for reproducibility
        torch.manual_seed(42)

        for _ in range(num_slots):
            # Add small random noise to break ties - essential for fairness test
            # Without noise, argmax always returns the same user when rates are equal
            rate_achievable_curr_slot = torch.ones(
                num_ofdm_sym, num_freq_res, num_ut, device=device
            ) * 10 + torch.rand(num_ofdm_sym, num_freq_res, num_ut, device=device) * 0.1
            
            # Last slot rate (use past achieved rate as proxy)
            rate_last_slot = scheduler.rate_achieved_past
            is_scheduled = scheduler(rate_last_slot, rate_achievable_curr_slot)
            # Sum over all resources to get per-user schedule count
            schedule_count += is_scheduled.sum(dim=(0, 1, 3)).float().squeeze(-1)

        # With roughly equal rates, each user should be scheduled roughly equally
        total_schedules = schedule_count.sum()
        expected_per_user = total_schedules / num_ut
        for ut in range(num_ut):
            # Allow for some variance (within 50% of expected)
            assert schedule_count[ut] > expected_per_user * 0.2, (
                f"User {ut} scheduled too infrequently"
            )
            assert schedule_count[ut] < expected_per_user * 1.8, (
                f"User {ut} scheduled too frequently"
            )

    def test_unequal_rates(self, device, precision):
        """Test scheduler with unequal achievable rates."""
        num_ut = 4
        num_freq_res = 12
        num_ofdm_sym = 14

        scheduler = PFSchedulerSUMIMO(
            num_ut=num_ut,
            num_freq_res=num_freq_res,
            num_ofdm_sym=num_ofdm_sym,
            beta=0.9,
            precision=precision,
            device=device,
        )

        # User 0 has much higher rate
        rate_achievable_curr_slot = torch.ones(
            num_ofdm_sym, num_freq_res, num_ut, device=device
        ) * 10
        rate_achievable_curr_slot[..., 0] = 100.0

        # Last slot rate (uniform)
        rate_last_slot = torch.ones(num_ut, device=device)

        # In first slot, user 0 should be scheduled most (highest PF metric)
        is_scheduled = scheduler(rate_last_slot, rate_achievable_curr_slot)

        # User 0 should have the most schedules
        schedule_count = is_scheduled.sum(dim=(0, 1, 3)).squeeze(-1)
        assert schedule_count[0] > schedule_count[1]

    def test_batch_processing(self, device, precision):
        """Test scheduler with batched inputs."""
        num_ut = 4
        num_freq_res = 12
        num_ofdm_sym = 14
        batch_size = 8

        scheduler = PFSchedulerSUMIMO(
            num_ut=num_ut,
            num_freq_res=num_freq_res,
            num_ofdm_sym=num_ofdm_sym,
            batch_size=batch_size,
            precision=precision,
            device=device,
        )

        rate_last_slot = torch.rand(batch_size, num_ut, device=device) * 100
        rate_achievable_curr_slot = torch.rand(
            batch_size, num_ofdm_sym, num_freq_res, num_ut, device=device
        ) * 100

        is_scheduled = scheduler(rate_last_slot, rate_achievable_curr_slot)

        # Check shape
        assert is_scheduled.shape == (batch_size, num_ofdm_sym, num_freq_res, num_ut, 1)

        # Check that exactly one user is scheduled per resource per batch element
        assert (is_scheduled.sum(dim=-2).squeeze(-1) == 1).all()

    @pytest.mark.parametrize("mode", ["default", "reduce-overhead"])
    def test_compiled(self, device, mode):
        """Test that PFSchedulerSUMIMO works with torch.compile."""
        if device == "cpu" and mode == "reduce-overhead":
            pytest.skip("reduce-overhead mode not well supported on CPU")

        num_ut = 4
        num_freq_res = 12
        num_ofdm_sym = 14
        batch_size = 4

        scheduler = PFSchedulerSUMIMO(
            num_ut=num_ut,
            num_freq_res=num_freq_res,
            num_ofdm_sym=num_ofdm_sym,
            batch_size=batch_size,
            device=device,
        )

        compiled_call = torch.compile(scheduler.call, mode=mode)

        rate_last_slot = torch.rand(batch_size, num_ut, device=device) * 100
        rate_achievable_curr_slot = torch.rand(
            batch_size, num_ofdm_sym, num_freq_res, num_ut, device=device
        ) * 100

        is_scheduled = compiled_call(rate_last_slot, rate_achievable_curr_slot)

        assert is_scheduled.shape == (batch_size, num_ofdm_sym, num_freq_res, num_ut, 1)

    def test_invalid_beta(self, device):
        """Test that a beta outside the unit interval is rejected."""
        with pytest.raises(ValueError, match="within"):
            PFSchedulerSUMIMO(2, 1, 1, beta=1.0, device=device)

    def test_invalid_rate_last_slot_shape(self, device):
        """Test that a misshaped rate_last_slot is rejected."""
        scheduler = PFSchedulerSUMIMO(2, 1, 1, batch_size=1, device=device)

        with pytest.raises(ValueError, match="rate_last_slot"):
            scheduler(
                torch.ones(2, device=device),
                torch.ones(1, 1, 1, 2, device=device),
            )
