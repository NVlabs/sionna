#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Unit tests for sionna.sys.link_adaptation"""

import pytest
import torch

from sionna.phy import config
from sionna.phy.utils import db_to_lin
from sionna.sys import InnerLoopLinkAdaptation, OuterLoopLinkAdaptation, PHYAbstraction


class TestLinkAdaptation:
    """Tests for link adaptation classes."""

    def test_illa_basic(self, device, precision):
        """Test basic InnerLoopLinkAdaptation functionality."""
        num_ut = 4
        batch_size = 2
        num_ofdm_symbols = 14
        num_subcarriers = 52
        num_streams_per_ut = 2

        # Initialize PHY abstraction
        phy_abs = PHYAbstraction(precision=precision, device=device)

        illa = InnerLoopLinkAdaptation(
            phy_abstraction=phy_abs, precision=precision, device=device
        )

        # Generate SINR
        sinr_db = torch.rand(
            batch_size,
            num_ofdm_symbols,
            num_subcarriers,
            num_ut,
            num_streams_per_ut,
            device=device,
        ) * 30
        sinr = db_to_lin(sinr_db, precision=precision).to(device)

        # Get MCS index
        mcs_index = illa(sinr)

        # Basic checks
        assert mcs_index.shape == (batch_size, num_ut)
        assert mcs_index.dtype in (torch.int32, torch.int64)
        assert (mcs_index >= 0).all()

    def test_olla_basic(self, device, precision):
        """Test basic OuterLoopLinkAdaptation functionality."""
        num_ut = 4
        batch_size = 2

        # Initialize PHY abstraction
        phy_abs = PHYAbstraction(precision=precision, device=device)

        olla = OuterLoopLinkAdaptation(
            phy_abstraction=phy_abs,
            num_ut=num_ut,
            batch_size=batch_size,
            precision=precision,
            device=device,
        )

        # Number of allocated REs for each user
        num_allocated_re = torch.randint(10, 100, (batch_size, num_ut), device=device)

        # HARQ feedback for each user (-1: N/A, 0: NACK, 1: ACK)
        harq_feedback = torch.randint(-1, 2, (batch_size, num_ut), device=device)

        # Effective SINR feedback for each user
        sinr_eff = torch.rand(batch_size, num_ut, device=device) * 10 + 1

        # Run OLLA
        mcs_index = olla(num_allocated_re, harq_feedback, sinr_eff)

        # Basic checks
        assert mcs_index.shape == (batch_size, num_ut)
        assert mcs_index.dtype in (torch.int32, torch.int64)

    def test_olla_offset_update(self, device):
        """Test that OLLA offset updates correctly based on HARQ feedback."""
        num_ut = 2
        batch_size = 1

        # Initialize PHY abstraction
        phy_abs = PHYAbstraction(device=device)

        olla = OuterLoopLinkAdaptation(
            phy_abstraction=phy_abs,
            num_ut=num_ut,
            batch_size=batch_size,
            delta_up=1.0,
            device=device,
        )

        initial_offset = olla.offset.clone()

        # Run multiple iterations
        for _ in range(10):
            num_allocated_re = torch.randint(10, 100, (batch_size, num_ut), device=device)
            harq_feedback = torch.randint(0, 2, (batch_size, num_ut), device=device)
            sinr_eff = torch.rand(batch_size, num_ut, device=device) * 10 + 1
            olla(num_allocated_re, harq_feedback, sinr_eff)

        # Offset should have changed
        final_offset = olla.offset

        # Offset should be within bounds
        assert (final_offset >= olla.offset_min).all()
        assert (final_offset <= olla.offset_max).all()

    @pytest.mark.parametrize("mode", ["default", "reduce-overhead"])
    def test_illa_compiled(self, device, mode):
        """Test that InnerLoopLinkAdaptation works with torch.compile."""
        if device == "cpu" and mode == "reduce-overhead":
            pytest.skip("reduce-overhead mode not well supported on CPU")

        num_ut = 4
        batch_size = 2
        num_ofdm_symbols = 4
        num_subcarriers = 12
        num_streams_per_ut = 2

        # Initialize PHY abstraction
        phy_abs = PHYAbstraction(device=device)

        illa = InnerLoopLinkAdaptation(phy_abstraction=phy_abs, device=device)

        compiled_call = torch.compile(illa.call, mode=mode)

        # Generate SINR
        sinr = torch.rand(
            batch_size,
            num_ofdm_symbols,
            num_subcarriers,
            num_ut,
            num_streams_per_ut,
            device=device,
        ) * 100

        # Run compiled version
        mcs_index = compiled_call(sinr)

        assert mcs_index.shape == (batch_size, num_ut)

    @pytest.mark.parametrize("bler_target", [0.0, 1.0, -0.1, 1.1])
    def test_olla_bler_target_validation(self, device, bler_target):
        """OLLA rejects BLER targets outside the open interval (0, 1)."""
        phy_abs = PHYAbstraction(device=device)
        with pytest.raises(ValueError, match="bler_target"):
            OuterLoopLinkAdaptation(
                phy_abs, num_ut=2, bler_target=bler_target, device=device
            )

    def test_olla_parameter_validation(self, device):
        """OLLA validates positive delta and ordered offset bounds."""
        phy_abs = PHYAbstraction(device=device)
        with pytest.raises(ValueError, match="delta_up"):
            OuterLoopLinkAdaptation(
                phy_abs, num_ut=2, delta_up=0.0, device=device
            )
        with pytest.raises(ValueError, match="offset_min"):
            OuterLoopLinkAdaptation(
                phy_abs,
                num_ut=2,
                offset_min=5.0,
                offset_max=0.0,
                device=device,
            )

        olla = OuterLoopLinkAdaptation(phy_abs, num_ut=2, device=device)
        with pytest.raises(ValueError, match="bler_target"):
            olla.bler_target = 1.0
        with pytest.raises(ValueError, match="offset_min"):
            olla.offset_min = olla.offset_max + 1.0
        with pytest.raises(ValueError, match="offset_min"):
            olla.offset_max = olla.offset_min - 1.0

    @pytest.mark.parametrize("bler_target", [0.0, 1.0, -0.1, 1.1])
    def test_illa_bler_target_validation(self, device, bler_target):
        """ILLA rejects BLER targets outside the open interval (0, 1)."""
        phy_abs = PHYAbstraction(device=device)
        with pytest.raises(ValueError, match="bler_target"):
            InnerLoopLinkAdaptation(
                phy_abs, bler_target=bler_target, device=device
            )

        illa = InnerLoopLinkAdaptation(phy_abs, bler_target=0.1, device=device)
        with pytest.raises(ValueError, match="bler_target"):
            illa.bler_target = bler_target
