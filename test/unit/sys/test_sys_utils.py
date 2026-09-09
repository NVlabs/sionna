#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Unit tests for sionna.sys.utils"""

import numpy as np
import pytest
import torch

from sionna.phy import config
from sionna.sys.utils import get_pathloss, is_scheduled_in_slot, spread_across_subcarriers


class TestSysUtils:
    """Tests for sys utility functions."""

    def test_is_scheduled_in_slot_sinr(self, device, precision):
        """Test is_scheduled_in_slot with SINR input."""
        batch_size = 2
        num_ofdm_symbols = 14
        num_subcarriers = 52
        num_ut = 4
        num_streams_per_ut = 2

        # Create SINR tensor with some zeros (unscheduled)
        sinr = torch.rand(
            batch_size,
            num_ofdm_symbols,
            num_subcarriers,
            num_ut,
            num_streams_per_ut,
            device=device,
        )
        # Set user 1 to unscheduled (all zeros)
        sinr[:, :, :, 1, :] = 0

        is_sched = is_scheduled_in_slot(sinr=sinr)

        # Check shape
        assert is_sched.shape == (batch_size, num_ut)

        # User 1 should not be scheduled
        assert not is_sched[:, 1].any()

        # Other users should be scheduled
        assert is_sched[:, 0].all()
        assert is_sched[:, 2].all()
        assert is_sched[:, 3].all()

    def test_is_scheduled_in_slot_num_re(self, device):
        """Test is_scheduled_in_slot with num_allocated_re input."""
        batch_size = 2
        num_ut = 4

        num_allocated_re = torch.tensor(
            [[10, 0, 5, 8], [0, 12, 3, 0]], dtype=torch.int32, device=device
        )

        is_sched = is_scheduled_in_slot(num_allocated_re=num_allocated_re)

        expected = torch.tensor(
            [[True, False, True, True], [False, True, True, False]], device=device
        )
        assert (is_sched == expected).all()

    def test_get_pathloss(self, device, precision):
        """Test get_pathloss function."""
        batch_size = 2
        num_rx = 4
        num_rx_ant = 1
        num_tx = 4
        num_tx_ant = 4
        num_ofdm_sym = 14
        num_subcarriers = 52

        cdtype = torch.complex64 if precision == "single" else torch.complex128
        h_freq = torch.randn(
            batch_size,
            num_rx,
            num_rx_ant,
            num_tx,
            num_tx_ant,
            num_ofdm_sym,
            num_subcarriers,
            dtype=cdtype,
            device=device,
        )

        pathloss_all, pathloss_serving = get_pathloss(h_freq, precision=precision)

        # Check shape
        assert pathloss_all.shape == (batch_size, num_rx, num_tx, num_ofdm_sym)

        # Check that pathloss is positive
        assert (pathloss_all > 0).all()

    def test_get_pathloss_with_association(self, device, precision):
        """Test get_pathloss with RX-TX association matrix."""
        batch_size = 2
        num_rx = 4
        num_rx_ant = 1
        num_tx = 4
        num_tx_ant = 4
        num_ofdm_sym = 14
        num_subcarriers = 52

        cdtype = torch.complex64 if precision == "single" else torch.complex128
        h_freq = torch.randn(
            batch_size,
            num_rx,
            num_rx_ant,
            num_tx,
            num_tx_ant,
            num_ofdm_sym,
            num_subcarriers,
            dtype=cdtype,
            device=device,
        )

        # Keep the association on CPU to verify cross-device normalization.
        rx_tx_association = torch.eye(num_rx, num_tx, dtype=torch.int32)

        pathloss_all, pathloss_serving = get_pathloss(
            h_freq, rx_tx_association=rx_tx_association, precision=precision
        )

        # Check shapes
        assert pathloss_all.shape == (batch_size, num_rx, num_tx, num_ofdm_sym)
        assert pathloss_serving.shape == (batch_size, num_rx, num_ofdm_sym)
        expected = torch.diagonal(
            pathloss_all, dim1=-3, dim2=-2
        ).movedim(-1, -2)
        torch.testing.assert_close(pathloss_serving, expected)
        assert pathloss_serving.device == h_freq.device

        invalid_association = rx_tx_association.clone()
        invalid_association[0, 0] = 2
        with pytest.raises(ValueError, match="binary"):
            get_pathloss(
                h_freq,
                rx_tx_association=invalid_association,
                precision=precision,
            )

        incomplete_association = rx_tx_association.clone()
        incomplete_association[0, 0] = 0
        with pytest.raises(ValueError, match="one serving link per user"):
            get_pathloss(
                h_freq,
                rx_tx_association=incomplete_association,
                precision=precision,
            )

    @pytest.mark.parametrize(
        "association",
        [
            torch.tensor(
                [[1, 1, 0, 0], [0, 0, 1, 1]], dtype=torch.int32
            ),
            torch.tensor(
                [[1, 0], [1, 0], [0, 1], [0, 1]], dtype=torch.int32
            ),
        ],
        ids=["uplink", "downlink"],
    )
    def test_get_pathloss_rectangular_association(
        self, device, precision, association
    ):
        """All serving links are selected for uplink and downlink layouts."""
        num_rx, num_tx = association.shape
        num_ofdm_sym = 4
        cdtype = (
            torch.complex64 if precision == "single" else torch.complex128
        )
        h_freq = torch.randn(
            2,
            num_rx,
            1,
            num_tx,
            2,
            num_ofdm_sym,
            8,
            dtype=cdtype,
            device=device,
        )

        pathloss_all, pathloss_serving = get_pathloss(
            h_freq,
            rx_tx_association=association,
            precision=precision,
        )

        mask = association.to(device=device, dtype=torch.bool)
        expected = pathloss_all[..., mask, :]
        assert pathloss_serving.shape == (2, 4, num_ofdm_sym)
        assert pathloss_serving.device == h_freq.device
        torch.testing.assert_close(pathloss_serving, expected)

    @pytest.mark.parametrize("batch_shape", [(), (2,)])
    def test_get_pathloss_batched_and_unbatched_association(
        self, device, precision, batch_shape
    ):
        """Association supports channels with or without batch dimensions."""
        num_rx = 3
        num_tx = 3
        num_ofdm_sym = 4
        cdtype = (
            torch.complex64 if precision == "single" else torch.complex128
        )
        h_freq = torch.randn(
            *batch_shape,
            num_rx,
            2,
            num_tx,
            2,
            num_ofdm_sym,
            8,
            dtype=cdtype,
            device=device,
        )
        association = torch.eye(
            num_rx, num_tx, dtype=torch.int32, device=device
        )

        pathloss_without, serving_without = get_pathloss(
            h_freq, precision=precision
        )
        pathloss_with, pathloss_serving = get_pathloss(
            h_freq,
            rx_tx_association=association,
            precision=precision,
        )

        assert serving_without is None
        assert pathloss_with.shape == batch_shape + (
            num_rx,
            num_tx,
            num_ofdm_sym,
        )
        assert pathloss_serving.shape == batch_shape + (
            num_rx,
            num_ofdm_sym,
        )
        torch.testing.assert_close(pathloss_with, pathloss_without)
        expected_serving = torch.diagonal(
            pathloss_with, dim1=-3, dim2=-2
        ).movedim(-1, -2)
        torch.testing.assert_close(pathloss_serving, expected_serving)

    def test_get_pathloss_compiles_fullgraph(self, device, precision):
        """Association validation and selection must remain in one graph."""
        cdtype = torch.complex64 if precision == "single" else torch.complex128
        association = torch.tensor(
            [[1, 1, 0, 0], [0, 0, 1, 1]], dtype=torch.int32
        )

        @torch.compile(fullgraph=True, dynamic=True)
        def compiled_get_pathloss(h_freq):
            return get_pathloss(
                h_freq,
                rx_tx_association=association,
                precision=precision,
            )

        for batch_size in (2, 3):
            h_freq = torch.randn(
                batch_size,
                2,
                1,
                4,
                2,
                4,
                12,
                dtype=cdtype,
                device=device,
            )
            actual = compiled_get_pathloss(h_freq)
            expected = get_pathloss(
                h_freq,
                rx_tx_association=association,
                precision=precision,
            )
            torch.testing.assert_close(actual, expected)
            assert actual[1].device == h_freq.device

    def test_spread_across_subcarriers(self, device, precision):
        """Test spread_across_subcarriers function."""
        from sionna.phy import dtypes
        dtype = dtypes[precision]["torch"]["dtype"]
        
        batch_size = 2
        num_ofdm_sym = 14
        num_ut = 4
        num_subcarriers = 52
        num_streams = 2

        tx_power_per_ut = torch.ones(
            batch_size, num_ofdm_sym, num_ut, dtype=dtype, device=device
        )
        is_scheduled = torch.ones(
            batch_size,
            num_ofdm_sym,
            num_subcarriers,
            num_ut,
            num_streams,
            dtype=torch.bool,
            device=device,
        )

        tx_power = spread_across_subcarriers(
            tx_power_per_ut, is_scheduled, precision=precision
        )

        # Check shape
        assert tx_power.shape == (batch_size, num_ut, num_streams, num_ofdm_sym, num_subcarriers)

        # Total power per user should equal input power
        total_power = tx_power.sum(dim=(-1, -2, -3))
        expected = tx_power_per_ut.sum(dim=1)
        assert torch.allclose(total_power, expected, rtol=1e-4)

    @pytest.mark.parametrize("mode", ["default", "reduce-overhead"])
    def test_is_scheduled_compiled(self, device, mode):
        """Test that is_scheduled_in_slot works with torch.compile."""
        if device == "cpu" and mode == "reduce-overhead":
            pytest.skip("reduce-overhead mode not well supported on CPU")

        batch_size = 2
        num_ofdm_symbols = 4
        num_subcarriers = 12
        num_ut = 4
        num_streams_per_ut = 2

        sinr = torch.rand(
            batch_size,
            num_ofdm_symbols,
            num_subcarriers,
            num_ut,
            num_streams_per_ut,
            device=device,
        )

        compiled_fn = torch.compile(is_scheduled_in_slot, mode=mode)

        is_sched = compiled_fn(sinr=sinr)

        assert is_sched.shape == (batch_size, num_ut)
