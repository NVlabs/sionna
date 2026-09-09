#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Unit tests for sionna.sys.metrics"""

import pytest
import torch

from sionna.sys import (
    coupling_loss_db,
    geometry_sinr_db,
    geometry_sir_db,
    received_power_dbm,
    serving_indices,
    wideband_sir_db,
)


class TestSysMetrics:
    """Tests for system-level metric helpers."""

    def test_coupling_loss_and_received_power(self, device, precision):
        path_gain = torch.tensor([-120.0, -100.0], device=device)
        coupling_loss = coupling_loss_db(path_gain, precision=precision, device=device)
        expected_cl = torch.tensor([120.0, 100.0], dtype=coupling_loss.dtype,
                                   device=device)
        torch.testing.assert_close(coupling_loss, expected_cl)

        rx_power = received_power_dbm(46.0, coupling_loss, precision=precision)
        expected_rx = torch.tensor([-74.0, -54.0], dtype=rx_power.dtype,
                                   device=device)
        torch.testing.assert_close(rx_power, expected_rx)

    def test_serving_indices_and_sir(self, device, precision):
        coupling_loss = torch.tensor([[100.0, 110.0, 120.0],
                                      [115.0, 105.0, 125.0]], device=device)
        serving = serving_indices(coupling_loss)
        expected_serving = torch.tensor([0, 1], dtype=torch.int64, device=device)
        torch.testing.assert_close(serving, expected_serving)

        sir = geometry_sir_db(coupling_loss, serving=serving,
                              precision=precision)
        expected_value = 10.0*torch.log10(
            torch.tensor(1.0/(0.1 + 0.01), dtype=sir.dtype, device=device)
        )
        expected = torch.ones(2, dtype=sir.dtype, device=device)*expected_value
        torch.testing.assert_close(sir, expected)

        wideband = wideband_sir_db(coupling_loss, serving=serving,
                                   precision=precision)
        torch.testing.assert_close(wideband, sir)

    def test_sir_preserves_weak_interference(self, device):
        """Avoid cancellation when desired power dominates in float32."""
        coupling_loss = torch.tensor(
            [[0.0, 100.0]], dtype=torch.float32, device=device
        )
        sir = geometry_sir_db(coupling_loss, precision="single")
        torch.testing.assert_close(
            sir,
            torch.tensor([100.0], dtype=torch.float32, device=device),
            rtol=1e-6,
            atol=1e-5,
        )

    @pytest.mark.parametrize("offset", [-10000.0, 10000.0])
    def test_sir_is_invariant_to_large_common_offsets(
        self, device, precision, offset
    ):
        coupling_loss = torch.tensor(
            [[0.0, 10.0, 20.0]], device=device
        )
        expected = geometry_sir_db(
            coupling_loss, precision=precision, device=device
        )
        shifted = geometry_sir_db(
            coupling_loss + offset, precision=precision, device=device
        )
        wideband = wideband_sir_db(
            coupling_loss + offset, precision=precision, device=device
        )

        assert torch.all(torch.isfinite(shifted))
        torch.testing.assert_close(shifted, expected)
        torch.testing.assert_close(wideband, expected)

    def test_sir_handles_extreme_dynamic_range(self, device, precision):
        coupling_loss = torch.tensor(
            [[0.0, 1000.0], [0.0, -1000.0]], device=device
        )
        serving = torch.zeros(2, dtype=torch.int64, device=device)
        sir = geometry_sir_db(
            coupling_loss,
            serving=serving,
            precision=precision,
            device=device,
        )
        expected = torch.tensor(
            [1000.0, -1000.0], dtype=sir.dtype, device=device
        )
        torch.testing.assert_close(sir, expected)

    def test_geometry_sinr(self, device, precision):
        coupling_loss = torch.tensor([[100.0, 110.0]], device=device)
        sinr = geometry_sinr_db(
            coupling_loss,
            tx_power_dbm=46.0,
            bandwidth_hz=20e6,
            noise_figure_db=9.0,
            precision=precision,
        )

        signal_mw = 10.0**((-54.0)/10.0)
        interference_mw = 10.0**((-64.0)/10.0)
        noise_mw = 10.0**((-174.0 + 10.0*torch.log10(torch.tensor(20e6)) + 9.0)/10.0)
        expected = 10.0*torch.log10(
            torch.tensor(signal_mw, dtype=sinr.dtype, device=device)
            / (torch.tensor(interference_mw, dtype=sinr.dtype, device=device)
               + noise_mw.to(dtype=sinr.dtype, device=device))
        )
        torch.testing.assert_close(sinr, expected.reshape(1))

    def test_geometry_sinr_compiles_fullgraph(self, device, precision):
        """Bandwidth validation must not split the compiled metric."""

        @torch.compile(fullgraph=True)
        def compiled_geometry(coupling_loss, tx_power, bandwidth, noise_figure):
            return geometry_sinr_db(
                coupling_loss,
                tx_power_dbm=tx_power,
                bandwidth_hz=bandwidth,
                noise_figure_db=noise_figure,
                precision=precision,
            )

        args = (
            torch.tensor([[100.0, 110.0]], device=device),
            torch.tensor(46.0, device=device),
            torch.tensor(20e6, device=device),
            torch.tensor(9.0, device=device),
        )
        actual = compiled_geometry(*args)
        expected = geometry_sinr_db(
            args[0],
            tx_power_dbm=args[1],
            bandwidth_hz=args[2],
            noise_figure_db=args[3],
            precision=precision,
        )
        torch.testing.assert_close(actual, expected)

    @pytest.mark.parametrize("bandwidth_hz", [1e-30, 1e30])
    def test_geometry_sinr_handles_finite_extremes(
        self, device, precision, bandwidth_hz
    ):
        sinr = geometry_sinr_db(
            [[10000.0]],
            tx_power_dbm=46.0,
            bandwidth_hz=bandwidth_hz,
            noise_figure_db=9.0,
            precision=precision,
            device=device,
        )
        bandwidth = torch.tensor(
            bandwidth_hz, dtype=sinr.dtype, device=device
        )
        desired_dbm = torch.tensor(
            46.0 - 10000.0, dtype=sinr.dtype, device=device
        )
        noise_dbm = -174.0 + 10.0 * torch.log10(bandwidth) + 9.0
        expected = (desired_dbm - noise_dbm).reshape(1)

        assert torch.all(torch.isfinite(sinr))
        torch.testing.assert_close(sinr, expected)

    @pytest.mark.parametrize(
        "bandwidth_hz", [0.0, -1.0, float("nan"), float("inf")]
    )
    def test_geometry_sinr_rejects_invalid_bandwidth(
        self, device, precision, bandwidth_hz
    ):
        with pytest.raises(ValueError, match="finite, positive"):
            geometry_sinr_db(
                [[100.0, 110.0]],
                tx_power_dbm=46.0,
                bandwidth_hz=bandwidth_hz,
                noise_figure_db=9.0,
                precision=precision,
                device=device,
            )
