#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Unit tests for TR 38.901 calibration metrics."""

from types import SimpleNamespace

import torch

from sionna.phy.channel.tr38901 import (
    angular_spreads_from_rays,
    circular_angular_spread,
    delay_spread_from_rays,
    prb_singular_values,
    rms_delay_spread,
)


class TestTR38901Metrics:
    """Tests for TR 38.901 metric helpers."""

    def test_rms_delay_spread(self, device, precision):
        dtype = torch.float64 if precision == "double" else torch.float32
        delays = torch.tensor([0.0, 1e-9], dtype=dtype, device=device)
        powers = torch.tensor([1.0, 1.0], dtype=dtype, device=device)
        ds = rms_delay_spread(delays, powers, precision=precision)
        torch.testing.assert_close(ds, torch.tensor(0.5e-9, dtype=dtype,
                                                   device=device))

    def test_circular_angular_spread_annex_a(self, device, precision):
        dtype = torch.float64 if precision == "double" else torch.float32
        angles = torch.deg2rad(torch.tensor([359.0, 1.0], dtype=dtype,
                                            device=device))
        powers = torch.ones(2, dtype=dtype, device=device)
        spread = circular_angular_spread(angles, powers, precision=precision)
        expected = torch.sqrt(
            -2.0*torch.log(torch.cos(torch.deg2rad(
                torch.tensor(1.0, dtype=dtype, device=device)
            )))
        )
        torch.testing.assert_close(spread, expected)

    def test_delay_spread_from_rays(self, device, precision):
        dtype = torch.float64 if precision == "double" else torch.float32
        rays = SimpleNamespace(
            delays=torch.tensor([[[[0.0, 1e-9]]]], dtype=dtype, device=device),
            powers=torch.tensor([[[[1.0, 1.0]]]], dtype=dtype, device=device),
        )
        lsp = SimpleNamespace(
            k_factor=torch.zeros((1, 1, 1), dtype=dtype, device=device)
        )

        class Scenario:
            los = torch.zeros((1, 1, 1), dtype=torch.bool, device=device)

            @staticmethod
            def get_param(name):
                assert name == "cDS"
                return torch.zeros((1, 1, 1), dtype=dtype, device=device)

        ds = delay_spread_from_rays(
            rays, lsp, Scenario(), include_subclusters=False
        )
        torch.testing.assert_close(
            ds, torch.full((1, 1, 1), 0.5e-9, dtype=dtype, device=device)
        )

    def test_delay_spread_splits_strongest_diffuse_clusters(
        self, device, precision
    ):
        """Select subclusters before adding deterministic LoS power."""
        dtype = torch.float64 if precision == "double" else torch.float32
        delays = torch.tensor(
            [[[[0.0, 10e-9, 20e-9]]]], dtype=dtype, device=device
        )
        powers = torch.tensor(
            [[[[0.1, 0.6, 0.3]]]], dtype=dtype, device=device
        )
        k_factor = torch.full((1, 1, 1), 100.0, dtype=dtype, device=device)
        rays = SimpleNamespace(delays=delays, powers=powers)
        lsp = SimpleNamespace(k_factor=k_factor)

        class Scenario:
            los = torch.ones((1, 1, 1), dtype=torch.bool, device=device)

            @staticmethod
            def get_param(name):
                assert name == "cDS"
                return torch.full(
                    (1, 1, 1), 5.0, dtype=dtype, device=device
                )

        actual = delay_spread_from_rays(rays, lsp, Scenario())

        fractions = torch.tensor(
            [10.0, 6.0, 4.0], dtype=dtype, device=device
        ) / 20.0
        offsets = torch.tensor(
            [0.0, 1.28, 2.56], dtype=dtype, device=device
        ) * 5e-9
        adjusted = powers.reshape(-1) / 101.0
        adjusted[0] += 100.0 / 101.0
        expected_delays = torch.cat(
            [delays.reshape(-1)[:1], delays.reshape(-1)[1:2] + offsets,
             delays.reshape(-1)[2:3] + offsets]
        )
        expected_powers = torch.cat(
            [adjusted[:1], adjusted[1:2] * fractions,
             adjusted[2:3] * fractions]
        )
        expected = rms_delay_spread(expected_delays, expected_powers)
        torch.testing.assert_close(actual, expected.reshape(1, 1, 1))

    def test_delay_spread_accepts_single_cluster(self, device, precision):
        """Skip subcluster expansion when fewer than two clusters exist."""
        dtype = torch.float64 if precision == "double" else torch.float32
        rays = SimpleNamespace(
            delays=torch.zeros((1, 1, 1, 1), dtype=dtype, device=device),
            powers=torch.ones((1, 1, 1, 1), dtype=dtype, device=device),
        )
        lsp = SimpleNamespace(
            k_factor=torch.zeros((1, 1, 1), dtype=dtype, device=device)
        )

        class Scenario:
            los = torch.zeros((1, 1, 1), dtype=torch.bool, device=device)

            @staticmethod
            def get_param(name):
                assert name == "cDS"
                return torch.ones((1, 1, 1), dtype=dtype, device=device)

        actual = delay_spread_from_rays(rays, lsp, Scenario())
        torch.testing.assert_close(actual, torch.zeros_like(actual))

    def test_angular_spreads_from_rays_folds_zenith(self, device, precision):
        dtype = torch.float64 if precision == "double" else torch.float32
        zeros = torch.zeros((1, 1, 1), dtype=dtype, device=device)
        powers = torch.ones((1, 1, 1, 1), dtype=dtype, device=device)
        z_rays = torch.deg2rad(
            torch.tensor([[[[[350.0, 10.0]]]]], dtype=dtype, device=device)
        )
        rays = SimpleNamespace(
            powers=powers,
            aod=torch.zeros((1, 1, 1, 1, 2), dtype=dtype, device=device),
            aoa=torch.zeros((1, 1, 1, 1, 2), dtype=dtype, device=device),
            zod=z_rays,
            zoa=z_rays,
        )
        lsp = SimpleNamespace(k_factor=zeros)
        scenario = SimpleNamespace(
            los=torch.zeros((1, 1, 1), dtype=torch.bool, device=device),
            los_aod=zeros,
            los_aoa=zeros,
            los_zod=torch.full((1, 1, 1), 350.0, dtype=dtype, device=device),
            los_zoa=torch.full((1, 1, 1), 350.0, dtype=dtype, device=device),
        )
        spreads = angular_spreads_from_rays(rays, lsp, scenario)
        torch.testing.assert_close(spreads["zsd"], torch.zeros_like(spreads["zsd"]))
        torch.testing.assert_close(spreads["zsa"], torch.zeros_like(spreads["zsa"]))

    def test_prb_singular_values_identity_channel(self, device, precision):
        cdtype = torch.complex128 if precision == "double" else torch.complex64
        h = torch.zeros((1, 2, 2, 1), dtype=cdtype, device=device)
        h[0, :, :, 0] = torch.eye(2, dtype=cdtype, device=device)
        tau = torch.zeros((1, 1), dtype=h.real.dtype, device=device)
        values = prb_singular_values(h, tau, 6e9, 15e3)
        expected = torch.zeros((1, 2), dtype=h.real.dtype, device=device)
        torch.testing.assert_close(values, expected)

    def test_prb_singular_values_ignores_carrier_frequency(self, device, precision):
        """Check the compatibility carrier-frequency argument is a no-op."""
        cdtype = torch.complex128 if precision == "double" else torch.complex64
        h = torch.ones((1, 1, 1, 2), dtype=cdtype, device=device)
        tau = torch.tensor([[0.0, 1e-6]], dtype=h.real.dtype, device=device)
        low = prb_singular_values(h, tau, 6e9, 15e3)
        high = prb_singular_values(h, tau, 70e9, 15e3)
        torch.testing.assert_close(low, high)
