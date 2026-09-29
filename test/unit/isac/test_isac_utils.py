#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Tests for Sionna PHY ISAC utility functions."""

import pytest
import torch

from sionna.phy import PI, dtypes
from sionna.phy.channel import (
    cir_to_ofdm_channel,
    ofdm_to_delay_doppler_channel,
    subcarrier_frequencies,
)
from sionna.phy.isac import (
    angular_delay_doppler_spectrum,
    steering_vectors,
)


class TestSteeringVectors:
    """Tests for steering_vectors."""

    def test_broadside_and_normalization(self, device, precision):
        """Broadside weights have equal phase and unit norm."""
        rdtype = dtypes[precision]["torch"]["dtype"]
        cdtype = dtypes[precision]["torch"]["cdtype"]
        positions = torch.tensor(
            [[0.0, -0.075, 0.0], [0.0, -0.025, 0.0],
             [0.0, 0.025, 0.0], [0.0, 0.075, 0.0]],
            dtype=rdtype,
            device=device,
        )

        w = steering_vectors(
            positions,
            PI / 2,
            0.0,
            wavelength=0.1,
            mode="paired",
            precision=precision,
        )

        expected = torch.full((4,), 0.5, dtype=cdtype, device=device)
        assert w.dtype == cdtype
        assert w.device == positions.device
        assert torch.allclose(w, expected)
        assert torch.allclose(
            torch.norm(w),
            torch.tensor(1.0, dtype=rdtype, device=device),
        )

    def test_known_phase(self, device):
        """A quarter-wavelength displacement produces a +pi/2 phase."""
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [0.0, 0.025, 0.0]], device=device
        )
        w = steering_vectors(
            positions, PI / 2, PI / 2, wavelength=0.1, mode="paired"
        )
        expected = torch.tensor(
            [1.0 + 0.0j, 0.0 + 1.0j],
            dtype=torch.complex64,
            device=device,
        ) / torch.sqrt(torch.tensor(2.0, device=device))
        assert torch.allclose(w, expected, atol=1e-6)

    def test_matches_channel_model_array_response(self, device):
        """The output is the Sionna array response, normalized to unit norm."""
        wavelength = 0.1
        positions = torch.randn(5, 3, device=device)
        theta = torch.tensor(1.1, device=device)
        phi = torch.tensor(0.4, device=device)

        direction = torch.tensor(
            [
                torch.sin(theta) * torch.cos(phi),
                torch.sin(theta) * torch.sin(phi),
                torch.cos(theta),
            ],
            device=device,
        )
        # Sionna channel models use exp(+j 2 pi / lambda * d . r_hat)
        phase = 2 * PI / wavelength * (positions @ direction)
        expected = torch.polar(torch.ones_like(phase), phase) / 5**0.5

        a = steering_vectors(
            positions, theta, phi, wavelength, mode="paired"
        )
        assert torch.allclose(a, expected, atol=1e-5)

    def test_paired_and_cartesian_modes(self, device):
        """Paired and Cartesian modes follow their documented ordering."""
        positions = torch.tensor(
            [[0.0, -0.025, 0.0], [0.0, 0.025, 0.0]], device=device
        )
        theta = torch.tensor([PI / 3, PI / 2], device=device)
        phi = torch.tensor([-0.4, 0.2, 0.7], device=device)

        # Cartesian is the default mode
        cartesian = steering_vectors(positions, theta, phi, 0.1)
        assert cartesian.shape == (2, 3, 2)
        for i in range(2):
            for j in range(3):
                expected = steering_vectors(
                    positions, theta[i], phi[j], 0.1, mode="paired"
                )
                assert torch.allclose(cartesian[i, j], expected)

        paired = steering_vectors(
            positions, theta, phi[:2], 0.1, mode="paired"
        )
        assert paired.shape == (2, 2)
        assert torch.allclose(paired[0], cartesian[0, 0])
        assert torch.allclose(paired[1], cartesian[1, 1])

    def test_cartesian_keeps_degenerate_axes(self, device):
        """Scalar angles contribute a Cartesian axis of length one."""
        positions = torch.zeros(3, 3, device=device)
        theta = torch.tensor([1.0, PI / 2, 2.0], device=device)
        phi = torch.tensor([-0.5, 0.0, 0.5], device=device)

        assert steering_vectors(positions, PI / 2, 0.0, 0.1).shape == (1, 1, 3)
        assert steering_vectors(positions, PI / 2, phi, 0.1).shape == (1, 3, 3)
        assert steering_vectors(
            positions, theta, 0.0, 0.1
        ).shape == (3, 1, 3)

    def test_paired_broadcasting(self, device):
        """A scalar angle broadcasts over the other paired angle."""
        positions = torch.zeros(3, 3, device=device)
        phi = torch.tensor([-0.5, 0.0, 0.5], device=device)
        w = steering_vectors(positions, PI / 2, phi, 0.1, mode="paired")
        assert w.shape == (3, 3)

    def test_validation(self):
        """Rounded angle endpoints are valid; invalid inputs are rejected."""
        positions = torch.zeros(2, 3)
        theta = torch.linspace(0.0, PI, 3, dtype=torch.float32)
        w = steering_vectors(positions, theta, 0.0, 0.1, precision="double")
        assert w.shape == (3, 1, 2)
        assert w.dtype == torch.complex128

        with pytest.raises(ValueError, match="shape"):
            steering_vectors(torch.zeros(2, 2), 0.0, 0.0, 0.1)
        with pytest.raises(ValueError, match="strictly positive"):
            steering_vectors(positions, 0.0, 0.0, 0.0)
        with pytest.raises(ValueError, match="must not be empty"):
            steering_vectors(positions, torch.tensor([]), 0.0, 0.1)
        with pytest.raises(ValueError, match="\\[0, pi\\]"):
            steering_vectors(positions, -0.1, 0.0, 0.1)
        with pytest.raises(ValueError, match="\\[0, pi\\]"):
            steering_vectors(positions, theta + 0.1, 0.0, 0.1,
                             precision="double")
        with pytest.raises(ValueError, match="cartesian"):
            steering_vectors(
                positions,
                torch.zeros(1, 2),
                torch.zeros(2),
                0.1,
                mode="cartesian",
            )
        with pytest.raises(ValueError, match="one of"):
            steering_vectors(positions, 0.0, 0.0, 0.1, mode="invalid")
        with pytest.raises(TypeError, match="`theta` must be real-valued"):
            steering_vectors(positions, torch.zeros(2, dtype=torch.cfloat),
                             0.0, 0.1)
        with pytest.raises(TypeError, match="`phi` must be real-valued"):
            steering_vectors(positions, 0.0,
                             torch.zeros(2, dtype=torch.cfloat), 0.1)

    def test_compile(self, device):
        """The tensor computation can be captured as a full graph."""
        positions = torch.randn(4, 3, device=device)
        theta = torch.tensor([0.5, 1.0], device=device)
        phi = torch.tensor([-0.2, 0.3], device=device)

        def function(p, t, a):
            return steering_vectors(p, t, a, 0.1)

        compiled = torch.compile(function, backend="inductor", fullgraph=True)
        assert torch.allclose(
            compiled(positions, theta, phi),
            function(positions, theta, phi),
        )


class TestAngularDelayDopplerSpectrum:
    """Tests for angular_delay_doppler_spectrum."""

    def test_cartesian_matches_explicit_loops(self, device):
        """Cartesian projection agrees with explicit matrix products."""
        h_dd = torch.randn(
            2, 2, 3, 2, 4, 5, 6, dtype=torch.complex64, device=device
        )
        w_rx = torch.randn(2, 3, dtype=torch.complex64, device=device)
        w_tx = torch.randn(3, 4, dtype=torch.complex64, device=device)

        spectrum = angular_delay_doppler_spectrum(
            h_dd, w_rx, w_tx, mode="cartesian"
        )
        expected = torch.empty(
            2, 2, 2, 2, 3, 5, 6, dtype=torch.float32, device=device
        )
        for b in range(2):
            for r in range(2):
                for i in range(2):
                    for t in range(2):
                        for j in range(3):
                            value = torch.einsum(
                                "m,mnql,n->ql",
                                w_rx[i].conj(),
                                h_dd[b, r, :, t],
                                w_tx[j].conj(),
                            )
                            expected[b, r, i, t, j] = value.abs().square()

        assert spectrum.shape == expected.shape
        assert torch.allclose(spectrum, expected, rtol=1e-5, atol=1e-5)

    def test_device_specific_banks(self, device):
        """Each transmitter and receiver can use a distinct steering bank."""
        h_dd = torch.randn(
            1, 2, 3, 2, 4, 2, 3, dtype=torch.complex64, device=device
        )
        w_rx = torch.randn(2, 5, 3, dtype=torch.complex64, device=device)
        w_tx = torch.randn(2, 6, 4, dtype=torch.complex64, device=device)

        spectrum = angular_delay_doppler_spectrum(
            h_dd, w_rx, w_tx, mode="cartesian"
        )

        assert spectrum.shape == (1, 2, 5, 2, 6, 2, 3)
        value = torch.einsum(
            "m,mn,n->",
            w_rx[1, 3].conj(),
            h_dd[0, 1, :, 0, :, 1, 2],
            w_tx[0, 4].conj(),
        )
        assert torch.allclose(spectrum[0, 1, 3, 0, 4, 1, 2], value.abs() ** 2)

        paired = angular_delay_doppler_spectrum(h_dd, w_rx, w_tx[:, :5])
        expected = torch.stack([spectrum[:, :, i, :, i] for i in range(5)], dim=3)
        assert torch.allclose(paired, expected, rtol=1e-5, atol=1e-5)

    def test_dtype_and_shared_bank_broadcasting(self, device, precision):
        """Output precision follows the channel and shared banks broadcast."""
        rdtype = dtypes[precision]["torch"]["dtype"]
        cdtype = dtypes[precision]["torch"]["cdtype"]
        h_dd = torch.randn(
            2, 2, 3, 2, 4, 2, 3, dtype=cdtype, device=device
        )
        w_rx = torch.randn(5, 3, dtype=cdtype, device=device)
        w_tx = torch.randn(6, 4, dtype=cdtype, device=device)

        shared = angular_delay_doppler_spectrum(
            h_dd, w_rx, w_tx, mode="cartesian"
        )
        specific = angular_delay_doppler_spectrum(
            h_dd,
            w_rx[None].expand(2, -1, -1),
            w_tx[None].expand(2, -1, -1),
            mode="cartesian",
        )

        assert shared.dtype == rdtype
        assert shared.device == h_dd.device
        assert torch.allclose(shared, specific)

    def test_dtype_promotion(self, device):
        """Mixed inputs are promoted rather than cast down to the channel."""
        h_dd = torch.randn(
            1, 1, 2, 1, 3, 2, 3, dtype=torch.complex64, device=device
        )
        w_rx = torch.randn(4, 2, dtype=torch.complex128, device=device)
        w_tx = torch.randn(4, 3, dtype=torch.complex128, device=device)

        promoted = angular_delay_doppler_spectrum(h_dd, w_rx, w_tx)
        assert promoted.dtype == torch.float64
        assert torch.allclose(
            promoted,
            angular_delay_doppler_spectrum(
                h_dd.to(torch.complex128), w_rx, w_tx
            ),
        )

        # A real-valued bank is promoted to complex with a zero imaginary part
        real_bank = angular_delay_doppler_spectrum(h_dd, w_rx.real, w_tx.real)
        assert real_bank.dtype == torch.float64
        assert torch.allclose(
            real_bank,
            angular_delay_doppler_spectrum(
                h_dd,
                w_rx.real.to(torch.complex128),
                w_tx.real.to(torch.complex128),
            ),
        )

    def test_paired_mode_and_singleton_broadcast(self, device):
        """All contraction orders preserve paired singleton broadcasting."""
        for num_rx_ant, num_tx_ant, num_doppler, num_delay in (
            (2, 3, 4, 5),  # Combine steering banks first
            (3, 2, 1, 1),  # Project receive antennas first
            (2, 3, 1, 1),  # Project transmit antennas first
        ):
            h_dd = torch.randn(
                1, 1, num_rx_ant, 1, num_tx_ant, num_doppler, num_delay,
                dtype=torch.complex64, device=device,
            )
            w_rx = torch.randn(4, num_rx_ant, dtype=torch.complex64, device=device)
            w_tx = torch.randn(1, num_tx_ant, dtype=torch.complex64, device=device)

            # Paired is the default mode
            spectrum = angular_delay_doppler_spectrum(h_dd, w_rx, w_tx)

            assert spectrum.shape == (1, 1, 1, 4, num_doppler, num_delay)
            for i in range(4):
                value = torch.einsum(
                    "m,mnql,n->ql",
                    w_rx[i].conj(),
                    h_dd[0, 0, :, 0],
                    w_tx[0].conj(),
                )
                assert torch.allclose(
                    spectrum[0, 0, 0, i], value.abs().square(),
                    rtol=1e-5, atol=1e-6,
                )

    def test_validation(self):
        """Invalid channel and steering-bank shapes are rejected."""
        h_dd = torch.zeros(1, 1, 2, 1, 3, 4, 5, dtype=torch.complex64)
        with pytest.raises(TypeError, match="complex"):
            angular_delay_doppler_spectrum(h_dd.real, torch.zeros(1, 2),
                                           torch.zeros(1, 3))
        with pytest.raises(ValueError, match="antenna coefficients"):
            angular_delay_doppler_spectrum(h_dd, torch.zeros(1, 2),
                                           torch.zeros(1, 4))
        with pytest.raises(TypeError, match="`rx_steering_vectors` must be"):
            angular_delay_doppler_spectrum(h_dd, [[0.0, 0.0]],
                                           torch.zeros(1, 3))
        with pytest.raises(ValueError, match="direction counts"):
            angular_delay_doppler_spectrum(
                h_dd,
                torch.zeros(3, 2),
                torch.zeros(2, 3),
            )

    def test_compile(self, device):
        """Paired and Cartesian spectra can be captured as a full graph."""
        h_dd = torch.randn(
            1, 1, 2, 1, 3, 4, 5, dtype=torch.complex64, device=device
        )
        w_rx = torch.randn(2, 2, dtype=torch.complex64, device=device)
        w_tx = torch.randn(3, 3, dtype=torch.complex64, device=device)

        def function(h, rx, tx):
            return (
                angular_delay_doppler_spectrum(h, rx, tx, mode="cartesian"),
                angular_delay_doppler_spectrum(h, rx, tx[:2]),
            )

        compiled = torch.compile(function, backend="inductor", fullgraph=True)
        for actual, expected in zip(
            compiled(h_dd, w_rx, w_tx), function(h_dd, w_rx, w_tx)
        ):
            assert torch.allclose(actual, expected)

    def test_recovers_angle_delay_and_doppler(self, device):
        """The spectrum peak recovers an on-grid monostatic target."""
        num_rx_ant = 4
        num_tx_ant = 5
        num_time_steps = 8
        fft_size = 16
        subcarrier_spacing = 1e6
        wavelength = 0.1
        delay_bin = 3
        doppler_bin = 2

        rx_positions = torch.zeros(num_rx_ant, 3, device=device)
        tx_positions = torch.zeros(num_tx_ant, 3, device=device)
        rx_positions[:, 1] = (
            torch.arange(num_rx_ant, device=device)
            - (num_rx_ant - 1) / 2
        ) * wavelength / 2
        tx_positions[:, 1] = (
            torch.arange(num_tx_ant, device=device)
            - (num_tx_ant - 1) / 2
        ) * wavelength / 2

        phi_candidates = torch.tensor(
            [-0.6, -0.2, 0.2, 0.6], device=device
        )
        theta_candidates = torch.full_like(phi_candidates, PI / 2)
        w_rx = steering_vectors(
            rx_positions,
            theta_candidates,
            phi_candidates,
            wavelength,
            mode="paired",
        )
        w_tx = steering_vectors(
            tx_positions,
            theta_candidates,
            phi_candidates,
            wavelength,
            mode="paired",
        )
        true_angle_index = 2
        a_rx = w_rx[true_angle_index] * num_rx_ant**0.5
        a_tx = w_tx[true_angle_index] * num_tx_ant**0.5
        spatial_response = a_rx[:, None] * a_tx[None, :]

        time_index = torch.arange(num_time_steps, device=device)
        doppler_phase = 2 * PI * doppler_bin * time_index / num_time_steps
        temporal_response = torch.polar(
            torch.ones_like(doppler_phase), doppler_phase
        )
        a = (
            spatial_response[None, None, :, None, :, None, None]
            * temporal_response[None, None, None, None, None, None, :]
        )
        bandwidth = fft_size * subcarrier_spacing
        tau = torch.tensor(
            [[[[delay_bin / bandwidth]]]], device=device
        )
        frequencies = subcarrier_frequencies(
            fft_size, subcarrier_spacing, device=device
        )
        h_f = cir_to_ofdm_channel(frequencies, a, tau)
        h_dd = ofdm_to_delay_doppler_channel(h_f)
        spectrum = angular_delay_doppler_spectrum(
            h_dd, w_rx, w_tx, mode="cartesian"
        )

        peak = torch.unravel_index(torch.argmax(spectrum), spectrum.shape)
        peak = tuple(int(index) for index in peak)
        centered_doppler_bin = (
            doppler_bin + num_time_steps // 2
        ) % num_time_steps
        assert peak == (
            0,
            0,
            true_angle_index,
            0,
            true_angle_index,
            centered_doppler_bin,
            delay_bin,
        )
