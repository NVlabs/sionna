#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Tests for sionna.phy.ofdm.equalization module."""

import pytest
import numpy as np
import torch

from sionna.phy import config
from sionna.phy.ofdm import (
    ResourceGrid,
    OFDMEqualizer,
    LMMSEEqualizer,
    ZFEqualizer,
    MFEqualizer,
    LMMSEPostEqualizationSINR,
    RZFPrecodedChannel,
    EyePrecodedChannel,
)
from sionna.phy.mimo import StreamManagement, lmmse_equalizer, lmmse_matrix
from sionna.phy.channel import (
    RayleighBlockFading,
    cir_to_ofdm_channel,
    subcarrier_frequencies,
)
from sionna.phy.utils import complex_normal, inv_cholesky, expand_to_rank


@pytest.fixture
def resource_grid():
    """Create a simple resource grid for testing."""
    return ResourceGrid(
        num_ofdm_symbols=14,
        fft_size=64,
        subcarrier_spacing=30e3,
        num_tx=2,
        num_streams_per_tx=2,
        pilot_pattern="kronecker",
        pilot_ofdm_symbol_indices=[2, 11],
    )


@pytest.fixture
def stream_management():
    """Create a StreamManagement for testing."""
    rx_tx_association = np.ones([1, 2])
    return StreamManagement(rx_tx_association, num_streams_per_tx=2)


class TestLMMSEEqualizer:
    """Tests for LMMSEEqualizer class."""

    def test_output_shape(self, device, precision, resource_grid, stream_management):
        """Test that LMMSEEqualizer produces correct output shape."""
        cdtype = torch.complex64 if precision == "single" else torch.complex128
        dtype = torch.float32 if precision == "single" else torch.float64
        config.precision = "double" if precision == "single" else "single"

        equalizer = LMMSEEqualizer(
            resource_grid=resource_grid,
            stream_management=stream_management,
            precision=precision,
            device=device,
        )

        batch_size = 4
        num_rx = 1
        num_rx_ant = 4
        num_ofdm_symbols = resource_grid.num_ofdm_symbols
        fft_size = resource_grid.fft_size
        num_tx = resource_grid.num_tx
        num_streams_per_tx = resource_grid.num_streams_per_tx
        num_effective_subcarriers = resource_grid.num_effective_subcarriers
        num_data_symbols = resource_grid.pilot_pattern.num_data_symbols

        y = complex_normal(
            (batch_size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size),
            precision=precision,
            device=device,
        )
        h_hat = complex_normal(
            (
                batch_size,
                num_rx,
                num_rx_ant,
                num_tx,
                num_streams_per_tx,
                num_ofdm_symbols,
                num_effective_subcarriers,
            ),
            precision=precision,
            device=device,
        )
        err_var = torch.ones(1, device=device) * 0.01
        no = torch.ones(1, device=device) * 0.1

        x_hat, no_eff = equalizer(y, h_hat, err_var, no)

        expected_shape = (batch_size, num_tx, num_streams_per_tx, num_data_symbols)
        assert x_hat.shape == expected_shape
        assert no_eff.shape == expected_shape
        assert x_hat.dtype == cdtype
        assert no_eff.dtype == dtype

    def test_preserves_callable_contract_and_casts_outputs(
        self, device, resource_grid, stream_management
    ):
        """Custom callables keep three inputs and outputs are normalized."""
        def single_precision_equalizer(y, h, s):
            return lmmse_equalizer(y, h, s, precision="single")

        equalizer = OFDMEqualizer(
            equalizer=single_precision_equalizer,
            resource_grid=resource_grid,
            stream_management=stream_management,
            precision="double",
            device=device,
        )
        batch_size = 1
        y = complex_normal(
            (
                batch_size,
                1,
                4,
                resource_grid.num_ofdm_symbols,
                resource_grid.fft_size,
            ),
            precision="double",
            device=device,
        )
        h_hat = complex_normal(
            (
                batch_size,
                1,
                4,
                resource_grid.num_tx,
                resource_grid.num_streams_per_tx,
                resource_grid.num_ofdm_symbols,
                resource_grid.num_effective_subcarriers,
            ),
            precision="double",
            device=device,
        )

        x_hat, no_eff = equalizer(y, h_hat, 0.01, 0.1)

        assert x_hat.dtype == torch.complex128
        assert no_eff.dtype == torch.float64

    def test_scalar_err_var(self, device, precision, resource_grid, stream_management):
        """Test that equalizer works with scalar float err_var (perfect CSI case)."""
        equalizer = LMMSEEqualizer(
            resource_grid=resource_grid,
            stream_management=stream_management,
            precision=precision,
            device=device,
        )

        batch_size = 4
        num_rx = 1
        num_rx_ant = 4
        num_ofdm_symbols = resource_grid.num_ofdm_symbols
        fft_size = resource_grid.fft_size
        num_tx = resource_grid.num_tx
        num_streams_per_tx = resource_grid.num_streams_per_tx
        num_effective_subcarriers = resource_grid.num_effective_subcarriers
        num_data_symbols = resource_grid.pilot_pattern.num_data_symbols

        y = complex_normal(
            (batch_size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size),
            precision=precision,
            device=device,
        )
        h_hat = complex_normal(
            (
                batch_size,
                num_rx,
                num_rx_ant,
                num_tx,
                num_streams_per_tx,
                num_ofdm_symbols,
                num_effective_subcarriers,
            ),
            precision=precision,
            device=device,
        )
        # Use scalar float for err_var (simulates perfect CSI case)
        err_var = 0.0
        no = torch.ones(1, device=device) * 0.1

        # Should not raise AttributeError
        x_hat, no_eff = equalizer(y, h_hat, err_var, no)

        expected_shape = (batch_size, num_tx, num_streams_per_tx, num_data_symbols)
        assert x_hat.shape == expected_shape
        assert no_eff.shape == expected_shape

    @pytest.mark.parametrize("whiten_interference", [True, False])
    def test_whiten_interference(
        self, device, precision, resource_grid, stream_management, whiten_interference
    ):
        """Test that interference whitening option works."""
        cdtype = torch.complex64 if precision == "single" else torch.complex128

        equalizer = LMMSEEqualizer(
            resource_grid=resource_grid,
            stream_management=stream_management,
            whiten_interference=whiten_interference,
            precision=precision,
            device=device,
        )

        batch_size = 2
        num_rx = 1
        num_rx_ant = 4
        num_ofdm_symbols = resource_grid.num_ofdm_symbols
        fft_size = resource_grid.fft_size
        num_tx = resource_grid.num_tx
        num_streams_per_tx = resource_grid.num_streams_per_tx
        num_effective_subcarriers = resource_grid.num_effective_subcarriers

        y = complex_normal(
            (batch_size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size),
            precision=precision,
            device=device,
        )
        h_hat = complex_normal(
            (
                batch_size,
                num_rx,
                num_rx_ant,
                num_tx,
                num_streams_per_tx,
                num_ofdm_symbols,
                num_effective_subcarriers,
            ),
            precision=precision,
            device=device,
        )
        err_var = torch.ones(1, device=device) * 0.01
        no = torch.ones(1, device=device) * 0.1

        x_hat, no_eff = equalizer(y, h_hat, err_var, no)

        # Should produce valid output
        assert not torch.isnan(x_hat).any()
        assert not torch.isnan(no_eff).any()


class TestZFEqualizer:
    """Tests for ZFEqualizer class."""

    def test_output_shape(self, device, precision, resource_grid, stream_management):
        """Test that ZFEqualizer produces correct output shape."""
        cdtype = torch.complex64 if precision == "single" else torch.complex128
        dtype = torch.float32 if precision == "single" else torch.float64
        config.precision = "double" if precision == "single" else "single"

        equalizer = ZFEqualizer(
            resource_grid=resource_grid,
            stream_management=stream_management,
            precision=precision,
            device=device,
        )

        batch_size = 4
        num_rx = 1
        num_rx_ant = 4
        num_ofdm_symbols = resource_grid.num_ofdm_symbols
        fft_size = resource_grid.fft_size
        num_tx = resource_grid.num_tx
        num_streams_per_tx = resource_grid.num_streams_per_tx
        num_effective_subcarriers = resource_grid.num_effective_subcarriers
        num_data_symbols = resource_grid.pilot_pattern.num_data_symbols

        y = complex_normal(
            (batch_size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size),
            precision=precision,
            device=device,
        )
        h_hat = complex_normal(
            (
                batch_size,
                num_rx,
                num_rx_ant,
                num_tx,
                num_streams_per_tx,
                num_ofdm_symbols,
                num_effective_subcarriers,
            ),
            precision=precision,
            device=device,
        )
        err_var = torch.ones(1, device=device) * 0.01
        no = torch.ones(1, device=device) * 0.1

        x_hat, no_eff = equalizer(y, h_hat, err_var, no)

        expected_shape = (batch_size, num_tx, num_streams_per_tx, num_data_symbols)
        assert x_hat.shape == expected_shape
        assert no_eff.shape == expected_shape
        assert x_hat.dtype == cdtype
        assert no_eff.dtype == dtype


class TestMFEqualizer:
    """Tests for MFEqualizer class."""

    def test_output_shape(self, device, precision, resource_grid, stream_management):
        """Test that MFEqualizer produces correct output shape."""
        cdtype = torch.complex64 if precision == "single" else torch.complex128
        dtype = torch.float32 if precision == "single" else torch.float64
        config.precision = "double" if precision == "single" else "single"

        equalizer = MFEqualizer(
            resource_grid=resource_grid,
            stream_management=stream_management,
            precision=precision,
            device=device,
        )

        batch_size = 4
        num_rx = 1
        num_rx_ant = 4
        num_ofdm_symbols = resource_grid.num_ofdm_symbols
        fft_size = resource_grid.fft_size
        num_tx = resource_grid.num_tx
        num_streams_per_tx = resource_grid.num_streams_per_tx
        num_effective_subcarriers = resource_grid.num_effective_subcarriers
        num_data_symbols = resource_grid.pilot_pattern.num_data_symbols

        y = complex_normal(
            (batch_size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size),
            precision=precision,
            device=device,
        )
        h_hat = complex_normal(
            (
                batch_size,
                num_rx,
                num_rx_ant,
                num_tx,
                num_streams_per_tx,
                num_ofdm_symbols,
                num_effective_subcarriers,
            ),
            precision=precision,
            device=device,
        )
        err_var = torch.ones(1, device=device) * 0.01
        no = torch.ones(1, device=device) * 0.1

        x_hat, no_eff = equalizer(y, h_hat, err_var, no)

        expected_shape = (batch_size, num_tx, num_streams_per_tx, num_data_symbols)
        assert x_hat.shape == expected_shape
        assert no_eff.shape == expected_shape
        assert x_hat.dtype == cdtype
        assert no_eff.dtype == dtype

    def test_positive_noise_variance(
        self, device, precision, resource_grid, stream_management
    ):
        """Test that MF equalizer produces non-negative effective noise variance."""
        cdtype = torch.complex64 if precision == "single" else torch.complex128

        equalizer = MFEqualizer(
            resource_grid=resource_grid,
            stream_management=stream_management,
            precision=precision,
            device=device,
        )

        batch_size = 4
        num_rx = 1
        num_rx_ant = 4
        num_ofdm_symbols = resource_grid.num_ofdm_symbols
        fft_size = resource_grid.fft_size
        num_tx = resource_grid.num_tx
        num_streams_per_tx = resource_grid.num_streams_per_tx
        num_effective_subcarriers = resource_grid.num_effective_subcarriers

        y = complex_normal(
            (batch_size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size),
            precision=precision,
            device=device,
        )
        h_hat = complex_normal(
            (
                batch_size,
                num_rx,
                num_rx_ant,
                num_tx,
                num_streams_per_tx,
                num_ofdm_symbols,
                num_effective_subcarriers,
            ),
            precision=precision,
            device=device,
        )
        err_var = torch.ones(1, device=device) * 0.01
        no = torch.ones(1, device=device) * 0.1

        _, no_eff = equalizer(y, h_hat, err_var, no)

        # Effective noise variance should be non-negative (real part)
        assert (no_eff.real >= 0).all()


class TestLMMSEPostEqualizationSINR:
    """Tests for LMMSEPostEqualizationSINR class."""

    def test_output_shape(self, device, precision, resource_grid, stream_management):
        """Test that LMMSEPostEqualizationSINR produces correct output shape."""
        cdtype = torch.complex64 if precision == "single" else torch.complex128

        sinr_computer = LMMSEPostEqualizationSINR(
            resource_grid=resource_grid,
            stream_management=stream_management,
            precision=precision,
            device=device,
        )

        batch_size = 4
        num_rx = 1
        num_rx_ant = 4
        num_ofdm_symbols = resource_grid.num_ofdm_symbols
        num_tx = resource_grid.num_tx
        num_streams_per_tx = resource_grid.num_streams_per_tx
        # For SINR, we use fft_size directly as effective subcarriers
        num_effective_subcarriers = resource_grid.fft_size

        h_eff = complex_normal(
            (
                batch_size,
                num_rx,
                num_rx_ant,
                num_tx,
                num_streams_per_tx,
                num_ofdm_symbols,
                num_effective_subcarriers,
            ),
            precision=precision,
            device=device,
        )
        no = torch.ones(1, device=device) * 0.1

        sinr = sinr_computer(h_eff, no)

        # Expected shape: [batch_size, num_ofdm_symbols, num_effective_subcarriers,
        #                  num_rx, num_streams_per_rx]
        num_streams_per_rx = stream_management.num_streams_per_rx
        expected_shape = (
            batch_size,
            num_ofdm_symbols,
            num_effective_subcarriers,
            num_rx,
            num_streams_per_rx,
        )
        assert sinr.shape == expected_shape

    def test_sinr_positive(self, device, precision, resource_grid, stream_management):
        """Test that SINR values are non-negative."""
        cdtype = torch.complex64 if precision == "single" else torch.complex128

        sinr_computer = LMMSEPostEqualizationSINR(
            resource_grid=resource_grid,
            stream_management=stream_management,
            precision=precision,
            device=device,
        )

        batch_size = 4
        num_rx = 1
        num_rx_ant = 4
        num_ofdm_symbols = resource_grid.num_ofdm_symbols
        num_tx = resource_grid.num_tx
        num_streams_per_tx = resource_grid.num_streams_per_tx
        num_effective_subcarriers = resource_grid.fft_size

        h_eff = complex_normal(
            (
                batch_size,
                num_rx,
                num_rx_ant,
                num_tx,
                num_streams_per_tx,
                num_ofdm_symbols,
                num_effective_subcarriers,
            ),
            precision=precision,
            device=device,
        )
        no = torch.ones(1, device=device) * 0.1

        sinr = sinr_computer(h_eff, no)

        # SINR should be non-negative
        assert (sinr >= 0).all()

    def test_siso_theoretical(self, device, precision):
        """Test against theoretical SINR for a SISO channel."""
        if precision == "single":
            pytest.skip("Requires double precision for accurate comparison")

        cdtype = torch.complex128

        # Create SISO configuration
        rg = ResourceGrid(
            num_ofdm_symbols=14,
            fft_size=64,
            subcarrier_spacing=15e3,
            num_tx=1,
            num_streams_per_tx=1,
        )
        rx_tx_association = np.ones([1, 1])
        sm = StreamManagement(rx_tx_association, num_streams_per_tx=1)

        sinr_computer = LMMSEPostEqualizationSINR(
            resource_grid=rg,
            stream_management=sm,
            precision=precision,
            device=device,
        )

        batch_size = 8
        num_rx = 1
        num_rx_ant = 1
        num_ofdm_symbols = rg.num_ofdm_symbols
        num_tx = rg.num_tx
        num_streams_per_tx = rg.num_streams_per_tx
        num_effective_subcarriers = rg.fft_size

        # Random channel
        h = complex_normal(
            (
                batch_size,
                num_rx,
                num_rx_ant,
                num_tx,
                num_streams_per_tx,
                num_ofdm_symbols,
                num_effective_subcarriers,
            ),
            precision=precision,
            device=device,
        )
        no = torch.ones(batch_size, device=device) * 0.1

        sinr = sinr_computer(h, no)
        sinr = sinr.squeeze()

        # For SISO, theoretical SINR = |h|^2 / no
        h_squeeze = h.squeeze()
        sinr_theo = h_squeeze.abs().square() / no.unsqueeze(-1).unsqueeze(-1)

        # Check relative error
        rel_err = (sinr - sinr_theo).abs() / sinr_theo.abs()
        assert rel_err.max() < 1e-5

    def test_against_alternative_implementation(self, device, precision):
        """Test LMMSEPostEqualizationSINR against alternative manual implementation.

        Uses RZFPrecodedChannel and RayleighBlockFading to create a realistic
        MIMO OFDM scenario and computes SINR using both the class and a manual
        alternative implementation to verify correctness.
        """
        if precision == "single":
            pytest.skip("Requires double precision for accurate comparison")

        cdtype = torch.complex128
        rdtype = torch.float64

        num_rx_per_tx = 2
        num_streams_per_rx = 2
        num_rx_ant = num_streams_per_rx
        num_tx = 2
        num_rx = num_rx_per_tx * num_tx
        num_streams_per_tx = num_rx_per_tx * num_streams_per_rx

        rx_tx_association = np.zeros((num_rx, num_tx), dtype=np.int32)
        for j in range(num_tx):
            rx_tx_association[j * num_rx_per_tx : (j + 1) * num_rx_per_tx, j] = 1

        sm = StreamManagement(rx_tx_association, num_streams_per_tx)
        rg = ResourceGrid(
            num_ofdm_symbols=14,
            fft_size=64,
            subcarrier_spacing=15e3,
            num_tx=num_tx,
            num_streams_per_tx=num_streams_per_tx,
            precision=precision,
            device=device,
        )

        channel = RayleighBlockFading(
            num_rx=num_rx,
            num_rx_ant=num_rx_ant,
            num_tx=num_tx,
            num_tx_ant=num_streams_per_tx * 2,
            precision=precision,
            device=device,
        )

        batch_size = 64
        torch.manual_seed(42)
        tx_power = torch.rand(
            batch_size, num_tx, num_streams_per_tx, rg.num_ofdm_symbols, rg.fft_size,
            dtype=rdtype, device=device
        )
        alpha = torch.rand(batch_size, num_tx, 1, 1, dtype=rdtype, device=device)
        cir = channel(batch_size=batch_size, num_time_steps=rg.num_ofdm_symbols)

        frequencies = subcarrier_frequencies(
            rg.fft_size, rg.subcarrier_spacing, precision=precision, device=device
        )
        h = cir_to_ofdm_channel(frequencies, *cir)

        precoded_channel = RZFPrecodedChannel(
            resource_grid=rg, stream_management=sm, precision=precision, device=device
        )
        h_eff = precoded_channel(h, tx_power=tx_power, alpha=alpha)

        sinr_computer = LMMSEPostEqualizationSINR(
            resource_grid=rg, stream_management=sm, precision=precision, device=device
        )
        no = torch.tensor(0.1, dtype=rdtype, device=device)
        sinr = sinr_computer(h_eff, no=no)

        # Alternative implementation
        sinr_alt = torch.zeros_like(sinr)
        num_effective_subcarriers = rg.num_effective_subcarriers
        for j in range(num_tx):
            # Get all rx indices for this tx
            rx_ind = np.where(rx_tx_association[:, j])[0]
            for rx_counter, i in enumerate(rx_ind):
                # Get desired channels from the transmitter
                # [batch_size, num_rx_ant, num_streams_per_rx, num_ofdm_symbols, num_effective_subcarriers]
                h_i_d = h_eff[:, i, :, j, rx_counter * num_streams_per_rx : (rx_counter + 1) * num_streams_per_rx]
                # [batch_size, num_ofdm_symbols, num_effective_subcarriers, num_rx_ant, num_streams_per_rx]
                h_i_d = h_i_d.permute(0, 3, 4, 1, 2)

                # Get interfering channels from the transmitter
                h_i_i = torch.cat([
                    h_eff[:, i, :, j, : rx_counter * num_streams_per_rx],
                    h_eff[:, i, :, j, (rx_counter + 1) * num_streams_per_rx :]
                ], dim=2)
                # [batch_size, num_ofdm_symbols, num_effective_subcarriers, num_rx_ant, num_streams_per_tx-num_streams_per_rx]
                h_i_i = h_i_i.permute(0, 3, 4, 1, 2)

                # Get all channels for this receiver
                # [batch_size, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols, num_effective_subcarriers]
                h_i = h_eff[:, i]

                # Take channels from all interfering transmitters
                # [batch_size, num_rx_ant, num_tx-1, num_streams_per_tx, num_ofdm_symbols, num_effective_subcarriers]
                h_i_ud = torch.cat([h_i[:, :, :j], h_i[:, :, j + 1 :]], dim=2)

                # [batch_size, num_ofdm_symbols, num_effective_subcarriers, num_rx_ant, num_tx-1, num_streams_per_tx]
                h_i_ud = h_i_ud.permute(0, 4, 5, 1, 2, 3)

                # [batch_size, num_ofdm_symbols, num_effective_subcarriers, num_rx_ant, (num_tx-1)*num_streams_per_tx]
                h_i_ud = h_i_ud.reshape(
                    batch_size, rg.num_ofdm_symbols, num_effective_subcarriers, num_rx_ant, -1
                )

                h_i_ti = torch.cat([h_i_ud, h_i_i], dim=-1)

                # Compute interference covariance matrix
                eye_matrix = torch.eye(
                    num_rx_ant, dtype=cdtype, device=device
                ).expand(*h_i_ti.shape[:-2], num_rx_ant, num_rx_ant)
                c_i = h_i_ti @ h_i_ti.mH + no * eye_matrix

                # Whiten channels
                l_inv = inv_cholesky(c_i)  # Compute whitening matrix
                h_i_d = l_inv @ h_i_d
                h_i_ti = l_inv @ h_i_ti

                # Compute equalization matrix
                eye_streams = torch.eye(
                    num_streams_per_rx, dtype=cdtype, device=device
                ).expand(*h_i_d.shape[:-2], num_streams_per_rx, num_streams_per_rx)
                hhs = h_i_d.mH @ h_i_d + eye_streams
                chol = torch.linalg.cholesky(hhs)
                g_i = torch.cholesky_solve(h_i_d.mH, chol)

                # Signal power
                p_i = g_i @ h_i_d
                p_i = torch.linalg.diagonal(p_i, dim1=-2, dim2=-1).abs().square()

                # Total power
                t_i = (g_i @ torch.cat([h_i_ti, h_i_d], dim=-1)).abs().square()
                t_i = t_i.sum(dim=-1)

                # Noise power
                noise_i = g_i.abs().square().sum(dim=-1)

                # SINR
                sinr_i = torch.where(
                    t_i - p_i + noise_i != 0,
                    p_i / (t_i - p_i + noise_i),
                    torch.zeros_like(p_i),
                )

                sinr_alt[..., i, :] = sinr_i

        assert torch.allclose(sinr, sinr_alt, atol=1e-6)

    def test_single_antenna_uplink(self, device, precision):
        """Test against theoretical SINR for a single antenna uplink channel.

        This test uses multiple UEs (TX) with single antenna each transmitting
        to a single BS (RX) with multiple antennas.
        """
        if precision == "single":
            pytest.skip("Requires double precision for accurate comparison")

        cdtype = torch.complex128
        rdtype = torch.float64

        num_rx = 1
        num_rx_ant = 16
        num_tx = 8
        num_streams_per_tx = 1

        rx_tx_association = np.ones((num_rx, num_tx), dtype=np.int32)

        sm = StreamManagement(rx_tx_association, num_streams_per_tx)
        rg = ResourceGrid(
            num_ofdm_symbols=14,
            fft_size=64,
            subcarrier_spacing=15e3,
            num_tx=num_tx,
            num_streams_per_tx=num_streams_per_tx,
            precision=precision,
            device=device,
        )

        channel = RayleighBlockFading(
            num_rx=num_rx,
            num_rx_ant=num_rx_ant,
            num_tx=num_tx,
            num_tx_ant=1,
            precision=precision,
            device=device,
        )

        batch_size = 16
        torch.manual_seed(123)
        tx_power = torch.rand(
            batch_size, num_tx, num_streams_per_tx, rg.num_ofdm_symbols,
            dtype=rdtype, device=device
        )
        cir = channel(batch_size=batch_size, num_time_steps=rg.num_ofdm_symbols)

        frequencies = subcarrier_frequencies(
            rg.fft_size, rg.subcarrier_spacing, precision=precision, device=device
        )
        h = cir_to_ofdm_channel(frequencies, *cir)

        precoded_channel = EyePrecodedChannel(
            resource_grid=rg, stream_management=sm, precision=precision, device=device
        )
        h_eff = precoded_channel(h, tx_power=tx_power)

        sinr_computer = LMMSEPostEqualizationSINR(
            resource_grid=rg, stream_management=sm, precision=precision, device=device
        )
        torch.manual_seed(456)
        no = torch.rand(batch_size, num_rx, num_rx_ant, dtype=rdtype, device=device)
        sinr = sinr_computer(h_eff, no=no, interference_whitening=True)
        sinr = sinr.squeeze()

        # Theoretical SINR computation
        # h shape: [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_ofdm_symbols, num_effective_subcarriers]
        # After squeeze (remove num_rx=1 and num_tx_ant=1):
        # [batch_size, num_rx_ant, num_tx, num_ofdm_symbols, num_effective_subcarriers]
        h_e = h.squeeze()  # Remove all size-1 dimensions

        # [batch_size, num_ofdm_symbols, num_effective_subcarriers, num_rx_ant, num_tx]
        h_e = h_e.permute(0, 3, 4, 1, 2)

        # Apply transmit power
        # tx_power: [batch_size, num_tx, num_streams_per_tx, num_ofdm_symbols]
        p = tx_power.squeeze(2)  # [batch_size, num_tx, num_ofdm_symbols]
        p = p.permute(0, 2, 1)  # [batch_size, num_ofdm_symbols, num_tx]
        p = expand_to_rank(p, h_e.dim(), axis=2)
        h_e = h_e * p.sqrt().to(dtype=h_e.dtype)

        # Whiten channels according to noise power
        # no: [batch_size, num_rx, num_rx_ant]
        n = no.squeeze(1)  # [batch_size, num_rx_ant]
        n = n.unsqueeze(-1)  # [batch_size, num_rx_ant, 1]
        n = expand_to_rank(n, h_e.dim(), axis=1)
        h_e = h_e / n.sqrt().to(dtype=h_e.dtype)

        f = lmmse_matrix(h_e, precision=precision)
        signal_power = (f @ h_e).diagonal(dim1=-2, dim2=-1).abs().square()
        total_power = (f @ h_e).abs().square().sum(dim=-1)
        noise_power = f.abs().square().sum(dim=-1)
        interference_power = total_power - signal_power
        sinr_theo = torch.where(
            interference_power + noise_power != 0,
            signal_power / (interference_power + noise_power),
            torch.zeros_like(signal_power),
        )

        rel_err = (sinr - sinr_theo).abs() / sinr_theo.abs().clamp(min=1e-10)
        assert rel_err.max() < 1e-5


class TestEqualizerCompilation:
    """Tests for torch.compile compatibility."""

    @pytest.mark.parametrize(
        "equalizer_class", [LMMSEEqualizer, ZFEqualizer, MFEqualizer]
    )
    def test_equalizer_compiles(
        self, device, precision, resource_grid, stream_management, equalizer_class
    ):
        """Test that OFDM equalizers can be compiled with torch.compile."""
        if device == "cpu":
            pytest.skip("Compilation tests may be slow on CPU")

        cdtype = torch.complex64 if precision == "single" else torch.complex128

        equalizer = equalizer_class(
            resource_grid=resource_grid,
            stream_management=stream_management,
            precision=precision,
            device=device,
        )

        batch_size = 2
        num_rx = 1
        num_rx_ant = 4
        num_ofdm_symbols = resource_grid.num_ofdm_symbols
        fft_size = resource_grid.fft_size
        num_tx = resource_grid.num_tx
        num_streams_per_tx = resource_grid.num_streams_per_tx
        num_effective_subcarriers = resource_grid.num_effective_subcarriers

        y = complex_normal(
            (batch_size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size),
            precision=precision,
            device=device,
        )
        h_hat = complex_normal(
            (
                batch_size,
                num_rx,
                num_rx_ant,
                num_tx,
                num_streams_per_tx,
                num_ofdm_symbols,
                num_effective_subcarriers,
            ),
            precision=precision,
            device=device,
        )
        err_var = torch.ones(1, device=device) * 0.01
        no = torch.ones(1, device=device) * 0.1

        # Compile the equalizer
        compiled_eq = torch.compile(equalizer)

        # Run compiled version
        x_hat_compiled, no_eff_compiled = compiled_eq(y, h_hat, err_var, no)

        # Run original version
        x_hat_orig, no_eff_orig = equalizer(y, h_hat, err_var, no)

        # Results should match
        assert torch.allclose(x_hat_compiled, x_hat_orig, atol=1e-4)
        assert torch.allclose(no_eff_compiled, no_eff_orig, atol=1e-4)

