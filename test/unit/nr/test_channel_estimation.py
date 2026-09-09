#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Tests for PUSCH channel estimation."""

import pytest
import numpy as np
import torch

from sionna.phy.nr import (
    PUSCHConfig, PUSCHTransmitter, PUSCHLSChannelEstimator, PUSCHLMMSEChannelEstimator
)
from sionna.phy.channel import RayleighBlockFading, OFDMChannel
from sionna.phy.utils import insert_dims


def check_channel_estimation(pusch_configs, add_awgn=False):
    """Check channel estimation accuracy."""
    pusch_transmitter = PUSCHTransmitter(pusch_configs)
    rayleigh = RayleighBlockFading(
        num_rx=1,
        num_rx_ant=16,
        num_tx=len(pusch_configs),
        num_tx_ant=pusch_configs[0].num_antenna_ports,
    )
    channel = OFDMChannel(
        rayleigh, pusch_transmitter._resource_grid, return_channel=True
    )

    channel_estimator = PUSCHLSChannelEstimator(
        pusch_transmitter._resource_grid,
        pusch_transmitter._dmrs_length,
        pusch_transmitter._dmrs_additional_position,
        pusch_transmitter._num_cdm_groups_without_data,
        interpolation_type="nn",
    )

    no = torch.tensor(0.01)
    x, b = pusch_transmitter(128)
    if not add_awgn:
        y, h = channel(x)
    else:
        y, h = channel(x, no.item())

    if pusch_configs[0].precoding == "codebook":
        # Compute precoded channel
        # h has shape:
        # [batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_ofdm_symbols, fft_size]

        # Reshape to put channel matrix dimensions last
        # [batch size, num_rx, num_tx, num_ofdm_symbols, fft_size, num_rx_ant, num_tx_ant]
        h = h.permute(0, 1, 3, 5, 6, 2, 4)

        # Multiply by precoding matrices to compute effective channels
        # [batch size, num_rx, num_tx, num_ofdm_symbols, fft_size, num_rx_ant, num_streams]
        w = pusch_transmitter._precoder._w
        w = insert_dims(w, 2, 1)
        h = torch.matmul(h, w)

        # [batch size, num_rx, num_rx_ant, num_tx, num_streams, num_ofdm_symbols, fft_size]
        h = h.permute(0, 1, 5, 2, 6, 3, 4)

    h_hat, err_var_hat = channel_estimator(y, no)

    # Compute empirical error variance
    err_var = torch.var(h - h_hat).item()

    if not add_awgn:
        return np.allclose(
            h.cpu().numpy(), h_hat.cpu().numpy(), atol=1e-6
        )
    else:
        return np.allclose(err_var, err_var_hat.cpu().numpy(), atol=1e-2)


class TestPUSCHLSChannelEstimator:
    """Tests for PUSCHLSChannelEstimator."""

    def test_01(self):
        """Test many configurations for a single transmitter without precoding."""
        pusch_config = PUSCHConfig()
        pusch_config.n_size_bwp = 4
        pusch_config.precoding = "non-codebook"

        for num_antenna_ports in [1, 2, 4]:
            pusch_config.num_antenna_ports = num_antenna_ports
            pusch_config.num_layers = num_antenna_ports

            for length in [1, 2]:
                pusch_config.dmrs.length = length
                max_additional_position = 3 if length == 1 else 1

                for additional_position in range(0, max_additional_position + 1):
                    pusch_config.dmrs.additional_position = additional_position

                    for config_type in [1, 2]:
                        pusch_config.dmrs.config_type = config_type
                        max_cdm_groups = 2 if config_type == 1 else 3
                        min_cdm_groups = 1 if num_antenna_ports < 4 else 2

                        for num_cdm_groups in range(min_cdm_groups, max_cdm_groups + 1):
                            pusch_config.dmrs.num_cdm_groups_without_data = num_cdm_groups
                            assert check_channel_estimation([pusch_config])
                            assert check_channel_estimation([pusch_config], add_awgn=True)

    def test_02(self):
        """Test many configurations for a single transmitter with precoding."""
        pusch_config = PUSCHConfig()
        pusch_config.n_size_bwp = 5
        pusch_config.precoding = "codebook"
        pusch_config.tpmi = 2

        for num_antenna_ports in [1, 2, 4]:
            pusch_config.num_antenna_ports = num_antenna_ports

            for num_layers in range(1, num_antenna_ports):
                pusch_config.num_layers = num_layers

                for length in [1, 2]:
                    pusch_config.dmrs.length = length
                    max_additional_position = 3 if length == 1 else 1

                    for additional_position in range(0, max_additional_position + 1):
                        pusch_config.dmrs.additional_position = additional_position

                        for config_type in [1, 2]:
                            pusch_config.dmrs.config_type = config_type
                            max_cdm_groups = 2 if config_type == 1 else 3
                            min_cdm_groups = 1 if num_antenna_ports < 4 else 2

                            for num_cdm_groups in range(min_cdm_groups, max_cdm_groups + 1):
                                pusch_config.dmrs.num_cdm_groups_without_data = num_cdm_groups
                                assert check_channel_estimation([pusch_config])
                                assert check_channel_estimation([pusch_config], add_awgn=True)

    def test_03(self):
        """Tests for multiple transmitters with precoding."""
        pusch_config = PUSCHConfig()
        pusch_config.n_size_bwp = 4
        pusch_config.num_antenna_ports = 4
        pusch_config.precoding = "codebook"
        pusch_config.tpmi = 2
        pusch_config.num_layers = 2
        pusch_config.dmrs.length = 2
        pusch_config.dmrs.config_type = 2
        pusch_config.dmrs.num_cdm_groups_without_data = 3
        pusch_config.dmrs.dmrs_port_set = [0, 1]
        pusch_config.dmrs.additional_position = 1

        pusch_config2 = pusch_config.clone()
        pusch_config2.dmrs.dmrs_port_set = [2, 3]
        pusch_config2.tpmi = 11

        pusch_config3 = pusch_config.clone()
        pusch_config3.dmrs.dmrs_port_set = [4, 5]
        pusch_config3.tpmi = 16

        pusch_configs = [pusch_config, pusch_config2, pusch_config3]
        assert check_channel_estimation(pusch_configs)
        assert check_channel_estimation(pusch_configs, add_awgn=True)


class TestPUSCHLMMSEChannelEstimator:
    """Tests for PUSCHLMMSEChannelEstimator."""

    def test_cdm_aware_ls_hook_and_full_grid_output(self):
        """Test PUSCH CDM processing and complete LMMSE estimation."""
        pusch_config = PUSCHConfig()
        pusch_config.n_size_bwp = 4
        pusch_config.precoding = "non-codebook"
        pusch_config.num_antenna_ports = 2
        pusch_config.num_layers = 2
        pusch_config.dmrs.length = 2
        pusch_config.dmrs.additional_position = 1
        pusch_config.dmrs.config_type = 1
        pusch_config.dmrs.num_cdm_groups_without_data = 2

        transmitter = PUSCHTransmitter(pusch_config)
        resource_grid = transmitter.resource_grid

        cov_mat_time = torch.eye(
            resource_grid.num_ofdm_symbols,
            dtype=torch.complex64,
        )
        cov_mat_freq = torch.eye(
            resource_grid.num_effective_subcarriers,
            dtype=torch.complex64,
        )

        lmmse_estimator = PUSCHLMMSEChannelEstimator(
            resource_grid,
            dmrs_length=pusch_config.dmrs.length,
            dmrs_additional_position=pusch_config.dmrs.additional_position,
            num_cdm_groups_without_data=(
                pusch_config.dmrs.num_cdm_groups_without_data
            ),
            cov_mat_time=cov_mat_time,
            cov_mat_freq=cov_mat_freq,
            order="f-t",
        )
        ls_estimator = PUSCHLSChannelEstimator(
            resource_grid,
            dmrs_length=pusch_config.dmrs.length,
            dmrs_additional_position=pusch_config.dmrs.additional_position,
            num_cdm_groups_without_data=(
                pusch_config.dmrs.num_cdm_groups_without_data
            ),
        )

        batch_size = 2
        num_rx = 1
        num_rx_ant = 3
        device = torch.device(lmmse_estimator.device)
        no = torch.tensor(0.1, device=device)

        # Verify that LMMSE estimation starts from the same CDM-aware pilot
        # estimates as the dedicated PUSCH LS estimator.
        num_pilots = resource_grid.pilot_pattern.pilots.shape[-1]
        y_pilots = torch.randn(
            batch_size,
            num_rx,
            num_rx_ant,
            resource_grid.num_tx,
            resource_grid.num_streams_per_tx,
            num_pilots,
            dtype=torch.complex64,
            device=device,
        )
        expected_h_ls, expected_err_var = (
            ls_estimator.estimate_at_pilot_locations(y_pilots, no)
        )
        actual_h_ls, actual_err_var = (
            lmmse_estimator._estimate_ls_at_pilot_locations(y_pilots, no)
        )

        torch.testing.assert_close(actual_h_ls, expected_h_ls)
        torch.testing.assert_close(actual_err_var, expected_err_var)

        # Verify the complete PUSCH LMMSE estimation path.
        y = torch.randn(
            batch_size,
            num_rx,
            num_rx_ant,
            resource_grid.num_ofdm_symbols,
            resource_grid.fft_size,
            dtype=torch.complex64,
            device=device,
        )
        h_hat, err_var = lmmse_estimator(y, no)

        expected_shape = (
            batch_size,
            num_rx,
            num_rx_ant,
            resource_grid.num_tx,
            resource_grid.num_streams_per_tx,
            resource_grid.num_ofdm_symbols,
            resource_grid.num_effective_subcarriers,
        )
        assert h_hat.shape == expected_shape
        assert err_var.shape == expected_shape
        assert torch.isfinite(h_hat).all()
        assert torch.isfinite(err_var).all()
        assert torch.all(err_var >= 0)


    @pytest.mark.parametrize(
        ("dmrs_length", "dmrs_additional_position"),
        [(1, 0), (2, 1)],
    )
    def test_noiseless_unit_channel(
        self,
        dmrs_length,
        dmrs_additional_position,
    ):
        """Recover a noiseless unit channel over the complete resource grid."""
        pusch_config = PUSCHConfig()
        pusch_config.n_size_bwp = 4
        pusch_config.precoding = "non-codebook"
        pusch_config.num_antenna_ports = 1
        pusch_config.num_layers = 1
        pusch_config.dmrs.length = dmrs_length
        pusch_config.dmrs.additional_position = dmrs_additional_position
        pusch_config.dmrs.config_type = 1
        pusch_config.dmrs.num_cdm_groups_without_data = 1

        transmitter = PUSCHTransmitter(pusch_config)
        resource_grid = transmitter.resource_grid
        device = torch.device(transmitter.device)

        # Rank-one covariance matrices model a channel that is constant
        # across all subcarriers and OFDM symbols.
        cov_mat_time = torch.ones(
            (
                resource_grid.num_ofdm_symbols,
                resource_grid.num_ofdm_symbols,
            ),
            dtype=torch.complex64,
            device=device,
        )
        cov_mat_freq = torch.ones(
            (
                resource_grid.num_effective_subcarriers,
                resource_grid.num_effective_subcarriers,
            ),
            dtype=torch.complex64,
            device=device,
        )

        estimator = PUSCHLMMSEChannelEstimator(
            resource_grid,
            dmrs_length=dmrs_length,
            dmrs_additional_position=dmrs_additional_position,
            num_cdm_groups_without_data=1,
            cov_mat_time=cov_mat_time,
            cov_mat_freq=cov_mat_freq,
            order="f-t",
        )

        # Passing the transmitted grid directly to the estimator corresponds
        # to a noiseless SISO channel with unit gain.
        y, _ = transmitter(2)
        no = torch.zeros((), dtype=torch.float32, device=device)

        h_hat, err_var = estimator(y, no)

        expected_shape = (
            2,
            1,
            1,
            resource_grid.num_tx,
            resource_grid.num_streams_per_tx,
            resource_grid.num_ofdm_symbols,
            resource_grid.num_effective_subcarriers,
        )
        assert h_hat.shape == expected_shape
        assert err_var.shape == expected_shape

        torch.testing.assert_close(
            h_hat,
            torch.ones_like(h_hat),
            rtol=1e-5,
            atol=1e-5,
        )
        torch.testing.assert_close(
            err_var,
            torch.zeros_like(err_var),
            rtol=1e-5,
            atol=1e-5,
        )