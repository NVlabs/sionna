#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Tests for sionna.phy.ofdm.channel_estimation module."""

import pytest
import numpy as np
import torch
from scipy.special import jv

from sionna.phy import PI, SPEED_OF_LIGHT, config
from sionna.phy.object import Object
from sionna.phy.ofdm import (
    ResourceGrid,
    ResourceGridMapper,
    BaseChannelEstimator,
    BasePilotChannelEstimator,
    LSChannelEstimator,
    BaseChannelInterpolator,
    NearestNeighborInterpolator,
    LinearInterpolator,
    LMMSEChannelEstimator,
    PilotPattern,
    KroneckerPilotPattern,
    tdl_freq_cov_mat,
    tdl_time_cov_mat,
)
from sionna.phy.channel import (
    ApplyOFDMChannel,
    subcarrier_frequencies,
    cir_to_ofdm_channel,
    exp_corr_mat,
    gen_single_sector_topology,
)
from sionna.phy.channel.tr38901 import PanelArray, UMi, TDL, models
from sionna.phy.mimo import StreamManagement
from sionna.phy.mapping import QAMSource
from sionna.phy.utils import complex_normal, ebnodb2no
from sionna.phy.ofdm.channel_estimation import _PilotGroupMixin


# ============================================================================
# Reference implementations for linear interpolation
# ============================================================================

def freq_int_reference(h, i, j):
    """Reference implementation: Linear interpolation along frequency axis.

    :param h: Channel matrix [num_ofdm_symbols, num_subcarriers]
    :param i: Row indices of nonzero pilots
    :param j: Column indices of nonzero pilots
    :return: Frequency-interpolated channel
    """
    h_int = np.zeros_like(h)
    h_int[i, j] = h[i, j]

    x_0 = np.zeros_like(h)
    x_1 = np.zeros_like(h)
    y_0 = np.zeros_like(h)
    y_1 = np.zeros_like(h)
    x = np.zeros_like(h)

    for a in range(h_int.shape[0]):
        x[a] = np.arange(0, h_int.shape[1])
        pilot_ind = np.where(h_int[a])[0]
        if len(pilot_ind) == 1:
            x_0[a] = x_1[a] = pilot_ind[0]
            y_0[a] = y_1[a] = h_int[a, pilot_ind[0]]
        elif len(pilot_ind) > 1:
            x0 = 0
            x1 = 1
            for b in range(h_int.shape[1]):
                x_0[a, b] = pilot_ind[x0]
                x_1[a, b] = pilot_ind[x1]
                y_0[a, b] = h_int[a, pilot_ind[x0]]
                y_1[a, b] = h_int[a, pilot_ind[x1]]
                if b == pilot_ind[x1] and x1 < len(pilot_ind) - 1:
                    x0 = x1
                    x1 += 1

    with np.errstate(divide='ignore', invalid='ignore'):
        slope = np.where(x_1 - x_0 != 0, (y_1 - y_0) / (x_1 - x_0), 0)
    h_int = (x - x_0) * slope + y_0
    return h_int


def time_int_reference(h, time_avg=False):
    """Reference implementation: Linear interpolation along time axis.

    :param h: Channel matrix [num_ofdm_symbols, num_subcarriers]
    :param time_avg: If True, average across OFDM symbols
    :return: Time-interpolated channel
    """
    x_0 = np.zeros_like(h)
    x_1 = np.zeros_like(h)
    y_0 = np.zeros_like(h)
    y_1 = np.zeros_like(h)
    x = np.repeat(
        np.expand_dims(np.arange(0, h.shape[0]), 1), [h.shape[1]], axis=1
    )

    pilot_ind = np.where(np.sum(np.abs(h), axis=-1))[0]

    if time_avg:
        hh = np.sum(h, axis=0) / len(pilot_ind)
        h[pilot_ind] = hh

    if len(pilot_ind) == 1:
        h_int = np.repeat(h[pilot_ind], [h.shape[0]], axis=0)
        return h_int
    elif len(pilot_ind) > 1:
        x0 = 0
        x1 = 1
        for a in range(h.shape[0]):
            x_0[a] = pilot_ind[x0]
            x_1[a] = pilot_ind[x1]
            y_0[a] = h[pilot_ind[x0]]
            y_1[a] = h[pilot_ind[x1]]
            if a == pilot_ind[x1] and x1 < len(pilot_ind) - 1:
                x0 = x1
                x1 += 1

    with np.errstate(divide='ignore', invalid='ignore'):
        slope = np.where(x_1 - x_0 != 0, (y_1 - y_0) / (x_1 - x_0), 0)
    h_int = (x - x_0) * slope + y_0
    return h_int


def linear_int_reference(h, i, j, time_avg=False):
    """Reference implementation: Full 2D linear interpolation.

    :param h: Channel matrix [num_ofdm_symbols, num_subcarriers]
    :param i: Row indices of nonzero pilots
    :param j: Column indices of nonzero pilots
    :param time_avg: If True, average across OFDM symbols
    :return: Interpolated channel
    """
    h_int = freq_int_reference(h, i, j)
    return time_int_reference(h_int, time_avg)


def check_linear_interpolation(pilot_pattern, time_avg=False):
    """Simulate channel estimation with linear interpolation for a 3GPP UMi channel model.

    :param pilot_pattern: PilotPattern instance
    :param time_avg: Whether to use time averaging
    :return: True if interpolation matches reference
    """
    scenario = "umi"
    carrier_frequency = 3.5e9
    direction = "uplink"
    num_ut = pilot_pattern.num_tx
    num_streams_per_tx = pilot_pattern.num_streams_per_tx
    num_ofdm_symbols = pilot_pattern.num_ofdm_symbols
    fft_size = pilot_pattern.num_effective_subcarriers
    batch_size = 1

    ut_array = PanelArray(
        num_rows_per_panel=1,
        num_cols_per_panel=1,
        polarization="single",
        polarization_type="V",
        antenna_pattern="omni",
        carrier_frequency=carrier_frequency,
    )

    bs_array = PanelArray(
        num_rows_per_panel=1,
        num_cols_per_panel=4,
        polarization="dual",
        polarization_type="VH",
        antenna_pattern="38.901",
        carrier_frequency=carrier_frequency,
    )

    channel_model = UMi(
        carrier_frequency=carrier_frequency,
        o2i_model="low",
        ut_array=ut_array,
        bs_array=bs_array,
        direction=direction,
        enable_pathloss=False,
        enable_shadow_fading=False,
    )

    topology = gen_single_sector_topology(
        batch_size, num_ut, scenario, min_ut_velocity=0, max_ut_velocity=30
    )
    channel_model.set_topology(*topology)

    rx_tx_association = np.zeros([1, num_ut])
    rx_tx_association[0, :] = 1
    sm = StreamManagement(rx_tx_association, num_streams_per_tx)

    rg = ResourceGrid(
        num_ofdm_symbols=num_ofdm_symbols,
        fft_size=fft_size,
        subcarrier_spacing=30e3,
        num_tx=num_ut,
        num_streams_per_tx=num_streams_per_tx,
        cyclic_prefix_length=0,
        pilot_pattern=pilot_pattern,
    )

    frequencies = subcarrier_frequencies(rg.fft_size, rg.subcarrier_spacing)
    channel_freq = ApplyOFDMChannel()
    rg_mapper = ResourceGridMapper(rg)

    if time_avg:
        ls_est = LSChannelEstimator(rg, interpolation_type="lin_time_avg")
    else:
        ls_est = LSChannelEstimator(rg, interpolation_type="lin")

    # Generate channel
    x = torch.zeros(
        [batch_size, num_ut, rg.num_streams_per_tx, rg.num_data_symbols],
        dtype=torch.complex64,
    )
    x_rg = rg_mapper(x)
    a, tau = channel_model(
        num_time_samples=rg.num_ofdm_symbols,
        sampling_frequency=1 / rg.ofdm_symbol_duration,
    )
    h_freq = cir_to_ofdm_channel(frequencies, a, tau, normalize=True)
    y = channel_freq(x_rg, h_freq)  # noiseless channel
    h_hat_lin, _ = ls_est(y, torch.tensor(0.0))

    # Verify against reference for each TX
    for tx in range(0, num_ut):
        # Get non-zero pilot indices
        i, j = np.where(np.abs(x_rg[0, tx, 0].cpu().numpy()))
        h = h_freq[0, 0, 0, tx, 0].cpu().numpy()
        h_hat_lin_numpy = linear_int_reference(h, i, j, time_avg)
        if not np.allclose(h_hat_lin_numpy, h_hat_lin[0, 0, 0, tx, 0].cpu().numpy()):
            return False
    return True


# ============================================================================
# Test fixtures
# ============================================================================

@pytest.fixture
def resource_grid(device, precision):
    """Create a simple resource grid for testing."""
    return ResourceGrid(
        num_ofdm_symbols=14,
        fft_size=64,
        subcarrier_spacing=30e3,
        num_tx=2,
        num_streams_per_tx=2,
        pilot_pattern="kronecker",
        pilot_ofdm_symbol_indices=[2, 11],
        precision=precision,
        device=device,
    )

# ============================================================================
# BaseChannelEstimator shape test
# ============================================================================

class _FullGridChannelEstimator(BaseChannelEstimator):
    def call(self, y, no):
        err_var = torch.ones(
            (*y.shape[:3], *self._pilot_pattern.mask.shape),
            device=self.device, dtype=y.real.dtype
        )
        h_hat = torch.zeros(
            (*y.shape[:3], *self._pilot_pattern.mask.shape),
            device=self.device, dtype=y.dtype
        )
        return h_hat, err_var

class TestBaseChannelEstimator:

    def test_output_shape(self, device, precision, resource_grid):
        estimator = _FullGridChannelEstimator(
            resource_grid=resource_grid,
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
        no = torch.ones(1, device=device) * 0.1

        h_hat, err_var = estimator(y, no)

        expected_shape = (
            batch_size,
            num_rx,
            num_rx_ant,
            num_tx,
            num_streams_per_tx,
            num_ofdm_symbols,
            num_effective_subcarriers,
        )
        assert h_hat.shape == expected_shape
        assert err_var.shape == expected_shape

# ============================================================================
# BasePilotChannelEstimator shape tests
# ============================================================================

class _RecordingPilotChannelEstimator(BasePilotChannelEstimator):
    def estimate_at_pilot_locations(self, y_pilots, no):
        self._y_pilots = y_pilots.detach().clone()
        self._no = no.detach().clone()

        h_hat = torch.zeros_like(y_pilots)
        err_var = torch.zeros_like(y_pilots, dtype=no.dtype)
        return h_hat, err_var

class _RecordingInterpolator(BaseChannelInterpolator):

    def __init__(self, num_ofdm_symbols, num_effective_subcarriers):
        self._was_called = False
        self._num_ofdm_symbols = num_ofdm_symbols
        self._num_effective_subcarriers = num_effective_subcarriers

    def __call__(self, h_hat, err_var):
        self._was_called = True

        h_est = torch.zeros(
            (*h_hat.shape[:-1], self._num_ofdm_symbols, self._num_effective_subcarriers),
            device=h_hat.device, dtype=h_hat.dtype
        )
        err_var_est = torch.zeros_like(h_est, device=err_var.device, dtype=err_var.dtype)

        return h_est, err_var_est

class TestBasePilotChannelEstimator:
    def test_output_shape(self, device, precision, resource_grid):
        interpolator = _RecordingInterpolator(
            resource_grid.num_ofdm_symbols,
            resource_grid.num_effective_subcarriers
        )
        estimator = _RecordingPilotChannelEstimator(
            resource_grid=resource_grid,
            interpolator=interpolator,
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
        no = torch.ones(1, device=device) * 0.1

        h_hat, err_var = estimator(y, no)

        expected_shape = (
            batch_size,
            num_rx,
            num_rx_ant,
            num_tx,
            num_streams_per_tx,
            num_ofdm_symbols,
            num_effective_subcarriers,
        )
        assert h_hat.shape == expected_shape
        assert err_var.shape == expected_shape
        assert estimator._y_pilots.shape == (batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, resource_grid.pilot_pattern.num_pilot_symbols)
        assert interpolator._was_called


# ============================================================================
# LSChannelEstimator shape tests
# ============================================================================

class TestLSChannelEstimator:
    """Tests for LSChannelEstimator class."""

    @pytest.mark.parametrize("interpolation_type", ["nn", "lin", "lin_time_avg"])
    def test_output_shape(self, device, precision, resource_grid, interpolation_type):
        """Test that LSChannelEstimator produces correct output shape."""
        estimator = LSChannelEstimator(
            resource_grid=resource_grid,
            interpolation_type=interpolation_type,
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
        no = torch.ones(1, device=device) * 0.1

        h_hat, err_var = estimator(y, no)

        expected_shape = (
            batch_size,
            num_rx,
            num_rx_ant,
            num_tx,
            num_streams_per_tx,
            num_ofdm_symbols,
            num_effective_subcarriers,
        )
        assert h_hat.shape == expected_shape
        assert err_var.shape == expected_shape


    def test_error_variance_positive(self, device, precision, resource_grid):
        """Test that error variance is non-negative."""
        estimator = LSChannelEstimator(
            resource_grid=resource_grid,
            interpolation_type="lin",
            precision=precision,
            device=device,
        )

        batch_size = 4
        num_rx = 1
        num_rx_ant = 4
        num_ofdm_symbols = resource_grid.num_ofdm_symbols
        fft_size = resource_grid.fft_size

        y = complex_normal(
            (batch_size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size),
            precision=precision,
            device=device,
        )
        no = torch.ones(1, device=device) * 0.1

        _, err_var = estimator(y, no)

        # Error variance should be non-negative
        assert (err_var >= 0).all()


# ============================================================================
# Linear interpolation accuracy tests (match TF version)
# ============================================================================

class TestLinearInterpolatorAccuracy:
    """Tests for linear interpolation accuracy against reference implementation."""

    def test_sparse_pilot_pattern(self):
        """One UT has two pilots, three others have just one."""
        num_ut = 4
        num_streams_per_tx = 1
        num_ofdm_symbols = 14
        fft_size = 64
        mask = np.zeros([num_ut, num_streams_per_tx, num_ofdm_symbols, fft_size], bool)
        mask[..., [2, 3, 10, 11], :] = True
        num_pilots = np.sum(mask[0, 0])
        pilots = np.zeros([num_ut, num_streams_per_tx, num_pilots])
        pilots[0, 0, 10] = 1
        pilots[0, 0, 234] = 1
        pilots[1, 0, 20] = 1
        pilots[2, 0, 70] = 1
        pilots[3, 0, 120] = 1
        pilot_pattern = PilotPattern(mask, pilots)
        assert check_linear_interpolation(pilot_pattern)

    def test_kronecker_pilot_patterns_01(self):
        """Standard Kronecker pilot pattern."""
        num_ut = 4
        num_streams_per_tx = 1
        num_ofdm_symbols = 14
        fft_size = 64
        pilot_ofdm_symbol_indices = [2, 11]
        rg = ResourceGrid(
            num_ofdm_symbols=num_ofdm_symbols,
            fft_size=fft_size,
            subcarrier_spacing=30e3,
            num_tx=num_ut,
            num_streams_per_tx=num_streams_per_tx,
            cyclic_prefix_length=0,
            pilot_pattern="kronecker",
            pilot_ofdm_symbol_indices=pilot_ofdm_symbol_indices,
        )
        assert check_linear_interpolation(rg.pilot_pattern)

    def test_kronecker_pilot_patterns_02(self):
        """Only a single pilot symbol."""
        num_ut = 4
        num_streams_per_tx = 1
        num_ofdm_symbols = 14
        fft_size = 64
        pilot_ofdm_symbol_indices = [2]
        rg = ResourceGrid(
            num_ofdm_symbols=num_ofdm_symbols,
            fft_size=fft_size,
            subcarrier_spacing=30e3,
            num_tx=num_ut,
            num_streams_per_tx=num_streams_per_tx,
            cyclic_prefix_length=0,
            pilot_pattern="kronecker",
            pilot_ofdm_symbol_indices=pilot_ofdm_symbol_indices,
        )
        assert check_linear_interpolation(rg.pilot_pattern)

    def test_kronecker_pilot_patterns_03(self):
        """Only one pilot per UT."""
        num_ut = 16
        num_streams_per_tx = 1
        num_ofdm_symbols = 14
        fft_size = 16
        pilot_ofdm_symbol_indices = [2]
        rg = ResourceGrid(
            num_ofdm_symbols=num_ofdm_symbols,
            fft_size=fft_size,
            subcarrier_spacing=30e3,
            num_tx=num_ut,
            num_streams_per_tx=num_streams_per_tx,
            cyclic_prefix_length=0,
            pilot_pattern="kronecker",
            pilot_ofdm_symbol_indices=pilot_ofdm_symbol_indices,
        )
        assert check_linear_interpolation(rg.pilot_pattern)

    def test_kronecker_pilot_patterns_04(self):
        """Multi UT, multi stream."""
        num_ut = 4
        num_streams_per_tx = 2
        num_ofdm_symbols = 14
        fft_size = 64
        pilot_ofdm_symbol_indices = [2, 5, 8]
        rg = ResourceGrid(
            num_ofdm_symbols=num_ofdm_symbols,
            fft_size=fft_size,
            subcarrier_spacing=30e3,
            num_tx=num_ut,
            num_streams_per_tx=num_streams_per_tx,
            cyclic_prefix_length=0,
            pilot_pattern="kronecker",
            pilot_ofdm_symbol_indices=pilot_ofdm_symbol_indices,
        )
        assert check_linear_interpolation(rg.pilot_pattern)

    def test_kronecker_pilot_patterns_05(self):
        """Single UT, only pilots."""
        num_ut = 1
        num_streams_per_tx = 1
        num_ofdm_symbols = 5
        fft_size = 64
        pilot_ofdm_symbol_indices = list(range(num_ofdm_symbols))
        rg = ResourceGrid(
            num_ofdm_symbols=num_ofdm_symbols,
            fft_size=fft_size,
            subcarrier_spacing=30e3,
            num_tx=num_ut,
            num_streams_per_tx=num_streams_per_tx,
            cyclic_prefix_length=0,
            pilot_pattern="kronecker",
            pilot_ofdm_symbol_indices=pilot_ofdm_symbol_indices,
        )
        assert check_linear_interpolation(rg.pilot_pattern)

    def test_kronecker_pilot_patterns_06(self):
        """Multiple pilot symbols."""
        num_ut = 4
        num_streams_per_tx = 1
        num_ofdm_symbols = 14
        fft_size = 64
        pilot_ofdm_symbol_indices = [2, 3, 8, 11]
        rg = ResourceGrid(
            num_ofdm_symbols=num_ofdm_symbols,
            fft_size=fft_size,
            subcarrier_spacing=30e3,
            num_tx=num_ut,
            num_streams_per_tx=num_streams_per_tx,
            cyclic_prefix_length=0,
            pilot_pattern="kronecker",
            pilot_ofdm_symbol_indices=pilot_ofdm_symbol_indices,
        )
        assert check_linear_interpolation(rg.pilot_pattern)

    def test_kronecker_pilot_patterns_with_time_averaging(self):
        """Kronecker with time averaging."""
        num_ut = 4
        num_streams_per_tx = 1
        num_ofdm_symbols = 14
        fft_size = 64
        pilot_ofdm_symbol_indices = [2, 11]
        rg = ResourceGrid(
            num_ofdm_symbols=num_ofdm_symbols,
            fft_size=fft_size,
            subcarrier_spacing=30e3,
            num_tx=num_ut,
            num_streams_per_tx=num_streams_per_tx,
            cyclic_prefix_length=0,
            pilot_pattern="kronecker",
            pilot_ofdm_symbol_indices=pilot_ofdm_symbol_indices,
        )
        assert check_linear_interpolation(rg.pilot_pattern, time_avg=True)


# ============================================================================
# Interpolator shape tests
# ============================================================================

class TestNearestNeighborInterpolator:
    """Tests for NearestNeighborInterpolator class."""

    def test_output_shape(self, device, precision, resource_grid):
        """Test that NearestNeighborInterpolator produces correct output shape."""
        interpolator = NearestNeighborInterpolator(resource_grid.pilot_pattern)

        batch_size = 4
        num_rx = 1
        num_rx_ant = 4
        num_tx = resource_grid.num_tx
        num_streams_per_tx = resource_grid.num_streams_per_tx
        num_pilot_symbols = resource_grid.pilot_pattern.num_pilot_symbols
        num_ofdm_symbols = resource_grid.num_ofdm_symbols
        num_effective_subcarriers = resource_grid.num_effective_subcarriers

        h_hat_pilots = complex_normal(
            (batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_pilot_symbols),
            precision=precision,
            device=device,
        )
        err_var_pilots = torch.ones_like(h_hat_pilots.real) * 0.1

        h_hat, err_var = interpolator(h_hat_pilots, err_var_pilots)

        expected_shape = (
            batch_size,
            num_rx,
            num_rx_ant,
            num_tx,
            num_streams_per_tx,
            num_ofdm_symbols,
            num_effective_subcarriers,
        )
        assert h_hat.shape == expected_shape
        assert err_var.shape == expected_shape

    def test_time_averaging_uses_pilot_bearing_symbols(self, device, precision):
        """Time averaging equally weights frequency-interpolated pilot symbols."""
        mask = torch.zeros(
            1,
            1,
            7,
            5,
            dtype=torch.bool,
            device=device,
        )
        mask[0, 0, 0, [0, 4]] = True
        mask[0, 0, 2, [1, 3]] = True
        pilots = torch.ones(
            1,
            1,
            4,
            dtype=torch.complex64 if precision == "single" else torch.complex128,
            device=device,
        )
        pilot_pattern = PilotPattern(
            mask,
            pilots,
            precision=precision,
            device=device,
        )
        interpolator = NearestNeighborInterpolator(
            pilot_pattern,
            time_avg=True,
        )

        channel_at_pilots = torch.tensor(
            [1.0, 5.0, 10.0, 14.0],
            dtype=pilots.dtype,
            device=device,
        ).reshape(1, 1, 1, 1, 1, 4)
        variance_at_pilots = torch.tensor(
            [2.0, 2.0, 6.0, 6.0],
            dtype=pilots.real.dtype,
            device=device,
        ).reshape(1, 1, 1, 1, 1, 4)

        channel_estimate, error_variance = interpolator(
            channel_at_pilots,
            variance_at_pilots,
        )

        expected_frequency_estimate = torch.tensor(
            [5.5, 5.5, 5.5, 9.5, 9.5],
            dtype=pilots.dtype,
            device=device,
        )
        expected_channel = expected_frequency_estimate.reshape(
            1, 1, 1, 1, 1, 1, 5
        ).expand(1, 1, 1, 1, 1, 7, 5)
        expected_variance = torch.full(
            (1, 1, 1, 1, 1, 7, 5),
            4.0,
            dtype=pilots.real.dtype,
            device=device,
        )

        torch.testing.assert_close(channel_estimate, expected_channel)
        torch.testing.assert_close(error_variance, expected_variance)


class TestLinearInterpolator:
    """Tests for LinearInterpolator class."""

    def test_output_shape(self, device, precision, resource_grid):
        """Test that LinearInterpolator produces correct output shape."""
        interpolator = LinearInterpolator(resource_grid.pilot_pattern)

        batch_size = 4
        num_rx = 1
        num_rx_ant = 4
        num_tx = resource_grid.num_tx
        num_streams_per_tx = resource_grid.num_streams_per_tx
        num_pilot_symbols = resource_grid.pilot_pattern.num_pilot_symbols
        num_ofdm_symbols = resource_grid.num_ofdm_symbols
        num_effective_subcarriers = resource_grid.num_effective_subcarriers

        h_hat_pilots = complex_normal(
            (batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_pilot_symbols),
            precision=precision,
            device=device,
        )
        err_var_pilots = torch.ones_like(h_hat_pilots.real) * 0.1

        h_hat, err_var = interpolator(h_hat_pilots, err_var_pilots)

        expected_shape = (
            batch_size,
            num_rx,
            num_rx_ant,
            num_tx,
            num_streams_per_tx,
            num_ofdm_symbols,
            num_effective_subcarriers,
        )
        assert h_hat.shape == expected_shape
        assert err_var.shape == expected_shape

    @pytest.mark.parametrize("time_avg", [False, True])
    def test_time_averaging(self, device, precision, resource_grid, time_avg):
        """Test that time averaging option works."""
        interpolator = LinearInterpolator(resource_grid.pilot_pattern, time_avg=time_avg)

        batch_size = 2
        num_rx = 1
        num_rx_ant = 2
        num_tx = resource_grid.num_tx
        num_streams_per_tx = resource_grid.num_streams_per_tx
        num_pilot_symbols = resource_grid.pilot_pattern.num_pilot_symbols

        h_hat_pilots = complex_normal(
            (batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_pilot_symbols),
            precision=precision,
            device=device,
        )
        err_var_pilots = torch.ones_like(h_hat_pilots.real) * 0.1

        h_hat, err_var = interpolator(h_hat_pilots, err_var_pilots)

        # Should produce valid output
        assert not torch.isnan(h_hat).any()
        assert not torch.isnan(err_var).any()


# ============================================================================
# TDL covariance matrix tests
# ============================================================================

class TestTDLCovarianceMatrices:
    """Tests for TDL covariance matrix functions."""

    def test_defaults_follow_config(self, precision, device):
        """Default covariance dtype and device follow the global configuration."""
        config.precision = precision
        expected_dtype = (
            torch.complex64 if precision == "single" else torch.complex128
        )

        freq_cov = tdl_freq_cov_mat("A", 30e3, 8, 100e-9)
        time_cov = tdl_time_cov_mat("A", 3.0, 3.5e9, 35.7e-6, 8)

        for cov_mat in (freq_cov, time_cov):
            assert cov_mat.dtype == expected_dtype
            assert cov_mat.device == torch.device(device)

    def test_parameter_resolver_is_used(self, monkeypatch):
        """Test both covariance helpers use the versioned TDL resolver."""
        resolved_files = []
        parameter_file = models.parameter_file

        def tracked_parameter_file(filename, spec_version="19.2"):
            resolved_files.append((filename, spec_version))
            return parameter_file(filename, spec_version)

        monkeypatch.setattr(models, "parameter_file", tracked_parameter_file)

        tdl_freq_cov_mat("A", 30e3, 8, 100e-9)
        tdl_time_cov_mat("A", 3.0, 3.5e9, 35.7e-6, 8)

        assert resolved_files == [
            ("TDL-A.json", "19.2"),
            ("TDL-A.json", "19.2"),
        ]

    @pytest.mark.parametrize("model", ["A", "B", "C", "D", "E"])
    def test_tdl_freq_cov_mat_shape(self, precision, model):
        """Test that tdl_freq_cov_mat produces correct shape."""
        fft_size = 64
        subcarrier_spacing = 30e3
        delay_spread = 100e-9

        cov_mat = tdl_freq_cov_mat(model, subcarrier_spacing, fft_size, delay_spread, precision)

        assert cov_mat.shape == (fft_size, fft_size)

    @pytest.mark.parametrize("model", ["A", "B", "C", "D", "E"])
    def test_tdl_freq_cov_mat_hermitian(self, precision, model):
        """Test that frequency covariance matrix is Hermitian."""
        fft_size = 32
        subcarrier_spacing = 30e3
        delay_spread = 100e-9
        atol = 1e-5 if precision == "single" else 1e-10

        cov_mat = tdl_freq_cov_mat(model, subcarrier_spacing, fft_size, delay_spread, precision)

        # Hermitian: R = R^H
        assert torch.allclose(cov_mat, cov_mat.mH, atol=atol)

    @pytest.mark.parametrize("model", ["A", "B", "C", "D", "E"])
    def test_tdl_time_cov_mat_shape(self, precision, model):
        """Test that tdl_time_cov_mat produces correct shape."""
        num_ofdm_symbols = 14
        speed = 10.0
        carrier_frequency = 3.5e9
        ofdm_symbol_duration = 35.7e-6

        cov_mat = tdl_time_cov_mat(
            model, speed, carrier_frequency, ofdm_symbol_duration, num_ofdm_symbols, precision=precision
        )

        assert cov_mat.shape == (num_ofdm_symbols, num_ofdm_symbols)

    @pytest.mark.parametrize("model", ["A", "B", "C", "D", "E"])
    def test_tdl_time_cov_mat_hermitian(self, precision, model):
        """Test that time covariance matrix is Hermitian."""
        num_ofdm_symbols = 14
        speed = 10.0
        carrier_frequency = 3.5e9
        ofdm_symbol_duration = 35.7e-6
        atol = 1e-5 if precision == "single" else 1e-10

        cov_mat = tdl_time_cov_mat(
            model, speed, carrier_frequency, ofdm_symbol_duration, num_ofdm_symbols, precision=precision
        )

        # Hermitian: R = R^H
        assert torch.allclose(cov_mat, cov_mat.mH, atol=atol)

    def test_tdl_freq_cov_mat_diagonal_unity(self, precision):
        """Test that diagonal elements of frequency covariance matrix are close to 1."""
        fft_size = 64
        subcarrier_spacing = 30e3
        delay_spread = 100e-9
        atol = 1e-5 if precision == "single" else 1e-10

        cov_mat = tdl_freq_cov_mat("A", subcarrier_spacing, fft_size, delay_spread, precision)

        # Diagonal should be 1 (normalized power)
        diag = torch.diag(cov_mat)
        assert torch.allclose(diag.real, torch.ones_like(diag.real), atol=atol)
        assert torch.allclose(diag.imag, torch.zeros_like(diag.imag), atol=atol)

    def test_tdl_time_cov_mat_diagonal_unity(self, precision):
        """Test that diagonal elements of time covariance matrix are close to 1."""
        num_ofdm_symbols = 14
        speed = 10.0
        carrier_frequency = 3.5e9
        ofdm_symbol_duration = 35.7e-6
        atol = 1e-5 if precision == "single" else 1e-10

        cov_mat = tdl_time_cov_mat(
            "A", speed, carrier_frequency, ofdm_symbol_duration, num_ofdm_symbols, precision=precision
        )

        # Diagonal should be 1 (normalized power)
        diag = torch.diag(cov_mat)
        assert torch.allclose(diag.real, torch.ones_like(diag.real), atol=atol)

    @pytest.mark.parametrize("model", ["D", "E"])
    def test_tdl_time_cov_mat_default_los_doppler_peak(self, precision, model):
        """Test LoS covariance uses the default TR 38.901 0.7 Doppler peak."""
        speed = 10.0
        carrier_frequency = 3.5e9
        ofdm_symbol_duration = 1e-3
        cov_mat = tdl_time_cov_mat(
            model,
            speed,
            carrier_frequency,
            ofdm_symbol_duration,
            2,
            precision=precision,
        )

        params = models.load_json(models.parameter_file(f"TDL-{model}.json"))
        powers = np.power(10.0, np.asarray(params["powers"]) / 10.0)
        powers /= powers.sum()
        doppler = 2.0 * PI * speed / SPEED_OF_LIGHT * carrier_frequency
        argument = doppler * ofdm_symbol_duration
        expected = (
            powers[1:].sum() * jv(0.0, argument)
            + powers[0] * np.exp(1j * argument * 0.7)
        )
        expected = torch.tensor(
            expected, dtype=cov_mat.dtype, device=cov_mat.device
        )
        atol = 1e-5 if precision == "single" else 1e-12

        torch.testing.assert_close(cov_mat[1, 0], expected, atol=atol, rtol=atol)


# ============================================================================
# Compilation tests
# ============================================================================

class TestChannelEstimatorCompilation:
    """Tests for torch.compile compatibility."""

    @pytest.mark.parametrize("interpolation_type", ["nn", "lin", "lin_time_avg"])
    def test_ls_estimator_compiles(self, device, precision, resource_grid, interpolation_type):
        """Test that LSChannelEstimator can be compiled with torch.compile."""
        estimator = LSChannelEstimator(
            resource_grid=resource_grid,
            interpolation_type=interpolation_type,
            precision=precision,
            device=device,
        )

        batch_size = 2
        num_rx = 1
        num_rx_ant = 4
        num_ofdm_symbols = resource_grid.num_ofdm_symbols
        fft_size = resource_grid.fft_size

        y = complex_normal(
            (batch_size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size),
            precision=precision,
            device=device,
        )
        no = torch.ones(1, device=device) * 0.1

        # Compile the estimator
        compiled_est = torch.compile(estimator)

        # Run compiled version
        h_hat_compiled, err_var_compiled = compiled_est(y, no)

        # Run original version
        h_hat_orig, err_var_orig = estimator(y, no)

        # Results should match
        assert torch.allclose(h_hat_compiled, h_hat_orig, atol=1e-4)
        assert torch.allclose(err_var_compiled, err_var_orig, atol=1e-4)

    def test_lmmse_estimator_compiles(self, device, precision):
        """Test that LMMSEChannelEstimator can be compiled."""
        rg = ResourceGrid(
            num_ofdm_symbols=14,
            fft_size=32,
            subcarrier_spacing=30e3,
            num_tx=1,
            num_streams_per_tx=1,
            pilot_pattern="kronecker",
            pilot_ofdm_symbol_indices=[2, 11],
            precision=precision,
            device=device,
        )

        cov_freq = tdl_freq_cov_mat("A", 30e3, 32, 100e-9, precision)
        cov_time = tdl_time_cov_mat("A", 10.0, 3.5e9, 35.7e-6, 14, precision=precision)

        estimator = LMMSEChannelEstimator(
            resource_grid=rg,
            cov_mat_time=cov_time,
            cov_mat_freq=cov_freq,
            order="f-t",
            precision=precision,
            device=device,
        )

        batch_size = 2
        num_rx = 1
        num_rx_ant = 4

        y = complex_normal(
            (batch_size, num_rx, num_rx_ant, rg.num_ofdm_symbols, rg.fft_size),
            precision=precision,
            device=device,
        )
        no = torch.ones(1, device=device) * 0.1

        # Compile the estimator
        compiled_est = torch.compile(estimator)

        # Run compiled version
        h_hat_compiled, err_var_compiled = compiled_est(y, no)

        # Run original version
        h_hat_orig, err_var_orig = estimator(y, no)

        # Results should match
        assert torch.allclose(h_hat_compiled, h_hat_orig, atol=1e-4)
        assert torch.allclose(err_var_compiled, err_var_orig, atol=1e-4)


# ============================================================================
# Multiple pilot pattern tests
# ============================================================================

class TestMultiplePilotPatterns:
    """Tests for various pilot pattern configurations."""

    def test_kronecker_single_pilot_symbol(self, device, precision):
        """Test with only a single pilot OFDM symbol."""
        rg = ResourceGrid(
            num_ofdm_symbols=14,
            fft_size=64,
            subcarrier_spacing=30e3,
            num_tx=4,
            num_streams_per_tx=1,
            pilot_pattern="kronecker",
            pilot_ofdm_symbol_indices=[2],
        )

        estimator = LSChannelEstimator(
            resource_grid=rg,
            interpolation_type="lin",
            precision=precision,
            device=device,
        )

        batch_size = 2
        num_rx = 1
        num_rx_ant = 4
        num_ofdm_symbols = rg.num_ofdm_symbols
        fft_size = rg.fft_size

        y = complex_normal(
            (batch_size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size),
            precision=precision,
            device=device,
        )
        no = torch.ones(1, device=device) * 0.1

        h_hat, err_var = estimator(y, no)

        assert not torch.isnan(h_hat).any()
        assert not torch.isnan(err_var).any()

    def test_kronecker_multiple_streams(self, device, precision):
        """Test with multiple streams per TX."""
        rg = ResourceGrid(
            num_ofdm_symbols=14,
            fft_size=64,
            subcarrier_spacing=30e3,
            num_tx=2,
            num_streams_per_tx=4,
            pilot_pattern="kronecker",
            pilot_ofdm_symbol_indices=[2, 5, 8],
        )

        estimator = LSChannelEstimator(
            resource_grid=rg,
            interpolation_type="nn",
            precision=precision,
            device=device,
        )

        batch_size = 2
        num_rx = 1
        num_rx_ant = 8
        num_ofdm_symbols = rg.num_ofdm_symbols
        fft_size = rg.fft_size

        y = complex_normal(
            (batch_size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size),
            precision=precision,
            device=device,
        )
        no = torch.ones(1, device=device) * 0.1

        h_hat, err_var = estimator(y, no)

        expected_shape = (
            batch_size,
            num_rx,
            num_rx_ant,
            rg.num_tx,
            rg.num_streams_per_tx,
            num_ofdm_symbols,
            rg.num_effective_subcarriers,
        )
        assert h_hat.shape == expected_shape


# ============================================================================
# LMMSE Channel Estimator tests
# ============================================================================

class TestLMMSEChannelEstimator:
    """Tests for LMMSEChannelEstimator class."""

    # Batch size for the tests
    BATCH_SIZE = 1

    # SNR values for which tests are run
    EBN0_DBS = [0.0]

    # Allowed absolute error
    ATOL_SINGLE = 1e-3
    ATOL_DOUBLE = 1e-10

    # ========================================================================
    # Reference implementations
    # ========================================================================

    @staticmethod
    def pilot_pattern_to_pilot_mask(pilot_pattern):
        """Convert a pilot pattern to a boolean mask indicating pilot locations."""
        data_mask = pilot_pattern.mask.cpu().numpy()
        pilots = pilot_pattern.pilots.cpu().numpy()

        num_tx = data_mask.shape[0]
        num_streams_per_tx = data_mask.shape[1]
        num_ofdm_symbols = data_mask.shape[2]
        num_effective_subcarriers = data_mask.shape[3]
        pilot_mask = np.zeros(
            [num_tx, num_streams_per_tx, num_ofdm_symbols, num_effective_subcarriers],
            bool
        )
        for tx in range(num_tx):
            for st in range(num_streams_per_tx):
                pil_ind = 0
                for sb in range(num_ofdm_symbols):
                    for sc in range(num_effective_subcarriers):
                        if data_mask[tx, st, sb, sc]:
                            if np.abs(pilots[tx, st, pil_ind]) > 0.0:
                                pilot_mask[tx, st, sb, sc] = True
                            pil_ind += 1
        return pilot_mask

    @staticmethod
    def map_estimates_to_rg(h_hat, err_var, pilot_pattern):
        """Map pilot channel estimates to the full resource grid."""
        data_mask = pilot_pattern.mask.cpu().numpy()
        pilots = pilot_pattern.pilots.cpu().numpy()

        batch_size = h_hat.shape[0]
        num_rx = h_hat.shape[1]
        num_rx_ant = h_hat.shape[2]
        num_tx = h_hat.shape[3]
        num_streams_per_tx = h_hat.shape[4]
        num_ofdm_symbols = data_mask.shape[2]
        num_effective_subcarriers = data_mask.shape[3]
        h_hat_rg = np.zeros(
            [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx,
             num_ofdm_symbols, num_effective_subcarriers],
            complex
        )
        err_var_rg = np.zeros(
            [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx,
             num_ofdm_symbols, num_effective_subcarriers],
            float
        )
        for bs in range(batch_size):
            for rx in range(num_rx):
                for ra in range(num_rx_ant):
                    for tx in range(num_tx):
                        for st in range(num_streams_per_tx):
                            pil_ind = 0
                            for sb in range(num_ofdm_symbols):
                                for sc in range(num_effective_subcarriers):
                                    if data_mask[tx, st, sb, sc]:
                                        if np.abs(pilots[tx, st, pil_ind]) > 0.0:
                                            h_hat_rg[bs, rx, ra, tx, st, sb, sc] = h_hat[bs, rx, ra, tx, st, pil_ind]
                                            err_var_rg[bs, rx, ra, tx, st, sb, sc] = err_var[bs, rx, ra, tx, st, pil_ind]
                                        pil_ind += 1
        return h_hat_rg, err_var_rg

    @staticmethod
    def reference_lmmse_1d_one_axis(cov_mat, h_hat, err_var, pattern, last_step):
        """Reference LMMSE interpolation along one axis for a single slice."""
        err_var_old = err_var

        # Build interpolation matrix
        dim_size = pattern.shape[0]
        pil_ind, = np.where(pattern)
        num_pil = pil_ind.shape[0]

        pi_mat = np.zeros([dim_size, num_pil])
        k = 0
        for i in range(dim_size):
            if pattern[i]:
                pi_mat[i, k] = 1.0
                k += 1

        int_mat = np.matmul(np.matmul(pi_mat.T, cov_mat), pi_mat)
        err_var = np.take(err_var, pil_ind, axis=0)
        int_mat = int_mat + np.diag(err_var)
        int_mat = np.linalg.inv(int_mat)
        int_mat = np.matmul(pi_mat, np.matmul(int_mat, pi_mat.T))
        int_mat = np.matmul(cov_mat, int_mat)

        # Interpolation
        h_hat = np.matmul(int_mat, h_hat)

        # Error variance
        mask_mat = np.zeros([dim_size, dim_size])
        for i in range(dim_size):
            if pattern[i]:
                mask_mat[i, i] = 1.0
        err_var = cov_mat - np.matmul(int_mat, np.matmul(mask_mat, cov_mat))
        err_var = np.diag(err_var).real

        # Scaling if not last step
        if not last_step:
            int_mat_h = np.conj(int_mat.T)
            h_hat_var = np.matmul(int_mat, np.matmul(cov_mat + np.diag(err_var_old), int_mat_h))
            h_hat_var = np.diag(h_hat_var).real
            s = 2.0 / (1.0 + h_hat_var - err_var)
            h_hat = s * h_hat
            err_var = s * (s - 1) * h_hat_var + (1.0 - s) + s * err_var

        return h_hat, err_var

    @staticmethod
    def reference_spatial_smoothing_one_re(cov_mat, h_hat, err_var, last_step):
        """Reference spatial smoothing for a single resource element."""
        A = cov_mat + np.diag(err_var)
        A = np.linalg.inv(A)
        A = np.matmul(cov_mat, A)

        h_hat = np.expand_dims(h_hat, axis=-1)
        h_hat = np.matmul(A, h_hat)
        h_hat = np.squeeze(h_hat, axis=-1)

        err_var_out = cov_mat - np.matmul(A, cov_mat)
        err_var_out = np.diag(err_var_out).real

        if not last_step:
            Ah = np.conj(A.T)
            h_hat_var = np.matmul(A, np.matmul(cov_mat + np.diag(err_var), Ah))
            h_hat_var = np.diag(h_hat_var).real
            s = 2.0 / (1.0 + h_hat_var - err_var_out)
            h_hat = s * h_hat
            err_var_out = s * (s - 1) * h_hat_var + (1.0 - s) + s * err_var_out

        return h_hat, err_var_out

    def reference_spatial_smoothing(self, cov_mat, h_hat, err_var, last_step):
        """Reference spatial smoothing for all resource elements."""
        # [batch_size, num_rx, num_tx, num_tx_streams, num_ofdm_symbols, num_effective_subcarriers, num_rx_ant]
        h_hat = np.transpose(h_hat, [0, 1, 3, 4, 5, 6, 2])
        err_var = np.transpose(err_var, [0, 1, 3, 4, 5, 6, 2])

        h_hat_shape = h_hat.shape
        num_rx_ant = h_hat.shape[-1]
        h_hat = np.reshape(h_hat, [-1, num_rx_ant])
        err_var = np.reshape(err_var, [-1, num_rx_ant])

        for i, (h_hat_, err_var_) in enumerate(zip(h_hat, err_var)):
            h_hat_new, err_var_new = self.reference_spatial_smoothing_one_re(cov_mat, h_hat_, err_var_, last_step)
            h_hat[i] = h_hat_new
            err_var[i] = err_var_new

        h_hat = np.reshape(h_hat, h_hat_shape)
        err_var = np.reshape(err_var, h_hat_shape)

        # [batch_size, num_rx, num_rx_ant, num_tx, num_tx_streams, num_ofdm_symbols, num_effective_subcarriers]
        h_hat = np.transpose(h_hat, [0, 1, 6, 2, 3, 4, 5])
        err_var = np.transpose(err_var, [0, 1, 6, 2, 3, 4, 5])

        return h_hat, err_var

    def reference_lmmse_1d(self, cov_mat, h_hat, err_var, pattern, last_step):
        """Reference 1D LMMSE interpolation along last axis."""
        import itertools

        batch_size = h_hat.shape[0]
        num_rx = h_hat.shape[1]
        num_rx_ant = h_hat.shape[2]
        num_tx = h_hat.shape[3]
        num_tx_streams = h_hat.shape[4]
        outer_dim_size = h_hat.shape[5]

        for b, rx, ra, tx, ts, od in itertools.product(
            range(batch_size), range(num_rx), range(num_rx_ant),
            range(num_tx), range(num_tx_streams), range(outer_dim_size)
        ):
            h_hat_ = h_hat[b, rx, ra, tx, ts, od]
            err_var_ = err_var[b, rx, ra, tx, ts, od]
            pattern_ = pattern[tx, ts, od]
            if np.any(pattern_):
                h_hat_, err_var_ = self.reference_lmmse_1d_one_axis(cov_mat, h_hat_, err_var_, pattern_, last_step)
                h_hat[b, rx, ra, tx, ts, od] = h_hat_
                err_var[b, rx, ra, tx, ts, od] = err_var_

        # Update pattern
        pattern_update_mask = np.any(pattern, axis=-1, keepdims=True)
        pattern = np.logical_or(pattern, pattern_update_mask)

        return h_hat, err_var, pattern

    def reference_lmmse_interpolation(self, cov_mat_time, cov_mat_freq, cov_mat_space, h_hat, err_var, pilot_pattern, order):
        """Reference LMMSE interpolation with specified order."""
        pilot_mask = self.pilot_pattern_to_pilot_mask(pilot_pattern)
        h_hat, err_var = self.map_estimates_to_rg(h_hat, err_var, pilot_pattern)

        if order == "f-t":
            h_hat, err_var, pilot_mask = self.reference_lmmse_1d(cov_mat_freq, h_hat, err_var, pilot_mask, False)
            h_hat = np.transpose(h_hat, [0, 1, 2, 3, 4, 6, 5])
            err_var = np.transpose(err_var, [0, 1, 2, 3, 4, 6, 5])
            pilot_mask = np.transpose(pilot_mask, [0, 1, 3, 2])
            h_hat, err_var, _ = self.reference_lmmse_1d(cov_mat_time, h_hat, err_var, pilot_mask, True)
            h_hat = np.transpose(h_hat, [0, 1, 2, 3, 4, 6, 5])
            err_var = np.transpose(err_var, [0, 1, 2, 3, 4, 6, 5])
        elif order == "t-f":
            h_hat = np.transpose(h_hat, [0, 1, 2, 3, 4, 6, 5])
            err_var = np.transpose(err_var, [0, 1, 2, 3, 4, 6, 5])
            pilot_mask = np.transpose(pilot_mask, [0, 1, 3, 2])
            h_hat, err_var, pilot_mask = self.reference_lmmse_1d(cov_mat_time, h_hat, err_var, pilot_mask, False)
            h_hat = np.transpose(h_hat, [0, 1, 2, 3, 4, 6, 5])
            err_var = np.transpose(err_var, [0, 1, 2, 3, 4, 6, 5])
            pilot_mask = np.transpose(pilot_mask, [0, 1, 3, 2])
            h_hat, err_var, _ = self.reference_lmmse_1d(cov_mat_freq, h_hat, err_var, pilot_mask, True)
        elif order == "t-s-f":
            h_hat = np.transpose(h_hat, [0, 1, 2, 3, 4, 6, 5])
            err_var = np.transpose(err_var, [0, 1, 2, 3, 4, 6, 5])
            pilot_mask = np.transpose(pilot_mask, [0, 1, 3, 2])
            h_hat, err_var, pilot_mask = self.reference_lmmse_1d(cov_mat_time, h_hat, err_var, pilot_mask, False)
            h_hat = np.transpose(h_hat, [0, 1, 2, 3, 4, 6, 5])
            err_var = np.transpose(err_var, [0, 1, 2, 3, 4, 6, 5])
            pilot_mask = np.transpose(pilot_mask, [0, 1, 3, 2])
            h_hat, err_var = self.reference_spatial_smoothing(cov_mat_space, h_hat, err_var, False)
            h_hat, err_var, _ = self.reference_lmmse_1d(cov_mat_freq, h_hat, err_var, pilot_mask, True)

        return h_hat, err_var

    # ========================================================================
    # Basic shape and functionality tests
    # ========================================================================

    def test_output_shape(self, device, precision):
        """Test that LMMSEChannelEstimator produces correct output shape."""
        rg = ResourceGrid(
            num_ofdm_symbols=14,
            fft_size=64,
            subcarrier_spacing=30e3,
            num_tx=2,
            num_streams_per_tx=2,
            pilot_pattern="kronecker",
            pilot_ofdm_symbol_indices=[2, 11],
            precision=precision,
            device=device,
        )

        cov_mat_freq = tdl_freq_cov_mat("A", 30e3, 64, 100e-9, precision=precision)
        cov_mat_time = tdl_time_cov_mat("A", 10.0, 3.5e9, 35.7e-6, 14, precision=precision)

        estimator = LMMSEChannelEstimator(
            resource_grid=rg,
            cov_mat_time=cov_mat_time,
            cov_mat_freq=cov_mat_freq,
            order="f-t",
            precision=precision,
            device=device,
        )

        batch_size = 4
        num_rx = 1
        num_rx_ant = 4
        num_tx = rg.num_tx
        num_streams_per_tx = rg.num_streams_per_tx
        num_ofdm_symbols = rg.num_ofdm_symbols
        num_effective_subcarriers = rg.num_effective_subcarriers
        fft_size = rg.fft_size

        y = complex_normal(
            (batch_size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size),
            precision=precision,
            device=device,
        )
        no = torch.ones(1, device=device) * 0.1

        h_hat, err_var = estimator(y, no)

        expected_shape = (
            batch_size,
            num_rx,
            num_rx_ant,
            num_tx,
            num_streams_per_tx,
            num_ofdm_symbols,
            num_effective_subcarriers,
        )
        assert h_hat.shape == expected_shape
        assert err_var.shape == expected_shape

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
    @pytest.mark.parametrize(
        ("config_device", "explicit_device"),
        [
            ("cpu", "cuda:0"),
            ("cuda:0", "cpu"),
        ],
    )
    def test_explicit_device_overrides_config_device(
        self,
        monkeypatch,
        config_device,
        explicit_device,
    ):
        """Test that explicit device and precision propagate to nested filters."""
        monkeypatch.setattr(config, "device", config_device)
        device = explicit_device
        precision = "double"

        resource_grid = ResourceGrid(
            num_ofdm_symbols=4,
            fft_size=8,
            subcarrier_spacing=30e3,
            num_tx=1,
            num_streams_per_tx=1,
            pilot_pattern="kronecker",
            pilot_ofdm_symbol_indices=[1, 3],
            precision=precision,
            device=device,
        )
        cov_mat_freq = tdl_freq_cov_mat(
            "A", 30e3, 8, 100e-9, precision=precision
        )
        cov_mat_time = tdl_time_cov_mat(
            "A", 3.0, 3.5e9, 35.7e-6, 4, precision=precision
        )
        estimator = LMMSEChannelEstimator(
            resource_grid,
            cov_mat_time,
            cov_mat_freq,
            order="f-t",
            precision=precision,
            device=device,
        )

        y = complex_normal(
            (2, 1, 1, 4, 8),
            precision=precision,
            device=device,
        )
        no = torch.tensor(0.1, dtype=torch.float64, device=device)

        h_hat, err_var = estimator(y, no)

        assert h_hat.device == torch.device(device)
        assert err_var.device == torch.device(device)
        assert h_hat.dtype == torch.complex128
        assert err_var.dtype == torch.float64

    @pytest.mark.parametrize("order", ["f-t", "t-f"])
    def test_output_valid(self, device, precision, order):
        """Test that LMMSEChannelEstimator produces valid (non-NaN) output."""
        rg = ResourceGrid(
            num_ofdm_symbols=14,
            fft_size=64,
            subcarrier_spacing=30e3,
            num_tx=2,
            num_streams_per_tx=1,
            pilot_pattern="kronecker",
            pilot_ofdm_symbol_indices=[2, 11],
            precision=precision,
            device=device,
        )

        cov_mat_freq = tdl_freq_cov_mat("A", 30e3, 64, 100e-9, precision=precision)
        cov_mat_time = tdl_time_cov_mat("A", 10.0, 3.5e9, 35.7e-6, 14, precision=precision)

        estimator = LMMSEChannelEstimator(
            resource_grid=rg,
            cov_mat_time=cov_mat_time,
            cov_mat_freq=cov_mat_freq,
            order=order,
            precision=precision,
            device=device,
        )

        batch_size = 2
        num_rx = 1
        num_rx_ant = 2
        num_ofdm_symbols = rg.num_ofdm_symbols
        fft_size = rg.fft_size

        y = complex_normal(
            (batch_size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size),
            precision=precision,
            device=device,
        )
        no = torch.ones(1, device=device) * 0.1

        h_hat, err_var = estimator(y, no)

        assert not torch.isnan(h_hat).any()
        assert not torch.isnan(err_var).any()
        assert (err_var >= 0).all()

    def test_with_spatial_smoothing(self, device, precision):
        """Test LMMSEChannelEstimator including spatial smoothing."""
        num_rx_ant = 4
        rg = ResourceGrid(
            num_ofdm_symbols=14,
            fft_size=32,
            subcarrier_spacing=30e3,
            num_tx=2,
            num_streams_per_tx=1,
            pilot_pattern="kronecker",
            pilot_ofdm_symbol_indices=[2, 11],
            precision=precision,
            device=device,
        )

        cov_mat_freq = tdl_freq_cov_mat("A", 30e3, 32, 100e-9, precision)
        cov_mat_time = tdl_time_cov_mat("A", 10.0, 3.5e9, 35.7e-6, 14, precision=precision)
        cov_mat_space = exp_corr_mat(0.9, num_rx_ant, precision=precision)

        estimator = LMMSEChannelEstimator(
            resource_grid=rg,
            cov_mat_time=cov_mat_time,
            cov_mat_freq=cov_mat_freq,
            cov_mat_space=cov_mat_space,
            order="t-s-f",
            precision=precision,
            device=device,
        )

        batch_size = 2
        num_rx = 1
        num_ofdm_symbols = rg.num_ofdm_symbols
        fft_size = rg.fft_size

        y = complex_normal(
            (batch_size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size),
            precision=precision,
            device=device,
        )
        no = torch.ones(1, device=device) * 0.1

        h_hat, err_var = estimator(y, no)

        assert not torch.isnan(h_hat).any()
        assert not torch.isnan(err_var).any()
        assert (err_var >= 0).all()

    # ========================================================================
    # Order validation tests
    # ========================================================================

    def test_order_validation_invalid_string(self, precision):
        """Test that invalid order strings raise an error."""
        rg = ResourceGrid(
            num_ofdm_symbols=14,
            fft_size=12,
            subcarrier_spacing=30e3,
            num_tx=4,
            num_streams_per_tx=1,
            pilot_pattern="kronecker",
            pilot_ofdm_symbol_indices=[2, 11],
            precision=precision,
        )

        cov_mat_freq = tdl_freq_cov_mat("A", 30e3, 12, 100e-9, precision)
        cov_mat_time = tdl_time_cov_mat("A", 10.0, 3.5e9, 35.7e-6, 14, precision=precision)

        with pytest.raises(ValueError):
            LMMSEChannelEstimator(rg, cov_mat_time, cov_mat_freq, order="hello")

    def test_order_validation_double_dash(self, precision):
        """Test that double dashes in order raise an error."""
        rg = ResourceGrid(
            num_ofdm_symbols=14,
            fft_size=12,
            subcarrier_spacing=30e3,
            num_tx=4,
            num_streams_per_tx=1,
            pilot_pattern="kronecker",
            pilot_ofdm_symbol_indices=[2, 11],
            precision=precision,
        )

        cov_mat_freq = tdl_freq_cov_mat("A", 30e3, 12, 100e-9, precision)
        cov_mat_time = tdl_time_cov_mat("A", 10.0, 3.5e9, 35.7e-6, 14, precision=precision)

        with pytest.raises(ValueError):
            LMMSEChannelEstimator(rg, cov_mat_time, cov_mat_freq, order="f--t")

    def test_order_validation_duplicate_dims(self, precision):
        """Test that duplicate dimensions in order raise an error."""
        rg = ResourceGrid(
            num_ofdm_symbols=14,
            fft_size=12,
            subcarrier_spacing=30e3,
            num_tx=4,
            num_streams_per_tx=1,
            pilot_pattern="kronecker",
            pilot_ofdm_symbol_indices=[2, 11],
            precision=precision,
        )

        cov_mat_freq = tdl_freq_cov_mat("A", 30e3, 12, 100e-9, precision)
        cov_mat_time = tdl_time_cov_mat("A", 10.0, 3.5e9, 35.7e-6, 14, precision=precision)

        with pytest.raises(ValueError):
            LMMSEChannelEstimator(rg, cov_mat_time, cov_mat_freq, order="f-f-t")

        with pytest.raises(ValueError):
            LMMSEChannelEstimator(rg, cov_mat_time, cov_mat_freq, order="f-t-t")

    def test_order_validation_missing_dims(self, precision):
        """Test that missing time or frequency dimensions raise an error."""
        num_rx_ant = 4
        rg = ResourceGrid(
            num_ofdm_symbols=14,
            fft_size=12,
            subcarrier_spacing=30e3,
            num_tx=4,
            num_streams_per_tx=1,
            pilot_pattern="kronecker",
            pilot_ofdm_symbol_indices=[2, 11],
            precision=precision,
        )

        cov_mat_freq = tdl_freq_cov_mat("A", 30e3, 12, 100e-9, precision)
        cov_mat_time = tdl_time_cov_mat("A", 10.0, 3.5e9, 35.7e-6, 14, precision=precision)
        cov_mat_space = exp_corr_mat(0.9, num_rx_ant, precision=precision)

        with pytest.raises(ValueError):
            LMMSEChannelEstimator(rg, cov_mat_time, cov_mat_freq, cov_mat_space, order="f-s")

        with pytest.raises(ValueError):
            LMMSEChannelEstimator(rg, cov_mat_time, cov_mat_freq, cov_mat_space, order="s-t")

    def test_order_validation_spatial_without_cov_mat(self, precision):
        """Test that spatial smoothing without spatial covariance matrix raises an error."""
        rg = ResourceGrid(
            num_ofdm_symbols=14,
            fft_size=12,
            subcarrier_spacing=30e3,
            num_tx=4,
            num_streams_per_tx=1,
            pilot_pattern="kronecker",
            pilot_ofdm_symbol_indices=[2, 11],
            precision=precision,
        )

        cov_mat_freq = tdl_freq_cov_mat("A", 30e3, 12, 100e-9, precision)
        cov_mat_time = tdl_time_cov_mat("A", 10.0, 3.5e9, 35.7e-6, 14, precision=precision)

        with pytest.raises(ValueError):
            LMMSEChannelEstimator(rg, cov_mat_time, cov_mat_freq, order="f-t-s")

    # ========================================================================
    # E2E tests against reference implementation
    # ========================================================================

    def run_e2e_lmmse_test(self, batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx,
                           num_ofdm_symbols, fft_size, mask, pilots, ebno_db, precision):
        """Run end-to-end test comparing Sionna LMMSE to reference implementation."""
        device = "cpu"  # Use CPU for reproducibility

        # Create pilot pattern with the same device
        pilot_pattern = PilotPattern(mask, pilots, precision=precision, device=device)
        tdl_model = "A"
        subcarrier_spacing = 30e3
        num_bits_per_symbol = 2
        delay_spread = 300e-9
        carrier_frequency = 3.5e9
        speed = 5.0
        los_angle_of_arrival = np.pi / 4.0

        sm = StreamManagement(np.ones([num_rx, num_tx]), num_streams_per_tx)
        rg = ResourceGrid(
            num_ofdm_symbols=num_ofdm_symbols,
            fft_size=fft_size,
            subcarrier_spacing=subcarrier_spacing,
            num_tx=num_tx,
            num_streams_per_tx=num_streams_per_tx,
            cyclic_prefix_length=0,
            pilot_pattern=pilot_pattern,
            precision=precision,
            device=device,
        )

        qam_source = QAMSource(num_bits_per_symbol, precision=precision, device=device)
        rg_mapper = ResourceGridMapper(rg, precision=precision, device=device)

        channel_model = TDL(
            tdl_model, delay_spread, carrier_frequency,
            min_speed=speed, max_speed=speed,
            los_angle_of_arrival=los_angle_of_arrival,
            precision=precision, device=device
        )
        channel_freq = ApplyOFDMChannel(add_awgn=True, precision=precision, device=device)
        frequencies = subcarrier_frequencies(fft_size, subcarrier_spacing, precision=precision, device=device)

        cov_mat_freq = tdl_freq_cov_mat(tdl_model, subcarrier_spacing, fft_size, delay_spread, precision)
        cov_mat_time = tdl_time_cov_mat(
            tdl_model, speed, carrier_frequency, rg.ofdm_symbol_duration,
            num_ofdm_symbols, los_angle_of_arrival, precision
        )
        cov_mat_space = exp_corr_mat(0.9, num_rx_ant, precision=precision)

        lmmse_est_ft = LMMSEChannelEstimator(rg, cov_mat_time, cov_mat_freq, order="f-t", precision=precision, device=device)


        lmmse_est_tf = LMMSEChannelEstimator(rg, cov_mat_time, cov_mat_freq, order="t-f", precision=precision, device=device)

        lmmse_est_tsf = LMMSEChannelEstimator(rg, cov_mat_time, cov_mat_freq, cov_mat_space, order="t-s-f", precision=precision, device=device)

        ls_estimator = LSChannelEstimator(rg, precision=precision, device=device)

        no = ebnodb2no(ebno_db, num_bits_per_symbol, coderate=1.0, device=device)
        x = qam_source([batch_size, num_tx, num_streams_per_tx, rg.num_data_symbols])
        x_rg = rg_mapper(x)

        a, tau = channel_model(batch_size, num_ofdm_symbols, sampling_frequency=1.0 / rg.ofdm_symbol_duration)
        h_freq = cir_to_ofdm_channel(frequencies, a, tau, normalize=True)
        y = channel_freq(x_rg, h_freq, no)

        h_hat_lmmse_ft, err_var_lmmse_ft = lmmse_est_ft(y, no)
        h_hat_lmmse_tf, err_var_lmmse_tf = lmmse_est_tf(y, no)
        h_hat_lmmse_tsf, err_var_lmmse_tsf = lmmse_est_tsf(y, no)
        y_pilots = ls_estimator._extract_pilots(y)
        h_hat_no_int, err_var_no_int = ls_estimator.estimate_at_pilot_locations(y_pilots, no)

        # Convert to numpy
        h_hat_no_int_np = h_hat_no_int.cpu().numpy()
        err_var_no_int_np = err_var_no_int.cpu().numpy()
        err_var_no_int_np = np.broadcast_to(err_var_no_int_np, h_hat_no_int_np.shape)

        # Reference estimates
        h_hat_lmmse_ft_ref, err_var_lmmse_ft_ref = self.reference_lmmse_interpolation(
            cov_mat_time.cpu().numpy(), cov_mat_freq.cpu().numpy(), cov_mat_space.cpu().numpy(),
            h_hat_no_int_np, err_var_no_int_np, pilot_pattern, "f-t"
        )
        h_hat_lmmse_tf_ref, err_var_lmmse_tf_ref = self.reference_lmmse_interpolation(
            cov_mat_time.cpu().numpy(), cov_mat_freq.cpu().numpy(), cov_mat_space.cpu().numpy(),
            h_hat_no_int_np, err_var_no_int_np, pilot_pattern, "t-f"
        )
        h_hat_lmmse_tsf_ref, err_var_lmmse_tsf_ref = self.reference_lmmse_interpolation(
            cov_mat_time.cpu().numpy(), cov_mat_freq.cpu().numpy(), cov_mat_space.cpu().numpy(),
            h_hat_no_int_np, err_var_no_int_np, pilot_pattern, "t-s-f"
        )

        # Compute errors
        max_err_h_hat_ft = np.max(np.abs(h_hat_lmmse_ft_ref - h_hat_lmmse_ft.cpu().numpy()))
        max_err_err_var_ft = np.max(np.abs(err_var_lmmse_ft_ref - err_var_lmmse_ft.cpu().numpy()))
        max_err_h_hat_tf = np.max(np.abs(h_hat_lmmse_tf_ref - h_hat_lmmse_tf.cpu().numpy()))
        max_err_err_var_tf = np.max(np.abs(err_var_lmmse_tf_ref - err_var_lmmse_tf.cpu().numpy()))
        max_err_h_hat_tsf = np.max(np.abs(h_hat_lmmse_tsf_ref - h_hat_lmmse_tsf.cpu().numpy()))
        max_err_err_var_tsf = np.max(np.abs(err_var_lmmse_tsf_ref - err_var_lmmse_tsf.cpu().numpy()))

        return (max_err_h_hat_ft, max_err_err_var_ft, max_err_h_hat_tf, max_err_err_var_tf,
                max_err_h_hat_tsf, max_err_err_var_tsf)

    def _run_lmmse_e2e_test(self, num_rx, num_rx_ant, num_tx, num_streams_per_tx,
                            num_ofdm_symbols, fft_size, mask, pilots, precision):
        """Helper to run LMMSE E2E test with given parameters."""
        atol = self.ATOL_SINGLE if precision == "single" else self.ATOL_DOUBLE

        for ebno_db in self.EBN0_DBS:
            errors = self.run_e2e_lmmse_test(
                self.BATCH_SIZE, num_rx, num_rx_ant, num_tx, num_streams_per_tx,
                num_ofdm_symbols, fft_size, mask, pilots, ebno_db, precision
            )

            assert np.allclose(errors[0], 0.0, atol=atol), f"h_hat f-t error: {errors[0]}"
            assert np.allclose(errors[1], 0.0, atol=atol), f"err_var f-t error: {errors[1]}"
            assert np.allclose(errors[2], 0.0, atol=atol), f"h_hat t-f error: {errors[2]}"
            assert np.allclose(errors[3], 0.0, atol=atol), f"err_var t-f error: {errors[3]}"
            assert np.allclose(errors[4], 0.0, atol=atol), f"h_hat t-s-f error: {errors[4]}"
            assert np.allclose(errors[5], 0.0, atol=atol), f"err_var t-s-f error: {errors[5]}"

    @pytest.mark.parametrize("precision", ["single", "double"])
    def test_lmmse_sparse_pilot_pattern(self, precision):
        """Test LMMSE with sparse pilot pattern."""
        num_tx = 4
        num_streams_per_tx = 1
        num_ofdm_symbols = 14
        fft_size = 12
        mask = np.zeros([num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size], bool)
        mask[..., 5, :] = True
        num_pilots = np.sum(mask[0, 0])
        pilots = np.zeros([num_tx, num_streams_per_tx, num_pilots])
        pilots[0, 0, [0, 11]] = 1
        pilots[1, 0, 1] = 1
        pilots[2, 0, 5] = 1
        pilots[3, 0, 10] = 1

        self._run_lmmse_e2e_test(1, 1, num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size, mask, pilots, precision)

    @pytest.mark.parametrize("precision", ["single", "double"])
    def test_lmmse_kronecker_01(self, precision):
        """Test LMMSE with Kronecker pilot pattern - standard case."""
        num_tx = 1
        num_streams_per_tx = 1
        num_ofdm_symbols = 14
        fft_size = 64
        pilot_ofdm_symbol_indices = [2, 11]

        rg = ResourceGrid(
            num_ofdm_symbols=num_ofdm_symbols,
            fft_size=fft_size,
            subcarrier_spacing=30e3,
            num_tx=num_tx,
            num_streams_per_tx=num_streams_per_tx,
            cyclic_prefix_length=0,
            pilot_pattern="kronecker",
            pilot_ofdm_symbol_indices=pilot_ofdm_symbol_indices,
            precision=precision,
        )
        pilot_pattern = rg.pilot_pattern

        self._run_lmmse_e2e_test(
            1, 1, num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size,
            pilot_pattern.mask.cpu().numpy(), pilot_pattern.pilots.cpu().numpy(), precision
        )

    @pytest.mark.parametrize("precision", ["single", "double"])
    def test_lmmse_kronecker_02(self, precision):
        """Test LMMSE with Kronecker pilot pattern - single pilot symbol."""
        num_tx = 4
        num_streams_per_tx = 1
        num_ofdm_symbols = 14
        fft_size = 64
        pilot_ofdm_symbol_indices = [2]

        rg = ResourceGrid(
            num_ofdm_symbols=num_ofdm_symbols,
            fft_size=fft_size,
            subcarrier_spacing=30e3,
            num_tx=num_tx,
            num_streams_per_tx=num_streams_per_tx,
            cyclic_prefix_length=0,
            pilot_pattern="kronecker",
            pilot_ofdm_symbol_indices=pilot_ofdm_symbol_indices,
            precision=precision,
        )
        pilot_pattern = rg.pilot_pattern

        self._run_lmmse_e2e_test(
            1, 1, num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size,
            pilot_pattern.mask.cpu().numpy(), pilot_pattern.pilots.cpu().numpy(), precision
        )

    @pytest.mark.parametrize("precision", ["single", "double"])
    def test_lmmse_kronecker_03(self, precision):
        """Test LMMSE with Kronecker pilot pattern - one pilot per TX."""
        num_tx = 16
        num_streams_per_tx = 1
        num_ofdm_symbols = 14
        fft_size = 16
        pilot_ofdm_symbol_indices = [2]

        rg = ResourceGrid(
            num_ofdm_symbols=num_ofdm_symbols,
            fft_size=fft_size,
            subcarrier_spacing=30e3,
            num_tx=num_tx,
            num_streams_per_tx=num_streams_per_tx,
            cyclic_prefix_length=0,
            pilot_pattern="kronecker",
            pilot_ofdm_symbol_indices=pilot_ofdm_symbol_indices,
            precision=precision,
        )
        pilot_pattern = rg.pilot_pattern

        self._run_lmmse_e2e_test(
            1, 1, num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size,
            pilot_pattern.mask.cpu().numpy(), pilot_pattern.pilots.cpu().numpy(), precision
        )

    @pytest.mark.parametrize("precision", ["single", "double"])
    def test_lmmse_kronecker_04(self, precision):
        """Test LMMSE with Kronecker pilot pattern - multi TX, multi stream."""
        num_tx = 4
        num_streams_per_tx = 2
        num_ofdm_symbols = 14
        fft_size = 64
        pilot_ofdm_symbol_indices = [2, 5, 8]

        rg = ResourceGrid(
            num_ofdm_symbols=num_ofdm_symbols,
            fft_size=fft_size,
            subcarrier_spacing=30e3,
            num_tx=num_tx,
            num_streams_per_tx=num_streams_per_tx,
            cyclic_prefix_length=0,
            pilot_pattern="kronecker",
            pilot_ofdm_symbol_indices=pilot_ofdm_symbol_indices,
            precision=precision,
        )
        pilot_pattern = rg.pilot_pattern

        self._run_lmmse_e2e_test(
            1, 1, num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size,
            pilot_pattern.mask.cpu().numpy(), pilot_pattern.pilots.cpu().numpy(), precision
        )

    @pytest.mark.parametrize("precision", ["single", "double"])
    def test_lmmse_kronecker_05(self, precision):
        """Test LMMSE with Kronecker pilot pattern - single TX, only pilots."""
        num_tx = 1
        num_streams_per_tx = 1
        num_ofdm_symbols = 5
        fft_size = 64
        pilot_ofdm_symbol_indices = list(range(num_ofdm_symbols))

        rg = ResourceGrid(
            num_ofdm_symbols=num_ofdm_symbols,
            fft_size=fft_size,
            subcarrier_spacing=30e3,
            num_tx=num_tx,
            num_streams_per_tx=num_streams_per_tx,
            cyclic_prefix_length=0,
            pilot_pattern="kronecker",
            pilot_ofdm_symbol_indices=pilot_ofdm_symbol_indices,
            precision=precision,
        )
        pilot_pattern = rg.pilot_pattern

        self._run_lmmse_e2e_test(
            1, 1, num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size,
            pilot_pattern.mask.cpu().numpy(), pilot_pattern.pilots.cpu().numpy(), precision
        )

    @pytest.mark.parametrize("precision", ["single", "double"])
    def test_lmmse_kronecker_06(self, precision):
        """Test LMMSE with Kronecker pilot pattern - multiple pilot symbols."""
        num_tx = 4
        num_streams_per_tx = 1
        num_ofdm_symbols = 14
        fft_size = 64
        pilot_ofdm_symbol_indices = [2, 3, 8, 11]

        rg = ResourceGrid(
            num_ofdm_symbols=num_ofdm_symbols,
            fft_size=fft_size,
            subcarrier_spacing=30e3,
            num_tx=num_tx,
            num_streams_per_tx=num_streams_per_tx,
            cyclic_prefix_length=0,
            pilot_pattern="kronecker",
            pilot_ofdm_symbol_indices=pilot_ofdm_symbol_indices,
            precision=precision,
        )
        pilot_pattern = rg.pilot_pattern

        self._run_lmmse_e2e_test(
            1, 1, num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size,
            pilot_pattern.mask.cpu().numpy(), pilot_pattern.pilots.cpu().numpy(), precision
        )

class _PilotGroupUser(_PilotGroupMixin, Object):
    """Minimal Sionna object exercising only PilotGroupMixin."""

    def __init__(self, pilot_pattern, group_over, device):
        super().__init__(device=device)
        self._init_pilot_groups(
            pilot_pattern=pilot_pattern,
            group_over=group_over,
        )

class TestPilotGroupMixin:
    """Tests for pilot grouping, coordinate access, and gather/scatter."""

    @staticmethod
    def _resource_grid(
        *,
        fft_size,
        num_ofdm_symbols,
        pilot_symbols,
        num_tx,
        num_streams,
        device,
    ):
        return ResourceGrid(
            num_ofdm_symbols=num_ofdm_symbols,
            fft_size=fft_size,
            subcarrier_spacing=30e3,
            num_tx=num_tx,
            num_streams_per_tx=num_streams,
            pilot_pattern="kronecker",
            pilot_ofdm_symbol_indices=list(pilot_symbols),
            device=device,
        )

    @pytest.mark.parametrize(
        "fft_size,num_symbols,pilot_symbols,num_tx,num_streams",
        [
            (12, 7, (1, 5), 2, 1),
            (16, 9, (0, 4, 8), 2, 2),
        ],
    )
    def test_subcarrier_groups_have_expected_coordinates(
        self, device, fft_size, num_symbols, pilot_symbols, num_tx, num_streams
    ):
        """Subcarrier groups follow deterministic TX-stream-symbol ordering."""
        grid = TestPilotGroupMixin._resource_grid(
            fft_size=fft_size,
            num_ofdm_symbols=num_symbols,
            pilot_symbols=pilot_symbols,
            num_tx=num_tx,
            num_streams=num_streams,
            device=device,
        )
        user = _PilotGroupUser(
            grid.pilot_pattern, ("subcarrier",), device
        )

        expected_tx = []
        expected_stream = []
        expected_symbol = []
        expected_subcarriers = []
        num_sequences = num_tx * num_streams

        for tx_index in range(num_tx):
            for stream_index in range(num_streams):
                sequence_index = tx_index * num_streams + stream_index
                for symbol_index in pilot_symbols:
                    expected_tx.append(tx_index)
                    expected_stream.append(stream_index)
                    expected_symbol.append(symbol_index)
                    expected_subcarriers.append(
                        torch.arange(
                            sequence_index,
                            fft_size,
                            num_sequences,
                            device=device,
                        )
                    )

        torch.testing.assert_close(
            user._pilot_group_constant_coord("tx"),
            torch.tensor(expected_tx, device=device),
        )
        torch.testing.assert_close(
            user._pilot_group_constant_coord("stream"),
            torch.tensor(expected_stream, device=device),
        )
        torch.testing.assert_close(
            user._pilot_group_constant_coord("symbol"),
            torch.tensor(expected_symbol, device=device),
        )
        torch.testing.assert_close(
            user._pilot_group_varying_coord("subcarrier"),
            torch.stack(expected_subcarriers),
        )

    @pytest.mark.parametrize("group_over", [("subcarrier",), ("symbol",)])
    def test_gather_scatter_is_lossless_for_active_pilots(self, device, group_over):
        """Gathering and scattering preserves every active compact pilot value."""
        grid = TestPilotGroupMixin._resource_grid(
            fft_size=12,
            num_ofdm_symbols=7,
            pilot_symbols=(1, 5),
            num_tx=2,
            num_streams=1,
            device=device,
        )
        user = _PilotGroupUser(grid.pilot_pattern, group_over, device)

        pilots = grid.pilot_pattern.pilots
        active_values = torch.arange(
            pilots.numel(), device=device, dtype=pilots.real.dtype
        ).reshape_as(pilots)
        values = torch.where(pilots.abs() > 0, active_values, 0)
        values = values.expand(3, 2, *values.shape)

        grouped = user._gather_pilot_groups(values)
        restored = user._scatter_pilot_groups(
            torch.zeros_like(values), grouped
        )
        torch.testing.assert_close(restored, values)

    def test_unknown_group_dimension_is_rejected(self, device):
        """Unknown coordinate names fail during construction."""
        grid = TestPilotGroupMixin._resource_grid(
            fft_size=12,
            num_ofdm_symbols=7,
            pilot_symbols=(1, 5),
            num_tx=2,
            num_streams=1,
            device=device,
        )
        with pytest.raises(ValueError, match="Unknown pilot group dimension"):
            _PilotGroupUser(grid.pilot_pattern, ("frequency",), device)

    def test_non_kronecker_groups(self, device):
        """Non-Kronecker pilot patterns are supported."""
        mask = torch.zeros((1, 1, 3, 6), dtype=torch.bool, device=device)
        mask[0, 0, 0, [0, 2, 5]] = True
        mask[0, 0, 2, [1, 3, 4]] = True
        pilots = torch.ones((1, 1, 6), dtype=torch.complex64, device=device)
        pattern = PilotPattern(mask, pilots, device=device)

        user = _PilotGroupUser(pattern, ("subcarrier",), device)

        torch.testing.assert_close(
            user._pilot_group_constant_coord("symbol"),
            torch.tensor([0, 2], device=device),
        )
        torch.testing.assert_close(
            user._pilot_group_varying_coord("subcarrier"),
            torch.tensor([[0, 2, 5], [1, 3, 4]], device=device),
        )

    def test_multi_dimensional_groups(self, device):
        """Multi-dimensional groups are supported."""
        grid = TestPilotGroupMixin._resource_grid(
            fft_size=12,
            num_ofdm_symbols=7,
            pilot_symbols=(1, 5),
            num_tx=2,
            num_streams=1,
            device=device,
        )
        user = _PilotGroupUser(
            grid.pilot_pattern,
            ("symbol", "subcarrier"),
            device,
        )
        torch.testing.assert_close(
            user._pilot_group_constant_coord("tx"),
            torch.tensor([0, 1], device=device)
        )
        torch.testing.assert_close(
            user._pilot_group_constant_coord("stream"),
            torch.tensor([0, 0], device=device)
        )
        with pytest.raises(
            ValueError,
            match="Pilot groups do not have a fixed symbol coordinate"
        ):
            user._pilot_group_constant_coord("symbol")

    def test_ragged_active_groups(self, device):
        """Ragged active groups are rejected."""
        mask = torch.zeros((1, 1, 3, 6), dtype=torch.bool, device=device)
        mask[0, 0, 0, [0, 2, 5]] = True
        mask[0, 0, 2, [1, 3, 4]] = True
        pilots = torch.ones((1, 1, 6), dtype=torch.complex64, device=device)
        pilots[..., -1] = 0

        with pytest.raises(ValueError, match="equal-length groups"):
            _PilotGroupUser(
                PilotPattern(mask, pilots, device=device),
                ("subcarrier",),
                device,
            )

    def test_gather_compiles_as_full_graph(self, device):
        """Pilot gathering produces the same eager and compiled result."""
        grid = TestPilotGroupMixin._resource_grid(
            fft_size=12,
            num_ofdm_symbols=7,
            pilot_symbols=(1, 5),
            num_tx=2,
            num_streams=1,
            device=device,
        )
        user = _PilotGroupUser(
            grid.pilot_pattern, ("subcarrier",), device
        )
        values = grid.pilot_pattern.pilots.expand(
            2, 3, *grid.pilot_pattern.pilots.shape
        )

        compiled = torch.compile(
            user._gather_pilot_groups,
            backend="eager",
            fullgraph=True,
        )
        torch.testing.assert_close(
            compiled(values),
            user._gather_pilot_groups(values)
        )
