#
# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0#

import unittest
import numpy as np
import tensorflow as tf
from sionna.phy.channel import exp_corr_mat, one_ring_corr_mat, \
    cir_to_time_channel, time_to_ofdm_channel, ApplyTimeChannel
from sionna.phy.channel.tr38901 import TDL
from sionna.phy.ofdm import ResourceGrid, ResourceGridMapper, OFDMModulator, OFDMDemodulator, LSChannelEstimator
from sionna.phy.mimo import StreamManagement
from sionna.phy.mapping import QAMSource
from sionna.phy import config, dtypes
from sionna.phy.utils import flatten_dims
from channel_test_utils import exp_corr_mat_numpy, one_ring_corr_numpy


class TestExpCorrMat(unittest.TestCase):
    def test_single_dim(self):
        "Tests for scalar inputs"
        values = [0.0, 0.9999, 0.5+1j*0.3]
        dims = [1, 2, 4, 7, 64, 128]
        for a in values:
            for n in dims:
                for precision in ["single", "double"]:
                    R1 = exp_corr_mat_numpy(a, n, precision)
                    R2 = exp_corr_mat(a, n, precision)
                    err = np.max(np.abs(R1-R2))
                    self.assertAlmostEqual(err, 0, 5)

    def test_catch_abs_val_error(self):
        """Absolute value of a greater than 1 should raise error"""
        with self.assertRaises(tf.errors.InvalidArgumentError):
            exp_corr_mat(1.1+1j*0.3, 12)

    def test_multiple_dims(self):
        values = config.np_rng.uniform(0, 1, [2, 4, 3])
        n = 11
        precision = "double"
        R2 = exp_corr_mat(values, n, precision)
        R2 = tf.reshape(R2, [-1, n, n])
        for i, a in enumerate(np.reshape(values, [-1])):
            R1 = exp_corr_mat_numpy(a, n, precision)
            err = np.max(np.abs(R1-R2[i]))
            self.assertAlmostEqual(err, 0)

class TestOneRingCorrMat(unittest.TestCase):
    def test_single_dim(self):
        "Tests for scalar inputs"
        phi_degs = [-180, -90, -45, -12, 0, 15, 45, 65, 90, 180, 360]
        num_ants = [1, 4, 16, 128]
        d_hs = [0,0.2, 0.5, 1, 3]
        sigma_phi_degs = [0,2,5,15]

        for phi_deg in phi_degs:
            for num_ant in num_ants:
                for d_h in d_hs:
                    for sigma_phi_deg in sigma_phi_degs:
                        R1 = one_ring_corr_numpy(phi_deg, num_ant, d_h, sigma_phi_deg)
                        R2 = one_ring_corr_mat(phi_deg, num_ant, d_h, sigma_phi_deg, precision="double")
                        self.assertTrue(np.allclose(R1, R2))

    def test_multiple_dims(self):
        phi_degs = config.np_rng.uniform(-np.pi, np.pi, [2, 4, 3])
        num_ant = 32
        d_h = 0.7
        sigma_phi_deg = 10
        R2 = one_ring_corr_mat(phi_degs, num_ant, d_h, sigma_phi_deg, precision="double")
        R2 = tf.reshape(R2, [-1, num_ant, num_ant])
        for i, phi_deg in enumerate(np.reshape(phi_degs, [-1])):
            R1 = one_ring_corr_numpy(phi_deg, num_ant, d_h, sigma_phi_deg)
            self.assertTrue(np.allclose(R1, R2[i]))

    def test_warning_large_asd(self):
        """Should warn when sigma_phi_deg>15"""
        phi_deg = 35
        num_ant = 32
        d_h = 0.7
        sigma_phi_deg = 16
        with self.assertWarns(Warning):
            one_ring_corr_mat(phi_deg, num_ant, d_h, sigma_phi_deg)


def run_time_to_ofdm_channel_test(l_min, l_max, cyclic_prefix_length):
    """Test that the theoretical channel frequency response matches the perfectly estimated one"""
    sm = StreamManagement(np.array([[1]]), 1)
    rg = ResourceGrid(num_ofdm_symbols=5,
                  fft_size=128,
                  subcarrier_spacing=15e3,
                  pilot_pattern="kronecker",
                  pilot_ofdm_symbol_indices=[0,2,4],
                  cyclic_prefix_length=cyclic_prefix_length)
    batch_size = 16
    l_tot = l_max-l_min+1
    tdl = TDL("A", 100e-9, 3.5e9, min_speed=0, max_speed=0)
    cir = tdl(batch_size=batch_size, num_time_steps=rg.num_time_samples+l_tot-1, sampling_frequency=rg.bandwidth)
    h_time = cir_to_time_channel(rg.bandwidth, *cir, l_min=l_min, l_max=l_max, normalize=True)
    
    # Compute theoretical channel frequency response
    h_freq = time_to_ofdm_channel(h_time, rg, l_min)

    # Compute channel estimates from time-domain OFDM simulations
    qam = QAMSource(4)
    rg_mapper = ResourceGridMapper(rg)
    mod = OFDMModulator(rg.cyclic_prefix_length)
    demod = OFDMDemodulator(fft_size=rg.fft_size,
                l_min=l_min,
                cyclic_prefix_length=rg.cyclic_prefix_length)
    time_channel = ApplyTimeChannel(rg.num_time_samples, l_tot)
    chest = LSChannelEstimator(rg, interpolation_type="lin")

    x = qam([batch_size, 1, 1, rg.num_data_symbols])
    x_map = rg_mapper(x)
    x_time = mod(x_map)
    y_time = time_channel(x_time, h_time)
    y_freq = demod(y_time)
    h_freq_hat, no_var = chest(y_freq, 0.0001)

    return np.allclose(h_freq, h_freq_hat, atol=1e-6)

class TestTimeToOFDMChannel(unittest.TestCase):
    def test_different_taps(self):
        """Test that the theoretical channel frequency response
           matches the perfectly estimated one for different numbers
           of positive and negative taps
        """
        cyclic_prefix_length = 10
        for l_max in range(0, cyclic_prefix_length+1):
            for l_min in range(l_max-cyclic_prefix_length, 1):
                self.assertTrue(run_time_to_ofdm_channel_test(l_min, l_max, cyclic_prefix_length))

