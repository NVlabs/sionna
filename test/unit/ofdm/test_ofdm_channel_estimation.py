#
# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

try:
    import sionna
except ImportError as e:
    import sys
    sys.path.append("../")

from sionna.mimo import StreamManagement
from sionna.ofdm import ResourceGrid, ResourceGridMapper, LSChannelEstimator, PilotPattern, KroneckerPilotPattern
from sionna.channel.tr38901 import Antenna, AntennaArray, UMi
from sionna.channel import gen_single_sector_topology as gen_topology
from sionna.channel import subcarrier_frequencies, cir_to_ofdm_channel, cir_to_time_channel
from sionna.channel import ApplyOFDMChannel

import pytest
import unittest
import numpy as np
import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
print('Number of GPUs available :', len(gpus))
if gpus:
    gpu_num = 0 # Number of the GPU to be used
    try:
        tf.config.set_visible_devices(gpus[gpu_num], 'GPU')
        print('Only GPU number', gpu_num, 'used.')
        tf.config.experimental.set_memory_growth(gpus[gpu_num], True)
    except RuntimeError as e:
        print(e)

def freq_int(h, i, j):
    """Linear interpolation along the second axis on a 2D resource grid
    - h is [num_ofdm_symbols, num_subcarriers]
    - i, j are arrays indicating the indices of nonzero pilots
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
        if len(pilot_ind)==1:
            x_0[a] = x_1[a] = pilot_ind[0]
            y_0[a] = y_1[a] = h_int[a, pilot_ind[0]]
        elif len(pilot_ind)>1:
            x0 = 0
            x1 = 1
            for b in range(h_int.shape[1]):
                x_0[a, b] = pilot_ind[x0]
                x_1[a, b] = pilot_ind[x1]
                y_0[a, b] = h_int[a, pilot_ind[x0]]
                y_1[a, b] = h_int[a, pilot_ind[x1]]
                if b==pilot_ind[x1] and x1<len(pilot_ind)-1:
                    x0 = x1
                    x1 += 1
    h_int = (x-x_0)*np.divide(y_1-y_0, x_1-x_0, out=np.zeros_like(h), where=x_1-x_0!=0) + y_0
    return h_int

def time_int(h, time_avg=False):
    """Linear interpolation along the first axis on a 2D resource grid
    - h is [num_ofdm_symbols, num_subcarriers]
    """
    x_0 = np.zeros_like(h)
    x_1 = np.zeros_like(h)
    y_0 = np.zeros_like(h)
    y_1 = np.zeros_like(h)
    x = np.repeat(np.expand_dims(np.arange(0, h.shape[0]), 1), [h.shape[1]], axis=1)

    pilot_ind = np.where(np.sum(np.abs(h), axis=-1))[0]

    if time_avg:
        hh = np.sum(h, axis=0)/len(pilot_ind)
        h[pilot_ind] = hh

    if len(pilot_ind)==1:
        h_int = np.repeat(h[pilot_ind], [h.shape[0]], axis=0)
        return h_int
    elif len(pilot_ind)>1:
        x0 = 0
        x1 = 1
        for a in range(h.shape[0]):
            x_0[a] = pilot_ind[x0]
            x_1[a] = pilot_ind[x1]
            y_0[a] = h[pilot_ind[x0]]
            y_1[a] = h[pilot_ind[x1]]
            if a==pilot_ind[x1] and x1<len(pilot_ind)-1:
                x0 = x1
                x1 += 1
    h_int = (x-x_0)*np.divide(y_1-y_0, x_1-x_0, out=np.zeros_like(h), where=x_1-x_0!=0) + y_0
    return h_int 

def linear_int(h, i, j, time_avg=False):
    """Linear interpolation on a 2D resource grid
    - h is [num_ofdm_symbols, num_subcarriers]
    - i, j are arrays indicating the indices of nonzero pilots
    """
    h_int = freq_int(h, i, j)
    return time_int(h_int, time_avg)

def check_linear_interpolation(self, pilot_pattern, time_avg=False, mode="eager"):
    "Simulate channel estimation with linear interpolation for a 3GPP UMi channel model"
    scenario = "umi"
    carrier_frequency = 3.5e9
    direction = "uplink"
    num_ut = pilot_pattern.num_tx
    num_streams_per_tx = pilot_pattern.num_streams_per_tx
    num_ofdm_symbols = pilot_pattern.num_ofdm_symbols
    fft_size = pilot_pattern.num_effective_subcarriers
    batch_size = 1
    tf.random.set_seed(1)
    ut_array = Antenna(polarization="single",
                    polarization_type="V",
                    antenna_pattern="omni",
                    carrier_frequency=carrier_frequency)

    bs_array = AntennaArray(num_rows=1,
                            num_cols=4,
                            polarization="dual",
                            polarization_type="VH",
                            antenna_pattern="38.901",
                            carrier_frequency=carrier_frequency)

    channel_model = UMi(carrier_frequency=carrier_frequency,
                        o2i_model="low",
                        ut_array=ut_array,
                        bs_array=bs_array,
                        direction=direction,
                        enable_pathloss=False,
                        enable_shadow_fading=False)

    topology = gen_topology(batch_size, num_ut, scenario, min_ut_velocity=0, max_ut_velocity=30)
    channel_model.set_topology(*topology)

    rx_tx_association = np.zeros([1, num_ut])
    rx_tx_association[0, :] = 1
    sm = StreamManagement(rx_tx_association, num_streams_per_tx)

    rg = ResourceGrid(num_ofdm_symbols=num_ofdm_symbols,
                  fft_size=fft_size,
                  subcarrier_spacing=30e3,
                  num_tx=num_ut,
                  num_streams_per_tx=num_streams_per_tx,
                  cyclic_prefix_length=0,
                  pilot_pattern=pilot_pattern)

    frequencies = subcarrier_frequencies(rg.fft_size, rg.subcarrier_spacing)
    channel_freq = ApplyOFDMChannel(add_awgn=False)
    rg_mapper = ResourceGridMapper(rg)
    if time_avg:
        ls_est = LSChannelEstimator(rg, interpolation_type="lin_time_avg")
    else:
        ls_est = LSChannelEstimator(rg, interpolation_type="lin")

    def fun():
        x = tf.zeros([batch_size, num_ut, rg.num_streams_per_tx, rg.num_data_symbols], tf.complex64)
        x_rg = rg_mapper(x)
        a, tau = channel_model(num_time_samples=rg.num_ofdm_symbols, sampling_frequency=1/rg.ofdm_symbol_duration)
        h_freq = cir_to_ofdm_channel(frequencies, a, tau, normalize=True)
        y = channel_freq([x_rg, h_freq]) # noiseless channel
        h_hat_lin, _n = ls_est([y, 0.])
        return x_rg, h_freq, h_hat_lin

    @tf.function
    def fun_graph():
        return fun()

    @tf.function(jit_compile=True)
    def fun_xla():
        return fun()

    if mode=="eager":
        x_rg, h_freq, h_hat_lin = fun()
    elif mode=="graph":
        x_rg, h_freq, h_hat_lin = fun_graph()
    elif mode=="xla":
        x_rg, h_freq, h_hat_lin = fun_xla()

    for tx in range(0, num_ut):
        # Get non-zero pilot indices
        i, j = np.where(np.abs(x_rg[0, tx, 0]))
        h = h_freq[0,0,0,tx,0].numpy()
        h_hat_lin_numpy = linear_int(h, i, j, time_avg)
        self.assertTrue(np.allclose(h_hat_lin_numpy, h_hat_lin[0,0,0,tx,0].numpy()))

class TestLinearInterpolator(unittest.TestCase):

    def test_sparse_pilot_pattern(self):
        "One UT has two pilots, three others have just one"
        num_ut = 4
        num_streams_per_tx = 1
        num_ofdm_symbols = 14
        fft_size = 64
        mask = np.zeros([num_ut, num_streams_per_tx, num_ofdm_symbols, fft_size], bool)
        mask[...,[2,3,10,11],:] = True
        num_pilots = np.sum(mask[0,0])
        pilots = np.zeros([num_ut, num_streams_per_tx, num_pilots])
        pilots[0,0,10] = 1
        pilots[0,0,234] = 1
        pilots[1,0,20] = 1
        pilots[2,0,70] = 1
        pilots[3,0,120] = 1
        pilot_pattern = PilotPattern(mask, pilots)
        check_linear_interpolation(self, pilot_pattern, mode="eager")
        check_linear_interpolation(self, pilot_pattern, mode="graph")
        check_linear_interpolation(self, pilot_pattern, mode="xla")

    def test_kronecker_pilot_patterns_01(self):
        num_ut = 4
        num_streams_per_tx = 1
        num_ofdm_symbols = 14
        fft_size = 64
        pilot_ofdm_symbol_indices = [2, 11]
        rg = ResourceGrid(num_ofdm_symbols=num_ofdm_symbols,
                  fft_size=fft_size,
                  subcarrier_spacing=30e3,
                  num_tx=num_ut,
                  num_streams_per_tx=num_streams_per_tx,
                  cyclic_prefix_length=0,
                  pilot_pattern="kronecker",
                  pilot_ofdm_symbol_indices=pilot_ofdm_symbol_indices)
        check_linear_interpolation(self, rg.pilot_pattern, mode="eager")
        check_linear_interpolation(self, rg.pilot_pattern, mode="graph")
        check_linear_interpolation(self, rg.pilot_pattern, mode="xla")

    def test_kronecker_pilot_patterns_02(self):
        "Only a single pilot symbol"
        num_ut = 4
        num_streams_per_tx = 1
        num_ofdm_symbols = 14
        fft_size = 64
        pilot_ofdm_symbol_indices = [2]
        rg = ResourceGrid(num_ofdm_symbols=num_ofdm_symbols,
                  fft_size=fft_size,
                  subcarrier_spacing=30e3,
                  num_tx=num_ut,
                  num_streams_per_tx=num_streams_per_tx,
                  cyclic_prefix_length=0,
                  pilot_pattern="kronecker",
                  pilot_ofdm_symbol_indices=pilot_ofdm_symbol_indices)
        check_linear_interpolation(self, rg.pilot_pattern, mode="eager")
        check_linear_interpolation(self, rg.pilot_pattern, mode="graph")
        check_linear_interpolation(self, rg.pilot_pattern, mode="xla")

    def test_kronecker_pilot_patterns_03(self):
        "Only one pilot per UT"
        num_ut = 16
        num_streams_per_tx = 1
        num_ofdm_symbols = 14
        fft_size = 16
        pilot_ofdm_symbol_indices = [2]
        rg = ResourceGrid(num_ofdm_symbols=num_ofdm_symbols,
                  fft_size=fft_size,
                  subcarrier_spacing=30e3,
                  num_tx=num_ut,
                  num_streams_per_tx=num_streams_per_tx,
                  cyclic_prefix_length=0,
                  pilot_pattern="kronecker",
                  pilot_ofdm_symbol_indices=pilot_ofdm_symbol_indices)
        check_linear_interpolation(self, rg.pilot_pattern, mode="eager")
        check_linear_interpolation(self, rg.pilot_pattern, mode="graph")
        check_linear_interpolation(self, rg.pilot_pattern, mode="xla")

    def test_kronecker_pilot_patterns_04(self):
        "Multi UT, multi stream"
        num_ut = 4
        num_streams_per_tx = 2
        num_ofdm_symbols = 14
        fft_size = 64
        pilot_ofdm_symbol_indices = [2, 5, 8]
        rg = ResourceGrid(num_ofdm_symbols=num_ofdm_symbols,
                  fft_size=fft_size,
                  subcarrier_spacing=30e3,
                  num_tx=num_ut,
                  num_streams_per_tx=num_streams_per_tx,
                  cyclic_prefix_length=0,
                  pilot_pattern="kronecker",
                  pilot_ofdm_symbol_indices=pilot_ofdm_symbol_indices)
        check_linear_interpolation(self, rg.pilot_pattern, mode="eager")
        check_linear_interpolation(self, rg.pilot_pattern, mode="graph")
        check_linear_interpolation(self, rg.pilot_pattern, mode="xla")

    def test_kronecker_pilot_patterns_05(self):
        "Single UT, only pilots"
        num_ut = 1
        num_streams_per_tx = 1
        num_ofdm_symbols = 5
        fft_size = 64
        pilot_ofdm_symbol_indices = np.arange(0, num_ofdm_symbols)
        rg = ResourceGrid(num_ofdm_symbols=num_ofdm_symbols,
                  fft_size=fft_size,
                  subcarrier_spacing=30e3,
                  num_tx=num_ut,
                  num_streams_per_tx=num_streams_per_tx,
                  cyclic_prefix_length=0,
                  pilot_pattern="kronecker",
                  pilot_ofdm_symbol_indices=pilot_ofdm_symbol_indices)
        check_linear_interpolation(self, rg.pilot_pattern, mode="eager")
        check_linear_interpolation(self, rg.pilot_pattern, mode="graph")
        check_linear_interpolation(self, rg.pilot_pattern, mode="xla")

    def test_kronecker_pilot_patterns_06(self):
        num_ut = 4
        num_streams_per_tx = 1
        num_ofdm_symbols = 14
        fft_size = 64
        pilot_ofdm_symbol_indices = [2,3,8, 11]
        rg = ResourceGrid(num_ofdm_symbols=num_ofdm_symbols,
                  fft_size=fft_size,
                  subcarrier_spacing=30e3,
                  num_tx=num_ut,
                  num_streams_per_tx=num_streams_per_tx,
                  cyclic_prefix_length=0,
                  pilot_pattern="kronecker",
                  pilot_ofdm_symbol_indices=pilot_ofdm_symbol_indices)
        check_linear_interpolation(self, rg.pilot_pattern, mode="eager")
        check_linear_interpolation(self, rg.pilot_pattern, mode="graph")
        check_linear_interpolation(self, rg.pilot_pattern, mode="xla")

    def test_kronecker_pilot_patterns_with_time_averaging(self):
        num_ut = 4
        num_streams_per_tx = 1
        num_ofdm_symbols = 14
        fft_size = 64
        pilot_ofdm_symbol_indices = [2,11]
        rg = ResourceGrid(num_ofdm_symbols=num_ofdm_symbols,
                  fft_size=fft_size,
                  subcarrier_spacing=30e3,
                  num_tx=num_ut,
                  num_streams_per_tx=num_streams_per_tx,
                  cyclic_prefix_length=0,
                  pilot_pattern="kronecker",
                  pilot_ofdm_symbol_indices=pilot_ofdm_symbol_indices)
        check_linear_interpolation(self, rg.pilot_pattern, mode="eager")
        check_linear_interpolation(self, rg.pilot_pattern, mode="graph")
        check_linear_interpolation(self, rg.pilot_pattern, mode="xla")
