#
# SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import unittest
import numpy as np
import tensorflow as tf
import itertools
import sionna
from sionna.mimo import StreamManagement
from sionna.ofdm import ResourceGrid, ResourceGridMapper, LSChannelEstimator, PilotPattern, KroneckerPilotPattern, LMMSEInterpolator, tdl_freq_cov_mat, tdl_time_cov_mat
from sionna.channel.tr38901 import Antenna, AntennaArray, UMi
from sionna.channel import gen_single_sector_topology as gen_topology
from sionna.channel import subcarrier_frequencies, cir_to_ofdm_channel
from sionna.channel import ApplyOFDMChannel, exp_corr_mat
from sionna.utils import QAMSource,ebnodb2no
from sionna.mapping import Mapper
from sionna.channel.tr38901 import TDL


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

#######################################################
# Test LMMSE interpolation
#######################################################

class TestLMMSEInterpolator(unittest.TestCase):

    # Batch size for the tests
    BATCH_SIZE = 1

    # SNR values for which tests are run
    EBN0DBs = [0.0]

    # Allowed absolute error
    # Single precision and XLA
    ATOL_LOW_PREC = 1e-3
    # Double precision without XLA
    ATOL_HIGH_PREC = 1e-10

    ########################################
    # Reference implementation
    ########################################

    def pilot_pattern_2_pilot_mask(self, pilot_pattern):
        # pilot_pattern : PilotPattern
        #    An instance of PilotPattern

        data_mask = pilot_pattern.mask
        pilots = pilot_pattern.pilots

        num_tx = data_mask.shape[0]
        num_steams_per_tx = data_mask.shape[1]
        num_ofdm_symbols = data_mask.shape[2]
        num_effective_subcarriers = data_mask.shape[3]
        pilot_mask = np.zeros([num_tx,num_steams_per_tx,num_ofdm_symbols,num_effective_subcarriers], bool)
        for tx in range(num_tx):
            for st in range(num_steams_per_tx):
                pil_ind = 0 # Pilot index for this stream
                for sb in range(num_ofdm_symbols):
                    for sc in range(num_effective_subcarriers):
                        if data_mask[tx,st,sb,sc]:
                            if np.abs(pilots[tx,st,pil_ind]) > 0.:
                                pilot_mask[tx,st,sb,sc] = True
                            pil_ind += 1
        return pilot_mask

    def map_estimates_to_rg(self, h_hat, err_var, pilot_pattern):
        # h_hat : [batch_size, num_tx, num_streams_per_tx, num_rx, num_rx_ant, num_pilots]
        #     Channel estimates
        #
        # err_var : [batch_size, num_tx, num_streams_per_tx, num_rx, num_rx_ant, num_pilots]
        #     Channel estimation error variances
        #
        # pilot_pattern : PilotPattern
        #    An instance of PilotPattern

        data_mask = pilot_pattern.mask
        pilots = pilot_pattern.pilots

        batch_size = h_hat.shape[0]
        num_rx = h_hat.shape[1]
        num_rx_ant = h_hat.shape[2]
        num_tx = h_hat.shape[3]
        num_steams_per_tx = h_hat.shape[4]
        num_ofdm_symbols = data_mask.shape[2]
        num_effective_subcarriers = data_mask.shape[3]
        h_hat_rg = np.zeros([batch_size,num_rx,num_rx_ant,num_tx,num_steams_per_tx,num_ofdm_symbols,num_effective_subcarriers], complex)
        err_var_rg = np.zeros([batch_size,num_rx,num_rx_ant,num_tx,num_steams_per_tx,num_ofdm_symbols,num_effective_subcarriers], float)
        for bs in range(batch_size):
            for rx in range(num_rx):
                for ra in range(num_rx_ant):
                    for tx in range(num_tx):
                        for st in range(num_steams_per_tx):
                            pil_ind = 0 # Pilot index for this stream
                            for sb in range(num_ofdm_symbols):
                                for sc in range(num_effective_subcarriers):
                                    if data_mask[tx,st,sb,sc]:
                                        if np.abs(pilots[tx,st,pil_ind]) > 0.:
                                            h_hat_rg[bs,rx,ra,tx,st,sb,sc] = h_hat[bs,rx,ra,tx,st,pil_ind]
                                            err_var_rg[bs,rx,ra,tx,st,sb,sc] = err_var[bs,rx,ra,tx,st,pil_ind]
                                        pil_ind += 1
        return h_hat_rg,err_var_rg

    def reference_lmmse_interpolation_1d_one_axis(self, cov_mat, h_hat, err_var, pattern, last_step):

        # cov_mat : [dim_size, dim_size]
        #  Covariance matrix
        #
        # h_hat : [dim_size]
        #  Channel estimate at pilot locations. Zeros elsewhere.
        #
        # err_var : [dim_size]
        #  Channel estimation error variance at pilot locations. Zero elsewhere.
        #
        # pattern : [dim_size]
        #  Mask indicating where a channel estimate is available.
        #
        # last_step : bool
        #  If `False`, this is not the last step, and an additional scaling is done
        #   to prepare for the next interpolation/smoothing.

        err_var_old = err_var

        #
        # Build interpolation matrix
        #
        dim_size = pattern.shape[0]
        pil_ind, = np.where(pattern)
        num_pil = pil_ind.shape[0]

        pi_mat = np.zeros([dim_size, num_pil])
        k = 0
        for i in range(dim_size):
            if pattern[i]:
                pi_mat[i,k] = 1.0
                k += 1

        int_mat = np.matmul(np.matmul(pi_mat.T, cov_mat), pi_mat)
        err_var = np.take(err_var, pil_ind, axis=0)
        int_mat = int_mat + np.diag(err_var)
        int_mat = np.linalg.inv(int_mat)
        int_mat = np.matmul(pi_mat, np.matmul(int_mat, pi_mat.T))
        int_mat = np.matmul(cov_mat, int_mat)

        #
        # Interpolation
        #
        h_hat = np.matmul(int_mat, h_hat)

        #
        # Error variance
        #
        mask_mat = np.zeros([dim_size, dim_size])
        for i in range(dim_size):
            if pattern[i]:
                mask_mat[i,i] = 1.0
        err_var = cov_mat - np.matmul(int_mat, np.matmul(mask_mat, cov_mat))
        err_var = np.diag(err_var).real

        #
        # Scaling if not last step
        #
        if not last_step:
            # Estimate covariance
            int_mat_h = np.conj(int_mat.T)
            h_hat_var = np.matmul(int_mat, np.matmul(cov_mat+np.diag(err_var_old), int_mat_h))
            h_hat_var = np.diag(h_hat_var).real
            # Scaling
            s = 2./(1.+h_hat_var-err_var)
            h_hat = s*h_hat
            # Update error variance
            err_var = s*(s-1)*h_hat_var + (1.-s) + s*err_var

        return h_hat, err_var

    def reference_spatial_smoothing_one_re(self, cov_mat, h_hat, err_var, last_step):

        # cov_mat : [num_rx_ant, num_rx_ant]
        #  Covariance matrix
        #
        # h_hat : [num_rx_ant]
        #  Channel estimate at pilot locations. Zeros elsewhere.
        #
        # err_var : [num_rx_ant]
        #  Channel estimation error variance at pilot locations. Zero elsewhere.
        #
        # last_step : bool
        #  If `False`, this is not the last step, and an additional scaling is done
        #   to prepare for the next interpolation/smoothing.

        A = cov_mat + np.diag(err_var)
        A = np.linalg.inv(A)
        A = np.matmul(cov_mat,A)

        h_hat = np.expand_dims(h_hat, axis=-1)
        h_hat = np.matmul(A, h_hat)
        h_hat = np.squeeze(h_hat, axis=-1)

        err_var_out = cov_mat - np.matmul(A,cov_mat)
        err_var_out = np.diag(err_var_out).real

        if not last_step:
            # Estimate covariance
            Ah = np.conj(A.T)
            h_hat_var = np.matmul(A, np.matmul(cov_mat+np.diag(err_var), Ah))
            h_hat_var = np.diag(h_hat_var).real
            # Scaling
            s = 2./(1.+h_hat_var-err_var_out)
            h_hat = s*h_hat
            # Update error variance
            err_var_out = s*(s-1)*h_hat_var + (1.-s) + s*err_var_out

        return h_hat, err_var_out

    def reference_spatial_smoothing(self, cov_mat, h_hat, err_var, last_step):

        # cov_mat : [num_rx_ant, num_rx_ant]
        #  Covariance matrix
        #
        # h_hat : [batch_size, num_rx, num_rx_ant, num_tx, num_tx_streams, num_ofdm_symbols, num_effectve_subcarriers]
        #  Channel estimate at pilot locations. Zeros elsewhere.
        #
        # err_var : [batch_size, num_rx, num_rx_ant, num_tx, num_tx_streams, num_ofdm_symbols, num_effectve_subcarriers]
        #  Channel estimation error variance at pilot locations. Zero elsewhere.
        #
        # last_step : bool
        #  If `False`, this is not the last step, and an additional scaling is done
        #   to prepare for the next interpolation/smoothing.

        # [batch_size, num_rx, num_tx, num_tx_streams, num_ofdm_symbols, num_effectve_subcarriers, num_rx_ant]
        h_hat = np.transpose(h_hat, [0, 1, 3, 4, 5, 6, 2])
        err_var = np.transpose(err_var, [0, 1, 3, 4, 5, 6, 2])

        h_hat_shape = h_hat.shape
        num_rx_ant = h_hat.shape[-1]
        h_hat = np.reshape(h_hat, [-1, num_rx_ant])
        err_var = np.reshape(err_var, [-1, num_rx_ant])

        i = 0
        for h_hat_, err_var_ in zip(h_hat, err_var):
            h_hat_new, err_var_new = self.reference_spatial_smoothing_one_re(cov_mat, h_hat_, err_var_, last_step)
            h_hat[i] = h_hat_new
            err_var[i] = err_var_new
            i = i + 1

        h_hat = np.reshape(h_hat, h_hat_shape)
        err_var = np.reshape(err_var, h_hat_shape)

        # [batch_size, num_rx, num_rx_ant, num_tx, num_tx_streams, num_ofdm_symbols, num_effectve_subcarriers]
        h_hat = np.transpose(h_hat, [0, 1, 6, 2, 3, 4, 5])
        err_var = np.transpose(err_var, [0, 1, 6, 2, 3, 4, 5])

        return h_hat, err_var

    def reference_lmmse_interpolation_1d(self, cov_mat, h_hat, err_var, pattern, last_step):

        # cov_mat : [dim_size, dim_size]
        #  Covariance matrix
        #
        # h_hat : [batch_size, num_rx, num_rx_ant, num_tx, num_tx_streams, outer_dim_size, inner_dim_size]
        #  Channel estimate at pilot locations. Zeros elsewhere.
        #
        # err_var : [batch_size, num_rx, num_rx_ant, num_tx, num_tx_streams, outer_dim_size, inner_dim_size]
        #  Channel estimation error variance at pilot locations. Zero elsewhere.
        #
        # pattern : [num_tx, num_tx_streams, outer_dim_size, inner_dim_size]
        #  Mask indicating where a channel estimate is available.
        #
        # last_step : bool
        #  If `False`, this is not the last step, and an additional scaling is done
        #   to prepare for the next interpolation/smoothing.

        batch_size = h_hat.shape[0]
        num_rx = h_hat.shape[1]
        num_rx_ant = h_hat.shape[2]
        num_tx = h_hat.shape[3]
        num_tx_streams = h_hat.shape[4]
        outer_dim_size = h_hat.shape[5]
        inner_dim_size = h_hat.shape[6]

        for b,rx,ra,tx,ts,od in itertools.product(range(batch_size),
                                                range(num_rx),
                                                range(num_rx_ant),
                                                range(num_tx),
                                                range(num_tx_streams),
                                                range(outer_dim_size)):
            h_hat_ = h_hat[b,rx,ra,tx,ts,od]
            err_var_ = err_var[b,rx,ra,tx,ts,od]
            pattern_ = pattern[tx,ts,od]
            if np.any(pattern_):
                h_hat_, err_var_ = self.reference_lmmse_interpolation_1d_one_axis(cov_mat, h_hat_, err_var_, pattern_, last_step)
                h_hat[b,rx,ra,tx,ts,od] = h_hat_
                err_var[b,rx,ra,tx,ts,od] = err_var_

        # Updating the pattern
        pattern_update_mask = np.any(pattern, axis=-1, keepdims=True)
        pattern = np.logical_or(pattern, pattern_update_mask)

        return h_hat, err_var, pattern

    def reference_lmmse_interpolation(self, cov_mat_time, cov_mat_freq, cov_mat_space, h_hat, err_var, pattern, order):

        # cov_mat_time : [num_ofdm_symbols, num_ofdm_symbols]
        #  Time covariance matrix
        #
        # cov_mat_freq : [num_effectve_subcarriers, num_effectve_subcarriers]
        #  Frequency covariance matrix
        #
        # h_hat : [batch_size, num_rx, num_rx_ant, num_tx, num_tx_streams, num_ofdm_symbols, num_effectve_subcarriers]
        #  Channel estimate at pilot locations. Zeros elsewhere.
        #
        # err_var : [batch_size, num_rx, num_rx_ant, num_tx, num_tx_streams, num_ofdm_symbols, num_effectve_subcarriers]
        #  Channel estimation error variance at pilot locations. Zero elsewhere.
        #
        # pattern : PilotPattern
        #  A Sionna pilot pattern
        #
        # order : 'freq_first' or 'time_first'
        #  Order in which to do the 1D interpolation

        pilot_mask = self.pilot_pattern_2_pilot_mask(pattern)
        h_hat,err_var = self.map_estimates_to_rg(h_hat, err_var, pattern)

        if order == 'f-t':
            h_hat, err_var, pilot_mask = self.reference_lmmse_interpolation_1d(cov_mat_freq, h_hat, err_var, pilot_mask, False)
            h_hat = np.transpose(h_hat, [0, 1, 2, 3, 4, 6, 5])
            err_var = np.transpose(err_var, [0, 1, 2, 3, 4, 6, 5])
            pilot_mask = np.transpose(pilot_mask, [0, 1, 3, 2])
            h_hat, err_var,_ = self.reference_lmmse_interpolation_1d(cov_mat_time, h_hat, err_var, pilot_mask, True)
            h_hat = np.transpose(h_hat, [0, 1, 2, 3, 4, 6, 5])
            err_var = np.transpose(err_var, [0, 1, 2, 3, 4, 6, 5])
        elif order == 't-f':
            h_hat = np.transpose(h_hat, [0, 1, 2, 3, 4, 6, 5])
            err_var = np.transpose(err_var, [0, 1, 2, 3, 4, 6, 5])
            pilot_mask = np.transpose(pilot_mask, [0, 1, 3, 2])
            h_hat, err_var,pilot_mask = self.reference_lmmse_interpolation_1d(cov_mat_time, h_hat, err_var, pilot_mask, False)
            h_hat = np.transpose(h_hat, [0, 1, 2, 3, 4, 6, 5])
            err_var = np.transpose(err_var, [0, 1, 2, 3, 4, 6, 5])
            pilot_mask = np.transpose(pilot_mask, [0, 1, 3, 2])
            h_hat, err_var,_ = self.reference_lmmse_interpolation_1d(cov_mat_freq, h_hat, err_var, pilot_mask, True)
        elif order == 't-s-f':
            h_hat = np.transpose(h_hat, [0, 1, 2, 3, 4, 6, 5])
            err_var = np.transpose(err_var, [0, 1, 2, 3, 4, 6, 5])
            pilot_mask = np.transpose(pilot_mask, [0, 1, 3, 2])
            h_hat, err_var,pilot_mask = self.reference_lmmse_interpolation_1d(cov_mat_time, h_hat, err_var, pilot_mask, False)
            h_hat = np.transpose(h_hat, [0, 1, 2, 3, 4, 6, 5])
            err_var = np.transpose(err_var, [0, 1, 2, 3, 4, 6, 5])
            pilot_mask = np.transpose(pilot_mask, [0, 1, 3, 2])
            h_hat, err_var = self.reference_spatial_smoothing(cov_mat_space, h_hat, err_var, False)
            h_hat, err_var,_ = self.reference_lmmse_interpolation_1d(cov_mat_freq, h_hat, err_var, pilot_mask, True)

        return h_hat, err_var

    ##########################################
    # Tests
    ##########################################

    # Run an E2E link with reference and Sionna LMMSE interpolation and compute
    # the maximums error for both the estimate and error variance and both
    # time first and frequency first interpolation
    def run_e2e_link(self, batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx,
        num_ofdm_symbols, fft_size, pilot_pattern, ebno_db, exec_mode, dtype):

        assert exec_mode in ('eager', 'graph', 'xla'), "Wrong execution mode"

        tdl_model = 'A'
        subcarrier_spacing = 30e3 # Hz
        num_bits_per_symbol = 2
        delay_spread = 300e-9 # s
        carrier_frequency = 3.5e9 # Hz
        speed = 5. # m/s

        sm = StreamManagement(np.ones([num_rx, num_tx]), num_streams_per_tx)
        rg = ResourceGrid(num_ofdm_symbols=num_ofdm_symbols,
                        fft_size=fft_size,
                        subcarrier_spacing=subcarrier_spacing,
                        num_tx=num_tx,
                        num_streams_per_tx=num_streams_per_tx,
                        cyclic_prefix_length=0,
                        pilot_pattern=pilot_pattern,
                        dtype=dtype)

        # Transmitter
        qam_source = QAMSource(num_bits_per_symbol, dtype=dtype)
        mapper = Mapper("qam", num_bits_per_symbol, dtype=dtype)
        rg_mapper = ResourceGridMapper(rg, dtype=dtype)

        # OFDM CHannel
        los_angle_of_arrival=np.pi/4.
        channel_model = TDL(tdl_model, delay_spread, carrier_frequency, min_speed=speed, max_speed=speed,
                            los_angle_of_arrival=los_angle_of_arrival, dtype=dtype)
        channel_freq = ApplyOFDMChannel(add_awgn=True, dtype=dtype)
        frequencies = subcarrier_frequencies(fft_size, subcarrier_spacing, dtype=dtype)

        # The LS channel estimator will provide channel estimates and error variances
        cov_mat_freq = tdl_freq_cov_mat(tdl_model, subcarrier_spacing, fft_size, delay_spread, dtype)
        cov_mat_time = tdl_time_cov_mat(tdl_model, speed, carrier_frequency, rg.ofdm_symbol_duration,
                                        num_ofdm_symbols, los_angle_of_arrival, dtype)
        cov_mat_space = exp_corr_mat(0.9, num_rx_ant, dtype)
        lmmse_inter_ft = LMMSEInterpolator(pilot_pattern, cov_mat_time, cov_mat_freq, order="f-t")
        ls_est_lmmse_ft = LSChannelEstimator(rg, interpolator=lmmse_inter_ft, dtype=dtype)
        lmmse_inter_tf = LMMSEInterpolator(pilot_pattern, cov_mat_time, cov_mat_freq, order="t-f")
        ls_est_lmmse_tf = LSChannelEstimator(rg, interpolator=lmmse_inter_tf, dtype=dtype)
        lmmse_inter_tsf = LMMSEInterpolator(pilot_pattern, cov_mat_time, cov_mat_freq, cov_mat_space, order="t-s-f")
        ls_est_lmmse_tsf = LSChannelEstimator(rg, interpolator=lmmse_inter_tsf, dtype=dtype)

        # For computing the reference interpolation
        ls_no_interp = LSChannelEstimator(rg, interpolation_type=None, dtype=dtype)

        def _run():
            no = ebnodb2no(ebno_db, num_bits_per_symbol, coderate=1.0)
            x = qam_source([batch_size, num_tx, num_streams_per_tx, rg.num_data_symbols])
            x_rg = rg_mapper(x)

            a, tau = channel_model(batch_size, num_ofdm_symbols, sampling_frequency=1./rg.ofdm_symbol_duration)
            h_freq = cir_to_ofdm_channel(frequencies, a, tau, normalize=True)
            y = channel_freq([x_rg, h_freq, no])

            h_hat_lmmse_ft,err_var_lmmse_ft = ls_est_lmmse_ft([y, no])
            h_hat_lmmse_tf,err_var_lmmse_tf = ls_est_lmmse_tf([y, no])
            h_hat_lmmse_tsf,err_var_lmmse_tsf = ls_est_lmmse_tsf([y, no])
            h_hat_no_int, err_var_no_int = ls_no_interp([y, no])

            return h_hat_no_int, err_var_no_int, h_hat_lmmse_ft, err_var_lmmse_ft, h_hat_lmmse_tf, err_var_lmmse_tf, h_hat_lmmse_tsf, err_var_lmmse_tsf, h_freq

        if exec_mode == 'eager':
            _run_compiled = _run
        elif exec_mode == 'graph':
            _run_compiled = tf.function(_run)
        elif exec_mode == 'xla':
            _run_compiled = tf.function(_run, jit_compile=True)

        run_output = _run_compiled()
        h_hat_no_int = run_output[0].numpy()
        err_var_no_int = run_output[1].numpy()
        err_var_no_int = np.broadcast_to(err_var_no_int, h_hat_no_int.shape)
        h_hat_lmmse_ft = run_output[2].numpy()
        err_var_lmmse_ft = run_output[3].numpy()
        h_hat_lmmse_tf = run_output[4].numpy()
        err_var_lmmse_tf = run_output[5].numpy()
        h_hat_lmmse_tsf = run_output[6].numpy()
        err_var_lmmse_tsf = run_output[7].numpy()
        h_freq = run_output[8].numpy()

        # Reference estimate
        h_hat_lmmse_ft_ref, err_var_lmmse_ft_ref = self.reference_lmmse_interpolation(cov_mat_time.numpy(),
                                                                                        cov_mat_freq.numpy(),
                                                                                        cov_mat_space.numpy(),
                                                                                        h_hat_no_int, err_var_no_int,
                                                                                        pilot_pattern, "f-t")
        h_hat_lmmse_tf_ref, err_var_lmmse_tf_ref = self.reference_lmmse_interpolation(cov_mat_time.numpy(),
                                                                                        cov_mat_freq.numpy(),
                                                                                        cov_mat_space.numpy(),
                                                                                        h_hat_no_int, err_var_no_int,
                                                                                        pilot_pattern, "t-f")
        h_hat_lmmse_tsf_ref, err_var_lmmse_tsf_ref = self.reference_lmmse_interpolation(cov_mat_time.numpy(),
                                                                                        cov_mat_freq.numpy(),
                                                                                        cov_mat_space.numpy(),
                                                                                        h_hat_no_int, err_var_no_int,
                                                                                        pilot_pattern, "t-s-f")

        # Compute errors
        max_err_h_hat_ft = np.max(np.abs(h_hat_lmmse_ft_ref-h_hat_lmmse_ft))
        max_err_err_var_lmmse_ft = np.max(np.abs(err_var_lmmse_ft_ref-err_var_lmmse_ft))
        max_err_h_hat_tf = np.max(np.abs(h_hat_lmmse_tf_ref-h_hat_lmmse_tf))
        max_err_err_var_lmmse_tf = np.max(np.abs(err_var_lmmse_tf_ref-err_var_lmmse_tf))
        max_err_h_hat_tsf = np.max(np.abs(h_hat_lmmse_tsf_ref-h_hat_lmmse_tsf))
        max_err_err_var_lmmse_tsf = np.max(np.abs(err_var_lmmse_tsf_ref-err_var_lmmse_tsf))

        return max_err_h_hat_ft,max_err_err_var_lmmse_ft,max_err_h_hat_tf,max_err_err_var_lmmse_tf,max_err_h_hat_tsf,max_err_err_var_lmmse_tsf

    def run_test(self, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols,
                    fft_size, mask, pilots):

        def _test(num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols,
                    fft_size, pilot_pattern, ebno_db, exec_mode, dtype):
            if exec_mode == 'xla':
                sionna.Config.xla_compat = True
            outputs = self.run_e2e_link(TestLMMSEInterpolator.BATCH_SIZE, num_rx, num_rx_ant, num_tx,
                num_streams_per_tx, num_ofdm_symbols, fft_size, pilot_pattern, ebno_db, exec_mode, dtype)
            if exec_mode == 'xla':
                sionna.Config.xla_compat = False

            if dtype == tf.complex64 or exec_mode == "xla":
                atol = TestLMMSEInterpolator.ATOL_LOW_PREC
            else:
                atol = TestLMMSEInterpolator.ATOL_HIGH_PREC

            max_err_h_hat_ft = outputs[0]
            self.assertTrue(np.allclose(max_err_h_hat_ft, 0.0, atol=atol))

            max_err_err_var_lmmse_ft = outputs[1]
            self.assertTrue(np.allclose(max_err_err_var_lmmse_ft, 0.0, atol=atol))

            max_err_h_hat_tf = outputs[2]
            self.assertTrue(np.allclose(max_err_h_hat_tf, 0.0, atol=atol))

            max_err_err_var_lmmse_tf = outputs[3]
            self.assertTrue(np.allclose(max_err_err_var_lmmse_tf, 0.0, atol=atol))

            max_err_h_hat_tsf = outputs[4]
            self.assertTrue(np.allclose(max_err_h_hat_tsf, 0.0, atol=atol))

            max_err_err_var_lmmse_tsf = outputs[5]
            self.assertTrue(np.allclose(max_err_err_var_lmmse_tsf, 0.0, atol=atol))

        for ebno_db in TestLMMSEInterpolator.EBN0DBs:
            # 32bit precision
            pilot_pattern = PilotPattern(mask, pilots, dtype=tf.complex64)
            ebno_db_sp = tf.cast(ebno_db, tf.float32)
            _test(num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols,
                    fft_size, pilot_pattern, ebno_db_sp, "eager", tf.complex64)
            _test(num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols,
                    fft_size, pilot_pattern, ebno_db_sp, "graph", tf.complex64)
            # XLA is not supported
            # _test(num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols,
            #         fft_size, pilot_pattern, ebno_db_sp, "xla", tf.complex64)
            # 64bit precision
            pilot_pattern = PilotPattern(mask, pilots, dtype=tf.complex128)
            ebno_db_dp = tf.cast(ebno_db, tf.float64)
            _test(num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols,
                    fft_size, pilot_pattern, ebno_db_dp, "eager", tf.complex128)
            _test(num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols,
                    fft_size, pilot_pattern, ebno_db_dp, "graph", tf.complex128)
            # XLA is not supported
            # _test(num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols,
            #         fft_size, pilot_pattern, ebno_db_dp, "xla", tf.complex128)

    def test_sparse_pilot_pattern(self):
        "One UT has two pilots, three others have just one"
        num_tx = 4
        num_streams_per_tx = 1
        num_ofdm_symbols = 14
        fft_size = 12
        mask = np.zeros([num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size], bool)
        mask[...,5,:] = True
        num_pilots = np.sum(mask[0,0])
        pilots = np.zeros([num_tx, num_streams_per_tx, num_pilots])
        pilots[0,0,[0,11]] = 1
        pilots[1,0,1] = 1
        pilots[2,0,5] = 1
        pilots[3,0,10] = 1
        self.run_test(1, 1, num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size, mask, pilots)

    def test_kronecker_pilot_patterns_01(self):
        num_tx = 1
        num_streams_per_tx = 1
        num_ofdm_symbols = 14
        fft_size = 64
        pilot_ofdm_symbol_indices = [2, 11]
        rg = ResourceGrid(num_ofdm_symbols=num_ofdm_symbols,
                  fft_size=fft_size,
                  subcarrier_spacing=30e3,
                  num_tx=num_tx,
                  num_streams_per_tx=num_streams_per_tx,
                  cyclic_prefix_length=0,
                  pilot_pattern="kronecker",
                  pilot_ofdm_symbol_indices=pilot_ofdm_symbol_indices)
        pilot_pattern = rg.pilot_pattern
        pilot_pattern = KroneckerPilotPattern(rg, pilot_ofdm_symbol_indices)
        self.run_test(1, 1, num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size,
                        pilot_pattern.mask, pilot_pattern.pilots)

    def test_kronecker_pilot_patterns_02(self):
        "Only a single pilot symbol"
        num_tx = 4
        num_streams_per_tx = 1
        num_ofdm_symbols = 14
        fft_size = 64
        pilot_ofdm_symbol_indices = [2]
        rg = ResourceGrid(num_ofdm_symbols=num_ofdm_symbols,
                  fft_size=fft_size,
                  subcarrier_spacing=30e3,
                  num_tx=num_tx,
                  num_streams_per_tx=num_streams_per_tx,
                  cyclic_prefix_length=0,
                  pilot_pattern="kronecker",
                  pilot_ofdm_symbol_indices=pilot_ofdm_symbol_indices)
        pilot_pattern = rg.pilot_pattern
        self.run_test(1, 1, num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size,
                        pilot_pattern.mask, pilot_pattern.pilots)

    def test_kronecker_pilot_patterns_03(self):
        "Only one pilot per UT"
        num_tx = 16
        num_streams_per_tx = 1
        num_ofdm_symbols = 14
        fft_size = 16
        pilot_ofdm_symbol_indices = [2]
        rg = ResourceGrid(num_ofdm_symbols=num_ofdm_symbols,
                  fft_size=fft_size,
                  subcarrier_spacing=30e3,
                  num_tx=num_tx,
                  num_streams_per_tx=num_streams_per_tx,
                  cyclic_prefix_length=0,
                  pilot_pattern="kronecker",
                  pilot_ofdm_symbol_indices=pilot_ofdm_symbol_indices)
        pilot_pattern = rg.pilot_pattern
        self.run_test(1, 1, num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size,
                        pilot_pattern.mask, pilot_pattern.pilots)

    def test_kronecker_pilot_patterns_04(self):
        "Multi UT, multi stream"
        num_tx = 4
        num_streams_per_tx = 2
        num_ofdm_symbols = 14
        fft_size = 64
        pilot_ofdm_symbol_indices = [2, 5, 8]
        rg = ResourceGrid(num_ofdm_symbols=num_ofdm_symbols,
                  fft_size=fft_size,
                  subcarrier_spacing=30e3,
                  num_tx=num_tx,
                  num_streams_per_tx=num_streams_per_tx,
                  cyclic_prefix_length=0,
                  pilot_pattern="kronecker",
                  pilot_ofdm_symbol_indices=pilot_ofdm_symbol_indices)
        pilot_pattern = rg.pilot_pattern
        self.run_test(1, 1, num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size,
                        pilot_pattern.mask, pilot_pattern.pilots)

    def test_kronecker_pilot_patterns_05(self):
        "Single UT, only pilots"
        num_tx = 1
        num_streams_per_tx = 1
        num_ofdm_symbols = 5
        fft_size = 64
        pilot_ofdm_symbol_indices = np.arange(0, num_ofdm_symbols)
        rg = ResourceGrid(num_ofdm_symbols=num_ofdm_symbols,
                  fft_size=fft_size,
                  subcarrier_spacing=30e3,
                  num_tx=num_tx,
                  num_streams_per_tx=num_streams_per_tx,
                  cyclic_prefix_length=0,
                  pilot_pattern="kronecker",
                  pilot_ofdm_symbol_indices=pilot_ofdm_symbol_indices)
        pilot_pattern = rg.pilot_pattern
        self.run_test(1, 1, num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size,
                        pilot_pattern.mask, pilot_pattern.pilots)

    def test_kronecker_pilot_patterns_06(self):
        num_tx = 4
        num_streams_per_tx = 1
        num_ofdm_symbols = 14
        fft_size = 64
        pilot_ofdm_symbol_indices = [2,3,8, 11]
        rg = ResourceGrid(num_ofdm_symbols=num_ofdm_symbols,
                  fft_size=fft_size,
                  subcarrier_spacing=30e3,
                  num_tx=num_tx,
                  num_streams_per_tx=num_streams_per_tx,
                  cyclic_prefix_length=0,
                  pilot_pattern="kronecker",
                  pilot_ofdm_symbol_indices=pilot_ofdm_symbol_indices)
        pilot_pattern = rg.pilot_pattern
        self.run_test(1, 1, num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size,
                        pilot_pattern.mask, pilot_pattern.pilots)

    def test_order_error(self):

        tdl_model = 'A'
        subcarrier_spacing = 30e3 # Hz
        num_bits_per_symbol = 2
        delay_spread = 300e-9 # s
        carrier_frequency = 3.5e9 # Hz
        speed = 5. # m/s
        los_angle_of_arrival=np.pi/4.
        fft_size = 12
        num_rx_ant = 16
        num_tx = 4
        num_streams_per_tx = 1
        num_ofdm_symbols = 14
        pilot_ofdm_symbol_indices = [2,3,8, 11]
        rg = ResourceGrid(num_ofdm_symbols=num_ofdm_symbols,
                  fft_size=fft_size,
                  subcarrier_spacing=subcarrier_spacing,
                  num_tx=num_tx,
                  num_streams_per_tx=num_streams_per_tx,
                  cyclic_prefix_length=0,
                  pilot_pattern="kronecker",
                  pilot_ofdm_symbol_indices=pilot_ofdm_symbol_indices)
        pilot_pattern = rg.pilot_pattern
        cov_mat_freq = tdl_freq_cov_mat(tdl_model, subcarrier_spacing, fft_size, delay_spread)
        cov_mat_time = tdl_time_cov_mat(tdl_model, speed, carrier_frequency, rg.ofdm_symbol_duration,
                                        num_ofdm_symbols, los_angle_of_arrival)
        cov_mat_space = exp_corr_mat(0.9, num_rx_ant)

        # Testing random input order
        with self.assertRaises(AssertionError):
            lmmse_inter_ft = LMMSEInterpolator(pilot_pattern, cov_mat_time, cov_mat_freq, order="hello")

        # Test multiple --
        with self.assertRaises(AssertionError):
            lmmse_inter_ft = LMMSEInterpolator(pilot_pattern, cov_mat_time, cov_mat_freq, order="f--t")

        # Test multiple s,f, or t
        with self.assertRaises(AssertionError):
            lmmse_inter_ft = LMMSEInterpolator(pilot_pattern, cov_mat_time, cov_mat_freq, order="f-f-t")
            lmmse_inter_ft = LMMSEInterpolator(pilot_pattern, cov_mat_time, cov_mat_freq, order="f-t-t")
            lmmse_inter_ft = LMMSEInterpolator(pilot_pattern, cov_mat_time, cov_mat_freq, cov_mat_space, order="f-s-s-t")

        # Test multiple s,f, or t
        with self.assertRaises(AssertionError):
            lmmse_inter_ft = LMMSEInterpolator(pilot_pattern, cov_mat_time, cov_mat_freq, order="f-f-t")
            lmmse_inter_ft = LMMSEInterpolator(pilot_pattern, cov_mat_time, cov_mat_freq, order="f-t-t")
            lmmse_inter_ft = LMMSEInterpolator(pilot_pattern, cov_mat_time, cov_mat_freq, cov_mat_space, order="f-s-s-t")

        # Test no t or no f
        with self.assertRaises(AssertionError):
            lmmse_inter_ft = LMMSEInterpolator(pilot_pattern, cov_mat_time, cov_mat_freq, cov_mat_space, order="f-s")
            lmmse_inter_ft = LMMSEInterpolator(pilot_pattern, cov_mat_time, cov_mat_freq, cov_mat_space, order="s-t")

        # Test s but no spatial covariance matrix
        with self.assertRaises(AssertionError):
            lmmse_inter_ft = LMMSEInterpolator(pilot_pattern, cov_mat_time, cov_mat_freq, order="f-t-s")

#######################################################
# Test utilities
#######################################################

# class TestUtilities(unittest.TestCase):

#     # Batch size for sampling channel models
#     BATCH_SIZE = 1000

#     # Num samples for every monte carlo estimate
#     NUM_SAMPLES = 1000000

#     # Tested subcarrier spacings
#     SUBCARRIER_SPACING = (15e3, 30e3, 120e3) # Hz

#     # Tested delay spreads
#     DELAY_SPREAD = (100e-9, 300e-9, 1000e-9) # s

#     # Tested FFT sizes
#     FFT_SIZE = 1024

#     # TDL models
#     TDL_MODELS = ('A', 'B', 'C', 'D', 'E')

#     # Tested speeds
#     SPEEDS = (0.0, 10.0, 100.)

#     # Tested number of OFDM symbols
#     NUM_SYMBOLS = 140

#     # Tested carrier frequencies
#     CARRIER_FREQS = (0.450e6, 3.5e9, 6.0e9)

#     # Absolute error tolerance
#     ATOL = 1e-2

#     def est_tdl_freq_cov_mat(self, num_samples, model, delay_spread, carrier_frequency,
#         subcarrier_spacing, ofdm_symbol_duration, fft_size):


#         channel_model = TDL(model, delay_spread, carrier_frequency)
#         frequencies = subcarrier_frequencies(fft_size, subcarrier_spacing)

#         batch_size = TestUtilities.BATCH_SIZE
#         num_it = (num_samples//batch_size) + 1
#         hs = []

#         @tf.function(jit_compile=True)
#         def _run():
#             cov_mat = tf.zeros([fft_size, fft_size], tf.complex64)
#             for _ in tf.range(num_it):
#                 a, tau = channel_model(batch_size, 1,
#                                         sampling_frequency=1./ofdm_symbol_duration)
#                 h = cir_to_ofdm_channel(frequencies, a, tau)[:,0,0,0,0] # [batch size, 1, fft size]
#                 h = tf.transpose(h, [0,2,1]) # [batch size, fft size, 1]
#                 cov_mat_ = tf.matmul(h, h, adjoint_b=True)
#                 cov_mat_ = tf.reduce_mean(cov_mat_, axis=0)
#                 cov_mat += cov_mat_
#             cov_mat = cov_mat / tf.cast(num_it,tf.complex64)
#             return cov_mat

#         cov_mat = _run().numpy()
#         return cov_mat

#     def test_tdl_freq_cov_mat(self):

#         fft_size = TestUtilities.FFT_SIZE

#         parameters =  itertools.product(TestUtilities.TDL_MODELS,
#                                         TestUtilities.SUBCARRIER_SPACING,
#                                         TestUtilities.DELAY_SPREAD)
#         for p in parameters:
#             model = p[0]        # Model
#             scs = p[1]          # subcarrier spacing
#             ds = p[2]           # delay spread
#             # Empirical covariance
#             cov_mat_emp = self.est_tdl_freq_cov_mat(TestUtilities.NUM_SAMPLES, model, ds, 3.5e9,
#                                             scs, 1.0, fft_size)
#             # Expected covariance
#             cov_mat = tdl_freq_cov_mat(model, scs,fft_size, ds)
#             cov_mat = cov_mat.numpy()
#             # Error
#             max_err = np.max(np.abs(cov_mat - cov_mat_emp))
#             self.assertTrue(max_err < TestUtilities.ATOL)

#     def est_tdl_time_cov_mat(self, num_samples, model, carrier_frequency,
#         subcarrier_spacing, speed, num_ofdm_symbols, los_angle_of_arrival):


#         channel_model = TDL(model, 300e-9, carrier_frequency, min_speed=speed, max_speed=speed,
#                             los_angle_of_arrival=los_angle_of_arrival)
#         frequencies = subcarrier_frequencies(1, subcarrier_spacing)

#         batch_size = TestUtilities.BATCH_SIZE
#         num_it = (num_samples//batch_size) + 1
#         hs = []

#         @tf.function(jit_compile=True)
#         def _run():
#             cov_mat = tf.zeros([num_ofdm_symbols, num_ofdm_symbols], tf.complex64)
#             for _ in tf.range(num_it):
#                 a, tau = channel_model(batch_size, num_ofdm_symbols,
#                                         sampling_frequency=subcarrier_spacing)
#                 h = cir_to_ofdm_channel(frequencies, a, tau)[:,0,0,0,0] # [batch size, num_ofdm_symbols, 1]
#                 cov_mat_ = tf.matmul(h, h, adjoint_b=True)
#                 cov_mat_ = tf.reduce_mean(cov_mat_, axis=0)
#                 cov_mat += cov_mat_
#             cov_mat = cov_mat / tf.cast(num_it,tf.complex64)
#             return cov_mat

#         cov_mat = _run().numpy()
#         return cov_mat

#     def test_tdl_time_cov_mat(self):

#         num_ofdm_symbols = TestUtilities.NUM_SYMBOLS
#         los_angle_of_arrival = np.pi/4.

#         parameters =  itertools.product(TestUtilities.TDL_MODELS,
#                                         TestUtilities.SPEEDS,
#                                         TestUtilities.SUBCARRIER_SPACING,
#                                         TestUtilities.CARRIER_FREQS)
#         for p in parameters:
#             model = p[0]                    # Model
#             speed = p[1]                    # Speed
#             subcarrier_spacing = p[2]       # Subcarrier spacing
#             carr_freq = p[3]                # Carrier frequency
#             # Empirical covariance
#             cov_mat_emp = self.est_tdl_time_cov_mat(TestUtilities.NUM_SAMPLES, model, carr_freq,
#                 subcarrier_spacing, speed, num_ofdm_symbols, los_angle_of_arrival)
#             # Expected covariance
#             cov_mat = tdl_time_cov_mat(model, speed, carr_freq, 1./subcarrier_spacing,
#                                         num_ofdm_symbols, los_angle_of_arrival)
#             cov_mat = cov_mat.numpy()
#             # # Error
#             max_err = np.max(np.abs(cov_mat - cov_mat_emp))
#             self.assertTrue(max_err < TestUtilities.ATOL)
