#
# SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
import pytest
import unittest
import numpy as np
from scipy.stats import kstest, rayleigh, rice
from scipy.special import jv
import sionna
from sionna.channel.tr38901 import TDL
from sionna.channel import exp_corr_mat
from channel_test_utils import *

@pytest.mark.usefixtures("only_gpu")
class TestTDL(unittest.TestCase):
    r"""Test the 3GPP TDL channel model.
    """

    # Batch size used to check the LSP distribution
    BATCH_SIZE = 10000

    # Carrier frequency
    CARRIER_FREQUENCY = 3.5e9 # Hz

    # Frequency at which the channel is sampled
    SAMPLING_FREQUENCY = 15e3 # Hz

    # Delay spread
    DELAY_SPREAD = 100e-9 # s

    #  Number of time steps per example
    NUM_TIME_STEPS = 100

    # Number of sinusoids for channel coefficient generation
    NUM_SINUSOIDS = 20

    # Speed
    SPEED = 150 # m/s
    MAX_DOPPLER = 2.*sionna.PI*SPEED/sionna.SPEED_OF_LIGHT*CARRIER_FREQUENCY

    # AoA
    LoS_AoA = np.pi/4

    # Maximum allowed deviation for distance calculation (relative error)
    MAX_ERR = 5e-2

    def setUpClass():

        # Dict for storing the samples
        TestTDL.channel_coeff = {}
        TestTDL.delays = {}

        ########## TDL-A
        tdl = TDL(  "A",
                    delay_spread=TestTDL.DELAY_SPREAD,
                    carrier_frequency=TestTDL.CARRIER_FREQUENCY,
                    num_sinusoids=TestTDL.NUM_SINUSOIDS,
                    min_speed=TestTDL.SPEED)
        h,tau = tdl(batch_size=TestTDL.BATCH_SIZE,
                    num_time_steps=TestTDL.NUM_TIME_STEPS,
                    sampling_frequency=TestTDL.SAMPLING_FREQUENCY)
        TestTDL.channel_coeff['A'] = h.numpy()[:,0,0,0,0,:,:]
        TestTDL.delays['A'] = tau.numpy()[:,0,0,:]

        ########## TDL-B
        tdl = TDL(  "B",
                    delay_spread=TestTDL.DELAY_SPREAD,
                    carrier_frequency=TestTDL.CARRIER_FREQUENCY,
                    num_sinusoids=TestTDL.NUM_SINUSOIDS,
                    min_speed=TestTDL.SPEED)
        h,tau = tdl(batch_size=TestTDL.BATCH_SIZE,
                    num_time_steps=TestTDL.NUM_TIME_STEPS,
                    sampling_frequency=TestTDL.SAMPLING_FREQUENCY)
        TestTDL.channel_coeff['B'] = h.numpy()[:,0,0,0,0,:,:]
        TestTDL.delays['B'] = tau.numpy()[:,0,0,:]

        ########## TDL-C
        tdl = TDL(  "C",
                    delay_spread=TestTDL.DELAY_SPREAD,
                    carrier_frequency=TestTDL.CARRIER_FREQUENCY,
                    num_sinusoids=TestTDL.NUM_SINUSOIDS,
                    min_speed=TestTDL.SPEED)
        h,tau = tdl(batch_size=TestTDL.BATCH_SIZE,
                    num_time_steps=TestTDL.NUM_TIME_STEPS,
                    sampling_frequency=TestTDL.SAMPLING_FREQUENCY)
        TestTDL.channel_coeff['C'] = h.numpy()[:,0,0,0,0,:,:]
        TestTDL.delays['C'] = tau.numpy()[:,0,0,:]

        ########## TDL-D
        tdl = TDL(  "D",
                    delay_spread=TestTDL.DELAY_SPREAD,
                    carrier_frequency=TestTDL.CARRIER_FREQUENCY,
                    num_sinusoids=TestTDL.NUM_SINUSOIDS,
                    los_angle_of_arrival=TestTDL.LoS_AoA,
                    min_speed=TestTDL.SPEED)
        h,tau = tdl(batch_size=TestTDL.BATCH_SIZE,
                    num_time_steps=TestTDL.NUM_TIME_STEPS,
                    sampling_frequency=TestTDL.SAMPLING_FREQUENCY)
        TestTDL.channel_coeff['D'] = h.numpy()[:,0,0,0,0,:,:]
        TestTDL.delays['D'] = tau.numpy()[:,0,0,:]

        ########## TDL-E
        tdl = TDL(  "E",
                    delay_spread=TestTDL.DELAY_SPREAD,
                    carrier_frequency=TestTDL.CARRIER_FREQUENCY,
                    num_sinusoids=TestTDL.NUM_SINUSOIDS,
                    los_angle_of_arrival=TestTDL.LoS_AoA,
                    min_speed=TestTDL.SPEED)
        h,tau = tdl(batch_size=TestTDL.BATCH_SIZE,
                    num_time_steps=TestTDL.NUM_TIME_STEPS,
                    sampling_frequency=TestTDL.SAMPLING_FREQUENCY)
        TestTDL.channel_coeff['E'] = h.numpy()[:,0,0,0,0,:,:]
        TestTDL.delays['E'] = tau.numpy()[:,0,0,:]

        ########## TDL-A30
        tdl = TDL(  "A30",
                    delay_spread=30e-9,
                    carrier_frequency=TestTDL.CARRIER_FREQUENCY,
                    num_sinusoids=TestTDL.NUM_SINUSOIDS,
                    los_angle_of_arrival=TestTDL.LoS_AoA,
                    min_speed=TestTDL.SPEED)
        h,tau = tdl(batch_size=TestTDL.BATCH_SIZE,
                    num_time_steps=TestTDL.NUM_TIME_STEPS,
                    sampling_frequency=TestTDL.SAMPLING_FREQUENCY)
        TestTDL.channel_coeff['A30'] = h.numpy()[:,0,0,0,0,:,:]
        TestTDL.delays['A30'] = tau.numpy()[:,0,0,:]

        ########## TDL-B100
        tdl = TDL(  "B100",
                    delay_spread=100e-9,
                    carrier_frequency=TestTDL.CARRIER_FREQUENCY,
                    num_sinusoids=TestTDL.NUM_SINUSOIDS,
                    los_angle_of_arrival=TestTDL.LoS_AoA,
                    min_speed=TestTDL.SPEED)
        h,tau = tdl(batch_size=TestTDL.BATCH_SIZE,
                    num_time_steps=TestTDL.NUM_TIME_STEPS,
                    sampling_frequency=TestTDL.SAMPLING_FREQUENCY)
        TestTDL.channel_coeff['B100'] = h.numpy()[:,0,0,0,0,:,:]
        TestTDL.delays['B100'] = tau.numpy()[:,0,0,:]

        ########## TDL-C300
        tdl = TDL(  "C300",
                    delay_spread=300e-9,
                    carrier_frequency=TestTDL.CARRIER_FREQUENCY,
                    num_sinusoids=TestTDL.NUM_SINUSOIDS,
                    los_angle_of_arrival=TestTDL.LoS_AoA,
                    min_speed=TestTDL.SPEED)
        h,tau = tdl(batch_size=TestTDL.BATCH_SIZE,
                    num_time_steps=TestTDL.NUM_TIME_STEPS,
                    sampling_frequency=TestTDL.SAMPLING_FREQUENCY)
        TestTDL.channel_coeff['C300'] = h.numpy()[:,0,0,0,0,:,:]
        TestTDL.delays['C300'] = tau.numpy()[:,0,0,:]


    @channel_test_on_models(('A', 'B', 'C', 'D', 'E', 'A30', 'B100', 'C300'), ('foo',))
    def test_pdp(self, model, submodel): # Submodel does not apply to TDL
        """Test power delay profiles"""
        # Checking powers
        h = TestTDL.channel_coeff[model]
        p = np.mean(np.square(np.abs(h[:,:,0])), axis=0)
        ref_p = np.power(10.0, TDL_POWERS[model]/10.0)
        ref_p = ref_p / np.sum(ref_p)
        max_err = np.max(np.abs(ref_p - p))
        self.assertLessEqual(max_err, TestTDL.MAX_ERR, f'{model}')
        # Checking delays
        if model in ('A30', 'B100', 'C300'):
            tau = TestTDL.delays[model]
            ref_tau = np.expand_dims(TDL_DELAYS[model], axis=0)*1e-9 # ns to s
        else:
            tau = TestTDL.delays[model]/TestTDL.DELAY_SPREAD
            ref_tau = np.expand_dims(TDL_DELAYS[model], axis=0)
        max_err = np.max(np.abs(ref_tau - tau))
        self.assertLessEqual(max_err, TestTDL.MAX_ERR, f'{model}')

    # Submodel does not apply to TDL
    @channel_test_on_models(('A', 'B', 'C', 'D', 'E', 'A30', 'B100', 'C300'), ('foo',))
    def test_taps_powers_distributions(self, model, submodel):
        """Test the distribution of the taps powers"""
        ref_powers = np.power(10.0, TDL_POWERS[model]/10.0)
        ref_powers = ref_powers / np.sum(ref_powers)
        h = TestTDL.channel_coeff[model]
        powers = np.abs(h)
        for i,p in enumerate(ref_powers):
            if i == 0 and (model == 'D' or model == 'E'):
                K = np.power(10.0, TDL_RICIAN_K[model]/10.0)
                P0 = ref_powers[0]
                s = np.sqrt(0.5*P0/(1+K))
                b = np.sqrt(K*2)
                D,_ = kstest(   powers[:,i,0].flatten(),
                                rice.cdf,
                                args=(b, 0.0, s))
            else:
                D,_ = kstest(   powers[:,i,0].flatten(),
                                rayleigh.cdf,
                                args=(0.0, np.sqrt(0.5*p)))
            self.assertLessEqual(D, TestTDL.MAX_ERR, f'{model}')

    def corr(self, x, max_lags):
        num_lags = x.shape[-1]//2
        c = np.zeros([max_lags], dtype=complex)
        for i in range(0, max_lags):
            c[i] = np.mean(np.conj(x[...,:num_lags])*x[...,i:num_lags+i])
        return c

    def auto_real(self, max_doppler, t, power): #Eq. (8a)
        return 0.5*power*jv(0,t*max_doppler)

    def auto_complex(self, max_doppler, t, power): #Eq. (8c)
        return power*jv(0,t*max_doppler)

    def auto_abs2(self, max_doppler, num_sinusoids, t, power): #Eq. (8d)
        return (1 + (1-1/num_sinusoids)*jv(0,t*max_doppler)**2)*np.square(power)

    def auto_complex_rice(self, max_doppler, K, theta_0, t):
        a = jv(0,t*max_doppler)
        b = K*np.cos(t*max_doppler*np.cos(theta_0))
        c = 1j*K*np.sin(t*max_doppler*np.cos(theta_0))
        return (a + b+ c)/(1+K)

    # Submodel does not apply to TDL
    @channel_test_on_models(('A', 'B', 'C', 'D', 'E', 'A30', 'B100', 'C300'), ('foo',))
    def test_autocorrelation(self, model, submodel):
        """Test the autocorrelation"""
        max_lag = TestTDL.NUM_TIME_STEPS//2
        ref_powers = np.power(10.0, TDL_POWERS[model]/10.0)
        ref_powers = ref_powers / np.sum(ref_powers)
        time = np.arange(max_lag)/TestTDL.SAMPLING_FREQUENCY
        #
        for i, p in enumerate(ref_powers):
            if i == 0 and (model == 'D' or model == 'E'):
                h = self.channel_coeff[model]
                h = h[:,i,:]
                r = self.corr(h, max_lag)
                #
                K = np.power(10.0, TDL_RICIAN_K[model]/10.0)
                ref_r = self.auto_complex_rice(TestTDL.MAX_DOPPLER, K,
                                                TestTDL.LoS_AoA, time)*p
                max_err = np.max(np.abs(r - ref_r))
                self.assertLessEqual(max_err, TestTDL.MAX_ERR, f'{model}')
            else:
                h = self.channel_coeff[model]
                h = h[:,i,:]
                r_real = self.corr(h.real, max_lag)
                r_imag = self.corr(h.imag, max_lag)
                r      = self.corr(h, max_lag)
                # r_abs2 = self.corr(np.square(np.abs(h)), max_lag)
                #
                ref_r_real = self.auto_real(TestTDL.MAX_DOPPLER, time, p)
                max_err = np.max(np.abs(r_real - ref_r_real))
                self.assertLessEqual(max_err, TestTDL.MAX_ERR, f'{model}')
                #
                max_err = np.max(np.abs(r_imag - ref_r_real))
                self.assertLessEqual(max_err, TestTDL.MAX_ERR, f'{model}')
                #
                ref_r = self.auto_complex(TestTDL.MAX_DOPPLER, time, p)
                max_err = np.max(np.abs(r - ref_r))
                self.assertLessEqual(max_err, TestTDL.MAX_ERR, f'{model}')
                #
                # ref_r_abs2 = self.auto_abs2(TestTDL.MAX_DOPPLER,
                #                             TestTDL.NUM_SINUSOIDS, time, p)
                # max_err = np.max(np.abs(r_abs2 - ref_r_abs2))
                # self.assertLessEqual(max_err, TestTDL.MAX_ERR, f'{model}')

    # No need to test on evey channel model for spatial correlation
    def test_spatial_correlation_separate_rx_tx(self):
        """Test spatial Correlation with separate RX and TX correlation"""

        # Instantiate the model
        num_rx_ant = 16
        num_tx_ant = 16
        rx_corr_mat = exp_corr_mat(0.9, num_rx_ant)
        tx_corr_mat = exp_corr_mat(0.5, num_tx_ant)
        tdl = TDL(model = "A",
                 delay_spread = 100e-9,
                 carrier_frequency = 3.5e9,
                 min_speed = 0.0, max_speed = 0.0,
                 num_rx_ant=num_rx_ant,num_tx_ant=num_tx_ant,
                 rx_corr_mat=rx_corr_mat, tx_corr_mat=tx_corr_mat)

        # Empirical estimation of the correlation matrices
        est_rx_cov = np.zeros([num_rx_ant,num_rx_ant], complex)
        est_tx_cov = np.zeros([num_tx_ant,num_tx_ant], complex)
        num_it = 1000
        batch_size = 1000
        for _ in range(num_it):
            h, _ = tdl(batch_size, 1, 1)

            h = np.transpose(h, [0,1,3,5,6,2,4]) # [..., rx ant, tx ant]
            h = h[:,0,0,0,0,:,:]/np.sqrt(tdl.mean_powers[0].numpy()) # [batch size, rx ant, tx ant]

            # RX correlation
            h_ = np.expand_dims(h[:,:,0], axis=-1) # [batch size, rx ant, 1]
            est_rx_cov_ = np.matmul(h_, np.conj(np.transpose(h_, [0,2,1])))
            est_rx_cov_ = np.mean(est_rx_cov_, axis=0) # [rx ant, rx ant]
            est_rx_cov += est_rx_cov_

            # TX correlation
            h_ = np.expand_dims(h[:,0,:], axis=-1) # [batch size, rx ant, 1]
            est_tx_cov_ = np.matmul(h_, np.conj(np.transpose(h_, [0,2,1])))
            est_tx_cov_ = np.mean(est_tx_cov_, axis=0) # [rx ant, rx ant]
            est_tx_cov += est_tx_cov_
        est_rx_cov /= num_it
        est_tx_cov /= num_it

        # Test
        max_err = np.max(np.abs(est_rx_cov - rx_corr_mat))
        self.assertLessEqual(max_err, TestTDL.MAX_ERR, f'Receiver correlation')
        max_err = np.max(np.abs(est_tx_cov - tx_corr_mat))
        self.assertLessEqual(max_err, TestTDL.MAX_ERR, f'Transmitter correlation')

    # No need to test on evey channel model for spatial correlation
    def test_spatial_correlation_joint_rx_tx(self):
        """Test spatial Correlation with joint filtering"""

        # Instantiate the model
        num_rx_ant = 16
        num_tx_ant = 16
        rx_corr_mat = exp_corr_mat(0.9, num_rx_ant//2).numpy()
        pol_corr_mat = np.array([[1.0, 0.8, 0.0, 0.0],
                                 [0.8, 1.0, 0.0, 0.0],
                                 [0.0, 0.0, 1.0, 0.8],
                                 [0.0, 0.0, 0.8, 1.0]])
        tx_corr_mat = exp_corr_mat(0.5, num_tx_ant//2).numpy()
        spatial_corr_mat = np.kron(pol_corr_mat, tx_corr_mat)
        spatial_corr_mat = np.kron(rx_corr_mat, spatial_corr_mat)
        tdl = TDL(model = "A",
                 delay_spread = 100e-9,
                 carrier_frequency = 3.5e9,
                 min_speed = 0.0, max_speed = 0.0,
                 num_rx_ant=num_rx_ant,num_tx_ant=num_tx_ant,
                 spatial_corr_mat=spatial_corr_mat)

        # Empirical estimation of the correlation matrices
        est_spatial_cov = np.zeros([num_tx_ant*num_rx_ant,
                                    num_tx_ant*num_rx_ant], complex)
        num_it = 1000
        batch_size = 1000
        for _ in range(num_it):
            h, _ = tdl(batch_size, 1, 1)

            h = np.transpose(h, [0,1,3,5,6,2,4]) # [..., rx ant, tx ant]
            h = h[:,0,0,0,0,:,:]/np.sqrt(tdl.mean_powers[0].numpy()) # [batch size, rx ant, tx ant]
            h = np.reshape(h, [batch_size, -1]) # [batch size, rx ant*tx ant]

            # Spatial correlation
            h_ = np.expand_dims(h, axis=-1) # [batch size, rx ant*tx ant, 1]
            est_spatial_cov_ = np.matmul(h_, np.conj(np.transpose(h_, [0,2,1])))
            est_spatial_cov_ = np.mean(est_spatial_cov_, axis=0) # [rx ant, rx ant]
            est_spatial_cov += est_spatial_cov_
        est_spatial_cov /= num_it

        # Test
        max_err = np.max(np.abs(est_spatial_cov - spatial_corr_mat))
        self.assertLessEqual(max_err, TestTDL.MAX_ERR)

    # No need to test on evey channel model for spatial correlation
    def test_no_spatial_correlation(self):
        """No spatial correlation specified leads to no spatial correlation observed"""

        # Instantiate the model
        num_rx_ant = 16
        num_tx_ant = 16
        tdl = TDL(model = "A",
                 delay_spread = 100e-9,
                 carrier_frequency = 3.5e9,
                 min_speed = 0.0, max_speed = 0.0,
                 num_rx_ant=num_rx_ant,num_tx_ant=num_tx_ant)

        # Empirical estimation of the correlation matrices
        est_spatial_cov = np.zeros([num_tx_ant*num_rx_ant,
                                    num_tx_ant*num_rx_ant], complex)
        num_it = 1000
        batch_size = 1000
        for _ in range(num_it):
            h, _ = tdl(batch_size, 1, 1)

            h = np.transpose(h, [0,1,3,5,6,2,4]) # [..., rx ant, tx ant]
            h = h[:,0,0,0,0,:,:]/np.sqrt(tdl.mean_powers[0].numpy()) # [batch size, rx ant, tx ant]
            h = np.reshape(h, [batch_size, -1]) # [batch size, rx ant*tx ant]

            # Spatial correlation
            h_ = np.expand_dims(h, axis=-1) # [batch size, rx ant*tx ant, 1]
            est_spatial_cov_ = np.matmul(h_, np.conj(np.transpose(h_, [0,2,1])))
            est_spatial_cov_ = np.mean(est_spatial_cov_, axis=0) # [rx ant, rx ant]
            est_spatial_cov += est_spatial_cov_
        est_spatial_cov /= num_it

        # Test
        spatial_corr_mat = np.eye(num_rx_ant*num_rx_ant)
        max_err = np.max(np.abs(est_spatial_cov - spatial_corr_mat))
        self.assertLessEqual(max_err, TestTDL.MAX_ERR)

    # No need to test on evey channel model for spatial correlation
    def test_rx_corr_only(self):
        """Test with RX spatial correlation only"""

        # Instantiate the model
        num_rx_ant = 16
        num_tx_ant = 16
        rx_corr_mat = exp_corr_mat(0.9, num_rx_ant)
        tx_corr_mat = np.eye(num_tx_ant)
        tdl = TDL(model = "A",
                 delay_spread = 100e-9,
                 carrier_frequency = 3.5e9,
                 min_speed = 0.0, max_speed = 0.0,
                 num_rx_ant=num_rx_ant,num_tx_ant=num_tx_ant,
                 rx_corr_mat=rx_corr_mat)

        # Empirical estimation of the correlation matrices
        est_rx_cov = np.zeros([num_rx_ant,num_rx_ant], complex)
        est_tx_cov = np.zeros([num_tx_ant,num_tx_ant], complex)
        num_it = 1000
        batch_size = 1000
        for _ in range(num_it):
            h, _ = tdl(batch_size, 1, 1)

            h = np.transpose(h, [0,1,3,5,6,2,4]) # [..., rx ant, tx ant]
            h = h[:,0,0,0,0,:,:]/np.sqrt(tdl.mean_powers[0].numpy()) # [batch size, rx ant, tx ant]

            # RX correlation
            h_ = np.expand_dims(h[:,:,0], axis=-1) # [batch size, rx ant, 1]
            est_rx_cov_ = np.matmul(h_, np.conj(np.transpose(h_, [0,2,1])))
            est_rx_cov_ = np.mean(est_rx_cov_, axis=0) # [rx ant, rx ant]
            est_rx_cov += est_rx_cov_

            # TX correlation
            h_ = np.expand_dims(h[:,0,:], axis=-1) # [batch size, rx ant, 1]
            est_tx_cov_ = np.matmul(h_, np.conj(np.transpose(h_, [0,2,1])))
            est_tx_cov_ = np.mean(est_tx_cov_, axis=0) # [rx ant, rx ant]
            est_tx_cov += est_tx_cov_
        est_rx_cov /= num_it
        est_tx_cov /= num_it

        # Test
        max_err = np.max(np.abs(est_rx_cov - rx_corr_mat))
        self.assertLessEqual(max_err, TestTDL.MAX_ERR, f'Receiver correlation')
        max_err = np.max(np.abs(est_tx_cov - tx_corr_mat))
        self.assertLessEqual(max_err, TestTDL.MAX_ERR, f'Transmitter correlation')

    # No need to test on evey channel model for spatial correlation
    def test_tx_corr_only(self):
        """Test with TX spatial Correlation only"""

        # Instantiate the model
        num_rx_ant = 16
        num_tx_ant = 16
        rx_corr_mat = np.eye(num_tx_ant)
        tx_corr_mat = exp_corr_mat(0.9, num_rx_ant)
        tdl = TDL(model = "A",
                 delay_spread = 100e-9,
                 carrier_frequency = 3.5e9,
                 min_speed = 0.0, max_speed = 0.0,
                 num_rx_ant=num_rx_ant,num_tx_ant=num_tx_ant,
                 tx_corr_mat=tx_corr_mat)

        # Empirical estimation of the correlation matrices
        est_rx_cov = np.zeros([num_rx_ant,num_rx_ant], complex)
        est_tx_cov = np.zeros([num_tx_ant,num_tx_ant], complex)
        num_it = 1000
        batch_size = 1000
        for _ in range(num_it):
            h, _ = tdl(batch_size, 1, 1)

            h = np.transpose(h, [0,1,3,5,6,2,4]) # [..., rx ant, tx ant]
            h = h[:,0,0,0,0,:,:]/np.sqrt(tdl.mean_powers[0].numpy()) # [batch size, rx ant, tx ant]

            # RX correlation
            h_ = np.expand_dims(h[:,:,0], axis=-1) # [batch size, rx ant, 1]
            est_rx_cov_ = np.matmul(h_, np.conj(np.transpose(h_, [0,2,1])))
            est_rx_cov_ = np.mean(est_rx_cov_, axis=0) # [rx ant, rx ant]
            est_rx_cov += est_rx_cov_

            # TX correlation
            h_ = np.expand_dims(h[:,0,:], axis=-1) # [batch size, rx ant, 1]
            est_tx_cov_ = np.matmul(h_, np.conj(np.transpose(h_, [0,2,1])))
            est_tx_cov_ = np.mean(est_tx_cov_, axis=0) # [rx ant, rx ant]
            est_tx_cov += est_tx_cov_
        est_rx_cov /= num_it
        est_tx_cov /= num_it

        # Test
        max_err = np.max(np.abs(est_rx_cov - rx_corr_mat))
        self.assertLessEqual(max_err, TestTDL.MAX_ERR, f'Receiver correlation')
        max_err = np.max(np.abs(est_tx_cov - tx_corr_mat))
        self.assertLessEqual(max_err, TestTDL.MAX_ERR, f'Transmitter correlation')

    # No need to test on evey channel model for spatial correlation
    def test_spatial_correlation_all_three_inputs(self):
        """Test spatial correlation with all three inputs"""

        # Instantiate the model
        num_rx_ant = 16
        num_tx_ant = 16
        rx_corr_mat = exp_corr_mat(0.9, num_rx_ant//2).numpy()
        pol_corr_mat = np.array([[1.0, 0.8, 0.0, 0.0],
                                 [0.8, 1.0, 0.0, 0.0],
                                 [0.0, 0.0, 1.0, 0.8],
                                 [0.0, 0.0, 0.8, 1.0]])
        tx_corr_mat = exp_corr_mat(0.5, num_tx_ant//2).numpy()
        spatial_corr_mat = np.kron(pol_corr_mat, tx_corr_mat)
        spatial_corr_mat = np.kron(rx_corr_mat, spatial_corr_mat)
        tdl = TDL(model = "A",
                 delay_spread = 100e-9,
                 carrier_frequency = 3.5e9,
                 min_speed = 0.0, max_speed = 0.0,
                 num_rx_ant=num_rx_ant,num_tx_ant=num_tx_ant,
                 spatial_corr_mat=spatial_corr_mat,
                 rx_corr_mat=np.eye(num_rx_ant), tx_corr_mat=np.eye(num_tx_ant))

        # Empirical estimation of the correlation matrices
        est_spatial_cov = np.zeros([num_tx_ant*num_rx_ant,
                                    num_tx_ant*num_rx_ant], complex)
        num_it = 1000
        batch_size = 1000
        for _ in range(num_it):
            h, _ = tdl(batch_size, 1, 1)

            h = np.transpose(h, [0,1,3,5,6,2,4]) # [..., rx ant, tx ant]
            h = h[:,0,0,0,0,:,:]/np.sqrt(tdl.mean_powers[0].numpy()) # [batch size, rx ant, tx ant]
            h = np.reshape(h, [batch_size, -1]) # [batch size, rx ant*tx ant]

            # Spatial correlation
            h_ = np.expand_dims(h, axis=-1) # [batch size, rx ant*tx ant, 1]
            est_spatial_cov_ = np.matmul(h_, np.conj(np.transpose(h_, [0,2,1])))
            est_spatial_cov_ = np.mean(est_spatial_cov_, axis=0) # [rx ant, rx ant]
            est_spatial_cov += est_spatial_cov_
        est_spatial_cov /= num_it

        # Test
        max_err = np.max(np.abs(est_spatial_cov - spatial_corr_mat))
        self.assertLessEqual(max_err, TestTDL.MAX_ERR)
