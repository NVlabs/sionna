#
# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
try:
    import sionna
except ImportError as e:
    import sys
    sys.path.append("../")
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

import sionna
import unittest
import numpy as np
from sionna.channel.tr38901 import TDL
from channel_test_utils import *
from scipy.stats import kstest, rayleigh, rice
from scipy.special import jv


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

        # Forcing the seed to make the tests deterministic
        tf.random.set_seed(42)
        np.random.seed(42)

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


    @channel_test_on_models(('A', 'B', 'C', 'D', 'E'), ('foo',))
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
        tau = TestTDL.delays[model]/TestTDL.DELAY_SPREAD
        ref_tau = np.expand_dims(TDL_DELAYS[model], axis=0)
        max_err = np.max(np.abs(ref_tau - tau))
        self.assertLessEqual(max_err, TestTDL.MAX_ERR, f'{model}')

    # Submodel does not apply to TDL
    @channel_test_on_models(('A', 'B', 'C', 'D', 'E'), ('foo',))
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
    @channel_test_on_models(('A', 'B', 'C', 'D', 'E'), ('foo',))
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
