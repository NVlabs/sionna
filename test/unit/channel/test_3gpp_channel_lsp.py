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

import unittest
import numpy as np
import sionna
from channel_test_utils import *
from scipy.stats import kstest, norm


class TestLSP(unittest.TestCase):
    r"""Test the distribution, cross-correlation, and spatial correlation of
    3GPP channel models' LSPs
    """

    # Carrier frequency
    CARRIER_FREQUENCY = 3.5e9 # Hz

    # Heigh of UTs
    H_UT = 1.5

    # Heigh of BSs
    H_BS = 35.0

    # Batch size for generating samples of LSPs and pathlosses
    BATCH_SIZE = 500000

    # More than one UT is required for testing the spatial and cross-correlation
    # of LSPs
    NB_UT = 5

    # The LSPs follow either a Gaussian or a truncated Gaussian
    # distribution. A Kolmogorov-Smirnov (KS) test is used to check that the
    # LSP follow the appropriate distribution. This is the threshold below
    # which the KS statistic `D` should be for passing the test.
    MAX_ERR_KS = 1e-2

    # # Maximum allowed deviation for cross-correlation of LSP parameters
    MAX_ERR_CROSS_CORR = 3e-2

    # # Maximum allowed deviation for spatial correlation of LSP parameters
    MAX_ERR_SPAT_CORR = 3e-2

    # LoS probability
    MAX_ERR_LOS_PROB = 1e-2

    # ZOD Offset maximum relative error
    MAX_ERR_ZOD_OFFSET = 1e-2

    # Maximum allowed deviation for pathloss
    MAX_ERR_PATHLOSS_MEAN = 1.0
    MAX_ERR_PATHLOSS_STD = 1e-1

    def limited_normal(self, batch_size, minval, maxval, mu, std):
        r"""
        Return a limited normal distribution. This is different from a truncated
        normal distribution, as the samples exceed ``minval`` and ``maxval`` are
        clipped.

        More precisely, ``x`` is generated as follows:
        1. Sample ``y`` of shape [``batch_size``] from a Gaussian distribution N(mu,std)
        2. x = max(min(x, maxval), minval)
        """
        x = np.random.normal(size=[batch_size])
        x = np.maximum(x, minval)
        x = np.minimum(x, maxval)
        x = std*x+mu
        return x

    def setUpClass():
        r"""Sample LSPs and pathlosses from all channel models for testing"""

        # Forcing the seed to make the tests deterministic
        tf.random.set_seed(42)
        np.random.seed(42)

        nb_bs = 1
        fc = TestLSP.CARRIER_FREQUENCY
        h_ut = TestLSP.H_UT
        h_bs = TestLSP.H_BS
        batch_size = TestLSP.BATCH_SIZE
        nb_ut = TestLSP.NB_UT

        # UT and BS arrays have no impact on LSP
        # However, these are needed to instantiate the model
        bs_array = sionna.channel.tr38901.PanelArray(num_rows_per_panel=2,
                                                    num_cols_per_panel=2,
                                                    polarization='dual',
                                                    polarization_type='VH',
                                                    antenna_pattern='38.901',
                                                    carrier_frequency=fc,
                                                    dtype=tf.complex128)
        ut_array = sionna.channel.tr38901.PanelArray(num_rows_per_panel=1,
                                                    num_cols_per_panel=1,
                                                    polarization='dual',
                                                    polarization_type='VH',
                                                    antenna_pattern='38.901',
                                                    carrier_frequency=fc,
                                                    dtype=tf.complex128)

        # The following quantities have no impact on LSP
        # However,these are needed to instantiate the model
        ut_orientations = tf.zeros([batch_size, nb_ut], dtype=tf.float64)
        bs_orientations = tf.zeros([batch_size, nb_ut], dtype=tf.float64)
        ut_velocities = tf.zeros([batch_size, nb_ut], dtype=tf.float64)

        # LSPs, ZoD offset, pathlosses
        TestLSP.lsp_samples = {}
        TestLSP.zod_offset = {}
        TestLSP.pathlosses = {}
        TestLSP.los_prob = {}

        ut_loc = generate_random_loc(batch_size, nb_ut, (100,2000),
                                     (100,2000), (h_ut, h_ut),
                                     share_loc=True, dtype=tf.float64)
        bs_loc = generate_random_loc(batch_size, nb_bs, (0,100),
                                     (0,100), (h_bs, h_bs),
                                     share_loc=True, dtype=tf.float64)

        ####### RMa
        TestLSP.lsp_samples['rma'] = {}
        TestLSP.zod_offset['rma'] = {}
        TestLSP.pathlosses['rma'] = {}
        scenario = sionna.channel.tr38901.RMaScenario(  fc,
                                                        ut_array,
                                                        bs_array,
                                                        "uplink",
                                                        dtype=tf.complex128)
        lsp_sampler = sionna.channel.tr38901.LSPGenerator(scenario)

        # LoS
        in_state = generate_random_bool(batch_size, nb_ut, 0.0)
        scenario.set_topology(ut_loc, bs_loc, ut_orientations, bs_orientations,
                              ut_velocities, in_state, True)
        lsp_sampler.topology_updated_callback()
        TestLSP.lsp_samples['rma']['los'] = lsp_sampler()
        TestLSP.zod_offset['rma']['los'] = scenario.zod_offset
        TestLSP.pathlosses['rma']['los'] = lsp_sampler.sample_pathloss()[:,0,:]

        # NLoS
        in_state = generate_random_bool(batch_size, nb_ut, 0.0)
        scenario.set_topology(ut_loc, bs_loc, ut_orientations, bs_orientations,
                              ut_velocities, in_state, False)
        lsp_sampler.topology_updated_callback()
        TestLSP.lsp_samples['rma']['nlos'] = lsp_sampler()
        TestLSP.zod_offset['rma']['nlos'] = scenario.zod_offset
        TestLSP.pathlosses['rma']['nlos'] = lsp_sampler.sample_pathloss()[:,0,:]

        # Indoor
        in_state = generate_random_bool(batch_size, nb_ut, 1.0)
        scenario.set_topology(ut_loc, bs_loc, ut_orientations, bs_orientations,
                              ut_velocities, in_state)
        lsp_sampler.topology_updated_callback()
        TestLSP.lsp_samples['rma']['o2i'] = lsp_sampler()
        TestLSP.zod_offset['rma']['o2i'] = scenario.zod_offset
        TestLSP.pathlosses['rma']['o2i'] = lsp_sampler.sample_pathloss()[:,0,:]

        TestLSP.los_prob['rma'] = scenario.los_probability.numpy()

        TestLSP.rma_w = scenario.average_street_width
        TestLSP.rma_h = scenario.average_building_height

        ####### UMi
        TestLSP.lsp_samples['umi'] = {}
        TestLSP.zod_offset['umi'] = {}
        TestLSP.pathlosses['umi'] = {}
        scenario = sionna.channel.tr38901.UMiScenario(  fc,
                                                        'low',
                                                        ut_array,
                                                        bs_array,
                                                        "uplink",
                                                        dtype=tf.complex128)
        lsp_sampler = sionna.channel.tr38901.LSPGenerator(scenario)

        # LoS
        in_state = generate_random_bool(batch_size, nb_ut, 0.0)
        scenario.set_topology(ut_loc, bs_loc, ut_orientations, bs_orientations,
                              ut_velocities, in_state, True)
        lsp_sampler.topology_updated_callback()
        TestLSP.lsp_samples['umi']['los'] = lsp_sampler()
        TestLSP.zod_offset['umi']['los'] = scenario.zod_offset
        TestLSP.pathlosses['umi']['los'] = lsp_sampler.sample_pathloss()[:,0,:]

        # NLoS
        in_state = generate_random_bool(batch_size, nb_ut, 0.0)
        scenario.set_topology(ut_loc, bs_loc, ut_orientations, bs_orientations,
                              ut_velocities, in_state, False)
        lsp_sampler.topology_updated_callback()
        TestLSP.lsp_samples['umi']['nlos'] = lsp_sampler()
        TestLSP.zod_offset['umi']['nlos'] = scenario.zod_offset
        TestLSP.pathlosses['umi']['nlos'] = lsp_sampler.sample_pathloss()[:,0,:]

        # Indoor
        in_state = generate_random_bool(batch_size, nb_ut, 1.0)
        scenario.set_topology(ut_loc, bs_loc, ut_orientations, bs_orientations,
                              ut_velocities, in_state)
        lsp_sampler.topology_updated_callback()
        TestLSP.lsp_samples['umi']['o2i'] = lsp_sampler()
        TestLSP.zod_offset['umi']['o2i'] = scenario.zod_offset
        TestLSP.pathlosses['umi']['o2i-low'] = lsp_sampler.sample_pathloss()[:,0,:]

        TestLSP.los_prob['umi'] = scenario.los_probability.numpy()

        ####### UMa
        TestLSP.lsp_samples['uma'] = {}
        TestLSP.zod_offset['uma'] = {}
        TestLSP.pathlosses['uma'] = {}
        scenario = sionna.channel.tr38901.UMaScenario(  fc,
                                                        'low',
                                                        ut_array,
                                                        bs_array,
                                                        "uplink",
                                                        dtype=tf.complex128)
        lsp_sampler = sionna.channel.tr38901.LSPGenerator(scenario)

        # LoS
        in_state = generate_random_bool(batch_size, nb_ut, 0.0)
        scenario.set_topology(ut_loc, bs_loc, ut_orientations, bs_orientations,
                              ut_velocities, in_state, True)
        lsp_sampler.topology_updated_callback()
        TestLSP.lsp_samples['uma']['los'] = lsp_sampler()
        TestLSP.zod_offset['uma']['los'] = scenario.zod_offset
        TestLSP.pathlosses['uma']['los'] = lsp_sampler.sample_pathloss()[:,0,:]

        # NLoS
        in_state = generate_random_bool(batch_size, nb_ut, 0.0)
        scenario.set_topology(ut_loc, bs_loc, ut_orientations, bs_orientations,
                              ut_velocities, in_state, False)
        lsp_sampler.topology_updated_callback()
        TestLSP.lsp_samples['uma']['nlos'] = lsp_sampler()
        TestLSP.zod_offset['uma']['nlos'] = scenario.zod_offset
        TestLSP.pathlosses['uma']['nlos'] = lsp_sampler.sample_pathloss()[:,0,:]

        # Indoor
        in_state = generate_random_bool(batch_size, nb_ut, 1.0)
        scenario.set_topology(ut_loc, bs_loc, ut_orientations, bs_orientations,
                              ut_velocities, in_state)
        lsp_sampler.topology_updated_callback()
        TestLSP.lsp_samples['uma']['o2i'] = lsp_sampler()
        TestLSP.zod_offset['uma']['o2i'] = scenario.zod_offset
        TestLSP.pathlosses['uma']['o2i-low'] = lsp_sampler.sample_pathloss()[:,0,:]

        TestLSP.los_prob['uma'] = scenario.los_probability.numpy()

        # Sample pathlosses with high O2I loss model. Only with UMi and UMa
        ####### UMi-High
        scenario = sionna.channel.tr38901.UMiScenario(  fc,
                                                        'high',
                                                        ut_array,
                                                        bs_array,
                                                        "uplink",
                                                        dtype=tf.complex128)
        lsp_sampler = sionna.channel.tr38901.LSPGenerator(scenario)
        in_state = generate_random_bool(batch_size, nb_ut, 1.0)
        scenario.set_topology(ut_loc, bs_loc, ut_orientations, bs_orientations,
                              ut_velocities, in_state)
        lsp_sampler.topology_updated_callback()
        TestLSP.pathlosses['umi']['o2i-high'] = lsp_sampler.sample_pathloss()[:,0,:]

        ####### UMa-high
        scenario = sionna.channel.tr38901.UMaScenario(  fc,
                                                        'high',
                                                        ut_array,
                                                        bs_array,
                                                        "uplink",
                                                        dtype=tf.complex128)
        lsp_sampler = sionna.channel.tr38901.LSPGenerator(scenario)
        in_state = generate_random_bool(batch_size, nb_ut, 1.0)
        scenario.set_topology(ut_loc, bs_loc, ut_orientations, bs_orientations,
                              ut_velocities, in_state)
        lsp_sampler.topology_updated_callback()
        TestLSP.pathlosses['uma']['o2i-high'] = lsp_sampler.sample_pathloss()[:,0,:]

        # The following values do not depend on the scenario
        TestLSP.d_2d = scenario.distance_2d.numpy()
        TestLSP.d_2d_ut = scenario.matrix_ut_distance_2d.numpy()
        TestLSP.d_2d_out = scenario.distance_2d_out.numpy()
        TestLSP.d_3d = scenario.distance_3d[0,0,:].numpy()

    @channel_test_on_models(('rma', 'umi', 'uma'), ('los', 'nlos', 'o2i'))
    def test_ds_dist(self, model, submodel):
        """Test the distribution of LSP DS"""
        samples = TestLSP.lsp_samples[model][submodel].ds[:,0,0].numpy()
        samples = np.log10(samples)
        mu, std = log10DS(model, submodel, TestLSP.CARRIER_FREQUENCY)
        D,_ = kstest(samples, norm.cdf, args=(mu, std))
        self.assertLessEqual(D, TestLSP.MAX_ERR_KS, f"{model}:{submodel}")

    @channel_test_on_models(('rma', 'umi', 'uma'), ('los', 'nlos', 'o2i'))
    def test_asa_dist(self, model, submodel):
        """Test the distribution of LSP ASA"""
        samples = TestLSP.lsp_samples[model][submodel].asa[:,0,0].numpy()
        samples = np.log10(samples)
        mu, std = log10ASA(model, submodel, TestLSP.CARRIER_FREQUENCY)
        a = -np.inf
        b = (np.log10(104)-mu)/std
        samples_ref = self.limited_normal(TestLSP.BATCH_SIZE, a, b, mu, std)
        # KS-test does not work great with discontinuties.
        # Therefore, we test only the continuous part of the CDF, and also test
        # that the maximum value allowed is not exceeded
        maxval = np.max(samples)
        samples = samples[samples < np.log10(104)]
        samples_ref = samples_ref[samples_ref < np.log10(104)]
        D,_ = kstest(samples, samples_ref)
        self.assertLessEqual(maxval, np.log10(104), f"{model}:{submodel}")
        self.assertLessEqual(D, TestLSP.MAX_ERR_KS, f"{model}:{submodel}")

    @channel_test_on_models(('rma', 'umi', 'uma'), ('los', 'nlos', 'o2i'))
    def test_asd_dist(self, model, submodel):
        """Test the distribution of LSP ASD"""
        samples = TestLSP.lsp_samples[model][submodel].asd.numpy()
        samples = np.log10(samples)
        mu, std = log10ASD(model, submodel, TestLSP.CARRIER_FREQUENCY)
        a = -np.inf
        b = (np.log10(104)-mu)/std
        samples_ref = self.limited_normal(TestLSP.BATCH_SIZE, a, b, mu, std)
        # KS-test does not work great with discontinuties.
        # Therefore, we test only the continuous part of the CDF, and also test
        # that the maximum value allowed is not exceeded
        maxval = np.max(samples)
        samples = samples[samples < np.log10(104)]
        samples_ref = samples_ref[samples_ref < np.log10(104)]
        D,_ = kstest(samples, samples_ref)
        self.assertLessEqual(maxval, np.log10(104), f"{model}:{submodel}")
        self.assertLessEqual(D, TestLSP.MAX_ERR_KS, f"{model}:{submodel}")

    @channel_test_on_models(('rma', 'umi', 'uma'), ('los', 'nlos', 'o2i'))
    def test_zsa_dist(self, model, submodel):
        """Test the distribution of LSP ZSA"""
        samples = TestLSP.lsp_samples[model][submodel].zsa[:,0,0].numpy()
        samples = np.log10(samples)
        mu, std = log10ZSA(model, submodel, TestLSP.CARRIER_FREQUENCY)
        a = -np.inf
        b = (np.log10(52)-mu)/std
        samples_ref = self.limited_normal(TestLSP.BATCH_SIZE, a, b, mu, std)
        # KS-test does not work great with discontinuties.
        # Therefore, we test only the continuous part of the CDF, and also test
        # that the maximum value allowed is not exceeded
        maxval = np.max(samples)
        samples = samples[samples < np.log10(52)]
        samples_ref = samples_ref[samples_ref < np.log10(52)]
        D,_ = kstest(samples, samples_ref)
        self.assertLessEqual(maxval, np.log10(52), f"{model}:{submodel}")
        self.assertLessEqual(D, TestLSP.MAX_ERR_KS, f"{model}:{submodel}")

    @channel_test_on_models(('rma', 'umi', 'uma'), ('los', 'nlos', 'o2i'))
    def test_zsd_dist(self, model, submodel):
        """Test the distribution of LSP ZSD"""
        d_2d = TestLSP.d_2d[0,0,0]
        samples = TestLSP.lsp_samples[model][submodel].zsd[:,0,0].numpy()
        samples = np.log10(samples)
        mu, std = log10ZSD(model, submodel, d_2d, TestLSP.CARRIER_FREQUENCY,
                            TestLSP.H_BS, TestLSP.H_UT)
        a = -np.inf
        b = (np.log10(52)-mu)/std
        samples_ref = self.limited_normal(TestLSP.BATCH_SIZE, a, b, mu, std)
        # KS-test does not work great with discontinuties.
        # Therefore, we test only the continuous part of the CDF, and also test
        # that the maximum value allowed is not exceeded
        maxval = np.max(samples)
        samples = samples[samples < np.log10(52)]
        samples_ref = samples_ref[samples_ref < np.log10(52)]
        D,_ = kstest(samples, samples_ref)
        self.assertLessEqual(maxval, np.log10(52), f"{model}:{submodel}")
        self.assertLessEqual(D, TestLSP.MAX_ERR_KS, f"{model}:{submodel}")

    @channel_test_on_models(('rma', 'umi', 'uma'), ('los', 'nlos', 'o2i'))
    def test_sf_dist(self, model, submodel):
        """Test the distribution of LSP SF"""
        d_2d = TestLSP.d_2d[0,0,0]
        samples = TestLSP.lsp_samples[model][submodel].sf[:,0,0].numpy()
        samples = 10.0*np.log10(samples)
        mu, std = log10SF_dB(model, submodel, d_2d, TestLSP.CARRIER_FREQUENCY,
                            TestLSP.H_BS, TestLSP.H_UT)
        D,_ = kstest(samples, norm.cdf, args=(mu, std))
        self.assertLessEqual(D, TestLSP.MAX_ERR_KS, f"{model}:{submodel}")

    @channel_test_on_models(('rma', 'umi', 'uma'), ('los',))
    def test_k_dist(self, model, submodel):
        """Test the distribution of LSP K"""
        samples = TestLSP.lsp_samples[model][submodel].k_factor[:,0,0].numpy()
        samples = 10.0*np.log10(samples)
        mu, std = log10K_dB(model, submodel)
        D,_ = kstest(samples, norm.cdf, args=(mu, std))
        self.assertLessEqual(D, TestLSP.MAX_ERR_KS, f"{model}:{submodel}")

    @channel_test_on_models(('rma', 'umi', 'uma'), ('los', 'nlos', 'o2i'))
    def test_cross_correlation(self, model, submodel):
        """Test the LSP cross correlation"""
        lsp_list = []
        ds_samples = TestLSP.lsp_samples[model][submodel].ds[:,0,0].numpy()
        ds_samples = np.log10(ds_samples)
        lsp_list.append(ds_samples)
        asd_samples = TestLSP.lsp_samples[model][submodel].asd[:,0,0].numpy()
        asd_samples = np.log10(asd_samples)
        lsp_list.append(asd_samples)
        asa_samples = TestLSP.lsp_samples[model][submodel].asa[:,0,0].numpy()
        asa_samples = np.log10(asa_samples)
        lsp_list.append(asa_samples)
        sf_samples = TestLSP.lsp_samples[model][submodel].sf[:,0,0].numpy()
        sf_samples = np.log10(sf_samples)
        lsp_list.append(sf_samples)
        if submodel == 'los':
            k_samples = TestLSP.lsp_samples[model][submodel].k_factor[:,0,0]
            k_samples = np.log10(k_samples.numpy())
            lsp_list.append(k_samples)
        zsa_samples = TestLSP.lsp_samples[model][submodel].zsa[:,0,0].numpy()
        zsa_samples = np.log10(zsa_samples)
        lsp_list.append(zsa_samples)
        zsd_samples = TestLSP.lsp_samples[model][submodel].zsd[:,0,0].numpy()
        zsd_samples = np.log10(zsd_samples)
        lsp_list.append(zsd_samples)
        lsp_list = np.stack(lsp_list, axis=-1)
        cross_corr_measured = np.corrcoef(lsp_list.T)
        abs_err = np.abs(cross_corr(model, submodel) - cross_corr_measured)
        max_err = np.max(abs_err)
        self.assertLessEqual(max_err, TestLSP.MAX_ERR_CROSS_CORR,
                            f"{model}:{submodel}")

    @channel_test_on_models(('rma', 'umi', 'uma'), ('los', 'nlos', 'o2i'))
    def test_spatial_correlation(self, model, submodel):
        """Test the spatial correlation of LSPs"""
        d_2d_ut = TestLSP.d_2d_ut[0,0]
        #### LoS
        ds_samples = TestLSP.lsp_samples[model][submodel].ds[:,0,:]
        ds_samples = np.log10(ds_samples.numpy())
        asd_samples = TestLSP.lsp_samples[model][submodel].asd[:,0,:]
        asd_samples = np.log10(asd_samples.numpy())
        asa_samples = TestLSP.lsp_samples[model][submodel].asa[:,0,:]
        asa_samples = np.log10(asa_samples.numpy())
        sf_samples = TestLSP.lsp_samples[model][submodel].sf[:,0,:]
        sf_samples = np.log10(sf_samples.numpy())
        if submodel == 'los':
            k_samples = TestLSP.lsp_samples[model][submodel].k_factor[:,0,:]
            k_samples = np.log10(k_samples.numpy())
        zsa_samples = TestLSP.lsp_samples[model][submodel].zsa[:,0,:]
        zsa_samples = np.log10(zsa_samples.numpy())
        zsd_samples = TestLSP.lsp_samples[model][submodel].zsd[:,0,:]
        zsd_samples = np.log10(zsd_samples.numpy())
        #
        C_ds_measured = np.corrcoef(ds_samples.T)[0]
        C_asd_measured = np.corrcoef(asd_samples.T)[0]
        C_asa_measured = np.corrcoef(asa_samples.T)[0]
        C_sf_measured = np.corrcoef(sf_samples.T)[0]
        if submodel == 'los':
            C_k_measured = np.corrcoef(k_samples.T)[0]
        C_zsa_measured = np.corrcoef(zsa_samples.T)[0]
        C_zsd_measured = np.corrcoef(zsd_samples.T)[0]
        #
        C_ds = np.exp(-d_2d_ut/corr_dist_ds(model, submodel))
        C_asd = np.exp(-d_2d_ut/corr_dist_asd(model, submodel))
        C_asa = np.exp(-d_2d_ut/corr_dist_asa(model, submodel))
        C_sf = np.exp(-d_2d_ut/corr_dist_sf(model, submodel))
        if submodel == 'los':
            C_k = np.exp(-d_2d_ut/corr_dist_k(model, submodel))
        C_zsa = np.exp(-d_2d_ut/corr_dist_zsa(model, submodel))
        C_zsd = np.exp(-d_2d_ut/corr_dist_zsd(model, submodel))
        #
        ds_max_err = np.max(np.abs(C_ds_measured - C_ds))
        self.assertLessEqual(ds_max_err, TestLSP.MAX_ERR_SPAT_CORR,
                                f"{model}:{submodel}")
        asd_max_err = np.max(np.abs(C_asd_measured - C_asd))
        self.assertLessEqual(asd_max_err, TestLSP.MAX_ERR_SPAT_CORR,
                                f"{model}:{submodel}")
        asa_max_err = np.max(np.abs(C_asa_measured - C_asa))
        self.assertLessEqual(asa_max_err, TestLSP.MAX_ERR_SPAT_CORR,
                                f"{model}:{submodel}")
        sf_max_err = np.max(np.abs(C_sf_measured - C_sf))
        self.assertLessEqual(sf_max_err, TestLSP.MAX_ERR_SPAT_CORR,
                                f"{model}:{submodel}")
        if submodel == 'los':
            k_max_err = np.max(np.abs(C_k_measured - C_k))
            self.assertLessEqual(k_max_err, TestLSP.MAX_ERR_SPAT_CORR,
                                f"{model}:{submodel}")
        zsa_max_err = np.max(np.abs(C_zsa_measured - C_zsa))
        self.assertLessEqual(zsa_max_err, TestLSP.MAX_ERR_SPAT_CORR,
                                f"{model}:{submodel}")
        zsd_max_err = np.max(np.abs(C_zsd_measured - C_zsd))
        self.assertLessEqual(zsd_max_err, TestLSP.MAX_ERR_SPAT_CORR,
                                f"{model}:{submodel}")

    # Submodel is not needed for LoS probability
    @channel_test_on_models(('rma', 'umi', 'uma'), ('foo',))
    def test_los_probability(self, model, submodel):
        """Test LoS probability"""
        d_2d_out = TestLSP.d_2d_out
        h_ut = TestLSP.H_UT
        #
        los_prob_ref = los_probability(model, d_2d_out, h_ut)
        los_prob = TestLSP.los_prob[model]
        #
        max_err = np.max(np.abs(los_prob_ref-los_prob))
        self.assertLessEqual(max_err, TestLSP.MAX_ERR_LOS_PROB,
                            f"{model}:{submodel}")

    @channel_test_on_models(('rma', 'umi', 'uma'), ('los', 'nlos', 'o2i'))
    def test_zod_offset(self, model, submodel):
        """Test ZOD offset"""
        d_2d = self.d_2d
        fc = TestLSP.CARRIER_FREQUENCY
        h_ut = TestLSP.H_UT
        samples = self.zod_offset[model][submodel]
        samples_ref = zod_offset(model, submodel, fc, d_2d, h_ut)
        max_err = np.max(np.abs(samples-samples_ref))
        self.assertLessEqual(max_err, TestLSP.MAX_ERR_ZOD_OFFSET,
                                f"{model}:{submodel}")

    @channel_test_on_models(('rma','umi', 'uma'), ('los', 'nlos', 'o2i'))
    def test_pathloss(self, model, submodel):
        """Test the pathloss"""
        fc = TestLSP.CARRIER_FREQUENCY
        h_ut = TestLSP.H_UT
        h_bs = TestLSP.H_BS
        if model == 'rma':
            samples = TestLSP.pathlosses[model][submodel]
            mean_samples = tf.reduce_mean(samples, axis=0).numpy()
            std_samples = tf.math.reduce_std(samples, axis=0).numpy()
            #
            d_2ds = TestLSP.d_2d[0,0]
            d_3ds = TestLSP.d_3d
            w = TestLSP.rma_w
            h = TestLSP.rma_h
            samples_ref = np.array([pathloss(model, submodel, d_2d, d_3d, fc,\
                 h_bs, h_ut, h, w) for d_2d, d_3d in zip(d_2ds, d_3ds)])
            #
            max_err = np.max(np.abs(mean_samples-samples_ref))
            self.assertLessEqual(max_err, TestLSP.MAX_ERR_PATHLOSS_MEAN,
                f"{model}:{submodel}")
            max_err = np.max(np.abs(std_samples-pathloss_std(model, submodel)))
            self.assertLessEqual(max_err, TestLSP.MAX_ERR_PATHLOSS_STD,
                f"{model}:{submodel}")
        elif model == 'umi':
            if submodel == 'o2i':
                for loss_model in ('low', 'high'):
                    samples = TestLSP.pathlosses[model][submodel+'-'+loss_model]
                    mean_samples = tf.reduce_mean(samples, axis=0).numpy()
                    std_samples = tf.math.reduce_std(samples, axis=0).numpy()
                    #
                    d_2ds = TestLSP.d_2d[0,0]
                    d_3ds = TestLSP.d_3d
                    samples_ref = np.array([pathloss(model, submodel, d_2d, d_3d,\
                        fc, h_bs, h_ut, loss_model) for d_2d, d_3d in zip(d_2ds, d_3ds)])
                    #
                    max_err = np.max(np.abs(mean_samples-samples_ref))
                    self.assertLessEqual(max_err, TestLSP.MAX_ERR_PATHLOSS_MEAN,
                        f"{model}:{submodel}")
                    max_err = np.max(np.abs(std_samples-pathloss_std(model, submodel, loss_model)))
                    self.assertLessEqual(max_err, TestLSP.MAX_ERR_PATHLOSS_STD,
                        f"{model}:{submodel}")
            else:
                samples = TestLSP.pathlosses[model][submodel]
                mean_samples = tf.reduce_mean(samples, axis=0).numpy()
                std_samples = tf.math.reduce_std(samples, axis=0).numpy()
                #
                d_2ds = TestLSP.d_2d[0,0]
                d_3ds = TestLSP.d_3d
                samples_ref = np.array([pathloss(model, submodel, d_2d, d_3d,\
                    fc, h_bs, h_ut) for d_2d, d_3d in zip(d_2ds, d_3ds)])
                #
                max_err = np.max(np.abs(mean_samples-samples_ref))
                self.assertLessEqual(max_err, TestLSP.MAX_ERR_PATHLOSS_MEAN,
                    f"{model}:{submodel}")
                max_err = np.max(np.abs(std_samples-pathloss_std(model, submodel)))
                self.assertLessEqual(max_err, TestLSP.MAX_ERR_PATHLOSS_STD,
                    f"{model}:{submodel}")
        elif model == 'uma':
            if submodel == 'o2i':
                for loss_model in ('low', 'high'):
                    samples = TestLSP.pathlosses[model][submodel+'-'+loss_model]
                    mean_samples = tf.reduce_mean(samples, axis=0).numpy()
                    std_samples = tf.math.reduce_std(samples, axis=0).numpy()
                    #
                    d_2ds = TestLSP.d_2d[0,0]
                    d_3ds = TestLSP.d_3d
                    samples_ref = np.array([pathloss(model, submodel, d_2d, d_3d,\
                        fc, h_bs, h_ut, loss_model) for d_2d, d_3d in zip(d_2ds, d_3ds)])
                    #
                    max_err = np.max(np.abs(mean_samples-samples_ref))
                    self.assertLessEqual(max_err, TestLSP.MAX_ERR_PATHLOSS_MEAN,
                        f"{model}:{submodel}")
                    max_err = np.max(np.abs(std_samples-pathloss_std(model, submodel, loss_model)))
                    self.assertLessEqual(max_err, TestLSP.MAX_ERR_PATHLOSS_STD,
                        f"{model}:{submodel}")
            else:
                samples = TestLSP.pathlosses[model][submodel]
                mean_samples = tf.reduce_mean(samples, axis=0).numpy()
                std_samples = tf.math.reduce_std(samples, axis=0).numpy()
                #
                d_2ds = TestLSP.d_2d[0,0]
                d_3ds = TestLSP.d_3d
                samples_ref = np.array([pathloss(model, submodel, d_2d, d_3d,\
                    fc, h_bs, h_ut) for d_2d, d_3d in zip(d_2ds, d_3ds)])
                #
                max_err = np.max(np.abs(mean_samples-samples_ref))
                self.assertLessEqual(max_err, TestLSP.MAX_ERR_PATHLOSS_MEAN,
                    f"{model}:{submodel}")
                max_err = np.max(np.abs(std_samples-pathloss_std(model, submodel)))
                self.assertLessEqual(max_err, TestLSP.MAX_ERR_PATHLOSS_STD,
                    f"{model}:{submodel}")
