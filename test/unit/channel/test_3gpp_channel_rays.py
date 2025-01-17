#
# SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
import pytest
import unittest
import numpy as np
from scipy.stats import kstest
import tensorflow as tf
import sionna
from sionna import config
from channel_test_utils import *

@pytest.mark.usefixtures("only_gpu")
class TestRays(unittest.TestCase):
    r"""Test the rays generated for 3GPP system level simulations
    """

    # Batch size used to check the LSP distribution
    BATCH_SIZE = 100000

    # Carrier frequency
    CARRIER_FREQUENCY = 3.5e9 # Hz

    # Maximum allowed deviation for distance calculation (relative error)
    MAX_ERR = 3e-2

    # Heigh of UTs
    H_UT = 1.5

    # Heigh of BSs
    H_BS = 35.0

    def setUpClass():
        r"""Sample rays from all LoS and NLoS channel models for testing"""

        batch_size = TestRays.BATCH_SIZE
        fc = TestRays.CARRIER_FREQUENCY

        # UT and BS arrays have no impact on LSP
        # However, these are needed to instantiate the model
        bs_array = sionna.channel.tr38901.PanelArray(num_rows_per_panel=1,
                                                    num_cols_per_panel=1,
                                                    polarization='single',
                                                    polarization_type='V',
                                                    antenna_pattern='38.901',
                                                    carrier_frequency=fc,
                                                    dtype=tf.complex128)
        ut_array = sionna.channel.tr38901.PanelArray(num_rows_per_panel=1,
                                                    num_cols_per_panel=1,
                                                    polarization='single',
                                                    polarization_type='V',
                                                    antenna_pattern='38.901',
                                                    carrier_frequency=fc,
                                                    dtype=tf.complex128)

        # The following quantities have no impact on the rays, but are
        # required to instantiate models
        ut_orientations = config.tf_rng.uniform([batch_size, 1, 3],
                                            -sionna.PI, sionna.PI,
                                            dtype=tf.float64)
        bs_orientations = config.tf_rng.uniform([batch_size, 1, 3],
                                            -sionna.PI, sionna.PI,
                                            dtype=tf.float64)
        ut_velocities = config.tf_rng.uniform([batch_size, 1, 3], -1.0, 1.0,
                                            dtype=tf.float64)

        # 1 UT and 1 BS
        ut_loc = generate_random_loc(batch_size, 1, (100,2000), (100,2000),
                                        (1.5, 1.5), share_loc=True,
                                        dtype=tf.float64)
        bs_loc = generate_random_loc(batch_size, 1, (0,100), (0,100),
                                        (35.0, 35.0), share_loc=True,
                                        dtype=tf.float64)

        # Force the LSPs
        TestRays.ds = np.power(10.0,-7.49)
        ds_ = tf.fill([batch_size, 1, 1], tf.cast(TestRays.ds, tf.float64))
        TestRays.asd = np.power(10.0, 0.90)
        asd_ = tf.fill([batch_size, 1, 1], tf.cast(TestRays.asd, tf.float64))
        TestRays.asa = np.power(10.0, 1.52)
        asa_ = tf.fill([batch_size, 1, 1], tf.cast(TestRays.asa, tf.float64))
        TestRays.zsa = np.power(10.0, 0.47)
        zsa_ = tf.fill([batch_size, 1, 1], tf.cast(TestRays.zsa, tf.float64))
        TestRays.zsd = np.power(10.0, -0.29)
        zsd_ = tf.fill([batch_size, 1, 1], tf.cast(TestRays.zsd, tf.float64))
        TestRays.k = np.power(10.0, 7./10.)
        k_ = tf.fill([batch_size, 1, 1], tf.cast(TestRays.k, tf.float64))
        sf_ = tf.zeros([batch_size, 1, 1], tf.float64)
        lsp = sionna.channel.tr38901.LSP(ds_, asd_, asa_, sf_, k_, zsa_, zsd_)

        # Store the sampled rays
        TestRays.delays = {}
        TestRays.powers = {}
        TestRays.aoa = {}
        TestRays.aod = {}
        TestRays.zoa = {}
        TestRays.zod = {}
        TestRays.xpr = {}
        TestRays.num_clusters = {}
        TestRays.los_aoa = {}
        TestRays.los_aod = {}
        TestRays.los_zoa = {}
        TestRays.los_zod = {}
        TestRays.mu_log_zsd = {}

        #################### RMa
        TestRays.delays['rma'] = {}
        TestRays.powers['rma'] = {}
        TestRays.aoa['rma'] = {}
        TestRays.aod['rma'] = {}
        TestRays.zoa['rma'] = {}
        TestRays.zod['rma'] = {}
        TestRays.xpr['rma'] = {}
        TestRays.num_clusters['rma'] = {}
        TestRays.los_aoa['rma'] = {}
        TestRays.los_aod['rma'] = {}
        TestRays.los_zoa['rma'] = {}
        TestRays.los_zod['rma'] = {}
        TestRays.mu_log_zsd['rma'] = {}
        scenario = sionna.channel.tr38901.RMaScenario(fc, ut_array, bs_array,
                                                        "downlink",
                                                        dtype=tf.complex128)
        ray_sampler = sionna.channel.tr38901.RaysGenerator(scenario)

        #### LoS
        in_state = generate_random_bool(batch_size, 1, 0.0)
        scenario.set_topology(ut_loc, bs_loc, ut_orientations, bs_orientations,
                                            ut_velocities, in_state, los=True)
        ray_sampler.topology_updated_callback()
        rays = ray_sampler(lsp)
        TestRays.delays['rma']['los'] = tf.squeeze(rays.delays).numpy()
        TestRays.powers['rma']['los'] = tf.squeeze(rays.powers).numpy()
        TestRays.aoa['rma']['los'] = tf.squeeze(rays.aoa).numpy()
        TestRays.aod['rma']['los'] = tf.squeeze(rays.aod).numpy()
        TestRays.zoa['rma']['los'] = tf.squeeze(rays.zoa).numpy()
        TestRays.zod['rma']['los'] = tf.squeeze(rays.zod).numpy()
        TestRays.xpr['rma']['los'] = tf.squeeze(rays.xpr).numpy()
        TestRays.num_clusters['rma']['los'] = 11
        TestRays.los_aoa['rma']['los'] = scenario.los_aoa.numpy()
        TestRays.los_aod['rma']['los'] = scenario.los_aod.numpy()
        TestRays.los_zoa['rma']['los'] = scenario.los_zoa.numpy()
        TestRays.los_zod['rma']['los'] = scenario.los_zod.numpy()
        TestRays.mu_log_zsd['rma']['los'] = scenario.lsp_log_mean[:,0,0,6].numpy()

        #### NLoS
        in_state = generate_random_bool(batch_size, 1, 0.0)
        scenario.set_topology(ut_loc, bs_loc, ut_orientations, bs_orientations,
                                            ut_velocities, in_state, los=False)
        ray_sampler.topology_updated_callback()
        rays = ray_sampler(lsp)
        TestRays.delays['rma']['nlos'] = tf.squeeze(rays.delays).numpy()
        TestRays.powers['rma']['nlos'] = tf.squeeze(rays.powers).numpy()
        TestRays.aoa['rma']['nlos'] = tf.squeeze(rays.aoa).numpy()
        TestRays.aod['rma']['nlos'] = tf.squeeze(rays.aod).numpy()
        TestRays.zoa['rma']['nlos'] = tf.squeeze(rays.zoa).numpy()
        TestRays.zod['rma']['nlos'] = tf.squeeze(rays.zod).numpy()
        TestRays.xpr['rma']['nlos'] = tf.squeeze(rays.xpr).numpy()
        TestRays.num_clusters['rma']['nlos'] = 10
        TestRays.los_aoa['rma']['nlos'] = scenario.los_aoa.numpy()
        TestRays.los_aod['rma']['nlos'] = scenario.los_aod.numpy()
        TestRays.los_zoa['rma']['nlos'] = scenario.los_zoa.numpy()
        TestRays.los_zod['rma']['nlos'] = scenario.los_zod.numpy()
        TestRays.mu_log_zsd['rma']['nlos'] = scenario.lsp_log_mean[:,0,0,6].numpy()

        #### O2I
        in_state = generate_random_bool(batch_size, 1, 1.0)
        scenario.set_topology(ut_loc, bs_loc, ut_orientations, bs_orientations,
                                            ut_velocities, in_state)
        ray_sampler.topology_updated_callback()
        rays = ray_sampler(lsp)
        TestRays.delays['rma']['o2i'] = tf.squeeze(rays.delays).numpy()
        TestRays.powers['rma']['o2i'] = tf.squeeze(rays.powers).numpy()
        TestRays.aoa['rma']['o2i'] = tf.squeeze(rays.aoa).numpy()
        TestRays.aod['rma']['o2i'] = tf.squeeze(rays.aod).numpy()
        TestRays.zoa['rma']['o2i'] = tf.squeeze(rays.zoa).numpy()
        TestRays.zod['rma']['o2i'] = tf.squeeze(rays.zod).numpy()
        TestRays.xpr['rma']['o2i'] = tf.squeeze(rays.xpr).numpy()
        TestRays.num_clusters['rma']['o2i'] = 10
        TestRays.los_aoa['rma']['o2i'] = scenario.los_aoa.numpy()
        TestRays.los_aod['rma']['o2i'] = scenario.los_aod.numpy()
        TestRays.los_zoa['rma']['o2i'] = scenario.los_zoa.numpy()
        TestRays.los_zod['rma']['o2i'] = scenario.los_zod.numpy()
        TestRays.mu_log_zsd['rma']['o2i'] = scenario.lsp_log_mean[:,0,0,6].numpy()

        #################### UMi
        TestRays.delays['umi'] = {}
        TestRays.powers['umi'] = {}
        TestRays.aoa['umi'] = {}
        TestRays.aod['umi'] = {}
        TestRays.zoa['umi'] = {}
        TestRays.zod['umi'] = {}
        TestRays.xpr['umi'] = {}
        TestRays.num_clusters['umi'] = {}
        TestRays.los_aoa['umi'] = {}
        TestRays.los_aod['umi'] = {}
        TestRays.los_zoa['umi'] = {}
        TestRays.los_zod['umi'] = {}
        TestRays.mu_log_zsd['umi'] = {}
        scenario = sionna.channel.tr38901.UMiScenario(  fc, 'low',
                                                        ut_array, bs_array,
                                                        "downlink",
                                                        dtype=tf.complex128)
        ray_sampler = sionna.channel.tr38901.RaysGenerator(scenario)

        #### LoS
        in_state = generate_random_bool(batch_size, 1, 0.0)
        scenario.set_topology(ut_loc, bs_loc, ut_orientations, bs_orientations,
                                            ut_velocities, in_state, los=True)
        ray_sampler.topology_updated_callback()
        rays = ray_sampler(lsp)
        TestRays.delays['umi']['los'] = tf.squeeze(rays.delays).numpy()
        TestRays.powers['umi']['los'] = tf.squeeze(rays.powers).numpy()
        TestRays.aoa['umi']['los'] = tf.squeeze(rays.aoa).numpy()
        TestRays.aod['umi']['los'] = tf.squeeze(rays.aod).numpy()
        TestRays.zoa['umi']['los'] = tf.squeeze(rays.zoa).numpy()
        TestRays.zod['umi']['los'] = tf.squeeze(rays.zod).numpy()
        TestRays.xpr['umi']['los'] = tf.squeeze(rays.xpr).numpy()
        TestRays.num_clusters['umi']['los'] = 12
        TestRays.los_aoa['umi']['los'] = scenario.los_aoa.numpy()
        TestRays.los_aod['umi']['los'] = scenario.los_aod.numpy()
        TestRays.los_zoa['umi']['los'] = scenario.los_zoa.numpy()
        TestRays.los_zod['umi']['los'] = scenario.los_zod.numpy()
        TestRays.mu_log_zsd['umi']['los'] = scenario.lsp_log_mean[:,0,0,6].numpy()

        #### NLoS
        in_state = generate_random_bool(batch_size, 1, 0.0)
        scenario.set_topology(ut_loc, bs_loc, ut_orientations, bs_orientations,
                                            ut_velocities, in_state, los=False)
        ray_sampler.topology_updated_callback()
        rays = ray_sampler(lsp)
        TestRays.delays['umi']['nlos'] = tf.squeeze(rays.delays).numpy()
        TestRays.powers['umi']['nlos'] = tf.squeeze(rays.powers).numpy()
        TestRays.aoa['umi']['nlos'] = tf.squeeze(rays.aoa).numpy()
        TestRays.aod['umi']['nlos'] = tf.squeeze(rays.aod).numpy()
        TestRays.zoa['umi']['nlos'] = tf.squeeze(rays.zoa).numpy()
        TestRays.zod['umi']['nlos'] = tf.squeeze(rays.zod).numpy()
        TestRays.xpr['umi']['nlos'] = tf.squeeze(rays.xpr).numpy()
        TestRays.num_clusters['umi']['nlos'] = 19
        TestRays.los_aoa['umi']['nlos'] = scenario.los_aoa.numpy()
        TestRays.los_aod['umi']['nlos'] = scenario.los_aod.numpy()
        TestRays.los_zoa['umi']['nlos'] = scenario.los_zoa.numpy()
        TestRays.los_zod['umi']['nlos'] = scenario.los_zod.numpy()
        TestRays.mu_log_zsd['umi']['nlos'] = scenario.lsp_log_mean[:,0,0,6].numpy()

        #### O2I
        in_state = generate_random_bool(batch_size, 1, 1.0)
        scenario.set_topology(ut_loc, bs_loc, ut_orientations, bs_orientations,
                                            ut_velocities, in_state)
        ray_sampler.topology_updated_callback()
        rays = ray_sampler(lsp)
        TestRays.delays['umi']['o2i'] = tf.squeeze(rays.delays).numpy()
        TestRays.powers['umi']['o2i'] = tf.squeeze(rays.powers).numpy()
        TestRays.aoa['umi']['o2i'] = tf.squeeze(rays.aoa).numpy()
        TestRays.aod['umi']['o2i'] = tf.squeeze(rays.aod).numpy()
        TestRays.zoa['umi']['o2i'] = tf.squeeze(rays.zoa).numpy()
        TestRays.zod['umi']['o2i'] = tf.squeeze(rays.zod).numpy()
        TestRays.xpr['umi']['o2i'] = tf.squeeze(rays.xpr).numpy()
        TestRays.num_clusters['umi']['o2i'] = 12
        TestRays.los_aoa['umi']['o2i'] = scenario.los_aoa.numpy()
        TestRays.los_aod['umi']['o2i'] = scenario.los_aod.numpy()
        TestRays.los_zoa['umi']['o2i'] = scenario.los_zoa.numpy()
        TestRays.los_zod['umi']['o2i'] = scenario.los_zod.numpy()
        TestRays.mu_log_zsd['umi']['o2i'] = scenario.lsp_log_mean[:,0,0,6].numpy()

        #################### UMa
        TestRays.delays['uma'] = {}
        TestRays.powers['uma'] = {}
        TestRays.aoa['uma'] = {}
        TestRays.aod['uma'] = {}
        TestRays.zoa['uma'] = {}
        TestRays.zod['uma'] = {}
        TestRays.xpr['uma'] = {}
        TestRays.num_clusters['uma'] = {}
        TestRays.los_aoa['uma'] = {}
        TestRays.los_aod['uma'] = {}
        TestRays.los_zoa['uma'] = {}
        TestRays.los_zod['uma'] = {}
        TestRays.mu_log_zsd['uma'] = {}
        scenario = sionna.channel.tr38901.UMaScenario(  fc, 'low',
                                                        ut_array, bs_array,
                                                        "downlink",
                                                        dtype=tf.complex128)
        ray_sampler = sionna.channel.tr38901.RaysGenerator(scenario)

        #### LoS
        in_state = generate_random_bool(batch_size, 1, 0.0)
        scenario.set_topology(ut_loc, bs_loc, ut_orientations, bs_orientations,
                                            ut_velocities, in_state, los=True)
        ray_sampler.topology_updated_callback()
        rays = ray_sampler(lsp)
        TestRays.delays['uma']['los'] = tf.squeeze(rays.delays).numpy()
        TestRays.powers['uma']['los'] = tf.squeeze(rays.powers).numpy()
        TestRays.aoa['uma']['los'] = tf.squeeze(rays.aoa).numpy()
        TestRays.aod['uma']['los'] = tf.squeeze(rays.aod).numpy()
        TestRays.zoa['uma']['los'] = tf.squeeze(rays.zoa).numpy()
        TestRays.zod['uma']['los'] = tf.squeeze(rays.zod).numpy()
        TestRays.xpr['uma']['los'] = tf.squeeze(rays.xpr).numpy()
        TestRays.num_clusters['uma']['los'] = 12
        TestRays.los_aoa['uma']['los'] = scenario.los_aoa.numpy()
        TestRays.los_aod['uma']['los'] = scenario.los_aod.numpy()
        TestRays.los_zoa['uma']['los'] = scenario.los_zoa.numpy()
        TestRays.los_zod['uma']['los'] = scenario.los_zod.numpy()
        TestRays.mu_log_zsd['uma']['los'] = scenario.lsp_log_mean[:,0,0,6].numpy()

        #### NLoS
        in_state = generate_random_bool(batch_size, 1, 0.0)
        scenario.set_topology(ut_loc, bs_loc, ut_orientations, bs_orientations,
                                            ut_velocities, in_state, los=False)
        ray_sampler.topology_updated_callback()
        rays = ray_sampler(lsp)
        TestRays.delays['uma']['nlos'] = tf.squeeze(rays.delays).numpy()
        TestRays.powers['uma']['nlos'] = tf.squeeze(rays.powers).numpy()
        TestRays.aoa['uma']['nlos'] = tf.squeeze(rays.aoa).numpy()
        TestRays.aod['uma']['nlos'] = tf.squeeze(rays.aod).numpy()
        TestRays.zoa['uma']['nlos'] = tf.squeeze(rays.zoa).numpy()
        TestRays.zod['uma']['nlos'] = tf.squeeze(rays.zod).numpy()
        TestRays.xpr['uma']['nlos'] = tf.squeeze(rays.xpr).numpy()
        TestRays.num_clusters['uma']['nlos'] = 20
        TestRays.los_aoa['uma']['nlos'] = scenario.los_aoa.numpy()
        TestRays.los_aod['uma']['nlos'] = scenario.los_aod.numpy()
        TestRays.los_zoa['uma']['nlos'] = scenario.los_zoa.numpy()
        TestRays.los_zod['uma']['nlos'] = scenario.los_zod.numpy()
        TestRays.mu_log_zsd['uma']['nlos'] = scenario.lsp_log_mean[:,0,0,6].numpy()

        #### O2I
        in_state = generate_random_bool(batch_size, 1, 1.0)
        scenario.set_topology(ut_loc, bs_loc, ut_orientations, bs_orientations,
                                            ut_velocities, in_state)
        ray_sampler.topology_updated_callback()
        rays = ray_sampler(lsp)
        TestRays.delays['uma']['o2i'] = tf.squeeze(rays.delays).numpy()
        TestRays.powers['uma']['o2i'] = tf.squeeze(rays.powers).numpy()
        TestRays.aoa['uma']['o2i'] = tf.squeeze(rays.aoa).numpy()
        TestRays.aod['uma']['o2i'] = tf.squeeze(rays.aod).numpy()
        TestRays.zoa['uma']['o2i'] = tf.squeeze(rays.zoa).numpy()
        TestRays.zod['uma']['o2i'] = tf.squeeze(rays.zod).numpy()
        TestRays.xpr['uma']['o2i'] = tf.squeeze(rays.xpr).numpy()
        TestRays.num_clusters['uma']['o2i'] = 12
        TestRays.los_aoa['uma']['o2i'] = scenario.los_aoa.numpy()
        TestRays.los_aod['uma']['o2i'] = scenario.los_aod.numpy()
        TestRays.los_zoa['uma']['o2i'] = scenario.los_zoa.numpy()
        TestRays.los_zod['uma']['o2i'] = scenario.los_zod.numpy()
        TestRays.mu_log_zsd['uma']['o2i'] = scenario.lsp_log_mean[:,0,0,6].numpy()

        ###### General
        TestRays.d_2d = scenario.distance_2d[0,0,0].numpy()

    @channel_test_on_models(('rma', 'umi', 'uma'), ('los', 'nlos', 'o2i'))
    def test_delays(self, model, submodel):
        """Test ray generation: Delays"""
        num_clusters = TestRays.num_clusters[model][submodel]
        batch_size = TestRays.BATCH_SIZE
        tau = TestRays.delays[model][submodel][:,:num_clusters].flatten()
        _, ref_tau = delays(model, submodel, batch_size, num_clusters,
                                    TestRays.ds, TestRays.k)
        ref_tau = ref_tau[:,:num_clusters].flatten()
        D,_ = kstest(tau,ref_tau)
        self.assertLessEqual(D, TestRays.MAX_ERR, f"{model}:{submodel}")

    @channel_test_on_models(('rma', 'umi', 'uma'), ('los', 'nlos', 'o2i'))
    def test_powers(self, model, submodel):
        """Test ray generation: Powers"""
        num_clusters = TestRays.num_clusters[model][submodel]
        batch_size = TestRays.BATCH_SIZE
        p = self.powers[model][submodel][:,:num_clusters].flatten()
        unscaled_tau, _ = delays(model, submodel, batch_size, num_clusters,
                                    TestRays.ds, TestRays.k)
        ref_p,_ = powers(model, submodel, batch_size, num_clusters,
                unscaled_tau, self.ds, self.k)
        ref_p = ref_p[:,:num_clusters].flatten()
        D,_ = kstest(ref_p,p)
        self.assertLessEqual(D, TestRays.MAX_ERR, f"{model}:{submodel}")

    @channel_test_on_models(('rma', 'umi', 'uma'), ('los', 'nlos', 'o2i'))
    def test_aoa(self, model, submodel):
        """Test ray generation: AoA"""
        num_clusters = TestRays.num_clusters[model][submodel]
        batch_size = TestRays.BATCH_SIZE
        k = None
        if submodel == 'los':
            k = TestRays.k
        unscaled_tau, _ = delays(model, submodel, batch_size, num_clusters,
                                    TestRays.ds, TestRays.k)
        _, ref_p_angles = powers(model, submodel, batch_size, num_clusters,
            unscaled_tau, TestRays.ds, TestRays.k)
        ref_p_angles = ref_p_angles[:,:num_clusters]
        ref_samples = aoa(model, submodel, batch_size, num_clusters,
            TestRays.asa, ref_p_angles, TestRays.los_aoa[model][submodel], k)
        ref_samples = ref_samples[:,:num_clusters].flatten()
        samples = TestRays.aoa[model][submodel][:,:num_clusters].flatten()
        D,_ = kstest(ref_samples, samples)
        self.assertLessEqual(D, TestRays.MAX_ERR, f"{model}:{submodel}")

    @channel_test_on_models(('rma', 'umi', 'uma'), ('los', 'nlos', 'o2i'))
    def test_aod(self, model, submodel):
        """Test ray generation: AoD"""
        num_clusters = TestRays.num_clusters[model][submodel]
        batch_size = TestRays.BATCH_SIZE
        k = None
        if submodel == 'los':
            k = TestRays.k
        unscaled_tau, _ = delays(model, submodel, batch_size, num_clusters,
                                    TestRays.ds, TestRays.k)
        _, ref_p_angles = powers(model, submodel, batch_size, num_clusters,
                unscaled_tau, TestRays.ds, TestRays.k)
        ref_p_angles = ref_p_angles[:,:num_clusters]
        ref_samples = aod(model, submodel, batch_size, num_clusters,
            TestRays.asd, ref_p_angles, TestRays.los_aod[model][submodel], k)
        ref_samples = ref_samples[:,:num_clusters].flatten()
        samples = TestRays.aod[model][submodel][:,:num_clusters].flatten()
        D,_ = kstest(ref_samples, samples)
        self.assertLessEqual(D, TestRays.MAX_ERR, f"{model}:{submodel}")

    @channel_test_on_models(('rma', 'umi', 'uma'), ('los', 'nlos', 'o2i'))
    def test_zoa(self, model, submodel):
        """Test ray generation: ZoA"""
        num_clusters = TestRays.num_clusters[model][submodel]
        batch_size = TestRays.BATCH_SIZE
        k = None
        if submodel == 'los':
            k = TestRays.k
        unscaled_tau, _ = delays(model, submodel, batch_size, num_clusters,
                                    TestRays.ds, TestRays.k)
        _, ref_p_angles = powers(model, submodel, batch_size, num_clusters,
                    unscaled_tau, TestRays.ds, TestRays.k)
        ref_p_angles = ref_p_angles[:,:num_clusters]
        ref_samples = zoa(model, submodel, batch_size, num_clusters,
            TestRays.zsa, ref_p_angles, TestRays.los_zoa[model][submodel], k)
        ref_samples = ref_samples[:,:num_clusters].flatten()
        samples = TestRays.zoa[model][submodel][:,:num_clusters].flatten()
        D,_ = kstest(ref_samples, samples)
        self.assertLessEqual(D, TestRays.MAX_ERR, f"{model}:{submodel}")

    @channel_test_on_models(('rma', 'umi', 'uma'), ('los', 'nlos', 'o2i'))
    def test_zod(self, model, submodel):
        """Test ray generation: ZoD"""
        num_clusters = TestRays.num_clusters[model][submodel]
        batch_size = TestRays.BATCH_SIZE
        fc = TestRays.CARRIER_FREQUENCY
        d_2d = TestRays.d_2d
        h_ut = TestRays.H_UT
        mu_log_zod = TestRays.mu_log_zsd[model][submodel]
        k = None
        if submodel == 'los':
            k = TestRays.k
        unscaled_tau, _ = delays(model, submodel, batch_size, num_clusters,
                                    TestRays.ds, TestRays.k)
        _, ref_p_angles = powers(model, submodel, batch_size, num_clusters,
                    unscaled_tau, TestRays.ds, TestRays.k)
        ref_p_angles = ref_p_angles[:,:num_clusters]
        offset = zod_offset(model, submodel, fc, d_2d, h_ut)
        ref_samples = zod(model, submodel, batch_size, num_clusters,
            TestRays.zsd, ref_p_angles, TestRays.los_zod[model][submodel],
            offset, mu_log_zod, k)
        ref_samples = ref_samples[:,:num_clusters].flatten()
        samples = TestRays.zod[model][submodel][:,:num_clusters].flatten()
        D,_ = kstest(ref_samples, samples)
        self.assertLessEqual(D, TestRays.MAX_ERR, f"{model}:{submodel}")

    @channel_test_on_models(('rma', 'umi', 'uma'), ('los', 'nlos', 'o2i'))
    def test_xpr(self, model, submodel):
        """Test ray generation: XPR"""
        num_clusters = TestRays.num_clusters[model][submodel]
        batch_size = TestRays.BATCH_SIZE
        samples = TestRays.xpr[model][submodel][:,:num_clusters].flatten()
        ref_samples = xpr(model, submodel, batch_size, num_clusters)
        ref_samples = ref_samples[:,:num_clusters].flatten()
        D,_ = kstest(ref_samples, samples)
        self.assertLessEqual(D, TestRays.MAX_ERR)
