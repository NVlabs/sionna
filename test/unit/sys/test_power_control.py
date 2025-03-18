#
# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0#

import unittest
import numpy as np
import tensorflow as tf

from sionna.phy import config, dtypes
from sionna.sys import open_loop_uplink_power_control, downlink_fair_power_control
from sionna.phy.utils import sample_bernoulli, db_to_lin, dbm_to_watt
from sys_utils import open_loop_uplink_power_control_xla, downlink_fair_power_control_xla


class TestPowerControl(unittest.TestCase):

    def test_uplink_open_loop(self):
        """Validate the per-UT power produced by sys.open_loop_uplink_power_control"""
        batch_size = 2
        num_ut = 1000
        precision = 'double'

        # N. allocated REs
        num_allocated_re = config.tf_rng.uniform(
            [batch_size, num_ut], minval=1, maxval=30, dtype=tf.int32)
        ut_is_scheduled = sample_bernoulli(num_allocated_re.shape, p=.8)
        num_allocated_re = num_allocated_re * \
            tf.cast(ut_is_scheduled, num_allocated_re.dtype)

        # Pathloss
        pathloss_db = config.tf_rng.uniform([batch_size, num_ut],
                                            minval=70, maxval=130)
        pathloss = db_to_lin(pathloss_db, precision=precision)

        # ULPC parameters
        alpha = config.tf_rng.uniform([batch_size, num_ut],
                                      minval=.5,
                                      maxval=1)
        p0_dbm = config.tf_rng.uniform([batch_size, num_ut],
                                       minval=-110.,
                                       maxval=-80)
        max_power_dbm = 26.

        for fun in [open_loop_uplink_power_control_xla,
                    open_loop_uplink_power_control]:
            # Uplink power control
            tx_power_tf = fun(
                pathloss,
                num_allocated_re,
                alpha=alpha,
                p0_dbm=p0_dbm,
                precision=precision).numpy()

            # Per-UT power control via Numpy
            # [batch_size, num_ut]
            num_allocated_prb = np.ceil(num_allocated_re / 12)
            tx_power_np_dbm = np.zeros([batch_size, num_ut])
            tx_power_np_dbm[ut_is_scheduled] = np.minimum(
                alpha[ut_is_scheduled] * pathloss_db.numpy()[ut_is_scheduled] +
                p0_dbm[ut_is_scheduled] + 10 *
                np.log10(num_allocated_prb[ut_is_scheduled]),
                max_power_dbm)

            # Convert to Watt
            tx_power_np = np.zeros([batch_size, num_ut])
            tx_power_np[ut_is_scheduled] = np.power(
                10, tx_power_np_dbm[ut_is_scheduled]/10) / 1e3

            # np/tf versions must be positive at the same positions
            self.assertEqual(
                (np.abs((tx_power_np > 0)*1 - (tx_power_tf > 0)*1)).sum(), 0)

            # Compute tf/np difference
            error = np.mean(np.abs(tx_power_np - tx_power_tf))

            self.assertAlmostEqual(error, 0, delta=1e-3)

    def test_downlink_fair(self):
        """
        Test ~sionna.sys.downlink_fair_power_control

        """
        precision = 'double'
        rdtype = dtypes[precision]["tf"]["rdtype"]

        batch_size = [2, 3]
        num_ut = 30

        interference_plus_noise_db = config.tf_rng.uniform(
            batch_size + [num_ut], minval=-120, maxval=-115)
        interference_plus_noise = db_to_lin(interference_plus_noise_db)

        # Note: since minval=0, some UTs are not scheduled
        num_resources = config.tf_rng.uniform(
            batch_size + [num_ut], minval=0, maxval=5, dtype=tf.int32)
        num_resources = tf.cast(num_resources, rdtype)

        pathloss_db = config.tf_rng.uniform(
            batch_size + [num_ut], minval=70, maxval=140)
        pathloss = db_to_lin(pathloss_db, precision=precision)

        bs_max_power_dbm = config.tf_rng.uniform(
            batch_size, minval=54, maxval=58)
        max_power_bs = dbm_to_watt(bs_max_power_dbm).numpy()

        # Channel quality
        cq = (1 / (pathloss * tf.cast(interference_plus_noise, rdtype))).numpy()

        # N. scheduled UTs
        n_scheduled_uts = tf.reduce_sum(
            tf.cast(num_resources > 0, rdtype), axis=-1)

        for guaranteed_power_ratio in [0, .5, 1]:
            for fairness in [0., 1., 3.]:
                for fun in [downlink_fair_power_control,
                            downlink_fair_power_control_xla]:
                    tx_power_tf, _, mu_inv_star = fun(
                        pathloss,
                        interference_plus_noise,
                        num_resources,
                        bs_max_power_dbm=bs_max_power_dbm,
                        fairness=fairness,
                        guaranteed_power_ratio=guaranteed_power_ratio,
                        return_lagrangian=True,
                        precision=precision)

                    # If num_resources=0 then power must be 0
                    power_must_be0 = tf.gather_nd(
                        tx_power_tf, tf.where(num_resources == 0))
                    err_pow0 = tf.abs(tf.reduce_sum(power_must_be0))
                    self.assertEqual(err_pow0.numpy(), 0)

                    # For scheduled UTs, power >= guaranteed power

                    # Min power per UT
                    min_power_per_ut = guaranteed_power_ratio * \
                        dbm_to_watt(bs_max_power_dbm, precision=precision) / \
                        n_scheduled_uts
                    # Min power for each resource of each UT
                    # [..., num_ut]
                    min_power_per_res_ut = tf.expand_dims(
                        min_power_per_ut, axis=-1)
                    min_power_per_res_ut = tf.tile(
                        min_power_per_res_ut, [1]*len(batch_size) + [num_ut])
                    min_power_per_res_ut = min_power_per_res_ut / num_resources
                    min_power_per_res_ut = tf.where(num_resources > 0,
                                                    min_power_per_res_ut,
                                                    tf.cast(0., rdtype))
                    power_must_be_guaranteed = tf.gather_nd(
                        tx_power_tf, tf.where(num_resources > 0))
                    self.assertTrue(tf.reduce_all((power_must_be_guaranteed >=
                                                   tf.gather_nd(min_power_per_res_ut,
                                                                tf.where(num_resources > 0)) -
                                                   tf.cast(1e-5, rdtype))
                                                  ))

                    # Power constraint
                    err_constr = tf.abs(tf.reduce_sum(tx_power_tf, axis=-1) -
                                        dbm_to_watt(bs_max_power_dbm, precision=precision)).numpy()
                    self.assertAlmostEqual(np.mean(err_constr), 0, delta=1e-3)

                    if fairness == 0:
                        # Compute optimal analytical solution
                        tx_power_np = np.maximum(
                            mu_inv_star.numpy()[..., np.newaxis] - 1 / cq, min_power_per_res_ut.numpy())
                        tx_power_np *= num_resources.numpy()
                        err_p_opt = np.mean(np.abs(tx_power_tf - tx_power_np))
                        self.assertAlmostEqual(err_p_opt, 0, delta=1e-4)
                    elif guaranteed_power_ratio < 1:
                        # Verify KKT conditions
                        # Note: at guaranteed_power_ratio=1, all UTs are at min power,
                        # so check is useless

                        tx_power_tf = tx_power_tf.numpy()
                        ut_is_not_at_min_power = tx_power_tf > \
                            min_power_per_ut.numpy()[..., np.newaxis] + 1e-5
                        # power per resource
                        tx_power_tf[num_resources > 0] = \
                            tx_power_tf[num_resources > 0] / \
                            num_resources[num_resources > 0]
                        term = (1 + tx_power_tf * cq)
                        term1 = np.power(
                            num_resources * np.log(term), fairness) * term
                        err_kkt = cq * mu_inv_star[..., np.newaxis] - term1
                        # Consider only UTs i) scheduled and ii) not at min power
                        ind_to_consider = (
                            num_resources > 0) & ut_is_not_at_min_power
                        err_kkt_rel = err_kkt[ind_to_consider] / \
                            np.linalg.norm(term1[ind_to_consider])

                        err_kkt_rel_mean = np.mean(np.abs(err_kkt_rel))
                        self.assertAlmostEqual(err_kkt_rel_mean, 0, delta=1e-2)
