#
# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0#

import unittest
import numpy as np
import tensorflow as tf
from sionna.phy import config
from sionna.phy.utils import insert_dims
from sionna.sys import PFSchedulerSUMIMO
from sys_utils import pf_scheduler_multislot, pf_scheduler_multislot_xla


class TestPFSchedulerSUMIMO(unittest.TestCase):

    def test_first_slot(self):
        """
        Checks that at the first slot the scheduler selects the UTs with maximum
        achievable rate
        """
        batch_size = [6, 2]
        num_ut = 5
        num_freq_res = 10
        num_time_samples = 3
        num_streams = 1
        beta = .98
        precision = 'double'

        pf_sched = PFSchedulerSUMIMO(num_ut,
                                     num_freq_res,
                                     num_time_samples,
                                     batch_size=batch_size,
                                     num_streams_per_ut=num_streams,
                                     beta=beta,
                                     precision=precision)
        # XLA scheduler

        def pf_sched_xla(rate_last_slot, rate_achievable):
            return pf_sched(rate_last_slot, rate_achievable)

        # Rate achievable in the current slot
        rate_achievable = tf.Variable(
            config.tf_rng.uniform(batch_size +
                                  [num_time_samples,
                                   num_freq_res,
                                   num_ut],
                                  minval=0,
                                  maxval=10))
        # Rate achieved over last slot
        rate_last_slot = tf.Variable(
            tf.zeros(batch_size + [num_ut]))

        for sched_fun in [pf_sched, pf_sched_xla]:
            # [batch_size, num_time_samples, num_freq_res, num_ut, num_streams]
            is_scheduled = sched_fun(rate_last_slot, rate_achievable)

            uts_with_max_achievable_rate = np.argmax(
                rate_achievable.numpy(), axis=-1)

            uts_scheduled = np.reshape(np.where(
                is_scheduled.numpy() == 1)[-2],
                batch_size + [num_time_samples, num_freq_res])

            # At first slot, scheduler selects UTs with max achievable rate
            self.assertEqual(np.sum(np.abs((uts_with_max_achievable_rate -
                                            uts_scheduled))), 0)
            # Either all streams are assigned to a UT, or none
            self.assertTrue(tf.reduce_all((tf.reduce_all(is_scheduled, axis=-1) ==
                                           tf.reduce_any(is_scheduled, axis=-1))))

    def test_fairness_multislot(self):
        """
        Checks that all UTs receive the same number of allocated resources
        although their achievable rate is different
        """
        batch_size = [10]
        num_ut = 20
        num_freq_res = 10
        num_time_samples = 3
        num_streams_per_ut = 2
        beta = .98
        precision = 'double'

        num_slots = 10000

        # Achievable rate, very different across UTs
        rate_achievable_avg = config.tf_rng.uniform(
            batch_size + [num_ut],
            minval=0,
            maxval=100)

        for pf_fun in [pf_scheduler_multislot_xla]:
            pf_sched = PFSchedulerSUMIMO(num_ut,
                                         num_freq_res,
                                         num_time_samples,
                                         batch_size=batch_size,
                                         num_streams_per_ut=num_streams_per_ut,
                                         beta=beta,
                                         precision=precision)

            hist = pf_fun(pf_sched, rate_achievable_avg, num_slots)

            num_allocated_res = tf.reduce_sum(
                hist['is_scheduled'], axis=[-1, -3, -4, -(5 + len(batch_size))]).numpy()

            # Resource unbalancedness must be small
            unbalancedness = (num_allocated_res.max(
                axis=1) - num_allocated_res.min(axis=1)) / num_allocated_res.sum(axis=1)

            self.assertTrue(np.all(unbalancedness < 1e-2))
