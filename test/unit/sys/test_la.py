
#
# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0#

import unittest
import numpy as np
import os
import tensorflow as tf

from sionna.phy import config, dtypes, Block
from sionna.phy.utils import db_to_lin, lin_to_db, sample_bernoulli
from sionna.sys import PHYAbstraction, InnerLoopLinkAdaptation
from sys_utils import MAC, gen_num_allocated_re, SINREffFeedback


class TestILLA(unittest.TestCase):

    def test_bler_smaller_target(self):
        r"""
        Checks that BLER stays below the target 
        """
        # Parameters
        batch_size = 30
        num_ofdm_symbols = 3
        num_ut = 100
        num_subcarriers = 30
        num_streams_per_ut = 2
        mcs_table_index = 1
        mcs_category = 0
        bler_target = .1
        prob_being_scheduled = 1

        # Instantiate the PHY abstraction object
        phy_abs = PHYAbstraction()

        # Generate random SINR
        sinr_db = config.tf_rng.uniform([batch_size,
                                         num_ofdm_symbols,
                                         num_subcarriers,
                                         num_ut,
                                         num_streams_per_ut],
                                        minval=-5,
                                        maxval=30)
        ten = tf.cast(10, sinr_db.dtype)
        sinr = tf.pow(ten, sinr_db / ten)

        # Generate random effective SINR
        sinr_eff_db = config.tf_rng.uniform([batch_size,
                                             num_ut],
                                            minval=-5,
                                            maxval=30)
        ten = tf.cast(10, sinr_eff_db.dtype)
        sinr_eff = tf.pow(ten, sinr_eff_db / ten)

        # Generate Transport Blocks with more than 1 Code Block 
        num_allocated_re = \
            gen_num_allocated_re(prob_being_scheduled,
                                      [batch_size, num_ut],
                                      bounds=[200, 20000])
        sinr_eff = tf.where(num_allocated_re > 0,
                            sinr_eff, tf.cast(0., sinr_eff.dtype))
        # ILLA
        illa = InnerLoopLinkAdaptation(phy_abs)

        mcs_index, lowest_available_mcs = illa(
            sinr_eff=sinr_eff,
            num_allocated_re=num_allocated_re,
            bler_target=bler_target,
            return_lowest_available_mcs=True)

        # Compute BLER via PHYAbstraction
        *_, tbler, _ = phy_abs(mcs_index,
                                         sinr_eff=sinr_eff,
                                         num_allocated_re=num_allocated_re,
                                         mcs_table_index=mcs_table_index,
                                         mcs_category=mcs_category)

        # The BLER at the MCS indices higher than the lowest available index
        # must not exceed the bler_target
        tbler_should_be_lower_target = tf.gather_nd(
            tbler,
            tf.where(mcs_index > lowest_available_mcs))

        self.assertTrue(tf.reduce_all(
            tbler_should_be_lower_target <= bler_target))

        # Call OLLA in SINR per stream mode
        mcs_index, lowest_available_mcs = illa(
            sinr=sinr,
            mcs_table_index=mcs_table_index,
            mcs_category=mcs_category,
            bler_target=bler_target,
            return_lowest_available_mcs=True)


class TestOLLA(unittest.TestCase):
    def test_convergence(self):
        r"""
        Checks that BLER converges to its target value 
        """
        bler_target = .17

        sinr_db_bounds = (4, 17)
        delta_up = 1
        num_ut = 40
        batch_size = 100
        prob_being_scheduled = .5
        prob_feedback = .5
        num_slots = 300
        precision = 'double'

        # Initialize SINR feedback object
        sinr_obj = SINREffFeedback(shape=[batch_size, num_ut],
                                   prob_feedback=prob_feedback,
                                   bounds=sinr_db_bounds,
                                   precision=precision)

        # Initalize MAC block
        mac = MAC(sinr_eff_init=db_to_lin(sinr_obj.true_val_db),
                  bler_target=bler_target,
                  olla_delta_up=delta_up,
                  precision=precision)

        # Initialize historical data
        harq_feedback_hist = np.zeros((batch_size, num_ut, num_slots))

        # Initialize empty HARQ and SINR feedback
        sinr_eff_feedback = tf.zeros([batch_size, num_ut], mac.rdtype)
        harq_feedback = tf.cast(
            tf.fill([batch_size, num_ut], -1), dtype=tf.int32)

        for slot in range(num_slots):
            # Scheduler allocates streams
            # Generate Transport Blocks with more than 1 Code Block
            num_allocated_re = gen_num_allocated_re(
                prob_being_scheduled,
                shape=[batch_size, num_ut],
                bounds=[24, 20000])

            # Link Adaptation + PHY Abstraction
            *_, harq_feedback, _ = \
                mac(num_allocated_re,
                    harq_feedback,
                    sinr_eff_true=db_to_lin(sinr_obj.true_val_db),
                    sinr_eff_feedback=sinr_eff_feedback)

            # SINR eff feedback
            sinr_eff_feedback = sinr_obj()

            # Record historical data
            harq_feedback_hist[:, :, slot] = harq_feedback.numpy()

        harq_feedback_hist = np.where(
            harq_feedback_hist == -1, np.nan, harq_feedback_hist)

        # Achieved BLER
        bler_achieved = 1 - np.nanmean(harq_feedback_hist, axis=2)

        bler_achieved_ = np.sort(bler_achieved.flatten())
        # Take 20-th and 80-th percentile
        x = [.1, .9]
        perc = [bler_achieved_[int(x[0]*len(bler_achieved_))],
                bler_achieved_[int(x[1]*len(bler_achieved_))]]
        print(f'{perc = }')
        self.assertAlmostEqual(perc[0], bler_target, delta=.005)
        self.assertAlmostEqual(perc[1], bler_target, delta=.005)
