
#
# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0#

import unittest
import numpy as np
import tensorflow as tf

from sionna.phy import config
from sionna.sys.utils import is_scheduled_in_slot, get_pathloss
from sionna.sys import gen_hexgrid_topology, spread_across_subcarriers
from sionna.phy.utils import sample_bernoulli
from sionna.phy.channel.tr38901 import UMi, PanelArray
from sys_utils import get_stream_management, get_pathloss_xla
from sionna.phy.channel import subcarrier_frequencies, cir_to_ofdm_channel


class TestUtils(unittest.TestCase):

    def test_is_scheduled(self):
        r"""
        Validate sionna.sys.utils.is_scheduled_in_slot function
        """
        shape_per_stream = [5, 3, 4, 10, 2]
        p_per_stream = .02

        sinr = config.tf_rng.uniform(
            shape=shape_per_stream,
            minval=1, maxval=50, dtype=tf.int32)
        mask = sample_bernoulli(shape_per_stream, p=p_per_stream)
        sinr = tf.where(
            mask, sinr, tf.cast(0., sinr.dtype))

        self.assertTrue(np.all(is_scheduled_in_slot(
            sinr=sinr).numpy() ==
            (np.sum(sinr.numpy(), axis=(-1, -3, -4)) > 0)))

    def test_pathloss(self):
        """
        Compare sys.utils.get_pathloss with its Numpy version
        """

        precision = 'double'
        direction = 'downlink'
        num_ut_per_sector = 3
        batch_size = 2
        num_time_samples = 3

        num_subcarriers = 32
        subcarrier_spacing = 15e3
        frequencies = subcarrier_frequencies(num_subcarriers,
                                             subcarrier_spacing,
                                             precision=precision)

        # Create antenna arrays
        bs_array = PanelArray(num_rows_per_panel=3,
                              num_cols_per_panel=3,
                              polarization='dual',
                              polarization_type='VH',
                              antenna_pattern='38.901',
                              carrier_frequency=3.5e9,
                              precision=precision)

        ut_array = PanelArray(num_rows_per_panel=1,
                              num_cols_per_panel=1,
                              polarization='single',
                              polarization_type='V',
                              antenna_pattern='omni',
                              carrier_frequency=3.5e9,
                              precision=precision)

        # Create channel model
        channel_model = UMi(carrier_frequency=3.5e9,
                            o2i_model='low',
                            ut_array=ut_array,
                            bs_array=bs_array,
                            direction=direction,
                            enable_pathloss=True,
                            enable_shadow_fading=True,
                            precision=precision)

        # Generate hexagonal multi-cell topology
        topology = gen_hexgrid_topology(batch_size=batch_size,
                                        num_rings=1,
                                        num_ut_per_sector=num_ut_per_sector,
                                        scenario='umi',
                                        los=True,
                                        precision=precision)
        channel_model.set_topology(*topology)

        # Compute channel taps and delays
        # a: [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_steps]
        a, tau = channel_model(num_time_samples=num_time_samples,
                               sampling_frequency=1)

        # Convert to channel matrix via OFDM waveform
        # [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_time_samples, num_subcarriers]
        h_freq = cir_to_ofdm_channel(frequencies,
                                     a,
                                     tau,
                                     normalize=False)
        # Normalize power across subcarriers
        norm_fact = tf.sqrt(tf.cast(num_subcarriers, tf.float32))
        h_freq = h_freq / tf.cast(norm_fact, h_freq.dtype)

        # Generate TX power
        num_rx, num_tx, num_tx_ant = a.shape[-6], \
            a.shape[-4], a.shape[-3]

        stream_management, _ = get_stream_management(
            direction, num_rx, num_tx, 1)
        # [num_rx, num_tx]
        rx_tx_association = stream_management.rx_tx_association
        # Rx power
        # [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_time_samples, num_subcarriers]
        rx_power_np = np.abs(h_freq.numpy())**2

        # Pathloss (Numpy version)

        ##############
        # Average subcarriers
        
        # [batch_size, num_rx, num_tx, num_time_steps]
        pathloss_all_pairs_avg_np = 1 / np.mean(
            rx_power_np, axis=(2, 4, 6))
        # # [batch_size, num_time_samples, num_rx, num_tx]
        # pathloss_all_pairs_avg_np = np.transpose(pathloss_all_pairs_avg_np, [0, 3, 1, 2])

        # Extract pathloss of serving cells
        # [1, num_rx, num_tx, 1]
        rx_tx_association = rx_tx_association[np.newaxis, ..., np.newaxis]
        # [batch_size, num_time_samples, num_rx, num_tx]
        rx_tx_association = np.tile(
            rx_tx_association, [batch_size, 1, 1, num_time_samples])
        pathloss_serving_tx_avg_np = pathloss_all_pairs_avg_np[rx_tx_association == 1]
        
        for fun in [get_pathloss, get_pathloss_xla]:
            # Compare against the Numpy version
            pathloss_all_pairs_avg, pathloss_serving_tx_avg = fun(
                h_freq,
                rx_tx_association=tf.convert_to_tensor(
                    stream_management.rx_tx_association),
                precision=precision)
            self.assertAlmostEqual(
                np.sum(np.abs(10*np.log10(pathloss_all_pairs_avg.numpy()) -
                              10*np.log10(pathloss_all_pairs_avg_np))), 0, delta=0.02)

            self.assertAlmostEqual(
                np.sum(np.abs(10*np.log10(pathloss_serving_tx_avg.numpy().flatten()) -
                              10*np.log10(pathloss_serving_tx_avg_np))),
                0,
                delta=0.02)

    def test_spread_across_subcarriers(self):
        
        batch_size = [6, 5]
        lbs = len(batch_size)
        num_ut = 10
        num_ofdm_sym = 3
        num_subcarriers = 5
        num_streams_per_ut = 2
        precision = 'double'

        tx_power_per_ut = config.tf_rng.uniform(
            batch_size + [num_ofdm_sym, num_ut], minval=1, maxval=6)

        for p in [.1, .8]:

            is_scheduled = sample_bernoulli(batch_size +
                                            [num_ofdm_sym, num_subcarriers, num_ut,
                                                num_streams_per_ut],
                                            p=p)

            # Whether a UE is scheduled at all
            # [..., num_ofdm_sym, num_ut]
            ut_is_scheduled = is_scheduled.numpy().sum(axis=(-3, -1)) > 0

            # [..., num_tx, num_streams_per_tx, num_ofdm_sym, num_subcarriers]
            tx_power = spread_across_subcarriers(
                tx_power_per_ut,
                is_scheduled,
                precision=precision)

            is_scheduled = tf.transpose(is_scheduled, list(range(lbs)) + [lbs +2, lbs+3, lbs, lbs+1])
            # Tx power must be positive when a UT is scheduled, 0 otherwise
            self.assertEqual(
                tf.reduce_all((tx_power > 0) == is_scheduled),
                True)

            # Retrieve per-UT tx power when summing across streams and
            # subcarriers for scheduled UTs
            # [..., num_tx, num_ofdm_sym]
            tx_power_per_ut_tf = tf.reduce_sum(tx_power, axis=[-3, -1]).numpy()
            tx_power_per_ut_tf = tf.experimental.numpy.swapaxes(tx_power_per_ut_tf, -1, -2)
            power_diff = tx_power_per_ut_tf[ut_is_scheduled] - \
                tx_power_per_ut.numpy()[ut_is_scheduled]
            self.assertAlmostEqual(tf.reduce_sum(
                tf.abs(power_diff)), 0, delta=1e-5)
            
