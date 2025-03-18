
#
# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0#

import unittest
import numpy as np
import os
import tensorflow as tf

from sionna.phy import config
from sionna.sys import PHYAbstraction
from sionna.phy.utils import DeepUpdateDict, random_tensor_from_values
from sionna.phy.mimo import StreamManagement
from sionna.phy.ofdm import RZFPrecodedChannel, LMMSEPostEqualizationSINR,\
                            ResourceGrid


class TestPHYAbstraction(unittest.TestCase):

    def test_write_and_load(self):
        """ Test the SNR to BER/BLER table generation """

        sim_set_1 = {'category': {
            0:
                    {'index': {
                        1: {'MCS': [10, 24]},
                        2: {'MCS': [12]}
                    }}}}
        snr_dbs_1 = [0, 20]
        cb_sizes_1 = [50, 100, 150]
        filename = 'test.json'

        # start from no loaded table
        with self.assertWarns(UserWarning) as cm:
            phy_abs = PHYAbstraction(load_bler_tables_from='')

        # compute tables and save them to file
        table_1 = phy_abs.new_bler_table(
            snr_dbs_1,
            cb_sizes_1,
            sim_set_1,
            filename=filename,
            max_mc_iter=15,
            batch_size=10,
            verbose=False)

        # check that results have been written to file
        self.assertTrue(os.path.isfile(filename))
        self.assertTrue(os.path.getsize(filename) > 0)

        # load tables
        table_loaded = PHYAbstraction.load_table(filename)

        # check that the two tables (dumped and loaded) are equal
        for category in table_1['category']:

            for table_index in table_1['category'][category]['index']:

                for mcs in table_1['category'][category]['index'][table_index]['MCS']:
                    res_mcs = table_1['category'][category]['index'][table_index]['MCS'][mcs]
                    res_mcs1 = table_loaded['category'][category]['index'][table_index]['MCS'][mcs]

                    for a, b in zip(res_mcs['SNR_db'], res_mcs1['SNR_db']):
                        self.assertEqual(a, b)

                    for cbs in res_mcs['CBS']:

                        res = res_mcs['CBS'][cbs]
                        res1 = res_mcs1['CBS'][cbs]

                        for a, b in zip(res['BLER'], res1['BLER']):
                            self.assertEqual(a, b)

        # remove the file
        if os.path.isfile(filename):
            os.remove(filename)

        # compute a new table
        sim_set_2 = {'category': {1:
                                  {'index': {
                                      1: {'MCS': [5, 24]},
                                      2: {'MCS': [12, 13]}
                                  }}}}
        snr_dbs_2 = [10, 15]
        cb_sizes_2 = [30, 75, 90]

        # compute tables and save them to file
        table_2 = phy_abs.new_bler_table(
            snr_dbs_2,
            cb_sizes_2,
            sim_set_2,
            max_mc_iter=15,
            batch_size=10,
            verbose=False)

        table1 = DeepUpdateDict(table_1)
        table1.deep_update(table_2)
        # check that the internal BLER table has been updated
        self.assertDictEqual(phy_abs.bler_table,
                             table1)

        # check that for conflicting MCS values only the last table is kept
        self.assertDictEqual(
            phy_abs.bler_table['category'][1]['index'][1]['MCS'][24],
            table_2['category'][1]['index'][1]['MCS'][24])
        self.assertDictEqual(
            phy_abs.bler_table['category'][1]['index'][2]['MCS'][12],
            table_2['category'][1]['index'][2]['MCS'][12])

        # check that new MCS values are added
        self.assertTrue(
            13 in phy_abs.bler_table['category'][1]['index'][2]['MCS'].keys())
        self.assertTrue(
            5 in phy_abs.bler_table['category'][1]['index'][1]['MCS'].keys())

    def test_bler_interpolation(self):
        """
        Validate the (CBS, SNR) -> BLER interpolation
        """

        categories = [1, 1, 1]
        table_index = [1, 1, 1]
        mcs = [10, 15, 16]

        # Instantiate the PHY abstraction object
        phy_abs = PHYAbstraction()

        assert len(categories) == len(table_index)
        assert len(table_index) == len(mcs)

        for k in range(len(categories)):

            table_tmp = phy_abs.bler_table['category'][categories[k]
                                                       ]['index'][table_index[k]]['MCS'][mcs[k]]

            # SNR/CBS values at which tables have been simulated
            snr_dbs_sim = table_tmp['SNR_db']
            cb_sizes_sim = list(table_tmp['CBS'].keys())

            # Redefine the interpolation grid
            # Note that the interpolation grid includes the simulation grid
            phy_abs.cbs_interp_min_max_delta = \
                (cb_sizes_sim[0],
                    cb_sizes_sim[-1],
                    (cb_sizes_sim[1] - cb_sizes_sim[0])/10)
            phy_abs._snr_db_interp_min_max_delta = \
                (snr_dbs_sim[0],
                    snr_dbs_sim[-1],
                    (snr_dbs_sim[1] - snr_dbs_sim[0])/10)

            # interpolated table
            table_interp = phy_abs.bler_table_interp.numpy(
            )[categories[k], table_index[k]-1, mcs[k], ::]

            for cbs in cb_sizes_sim:
                cbs_interp_ind = np.argmin(abs(phy_abs._cbs_interp - cbs))
                bler_sim = table_tmp['CBS'][cbs]['BLER']
                for ii, snr in enumerate(snr_dbs_sim):
                    snr_interp_ind = np.argmin(
                        abs(phy_abs._snr_dbs_interp - snr))

                    bler_interp = table_interp[cbs_interp_ind, snr_interp_ind]

                    cbs_interp = phy_abs._cbs_interp[cbs_interp_ind]
                    snr_interp = phy_abs._snr_dbs_interp[snr_interp_ind]

                    # check that interpolated value and original value coincide
                    self.assertAlmostEqual(
                        bler_sim[ii], bler_interp, delta=1e-2)
        # TODO: improve precision by defining integer SNR simulation values

        # check that interpolation is performed again after changing the grid
        phy_abs.snr_db_interp_min_max_delta = (0, 20, .01)
        n_snr = len(np.arange(phy_abs.snr_db_interp_min_max_delta[0],
                              phy_abs.snr_db_interp_min_max_delta[1],
                              phy_abs.snr_db_interp_min_max_delta[2]))
        self.assertTrue(phy_abs.bler_table_interp.shape[-1] == n_snr)

        phy_abs.cbs_interp_min_max_delta = [100, 1000, 15]
        n_cbs = len(np.arange(phy_abs.cbs_interp_min_max_delta[0],
                              phy_abs.cbs_interp_min_max_delta[1],
                              phy_abs.cbs_interp_min_max_delta[2]))
        self.assertTrue(phy_abs.bler_table_interp.shape[-2] == n_cbs)

    def test_snr_interpolation(self):
        """
        Validate the (CBS, BLER) -> SNR interpolation
        """
        categories = [1, 1, 0, 0]
        table_index = [1, 1, 2, 1]
        mcs = [5, 15, 12, 20]

        # Instantiate the PHY abstraction object
        phy_abs = PHYAbstraction()

        assert len(categories) == len(table_index)
        assert len(table_index) == len(mcs)

        for k in range(len(categories)):

            table_tmp = phy_abs.bler_table['category'][categories[k]
                                                       ]['index'][table_index[k]]['MCS'][mcs[k]]

            # SNR/CBS values at which tables have been simulated
            snr_sim = np.array(table_tmp['SNR_db'])
            cbs_sim = list(table_tmp['CBS'].keys())

            phy_abs.cbs_interp_min_max_delta = \
                (cbs_sim[0], cbs_sim[-1], 10)

            # interpolated table
            snr_interp_mat = phy_abs.snr_table_interp.numpy(
            )[categories[k], table_index[k]-1, mcs[k], ::]

            # Loop over simulated CBS
            for cbs in cbs_sim[:2]:
                # Find corresponding interpolated CBS
                cbs_interp_ind = np.argmin(abs(phy_abs._cbs_interp - cbs))
                cbs_interp_val = phy_abs._cbs_interp[cbs_interp_ind]
                # Simulated BLER: snr_sim <-> bler_sim
                bler_sim = table_tmp['CBS'][cbs]['BLER']

                # Loop over simulated BLERs
                for ind_sim, bler_sim_val in enumerate(bler_sim):
                    if (bler_sim_val < 0.02) | (bler_sim_val > 0.98):
                        continue
                    # Find corresponding interpolated BLER
                    bler_interp_ind = np.argmin(
                        abs(phy_abs._blers_interp - bler_sim_val))

                    self.assertAlmostEqual(
                        snr_interp_mat[cbs_interp_ind, bler_interp_ind],
                        snr_sim[ind_sim],
                        delta=1
                    )

    def test_get_bler(self):
        """Test get_bler method"""

        # Instantiate the PHY abstraction object
        cbs_delta = 99
        assert (cbs_delta % 2) != 0, 'cbs_delta must be odd'

        phy_abs = PHYAbstraction(
            cbs_interp_min_max_delta=(24, 8448, cbs_delta),
            precision="double")

        # Check that it does not throw errors with non-tensor inputs
        bler_float = phy_abs.get_bler(mcs_index=10,
                                      mcs_table_index=1,
                                      mcs_category=0,  # PUSCH
                                      cb_size=500,
                                      snr_eff=10)

        # Test with tensor inputs
        # Compare against its Numpy equivalent
        shape = [100, 100]
        snr_db = config.tf_rng.uniform(shape, minval=0,
                                       maxval=20)
        ten = tf.cast(10, snr_db.dtype)
        snr = tf.pow(ten, snr_db/10)
        table_index = random_tensor_from_values([1, 2], shape)
        mcs = random_tensor_from_values(list(range(10, 20)), shape)
        cbs = config.tf_rng.uniform(
            shape, minval=24, maxval=8000, dtype=tf.int32)
        category = random_tensor_from_values([0, 1], shape)
        bler_tf = phy_abs.get_bler(mcs,
                                   table_index,
                                   category,
                                   cbs,
                                   snr)
        bler_tf = bler_tf.numpy()

        for i1 in range(shape[0]):
            for i2 in range(shape[1]):
                table_idx = tf.gather_nd(table_index, [i1, i2]).numpy() - 1
                category_idx = tf.gather_nd(category, [i1, i2]).numpy()
                mcs_idx = tf.gather_nd(mcs, [i1, i2]).numpy()

                cbs_ = tf.gather_nd(cbs, [i1, i2])
                cbs_idx = np.argmin(abs(phy_abs._cbs_interp - cbs_))

                snr_db_ = tf.gather_nd(snr_db, [i1, i2])
                snr_db_idx = np.argmin(abs(phy_abs._snr_dbs_interp - snr_db_))

                bler_numpy = tf.gather_nd(phy_abs.bler_table_interp,
                                          [category_idx,
                                           table_idx,
                                           mcs_idx,
                                           cbs_idx,
                                           snr_db_idx]).numpy()
                self.assertAlmostEqual(bler_tf[i1, i2], bler_numpy)

    def test_call(self):
        r"""
        Ensure that 'call' method of PHYAbstraction does not throw any error
        """

        # N. of base stations
        num_bs = 5
        # N. of UT per base station
        num_ut_per_bs = 10
        # N. of streams per UT
        num_streams_per_ut = 2
        # Additive noise power
        no = 1e-5
        # 'DL' for downlink or 'UL' for uplink

        for link_direction in ['DL', 'UL']:

            if link_direction == 'DL':

                num_tx = num_bs
                num_rx_per_tx = num_ut_per_bs
                num_streams_per_rx = num_streams_per_ut
                num_rx = num_rx_per_tx * num_tx
                num_ut = num_rx
                num_streams_per_tx = num_streams_per_rx * num_rx_per_tx

                # RX-TX association matrix
                rx_tx_association = np.zeros([num_rx, num_tx])
                idx = np.array([[i1, i2] for i2 in range(num_tx) for i1 in
                                np.arange(i2*num_rx_per_tx, (i2+1)*num_rx_per_tx)])
                rx_tx_association[idx[:, 0], idx[:, 1]] = 1

            elif link_direction == 'UL':

                num_rx = num_bs
                num_tx_per_rx = num_ut_per_bs
                num_streams_per_tx = num_streams_per_ut
                num_tx = num_tx_per_rx * num_rx
                num_ut = num_tx
                num_streams_per_rx = num_streams_per_tx * num_tx_per_rx

                # RX-TX association matrix
                rx_tx_association = np.zeros([num_rx, num_tx])
                idx = np.array([[i1, i2] for i1 in range(num_rx) for i2 in
                                np.arange(i1*num_tx_per_rx, (i1+1)*num_tx_per_rx)])
                rx_tx_association[idx[:, 0], idx[:, 1]] = 1

            num_rx_ant = num_streams_per_rx
            num_tx_ant = num_streams_per_tx

            # Instantiate a StreamManagement object
            # Determines which data streams are intended for each receiver
            stream_management = StreamManagement(
                rx_tx_association, num_streams_per_tx)

            # Generate channel matrix
            batch_size = 1
            num_subcarriers = 12
            num_ofdm_symbols = 2
            shape = [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_ofdm_symbols, num_subcarriers]
            h_r = config.tf_rng.uniform(shape, minval=0, maxval=1)
            h_i = config.tf_rng.uniform(shape, minval=0, maxval=1)
            h = tf.complex(h_r, h_i)

            rg = ResourceGrid(num_ofdm_symbols=num_ofdm_symbols,
                             fft_size=num_subcarriers,
                             subcarrier_spacing=15e3,
                             num_tx=num_tx,
                             num_streams_per_tx=num_streams_per_tx)

            # Transmit power
            tx_power = config.tf_rng.uniform([batch_size,
                                              num_tx,
                                              num_streams_per_tx,
                                              num_ofdm_symbols,
                                              num_subcarriers],
                                             minval=.5, maxval=1.5,
                                             dtype=tf.float32)

            # Compute per-stream and per-subcarrier SINR
            # [..., num_subcarriers, num_ut, num_streams_per_ut]
            if link_direction == 'DL':
                precoded_channel = RZFPrecodedChannel(rg, stream_management)
                h_eff = precoded_channel(h, tx_power, alpha=0.01)
            else:
                h_eff = h
            sinr = LMMSEPostEqualizationSINR(rg, stream_management)(h_eff, tf.cast(no, h_eff.dtype))

            # MCS
            mcs_index = config.tf_rng.uniform([batch_size,
                                               num_rx],
                                               minval=3,
                                               maxval=5,
                                               dtype=tf.int32)

            mcs_table_index = 1
            mcs_category = 0 if link_direction == 'UL' else 1

            # Instantiate PHYAbstraction object
            phy_abs = PHYAbstraction(precision='double')

            num_decoded_bits, harq_feedback, sinr_eff, tbler, bler = \
                phy_abs(mcs_index,
                        sinr=sinr,
                        mcs_table_index=mcs_table_index,
                        mcs_category=mcs_category)
            # If HARQ=1 (ACK) then number of successfully decoded bits must be positive
            self.assertTrue(tf.reduce_all(tf.gather_nd(
                num_decoded_bits, tf.where(harq_feedback == 1)) > 0))
            self.assertEqual(tbler.dtype, tf.float64)
