#
# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0#

import unittest
import numpy as np
import tensorflow as tf
import os

from sionna.phy.mimo import StreamManagement
from sionna.sys import PHYAbstraction, EESM
from sionna.phy import config
from sionna.phy.nr.utils import CodedAWGNChannelNR, MCSDecoderNR, \
    TransportBlockNR
from sionna.phy.utils import SplineGriddataInterpolation, sample_bernoulli


class TestEffectiveSINR(unittest.TestCase):

    def test_sinr_eff_vs_numpy(self):
        """ 
        Check that the effective SINR computation matches its Numpy
        counterpart 
        """

        def get_sinr_eff_numpy(sinr_sel, beta_sel, sinr_eff_min_db, sinr_eff_max_db):
            """ Numpy version of effective SINR computation """
            n_used_res = (sinr_sel > 0).sum()
            if n_used_res == 0:
                return 0
            sinr_exp = np.exp(-sinr_sel / beta_sel)
            sinr_exp = sinr_exp * (sinr_sel > 0)
            sinr_eff_numpy = -beta_sel * np.log(np.sum(sinr_exp) / n_used_res)
            
            sinr_eff_min = np.power(10, sinr_eff_min_db/10)
            sinr_eff_max = np.power(10, sinr_eff_max_db/10)
            if sinr_eff_numpy > sinr_eff_max:
                sinr_eff_numpy = sinr_eff_max
            if sinr_eff_numpy<sinr_eff_min:
                sinr_eff_numpy = sinr_eff_min
            return sinr_eff_numpy

        batch_size = 30
        num_ofdm_symbols = 2
        num_subcarriers = 10
        num_ut = 50
        num_streams_per_ut = 3
        sinr_eff_min_db = -40
        sinr_eff_max_db = 40
        dtype = tf.float64
        precision = 'double'

        eff_sinr_obj = EESM(sinr_eff_min_db=sinr_eff_min_db,
                            sinr_eff_max_db=sinr_eff_max_db,
                            precision=precision)

        @tf.function(jit_compile=False)
        def effective_sinr_eesm_graph(sinr, mcs_index, mcs_table_index=1, per_stream=False):
            return eff_sinr_obj(sinr, mcs_index, mcs_table_index=mcs_table_index, per_stream=per_stream)

        @tf.function(jit_compile=True, reduce_retracing=True)
        def effective_sinr_eesm_xla(sinr, mcs_index, mcs_table_index=1, per_stream=False):
            return eff_sinr_obj(sinr, mcs_index, mcs_table_index=mcs_table_index, per_stream=per_stream)

        effective_sinr_eesm_dict = {'eager': eff_sinr_obj,
                                    'graph': effective_sinr_eesm_graph,
                                    'xla': effective_sinr_eesm_xla}

        for mode, eff_sinr_fun in effective_sinr_eesm_dict.items():
            print(f'\n{mode}')

            # generate SINR randomly
            sinr = config.tf_rng.uniform([batch_size,
                                          num_ofdm_symbols,
                                          num_subcarriers,
                                          num_ut,
                                          num_streams_per_ut],
                                         minval=0,
                                         maxval=10,
                                         dtype=dtype)
            # Mask SINR on some streams
            sinr = sinr * tf.cast(config.tf_rng.uniform([batch_size,
                                                         num_ofdm_symbols,
                                                         num_subcarriers,
                                                         num_ut,
                                                         num_streams_per_ut],
                                                        minval=0,
                                                        maxval=2,
                                                        dtype=tf.int32), dtype)

            # Generate MCS per user
            mcs = config.tf_rng.uniform(
                [batch_size, num_ut], minval=0, maxval=27, dtype=tf.int32)

            # Table index
            table_index = config.tf_rng.uniform(
                [batch_size, num_ut], minval=1, maxval=3, dtype=tf.int32)

            for per_stream in [True, False]:
                # TF version
                sinr_eff = eff_sinr_fun(
                    sinr, mcs, mcs_table_index=table_index, per_stream=per_stream).numpy()

                for batch in range(batch_size):
                    for ut in range(num_ut):
                        mcs_sel = mcs[batch, ut]
                        table_idx_sel = table_index[batch, ut]
                        beta_sel = eff_sinr_obj.beta_tensor.numpy()[
                            table_idx_sel-1, mcs_sel]

                        if per_stream:
                            for stream in range(num_streams_per_ut):
                                sinr_sel = sinr[batch, :, :, ut, stream].numpy()
                                # Numpy version
                                sinr_eff_numpy = get_sinr_eff_numpy(
                                    sinr_sel, beta_sel, sinr_eff_min_db, sinr_eff_max_db)

                                self.assertAlmostEqual(
                                    sinr_eff[batch, ut, stream], sinr_eff_numpy, delta=1e-5)
                        else:
                            sinr_sel = sinr[batch, :, :, ut, :].numpy()
                            # Numpy version
                            sinr_eff_numpy = get_sinr_eff_numpy(
                                sinr_sel, beta_sel, sinr_eff_min_db, sinr_eff_max_db)

                            self.assertAlmostEqual(
                                sinr_eff[batch, ut], sinr_eff_numpy, delta=1e-5)
