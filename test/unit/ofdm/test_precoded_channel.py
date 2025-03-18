#
# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0#
import unittest
import numpy as np
import tensorflow as tf
import sionna
from sionna.phy.ofdm import ResourceGrid, RZFPrecodedChannel
from sionna.phy.mimo import StreamManagement, rzf_precoding_matrix
from sionna.phy.channel import RayleighBlockFading, cir_to_ofdm_channel, subcarrier_frequencies
from sionna.phy.utils import expand_to_rank
tf_rng = sionna.phy.config.tf_rng

class TestRZFPrecodedChannel(unittest.TestCase):

    def test_against_alternative_implementation(self):
        num_rx_per_tx = 2
        num_streams_per_rx = 2
        num_rx_ant = num_streams_per_rx
        num_tx = 2
        num_rx = num_rx_per_tx * num_tx
        num_streams_per_tx = num_rx_per_tx * num_streams_per_rx

        rx_tx_association = np.zeros((num_rx, num_tx), dtype=np.int32)
        for j in range(num_tx):
            rx_tx_association[j*num_rx_per_tx:(j+1)*num_rx_per_tx, j] = 1

        sm = StreamManagement(rx_tx_association=rx_tx_association,
                            num_streams_per_tx=num_streams_per_tx)

        rg = ResourceGrid(num_ofdm_symbols=14,
                        fft_size=64,
                        subcarrier_spacing=15e3,
                        num_tx=sm.num_tx,
                        num_streams_per_tx=sm.num_streams_per_tx)

        channel = RayleighBlockFading(num_rx=sm.num_rx,
                                    num_rx_ant=num_rx_ant,
                                    num_tx=sm.num_tx,
                                    num_tx_ant=sm.num_streams_per_tx*2)

        batch_size=128
        tx_power = tf_rng.uniform([batch_size, num_tx, num_streams_per_tx, rg.num_ofdm_symbols, rg.fft_size])
        alpha = tf_rng.uniform([batch_size, num_tx, 1, 1])
        cir = channel(batch_size=batch_size, num_time_steps=rg.num_ofdm_symbols)

        frequencies = subcarrier_frequencies(rg.fft_size, rg.subcarrier_spacing)
        h = cir_to_ofdm_channel(frequencies, *cir)

        precoded_channel = RZFPrecodedChannel(resource_grid=rg, stream_management=sm)
        h_eff = precoded_channel(h, tx_power=tx_power, alpha=alpha)

        for j in range(num_tx):
            rx_ind = np.where(rx_tx_association[:,j])[0]
            h_des = h.numpy()[:, rx_ind, :, j:j+1, :, :, :]
            h_des = np.squeeze(h_des)
            h_des = np.reshape(h_des, (batch_size, -1, h.shape[4], rg.num_ofdm_symbols, rg.fft_size))
            h_des = np.transpose(h_des, (0, 3, 4, 1, 2))
            g = rzf_precoding_matrix(h_des, alpha=alpha[:,j])
            # [batch_size, num_streams_per_tx, rg.num_ofdm_symbols, rg.fft_size]
            power = tx_power[:, j, :]
            power = tf.transpose(power, (0, 2, 3, 1))
            power = expand_to_rank(power, tf.rank(g), axis=-2)
            g = tf.cast(tf.sqrt(power), g.dtype) * g
            for i in range(num_rx):
                h_i_j = h.numpy()[:, i, :, j, :, :, :]
                h_i_j = np.transpose(h_i_j, (0, 3, 4, 1, 2))
                h_eff_i_j = tf.matmul(h_i_j, g)
                q = h_eff.numpy()[:,i,:,j,:]
                q = np.transpose(q, (0, 3, 4, 1, 2))
                self.assertTrue(np.allclose(q, h_eff_i_j, atol=1e-6))