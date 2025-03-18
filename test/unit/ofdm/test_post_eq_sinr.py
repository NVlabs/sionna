#
# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0#
import unittest
import numpy as np
import tensorflow as tf
import sionna
from sionna.phy.ofdm import ResourceGrid, RZFPrecodedChannel, LMMSEPostEqualizationSINR, EyePrecodedChannel
from sionna.phy.mimo import StreamManagement, lmmse_matrix
from sionna.phy.channel import RayleighBlockFading, cir_to_ofdm_channel, subcarrier_frequencies
from sionna.phy.utils import inv_cholesky, expand_to_rank
tf_rng = sionna.phy.config.tf_rng

class TestLMMSEPostEqualizationSINR(unittest.TestCase):
    """
    Test the LMMSEPostEqualizationSINR class
    """

    def test_against_alternative_implementation(self):
        sionna.phy.config.precision = "double"
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

        lmmse_posteq_sinr = LMMSEPostEqualizationSINR(resource_grid=rg, stream_management=sm)
        no = tf.constant(0.1, dtype=sionna.phy.config.tf_rdtype)
        sinr = lmmse_posteq_sinr(h_eff, no=no)

        # Alternative implementation
        sinr_alt = np.zeros_like(sinr)
        for j in range(num_tx):
            # Get all rx indices for this tx
            rx_ind = np.where(rx_tx_association[:,j])[0]
            for rx_counter,i in enumerate(rx_ind):
                # Get desired channels from the transmitter
                #[batch_size, num_rx_ant, num_streams_per_rx, num_ofdm_symbols, num_effective_subcarriers]
                h_i_d = h_eff[:,i,:,j,rx_counter*num_streams_per_rx:(rx_counter+1)*num_streams_per_rx]
                #[batch_size, num_ofdm_symbols, num_effective_subcarriers, num_rx_ant, num_streams_per_rx]
                h_i_d = tf.transpose( h_i_d, (0, 3, 4, 1, 2))

                # Get interfering channels from the transmitter
                h_i_i = tf.concat([
                    h_eff[:,i,:,j,:rx_counter*num_streams_per_rx],
                    h_eff[:,i,:,j,(rx_counter+1)*num_streams_per_rx:]
                    ], axis=2)
                #[batch_size, num_ofdm_symbols, num_effective_subcarriers, num_rx_ant, num_streams_per_tx-num_streams_per_rx]
                h_i_i = tf.transpose( h_i_i, (0, 3, 4, 1, 2))

                # Get all channels for this receiver
                # [batch_size, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols, num_effective_subcarriers]
                h_i = h_eff[:,i]

                # Take channels from all interfering transmitters
                # [batch_size, num_rx_ant, num_tx-1, num_streams_per_tx, num_ofdm_symbols, num_effective_subcarriers]
                h_i_ud = tf.concat([h_i[:,:,:j], h_i[:,:,j+1:]], axis=2)

                # [batch_size, num_ofdm_symbols, num_effective_subcarriersm, num_rx_ant, num_tx-1, num_streams_per_tx]
                h_i_ud = tf.transpose(h_i_ud, (0, 4, 5, 1, 2, 3))

                # [batch_size, num_ofdm_symbols, num_effective_subcarriersm, num_rx_ant, (num_tx-1)*num_streams_per_tx)]
                h_i_ud = tf.reshape(h_i_ud, [batch_size, rg.num_ofdm_symbols, rg.num_effective_subcarriers, num_rx_ant, -1])

                h_i_ti = tf.concat([h_i_ud, h_i_i], axis=-1)

                # Compute interference covariance matrix
                c_i = tf.matmul(h_i_ti, h_i_ti, adjoint_b=True) + tf.cast(no*tf.eye(num_rx_ant, batch_shape=h_i_ti.shape[:-2], dtype=no.dtype), h_i_ti.dtype)

                # Whiten channels
                l_inv = inv_cholesky(c_i) # Compute whitening matrix
                h_i_d = tf.matmul(l_inv, h_i_d)
                h_i_ti = tf.matmul(l_inv, h_i_ti)

                # Compute equalization matrix
                hhs = tf.matmul(h_i_d, h_i_d, adjoint_a=True) + tf.cast(tf.eye(num_rx_ant, batch_shape=h_i_d.shape[:-2]), h_i_d.dtype)
                chol = tf.linalg.cholesky(hhs)
                g_i = tf.linalg.cholesky_solve(chol, tf.linalg.adjoint(h_i_d))

                # Signal power
                p_i = tf.matmul(g_i, h_i_d)
                p_i = tf.abs(tf.linalg.diag_part(p_i))**2

                # Total power
                t_i = tf.abs(tf.matmul(g_i, tf.concat([h_i_ti, h_i_d], axis=-1)))**2
                t_i = tf.reduce_sum(t_i, axis=-1)

                # Noise power
                noise_i = tf.reduce_sum(tf.abs(g_i)**2, axis=-1)

                # SINR
                sinr_i = tf.math.divide_no_nan(p_i, t_i-p_i + noise_i)

                sinr_alt[...,i,:] = sinr_i

        self.assertTrue(np.allclose(sinr, sinr_alt))

    def test_siso(self):
        """Test against theoretical SINR for a SISO channel

        It also tests the correct broadcasting of alpha, no and tx_power.
        """
        sionna.phy.config.precision = "double"
        num_rx_per_tx = 1
        num_streams_per_rx = 1
        num_rx_ant = num_streams_per_rx
        num_tx = 1
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
                                    num_tx_ant=sm.num_streams_per_tx)

        batch_size=16
        tx_power = tf_rng.uniform([batch_size, num_tx, num_streams_per_tx, rg.num_ofdm_symbols])
        alpha = tf_rng.uniform([batch_size, num_tx])
        cir = channel(batch_size=batch_size, num_time_steps=rg.num_ofdm_symbols)

        frequencies = subcarrier_frequencies(rg.fft_size, rg.subcarrier_spacing)
        h = cir_to_ofdm_channel(frequencies, *cir)

        precoded_channel = RZFPrecodedChannel(resource_grid=rg, stream_management=sm)
        h_eff = precoded_channel(h, tx_power=tx_power, alpha=alpha)
        lmmse_posteq_sinr = LMMSEPostEqualizationSINR(resource_grid=rg, stream_management=sm)
        no = tf_rng.uniform([batch_size, num_rx, num_rx_ant])
        sinr = lmmse_posteq_sinr(h_eff, no=no, interference_whitening=True)
        sinr = tf.squeeze(sinr)

        tx_power = expand_to_rank(tx_power, 5, axis=-1)
        tx_power = tf.broadcast_to(tx_power, [batch_size, num_tx, num_streams_per_tx, rg.num_ofdm_symbols, rg.fft_size])
        no = expand_to_rank(no, 5, axis=-1)
        no = tf.broadcast_to(no, [batch_size, num_rx, num_rx_ant, rg.num_ofdm_symbols, rg.fft_size])
        sinr_theo = tf.abs(tf.squeeze(h))**2*tf.cast(tf.squeeze(tx_power)/tf.squeeze(no), sionna.phy.config.tf_rdtype)
        self.assertLess(np.max(np.abs(sinr_theo-sinr)/sinr), 1e-6)

    def test_single_antenna_uplink(self):
        """Test against theoretical SINR for a single antenna uplink channel"""
        sionna.phy.config.precision = "double"
        num_rx = 1
        num_rx_ant = 16
        num_tx = 8
        num_streams_per_tx = 1
        rx_tx_association = np.zeros((num_rx, num_tx), dtype=np.int32)
        for j in range(num_tx):
            rx_tx_association[0] = 1

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
                                    num_tx_ant=1)

        batch_size=16
        tx_power = tf_rng.uniform([batch_size, num_tx, num_streams_per_tx, rg.num_ofdm_symbols])
        cir = channel(batch_size=batch_size, num_time_steps=rg.num_ofdm_symbols)

        frequencies = subcarrier_frequencies(rg.fft_size, rg.subcarrier_spacing)
        h = cir_to_ofdm_channel(frequencies, *cir)

        precoded_channel = EyePrecodedChannel(resource_grid=rg, stream_management=sm)
        h_eff = precoded_channel(h, tx_power=tx_power)
        lmmse_posteq_sinr = LMMSEPostEqualizationSINR(resource_grid=rg, stream_management=sm)
        no = tf_rng.uniform([batch_size, num_rx, num_rx_ant,])
        sinr = lmmse_posteq_sinr(h_eff, no=no, interference_whitening=True)
        sinr = tf.squeeze(sinr)
     
        # Theoretical SINR
        # [batch_size, num_rx_ant, num_tx, num_ofdm_symbols, num_effective_subcarriers]
        h_e = tf.squeeze(h)

        # [batch_size, num_ofdm_symbols, num_effective_subcarriers, num_rx_ant, num_tx]
        h_e = tf.transpose(h_e, [0,3, 4, 1,2 ])

        # Apply transmit power
        p = tf.transpose(tf.squeeze(tx_power), [0,2,1])
        p = expand_to_rank(p, tf.rank(h_e), axis=2)
        h_e *= tf.cast(tf.sqrt(p), h_e.dtype)

        # Whiten channels according to noise power
        n = tf.expand_dims(no, -1)
        n = expand_to_rank(n, tf.rank(h_e), axis=1)
        h_e /= tf.cast(tf.sqrt(n), h_e.dtype)

        f = lmmse_matrix(h_e)
        signal_power = tf.abs(tf.linalg.diag_part(tf.matmul(f, h_e)))**2
        total_power = tf.reduce_sum(tf.abs(tf.matmul(f, h_e))**2, axis=-1)
        noise_power = tf.reduce_sum(tf.abs(f)**2, axis=-1)
        interference_power = total_power - signal_power
        sinr_theo = tf.math.divide_no_nan(signal_power, interference_power + noise_power)

        self.assertTrue(np.max(np.abs(sinr-sinr_theo)/np.abs(sinr)) < 1e-6)

