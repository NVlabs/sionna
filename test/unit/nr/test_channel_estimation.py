#
# SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
import pytest
import unittest
import numpy as np
import tensorflow as tf
from sionna.nr import PUSCHConfig, PUSCHTransmitter, PUSCHLSChannelEstimator
from sionna.channel import RayleighBlockFading, OFDMChannel
from sionna.utils import insert_dims


def check_channel_estimation(pusch_configs, add_awgn=False):
    pusch_transmitter = PUSCHTransmitter(pusch_configs)
    rayleigh = RayleighBlockFading(num_rx=1,
                                   num_rx_ant=16,
                                   num_tx=len(pusch_configs),
                                   num_tx_ant=pusch_configs[0].num_antenna_ports)
    channel = OFDMChannel(rayleigh,
                          pusch_transmitter._resource_grid,
                          return_channel=True, add_awgn=add_awgn)

    channel_estimator = PUSCHLSChannelEstimator(
                            pusch_transmitter._resource_grid,
                            pusch_transmitter._dmrs_length,
                            pusch_transmitter._dmrs_additional_position,
                            pusch_transmitter._num_cdm_groups_without_data,
                            interpolation_type='nn')

    no = 0.01
    x, b = pusch_transmitter(128)
    inputs = x if not add_awgn else [x, no]
    y, h = channel(inputs)

    if pusch_configs[0].precoding == "codebook":
        # Compute precoded channel
        # h has shape:
        # [batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_ofdm_symbols, fft_size]

        # Reshape to mut channel matrix dimensions last
        # [batch size, num_rx, num_tx, num_ofdm_symbols, fft_size, num_rx_ant, num_tx_ant]
        h = tf.transpose(h, perm=[0,1,3,5,6,2,4])

        # Multiply by precoding matrices to compute effective channels
        # [batch size, num_rx, num_tx, num_ofdm_symbols, fft_size, num_rx_ant, num_streams]
        w = pusch_transmitter._precoder._w
        w = insert_dims(w, 2, 1)
        h = tf.matmul(h, w)

        # [batch size, num_rx, num_rx_ant, num_tx, num_streams, num_ofdm_symbols, fft_size]
        h = tf.transpose(h, perm=[0,1,5,2,6,3,4])

    h_hat, err_var_hat = channel_estimator([y, no])

    # Compute empirical error variance
    err_var = np.var(h-h_hat)

    return_val = np.allclose(h, h_hat, atol=1e-6) if not add_awgn else np.allclose(err_var, err_var_hat, atol=1e-2)

    return return_val

class TestPUSCHLSChannelEstimator(unittest.TestCase):
    """Tests for PUSCHLSChannelEstimator"""

    def test_01(self):
        """Test many configurations for a single transmitter without precoding"""
        pusch_config = PUSCHConfig()
        pusch_config.n_size_bwp = 4
        pusch_config.precoding = "non-codebook"
        for num_antenna_ports in [1,2,4]:
            pusch_config.num_antenna_ports = num_antenna_ports
            pusch_config.num_layers = num_antenna_ports
            for length in [1,2]:
                pusch_config.dmrs.length = length
                max_additional_position = 3 if length==1 else 1
                for additional_position in range(0,max_additional_position+1):
                    pusch_config.dmrs.additional_position = additional_position
                    for config_type in [1,2]:
                        pusch_config.dmrs.config_type = config_type
                        max_cdm_groups = 2 if config_type==1 else 3
                        min_cdm_groups = 1 if num_antenna_ports<4 else 2
                        for num_cdm_groups_without_data in range(min_cdm_groups,max_cdm_groups+1):
                            pusch_config.dmrs.num_cdm_groups_without_data = num_cdm_groups_without_data
                            self.assertTrue(check_channel_estimation([pusch_config]))
                            self.assertTrue(check_channel_estimation([pusch_config], add_awgn=True))

    def test_02(self):
        """Test many configurations for a single transmitter with precoding"""
        pusch_config = PUSCHConfig()
        pusch_config.n_size_bwp = 5
        pusch_config.precoding = "codebook"
        pusch_config.tpmi = 2
        for num_antenna_ports in [1,2,4]:
            pusch_config.num_antenna_ports = num_antenna_ports
            for num_layers in range(1,num_antenna_ports):
                pusch_config.num_layers = num_layers
                for length in [1,2]:
                    pusch_config.dmrs.length = length
                    max_additional_position = 3 if length==1 else 1
                    for additional_position in range(0,max_additional_position+1):
                        pusch_config.dmrs.additional_position = additional_position
                        for config_type in [1,2]:
                            pusch_config.dmrs.config_type = config_type
                            max_cdm_groups = 2 if config_type==1 else 3
                            min_cdm_groups = 1 if num_antenna_ports<4 else 2
                            for num_cdm_groups_without_data in range(min_cdm_groups,max_cdm_groups+1):
                                pusch_config.dmrs.num_cdm_groups_without_data = num_cdm_groups_without_data
                                self.assertTrue(check_channel_estimation([pusch_config]))
                                self.assertTrue(check_channel_estimation([pusch_config], add_awgn=True))


    def test_03(self):
        """Some tests for multiple transmitters with precoding"""
        pusch_config = PUSCHConfig()
        pusch_config.n_size_bwp = 4
        pusch_config.num_antenna_ports=4
        pusch_config.precoding = "codebook"
        pusch_config.tpmi = 2
        pusch_config.num_layers = 2
        pusch_config.dmrs.length = 2
        pusch_config.dmrs.config_type = 2
        pusch_config.dmrs.num_cdm_groups_without_data = 3
        pusch_config.dmrs.dmrs_port_set = [0,1]
        pusch_config.dmrs.additional_position = 1

        pusch_config2 = pusch_config.clone()
        pusch_config2.dmrs.dmrs_port_set = [2,3]
        pusch_config2.tpmi = 11

        pusch_config3 = pusch_config.clone()
        pusch_config3.dmrs.dmrs_port_set = [4,5]
        pusch_config3.tpmi = 16

        pusch_configs = [pusch_config, pusch_config2, pusch_config3]
        self.assertTrue(check_channel_estimation(pusch_configs))
        self.assertTrue(check_channel_estimation(pusch_configs, add_awgn=True))
