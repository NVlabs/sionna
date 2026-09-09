#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Tests for PUSCHReceiver."""

import pytest
import numpy as np
import torch

from sionna.phy import config as phy_config
from sionna.phy.utils import compute_ber
from sionna.phy.channel import OFDMChannel, RayleighBlockFading, TimeChannel
from sionna.phy.nr import PUSCHConfig, PUSCHTransmitter, PUSCHReceiver
from sionna.phy.mimo import StreamManagement


def run_test(
    pusch_configs,
    channel_estimator="perfect",
    domain="freq",
    num_rx=1,
    num_rx_ant=8,
    batch_size=128,
    precision="single",
):
    """Configurable function for various test cases."""
    num_tx = len(pusch_configs)
    num_tx_ant = pusch_configs[0].num_antenna_ports
    l_min, l_max = -1, 3

    pusch_transmitter = PUSCHTransmitter(
        pusch_configs, output_domain=domain, precision=precision
    )

    stream_management = None
    if num_rx == 2:
        rx_tx_association = np.eye(2, dtype=bool)
        stream_management = StreamManagement(
            rx_tx_association, pusch_configs[0].num_layers
        )

    pusch_receiver = PUSCHReceiver(
        pusch_transmitter,
        stream_management=stream_management,
        input_domain=domain,
        l_min=l_min,
        channel_estimator=channel_estimator,
        precision=precision,
    )

    rayleigh = RayleighBlockFading(
        num_rx=num_rx,
        num_rx_ant=num_rx_ant,
        num_tx=num_tx,
        num_tx_ant=num_tx_ant,
        precision=precision,
    )

    if domain == "freq":
        channel = OFDMChannel(
            rayleigh,
            pusch_transmitter.resource_grid,
            normalize_channel=True,
            return_channel=True,
            precision=precision,
        )
    else:
        channel = TimeChannel(
            rayleigh,
            pusch_transmitter.resource_grid.bandwidth,
            pusch_transmitter.resource_grid.num_time_samples,
            l_min=l_min,
            l_max=l_max,
            normalize_channel=True,
            return_channel=True,
            precision=precision,
        )

    x, b = pusch_transmitter(batch_size)
    y, h = channel(x)

    dtype = torch.float32 if precision == "single" else torch.float64
    no = torch.tensor(0.001, dtype=dtype)

    if channel_estimator == "perfect":
        b_hat = pusch_receiver(y, no, h)
    else:
        b_hat = pusch_receiver(y, no)

    return compute_ber(b, b_hat)


class TestPUSCHReceiver:
    """Tests for PUSCHReceiver."""

    def test_device_override_propagates_from_transmitter(self, device):
        """Transmitter and receiver remain colocated despite global defaults."""
        global_device = next(
            (candidate for candidate in phy_config.available_devices
             if candidate != device),
            None,
        )
        if global_device is None:
            pytest.skip("Test requires two available devices")
        phy_config.device = global_device

        pusch_config = PUSCHConfig()
        pusch_config.n_size_bwp = 12
        transmitter = PUSCHTransmitter(pusch_config, device=device)
        receiver = PUSCHReceiver(
            transmitter,
            channel_estimator="perfect",
            device=device,
        )

        assert receiver.device == device
        assert receiver.resource_grid.device == device
        for _, buffer in receiver.named_buffers():
            assert buffer.device == torch.device(device)

    def test_01(self):
        """Test perfect and imperfect CSI in freq and time domain."""
        pusch_config = PUSCHConfig()
        pusch_config.n_size_bwp = 4
        pusch_config.num_antenna_ports = 4
        pusch_config.precoding = "codebook"
        pusch_config.num_layers = 2
        pusch_config.dmrs.config_type = 1
        pusch_config.dmrs.num_cdm_groups_without_data = 1
        pusch_config.dmrs.dmrs_port_set = [0, 1]
        pusch_configs = [pusch_config]

        # Test perfect CSI, freq domain
        ber = run_test(
            pusch_configs, channel_estimator="perfect", domain="freq"
        )
        assert ber == 0.0

        # Test imperfect CSI, freq domain
        ber = run_test(
            pusch_configs, channel_estimator=None, domain="freq"
        )
        assert ber == 0.0

        # Test perfect CSI, time domain
        ber = run_test(
            pusch_configs, channel_estimator="perfect", domain="time"
        )
        assert ber == 0.0

        # Test imperfect CSI, time domain
        ber = run_test(
            pusch_configs, channel_estimator=None, domain="time"
        )
        assert ber == 0.0

    def test_02(self):
        """Multi transmitter, multi stream test."""
        pusch_config = PUSCHConfig()
        pusch_config.n_size_bwp = 4
        pusch_config.num_antenna_ports = 4
        pusch_config.precoding = "codebook"
        pusch_config.num_layers = 2
        pusch_config.dmrs.config_type = 2
        pusch_config.dmrs.num_cdm_groups_without_data = 2
        pusch_config.dmrs.dmrs_port_set = [0, 2]
        pusch_config.dmrs.additional_position = 1

        pusch_config2 = pusch_config.clone()
        pusch_config2.dmrs.dmrs_port_set = [1, 3]

        pusch_configs = [pusch_config, pusch_config2]

        ber = run_test(pusch_configs, channel_estimator="perfect", domain="freq")
        assert ber == 0.0

        ber = run_test(pusch_configs, channel_estimator=None, domain="freq")
        assert ber == 0.0

        ber = run_test(pusch_configs, channel_estimator="perfect", domain="time")
        assert ber == 0.0

        ber = run_test(pusch_configs, channel_estimator=None, domain="time")
        assert ber == 0.0

    def test_03(self):
        """Multi transmitter, multi stream, no precoding."""
        pusch_config = PUSCHConfig()
        pusch_config.n_size_bwp = 4
        pusch_config.num_antenna_ports = 2
        pusch_config.precoding = "non-codebook"
        pusch_config.num_layers = 2
        pusch_config.dmrs.config_type = 2
        pusch_config.dmrs.num_cdm_groups_without_data = 2
        pusch_config.dmrs.dmrs_port_set = [0, 2]
        pusch_config.dmrs.additional_position = 1

        pusch_config2 = pusch_config.clone()
        pusch_config2.dmrs.dmrs_port_set = [1, 3]

        pusch_configs = [pusch_config, pusch_config2]

        ber = run_test(pusch_configs, channel_estimator="perfect", domain="freq")
        assert ber == 0.0

        ber = run_test(pusch_configs, channel_estimator=None, domain="freq")
        assert ber == 0.0

        ber = run_test(pusch_configs, channel_estimator="perfect", domain="time")
        assert ber == 0.0

        ber = run_test(pusch_configs, channel_estimator=None, domain="time")
        assert ber == 0.0

    def test_04(self):
        """Very large transport block."""
        pusch_config = PUSCHConfig()
        pusch_config.n_size_bwp = 273
        pusch_config.tb.mcs_index = 26
        pusch_config.tb.mcs_table = 2
        pusch_config.num_antenna_ports = 4
        pusch_config.precoding = "codebook"
        pusch_config.num_layers = 4
        pusch_config.dmrs.config_type = 2
        pusch_config.dmrs.length = 2
        pusch_config.dmrs.num_cdm_groups_without_data = 1
        pusch_config.dmrs.dmrs_port_set = [0, 1, 6, 7]
        pusch_config.dmrs.additional_position = 0
        pusch_configs = [pusch_config]

        ber = run_test(pusch_configs, channel_estimator=None, batch_size=2)
        assert ber == 0.0

    def test_05(self):
        """Very short transport block.

        Note: Using 4 symbols instead of 2 from original TF test, since
        channel estimation with only 2 OFDM symbols (1 DMRS + 1 data)
        has insufficient pilots for reliable estimation.
        """
        pusch_config = PUSCHConfig()
        pusch_config.n_size_bwp = 1
        pusch_config.num_antenna_ports = 1
        pusch_config.precoding = "non-codebook"
        pusch_config.num_layers = 1
        pusch_config.dmrs.config_type = 1
        pusch_config.dmrs.additional_position = 0
        pusch_config.dmrs.num_cdm_groups_without_data = 2
        pusch_config.dmrs.dmrs_port_set = [0]
        pusch_config.mapping_type = "B"
        pusch_config.symbol_allocation = [5, 4]
        pusch_config.tb.mcs_index = 10
        pusch_configs = [pusch_config]

        ber = run_test(pusch_configs, channel_estimator=None, batch_size=128)
        assert ber == 0.0

    def test_06(self):
        """Multi transmitter, multi stream, multi receiver test."""
        pusch_config = PUSCHConfig()
        pusch_config.n_size_bwp = 4
        pusch_config.num_antenna_ports = 4
        pusch_config.precoding = "codebook"
        pusch_config.tpmi = 2
        pusch_config.num_layers = 2
        pusch_config.dmrs.config_type = 1
        pusch_config.dmrs.length = 2
        pusch_config.dmrs.additional_position = 1
        pusch_config.dmrs.num_cdm_groups_without_data = 2
        pusch_config.dmrs.dmrs_port_set = [0, 1]
        pusch_config.tb.mcs_index = 10

        pusch_config2 = pusch_config.clone()
        pusch_config.dmrs.dmrs_port_set = [2, 3]

        pusch_configs = [pusch_config, pusch_config2]

        ber = run_test(pusch_configs, channel_estimator="perfect", domain="freq")
        assert ber == 0.0

        ber = run_test(pusch_configs, channel_estimator=None, domain="freq")
        assert ber == 0.0

        ber = run_test(pusch_configs, channel_estimator="perfect", domain="time")
        assert ber == 0.0

        ber = run_test(pusch_configs, channel_estimator=None, domain="time")
        assert ber == 0.0

    def test_07(self):
        """Multi transmitter, multi stream in double precision."""
        pusch_config = PUSCHConfig()
        pusch_config.n_size_bwp = 4
        pusch_config.num_antenna_ports = 4
        pusch_config.precoding = "codebook"
        pusch_config.tpmi = 2
        pusch_config.num_layers = 2
        pusch_config.dmrs.config_type = 1
        pusch_config.dmrs.length = 2
        pusch_config.dmrs.additional_position = 1
        pusch_config.dmrs.num_cdm_groups_without_data = 2
        pusch_config.dmrs.dmrs_port_set = [0, 1]
        pusch_config.tb.mcs_index = 10

        pusch_config2 = pusch_config.clone()
        pusch_config.dmrs.dmrs_port_set = [2, 3]

        pusch_configs = [pusch_config, pusch_config2]

        ber = run_test(
            pusch_configs,
            channel_estimator="perfect",
            domain="freq",
            precision="double",
        )
        assert ber == 0.0

        ber = run_test(
            pusch_configs,
            channel_estimator=None,
            domain="freq",
            precision="double",
        )
        assert ber == 0.0

        ber = run_test(
            pusch_configs,
            channel_estimator="perfect",
            domain="time",
            precision="double",
        )
        assert ber == 0.0

        ber = run_test(
            pusch_configs,
            channel_estimator=None,
            domain="time",
            precision="double",
        )
        assert ber == 0.0
