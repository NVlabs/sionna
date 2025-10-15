#
# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0#
import pytest
import os
import numpy as np
import tensorflow as tf
import json
import numpy as np
from sionna.phy.nr import PUSCHConfig, PUSCHTransmitter

script_dir = os.path.dirname(os.path.abspath(__file__))

def run_test(test_name):
    # Load data
    b, grid = np.load(test_name + ".npy", allow_pickle=True)

    # Load config
    with open(test_name + ".json", 'r') as file:
        config = json.load(file)

    pusch_config = PUSCHConfig()

    pusch_config.carrier.n_cell_id = config["carrier"]["n_cell_id"]
    pusch_config.carrier.slot_number = config["carrier"]["slot_number"]

    pusch_config.n_size_bwp = config["pusch"]["n_size_bwp"]
    pusch_config.symbol_allocation = config["pusch"]["symbol_allocation"]
    pusch_config.n_rnti = config["pusch"]["n_rnti"]
    pusch_config.num_antenna_ports = config["pusch"]["num_antenna_ports"]
    pusch_config.num_layers = config["pusch"]["num_layers"]
    pusch_config.precoding = config["pusch"]["precoding"]

    if pusch_config.precoding=="codebook":
        pusch_config.tpmi = config["pusch"]["tpmi"]

    pusch_config.dmrs.length = config["pusch"]["dmrs"]["length"]
    pusch_config.dmrs.config_type = config["pusch"]["dmrs"]["config_type"]
    pusch_config.dmrs.additional_position = config["pusch"]["dmrs"]["additional_position"]
    pusch_config.dmrs.num_cdm_groups_without_data = config["pusch"]["dmrs"]["num_cdm_groups_without_data"]
    pusch_config.dmrs.dmrs_port_set = config["pusch"]["dmrs"]["dmrs_port_set"]
    pusch_config.dmrs.n_scid = config["pusch"]["dmrs"]["n_scid"]
    pusch_config.dmrs.n_id = config["pusch"]["dmrs"]["n_id"]

    pusch_config.tb.mcs_index = config["pusch"]["tb"]["mcs_index"]
    pusch_config.tb.mcs_table = config["pusch"]["tb"]["mcs_table"]

    pusch_transmitter = PUSCHTransmitter(pusch_config, return_bits=False)

    x_grid = pusch_transmitter(b)
    x_grid = tf.transpose(x_grid[0,0], perm=[2,1,0])
    return np.allclose(tf.squeeze(x_grid), grid)

@pytest.mark.parametrize("test_id", list(range(0,83)))
def tests_against_reference(test_id):
    """Test PUSCHTransmitter output against reference"""
    test_name = script_dir+f"/pusch_test_configs/test_{test_id}"
    assert run_test(test_name)
