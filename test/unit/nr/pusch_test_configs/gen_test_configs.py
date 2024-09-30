#
# SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Script to generate config files for the PUSCHTransmitter tests
"""
import json
import numpy as np
try:
    import sionna
except ImportError as e:
    import sys
    sys.path.append("../../../../")
import sionna
from sionna import config

def gen_config_file(filename,
                    n_cell_id,
                    slot_number,
                    n_size_bwp,
                    n_rnti,
                    num_antenna_ports,
                    num_layers,
                    precoding,
                    tpmi,
                    length,
                    config_type,
                    additional_position,
                    num_cdm_groups_without_data,
                    dmrs_port_set,
                    n_scid,
                    n_id,
                    mcs_index,
                    mcs_table):
    config = {
    "carrier" : {
        "n_cell_id" : n_cell_id,
        "slot_number" : slot_number
    },
    "pusch" : {
        "n_size_bwp" : n_size_bwp,
        "symbol_allocation" : [0, 14],
        "n_rnti" : n_rnti,
        "num_antenna_ports" : num_antenna_ports,
        "num_layers" : num_layers,
        "precoding" : precoding,
        "tpmi" : tpmi,
        "dmrs" : {
            "length" : length,
            "config_type" : config_type,
            "additional_position" : additional_position,
            "num_cdm_groups_without_data" : num_cdm_groups_without_data,
            "dmrs_port_set" : dmrs_port_set,
            "n_scid" : n_scid,
            "n_id" : n_id,
        },
        "tb" : {
            "mcs_index" : mcs_index,
            "mcs_table" : mcs_table,
        }
    }
    }
    
    num_bits_per_symbol, target_code_rate = sionna.nr.utils.select_mcs(config["pusch"]["tb"]["mcs_index"],
                                                                   config["pusch"]["tb"]["mcs_table"])
    config["pusch"]["tb"]["num_bits_per_symbol"] = num_bits_per_symbol
    config["pusch"]["tb"]["target_code_rate"] = target_code_rate

    json_object = json.dumps(config, indent=4)
    with open(filename + ".json", "w") as outfile:
        outfile.write(json_object)

i = 0
for n_size_bwp in [40, 273]:
    for precoding in ["non-codebook", "codebook"]:            
        ports = [1,2,4] if precoding=="non-codebook" else [2,4]
        for num_antenna_ports in ports:
            num_layers = num_antenna_ports
            dmrs_port_set = list(range(0,num_layers))
            for length in [1,2]:
                for config_type in [1,2]:
                    max_cdm_groups = 2 if config_type==1 else 3
                    min_cdm_groups = 1 if num_antenna_ports<4 else 2
                    for num_cdm_groups_without_data in range(min_cdm_groups,max_cdm_groups+1):
                        filename = f"test_{i}"
                        n_cell_id = config.np_rng.integers(0, 1008)
                        slot_number = config.np_rng.integers(0, 10)
                        n_rnti = config.np_rng.integers(0, 65536)
                        tpmi = 2
                        additional_position = 1
                        n_scid = config.np_rng.integers(0, 2)
                        n_id = config.np_rng.integers(0, 65536)
                        mcs_index = 14
                        mcs_table = 1
                        gen_config_file(filename,
                            n_cell_id,
                            slot_number,
                            n_size_bwp,
                            n_rnti,
                            num_antenna_ports,
                            num_layers,
                            precoding,
                            tpmi,
                            length,
                            config_type,
                            additional_position,
                            num_cdm_groups_without_data,
                            dmrs_port_set,
                            n_scid,
                            n_id,
                            mcs_index,
                            mcs_table)
                        i = i + 1
