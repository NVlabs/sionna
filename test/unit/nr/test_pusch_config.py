#
# SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
import sys
import os
import unittest
import numpy as np
import tensorflow as tf
import sionna
from sionna.nr import PUSCHConfig
from sionna import config

script_dir = os.path.dirname(os.path.abspath(__file__))

class TestPUSCHDMRS(unittest.TestCase):
    """Tests for the PUSCHDMRS Class"""

    def test_against_reference_1(self):
        """Test that DMRS pattenrs match a reference implementation"""
        reference_dmrs = np.load(script_dir+"/reference_dmrs_1.npy")
        pusch_config = PUSCHConfig()
        pusch_config.carrier.n_size_grid = 1
        pusch_config.dmrs.config_type = 2
        pusch_config.dmrs.num_cdm_groups_without_data=3
        pusch_config.dmrs.additional_position = 1
        pusch_config.dmrs.length = 2
        pusch_config.dmrs.n_id = [4,4]
        p = []
        for n_cell_id in [0,1,10,24,99,1006]:
            for slot_number in [0,1,5,9]:
                for port_set in [0,3,4,9,11]:
                    pusch_config.carrier.n_cell_id = n_cell_id
                    pusch_config.carrier.slot_number=slot_number
                    pusch_config.dmrs.dmrs_port_set = [port_set]
                    a = pusch_config.dmrs_grid
                    pilots = np.concatenate([a[0,:,2], a[0,:,3], a[0,:,10], a[0,:,11]])
                    pilots = pilots[np.where(pilots)]/np.sqrt(3)
                    p.append(pilots)
        pilots = np.transpose(np.array(p))
        self.assertTrue(np.allclose(pilots, reference_dmrs))

    def test_against_reference_2(self):
        """Test that DMRS pattenrs match a reference implementation"""
        reference_dmrs = np.load(script_dir+"/reference_dmrs_2.npy")

        pusch_config = PUSCHConfig()
        pusch_config.carrier.n_size_grid = 4
        pusch_config.dmrs.config_type = 2
        pusch_config.dmrs.num_cdm_groups_without_data=3
        pusch_config.dmrs.additional_position = 1
        pusch_config.dmrs.length = 2
        pusch_config.dmrs.n_id = [4,4]
        p = []
        for n_cell_id in [0,1,10,24,99,1006]:
            for slot_number in [0,1,5,9]:
                for port_set in [0,3,4,9,11]:
                    pusch_config.carrier.n_cell_id = n_cell_id
                    pusch_config.carrier.slot_number=slot_number
                    pusch_config.dmrs.dmrs_port_set = [port_set]
                    a = pusch_config.dmrs_grid
                    pilots = np.concatenate([a[0,:,2], a[0,:,3], a[0,:,10], a[0,:,11]])
                    pilots = pilots[np.where(pilots)]/np.sqrt(3)
                    p.append(pilots)
        pilots = np.transpose(np.array(p))
        self.assertTrue(np.allclose(pilots, reference_dmrs))

    def test_orthogonality_over_resource_grid(self):
        """Test that DMRS for different ports are orthogonal
           accross a resource grid by computing the LS estimate
           on a noise less block-constant channel
        """
        def ls_estimate(pusch_config):
            """Assigns a random channel coefficient to each port
               and computes the LS estimate
            """
            a = pusch_config.dmrs_grid
            channel = config.np_rng.random([a.shape[0], 1, 1])
            y = np.sum(channel*a, axis=0)
            for i, port in enumerate(a):
                ind = np.where(port)
                port = port[ind]

                # LS Estimate
                z = y[ind]*np.conj(port)/np.abs(port)**2

                # Time-domain averaging of CDMs for DMRSLength=2
                if pusch_config.dmrs.length==2: 
                    l = len(pusch_config.dmrs_symbol_indices)
                    z = np.reshape(z, [-1, l])
                    z = np.reshape(z, [-1, l//2, 2])
                    z = np.mean(z, axis=-1)

                # Frequency-domain averaging of CDMs
                if pusch_config.dmrs.config_type==1:
                    num_freq_pilots = 6 * pusch_config.carrier.n_size_grid
                else:
                    num_freq_pilots = 4 * pusch_config.carrier.n_size_grid
                z = np.reshape(z, [num_freq_pilots//2,2,-1])
                z = np.mean(z, axis=1)

                return np.allclose(z-channel[i], 0)

        pusch_config = PUSCHConfig()
        pusch_config.carrier.n_size_grid = 4
        pusch_config.dmrs.config_type = 2
        pusch_config.dmrs.length = 2
        pusch_config.dmrs.num_cdm_groups_without_data = 3
        pusch_config.dmrs.dmrs_port_set = [1,2,5,11]
        pusch_config.num_layers = 4
        pusch_config.num_antenna_ports = 4
        for mapping_type in ["A", "B"]:
            for additional_position in range(2):
                for type_a_position in [2,3]:
                    for num_symbols in range(1,15):
                        try :
                            pusch_config.mapping_type = mapping_type
                            pusch_config.dmrs.type_a_position = type_a_position
                            pusch_config.dmrs.additional_position = additional_position
                            pusch_config.symbol_allocation = [0, num_symbols]
                            pusch_config.check_config()
                        except:
                            continue
                        self.assertTrue(ls_estimate(pusch_config))

        pusch_config = PUSCHConfig()
        pusch_config.carrier.n_size_grid = 4
        pusch_config.dmrs.config_type = 2
        pusch_config.dmrs.length = 1
        pusch_config.dmrs.num_cdm_groups_without_data = 3
        pusch_config.dmrs.port_set = [2,3,4,5]
        pusch_config.num_layers = 4
        pusch_config.num_antenna_ports = 4
        for mapping_type in ["A", "B"]:
            for additional_position in range(2):
                for type_a_position in [2,3]:
                    for num_symbols in range(1,15):
                        try :
                            pusch_config.mapping_type = mapping_type
                            pusch_config.dmrs.type_a_position = type_a_position
                            pusch_config.dmrs.additional_position = additional_position
                            pusch_config.symbol_allocation = [0, num_symbols]
                            pusch_config.check_config()
                        except:
                            continue
                        self.assertTrue(ls_estimate(pusch_config))

        pusch_config = PUSCHConfig()
        pusch_config.carrier.n_size_grid = 4
        pusch_config.dmrs.config_type = 1
        pusch_config.dmrs.length = 1
        pusch_config.dmrs.num_cdm_groups_without_data = 2
        pusch_config.dmrs.port_set = [0,1,2,3]
        pusch_config.num_layers = 4
        pusch_config.num_antenna_ports = 4
        for mapping_type in ["A", "B"]:
            for additional_position in range(2):
                for type_a_position in [2,3]:
                    for num_symbols in range(1,15):
                        try :
                            pusch_config.mapping_type = mapping_type
                            pusch_config.dmrs.type_a_position = type_a_position
                            pusch_config.dmrs.additional_position = additional_position
                            pusch_config.symbol_allocation = [0, num_symbols]
                            pusch_config.check_config()
                        except:
                            continue
                        self.assertTrue(ls_estimate(pusch_config))


    def test_precoding_against_reference(self):
        "Test precoded DMRS against reference implementation"

        pusch_config = PUSCHConfig()
        pusch_config.carrier.n_size_grid = 1
        pusch_config.carrier.slot_number = 1
        pusch_config.dmrs.additional_position = 0
        pusch_config.dmrs.config_type = 2
        pusch_config.dmrs.num_cdm_groups_without_data=3
        pusch_config.dmrs.length = 2
        pusch_config.dmrs.n_id = [8,8]
        pusch_config.precoding = "codebook"

        # 1-Layer 2-Antenna Ports
        pusch_config.num_layers = 1
        pusch_config.num_antenna_ports = 2
        ref = np.load(script_dir+f"/pusch_dmrs_precoded_{pusch_config.num_layers}_layer_{pusch_config.num_antenna_ports}_ports.npy", allow_pickle=True)
        for i in range(6):
            pusch_config.tpmi = i
            self.assertTrue(np.allclose(pusch_config.dmrs_grid_precoded/np.sqrt(3), ref[i]))

        # 1-Layer 4-Antenna Ports
        pusch_config.num_layers = 1
        pusch_config.num_antenna_ports = 4
        ref = np.load(script_dir+f"/pusch_dmrs_precoded_{pusch_config.num_layers}_layer_{pusch_config.num_antenna_ports}_ports.npy", allow_pickle=True)
        for i in range(28):
            pusch_config.tpmi = i
            self.assertTrue(np.allclose(pusch_config.dmrs_grid_precoded/np.sqrt(3), ref[i]))

        # 2-Layer 2-Antenna Ports
        pusch_config.num_layers = 2
        pusch_config.num_antenna_ports = 2
        ref = np.load(script_dir+f"/pusch_dmrs_precoded_{pusch_config.num_layers}_layer_{pusch_config.num_antenna_ports}_ports.npy", allow_pickle=True)
        for i in range(3):
            pusch_config.tpmi = i
            self.assertTrue(np.allclose(pusch_config.dmrs_grid_precoded/np.sqrt(3), ref[i]))

        # 2-Layer 4-Antenna Ports
        pusch_config.num_layers = 2
        pusch_config.num_antenna_ports = 4
        ref = np.load(script_dir+f"/pusch_dmrs_precoded_{pusch_config.num_layers}_layer_{pusch_config.num_antenna_ports}_ports.npy", allow_pickle=True)
        for i in range(22):
            pusch_config.tpmi = i
            self.assertTrue(np.allclose(pusch_config.dmrs_grid_precoded/np.sqrt(3), ref[i]))

        # 3-Layer 4-Antenna Ports
        pusch_config.num_layers = 3
        pusch_config.num_antenna_ports = 4
        ref = np.load(script_dir+f"/pusch_dmrs_precoded_{pusch_config.num_layers}_layer_{pusch_config.num_antenna_ports}_ports.npy", allow_pickle=True)
        for i in range(7):
            pusch_config.tpmi = i
            self.assertTrue(np.allclose(pusch_config.dmrs_grid_precoded/np.sqrt(3), ref[i]))

        # 4-Layer 4-Antenna Ports
        pusch_config.num_layers = 4
        pusch_config.num_antenna_ports = 4
        ref = np.load(script_dir+f"/pusch_dmrs_precoded_{pusch_config.num_layers}_layer_{pusch_config.num_antenna_ports}_ports.npy", allow_pickle=True)
        for i in range(5):
            pusch_config.tpmi = i
            self.assertTrue(np.allclose(pusch_config.dmrs_grid_precoded/np.sqrt(3), ref[i]))
