#
# SPDX-FileCopyrightText: Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""PUSCH pilot pattern for the nr (5G) sub-package of the Sionna library.
"""
import warnings
from collections.abc import Sequence
import tensorflow as tf
import numpy as np
from sionna.ofdm import PilotPattern
from .pusch_config import PUSCHConfig

class PUSCHPilotPattern(PilotPattern):
    # pylint: disable=line-too-long
    r"""Class defining a pilot pattern for NR PUSCH.

    This class defines a :class:`~sionna.ofdm.PilotPattern`
    that is used to configure an OFDM :class:`~sionna.ofdm.ResourceGrid`.

    For every transmitter, a separte :class:`~sionna.nr.PUSCHConfig`
    needs to be provided from which the pilot pattern will be created.

    Parameters
    ----------
    pusch_configs : instance or list of :class:`~sionna.nr.PUSCHConfig`
        PUSCH Configurations according to which the pilot pattern
        will created. One configuration is needed for each transmitter.

    dtype : tf.Dtype
        Defines the datatype for internal calculations and the output
        dtype. Defaults to `tf.complex64`.
    """
    def __init__(self,
                 pusch_configs,
                 dtype=tf.complex64):

        # Check correct type of pusch_configs
        if isinstance(pusch_configs, PUSCHConfig):
            pusch_configs = [pusch_configs]
        elif isinstance(pusch_configs, Sequence):
            for c in pusch_configs:
                assert isinstance(c, PUSCHConfig), \
                    "Each element of pusch_configs must be a valide PUSCHConfig"
        else:
            raise ValueError("Invalid value for pusch_configs")

        # Check validity of provided pusch_configs
        num_tx = len(pusch_configs)
        num_streams_per_tx = pusch_configs[0].num_layers
        dmrs_grid = pusch_configs[0].dmrs_grid
        num_subcarriers = dmrs_grid[0].shape[0]
        num_ofdm_symbols = pusch_configs[0].l_d
        precoding = pusch_configs[0].precoding
        dmrs_ports = []
        num_pilots = np.sum(pusch_configs[0].dmrs_mask)
        for pusch_config in pusch_configs:
            assert pusch_config.num_layers==num_streams_per_tx, \
                "All pusch_configs must have the same number of layers"
            assert pusch_config.dmrs_grid[0].shape[0]==num_subcarriers, \
                "All pusch_configs must have the same number of subcarriers"
            assert pusch_config.l_d==num_ofdm_symbols, \
                "All pusch_configs must have the same number of OFDM symbols"
            assert pusch_config.precoding==precoding, \
                "All pusch_configs must have a the same precoding method"
            assert np.sum(pusch_config.dmrs_mask)==num_pilots, \
                "All pusch_configs must have a the same number of masked REs"
            with warnings.catch_warnings():
                warnings.simplefilter('always')
                for port in pusch_config.dmrs.dmrs_port_set:
                    if port in dmrs_ports:
                        msg = f"DMRS port {port} used by multiple transmitters"
                        warnings.warn(msg)
            dmrs_ports += pusch_config.dmrs.dmrs_port_set

        # Create mask and pilots tensors
        mask = np.zeros([num_tx,
                         num_streams_per_tx,
                         num_ofdm_symbols,
                         num_subcarriers], bool)
        num_pilots = np.sum(pusch_configs[0].dmrs_mask)
        pilots = np.zeros([num_tx, num_streams_per_tx, num_pilots], complex)
        for i, pusch_config in enumerate(pusch_configs):
            for j in range(num_streams_per_tx):
                ind0, ind1 = pusch_config.symbol_allocation
                mask[i,j] = np.transpose(
                                pusch_config.dmrs_mask[:, ind0:ind0+ind1])
                dmrs_grid = np.transpose(
                                pusch_config.dmrs_grid[j, :, ind0:ind0+ind1])
                pilots[i,j] = dmrs_grid[np.where(mask[i,j])]

        # Init PilotPattern class
        super().__init__(mask, pilots,
                         trainable=False,
                         normalize=False,
                         dtype=dtype)
