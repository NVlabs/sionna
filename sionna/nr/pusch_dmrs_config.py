#
# SPDX-FileCopyrightText: Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""PUSCH DMRS configuration for the nr (5G) sub-package of the Sionna library.
"""
# pylint: disable=line-too-long

from collections.abc import Sequence
import numpy as np
from .config import Config

class PUSCHDMRSConfig(Config):
    """
    The PUSCHDMRSConfig objects sets parameters related to the generation
    of demodulation reference signals (DMRS) for a physical uplink shared
    channel (PUSCH), as described in Section 6.4.1.1 [3GPP38211]_.

    All configurable properties can be provided as keyword arguments during the
    initialization or changed later.

    Example
    -------
    >>> dmrs_config = PUSCHDMRSConfig(config_type=2)
    >>> dmrs_config.additional_position = 1
    """

    def __init__(self, **kwargs):
        self._name = "PUSCH DMRS Configuration"
        super().__init__(**kwargs)
        self.check_config()

    #-----------------------------#
    #---Configurable parameters---#
    #-----------------------------#

    #---config_type---#
    @property
    def config_type(self):
        """
        int, 1 (default) | 2 : DMRS configuration type

            The configuration type determines the frequency density of
            DMRS signals. With configuration type 1, six subcarriers per PRB are
            used for each antenna port, with configuration type 2, four
            subcarriers are used.
        """
        self._ifndef("config_type", 1)
        return self._config_type

    @config_type.setter
    def config_type(self, value):
        assert value in [1,2], "config_type must be in [1,2]"
        self._config_type = value

    #---type_a_position---#
    @property
    def type_a_position(self):
        """
        int, 2 (default) | 3 :  Position of first DMRS OFDM symbol

            Defines the position of the first DMRS symbol within a slot.
            This parameter only applies if the property
            :class:`~sionna.nr.PUSCHConfig.mapping_type` of
            :class:`~sionna.nr.PUSCHConfig` is equal to "A".
        """
        self._ifndef("type_a_position", 2)
        return self._type_a_position

    @type_a_position.setter
    def type_a_position(self, value):
        assert value in [2,3], "type_a_position must be in [2,3]"
        self._type_a_position = value

    #---additional_position---#
    @property
    def additional_position(self):
        """
        int, 0 (default) | 1 | 2 | 3 : Maximum number of additional DMRS positions

            The actual number of used DMRS positions depends on
            the length of the PUSCH symbol allocation.
        """
        self._ifndef("additional_position", 0)
        return self._additional_position

    @additional_position.setter
    def additional_position(self, value):
        assert value in [0,1,2,3], "additional_position must be in [0,1,2,3]"
        self._additional_position = value

    #---length---#
    @property
    def length(self):
        """
        int, 1 (default) | 2 : Number of front-loaded DMRS symbols
            A value of 1 corresponds to "single-symbol" DMRS, a value
            of 2 corresponds to "double-symbol" DMRS.
        """
        self._ifndef("length", 1)
        return self._length

    @length.setter
    def length(self, value):
        assert value in [1,2], "Invalid DMRS length"
        self._length = value

    #---dmrs_port_set---#
    @property
    def dmrs_port_set(self):
        """
        list, [] (default) | [0,...,11] : List of used DMRS antenna ports

            The elements in this list must all be from the list of
            `allowed_dmrs_ports` which depends on the `config_type` as well as
            the `length`. If set to `[]`, the port set will be equal to
            [0,...,num_layers-1], where
            :class:`~sionna.nr.PUSCHConfig.num_layers` is a property of the
            parent :class:`~sionna.nr.PUSCHConfig` instance.
        """
        self._ifndef("dmrs_port_set", [])
        return self._dmrs_port_set

    @dmrs_port_set.setter
    def dmrs_port_set(self, value):
        if isinstance(value, int):
            value = [value]
        elif isinstance(value, Sequence):
            value = list(value)
        else:
            raise ValueError("dmrs_port_set must be an integer or list")
        self._dmrs_port_set = value

    #---n_id---#
    @property
    def n_id(self):
        r"""
        2-tuple, None (default), [[0,...,65535], [0,...,65535]]: Scrambling
            identities

            Defines the scrambling identities :math:`N_\text{ID}^0` and
            :math:`N_\text{ID}^1` as a 2-tuple of integers. If `None`,
            the property :class:`~sionna.nr.CarrierConfig.n_cell_id` of the
            :class:`~sionna.nr.CarrierConfig` is used.
        """
        self._ifndef("n_id", None)
        return self._n_id

    @n_id.setter
    def n_id(self, value):
        if value is None:
            self._n_id = None
        elif isinstance(value, int):
            assert value in list(range(65536)), "n_id must be in [0, 65535]"
            self._n_id = [value, value]
        else:
            assert len(value)==2, "n_id must be either [] or a two-tuple"
            for e in value:
                assert e in list(range(65536)), "Each element of n_id must be in [0, 65535]"
            self._n_id = value

    #---n_scid---#
    @property
    def n_scid(self):
        r"""
        int, 0 (default) | 1 : DMRS scrambling initialization
            :math:`n_\text{SCID}`
        """
        self._ifndef("n_scid", 0)
        return self._n_scid

    @n_scid.setter
    def n_scid(self, value):
        assert value in [0, 1], "n_scid must be 0 or 1"
        self._n_scid = value

    #---num_cdm_groups_without_data---#
    @property
    def num_cdm_groups_without_data(self):
        """
        int, 2 (default) | 1 | 3 : Number of CDM groups without data

            This parameter controls how many REs are available for data
            transmission in a DMRS symbol. It should be greater or equal to
            the maximum configured number of CDM groups. A value of
            1 corresponds to CDM group 0, a value of 2 corresponds to
            CDM groups 0 and 1, and a value of 3 corresponds to
            CDM groups 0, 1, and 2.
        """
        self._ifndef("num_cdm_groups_without_data", 2)
        return self._num_cdm_groups_without_data

    @num_cdm_groups_without_data.setter
    def num_cdm_groups_without_data(self, value):
        assert value in [1,2,3], \
            "num_cdm_groups_without_data must be in [1,2,3]"
        self._num_cdm_groups_without_data = value

    #-----------------------------#
    #---Read-only parameters------#
    #-----------------------------#

    @property
    def allowed_dmrs_ports(self):
        """
        list, [0,...,max_num_dmrs_ports-1], read-only : List of nominal antenna
            ports

            The maximum number of allowed antenna ports `max_num_dmrs_ports`
            depends on the DMRS `config_type` and `length`. It can be
            equal to 4, 6, 8, or 12.
        """
        if self.length==1:
            if self.config_type==1:
                if self.num_cdm_groups_without_data==1:
                    return [0,1]
                else:
                    return [0,1,2,3]
                #max_num_dmrs_ports = self.num_cdm_groups_without_data*2
            elif self.config_type==2:
                if self.num_cdm_groups_without_data==1:
                    return [0,1]
                elif self.num_cdm_groups_without_data==2:
                    return [0,1,2,3]
                else:
                    return [0,1,2,3,4,5]
                #max_num_dmrs_ports = self.num_cdm_groups_without_data*2
        elif self.length==2:
            if self.config_type==1:
                if self.num_cdm_groups_without_data==1:
                    return [0,1,4,5]
                else:
                    return [0,1,2,3,4,5,6,7]
                #max_num_dmrs_ports = self.num_cdm_groups_without_data*4
            elif self.config_type==2:
                if self.num_cdm_groups_without_data==1:
                    return [0,1,6,7]
                elif self.num_cdm_groups_without_data==2:
                    return [0,1,2,3,6,7,8,9]
                else:
                    return [0,1,2,3,4,5,6,7,8,9,10,11]
                #max_num_dmrs_ports = self.num_cdm_groups_without_data*4
        #return list(range(max_num_dmrs_ports))

    @property
    def cdm_groups(self):
        r"""
        list, elements in [0,1,2], read-only : List of CDM groups
            :math:`\lambda` for all ports
            in the `dmrs_port_set` as defined in
            Table 6.4.1.1.3-1 or 6.4.1.1.3-2 [3GPP38211]_

            Depends on the `config_type`.
        """
        if self.config_type==1:
            cdm_groups = [0,0,1,1,0,0,1,1]
        else:
            cdm_groups = [0,0,1,1,2,2,0,0,1,1,2,2]
        return [cdm_groups[port] for port in self.dmrs_port_set]

    @property
    def deltas(self):
        r"""
        list, elements in [0,1,2,4], read-only : List of delta (frequency)
            shifts :math:`\Delta` for all ports in the `port_set` as defined in
            Table 6.4.1.1.3-1 or 6.4.1.1.3-2 [3GPP38211]_

            Depends on the `config_type`.
        """
        if self.config_type==1:
            deltas = [0,0,1,1,0,0,1,1]
        else:
            deltas = [0,0,2,2,4,4,0,0,2,2,4,4]
        return [deltas[port] for port in self.dmrs_port_set]

    @property
    def w_f(self):
        r"""
        matrix, elements in [-1,1], read-only : Frequency weight vectors
            :math:`w_f(k')` for all ports in the port set as defined in
            Table 6.4.1.1.3-1 or 6.4.1.1.3-2 [3GPP38211]_
        """
        if self.config_type==1:
            w_f = np.array([[1, 1,1, 1,1, 1,1, 1],
                            [1,-1,1,-1,1,-1,1,-1]])
        elif self.config_type==2:
            w_f = np.array([[1, 1,1, 1,1, 1,1, 1,1, 1,1, 1],
                            [1,-1,1,-1,1,-1,1,-1,1,-1,1,-1]])
        return w_f[:, self.dmrs_port_set]

    @property
    def w_t(self):
        r"""
        matrix, elements in [-1,1], read-only : Time weight vectors
            :math:`w_t(l')` for all ports in the port set as defined in
            Table 6.4.1.1.3-1 or 6.4.1.1.3-2 [3GPP38211]_
        """
        if self.config_type==1:
            w_t = np.array([[1,1,1,1, 1, 1, 1, 1],
                           [1,1,1,1,-1,-1,-1,-1]])
        elif self.config_type==2:
            w_t = np.array([[1,1,1,1,1,1, 1, 1, 1, 1, 1, 1],
                            [1,1,1,1,1,1,-1,-1,-1,-1,-1,-1]])
        return w_t[:, self.dmrs_port_set]

    @property
    def beta(self):
        r"""
        float, read-only : Ratio of PUSCH energy per resource element
            (EPRE) to DMRS EPRE :math:`\beta^{\text{DMRS}}_\text{PUSCH}`
            Table 6.2.2-1 [3GPP38214]_
        """
        if self.num_cdm_groups_without_data==1:
            return 1.0
        elif self.num_cdm_groups_without_data==2:
            return np.sqrt(2)
        elif self.num_cdm_groups_without_data==3:
            if self.config_type==2:
                return np.sqrt(3)

    #-------------------#
    #---Class methods---#
    #-------------------#

    def check_config(self):
        """Test if configuration is valid"""

        if self.length==2:
            assert self.additional_position in [0, 1], \
                "additional_position must be in [0, 1] for length==2"

        for p in self.dmrs_port_set:
            assert p in self.allowed_dmrs_ports,\
                f"Unallowed DMRS port {p}. Not in {self.allowed_dmrs_ports}."

        if self.config_type==1:
            assert self.num_cdm_groups_without_data in [1, 2], \
            "num_cdm_groups_without_data must be in [1,2] for config_type 1"

        attr_list = ["config_type",
                     "type_a_position",
                     "additional_position",
                     "length",
                     "dmrs_port_set",
                     "n_id",
                     "n_scid",
                     "num_cdm_groups_without_data"
                    ]
        for attr in attr_list:
            value = getattr(self, attr)
            setattr(self, attr, value)
