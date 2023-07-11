#
# SPDX-FileCopyrightText: Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""PUSCH configuration for the nr (5G) sub-package of the Sionna library.
"""
# pylint: disable=line-too-long

import numpy as np
from .utils import generate_prng_seq
from .config import Config
from sionna import nr
from .utils import calculate_tb_size

class PUSCHConfig(Config):
    """
    The PUSCHConfig objects sets parameters for a physical uplink shared
    channel (PUSCH), as described in Sections 6.3 and 6.4 [3GPP38211]_.

    All configurable properties can be provided as keyword arguments during the
    initialization or changed later.

    Parameters
    ----------
    carrier_config : :class:`~sionna.nr.CarrierConfig` or `None`
        An instance of :class:`~sionna.nr.CarrierConfig`. If `None`, a
        :class:`~sionna.nr.CarrierConfig` instance with default settings
        will be created.

    pusch_dmrs_config : :class:`~sionna.nr.PUSCHDMRSConfig` or `None`
        An instance of :class:`~sionna.nr.PUSCHDMRSConfig`. If `None`, a
        :class:`~sionna.nr.PUSCHDMRSConfig` instance with default settings
        will be created.

    Example
    -------
    >>> pusch_config = PUSCHConfig(mapping_type="B")
    >>> pusch_config.dmrs.config_type = 2
    >>> pusch_config.carrier.subcarrier_spacing = 30
    """
    def __init__(self,
                 carrier_config=None,
                 pusch_dmrs_config=None,
                 tb_config=None,
                 **kwargs):
        super().__init__(**kwargs)
        self._name = "PUSCH Configuration"
        self.carrier = carrier_config
        self.dmrs = pusch_dmrs_config
        self.tb = tb_config
        self.check_config()

    #-----------------------------#
    #---Configurable parameters---#
    #-----------------------------#

    #---carrier---#
    @property
    def carrier(self):
        """
        :class:`~sionna.nr.CarrierConfig` : Carrier configuration
        """
        return self._carrier

    @carrier.setter
    def carrier(self, value):
        if value is None:
            value = nr.CarrierConfig()
        else:
            assert isinstance(value, nr.CarrierConfig), \
            "carrier must be an instance of CarrierConfig"
        self._carrier = value

    #---dmrs---#
    @property
    def dmrs(self):
        """
        :class:`~sionna.nr.PUSCHDMRSConfig` : PUSCH DMRS configuration
        """
        return self._dmrs

    @dmrs.setter
    def dmrs(self, value):
        if value is None:
            value = nr.PUSCHDMRSConfig()
        else:
            assert isinstance(value, nr.PUSCHDMRSConfig), \
            "pusch_dmrs_config must be an instance of PUSCHDMRSConfig"
        self._dmrs = value

    #---transport block---#
    @property
    def tb(self):
        """
        :class:`~sionna.nr.TBConfig` : Transport block configuration
        """
        return self._tb

    @tb.setter
    def tb(self, value):
        if value is None:
            value = nr.TBConfig(channel_type="PUSCH")
        else:
            assert isinstance(value, nr.TBConfig), \
            "tb must be an instance of TBConfig"
            assert value.channel_type=="PUSCH",\
                    'TBConfig must be configured for "PUSCH"'
        self._tb = value

    #---n_size_bwp---#
    @property
    def n_size_bwp(self):
        r"""
        int, None (default), [1,...,275] : Number of resource blocks in the
            bandwidth part (BWP) :math:`N^{\text{size},\mu}_{\text{BWP},i}`

            If set to `None`, the property
            :class:`~sionna.nr.CarrierConfig.n_size_grid` of
            `carrier` will be used.
        """
        self._ifndef("n_size_bwp", None)
        return self._n_size_bwp

    @n_size_bwp.setter
    def n_size_bwp(self, value):
        if value is not None:
            assert value in range(1,276),\
                "n_size_bwp must be in the range from 1 to 275"
        self._n_size_bwp = value

    #---n_start_bwp---#
    @property
    def n_start_bwp(self):
        r"""
        int, 0 (default) | [0,...,2199] : Start of BWP relative to
            common resource block (CRB) 0
            :math:`N^{\text{start},\mu}_{\text{BWP},i}`
        """
        self._ifndef("n_start_bwp", 0)
        return self._n_start_bwp

    @n_start_bwp.setter
    def n_start_bwp(self, value):
        assert value in range(0,2474), \
            "n_start_bwp must be in the range from 0 to 2473"
        self._n_start_bwp = value


    #---num-layers---#
    @property
    def num_layers(self):
        r"""
        int, 1 (default) | 2 | 3 | 4: Number of transmission layers
            :math:`\nu`

            Must be smaller than or equal to
            :class:`~sionna.nr.PUSCHConfig.num_antenna_ports`.
        """
        self._ifndef("num_layers", 1)
        return self._num_layers

    @num_layers.setter
    def num_layers(self, value):
        assert value in [1,2,3,4], "num_layers must be in [1,...,4]"
        self._num_layers = value

    #---num-antenna-ports---#
    @property
    def num_antenna_ports(self):
        """
        int, 1 (default) | 2 | 4: Number of antenna ports

            Must be larger than or equal to
            :class:`~sionna.nr.PUSCHConfig.num_layers`.
        """
        self._ifndef("num_antenna_ports", 1)
        return self._num_antenna_ports

    @num_antenna_ports.setter
    def num_antenna_ports(self, value):
        assert value in [1,2,4], "num_antenna_ports must be in [1,2,4]"
        self._num_antenna_ports = value

    #---mapping_type---#
    @property
    def mapping_type(self):
        """
        string, "A" (default) | "B": Mapping type
        """
        self._ifndef("mapping_type", "A")
        return self._mapping_type

    @mapping_type.setter
    def mapping_type(self, value):
        assert value in ["A","B"], "mapping_type must be A or B"
        self._mapping_type = value

    #---symbol_allocation---#
    @property
    def symbol_allocation(self):
        """
        2-tuple, int, [0, 14] (default) : PUSCH symbol allocation

            The first elements denotes the start of the symbol allocation.
            The second denotes the positive number of allocated OFDM symbols.
            For `mapping_type` "A", the first element must be zero.
            For `mapping_type` "B", the first element must be in
            [0,...,13]. The second element must be such that the index
            of the last allocated OFDM symbol is not larger than 13
            (for "normal" cyclic prefix) or 11 (for "extended" cyclic prefix).
        """
        self._ifndef("symbol_allocation", [0, 14])
        return self._symbol_allocation

    @symbol_allocation.setter
    def symbol_allocation(self, value):
        assert len(value)==2, "symbol_allocation must have two elements"
        self._symbol_allocation = value

    #---n_rnti---#
    @property
    def n_rnti(self):
        r"""
        int, 1 (default), [0,...,65535] : Radio network temporary identifier
            :math:`n_\text{RNTI}`
        """
        self._ifndef("n_rnti", 1)
        return self._n_rnti

    @n_rnti.setter
    def n_rnti(self, value):
        if value is not None:
            assert value in range(65536), "n_rnti must be in [0, 65535]"
        self._n_rnti = value

    #---transform_precoding---#
    @property
    def precoding(self):
        """
        str, "non-codebook" (default), "codebook" : PUSCH
            transmission scheme
        """
        self._ifndef("precoding", "non-codebook")
        return self._precoding

    @precoding.setter
    def precoding(self, value):
        assert value in ["codebook", "non-codebook"], \
            "Unknown value for precoding"
        self._precoding = value

    #---transform_precoding---#
    @property
    def transform_precoding(self):
        """
        bool, False (default): Use transform precoding
        """
        self._ifndef("transform_precoding", False)
        return self._transform_precoding

    @transform_precoding.setter
    def transform_precoding(self, value):
        assert isinstance(value, bool), \
            """transform_precoding must be bool"""
        self._transform_precoding = value

    #---tpmi---#
    @property
    def tpmi(self):
        r"""
        int,  0 (default) | [0,...,27] : Transmit precoding matrix indicator

            The allowed value depends on the number of layers and
            the number of antenna ports according to Table 6.3.1.5-1
            until Table 6.3.1.5-7 [3GPP38211]_.
        """
        self._ifndef("tpmi", 0)
        return self._tpmi

    @tpmi.setter
    def tpmi(self, value):
        assert value in range(28), "tpmi must be in [0,...,27]"
        self._tpmi = value

    #-----------------------------#
    #---Read-only parameters------#
    #-----------------------------#

    @property
    def frequency_hopping(self):
        """
        str, "neither" (default), read-only : Frequency hopping configuration
        """
        return "neither"

    @property
    def l_0(self):
        r"""
        int, read-only : Position of the first DMRS symbol :math:`l_0`
            relative to the reference `l_ref`.
        """
        if self.mapping_type=="A":
            return self.dmrs.type_a_position
        elif self.mapping_type=="B":
            return 0

    @property
    def l_d(self):
        r"""
        int, read-only : Length of the symbol allocation :math:`l_\text{d}`
        """
        return self.symbol_allocation[1]

    @property
    def l_ref(self):
        r"""
        int, read-only: Reference OFDM symbol index  used for DMRS
            generation
        """
        if self.mapping_type=="A":
            return 0
        elif self. mapping_type=="B":
            return self.symbol_allocation[0]

    @property
    def l_prime(self):
        r"""
        list, elements in [0,1], read-only : List of possible values of
            :math:`l'` used for DMRS generation
        """
        if self.dmrs.length==1:
            return [0]
        elif self.dmrs.length==2:
            return [0, 1]

    @property
    def l_bar(self):
        r"""
        list, elements in [0,...,11], read-only : List of possible values of
            :math:`\bar{l}` used for DMRS generation

            Defined in Tables 6.4.1.1.3-3 and 6.4.1.1.3-4 [3GPP38211]_.
        """
        l_0 = self.l_0
        ind = 0 if self.l_d<4 else self.l_d-3
        if self.mapping_type=="A":
            if self.dmrs.length==1:
                l_bar = [
                   [[],    [],        [],           []],
                   [[l_0], [l_0],     [l_0],        [l_0]],
                   [[l_0], [l_0],     [l_0],        [l_0]],
                   [[l_0], [l_0],     [l_0],        [l_0]],
                   [[l_0], [l_0],     [l_0],        [l_0]],
                   [[l_0], [l_0, 7],  [l_0, 7],     [l_0, 7]],
                   [[l_0], [l_0, 7],  [l_0, 7],     [l_0, 7]],
                   [[l_0], [l_0, 9],  [l_0, 6, 9],  [l_0, 6, 9]],
                   [[l_0], [l_0, 9],  [l_0, 6, 9],  [l_0, 6, 9]],
                   [[l_0], [l_0, 9],  [l_0, 6, 9],  [l_0, 5, 8, 11]],
                   [[l_0], [l_0, 11], [l_0, 7, 11], [l_0, 5, 8, 11]],
                   [[l_0], [l_0, 11], [l_0, 7, 11], [l_0, 5, 8, 11]]
                ]
            else:
                l_bar = [
                   [[],    []],
                   [[l_0], [l_0]],
                   [[l_0], [l_0]],
                   [[l_0], [l_0]],
                   [[l_0], [l_0]],
                   [[l_0], [l_0]],
                   [[l_0], [l_0]],
                   [[l_0], [l_0, 8]],
                   [[l_0], [l_0, 8]],
                   [[l_0], [l_0, 8]],
                   [[l_0], [l_0, 10]],
                   [[l_0], [l_0, 10]],
                ]
        elif self.mapping_type=="B":
            if self.dmrs.length==1:
                l_bar = [
                   [[l_0], [l_0],     [l_0],        [l_0]],
                   [[l_0], [l_0],     [l_0],        [l_0]],
                   [[l_0], [l_0, 4],  [l_0, 4],     [l_0, 4]],
                   [[l_0], [l_0, 4],  [l_0, 4],     [l_0, 4]],
                   [[l_0], [l_0, 4],  [l_0, 4],     [l_0, 4]],
                   [[l_0], [l_0, 6],  [l_0, 3, 6],  [l_0, 3, 6]],
                   [[l_0], [l_0, 6],  [l_0, 3, 6],  [l_0, 3, 6]],
                   [[l_0], [l_0, 8],  [l_0, 4, 8],  [l_0, 3, 6, 9]],
                   [[l_0], [l_0, 8],  [l_0, 4, 8],  [l_0, 3, 6, 9]],
                   [[l_0], [l_0, 10], [l_0, 5, 10], [l_0, 3, 6, 9]],
                   [[l_0], [l_0, 10], [l_0, 5, 10], [l_0, 3, 6, 9]],
                   [[l_0], [l_0, 10], [l_0, 5, 10], [l_0, 3, 6, 9]]
                ]
            else:
                l_bar = [
                   [[],    []],
                   [[],    []],
                   [[l_0], [l_0]],
                   [[l_0], [l_0]],
                   [[l_0], [l_0]],
                   [[l_0], [l_0, 5]],
                   [[l_0], [l_0, 5]],
                   [[l_0], [l_0, 7]],
                   [[l_0], [l_0, 7]],
                   [[l_0], [l_0, 9]],
                   [[l_0], [l_0, 9]],
                   [[l_0], [l_0, 9]],
                ]

        return l_bar[ind][self.dmrs.additional_position]

    @property
    def l(self):
        r"""
        list, int, read-only : List of possible values of the OFDM symbol
            indices :math:`l` carrying DMRS relative to :math:`l_0`
        """
        l = []
        for l_bar in self.l_bar:
            for l_prime in self.l_prime:
                l.append(l_bar + l_prime)
        return l

    @property
    def n(self):
        """
        list, int, read-only: List of possible values of n
            used for DMRS generation
        """
        if self.dmrs.config_type==1:
            n_max = self.num_resource_blocks*12//4 -1
        elif self.dmrs.config_type==2:
            n_max = self.num_resource_blocks*12//6 -1
        return list(range(n_max+1))

    @property
    def dmrs_symbol_indices(self):
        """
        list, int, read-only: Indices of DMRS symbols within a slot
        """
        return [l + self.l_ref for l in self.l]

    @property
    def num_resource_blocks(self):
        """
        int, read-only : Number of allocated resource blocks for the
            PUSCH transmissions.
        """
        if self.n_size_bwp is None:
            return self.carrier.n_size_grid
        else:
            return self.n_size_bwp

    @property
    def num_subcarriers(self):
        """
        int, read-only : Number of allocated subcarriers for the
            PUSCH transmissions
        """
        return 12*self.num_resource_blocks

    @property
    def num_res_per_prb(self):
        """
        int, read-only : Number of resource elements per PRB
            available for data
        """
        # Number of DMRS symbols
        num_dmrs = len(self.dmrs_symbol_indices)

        # Number of non-DMRS symbols
        num_data = self.symbol_allocation[1] - num_dmrs

        # Number of REs on DMRS symbols
        if self.dmrs.config_type==1:
            num_res_dmrs = 12 - 6*self.dmrs.num_cdm_groups_without_data
        elif self.dmrs.config_type==2:
            num_res_dmrs = 12 - 4*self.dmrs.num_cdm_groups_without_data

        # Number of REs on data symbols
        num_res_data = 12

        return num_data*num_res_data + num_dmrs*num_res_dmrs

    @property
    def dmrs_mask(self):
        """
        bool, [num_subcarriers, num_symbols_per_slot], read-only : Masked
            resource elements in the resource grid. `True` corresponds to
            resource elements on which no data is transmitted.
        """
        mask = np.zeros([self.num_subcarriers,
                         self.carrier.num_symbols_per_slot],
                         dtype=bool)

        num_cdm_groups = self.dmrs.num_cdm_groups_without_data
        if self.dmrs.config_type==1:
            cdm_ind = np.zeros([6, num_cdm_groups], np.int32)
            for i in range(num_cdm_groups):
                cdm_ind[:,i] = np.arange(i, 12, 2)
        else:
            cdm_ind = np.zeros([4, num_cdm_groups], np.int32)
            for i in range(num_cdm_groups):
                cdm_ind[:,i] = np.array([0,1, 6, 7])+2*i

        for i in self.dmrs_symbol_indices:
            for j in range(self.num_resource_blocks):
                for k in range(num_cdm_groups):
                    mask[cdm_ind[:, k] + 12*j, i] = True
        return mask

    @property
    def dmrs_grid(self):
        # pylint: disable=line-too-long
        """
        complex, [num_dmrs_ports, num_subcarriers, num_symbols_per_slot], read-only : Empty
            resource grid for each DMRS port, filled with DMRS signals

            This property returns for each configured DMRS port an empty
            resource grid filled with DMRS signals as defined in
            Section 6.4.1.1 [3GPP38211]. Not all possible options are implemented,
            e.g., frequency hopping and transform precoding are not available.

            This property provides the *unprecoded* DMRS for each configured DMRS port.
            Precoding might be applied to map the DMRS to the antenna ports. However,
            in this case, the number of DMRS ports cannot be larger than the number of
            layers.
        """
        # Check configuration
        self.check_config()

        # Configure DMRS ports set if it has not been set
        reset_dmrs_port_set = False
        if len(self.dmrs.dmrs_port_set)==0:
            self.dmrs.dmrs_port_set = list(range(self.num_layers))
            reset_dmrs_port_set = True

        # Generate empty resource grid for each port
        a_tilde = np.zeros([len(self.dmrs.dmrs_port_set),
                            self.num_subcarriers,
                            self.carrier.num_symbols_per_slot],
                            dtype=complex)

        # For every l_bar
        for l_bar in self.l_bar:

            # For every l_prime
            for l_prime in self.l_prime:

                # Compute c_init
                l = l_bar + l_prime
                c_init = self.c_init(l)

                # Generate RNG
                c = generate_prng_seq(2*self.num_subcarriers, c_init=c_init)

                # Map to QAM
                r = 1/np.sqrt(2)*((1-2*c[::2]) + 1j*(1-2*c[1::2]))

                # For every port in the dmrs port set
                for j_ind, _ in enumerate(self.dmrs.dmrs_port_set):

                    # For every n
                    for n in self.n:

                        # For every k_prime
                        for k_prime in [0, 1]:

                            if self.dmrs.config_type==1:
                                k = 4*n + 2*k_prime + \
                                    self.dmrs.deltas[j_ind]
                            elif self.dmrs.config_type==2:
                                k = 6*n + k_prime + \
                                    self.dmrs.deltas[j_ind]

                            a_tilde[j_ind, k, self.l_ref+l] = \
                                r[2*n + k_prime] * \
                                self.dmrs.w_f[k_prime][j_ind] * \
                                self.dmrs.w_t[l_prime][j_ind]

        # Amplitude scaling
        a = self.dmrs.beta*a_tilde

        # Reset DMRS port set if it was not set
        if reset_dmrs_port_set:
            self.dmrs.dmrs_port_set = []

        return a

    @property
    def dmrs_grid_precoded(self):
        if self.precoding=="non-codebook":
            return None

        w = np.expand_dims(np.expand_dims(self.precoding_matrix, 0), 0)
        a = np.expand_dims(np.transpose(self.dmrs_grid, [1,2,0]),-1)
        a = np.squeeze(np.matmul(w, a), -1)
        a = np.transpose(a, [2, 0, 1])

        return a

    @property
    def precoding_matrix(self):
        r"""
        nd_array, complex, [num_antenna_ports, numLayers] : Precoding matrix
            :math:`\mathbf{W}` as defined in
            Tables 6.3.1.5-1 to 6.3.1.5-7 [3GPP38211]_.

            Only relevant if :class:`~sionna.nr.PUSCHCONFIG.precoding`
            is "codebook".
        """
        if self.precoding=="non-codebook":
            return None
        if self.num_antenna_ports==1:
            return None
        w = None
        if self.num_layers==1:

            # Table 6.3.1.5-1
            if self.num_antenna_ports==2:
                w = np.zeros([6,2,1], complex)

                #TPMI index 0-5
                w[:,0,0] = [1,  0,  1,  1,  1,  1]
                w[:,1,0] = [0,  1,  1, -1, 1j,-1j]

                w /= np.sqrt(2)

            # Table 6.3.1.5-3
            elif self.num_antenna_ports==4:
                w = np.zeros([28,4,1], complex)

                # TPMI index 0-7
                w[:8,0,0] = [  1,  0,  0,  0,  1,  1,  1,  1]
                w[:8,1,0] = [  0,  1,  0,  0,  0,  0,  0,  0]
                w[:8,2,0] = [  0,  0,  1,  0,  1, -1, 1j,-1j]
                w[:8,3,0] = [  0,  0,  0,  1,  0,  0,  0,  0]

                # TPMI index 8-15
                w[8:16,0,0] = [  0,  0,  0,  0,  1,  1,  1,  1]
                w[8:16,1,0] = [  1,  1,  1,  1,  1,  1,  1,  1]
                w[8:16,2,0] = [  0,  0,  0,  0,  1, 1j, -1,-1j]
                w[8:16,3,0] = [  1, -1, 1j,-1j,  1, 1j, -1,-1j]

                # TPMI index 16-23
                w[16:24,0,0] = [  1,  1,  1,  1,  1,  1,  1,  1]
                w[16:24,1,0] = [ 1j, 1j, 1j, 1j, -1, -1, -1, -1]
                w[16:24,2,0] = [  1, 1j, -1,-1j,  1, 1j, -1,-1j]
                w[16:24,3,0] = [ 1j, -1,-1j,  1, -1,-1j,  1, 1j]

                # TPMI index 24-27
                w[24:28,0,0] = [  1,  1,  1,  1]
                w[24:28,1,0] = [-1j,-1j,-1j,-1j]
                w[24:28,2,0] = [  1, 1j, -1,-1j]
                w[24:28,3,0] = [-1j,  1, 1j, -1]

                w /= 2

        elif self.num_layers==2:

            # Table 6.3.1.5-4
            if self.num_antenna_ports==2:
                w = np.zeros([3,2,2], complex)

                # TPMI index 0-2
                w[0] = [[  1,  0], [  0,  1]]
                w[0] /= np.sqrt(2)
                w[1] = [[  1,  1], [  1, -1]]
                w[1] /= 2
                w[2] = [[  1,  1], [ 1j,-1j]]
                w[2] /= 2

            # Table 6.3.1.5-5
            elif self.num_antenna_ports==4:
                w = np.zeros([22,4,2], complex)

                # TPMI index 0-21
                w[0] = [[  1,  0], [  0,  1], [  0,  0], [  0,  0]]
                w[0] /= 2
                w[1] = [[  1,  0], [  0,  0], [  0,  1], [  0,  0]]
                w[1] /= 2
                w[2] = [[  1,  0], [  0,  0], [  0,  0], [  0,  1]]
                w[2] /= 2
                w[3] = [[  0,  0], [  1,  0], [  0,  1], [  0,  0]]
                w[3] /= 2
                w[4] = [[  0,  0], [  1,  0], [  0,  0], [  0,  1]]
                w[4] /= 2
                w[5] = [[  0,  0], [  0,  0], [  1,  0], [  0,  1]]
                w[5] /= 2
                w[6] = [[  1,  0], [  0,  1], [  1,  0], [  0,-1j]]
                w[6] /= 2
                w[7] = [[  1,  0], [  0,  1], [  1,  0], [  0, 1j]]
                w[7] /= 2
                w[8] = [[  1,  0], [  0,  1], [-1j,  0], [  0,  1]]
                w[8] /= 2
                w[9] = [[  1,  0], [  0,  1], [-1j,  0], [  0, -1]]
                w[9] /= 2
                w[10] = [[  1,  0], [  0,  1], [ -1,  0], [  0,-1j]]
                w[10] /= 2
                w[11] = [[  1,  0], [  0,  1], [ -1,  0], [  0, 1j]]
                w[11] /= 2
                w[12] = [[  1,  0], [  0,  1], [ 1j,  0], [  0,  1]]
                w[12] /= 2
                w[13] = [[  1,  0], [  0,  1], [ 1j,  0], [  0, -1]]
                w[13] /= 2
                w[14] = [[  1,  1], [  1,  1], [  1, -1], [  1, -1]]
                w[14] /= 2*np.sqrt(2)
                w[15] = [[  1,  1], [  1,  1], [ 1j,-1j], [ 1j,-1j]]
                w[15] /= 2*np.sqrt(2)
                w[16] = [[  1,  1], [ 1j, 1j], [  1, -1], [ 1j,-1j]]
                w[16] /= 2*np.sqrt(2)
                w[17] = [[  1,  1], [ 1j, 1j], [ 1j,-1j], [ -1,  1]]
                w[17] /= 2*np.sqrt(2)
                w[18] = [[  1,  1], [ -1, -1], [  1, -1], [ -1,  1]]
                w[18] /= 2*np.sqrt(2)
                w[19] = [[  1,  1], [ -1, -1], [ 1j,-1j], [-1j, 1j]]
                w[19] /= 2*np.sqrt(2)
                w[20] = [[  1,  1], [-1j,-1j], [  1, -1], [-1j, 1j]]
                w[20] /= 2*np.sqrt(2)
                w[21] = [[  1,  1], [-1j,-1j], [1j,-1j], [  1, -1]]
                w[21] /= 2*np.sqrt(2)

        elif self.num_layers==3:

            # Table 6.3.1.5-6
            if self.num_antenna_ports==4:
                w = np.zeros([7,4,3], complex)

                #TPMI index 0-6
                w[0] = [[  1,  0,  0],
                        [  0,  1,  0],
                        [  0,  0,  1],
                        [  0,  0,  0]]
                w[0] /= 2

                w[1] = [[  1,  0,  0],
                        [  0,  1,  0],
                        [  1,  0,  0],
                        [  0,  0,  1]]
                w[1] /= 2

                w[2] = [[  1,  0,  0],
                        [  0,  1,  0],
                        [ -1,  0,  0],
                        [  0,  0,  1]]
                w[2] /= 2

                w[3] = [[  1,  1,  1],
                        [  1, -1,  1],
                        [  1,  1, -1],
                        [  1, -1, -1]]
                w[3] /= (2*np.sqrt(3))

                w[4] = [[  1,  1,  1],
                        [  1, -1,  1],
                        [ 1j, 1j,-1j],
                        [ 1j,-1j,-1j]]
                w[4] /= (2*np.sqrt(3))

                w[5] = [[  1,  1,  1],
                        [ -1,  1, -1],
                        [  1,  1, -1],
                        [ -1,  1,  1]]
                w[5] /= (2*np.sqrt(3))

                w[6] = [[  1,  1,  1],
                        [ -1,  1, -1],
                        [ 1j, 1j,-1j],
                        [-1j, 1j, 1j]]
                w[6] /= (2*np.sqrt(3))

        elif self.num_layers==4:

            # Table 6.3.1.5-7
            if self.num_antenna_ports==4:
                w = np.zeros([5,4,4], complex)

                # TPMI index 0-4
                w[0] = [[  1,  0,  0,  0],
                        [  0,  1,  0,  0],
                        [  0,  0,  1,  0],
                        [  0,  0,  0,  1]]
                w[0] /= 2

                w[1] = [[  1,  1,  0,  0],
                        [  0,  0,  1,  1],
                        [  1, -1,  0,  0],
                        [  0,  0,  1, -1]]
                w[1] /= 2*np.sqrt(2)

                w[2] = [[  1,  1,  0,  0],
                        [  0,  0,  1,  1],
                        [ 1j,-1j,  0,  0],
                        [  0,  0, 1j,-1j]]
                w[2] /= 2*np.sqrt(2)

                w[3] = [[  1,  1,  1,  1],
                        [  1, -1,  1, -1],
                        [  1,  1, -1, -1],
                        [  1, -1, -1,  1]]
                w[3] /= 4

                w[4] = [[  1,  1,  1,  1],
                        [  1, -1,  1, -1],
                        [ 1j, 1j,-1j,-1j],
                        [ 1j,-1j,-1j, 1j]]
                w[4] /= 4

        if w is None:
            return w
        else:
            return w[self.tpmi]

    @property
    def num_ov(self):
        r"""
        int, 0 (default), read-only:  Number of unused resource elements due to additional overhead as specified by higher layer."""
        return 0

    @property
    def num_coded_bits(self):
        r"""
        int, read-only: Number of coded bits that fit into one PUSCH slot."""

        # compute number of data symbols
        n_re_per_prb = self.num_res_per_prb - self.num_ov

        # number of allocated REs
        n_re = n_re_per_prb * self.num_resource_blocks

        # total number of bits per slot
        num_coded_bits = int(self.tb.tb_scaling * self.tb.num_bits_per_symbol \
                             * self.num_layers * n_re)

        return num_coded_bits

    @property
    def tb_size(self):
        r"""int, read-only: Transport block size, i.e., how many information bits can be encoded into a slot for the given slot configuration."""

        # compute number of data symbols per prb
        n_re_per_prb = self.num_res_per_prb - self.num_ov

        # number of allocated REs
        # the max. number of REs per PRB is limited to 156 in 38.214
        n_re = min(156, n_re_per_prb) * self.num_resource_blocks

        # include tb_scaling as defined in Tab. 5.1.3.2-2 38.214
        target_tb_size = int(self.tb.target_coderate * self.tb.tb_scaling \
                        * n_re * self.tb.num_bits_per_symbol * self.num_layers)

        # and run tb_size calculation to account for quantization
        tb_size, _, _, _, _,_ = calculate_tb_size(
                            target_tb_size=target_tb_size,
                            num_coded_bits=self.num_coded_bits,
                            target_coderate=self.tb.target_coderate,
                            modulation_order=self.tb.num_bits_per_symbol,
                            verbose=False)

        return tb_size

    #-------------------#
    #---Class methods---#
    #-------------------#

    def c_init(self, l):
        # pylint: disable=line-too-long
        r"""Compute RNG initialization :math:`c_\text{init}` as in Section 6.4.1.1.1.1 [3GPP38211]_

        Input
        -----
            l : int
                OFDM symbol index relative to a reference :math:`l`

        Output
        ------
            c_init : int
                RNG initialization value
        """
        num_symbols_per_slot = self.carrier.num_symbols_per_slot
        slot_number = self.carrier.slot_number

        lambda_bar = 0
        n_scid_bar = self.dmrs.n_scid
        if self.dmrs.n_id is None:
            n_id = self.carrier.n_cell_id
        else:
            n_id = self.dmrs.n_id[n_scid_bar]

        c_init = np.mod(2**17 * (num_symbols_per_slot * slot_number + l + 1) \
                              * (2*n_id + 1) \
                        + 2**17 * np.floor(lambda_bar/2) \
                        + 2*n_id + n_scid_bar
                        , 2**31)

        return int(c_init)

    def show(self):
        """Print all properties of the PUSCHConfig and children"""
        self.carrier.show()
        Config.show(self)
        self.dmrs.show()
        self.tb.show()

    def check_config(self):
        """Test if the compound configuration is valid"""

        self.carrier.check_config()
        self.dmrs.check_config()
        if self.precoding=="codebook":
            # Check that dmrs_port_set matches number of layers
            if len(self.dmrs.dmrs_port_set)>0:
                assert len(self.dmrs.dmrs_port_set)==self.num_layers, \
                "num_layers must be equal to the number of dmrs ports"

            # Check that num_layers<=num_antenna_ports
            assert self.num_layers <= self.num_antenna_ports,\
                "num_layers must be <= num_antenna_ports"

            # Check that more than one antenna port is available
            assert self.num_antenna_ports>=2, \
                "precoding requires two or more antenna ports"
        else:
            # Check that num_layers==num_antenna_ports
            assert self.num_layers == self.num_antenna_ports,\
                "num_layers must be == num_antenna_ports"

        # Check Tables 6.4.1.1.3-3/4 are valid
        if self.dmrs.length==1:
            if self.mapping_type=="A":
                assert self.symbol_allocation[1]>=4, \
                    "Symbol allocation is too short"
        else:
            assert self.dmrs.additional_position<2, \
                "dmrs.additional_position must be <2 for this dmrs.length"
            assert self.symbol_allocation[1]>=4, "Symbol allocation too short"
            if self.mapping_type=="B":
                assert self.symbol_allocation[1]>=5, \
                    "Symbol allocation is too short"

        # Check type_a and additional_position position
        if self.mapping_type=="A":
            if self.dmrs.additional_position==3:
                assert self.dmrs.type_a_position==2,\
                "additional_position=3 only allowed for type_a_position=2"

        # Check for valid TMPI
        if self.num_layers==1:
            if self.num_antenna_ports==2:
                assert self.tpmi in range(6), "tpmi must be in [0,...,5]"
            elif self.num_antenna_ports==4:
                assert self.tpmi in range(28), "tpmi must be in [0,...,27]"
        elif self.num_layers==2:
            if self.num_antenna_ports==2:
                assert self.tpmi in range(3), "tpmi must be in [0,...,2]"
            elif self.num_antenna_ports==4:
                assert self.tpmi in range(22), "tpmi must be in [0,...,21]"
        elif self.num_layers==3:
            assert self.tpmi in range(7), "tpmi must be in [0,...,6]"
        elif self.num_layers==4:
            assert self.tpmi in range(5), "tpmi must be in [0,...,4]"

        # Check that symbol allocation is valid
        if self.carrier.cyclic_prefix=="normal":
            max_length = 14
        elif self.carrier.cyclic_prefix=="extended":
            max_length = 12
        if self.mapping_type=="A":
            assert self.symbol_allocation[0]==0, \
                "symbol_allocation[0] must be 0 for mapping_type A"
            assert 4<=self.symbol_allocation[1]<=max_length, \
                "symbol_allocation[1] must be in [4, 14 (or 12)]"
            if self.dmrs.length==2:
                assert self.symbol_allocation[1]>=4, \
                    "symbol_allocation[1] must be >=4 for dmrs.length==2"
        elif self.mapping_type=="B":
            assert 0<=self.symbol_allocation[0]<=13, \
                "symbol_allocation[0] must be in [0,13] for mapping_type B"
            assert 1<=self.symbol_allocation[1]<=max_length, \
                "symbol_allocation[1] must be in [1, 14 (or 12)]"
            if self.dmrs.length==2:
                assert self.symbol_allocation[1]>=5, \
                    "symbol_allocation[1] must be >=5 for dmrs.length==2"
        assert self.symbol_allocation[0] \
               + self.symbol_allocation[1]<=max_length, \
            "symbol_allocation[0]+symbol_allocation[1] must be < 14 (or 12)"

        attr_list = ["n_size_bwp",
                     "n_start_bwp",
                     "num_layers",
                     "mapping_type",
                     "symbol_allocation",
                     "n_rnti",
                     "precoding",
                     "transform_precoding",
                     "tpmi"
                    ]
        for attr in attr_list:
            value = getattr(self, attr)
            setattr(self, attr, value)

        # check that TBConfig is configured for "PUSCH"
        assert self.tb.channel_type=="PUSCH", \
                'TB_config must be configured for "PUSCH" transmission.'

        # Check that the number of DMRS ports equals the number of layers
        # if dmrs_port_set has been set. Otherwise, this is
        # automatically ensured.
        if len(self.dmrs.dmrs_port_set)>0:
            assert self.num_layers==len(self.dmrs.dmrs_port_set), \
                "num_layers must equal the number of DMRS ports"

        return True

def check_pusch_configs(pusch_configs):

    # Check that pusch_configs is a list
    assert isinstance(pusch_configs, list), \
        """pusch_configs must be a Sequence of instances of PUSCHConfig"""

    # Iterate through all pusch_configs and check their type and validity
    for pusch_config in pusch_configs:
        assert isinstance(pusch_config, PUSCHConfig), \
        """All elements of pusch_configs must be instances of PUSCHConfig"""

        pusch_config.check_config()

    # Create dictionary with extracted configuration parameters
    pc = pusch_configs[0]
    carrier = pc.carrier

    params = {
        "num_bits_per_symbol" : pc.tb.num_bits_per_symbol,
        "num_tx" : len(pusch_configs),
        "num_layers" : pc.num_layers,
        "num_subcarriers" : pc.num_subcarriers,
        "num_ofdm_symbols" : pc.symbol_allocation[1],
        "subcarrier_spacing" : pc.carrier.subcarrier_spacing*1e3,
        "num_antenna_ports" : pc.num_antenna_ports,
        "precoding" : pc.precoding,
        "precoding_matrices" : [],
        "pusch_config" : pc,
        "carrier_config" : pc.carrier,
        "num_coded_bits" : pc.num_coded_bits,
        "target_coderate" : pc.tb.target_coderate,
        "n_id" : [],
        "n_rnti" : [],
        "tb_size" : pc.tb_size,
        "dmrs_length" : pc.dmrs.length,
        "dmrs_additional_position" : pc.dmrs.additional_position,
        "num_cdm_groups_without_data" : pc.dmrs.num_cdm_groups_without_data
    }
    params["bandwidth"] = params["num_subcarriers"]*params["subcarrier_spacing"]
    params["cyclic_prefix_length"] = np.ceil(carrier.cyclic_prefix_length *
                                             params["bandwidth"])

    for pusch_config in pusch_configs:
        if params["precoding"]=="codebook":
            params["precoding_matrices"].append(pusch_config.precoding_matrix)

        # n_rnti and n_id are given per tx
        if pusch_config.tb.n_id is None:
            params["n_id"].append(pusch_config.carrier.n_cell_id)
        else:
            params["n_id"].append(pusch_config.tb.n_id)
        params["n_rnti"].append(pusch_config.n_rnti)

    return params
