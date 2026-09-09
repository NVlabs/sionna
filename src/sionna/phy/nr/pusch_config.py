#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""PUSCH configuration for the 5G NR module of Sionna PHY."""

from typing import List, Optional, Tuple, Union
import numpy as np

from sionna._validation import check_instance, check_sequence_of
from sionna.phy import nr
from .config import Config
from .utils import generate_prng_seq, calculate_tb_size


__all__ = ["PUSCHConfig", "check_pusch_configs"]

_TRANSFORM_PRECODING_ERROR = (
    "PUSCH transform precoding is not implemented"
)


class PUSCHConfig(Config):
    r"""Configuration for a physical uplink shared channel (PUSCH).

    Implements parameters as described in Sections 6.3 and 6.4 :cite:p:`3GPPTS38211`.
    All configurable properties can be provided as keyword arguments during
    initialization or changed later.

    :param carrier_config: If `None`, a
        :class:`~sionna.phy.nr.CarrierConfig` instance with default
        settings will be created.
    :param pusch_dmrs_config: If `None`, a
        :class:`~sionna.phy.nr.PUSCHDMRSConfig` instance with default
        settings will be created.
    :param tb_config: If `None`, a
        :class:`~sionna.phy.nr.TBConfig` instance with default
        settings will be created.

    .. rubric:: Examples

    .. code-block:: python

        from sionna.phy.nr import PUSCHConfig

        pusch_config = PUSCHConfig(mapping_type="B")
        pusch_config.dmrs.config_type = 2
        pusch_config.carrier.subcarrier_spacing = 30
    """

    def __init__(
        self,
        carrier_config: Optional["nr.CarrierConfig"] = None,
        pusch_dmrs_config: Optional["nr.PUSCHDMRSConfig"] = None,
        tb_config: Optional["nr.TBConfig"] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._name = "PUSCH Configuration"
        self.carrier = carrier_config
        self.dmrs = pusch_dmrs_config
        self.tb = tb_config
        self.check_config()

    #-----------------------------#
    #---Configurable parameters---#
    #-----------------------------#

    @property
    def carrier(self) -> "nr.CarrierConfig":
        """:class:`~sionna.phy.nr.CarrierConfig`: Carrier configuration."""
        return self._carrier

    @carrier.setter
    def carrier(self, value: Optional["nr.CarrierConfig"]) -> None:
        if value is None:
            value = nr.CarrierConfig()
        elif not isinstance(value, nr.CarrierConfig):
            raise TypeError("carrier must be an instance of CarrierConfig")
        self._carrier = value

    @property
    def dmrs(self) -> "nr.PUSCHDMRSConfig":
        """:class:`~sionna.phy.nr.PUSCHDMRSConfig`: PUSCH DMRS configuration."""
        return self._dmrs

    @dmrs.setter
    def dmrs(self, value: Optional["nr.PUSCHDMRSConfig"]) -> None:
        if value is None:
            value = nr.PUSCHDMRSConfig()
        elif not isinstance(value, nr.PUSCHDMRSConfig):
            raise TypeError(
                "pusch_dmrs_config must be an instance of PUSCHDMRSConfig")
        self._dmrs = value

    @property
    def tb(self) -> "nr.TBConfig":
        """:class:`~sionna.phy.nr.TBConfig`: Transport block configuration."""
        return self._tb

    @tb.setter
    def tb(self, value: Optional["nr.TBConfig"]) -> None:
        if value is None:
            value = nr.TBConfig(channel_type="PUSCH")
        elif not isinstance(value, nr.TBConfig):
            raise TypeError("tb must be an instance of TBConfig")
        elif value.channel_type != "PUSCH":
            raise ValueError('TBConfig must be configured for "PUSCH"')
        self._tb = value

    @property
    def n_size_bwp(self) -> Optional[int]:
        r"""Number of resource blocks in the BWP :math:`N^{\text{size},\mu}_{\text{BWP},i}`.

        Defaults to `None`. Must be in [1, ..., 275].
        If set to `None`, the property
        :attr:`~sionna.phy.nr.CarrierConfig.n_size_grid` of
        ``carrier`` will be used.
        """
        self._ifndef("n_size_bwp", None)
        return self._n_size_bwp

    @n_size_bwp.setter
    def n_size_bwp(self, value: Optional[int]) -> None:
        if value is not None and value not in range(1, 276):
            raise ValueError("n_size_bwp must be in the range from 1 to 275")
        self._n_size_bwp = value

    @property
    def n_start_bwp(self) -> int:
        r"""Start of BWP relative to common resource block (CRB) 0 :math:`N^{\text{start},\mu}_{\text{BWP},i}`.

        Defaults to 0. Must be in [0, ..., 2473].
        """
        self._ifndef("n_start_bwp", 0)
        return self._n_start_bwp

    @n_start_bwp.setter
    def n_start_bwp(self, value: int) -> None:
        if value not in range(0, 2474):
            raise ValueError("n_start_bwp must be in the range from 0 to 2473")
        self._n_start_bwp = value

    @property
    def num_layers(self) -> int:
        r"""Number of transmission layers :math:`\nu`.

        Defaults to 1. Must be in [1, 2, 3, 4].
        Must be smaller than or equal to
        :attr:`~sionna.phy.nr.PUSCHConfig.num_antenna_ports`.
        """
        self._ifndef("num_layers", 1)
        return self._num_layers

    @num_layers.setter
    def num_layers(self, value: int) -> None:
        if value not in [1, 2, 3, 4]:
            raise ValueError("num_layers must be in [1, 2, 3, 4]")
        self._num_layers = value

    @property
    def num_antenna_ports(self) -> int:
        """Number of antenna ports.

        Defaults to 1. Must be in [1, 2, 4].
        Must be larger than or equal to
        :attr:`~sionna.phy.nr.PUSCHConfig.num_layers`.
        """
        self._ifndef("num_antenna_ports", 1)
        return self._num_antenna_ports

    @num_antenna_ports.setter
    def num_antenna_ports(self, value: int) -> None:
        if value not in [1, 2, 4]:
            raise ValueError("num_antenna_ports must be in [1, 2, 4]")
        self._num_antenna_ports = value

    @property
    def mapping_type(self) -> str:
        """Mapping type. Defaults to ``"A"``. Must be ``"A"`` or ``"B"``."""
        self._ifndef("mapping_type", "A")
        return self._mapping_type

    @mapping_type.setter
    def mapping_type(self, value: str) -> None:
        if value not in ["A", "B"]:
            raise ValueError("mapping_type must be 'A' or 'B'")
        self._mapping_type = value

    @property
    def symbol_allocation(self) -> List[int]:
        """PUSCH symbol allocation.

        Defaults to [0, 14].
        The first element denotes the start of the symbol allocation.
        The second denotes the positive number of allocated OFDM symbols.
        For ``mapping_type`` ``"A"``, the first element must be zero.
        For ``mapping_type`` ``"B"``, the first element must be in
        [0, ..., 13]. The second element must be such that the index
        of the last allocated OFDM symbol is not larger than 13
        (for ``"normal"`` cyclic prefix) or 11 (for ``"extended"`` cyclic
        prefix).
        """
        self._ifndef("symbol_allocation", [0, 14])
        return self._symbol_allocation

    @symbol_allocation.setter
    def symbol_allocation(self, value: List[int]) -> None:
        if len(value) != 2:
            raise ValueError("symbol_allocation must have two elements")
        self._symbol_allocation = list(value)

    @property
    def n_rnti(self) -> int:
        r"""Radio network temporary identifier :math:`n_\text{RNTI}`.

        Defaults to 1. Must be in [0, ..., 65535].
        """
        self._ifndef("n_rnti", 1)
        return self._n_rnti

    @n_rnti.setter
    def n_rnti(self, value: Optional[int]) -> None:
        if value is not None and value not in range(65536):
            raise ValueError("n_rnti must be in [0, 65535]")
        self._n_rnti = value

    @property
    def precoding(self) -> str:
        """PUSCH transmission scheme.

        Defaults to ``"non-codebook"``. Must be ``"codebook"`` or
        ``"non-codebook"``.
        """
        self._ifndef("precoding", "non-codebook")
        return self._precoding

    @precoding.setter
    def precoding(self, value: str) -> None:
        if value not in ["codebook", "non-codebook"]:
            raise ValueError("precoding must be 'codebook' or 'non-codebook'")
        self._precoding = value

    @property
    def transform_precoding(self) -> bool:
        """Use transform precoding. Defaults to `False`.

        Only `False` is currently supported.
        """
        self._ifndef("transform_precoding", False)
        return self._transform_precoding

    @transform_precoding.setter
    def transform_precoding(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise TypeError("transform_precoding must be bool")
        if value:
            raise NotImplementedError(_TRANSFORM_PRECODING_ERROR)
        self._transform_precoding = value

    @property
    def tpmi(self) -> int:
        """Transmit precoding matrix indicator.

        Defaults to 0. Must be in [0, ..., 27].
        The allowed value depends on the number of layers and
        the number of antenna ports according to Tables 6.3.1.5-1
        to 6.3.1.5-7 :cite:p:`3GPPTS38211`.
        """
        self._ifndef("tpmi", 0)
        return self._tpmi

    @tpmi.setter
    def tpmi(self, value: int) -> None:
        if value not in range(28):
            raise ValueError("tpmi must be in [0, 27]")
        self._tpmi = value

    #-----------------------------#
    #---Read-only parameters------#
    #-----------------------------#

    @property
    def frequency_hopping(self) -> str:
        """Frequency hopping configuration. Read-only, defaults to ``"neither"``."""
        return "neither"

    @property
    def l_0(self) -> int:
        r"""Position of the first DMRS symbol :math:`l_0` relative to the reference ``l_ref``. Read-only."""
        if self.mapping_type == "A":
            return self.dmrs.type_a_position
        return 0

    @property
    def l_d(self) -> int:
        r"""Length of the symbol allocation :math:`l_\text{d}`. Read-only."""
        return self.symbol_allocation[1]

    @property
    def l_ref(self) -> int:
        """Reference OFDM symbol index used for DMRS generation. Read-only."""
        if self.mapping_type == "A":
            return 0
        return self.symbol_allocation[0]

    @property
    def l_prime(self) -> List[int]:
        r"""List of possible values of :math:`l'` used for DMRS generation. Read-only."""
        if self.dmrs.length == 1:
            return [0]
        return [0, 1]

    @property
    def l_bar(self) -> List[int]:
        r"""List of possible values of :math:`\bar{l}` used for DMRS generation. Read-only.

        Defined in Tables 6.4.1.1.3-3 and 6.4.1.1.3-4 :cite:p:`3GPPTS38211`.
        """
        l_0 = self.l_0
        ind = 0 if self.l_d < 4 else self.l_d - 3

        if self.mapping_type == "A":
            if self.dmrs.length == 1:
                l_bar = [
                    [[], [], [], []],
                    [[l_0], [l_0], [l_0], [l_0]],
                    [[l_0], [l_0], [l_0], [l_0]],
                    [[l_0], [l_0], [l_0], [l_0]],
                    [[l_0], [l_0], [l_0], [l_0]],
                    [[l_0], [l_0, 7], [l_0, 7], [l_0, 7]],
                    [[l_0], [l_0, 7], [l_0, 7], [l_0, 7]],
                    [[l_0], [l_0, 9], [l_0, 6, 9], [l_0, 6, 9]],
                    [[l_0], [l_0, 9], [l_0, 6, 9], [l_0, 6, 9]],
                    [[l_0], [l_0, 9], [l_0, 6, 9], [l_0, 5, 8, 11]],
                    [[l_0], [l_0, 11], [l_0, 7, 11], [l_0, 5, 8, 11]],
                    [[l_0], [l_0, 11], [l_0, 7, 11], [l_0, 5, 8, 11]],
                ]
            else:
                l_bar = [
                    [[], []],
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
        else:  # mapping_type == "B"
            if self.dmrs.length == 1:
                l_bar = [
                    [[l_0], [l_0], [l_0], [l_0]],
                    [[l_0], [l_0], [l_0], [l_0]],
                    [[l_0], [l_0, 4], [l_0, 4], [l_0, 4]],
                    [[l_0], [l_0, 4], [l_0, 4], [l_0, 4]],
                    [[l_0], [l_0, 4], [l_0, 4], [l_0, 4]],
                    [[l_0], [l_0, 6], [l_0, 3, 6], [l_0, 3, 6]],
                    [[l_0], [l_0, 6], [l_0, 3, 6], [l_0, 3, 6]],
                    [[l_0], [l_0, 8], [l_0, 4, 8], [l_0, 3, 6, 9]],
                    [[l_0], [l_0, 8], [l_0, 4, 8], [l_0, 3, 6, 9]],
                    [[l_0], [l_0, 10], [l_0, 5, 10], [l_0, 3, 6, 9]],
                    [[l_0], [l_0, 10], [l_0, 5, 10], [l_0, 3, 6, 9]],
                    [[l_0], [l_0, 10], [l_0, 5, 10], [l_0, 3, 6, 9]],
                ]
            else:
                l_bar = [
                    [[], []],
                    [[], []],
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
    def l(self) -> List[int]:
        r"""List of possible values of the OFDM symbol indices :math:`l` carrying DMRS relative to :math:`l_0`. Read-only."""
        result = []
        for l_bar in self.l_bar:
            for l_prime in self.l_prime:
                result.append(l_bar + l_prime)
        return result

    @property
    def n(self) -> List[int]:
        """List of possible values of n used for DMRS generation. Read-only."""
        if self.dmrs.config_type == 1:
            n_max = self.num_resource_blocks * 12 // 4 - 1
        else:  # config_type == 2
            n_max = self.num_resource_blocks * 12 // 6 - 1
        return list(range(n_max + 1))

    @property
    def dmrs_symbol_indices(self) -> List[int]:
        """Indices of DMRS symbols within a slot. Read-only."""
        return [sym + self.l_ref for sym in self.l]

    @property
    def num_resource_blocks(self) -> int:
        """Number of allocated resource blocks for the PUSCH transmissions. Read-only."""
        if self.n_size_bwp is None:
            return self.carrier.n_size_grid
        return self.n_size_bwp

    @property
    def num_subcarriers(self) -> int:
        """Number of allocated subcarriers for the PUSCH transmissions. Read-only."""
        return 12 * self.num_resource_blocks

    @property
    def num_res_per_prb(self) -> int:
        """Number of resource elements per PRB available for data. Read-only."""
        num_dmrs = len(self.dmrs_symbol_indices)
        num_data = self.symbol_allocation[1] - num_dmrs

        if self.dmrs.config_type == 1:
            num_res_dmrs = 12 - 6 * self.dmrs.num_cdm_groups_without_data
        else:  # config_type == 2
            num_res_dmrs = 12 - 4 * self.dmrs.num_cdm_groups_without_data

        num_res_data = 12
        return num_data * num_res_data + num_dmrs * num_res_dmrs

    @property
    def dmrs_mask(self) -> np.ndarray:
        """Masked resource elements in the resource grid. Read-only.

        Shape: [num_subcarriers, num_symbols_per_slot]. `True` corresponds to
        resource elements on which no data is transmitted.
        """
        mask = np.zeros(
            [self.num_subcarriers, self.carrier.num_symbols_per_slot],
            dtype=bool)

        num_cdm_groups = self.dmrs.num_cdm_groups_without_data
        if self.dmrs.config_type == 1:
            cdm_ind = np.zeros([6, num_cdm_groups], np.int32)
            for i in range(num_cdm_groups):
                cdm_ind[:, i] = np.arange(i, 12, 2)
        else:
            cdm_ind = np.zeros([4, num_cdm_groups], np.int32)
            for i in range(num_cdm_groups):
                cdm_ind[:, i] = np.array([0, 1, 6, 7]) + 2 * i

        for i in self.dmrs_symbol_indices:
            for j in range(self.num_resource_blocks):
                for k in range(num_cdm_groups):
                    mask[cdm_ind[:, k] + 12 * j, i] = True
        return mask

    @property
    def dmrs_grid(self) -> np.ndarray:
        """Empty resource grid for each DMRS port, filled with DMRS signals. Read-only.

        Shape: [num_dmrs_ports, num_subcarriers, num_symbols_per_slot].

        This property returns for each configured DMRS port an empty
        resource grid filled with DMRS signals as defined in
        Section 6.4.1.1 :cite:p:`3GPPTS38211`. Not all possible options are
        implemented, e.g., frequency hopping and transform precoding are
        not available.

        This property provides the *unprecoded* DMRS for each configured
        DMRS port. Precoding might be applied to map the DMRS to the
        antenna ports. However, in this case, the number of DMRS ports
        cannot be larger than the number of layers.
        """
        self.check_config()

        reset_dmrs_port_set = False
        if len(self.dmrs.dmrs_port_set) == 0:
            self.dmrs.dmrs_port_set = list(range(self.num_layers))
            reset_dmrs_port_set = True

        a_tilde = np.zeros(
            [len(self.dmrs.dmrs_port_set),
             self.num_subcarriers,
             self.carrier.num_symbols_per_slot],
            dtype=complex)

        for l_bar in self.l_bar:
            for l_prime in self.l_prime:
                sym = self.l_ref + l_bar + l_prime
                c_init = self.c_init(sym)
                c = generate_prng_seq(2 * self.num_subcarriers, c_init=c_init)
                r = 1 / np.sqrt(2) * ((1 - 2 * c[::2]) + 1j * (1 - 2 * c[1::2]))

                for j_ind, _ in enumerate(self.dmrs.dmrs_port_set):
                    for n in self.n:
                        for k_prime in [0, 1]:
                            if self.dmrs.config_type == 1:
                                k = 4 * n + 2 * k_prime + self.dmrs.deltas[j_ind]
                            else:  # config_type == 2
                                k = 6 * n + k_prime + self.dmrs.deltas[j_ind]

                            a_tilde[j_ind, k, sym] = (
                                r[2 * n + k_prime] *
                                self.dmrs.w_f[k_prime][j_ind] *
                                self.dmrs.w_t[l_prime][j_ind])

        a = self.dmrs.beta * a_tilde

        if reset_dmrs_port_set:
            self.dmrs.dmrs_port_set = []

        return a

    @property
    def dmrs_grid_precoded(self) -> Optional[np.ndarray]:
        """Precoded DMRS grid. Read-only.

        Returns `None` if :attr:`precoding` is ``"non-codebook"``.
        """
        if self.precoding == "non-codebook":
            return None

        w = np.expand_dims(np.expand_dims(self.precoding_matrix, 0), 0)
        a = np.expand_dims(np.transpose(self.dmrs_grid, [1, 2, 0]), -1)
        a = np.squeeze(np.matmul(w, a), -1)
        a = np.transpose(a, [2, 0, 1])
        return a

    @property
    def precoding_matrix(self) -> Optional[np.ndarray]:
        r"""Precoding matrix :math:`\mathbf{W}`. Read-only.

        Shape: [num_antenna_ports, num_layers]. Defined in
        Tables 6.3.1.5-1 to 6.3.1.5-7 :cite:p:`3GPPTS38211`.
        Only relevant if :attr:`~sionna.phy.nr.PUSCHConfig.precoding`
        is ``"codebook"``.
        """
        if self.precoding == "non-codebook":
            return None
        if self.num_antenna_ports == 1:
            return None

        w = None

        if self.num_layers == 1:
            if self.num_antenna_ports == 2:
                w = np.zeros([6, 2, 1], complex)
                w[:, 0, 0] = [1, 0, 1, 1, 1, 1]
                w[:, 1, 0] = [0, 1, 1, -1, 1j, -1j]
                w /= np.sqrt(2)
            elif self.num_antenna_ports == 4:
                w = np.zeros([28, 4, 1], complex)
                w[:8, 0, 0] = [1, 0, 0, 0, 1, 1, 1, 1]
                w[:8, 1, 0] = [0, 1, 0, 0, 0, 0, 0, 0]
                w[:8, 2, 0] = [0, 0, 1, 0, 1, -1, 1j, -1j]
                w[:8, 3, 0] = [0, 0, 0, 1, 0, 0, 0, 0]
                w[8:16, 0, 0] = [0, 0, 0, 0, 1, 1, 1, 1]
                w[8:16, 1, 0] = [1, 1, 1, 1, 1, 1, 1, 1]
                w[8:16, 2, 0] = [0, 0, 0, 0, 1, 1j, -1, -1j]
                w[8:16, 3, 0] = [1, -1, 1j, -1j, 1, 1j, -1, -1j]
                w[16:24, 0, 0] = [1, 1, 1, 1, 1, 1, 1, 1]
                w[16:24, 1, 0] = [1j, 1j, 1j, 1j, -1, -1, -1, -1]
                w[16:24, 2, 0] = [1, 1j, -1, -1j, 1, 1j, -1, -1j]
                w[16:24, 3, 0] = [1j, -1, -1j, 1, -1, -1j, 1, 1j]
                w[24:28, 0, 0] = [1, 1, 1, 1]
                w[24:28, 1, 0] = [-1j, -1j, -1j, -1j]
                w[24:28, 2, 0] = [1, 1j, -1, -1j]
                w[24:28, 3, 0] = [-1j, 1, 1j, -1]
                w /= 2

        elif self.num_layers == 2:
            if self.num_antenna_ports == 2:
                w = np.zeros([3, 2, 2], complex)
                w[0] = [[1, 0], [0, 1]]
                w[0] /= np.sqrt(2)
                w[1] = [[1, 1], [1, -1]]
                w[1] /= 2
                w[2] = [[1, 1], [1j, -1j]]
                w[2] /= 2
            elif self.num_antenna_ports == 4:
                w = np.zeros([22, 4, 2], complex)
                w[0] = [[1, 0], [0, 1], [0, 0], [0, 0]]
                w[0] /= 2
                w[1] = [[1, 0], [0, 0], [0, 1], [0, 0]]
                w[1] /= 2
                w[2] = [[1, 0], [0, 0], [0, 0], [0, 1]]
                w[2] /= 2
                w[3] = [[0, 0], [1, 0], [0, 1], [0, 0]]
                w[3] /= 2
                w[4] = [[0, 0], [1, 0], [0, 0], [0, 1]]
                w[4] /= 2
                w[5] = [[0, 0], [0, 0], [1, 0], [0, 1]]
                w[5] /= 2
                w[6] = [[1, 0], [0, 1], [1, 0], [0, -1j]]
                w[6] /= 2
                w[7] = [[1, 0], [0, 1], [1, 0], [0, 1j]]
                w[7] /= 2
                w[8] = [[1, 0], [0, 1], [-1j, 0], [0, 1]]
                w[8] /= 2
                w[9] = [[1, 0], [0, 1], [-1j, 0], [0, -1]]
                w[9] /= 2
                w[10] = [[1, 0], [0, 1], [-1, 0], [0, -1j]]
                w[10] /= 2
                w[11] = [[1, 0], [0, 1], [-1, 0], [0, 1j]]
                w[11] /= 2
                w[12] = [[1, 0], [0, 1], [1j, 0], [0, 1]]
                w[12] /= 2
                w[13] = [[1, 0], [0, 1], [1j, 0], [0, -1]]
                w[13] /= 2
                w[14] = [[1, 1], [1, 1], [1, -1], [1, -1]]
                w[14] /= 2 * np.sqrt(2)
                w[15] = [[1, 1], [1, 1], [1j, -1j], [1j, -1j]]
                w[15] /= 2 * np.sqrt(2)
                w[16] = [[1, 1], [1j, 1j], [1, -1], [1j, -1j]]
                w[16] /= 2 * np.sqrt(2)
                w[17] = [[1, 1], [1j, 1j], [1j, -1j], [-1, 1]]
                w[17] /= 2 * np.sqrt(2)
                w[18] = [[1, 1], [-1, -1], [1, -1], [-1, 1]]
                w[18] /= 2 * np.sqrt(2)
                w[19] = [[1, 1], [-1, -1], [1j, -1j], [-1j, 1j]]
                w[19] /= 2 * np.sqrt(2)
                w[20] = [[1, 1], [-1j, -1j], [1, -1], [-1j, 1j]]
                w[20] /= 2 * np.sqrt(2)
                w[21] = [[1, 1], [-1j, -1j], [1j, -1j], [1, -1]]
                w[21] /= 2 * np.sqrt(2)

        elif self.num_layers == 3:
            if self.num_antenna_ports == 4:
                w = np.zeros([7, 4, 3], complex)
                w[0] = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]]
                w[0] /= 2
                w[1] = [[1, 0, 0], [0, 1, 0], [1, 0, 0], [0, 0, 1]]
                w[1] /= 2
                w[2] = [[1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, 0, 1]]
                w[2] /= 2
                w[3] = [[1, 1, 1], [1, -1, 1], [1, 1, -1], [1, -1, -1]]
                w[3] /= (2 * np.sqrt(3))
                w[4] = [[1, 1, 1], [1, -1, 1], [1j, 1j, -1j], [1j, -1j, -1j]]
                w[4] /= (2 * np.sqrt(3))
                w[5] = [[1, 1, 1], [-1, 1, -1], [1, 1, -1], [-1, 1, 1]]
                w[5] /= (2 * np.sqrt(3))
                w[6] = [[1, 1, 1], [-1, 1, -1], [1j, 1j, -1j], [-1j, 1j, 1j]]
                w[6] /= (2 * np.sqrt(3))

        elif self.num_layers == 4:
            if self.num_antenna_ports == 4:
                w = np.zeros([5, 4, 4], complex)
                w[0] = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
                w[0] /= 2
                w[1] = [[1, 1, 0, 0], [0, 0, 1, 1], [1, -1, 0, 0], [0, 0, 1, -1]]
                w[1] /= 2 * np.sqrt(2)
                w[2] = [[1, 1, 0, 0], [0, 0, 1, 1], [1j, -1j, 0, 0], [0, 0, 1j, -1j]]
                w[2] /= 2 * np.sqrt(2)
                w[3] = [[1, 1, 1, 1], [1, -1, 1, -1], [1, 1, -1, -1], [1, -1, -1, 1]]
                w[3] /= 4
                w[4] = [[1, 1, 1, 1], [1, -1, 1, -1], [1j, 1j, -1j, -1j], [1j, -1j, -1j, 1j]]
                w[4] /= 4

        if w is None:
            return w
        return w[self.tpmi]

    @property
    def num_ov(self) -> int:
        """Number of unused resource elements due to additional overhead as specified by higher layer. Read-only.

        Defaults to 0.
        """
        return 0

    @property
    def num_coded_bits(self) -> int:
        """Number of coded bits that fit into one PUSCH slot. Read-only."""
        n_re_per_prb = self.num_res_per_prb - self.num_ov
        n_re = n_re_per_prb * self.num_resource_blocks
        num_coded_bits = int(
            self.tb.tb_scaling * self.tb.num_bits_per_symbol *
            self.num_layers * n_re)
        return num_coded_bits

    @property
    def tb_size(self) -> int:
        """Transport block size. Read-only.

        Number of information bits that can be encoded into a slot for the
        given slot configuration.
        """
        n_re_per_prb = self.num_res_per_prb - self.num_ov
        n_re = min(156, n_re_per_prb) * self.num_resource_blocks
        target_tb_size = int(
            self.tb.target_coderate * self.tb.tb_scaling *
            n_re * self.tb.num_bits_per_symbol * self.num_layers)

        tb_size, *_ = calculate_tb_size(
            target_tb_size=target_tb_size,
            num_coded_bits=self.num_coded_bits,
            target_coderate=self.tb.target_coderate,
            modulation_order=self.tb.num_bits_per_symbol,
            verbose=False)

        return int(tb_size)

    #-------------------#
    #---Class methods---#
    #-------------------#

    def c_init(self, l: int) -> int:
        r"""Compute RNG initialization :math:`c_\text{init}` as in Section 6.4.1.1.1.1 :cite:p:`3GPPTS38211`.

        :param l: OFDM symbol index within the slot.
        """
        num_symbols_per_slot = self.carrier.num_symbols_per_slot
        slot_number = self.carrier.slot_number

        lambda_bar = 0
        n_scid_bar = self.dmrs.n_scid
        if self.dmrs.n_id is None:
            n_id = self.carrier.n_cell_id
        else:
            n_id = self.dmrs.n_id[n_scid_bar]

        c_init = np.mod(
            2**17 * (num_symbols_per_slot * slot_number + l + 1) *
            (2 * n_id + 1) +
            2**17 * np.floor(lambda_bar / 2) +
            2 * n_id + n_scid_bar,
            2**31)

        return int(c_init)

    def show(self) -> None:
        """Print all properties of the PUSCHConfig and children configs."""
        self.carrier.show()
        Config.show(self)
        self.dmrs.show()
        self.tb.show()

    def check_config(self) -> bool:
        """Test if the compound configuration is valid."""
        self.carrier.check_config()
        self.dmrs.check_config()

        if self.transform_precoding:
            raise NotImplementedError(_TRANSFORM_PRECODING_ERROR)

        if self.precoding == "codebook":
            if len(self.dmrs.dmrs_port_set) > 0:
                if len(self.dmrs.dmrs_port_set) != self.num_layers:
                    raise ValueError(
                        "num_layers must equal the number of DMRS ports")
            if self.num_layers > self.num_antenna_ports:
                raise ValueError("num_layers must be <= num_antenna_ports")
            if self.num_antenna_ports < 2:
                raise ValueError(
                    "precoding requires two or more antenna ports")
        else:
            if self.num_layers != self.num_antenna_ports:
                raise ValueError("num_layers must equal num_antenna_ports")

        # Check Tables 6.4.1.1.3-3/4 validity
        if self.dmrs.length == 1:
            if self.mapping_type == "A" and self.symbol_allocation[1] < 4:
                raise ValueError("Symbol allocation is too short")
        else:
            if self.dmrs.additional_position >= 2:
                raise ValueError(
                    "dmrs.additional_position must be < 2 for this dmrs.length")
            if (
                self.mapping_type == "A"
                and self.l_d == 4
                and self.dmrs.type_a_position != 2
            ):
                raise ValueError(
                    "dmrs.type_a_position must be 2 for mapping type A, "
                    "dmrs.length=2, and symbol_allocation length 4")
            if self.symbol_allocation[1] < 4:
                raise ValueError("Symbol allocation too short")
            if self.mapping_type == "B" and self.symbol_allocation[1] < 5:
                raise ValueError("Symbol allocation is too short")

        # Check type_a and additional_position
        if self.mapping_type == "A" and self.dmrs.additional_position == 3:
            if self.dmrs.type_a_position != 2:
                raise ValueError(
                    "additional_position=3 only allowed for type_a_position=2")

        # Check TPMI validity
        if self.num_layers == 1:
            if self.num_antenna_ports == 2 and self.tpmi not in range(6):
                raise ValueError("tpmi must be in [0, 5]")
            elif self.num_antenna_ports == 4 and self.tpmi not in range(28):
                raise ValueError("tpmi must be in [0, 27]")
        elif self.num_layers == 2:
            if self.num_antenna_ports == 2 and self.tpmi not in range(3):
                raise ValueError("tpmi must be in [0, 2]")
            elif self.num_antenna_ports == 4 and self.tpmi not in range(22):
                raise ValueError("tpmi must be in [0, 21]")
        elif self.num_layers == 3:
            if self.tpmi not in range(7):
                raise ValueError("tpmi must be in [0, 6]")
        elif self.num_layers == 4:
            if self.tpmi not in range(5):
                raise ValueError("tpmi must be in [0, 4]")

        # Check symbol allocation
        max_length = 14 if self.carrier.cyclic_prefix == "normal" else 12

        if self.mapping_type == "A":
            if self.symbol_allocation[0] != 0:
                raise ValueError(
                    "symbol_allocation[0] must be 0 for mapping_type A")
            if not 4 <= self.symbol_allocation[1] <= max_length:
                raise ValueError(
                    f"symbol_allocation[1] must be in [4, {max_length}]")
            if self.dmrs.length == 2 and self.symbol_allocation[1] < 4:
                raise ValueError(
                    "symbol_allocation[1] must be >= 4 for dmrs.length == 2")
        elif self.mapping_type == "B":
            if not 0 <= self.symbol_allocation[0] <= 13:
                raise ValueError(
                    "symbol_allocation[0] must be in [0, 13] for mapping_type B")
            if not 1 <= self.symbol_allocation[1] <= max_length:
                raise ValueError(
                    f"symbol_allocation[1] must be in [1, {max_length}]")
            if self.dmrs.length == 2 and self.symbol_allocation[1] < 5:
                raise ValueError(
                    "symbol_allocation[1] must be >= 5 for dmrs.length == 2")

        if self.symbol_allocation[0] + self.symbol_allocation[1] > max_length:
            raise ValueError(
                f"symbol_allocation[0] + [1] must be <= {max_length}")

        # Validate all configurable attributes
        attr_list = [
            "n_size_bwp", "n_start_bwp", "num_layers", "mapping_type",
            "symbol_allocation", "n_rnti", "precoding", "transform_precoding",
            "tpmi"
        ]
        for attr in attr_list:
            value = getattr(self, attr)
            setattr(self, attr, value)

        if self.tb.channel_type != "PUSCH":
            raise ValueError('TB_config must be configured for "PUSCH"')

        if (len(self.dmrs.dmrs_port_set) > 0 and
                self.num_layers != len(self.dmrs.dmrs_port_set)):
            raise ValueError("num_layers must equal the number of DMRS ports")

        return True


def check_pusch_configs(pusch_configs: List[PUSCHConfig]) -> dict:
    """Validate a list of PUSCHConfig instances and extract common parameters.

    The returned ``cyclic_prefix_length`` is a single sample count derived from
    :attr:`~sionna.phy.nr.CarrierConfig.cyclic_prefix_length`. As documented
    there, Sionna uses one scalar CP duration for the whole slot rather than
    the per-symbol CP lengths of TS 38.211.

    :param pusch_configs: List of :class:`~sionna.phy.nr.PUSCHConfig` instances.

        All configurations must use the same carrier geometry, resource
        allocation, coding, layer, precoding, and DMRS structure.
        Transmitter-specific cell and scrambling identities, DMRS ports and
        sequences, and precoding-matrix values may differ.
    """
    check_instance(
        pusch_configs,
        list,
        name="pusch_configs",
        message="pusch_configs must be a list of PUSCHConfig instances",
    )
    if not pusch_configs:
        raise ValueError("pusch_configs must not be empty")

    check_sequence_of(
        pusch_configs,
        PUSCHConfig,
        name="pusch_configs",
        sequence_type=list,
        message="All elements must be instances of PUSCHConfig",
    )
    for pusch_config in pusch_configs:
        pusch_config.check_config()

    pc = pusch_configs[0]
    carrier = pc.carrier

    def scalar(value):
        """Convert scalar NumPy and tensor values to Python scalars."""
        if hasattr(value, "numel"):
            if value.numel() != 1:
                raise ValueError("Expected a scalar tensor value")
            return value.item()
        if isinstance(value, np.generic):
            return value.item()
        return value

    common_parameters = [
        ("num_layers", lambda c: c.num_layers),
        ("num_antenna_ports", lambda c: c.num_antenna_ports),
        ("precoding", lambda c: c.precoding),
        ("transform_precoding", lambda c: c.transform_precoding),
        ("mapping_type", lambda c: c.mapping_type),
        ("symbol_allocation", lambda c: tuple(c.symbol_allocation)),
        ("n_size_bwp", lambda c: c.n_size_bwp),
        ("n_start_bwp", lambda c: c.n_start_bwp),
        ("carrier.cyclic_prefix", lambda c: c.carrier.cyclic_prefix),
        ("carrier.subcarrier_spacing", lambda c: c.carrier.subcarrier_spacing),
        ("carrier.n_size_grid", lambda c: c.carrier.n_size_grid),
        ("carrier.n_start_grid", lambda c: c.carrier.n_start_grid),
        ("carrier.slot_number", lambda c: c.carrier.slot_number),
        ("carrier.frame_number", lambda c: c.carrier.frame_number),
        ("tb.mcs_table", lambda c: c.tb.mcs_table),
        ("tb.mcs_index", lambda c: c.tb.mcs_index),
        ("dmrs.config_type", lambda c: c.dmrs.config_type),
    ]
    if pc.mapping_type == "A":
        common_parameters.append(
            ("dmrs.type_a_position", lambda c: c.dmrs.type_a_position)
        )
    common_parameters += [
        ("dmrs.length", lambda c: c.dmrs.length),
        ("dmrs.additional_position", lambda c: c.dmrs.additional_position),
        (
            "dmrs.num_cdm_groups_without_data",
            lambda c: c.dmrs.num_cdm_groups_without_data,
        ),
        ("num_resource_blocks", lambda c: c.num_resource_blocks),
        ("num_subcarriers", lambda c: c.num_subcarriers),
        ("num_ofdm_symbols", lambda c: c.l_d),
        ("tb.num_bits_per_symbol", lambda c: scalar(c.tb.num_bits_per_symbol)),
        ("tb.target_coderate", lambda c: scalar(c.tb.target_coderate)),
        ("num_coded_bits", lambda c: c.num_coded_bits),
        ("tb_size", lambda c: c.tb_size),
        ("dmrs_mask", lambda c: c.dmrs_mask),
    ]
    expected_parameters = [
        (name, accessor, accessor(pc))
        for name, accessor in common_parameters
    ]

    for index, pusch_config in enumerate(pusch_configs[1:], start=1):
        for name, accessor, expected in expected_parameters:
            actual = accessor(pusch_config)
            if isinstance(expected, np.ndarray):
                matches = np.array_equal(actual, expected)
            else:
                matches = actual == expected
            if not matches:
                raise ValueError(
                    f"pusch_configs[{index}].{name} must match "
                    f"pusch_configs[0] ({actual!r} != {expected!r})")

    params = {
        "num_bits_per_symbol": scalar(pc.tb.num_bits_per_symbol),
        "num_tx": len(pusch_configs),
        "num_layers": pc.num_layers,
        "num_subcarriers": pc.num_subcarriers,
        "num_ofdm_symbols": pc.symbol_allocation[1],
        "subcarrier_spacing": pc.carrier.subcarrier_spacing * 1e3,
        "num_antenna_ports": pc.num_antenna_ports,
        "precoding": pc.precoding,
        "precoding_matrices": [],
        "pusch_config": pc,
        "carrier_config": pc.carrier,
        "num_coded_bits": pc.num_coded_bits,
        "target_coderate": scalar(pc.tb.target_coderate),
        "n_id": [],
        "n_rnti": [],
        "tb_size": pc.tb_size,
        "dmrs_length": pc.dmrs.length,
        "dmrs_additional_position": pc.dmrs.additional_position,
        "num_cdm_groups_without_data": pc.dmrs.num_cdm_groups_without_data,
    }
    params["bandwidth"] = params["num_subcarriers"] * params["subcarrier_spacing"]
    params["cyclic_prefix_length"] = np.ceil(
        carrier.cyclic_prefix_length * params["bandwidth"])

    for pusch_config in pusch_configs:
        if params["precoding"] == "codebook":
            params["precoding_matrices"].append(pusch_config.precoding_matrix)

        if pusch_config.tb.n_id is None:
            params["n_id"].append(pusch_config.carrier.n_cell_id)
        else:
            params["n_id"].append(pusch_config.tb.n_id)
        params["n_rnti"].append(pusch_config.n_rnti)

    return params

