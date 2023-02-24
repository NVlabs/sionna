#
# SPDX-FileCopyrightText: Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Carrier configuration for the nr (5G) sub-package of the Sionna library.
"""
# pylint: disable=line-too-long

from .config import Config

class CarrierConfig(Config):
    """
    The CarrierConfig objects sets parameters for a specific OFDM numerology,
    as described in Section 4 [3GPP38211]_.

    All configurable properties can be provided as keyword arguments during the
    initialization or changed later.

    Example
    -------
    >>> carrier_config = CarrierConfig(n_cell_id=41)
    >>> carrier_config.subcarrier_spacing = 30
    """
    def __init__(self, **kwargs):
        self._name = "Carrier Configuration"
        super().__init__(**kwargs)
        self.check_config()

    #-----------------------------#
    #---Configurable parameters---#
    #-----------------------------#

    #---n_cell_id---#
    @property
    def n_cell_id(self):
        r"""
        int, 1 (default) | [0,...,1007] : Physical layer cell identity
            :math:`N_\text{ID}^\text{cell}`
        """
        self._ifndef("n_cell_id", 1)
        return self._n_cell_id

    @n_cell_id.setter
    def n_cell_id(self, value):
        assert value in range(1008), \
        "n_cell_id must be in the range from 0 to 1007"
        self._n_cell_id = value

    #---cyclic_prefix---#
    @property
    def cyclic_prefix(self):
        """
        str, "normal" (default) | "extended" : Cyclic prefix length

            The option "normal" corresponds to 14 OFDM symbols per slot, while
            "extended" corresponds to 12 OFDM symbols. The latter option is
            only possible with a `subcarrier_spacing` of 60 kHz.
        """
        self._ifndef("cyclic_prefix", "normal")
        return self._cyclic_prefix

    @cyclic_prefix.setter
    def cyclic_prefix(self, value):
        assert value in ["normal", "extended"], "Invalid cyclic prefix"
        self._cyclic_prefix = value

    #---subcarrier_spacing---#
    @property
    def subcarrier_spacing(self):
        r"""
        float, 15 (default) | 30 | 60 | 120 | 240 | 480 | 960 : Subcarrier
            spacing :math:`\Delta f` [kHz]
        """
        self._ifndef("subcarrier_spacing", 15)
        return self._subcarrier_spacing

    @subcarrier_spacing.setter
    def subcarrier_spacing(self, value):
        assert value in [15, 30, 60, 120, 240, 480, 960], \
            "Invalid subcarrier spacing"
        self._subcarrier_spacing = value

    #---n_size_grid---#
    @property
    def n_size_grid(self):
        r"""
        int, 4 (default) | [1,...,275] : Number of resource blocks in the
            carrier resource grid :math:`N^{\text{size},\mu}_{\text{grid},x}`
        """
        self._ifndef("n_size_grid", 4)
        return self._n_size_grid

    @n_size_grid.setter
    def n_size_grid(self, value):
        assert value in range(1,276), \
            "n_size_grid must be in the range from 1 to 275"
        self._n_size_grid = value

    #---n_start_grid---#
    @property
    def n_start_grid(self):
        r"""
        int, 0 (default) | [0,...,2199] : Start of resource grid relative to
            common resource block (CRB) 0
            :math:`N^{\text{start},\mu}_{\text{grid},x}`
        """
        self._ifndef("n_start_grid", 0)
        return self._n_start_grid

    @n_start_grid.setter
    def n_start_grid(self, value):
        assert value in range(0,2200), \
            "n_start_grid must be in the range from 0 to 2199"
        self._n_start_grid = value

    #---slot_number---#
    @property
    def slot_number(self):
        r"""
        int, 0 (default), [0,...,num_slots_per_frame] : Slot number within a frame
            :math:`n^\mu_{s,f}`
        """
        self._ifndef("slot_number", 0)
        return self._slot_number

    @slot_number.setter
    def slot_number(self, value):
        assert 0<=value<self.num_slots_per_frame, \
            "slot_number cannot exceed the number of slots per frame-1"
        self._slot_number = value

    #---frame_number---#
    @property
    def frame_number(self):
        r"""
        int, 0 (default), [0,...,1023] : System frame number :math:`n_\text{f}`
        """
        self._ifndef("frame_number", 0)
        return self._frame_number

    @frame_number.setter
    def frame_number(self, value):
        assert value in range(0,1024), "frame_number must be in [0, 1023]"
        self._frame_number = value

    #--------------------------#
    #---Read-only parameters---#
    #--------------------------#

    @property
    def num_symbols_per_slot(self):
        r"""
        int, 14 (default) | 12, read-only : Number of OFDM symbols per slot
            :math:`N_\text{symb}^\text{slot}`

            Configured through the `cyclic_prefix`.
        """
        if self.cyclic_prefix=="normal":
            return 14
        else:
            return 12

    @property
    def num_slots_per_subframe(self):
        r"""
        int, 1 (default) | 2 | 4 | 8 | 16 | 32 | 64, read-only : Number of
            slots per subframe :math:`N_\text{slot}^{\text{subframe},\mu}`

            Depends on the `subcarrier_spacing`.
        """
        if self.subcarrier_spacing==15:
            return 1
        elif self.subcarrier_spacing==30:
            return 2
        elif self.subcarrier_spacing==60:
            return 4
        elif self.subcarrier_spacing==120:
            return 8
        elif self.subcarrier_spacing==240:
            return 16
        elif self.subcarrier_spacing==480:
            return 32
        elif self.subcarrier_spacing==960:
            return 64

    @property
    def num_slots_per_frame(self):
        r"""
        int, 10 (default) | 20 | 40 | 80 | 160 | 320 | 640, read-only : Number
            of slots per frame :math:`N_\text{slot}^{\text{frame},\mu}`

            Depends on the `subcarrier_spacing`.
        """
        return 10*self.num_slots_per_subframe

    @property
    def mu(self):
        r"""
        int, 0 (default) | 1 | 2 | 3 | 4 | 5 | 6, read-only : Subcarrier
            spacing configuration, :math:`\Delta f = 2^\mu 15` kHz
        """
        return [15, 30, 60, 120, 240, 480, 960].index(self.subcarrier_spacing)

    @property
    def frame_duration(self):
        r"""
        float, 10e-3 (default), read-only : Duration of a frame
            :math:`T_\text{f}` [s]
        """
        return 10e-3

    @property
    def sub_frame_duration(self):
        r"""
        float, 1e-3 (default), read-only : Duration of a subframe
            :math:`T_\text{sf}` [s]
        """
        return 1e-3

    @property
    def t_c(self):
        r"""
        float, 0.509e-9 [s], read-only : Sampling time :math:`T_\text{c}` for
            subcarrier spacing 480kHz.
        """
        return 1/(480e3*4096)

    @property
    def t_s(self):
        r"""
        float, 32.552e-9 [s], read-only : Sampling time :math:`T_\text{s}` for
            subcarrier spacing 15kHz.
        """
        return 1/(15e3*2048)

    @property
    def kappa(self):
        r"""
        float, 64, read-only : The constant
            :math:`\kappa = T_\text{s}/T_\text{c}`
        """
        return 64.

    @property
    def cyclic_prefix_length(self):
        r"""
        float, read-only : Cyclic prefix length
            :math:`N_{\text{CP},l}^{\mu} \cdot T_{\text{c}}` [s]
        """
        if self.cyclic_prefix=="extended":
            cp =  512*self.kappa*2**(-self.mu)
        else:
            cp = 144*self.kappa*2**(-self.mu)
            if self.slot_number in [0, 7*2**self.mu]:
                cp += 16*self.kappa
        return cp*self.t_c

    #-------------------#
    #---Class methods---#
    #-------------------#

    def check_config(self):
        """Test if configuration is valid"""

        if self.cyclic_prefix=="extended":
            assert self.subcarrier_spacing==60, \
            "Extended cyclic prefix only valid for 60kHz subcarrier spacing"

        attr_list = ["n_cell_id",
                     "cyclic_prefix",
                     "subcarrier_spacing",
                     "n_size_grid",
                     "slot_number",
                     "frame_number"
                    ]
        for attr in attr_list:
            value = getattr(self, attr)
            setattr(self, attr, value)
