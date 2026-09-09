#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Carrier configuration for the NR (5G) module of Sionna PHY."""

from sionna._validation import check_one_of, check_scalar_range

from .config import Config


__all__ = ["CarrierConfig"]


class CarrierConfig(Config):
    """Sets parameters for a specific OFDM numerology, as described in
    Section 4 :cite:p:`3GPPTS38211`.

    All configurable properties can be provided as keyword arguments during
    initialization or changed later.

    .. rubric:: Examples

    >>> from sionna.phy.nr import CarrierConfig
    >>> carrier_config = CarrierConfig(n_cell_id=41)
    >>> carrier_config.subcarrier_spacing = 30
    """

    def __init__(self, **kwargs):
        self._name = "Carrier Configuration"
        super().__init__(**kwargs)
        self.check_config()

    # -----------------------------
    # Configurable parameters
    # -----------------------------

    @property
    def n_cell_id(self) -> int:
        r"""`int`, (default 1) | [0,...,1007] : Physical layer cell identity
        :math:`N_\text{ID}^\text{cell}`."""
        self._ifndef("n_cell_id", 1)
        return self._n_cell_id

    @n_cell_id.setter
    def n_cell_id(self, value: int) -> None:
        check_one_of(
            value,
            range(1008),
            name="n_cell_id",
            message="n_cell_id must be in the range from 0 to 1007",
        )
        self._n_cell_id = value

    @property
    def cyclic_prefix(self) -> str:
        """'normal' (default) | 'extended' : Cyclic prefix length.

        The option 'normal' corresponds to 14 OFDM symbols per slot, while
        'extended' corresponds to 12 OFDM symbols. The latter option is
        only possible with a `subcarrier_spacing` of 60 kHz.
        """
        self._ifndef("cyclic_prefix", "normal")
        return self._cyclic_prefix

    @cyclic_prefix.setter
    def cyclic_prefix(self, value: str) -> None:
        check_one_of(
            value,
            ("normal", "extended"),
            name="cyclic_prefix",
            message="Invalid cyclic prefix",
        )
        self._cyclic_prefix = value

    @property
    def subcarrier_spacing(self) -> float:
        r"""`float`, (default 15) | 30 | 60 | 120 | 240 | 480 | 960 :
        Subcarrier spacing :math:`\Delta f` [kHz]."""
        self._ifndef("subcarrier_spacing", 15)
        return self._subcarrier_spacing

    @subcarrier_spacing.setter
    def subcarrier_spacing(self, value: float) -> None:
        check_one_of(
            value,
            (15, 30, 60, 120, 240, 480, 960),
            name="subcarrier_spacing",
            message="Invalid subcarrier spacing",
        )
        self._subcarrier_spacing = value

    @property
    def n_size_grid(self) -> int:
        r"""`int`, (default 4) | [1,...,275] : Number of resource blocks
        in the carrier resource grid
        :math:`N^{\text{size},\mu}_{\text{grid},x}`."""
        self._ifndef("n_size_grid", 4)
        return self._n_size_grid

    @n_size_grid.setter
    def n_size_grid(self, value: int) -> None:
        check_one_of(
            value,
            range(1, 276),
            name="n_size_grid",
            message="n_size_grid must be in the range from 1 to 275",
        )
        self._n_size_grid = value

    @property
    def n_start_grid(self) -> int:
        r"""`int`, (default 0) | [0,...,2199] : Start of resource grid
        relative to common resource block (CRB) 0
        :math:`N^{\text{start},\mu}_{\text{grid},x}`."""
        self._ifndef("n_start_grid", 0)
        return self._n_start_grid

    @n_start_grid.setter
    def n_start_grid(self, value: int) -> None:
        check_one_of(
            value,
            range(2200),
            name="n_start_grid",
            message="n_start_grid must be in the range from 0 to 2199",
        )
        self._n_start_grid = value

    @property
    def slot_number(self) -> int:
        r"""`int`, (default 0), [0,...,num_slots_per_frame] : Slot number
        within a frame :math:`n^\mu_{s,f}`."""
        self._ifndef("slot_number", 0)
        return self._slot_number

    @slot_number.setter
    def slot_number(self, value: int) -> None:
        check_scalar_range(
            value,
            name="slot_number",
            minimum=0,
            maximum=self.num_slots_per_frame,
            upper_inclusive=False,
            message=(
                "slot_number cannot exceed the number of slots per frame-1"
            ),
        )
        self._slot_number = value

    @property
    def frame_number(self) -> int:
        r"""`int`, (default 0), [0,...,1023] : System frame number
        :math:`n_\text{f}`."""
        self._ifndef("frame_number", 0)
        return self._frame_number

    @frame_number.setter
    def frame_number(self, value: int) -> None:
        check_one_of(
            value,
            range(1024),
            name="frame_number",
            message="frame_number must be in [0, 1023]",
        )
        self._frame_number = value

    # --------------------------
    # Read-only parameters
    # --------------------------

    @property
    def num_symbols_per_slot(self) -> int:
        r"""`int`, (default 14) | 12 : Number of OFDM symbols per slot
        :math:`N_\text{symb}^\text{slot}`.

        Configured through the `cyclic_prefix`.
        """
        if self.cyclic_prefix == "normal":
            return 14
        else:
            return 12

    @property
    def num_slots_per_subframe(self) -> int:
        r"""`int`, (default 1) | 2 | 4 | 8 | 16 | 32 | 64 : Number of
        slots per subframe :math:`N_\text{slot}^{\text{subframe},\mu}`.

        Depends on the `subcarrier_spacing`.
        """
        spacing_map = {15: 1, 30: 2, 60: 4, 120: 8, 240: 16, 480: 32, 960: 64}
        return spacing_map[self.subcarrier_spacing]

    @property
    def num_slots_per_frame(self) -> int:
        r"""`int`, (default 10) | 20 | 40 | 80 | 160 | 320 | 640 : Number
        of slots per frame :math:`N_\text{slot}^{\text{frame},\mu}`.

        Depends on the `subcarrier_spacing`.
        """
        return 10 * self.num_slots_per_subframe

    @property
    def mu(self) -> int:
        r"""`int`, (default 0) | 1 | 2 | 3 | 4 | 5 | 6 : Subcarrier
        spacing configuration, :math:`\Delta f = 2^\mu 15` kHz."""
        return [15, 30, 60, 120, 240, 480, 960].index(self.subcarrier_spacing)

    @property
    def frame_duration(self) -> float:
        r"""`float`, (default 10e-3) : Duration of a frame
        :math:`T_\text{f}` [s]."""
        return 10e-3

    @property
    def sub_frame_duration(self) -> float:
        r"""`float`, (default 1e-3), read-only : Duration of a subframe
        :math:`T_\text{sf}` [s]."""
        return 1e-3

    @property
    def t_c(self) -> float:
        r"""`float`, (default 0.509e-9) : Sampling time :math:`T_\text{c}` [s]
        for subcarrier spacing 480kHz."""
        return 1 / (480e3 * 4096)

    @property
    def t_s(self) -> float:
        r"""`float`, (default 32.552e-9) : Sampling time :math:`T_\text{s}` [s]
        for subcarrier spacing 15kHz."""
        return 1 / (15e3 * 2048)

    @property
    def kappa(self) -> float:
        r"""`float`, (default 64) : The constant
        :math:`\kappa = T_\text{s}/T_\text{c}`."""
        return 64.0

    @property
    def cyclic_prefix_length(self) -> float:
        r"""`float` : Cyclic prefix length
        :math:`N_{\text{CP},l}^{\mu} \cdot T_{\text{c}}` [s].

        .. note::

            In NR, the longer normal cyclic prefix applies only to specific
            OFDM symbols within a slot (see TS 38.211). Sionna exposes a
            single scalar duration for the whole slot: for normal CP, the
            extra :math:`16\kappa` samples are included when
            ``slot_number`` is in :math:`\{0, 7\cdot 2^{\mu}\}`, and that
            one value is then used for every symbol. This is a deliberate
            waveform simplification, not a per-symbol CP model.
        """
        if self.cyclic_prefix == "extended":
            cp = 512 * self.kappa * 2 ** (-self.mu)
        else:
            cp = 144 * self.kappa * 2 ** (-self.mu)
            if self.slot_number in [0, 7 * 2**self.mu]:
                cp += 16 * self.kappa
        return cp * self.t_c

    # -------------------
    # Class methods
    # -------------------

    def check_config(self) -> None:
        """Test if the configuration is valid."""
        if self.cyclic_prefix == "extended":
            if self.subcarrier_spacing != 60:
                raise ValueError(
                    "Extended cyclic prefix only valid for 60kHz subcarrier spacing"
                )

        attr_list = [
            "n_cell_id",
            "cyclic_prefix",
            "subcarrier_spacing",
            "n_size_grid",
            "slot_number",
            "frame_number",
        ]
        for attr in attr_list:
            value = getattr(self, attr)
            setattr(self, attr, value)

