#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""3GPP TR 38.901 antenna modeling"""

from typing import Optional, Sequence, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle
import torch

from sionna.phy import SPEED_OF_LIGHT, PI
from sionna.phy.object import Object

_ANTENNA_PATTERNS = ("omni", "38.901", "38.901-handheld")


def _spherical_unit_vector(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    """Unit vector for TR 38.901 spherical coordinates."""
    return torch.stack(
        [
            torch.sin(theta) * torch.cos(phi),
            torch.sin(theta) * torch.sin(phi),
            torch.cos(theta),
        ],
        dim=-1,
    )


def _spherical_basis(
    theta: torch.Tensor,
    phi: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Theta and phi unit vectors for TR 38.901 spherical coordinates."""
    e_theta = torch.stack(
        [
            torch.cos(theta) * torch.cos(phi),
            torch.cos(theta) * torch.sin(phi),
            -torch.sin(theta),
        ],
        dim=-1,
    )
    e_phi = torch.stack(
        [
            -torch.sin(phi),
            torch.cos(phi),
            torch.zeros_like(phi),
        ],
        dim=-1,
    )
    return e_theta, e_phi


def _radiation_pattern_38901_table(
    theta: torch.Tensor,
    phi: torch.Tensor,
    theta_3db_deg: float,
    phi_3db_deg: float,
    sla_v_db: float,
    a_max_db: float,
    g_e_max_db: float,
) -> torch.Tensor:
    """Radiation power pattern from TR 38.901 Tables 7.3-1/7.3-2."""
    theta_3db = theta_3db_deg / 180 * PI
    phi_3db = phi_3db_deg / 180 * PI
    sla_v = torch.as_tensor(sla_v_db, dtype=theta.dtype, device=theta.device)
    a_max = torch.as_tensor(a_max_db, dtype=theta.dtype, device=theta.device)
    a_v = -torch.minimum(12 * ((theta - PI / 2) / theta_3db) ** 2, sla_v)
    a_h = -torch.minimum(12 * (phi / phi_3db) ** 2, a_max)
    a_db = -torch.minimum(-(a_v + a_h), a_max) + g_e_max_db
    return 10 ** (a_db / 10)


class AntennaElement(Object):
    """Antenna element following the :cite:p:`TR38901V1920` specification

    :param pattern: Radiation pattern. One of ``"omni"``, ``"38.901"``,
        or ``"38.901-handheld"``. The ``"38.901"`` pattern follows
        Table 7.3-1 of TR 38.901 :cite:p:`TR38901V1920`. The
        ``"38.901-handheld"`` pattern follows Table 7.3-2 of
        :cite:p:`TR38901V1920` for handheld
        UT antenna elements.
    :param slant_angle: Polarization slant angle [radian]
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`, :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for computation (e.g., ``"cpu"``, ``"cuda:0"``).
        If `None`, :attr:`~sionna.phy.config.Config.device` is used.

    .. rubric:: Examples

    .. code-block:: python

        from sionna.phy.channel.tr38901 import AntennaElement
        import torch

        # Create an antenna element with 38.901 radiation pattern
        ant = AntennaElement(pattern="38.901", slant_angle=0.0)

        # Compute field at zenith angle pi/2 and azimuth angle 0
        theta = torch.tensor([1.5708])
        phi = torch.tensor([0.0])
        f_theta, f_phi = ant.field(theta, phi)
    """

    def __init__(
        self,
        pattern: str,
        slant_angle: float = 0.0,
        precision: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        super().__init__(precision=precision, device=device)
        if pattern not in _ANTENNA_PATTERNS:
            raise ValueError(
                f"pattern must be one of {list(_ANTENNA_PATTERNS)}"
            )

        self._pattern = pattern
        # Register as buffer for CUDAGraph compatibility
        self.register_buffer("_slant_angle", torch.tensor(slant_angle, dtype=self.dtype, device=self.device))

        # Select the radiation field corresponding to the requested pattern
        if pattern == "omni":
            self._radiation_pattern = self._radiation_pattern_omni
        elif pattern == "38.901":
            self._radiation_pattern = self._radiation_pattern_38901
        else:
            self._radiation_pattern = self._radiation_pattern_38901_handheld

    @property
    def pattern(self) -> str:
        """Radiation pattern type."""
        return self._pattern

    @property
    def slant_angle(self) -> torch.Tensor:
        """Polarization slant angle [radian]"""
        return self._slant_angle

    def field(self, theta: torch.Tensor, phi: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Field pattern in the vertical and horizontal polarization (7.3-4/5)

        :param theta: Zenith angle wrapped within (0, pi) [radian]
        :param phi: Azimuth angle wrapped within (-pi, pi) [radian]
        """
        theta = theta.to(dtype=self.dtype, device=self.device)
        phi = phi.to(dtype=self.dtype, device=self.device)
        a = torch.sqrt(self._radiation_pattern(theta, phi))
        f_theta = a * torch.cos(self._slant_angle)
        f_phi = a * torch.sin(self._slant_angle)
        return (f_theta, f_phi)

    def show(self) -> None:
        """Shows the field pattern of an antenna element"""
        theta = torch.linspace(0.0, PI, 361, dtype=self.dtype, device=self.device)
        phi = torch.linspace(-PI, PI, 361, dtype=self.dtype, device=self.device)
        a_v = 10 * torch.log10(self._radiation_pattern(theta, torch.zeros_like(theta)))
        a_h = 10 * torch.log10(self._radiation_pattern(PI / 2 * torch.ones_like(phi), phi))

        # Convert to numpy for plotting
        theta_np = theta.cpu().numpy()
        phi_np = phi.cpu().numpy()
        a_v_np = a_v.cpu().numpy()
        a_h_np = a_h.cpu().numpy()

        fig = plt.figure()
        plt.polar(theta_np, a_v_np)
        fig.axes[0].set_theta_zero_location("N")
        fig.axes[0].set_theta_direction(-1)
        plt.title(r"Vertical cut of the radiation pattern ($\phi = 0$)")
        plt.legend([f"{self._pattern}"])

        fig = plt.figure()
        plt.polar(phi_np, a_h_np)
        fig.axes[0].set_theta_zero_location("E")
        plt.title(r"Horizontal cut of the radiation pattern ($\theta = \pi/2$)")
        plt.legend([f"{self._pattern}"])

        theta = torch.linspace(0.0, PI, 50, dtype=self.dtype, device=self.device)
        phi = torch.linspace(-PI, PI, 50, dtype=self.dtype, device=self.device)
        phi_grid, theta_grid = torch.meshgrid(phi, theta, indexing='xy')
        a = self._radiation_pattern(theta_grid, phi_grid)
        x = a * torch.sin(theta_grid) * torch.cos(phi_grid)
        y = a * torch.sin(theta_grid) * torch.sin(phi_grid)
        z = a * torch.cos(theta_grid)

        # Convert to numpy for 3D plotting
        x_np = x.cpu().numpy()
        y_np = y.cpu().numpy()
        z_np = z.cpu().numpy()

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        ax.plot_surface(x_np, y_np, z_np, rstride=1, cstride=1,
                        linewidth=0, antialiased=False, alpha=0.5)
        ax.view_init(elev=30., azim=-45)
        plt.xlabel("x")
        plt.ylabel("y")
        ax.set_zlabel("z")
        plt.title(f"Radiation power pattern ({self._pattern})")

    def _radiation_pattern_omni(self, theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        """Radiation pattern of an omnidirectional 3D radiation pattern

        :param theta: Zenith angle
        :param phi: Azimuth angle
        """
        return torch.ones_like(theta)

    def _radiation_pattern_38901(self, theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        """Radiation pattern from TR 38.901 Table 7.3-1

        :param theta: Zenith angle wrapped within (0, pi) [radian]
        :param phi: Azimuth angle wrapped within (-pi, pi) [radian]
        """
        return _radiation_pattern_38901_table(
            theta, phi, 65.0, 65.0, 30.0, 30.0, 8.0
        )

    def _radiation_pattern_38901_handheld(self, theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        """Handheld UT radiation pattern from TR 38.901 Table 7.3-2

        :param theta: Zenith angle wrapped within (0, pi) [radian]
        :param phi: Azimuth angle wrapped within (-pi, pi) [radian]
        """
        return _radiation_pattern_38901_table(
            theta, phi, 125.0, 125.0, 22.5, 22.5, 5.3
        )

    def _compute_gain(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute antenna gain and directivity through numerical integration"""
        # Create angular meshgrid
        theta = torch.linspace(0.0, PI, 181, dtype=self.dtype, device=self.device)
        phi = torch.linspace(-PI, PI, 361, dtype=self.dtype, device=self.device)
        phi_grid, theta_grid = torch.meshgrid(phi, theta, indexing='xy')

        # Compute field strength over the grid
        f_theta, f_phi = self.field(theta_grid, phi_grid)
        u = f_theta ** 2 + f_phi ** 2
        gain_db = 10 * torch.log10(torch.max(u))

        # Numerical integration of the field components
        dtheta = theta[1] - theta[0]
        dphi = phi[1] - phi[0]
        po = torch.sum(u * torch.sin(theta_grid) * dtheta * dphi)

        # Compute directivity
        u_bar = po / (4 * PI)  # Equivalent isotropic radiator
        d = u / u_bar  # Directivity grid
        directivity_db = 10 * torch.log10(torch.max(d))
        return (gain_db, directivity_db)


class HandheldUTArray(Object):
    # pylint: disable=line-too-long
    r"""
    Handheld UT antenna array from TR 38.901 Clause 7.3.

    .. _handheld-ut-array-geometry:

    .. figure:: ../../../figures/handheld_ut_array.svg
       :align: center
       :width: 360px

       Candidate antenna locations on the handheld UT device in the local
       :math:`x-y` plane. The local :math:`z`-axis is the reference
       orientation vector and points out of the shown plane.

    This class implements the handheld UT placement and field-rotation rules
    introduced by TR 38.901 V19.2.0 :cite:p:`TR38901V1920`. The element power
    pattern is selected through :class:`~sionna.phy.channel.tr38901.AntennaElement`;
    for standard-compliant handheld operation, use
    ``antenna_pattern="38.901-handheld"``, which follows Table 7.3-2. Antenna
    ports are placed at the candidate locations of Figure 7.3-2 on a flat
    device in the local :math:`x-y` plane. The default dimensions are
    :math:`15\,\mathrm{cm}` along the local :math:`x`-axis and
    :math:`7\,\mathrm{cm}` along the local :math:`y`-axis. The reference
    orientation vector is the local :math:`z`-axis, as shown in Figure 7.3-3.

    Note that this differs from the convention of
    :class:`~sionna.phy.channel.tr38901.PanelArray`, whose elements lie in the
    local :math:`y-z` plane and share the boresight :math:`+x`. Here every port
    has its own boresight, pointing radially outward from the device center and
    hence lying within the device plane, so :math:`+z` is not a boresight but
    the reference normal from which the polarization directions are derived.

    This class implements the same array interface as
    :class:`~sionna.phy.channel.tr38901.PanelArray` and can be used as the UT
    array for the :class:`~sionna.phy.channel.tr38901.CDL`,
    :class:`~sionna.phy.channel.tr38901.UMi`,
    :class:`~sionna.phy.channel.tr38901.UMa`,
    :class:`~sionna.phy.channel.tr38901.RMa`,
    :class:`~sionna.phy.channel.tr38901.InH`, and
    :class:`~sionna.phy.channel.tr38901.InF` models. The antenna model is not
    tied to the channel-model ``spec_version`` argument. Selecting
    ``spec_version="19.2"`` enables the V19.2 propagation parameters, but is
    not required for this array class to run.

    TR 38.901 defines this geometry for a handheld UT. The common Sionna array
    interface also permits the object to be supplied in roles such as a BS
    array, but such use is not a standardized handheld-UT configuration.


    :param carrier_frequency: Carrier frequency [Hz]. Used to establish the
        wavelength; calibration-frequency restrictions are not enforced.
    :param polarization: Polarization. One of ``"single"`` or ``"dual"``.
    :param polarization_type: Polarization type. For single polarization,
        ``"V"`` uses the Clause 7.3 single-field polarization direction and
        ``"H"`` uses the orthogonal direction. For dual polarization,
        ``"cross"`` applies the 45 degree rotation described for two field
        patterns in Clause 7.3, while ``"VH"`` uses the unrotated orthogonal
        pair. The two-field configuration is not intended for FR1. Defaults to
        ``"V"`` for single polarization and ``"cross"`` for dual
        polarization.
    :param antenna_locations: Candidate antenna locations from Figure 7.3-2;
        see :numref:`handheld-ut-array-geometry`. This can be a sequence of
        integers from 1 to 8, ``"tr38901"`` for all eight candidates, or
        ``"tr38901-4"`` for the four-corner subset ``(1, 7, 3, 5)`` used by
        several TR 38.901 calibration assumptions. The numbering follows the
        figure: 1, 2, and 3 are on the left edge of the top-down view from top
        to bottom, 4 is the bottom edge center, 5, 6, and 7 are on the right
        edge from bottom to top, and 8 is the top edge center. Defaults to
        ``"tr38901"``.
    :param antenna_pattern: Element radiation pattern. One of ``"omni"``,
        ``"38.901"``, or ``"38.901-handheld"``. Defaults to
        ``"38.901-handheld"``.
    :param device_depth: Handheld-device extent along the local :math:`x`-axis
        [m]. Defaults to 0.15 m.
    :param device_width: Handheld-device width [m]. Defaults to 0.07 m.
    :param port_power_offsets_db: Optional per-port attenuation [dB] for
        antenna imbalance. Positive values reduce the field amplitude. If
        set to `None`, no imbalance is applied, as specified by default in
        Table 7.3-2.
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`, :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for computation (e.g., ``"cpu"``, ``"cuda:0"``).
        If `None`, :attr:`~sionna.phy.config.Config.device` is used.

    .. rubric:: Examples

    .. code-block:: python

        from sionna.phy.channel.tr38901 import HandheldUTArray

        # Four single-polarized handheld antenna ports at locations 1, 7, 3, 5
        ut_array = HandheldUTArray(carrier_frequency=7e9,
                                   polarization="single",
                                   antenna_locations="tr38901-4")
        ut_array.show_element_radiation_pattern()

        # Eight ports: two field patterns at the four corner locations
        ut_array_dual = HandheldUTArray(carrier_frequency=15e9,
                                        polarization="dual",
                                        antenna_locations="tr38901-4")
        ut_array_dual.show()

    .. figure:: ../../../figures/handheld_ut_array_show.png
       :align: center
       :width: 500px

       Output of :meth:`show` for the dual-polarized array of the example
       above. Selected locations are drawn in red, unselected candidates in
       grey. The red segment shows the in-plane polarization direction of the
       first port of a location. The two polarizations are orthogonal in three
       dimensions, but their projections onto this top-down view coincide, so
       the second port is drawn as a perpendicular segment that marks the
       polarization pair rather than its true orientation.

    .. figure:: ../../../figures/handheld_ut_radiation_pattern.png
       :align: center
       :width: 700px

       Example output of :meth:`show_element_radiation_pattern` for the
       handheld UT reference antenna element.
    """

    _CANDIDATE_FRACTIONS = {
        1: (-0.5, -0.5),
        2: (0.0, -0.5),
        3: (0.5, -0.5),
        4: (0.5, 0.0),
        5: (0.5, 0.5),
        6: (0.0, 0.5),
        7: (-0.5, 0.5),
        8: (-0.5, 0.0),
    }

    def __init__(
        self,
        carrier_frequency: float,
        polarization: str = "single",
        polarization_type: Optional[str] = None,
        antenna_locations: Union[str, Sequence[int]] = "tr38901",
        antenna_pattern: str = "38.901-handheld",
        device_depth: float = 0.15,
        device_width: float = 0.07,
        port_power_offsets_db: Optional[Sequence[float]] = None,
        precision: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        super().__init__(precision=precision, device=device)

        if polarization not in ("single", "dual"):
            raise ValueError("polarization must be either 'single' or 'dual'")
        if polarization_type is None:
            polarization_type = "V" if polarization == "single" else "cross"
        if polarization == "single":
            if polarization_type not in ("V", "H"):
                raise ValueError(
                    "For single polarization, polarization_type must be 'V' "
                    "or 'H'"
                )
        else:
            if polarization_type not in ("VH", "cross"):
                raise ValueError(
                    "For dual polarization, polarization_type must be 'VH' "
                    "or 'cross'"
                )

        candidate_indices = self._parse_antenna_locations(antenna_locations)
        positions = self._candidate_positions(candidate_indices,
                                              device_depth,
                                              device_width)
        port_positions, port_basis, port_components = self._build_ports(
            positions, polarization, polarization_type
        )
        num_locations = len(candidate_indices)
        num_ant = port_positions.shape[0]

        if port_power_offsets_db is None:
            port_power_offsets_db = torch.zeros(num_ant, dtype=self.dtype,
                                                device=self.device)
        else:
            port_power_offsets_db = torch.as_tensor(
                port_power_offsets_db, dtype=self.dtype, device=self.device
            )
        if tuple(port_power_offsets_db.shape) != (num_ant,):
            raise ValueError(
                "port_power_offsets_db must have one value per antenna port"
            )

        self._polarization = polarization
        self._polarization_type = polarization_type
        self._antenna_locations = tuple(candidate_indices)
        self._antenna_pattern = antenna_pattern
        self._num_ant = num_ant
        self._num_locations = num_locations
        self._num_panels = 1
        self._num_panel_ant = num_ant

        self.register_buffer("_lambda_0", torch.tensor(
            SPEED_OF_LIGHT / carrier_frequency, dtype=self.dtype,
            device=self.device))
        self.register_buffer("_device_size", torch.tensor(
            [device_depth, device_width], dtype=self.dtype,
            device=self.device))
        self.register_buffer("_candidate_indices", torch.tensor(
            candidate_indices, dtype=torch.int64, device=self.device))
        self.register_buffer("_candidate_pos", torch.tensor(
            positions, dtype=self.dtype, device=self.device))
        self.register_buffer("_ant_pos", torch.tensor(
            port_positions, dtype=self.dtype, device=self.device))
        self.register_buffer("_port_basis", torch.tensor(
            port_basis, dtype=self.dtype, device=self.device))
        self.register_buffer("_port_field_component", torch.tensor(
            port_components, dtype=torch.int64, device=self.device))
        self.register_buffer("_port_power_offsets_db", port_power_offsets_db)

        if polarization == "single":
            ant_ind_pol1 = np.arange(num_ant)
            ant_ind_pol2 = np.array([], dtype=np.int64)
        else:
            ant_ind_pol1 = np.arange(num_locations)
            ant_ind_pol2 = np.arange(num_locations, 2*num_locations)
        self.register_buffer("_ant_ind_pol1", torch.tensor(
            ant_ind_pol1, dtype=torch.int64, device=self.device))
        self.register_buffer("_ant_ind_pol2", torch.tensor(
            ant_ind_pol2, dtype=torch.int64, device=self.device))
        self.register_buffer("_ant_pos_pol1", self._ant_pos[self._ant_ind_pol1])
        self.register_buffer("_ant_pos_pol2",
                             self._ant_pos[self._ant_ind_pol2]
                             if polarization == "dual" else
                             torch.tensor([], dtype=self.dtype,
                                          device=self.device))

        self._ant_pol1 = AntennaElement(antenna_pattern, 0.0,
                                        precision=self.precision,
                                        device=self.device)
        self._ant_pol2 = None
        if polarization == "dual":
            self._ant_pol2 = AntennaElement(antenna_pattern, PI/2,
                                            precision=self.precision,
                                            device=self.device)

    @staticmethod
    def _parse_antenna_locations(
        antenna_locations: Union[str, Sequence[int]]
    ) -> Tuple[int, ...]:
        """Parse Figure 7.3-2 candidate antenna locations."""
        if isinstance(antenna_locations, str):
            if antenna_locations == "tr38901":
                candidate_indices = tuple(range(1, 9))
            elif antenna_locations == "tr38901-4":
                candidate_indices = (1, 7, 3, 5)
            else:
                raise ValueError(
                    "antenna_locations must be 'tr38901', 'tr38901-4', "
                    "or a sequence of candidate indices"
                )
        else:
            candidate_indices = tuple(int(i) for i in antenna_locations)
        if not candidate_indices:
            raise ValueError("At least one antenna candidate location is required")
        unsupported = [i for i in candidate_indices
                       if i not in HandheldUTArray._CANDIDATE_FRACTIONS]
        if unsupported:
            raise ValueError(
                "Handheld UT candidate locations must be integers from 1 to 8"
            )
        return candidate_indices

    @classmethod
    def _candidate_positions(
        cls,
        candidate_indices: Sequence[int],
        device_depth: float,
        device_width: float,
    ) -> np.ndarray:
        """Return candidate antenna positions in the UT local coordinate system."""
        positions = []
        for index in candidate_indices:
            x_frac, y_frac = cls._CANDIDATE_FRACTIONS[index]
            positions.append([x_frac*device_depth, y_frac*device_width, 0.0])
        return np.asarray(positions, dtype=np.float64)

    @staticmethod
    def _normalize_np(v: np.ndarray) -> np.ndarray:
        """Normalize a NumPy vector."""
        norm = np.linalg.norm(v)
        if norm == 0:
            raise ValueError("Candidate antenna position must not be the origin")
        return v / norm

    @staticmethod
    def _rotate_np(v: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
        """Rotate vector ``v`` about ``axis`` by ``angle`` using Rodrigues' rule."""
        axis = HandheldUTArray._normalize_np(axis)
        return (v*np.cos(angle)
                + np.cross(axis, v)*np.sin(angle)
                + axis*np.dot(axis, v)*(1.0 - np.cos(angle)))

    @staticmethod
    def _basis_from_boresight_and_polarization(
        boresight: np.ndarray,
        polarization_direction: np.ndarray,
    ) -> np.ndarray:
        """Build a right-handed local-to-device rotation matrix."""
        x_axis = HandheldUTArray._normalize_np(boresight)
        p_axis = HandheldUTArray._normalize_np(polarization_direction)
        z_axis = -p_axis
        y_axis = HandheldUTArray._normalize_np(np.cross(z_axis, x_axis))
        z_axis = HandheldUTArray._normalize_np(np.cross(x_axis, y_axis))
        return np.stack([x_axis, y_axis, z_axis], axis=-1)

    @classmethod
    def _build_ports(
        cls,
        positions: np.ndarray,
        polarization: str,
        polarization_type: str,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build per-port positions, local bases, and reference components."""
        device_normal = np.asarray([0.0, 0.0, 1.0])
        pol1_positions = []
        pol1_basis = []
        pol2_basis = []

        for pos in positions:
            boresight = cls._normalize_np(pos)
            single_pol = cls._normalize_np(np.cross(boresight,
                                                    device_normal))
            orthogonal_pol = cls._normalize_np(np.cross(boresight,
                                                        single_pol))

            if polarization == "single":
                pol = single_pol if polarization_type == "V" else orthogonal_pol
                pol1_positions.append(pos)
                pol1_basis.append(
                    cls._basis_from_boresight_and_polarization(boresight, pol)
                )
            else:
                if polarization_type == "cross":
                    pol1 = cls._rotate_np(single_pol, boresight, PI/4)
                else:
                    pol1 = single_pol
                pol1_positions.append(pos)
                pol1_basis.append(
                    cls._basis_from_boresight_and_polarization(boresight, pol1)
                )
                # The second port uses the same local basis but selects
                # F_phi' instead of F_theta', which is the orthogonal field
                # component in the local frame.
                pol2_basis.append(
                    cls._basis_from_boresight_and_polarization(boresight, pol1)
                )

        if polarization == "single":
            return (np.asarray(pol1_positions),
                    np.asarray(pol1_basis),
                    np.zeros(len(pol1_positions), dtype=np.int64))

        port_positions = np.concatenate([positions, positions], axis=0)
        port_basis = np.concatenate(
            [np.asarray(pol1_basis), np.asarray(pol2_basis)],
            axis=0,
        )
        port_components = np.concatenate(
            [
                np.zeros(len(positions), dtype=np.int64),
                np.ones(len(positions), dtype=np.int64),
            ],
            axis=0,
        )
        return port_positions, port_basis, port_components

    @property
    def antenna_locations(self) -> Tuple[int, ...]:
        """Figure 7.3-2 candidate antenna location indices."""
        return self._antenna_locations

    @property
    def antenna_pattern(self) -> str:
        """Element radiation pattern."""
        return self._antenna_pattern

    @property
    def device_size(self) -> torch.Tensor:
        """Handheld-device size as ``[depth, width]`` [m]."""
        return self._device_size

    @property
    def candidate_pos(self) -> torch.Tensor:
        """Candidate antenna positions in the local coordinate system [m]."""
        return self._candidate_pos

    @property
    def port_basis(self) -> torch.Tensor:
        """Per-port local-to-device rotation matrices."""
        return self._port_basis

    @property
    def port_field_component(self) -> torch.Tensor:
        """Reference field component per port, 0 for theta and 1 for phi."""
        return self._port_field_component

    @property
    def port_power_offsets_db(self) -> torch.Tensor:
        """Per-port antenna-imbalance attenuation [dB]."""
        return self._port_power_offsets_db

    @property
    def polarization(self) -> str:
        """Polarization ('single' or 'dual')."""
        return self._polarization

    @property
    def polarization_type(self) -> str:
        """Polarization type."""
        return self._polarization_type

    @property
    def num_panels(self) -> int:
        """Number of panels."""
        return self._num_panels

    @property
    def num_panels_ant(self) -> int:
        """Number of antenna elements per panel."""
        return self._num_panel_ant

    @property
    def num_ant(self) -> int:
        """Total number of antenna ports."""
        return self._num_ant

    @property
    def ant_pol1(self) -> AntennaElement:
        """Reference element for the first polarization direction."""
        return self._ant_pol1

    @property
    def ant_pol2(self) -> AntennaElement:
        """Reference element for the second polarization direction."""
        if self._polarization != "dual":
            raise RuntimeError(
                "This property is not defined with single polarization"
            )
        return self._ant_pol2

    @property
    def ant_pos(self) -> torch.Tensor:
        """Positions of the antenna ports [m]."""
        return self._ant_pos

    @property
    def ant_ind_pol1(self) -> torch.Tensor:
        """Indices of antenna ports with the first polarization direction."""
        return self._ant_ind_pol1

    @property
    def ant_ind_pol2(self) -> torch.Tensor:
        """Indices of antenna ports with the second polarization direction."""
        if self._polarization != "dual":
            raise RuntimeError(
                "This property is not defined with single polarization"
            )
        return self._ant_ind_pol2

    @property
    def ant_pos_pol1(self) -> torch.Tensor:
        """Positions of antenna ports with the first polarization direction."""
        return self._ant_pos_pol1

    @property
    def ant_pos_pol2(self) -> torch.Tensor:
        """Positions of antenna ports with the second polarization direction."""
        if self._polarization != "dual":
            raise RuntimeError(
                "This property is not defined with single polarization"
            )
        return self._ant_pos_pol2

    def element_field(
        self,
        theta: torch.Tensor,
        phi: torch.Tensor,
    ) -> torch.Tensor:
        r"""Compute per-port field vectors in the handheld-device LCS.

        The incoming direction is first rotated into the local coordinate
        system of each candidate antenna, the reference field
        :math:`F_{\theta^{\prime\prime}}^{\prime\prime}` or
        :math:`F_{\phi^{\prime\prime}}^{\prime\prime}` is evaluated, and the
        field is then rotated back according to (7.3-6)--(7.3-8) of
        TR 38.901.

        For the standard single-polarized handheld model, the boresight
        direction of candidate :math:`u` is

        .. math::

            \hat{\mathbf{b}}_u =
            \frac{\mathbf{r}_u}{\lVert\mathbf{r}_u\rVert}

        where :math:`\mathbf{r}_u` is the candidate position relative to the
        handset center. With the reference handheld normal
        :math:`\hat{\mathbf{z}}`, the polarization direction is

        .. math::

            \hat{\mathbf{p}}_u =
            \hat{\mathbf{b}}_u \times \hat{\mathbf{z}}

        which is parallel to the handheld plane and perpendicular to the
        direction from the handset center to the candidate location, as shown
        in Figure 7.3-7 of TR 38.901. The double-primed local basis is chosen
        such that the positive
        :math:`F_{\theta^{\prime\prime}}^{\prime\prime}` component at boresight
        is aligned with :math:`\hat{\mathbf{p}}_u`.

        :param theta: Zenith angle in the handheld-device LCS [radian].
        :param phi: Azimuth angle in the handheld-device LCS [radian].

        :output: Tensor of shape ``[num_ant] + theta.shape + [2]``. The last
            dimension contains
            :math:`(F_{\theta^\prime}^\prime,F_{\phi^\prime}^\prime)`.
        """
        theta = theta.to(dtype=self.dtype, device=self.device)
        phi = phi.to(dtype=self.dtype, device=self.device)

        rho = _spherical_unit_vector(theta, phi)
        rho_local = torch.einsum("aij,...i->a...j", self._port_basis, rho)
        theta_local = torch.acos(rho_local[..., 2].clamp(-1.0, 1.0))
        phi_local = torch.atan2(rho_local[..., 1], rho_local[..., 0])

        power = self._ant_pol1._radiation_pattern(theta_local, phi_local)
        amplitude = torch.sqrt(power)

        e_theta_local, e_phi_local = _spherical_basis(theta_local, phi_local)
        component = self._port_field_component.reshape(
            [-1] + [1]*theta.dim() + [1]
        )
        field_local = torch.where(component == 0,
                                  e_theta_local,
                                  e_phi_local)
        field_local = field_local * amplitude.unsqueeze(-1)

        field_device = torch.einsum(
            "aij,a...j->a...i", self._port_basis, field_local
        )
        e_theta, e_phi = _spherical_basis(theta, phi)
        f_theta = (field_device * e_theta.unsqueeze(0)).sum(dim=-1)
        f_phi = (field_device * e_phi.unsqueeze(0)).sum(dim=-1)

        loss = 10 ** (-self._port_power_offsets_db / 20)
        loss = loss.reshape([-1] + [1]*theta.dim())
        f_theta = f_theta * loss
        f_phi = f_phi * loss
        return torch.stack([f_theta, f_phi], dim=-1)

    def show(self) -> None:
        """Show the handheld-device geometry and selected antenna locations."""
        depth = float(self._device_size[0].detach().cpu().item())
        width = float(self._device_size[1].detach().cpu().item())
        all_indices = tuple(self._CANDIDATE_FRACTIONS.keys())
        all_positions = self._candidate_positions(all_indices, depth, width)
        selected_positions = self._candidate_pos.detach().cpu().numpy()
        selected_basis = self._port_basis.detach().cpu().numpy()
        selected_components = self._port_field_component.detach().cpu().numpy()
        num_selected_locations = len(self._antenna_locations)

        fig, ax = plt.subplots()
        rect = plt.Rectangle(
            (-width/2, -depth/2),
            width,
            depth,
            fill=False,
            linewidth=1.5,
            edgecolor="black",
            label="Device outline",
        )
        ax.add_patch(rect)
        ax.plot(
            all_positions[:, 1],
            all_positions[:, 0],
            marker="o",
            markerfacecolor="none",
            markeredgecolor="0.65",
            markersize="8",
            linestyle="None",
            label="Candidate locations",
        )
        for index, pos in zip(all_indices, all_positions):
            ax.annotate(str(index), (pos[1], pos[0]), xytext=(4, 4),
                        textcoords="offset points", color="0.35")

        ax.plot(
            selected_positions[:, 1],
            selected_positions[:, 0],
            marker="o",
            markerfacecolor="none",
            markeredgecolor="red",
            markersize="10",
            linestyle="None",
            markeredgewidth="2",
            label="Selected ports",
        )

        segment_length = 0.30 * min(depth, width)

        def in_plane_direction(port):
            """Unit direction of a port polarization within the plotted plane"""
            basis = selected_basis[port]
            if selected_components[port] == 0:
                pol_axis = -basis[:, 2]
            else:
                pol_axis = basis[:, 1]
            direction = pol_axis[:2]
            norm = np.linalg.norm(direction)
            return direction / norm if norm > 0 else np.array([1.0, 0.0])

        def draw_polarization_segment(pos, direction, color, label):
            start = pos - 0.5 * segment_length * direction
            stop = pos + 0.5 * segment_length * direction
            ax.plot([start[1], stop[1]], [start[0], stop[0]],
                    color=color, linewidth=2.0, label=label)

        label_pol1 = "Polarization 1"
        label_pol2 = "Polarization 2"
        for port in range(num_selected_locations):
            pos = selected_positions[port][:2]
            direction = in_plane_direction(port)
            draw_polarization_segment(pos, direction, "red", label_pol1)
            # Only the first segment of a polarization carries the label
            label_pol1 = "_nolegend_"
            if self._polarization == "dual":
                # The two polarizations are orthogonal in 3D, but their
                # projections onto this plane coincide. The second one is drawn
                # perpendicular to the first to mark the pair of ports of a
                # location; its shown direction is schematic
                draw_polarization_segment(
                    pos, np.array([-direction[1], direction[0]]), "black",
                    label_pol2)
                label_pol2 = "_nolegend_"

        ax.set_aspect("equal", adjustable="box")
        pad = 0.1 * max(width, depth)
        ax.set_xlim(-width/2 - pad, width/2 + pad)
        ax.set_ylim(-depth/2 - pad, depth/2 + pad)
        ax.invert_yaxis()
        ax.set_xlabel("y (m)")
        ax.set_ylabel("x (m)")
        ax.set_title("Handheld UT Array")
        # The device outline fills the axes, so an inside legend would hide
        # candidate locations
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0),
                  borderaxespad=0.0)
        fig.tight_layout()

    def show_element_radiation_pattern(self) -> None:
        """Show the radiation field of the reference antenna element."""
        self._ant_pol1.show()


class AntennaPanel(Object):
    """Antenna panel following the :cite:p:`TR38901V1920` specification

    :param num_rows: Number of rows forming the panel
    :param num_cols: Number of columns forming the panel
    :param polarization: Polarization. One of ``"single"`` or ``"dual"``.
    :param vertical_spacing: Vertical antenna element spacing
        [multiples of wavelength]
    :param horizontal_spacing: Horizontal antenna element spacing
        [multiples of wavelength]
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`, :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for computation (e.g., ``"cpu"``, ``"cuda:0"``).
        If `None`, :attr:`~sionna.phy.config.Config.device` is used.
    """

    def __init__(
        self,
        num_rows: int,
        num_cols: int,
        polarization: str,
        vertical_spacing: float,
        horizontal_spacing: float,
        precision: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        super().__init__(precision=precision, device=device)
        if polarization not in ("single", "dual"):
            raise ValueError("polarization must be either 'single' or 'dual'")

        self._num_rows = num_rows
        self._num_cols = num_cols
        self._polarization = polarization
        # Register as buffers for CUDAGraph compatibility
        self.register_buffer("_horizontal_spacing", torch.tensor(horizontal_spacing, dtype=self.dtype, device=self.device))
        self.register_buffer("_vertical_spacing", torch.tensor(vertical_spacing, dtype=self.dtype, device=self.device))

        # Place the antenna elements of the first polarization direction
        # on the y-z-plane
        p = 1 if polarization == 'single' else 2
        ant_pos = np.zeros([num_rows * num_cols * p, 3])
        for i in range(num_rows):
            for j in range(num_cols):
                ant_pos[i + j * num_rows] = [0,
                                              j * horizontal_spacing,
                                              -i * vertical_spacing]

        # Center the panel around the origin
        offset = [0,
                  -(num_cols - 1) * horizontal_spacing / 2,
                  (num_rows - 1) * vertical_spacing / 2]
        ant_pos += offset

        # Create the antenna elements of the second polarization direction
        if polarization == 'dual':
            ant_pos[num_rows * num_cols:] = ant_pos[:num_rows * num_cols]
        # Register as buffer for CUDAGraph compatibility
        self.register_buffer("_ant_pos", torch.tensor(ant_pos, dtype=self.dtype, device=self.device))

    @property
    def ant_pos(self) -> torch.Tensor:
        """Antenna positions in the local coordinate system"""
        return self._ant_pos

    @property
    def num_rows(self) -> int:
        """Number of rows"""
        return self._num_rows

    @property
    def num_cols(self) -> int:
        """Number of columns"""
        return self._num_cols

    @property
    def polarization(self) -> str:
        """Polarization, either ``"single"`` or ``"dual"``."""
        return self._polarization

    @property
    def vertical_spacing(self) -> torch.Tensor:
        """Vertical spacing between elements [multiple of wavelength]"""
        return self._vertical_spacing

    @property
    def horizontal_spacing(self) -> torch.Tensor:
        """Horizontal spacing between elements [multiple of wavelength]"""
        return self._horizontal_spacing

    def show(self) -> None:
        """Shows the panel geometry"""
        fig = plt.figure()
        pos = self._ant_pos[:self._num_rows * self._num_cols].cpu().numpy()
        plt.plot(pos[:, 1], pos[:, 2], marker="|", markeredgecolor='red',
                 markersize="20", linestyle="None", markeredgewidth="2")
        for i, p in enumerate(pos):
            fig.axes[0].annotate(i + 1, (p[1], p[2]))
        if self._polarization == 'dual':
            pos = self._ant_pos[self._num_rows * self._num_cols:].cpu().numpy()
            plt.plot(pos[:, 1], pos[:, 2], marker="_", markeredgecolor='black',
                     markersize="20", linestyle="None", markeredgewidth="1")
        plt.xlabel(r"y ($\lambda_0$)")
        plt.ylabel(r"z ($\lambda_0$)")
        plt.title("Antenna Panel")
        plt.legend(["Polarization 1", "Polarization 2"], loc="upper right")


class PanelArray(Object):
    # pylint: disable=line-too-long
    r"""
    Antenna panel array following the :cite:p:`TR38901V1920` specification

    This class is used to create models of the panel arrays used by the
    transmitters and receivers and that need to be specified when using the
    :class:`~sionna.phy.channel.tr38901.CDL`,
    :class:`~sionna.phy.channel.tr38901.UMi`,
    :class:`~sionna.phy.channel.tr38901.UMa`,
    :class:`~sionna.phy.channel.tr38901.RMa`, and
    :class:`~sionna.phy.channel.tr38901.InH`, and
    :class:`~sionna.phy.channel.tr38901.InF` models.

    An array is made of ``num_rows`` :math:`\times` ``num_cols`` panels, and
    every panel carries ``num_rows_per_panel`` :math:`\times`
    ``num_cols_per_panel`` elements. All elements lie in the :math:`y-z` plane
    of the local coordinate system and the array is centered on its origin.
    Element and panel spacings are specified in multiples of the wavelength
    that corresponds to ``carrier_frequency``. For dual polarization, the two
    orthogonally polarized elements of a position are co-located.

    .. _panel-array-geometry:

    .. figure:: ../../../figures/panel_array_geometry.svg
       :align: center
       :width: 480px

       Geometry of a panel array. The element spacings
       ``element_horizontal_spacing`` (:math:`d_\text{H}`) and
       ``element_vertical_spacing`` (:math:`d_\text{V}`) apply within a panel,
       whereas the panel spacings ``panel_horizontal_spacing``
       (:math:`D_\text{H}`) and ``panel_vertical_spacing`` (:math:`D_\text{V}`)
       are measured between corresponding elements of adjacent panels. The
       local :math:`x`-axis points out of the shown plane.

    :param num_rows_per_panel: Number of rows of elements per panel
    :param num_cols_per_panel: Number of columns of elements per panel
    :param polarization: Polarization. One of ``"single"`` or ``"dual"``.
    :param polarization_type: Type of polarization. For single polarization,
        must be ``"V"`` or ``"H"``.
        For dual polarization, must be ``"VH"`` or ``"cross"``.
    :param antenna_pattern: Element radiation pattern. One of ``"omni"``,
        ``"38.901"``, or ``"38.901-handheld"``.
    :param carrier_frequency: Carrier frequency [Hz]
    :param num_rows: Number of rows of panels. Defaults to 1.
    :param num_cols: Number of columns of panels. Defaults to 1.
    :param panel_vertical_spacing: Vertical spacing of panels
        [multiples of wavelength].
        Must be greater than the panel height.
        If set to `None`, it is set to the panel height + 0.5.
    :param panel_horizontal_spacing: Horizontal spacing of panels
        [in multiples of wavelength].
        Must be greater than the panel width.
        If set to `None`, it is set to the panel width + 0.5.
    :param element_vertical_spacing: Element vertical spacing
        [multiple of wavelength].
        Defaults to 0.5 if set to `None`.
    :param element_horizontal_spacing: Element horizontal spacing
        [multiple of wavelength].
        Defaults to 0.5 if set to `None`.
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`, :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for computation (e.g., ``"cpu"``, ``"cuda:0"``).
        If `None`, :attr:`~sionna.phy.config.Config.device` is used.

    .. rubric:: Examples

    .. code-block:: python

        from sionna.phy.channel.tr38901 import PanelArray

        array = PanelArray(num_rows_per_panel=4,
                           num_cols_per_panel=4,
                           polarization='dual',
                           polarization_type='VH',
                           antenna_pattern='38.901',
                           carrier_frequency=3.5e9,
                           num_cols=2,
                           panel_horizontal_spacing=3.)
        array.show()

    .. figure:: ../../../figures/panel_array_show.png
       :align: center
       :width: 600px

       Output of :meth:`show` for the array of the example above. The two
       panels of 4 :math:`\times` 4 vertically and horizontally polarized
       elements are separated by three wavelengths, and the annotations are the
       port indices.
    """

    def __init__(
        self,
        num_rows_per_panel: int,
        num_cols_per_panel: int,
        polarization: str,
        polarization_type: str,
        antenna_pattern: str,
        carrier_frequency: float,
        num_rows: int = 1,
        num_cols: int = 1,
        panel_vertical_spacing: Optional[float] = None,
        panel_horizontal_spacing: Optional[float] = None,
        element_vertical_spacing: Optional[float] = None,
        element_horizontal_spacing: Optional[float] = None,
        precision: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        super().__init__(precision=precision, device=device)

        if polarization not in ("single", "dual"):
            raise ValueError("polarization must be either 'single' or 'dual'")

        # Setting default values for antenna and panel spacings if not
        # specified by the user
        # Default spacing for antenna elements is half a wavelength
        if element_vertical_spacing is None:
            element_vertical_spacing = 0.5
        if element_horizontal_spacing is None:
            element_horizontal_spacing = 0.5
        # Default values of panel spacing is the panel size + 0.5
        if panel_vertical_spacing is None:
            panel_vertical_spacing = (num_rows_per_panel - 1) \
                * element_vertical_spacing + 0.5
        if panel_horizontal_spacing is None:
            panel_horizontal_spacing = (num_cols_per_panel - 1) \
                * element_horizontal_spacing + 0.5

        # Check that panel spacing is larger than panel dimensions
        if not (
            panel_horizontal_spacing
            > (num_cols_per_panel - 1) * element_horizontal_spacing
        ):
            raise ValueError(
                "Panel horizontal spacing must be larger than the panel width"
            )
        if not (
            panel_vertical_spacing
            > (num_rows_per_panel - 1) * element_vertical_spacing
        ):
            raise ValueError(
                "Panel vertical spacing must be larger than panel height"
            )

        self._num_rows = num_rows
        self._num_cols = num_cols
        self._num_rows_per_panel = num_rows_per_panel
        self._num_cols_per_panel = num_cols_per_panel
        self._polarization = polarization
        self._polarization_type = polarization_type
        # Register as buffers for CUDAGraph compatibility
        self.register_buffer("_panel_vertical_spacing", torch.tensor(panel_vertical_spacing, dtype=self.dtype, device=self.device))
        self.register_buffer("_panel_horizontal_spacing", torch.tensor(panel_horizontal_spacing, dtype=self.dtype, device=self.device))
        self.register_buffer("_element_vertical_spacing", torch.tensor(element_vertical_spacing, dtype=self.dtype, device=self.device))
        self.register_buffer("_element_horizontal_spacing", torch.tensor(element_horizontal_spacing, dtype=self.dtype, device=self.device))

        self._num_panels = num_cols * num_rows

        p = 1 if polarization == 'single' else 2
        self._num_panel_ant = num_cols_per_panel * num_rows_per_panel * p
        # Total number of antenna elements
        self._num_ant = self._num_panels * self._num_panel_ant

        # Wavelength (m)
        # Register as buffer for CUDAGraph compatibility
        self.register_buffer("_lambda_0", torch.tensor(SPEED_OF_LIGHT / carrier_frequency, dtype=self.dtype, device=self.device))

        # Create one antenna element for each polarization direction
        # polarization must be one of {"V", "H", "VH", "cross"}
        if polarization == 'single':
            if polarization_type not in ("V", "H"):
                raise ValueError(
                    "For single polarization, polarization_type must be 'V' "
                    "or 'H'"
                )
            slant_angle = 0 if polarization_type == "V" else PI / 2
            self._ant_pol1 = AntennaElement(antenna_pattern, slant_angle,
                                            precision=self.precision, device=self.device)
            self._ant_pol2 = None
        else:
            if polarization_type not in ("VH", "cross"):
                raise ValueError(
                    "For dual polarization, polarization_type must be 'VH' "
                    "or 'cross'"
                )
            slant_angle = 0 if polarization_type == "VH" else -PI / 4
            self._ant_pol1 = AntennaElement(antenna_pattern, slant_angle,
                                            precision=self.precision, device=self.device)
            self._ant_pol2 = AntennaElement(antenna_pattern, slant_angle + PI / 2,
                                            precision=self.precision, device=self.device)

        # Compose array from panels
        ant_pos = np.zeros([self._num_ant, 3])
        panel = AntennaPanel(num_rows_per_panel, num_cols_per_panel,
                             polarization, element_vertical_spacing, element_horizontal_spacing,
                             precision=self.precision, device=self.device)
        pos = panel.ant_pos.cpu().numpy()
        count = 0
        num_panel_ant = self._num_panel_ant
        for j in range(num_cols):
            for i in range(num_rows):
                offset = [0,
                          j * panel_horizontal_spacing,
                          -i * panel_vertical_spacing]
                new_pos = pos + offset
                ant_pos[count * num_panel_ant:(count + 1) * num_panel_ant] = new_pos
                count += 1

        # Center the entire panel array around the origin of the y-z plane
        offset = [0,
                  -(num_cols - 1) * panel_horizontal_spacing / 2,
                  (num_rows - 1) * panel_vertical_spacing / 2]
        ant_pos += offset

        # Scale antenna element positions by the wavelength
        ant_pos *= self._lambda_0.cpu().numpy()
        # Register as buffer for CUDAGraph compatibility
        self.register_buffer("_ant_pos", torch.tensor(ant_pos, dtype=self.dtype, device=self.device))

        # Compute indices of antennas for polarization directions
        ind = np.arange(0, self._num_ant)
        ind = np.reshape(ind, [self._num_panels * p, -1])
        # Register as buffers for CUDAGraph compatibility
        self.register_buffer("_ant_ind_pol1", torch.tensor(np.reshape(ind[::p], [-1]), dtype=torch.int64, device=self.device))
        if polarization == 'single':
            self.register_buffer("_ant_ind_pol2", torch.tensor(np.array([]), dtype=torch.int64, device=self.device))
        else:
            self.register_buffer("_ant_ind_pol2", torch.tensor(np.reshape(
                ind[1:self._num_panels * p:2], [-1]), dtype=torch.int64, device=self.device))

        # Get positions of antenna elements for each polarization direction
        self.register_buffer("_ant_pos_pol1", self._ant_pos[self._ant_ind_pol1])
        self.register_buffer("_ant_pos_pol2", self._ant_pos[self._ant_ind_pol2] if polarization == 'dual' else torch.tensor([], dtype=self.dtype, device=self.device))

    @property
    def num_rows(self) -> int:
        """Number of rows of panels"""
        return self._num_rows

    @property
    def num_cols(self) -> int:
        """Number of columns of panels"""
        return self._num_cols

    @property
    def num_rows_per_panel(self) -> int:
        """Number of rows of elements per panel"""
        return self._num_rows_per_panel

    @property
    def num_cols_per_panel(self) -> int:
        """Number of columns of elements per panel"""
        return self._num_cols_per_panel

    @property
    def polarization(self) -> str:
        """Polarization, either ``"single"`` or ``"dual"``."""
        return self._polarization

    @property
    def polarization_type(self) -> str:
        """Polarization type. ``"V"`` or ``"H"`` for single polarization.
        ``"VH"`` or ``"cross"`` for dual polarization."""
        return self._polarization_type

    @property
    def panel_vertical_spacing(self) -> torch.Tensor:
        """Vertical spacing between the panels [multiple of wavelength]"""
        return self._panel_vertical_spacing

    @property
    def panel_horizontal_spacing(self) -> torch.Tensor:
        """Horizontal spacing between the panels [multiple of wavelength]"""
        return self._panel_horizontal_spacing

    @property
    def element_vertical_spacing(self) -> torch.Tensor:
        """Vertical spacing between the antenna elements within a panel
        [multiple of wavelength]"""
        return self._element_vertical_spacing

    @property
    def element_horizontal_spacing(self) -> torch.Tensor:
        """Horizontal spacing between the antenna elements within a panel
        [multiple of wavelength]"""
        return self._element_horizontal_spacing

    @property
    def num_panels(self) -> int:
        """Number of panels"""
        return self._num_panels

    @property
    def num_panels_ant(self) -> int:
        """Number of antenna elements per panel"""
        return self._num_panel_ant

    @property
    def num_ant(self) -> int:
        """Total number of antenna elements"""
        return self._num_ant

    @property
    def ant_pol1(self) -> AntennaElement:
        """Field of an antenna element with the first polarization direction"""
        return self._ant_pol1

    @property
    def ant_pol2(self) -> AntennaElement:
        """Field of an antenna element with the second polarization direction.
        Only defined with dual polarization."""
        if self._polarization != 'dual':
            raise RuntimeError(
                "This property is not defined with single polarization"
            )
        return self._ant_pol2

    @property
    def ant_pos(self) -> torch.Tensor:
        """Positions of the antennas"""
        return self._ant_pos

    @property
    def ant_ind_pol1(self) -> torch.Tensor:
        """Indices of antenna elements with the first polarization direction"""
        return self._ant_ind_pol1

    @property
    def ant_ind_pol2(self) -> torch.Tensor:
        """Indices of antenna elements with the second polarization direction.
        Only defined with dual polarization."""
        if self._polarization != 'dual':
            raise RuntimeError(
                "This property is not defined with single polarization"
            )
        return self._ant_ind_pol2

    @property
    def ant_pos_pol1(self) -> torch.Tensor:
        """Positions of the antenna elements with the first polarization
        direction"""
        return self._ant_pos_pol1

    @property
    def ant_pos_pol2(self) -> torch.Tensor:
        """Positions of antenna elements with the second polarization direction.
        Only defined with dual polarization."""
        if self._polarization != 'dual':
            raise RuntimeError(
                "This property is not defined with single polarization"
            )
        return self._ant_pos_pol2

    def show(self) -> None:
        """Show the panel array geometry"""
        if self._polarization == 'single':
            if self._polarization_type == 'H':
                marker_p1 = MarkerStyle("_").get_marker()
            else:
                marker_p1 = MarkerStyle("|")
        else:  # 'dual'
            if self._polarization_type == 'cross':
                marker_p1 = (2, 0, -45)
                marker_p2 = (2, 0, 45)
            else:
                marker_p1 = MarkerStyle("_").get_marker()
                marker_p2 = MarkerStyle("|").get_marker()

        fig = plt.figure()
        pos_pol1 = self._ant_pos_pol1.cpu().numpy()
        plt.plot(pos_pol1[:, 1], pos_pol1[:, 2],
                 marker=marker_p1, markeredgecolor='red',
                 markersize="20", linestyle="None", markeredgewidth="2",
                 label="Polarization 1")
        ant_ind_pol1 = self._ant_ind_pol1.cpu().numpy()
        for i, p in enumerate(pos_pol1):
            fig.axes[0].annotate(ant_ind_pol1[i] + 1, (p[1], p[2]))
        if self._polarization == 'dual':
            pos_pol2 = self._ant_pos_pol2.cpu().numpy()
            plt.plot(pos_pol2[:, 1], pos_pol2[:, 2],
                     marker=marker_p2,  # pylint: disable=possibly-used-before-assignment
                     markeredgecolor='black',
                     markersize="20", linestyle="None", markeredgewidth="1",
                     label="Polarization 2")
        plt.xlabel("y (m)")
        plt.ylabel("z (m)")
        # Subclasses such as AntennaArray and Antenna share this method
        plt.title(type(self).__name__)
        # An inside legend would hide the port labels of the upper-right
        # elements
        plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0),
                   borderaxespad=0.0)
        fig.tight_layout()

    def show_element_radiation_pattern(self) -> None:
        """Show the radiation field of antenna elements forming the panel"""
        self._ant_pol1.show()


class Antenna(PanelArray):
    # pylint: disable=line-too-long
    r"""
    Single antenna following the :cite:p:`TR38901V1920` specification

    This class is a special case of :class:`~sionna.phy.channel.tr38901.PanelArray`,
    and can be used in lieu of it.

    :param polarization: Polarization. One of ``"single"`` or ``"dual"``.
    :param polarization_type: Type of polarization. For single polarization,
        must be ``"V"`` or ``"H"``.
        For dual polarization, must be ``"VH"`` or ``"cross"``.
    :param antenna_pattern: Element radiation pattern. One of ``"omni"``,
        ``"38.901"``, or ``"38.901-handheld"``.
    :param carrier_frequency: Carrier frequency [Hz]
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`, :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for computation (e.g., ``"cpu"``, ``"cuda:0"``).
        If `None`, :attr:`~sionna.phy.config.Config.device` is used.

    .. rubric:: Examples

    .. code-block:: python

        from sionna.phy.channel.tr38901 import Antenna

        ant = Antenna(polarization='single',
                      polarization_type='V',
                      antenna_pattern='omni',
                      carrier_frequency=3.5e9)
        print(ant.num_ant)
        # 1
    """

    def __init__(
        self,
        polarization: str,
        polarization_type: str,
        antenna_pattern: str,
        carrier_frequency: float,
        precision: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        super().__init__(
            num_rows_per_panel=1,
            num_cols_per_panel=1,
            polarization=polarization,
            polarization_type=polarization_type,
            antenna_pattern=antenna_pattern,
            carrier_frequency=carrier_frequency,
            precision=precision,
            device=device,
        )


class AntennaArray(PanelArray):
    # pylint: disable=line-too-long
    r"""
    Antenna array following the :cite:p:`TR38901V1920` specification

    This class is a special case of :class:`~sionna.phy.channel.tr38901.PanelArray`,
    and can be used in lieu of it.

    All ``num_rows`` :math:`\times` ``num_cols`` elements form a single panel in
    the :math:`y-z` plane of the local coordinate system, centered on its
    origin. Note that ``num_rows`` and ``num_cols`` count antenna elements here,
    whereas they count panels in
    :class:`~sionna.phy.channel.tr38901.PanelArray`.

    .. _antenna-array-geometry:

    .. figure:: ../../../figures/antenna_array_geometry.svg
       :align: center
       :width: 480px

       Geometry of an antenna array. The element spacings
       ``horizontal_spacing`` (:math:`d_\text{H}`) and ``vertical_spacing``
       (:math:`d_\text{V}`) are specified in multiples of the wavelength that
       corresponds to ``carrier_frequency``. For dual polarization, the two
       orthogonally polarized elements of a position are co-located. The local
       :math:`x`-axis points out of the shown plane.

    :param num_rows: Number of rows of elements
    :param num_cols: Number of columns of elements
    :param polarization: Polarization. One of ``"single"`` or ``"dual"``.
    :param polarization_type: Type of polarization. For single polarization,
        must be ``"V"`` or ``"H"``.
        For dual polarization, must be ``"VH"`` or ``"cross"``.
    :param antenna_pattern: Element radiation pattern. One of ``"omni"``,
        ``"38.901"``, or ``"38.901-handheld"``.
    :param carrier_frequency: Carrier frequency [Hz]
    :param vertical_spacing: Element vertical spacing [multiple of wavelength].
        Defaults to 0.5 if set to `None`.
    :param horizontal_spacing: Element horizontal spacing [multiple of wavelength].
        Defaults to 0.5 if set to `None`.
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`, :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for computation (e.g., ``"cpu"``, ``"cuda:0"``).
        If `None`, :attr:`~sionna.phy.config.Config.device` is used.

    .. rubric:: Examples

    .. code-block:: python

        from sionna.phy.channel.tr38901 import AntennaArray

        array = AntennaArray(num_rows=4,
                             num_cols=4,
                             polarization='dual',
                             polarization_type='cross',
                             antenna_pattern='38.901',
                             carrier_frequency=3.5e9)
        print(array.num_ant)
        # 32
        array.show()

    .. figure:: ../../../figures/antenna_array_show.png
       :align: center
       :width: 600px

       Output of :meth:`show` for the array of the example above. The 4
       :math:`\times` 4 positions each carry two cross-polarized elements, and
       the annotations are the port indices.
    """

    def __init__(
        self,
        num_rows: int,
        num_cols: int,
        polarization: str,
        polarization_type: str,
        antenna_pattern: str,
        carrier_frequency: float,
        vertical_spacing: Optional[float] = None,
        horizontal_spacing: Optional[float] = None,
        precision: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        super().__init__(
            num_rows_per_panel=num_rows,
            num_cols_per_panel=num_cols,
            polarization=polarization,
            polarization_type=polarization_type,
            antenna_pattern=antenna_pattern,
            carrier_frequency=carrier_frequency,
            element_vertical_spacing=vertical_spacing,
            element_horizontal_spacing=horizontal_spacing,
            precision=precision,
            device=device,
        )
