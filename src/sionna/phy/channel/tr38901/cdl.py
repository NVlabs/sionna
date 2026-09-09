#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Clustered delay line (CDL) channel model from 3GPP TR38.901 specification"""

from typing import Optional, Tuple

import numpy as np
import torch

from sionna.phy import PI, SPEED_OF_LIGHT
from sionna.phy.channel.channel_model import ChannelModel
from sionna.phy.channel.utils import deg_2_rad
from sionna.phy.utils import rand, normal
from .channel_coefficients import ChannelCoefficientsGenerator, Topology
from .rays import Rays
from . import models

__all__ = ["CDL"]


class CDL(ChannelModel):
    r"""
    Clustered delay line (CDL) channel model from the 3GPP :cite:p:`TR38901V1920`
    specification

    The power delay profiles (PDPs) are normalized to have a total energy
    of one.

    .. note::

        The CDL-A through CDL-E profile parameters are unchanged between
        TR 38.901 V16.1 and V19.2. Selecting either supported
        ``spec_version`` therefore produces the same CDL profile, but the
        resources remain versioned and are selected by ``spec_version``.

        The scalable TR 38.901 models apply from 0.5 GHz to 100 GHz and for
        system bandwidths of at most 2 GHz. These applicability limits are not
        enforced by this class.

    If a minimum speed and a maximum speed are specified such that the
    maximum speed is greater than the minimum speed, then UTs speeds are
    randomly and uniformly sampled from the specified interval for each link
    and each batch example.

    The CDL model only works for systems with a single transmitter and a
    single receiver. The transmitter and receiver can be equipped with
    multiple antennas.

    The channel coefficient generation is done following the procedure
    described in sections 7.7.1 and 7.7.3.

    :param model: CDL model to use. Must be ``"A"``, ``"B"``, ``"C"``,
        ``"D"``, or ``"E"``.
    :param delay_spread: RMS delay spread [s]. Ignored if
        ``normalize_delays`` is set to `False`.
    :param carrier_frequency: Carrier frequency [Hz]
    :param ut_array: Antenna array used by the UTs. All UTs share the same
        antenna array configuration.
    :param bs_array: Antenna array used by the base stations. All base stations share the same
        antenna array configuration.
    :param direction: Link direction. Must be ``"uplink"`` or
        ``"downlink"``.
    :param ut_orientation: Orientation of the UT. If set to `None`,
        [:math:`\pi`, 0, 0] is used. Shape [3] or [batch size, 3].
    :param bs_orientation: Orientation of the BS. If set to `None`,
        [0, 0, 0] is used. Shape [3] or [batch size, 3].
    :param ut_velocity: UT velocity vector [m/s]. If set to `None`,
        velocities are randomly sampled using ``min_speed`` and
        ``max_speed``. Shape [3] or [batch size, 3].
    :param min_speed: Minimum speed [m/s]. Ignored if ``ut_velocity`` is
        not `None`. Defaults to 0.
    :param max_speed: Maximum speed [m/s]. Ignored if ``ut_velocity`` is
        not `None`. Defaults to 0.
    :param normalize_delays: If set to `True`, the path delays are
        normalized such that the delay of the first path is zero.
        Defaults to `True`.
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`,
        :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for computation (e.g., ``"cpu"``, ``"cuda:0"``).
        If `None`, :attr:`~sionna.phy.config.Config.device` is used.
    :param spec_version: Version of the TR 38.901 parameter tables to use.
        Supported values are ``"16.1"`` and ``"19.2"``. Defaults to
        ``"19.2"``.

    :input batch_size: `int`.
        Batch size.

    :input num_time_steps: `int`.
        Number of time steps.

    :input sampling_frequency: `float`.
        Sampling frequency [Hz].

    :output a: [batch size, num_rx = 1, num_rx_ant, num_tx = 1, num_tx_ant, num_paths, num_time_steps], `torch.complex`.
        Path coefficients.

    :output tau: [batch size, num_rx = 1, num_tx = 1, num_paths], `torch.float`.
        Path delays [s].

    .. rubric:: Examples

    The following code snippet shows how to set up a CDL channel model and
    generate channel impulse responses:

    .. code-block:: python

        import torch
        from sionna.phy.channel.tr38901 import CDL, PanelArray

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        carrier_frequency = 3.5e9

        bs_array = PanelArray(num_rows_per_panel=1,
                              num_cols_per_panel=1,
                              polarization="dual",
                              polarization_type="cross",
                              antenna_pattern="38.901",
                              carrier_frequency=carrier_frequency,
                              device=device)
        ut_array = PanelArray(num_rows_per_panel=1,
                              num_cols_per_panel=1,
                              polarization="single",
                              polarization_type="V",
                              antenna_pattern="omni",
                              carrier_frequency=carrier_frequency,
                              device=device)

        channel_model = CDL(model="A",
                            delay_spread=300e-9,
                            carrier_frequency=carrier_frequency,
                            ut_array=ut_array,
                            bs_array=bs_array,
                            direction="downlink",
                            min_speed=0.0,
                            max_speed=3.0,
                            device=device)

        h, tau = channel_model(batch_size=32,
                               num_time_steps=14,
                               sampling_frequency=30.72e6)

    .. rubric:: Notes

    The following tables from :cite:p:`TR38901V1920` provide typical values for the
    delay spread.

    +--------------------------+-------------------+
    | Model                    | Delay spread [ns] |
    +==========================+===================+
    | Very short delay spread  | :math:`10`        |
    +--------------------------+-------------------+
    | Short delay spread       | :math:`30`        |
    +--------------------------+-------------------+
    | Nominal delay spread     | :math:`100`       |
    +--------------------------+-------------------+
    | Long delay spread        | :math:`300`       |
    +--------------------------+-------------------+
    | Very long delay spread   | :math:`1000`      |
    +--------------------------+-------------------+

    +-----------------------------------------------+------+------+----------+-----+----+-----+
    |              Delay spread [ns]                |             Frequency [GHz]             |
    +                                               +------+------+----+-----+-----+----+-----+
    |                                               |   2  |   6  | 15 |  28 |  39 | 60 |  70 |
    +========================+======================+======+======+====+=====+=====+====+=====+
    | Indoor office          | Short delay profile  | 20   | 16   | 16 | 16  | 16  | 16 | 16  |
    |                        +----------------------+------+------+----+-----+-----+----+-----+
    |                        | Normal delay profile | 39   | 30   | 24 | 20  | 18  | 16 | 16  |
    |                        +----------------------+------+------+----+-----+-----+----+-----+
    |                        | Long delay profile   | 59   | 53   | 47 | 43  | 41  | 38 | 37  |
    +------------------------+----------------------+------+------+----+-----+-----+----+-----+
    | UMi Street-canyon      | Short delay profile  | 65   | 45   | 37 | 32  | 30  | 27 | 26  |
    |                        +----------------------+------+------+----+-----+-----+----+-----+
    |                        | Normal delay profile | 129  | 93   | 76 | 66  | 61  | 55 | 53  |
    |                        +----------------------+------+------+----+-----+-----+----+-----+
    |                        | Long delay profile   | 634  | 316  | 307| 301 | 297 | 293| 291 |
    +------------------------+----------------------+------+------+----+-----+-----+----+-----+
    | UMa                    | Short delay profile  | 93   | 93   | 85 | 80  | 78  | 75 | 74  |
    |                        +----------------------+------+------+----+-----+-----+----+-----+
    |                        | Normal delay profile | 363  | 363  | 302| 266 | 249 |228 | 221 |
    |                        +----------------------+------+------+----+-----+-----+----+-----+
    |                        | Long delay profile   | 1148 | 1148 | 955| 841 | 786 | 720| 698 |
    +------------------------+----------------------+------+------+----+-----+-----+----+-----+
    | RMa / RMa O2I          | Short delay profile  | 32   | 32   | N/A| N/A | N/A | N/A| N/A |
    |                        +----------------------+------+------+----+-----+-----+----+-----+
    |                        | Normal delay profile | 37   | 37   | N/A| N/A | N/A | N/A| N/A |
    |                        +----------------------+------+------+----+-----+-----+----+-----+
    |                        | Long delay profile   | 153  | 153  | N/A| N/A | N/A | N/A| N/A |
    +------------------------+----------------------+------+------+----+-----+-----+----+-----+
    | UMi / UMa O2I          | Normal delay profile | 242                                     |
    |                        +----------------------+-----------------------------------------+
    |                        | Long delay profile   | 616                                     |
    +------------------------+----------------------+-----------------------------------------+
    """

    def __init__(
        self,
        model: str,
        delay_spread: float,
        carrier_frequency: float,
        ut_array=None,
        bs_array=None,
        direction: str = "downlink",
        ut_orientation: Optional[torch.Tensor] = None,
        bs_orientation: Optional[torch.Tensor] = None,
        ut_velocity: Optional[torch.Tensor] = None,
        min_speed: float = 0.0,
        max_speed: float = 0.0,
        normalize_delays: bool = True,
        precision: Optional[str] = None,
        device: Optional[str] = None,
        spec_version: str = "19.2",
    ) -> None:
        super().__init__(precision=precision, device=device)

        # Validate model
        if model not in ("A", "B", "C", "D", "E"):
            raise ValueError("Invalid CDL model")

        # Validate direction
        if direction not in ("uplink", "downlink"):
            raise ValueError("Invalid direction")

        self._model = model
        self._spec_version = models._validate_spec_version(spec_version)
        # Register as buffers for CUDAGraph compatibility
        self.register_buffer("_delay_spread", torch.tensor(
            delay_spread, dtype=self.dtype, device=self.device
        ))
        self.register_buffer("_carrier_frequency", torch.tensor(
            carrier_frequency, dtype=self.dtype, device=self.device
        ))
        self._direction = direction
        self._min_speed = min_speed
        self._max_speed = max_speed
        self._normalize_delays = normalize_delays

        # Wavelength (m)
        self.register_buffer("_lambda_0", torch.tensor(
            SPEED_OF_LIGHT / carrier_frequency, dtype=self.dtype, device=self.device
        ))

        # Set TX and RX arrays based on direction
        if direction == "downlink":
            self._tx_array = bs_array
            self._rx_array = ut_array
        else:  # uplink
            self._tx_array = ut_array
            self._rx_array = bs_array

        # Orientations
        # Default UT orientation is [π, 0, 0] to match TF implementation
        if ut_orientation is None:
            ut_orientation = torch.tensor([PI, 0.0, 0.0], dtype=self.dtype, device=self.device)
        else:
            ut_orientation = torch.as_tensor(
                ut_orientation, dtype=self.dtype, device=self.device
            )
        if bs_orientation is None:
            bs_orientation = torch.zeros(3, dtype=self.dtype, device=self.device)
        else:
            bs_orientation = torch.as_tensor(
                bs_orientation, dtype=self.dtype, device=self.device
            )

        self._ut_orientation = ut_orientation
        self._bs_orientation = bs_orientation

        # Velocity
        if ut_velocity is not None:
            self.register_buffer("_ut_velocity", torch.as_tensor(
                ut_velocity, dtype=self.dtype, device=self.device
            ))
        else:
            self.register_buffer("_ut_velocity", None)

        # Load CDL model parameters
        self._load_parameters()

        # Create the channel coefficients generator
        self._cir_gen = ChannelCoefficientsGenerator(
            carrier_frequency=carrier_frequency,
            tx_array=self._tx_array,
            rx_array=self._rx_array,
            subclustering=False,
            precision=precision,
            device=device,
        )

    def __call__(
        self,
        batch_size: int,
        num_time_steps: int,
        sampling_frequency: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate CDL channel impulse response"""
        # Generate random velocity if not provided
        velocity = self._get_velocity(batch_size)

        # Create topology
        topology = self._create_topology(batch_size, velocity)

        # Create rays from CDL model parameters
        rays = self._create_rays(batch_size)

        # K-factor (only for D and E models)
        k_factor = self._get_k_factor(batch_size)

        # Generate channel impulse response
        h, delays = self._cir_gen(
            num_time_samples=num_time_steps,
            sampling_frequency=sampling_frequency,
            k_factor=k_factor,
            rays=rays,
            topology=topology,
        )

        # Reshape output to match expected format
        # h: [batch, num_tx, num_rx, num_paths, num_rx_ant, num_tx_ant, num_time_steps]
        # -> [batch, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_steps]
        h = h.permute(0, 2, 4, 1, 5, 3, 6)

        # delays: [batch, num_tx, num_rx, num_paths]
        # -> [batch, num_rx, num_tx, num_paths]
        delays = delays.permute(0, 2, 1, 3)

        return h, delays

    @property
    def spec_version(self) -> str:
        """Version of the TR 38.901 parameter tables in use."""
        return self._spec_version

    ########################################
    # Internal utility methods
    ########################################

    def _load_parameters(self) -> None:
        r"""Load CDL model parameters from JSON files.

        The model parameters are stored as JSON files with the following keys:

        * ``los`` : boolean that indicates if the model is a LoS model
        * ``num_clusters`` : integer corresponding to the number of clusters
          (paths)
        * ``delays`` : List of path delays in ascending order normalized by
          the RMS delay spread
        * ``powers`` : List of path powers in dB scale
        * ``aod`` : Paths AoDs [degree]
        * ``aoa`` : Paths AoAs [degree]
        * ``zod`` : Paths ZoDs [degree]
        * ``zoa`` : Paths ZoAs [degree]
        * ``cASD`` : Cluster ASD
        * ``cASA`` : Cluster ASA
        * ``cZSD`` : Cluster ZSD
        * ``cZSA`` : Cluster ZSA
        * ``xpr`` : XPR in dB

        For LoS models, the two first paths have zero delay, and are assumed
        to correspond to the specular and NLoS component, in this order.
        """
        model = self._model
        delay_spread = self._delay_spread
        dtype = self.dtype
        device = self.device

        # Load JSON file for this model
        fname = f"CDL-{model}.json"
        source = models.parameter_file(fname, self.spec_version)
        params = models.load_json(source)

        # LoS scenario?
        self._has_los = bool(params["los"])

        # Number of clusters
        self._num_clusters = params["num_clusters"]
        self._rays_per_cluster = 20

        # Load delays and powers
        delays = torch.tensor(params["delays"], dtype=dtype, device=device)
        powers = torch.tensor(
            np.power(10.0, np.array(params["powers"]) / 10.0),
            dtype=dtype, device=device
        )

        # Normalize powers
        powers = powers / powers.sum()

        # Load angles (in degrees) and their cluster spreads
        aod = torch.tensor(params["aod"], dtype=dtype, device=device)
        aoa = torch.tensor(params["aoa"], dtype=dtype, device=device)
        zod = torch.tensor(params["zod"], dtype=dtype, device=device)
        zoa = torch.tensor(params["zoa"], dtype=dtype, device=device)

        # Load cluster angle spreads
        c_aod = torch.tensor(params["cASD"], dtype=dtype, device=device)
        c_aoa = torch.tensor(params["cASA"], dtype=dtype, device=device)
        c_zod = torch.tensor(params["cZSD"], dtype=dtype, device=device)
        c_zoa = torch.tensor(params["cZSA"], dtype=dtype, device=device)

        # Load XPR and convert from dB to linear
        xpr_db = params["xpr"]
        xpr = torch.tensor(np.power(10.0, xpr_db / 10.0), dtype=dtype, device=device)

        # If LoS, compute K-factor and extract LoS component
        if self._has_los:
            # Extract the specular component (first path)
            los_power = powers[0]
            powers = powers[1:]
            delays = delays[1:]
            los_aod = aod[0]
            aod = aod[1:]
            los_aoa = aoa[0]
            aoa = aoa[1:]
            los_zod = zod[0]
            zod = zod[1:]
            los_zoa = zoa[0]
            zoa = zoa[1:]

            # Re-normalize NLoS powers
            norm_fact = powers.sum()
            powers = powers / norm_fact

            # K-factor = (power of specular component) / (total NLoS power)
            # Register as buffer for CUDAGraph compatibility
            self.register_buffer("_k_factor", los_power / norm_fact)

            # Store LoS angles (in radians)
            self._los_aod = deg_2_rad(los_aod)
            self._los_aoa = deg_2_rad(los_aoa)
            self._los_zod = deg_2_rad(los_zod)
            self._los_zoa = deg_2_rad(los_zoa)

            # Note: num_clusters in JSON already excludes the LoS component,
            # so we don't need to decrement it
        else:
            # Register as buffer for CUDAGraph compatibility
            self.register_buffer("_k_factor", torch.tensor(0.0, dtype=dtype, device=device))

        # Scale delays if normalizing
        if self._normalize_delays:
            delays = delays * delay_spread

        # Generate cluster rays using equation 7.7-0a from TR38.901
        # This adds the per-ray offset angles from Table 7.5-3
        aod = self._generate_rays(aod, c_aod)  # [num_clusters, num_rays]
        aod = deg_2_rad(aod)
        aoa = self._generate_rays(aoa, c_aoa)  # [num_clusters, num_rays]
        aoa = deg_2_rad(aoa)
        zod = self._generate_rays(zod, c_zod)  # [num_clusters, num_rays]
        zod = deg_2_rad(zod)
        zoa = self._generate_rays(zoa, c_zoa)  # [num_clusters, num_rays]
        zoa = deg_2_rad(zoa)

        # Swap angles for uplink direction
        if self._direction == "uplink":
            aod, aoa = aoa, aod
            zod, zoa = zoa, zod
            if self._has_los:
                self._los_aod, self._los_aoa = self._los_aoa, self._los_aod
                self._los_zod, self._los_zoa = self._los_zoa, self._los_zod

        # Store parameters
        self._delays = delays
        self._powers = powers
        self._aod = aod
        self._aoa = aoa
        self._zod = zod
        self._zoa = zoa
        self._xpr = xpr

    def _get_velocity(self, batch_size: int) -> torch.Tensor:
        """Get UT velocity, either from a specified value or random sampling.

        :param batch_size: Batch size

        :output velocity: UT velocity [m/s], shape [batch_size, 3]
        """
        if self._ut_velocity is not None:
            velocity = self._ut_velocity
            if velocity.dim() == 1:
                velocity = velocity.unsqueeze(0).expand(batch_size, -1)
            return velocity

        v_r = (
            rand((batch_size, 1), dtype=self.dtype, device=self.device, generator=self.torch_rng)
            * (self._max_speed - self._min_speed) + self._min_speed
        )
        v_phi = (
            rand((batch_size, 1), dtype=self.dtype, device=self.device, generator=self.torch_rng)
            * 2.0 * PI
        )
        v_theta = (
            rand((batch_size, 1), dtype=self.dtype, device=self.device, generator=self.torch_rng)
            * PI
        )
        velocity = torch.cat(
            [
                v_r * torch.cos(v_phi) * torch.sin(v_theta),
                v_r * torch.sin(v_phi) * torch.sin(v_theta),
                v_r * torch.cos(v_theta),
            ],
            dim=-1,
        )
        return velocity

    def _create_topology(self, batch_size: int, velocity: torch.Tensor) -> Topology:
        """Create network topology for channel generation.

        :param batch_size: Batch size
        :param velocity: UT velocity [m/s], shape [batch_size, 3]

        :output topology: Network topology
        """
        # LoS angles (for LoS models D, E)
        if self._has_los:
            los_aoa = self._los_aoa.reshape(1, 1, 1)
            los_aod = self._los_aod.reshape(1, 1, 1)
            los_zoa = self._los_zoa.reshape(1, 1, 1)
            los_zod = self._los_zod.reshape(1, 1, 1)
        else:
            los_aoa = torch.zeros(1, 1, 1, dtype=self.dtype, device=self.device)
            los_aod = torch.zeros(1, 1, 1, dtype=self.dtype, device=self.device)
            los_zoa = torch.zeros(1, 1, 1, dtype=self.dtype, device=self.device)
            los_zod = torch.zeros(1, 1, 1, dtype=self.dtype, device=self.device)

        los_aoa = los_aoa.expand(batch_size, 1, 1)
        los_aod = los_aod.expand(batch_size, 1, 1)
        los_zoa = los_zoa.expand(batch_size, 1, 1)
        los_zod = los_zod.expand(batch_size, 1, 1)

        los = torch.full(
            (batch_size, 1, 1), self._has_los, dtype=torch.bool, device=self.device
        )

        # Distance (used for LoS phase computation)
        distance_3d = torch.zeros(
            (batch_size, 1, 1), dtype=self.dtype, device=self.device
        )

        # Orientations
        ut_orientation = self._ut_orientation
        if ut_orientation.dim() == 1:
            ut_orientation = ut_orientation.unsqueeze(0).expand(batch_size, -1)
        ut_orientation = ut_orientation.unsqueeze(1)  # Add UT dimension

        bs_orientation = self._bs_orientation
        if bs_orientation.dim() == 1:
            bs_orientation = bs_orientation.unsqueeze(0).expand(batch_size, -1)
        bs_orientation = bs_orientation.unsqueeze(1)  # Add BS dimension

        # Set TX and RX orientations based on direction
        if self._direction == "downlink":
            tx_orientations = bs_orientation
            rx_orientations = ut_orientation
        else:
            tx_orientations = ut_orientation
            rx_orientations = bs_orientation

        # Moving end
        if self._direction == "downlink":
            moving_end = "rx"
            velocities = velocity.unsqueeze(1)  # Add RX dimension
        else:
            moving_end = "tx"
            velocities = velocity.unsqueeze(1)  # Add TX dimension

        topology = Topology(
            velocities=velocities,
            moving_end=moving_end,
            los_aoa=los_aoa,
            los_aod=los_aod,
            los_zoa=los_zoa,
            los_zod=los_zod,
            los=los,
            distance_3d=distance_3d,
            tx_orientations=tx_orientations,
            rx_orientations=rx_orientations,
        )

        return topology

    def _create_rays(self, batch_size: int) -> Rays:
        """Create rays from CDL model parameters.

        :param batch_size: Batch size

        :output rays: Rays
        """
        num_clusters = self._num_clusters
        rays_per_cluster = self._rays_per_cluster

        # Expand to batch and BS/UT dimensions
        # Shape: [batch, num_tx=1, num_rx=1, num_clusters]
        delays = self._delays.reshape(1, 1, 1, num_clusters).expand(batch_size, 1, 1, -1)
        powers = self._powers.reshape(1, 1, 1, num_clusters).expand(batch_size, 1, 1, -1)

        # Angles are already [num_clusters, rays_per_cluster] after _generate_rays
        # Shape: [batch, num_tx=1, num_rx=1, num_clusters, rays_per_cluster]
        aod = self._aod.reshape(1, 1, 1, num_clusters, rays_per_cluster).expand(
            batch_size, 1, 1, -1, -1
        )
        aoa = self._aoa.reshape(1, 1, 1, num_clusters, rays_per_cluster).expand(
            batch_size, 1, 1, -1, -1
        )
        zod = self._zod.reshape(1, 1, 1, num_clusters, rays_per_cluster).expand(
            batch_size, 1, 1, -1, -1
        )
        zoa = self._zoa.reshape(1, 1, 1, num_clusters, rays_per_cluster).expand(
            batch_size, 1, 1, -1, -1
        )

        # Apply random coupling (Step 8 of TR38.901)
        aoa, aod, zoa, zod = self._random_coupling(aoa, aod, zoa, zod)

        # XPR: [batch, num_tx=1, num_rx=1, num_clusters, rays_per_cluster]
        xpr = self._xpr.reshape(1, 1, 1, 1, 1).expand(
            batch_size, 1, 1, num_clusters, rays_per_cluster
        )

        rays = Rays(
            delays=delays,
            powers=powers,
            aoa=aoa,
            aod=aod,
            zoa=zoa,
            zod=zod,
            xpr=xpr,
        )

        return rays

    def _get_k_factor(self, batch_size: int) -> torch.Tensor:
        """Get K-factor for the model.

        :param batch_size: Batch size

        :output k_factor: K-factor, shape [batch_size, 1, 1]
        """
        k_factor = self._k_factor.reshape(1, 1, 1).expand(batch_size, 1, 1)
        return k_factor

    def _generate_rays(
        self, angles: torch.Tensor, c: torch.Tensor
    ) -> torch.Tensor:
        """Generate rays using equation 7.7-0a of TR38.901 specifications.

        :param angles: Cluster angles [num_clusters]
        :param c: Cluster angle spread

        :output ray_angles: Ray angles [num_clusters, num_rays=20]
        """
        # Basis vector of offset angles from Table 7.5-3 of TR38.901
        basis_vector = torch.tensor(
            [
                0.0447, -0.0447,
                0.1413, -0.1413,
                0.2492, -0.2492,
                0.3715, -0.3715,
                0.5129, -0.5129,
                0.6797, -0.6797,
                0.8844, -0.8844,
                1.1481, -1.1481,
                1.5195, -1.5195,
                2.1551, -2.1551,
            ],
            dtype=self.dtype,
            device=self.device,
        )

        # Reshape for broadcasting
        # basis_vector: [1, num_rays=20]
        basis_vector = basis_vector.unsqueeze(0)
        # angles: [num_clusters, 1]
        angles = angles.unsqueeze(1)

        # Generate rays following equation 7.7-0a
        # ray_angles: [num_clusters, num_rays=20]
        ray_angles = angles + c * basis_vector

        return ray_angles

    def _shuffle_angles(
        self,
        angles: torch.Tensor,
    ) -> torch.Tensor:
        """Randomly shuffle angles of arrival/departure within each cluster.

        :param angles: Angles to shuffle,
            [batch_size, num_tx, num_rx, num_clusters, num_rays]

        :output shuffled_angles: Shuffled angles with the same shape
        """
        # Create randomly shuffled indices by arg-sorting samples from a random
        # normal distribution
        random_numbers = normal(
            angles.shape, dtype=self.dtype, device=self.device,
            generator=self.torch_rng
        )
        shuffled_indices = torch.argsort(random_numbers, dim=-1)

        # Shuffling the angles using gather
        shuffled_angles = torch.gather(angles, dim=-1, index=shuffled_indices)
        return shuffled_angles

    def _random_coupling(
        self,
        aoa: torch.Tensor,
        aod: torch.Tensor,
        zoa: torch.Tensor,
        zod: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Randomly couple angles within a cluster (Step 8 in TR38.901).

        :param aoa: Azimuth angles of arrival (AoA),
            [batch_size, num_tx, num_rx, num_clusters, num_rays]
        :param aod: Azimuth angles of departure (AoD),
            [batch_size, num_tx, num_rx, num_clusters, num_rays]
        :param zoa: Zenith angles of arrival (ZoA),
            [batch_size, num_tx, num_rx, num_clusters, num_rays]
        :param zod: Zenith angles of departure (ZoD),
            [batch_size, num_tx, num_rx, num_clusters, num_rays]

        :output shuffled_aoa: Shuffled azimuth angles of arrival.

        :output shuffled_aod: Shuffled azimuth angles of departure.

        :output shuffled_zoa: Shuffled zenith angles of arrival.

        :output shuffled_zod: Shuffled zenith angles of departure.
        """
        shuffled_aoa = self._shuffle_angles(aoa)
        shuffled_aod = self._shuffle_angles(aod)
        shuffled_zoa = self._shuffle_angles(zoa)
        shuffled_zod = self._shuffle_angles(zod)

        return shuffled_aoa, shuffled_aod, shuffled_zoa, shuffled_zod
