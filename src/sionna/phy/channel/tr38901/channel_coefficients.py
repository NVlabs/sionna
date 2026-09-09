#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""
Class for sampling channel impulse responses following 3GPP TR38.901
specifications and giving LSPs and rays
"""

from typing import Optional, Tuple

import torch

from sionna.phy import PI, SPEED_OF_LIGHT
from sionna.phy.object import Object
from sionna.phy.utils import rand
from .rays import Rays

__all__ = ["Topology", "ChannelCoefficientsGenerator"]


class Topology:
    r"""
    Class for conveniently storing the network topology information required
    for sampling channel impulse responses

    :param velocities: UT velocities,
        shape [batch size, number of UTs, 3], `torch.float`
    :param moving_end: Indicated which end of the channel (TX or RX) is
        moving. One of ``"tx"`` or ``"rx"``.
    :param los_aoa: Azimuth angle of arrival of LoS path [radian],
        shape [batch size, number of base stations, number of UTs], `torch.float`
    :param los_aod: Azimuth angle of departure of LoS path [radian],
        shape [batch size, number of base stations, number of UTs], `torch.float`
    :param los_zoa: Zenith angle of arrival for LoS path [radian],
        shape [batch size, number of base stations, number of UTs], `torch.float`
    :param los_zod: Zenith angle of departure for LoS path [radian],
        shape [batch size, number of base stations, number of UTs], `torch.float`
    :param los: Indicate for each BS-UT link if it is in LoS,
        shape [batch size, number of base stations, number of UTs], `torch.bool`
    :param distance_3d: Distance between the UTs in X-Y-Z space
        (not only X-Y plane),
        shape [batch size, number of base stations, number of UTs], `torch.float`
    :param tx_orientations: Orientations of the transmitters, which are
        either base stations or UTs depending on the link direction [radian],
        shape [batch size, number of TXs, 3], `torch.float`
    :param rx_orientations: Orientations of the receivers, which are
        either base stations or UTs depending on the link direction [radian],
        shape [batch size, number of RXs, 3], `torch.float`
    """

    def __init__(
        self,
        velocities: torch.Tensor,
        moving_end: str,
        los_aoa: torch.Tensor,
        los_aod: torch.Tensor,
        los_zoa: torch.Tensor,
        los_zod: torch.Tensor,
        los: torch.Tensor,
        distance_3d: torch.Tensor,
        tx_orientations: torch.Tensor,
        rx_orientations: torch.Tensor,
    ) -> None:
        self.velocities = velocities
        self.moving_end = moving_end
        self.los_aoa = los_aoa
        self.los_aod = los_aod
        self.los_zoa = los_zoa
        self.los_zod = los_zod
        self.los = los
        self.tx_orientations = tx_orientations
        self.rx_orientations = rx_orientations
        self.distance_3d = distance_3d


class ChannelCoefficientsGenerator(Object):
    r"""
    Sample channel impulse responses according to LSPs rays

    This class implements steps 10 and 11 from the TR 38.901 specifications,
    (section 7.5).

    :param carrier_frequency: Carrier frequency [Hz]
    :param tx_array: Array used by the transmitters.
        All transmitters share the same antenna array configuration.
    :param rx_array: Panel array used by the receivers.
        All receivers share the same antenna array configuration.
    :param subclustering: Use subclustering if set to `True` (see step 11
        for section 7.5 in TR 38.901). CDL does not use subclustering.
        System level models (UMa, UMi, RMa) do.
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`,
        :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for computation (e.g., ``"cpu"``, ``"cuda:0"``).
        If `None`, :attr:`~sionna.phy.config.Config.device` is used.

    :input num_time_samples: `int`.
        Number of samples.

    :input sampling_frequency: `float`.
        Sampling frequency [Hz].

    :input k_factor: [batch_size, number of TX, number of RX], `torch.float`.
        K-factor.

    :input rays: :class:`~sionna.phy.channel.tr38901.Rays`.
        Rays from which to compute the CIR.

    :input topology: :class:`~sionna.phy.channel.tr38901.Topology`.
        Topology of the network.

    :input c_ds: [batch size, number of TX, number of RX], `torch.float`.
        Cluster DS [ns]. Only needed when subclustering is used
        (``subclustering`` set to `True`), i.e., with system level models.
        Otherwise can be set to `None`.
        Defaults to `None`.

    :input debug: `bool`.
        If set to `True`, additional information is returned in addition to
        paths coefficients and delays: The random phase shifts (see step 10 of
        section 7.5 in TR38.901 specification), and the time steps at which the
        channel is sampled.

    :output h: [batch size, num TX, num RX, num paths, num RX antenna, num TX antenna, num samples], `torch.complex`.
        Paths coefficients.

    :output delays: [batch size, num TX, num RX, num paths], `torch.float`.
        Paths delays [s].

    :output phi: [batch size, number of base stations, number of UTs, 4], `torch.float`.
        Initial phases (see step 10 of section 7.5 in TR 38.901 specification).
        Last dimension corresponds to the four polarization combinations.

    :output sample_times: [number of time steps], `torch.float`.
        Sampling time steps.

    .. rubric:: Examples

    .. code-block:: python

        # Create the generator
        cir_gen = ChannelCoefficientsGenerator(
            carrier_frequency=3.5e9,
            tx_array=tx_array,
            rx_array=rx_array,
            subclustering=False,
        )

        # Generate channel impulse responses
        h, delays = cir_gen(
            num_time_samples=100,
            sampling_frequency=1e6,
            k_factor=k_factor,
            rays=rays,
            topology=topology,
        )
    """

    def __init__(
        self,
        carrier_frequency: float,
        tx_array,
        rx_array,
        subclustering: bool,
        precision: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        super().__init__(precision=precision, device=device)

        # Wavelength (m)
        self._lambda_0 = torch.as_tensor(
            SPEED_OF_LIGHT / carrier_frequency, dtype=self.dtype, device=self.device
        )
        self._tx_array = tx_array
        self._rx_array = rx_array
        self._subclustering = subclustering
        self._tx_array_has_element_field = callable(
            getattr(tx_array, "element_field", None)
        )
        self._rx_array_has_element_field = callable(
            getattr(rx_array, "element_field", None)
        )

        # Sub-cluster information for intra cluster delay spread clusters
        # This is hardcoded from Table 7.5-5
        # Register as buffers for CUDAGraph compatibility
        self.register_buffer("_sub_cl_1_ind", torch.tensor(
            [0, 1, 2, 3, 4, 5, 6, 7, 18, 19], dtype=torch.int64, device=self.device
        ))
        self.register_buffer("_sub_cl_2_ind", torch.tensor(
            [8, 9, 10, 11, 16, 17], dtype=torch.int64, device=self.device
        ))
        self.register_buffer("_sub_cl_3_ind", torch.tensor(
            [12, 13, 14, 15], dtype=torch.int64, device=self.device
        ))
        self.register_buffer("_sub_cl_delay_offsets", torch.tensor(
            [0, 1.28, 2.56], dtype=self.dtype, device=self.device
        ))

        # Pre-compute complex constant tensors for torch.compile compatibility
        # These cannot be created inside compiled functions with complex dtype
        # from Python lists - see https://github.com/pytorch/pytorch/issues/
        self.register_buffer("_v2_const", torch.tensor(
            [1 + 0j, 1j, 0], dtype=self.cdtype, device=self.device
        ))
        self.register_buffer("_h_phase_los_const", torch.tensor(
            [[1.0, 0.0], [0.0, -1.0]], dtype=self.cdtype, device=self.device
        ))

        # Pre-compute v1 constant for _gcs_to_lcs
        self.register_buffer("_v1_const", torch.tensor(
            [0, 0, 1], dtype=self.dtype, device=self.device
        ))

        # Pre-compute gather indices for antenna polarization (CUDA graph compatible)
        num_ant_tx = tx_array.num_ant
        if tx_array.polarization == "dual":
            gather_ind_tx = torch.zeros(num_ant_tx, dtype=torch.int64, device=self.device)
            gather_ind_tx[tx_array.ant_ind_pol2] = 1
            self.register_buffer("_gather_ind_tx", gather_ind_tx)
        else:
            self.register_buffer("_gather_ind_tx", None)

        num_ant_rx = rx_array.num_ant
        if rx_array.polarization == "dual":
            gather_ind_rx = torch.zeros(num_ant_rx, dtype=torch.int64, device=self.device)
            gather_ind_rx[rx_array.ant_ind_pol2] = 1
            self.register_buffer("_gather_ind_rx", gather_ind_rx)
        else:
            self.register_buffer("_gather_ind_rx", None)

    def __call__(
        self,
        num_time_samples: int,
        sampling_frequency: float,
        k_factor: torch.Tensor,
        rays: Rays,
        topology: Topology,
        c_ds: Optional[torch.Tensor] = None,
        debug: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate channel impulse responses."""
        sample_times = torch.arange(
            num_time_samples, dtype=self.dtype, device=self.device
        ) / sampling_frequency

        # Step 10
        if getattr(rays, "phases", None) is None:
            phi = self._step_10(rays.aoa.shape)
        else:
            phi = rays.phases.to(dtype=self.dtype, device=self.device)

        # Step 11
        h, delays = self._step_11(phi, topology, k_factor, rays, sample_times, c_ds)

        # Return additional information if requested
        if debug:
            return h, delays, phi, sample_times

        return h, delays

    ###########################################
    # Utility functions
    ###########################################

    def _unit_sphere_vector(
        self, theta: torch.Tensor, phi: torch.Tensor
    ) -> torch.Tensor:
        r"""
        Generate vector on unit sphere (7.1-6)

        :param theta: Zenith [radian], arbitrary shape
        :param phi: Azimuth [radian], same shape as ``theta``

        :output rho_hat: Vector on unit sphere, shape ``phi.shape`` + [3, 1]
        """
        rho_hat = torch.stack(
            [
                torch.sin(theta) * torch.cos(phi),
                torch.sin(theta) * torch.sin(phi),
                torch.cos(theta),
            ],
            dim=-1,
        )
        return rho_hat.unsqueeze(-1)

    def _forward_rotation_matrix(self, orientations: torch.Tensor) -> torch.Tensor:
        r"""
        Forward composite rotation matrix (7.1-4)

        :param orientations: Orientation to which to rotate [radian], shape [...,3]

        :output rot_mat: Rotation matrix, shape [...,3,3]
        """
        a, b, c = orientations[..., 0], orientations[..., 1], orientations[..., 2]

        row_1 = torch.stack(
            [
                torch.cos(a) * torch.cos(b),
                torch.cos(a) * torch.sin(b) * torch.sin(c) - torch.sin(a) * torch.cos(c),
                torch.cos(a) * torch.sin(b) * torch.cos(c) + torch.sin(a) * torch.sin(c),
            ],
            dim=-1,
        )

        row_2 = torch.stack(
            [
                torch.sin(a) * torch.cos(b),
                torch.sin(a) * torch.sin(b) * torch.sin(c) + torch.cos(a) * torch.cos(c),
                torch.sin(a) * torch.sin(b) * torch.cos(c) - torch.cos(a) * torch.sin(c),
            ],
            dim=-1,
        )

        row_3 = torch.stack(
            [
                -torch.sin(b),
                torch.cos(b) * torch.sin(c),
                torch.cos(b) * torch.cos(c),
            ],
            dim=-1,
        )

        rot_mat = torch.stack([row_1, row_2, row_3], dim=-2)
        return rot_mat

    def _rot_pos(
        self, orientations: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        r"""
        Rotate the ``positions`` according to the ``orientations``

        :param orientations: Orientation to which to rotate [radian], shape [...,3]
        :param positions: Positions to rotate, shape [...,3,1]

        :output positions_rotated: Rotated positions, shape [...,3,1]
        """
        rot_mat = self._forward_rotation_matrix(orientations)
        return torch.matmul(rot_mat, positions)

    def _reverse_rotation_matrix(self, orientations: torch.Tensor) -> torch.Tensor:
        r"""
        Reverse composite rotation matrix (7.1-4)

        :param orientations: Orientations to rotate to [radian], shape [...,3]

        :output rot_mat_inv: Inverse of the rotation matrix corresponding to ``orientations``,
            shape [...,3,3]
        """
        rot_mat = self._forward_rotation_matrix(orientations)
        rot_mat_inv = rot_mat.mT
        return rot_mat_inv

    def _gcs_to_lcs(
        self, orientations: torch.Tensor, theta: torch.Tensor, phi: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Compute the angles ``theta``, ``phi`` in LCS rotated according to
        ``orientations`` (7.1-7/8)

        :param orientations: Orientations to which to rotate to [radian],
            shape [...,3] of rank K
        :param theta: Zenith to rotate [radian], broadcastable to the first K-1
            dimensions of ``orientations``
        :param phi: Azimuth to rotate [radian], same dimension as ``theta``

        :output theta_prime: Rotated zenith.

        :output phi_prime: Rotated azimuth.
        """
        rho_hat = self._unit_sphere_vector(theta, phi)
        rot_inv = self._reverse_rotation_matrix(orientations)
        rot_rho = torch.matmul(rot_inv, rho_hat)

        # Use pre-allocated constant (CUDA graph compatible)
        v1 = self._v1_const.reshape([1] * (rot_rho.dim() - 1) + [3])
        v2 = self._v2_const.reshape([1] * (rot_rho.dim() - 1) + [3])

        z = torch.matmul(v1, rot_rho)
        z = z.clamp(-1.0, 1.0)
        theta_prime = torch.acos(z)
        phi_prime = torch.angle(torch.matmul(v2, rot_rho.to(self.cdtype)))
        theta_prime = theta_prime.squeeze(-1).squeeze(-1)
        phi_prime = phi_prime.squeeze(-1).squeeze(-1)

        return theta_prime, phi_prime

    def _compute_psi(
        self, orientations: torch.Tensor, theta: torch.Tensor, phi: torch.Tensor
    ) -> torch.Tensor:
        r"""
        Compute displacement angle :math:`Psi` for the transformation of LCS-GCS
        field components in (7.1-15) of TR38.901 specification

        :param orientations: Orientations to which to rotate to [radian], shape [...,3]
        :param theta: Spherical position zenith [radian], broadcastable to the
            first K-1 dimensions of ``orientations``
        :param phi: Spherical position azimuth [radian], same dimensions as ``theta``

        :output psi: Displacement angle :math:`Psi`, same shape as ``theta`` and ``phi``
        """
        a = orientations[..., 0]
        b = orientations[..., 1]
        c = orientations[..., 2]
        real = torch.sin(c) * torch.cos(theta) * torch.sin(phi - a)
        real = real + torch.cos(c) * (
            torch.cos(b) * torch.sin(theta) - torch.sin(b) * torch.cos(theta) * torch.cos(phi - a)
        )
        imag = torch.sin(c) * torch.cos(phi - a) + torch.sin(b) * torch.cos(c) * torch.sin(phi - a)
        psi = torch.angle(torch.complex(real, imag))
        return psi

    def _l2g_response(
        self,
        f_prime: torch.Tensor,
        orientations: torch.Tensor,
        theta: torch.Tensor,
        phi: torch.Tensor,
    ) -> torch.Tensor:
        r"""
        Transform field components from LCS to GCS (7.1-11)

        :param f_prime: Field components, shape [...,2]
        :param orientations: Orientations of LCS-GCS [radian], shape [...,3]
        :param theta: Spherical position zenith [radian]
        :param phi: Spherical position azimuth [radian]

        :output f: Transformed field components, shape [...,2,1]
        """
        psi = self._compute_psi(orientations, theta, phi)
        row1 = torch.stack([torch.cos(psi), -torch.sin(psi)], dim=-1)
        row2 = torch.stack([torch.sin(psi), torch.cos(psi)], dim=-1)
        mat = torch.stack([row1, row2], dim=-2)
        f = torch.matmul(mat, f_prime.unsqueeze(-1))
        return f

    def _step_11_get_tx_antenna_positions(self, topology: Topology) -> torch.Tensor:
        r"""
        Compute d_bar_tx in (7.5-22), i.e., the positions in GCS of elements
        forming the transmit panel

        :param topology: Topology of the network

        :output d_bar_tx: Positions of the antenna elements in the GCS,
            shape [batch_size, num TXs, num TX antenna, 3]
        """
        # Get BS orientations for broadcasting
        tx_orientations = topology.tx_orientations.unsqueeze(2)

        # Get antenna element positions in LCS and reshape for broadcasting
        tx_ant_pos_lcs = self._tx_array.ant_pos
        tx_ant_pos_lcs = tx_ant_pos_lcs.reshape(1, 1, -1, 3, 1)

        # Compute antenna element positions in GCS
        tx_ant_pos_gcs = self._rot_pos(tx_orientations, tx_ant_pos_lcs)
        tx_ant_pos_gcs = tx_ant_pos_gcs.squeeze(-1)

        d_bar_tx = tx_ant_pos_gcs

        return d_bar_tx

    def _step_11_get_rx_antenna_positions(self, topology: Topology) -> torch.Tensor:
        r"""
        Compute d_bar_rx in (7.5-22), i.e., the positions in GCS of elements
        forming the receive antenna panel

        :param topology: Topology of the network

        :output d_bar_rx: Positions of the antenna elements in the GCS,
            shape [batch_size, num RXs, num RX antenna, 3]
        """
        # Get UT orientations for broadcasting
        rx_orientations = topology.rx_orientations.unsqueeze(2)

        # Get antenna element positions in LCS and reshape for broadcasting
        rx_ant_pos_lcs = self._rx_array.ant_pos
        rx_ant_pos_lcs = rx_ant_pos_lcs.reshape(1, 1, -1, 3, 1)

        # Compute antenna element positions in GCS
        rx_ant_pos_gcs = self._rot_pos(rx_orientations, rx_ant_pos_lcs)
        rx_ant_pos_gcs = rx_ant_pos_gcs.squeeze(-1)

        d_bar_rx = rx_ant_pos_gcs

        return d_bar_rx

    def _step_10(self, shape: torch.Size) -> torch.Tensor:
        r"""
        Generate random and uniformly distributed phases for all rays and
        polarization combinations

        :param shape: Shape of the leading dimensions for the tensor of phases to generate
        :output phi: Phases for all polarization combinations, shape [shape] + [4]
        """
        phi = (
            rand(
                (*shape, 4),
                dtype=self.dtype,
                device=self.device,
                generator=self.torch_rng,
            )
            * 2
            * PI
            - PI
        )

        return phi

    def _step_11_phase_matrix(
        self, phi: torch.Tensor, rays: Rays
    ) -> torch.Tensor:
        r"""
        Compute matrix with random phases in (7.5-22)

        :param phi: Initial phases for all combinations of polarization,
            shape [batch size, num TXs, num RXs, num clusters, num rays, 4]
        :param rays: Rays

        :output h_phase: Matrix with random phases in (7.5-22),
            shape [batch size, num TXs, num RXs, num clusters, num rays, 2, 2]
        """
        xpr = rays.xpr

        xpr_scaling = torch.complex(
            torch.sqrt(1 / xpr),
            torch.zeros_like(xpr),
        )
        e0 = torch.exp(torch.complex(torch.zeros_like(phi[..., 0]), phi[..., 0]))
        e3 = torch.exp(torch.complex(torch.zeros_like(phi[..., 3]), phi[..., 3]))
        e1 = xpr_scaling * torch.exp(
            torch.complex(torch.zeros_like(phi[..., 1]), phi[..., 1])
        )
        e2 = xpr_scaling * torch.exp(
            torch.complex(torch.zeros_like(phi[..., 2]), phi[..., 2])
        )
        h_phase = torch.stack([e0, e1, e2, e3], dim=-1).reshape(*e0.shape, 2, 2)

        return h_phase

    def _step_11_doppler_matrix(
        self,
        topology: Topology,
        aoa: torch.Tensor,
        aod: torch.Tensor,
        zoa: torch.Tensor,
        zod: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        r"""
        Compute matrix with phase shifts due to mobility in (7.5-22)

        :param topology: Topology of the network
        :param aoa: Azimuth angles of arrivals [radian],
            shape [batch size, num TXs, num RXs, num clusters, num rays]
        :param aod: Azimuth angles of departure [radian],
            shape [batch size, num TXs, num RXs, num clusters, num rays]
        :param zoa: Zenith angles of arrivals [radian],
            shape [batch size, num TXs, num RXs, num clusters, num rays]
        :param zod: Zenith angles of departure [radian],
            shape [batch size, num TXs, num RXs, num clusters, num rays]
        :param t: Time steps at which the channel is sampled, shape [number of time steps]

        :output h_doppler: Matrix with phase shifts due to mobility in (7.5-22),
            shape [batch size, num_tx, num rx, num clusters, num rays, num time steps]
        """
        lambda_0 = self._lambda_0
        velocities = topology.velocities

        # Add an extra dimension to make v_bar broadcastable with the time
        # dimension
        # v_bar [batch size, num tx or num rx, 3, 1]
        v_bar = velocities.unsqueeze(-1)

        # Depending on which end of the channel is moving, tx or rx, we add an
        # extra dimension to make this tensor broadcastable with the other end
        if topology.moving_end == "rx":
            # v_bar [batch size, 1, num rx, 3, 1]
            v_bar = v_bar.unsqueeze(1)
        elif topology.moving_end == "tx":
            # v_bar [batch size, num tx, 1, 3, 1]
            v_bar = v_bar.unsqueeze(2)

        # v_bar [batch size, 1, num rx, 1, 1, 3, 1]
        # or    [batch size, num tx, 1, 1, 1, 3, 1]
        v_bar = v_bar.unsqueeze(-3).unsqueeze(-3)

        if topology.moving_end == "rx":
            direction = self._unit_sphere_vector(zoa, aoa)
        else:
            direction = self._unit_sphere_vector(zod, aod)

        # Compute phase shift due to doppler
        # [batch size, num_tx, num rx, num clusters, num rays, num time steps]
        exponent = 2 * PI / lambda_0 * (direction * v_bar).sum(dim=-2) * t
        h_doppler = torch.exp(torch.complex(torch.zeros_like(exponent), exponent))

        # [batch size, num_tx, num rx, num clusters, num rays, num time steps]
        return h_doppler

    def _step_11_array_offsets(
        self,
        topology: Topology,
        aoa: torch.Tensor,
        aod: torch.Tensor,
        zoa: torch.Tensor,
        zod: torch.Tensor,
    ) -> torch.Tensor:
        r"""
        Compute matrix accounting for phases offsets between antenna elements

        :param topology: Topology of the network
        :param aoa: Azimuth angles of arrivals [radian],
            shape [batch size, num TXs, num RXs, num clusters, num rays]
        :param aod: Azimuth angles of departure [radian],
            shape [batch size, num TXs, num RXs, num clusters, num rays]
        :param zoa: Zenith angles of arrivals [radian],
            shape [batch size, num TXs, num RXs, num clusters, num rays]
        :param zod: Zenith angles of departure [radian],
            shape [batch size, num TXs, num RXs, num clusters, num rays]

        :output h_array: Matrix accounting for phases offsets between antenna elements,
            shape [batch size, num_tx, num rx, num clusters, num rays, num rx antennas, num tx antennas]
        """
        lambda_0 = self._lambda_0

        r_hat_rx = self._unit_sphere_vector(zoa, aoa).squeeze(-1)
        r_hat_tx = self._unit_sphere_vector(zod, aod).squeeze(-1)
        d_bar_rx = self._step_11_get_rx_antenna_positions(topology)
        d_bar_tx = self._step_11_get_tx_antenna_positions(topology)

        # Reshape tensors for broadcasting
        # r_hat_rx/tx have
        # shape [batch_size, num_tx, num_rx, num_clusters, num_rays, 3]
        # and will be reshaped to
        # [batch_size, num_tx, num_rx, num_clusters, num_rays, 1, 3]
        r_hat_tx = r_hat_tx.unsqueeze(-2)
        r_hat_rx = r_hat_rx.unsqueeze(-2)

        # d_bar_tx has shape [batch_size, num_tx, num_tx_antennas, 3]
        # and will be reshaped to
        # [batch_size, num_tx, 1, 1, 1, num_tx_antennas, 3]
        d_bar_tx = d_bar_tx.unsqueeze(2).unsqueeze(3).unsqueeze(4)

        # d_bar_rx has shape [batch_size, num_rx, num_rx_antennas, 3]
        # and will be reshaped to
        # [batch_size, 1, num_rx, 1, 1, num_rx_antennas, 3]
        d_bar_rx = d_bar_rx.unsqueeze(1).unsqueeze(3).unsqueeze(4)

        # Compute all tensor elements
        # Explicitly broadcast for high-rank tensors
        d_bar_rx = d_bar_rx.expand(-1, r_hat_rx.shape[1], -1, -1, -1, -1, -1)

        exp_rx = 2 * PI / lambda_0 * (r_hat_rx * d_bar_rx).sum(dim=-1, keepdim=True)
        exp_rx = torch.exp(torch.complex(torch.zeros_like(exp_rx), exp_rx))

        exp_tx = 2 * PI / lambda_0 * (r_hat_tx * d_bar_tx).sum(dim=-1)
        exp_tx = torch.exp(torch.complex(torch.zeros_like(exp_tx), exp_tx))
        exp_tx = exp_tx.unsqueeze(-2)

        h_array = exp_rx * exp_tx

        return h_array

    def _step_11_field_matrix(
        self,
        topology: Topology,
        aoa: torch.Tensor,
        aod: torch.Tensor,
        zoa: torch.Tensor,
        zod: torch.Tensor,
        h_phase: torch.Tensor,
    ) -> torch.Tensor:
        r"""
        Compute matrix accounting for the element responses, random phases
        and xpr

        :param topology: Topology of the network
        :param aoa: Azimuth angles of arrivals [radian],
            shape [batch size, num TXs, num RXs, num clusters, num rays]
        :param aod: Azimuth angles of departure [radian],
            shape [batch size, num TXs, num RXs, num clusters, num rays]
        :param zoa: Zenith angles of arrivals [radian],
            shape [batch size, num TXs, num RXs, num clusters, num rays]
        :param zod: Zenith angles of departure [radian],
            shape [batch size, num TXs, num RXs, num clusters, num rays]
        :param h_phase: Matrix with phase shifts due to mobility in (7.5-22),
            shape [batch size, num_tx, num rx, num clusters, num rays, 2, 2]

        :output h_field: Matrix accounting for element responses, random phases and xpr,
            shape [batch size, num_tx, num rx, num clusters, num rays, num rx antennas, num tx antennas]
        """
        tx_orientations = topology.tx_orientations
        rx_orientations = topology.rx_orientations

        # Transform departure angles to the LCS
        tx_orientations = tx_orientations.reshape(
            tx_orientations.shape[0],
            tx_orientations.shape[1],
            1,
            1,
            1,
            tx_orientations.shape[-1],
        )
        zod_prime, aod_prime = self._gcs_to_lcs(tx_orientations, zod, aod)

        # Transform arrival angles to the LCS
        rx_orientations = rx_orientations.reshape(
            rx_orientations.shape[0],
            1,
            rx_orientations.shape[1],
            1,
            1,
            rx_orientations.shape[-1],
        )
        zoa_prime, aoa_prime = self._gcs_to_lcs(rx_orientations, zoa, aoa)

        # Compute transmitted and received field strength for all antennas
        # in the LCS and convert to GCS. Standard panel arrays use one shared
        # element pattern per polarization. Handheld arrays can provide
        # per-port field vectors through element_field().
        if self._tx_array_has_element_field:
            f_tx_prime = self._tx_array.element_field(zod_prime, aod_prime)
            f_tx = self._l2g_response(f_tx_prime, tx_orientations, zod, aod)
            f_tx = torch.complex(f_tx, torch.zeros_like(f_tx))
            f_tx_array = torch.matmul(h_phase, f_tx)
        else:
            f_tx_pol1_prime = torch.stack(
                self._tx_array.ant_pol1.field(zod_prime, aod_prime), dim=-1
            )
            f_tx_pol1 = self._l2g_response(
                f_tx_pol1_prime, tx_orientations, zod, aod
            )
            pol1_tx = torch.matmul(
                h_phase, torch.complex(f_tx_pol1, torch.zeros_like(f_tx_pol1))
            )

            if self._tx_array.polarization == "dual":
                f_tx_pol2_prime = torch.stack(
                    self._tx_array.ant_pol2.field(zod_prime, aod_prime), dim=-1
                )
                f_tx_pol2 = self._l2g_response(
                    f_tx_pol2_prime, tx_orientations, zod, aod
                )
                pol2_tx = torch.matmul(
                    h_phase, torch.complex(f_tx_pol2, torch.zeros_like(f_tx_pol2))
                )

            num_ant_tx = self._tx_array.num_ant
            if self._tx_array.polarization == "single":
                # Each BS antenna gets the polarization 1 response
                f_tx_array = pol1_tx.unsqueeze(0).expand(num_ant_tx,
                                                         *pol1_tx.shape)
            else:
                # Assign polarization response according to polarization to
                # each antenna using pre-computed gather indices (CUDA graph
                # compatible)
                pol_tx = torch.stack([pol1_tx, pol2_tx], dim=0)
                f_tx_array = pol_tx[self._gather_ind_tx]

        if self._rx_array_has_element_field:
            f_rx_prime = self._rx_array.element_field(zoa_prime, aoa_prime)
            f_rx = self._l2g_response(f_rx_prime, rx_orientations, zoa, aoa)
            f_rx_array = torch.complex(f_rx, torch.zeros_like(f_rx))
        else:
            f_rx_pol1_prime = torch.stack(
                self._rx_array.ant_pol1.field(zoa_prime, aoa_prime), dim=-1
            )
            f_rx_pol1 = self._l2g_response(
                f_rx_pol1_prime, rx_orientations, zoa, aoa
            )

            if self._rx_array.polarization == "dual":
                f_rx_pol2_prime = torch.stack(
                    self._rx_array.ant_pol2.field(zoa_prime, aoa_prime), dim=-1
                )
                f_rx_pol2 = self._l2g_response(
                    f_rx_pol2_prime, rx_orientations, zoa, aoa
                )

            num_ant_rx = self._rx_array.num_ant
            if self._rx_array.polarization == "single":
                # Each UT antenna gets the polarization 1 response
                f_rx_array = f_rx_pol1.unsqueeze(0).expand(num_ant_rx,
                                                           *f_rx_pol1.shape)
                f_rx_array = torch.complex(f_rx_array,
                                           torch.zeros_like(f_rx_array))
            else:
                # Assign polarization response according to polarization to
                # each antenna using pre-computed gather indices (CUDA graph
                # compatible)
                pol_rx = torch.stack([f_rx_pol1, f_rx_pol2], dim=0)
                f_rx_array = torch.complex(
                    pol_rx[self._gather_ind_rx],
                    torch.zeros_like(pol_rx[self._gather_ind_rx]))

        # Compute the scalar product between the field vectors through
        # reduce_sum and transpose to put antenna dimensions last
        h_field = (f_rx_array.unsqueeze(1) * f_tx_array.unsqueeze(0)).sum(dim=(-2, -1))
        h_field = h_field.permute(
            *range(2, h_field.dim()), 0, 1
        )

        return h_field

    def _step_11_nlos(
        self,
        phi: torch.Tensor,
        topology: Topology,
        rays: Rays,
        t: torch.Tensor,
    ) -> torch.Tensor:
        r"""
        Compute the full NLOS channel matrix (7.5-28)

        :param phi: Random initial phases [radian],
            shape [batch size, num TXs, num RXs, num clusters, num rays, 4]
        :param topology: Topology of the network
        :param rays: Rays
        :param t: Time samples, shape [num time samples]
        """
        h_phase = self._step_11_phase_matrix(phi, rays)
        h_field = self._step_11_field_matrix(
            topology, rays.aoa, rays.aod, rays.zoa, rays.zod, h_phase
        )
        h_array = self._step_11_array_offsets(
            topology, rays.aoa, rays.aod, rays.zoa, rays.zod
        )
        h_doppler = self._step_11_doppler_matrix(
            topology, rays.aoa, rays.aod, rays.zoa, rays.zod, t
        )
        h_full = (h_field * h_array).unsqueeze(-1) * h_doppler.unsqueeze(-2).unsqueeze(-2)

        blockage_loss = getattr(rays, "blockage_loss_db", None)
        if (
            blockage_loss is not None
            and not getattr(rays, "blockage_loss_applied_to_powers", False)
        ):
            blockage_scale = torch.pow(
                torch.tensor(10.0, dtype=self.dtype, device=self.device),
                -blockage_loss / 20.0,
            )
            blockage_scale = torch.complex(
                blockage_scale, torch.zeros_like(blockage_scale)
            )
            for _ in range(h_full.dim() - blockage_scale.dim()):
                blockage_scale = blockage_scale.unsqueeze(-1)
            h_full = h_full * blockage_scale

        power_scaling = torch.complex(
            torch.sqrt(rays.powers / h_full.shape[4]),
            torch.zeros_like(rays.powers),
        )
        # Reshape for broadcasting
        for _ in range(h_full.dim() - power_scaling.dim()):
            power_scaling = power_scaling.unsqueeze(-1)
        h_full = h_full * power_scaling

        return h_full

    def _step_11_reduce_nlos(
        self, h_full: torch.Tensor, rays: Rays, c_ds: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Compute the final NLOS matrix in (7.5-27)

        :param h_full: NLoS channel matrix,
            shape [batch size, num_tx, num rx, num clusters, num rays,
            num rx antennas, num tx antennas, num time steps]
        :param rays: Rays
        :param c_ds: Cluster delay spread,
            shape [batch size, num TX, num RX]

        :output h_nlos: NLoS channel coefficients,
            shape [batch size, num_tx, num rx, num clusters,
            num rx antennas, num tx antennas, num time steps].

        :output delays_nlos: NLoS path delays,
            shape [batch size, num_tx, num rx, num clusters].
        """
        if self._subclustering:
            powers = rays.powers
            delays = rays.delays

            strongest_2 = getattr(rays, "strongest_cluster_indices", None)
            if strongest_2 is None:
                strongest_clusters = torch.argsort(
                    powers, dim=-1, descending=True
                )
                strongest_2 = strongest_clusters[..., :2]
            else:
                remaining_powers = powers.scatter(
                    -1,
                    strongest_2,
                    torch.full_like(strongest_2, -torch.inf, dtype=powers.dtype),
                )
                strongest_rest = torch.argsort(
                    remaining_powers, dim=-1, descending=True
                )[..., : powers.shape[-1] - 2]
                strongest_clusters = torch.cat(
                    [strongest_2, strongest_rest], dim=-1
                )

            # Sort delays according to the same ordering
            delays_sorted = torch.gather(delays, dim=3, index=strongest_clusters)

            # Split into delays for strong and weak clusters
            delays_strong = delays_sorted[..., :2]
            delays_weak = delays_sorted[..., 2:]

            # Compute delays for sub-clusters
            offsets = self._sub_cl_delay_offsets.reshape(
                [1] * (delays_strong.dim() - 1) + [-1, 1]
            )
            delays_sub_cl = delays_strong.unsqueeze(-2) + offsets * c_ds.unsqueeze(-1).unsqueeze(-1)
            delays_sub_cl = delays_sub_cl.reshape(*delays_sub_cl.shape[:-2], -1)

            # Select the strongest two clusters for sub-cluster splitting
            # Expand indices for gather
            idx = strongest_2.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            idx = idx.expand(-1, -1, -1, -1, h_full.shape[4], h_full.shape[5], h_full.shape[6], h_full.shape[7])
            h_strong = torch.gather(h_full, dim=3, index=idx)

            # The other clusters are the weak clusters
            strongest_rest = strongest_clusters[..., 2:]
            idx = strongest_rest.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            idx = idx.expand(-1, -1, -1, -1, h_full.shape[4], h_full.shape[5], h_full.shape[6], h_full.shape[7])
            h_weak = torch.gather(h_full, dim=3, index=idx)

            # Sum specific rays for each sub-cluster
            h_sub_cl_1 = h_strong[:, :, :, :, self._sub_cl_1_ind, ...].sum(dim=4)
            h_sub_cl_2 = h_strong[:, :, :, :, self._sub_cl_2_ind, ...].sum(dim=4)
            h_sub_cl_3 = h_strong[:, :, :, :, self._sub_cl_3_ind, ...].sum(dim=4)

            # Sum all rays for the weak clusters
            h_weak = h_weak.sum(dim=4)

            # Concatenate the channel and delay tensors
            h_nlos = torch.cat([h_sub_cl_1, h_sub_cl_2, h_sub_cl_3, h_weak], dim=3)
            delays_nlos = torch.cat([delays_sub_cl, delays_weak], dim=3)
        else:
            # Sum over rays
            h_nlos = h_full.sum(dim=4)
            delays_nlos = rays.delays

        # Order the delays in ascending orders
        delays_ind = torch.argsort(delays_nlos, dim=-1)
        delays_nlos = torch.gather(delays_nlos, dim=3, index=delays_ind)

        # Order the channel clusters according to the delay, too
        idx = delays_ind.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        idx = idx.expand(-1, -1, -1, -1, h_nlos.shape[4], h_nlos.shape[5], h_nlos.shape[6])
        h_nlos = torch.gather(h_nlos, dim=3, index=idx)

        return h_nlos, delays_nlos

    def _step_11_los(
        self, topology: Topology, t: torch.Tensor
    ) -> torch.Tensor:
        r"""
        Compute the LOS channels from (7.5-29)

        :param topology: Network topology
        :param t: Time samples, shape [num time samples]

        :output h_los: Paths LoS coefficients,
            shape [batch size, num_tx, num rx, 1, num rx antennas, num tx antennas, num time steps]
        """
        aoa = topology.los_aoa
        aod = topology.los_aod
        zoa = topology.los_zoa
        zod = topology.los_zod

        # LoS departure and arrival angles
        aoa = aoa.unsqueeze(3).unsqueeze(4)
        zoa = zoa.unsqueeze(3).unsqueeze(4)
        aod = aod.unsqueeze(3).unsqueeze(4)
        zod = zod.unsqueeze(3).unsqueeze(4)

        # Field matrix
        h_phase = self._h_phase_los_const.reshape(1, 1, 1, 1, 1, 2, 2)
        h_field = self._step_11_field_matrix(topology, aoa, aod, zoa, zod, h_phase)

        # Array offset matrix
        h_array = self._step_11_array_offsets(topology, aoa, aod, zoa, zod)

        # Doppler matrix
        h_doppler = self._step_11_doppler_matrix(
            topology, aoa, aod, zoa, zod, t
        )

        # Phase shift due to propagation delay
        d3d = topology.distance_3d
        lambda_0 = self._lambda_0
        h_delay = torch.exp(
            torch.complex(
                torch.zeros_like(d3d), -2 * PI * d3d / lambda_0
            )
        )

        # Combining all to compute channel coefficient
        h_field = h_field.squeeze(4).unsqueeze(-1)
        h_array = h_array.squeeze(4).unsqueeze(-1)
        h_doppler = h_doppler.unsqueeze(4)
        h_delay = h_delay.unsqueeze(3).unsqueeze(4).unsqueeze(5).unsqueeze(6)

        h_los = h_field * h_array * h_doppler * h_delay
        return h_los

    def _step_11(
        self,
        phi: torch.Tensor,
        topology: Topology,
        k_factor: torch.Tensor,
        rays: Rays,
        t: torch.Tensor,
        c_ds: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Combine LOS and NLOS components to compute (7.5-30)

        :param phi: Random initial phases,
            shape [batch size, num TXs, num RXs, num clusters, num rays, 4]
        :param topology: Network topology
        :param k_factor: Rician K-factor, shape [batch size, num TX, num RX]
        :param rays: Rays
        :param t: Time samples, shape [num time samples]
        :param c_ds: Cluster delay spread, shape [batch size, num TX, num RX]

        :output h: Path coefficients.

        :output delays_nlos: Path delays.
        """
        h_full = self._step_11_nlos(phi, topology, rays, t)
        h_nlos, delays_nlos = self._step_11_reduce_nlos(h_full, rays, c_ds)

        # LoS scenario
        h_los_los_comp = self._step_11_los(topology, t)
        los_blockage = getattr(rays, "los_blockage_loss_db", None)
        if los_blockage is not None:
            los_blockage = torch.pow(
                torch.tensor(10.0, dtype=self.dtype, device=self.device),
                -los_blockage / 20.0,
            )
            los_blockage = torch.complex(
                los_blockage, torch.zeros_like(los_blockage)
            )
            for _ in range(h_los_los_comp.dim() - los_blockage.dim()):
                los_blockage = los_blockage.unsqueeze(-1)
            h_los_los_comp = h_los_los_comp * los_blockage

        k_factor_expanded = k_factor
        for _ in range(h_los_los_comp.dim() - k_factor.dim()):
            k_factor_expanded = k_factor_expanded.unsqueeze(-1)
        k_factor_complex = torch.complex(
            k_factor_expanded, torch.zeros_like(k_factor_expanded)
        )

        # Scale NLOS and LOS components according to K-factor
        h_los_los_comp = h_los_los_comp * torch.sqrt(k_factor_complex / (k_factor_complex + 1))
        h_los_nlos_comp = h_nlos * torch.sqrt(1 / (k_factor_complex + 1))

        # Add the LOS component to the zero-delay NLOS cluster
        h_los_cl = h_los_los_comp + h_los_nlos_comp[:, :, :, 0:1, ...]

        # Combine all clusters into a single tensor
        h_los = torch.cat([h_los_cl, h_los_nlos_comp[:, :, :, 1:, ...]], dim=3)

        # LoS or NLoS CIR according to link configuration
        los_indicator = topology.los.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        h = torch.where(los_indicator, h_los, h_nlos)

        return h, delays_nlos
