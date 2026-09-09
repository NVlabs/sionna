#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Blockage models for 3GPP TR 38.901 system-level channels."""

from typing import Optional, TYPE_CHECKING

import torch

from sionna._validation import check_tensor_all
from sionna.phy.object import Object
from sionna.phy.channel.utils import deg_2_rad, rad_2_deg, wrap_angle_0_360
from sionna.phy.utils import normal, rand

from .spatial_consistency import (
    spatial_consistency_correlation_matrix,
    spatial_consistency_matrix_sqrt,
)
from .utils import update_topology_buffer

if TYPE_CHECKING:
    from .system_level_scenario import SystemLevelScenario

__all__ = ["BlockageModelA", "BlockageModelB"]


class BlockageModelA(Object):
    r"""Stochastic blockage model A from 3GPP TR 38.901.

    This model implements the add-on blockage procedure from Section 7.6.4.1
    of :cite:p:`TR38901V1920`. It generates rectangular angular blocking regions
    around each UT and returns the corresponding blockage attenuation for the
    cluster arrival angles. The LOS/NLOS state of a link is not changed by this
    model.

    For non-self-blocking regions, the loss for a cluster with azimuth angle
    of arrival :math:`\phi_\mathrm{AOA}` and zenith angle of arrival
    :math:`\theta_\mathrm{ZOA}` is

    .. math::

        L_\mathrm{dB}
        = -20 \log_{10}\left(
            1 - (F_{A_1}+F_{A_2})(F_{Z_1}+F_{Z_2})
          \right)

    if :math:`|\phi_\mathrm{AOA}-\phi_k|<x_k` and
    :math:`|\theta_\mathrm{ZOA}-\theta_k|<y_k`, and zero otherwise. The terms
    :math:`F_{A_1}`, :math:`F_{A_2}`, :math:`F_{Z_1}`, and :math:`F_{Z_2}`
    follow Eq. (7.6-23) with the signs from Table 7.6.4.1-3 of
    :cite:p:`TR38901V1920`. A compliant Model A realization also
    includes one self-blocking region. An additional 30 dB loss is added for
    clusters inside the selected portrait or landscape region from Table
    7.6.4.1-1. The explicit ``"none"`` mode omits this region and is a
    non-standard extension.

    The non-self-blocker centre angles are generated as all-correlated uniform
    random variables over BS links, following Section 7.6.3.4 of
    :cite:p:`TR38901V1920`. The specified exponential
    correlation :math:`\rho_u` is imposed with a Gaussian copula whose latent
    correlation is :math:`\rho_g=2\sin(\pi\rho_u/6)`.
    Their spatial correlation distance is 10 m for outdoor UMi, UMa, and RMa
    UTs, 5 m for O2I UTs, and 5 m for InH UTs, as specified by
    Table 7.6.4.1-4.

    Model A is not available for InF because Tables 7.6.4.1-2 and 7.6.4.1-4
    provide neither blocker distributions nor spatial-correlation distances
    for InF. Use :class:`BlockageModelB` with explicit blocker geometry for
    indoor-factory channels.

    The optional, on-demand temporal variability of blockage described by TR
    38.901 is currently not supported. Blockage attenuation does not evolve
    over the time samples of a generated channel realization.

    :param scenario: System-level TR 38.901 scenario.
    :param self_blocking: Self-blocking mode. Must be ``"portrait"`` or
        ``"landscape"`` for a compliant Model A realization. The explicit
        value ``"none"`` disables self-blocking as a non-standard extension.
    :param num_non_self_blockers: Number of non-self-blocking regions. The
        default value of 4 follows Section 7.6.4.1 of
        :cite:p:`TR38901V1920`.
    :param precision: Precision used for internal calculations. If `None`,
        the scenario precision is used.
    :param device: Device for computation. If `None`, the scenario device is
        used.

    :input aoa: Cluster azimuth angles of arrival [degree], shape
        ``[batch size, num_bs, num_ut, num_clusters]``.
    :input zoa: Cluster zenith angles of arrival [degree], shape
        ``[batch size, num_bs, num_ut, num_clusters]``.
    :input los_aoa: Optional LOS azimuth angles of arrival [degree], shape
        ``[batch size, num_bs, num_ut]``.
    :input los_zoa: Optional LOS zenith angles of arrival [degree], shape
        ``[batch size, num_bs, num_ut]``.

    :output cluster_loss_db: Blockage attenuation [dB] for each cluster, shape
        ``[batch size, num_bs, num_ut, num_clusters]``.
    :output los_loss_db: Blockage attenuation [dB] for the deterministic LOS
        component, or `None` if ``los_aoa`` or ``los_zoa`` is `None`.
    """

    _SELF_BLOCKING_PARAMS = {
        "portrait": (260.0, 120.0, 100.0, 80.0),
        "landscape": (40.0, 160.0, 110.0, 75.0),
    }

    def __init__(
        self,
        scenario: "SystemLevelScenario",
        self_blocking: str,
        num_non_self_blockers: int = 4,
        precision: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        precision = scenario.precision if precision is None else precision
        device = scenario.device if device is None else device
        super().__init__(precision=precision, device=device)

        if self_blocking is None:
            raise ValueError(
                "self_blocking must explicitly select 'portrait' or "
                "'landscape'; use 'none' only for the non-standard no-self "
                "variant"
            )
        if self_blocking not in ("none", "portrait", "landscape"):
            raise ValueError(
                "self_blocking must be 'portrait', 'landscape', or 'none'"
            )
        if not isinstance(num_non_self_blockers, int):
            raise TypeError("num_non_self_blockers must be int")
        if num_non_self_blockers < 0:
            raise ValueError("num_non_self_blockers must be non-negative")

        if scenario.scenario_kind not in ("umi", "uma", "rma", "inh"):
            raise ValueError(
                "Blockage model A is specified for UMi, UMa, RMa, and InH "
                "system-level scenarios."
            )

        self._scenario = scenario
        self._self_blocking = self_blocking
        self._num_non_self_blockers = num_non_self_blockers
        self._matrix_sqrt = None
        self._blocker_phi = None
        self._blocker_x = None
        self._blocker_y = None
        if self_blocking == "none":
            self._self_blocking_params = None
        else:
            phi_sb, x_sb, theta_sb, y_sb = self._SELF_BLOCKING_PARAMS[self_blocking]
            self._self_blocking_params = (
                torch.tensor(phi_sb, dtype=self.dtype, device=self.device),
                torch.tensor(x_sb, dtype=self.dtype, device=self.device),
                torch.tensor(theta_sb, dtype=self.dtype, device=self.device),
                torch.tensor(y_sb, dtype=self.dtype, device=self.device),
            )

    @property
    def self_blocking(self) -> str:
        """Self-blocking mode. One of ``"none"``, ``"portrait"``, or
        ``"landscape"``. The ``"none"`` mode is non-standard."""
        return self._self_blocking

    @property
    def num_non_self_blockers(self) -> int:
        """Number of non-self-blocking regions."""
        return self._num_non_self_blockers

    @property
    def requires_ray_angles(self) -> bool:
        """`False` because model A computes one loss per cluster."""
        return False

    def __call__(
        self,
        aoa: torch.Tensor,
        zoa: torch.Tensor,
        los_aoa: Optional[torch.Tensor] = None,
        los_zoa: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Compute cluster and LOS blockage attenuation."""
        if self._blocker_phi is None:
            self.topology_updated_callback()

        cluster_loss = torch.zeros_like(aoa)
        if self._num_non_self_blockers > 0:
            cluster_loss = cluster_loss + self._non_self_blocking_loss(aoa, zoa)
        if self._self_blocking != "none":
            cluster_loss = cluster_loss + self._self_blocking_loss(aoa, zoa)

        los_loss = None
        if los_aoa is not None and los_zoa is not None:
            los_loss = torch.zeros_like(los_aoa)
            if self._num_non_self_blockers > 0:
                los_loss = los_loss + self._non_self_blocking_loss(
                    los_aoa.unsqueeze(-1), los_zoa.unsqueeze(-1)
                ).squeeze(-1)
            if self._self_blocking != "none":
                los_loss = los_loss + self._self_blocking_loss(
                    los_aoa.unsqueeze(-1), los_zoa.unsqueeze(-1)
                ).squeeze(-1)

        return cluster_loss, los_loss

    def topology_updated_callback(self) -> None:
        """Sample topology-dependent blockage parameters."""
        self._compute_matrix_sqrt()
        self._sample_non_self_blockers()

    def reset_topology(self) -> None:
        """Reset topology-dependent blockage buffers."""
        for name in ("_matrix_sqrt", "_blocker_phi", "_blocker_x", "_blocker_y"):
            if hasattr(self, name) and name in self._buffers:
                del self._buffers[name]
            setattr(self, name, None)

    def allocate_topology_tensors(
        self,
        batch_size: int,
        num_bs: int,
        num_ut: int,
    ) -> None:
        """Pre-allocate topology-dependent blockage buffers.

        Model A samples one blocker realization per UT and shares it across
        base stations. ``num_bs`` is retained for the common channel-allocation
        interface but does not determine a buffer dimension here.
        """
        _ = num_bs
        self.reset_topology()
        for name in ("_matrix_sqrt", "_blocker_phi", "_blocker_x", "_blocker_y"):
            if hasattr(self, name) and name not in self._buffers:
                delattr(self, name)
        blocker_shape = (batch_size, num_ut, self._num_non_self_blockers)
        self.register_buffer(
            "_matrix_sqrt",
            torch.zeros(
                batch_size,
                num_ut,
                num_ut,
                dtype=self.dtype,
                device=self.device,
            ),
        )
        self.register_buffer(
            "_blocker_phi",
            torch.zeros(blocker_shape, dtype=self.dtype, device=self.device),
        )
        self.register_buffer(
            "_blocker_x",
            torch.zeros(blocker_shape, dtype=self.dtype, device=self.device),
        )
        self.register_buffer(
            "_blocker_y",
            torch.zeros(blocker_shape, dtype=self.dtype, device=self.device),
        )

    def _scenario_is_indoor_hotspot(self) -> bool:
        """Return `True` for InH scenarios."""
        return self._scenario.scenario_kind == "inh"

    def _blockage_states(self) -> torch.Tensor:
        """UT-state labels for blockage spatial consistency."""
        scenario = self._scenario
        if self._scenario_is_indoor_hotspot():
            states = torch.zeros(
                (scenario.batch_size, scenario.num_ut),
                dtype=torch.int64,
                device=self.device,
            )
        else:
            indoor = scenario.indoor & bool(scenario.o2i_pathloss_enabled)
            states = indoor.to(dtype=torch.int64)
        return states + 2 * scenario.ut_spatial_region_ids

    def _blockage_correlation_distance(self) -> torch.Tensor:
        """Correlation distance from Table 7.6.4.1-4."""
        scenario = self._scenario
        if self._scenario_is_indoor_hotspot():
            return torch.full(
                (scenario.batch_size, scenario.num_ut),
                5.0,
                dtype=self.dtype,
                device=self.device,
            )
        return torch.where(
            scenario.indoor & bool(scenario.o2i_pathloss_enabled),
            torch.tensor(5.0, dtype=self.dtype, device=self.device),
            torch.tensor(10.0, dtype=self.dtype, device=self.device),
        )

    def _compute_matrix_sqrt(self) -> None:
        """Compute the latent-Gaussian UT-domain spatial filter."""
        scenario = self._scenario
        uniform_correlation = spatial_consistency_correlation_matrix(
            scenario.matrix_ut_distance_2d,
            self._blockage_correlation_distance(),
            states=self._blockage_states(),
            correlation_distance_layout="per_terminal",
            precision=self.precision,
            device=self.device,
        )
        gaussian_correlation = 2.0 * torch.sin(
            (torch.pi / 6.0) * uniform_correlation
        )
        gaussian_correlation = gaussian_correlation.clamp(-1.0, 1.0)
        matrix_sqrt = spatial_consistency_matrix_sqrt(
            gaussian_correlation,
            precision=self.precision,
            device=self.device,
        )
        self._update_buffer("_matrix_sqrt", matrix_sqrt)

    def _spatial_uniform(self, shape: tuple[int, ...]) -> torch.Tensor:
        """Generate all-correlated uniform random variables over UTs."""
        samples = normal(
            shape,
            dtype=self.dtype,
            device=self.device,
            generator=self.torch_rng,
        )
        batch_size, num_ut = samples.shape[:2]
        tail_shape = samples.shape[2:]
        samples = samples.reshape(batch_size, num_ut, -1)
        samples = torch.matmul(self._matrix_sqrt, samples)
        samples = samples.reshape(batch_size, num_ut, *tail_shape)
        sqrt_two = torch.sqrt(
            torch.tensor(2.0, dtype=self.dtype, device=self.device)
        )
        samples = 0.5 * torch.erfc(-samples / sqrt_two)
        eps = torch.finfo(self.dtype).eps
        return samples.clamp(eps, 1.0 - eps)

    def _sample_non_self_blockers(self) -> None:
        """Sample non-self-blocking regions from Table 7.6.4.1-2."""
        scenario = self._scenario
        shape = (scenario.batch_size, scenario.num_ut, self._num_non_self_blockers)
        if self._num_non_self_blockers == 0:
            empty = torch.empty(shape, dtype=self.dtype, device=self.device)
            self._update_buffer("_blocker_phi", empty)
            self._update_buffer("_blocker_x", empty)
            self._update_buffer("_blocker_y", empty)
            return

        phi = 360.0 * self._spatial_uniform(shape)
        if self._scenario_is_indoor_hotspot():
            x = 15.0 + 30.0 * rand(
                shape,
                dtype=self.dtype,
                device=self.device,
                generator=self.torch_rng,
            )
            y = 5.0 + 10.0 * rand(
                shape,
                dtype=self.dtype,
                device=self.device,
                generator=self.torch_rng,
            )
        else:
            x = 5.0 + 10.0 * rand(
                shape,
                dtype=self.dtype,
                device=self.device,
                generator=self.torch_rng,
            )
            y = torch.full(shape, 5.0, dtype=self.dtype, device=self.device)

        self._update_buffer("_blocker_phi", phi)
        self._update_buffer("_blocker_x", x)
        self._update_buffer("_blocker_y", y)

    def _update_buffer(self, name: str, value: torch.Tensor) -> None:
        """Update or register a topology-dependent buffer."""
        update_topology_buffer(self, name, value)

    @staticmethod
    def _unit_sphere_vector(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        """Vector on the TR 38.901 unit sphere."""
        return torch.stack(
            [
                torch.sin(theta) * torch.cos(phi),
                torch.sin(theta) * torch.sin(phi),
                torch.cos(theta),
            ],
            dim=-1,
        ).unsqueeze(-1)

    @staticmethod
    def _forward_rotation_matrix(orientations: torch.Tensor) -> torch.Tensor:
        """Forward composite rotation matrix from Eq. (7.1-4)."""
        a, b, c = orientations[..., 0], orientations[..., 1], orientations[..., 2]
        row_1 = torch.stack(
            [
                torch.cos(a) * torch.cos(b),
                torch.cos(a) * torch.sin(b) * torch.sin(c)
                - torch.sin(a) * torch.cos(c),
                torch.cos(a) * torch.sin(b) * torch.cos(c)
                + torch.sin(a) * torch.sin(c),
            ],
            dim=-1,
        )
        row_2 = torch.stack(
            [
                torch.sin(a) * torch.cos(b),
                torch.sin(a) * torch.sin(b) * torch.sin(c)
                + torch.cos(a) * torch.cos(c),
                torch.sin(a) * torch.sin(b) * torch.cos(c)
                - torch.cos(a) * torch.sin(c),
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
        return torch.stack([row_1, row_2, row_3], dim=-2)

    def _gcs_to_lcs(
        self, orientations: torch.Tensor, theta: torch.Tensor, phi: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Transform GCS angles to LCS angles following Eqs. (7.1-7)-(7.1-8)."""
        rho_hat = self._unit_sphere_vector(theta, phi)
        rot_inv = self._forward_rotation_matrix(orientations).mT
        rot_rho = torch.matmul(rot_inv, rho_hat)

        z = rot_rho[..., 2, 0].clamp(-1.0, 1.0)
        theta_prime = torch.acos(z)
        xy = torch.complex(rot_rho[..., 0, 0], rot_rho[..., 1, 0])
        phi_prime = torch.angle(xy)
        return theta_prime, phi_prime

    @staticmethod
    def _signed_angular_difference_deg(
        angle: torch.Tensor, reference: torch.Tensor
    ) -> torch.Tensor:
        """Signed circular difference in degrees within ``[-180, 180)``."""
        return torch.remainder(angle - reference + 180.0, 360.0) - 180.0

    def _self_blocking_loss(
        self, aoa: torch.Tensor, zoa: torch.Tensor
    ) -> torch.Tensor:
        """Compute the optional self-blocking loss."""
        phi_sb, x_sb, theta_sb, y_sb = self._self_blocking_params
        scenario = self._scenario
        orientations = scenario.ut_orientations.unsqueeze(1)
        while orientations.dim() < aoa.dim() + 1:
            orientations = orientations.unsqueeze(-2)

        theta_prime, phi_prime = self._gcs_to_lcs(
            orientations,
            deg_2_rad(zoa),
            deg_2_rad(aoa),
        )
        theta_prime = rad_2_deg(theta_prime)
        phi_prime = wrap_angle_0_360(rad_2_deg(phi_prime))
        phi_diff = torch.abs(
            self._signed_angular_difference_deg(
                phi_prime,
                phi_sb,
            )
        )
        theta_diff = torch.abs(theta_prime - theta_sb)
        blocked = (phi_diff < 0.5 * x_sb) & (theta_diff < 0.5 * y_sb)
        return torch.where(
            blocked,
            torch.tensor(30.0, dtype=self.dtype, device=self.device),
            torch.zeros((), dtype=self.dtype, device=self.device),
        )

    def _non_self_parameters(self, ndim: int) -> tuple[torch.Tensor, ...]:
        """Broadcast non-self-blocker parameters."""
        phi = self._blocker_phi
        x = self._blocker_x
        y = self._blocker_y
        # [batch, ut, blocker] -> [batch, 1, ut, 1, blocker]
        phi = phi.unsqueeze(1).unsqueeze(3)
        x = x.unsqueeze(1).unsqueeze(3)
        y = y.unsqueeze(1).unsqueeze(3)
        while phi.dim() < ndim:
            phi = phi.unsqueeze(3)
            x = x.unsqueeze(3)
            y = y.unsqueeze(3)
        return phi, x, y

    def _non_self_blocking_loss(
        self, aoa: torch.Tensor, zoa: torch.Tensor
    ) -> torch.Tensor:
        """Compute non-self-blocking loss from Eqs. (7.6-22)-(7.6-27)."""
        phi_k, x_k, y_k = self._non_self_parameters(aoa.dim() + 1)
        theta_k = torch.tensor(90.0, dtype=self.dtype, device=self.device)
        r = torch.tensor(
            2.0 if self._scenario_is_indoor_hotspot() else 10.0,
            dtype=self.dtype,
            device=self.device,
        )
        wavelength = self._scenario.lambda_0

        aoa = aoa.unsqueeze(-1)
        zoa = zoa.unsqueeze(-1)
        phi_delta = self._signed_angular_difference_deg(aoa, phi_k)
        theta_delta = zoa - theta_k

        active = (torch.abs(phi_delta) < x_k) & (torch.abs(theta_delta) < y_k)

        a1 = phi_delta - 0.5 * x_k
        a2 = phi_delta + 0.5 * x_k
        z1 = theta_delta - 0.5 * y_k
        z2 = theta_delta + 0.5 * y_k

        sign_a1 = torch.where(
            phi_delta > 0.5 * x_k,
            torch.tensor(-1.0, dtype=self.dtype, device=self.device),
            torch.tensor(1.0, dtype=self.dtype, device=self.device),
        )
        sign_a2 = torch.where(
            phi_delta < -0.5 * x_k,
            torch.tensor(-1.0, dtype=self.dtype, device=self.device),
            torch.tensor(1.0, dtype=self.dtype, device=self.device),
        )
        sign_z1 = torch.where(
            theta_delta > 0.5 * y_k,
            torch.tensor(-1.0, dtype=self.dtype, device=self.device),
            torch.tensor(1.0, dtype=self.dtype, device=self.device),
        )
        sign_z2 = torch.where(
            theta_delta < -0.5 * y_k,
            torch.tensor(-1.0, dtype=self.dtype, device=self.device),
            torch.tensor(1.0, dtype=self.dtype, device=self.device),
        )

        f_a1 = self._knife_edge_term(a1, sign_a1, r, wavelength)
        f_a2 = self._knife_edge_term(a2, sign_a2, r, wavelength)
        f_z1 = self._knife_edge_term(z1, sign_z1, r, wavelength)
        f_z2 = self._knife_edge_term(z2, sign_z2, r, wavelength)

        product = (f_a1 + f_a2) * (f_z1 + f_z2)
        eps = torch.finfo(self.dtype).eps
        argument = (1.0 - product).clamp_min(eps)
        loss = -20.0 * torch.log10(argument)
        loss = torch.where(active, loss, torch.zeros_like(loss))
        return loss.sum(dim=-1)

    def _knife_edge_term(
        self,
        angle_deg: torch.Tensor,
        sign: torch.Tensor,
        r: torch.Tensor,
        wavelength: torch.Tensor,
    ) -> torch.Tensor:
        """Term from Eq. (7.6-23)."""
        angle_rad = deg_2_rad(angle_deg)
        cos_angle = torch.cos(angle_rad).clamp_min(torch.finfo(self.dtype).eps)
        radicand = (torch.pi / wavelength) * r * (1.0 / cos_angle - 1.0)
        radicand = radicand.clamp_min(0.0)
        return torch.atan(sign * 0.5 * torch.pi * torch.sqrt(radicand)) / torch.pi


class BlockageModelB(Object):
    r"""Geometric blockage model B from 3GPP TR 38.901.

    This model implements the rectangular-screen blockage add-on from Section
    7.6.4.2 of :cite:p:`TR38901V1920`. A set of physical blocker screens is placed
    in the global coordinate system. For every sub-path, the screen is rotated
    around its centre such that the incoming ray is perpendicular to the screen
    and the loss is computed from the simple knife-edge diffraction model

    .. math::

        L_\mathrm{dB}
        = -20 \log_{10}\left(
            1 - (F_{h_1}+F_{h_2})(F_{w_1}+F_{w_2})
          \right)

    where the four :math:`F` terms follow Eq. (7.6-30) of
    :cite:p:`TR38901V1920`. Losses from multiple screens are summed in dB. The
    LOS/NLOS state of each link is not changed by this model.

    The positions and dimensions of the blockers are simulation assumptions.
    Recommended dimensions for humans, vehicles, AGVs, and industrial robots
    are listed in Table 7.6.4.2-5 of :cite:p:`TR38901V1920`.

    The optional, on-demand temporal variability of blockage described by TR
    38.901 is currently not supported. Blockage attenuation does not evolve
    over the time samples of a generated channel realization.

    When used with
    :class:`~sionna.phy.channel.tr38901.ChannelCoefficientsGenerator`, the
    returned losses are applied as per-ray amplitude factors
    :math:`10^{-L_\mathrm{dB}/20}` before the channel paths are coherently
    summed. This convention is also used by the bundled blockage Model B RSRP
    calibration curves.

    :param scenario: System-level TR 38.901 scenario.
    :param blocker_centers: Blocker screen centres [m], shape
        ``[num_blockers, 3]`` or ``[batch size, num_blockers, 3]``.
    :param blocker_widths: Blocker widths [m], shape ``[num_blockers]`` or
        ``[batch size, num_blockers]``.
    :param blocker_heights: Blocker heights [m], shape ``[num_blockers]`` or
        ``[batch size, num_blockers]``.
    :param precision: Precision used for internal calculations. If `None`,
        the scenario precision is used.
    :param device: Device for computation. If `None`, the scenario device is
        used.

    :input aoa: Ray azimuth angles of arrival [degree], shape
        ``[batch size, num_bs, num_ut, num_clusters, num_rays]``.
    :input zoa: Ray zenith angles of arrival [degree], shape
        ``[batch size, num_bs, num_ut, num_clusters, num_rays]``.
    :input los_aoa: Optional LOS azimuth angles of arrival [degree], shape
        ``[batch size, num_bs, num_ut]``.
    :input los_zoa: Optional LOS zenith angles of arrival [degree], shape
        ``[batch size, num_bs, num_ut]``.

    :output ray_loss_db: Blockage attenuation [dB] for each ray, shape
        ``[batch size, num_bs, num_ut, num_clusters, num_rays]``.
    :output los_loss_db: Blockage attenuation [dB] for the deterministic LOS
        component, or `None` if ``los_aoa`` or ``los_zoa`` is `None`.
    """

    def __init__(
        self,
        scenario: "SystemLevelScenario",
        blocker_centers: torch.Tensor,
        blocker_widths: torch.Tensor,
        blocker_heights: torch.Tensor,
        precision: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        precision = scenario.precision if precision is None else precision
        device = scenario.device if device is None else device
        super().__init__(precision=precision, device=device)

        self._scenario = scenario
        centers = torch.as_tensor(blocker_centers, dtype=self.dtype, device=self.device)
        widths = torch.as_tensor(blocker_widths, dtype=self.dtype, device=self.device)
        heights = torch.as_tensor(blocker_heights, dtype=self.dtype, device=self.device)

        if centers.dim() == 2:
            centers = centers.unsqueeze(0)
        if centers.dim() != 3 or centers.shape[-1] != 3:
            raise ValueError(
                "blocker_centers must have shape [num_blockers, 3] or "
                "[batch size, num_blockers, 3]"
            )

        if widths.dim() == 1:
            widths = widths.unsqueeze(0)
        if heights.dim() == 1:
            heights = heights.unsqueeze(0)
        if widths.dim() != 2 or heights.dim() != 2:
            raise ValueError(
                "blocker_widths and blocker_heights must have shape "
                "[num_blockers] or [batch size, num_blockers]"
            )
        if (
            widths.shape[-1] != centers.shape[-2]
            or heights.shape[-1] != centers.shape[-2]
        ):
            raise ValueError(
                "blocker_widths and blocker_heights must match the number of "
                "blocker_centers"
            )
        batch_dims = (centers.shape[0], widths.shape[0], heights.shape[0])
        max_batch_dim = max(batch_dims)
        if any(dim not in (1, max_batch_dim) for dim in batch_dims):
            raise ValueError(
                "Batch dimensions of blocker_centers, blocker_widths, and "
                "blocker_heights are inconsistent"
            )
        check_tensor_all(
            (widths > 0.0) & (heights > 0.0),
            name="blocker dimensions",
            message="blocker widths and heights must be positive",
        )

        self.register_buffer("_blocker_centers", centers)
        self.register_buffer("_blocker_widths", widths)
        self.register_buffer("_blocker_heights", heights)

    @property
    def requires_ray_angles(self) -> bool:
        """`True` because model B computes one loss per ray."""
        return True

    @property
    def num_blockers(self) -> int:
        """Number of configured blocker screens."""
        return self._blocker_centers.shape[-2]

    def __call__(
        self,
        aoa: torch.Tensor,
        zoa: torch.Tensor,
        los_aoa: Optional[torch.Tensor] = None,
        los_zoa: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Compute ray and LOS blockage attenuation."""
        if aoa.dim() != 5 or zoa.shape != aoa.shape:
            raise ValueError(
                "Model B expects ray angles with shape "
                "[batch size, num_bs, num_ut, num_clusters, num_rays]"
            )

        ray_loss = self._screen_loss(aoa, zoa, direct_path=False)
        los_loss = None
        if los_aoa is not None and los_zoa is not None:
            los_loss = self._screen_loss(
                los_aoa.unsqueeze(-1).unsqueeze(-1),
                los_zoa.unsqueeze(-1).unsqueeze(-1),
                direct_path=True,
            ).squeeze(-1).squeeze(-1)

        return ray_loss, los_loss

    def topology_updated_callback(self) -> None:
        """Model B blocker geometry is explicit and topology-independent."""

    def reset_topology(self) -> None:
        """Model B has no topology-dependent random buffers."""

    def _blockers_for_batch(self, batch_size: int) -> tuple[torch.Tensor, ...]:
        """Return blocker geometry broadcast to ``batch_size``."""
        centers = self._blocker_centers
        widths = self._blocker_widths
        heights = self._blocker_heights
        if centers.shape[0] == 1 and batch_size != 1:
            centers = centers.expand(batch_size, -1, -1)
        if widths.shape[0] == 1 and batch_size != 1:
            widths = widths.expand(batch_size, -1)
        if heights.shape[0] == 1 and batch_size != 1:
            heights = heights.expand(batch_size, -1)
        if (
            centers.shape[0] != batch_size
            or widths.shape[0] != batch_size
            or heights.shape[0] != batch_size
        ):
            raise ValueError(
                "Batch-specific blocker geometry must match the current "
                "topology batch size"
            )
        return centers, widths, heights

    @staticmethod
    def _unit_sphere_vector(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        """Vector on the TR 38.901 unit sphere."""
        return torch.stack(
            [
                torch.sin(theta) * torch.cos(phi),
                torch.sin(theta) * torch.sin(phi),
                torch.cos(theta),
            ],
            dim=-1,
        )

    def _screen_loss(
        self,
        aoa: torch.Tensor,
        zoa: torch.Tensor,
        direct_path: bool,
    ) -> torch.Tensor:
        """Compute total loss from all screens."""
        scenario = self._scenario
        batch_size = aoa.shape[0]
        centers, widths, heights = self._blockers_for_batch(batch_size)

        ray_dir = self._unit_sphere_vector(deg_2_rad(zoa), deg_2_rad(aoa))
        ray_dir = ray_dir.unsqueeze(-2)

        rx = scenario.ut_loc[:, None, :, None, None, None, :]
        tx = scenario.bs_virtual_loc[:, :, :, None, None, None, :]
        centers = centers[:, None, None, None, None, :, :]
        widths = widths[:, None, None, None, None, :]
        heights = heights[:, None, None, None, None, :]

        loss_top = self._projected_loss(
            ray_dir,
            rx,
            tx,
            centers,
            widths,
            direct_path=direct_path,
            projection="top",
        )
        loss_side = self._projected_loss(
            ray_dir,
            rx,
            tx,
            centers,
            heights,
            direct_path=direct_path,
            projection="side",
        )

        product = loss_side["f_sum"] * loss_top["f_sum"]
        eps = torch.finfo(self.dtype).eps
        loss = -20.0 * torch.log10((1.0 - product).clamp_min(eps))
        active = loss_side["active"] & loss_top["active"]
        loss = torch.where(active, loss, torch.zeros_like(loss))
        return loss.sum(dim=-1)

    def _projected_loss(
        self,
        ray_dir: torch.Tensor,
        rx: torch.Tensor,
        tx: torch.Tensor,
        centers: torch.Tensor,
        size: torch.Tensor,
        direct_path: bool,
        projection: str,
    ) -> dict[str, torch.Tensor]:
        """Compute projected knife-edge terms for one screen dimension."""
        eps = torch.finfo(self.dtype).eps
        if projection == "top":
            norm_xy = torch.linalg.norm(ray_dir[..., :2], dim=-1).clamp_min(eps)
            u = ray_dir[..., :2] / norm_xy.unsqueeze(-1)
            v = torch.stack([-u[..., 1], u[..., 0]], dim=-1)
            rel = centers[..., :2] - rx[..., :2]
            center_parallel = (rel * u).sum(dim=-1)
            center_offset = (rel * v).sum(dim=-1)
            if direct_path:
                tx_rel = tx[..., :2] - rx[..., :2]
                tx_parallel = (tx_rel * u).sum(dim=-1)
                tx_offset = (tx_rel * v).sum(dim=-1)
                path_length = torch.sqrt(tx_parallel.square() + tx_offset.square())
            else:
                tx_parallel = None
                tx_offset = None
                path_length = None
        elif projection == "side":
            norm_xy = torch.linalg.norm(ray_dir[..., :2], dim=-1).clamp_min(eps)
            u_parallel = norm_xy
            u_vertical = ray_dir[..., 2]
            rel_xy = centers[..., :2] - rx[..., :2]
            horizontal_dir = ray_dir[..., :2] / norm_xy.unsqueeze(-1)
            rel_horizontal = (rel_xy * horizontal_dir).sum(dim=-1)
            rel_vertical = centers[..., 2] - rx[..., 2]
            center_parallel = (
                rel_horizontal * u_parallel + rel_vertical * u_vertical
            )
            center_offset = (
                -rel_horizontal * u_vertical + rel_vertical * u_parallel
            )
            if direct_path:
                tx_rel_xy = tx[..., :2] - rx[..., :2]
                tx_horizontal = (tx_rel_xy * horizontal_dir).sum(dim=-1)
                tx_vertical = tx[..., 2] - rx[..., 2]
                tx_parallel = tx_horizontal * u_parallel + tx_vertical * u_vertical
                tx_offset = -tx_horizontal * u_vertical + tx_vertical * u_parallel
                path_length = torch.sqrt(tx_parallel.square() + tx_offset.square())
            else:
                tx_parallel = None
                tx_offset = None
                path_length = None
        else:
            raise ValueError("projection must be 'top' or 'side'")

        edge_1 = center_offset - 0.5 * size
        edge_2 = center_offset + 0.5 * size
        d1_1 = torch.sqrt(center_parallel.square() + edge_1.square())
        d1_2 = torch.sqrt(center_parallel.square() + edge_2.square())

        intersects = torch.abs(center_offset) <= 0.5 * size
        if direct_path:
            d2_1 = torch.sqrt((tx_parallel - center_parallel).square()
                              + (tx_offset - edge_1).square())
            d2_2 = torch.sqrt((tx_parallel - center_parallel).square()
                              + (tx_offset - edge_2).square())
            excess_1 = d1_1 + d2_1 - path_length
            excess_2 = d1_2 + d2_2 - path_length
            sign_metric_1 = d1_1 + d2_1
            sign_metric_2 = d1_2 + d2_2
            between_endpoints = (
                (center_parallel > 0.0)
                & (center_parallel < path_length.clamp_min(eps))
            )
            active = between_endpoints
        else:
            excess_1 = d1_1 - center_parallel
            excess_2 = d1_2 - center_parallel
            sign_metric_1 = d1_1
            sign_metric_2 = d1_2
            active = center_parallel > 0.0

        sign_1 = torch.where(
            intersects | (sign_metric_1 > sign_metric_2),
            torch.tensor(1.0, dtype=self.dtype, device=self.device),
            torch.tensor(-1.0, dtype=self.dtype, device=self.device),
        )
        sign_2 = torch.where(
            intersects | (sign_metric_2 > sign_metric_1),
            torch.tensor(1.0, dtype=self.dtype, device=self.device),
            torch.tensor(-1.0, dtype=self.dtype, device=self.device),
        )

        f_1 = self._knife_edge_term(excess_1, sign_1)
        f_2 = self._knife_edge_term(excess_2, sign_2)
        return {"f_sum": f_1 + f_2, "active": active}

    def _knife_edge_term(
        self,
        excess_distance: torch.Tensor,
        sign: torch.Tensor,
    ) -> torch.Tensor:
        """Knife-edge term from Eq. (7.6-30)."""
        radicand = (torch.pi / self._scenario.lambda_0) * excess_distance
        radicand = radicand.clamp_min(0.0)
        return torch.atan(sign * 0.5 * torch.pi * torch.sqrt(radicand)) / torch.pi
