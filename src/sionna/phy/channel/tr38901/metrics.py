#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Metrics for 3GPP TR 38.901 calibration and verification.

These diagnostic helpers are not used by channel generation and are not
required during normal channel simulation.
"""

from typing import Optional

import torch

from sionna.phy import config, dtypes
from sionna.phy.config import Precision

__all__ = [
    "rms_delay_spread",
    "circular_angular_spread",
    "delay_spread_from_rays",
    "angular_spreads_from_rays",
    "prb_singular_values",
]


def _dtype(precision: Optional[Precision]) -> torch.dtype:
    if precision is None:
        return config.dtype
    return dtypes[precision]["torch"]["dtype"]


def _to_real_tensor(
    value,
    precision: Optional[Precision] = None,
    device: Optional[str] = None,
) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        if precision is None:
            dtype = value.dtype
        else:
            dtype = _dtype(precision)
        if device is None:
            device = value.device
        return value.to(dtype=dtype, device=device)
    if device is None:
        device = config.device
    return torch.as_tensor(value, dtype=_dtype(precision), device=device)


def _linear_to_db(value: torch.Tensor) -> torch.Tensor:
    floor = torch.finfo(value.real.dtype).tiny
    return 10.0*torch.log10(torch.clamp(value.real, min=floor))


def _fold_zenith(angle: torch.Tensor) -> torch.Tensor:
    angle = torch.remainder(angle, 2.0*torch.pi)
    return torch.where(angle > torch.pi, 2.0*torch.pi - angle, angle)


def rms_delay_spread(
    delays,
    powers,
    precision: Optional[Precision] = None,
    device: Optional[str] = None,
) -> torch.Tensor:
    r"""Computes the RMS delay spread of a power-delay profile.

    For path delays :math:`\tau_n` and linear powers :math:`P_n`, the RMS delay
    spread is

    .. math::

        \sigma_\tau =
        \sqrt{\frac{\sum_n P_n \tau_n^2}{\sum_n P_n}
        - \left(\frac{\sum_n P_n \tau_n}{\sum_n P_n}\right)^2}.

    This is the power-weighted second central moment used for delay-spread
    calibration. TR 38.901 uses RMS delay spread in Clauses 7.7.3 and 7.7.6
    and in the calibration metrics of Tables 7.8-2 and 7.8-7, but does not
    assign this expression a separate equation number
    :cite:p:`TR38901V1920`.

    For a scalar channel impulse response with complex path coefficients
    :math:`a_n`, set :math:`P_n=|a_n|^2`. For MIMO or time-varying channel
    coefficients, the result then depends on how power is selected or averaged
    over antennas and time. Use :func:`delay_spread_from_rays` for the
    propagation-domain TR 38.901 calibration metric.

    :param delays: Path delays [s] with shape ``[..., num_paths]``.
    :param powers: Linear path powers with shape ``[..., num_paths]``.
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`, :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for computation. If `None`,
        :attr:`~sionna.phy.config.Config.device` is used.

    :output ds: RMS delay spread [s] with shape ``[...]``.
    """
    delays = _to_real_tensor(delays, precision=precision, device=device)
    powers = _to_real_tensor(powers, precision=precision, device=delays.device)
    powers = torch.where(torch.isfinite(powers) & (powers > 0.0), powers, 0.0)
    delays = torch.where(torch.isfinite(delays), delays, 0.0)

    total = torch.sum(powers, dim=-1)
    safe_total = torch.clamp(total, min=torch.finfo(powers.dtype).tiny)
    mean = torch.sum(powers*delays, dim=-1)/safe_total
    second = torch.sum(powers*delays*delays, dim=-1)/safe_total
    spread = torch.sqrt(torch.clamp(second - mean*mean, min=0.0))
    return torch.where(total > 0.0, spread, torch.zeros_like(spread))


def circular_angular_spread(
    angles,
    powers,
    precision: Optional[Precision] = None,
    device: Optional[str] = None,
) -> torch.Tensor:
    r"""Computes circular angular spread using the Annex A phasor formula.

    For angles :math:`\phi_n` in radians and linear powers :math:`P_n`, define
    normalized powers :math:`p_n=P_n/\sum_m P_m` and

    .. math::

        R = \left|\sum_n p_n e^{j\phi_n}\right|.

    The circular angular spread is

    .. math::

        \sigma_\phi = \sqrt{-2\ln(R)}.

    This is the angular-spread definition from Annex A.1, Eq. (A-1), of
    :cite:p:`TR38901V1920`, which follows :cite:p:`TR25996`.
    It differs from a wrapped RMS spread around the circular mean.

    :param angles: Angles [radian] with shape ``[..., num_angles]``.
    :param powers: Linear powers with shape ``[..., num_angles]``.
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`, :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for computation. If `None`,
        :attr:`~sionna.phy.config.Config.device` is used.

    :output angular_spread: Circular angular spread [radian] with shape ``[...]``.
    """
    angles = _to_real_tensor(angles, precision=precision, device=device)
    powers = _to_real_tensor(powers, precision=precision, device=angles.device)
    powers = torch.where(torch.isfinite(powers) & (powers > 0.0), powers, 0.0)
    angles = torch.where(torch.isfinite(angles), angles, 0.0)

    total = torch.sum(powers, dim=-1)
    safe_total = torch.clamp(total, min=torch.finfo(powers.dtype).tiny)
    weights = powers/safe_total.unsqueeze(-1)
    mean_sin = torch.sum(weights*torch.sin(angles), dim=-1)
    mean_cos = torch.sum(weights*torch.cos(angles), dim=-1)
    mean_abs = torch.sqrt(mean_sin*mean_sin + mean_cos*mean_cos)
    mean_abs = torch.clamp(mean_abs, min=torch.finfo(powers.dtype).tiny, max=1.0)
    tol = 1e-7 if powers.dtype == torch.float64 else 1e-6
    mean_abs = torch.where(mean_abs > 1.0 - tol,
                           torch.ones_like(mean_abs),
                           mean_abs)
    spread = torch.sqrt(torch.clamp(-2.0*torch.log(mean_abs), min=0.0))
    return torch.where(total > 0.0, spread, torch.zeros_like(spread))


def _gather_serving(values: torch.Tensor, serving: torch.Tensor) -> torch.Tensor:
    batch_size, _num_bs, num_ut = values.shape[:3]
    serving = serving.to(dtype=torch.int64, device=values.device)
    if serving.dim() == 1:
        serving = serving.reshape(batch_size, num_ut)
    batch = torch.arange(batch_size, device=values.device).unsqueeze(-1)
    ut = torch.arange(num_ut, device=values.device).unsqueeze(0)
    return values[batch, serving, ut]


def _delay_spread_from_flat_rays(
    delays: torch.Tensor,
    powers: torch.Tensor,
    c_ds_ns: torch.Tensor,
    k_factor: torch.Tensor,
    los: torch.Tensor,
    include_los_component: bool,
    include_subclusters: bool,
) -> torch.Tensor:
    """Vectorized delay-spread calculation over flattened BS-UT links."""

    valid = torch.isfinite(delays) & torch.isfinite(powers) & (powers > 0.0)
    diffuse_powers = torch.where(valid, powers, torch.zeros_like(powers))
    adjusted = diffuse_powers

    if include_los_component:
        has_valid = torch.any(valid, dim=-1)
        los_k = los & torch.isfinite(k_factor) & has_valid
        scale = torch.where(los_k, k_factor + 1.0, torch.ones_like(k_factor))
        adjusted = adjusted / scale.unsqueeze(-1)
        los_power = torch.where(
            los_k,
            k_factor / torch.clamp(k_factor + 1.0,
                                   min=torch.finfo(k_factor.dtype).tiny),
            torch.zeros_like(k_factor),
        )
        first_valid = torch.argmax(valid.to(torch.int64), dim=-1)
        link_idx = torch.arange(delays.shape[0], device=delays.device)
        adjusted[link_idx, first_valid] = adjusted[link_idx, first_valid] + los_power

    if include_subclusters:
        split_links = (c_ds_ns > 0.0) & (valid.sum(dim=-1) >= 2)
        num_strongest = min(2, powers.shape[-1])
        top_input = torch.where(
            valid,
            diffuse_powers,
            torch.full_like(diffuse_powers, -torch.inf),
        )
        top_indices = torch.topk(top_input, k=num_strongest, dim=-1).indices
        strongest = torch.zeros_like(valid)
        strongest.scatter_(dim=-1, index=top_indices, value=True)
        strongest = strongest & split_links.unsqueeze(-1)

        fractions = torch.tensor(
            [10.0, 6.0, 4.0], dtype=delays.dtype, device=delays.device
        ) / 20.0
        offsets = torch.tensor(
            [0.0, 1.28, 2.56], dtype=delays.dtype, device=delays.device
        )
        offsets = offsets.reshape(1, 1, 3) * c_ds_ns.reshape(-1, 1, 1) * 1e-9
        expanded_delays = delays.unsqueeze(-1) + offsets
        expanded_powers = torch.where(
            strongest.unsqueeze(-1),
            adjusted.unsqueeze(-1) * fractions.reshape(1, 1, 3),
            torch.cat(
                [
                    adjusted.unsqueeze(-1),
                    torch.zeros((*adjusted.shape, 2),
                                dtype=adjusted.dtype,
                                device=adjusted.device),
                ],
                dim=-1,
            ),
        )
        return rms_delay_spread(
            expanded_delays.reshape(delays.shape[0], -1),
            expanded_powers.reshape(powers.shape[0], -1),
        )

    return rms_delay_spread(
        torch.where(valid, delays, torch.zeros_like(delays)),
        adjusted,
    )


def delay_spread_from_rays(
    rays,
    lsp,
    scenario,
    serving: Optional[torch.Tensor] = None,
    include_los_component: bool = True,
    include_subclusters: bool = True,
) -> torch.Tensor:
    r"""Computes delay spread from 3GPP TR 38.901 :cite:p:`TR38901V1920` rays.

    This helper applies the LoS K-factor weighting from Eq. (7.5-30) and
    the optional sub-cluster expansion from Eq. (7.5-26) and Table 7.5-5
    of :cite:p:`TR38901V1920` before evaluating
    :func:`~sionna.phy.channel.tr38901.rms_delay_spread`.

    :param rays: :class:`~sionna.phy.channel.tr38901.Rays`.
        Rays for all BS-UT links.
    :param lsp: :class:`~sionna.phy.channel.tr38901.LSP`.
        Large-scale parameters associated with ``rays``.
    :param scenario: :class:`~sionna.phy.channel.tr38901.SystemLevelScenario`.
        Scenario used to generate ``rays`` and ``lsp``.
    :param serving: Serving BS indices with shape ``[batch size, num_ut]`` or
        ``[batch size*num_ut]``. If `None`, delay spread is computed for all
        BS-UT links.
    :param include_los_component: If `True`, include the deterministic LoS
        component according to the K-factor.
    :param include_subclusters: If `True`, split the two strongest clusters
        according to the subcluster delay offsets from :cite:p:`TR38901V1920`.

    :output ds: Delay spread [s]. Shape is ``[batch size, num_ut]`` if
        ``serving`` is provided and ``[batch size, num_bs, num_ut]`` otherwise.
    """
    delays = rays.delays
    powers = rays.powers
    k_factor = lsp.k_factor
    los = scenario.los
    c_ds = scenario.get_param("cDS")

    out_shape = delays.shape[:3]
    if serving is not None:
        serving = torch.as_tensor(serving, dtype=torch.int64, device=delays.device)
        delays = _gather_serving(delays, serving)
        powers = _gather_serving(powers, serving)
        k_factor = _gather_serving(k_factor.unsqueeze(-1), serving).squeeze(-1)
        los = _gather_serving(los.unsqueeze(-1), serving).squeeze(-1)
        c_ds = _gather_serving(c_ds.unsqueeze(-1), serving).squeeze(-1)
        out_shape = delays.shape[:2]

    flat_delays = delays.reshape(-1, delays.shape[-1])
    flat_powers = powers.reshape(-1, powers.shape[-1])
    flat_k = k_factor.reshape(-1)
    flat_los = los.reshape(-1)
    flat_c_ds = c_ds.reshape(-1)

    values = _delay_spread_from_flat_rays(
        flat_delays,
        flat_powers,
        flat_c_ds,
        flat_k,
        flat_los,
        include_los_component,
        include_subclusters,
    )
    return values.reshape(out_shape)


def _angular_spread_from_flat_rays(
    angles: torch.Tensor,
    cluster_powers: torch.Tensor,
    k_factor: torch.Tensor,
    los: torch.Tensor,
    los_angle: torch.Tensor,
    include_los_component: bool,
) -> torch.Tensor:
    """Vectorized angular-spread calculation over flattened BS-UT links."""

    valid = (
        torch.isfinite(cluster_powers)
        & (cluster_powers > 0.0)
        & torch.all(torch.isfinite(angles), dim=-1)
    )
    num_rays = angles.shape[-1]
    ray_angles = torch.where(
        torch.isfinite(angles),
        angles,
        torch.zeros_like(angles),
    ).reshape(angles.shape[0], -1)
    ray_powers = torch.where(
        valid,
        cluster_powers,
        torch.zeros_like(cluster_powers),
    ).unsqueeze(-1)
    ray_powers = (ray_powers / float(num_rays)).expand_as(angles)

    if include_los_component:
        los_k = los & torch.isfinite(k_factor)
        scale = torch.where(los_k, k_factor + 1.0, torch.ones_like(k_factor))
        ray_powers = ray_powers / scale.reshape(-1, 1, 1)
        los_power = torch.where(
            los_k,
            k_factor / torch.clamp(k_factor + 1.0,
                                   min=torch.finfo(k_factor.dtype).tiny),
            torch.zeros_like(k_factor),
        )
        ray_angles = torch.cat([ray_angles, los_angle.reshape(-1, 1)], dim=-1)
        ray_powers = torch.cat(
            [ray_powers.reshape(angles.shape[0], -1), los_power.reshape(-1, 1)],
            dim=-1,
        )
    else:
        ray_powers = ray_powers.reshape(angles.shape[0], -1)

    return circular_angular_spread(ray_angles, ray_powers)


def angular_spreads_from_rays(
    rays,
    lsp,
    scenario,
    serving: Optional[torch.Tensor] = None,
    include_los_component: bool = True,
) -> dict[str, torch.Tensor]:
    r"""Computes angular spreads from 3GPP TR 38.901 :cite:p:`TR38901V1920` rays.

    The returned angular spreads use the phasor formula from Annex A.1,
    Eq. (A-1), of :cite:p:`TR38901V1920`, which follows
    :cite:p:`TR25996`, as implemented by
    :func:`~sionna.phy.channel.tr38901.circular_angular_spread`. The
    deterministic LoS power is weighted according to Eq. (7.5-30).
    Zenith angles are folded to the physical interval :math:`[0,\pi]`.

    :param rays: :class:`~sionna.phy.channel.tr38901.Rays`.
        Rays for all BS-UT links.
    :param lsp: :class:`~sionna.phy.channel.tr38901.LSP`.
        Large-scale parameters associated with ``rays``.
    :param scenario: :class:`~sionna.phy.channel.tr38901.SystemLevelScenario`.
        Scenario used to generate ``rays`` and ``lsp``.
    :param serving: Serving BS indices with shape ``[batch size, num_ut]`` or
        ``[batch size*num_ut]``. If `None`, angular spreads are computed for
        all BS-UT links.
    :param include_los_component: If `True`, include the deterministic LoS
        component according to the K-factor.

    :output spreads: Dictionary containing ``"asd"``, ``"asa"``, ``"zsd"``,
        and ``"zsa"`` angular spreads [radian].
    """
    powers = rays.powers
    k_factor = lsp.k_factor
    los = scenario.los
    los_angles = {
        "asd": torch.deg2rad(scenario.los_aod),
        "asa": torch.deg2rad(scenario.los_aoa),
        "zsd": _fold_zenith(torch.deg2rad(scenario.los_zod)),
        "zsa": _fold_zenith(torch.deg2rad(scenario.los_zoa)),
    }
    angle_values = {
        "asd": rays.aod,
        "asa": rays.aoa,
        "zsd": _fold_zenith(rays.zod),
        "zsa": _fold_zenith(rays.zoa),
    }

    out_shape = powers.shape[:3]
    if serving is not None:
        serving = torch.as_tensor(serving, dtype=torch.int64, device=powers.device)
        powers = _gather_serving(powers, serving)
        k_factor = _gather_serving(k_factor.unsqueeze(-1), serving).squeeze(-1)
        los = _gather_serving(los.unsqueeze(-1), serving).squeeze(-1)
        for key in angle_values:
            angle_values[key] = _gather_serving(angle_values[key], serving)
            los_angles[key] = _gather_serving(los_angles[key].unsqueeze(-1),
                                              serving).squeeze(-1)
        out_shape = powers.shape[:2]

    flat_powers = powers.reshape(-1, powers.shape[-1])
    flat_k = k_factor.reshape(-1)
    flat_los = los.reshape(-1)
    spreads = {}
    for key, values in angle_values.items():
        flat_angles = values.reshape(-1, values.shape[-2], values.shape[-1])
        flat_los_angle = los_angles[key].reshape(-1)
        link_values = _angular_spread_from_flat_rays(
            flat_angles,
            flat_powers,
            flat_k,
            flat_los,
            flat_los_angle,
            include_los_component,
        )
        spreads[key] = link_values.reshape(out_shape)
    return spreads


def prb_singular_values(
    h: torch.Tensor,
    tau: torch.Tensor,
    carrier_frequency: float,
    subcarrier_spacing: float,
    prb_num_subcarriers: int = 12,
    prb_index: Optional[int] = None,
) -> torch.Tensor:
    r"""Computes PRB singular values for a frequency-selective MIMO channel.

    For a baseband channel impulse response :math:`h_{r,t,n}` with path delays
    :math:`\tau_n`, the frequency response on subcarrier :math:`k` is

    .. math::

        H_k = \sum_n h_n e^{-j2\pi f_k\tau_n}.

    The reported PRB singular values are the eigenvalues of
    :math:`H_kH_k^\mathsf{H}` averaged over all subcarriers of the selected
    physical resource block and expressed in dB. This is the MIMO metric used
    for Phase 2 Config 2 calibration. It is defined by the note following the
    metric list in Table 7.8-2 of
    :cite:p:`TR38901V1920`; the table does not assign it a
    numbered equation.

    :param h: Channel coefficients with shape
        ``[..., num_rx_ant, num_tx_ant, num_paths]``, `torch.complex`.
    :param tau: Path delays [s] with shape ``[..., num_paths]``.
    :param carrier_frequency: Carrier frequency [Hz]. This argument is ignored
        by the baseband computation and retained for backwards-compatible API
        symmetry with channel construction.
    :param subcarrier_spacing: Subcarrier spacing [Hz].
    :param prb_num_subcarriers: Number of subcarriers per PRB.
    :param prb_index: PRB index relative to DC. If `None`, the centered PRB is
        used.

    :output singular_values: Eigenvalues of the averaged covariance matrix [dB]
        with shape ``[..., num_rx_ant]``, sorted in descending order.
    """
    _ = carrier_frequency

    tau = tau.to(dtype=h.real.dtype, device=h.device)
    center = 0.0 if prb_index is None else float(prb_index)*prb_num_subcarriers
    offsets = (
        torch.arange(prb_num_subcarriers, dtype=h.real.dtype, device=h.device)
        - (prb_num_subcarriers - 1.0)/2.0
        + center
    ) * subcarrier_spacing
    phase = torch.exp(
        -1j*2.0*torch.pi*tau.unsqueeze(-2)*offsets.reshape(
            *([1]*(tau.dim() - 1)), prb_num_subcarriers, 1
        )
    )
    response = torch.sum(
        h.unsqueeze(-4) * phase.unsqueeze(-2).unsqueeze(-2),
        dim=-1,
    )
    covariance = response @ response.conj().transpose(-1, -2)
    covariance = torch.mean(covariance, dim=-3)
    eigvals = torch.linalg.eigvalsh(covariance).real
    eigvals = torch.flip(eigvals, dims=(-1,))
    return _linear_to_db(eigvals)
