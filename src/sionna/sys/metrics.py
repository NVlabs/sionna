#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""System-level metric utilities."""

import math
from typing import Optional

import torch

from sionna._validation import check_tensor_all
from sionna.phy import config, dtypes
from sionna.phy.config import Precision

__all__ = [
    "coupling_loss_db",
    "received_power_dbm",
    "serving_indices",
    "geometry_sir_db",
    "geometry_sinr_db",
    "wideband_sir_db",
]


def _dtype(precision: Optional[Precision]) -> torch.dtype:
    if precision is None:
        return config.dtype
    return dtypes[precision]["torch"]["dtype"]


def _to_tensor(
    value,
    precision: Optional[Precision] = None,
    device: Optional[str] = None,
) -> torch.Tensor:
    dtype = _dtype(precision)
    if isinstance(value, torch.Tensor) and device is None:
        device = value.device
    if device is None:
        device = config.device
    return torch.as_tensor(value, dtype=dtype, device=device)


def _log_interference_ratio(
    relative_power_db: torch.Tensor, serving: torch.Tensor
) -> torch.Tensor:
    """Return the natural logarithm of relative interference power."""
    bs_indices = torch.arange(
        relative_power_db.shape[-1], device=relative_power_db.device
    )
    non_serving = bs_indices != serving.unsqueeze(-1)
    relative_power_neper = relative_power_db * (math.log(10.0) / 10.0)
    return torch.logsumexp(
        torch.where(
            non_serving,
            relative_power_neper,
            torch.full_like(relative_power_neper, -torch.inf),
        ),
        dim=-1,
    )


def coupling_loss_db(
    path_gain_db,
    precision: Optional[Precision] = None,
    device: Optional[str] = None,
) -> torch.Tensor:
    r"""Computes the coupling loss from a path gain in dB.

    The coupling loss :math:`\mathrm{CL}_{b,u}` between BS :math:`b` and UT
    :math:`u` is the negative path gain,

    .. math::

        \mathrm{CL}_{b,u}[\mathrm{dB}]
        = -G_{b,u}[\mathrm{dB}],

    where :math:`G_{b,u}` includes all gains and losses between transmitter and
    receiver. It is used in the link budget

    .. math::

        P_{r,b,u}[\mathrm{dBm}]
        = P_{\mathrm{tx},b}[\mathrm{dBm}]
        - \mathrm{CL}_{b,u}[\mathrm{dB}].

    This convention follows the coupling-loss definition used for 3GPP system
    calibration in Section 7.8 of :cite:p:`TR38901V160100`. Coupling loss is one of
    the Phase 1 calibration metrics and is also used as input for the geometry
    SIR, geometry SINR, and wideband SIR metrics.

    :param path_gain_db: Path gain [dB].
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`, :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for computation. If `None`,
        :attr:`~sionna.phy.config.Config.device` is used.

    :output coupling_loss: Coupling loss [dB].

    .. rubric:: Examples

    .. code-block:: python

        from sionna.sys import coupling_loss_db

        cl = coupling_loss_db(-120.0)
        print(cl)
        # tensor(120.)
    """
    return -_to_tensor(path_gain_db, precision=precision, device=device)


def received_power_dbm(
    tx_power_dbm,
    coupling_loss,
    precision: Optional[Precision] = None,
    device: Optional[str] = None,
) -> torch.Tensor:
    r"""Computes received power from transmit power and coupling loss.

    For transmit power :math:`P_{\mathrm{tx},b}` and coupling loss
    :math:`\mathrm{CL}_{b,u}`, the received power is

    .. math::

        P_{r,b,u}[\mathrm{dBm}]
        = P_{\mathrm{tx},b}[\mathrm{dBm}]
        - \mathrm{CL}_{b,u}[\mathrm{dB}].

    This link-budget relation is used by the 3GPP TR 38.901 calibration
    metrics in Section 7.8 of :cite:p:`TR38901V160100` to convert coupling losses
    into desired and interfering received powers.

    :param tx_power_dbm: Transmit power [dBm].
    :param coupling_loss: Coupling loss [dB].
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`, :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for computation. If `None`,
        :attr:`~sionna.phy.config.Config.device` is used.

    :output rx_power_dbm: Received power [dBm].
    """
    cl = _to_tensor(coupling_loss, precision=precision, device=device)
    tx = _to_tensor(tx_power_dbm, precision=precision, device=cl.device)
    return tx - cl


def serving_indices(coupling_loss: torch.Tensor) -> torch.Tensor:
    r"""Returns the serving-cell indices for minimum coupling loss.

    The serving BS is selected as

    .. math::

        b^\star(u) = \arg\min_b \mathrm{CL}_{b,u}.

    This minimum-coupling-loss association is used by the 3GPP TR 38.901
    calibration metrics in Section 7.8 of :cite:p:`TR38901V160100` before computing
    geometry SIR and geometry SINR.

    :param coupling_loss: Coupling loss [dB] with shape ``[..., num_bs]``.

    :output serving: Serving BS indices with shape ``[...]``.
    """
    return torch.argmin(coupling_loss, dim=-1)


def geometry_sir_db(
    coupling_loss,
    serving: Optional[torch.Tensor] = None,
    precision: Optional[Precision] = None,
    device: Optional[str] = None,
) -> torch.Tensor:
    r"""Computes geometry SIR from coupling losses.

    Geometry SIR is the interference-limited signal-to-interference ratio

    .. math::

        \mathrm{SIR}_{u}
        = \frac{P_{r,b^\star,u}}
               {\sum_{b\ne b^\star} P_{r,b,u}},

    where :math:`b^\star` is the serving BS and received powers are obtained
    from coupling losses. Transmit powers are assumed equal for all base stations and
    therefore cancel.

    Geometry SIR is a 3GPP TR 38.901 system calibration metric from Section
    7.8 of :cite:p:`TR38901V160100`. It is used to verify interference geometry
    independently of thermal noise.

    The ratio is evaluated from relative gains in the log domain, making the
    result invariant to large common coupling-loss offsets without numerical
    underflow or overflow.

    :param coupling_loss: Coupling loss [dB] with shape ``[..., num_bs]``.
    :param serving: Serving BS indices with shape ``[...]``. If `None`, the
        serving BS is selected by :func:`~sionna.sys.serving_indices`.
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`, :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for computation. If `None`,
        :attr:`~sionna.phy.config.Config.device` is used.

    :output sir: Geometry SIR [dB] with shape ``[...]``.
    """
    cl = _to_tensor(coupling_loss, precision=precision, device=device)
    if serving is None:
        serving = serving_indices(cl)
    else:
        serving = torch.as_tensor(serving, dtype=torch.int64, device=cl.device)

    desired_cl = torch.take_along_dim(
        cl, serving.unsqueeze(-1), dim=-1
    ).squeeze(-1)
    relative_power_db = desired_cl.unsqueeze(-1) - cl
    log_interference_ratio = _log_interference_ratio(
        relative_power_db, serving
    )
    return -(10.0 / math.log(10.0)) * log_interference_ratio


def geometry_sinr_db(
    coupling_loss,
    tx_power_dbm,
    bandwidth_hz,
    noise_figure_db,
    serving: Optional[torch.Tensor] = None,
    precision: Optional[Precision] = None,
    device: Optional[str] = None,
) -> torch.Tensor:
    r"""Computes geometry SINR from coupling losses and receiver noise.

    Geometry SINR is

    .. math::

        \mathrm{SINR}_{u}
        = \frac{P_{r,b^\star,u}}
               {\sum_{b\ne b^\star} P_{r,b,u} + N},

    where :math:`N` is the thermal noise power including receiver noise figure
    over the configured bandwidth,

    .. math::

        N[\mathrm{dBm}]
        = -174 + 10\log_{10}(B[\mathrm{Hz}]) + F[\mathrm{dB}].

    Geometry SINR is used in the Phase 1 system calibration described in
    Section 7.8 of :cite:p:`TR38901V160100`. It verifies the combined effect of
    serving-cell association, interference geometry, transmit power, bandwidth,
    and receiver noise figure.

    :param coupling_loss: Coupling loss [dB] with shape ``[..., num_bs]``.
    :param tx_power_dbm: Transmit power [dBm].
    :param bandwidth_hz: Finite, positive bandwidth [Hz].
    :param noise_figure_db: Receiver noise figure [dB].
    :param serving: Serving BS indices with shape ``[...]``. If `None`, the
        serving BS is selected by :func:`~sionna.sys.serving_indices`.
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`, :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for computation. If `None`,
        :attr:`~sionna.phy.config.Config.device` is used.

    :output sinr: Geometry SINR [dB] with shape ``[...]``.
    """
    cl = _to_tensor(coupling_loss, precision=precision, device=device)
    if serving is None:
        serving = serving_indices(cl)
    else:
        serving = torch.as_tensor(serving, dtype=torch.int64, device=cl.device)

    rx_dbm = received_power_dbm(
        tx_power_dbm, cl, precision=precision, device=cl.device
    )
    desired_dbm = torch.take_along_dim(
        rx_dbm, serving.unsqueeze(-1), dim=-1
    ).squeeze(-1)
    relative_power_db = rx_dbm - desired_dbm.unsqueeze(-1)
    log_interference_ratio = _log_interference_ratio(
        relative_power_db, serving
    )

    bandwidth = _to_tensor(bandwidth_hz, precision=precision, device=cl.device)
    if bandwidth.numel() == 0:
        raise ValueError("`bandwidth_hz` must contain finite, positive values")
    check_tensor_all(
        torch.isfinite(bandwidth) & (bandwidth > 0.0),
        name="bandwidth_hz",
        message="`bandwidth_hz` must contain finite, positive values",
    )
    noise_figure = _to_tensor(noise_figure_db, precision=precision, device=cl.device)
    noise_dbm = -174.0 + 10.0*torch.log10(bandwidth) + noise_figure
    log_noise_ratio = (
        noise_dbm - desired_dbm
    ) * (math.log(10.0) / 10.0)
    log_denominator_ratio = torch.logaddexp(
        log_interference_ratio, log_noise_ratio
    )
    return -(10.0 / math.log(10.0)) * log_denominator_ratio


def wideband_sir_db(
    coupling_loss,
    serving: Optional[torch.Tensor] = None,
    precision: Optional[Precision] = None,
    device: Optional[str] = None,
) -> torch.Tensor:
    r"""Computes wideband SIR from coupling losses.

    For a serving BS :math:`b^\star(u)` selected by minimum coupling loss,

    .. math::

        b^\star(u) = \arg\min_b \mathrm{CL}_{b,u},

    the wideband SIR is

    .. math::

        \mathrm{SIR}^{\mathrm{WB}}_u
        = \frac{10^{-\mathrm{CL}_{b^\star,u}/10}}
               {\sum_{b\ne b^\star}10^{-\mathrm{CL}_{b,u}/10}}.

    The returned value is :math:`10\log_{10}(\mathrm{SIR}^{\mathrm{WB}}_u)`.
    Equal BS transmit powers are assumed and therefore cancel.

    With equal BS transmit powers and no small-scale fading, this expression is
    numerically identical to :func:`~sionna.sys.geometry_sir_db`. The separate
    name is kept because 3GPP TR 38.901 Phase 2 calibration in Section 7.8 of
    :cite:p:`TR38901V160100` refers to the corresponding quantity as wideband SIR. It
    verifies the large-scale interference geometry used before evaluating the
    Phase 2 small-scale fading metrics.

    :param coupling_loss: Coupling loss [dB] with shape ``[..., num_bs]``.
    :param serving: Serving BS indices with shape ``[...]``. If `None`, the
        serving BS is selected by :func:`~sionna.sys.serving_indices`.
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`, :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for computation. If `None`,
        :attr:`~sionna.phy.config.Config.device` is used.

    :output sir: Wideband SIR [dB] with shape ``[...]``.
    """
    return geometry_sir_db(
        coupling_loss, serving=serving, precision=precision, device=device
    )
