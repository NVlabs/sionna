#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Plotting functions for integrated sensing and communication."""

import math
import numbers
from typing import Literal, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from sionna._validation import check_instance, check_one_of, check_tensor_all
from sionna.phy.constants import SPEED_OF_LIGHT

__all__ = [
    "plot_delay_doppler",
    "plot_angular_scan",
]


def plot_delay_doppler(
    delay_doppler_spectrum: torch.Tensor,
    *,
    l_min: int = 0,
    fast_time_sample_rate: Optional[float] = None,
    slow_time_sample_rate: Optional[float] = None,
    wavelength: Optional[float] = None,
    domain: Literal[
        "auto", "index", "delay_doppler", "range_velocity"
    ] = "auto",
    scale: Literal["linear", "db"] = "db",
    normalize: bool = True,
    db_floor: float = -40.0,
    ax: Optional[Axes] = None,
    cmap: str = "viridis",
) -> Tuple[Figure, Axes]:
    r"""Plot a selected delay-Doppler spectrum.

    For :math:`N_\text{D}` Doppler bins and fast- and slow-time sample rates
    :math:`f_\text{fast}` and :math:`f_\text{slow}`, the physical coordinates
    are

    .. math::

        \tau_\ell &= \frac{\ell}{f_\text{fast}},\\
        \nu_q &= \frac{q f_\text{slow}}{N_\text{D}},

    where :math:`\ell=L_\text{min},\ldots,L_\text{min}+N_\text{L}-1` for
    :math:`N_\text{L}` delay bins and
    :math:`q=-\lfloor N_\text{D}/2\rfloor,\ldots,
    N_\text{D}-\lfloor N_\text{D}/2\rfloor-1`. For monostatic sensing, these
    coordinates can be converted to range and radial velocity according to

    .. math::

        R_\ell = \frac{c\tau_\ell}{2},
        \qquad
        v_q = \frac{\lambda\nu_q}{2}.

    A positive Doppler frequency, and hence a positive radial velocity,
    corresponds to a target moving towards the sensing device.

    Delays and Doppler frequencies are displayed in microseconds and
    kilohertz, ranges and radial velocities in meters and meters per second.

    :param delay_doppler_spectrum: Selected linear-power spectrum with shape
        [num_doppler_bins, num_delay_bins]. Doppler bins must use centered
        ordering.
    :param l_min: Time lag of the first delay bin
        (:math:`L_\text{min}`). Must match the ``l_min`` passed to
        :func:`~sionna.phy.channel.ofdm_to_delay_doppler_channel`. Defaults
        to 0.
    :param fast_time_sample_rate: Fast-time sample rate [Hz]. Required for
        physical delay, range, Doppler, or velocity axes. Must be finite and
        strictly positive.
    :param slow_time_sample_rate: Slow-time sample rate [Hz]. Required for
        physical delay, range, Doppler, or velocity axes. Must be finite and
        strictly positive.
    :param wavelength: Carrier wavelength [m]. Must be finite and strictly
        positive. Required when ``domain="range_velocity"`` and ignored
        otherwise.
    :param domain: Axis domain. ``"auto"`` uses indices if both sample rates
        are omitted and delay/Doppler otherwise. ``"index"`` always uses bin
        indices. Defaults to ``"auto"``.
    :param scale: Power-display scale, ``"linear"`` or ``"db"``. Defaults to
        ``"db"``.
    :param normalize: If `True`, normalize the displayed spectrum by its
        maximum. Defaults to `True`.
    :param db_floor: Smallest displayed value in decibels. Only used for
        ``scale="db"``. Defaults to -40.
    :param ax: Matplotlib axes into which the spectrum is drawn. If `None`, a
        new figure and axes are created.
    :param cmap: Matplotlib colormap.

    :output fig: `matplotlib.figure.Figure`. Figure containing the plot.
    :output ax: `matplotlib.axes.Axes`. Axes containing the plot.

    .. rubric:: Examples

    The following example plots a synthetic off-grid delay-Doppler spectrum.

    .. code-block:: python

        import matplotlib.pyplot as plt
        import torch
        from sionna.phy.isac import plot_delay_doppler

        delay = torch.arange(64)
        doppler = torch.arange(-16, 16)
        delay_doppler_spectrum = (
            torch.sinc(doppler[:, None]-2.35).square()
            * torch.sinc(delay[None, :]-10.4).square())
        fig, ax = plot_delay_doppler(delay_doppler_spectrum)
        plt.show()

    .. figure:: /phy/figures/plot_delay_doppler.png
        :align: center
        :width: 70%

        Synthetic off-grid delay-Doppler spectrum.
    """
    spectrum = _check_spectrum(
        delay_doppler_spectrum, name="delay_doppler_spectrum"
    )
    check_one_of(
        domain,
        ("auto", "index", "delay_doppler", "range_velocity"),
        name="domain",
    )
    check_one_of(scale, ("linear", "db"), name="scale")
    check_instance(l_min, numbers.Integral, name="l_min")

    rates = (fast_time_sample_rate, slow_time_sample_rate)
    if (rates[0] is None) != (rates[1] is None):
        raise ValueError(
            "`fast_time_sample_rate` and `slow_time_sample_rate` must either "
            "both be provided or both be omitted."
        )
    if rates[0] is not None:
        fast_time_sample_rate = _check_positive(
            rates[0], name="fast_time_sample_rate"
        )
        slow_time_sample_rate = _check_positive(
            rates[1], name="slow_time_sample_rate"
        )

    if domain == "auto":
        domain = "index" if rates[0] is None else "delay_doppler"
    if domain in ("delay_doppler", "range_velocity") and rates[0] is None:
        raise ValueError(
            "Physical axes require `fast_time_sample_rate` and "
            "`slow_time_sample_rate`."
        )
    if domain == "range_velocity":
        if wavelength is None:
            raise ValueError(
                "`wavelength` is required for range-velocity axes."
            )
        wavelength = _check_positive(wavelength, name="wavelength")

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    else:
        fig = ax.figure

    num_doppler_bins, num_delay_bins = spectrum.shape
    # Explicit bin edges retain a finite cell width for singleton axes.
    delay = np.arange(l_min, l_min + num_delay_bins + 1, dtype=np.float64) - 0.5
    doppler = (
        np.arange(num_doppler_bins + 1, dtype=np.float64)
        - num_doppler_bins // 2
        - 0.5
    )

    if domain == "index":
        xlabel = "Delay-bin index"
        ylabel = "Doppler-bin index"
        title = "Delay-Doppler Spectrum"
    else:
        delay /= fast_time_sample_rate
        doppler *= slow_time_sample_rate / num_doppler_bins
        if domain == "delay_doppler":
            # Sensing delays and Doppler shifts are conventionally reported in
            # microseconds and kilohertz
            delay *= 1e6
            doppler /= 1e3
            xlabel = r"Delay [$\mu$s]"
            ylabel = "Doppler frequency [kHz]"
            title = "Delay-Doppler Spectrum"
        else:
            delay *= SPEED_OF_LIGHT / 2
            doppler *= wavelength / 2
            xlabel = "Range [m]"
            ylabel = "Radial velocity [m/s]"
            title = "Range-Velocity Spectrum"

    values, colorbar_label, vmin, vmax = _display_values(
        spectrum, scale=scale, normalize=normalize, db_floor=db_floor
    )
    image = ax.pcolormesh(
        delay,
        doppler,
        values,
        shading="flat",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    fig.colorbar(image, ax=ax, label=colorbar_label)
    return fig, ax


def plot_angular_scan(
    angular_spectrum: torch.Tensor,
    theta: torch.Tensor,
    phi: torch.Tensor,
    *,
    display_unit: Literal["rad", "deg"] = "deg",
    scale: Literal["linear", "db"] = "db",
    normalize: bool = True,
    db_floor: float = -40.0,
    ax: Optional[Axes] = None,
    cmap: str = "viridis",
) -> Tuple[Figure, Axes]:
    r"""Plot a selected theta-phi angular spectrum.

    The map value at indices :math:`(i,j)` is displayed at zenith and azimuth
    coordinates :math:`(\theta_i,\varphi_j)`. Both are always provided in
    radians and can be converted to degrees for display.

    :param angular_spectrum: Selected linear-power angular spectrum with shape
        [num_theta, num_phi].
    :param theta: Zenith coordinates [rad], shape [num_theta].
    :param phi: Azimuth coordinates [rad], shape [num_phi].
    :param display_unit: Unit used for displaying ``theta`` and ``phi``,
        ``"rad"`` or ``"deg"``. Defaults to ``"deg"``.
    :param scale: Power-display scale, ``"linear"`` or ``"db"``. Defaults to
        ``"db"``.
    :param normalize: If `True`, normalize the displayed spectrum by its
        maximum. Defaults to `True`.
    :param db_floor: Smallest displayed value in decibels. Only used for
        ``scale="db"``. Defaults to -40.
    :param ax: Matplotlib axes into which the spectrum is drawn. If `None`, a
        new figure and axes are created.
    :param cmap: Matplotlib colormap.

    :output fig: `matplotlib.figure.Figure`. Figure containing the plot.
    :output ax: `matplotlib.axes.Axes`. Axes containing the plot.

    .. rubric:: Examples

    The following example plots a synthetic theta-phi spectrum.

    .. code-block:: python

        import matplotlib.pyplot as plt
        import torch
        from sionna.phy.isac import plot_angular_scan

        theta = torch.deg2rad(torch.linspace(60., 120., 31))
        phi = torch.deg2rad(torch.linspace(-60., 60., 61))
        angular_spectrum = torch.exp(
            -((theta[:, None]-1.41)/0.15)**2
            -((phi[None, :]-0.33)/0.2)**2)
        fig, ax = plot_angular_scan(angular_spectrum, theta, phi)
        plt.show()

    .. figure:: /phy/figures/plot_angular_scan.png
        :align: center
        :width: 70%

        Synthetic theta-phi angular spectrum.
    """
    spectrum = _check_spectrum(angular_spectrum, name="angular_spectrum")
    check_instance(theta, torch.Tensor, name="theta")
    check_instance(phi, torch.Tensor, name="phi")
    if theta.dim() != 1 or theta.shape[0] != spectrum.shape[0]:
        raise ValueError(f"`theta` must have length {spectrum.shape[0]}.")
    if phi.dim() != 1 or phi.shape[0] != spectrum.shape[1]:
        raise ValueError(f"`phi` must have length {spectrum.shape[1]}.")
    check_one_of(display_unit, ("rad", "deg"), name="display_unit")
    check_one_of(scale, ("linear", "db"), name="scale")

    theta = theta.detach().cpu().numpy()
    phi = phi.detach().cpu().numpy()
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    else:
        fig = ax.figure

    if display_unit == "deg":
        theta = np.rad2deg(theta)
        phi = np.rad2deg(phi)
        unit = "deg"
    else:
        unit = "rad"

    values, colorbar_label, vmin, vmax = _display_values(
        spectrum, scale=scale, normalize=normalize, db_floor=db_floor
    )
    num_theta, num_phi = spectrum.shape
    if num_theta == 1 or num_phi == 1:
        if num_theta == 1:
            coordinate = phi
            profile = values[0]
            xlabel = rf"Azimuth $\varphi$ [{unit}]"
        else:
            coordinate = theta
            profile = values[:, 0]
            xlabel = rf"Zenith $\theta$ [{unit}]"
        # A marker keeps a single retained direction visible
        ax.plot(coordinate, profile, marker="o", markersize=3)
        ax.set(xlabel=xlabel, ylabel=colorbar_label, title="Angular Spectrum")
        ax.grid()
    else:
        image = ax.pcolormesh(
            phi,
            theta,
            values,
            shading="auto",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        ax.set(
            xlabel=rf"Azimuth $\varphi$ [{unit}]",
            ylabel=rf"Zenith $\theta$ [{unit}]",
            title="Angular Spectrum",
        )
        fig.colorbar(image, ax=ax, label=colorbar_label)
    return fig, ax


def _check_positive(value: float, *, name: str) -> float:
    """Require a scalar to be finite and strictly positive."""
    if (
        isinstance(value, torch.Tensor)
        and value.numel() == 1
        and not value.is_complex()
    ):
        value = value.item()
    check_instance(value, numbers.Real, name=name)
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"`{name}` must be finite and strictly positive.")
    return value


def _check_spectrum(spectrum: torch.Tensor, *, name: str) -> torch.Tensor:
    """Require a rank-two, real, finite, non-negative power spectrum."""
    check_instance(spectrum, torch.Tensor, name=name)
    if spectrum.dim() != 2:
        raise ValueError(f"`{name}` must have rank two.")
    if spectrum.is_complex():
        raise TypeError(f"`{name}` must be real-valued.")
    spectrum = spectrum.detach()
    check_tensor_all(
        torch.isfinite(spectrum) & (spectrum >= 0),
        name=name,
        message=f"`{name}` must contain finite, non-negative powers.",
    )
    return spectrum


def _display_values(
    spectrum: torch.Tensor,
    *,
    scale: str,
    normalize: bool,
    db_floor: float,
) -> Tuple[np.ndarray, str, Optional[float], Optional[float]]:
    """Prepare spectrum values and color limits for plotting."""
    values = spectrum.to(dtype=torch.float64)
    maximum = float(values.max())

    # An all-zero spectrum cannot be normalized, so the label must not claim
    # that it was
    normalized = normalize and maximum > 0
    if normalized:
        values = values / maximum

    if scale == "linear":
        label = "Normalized power" if normalized else "Power"
        vmax = 1.0 if normalized else None
        return values.cpu().numpy(), label, 0.0, vmax

    if maximum == 0:
        values = torch.full_like(values, db_floor)
    else:
        tiny = torch.finfo(values.dtype).tiny
        values = 10 * torch.log10(torch.clamp(values, min=tiny))
        values = torch.clamp(values, min=db_floor)
    label = "Normalized power [dB]" if normalized else "Power [dB]"
    vmax = 0.0 if normalized else None
    return values.cpu().numpy(), label, db_floor, vmax
