#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Functional tests for Sionna PHY ISAC plotting utilities."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from sionna.phy.constants import SPEED_OF_LIGHT  # noqa: E402
from sionna.phy.isac import plot_angular_scan, plot_delay_doppler  # noqa: E402


def _mesh_edges(ax):
    """Return the x and y cell edges of the first pcolormesh in ``ax``."""
    coordinates = ax.collections[0].get_coordinates()
    return coordinates[0, :, 0], coordinates[:, 0, 1]


def test_plot_delay_doppler():
    """A single delay-Doppler bin has finite area and labeled axes."""
    spectrum = torch.ones(1, 1)
    fig, ax = plot_delay_doppler(
        spectrum,
        fast_time_sample_rate=70e6,
        slow_time_sample_rate=5e3,
        wavelength=0.1,
        domain="range_velocity",
    )

    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
    assert ax.get_xlabel() == "Range [m]"
    assert ax.get_ylabel() == "Radial velocity [m/s]"

    # The cell spans half a bin on each side: R = c tau / 2, v = lambda nu / 2
    x, y = _mesh_edges(ax)
    half_bin = np.array([-0.5, 0.5])
    np.testing.assert_allclose(x, half_bin / 70e6 * SPEED_OF_LIGHT / 2)
    np.testing.assert_allclose(y, half_bin * 5e3 * 0.1 / 2)
    plt.close(fig)


def test_plot_angular_scan_with_axes():
    """Angular plotting uses matching theta/phi vectors and supplied axes."""
    spectrum = torch.rand(3, 4)
    theta = torch.linspace(0.2, 1.0, 3)
    phi = torch.linspace(-0.5, 0.5, 4)
    expected_fig, expected_ax = plt.subplots()

    fig, ax = plot_angular_scan(
        spectrum, theta, phi, ax=expected_ax
    )

    assert fig is expected_fig
    assert ax is expected_ax
    assert "Azimuth" in ax.get_xlabel()
    assert "Zenith" in ax.get_ylabel()
    plt.close(fig)


def test_plot_delay_doppler_units():
    """Delays and Doppler shifts are shown in microseconds and kilohertz."""
    fast_time_sample_rate = 70e6
    slow_time_sample_rate = 5e3
    fig, ax = plot_delay_doppler(
        torch.rand(4, 8),
        fast_time_sample_rate=fast_time_sample_rate,
        slow_time_sample_rate=slow_time_sample_rate,
    )

    assert ax.get_xlabel() == r"Delay [$\mu$s]"
    assert ax.get_ylabel() == "Doppler frequency [kHz]"

    # Cells are centered on the lags 0..7 and the Doppler bins -2..1
    x, y = _mesh_edges(ax)
    np.testing.assert_allclose(
        x, (np.arange(9) - 0.5) / fast_time_sample_rate * 1e6
    )
    np.testing.assert_allclose(
        y, (np.arange(5) - 2.5) * slow_time_sample_rate / 4 / 1e3
    )
    plt.close(fig)


def test_plot_delay_doppler_index_domain():
    """Without sample rates, the axes fall back to bin indices."""
    fig, ax = plot_delay_doppler(torch.rand(5, 7))

    assert ax.get_xlabel() == "Delay-bin index"
    assert ax.get_ylabel() == "Doppler-bin index"
    plt.close(fig)

    # An explicit index domain overrides the rates
    fig, ax = plot_delay_doppler(
        torch.rand(5, 7),
        fast_time_sample_rate=70e6,
        slow_time_sample_rate=5e3,
        domain="index",
    )
    assert ax.get_xlabel() == "Delay-bin index"
    plt.close(fig)


def test_plot_delay_doppler_l_min_shifts_delay_axis():
    """A nonzero l_min offsets the delay axis by that many bins."""
    spectrum = torch.rand(4, 8)
    _, unshifted = plot_delay_doppler(spectrum)
    _, shifted = plot_delay_doppler(spectrum, l_min=-3)

    offset = shifted.get_xlim()[0] - unshifted.get_xlim()[0]
    assert offset == pytest.approx(-3.0)
    plt.close("all")


@pytest.mark.parametrize(
    "scale,normalize,expected",
    [
        ("db", True, "Normalized power [dB]"),
        ("db", False, "Power [dB]"),
        ("linear", True, "Normalized power"),
        ("linear", False, "Power"),
    ],
)
def test_plot_scale_and_normalize_labels(scale, normalize, expected):
    """The colorbar label reflects the scale and normalization."""
    fig, _ = plot_delay_doppler(
        torch.rand(4, 8) + 0.1, scale=scale, normalize=normalize
    )

    assert fig.axes[-1].get_ylabel() == expected
    plt.close(fig)


def test_plot_all_zero_spectrum_is_not_labeled_normalized():
    """An all-zero spectrum is not normalized, so the label must not say so."""
    for scale, expected in (("db", "Power [dB]"), ("linear", "Power")):
        fig, _ = plot_delay_doppler(torch.zeros(4, 8), scale=scale)
        assert fig.axes[-1].get_ylabel() == expected
        plt.close(fig)


def test_plot_db_floor_clamps_small_values():
    """Values below the floor are clamped and set the lower color limit."""
    spectrum = torch.tensor([[1.0, 1e-9], [1e-9, 1e-9]])
    fig, ax = plot_delay_doppler(spectrum, db_floor=-30.0)

    values = ax.collections[0].get_array()
    assert values.min() == pytest.approx(-30.0)
    assert values.max() == pytest.approx(0.0)
    assert ax.collections[0].get_clim() == (-30.0, 0.0)
    plt.close(fig)


@pytest.mark.parametrize("shape", [(1, 4), (3, 1)])
def test_plot_angular_scan_line_branch(shape):
    """A single retained angle is drawn as a visible line profile."""
    theta = torch.linspace(0.2, 1.0, shape[0])
    phi = torch.linspace(-0.5, 0.5, shape[1])
    fig, ax = plot_angular_scan(torch.rand(*shape), theta, phi)

    assert len(ax.lines) == 1
    assert ax.lines[0].get_marker() == "o"
    assert not ax.collections
    expected = "Azimuth" if shape[0] == 1 else "Zenith"
    assert expected in ax.get_xlabel()
    plt.close(fig)


def test_plot_angular_scan_display_unit():
    """Angles are shown in degrees by default and in radians on request."""
    theta = torch.linspace(0.2, 1.0, 3)
    phi = torch.linspace(-0.5, 0.5, 4)

    for kwargs, unit, convert in (
        ({}, "deg", np.rad2deg),
        ({"display_unit": "rad"}, "rad", np.asarray),
    ):
        fig, ax = plot_angular_scan(torch.rand(3, 4), theta, phi, **kwargs)
        assert unit in ax.get_xlabel()

        # Cells are centered on the uniformly spaced coordinates
        x, y = _mesh_edges(ax)
        np.testing.assert_allclose(
            (x[1:] + x[:-1]) / 2, convert(phi.numpy()), atol=1e-5
        )
        np.testing.assert_allclose(
            (y[1:] + y[:-1]) / 2, convert(theta.numpy()), atol=1e-5
        )
        plt.close(fig)


def test_plot_validation():
    """Unsliced spectra and mismatched angular coordinates are rejected."""
    with pytest.raises(ValueError, match="rank two"):
        plot_delay_doppler(torch.ones(1, 4, 8))

    with pytest.raises(ValueError, match="length 3"):
        plot_angular_scan(
            torch.ones(3, 4),
            torch.linspace(0.0, 1.0, 2),
            torch.linspace(-1.0, 1.0, 4),
        )


def test_plot_rejects_non_tensor_input():
    """Array-likes and non-numeric arguments are rejected with TypeError."""
    with pytest.raises(TypeError, match="must be an instance of Tensor"):
        plot_delay_doppler(np.random.rand(4, 8))

    with pytest.raises(TypeError, match="must be an instance of Tensor"):
        plot_angular_scan(
            np.random.rand(3, 4), torch.zeros(3), torch.zeros(4)
        )

    with pytest.raises(TypeError, match="`fast_time_sample_rate`"):
        plot_delay_doppler(
            torch.rand(4, 8),
            fast_time_sample_rate="70e6",
            slow_time_sample_rate=5e3,
        )

    with pytest.raises(TypeError, match="`l_min`"):
        plot_delay_doppler(torch.rand(4, 8), l_min=0.5)


@pytest.mark.parametrize("rate", [0.0, -1e6, float("inf"), float("nan")])
def test_plot_rejects_invalid_sample_rates(rate):
    """Sample rates must be finite and strictly positive."""
    with pytest.raises(ValueError, match="finite and strictly positive"):
        plot_delay_doppler(
            torch.rand(4, 8),
            fast_time_sample_rate=rate,
            slow_time_sample_rate=1e3,
        )

    with pytest.raises(ValueError, match="finite and strictly positive"):
        plot_delay_doppler(
            torch.rand(4, 8),
            fast_time_sample_rate=1e6,
            slow_time_sample_rate=rate,
        )


def test_plot_rejects_invalid_wavelength():
    """Range-velocity axes require a finite, strictly positive wavelength.

    A rejected call must not leave an empty figure open.
    """
    plt.close("all")
    kwargs = {
        "fast_time_sample_rate": 70e6,
        "slow_time_sample_rate": 5e3,
        "domain": "range_velocity",
    }
    with pytest.raises(ValueError, match="`wavelength` is required"):
        plot_delay_doppler(torch.rand(4, 8), **kwargs)

    with pytest.raises(ValueError, match="finite and strictly positive"):
        plot_delay_doppler(torch.rand(4, 8), wavelength=0.0, **kwargs)
    assert not plt.get_fignums()


def test_plot_rejects_invalid_spectra():
    """Non-finite, negative, and complex powers are rejected."""
    spectrum = torch.rand(4, 8)
    spectrum[0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite, non-negative"):
        plot_delay_doppler(spectrum)

    with pytest.raises(ValueError, match="finite, non-negative"):
        plot_delay_doppler(-torch.rand(4, 8))

    with pytest.raises(TypeError, match="must be real-valued"):
        plot_delay_doppler(torch.ones(4, 8, dtype=torch.complex64))

    with pytest.raises(ValueError, match="finite, non-negative"):
        plot_angular_scan(
            -torch.rand(3, 4), torch.zeros(3), torch.zeros(4)
        )


def test_plot_rejects_invalid_options():
    """Unknown domains, scales, and display units are rejected."""
    with pytest.raises(ValueError, match="one of"):
        plot_delay_doppler(torch.rand(4, 8), domain="polar")

    with pytest.raises(ValueError, match="one of"):
        plot_delay_doppler(torch.rand(4, 8), scale="log")

    with pytest.raises(ValueError, match="one of"):
        plot_angular_scan(
            torch.rand(3, 4),
            torch.zeros(3),
            torch.zeros(4),
            display_unit="grad",
        )


def test_plot_requires_both_sample_rates():
    """Providing only one sample rate is rejected without leaving a figure."""
    plt.close("all")
    with pytest.raises(ValueError, match="both be provided"):
        plot_delay_doppler(torch.rand(4, 8), fast_time_sample_rate=70e6)
    assert not plt.get_fignums()
