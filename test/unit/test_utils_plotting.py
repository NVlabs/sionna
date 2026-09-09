#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Unit tests for sionna.phy.utils.plotting"""

import numpy as np
import pytest
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend for testing
import matplotlib.pyplot as plt

from sionna.phy.utils import plot_ber, PlotBER


class TestPlotBer:
    """Tests for the plot_ber function."""

    def test_single_curve(self):
        """Test plotting a single BER curve."""
        snr = np.array([0, 2, 4, 6, 8, 10])
        ber = np.array([0.2, 0.1, 0.05, 0.01, 0.001, 0.0001])

        fig, ax = plot_ber(snr, ber, legend="AWGN", title="BER vs SNR")

        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        assert ax.get_title() == "BER vs SNR"
        plt.close(fig)

    def test_no_legend_box_without_labels(self):
        """An omitted legend leaves no empty legend box behind."""
        snr = np.array([0, 2, 4, 6, 8, 10])
        ber = np.array([0.2, 0.1, 0.05, 0.01, 0.001, 0.0001])

        fig, ax = plot_ber(snr, ber)
        assert ax.get_legend() is None

        fig_labeled, ax_labeled = plot_ber(snr, ber, legend="AWGN")
        assert ax_labeled.get_legend() is not None
        plt.close(fig)
        plt.close(fig_labeled)

    def test_multiple_curves(self):
        """Test plotting multiple BER curves."""
        snr1 = np.array([0, 2, 4, 6, 8])
        snr2 = np.array([0, 2, 4, 6, 8])
        ber1 = np.array([0.2, 0.1, 0.05, 0.01, 0.001])
        ber2 = np.array([0.15, 0.08, 0.04, 0.008, 0.0008])

        fig, _ = plot_ber(
            [snr1, snr2],
            [ber1, ber2],
            legend=["Curve 1", "Curve 2"],
            title="Multiple Curves",
        )

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_broadcast_snr(self):
        """Test that snr is broadcast when ber is a list but snr is not."""
        snr = np.array([0, 2, 4, 6, 8])
        ber1 = np.array([0.2, 0.1, 0.05, 0.01, 0.001])
        ber2 = np.array([0.15, 0.08, 0.04, 0.008, 0.0008])

        fig, _ = plot_ber(
            snr,  # Single snr array
            [ber1, ber2],  # List of ber arrays
            legend=["Curve 1", "Curve 2"],
        )

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_bler_dashed_line(self):
        """Test that BLER curves use dashed line style."""
        snr = np.array([0, 2, 4, 6, 8])
        ber = np.array([0.2, 0.1, 0.05, 0.01, 0.001])

        fig, ax = plot_ber(snr, ber, is_bler=True, legend="BLER")

        lines = ax.get_lines()
        assert len(lines) == 1
        # Dashed lines have non-empty dash pattern
        assert lines[0].get_linestyle() == "--"
        plt.close(fig)

    @pytest.mark.parametrize(
        ("is_bler", "expected_style"),
        [([True], "--"), ([False], "-"), (False, "-")],
    )
    def test_single_curve_is_bler_normalization(self, is_bler, expected_style):
        """Single curves accept scalar or one-element BLER flags."""
        snr = np.array([0, 1, 2])
        ber = np.array([0.1, 0.01, 0.001])
        fig, ax = plot_ber(snr, ber, is_bler=is_bler)
        assert ax.get_lines()[0].get_linestyle() == expected_style
        plt.close(fig)

    def test_scalar_is_bler_broadcasts_to_multiple_curves(self):
        """A scalar BLER flag applies to every curve."""
        snr = np.array([0, 1, 2])
        ber = np.array([0.1, 0.01, 0.001])
        fig, ax = plot_ber(snr, [ber, ber], is_bler=True)
        assert [line.get_linestyle() for line in ax.get_lines()] == ["--", "--"]
        plt.close(fig)

    def test_esno_xlabel(self):
        """Test that x-label changes when ebno=False."""
        snr = np.array([0, 2, 4, 6])
        ber = np.array([0.2, 0.1, 0.05, 0.01])

        fig, ax = plot_ber(snr, ber, ebno=False)

        assert r"$E_s/N_0$" in ax.get_xlabel()
        plt.close(fig)

    def test_ebno_xlabel(self):
        """Test that x-label is EbNo when ebno=True."""
        snr = np.array([0, 2, 4, 6])
        ber = np.array([0.2, 0.1, 0.05, 0.01])

        fig, ax = plot_ber(snr, ber, ebno=True)

        assert r"$E_b/N_0$" in ax.get_xlabel()
        plt.close(fig)

    def test_xlim_ylim(self):
        """Test that axis limits are applied correctly."""
        snr = np.array([0, 2, 4, 6, 8])
        ber = np.array([0.2, 0.1, 0.05, 0.01, 0.001])

        fig, ax = plot_ber(snr, ber, xlim=(0, 10), ylim=(1e-4, 1))

        assert ax.get_xlim() == (0, 10)
        assert ax.get_ylim() == (1e-4, 1)
        plt.close(fig)

    def test_invalid_legend_type(self):
        """Test that invalid legend type raises TypeError."""
        snr = np.array([0, 2, 4])
        ber = np.array([0.1, 0.05, 0.01])

        with pytest.raises(TypeError, match="legend must be str or list"):
            plot_ber(snr, ber, legend=123)

    def test_invalid_title_type(self):
        """Test that invalid title type raises TypeError."""
        snr = np.array([0, 2, 4])
        ber = np.array([0.1, 0.05, 0.01])

        with pytest.raises(TypeError, match="title must be str"):
            plot_ber(snr, ber, title=123)

    def test_is_bler_invalid_size(self):
        """Test that is_bler with wrong size raises ValueError."""
        snr = np.array([0, 2, 4])
        ber1 = np.array([0.1, 0.05, 0.01])
        ber2 = np.array([0.08, 0.04, 0.008])

        with pytest.raises(ValueError, match="is_bler has invalid size"):
            plot_ber([snr, snr], [ber1, ber2], is_bler=[True])  # Wrong length

    def test_is_bler_elements_must_be_bool(self):
        """List-valued BLER flags are type checked element-by-element."""
        snr = np.array([0, 1, 2])
        ber = np.array([0.1, 0.01, 0.001])
        with pytest.raises(TypeError, match="list of bool"):
            plot_ber(snr, ber, is_bler=[1])

    def test_curve_counts_must_match(self):
        """SNR, BER, and legend curve counts are validated."""
        snr = np.array([0, 1, 2])
        ber = np.array([0.1, 0.01, 0.001])
        with pytest.raises(ValueError, match="snr_db has invalid size"):
            plot_ber([snr], [ber, ber])
        with pytest.raises(ValueError, match="legend has invalid size"):
            plot_ber(snr, [ber, ber], legend=["one"])

    def test_curve_lengths_must_match(self):
        """Each SNR vector must match its corresponding error-rate vector."""
        with pytest.raises(ValueError, match="curve length"):
            plot_ber(np.arange(3), np.arange(4))


class TestPlotBERClass:
    """Tests for the PlotBER class."""

    def test_initialization(self):
        """Test PlotBER initialization with custom title."""
        ber_plot = PlotBER(title="Custom Title")
        assert ber_plot.title == "Custom Title"
        assert ber_plot.ber == []
        assert ber_plot.snr == []
        assert ber_plot.legend == []
        assert ber_plot.is_bler == []

    def test_invalid_title_type_init(self):
        """Test that invalid title type in init raises TypeError."""
        with pytest.raises(TypeError, match="title must be str"):
            PlotBER(title=123)

    def test_add_curve(self):
        """Test adding a curve to the PlotBER object."""
        ber_plot = PlotBER()
        snr = np.array([0, 2, 4, 6, 8])
        ber = np.array([0.2, 0.1, 0.05, 0.01, 0.001])

        ber_plot.add(snr, ber, legend="Test Curve")

        assert len(ber_plot.ber) == 1
        assert len(ber_plot.snr) == 1
        assert ber_plot.legend == ["Test Curve"]
        assert ber_plot.is_bler == [False]
        np.testing.assert_array_equal(ber_plot.ber[0], ber)
        np.testing.assert_array_equal(ber_plot.snr[0], snr)

    def test_add_bler_curve(self):
        """Test adding a BLER curve to the PlotBER object."""
        ber_plot = PlotBER()
        snr = np.array([0, 2, 4])
        bler = np.array([0.3, 0.15, 0.05])

        ber_plot.add(snr, bler, is_bler=True, legend="BLER Curve")

        assert ber_plot.is_bler == [True]
        assert ber_plot.legend == ["BLER Curve"]

    def test_add_mismatched_length(self):
        """Test that adding curves with mismatched lengths raises ValueError."""
        ber_plot = PlotBER()
        snr = np.array([0, 2, 4])
        ber = np.array([0.1, 0.05])  # Wrong length

        with pytest.raises(ValueError, match="same number of elements"):
            ber_plot.add(snr, ber)

    def test_add_invalid_legend_type(self):
        """Test that invalid legend type raises TypeError."""
        ber_plot = PlotBER()
        snr = np.array([0, 2, 4])
        ber = np.array([0.1, 0.05, 0.01])

        with pytest.raises(TypeError, match="legend must be str"):
            ber_plot.add(snr, ber, legend=123)

    def test_add_invalid_is_bler_type(self):
        """Test that invalid is_bler type raises TypeError."""
        ber_plot = PlotBER()
        snr = np.array([0, 2, 4])
        ber = np.array([0.1, 0.05, 0.01])

        with pytest.raises(TypeError, match="is_bler must be bool"):
            ber_plot.add(snr, ber, is_bler="yes")

    def test_reset(self):
        """Test that reset clears all internal data."""
        ber_plot = PlotBER()
        snr = np.array([0, 2, 4])
        ber = np.array([0.1, 0.05, 0.01])

        ber_plot.add(snr, ber, legend="Test")
        ber_plot.reset()

        assert ber_plot.ber == []
        assert ber_plot.snr == []
        assert ber_plot.legend == []
        assert ber_plot.is_bler == []

    def test_remove(self):
        """Test removing a curve by index."""
        ber_plot = PlotBER()
        snr1 = np.array([0, 2, 4])
        snr2 = np.array([0, 2, 4])
        ber1 = np.array([0.1, 0.05, 0.01])
        ber2 = np.array([0.08, 0.04, 0.008])

        ber_plot.add(snr1, ber1, legend="Curve 1")
        ber_plot.add(snr2, ber2, legend="Curve 2")

        ber_plot.remove(0)

        assert len(ber_plot.ber) == 1
        assert ber_plot.legend == ["Curve 2"]

    def test_remove_negative_index(self):
        """Test removing a curve with negative index."""
        ber_plot = PlotBER()
        snr1 = np.array([0, 2, 4])
        snr2 = np.array([0, 2, 4])
        ber1 = np.array([0.1, 0.05, 0.01])
        ber2 = np.array([0.08, 0.04, 0.008])

        ber_plot.add(snr1, ber1, legend="Curve 1")
        ber_plot.add(snr2, ber2, legend="Curve 2")

        ber_plot.remove(-1)

        assert len(ber_plot.ber) == 1
        assert ber_plot.legend == ["Curve 1"]

    def test_remove_invalid_type(self):
        """Test that invalid idx type raises TypeError."""
        ber_plot = PlotBER()
        snr = np.array([0, 2, 4])
        ber = np.array([0.1, 0.05, 0.01])
        ber_plot.add(snr, ber)

        with pytest.raises(TypeError, match="idx must be int"):
            ber_plot.remove("0")

    def test_title_setter(self):
        """Test setting the title property."""
        ber_plot = PlotBER()
        ber_plot.title = "New Title"
        assert ber_plot.title == "New Title"

    def test_title_setter_invalid_type(self):
        """Test that invalid title type in setter raises TypeError."""
        ber_plot = PlotBER()
        with pytest.raises(TypeError, match="title must be string"):
            ber_plot.title = 123

    def test_call_empty(self, tmp_path):
        """An empty PlotBER creates no figure and writes no file."""
        plt.close("all")
        out = tmp_path / "empty.png"
        ber_plot = PlotBER()
        ber_plot(save_fig=True, path=str(out))

        assert not out.exists()
        assert plt.get_fignums() == []

    def test_call_with_curves(self):
        """Calling PlotBER plots the stored curves."""
        ber_plot = PlotBER(title="Test Plot")
        snr = np.array([0, 2, 4, 6, 8])
        ber = np.array([0.2, 0.1, 0.05, 0.01, 0.001])

        ber_plot.add(snr, ber, legend="Test")
        ber_plot()

        ax = plt.gca()
        assert ax.get_title() == "Test Plot"
        assert len(ax.get_lines()) == 1
        legend = ax.get_legend()
        assert legend is not None
        assert [t.get_text() for t in legend.get_texts()] == ["Test"]
        plt.close("all")

    @pytest.mark.parametrize(
        "kwargs",
        [{}, {"legend": "new"}, {"is_bler": True}, {"legend": "new", "is_bler": True}],
    )
    def test_call_with_new_data_defaults(self, kwargs):
        """New curves may omit their label or BER/BLER flag."""
        ber_plot = PlotBER()
        snr = np.array([0, 2, 4])
        ber = np.array([0.2, 0.1, 0.01])

        ber_plot.add(snr, ber, legend="stored")
        ber_plot(snr_db=snr, ber=ber, **kwargs)
        plt.close("all")

    def test_call_rejects_oversized_legend(self):
        """A legend longer than the curve count is still an error."""
        ber_plot = PlotBER()
        snr = np.array([0, 2, 4])
        ber = np.array([0.2, 0.1, 0.01])

        ber_plot.add(snr, ber, legend="stored")
        with pytest.raises(ValueError, match="legend has invalid size"):
            ber_plot(snr_db=snr, ber=ber, legend=["a", "b", "c"])

    def test_call_show_ber_false(self):
        """show_ber=False keeps only BLER curves."""
        ber_plot = PlotBER()
        snr = np.array([0, 2, 4])
        ber = np.array([0.1, 0.05, 0.01])
        bler = np.array([0.3, 0.15, 0.05])

        ber_plot.add(snr, ber, legend="BER", is_bler=False)
        ber_plot.add(snr, bler, legend="BLER", is_bler=True)

        ber_plot(show_ber=False)

        ax = plt.gca()
        lines = ax.get_lines()
        assert len(lines) == 1
        assert lines[0].get_linestyle() == "--"
        legend = ax.get_legend()
        assert legend is not None
        assert [t.get_text() for t in legend.get_texts()] == ["BLER"]
        assert ax.get_ylabel() == "BLER"
        plt.close("all")

    def test_call_show_bler_false(self):
        """show_bler=False keeps only BER curves."""
        ber_plot = PlotBER()
        snr = np.array([0, 2, 4])
        ber = np.array([0.1, 0.05, 0.01])
        bler = np.array([0.3, 0.15, 0.05])

        ber_plot.add(snr, ber, legend="BER", is_bler=False)
        ber_plot.add(snr, bler, legend="BLER", is_bler=True)

        ber_plot(show_bler=False)

        ax = plt.gca()
        lines = ax.get_lines()
        assert len(lines) == 1
        assert lines[0].get_linestyle() == "-"
        legend = ax.get_legend()
        assert legend is not None
        assert [t.get_text() for t in legend.get_texts()] == ["BER"]
        assert ax.get_ylabel() == "BER"
        plt.close("all")

    def test_call_invalid_path_type(self):
        """Test that invalid path type raises TypeError."""
        ber_plot = PlotBER()
        with pytest.raises(TypeError, match="path must be str"):
            ber_plot(path=123)

    def test_call_invalid_save_fig_type(self):
        """Test that invalid save_fig type raises TypeError."""
        ber_plot = PlotBER()
        with pytest.raises(TypeError, match="save_fig must be bool"):
            ber_plot(save_fig="yes")

    def test_multiple_curves_same_snr(self):
        """Test adding multiple curves with the same SNR values."""
        ber_plot = PlotBER()
        snr = np.array([0, 2, 4, 6])
        ber1 = np.array([0.2, 0.1, 0.05, 0.01])
        ber2 = np.array([0.15, 0.08, 0.04, 0.008])
        ber3 = np.array([0.1, 0.05, 0.025, 0.005])

        ber_plot.add(snr, ber1, legend="Curve 1")
        ber_plot.add(snr, ber2, legend="Curve 2")
        ber_plot.add(snr, ber3, legend="Curve 3")

        assert len(ber_plot.ber) == 3
        ber_plot()
        plt.close("all")

    def test_docstring_example(self):
        """Test the example from the PlotBER docstring."""
        ber_plot = PlotBER(title="My BER Plot")
        snr = np.array([0, 2, 4, 6, 8])
        ber = np.array([0.1, 0.05, 0.01, 0.001, 0.0001])
        ber_plot.add(snr, ber, legend="Curve 1")
        ber_plot()
        plt.close("all")

        # Verify data was added correctly
        assert len(ber_plot.ber) == 1
        assert ber_plot.title == "My BER Plot"


class TestPlotBerDocstringExample:
    """Test the example from the plot_ber docstring."""

    def test_docstring_example(self):
        """Verify that the docstring example works correctly."""
        snr = np.array([0, 2, 4, 6, 8, 10])
        ber = np.array([0.2, 0.1, 0.05, 0.01, 0.001, 0.0001])
        fig, ax = plot_ber(snr, ber, legend="AWGN", title="BER vs SNR")

        assert fig is not None
        assert ax is not None
        assert ax.get_title() == "BER vs SNR"
        plt.close(fig)
