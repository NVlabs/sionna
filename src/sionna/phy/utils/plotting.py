#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Plotting functions for Sionna PHY"""

from itertools import compress
from typing import Callable, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch

from sionna.phy.utils import sim_ber

__all__ = ["plot_ber", "PlotBER"]


def _normalize_curves(snr_db, ber):
    """Normalize single and multiple error-rate curves."""
    ber_curves = ber if isinstance(ber, list) else [ber]
    if not ber_curves:
        raise ValueError("ber must contain at least one curve.")

    num_curves = len(ber_curves)
    snr_curves = snr_db if isinstance(snr_db, list) else [snr_db] * num_curves
    if len(snr_curves) != num_curves:
        raise ValueError("snr_db has invalid size.")
    for snr_curve, ber_curve in zip(snr_curves, ber_curves):
        if len(snr_curve) != len(ber_curve):
            raise ValueError("Each SNR curve must match its BER curve length.")
    return snr_curves, ber_curves


def _normalize_legend(legend, num_curves):
    """Normalize and validate curve labels."""
    if isinstance(legend, str):
        return [legend] * num_curves
    if not isinstance(legend, list):
        raise TypeError("legend must be str or list of str.")
    if len(legend) != num_curves:
        raise ValueError("legend has invalid size.")
    if not all(isinstance(entry, str) for entry in legend):
        raise TypeError("legend must be str or list of str.")
    return legend


def _normalize_is_bler(is_bler, num_curves):
    """Normalize and validate BER/BLER curve flags."""
    if is_bler is None:
        return [False] * num_curves
    if isinstance(is_bler, bool):
        return [is_bler] * num_curves
    if not isinstance(is_bler, list):
        raise TypeError("is_bler must be bool or list of bool.")
    if len(is_bler) != num_curves:
        raise ValueError("is_bler has invalid size.")
    if not all(isinstance(value, bool) for value in is_bler):
        raise TypeError("is_bler must be bool or list of bool.")
    return is_bler


def plot_ber(
    snr_db: Union[np.ndarray, List[np.ndarray]],
    ber: Union[np.ndarray, List[np.ndarray]],
    legend: Union[str, List[str]] = "",
    ylabel: str = "BER",
    title: str = "Bit Error Rate",
    ebno: bool = True,
    is_bler: Optional[Union[bool, List[bool]]] = None,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    save_fig: bool = False,
    path: str = "",
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot error-rates.

    :param snr_db: Array defining the simulated SNR points.
    :param ber: Array defining the BER/BLER per SNR point.
    :param legend: Legend entries.
    :param ylabel: Label for the y-axis.
    :param title: Figure title.
    :param ebno: If `True`, the x-label is set to
        "EbNo [dB]" instead of "EsNo [dB]".
    :param is_bler: If `True`, the corresponding curve is dashed.
    :param xlim: x-axis limits.
    :param ylim: y-axis limits.
    :param save_fig: If `True`, the figure is saved as `.png`.
    :param path: Path to save the figure if ``save_fig`` is `True`.

    :output fig: `matplotlib.figure.Figure`.
        Figure handle.

    :output ax: `matplotlib.axes.Axes`.
        Axes object.

    .. rubric:: Examples

    .. code-block:: python

        import numpy as np
        from sionna.phy.utils import plot_ber

        snr = np.array([0, 2, 4, 6, 8, 10])
        ber = np.array([0.2, 0.1, 0.05, 0.01, 0.001, 0.0001])
        fig, ax = plot_ber(snr, ber, legend="AWGN", title="BER vs SNR")
    """
    if not isinstance(title, str):
        raise TypeError("title must be str.")

    snr_curves, ber_curves = _normalize_curves(snr_db, ber)
    num_curves = len(ber_curves)
    legend = _normalize_legend(legend, num_curves)
    is_bler = _normalize_is_bler(is_bler, num_curves)

    fig, ax = plt.subplots(figsize=(16, 10))

    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)

    plt.title(title, fontsize=25)

    for snr_curve, ber_curve, curve_is_bler in zip(
        snr_curves, ber_curves, is_bler
    ):
        line_style = "--" if curve_is_bler else ""
        plt.semilogy(snr_curve, ber_curve, line_style, linewidth=2)

    plt.grid(which="both")
    if ebno:
        plt.xlabel(r"$E_b/N_0$ (dB)", fontsize=25)
    else:
        plt.xlabel(r"$E_s/N_0$ (dB)", fontsize=25)
    plt.ylabel(ylabel, fontsize=25)
    # Skip the legend rather than drawing an empty box for blank labels.
    if any(legend):
        plt.legend(legend, fontsize=20)

    if save_fig:
        plt.savefig(path)
        plt.close(fig)

    return fig, ax


class PlotBER:
    """Provides a plotting object to simulate and store BER/BLER curves.

    :param title: Figure title.

    :input snr_db: `numpy.ndarray` or `list` of `numpy.ndarray`.
        SNR values.

    :input ber: `numpy.ndarray` or `list` of `numpy.ndarray`.
        BER values corresponding to ``snr_db``.

    :input legend: `str` or `list` of `str`.
        Legend entries.

    :input is_bler: `bool` or `list` of `bool`.
        If `True`, ``ber`` will be interpreted as BLER.

    :input show_ber: `bool`.
        If `True`, BER curves will be plotted.

    :input show_bler: `bool`.
        If `True`, BLER curves will be plotted.

    :input xlim: `None` | (`float`, `float`).
        x-axis limits.

    :input ylim: `None` | (`float`, `float`).
        y-axis limits.

    :input save_fig: `bool`.
        If `True`, the figure is saved as `.png`.

    :input path: `str`.
        Path to save the figure if ``save_fig`` is `True`.

    .. rubric:: Examples

    .. code-block:: python

        import numpy as np
        from sionna.phy.utils import PlotBER

        ber_plot = PlotBER(title="My BER Plot")
        snr = np.array([0, 2, 4, 6, 8])
        ber = np.array([0.1, 0.05, 0.01, 0.001, 0.0001])
        ber_plot.add(snr, ber, legend="Curve 1")
        ber_plot()  # Display the plot
    """

    def __init__(self, title: str = "Bit/Block Error Rate"):
        if not isinstance(title, str):
            raise TypeError("title must be str.")
        self._title = title

        # init lists
        self._bers: List[np.ndarray] = []
        self._snrs: List[np.ndarray] = []
        self._legends: List[str] = []
        self._is_bler: List[bool] = []

    def __call__(
        self,
        snr_db: Union[np.ndarray, List[np.ndarray], float] = None,
        ber: Union[np.ndarray, List[np.ndarray], float] = None,
        legend: Union[str, List[str]] = None,
        is_bler: Union[bool, List[bool]] = None,
        show_ber: bool = True,
        show_bler: bool = True,
        xlim: Optional[Tuple[float, float]] = None,
        ylim: Optional[Tuple[float, float]] = None,
        save_fig: bool = False,
        path: str = "",
    ) -> None:
        """Plot BER curves."""
        # Handle None defaults
        if snr_db is None:
            snr_db = []
        if ber is None:
            ber = []
        if legend is None:
            legend = []
        if is_bler is None:
            is_bler = []

        if not isinstance(path, str):
            raise TypeError("path must be str.")
        if not isinstance(save_fig, bool):
            raise TypeError("save_fig must be bool.")

        # broadcast snr if ber is list
        if isinstance(ber, list):
            if not isinstance(snr_db, list):
                snr_db = [snr_db] * len(ber)

        if not isinstance(snr_db, list):
            snrs = self._snrs + [snr_db]
        else:
            snrs = self._snrs + snr_db
        if not isinstance(ber, list):
            bers = self._bers + [ber]
        else:
            bers = self._bers + ber
        if not isinstance(legend, list):
            legends = self._legends + [legend]
        else:
            legends = self._legends + legend
        if not isinstance(is_bler, list):
            is_bler_list = self._is_bler + [is_bler]
        else:
            is_bler_list = self._is_bler + is_bler

        # Curves supplied without a label or a BER/BLER flag get a default one,
        # so that omitting either argument is not a size mismatch.
        legends += [""] * (len(bers) - len(legends))
        is_bler_list += [False] * (len(bers) - len(is_bler_list))

        # deactivate BER/BLER
        if len(is_bler_list) > 0:
            if show_ber is False:
                snrs = list(compress(snrs, is_bler_list))
                bers = list(compress(bers, is_bler_list))
                legends = list(compress(legends, is_bler_list))
                is_bler_list = list(compress(is_bler_list, is_bler_list))

            if show_bler is False:
                inverted = np.invert(is_bler_list)
                snrs = list(compress(snrs, inverted))
                bers = list(compress(bers, inverted))
                legends = list(compress(legends, inverted))
                is_bler_list = list(compress(is_bler_list, inverted))

        # set ylabel
        ylabel = "BER / BLER"
        if len(is_bler_list) > 0:
            if np.all(is_bler_list):
                ylabel = "BLER"
            if not np.any(is_bler_list):
                ylabel = "BER"

        # plot the results
        if len(bers) > 0:
            plot_ber(
                snr_db=snrs,
                ber=bers,
                legend=legends,
                is_bler=is_bler_list,
                title=self._title,
                ylabel=ylabel,
                xlim=xlim,
                ylim=ylim,
                save_fig=save_fig,
                path=path,
            )

    @property
    def title(self) -> str:
        """Get/set title of the plot."""
        return self._title

    @title.setter
    def title(self, title: str) -> None:
        if not isinstance(title, str):
            raise TypeError("title must be string.")
        self._title = title

    @property
    def ber(self) -> List[np.ndarray]:
        """Stored BER/BLER values."""
        return self._bers

    @property
    def snr(self) -> List[np.ndarray]:
        """Stored SNR values."""
        return self._snrs

    @property
    def legend(self) -> List[str]:
        """Legend entries."""
        return self._legends

    @property
    def is_bler(self) -> List[bool]:
        """Indicates if a curve shall be interpreted as BLER."""
        return self._is_bler

    def simulate(
        self,
        mc_fun: Callable,
        ebno_dbs: torch.Tensor,
        batch_size: int,
        max_mc_iter: int,
        legend: str = "",
        add_ber: bool = True,
        add_bler: bool = False,
        soft_estimates: bool = False,
        num_target_bit_errors: Optional[int] = None,
        num_target_block_errors: Optional[int] = None,
        target_ber: Optional[float] = None,
        target_bler: Optional[float] = None,
        early_stop: bool = True,
        compile_mode: Optional[str] = None,
        add_results: bool = True,
        forward_keyboard_interrupt: bool = True,
        show_fig: bool = True,
        verbose: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Simulate BER/BLER curves for a given model and saves the results.

        Internally calls :func:`sionna.phy.utils.sim_ber`.

        :param mc_fun: Callable that yields the transmitted bits `b` and the
            receiver's estimate `b_hat` for a given ``batch_size`` and
            ``ebno_db``. If ``soft_estimates`` is `True`, `b_hat` is
            interpreted as logit.
        :param ebno_dbs: SNR points to be evaluated.
        :param batch_size: Batch-size for evaluation.
        :param max_mc_iter: Max. number of Monte-Carlo iterations per
            SNR point.
        :param legend: Name to appear in legend.
        :param add_ber: If `True`, BER will be added to plot.
        :param add_bler: If `True`, BLER will be added to plot.
        :param soft_estimates: If `True`, ``b_hat`` is interpreted as logit
            and additional hard-decision is applied internally.
        :param num_target_bit_errors: Target number of bit errors per SNR
            point until the simulation stops.
        :param num_target_block_errors: Target number of block errors per
            SNR point until the simulation stops.
        :param target_ber: The simulation stops after the first SNR point
            which achieves a lower bit error rate as specified by
            ``target_ber``. This requires ``early_stop`` to be `True`.
        :param target_bler: The simulation stops after the first SNR point
            which achieves a lower block error rate as specified by
            ``target_bler``. This requires ``early_stop`` to be `True`.
        :param early_stop: If `True`, the simulation stops after the first
            error-free SNR point (i.e., no error occurred after
            ``max_mc_iter`` Monte-Carlo iterations).
        :param compile_mode: Compilation mode for ``mc_fun``.
            If `None`, ``mc_fun`` is executed as is.
            Options: `None`, ``"default"``, ``"reduce-overhead"``,
            ``"max-autotune"``.
        :param add_results: If `True`, the simulation results will be
            appended to the internal list of results.
        :param forward_keyboard_interrupt: If `False`, `KeyboardInterrupts`
            will be caught internally and not forwarded (e.g., will not stop
            outer loops). If `True`, the simulation ends and returns the
            intermediate simulation results.
        :param show_fig: If `True`, a BER figure will be plotted.
        :param verbose: If `True`, the current progress will be printed.

        :output ber: `torch.float`.
            Simulated bit-error rates.

        :output bler: `torch.float`.
            Simulated block-error rates.
        """
        ber, bler = sim_ber(
            mc_fun,
            ebno_dbs,
            batch_size,
            soft_estimates=soft_estimates,
            max_mc_iter=max_mc_iter,
            num_target_bit_errors=num_target_bit_errors,
            num_target_block_errors=num_target_block_errors,
            target_ber=target_ber,
            target_bler=target_bler,
            early_stop=early_stop,
            compile_mode=compile_mode,
            verbose=verbose,
            forward_keyboard_interrupt=forward_keyboard_interrupt,
        )

        # Convert to numpy for storage
        ber_np = ber.cpu().numpy()
        bler_np = bler.cpu().numpy()
        ebno_np = (
            ebno_dbs.cpu().numpy() if isinstance(ebno_dbs, torch.Tensor) else ebno_dbs
        )

        if add_ber:
            self._bers.append(ber_np)
            self._snrs.append(ebno_np)
            self._legends.append(legend)
            self._is_bler.append(False)

        if add_bler:
            self._bers.append(bler_np)
            self._snrs.append(ebno_np)
            self._legends.append(legend + " (BLER)")
            self._is_bler.append(True)

        if show_fig:
            self()

        # remove current curve if add_results=False
        if add_results is False:
            if add_bler:
                self.remove(-1)
            if add_ber:
                self.remove(-1)

        return ber, bler

    def add(
        self,
        ebno_db: np.ndarray,
        ber: np.ndarray,
        is_bler: bool = False,
        legend: str = "",
    ) -> None:
        """Add static reference curves.

        :param ebno_db: SNR points.
        :param ber: BER corresponding to each SNR point.
        :param is_bler: If `True`, ``ber`` is interpreted as BLER.
        :param legend: Legend entry.
        """
        if len(ebno_db) != len(ber):
            raise ValueError("ebno_db and ber must have same number of elements.")
        if not isinstance(legend, str):
            raise TypeError("legend must be str.")
        if not isinstance(is_bler, bool):
            raise TypeError("is_bler must be bool.")

        self._bers.append(ber)
        self._snrs.append(ebno_db)
        self._legends.append(legend)
        self._is_bler.append(is_bler)

    def reset(self) -> None:
        """Remove all internal data."""
        self._bers = []
        self._snrs = []
        self._legends = []
        self._is_bler = []

    def remove(self, idx: int = -1) -> None:
        """Remove curve with index ``idx``.

        :param idx: Index of the dataset that should be removed. Negative
            indexing is possible.
        """
        if not isinstance(idx, int):
            raise TypeError("idx must be int.")

        del self._bers[idx]
        del self._snrs[idx]
        del self._legends[idx]
        del self._is_bler[idx]
