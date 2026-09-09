#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""EXIT chart visualization utilities for the FEC package."""

from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from sionna.phy.fec.utils import j_fun, j_fun_inv


__all__ = [
    "plot_trajectory",
    "plot_exit_chart",
    "get_exit_analytic",
]


def plot_trajectory(
    plot,
    mi_v: np.ndarray,
    mi_c: np.ndarray,
    ebno: Optional[float] = None,
) -> None:
    """Plots the trajectory of an EXIT-chart.

    This utility function plots the trajectory of mutual information values in
    an EXIT-chart, based on variable and check node mutual information values.

    :param plot: A matplotlib figure handle where the trajectory will be plotted.
    :param mi_v: Array of floats representing the variable node mutual
        information values.
    :param mi_c: Array of floats representing the check node mutual
        information values.
    :param ebno: The Eb/No value in dB, used for the legend entry.

    .. note::
        This function does not return a value; it modifies the provided
        ``plot`` figure in-place.
    """
    if len(mi_v) != len(mi_c):
        raise ValueError("mi_v and mi_c must have same length.")

    # number of decoding iterations to plot
    iters = np.shape(mi_v)[0] - 1

    x = np.zeros([2 * iters])
    y = np.zeros([2 * iters])

    # iterate between VN and CN MI value
    y[1] = mi_v[0]
    for i in range(1, iters):
        x[2 * i] = mi_c[i - 1]
        y[2 * i] = mi_v[i - 1]
        x[2 * i + 1] = mi_c[i - 1]
        y[2 * i + 1] = mi_v[i]

    label_str = "Actual trajectory"

    if ebno is not None:
        label_str += f" @ {ebno} dB"

    # plot trajectory
    plot.plot(x, y, "-", linewidth=3, color="g", label=label_str)

    # and show the legend
    plot.legend(fontsize=18)


def plot_exit_chart(
    mi_a: Optional[np.ndarray] = None,
    mi_ev: Optional[np.ndarray] = None,
    mi_ec: Optional[np.ndarray] = None,
    title: str = "EXIT-Chart",
):
    """Plots an EXIT-chart based on mutual information curves :cite:p:`tenBrinkEXIT`.

    This utility function generates an EXIT-chart plot. If all inputs are
    `None`, an empty EXIT chart is created; otherwise, mutual information
    curves are plotted.

    :param mi_a: Array of floats representing the a priori mutual information.
    :param mi_ev: Array of floats representing the variable node mutual
        information.
    :param mi_ec: Array of floats representing the check node mutual information.
    :param title: Title of the EXIT chart.

    :output plot: A handle to the generated matplotlib figure.
    """
    if not isinstance(title, str):
        raise TypeError("title must be a string.")

    if not (mi_ev is None and mi_ec is None):
        if mi_a is None:
            raise ValueError("mi_a cannot be None if mi_e is provided.")

    if mi_ev is not None:
        if len(mi_a) != len(mi_ev):
            raise ValueError("mi_a and mi_ev must have same length.")
    if mi_ec is not None:
        if len(mi_a) != len(mi_ec):
            raise ValueError("mi_a and mi_ec must have same length.")

    plt.figure(figsize=(10, 10))
    plt.title(title, fontsize=25)
    plt.xlabel("$I_{a}^v$, $I_{e}^c$", fontsize=25)
    plt.ylabel("$I_{e}^v$, $I_{a}^c$", fontsize=25)
    plt.grid(visible=True, which="major")

    # for MI, the x,y limits are always (0,1)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    # and plot EXIT curves
    if mi_ec is not None:
        plt.plot(mi_ec, mi_a, "r", linewidth=3, label="Check node")
        plt.legend()
    if mi_ev is not None:
        plt.plot(mi_a, mi_ev, "b", linewidth=3, label="Variable node")
        plt.legend()
    return plt


def get_exit_analytic(
    pcm: np.ndarray,
    ebno_db: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculates analytic EXIT curves for a given parity-check matrix.

    This function extracts the degree profile from the provided parity-check
    matrix ``pcm`` and calculates the EXIT (Extrinsic Information Transfer)
    curves for variable nodes (VN) and check nodes (CN) decoders. Note that this
    approach relies on asymptotic analysis, which requires a sufficiently large
    codeword length for accurate predictions.

    It assumes transmission over an AWGN channel with BPSK modulation at an
    SNR specified by ``ebno_db``. For more details on the equations, see
    :cite:p:`tenBrink` and :cite:p:`tenBrinkEXIT`.

    :param pcm: The parity-check matrix.
    :param ebno_db: Channel SNR in dB.

    :output mi_a: The a priori mutual information.

    :output mi_ev: The extrinsic mutual information of the variable node
        decoder.

    :output mi_ec: The extrinsic mutual information of the check node
        decoder.

    .. rubric:: Notes

    This function assumes random, unstructured parity-check matrices. Thus,
    applying it to parity-check matrices with specific structures or constraints
    may result in inaccurate EXIT predictions. Additionally, this function is
    based on asymptotic properties and performs best with large parity-check
    matrices. For more information, refer to :cite:p:`tenBrink`.
    """
    # calc coderate
    n = pcm.shape[1]
    k = n - pcm.shape[0]
    coderate = k / n

    # calc mean and noise_var of Gaussian distributed LLRs for given channel SNR
    ebno = 10 ** (ebno_db / 10)
    snr = ebno * coderate
    noise_var = 1 / (2 * snr)

    # For BiAWGN channels the LLRs follow a Gaussian distr. as given below
    sigma_llr = np.sqrt(4 / noise_var)
    mu_llr = sigma_llr**2 / 2

    # calculate max node degree
    # "+1" as the array indices later directly denote the node degrees
    c_max = int(np.max(np.sum(pcm, axis=1)) + 1)
    v_max = int(np.max(np.sum(pcm, axis=0)) + 1)

    # calculate degree profile (node perspective)
    c = np.histogram(
        np.sum(pcm, axis=1), bins=c_max, range=(0, c_max), density=False
    )[0]

    v = np.histogram(
        np.sum(pcm, axis=0), bins=v_max, range=(0, v_max), density=False
    )[0]

    # calculate degrees from edge perspective
    r = np.zeros([c_max])
    for i in range(1, c_max):
        r[i] = (i - 1) * c[i]
    r = r / np.sum(r)
    lam = np.zeros([v_max])
    for i in range(1, v_max):
        lam[i] = (i - 1) * v[i]
    lam = lam / np.sum(lam)

    mi_a = np.arange(0.002, 0.998, 0.001)  # quantize Ia with 0.001 resolution

    # Exit function of check node update
    mi_ec = np.zeros_like(mi_a)
    for i in range(1, c_max):
        # Convert to tensor, compute, convert back
        mi_a_tensor = torch.from_numpy(1 - mi_a)
        j_inv_result = j_fun_inv(mi_a_tensor)
        j_result = j_fun((i - 1.0) * j_inv_result)
        mi_ec += r[i] * j_result.numpy()
    mi_ec = 1 - mi_ec

    # Exit function of variable node update
    mi_ev = np.zeros_like(mi_a)
    for i in range(1, v_max):
        mi_a_tensor = torch.from_numpy(mi_a)
        j_inv_result = j_fun_inv(mi_a_tensor)
        j_result = j_fun(mu_llr + (i - 1.0) * j_inv_result)
        mi_ev += lam[i] * j_result.numpy()

    return mi_a, mi_ev, mi_ec
