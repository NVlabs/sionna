#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Utility functions and blocks for the Polar code package."""

from typing import List, Tuple
import numbers
import numpy as np
from scipy.special import comb
from importlib_resources import files, as_file
from . import codes


__all__ = [
    "generate_5g_ranking",
    "generate_polar_transform_mat",
    "generate_rm_code",
    "generate_dense_polar",
]


def _is_pow2(x: int) -> bool:
    """True iff ``x`` is a positive power of two."""
    return x > 0 and (x & (x - 1)) == 0


def generate_5g_ranking(
    k: int, n: int, sort: bool = True
) -> List[np.ndarray]:
    """Returns information and frozen bit positions of the 5G Polar code
    as defined in Tab. 5.3.1.2-1 in :cite:p:`3GPPTS38212` for given values of ``k``
    and ``n``.

    :param k: The number of information bits per codeword.
    :param n: The desired codeword length. Must be a power of two.
    :param sort: Indicates if the returned indices are sorted.

    :output frozen_pos: Array of ints of shape `[n-k]` containing the
        frozen position indices.

    :output info_pos: Array of ints of shape `[k]` containing the
        information position indices.

    :raises TypeError: If ``k`` or ``n`` are not positive ints.
    :raises TypeError: If ``sort`` is not bool.
    :raises ValueError: If ``k`` or ``n`` are larger than 1024.
    :raises ValueError: If ``n`` is less than 32.
    :raises ValueError: If the resulting coderate is invalid (>1.0).
    :raises ValueError: If ``n`` is not a power of 2.

    .. rubric:: Examples


    .. code-block:: python

        from sionna.phy.fec.polar.utils import generate_5g_ranking

        frozen_pos, info_pos = generate_5g_ranking(k=100, n=256)
        print(f"Frozen positions: {len(frozen_pos)}, Info positions: {len(info_pos)}")
        # Frozen positions: 156, Info positions: 100
    """
    if not isinstance(k, int):
        raise TypeError("k must be integer.")
    if not isinstance(n, int):
        raise TypeError("n must be integer.")
    if not isinstance(sort, bool):
        raise TypeError("sort must be bool.")
    if k < 0:
        raise ValueError("k cannot be negative.")
    if k > 1024:
        raise ValueError("k cannot be larger than 1024.")
    if n > 1024:
        raise ValueError("n cannot be larger than 1024.")
    if n < 32:
        raise ValueError("n must be >=32.")
    if n < k:
        raise ValueError("Invalid coderate (>1).")
    if not _is_pow2(n):
        raise ValueError("n must be a power of 2.")

    # Load the channel ranking from csv format in folder "codes"
    source = files(codes).joinpath("polar_5G.csv")
    with as_file(source) as codes_csv:
        ch_order = np.genfromtxt(codes_csv, delimiter=";")
    ch_order = ch_order.astype(int)

    # Find n smallest values of channel order (2nd row)
    ind = np.argsort(ch_order[:, 1])
    ch_order_sort = ch_order[ind, :]
    # Only consider the first n channels
    ch_order_sort_n = ch_order_sort[0:n, :]
    # And sort again according to reliability
    ind_n = np.argsort(ch_order_sort_n[:, 0])
    ch_order_n = ch_order_sort_n[ind_n, :]

    # Calculate frozen/information positions for given n, k
    frozen_pos = np.zeros(n - k)
    info_pos = np.zeros(k)
    # The n-k smallest positions of ch_order denote frozen pos.
    for i in range(n - k):
        frozen_pos[i] = ch_order_n[i, 1]  # 2nd row yields index to freeze
    for i in range(n - k, n):
        info_pos[i - (n - k)] = ch_order_n[i, 1]  # 2nd row yields index

    # Sort to have channels in ascending order
    if sort:
        info_pos = np.sort(info_pos)
        frozen_pos = np.sort(frozen_pos)

    return [frozen_pos.astype(int), info_pos.astype(int)]


def generate_polar_transform_mat(n_lift: int) -> np.ndarray:
    """Generate the polar transformation matrix (Kronecker product).

    :param n_lift: Defining the Kronecker power, i.e., how often the kernel
        is lifted.

    :output gm: Array of `0s` and `1s` of shape `[2^n_lift, 2^n_lift]`
        containing the Polar transformation matrix.

    :raises ValueError: If ``n_lift`` is not integer or negative.
    :raises ValueError: If ``n_lift`` >= 20 (code length too large).

    .. rubric:: Examples


    .. code-block:: python

        from sionna.phy.fec.polar.utils import generate_polar_transform_mat

        gm = generate_polar_transform_mat(3)
        print(gm.shape)
        # (8, 8)
    """
    if int(n_lift) != n_lift:
        raise ValueError("n_lift must be integer.")
    if n_lift < 0:
        raise ValueError("n_lift must be positive.")
    if n_lift >= 20:
        msg = "Warning: the resulting code length is large (=2^n_lift)."
        raise ValueError(msg)

    gm = np.array([[1, 0], [1, 1]])

    gm_l = np.copy(gm)
    for _ in range(n_lift - 1):
        gm_l_new = np.zeros([2 * np.shape(gm_l)[0], 2 * np.shape(gm_l)[1]])
        for j in range(np.shape(gm_l)[0]):
            for k in range(np.shape(gm_l)[1]):
                gm_l_new[2 * j : 2 * j + 2, 2 * k : 2 * k + 2] = gm_l[j, k] * gm
        gm_l = gm_l_new
    return gm_l


def generate_rm_code(
    r: int, m: int
) -> Tuple[np.ndarray, np.ndarray, int, int, int]:
    """Generate frozen positions of the (r, m) Reed Muller (RM) code.

    :param r: The order of the RM code.
    :param m: `log2` of the desired codeword length.

    :output frozen_pos: Array of ints of shape `[n-k]` containing the
        frozen position indices.

    :output info_pos: Array of ints of shape `[k]` containing the
        information position indices.

    :output n: The resulting codeword length.

    :output k: The number of information bits.

    :output d_min: The minimum distance of the code.

    :raises TypeError: If ``r`` or ``m`` are not int.
    :raises ValueError: If ``r`` > ``m`` or ``r`` < 0 or ``m`` < 0.

    .. rubric:: Examples


    .. code-block:: python

        from sionna.phy.fec.polar.utils import generate_rm_code

        frozen_pos, info_pos, n, k, d_min = generate_rm_code(r=1, m=4)
        print(f"n={n}, k={k}, d_min={d_min}")
        # n=16, k=5, d_min=8
    """
    if not isinstance(r, int):
        raise TypeError("r must be int.")
    if not isinstance(m, int):
        raise TypeError("m must be int.")
    if r > m:
        raise ValueError("order r cannot be larger than m.")
    if r < 0:
        raise ValueError("r must be positive.")
    if m < 0:
        raise ValueError("m must be positive.")

    n = 2**m
    d_min = 2 ** (m - r)

    # Calc k to verify results
    k = 0
    for i in range(r + 1):
        k += int(comb(m, i))

    # Select positions to freeze
    # Freeze all rows that have weight < m-r
    w = np.zeros(n)
    for i in range(n):
        x_bin = np.binary_repr(i)
        for x_i in x_bin:
            w[i] += int(x_i)
    frozen_vec = w < m - r
    info_vec = np.invert(frozen_vec)
    k_res = np.sum(info_vec)
    frozen_pos = np.arange(n)[frozen_vec]
    info_pos = np.arange(n)[info_vec]

    # Verify results
    if k_res != k:
        raise ValueError("Error: resulting k is inconsistent.")

    return frozen_pos, info_pos, n, k, d_min


def generate_dense_polar(
    frozen_pos: np.ndarray, n: int, verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate *naive* (dense) Polar parity-check and generator matrix.

    This function follows Lemma 1 in :cite:p:`Goala_LP` and returns a parity-check
    matrix for Polar codes.

    .. rubric:: Notes

    The resulting matrix can be used for decoding with the
    :class:`~sionna.phy.fec.ldpc.LDPCBPDecoder` class. However, the resulting
    parity-check matrix is (usually) not sparse and, thus, not suitable for
    belief propagation decoding as the graph has many short cycles.
    Please consider :class:`~sionna.phy.fec.polar.PolarBPDecoder` for iterative
    decoding over the encoding graph.

    :param frozen_pos: Array of `int` defining the ``n-k`` indices of the
        frozen positions.
    :param n: The codeword length.
    :param verbose: If `True`, the code properties are printed.

    :output pcm: The parity-check matrix of shape [n-k, n].

    :output gm: The generator matrix of shape [k, n].

    :raises TypeError: If ``n`` is not a number.
    :raises TypeError: If ``frozen_pos`` does not consist of ints.
    :raises ValueError: If number of elements in frozen_pos > n.
    :raises ValueError: If ``n`` is not a power of 2.

    .. rubric:: Examples


    .. code-block:: python

        from sionna.phy.fec.polar.utils import generate_5g_ranking, generate_dense_polar

        frozen_pos, _ = generate_5g_ranking(k=32, n=64)
        pcm, gm = generate_dense_polar(frozen_pos, n=64, verbose=False)
        print(f"PCM shape: {pcm.shape}, GM shape: {gm.shape}")
        # PCM shape: (32, 64), GM shape: (32, 64)
    """
    if not isinstance(n, numbers.Number):
        raise TypeError("n must be a number.")
    n = int(n)  # n can be float (e.g. as result of n=k*r)
    if not np.issubdtype(frozen_pos.dtype, int):
        raise TypeError("frozen_pos must consist of ints.")
    if len(frozen_pos) > n:
        msg = "Number of elements in frozen_pos cannot be greater than n."
        raise ValueError(msg)

    if not _is_pow2(n):
        raise ValueError("n must be a power of 2.")

    k = n - len(frozen_pos)

    # Generate info positions
    info_pos = np.setdiff1d(np.arange(n), frozen_pos)
    if k != len(info_pos):
        raise ArithmeticError("Internal error: invalid info_pos generated.")

    gm_mat = generate_polar_transform_mat(int(np.log2(n)))

    gm_true = gm_mat[info_pos, :]
    pcm = np.transpose(gm_mat[:, frozen_pos])

    if verbose:
        print("Shape of the generator matrix: ", gm_true.shape)
        print("Shape of the parity-check matrix: ", pcm.shape)

    # Verify result, i.e., check that H*G has an all-zero syndrome.
    s = np.mod(np.matmul(pcm, np.transpose(gm_true)), 2)
    if np.sum(s) != 0:
        raise ArithmeticError("Non-zero syndrome for H*G'.")

    return pcm, gm_true



