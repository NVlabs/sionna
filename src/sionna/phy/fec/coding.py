#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Code construction, matrix, and I/O utilities for the FEC package."""

from typing import List, Tuple
import warnings

import numpy as np
from importlib_resources import files, as_file

from sionna.phy import config
from sionna.phy.fec.ldpc import codes


__all__ = [
    "load_parity_check_examples",
    "alist2mat",
    "load_alist",
    "make_systematic",
    "gm2pcm",
    "pcm2gm",
    "verify_gm_pcm",
    "generate_reg_ldpc",
]


def load_parity_check_examples(
    pcm_id: int,
    verbose: bool = False,
) -> Tuple[np.ndarray, int, int, float]:
    """Loads parity-check matrices of built-in example codes.

    This utility function loads predefined example codes, including Hamming,
    BCH, and LDPC codes. The following codes are available:

    - ``pcm_id`` =0 : `(7,4)` Hamming code with `k=4` information bits and `n=7` codeword length.
    - ``pcm_id`` =1 : `(63,45)` BCH code with `k=45` information bits and `n=63` codeword length.
    - ``pcm_id`` =2 : `(127,106)` BCH code with `k=106` information bits and `n=127` codeword length.
    - ``pcm_id`` =3 : Random LDPC code with variable node degree 3 and check node degree 6, with `k=50` information bits and `n=100` codeword length.
    - ``pcm_id`` =4 : 802.11n LDPC code with `k=324` information bits and `n=648` codeword length.

    :param pcm_id: An integer identifying the code matrix to load.
    :param verbose: If `True`, prints the code parameters.

    :output pcm: The parity-check matrix (values are `0` and `1`).

    :output k: The number of information bits.

    :output n: The number of codeword bits.

    :output coderate: The code rate, assuming full rank of the
        parity-check matrix.

    .. rubric:: Examples


    .. code-block:: python

        from sionna.phy.fec.coding import load_parity_check_examples

        pcm, k, n, coderate = load_parity_check_examples(0)
        print(f"n={n}, k={k}, rate={coderate:.3f}")
    """
    source = files(codes).joinpath("example_codes.npz")
    with as_file(source) as code:
        with np.load(code) as pcms:
            pcm = np.array(pcms[f"pcm_{pcm_id}"])
    n = int(pcm.shape[1])  # number of codeword bits (codeword length)
    k = int(n - pcm.shape[0])  # number of information bits k per codeword
    coderate = k / n

    if verbose:
        print(f"\nn: {n}, k: {k}, coderate: {coderate:.3f}")
    return pcm, k, n, coderate


def alist2mat(
    alist: List[List[int]],
    verbose: bool = True,
) -> Tuple[np.ndarray, int, int, float]:
    r"""Converts an `alist` :cite:p:`MacKay` code definition to a NumPy parity-check matrix.

    This function converts an `alist` format representation of a code's
    parity-check matrix to a NumPy array. Many example codes in `alist`
    format can be found in :cite:p:`UniKL`.

    About the `alist` format (see :cite:p:`MacKay` for details):

        - Row 1: Defines the parity-check matrix dimensions `m x n`.
        - Row 2: Contains two integers, `max_CN_degree` and `max_VN_degree`.
        - Row 3: Lists the degrees of all `n` variable nodes (columns).
        - Row 4: Lists the degrees of all `m` check nodes (rows).
        - Next `n` rows: Non-zero entries of each column, zero-padded as needed.
        - Following `m` rows: Non-zero entries of each row, zero-padded as needed.

    :param alist: Nested list in `alist` format :cite:p:`MacKay` representing the
        parity-check matrix.
    :param verbose: If `True`, prints the code parameters.

    :output pcm: The parity-check matrix.

    :output k: The number of information bits.

    :output n: The number of codeword bits.

    :output coderate: The code rate.

    .. rubric:: Notes

    Use :func:`~sionna.phy.fec.coding.load_alist` to import an `alist` from a
    text file.

    .. rubric:: Examples


    .. code-block:: python

        from sionna.phy.fec.coding import load_alist, alist2mat

        al = load_alist(path="code.alist")
        pcm, k, n, coderate = alist2mat(al)
    """
    if len(alist) <= 4:
        raise ValueError("Invalid alist format.")

    n = alist[0][0]
    m = alist[0][1]
    v_max = alist[1][0]
    c_max = alist[1][1]
    k = n - m
    coderate = k / n

    vn_profile = alist[2]
    cn_profile = alist[3]

    # plausibility checks
    if np.sum(vn_profile) != np.sum(cn_profile):
        raise ValueError("Invalid alist format.")
    if np.max(vn_profile) != v_max:
        raise ValueError("Invalid alist format.")
    if np.max(cn_profile) != c_max:
        raise ValueError("Invalid alist format.")

    if len(alist) == len(vn_profile) + 4:
        warnings.warn(
            ".alist does not contain (redundant) CN perspective. "
            "Recovering parity-check matrix from VN only. "
            "Please verify the correctness of the results manually.",
            UserWarning,
            stacklevel=2,
        )
        vn_only = True
    else:
        if len(alist) != len(vn_profile) + len(cn_profile) + 4:
            raise ValueError("Invalid alist format.")
        vn_only = False

    pcm = np.zeros((m, n))
    num_edges = 0  # count number of edges

    for idx_v in range(n):
        for idx_i in range(vn_profile[idx_v]):
            # first 4 rows of alist contain meta information
            idx_c = alist[4 + idx_v][idx_i] - 1  # "-1" as this is python
            pcm[idx_c, idx_v] = 1
            num_edges += 1  # count number of edges (=each non-zero entry)

    # validate results from CN perspective
    if not vn_only:
        for idx_c in range(m):
            for idx_i in range(cn_profile[idx_c]):
                # first 4 rows of alist contain meta information
                # following n rows contained VN perspective
                idx_v = alist[4 + n + idx_c][idx_i] - 1  # "-1" as this is python
                if pcm[idx_c, idx_v] != 1:
                    raise ValueError(
                        "Inconsistent alist: CN and VN perspectives disagree."
                    )

    if verbose:
        print("Number of variable nodes (columns): ", n)
        print("Number of check nodes (rows): ", m)
        print("Number of information bits per cw: ", k)
        print("Number edges: ", num_edges)
        print("Max. VN degree: ", v_max)
        print("Max. CN degree: ", c_max)
        print("VN degree: ", vn_profile)
        print("CN degree: ", cn_profile)

    return pcm, k, n, coderate


def load_alist(path: str) -> List[List[int]]:
    """Reads an `alist` file and returns a nested list describing a code's
    parity-check matrix.

    This function reads a file in `alist` format :cite:p:`MacKay` and returns a nested
    list representing the parity-check matrix. Numerous example codes in
    `alist` format are available in :cite:p:`UniKL`.

    :param path: Path to the `alist` file to be loaded.

    :output alist: A nested list containing the imported `alist` data
        representing the parity-check matrix.

    .. rubric:: Examples


    .. code-block:: python

        from sionna.phy.fec.coding import load_alist

        alist = load_alist("code.alist")
    """
    alist = []
    with open(path, "r") as reader:
        # read list line by line (different length)
        for line in reader:
            row = []
            # append all entries
            for word in line.split():
                row.append(int(word))
            if row:  # ignore empty lines
                alist.append(row)

    return alist


def make_systematic(
    mat: np.ndarray,
    is_pcm: bool = False,
) -> Tuple[np.ndarray, List[List[int]]]:
    r"""Converts a binary matrix to its systematic form.

    This function transforms a binary matrix into systematic form, where the
    first `k` columns (or last `k` columns if ``is_pcm`` is `True`) form an
    identity matrix.

    :param mat: Binary matrix of shape [k, n] to be transformed to systematic
        form.
    :param is_pcm: If `True`, ``mat`` is treated as a parity-check matrix,
        and the identity part will be placed in the last `k` columns.

    :output mat_sys: The systematic matrix.

    :output column_swaps: A list of column swap operations performed.

    .. rubric:: Notes

    This function may swap columns of the input matrix to achieve systematic
    form. As a result, the output matrix represents a permuted version of the
    code, defined by the ``column_swaps`` list. To revert to the original
    column order, apply the inverse permutation in reverse order of the swaps.
    If ``is_pcm`` is `True`, indicating a parity-check matrix, the identity
    matrix portion will be arranged in the last `k` columns.
    """
    m = mat.shape[0]
    n = mat.shape[1]

    if m > n:
        raise ValueError("Invalid matrix dimensions.")

    # check for all-zero columns (=unchecked nodes)
    if is_pcm:
        c_node_deg = np.sum(mat, axis=0)
        if np.any(c_node_deg == 0):
            warnings.warn(
                "All-zero column in parity-check matrix detected. "
                "It seems as if the code contains unprotected nodes.",
                UserWarning,
                stacklevel=2,
            )

    mat = np.copy(mat)
    column_swaps = []  # store all column swaps

    # convert to bool for faster arithmetic
    mat = mat.astype(bool)

    # bring in upper triangular form
    for idx_c in range(m):
        success = mat[idx_c, idx_c]
        if not success:  # skip if leading "1" already occurred
            # step 1: find next leading "1"
            for idx_r in range(idx_c + 1, m):
                # skip if entry is "0"
                if mat[idx_r, idx_c]:
                    mat[[idx_c, idx_r]] = mat[[idx_r, idx_c]]  # swap rows
                    success = True
                    break

        # Could not find "1"-entry for column idx_c
        # => swap with columns from non-sys part
        if not success:
            for idx_cc in range(idx_c + 1, n):
                if mat[idx_c, idx_cc]:
                    # swap columns
                    mat[:, [idx_c, idx_cc]] = mat[:, [idx_cc, idx_c]]
                    column_swaps.append([idx_c, idx_cc])
                    success = True
                    break

        if not success:
            raise ValueError("Could not succeed; mat is not full rank?")

        # we can now assume a leading "1" at row idx_c
        for idx_r in range(idx_c + 1, m):
            if mat[idx_r, idx_c]:
                mat[idx_r, :] ^= mat[idx_c, :]  # bin. add of row idx_c to idx_r

    # remove upper triangle part in inverse order
    for idx_c in range(m - 1, -1, -1):
        for idx_r in range(idx_c - 1, -1, -1):
            if mat[idx_r, idx_c]:
                mat[idx_r, :] ^= mat[idx_c, :]  # bin. add of row idx_c to idx_r

    # verify results
    if not np.array_equal(mat[:, :m], np.eye(m)):
        raise RuntimeError("Internal error, could not find systematic matrix.")

    # bring identity part to end of matrix if parity-check matrix is provided
    if is_pcm:
        # individual column swaps instead of copying entire block
        for i in range(n - 1, (n - 1) - m, -1):
            j = i - (n - m)
            mat[:, [i, j]] = mat[:, [j, i]]
            column_swaps.append([i, j])

    # return integer array
    mat = mat.astype(int)
    return mat, column_swaps


def gm2pcm(gm: np.ndarray, verify_results: bool = True) -> np.ndarray:
    r"""Generates the parity-check matrix for a given generator matrix.

    This function converts the generator matrix ``gm`` (denoted as
    :math:`\mathbf{G}`) to systematic form and uses the following relationship
    to compute the parity-check matrix :math:`\mathbf{H}` over GF(2):

    .. math::

        \mathbf{G} = [\mathbf{I} |  \mathbf{M}]
        \Rightarrow \mathbf{H} = [\mathbf{M}^T | \mathbf{I}]. \tag{1}

    This is derived from the requirement for an all-zero syndrome, such that:

    .. math::

        \mathbf{H} \mathbf{c}^T = \mathbf{H} * (\mathbf{u} * \mathbf{G})^T =
        \mathbf{H} * \mathbf{G}^T * \mathbf{u}^T = \mathbf{0},

    where :math:`\mathbf{c}` represents an arbitrary codeword and
    :math:`\mathbf{u}` the corresponding information bits.

    This leads to:

    .. math::

        \mathbf{G} * \mathbf{H}^T = \mathbf{0}. \tag{2}

    It can be seen that (1) satisfies (2), as in GF(2):

    .. math::

        [\mathbf{I} |  \mathbf{M}] * [\mathbf{M}^T | \mathbf{I}]^T
         = \mathbf{M} + \mathbf{M} = \mathbf{0}.

    :param gm: Binary generator matrix of shape [k, n].
    :param verify_results: If `True`, verifies that the generated parity-check
        matrix is orthogonal to the generator matrix in GF(2).

    :output pcm: Binary parity-check matrix of shape [n - k, n].

    .. rubric:: Notes

    This function requires ``gm`` to have full rank. An error is raised if
    ``gm`` does not meet this requirement.
    """
    k = gm.shape[0]
    n = gm.shape[1]

    if k >= n:
        raise ValueError("Invalid matrix dimensions.")

    # bring gm in systematic form
    gm_sys, c_swaps = make_systematic(gm, is_pcm=False)

    m_mat = np.transpose(np.copy(gm_sys[:, -(n - k) :]))
    i_mat = np.eye(n - k)

    pcm = np.concatenate((m_mat, i_mat), axis=1)

    # undo column swaps
    for swap in c_swaps[::-1]:  # reverse ordering when going through list
        pcm[:, [swap[0], swap[1]]] = pcm[:, [swap[1], swap[0]]]  # swap columns

    if verify_results:
        if not verify_gm_pcm(gm=gm, pcm=pcm):
            raise RuntimeError(
                "Resulting parity-check matrix does not match to"
                " generator matrix."
            )

    return pcm


def pcm2gm(pcm: np.ndarray, verify_results: bool = True) -> np.ndarray:
    r"""Generates the generator matrix for a given parity-check matrix.

    This function converts the parity-check matrix ``pcm`` (denoted as
    :math:`\mathbf{H}`) to systematic form and uses the following relationship
    to compute the generator matrix :math:`\mathbf{G}` over GF(2):

    .. math::

        \mathbf{G} = [\mathbf{I} | \mathbf{M}]
        \Rightarrow \mathbf{H} = [\mathbf{M}^T | \mathbf{I}]. \tag{1}

    This derivation is based on the requirement for an all-zero syndrome:

    .. math::

        \mathbf{H} \mathbf{c}^T = \mathbf{H} * (\mathbf{u} * \mathbf{G})^T =
        \mathbf{H} * \mathbf{G}^T * \mathbf{u}^T = \mathbf{0},

    where :math:`\mathbf{c}` represents an arbitrary codeword and
    :math:`\mathbf{u}` the corresponding information bits.

    This leads to:

    .. math::

        \mathbf{G} * \mathbf{H}^T = \mathbf{0}. \tag{2}

    It can be shown that (1) satisfies (2), as in GF(2):

    .. math::

        [\mathbf{I} | \mathbf{M}] * [\mathbf{M}^T | \mathbf{I}]^T
         = \mathbf{M} + \mathbf{M} = \mathbf{0}.

    :param pcm: Binary parity-check matrix of shape [n - k, n].
    :param verify_results: If `True`, verifies that the generated generator
        matrix is orthogonal to the parity-check matrix in GF(2).

    :output gm: Binary generator matrix of shape [k, n].

    .. rubric:: Notes

    This function requires ``pcm`` to have full rank. An error is raised if
    ``pcm`` does not meet this requirement.
    """
    n = pcm.shape[1]
    k = n - pcm.shape[0]

    if k >= n:
        raise ValueError("Invalid matrix dimensions.")

    # bring pcm in systematic form
    pcm_sys, c_swaps = make_systematic(pcm, is_pcm=True)

    m_mat = np.transpose(np.copy(pcm_sys[:, :k]))
    i_mat = np.eye(k)
    gm = np.concatenate((i_mat, m_mat), axis=1)

    # undo column swaps
    for swap in c_swaps[::-1]:  # reverse ordering when going through list
        gm[:, [swap[0], swap[1]]] = gm[:, [swap[1], swap[0]]]  # swap columns

    if verify_results:
        if not verify_gm_pcm(gm=gm, pcm=pcm):
            raise RuntimeError(
                "Resulting parity-check matrix does not match to"
                " generator matrix."
            )
    return gm


def verify_gm_pcm(gm: np.ndarray, pcm: np.ndarray) -> bool:
    r"""Verifies that the generator matrix and parity-check matrix are orthogonal in GF(2).

    For a valid code with an all-zero syndrome, the following condition must
    hold:

    .. math::

        \mathbf{H} \mathbf{c}^T = \mathbf{H} * (\mathbf{u} * \mathbf{G})^T =
        \mathbf{H} * \mathbf{G}^T * \mathbf{u}^T = \mathbf{0},

    where :math:`\mathbf{c}` represents an arbitrary codeword and
    :math:`\mathbf{u}` the corresponding information bits.

    Since :math:`\mathbf{u}` can be arbitrary, this leads to the condition:

    .. math::
        \mathbf{H} * \mathbf{G}^T = \mathbf{0}.

    :param gm: Binary generator matrix of shape [k, n].
    :param pcm: Binary parity-check matrix of shape [n - k, n].

    :output is_valid: `True` if ``gm`` and ``pcm`` define a valid pair of
        orthogonal parity-check and generator matrices in GF(2).
    """
    # check for valid dimensions
    k = gm.shape[0]
    n = gm.shape[1]

    n_pcm = pcm.shape[1]
    k_pcm = n_pcm - pcm.shape[0]

    if k != k_pcm:
        raise ValueError("Inconsistent shape of gm and pcm.")
    if n != n_pcm:
        raise ValueError("Inconsistent shape of gm and pcm.")

    # check that both matrices are binary
    if not ((gm == 0) | (gm == 1)).all():
        raise ValueError("gm is not binary.")
    if not ((pcm == 0) | (pcm == 1)).all():
        raise ValueError("pcm is not binary.")

    # check for zero syndrome
    s = np.mod(np.matmul(pcm, np.transpose(gm)), 2)  # mod2 to account for GF(2)
    return np.sum(s) == 0  # Check for Non-zero syndrome of H*G'


def generate_reg_ldpc(
    v: int,
    c: int,
    n: int,
    allow_flex_len: bool = True,
    verbose: bool = True,
) -> Tuple[np.ndarray, int, int, float]:
    r"""Generates a random regular (v, c) LDPC code.

    This function generates a random Low-Density Parity-Check (LDPC)
    parity-check matrix of length ``n`` where each variable node (VN) has
    degree ``v`` and each check node (CN) has degree ``c``. Note that the
    generated LDPC code is not optimized to avoid short cycles, which may
    result in a non-negligible error floor. For encoding, the
    :class:`~sionna.phy.fec.linear.encoding.LinearEncoder` block can be used,
    but the construction does not guarantee that the parity-check matrix
    (``pcm``) has full rank.

    :param v: Desired degree of each variable node (VN).
    :param c: Desired degree of each check node (CN).
    :param n: Desired codeword length.
    :param allow_flex_len: If `True`, the resulting codeword length may be
        slightly increased to meet the degree requirements.
    :param verbose: If `True`, prints code parameters.

    :output pcm: The parity-check matrix.

    :output k: The number of information bits.

    :output n: The codeword length.

    :output coderate: The code rate.

    .. rubric:: Notes

    This algorithm is designed only for regular node degrees. To achieve
    state-of-the-art bit-error-rate performance, optimizing irregular degree
    profiles is usually necessary (see :cite:p:`tenBrink`).

    .. rubric:: Examples


    .. code-block:: python

        from sionna.phy.fec.coding import generate_reg_ldpc

        pcm, k, n, rate = generate_reg_ldpc(3, 6, 100, verbose=False)
        print(f"Generated LDPC code: n={n}, k={k}, rate={rate:.3f}")
    """
    # check input values for consistency
    if not isinstance(allow_flex_len, bool):
        raise TypeError("allow_flex_len must be bool.")

    # allow slight change in n to keep num edges
    # from CN and VN perspective an integer
    if allow_flex_len:
        for n_mod in range(n, n + 2 * c):
            if np.mod((v / c) * n_mod, 1.0) == 0:
                n = n_mod
                if verbose:
                    print("Setting n to: ", n)
                break

    # calculate number of nodes
    coderate = 1 - (v / c)
    n_v = n
    n_c = int((v / c) * n)
    k = n_v - n_c

    # generate sockets
    v_socks = np.tile(np.arange(n_v), v)
    c_socks = np.tile(np.arange(n_c), c)
    if verbose:
        print("Number of edges (VN perspective): ", len(v_socks))
        print("Number of edges (CN perspective): ", len(c_socks))
    if len(v_socks) != len(c_socks):
        raise ValueError(
            "Number of edges from VN and CN "
            "perspective does not match. Consider to (slightly) change n."
        )

    # apply random permutations
    config.np_rng.shuffle(v_socks)
    config.np_rng.shuffle(c_socks)

    # and generate matrix
    pcm = np.zeros([n_c, n_v])

    idx = 0
    shuffle_max = 200  # stop if no success
    shuffle_counter = 0
    cont = True
    while cont:
        # if edge is available, take it
        if pcm[c_socks[idx], v_socks[idx]] == 0:
            pcm[c_socks[idx], v_socks[idx]] = 1
            idx += 1  # and go to next socket
            shuffle_counter = 0  # reset counter
            if idx == len(v_socks):
                cont = False
        else:  # shuffle sockets
            shuffle_counter += 1
            if shuffle_counter < shuffle_max:
                config.np_rng.shuffle(v_socks[idx:])
                config.np_rng.shuffle(c_socks[idx:])
            else:
                warnings.warn(
                    "Stopping LDPC generation - no solution found.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                cont = False

    v_deg = np.sum(pcm, axis=0)
    c_deg = np.sum(pcm, axis=1)

    if not (v_deg == v).all():
        raise RuntimeError("VN degree not always v.")
    if not (c_deg == c).all():
        raise RuntimeError("CN degree not always c.")

    if verbose:
        print(f"Generated regular ({v},{c}) LDPC code of length n={n}")
        print(f"Code rate is r={coderate:.3f}.")

    return pcm, k, n, coderate
