#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Blocks for LDPC channel encoding and utility functions."""

from typing import List, Optional, Tuple
import numbers
import warnings

import numpy as np
import scipy as sp
import torch
from importlib_resources import files, as_file

from sionna._validation import check_binary
from sionna.phy import Block
from sionna.phy.fec.utils import int_mod_2
from . import codes


__all__ = ["LDPC5GEncoder"]


class LDPC5GEncoder(Block):
    # pylint: disable=line-too-long
    r"""5G NR LDPC Encoder following the 3GPP 38.212 including rate-matching.

    The implementation follows the 3GPP NR Initiative :cite:p:`3GPPTS38212`.

    :param k: Number of information bits per codeword.
    :param n: Desired codeword length.
    :param num_bits_per_symbol: Number of bits per QAM symbol. If provided,
        the codeword will be interleaved after rate-matching as specified
        in Sec. 5.4.2.2 in :cite:p:`3GPPTS38212`.
    :param bg: Basegraph to be used for the code construction.
        If `None`, the encoder will automatically select the basegraph
        according to :cite:p:`3GPPTS38212`.
    :param precision: Precision used for internal calculations and outputs.
        If `None`, :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for computation (e.g., 'cpu', 'cuda:0').
        If `None`, :attr:`~sionna.phy.config.Config.device` is used.
    :param check_input: If `True` (default), check that inputs are binary
        on every call. Set to `False` to skip the check and reduce overhead.

    :input bits: [..., k], `torch.float`.
        Binary tensor containing the information bits to be encoded.

    :input rv: `None` | list of int.
        List of redundancy version indices (0–3) for HARQ-IR
        rate-matching per TS 38.212 Table 5.4.2.1-2.
        If `None` (default), standard RV 0 encoding is used and the
        output shape is ``[..., n]``.  When provided, each RV produces
        an independent rate-matched codeword and the output shape
        becomes ``[..., len(rv), n]``.

    :output cw: [..., n] or [..., len(rv), n], `torch.float`.
        Encoded codeword bits.  Same shape as the input with the last
        dimension replaced by ``n``.  When ``rv`` is provided, an
        additional dimension of size ``len(rv)`` is inserted before the
        codeword dimension.

    .. rubric:: Notes

    As specified in :cite:p:`3GPPTS38212`, the encoder also performs
    rate-matching (puncturing and shortening). Thus, the corresponding
    decoder needs to `invert` these operations, i.e., must be compatible with
    the 5G encoding scheme.

    .. rubric:: Examples


    .. code-block:: python

        import torch
        from sionna.phy.fec.ldpc import LDPC5GEncoder

        # Create encoder for k=100 information bits and n=200 codeword bits
        encoder = LDPC5GEncoder(k=100, n=200)

        # Generate random information bits
        u = torch.randint(0, 2, (10, 100), dtype=torch.float32)
        c = encoder(u)
        print(c.shape)
        # [10, 200]

        # HARQ: produce two redundancy versions
        c_harq = encoder(u, rv=[0, 2])
        print(c_harq.shape)
        # [10, 2, 200]
    """

    def __init__(
        self,
        k: int,
        n: int,
        num_bits_per_symbol: Optional[int] = None,
        bg: Optional[str] = None,
        precision: Optional[str] = None,
        device: Optional[str] = None,
        check_input: bool = True,
        **kwargs,
    ):
        super().__init__(precision=precision, device=device, **kwargs)

        if not isinstance(k, numbers.Number):
            raise TypeError("k must be a number.")
        if not isinstance(n, numbers.Number):
            raise TypeError("n must be a number.")
        k = int(k)  # k or n can be float (e.g. as result of n=k*r)
        n = int(n)

        if k > 8448:
            raise ValueError("Unsupported code length (k too large).")
        if k < 12:
            raise ValueError("Unsupported code length (k too small).")

        if n > (316 * 384):
            raise ValueError("Unsupported code length (n too large).")
        if n < 0:
            raise ValueError("Unsupported code length (n negative).")

        # Init encoder parameters
        self._k = k  # number of input bits (= input shape)
        self._n = n  # the desired length (= output shape)
        self._coderate = k / n
        if not isinstance(check_input, bool):
            raise TypeError("check_input must be bool.")
        self._check_input = check_input

        # Allow actual code rates slightly larger than 948/1024
        # to account for the quantization procedure in 38.214 5.1.3.1
        if self._coderate > (948 / 1024):  # as specified in 38.212 5.4.2.1
            warnings.warn(
                f"Effective coderate r>948/1024 for n={n}, k={k}.",
                UserWarning,
                stacklevel=2,
            )
        if self._coderate > 0.95:  # as specified in 38.212 5.4.2.1
            raise ValueError(f"Unsupported coderate (r>0.95) for n={n}, k={k}.")
        if self._coderate < (1 / 5):
            # outer rep. coding currently not supported
            raise ValueError("Unsupported coderate (r<1/5).")

        # Construct the basegraph according to 38.212
        self._bg = self._sel_basegraph(self._k, self._coderate, bg)

        self._z, self._i_ls, self._k_b = self._sel_lifting(self._k, self._bg)
        self._bm = self._load_basegraph(self._i_ls, self._bg)

        # Total number of codeword bits
        self._n_ldpc = self._bm.shape[1] * self._z
        # If K_real < K_target puncturing must be applied earlier
        self._k_ldpc = self._k_b * self._z

        # Construct explicit graph via lifting
        pcm = self._lift_basegraph(self._bm, self._z)

        pcm_a, pcm_b_inv, pcm_c1, pcm_c2 = self._gen_submat(
            self._bm, self._k_b, self._z, self._bg
        )

        # Init sub-matrices for fast encoding ("RU"-method)
        self._pcm = pcm  # store the sparse parity-check matrix (for decoding)

        # Store indices for fast gathering (instead of explicit matmul)
        # Register as buffers for CUDAGraph compatibility
        self.register_buffer("_pcm_a_ind", torch.tensor(
            self._mat_to_ind(pcm_a), dtype=torch.int32, device=self.device
        ))
        self.register_buffer("_pcm_b_inv_ind", torch.tensor(
            self._mat_to_ind(pcm_b_inv), dtype=torch.int32, device=self.device
        ))
        self.register_buffer("_pcm_c1_ind", torch.tensor(
            self._mat_to_ind(pcm_c1), dtype=torch.int32, device=self.device
        ))
        self.register_buffer("_pcm_c2_ind", torch.tensor(
            self._mat_to_ind(pcm_c2), dtype=torch.int32, device=self.device
        ))

        self._num_bits_per_symbol = num_bits_per_symbol
        if num_bits_per_symbol is not None:
            out_int, out_int_inv = self.generate_out_int(
                self._n, self._num_bits_per_symbol
            )
            self.register_buffer("_out_int", torch.tensor(out_int, dtype=torch.int32, device=self.device))
            self.register_buffer("_out_int_inv", torch.tensor(
                out_int_inv, dtype=torch.int32, device=self.device
            ))

    ###############################
    # Public methods and properties
    ###############################

    @property
    def k(self) -> int:
        """Number of input information bits."""
        return self._k

    @property
    def n(self) -> int:
        """Number of output codeword bits."""
        return self._n

    @property
    def coderate(self) -> float:
        """Coderate of the LDPC code after rate-matching."""
        return self._coderate

    @property
    def k_ldpc(self) -> int:
        """Number of LDPC information bits after rate-matching."""
        return self._k_ldpc

    @property
    def n_ldpc(self) -> int:
        """Number of LDPC codeword bits before rate-matching."""
        return self._n_ldpc

    @property
    def pcm(self) -> sp.sparse.csr_matrix:
        """Parity-check matrix for given code parameters."""
        return self._pcm

    @property
    def z(self) -> int:
        """Lifting factor of the basegraph."""
        return self._z

    @property
    def num_bits_per_symbol(self) -> Optional[int]:
        """Modulation order used for the rate-matching output interleaver."""
        return self._num_bits_per_symbol

    @property
    def k_filler(self) -> int:
        """Number of filler bits added to pad ``k`` to ``k_ldpc``."""
        return self._k_ldpc - self._k

    @property
    def n_cb(self) -> int:
        """Circular buffer length (excludes first 2Z)."""
        return self._n_ldpc - 2 * self._z

    @property
    def n_cb_comp(self) -> int:
        """Compressed circular buffer length (excludes first 2Z and fillers)."""
        return self.n_cb - self.k_filler

    @property
    def rv_starts(self) -> List[int]:
        r"""RV start positions ``k0`` from TS 38.212 Table 5.4.2.1-2.

        Returns a list of length 4 indexed by ``rv_id``.  Values are in
        spec buffer space (after the first 2Z punctured positions,
        before filler removal).

        .. math::
            k_0 = \left\lfloor \frac{\text{coeff} \cdot N_{cb}}
                  {N_{\text{cols}} \cdot Z_c} \right\rfloor \cdot Z_c

        where :math:`N_{\text{cols}}` is 66 (BG1) or 50 (BG2).
        Currently assumes ``I_LBRM=0``, i.e., ``N_cb = N``.
        """
        n_cb = self.n_cb
        z = self._z
        if self._bg == "bg1":
            coeffs = [0, 17, 33, 56]
            n_cols = 66
        else:
            coeffs = [0, 13, 25, 43]
            n_cols = 50
        return [(c * n_cb // (n_cols * z)) * z for c in coeffs]

    @property
    def out_int(self) -> torch.Tensor:
        """Output interleaver sequence as defined in 5.4.2.2."""
        return self._out_int

    @property
    def out_int_inv(self) -> torch.Tensor:
        """Inverse output interleaver sequence as defined in 5.4.2.2."""
        return self._out_int_inv

    #################
    # Utility methods
    #################

    def _k0_comp(self, k0: int) -> int:
        """Map ``k0`` from spec buffer space to compressed RM buffer space.

        The compressed buffer omits the contiguous filler block.
        """
        filler_start = self._k - 2 * self._z
        filler_len = self.k_filler
        if filler_len <= 0 or k0 < filler_start:
            return k0
        if k0 < filler_start + filler_len:
            return filler_start
        return k0 - filler_len

    def get_start_positions_comp(self, rv_list: List[int]) -> List[int]:
        """Return compressed-buffer start positions for a list of RV indices.

        :param rv_list: List of RV indices (each 0–3).

        :output: Corresponding start positions in the compressed RM buffer.
        """
        self._check_rv(rv_list)
        rv_start_map = self.rv_starts
        return [self._k0_comp(rv_start_map[rv_id]) for rv_id in rv_list]

    @staticmethod
    def _check_rv(rv: List[int]) -> None:
        """Reject an empty RV list or indices outside ``{0, 1, 2, 3}``."""
        if not rv:
            raise ValueError("rv must be a non-empty list of RV indices.")
        for v in rv:
            if v not in (0, 1, 2, 3):
                raise ValueError(f"Invalid RV index {v}; must be 0–3.")

    def generate_out_int(
        self, n: int, num_bits_per_symbol: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generates LDPC output interleaver sequence as defined in
        Sec 5.4.2.2 in :cite:p:`3GPPTS38212`.

        :param n: Desired output sequence length.
        :param num_bits_per_symbol: Number of bits per QAM symbol,
            i.e., the modulation order.

        .. rubric:: Notes

        The interleaver pattern depends on the modulation order and helps to
        reduce dependencies in bit-interleaved coded modulation (BICM) schemes
        combined with higher order modulation.
        """
        # Allow float inputs, but verify that they represent integer
        if n % 1 != 0:
            raise ValueError("n must be int.")
        if num_bits_per_symbol % 1 != 0:
            raise ValueError("num_bits_per_symbol must be int.")
        n = int(n)
        if n <= 0:
            raise ValueError("n must be a positive integer.")
        if num_bits_per_symbol <= 0:
            raise ValueError("num_bits_per_symbol must be a positive integer.")
        num_bits_per_symbol = int(num_bits_per_symbol)

        if n % num_bits_per_symbol != 0:
            raise ValueError("n must be a multiple of num_bits_per_symbol.")

        # Pattern as defined in Sec 5.4.2.2
        perm_seq = np.zeros(n, dtype=int)
        for j in range(int(n / num_bits_per_symbol)):
            for i in range(num_bits_per_symbol):
                perm_seq[i + j * num_bits_per_symbol] = int(
                    i * int(n / num_bits_per_symbol) + j
                )

        perm_seq_inv = np.argsort(perm_seq)

        return perm_seq, perm_seq_inv

    def _sel_basegraph(
        self, k: int, r: float, bg_: Optional[str] = None
    ) -> str:
        """Select basegraph according to :cite:p:`3GPPTS38212` and check for
        consistency.
        """
        # If bg is explicitly provided, we only check for consistency
        if bg_ is None:
            if k <= 292:
                bg = "bg2"
            elif k <= 3824 and r <= 0.67:
                bg = "bg2"
            elif r <= 0.25:
                bg = "bg2"
            else:
                bg = "bg1"
        elif bg_ in ("bg1", "bg2"):
            bg = bg_
        else:
            raise ValueError("Basegraph must be bg1, bg2 or None.")

        # Check for consistency
        if bg == "bg1" and k > 8448:
            raise ValueError("K is not supported by BG1 (too large).")

        if bg == "bg2" and k > 3840:
            raise ValueError(f"K is not supported by BG2 (too large) k={k}.")

        if bg == "bg1" and r < 1 / 3:
            raise ValueError(
                "Only coderate>1/3 supported for BG1. "
                "Remark: Repetition coding is currently not supported."
            )

        if bg == "bg2" and r < 1 / 5:
            raise ValueError(
                "Only coderate>1/5 supported for BG2. "
                "Remark: Repetition coding is currently not supported."
            )

        return bg

    def _load_basegraph(self, i_ls: int, bg: str) -> np.ndarray:
        """Helper to load basegraph from csv files.

        ``i_ls`` is sub_index of the basegraph and fixed during lifting
        selection.
        """
        if i_ls > 7:
            raise ValueError("i_ls too large.")

        if i_ls < 0:
            raise ValueError("i_ls cannot be negative.")

        # csv files are taken from 38.212 and dimension is explicitly given
        if bg == "bg1":
            bm = np.zeros([46, 68]) - 1  # init matrix with -1 (None positions)
        elif bg == "bg2":
            bm = np.zeros([42, 52]) - 1  # init matrix with -1 (None positions)
        else:
            raise ValueError("Basegraph not supported.")

        # And load the basegraph from csv format in folder "codes"
        source = files(codes).joinpath(f"5G_{bg}.csv")
        with as_file(source) as codes_csv:
            bg_csv = np.genfromtxt(codes_csv, delimiter=";")

        # Reconstruct BG for given i_ls
        r_ind = 0
        for r in np.arange(2, bg_csv.shape[0]):
            # Check for next row index
            if not np.isnan(bg_csv[r, 0]):
                r_ind = int(bg_csv[r, 0])
            c_ind = int(bg_csv[r, 1])  # second column in csv is column index
            value = bg_csv[r, i_ls + 2]  # i_ls entries start at offset 2
            bm[r_ind, c_ind] = value

        return bm

    def _lift_basegraph(self, bm: np.ndarray, z: int) -> sp.sparse.csr_matrix:
        """Lift basegraph with lifting factor ``z`` and shifted identities as
        defined by the entries of ``bm``.
        """
        num_nonzero = np.sum(bm >= 0)  # num of non-neg elements in bm

        # Init all non-zero row/column indices
        r_idx = np.zeros(z * num_nonzero)
        c_idx = np.zeros(z * num_nonzero)
        data = np.ones(z * num_nonzero)

        # Row/column indices of identity matrix for lifting
        im = np.arange(z)

        idx = 0
        for r in range(bm.shape[0]):
            for c in range(bm.shape[1]):
                if bm[r, c] == -1:  # -1 is used as all-zero matrix placeholder
                    pass  # do nothing (sparse)
                else:
                    # Roll matrix by bm[r,c]
                    c_roll = np.mod(im + bm[r, c], z)
                    # Append rolled identity matrix to pcm
                    r_idx[idx * z : (idx + 1) * z] = r * z + im
                    c_idx[idx * z : (idx + 1) * z] = c * z + c_roll
                    idx += 1

        # Generate lifted sparse matrix from indices
        pcm = sp.sparse.csr_matrix(
            (data, (r_idx, c_idx)), shape=(z * bm.shape[0], z * bm.shape[1])
        )
        return pcm

    def _sel_lifting(self, k: int, bg: str) -> Tuple[int, int, int]:
        """Select lifting as defined in Sec. 5.2.2 in :cite:p:`3GPPTS38212`.

        We assume B < K_cb, thus B'= B and C = 1, i.e., no
        additional CRC is appended. Thus, K' = B'/C = B and B is our K.

        Z is the lifting factor.
        i_ls is the set index ranging from 0...7 (specifying the exact bg
        selection).
        k_b is the number of information bit columns in the basegraph.
        """
        # Lifting set according to 38.212 Tab 5.3.2-1
        s_val = [
            [2, 4, 8, 16, 32, 64, 128, 256],
            [3, 6, 12, 24, 48, 96, 192, 384],
            [5, 10, 20, 40, 80, 160, 320],
            [7, 14, 28, 56, 112, 224],
            [9, 18, 36, 72, 144, 288],
            [11, 22, 44, 88, 176, 352],
            [13, 26, 52, 104, 208],
            [15, 30, 60, 120, 240],
        ]

        if bg == "bg1":
            k_b = 22
        else:
            if k > 640:
                k_b = 10
            elif k > 560:
                k_b = 9
            elif k > 192:
                k_b = 8
            else:
                k_b = 6

        # Find the min of Z from Tab. 5.3.2-1 s.t. k_b*Z>=K'
        min_val = 100000
        z = 0
        i_ls = 0
        i = -1
        for s in s_val:
            i += 1
            for s1 in s:
                x = k_b * s1
                if x >= k:
                    # Valid solution
                    if x < min_val:
                        min_val = x
                        z = s1
                        i_ls = i

        # And set K=22*Z for bg1 and K=10Z for bg2
        if bg == "bg1":
            k_b = 22
        else:
            k_b = 10

        return z, i_ls, k_b

    def _gen_submat(
        self, bm: np.ndarray, k_b: int, z: int, bg: str
    ) -> Tuple[
        sp.sparse.csr_matrix,
        sp.sparse.csr_matrix,
        sp.sparse.csr_matrix,
        sp.sparse.csr_matrix,
    ]:
        """Split the basegraph into multiple sub-matrices such that efficient
        encoding is possible.
        """
        g = 4  # code property (always fixed for 5G)
        mb = bm.shape[0]  # number of CN rows in basegraph (BG property)

        bm_a = bm[0:g, 0:k_b]
        bm_b = bm[0:g, k_b : (k_b + g)]
        bm_c1 = bm[g:mb, 0:k_b]
        bm_c2 = bm[g:mb, k_b : (k_b + g)]

        # H could be sliced immediately (but easier to implement if based on B)
        hm_a = self._lift_basegraph(bm_a, z)
        hm_c1 = self._lift_basegraph(bm_c1, z)
        hm_c2 = self._lift_basegraph(bm_c2, z)

        hm_b_inv = self._find_hm_b_inv(bm_b, z, bg)

        return hm_a, hm_b_inv, hm_c1, hm_c2

    def _find_hm_b_inv(
        self, bm_b: np.ndarray, z: int, bg: str
    ) -> sp.sparse.csr_matrix:
        """For encoding we need to find the inverse of `hm_b` such that
        `hm_b^-1 * hm_b = I`.

        Could be done sparse.
        For BG1 the structure of hm_b is given as (for all values of i_ls)
        hm_b =
        [P_A I 0 0
         P_B I I 0
         0 0 I I
         P_A 0 0 I]
        where P_B and P_A are shifted identities.

        The inverse can be found by solving a linear system of equations
        hm_b_inv =
        [P_B^-1, P_B^-1, P_B^-1, P_B^-1,
         I + P_A*P_B^-1, P_A*P_B^-1, P_A*P_B^-1, P_A*P_B^-1,
         P_A*P_B^-1, P_A*P_B^-1, I+P_A*P_B^-1, I+P_A*P_B^-1,
         P_A*P_B^-1, P_A*P_B^-1, P_A*P_B^-1, I+P_A*P_B^-1].

        For bg2 the structure of hm_b is given as (for all values of i_ls)
        hm_b =
        [P_A I 0 0
         0 I I 0
         P_B 0 I I
         P_A 0 0 I]
        where P_B and P_A are shifted identities.

        The inverse can be found by solving a linear system of equations
        hm_b_inv =
        [P_B^-1, P_B^-1, P_B^-1, P_B^-1,
         I + P_A*P_B^-1, P_A*P_B^-1, P_A*P_B^-1, P_A*P_B^-1,
         I+P_A*P_B^-1, I+P_A*P_B^-1, P_A*P_B^-1, P_A*P_B^-1,
         P_A*P_B^-1, P_A*P_B^-1, P_A*P_B^-1, I+P_A*P_B^-1]

        Note: the inverse of B is simply a shifted identity matrix with
        negative shift direction.
        """
        # Permutation indices
        pm_a = int(bm_b[0, 0])
        if bg == "bg1":
            pm_b_inv = int(-bm_b[1, 0])
        else:  # structure of B is slightly different for bg2
            pm_b_inv = int(-bm_b[2, 0])

        hm_b_inv = np.zeros([4 * z, 4 * z])

        im = np.eye(z)

        am = np.roll(im, pm_a, axis=1)
        b_inv = np.roll(im, pm_b_inv, axis=1)
        ab_inv = np.matmul(am, b_inv)

        # Row 0
        hm_b_inv[0:z, 0:z] = b_inv
        hm_b_inv[0:z, z : 2 * z] = b_inv
        hm_b_inv[0:z, 2 * z : 3 * z] = b_inv
        hm_b_inv[0:z, 3 * z : 4 * z] = b_inv

        # Row 1
        hm_b_inv[z : 2 * z, 0:z] = im + ab_inv
        hm_b_inv[z : 2 * z, z : 2 * z] = ab_inv
        hm_b_inv[z : 2 * z, 2 * z : 3 * z] = ab_inv
        hm_b_inv[z : 2 * z, 3 * z : 4 * z] = ab_inv

        # Row 2
        if bg == "bg1":
            hm_b_inv[2 * z : 3 * z, 0:z] = ab_inv
            hm_b_inv[2 * z : 3 * z, z : 2 * z] = ab_inv
            hm_b_inv[2 * z : 3 * z, 2 * z : 3 * z] = im + ab_inv
            hm_b_inv[2 * z : 3 * z, 3 * z : 4 * z] = im + ab_inv
        else:  # for bg2 the structure is slightly different
            hm_b_inv[2 * z : 3 * z, 0:z] = im + ab_inv
            hm_b_inv[2 * z : 3 * z, z : 2 * z] = im + ab_inv
            hm_b_inv[2 * z : 3 * z, 2 * z : 3 * z] = ab_inv
            hm_b_inv[2 * z : 3 * z, 3 * z : 4 * z] = ab_inv

        # Row 3
        hm_b_inv[3 * z : 4 * z, 0:z] = ab_inv
        hm_b_inv[3 * z : 4 * z, z : 2 * z] = ab_inv
        hm_b_inv[3 * z : 4 * z, 2 * z : 3 * z] = ab_inv
        hm_b_inv[3 * z : 4 * z, 3 * z : 4 * z] = im + ab_inv

        # Return results as sparse matrix
        return sp.sparse.csr_matrix(hm_b_inv)

    def _mat_to_ind(self, mat: sp.sparse.csr_matrix) -> np.ndarray:
        """Helper to transform matrix into index representation for
        gather. An index pointing to the ``last_ind+1`` is used for
        non-existing edges due to irregular degrees.
        """
        m = mat.shape[0]
        n = mat.shape[1]

        # Transpose mat for sorted column format
        c_idx, r_idx, _ = sp.sparse.find(mat.transpose())

        # Sort indices explicitly, as scipy.sparse.find changed from column to
        # row sorting in scipy>=1.11
        idx = np.argsort(r_idx)
        c_idx = c_idx[idx]
        r_idx = r_idx[idx]

        # Find max number of non-zero entries
        n_max = np.max(mat.getnnz(axis=1))

        # Init index array with n (pointer to last_ind+1, will be a default
        # value)
        gat_idx = np.zeros([m, n_max]) + n

        r_val = -1
        c_val = 0
        for idx in range(len(c_idx)):
            # Check if same row or if a new row starts
            if r_idx[idx] != r_val:
                r_val = r_idx[idx]
                c_val = 0
            gat_idx[r_val, c_val] = c_idx[idx]
            c_val += 1

        return gat_idx.astype(np.int32)

    def _matmul_gather(self, mat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
        """Implements a fast sparse matmul via gather function."""
        # Add 0 entry for gather-reduce_sum operation
        # (otherwise ragged Tensors are required)
        bs = vec.shape[0]
        vec = torch.cat([vec, torch.zeros(bs, 1, dtype=vec.dtype, device=vec.device)], dim=1)

        # Gather and sum
        retval = vec[:, mat]  # [bs, m, n_max]
        retval = retval.sum(dim=-1)

        return retval

    def _encode_fast(self, s: torch.Tensor) -> torch.Tensor:
        """Main encoding function based on gathering function."""
        p_a = self._matmul_gather(self._pcm_a_ind, s)
        p_a = self._matmul_gather(self._pcm_b_inv_ind, p_a)

        # Calc second part of parity bits p_b
        # second parities are given by C_1*s' + C_2*p_a' + p_b' = 0
        p_b_1 = self._matmul_gather(self._pcm_c1_ind, s)
        p_b_2 = self._matmul_gather(self._pcm_c2_ind, p_a)
        p_b = p_b_1 + p_b_2

        c = torch.cat([s, p_a, p_b], dim=1)

        # Faster implementation of mod-2 operation
        c = int_mod_2(c)

        c = c.unsqueeze(-1)  # returns nx1 vector
        return c

    def build(self, input_shape: tuple, **kwargs) -> None:
        """Build block and check for valid input shapes."""
        if input_shape[-1] != self._k:
            raise ValueError(f"Last dimension must be of length k={self._k}.")

    def _validate_input(self, u: torch.Tensor) -> None:
        """Validate binary inputs when ``check_input`` is enabled."""
        if self._check_input:
            check_binary(
                u,
                name="bits",
                message="Input must be binary.",
            )

    def call(
        self,
        bits: torch.Tensor,
        /,
        *,
        rv: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """5G LDPC encoding including rate-matching.

        :param bits: ``[..., k]`` information bits to be encoded.
        :param rv: Redundancy version indices (0–3) for HARQ-IR.
            If `None`, standard RV 0 rate-matching is used.

        :output: ``[..., n]`` or ``[..., len(rv), n]`` encoded codewords.
        """
        # Validate rv early
        if rv is not None:
            rv = list(rv)
            self._check_rv(rv)

        input_shape = list(bits.shape)
        u = bits.reshape(-1, input_shape[-1])
        self._validate_input(u)
        batch_size = u.shape[0]

        # Pad with filler bits and encode
        u_fill = torch.cat(
            [
                u,
                torch.zeros(
                    batch_size, self._k_ldpc - self._k,
                    dtype=u.dtype, device=u.device,
                ),
            ],
            dim=1,
        )
        c = self._encode_fast(u_fill.to(self.dtype))
        c = c.reshape(batch_size, self._n_ldpc)

        # Remove filler bits → compressed codeword
        c_no_filler = torch.cat(
            [c[:, :self._k], c[:, self._k_ldpc:]], dim=1
        )

        # --- Rate matching ------------------------------------------------
        # Compressed RM buffer: skip first 2Z punctured positions
        c_rm = c_no_filler[:, 2 * self._z :]  # [batch, n_cb_comp]

        if rv is None:
            c_out = c_rm[:, :self._n]  # [batch, n]
            output_shape = input_shape[:-1] + [self.n]
        else:
            starts = self.get_start_positions_comp(rv)
            n_cb_comp = self.n_cb_comp
            rv_slices = []
            for start in starts:
                if start + self._n <= n_cb_comp:
                    rv_slices.append(c_rm[:, start:start + self._n])
                else:
                    first_len = n_cb_comp - start
                    rv_slices.append(torch.cat(
                        [c_rm[:, start:], c_rm[:, :self._n - first_len]],
                        dim=1,
                    ))
            c_out = torch.stack(rv_slices, dim=1)  # [batch, num_rvs, n]
            output_shape = input_shape[:-1] + [len(rv), self.n]

        # Output interleaver (Sec. 5.4.2.2) — works on last dim for any rank
        if self._num_bits_per_symbol is not None:
            c_out = c_out[..., self._out_int]

        output_shape[0] = -1
        return c_out.reshape(output_shape).to(bits.dtype)


