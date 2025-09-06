#
# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0#
"""Blocks for LDPC channel encoding and utility functions."""

import tensorflow as tf
import numpy as np
import scipy as sp
from importlib_resources import files, as_file
from . import codes # pylint: disable=relative-beyond-top-level
import numbers # to check if n, k are numbers
from sionna.phy import Block
from typing import Optional

class LDPC5GEncoder(Block):
    # pylint: disable=line-too-long
    """5G NR LDPC Encoder following the 3GPP 38.212 including rate-matching.

    The implementation follows the 3GPP NR Initiative [3GPPTS38212_LDPC]_.

    Parameters
    ----------
    k: int
        Defining the number of information bit per codeword.

    n: int
        Defining the desired codeword length.

    num_bits_per_symbol: `None` (default) | int
        Defining the number of bits per QAM symbol. If this parameter is
        explicitly provided, the codeword will be interleaved after
        rate-matching as specified in Sec. 5.4.2.2 in [3GPPTS38212_LDPC]_.

    bg: `None` (default) | "bg1" | "bg2"
        Basegraph to be used for the code construction.
        If `None` is provided, the encoder will automatically select
        the basegraph according to [3GPPTS38212_LDPC]_.

    precision : `None` (default) | 'single' | 'double'
        Precision used for internal calculations and outputs.
        If set to `None`, :py:attr:`~sionna.phy.config.precision` is used.

    allow_low_rates: bool, (default `False`)
        If set to `True`, the encoder will allow coding rates below 1/3 and 1/5,
        respectively for BG1 and BG2, consistent with 3GPP specifications.

    Input
    -----
    bits: [...,k], tf.float
        Binary tensor containing the information bits to be encoded.

    rv: list of str | None (default)
        List of redundancy version strings to generate. Can contain
        any combination of ["rv0", "rv1", "rv2", "rv3"] in any order,
        including repeats. If `None`, defaults to single `"rv0"` encoding without
        adding an RV dimension to the output.

    Output
    ------
    : [...,n] or [...,len(rv),n], tf.float
        Binary tensor of same shape as inputs besides last dimension has
        changed to `n` containing the encoded codeword bits. If `rv` is not `None`, 
        an additional dimension is adde to the output.
        
    Note
    ----
    As specified in [3GPPTS38212_LDPC]_, the encoder also performs
    rate-matching (puncturing and shortening). Thus, the corresponding
    decoder needs to `invert` these operations, i.e., must be compatible with
    the 5G encoding scheme.

    The default RV0 is assumed. But before each transmission, the RV can be
    set as needed.
    """

    def __init__(self,
                 k,
                 n,
                 num_bits_per_symbol=None,
                 bg=None,
                 precision=None,
                 allow_low_rates=False,
                 **kwargs):

        super().__init__(precision=precision, **kwargs)

        if not isinstance(k, numbers.Number):
            raise TypeError("k must be a number.")
        if not isinstance(n, numbers.Number):
            raise TypeError("n must be a number.")
        k = int(k) # k or n can be float (e.g. as result of n=k*r)
        n = int(n) # k or n can be float (e.g. as result of n=k*r)

        if k>8448:
            raise ValueError("Unsupported code length (k too large).")
        if k<12:
            raise ValueError("Unsupported code length (k too small).")

        if n>(68*384) or (bg=="bg2" and n>(52*384)):
            raise ValueError("Unsupported code length (n too large).")
        if n<0:
            raise ValueError("Unsupported code length (n negative).")

        # init encoder parameters
        self._k = k # number of input bits (= input shape)
        self._n = n # the desired length (= output shape)
        self._coderate = k / n
        self._check_input = True # check input for consistency (i.e., binary)

        self._allow_low_rates = allow_low_rates # allow rates below 1/3 (BG1) and 1/5 (BG2)

        # allow actual code rates slightly larger than 948/1024
        # to account for the quantization procedure in 38.214 5.1.3.1
        if self._coderate>(948/1024): # as specified in 38.212 5.4.2.1
            print(f"Warning: effective coderate r>948/1024 for n={n}, k={k}.")
        if self._coderate>(0.95): # as specified in 38.212 5.4.2.1
            raise ValueError(f"Unsupported coderate (r>0.95) for n={n}, k={k}.")
        if self._coderate<(1/5) and not self._allow_low_rates:
            # outer rep. coding currently not supported
            raise ValueError("Unsupported coderate (r<1/5).")

        # construct the basegraph according to 38.212
        # if bg is explicitly provided
        self._bg = self._sel_basegraph(self._k, self._coderate, bg)

        self._z, self._i_ls, self._k_b = self._sel_lifting(self._k, self._bg)

        self._bm = self._load_basegraph(self._i_ls, self._bg)
        bm_num_cols = self._bm.shape[1]

        # total number of codeword bits
        self._n_ldpc = bm_num_cols * self._z
        # if K_real < K _target puncturing must be applied earlier
        self._k_ldpc = self._k_b * self._z

        # construct explicit graph via lifting
        pcm = self._lift_basegraph(self._bm, self._z)

        pcm_a, pcm_b_inv, pcm_c1, pcm_c2 = self._gen_submat(self._bm,
                                                            self._k_b,
                                                            self._z,
                                                            self._bg)

        # init sub-matrices for fast encoding ("RU"-method)
        # note: dtype is tf.float32;
        self._pcm = pcm # store the sparse parity-check matrix (for decoding)

        # store indices for fast gathering (instead of explicit matmul)
        self._pcm_a_ind = self._mat_to_ind(pcm_a)
        self._pcm_b_inv_ind = self._mat_to_ind(pcm_b_inv)
        self._pcm_c1_ind = self._mat_to_ind(pcm_c1)
        self._pcm_c2_ind = self._mat_to_ind(pcm_c2)

        self._num_bits_per_symbol = num_bits_per_symbol
        if num_bits_per_symbol is not None:
            self._out_int, self._out_int_inv  = self.generate_out_int(self._n,
                                                    self._num_bits_per_symbol)

    ###############################
    # Public methods and properties
    ###############################

    @property
    def k(self):
        """Number of input information bits"""
        return self._k

    @property
    def n(self):
        "Number of output codeword bits"
        return self._n

    @property
    def coderate(self):
        """Coderate of the LDPC code after rate-matching"""
        return self._coderate

    @property
    def k_filler(self):
        """Number of (systematic) filler bits added to the input bits (and
        removed before rate-matching)"""
        return self._k_ldpc - self._k

    @property
    def k_ldpc(self):
        """Number of LDPC information bits including filler bits"""
        return self._k_ldpc

    @property
    def n_cb(self):
        """Circular buffer length for HARQ (excludes filler bits)."""
        return self._n_ldpc - self.k_filler

    @property
    def n_ldpc(self):
        """Number of codeword bits before rate-matching, including
        filler bits"""
        return self._n_ldpc

    @property
    def pcm(self):
        """Parity-check matrix for given code parameters"""
        return self._pcm

    @property
    def z(self):
        """Lifting factor of the basegraph"""
        return self._z

    @property
    def num_bits_per_symbol(self):
        """Modulation order used for the rate-matching output interleaver"""
        return self._num_bits_per_symbol

    @property
    def out_int(self):
        """Output interleaver sequence as defined in 5.4.2.2"""
        return self._out_int
    
    @property
    def out_int_inv(self):
        """Inverse output interleaver sequence as defined in 5.4.2.2"""
        return self._out_int_inv

    #################
    # Utility methods
    #################

    def generate_out_int(self, n, num_bits_per_symbol):
        """Generates LDPC output interleaver sequence as defined in
        Sec 5.4.2.2 in [3GPPTS38212_LDPC]_.

        Parameters
        ----------
        n: int
            Desired output sequence length.

        num_bits_per_symbol: int
            Number of symbols per QAM symbol, i.e., the modulation order.

        Output
        ------
        perm_seq: ndarray of length n
            Containing the permuted indices.

        perm_seq_inv: ndarray of length n
            Containing the inverse permuted indices.

        Note
        ----
        The interleaver pattern depends on the modulation order and helps to
        reduce dependencies in bit-interleaved coded modulation (BICM) schemes
        combined with higher order modulation.
        """
        # allow float inputs, but verify that they represent integer
        if n%1!=0:
            raise ValueError("n must be int.")
        if num_bits_per_symbol%1!=0:
            raise ValueError("num_bits_per_symbol must be int.")
        n = int(n)
        if n<=0:
            raise ValueError("n must be a positive integer.")
        if num_bits_per_symbol<=0:
            raise ValueError("num_bits_per_symbol must be a positive integer.")
        num_bits_per_symbol = int(num_bits_per_symbol)

        if n%num_bits_per_symbol!=0:
            raise ValueError("n must be a multiple of num_bits_per_symbol.")

        # pattern as defined in Sec 5.4.2.2
        perm_seq = np.zeros(n, dtype=int)
        for j in range(int(n/num_bits_per_symbol)):
            for i in range(num_bits_per_symbol):
                perm_seq[i + j*num_bits_per_symbol] \
                    = int(i * int(n/num_bits_per_symbol) + j)

        perm_seq_inv = np.argsort(perm_seq)

        return perm_seq, perm_seq_inv

    def _sel_basegraph(self, k, r, bg_=None):
        """Select basegraph according to [3GPPTS38212_LDPC]_ and check for consistency."""

        # if bg is explicitly provided, we only check for consistency
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

        # check for consistency
        if bg=="bg1" and k>8448:
            raise ValueError("K is not supported by BG1 (too large).")

        if bg=="bg2" and k>3840:
            raise ValueError(
                f"K is not supported by BG2 (too large) k ={k}.")

        if bg=="bg1" and r<1/3 and not self._allow_low_rates:
            raise ValueError("Only coderate>1/3 supported for BG1. \
            Remark: Repetition coding is currently not supported.")

        if bg=="bg2" and r<1/5 and not self._allow_low_rates:
            raise ValueError("Only coderate>1/5 supported for BG2. \
            Remark: Repetition coding is currently not supported.")

        return bg

    def _load_basegraph(self, i_ls, bg):
        """Helper to load basegraph from csv files.

        ``i_ls`` is sub_index of the basegraph and fixed during lifting
        selection.
        """

        if i_ls > 7:
            raise ValueError("i_ls too large.")

        if i_ls < 0:
            raise ValueError("i_ls cannot be negative.")

        # csv files are taken from 38.212 and dimension is explicitly given
        if bg=="bg1":
            bm = np.zeros([46, 68]) - 1 # init matrix with -1 (None positions)
        elif bg=="bg2":
            bm = np.zeros([42, 52]) - 1 # init matrix with -1 (None positions)
        else:
            raise ValueError("Basegraph not supported.")

        # and load the basegraph from csv format in folder "codes"
        source = files(codes).joinpath(f"5G_{bg}.csv")
        with as_file(source) as codes.csv:
            bg_csv = np.genfromtxt(codes.csv, delimiter=";")

        # reconstruct BG for given i_ls
        r_ind = 0
        for r in np.arange(2, bg_csv.shape[0]):
            # check for next row index
            if not np.isnan(bg_csv[r, 0]):
                r_ind = int(bg_csv[r, 0])
            c_ind = int(bg_csv[r, 1]) # second column in csv is column index
            value = bg_csv[r, i_ls + 2] # i_ls entries start at offset 2
            bm[r_ind, c_ind] = value

        return bm

    def _lift_basegraph(self, bm, z):
        """Lift basegraph with lifting factor ``z`` and shifted identities as
        defined by the entries of ``bm``."""

        num_nonzero = np.sum(bm>=0) # num of non-neg elements in bm

        # init all non-zero row/column indices
        r_idx = np.zeros(z*num_nonzero)
        c_idx = np.zeros(z*num_nonzero)
        data = np.ones(z*num_nonzero)

        # row/column indices of identity matrix for lifting
        im = np.arange(z)

        idx = 0
        for r in range(bm.shape[0]):
            for c in range(bm.shape[1]):
                if bm[r,c]==-1: # -1 is used as all-zero matrix placeholder
                    pass #do nothing (sparse)
                else:
                    # roll matrix by bm[r,c]
                    c_roll = np.mod(im+bm[r,c], z)
                    # append rolled identity matrix to pcm
                    r_idx[idx*z:(idx+1)*z] = r*z + im
                    c_idx[idx*z:(idx+1)*z] = c*z + c_roll
                    idx += 1

        # generate lifted sparse matrix from indices
        pcm = sp.sparse.csr_matrix((data,(r_idx, c_idx)),
                                   shape=(z*bm.shape[0], z*bm.shape[1]))
        return pcm

    def _sel_lifting(self, k, bg):
        """Select lifting as defined in Sec. 5.2.2 in [3GPPTS38212_LDPC]_.

        We assume B < K_cb, thus B'= B and C = 1, i.e., no
        additional CRC is appended. Thus, K' = B'/C = B and B is our K.

        Z is the lifting factor.
        i_ls is the set index ranging from 0...7 (specifying the exact bg
        selection).
        k_b is the number of information bit columns in the basegraph.
        """
        # lifting set according to 38.212 Tab 5.3.2-1
        s_val = [[2, 4, 8, 16, 32, 64, 128, 256],
                [3, 6, 12, 24, 48, 96, 192, 384],
                [5, 10, 20, 40, 80, 160, 320],
                [7, 14, 28, 56, 112, 224],
                [9, 18, 36, 72, 144, 288],
                [11, 22, 44, 88, 176, 352],
                [13, 26, 52, 104, 208],
                [15, 30, 60, 120, 240]]

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

        # find the min of Z from Tab. 5.3.2-1 s.t. k_b*Z>=K'
        min_val = 100000
        z = 0
        i_ls = 0
        i = -1
        for s in s_val:
            i += 1
            for s1 in s:
                x = k_b *s1
                if  x >= k:
                    # valid solution
                    if x < min_val:
                        min_val = x
                        z = s1
                        i_ls = i

        # and set K=22*Z for bg1 and K=10Z for bg2
        if bg == "bg1":
            k_b = 22
        else:
            k_b = 10

        return z, i_ls, k_b

    def _gen_submat(self, bm, k_b, z, bg):
        """Split the basegraph into multiple sub-matrices such that efficient
        encoding is possible.
        """
        g = 4 # code property (always fixed for 5G)
        mb = bm.shape[0] # number of CN rows in basegraph (BG property)

        bm_a = bm[0:g, 0:k_b]
        bm_b = bm[0:g, k_b:(k_b+g)]
        bm_c1 = bm[g:mb, 0:k_b]
        bm_c2 = bm[g:mb, k_b:(k_b+g)]

        # H could be sliced immediately (but easier to implement if based on B)
        hm_a = self._lift_basegraph(bm_a, z)

        # not required for encoding, but helpful for debugging
        # hm_b = self._lift_basegraph(bm_b, z)

        hm_c1 = self._lift_basegraph(bm_c1, z)
        hm_c2 = self._lift_basegraph(bm_c2, z)

        hm_b_inv = self._find_hm_b_inv(bm_b, z, bg)

        return hm_a, hm_b_inv, hm_c1, hm_c2

    def _find_hm_b_inv(self, bm_b, z, bg):
        """ For encoding we need to find the inverse of `hm_b` such that
        `hm_b^-1 * hm_b = I`.

        Could be done sparse
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
        where P_B and P_A are shifted identities

        The inverse can be found by solving a linear system of equations
        hm_b_inv =
        [P_B^-1, P_B^-1, P_B^-1, P_B^-1,
         I + P_A*P_B^-1, P_A*P_B^-1, P_A*P_B^-1, P_A*P_B^-1,
         I+P_A*P_B^-1, I+P_A*P_B^-1, P_A*P_B^-1, P_A*P_B^-1,
         P_A*P_B^-1, P_A*P_B^-1, P_A*P_B^-1, I+P_A*P_B^-1]

        Note: the inverse of B is simply a shifted identity matrix with
        negative shift direction.
        """

        # permutation indices
        pm_a= int(bm_b[0,0])
        if bg=="bg1":
            pm_b_inv = int(-bm_b[1, 0])
        else: # structure of B is slightly different for bg2
            pm_b_inv = int(-bm_b[2, 0])

        hm_b_inv = np.zeros([4*z, 4*z])

        im = np.eye(z)

        am = np.roll(im, pm_a, axis=1)
        b_inv = np.roll(im, pm_b_inv, axis=1)
        ab_inv = np.matmul(am, b_inv)

        # row 0
        hm_b_inv[0:z, 0:z] = b_inv
        hm_b_inv[0:z, z:2*z] = b_inv
        hm_b_inv[0:z, 2*z:3*z] = b_inv
        hm_b_inv[0:z, 3*z:4*z] = b_inv

        # row 1
        hm_b_inv[z:2*z, 0:z] = im + ab_inv
        hm_b_inv[z:2*z, z:2*z] = ab_inv
        hm_b_inv[z:2*z, 2*z:3*z] = ab_inv
        hm_b_inv[z:2*z, 3*z:4*z] = ab_inv

        # row 2
        if bg=="bg1":
            hm_b_inv[2*z:3*z, 0:z] = ab_inv
            hm_b_inv[2*z:3*z, z:2*z] = ab_inv
            hm_b_inv[2*z:3*z, 2*z:3*z] = im + ab_inv
            hm_b_inv[2*z:3*z, 3*z:4*z] = im + ab_inv
        else: # for bg2 the structure is slightly different
            hm_b_inv[2*z:3*z, 0:z] = im + ab_inv
            hm_b_inv[2*z:3*z, z:2*z] = im + ab_inv
            hm_b_inv[2*z:3*z, 2*z:3*z] = ab_inv
            hm_b_inv[2*z:3*z, 3*z:4*z] = ab_inv

        # row 3
        hm_b_inv[3*z:4*z, 0:z] = ab_inv
        hm_b_inv[3*z:4*z, z:2*z] = ab_inv
        hm_b_inv[3*z:4*z, 2*z:3*z] = ab_inv
        hm_b_inv[3*z:4*z, 3*z:4*z] = im + ab_inv

        # return results as sparse matrix
        return sp.sparse.csr_matrix(hm_b_inv)

    def _mat_to_ind(self, mat):
        """Helper to transform matrix into index representation for
        tf.gather. An index pointing to the `last_ind+1` is used for non-existing edges due to irregular degrees."""
        m = mat.shape[0]
        n = mat.shape[1]

        # transpose mat for sorted column format
        c_idx, r_idx, _ = sp.sparse.find(mat.transpose())

        # sort indices explicitly, as scipy.sparse.find changed from column to
        # row sorting in scipy>=1.11
        idx = np.argsort(r_idx)
        c_idx = c_idx[idx]
        r_idx = r_idx[idx]

        # find max number of no-zero entries
        n_max = np.max(mat.getnnz(axis=1))

        # init index array with n (pointer to last_ind+1, will be a default
        # value)
        gat_idx = np.zeros([m, n_max]) + n

        r_val = -1
        c_val = 0
        for idx in range(len(c_idx)):
            # check if same row or if a new row starts
            if r_idx[idx] != r_val:
                r_val = r_idx[idx]
                c_val = 0
            gat_idx[r_val, c_val] = c_idx[idx]
            c_val += 1

        gat_idx = tf.cast(tf.constant(gat_idx), tf.int32)
        return gat_idx

    def _matmul_gather(self, mat, vec):
        """Implements a fast sparse matmul via gather function."""

        # add 0 entry for gather-reduce_sum operation
        # (otherwise ragged Tensors are required)
        bs = tf.shape(vec)[0]
        vec = tf.concat([vec, tf.zeros([bs, 1], dtype=self.rdtype)], 1)

        retval = tf.gather(vec, mat, batch_dims=0, axis=1)
        retval = tf.reduce_sum(retval, axis=-1)

        return retval

    def _encode_fast(self, s):
        """Main encoding function based on gathering function."""
        p_a = self._matmul_gather(self._pcm_a_ind, s)
        p_a = self._matmul_gather(self._pcm_b_inv_ind, p_a)

        # calc second part of parity bits p_b
        # second parities are given by C_1*s' + C_2*p_a' + p_b' = 0
        p_b_1 = self._matmul_gather(self._pcm_c1_ind, s)
        p_b_2 = self._matmul_gather(self._pcm_c2_ind, p_a)
        p_b = p_b_1 + p_b_2

        c = tf.concat([s, p_a, p_b], 1)

        # faster implementation of mod-2 operation c = tf.math.mod(c, 2)
        c_uint8 = tf.cast(c, tf.uint8)
        c_bin = tf.bitwise.bitwise_and(c_uint8, tf.constant(1, tf.uint8))
        c = tf.cast(c_bin, self.rdtype)

        c = tf.expand_dims(c, axis=-1) # returns nx1 vector
        return c

    def _validate_rv_list(self, rv_list):
        """Validate that all RV names in the list are valid.

        Args:
            rv_list (list): List of RV name strings to validate.

        Raises:
            ValueError: If any RV name is invalid.
        """
        valid_rvs = {"rv0", "rv1", "rv2", "rv3"}
        for rv_name in rv_list:
            if rv_name not in valid_rvs:
                raise ValueError(
                    f"Invalid RV name '{rv_name}'. Valid RV names are: {sorted(valid_rvs)}"
                )

    def _get_rv_starts(self) -> dict:
        """Get RV starting positions mapping as per 3GPP TS 38.212.

        Returns
        -------
        dict: Mapping from RV names to starting positions.
        """
        return {
            "rv0": 2 * self.z,
            "rv1": self.n_cb // 4,
            "rv2": self.n_cb // 2,
            "rv3": 3 * self.n_cb // 4
        }

    def build(self, input_shape, **kwargs):
        """"Build block."""
        # check if k and input shape match
        if input_shape[-1]!=self._k:
            raise ValueError("Last dimension must be of length k.")

    def call(self, bits, rv=None):
        """5G LDPC encoding function including rate-matching.

        This function returns the encoded codewords as specified by the 3GPP NR Initiative [3GPPTS38212_LDPC]_ including puncturing and shortening.

        Args:

        bits (tf.float): Tensor of shape `[...,k]` containing the
                information bits to be encoded.
        rv (list, optional): List of redundancy version strings to generate. Can contain
                any combination of ["rv0", "rv1", "rv2", "rv3"] in any order,
                including repeats. If None, defaults to single RV0 encoding without
                adding an RV dimension to the output.

        Returns:

        `tf.float`: Tensor of shape `[..., n]` if rv is None (single RV0), or
                    `[..., num_rv, n]` if rv is provided, where num_rv is the
                    length of the rv list.
        """

        # Determine HARQ mode and set RV list
        harq_mode = rv is not None
        if rv is None:
            rv = ["rv0"]

        # Validate RV names
        self._validate_rv_list(rv)

        # Reshape inputs to [...,k]
        input_shape = bits.get_shape().as_list()
        new_shape = [-1, input_shape[-1]]
        u = tf.reshape(bits, new_shape)

        # assert if bits are non-binary
        if self._check_input:
            tf.debugging.assert_equal(
                tf.reduce_min(
                    tf.cast(
                        tf.logical_or(
                            tf.equal(u, tf.constant(0, self.rdtype)),
                            tf.equal(u, tf.constant(1, self.rdtype)),
                            ),
                        self.rdtype)),
                tf.constant(1, self.rdtype),
                "Input must be binary.")
            # input datatype consistency should be only evaluated once
            self._check_input = False

        batch_size = tf.shape(u)[0]

        # add "filler" bits to last positions to match info bit length k_ldpc
        u_fill = tf.concat([u,
            tf.zeros([batch_size, self._k_ldpc-self._k], self.rdtype)],axis=1)

        # use optimized encoding based on tf.gather
        c = self._encode_fast(u_fill)

        c = tf.reshape(c, [batch_size, self._n_ldpc]) # remove last dim

        # remove filler bits at pos (k, k_ldpc)
        c_no_filler1 = tf.slice(c, [0, 0], [batch_size, self._k])
        c_no_filler2 = tf.slice(c,
                               [0, self._k_ldpc],
                               [batch_size, self._n_ldpc - self._k_ldpc])

        c_no_filler = tf.concat([c_no_filler1, c_no_filler2], 1)

        # get RV starting positions mapping
        rv_starts = self._get_rv_starts()

        c_short_list = []
        
        for rv_name in rv:
            start = rv_starts[rv_name]

            # check if circular wrap occurs
            if start + self.n <= self.n_cb:
                # no wrap: simple slice from start to start+n
                c_short_rv = tf.slice(c_no_filler, [0, start], [batch_size, self.n])
            else:
                # wrap occurs: concatenate two slices
                # first part: from start to end of buffer
                first_part_size = self.n_cb - start
                first_part = tf.slice(c_no_filler, [0, start],
                                    [batch_size, first_part_size])

                # second part: from beginning of buffer
                second_part_size = self.n - first_part_size
                second_part = tf.slice(c_no_filler, [0, 0],
                                     [batch_size, second_part_size])

                # concatenate the two parts
                c_short_rv = tf.concat([first_part, second_part], axis=-1)

            # if num_bits_per_symbol is provided, apply output interleaver as
            # specified in Sec. 5.4.2.2 in 38.212
            if self._num_bits_per_symbol is not None:
                c_short_rv = tf.gather(c_short_rv, self._out_int, axis=-1)

            # Reshape to [batch_size, 1, n] for proper stacking
            c_short_rv = tf.expand_dims(c_short_rv, axis=1)
            c_short_list.append(c_short_rv)

        # stack all RV versions: [batch_size, num_rv, n]
        c_short = tf.concat(c_short_list, axis=1)

        # reshape to match original input dimensions
        output_shape = input_shape[0:-1] + [len(rv), self.n]
        output_shape[0] = -1  # It can be None
        c_reshaped = tf.reshape(c_short, output_shape)

        # remove RV dimension if not in HARQ mode
        if not harq_mode:
            c_reshaped = tf.squeeze(c_reshaped, axis=-2)  # Remove the RV dimension

        return c_reshaped
