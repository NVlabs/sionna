#
# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Layers for LDPC channel encoding and utility functions."""

import tensorflow as tf
import numpy as np
import scipy as sp
from tensorflow.keras.layers import Layer
from importlib_resources import files, as_file
from . import codes # pylint: disable=relative-beyond-top-level
import numbers # to check if n, k are numbers

class AllZeroEncoder(Layer):
    """AllZeroEncoder(k, n, dtype=tf.float32, **kwargs)

    Dummy encoder that always outputs the all-zero codeword of length ``n``.
    Note that this encoder is a dummy encoder and does NOT perform real
    encoding!

    The class inherits from the Keras layer class and can be used as layer in a
    Keras model.

    Parameters
    ----------
        k: int
            Defining the number of information bit per codeword.

        n: int
            Defining the desired codeword length.

        dtype: tf.DType
            Defaults to `tf.float32`. Defines the datatype for internal
            calculations and the output dtype.

    Input
    -----
        inputs: [...,k], tf.float32
            2+D tensor containing arbitrary values (not used!).

    Output
    ------
        : [...,n], tf.float32
            2+D tensor containing all-zero codewords.

    Raises
    ------
        AssertionError
            ``k`` and ``n`` must be positive integers and ``k`` must be smaller
            (or equal) than ``n``.

        AssertionError
            If ``k`` is not `int`.

        AssertionError
            If ``n`` is not `int`.

    Note
    ----
        As the all-zero codeword is part of any linear code, it is often used
        to simulate BER curves of arbitrary (LDPC) codes without the need of
        having access to the actual generator matrix. However, this `"all-zero
        codeword trick"` requires symmetric channels (such as BPSK), otherwise
        scrambling is required (cf. [Pfister]_ for further details).

        This encoder is a dummy encoder that is needed for some all-zero
        codeword simulations independent of the input. It does NOT perform
        real encoding although the information bits are taken as input.
        This is just to ensure compatibility with other encoding layers.
    """

    def __init__(self,
                 k,
                 n,
                 dtype=tf.float32,
                 **kwargs):

        super().__init__(dtype=dtype, **kwargs)

        #assert error if r>1 or k,n are negativ
        assert isinstance(k, numbers.Number), "k must be a number."
        assert isinstance(n, numbers.Number), "n must be a number."
        k = int(k) # k or n can be float (e.g. as result of n=k*r)
        n = int(n) # k or n can be float (e.g. as result of n=k*r)
        assert k>-1, "k cannot be negative."
        assert n>-1, "n cannot be negative."
        assert n>=k, "Invalid coderate (>1)."
        # init encoder parameters
        self._k = k
        self._n = n
        self._coderate = k / n

    #########################################
    # Public methods and properties
    #########################################

    @property
    def k(self):
        """Number of information bits per codeword."""
        return self._k

    @property
    def n(self):
        "Codeword length."
        return self._n

    @property
    def coderate(self):
        """Coderate of the LDPC code."""
        return self._coderate

    #########################
    # Keras layer functions
    #########################

    def build(self, input_shape):
        """Nothing to build."""
        pass

    def call(self, inputs):
        """Encoding function that outputs the all-zero codeword.

        This function returns the all-zero codeword of shape `[..., n]`.
        Note that this encoder is a dummy encoder and does NOT perform real
        encoding!

        Args:
            inputs (tf.float32): Tensor of arbitrary shape.

        Returns:
            `tf.float32`: Tensor of shape `[...,n]`.

        Note:
            This encoder is a dummy encoder that is needed for some all-zero
            codeword simulations independent of the input. It does NOT perform
            real encoding although the information bits are taken as input.
            This is just to ensure compatibility with other encoding layers.
        """
        # keep shape of first dimensions
        # return an all-zero tensor of shape [..., n]
        output_shape = tf.concat([tf.shape(inputs)[:-1],
                                  tf.constant(self._n, shape=[1])],
                                  0)
        c = tf.zeros(output_shape, dtype=super().dtype)
        return c

class LDPC5GEncoder(Layer):
    # pylint: disable=line-too-long
    """LDPC5GEncoder(k, n, num_bits_per_symbol=None, dtype=tf.float32, **kwargs)

    5G NR LDPC Encoder following the 3GPP NR Initiative [3GPPTS38212_LDPC]_
    including rate-matching.

    The class inherits from the Keras layer class and can be used as layer in a
    Keras model.

    Parameters
    ----------
        k: int
            Defining the number of information bit per codeword.

        n: int
            Defining the desired codeword length.

        num_bits_per_symbol: int or None
            Defining the number of bits per QAM symbol. If this parameter is
            explicitly provided, the codeword will be interleaved after
            rate-matching as specified in Sec. 5.4.2.2 in [3GPPTS38212_LDPC]_.

        dtype: tf.DType
            Defaults to `tf.float32`. Defines the output datatype of the layer
            (internal precision remains `tf.uint8`).

    Input
    -----
        inputs: [...,k], tf.float32
            2+D tensor containing the information bits to be
            encoded.

    Output
    ------
        : [...,n], tf.float32
            2+D tensor of same shape as inputs besides last dimension has
            changed to `n` containing the encoded codeword bits.

    Attributes
    ----------
        k: int
            Defining the number of information bit per codeword.

        n: int
            Defining the desired codeword length.

        coderate: float
            Defining the coderate r= ``k`` / ``n``.

        n_ldpc: int
            An integer defining the total codeword length (before
            punturing) of the lifted parity-check matrix.

        k_ldpc: int
            An integer defining the total information bit length
            (before zero removal) of the lifted parity-check matrix. Gap to
            ``k`` must be filled with so-called filler bits.

        num_bits_per_symbol: int or None.
            Defining the number of bits per QAM symbol. If this parameter is
            explicitly provided, the codeword will be interleaved after
            rate-matching as specified in Sec. 5.4.2.2 in [3GPPTS38212_LDPC]_.

        out_int: [n], ndarray of int
            Defining the rate-matching output interleaver sequence.

        out_int_inv: [n], ndarray of int
            Defining the inverse rate-matching output interleaver sequence.

        _check_input: bool
            A boolean that indicates whether the input vector
            during call of the layer should be checked for consistency (i.e.,
            binary).

        _bg: str
            Denoting the selected basegraph (either `bg1` or `bg2`).

        _z: int
            Denoting the lifting factor.

        _i_ls: int
            Defining which version of the basegraph to load.
            Can take values between 0 and 7.

        _k_b: int
            Defining the number of `information bit columns` in the
            basegraph. Determined by the code design procedure in
            [3GPPTS38212_LDPC]_.

        _bm: ndarray
            An ndarray defining the basegraph.

        _pcm: sp.sparse.csr_matrix
            A sparse matrix of shape `[k_ldpc-n_ldpc, n_ldpc]`
            containing the sparse parity-check matrix.

    Raises
    ------
        AssertionError
            If ``k`` is not `int`.

        AssertionError
            If ``n`` is not `int`.

        ValueError
            If ``code_length`` is not supported.

        ValueError
            If `dtype` is not supported.

        ValueError
            If ``inputs`` contains other values than `0` or `1`.

        InvalidArgumentError
            When rank(``inputs``)<2.

        InvalidArgumentError
            When shape of last dim is not ``k``.

    Note
    ----
        As specified in [3GPPTS38212_LDPC]_, the encoder also performs
        puncturing and shortening. Thus, the corresponding decoder needs to
        `invert` these operations, i.e., must be compatible with the 5G
        encoding scheme.
    """

    def __init__(self,
                 k,
                 n,
                 num_bits_per_symbol=None,
                 dtype=tf.float32,
                 **kwargs):

        super().__init__(dtype=dtype, **kwargs)

        assert isinstance(k, numbers.Number), "k must be a number."
        assert isinstance(n, numbers.Number), "n must be a number."
        k = int(k) # k or n can be float (e.g. as result of n=k*r)
        n = int(n) # k or n can be float (e.g. as result of n=k*r)

        if dtype is not tf.float32:
            print("Note: decoder uses tf.float32 for internal calculations.")

        if dtype not in (tf.float16, tf.float32, tf.float64, tf.int8,
            tf.int32, tf.int64, tf.uint8, tf.uint16, tf.uint32):
            raise ValueError("Unsupported dtype.")
        self._dtype = dtype

        if k>8448:
            raise ValueError("Unsupported code length (k too large).")
        if k<12:
            raise ValueError("Unsupported code length (k too small).")

        if n>(316*384):
            raise ValueError("Unsupported code length (n too large).")
        if n<0:
            raise ValueError("Unsupported code length (n negative).")

        # init encoder parameters
        self._k = k # number of input bits (= input shape)
        self._n = n # the desired length (= output shape)
        self._coderate = k / n
        self._check_input = True # check input for consistency (i.e., binary)

        if self._coderate>(11/12):
            raise ValueError(
                    f"Unsupported coderate (r>11/12); n={n}, k={k}.")
        if self._coderate<(1/5):
            # outer rep. coding currently not supported
            raise ValueError("Unsupported coderate (r<1/5).")

        # construct the basegraph according to 38.212
        self._bg = self._sel_basegraph(self._k, self._coderate)
        self._z, self._i_ls, self._k_b = self._sel_lifting(self._k, self._bg)
        self._bm = self._load_basegraph(self._i_ls, self._bg)

        # total number of codeword bits
        self._n_ldpc = self._bm.shape[1] * self._z
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
            self._out_int, self._out_int_inv  = self._generate_out_int(self._n,
                                                    self._num_bits_per_symbol)

    #########################################
    # Public methods and properties
    #########################################

    @property
    def k(self):
        """Number of input information bits."""
        return self._k

    @property
    def n(self):
        "Number of output codeword bits."
        return self._n

    @property
    def coderate(self):
        """Coderate of the LDPC code after rate-matching."""
        return self._coderate

    @property
    def k_ldpc(self):
        """Number of LDPC information bits after rate-matching."""
        return self._k_ldpc

    @property
    def n_ldpc(self):
        """Number of LDPC codeword bits before rate-matching."""
        return self._n_ldpc

    @property
    def pcm(self):
        """Parity-check matrix for given code parameters."""
        return self._pcm

    @property
    def z(self):
        """Lifting factor of the basegraph."""
        return self._z

    @property
    def num_bits_per_symbol(self):
        """Modulation order used for the rate-matching output interleaver."""
        return self._num_bits_per_symbol

    @property
    def out_int(self):
        """Output interleaver sequence as defined in 5.4.2.2."""
        return self._out_int
    @property
    def out_int_inv(self):
        """Inverse output interleaver sequence as defined in 5.4.2.2."""
        return self._out_int_inv

    #########################
    # Utility methods
    #########################

    def _generate_out_int(self, n, num_bits_per_symbol):
        """"Generates LDPC output interleaver sequence as defined in
        Sec 5.4.2.2 in [3GPPTS38212_LDPC]_.

        Parameters
        ----------
        n: int
            Desired output sequence length.

        num_bits_per_symbol: int
            Number of symbols per QAM symbol, i.e., the modulation order.

        Outputs
        -------
        (perm_seq, perm_seq_inv):
            Tuple:

        perm_seq: ndarray of length n
            Containing the permuted indices.

        perm_seq_inv: ndarray of length n
            Containing the inverse permuted indices.

        Note
        ----
        The interleaver pattern depends on the modulation order and helps to reduce
        dependencies in bit-interleaved coded modulation (BICM) schemes.
        """
        # allow float inputs, but verify that they represent integer
        assert(n%1==0), "n must be int."
        assert(num_bits_per_symbol%1==0), "num_bits_per_symbol must be int."
        n = int(n)
        assert(n>0), "n must be a positive integer."
        assert(num_bits_per_symbol>0), \
                    "num_bits_per_symbol must be a positive integer."
        num_bits_per_symbol = int(num_bits_per_symbol)

        assert(n%num_bits_per_symbol==0),\
            "n must be a multiple of num_bits_per_symbol."

        # pattern as defined in Sec 5.4.2.2
        perm_seq = np.zeros(n, dtype=int)
        for j in range(int(n/num_bits_per_symbol)):
            for i in range(num_bits_per_symbol):
                perm_seq[i + j*num_bits_per_symbol] \
                    = int(i * int(n/num_bits_per_symbol) + j)

        perm_seq_inv = np.argsort(perm_seq)

        return perm_seq, perm_seq_inv

    def _sel_basegraph(self, k, r):
        """Select basegraph according to [3GPPTS38212_LDPC]_."""

        if k <= 292:
            bg = "bg2"
        elif k <= 3824 and r <= 0.67:
            bg = "bg2"
        elif r <= 0.25:
            bg = "bg2"
        else:
            bg = "bg1"

        # add for consistency
        if bg=="bg1" and k>8448:
            raise ValueError("K is not supported by BG1 (too large).")

        if bg=="bg2" and k>3840:
            raise ValueError(
                f"K is not supported by BG2 (too large) k ={k}.")

        if bg=="bg1" and r<1/3:
            raise ValueError("Only coderate>1/3 supported for BG1. \
            Remark: Repetition coding is currently not supported.")

        if bg=="bg2" and r<1/5:
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

        # generate lifted sparse matrix from incides
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
        #hm_b = self._lift_basegraph(bm_b, z)

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
        where P_B and P_A are Shifted identities.

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
        where P_B and P_A are Shifted identities

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
        vec = tf.concat([vec, tf.zeros([bs, 1], dtype=self.dtype)], 1)

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
        c = tf.cast(c_bin, self.dtype)

        c = tf.expand_dims(c, axis=-1) # returns nx1 vector
        return c

    #########################
    # Keras layer functions
    #########################

    def build(self, input_shape):
        """"Build layer."""
        # check if k and input shape match
        assert (input_shape[-1]==self._k), "Last dimension must be of length k."
        assert (len(input_shape)>=2), "Rank of input must be at least 2."

    def call(self, inputs):
        """5G LDPC encoding function including rate-matching.

        This function returns the encoded codewords as specified by the 3GPP NR Initiative [3GPPTS38212_LDPC]_ including puncturing and shortening.

        Args:
            inputs (tf.float32): Tensor of shape `[...,k]` containing the
                information bits to be encoded.

        Returns:
            `tf.float32`: Tensor of shape `[...,n]`.

        Raises:
            ValueError: If ``inputs`` contains other values than `0` or `1`.

            InvalidArgumentError: When rank(``inputs``)<2.

            InvalidArgumentError: When shape of last dim is not ``k``.
        """

        tf.debugging.assert_type(inputs, self.dtype, "Invalid input dtype.")

        # Reshape inputs to [...,k]
        input_shape = inputs.get_shape().as_list()
        new_shape = [-1, input_shape[-1]]
        u = tf.reshape(inputs, new_shape)

        # assert if u is non binary
        if self._check_input:
            tf.debugging.assert_equal(
                tf.reduce_min(
                    tf.cast(
                        tf.logical_or(
                            tf.equal(u, tf.constant(0, self.dtype)),
                            tf.equal(u, tf.constant(1, self.dtype)),
                            ),
                        self.dtype)),
                tf.constant(1, self.dtype),
                "Input must be binary.")
            # input datatype consistency should be only evaluated once
            self._check_input = False

        batch_size = tf.shape(u)[0]

        # add "filler" bits to last positions to match info bit length k_ldpc
        u_fill = tf.concat([u,
                    tf.zeros([batch_size, self._k_ldpc-self._k], self.dtype)],
                            1)

        # use optimized encoding based on tf.gather
        c = self._encode_fast(u_fill)

        c = tf.reshape(c, [batch_size, self._n_ldpc]) # remove last dim

        # remove filler bits at pos (k, k_ldpc)
        c_no_filler1 = tf.slice(c, [0, 0], [batch_size, self._k])
        c_no_filler2 = tf.slice(c,
                               [0, self._k_ldpc],
                               [batch_size, self._n_ldpc-self._k_ldpc])

        c_no_filler = tf.concat([c_no_filler1, c_no_filler2], 1)

        # shorten the first 2*Z positions and end after n bits
        # (remaining parity bits can be used for IR-HARQ)
        c_short = tf.slice(c_no_filler, [0, 2*self._z], [batch_size, self.n])
        # incremental redundancy could be generated by accessing the last bits

        # if num_bits_per_symbol is provided, apply output interleaver as
        # specified in Sec. 5.4.2.2 in 38.212
        if self._num_bits_per_symbol is not None:
            c_short = tf.gather(c_short, self._out_int, axis=-1)

        # Reshape c_short so that it matches the original input dimensions
        output_shape = input_shape[0:-1] + [self.n]
        output_shape[0] = -1
        c_reshaped = tf.reshape(c_short, output_shape)

        return tf.cast(c_reshaped, self._dtype)
