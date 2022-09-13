#
# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Layers for Polar encoding including 5G compliant rate-matching and CRC
concatenation."""

from sionna.fec.crc import CRCEncoder
from sionna.fec.polar.utils import generate_5g_ranking
from numpy.core.numerictypes import issubdtype
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Layer
import numbers

class PolarEncoder(Layer):
    """PolarEncoder(frozen_pos, n, dtype=tf.float32)

    Polar encoder for given code parameters.

    This layer performs polar encoding for the given ``k`` information bits and
    the `frozen set` (i.e., indices of frozen positions) specified by
    ``frozen_pos``.

    The class inherits from the Keras layer class and can be used as layer in a
    Keras model.

    Parameters
    ----------
        frozen_pos: ndarray
            Array of `int` defining the `n-k` frozen indices, i.e., information
            bits are mapped onto the `k` complementary positions.

        n: int
            Defining the codeword length.

        dtype: tf.DType
            Defaults to `tf.float32`. Defines the output datatype of the layer
            (internal precision is `tf.uint8`).

    Input
    -----
        inputs: [...,k], tf.float32
            2+D tensor containing the information bits to be encoded.

    Output
    ------
        : [...,n], tf.float32
            2+D tensor containing the codeword bits.

    Raises
    ------
        AssertionError
            ``k`` and ``n`` must be positive integers and ``k`` must be smaller
            (or equal) than ``n``.

        AssertionError
            If ``n`` is not a power of 2.

        AssertionError
            If the number of elements in ``frozen_pos`` is great than ``n``.

        AssertionError
            If ``frozen_pos`` does not consists of `int`.

        ValueError
            If ``dtype`` is not supported.

        ValueError
            If ``inputs`` contains other values than `0` or `1`.

        TypeError
            If ``inputs`` is not `tf.float32`.

        InvalidArgumentError
            When rank(``inputs``)<2.

        InvalidArgumentError
            When shape of last dim is not ``k``.

    Note
    ----
        As commonly done, we assume frozen bits are set to `0`. Please note
        that - although its practical relevance is only little - setting frozen
        bits to `1` may result in `affine` codes instead of linear code as the
        `all-zero` codeword is not necessarily part of the code any more.
    """

    def __init__(self,
                 frozen_pos,
                 n,
                 dtype=tf.float32):

        if dtype not in (tf.float16, tf.float32, tf.float64, tf.int8,
            tf.int32, tf.int64, tf.uint8, tf.uint16, tf.uint32):
            raise ValueError("Unsupported dtype.")

        super().__init__(dtype=dtype)

        assert isinstance(n, numbers.Number), "n must be a number."
        n = int(n) # n can be float (e.g. as result of n=k*r)
        assert issubdtype(frozen_pos.dtype, int), "frozen_pos must \
                                                   consist of ints."
        assert len(frozen_pos)<=n, "Number of elements in frozen_pos cannot \
                                   be greater than n."

        assert np.log2(n)==int(np.log2(n)), "n must be a power of 2."

        self._k = n - len(frozen_pos)
        self._n = n
        self._frozen_pos = frozen_pos

        # generate info positions
        self._info_pos = np.setdiff1d(np.arange(self._n), frozen_pos)
        assert self._k==len(self._info_pos), "Internal error: invalid " \
                                              "info_pos generated."

        self._check_input = True # check input for bin. values during first call

        self._nb_stages = int(np.log2(self._n))
        self._ind_gather = self._gen_indices(self._n)

    #########################################
    # Public methods and properties
    #########################################

    @property
    def k(self):
        """Number of information bits."""
        return self._k

    @property
    def n(self):
        """Codeword length."""
        return self._n

    @property
    def frozen_pos(self):
        """Frozen positions for Polar decoding."""
        return self._frozen_pos

    @property
    def info_pos(self):
        """Information bit positions for Polar encoding."""
        return self._info_pos

    #########################
    # Utility methods
    #########################

    def _gen_indices(self, n):
        """Pre-calculate encoding indices stage-wise for tf.gather.
        """

        nb_stages = int(np.log2(n))
        # last position denotes empty placeholder (points to element n+1)
        ind_gather = np.ones([nb_stages, n+1]) * n

        for s in range(nb_stages):
            ind_range = np.arange(int(n/2))
            ind_dest = ind_range * 2 - np.mod(ind_range, 2**(s))
            ind_origin = ind_dest + 2**s
            ind_gather[s, ind_dest] = ind_origin # and update gather indices

        ind_gather = tf.constant(ind_gather, dtype=tf.int32)

        return ind_gather

    #########################
    # Keras layer functions
    #########################

    def build(self, input_shape):
        """build and check if ``k`` and ``input_shape`` match."""
        assert (input_shape[-1]==self._k), "Invalid input shape."

    def call(self, inputs):
        """Polar encoding function.

        This function returns the polar encoded codewords for the given
        information bits ``inputs``.

        Args:
            inputs (tf.float32): Tensor of shape `[...,k]` containing the
            information bits to be encoded.

        Returns:
            `tf.float32`: Tensor of shape `[...,n]`.

        Raises:
            ValueError: If ``inputs`` contains other values than `0` or `1`.

            TypeError: If ``inputs`` is not `tf.float32`.

            InvalidArgumentError: When rank(``inputs``)<2.

            InvalidArgumentError: When shape of last dim is not ``k``.
        """

        tf.debugging.assert_type(inputs, self.dtype,
                                 "Invalid input dtype.")

        # Reshape inputs to [...,k]
        tf.debugging.assert_greater(tf.rank(inputs), 1)
        input_shape = inputs.shape
        new_shape = [-1, input_shape[-1]]
        u = tf.reshape(inputs, new_shape)

        # last dim must be of length k
        tf.debugging.assert_equal(tf.shape(u)[-1],
                                  self._k,
                                  "Last dimension must be of length k.")

        # assert if binary=True and u is non binary
        if self._check_input:
            u_test = tf.cast(u, tf.float32) # only for internal check
            tf.debugging.assert_equal(tf.reduce_min(
                                        tf.cast(
                                            tf.logical_or(
                                                tf.equal(u_test, 0.),
                                                tf.equal(u_test, 1.)),
                                        tf.float32)),
                                      1.,
                                      "Input must be binary.")
            # input datatype consistency should be only evaluated once
            self._check_input = False

        # copy info bits to information set; other positions are frozen (=0)

        # return an all-zero tensor of shape [n,...]
        c = tf.zeros([self._n, tf.shape(u)[0]], self.dtype)

        # u has shape bs x k, we now want k x bs
        u_transpose = tf.transpose(u, (1,0)) # batch dim to last pos

        # index vector has at least two axis (= index_depth)
        info_pos_tf = tf.expand_dims(self.info_pos, axis=1)

        c = tf.tensor_scatter_nd_update(c, info_pos_tf, u_transpose)
        c = tf.transpose(c, (1,0))
        x_nan = tf.zeros([tf.shape(c)[0] ,1], self.dtype)
        x = tf.concat([c, x_nan], 1)
        x = tf.cast(x, tf.uint8)

        # loop over all stages
        for s in range(self._nb_stages):
            ind_helper = self._ind_gather[s,:]
            x_add = tf.gather(x, ind_helper, batch_dims=0, axis=1)
            #x = tf.math.logical_xor(x, x_add) # does not work well with XLA
            x = tf.bitwise.bitwise_xor(x, x_add)

        # remove last position
        c_out = x[:,0:self._n]

        # restore original shape
        input_shape_list = input_shape.as_list()
        output_shape = input_shape_list[0:-1] + [self._n]
        output_shape[0] = -1 # to support dynamic shapes
        c_reshaped = tf.reshape(c_out, output_shape)

        # cast to dtype for compatibility with other components
        return tf.cast(c_reshaped, self.dtype)

class Polar5GEncoder(PolarEncoder):
    """Polar5GEncoder(k, n, verbose=False, dtype=tf.float32)

    5G compliant Polar encoder including rate-matching following [3GPPTS38212]_
    for the uplink scenario (`UCI`).

    This layer performs polar encoding for ``k`` information bits and
    rate-matching such that the codeword lengths is ``n``. This includes the CRC
    concatenation and the interleaving as defined in [3GPPTS38212]_.

    Note: `block segmentation` is currently not supported (`I_seq=False`).

    We follow the basic structure from Fig. 6 in [Bioglio_Design]_ with disabled
    downlink interleaver (`I_IL=False`).

    ..  figure:: ../figures/PolarEncoding5G.png

        Fig. 1: Implemented 5G Polar encoding chain following Fig. 6 in
        [Bioglio_Design]_ for the uplink scenario without `block segmentation`.


    For further details we refer to [3GPPTS38212]_, [Bioglio_Design]_ and
    [Hui_ChannelCoding]_.

    The class inherits from the Keras layer class and can be used as layer in a
    Keras model. Further, the class inherits from PolarEncoder.

    Parameters
    ----------
        k: int
            Defining the number of information bit per codeword.

        n: int
            Defining the codeword length.

        verbose: bool
            Defaults to False. If True, rate-matching parameters will be
            printed.

        dtype: tf.DType
            Defaults to tf.float32. Defines the output datatype of the layer
            (internal precision remains tf.uint8).

    Input
    -----
        inputs: [...,k], tf.float32
            2+D tensor containing the information bits to be encoded.

    Output
    ------
        : [...,n], tf.float32
            2+D tensor containing the codeword bits.

    Raises
    ------
        AssertionError
            ``k`` and ``n`` must be positive integers and ``k`` must be smaller
            (or equal) than ``n``.

        AssertionError
            If ``n`` and ``k`` are invalid code parameters (see [3GPPTS38212]_).

        AssertionError
            If ``verbose`` is not `bool`.

        ValueError
            If ``dtype`` is not supported.

    Note
    ----
        We implement the `uplink` Polar coding (`UCI`) scheme from
        [3GPPTS38212]_. Downlink is currently not supported.

        For `12 <= k <= 19` the 3 additional parity bits as defined in
        [3GPPTS38212]_ are not implemented as it would also require a
        modified decoding procedure to materialize the potential gains.

        `Code segmentation` is currently not supported and, thus, ``n`` is
        limited to a maximum length of 1088 codeword bits.

    """

    def __init__(self,
                 k,
                 n,
                 verbose=False,
                 dtype=tf.float32):

        if dtype not in (tf.float16, tf.float32, tf.float64, tf.int8,
            tf.int32, tf.int64, tf.uint8, tf.uint16, tf.uint32):
            raise ValueError("Unsupported dtype.")

        assert isinstance(k, numbers.Number), "k must be a number."
        assert isinstance(n, numbers.Number), "n must be a number."
        k = int(k) # k or n can be float (e.g. as result of n=k*r)
        n = int(n) # k or n can be float (e.g. as result of n=k*r)
        assert n>=k, "Invalid coderate (>1)."
        assert isinstance(verbose, bool), "verbose must be bool."

        self._k_target = k
        self._n_target = n
        self._verbose = verbose

         # Initialize rate-matcher
        crc_degree, n_polar, frozen_pos, idx_rm  = self._init_rate_match(k, n)

        self._frozen_pos = frozen_pos # Required for decoder
        self._ind_rate_matching = idx_rm # Index for gather-based rate-matching

        # Initialize CRC encoder
        self._enc_crc = CRCEncoder(crc_degree, dtype=dtype)

        # Init super-class (PolarEncoder)
        super().__init__(frozen_pos, n_polar, dtype=dtype)

    #########################################
    # Public methods and properties
    #########################################

    @property
    def enc_crc(self):
        """CRC Encoder layer used for CRC concatenation."""
        return self._enc_crc

    @property
    def k_target(self):
        """Number of information bits including rate-matching."""
        return self._k_target

    @property
    def n_target(self):
        """Codeword length including rate-matching."""
        return self._n_target

    def subblock_interleaving(self, u):
        """Input bit interleaving as defined in Sec 5.4.1.1 [3GPPTS38212]_.

        Input
        -----
            u: ndarray
                1D array to be interleaved. Length of ``u`` must be a multiple
                of 32.

        Output
        ------
            : ndarray
                Interleaved version of ``u`` with same shape and dtype as ``u``.

        Raises
        ------
            AssertionError
                If length of ``u`` is not a multiple of 32.

        """

        k = u.shape[-1]
        assert np.mod(k,32)==0, \
            "length for sub-block interleaving must be a multiple of 32."
        y = np.zeros_like(u)

        # Permutation according to Tab 5.4.1.1.1-1 in 38.212
        perm = np.array([0, 1, 2, 4, 3, 5, 6, 7, 8, 16, 9, 17, 10, 18, 11, 19,
                         12, 20, 13, 21, 14, 22, 15, 23, 24, 25, 26, 28, 27,
                         29, 30, 31])

        for n in range(k):
            i = int(np.floor(32*n/k))
            j = perm[i] * k/32 + np.mod(n, k/32)
            j = int(j)
            y[n] = u[j]

        return y

    def channel_interleaver(self, c):
        """Triangular interleaver following Sec. 5.4.1.3 in [3GPPTS38212]_.

        Input
        -----
            c: ndarray
                1D array to be interleaved.

        Output
        ------
            : ndarray
                Interleaved version of ``c`` with same shape and dtype as ``c``.

        """

        n = c.shape[-1] # Denoted as E in 38.212
        c_int = np.zeros_like(c)

        # Find smallest T s.t. T*(T+1)/2 >= n
        t = 0
        while t*(t+1)/2 < n:
            t +=1

        v = np.zeros([t, t])
        ind_k = 0
        for ind_i in range(t):
            for ind_j in range(t-ind_i):
                if ind_k < n:
                    v[ind_i, ind_j] = c[ind_k]
                else:
                    v[ind_i, ind_j] = np.nan # NULL
                # Store nothing otherwise
                ind_k += 1
        ind_k = 0
        for ind_j in range(t):
            for ind_i in range(t-ind_j):
                if not np.isnan(v[ind_i, ind_j]):
                    c_int[ind_k] = v[ind_i, ind_j]
                    ind_k += 1
        return c_int

    #########################
    # Utility methods
    #########################

    def _init_rate_match(self, k_target, n_target):
        """Implementing polar rate matching according to [3GPPTS38212]_.

        Currently, only uplink Polar rate-matching is implemented.

        Please note that this part of the code only runs during the
        initialization and, thus, is not performance critical. For easier
        alignment and traceability with the standard document [3GPPTS38212]_
        the implementation prefers `for loop`-based indexing.

        The relation of terminology between [3GPPTS38212]_ and this code is
        given as:
        `A`...`k_target`
        `E`...`n_target`
        `K`...`k_polar`
        `N`...`n_polar`
        `L`...`k_crc`.
        """

        # Check input for consistency (see Sec. 6.3.1.2.1 for UL)

        # currently not relevant (segmentation not supported)
        # assert k_target<=1706, "Maximum supported codeword length for" \
        # "Polar  coding is 1706."

        assert n_target >=k_target, "n must be larger or equal k."
        assert k_target >= 12, \
                        "k<12 is not supported by the 5G Polar coding scheme."
        assert n_target >= 18, \
                        "n<18 is not supported by the 5G Polar coding scheme."
        assert k_target <= 1013, \
            "k too large - no codeword segmentation supported at the moment."
        assert n_target <= 1088, \
            "n too large - no codeword segmentation supported at the moment."

        # Select CRC polynomials (see Sec. 6.3.1.2.1 for UL)
        if 12<=k_target<=19:
            crc_pol = "CRC6"
            k_crc = 6
        elif k_target >=20:
            crc_pol = "CRC11"
            k_crc = 11
        else:
            raise ValueError("k_target<12 is not supported in 5G NR; please " \
                "use 'channel coding of small block lengths' scheme from " \
                "Sec. 5.3.3 in 3GPP 38.212 instead.")

        # PC bit for k_target = 12-19 bits (see Sec. 6.3.1.3.1 for UL)
        n_pc = 0
        #n_pc_wm = 0
        if k_target<=19:
            #n_pc = 3
            n_pc = 0 # Currently deactivated
            print("Warning: For 12<=k<=19 additional 3 parity-check bits " \
                  "are defined in 38.212. They are currently not " \
                  "implemented by this encoder and, thus, ignored.")
            if n_target-k_target>175:
                #n_pc_wm = 1
                pass

        # No input interleaving for uplink needed

        # Calculate Polar payload length (CRC bits are treated as info bits)
        k_polar = k_target + k_crc + n_pc

        assert k_polar <= n_target, "UE is not expected to be configured " \
                                    "with k_polar + k_crc + n_pc > n_target."

        # Select polar mother code length n_polar
        n_min = 5
        n_max = 10 # For uplink; otherwise 9

        # Select rate-matching scheme following Sec. 5.3.1
        if (n_target <= ((9/8) * 2**(np.ceil(np.log2(n_target))-1)) and
            k_polar/n_target < 9/16):
            n1 = np.ceil(np.log2(n_target))-1
        else:
            n1 = np.ceil(np.log2(n_target))
        n2 = np.ceil(np.log2(8*k_polar)) #Lower bound such that rate > 1/8
        n_polar = int(2**np.max((np.min([n1, n2, n_max]), n_min)))

        # Puncturing and shortening as defined in Sec. 5.4.1.1
        prefrozen_pos = [] # List containing the pre-frozen indices
        if n_target < n_polar:
            if k_polar/n_target <= 7/16:
                # Puncturing
                if self._verbose:
                    print("Using puncturing for rate-matching.")
                n_int =  32 * np.ceil((n_polar-n_target) / 32)
                int_pattern = self.subblock_interleaving(np.arange(n_int))
                for i in range(n_polar-n_target):
                    # Freeze additional bits
                    prefrozen_pos.append(int(int_pattern[i]))
                if n_target >= 3*n_polar/4:
                    t = int(np.ceil(3/4*n_polar - n_target/2) - 1)
                else:
                    t = int(np.ceil(9/16*n_polar - n_target/4) - 1)
                # Extra freezing
                for i in range(t):
                    prefrozen_pos.append(i)
            else:
                # Shortening ("through" sub-block interleaver)
                if self._verbose:
                    print("Using shortening for rate-matching.")
                n_int =  32 * np.ceil((n_polar) / 32)
                int_pattern = self.subblock_interleaving(np.arange(n_int))
                for i in range(n_target, n_polar):
                    prefrozen_pos.append(int_pattern[i])

        # Remove duplicates
        prefrozen_pos = np.unique(prefrozen_pos)

        # Find the remaining n_polar - k_polar - |frozen_set|

        # Load full channel ranking
        ch_ranking, _ = generate_5g_ranking(0, n_polar, sort=False)

        # Remove positions that are already frozen by `pre-freezing` stage
        info_cand = np.setdiff1d(ch_ranking, prefrozen_pos, assume_unique=True)

        # Identify k_polar most reliable positions from candidate positions
        info_pos = []
        for i in range(k_polar):
            info_pos.append(info_cand[-i-1])

        # Sort and create frozen positions for n_polar indices (no shortening)
        info_pos = np.sort(info_pos).astype(int)
        frozen_pos = np.setdiff1d(np.arange(n_polar),
                                  info_pos,
                                  assume_unique=True)

        # Generate tf.gather indices for sub-block interleaver
        ind_sub_int = self.subblock_interleaving(np.arange(n_polar))

        # Rate matching via circular buffer as defined in Sec. 5.4.1.2
        c_int = np.arange(n_polar)
        idx_c_matched = np.zeros([n_target])
        if n_target >= n_polar:
            # Repetition coding
            if self._verbose:
                print("Using repetition coding for rate-matching")
            for ind in range(n_target):
                idx_c_matched[ind] = c_int[np.mod(ind, n_polar)]
        else:
            if k_polar/n_target <= 7/16:
                # Puncturing
                for ind in range(n_target):
                    idx_c_matched[ind] = c_int[ind+n_polar-n_target]
            else:
                # Shortening
                for ind in range(n_target):
                    idx_c_matched[ind] = c_int[ind]

        ind_channel_int = self.channel_interleaver(np.arange(n_target))


        # Combine indices for single tf.gather operation
        ind_t = idx_c_matched[ind_channel_int].astype(int)
        idx_rate_matched= ind_sub_int[ind_t]

        if self._verbose:
            print("Code parameters after rate-matching: " \
                  f"k = {k_target}, n = {n_target}")
            print(f"Polar mother code: k_polar = {k_polar}, " \
                  f"n_polar = {n_polar}")
            print("Using", crc_pol)
            print("Frozen positions: ", frozen_pos)

        return crc_pol, n_polar, frozen_pos, idx_rate_matched

    #########################
    # Keras layer functions
    #########################

    def build(self, input_shape):
        """Build and check if ``k`` and ``input_shape`` match."""
        assert (input_shape[-1]==self._k_target), "Invalid input shape."

    def call(self, inputs):
        """Polar encoding function including rate-matching and CRC encoding.

        This function returns the polar encoded codewords for the given
        information bits ``inputs`` following [3GPPTS38212]_ including
        rate-matching.

        Args:
            inputs (tf.float32): Tensor of shape `[...,k]` containing the
            information bits to be encoded.

        Returns:
            `tf.float32`: Tensor of shape `[...,n]`.

        Raises:
            TypeError: If ``inputs`` is not `tf.float32`.

            InvalidArgumentError: When rank(``inputs``)<2.

            InvalidArgumentError: When shape of last dim is not ``k``.
        """
        # dtype check will be done by super() layer

        # Reshape inputs to [...,k]
        tf.debugging.assert_greater(tf.rank(inputs), 1)
        input_shape = inputs.shape
        new_shape = [-1, input_shape[-1]]
        u = tf.reshape(inputs, new_shape)

        # Consistency check (i.e., binary) of inputs will be done in super_class

        # CRC encode
        u_crc = self._enc_crc(u)

        # Encode bits (= channel allocation + Polar transform)
        c = super().call(u_crc)

        # Sub-block interleaving with 32 sub-blocks as in Sec. 5.4.1.1
        # Rate matching via circular buffer as defined in Sec. 5.4.1.2
        # Channel interleaving for uplink (i_bil=True)
        c_matched = tf.gather(c, self._ind_rate_matching, axis=1)

        # Restore original shape
        input_shape_list = input_shape.as_list()
        output_shape = input_shape_list[0:-1] + [self._n_target]
        output_shape[0] = -1 # To support dynamic shapes
        c_reshaped = tf.reshape(c_matched, output_shape)

        return c_reshaped
