#
# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Layers for decoding of linear codes."""

import tensorflow as tf
import numpy as np
import scipy as sp # for sparse H matrix computations
from tensorflow.keras.layers import Layer
from sionna.fec.utils import pcm2gm, int_mod_2, make_systematic
from sionna.utils import hard_decisions
import itertools

class OSDecoder(Layer):
    # pylint: disable=line-too-long
    r"""OSDecoder(enc_mat=None, t=0, is_pcm=False, encoder=None, dtype=tf.float32, **kwargs)

    Ordered statistics decoding (OSD) for binary, linear block codes.

    This layer implements the OSD algorithm as proposed in [Fossorier]_ and,
    thereby, approximates maximum likelihood decoding for a sufficiently large
    order :math:`t`. The algorithm works for arbitrary linear block codes, but
    has a high computational complexity for long codes.

    The algorithm consists of the following steps:

        1. Sort LLRs according to their reliability and apply the same column
        permutation to the generator matrix.

        2. Bring the permuted generator matrix into its systematic form
        (so-called *most-reliable basis*).

        3. Hard-decide and re-encode the :math:`k` most reliable bits and
        discard the remaining :math:`n-k` received positions.

        4. Generate all possible error patterns up to :math:`t` errors in the
        :math:`k` most reliable positions find the most likely codeword within
        these candidates.

    This implementation of the OSD algorithm uses the LLR-based distance metric
    from [Stimming_LLR_OSD]_ which simplifies the handling of higher-order
    modulation schemes.

    The class inherits from the Keras layer class and can be used as layer in a
    Keras model.

    Parameters
    ----------
    enc_mat : [k, n] or [n-k, n], ndarray
        Binary generator matrix of shape `[k, n]`. If ``is_pcm`` is
        True, ``enc_mat`` is interpreted as parity-check matrix of shape
        `[n-k, n]`.

    t : int
        Order of the OSD algorithm

    is_pcm: bool
        Defaults to False. If True, ``enc_mat`` is interpreted as parity-check
        matrix.

    encoder: Layer
        Keras layer that implements a FEC encoder.
        If not None, ``enc_mat`` will be ignored and the code as specified by he
        encoder is used to initialize OSD.

    dtype: tf.DType
        Defaults to `tf.float32`. Defines the datatype for the output dtype.

    Input
    -----
    llrs_ch: [...,n], tf.float32
        2+D tensor containing the channel logits/llr values.

    Output
    ------
        : [...,n], tf.float32
            2+D Tensor of same shape as ``llrs_ch`` containing
            binary hard-decisions of all codeword bits.

    Note
    ----
    OS decoding is of high complexity and is only feasible for small values of
    :math:`t` as :math:`{n \choose t}` patterns must be evaluated. The
    advantage of OSD is that it works for arbitrary linear block codes and
    provides an estimate of the expected ML performance for sufficiently large
    :math:`t`. However, for some code families, more efficient decoding
    algorithms with close to ML performance exist which can exploit certain
    code specific properties. Examples of such decoders are the
    :class:`~sionna.fec.conv.ViterbiDecoder` algorithm for  convolutional codes
    or the :class:`~sionna.fec.polar.decoding.PolarSCLDecoder` for Polar codes
    (for a sufficiently large list size).

    It is recommended to run the decoder in XLA mode as it
    significantly reduces the memory complexity.
    """

    def __init__(self,
                 enc_mat=None,
                 t=0,
                 is_pcm=False,
                 encoder=None,
                 dtype=tf.float32,
                 **kwargs):

        super().__init__(dtype=dtype, **kwargs)

        assert isinstance(is_pcm, bool), 'is_pcm must be bool.'

        self._llr_max = 100. # internal clipping value for llrs

        if enc_mat is not None:
            # check that gm is binary
            if isinstance(enc_mat, np.ndarray):
                assert np.array_equal(enc_mat, enc_mat.astype(bool)), \
                    'PC matrix must be binary.'
            elif isinstance(enc_mat, sp.sparse.csr_matrix):
                assert np.array_equal(enc_mat.data, enc_mat.data.astype(bool)),\
                    'PC matrix must be binary.'
            elif isinstance(enc_mat, sp.sparse.csc_matrix):
                assert np.array_equal(enc_mat.data, enc_mat.data.astype(bool)),\
                    'PC matrix must be binary.'
            else:
                raise TypeError("Unsupported dtype of pcm.")

        if dtype not in (tf.float16, tf.float32, tf.float64):
            raise ValueError(
                'dtype must be {tf.float16, tf.float32, tf.float64}.')

        assert (int(t)==t), "t must be int."
        self._t = int(t)

        if encoder is not None:
            # test that encoder is already initialized (relevant for conv codes)
            if encoder.k is None:
                raise AttributeError("It seems as if the encoder is not "\
                                     "initialized or has no attribute k.")
            # encode identity matrix to get k basis vectors of the code
            u = tf.expand_dims(tf.eye(encoder.k), axis=0)
            # encode and remove batch_dim
            self._gm = tf.cast(tf.squeeze(encoder(u), axis=0), self.dtype)
        else:
            assert (enc_mat is not None),\
                "enc_mat cannot be None if no encoder is provided."
            if is_pcm:
                gm = pcm2gm(enc_mat)
            else:
                # check if gm is of full rank (raise error otherwise)
                make_systematic(enc_mat)
                gm = enc_mat
            self._gm = tf.constant(gm, dtype=self.dtype)

        self._k = self._gm.shape[0]
        self._n = self._gm.shape[1]

        # init error patterns
        num_patterns = self._num_error_patterns(self._n, self._t)

        # storage/computational complexity scales with n
        num_symbols = num_patterns * self._n
        if num_symbols>1e9: # number still to be optimized
            print(f"Note: Required memory complexity is large for the "\
                  f"given code parameters and t={t}. Please consider small " \
                  f"batch-sizes to keep the inference complexity small and " \
                  f"activate XLA mode if possible." )
        if num_symbols>1e11: # number still to be optimized
            raise ResourceWarning("Due to its high complexity, OSD is not " \
                                 "feasible for the selected parameters. " \
                                 "Please consider using a smaller value for t.")

        # pre-compute all error patterns
        self._err_patterns = []
        for t_i in range(1, t+1):
            self._err_patterns.append(self._gen_error_patterns(self._k, t_i))

    #########################################
    # Public methods and properties
    #########################################

    @property
    def gm(self):
        """Generator matrix of the code"""
        return self._gm

    @property
    def n(self):
        """Codeword length"""
        return self._n

    @property
    def k(self):
        """Number of information bits per codeword"""
        return self._k

    @property
    def t(self):
        """Order of the OSD algorithm"""
        return self._t

    #########################
    # Utility methods
    #########################

    def _num_error_patterns(self, n, t):
        r"""Returns number of possible error patterns for t errors in n
        positions, i.e., calculates :math:`{n \choose t}`.

        Input
        -----
        n: int
            length of vector.

        t: int
            number of errors.
        """
        return sp.special.comb(n, t, exact=True, repetition=False)

    def _gen_error_patterns(self, n, t):
        r"""Returns list of all possible error patterns for t errors in n
        positions.

        Input
        -----
        n: int
            Length of vector.

        t: int
            Number of errors.

        Output
        ------
        : [num_patterns, t], tf.int32
            Tensor of size `num_patterns`=:math:`{n \choose t}` containing the
            t error indices.
        """

        err_patterns = []
        for p in itertools.combinations(range(n), t):
            err_patterns.append(p)

        return tf.constant(err_patterns)

    def _get_dist(self, llr, c_hat):
        """Distance function used for ML candidate selection.

        Currently, the distance metric from Polar decoding [Stimming_LLR_OSD]_
        literature is implemented.

        Input
        -----
        llr: [bs, n], tf.float32
            Received llrs of the channel observations.

        c_hat: [bs, num_cand, n], tf.float32
            Candidate codewords for which the distance to ``llr`` shall be
            evaluated.

        Output
        ------
        : [bs, num_cand], tf.float32
            Distance between ``llr`` and ``c_hat`` for each of the `num_cand`
            codeword candidates.

        Reference
        ---------
        [Stimming_LLR_OSD] Alexios Balatsoukas-Stimming, Mani Bastani Parizi,
        Andreas Burg, "LLR-Based Successive Cancellation List Decoding
        of Polar Codes." IEEE Trans Signal Processing, 2015.
        """

        # broadcast llr to all codeword candidates
        llr = tf.expand_dims(llr, axis=1)
        llr_sign = llr * (-2.*c_hat + 1.) # apply BPSK mapping

        d = tf.math.log(1. + tf.exp(llr_sign))
        return tf.reduce_mean(d, axis=2)

    def _find_min_dist(self, llr_ch, ep, gm_mrb, c):
        r"""Find error pattern which leads to minimum distance.

        Input
        -----
        llr_ch: [bs, n], tf.float32
            Channel observations as llrs after mrb sorting.

        ep: [num_patterns, t], tf.int32
            Tensor of size `num_patterns`=:math:`{n \choose t}` containing the
            t error indices.

        gm_mrb: [bs, k, n] tf.float32
            Most reliable basis for each batch example.

        c: [bs, n], tf.float32
            Most reliable base codeword.

        Output
        ------
        : [bs], tf.float32
            Distance of the most likely codeword to ``llr_ch`` after testing all
            ``ep`` error patterns.

        : [bs, n], tf.float32
            The most likely codeword after testing against all ``ep`` error
            patterns.
        """

        # generate all test candidates for each possible error pattern
        e = tf.gather(gm_mrb, ep, axis=1)
        e = tf.reduce_sum(e, axis=2)
        e += tf.expand_dims(c, axis=1) # add to mrb codeword
        c_cand = int_mod_2(e) # apply modulo-2 operation

        # calculate distance for each candidate
        # where c_cand has shape [bs, num_patterns, n]
        d = self._get_dist(llr_ch, c_cand)

        # find candidate index with smallest metric
        idx = tf.argmin(d, axis=1)
        c_hat = tf.gather(c_cand, idx, batch_dims=1)
        d = tf.gather(d, idx, batch_dims=1)
        return d, c_hat

    def _find_mrb(self, gm):
        """Find most reliable basis for all generator matrices in batch.

        Input
        -----
        gm: [bs, k, n] tf.float32
            Generator matrix for each batch example.

        Output
        ------
        gm_mrb: [bs, k, n] tf.float32
            Most reliable basis in systematic form for each batch example.

        idx_sort: [bs, n] tf.int64
            Indices of column permutations applied during mrb calculation.
        """

        bs = tf.shape(gm)[0]
        idx_pivot = tf.TensorArray(tf.int64, self._k, dynamic_size=False)

        #  bring gm in systematic form (by so-called pivot method)
        for idx_c in tf.range(self._k):

            # find pivot (i.e., first pos with index 1)
            idx_p = tf.argmax(gm[:, idx_c, :], axis=-1)

            # store pivot position
            idx_pivot = idx_pivot.write(idx_c, idx_p)

            # and eliminate the column in all other rows
            r = tf.gather(gm, idx_p, batch_dims=1, axis=-1)

            # ignore idx_c row itself by adding all-zero row
            rz = tf.zeros((bs, 1), dtype=self.dtype)
            r = tf.concat([r[:,:idx_c], rz , r[:,idx_c+1:]], axis=1)

            # mask is zero at all rows where pivot position of this row is zero
            mask = tf.tile(tf.expand_dims(r, axis=-1), (1, 1, self._n))
            gm_off = tf.expand_dims(gm[:,idx_c,:], axis=1)

            # update all row in parallel
            gm = int_mod_2(gm + mask * gm_off) # account for binary operations

        # pivot positions
        idx_pivot = tf.transpose(idx_pivot.stack())

        # find non-pivot positions (i.e., all indices that are not part of
        # idx_pivot)

        # solution 1: sets.difference() does not support XLA (unknown shapes)
        #idx_parity = tf.sets.difference(idx_range, idx_pivot)
        #idx_parity = tf.sparse.to_dense(idx_parity)
        #idx_pivot = tf.reshape(idx_pivot, (-1, self._n)) # ensure shape

        # solution 2: add large offset to pivot indices and sorting gives the
        # indices of interest
        idx_range = tf.tile(tf.expand_dims(
                                tf.range(self._n, dtype=tf.int64), axis=0),
                            (bs, 1))
        # large value to be added to irrelevant indices
        updates = self._n * tf.ones((bs, self._k), tf.int64)

        # generate indices for tf.scatter_nd_add
        s = tf.shape(idx_pivot, tf.int64)
        ii, _ = tf.meshgrid(tf.range(s[0]), tf.range(s[1]), indexing='ij')
        idx_updates = tf.stack([ii, idx_pivot], axis=-1)

        # add large value to pivot positions
        idx = tf.tensor_scatter_nd_add(idx_range, idx_updates, updates)

        # sort and slice first n-k indices (equals parity positions)
        idx_parity = tf.cast(tf.argsort(idx)[:,:self._n-self._k], tf.int64)

        idx_sort = tf.concat([idx_pivot, idx_parity], axis=1)

        # permute gm according to indices idx_sort
        gm = tf.gather(gm, idx_sort, batch_dims=1, axis=-1)

        return gm, idx_sort

    #########################
    # Keras layer functions
    #########################

    def build(self, input_shape):
        """Nothing to build, but check for valid shapes."""

        assert input_shape[-1]==self._n, "Invalid input shape."

    def call(self, inputs):
        r"""Applies ordered statistic decoding to inputs.

        Remark: the decoder is implemented with llr definition
        llr = p(x=1)/p(x=0).
        """

        # flatten batch-dim
        input_shape = tf.shape(inputs)
        llr_ch = tf.reshape(inputs, (-1, self._n))
        llr_ch = tf.cast(llr_ch, self.dtype)
        bs = tf.shape(llr_ch)[0]

        # clip inputs
        llr_ch = tf.clip_by_value(llr_ch, -self._llr_max, self._llr_max)

        # Step 1: sort LLRs
        idx_sort = tf.argsort(tf.abs(llr_ch), direction="DESCENDING")

        # permute gm per batch sample individually
        gm = tf.broadcast_to(tf.expand_dims(self._gm, axis=0),
                             (bs, self._k,self._n))
        gm_sort = tf.gather(gm, idx_sort, batch_dims=1, axis=-1)

        # Step 2: Find most reliable basis (MRB)
        gm_mrb, idx_mrb = self._find_mrb(gm_sort)

        # apply corresponding mrb permutations
        idx_sort = tf.gather(idx_sort, idx_mrb, batch_dims=1)
        llr_sort = tf.gather(llr_ch, idx_sort, batch_dims=1)

        # find inverse permutation for final output
        idx_sort_inv = tf.argsort(idx_sort)

        # hard-decide k most reliable positions and encode
        u_hd = hard_decisions(llr_sort[:,0:self._k])
        u_hd = tf.expand_dims(u_hd, axis=1)
        c = tf.squeeze(tf.matmul(u_hd, gm_mrb), axis=1)
        c = int_mod_2(c)

        # and search for most likely pattern
        # _get_dist expects a list of candidates, thus expand_dims to [bs, 1, n]
        d_best = self._get_dist(llr_sort, tf.expand_dims(c, axis=1))
        d_best = tf.squeeze(d_best, axis=1)
        c_hat_best = c

        # known in advance - can be unrolled
        for ep in self._err_patterns:
            # compute distance for all candidate codewords
            d, c_hat = self._find_min_dist(llr_sort, ep, gm_mrb, c)

            # select most likely candidate
            ind = tf.expand_dims(d<d_best, axis=1)
            c_hat_best = tf.where(ind, c_hat, c_hat_best)
            d_best = tf.where(d<d_best, d, d_best)

        # undo permutations for final codeword
        c_hat_best = tf.gather(c_hat_best, idx_sort_inv, axis=1, batch_dims=1)
        # input shape
        c_hat = tf.reshape(c_hat_best, input_shape)

        return c_hat
