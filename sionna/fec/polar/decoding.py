#
# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Layers for Polar decoding such as successive cancellation (SC), successive
cancellation list (SCL) and iterative belief propagation (BP) decoding."""

import tensorflow as tf
import numpy as np
from numpy.core.numerictypes import issubdtype
import warnings
from tensorflow.keras.layers import Layer
from sionna.fec.crc import CRCDecoder, CRCEncoder
from sionna.fec.polar.encoding import Polar5GEncoder
import numbers

class PolarSCDecoder(Layer):
    """PolarSCDecoder(frozen_pos, n, output_dtype=tf.float32, **kwargs)

    Successive cancellation (SC) decoder [Arikan_Polar]_ for Polar codes and
    Polar-like codes.

    The class inherits from the Keras layer class and can be used as layer in a
    Keras model.

    Parameters
    ----------
        frozen_pos: ndarray
            Array of `int` defining the ``n-k`` indices of the frozen positions.

        n: int
            Defining the codeword length.

       output_dtype: tf.DType
        Defaults to tf.float32. Defines the output datatype of the layer
        (internal precision remains tf.float32).

    Input
    -----
        inputs: [...,n], tf.float32
            2+D tensor containing the channel LLR values (as logits).

    Output
    ------
        : [...,k], tf.float32
            2+D tensor  containing hard-decided estimations of all ``k``
            information bits.

    Raises
    ------
        AssertionError
            If ``n`` is not `int`.

        AssertionError
            If ``n`` is not a power of 2.

        AssertionError
            If the number of elements in ``frozen_pos`` is greater than ``n``.

        AssertionError
            If ``frozen_pos`` does not consists of `int`.

        ValueError
            If ``output_dtype`` is not {tf.float16, tf.float32, tf.float64}.

    Note
    ----
        This layer implements the SC decoder as described in
        [Arikan_Polar]_. However, the implementation follows the `recursive
        tree` [Gross_Fast_SCL]_ terminology and combines nodes for increased
        throughputs without changing the outcome of the algorithm.

        As commonly done, we assume frozen bits are set to `0`. Please note
        that - although its practical relevance is only little - setting frozen
        bits to `1` may result in `affine` codes instead of linear code as the
        `all-zero` codeword is not necessarily part of the code any more.

    """

    def __init__(self, frozen_pos, n, output_dtype=tf.float32, **kwargs):

        if output_dtype not in (tf.float16, tf.float32, tf.float64):
            raise ValueError(
                'output_dtype must be {tf.float16, tf.float32, tf.float64}.')

        if output_dtype is not tf.float32:
            print('Note: decoder uses tf.float32 for internal calculations.')

        super().__init__(dtype=output_dtype, **kwargs)
        self._output_dtype = output_dtype

        # assert error if r>1 or k, n are negativ
        assert isinstance(n, numbers.Number), "n must be a number."
        n = int(n) # n can be float (e.g. as result of n=k*r)

        assert issubdtype(frozen_pos.dtype, int), "frozen_pos contains non int."
        assert len(frozen_pos)<=n, "Num. of elements in frozen_pos cannot " \
            "be greater than n."
        assert np.log2(n)==int(np.log2(n)), "n must be a power of 2."

        # store internal attributes
        self._n = n
        self._frozen_pos = frozen_pos
        self._k = self._n - len(self._frozen_pos)
        self._info_pos = np.setdiff1d(np.arange(self._n), self._frozen_pos)
        assert self._k==len(self._info_pos), "Internal error: invalid " \
                                              "info_pos generated."
        self._llr_max = 30. # internal max LLR value (uncritical for SC dec)
        # and create a frozen bit vector for simpler encoding
        self._frozen_ind = np.zeros(self._n)
        self._frozen_ind[self._frozen_pos] = 1

        # enable graph pruning
        self._use_fast_sc = False

    #########################################
    # Public methods and properties
    #########################################

    @property
    def n(self):
        """Codeword length."""
        return self._n

    @property
    def k(self):
        """Number of information bits."""
        return self._k

    @property
    def frozen_pos(self):
        """Frozen positions for Polar decoding."""
        return self._frozen_pos

    @property
    def info_pos(self):
        """Information bit positions for Polar encoding."""
        return self._info_pos

    @property
    def llr_max(self):
        """Maximum LLR value for internal calculations."""
        return self._llr_max

    @property
    def output_dtype(self):
        """Output dtype of decoder."""
        return self._output_dtype

    #########################
    # Utility methods
    #########################

    def _cn_op_tf(self, x, y):
        """Check-node update (boxplus) for LLR inputs.

        Operations are performed element-wise.

        See [Stimming_LLR]_ and [Hashemi_SSCL]_ for detailed equations.
        """
        x_in = tf.clip_by_value(x,
                                clip_value_min=-self._llr_max,
                                clip_value_max=self._llr_max)
        y_in = tf.clip_by_value(y,
                                clip_value_min=-self._llr_max,
                                clip_value_max=self._llr_max)

        # avoid division for numerical stability
        llr_out = tf.math.log(1 + tf.math.exp(x_in + y_in))
        llr_out -= tf.math.log(tf.math.exp(x_in) + tf.math.exp(y_in))

        return llr_out

    def _vn_op_tf(self, x, y, u_hat):
        """VN update for LLR inputs."""
        return tf.multiply((1-2*u_hat), x) + y

    def _polar_decode_sc_tf(self, llr_ch, frozen_ind):
        """Recursive SC decoding function.

        Recursively branch decoding tree and split into decoding of `upper`
        and `lower` path until reaching a leaf node.

        The function returns the u_hat decisions at stage `0` and the bit
        decisions of the intermediate stage `s` (i.e., the re-encoded version of
        `u_hat` until the current stage `s`).

        Note:
            This decoder parallelizes over the batch-dimension, i.e., the tree
            is processed for all samples in the batch in parallel. This yields a
            higher throughput, but does not improve the latency.
        """

        # calculate current codeword length
        n = len(frozen_ind)

        # branch if leaf is not reached yet
        if n>1:
            if self._use_fast_sc:
                if np.sum(frozen_ind)==n:
                    #print("rate-0 detected! Length: ", n)
                    u_hat = tf.zeros_like(llr_ch)
                    return u_hat, u_hat

            llr_ch1 = llr_ch[...,0:int(n/2)]
            llr_ch2 = llr_ch[...,int(n/2):]
            frozen_ind1 = frozen_ind[0:int(n/2)]
            frozen_ind2 = frozen_ind[int(n/2):]

            # upper path
            x_llr1_in = self._cn_op_tf(llr_ch1, llr_ch2)

            # and call the decoding function (with upper half)
            u_hat1, u_hat1_up = self._polar_decode_sc_tf(x_llr1_in, frozen_ind1)

            # lower path
            x_llr2_in = self._vn_op_tf(llr_ch1, llr_ch2, u_hat1_up)
            # and call the decoding function again (with lower half)
            u_hat2, u_hat2_up = self._polar_decode_sc_tf(x_llr2_in, frozen_ind2)

            # combine u_hat from both branches
            u_hat = tf.concat([u_hat1, u_hat2], -1)

            # calculate re-encoded version of u_hat at current stage
            # u_hat1_up = tf.math.mod(u_hat1_up + u_hat2_up, 2)
            # combine u_hat via bitwise_xor (more efficient than mod2)
            u_hat1_up_int = tf.cast(u_hat1_up, tf.int8)
            u_hat2_up_int = tf.cast(u_hat2_up, tf.int8)
            u_hat1_up_int = tf.bitwise.bitwise_xor(u_hat1_up_int,
                                                   u_hat2_up_int)
            u_hat1_up = tf.cast(u_hat1_up_int , tf.float32)
            u_hat_up = tf.concat([u_hat1_up, u_hat2_up], -1)

        else: # if leaf is reached perform basic decoding op (=decision)

            if frozen_ind==1: # position is frozen
                u_hat = tf.expand_dims(tf.zeros_like(llr_ch[:,0]), axis=-1)
                u_hat_up = u_hat
            else: # otherwise hard decide
                u_hat = 0.5 * (1. - tf.sign(llr_ch))
                #remove "exact 0 llrs" leading to u_hat=0.5
                u_hat = tf.where(tf.equal(u_hat, 0.5),
                                 tf.ones_like(u_hat),
                                 u_hat)
                u_hat_up = u_hat
        return u_hat, u_hat_up

    #########################
    # Keras layer functions
    #########################

    def build(self, input_shape):
        """Check if shape of input is invalid."""
        assert (input_shape[-1]==self._n), "Invalid input shape."
        assert (len(input_shape)>=2), 'Inputs must have at least 2 dimensions.'

    def call(self, inputs):
        """Successive cancellation (SC) decoding function.

        Performs successive cancellation decoding and returns the estimated
        information bits.

        Args:
            inputs (tf.float32): Tensor of shape `[...,n]` containing the
                channel LLR values (as logits).

        Returns:
            `tf.float32`: Tensor of shape `[...,k]` containing
            hard-decided estimations of all ``k`` information bits.

        Raises:
            ValueError: If ``inputs`` is not of shape `[..., n]`
                or `dtype` is not `tf.float32`.

            InvalidArgumentError: When rank(``inputs``)<2.

        Note:
            This function recursively unrolls the SC decoding tree, thus,
            for larger values of ``n`` building the decoding graph can become
            time consuming.
        """

        tf.debugging.assert_type(inputs, self.dtype, 'Invalid input dtype.')
        # internal calculations still in tf.float32
        inputs = tf.cast(inputs, tf.float32)

        # last dim must be of length n
        tf.debugging.assert_equal(tf.shape(inputs)[-1],
                                  self._n,
                                  "Last input dimension must be of length n.")

        # Reshape inputs to [-1, n]
        tf.debugging.assert_greater(tf.rank(inputs), 1)
        input_shape = inputs.shape
        new_shape = [-1, self._n]
        llr_ch = tf.reshape(inputs, new_shape)

        llr_ch = -1. * llr_ch # logits are converted into "true" llrs

        # and decode
        u_hat_n, _ = self._polar_decode_sc_tf(llr_ch, self._frozen_ind)

        # and recover the k information bit positions
        u_hat = tf.gather(u_hat_n, self._info_pos, axis=1)

        # and reconstruct input shape
        output_shape = input_shape.as_list()
        output_shape[-1] = self.k
        output_shape[0] = -1 # first dim can be dynamic (None)
        u_hat_reshape = tf.reshape(u_hat, output_shape)
        return tf.cast(u_hat_reshape, self._output_dtype)

class PolarSCLDecoder(Layer):
    # pylint: disable=line-too-long
    """PolarSCLDecoder(frozen_pos, n, list_size=8, crc_degree=None, use_hybrid_sc=False, use_fast_scl=True, cpu_only=False, use_scatter=False, output_dtype=tf.float32, **kwargs)

    Successive cancellation list (SCL) decoder [Tal_SCL]_ for Polar codes
    and Polar-like codes.

    The class inherits from the Keras layer class and can be used as layer in a
    Keras model.

    Parameters
    ----------
        frozen_pos: ndarray
            Array of `int` defining the ``n-k`` indices of the frozen positions.

        n: int
            Defining the codeword length.

        list_size: int
            Defaults to 8. Defines the list size of the decoder.

        crc_degree: str
            Defining the CRC polynomial to be used. Can be any value from
            `{CRC24A, CRC24B, CRC24C, CRC16, CRC11, CRC6}`.

        use_hybrid_sc: bool
            Defaults to False. If True, SC decoding is applied and only the
            codewords with invalid CRC are decoded with SCL. This option
            requires an outer CRC specified via ``crc_degree``.
            Remark: hybrid_sc does not support XLA optimization, i.e.,
            `@tf.function(jit_compile=True)`.

        use_fast_scl: bool
            Defaults to True. If True, Tree pruning is used to
            reduce the decoding complexity. The output is equivalent to the
            non-pruned version (besides numerical differences).

        cpu_only: bool
            Defaults to False. If True, `tf.py_function` embedding
            is used and the decoder runs on the CPU. This option is usually
            slower, but also more memory efficient and, in particular,
            recommended for larger blocklengths. Remark: cpu_only does not
            support XLA optimization `@tf.function(jit_compile=True)`.

        use_scatter: bool
            Defaults to False. If True, `tf.tensor_scatter_update` is used for
            tensor updates. This option is usually slower, but more memory
            efficient.

       output_dtype: tf.DType
            Defaults to tf.float32. Defines the output datatype of the layer
            (internal precision remains tf.float32).

    Input
    -----
        inputs: [...,n], tf.float32
            2+D tensor containing the channel LLR values (as logits).

    Output
    ------
        : [...,k], tf.float32
            2+D tensor containing hard-decided estimations of all `k`
            information bits.

    Raises:
        AssertionError
            If ``n`` is not `int`.

        AssertionError
            If ``n`` is not a power of 2.

        AssertionError
            If the number of elements in ``frozen_pos`` is greater than ``n``.

        AssertionError
            If ``frozen_pos`` does not consists of `int`.

        AssertionError
            If ``list_size`` is not `int`.

        AssertionError
            If ``cpu_only`` is not `bool`.

        AssertionError
            If ``use_scatter`` is not `bool`.

        AssertionError
            If ``use_fast_scl`` is not `bool`.

        AssertionError
            If ``use_hybrid_sc`` is not `bool`.

        AssertionError
            If ``list_size`` is not a power of 2.

        ValueError
            If ``output_dtype`` is not {tf.float16, tf.float32, tf.
            float64}.

        ValueError
            If ``inputs`` is not of shape `[..., n]` or `dtype` is not
            correct.

        InvalidArgumentError
            When rank(``inputs``)<2.

    Note
    ----
        This layer implements the successive cancellation list (SCL) decoder
        as described in [Tal_SCL]_ but uses LLR-based message updates
        [Stimming_LLR]_. The implementation follows the notation from
        [Gross_Fast_SCL]_, [Hashemi_SSCL]_. If option `use_fast_scl` is active
        tree pruning is used and tree nodes are combined if possible (see
        [Hashemi_SSCL]_ for details).

        Implementing SCL decoding as TensorFlow graph is a difficult task that
        requires several design tradeoffs to match the TF constraints while
        maintaining a reasonable throughput. Thus, the decoder minimizes
        the `control flow` as much as possible, leading to a strong memory
        occupation (e.g., due to full path duplication after each decision).
        For longer code lengths, the complexity of the decoding graph becomes
        large and we recommend to use the `CPU_only` option that uses an
        embedded Numpy decoder. Further, this function recursively unrolls the
        SCL decoding tree, thus, for larger values of ``n`` building the
        decoding graph can become time consuming. Please consider the
        ``cpu_only`` option if building the graph takes to long.

        A hybrid SC/SCL decoder as proposed in [Cammerer_Hybrid_SCL]_ (using SC
        instead of BP) can be activated with option ``use_hybrid_sc`` iff an
        outer CRC is available. Please note that the results are not exactly
        SCL performance caused by the false positive rate of the CRC.

        As commonly done, we assume frozen bits are set to `0`. Please note
        that - although its practical relevance is only little - setting frozen
        bits to `1` may result in `affine` codes instead of linear code as the
        `all-zero` codeword is not necessarily part of the code any more.
    """

    def __init__(self,
                 frozen_pos,
                 n,
                 list_size=8,
                 crc_degree=None,
                 use_hybrid_sc=False,
                 use_fast_scl=True,
                 cpu_only=False,
                 use_scatter=False,
                 output_dtype=tf.float32,
                 **kwargs):

        if output_dtype not in (tf.float16, tf.float32, tf.float64):
            raise ValueError(
                'output_dtype must be {tf.float16, tf.float32, tf.float64}.')

        if output_dtype is not tf.float32:
            print('Note: decoder uses tf.float32 for internal calculations.')

        super().__init__(dtype=output_dtype, **kwargs)
        self._output_dtype = output_dtype

        # assert error if r>1 or k, n are negative
        assert isinstance(n, numbers.Number), "n must be a number."
        n = int(n) # n can be float (e.g. as result of n=k*r)
        assert isinstance(list_size, int), "list_size must be integer."
        assert isinstance(cpu_only, bool), "cpu_only must be bool."
        assert isinstance(use_scatter, bool), "use_scatter must be bool."
        assert isinstance(use_fast_scl, bool), "use_fast_scl must be bool."
        assert isinstance(use_hybrid_sc, bool), "use_hybrid_sc must be bool."

        assert issubdtype(frozen_pos.dtype, int), "frozen_pos contains non int."
        assert len(frozen_pos)<=n, "Num. of elements in frozen_pos cannot " \
            "be greater than n."
        assert np.log2(n)==int(np.log2(n)), "n must be a power of 2."
        assert np.log2(list_size)==int(np.log2(list_size)), \
                                    "list_size must be a power of 2."

        # CPU mode is recommended for larger values of n
        if n>128 and cpu_only is False and use_hybrid_sc is False:
            warnings.warn("Required ressource allocation is large " \
            "for the selected blocklength. Consider option `cpu_only=True`.")

        # CPU mode is recommended for larger values of L
        if list_size>32 and cpu_only is False and use_hybrid_sc is False:
            warnings.warn("Ressource allocation is high for the " \
            "selected list_size. Consider option `cpu_only=True`.")

        # internal decoder parameters
        self._use_fast_scl = use_fast_scl # optimize rate-0 and rep nodes
        self._use_scatter = use_scatter # slower but more memory friendly
        self._cpu_only = cpu_only # run numpy decoder
        self._use_hybrid_sc = use_hybrid_sc

        # store internal attributes
        self._n = n
        self._frozen_pos = frozen_pos
        self._k = self._n - len(self._frozen_pos)
        self._list_size = list_size
        self._info_pos = np.setdiff1d(np.arange(self._n), self._frozen_pos)
        self._llr_max = 30. # internal max LLR value (not very critical for SC)
        assert self._k==len(self._info_pos), "Internal error: invalid " \
                                              "info_pos generated."
        # create a frozen bit vector
        self._frozen_ind = np.zeros(self._n)
        self._frozen_ind[self._frozen_pos] = 1
        self._cw_ind = np.arange(self._n)
        self._n_stages = int(np.log2(self._n)) # number of decoding stages

        # init CRC check (if needed)
        if crc_degree is not None:
            self._use_crc = True
            self._crc_decoder = CRCDecoder(CRCEncoder(crc_degree))
            self._k_crc = self._crc_decoder.encoder.crc_length
        else:
            self._use_crc = False
            self._k_crc = 0
        assert self._k>=self._k_crc, "Value of k is too small for \
            given CRC_degree."

        # use SC decoder first and use numpy-based SCL as "afterburner"
        if self._use_hybrid_sc:
            self._decoder_sc = PolarSCDecoder(frozen_pos, n)
            # Note: CRC required to detect SC success
            if not self._use_crc:
                raise ValueError("Hybrid SC requires outer CRC.")

    #########################################
    # Public methods and properties
    #########################################

    @property
    def n(self):
        """Codeword length."""
        return self._n

    @property
    def k(self):
        """Number of information bits."""
        return self._k

    @property
    def k_crc(self):
        """Number of CRC bits."""
        return self._k_crc

    @property
    def frozen_pos(self):
        """Frozen positions for Polar decoding."""
        return self._frozen_pos

    @property
    def info_pos(self):
        """Information bit positions for Polar encoding."""
        return self._info_pos

    @property
    def llr_max(self):
        """Maximum LLR value for internal calculations."""
        return self._llr_max

    @property
    def list_size(self):
        """List size for SCL decoding."""
        return self._list_size

    @property
    def output_dtype(self):
        """Output dtype of decoder."""
        return self._output_dtype

    #####################################
    # Helper functions for the TF decoder
    #####################################

    def _update_rate0_code(self, msg_pm, msg_uhat, msg_llr, cw_ind):
        """Update rate-0 sub-code (i.e., all frozen) at pos ``cw_ind``.

        See eq. (26) in [Hashemi_SSCL]_.

        Remark: bits are not explicitly set to `0` as ``msg_uhat`` is initalized
        with `0` already.
        """
        n = len(cw_ind)
        stage_ind = int(np.log2(n))

        llr = tf.gather(msg_llr[:, :, stage_ind, :], cw_ind, axis=2)
        llr_in = tf.clip_by_value(llr,
                                  clip_value_min=-self._llr_max,
                                  clip_value_max=self._llr_max)

        # update path metric for complete sub-block of length n
        pm_val = tf.math.log(1 + tf.math.exp(-1.*llr_in))
        msg_pm += tf.reduce_sum(pm_val, axis=-1)

        return msg_pm, msg_uhat, msg_llr

    def _update_rep_code(self, msg_pm, msg_uhat, msg_llr, cw_ind):
        """Update rep. code (i.e., only rightmost bit is non-frozen)
        sub-code at position ``ind_u``.

        See Eq. (31) in [Hashemi_SSCL]_.

        Remark: bits are not explicitly set to `0` as ``msg_uhat`` is
        initalized with `0` already.
        """
        n = len(cw_ind)
        stage_ind = int(np.log2(n))

        # update PM
        llr = tf.gather(msg_llr[:, :, stage_ind, :], cw_ind, axis=2)
        llr_in = tf.clip_by_value(llr,
                                  clip_value_min=-self._llr_max,
                                  clip_value_max=self._llr_max)

        # upper branch has negative llr values (bit is 1)
        llr_low =  llr_in[:, :self._list_size, :]
        llr_up = - llr_in[:, self._list_size:, :]
        llr_pm = tf.concat([llr_low, llr_up], 1)
        pm_val = tf.math.log(1 + tf.math.exp(-1.*llr_pm))
        msg_pm += tf.reduce_sum(pm_val, axis=-1)

        msg_uhat1 = msg_uhat[:, :self._list_size, :, :]
        msg_uhat21 = tf.expand_dims(
                        msg_uhat[:, self._list_size:, stage_ind, :cw_ind[0]],
                        axis=2)

        msg_uhat22= tf.expand_dims(
                        msg_uhat[:, self._list_size:, stage_ind, cw_ind[-1]+1:],
                        axis=2)
        # ones to insert
        msg_ones = tf.ones([tf.shape(msg_uhat)[0], self._list_size, 1, n],
                            tf.float32)

        msg_uhat23 = tf.concat([msg_uhat21, msg_ones, msg_uhat22], 3)
        msg_uhat24_1 = msg_uhat[:, self._list_size:, :stage_ind, :]
        msg_uhat24_2 = msg_uhat[:, self._list_size:, stage_ind+1:, :]

        msg_uhat2 = tf.concat([msg_uhat24_1, msg_uhat23, msg_uhat24_2], 2)
        msg_uhat = tf.concat([msg_uhat1, msg_uhat2], 1)

        # branch last bit and update pm at pos cw_ind[-1]
        msg_uhat = self._update_single_bit([cw_ind[-1]], msg_uhat)
        msg_pm, msg_uhat, msg_llr = self._sort_decoders(msg_pm,
                                                        msg_uhat,
                                                        msg_llr)
        msg_uhat, msg_llr, msg_pm = self._duplicate_paths(msg_uhat,
                                                          msg_llr,
                                                          msg_pm)
        return msg_pm, msg_uhat, msg_llr

    def _update_single_bit(self, ind_u, msg_uhat):
        """Update single bit at position ``ind_u`` for all decoders.

        Remark: bits are not explicitly set to `0` as ``msg_uhat`` is
        initalized with `0` already.

        Remark: Two versions are implemented (throughput vs. graph complexity):
        1.) use tensor_scatter_nd_update
        2.) explicitly split graph and concatenate again
        """
        # position is non-frozen
        if self._frozen_ind[ind_u[0]]==0:

            # msg_uhat[:, ind_up, 0, ind_u] = 1
            if self._use_scatter:
                ind_dec = np.arange(self._list_size, 2*self._list_size, 1)
                ind_stage = np.array([0])

                # transpose such that batch dim can be broadcasted
                msg_uhat_t = tf.transpose(msg_uhat, [1, 3, 2, 0])

                # generate index grid
                ind_u = tf.cast(ind_u, tf.int64)
                grid = tf.meshgrid(ind_dec, ind_u, ind_stage)
                ind = tf.reshape(tf.stack(grid, axis=-1), [-1, 3])

                updates = tf.ones([ind.shape[0], tf.shape(msg_uhat)[0]])
                msg_uhat_s = tf.tensor_scatter_nd_update(msg_uhat_t,
                                                         ind,
                                                         updates)
                # and restore original order
                msg_uhat = tf.transpose(msg_uhat_s, [3, 0, 2, 1])
            else:
                # alternative solution with split/concantenation of graph
                msg_uhat1 = msg_uhat[:, :self._list_size, :, :]
                msg_uhat21 = tf.expand_dims(
                                msg_uhat[:, self._list_size:, 0, :ind_u[0]],
                                axis=2)

                msg_uhat22= tf.expand_dims(
                                msg_uhat[:, self._list_size:, 0, ind_u[0]+1:],
                                axis=2)
                # ones to insert
                msg_ones = tf.ones_like(tf.reshape(
                                msg_uhat[:, self._list_size:, 0, ind_u[0]],
                                [-1, self._list_size, 1, 1]))

                msg_uhat23 = tf.concat([msg_uhat21, msg_ones, msg_uhat22], 3)
                msg_uhat24 = msg_uhat[:, self._list_size:, 1:, :]

                msg_uhat2 = tf.concat([msg_uhat23, msg_uhat24], 2)
                msg_uhat = tf.concat([msg_uhat1, msg_uhat2], 1)

        return msg_uhat

    def _update_pm(self, ind_u, msg_uhat, msg_llr, msg_pm):
        """Update path metric of all decoders after updating bit_pos ``ind_u``.

        We implement (10) from [Stimming_LLR]_.
        """
        u_hat = msg_uhat[:, :, 0, ind_u[0]]
        llr = msg_llr[:, :, 0, ind_u[0]]

        llr_in = tf.clip_by_value(llr,
                                  clip_value_min=-self._llr_max,
                                  clip_value_max=self._llr_max)

        pm_inner = tf.math.exp(-tf.multiply((1 - 2*u_hat), llr_in))
        msg_pm += tf.math.log(1 + pm_inner)
        return msg_pm

    def _sort_decoders(self, msg_pm, msg_uhat, msg_llr):
        """Sort decoders according to their path metric."""

        ind = tf.argsort(msg_pm, axis=-1)

        msg_pm = tf.gather(msg_pm, ind, batch_dims=1, axis=None)
        msg_uhat = tf.gather(msg_uhat, ind, batch_dims=1, axis=None)
        msg_llr = tf.gather(msg_llr, ind, batch_dims=1, axis=None)

        return msg_pm, msg_uhat, msg_llr

    def _cn_op(self, x, y):
        """Check-node update (boxplus) for LLR inputs.

        Operations are performed element-wise.

        See [Stimming_LLR]_ and [Hashemi_SSCL]_ for detailed equations.
        """
        x_in = tf.clip_by_value(x,
                                clip_value_min=-self._llr_max,
                                clip_value_max=self._llr_max)
        y_in = tf.clip_by_value(y,
                                clip_value_min=-self._llr_max,
                                clip_value_max=self._llr_max)

        # avoid division for numerical stability
        llr_out = tf.math.log(1 + tf.math.exp(x_in + y_in))
        llr_out -= tf.math.log(tf.math.exp(x_in) + tf.math.exp(y_in))

        return llr_out

    def _vn_op(self, x, y, u_hat):
        """Variable node update for LLR inputs.

        Operations are performed element-wise.

        See [Stimming_LLR]_ and [Hashemi_SSCL]_ for detailed equations.
        """
        return tf.multiply((1 - 2*u_hat), x) + y

    def _duplicate_paths(self, msg_uhat, msg_llr, msg_pm):
        """Duplicate paths by copying the upper branch into the lower one.
        """
        msg_uhat = tf.tile(msg_uhat[:, :self._list_size, :, :], [1, 2, 1, 1])
        msg_llr = tf.tile(msg_llr[:, :self._list_size, :, :], [1, 2, 1, 1])
        msg_pm = tf.tile(msg_pm[:, :self._list_size], [1, 2])

        return msg_uhat, msg_llr, msg_pm

    def _update_left_branch(self, msg_llr, stage_ind, cw_ind_left,cw_ind_right):
        """Update messages of left branch.

        Remark: Two versions are implemented (throughput vs. graph complexity):
        1.) use tensor_scatter_nd_update
        2.) explicitly split graph and concatenate again
        """

        llr_left_in = tf.gather(msg_llr[:, :, stage_ind, :],
                                cw_ind_left,
                                axis=2)
        llr_right_in = tf.gather(msg_llr[:, :, stage_ind, :],
                                 cw_ind_right,
                                 axis=2)

        llr_left_out = self._cn_op(llr_left_in, llr_right_in)

        if self._use_scatter:
            # self.msg_llr[:, :, stage_ind-1, cw_ind_left] = llr_left_out

            # transpose such that batch-dim can be broadcasted
            msg_llr_t = tf.transpose(msg_llr, [2, 3, 1, 0])
            llr_left_out_s = tf.transpose(llr_left_out, [2, 1, 0])

            # generate index grid
            stage_ind = tf.cast(stage_ind, tf.int64)
            cw_ind_left = tf.cast(cw_ind_left, tf.int64)
            grid = tf.meshgrid(stage_ind-1, cw_ind_left)
            ind = tf.reshape(tf.stack(grid, axis=-1), [-1, 2])

            # update values
            msg_llr_s = tf.tensor_scatter_nd_update(msg_llr_t,
                                                    ind,
                                                    llr_left_out_s)

            # and restore original order
            msg_llr = tf.transpose(msg_llr_s, [3, 2, 0, 1])
        else:
            # alternative solution with split/concatenation of graph
            # llr_left = msg_llr[:, :, stage_ind, cw_ind_left]
            llr_left0 = tf.gather(msg_llr[:, :, stage_ind-1, :],
                                  np.arange(0, cw_ind_left[0]),
                                  axis=2)

            llr_right = tf.gather(msg_llr[:, :, stage_ind-1, :],
                                  cw_ind_right,
                                  axis=2)
            llr_right1 = tf.gather(msg_llr[:, :, stage_ind-1, :],
                                   np.arange(cw_ind_right[-1] +1, self._n),
                                   axis=2)

            llr_s = tf.concat([llr_left0,
                               llr_left_out,
                               llr_right,
                               llr_right1], 2)

            llr_s = tf.expand_dims(llr_s, axis=2)

            msg_llr1 = msg_llr[:, :, 0:stage_ind-1, :]
            msg_llr2 = msg_llr[:, :, stage_ind:, :]
            msg_llr = tf.concat([msg_llr1, llr_s, msg_llr2], 2)

        return msg_llr

    def _update_right_branch(self, msg_llr, msg_uhat, stage_ind, cw_ind_left,
                             cw_ind_right):
        """Update messages for right branch.

        Remark: Two versions are implemented (throughput vs. graph complexity):
        1.) use tensor_scatter_nd_update
        2.) explicitly split graph and concatenate again
        """
        u_hat_left_up = tf.gather(msg_uhat[:, :, stage_ind-1, :],
                                  cw_ind_left,
                                  axis=2)

        llr_left_in = tf.gather(msg_llr[:, :, stage_ind, :],
                                cw_ind_left,
                                axis=2)

        llr_right = tf.gather(msg_llr[:, :, stage_ind, :],
                              cw_ind_right,
                              axis=2)

        llr_right_out = self._vn_op(llr_left_in, llr_right, u_hat_left_up)

        if self._use_scatter:
            # transpose such that batch dim can be broadcasted
            msg_llr_t = tf.transpose(msg_llr, [2, 3, 1, 0])
            llr_right_out_s = tf.transpose(llr_right_out, [2, 1, 0])

            # generate index grid
            stage_ind = tf.cast(stage_ind, tf.int64)
            cw_ind_left = tf.cast(cw_ind_right, tf.int64)
            grid = tf.meshgrid(stage_ind-1, cw_ind_right)
            ind = tf.reshape(tf.stack(grid, axis=-1), [-1, 2])

            msg_llr_s = tf.tensor_scatter_nd_update(msg_llr_t,
                                                    ind,
                                                    llr_right_out_s)

            # and restore original order
            msg_llr = tf.transpose(msg_llr_s, [3, 2, 0, 1])
        else:
            # alternative solution with split/concatenation of graph
            # llr_left = msg_llr[:, :, stage_ind, cw_ind_left]
            llr_left0 = tf.gather(msg_llr[:, :, stage_ind-1, :],
                                  np.arange(0, cw_ind_left[0]),
                                  axis=2)
            llr_left = tf.gather(msg_llr[:, :, stage_ind-1, :],
                                 cw_ind_left,
                                 axis=2)
            llr_right1 = tf.gather(msg_llr[:, :, stage_ind-1, :],
                                   np.arange(cw_ind_right[-1]+1, self._n),
                                   axis=2)

            llr_s = tf.concat([llr_left0, llr_left, llr_right_out,llr_right1],2)
            llr_s = tf.expand_dims(llr_s, axis=2)

            msg_llr1 = msg_llr[:, :, 0:stage_ind-1, :]
            msg_llr2 = msg_llr[:, :, stage_ind:, :]

            msg_llr = tf.concat([msg_llr1, llr_s, msg_llr2], 2)

        return msg_llr

    def _update_branch_u(self, msg_uhat, stage_ind, cw_ind_left, cw_ind_right):
        """Update ``u_hat`` messages after executing both branches.

        Remark: Two versions are implemented (throughput vs. graph complexity):
        1.) use tensor_scatter_nd_update
        2.) explicitly split graph and concatenate again
        """
        u_hat_left_up = tf.gather(msg_uhat[:, :, stage_ind-1, :],
                                  cw_ind_left,
                                  axis=2)

        u_hat_right_up = tf.gather(msg_uhat[:, :, stage_ind-1, :],
                                   cw_ind_right,
                                   axis=2)

        # combine u_hat via bitwise_xor (more efficient than mod2)
        u_hat_left_up_int = tf.cast(u_hat_left_up, tf.int32)
        u_hat_right_up_int = tf.cast(u_hat_right_up, tf.int32)
        u_hat_left = tf.bitwise.bitwise_xor(u_hat_left_up_int,
                                            u_hat_right_up_int)
        u_hat_left = tf.cast(u_hat_left, tf.float32)

        if self._use_scatter:
            cw_ind = np.concatenate([cw_ind_left, cw_ind_right])

            u_hat = tf.concat([u_hat_left, u_hat_right_up], -1)

            # self.msg_llr[:, stage_ind-1, cw_ind_left] = llr_left_out

            # transpose such that batch dim can be broadcasted
            msg_uhat_t = tf.transpose(msg_uhat, [2, 3, 1, 0])
            u_hat_s = tf.transpose(u_hat, [2, 1, 0])

            # generate index grid
            stage_ind = tf.cast(stage_ind, tf.int64)
            cw_ind = tf.cast(cw_ind, tf.int64)
            grid = tf.meshgrid(stage_ind, cw_ind)
            ind = tf.reshape(tf.stack(grid, axis=-1), [-1, 2])

            msg_uhat_s = tf.tensor_scatter_nd_update(msg_uhat_t,
                                                     ind,
                                                     u_hat_s)

            # and restore original order
            msg_uhat = tf.transpose(msg_uhat_s, [3, 2, 0, 1])
        else:
            # alternative solution with split/concantenation of graph
            u_hat_left_0 = tf.gather(msg_uhat[:, :, stage_ind, :],
                                     np.arange(0, cw_ind_left[0]),
                                     axis=2)
            u_hat_right_1 = tf.gather(msg_uhat[:, :, stage_ind, :],
                                      np.arange(cw_ind_right[-1]+1, self._n),
                                      axis=2)

            u_hat = tf.concat([u_hat_left_0,
                               u_hat_left,
                               u_hat_right_up,
                               u_hat_right_1], 2)

            # provide u_hat for next higher stage
            msg_uhat1 = msg_uhat[:, :, 0:stage_ind, :]
            msg_uhat2 = msg_uhat[:, :, stage_ind+1:, :]
            u_hat = tf.expand_dims(u_hat, axis=2)

            msg_uhat = tf.concat([msg_uhat1, u_hat, msg_uhat2], 2)

        return msg_uhat

    def _polar_decode_scl(self, cw_ind, msg_uhat, msg_llr, msg_pm):
        """Recursive decoding function for SCL decoding.

        We follow the terminology from [Hashemi_SSCL]_ and [Stimming_LLR]_
        and branch the messages into a `left` and `right` update paths until
        reaching a leaf node.

        Tree pruning as proposed in [Hashemi_SSCL]_ is used to minimize the
        tree depth while maintaining the same output.
        """
        # current sub-code length and stage index (= tree depth)
        n = len(cw_ind)
        stage_ind = int(np.log2(n))

        # recursively branch through decoding tree
        if n>1:
            # prune tree if rate-0 subcode is detected
            if self._use_fast_scl:
                if np.sum(self._frozen_ind[cw_ind])==n:
                    msg_pm, msg_uhat, msg_llr = self._update_rate0_code(msg_pm,
                                                                       msg_uhat,
                                                                       msg_llr,
                                                                       cw_ind)
                    return msg_uhat, msg_llr, msg_pm

                if (self._frozen_ind[cw_ind[-1]]==0 and
                    np.sum(self._frozen_ind[cw_ind[:-1]])==n-1):
                    msg_pm, msg_uhat, msg_llr, = self._update_rep_code(msg_pm,
                                                                       msg_uhat,
                                                                       msg_llr,
                                                                       cw_ind)
                    return msg_uhat, msg_llr, msg_pm

            # split index into left and right part
            cw_ind_left = cw_ind[0:int(n/2)]
            cw_ind_right = cw_ind[int(n/2):]

            # ----- left branch -----
            msg_llr = self. _update_left_branch(msg_llr,
                                                stage_ind,
                                                cw_ind_left,
                                                cw_ind_right)

            # call sub-graph decoder of left branch
            msg_uhat, msg_llr, msg_pm = self._polar_decode_scl(cw_ind_left,
                                                               msg_uhat,
                                                               msg_llr,
                                                               msg_pm)

            # ----- right branch -----
            msg_llr = self._update_right_branch(msg_llr,
                                                msg_uhat,
                                                stage_ind,
                                                cw_ind_left,
                                                cw_ind_right)

            # call sub-graph decoder of right branch
            msg_uhat, msg_llr, msg_pm = self._polar_decode_scl(cw_ind_right,
                                                               msg_uhat,
                                                               msg_llr,
                                                               msg_pm)
            # update uhat at current stage
            msg_uhat = self._update_branch_u(msg_uhat,
                                             stage_ind,
                                             cw_ind_left,
                                             cw_ind_right)

        # if leaf is reached perform basic decoding op (=decision)
        else:
            # update bit value at current position
            msg_uhat = self._update_single_bit(cw_ind, msg_uhat)

            # update PM
            msg_pm = self._update_pm(cw_ind, msg_uhat, msg_llr, msg_pm)

            if self._frozen_ind[cw_ind]==0: # position is non-frozen
                # sort list
                msg_pm, msg_uhat, msg_llr = self._sort_decoders(msg_pm,
                                                                msg_uhat,
                                                                msg_llr)

                # duplicate l best decoders to pos l:2*l (kill other decoders)
                msg_uhat, msg_llr, msg_pm = self._duplicate_paths(msg_uhat,
                                                                  msg_llr,
                                                                  msg_pm)

        return msg_uhat, msg_llr, msg_pm

    def _decode_tf(self, llr_ch):
        """Main decoding function in TF.

        Initializes memory and calls recursive decoding function.
        """

        batch_size = tf.shape(llr_ch)[0]

        # allocate memory for all 2*list_size decoders
        msg_uhat = tf.zeros([batch_size,
                             2*self._list_size,
                             self._n_stages+1,
                             self._n])
        msg_llr = tf.zeros([batch_size,
                            2*self._list_size,
                            self._n_stages,
                            self._n])
        # init all 2*l decoders with same llr_ch
        llr_ch = tf.reshape(llr_ch, [-1, 1, 1, self._n])
        llr_ch = tf.tile(llr_ch,[1, 2*self._list_size, 1, 1])

        # init last stage with llr_ch
        msg_llr = tf.concat([msg_llr, llr_ch], 2)

        # init all remaining L-1 decoders with high penalty
        pm0 = tf.zeros([batch_size, 1])
        pm1 = self._llr_max * tf.ones([batch_size, self._list_size-1])
        msg_pm = tf.concat([pm0, pm1, pm0, pm1], 1)

        # and call recursive graph function
        msg_uhat, msg_llr, msg_pm = self._polar_decode_scl(self._cw_ind,
                                                           msg_uhat,
                                                           msg_llr,
                                                           msg_pm)

        # and sort output
        msg_pm, msg_uhat, msg_llr = self._sort_decoders(msg_pm,
                                                        msg_uhat,
                                                        msg_llr)
        return [msg_uhat, msg_pm]

    ####################################
    # Helper functions for Numpy decoder
    ####################################

    def _update_rate0_code_np(self, cw_ind):
        """Update rate-0 (i.e., all frozen) sub-code at pos ``cw_ind`` in Numpy.

        See Eq. (26) in [Hashemi_SSCL]_.
        """
        n = len(cw_ind)
        stage_ind = int(np.log2(n))

        # update PM for each batch sample
        ind = np.expand_dims(self._dec_pointer, axis=-1)
        llr_in = np.take_along_axis(self.msg_llr[:, :, stage_ind, cw_ind],
                                    ind,
                                    axis=1)

        llr_clip = np.maximum(np.minimum(llr_in, self._llr_max), -self._llr_max)
        pm_val = np.log(1 + np.exp(-llr_clip))
        self.msg_pm += np.sum(pm_val, axis=-1)

    def _update_rep_code_np(self, cw_ind):
        """Update rep. code (i.e., only rightmost bit is non-frozen)
        sub-code at position ``ind_u`` in Numpy.

        See Eq. (31) in [Hashemi_SSCL]_.
        """
        n = len(cw_ind)
        stage_ind = int(np.log2(n))
        bs = self._dec_pointer.shape[0]

        # update PM
        llr = np.zeros([bs, 2*self._list_size, n])
        for i in range(bs):
            llr_i = self.msg_llr[i, self._dec_pointer[i, :], stage_ind, :]
            llr[i, :, :] = llr_i[:, cw_ind]

        # upper branch has negative llr values (bit is 1)
        llr[:, self._list_size:, :] = - llr[:, self._list_size:, :]
        llr_in = np.maximum(np.minimum(llr, self._llr_max), -self._llr_max)
        pm_val = np.sum(np.log(1 + np.exp(-llr_in)), axis=-1)
        self.msg_pm += pm_val

        for i in range(bs):
            ind_dec = self._dec_pointer[i, self._list_size:]
            for j in cw_ind:
                self.msg_uhat[i, ind_dec, stage_ind, j] = 1

        # branch last bit and update pm at pos cw_ind[-1]
        self._update_single_bit_np([cw_ind[-1]])
        self._sort_decoders_np()
        self._duplicate_paths_np()

    def _update_single_bit_np(self, ind_u):
        """Update single bit at position ``ind_u`` of all decoders in Numpy."""

        if self._frozen_ind[ind_u]==0: # position is non-frozen
            ind_dec = np.expand_dims(self._dec_pointer[:, self._list_size:],
                                     axis=-1)
            uhat_slice = self.msg_uhat[:, :, 0, ind_u]
            np.put_along_axis(uhat_slice, ind_dec, 1., axis=1)
            self.msg_uhat[:, :, 0, ind_u] = uhat_slice


    def _update_pm_np(self, ind_u):
        """ Update path metric of all decoders at bit position ``ind_u`` in
        Numpy.

        We apply Eq. (10) from [Stimming_LLR]_.
        """
        ind = np.expand_dims(self._dec_pointer, axis=-1)
        u_hat = np.take_along_axis(self.msg_uhat[:, :, 0, ind_u], ind, axis=1)
        u_hat = np.squeeze(u_hat, axis=-1)
        llr_in = np.take_along_axis(self.msg_llr[:, :, 0, ind_u], ind, axis=1)
        llr_in = np.squeeze(llr_in, axis=-1)

        llr_clip = np.maximum(np.minimum(llr_in, self._llr_max), -self._llr_max)
        self.msg_pm += np.log(1 + np.exp(-np.multiply((1-2*u_hat), llr_clip)))

    def _sort_decoders_np(self):
        """Sort decoders according to their path metric."""

        ind = np.argsort(self.msg_pm, axis=-1)
        self.msg_pm = np.take_along_axis(self.msg_pm, ind, axis=1)
        self._dec_pointer = np.take_along_axis(self._dec_pointer, ind, axis=1)

    def _cn_op_np(self, x, y):
        """Check node update (boxplus) for LLRs in Numpy.

        See [Stimming_LLR]_ and [Hashemi_SSCL]_ for detailed equations.
        """
        x_in = np.maximum(np.minimum(x, self._llr_max), -self._llr_max)
        y_in = np.maximum(np.minimum(y, self._llr_max), -self._llr_max)

        # avoid division for numerical stability
        llr_out = np.log(1 + np.exp(x_in + y_in))
        llr_out -= np.log(np.exp(x_in) + np.exp(y_in))

        return llr_out


    def _vn_op_np(self, x, y, u_hat):
        """Variable node update (boxplus) for LLRs in Numpy."""
        return np.multiply((1-2*u_hat), x) + y

    def _duplicate_paths_np(self):
        """Copy first ``list_size``/2 paths into lower part in Numpy.

        Decoder indices are encoded in ``self._dec_pointer``.
        """
        ind_low = self._dec_pointer[:, :self._list_size]
        ind_up = self._dec_pointer[:, self._list_size:]

        for i in range(ind_up.shape[0]):
            self.msg_uhat[i, ind_up[i,:], :, :] = self.msg_uhat[i,
                                                                ind_low[i,:],
                                                                :, :]
            self.msg_llr[i, ind_up[i,:],:,:] = self.msg_llr[i, ind_low[i,:],:,:]

        # pm must be sorted directly (not accessed via pointer)
        self.msg_pm[:, self._list_size:] = self.msg_pm[:, :self._list_size]

    def _polar_decode_scl_np(self, cw_ind):
        """Recursive decoding function in Numpy.

        We follow the terminology from [Hashemi_SSCL]_ and [Stimming_LLR]_
        and branch the messages into a `left` and `right` update paths until
        reaching a leaf node.

        Tree pruning as proposed in [Hashemi_SSCL]_ is used to minimize the
        tree depth while maintaining the same output.
        """
        n = len(cw_ind)
        stage_ind = int(np.log2(n))

        # recursively branch through decoding tree
        if n>1:
            # prune tree if rate-0 subcode or rep-code is detected
            if self._use_fast_scl:
                if np.sum(self._frozen_ind[cw_ind])==n:
                    # rate0 code detected
                    self._update_rate0_code_np(cw_ind)
                    return
                if (self._frozen_ind[cw_ind[-1]]==0 and
                    np.sum(self._frozen_ind[cw_ind[:-1]])==n-1):
                    # rep code detected
                    self._update_rep_code_np(cw_ind)
                    return
            cw_ind_left = cw_ind[0:int(n/2)]
            cw_ind_right = cw_ind[int(n/2):]

            # ----- left branch -----
            llr_left = self.msg_llr[:, :, stage_ind, cw_ind_left]
            llr_right = self.msg_llr[:, :, stage_ind, cw_ind_right]

            self.msg_llr[:, :, stage_ind-1, cw_ind_left] = self._cn_op_np(
                                                                    llr_left,
                                                                    llr_right)

            # call left branch decoder
            self._polar_decode_scl_np(cw_ind_left)

            # ----- right branch -----
            u_hat_left_up = self.msg_uhat[:, :, stage_ind-1, cw_ind_left]
            llr_left = self.msg_llr[:, :, stage_ind, cw_ind_left]
            llr_right = self.msg_llr[:, :, stage_ind, cw_ind_right]

            self.msg_llr[:, :, stage_ind-1, cw_ind_right] = self._vn_op_np(
                                                                llr_left,
                                                                llr_right,
                                                                u_hat_left_up)

            # call right branch decoder
            self._polar_decode_scl_np(cw_ind_right)

            # combine u_hat
            u_hat_left_up = self.msg_uhat[:, :, stage_ind-1, cw_ind_left]
            u_hat_right_up = self.msg_uhat[:, :, stage_ind-1, cw_ind_right]

            # u_hat_left_up XOR u_hat_right_up
            u_hat_left =  (u_hat_left_up != u_hat_right_up) + 0

            u_hat = np.concatenate([u_hat_left, u_hat_right_up], axis=-1)

            # provide u_hat for next higher stage
            self.msg_uhat[:, :, stage_ind,  cw_ind] = u_hat

        else: # if leaf is reached perform basic decoding op (=decision)

            self._update_single_bit_np(cw_ind)

            # update PM
            self._update_pm_np(cw_ind)

            # position is non-frozen
            if self._frozen_ind[cw_ind]==0:
                # sort list
                self._sort_decoders_np()
                # duplicate the best list_size decoders
                self._duplicate_paths_np()
        return

    def _decode_np_batch(self, llr_ch):
        """Decode batch of ``llr_ch`` with Numpy decoder."""

        bs = llr_ch.shape[0]

        # allocate memory for all 2*list_size decoders
        self.msg_uhat = np.zeros([bs,
                                  2*self._list_size,
                                  self._n_stages+1,
                                  self._n])
        self.msg_llr = np.zeros([bs,
                                 2*self._list_size,
                                 self._n_stages+1,
                                 self._n])
        self.msg_pm = np.zeros([bs,
                                2*self._list_size])

        # L-1 decoders start with high penalty
        self.msg_pm[:,1:self._list_size] = self._llr_max
        # same for the second half of the L-1 decoders
        self.msg_pm[:,self._list_size+1:] = self._llr_max

        # use pointers to avoid in-memory sorting
        self._dec_pointer = np.arange(2*self._list_size)
        self._dec_pointer = np.tile(np.expand_dims(self._dec_pointer, axis=0),
                                    [bs,1])

        # init llr_ch (broadcast via list dimension)
        self.msg_llr[:, :, self._n_stages, :] = np.expand_dims(llr_ch, axis=1)

        # call recursive graph function
        self._polar_decode_scl_np(self._cw_ind)

        # select most likely candidate
        self._sort_decoders_np()

        # remove pointers
        for ind in range(bs):
            self.msg_uhat[ind, :, :, :] = self.msg_uhat[ind,
                                                        self._dec_pointer[ind],
                                                        :, :]
        return self.msg_uhat, self.msg_pm

    def _decode_np_hybrid(self, llr_ch, u_hat_sc, crc_valid):
        """Hybrid SCL decoding stage that decodes iff CRC from previous SC
        decoding attempt failed.

        This option avoids the usage of the high-complexity SCL decoder in cases
        where SC would be sufficient. For further details we refer to
        [Cammerer_Hybrid_SCL]_ (we use SC instead of the proposed BP stage).

        Remark: This decoder does not exactly implement SCL as the CRC
        can be false positive after the SC stage. However, in these cases
        SCL+CRC may also yield the wrong results.

        Remark 2: Due to the excessive control flow (if/else) and the
        varying batch-sizes, this function is only availabe as Numpy
        decoder (i.e., runs on the CPU).
        """

        bs = llr_ch.shape[0]
        crc_valid = np.squeeze(crc_valid, axis=-1)
        # index of codewords that need SCL decoding
        ind_invalid = np.arange(bs)[np.invert(crc_valid)]

        # init SCL decoder for bs_hyb samples requiring SCL dec.
        llr_ch_hyb = np.take(llr_ch, ind_invalid, axis=0)
        msg_uhat_hyb, msg_pm_hyb = self._decode_np_batch(llr_ch_hyb)

        # merge results with previously decoded SC results
        msg_uhat = np.zeros([bs, 2*self._list_size, 1, self._n])
        msg_pm = np.ones([bs, 2*self._list_size]) * self._llr_max * self.k
        msg_pm[:, 0] = 0

        # copy SC data
        msg_uhat[:, 0, 0, self._info_pos] = u_hat_sc

        ind_hyb = 0
        for ind in range(bs):
            if not crc_valid[ind]:
                #copy data from SCL
                msg_uhat[ind, :, 0, :] = msg_uhat_hyb[ind_hyb, :, 0, :]
                msg_pm[ind, :] = msg_pm_hyb[ind_hyb, :]
                ind_hyb += 1

        return msg_uhat, msg_pm

    #########################
    # Keras layer functions
    #########################

    def build(self, input_shape):
        """Build and check if shape of input is invalid."""
        assert (input_shape[-1]==self._n), "Invalid input shape."
        assert (len(input_shape)>=2), 'Inputs must have at least 2 dimensions.'

    def call(self, inputs):
        """Successive cancellation list (SCL) decoding function.

        This function performs successive cancellation list decoding
        and returns the estimated information bits.

        An outer CRC can be applied optionally by setting ``crc_degree``.

        Args:
            inputs (tf.float32): Tensor of shape `[...,n]` containing the
                channel LLR values (as logits).

        Returns:
            `tf.float32`: Tensor of shape `[...,k]` containing
            hard-decided estimations of all ``k`` information bits.

        Raises:
            ValueError: If ``inputs`` is not of shape `[..., n]`
                or `dtype` is not `tf.float32`.

            InvalidArgumentError: When rank(``inputs``)<2.

        Note:
            This function recursively unrolls the SCL decoding tree, thus,
            for larger values of ``n`` building the decoding graph can become
            time consuming. Please consider the ``cpu_only`` option instead.
        """

        tf.debugging.assert_type(inputs, self._output_dtype,
                                 "Invalid input dtype.")
        # internal calculations still in tf.float32
        inputs = tf.cast(inputs, tf.float32)

        # last dim must be of length n
        tf.debugging.assert_equal(tf.shape(inputs)[-1],
                                  self._n,
                                  "Last input dimension must be of length n.")

        # Reshape inputs to [-1, n]
        tf.debugging.assert_greater(tf.rank(inputs), 1)
        input_shape = inputs.shape
        new_shape = [-1, self._n]
        llr_ch = tf.reshape(inputs, new_shape)

        llr_ch = -1. * llr_ch # logits are converted into "true" llrs

        # if activated use Numpy decoder
        if self._use_hybrid_sc:
            # use SC decoder to decode first
            u_hat = self._decoder_sc(-llr_ch)
            _, crc_valid = self._crc_decoder(u_hat)
            msg_uhat, msg_pm = tf.py_function(func=self._decode_np_hybrid,
                                              inp=[llr_ch, u_hat, crc_valid],
                                              Tout=[tf.float32, tf.float32])
            # note: return shape is only 1 in 3. dim (to avoid copy overhead)
            msg_uhat = tf.reshape(msg_uhat, [-1, 2*self._list_size, 1, self._n])
            msg_pm = tf.reshape(msg_pm, [-1, 2*self._list_size])
        else:
            if self._cpu_only:
                msg_uhat, msg_pm = tf.py_function(func=self._decode_np_batch,
                                                  inp=[llr_ch],
                                                  Tout=[tf.float32, tf.float32])
                # restore shape information
                msg_uhat = tf.reshape(msg_uhat,
                            [-1, 2*self._list_size, self._n_stages+1, self._n])
                msg_pm = tf.reshape(msg_pm, [-1, 2*self._list_size])
            else:
                msg_uhat, msg_pm = self._decode_tf(llr_ch)

        # check crc (and remove CRC parity bits)
        if self._use_crc:
            u_hat_list = tf.gather(msg_uhat[:, :, 0, :], self._info_pos,
                                   axis=-1)
            u_hat_list, crc_valid = self._crc_decoder(u_hat_list)
            # add penalty to pm if CRC fails
            pm_penalty = ((1. - tf.cast(crc_valid, tf.float32))
                       * self._llr_max * self.k)
            msg_pm += tf.squeeze(pm_penalty, axis=2)

        # select most likely candidate
        cand_ind = tf.argmin(msg_pm, axis=-1)
        c_hat = tf.gather(msg_uhat[:, :, 0, :], cand_ind, axis=1, batch_dims=1)
        u_hat = tf.gather(c_hat, self._info_pos, axis=-1)

        # and reconstruct input shape
        output_shape = input_shape.as_list()
        output_shape[-1] = self.k
        output_shape[0] = -1 # first dim can be dynamic (None)
        u_hat_reshape = tf.reshape(u_hat, output_shape)
        return tf.cast(u_hat_reshape, self._output_dtype)


class PolarBPDecoder(Layer):
    # pylint: disable=line-too-long
    """PolarBPDecoder(frozen_pos, n, num_iter=20, hard_out=True, output_dtype=tf.float32, **kwargs)

    Belief propagation (BP) decoder for Polar codes [Arikan_Polar]_ and
    Polar-like codes based on [Arikan_BP]_ and [Forney_Graphs]_.

    The class inherits from the Keras layer class and can be used as layer in a
    Keras model.

    Remark: The PolarBPDecoder does currently not support XLA.

    Parameters
    ----------
        frozen_pos: ndarray
            Array of `int` defining the ``n-k`` indices of the frozen positions.

        n: int
            Defining the codeword length.

        num_iter: int
            Defining the number of decoder iterations (no early stopping used
            at the moment).

        hard_out: bool
            Defaults to True. If True, the decoder provides hard-decided
            information bits instead of soft-values.

        output_dtype: tf.DType
            Defaults to tf.float32. Defines the output datatype of the layer
            (internal precision remains tf.float32).

    Input
    -----
        inputs: [...,n], tf.float32
            2+D tensor containing the channel logits/llr values.

    Output
    ------
        : [...,k], tf.float32
            2+D tensor containing bit-wise soft-estimates
            (or hard-decided bit-values) of all ``k`` information bits.

    Raises
    ------
        AssertionError
            If ``n`` is not `int`.

        AssertionError
            If ``n`` is not a power of 2.

        AssertionError
            If the number of elements in ``frozen_pos`` is greater than ``n``.

        AssertionError
            If ``frozen_pos`` does not consists of `int`.

        AssertionError
            If ``hard_out`` is not `bool`.

        ValueError
            If ``output_dtype`` is not {tf.float16, tf.float32, tf.float64}.

        AssertionError
            If ``num_iter`` is not `int`.

        AssertionError
            If ``num_iter`` is not a positive value.

    Note
    ----
        This decoder is fully differentiable and, thus, well-suited for
        gradient descent-based learning tasks such as `learned code design`
        [Ebada_Design]_.

        As commonly done, we assume frozen bits are set to `0`. Please note
        that - although its practical relevance is only little - setting frozen
        bits to `1` may result in `affine` codes instead of linear code as the
        `all-zero` codeword is not necessarily part of the code any more.

    """

    def __init__(self,
                 frozen_pos,
                 n,
                 num_iter=20,
                 hard_out=True,
                 output_dtype=tf.float32,
                 **kwargs):

        if output_dtype not in (tf.float16, tf.float32, tf.float64):
            raise ValueError(
                'output_dtype must be {tf.float16, tf.float32, tf.float64}.')

        if output_dtype is not tf.float32:
            print('Note: decoder uses tf.float32 for internal calculations.')

        super().__init__(dtype=output_dtype, **kwargs)
        self._output_dtype = output_dtype

        # assert error if r>1 or k, n are negative
        assert isinstance(n, numbers.Number), "n must be a number."
        n = int(n) # n can be float (e.g. as result of n=k*r)
        assert issubdtype(frozen_pos.dtype, int), "frozen_pos contains non int."
        assert len(frozen_pos)<=n, "Num. of elements in frozen_pos cannot " \
            "be greater than n."
        assert np.log2(n)==int(np.log2(n)), "n must be a power of 2."

        assert isinstance(hard_out, bool), "hard_out must be boolean."

        # store internal attributes
        self._n = n
        self._frozen_pos = frozen_pos
        self._k = self._n - len(self._frozen_pos)
        self._info_pos = np.setdiff1d(np.arange(self._n), self._frozen_pos)
        assert self._k==len(self._info_pos), "Internal error: invalid " \
                                             "info_pos generated."

        assert isinstance(num_iter, int), "num_iter must be integer."
        assert num_iter>0, "num_iter must be a positive value."
        self._num_iter = tf.constant(num_iter, dtype=tf.int32)

        self._llr_max = 19.3 # internal max LLR value
        self._hard_out = hard_out

        # depth of decoding graph
        self._n_stages = int(np.log2(self._n))

    #########################################
    # Public methods and properties
    #########################################

    @property
    def n(self):
        """Codeword length."""
        return self._n

    @property
    def k(self):
        """Number of information bits."""
        return self._k

    @property
    def frozen_pos(self):
        """Frozen positions for Polar decoding."""
        return self._frozen_pos

    @property
    def info_pos(self):
        """Information bit positions for Polar encoding."""
        return self._info_pos

    @property
    def llr_max(self):
        """Maximum LLR value for internal calculations."""
        return self._llr_max

    @property
    def num_iter(self):
        """Number of decoding iterations."""
        return self._num_iter

    @property
    def hard_out(self):
        """Indicates if decoder hard-decides outputs."""
        return self._hard_out

    @property
    def output_dtype(self):
        """Output dtype of decoder."""
        return self._output_dtype

    @num_iter.setter
    def num_iter(self, num_iter):
        "Number of decoding iterations."
        assert isinstance(num_iter, int), 'num_iter must be int.'
        assert num_iter>=0, 'num_iter cannot be negative.'
        self._num_iter = tf.constant(num_iter, dtype=tf.int32)

    #########################
    # Utility methods
    #########################

    def _boxplus_tf(self, x, y):
        """Check-node update (boxplus) for LLR inputs.

        Operations are performed element-wise.
        """
        x_in = tf.clip_by_value(x,
                                clip_value_min=-self._llr_max,
                                clip_value_max=self._llr_max)
        y_in = tf.clip_by_value(y,
                                clip_value_min=-self._llr_max,
                                clip_value_max=self._llr_max)

        # avoid division for numerical stability
        llr_out = tf.math.log(1 + tf.math.exp(x_in + y_in))
        llr_out -= tf.math.log(tf.math.exp(x_in) + tf.math.exp(y_in))

        return llr_out

    def _decode_bp(self, llr_ch, num_iter):
        """Iterative BP decoding function with LLR-values.

        Args:
            llr_ch (tf.float32): Tensor of shape `[batch_size, n]` containing
                the channel logits/llr values where `batch_size` denotes the
                batch-size.

            num_iter (int): Defining the number of decoder iteration
                (no early stopping used at the moment).
        Returns:
            `tf.float32`: Tensor of shape `[batch_size, k]` containing
            bit-wise soft-estimates (or hard-decided bit-values) of all
            information bits.
        """

        bs = tf.shape(llr_ch)[0]

        # store intermediate Tensors in TensorArray
        msg_l = tf.TensorArray(tf.float32,
                               size=num_iter*(self._n_stages+1),
                               dynamic_size=False,
                               clear_after_read=False)

        msg_r = tf.TensorArray(tf.float32,
                               size=num_iter*(self._n_stages+1),
                               dynamic_size=False,
                               clear_after_read=False)

        # init frozen positions with infinity
        msg_r_in = np.zeros([1, self._n])
        msg_r_in[:, self._frozen_pos] = self._llr_max
        # copy for all batch-samples
        msg_r_in = tf.tile(tf.constant(msg_r_in, tf.float32), [bs, 1])

        # perform decoding iterations
        for ind_it in tf.range(self._num_iter):
            # update left-to-right messages
            for ind_s in range(self._n_stages):
                # calc indices
                ind_range = np.arange(int(self._n/2))
                ind_1 = ind_range * 2 - np.mod(ind_range, 2**ind_s)
                ind_2 = ind_1 + 2**ind_s
                # simplify gather with concatenated outputs
                ind_inv = np.argsort(np.concatenate([ind_1, ind_2], axis=0))

                # load incoming l messages
                if ind_s==self._n_stages-1:
                    l1_in = tf.gather(llr_ch, ind_1, axis=1)
                    l2_in = tf.gather(llr_ch, ind_2, axis=1)
                elif ind_it==0:
                    l1_in = tf.zeros([bs, int(self._n/2)])
                    l2_in = tf.zeros([bs, int(self._n/2)])
                else:
                    l_in = msg_l.read((ind_s+1) + (ind_it-1)*(self._n_stages+1))
                    l1_in = tf.gather(l_in, ind_1, axis=1)
                    l2_in = tf.gather(l_in, ind_2, axis=1)

                # load incoming r messages
                if ind_s==0:
                    r1_in = tf.gather(msg_r_in, ind_1, axis=1)
                    r2_in = tf.gather(msg_r_in, ind_2, axis=1)
                else:
                    r_in = msg_r.read(ind_s + ind_it*(self._n_stages+1))
                    r1_in = tf.gather(r_in, ind_1, axis=1)
                    r2_in = tf.gather(r_in, ind_2, axis=1)

                r1_out = self._boxplus_tf(r1_in, l2_in + r2_in)
                r2_out = self._boxplus_tf(r1_in, l1_in) + r2_in

                # and re-concatenate output
                r_out = tf.concat([r1_out, r2_out], 1)
                r_out = tf.gather(r_out, ind_inv, axis=1)
                msg_r = msg_r.write((ind_s+1)
                                     + ind_it*(self._n_stages+1), r_out)

            # update right-to-left messages
            for ind_s in range(self._n_stages-1, -1, -1):
                ind_range = np.arange(int(self._n/2))
                ind_1 = ind_range * 2 - np.mod(ind_range, 2**ind_s)
                ind_2 = ind_1 + 2**ind_s
                ind_inv = np.argsort(np.concatenate([ind_1, ind_2], axis=0))

                # load messages
                if ind_s==self._n_stages-1:
                    l1_in = tf.gather(llr_ch, ind_1, axis=1)
                    l2_in = tf.gather(llr_ch, ind_2, axis=1)
                else:
                    l_in = msg_l.read((ind_s+1)+ind_it*(self._n_stages+1))
                    l1_in = tf.gather(l_in, ind_1, axis=1)
                    l2_in = tf.gather(l_in, ind_2, axis=1)

                if ind_s==0:
                    r1_in = tf.gather(msg_r_in, ind_1, axis=1)
                    r2_in = tf.gather(msg_r_in, ind_2, axis=1)
                else:
                    r_in = msg_r.read(ind_s + ind_it*(self._n_stages+1))
                    r1_in = tf.gather(r_in, ind_1, axis=1)
                    r2_in = tf.gather(r_in, ind_2, axis=1)

                # node update functions
                l1_out = self._boxplus_tf(l1_in, l2_in + r2_in)
                l2_out = self._boxplus_tf(r1_in, l1_in) + l2_in

                l_out = tf.concat([l1_out, l2_out], 1)
                l_out = tf.gather(l_out, ind_inv, axis=1)
                msg_l = msg_l.write(ind_s + ind_it*(self._n_stages+1), l_out)

        # recover u_hat
        u_hat = tf.gather(msg_l.read((num_iter-1)*(self._n_stages+1)),
                          self._info_pos,
                          axis=1)
        # if active, hard-decide output bits
        if self._hard_out:
            u_hat = tf.where(u_hat>0, 0., 1.)
        else: # re-transform soft output to logits (instead of llrs)
            u_hat = -1. * u_hat
        return u_hat

    #########################
    # Keras layer functions
    #########################

    def build(self, input_shape):
        """Build and check if shape of input is invalid."""
        assert (input_shape[-1]==self._n), "Invalid input shape"
        assert (len(input_shape)>=2), 'Inputs must have at least 2 dimensions.'

    def call(self, inputs):
        """Iterative BP decoding function.

        This function performs `num_iter` belief propagation decoding iterations
        and returns the estimated information bits.

        Args:
            inputs (tf.float32): Tensor of shape `[...,n]` containing the
                channel logits/llr values.

        Returns:
            `tf.float32`: Tensor of shape `[...,k]` containing
                bit-wise soft-estimates (or hard-decided bit-values) of all
                ``k`` information bits.

        Raises:
            ValueError: If ``inputs`` is not of shape `[..., n]`
                or `dtype` is not `output_dtype`.

            InvalidArgumentError: When rank(``inputs``)<2.

        Note:
            This function recursively unrolls the BP decoding graph, thus,
            for larger values of ``n`` or more iterations, building the
            decoding graph can become time and memory consuming.
        """

        tf.debugging.assert_type(inputs, self._output_dtype,
                                 "Invalid input dtype.")
        # internal calculations still in tf.float32
        inputs = tf.cast(inputs, tf.float32)

        # Reshape inputs to [-1, n]
        input_shape = inputs.shape
        new_shape = [-1, self._n]
        llr_ch = tf.reshape(inputs, new_shape)

        llr_ch = -1. * llr_ch # logits are converted into "true" llrs

        # and decode
        u_hat = self._decode_bp(llr_ch, self._num_iter)

        # and reconstruct input shape
        output_shape = input_shape.as_list()
        output_shape[-1] = self.k
        output_shape[0] = -1 # first dim can be dynamic (None)
        u_hat_reshape = tf.reshape(u_hat, output_shape)
        return tf.cast(u_hat_reshape, self._output_dtype)


class Polar5GDecoder(Layer):
    # pylint: disable=line-too-long
    """Polar5GDecoder(enc_polar, dec_type="SC", list_size=8, num_iter=20, output_dtype=tf.float32, **kwargs)

    Wrapper for 5G compliant decoding including rate-recovery and CRC removal.

    The class inherits from the Keras layer class and can be used as layer in a
    Keras model.

    Parameters
    ----------
        enc_polar: Polar5GEncoder
            Instance of the :class:`~sionna.fec.polar.encoding.Polar5GEncoder`
            used for encoding including rate-matching.

        dec_type: str
            Defaults to `"SC"`. Defining the decoder to be used.
            Must be one of the following `{"SC", "SCL", "hybSCL", "BP"}`.

        list_size: int
            Defaults to 8. Defining the list size `iff` list-decoding is used.
            Only required for ``dec_types`` `{"SCL", "hybSCL"}`.

        num_iter: int
            Defaults to 20. Defining the number of BP iterations. Only required
            for ``dec_type`` `"BP"`.

        output_dtype: tf.DType
            Defaults to tf.float32. Defines the output datatype of the layer
            (internal precision remains tf.float32).

    Input
    -----
        inputs: [...,n], tf.float32
            2+D tensor containing the channel logits/llr values.

    Output
    ------
        : [...,k], tf.float32
            2+D tensor  containing hard-decided estimates of all `k`
            information bits.

    Raises
    ------
        AssertionError
            If ``enc_polar`` is not `Polar5GEncoder`.

        ValueError
            If ``dec_type`` is not `{"SC", "SCL", "SCL8", "SCL32", "hybSCL",
            "BP"}`.

        AssertionError
            If ``dec_type`` is not `str`.

        ValueError
            If ``inputs`` is not of shape `[..., n]` or `dtype` is not
            the same as ``output_dtype``.

        InvalidArgumentError
            When rank(``inputs``)<2.

    Note
    ----
        This layer only supports the uplink polar rate-matching scheme without
        `codeword segmentation`.

        Although the decoding `list size` is not provided by 3GPP
        [3GPPTS38212]_, the consortium has agreed on a `list size` of 8 for the
        5G decoding reference curves [Bioglio_Design]_.

        All list-decoders apply `CRC-aided` decoding, however, the non-list
        decoders (`"SC"` and `"BP"`) cannot materialize the CRC leading to an
        effective rate-loss.

    """

    def __init__(self,
                 enc_polar,
                 dec_type="SC",
                 list_size=8,
                 num_iter=20,
                 output_dtype=tf.float32,
                 **kwargs):

        if output_dtype not in (tf.float16, tf.float32, tf.float64):
            raise ValueError(
                'output_dtype must be {tf.float16, tf.float32, tf.float64}.')

        if output_dtype is not tf.float32:
            print('Note: decoder uses tf.float32 for internal calculations.')
        self._output_dtype = output_dtype

        super().__init__(dtype=output_dtype, **kwargs)

        assert isinstance(enc_polar, Polar5GEncoder), \
                                    "enc_polar must be Polar5GEncoder."
        assert isinstance(dec_type, str), "dec_type must be str."
        # list_size and num_iter are not checked here (done during decoder init)

        # Store internal attributes
        self._n_target = enc_polar.n_target
        self._k_target = enc_polar.k_target
        self._n_polar = enc_polar.n
        self._k_polar = enc_polar.k
        self._k_crc = enc_polar.enc_crc.crc_length
        self._llr_max = 100 # Internal max LLR value (for punctured positions)
        self._enc_polar = enc_polar
        self._dec_type = dec_type

        # Initialize the de-interleaver patterns
        self._init_interleavers()

        # Initialize decoder
        if dec_type=="SC":
            print("Warning: 5G Polar codes use an integrated CRC that " \
                  "cannot be materialized with SC decoding and, thus, " \
                  "causes a degraded performance. Please consider SCL " \
                  "decoding instead.")
            self._polar_dec = PolarSCDecoder(self._enc_polar.frozen_pos,
                                             self._n_polar)
        elif dec_type=="SCL":
            self._polar_dec = PolarSCLDecoder(self._enc_polar.frozen_pos,
                                self._n_polar,
                                crc_degree=self._enc_polar.enc_crc.crc_degree,
                                list_size=list_size)
        elif dec_type=="hybSCL":
            self._polar_dec = PolarSCLDecoder(self._enc_polar.frozen_pos,
                                self._n_polar,
                                crc_degree=self._enc_polar.enc_crc.crc_degree,
                                list_size=list_size,
                                use_hybrid_sc=True)
        elif dec_type=="BP":
            print("Warning: 5G Polar codes use an integrated CRC that " \
                  "cannot be materialized with BP decoding and, thus, " \
                  "causes a degraded performance. Please consider SCL " \
                  " decoding instead.")
            assert isinstance(num_iter, int), "num_iter must be int."
            assert num_iter > 0, "num_iter must be positive."
            self._num_iter = num_iter
            self._polar_dec = PolarBPDecoder(self._enc_polar.frozen_pos,
                                             self._n_polar,
                                             num_iter=num_iter,
                                             hard_out=True)
        else:
            raise ValueError("Unkown value for dec_type.")

    #########################################
    # Public methods and properties
    #########################################

    @property
    def k_target(self):
        """Number of information bits including rate-matching."""
        return self._k_target

    @property
    def n_target(self):
        """Codeword length including rate-matching."""
        return self._n_target

    @property
    def k_polar(self):
        """Number of information bits of mother Polar code."""
        return self._k_polar

    @property
    def n_polar(self):
        """Codeword length of mother Polar code."""
        return self._n_polar

    @property
    def frozen_pos(self):
        """Frozen positions for Polar decoding."""
        return self._frozen_pos

    @property
    def info_pos(self):
        """Information bit positions for Polar encoding."""
        return self._info_pos

    @property
    def llr_max(self):
        """Maximum LLR value for internal calculations."""
        return self._llr_max

    @property
    def dec_type(self):
        """Decoder type used for decoding as str."""
        return self._dec_type

    @property
    def polar_dec(self):
        """Decoder instance used for decoding."""
        return self._polar_dec

    @property
    def output_dtype(self):
        """Output dtype of decoder."""
        return self._output_dtype

    #########################
    # Utility methods
    #########################

    def _init_interleavers(self):
        """Initialize inverse interleaver patterns for rate-recovery."""

        # Channel interleaver
        ind_ch_int = self._enc_polar.channel_interleaver(
                                                np.arange(self._n_target))
        self.ind_ch_int_inv = np.argsort(ind_ch_int) # Find inverse perm

        # Sub-block interleaver
        ind_sub_int = self._enc_polar.subblock_interleaving(
                                                np.arange(self._n_polar))
        self.ind_sub_int_inv = np.argsort(ind_sub_int) # Find inverse perm

    #########################
    # Keras layer functions
    #########################

    def build(self, input_shape):
        """Build and check if shape of input is invalid."""
        assert (input_shape[-1]==self._n_target), "Invalid input shape."
        assert (len(input_shape)>=2), 'Inputs must have at least 2 dimensions.'

    def call(self, inputs):
        """Polar decoding and rate-recovery for uplink 5G Polar codes.

        Args:
            inputs (tf.float32): Tensor of shape `[...,n]` containing the
                channel logits/llr values.

        Returns:
            `tf.float32`: Tensor of shape `[...,k]` containing
                hard-decided estimates of all ``k`` information bits.

        Raises:
            ValueError: If ``inputs`` is not of shape `[..., n]`
                or `dtype` is not `output_dtype`.

            InvalidArgumentError: When rank(``inputs``)<2.
        """

        tf.debugging.assert_type(inputs, self._output_dtype,
                                 "Invalid input dtype.")
        # internal calculations still in tf.float32
        inputs = tf.cast(inputs, tf.float32)

        # Reshape inputs to [-1, n]
        tf.debugging.assert_greater(tf.rank(inputs), 1)
        input_shape = inputs.shape
        new_shape = [-1, self._n_target]
        llr_ch = tf.reshape(inputs, new_shape)

        # Note: logits are not inverted here; this is done in the decoder itself

        # 1.) Undo channel interleaving
        llr_deint = tf.gather(llr_ch, self.ind_ch_int_inv, axis=1)

        # 2.) Remove puncturing, shortening, repetition (see Sec. 5.4.1.2)
        # a) Puncturing: set LLRs to 0
        # b) Shortening: set LLRs to infinity
        # c) Repetition: combine LLRs
        if self._n_target >= self._n_polar:
            # Repetition coding
            # Add the last n_rep positions to the first llr positions
            n_rep = self._n_target - self._n_polar
            llr_1 = llr_deint[:,:n_rep]
            llr_2 = llr_deint[:,n_rep:self._n_polar]
            llr_3 = llr_deint[:,self._n_polar:]
            llr_dematched = tf.concat([llr_1+llr_3, llr_2], 1)
        else:
            if self._k_polar/self._n_target <= 7/16:
                # Puncturing
                # Append n_polar - n_target "zero" llrs to first positions
                llr_zero = tf.zeros([tf.shape(llr_deint)[0],
                                     self._n_polar-self._n_target])
                llr_dematched = tf.concat([llr_zero, llr_deint], 1)
            else:
                # Shortening
                # Append n_polar - n_target "-infinity" llrs to last positions
                # Remark: we still operate with logits here, thus the neg. sign
                llr_infty = -self._llr_max * tf.ones([tf.shape(llr_deint)[0],
                                                self._n_polar-self._n_target])
                llr_dematched = tf.concat([llr_deint, llr_infty], 1)

        # 3.) Remove subblock interleaving
        llr_dec = tf.gather(llr_dematched, self.ind_sub_int_inv, axis=1)
        # 4.) Run main decoder
        u_hat_crc = self._polar_dec(llr_dec)

        # 5.) Shortening should be implicitely recovered by decoder

        # 6.) Remove CRC (and PC)
        u_hat = u_hat_crc[:,:-self._k_crc]

        # And reconstruct input shape
        output_shape = input_shape.as_list()
        output_shape[-1] = self._k_target
        output_shape[0] = -1 # First dim can be dynamic (None)
        u_hat_reshape = tf.reshape(u_hat, output_shape)

        return tf.cast(u_hat_reshape, dtype=self._output_dtype)
