#
# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Layer for Convolutional Code Viterbi Decoding."""

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer
from sionna.fec.utils import int2bin
from sionna.fec.conv.utils import polynomial_selector, Trellis

class ViterbiDecoder(Layer):
    # pylint: disable=line-too-long
    r"""ViterbiDecoder(gen_poly=None, rate=1/2, constraint_length=3, method='soft_llr', output_dtype=tf.float32, **kwargs)

    Implements the Viterbi decoding algorithm [Viterbi]_ that returns an
    estimate of the information bits for a noisy convolutional codeword.
    Takes as input either LLR values (`method` = `soft_llr`) or hard bit values
    (`method` = `hard`) and returns the hard decided estimate of information
    bits.

    The class inherits from the Keras layer class and can be used as layer in
    a Keras model.

    Parameters
    ----------
    gen_poly: tuple
         tuple of strings with each string being a 0, 1 sequence. If `None`,
        ``rate`` and ``constraint_length`` must be provided.

    rate: float
        Valid values are 1/3 and 0.5. Only required if ``gen_poly`` is `None`.

    constraint_length: int
        Valid values are between 3 and 8 inclusive. Only required if
        ``gen_poly`` is `None`.

    method: str
        Choices are `soft_llr' or `hard` or `soft`. In computing path
        metrics, `soft_llr` expects channel LLRs whereas `hard` assumes a `binary symmetric channel` (BSC). In case of `hard`, `inputs` will be quantized to 0/1 values.

    output_dtype: tf.DType
        Defaults to tf.float32. Defines the output datatype of the layer.

    Input
    -----
        inputs: [...,n], tf.float32
            2+D tensor containing the (noisy) channel output symbols where `n`
            denotes the codeword length.

    Output
    ------
        : [...,rate*n], tf.float32
            2+D tensor containing the estimates of the information bit tensor.

    Note
    ----
    A full implementation of the decoder rather than a windowed approach
    is used. For a given codeword of duration `T`, the path metric is
    computed from time `0` to `T` and the path with optimal metric at time
    `T` is selected. The optimal path is then traced back from `T` to `0`
    to output the estimate of the information bit vector used to encode.
    For larger codewords, note that the current method is sub-optimal
    in terms of memory utilization and latency.
    """

    def __init__(self,
                 gen_poly=None,
                 rate=1/2,
                 constraint_length=3,
                 method='soft_llr',
                 return_info_bits=True,
                 output_dtype=tf.float32,
                 **kwargs):

        super().__init__(**kwargs)
        valid_rates = (1/2, 1/3)
        valid_constraint_length = (3, 4, 5, 6, 7, 8)

        if gen_poly is not None:
            assert all(isinstance(poly, str) for poly in gen_poly), \
                "Each polynomial must be a string."
            assert all(len(poly)==len(gen_poly[0]) for poly in gen_poly), \
                "Each polynomial must be of same length."
            assert all(all(
                char in ['0','1'] for char in poly) for poly in gen_poly),\
                "Each polynomial must be a string of 0's and 1's."
            self._gen_poly = gen_poly
        else:
            valid_rates = (1/2, 1/3)
            valid_constraint_length = (3, 4, 5, 6, 7, 8)

            assert constraint_length in valid_constraint_length, \
                "Constraint length must be between 3 and 8."
            assert rate in valid_rates, \
                "Rate must be 1/3 or 1/2."
            self._gen_poly = polynomial_selector(rate, constraint_length)

        assert method in ('soft_llr', 'hard'), \
            "method must be `soft_llr` or `hard`."

        # init Trellis parameters
        self._trellis = Trellis(self.gen_poly, rsc=False)
        self._coderate = 1/len(self.gen_poly)

        # conv_k denotes number of input bit streams
        # can only be 1 in current implementation
        self._conv_k = self._trellis.conv_k

        # conv_n denotes number of output bits for conv_k input bits
        self._conv_n = self._trellis.conv_n

        self._k = None
        self._n = None
        # num_syms denote number of encoding periods or state transitions.
        self._num_syms = None

        self._ni = 2**self._conv_k
        self._no = 2**self._conv_n
        self._ns = self._trellis.ns

        self._method = method
        self._return_info_bits = return_info_bits
        self.output_dtype = output_dtype
        # If i->j state transition emits symbol k, tf.gather with ipst_op_idx
        # gathers (i,k) element from input in row j.
        self.ipst_op_idx = self._mask_by_tonode()

    #########################################
    # Public methods and properties
    #########################################

    @property
    def gen_poly(self):
        """The generator polynomial used by the encoder."""
        return self._gen_poly

    @property
    def coderate(self):
        """Rate of the code used in the encoder."""
        return self._coderate

    @property
    def trellis(self):
        """Trellis object used during encoding."""
        return self._trellis

    #########################
    # Utility functions
    #########################

    def _mask_by_tonode(self):
        r"""
         _Ns x _No index matrix, each element of shape (2,)
         where num_ops = 2**conv_n
         When applied as tf.gather index on a Ns x  num_ops matrix
         ((i,j) denoting metric for prev_st=i and output=j)
         the output is matrix sorted by next_state. Row i in output
         denotes the 2 possible metrics for transition to state i.
        """
        cnst = self._ns * self._ni
        from_nodes_vec = tf.reshape(self._trellis.from_nodes,(cnst,))
        op_idx = tf.reshape(self._trellis.op_by_tonode, (cnst,))
        st_op_idx = tf.transpose(tf.stack([from_nodes_vec, op_idx]))
        st_op_idx = tf.reshape(st_op_idx[None,:,:],(self._ns, self._ni, 2))

        return st_op_idx

    def _update(self, cum_tminus1, metrics_t):
        r"""
        Update optimal cumulative path metrics at time t given optimal
        cumulative metrics at time t-1.

        Also returns tb_states, the traceback states at t-1 that result
        in optimal cumulative metric at time t, for each state.
        """
        state_vec = tf.tile(tf.range(self._ns, dtype=tf.int32)[None,:],
                            [tf.shape(cum_tminus1)[0], 1])
        # Ns x No matrix. Element (s,j) is path_metric at state s,tminus1
        # with transition output j
        sum_metric = tf.math.add(
            tf.expand_dims(metrics_t, axis=1),
            tf.cast(tf.expand_dims(cum_tminus1, axis=-1),tf.float32))

        sum_metric_bytonode = tf.gather_nd(sum_metric,
            tf.tile(self.ipst_op_idx[None,:],
                    [tf.shape(cum_tminus1)[0],1,1,1]),
                    batch_dims=1)
        tb_state_idx = tf.cast(tf.math.argmin(sum_metric_bytonode,axis=2),
                               tf.int32)
        # Transition to States argmin state index
        from_st_idx = tf.transpose(tf.stack([state_vec, tb_state_idx]),
                                   perm=[1, 2,0])
        tb_states = tf.gather_nd(self._trellis.from_nodes, from_st_idx)
        cum_t = tf.math.reduce_min(sum_metric_bytonode,axis=2)
        return cum_t, tb_states

    def _op_bits_path(self, paths):
        r"""
        Given a path, compute the input bit stream that results in the path.
        Used in call() where the input is optimal path (seq of states) such
        as the path returned by _return_optimal.
        """
        paths = tf.cast(paths, tf.int32)
        ip_bits = tf.TensorArray(tf.int32,
                                  size=paths.shape[-1]-1,
                                  dynamic_size=False,
                                  clear_after_read=False)
        dec_syms = tf.TensorArray(tf.int32,
                                  size=paths.shape[-1]-1,
                                  dynamic_size=False,
                                  clear_after_read=False)
        ni = self._trellis.ni
        ip_sym_mask = tf.range(ni)[None, :]

        for sym in tf.range(1, paths.shape[-1]):

            # gather index from paths to enable XLA
            # replaces p_idx = paths[:,sym-1:sym+1]
            p_idx = tf.gather(paths, [sym-1, sym], axis=-1)
            dec_ = tf.gather_nd(self._trellis.op_mat, p_idx)

            dec_syms = dec_syms.write(sym-1, value=dec_)
            # bs x ni boolean tensor. Each row has a True and False. True
            # corresponds to input_bit which produced the next state (t=sym)
            match_st = tf.math.equal(
                tf.gather(self._trellis.to_nodes,paths[:, sym-1]),
                tf.tile(paths[:, sym][:, None], [1, 2])
                )

            # tf.boolean_mask throws error in XLA mode
            #ip_bit = tf.boolean_mask(ip_sym_mask, match_st)

            ip_bit_ = tf.where(match_st,
                               ip_sym_mask,
                               tf.zeros_like(ip_sym_mask))
            ip_bit = tf.reduce_sum(ip_bit_, axis=-1)
            ip_bits = ip_bits.write(sym-1, ip_bit)

        ip_bit_vec_est = tf.transpose(ip_bits.stack())
        ip_sym_vec_est = tf.transpose(dec_syms.stack())

        return ip_bit_vec_est, ip_sym_vec_est

    def _optimal_path(self, cm_, tb_):
        r"""
        Compute optimal path (state at each time t) given tensors cm_ & tb_
        of shapes (None, Ns, T). Output is of shape (None, T)
        cm_: cumulative metrics for each state at time t(0 to T)
        tb_: traceback state for each state at time t(0 to T)
        """
        # tb and ca are of shape batch x self._ns x num_syms
        assert(tb_.get_shape()[1] == self._ns), "Invalid shape."
        optst_ta = tf.TensorArray(tf.int32, size=tb_.shape[-1],
                                  dynamic_size=False,
                                  clear_after_read=False)

        opt_term_state =tf.cast(tf.argmin(cm_[:, :, -1], axis=1), tf.int32)
        optst_ta= optst_ta.write(tb_.shape[-1]-1,opt_term_state)

        for sym in tf.range(tb_.shape[-1]-1, 0, -1):
            opt_st = optst_ta.read(sym)[:,None]

            idx_ = tf.concat([tf.range(tf.shape(cm_)[0])[:,None], opt_st],
                             axis=1)
            opt_st_tminus1 = tf.gather_nd(tb_[:, :, sym], idx_)

            optst_ta = optst_ta.write(sym-1, opt_st_tminus1)

        return tf.transpose(optst_ta.stack())

    def _bmcalc(self, y):
        """
        Calculate branch metrics for a given noisy codeword tensor.
        For each time period t, _bmcalc computes the distance of symbol
        vector y[t] from each possible output symbol.
        The distance metric is L2 distance if decoder parameter `method` is
        "soft".

        The distance metric is L1 distance if parameter `method` is "hard".
        """

        op_bits = np.stack(
                    [int2bin(op, self._conv_n) for op in range(self._no)])
        op_mat = tf.cast(tf.tile(op_bits, [1,self._num_syms]), tf.float32)
        op_mat = tf.expand_dims(op_mat, axis=0)
        y = tf.expand_dims(y, axis=1)
        if self._method=='soft_llr':
            op_mat_sign = 1 - 2.*op_mat
            llr_sign = -1. * tf.math.multiply(y, op_mat_sign)
            llr_sign = tf.reshape(llr_sign,
                (-1, self._no, self._num_syms, self._conv_n))
            # Sum of LLR*(sign of bit) for each symbol
            bm = tf.math.reduce_sum(llr_sign, axis=-1)

        elif self._method == 'hard':
            diffabs = tf.math.abs(y-op_mat)
            diffabs = tf.reshape(diffabs,
                                 (-1, self._no, self._num_syms, self._conv_n))
            # Manhattan distance of symbols
            bm = tf.math.reduce_sum(diffabs, axis=-1)

        return bm

    #########################
    # Keras layer functions
    #########################

    def build(self, input_shape):
        """Build layer and check dimensions."""
        # assert rank must be two
        tf.debugging.assert_greater_equal(len(input_shape), 2)

        self._n = input_shape[-1]
        self._k = int(self._n*self.coderate)

        divisible = tf.math.floormod(self._n, self._conv_n)
        assert divisible==0, 'length of codeword should be divisible by \
            number of output bits per symbol.'

        self._num_syms = int(self._n/self._conv_n)

    def call(self, inputs):
        """
        Viterbi decoding function.

        inputs is the (noisy) codeword tensor where the last dimension should
        equal n. All the leading dimensions are assumed as batch dimensions.

        """
        LARGEDIST = 2.**20 # pylint: disable=invalid-name

        tf.debugging.assert_type(inputs, tf.float32,
                                 message="input must be tf.float32.")
        if self._method == 'hard':
            inputs = tf.math.floormod(tf.cast(inputs, tf.int32),2)
        elif self._method == 'soft_llr':
            inputs = -1. * inputs

        inputs = tf.cast(inputs, tf.float32)

        output_shape = inputs.get_shape().as_list()
        y_resh = tf.reshape(inputs, [-1, self._n])

        output_shape[0] = -1
        if self._return_info_bits:
            output_shape[-1] = self._k # assign k to the last dimension
        else:
            output_shape[-1] = self._n
        # Branch metrics matrix for a given y
        bm_mat = self._bmcalc(y_resh)

        cm_ta = tf.TensorArray(tf.float32,
                               size=self._num_syms,
                               dynamic_size=False,
                               clear_after_read=False)
        tb_ta = tf.TensorArray(tf.int32,
                               size=self._num_syms,
                               dynamic_size=False,
                               clear_after_read=False)

        prev_cm_np = np.full((self._ns,), LARGEDIST)
        prev_cm_np[0] = 0.0
        prev_cm_ = tf.convert_to_tensor(prev_cm_np, dtype=tf.float32)

        prev_cm = tf.tile(prev_cm_[None,:], [tf.shape(y_resh)[0], 1])

        for idx in tf.range(0, self._n, self._conv_n):
            sym = idx//self._conv_n

            cum_t, tb_states = self._update(prev_cm, bm_mat[..., sym])
            cm_ta = cm_ta.write(sym, cum_t)
            tb_ta = tb_ta.write(sym, tb_states)

            prev_cm = cum_t
        cm = tf.transpose(cm_ta.stack(), perm=[1,2,0])
        tb = tf.transpose(tb_ta.stack(),perm=[1,2,0])
        del cm_ta, tb_ta

        opt_path = self._optimal_path(cm, tb)
        opt_path = tf.concat(
            (tf.zeros((tf.shape(cm)[0], 1), tf.int32),
             opt_path), axis=1)
        del cm, tb
        msghat, cwhat = self._op_bits_path(opt_path)

        if self._return_info_bits:
            output = tf.cast(msghat, self.output_dtype)
        else:
            output = tf.cast(cwhat, self.output_dtype)
        output_reshaped = tf.reshape(output, output_shape)

        return output_reshaped


class BCJRDecoder(Layer):
    # pylint: disable=line-too-long
    """BCJRDecoder(gen_poly=None, rate=1/2, constraint_length=3,, rsc=False, terminate=False, hard_out=True, output_dtype=tf.float32, **kwargs)

    Implements the BCJR decoding algorithm [BCJR]_ that returns an
    estimate of the information bits for a noisy convolutional codeword.
    Takes as input either channel LLRs or a tuple of
    (channel LLRs, a priori LLRs). Returns an estimate of the information bits, either the as LLRs (if ``hard_out`` =False) or hard decoded
    bits (if ``hard_out`` =True), respectively.

    The class inherits from the Keras layer class and can be used as layer in
    a Keras model.

    Parameters
    ----------
    gen_poly: tuple
        tuple of strings with each string being a 0, 1 sequence. If `None`,
        ``rate`` and ``constraint_length`` must be provided.

    rate: float
        Valid values are 1/3 and 0.5. Only required if ``gen_poly`` is `None`.

    constraint_length: int
        Valid values are between 3 and 8 inclusive. Only required if
        ``gen_poly`` is `None`.

    rsc: boolean
        Boolean flag indicating whether the encoder is recursive-systematic for
        given generator polynomials.
        `"True"` indicates encoder is recursive-systematic.
        `"False"` indicates encoder is feed-forward non-systematic.

    terminate: boolean
        Boolean flag indicating whether the codeword is terminated.
        `"True"` indicates codeword is terminated to all-zero state.
        `"False"` indicates codeword is not terminated

    hard_out: boolean
        Boolean flag indicating whether to output hard or soft decisions on
        the decoded information vector. `"True"` implies a hard-decoded
        information vector of 0/1's as output. `"False"` implies output is
        decoded LLR's of the information.

    output_dtype: tf.DType
        Defaults to tf.float32. Defines the output datatype of the layer.

    Input
    -----
    llr_ch or (llr_ch, llr_a) :
        Tensor or Tuple:

    llr_ch: [...,n], tf.float32
        2+D tensor containing the (noisy) channel
        LLRs where `n` denotes the codeword length.

    llr_a: [...,k], tf.float32
        2+D tensor containing the a priori information of each information bit.
        Implicitly assumed to be 0 if only ``llr_ch`` is provided.

    Output
    ------
    : tf.float32
        2+D tensor of shape `[...,rate*n]` containing the estimates of the
        information bit tensor.

    Note
    ----
    A full implementation of the decoder rather than a windowed approach
    is used. For a given codeword of duration `T`, the path metric is
    computed from time `0` to `T` and the path with optimal metric at time
    `T` is selected. The optimal path is then traced back from `T` to `0`
    to output the estimate of the information bit vector used to encode.
    For larger codewords, note that the current method is sub-optimal
    in terms of memory utilization and latency.
    """

    def __init__(self,
                 gen_poly=None,
                 rate=1/2,
                 constraint_length=3,
                 rsc=False,
                 terminate=False,
                 hard_out=True,
                 output_dtype=tf.float32,
                 **kwargs):

        super().__init__(**kwargs)

        if gen_poly is not None:
            assert all(isinstance(poly, str) for poly in gen_poly), \
                "Each polynomial must be a string."
            assert all(len(poly)==len(gen_poly[0]) for poly in gen_poly), \
                "Each polynomial must be of same length."
            assert all(all(
                char in ['0','1'] for char in poly) for poly in gen_poly),\
                "Each polynomial must be a string of 0's and 1's."
            self._gen_poly = gen_poly
        else:
            valid_rates = (1/2, 1/3)
            valid_constraint_length = (3, 4, 5, 6, 7, 8)

            assert constraint_length in valid_constraint_length, \
                "Constraint length must be between 3 and 8."
            assert rate in valid_rates, \
                "Rate must be 1/3 or 1/2."
            self._gen_poly = polynomial_selector(rate, constraint_length)

        # init Trellis parameters
        self._trellis = Trellis(self.gen_poly, rsc=rsc)
        self._coderate = 1/len(self._gen_poly)
        self._mu = len(self._gen_poly[0])-1

        self._terminate = terminate
        self._num_term_bits = None
        self._num_term_syms = None

        # conv_k denotes number of input bit streams
        # can only be 1 in current implementation
        self._conv_k = self._trellis.conv_k
        assert self._conv_k ==  1
        self._mu = self._trellis._mu
        # conv_n denotes number of output bits for conv_k input bits
        self._conv_n = self._trellis.conv_n

        # Length of Info-bit vector. Equal to _num_syms if terminate=False,
        # else < _num_syms
        self._k = None
        # Length of Turbo codeword, including termination bits
        self._n = None
        # num_syms denote number of encoding periods or state transitions.
        self._num_syms = None

        self._ni = 2**self._conv_k
        self._no = 2**self._conv_n
        self._ns = self._trellis.ns

        self._hard_out = hard_out
        self._output_dtype = output_dtype
        self.ipst_op_idx, self.ipst_ip_idx = self._mask_by_tonode()

    #########################################
    # Public methods and properties
    #########################################

    @property
    def gen_poly(self):
        """The generator polynomial used by the encoder."""
        return self._gen_poly

    @property
    def coderate(self):
        """Rate of the code used in the encoder."""
        return self._coderate

    @property
    def trellis(self):
        """Trellis object used during encoding."""
        return self._trellis

    #########################
    # Utility functions
    #########################

    def _mask_by_tonode(self):
        """
        Assume i->j a valid state transition given info-bit b & emits symbol k
        returns following two _ns x _no matrices, each element of shape (2,).
        - st_op_idx: jth row contains (i,k) tuples
        - st_ip_idx: jth row contains (i,b) tuples

        When applied as tf.gather on a _ns x _no matrix, the output is
        matrix sorted by next_state.

        For e.g., tf.gather when applied on "input" (shape _ns x _no), with mask
        - st_op_idx: gathers input[i][k] in row j,
        - st_ip_idx: gathers input[i][b] in row j.
        """

        cnst = self._ns * self._ni
        from_nodes_vec = tf.reshape(self._trellis.from_nodes,(cnst,))
        op_idx = tf.reshape(self._trellis.op_by_tonode, (cnst,))
        st_op_idx = tf.transpose(tf.stack([from_nodes_vec, op_idx]))
        st_op_idx = tf.reshape(st_op_idx[None,:,:],(self._ns, self._ni, 2))

        ip_idx = tf.reshape(self._trellis.ip_by_tonode, (cnst,))
        st_ip_idx = tf.transpose(tf.stack([from_nodes_vec, ip_idx]))
        st_ip_idx = tf.reshape(st_ip_idx[None,:,:],(self._ns, self._ni, 2))

        return st_op_idx, st_ip_idx

    def _bmcalc(self, llr_in):
        """
        Calculate branch gamma metrics for a given noisy codeword tensor.
        For each time period t, _bmcalc computes the "distance" of symbol
        vector y[t] from each possible output symbol i.e.,
        (2*Eb/N0)* sum_i x_y*y_i for i=1,2,...,conv_n

        The above metric is used in calculation of gamma.
        If the input is llr, which is nothing but 2*Eb*y/N0.
        """
        op_bits = np.stack(
                    [int2bin(op, self._conv_n) for op in range(self._no)])
        op_mat = tf.cast(tf.tile(op_bits, [1, self._num_syms]), tf.float32)
        op_mat = tf.expand_dims(op_mat, axis=0)
        llr_in = tf.expand_dims(llr_in, axis=1)
        op_mat_sign = 1. - 2. * op_mat

        llr_sign = tf.math.multiply(llr_in, op_mat_sign)
        half_llr_sign = tf.reshape(0.5 * llr_sign,
            (-1, self._no, self._num_syms, self._conv_n))
        bm = tf.math.exp(tf.math.reduce_sum(half_llr_sign, axis=-1))

        return bm

    def _update_fwd(self, alph_init, bm_mat, llr):
        """
        Run forward update from time t=0 to t=k-1.
        At each time t, computes alpha_t using alpha_t-1 and gamma_t.

        Returns tensor array of alpha_t, t-0,1,2...,k-1
        """
        alph_ta = tf.TensorArray(tf.float32, size=self._num_syms+1,
                                  dynamic_size=False, clear_after_read=False)

        alph_prev = alph_init
        alph_prev = tf.cast(alph_prev, tf.float32)

        # (bs, _Ns, _ni, 2) matrix
        ipst_ip_mask = tf.tile(
            self.ipst_ip_idx[None,:],[tf.shape(alph_init)[0],1,1,1])
        # (bs, _Ns, _ni) matricx, by from state
        op_mask = tf.tile(self.trellis.op_by_fromnode[None,:,:],
                          [tf.shape(alph_init)[0],1,1])
        ipbit_mat = tf.tile(tf.range(self._ni)[None, None, :],
                            [tf.shape(alph_init)[0], self._ns, 1])
        ipbitsign_mat = 1.0 - 2.0*tf.cast(ipbit_mat, tf.float32)
        alph_ta = alph_ta.write(0, alph_prev)
        for t in tf.range(0, self._num_syms):
            bm_t = bm_mat[..., t]
            llr_t = 0.5 * llr[...,t][:, None,None]

            bm_byfromst = tf.gather(bm_t, op_mask, batch_dims=1)
            llr_byfromst = tf.math.exp(tf.math.multiply(
                tf.tile(llr_t,[1, self._ns, self._ni]), ipbitsign_mat))
            gamma_byfromst = tf.multiply(llr_byfromst, bm_byfromst)

            alph_gam_prod = tf.math.multiply(gamma_byfromst,
                                             alph_prev[:,:,None])
            alphgam_bytost = tf.gather_nd(alph_gam_prod,
                                          ipst_ip_mask,
                                          batch_dims=1)
            alph_t = tf.math.reduce_sum(alphgam_bytost, axis=-1)
            alph_t_sum = tf.reduce_sum(alph_t, axis=-1)
            alph_t = tf.divide(alph_t, tf.tile(alph_t_sum[:,None],[1,self._ns]))

            alph_prev = alph_t
            alph_ta = alph_ta.write(t+1, alph_t)
        return alph_ta

    def _update_bwd(self, beta_init, bm_mat, llr, alpha_ta):
        """
        Run backward update from time t=k-1 to t=0.
        At each time t, computes beta_t-1 using beta_t and gamma_t.

        Returns llr for information bits for t=0,1,...,k-1
        """

        beta_next = beta_init
        llr_op_ta = tf.TensorArray(tf.float32,
                                  size=self._num_syms,
                                  dynamic_size=False,
                                  clear_after_read=False)
        beta_next = tf.cast(beta_next, tf.float32)

        # (bs, _Ns, _ni) matrix, by from state
        op_mask = tf.tile(self.trellis.op_by_fromnode[None,:,:],
                          [tf.shape(beta_init)[0],1,1])
        tonode_mask = tf.tile(self.trellis.to_nodes[None,:,:],
                              [tf.shape(beta_init)[0], 1, 1])

        ipbit_mat = tf.tile(tf.range(self._ni)[None, None, :],
                            [tf.shape(beta_init)[0], self._ns, 1])
        ipbitsign_mat = 1.0 - 2.0 * tf.cast(ipbit_mat, tf.float32)

        for t in tf.range(self._num_syms-1, -1, -1):
            bm_t = bm_mat[..., t]
            llr_t = 0.5 * llr[...,t][:, None,None]
            bm_byfromst = tf.gather(bm_t, op_mask, batch_dims=1)
            llr_byfromst = tf.math.exp(tf.math.multiply(
                tf.tile(llr_t,[1, self._ns, self._ni]), ipbitsign_mat))
            gamma_byfromst = tf.multiply(bm_byfromst, llr_byfromst)

            beta_bytonode = tf.gather(beta_next, tonode_mask, batch_dims=1)
            beta_gam_prod = tf.math.multiply(gamma_byfromst, beta_bytonode)
            beta_t = tf.math.reduce_sum(beta_gam_prod, axis=-1)
            beta_t_sum = tf.reduce_sum(beta_t, axis=-1)
            beta_t = tf.divide(beta_t, tf.tile(beta_t_sum[:,None],[1,self._ns]))

            alph_t = alpha_ta.read(t)
            llr_op_t0 = tf.math.multiply(
                            tf.math.multiply(alph_t, gamma_byfromst[...,0]),
                                         beta_bytonode[...,0])
            llr_op_t1 = tf.math.multiply(
                            tf.math.multiply(alph_t,gamma_byfromst[...,1]),
                                         beta_bytonode[...,1])
            llr_op_t = tf.math.log(tf.divide(tf.reduce_sum(llr_op_t0, axis=-1),
                                             tf.reduce_sum(llr_op_t1,axis=-1)))
            llr_op_ta = llr_op_ta.write(t, llr_op_t)

            beta_next = beta_t

        llr_op = tf.transpose(llr_op_ta.stack())
        return llr_op

    #########################
    # Keras layer functions
    #########################

    def build(self, input_shape):
        """Build layer and check dimensions."""
        # assert rank must be two
        tf.debugging.assert_greater_equal(len(input_shape), 2)

        if isinstance(input_shape, tf.TensorShape):
            self._n = input_shape[-1]
        else:
            self._n = input_shape[0][-1]

        self._num_syms = int(self._n*self.coderate)
        if self._terminate:
            self._num_term_syms = self._mu
            self._num_term_bits = self._num_term_syms * 2
        else:
            self._num_term_syms = 0
            self._num_term_bits = 0

        self._k = self._num_syms - self._num_term_syms

    def call(self, inputs):
        """
        BCJR decoding function.
        inputs is the (noisy) codeword tensor where the last dimension should
        equal n. All the leading dimensions are assumed as batch dimensions.
        """
        if isinstance(inputs, (tuple, list)):
            assert(len(inputs)) == 2
            llr_ch, llr_apr = inputs
        else:
            tf.debugging.assert_greater(tf.rank(inputs), 1)
            llr_ch = inputs
            llr_apr = None

        tf.debugging.assert_type(llr_ch,
                                 tf.float32,
                                 message="input must be tf.float32.")

        output_shape = llr_ch.get_shape().as_list()

        # allow different codeword lenghts in eager mode
        if output_shape[-1] != self._n:
            if isinstance(inputs, (tuple, list)):
                self.build((inputs[0].get_shape(),
                            inputs[1].get_shape()))
            else:
                self.build(llr_ch.get_shape().as_list())

        output_shape[0] = -1
        output_shape[-1] = self._k # assign k to the last dimension
        llr_ch = tf.reshape(llr_ch, [-1, self._n])

        if llr_apr is None:
            llr_apr = tf.zeros((tf.shape(llr_ch)[0], self._num_syms),
                               dtype=tf.float32)
        llr_ch = -1. * llr_ch
        llr_apr = -1. * llr_apr
        # Branch metrics matrix for a given y
        bm_mat = self._bmcalc(llr_ch)

        alpha_prev_np = np.full((self._ns,), 0.0)
        alpha_prev_np[0] = 1.0
        alpha_init = tf.convert_to_tensor(alpha_prev_np, dtype=tf.float32)
        alpha_init = tf.tile(alpha_init[None,:], [tf.shape(llr_ch)[0], 1])
        if self._terminate:
            beta_init = alpha_init
        else:
            eq_prob = 1./self._ns
            beta_init = tf.convert_to_tensor(np.full((self._ns,), eq_prob),
                                             dtype=tf.float32)
            beta_init = tf.tile(beta_init[None,:], [tf.shape(llr_ch)[0], 1])

        alph_ta = self._update_fwd(alpha_init, bm_mat, llr_apr)
        llr_op = self._update_bwd(beta_init, bm_mat, llr_apr, alph_ta)

        msghat = -1. * llr_op[...,:self._k]
        if self._hard_out: # hard decide decoder output if required
            msghat = tf.less(0.0, msghat)
        msghat = tf.cast(msghat, self._output_dtype)
        msghat_reshaped = tf.reshape(msghat, output_shape)

        return msghat_reshaped
