#
# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0#
"""Blocks for Convolutional Code Viterbi Decoding."""

import numpy as np
import tensorflow as tf
from sionna.phy import Block
from sionna.phy.fec.utils import int2bin, int_mod_2
from sionna.phy.fec.conv.utils import polynomial_selector, Trellis


class ViterbiDecoder(Block):
    # pylint: disable=line-too-long
    r"""Applies Viterbi decoding to a sequence of noisy codeword bits

    Implements the Viterbi decoding algorithm [Viterbi]_ that returns an
    estimate of the information bits for a noisy convolutional codeword.
    Takes as input either LLR values (`method` = `soft_llr`) or hard bit values
    (`method` = `hard`) and returns a hard decided estimation of the information
    bits.

    Parameters
    ----------
    encoder: :class:`~sionna.phy.fec.conv.encoding.ConvEncoder`
        If ``encoder`` is provided as input, the following input parameters
        are not required and will be ignored: ``gen_poly``, ``rate``,
        ``constraint_length``, ``rsc``, ``terminate``. They will be inferred
        from the ``encoder``  object itself. If ``encoder`` is `None`, the
        above parameters must be provided explicitly.

    gen_poly: tuple | None
        tuple of strings with each string being a 0, 1 sequence. If `None`,
        ``rate`` and ``constraint_length`` must be provided.

    rate: float, 1/2 | 1/3
        Valid values are 1/3 and 0.5. Only required if ``gen_poly`` is `None`.

    constraint_length: int, 3....8
        Valid values are between 3 and 8 inclusive. Only required if
        ``gen_poly`` is `None`.

    rsc: `bool`, (default `False`)
        Boolean flag indicating whether the encoder is recursive-systematic for
        given generator polynomials.
        `True` indicates encoder is recursive-systematic.
        `False` indicates encoder is feed-forward non-systematic.

    terminate: `bool`, (default `False`)
        Boolean flag indicating whether the codeword is terminated.
        `True` indicates codeword is terminated to all-zero state.
        `False` indicates codeword is not terminated.

    method: str, `soft_llr` | `hard`
        Valid values are `soft_llr` or `hard`. In computing path
        metrics, `soft_llr` expects channel LLRs as input
        `hard` assumes a `binary symmetric channel` (BSC) with 0/1 values are
        inputs. In case of `hard`, `inputs` will be quantized to 0/1 values.

    return_info_bots: `bool`, (default `True`)
        Boolean flag indicating whether only the information bits or all
        codeword bits are returned.

    precision : `None` (default) | 'single' | 'double'
        Precision used for internal calculations and outputs.
        If set to `None`, :py:attr:`~sionna.phy.config.precision` is used.

    Input
    -----
    inputs: [...,n], tf.float
        Tensor containing the (noisy) channel output symbols where `n`
        denotes the codeword length

    Output
    ------
    : [...,rate*n], tf.float
        Binary tensor containing the estimates of the information bit tensor

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
                 *,
                 encoder=None,
                 gen_poly=None,
                 rate=1/2,
                 constraint_length=3,
                 rsc=False,
                 terminate=False,
                 method='soft_llr',
                 return_info_bits=True,
                 precision=None,
                 **kwargs):

        super().__init__(precision=precision, **kwargs)

        if encoder is not None:
            self._gen_poly = encoder.gen_poly
            self._trellis = encoder.trellis
            self._terminate = encoder.terminate
        else:
            valid_rates = (1/2, 1/3)
            valid_constraint_length = (3, 4, 5, 6, 7, 8)

            if gen_poly is not None:
                if not all(isinstance(poly, str) for poly in gen_poly):
                    raise TypeError("Each polynomial must be a string.")
                if not all(len(poly)==len(gen_poly[0]) for poly in gen_poly):
                    raise ValueError("Each polynomial must be of same length.")
                if not all(all(
                char in ['0','1'] for char in poly) for poly in gen_poly):
                    msg = "Each polynomial must be a string of 0's and 1's."
                    raise ValueError(msg)
                self._gen_poly = gen_poly
            else:
                valid_rates = (1/2, 1/3)
                valid_constraint_length = (3, 4, 5, 6, 7, 8)

                if constraint_length not in valid_constraint_length:
                    msg = "Constraint length must be between 3 and 8."
                    raise ValueError(msg)
                if rate not in valid_rates:
                    raise ValueError("Rate must be 1/3 or 1/2.")
                self._gen_poly = polynomial_selector(rate, constraint_length)

            # init Trellis parameters
            self._trellis = Trellis(self.gen_poly, rsc=rsc)
            self._terminate = terminate

        self._coderate_desired = 1/len(self.gen_poly)
        self._mu = len(self._gen_poly[0])-1
        if method not in ('soft_llr', 'hard'):
            raise ValueError("method must be `soft_llr` or `hard`.")

        # conv_k denotes number of input bit streams
        # can only be 1 in current implementation
        self._conv_k = self._trellis.conv_k

        # conv_n denotes number of output bits for conv_k input bits
        self._conv_n = self._trellis.conv_n

        # for conv codes, the code dimensions are unknown during initialization
        self._k = None
        self._n = None
        # num_syms denote number of encoding periods or state transitions.
        self._num_syms = None

        self._ni = 2**self._conv_k
        self._no = 2**self._conv_n
        self._ns = self._trellis.ns

        self._method = method
        self._return_info_bits = return_info_bits
        # If i->j state transition emits symbol k, tf.gather with ipst_op_idx
        # gathers (i,k) element from input in row j.
        self.ipst_op_idx = self._mask_by_tonode()

    #########################################
    # Public methods and properties
    #########################################

    @property
    def gen_poly(self):
        """Generator polynomial used by the encoder"""
        return self._gen_poly

    @property
    def coderate(self):
        """Rate of the code used in the encoder"""
        if self.terminate and self._n is None:
            print("Note that, due to termination, the true coderate is lower "\
                  "than the returned design rate. "\
                  "The exact true rate is dependent on the value of n and "\
                  "hence cannot be computed before the first call().")
            self._coderate = self._coderate_desired
        elif self.terminate and self._n is not None:
            k = self._coderate_desired*self._n - self._mu
            self._coderate = k/self._n
        return self._coderate

    @property
    def trellis(self):
        """Trellis object used during encoding"""
        return self._trellis

    @property
    def terminate(self):
        """Indicates if the encoder is terminated during codeword generation"""
        return self._terminate

    @property
    def k(self):
        """Number of information bits per codeword"""
        if self._k is None:
            print("Note: The value of k cannot be computed before the first " \
                  "call().")
        return self._k

    @property
    def n(self):
        """Number of codeword bits"""
        if self._n is None:
            print("Note: The value of n cannot be computed before the first " \
                  "call().")
        return self._n

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

    def _update_fwd(self, init_cm, bm_mat):
        state_vec = tf.tile(tf.range(self._ns, dtype=tf.int32)[None,:],
                            [tf.shape(init_cm)[0], 1])
        ipst_op_mask = tf.tile(self.ipst_op_idx[None,:],
                               [tf.shape(init_cm)[0], 1, 1, 1])

        cm_ta = tf.TensorArray(self.rdtype, size=self._num_syms,
                               dynamic_size=False, clear_after_read=False)
        tb_ta = tf.TensorArray(tf.int32, size=self._num_syms,
                               dynamic_size=False, clear_after_read=False)

        prev_cm = init_cm
        for idx in tf.range(0, self._n, self._conv_n):
            sym = idx//self._conv_n
            metrics_t = bm_mat[..., sym]
            # Ns x No matrix- (s,j) is path_metric at state s with transition
            # op=j
            sum_metric = prev_cm[:,:,None] + metrics_t[:,None,:]
            sum_metric_bytonode = tf.gather_nd(sum_metric, ipst_op_mask,
                                               batch_dims=1)

            tb_state_idx = tf.math.argmin(sum_metric_bytonode, axis=2)
            tb_state_idx = tf.cast(tb_state_idx, tf.int32)

            # Transition to states argmin state index
            from_st_idx = tf.transpose(tf.stack([state_vec, tb_state_idx]),
                                       perm=[1, 2, 0])

            tb_states = tf.gather_nd(self._trellis.from_nodes, from_st_idx)
            cum_t = tf.math.reduce_min(sum_metric_bytonode, axis=2)

            cm_ta = cm_ta.write(sym, cum_t)
            tb_ta = tb_ta.write(sym, tb_states)

            prev_cm = cum_t

        return cm_ta, tb_ta


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
        # tb and ca are of shape (batch x self._ns x num_syms)
        assert(tb_.get_shape()[1] == self._ns), "Invalid shape."
        optst_ta = tf.TensorArray(tf.int32, size=tb_.shape[-1],
                                  dynamic_size=False,
                                  clear_after_read=False)
        if self._terminate:
            opt_term_state = tf.zeros((tf.shape(cm_)[0],), tf.int32)
        else:
            opt_term_state =tf.cast(tf.argmin(cm_[:, :, -1], axis=1), tf.int32)
        optst_ta = optst_ta.write(tb_.shape[-1]-1,opt_term_state)

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
        op_mat = tf.cast(tf.tile(op_bits, [1,self._num_syms]), self.rdtype)
        op_mat = tf.expand_dims(op_mat, axis=0)
        y = tf.expand_dims(y, axis=1)
        if self._method=='soft_llr':
            op_mat_sign = 1 - 2.*op_mat
            llr_sign = -1. * tf.math.multiply(y, op_mat_sign)
            llr_sign = tf.reshape(llr_sign,
                (-1, self._no, self._num_syms, self._conv_n))
            # Sum of LLR*(sign of bit) for each symbol
            bm = tf.math.reduce_sum(llr_sign, axis=-1)

        else: # method == 'hard'
            diffabs = tf.math.abs(y-op_mat)
            diffabs = tf.reshape(diffabs,
                                 (-1, self._no, self._num_syms, self._conv_n))
            # Manhattan distance of symbols
            bm = tf.math.reduce_sum(diffabs, axis=-1)

        return bm

    ########################
    # Sionna Block functions
    ########################

    def build(self, input_shape):
        """Build block and check dimensions."""

        self._n = input_shape[-1]

        divisible = tf.math.floormod(self._n, self._conv_n)
        if divisible!=0:
            raise ValueError('Length of codeword should be divisible by \
            number of output bits per symbol.')

        self._num_syms = int(self._n*self._coderate_desired)

        self._num_term_syms = self._mu if self.terminate else 0
        self._k = self._num_syms - self._num_term_syms

    def call(self, inputs, /):
        """
        Viterbi decoding function.

        inputs is the (noisy) codeword tensor where the last dimension should
        equal n. All the leading dimensions are assumed as batch dimensions.

        """
        LARGEDIST = 2.**20 # pylint: disable=invalid-name

        if self._method == 'hard':
            # Ensure binary values
            inputs = int_mod_2(inputs)
        elif self._method == 'soft_llr':
            inputs = -1. * inputs

        output_shape = inputs.get_shape().as_list()
        y_resh = tf.reshape(inputs, [-1, self._n])
        output_shape[0] = -1
        if self._return_info_bits:
            output_shape[-1] = self._k # assign k to the last dimension
        else:
            output_shape[-1] = self._n

        # Branch metrics matrix for a given y
        bm_mat = self._bmcalc(y_resh)

        init_cm_np = np.full((self._ns,), LARGEDIST)
        init_cm_np[0] = 0.0
        prev_cm_ = tf.convert_to_tensor(init_cm_np, dtype=self.rdtype)
        prev_cm = tf.tile(prev_cm_[None,:], [tf.shape(y_resh)[0], 1])

        cm_ta, tb_ta = self._update_fwd(prev_cm, bm_mat)

        cm = tf.transpose(cm_ta.stack(), perm=[1,2,0])
        tb = tf.transpose(tb_ta.stack(),perm=[1,2,0])
        del cm_ta, tb_ta

        zero_st = tf.zeros((tf.shape(y_resh)[0], 1), tf.int32)
        opt_path = self._optimal_path(cm, tb)
        opt_path = tf.concat((zero_st, opt_path), axis=1)
        del cm, tb
        msghat, cwhat = self._op_bits_path(opt_path)
        if self._return_info_bits:
            msghat = msghat[...,:self._k]
            output = tf.cast(msghat, self.rdtype)
        else:
            output = tf.cast(cwhat, self.rdtype)
        output_reshaped = tf.reshape(output, output_shape)

        return output_reshaped


class BCJRDecoder(Block):
    # pylint: disable=line-too-long
    r"""Applies BCJR decoding to a sequence of noisy codeword bits

    Implements the BCJR decoding algorithm [BCJR]_ that returns an
    estimate of the information bits for a noisy convolutional codeword.
    Takes as input either channel LLRs  and a priori LLRs (optional).
    Returns an estimate of the information bits, either output LLRs (
    ``hard_out`` = `False`) or hard decoded bits ( ``hard_out`` = `True`),
    respectively.

    Parameters
    ----------
    encoder: :class:`~sionna.phy.fec.conv.encoding.ConvEncoder`
        If ``encoder`` is provided as input, the following input parameters
        are not required and will be ignored: ``gen_poly``, ``rate``,
        ``constraint_length``, ``rsc``, ``terminate``. They will be inferred
        from the ``encoder``  object itself. If ``encoder`` is `None`, the
        above parameters must be provided explicitly.

    gen_poly: tuple | None
        tuple of strings with each string being a 0, 1 sequence. If `None`,
        ``rate`` and ``constraint_length`` must be provided.

    rate: float, None (default) | 1/3 | 1/2
        Valid values are 1/3 and 1/2. Only required if ``gen_poly`` is `None`.

    constraint_length: int, 3...8
        Valid values are between 3 and 8 inclusive. Only required if
        ``gen_poly`` is `None`.

    rsc: `bool`, (default `False`)
        Boolean flag indicating whether the encoder is recursive-systematic for
        given generator polynomials. `True` indicates encoder is
        recursive-systematic. `False` indicates encoder is feed-forward
        non-systematic.

    terminate: `bool`, (default `False`)
        Boolean flag indicating whether the codeword is terminated.
        `True` indicates codeword is terminated to all-zero state.
        `False` indicates codeword is not terminated.

    hard_out: `bool`, (default `True`)
        Boolean flag indicating whether to output hard or soft decisions on
        the decoded information vector.
        `True` implies a hard-decoded information vector of 0/1's as output.
        `False` implies output is decoded LLR's of the information.

    algorithm: str, `map` (default) | `log` | `maxlog`
        Indicates the implemented BCJR algorithm,
        where `map` denotes the exact MAP algorithm, `log` indicates the
        exact MAP implementation, but in log-domain, and
        `maxlog` indicates the approximated MAP implementation in log-domain,
        where :math:`\log(e^{a}+e^{b}) \sim \max(a,b)`.

    precision : `None` (default) | 'single' | 'double'
        Precision used for internal calculations and outputs.
        If set to `None`, :py:attr:`~sionna.phy.config.precision` is used.

    Input
    -----
    llr_ch: [...,n], tf.float
        Tensor containing the (noisy) channel
        LLRs, where `n` denotes the codeword length

    llr_a: [...,k], None (default) | tf.float
        Tensor containing the a priori information of each information bit.
        Implicitly assumed to be 0 if only ``llr_ch`` is provided.

    Output
    ------
    : tf.float
        Tensor of shape `[...,coderate*n]` containing the estimates of the
        information bit tensor

    """

    def __init__(self,
                 encoder=None,
                 gen_poly=None,
                 rate=1/2,
                 constraint_length=3,
                 rsc=False,
                 terminate=False,
                 hard_out=True,
                 algorithm='map',
                 precision=None,
                 **kwargs):

        super().__init__(precision=precision, **kwargs)

        if encoder is not None:
            self._gen_poly = encoder.gen_poly
            self._trellis = encoder.trellis
            self._terminate = encoder.terminate
        else:
            if gen_poly is not None:
                if not all(isinstance(poly, str) for poly in gen_poly):
                    raise TypeError("Each polynomial must be a string.")
                if not all(len(poly)==len(gen_poly[0]) for poly in gen_poly):
                    raise ValueError("Each polynomial must be of same length.")
                if not all(all(
                    char in ['0','1'] for char in poly) for poly in gen_poly):
                    msg = "Each polynomial must be a string of 0's and 1's."
                    raise ValueError(msg)
                self._gen_poly = gen_poly
            else:
                valid_rates = (1/2, 1/3)
                valid_constraint_length = (3, 4, 5, 6, 7, 8)

                if constraint_length not in valid_constraint_length:
                    msg = "Constraint length must be between 3 and 8."
                    raise ValueError(msg)
                if rate not in valid_rates:
                    raise ValueError("Rate must be 1/3 or 1/2.")
                self._gen_poly = polynomial_selector(rate, constraint_length)

                # init Trellis parameters
            self._trellis = Trellis(self.gen_poly, rsc=rsc)
            self._terminate = terminate

        valid_algorithms = ['map', 'log', 'maxlog']
        if algorithm not in valid_algorithms:
            raise ValueError("algorithm must be one of map, log or maxlog")

        self._coderate_desired = 1/len(self._gen_poly)
        self._mu = len(self._gen_poly[0])-1

        self._num_term_bits = None
        self._num_term_syms = None

        # conv_k denotes number of input bit streams
        # can only be 1 in current implementation
        self._conv_k = self._trellis.conv_k
        if self._conv_k!=1:
            raise NotImplementedError("Only conv_k=1 currently supported.")

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
        self._algorithm = algorithm

        self.ipst_op_idx, self.ipst_ip_idx = self._mask_by_tonode()

    #########################################
    # Public methods and properties
    #########################################

    @property
    def gen_poly(self):
        """Generator polynomial used by the encoder"""
        return self._gen_poly

    @property
    def coderate(self):
        """Rate of the code used in the encoder"""
        if self.terminate and self._n is None:
            print("Note that, due to termination, the true coderate is lower "\
                  "than the returned design rate. "\
                  "The exact true rate is dependent on the value of n and "\
                  "hence cannot be computed before the first call().")
            self._coderate = self._coderate_desired
        elif self.terminate and self._n is not None:
            k = self._coderate_desired*self._n - self._mu
            self._coderate = k/self._n
        return self._coderate

    @property
    def trellis(self):
        """Trellis object used during encoding"""
        return self._trellis

    @property
    def terminate(self):
        """Indicates if the encoder is terminated during codeword generation"""
        return self._terminate

    @property
    def k(self):
        """Number of information bits per codeword"""
        if self._k is None:
            print("Note: The value of k cannot be computed before the first " \
                  "call().")
        return self._k

    @property
    def n(self):
        """Number of codeword bits"""
        if self._n is None:
            print("Note: The value of n cannot be computed before the first " \
                  "call().")
        return self._n

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
        op_mat = tf.cast(tf.tile(op_bits, [1, self._num_syms]), self.rdtype)
        op_mat = tf.expand_dims(op_mat, axis=0)
        llr_in = tf.expand_dims(llr_in, axis=1)
        op_mat_sign = 1. - 2. * op_mat

        llr_sign = tf.math.multiply(llr_in, op_mat_sign)
        half_llr_sign = tf.reshape(0.5 * llr_sign,
            (-1, self._no, self._num_syms, self._conv_n))

        if self._algorithm in ['log', 'maxlog']:
            bm = tf.math.reduce_sum(half_llr_sign, axis=-1)
        else:
            bm = tf.math.exp(tf.math.reduce_sum(half_llr_sign, axis=-1))

        return bm

    def _initialize(self, llr_ch):
        if self._algorithm in ['log', 'maxlog']:
            init_vals = -np.inf, 0.0
        else:
            init_vals = 0.0, 1.0
        alpha_init_np = np.full((self._ns,), init_vals[0])
        alpha_init_np[0] = init_vals[1]

        beta_init_np = alpha_init_np
        if not self._terminate:
            eq_prob = 1./self._ns
            if self._algorithm in ['log', 'maxlog']:
                eq_prob = np.log(eq_prob)
            beta_init_np = np.full((self._ns,), eq_prob)

        alpha_init = tf.convert_to_tensor(alpha_init_np, dtype=self.rdtype)
        alpha_init = tf.tile(alpha_init[None,:], [tf.shape(llr_ch)[0], 1])
        beta_init = tf.convert_to_tensor(beta_init_np, dtype=self.rdtype)
        beta_init = tf.tile(beta_init[None,:], [tf.shape(llr_ch)[0], 1])
        return alpha_init, beta_init

    def _update_fwd(self, alph_init, bm_mat, llr):
        """
        Run forward update from time t=0 to t=k-1.
        At each time t, computes alpha_t using alpha_t-1 and gamma_t.

        Returns tensor array of alpha_t, t-0,1,2...,k-1
        """
        alph_ta = tf.TensorArray(self.rdtype, size=self._num_syms+1,
                                  dynamic_size=False, clear_after_read=False)
        alph_prev = tf.cast(alph_init, self.rdtype)

        # (bs, _Ns, _ni, 2) matrix
        ipst_ip_mask = tf.tile(
            self.ipst_ip_idx[None,:],[tf.shape(alph_init)[0],1,1,1])
        # (bs, _Ns, _ni) matrix, by from state
        op_mask = tf.tile(self.trellis.op_by_fromnode[None,:,:],
                          [tf.shape(alph_init)[0],1,1])
        ipbit_mat = tf.tile(tf.range(self._ni)[None, None, :],
                            [tf.shape(alph_init)[0], self._ns, 1])
        ipbitsign_mat = 1. - 2. * tf.cast(ipbit_mat, self.rdtype)
        alph_ta = alph_ta.write(0, alph_prev)
        for t in tf.range(self._num_syms):
            bm_t = bm_mat[..., t]
            llr_t = 0.5 * llr[...,t][:, None,None]

            bm_byfromst = tf.gather(bm_t, op_mask, batch_dims=1)
            signed_half_llr = tf.math.multiply(
                tf.tile(llr_t, [1, self._ns, self._ni]), ipbitsign_mat)
            if self._algorithm in ['log', 'maxlog']:
                llr_byfromst = signed_half_llr
                gamma_byfromst = llr_byfromst + bm_byfromst
                alph_gam_prod = gamma_byfromst + alph_prev[:,:,None]
            else:
                llr_byfromst = tf.math.exp(signed_half_llr)
                gamma_byfromst = tf.multiply(llr_byfromst, bm_byfromst)
                alph_gam_prod = tf.math.multiply(gamma_byfromst,
                                             alph_prev[:,:,None])

            alphgam_bytost = tf.gather_nd(alph_gam_prod,
                                          ipst_ip_mask,
                                          batch_dims=1)
            if self._algorithm =='map':
                alph_t = tf.math.reduce_sum(alphgam_bytost, axis=-1)
                alph_t_sum = tf.reduce_sum(alph_t, axis=-1)
                alph_t = tf.divide(alph_t,
                                   tf.tile(alph_t_sum[:,None], [1,self._ns]))
            elif self._algorithm == 'log':
                alph_t = tf.math.reduce_logsumexp(alphgam_bytost, axis=-1)
            else:  # self._algorithm = 'maxlog'
                alph_t = tf.math.reduce_max(alphgam_bytost, axis=-1)

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
        llr_op_ta = tf.TensorArray(self.rdtype,
                                  size=self._num_syms,
                                  dynamic_size=False,
                                  clear_after_read=False)
        beta_next = tf.cast(beta_next, self.rdtype)

        # (bs, _Ns, _ni) matrix, by from state
        op_mask = tf.tile(self.trellis.op_by_fromnode[None,:,:],
                          [tf.shape(beta_init)[0],1,1])
        tonode_mask = tf.tile(self.trellis.to_nodes[None,:,:],
                              [tf.shape(beta_init)[0], 1, 1])

        ipbit_mat = tf.tile(tf.range(self._ni)[None, None, :],
                            [tf.shape(beta_init)[0], self._ns, 1])
        ipbitsign_mat = 1.0 - 2.0 * tf.cast(ipbit_mat, self.rdtype)

        for t in tf.range(self._num_syms-1, -1, -1):
            bm_t = bm_mat[..., t]
            llr_t = 0.5 * llr[...,t][:, None,None]
            signed_half_llr = tf.math.multiply(
                tf.tile(llr_t,[1, self._ns, self._ni]), ipbitsign_mat)
            bm_byfromst = tf.gather(bm_t, op_mask, batch_dims=1)

            if self._algorithm in ['log', 'maxlog']:
                llr_byfromst = signed_half_llr
                gamma_byfromst = tf.math.add(llr_byfromst, bm_byfromst)
            else:
                llr_byfromst = tf.math.exp(signed_half_llr)
                gamma_byfromst = tf.multiply(llr_byfromst, bm_byfromst)

            beta_bytonode = tf.gather(beta_next, tonode_mask, batch_dims=1)

            if self._algorithm not in ['log', 'maxlog']:
                beta_gam_prod = tf.math.multiply(gamma_byfromst, beta_bytonode)
                beta_t = tf.math.reduce_sum(beta_gam_prod, axis=-1)
                beta_t_sum = tf.reduce_sum(beta_t, axis=-1)
                beta_t = tf.divide(beta_t,
                                   tf.tile(beta_t_sum[:,None],[1,self._ns]))
            elif self._algorithm == 'log':
                beta_gam_prod = gamma_byfromst + beta_bytonode
                beta_t = tf.math.reduce_logsumexp(beta_gam_prod,
                                                  axis=-1, keepdims=False)
            else: #self._algorithm = 'maxlog'
                beta_gam_prod = gamma_byfromst + beta_bytonode
                beta_t = tf.math.reduce_max(beta_gam_prod, axis=-1)

            alph_t = alpha_ta.read(t)
            if self._algorithm not in ['log', 'maxlog']:
                llr_op_t0 = tf.math.multiply(
                                tf.math.multiply(alph_t, gamma_byfromst[...,0]),
                                            beta_bytonode[...,0])
                llr_op_t1 = tf.math.multiply(
                                tf.math.multiply(alph_t,gamma_byfromst[...,1]),
                                            beta_bytonode[...,1])
                llr_op_t = tf.math.log(tf.divide(
                                tf.reduce_sum(llr_op_t0, axis=-1),
                                tf.reduce_sum(llr_op_t1,axis=-1)))
            else:
                llr_op_t0 = alph_t + gamma_byfromst[...,0] +beta_bytonode[...,0]
                llr_op_t1 = alph_t + gamma_byfromst[...,1] +beta_bytonode[...,1]
                if self._algorithm == 'log':
                    llr_op_t = tf.math.subtract(
                        tf.math.reduce_logsumexp(llr_op_t0, axis=-1),
                        tf.math.reduce_logsumexp(llr_op_t1, axis=-1))
                else:
                    llr_op_t = tf.math.subtract(
                        tf.math.reduce_max(llr_op_t0, axis=-1),
                        tf.math.reduce_max(llr_op_t1, axis=-1))

            llr_op_ta = llr_op_ta.write(t, llr_op_t)
            beta_next = beta_t

        llr_op = tf.transpose(llr_op_ta.stack())
        return llr_op

    ########################
    # Sionna Block functions
    ########################

    # kwargs catchs additional llr_a input which is not required here
    # pylint: disable=unused-argument
    def build(self, input_shape, **kwargs):
        """Build block and check dimensions."""

        self._n = input_shape[-1]
        self._num_syms = int(self._n*self._coderate_desired)

        self._num_term_syms = self._mu if self._terminate else 0
        self._num_term_bits = int(self._num_term_syms/self._coderate_desired)

        self._k = self._num_syms - self._num_term_syms

    def call(self, llr_ch, /, *, llr_a=None):
        """
        BCJR decoding function.
        Inputs is the (noisy) codeword tensor where the last dimension should
        equal n. All the leading dimensions are assumed as batch dimensions.
        """

        output_shape = llr_ch.get_shape().as_list()

        # detect changes and triger rebuild if required (in Eager)

        # allow different codeword lengths in eager mode
        if output_shape[-1] != self._n:
            self.build(output_shape)

        output_shape[0] = -1
        output_shape[-1] = self._k # assign k to the last dimension
        llr_ch = tf.reshape(llr_ch, [-1, self._n])

        if llr_a is None:
            llr_a = tf.zeros((tf.shape(llr_ch)[0], self._num_syms),
                               dtype=self.rdtype)
        else:
            llr_a = tf.reshape(llr_a, [-1, self._num_syms])

        # internally, we use more common llr definition log(x)=p(x=0)/p(x=1)
        llr_ch = -1. * llr_ch
        llr_a = -1. * llr_a

        # Branch metrics matrix for a given y
        bm_mat = self._bmcalc(llr_ch)
        alpha_init, beta_init = self._initialize(llr_ch)

        alph_ta = self._update_fwd(alpha_init, bm_mat, llr_a)
        llr_op = self._update_bwd(beta_init, bm_mat, llr_a, alph_ta)

        # revert llr definition
        msghat = -1. * llr_op[...,:self._k]

        if self._hard_out: # hard decide decoder output if required
            msghat = tf.less(0.0, msghat)
            msghat = tf.cast(msghat, self.rdtype)
        msghat_reshaped = tf.reshape(msghat, output_shape)

        return msghat_reshaped
