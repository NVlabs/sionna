#
# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Layer for Convolutional Code Viterbi Decoding."""

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer
from sionna.fec.conv.utils import int2bin, polynomial_selector, Trellis

class ViterbiDecoder(Layer):
    # pylint: disable=line-too-long
    """ViterbiDecoder(gen_poly=None, rate=1/2, constraint_length=3, method='soft_llr', output_dtype=tf.float32, **kwargs)

    Decodes a noisy convolutional codeword to the information tensor.
    Takes as input either llr_values (`method` = `soft_llr`) or soft bit
    values centered on 0/1 (`method` = `soft`) or hard bit values
    (`method` = `hard`) and returns hard decision 0/1 bits, an estimate
    of information tensor.

    The class inherits from the Keras layer class and can be used as layer in
    a Keras model.

    Parameters
    ----------
        gen_poly: tuple
            tuple of strings with each string being a 0, 1 sequence. If `None`,
            ``rate`` and ``constraint_length`` must be provided.

        rate: float
            Valid values are 1/3 and 0.5. Only required if ``gen_poly`` is
            `None`.

        constraint_length: int
            Valid values are between 3 and 8 inclusive. Only required if
            ``gen_poly`` is `None`.

        method: str
            `soft_llr`, `soft` or `hard`. In computing path metrics, `soft_llr`
            and `soft` assume an additive white Gaussian channel, whereas
            `hard` assumes a `binary symmetric channel` (BSC). In case of
            `hard`, `inputs` will be quantized to 0/1 values.

        output_dtype: tf.DType
            Defaults to tf.float32. Defines the output datatype of the layer.

    Input
    -----
        inputs: [...,n], tf.float32
            2+D tensor containing the (noisy) channel output symbols. `n` is
            the codeword length.

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

        assert method in ('soft_llr', 'soft', 'hard'), \
            "method must be `soft_llr` `soft` or `hard`."

        # init Trellis parameters
        self._trellis = Trellis(self.gen_poly)
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
        self.output_dtype = output_dtype
        # If i->j state transition emits symbol k, tf.gather with ipst_op_idx
        # gathers (i,k) element from input in row j.
        self.ipst_op_idx = self._mask_by_tonode()

    #########################################
    # Public methods and properties
    #########################################

    @property
    def gen_poly(self):
        """The generator polynomail used by the encoder."""
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
        """
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
        """
        Given a path, compute the input bit stream that results in the path.
        Used in call() where the input is optimal path (seq of states) such
        as the path returned by _return_optimal.
        """
        ip_bits = []
        dec_syms = []
        paths = tf.cast(paths, tf.int32)
        ni = self._trellis.ni
        ip_sym_mask = tf.range(ni)[None, :]
        for sym in range(1, paths.shape[-1]):
            dec_ = tf.gather_nd(self._trellis.op_mat, paths[:, sym-1:sym+1])
            dec_syms.append(dec_)

            # bs x ni boolean tensor. Each row has a True and False. True
            # corresponds to input_bit which produced the next state (t=sym)
            match_st = tf.math.equal(
                tf.gather(self._trellis.to_nodes,paths[:, sym-1]),
                tf.tile(paths[:, sym][:, None], [1, 2])
                )

            # tf.boolean_mask throws error in XLA mode
            #ip_bit = tf.boolean_mask(ip_sym_mask, match_st)

            ip_bit_ = tf.where(match_st, ip_sym_mask, tf.zeros_like(ip_sym_mask))
            ip_bit = tf.reduce_sum(ip_bit_, axis=-1)
            ip_bits.append(ip_bit)

        ip_bit_vec_est = tf.stack(ip_bits, axis=-1)
        ip_sym_vec_est = tf.stack(dec_syms, axis=-1)

        return ip_bit_vec_est, ip_sym_vec_est

    def _optimal_path(self, cm_, tb_):
        """
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

        for sym in range(tb_.shape[-1]-1, 0, -1):
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
        The distance metric is L2 distance if decoder parameter `method` is "soft".

        The distance metric is L1 distance if parameter `method` is "hard".
        """

        op_bits = np.stack(
                    [int2bin(op, self._conv_n) for op in range(self._no)])
        op_mat = tf.cast(tf.tile(op_bits, [1,self._num_syms]), tf.float32)
        op_mat = tf.expand_dims(op_mat, axis=0)
        y = tf.expand_dims(y, axis=1)
        if self._method=='soft':
            diffsq = tf.experimental.numpy.power(y-op_mat, 2)
            diffsq = tf.reshape(diffsq,
                (-1, self._no, self._num_syms, self._conv_n))
            # Distance Squared of symbols
            bm = tf.math.reduce_sum(diffsq, axis=-1)
        elif self._method=='soft_llr':
            op_mat_sign = 1- 2.*op_mat
            llr_sign = tf.math.multiply(y, op_mat_sign)
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
            inputs = tf.cast(inputs, tf.float32)

        output_shape = inputs.get_shape().as_list()
        y_resh = tf.reshape(inputs, [-1, self._n])

        output_shape[0] = -1
        output_shape[-1] = self._k # assign k to the last dimension

        # Branch metrics matrix for a given y
        bm_mat = self._bmcalc(y_resh)

        cm_ta = tf.TensorArray(tf.float32, size=self._num_syms,
                            dynamic_size=False, clear_after_read=False)
        tb_ta = tf.TensorArray(tf.int32, size=self._num_syms,
                            dynamic_size=False)

        prev_cm_np = np.full((self._ns,), LARGEDIST)
        prev_cm_np[0] = 0.0
        prev_cm_ = tf.convert_to_tensor(prev_cm_np)

        prev_cm = tf.tile(prev_cm_[None,:], [tf.shape(y_resh)[0], 1])

        for idx in range(0, self._n, self._conv_n):
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

        msghat = tf.cast(msghat, self.output_dtype)
        cwhat = tf.cast(cwhat, self.output_dtype)

        msghat_reshaped = tf.reshape(msghat, output_shape)

        return msghat_reshaped
