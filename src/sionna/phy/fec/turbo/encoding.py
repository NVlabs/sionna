#
# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0#
"""Blocks for Turbo Code Encoding."""

import math
import tensorflow as tf
from sionna.phy import Block
from sionna.phy.fec import interleaving
from sionna.phy.fec.utils import bin2int_tf, int2bin_tf
from sionna.phy.fec.conv.encoding import ConvEncoder
from sionna.phy.fec.conv.utils import Trellis
from sionna.phy.fec.turbo.utils import polynomial_selector, puncture_pattern, \
                                       TurboTermination

class TurboEncoder(Block):
    # pylint: disable=line-too-long
    r"""Performs encoding of information bits to a Turbo code codeword

    Implements the standard Turbo code framework [Berrou]_: Two identical
    rate-1/2 convolutional encoders
    :class:`~sionna.phy.fec.conv.encoding.ConvEncoder` are combined to produce a
    rate-1/3 Turbo code. Further, puncturing to attain a rate-1/2 Turbo code is
    supported.

    Parameters
    ----------
    gen_poly: tuple | `None` (default)
        Tuple of strings with each string being a 0,1 sequence. If
        `None`, ``constraint_length`` must be provided.

    constraint_length: int, 3...6
        Valid values are between 3 and 6 inclusive. Only required if
        ``gen_poly`` is `None`.

    rate: float, 1/2 | 1/3
        Valid values are 1/3 and 1/2. Note that ``rate`` here denotes
        the `design` rate of the Turbo code. If ``terminate`` is `True`, a
        small rate-loss occurs.

    terminate: `bool`, (default `False`)
        Underlying convolutional encoders are terminated to all zero state
        if `True`. If terminated, the true rate of the code is slightly lower
        than ``rate``.

    interleaver_type: str, "3GPP" | "random"
        Valid values are `"3GPP"` or `"random"`. Determines the choice of
        the interleaver to interleave the message bits before input to the
        second convolutional encoder. If `"3GPP"`, the Turbo code interleaver
        from the 3GPP LTE standard [3GPPTS36212_Turbo]_ is used. If `"random"`,
        a random interleaver is used.

    precision : `None` (default) | 'single' | 'double'
        Precision used for internal calculations and outputs.
        If set to `None`, :py:attr:`~sionna.phy.config.precision` is used.

    Input
    -----
    inputs : [...,k], tf.float32
        Tensor of information bits where `k` is the information length

    Output
    ------
    : `[...,k/rate]`, tf.float32
        Tensor where `rate` is provided as input parameter. The output is the
        encoded codeword for the input information tensor. When ``terminate``
        is `True`, the effective rate of the Turbo code is slightly less than
        ``rate``.

    Note
    ----
    Various notations are used in literature to represent the generator
    polynomials for convolutional codes. For simplicity
    :class:`~sionna.phy.fec.turbo.encoding.TurboEncoder` only
    accepts the binary format, i.e., `10011`, for the ``gen_poly`` argument
    which corresponds to the polynomial :math:`1 + D^3 + D^4`.

    Note that Turbo codes require the underlying convolutional encoders
    to be recursive systematic encoders. Only then the channel output
    from the systematic part of the first encoder can be used to decode
    the second encoder.

    Also note that ``constraint_length`` and ``memory`` are two different
    terms often used to denote the strength of the convolutional code. In
    this sub-package we use ``constraint_length``. For example, the polynomial
    `10011` has a ``constraint_length`` of 5, however its ``memory`` is
    only 4.

    When ``terminate`` is `True`, the true rate of the Turbo code is
    slightly lower than ``rate``. It can be computed as
    :math:`\frac{k}{\frac{k}{r}+\frac{4\mu}{3r}}` where `r` denotes
    ``rate`` and :math:`\mu` is the ``constraint_length`` - 1. For example, in
    3GPP, ``constraint_length`` = 4, ``terminate`` = `True`, for
    ``rate`` = 1/3, true rate is equal to  :math:`\frac{k}{3k+12}`.
    """

    def __init__(self,
                 gen_poly=None,
                 constraint_length=3,
                 rate=1/3,
                 terminate=False,
                 interleaver_type='3GPP',
                 precision=None,
                 **kwargs):

        super().__init__(precision=precision, **kwargs)

        if gen_poly is not None:
            if not all(isinstance(poly, str) for poly in gen_poly):
                raise TypeError("Each element of gen_poly must be a string.")
            if not all(len(poly)==len(gen_poly[0]) for poly in gen_poly):
                raise ValueError("Each polynomial must be of same length.")
            if not all(all(
                char in ['0','1'] for char in poly) for poly in gen_poly):
                raise ValueError("Each Polynomial must be a string of 0/1 s.")
            if len(gen_poly)!=2:
                raise ValueError("Generator polynomials need to be of rate-1/2")
            self._gen_poly = gen_poly
        else:
            valid_constraint_length = (3, 4, 5, 6)
            if constraint_length not in valid_constraint_length:
                raise ValueError("Constraint length must be between 3 and 6.")
            self._gen_poly = polynomial_selector(constraint_length)

        valid_rates = (1/2, 1/3)
        if rate not in valid_rates:
            raise ValueError("Invalid coderate.")
        if not isinstance(terminate, bool):
            raise TypeError("terminate must be bool.")
        if interleaver_type not in ('3GPP', 'random'):
            raise ValueError("Invalid interleaver_type.")

        self._coderate_desired = rate
        self._coderate = self._coderate_desired
        self._terminate = terminate
        self._interleaver_type = interleaver_type

        # Underlying convolutional encoders to be rsc or not
        rsc = True

        self._coderate_conv = 1/len(self.gen_poly)
        self._punct_pattern = puncture_pattern(rate, self._coderate_conv)

        self._trellis = Trellis(self.gen_poly, rsc=rsc)
        self._mu = self.trellis._mu

        # conv_n denotes number of output bits for conv_k input bits.
        self._conv_k = self._trellis.conv_k
        self._conv_n = self._trellis.conv_n

        self._ni = 2**self._conv_k
        self._no  = 2**self._conv_n
        self._ns = self._trellis.ns

        # for conv codes, the code dimensions are unknown during initialization
        self._k = None
        self._n = None

        if self.terminate:
            self.turbo_term = TurboTermination(self._mu+1, conv_n=self._conv_n)

        if self._interleaver_type == '3GPP':
            self.internal_interleaver = interleaving.Turbo3GPPInterleaver()
        else:
            self.internal_interleaver = interleaving.RandomInterleaver(
                                                    keep_batch_constant=True,
                                                    keep_state=True,
                                                    axis=-1)

        if self.punct_pattern is not None:
            self.punct_idx = tf.where(self.punct_pattern)

        self.convencoder = ConvEncoder(gen_poly=self._gen_poly,
                                       rsc=rsc,
                                       terminate=self._terminate)

    #########################################
    # Public methods and properties
    #########################################

    @property
    def gen_poly(self):
        """Generator polynomial used by the encoder"""
        return self._gen_poly

    @property
    def constraint_length(self):
        """Constraint length of the encoder"""
        return self._mu + 1

    @property
    def coderate(self):
        """Rate of the code used in the encoder"""
        if self.terminate and self._k is None:
            print("Note that, due to termination, the true coderate is lower "\
                  "than the returned design rate. "\
                  "The exact true rate is dependent on the value of k and "\
                  "hence cannot be computed before the first call().")
        elif self.terminate and self._k is not None:
            term_factor = 1+math.ceil(4*self._mu/3)/self._k
            self._coderate = self._coderate_desired/term_factor
        return self._coderate

    @property
    def trellis(self):
        """Trellis object used during encoding"""
        return self._trellis

    @property
    def terminate(self):
        """Indicates if the convolutional encoders are terminated"""
        return self._terminate

    @property
    def punct_pattern(self):
        """Puncturing pattern for the Turbo codeword"""
        return self._punct_pattern

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

    def _conv_enc(self, info_vec, terminate):
        """
        This method encodes the information tensor info_vec using the
        underlying convolutional encoder. Returns the encoded codeword tensor
        array ta, and the tensor array containing termination bits ta_term.
        If the terminate variable is False, ta_term is array of length 0.
        """
        msg = tf.cast(info_vec, tf.int32)

        msg_reshaped = tf.reshape(msg, [-1, self._k])
        term_syms = int(self._mu) if terminate else 0

        prev_st = tf.zeros([tf.shape(msg_reshaped)[0]], tf.int32)
        ta = tf.TensorArray(tf.int32, size=self.num_syms, dynamic_size=False)

        idx_offset = range(0, self._conv_k)
        for idx in tf.range(0, self._k, self._conv_k):
            msg_bits_idx = tf.gather(msg_reshaped,
                                     idx + idx_offset,
                                     axis=-1)

            #msg_bits_idx = tf.experimental.numpy.take_along_axis(msg_reshaped)

            msg_idx = bin2int_tf(msg_bits_idx)

            indices = tf.stack([prev_st, msg_idx], -1)
            new_st = tf.gather_nd(self._trellis.to_nodes, indices=indices)

            idx_syms = tf.gather_nd(self._trellis.op_mat,
                                    tf.stack([prev_st, new_st], -1))
            idx_bits = int2bin_tf(idx_syms, self._conv_n)
            ta = ta.write(idx//self._conv_k, idx_bits)
            prev_st = new_st

        ta_term = tf.TensorArray(tf.int32, size=term_syms, dynamic_size=False)
        # Termination
        if terminate:
            fb_poly = tf.constant([int(x) for x in self.gen_poly[0][1:]])
            fb_poly_tiled = tf.tile(
                tf.expand_dims(fb_poly,0),[tf.shape(prev_st)[0],1])
            for idx in tf.range(0, term_syms, self._conv_k):
                prev_st_bits = int2bin_tf(prev_st, self._mu)
                msg_idx = tf.math.reduce_sum(
                                    tf.multiply(fb_poly_tiled, prev_st_bits),-1)
                msg_idx = tf.squeeze(int2bin_tf(msg_idx,1),-1)

                indices = tf.stack([prev_st, msg_idx], -1)
                new_st = tf.gather_nd(self._trellis.to_nodes, indices=indices)
                idx_syms = tf.gather_nd(self._trellis.op_mat,
                                        tf.stack([prev_st, new_st], -1))
                idx_bits = int2bin_tf(idx_syms, self._conv_n)
                ta_term = ta_term.write(idx//self._conv_k, idx_bits)
                prev_st = new_st

        return ta, ta_term

    def _puncture_cw(self, cw):
        """
        Given the codeword ``cw``, this method punctures ``cw`` using the
        puncturing pattern defined in self.punct_pattern. A simple tile
        operation of self.punct_pattern followed by tf.boolean_mask(cw, mask_)
        works. However this fails in XLA mode as the dimension of the above
        operation is unknown.

        Hence, idx is obtained from `tf.where(self.punct_pattern)` during
        initialization. This way the dimension of idx is known during graph
        creation. Then during the call(), idx is tiled followed by row offset
        addition to idx (the indices tensor) will achieve the same result as
        applying a tiled boolean_mask.
        """
        # cw shape: (bs, n, 3)- transpose to (n, 3, bs)
        cw = tf.transpose(cw, perm=[1, 2, 0])
        cw_n = cw.get_shape()[0]

        punct_period = self.punct_pattern.shape[0]
        mask_reps = cw_n//punct_period
        idx = tf.tile(self.punct_idx, [mask_reps, 1])

        idx_per_period = self.punct_idx.shape[0]
        idx_per_time = idx_per_period/punct_period

        # When tiling punct_pattern doesn't cover cw, delta_times > 0
        delta_times  = cw_n - (mask_reps * punct_period)
        delta_idx_rows = int(delta_times*idx_per_time)

        time_offset = punct_period * tf.range(mask_reps)[None,:]
        row_idx = tf.transpose(tf.tile(time_offset,[idx_per_period,1]))
        row_idx = tf.reshape(row_idx, (-1, 1))

        total_indices = mask_reps*idx_per_period + delta_idx_rows
        col_idx = tf.zeros((total_indices,1), tf.int32)

        if delta_times > 0:
            idx = tf.concat([idx, self.punct_idx[:delta_idx_rows]], axis=0)
            # Additional index row offsets if delta_times > 0
            time_n = punct_period*mask_reps
            row_idx_delta = tf.tile(
                                tf.range(time_n, time_n+delta_times)[None, :],
                                [delta_idx_rows, 1])
            row_idx = tf.concat([row_idx, row_idx_delta], axis=0)

        idx_offset = tf.cast(tf.concat([row_idx, col_idx], axis=1), tf.int64)
        idx = tf.add(idx, idx_offset)

        cw = tf.gather_nd(cw, idx)
        cw = tf.transpose(cw)
        return cw

    ########################
    # Sionna Block functions
    ########################

    def build(self, input_shape):
        """Build Block and check dimensions.

        Args:
            input_shape: shape of input tensor (...,k).
        """
        self._k = input_shape[-1]
        self._n = int(self._k/self._coderate_desired)
        if self._interleaver_type == '3GPP':
            if self._k > 6144:
                msg = '3GPP Turbo Codes define Interleavers only'\
                      'upto frame lengths of 6144'
                raise ValueError(msg)

        # Num. of encoding periods/state transitions.
        # Not equal to _k if_conv_k>1.
        self.num_syms = int(self._k//self._conv_k)

    def call(self, bits):
        """Turbo code encoding function.
        Args:
            bits (tf.float): Information tensor of shape `[...,k]`.

        Returns:
            `tf.float`: Encoded codeword tensor of shape `[...,n]`.
        """

        if bits.shape[-1] != self._k:
            self.build(bits.shape)

        if self._terminate:
            num_term_bits_ = int(
                self.turbo_term.get_num_term_syms()/self._coderate_conv)
            num_term_bits_punct = int(
                num_term_bits_*self._coderate_conv/self._coderate_desired)
        else:
            num_term_bits_ = 0
            num_term_bits_punct = 0

        output_shape = bits.get_shape().as_list()
        output_shape[0] = -1
        output_shape[-1] = self._n + num_term_bits_punct

        preterm_n = int(self._k/self._coderate_conv)
        msg = tf.cast(tf.reshape(bits, [-1, self._k]), tf.int32)
        msg2 = self.internal_interleaver(msg)

        cw1_ = self.convencoder(msg)
        cw2_ = self.convencoder(msg2)

        cw1, term1 = cw1_[:, :preterm_n], cw1_[:, preterm_n:]
        cw2, term2 = cw2_[:, :preterm_n], cw2_[:, preterm_n:]

        # Gather parity stream from 2nd enc
        par_idx = tf.range(1, preterm_n, delta=self._conv_n)
        cw2_par = tf.gather(cw2, indices=par_idx, axis=-1)

        cw1 = tf.reshape(cw1,(-1, self._k, self._conv_n))
        cw2_par = tf.reshape(cw2_par, (-1, self._k, 1))

        # Concatenate 2nd enc parity to _conv_n streams from first encoder
        cw = tf.concat([cw1, cw2_par], axis=-1)

        if self.terminate:
            term_syms_turbo = self.turbo_term.termbits_conv2turbo(term1, term2)
            term_syms_turbo = tf.reshape(
                term_syms_turbo, (-1, num_term_bits_//2, 3))
            cw = tf.concat([cw, term_syms_turbo], axis=-2)

        if self.punct_pattern is not None:
            cw = self._puncture_cw(cw)

        cw = tf.cast(cw, self.rdtype)
        cw_reshaped = tf.reshape(cw, output_shape)
        return cw_reshaped
