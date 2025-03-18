#
# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0#
"""Blocks for Turbo Decoding."""

import numpy as np
import tensorflow as tf
from sionna.phy import Block
from sionna.phy.fec import interleaving
from sionna.phy.fec.conv.decoding import BCJRDecoder
from sionna.phy.fec.conv.utils import Trellis
from sionna.phy.fec.turbo.utils import TurboTermination, polynomial_selector, \
                                       puncture_pattern

class TurboDecoder(Block):
    # pylint: disable=line-too-long
    r"""Turbo code decoder based on BCJR component decoders [Berrou]_.

    Takes as input LLRs and returns LLRs or hard decided bits, i.e., an
    estimate of the information tensor.

    This decoder is based on the
    :class:`~sionna.phy.fec.conv.decoding.BCJRDecoder` and, thus, internally
    instantiates two :class:`~sionna.phy.fec.conv.decoding.BCJRDecoder` blocks.

    Parameters
    ----------
    encoder: :class:`~sionna.phy.fec.turbo.encoding.TurboEncoder`
        If ``encoder`` is provided as input, the following input parameters
        are not required and will be ignored: `gen_poly`, `rate`,
        `constraint_length`, `terminate`, `interleaver`. They will be inferred
        from the ``encoder`` object itself.
        If ``encoder`` is `None`, the above parameters must be provided
        explicitly.

    gen_poly: tuple or None
        Tuple of strings with each string being a 0, 1 sequence. If `None`,
        ``rate`` and ``constraint_length`` must be provided.

    rate: float, 1/3 | 1/2
        Rate of the Turbo code. Valid values are 1/3 and 1/2. Note that
        ``gen_poly``, if provided, is used to encode the underlying
        convolutional code, which traditionally has rate 1/2.

    constraint_length: int, 3...6
        Valid values are between 3 and 6 inclusive. Only required if
        ``encoder`` and ``gen_poly`` are `None`.

    interleaver: str, `None` (default) | "3GPP" | "Random"
        `"3GPP"` or `"Random"`. If `"3GPP"`, the internal interleaver for Turbo
        codes as specified in [3GPPTS36212_Turbo]_ will be used. Only required
        if ``encoder`` is `None`.

    terminate: `bool`, (default `False`)
        If `True`, the two underlying convolutional encoders are assumed
        to have terminated to all zero state.

    num_iter: int
        Number of iterations for the Turbo decoding to run. Each iteration of
        Turbo decoding entails one BCJR decoder for each of the underlying
        convolutional code components.

    hard_out: `bool`, (default `True`)
        Indicates whether to output hard or soft
        decisions on the decoded information vector. `True` implies a hard-
        decoded information vector of 0/1's is output. `False` implies
        decoded LLRs of the information is output.

    algorithm: str, "map" (default) | "log" | "maxlog"
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
    llr_ch: tf.float
        Tensor of shape `[...,n]` containing the (noisy) channel
        output symbols where `n` is the codeword length.

    Output
    ------
    : tf.float
        Tensor of shape `[..., coderate * n]` containing the estimates of the
        information bit tensor.

    Note
    ----
    For decoding, input `logits` defined as
    :math:`\operatorname{log} \frac{p(x=1)}{p(x=0)}` are assumed for
    compatibility with the rest of Sionna. Internally,
    log-likelihood ratios (LLRs) with definition
    :math:`\operatorname{log} \frac{p(x=0)}{p(x=1)}` are used.
    """

    def __init__(self,
                 encoder=None,
                 gen_poly=None,
                 rate=1/3,
                 constraint_length=None,
                 interleaver='3GPP',
                 terminate=False,
                 num_iter=6,
                 hard_out=True,
                 algorithm='map',
                 precision=None,
                 **kwargs):

        super().__init__(precision=precision, **kwargs)

        if encoder is not None:
            self._coderate = encoder._coderate
            self._gen_poly = encoder._gen_poly
            self._terminate = encoder.terminate
            self._trellis = encoder.trellis
            assert self._trellis.rsc is True
            self.rsc = True
            self.internal_interleaver = encoder.internal_interleaver
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
                valid_constraint_length = (3, 4, 5, 6)
                if not constraint_length in valid_constraint_length:
                    msg = "Constraint length must be between 3 and 6."
                    raise ValueError(msg)
                self._gen_poly = polynomial_selector(constraint_length)

            valid_rates = (1/2, 1/3)
            if rate not in valid_rates:
                raise ValueError("rate must be 1/3 or 1/2.")
            self._coderate = rate

            if not isinstance(terminate, bool):
                raise TypeError("terminate must be bool.")
            self._terminate = terminate

            if interleaver not in ('3GPP', 'random'):
                raise ValueError("interleaver must be 3GPP or random.")

            if interleaver == '3GPP':
                self.internal_interleaver = \
                    interleaving.Turbo3GPPInterleaver(precision=precision)
            else:
                self.internal_interleaver = interleaving.RandomInterleaver(
                    keep_batch_constant=True,
                    keep_state=True,
                    axis=-1,
                    precision=precision)

            self.rsc = True
            self._trellis = Trellis(self._gen_poly, rsc=self.rsc)

        if not isinstance(hard_out, bool):
            raise TypeError('hard_out must be bool.')

        self._coderate_conv = 1/len(self._gen_poly)
        self._mu = len(self._gen_poly[0])-1
        self.punct_pattern = puncture_pattern(self._coderate,
                                              self._coderate_conv)

        # num. of input bit streams, only 1 in current implementation
        self._conv_k = self._trellis.conv_k
        self._mu = self._trellis._mu
        # num. of output bits for conv_k input bits
        self._conv_n = self._trellis.conv_n
        self._ni = 2**self._conv_k
        self._no = 2**self._conv_n
        self._ns = self._trellis.ns

        if self._conv_k != 1:
            raise NotImplementedError("Only single bit stream support.")
        if self._conv_n != 2:
            raise NotImplementedError("Only single bit stream support.")

        # for conv codes, the code dimensions are unknown during initialization
        self._k = None # Length of Info-bit vector
        self._n = None # Length of Turbo codeword, including termination bits

        if self._terminate:
            self.turbo_term = TurboTermination(self._mu+1,
                                               conv_n=self._conv_n)
            self._num_term_bits = 3 * self.turbo_term.get_num_term_syms()
        else:
            self._num_term_bits = 0

        self.num_iter = num_iter
        self._hard_out = hard_out

        self.bcjrdecoder = BCJRDecoder(gen_poly=self._gen_poly,
                                       rsc=self.rsc,
                                       hard_out=False,
                                       terminate=self._terminate,
                                       algorithm=algorithm,
                                       precision=precision)

    ###############################
    # Public methods and properties
    ###############################

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
        return self._coderate

    @property
    def trellis(self):
        """Trellis object used during encoding"""
        return self._trellis

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

    ###################
    # Utility functions
    ###################

    def depuncture(self, y):
        """
        Given a tensor `y` of shape `[batch, n]`, depuncture() scatters `y`
        elements into shape `[batch, 3*rate*n]` where the
        extra elements are filled with 0.

        For e.g., if input is `y`, rate is 1/2 and
        `punct_pattern` is [1, 1, 0, 1, 0, 1], then the
        output is [y[0], y[1], 0., y[2], 0., y[3], y[4], y[5], 0., ... ,].
        """

        y_depunct = tf.scatter_nd(self._punct_indices,
                                  tf.transpose(y),
                                  shape=(self._depunct_len, tf.shape(y)[0]))
        y_depunct = tf.transpose(y_depunct)
        return y_depunct

    def _convenc_cws(self, y_turbo):
        """
        _convenc_cws() re-arranges Turbo Codeword to the two Convolutional
        codewords format.
        Given the channel output of a Turbo codeword y_turbo, this method
        re-arranges y_turbo such that y1_cw contains the symbols corresponding
        to Conv. Encoder 1 & similarly y2_cw contains the symbols corresponding
        to Conv. Encoder 2
        """
        y_turbo = self.depuncture(y_turbo)
        prepunct_n = int(self._n * 3 * self._coderate)

        # Separate Pre-termination & Termination parts of Y
        msg_idx = tf.range(0, prepunct_n - self._num_term_bits)
        term_idx = tf.range(prepunct_n-self._num_term_bits, prepunct_n)

        # Pre-termination & Termination parts of Y
        y_cw = tf.gather(y_turbo, msg_idx, axis=-1)
        y_term = tf.gather(y_turbo, term_idx, axis=-1)

        # Gather Encoder1 corresp. from Y (pre-termination part)
        enc1_sys_idx = tf.expand_dims(tf.range(0, self._k*3, delta=3), 1)
        enc1_cw_idx = tf.stack([enc1_sys_idx, enc1_sys_idx+1], axis=1)
        enc1_cw_idx = tf.squeeze(tf.reshape(enc1_cw_idx, (-1, 2*self._k)))
        y1_cw = tf.gather(y_cw, enc1_cw_idx, axis=-1)

        # Gather systematic part of codeword from encoder1 & Inverse-interleave
        y1_sys_cw = tf.gather(y_cw, enc1_sys_idx, axis=-1)
        y2_sys_cw = self.internal_interleaver(
                            tf.squeeze(y1_sys_cw, -1))[:,:,None]

        # Using above, gather Encoder2 corresp. from Y (pre-termination part)
        y2_nonsys_cw = tf.gather(y_cw, enc1_sys_idx+2, axis=-1)
        y2_cw = tf.squeeze(tf.stack([y2_sys_cw, y2_nonsys_cw], axis=-2))
        y2_cw = tf.reshape(y2_cw, [-1, 2*self._k])

        # Separate Termination bits to encoders 1 & 2
        if self._terminate:
            term_vec1, term_vec2 = self.turbo_term.term_bits_turbo2conv(y_term)
            y1_cw = tf.concat([y1_cw, term_vec1],axis=1)
            y2_cw = tf.concat([y2_cw, term_vec2],axis=-1)
        return y1_cw, y2_cw

    ########################
    # Sionna Block functions
    ########################

    def build(self, input_shape):
        """Build block and check dimensions."""

        self._n = input_shape[-1]
        if self.coderate == 1/2:
            if self._n%2 != 0:
                raise ValueError("Codeword length should be a multiple of 2")

        codefactor = self.coderate * 3
        turbo_n = int(self._n * codefactor)
        turbo_n_preterm = turbo_n - self._num_term_bits
        if turbo_n_preterm%3 != 0:
            msg = "Invalid codeword length for a terminated Turbo code"
            raise ValueError(msg)

        self._k = int(turbo_n_preterm/3)

        # num of symbols for the convolutional codes.
        self._convenc_numsyms = self._k
        if self._terminate:
            self._convenc_numsyms += self._mu

        # generate puncturing mask
        rate_factor = 3. * self._coderate

        self._depunct_len = int(rate_factor * self._n)
        punct_size = np.prod(self.punct_pattern.get_shape().as_list())
        rep_times = int(self._depunct_len//punct_size)

        mask_ = tf.tile(self.punct_pattern, [rep_times, 1])
        extra_bits  = int(self._depunct_len - rep_times*punct_size)
        if extra_bits  > 0:
            extra_periods = int(extra_bits/3)
            mask_ = tf.concat([mask_, self.punct_pattern[:extra_periods,:]],
                              axis=0)

        mask_ = tf.squeeze(tf.reshape(mask_, (-1, )))
        self._punct_indices = tf.cast(tf.where(mask_), tf.int32)

    def call(self, llr_ch, /):
        """
        Decoder for Turbo code.

        Runs BCJR decoder on both the constituent convolutional codes
        iteratively `num_iter` times. At the end, the resultant LLRs are
        computed and the decoded message vector (termination bits are
        excluded) is output.
        """
        print(llr_ch.dtype)
        llr_max = 20.

        output_shape = llr_ch.get_shape().as_list()

        # allow different codeword lengths in eager mode
        if output_shape[-1] != self._n:
            self.build(output_shape)

        llr_ch = tf.reshape(llr_ch, [-1, self._n])

        output_shape[0] = -1
        output_shape[-1] = self._k # assign k to the last dimension

        # llr's inside TurboDecoder are not sign-inverted after input,
        # unlike BCJR & LDPC decoders. They represent P(x=1)/P(x=0) as
        # convention in Sionna.
        y1_cw, y2_cw = self._convenc_cws(llr_ch)

        sys_idx = tf.expand_dims(tf.range(0, self._k*2, delta=2), 1)
        llr_ch = tf.gather(y1_cw, sys_idx, axis=-1)
        llr_ch = tf.squeeze(llr_ch, -1)
        llr_ch2 = tf.gather(y2_cw, sys_idx, axis=-1)
        llr_ch2 = tf.squeeze(llr_ch2, -1)

        llr_1e = tf.zeros((tf.shape(llr_ch)[0], self._convenc_numsyms),
                          dtype=self.rdtype)
        # define zero LLR's for termination info bits
        term_info_bits = self._mu if self._terminate else 0
        llr_terminfo = tf.zeros(
                        (tf.shape(llr_ch)[0], term_info_bits), self.rdtype)

        # needs to be initialized for XLA before entering the loop
        llr_2i = tf.zeros_like(llr_ch2)

        # run decoding loop
        for _ in tf.range(self.num_iter):

            # run 1st component decoder
            llr_1i = self.bcjrdecoder(y1_cw, llr_a=llr_1e)
            llr_1i = llr_1i[...,:self._k]
            llr_extr = llr_1i - llr_ch - llr_1e[...,:self._k]

            llr_2e = self.internal_interleaver(llr_extr)
            llr_2e = tf.concat([llr_2e, llr_terminfo], axis=-1)
            llr_2e = tf.clip_by_value(llr_2e,
                                      clip_value_min=-llr_max,
                                      clip_value_max=llr_max)
            # run 2nd component decoder
            llr_2i = self.bcjrdecoder(y2_cw, llr_a=llr_2e)
            llr_2i = llr_2i[...,:self._k]
            llr_extr = llr_2i - llr_2e[...,:self._k] - llr_ch2

            llr_1e = self.internal_interleaver(llr_extr, inverse=True)

            llr_1e = tf.clip_by_value(llr_1e,
                                      clip_value_min=-llr_max,
                                      clip_value_max=llr_max)

            llr_1e = tf.concat([llr_1e, llr_terminfo], axis=-1)

        # use latest output of 2nd decoder
        output = self.internal_interleaver(llr_2i, inverse=True)

        if self._hard_out: # hard decide decoder output if required
            output = tf.less(0.0, output)
        output = tf.cast(output, self.rdtype)

        output_reshaped = tf.reshape(output, output_shape)
        return output_reshaped
