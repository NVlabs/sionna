#
# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Layer for Convolutional Code Encoding."""

import tensorflow as tf
from tensorflow.keras.layers import Layer
from sionna.fec.utils import bin2int_tf, int2bin_tf
from sionna.fec.conv.utils import polynomial_selector, Trellis

class ConvEncoder(Layer):
    # pylint: disable=line-too-long
    r"""ConvEncoder(gen_poly=None, rate= 1/2, constraint_length=3, rsc=False, terminate=False, output_dtype=tf.float32, **kwargs)

    Encodes an information binary tensor to a convolutional codeword. Currently,
    only generator polynomials for codes of rate=1/n for n=2,3,4,... are allowed.

    The class inherits from the Keras layer class and can be used as layer in a
    Keras model.

    Parameters
    ----------
        gen_poly: tuple
            Sequence of strings with each string being a 0,1 sequence. If
            `None`, ``rate`` and ``constraint_length`` must be provided.

        rate: float
            Valid values are 1/3 and 0.5. Only required if ``gen_poly`` is
            `None`.

        constraint_length: int
            Valid values are between 3 and 8 inclusive. Only required if
            ``gen_poly`` is `None`.

        rsc: boolean
            Boolean flag indicating whether the Trellis generated is recursive
            systematic or not. If `True`, the encoder is recursive-systematic.
            In this case first polynomial in ``gen_poly`` is used as the
            feedback polynomial. Defaults to `False`.

        terminate: boolean
            Encoder is terminated to all zero state if `True`.
            If terminated, the `true` rate of the code is slightly lower than
            ``rate``.

        output_dtype: tf.DType
            Defaults to `tf.float32`. Defines the output datatype of the layer.

    Input
    -----
        inputs : [...,k], tf.float32
            2+D tensor containing the information bits where `k` is the
            information length

    Output
    ------
        : [...,k/rate], tf.float32
            2+D tensor containing the encoded codeword for the given input
            information tensor where `rate` is
            :math:`\frac{1}{\textrm{len}\left(\textrm{gen_poly}\right)}`
            (if ``gen_poly`` is provided).

    Note
    ----
        The generator polynomials from [Moon]_ are available for various
        rate and constraint lengths. To select them, use the ``rate`` and
        ``constraint_length`` arguments.

        In addition, polynomials for any non-recursive convolutional encoder
        can be given as input via ``gen_poly`` argument. Currently, only
        polynomials with rate=1/n are supported. When the ``gen_poly`` argument
        is given, the ``rate`` and ``constraint_length`` arguments are ignored.

        Various notations are used in the literature to represent the generator
        polynomials for convolutional codes. In [Moon]_, the octal digits
        format is primarily used. In the octal format, the generator polynomial
        `10011` corresponds to 46. Another widely used format
        is decimal notation with MSB. In this notation, polynomial `10011`
        corresponds to 19. For simplicity, the
        :class:`~sionna.fec.conv.encoding.ConvEncoder` only accepts the bit
        format i.e. `10011` as ``gen_poly`` argument.

        Also note that ``constraint_length`` and ``memory`` are two different
        terms often used to denote the strength of a convolutional code. In this
        sub-package, we use ``constraint_length``. For example, the
        polynomial `10011` has a ``constraint_length`` of 5, however its
        ``memory`` is only 4.

        When ``terminate`` is `True`, the true rate of the convolutional
        code is slightly lower than ``rate``. It equals
        :math:`\frac{r*k}{k+\mu}` where `r` denotes ``rate`` and
        :math:`\mu` is ``constraint_length`` - 1. For example when
        ``terminate`` is `True`, ``k=100``,
        :math:`\mu=4` and ``rate`` =0.5, true rate equals
        :math:`\frac{0.5*100}{104}=0.481`.
    """

    def __init__(self,
                 gen_poly=None,
                 rate=1/2,
                 constraint_length=3,
                 rsc=False,
                 terminate=False,
                 output_dtype=tf.float32,
                 **kwargs):

        super().__init__(**kwargs)

        if gen_poly is not None:
            assert all(isinstance(poly, str) for poly in gen_poly), \
                "Each element of gen_poly must be a string."
            assert all(len(poly)==len(gen_poly[0]) for poly in gen_poly), \
                "Each polynomial must be of same length."
            assert all(all(
                char in ['0','1'] for char in poly) for poly in gen_poly),\
                "Each Polynomial must be a string of 0/1 s."
            self._gen_poly = gen_poly
        else:
            valid_rates = (1/2, 1/3)
            valid_constraint_length = (3, 4, 5, 6, 7, 8)

            assert constraint_length in valid_constraint_length, \
                "Constraint length must be between 3 and 8."
            assert rate in valid_rates, \
                "Rate must be 1/3 or 1/2."
            self._gen_poly = polynomial_selector(rate, constraint_length)

        self._rsc = rsc
        self._terminate = terminate

        self._coderate_desired = 1/len(self.gen_poly)
        # Differ when terminate is True
        self._coderate = self._coderate_desired

        self._trellis = Trellis(self.gen_poly,rsc=self._rsc)
        self._mu = self.trellis._mu

        # conv_k denotes number of input bit streams.
        # Only 1 allowed in current implementation
        self._conv_k = self._trellis.conv_k

        # conv_n denotes number of output bits for conv_k input bits
        self._conv_n = self._trellis.conv_n

        self._ni = 2**self._conv_k
        self._no  = 2**self._conv_n
        self._ns = self._trellis.ns

        self._k = None
        self._n = None
        self.output_dtype = output_dtype

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
        if self.terminate and self._k is None:
            print("Note that, due to termination, the true coderate is lower "\
                  "than the returned design rate. "\
                  "The exact true rate is dependent on the value of k and "\
                  "hence cannot be computed before the first call().")
        elif self.terminate and self._k is not None:
            term_factor = (self._k/(self._k + self._mu))
            self._coderate = self._coderate_desired*term_factor
        return self._coderate

    @property
    def trellis(self):
        """Trellis object used during encoding"""
        return self._trellis

    @property
    def terminate(self):
        """Indicates if the convolutional encoder is terminated"""
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
    # Keras layer functions
    #########################

    def build(self, input_shape):
        """Build layer and check dimensions.

        Args:
            input_shape: shape of input tensor (...,k).
        """
        self._k = input_shape[-1]
        self._n = int(self._k/self.coderate)

        # num_syms denote number of encoding periods or state transitions.
        # different from _k when _conv_k > 1.
        self.num_syms = int(self._k//self._conv_k)

    def call(self, inputs):
        """Convolutional code encoding function.

        Args:
            inputs (tf.float32): Information tensor of shape `[...,k]`.

        Returns:
            `tf.float32`: Encoded codeword tensor of shape `[...,n]`.
        """
        tf.debugging.assert_greater(tf.rank(inputs), 1)

        if inputs.shape[-1] != self._k:
            self.build(inputs.shape)

        msg = tf.cast(inputs, tf.int32)
        output_shape = msg.get_shape().as_list()
        output_shape[0] = -1 # overwrite batch dim (can be none in keras)
        output_shape[-1] = self._n # assign n to the last dim

        msg_reshaped = tf.reshape(msg, [-1, self._k])
        term_syms = int(self._mu) if self._terminate else 0

        prev_st = tf.zeros([tf.shape(msg_reshaped)[0]], tf.int32)
        ta = tf.TensorArray(tf.int32, size=self.num_syms, dynamic_size=False)

        idx_offset = range(0, self._conv_k)
        for idx in tf.range(0, self._k, self._conv_k):
            msg_bits_idx = tf.gather(msg_reshaped,
                                     idx + idx_offset,
                                     axis=-1)

            msg_idx = bin2int_tf(msg_bits_idx)

            indices = tf.stack([prev_st, msg_idx], -1)
            new_st = tf.gather_nd(self._trellis.to_nodes, indices=indices)

            idx_syms = tf.gather_nd(self._trellis.op_mat,
                                    tf.stack([prev_st, new_st], -1))
            idx_bits = int2bin_tf(idx_syms, self._conv_n)
            ta = ta.write(idx//self._conv_k, idx_bits)
            prev_st = new_st
        cw = tf.concat(tf.unstack(ta.stack()), axis=1)

        ta_term = tf.TensorArray(tf.int32, size=term_syms, dynamic_size=False)
        # Termination
        if self._terminate:
            if self._rsc:
                fb_poly = tf.constant([int(x) for x in self.gen_poly[0][1:]])
                fb_poly_tiled = tf.tile(
                        tf.expand_dims(fb_poly,0),[tf.shape(prev_st)[0],1])

            for idx in tf.range(0, term_syms, self._conv_k):
                prev_st_bits = int2bin_tf(prev_st, self._mu)
                if self._rsc:
                    msg_idx = tf.math.reduce_sum(
                                        tf.multiply(fb_poly_tiled, prev_st_bits),-1)
                    msg_idx = tf.squeeze(int2bin_tf(msg_idx,1),-1)
                else:
                    msg_idx = tf.zeros((tf.shape(prev_st)[0],), dtype=tf.int32)

                indices = tf.stack([prev_st, msg_idx], -1)
                new_st = tf.gather_nd(self._trellis.to_nodes, indices=indices)
                idx_syms = tf.gather_nd(self._trellis.op_mat,
                                        tf.stack([prev_st, new_st], -1))
                idx_bits = int2bin_tf(idx_syms, self._conv_n)
                ta_term = ta_term.write(idx//self._conv_k, idx_bits)
                prev_st = new_st

            term_bits = tf.concat(tf.unstack(ta_term.stack()), axis=1)
            cw = tf.concat([cw, term_bits], axis=-1)

        cw = tf.cast(cw, self.output_dtype)
        cw_reshaped = tf.reshape(cw, output_shape)

        return cw_reshaped

