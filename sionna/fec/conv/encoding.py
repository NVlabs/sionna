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
    r"""ConvEncoder(gen_poly=None, rate= 1/2, constraint_length=3, output_dtype=tf.float32, **kwargs)

    Encodes an information binary tensor to a convolutional codeword.
    Only non-recursive encoding is available. Currently, only generator
    polynomials for codes of rate=1/n for n=2,3,4,... are allowed.

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

        output_dtype: tf.DType
            Defaults to `tf.float32`. Defines the output datatype of the layer.

    Input
    -----
        inputs : [...,k], tf.float32
            2+D tensor containing the information bits where `k` is the
            information length.

    Output
    ------
        : [...,k/rate], tf.float32
            2+D tensor containing the encoded codeword for the given input
            information tensor where `rate` is
            :math:`\frac{1}{len\left(\textrm{gen_poly}\right)}`
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

    """

    def __init__(self,
                 gen_poly=None,
                 rate=1/2,
                 constraint_length=3,
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

        self._coderate = 1/len(self.gen_poly)
        self._trellis = Trellis(self.gen_poly,rsc=False)

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
        output_shape[-1] = self._n # assign n to the last dimension

        msg_reshaped = tf.reshape(msg, [-1, self._k])

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
        cw = tf.cast(cw, self.output_dtype)
        cw_reshaped = tf.reshape(cw, output_shape)

        return cw_reshaped

