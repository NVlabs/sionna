#
# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Layer for utility functions needed for Turbo Codes."""

import math
import numpy as np
import tensorflow as tf

def polynomial_selector(constraint_length):
    r"""Returns the generator polynomials for rate-1/2 convolutional codes
    for a given ``constraint_length``.

    Input
    -----
    constraint_length: int
        An integer defining the desired constraint length of the encoder.
        The memory of the encoder is ``constraint_length`` - 1.

    Output
    ------
    gen_poly: tuple
        Tuple of strings with each string being a 0,1 sequence where
        each polynomial is represented in binary form.

    Note
    ----
    Please note that the polynomials are optimized for rsc codes and are
    not necessarily the same as used in the polynomial selector
    :class:`~sionna.fec.conv.utils.polynomial_selector` of the
    convolutional codes.
    """
    assert(isinstance(constraint_length, int)),\
        "constraint_length must be int."
    assert(2 < constraint_length < 7),\
        "Unsupported constraint_length."

    gen_poly_dict = {
            3: ('111', '101'), # (7, 5)
            4: ('1011', '1101'), # (13, 15)
            5: ('10011','11011'), # (23, 33)
            6: ('111101', '101011'), # (75, 53)
    }
    gen_poly = gen_poly_dict[constraint_length]
    return gen_poly


def puncture_pattern(turbo_coderate, conv_coderate):
    r"""This method returns puncturing pattern such that the
    Turbo code has rate ``turbo_coderate`` given the underlying
    convolutional encoder is of rate ``conv_coderate``.

    Input
    -----
    turbo_coderate: float
        Desired coderate of the Turbo code

    conv_coderate: float
        Coderate of the underlying convolutional encoder.

    Output
    ------
    : tf.bool
        2D tensor indicating the positions to be punctured.
    """
    tf.debugging.assert_equal(conv_coderate, 1/2)
    if turbo_coderate == 1/2:
        pattern = [[1, 1, 0], [1, 0, 1]]
    elif turbo_coderate == 1/3:
        pattern = [[1, 1, 1]]
    turbo_punct_pattern = tf.convert_to_tensor(
        np.asarray(pattern), dtype=bool)
    return turbo_punct_pattern


class TurboTermination(object):
    # pylint: disable=line-too-long
    r"""TurboTermination(constraint_length, conv_n=2, num_conv_encs=2, num_bit_streams=3)

    Termination object, handles the transformation of termination bits from
    the convolutional encoders to a Turbo codeword. Similarly, it handles the
    transformation of channel symbols corresponding to the termination of a
    Turbo codeword to the underlying convolutional codewords.

    Parameters
    ----------
    constraint_length: int
        Constraint length of the convolutional encoder used in the Turbo code.
        Note that the memory of the encoder is ``constraint_length`` - 1.

    conv_n: int
        Number of output bits for one state transition in the underlying
        convolutional encoder.

    num_conv_encs: int
        Number of parallel convolutional encoders used in the Turbo code.

    num_bit_streams: int
        Number of output bit streams from Turbo code.
    """

    def __init__(self,
                constraint_length,
                conv_n=2,
                num_conv_encs=2,
                num_bitstreams=3):
        tf.debugging.assert_type(constraint_length, tf.int32)
        tf.debugging.assert_type(conv_n, tf.int32)
        tf.debugging.assert_type(num_conv_encs, tf.int32)
        tf.debugging.assert_type(num_bitstreams, tf.int32)

        self.mu_ = constraint_length - 1
        self.conv_n = conv_n
        tf.debugging.assert_equal(num_conv_encs, 2)
        self.num_conv_encs = num_conv_encs
        self.num_bitstreams = num_bitstreams

    def get_num_term_syms(self):
        r"""
        Computes the number of termination symbols for the Turbo
        code based on the underlying convolutional code parameters,
        primarily the memory :math:`\mu`.
        Note that it is assumed that one Turbo symbol implies
        ``num_bitstreams`` bits.

        Input
        -----
        None

        Output
        ------
        turbo_term_syms: int
            Total number of termination symbols for the Turbo Code. One
            symbol equals ``num_bitstreams`` bits.
        """
        total_term_bits = self.conv_n * self. num_conv_encs * self.mu_
        turbo_term_syms = math.ceil(total_term_bits/self.num_bitstreams)
        return turbo_term_syms

    def termbits_conv2turbo(self, term_bits1, term_bits2):
        # pylint: disable=line-too-long
        r"""
        This method merges ``term_bits1`` and ``term_bits2``, termination
        bit streams from the two convolutional encoders, to a bit stream
        corresponding to the Turbo codeword.

        Let ``term_bits1`` and ``term_bits2`` be:

        :math:`[x_1(K), z_1(K), x_1(K+1), z_1(K+1),..., x_1(K+\mu-1),z_1(K+\mu-1)]`

        :math:`[x_2(K), z_2(K), x_2(K+1), z_2(K+1),..., x_2(K+\mu-1), z_2(K+\mu-1)]`

        where :math:`x_i, z_i` are the systematic and parity bit streams
        respectively for a rate-1/2 convolutional encoder i, for i = 1, 2.

        In the example output below, we assume :math:`\mu=4` to demonstrate zero
        padding at the end. Zero padding is done such that the total length is
        divisible by ``num_bitstreams`` (defaults to  3) which is the number of
        Turbo bit streams.

        Assume ``num_bitstreams`` = 3. Then number of termination symbols for
        the TurboEncoder is :math:`\lceil \frac{2*conv\_n*\mu}{3} \rceil`:

        :math:`[x_1(K), z_1(K), x_1(K+1)]`

        :math:`[z_1(K+1), x_1(K+2, z_1(K+2)]`

        :math:`[x_1(K+3), z_1(K+3), x_2(K)]`

        :math:`[z_2(K), x_2(K+1), z_2(K+1)]`

        :math:`[x_2(K+2), z_2(K+2), x_2(K+3)]`

        :math:`[z_2(K+3), 0, 0]`

        Therefore, the output from this method is a single dimension vector
        where all Turbo symbols are concatenated together.

        :math:`[x_1(K), z_1(K), x_1(K+1), z_1(K+1), x_1(K+2, z_1(K+2), x_1(K+3),`

        :math:`z_1(K+3), x_2(K),z_2(K), x_2(K+1), z_2(K+1), x_2(K+2), z_2(K+2),`

        :math:`x_2(K+3), z_2(K+3), 0, 0]`

        Input
        -----
        term_bits1: tf.int32
            2+D Tensor containing termination bits from convolutional encoder 1.

        term_bits2: tf.int32
            2+D Tensor containing termination bits from convolutional encoder 2.

        Output
        ------
        : tf.int32
            1+D tensor of termination bits. The output is obtained by
            concatenating the inputs and then adding right zero-padding if
            needed.
        """
        term_bits = tf.concat([term_bits1, term_bits2],axis=-1)

        num_term_bits = term_bits.get_shape()[-1]
        num_term_syms = math.ceil(num_term_bits/self.num_bitstreams)

        extra_bits = self.num_bitstreams*num_term_syms - num_term_bits
        if extra_bits > 0:
            zer_shape = tf.stack([tf.shape(term_bits)[0],
                                  tf.constant(extra_bits)],
                                   axis=0)
            term_bits = tf.concat(
                        [term_bits, tf.zeros(zer_shape, tf.int32)], axis=-1)
        return term_bits

    def term_bits_turbo2conv(self, term_bits):
        # pylint: disable=line-too-long
        r"""
        This method splits the termination symbols from a Turbo codeword
        to the termination symbols corresponding to the two convolutional
        encoders, respectively.

        Let's assume :math:`\mu=4` and the underlying convolutional encoders
        are systematic and rate-1/2, for demonstration purposes.

        Let ``term_bits`` tensor, corresponding to the termination symbols of
        the Turbo codeword be as following:

        :math:`y = [x_1(K), z_1(K), x_1(K+1), z_1(K+1), x_1(K+2), z_1(K+2)`,
        :math:`x_1(K+3), z_1(K+3), x_2(K), z_2(K), x_2(K+1), z_2(K+1),`
        :math:`x_2(K+2), z_2(K+2), x_2(K+3), z_2(K+3), 0, 0]`

        The two termination tensors corresponding to the convolutional encoders
        are:
        :math:`y[0,..., 2\mu]`, :math:`y[2\mu,..., 4\mu]`. The output from this method is a tuple of two tensors, each of
        size :math:`2\mu` and shape :math:`[\mu,2]`.

        :math:`[[x_1(K), z_1(K)]`,

        :math:`[x_1(K+1), z_1(K+1)]`,

        :math:`[x_1(K+2, z_1(K+2)]`,

        :math:`[x_1(K+3), z_1(K+3)]]`

        and

        :math:`[[x_2(K), z_2(K)],`

        :math:`[x_2(K+1), z_2(K+1)]`,

        :math:`[x_2(K+2), z_2(K+2)]`,

        :math:`[x_2(K+3), z_2(K+3)]]`

        Input
        -----
        term_bits: tf.float32
            Channel output of the Turbo codeword, corresponding to the
            termination part.

        Output
        ------
        : tf.float32
            Two tensors of channel outputs, corresponding to encoders 1 and 2,
            respectively.
        """
        input_len = tf.shape(term_bits)[-1]
        divisible = tf.math.floormod(input_len, self.num_bitstreams)
        tf.assert_equal(divisible, 0, 'Programming Error.')

        enc1_term_idx = tf.range(0, self.conv_n*self.mu_)
        enc2_term_idx = tf.range(self.conv_n*self.mu_, 2*self.conv_n*self.mu_)

        term_bits1 = tf.gather(term_bits, enc1_term_idx, axis=-1)
        term_bits2 = tf.gather(term_bits, enc2_term_idx, axis=-1)

        return term_bits1, term_bits2

