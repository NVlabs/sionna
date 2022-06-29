#
# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Layers for cyclic redundancy checks (CRC) and utility functions"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer

class CRCEncoder(Layer):
    """CRCEncoder(crc_degree, output_dtype=tf.float32, **kwargs)

    Adds cyclic redundancy check to input sequence.

    The CRC polynomials from Sec. 5.1 in [3GPPTS38212_CRC]_ are available:
    `{CRC24A, CRC24B, CRC24C, CRC16, CRC11, CRC6}`.

    The class inherits from the Keras layer class and can be used as layer in a
    Keras model.

    Parameters
    ----------
        crc_degree: str
            Defining the CRC polynomial to be used. Can be any value from
            `{CRC24A, CRC24B, CRC24C, CRC16, CRC11, CRC6}`.

        dtype: tf.DType
            Defaults to `tf.float32`. Defines the output dtype.

    Input
    -----
        inputs : [...,k], tf.float32
            2+D tensor of arbitrary shape where the last dimension is
            `[...,k]`. Must have at least rank two.

    Output
    ------
        x_crc : [...,k+crc_degree], tf.float32
            2+D tensor containing CRC encoded bits of same shape as
            ``inputs`` except the last dimension changes to
            `[...,k+crc_degree]`.

    Raises
    ------
        AssertionError
            If ``crc_degree`` is not `str`.

        ValueError
            If requested CRC polynomial is not available in [3GPPTS38212_CRC]_.

        InvalidArgumentError
            When rank(``inputs``)<2.

    Note
    ----
        For performance enhancements, we implement a generator-matrix based
        implementation for fixed `k` instead of the more common shift
        register-based operations. Thus, the encoder need to trigger an
        (internal) rebuild if `k` changes.

    """

    def __init__(self, crc_degree, dtype=tf.float32, **kwargs):

        super().__init__(dtype=dtype, **kwargs)

        assert isinstance(crc_degree, str), "crc_degree must be str"
        self._crc_degree = crc_degree

        # init 5G CRC polynomial
        self._crc_pol, self._crc_length = self._select_crc_pol(self._crc_degree)

    #########################################
    # Public methods and properties
    #########################################

    @property
    def crc_degree(self):
        """CRC degree as string."""
        return self._crc_degree

    @property
    def crc_length(self):
        """Length of CRC. Equals number of CRC parity bits."""
        return self._crc_length

    @property
    def crc_pol(self):
        """CRC polynomial in binary representation."""
        return self._crc_pol

    #########################
    # Utility methods
    #########################

    def _select_crc_pol(self, crc_degree):
        """Select 5G CRC polynomial according to Sec. 5.1 [3GPPTS38212_CRC]_."""
        if crc_degree=="CRC24A":
            crc_length = 24
            crc_coeffs = [24, 23, 18, 17, 14, 11, 10, 7, 6, 5, 4, 3, 1, 0]
        elif crc_degree=="CRC24B":
            crc_length = 24
            crc_coeffs = [24, 23, 6, 5, 1, 0]
        elif crc_degree=="CRC24C":
            crc_length = 24
            crc_coeffs = [24, 23, 21, 20, 17, 15, 13, 12, 8, 4, 2, 1, 0]
        elif crc_degree=="CRC16":
            crc_length = 16
            crc_coeffs = [16, 12, 5, 0]
        elif crc_degree=="CRC11":
            crc_length = 11
            crc_coeffs = [11, 10, 9, 5, 0]
        elif crc_degree=="CRC6":
            crc_length = 6
            crc_coeffs = [6, 5, 0]
        else:
            raise ValueError("Invalid CRC Polynomial")

        crc_pol = np.zeros(crc_length+1)
        for c in crc_coeffs:
            crc_pol[c] = 1

        # invert array (MSB instead of LSB)
        crc_pol_inv = np.zeros(crc_length+1)
        for i in range(crc_length+1):
            crc_pol_inv[crc_length-i] = crc_pol[i]

        return crc_pol_inv.astype(int), crc_length

    def _gen_crc_mat(self, k, pol_crc):
        """ Build (dense) generator matrix for CRC parity bits.

        The principle idea is to treat the CRC as systematic linear code, i.e.,
        the generator matrix can be composed out of ``k`` linear independent
        (valid) codewords. For this, we CRC encode all ``k`` unit-vectors
        `[0,...1,...,0]` and build the generator matrix.
        To avoid `O(k^2)` complexity, we start with the last unit vector
        given as `[0,...,0,1]` and can generate the result for next vector
        `[0,...,1,0]` via another polynom division of the remainder from the
        previous result. This allows to successively build the generator matrix
        at linear complexity `O(k)`.
        """
        crc_length = len(pol_crc) - 1
        g_mat = np.zeros([k, crc_length])

        x_crc = np.zeros(crc_length).astype(int)
        x_crc[0] = 1
        for i in range(k):
            # shift by one position
            x_crc = np.concatenate([x_crc, [0]])
            if x_crc[0]==1:
                x_crc = np.bitwise_xor(x_crc, pol_crc)
            x_crc = x_crc[1:]
            g_mat[k-i-1,:] = x_crc

        return g_mat

    #########################
    # Keras layer functions
    #########################

    def build(self, input_shape):
        """Build the generator matrix

        The CRC is always added to the last dimension of the input.
        """
        k = input_shape[-1] # we perform the CRC check on the last dimension
        assert k is not None, "Shape of last dimension cannot be None."
        g_mat_crc = self._gen_crc_mat(k, self.crc_pol)
        self._g_mat_crc = tf.constant(g_mat_crc, dtype=tf.float32)

    def call(self, inputs):
        """cyclic redundancy check function.

        This function add the CRC parity bits to ``inputs`` and returns the
        result of the CRC validation.

        Args:
            inputs (tf.float32): Tensor of arbitrary shape `[...,k]`.
                Must have at least rank two.

        Returns:
            `tf.float32`: CRC encoded bits ``x_crc``of shape
                `[...,k+crc_degree]`.

        Raises:
            InvalidArgumentError: When rank(``x``)<2.

        """

        # assert rank must be two
        tf.debugging.assert_greater(tf.rank(inputs), 1)

        # re-init if shape has changed, update generator matrix
        if inputs.shape[-1] != self._g_mat_crc.shape[0]:
            self.build(inputs.shape)

        # note: as the code is systematic, we only encode the crc positions
        # this the generator matrix is non-sparse and a "full" matrix
        # multiplication is probably the fastest implementation.

        x_exp = tf.expand_dims(inputs, axis=-2) # row vector of shape 1xk

        # tf.matmul onl supports floats (and int32 but not uint8 etc.)
        x_exp32 = tf.cast(x_exp, tf.float32)
        x_crc = tf.matmul(x_exp32, self._g_mat_crc) # calculate crc bits

        # take modulo 2 of x_crc (bitwise operations instead of tf.mod)
        x_crc_uint8 = tf.cast(x_crc, tf.uint8)
        x_bin = tf.bitwise.bitwise_and(x_crc_uint8, tf.constant(1, tf.uint8))
        x_crc = tf.cast(x_bin, dtype=self.dtype)

        x_conc = tf.concat([x_exp, x_crc], -1)
        x_out = tf.squeeze(x_conc, axis=-2)

        return x_out


class CRCDecoder(Layer):
    """CRCDecoder(crc_encoder, dtype=None, **kwargs)

    Allows cyclic redundancy check verification and removes parity-bits.

    The CRC polynomials from Sec. 5.1 in [3GPPTS38212_CRC]_ are available:
    `{CRC24A, CRC24B, CRC24C, CRC16, CRC11, CRC6}`.

    The class inherits from the Keras layer class and can be used as layer in a
    Keras model.

    Parameters
    ----------
        crc_encoder: CRCEncoder
            An instance of :class:`~sionna.fec.crc.CRCEncoder` to which the
            CRCDecoder is associated.

        dtype: tf.DType
            Defaults to `None`. Defines the datatype for internal calculations
            and the output dtype. If no explicit dtype is provided the dtype
            from the associated interleaver is used.

    Input
    -----
        inputs: [...,k+crc_degree], tf.float32
            2+D Tensor containing the CRC encoded bits (i.e., the last
            `crc_degree` bits are parity bits). Must have at least rank two.

    Output
    ------
        (x, crc_valid):
            Tuple:

        x : [...,k], tf.float32
            2+D tensor containing the information bit sequence without CRC
            parity bits.

        crc_valid : [...,1], tf.bool
            2+D tensor containing the result of the CRC per codeword.

    Raises
    ------
        AssertionError
            If ``crc_encoder`` is not `CRCEncoder`.

        InvalidArgumentError
            When rank(``x``)<2.
    """

    def __init__(self,
                 crc_encoder,
                 dtype=tf.float32,
                 **kwargs):

        assert isinstance(crc_encoder, CRCEncoder), \
             "crc_encoder must be an instance of CRCEncoder."
        self._encoder = crc_encoder

        # if dtype is None, use same dtype as associated encoder
        if dtype is None:
            dtype = self._encoder.dtype

        super().__init__(dtype=dtype, **kwargs)

    #########################################
    # Public methods and properties
    #########################################

    @property
    def crc_degree(self):
        """CRC degree as string."""
        return self._encoder.crc_degree

    @property
    def encoder(self):
        """CRC Encoder used for internal validation."""
        return self._encoder

    #########################
    # Keras layer functions
    #########################

    def build(self, input_shape):
        """Nothing to build."""
        pass

    def call(self, inputs):
        """cyclic redundancy check verification

        This function verifies the CRC of ``inputs``. Returns the result of
        the CRC validation and removes parity bits from ``inputs``.

        Args:
            inputs (tf.float32): Tensor of arbitrary shape `[...,k+crc_degree]`.
                Must have at least rank two.

        Returns:
            List(`tf.float32`, `tf.bool`): ``[x, crc_valid]`` list of the
            information bits ``x`` and the result of the parity check
            validation ``crc_valid`` of each codeword, where ``x`` has shape
            `[...,k]` and ``crc_valid`` has shape `[...,1]`.

        Raises:
            InvalidArgumentError: When rank(``inputs``)<2.

        """

        # assert rank must be two
        tf.debugging.assert_greater(tf.rank(inputs), 1)

        # last dim must be at least crc_bits long
        tf.debugging.assert_greater_equal(tf.shape(inputs)[-1],
                                          self._encoder.crc_length)

        # re-encode information bits of x and verify that CRC bits are correct

        x_info = inputs[...,0:-self._encoder.crc_length]
        x_parity = self._encoder(inputs)[...,-self._encoder.crc_length:]

        # return if x fulfils the CRC
        crc_check = tf.reduce_sum(x_parity, axis=-1, keepdims=True)
        crc_check = tf.where(crc_check>0, False, True)

        return [x_info, crc_check]
