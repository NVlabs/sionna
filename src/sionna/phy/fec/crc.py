#
# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0#
"""Blocks for cyclic redundancy checks (CRC) and utility functions"""

import numpy as np
import tensorflow as tf
from sionna.phy import Block
from sionna.phy.fec.utils import int_mod_2

class CRCEncoder(Block):
    """Adds a Cyclic Redundancy Check (CRC) to the input sequence.

    The CRC polynomials from Sec. 5.1 in [3GPPTS38212_CRC]_ are available:
    `{CRC24A, CRC24B, CRC24C, CRC16, CRC11, CRC6}`.

    Parameters
    ----------
    crc_degree: str, 'CRC24A' | 'CRC24B' | 'CRC24C' | 'CRC16' | 'CRC11' | 'CRC6'
        Defines the CRC polynomial to be used. Can be any value from
        `{CRC24A, CRC24B, CRC24C, CRC16, CRC11, CRC6}`.

    precision : `None` (default) | 'single' | 'double'
        Precision used for internal calculations and outputs.
        If set to `None`, :py:attr:`~sionna.phy.config.precision` is used.

    Input
    -----
    bits : [...,k], tf.float
        Binary tensor of arbitrary shape where the last dimension is
        `[...,k]`.

    Output
    ------
    x_crc : [...,k+crc_degree], tf.float
        Binary tensor containing CRC-encoded bits of the same shape as
        ``inputs`` except the last dimension changes to
        `[...,k+crc_degree]`.

    Note
    ----
        For performance enhancements, a generator-matrix-based
        implementation is used for fixed `k` instead of the more common shift
        register-based operations. Thus, the encoder must trigger an
        (internal) rebuild if `k` changes.
    """

    def __init__(self, crc_degree, *, precision=None, **kwargs):

        super().__init__(precision=precision, **kwargs)

        assert isinstance(crc_degree, str), "crc_degree must be a string."
        self._crc_degree = crc_degree

        # init 5G CRC polynomial
        self._crc_pol, self._crc_length = self._select_crc_pol(self._crc_degree)

        self._k = None
        self._n = None

    #########################################
    # Public methods and properties
    #########################################

    @property
    def crc_degree(self):
        """CRC degree as string"""
        return self._crc_degree

    @property
    def crc_length(self):
        """Length of CRC. Equals number of CRC parity bits."""
        return self._crc_length

    @property
    def crc_pol(self):
        """CRC polynomial in binary representation"""
        return self._crc_pol

    @property
    def k(self):
        """Number of information bits per codeword"""
        if self._k is None:
            print("Warning: CRC encoder is not initialized yet."\
                  "Input dimensions are unknown.")
        return self._k

    @property
    def n(self):
        """Number of codeword bits after CRC encoding."""
        if self._n is None:
            print("Warning: CRC encoder is not initialized yet."\
                  "Output dimensions are unknown.")
        return self._n

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

        # invert array (MSB instead of LSB)
        crc_pol_inv = np.zeros(crc_length + 1)
        crc_pol_inv[[crc_length - c for c in crc_coeffs]] = 1

        return crc_pol_inv.astype(int), crc_length

    def _gen_crc_mat(self, k, pol_crc):
        """ Build (dense) generator matrix for CRC parity bits.

        The principle idea is to treat the CRC as systematic linear code, i.e.,
        the generator matrix can be composed out of ``k`` linear independent
        (valid) codewords. For this, we CRC encode all ``k`` unit-vectors
        `[0,...1,...,0]` and build the generator matrix.
        To avoid `O(k^2)` complexity, we start with the last unit vector
        given as `[0,...,0,1]` and can generate the result for next vector
        `[0,...,1,0]` via another polynomial division of the remainder from the
        previous result. This allows to successively build the generator matrix
        at linear complexity `O(k)`.
        """
        crc_length = len(pol_crc) - 1
        g_mat = np.zeros([k, crc_length])

        x_crc = np.zeros(crc_length, dtype=int)
        x_crc[0] = 1
        for i in range(k):
            # shift by one position
            x_crc = np.concatenate([x_crc, [0]])
            if x_crc[0]==1:
                x_crc = np.bitwise_xor(x_crc, pol_crc)
            x_crc = x_crc[1:]
            g_mat[k-i-1,:] = x_crc

        return g_mat

    ########################
    # Sionna Block functions
    ########################

    def build(self, input_shape):
        """Build the generator matrix

        The CRC is always added to the last dimension of the input.
        """
        k = input_shape[-1] # we perform the CRC check on the last dimension
        assert k is not None, "Shape of last dimension cannot be None."
        g_mat_crc = self._gen_crc_mat(k, self.crc_pol)
        self._g_mat_crc = tf.constant(g_mat_crc, dtype=self.rdtype)

        self._k = k
        self._n = k + g_mat_crc.shape[1]

    def call(self, bits, /):
        """Cyclic Redundancy Check (CRC) function.

        This function adds the CRC parity bits to ``inputs`` and returns the
        result of the CRC validation.

        Args:
            bits (tf.float): Tensor of arbitrary shape `[...,k]`.

        Returns:
            `tf.float`: CRC encoded bits ``x_crc`` of shape
                `[...,k+crc_degree]`.

        """

        # re-init if shape has changed, update generator matrix
        if bits.shape[-1] != self._g_mat_crc.shape[0]:
            self.build(bits.shape)

        # note: as the code is systematic, we only encode the crc positions
        # thus, the generator matrix is non-sparse and a "full" matrix
        # multiplication is probably the fastest TF implementation.
        x_exp = tf.expand_dims(bits, axis=-2) # row vector of shape 1xk

        # tf.matmul only supports floats (and int32 but not uint8 etc.)
        x_crc = tf.matmul(tf.cast(x_exp, self.rdtype),
                          self._g_mat_crc) # calculate crc bits

        # take modulo 2 of x_crc (bitwise operations instead of tf.mod)

        # cast to tf.int64 first as TF 2.15 has an XLA bug with casting directly
        # to tf.int32
        x_crc = tf.cast(x_crc, dtype=tf.int64)
        x_crc = int_mod_2(x_crc)
        # cast back to original dtype (to support also int8 inputs etc.)
        x_crc = tf.cast(x_crc, dtype=x_exp.dtype)

        x_conc = tf.concat([x_exp, x_crc], -1)
        x_out = tf.squeeze(x_conc, axis=-2)

        return x_out


class CRCDecoder(Block):
    # pylint: disable=line-too-long
    """Allows Cyclic Redundancy Check (CRC) verification and removes parity bits.

    The CRC polynomials from Sec. 5.1 in [3GPPTS38212_CRC]_ are available:
    `{CRC24A, CRC24B, CRC24C, CRC16, CRC11, CRC6}`.

    Parameters
    ----------
    crc_encoder: CRCEncoder
        An instance of :class:`~sionna.phy.fec.crc.CRCEncoder` associated with
        the CRCDecoder.

    precision : `None` (default) | 'single' | 'double'
        Precision used for internal calculations and outputs.
        If set to `None`, :py:attr:`~sionna.phy.config.precision` is used.

    Input
    -----
    x_crc: [...,k+crc_degree], tf.float
        Binary tensor containing the CRC-encoded bits (the last
        `crc_degree` bits are parity bits).

    Output
    ------
    bits : [...,k], tf.float
        Binary tensor containing the information bit sequence without CRC
        parity bits.

    crc_valid : [...,1], tf.bool
        Boolean tensor containing the result of the CRC check per codeword.

    """

    def __init__(self, crc_encoder, *, precision=None, **kwargs):

        super().__init__(precision=precision, **kwargs)

        assert isinstance(crc_encoder, CRCEncoder), \
                "crc_encoder must be a CRCEncoder instance."
        self._encoder = crc_encoder

        # to detect changing input dimensions
        self._bit_shape = None

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

    ########################
    # Sionna Block functions
    ########################

    def build(self, input_shape):
        """Nothing to build but check shapes."""
        self._bit_shape = input_shape
        if input_shape[-1] < self._encoder.crc_length:
            msg ="Input length must be greater than or equal to the CRC length."
            raise ValueError(msg)


    def call(self, x_crc, /):
        """Cyclic Redundancy Check (CRC) verification function.

        This function verifies the CRC of ``x_crc``. It returns the result of
        the CRC validation and removes parity bits from ``x_crc``.

        Args:
            x_crc (tf.float32): Tensor of arbitrary shape `[...,k+crc_degree]`.

        Returns:
            Tuple [`tf.float32`, `tf.bool`]: ``[x_info, crc_valid]`` list, where
            ``x_info`` contains the information bits without CRC parity bits,
            of shape `[...,k]`, and ``crc_valid`` contains the result of the CRC
            validation for each codeword, of shape `[...,1]`.

        """

        if x_crc.shape[-1] != self._bit_shape:
            self.build(x_crc.shape)

        # re-encode information bits of x and verify that CRC bits are correct
        x_info = x_crc[...,0:-self._encoder.crc_length]
        x_parity = self._encoder(x_crc)[...,-self._encoder.crc_length:]

        # cast output to desired precision as encoder can have a different
        # precision
        x_parity = tf.cast(x_parity, self.rdtype)

        # return if x fulfils the CRC
        crc_check = tf.reduce_sum(x_parity, axis=-1, keepdims=True)
        crc_check = tf.where(crc_check>0, False, True)

        return x_info, crc_check
