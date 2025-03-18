#
# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0#
"""Blocks for encoding of linear codes."""

import tensorflow as tf
from sionna.phy import Block
from sionna.phy.fec.utils import pcm2gm, int_mod_2

class LinearEncoder(Block):
    # pylint: disable=line-too-long
    r"""Linear binary encoder for a given generator or parity-check matrix.

    If ``is_pcm`` is `True`,  ``enc_mat`` is interpreted as parity-check
    matrix and internally converted to a corresponding generator matrix.

    Parameters
    ----------
    enc_mat : [k, n] or [n-k, n], ndarray
        Binary generator matrix of shape `[k, n]`. If ``is_pcm`` is
        `True`,  ``enc_mat`` is interpreted as parity-check matrix of shape
        `[n-k, n]`.

    is_pcm: `bool`, (default `False`)
        If `True`,  the `enc_mat` is interpreted as parity-check matrix instead of
        a generator matrix

    precision : `None` (default) | 'single' | 'double'
        Precision used for internal calculations and outputs.
        If set to `None`, :py:attr:`~sionna.phy.config.precision` is used.

    Input
    -----
    info_bits: [...,k], tf.float or tf.int
        Binary tensor containing the information bits.

    Output
    ------
    : [...,n], same dtype as info_bits
        Binary tensor containing codewords with same shape as inputs,
        except the last dimension changes to `[...,n]`.

    Note
    ----
        If ``is_pcm`` is `True`,  this block uses
        :class:`~sionna.phy.fec.utils.pcm2gm` to find the generator matrix for
        encoding. Please note that this imposes a few constraints on the
        provided parity-check matrix such as full rank and it must be binary.

        Note that this encoder is generic for all binary linear block codes
        and, thus, cannot implement any code specific optimizations. As a
        result, the encoding complexity is :math:`O(k^2)`. Please consider code
        specific encoders such as the
        :class:`~sionna.phy.fec.polar.encoding.Polar5GEncoder` or
        :class:`~sionna.phy.fec.ldpc.encoding.LDPC5GEncoder` for an improved
        encoding performance.
    """

    def __init__(self,
                 enc_mat,
                 *,
                 is_pcm=False,
                 precision=None,
                 **kwargs):

        super().__init__(precision=precision, **kwargs)

        # check input values for consistency
        if not isinstance(is_pcm, bool):
            raise TypeError("is_parity_check must be bool.")

        # verify that enc_mat is binary
        if not ((enc_mat==0) | (enc_mat==1)).all():
            raise ValueError("enc_mat is not binary.")
        if not len(enc_mat.shape)==2:
            raise ValueError("enc_mat must be 2-D array.")

        # in case parity-check matrix is provided, convert to generator matrix
        if is_pcm:
            self._gm = pcm2gm(enc_mat, verify_results=True)
        else:
            self._gm = enc_mat
        self._gm = tf.cast(self._gm, dtype=self.rdtype)

        self._k = self._gm.shape[0]
        self._n = self._gm.shape[1]
        self._coderate = self._k / self._n

        if not self._k<=self._n:
            raise AttributeError("Invalid matrix dimensions.")

    ###############################
    # Public methods and properties
    ###############################

    @property
    def k(self):
        """Number of information bits per codeword"""
        return self._k

    @property
    def n(self):
        "Codeword length"
        return self._n

    @property
    def gm(self):
        "Generator matrix used for encoding"
        return self._gm

    @property
    def coderate(self):
        """Coderate of the code"""
        return self._coderate

    def build(self, input_shapes):
        """Nothing to build, but check for valid shapes."""

        if not input_shapes[-1]==self._k:
            raise ValueError(f"Last dimension must be of size k={self._k}.")

    def call(self, bits,/):
        """Generic encoding function based on generator matrix multiplication.
        """

        # add batch_dim if not provided (will be removed afterwards)
        if len(bits.shape)==1:
            bits = tf.expand_dims(bits, axis=0)
            no_batch_dim = True
        else:
            no_batch_dim = False

        # encode via matrix multiplication: c = u*G
        c = tf.linalg.matmul(bits, tf.cast(self._gm, bits.dtype))
        c = int_mod_2(c)

        # remove batch_dim if not desired
        if no_batch_dim:
            c = tf.squeeze(c, axis=0)
        return c
