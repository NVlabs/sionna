#
# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Layers for encoding of linear codes."""

import tensorflow as tf
from tensorflow.keras.layers import Layer
from sionna.fec.utils import pcm2gm
import numbers # to check if n, k are numbers

class LinearEncoder(Layer):
    # pylint: disable=line-too-long
    r"""LinearEncoder(enc_mat, is_pcm=False, dtype=tf.float32, **kwargs)

    Linear binary encoder for a given generator or parity-check matrix ``enc_mat``.

    If ``is_pcm`` is True, ``enc_mat`` is interpreted as parity-check
    matrix and internally converted to a corresponding generator matrix.

    The class inherits from the Keras layer class and can be used as layer in a
    Keras model.

    Parameters
    ----------
    enc_mat : [k, n] or [n-k, n], ndarray
        Binary generator matrix of shape `[k, n]`. If ``is_pcm`` is
        True, ``enc_mat`` is interpreted as parity-check matrix of shape
        `[n-k, n]`.

    dtype: tf.DType
        Defaults to `tf.float32`. Defines the datatype for the output dtype.

    Input
    -----
    inputs: [...,k], tf.float32
        2+D tensor containing information bits.

    Output
    ------
    : [...,n], tf.float32
        2+D tensor containing codewords with same shape as inputs, except the
        last dimension changes to `[...,n]`.

    Raises
    ------
    AssertionError
        If the encoding matrix is not a valid binary 2-D matrix.

    Note
    ----
        If ``is_pcm`` is True, this layer uses
        :class:`~sionna.fec.utils.pcm2gm` to find the generator matrix for
        encoding. Please note that this imposes a few constraints on the
        provided parity-check matrix such as full rank and it must be binary.

        Note that this encoder is generic for all binary linear block codes
        and, thus, cannot implement any code specific optimizations. As a
        result, the encoding complexity is :math:`O(k^2)`. Please consider code
        specific encoders such as the
        :class:`~sionna.fec.polar.encoding.Polar5GEncoder` or
        :class:`~sionna.fec.ldpc.encoding.LDPC5GEncoder` for an improved
        encoding performance.
    """

    def __init__(self,
                 enc_mat,
                 is_pcm=False,
                 dtype=tf.float32,
                 **kwargs):

        super().__init__(dtype=dtype, **kwargs)

        # tf.int8 currently not supported by tf.matmult
        assert (dtype in
               (tf.float16, tf.float32, tf.float64, tf.int32, tf.int64)), \
               "Unsupported dtype."

        # check input values for consistency
        assert isinstance(is_pcm, bool), \
                                    'is_parity_check must be bool.'

        # verify that enc_mat is binary
        assert ((enc_mat==0) | (enc_mat==1)).all(), "enc_mat is not binary."
        assert (len(enc_mat.shape)==2), "enc_mat must be 2-D array."

        # in case parity-check matrix is provided, convert to generator matrix
        if is_pcm:
            self._gm = pcm2gm(enc_mat, verify_results=True)
        else:
            self._gm = enc_mat

        self._k = self._gm.shape[0]
        self._n = self._gm.shape[1]
        self._coderate = self._k / self._n

        assert (self._k<=self._n), "Invalid matrix dimensions."

        self._gm = tf.cast(self._gm, dtype=self.dtype)

    #########################################
    # Public methods and properties
    #########################################

    @property
    def k(self):
        """Number of information bits per codeword."""
        return self._k

    @property
    def n(self):
        "Codeword length."
        return self._n

    @property
    def gm(self):
        "Generator matrix used for encoding."
        return self._gm

    @property
    def coderate(self):
        """Coderate of the code."""
        return self._coderate

    #########################
    # Keras layer functions
    #########################

    def build(self, input_shape):
        """Nothing to build, but check for valid shapes."""
        assert input_shape[-1]==self._k, "Invalid input shape."
        assert (len(input_shape)>=2), 'The inputs must have at least rank 2.'

    def call(self, inputs):
        """Generic encoding function based on generator matrix multiplication.
        """

        c = tf.linalg.matmul(inputs, self._gm)

        # faster implementation of tf.math.mod(c, 2)
        c_uint8 = tf.cast(c, tf.uint8)
        c_bin = tf.bitwise.bitwise_and(c_uint8, tf.constant(1, tf.uint8))
        c = tf.cast(c_bin, self.dtype)

        return c

class AllZeroEncoder(Layer):
    r"""AllZeroEncoder(k, n, dtype=tf.float32, **kwargs)
    Dummy encoder that always outputs the all-zero codeword of length ``n``.

    Note that this encoder is a dummy encoder and does NOT perform real
    encoding!

    The class inherits from the Keras layer class and can be used as layer in a
    Keras model.

    Parameters
    ----------
        k: int
            Defining the number of information bit per codeword.

        n: int
            Defining the desired codeword length.

        dtype: tf.DType
            Defaults to `tf.float32`. Defines the datatype for internal
            calculations and the output dtype.

    Input
    -----
        inputs: [...,k], tf.float32
            2+D tensor containing arbitrary values (not used!).

    Output
    ------
        : [...,n], tf.float32
            2+D tensor containing all-zero codewords.

    Raises
    ------
        AssertionError
            ``k`` and ``n`` must be positive integers and ``k`` must be smaller
            (or equal) than ``n``.

    Note
    ----
        As the all-zero codeword is part of any linear code, it is often used
        to simulate BER curves of arbitrary (LDPC) codes without the need of
        having access to the actual generator matrix. However, this `"all-zero
        codeword trick"` requires symmetric channels (such as BPSK), otherwise
        scrambling is required (cf. [Pfister]_ for further details).

        This encoder is a dummy encoder that is needed for some all-zero
        codeword simulations independent of the input. It does NOT perform
        real encoding although the information bits are taken as input.
        This is just to ensure compatibility with other encoding layers.
    """

    def __init__(self,
                 k,
                 n,
                 dtype=tf.float32,
                 **kwargs):

        super().__init__(dtype=dtype, **kwargs)

        #assert error if r>1 or k,n are negativ
        assert isinstance(k, numbers.Number), "k must be a number."
        assert isinstance(n, numbers.Number), "n must be a number."
        k = int(k) # k or n can be float (e.g. as result of n=k*r)
        n = int(n) # k or n can be float (e.g. as result of n=k*r)
        assert k>-1, "k cannot be negative."
        assert n>-1, "n cannot be negative."
        assert n>=k, "Invalid coderate (>1)."
        # init encoder parameters
        self._k = k
        self._n = n
        self._coderate = k / n

    #########################################
    # Public methods and properties
    #########################################

    @property
    def k(self):
        """Number of information bits per codeword."""
        return self._k

    @property
    def n(self):
        "Codeword length."
        return self._n

    @property
    def coderate(self):
        """Coderate of the LDPC code."""
        return self._coderate

    #########################
    # Keras layer functions
    #########################

    def build(self, input_shape):
        """Nothing to build."""
        pass

    def call(self, inputs):
        """Encoding function that outputs the all-zero codeword.

        This function returns the all-zero codeword of shape `[..., n]`.
        Note that this encoder is a dummy encoder and does NOT perform real
        encoding!

        Args:
            inputs (tf.float32): Tensor of arbitrary shape.

        Returns:
            `tf.float32`: Tensor of shape `[...,n]`.

        Note:
            This encoder is a dummy encoder that is needed for some all-zero
            codeword simulations independent of the input. It does NOT perform
            real encoding although the information bits are taken as input.
            This is just to ensure compatibility with other encoding layers.
        """
        # keep shape of first dimensions
        # return an all-zero tensor of shape [..., n]
        output_shape = tf.concat([tf.shape(inputs)[:-1],
                                  tf.constant(self._n, shape=[1])],
                                  0)
        c = tf.zeros(output_shape, dtype=super().dtype)
        return c
