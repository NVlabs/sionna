#
# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Classes and functions related to MIMO channel detection"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer
from sionna.utils import expand_to_rank, matrix_sqrt_inv
from sionna.mapping import Constellation, SymbolLogits2LLRs


class MaximumLikelihoodDetector(Layer):
    # pylint: disable=line-too-long
    r"""
    MaximumLikelihoodDetector(output, demapping_method, k, constellation_type=None, num_bits_per_symbol=None, constellation=None, hard_out=False, dtype=tf.complex64, **kwargs)

    MIMO maximum-likelihood (ML) detector.

    This layer implements MIMO maximum-likelihood (ML) detection assuming the
    following channel model:

    .. math::
        \mathbf{y} = \mathbf{H}\mathbf{x} + \mathbf{n}

    where :math:`\mathbf{y}\in\mathbb{C}^M` is the received signal vector,
    :math:`\mathbf{x}\in\mathcal{C}^K` is the vector of transmitted symbols which
    are uniformly and independently drawn from the constellation :math:`\mathcal{C}`,
    :math:`\mathbf{H}\in\mathbb{C}^{M\times K}` is the known channel matrix,
    and :math:`\mathbf{n}\in\mathbb{C}^M` is a complex Gaussian noise vector.
    It is assumed that :math:`\mathbb{E}\left[\mathbf{n}\right]=\mathbf{0}` and
    :math:`\mathbb{E}\left[\mathbf{n}\mathbf{n}^{\mathsf{H}}\right]=\mathbf{S}`,
    where :math:`\mathbf{S}` has full rank.

    Prior to demapping, the received signal is whitened:

    .. math::
        \tilde{\mathbf{y}} &= \mathbf{S}^{-\frac{1}{2}}\mathbf{y}\\
        &=  \mathbf{S}^{-\frac{1}{2}}\mathbf{H}\mathbf{x} + \mathbf{S}^{-\frac{1}{2}}\mathbf{n}\\
        &= \tilde{\mathbf{H}}\mathbf{x} + \tilde{\mathbf{n}}

    The layer can compute ML detection of symbols or bits with either
    soft- or hard-decisions. Note that decisions are computed symbol-/bit-wise
    and not jointly for the entire vector :math:`\textbf{x}` (or the underlying vector
    of bits).

    **\ML detection of bits:**

    Soft-decisions on bits are called log-likelihood ratios (LLR).
    With the “app” demapping method, the LLR for the :math:`i\text{th}` bit
    of the :math:`k\text{th}` user is then computed according to

    .. math::
        LLR(k,i) = \ln\left(\frac{\Pr\left(b_{k,i}=1\lvert \mathbf{y},\mathbf{H}\right)}{\Pr\left(b_{k,i}=0\lvert \mathbf{y},\mathbf{H}\right)}\right) =\ln\left(\frac{
                \sum_{\mathbf{c}\in\mathcal{C}_{k,i,1}} \exp\left(
                    -\left\lVert\tilde{\mathbf{y}}-\tilde{\mathbf{H}}\mathbf{c}\right\rVert^2
                    \right)
                }{
                \sum_{\mathbf{c}\in\mathcal{C}_{k,i,0}} \exp\left(
                    -\left\lVert\tilde{\mathbf{y}}-\tilde{\mathbf{H}}\mathbf{c}\right\rVert^2
                    \right)
                }\right)

    where :math:`\mathcal{C}_{k,i,1}` and :math:`\mathcal{C}_{k,i,0}` are the
    sets of vectors of constellation points for which the :math:`i\text{th}` bit
    of the :math:`k\text{th}` user is equal to 1 and 0, respectively. The definition of the LLR has been
    chosen such that it is equivalent with that of logit. This is
    different from many textbooks in communications, where the LLR is
    defined as :math:`LLR(k,i) = \ln\left(\frac{\Pr\left(b_{k,i}=0\lvert \mathbf{y},\mathbf{H}\right)}{\Pr\left(b_{k,i}=1\lvert \mathbf{y},\mathbf{H}\right)}\right)`.

    With the "maxlog" demapping method, the LLR for the :math:`i\text{th}` bit
    of the :math:`k\text{th}` user is approximated like

    .. math::
        \begin{align}
            LLR(k,i) \approx&\ln\left(\frac{
                \max_{\mathbf{c}\in\mathcal{C}_{k,i,1}} \exp\left(
                    -\left\lVert\tilde{\mathbf{y}}-\tilde{\mathbf{H}}\mathbf{c}\right\rVert^2
                    \right)
                }{
                \max_{\mathbf{c}\in\mathcal{C}_{k,i,0}} \exp\left(
                    -\left\lVert\tilde{\mathbf{y}}-\tilde{\mathbf{H}}\mathbf{c}\right\rVert^2
                    \right)
                }\right)\\
                = &\min_{\mathbf{c}\in\mathcal{C}_{k,i,0}}\left\lVert\tilde{\mathbf{y}}-\tilde{\mathbf{H}}\mathbf{c}\right\rVert^2-
                    \min_{\mathbf{c}\in\mathcal{C}_{k,i,1}}\left\lVert\tilde{\mathbf{y}}-\tilde{\mathbf{H}}\mathbf{c}\right\rVert^2.
            \end{align}

    **ML detection of symbols:**

    Soft-decisions on symbols are called logits (i.e., unnormalized log-probability).

    With the “app” demapping method, the logit for the
    constellation point :math:`c \in \mathcal{C}` of the :math:`k\text{th}` user  is computed according to

    .. math::
        \begin{align}
            \text{logit}(k,c) &= \ln\left(\sum_{\mathbf{x} : x_k = c} \exp\left(
                        -\left\lVert\tilde{\mathbf{y}}-\tilde{\mathbf{H}}\mathbf{x}\right\rVert^2
                        \right)\right)\\
                            &= \ln\left( \Pr\left(x_k = c \lvert \mathbf{y} \right) \right) + C
        \end{align}

    where :math:`C` is a constant.

    With the "maxlog" demapping method, the logit for the constellation point :math:`c \in \mathcal{C}`
    of the :math:`k\text{th}` user  is approximated like

    .. math::
        \text{logit}(k,c) \approx \max_{\mathbf{x} : x_k = c} \left(
                -\left\lVert\tilde{\mathbf{y}}-\tilde{\mathbf{H}}\mathbf{x}\right\rVert^2
                \right).

    When hard decisions are requested, this layer returns for the :math:`k` th stream

    .. math::
        \hat{c}_k = \underset{c \in \mathcal{C}}{\text{argmax}} \Pr\left(x_k = c \lvert \mathbf{y} \right)

    where :math:`\mathcal{C}` is the set of constellation points.
    This is not the same as returning the vector :math:`\hat{\mathbf{x}} = \left[ x_0,\dots,x_{K-1} \right]` such that

    .. math::
        \hat{\mathbf{x}} = \min_{\mathbf{x} \in \mathcal{C}^K} \lVert \mathbf{y} - \mathbf{H}\mathbf{x} \rVert^2.

    Parameters
    -----------
    output : One of ["bit", "symbol"], str
        The type of output, either LLRs on bits or logits on constellation symbols.

    demapping_method : One of ["app", "maxlog"], str
        The demapping method used.

    k : tf.int
        Number of transmit streams.

    constellation_type : One of ["qam", "pam", "custom"], str
        For "custom", an instance of :class:`~sionna.mapping.Constellation`
        must be provided.

    num_bits_per_symbol : int
        The number of bits per constellation symbol, e.g., 4 for QAM16.
        Only required for ``constellation_type`` in ["qam", "pam"].

    constellation : Constellation
        An instance of :class:`~sionna.mapping.Constellation` or `None`.
        In the latter case, ``constellation_type``
        and ``num_bits_per_symbol`` must be provided.

    hard_out : bool
        If `True`, the detector computes hard-decided bit values or
        constellation point indices instead of soft-values.
        Defaults to `False`.

    dtype : One of [tf.complex64, tf.complex128] tf.DType (dtype)
        The dtype of ``y``. Defaults to tf.complex64.
        The output dtype is the corresponding real dtype (tf.float32 or tf.float64).

    Input
    ------
    (y, h, s) :
        Tuple:

    y : [...,M], tf.complex
        1+D tensor containing the received signals.

    h : [...,M,K], tf.complex
        2+D tensor containing the channel matrices.

    s : [...,M,M], tf.complex
        2+D tensor containing the noise covariance matrices.

    Output
    ------
    One of:

    : [..., K, num_bits_per_symbol], tf.float
        LLRs or hard-decisions for every bit of every stream, if ``output`` equals `"bit"`.

    : [..., K, num_points], tf.float or [..., K], tf.int
       Logits or hard-decisions on constellation symbols for every stream, if ``output`` equals `"symbol"`.
       Hard-decisions correspond to the symbol indices.

    Note
    ----
    If you want to use this layer in Graph mode with XLA, i.e., within
    a function that is decorated with ``@tf.function(jit_compile=True)``,
    you must set ``sionna.Config.xla_compat=true``.
    See :py:attr:`~sionna.Config.xla_compat`.
    """

    def __init__(self,
                 output,
                 demapping_method,
                 k,
                 constellation_type=None,
                 num_bits_per_symbol=None,
                 constellation=None,
                 hard_out=False,
                 dtype=tf.complex64,
                 **kwargs):
        super().__init__(dtype=dtype, **kwargs)

        assert dtype in [tf.complex64, tf.complex128],\
            "dtype must be tf.complex64 or tf.complex128"

        assert output in ("bit", "symbol"), "Unknown output"

        assert demapping_method in ("app","maxlog"), "Unknown demapping method"

        self._output = output
        self._demapping_method = demapping_method
        self._hard_out = hard_out

        # Determine the reduce function for LLR computation
        if self._demapping_method == "app":
            self._reduce = tf.reduce_logsumexp
        else:
            self._reduce = tf.reduce_max

        # Create constellation object
        self._constellation = Constellation.create_or_check_constellation(
                                                        constellation_type,
                                                        num_bits_per_symbol,
                                                        constellation)

        # Utility function to compute
        # vecs : [num_vecs, K] The list of all possible transmitted vectors.
        # c : [num_vecs/num_points, K, num_points] Which is such that `c[:,k,s]`
        #   gives the symbol indices in the first dimension of `vecs` for which
        #   the `k`th stream transmitted the `s`th constellation point.
        vecs, c = self._build_vecs(k)
        self._vecs = tf.cast(vecs, dtype)
        self._c = tf.cast(c, tf.int32)

        if output == 'bit':
            self._logits2llr = SymbolLogits2LLRs(
                                    method=demapping_method,
                                    num_bits_per_symbol=num_bits_per_symbol,
                                    hard_out=hard_out,
                                    dtype=dtype.real_dtype,
                                    **kwargs)

    @property
    def constellation(self):
        return self._constellation

    def _build_vecs(self, k):
        """
        Utility function for building the list of all possible transmitted
        vectors of constellation points and the symbol indices corresponding to
        all possibly transmitted constellation points for every stream.

        Input
        ------
        k : int
            Number of transmit streams.

        Output
        -------
        vecs : [num_vecs, K], tf.complex
            List of all possible transmitted vectors.

        c : [num_vecs/num_points, K, num_points], int
            `c[:,k,s]` gives the symbol indices in the first dimension of `vecs`
            for which the `k`th stream transmitted the `s`th symbol.
        """

        points = self._constellation.points

        # Recursive function for generating all possible transmitted
        # vector of symbols.
        # `n` is the remaining number of stream to process
        def _build_vecs_(n):
            if n == 1:
                # If there is a single stream, then the list of possibly
                # transmitted vectors corresponds to the constellation points.
                # No recusrion is needed.
                vecs = np.expand_dims(points, axis=1)
            else:
                # If the number of streams is `n >= 2` streams, then the list
                # of possibly transmitted vectors is
                #
                # [c_1 v , c_2 v, ..., c_N v]
                #
                # where `[c_1, ..., c_N]` is the constellation of size N, and
                # `v` is the list of possible vectors for `n-1` streams.
                # This list has therefore length `N x len(v)`.
                #
                # Building the list for `n-1` streams, recursively.
                v = _build_vecs_(n-1)
                # Building the list of `n` streams by appending the
                # constellation points.
                vecs = []
                for p in points:
                    vecs.append(np.concatenate([np.full([v.shape[0], 1], p),
                                                v], axis=1))
                vecs = np.concatenate(vecs, axis=0)
            return vecs

        # Building the list of possible vectors for the `k` streams.
        # [num_vecs, K]
        vecs = _build_vecs_(k)

        # Compute symbol indices for every stream.
        # For every constellation point `p` and for every stream `j`, we gather
        # the list of vector indices from `vecs` corresponding the vectors for
        # which the `jth` stream transmitted `p`.
        # [num_vecs/num_points, K, num_points]
        c = []
        for p in points:
            c_ = []
            for j in range(k):
                c_.append(np.where(vecs[:,j]==p)[0])
            c_ = np.stack(c_, axis=-1)
            c.append(c_)
        c = np.stack(c, axis=-1)

        return vecs, c

    def call(self, inputs):
        y, h, s = inputs

        # Compute square-root of interference covariance matrix
        s_inv = matrix_sqrt_inv(s)

        # Whiten the observation
        y = tf.expand_dims(y, -1)
        y = tf.squeeze(tf.matmul(s_inv, y), axis=-1)

        # Compute channel after whitening
        h = tf.matmul(s_inv, h)

        # Add extra dims for broadcasting with the dimensions corresponding
        # to all possible transmimtted vectors
        # Shape: [..., 1, M, K]
        h = tf.expand_dims(h, axis=-3)

        # Add extra dims for broadcasting with the dimensions corresponding
        # to all possible transmimtted vectors
        # Shape: [..., 1, M]
        y = tf.expand_dims(y, axis=-2)

        # Reshape list of all possible vectors from
        # [num_vecs, K]
        # to
        # [1,...,1, num_vecs, K, 1]
        vecs = self._vecs
        vecs = tf.expand_dims(vecs, axis=-1)
        vecs = expand_to_rank(vecs, tf.rank(h), 0)

        # Compute exponents
        # [..., num_vecs]
        diff = y - tf.squeeze(h@vecs, axis=-1)
        exponents = -tf.reduce_sum(tf.square(tf.abs(diff)), axis=-1)

        # Gather exponents for all symbols
        # [..., num_vecs/num_points, K, num_points]
        exp = tf.gather(exponents, self._c, axis=-1)

        # Compute logits on constellation points
        # [..., K, num_points]
        logits = self._reduce(exp, axis=-3)

        if self._output == 'bit':
            # Compute LLRs or hard decisions
            return self._logits2llr(logits)
        else:
            if self._hard_out:
                return tf.argmax(logits, axis=-1)
            else:
                return logits
