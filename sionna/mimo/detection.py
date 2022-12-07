#
# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Classes and functions related to MIMO channel detection"""

import warnings
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer
from sionna.utils import expand_to_rank, matrix_sqrt_inv, flatten_last_dims, flatten_dims, split_dim, insert_dims, hard_decisions
from sionna.mapping import Constellation, SymbolLogits2LLRs, LLRs2SymbolLogits, PAM2QAM, Demapper, SymbolDemapper, SymbolInds2Bits, DemapperWithPrior, SymbolLogits2Moments
from sionna.mimo.utils import complex2real_channel, whiten_channel, List2LLR, List2LLRSimple, complex2real_matrix, complex2real_vector, real2complex_vector
from sionna.mimo.equalization import lmmse_equalizer, zf_equalizer, mf_equalizer

class LinearDetector(Layer):
    # pylint: disable=line-too-long
    r"""LinearDetector(equalizer, output, demapping_method, constellation_type=None, num_bits_per_symbol=None, constellation=None, hard_out=False, dtype=tf.complex64, **kwargs)

    Convenience class that combines an equalizer,
    such as :func:`~sionna.mimo.lmmse_equalizer`, and a :class:`~sionna.mapping.Demapper`.

    Parameters
    ----------
    equalizer : str, one of ["lmmse", "zf", "mf"], or an equalizer function
        The equalizer to be used. Either one of the existing equalizers
        :func:`~sionna.mimo.lmmse_equalizer`, :func:`~sionna.mimo.zf_equalizer`, or
        :func:`~sionna.mimo.mf_equalizer` can be used, or a custom equalizer
        callable provided that has the same input/output specification.

    output : One of ["bit", "symbol"], str
        The type of output, either LLRs on bits or logits on constellation symbols.

    demapping_method : One of ["app", "maxlog"], str
        The demapping method used.

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
        1+D tensor containing the received signals

    h : [...,M,num_streams], tf.complex
        2+D tensor containing the channel matrices

    s : [...,M,M], tf.complex
        2+D tensor containing the noise covariance matrices

    Output
    ------
    One of:

    : [..., num_streams, num_bits_per_symbol], tf.float
        LLRs or hard-decisions for every bit of every stream, if ``output`` equals `"bit"`

    : [..., num_streams, num_points], tf.float or [..., num_streams], tf.int
       Logits or hard-decisions for constellation symbols for every stream, if ``output`` equals `"symbol"`
       Hard-decisions correspond to the symbol indices.

    Note
    ----
    If you want to use this layer in Graph mode with XLA, i.e., within
    a function that is decorated with ``@tf.function(jit_compile=True)``,
    you might need to set ``sionna.Config.xla_compat=true``. This depends on the
    chosen equalizer function. See :py:attr:`~sionna.Config.xla_compat`.
    """
    def __init__(self,
                 equalizer,
                 output,
                 demapping_method,
                 constellation_type=None,
                 num_bits_per_symbol=None,
                 constellation=None,
                 hard_out=False,
                 dtype=tf.complex64,
                 **kwargs):
        super().__init__(dtype=dtype, **kwargs)
        self._output = output
        self._hard_out = hard_out

        # Determine the equalizer to use
        if isinstance(equalizer, str):
            assert equalizer in ["lmmse", "zf", "mf"], "Unknown equalizer."
            if equalizer=="lmmse":
                self._equalizer = lmmse_equalizer
            elif equalizer=="zf":
                self._equalizer = zf_equalizer
            else:
                self._equalizer = mf_equalizer
        else:
            self._equalizer = equalizer

        assert output in ("bit", "symbol"), "Unknown output"
        assert demapping_method in ("app","maxlog"), "Unknown demapping method"

        constellation = Constellation.create_or_check_constellation(
                                                            constellation_type,
                                                            num_bits_per_symbol,
                                                            constellation,
                                                            dtype=dtype)
        self._constellation = constellation

        # Determine the demapper to use
        if output=="bit":
            self._demapper = Demapper(demapping_method,
                                      constellation=constellation,
                                      hard_out=hard_out,
                                      dtype=dtype)
        else:
            self._demapper = SymbolDemapper(constellation=constellation,
                                            hard_out=hard_out,
                                            dtype=dtype)

    def call(self, inputs):
        x_hat, no_eff = self._equalizer(*inputs)
        z = self._demapper([x_hat, no_eff])

        # Reshape to the expected output shape
        num_streams = tf.shape(inputs[1])[-1]
        if self._output == 'bit':
            num_bits_per_symbol = self._constellation.num_bits_per_symbol
            z = split_dim(z, [num_streams, num_bits_per_symbol], tf.rank(z)-1)

        return z

class MaximumLikelihoodDetector(Layer):
    # pylint: disable=line-too-long
    r"""
    MaximumLikelihoodDetector(output, demapping_method, num_streams, constellation_type=None, num_bits_per_symbol=None, constellation=None, hard_out=False, with_prior=False, dtype=tf.complex64, **kwargs)

    MIMO maximum-likelihood (ML) detector.
    If the ``with_prior`` flag is set, prior knowledge on the bits or constellation points is assumed to be available.

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
    If the ``with_prior`` flag is set, it is assumed that prior information of the transmitted signal :math:`\mathbf{x}` is available,
    provided either as LLRs on the bits mapped onto :math:`\mathbf{x}` or as logits on the individual
    constellation points forming :math:`\mathbf{x}`.

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
        \begin{align}
            LLR(k,i)&= \ln\left(\frac{\Pr\left(b_{k,i}=1\lvert \mathbf{y},\mathbf{H}\right)}{\Pr\left(b_{k,i}=0\lvert \mathbf{y},\mathbf{H}\right)}\right)\\
                    &=\ln\left(\frac{
                    \sum_{\mathbf{x}\in\mathcal{C}_{k,i,1}} \exp\left(
                        -\left\lVert\tilde{\mathbf{y}}-\tilde{\mathbf{H}}\mathbf{x}\right\rVert^2
                        \right) \Pr\left( \mathbf{x} \right)
                    }{
                    \sum_{\mathbf{x}\in\mathcal{C}_{k,i,0}} \exp\left(
                        -\left\lVert\tilde{\mathbf{y}}-\tilde{\mathbf{H}}\mathbf{x}\right\rVert^2
                        \right) \Pr\left( \mathbf{x} \right)
                    }\right)
        \end{align}

    where :math:`\mathcal{C}_{k,i,1}` and :math:`\mathcal{C}_{k,i,0}` are the
    sets of vectors of constellation points for which the :math:`i\text{th}` bit
    of the :math:`k\text{th}` user is equal to 1 and 0, respectively.
    :math:`\Pr\left( \mathbf{x} \right)` is the prior distribution of the vector of
    constellation points :math:`\mathbf{x}`. Assuming that the constellation points and
    bit levels are independent, it is computed from the prior of the bits according to

    .. math::
        \Pr\left( \mathbf{x} \right) = \prod_{k=1}^K \prod_{i=1}^{I} \sigma \left( LLR_p(k,i) \right)

    where :math:`LLR_p(k,i)` is the prior knowledge of the :math:`i\text{th}` bit of the
    :math:`k\text{th}` user given as an LLR and which is set to :math:`0` if no prior knowledge is assumed to be available,
    and :math:`\sigma\left(\cdot\right)` is the sigmoid function.
    The definition of the LLR has been chosen such that it is equivalent with that of logit. This is
    different from many textbooks in communications, where the LLR is
    defined as :math:`LLR(k,i) = \ln\left(\frac{\Pr\left(b_{k,i}=0\lvert \mathbf{y},\mathbf{H}\right)}{\Pr\left(b_{k,i}=1\lvert \mathbf{y},\mathbf{H}\right)}\right)`.

    With the "maxlog" demapping method, the LLR for the :math:`i\text{th}` bit
    of the :math:`k\text{th}` user is approximated like

    .. math::
        \begin{align}
            LLR(k,i) \approx&\ln\left(\frac{
                \max_{\mathbf{x}\in\mathcal{C}_{k,i,1}} \left( \exp\left(
                    -\left\lVert\tilde{\mathbf{y}}-\tilde{\mathbf{H}}\mathbf{x}\right\rVert^2
                    \right) \Pr\left( \mathbf{x} \right) \right)
                }{
                \max_{\mathbf{x}\in\mathcal{C}_{k,i,0}} \left( \exp\left(
                    -\left\lVert\tilde{\mathbf{y}}-\tilde{\mathbf{H}}\mathbf{x}\right\rVert^2
                    \right) \Pr\left( \mathbf{x} \right) \right)
                }\right)\\
                = &\min_{\mathbf{x}\in\mathcal{C}_{k,i,0}} \left( \left\lVert\tilde{\mathbf{y}}-\tilde{\mathbf{H}}\mathbf{x}\right\rVert^2 - \ln \left(\Pr\left( \mathbf{x} \right) \right) \right) -
                    \min_{\mathbf{x}\in\mathcal{C}_{k,i,1}} \left( \left\lVert\tilde{\mathbf{y}}-\tilde{\mathbf{H}}\mathbf{x}\right\rVert^2 - \ln \left( \Pr\left( \mathbf{x} \right) \right) \right).
            \end{align}

    **ML detection of symbols:**

    Soft-decisions on symbols are called logits (i.e., unnormalized log-probability).

    With the “app” demapping method, the logit for the
    constellation point :math:`c \in \mathcal{C}` of the :math:`k\text{th}` user  is computed according to

    .. math::
        \begin{align}
            \text{logit}(k,c) &= \ln\left(\sum_{\mathbf{x} : x_k = c} \exp\left(
                        -\left\lVert\tilde{\mathbf{y}}-\tilde{\mathbf{H}}\mathbf{x}\right\rVert^2
                        \right)\Pr\left( \mathbf{x} \right)\right).
        \end{align}

    With the "maxlog" demapping method, the logit for the constellation point :math:`c \in \mathcal{C}`
    of the :math:`k\text{th}` user  is approximated like

    .. math::
        \text{logit}(k,c) \approx \max_{\mathbf{x} : x_k = c} \left(
                -\left\lVert\tilde{\mathbf{y}}-\tilde{\mathbf{H}}\mathbf{x}\right\rVert^2 + \ln \left( \Pr\left( \mathbf{x} \right) \right)
                \right).

    When hard decisions are requested, this layer returns for the :math:`k` th stream

    .. math::
        \hat{c}_k = \underset{c \in \mathcal{C}}{\text{argmax}} \left( \sum_{\mathbf{x} : x_k = c} \exp\left(
                        -\left\lVert\tilde{\mathbf{y}}-\tilde{\mathbf{H}}\mathbf{x}\right\rVert^2
                        \right)\Pr\left( \mathbf{x} \right) \right)

    where :math:`\mathcal{C}` is the set of constellation points.

    Parameters
    -----------
    output : One of ["bit", "symbol"], str
        The type of output, either LLRs on bits or logits on constellation symbols.

    demapping_method : One of ["app", "maxlog"], str
        The demapping method used.

    num_streams : tf.int
        Number of transmitted streams

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

    with_prior : bool
        If `True`, it is assumed that prior knowledge on the bits or constellation points is available.
        This prior information is given as LLRs (for bits) or log-probabilities (for constellation points) as an
        additional input to the layer.
        Defaults to `False`.

    dtype : One of [tf.complex64, tf.complex128] tf.DType (dtype)
        The dtype of ``y``. Defaults to tf.complex64.
        The output dtype is the corresponding real dtype (tf.float32 or tf.float64).

    Input
    ------
    (y, h, s) or (y, h, prior, s) :
        Tuple:

    y : [...,M], tf.complex
        1+D tensor containing the received signals.

    h : [...,M,num_streams], tf.complex
        2+D tensor containing the channel matrices.

    prior : [...,num_streams,num_bits_per_symbol] or [...,num_streams,num_points], tf.float
        Prior of the transmitted signals.
        If ``output`` equals "bit", then LLRs of the transmitted bits are expected.
        If ``output`` equals "symbol", then logits of the transmitted constellation points are expected.
        Only required if the ``with_prior`` flag is set.

    s : [...,M,M], tf.complex
        2+D tensor containing the noise covariance matrices.

    Output
    ------
    One of:

    : [..., num_streams, num_bits_per_symbol], tf.float
        LLRs or hard-decisions for every bit of every stream, if ``output`` equals `"bit"`.

    : [..., num_streams, num_points], tf.float or [..., num_streams], tf.int
       Logits or hard-decisions for constellation symbols for every stream, if ``output`` equals `"symbol"`.
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
                 num_streams,
                 constellation_type=None,
                 num_bits_per_symbol=None,
                 constellation=None,
                 hard_out=False,
                 with_prior=False,
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
        self._with_prior = with_prior

        # Determine the reduce function for LLR computation
        if self._demapping_method == "app":
            self._reduce = tf.reduce_logsumexp
        else:
            self._reduce = tf.reduce_max

        # Create constellation object
        self._constellation = Constellation.create_or_check_constellation(
                                                        constellation_type,
                                                        num_bits_per_symbol,
                                                        constellation,
                                                        dtype=dtype)

        # Utility function to compute
        # vecs : [num_vecs, num_streams] The list of all possible transmitted vectors.
        # vecs_ind : [num_vecs, num_streams] The list of all possible transmitted vectors
        #   constellation indices
        # c : [num_vecs/num_points, num_streams, num_points] Which is such that `c[:,k,s]`
        #   gives the symbol indices in the first dimension of `vecs` for which
        #   the `k`th stream transmitted the `s`th constellation point.
        vecs, vecs_ind, c = self._build_vecs(num_streams)
        self._vecs = tf.cast(vecs, dtype)
        self._vecs_ind = tf.cast(vecs_ind, tf.int32)
        self._c = tf.cast(c, tf.int32)

        if output == 'bit':
            num_bits_per_symbol = self._constellation.num_bits_per_symbol
            self._logits2llr = SymbolLogits2LLRs(
                                    method=demapping_method,
                                    num_bits_per_symbol=num_bits_per_symbol,
                                    hard_out=hard_out,
                                    dtype=dtype.real_dtype,
                                    **kwargs)
            self._llrs2logits = LLRs2SymbolLogits(
                                    num_bits_per_symbol=num_bits_per_symbol,
                                    hard_out=False,
                                    dtype=dtype.real_dtype,
                                    **kwargs)

    @property
    def constellation(self):
        return self._constellation

    def _build_vecs(self, num_streams):
        """
        Utility function for building the list of all possible transmitted
        vectors of constellation points and the symbol indices corresponding to
        all possibly transmitted constellation points for every stream.

        Input
        ------
        num_streams : int
            Number of transmitted streams

        Output
        -------
        vecs : [num_vecs, K], tf.complex
            List of all possible transmitted vectors.

        c : [num_vecs/num_points, num_streams, num_points], int
            `c[:,k,s]` gives the symbol indices in the first dimension of `vecs`
            for which the `k`th stream transmitted the `s`th symbol.
        """

        points = self._constellation.points
        num_points = points.shape[0]

        # Recursive function for generating all possible transmitted
        # vector of symbols and indices
        # `n` is the remaining number of stream to process
        def _build_vecs_(n):
            if n == 1:
                # If there is a single stream, then the list of possibly
                # transmitted vectors corresponds to the constellation points.
                # No recusrion is needed.
                vecs = np.expand_dims(points, axis=1)
                vecs_ind = np.expand_dims(np.arange(num_points), axis=1)
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
                v, vi = _build_vecs_(n-1)
                # Building the list of `n` streams by appending the
                # constellation points.
                vecs = []
                vecs_ind = []
                for i,p in enumerate(points):
                    vecs.append(np.concatenate([np.full([v.shape[0], 1], p),
                                                v], axis=1))
                    vecs_ind.append(np.concatenate([np.full([v.shape[0], 1], i),
                                                vi], axis=1))
                vecs = np.concatenate(vecs, axis=0)
                vecs_ind = np.concatenate(vecs_ind, axis=0)
            return vecs, vecs_ind

        # Building the list of possible vectors for the `k` streams.
        # [num_vecs, K]
        vecs, vecs_ind = _build_vecs_(num_streams)

        tx_ind = np.arange(num_streams)
        tx_ind = np.expand_dims(tx_ind, axis=0)
        tx_ind = np.tile(tx_ind, [vecs_ind.shape[0], 1])
        vecs_ind = np.stack([tx_ind, vecs_ind], axis=-1)

        # Compute symbol indices for every stream.
        # For every constellation point `p` and for every stream `j`, we gather
        # the list of vector indices from `vecs` corresponding the vectors for
        # which the `jth` stream transmitted `p`.
        # [num_vecs/num_points, num_streams, num_points]
        c = []
        for p in points:
            c_ = []
            for j in range(num_streams):
                c_.append(np.where(vecs[:,j]==p)[0])
            c_ = np.stack(c_, axis=-1)
            c.append(c_)
        c = np.stack(c, axis=-1)

        return vecs, vecs_ind, c

    def call(self, inputs):
        if self._with_prior:
            y, h, prior, s = inputs

            # If operating on bits, computes prior on symbols from the prior
            # on bits
            if self._output == 'bit':
                # [..., K, num_points]
                prior = self._llrs2logits(prior)
        else:
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

        # Add prior
        if self._with_prior:
            # [..., num_vecs, K]
            prior = expand_to_rank(prior, tf.rank(exponents), axis=0)
            prior_rank = tf.rank(prior)
            transpose_ind = tf.concat([[prior_rank-2, prior_rank-1],
                                        tf.range(prior_rank-2)], axis=0)
            prior = tf.transpose(prior, transpose_ind)
            prior = tf.gather_nd(prior, self._vecs_ind)
            transpose_ind = tf.concat([ tf.range(2, prior_rank),
                                        [0, 1]], axis=0)
            prior = tf.transpose(prior, transpose_ind)
            # [..., num_vecs]
            prior = tf.reduce_sum(prior, axis=-1)
            exponents = exponents + prior

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
                return tf.argmax(logits, axis=-1, output_type=tf.int32)
            else:
                return logits

class MaximumLikelihoodDetectorWithPrior(MaximumLikelihoodDetector):
    # pylint: disable=line-too-long
    r"""
    MaximumLikelihoodDetectorWithPrior(output, demapping_method, num_streams, constellation_type=None, num_bits_per_symbol=None, constellation=None, hard_out=False, dtype=tf.complex64, **kwargs)

    MIMO maximum-likelihood (ML) detector, assuming prior
    knowledge on the bits or constellation points is available.

    This class is deprecated as the functionality has been integrated
    into :class:`~sionna.mimo.MaximumLikelihoodDetector`.

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
    It is assumed that prior information of the transmitted signal :math:`\mathbf{x}` is available,
    provided either as LLRs on the bits modulated onto :math:`\mathbf{x}` or as logits on the individual
    constellation points forming :math:`\mathbf{x}`.

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
        \begin{align}
            LLR(k,i)&= \ln\left(\frac{\Pr\left(b_{k,i}=1\lvert \mathbf{y},\mathbf{H}\right)}{\Pr\left(b_{k,i}=0\lvert \mathbf{y},\mathbf{H}\right)}\right)\\
                    &=\ln\left(\frac{
                    \sum_{\mathbf{x}\in\mathcal{C}_{k,i,1}} \exp\left(
                        -\left\lVert\tilde{\mathbf{y}}-\tilde{\mathbf{H}}\mathbf{x}\right\rVert^2
                        \right) \Pr\left( \mathbf{x} \right)
                    }{
                    \sum_{\mathbf{x}\in\mathcal{C}_{k,i,0}} \exp\left(
                        -\left\lVert\tilde{\mathbf{y}}-\tilde{\mathbf{H}}\mathbf{x}\right\rVert^2
                        \right) \Pr\left( \mathbf{x} \right)
                    }\right)
        \end{align}

    where :math:`\mathcal{C}_{k,i,1}` and :math:`\mathcal{C}_{k,i,0}` are the
    sets of vectors of constellation points for which the :math:`i\text{th}` bit
    of the :math:`k\text{th}` user is equal to 1 and 0, respectively.
    :math:`\Pr\left( \mathbf{x} \right)` is the prior distribution of the vector of
    constellation points :math:`\mathbf{x}`. Assuming that the constellation points and
    bit levels are independent, it is computed from the prior of the bits according to

    .. math::
        \Pr\left( \mathbf{x} \right) = \prod_{k=1}^K \prod_{i=1}^{I} \sigma \left( LLR_p(k,i) \right)

    where :math:`LLR_p(k,i)` is the prior knowledge of the :math:`i\text{th}` bit of the
    :math:`k\text{th}` user given as an LLR, and :math:`\sigma\left(\cdot\right)` is the sigmoid function.
    The definition of the LLR has been chosen such that it is equivalent with that of logit. This is
    different from many textbooks in communications, where the LLR is
    defined as :math:`LLR(k,i) = \ln\left(\frac{\Pr\left(b_{k,i}=0\lvert \mathbf{y},\mathbf{H}\right)}{\Pr\left(b_{k,i}=1\lvert \mathbf{y},\mathbf{H}\right)}\right)`.

    With the "maxlog" demapping method, the LLR for the :math:`i\text{th}` bit
    of the :math:`k\text{th}` user is approximated like

    .. math::
        \begin{align}
            LLR(k,i) \approx&\ln\left(\frac{
                \max_{\mathbf{x}\in\mathcal{C}_{k,i,1}} \left( \exp\left(
                    -\left\lVert\tilde{\mathbf{y}}-\tilde{\mathbf{H}}\mathbf{x}\right\rVert^2
                    \right) \Pr\left( \mathbf{x} \right) \right)
                }{
                \max_{\mathbf{x}\in\mathcal{C}_{k,i,0}} \left( \exp\left(
                    -\left\lVert\tilde{\mathbf{y}}-\tilde{\mathbf{H}}\mathbf{x}\right\rVert^2
                    \right) \Pr\left( \mathbf{x} \right) \right)
                }\right)\\
                = &\min_{\mathbf{x}\in\mathcal{C}_{k,i,0}} \left( \left\lVert\tilde{\mathbf{y}}-\tilde{\mathbf{H}}\mathbf{x}\right\rVert^2 - \ln \left(\Pr\left( \mathbf{x} \right) \right) \right) -
                    \min_{\mathbf{x}\in\mathcal{C}_{k,i,1}} \left( \left\lVert\tilde{\mathbf{y}}-\tilde{\mathbf{H}}\mathbf{x}\right\rVert^2 - \ln \left( \Pr\left( \mathbf{x} \right) \right) \right).
            \end{align}

    **ML detection of symbols:**

    Soft-decisions on symbols are called logits (i.e., unnormalized log-probability).

    With the “app” demapping method, the logit for the
    constellation point :math:`c \in \mathcal{C}` of the :math:`k\text{th}` user  is computed according to

    .. math::
        \begin{align}
            \text{logit}(k,c) &= \ln\left(\sum_{\mathbf{x} : x_k = c} \exp\left(
                        -\left\lVert\tilde{\mathbf{y}}-\tilde{\mathbf{H}}\mathbf{x}\right\rVert^2
                        \right)\Pr\left( \mathbf{x} \right)\right).
        \end{align}

    With the "maxlog" demapping method, the logit for the constellation point :math:`c \in \mathcal{C}`
    of the :math:`k\text{th}` user  is approximated like

    .. math::
        \text{logit}(k,c) \approx \max_{\mathbf{x} : x_k = c} \left(
                -\left\lVert\tilde{\mathbf{y}}-\tilde{\mathbf{H}}\mathbf{x}\right\rVert^2 + \ln \left( \Pr\left( \mathbf{x} \right) \right)
                \right).

    When hard decisions are requested, this layer returns for the :math:`k` th stream

    .. math::
        \hat{c}_k = \underset{c \in \mathcal{C}}{\text{argmax}} \left( \sum_{\mathbf{x} : x_k = c} \exp\left(
                        -\left\lVert\tilde{\mathbf{y}}-\tilde{\mathbf{H}}\mathbf{x}\right\rVert^2
                        \right)\Pr\left( \mathbf{x} \right) \right)

    where :math:`\mathcal{C}` is the set of constellation points.

    Parameters
    -----------
    output : One of ["bit", "symbol"], str
        The type of output, either LLRs on bits or logits on constellation symbols.

    demapping_method : One of ["app", "maxlog"], str
        The demapping method used.

    num_streams : tf.int
        Number of transmitted streams

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
    (y, h, prior, s) :
        Tuple:

    y : [...,M], tf.complex
        1+D tensor containing the received signals.

    h : [...,M,num_streams], tf.complex
        2+D tensor containing the channel matrices.

    prior : [...,num_streams,num_bits_per_symbol] or [...,num_streams,num_points], tf.float
        Prior of the transmitted signals.
        If ``output`` equals "bit", then LLRs of the transmitted bits are expected.
        If ``output`` equals "symbol", then logits of the transmitted constellation points are expected.

    s : [...,M,M], tf.complex
        2+D tensor containing the noise covariance matrices.

    Output
    ------
    One of:

    : [..., num_streams, num_bits_per_symbol], tf.float
        LLRs or hard-decisions for every bit of every stream, if ``output`` equals `"bit"`.

    : [..., num_streams, num_points], tf.float or [..., num_streams], tf.int
       Logits or hard-decisions for constellation symbols for every stream, if ``output`` equals `"symbol"`.
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
                 num_streams,
                 constellation_type=None,
                 num_bits_per_symbol=None,
                 constellation=None,
                 hard_out=False,
                 dtype=tf.complex64,
                 **kwargs):
        super().__init__(   output=output,
                            demapping_method=demapping_method,
                            num_streams=num_streams,
                            constellation_type=constellation_type,
                            num_bits_per_symbol=num_bits_per_symbol,
                            constellation=constellation,
                            hard_out=hard_out,
                            with_prior=True,
                            dtype=dtype,
                            **kwargs)

class KBestDetector(Layer):
    # pylint: disable=line-too-long
    r"""KBestDetector(output, num_streams, k, constellation_type=None, num_bits_per_symbol=None, constellation=None, hard_out=False, use_real_rep=False, list2llr=None, dtype=tf.complex64)

    MIMO K-Best detector

    This layer implements K-Best MIMO detection as described
    in (Eq. 4-5) [FT2015]_. It can either generate hard decisions (for symbols
    or bits) or compute LLRs.

    The algorithm operates in either the complex or real-valued domain.
    Although both options produce identical results, the former has the advantage
    that it can be applied to arbitrary non-QAM constellations. It also reduces
    the number of streams (or depth) by a factor of two.

    The way soft-outputs (i.e., LLRs) are computed is determined by the
    ``list2llr`` function. The default solution
    :class:`~sionna.mimo.List2LLRSimple` assigns a predetermined
    value to all LLRs without counter-hypothesis.

    This layer assumes the following channel model:

    .. math::
        \mathbf{y} = \mathbf{H}\mathbf{x} + \mathbf{n}

    where :math:`\mathbf{y}\in\mathbb{C}^M` is the received signal vector,
    :math:`\mathbf{x}\in\mathcal{C}^S` is the vector of transmitted symbols which
    are uniformly and independently drawn from the constellation :math:`\mathcal{C}`,
    :math:`\mathbf{H}\in\mathbb{C}^{M\times S}` is the known channel matrix,
    and :math:`\mathbf{n}\in\mathbb{C}^M` is a complex Gaussian noise vector.
    It is assumed that :math:`\mathbb{E}\left[\mathbf{n}\right]=\mathbf{0}` and
    :math:`\mathbb{E}\left[\mathbf{n}\mathbf{n}^{\mathsf{H}}\right]=\mathbf{S}`,
    where :math:`\mathbf{S}` has full rank.

    In a first optional step, the channel model is converted to its real-valued equivalent,
    see :func:`~sionna.mimo.complex2real_channel`. We assume in the sequel the complex-valued
    representation. Then, the channel is whitened using :func:`~sionna.mimo.whiten_channel`:

    .. math::
        \tilde{\mathbf{y}} &= \mathbf{S}^{-\frac{1}{2}}\mathbf{y}\\
        &=  \mathbf{S}^{-\frac{1}{2}}\mathbf{H}\mathbf{x} + \mathbf{S}^{-\frac{1}{2}}\mathbf{n}\\
        &= \tilde{\mathbf{H}}\mathbf{x} + \tilde{\mathbf{n}}.

    Next, the columns of :math:`\tilde{\mathbf{H}}` are sorted according
    to their norm in descending order. Then, the QR decomposition of the
    resulting channel matrix is computed:

    .. math::
        \tilde{\mathbf{H}} = \mathbf{Q}\mathbf{R}

    where :math:`\mathbf{Q}\in\mathbb{C}^{M\times S}` is unitary and
    :math:`\mathbf{R}\in\mathbb{C}^{S\times S}` is upper-triangular.
    The channel outputs are then pre-multiplied by :math:`\mathbf{Q}^{\mathsf{H}}`.
    This leads to the final channel model on which the K-Best detection algorithm operates:

    .. math::
        \bar{\mathbf{y}} = \mathbf{R}\bar{\mathbf{x}} + \bar{\mathbf{n}}

    where :math:`\bar{\mathbf{y}}\in\mathbb{C}^S`,
    :math:`\bar{\mathbf{x}}\in\mathbb{C}^S`, and :math:`\bar{\mathbf{n}}\in\mathbb{C}^S`
    with :math:`\mathbb{E}\left[\bar{\mathbf{n}}\right]=\mathbf{0}` and
    :math:`\mathbb{E}\left[\bar{\mathbf{n}}\bar{\mathbf{n}}^{\mathsf{H}}\right]=\mathbf{I}`.

    **LLR Computation**

    The K-Best algorithm produces :math:`K` candidate solutions :math:`\bar{\mathbf{x}}_k\in\mathcal{C}^S`
    and their associated distance metrics :math:`d_k=\lVert \bar{\mathbf{y}} - \mathbf{R}\bar{\mathbf{x}}_k \rVert^2`
    for :math:`k=1,\dots,K`. If the real-valued channel representation is used, the distance
    metrics are scaled by 0.5 to account for the reduced noise power in each complex dimension.
    A hard-decision is simply the candidate with the shortest distance.
    Various ways to compute LLRs from this list (and possibly
    additional side-information) are possible. The (sub-optimal) default solution
    is :class:`~sionna.mimo.List2LLRSimple`. Custom solutions can be provided.

    Parameters
    -----------
    output : One of ["bit", "symbol"], str
        The type of output, either bits or symbols. Whether soft- or
        hard-decisions are returned can be configured with the
        ``hard_out`` flag.

    num_streams : tf.int
        Number of transmitted streams

    k : tf.int
        The number of paths to keep. Cannot be larger than the
        number of constellation points to the power of the number of
        streams.

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
        Defaults to `False`. The detector cannot compute soft-symbols.

    use_real_rep : bool
        If `True`, the detector use the real-valued equivalent representation
        of the channel. Note that this only works with a QAM constellation.
        Defaults to `False`.

    list2llr: `None` or instance of :class:`~sionna.mimo.List2LLR`
        The function to be used to compute LLRs from a list of candidate solutions.
        If `None`, the default solution :class:`~sionna.mimo.List2LLRSimple`
        is used.

    dtype : One of [tf.complex64, tf.complex128] tf.DType (dtype)
        The dtype of ``y``. Defaults to tf.complex64.
        The output dtype is the corresponding real dtype (tf.float32 or tf.float64).

    Input
    -----
    (y, h, s) :
        Tuple:

    y : [...,M], tf.complex
        1+D tensor containing the received signals

    h : [...,M,num_streams], tf.complex
        2+D tensor containing the channel matrices

    s : [...,M,M], tf.complex
        2+D tensor containing the noise covariance matrices

    Output
    ------
    One of:

    : [...,num_streams,num_bits_per_symbol], tf.float
        LLRs or hard-decisions for every bit of every stream, if ``output`` equals `"bit"`

    : [...,num_streams,2**num_points], tf.float or [...,num_streams], tf.int
       Logits or hard-decisions for constellation symbols for every stream, if ``output`` equals `"symbol"`
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
                 num_streams,
                 k,
                 constellation_type=None,
                 num_bits_per_symbol=None,
                 constellation=None,
                 hard_out=False,
                 use_real_rep=False,
                 list2llr="default",
                 dtype=tf.complex64,
                 **kwargs):
        super().__init__(dtype=dtype, **kwargs)
        assert dtype in [tf.complex64, tf.complex128],\
            "dtype must be tf.complex64 or tf.complex128."

        assert output in ("bit", "symbol"), "Unknown output"

        err_msg = "You must provide either constellation or " + \
                  "constellation_type and num_bits_per_symbol."
        if constellation is None:
            assert constellation_type is not None and \
                   num_bits_per_symbol is not None, err_msg
        else:
            assert constellation_type is None and \
                   num_bits_per_symbol is None, err_msg

        if constellation is not None:
            assert constellation.points.dtype==dtype, \
                "Constellation has wrong dtype."

        self._output = output
        self._hard_out = hard_out
        self._use_real_rep = use_real_rep

        if self._use_real_rep:
            # Real-valued representation is used
            err_msg = "Only QAM can be used for the real-valued representation"
            if constellation_type is not None:
                assert constellation_type=="qam", err_msg
            else:
                assert constellation._constellation_type=="qam", err_msg

            # Double the number of streams to dectect
            self._num_streams = 2*num_streams

            # Half the number of bits for the PAM constellation
            if num_bits_per_symbol is None:
                n = constellation.num_bits_per_symbol//2
                self._num_bits_per_symbol = n
            else:
                self._num_bits_per_symbol = num_bits_per_symbol//2

            # Geerate a PAM constellation with 0.5 energy
            c = Constellation("pam",
                                self._num_bits_per_symbol,
                                normalize=False,
                                dtype=dtype)
            c._points /= tf.cast(np.std(c._points)*np.sqrt(2), c._points.dtype)
            self._constellation = tf.cast(c.points, dtype.real_dtype)

            self._pam2qam = PAM2QAM(2*self._num_bits_per_symbol)

        else:
            # Complex-valued representation is used
            # Number of streams is equal to number of transmitters
            self._num_streams = num_streams

            # Create constellation or take the one provided
            c = Constellation.create_or_check_constellation(
                                                        constellation_type,
                                                        num_bits_per_symbol,
                                                        constellation,
                                                        dtype=dtype)
            self._constellation = c.points
            self._num_bits_per_symbol = c.num_bits_per_symbol

        # Number of constellation symbols
        self._num_symbols = self._constellation.shape[0]

        # Number of best paths to keep
        self._k = np.minimum(k, self._num_symbols**self._num_streams)
        if self._k < k:
            msg = "KBestDetector: " + \
                  f"The provided value of k={k} is larger than " + \
                  "the possible maximum number of paths. " + \
                  f"It has been set to k={self._k}."
            warnings.warn(msg)

        # Compute the number of previous paths a layer needs to consider
        num_paths = [1] # The first layer considers a single path
        for l in range(1, self._num_streams+1):
            # The lth layer considers min(k, num_symbols**l) paths
            num_paths.append(np.minimum(self._k, self._num_symbols**l))
        self._num_paths = tf.constant(tf.stack(num_paths, 0), tf.int32)

        # The symbols and indices for all paths will be stored in tensors
        # of shape [batch_size, k, num_streams]. However, only
        # a subset of the available entries are updated by each stream.
        # To enable XLA, we need to compute the relevant indices of the tensors
        # that will be updated through tf.tensor_scatter_nd_update.
        indices = np.zeros([self._num_streams, self._k*self._num_streams, 2],
                           np.int32)
        for l in range(0, self._num_streams):
            ind = np.zeros([self._num_paths[l+1], self._num_streams])
            ind[:, :l+1] = 1
            ind = np.stack(np.where(ind), -1)
            indices[l,:ind.shape[0],:ind.shape[1]] = ind
        self._indices = tf.constant(indices, dtype=tf.int32)

        if self._output=="bit":
            if self._hard_out is False:
                if list2llr=="default":
                    self.list2llr = List2LLRSimple(self._num_bits_per_symbol)
                else:
                    self.list2llr = list2llr
            else:
                if self._use_real_rep:
                    n = 2*self._num_bits_per_symbol
                else:
                    n = self._num_bits_per_symbol
                self._symbolinds2bits = SymbolInds2Bits(n,
                                             dtype=dtype.real_dtype)
        else:
            assert self._hard_out is True, \
                "Soft-symbols are not supported for this detector."

    @property
    def list2llr(self):
        return self._list2llr

    @list2llr.setter
    def list2llr(self, value):
        assert isinstance(value, List2LLR)
        self._list2llr = value

    def _preprocessing(self, inputs):

        y, h, s = inputs

        # Convert to real-valued representation if desired
        if self._use_real_rep:
            y, h, s = complex2real_channel(y, h, s)

        # Whiten channel
        y, h = whiten_channel(y, h, s, return_s=False) # pylint: disable=W0632

        # Order columns of H in order of decreasing norm
        h_norm = tf.reduce_sum(tf.abs(h)**2, axis=1)
        column_order = tf.argsort(h_norm, axis=-1, direction="DESCENDING")
        h = tf.gather(h, column_order, axis=-1, batch_dims=1)

        # Compute QR decomposition of sorted channel
        # r is upper triangular
        q, r = tf.linalg.qr(h)

        # Project y on Q'
        y = tf.squeeze(tf.matmul(q, tf.expand_dims(y, -1), adjoint_a=True),
                       -1)

        return y, r, column_order

    def _select_best_paths(self, dists, path_syms, path_inds):

        # Determine the number of paths to keep (either all or k)
        num_paths = tf.shape(path_syms)[1]
        k = tf.minimum(num_paths, self._k)

        # Get the k paths with the shortest distance
        dists, ind = tf.math.top_k(-dists, k=k, sorted=True)
        dists = -dists

        # Select the same best paths for the symbols and symbol indices
        path_syms = tf.gather(path_syms, ind, axis=1, batch_dims=1)
        path_inds = tf.gather(path_inds, ind, axis=1, batch_dims=1)

        return dists, path_syms, path_inds

    def _next_layer(self, y, r, dists, path_syms, path_inds, stream):

        batch_size = tf.shape(y)[0]

        # Streams are processed in reverse order
        stream_ind = self._num_streams-1-stream

        # Current number of considered paths
        num_paths = tf.gather(self._num_paths, stream)

        # Store input tensors for scatter update later on
        dists_o = dists
        path_syms_o = path_syms
        path_inds_o = path_inds

        # Extract relevant values from input tensor
        dists = dists[..., :num_paths]
        path_syms = path_syms[..., :num_paths, :stream]
        path_inds = path_inds[..., :num_paths, :stream]

        # Each path creates num_symbols branches
        dists     = tf.repeat(dists,     repeats=self._num_symbols, axis=1)
        path_syms = tf.repeat(path_syms, repeats=self._num_symbols, axis=1)
        path_inds = tf.repeat(path_inds, repeats=self._num_symbols, axis=1)

        # Append to each path the symbols corresponding to the branch
        syms = tf.reshape(self._constellation, [1,-1])
        syms = tf.repeat(syms, self._k, 0)
        syms = tf.reshape(syms, [1, -1, 1])
        syms = tf.repeat(syms, batch_size, 0)
        syms = syms[:,:num_paths*self._num_symbols]
        path_syms = tf.concat([path_syms, syms], axis=-1)

        # Do the same for the symbol indices
        inds = tf.reshape(tf.range(0, self._num_symbols), [1, -1])
        inds = tf.repeat(inds, self._k, 0)
        inds = tf.reshape(inds, [1, -1, 1])
        inds = tf.repeat(inds, batch_size, 0)
        inds = inds[:,:num_paths*self._num_symbols]
        path_inds = tf.concat([path_inds, inds], axis=-1)

        # Compute partial distances
        # Extract the row of r corresponding to layer and reverse the order
        y = tf.expand_dims(y[:, stream_ind], axis=-1)
        r = tf.expand_dims(tf.reverse(r[:, stream_ind, stream_ind:], [-1]), 1)
        delta = tf.pow(tf.abs(y - tf.reduce_sum(r*path_syms, axis=-1)), 2)

        # Update distances
        dists += delta

        # Get k best paths
        dists, path_syms, path_inds = self._select_best_paths(dists, path_syms, path_inds)

        # Scatter updates of dists
        tensor = tf.transpose(dists_o, perm=[1, 0])
        updates = tf.transpose(dists, perm=[1, 0])
        indices = tf.expand_dims(tf.range(tf.shape(updates)[0], dtype=tf.int32), -1)
        dists = tf.tensor_scatter_nd_update(tensor, indices, updates)
        dists = tf.transpose(dists, perm=[1, 0])

        # Scatter update of path_syms
        tensor = tf.transpose(path_syms_o, [1, 2, 0])
        updates = tf.transpose(path_syms, [1, 2, 0])
        updates = tf.reshape(updates, [-1, batch_size])
        indices = self._indices[stream, :self._num_paths[stream+1]*(stream+1)]
        path_syms = tf.tensor_scatter_nd_update(tensor, indices, updates)
        path_syms = tf.transpose(path_syms, perm=[2, 0, 1])

        # Scatter update of path_inds
        tensor = tf.transpose(path_inds_o, [1, 2, 0])
        updates = tf.transpose(path_inds, [1, 2, 0])
        updates = tf.reshape(updates, [-1, batch_size])
        path_inds = tf.tensor_scatter_nd_update(tensor, indices, updates)
        path_inds = tf.transpose(path_inds, perm=[2, 0, 1])

        return dists, path_syms, path_inds

    def _unsort(self, column_order, tensor, transpose=True):
        # Undo the column sorting
        # If transpose=True, the unsorting is done along the last dimension
        # Otherwise, sorting is done along the second-last index
        unsort_inds = tf.argsort(column_order, axis=-1)
        if transpose:
            tensor = tf.transpose(tensor, perm=[0, 2, 1])
        tensor = tf.gather(tensor, unsort_inds, axis=-2, batch_dims=1)
        if transpose:
            tensor = tf.transpose(tensor, perm=[0, 2, 1])
        return tensor

    def build(self, input_shape):
        assert input_shape[1][-2]>=input_shape[1][-1], \
                "The number of receive antennas cannot be smaller \
                 than the number of streams"

    def call(self, inputs):

        # Flatten the batch dimensions
        y, h, s = inputs
        batch_shape = tf.shape(y)[:-1]
        num_batch_dims = len(batch_shape)
        if num_batch_dims > 1:
            y = flatten_dims(y, num_batch_dims, 0)
            h = flatten_dims(h, num_batch_dims, 0)
            s = flatten_dims(s, num_batch_dims, 0)
            inputs = (y,h,s)

        # Initialization
        # (i) (optional) Convert to real-valued representation
        # (ii) Whiten channel
        # (iii) Sort columns of H by decreasing column norm
        # (iv) QR Decomposition of H
        # (v) Project y onto Q'
        y, r, column_order = self._preprocessing(inputs)

        batch_size = tf.shape(y)[0]

        # Tensor to keep track of the aggregate distances of all paths
        dists = tf.zeros([batch_size, self._k], y.dtype.real_dtype)

        # Tensor to store constellation symbols of all paths
        path_syms = tf.zeros([batch_size, self._k, self._num_streams], y.dtype)

        # Tensor to store constellation symbol indices of all paths
        path_inds = tf.zeros([batch_size, self._k, self._num_streams],tf.int32)

        # Sequential K-Best algorithm
        for stream in range(0, self._num_streams):
            dists, path_syms, path_inds = self._next_layer(y,
                                                           r,
                                                           dists,
                                                           path_syms,
                                                           path_inds,
                                                           stream)

        # Reverse order as detection started with the last symbol first
        path_syms = tf.reverse(path_syms, axis=[-1])
        path_inds = tf.reverse(path_inds, axis=[-1])

        # Processing for hard-decisions
        if self._hard_out:
            path_inds = self._unsort(column_order, path_inds)
            hard_dec = path_inds[:,0,:]

            # Real-valued representation
            if self._use_real_rep:
                hard_dec = \
                    self._pam2qam(hard_dec[...,:self._num_streams//2],
                                  hard_dec[...,self._num_streams//2:])

            # Hard decisions on bits
            if self._output=="bit":
                hard_dec = self._symbolinds2bits(hard_dec)

            # Reshape batch dimensions
            if num_batch_dims > 1:
                hard_dec = split_dim(hard_dec, batch_shape, 0)

            return hard_dec

        # Processing for soft-decisions
        else:
            # Real-valued representation
            if self._use_real_rep:
                llr = self.list2llr([y, r, dists, path_inds, path_syms])
                llr = self._unsort(column_order, llr, transpose=False)

                # Combine LLRs from PAM symbols in the correct order
                llr1 = llr[:,:self._num_streams//2]
                llr2 = llr[:,self._num_streams//2:]
                llr1 = tf.expand_dims(llr1, -1)
                llr2 = tf.expand_dims(llr2, -1)
                llr = tf.concat([llr1, llr2], -1)
                llr = tf.reshape(llr, [-1, self._num_streams//2,
                                   2*self._num_bits_per_symbol])

            # Complex-valued representation
            else:
                llr = self.list2llr([y, r, dists, path_inds, path_syms])
                llr = self._unsort(column_order, llr, transpose=False)

            # Reshape batch dimensions
            if num_batch_dims > 1:
                llr = split_dim(llr, batch_shape, 0)

            return llr

class EPDetector(Layer):
    # pylint: disable=line-too-long
    r"""EPDetector(output, num_bits_per_symbol, hard_out=False, l=10, beta=0.9, dtype=tf.complex64)

    MIMO Expectation Propagation (EP) detector

    This layer implements Expectation Propagation (EP) MIMO detection as described
    in [EP2014]_. It can generate hard- or soft-decisions for symbols or bits.

    This layer assumes the following channel model:

    .. math::
        \mathbf{y} = \mathbf{H}\mathbf{x} + \mathbf{n}

    where :math:`\mathbf{y}\in\mathbb{C}^M` is the received signal vector,
    :math:`\mathbf{x}\in\mathcal{C}^S` is the vector of transmitted symbols which
    are uniformly and independently drawn from the constellation :math:`\mathcal{C}`,
    :math:`\mathbf{H}\in\mathbb{C}^{M\times S}` is the known channel matrix,
    and :math:`\mathbf{n}\in\mathbb{C}^M` is a complex Gaussian noise vector.
    It is assumed that :math:`\mathbb{E}\left[\mathbf{n}\right]=\mathbf{0}` and
    :math:`\mathbb{E}\left[\mathbf{n}\mathbf{n}^{\mathsf{H}}\right]=\mathbf{S}`,
    where :math:`\mathbf{S}` has full rank.

    The channel model is first whitened using :func:`~sionna.mimo.whiten_channel`
    and then converted to its real-valued equivalent,
    see :func:`~sionna.mimo.complex2real_channel`, prior to MIMO detection.

    The computation of LLRs is done by converting the symbol logits
    that naturally arise in the algorithm to LLRs using
    :func:`~sionna.mapping.PAM2QAM`. Custom conversions of symbol logits to LLRs
    can be implemented by using the soft-symbol output.

    Parameters
    -----------
    output : One of ["bit", "symbol"], str
        The type of output, either bits or symbols. Whether soft- or
        hard-decisions are returned can be configured with the
        ``hard_out`` flag.

    num_bits_per_symbol : int
        The number of bits per QAM constellation symbol, e.g., 4 for QAM16.

    hard_out : bool
        If `True`, the detector computes hard-decided bit values or
        constellation point indices instead of soft-values.
        Defaults to `False`.

    l : int
        Number of iterations. Defaults to 10.

    beta : float
        Parameter :math:`\beta\in[0,1]` for update smoothing.
        Defaults to 0.9.

    dtype : One of [tf.complex64, tf.complex128] tf.DType (dtype)
        Precision used for internal computations. Defaults to ``tf.complex64``.
        Especially for large MIMO setups, the precision can make a significant
        performance difference.

    Input
    -----
    (y, h, s) :
        Tuple:

    y : [...,M], tf.complex
        1+D tensor containing the received signals

    h : [...,M,num_streams], tf.complex
        2+D tensor containing the channel matrices

    s : [...,M,M], tf.complex
        2+D tensor containing the noise covariance matrices

    Output
    ------
    One of:

    : [...,num_streams,num_bits_per_symbol], tf.float
        LLRs or hard-decisions for every bit of every stream, if ``output`` equals `"bit"`

    : [...,num_streams,2**num_bits_per_symbol], tf.float or [...,num_streams], tf.int
       Logits or hard-decisions for constellation symbols for every stream, if ``output`` equals `"symbol"`

    Note
    ----
    For numerical stability, we do not recommend to use this function in Graph
    mode with XLA, i.e., within a function that is decorated with
    ``@tf.function(jit_compile=True)``.
    However, it is possible to do so by setting
    ``sionna.Config.xla_compat=true``.
    See :py:attr:`~sionna.Config.xla_compat`.
    """
    def __init__(self,
                 output,
                 num_bits_per_symbol,
                 hard_out=False,
                 l=10,
                 beta=0.9,
                 dtype=tf.complex64,
                 **kwargs):
        super().__init__(dtype=dtype, **kwargs)
        assert dtype in [tf.complex64, tf.complex128], \
            "Invalid dtype"
        self._cdtype = tf.dtypes.as_dtype(dtype)
        self._rdtype = self._cdtype.real_dtype

        # Variable used to avoid numerical instabilities
        # See paragraph after Eq. (38)
        if self.dtype=="complex64":
            self._prec = 1e-6
        else:
            self._prec = 1e-12

        assert output in ("bit", "symbol"), "Unknown output"
        self._output = output

        self._hard_out = hard_out

        if self._output=="symbol":
            self._pam2qam = PAM2QAM(num_bits_per_symbol, hard_out)
        else:
            self._symbollogits2llrs = SymbolLogits2LLRs("maxlog",
                                                        num_bits_per_symbol//2,
                                                        hard_out=hard_out)
            self._demapper = Demapper("maxlog", "pam", num_bits_per_symbol//2)

        assert l>=1, "l must be a positive integer"
        self._l = l

        assert 0.0<= beta <=1.0, "beta must be in [0,1]"
        self._beta = beta

        # Create PAM constellations for real-valued detection
        self._num_bits_per_symbol = num_bits_per_symbol//2
        points = Constellation("pam", int(self._num_bits_per_symbol)).points

        # Scale constellation points to half the energy because QAM is assumed
        self._points = tf.cast(points/np.sqrt(2.0), self._rdtype)

        # Average symbol energy
        self._es = tf.constant(np.var(self._points), self._rdtype)

    def compute_sigma_mu(self, h_t_h, h_t_y, no, lam, gam):
        """Equations (28) and (29)"""

        # Prepare inputs
        lam = tf.linalg.diag(lam)
        gam = tf.expand_dims(gam, axis=-1)

        # Computations
        sigma = tf.linalg.inv(h_t_h + no*lam)
        mu = tf.squeeze(tf.matmul(sigma, h_t_y + no*gam), axis=-1)
        sigma *= no
        sigma = tf.linalg.diag_part(sigma)

        return sigma, mu

    def compute_v_x_obs(self, sigma, mu, lam, gam):
        """Equations (31) and (32)"""

        v_obs = tf.maximum(1/(1/sigma-lam), self._prec)
        x_obs = v_obs*(mu/sigma-gam)

        return v_obs, x_obs

    def compute_v_x(self, v_obs, x_obs):
        """Equation (33)"""

        # Compute probability mass function for the symbols
        x_obs = tf.expand_dims(x_obs, -1)
        v_obs = tf.expand_dims(v_obs, -1)

        points = expand_to_rank(self._points, tf.rank(x_obs), axis=0)
        logits = -tf.pow(x_obs-points, 2) / (tf.cast(2, self._rdtype)*v_obs)
        pmf = tf.math.softmax(logits)

        # Compute mean and variance of all symbols
        x = tf.reduce_sum(points * pmf, axis=-1, keepdims=True)
        v = tf.reduce_sum((points-x)**2 * pmf, axis=-1)
        v = tf.maximum(v, self._prec)
        x = tf.squeeze(x, axis=-1)

        return v, x, logits

    def update_lam_gam(self, v, v_obs, x, x_obs, lam, gam):
        """Equations (35), (36), (37), (38)"""

        # Save old values of lam, and gam
        lam_old = lam
        gam_old = gam

        # Compute potential new values (35), (36)
        lam = 1/v - 1/v_obs
        gam = x/v - x_obs/v_obs

        # Only update nonnegative values
        lam_new = tf.where(lam<0, lam_old, lam)
        gam_new = tf.where(lam<0, gam_old, gam)

        # Damp updates (37), (38)
        lam_damp = (1-self._beta)*lam_new + self._beta*lam_old
        gam_damp = (1-self._beta)*gam_new + self._beta*gam_old

        return lam_damp, gam_damp

    def call(self, inputs):

        # Flatten the batch dimensions
        y, h, s = inputs
        batch_shape = tf.shape(y)[:-1]
        num_batch_dims = len(batch_shape)
        if num_batch_dims > 1:
            y = flatten_dims(y, num_batch_dims, 0)
            h = flatten_dims(h, num_batch_dims, 0)
            s = flatten_dims(s, num_batch_dims, 0)
            inputs = (y,h,s)

        # Number of transmit streams
        n_t = tf.shape(h)[-1]

        # Whiten channel
        y, h, s = whiten_channel(y, h, s)

        # Convert channel to real-valued representation
        y, h, s = complex2real_channel(y,h,s)

        # Convert all inputs to desired dtypes
        y = tf.cast(y, self._rdtype)
        h = tf.cast(h, self._rdtype)
        no = tf.cast(0.5, self._rdtype)

        # Gather relevant parameters
        batch_dims = tf.shape(y)[:-1]
        n_t_r = tf.shape(h)[-1]

        # Initialize gamma and lambda (Paragraph after Eq. (29))
        gam = tf.zeros(tf.concat([batch_dims, [n_t_r]], axis=0), y.dtype)
        lam = tf.ones(tf.concat([batch_dims, [n_t_r]], axis=0), y.dtype)
        lam /= tf.cast(self._es, y.dtype)

        # Precompute values that are repeatedly needed
        h_t_h = tf.matmul(h, h, transpose_a=True)
        y = tf.expand_dims(y, axis=-1)
        h_t_y = tf.matmul(h, y, transpose_a=True)
        no = expand_to_rank(no, tf.rank(h), axis=-1)

        for _ in range(self._l):
            sigma, mu = self.compute_sigma_mu(h_t_h, h_t_y, no, lam, gam)
            v_obs, x_obs = self.compute_v_x_obs(sigma, mu, lam, gam)
            v, x, logits = self.compute_v_x(v_obs, x_obs)
            lam, gam = self.update_lam_gam(v, v_obs, x, x_obs, lam, gam)

        # Extract the logits for the 2 PAM constellations for each streams
        pam1_logits = logits[...,:n_t,:]
        pam2_logits = logits[...,n_t:,:]

        if self._output=="symbol" and self._hard_out:
            # Take hard decisions on PAM symbol;s
            pam1_ind = tf.argmax(pam1_logits, axis=-1, output_type=tf.int32)
            pam2_ind = tf.argmax(pam2_logits, axis=-1, output_type=tf.int32)

            # Transform to QAM indices
            qam_ind = self._pam2qam(pam1_ind, pam2_ind)

            # Reshape batch dimensions
            if num_batch_dims > 1:
                qam_ind = split_dim(qam_ind, batch_shape, 0)

            return qam_ind

        elif self._output=="symbol" and not self._hard_out:
            qam_logits = self._pam2qam(pam1_logits, pam2_logits)

            # Reshape batch dimensions
            if num_batch_dims > 1:
                qam_logits = split_dim(qam_logits, batch_shape, 0)

            return qam_logits

        elif self._output=="bit":
            # Compute LLRs for both PAM constellations
            llr1 = self._symbollogits2llrs(pam1_logits)
            llr2 = self._symbollogits2llrs(pam2_logits)

            # Put LLRs in the correct order and shape
            llr = tf.stack([llr1, llr2], -1)
            llr = flatten_last_dims(llr)

            # Reshape batch dimensions
            if num_batch_dims > 1:
                llr = split_dim(llr, batch_shape, 0)

            return llr

class MMSEPICDetector(Layer):
    # pylint: disable=line-too-long
    r"""MMSEPICDetector(output, demapping_method="maxlog", num_iter=1, constellation_type=None, num_bits_per_symbol=None, constellation=None, hard_out=False, dtype=tf.complex64, **kwargs)

    Minimum mean square error (MMSE) with parallel interference cancellation (PIC) detector

    This layer implements the MMSE PIC detector, as proposed in [CST2011]_.
    For ``num_iter``>1, this implementation performs MMSE PIC self-iterations.
    MMSE PIC self-iterations can be understood as a concatenation of MMSE PIC
    detectors from [CST2011]_, which forward intrinsic LLRs to the next
    self-iteration.

    Compared to [CST2011]_, this implementation also accepts priors on the
    constellation symbols as an alternative to priors on the bits.

    This layer assumes the following channel model:

    .. math::
        \mathbf{y} = \mathbf{H}\mathbf{x} + \mathbf{n}

    where :math:`\mathbf{y}\in\mathbb{C}^M` is the received signal vector,
    :math:`\mathbf{x}\in\mathcal{C}^S` is the vector of transmitted symbols which
    are uniformly and independently drawn from the constellation :math:`\mathcal{C}`,
    :math:`\mathbf{H}\in\mathbb{C}^{M\times S}` is the known channel matrix,
    and :math:`\mathbf{n}\in\mathbb{C}^M` is a complex Gaussian noise vector.
    It is assumed that :math:`\mathbb{E}\left[\mathbf{n}\right]=\mathbf{0}` and
    :math:`\mathbb{E}\left[\mathbf{n}\mathbf{n}^{\mathsf{H}}\right]=\mathbf{S}`,
    where :math:`\mathbf{S}` has full rank.

    The algorithm starts by computing the soft symbols
    :math:`\bar{x}_s=\mathbb{E}\left[ x_s \right]` and
    variances :math:`v_s=\mathbb{E}\left[ |e_s|^2\right]` from the priors,
    where :math:`e_s = x_s - \bar{x}_s`, for all :math:`s=1,\dots,S`.

    Next, for each stream, the interference caused by all other streams is cancelled
    from the observation :math:`\mathbf{y}`, leading to

    .. math::
        \hat{\mathbf{y}}_s = \mathbf{y} - \sum_{j\neq s} \mathbf{h}_j x_j = \mathbf{h}_s x_s + \tilde{\mathbf{n}}_s,\quad s=1,\dots,S

    where :math:`\tilde{\mathbf{n}}_s=\sum_{j\neq s} \mathbf{h}_j e_j + \mathbf{n}`.

    Then, a linear MMSE filter :math:`\mathbf{w}_s` is computed to reduce the resdiual noise
    for each observation :math:`\hat{\mathbf{y}}_s`, which is given as

    .. math::
        \mathbf{w}_s = \mathbf{h}_s^{\mathsf{H}}\left( \mathbf{H} \mathbf{D}_s\mathbf{H}^{\mathsf{H}} +\mathbf{S} \right)^{-1}

    where :math:`\mathbf{D}_s \in \mathbb{C}^{S\times S}` is diagonal with entries

    .. math::
        \left[\mathbf{D}_s\right]_{i,i} = \begin{cases}
                                            v_i & i\neq s \\
                                            1 & i=s.
                                          \end{cases}

    The filtered observations

    .. math::
        \tilde{z}_s = \mathbf{w}_s^{\mathsf{H}} \hat{\mathbf{y}}_s = \tilde{\mu}_s x_s + \mathbf{w}_s^{\mathsf{H}}\tilde{\mathbf{n}}_s

    where :math:`\tilde{\mu}_s=\mathbf{w}_s^{\mathsf{H}} \mathbf{h}_s`, are then demapped to either symbol logits or LLRs, assuming that the remaining noise is Gaussian with variance

    .. math::
        \nu_s^2 = \mathop{\text{Var}}\left[\tilde{z}_s\right] = \mathbf{w}_s^{\mathsf{H}} \left(\sum_{j\neq s} \mathbf{h}_j \mathbf{h}_j^{\mathsf{H}} v_j +\mathbf{S} \right)\mathbf{w}_s.

    The resulting soft-symbols can then be used for the next self-iteration of the algorithm.

    Note that this algorithm can be substantially simplified as described in [CST2011]_ to avoid
    the computation of different matrix inverses for each stream. This is the version which is
    implemented.

    Parameters
    -----------
    output : One of ["bit", "symbol"], str
        The type of output, either LLRs on bits or logits on constellation
        symbols.

    demapping_method : One of ["app", "maxlog"], str
        The demapping method used.
        Defaults to "maxlog".

    num_iter : int
        Number of MMSE PIC iterations.
        Defaults to 1.

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
        The output dtype is the corresponding real dtype
        (tf.float32 or tf.float64).

    Input
    -----
    (y, h, prior, s) :
        Tuple:

    y : [...,M], tf.complex
        1+D tensor containing the received signals

    h : [...,M,S], tf.complex
        2+D tensor containing the channel matrices

    prior : [...,S,num_bits_per_symbol] or [...,S,num_points], tf.float
        Prior of the transmitted signals.
        If ``output`` equals "bit", then LLRs of the transmitted bits are expected.
        If ``output`` equals "symbol", then logits of the transmitted constellation points are expected.

    s : [...,M,M], tf.complex
        2+D tensor containing the noise covariance matrices

    Output
    ------
    One of:

    : [...,S,num_bits_per_symbol], tf.float
        LLRs or hard-decisions for every bit of every stream, if ``output`` equals `"bit"`

    : [...,S,2**num_bits_per_symbol], tf.float or [...,S], tf.int
       Logits or hard-decisions for constellation symbols for every stream, if ``output`` equals `"symbol"`

    Note
    ----
    For numerical stability, we do not recommend to use this function in Graph
    mode with XLA, i.e., within a function that is decorated with
    ``@tf.function(jit_compile=True)``.
    However, it is possible to do so by setting
    ``sionna.Config.xla_compat=true``.
    See :py:attr:`~sionna.Config.xla_compat`.
    """
    def __init__(self,
                 output,
                 demapping_method="maxlog",
                 num_iter=1,
                 constellation_type=None,
                 num_bits_per_symbol=None,
                 constellation=None,
                 hard_out=False,
                 dtype=tf.complex64,
                 **kwargs):
        super().__init__(dtype=dtype, **kwargs)

        assert isinstance(num_iter, int), "num_iter must be an integer"
        assert output in ("bit", "symbol"), "Unknown output"
        assert demapping_method in ("app", "maxlog"), "Unknown demapping method"

        assert dtype in [tf.complex64, tf.complex128], \
            "dtype must be tf.complex64 or tf.complex128"

        self._num_iter = num_iter
        self._output = output
        self._epsilon = 1e-4
        self._realdtype = dtype.real_dtype
        self._demapping_method = demapping_method
        self._hard_out = hard_out

        # Create constellation object
        self._constellation = Constellation.create_or_check_constellation(
            constellation_type,
            num_bits_per_symbol,
            constellation,
            dtype=dtype)

        # Soft symbol mapping
        self._llr_2_symbol_logits = LLRs2SymbolLogits(
                                        self._constellation.num_bits_per_symbol,
                                        dtype=self._realdtype)

        if self._output == "symbol":
            self._llr_2_symbol_logits_output = LLRs2SymbolLogits(
                                    self._constellation.num_bits_per_symbol,
                                    dtype=self._realdtype,
                                    hard_out=hard_out)
            self._symbol_logits_2_llrs = SymbolLogits2LLRs(
                method=demapping_method,
                num_bits_per_symbol=self._constellation.num_bits_per_symbol)
        self._symbol_logits_2_moments = SymbolLogits2Moments(
                                            constellation=self._constellation,
                                            dtype=self._realdtype)

        # soft output demapping
        self._bit_demapper = DemapperWithPrior(
                                            demapping_method=demapping_method,
                                            constellation=self._constellation,
                                            dtype=dtype)


    def call(self, inputs):
        y, h, prior, s = inputs
        # y is unwhitened receive signal
        #   [..., M]
        # h the channel estimate
        #   [..., M, K]
        # prior is either the soft input LLRs
        #   [..., K, num_bits_per_symbol] or symbol logits [..., K, Q]
        # s the noise covariance matrix
        #   [..., M, M]

        ## Preprocessing
        # Whiten channel
        # y : [..., M]
        # s : [..., M, M]
        y, h = whiten_channel(y, h, s, return_s=False)  # pylint: disable=unbalanced-tuple-unpacking

        # matched filtering of y
        # [..., K, 1]
        y_mf = insert_dims(tf.linalg.matvec(h, y, adjoint_a=True),
                            num_dims=1, axis=-1)

        ## Step 1: compute Gramm matrix
        # [..., K, K]
        g = tf.matmul(h, h, adjoint_a=True)

        # For XLA compatibility, this implementation performs the MIMO
        # equalization in the real-valued domain
        # [..., 2M, 2K]
        hr = complex2real_matrix(h)
        # [..., 2K, 2K]
        gr = tf.matmul(hr, hr, adjoint_a=True)

        # Compute a priori LLRs
        if self._output == "symbol":
            llr_a = self._symbol_logits_2_llrs(prior)
        else:
            llr_a = prior
        # llr_a is [..., K, num_bits_per_symbol]
        llr_shape = tf.shape(llr_a)

        def mmse_pic_self_iteration(llr_d, llr_a, it):
            # MMSE PIC takes in a priori LLRs
            llr_a = llr_d

            # Step 2: compute soft symbol estimates and variances
            # x_hat, var_x : [..., K]
            x_logits = self._llr_2_symbol_logits(llr_a)
            x_hat, var_x = self._symbol_logits_2_moments(x_logits)

            # Step 3: perform parallel interference cancellation
            # H^H y_hat_i = y_mf - sum_j!=i gj x_hat_j = y + g_i x_hat_i
            #               - sum_j g_j x_hat_j
            # [..., K, K]
            y_mf_pic = y_mf + g * insert_dims(x_hat, num_dims=1, axis=-2) \
                - tf.linalg.matmul(g, insert_dims(x_hat, num_dims=1, axis=-1))

            # Step 4: compute A^-1 matrix
            # Calculate MMSE Filter (efficiently)
            # W^H = A^-1 H^H
            # A = H^H H \Lambda + N_0 I_Mt
            # \Lambda_ii is a diagonal matrix with \Lambda_ii = E_i = error_var

            # Stack error variances and make it real
            # Note: Imaginary part is zero
            var_x = tf.cast(tf.concat([var_x, var_x], axis=-1),
                            dtype=self._realdtype)
            var_x_row_vec = insert_dims(var_x, num_dims=1, axis=-2)
            # [..., 2K, 2K]
            a = gr * var_x_row_vec

            i = expand_to_rank(tf.eye(tf.shape(a)[-1], dtype=a.dtype),
                                tf.rank(a), 0)
            a = a + i

            # a is non-hermitian! that's why we can't use sn.utils.matrix_inv
            # XLA can't invert complex matrices, that's why we work with the
            # real valued domain
            a_inv = tf.linalg.inv(a)

            # Step 5: compute unbiased MMSE filter and outputs, calculate A\H^H

            # Calculate bias mu_i = diag(A^-1 H^H H) = diag(A^-1 G)
            # Diagonal elements of matrix matrix multiplication simplified
            # to sum and dot-product
            # [..., 2K]
            mu = tf.reduce_sum(a_inv * tf.linalg.matrix_transpose(gr), axis=-1)

            # Make y_mf_pic columns real (after transposition,
            # the last dimension corresponds to vectors)
            # [..., K, 2K]
            y_mf_pic_trans = tf.linalg.matrix_transpose(y_mf_pic)
            y_mf_pic_trans = complex2real_vector(y_mf_pic_trans)
            # stack them such that y_mf_pic_trans has shape [..., 2K, 2K]
            y_mf_pic_trans = tf.concat([y_mf_pic_trans, y_mf_pic_trans],
                                        axis=-2)

            # Efficient parallel equalization after PIC
            # z_i = i'th row of a_inv * y_MF_PIC_i
            # boils down to tf.reduce_sum(a_inv * y_mf_pic_trans, axis=-1)
            # divide by mu_i for unbiasedness
            # [..., K]
            x_hat = real2complex_vector(tf.reduce_sum(a_inv * y_mf_pic_trans,
                                    axis=-1) / tf.cast(mu, dtype=a_inv.dtype))

            # Compute post equalization signal error estimate:
            # rho_i = mu_i / (1 - var_x_i * mu_i)
            # 1 - var_x_i * mu_i can become numerically 0, or even slightly
            # smaller than zero due to limited numerical precision
            # [..., 2K]
            var_x = tf.divide(mu, tf.maximum(1 - var_x * mu, self._epsilon))
            # real variances map to the same complex valued variances in this
            # model
            var_x, _ = tf.split(var_x, 2, -1)

            no_eff = 1. / var_x

            # Step 6: LLR demapping (extrinsic LLRs)
            # [..., K, num_bits_per_symbols]
            llr_d = tf.reshape(self._bit_demapper([x_hat, llr_a, no_eff]),
                                llr_shape)

            return llr_d, llr_a, it

        # Stopping condition (required for tf.while_loop)
        def dec_stop(llr_d, llr_a, it):  # pylint: disable=W0613
            return tf.less(it, self._num_iter)

        # start decoding iterations
        it = tf.constant(0)
        null_prior = tf.zeros(llr_shape, dtype=self._realdtype)
        llr_d, llr_a, _ = tf.while_loop(dec_stop,
                                    mmse_pic_self_iteration,
                                    (llr_a, null_prior, it),
                                    parallel_iterations=1,
                                    maximum_iterations=self._num_iter)
        llr_e = llr_d - llr_a
        if self._output == "symbol":
            # convert back to symbols if requested.
             # output symbol logits computed on extrinsic LLRs
            out = self._llr_2_symbol_logits_output(llr_e)
        else:
            # output extrinsic LLRs
            out = llr_e
            if self._hard_out:
                out = hard_decisions(out)

        return out
