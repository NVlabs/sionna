#
# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0#
"""Classes and functions related to MIMO channel equalization"""

import tensorflow as tf
from sionna.phy import config, dtypes
from sionna.phy.utils import expand_to_rank, matrix_pinv
from sionna.phy.mimo.utils import whiten_channel

def lmmse_matrix(h, s=None, precision=None):
    # pylint: disable=line-too-long
    r"""MIMO LMMSE Equalization matrix

    This function computes the LMMSE equalization matrix for a MIMO link,
    assuming the following model:

    .. math::

        \mathbf{y} = \mathbf{H}\mathbf{x} + \mathbf{n}

    where :math:`\mathbf{y}\in\mathbb{C}^M` is the received signal vector,
    :math:`\mathbf{x}\in\mathbb{C}^K` is the vector of transmitted symbols,
    :math:`\mathbf{H}\in\mathbb{C}^{M\times K}` is the known channel matrix,
    and :math:`\mathbf{n}\in\mathbb{C}^M` is a noise vector.
    It is assumed that :math:`\mathbb{E}\left[\mathbf{x}\right]=\mathbb{E}\left[\mathbf{n}\right]=\mathbf{0}`,
    :math:`\mathbb{E}\left[\mathbf{x}\mathbf{x}^{\mathsf{H}}\right]=\mathbf{I}_K` and
    :math:`\mathbb{E}\left[\mathbf{n}\mathbf{n}^{\mathsf{H}}\right]=\mathbf{S}`.

    This function returns the LLMSE equalization matrix:

    .. math::

        \mathbf{G} = \mathbf{H}^{\mathsf{H}} \left(\mathbf{H}\mathbf{H}^{\mathsf{H}} + \mathbf{S}\right)^{-1}.


    If :math:`\mathbf{S}=\mathbf{I}_M`, a numerically more stable version of the equalization matrix is computed:

    .. math::

        \mathbf{G} = \left(\mathbf{H}^{\mathsf{H}}\mathbf{H} + \mathbf{I}\right)^{-1}\mathbf{H}^{\mathsf{H}} .

    Input
    -----

    h : [...,M,K], `tf.complex`
        Channel matrices

    s : `None` (default) | [...,M,M], `tf.complex`
        Noise covariance matrices. If `None`, the noise is assumed to be white
        with unit variance.

    precision : `None` (default) | "single" | "double"
        Precision used for internal calculations and outputs.
        If set to `None`,
        :attr:`~sionna.phy.config.Config.precision` is used.

    Output
    ------
    g : [...,K,M], `tf.complex`
        LLMSE equalization matrices

    """
    # Cast inputs
    if precision is None:
        cdtype = config.tf_cdtype
    else:
        cdtype = dtypes[precision]["tf"]["cdtype"]
    h = tf.cast(h, dtype=cdtype)
    if s is not None:
        s = tf.cast(s, dtype=cdtype)
        s_none = False
    else:
        s = expand_to_rank(tf.eye(h.shape[-1], dtype=h.dtype), tf.rank(h), 0)
        s_none = True

    if not s_none:
        #------------------------------------#
        # Compute g = h^* @ (h @ h^* + s)^-1 #
        #------------------------------------#
        # hhs = h @ h^* + s.
        # Note that hhs^* = hhs, hence it admits a Cholesky decomposition
        hhs = tf.matmul(h, h, adjoint_b=True) + s

        # Solve hhs @ g_t = h in the unknown g_t
        chol = tf.linalg.cholesky(hhs)
        g_t = tf.linalg.cholesky_solve(chol, h)

        # Compute g = g_t^* = (hhs^-1 @ h)^* = h^* @ hhs^-1
        g = tf.linalg.adjoint(g_t)
    else:
        #------------------------------------#
        # Compute g = (h^* @ h + I)^-1 @ h^* #
        #------------------------------------#
        hhs = tf.matmul(h, h, adjoint_a=True) + s
        chol = tf.linalg.cholesky(hhs)
        g = tf.linalg.cholesky_solve(chol, tf.linalg.adjoint(h))

    return g

def lmmse_equalizer(y, h, s, whiten_interference=True, precision=None):
    # pylint: disable=line-too-long
    r""" MIMO LMMSE Equalizer

    This function implements LMMSE equalization for a MIMO link, assuming the
    following model:

    .. math::

        \mathbf{y} = \mathbf{H}\mathbf{x} + \mathbf{n}

    where :math:`\mathbf{y}\in\mathbb{C}^M` is the received signal vector,
    :math:`\mathbf{x}\in\mathbb{C}^K` is the vector of transmitted symbols,
    :math:`\mathbf{H}\in\mathbb{C}^{M\times K}` is the known channel matrix,
    and :math:`\mathbf{n}\in\mathbb{C}^M` is a noise vector.
    It is assumed that :math:`\mathbb{E}\left[\mathbf{x}\right]=\mathbb{E}\left[\mathbf{n}\right]=\mathbf{0}`,
    :math:`\mathbb{E}\left[\mathbf{x}\mathbf{x}^{\mathsf{H}}\right]=\mathbf{I}_K` and
    :math:`\mathbb{E}\left[\mathbf{n}\mathbf{n}^{\mathsf{H}}\right]=\mathbf{S}`.

    The estimated symbol vector :math:`\hat{\mathbf{x}}\in\mathbb{C}^K` is given as
    (Lemma B.19) [BHS2017]_ :

    .. math::

        \hat{\mathbf{x}} = \mathop{\text{diag}}\left(\mathbf{G}\mathbf{H}\right)^{-1}\mathbf{G}\mathbf{y}

    where

    .. math::

        \mathbf{G} = \mathbf{H}^{\mathsf{H}} \left(\mathbf{H}\mathbf{H}^{\mathsf{H}} + \mathbf{S}\right)^{-1}.

    This leads to the post-equalized per-symbol model:

    .. math::

        \hat{x}_k = x_k + e_k,\quad k=0,\dots,K-1

    where the variances :math:`\sigma^2_k` of the effective residual noise
    terms :math:`e_k` are given by the diagonal elements of

    .. math::

        \mathop{\text{diag}}\left(\mathbb{E}\left[\mathbf{e}\mathbf{e}^{\mathsf{H}}\right]\right)
        = \mathop{\text{diag}}\left(\mathbf{G}\mathbf{H} \right)^{-1} - \mathbf{I}.

    Note that the scaling by :math:`\mathop{\text{diag}}\left(\mathbf{G}\mathbf{H}\right)^{-1}`
    is important for the :class:`~sionna.phy.mapping.Demapper` although it does
    not change the signal-to-noise ratio.

    The function returns :math:`\hat{\mathbf{x}}` and
    :math:`\boldsymbol{\sigma}^2=\left[\sigma^2_0,\dots, \sigma^2_{K-1}\right]^{\mathsf{T}}`.

    Input
    -----
    y : [...,M], `tf.complex`
        Received signals

    h : [...,M,K], `tf.complex`
        Channel matrices

    s : [...,M,M], `tf.complex`
        Noise covariance matrices

    whiten_interference : `bool`, (default `True`)
        If `True`, the interference is first whitened before equalization.
        In this case, an alternative expression for the receive filter is used that
        can be numerically more stable.

    precision : `None` (default) | "single" | "double"
        Precision used for internal calculations and outputs.
        If set to `None`,
        :attr:`~sionna.phy.config.Config.precision` is used.

    Output
    ------
    x_hat : [...,K], `tf.complex`
        Estimated symbol vectors

    no_eff : `tf.float`
        Effective noise variance estimates
    """
    # Cast inputs
    if precision is None:
        cdtype = config.tf_cdtype
    else:
        cdtype = dtypes[precision]["tf"]["cdtype"]
    y = tf.cast(y, dtype=cdtype)
    h = tf.cast(h, dtype=cdtype)
    s = tf.cast(s, dtype=cdtype)

    # We assume the model:
    # y = Hx + n, where E[nn']=S.
    # E[x]=E[n]=0
    #
    # The LMMSE estimate of x is given as:
    # x_hat = diag(GH)^(-1)Gy
    # with G=H'(HH'+S)^(-1).
    #
    # This leads us to the per-symbol model;
    #
    # x_hat_k = x_k + e_k
    #
    # The elements of the residual noise vector e have variance:
    # diag(E[ee']) = diag(GH)^(-1) - I
    if not whiten_interference:
        # Compute equalizer matrix G
        g = lmmse_matrix(h, s, precision=precision)
    else:
        # Whiten channel
        y, h = whiten_channel(y, h, s, return_s=False) # pylint: disable=unbalanced-tuple-unpacking

        # Compute equalizer matrix G
        g = lmmse_matrix(h, s=None, precision=precision)

    # Compute G @ y
    y = tf.expand_dims(y, -1)
    gy = tf.squeeze(tf.matmul(g, y), axis=-1)

    # Compute G @ H
    gh = tf.matmul(g, h)

    # Compute diag(G @ H)
    d = tf.linalg.diag_part(gh)

    # Compute x_hat = diag(G @ H)^-1 @ G @ y
    x_hat = gy / d

    # Compute residual error variance
    one = tf.cast(1, dtype=d.dtype)
    no_eff = tf.math.real(one/d - one)

    return x_hat, no_eff

def zf_equalizer(y, h, s, precision=None):
    # pylint: disable=line-too-long
    r"""Applies MIMO ZF Equalizer

    This function implements zero-forcing (ZF) equalization for a MIMO link, assuming the
    following model:

    .. math::

        \mathbf{y} = \mathbf{H}\mathbf{x} + \mathbf{n}

    where :math:`\mathbf{y}\in\mathbb{C}^M` is the received signal vector,
    :math:`\mathbf{x}\in\mathbb{C}^K` is the vector of transmitted symbols,
    :math:`\mathbf{H}\in\mathbb{C}^{M\times K}` is the known channel matrix,
    and :math:`\mathbf{n}\in\mathbb{C}^M` is a noise vector.
    It is assumed that :math:`\mathbb{E}\left[\mathbf{x}\right]=\mathbb{E}\left[\mathbf{n}\right]=\mathbf{0}`,
    :math:`\mathbb{E}\left[\mathbf{x}\mathbf{x}^{\mathsf{H}}\right]=\mathbf{I}_K` and
    :math:`\mathbb{E}\left[\mathbf{n}\mathbf{n}^{\mathsf{H}}\right]=\mathbf{S}`.

    The estimated symbol vector :math:`\hat{\mathbf{x}}\in\mathbb{C}^K` is given as
    (Eq. 4.10) [BHS2017]_ :

    .. math::

        \hat{\mathbf{x}} = \mathbf{G}\mathbf{y}

    where

    .. math::

        \mathbf{G} = \left(\mathbf{H}^{\mathsf{H}}\mathbf{H}\right)^{-1}\mathbf{H}^{\mathsf{H}}.

    This leads to the post-equalized per-symbol model:

    .. math::

        \hat{x}_k = x_k + e_k,\quad k=0,\dots,K-1

    where the variances :math:`\sigma^2_k` of the effective residual noise
    terms :math:`e_k` are given by the diagonal elements of the matrix

    .. math::

        \mathbb{E}\left[\mathbf{e}\mathbf{e}^{\mathsf{H}}\right]
        = \mathbf{G}\mathbf{S}\mathbf{G}^{\mathsf{H}}.

    The function returns :math:`\hat{\mathbf{x}}` and
    :math:`\boldsymbol{\sigma}^2=\left[\sigma^2_0,\dots, \sigma^2_{K-1}\right]^{\mathsf{T}}`.

    Input
    -----
    y : [...,M], `tf.complex`
        Received signals

    h : [...,M,K], `tf.complex`
        Channel matrices

    s : [...,M,M], `tf.complex`
        Noise covariance matrices

    precision : `None` (default) | "single" | "double"
        Precision used for internal calculations and outputs.
        If set to `None`,
        :attr:`~sionna.phy.config.Config.precision` is used.

    Output
    ------
    x_hat : [...,K], `tf.complex`
        Estimated symbol vectors

    no_eff : tf.float
        Effective noise variance estimates
    """
    # Cast inputs
    if precision is None:
        cdtype = config.tf_cdtype
    else:
        cdtype = dtypes[precision]["tf"]["cdtype"]
    y = tf.cast(y, dtype=cdtype)
    h = tf.cast(h, dtype=cdtype)
    s = tf.cast(s, dtype=cdtype)

    # We assume the model:
    # y = Hx + n, where E[nn']=S.
    # E[x]=E[n]=0
    #
    # The ZF estimate of x is given as:
    # x_hat = Gy
    # with G=(H'H')^(-1)H'.
    #
    # This leads us to the per-symbol model;
    #
    # x_hat_k = x_k + e_k
    #
    # The elements of the residual noise vector e have variance:
    # E[ee'] = GSG'

    # Compute G
    g = matrix_pinv(h)

    # Compute x_hat
    y = tf.expand_dims(y, -1)
    x_hat = tf.squeeze(tf.matmul(g, y), axis=-1)

    # Compute residual error variance
    gsg = tf.matmul(tf.matmul(g, s), g, adjoint_b=True)
    no_eff = tf.math.real(tf.linalg.diag_part(gsg))

    return x_hat, no_eff

def mf_equalizer(y, h, s, precision=None):
    # pylint: disable=line-too-long
    r"""MIMO Matched Filter (MF) Equalizer

    This function implements matched filter (MF) equalization for a
    MIMO link, assuming the following model:

    .. math::

        \mathbf{y} = \mathbf{H}\mathbf{x} + \mathbf{n}

    where :math:`\mathbf{y}\in\mathbb{C}^M` is the received signal vector,
    :math:`\mathbf{x}\in\mathbb{C}^K` is the vector of transmitted symbols,
    :math:`\mathbf{H}\in\mathbb{C}^{M\times K}` is the known channel matrix,
    and :math:`\mathbf{n}\in\mathbb{C}^M` is a noise vector.
    It is assumed that :math:`\mathbb{E}\left[\mathbf{x}\right]=\mathbb{E}\left[\mathbf{n}\right]=\mathbf{0}`,
    :math:`\mathbb{E}\left[\mathbf{x}\mathbf{x}^{\mathsf{H}}\right]=\mathbf{I}_K` and
    :math:`\mathbb{E}\left[\mathbf{n}\mathbf{n}^{\mathsf{H}}\right]=\mathbf{S}`.

    The estimated symbol vector :math:`\hat{\mathbf{x}}\in\mathbb{C}^K` is given as
    (Eq. 4.11) [BHS2017]_ :

    .. math::

        \hat{\mathbf{x}} = \mathbf{G}\mathbf{y}

    where

    .. math::

        \mathbf{G} = \mathop{\text{diag}}\left(\mathbf{H}^{\mathsf{H}}\mathbf{H}\right)^{-1}\mathbf{H}^{\mathsf{H}}.

    This leads to the post-equalized per-symbol model:

    .. math::

        \hat{x}_k = x_k + e_k,\quad k=0,\dots,K-1

    where the variances :math:`\sigma^2_k` of the effective residual noise
    terms :math:`e_k` are given by the diagonal elements of the matrix

    .. math::

        \mathbb{E}\left[\mathbf{e}\mathbf{e}^{\mathsf{H}}\right]
        = \left(\mathbf{I}-\mathbf{G}\mathbf{H} \right)\left(\mathbf{I}-\mathbf{G}\mathbf{H} \right)^{\mathsf{H}} + \mathbf{G}\mathbf{S}\mathbf{G}^{\mathsf{H}}.

    Note that the scaling by :math:`\mathop{\text{diag}}\left(\mathbf{H}^{\mathsf{H}}\mathbf{H}\right)^{-1}`
    in the definition of :math:`\mathbf{G}`
    is important for the :class:`~sionna.phy.mapping.Demapper` although it does
    not change the signal-to-noise ratio.

    The function returns :math:`\hat{\mathbf{x}}` and
    :math:`\boldsymbol{\sigma}^2=\left[\sigma^2_0,\dots, \sigma^2_{K-1}\right]^{\mathsf{T}}`.

    Input
    -----
    y : [...,M], `tf.complex`
        Received signals

    h : [...,M,K], `tf.complex`
        Channel matrices

    s : [...,M,M], `tf.complex`
        Noise covariance matrices

    precision : `None` (default) | "single" | "double"
        Precision used for internal calculations and outputs.
        If set to `None`,
        :attr:`~sionna.phy.config.Config.precision` is used.

    Output
    ------
    x_hat : [...,K], `tf.complex`
        Estimated symbol vectors

    no_eff : tf.float
        Effective noise variance estimates
    """
    # Cast inputs
    if precision is None:
        cdtype = config.tf_cdtype
    else:
        cdtype = dtypes[precision]["tf"]["cdtype"]
    y = tf.cast(y, dtype=cdtype)
    h = tf.cast(h, dtype=cdtype)
    s = tf.cast(s, dtype=cdtype)

    # We assume the model:
    # y = Hx + n, where E[nn']=S.
    # E[x]=E[n]=0
    #
    # The MF estimate of x is given as:
    # x_hat = Gy
    # with G=diag(H'H)^-1 H'.
    #
    # This leads us to the per-symbol model;
    #
    # x_hat_k = x_k + e_k
    #
    # The elements of the residual noise vector e have variance:
    # E[ee'] = (I-GH)(I-GH)' + GSG'

    # Compute G
    hth = tf.matmul(h, h, adjoint_a=True)
    d = tf.linalg.diag(tf.cast(1, h.dtype)/tf.linalg.diag_part(hth))
    g = tf.matmul(d, h, adjoint_b=True)

    # Compute x_hat
    y = tf.expand_dims(y, -1)
    x_hat = tf.squeeze(tf.matmul(g, y), axis=-1)

    # Compute residual error variance
    gsg = tf.matmul(tf.matmul(g, s), g, adjoint_b=True)
    gh = tf.matmul(g, h)
    i = expand_to_rank(tf.eye(gsg.shape[-2], dtype=gsg.dtype), tf.rank(gsg), 0)

    no_eff = tf.abs(tf.linalg.diag_part(tf.matmul(i-gh, i-gh, adjoint_b=True) + gsg))

    return x_hat, no_eff
