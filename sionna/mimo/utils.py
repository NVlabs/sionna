#
# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Utility functions and layers for the MIMO package."""

import tensorflow as tf
from sionna.utils import matrix_sqrt_inv, expand_to_rank

def complex2real_vector(z):
    # pylint: disable=line-too-long
    r"""Transforms a complex-valued vector into its real-valued equivalent.

    Transforms the last dimension of a complex-valued tensor into
    its real-valued equivalent by stacking the real and imaginary
    parts on top of each other.

    For a vector :math:`\mathbf{z}\in \mathbb{C}^M` with real and imaginary
    parts :math:`\mathbf{x}\in \mathbb{R}^M` and
    :math:`\mathbf{y}\in \mathbb{R}^M`, respectively, this function returns
    the vector :math:`\left[\mathbf{x}^{\mathsf{T}}, \mathbf{y}^{\mathsf{T}} \right ]^{\mathsf{T}}\in\mathbb{R}^{2M}`.

    Input
    -----
    : [...,M], tf.complex

    Output
    ------
    : [...,2M], tf.complex.real_dtype
    """
    x = tf.math.real(z)
    y = tf.math.imag(z)
    return tf.concat([x, y], axis=-1)

def real2complex_vector(z):
# pylint: disable=line-too-long
    r"""Transforms a real-valued vector into its complex-valued equivalent.

    Transforms the last dimension of a real-valued tensor into
    its complex-valued equivalent by interpreting the first half
    as the real and the second half as the imaginary part.

    For a vector :math:`\mathbf{z}=\left[\mathbf{x}^{\mathsf{T}}, \mathbf{y}^{\mathsf{T}} \right ]^{\mathsf{T}}\in \mathbb{R}^{2M}`
    with :math:`\mathbf{x}\in \mathbb{R}^M` and :math:`\mathbf{y}\in \mathbb{R}^M`,
    this function returns
    the vector :math:`\mathbf{x}+j\mathbf{y}\in\mathbb{C}^M`.

    Input
    -----
    : [...,2M], tf.real

    Output
    ------
    : [...,M], tf.complex
    """
    x, y = tf.split(z, 2, -1)
    return tf.complex(x, y)

def complex2real_matrix(z):
    # pylint: disable=line-too-long
    r"""Transforms a complex-valued matrix into its real-valued equivalent.

    Transforms the last two dimensions of a complex-valued tensor into
    their real-valued matrix equivalent representation.

    For a matrix :math:`\mathbf{Z}\in \mathbb{C}^{M\times K}` with real and imaginary
    parts :math:`\mathbf{X}\in \mathbb{R}^{M\times K}` and
    :math:`\mathbf{Y}\in \mathbb{R}^{M\times K}`, respectively, this function returns
    the matrix :math:`\tilde{\mathbf{Z}}\in \mathbb{R}^{2M\times 2K}`, given as

    .. math::

        \tilde{\mathbf{Z}} = \begin{pmatrix}
                                \mathbf{X} & -\mathbf{Y}\\
                                \mathbf{Y} & \mathbf{X}
                             \end{pmatrix}.

    Input
    -----
    : [...,M,K], tf.complex

    Output
    ------
    : [...,2M, 2K], tf.complex.real_dtype
    """
    x = tf.math.real(z)
    y = tf.math.imag(z)
    row1 = tf.concat([x, -y], axis=-1)
    row2 = tf.concat([y, x], axis=-1)
    return tf.concat([row1, row2], axis=-2)

def real2complex_matrix(z):
    # pylint: disable=line-too-long
    r"""Transforms a real-valued matrix into its complex-valued equivalent.

    Transforms the last two dimensions of a real-valued tensor into
    their complex-valued matrix equivalent representation.

    For a matrix :math:`\tilde{\mathbf{Z}}\in \mathbb{R}^{2M\times 2K}`,
    satisfying

    .. math::

        \tilde{\mathbf{Z}} = \begin{pmatrix}
                                \mathbf{X} & -\mathbf{Y}\\
                                \mathbf{Y} & \mathbf{X}
                             \end{pmatrix}

    with :math:`\mathbf{X}\in \mathbb{R}^{M\times K}` and
    :math:`\mathbf{Y}\in \mathbb{R}^{M\times K}`, this function returns
    the matrix :math:`\mathbf{Z}=\mathbf{X}+j\mathbf{Y}\in\mathbb{C}^{M\times K}`.

    Input
    -----
    : [...,2M,2K], tf.real

    Output
    ------
    : [...,M, 2], tf.complex
    """
    m = tf.shape(z)[-2]//2
    k = tf.shape(z)[-1]//2
    x = z[...,:m,:k]
    y = z[...,m:,:k]
    return tf.complex(x, y)

def complex2real_covariance(r):
    # pylint: disable=line-too-long
    r"""Transforms a complex-valued covariance matrix to its real-valued equivalent.

    Assume a proper complex random variable :math:`\mathbf{z}\in\mathbb{C}^M` [ProperRV]_
    with covariance matrix :math:`\mathbf{R}= \in\mathbb{C}^{M\times M}`
    and real and imaginary parts :math:`\mathbf{x}\in \mathbb{R}^M` and
    :math:`\mathbf{y}\in \mathbb{R}^M`, respectively.
    This function transforms the given :math:`\mathbf{R}` into the covariance matrix of the real-valued equivalent
    vector :math:`\tilde{\mathbf{z}}=\left[\mathbf{x}^{\mathsf{T}}, \mathbf{y}^{\mathsf{T}} \right ]^{\mathsf{T}}\in\mathbb{R}^{2M}`, which
    is computed as [CovProperRV]_

    .. math::

        \mathbb{E}\left[\tilde{\mathbf{z}}\tilde{\mathbf{z}}^{\mathsf{H}} \right] =
        \begin{pmatrix}
            \frac12\Re\{\mathbf{R}\} & -\frac12\Im\{\mathbf{R}\}\\
            \frac12\Im\{\mathbf{R}\} & \frac12\Re\{\mathbf{R}\}
        \end{pmatrix}.

    Input
    -----
    : [...,M,M], tf.complex

    Output
    ------
    : [...,2M, 2M], tf.complex.real_dtype
    """
    q = complex2real_matrix(r)
    scale = tf.cast(2, q.dtype)
    return q/scale

def real2complex_covariance(q):
    # pylint: disable=line-too-long
    r"""Transforms a real-valued covariance matrix to its complex-valued equivalent.

    Assume a proper complex random variable :math:`\mathbf{z}\in\mathbb{C}^M` [ProperRV]_
    with covariance matrix :math:`\mathbf{R}= \in\mathbb{C}^{M\times M}`
    and real and imaginary parts :math:`\mathbf{x}\in \mathbb{R}^M` and
    :math:`\mathbf{y}\in \mathbb{R}^M`, respectively.
    This function transforms the given covariance matrix of the real-valued equivalent
    vector :math:`\tilde{\mathbf{z}}=\left[\mathbf{x}^{\mathsf{T}}, \mathbf{y}^{\mathsf{T}} \right ]^{\mathsf{T}}\in\mathbb{R}^{2M}`, which
    is given as [CovProperRV]_

    .. math::

        \mathbb{E}\left[\tilde{\mathbf{z}}\tilde{\mathbf{z}}^{\mathsf{H}} \right] =
        \begin{pmatrix}
            \frac12\Re\{\mathbf{R}\} & -\frac12\Im\{\mathbf{R}\}\\
            \frac12\Im\{\mathbf{R}\} & \frac12\Re\{\mathbf{R}\}
        \end{pmatrix},

    into is complex-valued equivalent :math:`\mathbf{R}`.

    Input
    -----
    : [...,2M,2M], tf.real

    Output
    ------
    : [...,M, M], tf.complex
    """
    r = real2complex_matrix(q)
    scale = tf.cast(2, r.dtype)
    return r*scale

def complex2real_channel(y, h, s):
    # pylint: disable=line-too-long
    r"""Transforms a complex-valued MIMO channel into its real-valued equivalent.

    Assume the canonical MIMO channel model

    .. math::

        \mathbf{y} = \mathbf{H}\mathbf{x} + \mathbf{n}

    where :math:`\mathbf{y}\in\mathbb{C}^M` is the received signal vector,
    :math:`\mathbf{x}\in\mathbb{C}^K` is the vector of transmitted symbols,
    :math:`\mathbf{H}\in\mathbb{C}^{M\times K}` is the known channel matrix,
    and :math:`\mathbf{n}\in\mathbb{C}^M` is a noise vector with covariance
    matrix :math:`\mathbf{S}\in\mathbb{C}^{M\times M}`.

    This function returns the real-valued equivalent representations of
    :math:`\mathbf{y}`, :math:`\mathbf{H}`, and :math:`\mathbf{S}`,
    which are used by a wide variety of MIMO detection algorithms (Section VII) [YH2015]_.
    These are obtained by applying :meth:`~sionna.mimo.complex2real_vector` to :math:`\mathbf{y}`,
    :meth:`~sionna.mimo.complex2real_matrix` to :math:`\mathbf{H}`,
    and :meth:`~sionna.mimo.complex2real_covariance` to :math:`\mathbf{S}`.

    Input
    -----
    y : [...,M], tf.complex
        1+D tensor containing the received signals.

    h : [...,M,K], tf.complex
        2+D tensor containing the channel matrices.

    s : [...,M,M], tf.complex
        2+D tensor containing the noise covariance matrices.

    Output
    ------
    : [...,2M], tf.complex.real_dtype
        1+D tensor containing the real-valued equivalent received signals.

    : [...,2M,2K], tf.complex.real_dtype
        2+D tensor containing the real-valued equivalent channel matrices.

    : [...,2M,2M], tf.complex.real_dtype
        2+D tensor containing the real-valued equivalent noise covariance matrices.
    """
    yr = complex2real_vector(y)
    hr = complex2real_matrix(h)
    sr = complex2real_covariance(s)
    return yr, hr, sr

def real2complex_channel(y, h, s):
    # pylint: disable=line-too-long
    r"""Transforms a real-valued MIMO channel into its complex-valued equivalent.

    Assume the canonical MIMO channel model

    .. math::

        \mathbf{y} = \mathbf{H}\mathbf{x} + \mathbf{n}

    where :math:`\mathbf{y}\in\mathbb{C}^M` is the received signal vector,
    :math:`\mathbf{x}\in\mathbb{C}^K` is the vector of transmitted symbols,
    :math:`\mathbf{H}\in\mathbb{C}^{M\times K}` is the known channel matrix,
    and :math:`\mathbf{n}\in\mathbb{C}^M` is a noise vector with covariance
    matrix :math:`\mathbf{S}\in\mathbb{C}^{M\times M}`.

    This function transforms the real-valued equivalent representations of
    :math:`\mathbf{y}`, :math:`\mathbf{H}`, and :math:`\mathbf{S}`, as, e.g.,
    obtained with the function :meth:`~sionna.mimo.complex2real_channel`,
    back to their complex-valued equivalents (Section VII) [YH2015]_.

    Input
    -----
    y : [...,2M], tf.real
        1+D tensor containing the real-valued received signals.

    h : [...,2M,2K], tf.real
        2+D tensor containing the real-valued channel matrices.

    s : [...,2M,2M], tf.real
        2+D tensor containing the real-valued noise covariance matrices.

    Output
    ------
    : [...,M], tf.complex
        1+D tensor containing the complex-valued equivalent received signals.

    : [...,M,K], tf.complex
        2+D tensor containing the complex-valued equivalent channel matrices.

    : [...,M,M], tf.complex
        2+D tensor containing the complex-valued equivalent noise covariance matrices.
    """
    yc = real2complex_vector(y)
    hc = real2complex_matrix(h)
    sc = real2complex_covariance(s)
    return yc, hc, sc

def whiten_channel(y, h, s, return_s=True):
    # pylint: disable=line-too-long
    r"""Whitens a canonical MIMO channel.

    Assume the canonical MIMO channel model

    .. math::

        \mathbf{y} = \mathbf{H}\mathbf{x} + \mathbf{n}

    where :math:`\mathbf{y}\in\mathbb{C}^M(\mathbb{R}^M)` is the received signal vector,
    :math:`\mathbf{x}\in\mathbb{C}^K(\mathbb{R}^K)` is the vector of transmitted symbols,
    :math:`\mathbf{H}\in\mathbb{C}^{M\times K}(\mathbb{R}^{M\times K})` is the known channel matrix,
    and :math:`\mathbf{n}\in\mathbb{C}^M(\mathbb{R}^M)` is a noise vector with covariance
    matrix :math:`\mathbf{S}\in\mathbb{C}^{M\times M}(\mathbb{R}^{M\times M})`.

    This function whitens this channel by multiplying :math:`\mathbf{y}` and
    :math:`\mathbf{H}` from the left by :math:`\mathbf{S}^{-\frac{1}{2}}`.
    Optionally, the whitened noise covariance matrix :math:`\mathbf{I}_M`
    can be returned.

    Input
    -----
    y : [...,M], tf.real or tf.complex
        1+D tensor containing the received signals.

    h : [...,M,K], tf.real or tf.complex
        2+D tensor containing the  channel matrices.

    s : [...,M,M], tf.real or complex
        2+D tensor containing the noise covariance matrices.

    return_s : bool
        If `True`, the whitened covariance matrix is returned.
        Defaults to `True`.

    Output
    ------
    : [...,M], tf.real or tf.complex
        1+D tensor containing the whitened received signals.

    : [...,M,K], tf.real or tf.complex
        2+D tensor containing the whitened channel matrices.

    : [...,M,M], tf.real or tf.complex
        2+D tensor containing the whitened noise covariance matrices.
        Only returned if ``return_s`` is `True`.
    """
    # Compute whitening matrix
    s_inv_1_2 = matrix_sqrt_inv(s)
    s_inv_1_2 = expand_to_rank(s_inv_1_2, tf.rank(h), 0)

    # Whiten obervation and channel matrix
    yw = tf.expand_dims(y, -1)
    yw = tf.matmul(s_inv_1_2, yw)
    yw = tf.squeeze(yw, axis=-1)

    hw = tf.matmul(s_inv_1_2, h)

    if return_s:
        # Ideal interference covariance matrix after whitening
        sw = tf.eye(tf.shape(s)[-2], dtype=s.dtype)
        sw = expand_to_rank(sw, tf.rank(s), 0)
        return yw, hw, sw
    else:
        return yw, hw
