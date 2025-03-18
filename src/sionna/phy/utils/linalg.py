#
# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0#
"""Functions extending TensorFlow linear algebra operations"""

import tensorflow as tf

def inv_cholesky(tensor):
    r"""Inverse of the Cholesky decomposition of a matrix

    Given a batch of :math:`M \times M` Hermitian positive definite
    matrices :math:`\mathbf{A}`, this function computes
    :math:`\mathbf{L}^{-1}`, where :math:`\mathbf{L}` is
    the Cholesky decomposition, such that
    :math:`\mathbf{A}=\mathbf{L}\mathbf{L}^{\textsf{H}}`.

    Input
    -----
    tensor : [..., M, M], `tf.float` | `tf.complex`
        Input tensor of rank greater than one

    Output
    ------
    : [..., M, M], `tf.float` | `tf.complex`
        A tensor of the same shape and type as ``tensor`` containing
        the inverse of the Cholesky decomposition of its last two dimensions
    """
    l = tf.linalg.cholesky(tensor)
    rhs = tf.eye(num_rows=tf.shape(l)[-1],
                 batch_shape=tf.shape(l)[:-2],
                 dtype=l.dtype)
    return tf.linalg.triangular_solve(l, rhs, lower=True)

def matrix_pinv(tensor):
    r"""Computes the Moore–Penrose (or pseudo) inverse of a matrix

    Given a batch of :math:`M \times K` matrices :math:`\mathbf{A}` with rank
    :math:`K` (i.e., linearly independent columns), the function returns
    :math:`\mathbf{A}^+`, such that
    :math:`\mathbf{A}^{+}\mathbf{A}=\mathbf{I}_K`.

    The two inner dimensions are assumed to correspond to the matrix rows
    and columns, respectively.

    Input
    -----
    tensor : [..., M, K], `tf.Tensor`
        Input tensor of rank greater than or equal to two

    Output
    ------
    : [..., M, K], `tf.Tensor`
        A tensor of the same shape and type as ``tensor`` containing
        the matrix pseudo inverse of its last two dimensions
    """
    tensor_tensor_h = tf.matmul(tensor, tensor, adjoint_a=True)
    l = tf.linalg.cholesky(tensor_tensor_h)
    return tf.linalg.cholesky_solve(l, tf.linalg.adjoint(tensor))

