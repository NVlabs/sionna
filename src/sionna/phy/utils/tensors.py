#
# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0#
"""Functions extending TensorFlow tensor operations for Sionna PHY and SYS"""

import tensorflow as tf
from sionna.phy import config

def expand_to_rank(tensor, target_rank, axis=-1):
    """Inserts as many axes to a tensor as needed to achieve a desired rank

    This operation inserts additional dimensions to a ``tensor`` starting at
    ``axis``, so that so that the rank of the resulting tensor has rank
    ``target_rank``. The dimension index follows Python indexing rules, i.e.,
    zero-based, where a negative index is counted backward from the end.

    Input
    -----
    tensor : `tf.Tensor`
        Input tensor

    target_rank : `int`
        Rank of the output tensor.
        If ``target_rank`` is smaller than the rank of ``tensor``,
        the function does nothing.

    axis : `int`
        Dimension index at which to expand the
        shape of ``tensor``. Given a ``tensor`` of `D` dimensions,
        ``axis`` must be within the range `[-(D+1), D]` (inclusive).

    Output
    ------
     : `tf.Tensor`
        A tensor with the same data as ``tensor``, with
        ``target_rank``- rank(``tensor``) additional dimensions inserted at the
        index specified by ``axis``.
        If ``target_rank`` <= rank(``tensor``), ``tensor`` is returned.
    """
    num_dims = tf.maximum(target_rank - tf.rank(tensor), 0)
    output = insert_dims(tensor, num_dims, axis)
    return output

def flatten_dims(tensor, num_dims, axis):
    """
    Flattens a specified set of dimensions of a tensor

    This operation flattens ``num_dims`` dimensions of a ``tensor``
    starting at a given ``axis``.

    Input
    -----
    tensor : `tf.Tensor`
        Input tensor

    num_dims : `int`
        Number of dimensions to combine. Must be larger than
        two and less or equal than the rank of ``tensor``.

    axis : `int` 
        Index of the dimension from which to start

    Output
    ------
    : `tf.Tensor`
        A tensor of the same type as ``tensor`` with ``num_dims``-1 lesser
        dimensions, but the same number of elements
    """
    msg = "`num_dims` must be >= 2"
    tf.debugging.assert_greater_equal(num_dims, 2, msg)

    msg = "`num_dims` must <= rank(`tensor`)"
    tf.debugging.assert_less_equal(num_dims, tf.rank(tensor), msg)

    msg = "0<= `axis` <= rank(tensor)-1"
    tf.debugging.assert_less_equal(axis, tf.rank(tensor)-1, msg)
    tf.debugging.assert_greater_equal(axis, 0, msg)

    msg ="`num_dims`+`axis` <= rank(`tensor`)"
    tf.debugging.assert_less_equal(num_dims + axis, tf.rank(tensor), msg)

    if num_dims==len(tensor.shape):
        new_shape = [-1]
    elif axis==0:
        shape = tf.shape(tensor)
        new_shape = tf.concat([[-1], shape[axis+num_dims:]], 0)
    else:
        shape = tf.shape(tensor)
        flat_dim = tf.reduce_prod(tensor.shape[axis:axis+num_dims])
        new_shape = tf.concat([shape[:axis],
                               [flat_dim],
                               shape[axis+num_dims:]], 0)

    return tf.reshape(tensor, new_shape)

def flatten_last_dims(tensor, num_dims=2):
    """
    Flattens the last `n` dimensions of a tensor

    This operation flattens the last ``num_dims`` dimensions of a ``tensor``.
    It is a simplified version of the function ``flatten_dims``.

    Input
    -----
    tensor : `tf.Tensor`
        Input tensor

    num_dims : `int`
        Number of dimensions to combine.
        Must be greater than or equal to two and less or equal
        than the rank of ``tensor``.

    Output
    ------
    : `tf.Tensor`
        A tensor of the same type as ``tensor`` with ``num_dims``-1 lesser
        dimensions, but the same number of elements
    """
    msg = "`num_dims` must be >= 2"
    tf.debugging.assert_greater_equal(num_dims, 2, msg)

    msg = "`num_dims` must <= rank(`tensor`)"
    tf.debugging.assert_less_equal(num_dims, tf.rank(tensor), msg)

    if num_dims==len(tensor.shape):
        new_shape = [-1]
    else:
        shape = tf.shape(tensor)
        last_dim = tf.reduce_prod(tensor.shape[-num_dims:])
        new_shape = tf.concat([shape[:-num_dims], [last_dim]], 0)

    return tf.reshape(tensor, new_shape)

def insert_dims(tensor, num_dims, axis=-1):
    """Adds multiple length-one dimensions to a tensor

    This operation is an extension to TensorFlow`s ``expand_dims`` function.
    It inserts ``num_dims`` dimensions of length one starting from the
    dimension ``axis`` of a ``tensor``. The dimension
    index follows Python indexing rules, i.e., zero-based, where a negative
    index is counted backward from the end.

    Input
    -----
    tensor : `tf.Tensor`
        Input tensor

    num_dims : `int`
        Number of dimensions to add

    axis : `int`
        Dimension index at which to expand the
        shape of ``tensor``. Given a ``tensor`` of `D` dimensions,
        ``axis`` must be within the range `[-(D+1), D]` (inclusive).

    Output
    ------
    : `tf.tensor`
        A tensor with the same data as ``tensor``, with ``num_dims`` additional
        dimensions inserted at the index specified by ``axis``
    """
    msg = "`num_dims` must be nonnegative."
    tf.debugging.assert_greater_equal(num_dims, 0, msg)

    rank = tf.rank(tensor)
    msg = "`axis` is out of range `[-(D+1), D]`)"
    tf.debugging.assert_less_equal(axis, rank, msg)
    tf.debugging.assert_greater_equal(axis, -(rank+1), msg)

    axis = axis if axis>=0 else rank+axis+1
    shape = tf.shape(tensor)
    new_shape = tf.concat([shape[:axis],
                           tf.ones([num_dims], tf.int32),
                           shape[axis:]], 0)
    output = tf.reshape(tensor, new_shape)

    return output

def split_dim(tensor, shape, axis):
    """Reshapes a dimension of a tensor into multiple dimensions

    This operation splits the dimension ``axis`` of a ``tensor`` into
    multiple dimensions according to ``shape``.

    Input
    -----
    tensor : `tf.Tensor`
        Input tensor

    shape : (list or TensorShape)
        Shape to which the dimension should
        be reshaped

    axis : `int`
        Index of the axis to be reshaped

    Output
    ------
    : `tf.Tensor`
        A tensor of the same type as ``tensor`` with len(``shape``)-1
        additional dimensions, but the same number of elements
    """
    msg = "0<= `axis` <= rank(tensor)-1"
    tf.debugging.assert_less_equal(axis, tf.rank(tensor)-1, msg)
    tf.debugging.assert_greater_equal(axis, 0, msg)

    s = tf.shape(tensor)
    new_shape = tf.concat([s[:axis], shape, s[axis+1:]], 0)
    output = tf.reshape(tensor, new_shape)

    return output

def diag_part_axis(tensor,
                   axis,
                   **kwargs):
    # pylint: disable=line-too-long
    r"""
    Extracts the batched diagonal part of a batched tensor over the specified axis
    
    This is an extension of TensorFlow`s ``tf.linalg.diag_part`` function, which
    extracts the diagonal over the last two dimensions. This behavior can be
    reproduced by setting ``axis`` =-2.

    Input
    -----

    tensor : [s(1), ..., s(N)], `any`
        A tensor of rank greater than or equal to two (:math:`N\ge 2`)

    axis : `int`
        Axis index starting from which the diagonal part is
        extracted

    kwargs : `dict`
        Optional inputs for TensorFlow's
        `linalg.diag_part`, such as the diagonal offset ``k`` or the
        padding value ``padding_value``. See TensorFlow's `linalg.diag_part` for
        more details.

    Output
    ------
    : [s(1), ..., min[s(``axis``),s(``axis`` +1)], s(``axis`` +2), ..., s(N))], `any`
        Tensor containing the diagonal part of input ``tensor`` over axis
        (``axis``, ``axis`` +1)

    Example
    -------

    .. code-block:: Python

        import tensorflow as tf
        from sionna.phy.utils import diag_part_axis

        a = tf.reshape(tf.range(27), [3,3,3])
        print(a.numpy())
        #  [[[ 0  1  2]
        #    [ 3  4  5]
        #    [ 6  7  8]]
        #
        #    [[ 9 10 11]
        #    [12 13 14]
        #    [15 16 17]]
        #
        #    [[18 19 20]
        #    [21 22 23]
        #    [24 25 26]]]
        
        dp_0 = diag_part_axis(a, axis=0)
        print(dp_0.numpy())
        # [[ 0  1  2]
        #  [12 13 14]
        #  [24 25 26]]
        
        dp_1 = diag_part_axis(a, axis=1)
        print(dp_1.numpy())
        # [[ 0  4  8]
        #  [ 9 13 17]
        #  [18 22 26]]
    """
    tf.debugging.assert_rank_at_least(tensor, 2,
        message='The input tensor must have rank >= 2.')

    if axis < 0:
        axis = tf.rank(tensor) + axis

    shape_in = tf.shape(tensor)
    tf.debugging.assert_equal((axis < 0) | (axis > len(shape_in)-2),
                              False,
                              message="Input value of 'axis' out of boundaries.")

    if 'k' not in kwargs:
        len_k = 0
    else:
        if hasattr(kwargs['k'], '__len__'):
            len_k = 1
        else:
            len_k = 0

    # Push the axis (axis,axis+1) to the last dimensions
    index1 = tf.concat([range(axis),
                        range(axis+2, len(shape_in)),
                        range(axis, axis+2)], 0)
    tensor_out = tf.transpose(tensor, index1)

    # Extract the diagonal part out of the 2 innermost dimensions
    tensor_out = tf.linalg.diag_part(tensor_out,
                                     **kwargs)

    # Push the two last dimensions back to the original location
    index2 = tf.concat([range(axis),
                        range(len(tensor_out.shape)-1-len_k,
                              len(tensor_out.shape)),
                        range(axis, len(tensor_out.shape)-1-len_k)], 0)
    tensor_out = tf.transpose(tensor_out, index2)

    return tensor_out


def flatten_multi_index(indices, shape):
    # pylint: disable=line-too-long
    r"""
    Converts a tensor of index arrays into an tensor of flat indices

    Input
    -----

    indices : [..., N], `tf.int32`
        Indices to flatten

    shape : [N], `tf.int32`
        Shape of each index dimension.
        Note that it must hold that ``indices[..., n]<shape[n]`` for all n and
        batch dimension

    Output
    ------

    flat_indices : [...], `tf.int32`
        Flattened indices

    Example
    -------

    .. code-block:: Python

        import tensorflow as tf
        from sionna.phy.utils import flatten_multi_index

        indices = tf.constant([2, 3])
        shape = [5, 6]
        print(flatten_multi_index(indices, shape).numpy())
        # 15 = 2*6 + 3

    """
    indices = tf.cast(indices, tf.int32)
    batch_rank = tf.rank(indices) - 1

    # Assert that indices are within valid bounds
    tf.debugging.assert_less(indices, insert_dims(shape, batch_rank, axis=0))
    tf.debugging.assert_non_negative(indices)

    # strides = [prod(shape[1:]), prod(shape[2,:]),...,shape[-1], 1]
    strides = tf.math.cumprod([1] + shape[::-1][:-1])[::-1]
    strides = insert_dims(strides, batch_rank, axis=0)

    flat_indices = tf.reduce_sum(strides * indices, axis=-1)
    return flat_indices


def gather_from_batched_indices(params, indices):
    # pylint: disable=line-too-long
    r"""
    Gathers the values of a tensor ``params`` according to batch-specific
    ``indices`` 

    Input
    -----
    params : [s(1), ..., s(N)], `any`
        Tensor containing the values to gather

    indices : [..., N], `tf.int32`
        Tensor containing, for each batch `[...]`, the indices at which
        ``params`` is gathered. Note that 0 :math:`\le`
        ``indices[...,n]`` :math:`<` `s(n)` must hold for all `n=1,...,N`

    Output
    ------
    : [...], `any`
        Tensor containing the gathered values

    Example
    -------

    .. code-block:: Python

        import tensorflow as tf
        from sionna.phy.utils import gather_from_batched_indices
        
        params = tf.constant([[10, 20, 30], [40, 50, 60], [70, 80, 90]])
        print(params.shape)
        # TensorShape([3, 3])
        
        indices = tf.constant([[[0, 1], [1, 2], [2, 0], [0, 0]],
                               [[0, 0], [2, 2], [2, 1], [0, 1]]])
        print(indices.shape)
        # TensorShape([2, 4, 2])
        # Note that the batch shape is [2, 4]. Each batch contains a list of 2 indices 

        print(gather_from_batched_indices(params, indices).numpy())
        # [[20, 60, 70, 10],
        #  [10, 90, 80, 20]]
        # Note that the output shape coincides with the batch shape.
        # Element [i,j] coincides with params[indices[i,j,:]]
    
    """
    # flatten indices
    flat_indices = flatten_multi_index(indices, shape=params.shape)
    # gather according to the flattened indices
    return tf.gather(tf.reshape(params, [-1]), flat_indices)


def tensor_values_are_in_set(tensor,
                             admissible_set):
    r"""
    Checks if the input ``tensor`` values are contained in the
    specified ``admissible_set`` 

    Input
    -----

    tensor : `tf.Tensor` | `list`
        Tensor to validate
    
    admissible_set : `tf.Tensor` | `list`
        Set of valid values that the input ``tensor`` must be composed of 

    Output
    ------
    : `bool`
        Returns `True` if and only if ``tensor`` values are contained in
        ``admissible_set`` 
    
    Example
    -------

    .. code-block:: Python

        import tensorflow as tf
        from sionna.phy.utils import tensor_values_are_in_set

        tensor = tf.Variable([[1, 0], [0, 1]])
        
        print(tensor_values_are_in_set(tensor, [0, 1, 2]).numpy())
        # True

        print(tensor_values_are_in_set(tensor, [0, 2]).numpy())
        # False
    """

    # Flatten tensors
    tensor_flat = tf.reshape(tensor, [-1])  # Shape: [num_values]
    admissible_set_flat = tf.reshape(admissible_set, [-1])  # Shape: [set_size]

    # element [i] = 1 if tensor_flat[i] is found in admissible_set, else 0
    # [len(tensor_unique)]
    value_is_admissible = tf.reduce_any(
        tf.equal(tf.expand_dims(tensor_flat, axis=0),  # [1, -1]
                 tf.expand_dims(admissible_set_flat, axis=1)),  # [-1, 1]
        axis=0)

    # Whether all tensor values are contained in admissible set
    return tf.reduce_all(value_is_admissible)


def random_tensor_from_values(values, shape, dtype=None):
    r"""
    Generates a tensor of the specified ``shape``, with elements randomly
    sampled  from the provided set of ``values``

    Input
    -----
    values : `tf.Tensor` | `list`
        The set of values to sample from

    shape : `tf.Tensor` | `list`
        The desired shape of the output tensor

    dtype : `tf.dtype`
        Desired dtype of the output

    Returns
    -------
    : `tf.Tensor`
        A tensor with the specified shape, where each element is randomly 
        selected from ``values``

    Example
    -------

    .. code-block:: Python

        from sionna.phy.utils import random_tensor_from_values

        values = [0, 10, 20]
        shape = [2, 3]
        print(random_tensor_from_values(values, shape).numpy())
        # array([[ 0, 20,  0],
        #        [10,  0, 20]], dtype=int32)
    """
    num_elements = tf.reduce_prod(shape)
    indices = config.tf_rng.uniform(shape=(num_elements,),
                                       minval=0,
                                       maxval=len(values),
                                       dtype=tf.int32)
    tensor = tf.reshape(tf.gather(values, indices), shape)
    if dtype is not None:
        tensor = tf.cast(tensor, dtype)
    return tensor


def enumerate_indices(bounds):
    r"""
    Enumerates all indices between 0 (included) and ``bounds`` (excluded) in
    lexicographic order

    Input
    -----

    bounds : `list` | `tf.Tensor` | `np.array`, `int`
        Collection of index bounds
    
    Output
    ------

    : [prod(bounds), len(bounds)]
        Collection of all indices, in lexicographic order

    Example
    -------

    .. code-block:: Python

        from sionna.phy.utils import enumerate_indices
    
        print(enumerate_indices([2, 3]).numpy())
        # [[0 0]
        #  [0 1]
        #  [0 2]
        #  [1 0]
        #  [1 1]
        #  [1 2]]
    """
    # Flattened indices: range from 0 to total number of elements
    idx_flat = tf.range(tf.reduce_prod(bounds))

    # Convert flattened indices to multi-dimensional indices
    idx = tf.unravel_index(idx_flat, dims=bounds)

    # Transpose
    return tf.transpose(idx, [1, 0])


def find_true_position(bool_tensor,
                       side='last',
                       axis=-1):
    """
    Finds the index of the first or last (according to the value of ``side``)
    `True` value along the specified axis. 
    When no `True` value is present, it returns -1 

    Input
    -----

    bool_tensor : `tf.bool`
        Boolean tensor of any shape

    side : "first" | "last"
        If "first", the first `True` position is found, else the last

    axis : `int` (default: -1)
        Axis along which to find the last `True` value

    Output
    ------

    index : `tf.int32`
        Tensor of indices, containing the index of the first or last `True`
        value.
        Its shape is ``bool_tensor.shape`` with specified ``axis`` removed 
    """
    tf.debugging.assert_equal(side in ['first', 'last'],
                              True,
                              message="input side must be 'first' or 'last'")
    rank = tf.rank(bool_tensor)
    shape = tf.shape(bool_tensor)

    # Convert to positive axis
    axis = rank + axis if axis < 0 else axis

    # Create sequence of indices
    # [1, ..., shape[axis], 1, ..., 1]
    indices = tf.range(shape[axis], dtype=tf.int32)
    indices = insert_dims(indices, axis, axis=0)
    indices = insert_dims(indices, rank - axis - 1, axis=-1)

    # Broadcast to shape
    # multiples = tf.tensor_scatter_nd_update(shape, [axis], [1])
    # indices = tf.tile(indices, multiples)
    indices = tf.broadcast_to(indices, shape)

    if side == 'last':
        # Where tensor is True, use computed indices, else set to -1
        masked_indices = tf.where(bool_tensor, indices, -1)

        # Get maximum index (last True position)
        index = tf.reduce_max(masked_indices, axis=axis)
    else:
        # Where tensor is True, use computed indices, else set to shape[axis]
        masked_indices = tf.where(bool_tensor, indices, shape[axis])

        # Get minimum index (first True position)
        index = tf.reduce_min(masked_indices, axis=axis)
        # If not found, return -1
        index = tf.where(index != shape[axis], index, -1)
    return index
