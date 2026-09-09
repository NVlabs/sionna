#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Functions extending PyTorch tensor operations for Sionna PHY and SYS"""

from typing import Any, List, Optional, Union

import torch

from sionna._validation import check_tensor_all
from sionna.phy import config
from sionna.phy.utils.random import randint as _randint

__all__ = [
    "expand_to_rank",
    "flatten_dims",
    "flatten_last_dims",
    "insert_dims",
    "split_dim",
    "diag_part_axis",
    "flatten_multi_index",
    "gather_from_batched_indices",
    "tensor_values_are_in_set",
    "random_tensor_from_values",
    "enumerate_indices",
    "find_true_position",
]


def expand_to_rank(
    tensor: torch.Tensor, target_rank: int, axis: int = -1
) -> torch.Tensor:
    """Inserts as many axes to a tensor as needed to achieve a desired rank.

    This operation inserts additional dimensions to a ``tensor`` starting at
    ``axis``, so that the rank of the resulting tensor has rank
    ``target_rank``. The dimension index follows Python indexing rules, i.e.,
    zero-based, where a negative index is counted backward from the end.

    :param tensor: Input tensor
    :param target_rank: Rank of the output tensor.
        If ``target_rank`` is smaller than the rank of ``tensor``,
        the function does nothing.
    :param axis: Dimension index at which to expand the
        shape of ``tensor``. Given a ``tensor`` of `D` dimensions,
        ``axis`` must be within the range `[-(D+1), D]` (inclusive).

    :output tensor: A tensor with the same data as ``tensor``, with
        ``target_rank`` - rank(``tensor``) additional dimensions inserted at the
        index specified by ``axis``.
        If ``target_rank`` <= rank(``tensor``), ``tensor`` is returned.

    .. rubric:: Examples

    .. code-block:: python

        import torch
        from sionna.phy.utils import expand_to_rank

        x = torch.ones([3, 4])
        print(x.shape)
        # torch.Size([3, 4])

        y = expand_to_rank(x, 4, axis=-1)
        print(y.shape)
        # torch.Size([3, 4, 1, 1])
    """
    num_dims = max(target_rank - tensor.dim(), 0)
    return insert_dims(tensor, num_dims, axis)


def flatten_dims(tensor: torch.Tensor, num_dims: int, axis: int) -> torch.Tensor:
    """Flattens a specified set of dimensions of a tensor.

    This operation flattens ``num_dims`` dimensions of a ``tensor``
    starting at a given ``axis``.

    :param tensor: Input tensor
    :param num_dims: Number of dimensions to combine. Must be larger than
        two and less than or equal to the rank of ``tensor``.
    :param axis: Index of the dimension from which to start

    :output tensor: A tensor of the same type as ``tensor`` with ``num_dims`` - 1 lesser
        dimensions, but the same number of elements

    .. rubric:: Examples

    .. code-block:: python

        import torch
        from sionna.phy.utils import flatten_dims

        x = torch.ones([2, 3, 4, 5])
        print(x.shape)
        # torch.Size([2, 3, 4, 5])

        y = flatten_dims(x, num_dims=2, axis=1)
        print(y.shape)
        # torch.Size([2, 12, 5])
    """
    if not (num_dims >= 2):
        raise ValueError("`num_dims` must be >= 2")
    if not (num_dims <= tensor.dim()):
        raise ValueError("`num_dims` must be <= rank(`tensor`)")
    if not 0 <= axis <= tensor.dim() - 1:
        raise ValueError("`axis` must satisfy 0 <= `axis` <= rank(`tensor`) - 1")
    if not (num_dims + axis <= tensor.dim()):
        raise ValueError("`num_dims` + `axis` must be <= rank(`tensor`)")

    return torch.flatten(tensor, start_dim=axis, end_dim=axis + num_dims - 1)


def flatten_last_dims(tensor: torch.Tensor, num_dims: int = 2) -> torch.Tensor:
    """Flattens the last `n` dimensions of a tensor.

    This operation flattens the last ``num_dims`` dimensions of a ``tensor``.
    It is a simplified version of the function ``flatten_dims``.

    :param tensor: Input tensor
    :param num_dims: Number of dimensions to combine. Must be greater than
        or equal to two and less than or equal to the rank of ``tensor``.

    :output tensor: A tensor of the same type as ``tensor`` with ``num_dims`` - 1 lesser
        dimensions, but the same number of elements

    .. rubric:: Examples

    .. code-block:: python

        import torch
        from sionna.phy.utils import flatten_last_dims

        x = torch.ones([2, 3, 4])
        print(x.shape)
        # torch.Size([2, 3, 4])

        y = flatten_last_dims(x, num_dims=2)
        print(y.shape)
        # torch.Size([2, 12])
    """
    if not (num_dims >= 2):
        raise ValueError("`num_dims` must be >= 2")
    if not (num_dims <= tensor.dim()):
        raise ValueError("`num_dims` must be <= rank(`tensor`)")

    return torch.flatten(tensor, start_dim=-num_dims)


def insert_dims(tensor: torch.Tensor, num_dims: int, axis: int = -1) -> torch.Tensor:
    """Adds multiple length-one dimensions to a tensor.

    This operation is an extension to PyTorch's ``unsqueeze`` function.
    It inserts ``num_dims`` dimensions of length one starting from the
    dimension ``axis`` of a ``tensor``. The dimension index follows Python
    indexing rules, i.e., zero-based, where a negative index is counted
    backward from the end.

    :param tensor: Input tensor
    :param num_dims: Number of dimensions to add
    :param axis: Dimension index at which to expand the
        shape of ``tensor``. Given a ``tensor`` of `D` dimensions,
        ``axis`` must be within the range `[-(D+1), D]` (inclusive).

    :output tensor: A tensor with the same data as ``tensor``, with ``num_dims``
        additional dimensions inserted at the index specified by ``axis``

    .. rubric:: Examples

    .. code-block:: python

        import torch
        from sionna.phy.utils import insert_dims

        x = torch.ones([3, 4])
        print(x.shape)
        # torch.Size([3, 4])

        y = insert_dims(x, num_dims=2, axis=-1)
        print(y.shape)
        # torch.Size([3, 4, 1, 1])
    """
    if not (num_dims >= 0):
        raise ValueError("`num_dims` must be nonnegative")

    rank = tensor.dim()
    if not -(rank + 1) <= axis <= rank:
        raise ValueError("`axis` must be in the range [-(D+1), D]")

    axis = axis if axis >= 0 else rank + axis + 1
    shape = list(tensor.shape)
    new_shape = shape[:axis] + [1] * num_dims + shape[axis:]
    return tensor.reshape(new_shape)


def split_dim(
    tensor: torch.Tensor, shape: Union[List[int], torch.Size], axis: int
) -> torch.Tensor:
    """Reshapes a dimension of a tensor into multiple dimensions.

    This operation splits the dimension ``axis`` of a ``tensor`` into
    multiple dimensions according to ``shape``.

    :param tensor: Input tensor
    :param shape: Shape to which the dimension should be reshaped
    :param axis: Index of the axis to be reshaped

    :output tensor: A tensor of the same type as ``tensor`` with
        len(``shape``) - 1 additional dimensions, but the same number of
        elements

    .. rubric:: Examples

    .. code-block:: python

        import torch
        from sionna.phy.utils import split_dim

        x = torch.ones([2, 12])
        print(x.shape)
        # torch.Size([2, 12])

        y = split_dim(x, shape=[3, 4], axis=1)
        print(y.shape)
        # torch.Size([2, 3, 4])
    """
    if not 0 <= axis <= tensor.dim() - 1:
        raise ValueError("`axis` must satisfy 0 <= `axis` <= rank(`tensor`) - 1")

    s = tensor.shape
    new_shape = list(s[:axis]) + list(shape) + list(s[axis + 1 :])
    return tensor.reshape(new_shape)


def diag_part_axis(tensor: torch.Tensor, axis: int, offset: int = 0) -> torch.Tensor:
    r"""Extracts the batched diagonal part of a batched tensor over the specified axis.

    This is an extension of PyTorch's ``torch.diagonal`` function, which
    extracts the diagonal over the last two dimensions. This behavior can be
    reproduced by setting ``axis`` = -2.

    :param tensor: A tensor of rank greater than or equal to
        two (:math:`N\ge 2`) with shape [s(1), ..., s(N)]
    :param axis: Axis index starting from which the diagonal part is
        extracted
    :param offset: Offset of the diagonal from the main diagonal. Positive
        values select superdiagonals, negative values select subdiagonals.

    :output tensor: Tensor containing the diagonal part of input ``tensor`` over
        axis (``axis``, ``axis`` + 1), with shape
        [s(1), ..., min[s(``axis``), s(``axis`` + 1)],
        s(``axis`` + 2), ..., s(N)]

    .. rubric:: Examples

    .. code-block:: python

        import torch
        from sionna.phy.utils import diag_part_axis

        a = torch.arange(27).reshape(3, 3, 3)
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
    if tensor.dim() < 2:
        raise ValueError("The input tensor must have rank >= 2")

    rank = tensor.dim()
    if axis < 0:
        axis = rank + axis

    if not 0 <= axis <= rank - 2:
        raise ValueError("`axis` must satisfy 0 <= `axis` <= rank(`tensor`) - 2")

    # torch.diagonal extracts diagonal from dim1 and dim2, placing it at the end
    diag = torch.diagonal(tensor, offset=offset, dim1=axis, dim2=axis + 1)

    # Move the last dimension (diagonal) back to position axis
    return torch.movedim(diag, -1, axis)


def flatten_multi_index(
    indices: torch.Tensor, shape: Union[List[int], torch.Size]
) -> torch.Tensor:
    r"""Converts a tensor of index arrays into a tensor of flat indices.

    :param indices: Indices to flatten with shape [..., N] and dtype
        `torch.int32` or `torch.int64`
    :param shape: Shape of each index dimension [N].
        Note that it must hold that ``indices[..., n]`` < ``shape[n]``
        for all n and batch dimension.

    :output flat_indices: Flattened indices with shape [...]

    .. rubric:: Examples

    .. code-block:: python

        import torch
        from sionna.phy.utils import flatten_multi_index

        indices = torch.tensor([2, 3])
        shape = [5, 6]
        print(flatten_multi_index(indices, shape).numpy())
        # 15 = 2*6 + 3
    """
    indices = indices.to(torch.int64)
    shape_tensor = torch.tensor(shape, dtype=torch.int64, device=indices.device)

    check_tensor_all(
        indices >= 0,
        name="indices",
        message="`indices` must be non-negative",
    )
    check_tensor_all(
        indices < shape_tensor,
        name="indices",
        message="`indices` are out of bounds for `shape`",
    )

    # Compute strides: [prod(shape[1:]), prod(shape[2:]), ..., shape[-1], 1]
    ones = torch.ones(1, dtype=torch.int64, device=indices.device)
    strides = torch.cat([shape_tensor[1:], ones]).flip(0).cumprod(0).flip(0)

    return (indices * strides).sum(dim=-1)


def gather_from_batched_indices(
    params: torch.Tensor, indices: torch.Tensor
) -> torch.Tensor:
    r"""Gathers the values of a tensor ``params`` according to batch-specific ``indices``.

    :param params: Tensor containing the values to gather with shape
        [s(1), ..., s(N)]
    :param indices: Tensor containing, for each batch [...], the indices at
        which ``params`` is gathered with shape [..., N] and dtype
        `torch.int32` or `torch.int64`.
        Note that 0 :math:`\le` ``indices[...,n]`` :math:`<` s(n) must hold
        for all n=1,...,N.

    :output values: Tensor containing the gathered values with shape [...]

    .. rubric:: Examples

    .. code-block:: python

        import torch
        from sionna.phy.utils import gather_from_batched_indices

        params = torch.tensor([[10, 20, 30], [40, 50, 60], [70, 80, 90]])
        print(params.shape)
        # torch.Size([3, 3])

        indices = torch.tensor([[[0, 1], [1, 2], [2, 0], [0, 0]],
                               [[0, 0], [2, 2], [2, 1], [0, 1]]])
        print(indices.shape)
        # torch.Size([2, 4, 2])
        # Note that the batch shape is [2, 4]. Each batch contains a list
        # of 2 indices

        print(gather_from_batched_indices(params, indices).numpy())
        # [[20, 60, 70, 10],
        #  [10, 90, 80, 20]]
        # Note that the output shape coincides with the batch shape.
        # Element [i,j] coincides with params[indices[i,j,:]]
    """
    flat_indices = flatten_multi_index(indices, shape=list(params.shape))
    return params.reshape(-1)[flat_indices]


def tensor_values_are_in_set(
    tensor: torch.Tensor, admissible_set: Union[torch.Tensor, List[Any]]
) -> torch.Tensor:
    r"""Checks if the input ``tensor`` values are contained in the specified ``admissible_set``.

    :param tensor: Tensor to validate
    :param admissible_set: Set of valid values that the input ``tensor``
        must be composed of

    :output result: Returns `True` if and only if ``tensor`` values are contained
        in ``admissible_set``

    .. rubric:: Examples

    .. code-block:: python

        import torch
        from sionna.phy.utils import tensor_values_are_in_set

        tensor = torch.tensor([[1, 0], [0, 1]])

        print(tensor_values_are_in_set(tensor, [0, 1, 2]).item())
        # True

        print(tensor_values_are_in_set(tensor, [0, 2]).item())
        # False
    """
    if not isinstance(admissible_set, torch.Tensor):
        admissible_set = torch.tensor(admissible_set, device=tensor.device)
    return torch.all(torch.isin(tensor, admissible_set))


def random_tensor_from_values(
    values: Union[torch.Tensor, List[Any]],
    shape: Union[List[int], torch.Size],
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    r"""Generates a tensor of the specified ``shape``, with elements randomly sampled from the provided set of ``values``.

    :param values: The set of values to sample from. The output is placed on
        the device of ``values``, or on
        :attr:`~sionna.phy.config.Config.device` for list inputs.
    :param shape: The desired shape of the output tensor
    :param dtype: Desired dtype of the output

    :output tensor: A tensor with the specified shape, where each element is
        randomly selected from ``values``

    .. rubric:: Examples

    .. code-block:: python

        from sionna.phy.utils import random_tensor_from_values

        values = [0, 10, 20]
        shape = [2, 3]
        print(random_tensor_from_values(values, shape).cpu().numpy())
        # array([[ 0, 20,  0],
        #        [10,  0, 20]], dtype=int32)
    """
    if not isinstance(values, torch.Tensor):
        values = torch.tensor(values, device=config.device)

    device = values.device
    generator = (
        None if torch.compiler.is_compiling() else config.torch_rng(str(device))
    )
    indices = _randint(
        low=0,
        high=len(values),
        size=tuple(shape),
        dtype=torch.int64,
        device=device,
        generator=generator,
    )
    tensor = values[indices]
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    return tensor


def enumerate_indices(bounds: Union[List[int], torch.Tensor]) -> torch.Tensor:
    r"""Enumerates all indices between 0 (included) and ``bounds`` (excluded) in lexicographic order.

    :param bounds: Collection of index bounds. The output is placed on the
        device of ``bounds``, or on
        :attr:`~sionna.phy.config.Config.device` for list inputs.

    :output indices: Collection of all indices, in lexicographic order, with shape
        [prod(bounds), len(bounds)]

    .. rubric:: Examples

    .. code-block:: python

        from sionna.phy.utils import enumerate_indices

        print(enumerate_indices([2, 3]).cpu().numpy())
        # [[0 0]
        #  [0 1]
        #  [0 2]
        #  [1 0]
        #  [1 1]
        #  [1 2]]
    """
    if isinstance(bounds, torch.Tensor):
        bounds_list = bounds.tolist()
        device = bounds.device
    else:
        bounds_list = list(bounds)
        device = config.device

    ranges = [torch.arange(b, device=device) for b in bounds_list]
    return torch.cartesian_prod(*ranges)


def find_true_position(
    bool_tensor: torch.Tensor, side: str = "last", axis: int = -1
) -> torch.Tensor:
    """Finds the index of the first or last `True` value along the specified axis.

    When no `True` value is present, it returns -1.

    :param bool_tensor: Boolean tensor of any shape
    :param side: ``'first'`` | ``'last'``. If ``'first'``, the first `True`
        position is found, else the last.
    :param axis: Axis along which to find the last `True` value

    :output position: Tensor of indices, containing the index of the first or last
        `True` value. Its shape is ``bool_tensor.shape`` with specified
        ``axis`` removed.

    .. rubric:: Examples

    .. code-block:: python

        import torch
        from sionna.phy.utils import find_true_position

        x = torch.tensor([True, False, True, False, True])

        print(find_true_position(x, side='first').item())
        # 0

        print(find_true_position(x, side='last').item())
        # 4
    """
    if side not in ("first", "last"):
        raise ValueError("side must be 'first' or 'last'")

    # Check if any True exists along the axis
    any_true = torch.any(bool_tensor, dim=axis)
    size_along_axis = bool_tensor.shape[axis]

    if side == "first":
        # argmax returns the first True position (treats True as 1, False as 0)
        idx = torch.argmax(bool_tensor.to(torch.int32), dim=axis)
    else:
        # Flip, find first, then convert to last position
        flipped = torch.flip(bool_tensor, dims=[axis])
        idx_from_end = torch.argmax(flipped.to(torch.int32), dim=axis)
        idx = size_along_axis - 1 - idx_from_end

    # Return -1 where no True was found
    return torch.where(any_true, idx, torch.tensor(-1, device=bool_tensor.device))
