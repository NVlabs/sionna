#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
import pytest
import numpy as np
import torch

from sionna.phy import config
from sionna.phy.utils import (
    expand_to_rank,
    insert_dims,
    flatten_dims,
    flatten_last_dims,
    split_dim,
    diag_part_axis,
    flatten_multi_index,
    gather_from_batched_indices,
    tensor_values_are_in_set,
    random_tensor_from_values,
    enumerate_indices,
    find_true_position,
)


class TestExpandToRank:
    """Tests for the expand_to_rank function."""

    def test_expand_basic(self, device):
        """Test basic expansion to higher rank."""
        x = torch.ones([2, 3], device=device)
        y = expand_to_rank(x, 4, axis=-1)
        assert y.shape == torch.Size([2, 3, 1, 1])

    def test_expand_at_beginning(self, device):
        """Test expansion at the beginning of tensor."""
        x = torch.ones([2, 3], device=device)
        y = expand_to_rank(x, 5, axis=0)
        assert y.shape == torch.Size([1, 1, 1, 2, 3])

    def test_expand_at_middle(self, device):
        """Test expansion at a middle axis."""
        x = torch.ones([2, 3, 4], device=device)
        y = expand_to_rank(x, 5, axis=1)
        assert y.shape == torch.Size([2, 1, 1, 3, 4])

    def test_no_expansion_needed(self, device):
        """Test that no expansion occurs when target_rank <= current rank."""
        x = torch.ones([2, 3, 4], device=device)
        y = expand_to_rank(x, 2, axis=-1)
        assert y.shape == x.shape

    def test_negative_axis(self, device):
        """Test expansion with negative axis."""
        x = torch.ones([2, 3], device=device)
        y = expand_to_rank(x, 4, axis=-1)
        assert y.shape == torch.Size([2, 3, 1, 1])


class TestInsertDims:
    """Tests for the insert_dims function."""

    def test_insert_at_end(self, device):
        """Test inserting dimensions at the end."""
        x = torch.ones([2, 3], device=device)
        y = insert_dims(x, 2, axis=-1)
        assert y.shape == torch.Size([2, 3, 1, 1])

    def test_insert_at_beginning(self, device):
        """Test inserting dimensions at the beginning."""
        x = torch.ones([2, 3], device=device)
        y = insert_dims(x, 3, axis=0)
        assert y.shape == torch.Size([1, 1, 1, 2, 3])

    def test_insert_at_middle(self, device):
        """Test inserting dimensions at a middle position."""
        x = torch.ones([2, 3, 4], device=device)
        y = insert_dims(x, 2, axis=1)
        assert y.shape == torch.Size([2, 1, 1, 3, 4])

    def test_insert_zero_dims(self, device):
        """Test inserting zero dimensions (no change)."""
        x = torch.ones([2, 3], device=device)
        y = insert_dims(x, 0, axis=0)
        assert y.shape == x.shape

    def test_data_preserved(self, device):
        """Test that data is preserved after inserting dimensions."""
        x = torch.arange(6, device=device).reshape(2, 3)
        y = insert_dims(x, 2, axis=1)
        assert torch.allclose(y.squeeze(), x)


class TestFlattenDims:
    """Tests for the flatten_dims function."""

    def test_output_shapes(self, device):
        """Test that output shapes are correct for various inputs."""
        dims_list = [[100], [10, 100], [20, 30, 40], [20, 30, 40, 50]]
        batch_size = 128

        for dims in dims_list:
            for axis in range(len(dims) + 1):
                for num_dims in range(2, len(dims) + 2 - axis):
                    shape = [batch_size] + dims
                    x = torch.ones(shape, device=device)
                    r = flatten_dims(x, num_dims, axis)

                    expected_shape = (
                        shape[:axis]
                        + [int(np.prod(shape[axis : axis + num_dims]))]
                        + shape[axis + num_dims :]
                    )
                    assert list(r.shape) == expected_shape

    def test_data_preserved(self, device):
        """Test that flattening preserves tensor data."""
        x = torch.arange(24, device=device).reshape(2, 3, 4)
        y = flatten_dims(x, 2, axis=1)
        assert y.shape == torch.Size([2, 12])
        assert torch.allclose(y, x.reshape(2, 12))


class TestFlattenLastDims:
    """Tests for the flatten_last_dims function."""

    def test_output_shapes(self, device):
        """Test that output shapes are correct."""
        dims_list = [[100], [10, 100], [20, 30, 40]]
        batch_size = 128

        for dims in dims_list:
            for num_dims in range(2, len(dims) + 2):
                shape = [batch_size] + dims
                x = torch.ones(shape, device=device)
                r = flatten_last_dims(x, num_dims)
                assert r.shape[-1] == int(np.prod(shape[-num_dims:]))

    def test_full_flatten(self, device):
        """Test flattening entire tensor to vector."""
        dims_list = [[100], [10, 100], [20, 30, 40]]
        batch_size = 128

        for dims in dims_list:
            num_dims = len(dims) + 1
            shape = [batch_size] + dims
            x = torch.ones(shape, device=device)
            r = flatten_last_dims(x, num_dims)
            assert r.shape[-1] == int(np.prod(shape[-num_dims:]))

    def test_error_num_dims_too_small(self, device):
        """Test that num_dims < 2 raises a value error."""
        x = torch.ones([2, 3, 4], device=device)
        with pytest.raises(ValueError):
            flatten_last_dims(x, 1)


class TestSplitDim:
    """Tests for the split_dim function."""

    def test_split_basic(self, device):
        """Test basic dimension splitting."""
        x = torch.arange(24, device=device).reshape(2, 12)
        y = split_dim(x, [3, 4], axis=1)
        assert y.shape == torch.Size([2, 3, 4])

    def test_split_first_dim(self, device):
        """Test splitting the first dimension."""
        x = torch.arange(24, device=device).reshape(6, 4)
        y = split_dim(x, [2, 3], axis=0)
        assert y.shape == torch.Size([2, 3, 4])

    def test_data_preserved(self, device):
        """Test that data is preserved after splitting."""
        x = torch.arange(24, device=device).reshape(2, 12)
        y = split_dim(x, [3, 4], axis=1)
        assert torch.allclose(y.reshape(2, 12), x)


class TestDiagPartAxis:
    """Tests for the diag_part_axis function."""

    def test_matches_torch_diagonal_last_axes(self, device):
        """Test that axis=-2 produces correct diagonal extraction from last two dims."""
        a = torch.randint(0, 10000, (10, 8, 7, 6), device=device)

        a_diag1 = diag_part_axis(a, axis=-2)
        # For axis=-2, we extract diagonal from dims (2, 3) of shape (7, 6)
        # The diagonal has length min(7, 6) = 6
        # Result should have shape (10, 8, 6)
        assert a_diag1.shape == torch.Size([10, 8, 6])

        # Verify values: diagonal element [i, j, k] should equal a[i, j, k, k]
        for k in range(6):
            assert torch.all(a_diag1[:, :, k] == a[:, :, k, k])

    def test_with_offset(self, device):
        """Test diagonal extraction with offset."""
        a = torch.randint(0, 10000, (10, 8, 7, 6), device=device)

        a_diag1 = diag_part_axis(a, axis=-2, offset=1)
        # With offset=1, we get the superdiagonal
        # Diagonal length is min(7, 6-1) = 5
        assert a_diag1.shape == torch.Size([10, 8, 5])

        # Verify values: superdiagonal element [i, j, k] should equal a[i, j, k, k+1]
        for k in range(5):
            assert torch.all(a_diag1[:, :, k] == a[:, :, k, k + 1])

    def test_axis_0(self, device):
        """Test diagonal extraction starting at axis 0."""
        a = torch.arange(27, device=device).reshape(3, 3, 3)
        dp_0 = diag_part_axis(a, axis=0)

        expected = torch.tensor([[0, 1, 2], [12, 13, 14], [24, 25, 26]], device=device)
        assert torch.all(dp_0 == expected)

    def test_axis_1(self, device):
        """Test diagonal extraction starting at axis 1."""
        a = torch.arange(27, device=device).reshape(3, 3, 3)
        dp_1 = diag_part_axis(a, axis=1)

        expected = torch.tensor([[0, 4, 8], [9, 13, 17], [18, 22, 26]], device=device)
        assert torch.all(dp_1 == expected)

    def test_compiled(self, device):
        """Test that diag_part_axis works when compiled."""
        a = torch.randint(0, 10000, (10, 8, 7, 6), device=device)

        @torch.compile
        def diag_part_axis_compiled(tensor, axis, offset=0):
            return diag_part_axis(tensor, axis=axis, offset=offset)

        a_diag = diag_part_axis(a, axis=-2)
        a_diag_compiled = diag_part_axis_compiled(a, axis=-2)

        assert torch.all(a_diag == a_diag_compiled)


class TestFlattenMultiIndex:
    """Tests for the flatten_multi_index function."""

    def test_simple_example(self, device):
        """Test simple 2D index flattening."""
        indices = torch.tensor([2, 3], device=device)
        shape = [5, 6]

        flat_indices = flatten_multi_index(indices, shape)
        assert flat_indices.item() == 15  # 2*6 + 3 = 15

    def test_batched_indices(self, device):
        """Test flattening of batched multi-indices."""
        shape = [5, 6, 10, 12]
        batch_size = [20, 5]

        indices_np = np.random.randint(
            [0, 0, 0, 0], shape, size=batch_size + [len(shape)]
        )
        indices = torch.tensor(indices_np, device=device)
        flat_indices = flatten_multi_index(indices, shape)

        # Check output dimension
        assert list(flat_indices.shape) == batch_size

        # Compute expected strides
        strides = [np.prod(shape[i:]) for i in range(1, len(shape))] + [1]

        # Check computation correctness
        for b1 in range(batch_size[0]):
            for b2 in range(batch_size[1]):
                expected_flat = sum(indices_np[b1, b2] * np.array(strides))
                assert flat_indices[b1, b2].item() == expected_flat

    def test_compiled(self, device):
        """Test that flatten_multi_index works when compiled."""

        @torch.compile
        def flatten_multi_index_compiled(indices, shape):
            return flatten_multi_index(indices, shape)

        indices = torch.tensor([2, 3], device=device)
        shape = [5, 6]

        result = flatten_multi_index(indices, shape)
        result_compiled = flatten_multi_index_compiled(indices, shape)

        assert result.item() == result_compiled.item()

    def test_compiled_no_graph_break_on_validation(self, device):
        """Bounds checks must not graph-break Dynamo (SYS End-to-End regression)."""

        @torch.compile(fullgraph=True)
        def flatten_multi_index_compiled(indices, shape):
            return flatten_multi_index(indices, shape)

        indices = torch.tensor([[2, 3], [1, 4]], device=device)
        shape = [5, 6]
        result = flatten_multi_index_compiled(indices, shape)
        expected = flatten_multi_index(indices, shape)
        assert torch.equal(result, expected)


class TestGatherFromBatchedIndices:
    """Tests for the gather_from_batched_indices function."""

    def test_basic_gather(self, device):
        """Test basic batched index gathering."""
        shape = [3, 5, 2, 8]
        batch_size = [5, 6]

        params = torch.rand(shape, device=device)
        indices_np = np.random.randint(
            [0, 0, 0, 0], shape, size=batch_size + [len(shape)]
        )
        indices = torch.tensor(indices_np, dtype=torch.int32, device=device)
        params_gathered = gather_from_batched_indices(params, indices)

        # Check that for each batch (b1, b2), it extracts params[indices[b1,b2,:]]
        for b1 in range(batch_size[0]):
            for b2 in range(batch_size[1]):
                expected = params[tuple(indices_np[b1, b2, :])]
                assert params_gathered[b1, b2] == expected

    def test_compiled(self, device):
        """Test that gather_from_batched_indices works when compiled."""

        @torch.compile
        def gather_compiled(params, indices):
            return gather_from_batched_indices(params, indices)

        params = torch.tensor([[10, 20, 30], [40, 50, 60], [70, 80, 90]], device=device)
        indices = torch.tensor(
            [[[0, 1], [1, 2]], [[2, 0], [0, 0]]], dtype=torch.int32, device=device
        )

        result = gather_from_batched_indices(params, indices)
        result_compiled = gather_compiled(params, indices)

        assert torch.all(result == result_compiled)


class TestTensorValuesAreInSet:
    """Tests for the tensor_values_are_in_set function."""

    def test_values_in_set(self, device):
        """Test that function returns True when all values are in the set."""
        admissible_set = torch.tensor([10, 20, 30, 40], device=device)
        tensor = torch.tensor([[10, 30], [40, 20]], device=device)

        assert tensor_values_are_in_set(tensor, admissible_set).item() is True

    def test_values_not_in_set(self, device):
        """Test that function returns False when some values are not in the set."""
        admissible_set = torch.tensor([10, 20, 30, 40], device=device)
        tensor = torch.tensor([[10, 5], [40, 20]], device=device)  # 5 not in set

        assert tensor_values_are_in_set(tensor, admissible_set).item() is False

    def test_with_list_input(self, device):
        """Test that function works with list inputs."""
        tensor = torch.tensor([[1, 0], [0, 1]], device=device)

        assert tensor_values_are_in_set(tensor, [0, 1, 2]).item() is True
        assert tensor_values_are_in_set(tensor, [0, 2]).item() is False

    def test_compiled(self, device):
        """Test that tensor_values_are_in_set works when compiled."""

        @torch.compile
        def check_compiled(tensor, admissible_set):
            return tensor_values_are_in_set(tensor, admissible_set)

        tensor = torch.tensor([[1, 0], [0, 1]], device=device)
        admissible_set = torch.tensor([0, 1, 2], device=device)

        result = tensor_values_are_in_set(tensor, admissible_set)
        result_compiled = check_compiled(tensor, admissible_set)

        assert result.item() == result_compiled.item()


class TestRandomTensorFromValues:
    """Tests for the random_tensor_from_values function."""

    def test_output_shape(self, device):
        """Test that output has correct shape."""
        old_device = config.device
        config.device = device

        try:
            values = [0, 10, 20]
            shape = [2, 3, 4]

            tensor = random_tensor_from_values(values, shape)
            assert list(tensor.shape) == shape
        finally:
            config.device = old_device

    def test_values_from_set(self, device):
        """Test that all output values come from the input set."""
        old_device = config.device
        config.device = device

        try:
            values = [0, 10, 20, 30]
            shape = [10, 10]

            tensor = random_tensor_from_values(values, shape)
            assert tensor_values_are_in_set(tensor, values).item() is True
        finally:
            config.device = old_device

    def test_dtype_conversion(self, device):
        """Test that dtype parameter works correctly."""
        old_device = config.device
        config.device = device

        try:
            values = [0, 1, 2]
            shape = [3, 3]

            tensor = random_tensor_from_values(values, shape, dtype=torch.float32)
            assert tensor.dtype == torch.float32
        finally:
            config.device = old_device

    def test_tensor_input_owns_device(self, device):
        """Tensor values own output placement even when config differs."""
        values = torch.tensor([0, 10, 20], device="cpu")
        result = random_tensor_from_values(values, [32])
        assert result.device.type == "cpu"

        old_device = config.device
        config.device = "cpu"
        try:
            values = values.to(device)
            result = random_tensor_from_values(values, [32])
            assert result.device == torch.device(device)
        finally:
            config.device = old_device

    def test_tensor_input_reproducible(self, device):
        """Sampling uses the configured generator for the input device."""
        values = torch.tensor([0, 10, 20], device=device)
        config.seed = 123
        first = random_tensor_from_values(values, [64])
        config.seed = 123
        second = random_tensor_from_values(values, [64])
        assert torch.equal(first, second)

    def test_compiled(self, device):
        """Tensor-owned random sampling remains compile-aware."""

        @torch.compile(fullgraph=True)
        def sample(values):
            return random_tensor_from_values(values, [16])

        values = torch.tensor([0, 10, 20], device=device)
        result = sample(values)
        assert result.device == values.device
        assert tensor_values_are_in_set(result, values)


class TestEnumerateIndices:
    """Tests for the enumerate_indices function."""

    def test_simple_enumeration(self, device):
        """Test simple index enumeration."""
        old_device = config.device
        config.device = device

        try:
            bounds = [2, 3]
            result = enumerate_indices(bounds)

            expected = torch.tensor(
                [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]], device=device
            )
            assert torch.all(result == expected)
        finally:
            config.device = old_device

    def test_output_shape(self, device):
        """Test that output shape is correct."""
        old_device = config.device
        config.device = device

        try:
            bounds = [3, 4, 5]
            result = enumerate_indices(bounds)

            expected_rows = 3 * 4 * 5
            assert result.shape == torch.Size([expected_rows, 3])
        finally:
            config.device = old_device

    def test_tensor_input(self, device):
        """Test that tensor input works."""
        old_device = config.device
        config.device = device

        try:
            bounds = torch.tensor([2, 3])
            result = enumerate_indices(bounds)

            assert result.shape == torch.Size([6, 2])
        finally:
            config.device = old_device

    def test_tensor_input_owns_device(self, device):
        """Tensor bounds own output placement even when config differs."""
        bounds = torch.tensor([2, 3], device="cpu")
        result = enumerate_indices(bounds)
        assert result.device.type == "cpu"

        old_device = config.device
        config.device = "cpu"
        try:
            bounds = bounds.to(device)
            result = enumerate_indices(bounds)
            assert result.device == torch.device(device)
        finally:
            config.device = old_device


class TestFindTruePosition:
    """Tests for the find_true_position function."""

    @pytest.fixture
    def tensors(self, device):
        """Common test tensors."""
        return {
            "tensor_1d": torch.tensor([True, False, True, False, True], device=device),
            "tensor_2d": torch.tensor(
                [[True, False, True], [False, False, False], [True, True, False]],
                device=device,
            ),
            "tensor_3d": torch.tensor(
                [
                    [[True, False], [True, True], [False, True]],
                    [[True, True], [False, False], [True, False]],
                ],
                device=device,
            ),
            "all_false": torch.zeros((3, 4), dtype=torch.bool, device=device),
            "all_true": torch.ones((3, 4), dtype=torch.bool, device=device),
        }

    def test_basic_last(self, tensors):
        """Test basic last position finding."""
        result = find_true_position(tensors["tensor_1d"], side="last")
        assert result.item() == 4

    def test_basic_first(self, tensors):
        """Test basic first position finding."""
        result = find_true_position(tensors["tensor_1d"], side="first")
        assert result.item() == 0

    def test_2d_last_axis_minus1(self, device, tensors):
        """Test last position in 2D tensor along last axis."""
        result = find_true_position(tensors["tensor_2d"], side="last", axis=-1)
        expected = torch.tensor([2, -1, 1], device=device)
        assert torch.all(result == expected)

    def test_2d_last_axis_0(self, device, tensors):
        """Test last position in 2D tensor along first axis."""
        result = find_true_position(tensors["tensor_2d"], side="last", axis=0)
        expected = torch.tensor([2, 2, 0], device=device)
        assert torch.all(result == expected)

    def test_2d_first_axis_minus1(self, device, tensors):
        """Test first position in 2D tensor along last axis."""
        result = find_true_position(tensors["tensor_2d"], side="first", axis=-1)
        expected = torch.tensor([0, -1, 0], device=device)
        assert torch.all(result == expected)

    def test_2d_first_axis_0(self, device, tensors):
        """Test first position in 2D tensor along first axis."""
        result = find_true_position(tensors["tensor_2d"], side="first", axis=0)
        expected = torch.tensor([0, 2, 0], device=device)
        assert torch.all(result == expected)

    def test_3d_last(self, device, tensors):
        """Test last position in 3D tensor along last axis."""
        result = find_true_position(tensors["tensor_3d"], side="last")
        expected = torch.tensor([[0, 1, 1], [1, -1, 0]], device=device)
        assert torch.all(result == expected)

    def test_3d_first_axis_1(self, device, tensors):
        """Test first position in 3D tensor along middle axis."""
        result = find_true_position(tensors["tensor_3d"], side="first", axis=1)
        expected = torch.tensor([[0, 1], [0, 0]], device=device)
        assert torch.all(result == expected)

    def test_all_false(self, device, tensors):
        """Test behavior when no True values exist."""
        result_last = find_true_position(tensors["all_false"], side="last")
        expected = torch.tensor([-1, -1, -1], device=device)
        assert torch.all(result_last == expected)

        result_first = find_true_position(tensors["all_false"], side="first")
        assert torch.all(result_first == expected)

    def test_all_true(self, device, tensors):
        """Test behavior when all values are True."""
        result_last = find_true_position(tensors["all_true"], side="last")
        expected_last = torch.tensor([3, 3, 3], device=device)
        assert torch.all(result_last == expected_last)

        result_first = find_true_position(tensors["all_true"], side="first")
        expected_first = torch.tensor([0, 0, 0], device=device)
        assert torch.all(result_first == expected_first)

    def test_invalid_side(self, tensors):
        """Test that invalid side parameter raises error."""
        with pytest.raises(ValueError):
            find_true_position(tensors["tensor_1d"], side="middle")

    def test_positive_negative_axis_equivalence(self, device):
        """Test that positive and negative axis specifications are equivalent."""
        tensor = torch.tensor(
            [[[True, False], [False, True]], [[False, True], [True, False]]],
            device=device,
        )

        result_pos = find_true_position(tensor, side="last", axis=1)
        result_neg = find_true_position(tensor, side="last", axis=-2)
        assert torch.all(result_pos == result_neg)

    def test_compiled(self, tensors):
        """Test that find_true_position works when compiled."""

        @torch.compile
        def find_compiled(tensor, side, axis):
            return find_true_position(tensor, side=side, axis=axis)

        result = find_true_position(tensors["tensor_2d"], side="last", axis=-1)
        result_compiled = find_compiled(tensors["tensor_2d"], side="last", axis=-1)

        assert torch.all(result == result_compiled)


class TestDocstringExamples:
    """Tests verifying that docstring examples produce correct output."""

    def test_expand_to_rank_example(self):
        """Verify the expand_to_rank docstring example."""
        x = torch.ones([3, 4])
        assert x.shape == torch.Size([3, 4])

        y = expand_to_rank(x, 4, axis=-1)
        assert y.shape == torch.Size([3, 4, 1, 1])

    def test_flatten_dims_example(self):
        """Verify the flatten_dims docstring example."""
        x = torch.ones([2, 3, 4, 5])
        assert x.shape == torch.Size([2, 3, 4, 5])

        y = flatten_dims(x, num_dims=2, axis=1)
        assert y.shape == torch.Size([2, 12, 5])

    def test_flatten_last_dims_example(self):
        """Verify the flatten_last_dims docstring example."""
        x = torch.ones([2, 3, 4])
        assert x.shape == torch.Size([2, 3, 4])

        y = flatten_last_dims(x, num_dims=2)
        assert y.shape == torch.Size([2, 12])

    def test_insert_dims_example(self):
        """Verify the insert_dims docstring example."""
        x = torch.ones([3, 4])
        assert x.shape == torch.Size([3, 4])

        y = insert_dims(x, num_dims=2, axis=-1)
        assert y.shape == torch.Size([3, 4, 1, 1])

    def test_split_dim_example(self):
        """Verify the split_dim docstring example."""
        x = torch.ones([2, 12])
        assert x.shape == torch.Size([2, 12])

        y = split_dim(x, shape=[3, 4], axis=1)
        assert y.shape == torch.Size([2, 3, 4])

    def test_diag_part_axis_example(self):
        """Verify the diag_part_axis docstring example."""
        a = torch.arange(27).reshape(3, 3, 3)

        dp_0 = diag_part_axis(a, axis=0)
        expected_dp_0 = torch.tensor([[0, 1, 2], [12, 13, 14], [24, 25, 26]])
        assert torch.all(dp_0 == expected_dp_0)

        dp_1 = diag_part_axis(a, axis=1)
        expected_dp_1 = torch.tensor([[0, 4, 8], [9, 13, 17], [18, 22, 26]])
        assert torch.all(dp_1 == expected_dp_1)

    def test_flatten_multi_index_example(self):
        """Verify the flatten_multi_index docstring example."""
        indices = torch.tensor([2, 3])
        shape = [5, 6]

        result = flatten_multi_index(indices, shape)
        assert result.item() == 15  # 2*6 + 3 = 15

    def test_gather_from_batched_indices_example(self):
        """Verify the gather_from_batched_indices docstring example."""
        params = torch.tensor([[10, 20, 30], [40, 50, 60], [70, 80, 90]])
        assert params.shape == torch.Size([3, 3])

        indices = torch.tensor(
            [[[0, 1], [1, 2], [2, 0], [0, 0]], [[0, 0], [2, 2], [2, 1], [0, 1]]]
        )
        assert indices.shape == torch.Size([2, 4, 2])

        result = gather_from_batched_indices(params, indices)
        expected = torch.tensor([[20, 60, 70, 10], [10, 90, 80, 20]])
        assert torch.all(result == expected)

    def test_tensor_values_are_in_set_example(self):
        """Verify the tensor_values_are_in_set docstring example."""
        tensor = torch.tensor([[1, 0], [0, 1]])

        assert tensor_values_are_in_set(tensor, [0, 1, 2]).item() is True
        assert tensor_values_are_in_set(tensor, [0, 2]).item() is False

    def test_random_tensor_from_values_example(self):
        """Verify the random_tensor_from_values docstring example produces valid output."""
        values = [0, 10, 20]
        shape = [2, 3]

        result = random_tensor_from_values(values, shape)
        # Cannot verify exact values due to randomness, but verify shape and value membership
        assert list(result.shape) == shape
        assert tensor_values_are_in_set(result, values).item() is True

    def test_enumerate_indices_example(self):
        """Verify the enumerate_indices docstring example."""
        result = enumerate_indices([2, 3])
        expected = torch.tensor(
            [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]], device=result.device
        )
        assert torch.all(result == expected)

    def test_find_true_position_example(self):
        """Verify the find_true_position docstring example."""
        x = torch.tensor([True, False, True, False, True])

        assert find_true_position(x, side="first").item() == 0
        assert find_true_position(x, side="last").item() == 4
