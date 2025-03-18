#
# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0#

import unittest
import numpy as np
import tensorflow as tf

from sionna.phy import config, Block, dtypes
from sionna.phy.utils import complex_normal
from sionna.phy.utils import matrix_pinv, flatten_last_dims, \
    flatten_dims, expand_to_rank, diag_part_axis, flatten_multi_index, \
    gather_from_batched_indices, tensor_values_are_in_set, find_true_position

class TestFlattenLastDims(unittest.TestCase):
    def test_jit_mode(self):
        """Test that all but first dim are not None"""
        class F(Block):
            def __init__(self, dims, num_dims):
                super().__init__()
                self._dims = dims
                self._num_dims = num_dims

            @tf.function(jit_compile=True)
            def call(self, batch_size):
                shape = tf.concat([[batch_size], self._dims], -1)
                x = tf.ones(shape)
                x = flatten_last_dims(x, self._num_dims)
                tf.debugging.assert_equal(x.shape[-1], tf.reduce_prod(shape[-self._num_dims:]))
                return x

        dimsl = [[100],
                 [10, 100],
                 [20, 30, 40]
                ]
        batch_size = tf.constant(128, tf.int32)
        for dims in dimsl:
            for num_dims in range(2,len(dims)+1):
                f = F(dims, num_dims)
                r = f(batch_size)
                shape = [batch_size]+dims
                self.assertEqual(r.shape[-1], np.prod(shape[-num_dims:]))

        f = F([30], 2)
        with self.assertRaises(ValueError):
            f(batch_size)

    def test_full_flatten(self):
        """Test flattening to vector"""
        class F(Block):
            def __init__(self, dims, num_dims):
                super().__init__()
                self._dims = dims
                self._num_dims = num_dims

            @tf.function(jit_compile=True)
            def call(self, batch_size):
                shape = tf.concat([[batch_size], self._dims], -1)
                x = tf.ones(shape)
                x = flatten_last_dims(x, self._num_dims)
                return x

        dimsl = [[100],
                 [10, 100],
                 [20, 30, 40]
                ]
        batch_size = tf.constant(128, tf.int32)
        for dims in dimsl:
            num_dims = len(dims)+1
            f = F(dims, num_dims)
            r = f(batch_size)
            shape = [batch_size]+dims
            self.assertEqual(r.shape[-1], np.prod(shape[-num_dims:]))

class TestFlattenDims(unittest.TestCase):
    def test_jit_mode(self):
        """Test output shapes"""
        class F(Block):
            def __init__(self, dims, num_dims, axis):
                super().__init__()
                self._dims = dims
                self._num_dims = num_dims
                self._axis = axis

            @tf.function(jit_compile=True)
            def call(self, batch_size):
                shape = tf.concat([[batch_size], self._dims], -1)
                x = tf.ones(shape)
                x = flatten_dims(x, self._num_dims, self._axis)
                return x

        dimsl = [[100],
                 [10, 100],
                 [20, 30, 40],
                 [20, 30, 40, 50]
                ]
        batch_size = tf.constant(128, tf.int32)
        for dims in dimsl:
            for axis in range(0, len(dims)+1):
                for num_dims in range(2,len(dims)+2-axis):
                    f = F(dims, num_dims, axis)
                    r = f(batch_size)
                    shape = [batch_size]+dims
                    new_shape = shape[:axis] + [np.prod(shape[axis:axis+num_dims])] + shape[axis+num_dims:]
                    self.assertEqual(r.shape, new_shape)

class TestMatrixPinv(unittest.TestCase):
    """Unittest for the matrix_pinv function"""
    def test_single_dim(self):
        av = [0, 0.2, 0.9, 0.99]
        n = 64
        for a in av: 
            A = complex_normal([n, n//2], precision="double")
            A_pinv = matrix_pinv(A)
            I = tf.matmul(A_pinv, A)
            self.assertTrue(np.allclose(I, tf.eye(n//2, dtype=A.dtype)))

    def test_multi_dim(self):
        a = [2, 4, 3]
        n = 32
        A = complex_normal(a + [n, n//2], precision="double")
        A_pinv = matrix_pinv(A)
        I = tf.matmul(A_pinv, A)
        I_target = tf.eye(n//2, dtype=A.dtype)
        I_target = expand_to_rank(I_target, tf.rank(I), 0)
        self.assertTrue(np.allclose(I, I_target))

    def test_xla(self):
        a = [2, 4, 3]
        n = 32
        A64 = complex_normal(a + [n, n//2], precision="single")
        A128 = complex_normal(a + [n, n//2], precision="double")
        @tf.function(jit_compile=True)
        def func(A):
            return matrix_pinv(A)

        self.assertTrue(func(A64).dtype==dtypes['single']['tf']['cdtype'])
        self.assertTrue(func(A128).dtype==dtypes['double']['tf']['cdtype'])


class TestDiagPartAxis(unittest.TestCase):
    """Unittest for the diag_part_axis function. 
    Test that when axis=-2 the original behavior of tf.linalg.diag_part is
    retrieved"""
    def test_diag_part(self):
        #  
        @tf.function
        def diag_part_axis_graph(a, axis, **kwargs):
            return diag_part_axis(a, axis=axis, **kwargs)

        @tf.function(jit_compile=True)
        def diag_part_axis_xla(a, axis, **kwargs):
            return diag_part_axis(a, axis=axis, **kwargs)
        
        a = config.tf_rng.uniform(shape=[10,8,7,6],
                                  dtype=tf.int32, maxval=10000)

        a_diag1 = diag_part_axis(a, axis=-2)
        a_diag1_graph = diag_part_axis_graph(a, axis=-2)
        a_diag1_xla = diag_part_axis_xla(a, axis=-2)
        a_diag2 = tf.linalg.diag_part(a)

        self.assertEqual(tf.reduce_max(abs(a_diag1-a_diag2)).numpy(), 0)
        self.assertEqual(tf.reduce_max(abs(a_diag1_graph-a_diag2)).numpy(), 0)
        self.assertEqual(tf.reduce_max(abs(a_diag1_xla-a_diag2)).numpy(), 0)

        a_diag3 = diag_part_axis(a, axis=-2, k=1)
        a_diag3_graph = diag_part_axis_graph(a, axis=-2, k=1)
        a_diag3_xla = diag_part_axis_xla(a, axis=-2, k=1)
        a_diag4 = tf.linalg.diag_part(a, k=1)
        
        self.assertEqual(tf.reduce_max(abs(a_diag3-a_diag4)).numpy(), 0)
        self.assertEqual(tf.reduce_max(abs(a_diag3_graph-a_diag4)).numpy(), 0)
        self.assertEqual(tf.reduce_max(abs(a_diag3_xla-a_diag4)).numpy(), 0)

        a_diag5 = diag_part_axis(a, axis=-2, k=(1,3))
        a_diag5_graph = diag_part_axis_graph(a, axis=-2, k=(1,3))
        a_diag5_xla = diag_part_axis_xla(a, axis=-2, k=(1,3))
        a_diag6 = tf.linalg.diag_part(a, k=(1,3))

        self.assertEqual(tf.reduce_max(abs(a_diag5-a_diag6)).numpy(), 0)
        self.assertEqual(tf.reduce_max(abs(a_diag5_graph-a_diag6)).numpy(), 0)
        self.assertEqual(tf.reduce_max(abs(a_diag5_xla-a_diag6)).numpy(), 0)

class TestFlattenMultiIndex(unittest.TestCase):
    """Unittest for the flatten_multi_index function"""
    def test_flatten_multi_index(self):
        
        @tf.function
        def flatten_multi_index_graph(indices, shape):
            return flatten_multi_index(indices, shape)

        @tf.function(jit_compile=True)
        def flatten_multi_index_xla(indices, shape):
            return flatten_multi_index(indices, shape)
        
        # simple example
        indices = tf.constant([2, 3])
        shape = [5, 6]

        flat_indices = flatten_multi_index(indices, shape).numpy()
        self.assertTrue(flat_indices==15)

        flat_indices_graph = flatten_multi_index_graph(indices, shape).numpy()
        self.assertTrue(flat_indices_graph==15)

        flat_indices_xla = flatten_multi_index_xla(indices, shape).numpy()
        self.assertTrue(flat_indices_xla==15)

        # more articulate example
        shape = [5, 6, 10, 12]
        
        batch_size = [20, 5]
        indices = np.random.randint(shape, size=batch_size +[len(shape)])
        flat_indices = flatten_multi_index(indices, shape).numpy()
        flat_indices_graph = flatten_multi_index_graph(indices, shape).numpy()
        flat_indices_xla = flatten_multi_index_xla(indices, shape).numpy()

        # check output dimension consistency
        self.assertEqual(list(flat_indices.shape), batch_size)
        self.assertEqual(list(flat_indices_graph.shape), batch_size)
        self.assertEqual(list(flat_indices_xla.shape), batch_size)

        strides = [np.prod(shape[i:]) for i in range(1, len(shape))] + [1]

        # check that the computation is correct
        for b1 in range(batch_size[0]):
            for b2 in range(batch_size[1]):
                flat_idx = sum(indices[b1, b2] * strides)
                self.assertEqual(flat_indices[b1, b2], flat_idx)
                self.assertEqual(flat_indices_graph[b1, b2], flat_idx)
                self.assertEqual(flat_indices_xla[b1, b2], flat_idx)


class TestGatherBatchedIndices(unittest.TestCase):
    """Unittest for the gather_from_batched_indices function"""
    def test_gather_from_batched_indices(self):
        
        @tf.function(jit_compile=True)
        def gather_from_batched_indices_xla(params, indices):
            return gather_from_batched_indices(params, indices)
        
        @tf.function(jit_compile=False)
        def gather_from_batched_indices_graph(params, indices):
            return gather_from_batched_indices(params, indices)
        
        gather_from_batched_indices_dict = {'eager': gather_from_batched_indices,
                                            'graph': gather_from_batched_indices_graph,
                                            'xla': gather_from_batched_indices_xla}
        for mode in gather_from_batched_indices_dict:
            print(mode)
            fun = gather_from_batched_indices_dict[mode]

            shape = [3, 5, 2, 8]
            batch_size = [5, 6]
            params = config.tf_rng.uniform(shape=shape)
            indices_np = config.np_rng.integers(shape, size = batch_size + [len(shape)])
            indices = tf.convert_to_tensor(indices_np, dtype=tf.int32)
            params_gathered = fun(params, indices)

            # check that, bor each batch (b1,b2), it extracts
            # params[indices[b1,b1,:]] 
            for b1 in range(batch_size[0]):
                for b2 in range(batch_size[1]):
                    self.assertEqual(params[tuple(indices_np[b1,b2,:])],
                                    params_gathered[b1,b2])


class TestAssertValuesInSet(unittest.TestCase):
    """
    Unittest for the assert_values_in_set function
    """
    def test_assert_values_in_set(self):
        @tf.function(jit_compile=True)
        def tensor_values_are_in_set_xla(tensor, admissible_set):
            return tensor_values_are_in_set(tensor, admissible_set)
        
        @tf.function(jit_compile=False)
        def tensor_values_are_in_set_graph(tensor, admissible_set):
            return tensor_values_are_in_set(tensor, admissible_set)
        
        tensor_values_are_in_set_dict = {'eager': tensor_values_are_in_set,
                                     'graph': tensor_values_are_in_set_graph,
                                     'xla': tensor_values_are_in_set_xla}
        
        shape = [4, 5, 3]
        num_elements = np.prod(shape)  # Number of elements in the output tensor
        for mode, fun in tensor_values_are_in_set_dict.items():
            print(mode)

            # Example set of values
            admissible_set = tf.constant([10, 20, 30, 40])

            # Positive example: values are contained in admissible_set
            tensor_vals = tf.constant([10, 30, 40])
            indices = config.tf_rng.uniform(shape=(num_elements,),
                                        minval=0,
                                        maxval=tf.shape(tensor_vals)[0],
                                        dtype=tf.int32)
            tensor = tf.reshape(tf.gather(tensor_vals, indices), shape)

            self.assertTrue(fun(tensor, admissible_set))

            # Negative example: values are NOT contained in admissible_set
            tensor_vals = tf.constant([0, 10, 2])
            indices = config.tf_rng.uniform(shape=(num_elements,),
                                        minval=0,
                                        maxval=tf.shape(tensor_vals)[0],
                                        dtype=tf.int32)
            tensor = tf.reshape(tf.gather(tensor_vals, indices), shape)

            self.assertFalse(fun(tensor, admissible_set))


def find_true_position_xla(bool_tensor,
                           side='last',
                           axis=-1):
    return find_true_position(bool_tensor,
                              side=side,
                              axis=axis)
class TestFindTruePosition(unittest.TestCase):
    def setUp(self):
        # Common test tensors
        self.tensor_1d = tf.constant([True, False, True, False, True])
        self.tensor_2d = tf.constant([
            [True, False, True],
            [False, False, False],
            [True, True, False]
        ])
        self.tensor_3d = tf.constant([
            [[True, False],
             [True, True],
             [False, True]],
            
            [[True, True],
             [False, False],
             [True, False]]
        ])
        self.all_false = tf.zeros((3, 4), dtype=bool)
        self.all_true = tf.ones((3, 4), dtype=bool)

    def test_basic_last(self):
        """Test basic last position finding"""
        result = find_true_position_xla(self.tensor_1d, side='last')
        self.assertEqual(result.numpy(), 4)

    def test_basic_first(self):
        """Test basic first position finding"""
        result = find_true_position_xla(self.tensor_1d, side='first')
        self.assertEqual(result.numpy(), 0)

    def test_2d_last(self):
        """Test last position in 2D tensor along different axes"""
        # Along last axis
        result = find_true_position_xla(self.tensor_2d, side='last', axis=-1)
        expected = tf.constant([2, -1, 1])
        self.assertTrue(tf.reduce_all(result==expected))
        
        # Along first axis
        result = find_true_position_xla(self.tensor_2d, side='last', axis=0)
        expected = tf.constant([2, 2, 0])
        self.assertTrue(tf.reduce_all(result==expected))

    def test_2d_first(self):
        """Test first position in 2D tensor along different axes"""
        # Along last axis
        result = find_true_position_xla(self.tensor_2d, side='first', axis=-1)
        expected = tf.constant([0, -1, 0])
        self.assertTrue(tf.reduce_all(result==expected))
        
        # Along first axis
        result = find_true_position_xla(self.tensor_2d, side='first', axis=0)
        expected = tf.constant([0, 2, 0])
        self.assertTrue(tf.reduce_all(result==expected))

    def test_3d_tensor(self):
        """Test with 3D tensor"""
        # Last position along last axis
        result = find_true_position_xla(self.tensor_3d, side='last')
        expected = tf.constant([[0, 1, 1], [1, -1, 0]])
        self.assertTrue(tf.reduce_all(result==expected))

        # First position along middle axis
        result = find_true_position_xla(self.tensor_3d, side='first', axis=1)
        expected = tf.constant([[0, 1], [0, 0]])
        self.assertTrue(tf.reduce_all(result==expected))

    def test_all_false(self):
        """Test behavior when no True values exist"""
        result_last = find_true_position_xla(self.all_false, side='last')
        expected = tf.constant([-1, -1, -1])
        self.assertTrue(tf.reduce_all(result_last==expected))

        result_first = find_true_position_xla(self.all_false, side='first')
        expected = tf.constant([-1, -1, -1])
        self.assertTrue(tf.reduce_all(result_first==expected))

    def test_all_true(self):
        """Test behavior when all values are True"""
        result_last = find_true_position_xla(self.all_true, side='last')
        expected = tf.constant([3, 3, 3])
        self.assertTrue(tf.reduce_all(result_last==expected))

        result_first = find_true_position_xla(self.all_true, side='first')
        expected = tf.constant([0, 0, 0])
        self.assertTrue(tf.reduce_all(result_first==expected))

    def test_invalid_side(self):
        """Test that invalid side parameter raises error"""
        with self.assertRaises(tf.errors.InvalidArgumentError):
            find_true_position_xla(self.tensor_1d, side='middle')

    def test_different_axes(self):
        """Test with different axis specifications"""
        tensor = tf.constant([[[True, False], [False, True]], 
                            [[False, True], [True, False]]])
        
        # Test positive and negative axis specifications
        result_pos = find_true_position_xla(tensor, side='last', axis=1)
        result_neg = find_true_position_xla(tensor, side='last', axis=-2)
        self.assertTrue(tf.reduce_all(result_pos==result_neg))