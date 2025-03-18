#
# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0#

import unittest
import numpy as np
import tensorflow as tf
from sionna.phy.utils.metrics import compute_ber, compute_bler, count_block_errors, count_errors
from sionna.phy.fec.interleaving import RandomInterleaver
from sionna.phy.utils import sim_ber, complex_normal, DeepUpdateDict, dict_keys_to_int, to_list
from sionna.phy.mapping import SymbolDemapper, Demapper, Constellation, SymbolSource, BinarySource, QAMSource, PAMSource
from sionna.phy.channel import AWGN
from sionna.phy import dtypes, config

class ber_tester():
    """Utility class to emulate monte-carlo simulation with predefined
    num_errors."""
    def __init__(self, nb_errors, shape):
        self.shape = shape # shape
        self.errors = nb_errors # [1000, 400, 200, 100, 1, 0]
        self.idx = 0

    def reset(self):
        self.idx = 0

    def get_samples(self, batch_size, ebno_db):
        """Helper function to test sim_ber.

         Both inputs will be ignored but are required as placeholder to test
         sim_ber."""

        nb_errors = self.errors[self.idx]
        # increase internal counter for next call
        self.idx += 1
        x = np.zeros(np.prod(self.shape))

        # distribute nb_errors
        for i in range(nb_errors):
            x[i] = 1

        # permute
        interleaver = RandomInterleaver(axis=1, keep_batch_constant=False)
        x = tf.expand_dims(x,0)
        x = interleaver(x)
        x = tf.reshape(x, self.shape)

        return tf.zeros(self.shape), x


class TestUtils(unittest.TestCase):

    def test_ber_sim(self):
        """Test that ber_sim returns correct number of errors"""

        shape = [500, 200]
        errors = [1000, 400, 200, 100, 1, 0, 10]
        ber_true = errors / np.prod(shape)

        # init tester
        tester = ber_tester(errors, shape)

        # --- no stopping cond ---
        ber, _ = sim_ber(tester.get_samples, np.zeros_like(errors), max_mc_iter=1, early_stop=False, batch_size=1)
        # check if ber is correct
        self.assertTrue(np.allclose(ber, ber_true))

        # --- test early stopping ---
        tester.reset() # reset tester (set internal snr index to 0)

        ber, _ = sim_ber(tester.get_samples, np.zeros_like(errors), max_mc_iter=1, early_stop=True, batch_size=1)

        ber_true = errors / np.prod(shape)
        # test that all bers except last position are equal
        # last position differs as early stop triggered at 2. last point
        self.assertTrue(np.allclose(ber[:-1], ber_true[:-1]))

        # check that last ber is 0
        self.assertTrue(np.allclose(ber[-1], np.zeros_like(ber[-1])))

    def test_ber_sim_multi(self):
        "test multi GPU support of sim_ber function"

        channel = AWGN()
        ebno_dbs = np.arange(0, 3, 1)

        def _run_sim(batch_size, ebno_db):
            no = 10**(-ebno_db/10)
            b = tf.ones((batch_size, 1),tf.complex64)
            return tf.math.real(b), tf.math.real(channel(b, no))

        ber_ref, _ = sim_ber(_run_sim,
                             ebno_dbs,
                             max_mc_iter=128,
                             early_stop=False,
                             soft_estimates=True,
                             batch_size=100000)

        for mode in [None, "graph", "xla"]:
            ber_dist, _ = sim_ber(_run_sim,
                                    ebno_dbs,
                                    max_mc_iter=128,
                                    early_stop=False,
                                    graph_mode=mode,
                                    soft_estimates=True,
                                    distribute="all",
                                    batch_size=100000)

            # allow relativ tolerance due to Monte Carlo
            self.assertTrue(np.allclose(ber_ref.numpy(),
                                        ber_dist.numpy(),rtol=0.03))

        # Test for random seeds per simulation (i.e., non-equal results per gpu)
        # The idea is to run many simulations with a only 2 bit and very low SNR
        # we expect all cases {0,1,2} erroneous bits, i.e., the BER takes
        # the values {0.0, 0.5, 1.0}
        # if two GPUs have the same seed, we will never see an odd number of
        # errors.
        ebno_dbs = np.arange(-30, 0, 1)
        for mode in [None, "graph", "xla"]:
            ber, _ = sim_ber(_run_sim,
                            ebno_dbs,
                            max_mc_iter=2,
                            early_stop=False,
                            soft_estimates=True,
                            graph_mode="xla",
                            distribute="all",
                            batch_size=1)
            self.assertTrue(np.any(ber==0.5))

    def test_compute_ber(self):
        """Test that compute_ber returns the correct value."""

        shape = [500, 20, 40]
        errors = [1000, 400, 200, 100, 1, 0, 10]
        bers_true = errors / np.prod(shape)

        tester = ber_tester(errors, shape)

        for _,ber in enumerate(bers_true):
            b, b_hat = tester.get_samples(0, 0)
            ber_hat = compute_ber(b, b_hat)
            self.assertTrue(np.allclose(ber, ber_hat))

    def test_count_errors(self):
        """Test that count_errors returns the correct value."""

        shape = [500, 20, 40]
        errors = [1000, 400, 200, 100, 1, 0, 10]

        tester = ber_tester(errors, shape)

        for _,e in enumerate(errors):
            b, b_hat = tester.get_samples(0, 0)
            errors_hat = count_errors(b, b_hat)
            self.assertTrue(np.allclose(e, errors_hat))

    def test_count_block_errors(self):
        """Test that count_block_errors returns the correct value."""

        shape = [50, 400]
        errors = [1000, 400, 200, 100, 1, 0, 10]

        tester = ber_tester(errors, shape)

        for _,e in enumerate(errors):
            b, b_hat = tester.get_samples(0, 0)
            bler_hat = count_block_errors(b, b_hat)

            # ground truth
            bler = 0
            for idx in range(shape[0]):
                if not np.allclose(b[idx,:], b_hat[idx,:]):
                    bler +=1

            self.assertTrue(np.allclose(bler, bler_hat))

    def test_compute_bler(self):
        """Test that compute_bler returns the correct value."""

        shape = [50, 400]
        errors = [1000, 400, 200, 100, 1, 0, 10]

        tester = ber_tester(errors, shape)

        for _,e in enumerate(errors):
            b, b_hat = tester.get_samples(0, 0)
            bler_hat = compute_bler(b, b_hat)

            # ground truth
            bler = 0
            for idx in range(shape[0]):
                if not np.allclose(b[idx,:], b_hat[idx,:]):
                    bler +=1
            bler /= shape[0]
            self.assertTrue(np.allclose(bler, bler_hat))


class TestComplexNormal(unittest.TestCase):
    """Test cases for the complex_normal function"""
    def test_variance(self):
        shape = [100000000]
        v = [0, 0.5, 1.0, 2.3, 25]
        for var in v:
            x = complex_normal(shape, var)
            self.assertTrue(np.allclose(var, np.var(x), rtol=1e-3))
            self.assertTrue(np.allclose(np.var(np.real(x)), np.var(np.imag(x)), rtol=1e-3))

        # Default variance
        var_hat = np.var(complex_normal(shape))
        self.assertTrue(np.allclose(1.0, var_hat, rtol=1e-3))

    def test_precision(self):
        for precision in ["single", "double"]:
            x = complex_normal([100], precision=precision)
            self.assertEqual(dtypes[precision]['tf']['cdtype'], x.dtype)

    def test_dims(self):
        dims = [
                [100],
                [7, 8, 5],
                [4, 5, 67, 8]
                ]
        for d in dims:
            x = complex_normal(d)
            self.assertEqual(d, x.shape)

    def test_xla(self):
        @tf.function(jit_compile=True)
        def func(batch_size, var):
            return complex_normal([batch_size, 1000], var, precision="double")
        var = 0.3
        var_hat = np.var(func(100000, var))
        self.assertTrue(np.allclose(var, var_hat, rtol=1e-3))

        var = 1
        var_hat = np.var(func(100000, var))
        self.assertTrue(np.allclose(var, var_hat, rtol=1e-3))

        var = tf.cast(0.3, tf.int32)
        var_hat = np.var(func(100000, var))
        self.assertTrue(np.allclose(var, var_hat, rtol=1e-3))

class TestSources(unittest.TestCase):

    def test_binary_source(self):
        shapes = [[10], [10, 20], [10, 20, 30], [10,20,30,40]]
        precisions = ["single", "double"]
        seeds = [None, 1, 2]
        for precision in precisions:
            for seed in seeds:
                binary_source = BinarySource(precision=precision, seed=seed)
                binary_source2 = BinarySource(precision=precision, seed=seed)
                binary_source3 = BinarySource(precision=precision, seed=3)
                for shape in shapes:
                    b = binary_source(shape)
                    self.assertTrue(np.array_equal(b.shape, shape))
                    self.assertTrue(b.dtype==dtypes[precision]['tf']['rdtype'])
                    if seed is not None:
                        b2 = binary_source2(shape)
                        b3 = binary_source3(shape)
                        self.assertTrue(np.array_equal(b, b2))
                        self.assertFalse(np.array_equal(b, b3))

    def test_symbol_source_pam(self):
        shapes = [[10], [10, 20], [10, 20, 30], [10,20,30,40]]
        precisions = ["single", "double"]
        seeds = [None, 1, 2]
        bits_per_symbol = [1, 2, 3, 4]
        for precision in precisions:
            for seed in seeds:
                for num_bits_per_symbol in bits_per_symbol:
                    symbol_source = SymbolSource("pam", num_bits_per_symbol, precision=precision, seed=seed)
                    symbol_source2 = PAMSource(num_bits_per_symbol, precision=precision, seed=seed)
                    symbol_source3 = SymbolSource("pam", num_bits_per_symbol,precision=precision, seed=3)
                    for shape in shapes:
                        x = symbol_source(shape)
                        self.assertTrue(np.array_equal(x.shape, shape))
                        self.assertTrue(x.dtype==dtypes[precision]['tf']['cdtype'])
                        if seed is not None:
                            x2 = symbol_source2(shape)
                            x3 = symbol_source3(shape)
                            # Same seed must lead to same result
                            self.assertTrue(np.array_equal(x, x2))
                            # Different seed must lead to different results
                            self.assertFalse(np.array_equal(x, x3))

    def test_symbol_source_qam(self):
        shapes = [[10], [10, 20], [10, 20, 30], [10,20,30,40]]
        precisions = ["single", "double"]
        seeds = [None, 1, 2]
        bits_per_symbol = [2, 4, 6]
        for precision in precisions:
            for seed in seeds:
                for num_bits_per_symbol in bits_per_symbol:
                    symbol_source = SymbolSource("qam", num_bits_per_symbol, precision=precision, seed=seed)
                    symbol_source2 = QAMSource(num_bits_per_symbol, precision=precision, seed=seed)
                    symbol_source3 = SymbolSource("qam", num_bits_per_symbol, precision=precision, seed=3)
                    for shape in shapes:
                        x = symbol_source(shape)
                        self.assertTrue(np.array_equal(x.shape, shape))
                        self.assertTrue(x.dtype==dtypes[precision]['tf']['cdtype'])
                        if seed is not None:
                            x2 = symbol_source2(shape)
                            x3 = symbol_source3(shape)
                            # Same seed must lead to same result
                            self.assertTrue(np.array_equal(x, x2))
                            # Different seed must lead to different results
                            self.assertFalse(np.array_equal(x, x3))

    def test_symbol_source_custom(self):
        shapes = [[10], [10, 20], [10, 20, 30], [10,20,30,40]]
        precisions = ["single", "double"]
        seeds = [None, 1, 2]
        bits_per_symbol = [2, 4, 6]
        for precision in precisions:
            for seed in seeds:
                for num_bits_per_symbol in bits_per_symbol:
                    points = config.tf_rng.normal([2**num_bits_per_symbol])
                    constellation = Constellation("custom", num_bits_per_symbol, points=points, precision=precision)
                    symbol_source = SymbolSource("custom", num_bits_per_symbol, constellation=constellation, precision=precision, seed=seed)
                    symbol_source2 = SymbolSource("custom", num_bits_per_symbol, constellation=constellation, precision=precision, seed=seed)
                    symbol_source3 = SymbolSource("custom", num_bits_per_symbol, constellation=constellation, precision=precision, seed=3)
                    for shape in shapes:
                        x = symbol_source(shape)
                        self.assertTrue(np.array_equal(x.shape, shape))
                        self.assertTrue(x.dtype==dtypes[precision]['tf']['cdtype'])
                        if seed is not None:
                            x2 = symbol_source2(shape)
                            x3 = symbol_source3(shape)
                            # Same seed must lead to same result
                            self.assertTrue(np.array_equal(x, x2))
                            # Different seed must lead to different results
                            self.assertFalse(np.array_equal(x, x3))

    def test_symbol_source_output_flags(self):
        shapes = [[10], [10, 20], [10, 20, 30], [10,20,30,40]]
        seeds = [None, 1, 2]
        bits_per_symbol = [2, 4, 6]
        return_indices = [False, True]
        return_bits = [False, True]
        for num_bits_per_symbol in bits_per_symbol:
            for ri in return_indices:
                for rb in return_bits:
                    symbol_source = SymbolSource("qam", num_bits_per_symbol, return_indices=ri, return_bits=rb)
                    symbol_demapper = SymbolDemapper("qam", num_bits_per_symbol, hard_out=True)
                    demapper = Demapper("app","qam", num_bits_per_symbol, hard_out=True)
                    for shape in shapes:
                        if not ri and not rb:
                            x = symbol_source(shape)
                            self.assertTrue(np.array_equal(x.shape, shape))
                        if ri and not rb:
                            x, ind = symbol_source(shape)
                            self.assertTrue(np.array_equal(x.shape, shape))
                            self.assertTrue(np.array_equal(ind.shape, shape))
                            ind2 = symbol_demapper(x, tf.cast(0.01, tf.float32))
                            self.assertTrue(np.array_equal(ind, ind2))
                        if not ri and rb:
                            x, b = symbol_source(shape)
                            self.assertTrue(np.array_equal(x.shape, shape))
                            self.assertTrue(np.array_equal(b.shape, shape+[num_bits_per_symbol]))
                            b2 = demapper(x, tf.cast(0.01, tf.float32))
                            b2 = tf.reshape(b2, b.shape)
                            self.assertTrue(np.array_equal(b, b2))

                        if ri and rb:
                            x, ind, b = symbol_source(shape)
                            self.assertTrue(np.array_equal(x.shape, shape))
                            self.assertTrue(np.array_equal(ind.shape, shape))
                            self.assertTrue(np.array_equal(b.shape, shape+[num_bits_per_symbol]))
                            ind2 = symbol_demapper(x, tf.cast(0.01, tf.float32))
                            self.assertTrue(np.array_equal(ind, ind2))
                            b2 = demapper(x, tf.cast(0.01, tf.float32))
                            b2 = tf.reshape(b2, b.shape)
                            self.assertTrue(np.array_equal(b, b2))


class TestMisc(unittest.TestCase):
    """Test cases for miscellaneous utility functions"""
    
    def test_to_list(self):
        """Test sionna.phy.utils.to_list"""

        self.assertEqual(to_list(None), None)
        self.assertEqual(to_list(1), [1])
        self.assertEqual(to_list([1, 2]), [1, 2])
        self.assertEqual(to_list('abc'), ['abc'])
        self.assertEqual(to_list(np.array([1, 2])), [1, 2])

    def test_dict_keys_to_int(self):
        """Test sionna.phy.utils.dict_keys_to_int"""

        dict1 = {'1': {'2.3': [45, '3']}, 4: 6, 'ciao': [5, '87']}
        dict_out = dict_keys_to_int(dict1)
        self.assertDictEqual(dict_out, 
                             {1: {'2.3': [45, '3']}, 4: 6, 'ciao': [5, '87']})
        
    def test_deep_merge_dict(self):
        """Test sionna.phy.utils.deep_merge_dict"""
        
        # Merge without conflicts
        dict1 = DeepUpdateDict(
            {'a': 1,
            'b':
            {'b1': 10,
            'b2': 20}})
        dict_delta1 = {'c': -2,
                    'b':
                    {'b3': 30}}
        dict1.deep_update(dict_delta1)
        self.assertDictEqual(
            dict1,
            {'a': 1, 'b': {'b1': 10, 'b2': 20, 'b3': 30}, 'c': -2})

        # Handle key conflicts
        dict2 = DeepUpdateDict(
            {'a': 1,
            'b':
            {'b1': 10,
            'b2': 20}})
        dict_delta2 = {'a': -2,
                    'b':
                    {'b1': {'f': 3, 'g': 4}}}
        dict2.deep_update(dict_delta2)
        self.assertDictEqual(
            dict2,
            {'a': -2, 'b': {'b1': {'f': 3, 'g': 4}, 'b2': 20}}
        )

        # Merge with conflicts (1)
        dict3 = DeepUpdateDict({'a': {3: 50},
                'b':
                {'b1': 10,
                'b2': 20}})
        dict4 = {'a': -2,
                'b':
                {'b1': {'f': 3, 'g': 4}}}
        dict3.deep_update(dict4)
        self.assertDictEqual(dict3,
                             {'a': -2, 'b': {'b1': {'f': 3, 'g': 4}, 'b2': 20}})

        # Merge at intermediate keys
        dict2 = DeepUpdateDict(
            {'a': 1,
            'b':
            {'b1': 10,
            'b2': 20}})
        dict2.deep_update(dict_delta2, stop_at_keys='b')
        self.assertDictEqual(
            dict2,
            {'a': -2, 'b': {'b1': {'f': 3, 'g': 4}}}
        )
