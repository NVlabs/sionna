#
# SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
from sionna import config
import unittest
import numpy as np
import tensorflow as tf
from sionna.fec.interleaving import RandomInterleaver, RowColumnInterleaver, Deinterleaver, Turbo3GPPInterleaver
from sionna.utils import BinarySource
from sionna.fec.scrambling import Scrambler

class TestRandomInterleaver(unittest.TestCase):
    """Test random interleaver for consistency."""

    def test_sequence_dimension(self):
        """Test against correct dimensions of the sequence."""
        seq_lengths = [1, 100, 256, 1000]
        batch_sizes = [1, 100, 256, 1000]
        for m in [True, False]: # keep_batch mode
            for inv in [True, False]: # inverse mode
                i = RandomInterleaver(keep_batch_constant=m, inverse=inv)
                for seq_length in seq_lengths:
                    for batch_size in batch_sizes:
                        x = i(tf.zeros([batch_size, seq_length]))
                        self.assertEqual(x.shape,
                                        [int(batch_size),int(seq_length)])

    def test_inverse(self):
        """Test that inverse permutation matches to permutation."""
        seq_length = int(1e3)
        batch_size = int(1e2)
        for m in [True, False]:
            inter = RandomInterleaver(keep_batch_constant=m, seed=123)
            inter2 = RandomInterleaver(keep_batch_constant=m, inverse=True,
                                       seed=123)

            x = np.arange(seq_length)
            x = np.expand_dims(x, axis=0)
            x = np.tile(x, [batch_size, 1])
            y = inter(x)
            z = inter2(y)
            for i in range(batch_size):
                # result must be sorted integers
                self.assertTrue(np.array_equal(z[i,:], np.arange(seq_length)))

            # also test explicit seed
            y = inter([x, 12345])
            z = inter2([y, 12345])
            for i in range(batch_size):
                # result must be sorted integers
                self.assertTrue(np.array_equal(z[i,:], np.arange(seq_length)))

    def test_sequence_batch(self):
        """Test that interleaver sequence is random per batch sample.
        Remark: this tests must fail for keep_batch_constant=True."""
        seq_length = int(1e3)
        batch_size = int(1e1)
        i1 = RandomInterleaver(keep_batch_constant=False) # test valid iff False
        i2 = RandomInterleaver(keep_batch_constant=True) # test valid iff False
        x = np.arange(seq_length)
        x = np.expand_dims(x, axis=0)
        x = np.tile(x, [batch_size, 1])
        y1 = i1(x)
        y2 = i2(x)
        for i in range(batch_size-1):
            for j in range(i+1,batch_size):
                self.assertFalse(np.array_equal(y1[i,:],y1[j,:]))
                self.assertTrue(np.array_equal(y2[i,:],y2[j,:]))

    def test_sequence_realization(self):
        """Test that interleaver sequence are random for each new realization
        iff keep_state==False."""

        seq_length = int(1e3)
        batch_size = int(1e1)
        for m in [True, False]:
            i = RandomInterleaver(keep_batch_constant=m, keep_state=True)
            x = np.arange(seq_length)
            x = np.expand_dims(x, axis=0)
            x = np.tile(x, [batch_size, 1])

            # same results if keep_state=True
            x1 = i(x).numpy()
            x2 = i(x).numpy()
            self.assertTrue(np.array_equal(x1, x2))

            i = RandomInterleaver(keep_batch_constant=m, keep_state=False)
            # different results if keep_state=False
            x1 = i(x).numpy()
            x2 = i(x).numpy()
            self.assertFalse(np.array_equal(x1, x2))

    def test_dimension(self):
        """Test that dimensions can be changed."""
        seq_length = int(1e1)
        batch_size = int(1e2)

        cases = np.array([[1e2, 1e1-1],[1e2, 1e1+1]])

        # test that bs can be variable
        cases = np.array([[1e2+2, 1e1],[1e2+1, 1e1+1]])

        for m in [True, False]:
            llr = config.tf_rng.uniform([tf.cast(batch_size, dtype=tf.int32),
                                     tf.cast(seq_length, dtype=tf.int32)])
            for c in cases:
                for states in [True, False]:
                    S = RandomInterleaver(keep_batch_constant=m,
                                          keep_state=states)
                    llr = config.tf_rng.uniform([tf.cast(c[0], dtype=tf.int32),
                                            tf.cast(c[1], dtype=tf.int32)])
                    S(llr)

    def test_multi_dim(self):
        """Test that 2+D Tensors permutation can be inverted/removed.
        Inherently tests that the output dimensions match.
        """

        # note: test can fail for small dimension_sizes as it may
        # randomly result in the identity interleaver pattern
        shapes=[[10,20,30],[10,22,33,44],[20,10,10,10,6]]

        for s in shapes:
            #check soft-value scrambling (flipp sign)
            llr = config.tf_rng.uniform(tf.constant(s, dtype=tf.int32),
                                    minval=-100,
                                    maxval=100)
            for a in range(0, len(s)):
                for m in [True, False]:
                    if a==0: # check that axis=-1 works as well...axis=0 is
                        # invalid (=batch_dim) and does not need to be checked
                        i = RandomInterleaver(keep_batch_constant=m,
                                              axis=-1,
                                              keep_state=True)
                        i2 = RandomInterleaver(keep_batch_constant=m,
                                               axis=-1,
                                               keep_state=True,
                                               inverse=True)
                    else:
                        i = RandomInterleaver(keep_batch_constant=m,
                                              axis=a,
                                              keep_state=True)
                        i2 = RandomInterleaver(keep_batch_constant=m,
                                               axis=a,
                                               keep_state=True,
                                               inverse=True)

                    x = i([llr, 1234])
                    # after interleaving arrays must be different
                    self.assertTrue(np.any(np.not_equal(x.numpy(),llr.numpy())))

                    # after deinterleaving arrays should be equal again
                    x = i2([x, 1234])
                    self.assertIsNone(np.testing.assert_array_equal(x.numpy(), llr.numpy()))

    def test_invalid_shapes(self):
        """Test that invalid shapes/axis parameter raise error.
        """
        # axis 0 not allowed
        with self.assertRaises(AssertionError):
            RandomInterleaver(axis=0)

        shapes=[[10,20,30],[10,22,33,44],[20,10,10,10,6]]

        for s in shapes:
            with self.assertRaises(AssertionError):
                # axis out bounds...must raise error
                i = RandomInterleaver(axis=len(s))
                llr = config.tf_rng.uniform(tf.constant(s, dtype=tf.int32))
                i(llr)

        # cannot permute batch_dim only
        with self.assertRaises(AssertionError):
            i = RandomInterleaver(axis=1)
            llr = config.tf_rng.uniform(tf.constant([10], dtype=tf.int32),
                                    minval=-10,
                                    maxval=10)
            i(llr)

    def test_keras(self):
        """Test that Keras model can be compiled (supports dynamic shapes)."""
        bs = 10
        k = 100
        source = BinarySource()
        modes = [True, False]
        for m in modes:
            inputs = tf.keras.Input(shape=(k), dtype=tf.float32)
            x = RandomInterleaver(keep_batch_constant=m)(inputs)
            model = tf.keras.Model(inputs=inputs, outputs=x)
            # test that output batch dim is none
            self.assertTrue(model.output_shape[0] is None)

            # test that model can be called
            b = source([bs, k])
            model(b)
            # call twice to see that bs can change
            b2 = source([bs+1, k])
            model(b2)
            model.summary()

    def test_tf_fun(self):
        """Test that tf.function works as expected and XLA work as expected.

        Also tests that arrays are different.
        """
        @tf.function()
        def run_graph(llr):
            return i(llr)

        @tf.function(jit_compile=True)
        def run_graph_xla(llr):
           return i(llr)

        shapes=[[10,20,30], [10,22,33,44], [20,10,10,10,9]]
        modes = [True, False]
        for m in modes:
            for s in shapes:
                #check soft-value scrambling (flipp sign)
                llr = config.tf_rng.uniform(tf.constant(s, dtype=tf.int32))
                i = RandomInterleaver(keep_batch_constant=m)
                x1 = run_graph(llr)
                x2 = run_graph_xla(llr)
                # after interleaving arrays must be different
                self.assertTrue(np.any(np.not_equal(x1.numpy(),llr.numpy())))
                self.assertTrue(np.any(np.not_equal(x2.numpy(),llr.numpy())))

    def test_seed(self):
        """Test that seed can be fed.

        Remark: this test generates multiple interleavers to test the
        influence of different seeds."""

        seq_length = int(1e3)
        batch_size = int(1e1)

        seed = 123456

        for m in [True, False]:
            i1 = RandomInterleaver(keep_batch_constant=m,
                                   seed=seed,
                                   keep_state=True)
            i2 = RandomInterleaver(keep_batch_constant=m, keep_state=True)
            i3 = RandomInterleaver(keep_batch_constant=m,
                                   seed=seed,
                                   keep_state=True)
            i4 = RandomInterleaver(keep_batch_constant=m,
                                   seed=seed+1,
                                   keep_state=True)
            x = np.arange(seq_length)
            x = np.expand_dims(x, axis=0)
            x = np.tile(x, [batch_size, 1])

            # same results if keep_state=True
            x1 = i1(x).numpy()
            x2 = i2(x).numpy()
            x3 = i3(x).numpy()
            x4 = i4(x).numpy()

            #x1 and x3 must be the same (same seed)
            self.assertTrue(np.array_equal(x1, x3))

            #x1 and x2/x4  are not the same (different seed)
            self.assertFalse(np.array_equal(x1, x2))
            self.assertFalse(np.array_equal(x1, x4))

            i11 = RandomInterleaver(keep_batch_constant=m,
                                   seed=seed,
                                   keep_state=False)
            i31 = RandomInterleaver(keep_batch_constant=m,
                                    seed=seed,
                                    keep_state=True)

            # different results if keep_state=False
            x5 = i11(x).numpy()
            x6 = i31(x).numpy()
            self.assertFalse(np.array_equal(x5, x6))

            # test that seed can be also provided to call
            seed = 987654
            x7 = i11([x, seed]).numpy()
            x8 = i11([x, seed+1]).numpy()
            x9 = i11([x, seed]).numpy()
            x10 = i1([x, seed]).numpy()
            self.assertFalse(np.array_equal(x7, x8)) # different seed
            self.assertTrue(np.array_equal(x7, x9)) # same seed
            self.assertTrue(np.array_equal(x7, x10)) # same seed (keep_state=f)

            # test that random seed allows inverse
            x11 = i11([x, seed])
            i21 = RandomInterleaver(keep_batch_constant=m,
                                    keep_state=False,
                                    inverse=True)
            # use different interleaver with same seed to de-interleave
            x12 = i21([x11, seed]).numpy()
            self.assertTrue(np.array_equal(x, x12)) # identity

    def test_s_param(self):
        """Test that interleaver outputs correct S parameter for given seed."""
        N_tests = 100
        k = 100
        inter = RandomInterleaver()

        for s in range(N_tests):
            x = np.arange(k)
            x = np.expand_dims(x, axis=0)
            x_int = inter([x, s]).numpy()
            x_int = np.squeeze(x_int, axis=0)

            s_inter = inter.find_s_min(seed=s, seq_length=k)

            #s_min = 0
            cont = True
            for s_min in range(1, k, 1):
                for i in range(k):
                    a = x_int[i]
                    if i-s_min>=0:
                        b = x_int[i-s_min]
                        if np.abs(a-b)<=s_min:
                            cont=False
                            #break
                    if i+s_min<k:
                        b = x_int[i+s_min]
                        if np.abs(a-b)<=s_min:
                            cont=False
                            #break
                if not cont:
                    break

            self.assertTrue(s_inter==s_min)

    def test_dtype(self):
        """Test that variable dtypes are supported."""
        seq_length = int(1e1)
        batch_size = int(1e2)

        dt_supported = [tf.float16, tf.float32, tf.float64]
        for dt in dt_supported:
            for dt_in in dt_supported:
                b = tf.zeros([batch_size, seq_length], dtype=dt_in)
                inter = RandomInterleaver(dtype=dt)
                x = inter(b)
                assert (x.dtype==dt)

class TestInterleaverRC(unittest.TestCase):
    def test_sequence_dimension(self):
        """Test against correct dimensions of the perm sequence"""
        seq_lengths = [1, 100, 256, 1000]
        depths = [1, 2, 4, 7, 8]
        for d in depths:
            i = RowColumnInterleaver(row_depth=d)
            for seq_length in seq_lengths:
                x, y = i._generate_perm_rc(int(seq_length), d)
                self.assertEqual(x.shape[0],int(seq_length))
                self.assertEqual(y.shape[0],int(seq_length))


    def test_dimension(self):
        """Test against dimension mismatches"""
        seq_length = int(1e1)
        batch_size = int(1e2)

        cases = np.array([[1e2, 1e1-1],[1e2, 1e1+1]])
        depths = [1, 2, 4, 7, 8]
        for d in depths:
            i = RowColumnInterleaver(row_depth=d)
            llr = config.tf_rng.uniform([tf.cast(batch_size, dtype=tf.int32),
                                     tf.cast(seq_length, dtype=tf.int32)])
            i(llr)
            for c in cases:
                llr = config.tf_rng.uniform([tf.cast(c[0], dtype=tf.int32),
                                         tf.cast(c[1], dtype=tf.int32)])
                # should run without error
                i(llr)

        # test that bs can be changed
        cases = np.array([[1e2+2, 1e1],[1e2+1, 1e1]])

        for d in depths:
            i = RowColumnInterleaver(row_depth=d)
            llr = config.tf_rng.uniform([tf.cast(batch_size, dtype=tf.int32),
                            tf.cast(seq_length, dtype=tf.int32)])
            i(llr)
            for c in cases:
                llr = config.tf_rng.uniform([tf.cast(c[0], dtype=tf.int32),
                                         tf.cast(c[1], dtype=tf.int32)])
                i(llr)

    def test_inverse(self):
        """Test that permutation can be inverted/removed"""

        seq_length = int(1e3)
        batch_size = int(1e2)

        # check soft-value scrambling (flip sign)
        llr = config.tf_rng.uniform([tf.cast(batch_size, dtype=tf.int32),
                            tf.cast(seq_length, dtype=tf.int32)])

        depths = [1, 2, 4, 7, 8]
        for d in depths:
            i = RowColumnInterleaver(row_depth=d)
            i2 = RowColumnInterleaver(row_depth=d, inverse=True)
            x = i(llr)
            y = i2(x)
            self.assertIsNone(np.testing.assert_array_equal(y.numpy(),
                                                            llr.numpy()))

    def test_multi_dim(self):
        """Test that 2+D Tensors permutation can be inverted/removed.
        inherently tests that the output dimensions match.

        Also tests that arrays are different.
        """

        shapes=[[10,20,30],[10,22,33,44],[20,10,10,10,9]]

        for s in shapes:
            #check soft-value scrambling (flip sign)
            llr = config.tf_rng.uniform(tf.constant(s, dtype=tf.int32))
            for a in range(0,len(s)):
                depths = [2, 4, 7, 8]
                for d in depths:
                    i = RowColumnInterleaver(row_depth=d, axis=a)
                    i2 = RowColumnInterleaver(row_depth=d, axis=a, inverse=True)
                    x = i(llr)
                    # after interleaving arrays must be different
                    self.assertTrue(np.any(np.not_equal(x.numpy(),llr.numpy())))

                    # after deinterleaving it should be equal again
                    x = i2(x)
                    self.assertIsNone(np.testing.assert_array_equal(x.numpy(), llr.numpy()))

    def test_tf_fun(self):
        """Test that tf.function works as expected and XLA work as expected.

        Also tests that arrays are different.
        """
        @tf.function()
        def run_graph(llr):
            return i(llr)

        @tf.function(jit_compile=True)
        def run_graph_xla(llr):
            return i(llr)

        shapes=[[10,20,30],[10,22,33,44],[20,10,10,10,9]]

        for s in shapes:
            #check soft-value scrambling (flip sign)
            llr = config.tf_rng.uniform(tf.constant(s, dtype=tf.int32))
            for a in range(0,len(s)):
                depths = [2, 4, 7, 8]
                for d in depths:
                    i = RowColumnInterleaver(row_depth=d, axis=a)
                    x1 = run_graph(llr)
                    x2 = run_graph_xla(llr)
                    # after interleaving arrays must be different
                    self.assertTrue(np.any(np.not_equal(x1.numpy(),
                                                        llr.numpy())))
                    self.assertTrue(np.any(np.not_equal(x2.numpy(),
                                                        llr.numpy())))


    def test_invalid_axis(self):
        """Test that 2+D Tensors and invalid axis raise error
        """

        shapes=[[10,20,30],[10,22,33,44],[20,10,10,10,6]]

        for s in shapes:
            with self.assertRaises(AssertionError):
                i = RowColumnInterleaver(row_depth=4, axis=len(s))
                # axis is out bounds; must raise an error
                llr = config.tf_rng.uniform(tf.constant(s, dtype=tf.int32))
                i(llr)

    def test_keras(self):
        """Test that Keras model can be compiled (supports dynamic shapes)"""
        bs = 10
        k = 100
        source = BinarySource()

        inputs = tf.keras.Input(shape=(k), dtype=tf.float32)
        x = RowColumnInterleaver(row_depth=4)(inputs)
        model = tf.keras.Model(inputs=inputs, outputs=x)
        # test that output batch dim is none
        self.assertTrue(model.output_shape[0] is None)

        # test that model can be called
        b = source([bs,k])
        model(b)
        # call twice to see that bs can change
        b2 = source([bs+1,k])
        model(b2)
        model.summary()

    def test_dtype(self):
        """Test that variable dtypes are supported."""
        seq_length = int(1e1)
        batch_size = int(1e2)

        dt_supported = [tf.float16, tf.float32, tf.float64]
        for dt in dt_supported:
            for dt_in in dt_supported:
                b = tf.zeros([batch_size, seq_length], dtype=dt_in)
                inter = RowColumnInterleaver(row_depth=4, dtype=dt)
                x = inter(b)
                assert (x.dtype==dt)

class TestDeinterleaver(unittest.TestCase):
    """Test Deinterleaver class."""

    def test_identity(self):
        """Test that deinterleave can invert Random-/RCInterleaver."""
        seq_length = int(1e1)
        batch_size = int(1e2)

        # test RowColumnInterleaver
        inter_rc = RowColumnInterleaver(row_depth=3)
        deinter_rc = Deinterleaver(inter_rc)

        x = config.tf_rng.uniform([tf.cast(batch_size, dtype=tf.int32),
                               tf.cast(seq_length, dtype=tf.int32)])

        y = inter_rc(x)
        z = deinter_rc(y)

        self.assertFalse(np.array_equal(x.numpy(), y.numpy()))
        self.assertTrue(np.array_equal(x.numpy(), z.numpy()))

        # test RandomInterleaver
        for k in (True, False): # same sequence per batch
            for s in (None, 1234, 876): # test different seeds
                inter_random = RandomInterleaver(keep_batch_constant=k, seed=s)
                deinter_random = Deinterleaver(inter_random)

                y = inter_random(x)
                z = deinter_random(y)

                self.assertFalse(np.array_equal(x.numpy(), y.numpy()))
                self.assertTrue(np.array_equal(x.numpy(), z.numpy()))


    def test_tf_fun(self):
        """Test that tf.function works as expected and XLA work as expected.
        """
        @tf.function()
        def run_graph(llr):
            return de_int_rc(int_rc(llr)), de_int_rand(int_rand(llr))

        @tf.function(jit_compile=True)
        def run_graph_xla(llr):
            return de_int_rc(int_rc(llr)), de_int_rand(int_rand(llr))

        shapes=[[10,20,30],[10,22,33,44],[20,10,10,10,9]]

        for s in shapes:
            # check soft-value scrambling (flip sign)
            llr = config.tf_rng.uniform(tf.constant(s, dtype=tf.int32))
            for a in range(0,len(s)):
                depths = [2, 4, 7, 8]
                for d in depths:
                    int_rc = RowColumnInterleaver(row_depth=d, axis=a)
                    int_rand = RandomInterleaver()

                    de_int_rc = Deinterleaver(int_rc)
                    de_int_rand = Deinterleaver(int_rand)

                    x1, x2 = run_graph(llr)
                    x3, x4 = run_graph_xla(llr)
                    # after interleaving arrays must be different
                    for x in (x1, x2, x3, x4):
                        self.assertTrue(np.array_equal(llr.numpy(), x.numpy()))

    def test_axis(self):
        """Test that deinterleaver operates on correct axis."""

        x = config.tf_rng.uniform([10, 20, 20, 20])

        for a in (1, 2, 3, -1, -2):

            # test RowColumnInterleaver
            inter_rc = RowColumnInterleaver(row_depth=3, axis=a)
            deinter_rc = Deinterleaver(inter_rc)

            y = inter_rc(x)
            z = deinter_rc(y)

            self.assertFalse(np.array_equal(x.numpy(), y.numpy()))
            self.assertTrue(np.array_equal(x.numpy(), z.numpy()))

            # test RandomInterleaver

            inter_random = RandomInterleaver(axis=a)
            deinter_random = Deinterleaver(inter_random)

            y = inter_random(x)
            z = deinter_random(y)

            self.assertFalse(np.array_equal(x.numpy(), y.numpy()))
            self.assertTrue(np.array_equal(x.numpy(), z.numpy()))

    def test_dtype(self):
        """test that arbitrary dtypes are supported."""

        dtypes_supported = (tf.float16, tf.float32, tf.float64, tf.int32,
                            tf.int64, tf.complex128, tf.complex64)

        for dt_in in dtypes_supported:

            # tf.uniform does not support complex dtypes
            if dt_in is (tf.complex64):
                x = config.tf_rng.uniform([10, 20], maxval=10, dtype=tf.float32)
                x = tf.complex(x, tf.zeros_like(x))
            elif dt_in is (tf.complex128):
                x = config.tf_rng.uniform([10, 20], maxval=10, dtype=tf.float64)
                x = tf.complex(x, tf.zeros_like(x))
            else:
                x = config.tf_rng.uniform([10, 20], maxval=10, dtype=dt_in)

            # test RowColumnInterleaver
            inter_rc = RowColumnInterleaver(row_depth=3,
                                            dtype=dt_in)

            # inherits dtype from inter
            deinter_rc1 = Deinterleaver(inter_rc)
            # custom dtype
            deinter_rc2 = Deinterleaver(inter_rc, dtype=dt_in)

            y = inter_rc(x)
            z1 = deinter_rc1(y)
            z2 = deinter_rc2(y)

            self.assertTrue(y.dtype==dt_in)
            self.assertTrue(z1.dtype==dt_in)
            self.assertTrue(z2.dtype==dt_in)

            # test RandomInterleaver
            inter_rand = RandomInterleaver(dtype=dt_in)

            # inherits dtype from inter
            deinter_rand1 = Deinterleaver(inter_rand)
            # custom dtype
            deinter_rand2 = Deinterleaver(inter_rand,
                                            dtype=dt_in)

            y = inter_rand(x)
            z1 = deinter_rand1(y)
            z2 = deinter_rand2(y)

            self.assertTrue(y.dtype==dt_in)
            self.assertTrue(z1.dtype==dt_in)
            self.assertTrue(z2.dtype==dt_in)

    def test_invalid_input(self):
        """test against invalid parameters."""

        inter1 = RandomInterleaver()
        inter2 = RowColumnInterleaver(3)
        scram = Scrambler()

        # invalid input
        for s in (scram, None, 124):
            with self.assertRaises(ValueError):
                x = Deinterleaver(s)

class TestTurbo3GPPInterleaver(unittest.TestCase):
    """Test Turbo3GPP interleaver for consistency."""

    def test_sequence_dimension(self):
        """Test against correct dimensions of the sequence."""

        seq_lengths = [1, 100, 256, 1000]
        batch_sizes = [1, 100, 256, 1000]
        for inv in [True, False]: # inverse mode
            i = Turbo3GPPInterleaver(inverse=inv)
            for seq_length in seq_lengths:
                for batch_size in batch_sizes:
                    x = i(tf.zeros([batch_size, seq_length]))
                    self.assertEqual(x.shape,
                                    [int(batch_size),int(seq_length)])

    def test_inverse(self):
        """Test that inverse permutation matches to permutation."""
        seq_length = int(1e3)
        batch_size = int(1e2)

        inter = Turbo3GPPInterleaver()
        inter2 = Turbo3GPPInterleaver(inverse=True)
        # also test that the deinterleaver can be used
        deinter = Deinterleaver(inter)

        x = np.arange(seq_length)
        x = np.expand_dims(x, axis=0)
        x = np.tile(x, [batch_size, 1])
        y = inter(x)
        z = inter2(y)
        z2 = deinter(y)
        for i in range(batch_size):
            # result must be sorted integers
            self.assertTrue(np.array_equal(z[i,:], np.arange(seq_length)))
            self.assertTrue(np.array_equal(z2[i,:], np.arange(seq_length)))

    def test_dimension(self):
        """Test that dimensions can be changed."""
        seq_length = int(1e1)
        batch_size = int(1e2)

        cases = np.array([[1e2, 1e1-1], [1e2, 1e1+1]])

        # test that bs can be variable
        cases = np.array([[1e2+2, 1e1], [1e2+1, 1e1+1]])

        llr = config.tf_rng.uniform([tf.cast(batch_size, dtype=tf.int32),
                                tf.cast(seq_length, dtype=tf.int32)])
        for c in cases:
                i = Turbo3GPPInterleaver()
                llr = config.tf_rng.uniform([tf.cast(c[0], dtype=tf.int32),
                                         tf.cast(c[1], dtype=tf.int32)])
                i(llr)

    def test_multi_dim(self):
        """Test that 2+D Tensors permutation can be inverted/removed.
        Inherently tests that the output dimensions match.
        """

        shapes=[[10,20,30], [10,22,33,44], [20,10,10,10,6]]

        for s in shapes:
            #check soft-value scrambling (flipp sign)
            llr = config.tf_rng.uniform(tf.constant(s, dtype=tf.int32),
                                    minval=-100,
                                    maxval=100)
            for a in range(0, len(s)):
                    if a==0: # check that axis=-1 works as well...axis=0 is
                        # invalid (=batch_dim) and does not need to be checked
                        i = Turbo3GPPInterleaver(axis=-1)
                        i2 = Turbo3GPPInterleaver(axis=-1,
                                                  inverse=True)
                    else:
                        i = Turbo3GPPInterleaver(axis=a)
                        i2 = Turbo3GPPInterleaver(axis=a,
                                                  inverse=True)

                    x = i(llr)
                    # after interleaving arrays must be different
                    self.assertTrue(np.any(np.not_equal(x.numpy(),llr.numpy())))

                    # after deinterleaving arrays should be equal again
                    x = i2(x)
                    self.assertIsNone(np.testing.assert_array_equal(x.numpy(), llr.numpy()))

    def test_invalid_shapes(self):
        """Test that invalid shapes/axis parameter raise error.
        """
        # axis 0 not allowed
        with self.assertRaises(AssertionError):
            Turbo3GPPInterleaver(axis=0)

        # k>6144
        i = Turbo3GPPInterleaver(axis=-1)
        s = [10, 6145]
        llr = config.tf_rng.uniform(tf.constant(s, dtype=tf.int32))
        with self.assertRaises(AssertionError):
            i(llr)

        shapes=[[10,20,30], [10,22,33,44], [20,10,10,10,6]]

        for s in shapes:
            with self.assertRaises(AssertionError):
                # axis out bounds...must raise error
                i = Turbo3GPPInterleaver(axis=len(s))
                llr = config.tf_rng.uniform(tf.constant(s, dtype=tf.int32))
                i(llr)

        # cannot permute batch_dim only
        with self.assertRaises(AssertionError):
            i = Turbo3GPPInterleaver(axis=1)
            llr = config.tf_rng.uniform(tf.constant([10], dtype=tf.int32),
                                    minval=-10,
                                    maxval=10)
            i(llr)

    def test_keras(self):
        """Test that Keras model can be compiled (supports dynamic shapes)."""
        bs = 10
        k = 100
        source = BinarySource()

        inputs = tf.keras.Input(shape=(k), dtype=tf.float32)
        x = Turbo3GPPInterleaver()(inputs)
        model = tf.keras.Model(inputs=inputs, outputs=x)
        # test that output batch dim is none
        self.assertTrue(model.output_shape[0] is None)

        # test that model can be called
        b = source([bs, k])
        model(b)
        # call twice to see that bs can change
        b2 = source([bs+1, k])
        model(b2)
        model.summary()

    def test_tf_fun(self):
        """Test that tf.function works as expected and XLA work as expected.

        Also tests that arrays are different.
        """
        @tf.function()
        def run_graph(llr):
            return i(llr)

        @tf.function(jit_compile=True)
        def run_graph_xla(llr):
           return i(llr)

        shapes=[[10,20,30], [10,22,33,44], [20,10,10,10,9]]
        for s in shapes:
            #check soft-value scrambling (flip sign)
            llr = config.tf_rng.uniform(tf.constant(s, dtype=tf.int32))
            i = Turbo3GPPInterleaver()
            x1 = run_graph(llr)
            x2 = run_graph_xla(llr)
            # after interleaving arrays must be different
            self.assertTrue(np.any(np.not_equal(x1.numpy(),llr.numpy())))
            self.assertTrue(np.any(np.not_equal(x2.numpy(),llr.numpy())))

            # XLA and graph mode should result in the same array
            self.assertTrue(np.array_equal(x1.numpy(),x2.numpy()))

        # test for variable lengths
        i = Turbo3GPPInterleaver()
        llr = config.tf_rng.uniform(tf.constant((10,100), dtype=tf.int32))
        x = run_graph(llr)
        x = run_graph_xla(llr)
        llr = config.tf_rng.uniform(tf.constant((10,101), dtype=tf.int32))
        x = run_graph(llr)
        x = run_graph_xla(llr)

    def test_dtype(self):
        """Test that variable dtypes are supported."""
        seq_length = int(1e1)
        batch_size = int(1e2)

        dt_supported = [tf.float16, tf.float32, tf.float64]
        for dt in dt_supported:
            for dt_in in dt_supported:
                b = tf.zeros([batch_size, seq_length], dtype=dt_in)
                inter = Turbo3GPPInterleaver(dtype=dt)
                x = inter(b)
                assert (x.dtype==dt)
