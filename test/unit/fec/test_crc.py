#
# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
try:
    import sionna
except ImportError as e:
    import sys
    sys.path.append("../")

import unittest
import numpy as np
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
print('Number of GPUs available :', len(gpus))
if gpus:
    gpu_num = 0 # Number of the GPU to be used
    try:
        tf.config.set_visible_devices(gpus[gpu_num], 'GPU')
        print('Only GPU number', gpu_num, 'used.')
        tf.config.experimental.set_memory_growth(gpus[gpu_num], True)
    except RuntimeError as e:
        print(e)
from sionna.fec.crc import CRCEncoder, CRCDecoder
from sionna.utils import BinarySource

VALID_POLS = ["CRC24A", "CRC24B", "CRC24C", "CRC16", "CRC11", "CRC6"]

class TestCRC(unittest.TestCase):
    """"Unittests for the CRC encoder/decoder class."""

    def test_polynomials(self):
        """Check that all valid polynomials from 38.212 are supported.
        Also check against invalid input parameters."""

        crc_polys =[
            [1,1,0,0,0,0,1,1,0,0,1,0,0,1,1,0,0,1,1,1,1,1,0,1,1],
            [1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,1,1],
            [1,1,0,1,1,0,0,1,0,1,0,1,1,0,0,0,1,0,0,0,1,0,1,1,1],
            [1,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,1],
            [1,1,1,0,0,0,1,0,0,0,0,1],
            [1,1,0,0,0,0,1]]

        for idx, p in enumerate(VALID_POLS):
            c = CRCEncoder(p)
            # test that result is correct
            self.assertTrue(np.array_equal(c.crc_pol, crc_polys[idx]))
            # test that crc_length has correct length of polynomial
            self.assertTrue(c.crc_length == (len(crc_polys[idx])-1))

        # test that invalid input raises an Exception
        with self.assertRaises(AssertionError):
            CRCEncoder(24)
        # test against unknown CRC polynomial
        with self.assertRaises(ValueError):
            CRCEncoder("CRC17")

    def test_output_dim(self):
        """Test that output dims are correct (=k+crc_len)."""

        # test against random shapes
        shapes = [[1, 10],[10, 3, 3],[1, 2, 3, 4, 1000]]

        for pol in VALID_POLS:
            crc_enc = CRCEncoder(pol)
            crc_dec = CRCDecoder(crc_enc)
            for s in shapes:
                u = tf.zeros(s)
                c = crc_enc(u)
                u_hat, crc_indicator = crc_dec(c)
                c = c.numpy()
                crc_indicator = crc_indicator.numpy()
                u_hat = u_hat.numpy()
                # output shapes are equal to input shape (besides last dim)
                self.assertTrue(np.array_equal(c.shape[0:-1], s[0:-1]))
                # last dimension of output is increased by 'crc_length'
                self.assertTrue(c.shape[-1]==s[-1]+crc_enc.crc_length)
                #check dimensions of "crc_valid indicator" (boolean)
                self.assertTrue(np.array_equal(crc_indicator.shape[0:-1],
                                               s[0:-1]))
                self.assertTrue(crc_indicator.shape[-1]==1)
                # check that decoder removes parity bits (=original shape)
                self.assertTrue(np.array_equal(u_hat.shape, s))

    def test_rank1(self):
        """Test that rank 1 input raises error."""

        shapes = [[1],[10]]
        for pol in VALID_POLS:
            crc_enc = CRCEncoder(pol)
            for s in shapes:
                with self.assertRaises(tf.errors.InvalidArgumentError):
                    u = tf.zeros(s)
                    crc_enc(u)

    def test_tf_fun(self):
        """Test that tf.function works and XLA is supported."""

        # graph mode
        @tf.function()
        def run_graph(s, pol):
            crc_enc = CRCEncoder(pol)
            crc_dec = CRCDecoder(crc_enc)
            u = tf.zeros(s)
            x = crc_enc(u)
            y, z = crc_dec(x)
            return y

        # XLA mode
        @tf.function(jit_compile=True)
        def run_graph_xla(s, pol):
            crc_enc = CRCEncoder(pol)
            crc_dec = CRCDecoder(crc_enc)
            u = tf.zeros(s)
            x = crc_enc(u)
            y, z = crc_dec(x)
            return y

        shapes = [[1, 10], [10, 3, 3], [1 ,2, 3, 4, 1000]]
        # test for different shapes
        for pol in VALID_POLS:
            for s in shapes:
                x = run_graph(s, pol)
                x = run_graph_xla(s, pol)


    def test_valid_crc(self):
        """Test that CRC of error-free codewords always holds, i.e.,
        re-encoding always yields a valid CRC.
        """

        shapes = [[100, 10], [50, 3, 3], [100, 2, 3, 4, 100], [1,int(1e6)]]
        source = BinarySource()

        for pol in VALID_POLS:
            crc_enc = CRCEncoder(pol)
            crc_dec = CRCDecoder(crc_enc)
            for s in shapes:
                u = source(s)
                x = crc_enc(u) # add CRC parity bits
                _, y2 = crc_dec(x) # perform CRC check

                # CRC check for CRC encoded data x must always hold
                self.assertTrue(np.all(y2.numpy()))

    def test_error_patters(self):
        """"Test that CRC detects random error patterns."""

        shapes = [100, 100]
        source = BinarySource()

        for pol in VALID_POLS:
            crc_enc = CRCEncoder(pol)
            crc_dec = CRCDecoder(crc_enc)
            u = source(shapes)
            x = crc_enc(u) # add CRC

            # add error patters with 3 random errors per CW
            e = tf.concat([tf.ones([100, 3]),
                          tf.zeros([100, (97+crc_enc.crc_length)])], axis=1)
            # shuffling permutes first dim, but we need the second dim
            e = tf.transpose(e, (1,0))
            e = tf.random.shuffle(e)
            e = tf.transpose(e, (1,0))
            # add error vector
            x += e
            x = tf.math.mod(x, 2) # take mod2 as we are in GF(2)

            _, y2 = crc_dec(x) # perform CRC check

            # CRC should detect all errors (= all checks return False)
            self.assertFalse(np.any(y2.numpy()))

    def test_examples(self):
        """Test against some manually calculated examples."""

        crc_polys =[
            [1,0,0,0,0,1,1,0,0,1,0,0,1,1,0,0,1,1,1,1,1,0,1,1],
            [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,1,1],
            [1,0,1,1,0,0,1,0,1,0,1,1,0,0,0,1,0,0,0,1,0,1,1,1],
            [0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,1],
            [1,1,0,0,0,1,0,0,0,0,1],
            [1,0,0,0,0,1]]

        for idx, pol in enumerate(VALID_POLS):
            crc_length = len(crc_polys[idx])
            crc_enc = CRCEncoder(pol) # init encoder
            u = tf.constant(1., dtype=tf.float32) # encode single "1"
            u = tf.reshape(u, [1,1])
            x = crc_enc(u)
            x = tf.reshape(x, (-1)).numpy()
            x = x[-crc_length:] # slice only CRC bits
            # calculate results by hand
            x_ref = np.array(crc_polys[idx])
            self.assertTrue(np.array_equal(x, x_ref))

    def test_valid_encoding(self):
        """Check all valid polynomials from 38.212 against
        a dataset from a reference implementation."""

        for pol in VALID_POLS:

            ref_path = 'codes/crc/'

            # load reference codewords
            u = np.load(ref_path + "crc_u_" + pol + ".npy")
            x_ref_np = np.load(ref_path + "crc_x_ref_np_" + pol + ".npy")

            crc_enc = CRCEncoder(pol)

            x = crc_enc(u) # add CRC
            x = tf.reshape(x, (-1))
            x_crc = x.numpy()[-crc_enc.crc_length:] #consider only CRC positions

            self.assertTrue(np.array_equal(x_crc, x_ref_np))

            # test properties k,n
            self.assertTrue(crc_enc.k==u.shape[-1])
            self.assertTrue(crc_enc.n==x.shape[-1])

    def test_keras(self):
        """Test that Keras model can be compiled (=supports dynamic shapes)."""

        bs = 10
        k = 100
        source = BinarySource()

        inputs = tf.keras.Input(shape=(k), dtype=tf.float32)
        x = CRCEncoder("CRC24A")(inputs)
        y, _ = CRCDecoder(CRCEncoder("CRC24A"))(x)
        model = tf.keras.Model(inputs=inputs, outputs=y)
        # test that output batch dim is none
        self.assertTrue(model.output_shape[0] is None)

        b = source([bs, k])
        model(b)
        # call twice to see that bs can change
        b2 = source([bs+1, k])
        model(b2)

        model.summary()

    def test_dtype(self):
        """Test support for variable dtypes."""

        dt = [tf.float16, tf.float32, tf.float64]
        pol = "CRC24A"
        shape = [100, 10]
        source = BinarySource()

        for d_in in dt:
            for d_enc in dt:
                for d_dec in dt:
                    crc_enc = CRCEncoder(pol, dtype=d_enc)
                    crc_dec = CRCDecoder(crc_enc, dtype=d_dec)

                    u = tf.cast(source(shape), d_in)
                    x = crc_enc(u) # add CRC parity bits
                    y, _ = crc_dec(x) # perform CRC check

                    self.assertTrue(u.dtype==d_in)
                    self.assertTrue(x.dtype==d_enc)
                    self.assertTrue(y.dtype==d_dec)
