#
# SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import unittest
import numpy as np
import tensorflow as tf
from sionna import config
from sionna.mapping import pam_gray, pam, qam, Constellation

def bpsk(b):
    return 1-2*b[0]

def pam4(b):
    return (1-2*b[0])*(2-(1-2*b[1]))

def pam8(b):
    return (1-2*b[0])*(4-(1-2*b[1])*(2-(1-2*b[2])))

def pam16(b):
    return (1-2*b[0])*(8-(1-2*b[1])*(4-(1-2*b[2])*(2-(1-2*b[3]))))

class TestPAMGray(unittest.TestCase):
    def test_pam(self):
        """Test against 5G formulars"""
        for n, comp in enumerate([bpsk, pam4, pam8, pam16]):
            for i in range(0, 2**n):
                b = np.array(list(np.binary_repr(i,n+1)), dtype=np.int16)
                self.assertEqual(pam_gray(b), comp(b))

class TestPAM(unittest.TestCase):
    def test_pam(self):
        """Test constellations against 5G formulars"""
        for n, comp in enumerate([bpsk, pam4, pam8, pam16]):
            num_bits_per_symbol = n+1
            c = pam(num_bits_per_symbol, normalize=False)
            for i in range(0, 2**num_bits_per_symbol):
                b = np.array(list(np.binary_repr(i,num_bits_per_symbol)), dtype=np.int16)
                self.assertTrue(np.equal(c[i], comp(b)))

    def test_normalization(self):
        """Test that constellations have unit energy"""
        for num_bits_per_symbol in np.arange(1,17):
            c = pam(num_bits_per_symbol, normalize=True)
            self.assertAlmostEqual(1, np.mean(np.abs(c)**2), 4)

class TestQAM(unittest.TestCase):
    def test_normalization(self):
        """Test that constellations have unit energy"""
        for num_bits_per_symbol in [2, 4, 6, 8, 10, 12, 14]:
            c = qam(num_bits_per_symbol, normalize=True)
            self.assertAlmostEqual(1, np.mean(np.abs(c)**2), 4)

    def test_qam(self):
        """Test constellations against 5G formulars"""
        for n, pam in enumerate([bpsk, pam4, pam8, pam16]):
            num_bits_per_symbol = 2*(n+1)
            c = qam(num_bits_per_symbol, normalize=False)
            for i in range(0, 2**num_bits_per_symbol):
                b = np.array(list(np.binary_repr(i,2*(n+1))), dtype=np.int16)
                self.assertTrue(np.equal(c[i], pam(b[0::2]) + 1j*pam(b[1::2])))

class TestConstellation(unittest.TestCase):
    def test_assertions(self):
        with self.assertRaises(AssertionError):
            Constellation("custom2", 2)
        with self.assertRaises(AssertionError):
            Constellation("custom", 0)
        with self.assertRaises(AssertionError):
            Constellation("qam", 0)
        with self.assertRaises(AssertionError):
            Constellation("qam", 3)
        with self.assertRaises(AssertionError):
            Constellation("qam", 2.1)
        with self.assertRaises(AssertionError):
            Constellation("custom", 3.7)
        with self.assertRaises(AssertionError):
            num_bits_per_symbol = 3
            initial_value = np.zeros([2**num_bits_per_symbol-1])
            Constellation("custom", num_bits_per_symbol, initial_value)
        with self.assertRaises(AssertionError):
            num_bits_per_symbol = 3
            initial_value = np.zeros([2**num_bits_per_symbol])
            Constellation("qam", num_bits_per_symbol, initial_value)

    def test_outputdimensions(self):
        for num_bits_per_symbol in [2, 4, 6, 8, 10]:
            c = Constellation("qam", num_bits_per_symbol)
            self.assertTrue(np.array_equal(c.points.shape, [2**num_bits_per_symbol]))

        for num_bits_per_symbol in range(1,11):
            c = Constellation("custom", num_bits_per_symbol)
            self.assertTrue(np.array_equal(c.points.shape, [2**num_bits_per_symbol]))

    def test_normalization_and_centering(self):
        for constellation_type in ("qam", "custom"):
            for num_bits_per_symbol in [2, 4, 6, 8, 10]:
                c = Constellation(constellation_type, num_bits_per_symbol, normalize=False, center=False)
                self.assertNotAlmostEqual(np.mean(np.abs(c.points)**2), 1.0, 5)
                if constellation_type == "custom":
                    self.assertNotAlmostEqual(np.mean(c.points), 0.0 + 1j*0.0, 5)
                else:
                    self.assertAlmostEqual(np.mean(c.points), 0.0 + 1j*0.0, 5)

                c = Constellation(constellation_type, num_bits_per_symbol, normalize=True, center=False)
                if constellation_type == "custom":
                    self.assertNotAlmostEqual(np.mean(c.points), 0.0 + 1j*0.0, 5)
                else:
                    self.assertAlmostEqual(np.mean(c.points), 0.0 + 1j*0.0, 5)

                c = Constellation(constellation_type, num_bits_per_symbol, normalize=False, center=True)
                self.assertNotAlmostEqual(np.mean(np.abs(c.points)**2), 1.0, 5)
                self.assertAlmostEqual(np.mean(c.points), 0.0 + 1j*0.0, 5)

                c = Constellation(constellation_type, num_bits_per_symbol, normalize=True, center=True)
                self.assertAlmostEqual(np.mean(np.abs(c.points)**2), 1.0, 5)
                self.assertAlmostEqual(np.mean(c.points), 0.0 + 1j*0.0, 5)

    def test_initial_value(self):
        constellation_type = "custom"
        for num_bits_per_symbol in range(1,10):
            initial_value = config.np_rng.normal(size=[2**num_bits_per_symbol]) + 1j*config.np_rng.normal(size=[2**num_bits_per_symbol])
            
            c = Constellation(constellation_type, num_bits_per_symbol, initial_value=initial_value, normalize=False, center=False)
            self.assertTrue (np.allclose(c.points, initial_value))
            initial_value = tf.Variable(initial_value)
            
            c = Constellation(constellation_type, num_bits_per_symbol, initial_value=initial_value, normalize=False, center=False)
            self.assertTrue (np.allclose(c.points, initial_value))
            self.assertNotAlmostEqual(np.mean(np.abs(c.points)**2), 1.0, 5)
            self.assertNotAlmostEqual(np.mean(c.points), 0.0 + 1j*0.0, 5)
            
            c = Constellation(constellation_type, num_bits_per_symbol, initial_value=initial_value, normalize=True, center=False)
            self.assertAlmostEqual(np.mean(np.abs(c.points)**2), 1.0, 5)
            self.assertNotAlmostEqual(np.mean(c.points), 0.0 + 1j*0.0, 5)

            c = Constellation(constellation_type, num_bits_per_symbol, initial_value=initial_value, normalize=False, center=True)
            self.assertNotAlmostEqual(np.mean(np.abs(c.points)**2), 1.0, 5)
            self.assertAlmostEqual(np.mean(c.points), 0.0 + 1j*0.0, 5)
            
            c = Constellation(constellation_type, num_bits_per_symbol, initial_value=initial_value, normalize=True, center=True)
            self.assertAlmostEqual(np.mean(np.abs(c.points)**2), 1.0, 5)
            self.assertAlmostEqual(np.mean(c.points), 0.0 + 1j*0.0, 5)



class TestConstellationDTypes(unittest.TestCase):

    def test_qam(self):
        "Test QAM constellation with all possible configurations"
        for num_bits_per_symbol in [2,4,6,8, 10]:
            for normalize in [True, False]:
                for center in [True, False]:
                    for trainable in [True, False]:
                        for dtype in [tf.complex64, tf.complex128]:
                            c = Constellation("qam",
                                              num_bits_per_symbol,
                                              None,
                                              normalize,
                                              center,
                                              trainable,
                                              dtype)
                            self.assertEqual(c.points.dtype, dtype)

    def test_pam(self):
        "Test PAM constellation with all possible configurations"
        for num_bits_per_symbol in [1,2,3,4,5,6,7,8,9,10]:
            for normalize in [True, False]:
                for center in [True, False]:
                    for trainable in [True, False]:
                        for dtype in [tf.complex64, tf.complex128]:
                            c = Constellation("pam",
                                              num_bits_per_symbol,
                                              None,
                                              normalize,
                                              center,
                                              trainable,
                                              dtype)
                            self.assertEqual(c.points.dtype, dtype)

    def test_custom_no_initial_value(self):
        "Test custom constellation without initial_value with all possible configurations"
        for num_bits_per_symbol in [1,2,3,4,5,6,7,8]:
            for normalize in [True, False]:
                for center in [True, False]:
                    for trainable in [True, False]:
                        for dtype in [tf.complex64, tf.complex128]:
                            c = Constellation("custom",
                                              num_bits_per_symbol,
                                              None,
                                              normalize,
                                              center,
                                              trainable,
                                              dtype)
                            self.assertEqual(c.points.dtype, dtype)
                            if center:
                                self.assertAlmostEqual(np.mean(c.points), 0, 6)
                            if normalize:
                                self.assertAlmostEqual(np.mean(np.abs(c.points)**2), 1, 6)

    def test_custom_with_initial_value(self):
        "Test custom constellation with initial_value with all possible configurations"
        for num_bits_per_symbol in [1,2,3,4,5,6,7,8]:
            for normalize in [True, False]:
                for center in [True, False]:
                    for trainable in [True, False]:
                        for dtype in [tf.complex64, tf.complex128]:
                            initial_value = config.np_rng.normal(size=[2**num_bits_per_symbol]) + \
                                            1j*config.np_rng.normal(size=[2**num_bits_per_symbol])
                            c = Constellation("custom",
                                              num_bits_per_symbol,
                                              initial_value,
                                              normalize,
                                              center,
                                              trainable,
                                              dtype)
                            self.assertEqual(c.points.dtype, dtype)
                            if center:
                                self.assertAlmostEqual(np.mean(c.points), 0, 5)
                            if normalize:
                                self.assertAlmostEqual(np.mean(np.abs(c.points)**2), 1, 5)
