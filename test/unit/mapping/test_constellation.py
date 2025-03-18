#
# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0#

import unittest
import numpy as np
import tensorflow as tf
from sionna.phy.config import config, dtypes
from sionna.phy.mapping import pam_gray, pam, qam, Constellation

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
                b = np.array(list(np.binary_repr(i,n+1)), dtype=np.int32)
                self.assertEqual(pam_gray(b), comp(b))

class TestPAM(unittest.TestCase):
    def test_pam(self):
        """Test constellations against 5G formulars"""
        for n, comp in enumerate([bpsk, pam4, pam8, pam16]):
            num_bits_per_symbol = n+1
            c = pam(num_bits_per_symbol, normalize=False)
            for i in range(0, 2**num_bits_per_symbol):
                b = np.array(list(np.binary_repr(i,num_bits_per_symbol)), dtype=np.int32)
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
                b = np.array(list(np.binary_repr(i,2*(n+1))), dtype=np.int32)
                self.assertTrue(np.equal(c[i], pam(b[0::2]) + 1j*pam(b[1::2])))

class TestConstellation(unittest.TestCase):
    def test_assertions(self):
        with self.assertRaises(ValueError):
            Constellation("custom2", 2)
        with self.assertRaises(ValueError):
            Constellation("custom", 0)
        with self.assertRaises(ValueError):
            Constellation("qam", 0)
        with self.assertRaises(ValueError):
            Constellation("qam", 3)
        with self.assertRaises(ValueError):
            Constellation("qam", 2.1)
        with self.assertRaises(ValueError):
            Constellation("custom", 3.7)
        with self.assertRaises(ValueError):
            num_bits_per_symbol = 3
            points = np.zeros([2**num_bits_per_symbol-1])
            Constellation("custom", num_bits_per_symbol, points=points)
        with self.assertRaises(ValueError):
            num_bits_per_symbol = 3
            points= np.zeros([2**num_bits_per_symbol])
            Constellation("qam", num_bits_per_symbol, points=points)

    def test_outputdimensions(self):
        for num_bits_per_symbol in [2, 4, 6, 8, 10]:
            c = Constellation("qam", num_bits_per_symbol)
            self.assertTrue(np.array_equal(c().shape, [2**num_bits_per_symbol]))

        for num_bits_per_symbol in range(1,11):
            points = tf.complex(config.tf_rng.normal([2**num_bits_per_symbol]),
                                config.tf_rng.normal([2**num_bits_per_symbol]))
            c = Constellation("custom", num_bits_per_symbol, points=points)
            self.assertTrue(np.array_equal(c().shape, [2**num_bits_per_symbol]))

    def test_normalization_and_centering(self):
        for constellation_type in ("pam", "qam", "custom"):
            for num_bits_per_symbol in [2, 4, 6, 8, 10]:
                if constellation_type == "custom":
                    points = tf.complex(config.tf_rng.normal([2**num_bits_per_symbol]),
                                        config.tf_rng.normal([2**num_bits_per_symbol]))
                else:
                    points = None
                c = Constellation(constellation_type, num_bits_per_symbol, points=points, normalize=False, center=False)
                if constellation_type == "custom":
                    self.assertNotAlmostEqual(np.mean(c()), 0.0 + 1j*0.0, 5)
                    self.assertNotAlmostEqual(np.mean(np.abs(c())**2), 1.0, 5)
                else:
                    self.assertAlmostEqual(np.mean(np.abs(c())**2), 1.0, 5)
                    self.assertAlmostEqual(np.mean(c()), 0.0 + 1j*0.0, 5)

                c = Constellation(constellation_type, num_bits_per_symbol, points=points, normalize=True, center=False)
                if constellation_type == "custom":
                    self.assertNotAlmostEqual(np.mean(c()), 0.0 + 1j*0.0, 5)
                else:
                    self.assertAlmostEqual(np.mean(c()), 0.0 + 1j*0.0, 5)

                c = Constellation(constellation_type, num_bits_per_symbol, points=points, normalize=False, center=True)
                if constellation_type == "custom":
                    self.assertNotAlmostEqual(np.mean(np.abs(c())**2), 1.0, 5)
                else:
                    self.assertAlmostEqual(np.mean(np.abs(c())**2), 1.0, 5)
                self.assertAlmostEqual(np.mean(c()), 0.0 + 1j*0.0, 5)

                c = Constellation(constellation_type, num_bits_per_symbol, points=points, normalize=True, center=True)
                self.assertAlmostEqual(np.mean(np.abs(c())**2), 1.0, 5)
                self.assertAlmostEqual(np.mean(c()), 0.0 + 1j*0.0, 5)


class TestConstellationDTypes(unittest.TestCase):

    def test_qam(self):
        "Test QAM constellation with all possible configurations"
        for num_bits_per_symbol in [2,8]:
            for normalize in [True, False]:
                for center in [True, False]:
                    for precision in ["single", "double"]:
                        c = Constellation("qam",
                                            num_bits_per_symbol,
                                            normalize=normalize,
                                            center=center,
                                            precision=precision)
                        self.assertEqual(c().dtype, dtypes[precision]["tf"]["cdtype"])

    def test_pam(self):
        "Test PAM constellation with all possible configurations"
        for num_bits_per_symbol in [1,2,8]:
            for normalize in [True, False]:
                for center in [True, False]:
                    for precision in ["single", "double"]:
                        c = Constellation("pam",
                                        num_bits_per_symbol,
                                        normalize=normalize,
                                        center=center,
                                        precision=precision)
                        self.assertEqual(c().dtype, dtypes[precision]["tf"]["cdtype"])


    def test_custom_with_initial_value(self):
        "Test custom constellation with initial_value with all possible configurations"
        for num_bits_per_symbol in [1,4,7]:
            for normalize in [True, False]:
                for center in [True, False]:
                    for precision in ["single", "double"]:
                        points = config.np_rng.normal(size=[2**num_bits_per_symbol]) + \
                                 1j*config.np_rng.normal(size=[2**num_bits_per_symbol])
                        c = Constellation("custom",
                                          num_bits_per_symbol,
                                          points=points,
                                          normalize=normalize,
                                          center=center,
                                          precision=precision)
                        self.assertEqual(c().dtype, dtypes[precision]["tf"]["cdtype"])
                        if center:
                            self.assertAlmostEqual(np.mean(c()), 0, 5)
                        if normalize:
                            self.assertAlmostEqual(np.mean(np.abs(c())**2), 1, 5)
