#
# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

try:
    import sionna
except ImportError as e:
    import sys
    sys.path.append("../")

from sionna.channel import exp_corr_mat, one_ring_corr_mat

import pytest
import unittest
import warnings
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


def exp_corr_mat_numpy(a, n, dtype=tf.complex64):
    R = np.eye(n, dtype=np.complex64)
    for i in range(1, n):
        for j in range(i+1):
            R[i,j] = a**(i-j)
            R[j,i] = np.conj(a)**(i-j)
    return tf.cast(R, dtype)

def one_ring_corr_numpy(phi_deg, n, d_h=0.5, sigma_phi_deg=15):
    R = np.zeros([n, n], dtype=np.complex128)
    c = 2*np.pi*d_h
    phi = phi_deg/180*np.pi
    sigma_phi = sigma_phi_deg/180*np.pi    
    for l in range(n):
        for m in range(n):
            tmp = c*(l-m) 
            a = np.exp(1j * tmp*np.sin(phi) )
            b = np.exp(-0.5 * (sigma_phi*tmp*np.cos(phi))**2 )
            R[l, m] = a*b
    return R


class TestExpCorrMat(unittest.TestCase):
    def test_single_dim(self):
        "Tests for scalar inputs"
        values = [0.0, 0.9999, 0.5+1j*0.3]
        dims = [1, 2, 4, 7, 64, 128]
        for a in values:
            for n in dims:
                for dtype in [tf.complex64, tf.complex128]:
                    R1 = exp_corr_mat_numpy(a, n, dtype)
                    R2 = exp_corr_mat(a, n, dtype)
                    err = np.max(np.abs(R1-R2))
                    self.assertAlmostEqual(err, 0, 5)

    def test_catch_abs_val_error(self):
        """Absolute value of a greater than 1 should raise error"""
        with self.assertRaises(tf.errors.InvalidArgumentError):
            exp_corr_mat(1.1+1j*0.3, 12)

    def test_multiple_dims(self):
        values = np.random.uniform(0, 1, [2, 4, 3])
        n = 11
        dtype = tf.complex128
        R2 = exp_corr_mat(values, n, dtype)
        R2 = tf.reshape(R2, [-1, n, n])
        for i, a in enumerate(np.reshape(values, [-1])):
            R1 = exp_corr_mat_numpy(a, n, dtype)
            err = np.max(np.abs(R1-R2[i]))
            self.assertAlmostEqual(err, 0)

class TestOneRingCorrMat(unittest.TestCase):
    def test_single_dim(self):
        "Tests for scalar inputs"
        phi_degs = [-180, -90, -45, -12, 0, 15, 45, 65, 90, 180, 360]
        num_ants = [1, 4, 16, 128]
        d_hs = [0,0.2, 0.5, 1, 3]
        sigma_phi_degs = [0,2,5,15]

        for phi_deg in phi_degs:
            for num_ant in num_ants:
                for d_h in d_hs:
                    for sigma_phi_deg in sigma_phi_degs:
                        R1 = one_ring_corr_numpy(phi_deg, num_ant, d_h, sigma_phi_deg)
                        R2 = one_ring_corr_mat(phi_deg, num_ant, d_h, sigma_phi_deg, dtype=tf.complex128)
                        self.assertTrue(np.allclose(R1, R2))

    def test_multiple_dims(self):
        phi_degs = np.random.uniform(-np.pi, np.pi, [2, 4, 3])
        num_ant = 32
        d_h = 0.7
        sigma_phi_deg = 10
        R2 = one_ring_corr_mat(phi_degs, num_ant, d_h, sigma_phi_deg, dtype=tf.complex128)
        R2 = tf.reshape(R2, [-1, num_ant, num_ant])
        for i, phi_deg in enumerate(np.reshape(phi_degs, [-1])):
            R1 = one_ring_corr_numpy(phi_deg, num_ant, d_h, sigma_phi_deg)
            self.assertTrue(np.allclose(R1, R2[i]))

    def test_warning_large_asd(self):
        """Should warn when sigma_phi_deg>15"""
        phi_deg = 35
        num_ant = 32
        d_h = 0.7
        sigma_phi_deg = 16
        with self.assertWarns(Warning):
            one_ring_corr_mat(phi_deg, num_ant, d_h, sigma_phi_deg)
