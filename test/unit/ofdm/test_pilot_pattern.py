#
# SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
try:
    import sionna
except ImportError as e:
    import sys
    sys.path.append("../")
import unittest
import pytest
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
from sionna.ofdm import ResourceGrid, PilotPattern, EmptyPilotPattern, KroneckerPilotPattern

class TestPilotPattern(unittest.TestCase):
    """Unittest for the PilotPattern Class"""
    def test_check_settings(self):
        with self.assertRaises(AssertionError):
            # mask does not have rank 4
            mask = np.zeros([1,10], bool)
            pilots = np.zeros([1,10, 20], np.complex64)
            PilotPattern(mask, pilots)
        with self.assertRaises(AssertionError):
            # pilots does not have rank 3
            mask = np.zeros([4,2,10,46], bool)
            pilots = np.zeros([1,10,20,2], np.complex64)
            PilotPattern(mask, pilots)
        with self.assertRaises(AssertionError):
            # pilots and mask haves different first two dimensions
            mask = np.zeros([1,2,14,64], bool)
            mask[0,0,0,:] = True
            mask[0,1,1,:] = True
            num_pilots = np.max(np.sum(mask, (-2,-1)))
            pilots = np.zeros([1,3,num_pilots], np.complex64)
            PilotPattern(mask, pilots)
        with self.assertRaises(AssertionError):
            # mask has a different number of Trues
            mask = np.zeros([1,2,14,64], bool)
            mask[0,0,0,:] = True
            mask[0,1,1:3,:] = True
            num_pilots = np.max(np.sum(mask, (-2,-1)))
            pilots = np.zeros([1,2,num_pilots], np.complex64)
            PilotPattern(mask, pilots)
        with self.assertRaises(AssertionError):
            # pilots has a the wrong last dimension
            mask = np.zeros([1,2,14,64], bool)
            mask[0,0,0,:] = True
            mask[0,1,1,:] = True
            num_pilots = np.max(np.sum(mask, (-2,-1)))
            pilots = np.zeros([1,2,num_pilots+1], np.complex64)
            PilotPattern(mask, pilots)

    def test_properties(self):
        mask = np.zeros([1,2,14,64], bool)
        mask[0,0,0,:] = True
        mask[0,1,1,:] = True
        num_pilots = np.max(np.sum(mask, (-2,-1)))
        pilots = np.zeros([1,2,num_pilots], np.complex64)
        pp = PilotPattern(mask, pilots)
        self.assertEqual(pp.num_pilot_symbols, 64)
        self.assertEqual(pp.num_data_symbols, 13*64)

        mask = np.zeros([1,2,14,64], bool)
        mask[0,0,:2,:] = True
        mask[0,1,1:3,:] = True
        num_pilots = np.max(np.sum(mask, (-2,-1)))
        pilots = np.zeros([1,2,num_pilots], np.complex64)
        pp = PilotPattern(mask, pilots)
        self.assertEqual(pp.num_pilot_symbols, 128)
        self.assertEqual(pp.num_data_symbols, 12*64)

    def test_trainable_pilots(self):
        mask = np.zeros([1,2,14,64], bool)
        mask[0,0,0,:] = True
        mask[0,1,1,:] = True
        num_pilots = np.max(np.sum(mask, (-2,-1)))
        pilots = np.zeros([1,2,num_pilots], np.complex64)
        pp = PilotPattern(mask, pilots, trainable=True)
        self.assertTrue(pp.pilots.trainable)

    def test_normalized_pilots(self):
        mask = np.zeros([1,2,14,64], bool)
        mask[0,0,0,:] = True
        mask[0,1,1,:] = True
        num_pilots = np.max(np.sum(mask, (-2,-1)))
        pilots = 3*np.ones([1,2,num_pilots], np.complex64)
        pp = PilotPattern(mask, pilots, normalize=True)
        self.assertTrue(np.allclose(np.mean(np.abs(pp.pilots)**2, -1), 1.0))

class TestEmptyPilotPattern(unittest.TestCase):
    """Unittest for the EmptyPilotPattern Class"""

    def test_properties(self):
        pp = EmptyPilotPattern(4,2,14,55)
        self.assertEqual(pp.num_pilot_symbols, 0)
        self.assertEqual(pp.num_data_symbols,14*55)
