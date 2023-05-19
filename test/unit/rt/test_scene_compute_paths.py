#
# SPDX-FileCopyrightText: Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

try:
    import sionna
except ImportError as e:
    import sys
    sys.path.append("..")
    import sionna

import pytest
import unittest
import numpy as np
import tensorflow as tf
import itertools

gpus = tf.config.list_physical_devices('GPU')
print('Number of GPUs available :', len(gpus))
if gpus:
    gpu_num = 0
    try:
        tf.config.set_visible_devices(gpus[gpu_num], 'GPU')
        print('Only GPU number', gpu_num, 'used.')
        tf.config.experimental.set_memory_growth(gpus[gpu_num], True)
    except RuntimeError as e:
        print(e)

from sionna.rt import load_scene, Transmitter, Receiver, PlanarArray
from utils import *

class TestSingleReflectionWithoutLoS(unittest.TestCase):
    def test_single_reflection_with_obstructed_los(self):
        """This is a corner-case test as the reflection
           occurs at the boundary between two triangles.
           This leads to one invalid path. 
        """
        scene = load_scene(sionna.rt.scene.floor_wall)
        scene.tx_array = PlanarArray(1, 1, 0.5, 0.5, pattern="iso", polarization="V")
        scene.rx_array = PlanarArray(1, 1, 0.5, 0.5, pattern="iso", polarization="V")
        tx = Transmitter("tx", [-1, 0, 2], [0, 0, 0])
        rx = Receiver("rx", [1, 0, 2], [0, 0, 0])
        scene.add(tx)
        scene.add(rx)
        paths = scene.compute_paths(max_depth=1, method="exhaustive")

        num_paths = paths.objects.shape[-1]
        for i in range(len(np.squeeze(paths.mat_t))):
            if np.squeeze(paths.mask)[i]: # Check that paths is valid
                    a, b = validate_path(i, paths, scene)
                    self.assertTrue(np.allclose(a,b))

    def test_single_reflection_with_colocated_tx_and_rx(self):
        """Transmitter and receiver are colocated aboved a floor plane.
           leading to a single reflected path with normal incidence. 
        """
        scene = load_scene(sionna.rt.scene.floor_wall)
        scene.tx_array = PlanarArray(1, 1, 0.5, 0.5, pattern="iso", polarization="V")
        scene.rx_array = PlanarArray(1, 1, 0.5, 0.5, pattern="iso", polarization="V")
        tx = Transmitter("tx", [-1, 1, 3], [0, 0, 0])
        rx = Receiver("rx", [-1, 1, 3], [0, 0, 0])
        scene.add(tx)
        scene.add(rx)
        paths = scene.compute_paths(max_depth=1)
        num_paths = paths.objects.shape[-1]
        for i in range(num_paths):
            a, b = validate_path(i, paths, scene)
            self.assertTrue(np.allclose(a,b))

    def test_multiple_reflections(self):
        """Test transfer matrices of paths in a larger scene"""
        dtype = tf.complex128
        scene = load_scene(sionna.rt.scene.simple_street_canyon, dtype=dtype)
        scene.tx_array = PlanarArray(1, 1, 0.5, 0.5, pattern="iso", polarization="V", dtype=dtype)
        scene.rx_array = PlanarArray(1, 1, 0.5, 0.5, pattern="iso", polarization="V", dtype=dtype)
        tx = Transmitter("tx", [-10, 1, 4], [0, 0, 0], dtype=dtype)
        rx = Receiver("rx", [-2, 1.5, 4], [0, 0, 0], dtype=dtype)
        scene.add(tx)
        scene.add(rx)
        paths = scene.compute_paths(max_depth=5, method="stochastic")
        num_paths = paths.objects.shape[-1]
        for i in range(num_paths):
            a, b = validate_path(i, paths, scene)
            self.assertTrue(np.allclose(a,b))

    def test_masked_paths(self):
        """Test that mat_t of masked paths are equal to zero"""
        scene = load_scene(sionna.rt.scene.floor_wall)
        scene.synthetic_array = False
        scene.tx_array = PlanarArray(2, 2, 0.5, 0.5, pattern="iso", polarization="VH")
        scene.rx_array = PlanarArray(1, 2, 0.5, 0.5, pattern="iso", polarization="VH")
        tx_1 = Transmitter("tx_1", [-1.3, -0.3, 2], [0, 0, 0])
        tx_2 = Transmitter("tx_2", [1.5, 1, 2.5], [0, 0, 0])
        rx_1 = Receiver("rx-1", [0.7, 0.3, 1.3], [0, 0, 0])
        rx_2 = Receiver("rx-2", [-0.5, -0.7, 1.6], [0, 0, 0])
        scene.add(tx_1)
        scene.add(tx_2)
        scene.add(rx_1)
        scene.add(rx_2)
        paths = scene.compute_paths(max_depth=3)
        num_tx = len(list(scene.transmitters.values()))
        num_rx = len(list(scene.receivers.values()))
        num_tx_ant = scene.tx_array.array_size
        num_rx_ant = scene.rx_array.array_size
        for tx in range(num_tx):
            for tx_ant in range(num_tx_ant):
                source = tx*num_tx_ant + tx_ant
                for rx in range(num_rx):
                    for rx_ant in range(num_rx_ant):
                        target = rx*num_rx_ant + rx_ant
                        for path in range(0, paths.mask.shape[-1]):
                            if not paths.mask[target, source, path]:
                                mat_t = paths.mat_t[0, rx, rx_ant, tx, tx_ant, path]
                                self.assertTrue(np.array_equal(mat_t, np.zeros_like(mat_t)))
