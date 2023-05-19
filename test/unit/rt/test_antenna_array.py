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

from sionna.rt import *
from sionna.constants import PI

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

class TestAntennaArray(unittest.TestCase):

    def test_polarization_01(self):
        """Perfectly aligned, and co-polarized"""
        dtype = tf.complex128
        distance = 100
        for pattern1 in ["iso", "dipole", "hw_dipole", "tr38901"]:
            for pattern2 in ["iso", "dipole", "hw_dipole", "tr38901"]:
                for polarization in ["V", "H"]:
                    scene = load_scene(dtype=dtype)
                    scene.tx_array = PlanarArray(1, 1, 0.5, 0.5, pattern=pattern1, polarization=polarization, dtype=dtype)
                    scene.rx_array = PlanarArray(1, 1, 0.5, 0.5, pattern=pattern2, polarization=polarization, dtype=dtype)
                    _, g_tx, _ = compute_gain(scene.tx_array.antenna.patterns[0], dtype)
                    _, g_rx, _ = compute_gain(scene.rx_array.antenna.patterns[0], dtype)
                    tx = Transmitter("tx", [0,0,0], [0, 0, 0], dtype=dtype)
                    rx = Receiver("rx", [distance, 0,0], [PI, 0, 0], dtype=dtype)
                    scene.add(tx)
                    scene.add(rx)
                    p2c = Paths2CIR(1, scene=scene, dtype=dtype)
                    paths = scene.compute_paths()
                    a, _  = p2c(paths.as_tuple())
                    a = tf.squeeze(a)
                    a /= tf.cast(scene.wavelength/4/PI/distance*tf.sqrt(g_tx*g_rx), a.dtype)
                    self.assertTrue(np.abs(np.abs(a)-1)<1e-5)

    def test_polarization_02(self):
        """Orthogonal polarization"""
        dtype = tf.complex128
        distance = 100
        for pattern1 in ["iso", "dipole", "hw_dipole", "tr38901"]:
            for pattern2 in ["iso", "dipole", "hw_dipole", "tr38901"]:
                scene = load_scene(dtype=dtype)
                scene.tx_array = PlanarArray(1, 1, 0.5, 0.5, pattern=pattern1, polarization="V", dtype=dtype)
                scene.rx_array = PlanarArray(1, 1, 0.5, 0.5, pattern=pattern2, polarization="H", dtype=dtype)
                tx = Transmitter("tx", [0,0,0], [0, 0, 0], dtype=dtype)
                rx = Receiver("rx", [distance, 0,0], [PI, 0, 0], dtype=dtype)
                scene.add(tx)
                scene.add(rx)
                p2c = Paths2CIR(1, scene=scene, dtype=dtype)
                paths = scene.compute_paths()
                a, _  = p2c(paths.as_tuple())
                a = tf.squeeze(a)
                a /= tf.cast(scene.wavelength/4/PI/distance, a.dtype)
                self.assertTrue(np.abs(a)<1e-6)

    def test_rotation_polarization(self):
        """Test that we can create a polarized antenna by rotation"""
        dtype = tf.complex128
        distance = 100
        pattern = dipole_pattern
        slant_angle = 0.3
        for slant_angle in tf.linspace(-PI/2, PI/2, 10):
            pattern1 = lambda theta, phi: pattern(theta, phi, slant_angle=slant_angle, dtype=dtype)
            pattern2 = lambda theta, phi: pattern(theta, phi, slant_angle=0., dtype=dtype)
            scene = load_scene(dtype=dtype)
            scene.tx_array = PlanarArray(1, 1, 0.5, 0.5, pattern=[pattern1], dtype=dtype)
            scene.rx_array = PlanarArray(1, 1, 0.5, 0.5, pattern=[pattern2], dtype=dtype)
            _, g_tx, _ = compute_gain(scene.tx_array.antenna.patterns[0], dtype)
            _, g_rx, _ = compute_gain(scene.rx_array.antenna.patterns[0], dtype)
            tx = Transmitter("tx", [0,0,0], [0, 0, 0], dtype=dtype)
            rx = Receiver("rx", [distance, 0,0], [PI, 0, -slant_angle], dtype=dtype)
            scene.add(tx)
            scene.add(rx)
            p2c = Paths2CIR(1, scene=scene, dtype=dtype)
            paths = scene.compute_paths()
            a, _  = p2c(paths.as_tuple())
            a = tf.squeeze(a)
            a /= tf.cast(scene.wavelength/4/PI/distance*tf.sqrt(g_tx*g_rx), a.dtype)
            self.assertTrue(np.abs(a-1)<1e-6)

    def test_dual_polarization_01(self):
        """Test that dual-polarized antennas have the correct behaviour"""
        dtype = tf.complex128
        distance = 100
        scene = load_scene(dtype=dtype)
        scene.tx_array = PlanarArray(1, 1, 0.5, 0.5, pattern="dipole", polarization="VH", dtype=dtype)
        scene.rx_array = PlanarArray(1, 1, 0.5, 0.5, pattern="dipole", polarization="VH", dtype=dtype)
        _, g_tx, _ = compute_gain(scene.tx_array.antenna.patterns[0], dtype)
        _, g_rx, _ = compute_gain(scene.rx_array.antenna.patterns[0], dtype)
        tx = Transmitter("tx", [0,0,0], [PI, 0, 0], dtype=dtype)
        rx = Receiver("rx", [distance, 0,0], [0, 0, 0], dtype=dtype)
        scene.add(tx)
        scene.add(rx)
        p2c = Paths2CIR(1, scene=scene, dtype=dtype)
        paths = scene.compute_paths()
        a, _  = p2c(paths.as_tuple())
        a = tf.squeeze(a)
        a /= tf.cast(scene.wavelength/4/PI/distance*tf.sqrt(g_tx*g_rx), a.dtype)
        self.assertTrue(np.abs(a[0,0]-1)<1e-6)
        self.assertTrue(np.abs(a[1,1]+1)<1e-6)
        self.assertTrue(np.abs(a[0,1])<1e-6)
        self.assertTrue(np.abs(a[1,0])<1e-6)

    def test_dual_polarization_02(self):
        """Test that dual-polarized antennas have the correct behaviour"""
        dtype = tf.complex128
        distance = 100
        scene = load_scene(dtype=dtype)
        scene.tx_array = PlanarArray(1, 1, 0.5, 0.5, pattern="dipole", polarization="cross", dtype=dtype)
        scene.rx_array = PlanarArray(1, 1, 0.5, 0.5, pattern="dipole", polarization="cross", dtype=dtype)
        _, g_tx, _ = compute_gain(scene.tx_array.antenna.patterns[0], dtype)
        _, g_rx, _ = compute_gain(scene.rx_array.antenna.patterns[0], dtype)
        tx = Transmitter("tx", [0,0,0], [PI, 0, 0], dtype=dtype)
        rx = Receiver("rx", [distance, 0,0], [0, 0, 0], dtype=dtype)
        scene.add(tx)
        scene.add(rx)
        p2c = Paths2CIR(1, scene=scene, dtype=dtype)
        paths = scene.compute_paths()
        a, _  = p2c(paths.as_tuple())
        a = tf.squeeze(a)
        a /= tf.cast(scene.wavelength/4/PI/distance*tf.sqrt(g_tx*g_rx), a.dtype)
        self.assertTrue(np.abs(a[0,0])<1e-6)
        self.assertTrue(np.abs(a[1,1])<1e-6)
        self.assertTrue(np.abs(a[0,1]-1)<1e-6)
        self.assertTrue(np.abs(a[1,0]-1)<1e-6)

    def test_synthetic_vs_real_array_01(self):
        """Test that synthetic and real arrays have similar channel impulse responses"""
        dtype=tf.complex128
        scene = load_scene(dtype=dtype)
        scene.tx_array = PlanarArray(3, 3, 0.5, 0.5, "tr38901", "VH", dtype=dtype)
        scene.rx_array = PlanarArray(3, 3, 0.5, 0.5, "dipole", "cross", dtype=dtype)
        tx = Transmitter("tx", [0,0,25], [0, 0, 0], dtype=dtype)
        rx = Receiver("rx", [100,50,1.5], [PI, 0, 0], dtype=dtype)
        scene.add(tx)
        scene.add(rx)
        distance = normalize(tx.position-rx.position)[1]

        scene.synthetic_array = True
        paths = scene.compute_paths()
        p2c = Paths2CIR(1, scene=scene, dtype=dtype, normalize_delays=False)
        a_syn, tau_syn = [tf.squeeze(t) for t in p2c(paths.as_tuple())]

        scene.synthetic_array = False
        paths = scene.compute_paths()
        p2c = Paths2CIR(1, scene=scene, dtype=dtype, normalize_delays=False)
        a, tau = [tf.squeeze(t) for t in p2c(paths.as_tuple())]

        # Check that the delay of the antenna in the array center coincides with the delay for the synthetic array
        assert tau_syn==tau[4][4]

        # Add phase shifts realted to differences with the syn thetic array to the gains
        a_comp = a*tf.exp(tf.complex(tf.cast(0, dtype.real_dtype), (tau_syn-tau)*2*PI*scene.frequency))

        # Test that antenna gains are not oo different
        assert np.max(np.abs(np.imag(a_syn)-np.imag(a_comp)))<1e-2
        assert np.mean(np.abs(a_comp-a_syn)**2) / np.mean(np.abs(a_syn)**2) <1e-4

    def test_synthetic_vs_real_array_02(self):
        """Test that synthetic and real arrays have similar channel impulse responses"""
        dtype=tf.complex128
        scene = load_scene(dtype=dtype)
        scene.tx_array = PlanarArray(3, 3, 0.5, 0.5, "tr38901", "VH", dtype=dtype)
        scene.rx_array = PlanarArray(3, 3, 0.5, 0.5, "dipole", "cross", dtype=dtype)
        tx = Transmitter("tx", [0,0,0], [0, 0, 0], dtype=dtype)
        rx = Receiver("rx", [1000,0,0], [PI, 0, 0], dtype=dtype)
        scene.add(tx)
        scene.add(rx)
        distance = normalize(tx.position-rx.position)[1]

        scene.synthetic_array = True
        paths = scene.compute_paths()
        p2c = Paths2CIR(1, scene=scene, dtype=dtype, normalize_delays=False)
        a_syn, tau_syn = [tf.squeeze(t) for t in p2c(paths.as_tuple())]

        scene.synthetic_array = False
        paths = scene.compute_paths()
        p2c = Paths2CIR(1, scene=scene, dtype=dtype, normalize_delays=False)
        a, tau = [tf.squeeze(t) for t in p2c(paths.as_tuple())]

        # Check that the delay of the antenna in the array center coincides with the delay for the synthetic array
        assert tau_syn==tau[4][4]

        # Add phase shifts realted to differences with the syn thetic array to the gains
        a_comp = a*tf.exp(tf.complex(tf.cast(0, dtype.real_dtype), (tau_syn-tau)*2*PI*scene.frequency))

        # Test that antenna gains are not oo different
        assert np.max(np.abs(np.imag(a_syn)-np.imag(a_comp)))<1e-2
        assert np.mean(np.abs(a_comp-a_syn)**2) / np.mean(np.abs(a_syn)**2) <1e-4
