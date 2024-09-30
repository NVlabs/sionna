#
# SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import unittest
import numpy as np
import tensorflow as tf
from sionna.rt import *
from sionna.constants import SPEED_OF_LIGHT, PI


class TestEmptyScene(unittest.TestCase):

    def test_tau(self):
        """Verify that tau corresponds to the speed of light on the LOS paths"""
        dtype = tf.complex128
        scene = load_scene(dtype=dtype)
        scene.tx_array = PlanarArray(1, 1, 0.5, 0.5, "iso", "H", dtype=dtype)
        scene.rx_array = PlanarArray(1, 1, 0.5, 0.5, "iso", "H", dtype=dtype)
        tx = Transmitter("tx", [0,0,0], [0, 0, 0], dtype=dtype)
        scene.add(tx)
        distances = np.array([1, 10, 100, 1000, 10000, 100000])
        for d in distances:
            scene.add(Receiver(f"rx_{d}", [d,0,0], [0, 0, 0], dtype=dtype))
        paths = scene.compute_paths()
        paths.normalize_delays = False
        tau = tf.squeeze(paths.tau)
        tau_theo = distances/SPEED_OF_LIGHT
        self.assertTrue(np.array_equal(tau, tau_theo))

    def test_frijs_equation(self):
        """Test that Frijs equation holds in free space"""
        dtype=tf.complex128
        for pattern in ["hw_dipole", "tr38901"]:
            scene = load_scene(dtype=dtype)
            scene.tx_array = PlanarArray(1, 1, 0.5, 0.5, pattern, "V", dtype=dtype)
            scene.rx_array = PlanarArray(1, 1, 0.5, 0.5, pattern, "V", dtype=dtype)
            _, g_tx, _ = compute_gain(scene.tx_array.antenna.patterns[0], dtype=dtype)
            _, g_rx, _ = compute_gain(scene.rx_array.antenna.patterns[0], dtype=dtype)
            tx = Transmitter("tx", [10,-10, 4], [0, 0, 0], dtype=dtype)
            rx = Receiver("rx", [10,0,0], [0, 0, 0], dtype=dtype)
            scene.add(tx)
            scene.add(rx)
            r = 200.
            thetas = tf.cast(tf.linspace(0., PI, 10), dtype.real_dtype)
            phis = tf.cast(tf.linspace(0., 2*PI, 10), dtype.real_dtype)
            for theta in thetas:
                for phi in phis:
                    rx.position = r_hat(theta, phi)*r + tx.position
                    rx.look_at("tx")
                    tx.look_at("rx")
                    paths = scene.compute_paths(method="exhaustive",
                                                scattering=False)
                    a = paths.a
                    sim = 10*np.log10(np.abs(np.squeeze(a))**2)
                    theo = 10*np.log10((scene.wavelength/4/PI/r)**2*g_tx*g_rx)
                    self.assertTrue(np.abs(sim-theo)< 1e-3)

    def test_measure_antenna_pattern(self):
        """Measure TX antenna pattern with an isotropic receive antenna"""
        dtype=tf.complex128
        scene = load_scene(dtype=dtype)
        scene.tx_array = PlanarArray(1, 1, 0.5, 0.5, "tr38901", "V", dtype=dtype)
        scene.rx_array = PlanarArray(1, 1, 0.5, 0.5, "iso", "V", dtype=dtype)
        _, g_tx, _ = compute_gain(scene.tx_array.antenna.patterns[0], dtype=dtype)
        _, g_rx, _ = compute_gain(scene.rx_array.antenna.patterns[0], dtype=dtype)
        tx = Transmitter("tx", [-2,3,5], [0, 0, 0], dtype=dtype)
        scene.add(tx)

        # Add receivers on a sphere around the transmitter
        r = tf.cast(100, dtype.real_dtype)
        thetas = tf.linspace(0.01, np.pi-0.01, 10)
        phis = tf.linspace(0., 2*np.pi, 10)
        i = 0
        for theta in thetas:
            for phi in phis:
                theta = tf.cast(theta, dtype.real_dtype)
                phi = tf.cast(phi, dtype.real_dtype)

                # Add receiver
                rx = Receiver(f"rx_{i}", [0,0,0], [0, 0, 0], dtype=dtype)
                rx.position = r_hat(theta, phi)*r + tx.position
                scene.add(rx)
                i += 1

        # Compute paths
        paths = scene.compute_paths()

        # Compute normalized channel gain
        a = tf.squeeze(paths.a).numpy()
        a /= tf.cast(scene.wavelength/4/PI/r, a.dtype)

        # Compute transmitted pattern
        thetas, phis = np.meshgrid(thetas, phis, indexing='ij')
        thetas = tf.cast(thetas, tf.float32)
        phis = tf.cast(phis, tf.float32)
        c_t_theta, _ = scene.tx_array.antenna.patterns[0](thetas, phis)

        # Check that measured and transmitted patterns are the same
        self.assertTrue(np.max(np.abs(tf.reshape(c_t_theta, [-1])-a)), 1e-12)

    def test_colocated_tx_rx(self):
        """Test that a colocated transmitter and receiver do not have LOS path"""
        scene = load_scene()
        scene.tx_array = PlanarArray(1, 1, 0.5, 0.5, pattern="iso", polarization="V")
        scene.rx_array = PlanarArray(1, 1, 0.5, 0.5, pattern="iso", polarization="V")
        scene.add(Transmitter("tx", [0,0,0], [0, 0, 0]))
        scene.add(Receiver("rx", [0,0,0], [0, 0, 0]))
        paths = scene.compute_paths()
        self.assertTrue(paths.tau.numpy().size == 0)
