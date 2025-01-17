#
# SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import unittest
import numpy as np
import tensorflow as tf
from sionna.rt import rotation_matrix, r_hat, theta_phi_from_unit_vec, \
                      load_scene, PlanarArray, Transmitter, Receiver, compute_gain
from sionna.constants import PI


def theta_prime_phi_prime(angles, theta, phi):
    rot_mat = rotation_matrix(angles)
    v = r_hat(theta, phi)
    v_prime = tf.linalg.matvec(rot_mat, v, transpose_a=True)
    return theta_phi_from_unit_vec(v_prime)


class TestRadioDevice(unittest.TestCase):

    def test_add_remove(self):
        """Test adding and removing of radio devices"""
        scene = load_scene()
        tx = Transmitter("tx", [1,2,-3], [0, 0, 0])
        rx = Receiver("rx", [0,0,0], [0, 0, 0])
        scene.add(tx)
        scene.add(rx)
        self.assertTrue(len(scene.transmitters)==1)
        self.assertTrue(len(scene.receivers)==1)

        scene.remove("tx")
        scene.remove("rx")
        self.assertTrue(len(scene.transmitters)==0)
        self.assertTrue(len(scene.receivers)==0)

    def test_trainable_properties(self):
        """Test that trainable properties are correctly identified by the gradient tape"""
        scene = load_scene()
        tx = Transmitter("tx", [1,2,-3], [0, 0, 0])
        tx.position = tf.Variable([1,2,-3], dtype=tf.float32, trainable=True)
        tx.orientation = tf.Variable([1,2,-3], dtype=tf.float32, trainable=True)
        scene.add(tx)

        with tf.GradientTape() as tape:
            a = tx.position
            b = tx.orientation
            c = b+a
        self.assertTrue(len(tape.watched_variables())==2)

        tx.orientation = tf.zeros([3], dtype=tf.float32)
        with tf.GradientTape() as tape:
            a = tx.position
            b = tx.orientation
            c = b+a
        self.assertTrue(len(tape.watched_variables())==1)

        tx.position = tf.zeros([3], dtype=tf.float32)
        with tf.GradientTape() as tape:
            a = tx.position
            b = tx.orientation
            c = b+a
        self.assertTrue(len(tape.watched_variables())==0)


    def test_look_at(self):
        """Test that we obtain perfectly aligned readio devices when using the look_at function"""
        scene = load_scene()
        scene.tx_array = PlanarArray(1, 1, 0.5, 0.5, "tr38901", "H")
        scene.rx_array = PlanarArray(1, 1, 0.5, 0.5, "dipole", "H")
        _, g_tx, _ = compute_gain(scene.tx_array.antenna.patterns[0])
        _, g_rx, _ = compute_gain(scene.rx_array.antenna.patterns[0])
        tx = Transmitter("tx", [1,2,-3], [0, 0, 0])
        rx = Receiver("rx", [0,0,0], [0, 0, 0])
        scene.add(tx)
        scene.add(rx)

        r = 100
        thetas = tf.linspace(0., PI, 10)
        phis = tf.linspace(0., 2*PI, 10)

        # Compute theoretical gain (Frijs equation)
        a_theo_db = 10*np.log10((scene.wavelength/4/np.pi/r)**2*g_tx*g_rx)

        for theta in thetas:
            for phi in phis:

                # Move receiver to the next position on the sphere around the transmitter
                rx.position = r_hat(theta, phi)*r + tx.position

                # Make transmitter and receiver look at each other
                rx.look_at("tx")
                tx.look_at("rx")

                # Compute paths
                paths = scene.compute_paths()
                theta_t = tf.squeeze(paths.theta_t)
                phi_t = tf.squeeze(paths.phi_t)
                theta_r = tf.squeeze(paths.theta_r)
                phi_r = tf.squeeze(paths.phi_r)

                # Compute AODs and AoAs in LCS of the transmitter and receiver
                theta_prime_t, phi_prime_t = theta_prime_phi_prime(tx.orientation, theta_t, phi_t)
                theta_prime_r, phi_prime_r = theta_prime_phi_prime(rx.orientation, theta_r, phi_r)

                # Check that the LCS angles are always constant
                self.assertTrue(np.abs(theta_prime_t - PI/2)<1e-5)
                self.assertTrue(np.abs(theta_prime_r - PI/2)<1e-5)
                self.assertTrue(np.abs(phi_prime_t)<1e-5)
                self.assertTrue(np.abs(phi_prime_r)<1e-5)

                # Compute channel impulse response and make
                # sure that it matches the theoretical
                a = tf.squeeze(paths.a)
                a_db = 20*np.log10(np.abs(a.numpy()))
                self.assertTrue(np.abs(a_db-a_theo_db)< 1e-4)

    def test_default_coloring(self):
        """Test default coloring of radio devices"""
        scene = load_scene()
        tx = Transmitter("tx", [1,2,-3], [0, 0, 0])
        rx = Receiver("rx", [0,0,0], [0, 0, 0])
        scene.add(tx)
        scene.add(rx)

        color_tx = tf.cast((0.160, 0.502, 0.725), tf.float32)
        color_rx = tf.cast((0.153, 0.682, 0.375), tf.float32)
        self.assertTrue(tf.reduce_all(list(scene.transmitters.values())[0].color==color_tx))
        self.assertTrue(tf.reduce_all(list(scene.receivers.values())[0].color==color_rx))

    def test_custom_coloring(self):
        """Test custom coloring of radio devices"""
        color_tx = tf.cast((0.8, 0., 0.), tf.float32)
        color_rx = tf.cast((1., 1., 0.), tf.float32)
        scene = load_scene()
        tx = Transmitter("tx", [1,2,-3], [0, 0, 0], color=color_tx)
        rx = Receiver("rx", [0,0,0], [0, 0, 0], color=color_rx)
        scene.add(tx)
        scene.add(rx)

        self.assertTrue(tf.reduce_all(list(scene.transmitters.values())[0].color==color_tx))
        self.assertTrue(tf.reduce_all(list(scene.receivers.values())[0].color==color_rx))
