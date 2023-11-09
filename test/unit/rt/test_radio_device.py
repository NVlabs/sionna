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

import unittest
import numpy as np
import tensorflow as tf

from sionna.rt import rotation_matrix, r_hat, theta_phi_from_unit_vec, \
                      load_scene, PlanarArray, Transmitter, Receiver, compute_gain
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
        tx = Transmitter("tx", [1,2,-3], [0, 0, 0], trainable_orientation=True, trainable_position=True)
        scene.add(tx)

        with tf.GradientTape() as tape:
            a = tx.position
            b = tx.orientation
        self.assertTrue(len(tape.watched_variables())==2)

        tx.trainable_orientation = False
        with tf.GradientTape() as tape:
            a = tx.position
            b = tx.orientation
        self.assertTrue(len(tape.watched_variables())==1)

        tx.trainable_position = False
        with tf.GradientTape() as tape:
            a = tx.position
            b = tx.orientation
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

        self.assertTrue(list(scene.transmitters.values())[0].color==(40, 128, 184))
        self.assertTrue(list(scene.receivers.values())[0].color==(39, 173, 95))

    def test_custom_coloring(self):
        """Test custom coloring of radio devices"""
        scene = load_scene()
        tx = Transmitter("tx", [1,2,-3], [0, 0, 0], color=(204, 0, 0))
        rx = Receiver("rx", [0,0,0], [0, 0, 0], color=(255, 255, 0))
        scene.add(tx)
        scene.add(rx)
        
        self.assertTrue(list(scene.transmitters.values())[0].color==(204, 0, 0))
        self.assertTrue(list(scene.receivers.values())[0].color==(255, 255, 0))