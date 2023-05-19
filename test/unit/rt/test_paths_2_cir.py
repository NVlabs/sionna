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

from sionna.rt import load_scene, Transmitter, Receiver, PlanarArray, Paths2CIR
from utils import *

class TestPaths2CIR(unittest.TestCase):

    def test_doppler_full_array(self):
        """Test that Doppler shifts are correctly applied to all paths for a non-synthetic array"""
        dtype = tf.complex128
        scene = load_scene(sionna.rt.scene.simple_street_canyon, dtype=dtype)
        scene.synthetic_array = False
        scene.tx_array = PlanarArray(1, 2, 0.5, 0.5, pattern="iso", polarization="VH", dtype=dtype)
        scene.rx_array = PlanarArray(2, 1, 0.5, 0.5, pattern="iso", polarization="cross", dtype=dtype)
        tx = Transmitter("tx", [50, 1, 4], [0, 0, 0], dtype=dtype)
        rx = Receiver("rx", [-50, 1.5, 4], [0, 0, 0], dtype=dtype)
        scene.add(tx)
        scene.add(rx)
        paths = scene.compute_paths(max_depth=5, method="stochastic")
        v_tx = tf.reshape(tf.constant([3.5, 0.4, 0.1], tf.float64), [1, 1, 3])
        v_rx = tf.reshape(tf.constant([-4.5, 0.4, -0.1], tf.float64), [1, 1, 3])
        bandwidth = 1e6
        num_time_steps = 100
        p2c = Paths2CIR(bandwidth, v_tx, v_rx, num_time_steps, scene, dtype=dtype)
        a_ref, tau_ref = p2c(paths.as_tuple())
        t = np.arange(0, num_time_steps)/bandwidth
        num_tx = len(list(scene.transmitters.values()))
        num_rx = len(list(scene.receivers.values()))
        tx_array_size = scene.tx_array.array_size
        rx_array_size = scene.rx_array.array_size
        num_tx_ant = scene.tx_array.num_ant
        num_rx_ant = scene.rx_array.num_ant
        tx_dual = num_tx_ant>tx_array_size 
        rx_dual = num_rx_ant>rx_array_size
        num_tx_pol = 2 if tx_dual else 1
        num_rx_pol = 2 if rx_dual else 1

        for tx in range(num_tx):
            for tx_el in range(tx_array_size):
                source = tx*tx_array_size + tx_el
                for tx_pol in range(num_tx_pol):
                    tx_ant = tx*num_tx_ant + tx_el*num_tx_pol + tx_pol
                    for rx in range(num_rx):
                        for rx_el in range(rx_array_size):
                            target = rx*rx_array_size + rx_el
                            for rx_pol in range(num_rx_pol):
                                rx_ant = rx*num_rx_ant + rx_el*num_rx_pol + rx_pol
                                for path in range(paths.mask.shape[-1]):
                                    if paths.mask[target, source, path]: 
                                        a = a_ref[0, rx, rx_ant, tx, tx_ant, path]
                                        theta_t = paths.theta_t[0, rx, rx_el, tx, tx_el, path]
                                        phi_t = paths.phi_t[0, rx, rx_el, tx, tx_el, path]
                                        theta_r = paths.theta_r[0, rx, rx_el, tx, tx_el, path]
                                        phi_r = paths.phi_r[0, rx, rx_el, tx, tx_el, path]
                                        r_hat_t = r_hat(theta_t, phi_t)
                                        r_hat_r = r_hat(theta_r, phi_r)
                                        f_t = dot(r_hat_t, np.squeeze(v_tx))/scene.wavelength.numpy()
                                        f_r = dot(r_hat_r, np.squeeze(v_rx))/scene.wavelength.numpy()
                                        a_theo = a[0]*np.exp(1j*2*np.pi*(f_t+f_r)*t)
                                        self.assertTrue(np.allclose(a, a_theo, atol=1e-6))


        def test_doppler_synthetic_array(self):
            """Test that Doppler shifts are correctly applied to all paths for a synthetic array"""
            dtype = tf.complex128
            scene = load_scene(sionna.rt.scene.simple_street_canyon, dtype=dtype)
            scene.tx_array = PlanarArray(1, 2, 0.5, 0.5, pattern="iso", polarization="VH", dtype=dtype)
            scene.rx_array = PlanarArray(2, 1, 0.5, 0.5, pattern="iso", polarization="cross", dtype=dtype)
            tx = Transmitter("tx", [50, 1, 4], [0, 0, 0], dtype=dtype)
            rx = Receiver("rx", [-50, 1.5, 4], [0, 0, 0], dtype=dtype)
            scene.add(tx)
            scene.add(rx)
            scene.synthetic_array = True
            paths = scene.compute_paths(max_depth=5, method="stochastic")
            v_tx = tf.reshape(tf.constant([3.5, 0.4, 0.1], tf.float64), [1, 1, 3])
            v_rx = tf.reshape(tf.constant([-4.5, 0.4, -0.1], tf.float64), [1, 1, 3])
            bandwidth = 1e6
            num_time_steps = 100
            p2c = Paths2CIR(bandwidth, v_tx, v_rx, num_time_steps, scene, dtype=dtype)
            a_ref, tau_ref = p2c(paths.as_tuple())
            t = np.arange(0, num_time_steps)/bandwidth
            for tx in range(num_tx):
                for tx_el in range(tx_array_size):
                    source = tx
                    for tx_pol in range(num_tx_pol):
                        tx_ant = tx*num_tx_ant + tx_el*num_tx_pol + tx_pol
                        for rx in range(num_rx):
                            for rx_el in range(rx_array_size):
                                target = rx
                                for rx_pol in range(num_rx_pol):
                                    rx_ant = rx*num_rx_ant + rx_el*num_rx_pol + rx_pol
                                    for path in range(paths.mask.shape[-1]):
                                        if paths.mask[target, source, path]: 
                                            a = a_ref[0, rx, rx_ant, tx, tx_ant, path]
                                            theta_t = paths.theta_t[0, rx, tx, path]
                                            phi_t = paths.phi_t[0, rx, tx, path]
                                            theta_r = paths.theta_r[0, rx, tx, path]
                                            phi_r = paths.phi_r[0, rx, tx, path]
                                            r_hat_t = r_hat(theta_t, phi_t)
                                            r_hat_r = r_hat(theta_r, phi_r)
                                            f_t = dot(r_hat_t, np.squeeze(v_tx))/scene.wavelength.numpy()
                                            f_r = dot(r_hat_r, np.squeeze(v_rx))/scene.wavelength.numpy()
                                            a_theo = a[0]*np.exp(1j*2*np.pi*(f_t+f_r)*t)
                                            self.assertTrue(np.allclose(a, a_theo, atol=1e-6))
