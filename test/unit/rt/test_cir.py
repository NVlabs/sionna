#
# SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
import unittest
import numpy as np
import tensorflow as tf
import sionna
from sionna.rt import load_scene, Transmitter, Receiver, PlanarArray, RIS
from utils import r_hat, dot

class TestCIR(unittest.TestCase):

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
        paths = scene.compute_paths(max_depth=5, method="fibonacci", scattering=False)
        v_tx = tf.reshape(tf.constant([3.5, 0.4, 0.1], tf.float64), [1, 1, 3])
        v_rx = tf.reshape(tf.constant([-4.5, 0.4, -0.1], tf.float64), [1, 1, 3])
        bandwidth = 1e6
        num_time_steps = 100
        paths.apply_doppler(bandwidth, num_time_steps, v_tx, v_rx)
        a_ref = paths.a
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
                                    if paths.targets_sources_mask[target, source, path]:
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
        paths = scene.compute_paths(max_depth=5, method="fibonacci", scattering=False)
        v_tx = tf.reshape(tf.constant([3.5, 0.4, 0.1], tf.float64), [1, 1, 3])
        v_rx = tf.reshape(tf.constant([-4.5, 0.4, -0.1], tf.float64), [1, 1, 3])
        bandwidth = 1e6
        num_time_steps = 100
        paths.apply_doppler(bandwidth, num_time_steps, v_tx, v_rx)
        a_ref = paths.a
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
                source = tx
                for tx_pol in range(num_tx_pol):
                    tx_ant = tx*num_tx_ant + tx_el*num_tx_pol + tx_pol
                    for rx in range(num_rx):
                        for rx_el in range(rx_array_size):
                            target = rx
                            for rx_pol in range(num_rx_pol):
                                rx_ant = rx*num_rx_ant + rx_el*num_rx_pol + rx_pol
                                for path in range(paths.mask.shape[-1]):
                                    if paths.targets_sources_mask[target, source, path]:
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

    def test_pad_or_crop_synthetic_array(self):
        """Test padding and cropping of CIR"""
        scene = load_scene(sionna.rt.scene.simple_street_canyon)
        scene.synthetic_array = True
        scene.tx_array = PlanarArray(num_rows=2,
                                    num_cols=2,
                                    vertical_spacing=0.5,
                                    horizontal_spacing=0.5,
                                    pattern="iso",
                                    polarization="V")
        scene.rx_array = PlanarArray(num_rows=2,
                                    num_cols=1,
                                    vertical_spacing=0.5,
                                    horizontal_spacing=0.5,
                                    pattern="iso",
                                    polarization="V")
        tx = Transmitter("tx",[0,0,15])
        scene.add(tx)
        tx = Transmitter("tx2",[10, 0, 15])
        scene.add(tx)
        rx = Receiver("rx",[-25,-10,1.5],[0.0,0.0,0.0])
        scene.add(rx)
        rx = Receiver("rx2",[-25,15,1.5],[0.0,0.0,0.0])
        scene.add(rx)
        paths = scene.compute_paths(max_depth=5, num_samples=10e6, diffraction=True)
        a, tau = paths.cir()
        max_num_paths = a.shape[-2]

        # Pad
        num_paths = max_num_paths+10
        a_pad, tau_pad = paths.cir(num_paths, num_paths=num_paths)
        self.assertTrue(tf.reduce_all(a_pad[...,max_num_paths:,:]==0))
        self.assertTrue(tf.reduce_all(tau_pad[...,max_num_paths:]==-1))

        # Crop
        num_paths = max_num_paths-3
        a_crop, tau_crop = paths.cir(num_paths, num_paths=num_paths)
        ind = np.flip(np.argsort(np.abs(a), axis=-2), axis=-2)[...,:num_paths,:]
        a_test = np.take_along_axis(a.numpy(), ind, axis=-2)
        ind = ind[:,:,0,:,0,:, 0]
        tau_test = np.take_along_axis(tau.numpy(), ind, axis=-1)
        self.assertTrue(np.allclose(a_test, a_crop))
        self.assertTrue(np.allclose(tau_test, tau_crop))

    def test_pad_or_crop_non_synthetic_array(self):
        """Test padding and cropping of CIR"""
        scene = load_scene(sionna.rt.scene.simple_street_canyon)
        scene.synthetic_array = False
        scene.tx_array = PlanarArray(num_rows=2,
                                    num_cols=2,
                                    vertical_spacing=0.5,
                                    horizontal_spacing=0.5,
                                    pattern="iso",
                                    polarization="V")
        scene.rx_array = PlanarArray(num_rows=2,
                                    num_cols=1,
                                    vertical_spacing=0.5,
                                    horizontal_spacing=0.5,
                                    pattern="iso",
                                    polarization="V")
        tx = Transmitter("tx",[0,0,15])
        scene.add(tx)
        tx = Transmitter("tx2",[10, 0, 15])
        scene.add(tx)
        rx = Receiver("rx",[-25,-10,1.5],[0.0,0.0,0.0])
        scene.add(rx)
        rx = Receiver("rx2",[-25,15,1.5],[0.0,0.0,0.0])
        scene.add(rx)
        paths = scene.compute_paths(max_depth=5, num_samples=10e6, diffraction=True)
        a, tau = paths.cir()
        max_num_paths = a.shape[-2]

        # Pad
        num_paths = max_num_paths+10
        a_pad, tau_pad = paths.cir(num_paths, num_paths=num_paths)
        self.assertTrue(tf.reduce_all(a_pad[...,max_num_paths:,:]==0))
        self.assertTrue(tf.reduce_all(tau_pad[...,max_num_paths:]==-1))

        # Crop
        num_paths = max_num_paths-3
        a_crop, tau_crop = paths.cir(num_paths, num_paths=num_paths)
        ind = np.flip(np.argsort(np.abs(a), axis=-2), axis=-2)[...,:num_paths,:]
        a_test = np.take_along_axis(a.numpy(), ind, axis=-2)
        ind = ind[...,0]
        tau_test = np.take_along_axis(tau.numpy(), ind, axis=-1)
        self.assertTrue(np.allclose(a_test, a_crop))
        self.assertTrue(np.allclose(tau_test, tau_crop))

    def test_ris(self):
        # Tests that shapes of impulse response match the desired results
        scene = load_scene(sionna.rt.scene.simple_street_canyon)
        scene.frequency = 3e9 
        scene.tx_array = PlanarArray(3,1,0.5,0.5,"iso", "V")
        scene.rx_array = PlanarArray(1,4,0.5,0.5,"iso", "V")
        tx1 = Transmitter("tx1", position=[-32,10,32], look_at=[0,0,0])
        scene.add(tx1)
        tx2 = Transmitter("tx2", position=[-32,10,33], look_at=[0,0,0])
        scene.add(tx2)
        rx1 = Receiver("rx1", position=[22,52,1.7])
        scene.add(rx1)
        rx2 = Receiver("rx2", position=[22,52,2.2])
        scene.add(rx2)

        ris1 = RIS(name="ris1",
                position=[32,-9,32],
                num_rows=100,
                num_cols=100,
                num_modes=2,
                look_at=(tx1.position+rx1.position)/2) # Look in between TX and RX
        scene.add(ris1)

        ris2 = RIS(name="ris2",
                position=[32,-9,33],
                num_rows=5,
                num_cols=5,
                num_modes=1,
                look_at=(tx2.position+rx2.position)/2) # Look in between TX and RX
        scene.add(ris2)

        ris1.phase_gradient_reflector([tx1.position, tx2.position],
                                     [rx1.position, rx2.position])
        ris2.phase_gradient_reflector(tx2.position, rx2.position)

        # Test with non-synthetic array
        scene.synthetic_array = False
        paths = scene.compute_paths(max_depth=1, num_samples=1e6, scattering=False,
                                    ris=True, los=False, reflection=False, diffraction=False)
        paths.apply_doppler(100e6, 7, [10,0,0], [0,10,0])
        paths.normalize_delays = False

        # Non clustering of RIS paths
        a, tau = paths.cir(cluster_ris_paths=False)
        a_target_shape = [1, 2, 4, 2, 3, 10025, 7]
        tau_target_shape = [1, 2, 4, 2, 3, 10025]
        self.assertTrue(np.array_equal(a_target_shape, a.shape))
        self.assertTrue(np.array_equal(tau_target_shape, tau.shape))

        # Clustering of RIS paths
        a, tau = paths.cir(cluster_ris_paths=True)
        a_target_shape = [1, 2, 4, 2, 3, 2, 7]
        tau_target_shape = [1, 2, 4, 2, 3, 2]
        self.assertTrue(np.array_equal(a_target_shape, a.shape))
        self.assertTrue(np.array_equal(tau_target_shape, tau.shape))

        # Test with synthetic array
        scene.synthetic_array = True
        paths = scene.compute_paths(max_depth=1, num_samples=1e6, scattering=False,
                                    ris=True, los=False, reflection=False, diffraction=False)
        paths.apply_doppler(100e6, 7, [10,0,0], [0,10,0])
        paths.normalize_delays = False

        # Non clustering of RIS paths
        a, tau = paths.cir(cluster_ris_paths=False)
        a_target_shape = [1, 2, 4, 2, 3, 10025, 7]
        tau_target_shape = [1, 2, 2, 10025]
        self.assertTrue(np.array_equal(a_target_shape, a.shape))
        self.assertTrue(np.array_equal(tau_target_shape, tau.shape))

        # Clustering of RIS paths
        a, tau = paths.cir(cluster_ris_paths=True)
        a_target_shape = [1, 2, 4, 2, 3, 2, 7]
        tau_target_shape = [1, 2, 2, 2]
        self.assertTrue(np.array_equal(a_target_shape, a.shape))
        self.assertTrue(np.array_equal(tau_target_shape, tau.shape))