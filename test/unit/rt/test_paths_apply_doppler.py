#
# SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import unittest
import numpy as np
import tensorflow as tf
from tensorflow.experimental.numpy import swapaxes
import sionna
from sionna.rt import load_scene, Transmitter, Receiver, PlanarArray
from sionna.rt.utils import r_hat
from sionna.channel import cir_to_ofdm_channel, subcarrier_frequencies
from sionna.signal import fft
from sionna.constants import SPEED_OF_LIGHT

def compute_doppler_spectrum(scene, paths, tx_velocities, rx_velocities):
    num_subcarriers = 128
    subcarrier_spacing = 15e3
    bandwidth = num_subcarriers * subcarrier_spacing
    frequencies = subcarrier_frequencies(num_subcarriers, subcarrier_spacing)
    num_time_steps = 1000
    sampling_frequency = bandwidth / num_subcarriers
    sampling_time = 1/sampling_frequency

    paths.apply_doppler(sampling_frequency=sampling_frequency,
                        num_time_steps=num_time_steps,
                        tx_velocities=tx_velocities,
                        rx_velocities=rx_velocities)
    a, tau = paths.cir()

    # Compute Doppler spectrum
    h_freq = cir_to_ofdm_channel(frequencies, a, tau, normalize=True)
    h_freq = tf.squeeze(h_freq, axis=0)
    h_freq = swapaxes(h_freq, -1, -2)

    ds = tf.abs(fft(h_freq, axis=-1))**2
    ds = tf.signal.fftshift(ds, axes=[-1])
    ds = tf.reduce_mean(ds, axis=-2)

    # Compute delay-to-speed transformation
    doppler_resolution = 1/(sampling_time*num_time_steps)
    if tf.equal(tf.math.floormod(num_time_steps, 2), 0):
        start=-num_time_steps/2
        limit=num_time_steps/2
    else:
        start=-(num_time_steps-1)/2
        limit=(num_time_steps-1)/2+1

    doppler_frequencies = tf.range(start=start, limit=limit) * doppler_resolution
    velocities = doppler_frequencies/scene.frequency*SPEED_OF_LIGHT

    return velocities, ds


class TestApplyDoppler(unittest.TestCase):

    def test_moving_tx(self):
        """Test that the TX speed can be correctly estimated from the Doppler spectrum"""
        scene = load_scene()
        scene.tx_array = PlanarArray(num_rows=1,
                                num_cols=1,
                                vertical_spacing=0.5,
                                horizontal_spacing=0.5,
                                pattern="iso",
                                polarization="V")
        scene.rx_array = scene.tx_array
        scene.add(Transmitter(name="tx1",
                            position=[0,0,0],
                            orientation=[0,0,0]))
        scene.add(Receiver(name="rx1",
                            position=[10,0,0],
                            orientation=[0,0,0]))
        paths = scene.compute_paths()
        rx_velocities = np.array([0,0,0])
        v = 100
        tx_velocities = np.array([v,0,0])
        velocities, ds = compute_doppler_spectrum(scene, paths, tx_velocities, rx_velocities)
        v_hat = velocities[tf.squeeze(tf.argmax(ds, axis=-1))]
        self.assertLessEqual(np.abs(v_hat-v)/v, 0.05)

    def test_moving_rx(self):
        """Test that the RX speed can be correctly estimated from the Doppler spectrum"""
        scene = load_scene()
        scene.tx_array = PlanarArray(num_rows=1,
                                num_cols=1,
                                vertical_spacing=0.5,
                                horizontal_spacing=0.5,
                                pattern="iso",
                                polarization="V")
        scene.rx_array = scene.tx_array
        scene.add(Transmitter(name="tx1",
                            position=[0,0,0],
                            orientation=[0,0,0]))
        scene.add(Receiver(name="rx1",
                            position=[10,0,0],
                            orientation=[0,0,0]))
        paths = scene.compute_paths()
        v = -100
        rx_velocities = np.array([v,0,0])
        tx_velocities = np.array([0,0,0])
        velocities, ds = compute_doppler_spectrum(scene, paths, tx_velocities, rx_velocities)
        v_hat = velocities[tf.squeeze(tf.argmax(ds, axis=-1))]
        self.assertLessEqual(np.abs(v_hat-v)/v, 0.05)

    def test_moving_tx_rx(self):
        """Test that the differentia TX-RX speed can be correctly estimated from the Doppler spectrum"""
        scene = load_scene()
        scene.tx_array = PlanarArray(num_rows=1,
                                num_cols=1,
                                vertical_spacing=0.5,
                                horizontal_spacing=0.5,
                                pattern="iso",
                                polarization="V")
        scene.rx_array = scene.tx_array
        scene.add(Transmitter(name="tx1",
                            position=[0,0,0],
                            orientation=[0,0,0]))
        scene.add(Receiver(name="rx1",
                            position=[10,0,0],
                            orientation=[0,0,0]))
        paths = scene.compute_paths()
        v_rx = 10
        v_tx = 30
        rx_velocities = np.array([v_rx,0,0])
        tx_velocities = np.array([v_tx,0,0])
        velocities, ds = compute_doppler_spectrum(scene, paths, tx_velocities, rx_velocities)
        v_hat = velocities[tf.squeeze(tf.argmax(ds, axis=-1))]
        self.assertLessEqual(np.abs(v_hat-(v_tx-v_rx))/(v_tx-v_rx), 0.05)

    def test_multi_tx_rx_synthetic_array(self):
        """Check that the doppler spectra for all pairs of antennas of each link
           match, in a multi-TX multi-RX scenario with a synthetic array.
        """
        scene = load_scene(sionna.rt.scene.simple_wedge)
        scene.synthetic_array=True
        scene.tx_array = PlanarArray(num_rows=2,
                                num_cols=2,
                                vertical_spacing=0.5,
                                horizontal_spacing=0.5,
                                pattern="iso",
                                polarization="V")
        scene.rx_array = PlanarArray(num_rows=1,
                                num_cols=2,
                                vertical_spacing=0.5,
                                horizontal_spacing=0.5,
                                pattern="iso",
                                polarization="V")
        scene.add(Transmitter(name="tx1",
                    position=[20,-20,0],
                    orientation=[0,0,0]))
        scene.add(Transmitter(name="tx2",
                    position=[10,-30,0],
                    orientation=[0,0,0]))
        scene.add(Receiver(name="rx1",
                position=[20,-20, 0],
                orientation=[0,0,0]))
        scene.add(Receiver(name="rx2",
                position=[20,-20, 0],
                orientation=[0,0,0]))
        scene.add(Receiver(name="rx3",
                position=[20,-20, 0],
                orientation=[0,0,0]))
        paths = scene.compute_paths(max_depth=1, los=False)
        rx_velocities = np.zeros([3,3])
        rx_velocities[0] = [10, 10, 0]
        rx_velocities[1] = [-50, -50, 0]

        tx_velocities = np.zeros([2, 3])
        tx_velocities[0] = [0, 0, 0]
        tx_velocities[1] = [110, 0, 0]
        velocities, ds = compute_doppler_spectrum(scene, paths, tx_velocities, rx_velocities)
        ds_max = tf.reduce_max(tf.abs(ds), axis=[1, 3, 4], keepdims=True)
        ds = tf.where(ds<ds_max/0.5, 0.0, 1.0)

        for i, _ in enumerate(scene.receivers):
            for j, _ in enumerate(scene.transmitters):
                ref = ds[i,0,j,0]
                for k in range(scene.rx_array.num_ant):
                    for l in range(scene.tx_array.num_ant):
                        self.assertTrue(tf.reduce_all(ref==ds[i,k,j,l]))

    def test_multi_tx_rx_non_synthetic_array(self):
        """Check that the doppler spectra for all pairs of antennas of each link
           match, in a multi-TX multi-RX scenario with a non-synthetic array.
        """
        scene = load_scene(sionna.rt.scene.simple_wedge)
        scene.synthetic_array=False
        scene.tx_array = PlanarArray(num_rows=2,
                                num_cols=2,
                                vertical_spacing=0.5,
                                horizontal_spacing=0.5,
                                pattern="iso",
                                polarization="V")
        scene.rx_array = PlanarArray(num_rows=1,
                                num_cols=2,
                                vertical_spacing=0.5,
                                horizontal_spacing=0.5,
                                pattern="iso",
                                polarization="V")
        scene.add(Transmitter(name="tx1",
                    position=[20,-20,0],
                    orientation=[0,0,0]))
        scene.add(Transmitter(name="tx2",
                    position=[10,-30,0],
                    orientation=[0,0,0]))
        scene.add(Receiver(name="rx1",
                position=[20,-20, 0],
                orientation=[0,0,0]))
        scene.add(Receiver(name="rx2",
                position=[20,-20, 0],
                orientation=[0,0,0]))
        scene.add(Receiver(name="rx3",
                position=[20,-20, 0],
                orientation=[0,0,0]))
        paths = scene.compute_paths(max_depth=1, los=False)
        rx_velocities = np.zeros([3,3])
        rx_velocities[0] = [10, 10, 0]
        rx_velocities[1] = [-50, -50, 0]

        tx_velocities = np.zeros([2, 3])
        tx_velocities[0] = [0, 0, 0]
        tx_velocities[1] = [110, 0, 0]
        velocities, ds = compute_doppler_spectrum(scene, paths, tx_velocities, rx_velocities)
        ds_max = tf.reduce_max(tf.abs(ds), axis=[1, 3, 4], keepdims=True)
        ds = tf.where(ds<ds_max/0.5, 0.0, 1.0)

        for i, _ in enumerate(scene.receivers):
            for j, _ in enumerate(scene.transmitters):
                ref = ds[i,0,j,0]
                for k in range(scene.rx_array.num_ant):
                    for l in range(scene.tx_array.num_ant):
                        self.assertTrue(tf.reduce_all(ref==ds[i,k,j,l]))

    def test_moving_reflector(self):
        """Test that moving reflector has the right Doppler shift"""
        scene = load_scene(sionna.rt.scene.simple_reflector)
        scene.get("reflector").velocity = [0, 0, -20]
        scene.tx_array = PlanarArray(1,1,0.5,0.5,"iso","V")
        scene.rx_array = scene.tx_array
        scene.add(Transmitter("tx", [-25,0.1,50]))
        scene.add(Receiver("rx",    [ 25,0.1,50]))      
        
        # Compute the reflected path
        paths = scene.compute_paths(max_depth=1, los=False)

        # Compute theoretical Doppler shift for this path
        theta_t = tf.squeeze(paths.theta_t)
        phi_t = tf.squeeze(paths.phi_t)
        k_0 = r_hat(theta_t, phi_t)
        theta_r = tf.squeeze(paths.theta_r)
        phi_r = tf.squeeze(paths.phi_r)
        k_1 = -r_hat(theta_r, phi_r)
        doppler_theo = np.sum((k_1-k_0)*scene.get("reflector").velocity)/scene.wavelength

        self.assertAlmostEqual(tf.squeeze(paths.doppler), doppler_theo)