#
# SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import unittest
import numpy as np
import tensorflow as tf
import sionna
from sionna.rt import load_scene, Transmitter, Receiver, PlanarArray, RIS
from utils import *

class TestSingleReflectionWithoutLoS(unittest.TestCase):
    def test_single_reflection_with_obstructed_los(self):
        """This is a corner-case test as the reflection
           occurs at the boundary between two triangles.
           This leads to one invalid path.
        """
        scene = load_scene(sionna.rt.scene.floor_wall)
        scene.synthetic_array = True
        scene.tx_array = PlanarArray(1, 1, 0.5, 0.5, pattern="iso", polarization="V")
        scene.rx_array = PlanarArray(1, 1, 0.5, 0.5, pattern="iso", polarization="V")
        tx = Transmitter("tx", [-1, 0, 2], [0, 0, 0])
        rx = Receiver("rx", [1, 0, 2], [0, 0, 0])
        scene.add(tx)
        scene.add(rx)
        paths = scene.compute_paths(max_depth=1, method="exhaustive", diffraction=False, scattering=False, testing=True)
        mask = np.squeeze(paths.targets_sources_mask, axis=(0,1))
        for i in range(mask.shape[0]):
            if mask[i]: # Check that paths is valid
                a, b = validate_path(i, paths, scene)
                self.assertTrue(np.allclose(a,b))

    def test_single_reflection_with_colocated_tx_and_rx(self):
        """Transmitter and receiver are colocated aboved a floor plane.
           leading to a single reflected path with normal incidence.
        """
        scene = load_scene(sionna.rt.scene.floor_wall)
        scene.synthetic_array = True
        scene.tx_array = PlanarArray(1, 1, 0.5, 0.5, pattern="iso", polarization="V")
        scene.rx_array = PlanarArray(1, 1, 0.5, 0.5, pattern="iso", polarization="V")
        tx = Transmitter("tx", [-1, 1, 3], [0, 0, 0])
        rx = Receiver("rx", [-1, 1, 3], [0, 0, 0])
        scene.add(tx)
        scene.add(rx)
        paths = scene.compute_paths(max_depth=1, diffraction=False, scattering=False, testing=True)
        mask = np.squeeze(paths.targets_sources_mask, axis=(0,1))
        for i in range(mask.shape[0]):
            if mask[i]: # Check that paths is valid
                a, b = validate_path(i, paths, scene)
                self.assertTrue(np.allclose(a,b))

    def test_multiple_reflections(self):
        """Test transfer matrices of paths in a larger scene"""
        dtype = tf.complex128
        scene = load_scene(sionna.rt.scene.simple_street_canyon, dtype=dtype)
        scene.synthetic_array = True
        scene.tx_array = PlanarArray(1, 1, 0.5, 0.5, pattern="iso", polarization="V", dtype=dtype)
        scene.rx_array = PlanarArray(1, 1, 0.5, 0.5, pattern="iso", polarization="V", dtype=dtype)
        tx = Transmitter("tx", [-10, 1, 4], [0, 0, 0], dtype=dtype)
        rx = Receiver("rx", [-2, 1.5, 4], [0, 0, 0], dtype=dtype)
        scene.add(tx)
        scene.add(rx)
        paths = scene.compute_paths(max_depth=5, method="fibonacci", diffraction=False, scattering=False, testing=True)
        mask = np.squeeze(paths.targets_sources_mask, axis=(0,1))
        for i in range(mask.shape[0]):
            if mask[i]: # Check that paths is valid
                a, b = validate_path(i, paths, scene)
                self.assertTrue(np.allclose(a,b))

    def test_masked_paths(self):
        """Test that a of masked paths are equal to zero"""
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
        paths = scene.compute_paths(max_depth=3, scattering=True,
                                    edge_diffraction=True, testing=True)
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
                        for path in range(0, paths.targets_sources_mask.shape[-1]):
                            if paths.tau[0, rx, rx_ant, tx, tx_ant, path] < 0:
                                a = paths.a[0, rx, rx_ant, tx, tx_ant, path]
                                self.assertTrue(np.array_equal(a, np.zeros_like(a)))
                            if not paths.targets_sources_mask[target, source, path]:
                                a = paths.a[0, rx, rx_ant, tx, tx_ant, path]
                                tau = paths.tau[0, rx, rx_ant, tx, tx_ant, path]
                                self.assertTrue(np.array_equal(a, np.zeros_like(a)))
                                self.assertTrue(np.array_equal(tau, -np.ones_like(tau)))

    def test_diffracted_paths(self):

        for dtype in (tf.complex64, tf.complex128):

            scene = load_scene(sionna.rt.scene.simple_wedge, dtype=dtype)
            scene.synthetic_array = True

            scene.tx_array = PlanarArray(num_rows=1,
                        num_cols=1,
                        vertical_spacing=0.5,
                        horizontal_spacing=0.5,
                        pattern="iso",
                        polarization="V",
                                        dtype=dtype)
            scene.rx_array = PlanarArray(num_rows=1,
                                        num_cols=1,
                                        vertical_spacing=0.5,
                                        horizontal_spacing=0.5,
                                        pattern="iso",
                                        polarization="V",
                                        dtype=dtype)

            # Unique transmitter with angle PI/4
            tx = Transmitter(name="tx",
                            position=[5.0, 5.0, 0.0],
                            orientation=[0,0,0],
                            dtype=dtype)
            scene.add(tx)

            # Add many receivers on a circle
            num_rx = 300
            dist = 1.0
            angles_boundary_1 = np.linspace(130.0*np.pi/180.0, 140.0*np.pi/180.0, num_rx)
            angles_boundary_2 = np.linspace(220.0*np.pi/180.0, 230.0*np.pi/180.0, num_rx)
            angles = np.concatenate([angles_boundary_1, angles_boundary_2], axis=0)
            xs = dist*np.cos(angles)
            ys = dist*np.sin(angles)
            positions = np.stack([xs, ys, np.zeros_like(xs)], axis=1)
            for i,p in enumerate(positions):
                rx = Receiver(name=f"rx-{i}",
                            position=p,
                            orientation=[0,0,0],
                            dtype=dtype)
                scene.add(rx)

            paths = scene.compute_paths(max_depth=1, scattering=False, reflection=True, los=True, diffraction=True, testing=True)

            spec_mat_t = paths.spec_tmp.mat_t.numpy()
            diff_mat_t = paths.diff_tmp.mat_t.numpy()
            mat_t = np.concatenate([spec_mat_t,diff_mat_t], axis=2)
            mat_t = np.squeeze(mat_t, axis=1)
            mat_t_1 = mat_t[:num_rx]
            mat_t_2 = mat_t[num_rx:]

            mask = np.squeeze(paths.targets_sources_mask.numpy(), axis=1)
            mask = np.expand_dims(mask, axis=(2,3))
            mask_1 = mask[:num_rx]
            mask_2 = mask[num_rx:]

            total_energy_1 = np.sum(np.square(np.abs(np.sum(mask_1*mat_t_1, axis=1))), axis=(1,2))
            diff_total_energy_1 = np.abs(total_energy_1[1:] - total_energy_1[:-1])
            avg_energy_1 = 0.5*(total_energy_1[1:] + total_energy_1[:-1])
            rel_err_1 = diff_total_energy_1 / avg_energy_1

            total_energy_2 = np.sum(np.square(np.abs(np.sum(mask_2*mat_t_2, axis=1))), axis=(1,2))
            diff_total_energy_2 = np.abs(total_energy_2[1:] - total_energy_2[:-1])
            avg_energy_2 = 0.5*(total_energy_2[1:] + total_energy_2[:-1])
            rel_err_2 = diff_total_energy_2 / avg_energy_2

            max_rel_err = np.maximum(np.max(rel_err_1), np.max(rel_err_2))
            self.assertLess(max_rel_err, 1e-2)

class TestRISOcclusion(unittest.TestCase):

    def test_ris_occlude_los(self):
        """In a scene containing RIS, test that RIS can occlude the LoS
        """
        scene = load_scene(sionna.rt.scene.simple_reflector)
        ris0 = RIS("ris-0",
                [0.0, 0, 2.0],
                orientation=[0., 0., 0.],
                num_rows=32,
                num_cols=32)
        scene.add(ris0)

        tx = Transmitter("tx",
                        [1.0, 0.0, 2.1])
        scene.add(tx)

        rx = Receiver("rx",
                        [-1.0, 0.0, 2.])
        scene.add(rx)

        # Configure antenna array for all transmitters
        scene.tx_array = PlanarArray(num_rows=1,
                                    num_cols=1,
                                    vertical_spacing=0.5,
                                    horizontal_spacing=0.5,
                                    pattern="iso",
                                    polarization="V")

        # Configure antenna array for all receivers
        scene.rx_array = PlanarArray(num_rows=1,
                                    num_cols=1,
                                    vertical_spacing=0.5,
                                    horizontal_spacing=0.5,
                                    pattern="iso",
                                    polarization="V")

        # Compute paths
        paths = scene.compute_paths(ris=True, reflection=True, los=True, max_depth=1, diffraction=True, edge_diffraction=True)

        paths_types = paths.types[0]
        self.assertTrue(np.where(paths_types == 0)[0].size == 0)
        self.assertTrue(np.where(paths_types == 1)[0].size == 1)
        self.assertTrue(np.where(paths_types == 2)[0].size == 4)
        self.assertTrue(np.where(paths_types == 4)[0].size == 32*32)

    def test_ris_occlude_specular_diffuse_diffracted(self):
        """In a scene containing RIS, test that RIS can occlude the reflected
        (diffuse and specular) and diffracted paths
        """
        scene = sionna.rt.load_scene(sionna.rt.scene.simple_reflector)
        scene.frequency = 1e9
        ris0 = RIS("ris-0",
                [0.0, 0, 0.2],
                orientation=[0., -0.5*np.pi, 0.],
                num_rows=8,
                num_cols=8)
        scene.add(ris0)

        tx = Transmitter("tx",
                        [1.0, 0.0, 2.])
        scene.add(tx)

        rx = Receiver("rx",
                        [-1.0, 0.0, 2.])
        scene.add(rx)

        # Configure antenna array for all transmitters
        scene.tx_array = PlanarArray(num_rows=1,
                                    num_cols=1,
                                    vertical_spacing=0.5,
                                    horizontal_spacing=0.5,
                                    pattern="iso",
                                    polarization="V")

        # Configure antenna array for all receivers
        scene.rx_array = PlanarArray(num_rows=1,
                                    num_cols=1,
                                    vertical_spacing=0.5,
                                    horizontal_spacing=0.5,
                                    pattern="iso",
                                    polarization="V")

        # Compute pats
        paths = scene.compute_paths(ris=True, reflection=True, los=True, max_depth=2, diffraction=True, scattering=False, edge_diffraction=True)

        paths_types = paths.types[0]
        self.assertTrue(np.where(np.logical_and(paths_types == 0, paths.mask[0,0,0]))[0].size == 1)
        self.assertTrue(np.where(np.logical_and(paths_types == 1, paths.mask[0,0,0]))[0].size == 0)
        self.assertTrue(np.where(np.logical_and(paths_types == 2, paths.mask[0,0,0]))[0].size == 0)
        self.assertTrue(np.where(np.logical_and(paths_types == 3, paths.mask[0,0,0]))[0].size == 0)
        self.assertTrue(np.where(np.logical_and(paths_types == 4, paths.mask[0,0,0]))[0].size == 8*8)

    def test_ris_occlude_ris(self):
        """In a scene containing RIS, test that RIS can occlude the paths reflected
        by another RIS
        """
        scene = sionna.rt.load_scene(sionna.rt.scene.simple_reflector)
        scene.frequency = 1e9

        ris0 = RIS("ris-0",
                [0.0, 0, 0.1],
                orientation=[0., -0.5*np.pi, 0.],
                num_rows=8,
                num_cols=8)
        scene.add(ris0)

        ris1 = RIS("ris-1",
                [0.0, 0, 1.0],
                orientation=[0., -0.5*np.pi, 0.],
                num_rows=11,
                num_cols=11)
        scene.add(ris1)

        tx = Transmitter("tx",
                        [1.0, 0.0, 2.])
        scene.add(tx)

        rx = Receiver("rx",
                        [-1.0, 0.0, 2.])
        scene.add(rx)

        # Configure antenna array for all transmitters
        scene.tx_array = PlanarArray(num_rows=1,
                                    num_cols=1,
                                    vertical_spacing=0.5,
                                    horizontal_spacing=0.5,
                                    pattern="iso",
                                    polarization="V")

        # Configure antenna array for all receivers
        scene.rx_array = PlanarArray(num_rows=1,
                                    num_cols=1,
                                    vertical_spacing=0.5,
                                    horizontal_spacing=0.5,
                                    pattern="iso",
                                    polarization="V")

        # Compute pats
        paths = scene.compute_paths(ris=True, reflection=True, los=True, max_depth=1, diffraction=True, scattering=True, edge_diffraction=True, scat_keep_prob=1e-1)

        paths_types = paths.types[0]
        self.assertTrue(np.where(np.logical_and(paths_types == 0, paths.mask[0,0,0]))[0].size == 1)
        self.assertTrue(np.where(np.logical_and(paths_types == 1, paths.mask[0,0,0]))[0].size == 0)
        self.assertTrue(np.where(np.logical_and(paths_types == 2, paths.mask[0,0,0]))[0].size == 0)
        self.assertTrue(np.where(np.logical_and(paths_types == 3, paths.mask[0,0,0]))[0].size == 0)
        self.assertTrue(np.where(np.logical_and(paths_types == 4, paths.mask[0,0,0]))[0].size == 11*11)
