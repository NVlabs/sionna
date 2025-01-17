#
# SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import unittest
import numpy as np
import tensorflow as tf
import sionna
from sionna.rt import load_scene, Transmitter, Receiver, PlanarArray, \
                      RadioMaterial, LambertianPattern, BackscatteringPattern
from sionna.rt.utils import dbm_to_watt

class TestCovMap(unittest.TestCase):

    def test_random_positions(self):
        """test that random positions have a valid path loss and min/max
        distance is correctly set."""

        cm_cell_size = np.array([4., 5.])
        batch_size = 100
        tx_pos = np.array([-210,73,105]) # top of Frauenkirche

        scene = load_scene(sionna.rt.scene.munich)

        tx = Transmitter(name="tx",
                         position=tx_pos,)
        scene.add(tx)

        # dummy - not needed
        rx = Receiver("rx", position=[0,0,0])
        scene.add(rx)

        scene.tx_array = PlanarArray(num_rows=4,
                                     num_cols=4,
                                     vertical_spacing=0.5,
                                     horizontal_spacing=0.5,
                                     pattern="iso",
                                     polarization="V")
        scene.rx_array = PlanarArray(num_rows=4,
                                     num_cols=4,
                                     vertical_spacing=0.5,
                                     horizontal_spacing=0.5,
                                     pattern="iso",
                                     polarization="V")

        rx_pos = scene.transmitters["tx"].position.numpy()
        rx_pos[-1] = 1.5 # set height of coverage map to 1.5m

        # generate coverage map
        cm = scene.coverage_map(
                            rx_orientation=(0., 0., 0.),
                            max_depth=5,
                            cm_center=rx_pos,
                            cm_orientation=(0., 0., 0.), # no rotation
                            cm_size=(500., 500.),
                            cm_cell_size=cm_cell_size,
                            combining_vec=None,
                            precoding_vec=None,
                            los=True,
                            reflection=True,
                            scattering=True,
                            diffraction=True,
                            num_samples=int(1e6))

        ### centering is True
        pos, _ = cm.sample_positions(
                    batch_size,
                    min_val_db=-110,
                    #max_val_db=-100,
                    #min_dist=100,
                    #max_dist=250,
                    center_pos=True
                    )
        pos = pos.numpy()
        cpos = cm.cell_centers.numpy()

        for tx in range(pos.shape[0]):
            for i in range(batch_size):
                # find closest point in coverage map
                success = False
                for j in range(cpos.shape[0]):
                    for k in range(cpos.shape[1]):
                        d = np.abs(pos[tx, i]-cpos[j,k])
                        if np.sum(d)==0.:
                            success=True
                            break
                    if success:
                        break
                self.assertTrue(success) # "position not centered on grid"

        ### centering is False
        pos, _ = cm.sample_positions(
                    batch_size,
                    min_val_db=-110,
                    #max_val_db=-100,
                    #min_dist=100,
                    #max_dist=250,
                    center_pos=False
                    )
        pos = pos.numpy()
        cpos = cm.cell_centers.numpy()

        for tx in range(pos.shape[0]):
            for i in range(batch_size):
                # find closest point in coverage map
                success = False
                for j in range(cpos.shape[0]):
                    for k in range(cpos.shape[1]):
                        d = np.abs(pos[tx, i]-cpos[j,k])
                        if  d[0]<=cm_cell_size[0]/2 and d[1]<=cm_cell_size[1]/2: # no z-direction in this example
                            success=True
                            break
                    if success:
                        break
                self.assertTrue(success) # "position not within valid cell"

        ### test min and max distance
        batch_size = 1000 # this test is simple and can run with more samples
        d_min = 150
        d_max = 300
        # max distance offset due to cell size quantization
        # dist can be off at most by factor 0.5 of diagonal
        d_cell = np.sqrt(np.sum((cm_cell_size)**2))/2

        pos, _ = cm.sample_positions(
                    batch_size,
                    min_dist=d_min,
                    max_dist=d_max,
                    center_pos=False)
        pos = pos.numpy()

        for tx in range(pos.shape[0]):
            for i in range(batch_size):
                d = np.sqrt(np.sum((pos[tx, i]-tx_pos)**2))
                valid_dist = True
                if d<(d_min-d_cell):
                    valid_dist = False
                if d>(d_max+d_cell):
                    valid_dist = False
                self.assertTrue(valid_dist)

        ########################
        # test TX associations #
        ########################

        # Remove all transmitters
        [scene.remove(tx_) for tx_ in scene.transmitters]

        # Add 3 transmitters
        tx_pos = [
                [150, -100, 20],
                [-150, -100, 20],
                [0, 150 * np.tan(np.pi/3) - 100, 20]
        ]
        # Add the first transmitter
        tx0 = Transmitter(name='tx0',
                          position=tx_pos[0],
                          orientation=[np.pi*5/6, 0, 0],
                          power_dbm=44)
        scene.add(tx0)

        # Add the second transmitter
        tx1 = Transmitter(name='tx1',
                          position=tx_pos[1],
                          orientation=[np.pi/6, 0, 0],
                          power_dbm=44)
        scene.add(tx1)

        # Add the third transmitter
        tx2 = Transmitter(name='tx2',
                          # symmetric positions
                          position=tx_pos[2],
                          orientation=[-np.pi/2, 0, 0],
                          power_dbm=44)
        scene.add(tx2)

        # generate coverage map
        cm = scene.coverage_map(
                            rx_orientation=(0., 0., 0.),
                            max_depth=5,
                            cm_center=[0, 0, 0],
                            cm_orientation=(0., 0., 0.), # no rotation
                            cm_size=(500., 500.),
                            cm_cell_size=cm_cell_size,
                            combining_vec=None,
                            precoding_vec=None,
                            los=True,
                            reflection=True,
                            scattering=True,
                            diffraction=True,
                            num_samples=int(1e6))
        
        metric = 'sinr'
        sm = getattr(cm, 'sinr').numpy()
        cell_to_tx_ideal = np.argmax(sm, axis=0)
        cell_to_tx_ideal[np.max(sm, axis=0)==0] = -1

        pos, pos_idx = cm.sample_positions(
                    batch_size,
                    center_pos=False,
                    tx_association=True,
                    metric=metric
                    )
        # invert last dimensions to get (row, column) cell index pairs
        pos_idx = tf.gather(pos_idx, [1, 0], axis=-1).numpy().astype(int)

        for tx in range(sm.shape[0]):
            success = True
            for pos in pos_idx[tx]:
                if cell_to_tx_ideal[pos[0], pos[1]] != tx:
                    success = False
                    break
            self.assertTrue(success) # "tx association is incorrect"

    def test_sinr_map(self):
        """ Test that SINR map computation is correct """
        cm_cell_size = np.array([4., 5.])

        scene = load_scene(sionna.rt.scene.munich)

        tx_pos = [
            [150, -100, 20],
            [-150, -100, 20],
            [0, 150 * np.tan(np.pi/3) - 100, 20]
        ]
        # Add the first transmitter
        tx0 = Transmitter(name='tx0',
                          position=tx_pos[0],
                          orientation=[np.pi*5/6, 0, 0],
                          power_dbm=44)
        scene.add(tx0)

        # Add the second transmitter
        tx1 = Transmitter(name='tx1',
                          position=tx_pos[1],
                          orientation=[np.pi/6, 0, 0],
                          power_dbm=44)
        scene.add(tx1)

        # Add the third transmitter
        tx2 = Transmitter(name='tx2',
                          # symmetric positions
                          position=tx_pos[2],
                          orientation=[-np.pi/2, 0, 0],
                          power_dbm=44)
        scene.add(tx2)

        # dummy - not needed
        rx = Receiver("rx", position=[0, 0, 0])
        scene.add(rx)

        scene.tx_array = PlanarArray(num_rows=4,
                                     num_cols=4,
                                     vertical_spacing=0.5,
                                     horizontal_spacing=0.5,
                                     pattern="iso",
                                     polarization="V")
        scene.rx_array = PlanarArray(num_rows=4,
                                     num_cols=4,
                                     vertical_spacing=0.5,
                                     horizontal_spacing=0.5,
                                     pattern="iso",
                                     polarization="V")

        rx_pos = scene.transmitters["tx0"].position.numpy()
        rx_pos[-1] = 1.5  # set height of coverage map to 1.5m

        # generate coverage map
        cm = scene.coverage_map(
            rx_orientation=(0., 0., 0.),
            max_depth=5,
            cm_center=rx_pos,
            cm_orientation=(0., 0., 0.),  # no rotation
            cm_size=(500., 500.),
            cm_cell_size=cm_cell_size,
            combining_vec=None,
            precoding_vec=None,
            los=True,
            reflection=True,
            scattering=True,
            diffraction=True,
            num_samples=int(1e6))

        # retrieve path gain map
        path_gain = getattr(cm, 'path_gain').numpy()

        # retrieve transmit power
        tx_power = [dbm_to_watt(tx.power_dbm)
                    for tx in scene.transmitters.values()]

        # retrieve noise power
        n0 = scene.thermal_noise_power

        num_tx = cm.num_tx
        sinr_map_per_tx_ideal = np.zeros(path_gain.shape)

        # compute the SINR via Numpy operations
        for tx in range(num_tx):
            interf_mat = np.zeros(path_gain.shape[1:])
            for tx_interf in range(num_tx):
                # compute interference
                if tx_interf != tx:
                    interf_mat += tx_power[tx_interf] * \
                        path_gain[tx_interf, ::]
            # SINR per tx,  assuming that all tiles connect to transmitter tx
            sinr_map_per_tx_ideal[tx, ::] = tx_power[tx] * \
                path_gain[tx, ::] / (interf_mat + n0)

        # Tile to tx association
        tile_to_tx_ideal = np.argmax(sinr_map_per_tx_ideal, axis=0)
        tile_to_tx_ideal[np.max(sinr_map_per_tx_ideal, axis=0) == 0] = -1

        # Transform to numpy
        sinr_map_per_tx = getattr(cm, 'sinr').numpy()
        tile_to_tx = cm.cell_to_tx('sinr').numpy()

        # Gap between Tensor and Numpy versions
        err_sinr_map_per_tx = np.sort(
            abs((sinr_map_per_tx - sinr_map_per_tx_ideal)/sinr_map_per_tx_ideal).flatten())
        err_sinr_map_per_tx = err_sinr_map_per_tx[~np.isnan(
            err_sinr_map_per_tx)]

        err_tile_to_tx = np.max(abs(tile_to_tx_ideal - tile_to_tx))

        # check that 98-th percentile SINR has small error
        self.assertTrue(
            err_sinr_map_per_tx[int(.98*len(err_sinr_map_per_tx))] < .01)
        self.assertTrue(err_tile_to_tx == 0)


    def test_dtype(self):
        """test against different dtypes"""

        cm_cell_size = np.array([1., 1.])
        batch_size = 100
        tx_pos = np.array([1.,1.,2.])

        # load simple scene with different dtypes
        for dt in (tf.complex64, tf.complex128):
            scene = load_scene(sionna.rt.scene.floor_wall, dtype=dt)

            tx = Transmitter(name="tx",
                             position=tx_pos,
                             dtype=dt)
            scene.add(tx)

            # dummy - not needed
            rx = Receiver("rx", position=[0,0,0],dtype=dt)
            scene.add(rx)

            scene.tx_array = PlanarArray(num_rows=1,
                                        num_cols=1,
                                        vertical_spacing=0.5,
                                        horizontal_spacing=0.5,
                                        pattern="iso",
                                        polarization="V",
                                        dtype=dt)
            scene.rx_array = PlanarArray(num_rows=1,
                                        num_cols=1,
                                        vertical_spacing=0.5,
                                        horizontal_spacing=0.5,
                                        pattern="iso",
                                        polarization="V",
                                        dtype=dt)

            # generate coverage map
            cm = scene.coverage_map(
                                rx_orientation=(0., 0., 0.),
                                max_depth=3,
                                cm_center=tx_pos,
                                cm_orientation=(0., 0., 0.), # no rotation
                                cm_size=(50., 50.),
                                cm_cell_size=cm_cell_size,
                                los=True,
                                reflection=True,
                                scattering=True,
                                diffraction=True,
                                edge_diffraction=True) # To get some diffraction with this scene

            # and sample positions
            pos, _ = cm.sample_positions(
                        batch_size,
                        min_val_db=-110,
                        max_val_db=-100,
                        min_dist=1,
                        max_dist=25,
                        center_pos=True)

    def test_diffraction_coverage_map(self):
        for dtype in (tf.complex64,tf.complex128):

            scene = load_scene(sionna.rt.scene.simple_wedge, dtype=dtype)

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
                            position=[1.0, 1.0, 0.0],
                            orientation=[0,0,0],
                            dtype=dtype)
            scene.add(tx)

            # Coverage map properties
            cm_center = np.array([-1.0,-2.5, -10.0])
            cm_orientation = np.array([0.,0.,0.])
            cm_size = np.array([1.0,1.0])
            cm_cell_size = np.array([0.1,0.1])
            num_cells_x = int(np.ceil(cm_size[0]/cm_cell_size[0]))
            num_cells_y = int(np.ceil(cm_size[1]/cm_cell_size[1]))

            # Add receivers in the center of every of the coverage map

            rx_pos_x_min = cm_center[0] - cm_size[0]*0.5 + cm_cell_size[0]*0.5
            rx_pos_x_max = cm_center[0] + cm_size[0]*0.5 - cm_cell_size[0]*0.5

            rx_pos_y_min = cm_center[1] - cm_size[1]*0.5 + cm_cell_size[1]*0.5
            rx_pos_y_max = cm_center[1] + cm_size[1]*0.5 - cm_cell_size[1]*0.5

            rx_pos_xs = np.linspace(rx_pos_x_min, rx_pos_x_max, num_cells_x, endpoint=True)
            rx_pos_ys = np.linspace(rx_pos_y_min, rx_pos_y_max, num_cells_y, endpoint=True)
            rx_pos_xs, rx_pos_ys = np.meshgrid(rx_pos_xs, rx_pos_ys)
            rx_pos_xs = np.reshape(rx_pos_xs, [-1])
            rx_pos_ys = np.reshape(rx_pos_ys, [-1])
            rx_pos = np.stack([rx_pos_xs, rx_pos_ys, np.full(rx_pos_xs.shape, cm_center[2])], axis=-1)
            for i,p in enumerate(rx_pos):
                rx = Receiver(name=f"rx-{i}",
                            position=p,
                            orientation=[0,0,0],
                            dtype=dtype)
                scene.add(rx)

            # Compute the diffracted field energy using compute_paths()
            paths = scene.compute_paths(los=False, reflection=False, scattering=False, diffraction=True)
            a, _ = paths.cir()
            a = np.squeeze(a.numpy())
            ref_en = np.mean(np.square(np.abs(a)))

            # Compute the diffracted field energy using the coverage map
            cm = scene.coverage_map(cm_center=cm_center, cm_orientation=cm_orientation, cm_size=cm_size, cm_cell_size=cm_cell_size, num_samples=int(15e6), los=False, reflection=False, scattering=False, diffraction=True)
            cm_np = getattr(cm, 'path_gain').numpy()[0]
            en_cm = np.mean(cm_np)

            rel_err = np.abs(ref_en-en_cm)/ref_en
            self.assertTrue(rel_err < 1e-2)


def paths_to_coverage_map(paths):
    """
    Converts paths into the equivalent coverage map values.
    The coverage map is assumed to be quadratic.
    """
    # [1, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths]
    a, _ = paths.cir()

    # Remove batch dim
    # [num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths]
    a = tf.squeeze(a, axis=(0,6))

    # Receive and transmit combining
    # [num_rx, num_tx, num_paths]
    a /= np.sqrt(a.shape[1]*a.shape[3])
    a = tf.reduce_sum(a, axis=[1, 3])

    # Sum energu of paths
    a = tf.abs(a)**2
    # [num_rx, num_tx]
    a = tf.reduce_sum(a, axis=-1)

    # Swap dims
    # [num_tx, num_rx]
    a = tf.transpose(a, perm=[1, 0])

    # Reshape to coverage map
    n = int(np.sqrt(a.shape[1]))
    shape = [a.shape[0], n, n]
    a = tf.reshape(a, shape)

    return a

def validate_cm(los=False,
                reflection=False,
                scattering=False,
                diffraction=False,
                tx_pattern="iso",
                rx_pattern="iso",
                tx_pol="V",
                rx_pol="V"):
    """Compares coverage map against exact path calculation for
    different propagation phenomena.
    """
    dtype=tf.complex64

    scene = load_scene(sionna.rt.scene.simple_reflector, dtype=dtype)

    scene.get("reflector").radio_material = "itu_concrete"
    scene.get("reflector").radio_material.scattering_coefficient = 1/np.sqrt(2)
    scene.get("reflector").radio_material.xpd_coefficient = 0.3

    scene.tx_array = PlanarArray(num_rows=1,
                             num_cols=1,
                             vertical_spacing=0.5,
                             horizontal_spacing=0.5,
                             pattern=tx_pattern,
                             polarization=tx_pol,
                             polarization_model=2,
                             dtype=dtype)

    scene.rx_array =  PlanarArray(num_rows=1,
                             num_cols=1,
                             vertical_spacing=0.5,
                             horizontal_spacing=0.5,
                             pattern=rx_pattern,
                             polarization=rx_pol,
                             polarization_model=2,
                             dtype=dtype)

    delta = 0.1
    width = 3
    tx = Transmitter(name="tx",
                     position=[0, 0, .1],
                     orientation=[0,0,0],
                     dtype=dtype)
    scene.add(tx)

    cm = scene.coverage_map([0,0,0],
                            max_depth=1,
                            cm_cell_size=[delta, delta],
                            cm_center=[0,0,1],
                            cm_orientation=[0.,0.,0.],
                            cm_size=[width, width],
                            num_samples=10e6,
                            los=los,
                            reflection=reflection,
                            diffraction=diffraction,
                            edge_diffraction=True,
                            scattering=scattering, )

    for i, pos in enumerate(np.reshape(cm.cell_centers, [-1,3])):
        scene.add(Receiver(name=f"rx-{i}",
                           position=pos,
                           orientation=[0,0,0],
                           dtype=dtype))

    paths = scene.compute_paths(num_samples = 10000,
                                max_depth=1,
                                los=los,
                                reflection=reflection,
                                diffraction=diffraction,
                                edge_diffraction=True,
                                scattering=scattering,
                                scat_keep_prob=1.)

    cm_theo = paths_to_coverage_map(paths)

    err = tf.math.divide_no_nan(cm.path_gain[0]-cm_theo[0], cm_theo[0])

    nmse_db = 10*np.log10(np.mean(np.abs(err)**2))
    return cm, cm_theo, nmse_db


class TestCovMapVsPaths(unittest.TestCase):
    """Tests comparing the coverage against exact path computations
    for receivers placed in the middle of the coverage map cells."""

    def test_los(self):
        """Test that LoS coverage map is close to exact path calculation"""
        cm, cm_theo, nmse_db = validate_cm(los=True, tx_pol="V", rx_pol="V")
        self.assertLess(nmse_db, -20)
        cm, cm_theo, nmse_db = validate_cm(los=True, tx_pol="cross", rx_pol="cross")
        self.assertLess(nmse_db, -20)

    def test_reflection(self):
        """Test that reflection coverage map is close to exact path calculation"""
        cm, cm_theo, nmse_db = validate_cm(reflection=True, tx_pol="V", rx_pol="V")
        self.assertLess(nmse_db, -20)
        cm, cm_theo, nmse_db = validate_cm(reflection=True, tx_pol="cross", rx_pol="cross")
        self.assertLess(nmse_db, -20)

    def test_scattering(self):
        """Test that scattering coverage map is close to exact path calculation"""
        cm, cm_theo, nmse_db = validate_cm(scattering=True, tx_pol="V", rx_pol="V")
        self.assertLess(nmse_db, -20)
        cm, cm_theo, nmse_db = validate_cm(scattering=True, tx_pol="V", rx_pol="H")
        self.assertLess(nmse_db, -20)

    def test_diffraction(self):
        """Test that diffraction coverage map is close to exact path calculation"""
        cm, cm_theo, nmse_db = validate_cm(diffraction=True, tx_pol="V", rx_pol="V")
        self.assertLess(nmse_db, -20)
        cm, cm_theo, nmse_db = validate_cm(diffraction=True, tx_pol="H", rx_pol="V")
        self.assertLess(nmse_db, -20)

    def test_box_01(self):
        """Test that field scattered and reflected fields jointly match for
        max_depth=1 in the box scene.
        This test also applies orientation, directive antenna patterns, as well as
        depolariation during scattering.
        """
        los = False
        reflection = True
        diffraction = False
        scattering = True
        max_depth = 1 # Test only works for max_depth=1
        width=10
        delta = 1

        dtype=tf.complex64
        scene = load_scene(sionna.rt.scene.box, dtype=dtype)
        scene.objects["floor"].radio_material = "itu_concrete"
        scene.objects["floor"].radio_material.scattering_coefficient = 0.5
        scene.objects["floor"].radio_material.scattering_pattern = BackscatteringPattern(30, 10, 0.5, dtype=dtype)

        scene.tx_array = PlanarArray(num_rows=1,
                                    num_cols=1,
                                    vertical_spacing=0.5,
                                    horizontal_spacing=0.5,
                                    pattern="tr38901",
                                    polarization="V",
                                    polarization_model=2,
                                    dtype=dtype)

        scene.rx_array =  PlanarArray(num_rows=1,
                                    num_cols=1,
                                    vertical_spacing=0.5,
                                    horizontal_spacing=0.5,
                                    pattern="tr38901",
                                    polarization="H",
                                    polarization_model=2,
                                    dtype=dtype)

        scene.add(Transmitter(name="tx",
                      position=[1.1, 0.8, 2],
                      orientation=[0,0,0],
                      dtype=dtype))
        scene.get("tx").look_at([5,5,5])

        cm = scene.coverage_map(max_depth=max_depth,
                                cm_cell_size=[delta, delta],
                                cm_center=[0,0,1.5],
                                cm_orientation=[0.,0.,0],
                                cm_size=[width, width],
                                num_samples=10e6,
                                los=los,
                                reflection=reflection,
                                diffraction=diffraction,
                                edge_diffraction=True,
                                scattering=scattering)

        for i, pos in enumerate(np.reshape(cm.cell_centers, [-1,3])):
            scene.add(Receiver(name=f"rx-{i}",
                                position=pos,
                                orientation=[0, 0, 0],
                                dtype=dtype))

        paths = scene.compute_paths(num_samples = 10000,
                                    max_depth=max_depth,
                                    los=los,
                                    reflection=reflection,
                                    diffraction=diffraction,
                                    edge_diffraction=True,
                                    scattering=scattering)

        a = paths_to_coverage_map(paths)[0]

        nmse_db = 10*np.log10(np.mean( ((cm.path_gain[0]-a)/a)**2 ))

        self.assertLess(nmse_db, -20)


    def test_box_02(self):
        """Test a multiple reflections and LoS in the box scene.
        It includes directive antenna pattern and a complex material"""
        dtype=tf.complex64
        scene = load_scene(sionna.rt.scene.box, dtype=dtype)
        scene.objects["floor"].radio_material = "itu_concrete"
        scene.objects["floor"].radio_material.scattering_coefficient = 0.3


        scene.tx_array = PlanarArray(num_rows=1,
                                    num_cols=1,
                                    vertical_spacing=0.5,
                                    horizontal_spacing=0.5,
                                    pattern="tr38901",
                                    polarization="V",
                                    polarization_model=2,
                                    dtype=dtype)

        scene.rx_array =  PlanarArray(num_rows=1,
                                    num_cols=1,
                                    vertical_spacing=0.5,
                                    horizontal_spacing=0.5,
                                    pattern="iso",
                                    polarization="V",
                                    polarization_model=2,
                                    dtype=dtype)

        los = True
        reflection = True
        diffraction = False
        scattering = False
        width=9
        num_cells_x = 20
        delta = width/num_cells_x
        max_depth = 5

        scene.add(Transmitter(name="tx",
                            position=[-3, -0.3, 2],
                            orientation=[0,0,0],
                            dtype=dtype))

        cm = scene.coverage_map(max_depth=max_depth,
                                cm_cell_size=[delta, delta],
                                cm_center=[0.1,0.2,1.5],
                                cm_orientation=[0.,0.,0],
                                cm_size=[width, width],
                                num_samples=10e6,
                                los=los,
                                reflection=reflection,
                                diffraction=diffraction,
                                edge_diffraction=True,
                                scattering=scattering)


        for i, pos in enumerate(np.reshape(cm.cell_centers, [-1,3])):
            scene.add(Receiver(name=f"rx-{i}",
                                position=pos,
                                orientation=[0, 0, 0],
                                dtype=dtype))

        paths = scene.compute_paths(num_samples = 10000,
                                    max_depth=max_depth,
                                    los=los,
                                    reflection=reflection,
                                    diffraction=diffraction,
                                    edge_diffraction=True,
                                    scattering=scattering)

        a = paths_to_coverage_map(paths)[0]

        nmse_db = 10*np.log10(np.mean( ((cm.path_gain[0]-a)/a)**2 ))

        self.assertLess(nmse_db, -20)

class TestCovMapGradients(unittest.TestCase):
    """Tests comparing the coverage against exact path computations
    for receivers placed in the middle of the coverage map cells."""

    def test_all_gradients_exist(self):
        """Defines a setup for which gradients for all trainable parameters
        of the radio material should exist, i.e., not be None.
        Check that this is the case.
        """
        dtype=tf.complex64
        rdtype = dtype.real_dtype
        scene = load_scene(sionna.rt.scene.double_reflector, dtype=dtype)

        # Define a trainable radio material
        sp = BackscatteringPattern(5, 5, 0.5, dtype=dtype)
        sp.lambda_ = tf.Variable(0.5, dtype=rdtype, trainable=True)
        mat = RadioMaterial("my_mat",
                            relative_permittivity=1,
                            conductivity=1,
                            scattering_coefficient=0.7,
                            xpd_coefficient=0.2,
                            scattering_pattern=sp,
                            dtype=dtype)
        mat.conductivity = tf.Variable(1.0, dtype=rdtype, trainable=True)
        mat.relative_permittivity = tf.Variable(1.0, dtype=rdtype, trainable=True)
        mat.scattering_coefficient = tf.Variable(0.7, dtype=rdtype, trainable=True)
        mat.xpd_coefficient = tf.Variable(0.2, dtype=rdtype, trainable=True)
        scene.get("large_reflector").radio_material = mat

        scene.tx_array = PlanarArray(num_rows=1,
                                    num_cols=1,
                                    vertical_spacing=0.5,
                                    horizontal_spacing=0.5,
                                    pattern="iso",
                                    polarization="V",
                                    polarization_model=2,
                                    dtype=dtype)

        scene.rx_array =  PlanarArray(num_rows=1,
                                    num_cols=1,
                                    vertical_spacing=0.5,
                                    horizontal_spacing=0.5,
                                    pattern="iso",
                                    polarization="V",
                                    polarization_model=2,
                                    dtype=dtype)


        tx = Transmitter(name="tx",
                        position=[-20., 0., 11.],
                        orientation=[0,0,0],
                        dtype=dtype)
        scene.add(tx)

        # To make sure that we get gradients for the scattering coefficient
        # through the reflection and scattered paths, we disable diffraction.
        with tf.GradientTape() as tape:
            cm = scene.coverage_map([0, 0, 0],
                                max_depth=3,
                                cm_cell_size=[0.5, 0.5],
                                cm_center=[0., 0., 3.],
                                cm_orientation=[0.,0.,0.],
                                cm_size=[0.5, 0.5],
                                num_samples=int(10e6),
                                los=True,
                                reflection=True,
                                diffraction=False,
                                edge_diffraction=False,
                                scattering=True)
            tf_cm = cm.path_gain[0]
            loss = tf.reduce_sum(tf_cm)

        watched_variables = tape.watched_variables()
        # Make sure we have all the trainable variables
        self.assertTrue(len(watched_variables) == 5)
        grads = tape.gradient(loss,watched_variables)
        # Make sure all the gradients are well defined
        self.assertFalse(None in grads)

        # Second test with diffraction enabled
        with tf.GradientTape() as tape:
            cm = scene.coverage_map([0, 0, 0],
                                max_depth=3,
                                cm_cell_size=[0.5, 0.5],
                                cm_center=[0., 0., 3.],
                                cm_orientation=[0.,0.,0.],
                                cm_size=[0.5, 0.5],
                                num_samples=int(10e6),
                                los=True,
                                reflection=True,
                                diffraction=True,
                                edge_diffraction=True,
                                scattering=True)
            tf_cm = cm.path_gain[0]
            loss = tf.reduce_sum(tf_cm)

        watched_variables = tape.watched_variables()
        # Make sure we have all the trainable variables
        self.assertTrue(len(watched_variables) == 5)
        grads = tape.gradient(loss,watched_variables)
        # Make sure all the gradients are well defined
        self.assertFalse(None in grads)

    def test_scattering_coefficient_gradient_sign_1(self):
        r"""
        Enables scattering and reflection only and sets the scattering
        coefficient to a value between 0 (excluded) and 1 (excluded).
        The objective function is the power measured for a coverage map made of
        a single small cell.
        The gradient of the scattering coefficient should be negative as
        the reflected paths are stronger than the scattered paths, and therefore
        maximizing the energy requires reducing the scattering coefficient.
        """
        dtype=tf.complex64
        rdtype = dtype.real_dtype
        scene = load_scene(sionna.rt.scene.double_reflector, dtype=dtype)

        # Defines a radio material
        mat = RadioMaterial("my_mat",
                            relative_permittivity=1,
                            conductivity=1,
                            scattering_coefficient=0.7,
                            xpd_coefficient=0.2,
                            scattering_pattern=LambertianPattern(dtype=dtype),
                            dtype=dtype)
        mat.conductivity = tf.Variable(1.0, dtype=rdtype, trainable=True)
        mat.relative_permittivity = tf.Variable(1.0, dtype=rdtype, trainable=True)
        mat.scattering_coefficient = tf.Variable(0.7, dtype=rdtype, trainable=True)
        mat.xpd_coefficient = tf.Variable(0.2, dtype=rdtype, trainable=True)
        scene.get("large_reflector").radio_material = mat

        scene.tx_array = PlanarArray(num_rows=1,
                                    num_cols=1,
                                    vertical_spacing=0.5,
                                    horizontal_spacing=0.5,
                                    pattern="iso",
                                    polarization="V",
                                    polarization_model=2,
                                    dtype=dtype)

        scene.rx_array =  PlanarArray(num_rows=1,
                                    num_cols=1,
                                    vertical_spacing=0.5,
                                    horizontal_spacing=0.5,
                                    pattern="iso",
                                    polarization="V",
                                    polarization_model=2,
                                    dtype=dtype)

        tx = Transmitter(name="tx",
                        position=[-20., 0., 11.],
                        orientation=[0,0,0],
                        dtype=dtype)
        scene.add(tx)

        with tf.GradientTape() as tape:
            cm = scene.coverage_map([0, 0, 0],
                                max_depth=3,
                                cm_cell_size=[0.5, 0.5],
                                cm_center=[0., 0., 3.],
                                cm_orientation=[0.,0.,0.],
                                cm_size=[0.5, 0.5],
                                num_samples=int(10e6),
                                los=False,
                                reflection=True,
                                diffraction=False,
                                scattering=True)
            tf_cm = cm.path_gain[0]
            power = tf.reduce_sum(tf_cm)
        # Computes gradient with respect to the scattering coefficient only
        grads = tape.gradient(power, [mat.scattering_coefficient])[0]
        self.assertLess(grads, 0.0)

    def test_scattering_coefficient_gradient_sign_2(self):
        r"""
        Enables scattering only and sets the scattering coefficient to a value
        between 0 (excluded) and 1 (excluded).
        The objective function is the power measured for a coverage map made of
        a single small cell.
        The gradient of the scattering coefficient should be positive to
        increase the total power, as reflection is disabled.
        """
        dtype=tf.complex64
        rdtype = dtype.real_dtype
        scene = load_scene(sionna.rt.scene.double_reflector, dtype=dtype)

        # Defines a radio material
        mat = RadioMaterial("my_mat",
                            relative_permittivity=1,
                            conductivity=1,
                            scattering_coefficient=0.7,
                            xpd_coefficient=0.2,
                            scattering_pattern=LambertianPattern(dtype=dtype),
                            dtype=dtype)
        mat.conductivity = tf.Variable(1.0, dtype=rdtype, trainable=True)
        mat.relative_permittivity = tf.Variable(1.0, dtype=rdtype, trainable=True)
        mat.scattering_coefficient = tf.Variable(0.7, dtype=rdtype, trainable=True)
        mat.xpd_coefficient = tf.Variable(0.2, dtype=rdtype, trainable=True)
        scene.get("large_reflector").radio_material = mat

        scene.tx_array = PlanarArray(num_rows=1,
                                    num_cols=1,
                                    vertical_spacing=0.5,
                                    horizontal_spacing=0.5,
                                    pattern="iso",
                                    polarization="V",
                                    polarization_model=2,
                                    dtype=dtype)

        scene.rx_array =  PlanarArray(num_rows=1,
                                    num_cols=1,
                                    vertical_spacing=0.5,
                                    horizontal_spacing=0.5,
                                    pattern="iso",
                                    polarization="V",
                                    polarization_model=2,
                                    dtype=dtype)

        tx = Transmitter(name="tx",
                        position=[-20., 0., 11.],
                        orientation=[0,0,0],
                        dtype=dtype)
        scene.add(tx)

        with tf.GradientTape() as tape:
            cm = scene.coverage_map([0, 0, 0],
                                max_depth=3,
                                cm_cell_size=[0.5, 0.5],
                                cm_center=[0., 0., 3.],
                                cm_orientation=[0.,0.,0.],
                                cm_size=[0.5, 0.5],
                                num_samples=int(10e6),
                                los=False,
                                reflection=False,
                                diffraction=False,
                                scattering=True)
            tf_cm = cm.path_gain[0]
            power = tf.reduce_sum(tf_cm)
        # Computes gradient with respect to the scattering coefficient only
        grads = tape.gradient(power, [mat.scattering_coefficient])[0]
        self.assertGreater(grads, 0.0)

    def test_scattering_coefficient_gradient_sign_3(self):
        r"""
        Enables reflection only and sets the scattering coefficient to a value
        between 0 (excluded) and 1 (excluded).
        The objective function is the power measured for a coverage map made of
        a single small cell.
        The gradient of the scattering coefficient should be negative to
        increase the total power, as scattering is disabled.
        """
        dtype=tf.complex64
        rdtype = dtype.real_dtype
        scene = load_scene(sionna.rt.scene.double_reflector, dtype=dtype)

        # Defines a radio material
        mat = RadioMaterial("my_mat",
                            relative_permittivity=1,
                            conductivity=1,
                            scattering_coefficient=0.7,
                            xpd_coefficient=0.2,
                            scattering_pattern=LambertianPattern(dtype=dtype),
                            dtype=dtype)
        mat.conductivity = tf.Variable(1.0, dtype=rdtype, trainable=True)
        mat.relative_permittivity = tf.Variable(1.0, dtype=rdtype, trainable=True)
        mat.scattering_coefficient = tf.Variable(0.7, dtype=rdtype, trainable=True)
        mat.xpd_coefficient = tf.Variable(0.2, dtype=rdtype, trainable=True)
        scene.get("large_reflector").radio_material = mat

        scene.tx_array = PlanarArray(num_rows=1,
                                    num_cols=1,
                                    vertical_spacing=0.5,
                                    horizontal_spacing=0.5,
                                    pattern="iso",
                                    polarization="V",
                                    polarization_model=2,
                                    dtype=dtype)

        scene.rx_array =  PlanarArray(num_rows=1,
                                    num_cols=1,
                                    vertical_spacing=0.5,
                                    horizontal_spacing=0.5,
                                    pattern="iso",
                                    polarization="V",
                                    polarization_model=2,
                                    dtype=dtype)

        tx = Transmitter(name="tx",
                        position=[-20., 0., 11.],
                        orientation=[0,0,0],
                        dtype=dtype)
        scene.add(tx)

        with tf.GradientTape() as tape:
            cm = scene.coverage_map([0, 0, 0],
                                max_depth=3,
                                cm_cell_size=[0.5, 0.5],
                                cm_center=[0., 0., 3.],
                                cm_orientation=[0.,0.,0.],
                                cm_size=[0.5, 0.5],
                                num_samples=int(10e6),
                                los=False,
                                reflection=True,
                                diffraction=False,
                                scattering=False)
            tf_cm = cm.path_gain[0]
            power = tf.reduce_sum(tf_cm)
        # Computes gradient with respect to the scattering coefficient only
        grads = tape.gradient(power, [mat.scattering_coefficient])[0]
        self.assertLess(grads, 0.0)
