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

from sionna.rt import load_scene, Transmitter, Receiver, PlanarArray, CoverageMap

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
                                     pattern="iso",                             polarization="V")
        scene.rx_array = PlanarArray(num_rows=4,
                                     num_cols=4,
                                     vertical_spacing=0.5,  horizontal_spacing=0.5,
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
                            num_samples=int(1e6),
                            seed=42)

        ### centering is True
        pos = cm.sample_positions(
                    batch_size,
                    min_gain_db=-110,
                    #max_gain_db=-100,
                    #min_dist=100,
                    #max_dist=250,
                    tx=0,
                    center_pos=True
                    )
        pos = pos.numpy()
        cpos = cm.cell_centers.numpy()

        for i in range(batch_size):
            # find closest point in coverage map
            success = False
            for j in range(cpos.shape[0]):
                for k in range(cpos.shape[1]):
                    d = np.abs(pos[i]-cpos[j,k])
                    if np.sum(d)==0.:
                        print("success")
                        success=True
                        break
                if success:
                    break
            self.assertTrue(success) # "position not centered on grid"

        ### centering is False
        pos = cm.sample_positions(
                    batch_size,
                    min_gain_db=-110,
                    #max_gain_db=-100,
                    #min_dist=100,
                    #max_dist=250,
                    tx=0,
                    center_pos=False
                    )
        pos = pos.numpy()
        cpos = cm.cell_centers.numpy()

        for i in range(batch_size):
            # find closest point in coverage map
            success = False
            for j in range(cpos.shape[0]):
                for k in range(cpos.shape[1]):
                    d = np.abs(pos[i]-cpos[j,k])
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

        pos = cm.sample_positions(
                    batch_size,
                    min_dist=d_min,
                    max_dist=d_max,
                    tx=0,
                    center_pos=False)
        pos = pos.numpy()

        for i in range(batch_size):
            d = np.sqrt(np.sum((pos[i]-tx_pos)**2))
            valid_dist = True
            if d<(d_min-d_cell):
                valid_dist = False
            if d>(d_max+d_cell):
                valid_dist = False
            self.assertTrue(valid_dist)

    def test_dtype(self):
        """test against different dtypes"""

        cm_cell_size = np.array([1., 1.])
        batch_size = 100
        tx_pos = np.array([0.,0.,0.])

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
                                        pattern="iso",                          polarization="V",
                                        dtype=dt)
            scene.rx_array = PlanarArray(num_rows=1,
                                        num_cols=1,
                                        vertical_spacing=0.5,  horizontal_spacing=0.5,
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
                                cm_cell_size=cm_cell_size)

            # and sample positions
            pos = cm.sample_positions(
                        batch_size,
                        min_gain_db=-110,
                        max_gain_db=-100,
                        min_dist=1,
                        max_dist=25,
                        tx=0,
                        center_pos=True)
