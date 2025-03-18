#
# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0#

import unittest
import numpy as np
import tensorflow as tf
from sionna.sys.topology import HexGrid
from sionna.phy.channel.tr38901 import TDL
from sionna.phy.ofdm import ResourceGrid, ResourceGridMapper, OFDMModulator, OFDMDemodulator, LSChannelEstimator
from sionna.phy import config, dtypes
from sionna.phy.utils import flatten_dims
from sys_utils import wraparound_dist_np


class TestHexagonalGrid(unittest.TestCase):
    
    def test_hexagonal_grid(self):
        """ Checks that the centers are aligned with pre-computed ones """
        grid = HexGrid(cell_radius=4, num_rings=3, center_loc=(-2,3), precision='double')
        grid.cell_radius = 1
        grid.num_rings = 2
        grid.center_loc = (0, 0)
        centers_precomputed = np.array([(0.0, 0.0),
                               (-1.5, 0.8660254037844386),
                               (0.0, 1.7320508075688772),
                               (1.5, 0.8660254037844386),
                               (1.5, -0.8660254037844386),
                               (0.0, -1.7320508075688772),
                               (-1.5, -0.8660254037844386),
                               (-3.0, 1.7320508075688772),
                               (-1.5, 2.598076211353316),
                               (0.0, 3.4641016151377544),
                               (1.5, 2.598076211353316),
                               (3.0, 1.7320508075688772),
                               (3.0, 0.0),
                               (3.0, -1.7320508075688772),
                               (1.5, -2.598076211353316),
                               (0.0, -3.4641016151377544),
                               (-1.5, -2.598076211353316),
                               (-3.0, -1.7320508075688772),
                               (-3.0, 0.0)])
        centers = grid.cell_loc.numpy()[:, :2]
        is_found = np.zeros(len(centers))
        for c in centers:
            dist_c_centers = np.linalg.norm(np.array([c]) - centers_precomputed, axis=1)
            closest_center = np.argmin(dist_c_centers)
            self.assertAlmostEqual(dist_c_centers[closest_center], 0, delta=1e-5)  # center is found among pre-computed ones
            is_found[closest_center] = 1
        self.assertTrue(np.sum(is_found)==len(is_found))  # all centers have been found

    def test_drop_uts(self):
        """
        Validate ut locations from drop_uts method
        """
        isd = 50
        bs_height = 10
        num_rings = 1

        grid = HexGrid(isd=isd,
                    num_rings=num_rings,
                    cell_height=bs_height,
                    precision='double')
        # Drop users

        num_ut_per_sector = 100
        
        min_bs_ut_dist_vec = [20, 20, 20]
        max_bs_ut_dist_vec = [30, 35, 40]
        min_ut_height_vec = [1, 9, 12]
        max_ut_height_vec = [2, 11, 15]

        assert len(np.unique([len(min_bs_ut_dist_vec),
                              len(min_ut_height_vec),
                              len(max_ut_height_vec)])) == 1

        for ii in range(len(min_bs_ut_dist_vec)):
            # [batch_size, num_cells, 3, num_ut_per_sector, 3]
            ut_loc, *_ = grid(1,
                            num_ut_per_sector,
                            min_bs_ut_dist_vec[ii],
                            max_bs_ut_dist=max_bs_ut_dist_vec[ii],
                            min_ut_height=min_ut_height_vec[ii],
                            max_ut_height=max_ut_height_vec[ii])
            # [num_cells, num_ut_per_cell, 3]
            ut_loc = flatten_dims(ut_loc, num_dims=2, axis=2)[0, ::].numpy()

            cell_loc = grid.cell_loc.numpy()

            for cell in range(grid.num_cells):
                for ut in range(ut_loc.shape[1]):
                    ut_cell_dist_3d = np.linalg.norm(cell_loc[cell, :] - ut_loc[cell, ut, :])
                    ut_cell_dist_2d = np.linalg.norm(cell_loc[cell, :2] - ut_loc[cell, ut, :2])

                    # 2D UT-cell center distance must be at most ISD / sqrt(3)
                    self.assertLessEqual(ut_cell_dist_2d,grid.isd / np.sqrt(3))

                    # 3D UT-cell center distance must be >= min_bs_ut_dist
                    self.assertGreaterEqual(ut_cell_dist_3d, min_bs_ut_dist_vec[ii])

                    # 3D UT-cell center distance must be ,= min_bs_ut_dist
                    self.assertLessEqual(ut_cell_dist_3d, max_bs_ut_dist_vec[ii])

    def test_wraparound(self):
        """ Validate wraparound method against its non-TensorFlow
        version """ 

        def drop_uts(isd, num_rings, batch_size, num_ut_per_sector, 
                     min_bs_ut_dist, min_ut_height, max_ut_height):
            grid = HexGrid(isd=isd,
                           num_rings=num_rings,
                           precision='double')
            # Drop users
            ut_loc, cell_mirror_coord, wrap_dist_tf = \
                grid(batch_size,
                    num_ut_per_sector,
                    min_bs_ut_dist,
                    min_ut_height=min_ut_height,
                    max_ut_height=max_ut_height)
            return ut_loc, cell_mirror_coord, wrap_dist_tf
        
        @tf.function
        def drop_uts_graph(isd, num_rings, batch_size, num_ut_per_sector, 
                     min_bs_ut_dist, min_ut_height, max_ut_height):
            return drop_uts(isd, num_rings, batch_size, num_ut_per_sector, 
                     min_bs_ut_dist, min_ut_height, max_ut_height)

        @tf.function(jit_compile=True)
        def drop_uts_xla(isd, num_rings, batch_size, num_ut_per_sector, 
                     min_bs_ut_dist, min_ut_height, max_ut_height):
            return drop_uts(isd, num_rings, batch_size, num_ut_per_sector, 
                     min_bs_ut_dist, min_ut_height, max_ut_height)
        
        fun_dict = {'eager': drop_uts,
                    'graph': drop_uts_graph,
                    'xla': drop_uts_xla}
        batch_size = 1
        num_ut_per_sector = 5
        min_bs_ut_dist = 20
        isd = 50
        bs_height = 10
        min_ut_height = 1
        max_ut_height = 2
        num_rings = 1

        # generate grid also outside fun, since XLA does not allow for
        # non-tensor outputs
        grid = HexGrid(isd=isd,
                           num_rings=num_rings,
                           precision='double')
        for mode, fun in fun_dict.items():
            ut_loc, cell_mirror_coord, wrap_dist_tf = fun(
                isd, num_rings,
                batch_size, num_ut_per_sector,
                min_bs_ut_dist, min_ut_height, max_ut_height)
            
            # [..., 2]
            ut_loc = flatten_dims(ut_loc, num_dims=4, axis=0).numpy()
            # [..., num_cells, 2]
            cell_mirror_coord = flatten_dims(cell_mirror_coord, num_dims=4, axis=0).numpy()
            # [..., num_cells]
            wrap_dist_tf = flatten_dims(wrap_dist_tf, num_dims=4, axis=0).numpy()

            # Compare wraparound distance against the Numpy version
            for ut in range(ut_loc.shape[0]):
                wrap_dist_np_vec = wraparound_dist_np(grid, ut_loc[ut, :])
                for cell in range(grid.num_cells):
                    wrap_dist_np1 = np.linalg.norm(cell_mirror_coord[ut, cell, :] - ut_loc[ut, :])
                    self.assertAlmostEqual(wrap_dist_np_vec[cell], wrap_dist_np1, delta=1e-5)
                    self.assertAlmostEqual(wrap_dist_np_vec[cell], wrap_dist_tf[ut, cell], delta=1e-5)

