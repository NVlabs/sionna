#
# SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import unittest
import numpy as np
import tensorflow as tf
from sionna.nr import LayerMapper, LayerDemapper
from sionna.utils import BinarySource, hard_decisions
from sionna.mapping import Mapper, Demapper


class TestLayerMapper(unittest.TestCase):
    """Tests for LayerMapper"""

    def test_ref(self):
        """Test against predefined sequences."""

        # single layer
        u = np.array([[1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0,
                       0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0,
                       0, 1, 1, 0, 1, 0,1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0,
                       1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1,
                       0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1,
                       1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1,
                       0, 1, 0, 0, 1, 1]])

        o1 = np.array([[[1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1,
                         0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1,
                         1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1,
                         1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0,
                         1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1,
                         1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0,
                         0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1]]])

        o2 = np.array([[[1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1,
                         1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1,
                         1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1,
                         0, 1, 1, 0, 0, 1], [1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0,
                         1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1,
                         1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0,
                         1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1]]])

        o3 = np.array([[[1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1,
                         1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1,
                         0, 1, 0, 0], [1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0,
                         1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0,
                         1, 1, 0, 0, 1, 1, 1, 1, 1], [0, 1, 0, 1, 0, 1, 1, 1,
                         1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1,
                         0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1]]])

        o4=np.array([[[1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1,
                       0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0], [1, 1, 0, 1, 1, 0, 1,
                       1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0,
                       1, 1, 1, 0], [0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1,
                       0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1], [1, 0,
                       1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0,
                       1, 0, 1, 1, 0, 0, 1, 1, 1]]])

        # 1 layer
        mapper = LayerMapper(num_layers=1)
        o = mapper(u)
        self.assertTrue(np.array_equal(o.numpy(), o1))

        # 2 layer
        mapper = LayerMapper(num_layers=2)
        o = mapper(u)
        self.assertTrue(np.array_equal(o.numpy(), o2))

        # 3 layer
        mapper = LayerMapper(num_layers=3)
        o = mapper(u)
        self.assertTrue(np.array_equal(o.numpy(), o3))

        # 4 layer
        mapper = LayerMapper(num_layers=4)
        o = mapper(u)
        self.assertTrue(np.array_equal(o.numpy(), o4))

        ####### dual codeword scenario ######

        # 5 layer (dual codeword)
        u1 = np.array([[0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0,
                        0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0,
                        0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1,
                        0, 0, 0, 1, 1, 1]])

        u2 = np.array([[1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1,
                        1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1,
                        1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1,
                        1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1,
                        0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0]])

        o5 = np.array([[[0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0,
                         0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1], [1, 0, 0, 1, 0,
                         1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1,
                         0, 1, 1, 1, 0, 1, 1], [1, 1, 1, 1, 0, 0, 1, 0, 0, 0,
                         1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0,
                         0, 0], [0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1,
                         0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0], [1, 1,
                         1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0,
                         1, 1, 1, 1, 1, 0, 1, 0, 1, 0]]])

        mapper = LayerMapper(num_layers=5)
        o = mapper([u1, u2])
        self.assertTrue(np.array_equal(o.numpy(), o5))

        # 6 layer (dual codeword)
        u1 = np.array([[1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1,
                        0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0,
                        0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1,
                        1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1,
                        0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0]])


        u2 = np.array([[0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0,
                        0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0,
                        1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1,
                        1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1,
                        0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0]])

        o6 = np.array([[[1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1,
                         1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1], [0, 1, 1, 1, 0,
                         1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0,
                         0, 1, 0, 1, 1, 0, 1], [0, 1, 1, 0, 1, 1, 1, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1,
                         0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0,
                         1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1], [1, 1,
                         1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1,
                         1, 0, 0, 0, 0, 0, 0, 1, 1, 0], [1, 1, 0, 1, 1, 0, 0,
                         1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0,
                         1, 0, 0, 1, 0]]])

        mapper = LayerMapper(num_layers=6)
        o = mapper([u1, u2])
        self.assertTrue(np.array_equal(o.numpy(), o6))


        # 7 layer (dual codeword)
        u1 = np.array([[1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1,
                        1, 1, 1, 1,0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1,
                        0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0,
                        0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1,
                        1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1]])

        u2 = np.array([[0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0,
                        1, 1, 1, 1,0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1,
                        0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1,
                        1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1,
                        1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1,
                        0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0,
                        0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0]])

        o7 = np.array([[[1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1,
                      0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0], [0, 1, 0, 1, 1, 0,
                      1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1,
                      0, 0, 1, 0, 1], [1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1,
                      1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1], [0,
                      1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0,
                      1, 1, 1, 0, 1, 1, 1, 0, 0, 0], [0, 0, 0, 0, 0, 1, 1, 1,
                      1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0,
                      0, 1, 0], [1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0,
                      1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1], [1, 1, 1,
                      1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1,
                      0, 0, 0, 1, 1, 1, 1, 0]]])

        mapper = LayerMapper(num_layers=7)
        o = mapper([u1, u2])
        self.assertTrue(np.array_equal(o.numpy(), o7))

        # 8 layer (dual codeword)
        u1 = np.array([[0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0,
                        0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0,
                        1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1,
                        0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1,
                        1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0,
                        0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0,
                        1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1]])

        u2 = np.array([[0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0,
                        0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1,
                        0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0,
                        1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1,
                        1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
                        1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0,
                        1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1]])

        o8 = np.array([[[0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0,
                      1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0], [0, 1, 0, 0, 0, 0,
                      1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1,
                      1, 0, 0, 1, 1], [0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0,
                      0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0], [1,
                      0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0,
                      1, 0, 1, 1, 1, 0, 0, 1, 1, 1], [0, 1, 1, 1, 1, 0, 1, 0,
                      1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0,
                      1, 1, 0], [1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1,
                      0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1], [0, 0, 0,
                      0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0,
                      1, 0, 0, 1, 1, 0, 0, 1], [1, 0, 1, 0, 0, 1, 1, 1, 1, 0,
                      1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1,
                      1]]])

        mapper = LayerMapper(num_layers=8)
        o = mapper([u1, u2])
        self.assertTrue(np.array_equal(o.numpy(), o8))


class TestLayerDemapper(unittest.TestCase):
    """Tests for LayerDeMapper"""

    def test_identity(self):
        """Test that original sequence can be recovered."""

        bs = 7
        source = BinarySource()

        # single cw
        num_layers = [1,2,3,4]
        n = np.prod(num_layers)
        for l in num_layers:
            mapper = LayerMapper(num_layers=l)
            demapper = LayerDemapper(mapper)

            x = source([bs, n])
            y = mapper(x)
            z = demapper(y)

            self.assertTrue(np.array_equal(x.numpy(), z.numpy()))

        # dual cw (only available for PDSCH)
        num_layers = [5, 6, 7, 8]
        n0s = [60, 90, 90, 120]
        n1s = [90, 90, 120, 120]
        n = np.prod(num_layers)
        for l, n0, n1 in zip(num_layers, n0s, n1s):
            mapper = LayerMapper(num_layers=l)
            demapper = LayerDemapper(mapper)

            x0 = source([bs, n0])
            x1 = source([bs, n1])

            y = mapper([x0, x1])
            z0, z1 = demapper(y)

            self.assertTrue(np.array_equal(x0.numpy(), z0.numpy()))
            self.assertTrue(np.array_equal(x1.numpy(), z1.numpy()))

    def test_higher_order(self):
        """Test LRRs are correctly grouped when demapping is applied."""

        bs  = 10
        mod_order = 4

        source = BinarySource()
        mapper = Mapper("qam", num_bits_per_symbol=mod_order)
        demapper = Demapper("maxlog", "qam",num_bits_per_symbol=mod_order)

        # single cw
        num_layers = [1,2,3,4]

        for l in num_layers:
            l_mapper = LayerMapper(num_layers=l)
            l_demapper = LayerDemapper(l_mapper, num_bits_per_symbol=mod_order)
            u = source((bs, 5, 7, 13*l*mod_order)) # arbitrary dimensions
            x = mapper(u)
            x_l = l_mapper(x)
            llr_l = demapper((x_l, 0.1))
            l_hat = l_demapper(llr_l)
            u_hat = hard_decisions(l_hat)

            self.assertTrue(np.array_equal(u.numpy(), u_hat.numpy()))

        # dual cw
        num_layers = [5, 6, 7, 8]
        n0s = [60, 90, 90, 120]
        n1s = [90, 90, 120, 120]
        n = np.prod(num_layers)
        for l, n0, n1 in zip(num_layers, n0s, n1s):
            lmapper = LayerMapper(num_layers=l)
            ldemapper = LayerDemapper(lmapper, num_bits_per_symbol=mod_order)

            u0 = source([bs, n0*mod_order*l])
            u1 = source([bs, n1*mod_order*l])
            x0 = mapper(u0)
            x1 = mapper(u1)

            y = lmapper([x0, x1])
            llr = demapper((y, 0.01))
            z0, z1 = ldemapper(llr)

            u_hat0 = hard_decisions(z0)
            u_hat1 = hard_decisions(z1)

            self.assertTrue(np.array_equal(u0.numpy(), u_hat0.numpy()))
            self.assertTrue(np.array_equal(u1.numpy(), u_hat1.numpy()))
