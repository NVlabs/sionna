#
# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Layer for utility functions needed for Convolutional Codes."""

import numpy as np
import tensorflow as tf
from sionna.fec.utils import int2bin, bin2int

def polynomial_selector(rate, constraint_length):
    """Returns generator polynomials for given code parameters. The
    polynomials are chosen from [Moon] which are tabulated by searching
    for polynomials with best free distances for a given rate and
    constraint length.

    Input
    -----
        rate: float
            A float defining the desired rate of the code.

        constraint_length: int
            An integer defining the desired constraint length of the encoder.

    Output
    ------
        : tuple
            Tuple of strings with each string being a 0,1 sequence where
            each polynomial is represented in binary form.

    """
    rate_half_dict = {
            3: ('101', '111'), # (5,7)
            4: ('1101', '1111'), # (64, 74)
            5: ('10011', '11101'), # (46, 72)
            6: ('110101', '101111'), # (65, 57)
            7: ('1011011', '1111001'), # (554, 744)
            8: ('11100101', '10011111'), # (712, 476)
    }
    rate_third_dict = {
            3: ('101', '111', '111'), # (5,7,7)
            4: ('1011', '1101', '1111'),# (54, 64, 74)
            5: ('10101', '11011', '11111'), # (52, 66, 76)
            6: ('100111', '101011', '111101'), # (47,53,75)
            7: ('1111001','1100101','1011011'), # (554, 744)
            8: ('10010101', '11011001', '11110111') # (452, 662, 756)
    }

    gen_poly_dict = {
            1/2: rate_half_dict,
            1/3: rate_third_dict
    }
    gen_poly = gen_poly_dict[rate][constraint_length]
    return gen_poly


class Trellis(object):
    """Trellis(gen_poly)

    Trellis structure for a given generator polynomial. Defines
    state transitions and output symbols (and bits) for each current
    state and input.

    Parameters
    ----------
        gen_poly: tuple
            sequence of strings with each string being a 0,1 sequence. If None,
            ``rate`` and ``constraint_length`` must be provided.

    """
    def __init__(self, gen_poly):
        self.gen_poly = gen_poly
        self.constraint_length = len(self.gen_poly[0])

        self.conv_k = 1
        self.conv_n = len(self.gen_poly)
        self.ni = 2**self.conv_k
        self.ns = 2**(self.constraint_length-1)

        #For current state i and input j, state transitions i->to_nodes[i][j]
        self.to_nodes = None

        #For current state i, valid state transitions are from_nodes[i][:]-> i
        self.from_nodes = None

        # Given states i and j, Trellis emits ops[i][j] symbol if neq -1
        self.op_mat = None

        # Given next state as i, trellis emits op_by_tonode[i][:] symbols
        self.op_by_tonode = None

        self._generate_transitions()

    def _binary_matmul(self, st):
        """
        For a given state st, this method multiplies each generator
        polynomial with st and returns the sum modulo 2 bit as output
        """
        op = np.zeros(self.conv_n, int)
        assert len(st) == len(self.gen_poly[0])
        for i, poly in enumerate(self.gen_poly):
            op_int = sum(
                [int(char)*int(poly[idx]) for idx,char in enumerate(st)])
            op[i] = int2bin(op_int % 2, 1)[0]
        return op

    def _generate_transitions(self):
        """Utility method that generates state transitions for different
        input symbols. This depends only on constraint_length and independent
        of the generator polynomials.
        """
        to_nodes = np.full((self.ns, self.ni), -1, int)
        from_nodes = np.full((self.ns, self.ni), -1, int)
        op_mat = np.full((self.ns, self.ns), -1, int)
        op_by_tonode =  np.full((self.ns, self.ni), -1, int)

        from_nodes_ctr = np.zeros(self.ns, int)
        for i in range(self.ni):
            ip_bits = int2bin(i, self.conv_k)
            for j in range(self.ns):
                state_bits = ip_bits + int2bin(j, self.constraint_length-1)
                j_to = bin2int(state_bits[:-1])

                to_nodes[j][i] = j_to
                from_nodes[j_to][from_nodes_ctr[j_to]] = j

                op_bits = self._binary_matmul(state_bits)
                op_sym = bin2int(op_bits)
                op_mat[j, j_to] = op_sym
                op_by_tonode[j_to, from_nodes_ctr[j_to]] = op_sym
                from_nodes_ctr[j_to] += 1

        self.to_nodes = tf.convert_to_tensor(to_nodes, dtype=tf.int32)
        self.from_nodes = tf.convert_to_tensor(from_nodes, dtype=tf.int32)
        self.op_mat = tf.convert_to_tensor(op_mat, dtype=tf.int32)
        self.op_by_tonode = tf.convert_to_tensor(op_by_tonode, dtype=tf.int32)
