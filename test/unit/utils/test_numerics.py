#
# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0#

import unittest
import numpy as np
import tensorflow as tf

from sionna.phy.utils import bisection_method
from sionna.phy import config


@tf.function(jit_compile=True)
def bisection_method_xla(f,
                         left,
                         right,
                         regula_falsi=False,
                         expand_to_left=True,
                         expand_to_right=True,
                         step_expand=2.,
                         eps_x=1e-5,
                         eps_y=1e-5,
                         max_n_iter=100,
                         return_brackets=False,
                         precision=None,
                         **kwargs):
    return bisection_method(f,
                            left,
                            right,
                            regula_falsi=regula_falsi,
                            expand_to_left=expand_to_left,
                            expand_to_right=expand_to_right,
                            step_expand=step_expand,
                            eps_x=eps_x,
                            eps_y=eps_y,
                            max_n_iter=max_n_iter,
                            return_brackets=return_brackets,
                            precision=precision,
                            **kwargs)


class TestNumerics(unittest.TestCase):

    def test_bisection(self):
        """
        Validate bisection_method
        """
        def f1(x, a, b):
            return a - b * x

        def f2(x, a, b):
            return - tf.abs(b) * tf.math.pow(x - a, 3)

        batch_size = [50, 50]
        par = {f1: {}, f2: {}}
        # Prameters for f1
        par[f1]['a'] = config.tf_rng.uniform(batch_size, minval=-.5, maxval=1)
        par[f1]['b'] = config.tf_rng.uniform(batch_size, minval=0, maxval=1)

        # Parameters for f2
        par[f2]['a'] = config.tf_rng.uniform(batch_size, minval=-.5, maxval=1.5)
        par[f2]['b'] = config.tf_rng.uniform(batch_size, minval=.1, maxval=10)

        left = tf.fill(batch_size, 0.)
        right = tf.fill(batch_size, 1.)
        eps_x = 1e-4
        eps_y = 1e-5
        precision = 'single'

        # ------------------------ #
        # W/O bracketing expansion #
        # ------------------------ #
        for f in [f1, f2]:
            for fun in [bisection_method, bisection_method_xla]:

                x_opt, _, left_opt, right_opt = fun(
                    f,
                    left,
                    right,
                    eps_x=eps_x,
                    eps_y=eps_y,
                    expand_to_left=False,
                    expand_to_right=False,
                    max_n_iter=10000,
                    regula_falsi=False,
                    return_brackets=True,
                    precision=precision,
                    a=par[f]['a'],
                    b=par[f]['b'])

                # If f(right) >= 0, then must return right
                ind = tf.where(f(right, par[f]['a'], par[f]['b']) >= 0)
                self.assertTrue(
                    tf.reduce_all(tf.gather_nd((x_opt == right), ind)))

                # If f(left) <= 0, then must return left
                ind = tf.where(f(left, par[f]['a'], par[f]['b']) <= 0)
                self.assertTrue(
                    tf.reduce_all(tf.gather_nd((x_opt == left), ind)))

                # Else, it returns a point within the bounds with f roughly 0
                ind_middle = tf.where((f(left, par[f]['a'], par[f]['b']) > 0)
                                    & (f(right, par[f]['a'], par[f]['b']) < 0))
                f_x_opt_middle = tf.gather_nd(f(x_opt, par[f]['a'], par[f]['b']), ind_middle)
                right_opt_middle = tf.gather_nd(right_opt, ind_middle)
                left_opt_middle = tf.gather_nd(left_opt, ind_middle)
                self.assertTrue(
                    tf.reduce_all((tf.abs(f_x_opt_middle) < eps_y) |
                                (tf.abs(right_opt_middle - left_opt_middle) < eps_x)))

        # ------------------------- #
        # WITH bracketing expansion #
        # ------------------------- #
        for f in [f1, f2]:
            for fun in [bisection_method, bisection_method_xla]:

                x_opt, _, left_opt, right_opt = fun(
                    f,
                    left,
                    right,
                    eps_x=eps_x,
                    eps_y=eps_y,
                    expand_to_left=True,
                    expand_to_right=True,
                    max_n_iter=10000,
                    regula_falsi=False,
                    return_brackets=True,
                    precision=precision,
                    a=par[f]['a'],
                    b=par[f]['b'])
                # All rots must be found
                f_x_opt = f(x_opt, par[f]['a'], par[f]['b'])
                if f==f1:
                    root = par[f1]['a'] / par[f1]['b']
                else:
                    root = par[f2]['a']
                self.assertTrue(
                    tf.reduce_all((tf.abs(f_x_opt) < eps_y) |
                                (tf.abs(x_opt - root) < eps_x)))
