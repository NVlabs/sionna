#
# SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Unit tests for profile interpolation"""

import unittest
import numpy as np
import tensorflow as tf
from sionna.rt import CellGrid, DiscreteProfile


def nmse(x, x_hat):
    """Computes the NMSE between two input tensors"""
    return 20*np.log10(tf.reduce_mean(tf.abs(x-x_hat))) - 20*np.log10(tf.reduce_mean(tf.abs(x)))

def compute_derivatives(f, y_val, z_val, dtype=tf.complex64):
    """
    Computes the first and second order derivatives, including mixed derivatives,
    of a function f(y, z) with respect to y and z.

    Parameters:
    - f: A callable representing a function f(y, z).
    - y_val: The value of y at which to evaluate the derivatives.
    - z_val: The value of z at which to evaluate the derivatives.

    Returns:
    A dictionary with keys 'dy', 'dz', 'd2y', 'd2z', and 'd2y_dz' representing
    the first-order derivatives, the second-order derivatives, and the mixed
    second-order derivative, respectively.
    """
    # Convert inputs to TensorFlow variables if they are not already
    y = tf.Variable(y_val, dtype=dtype.real_dtype)
    z = tf.Variable(z_val, dtype=dtype.real_dtype)

    with tf.GradientTape(persistent=True) as tape2:
        tape2.watch([y, z])
        with tf.GradientTape(persistent=True) as tape1:
            tape1.watch([y, z])
            value = f(y, z)
        dy, dz = tape1.gradient(value, [y, z])
    d2y = tape2.gradient(dy, y)
    d2z = tape2.gradient(dz, z)
    d2y_dz = tape2.gradient(dy, z)  # Mixed partial derivative
    if dy is None:
        dy = tf.zeros_like(y_val)
    if dz is None:
        dz = tf.zeros_like(y_val)
    if d2y is None:
        d2y = tf.zeros_like(y_val)
    if d2z is None:
        d2z = tf.zeros_like(y_val)
    if d2y_dz is None:
        d2y_dz = tf.zeros_like(y_val)
    return {'dy': dy, 'dz': dz, 
            'd2y': d2y, 'd2z': d2z, 'd2y_dz': d2y_dz}

def interpolate(profile_fun, scales, dtype):
    """Creates a discrete profile from a profile function
    and returns interpolated values, grads, and Hessians
    together with analytical and ground truth.
    """
    # Create a cell grid
    num_rows = 20
    num_cols = 30
    num_modes = len(scales)
    cell_grid = CellGrid(num_rows, num_cols, dtype)
    
    # Create discrete profile
    profile = DiscreteProfile(cell_grid=cell_grid,
                              num_modes=num_modes,
                              dtype=dtype)

    # Assign profile values 
    y_supp, z_supp = tf.meshgrid(profile._interpolator.cell_y_positions,
                                 profile._interpolator.cell_z_positions)

    profile_values = []
    for scale in scales:
        profile_values.append(profile_fun(y_supp, z_supp, scale))
    profile.values = tf.stack(profile_values, axis=0)
    f = profile.values

    derivatives = []
    for scale in scales:
        ff = lambda y, z: profile_fun(y, z, scale) 
        derivatives.append(compute_derivatives(ff, y_supp, z_supp, dtype))

    # Compute a meshgrid on which the profile should be interpolated
    num_int_steps = 200
    y_int = tf.cast(np.linspace(profile._interpolator.cell_y_positions[0],
                                profile._interpolator.cell_y_positions[-1],
                                num_int_steps), dtype.real_dtype)
    z_int = tf.cast(np.linspace(profile._interpolator.cell_z_positions[0],
                                profile._interpolator.cell_z_positions[-1],
                                num_int_steps), dtype.real_dtype)
    y_int, z_int = np.meshgrid(y_int, z_int)

    # Flatten meshgrid into a batch of positions
    sample_positions = tf.stack([tf.reshape(y_int, [-1]),
                                 tf.reshape(z_int, [-1])], axis=-1)

    # Compute interpolated profile together with gradients and Hessians
    f_int, grads, hessians = profile(sample_positions,
                                     mode=None,
                                     return_grads=True)
    # Reshape to meshgrid dimensions
    f_int = tf.reshape(f_int, [num_modes, num_int_steps, num_int_steps])
    grads = tf.reshape(grads, [num_modes, num_int_steps, num_int_steps, 3])
    hessians = tf.reshape(hessians, [num_modes, num_int_steps, num_int_steps, 3, 3])

    # Compute ground truth for interpolated profile
    f_ground_truth = []
    for scale in scales:
        f_ground_truth.append(profile_fun(y_int, z_int, scale))
    f_ground_truth = tf.stack(f_ground_truth, axis=0)

    derivatives_ground_truth = []
    for scale in scales:
        ff = lambda y, z: profile_fun(y, z, scale) 
        derivatives_ground_truth.append(compute_derivatives(ff, y_int, z_int, dtype))

    return f_ground_truth, derivatives_ground_truth, f_int, grads, hessians


class TestProfileInterpolation(unittest.TestCase):
    """Unit test for profile interpolation, gradients, and Hessians"""

    def test_interpolation(self):
        """Test for a profile with multiple modes and for both dtypes"""

        def profile_fun(y, z, scale=1.):
            y = y*scale
            z = z*scale
            return y**3 + z**3 - y*z + y**2

        scales = [0.01, 0.1, 1.0]

        for dtype in [tf.complex64, tf.complex128]:
            res = interpolate(profile_fun, scales, dtype)
            f_ground_truth = res[0]
            derivatives_ground_truth = res[1]
            f_int = res[2]
            grads = res[3]
            hessians = res[4]

            for mode in range(len(scales)):
                # Interpolation
                self.assertLess(nmse(f_ground_truth[mode],
                                     f_int[mode]), -60)
                # First derivative w.r.t. dy
                self.assertLess(nmse(derivatives_ground_truth[mode]["dy"],
                                     grads[mode,:,:,1]), -40)
                # First derivative w.r.t. dz
                self.assertLess(nmse(derivatives_ground_truth[mode]["dz"],
                                     grads[mode,:,:,2]), -40)
                # 2nd-order derivative w.r.t. dy^2
                self.assertLess(nmse(derivatives_ground_truth[mode]["d2y"],
                                     hessians[mode,:,:,1,1]), -24)
                # 2nd-order derivative w.r.t. dz^2
                self.assertLess(nmse(derivatives_ground_truth[mode]["d2z"],
                                     hessians[mode,:,:,2,2]), -24)
                # 2nd-order derivative w.r.t. dy*dz
                self.assertLess(nmse(derivatives_ground_truth[mode]["d2y_dz"],
                                     hessians[mode,:,:,1,2]), -24)
                self.assertLess(nmse(derivatives_ground_truth[mode]["d2y_dz"],
                                     hessians[mode,:,:,2,1]), -24)
