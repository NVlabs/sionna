#
# SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
import pytest
import unittest
import numpy as np
import tensorflow as tf
from sionna.constants import PI
from sionna.rt import ScatteringPattern, LambertianPattern, DirectivePattern, BackscatteringPattern
from sionna.rt.utils import r_hat

@tf.function(jit_compile=True)
def integrate_pattern(pattern, k_i, dtype):
    """
    Integrates an scattering pattern over the hemisphere.
    SHould be close to one for all inputs.
    """
    n_hat = tf.constant((0,0,1), dtype.real_dtype)

    theta = tf.cast(tf.linspace(0.0, PI/2, 5000), dtype.real_dtype)
    phi = tf.cast(tf.linspace(-PI, PI, 5000), dtype.real_dtype)
    delta_theta = theta[1]-theta[0]
    delta_phi = phi[1]-phi[0]
    theta_grid, phi_grid = tf.meshgrid(theta, phi, indexing='ij')

    theta_s =tf.reshape(theta_grid, [-1])
    phi_s = tf.reshape(phi_grid, [-1])
    k_s = r_hat(theta_s, phi_s)
    k_i = tf.broadcast_to(k_i, k_s.shape)
    n_hat = tf.broadcast_to(n_hat, k_s.shape)

    return tf.reduce_sum(pattern(k_i, k_s, n_hat) * tf.sin(theta_s) * delta_theta * delta_phi)

class TestNormalization(unittest.TestCase):
    """Test validating that the scattering patterns integrate to one
       over the hemisphere for different angles of arrival.
    """

    @pytest.mark.usefixtures("only_gpu")
    def test_lambertian_pattern(self):
        dtypes = [tf.complex64]
        k_is = [[-0.7071,0., -0.7071],
                [0.7071,0., -0.7071],
                [0, -0.7071, -0.7071],
                [0, 0.7071, -0.7071],
                [0,0,-1]]
        for dtype in dtypes:
            for k_i in k_is:
                k_i = tf.constant(k_i, dtype.real_dtype)
                pattern = LambertianPattern(dtype=dtype)
                res = integrate_pattern(pattern, k_i, dtype)
                self.assertTrue(np.abs(res-1)<1e-3)

    @pytest.mark.usefixtures("only_gpu")
    def test_directive_pattern(self):
        dtypes = [tf.complex64]
        k_is = [[-0.7071,0., -0.7071],
                [0.7071,0., -0.7071],
                [0, -0.7071, -0.7071],
                [0, 0.7071, -0.7071],
                [0,0,-1]]
        alpha_rs = [1,5,10,30,100]
        for dtype in dtypes:
            for k_i in k_is:
                for alpha_r in alpha_rs:
                    k_i = tf.constant(k_i, dtype.real_dtype)
                    pattern = DirectivePattern(alpha_r,dtype=dtype)
                    res = integrate_pattern(pattern, k_i, dtype)
                    self.assertTrue(np.abs(res-1)<1e-2)

    @pytest.mark.usefixtures("only_gpu")
    def test_backscattering_pattern(self):
        dtypes = [tf.complex64]
        k_is = [[-0.7071,0., -0.7071],
                [0.7071,0., -0.7071],
                [0, -0.7071, -0.7071],
                [0, 0.7071, -0.7071],
                [0,0,-1]]
        alpha_rs = [1,5,10,30,100]
        alpha_is = [1,9,12,20,90]
        lambdas = [0.5, 0.3, 0.4, 0.8, 0.6]
        for dtype in dtypes:
            for k_i in k_is:
                for i, alpha_r in enumerate(alpha_rs):
                    k_i = tf.constant(k_i, dtype.real_dtype)
                    pattern = BackscatteringPattern(alpha_r, alpha_is[i],
                                                    lambdas[i], dtype=dtype)
                    res = integrate_pattern(pattern, k_i, dtype)
                    self.assertTrue(np.abs(res-1)<1e-2)
