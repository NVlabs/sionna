#
# SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import unittest
import numpy as np
import tensorflow as tf
import sionna
from sionna.rt import load_scene, Transmitter, Receiver, PlanarArray,\
    BackscatteringPattern, LambertianPattern, ScatteringPattern

class TestSceneCallableProp(unittest.TestCase):

    def test_radio_material_callable_paths(self):
        """Test the callable that computes the radio material properties from
        the intersection point properties for paths.
        """
        ## Setup the scene

        # Load integrated scene
        scene = load_scene(sionna.rt.scene.floor_wall)

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

        # Create transmitter
        tx = Transmitter(name="tx",
                        position=[1.0,0.5,1.0])
        # Add transmitter instance to scene
        scene.add(tx)

        # Create a receiver
        rx = Receiver(name="rx",
                    position=[1.0,-0.5,1.0],
                    orientation=[0,0,0])
        # Add receiver instance to scene
        scene.add(rx)
        tx.look_at(rx) # Transmitter points towards receiver

        # Store the objects id
        objects_id = {'floor' : scene.objects['floor'].object_id,
                    'wall' : scene.objects['wall'].object_id}
        # Material properties to set for testing
        rms = {'floor' : 'itu_concrete',
            'wall' : 'itu_brick'}
        scs = {'floor' : tf.constant(np.sqrt(0.5), tf.float32),
            'wall' : tf.constant(np.sqrt(0.3), tf.float32)}
        xpd = {'floor' : tf.constant(np.sqrt(0.3), tf.float32),
            'wall' : tf.constant(np.sqrt(0.5), tf.float32)}

        ## Reference paths

        # Use objects radio materials
        scene.radio_material_callable = None
        # Set the materials
        scene.objects['floor'].radio_material = rms['floor']
        scene.objects['wall'].radio_material = rms['wall']
        scene.objects['floor'].radio_material.scattering_coefficient = scs['floor']
        scene.objects['wall'].radio_material .scattering_coefficient = scs['wall']
        scene.objects['floor'].radio_material.xpd_coefficient = xpd['floor']
        scene.objects['wall'].radio_material.xpd_coefficient = xpd['wall']
        sionna.config.seed=1
        paths = scene.compute_paths(max_depth=3, diffraction=True, edge_diffraction=True, scattering=True)
        a_ref = tf.squeeze(paths.a)

        ## Callable for radio material

        # Radio material callable
        class RadioMatCallable:

            def __init__(self, test_class):

                floor_id = objects_id['floor']
                wall_id = objects_id['wall']
                size = np.maximum(floor_id, wall_id) + 1
                indices = [[floor_id], [wall_id]]

                # Use the materials complex relative permittivities
                etas = {floor_id : scene.radio_materials[rms['floor']].complex_relative_permittivity,
                        wall_id : scene.radio_materials[rms['wall']].complex_relative_permittivity}
                s = {floor_id : scs['floor'],
                    wall_id : scs['wall']}
                x = {floor_id : xpd['floor'],
                    wall_id : xpd['wall']}
                self._etas = tf.scatter_nd(indices, [etas[floor_id], etas[wall_id]], [size])
                self._s = tf.scatter_nd(indices, [s[floor_id], s[wall_id]], [size])
                self._x = tf.scatter_nd(indices, [x[floor_id], x[wall_id]], [size])

                self.test_class = test_class

            def __call__(self, objects, points):

                self.test_class.assertTrue(objects.dtype == tf.int32)
                self.test_class.assertTrue(points.dtype == tf.float32)
                self.test_class.assertTrue(points.shape[:-1] == objects.shape)
                self.test_class.assertTrue(points.shape[-1] == 3)

                #Eliminate -1 from objects which don't work on CPU
                objects = tf.where(objects<0, 0, objects)

                s = tf.gather(self._s, objects)
                x = tf.gather(self._x, objects)
                e = tf.gather(self._etas, objects)

                return e, s, x

        # Assign the radio material callable to the scene
        scene.radio_material_callable = RadioMatCallable(self)
        # Set different radio materials to objects to ensure the test fails if the callable is not used
        scene.objects['floor'].radio_material = 'itu_wood'
        scene.objects['wall'].radio_material = 'itu_marble'
        # Compute paths powers
        sionna.config.seed=1
        paths = scene.compute_paths(max_depth=3, diffraction=True, edge_diffraction=True, scattering=True)
        a_call = tf.squeeze(paths.a)

        max_err = tf.reduce_max(tf.abs(a_call - a_ref))
        self.assertEqual(max_err, 0.0)

    def test_radio_material_callable_cm(self):
        """Test the callable that computes the radio material properties from
        the intersection point properties for coverage map.
        """
        ## Setup the scene

        # Load integrated scene
        scene = load_scene(sionna.rt.scene.floor_wall)

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

        # Create transmitter
        tx = Transmitter(name="tx",
                        position=[1.0,0.5,2.0])

        # Add transmitter instance to scene
        scene.add(tx)

        # Store the objects id
        objects_id = {'floor' : scene.objects['floor'].object_id,
                    'wall' : scene.objects['wall'].object_id}
        # Material properties to set for testing
        rms = {'floor' : 'itu_concrete',
            'wall' : 'itu_brick'}
        scs = {'floor' : tf.constant(np.sqrt(0.5), tf.float32),
            'wall' : tf.constant(np.sqrt(0.3), tf.float32)}
        xpd = {'floor' : tf.constant(np.sqrt(0.3), tf.float32),
            'wall' : tf.constant(np.sqrt(0.5), tf.float32)}

        ## Reference paths

        # Use objects radio materials
        scene.radio_material_callable = None
        # Set the materials
        scene.objects['floor'].radio_material = rms['floor']
        scene.objects['wall'].radio_material = rms['wall']
        scene.objects['floor'].radio_material.scattering_coefficient = scs['floor']
        scene.objects['wall'].radio_material .scattering_coefficient = scs['wall']
        scene.objects['floor'].radio_material.xpd_coefficient = xpd['floor']
        scene.objects['wall'].radio_material.xpd_coefficient = xpd['wall']
        # Compute the coverage map
        sionna.config.seed=1
        cm_ref = scene.coverage_map(max_depth=3, los=False, diffraction=True,
                                    edge_diffraction=True, scattering=True,
                                    reflection=True, cm_cell_size=(0.1, 0.1))
        cm_ref = cm_ref.path_gain[0]

        ## Callable for radio material

        # Radio material callable
        class RadioMatCallable:

            def __init__(self, test_class):

                floor_id = objects_id['floor']
                wall_id = objects_id['wall']
                size = np.maximum(floor_id, wall_id) + 1
                indices = [[floor_id], [wall_id]]

                # Use the materials complex relative permittivities
                etas = {floor_id : scene.radio_materials[rms['floor']].complex_relative_permittivity,
                        wall_id : scene.radio_materials[rms['wall']].complex_relative_permittivity}
                s = {floor_id : scs['floor'],
                     wall_id : scs['wall']}
                x = {floor_id : xpd['floor'],
                     wall_id : xpd['wall']}
                self._etas = tf.scatter_nd(indices, [etas[floor_id], etas[wall_id]], [size])
                self._s = tf.scatter_nd(indices, [s[floor_id], s[wall_id]], [size])
                self._x = tf.scatter_nd(indices, [x[floor_id], x[wall_id]], [size])

                self.test_class = test_class

            def __call__(self, objects, points):

                self.test_class.assertTrue(objects.dtype == tf.int32)
                self.test_class.assertTrue(points.dtype == tf.float32)
                self.test_class.assertTrue(points.shape[:-1] == objects.shape)
                self.test_class.assertTrue(points.shape[-1] == 3)

                s = tf.gather(self._s, objects)
                x = tf.gather(self._x, objects)
                e = tf.gather(self._etas, objects)

                return e, s, x

        # Assign the radio material callable to the scene
        scene.radio_material_callable = RadioMatCallable(self)
        # # Set different radio materials to objects to ensure the test fails if the callable is not used
        scene.objects['floor'].radio_material = 'itu_wood'
        scene.objects['wall'].radio_material = 'itu_marble'
        # Compute paths powers
        sionna.config.seed=1
        cm_call = scene.coverage_map(max_depth=3, los=False, diffraction=True,
                                     edge_diffraction=True, scattering=True,
                                     reflection=True, cm_cell_size=(0.1, 0.1))
        cm_call = cm_call.path_gain[0]

        # The coverage map solver uses tf.tensor_scatter_nd_add() to add the
        # contributions of paths to cells of the coverage map.
        # This function does not add the contributions that fall into the same
        # cell in a deterministic order, leading to slighlty different results:
        # https://www.itu.dk/~sestoft/bachelor/IEEE754_article.pdf
        max_rel_err = tf.reduce_max(tf.abs(cm_call-cm_ref))/tf.reduce_min(cm_ref)
        self.assertLess(max_rel_err, 1e-3)

    def test_scattering_pattern_callable_paths(self):

        dtype=tf.complex64
        alpha_r=10
        alpha_i=5
        lambda_=0.4

        scene = load_scene(sionna.rt.scene.simple_street_canyon, dtype=dtype)
        for obj in scene.objects.values():
            obj.radio_material.scattering_coefficient = 1.0
            obj.radio_material.scattering_pattern = BackscatteringPattern(alpha_r=alpha_r,
                                                                          alpha_i=alpha_i,
                                                                          lambda_=lambda_)
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

        # Place tx and rx
        scene.add(Transmitter("tx", [60., 0. , 1.5], dtype=dtype))
        scene.add(Receiver("rx", [-40., 0.,  1.5], dtype=dtype))
        scene.add(Receiver("rx2", [-22., -22., 1.5], dtype=dtype))

        # Reference paths
        sionna.config.seed=1
        paths_ref = scene.compute_paths(los=False, diffraction=False,
                                        scattering=True, reflection=False)

        # Define a scattering pattern callable
        class ScatteringPatternCallable:

            def __init__(self, alpha_r, alpha_i, lambda_, test_class):

                self._alpha_r = alpha_r
                self._alpha_i = alpha_i
                self._lambda_ = lambda_

                self._test_class = test_class

            def __call__(self, objects, points, k_i, k_s, n):

                self._test_class.assertTrue(objects.dtype == tf.int32)
                self._test_class.assertTrue(points.dtype == tf.float32)
                self._test_class.assertTrue(k_i.dtype == tf.float32)
                self._test_class.assertTrue(k_s.dtype == tf.float32)
                self._test_class.assertTrue(n.dtype == tf.float32)
                self._test_class.assertTrue(points.shape[:-1] == objects.shape)
                self._test_class.assertTrue(points.shape[-1] == 3)
                self._test_class.assertTrue(k_i.shape[-1] == 3)
                self._test_class.assertTrue(k_s.shape[-1] == 3)
                self._test_class.assertTrue(n.shape[-1] == 3)

                cos_i = tf.reduce_all(tf.reduce_sum(k_i*n, axis=-1) <= 0.)
                cos_s = tf.reduce_all(tf.reduce_sum(k_s*n, axis=-1) >= 0.)
                self._test_class.assertTrue(cos_i)
                self._test_class.assertTrue(cos_s)

                k_i = tf.reshape(k_i, [-1, 3])
                k_s = tf.reshape(k_s, [-1, 3])
                n = tf.reshape(n, [-1, 3])
                batch_size = tf.shape(n)[0]

                alpha_r = tf.fill([batch_size], self._alpha_r)
                alpha_i = tf.fill([batch_size], self._alpha_i)
                lambda_ = tf.fill([batch_size], self._lambda_)

                f_s = ScatteringPattern.pattern(k_i, k_s, n, alpha_r, alpha_i, lambda_)
                f_s = tf.reshape(f_s, tf.shape(objects))

                return f_s

        # Set the scattering pattern callable
        scene.scattering_pattern_callable = ScatteringPatternCallable(alpha_r,
                                                                      alpha_i,
                                                                      lambda_,
                                                                      self)
        # Change the scattering pattern of the scene, to ensure that the test fails
        # if the callable is not called
        for obj in scene.objects.values():
            obj.radio_material.scattering_pattern = LambertianPattern()

        # Generate paths using the scattering pattern callable
        sionna.config.seed=1
        paths_callable = scene.compute_paths(los=False, diffraction=False,
                                             scattering=True, reflection=False)

        # Test error
        max_err = tf.reduce_max(tf.abs(paths_callable.a - paths_ref.a)).numpy()
        self.assertTrue(max_err == 0.0)

    def test_scattering_pattern_callable_cm(self):

        dtype=tf.complex64
        alpha_r=10
        alpha_i=5
        lambda_=0.4

        scene = load_scene(sionna.rt.scene.simple_street_canyon, dtype=dtype)
        for obj in scene.objects.values():
            obj.radio_material.scattering_coefficient = 1.0
            obj.radio_material.scattering_pattern = BackscatteringPattern(alpha_r=alpha_r,
                                                                          alpha_i=alpha_i,
                                                                          lambda_=lambda_)
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

        # Place tx and rx
        scene.add(Transmitter("tx", [60., 0. , 2.5], dtype=dtype))
        # Reference paths
        sionna.config.seed=1
        cm_ref = scene.coverage_map(los=False,
                                    diffraction=False,
                                    scattering=True,
                                    reflection=False,
                                    cm_cell_size=(1, 1)).path_gain[0]

        # Define a scattering pattern callable
        class ScatteringPatternCallable:

            def __init__(self, alpha_r, alpha_i, lambda_, test_class):

                self._alpha_r = alpha_r
                self._alpha_i = alpha_i
                self._lambda_ = lambda_

                self._test_class = test_class

            def __call__(self, objects, points, k_i, k_s, n):

                self._test_class.assertTrue(objects.dtype == tf.int32)
                self._test_class.assertTrue(points.dtype == tf.float32)
                self._test_class.assertTrue(k_i.dtype == tf.float32)
                self._test_class.assertTrue(k_s.dtype == tf.float32)
                self._test_class.assertTrue(n.dtype == tf.float32)
                self._test_class.assertTrue(points.shape[:-1] == objects.shape)
                self._test_class.assertTrue(points.shape[-1] == 3)
                self._test_class.assertTrue(k_i.shape[-1] == 3)
                self._test_class.assertTrue(k_s.shape[-1] == 3)
                self._test_class.assertTrue(n.shape[-1] == 3)

                cos_i = tf.reduce_all(tf.reduce_sum(k_i*n, axis=-1) <= 0.)
                cos_s = tf.reduce_all(tf.reduce_sum(k_s*n, axis=-1) >= 0.)
                self._test_class.assertTrue(cos_i)
                self._test_class.assertTrue(cos_s)

                k_i = tf.reshape(k_i, [-1, 3])
                k_s = tf.reshape(k_s, [-1, 3])
                n = tf.reshape(n, [-1, 3])
                batch_size = tf.shape(n)[0]

                alpha_r = tf.fill([batch_size], self._alpha_r)
                alpha_i = tf.fill([batch_size], self._alpha_i)
                lambda_ = tf.fill([batch_size], self._lambda_)

                f_s = ScatteringPattern.pattern(k_i, k_s, n, alpha_r, alpha_i, lambda_)
                f_s = tf.reshape(f_s, tf.shape(objects))

                return f_s

        # Set the scattering pattern callable
        scene.scattering_pattern_callable = ScatteringPatternCallable(alpha_r,
                                                                     alpha_i,
                                                                     lambda_,
                                                                     self)
        # Change the scattering pattern of the scene, to ensure that the test fails
        # if the callable is not called
        for obj in scene.objects.values():
            obj.radio_material.scattering_pattern = LambertianPattern()

        # Generate paths using the scattering pattern callable
        sionna.config.seed=1
        cm_callable = scene.coverage_map(los=False,
                                        diffraction=False,
                                        scattering=True,
                                        reflection=False,
                                        cm_cell_size=(1, 1)).path_gain[0]

        # Test error
        max_err = tf.reduce_max(tf.math.divide_no_nan(tf.abs(cm_callable - cm_ref), cm_ref)).numpy()
        self.assertTrue(max_err < 1e-3)
