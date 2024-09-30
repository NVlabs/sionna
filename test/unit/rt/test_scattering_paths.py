#
# SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import unittest
import numpy as np
import tensorflow as tf
import sionna
from sionna.rt import load_scene, RadioMaterial, LambertianPattern, DirectivePattern, BackscatteringPattern, PlanarArray, Transmitter, Receiver, theta_phi_from_unit_vec, dot
from sionna.constants import PI

def comp_power(scattering_coefficient, xpd_coefficient, scattering_pattern, polarization, distance, dtype=tf.complex64):
    """
    Function to compute the scattered paths from a single surface.
    TX and RX are both located at the same distance away from the origin,
    with an angle of 45 degrees with respect to the surfce normal.

    Different scattering parameters can be congigured.

    Also the theoretical ray making a far-field approximation, i.e.,
    a single scattering tile, is returned.
    """
    scene = load_scene(sionna.rt.scene.simple_reflector, dtype=dtype)
    rm = RadioMaterial("MyMaterial", dtype=dtype)
    scene.get("reflector").radio_material = rm
    rm.scattering_pattern = scattering_pattern
    rm.scattering_coefficient = scattering_coefficient
    rm.xpd_coefficient = xpd_coefficient
    rm.conductivity = 1e24 # Perfect reflector

    # Position transmitter and receiver
    scene.tx_array = PlanarArray(num_rows=1,
                                 num_cols=1,
                                 vertical_spacing=0.5,
                                 horizontal_spacing=0.5,
                                 pattern="iso",
                                 polarization=polarization,
                                 polarization_model=2,
                                 dtype=dtype)

    scene.rx_array = PlanarArray(num_rows=1,
                                 num_cols=1,
                                 vertical_spacing=0.5,
                                 horizontal_spacing=0.5,
                                 pattern="iso",
                                 polarization=polarization,
                                 polarization_model=2,
                                 dtype=dtype)

    scene.synthetic_array = False

    dist = distance
    d = dist/np.sqrt(2)
    tx = Transmitter(name="tx",
              position=[-d,0,d],
              orientation=[0,0,0], dtype=dtype)
    scene.add(tx)

    rx = Receiver(name="rx",
              position=[d,0,d],
              orientation=[0,0,0], dtype=dtype)
    scene.add(rx)

    scene.frequency = 3.5e9
    paths = scene.compute_paths(los=False, reflection=False, diffraction=False, edge_diffraction=True, scattering=True, scat_keep_prob=1, num_samples=10e6)
    a, tau = paths.cir()
    
    # Theoretical far field approximation (a single scattering tile
    # With Lambertian scattering pattern
    r_i = np.linalg.norm(tx.position)
    r_s = np.linalg.norm(rx.position)
    k_i = -tx.position/r_i
    k_s = rx.position/r_s
    dA = 1
    theta_i, _ = theta_phi_from_unit_vec(-k_i)
    k_i = tf.expand_dims(k_i, 0)
    k_s = tf.expand_dims(k_s, 0)
    n_hat = tf.constant([[0,0,1]], dtype.real_dtype)

    a_theo = scene.wavelength/(4*PI)/(r_i*r_s)
    a_theo *= np.sqrt( np.cos(theta_i)* dA * scattering_pattern(k_i, k_s, n_hat) )
    a_theo *= scattering_coefficient

    return a, tau, a_theo
def comp_power_double_reflection(rx_x_pos, scattering_coefficient, xpd_coefficient, scattering_pattern, polarization, dtype=tf.complex128):
    """
    Function to compute the scattered paths from a single surface that is reached
    by paths undergoing a single perfect reflection.

    Different scattering parameters can be congigured.

    Also the theoretical ray making a far-field approximation, i.e.,
    a single scattering tile, is returned.
    """

    scene = load_scene(sionna.rt.scene.double_reflector, dtype=dtype)

    # Create perfect reflector material
    perfect_reflector = RadioMaterial("PerfectReflector", dtype=dtype)
    perfect_reflector.scattering_coefficient = 0
    perfect_reflector.conductivity = 1e24

    # Create pure perfectly scattering material
    scatterer = RadioMaterial("Scatterer", dtype=dtype)
    scatterer.scattering_pattern = scattering_pattern
    scatterer.scattering_coefficient = scattering_coefficient
    scatterer.xpd_coefficient = xpd_coefficient
    scatterer.conductivity = 1e24 # Perfect 

    # Assign materials to objects
    scene.get("large_reflector").radio_material = perfect_reflector
    scene.get("small_reflector").radio_material = scatterer

    # Position transmitter and receiver
    scene.tx_array = PlanarArray(num_rows=1,
                                     num_cols=1,
                                     vertical_spacing=0.5,
                                     horizontal_spacing=0.5,
                                     pattern="iso",
                                     polarization=polarization,
                                     polarization_model=2,
                                     dtype=dtype)

    scene.rx_array = PlanarArray(num_rows=1,
                                     num_cols=1,
                                     vertical_spacing=0.5,
                                     horizontal_spacing=0.5,
                                     pattern="iso",
                                     polarization=polarization,
                                     polarization_model=2,
                                     dtype=dtype)

    tx = Transmitter(name="tx",
              position=[-20,0,10],
              orientation=[0,0,0], dtype=dtype)
    scene.add(tx)

    rx = Receiver(name="rx",
              position=[rx_x_pos,0,0],
              orientation=[0,0,0], dtype=dtype)
    scene.add(rx)

    # Compute propagation paths
    scene.frequency = 3.5e9
    paths = scene.compute_paths(los=False,
                                reflection=False,
                                diffraction=False,
                                edge_diffraction=False,
                                scattering=True,
                                scat_keep_prob=1,
                                 num_samples=1e6)
    a, _ = paths.cir()
    p = tf.reduce_sum(tf.abs(tf.squeeze(a))**2,-1) # Power


    # Theoretical far field approximation (a single scattering tile)
    hitpoint = [0,0,10]
    k_s =  rx.position-hitpoint
    r_s = np.linalg.norm(k_s)
    k_s /= r_s

    refpoint = np.array([-10.,0,0])
    r_i_1 = np.linalg.norm(refpoint-tx.position)
    k_i = tf.constant(hitpoint - refpoint, dtype=dtype.real_dtype)
    r_i_2 = np.linalg.norm(k_i)
    k_i /= r_i_2
    r_i = r_i_1 + r_i_2

    dA = 1 # The scatterer has a size of 1x1m
    n_hat = tf.constant([[0,0,-1]], dtype.real_dtype)

    a_theo = scene.wavelength/(4*PI)/(r_i*r_s)
    a_theo *= np.squeeze(np.sqrt(dot(-k_i, n_hat)* dA * scattering_pattern(k_i, k_s, n_hat) ))
    p_theo = np.abs(a_theo)**2

    return p, p_theo


class TestScatteringPaths(unittest.TestCase):
    """Tests related to the computation of scattered paths
    """

    def test_01(self):
        """
        Test that received power from a scattering surface matches the
        theoretical far-field approximantion for various different
        configurations.
        """
        xpd_coefficient = 0.0
        distance = 100
        for dtype in [tf.complex64, tf.complex128]:
            patterns = [LambertianPattern(dtype=dtype),
                        DirectivePattern(20, dtype=dtype),
                        BackscatteringPattern(20, 30, 0.5, dtype=dtype)]
            for pattern in patterns:
                for polarization in ["V", "H"]:
                    for scattering_coefficient in [0.3, 0.7, 1.0]:
                        a, _, a_theo = comp_power(
                                        scattering_coefficient,
                                        xpd_coefficient,
                                        pattern,
                                        polarization,
                                        distance,
                                        dtype)

                        p = tf.reduce_sum(tf.abs(tf.squeeze(a))**2,-1)
                        p_theo = np.abs(a_theo)**2
                        err_rel = 20*np.log10((np.abs(p-p_theo)/p_theo))
                        if dtype==tf.complex64:
                            self.assertTrue(err_rel<-20)
                        else:
                            self.assertTrue(err_rel<-30)

    def test_02(self):
        """
        Test that the XPD-coefficient achieves the desired
        behaviour for VH polarization.
        """
        dtype = tf.complex128
        scattering_coefficient = 1.0
        scattering_pattern = DirectivePattern(30, dtype=dtype)
        distance = 100
        polarization = "VH"

        # No XPD
        xpd_coefficient = 0.0
        a, tau, a_theo = comp_power(scattering_coefficient,
                                    xpd_coefficient,
                                    scattering_pattern,
                                    polarization, 
                                    distance,
                                    dtype)

        p = tf.reduce_sum(tf.abs(tf.squeeze(a))**2,-1)
        p_theo = np.abs(a_theo)**2
        err_rel = 20*np.log10((np.abs(p-p_theo)/p_theo))
        self.assertTrue(err_rel[0,0]<-30)
        self.assertTrue(err_rel[1,1]<-30)

        # XPD so that both polarizations get half the power
        xpd_coefficient = 0.5
        a, tau, a_theo = comp_power(scattering_coefficient,
                                    xpd_coefficient,
                                    scattering_pattern,
                                    polarization, 
                                    distance,
                                    dtype)

        p = tf.reduce_sum(tf.abs(tf.squeeze(a))**2,-1)
        p_theo = np.abs(a_theo)**2
        err_rel = 20*np.log10((np.abs(p-p_theo/2)/p_theo))
        self.assertTrue(np.max(err_rel)<-30)

        # XPD so that both polarizations get inverted
        xpd_coefficient = 1.0
        a, tau, a_theo = comp_power(scattering_coefficient,
                                    xpd_coefficient,
                                    scattering_pattern,
                                    polarization,
                                    distance,
                                    dtype)

        p = tf.reduce_sum(tf.abs(tf.squeeze(a))**2,-1)
        p_theo = np.abs(a_theo)**2
        err_rel = 20*np.log10((np.abs(p-p_theo)/p_theo))
        self.assertTrue(err_rel[0,1]<-30)
        self.assertTrue(err_rel[1,0]<-30)

    def test_03(self):
        """
        Test that received power from a scattering surface matches the
        theoretical far-field approximantion after reflection+scattering.
        """
        dtype = tf.complex128
        scattering_coefficient = 1.0
        xpd_coefficient = 0
        scattering_pattern = DirectivePattern(10, dtype=dtype)

        for rx_x_pos in [1., 5., 20.]:
            for polarization in ["V", "H"]:
                p, p_theo = comp_power_double_reflection(rx_x_pos,
                                                         scattering_coefficient,
                                                         xpd_coefficient,
                                                         scattering_pattern,
                                                         polarization,
                                                         dtype)
                err_rel = 20*np.log10((np.abs(p-p_theo)/p_theo))
                self.assertTrue(err_rel<-30)

    def test_04(self):
        """
        Test that received power from a scattering surface matches the
        theoretical far-field approximantion after reflection+scattering
        and another 3xreflection+scattering.
        """
        dtype = tf.complex128
        scene = load_scene(sionna.rt.scene.triple_reflector, dtype=dtype)

        scattering_coefficient = 1
        ref_scat_coeff = 0.7 # scattering coefficient of the perfect reflectors
        xpd_coefficient = 0
        scattering_pattern = DirectivePattern(10, dtype=dtype)
        polarization = "V"

        perfect_reflector = RadioMaterial("PerfectReflector", dtype=dtype)
        perfect_reflector.scattering_coefficient = ref_scat_coeff
        perfect_reflector.conductivity = 1e24

        scatterer = RadioMaterial("Scatterer", dtype=dtype)
        scatterer.scattering_pattern = scattering_pattern
        scatterer.scattering_coefficient = scattering_coefficient
        scatterer.xpd_coefficient = xpd_coefficient
        scatterer.conductivity = 1e24

        scene.get("floor").radio_material = perfect_reflector
        scene.get("large_reflector").radio_material = perfect_reflector
        scene.get("small_reflector").radio_material = scatterer

        # Position transmitter and receiver
        scene.tx_array = PlanarArray(num_rows=1,
                                        num_cols=1,
                                        vertical_spacing=0.5,
                                        horizontal_spacing=0.5,
                                        pattern="iso",
                                        polarization=polarization,
                                        polarization_model=2,
                                        dtype=dtype)

        scene.rx_array = PlanarArray(num_rows=1,
                                        num_cols=1,
                                        vertical_spacing=0.5,
                                        horizontal_spacing=0.5,
                                        pattern="iso",
                                        polarization=polarization,
                                        polarization_model=2,
                                        dtype=dtype)

        tx = Transmitter(name="tx",
                    position=[-20,0,10],
                    orientation=[0,0,0], dtype=dtype)
        scene.add(tx)

        rx = Receiver(name="rx",
                    position=[30,0,-10],
                    orientation=[0,0,0], dtype=dtype)
        scene.add(rx)

        scene.frequency = 3.5e9
        paths = scene.compute_paths(max_depth=4, los=False, reflection=False, diffraction=False, edge_diffraction=False, scattering=True, scat_keep_prob=1, num_samples=10e6)
        a, _ = paths.cir()
        p = tf.reduce_sum(tf.abs(tf.squeeze(a))**2,-1)

        # Compute theoretical results
        dA = 1
        n_hat = tf.constant([[0,0,-1]], dtype.real_dtype)

        # Reflection-Scattering path
        refpoint_1 = np.array([0, 0, 0])
        hitpoint_1 = np.array([20, 0, 10])
        k_s_1 =  rx.position-hitpoint_1
        r_s_1 = np.linalg.norm(k_s_1)
        k_s_1 /= r_s_1

        r_i_1_1 = np.linalg.norm(refpoint_1-tx.position)
        k_i_1 = tf.constant(hitpoint_1 - refpoint_1, dtype=dtype.real_dtype)
        r_i_1_2 = np.linalg.norm(k_i_1)
        k_i_1 /= r_i_1_2
        r_i_1 = r_i_1_1 + r_i_1_2

        a_theo_1 = scene.wavelength/(4*PI)/(r_i_1*r_s_1)
        a_theo_1 *= np.squeeze(np.sqrt(dot(-k_i_1, n_hat)* dA * scattering_pattern(k_i_1, k_s_1, n_hat) ))*np.sqrt(1-ref_scat_coeff**2)
        p_theo_1 = np.abs(a_theo_1)**2

        # 3xReflection-Scattering path
        refpoint_2 = np.array([-10, 0, 0])
        r_i_2 = 4*np.linalg.norm(refpoint_2-tx.position)
        r_s_2 = r_s_1
        last_reflection = np.array([10,0,0])
        k_i_2 = tf.constant(hitpoint_1 - last_reflection, dtype=dtype.real_dtype)
        k_i_2 /= np.linalg.norm(k_i_2)
        a_theo_2 = scene.wavelength/(4*PI)/(r_i_2*r_s_2)
        a_theo_2 *= np.squeeze(np.sqrt(dot(-k_i_2, n_hat)* dA * scattering_pattern(k_i_2, k_s_1, n_hat) ))*np.sqrt(1-ref_scat_coeff**2)**3
        p_theo_2 = np.abs(a_theo_2)**2

        # Total theoretically received power
        p_theo = p_theo_1 + p_theo_2

        # Normalized error
        err_rel = 20*np.log10((np.abs(p-p_theo)/p_theo))
        self.assertTrue(err_rel<-30)
