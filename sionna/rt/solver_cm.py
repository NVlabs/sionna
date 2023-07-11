#
# SPDX-FileCopyrightText: Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""
Ray tracing algorithm that uses the image method to compute all pure reflection
paths.
"""

import mitsuba as mi
import drjit as dr
import tensorflow as tf
from sionna.constants import PI
from sionna.utils.tensors import expand_to_rank, insert_dims, flatten_dims
from .utils import dot, phi_hat, theta_hat, theta_phi_from_unit_vec,\
    normalize, rotation_matrix, mi_to_tf_tensor, compute_field_unit_vectors,\
        reflection_coefficient, component_transform, fibonacci_lattice, r_hat,\
            cross, cot, sign, sample_points_on_hemisphere, acos_diff
from .solver_base import SolverBase
from .coverage_map import CoverageMap, coverage_map_rectangle_to_world
from .scattering_pattern import ScatteringPattern


class SolverCoverageMap(SolverBase):
    # pylint: disable=line-too-long
    r"""SolverCoverageMap(scene, solver=None, dtype=tf.complex64)

    Generates a coverage map consisting of the squared amplitudes of the channel
    impulse response considering the LoS and reflection paths.

    The main inputs of the solver are:

    * The properties of the rectangle defining the coverage map, i.e., its
    position, scale, and orientation, and the resolution of the coverage map

    * The receiver orientation

    * A maximum depth, corresponding to the maximum number of reflections. A
    depth of zero corresponds to LoS only.

    Generation of a coverage map is carried-out for every transmitter in the
    scene. The antenna arrays of the transmitter and receiver are used.

    The generation of a coverage map consists in two steps:

    1. Shoot-and bounce ray tracing where rays are generated from the
    transmitters and the intersection with the rectangle defining the coverage
    map are recorded.
    Initial rays direction are arranged in a Fibonacci lattice on the unit
    sphere.

    2. The transfer matrices of every ray that intersect the coverage map are
    computed considering the materials of the objects that make the scene.
    The antenna patterns, synthetic phase shifts due to the array geometry, and
    combining and precoding vectors are then applied to obtain channel
    coefficients. The squared amplitude of the channel coefficients are then
    added to the value of the output corresponding to the cell of the coverage
    map within which the intersection between the ray and the coverage map
    occured.

    Note: Only triangle mesh are supported.

    Parameters
    -----------
    scene : :class:`~sionna.rt.Scene`
        Sionna RT scene

    solver : :class:`~sionna.rt.BaseSolver` | None
        Another solver from which to re-use some structures to avoid useless
        compute and memory use

    dtype : tf.complex64 | tf.complex128
        Datatype for all computations, inputs, and outputs.
        Defaults to `tf.complex64`.

    Input
    ------
    max_depth : int
        Maximum depth (i.e., number of bounces) allowed for tracing the
        paths

    rx_orientation : [3], tf.float
        Orientation of the receiver.
        This is used to compute the antenna response and antenna pattern
        for an imaginary receiver located on the coverage map.

    cm_center : [3], tf.float
        Center of the coverage map

    cm_orientation : [3], tf.float
        Orientation of the coverage map

    cm_size : [2], tf.float
        Scale of the coverage map.
        The width of the map (in the local X direction) is scale[0]
        and its map (in the local Y direction) scale[1].

    cm_cell_size : [2], tf.float
        Resolution of the coverage map, i.e., width
        (in the local X direction) and height (in the local Y direction) in
        meters of a cell of the coverage map

    combining_vec : [num_rx_ant], tf.complex
        Combining vector.
        This is used to combine the signal from the receive antennas for
        an imaginary receiver located on the coverage map.

    precoding_vec : [num_tx or 1, num_tx_ant], tf.complex
        Precoding vectors of the transmitters

    num_samples : int
        Number of rays initially shooted from the transmitters.
        This number is shared by all transmitters, i.e.,
        ``num_samples/num_tx`` are shooted for each transmitter.

    los : bool
        If set to `True`, then the LoS paths are computed.

    reflection : bool
        If set to `True`, then the reflected paths are computed.

    diffraction : bool
        If set to `True`, then the diffracted paths are computed.

    scattering : bool
        if set to `True`, then the scattered paths are computed.

    edge_diffraction : bool
        If set to `False`, only diffraction on wedges, i.e., edges that
        connect two primitives, is considered.

    Output
    -------
    :cm : :class:`~sionna.rt.CoverageMap`
        The coverage maps
    """

    def __call__(self, max_depth, rx_orientation,
                 cm_center, cm_orientation, cm_size, cm_cell_size,
                 combining_vec, precoding_vec, num_samples,
                 los, reflection, diffraction, scattering, edge_diffraction):

        # If reflection and scattering are disabled, no need for a max_depth
        # higher than 1.
        # This clipping can save some compute for the shoot-and-bounce
        if (not reflection) and (not scattering):
            max_depth = tf.minimum(max_depth, 1)

        # Transmitters positions and orientations
        # sources_positions : [num_tx, 3]
        # sources_orientations : [num_tx, 3]
        sources_positions = []
        sources_orientations = []
        for tx in self._scene.transmitters.values():
            sources_positions.append(tx.position)
            sources_orientations.append(tx.orientation)
        sources_positions = tf.stack(sources_positions, axis=0)
        sources_orientations = tf.stack(sources_orientations, axis=0)

       # EM properties of the materials
        # Returns: relative_permittivities, denoted by `etas`
        # scattering_coefficients, xpd_coefficients,
        # alpha_r, alpha_i and lambda_
        object_properties = self._build_scene_object_properties_tensors()
        etas = object_properties[0]
        scattering_coefficient = object_properties[1]
        xpd_coefficient = object_properties[2]
        alpha_r = object_properties[3]
        alpha_i = object_properties[4]
        lambda_ = object_properties[5]

        # Measurement plane defining the coverage map
        # meas_plane : mi.Shape
        #     Mitsuba rectangle defining the measurement plane
        meas_plane = self._build_mi_measurement_plane(cm_center,
                                                      cm_orientation,
                                                      cm_size)

        ####################################################
        # Shooting-and-bouncing
        # Computes the coverage map for LoS, reflection,
        # and scattering.
        # Also returns the primitives found in LoS of the
        # transmitters to shoot diffracted rays.
        ####################################################

        cm, los_primitives = self._shoot_and_bounce(meas_plane,
                                                    rx_orientation,
                                                    sources_positions,
                                                    sources_orientations,
                                                    max_depth,
                                                    num_samples,
                                                    combining_vec,
                                                    precoding_vec,
                                                    cm_center,
                                                    cm_orientation,
                                                    cm_size,
                                                    cm_cell_size,
                                                    los,
                                                    reflection,
                                                    diffraction,
                                                    scattering,
                                                    etas,
                                                    scattering_coefficient,
                                                    xpd_coefficient,
                                                    alpha_r,
                                                    alpha_i,
                                                    lambda_)

        # ############################################
        # # Diffracted
        # ############################################

        if los_primitives is not None:

            cm_diff = self._diff_samples_2_coverage_map(los_primitives,
                                                        edge_diffraction,
                                                        num_samples,
                                                        sources_positions,
                                                        meas_plane,
                                                        cm_center,
                                                        cm_orientation,
                                                        cm_size,
                                                        cm_cell_size,
                                                        sources_orientations,
                                                        rx_orientation,
                                                        combining_vec,
                                                        precoding_vec,
                                                        etas,
                                                        scattering_coefficient)

            cm = cm + cm_diff

        # ############################################
        # # Combine the coverage maps.
        # # Coverage maps are combined non-coherently
        # ############################################
        cm = CoverageMap(cm_center,
                         cm_orientation,
                         cm_size,
                         cm_cell_size,
                         cm,
                         scene=self._scene,
                         dtype=self._dtype)
        return cm

    ##################################################################
    # Internal methods
    ##################################################################

    def _build_mi_measurement_plane(self, cm_center, cm_orientation, cm_size):
        r"""
        Builds the Mitsuba rectangle defining the measurement plane
        corresponding to the coverage map.

        Input
        ------
        cm_center : [3], tf.float
            Center of the rectangle

        cm_orientation : [3], tf.float
            Orientation of the rectangle

        cm_size : [2], tf.float
            Scale of the rectangle.
            The width of the rectangle (in the local X direction) is scale[0]
            and its height (in the local Y direction) scale[1].

        Output
        ------
        mi_meas_plane : mi.Shape
            Mitsuba rectangle defining the measurement plane
        """
        # Rectangle defining the coverage map
        mi_meas_plane = mi.load_dict({
            'type': 'rectangle',
            'to_world': coverage_map_rectangle_to_world(cm_center,
                                                        cm_orientation,
                                                        cm_size),
        })

        return mi_meas_plane

    def _mp_hit_point_2_cell_ind(self, rot_gcs_2_mp, cm_center, cm_size,
                                 cm_cell_size, num_cells, hit_point):
        r"""
        Computes the indices of the cells to which points ``hit_point`` on the
        measurement plane belongs.

        Input
        ------
        rot_gcs_2_mp : [3, 3], tf.float
            Rotation matrix for going from the measurement plane LCS to the GCS

        cm_center : [3], tf.float
            Center of the coverage map

        cm_size : [2], tf.float
            Size of the coverage map

        cm_cell_size : [2], tf.float
            Size of of the cells of the ceverage map

        num_cells : [2], tf.int
            Number of cells in the coverage map

        hit_point : [...,3]
            Intersection points

        Output
        -------
        cell_ind : [..., 2], tf.int
            Indices of the cells
        """

        # Expand for broadcasting
        # [..., 3, 3]
        rot_gcs_2_mp = expand_to_rank(rot_gcs_2_mp, tf.rank(hit_point)+1, 0)
        # [..., 3]
        cm_center = expand_to_rank(cm_center, tf.rank(hit_point), 0)

        # Coverage map cells' indices
        # Coordinates of the hit point in the coverage map LCS
        # [..., 3]
        hit_point = tf.linalg.matvec(rot_gcs_2_mp, hit_point - cm_center)

        # In the local coordinate system of the coverage map, z should be 0
        # as the coverage map is in XY

        # x
        # [...]
        cell_x = hit_point[...,0] + cm_size[0]*0.5
        cell_x = tf.cast(tf.math.floor(cell_x/cm_cell_size[0]), tf.int32)
        cell_x = tf.where(tf.less(cell_x, num_cells[0]), cell_x, num_cells[0])
        cell_x = tf.where(tf.greater_equal(cell_x, 0), cell_x, num_cells[0])

        # y
        # [...]
        cell_y = hit_point[...,1] + cm_size[1]*0.5
        cell_y = tf.cast(tf.math.floor(cell_y/cm_cell_size[1]), tf.int32)
        cell_y = tf.where(tf.less(cell_y, num_cells[1]), cell_y, num_cells[1])
        cell_y = tf.where(tf.greater_equal(cell_y, 0), cell_y, num_cells[1])

        # [..., 2]
        cell_ind = tf.stack([cell_y, cell_x], axis=-1)

        return cell_ind

    def _compute_antenna_patterns(self, rot_mat, patterns, k):
        r"""
        Evaluates the antenna ``patterns`` of a radio device with oriented
        following ``orientation``, and for a incident field direction
        ``k``.

        Input
        ------
        rot_mat : [..., 3, 3] or [3,3], tf.float
            Rotation matrix built from the orientation of the radio device

        patterns : [f(theta, phi)], list of callable
            List of antenna patterns

        k : [..., 3], tf.float
            Direction of departure/arrival in the GCS.
            Must point away from the radio device

        Output
        -------
        fields_hat : [..., num_patterns, 2], tf.complex
            Antenna fields theta_hat and phi_hat components in the GCS

        theta_hat : [..., 3], tf.float
            Theta hat direction in the GCS

        phi_hat : [..., 3], tf.float
            Phi hat direction in the GCS

        """

        # [..., 3, 3]
        rot_mat = expand_to_rank(rot_mat, tf.rank(k)+1, 0)

        # [...]
        theta, phi = theta_phi_from_unit_vec(k)

        # Normalized direction vector in the LCS of the radio device
        # [..., 3]
        k_prime = tf.linalg.matvec(rot_mat, k, transpose_a=True)

        # Angles of departure in the local coordinate system of the
        # radio device
        # [...]
        theta_prime, phi_prime = theta_phi_from_unit_vec(k_prime)

        # Spherical global frame vectors
        # [..., 3]
        theta_hat_ = theta_hat(theta, phi)
        phi_hat_ = phi_hat(phi)

        # Spherical local frame vectors
        # [..., 3]
        theta_hat_prime = theta_hat(theta_prime, phi_prime)
        phi_hat_prime = phi_hat(phi_prime)

        # Rotate the LCS according to the radio device orientation
        # [..., 3]
        theta_hat_prime = tf.linalg.matvec(rot_mat, theta_hat_prime)
        phi_hat_prime = tf.linalg.matvec(rot_mat, phi_hat_prime)

        # Rotation matrix for going from the spherical radio device LCS to the
        # spherical GCS
        # [..., 2, 2]
        lcs2gcs = component_transform(theta_hat_prime, phi_hat_prime, # LCS
                                      theta_hat_, phi_hat_) # GCS
        lcs2gcs = tf.complex(lcs2gcs, tf.zeros_like(lcs2gcs))

        # Compute the fields in the LCS
        fields_hat = []
        for pattern in patterns:
            # [..., 2]
            field_ = tf.stack(pattern(theta_prime, phi_prime), axis=-1)
            fields_hat.append(field_)

        # Stacking the patterns, corresponding to different polarization
        # directions, as an additional dimension
        # [..., num_patterns, 2]
        fields_hat = tf.stack(fields_hat, axis=-2)

        # Fields in the GCS
        # [..., 1, 2, 2]
        lcs2gcs = tf.expand_dims(lcs2gcs, axis=-3)
        # [..., num_patterns, 2]
        fields_hat = tf.linalg.matvec(lcs2gcs, fields_hat)

        return fields_hat, theta_hat_, phi_hat_

    def _apply_synthetic_array(self, tx_rot_mat, rx_rot_mat, k_rx,
                               k_tx, a):
        # pylint: disable=line-too-long
        r"""
        Synthetically apply transmitter and receiver arrays to the channel
        coefficients ``a``

        Input
        ------
        tx_rot_mat : [..., 3, 3], tf.float
            Rotation matrix built from the orientation of the transmitters

        rx_rot_mat : [3, 3], tf.float
            Rotation matrix built from the orientation of the receivers

        k_rx : [..., 3], tf.float
            Directions of arrivals of the rays

        k_tx : [..., 3], tf.float
            Directions of departure of the rays

        a : [..., num_rx_patterns, num_tx_patterns], tf.complex
            Channel coefficients

        Output
        -------
        a : [..., num_rx_ant, num_tx_ant], tf.complex
            Channel coefficients with the antenna array applied
        """

        two_pi = tf.cast(2.*PI, self._rdtype)

        # Rotated position of the TX antenna elements
        # [..., tx_array_size, 3]
        tx_rel_ant_pos = expand_to_rank(self._scene.tx_array.positions,
                                        tf.rank(tx_rot_mat), 0)
        # [..., 1, 3, 3]
        tx_rot_mat_ = tf.expand_dims(tx_rot_mat, axis=-3)
        # [..., tx_array_size, 3]
        tx_rel_ant_pos = tf.linalg.matvec(tx_rot_mat_, tx_rel_ant_pos)

        # Rotated position of the RX antenna elements
        # [1, rx_array_size, 3]
        rx_rel_ant_pos = self._scene.rx_array.positions
        # [1, 3, 3]
        rx_rot_mat = tf.expand_dims(rx_rot_mat, axis=0)
        # [rx_array_size, 3]
        rx_rel_ant_pos = tf.linalg.matvec(rx_rot_mat, rx_rel_ant_pos)
        # [..., rx_array_size, 3]
        rx_rel_ant_pos = expand_to_rank(rx_rel_ant_pos, tf.rank(tx_rel_ant_pos),
                                        0)

        # Expand dims for broadcasting with antennas
        # [..., 1, 3]
        k_rx = tf.expand_dims(k_rx, axis=-2)
        k_tx = tf.expand_dims(k_tx, axis=-2)
        # Compute the synthetic phase shifts due to the antenna array
        # Transmitter side
        # [..., tx_array_size]
        tx_phase_shifts = dot(tx_rel_ant_pos, k_tx)
        # Receiver side
        # [..., rx_array_size]
        rx_phase_shifts = dot(rx_rel_ant_pos, k_rx)
        # Total phase shift
        # [..., rx_array_size, 1]
        rx_phase_shifts = tf.expand_dims(rx_phase_shifts, axis=-1)
        # [..., 1, tx_array_size]
        tx_phase_shifts = tf.expand_dims(tx_phase_shifts, axis=-2)
        # [..., rx_array_size, tx_array_size]
        phase_shifts = rx_phase_shifts + tx_phase_shifts
        phase_shifts = two_pi*phase_shifts/self._scene.wavelength
        # Apply the phase shifts
        # [..., 1, rx_array_size, 1, tx_array_size]
        phase_shifts = tf.expand_dims(phase_shifts, axis=-2)
        phase_shifts = tf.expand_dims(phase_shifts, axis=-4)
        # [..., num_rx_patterns, 1, num_tx_patterns, 1]
        a = tf.expand_dims(a, axis=-1)
        a = tf.expand_dims(a, axis=-3)
        # [..., num_rx_patterns, rx_array_size, num_tx_patterns, tx_array_size]
        a = a*tf.exp(tf.complex(tf.zeros_like(phase_shifts), phase_shifts))
        # Reshape to merge antenna patterns and array
        # [...,
        #   num_rx_ant=num_rx_patterns*rx_array_size,
        #   num_tx_ant=num_tx_patterns*tx_array_size ]
        a = flatten_dims(a, 2, len(a.shape)-4)
        a = flatten_dims(a, 2, len(a.shape)-2)

        return a

    def _update_coverage_map(self, cm_center, cm_size, cm_cell_size, num_cells,
                             rot_gcs_2_mp, cm_normal, tx_rot_mat,
                             rx_rot_mat, precoding_vec, combining_vec,
                             samples_tx_indices, e_field, field_es, field_ep,
                             mp_hit_point, hit_mp, k_tx, previous_int_point,cm):
        r"""
        Updates the coverage map with the power of the paths that hit it.

        Input
        ------
        cm_center : [3], tf.float
            Center of the coverage map

        cm_size : [2], tf.float
            Scale of the coverage map.
            The width of the map (in the local X direction) is ``cm_size[0]``
            and its map (in the local Y direction) ``cm_size[1]``.

        cm_cell_size : [2], tf.float
            Resolution of the coverage map, i.e., width
            (in the local X direction) and height (in the local Y direction) in
            meters of a cell of the coverage map

        num_cells : [2], tf.int
            Number of cells in the coverage map

        rot_gcs_2_mp : [3, 3], tf.float
            Rotation matrix for going from the measurement plane LCS to the GCS

        cm_normal : [3], tf.float
            Normal to the measurement plane

        tx_rot_mat : [num_tx, 3, 3], tf.float
            Rotation matrix built from the orientation of the transmitters

        rx_rot_mat : [3, 3], tf.float
            Rotation matrix built from the orientation of the receivers

        precoding_vec : [num_tx, num_tx_ant] or [1, num_tx_ant], tf.complex
            Vector used for transmit-precoding

        combining_vec : [num_rx_ant], tf.complex
            Vector used for receive-combing

        samples_tx_indices : [num_samples], tf.int
            Transmitter indices that correspond to evey sample, i.e., from
            which the ray was shot.

        e_field : [num_samples, num_tx_patterns, 2], tf.float
            Incoming electric field. These are the e_s and e_p components.
            The e_s and e_p directions are given thereafter.

        field_es : [num_samples, 3], tf.float
            S direction for the incident field

        field_ep : [num_samples, 3], tf.float
            P direction for the incident field

        mp_hit_point : [num_samples, 3], tf.float
            Positions of the hit points with the measurement plane.

        hit_mp : [num_samples], tf.bool
            Set to `True` for samples that hit the measurement plane

        k_tx : [num_samples, 3], tf.float
            Direction of departure from the transmitters

        previous_int_point : [num_samples, 3], tf.float
            Position of the previous interaction with the scene

        cm : [num_tx, num_cells_y+1, num_cells_x+1], tf.float
            Coverage map

        Output
        -------
        cm : [num_tx, num_cells_y+1, num_cells_x+1], tf.float
            Updated coverage map
        """

        # Extract the samples that hit the coverage map.
        # This is to avoid computing the channel coefficients for all the
        # samples.
        # Indices of the samples that hit the coverage map
        # [num_hits]
        hit_mp_ind = tf.where(hit_mp)[:,0]
        # Indices of the transmitters corresponding to the rays that hit
        # [num_hits]
        hit_mp_tx_ind = tf.gather(samples_tx_indices, hit_mp_ind)
        # the coverage map
        # [num_hits, 3]
        mp_hit_point = tf.gather(mp_hit_point, hit_mp_ind, axis=0)
        # [num_hits, 3]
        previous_int_point = tf.gather(previous_int_point, hit_mp_ind, axis=0)
        # [num_hits, 3]
        k_tx = tf.gather(k_tx, hit_mp_ind, axis=0)
        # [num_hits, 3]
        precoding_vec = tf.gather(precoding_vec, hit_mp_tx_ind, axis=0)
        # [num_hits, 3, 3]
        tx_rot_mat = tf.gather(tx_rot_mat, hit_mp_tx_ind, axis=0)
        # [num_hits, num_tx_patterns, 2]
        e_field = tf.gather(e_field, hit_mp_ind, axis=0)
        # [num_hits, 3]
        field_es = tf.gather(field_es, hit_mp_ind, axis=0)
        # [num_hits, 3]
        field_ep = tf.gather(field_ep, hit_mp_ind, axis=0)

        # Cell indices
        # [num_hits, 2]
        hit_cells = self._mp_hit_point_2_cell_ind(rot_gcs_2_mp, cm_center,
                                                  cm_size, cm_cell_size,
                                                  num_cells, mp_hit_point)
        # Receive direction
        # k_rx : [num_hits, 3]
        k_rx,_ = normalize(mp_hit_point - previous_int_point)

        # Compute the receive field in the GCS
        # rx_field : [num_hits, num_rx_patterns, 2]
        # rx_es_hat, rx_ep_hat : [num_hits, 3]
        rx_field, rx_es_hat, rx_ep_hat = self._compute_antenna_patterns(
            rx_rot_mat, self._scene.rx_array.antenna.patterns, -k_rx)
        # Move the incident field to the receiver basis
        # Change of basis of the field
        # [num_hits, 2, 2]
        to_rx_mat = component_transform(field_es, field_ep,
                                        rx_es_hat, rx_ep_hat)
        # [num_hits, 1, 2, 2]
        to_rx_mat = tf.expand_dims(to_rx_mat, axis=1)
        to_rx_mat = tf.complex(to_rx_mat, tf.zeros_like(to_rx_mat))
        # [num_hits, num_tx_patterns, 2]
        e_field = tf.linalg.matvec(to_rx_mat, e_field)
        # Apply the receiver antenna field to compute the channel coefficient
        # [num_hits num_rx_patterns, 1, 2]
        rx_field = tf.expand_dims(rx_field, axis=2)
        # [num_hits, 1, num_tx_patterns, 2]
        e_field = tf.expand_dims(e_field, axis=1)
        # [num_hits, num_rx_patterns, num_tx_patterns]
        a = tf.reduce_sum(tf.math.conj(rx_field)*e_field, axis=-1)

        # Apply synthetic array
        # [num_hits, num_rx_ant, num_tx_ant]
        a = self._apply_synthetic_array(tx_rot_mat, rx_rot_mat,
                                        k_rx, k_tx, a)

        # Apply precoding and combining
        # [1, num_rx_ant]
        combining_vec = tf.expand_dims(combining_vec, 0)
        # [num_hits, 1, num_tx_ant]
        precoding_vec = tf.expand_dims(precoding_vec, 1)
        # [num_hits, num_rx_ant]
        a = tf.reduce_sum(a*precoding_vec, axis=-1)
        # [num_hits]
        a = tf.reduce_sum(tf.math.conj(combining_vec)*a, axis=-1)

        # Compute the amplitude of the path
        # [num_hits]
        a = tf.square(tf.abs(a))

        # Add the rays contribution to the coverage map
        # We just divide by cos(aoa) instead of dividing by the square distance
        # to apply the propagation loss, to then multiply by the square distance
        # over cos(aoa) to compute the ray weight.
        # Ray weighting
        # Cosine of the angle of arrival with respect to the normal of
        # the plan
        # [num_hits]
        cos_aoa = tf.abs(dot(k_rx, cm_normal))
        # [num_hits]
        ray_weights = tf.math.divide_no_nan(tf.ones_like(cos_aoa), cos_aoa)
        # Add the contribution to the coverage map
        # [num_hits, 3]
        hit_cells = tf.concat([tf.expand_dims(hit_mp_tx_ind, axis=-1),
                                hit_cells], axis=-1)
        # [num_tx, num_cells_y+1, num_cells_x+1]
        cm = tf.tensor_scatter_nd_add(cm, hit_cells, ray_weights*a)

        return cm

    def _compute_reflected_field(self, normals, etas, scattering_coefficient,
                                 k_i, e_field, field_es, field_ep, scattering):
        r"""
        Computes the reflected field at the intersections.

        Input
        ------
        normals : [num_active_samples, 3], tf.float
            Normals to the intersected primitives

        etas : [num_active_samples], tf.complex
            Relative permittivities of the intersected primitives

        scattering_coefficient : [num_active_samples], tf.float
            Scattering coefficients of the intersected primitives

        k_i : [num_active_samples, 3], tf.float
            Direction of arrival of the ray

        e_field : [num_active_samples, num_tx_patterns, 2], tf.complex
            S and P components of the incident field

        field_es : [num_active_samples, 3], tf.float
            Direction of the S component of the incident field

        field_ep : [num_active_samples, 3], tf.float
            Directino of the P component of the incident field

        scattering : bool
            Set to `True` if scattering is enabled

        Output
        -------
        e_field : [num_active_samples, num_tx_patterns, 2], tf.complex
            S and P components of the reflected field

        field_es : [num_active_samples, 3], tf.float
            Direction of the S component of the reflected field

        field_ep : [num_active_samples, 3], tf.float
            Directino of the P component of the reflected field

        k_r : [num_active_samples, 3], tf.float
            Direction of the reflected ray
        """

        # [num_active_samples, 3]
        k_r = k_i - 2.*dot(k_i, normals, keepdim=True)*normals

        # S/P direction for the incident/reflected field
        # [num_active_samples, 3]
        # pylint: disable=unbalanced-tuple-unpacking
        e_i_s, e_i_p, e_r_s, e_r_p = compute_field_unit_vectors(k_i, k_r,
                                            normals, SolverBase.EPSILON)

        # Move to the incident S/P component
        # [num_active_samples, 2, 2]
        to_incident = component_transform(field_es, field_ep,
                                        e_i_s, e_i_p)
        # [num_active_samples, 1, 2, 2]
        to_incident = tf.expand_dims(to_incident, axis=1)
        to_incident = tf.complex(to_incident, tf.zeros_like(to_incident))
        # [num_active_samples, num_tx_patterns, 2]
        e_field = tf.linalg.matvec(to_incident, e_field)

        # Compute the reflection coefficients
        # [num_active_samples]
        cos_theta = -dot(k_i, normals)
        # [num_active_samples]
        r_s, r_p = reflection_coefficient(etas, cos_theta)

        # If scattering is enabled, then the rays are randomly
        # allocated to reflection or scattering by sampling according to the
        # scattering coefficient. An oversampling factor is applied to keep
        # differentiability with respect to the scattering coefficient.
        # This oversampling factor is the ratio between the reduction factor
        # and the (non-differientiable) probability with which a
        # reflection phenomena is selected. In our case, this probability is the
        # reduction factor.
        # If scattering is disabled, all samples are allocated to reflection to
        # maximize sample-efficiency. However, this requires correcting the
        # contribution of the reflected rays by applying the reduction factor.
        # [num_active_samples]
        reduction_factor = tf.sqrt(1. - tf.square(scattering_coefficient))
        reduction_factor = tf.complex(reduction_factor,
                                      tf.zeros_like(reduction_factor))
        if scattering:
            # [num_active_samples]
            ovs_factor = tf.math.divide_no_nan(reduction_factor,
                                            tf.stop_gradient(reduction_factor))
            r_s *= ovs_factor
            r_p *= ovs_factor
        else:
            # [num_active_samples]
            r_s *= reduction_factor
            r_p *= reduction_factor

        # Apply the reflection coefficients
        # [num_active_samples, 2]
        r = tf.stack([r_s, r_p], -1)
        # [num_active_samples, 1, 2]
        r = tf.expand_dims(r, axis=-2)
        # [num_active_samples, num_tx_patterns, 2]
        e_field *= r

        # Update S/P directions
        # [num_active_samples, 3]
        field_es = e_r_s
        field_ep = e_r_p

        return e_field, field_es, field_ep, k_r

    def _compute_scattered_field(self, normals, etas, scattering_coefficient,
            xpd_coefficient, alpha_r, alpha_i, lambda_, k_i, e_field, field_es,
            field_ep, reflection):
        r"""
        Computes the scattered field at the intersections.

        Input
        ------
        normals : [num_active_samples, 3], tf.float
            Normals to the intersected primitives

        etas : [num_active_samples], tf.complex
            Relative permittivities of the intersected primitives

        scattering_coefficient : [num_active_samples], tf.float
            Scattering coefficients of the intersected primitives

        xpd_coefficient : [num_active_samples], tf.float
            Tensor containing the cross-polarization discrimination
            coefficients of all shapes

        alpha_r : [num_active_samples], tf.float
            Tensor containing the alpha_r scattering parameters of all shapes

        alpha_i : [num_active_samples], tf.float
            Tensor containing the alpha_i scattering parameters of all shapes

        lambda_ : [num_shape], tf.float
            Tensor containing the lambda_ scattering parameters of all shapes

        k_i : [num_active_samples, 3], tf.float
            Direction of arrival of the ray

        e_field : [num_active_samples, num_tx_patterns, 2], tf.complex
            S and P components of the incident field

        field_es : [num_active_samples, 3], tf.float
            Direction of the S component of the incident field

        field_ep : [num_active_samples, 3], tf.float
            Directins of the P component of the incident field

        reflection : bool
            Set to `True` if reflection is enabled

        Output
        -------
        e_field : [num_active_samples, num_tx_patterns, 2], tf.complex
            S and P components of the scattered field

        field_es : [num_active_samples, 3], tf.float
            Direction of the S component of the scattered field

        field_ep : [num_active_samples, 3], tf.float
            Direction of the P component of the scattered field

        k_s : [num_active_samples, 3], tf.float
            Direction of the scattered ray
        """

        # Represent incomning field in the basis for reflection
        e_i_s, e_i_p = compute_field_unit_vectors(k_i, None,
                                            normals, SolverBase.EPSILON,
                                            return_e_r=False)

        # [num_active_samples, 2, 2]
        to_incident = component_transform(field_es, field_ep,
                                        e_i_s, e_i_p)
        # [num_active_samples, 1, 2, 2]
        to_incident = tf.expand_dims(to_incident, axis=1)
        to_incident = tf.complex(to_incident, tf.zeros_like(to_incident))
        # [num_active_samples, num_tx_patterns, 2]
        e_field_ref = tf.linalg.matvec(to_incident, e_field)

        # Compute Fresnel reflection coefficients
        # [num_active_samples]
        cos_theta = -dot(k_i, normals)
        # [num_active_samples]
        r_s, r_p = reflection_coefficient(etas, cos_theta)

        # [num_active_samples, 2]
        r = tf.stack([r_s, r_p], axis=-1)
        # [num_active_samples, 1, 2]
        r = tf.expand_dims(r, axis=-2)

        # Compute amplitude of the reflected field
        # [num_active_samples, num_tx_patterns]
        ref_amp = tf.sqrt(tf.reduce_sum(tf.abs(r*e_field_ref)**2, axis=-1))

        # Compute incoming field and polarization vectors
        # [num_active_samples, num_tx_patterns, 1]
        e_field_s, e_field_p = tf.split(e_field, 2, axis=-1)

        # [num_active_samples, 1, 3]
        field_es = tf.expand_dims(field_es, axis=1)
        field_es = tf.complex(field_es, tf.zeros_like(field_es))
        field_ep = tf.expand_dims(field_ep, axis=1)
        field_ep = tf.complex(field_ep, tf.zeros_like(field_ep))

        # Incoming field vector
        # [num_active_samples, num_tx_patterns, 3]
        e_in = e_field_s*field_es + e_field_p*field_ep

        # Polarization vectors
        # [num_active_samples, num_tx_patterns, 3]
        e_pol_hat, _ = normalize(tf.math.real(e_in))
        e_xpol_hat = cross(e_pol_hat, tf.expand_dims(k_i, 1))

        # Compute incoming spherical unit vectors in GCS
        theta_i, phi_i = theta_phi_from_unit_vec(-k_i)
        # [num_active_samples, 1, 3]
        theta_hat_i = tf.expand_dims(theta_hat(theta_i, phi_i), axis=1)
        phi_hat_i = tf.expand_dims(phi_hat(phi_i), axis=1)

        # Transformation from e_pol_hat, e_xpol_hat to theta_hat_i,phi_hat_i
        # [num_active_samples, num_tx_patterns, 2, 2]
        trans_mat = component_transform(e_pol_hat, e_xpol_hat,
                                        theta_hat_i, phi_hat_i)
        trans_mat = tf.complex(trans_mat, tf.zeros_like(trans_mat))

        # Generate random phases
        # All tx_patterns get the same phases
        num_active_samples = tf.shape(e_field)[0]
        phase_shape = [num_active_samples, 1, 2]
        # [num_active_samples, 1, 2]
        phases = tf.random.uniform(phase_shape, maxval=2*PI, dtype=self._rdtype)

        # Compute XPD weighting
        # [num_active_samples, 2]
        xpd_weights = tf.stack([tf.sqrt(1-xpd_coefficient),
                                        tf.sqrt(xpd_coefficient)],
                                       axis=-1)
        xpd_weights = tf.complex(xpd_weights, tf.zeros_like(xpd_weights))
        # [num_active_samples, 1, 2]
        xpd_weights = tf.expand_dims(xpd_weights, axis=1)

        # Create scattered field components from phases and xpd_weights
        # [num_active_samples, 1, 2]
        e_field = tf.exp(tf.complex(tf.zeros_like(phases), phases))
        e_field *= xpd_weights

        # Apply transformation to field vector
        # [num_active_samples, num_tx_patterns, 2]
        e_field = tf.linalg.matvec(trans_mat, e_field)

        # Draw random directions for scattered paths
        # [num_active_samples, 3]
        k_s = sample_points_on_hemisphere(normals)

        # Evaluate scattering pattern
        # [num_active_samples]
        f_s = ScatteringPattern.pattern(k_i,
                                        k_s,
                                        normals,
                                        alpha_r,
                                        alpha_i,
                                        lambda_)

        # Compute scaled scattered field
        # [num_active_samples, num_tx_patterns, 2]
        ref_amp = tf.expand_dims(ref_amp, -1)
        e_field *= tf.complex(ref_amp, tf.zeros_like(ref_amp))
        f_s = tf.reshape(tf.sqrt(f_s), [-1, 1, 1])
        e_field *= tf.complex(f_s, tf.zeros_like(f_s))
        e_field *= tf.cast(tf.sqrt(2*PI), self._dtype)

        # If reflection is enabled, then the rays are randomly
        # allocated to reflection or scattering by sampling according to the
        # scattering coefficient. An oversampling factor is applied to keep
        # differentiability with respect to the scattering coefficient.
        # This oversampling factor is the ratio between the scattering factor
        # and the (non-differientiable) probability with which a
        # scattering phenomena is selected. In our case, this probability is the
        # scattering factor.
        # If reflection is disabled, all samples are allocated to scattering to
        # maximize sample-efficiency. However, this requires correcting the
        # contribution of the reflected rays by applying the scattering factor.
        # [num_active_samples]
        scattering_factor = tf.complex(scattering_coefficient,
                                       tf.zeros_like(scattering_coefficient))
        # [num_active_samples, 1, 1]
        scattering_factor = tf.reshape(scattering_factor, [-1, 1, 1])
        if reflection:
           # [num_active_samples]
            ovs_factor = tf.math.divide_no_nan(scattering_factor,
                                            tf.stop_gradient(scattering_factor))
            # [num_active_samples, num_tx_patterns, 2]
            e_field *= ovs_factor
        else:
            # [num_active_samples, num_tx_patterns, 2]
            e_field *= scattering_factor

        # Compute outgoing spherical unit vectors in GCS
        theta_s, phi_s = theta_phi_from_unit_vec(k_s)
        # # [num_active_samples, 3]
        field_es = theta_hat(theta_s, phi_s)
        field_ep = phi_hat(phi_s)

        return e_field, field_es, field_ep, k_s


    def _init_e_field(self, valid_ray, samples_tx_indices, k_tx, tx_rot_mat):
        r"""
        Initialize the electric field for the rays flagged as valid.

        Input
        -----
        valid_ray : [num_samples], tf.bool
            Flag set to `True` if the ray is valid

        samples_tx_indices : [num_samples]. tf.int
            Index of the transmitter from which the ray originated

        k_tx : [num_samples, 3]. tf.float
            Direction of departure

        tx_rot_mat : [num_tx, 3, 3], tf.float
            Matrix to go transmitter LCS to the GCS

        Output
        -------
        e_field : [num_valid_samples, num_tx_patterns, 2], tf.complex
            Emitted electric field S and P components

        field_es : [num_valid_samples, 3], tf.float
            Direction of the S component of the electric field

        field_ep : [num_valid_samples, 3], tf.float
            Direction of the P component of the electric field
        """

        num_samples = tf.shape(valid_ray)[0]

        # [num_valid_samples]
        valid_ind = tf.where(valid_ray)[:,0]
        # [num_valid_samples]
        valid_tx_ind = tf.gather(samples_tx_indices, valid_ind, axis=0)
        # [num_valid_samples, 3]
        k_tx = tf.gather(k_tx, valid_ind, axis=0)
        # [num_valid_samples, 3, 3]
        tx_rot_mat = tf.gather(tx_rot_mat, valid_tx_ind, axis=0)

        # val_e_field : [num_valid_samples, num_tx_patterns, 2]
        # val_field_es, val_field_ep : [num_valid_samples, 3]
        val_e_field, val_field_es, val_field_ep =\
            self._compute_antenna_patterns(tx_rot_mat,
                            self._scene.tx_array.antenna.patterns, k_tx)

        valid_ind = tf.expand_dims(valid_ind, axis=-1)
        # [num_samples, num_tx_patterns, 2]
        e_field = tf.scatter_nd(valid_ind, val_e_field,
                                [num_samples, val_e_field.shape[1], 2])
        # [num_samples, 3]
        field_es = tf.scatter_nd(valid_ind, val_field_es,
                                 [num_samples, 3])
        field_ep = tf.scatter_nd(valid_ind, val_field_ep,
                                 [num_samples, 3])

        return e_field, field_es, field_ep

    def _extract_active_rays(self, active_ind, int_point, previous_int_point,
        primitives, e_field, field_es, field_ep, etas, scattering_coefficient,
        xpd_coefficient, alpha_r, alpha_i, lambda_):
        r"""
        Extracts the active rays.

        Input
        ------
        active_ind : [num_active_samples], tf.int
            Indices of the active rays

        int_point : [num_samples, 3], tf.float
            Positions at which the rays intersect with the scene. For the rays
            that did not intersect the scene, the corresponding position should
            be ignored.

        previous_int_point : [num_samples, 3], tf.float
            Positions of the previous intersection points of the rays with
            the scene

        primitives : [num_samples], tf.int
            Indices of the intersected primitives

        e_field : [num_samples, num_tx_patterns, 2], tf.complex
            S and P components of the electric field

        field_es : [num_samples, 3], tf.float
            Direction of the S component of the field

        field_ep : [num_samples, 3], tf.float
            Direction of the P component of the field

        etas : [num_shape], tf.complex
            Tensor containing the complex relative permittivities of all shapes

        scattering_coefficient : [num_shape], tf.float
            Tensor containing the scattering coefficients of all shapes

        xpd_coefficient : [num_shape], tf.float | `None`
            Tensor containing the cross-polarization discrimination
            coefficients of all shapes

        alpha_r : [num_shape], tf.float | `None`
            Tensor containing the alpha_r scattering parameters of all shapes

        alpha_i : [num_shape], tf.float | `None`
            Tensor containing the alpha_i scattering parameters of all shapes

        lambda_ : [num_shape], tf.float | `None`
            Tensor containing the lambda_ scattering parameters of all shapes

        Output
        -------
        act_e_field : [num_active_samples, num_tx_patterns, 2], tf.complex
            S and P components of the electric field of the active rays

        act_field_es : [num_active_samples, 3], tf.float
            Direction of the S component of the field of the active rays

        act_field_ep : [num_active_samples, 3], tf.float
            Direction of the P component of the field of the active rays

        act_point : [num_active_samples, 3], tf.float
            Positions at which the rays intersect with the scene

        act_normals : [num_active_samples, 3], tf.float
            Normals at the intersection point. The normals are oriented to match
            the direction opposite to the incident ray

        act_etas : [num_active_samples], tf.complex
            Relative permittivity of the intersected primitives

        act_scat_coeff : [num_active_samples], tf.float
            Scattering coefficient of the intersected primitives

        act_k_i : [num_active_samples, 3], tf.float
            Direction of the active incident ray

        act_xpd_coefficient : [num_active_samples], tf.float | `None`
            Tensor containing the cross-polarization discrimination
            coefficients of all shapes.
            Only returned if ``xpd_coefficient`` is not `None`.

        act_alpha_r : [num_active_samples], tf.float
            Tensor containing the alpha_r scattering parameters of all shapes.
            Only returned if ``alpha_r`` is not `None`.

        act_alpha_i : [num_active_samples], tf.float
            Tensor containing the alpha_i scattering parameters of all shapes
            Only returned if ``alpha_i`` is not `None`.

        act_lambda_ : [num_active_samples], tf.float
            Tensor containing the lambda_ scattering parameters of all shapes
            Only returned if ``lambda_`` is not `None`.
        """

        # Extract the rays that interact the scene
        # [num_active_samples, num_tx_patterns, 2]
        act_e_field = tf.gather(e_field, active_ind, axis=0)
        # [num_active_samples, 3]
        act_field_es = tf.gather(field_es, active_ind, axis=0)
        # [num_active_samples, 3]
        act_field_ep = tf.gather(field_ep, active_ind, axis=0)
        # [num_active_samples, 3]
        act_previous_int_point = tf.gather(previous_int_point, active_ind,
                                            axis=0)
        # Current intersection point
        # [num_active_samples, 3]
        int_point = tf.gather(int_point, active_ind, axis=0)
        # [num_active_samples]
        act_primitives = tf.gather(primitives, active_ind, axis=0)
        # [num_active_samples]
        act_objects = tf.gather(self._primitives_2_objects, act_primitives,
                                axis=0)
        # Extract the normals to the intersected primitves
        # [num_active_samples, 3]
        act_normals = tf.gather(self._normals, act_primitives, axis=0)
        # Extract the material properties of the intersected normals
        # [num_active_samples]
        act_etas = tf.gather(etas, act_objects)
        # [num_active_samples]
        act_scat_coeff = tf.gather(scattering_coefficient, act_objects)
        if xpd_coefficient is not None:
            act_xpd_coefficient = tf.gather(xpd_coefficient, act_objects)
            act_alpha_r = tf.gather(alpha_r, act_objects)
            act_alpha_i = tf.gather(alpha_i, act_objects)
            act_lambda_ = tf.gather(lambda_, act_objects)

        # Direction of arrival
        # [num_active_samples, 3]
        act_k_i,_ = normalize(int_point - act_previous_int_point)

        # Ensure the normal points in the direction -k_i
        # [num_active_samples, 1]
        flip_normal = -tf.math.sign(dot(act_k_i, act_normals, keepdim=True))
        # [num_active_samples, 3]
        act_normals = flip_normal*act_normals

        if xpd_coefficient is None:
            output = (act_e_field, act_field_es, act_field_ep, int_point,
                    act_normals, act_etas, act_scat_coeff, act_k_i)
        else:
            output = (act_e_field, act_field_es, act_field_ep, int_point,
                    act_normals, act_etas, act_scat_coeff, act_k_i,
                    act_xpd_coefficient, act_alpha_r, act_alpha_i, act_lambda_)

        return output

    def _sample_interaction_phenomena(self, active, primitives,
                            scattering_coefficient, reflection, scattering):
        r"""
        Samples the interaction phenoema to apply to each active ray, among
        scattering or reflection.

        This is done by sampling a Bernouilli distribution with probablity p
        equal to the square of the scattering coefficient amplitude, as it
        corresponds to the ratio of the reflected energy that goes to
        scattering. With probability p, the ray is scattered. Otherwise, it is
        reflected.

        Input
        ------
        active : [num_samples], tf.bool
            Flag indicating if a ray is active

        scattering_coefficient : [num_shape], tf.complex
            Scattering coefficients of all shapes

        reflection : bool
            Set to `True` if reflection is enabled

        scattering : bool
            Set to `True` if scattering is enabled

        Output
        -------
        reflect_ind : [num_reflected_samples], tf.int
            Indices of the rays that are reflected

        scatter_ind : [num_scattered_samples], tf.int
            Indices of the rays that are scattered
        """

        assert reflection or scattering,\
            "This function should not be called if neither reflection nor"\
                " scattering is enabled"

        # Indices of the active samples
        # [num_active_samples]
        active_ind = tf.where(active)[:,0]

        # If only one of reflection or scattering is enabled, then all the
        # samples are used for the enabled phenomena to avoid wasting samples
        # by allocating them to a phenomena that is not requested by the users.
        # This approach, however, requires to correct later the contribution
        # of the rays by weighting them by the square of the scattering or
        # reduction factor, depending on the selected phenomena.
        # This is done in the functions that compute the reflected and scattered
        # field.
        if not reflection:
            reflect_ind = tf.zeros([0], tf.int32)
            scatter_ind = active_ind
        elif not scattering:
            reflect_ind = active_ind
            scatter_ind = tf.zeros([0], tf.int32)
        else:
            # Scattering coefficients of the intersected objects
            # [num_active_samples]
            act_primitives = tf.gather(primitives, active_ind, axis=0)
            act_objects = tf.gather(self._primitives_2_objects, act_primitives,
                                    axis=0)
            act_scat_coeff = tf.gather(scattering_coefficient, act_objects)

            # Probability of scattering
            # [num_active_samples]
            prob_scatter = tf.square(tf.abs(act_scat_coeff))

            # Sampling a Bernoulli distribution
            # [num_active_samples]
            scatter = tf.random.uniform(tf.shape(prob_scatter),
                                        tf.zeros((), self._rdtype),
                                        tf.ones((), self._rdtype),
                                        dtype=self._rdtype)
            scatter = tf.less(scatter, prob_scatter)

            # Extract indices of the reflected and scattered rays
            # [num_reflected_samples]
            reflect_ind = tf.gather(active_ind, tf.where(~scatter)[:,0])
            # [num_scattered_samples]
            scatter_ind = tf.gather(active_ind, tf.where(scatter)[:,0])

        return reflect_ind, scatter_ind

    def _apply_reflection(self, active_ind, int_point, previous_int_point,
        primitives, e_field, field_es, field_ep, etas, scattering_coefficient,
        scattering):
        r"""
        Apply reflection.

        Input
        ------
        active_ind : [num_reflected_samples], tf.int
            Indices of the *active* rays to which reflection must be applied.

        int_point : [num_samples, 3], tf.float
            Locations of the intersection point

        previous_int_point : [num_samples, 3], tf.float
            Locations of the intersection points of the previous interaction.

        primitives : [num_samples], tf.int
            Indices of the intersected primitives

        e_field : [num_samples, num_tx_patterns, 2], tf.complex
            S and P components of the electric field

        field_es : [num_samples, 3], tf.float
            Direction of the S component of the field

        field_ep : [num_samples, 3], tf.float
            Direction of the P component of the field

        etas : [num_shape], tf.complex
            Complex relative permittivities of all shapes

        scattering_coefficient : [num_shape], tf.float
            Scattering coefficients of all shapes

        scattering : bool
            Set to `True` if scattering is enabled

        Output
        -------
        e_field : [num_reflected_samples, num_tx_patterns, 2], tf.complex
            S and P components of the reflected electric field

        field_es : [num_reflected_samples, 3], tf.float
            Direction of the S component of the reflected field

        field_ep : [num_reflected_samples, 3], tf.float
            Direction of the P component of the reflected field

        int_point : [num_reflected_samples, 3], tf.float
            Locations of the intersection point

        k_r : [num_reflected_samples, 3], tf.float
            Direction of the reflected ray
        """

        # Prepare field computation
        # This function extract the data for the rays to which reflection
        # must be applied, and ensures that the normals are correcly oriented.
        act_data = self._extract_active_rays(active_ind, int_point,
            previous_int_point, primitives, e_field, field_es, field_ep,
            etas, scattering_coefficient, None, None, None, None)
        # [num_reflected_samples, num_tx_patterns, 2]
        e_field = act_data[0]
        # [num_reflected_samples, 3]
        field_es = act_data[1]
        field_ep = act_data[2]
        int_point = act_data[3]
        # [num_reflected_samples, 3]
        act_normals = act_data[4]
        # [num_reflected_samples]
        act_etas = act_data[5]
        act_scat_coeff = act_data[6]
        # [num_reflected_samples, 3]
        k_i = act_data[7]

        # Compute the reflected field
        e_field, field_es, field_ep, k_r = self._compute_reflected_field(
            act_normals, act_etas, act_scat_coeff, k_i, e_field, field_es,
            field_ep, scattering)

        return e_field, field_es, field_ep, int_point, k_r

    def _apply_scattering(self, active_ind, int_point, previous_int_point,
        primitives, e_field, field_es, field_ep, etas, scattering_coefficient,
        xpd_coefficient, alpha_r, alpha_i, lambda_, reflection):
        r"""
        Apply scattering.

        Input
        ------
        active_ind : [num_scattered_samples], tf.int
            Indices of the *active* rays to which scattering must be applied.

        int_point : [num_samples, 3], tf.float
            Locations of the intersection point

        previous_int_point : [num_samples, 3], tf.float
            Locations of the intersection points of the previous interaction.

        primitives : [num_samples], tf.int
            Indices of the intersected primitives

        e_field : [num_samples, num_tx_patterns, 2], tf.complex
            S and P components of the electric field

        field_es : [num_samples, 3], tf.float
            Direction of the S component of the field

        field_ep : [num_samples, 3], tf.float
            Direction of the P component of the field

        etas : [num_shape], tf.complex
            Complex relative permittivities of all shapes

        scattering_coefficient : [num_shape], tf.float
            Scattering coefficients of all shapes

        xpd_coefficient : [num_shape], tf.float | `None`
            Cross-polarization discrimination coefficients of all shapes

        alpha_r : [num_shape], tf.float | `None`
            alpha_r scattering parameters of all shapes

        alpha_i : [num_shape], tf.float | `None`
            alpha_i scattering parameters of all shapes

        lambda_ : [num_shape], tf.float | `None`
            lambda_ scattering parameters of all shapes

        reflection : bool
            Set to `True` if reflection is enabled

        Output
        -------

        Output
        -------
        e_field : [num_scattered_samples, num_tx_patterns, 2], tf.complex
            S and P components of the scattered electric field

        field_es : [num_scattered_samples, 3], tf.float
            Direction of the S component of the scattered field

        field_ep : [num_scattered_samples, 3], tf.float
            Direction of the P component of the scattered field

        int_point : [num_scattered_samples, 3], tf.float
            Locations of the intersection point

        k_r : [num_scattered_samples, 3], tf.float
            Direction of the scattered ray
        """

        # Prepare field computation
        # This function extract the data for the rays to which scattering
        # must be applied, and ensures that the normals are correcly oriented.
        act_data = self._extract_active_rays(active_ind, int_point,
            previous_int_point, primitives, e_field, field_es, field_ep,
            etas, scattering_coefficient, xpd_coefficient, alpha_r, alpha_i,
            lambda_)
        # [num_scattered_samples, num_tx_patterns, 2]
        e_field = act_data[0]
        # [num_scattered_samples, 3]
        field_es = act_data[1]
        field_ep = act_data[2]
        int_point = act_data[3]
        # [num_scattered_samples, 3]
        act_normals = act_data[4]
        # [num_scattered_samples]
        act_etas = act_data[5]
        act_scat_coeff = act_data[6]
        # [num_scattered_samples, 3]
        k_i = act_data[7]
        # [num_scattered_samples]
        act_xpd_coefficient = act_data[8]
        act_alpha_r = act_data[9]
        act_alpha_i = act_data[10]
        act_lambda_ = act_data[11]

        # Compute the reflected field
        e_field, field_es, field_ep, k_r = self._compute_scattered_field(
            act_normals, act_etas, act_scat_coeff, act_xpd_coefficient,
            act_alpha_r, act_alpha_i, act_lambda_, k_i, e_field, field_es,
            field_ep, reflection)

        return e_field, field_es, field_ep, int_point, k_r

    def _shoot_and_bounce(self,
                          meas_plane,
                          rx_orientation,
                          sources_positions,
                          sources_orientations,
                          max_depth,
                          num_samples,
                          combining_vec,
                          precoding_vec,
                          cm_center, cm_orientation, cm_size, cm_cell_size,
                          los,
                          reflection,
                          diffraction,
                          scattering,
                          etas,
                          scattering_coefficient,
                          xpd_coefficient,
                          alpha_r,
                          alpha_i,
                          lambda_):
        r"""
        Runs shoot-and-bounce to build the coverage map for LoS, reflection,
        and scattering.

        If ``diffraction`` is set to `True`, this function also returns the
        primitives in LoS with at least one transmitter.

        Input
        ------
        meas_plane : mi.Shape
            Mitsuba rectangle defining the measurement plane

        rx_orientation : [3], tf.float
            Orientation of the receiver.

        sources_positions : [num_tx, 3], tf.float
            Coordinates of the sources.

        max_depth : int
            Maximum number of reflections

        num_samples : int
            Number of rays initially shooted from the transmitters.
            This number is shared by all transmitters, i.e.,
            ``num_samples/num_tx`` are shooted for each transmitter.

        combining_vec : [num_rx_ant], tf.complex
            Combining vector.
            This is used to combine the signal from the receive antennas for
            an imaginary receiver located on the coverage map.

        precoding_vec : [num_tx or 1, num_tx_ant], tf.complex
            Precoding vectors of the transmitters

        cm_center : [3], tf.float
            Center of the coverage map

        cm_orientation : [3], tf.float
            Orientation of the coverage map

        cm_size : [2], tf.float
            Scale of the coverage map.
            The width of the map (in the local X direction) is scale[0]
            and its map (in the local Y direction) scale[1].

        cm_cell_size : [2], tf.float
            Resolution of the coverage map, i.e., width
            (in the local X direction) and height (in the local Y direction) in
            meters of a cell of the coverage map

        los : bool
            If set to `True`, then the LoS paths are computed.

        reflection : bool
            If set to `True`, then the reflected paths are computed.

        diffraction : bool
            If set to `True`, then the diffracted paths are computed.

        scattering : bool
            If set to `True`, then the scattered paths are computed.

        etas : [num_shape], tf.complex
            Tensor containing the complex relative permittivities of all shapes

        scattering_coefficient : [num_shape], tf.float
            Tensor containing the scattering coefficients of all shapes

        xpd_coefficient : [num_shape], tf.float
            Tensor containing the cross-polarization discrimination
            coefficients of all shapes

        alpha_r : [num_shape], tf.float
            Tensor containing the alpha_r scattering parameters of all shapes

        alpha_i : [num_shape], tf.float
            Tensor containing the alpha_i scattering parameters of all shapes

        lambda_ : [num_shape], tf.float
            Tensor containing the lambda_ scattering parameters of all shapes

        Output
        ------
        cm : [num_tx, num_cells_y, num_cells_x], tf.float
            Coverage map for every transmitter.
            Includes LoS, reflection, and scattering.

        los_primitives: [num_los_primitives], int | `None`
            Primitives in LoS.
            `None` is returned if ``diffraction`` is set to `False`.
        """

        # Ensure that sample count can be distributed over the emitters
        num_tx = sources_positions.shape[0]
        samples_per_tx_float = tf.math.ceil(num_samples / num_tx)
        samples_per_tx = int(samples_per_tx_float)
        num_samples = num_tx * samples_per_tx

        # Transmitters and receivers rotation matrices
        # [3, 3]
        rx_rot_mat = rotation_matrix(rx_orientation)
        # [num_tx, 3, 3]
        tx_rot_mat = rotation_matrix(sources_orientations)

        # Rotation matrix to go from the measurement plane LCS to the GCS, and
        # the othwer way around
        # [3,3]
        rot_mp_2_gcs = rotation_matrix(cm_orientation)
        rot_gcs_2_mp = tf.transpose(rot_mp_2_gcs)
        # Normal to the CM
        # Add a dimension for broadcasting
        # [1, 3]
        cm_normal = tf.expand_dims(rot_mp_2_gcs[:,2], axis=0)

        # Number of cells in the coverage map
        # [2,2]
        num_cells_x = tf.cast(tf.math.ceil(cm_size[0]/cm_cell_size[0]),
                              tf.int32)
        num_cells_y = tf.cast(tf.math.ceil(cm_size[1]/cm_cell_size[1]),
                              tf.int32)
        num_cells = tf.stack([num_cells_x, num_cells_y], axis=-1)

        # Primitives in LoS are required for diffraction
        los_primitives = None

        # Initialize rays.
        # Direction arranged in a Fibonacci lattice on the unit
        # sphere.
        # [num_samples, 3]
        k_tx = fibonacci_lattice(samples_per_tx, self._rdtype)
        k_tx = tf.tile(k_tx, [num_tx, 1])
        k_tx_dr = self._mi_vec_t(k_tx)
        # Origin placed on the given transmitters
        # [num_samples]
        samples_tx_indices_dr = dr.linspace(self._mi_scalar_t, 0, num_tx-1e-7,
                                            num=num_samples, endpoint=False)
        samples_tx_indices_dr = mi.Int32(samples_tx_indices_dr)
        samples_tx_indices = mi_to_tf_tensor(samples_tx_indices_dr, tf.int32)
        # [num_samples, 3]
        rays_origin_dr = dr.gather(self._mi_vec_t,
                                   self._mi_tensor_t(sources_positions).array,
                                   samples_tx_indices_dr)
        rays_origin = mi_to_tf_tensor(rays_origin_dr, self._rdtype)
        # Rays
        ray = mi.Ray3f(o=rays_origin_dr, d=k_tx_dr)

        # Previous intersection point. Initialized to the transmitter position
        # [num_samples, 3]
        previous_int_point = rays_origin

        # Initializing the coverage map
        # Add dummy row and columns to store the items that are out of the
        # coverage map
        # [num_tx, num_cells_y+1, num_cells_x+1]
        cm = tf.zeros([num_tx, num_cells_y+1, num_cells_x+1],dtype=self._rdtype)

        for depth in tf.range(max_depth+1):

            ################################################
            # Intersection test
            ################################################

            # Intersect with scene
            si_scene = self._mi_scene.ray_intersect(ray)

            # Intersect with the measurement plane
            si_mp = meas_plane.ray_intersect(ray)

            # Hit the measurement plane?
            # An intersection with the coverage map is only valid if it was
            # not obstructed
            # [num_samples]
            hit_mp_dr = (si_mp.t < si_scene.t) & si_mp.is_valid()
            # [num_samples]
            hit_mp = mi_to_tf_tensor(hit_mp_dr, tf.bool)

            # A ray is active if it interacted with the scene
            # [num_samples]
            active_dr = si_scene.is_valid()
            # [num_samples]
            active = mi_to_tf_tensor(active_dr, tf.bool)

            # Discard LoS if requested
            # [num_samples]
            hit_mp &= (los or (depth > 0))

            ################################################
            # Initialize the electric field
            ################################################

            # The field is initialized with the transmit field in the GCS
            # at the first iteration for rays that either hit the coverage map
            # or are active
            if depth == 0:
                init_ray_dr = active_dr | si_mp.is_valid()
                init_ray = mi_to_tf_tensor(init_ray_dr, tf.bool)
                e_field, field_es, field_ep = self._init_e_field(init_ray,
                samples_tx_indices, k_tx, tx_rot_mat)

            ################################################
            # Update the coverage map
            ################################################

            # Intersection point with the measurement plane
            # [num_samples, 3]
            mp_hit_point = ray.o + si_mp.t*ray.d
            mp_hit_point = mi_to_tf_tensor(mp_hit_point, self._rdtype)

            cm = self._update_coverage_map(cm_center, cm_size,
                cm_cell_size, num_cells, rot_gcs_2_mp, cm_normal, tx_rot_mat,
                rx_rot_mat, precoding_vec, combining_vec, samples_tx_indices,
                e_field, field_es, field_ep, mp_hit_point, hit_mp, k_tx,
                previous_int_point, cm)

            # If the maximum requested depth is reached, we stop, as we just
            # updated the coverage map with the last requested contribution from
            # the rays.
            # We also stop if there is no remaining active ray.
            if (depth == max_depth) or (not tf.reduce_any(active)):
                break

            #############################################
            # Extract primitives that were hit by
            # active rays.
            #############################################

            # Extract the primitives that were hit
            # Primitives that were hit
            shape_i = dr.gather(mi.Int32, self._shape_indices,
                                dr.reinterpret_array_v(mi.UInt32,
                                                       si_scene.shape),
                                active_dr)
            offsets = dr.gather(mi.Int32, self._prim_offsets, shape_i,
                                active_dr)
            # [num_samples]
            primitives = dr.select(active_dr, offsets + si_scene.prim_index, -1)
            primitives = mi_to_tf_tensor(primitives, tf.int32)

            # If diffraction is enabled, stores the primitives in LoS
            # for sampling their wedges. These are needed to compute the
            # coverage map for diffraction (not in this function).
            if diffraction and (depth == 0):
                # [num_samples]
                los_primitives = primitives

            # At this point, max_depth > 0 and there are still active rays.
            # However, we can stop if neither reflection or scattering is
            # enabled, as only these phenomena require to go further.
            if not (reflection or scattering):
                break

            #############################################
            # Update the field.
            # Only active rays are updated.
            #############################################

            # Intersection point
            # [num_samples, 3]
            int_point = ray.o + si_scene.t*ray.d
            int_point = mi_to_tf_tensor(int_point, self._rdtype)

            # Sample scattering/reflection phenomena.
            # reflect_ind : [num_reflected_samples]
            #   Indices of the rays that are reflected
            #  scatter_ind : [num_scattered_samples]
            #   Indices of the rays that are scattered
            reflect_ind, scatter_ind = self._sample_interaction_phenomena(
                                active, primitives, scattering_coefficient,
                                reflection, scattering)

            updated_e_field = tf.zeros([0, e_field.shape[1], 2], self._dtype)
            updated_field_es = tf.zeros([0, 3], self._rdtype)
            updated_field_ep = tf.zeros([0, 3], self._rdtype)
            updated_int_point = tf.zeros([0, 3], self._rdtype)
            updated_k_r = tf.zeros([0, 3], self._rdtype)

            if tf.shape(reflect_ind)[0] > 0:
                # ref_e_field : [num_reflected_samples, num_tx_patterns, 2]
                # ref_field_es : [num_reflected_samples, 3]
                # ref_field_ep : [num_reflected_samples, 3]
                # ref_int_point : [num_reflected_samples, 3]
                # ref_k_r : [num_reflected_samples, 3]
                ref_e_field, ref_field_es, ref_field_ep, ref_int_point, ref_k_r\
                    = self._apply_reflection(reflect_ind, int_point,
                        previous_int_point, primitives, e_field, field_es,
                        field_ep, etas, scattering_coefficient, scattering)

                updated_e_field = tf.concat([updated_e_field, ref_e_field],
                                            axis=0)
                updated_field_es = tf.concat([updated_field_es, ref_field_es],
                                                axis=0)
                updated_field_ep = tf.concat([updated_field_ep, ref_field_ep],
                                                axis=0)
                updated_int_point = tf.concat([updated_int_point,ref_int_point],
                                                axis=0)
                updated_k_r = tf.concat([updated_k_r, ref_k_r], axis=0)

            if tf.shape(scatter_ind)[0] > 0:
                # scat_e_field : [num_reflected_samples, num_tx_patterns, 2]
                # scat_field_es : [num_reflected_samples, 3]
                # scat_field_ep : [num_reflected_samples, 3]
                # scat_int_point : [num_reflected_samples, 3]
                # scat_k_r : [num_reflected_samples, 3]
                scat_e_field, scat_field_es, scat_field_ep, scat_int_point,\
                    scat_k_r = self._apply_scattering(scatter_ind,
                        int_point, previous_int_point, primitives, e_field,
                        field_es, field_ep, etas, scattering_coefficient,
                        xpd_coefficient, alpha_r, alpha_i, lambda_, reflection)

                updated_e_field = tf.concat([updated_e_field, scat_e_field],
                                            axis=0)
                updated_field_es = tf.concat([updated_field_es, scat_field_es],
                                                axis=0)
                updated_field_ep = tf.concat([updated_field_ep, scat_field_ep],
                                                axis=0)
                updated_int_point = tf.concat([updated_int_point,
                                                scat_int_point], axis=0)
                updated_k_r = tf.concat([updated_k_r, scat_k_r], axis=0)

            e_field = updated_e_field
            field_es = updated_field_es
            field_ep = updated_field_ep
            k_r = updated_k_r
            int_point = updated_int_point

            ###############################################
            # Reflect or scatter the current ray
            ###############################################

            # Spawn a new rays
            # [num_active_samples, 3]
            k_r_dr = self._mi_vec_t(k_r)
            rays_origin_dr = self._mi_vec_t(int_point)
            rays_origin_dr += SolverBase.EPSILON_OBSTRUCTION*k_r_dr
            ray = mi.Ray3f(o=rays_origin_dr, d=k_r_dr)
            # Update previous intersection point
            # [num_active_samples, 3]
            previous_int_point = int_point

        #################################################
        # Finalize the computation of the coverage map
        #################################################

        # Scaling factor
        cell_area = cm_cell_size[0]*cm_cell_size[1]
        cst = tf.cast(4.*PI*cell_area*samples_per_tx_float, self._rdtype)
        cm_scaling = tf.square(self._scene.wavelength)/cst
        cm_scaling = tf.cast(cm_scaling, self._rdtype)

        # Dump the dummy line and row and apply the scaling factor
        # [num_tx, num_cells_y, num_cells_x]
        cm = cm_scaling*cm[:,:num_cells_y,:num_cells_x]

        # For diffraction, we need only primitives in LoS
        # [num_los_primitives]
        if los_primitives is not None:
            los_primitives,_ = tf.unique(los_primitives)

        return cm, los_primitives

    def _discard_obstructing_wedges(self, candidate_wedges, sources_positions):
        r"""
        Discard wedges for which the source is "inside" the wedge

        Input
        ------
        candidate_wedges : [num_candidate_wedges], int
            Candidate wedges.
            Entries correspond to wedges indices.

        sources_positions : [num_tx, 3], tf.float
            Coordinates of the sources.

        Output
        -------
        diff_mask : [num_tx, num_candidate_wedges], tf.bool
            Mask set to False for invalid wedges

        diff_wedges_ind : [num_candidate_wedges], tf.int
            Indices of the wedges that interacted with the diffracted paths
        """

        epsilon = tf.cast(SolverBase.EPSILON, self._rdtype)

        # [num_candidate_wedges, 3]
        origins = tf.gather(self._wedges_origin, candidate_wedges)

        # Expand to broadcast with sources/targets and 0/n faces
        # [1, num_candidate_wedges, 1, 3]
        origins = tf.expand_dims(origins, axis=0)
        origins = tf.expand_dims(origins, axis=2)

        # Normals
        # [num_candidate_wedges, 2, 3]
        # [:,0,:] : 0-face
        # [:,1,:] : n-face
        normals = tf.gather(self._wedges_normals, candidate_wedges)
        # Expand to broadcast with the sources or targets
        # [1, num_candidate_wedges, 2, 3]
        normals = tf.expand_dims(normals, axis=0)

        # Expand to broadcast with candidate and 0/n faces wedges
        # [num_tx, 1, 1, 3]
        sources_positions = expand_to_rank(sources_positions, 4, 1)
        # Sources vectors
        # [num_tx, num_candidate_wedges, 1, 3]
        u_t = sources_positions - origins

        # [num_tx, num_candidate_wedges, 2]
        mask = dot(u_t, normals)
        mask = tf.greater(mask, tf.fill(tf.shape(mask), epsilon))
        # [num_tx, num_candidate_wedges]
        mask = tf.reduce_any(mask, axis=2)

        # Discard wedges with no valid link
        # [num_candidate_wedges]
        valid_wedges = tf.where(tf.reduce_any(mask, axis=0))[:,0]
        # [num_tx, num_candidate_wedges]
        mask = tf.gather(mask, valid_wedges, axis=1)
        # [num_candidate_wedges]
        diff_wedges_ind = tf.gather(candidate_wedges, valid_wedges, axis=0)

        return mask, diff_wedges_ind

    def _sample_wedge_points(self, diff_mask, diff_wedges_ind, num_samples):
        r"""

        Samples equally spaced candidate diffraction points on the candidate
        wedges.

        The distance between two points is the cumulative length of the
        candidate wedges divided by ``num_samples``, i.e., the density of
        samples is the same for all wedges.

        The `num_samples` dimension of the output tensors is in general slighly
        smaller than input `num_samples` because of roundings.

        Input
        ------
        diff_mask : [num_tx, num_samples], tf.bool
            Mask set to False for invalid samples

        diff_wedges_ind : [num_candidate_wedges], int
            Candidate wedges indices

        num_samples : int
            Number of samples to shoot

        Output
        ------
        diff_mask : [num_tx, num_samples], tf.bool
            Mask set to False for invalid wedges

        diff_wedges_ind : [num_samples], tf.int
            Indices of the wedges that interacted with the diffracted paths

        diff_ells : [num_samples], tf.float
            Positions of the diffraction points on the wedges.
            These positions are given as an offset from the wedges origins.

        diff_vertex : [num_samples, 3], tf.float
            Positions of the diffracted points in the GCS

        diff_num_samples_per_wedge : [num_samples], tf.int
            For each sample, total mumber of samples that were sampled on the
            same wedge
        """

        zero_dot_five = tf.cast(0.5, self._rdtype)

        # [num_candidate_wedges]
        wedges_length = tf.gather(self._wedges_length, diff_wedges_ind)
        # Total length of the wedges
        # ()
        wedges_total_length = tf.reduce_sum(wedges_length)
        # Spacing between the samples
        # ()
        delta_ell = wedges_total_length/tf.cast(num_samples,self._rdtype)
        # Number of samples for each wedge
        # [num_candidate_wedges]
        samples_per_wedge = tf.math.divide_no_nan(wedges_length, delta_ell)
        samples_per_wedge = tf.cast(tf.floor(samples_per_wedge), tf.int32)
        # Maximum number of samples for a wedge
        # tf.maximum() required for the case where samples_per_wedge is empty
        # ()
        max_samples_per_wedge = tf.maximum(tf.reduce_max(samples_per_wedge), 0)
        # Sequence used to build the equally spaced samples on the wedges
        # [max_samples_per_wedge]
        cseq = tf.cumsum(tf.ones([max_samples_per_wedge], dtype=tf.int32)) - 1
        # [1, max_samples_per_wedge]
        cseq = tf.expand_dims(cseq, axis=0)
        # [num_candidate_wedges, 1]
        samples_per_wedge_ = tf.expand_dims(samples_per_wedge, axis=1)
        # [num_candidate_wedges, max_samples_per_wedge]
        ells_i = tf.where(cseq < samples_per_wedge_, cseq,
                          max_samples_per_wedge)
        # Compute the relative offset of the diffraction point on the wedge
        # [num_candidate_wedges, max_samples_per_wedge]
        ells = (tf.cast(ells_i, self._rdtype) + zero_dot_five)*delta_ell
        # [num_candidate_wedges x max_samples_per_wedge]
        ells_i = tf.reshape(ells_i, [-1])
        ells = tf.reshape(ells, [-1])
        # Extract only relevant indices
        # [num_samples]. Smaller but close than input num_samples in general
        # because of previous floor() op
        ells = tf.gather(ells, tf.where(ells_i < max_samples_per_wedge))[:,0]

        # Compute the corresponding points coordinates in the GCS
        # Wedges origin
        # [num_candidate_wedges, 3]
        origins = tf.gather(self._wedges_origin, diff_wedges_ind)
        # Wedges directions
        # [num_candidate_wedges, 3]
        e_hat = tf.gather(self._wedges_e_hat, diff_wedges_ind)
        # Match each sample to the corresponding wedge origin and vector
        # First, generate the indices for the gather op
        # ()
        num_candidate_wedges = diff_wedges_ind.shape[0]
        # [num_candidate_wedges]
        gather_ind = tf.range(num_candidate_wedges)
        gather_ind = tf.expand_dims(gather_ind, axis=1)
        # [num_candidate_wedges, max_samples_per_wedge]
        gather_ind = tf.where(cseq < samples_per_wedge_, gather_ind,
                              num_candidate_wedges)
        # [num_candidate_wedges x max_samples_per_wedge]
        gather_ind = tf.reshape(gather_ind, [-1])
        # [num_samples]
        gather_ind = tf.gather(gather_ind,
                               tf.where(ells_i < max_samples_per_wedge))[:,0]
        # [num_samples, 3]
        origins = tf.gather(origins, gather_ind, axis=0)
        e_hat = tf.gather(e_hat, gather_ind, axis=0)
        # [num_samples]
        diff_wedges_ind = tf.gather(diff_wedges_ind, gather_ind, axis=0)
        # [num_tx, num_samples]
        diff_mask = tf.gather(diff_mask, gather_ind, axis=1)
        # Positions of the diffracted points in the GCS
        # [num_samples, 3]
        diff_points = origins + tf.expand_dims(ells, axis=1)*e_hat
        # Number of samples per wedge
        # [num_samples]
        samples_per_wedge = tf.gather(samples_per_wedge, gather_ind, axis=0)

        return diff_mask, diff_wedges_ind, ells, diff_points, samples_per_wedge

    def _test_tx_visibility(self, diff_mask, diff_wedges_ind, diff_ells,
                            diff_vertex, diff_num_samples_per_wedge,
                            sources_positions):
        r"""
        Test for blockage between the diffraction points and the transmitters.
        Blocked samples are discarded.

        Input
        ------
        diff_mask : [num_tx, num_samples], tf.bool
            Mask set to False for invalid samples

        diff_wedges_ind : [num_samples], tf.int
            Indices of the wedges that interacted with the diffracted paths

        diff_ells : [num_samples], tf.float
            Positions of the diffraction points on the wedges.
            These positions are given as an offset from the wedges origins.

        diff_vertex : [num_samples, 3], tf.float
            Positions of the diffracted points in the GCS

        diff_num_samples_per_wedge : [num_samples], tf.int
                For each sample, total mumber of samples that were sampled on
                the same wedge

        sources_positions : [num_tx, 3], tf.float
            Positions of the transmitters.

        Output
        -------
        diff_mask : [num_tx, num_samples], tf.bool
            Mask set to False for invalid wedges

        diff_wedges_ind : [num_samples], tf.int
            Indices of the wedges that interacted with the diffracted paths

        diff_ells : [num_samples], tf.float
            Positions of the diffraction points on the wedges.
            These positions are given as an offset from the wedges origins.

        diff_vertex : [num_samples, 3], tf.float
            Positions of the diffracted points in the GCS

        diff_num_samples_per_wedge : [num_samples], tf.int
                For each sample, total mumber of samples that were sampled on
                the same wedge
        """

        num_tx = sources_positions.shape[0]
        num_samples = diff_vertex.shape[0]

        # [num_tx, 1, 3]
        sources_positions = tf.expand_dims(sources_positions, axis=1)
        # [1, num_samples, 3]
        wedges_diff_points_ = tf.expand_dims(diff_vertex, axis=0)
        # Ray directions and maximum distance for obstruction test
        # ray_dir : [num_tx, num_samples, 3]
        # maxt : [num_tx, num_samples]
        ray_dir,maxt = normalize(sources_positions - wedges_diff_points_)
        # Ray origins
        # [num_tx, num_samples, 3]
        ray_org = tf.tile(wedges_diff_points_, [num_tx, 1, 1])

        # Test for obstruction
        # [num_tx, num_samples]
        ray_org = tf.reshape(ray_org, [-1,3])
        ray_dir = tf.reshape(ray_dir, [-1,3])
        maxt = tf.reshape(maxt, [-1])
        invalid = self._test_obstruction(ray_org, ray_dir, maxt)
        invalid = tf.reshape(invalid, [num_tx, num_samples])

        # Remove discarded paths
        # [num_tx, num_samples]
        diff_mask = tf.logical_and(diff_mask, ~invalid)
        # Discard samples with no valid link
        # [num_candidate_wedges]
        valid_samples = tf.where(tf.reduce_any(diff_mask, axis=0))[:,0]
        # [num_tx, num_samples]
        diff_mask = tf.gather(diff_mask, valid_samples, axis=1)
        # [num_samples]
        diff_wedges_ind = tf.gather(diff_wedges_ind, valid_samples, axis=0)
        # [num_samples]
        diff_vertex = tf.gather(diff_vertex, valid_samples,
                                       axis=0)
        # [num_samples]
        diff_ells = tf.gather(diff_ells, valid_samples, axis=0)
        # [num_samples]
        diff_num_samples_per_wedge = tf.gather(diff_num_samples_per_wedge,
                                               valid_samples, axis=0)

        return diff_mask, diff_wedges_ind, diff_ells, diff_vertex,\
            diff_num_samples_per_wedge

    def _sample_diff_angles(self, diff_wedges_ind):
        r"""
        Samples angles of diffracted ray on the diffraction cone

        Input
        ------
        diff_wedges_ind : [num_samples], tf.int
            Indices of the wedges that interacted with the diffracted paths

        Output
        -------
        diff_phi : [num_samples], tf.float
            Sampled angles of diffracted rays on the diffraction cone
        """

        num_samples = diff_wedges_ind.shape[0]

        # [num_samples, 2, 3]
        normals = tf.gather(self._wedges_normals, diff_wedges_ind,  axis=0)

        # Compute the wedges angle
        # [num_samples]
        wedges_angle = PI + tf.math.acos(dot(normals[:,0,:],normals[:,1,:]))

        # Uniformly sample angles for shooting rays on the diffraction cone
        # [num_samples]
        phis = tf.random.uniform([num_samples],
                                 minval=tf.zeros_like(wedges_angle),
                                 maxval=wedges_angle,
                                 dtype=self._rdtype)

        return phis

    def _shoot_diffracted_rays(self, diff_mask, diff_wedges_ind, diff_ells,
                               diff_vertex, diff_num_samples_per_wedge,
                               diff_phi, sources_positions, meas_plane):
        r"""
        Shoots the diffracted rays and computes their intersection with the
        coverage map, if any. Rays blocked by the scene are discarded. Rays
        that do not hit the coverage map are discarded.

        Input
        ------
        diff_mask : [num_tx, num_samples], tf.bool
            Mask set to False for invalid samples

        diff_wedges_ind : [num_samples], tf.int
            Indices of the wedges that interacted with the diffracted paths

        diff_ells : [num_samples], tf.float
            Positions of the diffraction points on the wedges.
            These positions are given as an offset from the wedges origins.

        diff_vertex : [num_samples, 3], tf.float
            Positions of the diffracted points in the GCS

        diff_num_samples_per_wedge : [num_samples], tf.int
            For each sample, total mumber of samples that were sampled on the
            same wedge

        diff_phi : [num_samples], tf.float
            Sampled angles of diffracted rays on the diffraction cone

        sources_positions : [num_tx, 3], tf.float
            Positions of the transmitters.

        meas_plane : mi.Shape
            Mitsuba rectangle defining the measurement plane

        Output
        -------
        diff_mask : [num_tx, num_samples], tf.bool
            Mask set to False for invalid samples

        diff_wedges_ind : [num_samples], tf.int
            Indices of the wedges that interacted with the diffracted paths

        diff_ells : [num_samples], tf.float
            Positions of the diffraction points on the wedges.
            These positions are given as an offset from the wedges origins.

        diff_phi : [num_samples], tf.float
            Sampled angles of diffracted rays on the diffraction cone

        diff_vertex : [num_tx, num_samples, 3], tf.float
            Positions of the diffracted points in the GCS

        diff_num_samples_per_wedge : [num_samples], tf.int
            For each sample, total mumber of samples that were sampled on the
            same wedge

        diff_hit_points : [num_tx, num_samples, 3], tf.float
            Positions of the intersection of the diffracted rays and coverage
            map

        diff_cone_angle : [num_tx, num_samples], tf.float
            Angle between e_hat and the diffracted ray direction.
            Takes value in (0,pi).
        """

        # [num_tx, 1, 3]
        sources_positions = tf.expand_dims(sources_positions, axis=1)
        # [1, num_samples, 3]
        diff_vertex = tf.expand_dims(diff_vertex, axis=0)
        # Ray directions and maximum distance for obstruction test
        # ray_dir : [num_tx, num_samples, 3]
        # maxt : [num_tx, num_samples]
        ray_dir,_ = normalize(diff_vertex - sources_positions)

        # Edge vector
        # [num_samples, 3]
        e_hat = tf.gather(self._wedges_e_hat, diff_wedges_ind)
        # [1, num_samples, 3]
        e_hat_ = tf.expand_dims(e_hat, axis=0)
        # Angles between the incident ray and wedge.
        # This angle is not beta_0. It takes values in (0,pi), and is the angle
        # with respect to e_hat in which to shoot the diffracted ray.
        # [num_tx, num_samples]
        theta_shoot_dir = acos_diff(dot(ray_dir, e_hat_))

        # Discard paths for which the incident ray is aligned or perpendicular
        # to the edge
        # [num_tx, num_samples, 3]
        invalid_angle = tf.stack([
            theta_shoot_dir < SolverBase.EPSILON,
            theta_shoot_dir > PI - SolverBase.EPSILON,
            tf.abs(theta_shoot_dir - 0.5*PI) < SolverBase.EPSILON],
                                 axis=-1)
        # [num_tx, num_samples]
        invalid_angle = tf.reduce_any(invalid_angle, axis=-1)

        num_tx = diff_mask.shape[0]

        # Build the direction of the diffracted ray in the LCS
        # The LCS is defined by (t_0_hat, n0_hat, e_hat)

        # Direction of the diffracted ray
        # [1, num_samples]
        phis = tf.expand_dims(diff_phi, axis=0)
        # [num_tx, num_samples, 3]
        diff_dir = r_hat(theta_shoot_dir, phis)

        # Matrix for going from the LCS to the GCS

        # Normals to face 0
        # [num_samples, 2, 3]
        normals = tf.gather(self._wedges_normals, diff_wedges_ind, axis=0)
        # [num_samples, 3]
        normals = normals[:,0,:]
        # Tangent vector t_hat
        # [num_samples, 3]
        t_hat = cross(normals, e_hat)
        # Matrix for going from LCS to GCS
        # [num_samples, 3, 3]
        lcs2gcs = tf.stack([t_hat, normals, e_hat], axis=-1)
        # [1, num_samples, 3, 3]
        lcs2gcs = tf.expand_dims(lcs2gcs, axis=0)

        # Direction of diffracted rays in CGS

        # [num_tx, num_samples, 3]
        diff_dir = tf.linalg.matvec(lcs2gcs, diff_dir)

        # Origin of the diffracted rays

        # [num_tx, num_samples, 3]
        diff_points = tf.tile(diff_vertex, [num_tx, 1, 1])

        # Test of intersection of the diffracted rays with the measurement
        # plane
        mi_diff_dir = self._mi_vec_t(tf.reshape(diff_dir, [-1, 3]))
        mi_diff_points = self._mi_vec_t(tf.reshape(diff_points, [-1, 3]))
        rays = mi.Ray3f(o=mi_diff_points, d=mi_diff_dir)
        # Intersect with the coverage map
        si_mp = meas_plane.ray_intersect(rays)

        # Check for obstruction
        # [num_tx x num_samples]
        obstructed = self._test_obstruction(mi_diff_points, mi_diff_dir,
                                            si_mp.t)

        # Mask invalid rays, i.e., rays that are obstructed or do that not hit
        # the measurement plane, and discard rays that are invalid for all TXs

        # [num_tx x num_samples]
        maxt = mi_to_tf_tensor(si_mp.t, dtype=self._rdtype)
        # [num_tx x num_samples]
        invalid = tf.logical_or(tf.math.is_inf(maxt), obstructed)
        # [num_tx, num_samples]
        invalid = tf.reshape(invalid, [num_tx, -1])
        # [num_tx, num_samples]
        invalid = tf.logical_or(invalid, invalid_angle)
        # [num_tx, num_samples]
        diff_mask = tf.logical_and(diff_mask, ~invalid)
        # Discard samples with no valid link
        # [num_candidate_wedges]
        valid_samples = tf.where(tf.reduce_any(diff_mask, axis=0))[:,0]
        # [num_tx, num_samples]
        diff_mask = tf.gather(diff_mask, valid_samples, axis=1)
        # [num_samples]
        diff_wedges_ind = tf.gather(diff_wedges_ind, valid_samples, axis=0)
        # [num_samples]
        diff_ells = tf.gather(diff_ells, valid_samples, axis=0)
        # [num_samples]
        diff_phi = tf.gather(diff_phi, valid_samples, axis=0)
        # [num_tx, num_samples]
        theta_shoot_dir = tf.gather(theta_shoot_dir, valid_samples, axis=1)
        # [num_samples]
        diff_num_samples_per_wedge = tf.gather(diff_num_samples_per_wedge,
                                               valid_samples, axis=0)

        # Compute intersection point with the coverage map
        # [num_tx, num_samples]
        maxt = tf.reshape(maxt, [num_tx, -1])
        # [num_tx, num_samples]
        maxt = tf.gather(maxt, valid_samples, axis=1)
        # Zeros invalid samples to avoid numeric issues
        # [num_tx, num_samples]
        maxt = tf.where(diff_mask, maxt, tf.zeros_like(maxt))
        # [num_tx, num_samples, 1]
        maxt = tf.expand_dims(maxt, -1)
        # [num_tx, num_samples, 3]
        diff_dir = tf.gather(diff_dir, valid_samples, axis=1)
        # [num_tx, num_samples, 3]
        diff_points = tf.gather(diff_points, valid_samples, axis=1)
        # [num_tx, num_samples, 3]
        diff_hit_points = diff_points + maxt*diff_dir

        return diff_mask, diff_wedges_ind, diff_ells, diff_phi,\
            diff_points, diff_num_samples_per_wedge, diff_hit_points,\
                theta_shoot_dir

    def _compute_samples_weights(self, cm_center, cm_orientation,
        sources_positions, diff_wedges_ind, diff_ells, diff_phi,
        diff_cone_angle):
        r"""
        Computes the weights for averaging the field powers of the samples to
        compute the Monte Carlo estimate of the integral of the diffracted field
        power over the measurement plane.

        These weights are required as the measurement plane is parametrized by
        the angle on the diffraction cones (phi) and position on the wedges
        (ell).

        Input
        ------
        cm_center : [3], tf.float
            Center of the coverage map

        cm_orientation : [3], tf.float
            Orientation of the coverage map

        sources_positions : [num_tx, 3], tf.float
            Coordinates of the sources

        diff_wedges_ind : [num_samples], tf.int
            Indices of the wedges that interacted with the diffracted paths

        diff_ells : [num_samples], tf.float
            Positions of the diffraction points on the wedges.
            These positions are given as an offset from the wedges origins

        diff_phi : [num_samples], tf.float
            Sampled angles of diffracted rays on the diffraction cone

        diff_cone_angle : [num_tx, num_samples], tf.float
            Angle between e_hat and the diffracted ray direction.
            Takes value in (0,pi).

        Output
        ------
        diff_samples_weights : [num_tx, num_samples], tf.float
            Weights for averaging the field powers of the samples.
        """
        cos = tf.math.cos
        sin = tf.math.sin

        # [1, 1, 3]
        cm_center = expand_to_rank(cm_center, 3, 0)
        # [num_tx, 1, 3]
        sources_positions = tf.expand_dims(sources_positions, axis=1)

        # Normal to the coverage map
        # [3]
        cmo_z = cm_orientation[0]
        cmo_y = cm_orientation[1]
        cmo_x = cm_orientation[2]
        cm_normal = tf.stack([
            cos(cmo_z)*sin(cmo_y)*cos(cmo_x) + sin(cmo_z)*sin(cmo_x),
            sin(cmo_z)*sin(cmo_y)*cos(cmo_x) - cos(cmo_z)*sin(cmo_x),
            cos(cmo_y)*cos(cmo_x)],
                             axis=0)
        # [1, 1, 3]
        cm_normal = expand_to_rank(cm_normal, 3, 0)


        # Origins
        # [num_samples, 3]
        origins = tf.gather(self._wedges_origin, diff_wedges_ind)
        # [1, num_samples, 3]
        origins = tf.expand_dims(origins, axis=0)

        # Distance of the wedge to the measurement plane
        # [num_tx, num_samples]
        wedge_cm_dist = dot(cm_center - origins, cm_normal)

        # Edges vectors
        # [num_samples, 3]
        e_hat = tf.gather(self._wedges_e_hat, diff_wedges_ind)

        # Normals to face 0
        # [num_samples, 2, 3]
        normals = tf.gather(self._wedges_normals, diff_wedges_ind, axis=0)
        # [num_samples, 3]
        normals = normals[:,0,:]
        # Tangent vector t_hat
        # [num_samples, 3]
        t_hat = cross(normals, e_hat)
        # Matrix for going from LCS to GCS
        # [num_samples, 3, 3]
        gcs2lcs = tf.stack([t_hat, normals, e_hat], axis=-2)
        # [1, num_samples, 3, 3]
        gcs2lcs = tf.expand_dims(gcs2lcs, axis=0)
        # Normal in LCS
        # [1, num_samples, 3]
        cm_normal = tf.linalg.matvec(gcs2lcs, cm_normal)

        # Projections of the transmitters on the wedges
        # [1, num_samples, 3]
        e_hat = tf.expand_dims(e_hat, axis=0)
        # [num_tx, num_samples]
        tx_proj_org_dist = dot(sources_positions - origins, e_hat)
        # [num_tx, num_samples, 1]
        tx_proj_org_dist_ = tf.expand_dims(tx_proj_org_dist, axis=2)

        # Position of the sources projections on the wedges
        # [num_tx, num_samples, 3]
        tx_proj_pos = origins + tx_proj_org_dist_*e_hat
        # Distance of transmitters to wedges
        # [num_tx, num_samples]
        tx_wedge_dist = tf.linalg.norm(tx_proj_pos - sources_positions, axis=-1)

        # Building the derivatives of the parametrization of the intersection
        # of the diffraction cone and measurement plane
        # [1, num_samples]
        diff_phi = tf.expand_dims(diff_phi, axis=0)
        # [1, num_samples]
        diff_ells = tf.expand_dims(diff_ells, axis=0)

        # [1, num_samples]
        cos_phi = cos(diff_phi)
        # [1, num_samples]
        sin_phi = sin(diff_phi)
        # [1, num_samples]
        xy_dot = cm_normal[...,0]*cos_phi + cm_normal[...,1]*sin_phi
        # [num_tx, num_samples]
        ell_min_d = diff_ells - tx_proj_org_dist
        # [num_tx, num_samples]
        u = tf.math.sign(ell_min_d)
        # [num_tx, num_samples]
        ell_min_d = tf.math.abs(ell_min_d)
        # [num_tx, num_samples]
        s = tf.where(diff_cone_angle < 0.5*PI,
                     tf.ones_like(diff_cone_angle),
                     -tf.ones_like(diff_cone_angle))
        # [num_tx, num_samples]
        q = s*tx_wedge_dist*xy_dot + cm_normal[...,2]*ell_min_d
        q_square = tf.square(q)
        inv_q = tf.math.divide_no_nan(tf.ones_like(q), q)
        # [num_tx, num_samples]
        big_d_min_lz = wedge_cm_dist - diff_ells*cm_normal[...,2]

        # [num_tx, num_samples, 3]
        v1 = tf.stack([
                s*big_d_min_lz*tx_wedge_dist*cos_phi,
                s*big_d_min_lz*tx_wedge_dist*sin_phi,
                wedge_cm_dist*ell_min_d + s*diff_ells*tx_wedge_dist*xy_dot],
                      axis=-1)
        # [num_tx, num_samples, 3]
        v2 = tf.stack([-s*cm_normal[...,2]*tx_wedge_dist*cos_phi,
                       -s*cm_normal[...,2]*tx_wedge_dist*sin_phi,
                       u*wedge_cm_dist + s*tx_wedge_dist*xy_dot],
                      axis=-1)
        # Derivative with respect to ell
        # [num_tx, num_samples, 3]
        ds_dl = tf.expand_dims(tf.math.divide_no_nan(-u*cm_normal[...,2],
                                                      q_square), axis=-1)*v1
        ds_dl = ds_dl + tf.expand_dims(inv_q, axis=-1)*v2

        # Derivative with respect to phi
        # [num_tx, num_samples]
        w = -cm_normal[...,0]*sin_phi + cm_normal[...,1]*cos_phi
        # [num_tx, num_samples, 3]
        v3 = tf.stack([-s*big_d_min_lz*tx_wedge_dist*sin_phi,
                       s*big_d_min_lz*tx_wedge_dist*cos_phi,
                       s*diff_ells*tx_wedge_dist*w],
                      axis=-1)
        # [num_tx, num_samples, 3]
        ds_dphi = tf.expand_dims(tf.math.divide_no_nan(
            -s*tx_wedge_dist*w, q_square), axis=-1)*v1
        ds_dphi = ds_dphi + tf.expand_dims(inv_q, axis=-1)*v3

        # Weighting
        # [num_tx, num_samples]
        diff_samples_weights = tf.linalg.norm(cross(ds_dl, ds_dphi), axis=-1)

        return diff_samples_weights

    def _compute_diffracted_path_power(self,
                                       sources_positions,
                                       sources_orientations,
                                       rx_orientation,
                                       combining_vec,
                                       precoding_vec,
                                       diff_mask,
                                       diff_wedges_ind,
                                       diff_vertex,
                                       diff_hit_points,
                                       relative_permittivity,
                                       scattering_coefficient):
        """
        Computes the power of the diffracted paths.

        Input
        ------
        sources_positions : [num_tx, 3], tf.float
            Positions of the transmitters.

        sources_orientations : [num_tx, 3], tf.float
            Orientations of the sources.

        rx_orientation : [3], tf.float
            Orientation of the receiver.
            This is used to compute the antenna response and antenna pattern
            for an imaginary receiver located on the coverage map.

        combining_vec : [num_rx_ant], tf.complex
            Combining vector.
            This is used to combine the signal from the receive antennas for
            an imaginary receiver located on the coverage map.

        precoding_vec : [num_tx or 1, num_tx_ant], tf.complex
            Precoding vectors of the transmitters

        diff_mask : [num_tx, num_samples], tf.bool
            Mask set to False for invalid samples

        diff_wedges_ind : [num_samples], tf.int
            Indices of the wedges that interacted with the diffracted paths

        diff_vertex : [num_tx, num_samples, 3], tf.float
            Positions of the diffracted points in the GCS

        diff_hit_points : [num_tx, num_samples, 3], tf.float
            Positions of the intersection of the diffracted rays and coverage
            map

        relative_permittivity : [num_shape], tf.complex
            Tensor containing the complex relative permittivity of all objects

        scattering_coefficient : [num_shape], tf.float
            Tensor containing the scattering coefficients of all objects

        Output
        ------
        diff_samples_power : [num_tx, num_samples], tf.float
            Powers of the samples of diffracted rays.
        """

        def f(x):
            """F(x) Eq.(88) in [ITUR_P526]
            """
            sqrt_x = tf.sqrt(x)
            sqrt_pi_2 = tf.cast(tf.sqrt(PI/2.), x.dtype)

            # Fresnel integral
            arg = sqrt_x/sqrt_pi_2
            s = tf.math.special.fresnel_sin(arg)
            c = tf.math.special.fresnel_cos(arg)
            f = tf.complex(s, c)

            zero = tf.cast(0, x.dtype)
            one = tf.cast(1, x.dtype)
            two = tf.cast(2, f.dtype)
            factor = tf.complex(sqrt_pi_2*sqrt_x, zero)
            factor = factor*tf.exp(tf.complex(zero, x))
            res =  tf.complex(one, one) - two*f

            return factor* res

        wavelength = self._scene.wavelength
        k = 2.*PI/wavelength

        # On CPU, indexing with -1 does not work. Hence we replace -1 by 0.
        # This makes no difference on the resulting paths as such paths
        # are not flaged as active.
        # [num_samples]
        valid_wedges_idx = tf.where(diff_wedges_ind == -1, 0, diff_wedges_ind)

        # [num_tx, 1, 3]
        sources_positions = tf.expand_dims(sources_positions, axis=1)

        # Normals
        # [num_samples, 2, 3]
        normals = tf.gather(self._wedges_normals, valid_wedges_idx, axis=0)

        # Compute the wedges angle
        # [num_samples]
        wedges_angle = PI - tf.math.acos(dot(normals[...,0,:],normals[...,1,:]))
        n = (2.*PI-wedges_angle)/PI
        # [1, num_samples]
        n = tf.expand_dims(n, axis=0)

        # [num_samples, 3]
        e_hat = tf.gather(self._wedges_e_hat, valid_wedges_idx)
        # [1, num_samples, 3]
        e_hat = tf.expand_dims(e_hat, axis=0)

        # Extract surface normals
        # [num_samples, 3]
        n_0_hat = normals[:,0,:]
        # [1, num_samples, 3]
        n_0_hat = tf.expand_dims(n_0_hat, axis=0)
        # [num_samples, 3]
        n_n_hat = normals[:,1,:]
        # [1, num_samples, 3]
        n_n_hat = tf.expand_dims(n_n_hat, axis=0)

        # Relative permitivities
        # [num_samples, 2]
        objects_indices = tf.gather(self._wedges_objects, valid_wedges_idx,
                                    axis=0)
        # [num_samples, 2]
        etas = tf.gather(relative_permittivity, objects_indices)
        # [num_samples]
        eta_0 = etas[:,0]
        eta_n = etas[:,1]
        # [1, num_samples]
        eta_0 = tf.expand_dims(eta_0, axis=0)
        eta_n = tf.expand_dims(eta_n, axis=0)

        # Get scattering coefficients
        # [num_samples, 2]
        scattering_coefficient = tf.gather(scattering_coefficient,
                                           objects_indices)
        # [num_samples]
        scattering_coefficient_0 = scattering_coefficient[...,0]
        scattering_coefficient_n = scattering_coefficient[...,1]

        # [1, num_samples]
        scattering_coefficient_0 = tf.expand_dims(scattering_coefficient_0,
                                                  axis=0)
        scattering_coefficient_n = tf.expand_dims(scattering_coefficient_n,
                                                  axis=0)

        # Compute s_prime_hat, s_hat, s_prime, s
        # s_prime_hat : [num_tx, num_samples, 3]
        # s_prime : [num_tx, num_samples]
        s_prime_hat, s_prime = normalize(diff_vertex-sources_positions)
        # s_hat : [num_tx, num_samples, 3]
        # s : [num_tx, num_samples]
        s_hat, s = normalize(diff_hit_points-diff_vertex)

        # Compute phi_prime_hat, beta_0_prime_hat, phi_hat, beta_0_hat
        # [num_tx, num_samples, 3]
        phi_prime_hat, _ = normalize(cross(s_prime_hat, e_hat))
        # [num_tx, num_samples, 3]
        beta_0_prime_hat = cross(phi_prime_hat, s_prime_hat)

        # [num_tx, num_samples, 3]
        phi_hat_, _ = normalize(-cross(s_hat, e_hat))
        beta_0_hat = cross(phi_hat_, s_hat)

        # Compute tangent vector t_0_hat
        # [1, num_samples, 3]
        t_0_hat = cross(n_0_hat, e_hat)

        # Compute s_t_prime_hat and s_t_hat
        # [num_tx, num_samples, 3]
        s_t_prime_hat, _ = normalize(s_prime_hat
                                - dot(s_prime_hat,e_hat, keepdim=True)*e_hat)
        # [num_tx, num_samples, 3]
        s_t_hat, _ = normalize(s_hat - dot(s_hat,e_hat, keepdim=True)*e_hat)

        # Compute phi_prime and phi
        # [num_tx, num_samples]
        phi_prime = PI -\
            (PI-acos_diff(-dot(s_t_prime_hat, t_0_hat)))*\
                sign(-dot(s_t_prime_hat, n_0_hat))
        # [num_tx, num_samples]
        phi = PI - (PI-acos_diff(dot(s_t_hat, t_0_hat)))\
            *sign(dot(s_t_hat, n_0_hat))

        # Compute field component vectors for reflections at both surfaces
        # [num_tx, num_samples, 3]
        # pylint: disable=unbalanced-tuple-unpacking
        e_i_s_0, e_i_p_0, e_r_s_0, e_r_p_0 = compute_field_unit_vectors(
            s_prime_hat,
            s_hat,
            n_0_hat,#*sign(-dot(s_t_prime_hat, n_0_hat, keepdim=True)),
            SolverBase.EPSILON
            )
        # [num_tx, num_samples, 3]
        # pylint: disable=unbalanced-tuple-unpacking
        e_i_s_n, e_i_p_n, e_r_s_n, e_r_p_n = compute_field_unit_vectors(
            s_prime_hat,
            s_hat,
            n_n_hat,#*sign(-dot(s_t_prime_hat, n_n_hat, keepdim=True)),
            SolverBase.EPSILON
            )

        # Compute Fresnel reflection coefficients for 0- and n-surfaces
        # [num_tx, num_samples]
        r_s_0, r_p_0 = reflection_coefficient(eta_0, tf.abs(tf.sin(phi_prime)))
        r_s_n, r_p_n = reflection_coefficient(eta_n, tf.abs(tf.sin(n*PI-phi)))

        # Multiply the reflection coefficients with the
        # corresponding reflection reduction factor
        reduction_factor_0 = tf.sqrt(1 - scattering_coefficient_0**2)
        reduction_factor_0 = tf.complex(reduction_factor_0,
                                        tf.zeros_like(reduction_factor_0))
        reduction_factor_n = tf.sqrt(1 - scattering_coefficient_n**2)
        reduction_factor_n = tf.complex(reduction_factor_n,
                                        tf.zeros_like(reduction_factor_n))
        r_s_0 *= reduction_factor_0
        r_p_0 *= reduction_factor_0
        r_s_n *= reduction_factor_n
        r_p_n *= reduction_factor_n

        # Compute matrices R_0, R_n
        # [num_tx, num_samples, 2, 2]
        w_i_0  = component_transform(phi_prime_hat,
                                     beta_0_prime_hat,
                                     e_i_s_0,
                                     e_i_p_0)
        w_i_0 = tf.complex(w_i_0, tf.zeros_like(w_i_0))
        # [num_tx, num_samples, 2, 2]
        w_r_0 = component_transform(e_r_s_0,
                                    e_r_p_0,
                                    phi_hat_,
                                    beta_0_hat)
        w_r_0 = tf.complex(w_r_0, tf.zeros_like(w_r_0))
        # [num_tx, num_samples, 2, 2]
        r_0 = tf.expand_dims(tf.stack([r_s_0, r_p_0], -1), -1) * w_i_0
        # [num_tx, num_samples, 2, 2]
        r_0 = -tf.matmul(w_r_0, r_0)

        # [num_tx, num_samples, 2, 2]
        w_i_n = component_transform(phi_prime_hat,
                                    beta_0_prime_hat,
                                    e_i_s_n,
                                    e_i_p_n)
        w_i_n = tf.complex(w_i_n, tf.zeros_like(w_i_n))
        # [num_tx, num_samples, 2, 2]
        w_r_n = component_transform(e_r_s_n,
                                    e_r_p_n,
                                    phi_hat_,
                                    beta_0_hat)
        w_r_n = tf.complex(w_r_n, tf.zeros_like(w_r_n))
        # [num_tx, num_samples, 2, 2]
        r_n = tf.expand_dims(tf.stack([r_s_n, r_p_n], -1), -1) * w_i_n
        # [num_tx, num_samples, 2, 2]
        r_n = -tf.matmul(w_r_n, r_n)

        # Compute D_1, D_2, D_3, D_4
        # [num_tx, num_samples]
        phi_m = phi - phi_prime
        phi_p = phi + phi_prime

        # [num_tx, num_samples]
        cot_1 = cot((PI + phi_m)/(2*n))
        cot_2 = cot((PI - phi_m)/(2*n))
        cot_3 = cot((PI + phi_p)/(2*n))
        cot_4 = cot((PI - phi_p)/(2*n))

        def n_p(beta, n):
            return tf.math.round((beta + PI)/(2.*n*PI))

        def n_m(beta, n):
            return tf.math.round((beta - PI)/(2.*n*PI))

        def a_p(beta, n):
            return 2*tf.cos((2.*n*PI*n_p(beta, n)-beta)/2.)**2

        def a_m(beta, n):
            return 2*tf.cos((2.*n*PI*n_m(beta, n)-beta)/2.)**2

        # [1, num_samples]
        d_mul = - tf.cast(tf.exp(-1j*PI/4.), self._dtype)/\
            tf.cast((2*n)*tf.sqrt(2*PI*k), self._dtype)

        # [num_tx, num_samples]
        ell = s_prime*s/(s_prime + s)

        # [num_tx, num_samples]
        cot_1 = tf.complex(cot_1, tf.zeros_like(cot_1))
        cot_2 = tf.complex(cot_2, tf.zeros_like(cot_2))
        cot_3 = tf.complex(cot_3, tf.zeros_like(cot_3))
        cot_4 = tf.complex(cot_4, tf.zeros_like(cot_4))
        d_1 = d_mul*cot_1*f(k*ell*a_p(phi_m, n))
        d_2 = d_mul*cot_2*f(k*ell*a_m(phi_m, n))
        d_3 = d_mul*cot_3*f(k*ell*a_p(phi_p, n))
        d_4 = d_mul*cot_4*f(k*ell*a_m(phi_p, n))

        # [num_tx, num_samples, 1, 1]
        d_1 = tf.reshape(d_1, tf.concat([tf.shape(d_1), [1, 1]], axis=0))
        d_2 = tf.reshape(d_2, tf.concat([tf.shape(d_2), [1, 1]], axis=0))
        d_3 = tf.reshape(d_3, tf.concat([tf.shape(d_3), [1, 1]], axis=0))
        d_4 = tf.reshape(d_4, tf.concat([tf.shape(d_4), [1, 1]], axis=0))

        # [num_tx, num_samples]
        spreading_factor = tf.sqrt(1.0 / (s*s_prime*(s_prime + s)))
        spreading_factor = tf.complex(spreading_factor,
                                      tf.zeros_like(spreading_factor))
        # [num_tx, num_samples, 1, 1]
        spreading_factor = tf.reshape(spreading_factor, tf.shape(d_1))

        # [num_tx, num_samples, 2, 2]
        mat_t = (d_1+d_2)*tf.eye(2,2, batch_shape=tf.shape(r_0)[:2],
                                 dtype=self._dtype)
        # [num_tx, num_samples, 2, 2]
        mat_t += d_3*r_n + d_4*r_0
        # [num_tx, num_samples, 2, 2]
        mat_t *= -spreading_factor

        # Convert from/to GCS
        # [num_tx, num_samples]
        theta_t, phi_t = theta_phi_from_unit_vec(s_prime_hat)
        theta_r, phi_r = theta_phi_from_unit_vec(-s_hat)

        # [num_tx, num_samples, 2, 2]
        mat_from_gcs = component_transform(
                            theta_hat(theta_t, phi_t), phi_hat(phi_t),
                            phi_prime_hat, beta_0_prime_hat)
        mat_from_gcs = tf.complex(mat_from_gcs,
                                  tf.zeros_like(mat_from_gcs))

        # [num_tx, num_samples, 2, 2]
        mat_to_gcs = component_transform(phi_hat_, beta_0_hat,
                                         theta_hat(theta_r, phi_r),
                                         phi_hat(phi_r))
        mat_to_gcs = tf.complex(mat_to_gcs,
                                tf.zeros_like(mat_to_gcs))

        # [num_tx, num_samples, 2, 2]
        mat_t = tf.linalg.matmul(mat_t, mat_from_gcs)
        mat_t = tf.linalg.matmul(mat_to_gcs, mat_t)

        # Set invalid paths to 0
        # Expand masks to broadcast with the field components
        # [num_tx, num_samples, 1, 1]
        mask_ = expand_to_rank(diff_mask, 4, axis=-1)
        # Zeroing coefficients corresponding to non-valid paths
        # [num_tx, num_samples, 2, 2]
        mat_t = tf.where(mask_, mat_t, tf.zeros_like(mat_t))

        # Compute transmitters antenna pattern in the GCS
        # [num_tx, 3, 3]
        tx_rot_mat = rotation_matrix(sources_orientations)
        # [num_tx, 1, 3, 3]
        tx_rot_mat = tf.expand_dims(tx_rot_mat, axis=1)
        # tx_field : [num_tx, num_samples, num_tx_patterns, 2]
        # tx_es, ex_ep : [num_tx, num_samples, 3]
        tx_field, _, _ = self._compute_antenna_patterns(tx_rot_mat,
                            self._scene.tx_array.antenna.patterns, s_prime_hat)

        # Compute receiver antenna pattern in the GCS
        # [3, 3]
        rx_rot_mat = rotation_matrix(rx_orientation)
        # tx_field : [num_tx, num_samples, num_rx_patterns, 2]
        # tx_es, ex_ep : [num_tx, num_samples, 3]
        rx_field, _, _ = self._compute_antenna_patterns(rx_rot_mat,
                            self._scene.rx_array.antenna.patterns, -s_hat)

        # Compute the channel coefficients for every transmitter-receiver
        # pattern pairs
        # [num_tx, num_samples, 1, 1, 2, 2]
        mat_t = insert_dims(mat_t, 2, 2)
        # [num_tx, num_samples, 1, num_tx_patterns, 1, 2]
        tx_field = tf.expand_dims(tf.expand_dims(tx_field, axis=2), axis=4)
        # [num_tx, num_samples, num_rx_patterns, 1, 2]
        rx_field = tf.expand_dims(rx_field, axis=3)
        # [num_tx, num_samples, 1, num_tx_patterns, 2]
        a = tf.reduce_sum(mat_t*tx_field, axis=-1)
        # [num_tx, num_samples, num_rx_patterns, num_tx_patterns]
        a = tf.reduce_sum(tf.math.conj(rx_field)*a, axis=-1)

        # Apply synthetic array
        # [num_tx, num_samples, num_rx_antenna, num_tx_antenna]
        a = self._apply_synthetic_array(tx_rot_mat, rx_rot_mat, -s_hat,
                                        s_prime_hat, a)

        # Apply spatial precoding and combining
        # Precoding and combing
        # [1, 1, num_rx_ant]
        combining_vec = insert_dims(combining_vec, 2, 0)
        # [num_tx/1, 1, 1, num_tx_ant]
        precoding_vec = insert_dims(precoding_vec, 2, 1)
        # [num_tx, samples_per_tx, num_rx_ant]
        a = tf.reduce_sum(a*precoding_vec, axis=-1)
        # [num_tx, samples_per_tx]
        a = tf.reduce_sum(tf.math.conj(combining_vec)*a, axis=-1)

        # [num_tx, samples_per_tx]
        a = tf.square(tf.abs(a))

        # [num_tx, samples_per_tx]
        cst = tf.square(self._scene.wavelength/(4.*PI))
        a = a*cst

        return a

    def _build_diff_coverage_map(self, cm_center, cm_orientation, cm_size,
                                 cm_cell_size, diff_wedges_ind, diff_hit_points,
                                 diff_samples_power, diff_samples_weights,
                                 diff_num_samples_per_wedge):
        r"""
        Builds the coverage map for diffraction

        Input
        ------
        cm_center : [3], tf.float
            Center of the coverage map

        cm_orientation : [3], tf.float
            Orientation of the coverage map

        cm_size : [2], tf.float
            Scale of the coverage map.
            The width of the map (in the local X direction) is ``cm_size[0]``
            and its map (in the local Y direction) ``cm_size[1]``.

        cm_cell_size : [2], tf.float
            Resolution of the coverage map, i.e., width
            (in the local X direction) and height (in the local Y direction) in
            meters of a cell of the coverage map

        diff_wedges_ind : [num_samples], tf.int
            Indices of the wedges that interacted with the diffracted paths

        diff_hit_points : [num_tx, num_samples, 3], tf.float
            Positions of the intersection of the diffracted rays and coverage
            map

        diff_samples_power : [num_tx, num_samples], tf.float
            Powers of the samples of diffracted rays.

        diff_samples_weights : [num_tx, num_samples], tf.float
            Weights for averaging the field powers of the samples.

        diff_num_samples_per_wedge : [num_samples], tf.int
            For each sample, total mumber of samples that were sampled on the
            same wedge

        Output
        ------
        :cm : :class:`~sionna.rt.CoverageMap`
            The coverage maps
        """
        num_tx = diff_hit_points.shape[0]
        num_samples = diff_hit_points.shape[1]
        cell_area = cm_cell_size[0]*cm_cell_size[1]

        # [num_tx, num_samples]
        diff_wedges_ind = tf.tile(tf.expand_dims(diff_wedges_ind, axis=0),
                                  [num_tx, 1])

        # Transformation matrix required for computing the cell
        # indices of the intersection points
        # [3,3]
        rot_cm_2_gcs = rotation_matrix(cm_orientation)
        # [3,3]
        rot_gcs_2_cm = tf.transpose(rot_cm_2_gcs)

        # Initializing the coverage map
        num_cells_x = tf.cast(tf.math.ceil(cm_size[0]/cm_cell_size[0]),
                              tf.int32)
        num_cells_y = tf.cast(tf.math.ceil(cm_size[1]/cm_cell_size[1]),
                              tf.int32)
        num_cells = tf.stack([num_cells_x, num_cells_y], axis=-1)
        # [num_tx, num_cells_y+1, num_cells_x+1]
        # Add dummy row and columns to store the items that are out of the
        # coverage map
        cm = tf.zeros([num_tx, num_cells_y+1, num_cells_x+1],
                      dtype=self._rdtype)

        # Coverage map cells' indices
        # [num_tx, num_samples, 2 : xy]
        cell_ind = self._mp_hit_point_2_cell_ind(rot_gcs_2_cm, cm_center,
                            cm_size, cm_cell_size, num_cells, diff_hit_points)
        # Add the transmitter index to the coverage map
        # [num_tx]
        tx_ind = tf.range(num_tx, dtype=tf.int32)
        # [num_tx, 1, 1]
        tx_ind = expand_to_rank(tx_ind, 3)
        # [num_tx, num_samples, 1]
        tx_ind = tf.tile(tx_ind, [1, num_samples, 1])
        # [num_tx, num_samples, 3]
        cm_ind = tf.concat([tx_ind, cell_ind], axis=-1)

        # Wedges lengths
        # [num_tx, num_samples]
        lengths = tf.gather(self._wedges_length, diff_wedges_ind)

        # Wedges opening angles
        # [num_tx, num_samples, 2, 3]
        normals = tf.gather(self._wedges_normals, diff_wedges_ind)
        # [num_tx, num_samples]
        op_angles = PI + tf.math.acos(dot(normals[...,0,:],normals[...,1,:]))

        # Update the weights of each ray power
        # [1, num_samples]
        diff_num_samples_per_wedge = tf.expand_dims(diff_num_samples_per_wedge,
                                                    axis=0)
        diff_num_samples_per_wedge = tf.cast(diff_num_samples_per_wedge,
                                             self._rdtype)
        # [num_tx, num_samples]
        diff_samples_weights = tf.math.divide_no_nan(diff_samples_weights,
                                                     diff_num_samples_per_wedge)
        diff_samples_weights = diff_samples_weights*lengths*op_angles

        # Add the weighted powers to the coverage map
        # [num_tx, num_samples]
        weighted_sample_power = diff_samples_power*diff_samples_weights
        # [num_tx, num_cells_y+1, num_cells_x+1]
        cm = tf.tensor_scatter_nd_add(cm, cm_ind, weighted_sample_power)

        # Dump the dummy line and row
        # [num_tx, num_cells_y, num_cells_x]
        cm = cm[:,:num_cells_y,:num_cells_x]

        # Scaling by area of a cell
        # [num_tx, num_cells_y, num_cells_x]
        cm = cm / cell_area

        return cm

    def _diff_samples_2_coverage_map(self, los_primitives, edge_diffraction,
                                     num_samples, sources_positions, meas_plane,
                                     cm_center, cm_orientation, cm_size,
                                     cm_cell_size, sources_orientations,
                                     rx_orientation, combining_vec,
                                     precoding_vec, etas,
                                     scattering_coefficient):
        r"""
        Computes the coverage map for diffraction.

        Input
        ------
        los_primitives: [num_los_primitives], int
            Primitives in LoS.

        edge_diffraction : bool
            If set to `False`, only diffraction on wedges, i.e., edges that
            connect two primitives, is considered.

        num_samples : int
            Number of rays initially shooted from the wedges.

        sources_positions : [num_tx, 3], tf.float
            Coordinates of the sources.

        meas_plane : mi.Shape
            Mitsuba rectangle defining the measurement plane

        cm_center : [3], tf.float
            Center of the coverage map

        cm_orientation : [3], tf.float
            Orientation of the coverage map

        cm_size : [2], tf.float
            Scale of the coverage map.
            The width of the map (in the local X direction) is ``cm_size[0]``
            and its map (in the local Y direction) ``cm_size[1]``.

        cm_cell_size : [2], tf.float
            Resolution of the coverage map, i.e., width
            (in the local X direction) and height (in the local Y direction) in
            meters of a cell of the coverage map

        sources_orientations : [num_tx, 3], tf.float
            Orientations of the sources.

        rx_orientation : [3], tf.float
            Orientation of the receiver.

        combining_vec : [num_rx_ant], tf.complex
            Combining vector.
            This is used to combine the signal from the receive antennas for
            an imaginary receiver located on the coverage map.

        precoding_vec : [num_tx or 1, num_tx_ant], tf.complex
            Precoding vectors of the transmitters

        etas : [num_shape], tf.complex
            Tensor containing the complex relative permittivities of all shapes

        scattering_coefficient : [num_shape], tf.float
            Tensor containing the scattering coefficients of all shapes

        Output
        -------
        :cm : :class:`~sionna.rt.CoverageMap`
            The coverage maps
        """

        # Build empty coverage map
        num_cells_x = tf.cast(tf.math.ceil(cm_size[0]/cm_cell_size[0]),
                              tf.int32)
        num_cells_y = tf.cast(tf.math.ceil(cm_size[1]/cm_cell_size[1]),
                              tf.int32)
        # [num_tx, num_cells_y, num_cells_x]
        cm_null = tf.zeros([sources_positions.shape[0], num_cells_y,
                            num_cells_x], dtype=self._rdtype)

        # Get the candidate wedges for diffraction
        # diff_wedges_ind : [num_candidate_wedges], int
        #     Candidate wedges indices
        diff_wedges_ind = self._wedges_from_primitives(los_primitives,
                                                        edge_diffraction)
        # Early stop if there are no wedges
        if diff_wedges_ind.shape[0] == 0:
            return cm_null

        # Discard wedges for which the tx is inside the wedge
        # diff_mask : [num_tx, num_candidate_wedges], bool
        #   Mask set to False if the wedge is invalid
        # wedges : [num_candidate_wedges], int
        #     Candidate wedges indices
        output = self._discard_obstructing_wedges(diff_wedges_ind,
                                                    sources_positions)
        diff_mask = output[0]
        diff_wedges_ind = output[1]
        # Early stop if there are no wedges
        if diff_wedges_ind.shape[0] == 0:
            return cm_null

        # Sample diffraction points on the wedges
        # diff_mask : [num_tx, num_candidate_wedges], bool
        #   Mask set to False if the wedge is invalid
        # diff_wedges_ind : [num_candidate_wedges], int
        #     Candidate wedges indices
        # diff_ells : [num_samples], float
        #   Positions of the diffraction points on the wedges.
        #   These positionsare given as an offset from the wedges origins.
        #   The size of this tensor is in general slighly smaller than
        #   `num_samples` because of roundings.
        # diff_vertex : [num_samples, 3], tf.float
        #   Positions of the diffracted points in the GCS
        # diff_num_samples_per_wedge : [num_samples], tf.int
        #         For each sample, total mumber of samples that were sampled
        #         on the same wedge
        output = self._sample_wedge_points(diff_mask, diff_wedges_ind,
                                            num_samples)
        diff_mask = output[0]
        diff_wedges_ind = output[1]
        diff_ells = output[2]
        diff_vertex = output[3]
        diff_num_samples_per_wedge = output[4]

        # Test for blockage between the transmitters and diffraction points.
        # Discarted blocked samples.
        # diff_mask : [num_tx, num_candidate_wedges], bool
        #   Mask set to False if the wedge is invalid
        # diff_wedges_ind : [num_samples], int
        #     Candidate wedges indices
        # diff_ells : [num_samples], float
        #   Positions of the diffraction points on the wedges.
        #   These positionsare given as an offset from the wedges origins.
        #   The size of this tensor is in general slighly smaller than
        #   `num_samples` because of roundings.
        # diff_vertex : [num_samples, 3], float
        #   Positions of the diffracted points in the GCS
        # diff_num_samples_per_wedge : [num_samples], tf.int
        #         For each sample, total mumber of samples that were sampled
        #         on the same wedge
        output = self._test_tx_visibility(diff_mask, diff_wedges_ind,
                                            diff_ells,
                                            diff_vertex,
                                            diff_num_samples_per_wedge,
                                            sources_positions)
        diff_mask = output[0]
        diff_wedges_ind = output[1]
        diff_ells = output[2]
        diff_vertex = output[3]
        diff_num_samples_per_wedge = output[4]
        # Early stop if there are no wedges
        if diff_wedges_ind.shape[0] == 0:
            return cm_null

        # Samples angles for departure on the diffraction cone
        # diff_phi : [num_samples, 3], tf.float
        #   Sampled angles on the diffraction cone used for shooting rays
        diff_phi = self._sample_diff_angles(diff_wedges_ind)

        # Shoot rays in the sampled directions and test for intersection
        # with the coverage map.
        # Discard rays that miss it.
        # diff_mask : [num_tx, num_samples], tf.bool
        #     Mask set to False for invalid samples
        # diff_wedges_ind : [num_samples], tf.int
        #     Indices of the wedges that interacted with the diffracted
        #     paths
        # diff_ells : [num_samples], tf.float
        #     Positions of the diffraction points on the wedges.
        #     These positions are given as an offset from the wedges
        #     origins.
        # diff_phi : [num_samples], tf.float
        #     Sampled angles of diffracted rays on the diffraction cone
        # diff_vertex : [num_tx, num_samples, 3], tf.float
        #     Positions of the diffracted points in the GCS
        # diff_num_samples_per_wedge : [num_samples], tf.int
        #         For each sample, total mumber of samples that were sampled
        #         on the same wedge
        # diff_hit_points : [num_tx, num_samples, 3], tf.float
        #     Positions of the intersection of the diffracted rays and
        #     coverage map
        # diff_cone_angle : [num_tx, num_samples], tf.float
        #     Angle between e_hat and the diffracted ray direction.
        #     Takes value in (0,pi).
        output = self._shoot_diffracted_rays(diff_mask, diff_wedges_ind,
                                             diff_ells,
                                             diff_vertex,
                                             diff_num_samples_per_wedge,
                                             diff_phi,
                                             sources_positions,
                                             meas_plane)
        diff_mask = output[0]
        diff_wedges_ind = output[1]
        diff_ells = output[2]
        diff_phi = output[3]
        diff_vertex = output[4]
        diff_num_samples_per_wedge = output[5]
        diff_hit_points = output[6]
        diff_cone_angle = output[7]

        # Computes the weights for averaging the field powers of the samples
        # to compute the Monte Carlo estimate of the integral of the
        # diffracted field power over the measurement plane.
        # These weights are required as the measurement plane is
        # parametrized by the angle on the diffraction cones (phi) and
        # position on the wedges (ell).
        #
        # diff_samples_weights : [num_tx, num_samples], tf.float
        #     Weights for averaging the field powers of the samples.
        output = self._compute_samples_weights(cm_center,
                                               cm_orientation,
                                               sources_positions,
                                               diff_wedges_ind,
                                               diff_ells,
                                               diff_phi,
                                               diff_cone_angle)
        diff_samples_weights = output

        # Computes the power of the diffracted paths.
        #
        # diff_samples_power : [num_tx, num_samples], tf.float
        #   Powers of the samples of diffracted rays.
        output = self._compute_diffracted_path_power(sources_positions,
                                                     sources_orientations,
                                                     rx_orientation,
                                                     combining_vec,
                                                     precoding_vec,
                                                     diff_mask,
                                                     diff_wedges_ind,
                                                     diff_vertex,
                                                     diff_hit_points,
                                                     etas,
                                                     scattering_coefficient)
        diff_samples_power = output

        # Builds the coverage map for the diffracted field
        cm_diff = self._build_diff_coverage_map(cm_center,
                                                cm_orientation,
                                                cm_size,
                                                cm_cell_size,
                                                diff_wedges_ind,
                                                diff_hit_points,
                                                diff_samples_power,
                                                diff_samples_weights,
                                                diff_num_samples_per_wedge)

        return cm_diff
