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
    normalize, rotation_matrix
from .solver_base import SolverBase
from .coverage_map import CoverageMap, coverage_map_rectangle_to_world


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
    map are recorded. A ray can intersect the coverage map multiple times.
    Initial rays direction are randomly sampled.

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

    solver : :class:`~sionna.rt.SolverBase` | None
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

    seed : int
        Seed for sampling initial rays direction

    Output
    -------
    :cm : :class:`~sionna.rt.CoverageMap`
        The coverage maps
    """

    def __call__(self, max_depth, rx_orientation,
                 cm_center, cm_orientation, cm_size, cm_cell_size,
                 combining_vec, precoding_vec, num_samples, seed):

        # EM properties of the materials
        # Returns: relative_permittivities, denoted by `etas`
        etas = self._build_scene_object_properties_tensors()

        # Shoot and bounce
        # hit : [max_depth+1, num_tx, samples_per_tx]
        # hit_point : [max_depth+1, num_tx, samples_per_tx, 3]
        # objects : [max_depth, num_tx, samples_per_tx]
        # int_point : [max_depth, num_tx, samples_per_tx, 3]
        rays = self._shoot_and_bounce(cm_center, cm_orientation, cm_size,
                                      max_depth,
                                      num_samples, seed)

        cm = self._rays_2_coverage_map(
                            cm_center, cm_orientation, cm_size, cm_cell_size,
                            rx_orientation,
                            combining_vec, precoding_vec,
                            *rays,
                            etas)
        return cm

    ##################################################################
    # Internal methods
    ##################################################################

    def _shoot_and_bounce(self,
                          cm_center, cm_orientation, cm_size,
                          max_depth,
                          num_samples, seed):
        r"""
        Runs shoot-and-bounce to build the coverage map.

        The number of samples returned correspond to rays that made at least one
        hit with the coverage map for at least one transmitter, i.e., rays that
        made no hit are discarded.

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

        max_depth : int
            Maximum number of reflections

        num_samples : int
            Number of rays initially shooted from the transmitters.
            This number is shared by all transmitters, i.e.,
            ``num_samples/num_tx`` are shooted for each transmitter.

        seed : int
            Seed for sampling initial rays direction

        Output
        ------
        hit : [max_depth+1, num_tx, samples_per_tx], tf.bool
            Flag indicating if the ray intersect with the coverage map after
            the ``d``th reflection, where ``0 <= d <= max_depth`` (first
            dimension). ``d = 0`` corresponds to a LoS from the transmitter to
            the coverage map.
            A ray can intersect the coverage map multiple times.

        hit_point : [max_depth+1, num_tx, samples_per_tx, 3], tf.float
            Intersection point between theray and the coverage map after
            the ``d``th reflection, where ``0 <= d <= max_depth`` (first
            dimension). ``d = 0`` corresponds to a LoS from the transmitter to
            the coverage map.
            A ray can intersect the coverage map multiple times.

        objects : [max_depth, num_tx, samples_per_tx], tf.int
            Indices of the objects from the scene intersected by the rays at
            evey reflection.

        int_point : [max_depth, num_tx, samples_per_tx, 3], tf.float
            Intersection point between the rays and the coverage map.

        normals : [max_depth, num_tx, samples_per_tx, 3], tf.float
            Normals at the intersection with the objects from the scene.
        """
        mask_t = dr.mask_t(self._mi_scalar_t)

        # Transmitters positions
        # [num_tx, 3]
        tx_position = []
        for tx in self._scene.transmitters.values():
            tx_position.append(tx.position)
        tx_position = tf.stack(tx_position, axis=0)

        # Ensure that sample count can be distributed over the emitters
        num_tx = tx_position.shape[0]
        samples_per_tx = int(dr.ceil(num_samples / num_tx))
        num_samples = num_tx * samples_per_tx

        # Samples for sampling random directions for shooting the rays from the
        # transmitters.
        sampler = mi.load_dict({'type': 'independent'})
        sampler.seed(seed, num_samples)

        # Keep track of which rays are still active after every intersection,
        # i.e., didn't bounce out of the scene ("to the sky")
        # [num_tx, num_samples]
        active = dr.full(mask_t, True, num_samples)
        # Flag indicating if the ray hits the coverage map after the dth bounce,
        # where d = 0 means in LoS
        # [max_depth+1, num_tx, samples_per_tx]
        hit = []
        # Intersection point with the coverage map. Only valid if hit is True
        # for this item
        # [max_depth+1, num_tx, samples_per_tx, 3]
        hit_point = []
        # Index of the intersected primitive dth bounce. -1 Means no
        # intersection, i.e., the ray becomes invalid
        # [max_depth, num_tx, samples_per_tx]
        primitives = []
        # Intersection point with the scene. Only valid if object is not -1.
        # [max_depth, num_tx, samples_per_tx, 3]
        int_point = []

        # Rectangle defining the coverage map
        mi_rect = mi.load_dict({
            'type': 'rectangle',
            'to_world': coverage_map_rectangle_to_world(cm_center, cm_orientation, cm_size),
        })

        # Initial ray: direction sampled at random, origin placed on the
        # given transmitters
        tx_i = dr.linspace(self._mi_scalar_t, 0, num_tx, num=num_samples,
                            endpoint=False)
        tx_i = mi.Int32(tx_i)
        sampled_d = mi.warp.square_to_uniform_sphere(sampler.next_2d())
        transmitters_dr = self._mi_tensor_t(tx_position)
        ray = mi.Ray3f(
            o=dr.gather(self._mi_vec_t, transmitters_dr.array, tx_i),
            d=sampled_d,
        )

        for depth in range(max_depth+1):

            # Intersect with the coverage map
            si_rect = mi_rect.ray_intersect(ray, active=active)

            # Intersect with scene
            si_scene = self._mi_scene.ray_intersect(ray, active=active)

            # Hit the coverage map?
            # An intersection with the coverage map is only valid if it was
            # not obstructed
            # [num_samples]
            hit_rect = (si_rect.t < si_scene.t) & (si_rect.t < dr.inf)
            # [depth+1, num_tx, samples_per_tx]
            hit_rect = self._mi_to_tf_tensor(hit_rect, dtype=tf.bool)
            hit_rect = tf.reshape(hit_rect, [num_tx, samples_per_tx])
            hit.append(hit_rect)
            # Position of intersection with the coverage map
            # [num_samples, 3]
            hit_pos_rect = ray.o + si_rect.t*ray.d
            # [num_samples, 3]
            hit_pos_rect = self._mi_to_tf_tensor(hit_pos_rect,
                                                dtype=self._rdtype)
            # [num_tx, samples_per_tx, 3]
            hit_pos_rect = tf.reshape(hit_pos_rect,
                                    [num_tx, samples_per_tx, 3])
            # [depth+1, num_tx, samples_per_tx, 3]
            hit_point.append(hit_pos_rect)

            # Stop if reached max depth
            if depth == max_depth:
                break

            # A ray if sill active if it intersected with the scene
            # [num_samples]
            active &= si_scene.is_valid()

            # Stop tracing if no ray is active.
            if not dr.any(active):
                break

            # Record which primitives were hit
            shape_i = dr.gather(mi.Int32, self._shape_indices,
                                dr.reinterpret_array_v(mi.UInt32,
                                                    si_scene.shape),
                                active)
            offsets = dr.gather(mi.Int32, self._prim_offsets, shape_i,
                                active)
            prims_i = dr.select(active, offsets + si_scene.prim_index, -1)
            prims_i = self._mi_to_tf_tensor(prims_i, tf.int32)
            prims_i = tf.reshape(prims_i, [num_tx, samples_per_tx])
            primitives.append(prims_i)
            # Position of intersection with the scene
            # [num_samples, 3]
            hit_pos_scene = ray.o + si_scene.t*ray.d
            # [num_samples, 3]
            hit_pos_scene = self._mi_to_tf_tensor(hit_pos_scene,
                                                dtype=self._rdtype)
            # [num_tx, samples_per_tx, 3]
            hit_pos_scene = tf.reshape(hit_pos_scene,
                                    [num_tx, samples_per_tx, 3])
            int_point.append(hit_pos_scene)

            # Prepare the next interaction, assuming purely specular
            # reflection
            ray = si_scene.spawn_ray(si_scene.to_world(
                mi.reflect(si_scene.wi)))

        if (max_depth == 0) or (len(primitives) == 0):
            # If only LoS is requested or if no interaction was found
            # (empty scene), then the only candidate is the LoS
            primitives = tf.fill([0, num_tx, samples_per_tx], -1)
            int_point = tf.fill([0, num_tx, samples_per_tx, 3],
                                tf.zeros((),self._rdtype))
            # Set the max_depth to 0
            max_depth = 0
        else:
            # Stack all found interactions along the depth dimension
            # [max_depth, num_tx, samples_per_tx]
            primitives = tf.stack(primitives, axis=0)
            # [max_depth, num_tx, samples_per_tx, 3]
            int_point = tf.stack(int_point, axis=0)
            # Max depth is updated to the highest number of reflections that
            # was found for a path
            max_depth = primitives.shape[0]

        if len(hit) == 0:
            hit = tf.fill([0, num_tx, samples_per_tx], -1)
            hit_point = tf.fill([0, num_tx, samples_per_tx, 3], 0.0)
        else:
            # [max_depth+1, num_tx, samples_per_tx]
            hit = tf.stack(hit, axis=0)
            # [max_depth+1, num_tx, samples_per_tx, 3]
            hit_point = tf.stack(hit_point, axis=0)

        # Map primitives to their corresponding object.
        # Add a dummy entry to primitives_2_objects with value -1 for invalid
        # reflection.
        # Invalid reflection, i.e., corresponding to paths with a depth lower
        # than max_depth, will be assigned -1 as index of the intersected
        # shape.
        # [num_primitives + 1]
        primitives_2_objects = tf.pad(self._primitives_2_objects, [[0,1]],
                                        constant_values=-1)
        # [num_primitives + 1, 3]
        normals = tf.pad(self._normals, [[0,1], [0,0]],
                         constant_values=tf.ones((), self._rdtype))
        # Replace all -1 by num_primitives
        num_primitives = self._primitives_2_objects.shape[0]
        # [max_depth, num_tx, samples_per_tx]
        primitives = tf.where(tf.equal(primitives,-1),
                                    num_primitives,
                                    primitives)
        # [max_depth, num_tx, samples_per_tx]
        objects = tf.gather(primitives_2_objects, primitives)

        # Extract the normals of the intersected primitives
        # [max_depth, num_tx, samples_per_tx, 3]
        normals = tf.gather(normals, primitives)

        # Remove rays that never hit the coverage map
        # [samples_per_tx]
        hit_once = tf.reduce_any(hit, axis=(0,1))
        # [num_samples_hit]
        hit_once = tf.where(hit_once)[:,0]
        # [max_depth+1, num_tx, num_samples_hit]
        hit = tf.gather(hit, hit_once, axis=2)
        # [max_depth+1, num_tx, num_samples_hit, 3]
        hit_point = tf.gather(hit_point, hit_once, axis=2)
        # [max_depth, num_tx, num_samples_hit]
        objects = tf.gather(objects, hit_once, axis=2)
        # [max_depth, num_tx, num_samples_hit, 3]
        int_point = tf.gather(int_point, hit_once, axis=2)
        # [max_depth, num_tx, num_samples_hit, 3]
        normals = tf.gather(normals, hit_once, axis=2)

        # Ensure output values are numerically valid
        # [max_depth+1, num_tx, num_samples_hit, 1]
        hit_ = tf.expand_dims(hit, axis=-1)
        # [max_depth+1, num_tx, num_samples_hit, 3]
        hit_point = tf.where(hit_, hit_point, tf.ones_like(hit_point))
        # [max_depth, num_tx, num_samples_hit, 1]
        valid_inter = tf.expand_dims(tf.not_equal(objects, -1), axis=-1)
        # [max_depth, num_tx, num_samples_hit, 3]
        int_point = tf.where(valid_inter, int_point, tf.ones_like(int_point))
        # [max_depth, num_tx, num_samples_hit, 3]
        normals = tf.where(valid_inter, normals, tf.ones_like(normals))

        return hit, hit_point, objects, int_point, normals

    def _ray_contribution(self, mat_t,
                          k_t, theta_t, phi_t,
                          k_r, theta_r, phi_r,
                          rx_orientation,
                          combining_vec, precoding_vec):
        """
        Compute the individual contribution of rays to the coverage map.

        This function applies the phase shit to simulate a synthetic array and
        the antenna patterns.

        Input
        ------
        mat_t : [num_tx, samples_per_tx, 2, 2], tf.complex
            Transfer matrices

        k_t : [num_tx, samples_per_tx, 3], tf.float
            Normalized wave transmit vector in the global coordinate system

        theta_t : [num_tx, samples_per_tx], tf.float
            Zenith of departure in radian

        phi_t : [num_tx, samples_per_tx], tf.float
            Azimuth of departure in radian

        k_r : [num_tx, samples_per_tx, 3], tf.float
            Normalized wave receive vector in the global coordinate system

        theta_r : [num_tx, samples_per_tx], tf.float
            Zenith of arrival in radian

        phi_r : [num_tx, samples_per_tx], tf.float
            Azimuth of arrival in radian

        rx_orientation : [3], tf.float
            Orientation of the receiver.
            This is used to compute the antenna response and antenna pattern
            for an imaginary receiver located on the coverage map.

        combining_vec : [num_rx_ant], tf.complex
            Vector used for receive-combing

        precoding_vec : [num_tx, num_tx_ant] or [1, num_tx_ant], tf.complex
            Vector used for transmit-precoding

        Output
        -------
        a : [num_tx, samples_per_tx], tf.float
            Amplitudes of the contribution of the individual rays
        """

        # Apply multiplication by wavelength/4pi
        # [num_tx, samples_per_tx, 2, 2]
        cst = tf.cast(self._scene.wavelength/(4.*PI), self._dtype)
        h = cst*mat_t

        two_pi = tf.cast(2.*PI, self._rdtype)

        # Transmitters positions and orientations
        # tx_position : [num_tx, 3]
        # tx_orientation : [num_tx, 3]
        tx_position = []
        tx_orientation = []
        for tx in self._scene.transmitters.values():
            tx_position.append(tx.position)
            tx_orientation.append(tx.orientation)
        tx_position = tf.stack(tx_position, axis=0)
        tx_orientation = tf.stack(tx_orientation, axis=0)

        ######################################################
        # Synthetic array are assumed for coverage map
        # computation. Applies the phase shift due to the
        # antenna array
        ######################################################

        # Rotated position of the TX and RX antenna elements
        # [num_tx, tx_array_size, 3]
        tx_rel_ant_pos = [self._scene.tx_array.rotated_positions(tx.orientation)
                            for tx in self._scene.transmitters.values()]
        tx_rel_ant_pos = tf.stack(tx_rel_ant_pos, axis=0)
        # [1, rx_array_size, 3]
        rx_rel_ant_pos = self._scene.rx_array.rotated_positions(rx_orientation)
        rx_rel_ant_pos = tf.expand_dims(rx_rel_ant_pos, 0)

        # Expand dims for broadcasting with antennas
        # [num_tx, 1, 1, samples_per_tx, 3]
        k_r_ = insert_dims(k_r, 2, 1)
        k_t_ = insert_dims(k_t, 2, 1)
        # Compute the synthetic phase shifts due to the antenna array
        # Transmitter side
        # Expand for broadcasting with receive antennas and samples
        # [num_tx, 1, tx_array_size, 1, 3]
        tx_rel_ant_pos = tf.expand_dims(tx_rel_ant_pos, axis=1)
        tx_rel_ant_pos = tf.expand_dims(tx_rel_ant_pos, axis=3)
        # [num_tx, 1, tx_array_size, samples_per_tx]
        tx_phase_shifts = dot(tx_rel_ant_pos, k_t_)
        # Receiver side
        # Expand for broadcasting with transmit antennas and samples
        # [1, rx_array_size, 1, 1, 3]
        rx_rel_ant_pos = insert_dims(rx_rel_ant_pos, 2, 2)
        # [num_tx, rx_array_size, 1, samples_per_tx]
        rx_phase_shifts = dot(rx_rel_ant_pos, k_r_)
        # Total phase shift
        # [num_tx, rx_array_size, tx_array_size, samples_per_tx]
        phase_shifts = rx_phase_shifts + tx_phase_shifts
        phase_shifts = two_pi*phase_shifts/self._scene.wavelength
        # Apply the phase shifts
        # Expand field for broadcasting with antennas
        # [num_tx, 1, 1, samples_per_tx, 2, 2]
        h = insert_dims(h, 2, 1)
        # Expand phase shifts for broadcasting with transfer matrix
        # [num_tx, rx_array_size, tx_array_size, samples_per_tx, 1, 1]
        phase_shifts = expand_to_rank(phase_shifts, tf.rank(h))
        # [num_tx, rx_array_size, tx_array_size, samples_per_tx, 2, 2]
        h = h*tf.exp(tf.complex(tf.zeros_like(phase_shifts), phase_shifts))

        ######################################################
        # Compute and apply antenna patterns
        ######################################################

        # Rotation matrices for transmitters
        # [num_tx, 3, 3]
        tx_rot_mat = rotation_matrix(tx_orientation)
        # Rotation matrices for receivers
        # [3, 3]
        rx_rot_mat = rotation_matrix(rx_orientation)

        # Normalized wave transmit vector in the local coordinate system of
        # the transmitters
        # [num_tx, 1, 3, 3]
        tx_rot_mat = tf.expand_dims(tx_rot_mat, axis=1)
        # [num_tx, samples_per_tx, 3]
        k_prime_t = tf.linalg.matvec(tx_rot_mat, k_t, transpose_a=True)

        # Normalized wave receiver vector in the local coordinate system of
        # the receivers
        # [1, 1, 3, 3]
        rx_rot_mat = insert_dims(rx_rot_mat, 2, 0)
        # [num_tx, samples_per_tx, 3]
        k_prime_r = tf.linalg.matvec(rx_rot_mat, k_r, transpose_a=True)

        # Angles of departure in the local coordinate system of the
        # transmitters
        # [num_tx, samples_per_tx]
        theta_prime_t, phi_prime_t = theta_phi_from_unit_vec(k_prime_t)

        # Angles of arrival in the local coordinate system of the
        # receivers
        # [num_tx, samples_per_tx]
        theta_prime_r, phi_prime_r = theta_phi_from_unit_vec(k_prime_r)

        # Spherical global frame vectors for tx and rx
        # [num_tx, samples_per_tx, 3]
        theta_hat_t = theta_hat(theta_t, phi_t)
        # [num_tx, samples_per_tx, 3]
        phi_hat_t = phi_hat(phi_t)
        # [num_tx, samples_per_tx, 3]
        theta_hat_r = theta_hat(theta_r, phi_r)
        # [num_tx, samples_per_tx, 3]
        phi_hat_r = phi_hat(phi_r)

        # Spherical local frame vectors for tx and rx
        # [num_tx, samples_per_tx, 3]
        theta_hat_prime_t = theta_hat(theta_prime_t, phi_prime_t)
        # [num_tx, samples_per_tx, 3]
        phi_hat_prime_t = phi_hat(phi_prime_t)
        # [num_tx, samples_per_tx, 3]
        theta_hat_prime_r = theta_hat(theta_prime_r, phi_prime_r)
        # [num_tx, samples_per_tx, 3]
        phi_hat_prime_r = phi_hat(phi_prime_r)

        # Rotation matrix for going from the spherical LCS to the spherical GCS
        # For transmitters
        # [num_tx, samples_per_tx]
        tx_lcs2gcs_11 = dot(theta_hat_t,
                            tf.linalg.matvec(tx_rot_mat, theta_hat_prime_t))
        # [num_tx, samples_per_tx]
        tx_lcs2gcs_12 = dot(theta_hat_t,
                            tf.linalg.matvec(tx_rot_mat, phi_hat_prime_t))
        # [num_tx, samples_per_tx]
        tx_lcs2gcs_21 = dot(phi_hat_t,
                            tf.linalg.matvec(tx_rot_mat, theta_hat_prime_t))
        # [num_tx, samples_per_tx]
        tx_lcs2gcs_22 = dot(phi_hat_t,
                            tf.linalg.matvec(tx_rot_mat, phi_hat_prime_t))
        # [num_tx, samples_per_tx, 2, 2]
        tx_lcs2gcs = tf.stack(
                    [tf.stack([tx_lcs2gcs_11, tx_lcs2gcs_12], axis=-1),
                     tf.stack([tx_lcs2gcs_21, tx_lcs2gcs_22], axis=-1)],
                    axis=-2)
        tx_lcs2gcs = tf.complex(tx_lcs2gcs, tf.zeros_like(tx_lcs2gcs))
        # For receivers
        # [num_tx, samples_per_tx]
        rx_lcs2gcs_11 = dot(theta_hat_r,
                            tf.linalg.matvec(rx_rot_mat, theta_hat_prime_r))
        # [num_tx, samples_per_tx]
        rx_lcs2gcs_12 = dot(theta_hat_r,
                            tf.linalg.matvec(rx_rot_mat, phi_hat_prime_r))
        # [num_tx, samples_per_tx]
        rx_lcs2gcs_21 = dot(phi_hat_r,
                            tf.linalg.matvec(rx_rot_mat, theta_hat_prime_r))
        # [num_tx, samples_per_tx]
        rx_lcs2gcs_22 = dot(phi_hat_r,
                            tf.linalg.matvec(rx_rot_mat, phi_hat_prime_r))
        # [num_tx, samples_per_tx, 2, 2]
        rx_lcs2gcs = tf.stack(
                    [tf.stack([rx_lcs2gcs_11, rx_lcs2gcs_12], axis=-1),
                     tf.stack([rx_lcs2gcs_21, rx_lcs2gcs_22], axis=-1)],
                    axis=-2)
        rx_lcs2gcs = tf.complex(rx_lcs2gcs, tf.zeros_like(rx_lcs2gcs))

        # List of antenna patterns (callables)
        tx_patterns = self._scene.tx_array.antenna.patterns
        rx_patterns = self._scene.rx_array.antenna.patterns

        tx_ant_fields_hat = []
        for pattern in tx_patterns:
            # [num_tx, samples_per_tx, 2]
            tx_ant_f = tf.stack(pattern(theta_prime_t, phi_prime_t), axis=-1)
            tx_ant_fields_hat.append(tx_ant_f)

        rx_ant_fields_hat = []
        for pattern in rx_patterns:
            # [num_tx, samples_per_tx, 2]
            rx_ant_f = tf.stack(pattern(theta_prime_r, phi_prime_r), axis=-1)
            rx_ant_fields_hat.append(rx_ant_f)

        # Stacking the patterns, corresponding to different polarization
        # directions, as an additional dimension
        # [num_tx, num_rx_patterns, samples_per_tx, 2]
        rx_ant_fields_hat = tf.stack(rx_ant_fields_hat, axis=1)
        # Expand for broadcasting with tx polarization
        # [num_tx, num_rx_patterns, 1, samples_per_tx, 2]
        rx_ant_fields_hat = tf.expand_dims(rx_ant_fields_hat, axis=2)

        # Stacking the patterns, corresponding to different polarization
        # [num_tx, num_tx_patterns, samples_per_tx, 2]
        tx_ant_fields_hat = tf.stack(tx_ant_fields_hat, axis=1)
        # Expand for broadcasting with rx polarization
        # [num_tx, 1, num_tx_patterns, samples_per_tx, 2]
        tx_ant_fields_hat = tf.expand_dims(tx_ant_fields_hat, axis=1)

        # Antenna patterns to spherical global coordinate system
        # Expand to broadcast with antenna patterns
        # [num_tx, 1, 1, samples_per_tx, 2, 2]
        rx_lcs2gcs = insert_dims(rx_lcs2gcs, 2, 1)
        # [num_tx, num_rx_patterns, 1, samples_per_tx, 2]
        rx_ant_fields = tf.linalg.matvec(rx_lcs2gcs, rx_ant_fields_hat)
        # Expand to broadcast with antenna patterns
        tx_lcs2gcs = insert_dims(tx_lcs2gcs, 2, 1)
        # [num_tx, 1, num_tx_patterns, samples_per_tx, 2]
        tx_ant_fields = tf.linalg.matvec(tx_lcs2gcs, tx_ant_fields_hat)

        # Expand the transfer matrix to broadcast with the antenna patterns
        # [num_tx, rx_array_size, 1, tx_array_size, 1, samples_per_tx, 2, 2]
        h = tf.expand_dims(tf.expand_dims(h, 2), 4)

        # Expand the field to broadcast with the antenna array
        # [num_tx, 1, 1, 1, num_tx_patterns, samples_per_tx, 2]
        tx_ant_fields = insert_dims(tx_ant_fields, 2, 1)

        # Expand the field to broadcast with the antenna array
        # [num_tx, 1, num_rx_patterns, 1, 1, samples_per_tx, 2]
        rx_ant_fields = tf.expand_dims(tf.expand_dims(rx_ant_fields, 1), 3)

        # Manual broadcast to speed-up computation
        rx_array_size = tf.shape(h)[1]
        tx_array_size = tf.shape(h)[3]
        num_tx_patterns = tf.shape(tx_ant_fields)[4]
        # [num_tx, rx_array_size, 1, tx_array_size, num_tx_patterns,
        #   samples_per_tx, 2, 2]
        h = tf.tile(h, [1, 1, 1, 1, num_tx_patterns, 1, 1, 1])
        # [num_tx, rx_array_size, 1, tx_array_size, num_tx_patterns,
        # samples_per_tx, 2]
        tx_ant_fields = tf.tile(tx_ant_fields,
                                [1, rx_array_size, 1, tx_array_size, 1, 1, 1])
        # [num_tx, rx_array_size, num_rx_patterns, tx_array_size,
        #   num_tx_patterns, samples_per_tx]
        h = tf.linalg.matvec(h, tx_ant_fields)
        h = dot(rx_ant_fields, h)

        ######################################################
        # Spatial precoding and combining
        ######################################################

        # Flatten the antenna and pattern dimensions
        # [num_tx,
        # num_rx_ant = rx_array_size*num_rx_patterns,
        # num_tx_ant = tx_array_size*num_tx_patterns,
        # samples_per_tx]
        h = flatten_dims(flatten_dims(h, 2, 1), 2, 2)

        # Precoding and combing
        # [num_tx, samples_per_tx, num_rx_ant, num_tx_ant]
        h = tf.transpose(h, [0, 3, 1, 2])
        # [1, 1, num_rx_ant]
        combining_vec = insert_dims(combining_vec, 2, 0)
        # [num_tx or 1, 1, num_tx_ant, 1]
        precoding_vec = tf.expand_dims(precoding_vec, 1)
        # [num_tx, samples_per_tx, num_rx_ant]
        h = tf.linalg.matvec(h, precoding_vec)
        # [num_tx, samples_per_tx]
        h = dot(combining_vec, h)

        ######################################################
        # Only the squared amplitude is returned
        ######################################################

        # Compute and return the amplitudes
        # [num_tx, samples_per_tx]
        a = tf.square(tf.abs(h))
        return a

    def _rays_2_coverage_map(self,
                             cm_center, cm_orientation, cm_size, cm_cell_size,
                             rx_orientation,
                             combining_vec, precoding_vec,
                             hit, hit_point, objects, int_point, normals,
                             relative_permittivity):
        r"""
        Builds the coverage map from the rays.

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

        hit : [max_depth+1, num_tx, samples_per_tx], tf.bool
            Flag indicating if the ray intersect with the coverage map after
            the ``d``th reflection, where ``0 <= d <= max_depth`` (first
            dimension). ``d = 0`` corresponds to a LoS from the transmitter to
            the coverage map.
            A ray can intersect the coverage map multiple times.

        hit_point : [max_depth+1, num_tx, samples_per_tx, 3], tf.float
            Intersection point between theray and the coverage map after
            the ``d``th reflection, where ``0 <= d <= max_depth`` (first
            dimension). ``d = 0`` corresponds to a LoS from the transmitter to
            the coverage map.
            A ray can intersect the coverage map multiple times.

        objects : [max_depth, num_tx, samples_per_tx], tf.int
            Indices of the objects from the scene intersected by the rays at
            evey reflection.

        int_point : [max_depth, num_tx, samples_per_tx, 3], tf.float
            Intersection point between the rays and the coverage map.

        normals : [max_depth, num_tx, samples_per_tx, 3], tf.float
            Normals at the intersection with the objects from the scene.

        relative_permittivity : [num_shape], tf.complex
            Tensor containing the complex relative permittivity of all objects

        Output
        -------
        :cm : :class:`~sionna.rt.CoverageMap`
            The coverage maps
        """

        # Transmitters positions and orientations
        # [num_tx, 3]
        tx_position = []
        for tx in self._scene.transmitters.values():
            tx_position.append(tx.position)
        tx_position = tf.stack(tx_position, axis=0)

        # Maximum depth
        max_depth = objects.shape[0]
        # Number of transmitters
        num_tx = tx_position.shape[0]
        # Maximum number of paths
        samples_per_tx = hit.shape[2]

        # Offset for computing the cell indices of the coverage map
        scene_min = tf.cast(self._mi_scene.bbox().min, self._rdtype)
        # In case of empty scene, bbox min is -inf
        scene_min = tf.where(tf.math.is_inf(scene_min),
                             -tf.ones_like(scene_min), scene_min)

        # Flag that indicates if a ray is valid
        # [max_depth, num_tx, samples_per_tx]
        valid_ray = tf.not_equal(objects, -1)

        # Tensor with relative perimittivities values for all reflection points
        # On CPU, indexing with -1 does not work. Hence we replace -1 by 0.
        # This makes no difference on the resulting paths as such paths
        # are not flaged as active.
        # [max_depth, num_tx, samples_per_tx]
        object_idx = tf.where(objects == -1, 0, objects)
        # [max_depth, num_tx, samples_per_tx]
        if relative_permittivity.shape[0] == 0:
            etas = tf.zeros_like(object_idx, dtype=self._dtype)
        else:
            etas = tf.gather(relative_permittivity, object_idx)

        # Vertices updated with the transmitters and receivers
        # [num_tx, 1, 3]
        tx_position = tf.expand_dims(tx_position, axis=1)
        # [num_tx, samples_per_tx, 3]
        tx_position = tf.broadcast_to(tx_position, int_point.shape[1:])
        # [1, num_tx, samples_per_tx, 3]
        tx_position = tf.expand_dims(tx_position, axis=0)
        # [1 + max_depth, num_tx, samples_per_tx, 3]
        vertices = tf.concat([tx_position, int_point], axis=0)

        # Direction of arrivals (k_i)
        # k_i : [max_depth, num_tx, samples_per_tx, 3]
        # ray_lengths : [max_depth, num_tx, samples_per_tx]
        k_i = tf.roll(vertices, -1, axis=0) - vertices
        k_i,ray_lengths = normalize(k_i)
        k_i = k_i[:max_depth]
        ray_lengths = ray_lengths[:max_depth]

        # Compute distance travelled by every ray after `d` interactions with
        # the scene.
        # [max_depth, num_tx, samples_per_tx]
        distance = tf.cumsum(ray_lengths, axis=0)
        # The first entry in `distance` must correspond to 0, i.e., after 0
        # interaction.
        # [max_depth+1, num_tx, samples_per_tx]
        distance = tf.pad(distance, [[1,0], [0,0], [0,0]], "CONSTANT",
                          tf.zeros((), self._rdtype))

        # Direction of departures (k_r) at interaction points.
        # We do not need the direction of departure at the transmitter.
        # Therefore `k_r` only stores the directions of
        # departures at the `max_depth` interaction points.
        # [max_depth, num_tx, samples_per_tx, 3]
        k_r = tf.roll(vertices, -2, axis=0) - tf.roll(vertices, -1, axis=0)
        k_r,_ = normalize(k_r)
        k_r = k_r[:max_depth]

        # Compute cos(theta) at each reflection point
        # [max_depth, num_tx, samples_per_tx]
        cos_theta = -dot(k_i, normals)

        # Compute e_i_s, e_i_p, e_r_s, e_r_p at each reflection point
        # all : [max_depth, num_tx, samples_per_tx, 3]
        e_i_s, e_i_p, e_r_s, e_r_p = self._compute_field_unit_vectors(
            k_i, k_r, normals)

        # Compute r_s, r_p at each reflection point
        # [max_depth, num_tx, samples_per_tx]
        r_s, r_p = self._reflection_coefficient(etas, cos_theta)

        # Precompute the transformation matrix required for computing the cell
        # indices of the intersection points
        # [3,3]
        rot_cm_2_gcs = rotation_matrix(cm_orientation)
        # [3,3]
        inv_rot_mat = tf.transpose(rot_cm_2_gcs)
        # Expand for broadcasting
        # [1, 1, 3, 3]
        inv_transform = insert_dims(inv_rot_mat, 2, 0)
        # [1, 1, 3]
        cm_center = insert_dims(cm_center, 2, 0)

        # Compute the field transfer matrix.
        # It is initialized with the identity matrix of size 2 (S and P
        # polarization components)
        # [num_tx, samples_per_tx, 2, 2]
        mat_t = tf.eye(num_rows=2,
                       batch_shape=[num_tx, samples_per_tx],
                       dtype=self._dtype)

        # Initializing the coverage map
        num_cells_x = tf.cast(tf.math.ceil(cm_size[0]/cm_cell_size[0]),
                              tf.int32)
        num_cells_y = tf.cast(tf.math.ceil(cm_size[1]/cm_cell_size[1]),
                              tf.int32)
        # [num_tx, num_cells_y+1, num_cells_x+1]
        # Add dummy row and columns to store the items that are out of the
        # coverage map
        cm = tf.zeros([num_tx, num_cells_y+1, num_cells_x+1],
                      dtype=self._rdtype)

        if max_depth > 0:
            # [num_tx, samples_per_tx, 3]
            k_i_0 = k_i[0]
            # Compute angles of departures
            # theta_t, phi_t: [num_tx, samples_per_tx]
            theta_t, phi_t = theta_phi_from_unit_vec(k_i_0)
            # Initialize last field unit vector with outgoing ones
            # [num_tx, samples_per_tx, 3]
            last_e_r_s = theta_hat(theta_t, phi_t)
            last_e_r_p = phi_hat(phi_t)
        for depth in tf.range(0,max_depth+1):

            # After `depth` interactions with the scene, did the ray hit
            # the coverage map?
            # [num_tx, samples_per_tx]
            hit_ = hit[depth]

            # Vector representing the last section of the path
            # [num_tx, samples_per_tx]
            last_step = vertices[depth] - hit_point[depth]
            # last_step_norm : [num_tx, samples_per_tx, 3]
            # last_step_dist : [num_tx, samples_per_tx]
            last_step_norm,last_step_dist = normalize(last_step)
            # Angles of arrival at the coverage map
            # theta_t, phi_t: [num_tx, samples_per_tx]
            theta_r, phi_r = theta_phi_from_unit_vec(last_step_norm)

            # Distance from transmitter to coverage map
            # [num_tx, samples_per_tx]
            last_step_dist = distance[depth] + last_step_dist

            # Update transfer matrices with the propagation loss
            # [num_tx, samples_per_tx, 1, 1]
            last_step_dist = expand_to_rank(last_step_dist, tf.rank(mat_t))
            last_step_dist = tf.cast(last_step_dist, self._dtype)
            # [num_tx, samples_per_tx, 2, 2]
            mat_t_hit = tf.math.divide_no_nan(mat_t, last_step_dist)

            # Compute the rays' individual contribution
            # [num_tx, samples_per_tx]
            if depth == 0:
                # In LoS, the transmit wave vector is calculated from the
                # intersection point with the coverage map and not from the
                # first intersection with the scene (k_i[0])
                # [num_tx, samples_per_tx, 3]
                k_i_0 = -last_step_norm
                # theta_t, phi_t: [num_tx, samples_per_tx]
                theta_t_0, phi_t_0 = theta_phi_from_unit_vec(k_i_0)
            else:
                # [num_tx, samples_per_tx, 3]
                k_i_0 = k_i[0]
                # theta_t, phi_t: [num_tx, samples_per_tx]
                theta_t_0, phi_t_0 = theta_t, phi_t
            # [num_tx, samples_per_tx]
            ray_contrib = self._ray_contribution(mat_t_hit,
                                                 k_i_0, theta_t_0, phi_t_0,
                                                 last_step_norm, theta_r, phi_r,
                                                 rx_orientation,
                                                 combining_vec, precoding_vec)
            # Zeros the contribution of rays that didn't hit the coverage map
            # [num_tx, samples_per_tx]
            ray_contrib = tf.where(hit_, ray_contrib,
                                   tf.zeros_like(ray_contrib))

            # Coverage map cells' indices
            # Coordinates of the hit point in the coverage map local coordinate
            # system
            # [num_tx, samples_per_tx, 3]
            cm_hit_point = tf.linalg.matvec(inv_transform,
                                            hit_point[depth]-cm_center)
            # In the local coordinate system of the coverage map, z should be 0
            # as the coverage map is in XY
            # [num_tx, samples_per_tx]
            cell_x = cm_hit_point[...,0] + cm_size[0]*0.5
            cell_x = tf.cast(tf.math.floor(cell_x/cm_cell_size[0]), tf.int32)
            cell_x = tf.where(tf.less(cell_x, num_cells_x), cell_x, num_cells_x)
            # [num_tx, samples_per_tx]
            cell_y = cm_hit_point[...,1] + cm_size[1]*0.5
            cell_y = tf.cast(tf.math.floor(cell_y/cm_cell_size[1]), tf.int32)
            cell_y = tf.where(tf.less(cell_y, num_cells_y), cell_y, num_cells_y)
            # [num_tx, samples_per_tx, 2 : xy]
            cell_ind = tf.stack([cell_y, cell_x], axis=2)
            # Add the transmitter index to the coverage map
            # [num_tx]
            tx_ind = tf.range(num_tx, dtype=tf.int32)
            # [num_tx, 1, 1]
            tx_ind = expand_to_rank(tx_ind, 3)
            # [num_tx, samples_per_tx, 1]
            tx_ind = tf.tile(tx_ind, [1, samples_per_tx, 1])
            # [num_tx, samples_per_tx, 3]
            cell_ind = tf.concat([tx_ind, cell_ind], axis=-1)

            # Add the contribution to the coverage map
            # [num_tx, num_cells_y+1, num_cells_x+1]
            cm = tf.tensor_scatter_nd_add(cm, cell_ind, ray_contrib)

            # Break the loop if the maximum depth was reached
            if depth == max_depth:
                break

            # Is this a valid reflection?
            # [num_tx, samples_per_tx]
            valid = valid_ray[depth]

            # Early stopping if no active rays
            if not tf.reduce_any(valid):
                break

            # Add dimension for broadcasting with coordinates
            # [num_tx, samples_per_tx, 1]
            valid = tf.expand_dims(valid, axis=-1)

            # Change of basis matrix
            # [num_tx, samples_per_tx, 2, 2]
            mat_cob = self._component_transform(last_e_r_s, last_e_r_p,
                                     e_i_s[depth], e_i_p[depth])
            # Only apply transform if valid reflection
            # [num_tx, samples_per_tx, 1, 1]
            valid_ = tf.expand_dims(valid, axis=-1)
            # [num_tx, samples_per_tx, 2, 2]
            e = tf.where(valid_, tf.linalg.matmul(mat_cob, mat_t), mat_t)
            # Only update ongoing direction for next iteration if this
            # reflection is valid and if this is not the last step
            last_e_r_s = tf.where(valid, e_r_s[depth], last_e_r_s)
            last_e_r_p = tf.where(valid, e_r_p[depth], last_e_r_p)

            # Fresnel coefficients or receive antenna pattern
            # [num_tx, samples_per_tx, 2]
            r = tf.stack([r_s[depth], r_p[depth]], -1)
            # Set the coefficients to one if non-valid reflection
            # [num_tx, samples_per_tx, 2]
            r = tf.where(valid, r, tf.ones_like(r))
            # Add a dimension to broadcast with mat_t
            # [num_tx, samples_per_tx, 2, 1]
            r = tf.expand_dims(r, axis=-1)
            # Apply Fresnel coefficient or receive antenna pattern
            # [num_tx, samples_per_tx, 2, 2]
            mat_t = r*e

        # Dump the dummy line and row
        # [num_tx, num_cells_y, num_cells_x]
        cm = cm[:,:num_cells_y,:num_cells_x]

        # Create a CoverageMap object
        cm = CoverageMap(cm_center[0,0],
                         cm_orientation,
                         cm_size,
                         cm_cell_size,
                         cm,
                         scene=self._scene,
                         dtype=self._dtype)

        return cm
