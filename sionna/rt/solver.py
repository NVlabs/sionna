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

from sionna.constants import SPEED_OF_LIGHT
from sionna.utils.tensors import expand_to_rank
from .paths import Paths
from .utils import dot, cross, phi_hat, theta_hat, theta_phi_from_unit_vec,\
    normalize
from .solver_base import SolverBase


class Solver(SolverBase):
    # pylint: disable=line-too-long
    r"""Solver(scene, solver=None, dtype=tf.complex64)

    Generates propagation paths consisting of the line-of-sight (LoS) paths and
    reflections-only paths for the currently loaded scene.

    The main inputs of the solver are:

    * A set of sources, from which rays are emitted.

    * A set of targets, at which rays are received.

    * A maximum depth, corresponding to the maximum number of reflections. A
    depth of zero corresponds to LoS.

    Generation of paths is carried-out for every link, i.e., for every pair of
    source and target.

    The genration of paths consists in three steps:

    1. A list of candidate paths is generated. A candidate consists in a
    sequence of primitives on which a ray emitted by a source sequentially
    reflects until it reaches a target.

    2. The image method is applied to every candidates (in parallel) to discard
    candidates that do not correspond to valid paths, either because they
    are obstructed by another object in the scene, or because a reflection on
    one of the primitive in the sequence is impossible (reflection point outside
    of the primitive).

    3. For the valid paths, Fresnel coefficients for reflections are computed,
    considering the materials of the intersected objects, to compute transfer
    matrices for every paths.

    The output of the solver consists in, for every valid path that was found:

    * A transfer matrix, which is a 2x2 complex-valued matrix that describes the
    linear transformation incurred by the emitted field. The two dimensions
    correspond to the two polarization components (S and P).

    * A delay

    * Azimuth and zenith angles of arrival

    * Azimuth and zenith angles of departure

    Concerning the first step, two search methods are available for the
    listing of candidates:

    * Exhaustive search, which lists all possible combinations of primitives up
    to the requested maximum depth. This method is deterministic and ensures
    that all paths are found. However, its complexity increases exponentially
    with the number of primitives and with the maximum depth. Therefore, it
    only works for scenes of low complexity and/or for small depth values.

    * Stochastic search, which find candidates by shooting and bouncing rays,
    i.e., by randomly and uniformly sampling directions from the sources at
    which rays are shoot, and bouncing the rays on the intersected primitives
    assuming perfectly specular reflections until the maximum depth is reached.
    The intersected primitives makes the candidate. This method can be applied
    to very large scenes. However, there is no guarantee that all possible
    paths are found.

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
        paths.

    sources : [num_sources, 3], tf.float
        Coordinates of the sources.

    targets : [num_targets, 3], tf.float
        Coordinates of the targets.

    method : str ("exhaustive"|"stochastic")
        Method to be used to list candidate paths.
        The "exhaustive" method uses a
        brute-force approach and considers all possible chains of
        primitives up to ``max_depth``. It can therefore only be used
        for very small scenes.
        The "stochastic" method uses a shoot-and-bounce approach to find
        candidate chains of primitives. This method can be applied to very
        large scenes. However, there is no guarantee that all possible
        paths are found.

    num_samples: int
        Number of random rays to trace in order to generate candidates.
        A large sample count may exhaust GPU memory.

    seed: int
        Controls the random number generator. Using the same seed will
        always result in the same candidates.

    Output
    -------
    paths : Paths
        The simulated paths.
    """

    def __call__(self, max_depth, sources, targets, method, num_samples, seed):

        # EM properties of the materials
        # Returns: relative_permittivities, denoted by `etas`
        etas = self._build_scene_object_properties_tensors()

        # Candidates are generated according to the specified `method`.
        if method == 'exhaustive':
            # List all possible sequences of primitives with length up to
            # ``max_depth``
            candidates = self._list_candidates_exhaustive(max_depth)
        elif method == 'stochastic':
            # Sample sequences of primitives randomly using shoot-and-bounce
            # with length up to ``max_depth``
            candidates = self._list_candidates_stochastic(max_depth, sources,
                                num_samples=num_samples, seed=seed)

        # Compute specular paths from candidates
        # Returns: mask, vertices, normals, objects
        valid_candidates = self._image_method(candidates, sources, targets)

        # Compute channel coefficients and delays
        # Returns: mask, mat_t, tau, theta_t, phi_t, theta_r, phi_r
        paths = self._fresnel_coefficients(sources, targets, *valid_candidates,
                                           etas)

        # Paths object to return
        # Replace the mask
        valid_candidates = (paths[0], *valid_candidates[1:])
        paths = paths[1:]
        paths = Paths(*paths, sources, targets, *valid_candidates)

        return paths

    ##################################################################
    # Internal methods
    ##################################################################

    def _list_candidates_exhaustive(self, max_depth):
        r"""
        Generate all possible candidate paths made of reflections only and the
        LoS.

        The number of candidate paths equals

            num_triangles**max_depth + 1

        where the additional path (+1) is the LoS.

        This can easily exhaust GPU memory if the number of triangles in the
        scene or the `max_depth` are too large.

        Input
        ------
        max_depth: int
            Maximum number of reflections.
            Set to 0 for LoS only.

        Output
        -------
        candidates: [max_depth, num_samples], int
            All possible candidate paths with depth up to ``max_depth``.
            Entries correspond to primitives indices.
            For paths with depth lower than ``max_depth``, -1 is used as
            padding value.
            The first path is the LoS one.
        """
        # Number of triangles
        n_prims = self._primitives.shape[0]
        # List of all triangles
        # [n_prims]
        all_prims = tf.range(n_prims, dtype=tf.int32)
        # Number of candidate paths made of reflections only
        # num_samples = n_prims + n_prims^2 + ... + n_prims^max_depth
        if n_prims == 0:
            num_samples = 0
        elif n_prims == 1:
            num_samples = max_depth
        else:
            num_samples = (n_prims * (n_prims ** max_depth - 1))//(n_prims - 1)
        # Add LoS path
        num_samples += 1
        # Tensor of all possible reflections
        # Shape : [max_depth , num_samples]
        # It is transposed to fit the expected output shape at the end of this
        # function.
        # all_candidates[i,j] correspond to the triangle index intersected
        # by the i^th path for at j^th reflection.
        # The first column corresponds to LoS, i.e., no interaction.
        # -1 is used as padding value for path with depth lower than
        # max_depth.
        # Initialized with -1.
        all_candidates = tf.fill([num_samples, max_depth], -1)
        # The next loop fill all_candidates with the list of intersected
        # primitives for all possible paths made of reflections only.
        # It starts from the paths with the 1 reflection, up to max_depth.
        # The variable offset corresponds to the index offset for storing the
        # paths in all_candidates.
        # It is initialized to 1 as the first path (depth = 0) corresponds to
        # LoS.
        offset = 1
        for depth in range(1, max_depth+1):
            # Enumerate all possible interactions for this depth
            # List of `depth` tensors with shape
            # [n_prims, ..., n_prims] and rank `depth`
            candidates = tf.meshgrid(*([all_prims] * depth), indexing='ij')

            # Reshape to
            # [n_prims**depth,depth]
            candidates = tf.stack([tf.reshape(c, [-1]) for c in candidates],
                                    axis=1)

            # Pad with -1 for paths shorter than max_depth
            # [n_prims**depth,max_depth]
            candidates = tf.pad(candidates, [[0,0],[0,max_depth-depth]],
                                mode='CONSTANT', constant_values=-1)

            # Update all_candidates
            # Number of candidate paths for this depth
            num_candidates = candidates.shape[0]
            # Corresponding row indices in the all_candidates tensor
            indices = tf.range(offset, offset+num_candidates, dtype=tf.int32)
            indices = tf.expand_dims(indices, -1)
            # all_candidates : [max_depth , num_samples]
            all_candidates = tf.tensor_scatter_nd_update(all_candidates,
                                                         indices, candidates)

            # Prepare for next iteration
            offset += num_candidates

        # Transpose to fit the expected output shape.
        # [max_depth, num_samples]
        all_candidates = tf.transpose(all_candidates)
        return all_candidates


    def _list_candidates_stochastic(self, max_depth, sources,
                                    num_samples, seed):
        r"""
        Generate potential candidate paths made of reflections only and the
        LoS. Unlike `_list_candidates_exhaustive`, this search is not
        exhaustive but stochastic.

        This can be used when the triangle count or maximum depth make the
        exhaustive method impractical.

        A budget of ``num_samples`` rays is split equally over the given
        sources. Starting directions are sampled uniformly at random.
        Paths are simulated until the maximum depth is reached.
        We record all sequences of primitives hit and the prefixes of these
        sequences, and return unique sequences.

        If a `catch_sphere_radius` is specified, we further restrict the set of
        candidates returned to the ones intersecting a virtual sphere placed
        around each target.

        Input
        ------
        max_depth: int
            Maximum number of reflections.
            Set to 0 for LoS only.

        sources : [num_sources, 3], tf.float
            Coordinates of the sources.

        num_samples: int
            Number of random rays to trace in order to generate candidates.
            A large sample count may exhaust GPU memory.

        seed: int
            Controls the random number generator. Using the same seed will
            always result in the same candidates.

        Output
        -------
        candidates: [max_depth, num paths], int
            All unique candidate paths found by random sampling, with
            depth up to ``max_depth``.
            Entries correspond to triangle indices.
            For paths with depth lower than max_depth, -1 is used as
            padding value. The first path is LoS.
        """
        mask_t = dr.mask_t(self._mi_scalar_t)

        # Ensure that sample count can be distributed over the emitters
        num_sources = sources.shape[0]
        samples_per_source = int(dr.ceil(num_samples / num_sources))
        num_samples = num_sources * samples_per_source

        # Samples for sampling random directions for shooting the rays from the
        # sources.
        sampler = mi.load_dict({'type': 'independent'})
        sampler.seed(seed, num_samples)

        # List of candidates
        results = []

        # Is the scene empty?
        is_empty = dr.shape(self._shape_indices)[0] == 0

        # Only shoot if the scene is not empty
        if not is_empty:

            # Keep track of which paths are still active
            active = dr.full(mask_t, True, num_samples)

            # Initial ray: direction sampled at random, origin placed on the
            # given sources
            source_i = dr.linspace(self._mi_scalar_t, 0, num_sources,
                                num=num_samples, endpoint=False)
            source_i = mi.Int32(source_i)
            sampled_d = mi.warp.square_to_uniform_sphere(sampler.next_2d())
            sources_dr = self._mi_tensor_t(sources)
            ray = mi.Ray3f(
                o=dr.gather(self._mi_vec_t, sources_dr.array, source_i),
                d=sampled_d,
            )

            for depth in range(max_depth):

                # Intersect ray against the scene to find the next hitted
                # primitive
                si = self._mi_scene.ray_intersect(ray, active)

                active &= si.is_valid()

                # Record which primitives were hit
                shape_i = dr.gather(mi.Int32, self._shape_indices,
                                    dr.reinterpret_array_v(mi.UInt32, si.shape),
                                    active)
                offsets = dr.gather(mi.Int32, self._prim_offsets, shape_i,
                                    active)
                prims_i = dr.select(active, offsets + si.prim_index, -1)
                results.append(prims_i)

                # Prepare the next interaction, assuming purely specular
                # reflection
                ray = si.spawn_ray(si.to_world(mi.reflect(si.wi)))

        if (max_depth == 0) or (len(results) == 0):
            # If only LoS is requested or if no interaction was found
            # (empty scene), then the only candidate is the LoS
            results_tf = tf.fill([0, 1], -1)
            # Set the max_depth to 0
            max_depth = 0
        else:
            # Stack all found interactions along the depth dimension
            # [max_depth, num_samples]
            results_tf = tf.stack([self._mi_to_tf_tensor(r, tf.int32)
                                   for r in results], axis=0)
            # Max depth is updated to the highest number of reflections that
            # was found for a path
            max_depth = results_tf.shape[0]
            # Add line-of-sight
            results_tf = tf.concat([tf.fill((max_depth, 1), -1), results_tf],
                                axis=1)

        # Remove duplicates
        if max_depth > 0:
            results_tf, _ = tf.raw_ops.UniqueV2(
                x=results_tf,
                axis=[1]
            )

        # The previous shoot and bounce process does not do next-event
        # estimation, and continues to trace until max_depth reflections occurs
        # or the ray does not intersect any primitive.
        # Therefore, we extend the set of rays with the prefixes of all
        # rays in `results_tf` to ensure we don't miss shorter paths than the
        # ones found.
        results = [results_tf]
        for depth in range(1, max_depth):
            # Extract prefix of length depth
            # [depth, num_samples]
            prefix = results_tf[:depth]
            # Pad with -1, i.e., not intersection
            # [max_depth, num_samples]
            prefix = tf.pad(prefix, [[0, max_depth-depth], [0,0]],
                            constant_values=-1)
            # Add to the list of rays
            results.insert(0, prefix)
        # [max_depth, num_samples]
        results = tf.concat(results, axis=1)

        # Extending the rays with prefixes might have created duplicates.
        # Remove duplicates
        if results.shape[0] > 0:
            results, _ = tf.raw_ops.UniqueV2(
                x=results,
                axis=[1]
            )

        return results

    def _moller_trumbore(self, o, d, p0, p1, p2):
        r"""
        Computes the intersection between a ray ``ray`` and a triangle defined
        by its vertices ``p0``, ``p1``, and ``p2`` using the Moller–Trumbore
        intersection algorithm.

        Input
        -----
        o, d: [..., 3], tf.float
            Ray origin and direction.
            The direction `d` must be a unit vector.

        p0, p1, p2: [..., 3], tf.float
            Vertices defining the triangle

        Output
        -------
        t : [...], tf.float
            Position along the ray from the origin at which the intersection
            occurs (if any)

        hit : [...], bool
            `True` if the ray intersects the triangle. `False` otherwise.
        """

        zero = tf.cast(0.0, self._rdtype)
        epsilon = tf.cast(SolverBase.EPSILON, self._rdtype)
        one = tf.ones((), self._rdtype)

        # [..., 3]
        e1 = p1 - p0
        e2 = p2 - p0

        # [...,3]
        pvec = cross(d, e2)
        # [...,1]
        det = dot(e1, pvec, keepdim=True)

        # If the ray is parallel to the triangle, then det = 0.
        hit = tf.greater(tf.abs(det), zero)

        # [...,3]
        tvec = o - p0
        # [...,1]
        u = tf.math.divide_no_nan(dot(tvec, pvec, keepdim=True), det)
        # [...,1]
        hit = tf.logical_and(hit,
            tf.logical_and(tf.greater_equal(u, zero), tf.less_equal(u, one)))
        # hit = tf.logical_and(tf.greater_equal(u, zero),
        #                      tf.less_equal(u, one))

        # [..., 3]
        qvec = cross(tvec, e1)
        # [...,1]
        v = tf.math.divide_no_nan(dot(d, qvec, keepdim=True), det)
        # [..., 1]
        hit = tf.logical_and(hit,
                             tf.logical_and(tf.greater_equal(v, zero),
                                            tf.less_equal(u + v, one)))
        # [..., 1]
        t = tf.math.divide_no_nan(dot(e2, qvec, keepdim=True), det)
        # [..., 1]
        hit = tf.logical_and(hit, tf.greater_equal(t, epsilon))

        # [...]
        t = tf.squeeze(t, axis=-1)
        hit = tf.squeeze(hit, axis=-1)

        return t, hit

    def _test_obstruction(self, o, d, maxt):
        r"""
        Test obstruction of a batch of rays using Mitsuba.

        Input
        -----
        o: [batch_size, 3], tf.float
            Origin of the rays

        d: [batch_size, 3], tf.float
            Direction of the rays.
            Must be unit vectors.

        maxt: [batch_size], tf.float
            Length of the ray.

        Output
        -------
        val: [batch_size], tf.bool
            `True` if the ray is obstructed, i.e., hits a primitive.
            `False` otherwise.
        """
        # Translate the origin a bit along the ray direction to avoid
        # consecutive intersection with the same primitive
        o = o + SolverBase.EPSILON*d
        # [batch_size, 3]
        mi_o = self._mi_point_t(o)
        # Ray direction
        # [batch_size, 3]
        mi_d = self._mi_vec_t(d)
        # [batch_size]
        # Reduce the ray length by a small value to avoid false positive when
        # testing for LoS to a primitive due to hitting the primitive we are
        # testing visibility to.
        maxt = maxt * (1. - 2.*SolverBase.EPSILON)
        mi_maxt = self._mi_scalar_t(maxt)
        # Mitsuba ray
        mi_ray = mi.Ray3f(o=mi_o, d=mi_d, maxt=mi_maxt, time=0.,
                          wavelengths=mi.Color0f(0.))
        # Test for obstruction using Mitsuba
        # [batch_size]
        mi_val = self._mi_scene.ray_test(mi_ray)
        val = self._mi_to_tf_tensor(mi_val, tf.bool)
        return val

    def _image_method(self, candidates, sources, targets):
        # pylint: disable=line-too-long
        r"""
        Evaluates a list of candidate paths ``candidates`` and keep only the
        valid ones, i.e., the non-obstricted ones with valid reflections only,
        using the image method.

        Input
        -----
        candidates: [max_depth, num_samples], tf.int
            Set of candidate paths with depth up to ``max_depth``.
            For paths with depth lower than ``max_depth``, -1 must be used as
            padding value.
            The first path is LoS.

        sources: [num_src, 3]
            Positions of the sources of rays

        targets: [num_targets, 3]
            Positions of the targets of rays

        Output
        -------
        mask: [num_targets, num_sources, max_num_paths], tf.bool
            Mask indicating if a path is valid.

        valid_vertices: [max_depth, num_targets, num_sources, max_num_paths, 3], tf.float
            Positions of intersection points.

        valid_normals: [max_depth, num_targets, num_sources, max_num_paths, 3], tf.float
            Normals to the primitives at the intersection points.

        valid_objects: [max_depth, num_targets, num_sources, max_num_paths], tf.int
            Indices of the intersected scene objects.
        """

        # Max depth
        max_depth = candidates.shape[0]

        # Number of candidates
        num_samples = candidates.shape[1]

        # Number of sources and number of receivers
        num_src = len(sources)
        num_targets = len(targets)

        # --- Phase 1
        # Starting from the sources, mirror each point against the
        # given candidate primitive. At this stage, we do not carry
        # any verification about the visibility of the ray.
        # Loop through the max_depth interactions. All candidate paths are
        # processed in parallel.

        # Sturctures are filled by the following loop
        # Indicates if a path is discarded
        # [num_samples]
        invalid = tf.fill([num_samples], False)
        # Coordinates of the first vertex of potentially hitted triangles
        # [max_depth, num_src, num_samples, 3]
        tri_p0 = tf.zeros([max_depth, num_src, num_samples, 3],
                            dtype=self._rdtype)
        # Coordinates of the mirrored vertices
        # [max_depth, num_src, num_samples, 3]
        mirrored_vertices = tf.zeros([max_depth, num_src, num_samples, 3],
                                        dtype=self._rdtype)
        # Normals to the potentially hitted triangles
        # [max_depth, num_src, num_samples, 3]
        normals = tf.zeros([max_depth, num_src, num_samples, 3]
                            ,dtype=self._rdtype)

        # Position of the last interaction.
        # It is initialized with the sources position
        # Add an additional dimension for broadcasting with the paths
        # [num_src, 1, xyz : 1]
        current = tf.expand_dims(sources, axis=1)
        # Index of the last hit primitive
        prev_prim_idx = tf.fill([num_samples], -1)
        for depth in tf.range(max_depth):

            # Primitive indices with which paths interact at this depth
            # [num_samples]
            prim_idx = tf.gather(candidates, depth, axis=0)

            # Flag indicating which paths are still active, i.e., should be
            # tested.
            # Paths that are shorter than depth are marked as inactive
            # [num_samples]
            active = tf.not_equal(prim_idx, -1)

            # Break the loop if no active paths
            # Could happen with empty scenes, where we have only LoS
            if tf.logical_not(tf.reduce_any(active)):
                break

            # Eliminate paths that go through the same prim twice in a row
            # [num_samples]
            invalid = tf.logical_or(invalid,
                    tf.logical_and(active, tf.equal(prim_idx, prev_prim_idx)))

            # On CPU, indexing with -1 does not work. Hence we replace -1 by 0.
            # This makes no difference on the resulting paths as such paths
            # are not flaged as active.
            # valid_prim_idx = prim_idx
            valid_prim_idx = tf.where(prim_idx == -1, 0, prim_idx)

            # Mirroring of the current point with respected to the potentially
            # hitted triangle.
            # We need the coordinate of the first vertex of the potentially
            # hitted triangle.
            # To get this, we build the indexing tensor to gather only the
            # coordinate of the first index
            # [[num_samples, 1]]
            p0_index = tf.expand_dims(valid_prim_idx, axis=1)
            p0_index = tf.pad(p0_index, [[0,0], [0,1]], mode='CONSTANT',
                                constant_values=0) # First vertex
            # [num_samples, xyz : 3]
            p0 = tf.gather_nd(self._primitives, p0_index)
            # Expand rank and tile to broadcast with the number of transmitters
            # [num_src, num_samples, xyz : 3]
            p0 = tf.expand_dims(p0, axis=0)
            p0 = tf.tile(p0, [num_src, 1, 1])
            # Gather normals to potentially intersected triangles
            # [num_samples, xyz : 3]
            normal = tf.gather(self._normals, valid_prim_idx)
            # Expand rank and tile to broadcast with the number of transmitters
            # [1, num_samples, xyz : 3]
            normal = tf.expand_dims(normal, axis=0)
            normal = tf.tile(normal, [num_src, 1, 1])

            # Distance between the current intersection point (or sources)
            # and the plane the triangle is part of.
            # Note: `dist` is signed to compensate for backfacing normals when
            # needed.
            # [num_src, num_samples, 1]
            dist = dot(current, normal, keepdim=True)\
                        - dot(p0, normal, keepdim=True)
            # Coordinates of the mirrored point
            # [num_src, num_samples, xyz : 3]
            mirrored = current - 2. * dist * normal

            # Store these results
            # [max_depth, num_src, num_samples, 3]
            mirrored_vertices = tf.tensor_scatter_nd_update(mirrored_vertices,
                                                        [[depth]], [mirrored])
            # [max_depth, num_src, num_samples, 3]
            tri_p0 = tf.tensor_scatter_nd_update(tri_p0, [[depth]], [p0])
            # [max_depth, num_src, num_samples, 3]
            normals = tf.tensor_scatter_nd_update(normals, [[depth]], [normal])

            # Prepare for the next interaction
            # [num_src, num_samples, xyz : 3]
            current = mirrored
            # [num_samples]
            prev_prim_idx = prim_idx

        # --- Phase 2
        # Starting from the receivers, go over the vertices in reverse
        # and check that connections are possible.

        # Positions of the last interactions.
        # Initialized with the positions of the receivers.
        # Add two additional dimensions for broadcasting with transmitters and
        # paths.
        # [num_targets, 1, 1, xyz : 3]
        current = expand_to_rank(targets, 4, axis=1)
        # Expand `invalid` for broadcasting with receivers and transmitters
        # [1, 1, num_samples]
        invalid = expand_to_rank(invalid, 3, axis=0)
        # Positions of the interactions.
        # [max_depth, num_targets, num_src, num_samples, xyz : 3]
        path_vertices = tf.zeros([max_depth, num_targets, num_src,
                                    num_samples, 3], dtype=self._rdtype)
        # Normals at the interactions.
        # [max_depth, num_targets, num_src, num_samples, xyz : 3]
        path_normals = tf.zeros([max_depth, num_targets, num_src,
                                 num_samples, 3],
                                    dtype=self._rdtype)
        for depth in tf.range(max_depth-1, -1, -1):

            # Primitive indices with which paths interact at this depth
            # [num_samples]
            prim_idx = tf.gather(candidates, depth, axis=0)

            # Since paths can have different depths, we have to mask out paths
            # that have not started yet.
            # [num_samples]
            active = tf.not_equal(prim_idx, -1)

            # Break the loop if no active paths
            # Could happen with empty scenes, where we have only LoS
            if tf.logical_not(tf.reduce_any(active)):
                break

            # Expand rank to broadcast with receivers and transmitters
            # [1, 1, num_samples]
            active = expand_to_rank(active, 3, axis=0)
            active = tf.logical_and(active, tf.logical_not(invalid))

            # On CPU, indexing with -1 does not work. Hence we replace -1 by 0.
            # This makes no difference on the resulting paths as such paths
            # are not flaged as active.
            # valid_prim_idx = prim_idx
            valid_prim_idx = tf.where(prim_idx == -1, 0, prim_idx)

            # Trace a direct line from the current position to the next path
            # vertex.

            # Next interaction point
            # [num_src, num_samples, 3]
            next_pos = tf.gather(mirrored_vertices, depth, axis=0)
            # Expand rank to broadcast with receivers
            # [1, num_src, num_samples, 3]
            next_pos = tf.expand_dims(next_pos, axis=0)

            # Ray direction
            # [num_targets, num_src, num_samples, 3]
            d,_ = normalize(next_pos - current)

            # Find where it intersects the primitive that we mirrored against.
            # If that falls out of the primitive, this whole path is invalid.

            # Vertices forming the triangle.
            # [num_src, num_samples, xyz : 3]
            p0 = tf.gather(tri_p0, depth, axis=0)
            # Expand rank to broadcast with the target dimension
            # [1, num_src, num_samples, xyz : 3]
            p0 = tf.expand_dims(p0, axis=0)
            # Build the indexing tensor to gather only the coordinate of the
            # second index
            # [[num_samples, 1]]
            p1_index = tf.expand_dims(valid_prim_idx, axis=1)
            p1_index = tf.pad(p1_index, [[0,0], [0,1]], mode='CONSTANT',
                                constant_values=1) # Second vertex
            # [num_samples, xyz : 3]
            p1 = tf.gather_nd(self._primitives, p1_index)
            # Expand rank to broadcast with the target and sources
            # dimensions
            # [1, 1, num_samples, xyz : 3]
            p1 = expand_to_rank(p1, 4, axis=0)
            # Build the indexing tensor to gather only the coordinate of the
            # third index
            # [[num_samples, 1]]
            p2_index = tf.expand_dims(valid_prim_idx, axis=1)
            p2_index = tf.pad(p2_index, [[0,0], [0,1]], mode='CONSTANT',
                                constant_values=2) # Third vertex
            # [num_samples, xyz : 3]
            p2 = tf.gather_nd(self._primitives, p2_index)
            # Expand rank to broadcast with the target and sources
            # dimensions
            # [1, 1, num_samples, xyz : 3]
            p2 = expand_to_rank(p2, 4, axis=0)
            # Intersection test.
            # We use The Moeller Trumbore algorithm, implemented
            # t : [num_targets, num_src, num_samples]
            # hit : [num_targets, num_src, num_samples]
            t, hit = self._moller_trumbore(current, d, p0, p1, p2)
            invalid = tf.logical_or(invalid,
                                    tf.logical_and(active, ~hit))
            active = tf.logical_and(active, ~invalid)

            # Force normal to point towards our current position
            # [num_src, num_samples, 3]
            n = tf.gather(normals, depth, axis=0)
            # Add dimension for broadcasting with receivers
            # [1, num_src, num_samples, 3]
            n = tf.expand_dims(n, axis=0)
            # Force to point towards current position
            # [num_targets, num_src, num_samples, 3]
            s = tf.sign(dot(n, current-p0, keepdim=True))
            n = n * s
            # Intersection point
            # [num_targets, num_src, num_samples, 3]
            t = tf.expand_dims(t, axis=3)
            p = current + t*d
            # Store the intersection point
            # [max_depth, num_targets, num_src, num_samples, 3]
            path_vertices = tf.tensor_scatter_nd_update(path_vertices,
                [[depth]], [p])
            # Store the intersection normals
            # [max_depth, num_targets, num_src, num_samples, 3]
            path_normals = tf.tensor_scatter_nd_update(path_normals,
                [[depth]], [n])

            # Moreover, there should be no obstruction between the actual
            # interaction point and the current point.
            # We use Mitsuba to test for obstruction efficiently.

            # Ensure current is already broadcasted
            # [num_targets, num_src, num_samples, 3]
            current = tf.broadcast_to(current, [num_targets, num_src,
                                                num_samples, 3])
            # Distance from current to intersection point
            # [num_targets, num_src, num_samples]
            maxt = tf.norm(current - p, axis=-1)
            # Test for obstruction using Mitsuba
            # As Mitsuba only hanldes a single batch dimension, we flatten the
            # batch dims [num_targets, num_src, num_samples]
            # [num_targets*num_src*num_samples]
            val = self._test_obstruction(tf.reshape(current, [-1, 3]),
                                         tf.reshape(d, [-1, 3]),
                                         tf.reshape(maxt, [-1]))
            # [num_targets, num_src, num_samples]
            val = tf.reshape(val, [num_targets, num_src, num_samples])
            invalid = tf.logical_or(invalid, tf.logical_and(active, val))
            active = tf.logical_and(active, ~invalid)

            # Discard paths for which the shooted ray has zero-length, i.e.,
            # when two consecutive intersection points have the same location,
            # or when the source and target have the same locations (RADAR).
            # [num_targets, num_src, num_samples]
            val = tf.less(maxt, SolverBase.EPSILON)
            # [num_targets, num_src, num_samples]
            invalid = tf.logical_or(invalid, tf.logical_and(active, val))
            active = tf.logical_and(active, ~invalid)

            # We must also ensure that the current point and the next_pos are
            # not on the same side, as this would mean that the path is going
            # through the surface

            # Vector from the intersection point to the current point
            # [num_targets, num_src, num_samples, 3]
            v1 = current - p
            # Vector from the intersection point to the next point
            # [num_targets, num_src, num_samples, 3]
            v2 = next_pos - p
            # Compute the scalar product. It must be negative, as we are using
            # the image (next_pos)
            # [num_targets, num_src, num_samples]
            val = dot(v1, v2)
            val = tf.greater_equal(val, tf.zeros_like(val))
            invalid = tf.logical_or(invalid, tf.logical_and(active, val))
            active = tf.logical_and(active, ~invalid)

            # Prepare for next path segment
            # [num_targets, num_src, num_samples, 3]
            current = tf.where(tf.expand_dims(active, axis=-1), p, current)

        # Finally, check visibility to the transmitters
        # [1, num_src, 1, 3]
        sources_ = tf.expand_dims(tf.expand_dims(sources, axis=0),
                                        axis=2)
        # Direction vector and distance to the transmitters
        # d : [num_targets, num_src, num_samples, 3]
        # maxt : [num_targets, num_src, num_samples]
        d,maxt = normalize(sources_ - current)
        # Ensure current is already broadcasted
        # [num_targets, num_src, num_samples, 3]
        current = tf.broadcast_to(current, [num_targets, num_src,
                                            num_samples, 3])
        d = tf.broadcast_to(d, [num_targets, num_src, num_samples, 3])
        maxt = tf.broadcast_to(maxt, [num_targets, num_src, num_samples])
        # Test for obstruction using Mitsuba
        # [num_targets*num_src*num_samples]
        val = self._test_obstruction(tf.reshape(current, [-1, 3]),
                                     tf.reshape(d, [-1, 3]),
                                     tf.reshape(maxt, [-1]))
        # [num_targets, num_src, num_samples, 3]
        val = tf.reshape(val, [num_targets, num_src, num_samples])
        invalid = tf.logical_or(invalid, val)
        # Discard paths for which the shooted ray has zero-length, i.e., when
        # two consecutive intersection points have the same location, or when
        # the source and target have the same locations (RADAR).
        # [num_targets, num_src, num_samples]
        val = tf.less(maxt, SolverBase.EPSILON)
        # [num_targets, num_src, num_samples]
        invalid = tf.logical_or(invalid, val)

        # --- Phase 3
        # Output the valid paths, from transmitters to receivers.

        # [num_targets, num_src, num_samples]
        valid = tf.logical_not(invalid)

        # [num_targets, num_src]
        num_paths = tf.reduce_sum(tf.cast(valid, tf.int32), axis=-1)
        # Maximum number of paths
        # ()
        max_num_paths = tf.reduce_max(num_paths)

        # Build indices for keeping only valid path
        gather_indices = tf.where(valid)
        path_indices = tf.cumsum(tf.cast(valid, tf.int32), axis=-1)
        path_indices = tf.gather_nd(path_indices, gather_indices) - 1
        scatter_indices = tf.transpose(gather_indices, [1,0])
        if not tf.size(scatter_indices) == 0:
            scatter_indices = tf.tensor_scatter_nd_update(scatter_indices,
                                [[2]], [path_indices])
        scatter_indices = tf.transpose(scatter_indices, [1,0])

        # Mask of valid paths
        # [num_targets, num_src, max_num_paths]
        mask = tf.fill([num_targets, num_src, max_num_paths], False)
        mask = tf.tensor_scatter_nd_update(mask, scatter_indices,
                tf.fill([scatter_indices.shape[0]], True))
        # Locations of the interactions
        # [max_depth, num_targets, num_src, max_num_paths, 3]
        valid_vertices = tf.zeros([max_depth, num_targets, num_src,
                                    max_num_paths, 3], dtype=self._rdtype)
        # Normals at the intersection points
        # [max_depth, num_targets, num_src, max_num_paths, 3]
        valid_normals = tf.zeros([max_depth, num_targets, num_src,
                                    max_num_paths, 3], dtype=self._rdtype)
        # [max_depth, num_targets, num_src, max_num_paths]
        valid_primitives = tf.fill([max_depth, num_targets, num_src,
                                        max_num_paths], -1)

        for depth in tf.range(max_depth, dtype=tf.int64):

            # Indices for storing the valid vertices/normals/primitives for
            # this depth
            scatter_indices_ = tf.pad(scatter_indices, [[0,0], [1,0]],
                            mode='CONSTANT', constant_values=depth)

            # Loaction of the interactions
            # Extract only the valid paths
            # [num_targets, num_src, num_samples, 3]
            vertices_ = tf.gather(path_vertices, depth, axis=0)
            # [total_num_valid_paths, 3]
            vertices_ = tf.gather_nd(vertices_, gather_indices)
            # Store the valid intersection points
            # [max_depth, num_targets, num_src, max_num_paths, 3]
            valid_vertices = tf.tensor_scatter_nd_update(valid_vertices,
                                            scatter_indices_, vertices_)

            # Normals at the interactions
            # Extract only the valid paths
            # [num_targets, num_src, num_samples, 3]
            normals_ = tf.gather(path_normals, depth, axis=0)
            # [total_num_valid_paths, 3]
            normals_ = tf.gather_nd(normals_, gather_indices)
            # Store the valid normals
            # [max_depth, num_targets, num_src, max_num_paths, 3]
            valid_normals = tf.tensor_scatter_nd_update(valid_normals,
                                    scatter_indices_, normals_)

            # Intersected primitives
            # Extract only the valid paths
            # [num_samples]
            primitives_ = tf.gather(candidates, depth, axis=0)
            # [total_num_valid_paths]
            primitives_ = tf.gather(primitives_, gather_indices[:,2])
            # Store the valid primitives]
            # [max_depth, num_targets, num_src, max_num_paths]
            valid_primitives = tf.tensor_scatter_nd_update(valid_primitives,
                                    scatter_indices_, primitives_)

        # Add a dummy entry to primitives_2_objects with value -1 for invalid
        # reflection.
        # Invalid reflection, i.e., corresponding to paths with a depth lower
        # than max_depth, will be assigned -1 as index of the intersected
        # shape.
        # [num_samples + 1]
        primitives_2_objects = tf.pad(self._primitives_2_objects, [[0,1]],
                                        constant_values=-1)
        # Replace all -1 by num_samples
        num_samples = self._primitives_2_objects.shape[0]
        # [max_depth, num_targets, num_src, max_num_paths]
        valid_primitives = tf.where(tf.equal(valid_primitives,-1),
                                    num_samples,
                                    valid_primitives)
        # [max_depth, num_targets, num_src, max_num_paths]
        valid_objects = tf.gather(primitives_2_objects, valid_primitives)

        return mask, valid_vertices, valid_normals, valid_objects

    def _fresnel_coefficients(self, sources, targets, mask, vertices, normals,
                                objects, relative_permittivity):
        # pylint: disable=line-too-long
        """
        Compute the transfer matrices, delays, angles of departures, and angles
        of arrivals, of paths from a set of valid reflection paths and the
        EM properties of the materials.

        Input
        ------
        sources: [num_src, 3]
            Positions of the sources of rays

        targets: [num_targets, 3]
            Positions of the targets of rays

        mask: [num_targets, num_sources, max_num_paths], tf.bool
            Mask indicating if a path is valid.

        vertices: [max_depth, num_targets, num_sources, max_num_paths, 3], tf.float
            Positions of intersection points.

        normals: [max_depth, num_targets, num_sources, max_num_paths, 3], tf.float
            Normals to the triangles at the intersection points.

        objects: [max_depth, num_targets, num_sources, max_num_paths], tf.int
            Indices of the intersected primitives at the interaction points.

        relative_permittivity : [num_shape], tf.complex
            Tensor containing the relative permittivity of all shapes

        Output
        -------
        mask: [num_targets, num_sources, max_num_paths], tf.bool
            Mask indicating if a path is valid.

        mat_t : [num_targets, num_sources, max_num_paths, 2, 2], tf.complex
            Transfer matrices for the paths.
            These are 2x2 complex-valued matrices modeling the transformation
            experienced by the field due to propagation. The two components
            correspond to the S and P polarization components.
            Invalid paths, i.e., non-existing paths for links with a number of
            paths lower than ``max_num_path``, are padded with zeros.

        tau : [num_targets, num_sources, max_num_paths], tf.float
            Delays in seconds.
            Invalid paths, i.e., non-existing paths for links with a number of
            paths lower than ``max_num_path``, are padded with -1.


        theta_t : [num_targets, num_sources, max_num_paths], tf.float
            Zenith departure angle in radian.

        phi_t : [num_targets, num_sources, max_num_paths], tf.float
            Azimuth departure angle in radian.

        theta_r : [num_targets, num_sources, max_num_paths], tf.float
            Zenith arrival angle in radian.

        phi_r : [num_targets, num_sources, max_num_paths], tf.float
            Azimuth departure angle in radian.
        """

        # Maximum depth
        max_depth = vertices.shape[0]
        # Number of targets
        num_targets = targets.shape[0]
        # Number of sources
        num_src = sources.shape[0]
        # Maximum number of paths
        max_num_paths = mask.shape[2]

        # Flag that indicates if a ray is valid
        # [max_depth, num_targets, num_sources, max_num_paths]
        valid_ray = tf.not_equal(objects, -1)

        # Tensor with relative perimittivities values for all reflection points
        # On CPU, indexing with -1 does not work. Hence we replace -1 by 0.
        # This makes no difference on the resulting paths as such paths
        # are not flaged as active.
        # [max_depth, num_targets, num_sources, max_num_paths]
        valid_object_idx = tf.where(objects == -1, 0, objects)
        # [max_depth, num_targets, num_src, max_num_paths]
        if relative_permittivity.shape[0] == 0:
            etas = tf.zeros_like(valid_object_idx, dtype=self._dtype)
        else:
            etas = tf.gather(relative_permittivity, valid_object_idx)

        # Vertices updated with the sources and targets
        # [1, num_src, 1, 3]
        sources = tf.expand_dims(tf.expand_dims(sources, axis=0), axis=2)
        # [num_targets, num_src, max_num_paths, 3]
        sources = tf.broadcast_to(sources, vertices.shape[1:])
        # [1, num_targets, num_src, max_num_paths, 3]
        sources = tf.expand_dims(sources, axis=0)
        # [1 + max_depth, num_targets, num_sources, max_num_paths, 3]
        vertices = tf.concat([sources, vertices], axis=0)
        # For the targets, we need to account for the paths having different
        # depths.
        # Pad vertices with dummy values to create the required extra depth
        # [1 + max_depth + 1, num_targets, num_sources, max_num_paths, 3]
        vertices = tf.pad(vertices, [[0,1],[0,0],[0,0],[0,0],[0,0]])
        # [num_targets, 1, 1, 3]
        targets = tf.expand_dims(tf.expand_dims(targets, axis=1), axis=2)
        # [num_targets, num_src, max_num_paths, 3]
        targets = tf.broadcast_to(targets, vertices.shape[1:])

        #  [max_depth, num_targets, num_sources, max_num_paths]
        target_indices = tf.cast(valid_ray, tf.int64)
        #  [num_targets, num_sources, max_num_paths]
        target_indices = tf.reduce_sum(target_indices, axis=0) + 1
        # [num_targets*num_sources*max_num_paths]
        target_indices = tf.reshape(target_indices, [-1,1])
        # Indices of all (target, source,paths) entries
        # [num_targets*num_sources*max_num_paths, 3]
        target_indices_ = tf.where(tf.fill(vertices.shape[1:4], True))
        # Indices of all entries in vertices
        # [num_targets*num_sources*max_num_paths, 4]
        target_indices = tf.concat([target_indices, target_indices_], axis=1)
        # Reshape targets
        # vertices : [max_depth + 1, num_targets, num_sources, max_num_paths, 3]
        targets = tf.reshape(targets, [-1,3])
        vertices = tf.tensor_scatter_nd_update(vertices, target_indices,
                                                targets)

        # Direction of arrivals (k_i)
        # The last item (k_i[depth]) correspond to the direction of arrival
        # at the target. Therefore, k_i is a tensor of length `max_depth + 1`,
        # where `max_depth` is the number of maximum interaction (which could be
        # zero if only LoS is requested).
        # k_i : [max_depth + 1, num_targets, num_sources, max_num_paths, 3]
        # ray_lengths : [max_depth + 1, num_targets, num_sources, max_num_paths]
        k_i = tf.roll(vertices, -1, axis=0) - vertices
        k_i,ray_lengths = normalize(k_i)
        k_i = k_i[:max_depth+1]
        ray_lengths = ray_lengths[:max_depth+1]

        # Compute the distance
        # [max_depth, num_targets, num_sources, max_num_paths]
        lengths_mask = tf.cast(valid_ray, self._rdtype)
        # First ray is always valid (LoS)
        # [1 + max_depth, num_targets, num_sources, max_num_paths]
        lengths_mask = tf.pad(lengths_mask, [[1,0],[0,0],[0,0],[0,0]],
                                constant_values=tf.ones((),self._rdtype))
        # Compute path distance
        # [num_targets, num_sources, max_num_paths]
        distance = tf.reduce_sum(lengths_mask*ray_lengths, axis=0)

        # Direction of departures (k_r) at interaction points.
        # We do not need the direction of departure at the source, as it
        # is the same as k_i[0]. Therefore `k_r` only stores the directions of
        # departures at the `max_depth` interaction points.
        # [max_depth, num_targets, num_sources, max_num_paths, 3]
        k_r = tf.roll(vertices, -2, axis=0) - tf.roll(vertices, -1, axis=0)
        k_r,_ = normalize(k_r)
        k_r = k_r[:max_depth]

        # Compute angles of departures and arrival
        # theta_t, phi_t: [num_targets, num_sources, max_num_paths]
        theta_t, phi_t = theta_phi_from_unit_vec(k_i[0])
        # Depth of the rays
        # [num_targets, num_sources, max_num_paths]
        ray_depth = tf.reduce_sum(tf.cast(valid_ray, tf.int32), axis=0)
        last_k_r = -tf.gather(tf.transpose(k_i, [1,2,3,0,4]), ray_depth,
                                batch_dims=3, axis=3)
        # theta_r, phi_r: [num_targets, num_sources, max_num_paths]
        theta_r, phi_r = theta_phi_from_unit_vec(last_k_r)

        # Compute cos(theta) at each reflection point
        # [max_depth, num_targets, num_sources, max_num_paths]
        cos_theta = -dot(k_i[:max_depth], normals)

        # Compute e_i_s, e_i_p, e_r_s, e_r_p at each reflection point
        # all : [max_depth, num_targets, num_sources, max_num_paths,3]
        e_i_s, e_i_p, e_r_s, e_r_p = self._compute_field_unit_vectors(
                                                k_i[:max_depth], k_r, normals)

        # Compute r_s, r_p at each reflection point
        # [max_depth, num_targets, num_src, max_num_paths]
        r_s, r_p = self._reflection_coefficient(etas, cos_theta)

        # Compute the field transfer matrix.
        # It is initialized with the identity matrix of size 2 (S and P
        # polarization components)
        # [num_targets, num_src, max_num_paths, 2, 2]
        mat_t = tf.eye(num_rows=2,
                       batch_shape=[num_targets, num_src,max_num_paths],
                       dtype=self._dtype)
        # Initialize last field unit vector with outgoing ones
        # [num_targets, num_sources, max_num_paths, 3]
        last_e_r_s = theta_hat(theta_t, phi_t)
        last_e_r_p = phi_hat(phi_t)
        for depth in tf.range(0,max_depth):
            # Is this a valid reflection?
            # [num_targets, num_sources, max_num_paths]
            valid = valid_ray[depth]

            # Early stopping if no active rays
            if not tf.reduce_any(valid):
                break

            # Add dimension for broadcasting with coordinates
            # [num_targets, num_sources, max_num_paths, 1]
            valid = tf.expand_dims(valid, axis=-1)

            # Change of basis matrix
            # [num_targets, num_sources, max_num_paths, 2, 2]
            mat_cob = self._component_transform(last_e_r_s, last_e_r_p,
                                     e_i_s[depth], e_i_p[depth])
            # Only apply transform if valid reflection
            # [num_targets, num_sources, max_num_paths, 1, 1]
            valid_ = tf.expand_dims(valid, axis=-1)
            # [num_targets, num_sources, max_num_paths, 2, 2]
            e = tf.where(valid_, tf.linalg.matmul(mat_cob, mat_t), mat_t)
            # Only update ongoing direction for next iteration if this
            # reflection is valid and if this is not the last step
            last_e_r_s = tf.where(valid, e_r_s[depth], last_e_r_s)
            last_e_r_p = tf.where(valid, e_r_p[depth], last_e_r_p)

            # Fresnel coefficients or receive antenna pattern
            # [num_targets, num_src, max_num_paths, 2]
            r = tf.stack([r_s[depth], r_p[depth]], -1)
            # Set the coefficients to one if non-valid reflection
            # [num_targets, num_src, max_num_paths, 2]
            r = tf.where(valid, r, tf.ones_like(r))
            # Add a dimension to broadcast with mat_t
            # [num_targets, num_src, max_num_paths, 2, 1]
            r = tf.expand_dims(r, axis=-1)
            # Apply Fresnel coefficient or receive antenna pattern
            # [num_targets, num_sources, max_num_paths, 2, 2]
            mat_t = r*e

        # Move to the targets frame
        # Transformation matrix
        # [num_targets, num_sources, max_num_paths, 2, 2]
        mat_cob = self._component_transform(last_e_r_s, last_e_r_p,
                theta_hat(theta_r, phi_r), phi_hat(phi_r))
        # Apply transformation
        # [num_targets, num_sources, max_num_paths, 2, 2]
        mat_t = tf.linalg.matmul(mat_cob, mat_t)

        # Divide by total distance to account for propagation loss
        # [num_targets, num_sources, max_num_paths, 1, 1]
        distance_ = expand_to_rank(distance, 5, axis=3)
        # [num_targets, num_src, max_num_paths, 2, 2]
        mat_t = tf.math.divide_no_nan(mat_t, tf.cast(distance_, self._dtype))

        # Remove duplicate paths.
        # Paths intersecting an edge belonging to two different triangles can
        # be considered for twice,.
        # Note that this is rare, as intersections rarely occur on edges.
        # The similarity measure used to distinguish paths if the distance
        # between the angles of arrivals and departures.
        # [num_targets, num_src, max_num_paths, 4]
        sim = tf.stack([theta_t, phi_t, theta_r, phi_r], axis=3)
        # [num_targets, num_src, max_num_paths, max_num_paths, 4]
        sim = tf.expand_dims(sim, axis=2) - tf.expand_dims(sim, axis=3)
        # [num_targets, num_src, max_num_paths, max_num_paths]
        sim = tf.reduce_sum(tf.square(sim), axis=4)
        sim = tf.equal(sim, tf.zeros_like(sim))
        # Keep only the paths with no duplicates.
        # If many paths are identical, keep the one with the highest index
        # [num_targets, num_src, max_num_paths, max_num_paths]
        sim = tf.logical_and(tf.linalg.band_part(sim, 0, -1),
                             ~tf.eye(sim.shape[-1],
                                     dtype=tf.bool,
                                     batch_shape=sim.shape[:2]))
        sim = tf.logical_and(sim, tf.expand_dims(mask, axis=-2))
        # [num_targets, num_src, max_num_paths]
        uniques = tf.reduce_all(~sim, axis=3)
        # Keep only the unique paths
        # [num_targets, num_src, max_num_paths]
        mask = tf.logical_and(uniques, mask)

        # Set invalid paths to 0
        # Expand masks to broadcast with the field components
        # [num_targets, num_sources, max_num_paths, 1, 1]
        mask_ = expand_to_rank(mask, 5, axis=3)
        # Zeroing coefficients corresponding to non-valid paths
        # [num_targets, num_src, max_num_paths, 2]
        mat_t = tf.where(mask_, mat_t, tf.zeros_like(mat_t))
        # Propagation delay [s]
        # [num_targets, num_src, max_num_paths]
        tau = distance / SPEED_OF_LIGHT
        # Setting -1 for delays corresponding to non-valid paths
        # [num_targets, num_src, max_num_paths]
        tau = tf.where(mask, tau, -tf.ones_like(tau))

        return mask, mat_t, tau, theta_t, phi_t, theta_r, phi_r
