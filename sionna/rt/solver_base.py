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

from sionna.utils.tensors import expand_to_rank
from .utils import dot, cross, normalize

class SolverBase:
    # pylint: disable=line-too-long
    r"""SolverBase(scene, solver=None, dtype=tf.complex64)

    Base class for implementing a solver. If another ``solver`` is specified at
    instantiation, then it re-uses the structure to avoid useless compute and
    memory use.

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
    """

    # Small value used to discard intersection with edges, avoid
    # self-intersection, etc.
    EPSILON = 1e-5

    def __init__(self, scene, solver=None, dtype=tf.complex64):

        # Computes the quantities required for generating the paths.
        # More pricisely:

        # _primitives : [num triangles, 3, 3], float
        #     The triangles: x-y-z coordinates of the 3 vertices for every
        #     triangle

        # _normals : [num triangles, 3], float
        #     The normals of the triangles

        # _pimitives_2_objects : [num_triangles], int
        #     Index of the shape containing the triangle

        # _prim_offsets : [num_objects], int
        #     Indices offsets for accessing the triangles making each shape.

        # _shape_indices : [num_objects], int
        #     Map Mitsuba shape indices to indices that can be used to access
        #    _prim_offsets

        assert dtype in (tf.complex64, tf.complex128),\
            "`dtype` must be tf.complex64 or tf.complex64`"
        self._dtype = dtype
        self._rdtype = dtype.real_dtype

        # Mitsuba types depend on the used precision
        if dtype == tf.complex64:
            self._mi_point_t = mi.Point3f
            self._mi_vec_t = mi.Vector3f
            self._mi_scalar_t = mi.Float
            self._mi_tensor_t = mi.TensorXf
        else:
            self._mi_point_t = mi.Point3d
            self._mi_vec_t = mi.Vector3d
            self._mi_scalar_t = mi.Float64
            self._mi_tensor_t = mi.TensorXd

        self._scene = scene
        mi_scene = scene.mi_scene
        self._mi_scene = mi_scene

        # If a solver is provided, then link to the same structures to avoid
        # useless compute and memory use
        if solver is not None:
            self._primitives = solver._primitives
            self._normals = solver._normals
            self._primitives_2_objects = solver._primitives_2_objects
            self._prim_offsets = solver._prim_offsets
            self._shape_indices = solver._shape_indices
            return

        ###################################################
        # Extract triangles, their normals, and a
        # look-up-table to map primitives to the scene
        # object they belong to.
        ###################################################

        # Tensor mapping primitives to corresponding objects
        # [num_triangles]
        primitives_2_objects = []

        # Number of triangles
        n_prims = 0
        # Triangles of each object (shape) in the scene are stacked.
        # This list tracks the indices offsets for accessing the triangles
        # making each shape.
        prim_offsets = []
        for i,s in enumerate(mi_scene.shapes()):
            if not isinstance(s, mi.Mesh):
                raise ValueError('Only triangle meshes are supported')
            prim_offsets.append(n_prims)
            n_prims += s.face_count()
            primitives_2_objects += [i]*s.face_count()
        # [num_objects]
        prim_offsets = tf.cast(prim_offsets, tf.int32)

        # Tensor of triangles vertices
        # [n_prims, number of vertices : 3, coordinates : 3]
        prims = tf.zeros([n_prims, 3, 3], self._rdtype)
        # Normals to the triangles
        normals = tf.zeros([n_prims, 3], self._rdtype)
        # Loop through the objects in the scene
        for prim_offset, s in zip(prim_offsets, mi_scene.shapes()):
            # Extract the vertices of the shape.
            # Dr.JIT/Mitsuba is used here.
            # Indices of the vertices
            # [n_prims, num of vertices per triangle : 3]
            face_indices3 = s.face_indices(dr.arange(mi.UInt32, s.face_count()))
            # Flatten. This is required for calling vertex_position
            # [n_prims*3]
            face_indices = dr.ravel(face_indices3)
            # Get vertices coordinates
            # [n_prims*3, 3]
            vertex_coords = s.vertex_position(face_indices)
            # Move to TensorFlow
            # [n_prims*3, 3]
            vertex_coords = self._mi_to_tf_tensor(vertex_coords, self._rdtype)
            # Unflatten
            # [n_prims, vertices per triangle : 3, 3]
            vertex_coords = tf.reshape(vertex_coords, [s.face_count(), 3, 3])
            # Update the `prims` tensor
            sl = tf.range(prim_offset, prim_offset + s.face_count(),
                          dtype=tf.int32)
            sl = tf.expand_dims(sl, axis=1)
            prims = tf.tensor_scatter_nd_update(prims, sl, vertex_coords)
            # Compute the normals to the triangles
            # Coordinate of the first vertices of every triangle making the
            # shape
            # [n_prims, xyz : 3]
            v0 = s.vertex_position(face_indices3.x)
            # Coordinate of the second vertices of every triangle making the
            # shape
            # [n_prims, xyz : 3]
            v1 = s.vertex_position(face_indices3.y)
            # Coordinate of the third vertices of every triangle making the
            # shape
            # [n_prims, xyz : 3]
            v2 = s.vertex_position(face_indices3.z)
            # Compute the normals
            # [n_prims, xyz : 3]
            mi_n = dr.normalize(dr.cross(
                v1 - v0,
                v2 - v0,
            ))
            # Move to TensorFlow
            # [n_prims, 3]
            n = self._mi_to_tf_tensor(mi_n, self._rdtype)
            # Update the 'normals' tensor
            normals = tf.tensor_scatter_nd_update(normals, sl, n)

        self._primitives = prims
        self._normals = normals
        self._primitives_2_objects = tf.cast(primitives_2_objects, tf.int32)

        ####################################################
        # Used by the shoot & bounce method to map from
        # (shape, local primitive index) to the
        # corresponding global primitive index.
        ####################################################

        # [num_objects]
        self._prim_offsets = mi.Int32(prim_offsets.numpy())
        dest = dr.reinterpret_array_v(mi.UInt32, mi_scene.shapes_dr())
        if dr.width(dest) == 0:
            self._shape_indices = mi.Int32([])
        else:
            # [num_objects]
            shape_indices = dr.full(mi.Int32, -1, dr.max(dest)[0] + 1)
            dr.scatter(shape_indices, dr.arange(mi.Int32, 0,
                       dr.width(dest)), dest)
            dr.eval(shape_indices)
            # [num_objects]
            self._shape_indices = shape_indices

    ##################################################################
    # Internal methods
    ##################################################################

    def _build_scene_object_properties_tensors(self):
        r"""
        Build tensor containing the shape properties

        Input
        ------
        None

        Output
        -------
        relative_permittivity : [num_shape], tf.complex
            Tensor containing the complex relative permittivity of all shapes
        """

        relative_permittivity = []
        for s in self._scene.objects.values():
            rm = s.radio_material
            relative_permittivity.append(rm.complex_relative_permittivity)
        relative_permittivity = tf.stack(relative_permittivity, axis=0)
        relative_permittivity = tf.cast(relative_permittivity, self._dtype)

        return relative_permittivity

    def _mi_to_tf_tensor(self, mi_tensor, dtype):
        """
        Get a TensorFlow eager tensor from a Mitsuba/DrJIT tensor
        """
        # When there is only one input, the .tf() methods crashes.
        # The following hack takes care of this corner case
        dr.eval(mi_tensor)
        dr.sync_thread()
        if dr.shape(mi_tensor)[-1] == 1:
            mi_tensor = dr.repeat(mi_tensor, 2)
            tf_tensor = tf.cast(mi_tensor.tf(), dtype)[:1]
        else:
            tf_tensor = tf.cast(mi_tensor.tf(), dtype)
        return tf_tensor

    def _gen_orthogonal_vector(self, k):
        """
        Generate an arbitrary vector that is orthogonal to ``k``.

        Input
        ------
        k : [..., 3], tf.float
            Vector

        Output
        -------
        : [..., 3], tf.float
            Vector orthogonal to ``k``
        """
        ex = tf.cast([1.0, 0.0, 0.0], self._rdtype)
        ex = expand_to_rank(ex, tf.rank(k), 0)

        ey = tf.cast([0.0, 1.0, 0.0], self._rdtype)
        ey = expand_to_rank(ey, tf.rank(k), 0)

        n1 = cross(k, ex)
        n1_norm = tf.norm(n1, axis=-1, keepdims=True)
        n2 = cross(k, ey)
        return tf.where(tf.greater(n1_norm, SolverBase.EPSILON), n1, n2)

    def _compute_field_unit_vectors(self, k_i, k_r, n):
        """
        Compute unit vector parallel and orthogonal to incident plane

        Input
        ------
        k_i : [..., 3], tf.float
            Direction of arrival

        k_r : [..., 3], tf.float
            Direction of reflection

        n : [..., 3], tf.float
            Surface normal

        Output
        ------
        e_i_s : [..., 3], tf.float
            Incident unit field vector for S polarization

        e_i_p : [..., 3], tf.float
            Incident unit field vector for P polarization

        e_r_s : [..., 3], tf.float
            Reflection unit field vector for S polarization

        e_r_p : [..., 3], tf.float
            Reflection unit field vector for P polarization
        """
        e_i_s = cross(k_i, n)
        e_i_s_norm = tf.norm(e_i_s, axis=-1, keepdims=True)
        # In case of normal incidence, the incidence plan is not uniquely
        # define and the Fresnel coefficent is the same for both polarization
        # (up to a sign flip for the parallel component due to the definition of
        # polarization).
        # It is required to detect such scenarios and define an arbitrary valid
        # e_i_s to fix an incidence plane, as the result from previous
        # computation leads to e_i_s = 0.
        e_i_s = tf.where(tf.greater(e_i_s_norm, SolverBase.EPSILON), e_i_s,
                        self._gen_orthogonal_vector(n))

        e_i_s,_ = normalize(e_i_s)
        e_i_p,_ = normalize(cross(e_i_s, k_i))
        e_r_s = e_i_s
        e_r_p,_ = normalize(cross(e_r_s, k_r))
        return e_i_s, e_i_p, e_r_s, e_r_p

    def _reflection_coefficient(self, eta, cos_theta):
        """
        Compute simplified reflection coefficients

        Input
        ------
        eta : Any shape, tf.complex
            Real part of the relative permittivity

        cos_thehta : Same as ``eta``, tf.float
            Cosine of the incident angle

        Output
        -------
        r_te : Same as input, tf.complex
            Fresnel reflection coefficient for S direction

        r_tm : Same as input, tf.complex
            Fresnel reflection coefficient for P direction
        """
        cos_theta = tf.cast(cos_theta, self._dtype)

        # Fresnel equations
        a = cos_theta
        b = tf.sqrt(eta-1.+cos_theta**2)
        r_te = tf.math.divide_no_nan(a-b, a+b)

        c = eta*a
        d = b
        r_tm = tf.math.divide_no_nan(c-d, c+d)
        return r_te, r_tm

    def _component_transform(self, e_s, e_p, e_i_s, e_i_p):
        """
        Compute basis change matrix for reflections

        Input
        -----
        e_s : [..., 3], tf.float
            Source unit vector for S polarization

        e_p : [..., 3], tf.float
            Source unit vector for P polarization

        e_i_s : [..., 3], tf.float
            Target unit vector for S polarization

        e_i_p : [..., 3], tf.float
            Target unit vector for P polarization

        Output
        -------
        r : [..., 2, 2], tf.float
            Change of basis matrix for going from (e_s, e_p) to (e_i_s, e_i_p)
        """
        r_11 = dot(e_i_s, e_s)
        r_12 = dot(e_i_s, e_p)
        r_21 = dot(e_i_p, e_s)
        r_22 = dot(e_i_p, e_p)
        r1 = tf.stack([r_11, r_12], axis=-1)
        r2 = tf.stack([r_21, r_22], axis=-1)
        r = tf.stack([r1, r2], axis=-2)
        return tf.cast(r, self._dtype)
