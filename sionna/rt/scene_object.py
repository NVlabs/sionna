#
# SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""
Class representing objects in the scene
"""
import tensorflow as tf

from .object import Object
from .radio_material import RadioMaterial
import drjit as dr
import mitsuba as mi
from .utils import mi_to_tf_tensor, angles_to_mitsuba_rotation, normalize,\
    theta_phi_from_unit_vec
from sionna.constants import PI


class SceneObject(Object):
    # pylint: disable=line-too-long
    r"""
    SceneObject()

    Every object in the scene is implemented by an instance of this class
    """

    def __init__(self,
                 name,
                 object_id=None,
                 mi_shape=None,
                 radio_material=None,
                 orientation=(0,0,0),
                 dtype=tf.complex64,
                 **kwargs):

        if dtype not in (tf.complex64, tf.complex128):
            raise ValueError("`dtype` must be tf.complex64 or tf.complex128`")
        self._dtype = dtype
        self._rdtype = dtype.real_dtype

        # Orientation of the object is initialized to (0,0,0)
        self._orientation = tf.cast(orientation, dtype=self._rdtype)

        # Initialize the base class Object
        super().__init__(name, **kwargs)

        # Set the radio material
        self.radio_material = radio_material

        # Set the object id
        self.object_id = object_id

        # Set the Mitsuba shape
        self._mi_shape = mi_shape

        # Set velocity vector
        self.velocity = tf.cast([0,0,0], dtype=self._rdtype)

        if self._dtype == tf.complex64:
            self._mi_point_t = mi.Point3f
            self._mi_vec_t = mi.Vector3f
            self._mi_scalar_t = mi.Float
            self._mi_transform_t = mi.Transform4f
        else:
            self._mi_point_t = mi.Point3d
            self._mi_vec_t = mi.Vector3d
            self._mi_scalar_t = mi.Float64
            self._mi_transform_t = mi.Transform4d

    @property
    def object_id(self):
        r"""
        int : Get/set the identifier of this object
        """
        return self._object_id

    @object_id.setter
    def object_id(self, v):
        self._object_id = v

    @property
    def radio_material(self):
        r"""
        :class:`~sionna.rt.RadioMaterial` : Get/set the radio material of the
        object. Setting can be done by using either an instance of
        :class:`~sionna.rt.RadioMaterial` or the material name (`str`).
        If the radio material is not part of the scene, it will be added. This
        can raise an error if a different radio material with the same name was
        already added to the scene.
        """
        return self._radio_material

    @radio_material.setter
    def radio_material(self, mat):
        # Note: _radio_material is set at __init__, but pylint doesn't see it.
        if mat is None:
            mat_obj = None

        elif isinstance(mat, str):
            mat_obj = self.scene.get(mat)
            if (mat_obj is None) or (not isinstance(mat_obj, RadioMaterial)):
                err_msg = f"Unknown radio material '{mat}'"
                raise TypeError(err_msg)

        elif not isinstance(mat, RadioMaterial):
            err_msg = ("The material must be a material name (str) or an "
                        "instance of RadioMaterial")
            raise TypeError(err_msg)

        else:
            mat_obj = mat

        # Remove the object from the set of the currently used material, if any
        # pylint: disable=access-member-before-definition
        if hasattr(self, '_radio_material') and self._radio_material:
            self._radio_material.discard_object_using(self.object_id)
        # Assign the new material
        # pylint: disable=access-member-before-definition
        self._radio_material = mat_obj

        # If the radio material is set to None, we can stop here
        # pylint: disable=access-member-before-definition
        if not self._radio_material:
            return

        # Add the object to the set of the newly used material
        # pylint: disable=access-member-before-definition
        self._radio_material.add_object_using(self.object_id)

        # Add the RadioMaterial to the scene if not already done
        self.scene.add(self._radio_material)

    @property
    def velocity(self):
        """
        [3], tf.float : Get/set the velocity vector [m/s]
        """
        return self._velocity

    @velocity.setter
    def velocity(self, v):
        if not tf.shape(v)==3:
            raise ValueError("`velocity` must have shape [3]")
        self._velocity = tf.cast(v, self._rdtype)

    @property
    def position(self):
        """
        [3], tf.float : Get/set the position vector [m] of the center
            of the object. The center is defined as the object's axis-aligned
            bounding box (AABB).
        """
        dr.sync_thread()
        rdtype = self._scene.dtype.real_dtype
        # Bounding box
        # [3]
        bbox_min = tf.cast(self._mi_shape.bbox().min, rdtype)
        # [3]
        bbox_max = tf.cast(self._mi_shape.bbox().max, rdtype)
        # [3]
        half = tf.cast(0.5, self._rdtype)
        position = half*(bbox_min + bbox_max)
        return position

    @position.setter
    def position(self, new_position):

        ## Update Mitsuba vertices

        # Scene parameters
        scene_params = self._scene.mi_scene_params
        # Real dtype
        rdtype = self._scene.dtype.real_dtype
        new_position = tf.cast(new_position, rdtype)
        # [num_vertices*3]
        vertices = scene_params[f'mesh-{self.name}.vertex_positions']
        # [num_vertices,3]
        vertices = mi_to_tf_tensor(vertices, rdtype)
        vertices = tf.reshape(vertices, [-1, 3])
        # [3]
        position = self.position
        # [3]
        translation_vector = new_position - position
        # [1,3]
        translation_vector = tf.expand_dims(translation_vector, axis=0)
        # [num_vertices,3]
        translated_vertices = vertices + translation_vector
        # Cast to Mitsuba type to object the Mitsuba scene
        fltn_translated_vertices = tf.reshape(translated_vertices, [-1])
        fltn_translated_vertices = self._mi_scalar_t(fltn_translated_vertices)
        #
        scene_params[f'mesh-{self.name}.vertex_positions'] =\
            fltn_translated_vertices
        scene_params.update()

        ## Update Sionna vertices

        obj_id = self.object_id
        mi_shape = self._mi_shape
        solver_paths = self._scene.solver_paths

        shape_ind = solver_paths.shape_indices[obj_id]
        prim_offset = solver_paths.prim_offsets[shape_ind]

        face_indices3 = mi_shape.face_indices(dr.arange(mi.UInt32,
                                                        mi_shape.face_count()))
        # Flatten. This is required for calling vertex_position
        # [n_prims*3]
        face_indices = dr.ravel(face_indices3)
        # Get vertices coordinates
        # [n_prims*3, 3]
        vertex_coords = mi_shape.vertex_position(face_indices)
        # Cast to TensorFlow type
        # [n_prims*3, 3]
        vertex_coords = mi_to_tf_tensor(vertex_coords, rdtype)
        # Unflatten
        # [n_prims, vertices per triangle : 3, 3]
        vertex_coords = tf.reshape(vertex_coords, [mi_shape.face_count(), 3, 3])
        # Update the tensor storing the primitive vertices
        sl = tf.range(prim_offset, prim_offset + mi_shape.face_count(),
                    dtype=tf.int32)
        sl = tf.expand_dims(sl, axis=1)
        solver_paths.primitives.scatter_nd_update(sl, vertex_coords)

        ## Update Sionna wedges

        wedges_objects = solver_paths.wedges_objects
        wedges_origin = solver_paths.wedges_origin

        # Indices of the wedges corresponding to this object
        # [num_wedges]
        wedges_ind, _ = tf.unique(tf.where(wedges_objects == obj_id)[:,0])

        # Corresponding origins
        # [num_wedges, 3]
        wedges_origin = tf.gather(wedges_origin, wedges_ind, axis=0)

        # Translates the wedges
        # [num_wedges, 3]
        wedges_origin += translation_vector

        # Updates the wedges
        wedges_ind = tf.expand_dims(wedges_ind, axis=1)
        solver_paths.wedges_origin.scatter_nd_update(wedges_ind, wedges_origin)

        # Trigger scene callback
        self._scene.scene_geometry_updated()

    @property
    def orientation(self):
        r"""
        [3], tf.float : Get/set the orientation :math:`(\alpha, \beta, \gamma)`
            [rad] specified through three angles corresponding to a
            3D rotation as defined in :eq:`rotation`.
        """
        return self._orientation

    @orientation.setter
    def orientation(self, new_orient):

        # Real dtype
        new_orient = tf.cast(new_orient, self._rdtype)

        # Build the transformtation corresponding to the new rotation
        new_rotation = angles_to_mitsuba_rotation(new_orient)

        # Invert the current orientation
        cur_rotation = angles_to_mitsuba_rotation(self._orientation.numpy())
        inv_cur_rotation = cur_rotation.inverse()

        # Build the transform.
        # The object is first translated to the origin, then rotated, then
        # translated back to its current position
        transform =  (  self._mi_transform_t.translate(self.position.numpy())
                      @ new_rotation
                      @ inv_cur_rotation
                      @ self._mi_transform_t.translate(-self.position.numpy()) )

        ## Update Mitsuba vertices

        # Scene parameters
        scene_params = self._scene.mi_scene_params
        # [num_vertices*3]
        vertices = scene_params[f'mesh-{self.name}.vertex_positions']
        # [num_vertices,3]
        vertices = dr.unravel(self._mi_point_t, vertices)
        # Apply the transform
        vertices = transform.transform_affine(vertices)
        # Cast to Mitsuba type to object the Mitsuba scene
        fltn_vertices = tf.reshape(vertices, [-1])
        fltn_vertices = tf.cast(fltn_vertices, tf.float32)
        scene_params[f'mesh-{self.name}.vertex_positions'] = fltn_vertices
        scene_params.update()

        ## Update Sionna vertices

        obj_id = self.object_id
        mi_shape = self._mi_shape
        solver_paths = self._scene.solver_paths

        shape_ind = solver_paths.shape_indices[obj_id]
        prim_offset = solver_paths.prim_offsets[shape_ind]

        face_indices3 = mi_shape.face_indices(dr.arange(mi.UInt32,
                                                        mi_shape.face_count()))
        # Flatten. This is required for calling vertex_position
        # [n_prims*3]
        face_indices = dr.ravel(face_indices3)
        # Get vertices coordinates
        # [n_prims*3, 3]
        vertex_coords = mi_shape.vertex_position(face_indices)
        # Cast to TensorFlow type
        # [n_prims*3, 3]
        vertex_coords = mi_to_tf_tensor(vertex_coords, self._rdtype)
        # Unflatten
        # [n_prims, vertices per triangle : 3, 3]
        vertex_coords = tf.reshape(vertex_coords, [mi_shape.face_count(), 3, 3])
        # Update the tensor storing the primitive vertices
        sl = tf.range(prim_offset, prim_offset + mi_shape.face_count(),
                    dtype=tf.int32)
        sl = tf.expand_dims(sl, axis=1)
        solver_paths.primitives.scatter_nd_update(sl, vertex_coords)

        ## Update Sionna normals

        # Get vertices coordinates
        # [n_prims, 3]
        normals = solver_paths.normals.gather_nd(sl)
        # Cast to Mitsuba Vector
        # [n_prims, 3]
        normals = self._mi_vec_t(normals)
        # Rotate the normals
        normals = transform.transform_affine(normals)
        # Cast to Tensorflow type
        # [n_prims, 3]
        normals = mi_to_tf_tensor(normals, self._rdtype)
        # Update the tensor storing the primitive vertices
        solver_paths.normals.scatter_nd_update(sl, normals)

        ## Update Sionna wedges

        wedges_objects = solver_paths.wedges_objects
        wedges_origin = solver_paths.wedges_origin
        wedges_e_hat = solver_paths.wedges_e_hat
        wedges_normals = solver_paths.wedges_normals

        # Indices of the wedges corresponding to this object
        # [num_wedges]
        wedges_ind, _ = tf.unique(tf.where(wedges_objects == obj_id)[:,0])

        # Corresponding origins, e_hat, and normals
        # [num_wedges, 3]
        wedges_origin = tf.gather(wedges_origin, wedges_ind, axis=0)
        # [num_wedges, 3]
        wedges_e_hat = tf.gather(wedges_e_hat, wedges_ind, axis=0)
        # [num_wedges, 3]
        wedges_normals = tf.gather(wedges_normals, wedges_ind, axis=0)
        # [num_wedges*2, 3]
        wedges_normals = tf.reshape(wedges_normals, [-1, 3])

        # Cast to Mitsuba types
        # [num_wedges, 3]
        wedges_origin = self._mi_point_t(wedges_origin)
        # [num_wedges, 3]
        wedges_e_hat = self._mi_vec_t(wedges_e_hat)
        # [num_wedges*2, 3]
        wedges_normals = self._mi_vec_t(wedges_normals)

        # Rotate all quantities
        # [num_wedges, 3]
        wedges_origin = transform.transform_affine(wedges_origin)
         # [num_wedges, 3]
        wedges_e_hat = transform.transform_affine(wedges_e_hat)
         # [num_wedges*2, 3]
        wedges_normals = transform.transform_affine(wedges_normals)

        # Cast to Tensorflow type
        # [num_wedges, 3]
        wedges_origin = mi_to_tf_tensor(wedges_origin, self._rdtype)
        # [num_wedges, 3]
        wedges_e_hat = mi_to_tf_tensor(wedges_e_hat, self._rdtype)
        # [num_wedges*2, 3]
        wedges_normals = mi_to_tf_tensor(wedges_normals, self._rdtype)
        # [num_wedges, 2, 3]
        wedges_normals = tf.reshape(wedges_normals, [-1, 2, 3])

        # Updates the wedges
        wedges_ind = tf.expand_dims(wedges_ind, axis=1)
        solver_paths.wedges_origin.scatter_nd_update(wedges_ind, wedges_origin)
        solver_paths.wedges_e_hat.scatter_nd_update(wedges_ind, wedges_e_hat)
        solver_paths.wedges_normals.scatter_nd_update(wedges_ind,
                                                      wedges_normals)

        self._orientation = new_orient

        # Trigger scene callback
        self._scene.scene_geometry_updated()

    def look_at(self, target):
        # pylint: disable=line-too-long
        r"""
        Sets the orientation so that the x-axis points toward an
        ``Object``.

        Input
        -----
        target : [3], float | :class:`sionna.rt.Object` | str
            A position or the name or instance of an
            :class:`sionna.rt.Object` in the scene to point toward to
        """
        # Get position to look at
        if isinstance(target, str):
            obj = self.scene.get(target)
            if not isinstance(obj, Object):
                raise ValueError(f"No camera, device, or object named '{target}' found.")
            else:
                target = obj.position
        elif isinstance(target, Object):
            target = target.position
        else:
            target = tf.cast(target, dtype=self._rdtype)
            if not target.shape[0]==3:
                raise ValueError("`target` must be a three-element vector)")

        # Compute angles relative to LCS
        x = target - self.position
        x, _ = normalize(x)
        theta, phi = theta_phi_from_unit_vec(x)
        alpha = phi # Rotation around z-axis
        beta = theta-PI/2 # Rotation around y-axis
        gamma = 0.0 # Rotation around x-axis
        self.orientation = (alpha, beta, gamma)
