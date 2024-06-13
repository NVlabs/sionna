#
# SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""
Ray tracer utilities
"""

import tensorflow as tf
import mitsuba as mi
import drjit as dr

from sionna.utils import expand_to_rank
from sionna import PI

def rotation_matrix(angles):
    r"""
    Computes rotation matrices as defined in :eq:`rotation`

    The closed-form expression in (7.1-4) [TR38901]_ is used.

    Input
    ------
    angles : [...,3], tf.float
        Angles for the rotations [rad].
        The last dimension corresponds to the angles
        :math:`(\alpha,\beta,\gamma)` that define
        rotations about the axes :math:`(z, y, x)`,
        respectively.

    Output
    -------
    : [...,3,3], tf.float
        Rotation matrices
    """

    a = angles[...,0]
    b = angles[...,1]
    c = angles[...,2]
    cos_a = tf.cos(a)
    cos_b = tf.cos(b)
    cos_c = tf.cos(c)
    sin_a = tf.sin(a)
    sin_b = tf.sin(b)
    sin_c = tf.sin(c)

    r_11 = cos_a*cos_b
    r_12 = cos_a*sin_b*sin_c - sin_a*cos_c
    r_13 = cos_a*sin_b*cos_c + sin_a*sin_c
    r_1 = tf.stack([r_11, r_12, r_13], axis=-1)

    r_21 = sin_a*cos_b
    r_22 = sin_a*sin_b*sin_c + cos_a*cos_c
    r_23 = sin_a*sin_b*cos_c - cos_a*sin_c
    r_2 = tf.stack([r_21, r_22, r_23], axis=-1)

    r_31 = -sin_b
    r_32 = cos_b*sin_c
    r_33 = cos_b*cos_c
    r_3 = tf.stack([r_31, r_32, r_33], axis=-1)

    rot_mat = tf.stack([r_1, r_2, r_3], axis=-2)
    return rot_mat

def rotate(p, angles, inverse=False):
    r"""
    Rotates points ``p`` by the ``angles`` according
    to the 3D rotation defined in :eq:`rotation`

    Input
    -----
    p : [...,3], tf.float
        Points to rotate

    angles : [..., 3]
        Angles for the rotations [rad].
        The last dimension corresponds to the angles
        :math:`(\alpha,\beta,\gamma)` that define
        rotations about the axes :math:`(z, y, x)`,
        respectively.

    inverse : bool
        If `True`, the inverse rotation is applied,
        i.e., the transpose of the rotation matrix is used.
        Defaults to `False`

    Output
    ------
    : [...,3]
        Rotated points ``p``
    """

    # Rotation matrix
    # [..., 3, 3]
    rot_mat = rotation_matrix(angles)
    rot_mat = expand_to_rank(rot_mat, tf.rank(p)+1, 0)

    # Rotation around ``center``
    # [..., 3]
    rot_p = tf.linalg.matvec(rot_mat, p, transpose_a=inverse)

    return rot_p

def theta_phi_from_unit_vec(v):
    r"""
    Computes zenith and azimuth angles (:math:`\theta,\varphi`)
    from unit-norm vectors as described in :eq:`theta_phi`

    Input
    ------
    v : [...,3], tf.float
        Tensor with unit-norm vectors in the last dimension

    Output
    -------
    theta : [...], tf.float
        Zenith angles :math:`\theta`

    phi : [...], tf.float
        Azimuth angles :math:`\varphi`
    """
    x = v[...,0]
    y = v[...,1]
    z = v[...,2]

    # If v = z, then x = 0 and y = 0. In this case, atan2 is not differentiable,
    # leading to NaN when computing the gradients.
    # The following lines force x to one this case. Note that this does not
    # impact the output meaningfully, as in that case theta = 0 and phi can
    # take any value.
    zero = tf.zeros_like(x)
    is_unit_z = tf.logical_and(tf.equal(x, zero), tf.equal(y, zero))
    is_unit_z = tf.cast(is_unit_z, x.dtype)
    x += is_unit_z

    theta = acos_diff(z)
    phi = tf.math.atan2(y, x)
    return theta, phi

def r_hat(theta, phi):
    r"""
    Computes the spherical unit vetor :math:`\hat{\mathbf{r}}(\theta, \phi)`
    as defined in :eq:`spherical_vecs`

    Input
    -------
    theta : arbitrary shape, tf.float
        Zenith angles :math:`\theta` [rad]

    phi : same shape as ``theta``, tf.float
        Azimuth angles :math:`\varphi` [rad]

    Output
    --------
    rho_hat : ``phi.shape`` + [3], tf.float
        Vector :math:`\hat{\mathbf{r}}(\theta, \phi)`  on unit sphere
    """
    rho_hat = tf.stack([tf.sin(theta)*tf.cos(phi),
                        tf.sin(theta)*tf.sin(phi),
                        tf.cos(theta)], axis=-1)
    return rho_hat

def theta_hat(theta, phi):
    r"""
    Computes the spherical unit vector
    :math:`\hat{\boldsymbol{\theta}}(\theta, \varphi)`
    as defined in :eq:`spherical_vecs`

    Input
    -------
    theta : arbitrary shape, tf.float
        Zenith angles :math:`\theta` [rad]

    phi : same shape as ``theta``, tf.float
        Azimuth angles :math:`\varphi` [rad]

    Output
    --------
    theta_hat : ``phi.shape`` + [3], tf.float
        Vector :math:`\hat{\boldsymbol{\theta}}(\theta, \varphi)`
    """
    x = tf.cos(theta)*tf.cos(phi)
    y = tf.cos(theta)*tf.sin(phi)
    z = -tf.sin(theta)
    return tf.stack([x,y,z], -1)

def phi_hat(phi):
    r"""
    Computes the spherical unit vector
    :math:`\hat{\boldsymbol{\varphi}}(\theta, \varphi)`
    as defined in :eq:`spherical_vecs`

    Input
    -------
    phi : same shape as ``theta``, tf.float
        Azimuth angles :math:`\varphi` [rad]

    Output
    --------
    theta_hat : ``phi.shape`` + [3], tf.float
        Vector :math:`\hat{\boldsymbol{\varphi}}(\theta, \varphi)`
    """
    x = -tf.sin(phi)
    y = tf.cos(phi)
    z = tf.zeros_like(x)
    return tf.stack([x,y,z], -1)

def cross(u, v):
    r"""
    Computes the cross (or vector) product between u and v

    Input
    ------
    u : [...,3]
        First vector

    v : [...,3]
        Second vector

    Output
    -------
    : [...,3]
        Cross product between ``u`` and ``v``
    """
    u_x = u[...,0]
    u_y = u[...,1]
    u_z = u[...,2]

    v_x = v[...,0]
    v_y = v[...,1]
    v_z = v[...,2]

    w = tf.stack([u_y*v_z - u_z*v_y,
                  u_z*v_x - u_x*v_z,
                  u_x*v_y - u_y*v_x], axis=-1)

    return w

def dot(u, v, keepdim=False, clip=False):
    r"""
    Computes and the dot (or scalar) product between u and v

    Input
    ------
    u : [...,3]
        First vector

    v : [...,3]
        Second vector

    keepdim : bool
        If `True`, keep the last dimension.
        Defaults to `False`.

    clip : bool
        If `True`, clip output to [-1,1].
        Defaults to `False`.

    Output
    -------
    : [...,1] or [...]
        Dot product between ``u`` and ``v``.
        The last dimension is removed if ``keepdim``
        is set to `False`.
    """
    res = tf.reduce_sum(u*v, axis=-1, keepdims=keepdim)
    if clip:
        one = tf.ones((), u.dtype)
        res = tf.clip_by_value(res, -one, one)
    return res

def outer(u,v):
    r"""
    Computes the outer product between u and v

    Input
    ------
    u : [...,3]
        First vector

    v : [...,3]
        Second vector

    Output
    -------
    : [...,3,3]
        Outer product between ``u`` and ``v``
    """
    return u[...,tf.newaxis] * v[...,tf.newaxis,:]

def normalize(v):
    r"""
    Normalizes ``v`` to unit norm

    Input
    ------
    v : [...,3], tf.float
        Vector

    Output
    -------
    : [...,3], tf.float
        Normalized vector

    : [...], tf.float
        Norm of the unnormalized vector
    """
    norm = tf.norm(v, axis=-1, keepdims=True)
    n_v = tf.math.divide_no_nan(v, norm)
    norm = tf.squeeze(norm, axis=-1)
    return n_v, norm

def moller_trumbore(o, d, p0, p1, p2, epsilon):
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

    epsilon : (), tf.float
        Small value used to avoid errors due to numerical precision

    Output
    -------
    t : [...], tf.float
        Position along the ray from the origin at which the intersection
        occurs (if any)

    hit : [...], bool
        `True` if the ray intersects the triangle. `False` otherwise.
    """

    rdtype = o.dtype
    zero = tf.cast(0.0, rdtype)
    one = tf.ones((), rdtype)

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
        tf.logical_and(tf.greater_equal(u, -epsilon),
                       tf.less_equal(u, one + epsilon)))

    # [..., 3]
    qvec = cross(tvec, e1)
    # [...,1]
    v = tf.math.divide_no_nan(dot(d, qvec, keepdim=True), det)
    # [..., 1]
    hit = tf.logical_and(hit,
                            tf.logical_and(tf.greater_equal(v, -epsilon),
                                        tf.less_equal(u + v, one + epsilon)))
    # [..., 1]
    t = tf.math.divide_no_nan(dot(e2, qvec, keepdim=True), det)
    # [..., 1]
    hit = tf.logical_and(hit, tf.greater_equal(t, epsilon))

    # [...]
    t = tf.squeeze(t, axis=-1)
    hit = tf.squeeze(hit, axis=-1)

    return t, hit

def component_transform(e_s, e_p, e_i_s, e_i_p):
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
    return r

def mi_to_tf_tensor(mi_tensor, dtype):
    """
    Get a TensorFlow eager tensor from a Mitsuba/DrJIT tensor
    """
    dr.eval(mi_tensor)
    dr.sync_thread()
    # When there is only one input, the .tf() methods crashes.
    # The following hack takes care of this corner case
    if dr.shape(mi_tensor)[-1] == 1:
        mi_tensor = dr.repeat(mi_tensor, 2)
        tf_tensor = tf.cast(mi_tensor.tf(), dtype)[:1]
    else:
        tf_tensor = tf.cast(mi_tensor.tf(), dtype)
    return tf_tensor

def gen_orthogonal_vector(k, epsilon):
    """
    Generate an arbitrary vector that is orthogonal to ``k``.

    Input
    ------
    k : [..., 3], tf.float
        Vector

    epsilon : (), tf.float
        Small value used to avoid errors due to numerical precision

    Output
    -------
    : [..., 3], tf.float
        Vector orthogonal to ``k``
    """
    rdtype = k.dtype
    ex = tf.cast([1.0, 0.0, 0.0], rdtype)
    ex = expand_to_rank(ex, tf.rank(k), 0)

    ey = tf.cast([0.0, 1.0, 0.0], rdtype)
    ey = expand_to_rank(ey, tf.rank(k), 0)

    n1 = cross(k, ex)
    n1_norm = tf.norm(n1, axis=-1, keepdims=True)
    n2 = cross(k, ey)
    return tf.where(tf.greater(n1_norm, epsilon), n1, n2)

def compute_field_unit_vectors(k_i, k_r, n, epsilon, return_e_r=True):
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

    epsilon : (), tf.float
        Small value used to avoid errors due to numerical precision

    return_e_r : bool
        If `False`, only ``e_i_s`` and ``e_i_p`` are returned.

    Output
    ------
    e_i_s : [..., 3], tf.float
        Incident unit field vector for S polarization

    e_i_p : [..., 3], tf.float
        Incident unit field vector for P polarization

    e_r_s : [..., 3], tf.float
        Reflection unit field vector for S polarization.
        Only returned if ``return_e_r`` is `True`.

    e_r_p : [..., 3], tf.float
        Reflection unit field vector for P polarization
        Only returned if ``return_e_r`` is `True`.
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
    e_i_s = tf.where(tf.greater(e_i_s_norm, epsilon), e_i_s,
                     gen_orthogonal_vector(n, epsilon))

    e_i_s,_ = normalize(e_i_s)
    e_i_p,_ = normalize(cross(e_i_s, k_i))
    if not return_e_r:
        return e_i_s, e_i_p
    else:
        e_r_s = e_i_s
        e_r_p,_ = normalize(cross(e_r_s, k_r))
        return e_i_s, e_i_p, e_r_s, e_r_p

def reflection_coefficient(eta, cos_theta):
    """
    Compute simplified reflection coefficients

    Input
    ------
    eta : Any shape, tf.complex
        Complex relative permittivity

    cos_theta : Same as ``eta``, tf.float
        Cosine of the incident angle

    Output
    -------
    r_te : Same as input, tf.complex
        Fresnel reflection coefficient for S direction

    r_tm : Same as input, tf.complex
        Fresnel reflection coefficient for P direction
    """
    cos_theta = tf.complex(cos_theta, tf.zeros_like(cos_theta))

    # Fresnel equations
    a = cos_theta
    b = tf.sqrt(eta-1.+cos_theta**2)
    r_te = tf.math.divide_no_nan(a-b, a+b)

    c = eta*a
    d = b
    r_tm = tf.math.divide_no_nan(c-d, c+d)
    return r_te, r_tm

def paths_to_segments(paths):
    """
    Extract the segments corresponding to a set of ``paths``

    Input
    -----
    paths : :class:`~sionna.rt.Paths`
        A set of paths

    Output
    -------
    starts, ends : [n,3], float
        Endpoints of the segments making the paths.
    """

    vertices = paths.vertices.numpy()
    objects = paths.objects.numpy()
    mask = paths.targets_sources_mask
    sources, targets = paths.sources.numpy(), paths.targets.numpy()

    # Emit directly two lists of the beginnings and endings of line segments
    starts = []
    ends = []
    for rx in range(vertices.shape[1]): # For each receiver
        for tx in range(vertices.shape[2]): # For each transmitter
            for p in range(vertices.shape[3]): # For each path depth
                if not mask[rx, tx, p]:
                    continue

                start = sources[tx]
                i = 0
                while ( (i < objects.shape[0])
                    and (objects[i, rx, tx, p] != -1) ):
                    end = vertices[i, rx, tx, p]
                    starts.append(start)
                    ends.append(end)
                    start = end
                    i += 1
                # Explicitly add the path endpoint
                starts.append(start)
                ends.append(targets[rx])
    return starts, ends

def scene_scale(scene):
    bbox = scene.mi_scene.bbox()
    tx_positions, rx_positions, ris_positions = {}, {}, {}
    devices = ((scene.transmitters, tx_positions),
               (scene.receivers, rx_positions),
               (scene.ris, ris_positions)
              )
    for source, destination in devices:
        for k, rd in source.items():
            p = rd.position.numpy()
            bbox.expand(p)
            destination[k] = p

    sc = 2. * bbox.bounding_sphere().radius
    return sc, tx_positions, rx_positions, ris_positions, bbox

def fibonacci_lattice(num_points, dtype=tf.float32):
    """
    Generates a Fibonacci lattice for the unit square

    Input
    -----
    num_points : int
        Number of points

    type : tf.DType
        Datatype to use for the output

    Output
    -------
    points : [num_points, 2]
        Generated rectangular coordinates of the lattice points
    """

    golden_ratio = (1.+tf.sqrt(tf.cast(5., tf.float64)))/2.
    ns = tf.range(0, num_points, dtype=tf.float64)

    x = ns/golden_ratio
    x = x - tf.floor(x)
    y = ns/(num_points-1)
    points = tf.stack([x,y], axis=1)

    points = tf.cast(points, dtype)

    return points

def cot(x):
    """
    Cotangens function

    Input
    ------
    x : [...], tf.float

    Output
    -------
    : [...], tf.float
        Cotangent of x
    """
    return tf.math.divide_no_nan(tf.ones_like(x), tf.math.tan(x))

def sign(x):
    """
    Returns +1 if ``x`` is non-negative, -1 otherwise

    Input
    ------
    x : [...], tf.float
        A real-valued number

    Output
    -------
    : [...], tf.float
        +1 if ``x`` is non-negative, -1 otherwise
    """
    two = tf.cast(2, x.dtype)
    one = tf.cast(1, x.dtype)
    return two*tf.cast(tf.greater_equal(x, 0), x.dtype)-one

def rot_mat_from_unit_vecs(a, b):
    r"""
    Computes Rodrigues` rotation formula :eq:`rodrigues_matrix`

    Input
    ------
    a : [...,3], tf.float
        First unit vector

    b : [...,3], tf.float
        Second unit vector

    Output
    -------
    : [...,3,3], tf.float
        Rodrigues' rotation matrix
    """

    rdtype = a.dtype

    # Compute rotation axis vector
    k, _ = normalize(cross(a, b))

    # Deal with special case where a and b are parallel
    o = gen_orthogonal_vector(a, 1e-6)
    k = tf.where(tf.reduce_sum(tf.abs(k), axis=-1, keepdims=True)==0, o, k)

    # Compute K matrix
    shape = tf.concat([tf.shape(k)[:-1],[1]], axis=-1)
    zeros = tf.zeros(shape, rdtype)
    kx, ky, kz = tf.split(k, 3, axis=-1)
    l1 = tf.concat([zeros, -kz, ky], axis=-1)
    l2 = tf.concat([kz, zeros, -kx], axis=-1)
    l3 = tf.concat([-ky, kx, zeros], axis=-1)
    k_mat = tf.stack([l1, l2, l3], axis=-2)

    # Assemble full rotation matrix
    eye = tf.eye(3, batch_shape=tf.shape(k)[:-1], dtype=rdtype)
    cos_theta = dot(a, b, clip=True)
    sin_theta = tf.sin(acos_diff(cos_theta))
    cos_theta = expand_to_rank(cos_theta, tf.rank(eye), axis=-1)
    sin_theta = expand_to_rank(sin_theta, tf.rank(eye), axis=-1)
    rot_mat = eye + k_mat*sin_theta + \
                      tf.linalg.matmul(k_mat, k_mat) * (1-cos_theta)
    return rot_mat

def sample_points_on_hemisphere(normals, num_samples=1):
    # pylint: disable=line-too-long
    r"""
    Randomly sample points on hemispheres defined by their normal vectors

    Input
    -----
    normals : [batch_size, 3], tf.float
        Normal vectors defining hemispheres

    num_samples : int
        Number of random samples to draw for each hemisphere
        defined by its normal vector.
        Defaults to 1.

    Output
    ------
    points : [batch_size, num_samples, 3], tf.float or [batch_size, 3], tf.float if num_samples=1.
        Random points on the hemispheres
    """
    dtype = normals.dtype
    batch_size = tf.shape(normals)[0]
    shape = [batch_size, num_samples]

    # Sample phi uniformly distributed on [0,2*PI]
    phi = tf.random.uniform(shape, maxval=2*PI, dtype=dtype)

    # Generate samples of theta for uniform distribution on the hemisphere
    u = tf.random.uniform(shape, maxval=1, dtype=dtype)
    theta = tf.acos(u)

    # Transform spherical to Cartesian coordinates
    points = r_hat(theta, phi)

    # Compute rotation matrices
    z_hat = tf.constant([[0,0,1]], dtype=dtype)
    z_hat = tf.broadcast_to(z_hat, tf.shape(normals))
    rot_mat = rot_mat_from_unit_vecs(z_hat, normals)
    rot_mat = tf.expand_dims(rot_mat, axis=1)

    # Compute rotated points
    points = tf.linalg.matvec(rot_mat, points)

    # Numerical errors can cause sampling from the other hemisphere.
    # Correct the sampled vector to avoid sampling in the wrong hemisphere.
    normals = tf.expand_dims(normals, axis=1)
    s = dot(points, normals, keepdim=True)
    s = tf.where(s < 0., s, 0.)
    points = points - 2.*s*normals

    if num_samples==1:
        points = tf.squeeze(points, axis=1)

    return points

def acos_diff(x, epsilon=1e-7):
    r"""
    Implementation of arccos(x) that avoids evaluating the gradient at x
    -1 or 1 by using straight through estimation, i.e., in the
    forward pass, x is clipped to (-1, 1), but in the backward pass, x is
    clipped to (-1 + epsilon, 1 - epsilon).

    Input
    ------
    x : any shape, tf.float
        Value at which to evaluate arccos

    epsilon : tf.float
        Small backoff to avoid evaluating the gradient at -1 or 1.
        Defaults to 1e-7.

    Output
    -------
     : same shape as x, tf.float
        arccos(x)
    """

    x_clip_1 = tf.clip_by_value(x, -1., 1.)
    x_clip_2 = tf.clip_by_value(x, -1. + epsilon, 1. - epsilon)
    eps = tf.stop_gradient(x - x_clip_2)
    x_1 =  x - eps
    acos_x_1 =  tf.acos(x_1)
    y = acos_x_1 + tf.stop_gradient(tf.acos(x_clip_1)-acos_x_1)
    return y

def angles_to_mitsuba_rotation(angles):
    """
    Build a Mitsuba transform from angles in radian

    Input
    ------
    angles : [3], tf.float
        Angles [rad]

    Output
    -------
    : :class:`mitsuba.ScalarTransform4f`
        Mitsuba rotation
    """

    angles = 180. * angles / PI

    if angles.dtype == tf.float32:
        mi_transform_t = mi.Transform4f
        angles = mi.Float(angles)
    else:
        mi_transform_t = mi.Transform4d
        angles = mi.Float64(angles)

    return (
          mi_transform_t.rotate(axis=[0., 0., 1.], angle=angles[0])
        @ mi_transform_t.rotate(axis=[0., 1., 0.], angle=angles[1])
        @ mi_transform_t.rotate(axis=[1., 0., 0.], angle=angles[2])
    )

def gen_basis_from_z(z, epsilon):
    """
    Generate a pair of vectors (x,y) such that (x,y,z) is an orthonormal basis.

    Input
    ------
    z : [..., 3], tf.float
        Unit vector

    epsilon : (), tf.float
        Small value used to avoid errors due to numerical precision

    Output
    -------
    x : [..., 3], tf.float
        Unit vector

    y : [..., 3], tf.float
        Unit vector
    """
    x = gen_orthogonal_vector(z, epsilon)
    x,_ = normalize(x)
    y = cross(z, x)
    return x,y

def compute_spreading_factor(rho_1, rho_2, s):
    r"""
    Computes the spreading factor
    :math:`\sqrt{\frac{\rho_1 \rho_2}{(\rho_1 + s)(\rho_2 + s)}}`

    Input
    ------
    rho_1, rho_2 : [...], tf.float
        Principal radii of curvature

    s : [...], tf.float
        Position along the axial ray at which to evaluate the squared
        spreading factor

    Output
    -------
    : float
        Squared spreading factor
    """

    # In the case of a spherical wave, when the origin (s = 0) is set to unique
    # caustic point, then both principal radii of curvature are set to zero.
    # The spreading factor is then equal to 1/s.
    spherical = tf.logical_and(tf.equal(rho_1, 0.), tf.equal(rho_2, 0.))
    a2_spherical = tf.math.reciprocal_no_nan(s)

    # General formula for the spreading factor
    a2 = tf.sqrt(rho_1*rho_2/((rho_1+s)*(rho_2+s)))

    a2 = tf.where(spherical, a2_spherical, a2)
    return a2

def mitsuba_rectangle_to_world(center, orientation, size, ris=False):
    """
    Build the `to_world` transformation that maps a default Mitsuba rectangle
    to the rectangle that defines the coverage map surface.

    Input
    ------
    center : [3], tf.float
        Center of the rectangle

    orientation : [3], tf.float
        Orientation of the rectangle.
        An orientation of `(0,0,0)` correspond to a rectangle oriented such that
        z+ is its normal.

    size : [2], tf.float
        Scale of the rectangle.
        The width of the rectangle (in the local X direction) is scale[0]
        and its height (in the local Y direction) scale[1].

    Output
    -------
    to_world : :class:`mitsuba.ScalarTransform4f`
        Rectangle to world transformation.
    """
    orientation = 180. * orientation / PI

    trans = \
        mi.ScalarTransform4f.translate(center.numpy())\
        @ mi.ScalarTransform4f.rotate(axis=[0, 0, 1], angle=orientation[0])\
        @ mi.ScalarTransform4f.rotate(axis=[0, 1, 0], angle=orientation[1])\
        @ mi.ScalarTransform4f.rotate(axis=[1, 0, 0], angle=orientation[2])

    if ris:
        # The RIS normal points at [1,0,0].
        # We hence rotate the normal of the rectangle which points
        # at [0,0,1] by 90 degrees around the [0,1,0] axis.
        trans = trans\
            @mi.ScalarTransform4f.rotate(axis=[0, 1, 0], angle=90)

        # size = [width (=y), height (=z)]
        # Since the RIS is rotated w.r.t to rectangle,
        # The z axis corresponds to the x axis
        size = [size[1], size[0]]

    return (trans
            @mi.ScalarTransform4f.scale([0.5 * size[0], 0.5 * size[1], 1])
    )
