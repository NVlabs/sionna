#
# SPDX-FileCopyrightText: Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""
Ray tracer utilities
"""

import tensorflow as tf

from sionna.utils import expand_to_rank


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

def rotate(p, angles):
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
    rot_p = tf.linalg.matvec(rot_mat, p)

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

    # Clip to ensure numerical stability
    z = tf.clip_by_value(z, -1., 1.)
    theta = tf.acos(z)
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

def dot(u, v, keepdim=False):
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

    Output
    -------
    : [...,1] or [...]
        Dot product between ``u`` and ``v``.
        The last dimension is removed if ``keepdim``
        is set to `False`.
    """
    return tf.reduce_sum(u*v, axis=-1, keepdims=keepdim)

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
        Norm of the normalized vector
    """
    norm = tf.norm(v, axis=-1, keepdims=True)
    n_v = tf.math.divide_no_nan(v, norm)
    norm = tf.squeeze(norm, axis=-1)
    return n_v, norm

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

    mask = paths.mask.numpy()
    vertices = paths.vertices.numpy()
    objects = paths.objects.numpy()
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
    tx_positions, rx_positions = {}, {}
    devices = ( (scene.transmitters, tx_positions),
                (scene.receivers, rx_positions))
    for source, destination in devices:
        for k, rd in source.items():
            p = rd.position.numpy()
            bbox.expand(p)
            destination[k] = p

    sc = 2. * bbox.bounding_sphere().radius
    return sc, tx_positions, rx_positions, bbox
