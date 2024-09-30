#
# SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Utility functions used for testing of the sionna.rt module"""

try:
    import sionna
except ImportError as e:
    import sys
    sys.path.append("../..")
    import sionna

from sionna import config

import numpy as np


def normalize(v, return_norm=False):
    """Normalizes a vector and optionally returns its norm"""
    norm = np.sqrt(np.sum(np.abs(v)**2))
    if return_norm:
        return v/norm, norm
    else:
        return v/norm

def theat_phi_from_unit_vec(v):
    """Computes spherical angles theta and phi of a unit vector"""
    theta = np.arccos(v[2])
    phi = np.arctan2(v[1], v[0])
    return theta, phi

def cross(u, v):
    """Cross product between two vectors"""
    u_x = u[0]
    u_y = u[1]
    u_z = u[2]
    v_x = v[0]
    v_y = v[1]
    v_z = v[2]
    r_x = u_y*v_z - u_z*v_y
    r_y = u_z*v_x - u_x*v_z
    r_z = u_x*v_y - u_y*v_x
    return np.array([r_x, r_y, r_z])

def dot(u, v):
    """Dot product between two vectors"""
    return np.sum(u*v)

def r_hat(theta, phi):
    """Computes spherical unit vector r_hat from spherical angles"""
    x = np.sin(theta)*np.cos(phi)
    y = np.sin(theta)*np.sin(phi)
    z = np.cos(theta)
    return np.array([x,y,z])

def theta_hat(theta, phi):
    """Computes spherical unit vector theta_hat from spherical angles"""
    x = np.cos(theta)*np.cos(phi)
    y = np.cos(theta)*np.sin(phi)
    z = -np.sin(theta)
    return np.array([x,y,z])

def phi_hat(theta, phi):
    """Computes spherical unit vector phi_hat from spherical angles"""
    x = -np.sin(phi)
    y = np.cos(phi)
    z = 0.0
    return np.array([x,y,z])

def reflection_coefficient(eta, cos_theta):
    """Computes simplified reflection coeffient from cosine angle and complex permittivity"""
    a = cos_theta
    b = np.sqrt(eta-1.+cos_theta**2)
    r_te = (a-b)/(a+b)
    c = eta*a
    d = b
    r_tm =(c-d)/(c+d)
    return r_te, r_tm

def projection_matrix(e_s, e_p, e_i_s, e_i_p):
    """Projection matrix from basis (e_s, e_p) to (e_i_s, e_i_p)"""
    r_1_1 = dot(e_i_s, e_s)
    r_1_2 = dot(e_i_s, e_p)
    r_2_1 = dot(e_i_p, e_s)
    r_2_2 = dot(e_i_p, e_p)
    return np.array([[r_1_1, r_1_2], [r_2_1, r_2_2]], complex)

def compute_field_component_vectors(k_i, k_r, n_hat):
    """Computes the basis vectors of the incoming and reflected field components"""
    # Basis vectors in the incoming plane
    e_i_s = cross(k_i, n_hat)
    if np.sum(np.abs(e_i_s))<1e-4: # In case of normal incidence e_i_s is not uniquely defined, so we select a random orientation
        e_i_s = cross(n_hat+config.np_rng.normal([3]), n_hat)
    e_i_s = normalize(e_i_s)
    e_i_p = normalize(cross(e_i_s, k_i))

    # Basis vectors of the reflected field
    e_r_s = e_i_s
    e_r_p = normalize(cross(e_r_s, k_r))

    return e_i_s, e_i_p, e_r_s, e_r_p

def reflection(k_i, k_r, n_hat, eta, e_s, e_p):
    """Computes the transfer matrix of a reflection and basis vectors of the refleceted field components"""

    # Compute unit vectors representing the incoming.outgoing field
    e_i_s, e_i_p, e_r_s, e_r_p = compute_field_component_vectors(k_i, k_r, n_hat)

    # Compute matrix for base transformation
    d = projection_matrix(e_s, e_p, e_i_s, e_i_p)

    # Compute reflection coefficients
    cos_theta = dot(-k_i, n_hat)
    r_te, r_tm = reflection_coefficient(eta, cos_theta)
    t = np.array([[r_te, 0], [0, r_tm]], complex)

    # Reflection transfer matrix
    mat_t = np.matmul(t, d)

    return mat_t, e_r_s, e_r_p

def validate_path(path_ind, paths, scene):
    """Computes the transfer matrix mat_t of a specific path"""
    normals = paths.spec_tmp.normals[:,0,0,path_ind,:]
    objects = paths.objects[:,0,0, path_ind]
    etas = []
    for i in objects:
        if i==-1:
            break
        for obj in list(scene.objects.values()):
            if obj.object_id == i:
                etas.append(obj.radio_material.complex_relative_permittivity.numpy())
                break

    num_bounces = len(etas)
    num_paths = paths.spec_tmp.normals.shape[2]
    vertices = paths.vertices[:num_bounces,0,0,path_ind,:].numpy()
    theta_t = np.squeeze(paths.theta_t, axis=(0,1,2))[path_ind]
    phi_t = np.squeeze(paths.phi_t, axis=(0,1,2))[path_ind]
    theta_r = np.squeeze(paths.theta_r, axis=(0,1,2))[path_ind]
    phi_r = np.squeeze(paths.phi_r, axis=(0,1,2))[path_ind]

    e_s = theta_hat(theta_t, phi_t)
    e_p = phi_hat(theta_t, phi_t)

    tx_pos = scene.transmitters["tx"].position
    rx_pos = scene.receivers["rx"].position

    start_point = tx_pos
    mat_t = np.eye(2, dtype=complex)
    if num_bounces>0:
        # Not LoS
        for i in range(num_bounces):
            hit_point = vertices[i]
            n_hat = normals[i]
            eta = etas[i]
            if i==num_bounces-1:
                end_point = rx_pos
            else:
                end_point = vertices[i+1]

            # Compute incoming and reflected wave vectors
            k_i = normalize(hit_point - start_point)
            k_r = normalize(end_point - hit_point)

            # Compute transfer matrix of reflection
            t, e_s, e_p = reflection(k_i, k_r, n_hat, eta, e_s, e_p)
            mat_t = np.matmul(t, mat_t)


            # Set new start point
            start_point = hit_point

        # Compute total distance
        d = 0
        last_point = tx_pos
        for v in vertices:
            d += np.sqrt(np.sum(np.abs(v-last_point)**2))
            last_point = v
        d += np.sqrt(np.sum(np.abs(rx_pos-last_point)**2))
    else:
        # LoS
        d = np.sqrt(np.sum(np.abs(rx_pos-tx_pos)**2))

    mat_t /= d

    mat_d = projection_matrix(e_s,
                      e_p,
                      theta_hat(theta_r, phi_r),
                      phi_hat(theta_r, phi_r))

    mat_t = np.matmul(mat_d, mat_t)

    # Return reference transfer matrix from ray tracer
    mat_t_ref = np.squeeze(paths.spec_tmp.mat_t, axis=(0,1))[path_ind]
    if num_paths>1:
        mat_t_ref = mat_t_ref[path_ind]

    return mat_t, mat_t_ref
