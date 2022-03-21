#
# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""
Class for sampling channel impulse responses following 3GPP TR38.901
specifications and giving LSPs and rays.
"""

import tensorflow as tf
from tensorflow import sin, cos, acos

from sionna import PI, SPEED_OF_LIGHT

class Topology:
    # pylint: disable=line-too-long
    r"""
    Class for conveniently storing the network topology information required
    for sampling channel impulse responses

    Parameters
    -----------

    velocities : [batch size, number of UTs], tf.float
        UT velocities

    moving_end : str
        Indicated which end of the channel (TX or RX) is moving. Either "tx" or
        "rx".

    los_aoa : [batch size, number of BSs, number of UTs], tf.float
        Azimuth angle of arrival of LoS path [radian]

    los_aod : [batch size, number of BSs, number of UTs], tf.float
        Azimuth angle of departure of LoS path [radian]

    los_zoa : [batch size, number of BSs, number of UTs], tf.float
        Zenith angle of arrival for of path [radian]

    los_zod : [batch size, number of BSs, number of UTs], tf.float
        Zenith angle of departure for of path [radian]

    los : [batch size, number of BSs, number of UTs], tf.bool
        Indicate for each BS-UT link if it is in LoS

    distance_3d : [batch size, number of UTs, number of UTs], tf.float
        Distance between the UTs in X-Y-Z space (not only X-Y plan).

    tx_orientations : [batch size, number of TXs, 3], tf.float
        Orientations of the transmitters, which are either BSs or UTs depending
        on the link direction [radian].

    rx_orientations : [batch size, number of RXs, 3], tf.float
        Orientations of the receivers, which are either BSs or UTs depending on
        the link direction [radian].
    """

    def __init__(self,  velocities,
                        moving_end,
                        los_aoa,
                        los_aod,
                        los_zoa,
                        los_zod,
                        los,
                        distance_3d,
                        tx_orientations,
                        rx_orientations):
        self.velocities = velocities
        self.moving_end = moving_end
        self.los_aoa = los_aoa
        self.los_aod = los_aod
        self.los_zoa = los_zoa
        self.los_zod = los_zod
        self.los = los
        self.tx_orientations = tx_orientations
        self.rx_orientations = rx_orientations
        self.distance_3d = distance_3d

class ChannelCoefficientsGenerator:
    # pylint: disable=line-too-long
    r"""
    Sample channel impulse responses according to LSPs rays.

    This class implements steps 10 and 11 from the TR 38.901 specifications,
    (section 7.5).

    Parameters
    ----------
    carrier_frequency : float
        Carrier frequency [Hz]

    tx_array : PanelArray
        Panel array used by the transmitters.
        All transmitters share the same antenna array configuration.

    rx_array : PanalArray
        Panel array used by the receivers.
        All transmitters share the same antenna array configuration.

    subclustering : bool
        Use subclustering if set to `True` (see step 11 for section 7.5 in
        TR 38.901). CDL does not use subclustering. System level models (UMa,
        UMi, RMa) do.

    dtype : Complex tf.DType
        Defines the datatype for internal calculations and the output
        dtype. Defaults to `tf.complex64`.

    Input
    -----
    num_time_samples : int
        Number of samples

    sampling_frequency : float
        Sampling frequency [Hz]

    k_factor : [batch_size, number of TX, number of RX]
        K-factor

    rays : Rays
        Rays from which to compute thr CIR

    topology : Topology
        Topology of the network

    c_ds : [batch size, number of TX, number of RX]
        Cluster DS [ns]. Only needed when subclustering is used
        (``subclustering`` set to `True`), i.e., with system level models.
        Otherwise can be set to None.
        Defaults to None.

    debug : bool
        If set to `True`, additional information is returned in addition to
        paths coefficients and delays: The random phase shifts (see step 10 of
        section 7.5 in TR38.901 specification), and the time steps at which the
        channel is sampled.

    Output
    ------
    h : [batch size, num TX, num RX, num paths, num RX antenna, num TX antenna, num samples], tf.complex
        Paths coefficients

    delays : [batch size, num TX, num RX, num paths], tf.real
        Paths delays [s]

    phi : [batch size, number of BSs, number of UTs, 4], tf.real
        Initial phases (see step 10 of section 7.5 in TR 38.901 specification).
        Last dimension corresponds to the four polarization combinations.

    sample_times : [number of time steps], tf.float
        Sampling time steps
    """

    def __init__(self,  carrier_frequency,
                        tx_array, rx_array,
                        subclustering,
                        dtype=tf.complex64):
        assert dtype.is_complex, "dtype must be a complex datatype"
        self._dtype = dtype

        # Wavelength (m)
        self._lambda_0 = tf.constant(SPEED_OF_LIGHT/carrier_frequency,
            dtype.real_dtype)
        self._tx_array = tx_array
        self._rx_array = rx_array
        self._subclustering = subclustering

        # Sub-cluster information for intra cluster delay spread clusters
        # This is hardcoded from Table 7.5-5
        self._sub_cl_1_ind = tf.constant([0,1,2,3,4,5,6,7,18,19], tf.int32)
        self._sub_cl_2_ind = tf.constant([8,9,10,11,16,17], tf.int32)
        self._sub_cl_3_ind = tf.constant([12,13,14,15], tf.int32)
        self._sub_cl_delay_offsets = tf.constant([0, 1.28, 2.56],
                                                    dtype.real_dtype)

    def __call__(self, num_time_samples, sampling_frequency, k_factor, rays,
                 topology, c_ds=None, debug=False):
        # Sample times
        sample_times = (tf.range(num_time_samples,
                dtype=self._dtype.real_dtype)/sampling_frequency)

        # Step 10
        phi = self._step_10(tf.shape(rays.aoa))

        # Step 11
        h, delays = self._step_11(phi, topology, k_factor, rays, sample_times,
                                                                        c_ds)

        # Return additional information if requested
        if debug:
            return h, delays, phi, sample_times

        return h, delays

    ###########################################
    # Utility functions
    ###########################################

    def _unit_sphere_vector(self, theta, phi):
        r"""
        Generate vector on unit sphere (7.1-6)

        Input
        -------
        theta : Arbitrary shape, tf.float
            Zenith [radian]

        phi : Same shape as ``theta``, tf.float
            Azimuth [radian]

        Output
        --------
        rho_hat : ``phi.shape`` + [3, 1]
            Vector on unit sphere

        """
        rho_hat = tf.stack([sin(theta)*cos(phi),
                            sin(theta)*sin(phi),
                            cos(theta)], axis=-1)
        return tf.expand_dims(rho_hat, axis=-1)

    def _forward_rotation_matrix(self, orientations):
        r"""
        Forward composite rotation matrix (7.1-4)

        Input
        ------
            orientations : [...,3], tf.float
                Orientation to which to rotate [radian]

        Output
        -------
        R : [...,3,3], tf.float
            Rotation matrix
        """
        a, b, c = orientations[...,0], orientations[...,1], orientations[...,2]

        row_1 = tf.stack([cos(a)*cos(b),
            cos(a)*sin(b)*sin(c)-sin(a)*cos(c),
            cos(a)*sin(b)*cos(c)+sin(a)*sin(c)], axis=-1)

        row_2 = tf.stack([sin(a)*cos(b),
            sin(a)*sin(b)*sin(c)+cos(a)*cos(c),
            sin(a)*sin(b)*cos(c)-cos(a)*sin(c)], axis=-1)

        row_3 = tf.stack([-sin(b),
            cos(b)*sin(c),
            cos(b)*cos(c)], axis=-1)

        rot_mat = tf.stack([row_1, row_2, row_3], axis=-2)
        return rot_mat

    def _rot_pos(self, orientations, positions):
        r"""
        Rotate the ``positions`` according to the ``orientations``

        Input
        ------
        orientations : [...,3], tf.float
            Orientation to which to rotate [radian]

        positions : [...,3,1], tf.float
            Positions to rotate

        Output
        -------
        : [...,3,1], tf.float
            Rotated positions
        """
        rot_mat = self._forward_rotation_matrix(orientations)
        return tf.matmul(rot_mat, positions)

    def _reverse_rotation_matrix(self, orientations):
        r"""
        Reverse composite rotation matrix (7.1-4)

        Input
        ------
        orientations : [...,3], tf.float
            Orientations to rotate to  [radian]

        Output
        -------
        R_inv : [...,3,3], tf.float
            Inverse of the rotation matrix corresponding to ``orientations``
        """
        rot_mat = self._forward_rotation_matrix(orientations)
        rot_mat_inv = tf.linalg.matrix_transpose(rot_mat)
        return rot_mat_inv

    def _gcs_to_lcs(self, orientations, theta, phi):
        # pylint: disable=line-too-long
        r"""
        Compute the angles ``theta``, ``phi`` in LCS rotated according to
        ``orientations`` (7.1-7/8)

        Input
        ------
        orientations : [...,3] of rank K, tf.float
            Orientations to which to rotate to [radian]

        theta : Broadcastable to the first K-1 dimensions of ``orientations``, tf.float
            Zenith to rotate [radian]

        phi : Same dimension as ``theta``, tf.float
            Azimuth to rotate [radian]

        Output
        -------
        theta_prime : Same dimension as ``theta``, tf.float
            Rotated zenith

        phi_prime : Same dimensions as ``theta`` and ``phi``, tf.float
            Rotated azimuth
        """

        rho_hat = self._unit_sphere_vector(theta, phi)
        rot_inv = self._reverse_rotation_matrix(orientations)
        rot_rho = tf.matmul(rot_inv, rho_hat)
        v1 = tf.constant([0,0,1], self._dtype.real_dtype)
        v1 = tf.reshape(v1, [1]*(rot_rho.shape.rank-1)+[3])
        v2 = tf.constant([1+0j,1j,0], self._dtype)
        v2 = tf.reshape(v2, [1]*(rot_rho.shape.rank-1)+[3])
        z = tf.matmul(v1, rot_rho)
        z = tf.clip_by_value(z, tf.constant(-1., self._dtype.real_dtype),
                             tf.constant(1., self._dtype.real_dtype))
        theta_prime = acos(z)
        phi_prime = tf.math.angle((tf.matmul(v2, tf.cast(rot_rho,
            self._dtype))))
        theta_prime = tf.squeeze(theta_prime, axis=[phi.shape.rank,
            phi.shape.rank+1])
        phi_prime = tf.squeeze(phi_prime, axis=[phi.shape.rank,
            phi.shape.rank+1])

        return (theta_prime, phi_prime)

    def _compute_psi(self, orientations, theta, phi):
        # pylint: disable=line-too-long
        r"""
        Compute displacement angle :math:`Psi` for the transformation of LCS-GCS
        field components in (7.1-15) of TR38.901 specification

        Input
        ------
        orientations : [...,3], tf.float
            Orientations to which to rotate to [radian]

        theta :  Broadcastable to the first K-1 dimensions of ``orientations``, tf.float
            Spherical position zenith [radian]

        phi : Same dimensions as ``theta``, tf.float
            Spherical position azimuth [radian]

        Output
        -------
            Psi : Same shape as ``theta`` and ``phi``, tf.float
                Displacement angle :math:`Psi`
        """
        a = orientations[...,0]
        b = orientations[...,1]
        c = orientations[...,2]
        real = sin(c)*cos(theta)*sin(phi-a)
        real += cos(c)*(cos(b)*sin(theta)-sin(b)*cos(theta)*cos(phi-a))
        imag = sin(c)*cos(phi-a) + sin(b)*cos(c)*sin(phi-a)
        psi = tf.math.angle(tf.complex(real, imag))
        return psi

    def _l2g_response(self, f_prime, orientations, theta, phi):
        # pylint: disable=line-too-long
        r"""
        Transform field components from LCS to GCS (7.1-11)

        Input
        ------
        f_prime : K-Dim Tensor of shape [...,2], tf.float
            Field components

        orientations : K-Dim Tensor of shape [...,3], tf.float
            Orientations of LCS-GCS [radian]

        theta : K-1-Dim Tensor with matching dimensions to ``f_prime`` and ``phi``, tf.float
            Spherical position zenith [radian]

        phi : Same dimensions as ``theta``, tf.float
            Spherical position azimuth [radian]

        Output
        ------
            F : K+1-Dim Tensor with shape [...,2,1], tf.float
                The first K dimensions are identical to those of ``f_prime``
        """
        psi = self._compute_psi(orientations, theta, phi)
        row1 = tf.stack([cos(psi), -sin(psi)], axis=-1)
        row2 = tf.stack([sin(psi), cos(psi)], axis=-1)
        mat = tf.stack([row1, row2], axis=-2)
        f = tf.matmul(mat, tf.expand_dims(f_prime, -1))
        return f

    def _step_11_get_tx_antenna_positions(self, topology):
        r"""Compute d_bar_tx in (7.5-22), i.e., the positions in GCS of elements
        forming the transmit panel

        Input
        -----
        topology : Topology
            Topology of the network

        Output
        -------
        d_bar_tx : [batch_size, num TXs, num TX antenna, 3]
            Positions of the antenna elements in the GCS
        """
        # Get BS orientations got broadcasting
        tx_orientations = topology.tx_orientations
        tx_orientations = tf.expand_dims(tx_orientations, 2)

        # Get antenna element positions in LCS and reshape for broadcasting
        tx_ant_pos_lcs = self._tx_array.ant_pos
        tx_ant_pos_lcs = tf.reshape(tx_ant_pos_lcs,
            [1,1]+tx_ant_pos_lcs.shape+[1])

        # Compute antenna element positions in GCS
        tx_ant_pos_gcs = self._rot_pos(tx_orientations, tx_ant_pos_lcs)
        tx_ant_pos_gcs = tf.reshape(tx_ant_pos_gcs,
            tf.shape(tx_ant_pos_gcs)[:-1])

        d_bar_tx = tx_ant_pos_gcs

        return d_bar_tx

    def _step_11_get_rx_antenna_positions(self, topology):
        r"""Compute d_bar_rx in (7.5-22), i.e., the positions in GCS of elements
        forming the receive antenna panel

        Input
        -----
        topology : Topology
            Topology of the network

        Output
        -------
        d_bar_rx : [batch_size, num RXs, num RX antenna, 3]
            Positions of the antenna elements in the GCS
        """
        # Get UT orientations got broadcasting
        rx_orientations = topology.rx_orientations
        rx_orientations = tf.expand_dims(rx_orientations, 2)

        # Get antenna element positions in LCS and reshape for broadcasting
        rx_ant_pos_lcs = self._rx_array.ant_pos
        rx_ant_pos_lcs = tf.reshape(rx_ant_pos_lcs,
            [1,1]+rx_ant_pos_lcs.shape+[1])

        # Compute antenna element positions in GCS
        rx_ant_pos_gcs = self._rot_pos(rx_orientations, rx_ant_pos_lcs)
        rx_ant_pos_gcs = tf.reshape(rx_ant_pos_gcs,
            tf.shape(rx_ant_pos_gcs)[:-1])

        d_bar_rx = rx_ant_pos_gcs

        return d_bar_rx

    def _step_10(self, shape):
        r"""
        Generate random and uniformly distributed phases for all rays and
        polarization combinations

        Input
        -----
        shape : Shape tensor
            Shape of the leading dimensions for the tensor of phases to generate

        Output
        ------
        phi : [shape] + [4], tf.float
            Phases for all polarization combinations
        """
        phi = tf.random.uniform(tf.concat([shape, [4]], axis=0), minval=-PI,
            maxval=PI, dtype=self._dtype.real_dtype)

        return phi

    def _step_11_phase_matrix(self, phi, rays):
        # pylint: disable=line-too-long
        r"""
        Compute matrix with random phases in (7.5-22)

        Input
        -----
        phi : [batch size, num TXs, num RXs, num clusters, num rays, 4], tf.float
            Initial phases for all combinations of polarization

        rays : Rays
            Rays

        Output
        ------
        h_phase : [batch size, num TXs, num RXs, num clusters, num rays, 2, 2], tf.complex
            Matrix with random phases in (7.5-22)
        """
        xpr = rays.xpr

        xpr_scaling = tf.complex(tf.sqrt(1/xpr),
            tf.constant(0., self._dtype.real_dtype))
        e0 = tf.exp(tf.complex(tf.constant(0., self._dtype.real_dtype),
            phi[...,0]))
        e3 = tf.exp(tf.complex(tf.constant(0., self._dtype.real_dtype),
            phi[...,3]))
        e1 = xpr_scaling*tf.exp(tf.complex(tf.constant(0.,
                                self._dtype.real_dtype), phi[...,1]))
        e2 = xpr_scaling*tf.exp(tf.complex(tf.constant(0.,
                                self._dtype.real_dtype), phi[...,2]))
        shape = tf.concat([tf.shape(e0), [2,2]], axis=-1)
        h_phase = tf.reshape(tf.stack([e0, e1, e2, e3], axis=-1), shape)

        return h_phase

    def _step_11_doppler_matrix(self, topology, aoa, zoa, t):
        # pylint: disable=line-too-long
        r"""
        Compute matrix with phase shifts due to mobility in (7.5-22)

        Input
        -----
        topology : Topology
            Topology of the network

        aoa : [batch size, num TXs, num RXs, num clusters, num rays], tf.float
            Azimuth angles of arrivals [radian]

        zoa : [batch size, num TXs, num RXs, num clusters, num rays], tf.float
            Zenith angles of arrivals [radian]

        t : [number of time steps]
            Time steps at which the channel is sampled

        Output
        ------
        h_doppler : [batch size, num_tx, num rx, num clusters, num rays, num time steps], tf.complex
            Matrix with phase shifts due to mobility in (7.5-22)
        """
        lambda_0 = self._lambda_0
        velocities = topology.velocities

        # Add an extra dimension to make v_bar broadcastable with the time
        # dimension
        # v_bar [batch size, num tx or num rx, 3, 1]
        v_bar = velocities
        v_bar = tf.expand_dims(v_bar, axis=-1)

        # Depending on which end of the channel is moving, tx or rx, we add an
        # extra dimension to make this tensor broadcastable with the other end
        if topology.moving_end == 'rx':
            # v_bar [batch size, 1, num rx, num tx, 1]
            v_bar = tf.expand_dims(v_bar, 1)
        elif topology.moving_end == 'tx':
            # v_bar [batch size, num tx, 1, num tx, 1]
            v_bar = tf.expand_dims(v_bar, 2)

        # v_bar [batch size, 1, num rx, 1, 1, 3, 1]
        # or    [batch size, num tx, 1, 1, 1, 3, 1]
        v_bar = tf.expand_dims(tf.expand_dims(v_bar, -3), -3)

        # v_bar [batch size, num_tx, num rx, num clusters, num rays, 3, 1]
        r_hat_rx = self._unit_sphere_vector(zoa, aoa)

        # Compute phase shift due to doppler
        # [batch size, num_tx, num rx, num clusters, num rays, num time steps]
        exponent = 2*PI/lambda_0*tf.reduce_sum(r_hat_rx*v_bar, -2)*t
        h_doppler = tf.exp(tf.complex(tf.constant(0.,
                                    self._dtype.real_dtype), exponent))

        # [batch size, num_tx, num rx, num clusters, num rays, num time steps]
        return h_doppler

    def _step_11_array_offsets(self, topology, aoa, aod, zoa, zod):
        # pylint: disable=line-too-long
        r"""
        Compute matrix accounting for phases offsets between antenna elements

        Input
        -----
        topology : Topology
            Topology of the network

        aoa : [batch size, num TXs, num RXs, num clusters, num rays], tf.float
            Azimuth angles of arrivals [radian]

        aod : [batch size, num TXs, num RXs, num clusters, num rays], tf.float
            Azimuth angles of departure [radian]

        zoa : [batch size, num TXs, num RXs, num clusters, num rays], tf.float
            Zenith angles of arrivals [radian]

        zod : [batch size, num TXs, num RXs, num clusters, num rays], tf.float
            Zenith angles of departure [radian]
        Output
        ------
        h_array : [batch size, num_tx, num rx, num clusters, num rays, num rx antennas, num tx antennas], tf.complex
            Matrix accounting for phases offsets between antenna elements
        """

        lambda_0 = self._lambda_0

        r_hat_rx = self._unit_sphere_vector(zoa, aoa)
        r_hat_rx = tf.squeeze(r_hat_rx, axis=r_hat_rx.shape.rank-1)
        r_hat_tx = self._unit_sphere_vector(zod, aod)
        r_hat_tx = tf.squeeze(r_hat_tx, axis=r_hat_tx.shape.rank-1)
        d_bar_rx = self._step_11_get_rx_antenna_positions(topology)
        d_bar_tx = self._step_11_get_tx_antenna_positions(topology)

        # Reshape tensors for broadcasting
        # r_hat_rx/tx have
        # shape [batch_size, num_tx, num_rx, num_clusters, num_rays,    3]
        # and will be reshaoed to
        # [batch_size, num_tx, num_rx, num_clusters, num_rays, 1, 3]
        r_hat_tx = tf.expand_dims(r_hat_tx, -2)
        r_hat_rx = tf.expand_dims(r_hat_rx, -2)

        # d_bar_tx has shape [batch_size, num_tx,          num_tx_antennas, 3]
        # and will be reshaped to
        # [batch_size, num_tx, 1, 1, 1, num_tx_antennas, 3]
        s = tf.shape(d_bar_tx)
        shape = tf.concat([s[:2], [1,1,1], s[2:]], 0)
        d_bar_tx = tf.reshape(d_bar_tx, shape)

        # d_bar_rx has shape [batch_size,    num_rx,       num_rx_antennas, 3]
        # and will be reshaped to
        # [batch_size, 1, num_rx, 1, 1, num_rx_antennas, 3]
        s = tf.shape(d_bar_rx)
        shape = tf.concat([[s[0]], [1, s[1], 1,1], s[2:]], 0)
        d_bar_rx = tf.reshape(d_bar_rx, shape)

        # Compute all tensor elements

        # As broadcasting of such high-rank tensors is not fully supported
        # in all cases, we need to do a hack here by explicitly
        # broadcasting one dimension:
        s = tf.shape(d_bar_rx)
        shape = tf.concat([ [s[0]], [tf.shape(r_hat_rx)[1]], s[2:]], 0)
        d_bar_rx = tf.broadcast_to(d_bar_rx, shape)
        exp_rx = 2*PI/lambda_0*tf.reduce_sum(r_hat_rx*d_bar_rx,
            axis=-1, keepdims=True)
        exp_rx = tf.exp(tf.complex(tf.constant(0.,
                                    self._dtype.real_dtype), exp_rx))

        # The hack is for some reason not needed for this term
        # exp_tx = 2*PI/lambda_0*tf.reduce_sum(r_hat_tx*d_bar_tx,
        #     axis=-1, keepdims=True)
        exp_tx = 2*PI/lambda_0*tf.reduce_sum(r_hat_tx*d_bar_tx,
            axis=-1)
        exp_tx = tf.exp(tf.complex(tf.constant(0.,
                                    self._dtype.real_dtype), exp_tx))
        exp_tx = tf.expand_dims(exp_tx, -2)

        h_array = exp_rx*exp_tx

        return h_array

    def _step_11_field_matrix(self, topology, aoa, aod, zoa, zod, h_phase):
        # pylint: disable=line-too-long
        r"""
        Compute matrix accounting for the element responses, random phases
        and xpr

        Input
        -----
        topology : Topology
            Topology of the network

        aoa : [batch size, num TXs, num RXs, num clusters, num rays], tf.float
            Azimuth angles of arrivals [radian]

        aod : [batch size, num TXs, num RXs, num clusters, num rays], tf.float
            Azimuth angles of departure [radian]

        zoa : [batch size, num TXs, num RXs, num clusters, num rays], tf.float
            Zenith angles of arrivals [radian]

        zod : [batch size, num TXs, num RXs, num clusters, num rays], tf.float
            Zenith angles of departure [radian]

        h_phase : [batch size, num_tx, num rx, num clusters, num rays, num time steps], tf.complex
            Matrix with phase shifts due to mobility in (7.5-22)

        Output
        ------
        h_field : [batch size, num_tx, num rx, num clusters, num rays, num rx antennas, num tx antennas], tf.complex
            Matrix accounting for element responses, random phases and xpr
        """

        tx_orientations = topology.tx_orientations
        rx_orientations = topology.rx_orientations

        # Transform departure angles to the LCS
        s = tf.shape(tx_orientations)
        shape = tf.concat([s[:2], [1,1,1,s[-1]]], 0)
        tx_orientations = tf.reshape(tx_orientations, shape)
        zod_prime, aod_prime = self._gcs_to_lcs(tx_orientations, zod, aod)

        # Transform arrival angles to the LCS
        s = tf.shape(rx_orientations)
        shape = tf.concat([[s[0],1],[s[1],1,1,s[-1]]], 0)
        rx_orientations = tf.reshape(rx_orientations, shape)
        zoa_prime, aoa_prime = self._gcs_to_lcs(rx_orientations, zoa, aoa)

        # Compute transmitted and received field strength for all antennas
        # in the LCS  and convert to GCS
        f_tx_pol1_prime = tf.stack(self._tx_array.ant_pol1.field(zod_prime,
                                                            aod_prime), axis=-1)
        f_rx_pol1_prime = tf.stack(self._rx_array.ant_pol1.field(zoa_prime,
                                                            aoa_prime), axis=-1)

        f_tx_pol1 = self._l2g_response(f_tx_pol1_prime, tx_orientations,
            zod, aod)

        f_rx_pol1 = self._l2g_response(f_rx_pol1_prime, rx_orientations,
            zoa, aoa)

        if self._tx_array.polarization == 'dual':
            f_tx_pol2_prime = tf.stack(self._tx_array.ant_pol2.field(
                zod_prime, aod_prime), axis=-1)
            f_tx_pol2 = self._l2g_response(f_tx_pol2_prime, tx_orientations,
                zod, aod)

        if self._rx_array.polarization == 'dual':
            f_rx_pol2_prime = tf.stack(self._rx_array.ant_pol2.field(
                zoa_prime, aoa_prime), axis=-1)
            f_rx_pol2 = self._l2g_response(f_rx_pol2_prime, rx_orientations,
                zoa, aoa)

        # Fill the full channel matrix with field responses
        pol1_tx = tf.matmul(h_phase, tf.complex(f_tx_pol1,
            tf.constant(0., self._dtype.real_dtype)))
        if self._tx_array.polarization == 'dual':
            pol2_tx = tf.matmul(h_phase, tf.complex(f_tx_pol2, tf.constant(0.,
                                            self._dtype.real_dtype)))

        num_ant_tx = self._tx_array.num_ant
        if self._tx_array.polarization == 'single':
            # Each BS antenna gets the polarization 1 response
            f_tx_array = tf.tile(tf.expand_dims(pol1_tx, 0),
                tf.concat([[num_ant_tx], tf.ones([tf.rank(pol1_tx)], tf.int32)],
                axis=0))
        else:
            # Assign polarization reponse according to polarization to each
            # antenna
            pol_tx = tf.stack([pol1_tx, pol2_tx], 0)
            ant_ind_pol2 = self._tx_array.ant_ind_pol2
            num_ant_pol2 = ant_ind_pol2.shape[0]
            # O = Pol 1, 1 = Pol 2, we only scatter the indices for Pol 1,
            # the other elements are already 0
            gather_ind = tf.scatter_nd(tf.reshape(ant_ind_pol2, [-1,1]),
                tf.ones([num_ant_pol2], tf.int32), [num_ant_tx])
            f_tx_array = tf.gather(pol_tx, gather_ind, axis=0)

        num_ant_rx = self._rx_array.num_ant
        if self._rx_array.polarization == 'single':
            # Each UT antenna gets the polarization 1 response
            f_rx_array = tf.tile(tf.expand_dims(f_rx_pol1, 0),
                tf.concat([[num_ant_rx], tf.ones([tf.rank(f_rx_pol1)],
                                                 tf.int32)], axis=0))
            f_rx_array = tf.complex(f_rx_array,
                                    tf.constant(0., self._dtype.real_dtype))
        else:
            # Assign polarization response according to polarization to each
            # antenna
            pol_rx = tf.stack([f_rx_pol1, f_rx_pol2], 0)
            ant_ind_pol2 = self._rx_array.ant_ind_pol2
            num_ant_pol2 = ant_ind_pol2.shape[0]
            # O = Pol 1, 1 = Pol 2, we only scatter the indices for Pol 1,
            # the other elements are already 0
            gather_ind = tf.scatter_nd(tf.reshape(ant_ind_pol2, [-1,1]),
                tf.ones([num_ant_pol2], tf.int32), [num_ant_rx])
            f_rx_array = tf.complex(tf.gather(pol_rx, gather_ind, axis=0),
                            tf.constant(0., self._dtype.real_dtype))

        # Compute the scalar product between the field vectors through
        # reduce_sum and transpose to put antenna dimensions last
        h_field = tf.reduce_sum(tf.expand_dims(f_rx_array, 1)*tf.expand_dims(
            f_tx_array, 0), [-2,-1])
        h_field = tf.transpose(h_field, tf.roll(tf.range(tf.rank(h_field)),
            -2, 0))

        return h_field

    def _step_11_nlos(self, phi, topology, rays, t):
        # pylint: disable=line-too-long
        r"""
        Compute the full NLOS channel matrix (7.5-28)

        Input
        -----
        phi: [batch size, num TXs, num RXs, num clusters, num rays, 4], tf.float
            Random initial phases [radian]

        topology : Topology
            Topology of the network

        rays : Rays
            Rays

        t : [num time samples], tf.float
            Time samples

        Output
        ------
        h_full : [batch size, num_tx, num rx, num clusters, num rays, num rx antennas, num tx antennas, num time steps], tf.complex
            NLoS channel matrix
        """

        h_phase = self._step_11_phase_matrix(phi, rays)
        h_field = self._step_11_field_matrix(topology, rays.aoa, rays.aod,
                                                    rays.zoa, rays.zod, h_phase)
        h_array = self._step_11_array_offsets(topology, rays.aoa, rays.aod,
                                                            rays.zoa, rays.zod)
        h_doppler = self._step_11_doppler_matrix(topology, rays.aoa, rays.zoa,
                                                                            t)
        h_full = tf.expand_dims(h_field*h_array, -1) * tf.expand_dims(
            tf.expand_dims(h_doppler, -2), -2)

        power_scaling = tf.complex(tf.sqrt(rays.powers/
            tf.cast(tf.shape(h_full)[4], self._dtype.real_dtype)),
                            tf.constant(0., self._dtype.real_dtype))
        shape = tf.concat([tf.shape(power_scaling), tf.ones(
            [tf.rank(h_full)-tf.rank(power_scaling)], tf.int32)], 0)
        h_full *= tf.reshape(power_scaling, shape)

        return h_full

    def _step_11_reduce_nlos(self, h_full, rays, c_ds):
        # pylint: disable=line-too-long
        r"""
        Compute the final NLOS matrix in (7.5-27)

        Input
        ------
        h_full : [batch size, num_tx, num rx, num clusters, num rays, num rx antennas, num tx antennas, num time steps], tf.complex
            NLoS channel matrix

        rays : Rays
            Rays

        c_ds : [batch size, num TX, num RX], tf.float
            Cluster delay spread

        Output
        -------
        h_nlos : [batch size, num_tx, num rx, num clusters, num rx antennas, num tx antennas, num time steps], tf.complex
            Paths NLoS coefficients

        delays_nlos : [batch size, num_tx, num rx, num clusters], tf.float
            Paths NLoS delays
        """

        if self._subclustering:

            powers = rays.powers
            delays = rays.delays

            # Sort all clusters along their power
            strongest_clusters = tf.argsort(powers, axis=-1,
                direction="DESCENDING")

            # Sort delays according to the same ordering
            delays_sorted = tf.gather(delays, strongest_clusters,
                batch_dims=3, axis=3)

            # Split into delays for strong and weak clusters
            delays_strong = delays_sorted[...,:2]
            delays_weak = delays_sorted[...,2:]

            # Compute delays for sub-clusters
            offsets = tf.reshape(self._sub_cl_delay_offsets,
                (delays_strong.shape.rank-1)*[1]+[-1]+[1])
            delays_sub_cl = (tf.expand_dims(delays_strong, -2) +
                offsets*tf.expand_dims(tf.expand_dims(c_ds, axis=-1), axis=-1))
            delays_sub_cl = tf.reshape(delays_sub_cl,
                tf.concat([tf.shape(delays_sub_cl)[:-2], [-1]],0))

            # Select the strongest two clusters for sub-cluster splitting
            h_strong = tf.gather(h_full, strongest_clusters[...,:2],
                batch_dims=3, axis=3)

            # The other clusters are the weak clusters
            h_weak = tf.gather(h_full, strongest_clusters[...,2:],
                batch_dims=3, axis=3)

            # Sum specific rays for each sub-cluster
            h_sub_cl_1 = tf.reduce_sum(tf.gather(h_strong,
                self._sub_cl_1_ind, axis=4), axis=4)
            h_sub_cl_2 = tf.reduce_sum(tf.gather(h_strong,
                self._sub_cl_2_ind, axis=4), axis=4)
            h_sub_cl_3 = tf.reduce_sum(tf.gather(h_strong,
                self._sub_cl_3_ind, axis=4), axis=4)

            # Sum all rays for the weak clusters
            h_weak = tf.reduce_sum(h_weak, axis=4)

            # Concatenate the channel and delay tensors
            h_nlos = tf.concat([h_sub_cl_1, h_sub_cl_2, h_sub_cl_3, h_weak],
                axis=3)
            delays_nlos = tf.concat([delays_sub_cl, delays_weak], axis=3)
        else:
            # Sum over rays
            h_nlos = tf.reduce_sum(h_full, axis=4)
            delays_nlos = rays.delays

        # Order the delays in ascending orders
        delays_ind = tf.argsort(delays_nlos, axis=-1,
            direction="ASCENDING")
        delays_nlos = tf.gather(delays_nlos, delays_ind, batch_dims=3,
            axis=3)

        # # Order the channel clusters according to the delay, too
        h_nlos = tf.gather(h_nlos, delays_ind, batch_dims=3, axis=3)

        return h_nlos, delays_nlos

    def _step_11_los(self, topology, t):
        # pylint: disable=line-too-long
        r"""Compute the LOS channels from (7.5-29)

        Intput
        ------
        topology : Topology
            Network topology

        t : [num time samples], tf.float
            Number of time samples

        Output
        ------
        h_los : [batch size, num_tx, num rx, 1, num rx antennas, num tx antennas, num time steps], tf.complex
            Paths LoS coefficients
        """

        aoa = topology.los_aoa
        aod = topology.los_aod
        zoa = topology.los_zoa
        zod = topology.los_zod

         # LoS departure and arrival angles
        aoa = tf.expand_dims(tf.expand_dims(aoa, axis=3), axis=4)
        zoa = tf.expand_dims(tf.expand_dims(zoa, axis=3), axis=4)
        aod = tf.expand_dims(tf.expand_dims(aod, axis=3), axis=4)
        zod = tf.expand_dims(tf.expand_dims(zod, axis=3), axis=4)

        # Field matrix
        h_phase = tf.reshape(tf.constant([[1.,0.],
                                         [0.,-1.]],
                                         self._dtype),
                                         [1,1,1,1,1,2,2])
        h_field = self._step_11_field_matrix(topology, aoa, aod, zoa, zod,
                                                                    h_phase)

        # Array offset matrix
        h_array = self._step_11_array_offsets(topology, aoa, aod, zoa, zod)

        # Doppler matrix
        h_doppler = self._step_11_doppler_matrix(topology, aoa, zoa, t)

        # Phase shift due to propagation delay
        d3d = topology.distance_3d
        lambda_0 = self._lambda_0
        h_delay = tf.exp(tf.complex(tf.constant(0.,
                        self._dtype.real_dtype), 2*PI*d3d/lambda_0))

        # Combining all to compute channel coefficient
        h_field = tf.expand_dims(tf.squeeze(h_field, axis=4), axis=-1)
        h_array = tf.expand_dims(tf.squeeze(h_array, axis=4), axis=-1)
        h_doppler = tf.expand_dims(h_doppler, axis=4)
        h_delay = tf.expand_dims(tf.expand_dims(tf.expand_dims(
            tf.expand_dims(h_delay, axis=3), axis=4), axis=5), axis=6)

        h_los = h_field*h_array*h_doppler*h_delay
        return h_los

    def _step_11(self, phi, topology, k_factor, rays, t, c_ds):
        # pylint: disable=line-too-long
        r"""
        Combine LOS and LOS components to compute (7.5-30)

        Input
        -----
        phi: [batch size, num TXs, num RXs, num clusters, num rays, 4], tf.float
            Random initial phases

        topology : Topology
            Network topology

        k_factor : [batch size, num TX, num RX], tf.float
            Rician K-factor

        rays : Rays
            Rays

        t : [num time samples], tf.float
            Number of time samples

        c_ds : [batch size, num TX, num RX], tf.float
            Cluster delay spread
        """

        h_full = self._step_11_nlos(phi, topology, rays, t)
        h_nlos, delays_nlos = self._step_11_reduce_nlos(h_full, rays, c_ds)

        ####  LoS scenario

        h_los_los_comp = self._step_11_los(topology, t)
        k_factor = tf.reshape(k_factor, tf.concat([tf.shape(k_factor),
            tf.ones([tf.rank(h_los_los_comp)-tf.rank(k_factor)], tf.int32)],0))
        k_factor = tf.complex(k_factor, tf.constant(0.,
                                            self._dtype.real_dtype))

        # Scale NLOS and LOS components according to K-factor
        h_los_los_comp = h_los_los_comp*tf.sqrt(k_factor/(k_factor+1))
        h_los_nlos_comp = h_nlos*tf.sqrt(1/(k_factor+1))

        # Add the LOS component to the zero-delay NLOS cluster
        h_los_cl = h_los_los_comp + tf.expand_dims(
            h_los_nlos_comp[:,:,:,0,...], 3)

        # Combine all clusters into a single tensor
        h_los = tf.concat([h_los_cl, h_los_nlos_comp[:,:,:,1:,...]], axis=3)

        #### LoS or NLoS CIR according to link configuration
        los_indicator = tf.reshape(topology.los,
            tf.concat([tf.shape(topology.los), [1,1,1,1]], axis=0))
        h = tf.where(los_indicator, h_los, h_nlos)

        return h, delays_nlos
