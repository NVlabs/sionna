#
# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""
Class for sampling rays following 3GPP TR38.901 specifications and giving a
channel simulation scenario and LSPs.
"""

import tensorflow as tf

from sionna.utils import log10
from sionna.channel.utils import deg_2_rad, wrap_angle_0_360


class Rays:
    # pylint: disable=line-too-long
    r"""
    Class for conveniently storing rays

    Parameters
    -----------

    delays : [batch size, number of BSs, number of UTs, number of clusters], tf.float
        Paths delays [s]

    powers : [batch size, number of BSs, number of UTs, number of clusters], tf.float
        Normalized path powers

    aoa : (batch size, number of BSs, number of UTs, number of clusters, number of rays], tf.float
        Azimuth angles of arrival [radian]

    aod : [batch size, number of BSs, number of UTs, number of clusters, number of rays], tf.float
        Azimuth angles of departure [radian]

    zoa : [batch size, number of BSs, number of UTs, number of clusters, number of rays], tf.float
        Zenith angles of arrival [radian]

    zod : [batch size, number of BSs, number of UTs, number of clusters, number of rays], tf.float
        Zenith angles of departure [radian]

    xpr [batch size, number of BSs, number of UTs, number of clusters, number of rays], tf.float
        Coss-polarization power ratios.
    """

    def __init__(self, delays, powers, aoa, aod, zoa, zod, xpr):
        self.delays = delays
        self.powers = powers
        self.aoa = aoa
        self.aod = aod
        self.zoa = zoa
        self.zod = zod
        self.xpr = xpr


class RaysGenerator:
    """
    Sample rays according to a given channel scenario and large scale
    parameters (LSP).

    This class implements steps 6 to 9 from the TR 38.901 specifications,
    (section 7.5).

    Note that a global scenario is set for the entire batches when instantiating
    this class (UMa, UMi, or RMa). However, each UT-BS link can have its
    specific state (LoS, NLoS, or indoor).

    The batch size is set by the ``scenario`` given as argument when
    constructing the class.

    Parameters
    ----------
    scenario : :class:`~sionna.channel.tr38901.SystemLevelScenario``
        Scenario used to generate LSPs

    Input
    -----
    lsp : :class:`~sionna.channel.tr38901.LSP`
        LSPs samples

    Output
    ------
    rays : :class:`~sionna.channel.tr38901.Rays`
        Rays samples
    """

    def __init__(self, scenario):
        # Scenario
        self._scenario = scenario

        # For AoA, AoD, ZoA, and ZoD, offset to add to cluster angles to get ray
        # angles. This is hardcoded from table 7.5-3 for 3GPP 38.901
        # specification.
        self._ray_offsets = tf.constant([0.0447, -0.0447,
                                         0.1413, -0.1413,
                                         0.2492, -0.2492,
                                         0.3715, -0.3715,
                                         0.5129, -0.5129,
                                         0.6797, -0.6797,
                                         0.8844, -0.8844,
                                         1.1481, -0.1481,
                                         1.5195, -1.5195,
                                         2.1551, -2.1551],
                                         self._scenario.dtype.real_dtype)

    #########################################
    # Public methods and properties
    #########################################

    def __call__(self, lsp):
        # Sample cluster delays
        delays, delays_unscaled = self._cluster_delays(lsp.ds, lsp.k_factor)

        # Sample cluster powers
        powers, powers_for_angles_gen = self._cluster_powers(lsp.ds,
                                            lsp.k_factor, delays_unscaled)

        # Sample AoA
        aoa = self._azimuth_angles_of_arrival(lsp.asa, lsp.k_factor,
                                                powers_for_angles_gen)

        # Sample AoD
        aod = self._azimuth_angles_of_departure(lsp.asd, lsp.k_factor,
                                                powers_for_angles_gen)

        # Sample ZoA
        zoa = self._zenith_angles_of_arrival(lsp.zsa, lsp.k_factor,
                                                powers_for_angles_gen)

        # Sample ZoD
        zod = self._zenith_angles_of_departure(lsp.zsd, lsp.k_factor,
                                                powers_for_angles_gen)

        # XPRs
        xpr = self._cross_polarization_power_ratios()

        # Random coupling
        aoa, aod, zoa, zod = self._random_coupling(aoa, aod, zoa, zod)

        # Convert angles of arrival and departure from degree to radian
        aoa = deg_2_rad(aoa)
        aod = deg_2_rad(aod)
        zoa = deg_2_rad(zoa)
        zod = deg_2_rad(zod)

        # Storing and returning rays
        rays = Rays(delays = delays,
                    powers = powers,
                    aoa    = aoa,
                    aod    = aod,
                    zoa    = zoa,
                    zod    = zod,
                    xpr    = xpr)

        return rays

    def topology_updated_callback(self):
        """
        Updates internal quantities. Must be called at every update of the
        scenario that changes the state of UTs or their locations.

        Input
        ------
        None

        Output
        ------
        None
        """
        self._compute_clusters_mask()

    ########################################
    # Internal utility methods
    ########################################

    def _compute_clusters_mask(self):
        """
        Given a scenario (UMi, UMa, RMa), the number of clusters is different
        for different state of UT-BS links (LoS, NLoS, indoor).

        Because we use tensors with predefined dimension size (not ragged), the
        cluster dimension is always set to the maximum number of clusters the
        scenario requires. A mask is then used to discard not required tensors,
        depending on the state of each UT-BS link.

        This function computes and stores this mask of size
        [batch size, number of BSs, number of UTs, maximum number of cluster]
        where an element equals 0 if the cluster is used, 1 otherwise.
        """

        scenario = self._scenario
        num_clusters_los = scenario.num_clusters_los
        num_clusters_nlos = scenario.num_clusters_nlos
        num_clusters_o2i = scenario.num_clusters_indoor
        num_clusters_max = tf.reduce_max([num_clusters_los, num_clusters_nlos,
            num_clusters_o2i])


        # Initialize an empty mask
        mask = tf.zeros(shape=[scenario.batch_size, scenario.num_bs,
            scenario.num_ut, num_clusters_max],
            dtype=self._scenario.dtype.real_dtype)

        # Indoor mask
        mask_indoor = tf.concat((tf.zeros([num_clusters_o2i],
                                          self._scenario.dtype.real_dtype),
                                 tf.ones([num_clusters_max-num_clusters_o2i],
                                    self._scenario.dtype.real_dtype)), axis=0)
        mask_indoor = tf.reshape(mask_indoor, [1, 1, 1, num_clusters_max])
        indoor = tf.expand_dims(scenario.indoor, axis=1) # Broadcasting with BS
        o2i_slice_mask = tf.cast(indoor, self._scenario.dtype.real_dtype)
        o2i_slice_mask = tf.expand_dims(o2i_slice_mask, axis=3)
        mask = mask + o2i_slice_mask*mask_indoor

        # LoS
        mask_los = tf.concat([tf.zeros([num_clusters_los],
            self._scenario.dtype.real_dtype),
            tf.ones([num_clusters_max-num_clusters_los],
            self._scenario.dtype.real_dtype)], axis=0)
        mask_los = tf.reshape(mask_los, [1, 1, 1, num_clusters_max])
        los_slice_mask = scenario.los
        los_slice_mask = tf.cast(los_slice_mask,
                                    self._scenario.dtype.real_dtype)
        los_slice_mask = tf.expand_dims(los_slice_mask, axis=3)
        mask = mask + los_slice_mask*mask_los

        # NLoS
        mask_nlos = tf.concat([tf.zeros([num_clusters_nlos],
            self._scenario.dtype.real_dtype),
            tf.ones([num_clusters_max-num_clusters_nlos],
            self._scenario.dtype.real_dtype)], axis=0)
        mask_nlos = tf.reshape(mask_nlos, [1, 1, 1, num_clusters_max])
        nlos_slice_mask = tf.logical_and(tf.logical_not(scenario.los),
            tf.logical_not(indoor))
        nlos_slice_mask = tf.cast(nlos_slice_mask,
                                    self._scenario.dtype.real_dtype)
        nlos_slice_mask = tf.expand_dims(nlos_slice_mask, axis=3)
        mask = mask + nlos_slice_mask*mask_nlos

        # Save the mask
        self._cluster_mask = mask

    def _cluster_delays(self, delay_spread, rician_k_factor):
        # pylint: disable=line-too-long
        """
        Generate cluster delays.
        See step 5 of section 7.5 from TR 38.901 specification.

        Input
        ------
        delay_spread : [batch size, num of BSs, num of UTs], tf.float
            RMS delay spread of each BS-UT link.

        rician_k_factor : [batch size, num of BSs, num of UTs], tf.float
            Rician K-factor of each BS-UT link. Used only for LoS links.

        Output
        -------
        delays : [batch size, num of BSs, num of UTs, maximum number of clusters], tf.float
            Path delays [s]

        unscaled_delays [batch size, num of BSs, num of UTs, maximum number of clusters], tf.float
            Unscaled path delays [s]
        """

        scenario = self._scenario

        batch_size = scenario.batch_size
        num_bs = scenario.num_bs
        num_ut = scenario.num_ut

        num_clusters_max = scenario.num_clusters_max

        # Getting scaling parameter according to each BS-UT link scenario
        delay_scaling_parameter = scenario.get_param("rTau")
        delay_scaling_parameter = tf.expand_dims(delay_scaling_parameter,
            axis=3)

        # Generating random cluster delays
        # We don't start at 0 to avoid numerical errors
        delay_spread = tf.expand_dims(delay_spread, axis=3)
        x = tf.random.uniform(shape=[batch_size, num_bs, num_ut,
            num_clusters_max], minval=1e-6, maxval=1.0,
            dtype=self._scenario.dtype.real_dtype)

        # Moving to linear domain
        unscaled_delays = -delay_scaling_parameter*delay_spread*tf.math.log(x)
        # Forcing the cluster that should not exist to huge delays (1s)
        unscaled_delays = (unscaled_delays*(1.-self._cluster_mask)
            + self._cluster_mask)

        # Normalizing and sorting the delays
        unscaled_delays = unscaled_delays - tf.reduce_min(unscaled_delays,
            axis=3, keepdims=True)
        unscaled_delays = tf.sort(unscaled_delays, axis=3)

        # Additional scaling applied to LoS links
        rician_k_factor_db = 10.0*log10(rician_k_factor) # to dB
        scaling_factor = (0.7705 - 0.0433*rician_k_factor_db
            + 0.0002*tf.square(rician_k_factor_db)
            + 0.000017*tf.math.pow(rician_k_factor_db, tf.constant(3.,
            self._scenario.dtype.real_dtype)))
        scaling_factor = tf.expand_dims(scaling_factor, axis=3)
        delays = tf.where(tf.expand_dims(scenario.los, axis=3),
            unscaled_delays / scaling_factor, unscaled_delays)

        return delays, unscaled_delays

    def _cluster_powers(self, delay_spread, rician_k_factor, unscaled_delays):
        # pylint: disable=line-too-long
        """
        Generate cluster powers.
        See step 6 of section 7.5 from TR 38.901 specification.

        Input
        ------
        delays : [batch size, num of BSs, num of UTs, maximum number of clusters], tf.float
            Path delays [s]

        rician_k_factor : [batch size, num of BSs, num of UTs], tf.float
            Rician K-factor of each BS-UT link. Used only for LoS links.

        unscaled_delays [batch size, num of BSs, num of UTs, maximum number of clusters], tf.float
            Unscaled path delays [s]. Required to compute the path powers.

        Output
        -------
        powers : [batch size, num of BSs, num of UTs, maximum number of clusters], tf.float
            Normalized path powers
        """

        scenario = self._scenario

        batch_size = scenario.batch_size
        num_bs = scenario.num_bs
        num_ut = scenario.num_ut

        num_clusters_max = scenario.num_clusters_max

        delay_scaling_parameter = scenario.get_param("rTau")
        cluster_shadowing_std_db = scenario.get_param("zeta")
        delay_spread = tf.expand_dims(delay_spread, axis=3)
        cluster_shadowing_std_db = tf.expand_dims(cluster_shadowing_std_db,
            axis=3)
        delay_scaling_parameter = tf.expand_dims(delay_scaling_parameter,
            axis=3)

        # Generate unnormalized cluster powers
        z = tf.random.normal(shape=[batch_size, num_bs, num_ut,
            num_clusters_max], mean=0.0, stddev=cluster_shadowing_std_db,
            dtype=self._scenario.dtype.real_dtype)

        # Moving to linear domain
        powers_unnormalized = (tf.math.exp(-unscaled_delays*
            (delay_scaling_parameter - 1.0)/
            (delay_scaling_parameter*delay_spread))*tf.math.pow(tf.constant(10.,
            self._scenario.dtype.real_dtype), -z/10.0))

        # Force the power of unused cluster to zero
        powers_unnormalized = powers_unnormalized*(1.-self._cluster_mask)

        # Normalizing cluster powers
        powers = (powers_unnormalized/
            tf.reduce_sum(powers_unnormalized, axis=3, keepdims=True))

        # Additional specular component for LoS
        rician_k_factor = tf.expand_dims(rician_k_factor, axis=3)
        p_nlos_scaling = 1.0/(rician_k_factor + 1.0)
        p_1_los = rician_k_factor*p_nlos_scaling
        powers_1 = p_nlos_scaling*powers[:,:,:,:1] + p_1_los
        powers_n = p_nlos_scaling*powers[:,:,:,1:]
        powers_for_angles_gen = tf.where(tf.expand_dims(scenario.los, axis=3),
            tf.concat([powers_1, powers_n], axis=3), powers)

        return powers, powers_for_angles_gen

    def _azimuth_angles(self, azimuth_spread, rician_k_factor, cluster_powers,
                        angle_type):
        # pylint: disable=line-too-long
        """
        Generate departure or arrival azimuth angles (degrees).
        See step 7 of section 7.5 from TR 38.901 specification.

        Input
        ------
        azimuth_spread : [batch size, num of BSs, num of UTs], tf.float
            Angle spread, (ASD or ASA) depending on ``angle_type`` [deg]

        rician_k_factor : [batch size, num of BSs, num of UTs], tf.float
            Rician K-factor of each BS-UT link. Used only for LoS links.

        cluster_powers : [batch size, num of BSs, num of UTs, maximum number of clusters], tf.float
            Normalized path powers

        angle_type : str
            Type of angle to compute. Must be 'aoa' or 'aod'.

        Output
        -------
        azimuth_angles : [batch size, num of BSs, num of UTs, maximum number of clusters, number of rays], tf.float
            Paths azimuth angles wrapped within (-180, 180) [degree]. Either the AoA or AoD depending on ``angle_type``.
        """

        scenario = self._scenario

        batch_size = scenario.batch_size
        num_bs = scenario.num_bs
        num_ut = scenario.num_ut

        num_clusters_max = scenario.num_clusters_max

        azimuth_spread = tf.expand_dims(azimuth_spread, axis=3)

        # Loading the angle spread
        if angle_type == 'aod':
            azimuth_angles_los = scenario.los_aod
            cluster_angle_spread = scenario.get_param('cASD')
        else:
            azimuth_angles_los = scenario.los_aoa
            cluster_angle_spread = scenario.get_param('cASA')
        # Adding cluster dimension for broadcasting
        azimuth_angles_los = tf.expand_dims(azimuth_angles_los, axis=3)
        cluster_angle_spread = tf.expand_dims(tf.expand_dims(
            cluster_angle_spread, axis=3), axis=4)

        # Compute C-phi constant
        rician_k_factor = tf.expand_dims(rician_k_factor, axis=3)
        rician_k_factor_db = 10.0*log10(rician_k_factor) # to dB
        c_phi_nlos = tf.expand_dims(scenario.get_param("CPhiNLoS"), axis=3)
        c_phi_los = c_phi_nlos*(1.1035- 0.028*rician_k_factor_db
            - 0.002*tf.square(rician_k_factor_db)
            + 0.0001*tf.math.pow(rician_k_factor_db, 3.))
        c_phi = tf.where(tf.expand_dims(scenario.los, axis=3),
            c_phi_los, c_phi_nlos)

        # Inverse Gaussian function
        z = cluster_powers/tf.reduce_max(cluster_powers, axis=3, keepdims=True)
        z = tf.clip_by_value(z, 1e-6, 1.0)
        azimuth_angles_prime = (2.*azimuth_spread/1.4)*(tf.sqrt(-tf.math.log(z)
                                                                )/c_phi)

        # Introducing random variation
        random_sign = tf.random.uniform(shape=[batch_size, num_bs, 1,
            num_clusters_max], minval=0, maxval=2, dtype=tf.int32)
        random_sign = 2*random_sign - 1
        random_sign = tf.cast(random_sign, self._scenario.dtype.real_dtype)
        random_comp = tf.random.normal(shape=[batch_size, num_bs, num_ut,
            num_clusters_max], mean=0.0, stddev=azimuth_spread/7.0,
            dtype=self._scenario.dtype.real_dtype)
        azimuth_angles = (random_sign*azimuth_angles_prime + random_comp
            + azimuth_angles_los)
        azimuth_angles = (azimuth_angles -
            tf.where(tf.expand_dims(scenario.los, axis=3),
            random_sign[:,:,:,:1]*azimuth_angles_prime[:,:,:,:1]
            + random_comp[:,:,:,:1], 0.0))

        # Add offset angles to cluster angles to get the ray angles
        ray_offsets = self._ray_offsets[:scenario.rays_per_cluster]
        # Add dimensions for batch size, num bs, num ut, num clusters
        ray_offsets = tf.reshape(ray_offsets, [1,1,1,1,
                                                scenario.rays_per_cluster])
        # Rays angles
        azimuth_angles = tf.expand_dims(azimuth_angles, axis=4)
        azimuth_angles = azimuth_angles + cluster_angle_spread*ray_offsets

        # Wrapping to (-180, 180)
        azimuth_angles = wrap_angle_0_360(azimuth_angles)
        azimuth_angles = tf.where(tf.math.greater(azimuth_angles, 180.),
            azimuth_angles-360., azimuth_angles)

        return azimuth_angles

    def _azimuth_angles_of_arrival(self, azimuth_spread_arrival,
                                   rician_k_factor, cluster_powers):
        # pylint: disable=line-too-long
        """
        Compute the azimuth angle of arrival (AoA)
        See step 7 of section 7.5 from TR 38.901 specification.

        Input
        ------
        azimuth_spread_arrival : [batch size, num of BSs, num of UTs], tf.float
            Azimuth angle spread of arrival (ASA) [deg]

        rician_k_factor : [batch size, num of BSs, num of UTs], tf.float
            Rician K-factor of each BS-UT link. Used only for LoS links.

        cluster_powers : [batch size, num of BSs, num of UTs, maximum number of clusters], tf.float
            Normalized path powers

        Output
        -------
        aoa : [batch size, num of BSs, num of UTs, maximum number of clusters, number of rays], tf.float
            Paths azimuth angles of arrival (AoA) wrapped within (-180,180) [degree]
        """
        return self._azimuth_angles(azimuth_spread_arrival,
                                    rician_k_factor, cluster_powers, 'aoa')

    def _azimuth_angles_of_departure(self, azimuth_spread_departure,
                                     rician_k_factor, cluster_powers):
        # pylint: disable=line-too-long
        """
        Compute the azimuth angle of departure (AoD)
        See step 7 of section 7.5 from TR 38.901 specification.

        Input
        ------
        azimuth_spread_departure : [batch size, num of BSs, num of UTs], tf.float
            Azimuth angle spread of departure (ASD) [deg]

        rician_k_factor : [batch size, num of BSs, num of UTs], tf.float
            Rician K-factor of each BS-UT link. Used only for LoS links.

        cluster_powers : [batch size, num of BSs, num of UTs, maximum number of clusters], tf.float
            Normalized path powers

        Output
        -------
        aod : [batch size, num of BSs, num of UTs, maximum number of clusters, number of rays], tf.float
            Paths azimuth angles of departure (AoD) wrapped within (-180,180) [degree]
        """
        return self._azimuth_angles(azimuth_spread_departure,
                                    rician_k_factor, cluster_powers, 'aod')

    def _zenith_angles(self, zenith_spread, rician_k_factor, cluster_powers,
                       angle_type):
        # pylint: disable=line-too-long
        """
        Generate departure or arrival zenith angles (degrees).
        See step 7 of section 7.5 from TR 38.901 specification.

        Input
        ------
        zenith_spread : [batch size, num of BSs, num of UTs], tf.float
            Angle spread, (ZSD or ZSA) depending on ``angle_type`` [deg]

        rician_k_factor : [batch size, num of BSs, num of UTs], tf.float
            Rician K-factor of each BS-UT link. Used only for LoS links.

        cluster_powers : [batch size, num of BSs, num of UTs, maximum number of clusters], tf.float
            Normalized path powers

        angle_type : str
            Type of angle to compute. Must be 'zoa' or 'zod'.

        Output
        -------
        zenith_angles : [batch size, num of BSs, num of UTs, maximum number of clusters, number of rays], tf.float
            Paths zenith angles wrapped within (0,180) [degree]. Either the ZoA or ZoD depending on ``angle_type``.
        """

        scenario = self._scenario

        batch_size = scenario.batch_size
        num_bs = scenario.num_bs
        num_ut = scenario.num_ut

        # Tensors giving UTs states
        los = scenario.los
        indoor_uts = tf.expand_dims(scenario.indoor, axis=1)
        los_uts = tf.logical_and(los, tf.logical_not(indoor_uts))
        nlos_uts = tf.logical_and(tf.logical_not(los),
                            tf.logical_not(indoor_uts))

        num_clusters_max = scenario.num_clusters_max

        # Adding cluster dimension for broadcasting
        zenith_spread = tf.expand_dims(zenith_spread, axis=3)
        rician_k_factor = tf.expand_dims(rician_k_factor, axis=3)
        indoor_uts = tf.expand_dims(indoor_uts, axis=3)
        los_uts = tf.expand_dims(los_uts, axis=3)
        nlos_uts = tf.expand_dims(nlos_uts, axis=3)

        # Loading angle spread
        if angle_type == 'zod':
            zenith_angles_los = scenario.los_zod
            cluster_angle_spread = (3./8.)*tf.math.pow(tf.constant(10.,
                self._scenario.dtype.real_dtype),
                scenario.lsp_log_mean[:,:,:,6])
        else:
            cluster_angle_spread = scenario.get_param('cZSA')
            zenith_angles_los = scenario.los_zoa
        zod_offset = scenario.zod_offset
        # Adding cluster dimension for broadcasting
        zod_offset = tf.expand_dims(zod_offset, axis=3)
        zenith_angles_los = tf.expand_dims(zenith_angles_los, axis=3)
        cluster_angle_spread = tf.expand_dims(cluster_angle_spread, axis=3)

        # Compute the C_theta
        rician_k_factor_db = 10.0*log10(rician_k_factor) # to dB
        c_theta_nlos = tf.expand_dims(scenario.get_param("CThetaNLoS"),axis=3)
        c_theta_los = c_theta_nlos*(1.3086 + 0.0339*rician_k_factor_db
            - 0.0077*tf.square(rician_k_factor_db)
            + 0.0002*tf.math.pow(rician_k_factor_db, 3.))
        c_theta = tf.where(los_uts, c_theta_los, c_theta_nlos)

        # Inverse Laplacian function
        z = cluster_powers/tf.reduce_max(cluster_powers, axis=3, keepdims=True)
        z = tf.clip_by_value(z, 1e-6, 1.0)
        zenith_angles_prime = (-zenith_spread*tf.math.log(z)/c_theta)

        # Random component
        random_sign = tf.random.uniform(shape=[batch_size, num_bs, 1,
            num_clusters_max], minval=0, maxval=2, dtype=tf.int32)
        random_sign = 2*random_sign - 1
        random_sign = tf.cast(random_sign, self._scenario.dtype.real_dtype)
        random_comp = tf.random.normal(shape=[batch_size, num_bs, num_ut,
            num_clusters_max], mean=0.0, stddev=zenith_spread/7.0,
            dtype=self._scenario.dtype.real_dtype)

        # The center cluster angles depend on the UT scenario
        zenith_angles = random_sign*zenith_angles_prime + random_comp
        los_additinoal_comp = -(random_sign[:,:,:,:1]*
            zenith_angles_prime[:,:,:,:1] + random_comp[:,:,:,:1]
            - zenith_angles_los)
        if angle_type == 'zod':
            additional_comp = tf.where(los_uts, los_additinoal_comp,
                zenith_angles_los + zod_offset)
        else:
            additional_comp = tf.where(los_uts, los_additinoal_comp,
                0.0)
            additional_comp = tf.where(nlos_uts, zenith_angles_los,
                additional_comp)
            additional_comp = tf.where(indoor_uts, tf.constant(90.0,
                self._scenario.dtype.real_dtype),
                additional_comp)
        zenith_angles = zenith_angles + additional_comp

        # Generating rays for every cluster
        # Add offset angles to cluster angles to get the ray angles
        ray_offsets = self._ray_offsets[:scenario.rays_per_cluster]
        # # Add dimensions for batch size, num bs, num ut, num clusters
        ray_offsets = tf.reshape(ray_offsets, [1,1,1,1,
                                                scenario.rays_per_cluster])
        # Adding ray dimension for broadcasting
        zenith_angles = tf.expand_dims(zenith_angles, axis=4)
        cluster_angle_spread = tf.expand_dims(cluster_angle_spread, axis=4)
        zenith_angles = zenith_angles + cluster_angle_spread*ray_offsets

        # Wrapping to (0, 180)
        zenith_angles = wrap_angle_0_360(zenith_angles)
        zenith_angles = tf.where(tf.math.greater(zenith_angles, 180.),
            360.-zenith_angles, zenith_angles)

        return zenith_angles

    def _zenith_angles_of_arrival(self, zenith_spread_arrival, rician_k_factor,
        cluster_powers):
        # pylint: disable=line-too-long
        """
        Compute the zenith angle of arrival (ZoA)
        See step 7 of section 7.5 from TR 38.901 specification.

        Input
        ------
        zenith_spread_arrival : [batch size, num of BSs, num of UTs], tf.float
            Zenith angle spread of arrival (ZSA) [deg]

        rician_k_factor : [batch size, num of BSs, num of UTs], tf.float
            Rician K-factor of each BS-UT link. Used only for LoS links.

        cluster_powers : [batch size, num of BSs, num of UTs, maximum number of clusters], tf.float
            Normalized path powers

        Output
        -------
        zoa : [batch size, num of BSs, num of UTs, maximum number of clusters, number of rays], tf.float
            Paths zenith angles of arrival (ZoA) wrapped within (0,180) [degree]
        """
        return self._zenith_angles(zenith_spread_arrival, rician_k_factor,
                                   cluster_powers, 'zoa')

    def _zenith_angles_of_departure(self, zenith_spread_departure,
                                    rician_k_factor, cluster_powers):
        # pylint: disable=line-too-long
        """
        Compute the zenith angle of departure (ZoD)
        See step 7 of section 7.5 from TR 38.901 specification.

        Input
        ------
        zenith_spread_departure : [batch size, num of BSs, num of UTs], tf.float
            Zenith angle spread of departure (ZSD) [deg]

        rician_k_factor : [batch size, num of BSs, num of UTs], tf.float
            Rician K-factor of each BS-UT link. Used only for LoS links.

        cluster_powers : [batch size, num of BSs, num of UTs, maximum number of clusters], tf.float
            Normalized path powers

        Output
        -------
        zod : [batch size, num of BSs, num of UTs, maximum number of clusters, number of rays], tf.float
            Paths zenith angles of departure (ZoD) wrapped within (0,180) [degree]
        """
        return self._zenith_angles(zenith_spread_departure, rician_k_factor,
                                   cluster_powers, 'zod')

    def _shuffle_angles(self, angles):
        # pylint: disable=line-too-long
        """
        Randomly shuffle a tensor carrying azimuth/zenith angles
        of arrival/departure.

        Input
        ------
        angles : [batch size, num of BSs, num of UTs, maximum number of clusters, number of rays], tf.float
            Angles to shuffle

        Output
        -------
        shuffled_angles : [batch size, num of BSs, num of UTs, maximum number of clusters, number of rays], tf.float
            Shuffled ``angles``
        """

        scenario = self._scenario

        batch_size = scenario.batch_size
        num_bs = scenario.num_bs
        num_ut = scenario.num_ut

        # Create randomly shuffled indices by arg-sorting samples from a random
        # normal distribution
        random_numbers = tf.random.normal([batch_size, num_bs, 1,
                scenario.num_clusters_max, scenario.rays_per_cluster])
        shuffled_indices = tf.argsort(random_numbers)
        shuffled_indices = tf.tile(shuffled_indices, [1, 1, num_ut, 1, 1])
        # Shuffling the angles
        shuffled_angles = tf.gather(angles,shuffled_indices, batch_dims=4)
        return shuffled_angles

    def _random_coupling(self, aoa, aod, zoa, zod):
        # pylint: disable=line-too-long
        """
        Randomly couples the angles within a cluster for both azimuth and
        elevation.

        Step 8 in TR 38.901 specification.

        Input
        ------
        aoa : [batch size, num of BSs, num of UTs, maximum number of clusters, number of rays], tf.float
            Paths azimuth angles of arrival [degree] (AoA)

        aod : [batch size, num of BSs, num of UTs, maximum number of clusters, number of rays], tf.float
            Paths azimuth angles of departure (AoD) [degree]

        zoa : [batch size, num of BSs, num of UTs, maximum number of clusters, number of rays], tf.float
            Paths zenith angles of arrival [degree] (ZoA)

        zod : [batch size, num of BSs, num of UTs, maximum number of clusters, number of rays], tf.float
            Paths zenith angles of departure [degree] (ZoD)

        Output
        -------
        shuffled_aoa : [batch size, num of BSs, num of UTs, maximum number of clusters, number of rays], tf.float
            Shuffled `aoa`

        shuffled_aod : [batch size, num of BSs, num of UTs, maximum number of clusters, number of rays], tf.float
            Shuffled `aod`

        shuffled_zoa : [batch size, num of BSs, num of UTs, maximum number of clusters, number of rays], tf.float
            Shuffled `zoa`

        shuffled_zod : [batch size, num of BSs, num of UTs, maximum number of clusters, number of rays], tf.float
            Shuffled `zod`
        """
        shuffled_aoa = self._shuffle_angles(aoa)
        shuffled_aod = self._shuffle_angles(aod)
        shuffled_zoa = self._shuffle_angles(zoa)
        shuffled_zod = self._shuffle_angles(zod)

        return shuffled_aoa, shuffled_aod, shuffled_zoa, shuffled_zod

    def _cross_polarization_power_ratios(self):
        # pylint: disable=line-too-long
        """
        Generate cross-polarization power ratios.

        Step 9 in TR 38.901 specification.

        Input
        ------
        None

        Output
        -------
        cross_polarization_power_ratios : [batch size, num of BSs, num of UTs, maximum number of clusters, number of rays], tf.float
            Polarization power ratios
        """

        scenario = self._scenario

        batch_size = scenario.batch_size
        num_bs = scenario.num_bs
        num_ut = scenario.num_ut
        num_clusters = scenario.num_clusters_max
        num_rays_per_cluster = scenario.rays_per_cluster

        # Loading XPR mean and standard deviation
        mu_xpr = scenario.get_param("muXPR")
        std_xpr = scenario.get_param("sigmaXPR")
        # Expanding for broadcasting with clusters and rays dims
        mu_xpr = tf.expand_dims(tf.expand_dims(mu_xpr, axis=3), axis=4)
        std_xpr = tf.expand_dims(tf.expand_dims(std_xpr, axis=3), axis=4)

        # XPR are assumed to follow a log-normal distribution.
        # Generate XPR in log-domain
        x = tf.random.normal(shape=[batch_size, num_bs, num_ut, num_clusters,
            num_rays_per_cluster], mean=mu_xpr, stddev=std_xpr,
            dtype=self._scenario.dtype.real_dtype)
        # To linear domain
        cross_polarization_power_ratios = tf.math.pow(tf.constant(10.,
            self._scenario.dtype.real_dtype), x/10.0)
        return cross_polarization_power_ratios
