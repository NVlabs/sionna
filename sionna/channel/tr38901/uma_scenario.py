#
# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""3GPP TR39.801 urban macrocell (UMa) channel model"""

import tensorflow as tf

from sionna import SPEED_OF_LIGHT
from sionna.utils import log10
from . import SystemLevelScenario


class UMaScenario(SystemLevelScenario):
    r"""
    3GPP TR 38.901 urban macrocell (UMa) channel model scenario.

    Parameters
    -----------
    carrier_frequency : float
        Carrier frequency [Hz]

    o2i_model : str
        Outdoor to indoor (O2I) pathloss model, used for indoor UTs.
        Either "low" or "high" (see section 7.4.3 from 38.901 specification)

    ut_array : PanelArray
        Panel array configuration used by UTs

    bs_array : PanelArray
        Panel array configuration used by BSs

    direction : str
        Link direction. Either "uplink" or "downlink"

    enable_pathloss : bool
        If set to `True`, apply pathloss. Otherwise, does not. Defaults to True.

    enable_shadow_fading : bool
        If set to `True`, apply shadow fading. Otherwise, does not.
        Defaults to True.

    dtype : tf.DType
        Defines the datatype for internal calculations and the output
        dtype. Defaults to `tf.complex64`.
    """

    #########################################
    # Public methods and properties
    #########################################

    def clip_carrier_frequency_lsp(self, fc):
        r"""Clip the carrier frequency ``fc`` in GHz for LSP calculation

        Input
        -----
        fc : float
            Carrier frequency [GHz]

        Output
        -------
        : float
            Clipped carrier frequency, that should be used for LSp computation
        """
        if fc < 6.:
            fc = tf.cast(6., self._dtype.real_dtype)
        return fc

    @property
    def min_2d_in(self):
        r"""Minimum indoor 2D distance for indoor UTs [m]"""
        return tf.constant(0.0, self._dtype.real_dtype)

    @property
    def max_2d_in(self):
        r"""Maximum indoor 2D distance for indoor UTs [m]"""
        return tf.constant(25.0, self._dtype.real_dtype)

    @property
    def los_probability(self):
        r"""Probability of each UT to be LoS. Used to randomly generate LoS
        status of outdoor UTs.

        Computed following section 7.4.2 of TR 38.901.

        [batch size, num_ut]"""

        h_ut = self.h_ut
        c = tf.math.pow((h_ut-13.)/10., 1.5)
        c = tf.where(tf.math.less(h_ut, 13.0),
                        tf.constant(0., self._dtype.real_dtype), c)
        c = tf.expand_dims(c, axis=1)

        distance_2d_out = self._distance_2d_out
        los_probability = ((18.0/distance_2d_out
            + tf.math.exp(-distance_2d_out/63.0)*(1.-18./distance_2d_out))
            *(1.+c*5./4.*tf.math.pow(distance_2d_out/100., 3)
                *tf.math.exp(-distance_2d_out/150.0)))

        los_probability = tf.where(tf.math.less(distance_2d_out, 18.0),
            tf.constant(1.0, self._dtype.real_dtype), los_probability)
        return los_probability

    @property
    def rays_per_cluster(self):
        r"""Number of rays per cluster"""
        return tf.constant(20, tf.int32)

    @property
    def los_parameter_filepath(self):
        r""" Path of the configuration file for LoS scenario"""
        return "UMa_LoS.json"

    @property
    def nlos_parameter_filepath(self):
        r""" Path of the configuration file for NLoS scenario"""
        return"UMa_NLoS.json"

    @property
    def o2i_parameter_filepath(self):
        r""" Path of the configuration file for indoor scenario"""
        return "UMa_O2I.json"

    #########################
    # Utility methods
    #########################

    def _compute_lsp_log_mean_std(self):
        r"""Computes the mean and standard deviations of LSPs in log-domain"""

        batch_size = self.batch_size
        num_bs = self.num_bs
        num_ut = self.num_ut
        distance_2d = self.distance_2d
        h_bs = self.h_bs
        h_bs = tf.expand_dims(h_bs, axis=2) # For broadcasting
        h_ut = self.h_ut
        h_ut = tf.expand_dims(h_ut, axis=1) # For broadcasting

        ## Mean
        # DS
        log_mean_ds = self.get_param("muDS")
        # ASD
        log_mean_asd = self.get_param("muASD")
        # ASA
        log_mean_asa = self.get_param("muASA")
        # SF.  Has zero-mean.
        log_mean_sf = tf.zeros([batch_size, num_bs, num_ut],
                                self._dtype.real_dtype)
        # K.  Given in dB in the 3GPP tables, hence the division by 10
        log_mean_k = self.get_param("muK")/10.0
        # ZSA
        log_mean_zsa = self.get_param("muZSA")
        # ZSD
        log_mean_zsd_los = tf.math.maximum(tf.constant(-0.5,
                                                    self._dtype.real_dtype),
            -2.1*(distance_2d/1000.0) - 0.01*tf.abs(h_ut-1.5)+0.75)
        log_mean_zsd_nlos = tf.math.maximum(tf.constant(-0.5,
                                                    self._dtype.real_dtype),
            -2.1*(distance_2d/1000.0) - 0.01*tf.abs(h_ut-1.5)+0.9)
        log_mean_zsd = tf.where(self.los, log_mean_zsd_los, log_mean_zsd_nlos)

        lsp_log_mean = tf.stack([log_mean_ds,
                                log_mean_asd,
                                log_mean_asa,
                                log_mean_sf,
                                log_mean_k,
                                log_mean_zsa,
                                log_mean_zsd], axis=3)

        ## STD
        # DS
        log_std_ds = self.get_param("sigmaDS")
        # ASD
        log_std_asd = self.get_param("sigmaASD")
        # ASA
        log_std_asa = self.get_param("sigmaASA")
        # SF. Given in dB in the 3GPP tables, hence the division by 10
        # O2I and NLoS cases just require the use of a predefined value
        log_std_sf = self.get_param("sigmaSF")/10.0
        # K. Given in dB in the 3GPP tables, hence the division by 10.
        log_std_k = self.get_param("sigmaK")/10.0
        # ZSA
        log_std_zsa = self.get_param("sigmaZSA")
        # ZSD
        log_std_zsd = self.get_param("sigmaZSD")

        lsp_log_std = tf.stack([log_std_ds,
                               log_std_asd,
                               log_std_asa,
                               log_std_sf,
                               log_std_k,
                               log_std_zsa,
                               log_std_zsd], axis=3)

        self._lsp_log_mean = lsp_log_mean
        self._lsp_log_std = lsp_log_std

        # ZOD offset
        fc = self.carrier_frequency/1e9
        if fc < 6.:
            fc = tf.cast(6., self._dtype.real_dtype)
        a = 0.208*log10(fc)-0.782
        b = tf.constant(25., self._dtype.real_dtype)
        c = -0.13*log10(fc)+2.03
        e = 7.66*log10(fc)-5.96
        zod_offset =(e-tf.math.pow(tf.constant(10., self._dtype.real_dtype),
            a*log10(tf.maximum(b, distance_2d)) + c - 0.07*(h_ut-1.5)))
        zod_offset = tf.where(self.los,
                            tf.constant(0., self._dtype.real_dtype),zod_offset)
        self._zod_offset = zod_offset

    def _compute_pathloss_basic(self):
        r"""Computes the basic component of the pathloss [dB]"""

        batch_size = self.batch_size
        num_bs = self.num_bs
        num_ut = self.num_ut
        distance_2d = self.distance_2d
        distance_3d = self.distance_3d
        fc = self.carrier_frequency # Carrier frequency (Hz)
        h_bs = self.h_bs
        h_bs = tf.expand_dims(h_bs, axis=2) # For broadcasting
        h_ut = self.h_ut
        h_ut = tf.expand_dims(h_ut, axis=1) # For broadcasting

        # Beak point distance
        g = ((5./4.)*tf.math.pow(distance_2d/100., 3.)
            *tf.math.exp(-distance_2d/150.0))
        g = tf.where(tf.math.less(distance_2d, 18.),
            tf.constant(0.0, self._dtype.real_dtype), g)
        c = g*tf.math.pow((h_ut-13.)/10., 1.5)
        c = tf.where(tf.math.less(h_ut, 13.),
            tf.constant(0., self._dtype.real_dtype), c)
        p = 1./(1.+c)
        r = tf.random.uniform(shape=[batch_size, num_bs, num_ut], minval=0.0,
            maxval=1.0, dtype=self._dtype.real_dtype)
        r = tf.where(tf.math.less(r, p),
            tf.constant(1.0, self._dtype.real_dtype),
            tf.constant(0.0, self._dtype.real_dtype))

        max_value = h_ut- 1.5
        # Random uniform integer generation is not supported when maxval and
        # are not scalar. The following commented would therefore not work.
        # Therefore, for now, we just sample from a continuous
        # distribution.
        #delta = tf.cast(tf.math.floor((max_value-min_value)/3.), tf.int32)
        #s = tf.random.uniform(shape=[batch_size, num_bs, num_ut],
        #    minval=0, maxval=delta+1, dtype=tf.int32)
        #s = tf.cast(s, tf.float32)*3.+min_value
        s = tf.random.uniform(shape=[batch_size, num_bs, num_ut],
           minval=12., maxval=max_value, dtype=self._dtype.real_dtype)
        # Itc could happen that h_ut = 13m, and therefore max_value < 13m
        s = tf.where(tf.math.less(s, 12.0),
            tf.constant(12.0, self._dtype.real_dtype), s)

        h_e = r + (1.-r)*s
        h_bs_prime = h_bs - h_e
        h_ut_prime = h_ut - h_e
        distance_breakpoint = 4*h_bs_prime*h_ut_prime*fc/SPEED_OF_LIGHT

        ## Basic path loss for LoS

        pl_1 = 28.0 + 22.0*log10(distance_3d) + 20.0*log10(fc/1e9)
        pl_2 = (28.0 + 40.0*log10(distance_3d) + 20.0*log10(fc/1e9)
         - 9.0*log10(tf.square(distance_breakpoint)+tf.square(h_bs-h_ut)))
        pl_los = tf.where(tf.math.less(distance_2d, distance_breakpoint),
            pl_1, pl_2)

        ## Basic pathloss for NLoS and O2I

        pl_3 = (13.54 + 39.08*log10(distance_3d) + 20.0*log10(fc/1e9)
            - 0.6*(h_ut-1.5))
        pl_nlos = tf.math.maximum(pl_los, pl_3)

        ## Set the basic pathloss according to UT state

        # Expand to allow broadcasting with the BS dimension
        # LoS
        pl_b = tf.where(self.los, pl_los, pl_nlos)

        self._pl_b = pl_b
