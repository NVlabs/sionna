#
# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Class used to define a system level 3GPP channel simulation scenario"""

import json
from importlib_resources import files
import tensorflow as tf
from abc import ABC, abstractmethod

from sionna import SPEED_OF_LIGHT, PI
from sionna.utils import log10
from sionna.channel.utils import sample_bernoulli, rad_2_deg, wrap_angle_0_360
from .antenna import PanelArray

from . import models # pylint: disable=relative-beyond-top-level


class SystemLevelScenario(ABC):
    r"""
    This class is used to set up the scenario for system level 3GPP channel
    simulation.

    Scenarios for system level channel simulation, such as UMi, UMa, or RMa,
    are defined by implementing this base class.

    Input
    ------
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

    def __init__(self, carrier_frequency, o2i_model, ut_array, bs_array,
        direction, enable_pathloss=True, enable_shadow_fading=True,
        dtype=tf.complex64):

        # Carrier frequency (Hz)
        self._carrier_frequency = tf.constant(carrier_frequency,
            dtype.real_dtype)

        # Wavelength (m)
        self._lambda_0 = tf.constant(SPEED_OF_LIGHT/carrier_frequency,
            dtype.real_dtype)

        # O2I model
        assert o2i_model in ('low', 'high'), "o2i_model must be 'low' or 'high'"
        self._o2i_model = o2i_model

        # UTs and BSs arrays
        assert isinstance(ut_array, PanelArray), \
            "'ut_array' must be an instance of PanelArray"
        assert isinstance(bs_array, PanelArray), \
            "'bs_array' must be an instance of PanelArray"
        self._ut_array = ut_array
        self._bs_array = bs_array

        # data type
        assert dtype.is_complex, "'dtype' must be complex type"
        self._dtype = dtype

        # Direction
        assert direction in ("uplink", "downlink"), \
            "'direction' must be 'uplink' or 'downlink'"
        self._direction = direction

        # Pathloss and shadow fading
        self._enable_pathloss = enable_pathloss
        self._enable_shadow_fading = enable_shadow_fading

        # Scenario
        self._ut_loc = None
        self._bs_loc = None
        self._ut_orientations = None
        self._bs_orientations = None
        self._ut_velocities = None
        self._in_state = None
        self._requested_los = None

        # Load parameters for this scenario
        self._load_params()

    @property
    def carrier_frequency(self):
        r"""Carrier frequency [Hz]"""
        return self._carrier_frequency

    @property
    def direction(self):
        r"""Direction of communication. Either "uplink" or "downlink"."""
        return self._direction

    @property
    def pathloss_enabled(self):
        r"""`True` is pathloss is enabled. `False` otherwise."""
        return self._enable_pathloss

    @property
    def shadow_fading_enabled(self):
        r"""`True` is shadow fading is enabled. `False` otherwise."""
        return self._enable_shadow_fading

    @property
    def lambda_0(self):
        r"""Wavelength [m]"""
        return self._lambda_0

    @property
    def batch_size(self):
        """Batch size"""
        return tf.shape(self._ut_loc)[0]

    @property
    def num_ut(self):
        """Number of UTs."""
        return tf.shape(self._ut_loc)[1]

    @property
    def num_bs(self):
        """
        Number of BSs.
        """
        return tf.shape(self._bs_loc)[1]

    @property
    def h_ut(self):
        r"""Heigh of UTs [m]. [batch size, number of UTs]"""
        return self._ut_loc[:,:,2]

    @property
    def h_bs(self):
        r"""Heigh of BSs [m].[batch size, number of BSs]"""
        return self._bs_loc[:,:,2]

    @property
    def ut_loc(self):
        r"""Locations of UTs [m]. [batch size, number of UTs, 3]"""
        return self._ut_loc

    @property
    def bs_loc(self):
        r"""Locations of BSs [m]. [batch size, number of BSs, 3]"""
        return self._bs_loc

    @property
    def ut_orientations(self):
        r"""Orientations of UTs [radian]. [batch size, number of UTs, 3]"""
        return self._ut_orientations

    @property
    def bs_orientations(self):
        r"""Orientations of BSs [radian]. [batch size, number of BSs, 3]"""
        return self._bs_orientations

    @property
    def ut_velocities(self):
        r"""UTs velocities [m/s]. [batch size, number of UTs, 3]"""
        return self._ut_velocities

    @property
    def ut_array(self):
        r"""PanelArray used by UTs."""
        return self._ut_array

    @property
    def bs_array(self):
        r"""PanelArray used by BSs."""
        return self._bs_array

    @property
    def indoor(self):
        r"""
        Indoor state of UTs. `True` is indoor, `False` otherwise.
        [batch size, number of UTs]"""
        return self._in_state

    @property
    def los(self):
        r"""LoS state of BS-UT links. `True` if LoS, `False` otherwise.
        [batch size, number of BSs, number of UTs]"""
        return self._los

    @property
    def distance_2d(self):
        r"""
        Distance between each UT and each BS in the X-Y plan [m].
        [batch size, number of BSs, number of UTs]"""
        return self._distance_2d

    @property
    def distance_2d_in(self):
        r"""Indoor distance between each UT and BS in the X-Y plan [m], i.e.,
        part of the total distance that corresponds to indoor propagation in the
        X-Y plan.
        Set to 0 for UTs located ourdoor.
        [batch size, number of BSs, number of UTs]"""
        return self._distance_2d_in

    @property
    def distance_2d_out(self):
        r"""Outdoor distance between each UT and BS in the X-Y plan [m], i.e.,
        part of the total distance that corresponds to outdoor propagation in
        the X-Y plan.
        Equals to ``distance_2d`` for UTs located outdoor.
        [batch size, number of BSs, number of UTs]"""
        return self._distance_2d_out

    @property
    def distance_3d(self):
        r"""
        Distance between each UT and each BS [m].
        [batch size, number of BSs, number of UTs]"""
        return self._distance_2d

    @property
    def distance_3d_in(self):
        r"""Indoor distance between each UT and BS [m], i.e.,
        part of the total distance that corresponds to indoor propagation.
        Set to 0 for UTs located ourdoor.
        [batch size, number of BSs, number of UTs]"""
        return self._distance_3d_in

    @property
    def distance_3d_out(self):
        r"""Outdoor distance between each UT and BS [m], i.e.,
        part of the total distance that corresponds to outdoor propagation.
        Equals to ``distance_3d`` for UTs located outdoor.
        [batch size, number of BSs, number of UTs]"""
        return self._distance_3d_out

    @property
    def matrix_ut_distance_2d(self):
        r"""Distance between all pairs for UTs in the X-Y plan [m].
        [batch size, number of UTs, number of UTs]"""
        return self._matrix_ut_distance_2d

    @property
    def los_aod(self):
        r"""LoS azimuth angle of departure of each BS-UT link [deg].
        [batch size, number of BSs, number of UTs]"""
        return self._los_aod

    @property
    def los_aoa(self):
        r"""LoS azimuth angle of arrival of each BS-UT link [deg].
        [batch size, number of BSs, number of UTs]"""
        return self._los_aoa

    @property
    def los_zod(self):
        r"""LoS zenith angle of departure of each BS-UT link [deg].
        [batch size, number of BSs, number of UTs]"""
        return self._los_zod

    @property
    def los_zoa(self):
        r"""LoS zenith angle of arrival of each BS-UT link [deg].
        [batch size, number of BSs, number of UTs]"""
        return self._los_zoa

    @property
    @abstractmethod
    def los_probability(self):
        r"""Probability of each UT to be LoS. Used to randomly generate LoS
        status of outdoor UTs. [batch size, number of UTs]"""
        pass

    @property
    @abstractmethod
    def min_2d_in(self):
        r"""Minimum indoor 2D distance for indoor UTs [m]"""
        pass

    @property
    @abstractmethod
    def max_2d_in(self):
        r"""Maximum indoor 2D distance for indoor UTs [m]"""
        pass

    @property
    def lsp_log_mean(self):
        r"""
        Mean of LSPs in the log domain.
        [batch size, number of BSs, number of UTs, 7].
        The last dimension corresponds to the LSPs, in the following order:
        DS - ASD - ASA - SF - K - ZSA - ZSD - XPR"""
        return self._lsp_log_mean

    @property
    def lsp_log_std(self):
        r"""
        STD of LSPs in the log domain.
        [batch size, number of BSs, number of UTs, 7].
        The last dimension corresponds to the LSPs, in the following order:
        DS - ASD - ASA - SF - K - ZSA - ZSD - XPR"""
        return self._lsp_log_std

    @property
    @abstractmethod
    def rays_per_cluster(self):
        r"""Number of rays per cluster"""
        pass

    @property
    def zod_offset(self):
        r"""Zenith angle of departure offset"""
        return self._zod_offset

    @property
    def num_clusters_los(self):
        r"""Number of clusters for LoS scenario"""
        return self._params_los["numClusters"]

    @property
    def num_clusters_nlos(self):
        r"""Number of clusters for NLoS scenario"""
        return self._params_nlos["numClusters"]

    @property
    def num_clusters_indoor(self):
        r"""Number of clusters indoor scenario"""
        return self._params_o2i["numClusters"]

    @property
    def num_clusters_max(self):
        r"""Maximum number of clusters over indoor, LoS, and NLoS scenarios"""
        # Different models have different number of clusters
        num_clusters_los = self._params_los["numClusters"]
        num_clusters_nlos = self._params_nlos["numClusters"]
        num_clusters_o2i = self._params_o2i["numClusters"]
        num_clusters_max = tf.reduce_max([num_clusters_los, num_clusters_nlos,
            num_clusters_o2i])
        return num_clusters_max

    @property
    def basic_pathloss(self):
        r"""Basic pathloss component [dB].
        See section 7.4.1 of 38.901 specification.
        [batch size, num BS, num UT]"""
        return self._pl_b

    def set_topology(self, ut_loc=None, bs_loc=None, ut_orientations=None,
        bs_orientations=None, ut_velocities=None, in_state=None, los=None):
        r"""
        Set the network topology.

        It is possible to set up a different network topology for each batch
        example.

        When calling this function, not specifying a parameter leads to the
        reuse of the previously given value. Not specifying a value that was not
        set at a former call rises an error.

        Input
        ------
            ut_loc : [batch size, number of UTs, 3], tf.float
                Locations of the UTs [m]

            bs_loc : [batch size, number of BSs, 3], tf.float
                Locations of BSs [m]

            ut_orientations : [batch size, number of UTs, 3], tf.float
                Orientations of the UTs arrays [radian]

            bs_orientations : [batch size, number of BSs, 3], tf.float
                Orientations of the BSs arrays [radian]

            ut_velocities : [batch size, number of UTs, 3], tf.float
                Velocity vectors of UTs [m/s]

            in_state : [batch size, number of UTs], tf.bool
                Indoor/outdoor state of UTs. `True` means indoor and `False`
                means outdoor.

            los : tf.bool or `None`
                If not `None` (default value), all UTs located outdoor are
                forced to be in LoS if ``los`` is set to `True`, or in NLoS
                if it is set to `False`. If set to `None`, the LoS/NLoS states
                of UTs is set following 3GPP specification
                (Section 7.4.2 of TR 38.901).
        """

        assert (ut_loc is not None) or (self._ut_loc is not None),\
            "`ut_loc` is None and was not previously set"

        assert (bs_loc is not None) or (self._bs_loc is not None),\
            "`bs_loc` is None and was not previously set"

        assert (in_state is not None) or (self._in_state is not None),\
            "`in_state` is None and was not previously set"

        assert (ut_orientations is not None)\
            or (self._ut_orientations is not None),\
            "`ut_orientations` is None and was not previously set"

        assert (bs_orientations is not None)\
            or (self._bs_orientations is not None),\
            "`bs_orientations` is None and was not previously set"

        assert (ut_velocities is not None)\
            or (self._ut_velocities is not None),\
            "`ut_velocities` is None and was not previously set"

        # Boolean used to keep track of whether or not we need to (re-)compute
        # the distances between users, correlation matrices...
        # This is required if the UT locations, BS locations, indoor/outdoor
        # state of UTs, or LoS/NLoS states of outdoor UTs are updated.
        need_for_update = False

        if ut_loc is not None:
            self._ut_loc = ut_loc
            need_for_update = True

        if bs_loc is not None:
            self._bs_loc = bs_loc
            need_for_update = True

        if bs_orientations is not None:
            self._bs_orientations = bs_orientations

        if ut_orientations is not None:
            self._ut_orientations = ut_orientations

        if ut_velocities is not None:
            self._ut_velocities = ut_velocities

        if in_state is not None:
            self._in_state = in_state
            need_for_update = True

        if los is not None:
            self._requested_los = los
            need_for_update = True

        if need_for_update:
            # Update topology-related quantities
            self._compute_distance_2d_3d_and_angles()
            self._sample_indoor_distance()
            self._sample_los()

            # Compute the LSPs means and stds
            self._compute_lsp_log_mean_std()

            # Compute the basic path-loss
            self._compute_pathloss_basic()

        return need_for_update

    def spatial_correlation_matrix(self, correlation_distance):
        r"""Computes and returns a 2D spatial exponential correlation matrix
        :math:`C` over the UTs, such that :math:`C`has shape
        (number of UTs)x(number of UTs), and

        .. math::
            C_{n,m} = \exp{-\frac{d_{n,m}}{D}}

        where :math:`d_{n,m}` is the distance between UT :math:`n` and UT
        :math:`m` in the X-Y plan, and :math:`D` the correlation distance.

        Input
        ------
        correlation_distance : float
            Correlation distance, i.e., distance such that the correlation
            is :math:`e^{-1} \approx 0.37`

        Output
        --------
        : [batch size, number of UTs, number of UTs], float
            Spatial correlation :math:`C`
        """
        spatial_correlation_matrix = tf.math.exp(-self.matrix_ut_distance_2d/
                                                 correlation_distance)
        return spatial_correlation_matrix


    @property
    @abstractmethod
    def los_parameter_filepath(self):
        r""" Path of the configuration file for LoS scenario"""
        pass

    @property
    @abstractmethod
    def nlos_parameter_filepath(self):
        r""" Path of the configuration file for NLoS scenario"""
        pass

    @property
    @abstractmethod
    def o2i_parameter_filepath(self):
        r""" Path of the configuration file for indoor scenario"""
        pass

    @property
    def o2i_model(self):
        r"""O2I model used for pathloss computation of indoor UTs. Either "low"
        or "high". See section 7.4.3 or TR 38.901."""
        return self._o2i_model

    @property
    def dtype(self):
        r"""Complex datatype used for internal calculation and tensors"""
        return self._dtype

    @abstractmethod
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
        pass

    def get_param(self, parameter_name):
        r"""
        Given a ``parameter_name`` used in the configuration file, returns a
        tensor with shape [batch size, number of BSs, number of UTs] of the
        parameter value according to each BS-UT link state (LoS, NLoS, indoor).

        Input
        ------
        parameter_name : str
            Name of the parameter used in the configuration file

        Output
        -------
        : [batch size, number of BSs, number of UTs], tf.float
            Parameter value for each BS-UT link
        """

        fc = self._carrier_frequency/1e9
        fc = self.clip_carrier_frequency_lsp(fc)

        parameter_tensor = tf.zeros(shape=[self.batch_size,
                                            self.num_bs,
                                            self.num_ut],
                                            dtype=self._dtype.real_dtype)

        # Parameter value
        if parameter_name in ('muDS', 'sigmaDS', 'muASD', 'sigmaASD', 'muASA',
                             'sigmaASA', 'muZSA', 'sigmaZSA'):

            pa_los = self._params_los[parameter_name + 'a']
            pb_los = self._params_los[parameter_name + 'b']
            pc_los = self._params_los[parameter_name + 'c']

            pa_nlos = self._params_nlos[parameter_name + 'a']
            pb_nlos = self._params_nlos[parameter_name + 'b']
            pc_nlos = self._params_nlos[parameter_name + 'c']

            pa_o2i = self._params_o2i[parameter_name + 'a']
            pb_o2i = self._params_o2i[parameter_name + 'b']
            pc_o2i = self._params_o2i[parameter_name + 'c']

            parameter_value_los = pa_los*log10(pb_los+fc) + pc_los
            parameter_value_nlos = pa_nlos*log10(pb_nlos+fc) + pc_nlos
            parameter_value_o2i = pa_o2i*log10(pb_o2i+fc) + pc_o2i
        elif parameter_name == "cDS":

            pa_los = self._params_los[parameter_name + 'a']
            pb_los = self._params_los[parameter_name + 'b']
            pc_los = self._params_los[parameter_name + 'c']

            pa_nlos = self._params_nlos[parameter_name + 'a']
            pb_nlos = self._params_nlos[parameter_name + 'b']
            pc_nlos = self._params_nlos[parameter_name + 'c']

            pa_o2i = self._params_o2i[parameter_name + 'a']
            pb_o2i = self._params_o2i[parameter_name + 'b']
            pc_o2i = self._params_o2i[parameter_name + 'c']

            parameter_value_los = tf.math.maximum(pa_los,
                pb_los - pc_los*log10(fc))
            parameter_value_nlos = tf.math.maximum(pa_nlos,
                pb_nlos - pc_nlos*log10(fc))
            parameter_value_o2i = tf.math.maximum(pa_o2i,
                pb_o2i - pc_o2i*log10(fc))
        else:
            parameter_value_los = self._params_los[parameter_name]
            parameter_value_nlos = self._params_nlos[parameter_name]
            parameter_value_o2i = self._params_o2i[parameter_name]

        # Expand to allow broadcasting with the BS dimension
        indoor = tf.expand_dims(self.indoor, axis=1)
        # LoS
        parameter_value_los = tf.cast(parameter_value_los,
                                        self._dtype.real_dtype)
        parameter_tensor = tf.where(self.los, parameter_value_los,
            parameter_tensor)
        # NLoS
        parameter_value_nlos = tf.cast(parameter_value_nlos,
                                        self._dtype.real_dtype)
        parameter_tensor = tf.where(
            tf.logical_and(tf.logical_not(self.los),
            tf.logical_not(indoor)), parameter_value_nlos,
            parameter_tensor)
        # O2I
        parameter_value_o2i = tf.cast(parameter_value_o2i,
                                        self._dtype.real_dtype)
        parameter_tensor = tf.where(indoor, parameter_value_o2i,
            parameter_tensor)

        return parameter_tensor

    #####################################################
    # Internal utility methods
    #####################################################

    def _compute_distance_2d_3d_and_angles(self):
        r"""
        Computes the following internal values:
        * 2D distances for all BS-UT pairs in the X-Y plane
        * 3D distances for all BS-UT pairs
        * 2D distances for all pairs of UTs in the X-Y plane
        * LoS AoA, AoD, ZoA, ZoD for all BS-UT pairs

        This function is called at every update of the topology.
        """

        ut_loc = self._ut_loc
        ut_loc = tf.expand_dims(ut_loc, axis=1)

        bs_loc = self._bs_loc
        bs_loc = tf.expand_dims(bs_loc, axis=2)

        delta_loc_xy = ut_loc[:,:,:,:2] - bs_loc[:,:,:,:2]
        delta_loc = ut_loc - bs_loc

        # 2D distances for all BS-UT pairs in the (x-y) plane
        distance_2d = tf.sqrt(tf.reduce_sum(tf.square(delta_loc_xy), axis=3))
        self._distance_2d = distance_2d

        # 3D distances for all BS-UT pairs
        distance_3d = tf.sqrt(tf.reduce_sum(tf.square(delta_loc), axis=3))
        self._distance_3d = distance_3d

        # LoS AoA, AoD, ZoA, ZoD
        los_aod = tf.atan2(delta_loc[:,:,:,1], delta_loc[:,:,:,0])
        los_aoa = los_aod + PI
        los_zod = tf.atan2(distance_2d, delta_loc[:,:,:,2])
        los_zoa = los_zod - PI
        # Angles are converted to degrees and wrapped to (0,360)
        self._los_aod = wrap_angle_0_360(rad_2_deg(los_aod))
        self._los_aoa = wrap_angle_0_360(rad_2_deg(los_aoa))
        self._los_zod = wrap_angle_0_360(rad_2_deg(los_zod))
        self._los_zoa = wrap_angle_0_360(rad_2_deg(los_zoa))

        # 2D distances for all pairs of UTs in the (x-y) plane
        ut_loc_xy = self._ut_loc[:,:,:2]

        ut_loc_xy_expanded_1 = tf.expand_dims(ut_loc_xy, axis=1)
        ut_loc_xy_expanded_2 = tf.expand_dims(ut_loc_xy, axis=2)

        delta_loc_xy = ut_loc_xy_expanded_1 - ut_loc_xy_expanded_2

        matrix_ut_distance_2d = tf.sqrt(tf.reduce_sum(tf.square(delta_loc_xy),
                                                       axis=3))
        self._matrix_ut_distance_2d = matrix_ut_distance_2d

    def _sample_los(self):
        r"""Set the LoS state of each UT randomly, following the procedure
        described in section 7.4.2 of TR 38.901.
        LoS state of each UT is randomly assigned according to a Bernoulli
        distribution, which probability depends on the channel model.
        """
        if self._requested_los is None:
            los_probability = self.los_probability
            los = sample_bernoulli([self.batch_size, self.num_bs,
                                        self.num_ut], los_probability,
                                        self._dtype.real_dtype)
        else:
            los = tf.fill([self.batch_size, self.num_bs, self.num_ut],
                            self._requested_los)

        self._los = tf.logical_and(los,
            tf.logical_not(tf.expand_dims(self._in_state, axis=1)))

    def _sample_indoor_distance(self):
        r"""Sample 2D indoor distances for indoor devices, according to section
        7.4.3.1 of TR 38.901.
        """

        indoor = self.indoor
        indoor = tf.expand_dims(indoor, axis=1) # For broadcasting with BS dim
        indoor_mask = tf.where(indoor, tf.constant(1.0, self._dtype.real_dtype),
            tf.constant(0.0, self._dtype.real_dtype))

        # Sample the indoor 2D distances for each BS-UT link
        self._distance_2d_in = tf.random.uniform(shape=[self.batch_size,
            self.num_bs, self.num_ut], minval=self.min_2d_in,
            maxval=self.max_2d_in, dtype=self._dtype.real_dtype)*indoor_mask
        # Compute the outdoor 2D distances
        self._distance_2d_out = self.distance_2d - self._distance_2d_in
        # Compute the indoor 3D distances
        self._distance_3d_in = ((self._distance_2d_in/self.distance_2d)
            *self.distance_3d)
        # Compute the outdoor 3D distances
        self._distance_3d_out = self.distance_3d - self._distance_3d_in

    def _load_params(self):
        r"""Load the configuration files corresponding to the 3 possible states
        of UTs: LoS, NLoS, and O2I"""

        source = files(models).joinpath(self.o2i_parameter_filepath)
        # pylint: disable=unspecified-encoding
        with open(source) as f:
            self._params_o2i = json.load(f)

        for param_name in self._params_o2i :
            v = self._params_o2i[param_name]
            if isinstance(v, float):
                self._params_o2i[param_name] = tf.constant(v,
                                                    self._dtype.real_dtype)
            elif isinstance(v, int):
                self._params_o2i[param_name] = tf.constant(v, tf.int32)

        source = files(models).joinpath(self.los_parameter_filepath)
        # pylint: disable=unspecified-encoding
        with open(source) as f:
            self._params_los = json.load(f)

        for param_name in self._params_los :
            v = self._params_los[param_name]
            if isinstance(v, float):
                self._params_los[param_name] = tf.constant(v,
                                                    self._dtype.real_dtype)
            elif isinstance(v, int):
                self._params_los[param_name] = tf.constant(v, tf.int32)

        source = files(models).joinpath(self.nlos_parameter_filepath)
        # pylint: disable=unspecified-encoding
        with open(source) as f:
            self._params_nlos = json.load(f)

        for param_name in self._params_nlos :
            v = self._params_nlos[param_name]
            if isinstance(v, float):
                self._params_nlos[param_name] = tf.constant(v,
                                                        self._dtype.real_dtype)
            elif isinstance(v, int):
                self._params_nlos[param_name] = tf.constant(v, tf.int32)

    @abstractmethod
    def _compute_lsp_log_mean_std(self):
        r"""Computes the mean and standard deviations of LSPs in log-domain"""
        pass

    @abstractmethod
    def _compute_pathloss_basic(self):
        r"""Computes the basic component of the pathloss [dB]"""
        pass
