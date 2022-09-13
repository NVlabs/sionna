#
# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Clustered delay line (CDL) channel model from 3GPP TR38.901 specification"""


import json
from importlib_resources import files
import tensorflow as tf
from tensorflow import cos, sin
import numpy as np

from sionna.channel.utils import deg_2_rad
from sionna.channel import ChannelModel
from sionna import PI
from sionna.utils.tensors import insert_dims
from . import Topology, ChannelCoefficientsGenerator
from . import Rays

from . import models # pylint: disable=relative-beyond-top-level

class CDL(ChannelModel):
    # pylint: disable=line-too-long
    r"""CDL(model, delay_spread, carrier_frequency, ut_array, bs_array, direction, min_speed=0., max_speed=None, dtype=tf.complex64)

    Clustered delay line (CDL) channel model from the 3GPP [TR38901]_ specification.

    The power delay profiles (PDPs) are normalized to have a total energy of one.

    If a minimum speed and a maximum speed are specified such that the
    maximum speed is greater than the minimum speed, then UTs speeds are
    randomly and uniformly sampled from the specified interval for each link
    and each batch example.

    The CDL model only works for systems with a single transmitter and a single
    receiver. The transmitter and receiver can be equipped with multiple
    antennas.

    Example
    --------

    The following code snippet shows how to setup a CDL channel model assuming
    an OFDM waveform:

    >>> # Panel array configuration for the transmitter and receiver
    >>> bs_array = PanelArray(num_rows_per_panel = 4,
    ...                       num_cols_per_panel = 4,
    ...                       polarization = 'dual',
    ...                       polarization_type = 'cross',
    ...                       antenna_pattern = '38.901',
    ...                       carrier_frequency = 3.5e9)
    >>> ut_array = PanelArray(num_rows_per_panel = 1,
    ...                       num_cols_per_panel = 1,
    ...                       polarization = 'single',
    ...                       polarization_type = 'V',
    ...                       antenna_pattern = 'omni',
    ...                       carrier_frequency = 3.5e9)
    >>> # CDL channel model
    >>> cdl = CDL(model = "A",
    >>>           delay_spread = 300e-9,
    ...           carrier_frequency = 3.5e9,
    ...           ut_array = ut_array,
    ...           bs_array = bs_array,
    ...           direction = 'uplink')
    >>> channel = OFDMChannel(channel_model = cdl,
    ...                       resource_grid = rg)

    where ``rg`` is an instance of :class:`~sionna.ofdm.ResourceGrid`.

    Notes
    ------

    The following tables from [TR38901]_ provide typical values for the delay
    spread.

    +--------------------------+-------------------+
    | Model                    | Delay spread [ns] |
    +==========================+===================+
    | Very short delay spread  | :math:`10`        |
    +--------------------------+-------------------+
    | Short short delay spread | :math:`10`        |
    +--------------------------+-------------------+
    | Nominal delay spread     | :math:`100`       |
    +--------------------------+-------------------+
    | Long delay spread        | :math:`300`       |
    +--------------------------+-------------------+
    | Very long delay spread   | :math:`1000`      |
    +--------------------------+-------------------+

    +-----------------------------------------------+------+------+----------+-----+----+-----+
    |              Delay spread [ns]                |             Frequency [GHz]             |
    +                                               +------+------+----+-----+-----+----+-----+
    |                                               |   2  |   6  | 15 |  28 |  39 | 60 |  70 |
    +========================+======================+======+======+====+=====+=====+====+=====+
    | Indoor office          | Short delay profile  | 20   | 16   | 16 | 16  | 16  | 16 | 16  |
    |                        +----------------------+------+------+----+-----+-----+----+-----+
    |                        | Normal delay profile | 39   | 30   | 24 | 20  | 18  | 16 | 16  |
    |                        +----------------------+------+------+----+-----+-----+----+-----+
    |                        | Long delay profile   | 59   | 53   | 47 | 43  | 41  | 38 | 37  |
    +------------------------+----------------------+------+------+----+-----+-----+----+-----+
    | UMi Street-canyon      | Short delay profile  | 65   | 45   | 37 | 32  | 30  | 27 | 26  |
    |                        +----------------------+------+------+----+-----+-----+----+-----+
    |                        | Normal delay profile | 129  | 93   | 76 | 66  | 61  | 55 | 53  |
    |                        +----------------------+------+------+----+-----+-----+----+-----+
    |                        | Long delay profile   | 634  | 316  | 307| 301 | 297 | 293| 291 |
    +------------------------+----------------------+------+------+----+-----+-----+----+-----+
    | UMa                    | Short delay profile  | 93   | 93   | 85 | 80  | 78  | 75 | 74  |
    |                        +----------------------+------+------+----+-----+-----+----+-----+
    |                        | Normal delay profile | 363  | 363  | 302| 266 | 249 |228 | 221 |
    |                        +----------------------+------+------+----+-----+-----+----+-----+
    |                        | Long delay profile   | 1148 | 1148 | 955| 841 | 786 | 720| 698 |
    +------------------------+----------------------+------+------+----+-----+-----+----+-----+
    | RMa / RMa O2I          | Short delay profile  | 32   | 32   | N/A| N/A | N/A | N/A| N/A |
    |                        +----------------------+------+------+----+-----+-----+----+-----+
    |                        | Normal delay profile | 37   | 37   | N/A| N/A | N/A | N/A| N/A |
    |                        +----------------------+------+------+----+-----+-----+----+-----+
    |                        | Long delay profile   | 153  | 153  | N/A| N/A | N/A | N/A| N/A |
    +------------------------+----------------------+------+------+----+-----+-----+----+-----+
    | UMi / UMa O2I          | Normal delay profile | 242                                     |
    |                        +----------------------+-----------------------------------------+
    |                        | Long delay profile   | 616                                     |
    +------------------------+----------------------+-----------------------------------------+

    Parameters
    -----------

    model : str
        CDL model to use. Must be one of "A", "B", "C", "D" or "E".

    delay_spread : float
        RMS delay spread [s].

    carrier_frequency : float
        Carrier frequency [Hz].

    ut_array : PanelArray
        Panel array used by the UTs. All UTs share the same antenna array
        configuration.

    bs_array : PanelArray
        Panel array used by the Bs. All BSs share the same antenna array
        configuration.

    direction : str
        Link direction. Must be either "uplink" or "downlink".

    ut_orientation : `None` or Tensor of shape [3], tf.float
        Orientation of the UT. If set to `None`, [:math:`\pi`, 0, 0] is used.
        Defaults to `None`.

    bs_orientation : `None` or Tensor of shape [3], tf.float
        Orientation of the BS. If set to `None`, [0, 0, 0] is used.
        Defaults to `None`.

    min_speed : float
        Minimum speed [m/s]. Defaults to 0.

    max_speed : None or float
        Maximum speed [m/s]. If set to `None`,
        then ``max_speed`` takes the same value as ``min_speed``.
        Defaults to `None`.

    dtype : Complex tf.DType
        Defines the datatype for internal calculations and the output
        dtype. Defaults to `tf.complex64`.

    Input
    -----

    batch_size : int
        Batch size

    num_time_steps : int
        Number of time steps

    sampling_frequency : float
        Sampling frequency [Hz]

    Output
    -------
    a : [batch size, num_rx = 1, num_rx_ant, num_tx = 1, num_tx_ant, num_paths, num_time_steps], tf.complex
        Path coefficients

    tau : [batch size, num_rx = 1, num_tx = 1, num_paths], tf.float
        Path delays [s]

    """

    # Number of rays per cluster is set to 20 for CDL
    NUM_RAYS = 20

    def __init__(   self,
                    model,
                    delay_spread,
                    carrier_frequency,
                    ut_array,
                    bs_array,
                    direction,
                    ut_orientation=None,
                    bs_orientation=None,
                    min_speed=0.,
                    max_speed=None,
                    dtype=tf.complex64):

        assert dtype.is_complex, "dtype must be a complex datatype"
        self._dtype = dtype
        real_dtype = dtype.real_dtype
        self._real_dtype = real_dtype

        assert direction in('uplink', 'downlink'), "Invalid link direction"
        self._direction = direction

        # If no orientation is defined by the user, set to default values
        # that make sense
        if ut_orientation is None:
            ut_orientation = tf.constant([PI, 0.0, 0.0], real_dtype)
        if bs_orientation is None:
            bs_orientation = tf.zeros([3], real_dtype)

        # Setting which from UT or BS is the transmitter and which is the
        # receiver according to the link direction
        if self._direction == 'downlink':
            self._moving_end = 'rx'
            self._tx_array = bs_array
            self._rx_array = ut_array
            self._tx_orientation = bs_orientation
            self._rx_orientation = ut_orientation
        elif self._direction == 'uplink':
            self._moving_end = 'tx'
            self._tx_array = ut_array
            self._rx_array = bs_array
            self._tx_orientation = ut_orientation
            self._rx_orientation = bs_orientation

        self._carrier_frequency = tf.constant(carrier_frequency, real_dtype)
        self._delay_spread = tf.constant(delay_spread, real_dtype)
        self._min_speed = tf.constant(min_speed, real_dtype)
        if max_speed is None:
            self._max_speed = self._min_speed
        else:
            assert max_speed >= min_speed, \
                "min_speed cannot be larger than max_speed"
            self._max_speed = tf.constant(max_speed, real_dtype)

        # Loading the model parameters
        assert model in ("A", "B", "C", "D", "E"), "Invalid CDL model"
        if model == 'A':
            parameters_fname = "CDL-A.json"
        elif model == 'B':
            parameters_fname = "CDL-B.json"
        elif model == 'C':
            parameters_fname = "CDL-C.json"
        elif model == 'D':
            parameters_fname = "CDL-D.json"
        elif model == 'E':
            parameters_fname = "CDL-E.json"
        self._load_parameters(parameters_fname)

        # Channel coefficient generator for sampling channel impulse responses
        self._cir_sampler = ChannelCoefficientsGenerator(carrier_frequency,
                                                         self._tx_array,
                                                         self._rx_array,
                                                         subclustering=False,
                                                         dtype=dtype)

    def __call__(self, batch_size, num_time_steps, sampling_frequency):

        ## Topology for generating channel coefficients
        # Sample random velocities
        v_r = tf.random.uniform(shape=[batch_size, 1],
                                minval=self._min_speed,
                                maxval=self._max_speed,
                                dtype=self._real_dtype)
        v_phi = tf.random.uniform(  shape=[batch_size, 1],
                                    minval=0.0,
                                    maxval=2.*PI,
                                    dtype=self._real_dtype)
        v_theta = tf.random.uniform(    shape=[batch_size, 1],
                                        minval=0.0,
                                        maxval=PI,
                                        dtype=self._real_dtype)
        velocities = tf.stack([ v_r*cos(v_phi)*sin(v_theta),
                                v_r*sin(v_phi)*sin(v_theta),
                                v_r*cos(v_theta)], axis=-1)
        los = tf.fill([batch_size, 1, 1], self._los)
        los_aoa = tf.tile(self._los_aoa, [batch_size, 1, 1])
        los_zoa = tf.tile(self._los_zoa, [batch_size, 1, 1])
        los_aod = tf.tile(self._los_aod, [batch_size, 1, 1])
        los_zod = tf.tile(self._los_zod, [batch_size, 1, 1])
        distance_3d = tf.zeros([batch_size, 1, 1], self._real_dtype)
        tx_orientation = tf.tile(insert_dims(self._tx_orientation, 2, 0),
                                 [batch_size, 1, 1])
        rx_orientation = tf.tile(insert_dims(self._rx_orientation, 2, 0),
                                 [batch_size, 1, 1])
        k_factor = tf.tile(self._k_factor, [batch_size, 1, 1])
        topology = Topology(velocities=velocities,
                            moving_end=self._moving_end,
                            los_aoa=los_aoa,
                            los_zoa=los_zoa,
                            los_aod=los_aod,
                            los_zod=los_zod,
                            los=los,
                            distance_3d=distance_3d,
                            tx_orientations=tx_orientation,
                            rx_orientations=rx_orientation)

        # Rays used to generate the channel model
        delays = tf.tile(self._delays*self._delay_spread, [batch_size, 1, 1, 1])
        powers = tf.tile(self._powers, [batch_size, 1, 1, 1])
        aoa = tf.tile(self._aoa, [batch_size, 1, 1, 1, 1])
        aod = tf.tile(self._aod, [batch_size, 1, 1, 1, 1])
        zoa = tf.tile(self._zoa, [batch_size, 1, 1, 1, 1])
        zod = tf.tile(self._zod, [batch_size, 1, 1, 1, 1])
        xpr = tf.tile(self._xpr, [batch_size, 1, 1, 1, 1])

       # Random coupling
        aoa, aod, zoa, zod = self._random_coupling(aoa, aod, zoa, zod)

        rays = Rays(delays=delays,
                    powers=powers,
                    aoa=aoa,
                    aod=aod,
                    zoa=zoa,
                    zod=zod,
                    xpr=xpr)

        # Sampling channel impulse responses
        # pylint: disable=unbalanced-tuple-unpacking
        h, delays = self._cir_sampler(num_time_steps, sampling_frequency,
                                      k_factor, rays, topology)

        # Reshaping to match the expected output
        h = tf.transpose(h, [0, 2, 4, 1, 5, 3, 6])
        delays = tf.transpose(delays, [0, 2, 1, 3])

        # Stop gadients to avoid useless backpropagation
        h = tf.stop_gradient(h)
        delays = tf.stop_gradient(delays)

        return h, delays

    @property
    def num_clusters(self):
        r"""Number of paths (:math:`M`)"""
        return self._num_clusters

    @property
    def los(self):
        r"""`True` is this is a LoS model. `False` otherwise."""
        return self._los

    @property
    def k_factor(self):
        r"""K-factor in linear scale. Only available with LoS models."""
        assert self._los, "This property is only available for LoS models"
        # We return the K-factor for the path with zero-delay, and not for the
        # entire PDP.
        return self._k_factor[0,0,0]/self._powers[0,0,0,0]

    @property
    def delays(self):
        r"""Path delays [s]"""
        return self._delays[0,0,0]*self._delay_spread

    @property
    def powers(self):
        r"""Path powers in linear scale"""
        if self.los:
            k_factor = self._k_factor[0,0,0]
            nlos_powers = self._powers[0,0,0]
            # Power of the LoS path
            p0 = k_factor + nlos_powers[0]
            returned_powers = tf.tensor_scatter_nd_update(nlos_powers,
                                                            [[0]], [p0])
            returned_powers = returned_powers / (k_factor+1.)
        else:
            returned_powers = self._powers[0,0,0]
        return returned_powers

    @property
    def delay_spread(self):
        r"""RMS delay spread [s]"""
        return self._delay_spread

    @delay_spread.setter
    def delay_spread(self, value):
        self._delay_spread = value

    ###########################################
    # Utility functions
    ###########################################

    def _load_parameters(self, fname):
        r"""Load parameters of a CDL model.

        The model parameters are stored as JSON files with the following keys:
        * los : boolean that indicates if the model is a LoS model
        * num_clusters : integer corresponding to the number of clusters (paths)
        * delays : List of path delays in ascending order normalized by the RMS
            delay spread
        * powers : List of path powers in dB scale
        * aod : Paths AoDs [degree]
        * aoa : Paths AoAs [degree]
        * zod : Paths ZoDs [degree]
        * zoa : Paths ZoAs [degree]
        * cASD : Cluster ASD
        * cASA : Cluster ASA
        * cZSD : Cluster ZSD
        * cZSA : Cluster ZSA
        * xpr : XPR in dB

        For LoS models, the two first paths have zero delay, and are assumed
        to correspond to the specular and NLoS component, in this order.

        Input
        ------
        fname : str
            File from which to load the parameters.

        Output
        ------
        None
        """

        # Load the JSON configuration file
        source = files(models).joinpath(fname)
        # pylint: disable=unspecified-encoding
        with open(source) as parameter_file:
            params = json.load(parameter_file)

        # LoS scenario ?
        self._los = tf.cast(params['los'], tf.bool)

        # Loading cluster delays and powers
        self._num_clusters = tf.constant(params['num_clusters'], tf.int32)

        # Loading the rays components, all of shape [num clusters]
        delays = tf.constant(params['delays'], self._real_dtype)
        powers = tf.constant(np.power(10.0, np.array(params['powers'])/10.0),
                                                            self._real_dtype)

        # Normalize powers
        norm_fact = tf.reduce_sum(powers)
        powers = powers / norm_fact

        # Loading the angles and angle spreads of arrivals and departure
        c_aod = tf.constant(params['cASD'], self._real_dtype)
        aod = tf.constant(params['aod'], self._real_dtype)
        c_aoa = tf.constant(params['cASA'], self._real_dtype)
        aoa = tf.constant(params['aoa'], self._real_dtype)
        c_zod = tf.constant(params['cZSD'], self._real_dtype)
        zod = tf.constant(params['zod'], self._real_dtype)
        c_zoa = tf.constant(params['cZSA'], self._real_dtype)
        zoa = tf.constant(params['zoa'], self._real_dtype)

        # If LoS, compute the model K-factor following 7.7.6 of TR38.901 and
        # the LoS path angles of arrival and departure.
        # We remove the specular component from the arrays, as it will be added
        # separately when computing the channel coefficients
        if self._los:
            # Extract the specular component, as it will be added separately by
            # the CIR generator.
            los_power = powers[0]
            powers = powers[1:]
            delays = delays[1:]
            los_aod = aod[0]
            aod = aod[1:]
            los_aoa = aoa[0]
            aoa = aoa[1:]
            los_zod = zod[0]
            zod = zod[1:]
            los_zoa = zoa[0]
            zoa = zoa[1:]

            # The CIR generator scales all NLoS powers by 1/(K+1),
            # where K = k_factor, and adds to the path with zero delay a
            # specular component with power K/(K+1).
            # Note that all the paths are scaled by 1/(K+1), including the ones
            # with non-zero delays.
            # We re-normalized the NLoS power paths to ensure total unit energy
            # after scaling
            norm_fact = tf.reduce_sum(powers)
            powers = powers / norm_fact
            # To ensure that the path with zero delay the ratio between the
            # specular component and the NLoS component has the same ratio as
            # in the CDL PDP, we need to set the K-factor to to the value of
            # the specular component. The ratio between the other paths is
            # preserved as all paths are scaled by 1/(K+1).
            # Note that because of the previous normalization of the NLoS paths'
            # powers, which ensured that their total power is 1,
            # this is equivalent to defining the K factor as done in 3GPP
            # specifications (see step 11):
            # K = (power of specular component)/(total power of the NLoS paths)
            k_factor = los_power/norm_fact

            los_aod = deg_2_rad(los_aod)
            los_aoa = deg_2_rad(los_aoa)
            los_zod = deg_2_rad(los_zod)
            los_zoa = deg_2_rad(los_zoa)
        else:
            # For NLoS models, we need to give value to the K-factor and LoS
            # angles, but they will not be used.
            k_factor = tf.ones((), self._real_dtype)

            los_aod = tf.zeros((), self._real_dtype)
            los_aoa = tf.zeros((), self._real_dtype)
            los_zod = tf.zeros((), self._real_dtype)
            los_zoa = tf.zeros((), self._real_dtype)

        # Generate clusters rays and convert angles to radian
        aod = self._generate_rays(aod, c_aod) # [num clusters, num rays]
        aod = deg_2_rad(aod) # [num clusters, num rays]
        aoa = self._generate_rays(aoa, c_aoa) # [num clusters, num rays]
        aoa = deg_2_rad(aoa) # [num clusters, num rays]
        zod = self._generate_rays(zod, c_zod) # [num clusters, num rays]
        zod = deg_2_rad(zod) # [num clusters, num rays]
        zoa = self._generate_rays(zoa, c_zoa) # [num clusters, num rays]
        zoa = deg_2_rad(zoa) # [num clusters, num rays]

        # Store LoS power
        if self._los:
            self._los_power = los_power

        # Reshape the as expected by the channel impulse response generator
        self._k_factor = self._reshape_for_cir_computation(k_factor)
        los_aod  = self._reshape_for_cir_computation(los_aod)
        los_aoa  = self._reshape_for_cir_computation(los_aoa)
        los_zod  = self._reshape_for_cir_computation(los_zod)
        los_zoa  = self._reshape_for_cir_computation(los_zoa)
        self._delays = self._reshape_for_cir_computation(delays)
        self._powers = self._reshape_for_cir_computation(powers)
        aod = self._reshape_for_cir_computation(aod)
        aoa = self._reshape_for_cir_computation(aoa)
        zod = self._reshape_for_cir_computation(zod)
        zoa = self._reshape_for_cir_computation(zoa)

        # Setting angles of arrivals and departures according to the link
        # direction
        if self._direction == 'downlink':
            self._los_aoa = los_aoa
            self._los_zoa = los_zoa
            self._los_aod = los_aod
            self._los_zod = los_zod
            self._aoa = aoa
            self._zoa = zoa
            self._aod = aod
            self._zod = zod
        elif self._direction == 'uplink':
            self._los_aoa = los_aod
            self._los_zoa = los_zod
            self._los_aod = los_aoa
            self._los_zod = los_zoa
            self._aoa = aod
            self._zoa = zod
            self._aod = aoa
            self._zod = zoa

        # XPR
        xpr = params['xpr']
        xpr = np.power(10.0, xpr/10.0)
        xpr = tf.constant(xpr, self._real_dtype)
        xpr = tf.fill([self._num_clusters, CDL.NUM_RAYS], xpr)
        self._xpr = self._reshape_for_cir_computation(xpr)

    def _generate_rays(self, angles, c):
        r"""
        Generate rays from ``angles`` (which could be ZoD, ZoA, AoD, or AoA) and
        the angle spread ``c`` using equation 7.7-0a of TR38.901 specifications

        Input
        -------
        angles : [num cluster], float
            Tensor of angles with shape `[num_clusters]`

        c : float
            Angle spread

        Output
        -------
        ray_angles : float
            A tensor of shape [num clusters, num rays] containing the angle of
            each ray
        """

        # Basis vector of offset angle from table 7.5-3 from specfications
        # TR38.901
        basis_vector = tf.constant([0.0447, -0.0447,
                                    0.1413, -0.1413,
                                    0.2492, -0.2492,
                                    0.3715, -0.3715,
                                    0.5129, -0.5129,
                                    0.6797, -0.6797,
                                    0.8844, -0.8844,
                                    1.1481, -1.1481,
                                    1.5195, -1.5195,
                                    2.1551, -2.1551], self._real_dtype)

        # Reshape for broadcasting
        # [1, num rays = 20]
        basis_vector = tf.expand_dims(basis_vector, axis=0)
        # [num clusters, 1]
        angles = tf.expand_dims(angles, axis=1)

        # Generate rays following 7.7-0a
        # [num clusters, num rays = 20]
        ray_angles = angles + c*basis_vector

        return ray_angles

    def _reshape_for_cir_computation(self, array):
        r"""
        Add three leading dimensions to array, with shape [1, num_tx, num_rx],
        to reshape it as expected by the channel impulse response sampler.

        Input
        -------
        array : Any shape, float
            Array to reshape

        Output
        -------
        reshaped_array : Tensor, float
            The tensor ``array`` expanded with 3 dimensions for the batch,
            number of tx, and number of rx.
        """

        array_rank = tf.rank(array)
        tiling = tf.constant([1, 1, 1], tf.int32)
        if array_rank > 0:
            tiling = tf.concat([tiling, tf.ones([array_rank],tf.int32)], axis=0)

        array = insert_dims(array, 3, 0)
        array = tf.tile(array, tiling)

        return array

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

        # Create randomly shuffled indices by arg-sorting samples from a random
        # normal distribution
        random_numbers = tf.random.normal(tf.shape(angles))
        shuffled_indices = tf.argsort(random_numbers)
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
