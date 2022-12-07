#
# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Tapped delay line (TDL) channel model from 3GPP TR38.901 specification"""

import json
from importlib_resources import files
import numpy as np

import tensorflow as tf

from sionna import PI, SPEED_OF_LIGHT
from sionna.utils import insert_dims, expand_to_rank, matrix_sqrt, split_dim, flatten_last_dims
from sionna.channel import ChannelModel

from . import models # pylint: disable=relative-beyond-top-level

class TDL(ChannelModel):
    # pylint: disable=line-too-long
    r"""TDL(model, delay_spread, carrier_frequency, num_sinusoids=20, los_angle_of_arrival=PI/4., min_speed=0., max_speed=None, num_rx_ant=1, num_tx_ant=1, spatial_corr_mat=None, rx_corr_mat=None, tx_corr_mat=None, dtype=tf.complex64)

    Tapped delay line (TDL) channel model from the 3GPP [TR38901]_ specification.

    The power delay profiles (PDPs) are normalized to have a total energy of one.

    Channel coefficients are generated using a sum-of-sinusoids model [SoS]_.
    Channel aging is simulated in the event of mobility.

    If a minimum speed and a maximum speed are specified such that the
    maximum speed is greater than the minimum speed, then speeds are randomly
    and uniformly sampled from the specified interval for each link and each
    batch example.

    The TDL model only works for systems with a single transmitter and a single
    receiver. The transmitter and receiver can be equipped with multiple
    antennas. Spatial correlation is simulated through filtering by specified
    correlation matrices.

    The ``spatial_corr_mat`` parameter can be used to specify an arbitrary
    spatial correlation matrix. In particular, it can be used to model
    correlated cross-polarized transmit and receive antennas as follows
    (see, e.g., Annex G.2.3.2.1 [TS38141-1]_):

    .. math::

        \mathbf{R} = \mathbf{R}_{\text{rx}} \otimes \mathbf{\Gamma} \otimes \mathbf{R}_{\text{tx}}

    where :math:`\mathbf{R}` is the spatial correlation matrix ``spatial_corr_mat``,
    :math:`\mathbf{R}_{\text{rx}}` the spatial correlation matrix at the receiver
    with same polarization, :math:`\mathbf{R}_{\text{tx}}` the spatial correlation
    matrix at the transmitter with same polarization, and :math:`\mathbf{\Gamma}`
    the polarization correlation matrix. :math:`\mathbf{\Gamma}` is 1x1 for single-polarized
    antennas, 2x2 when only the transmit or receive antennas are cross-polarized, and 4x4 when
    transmit and receive antennas are cross-polarized.

    It is also possible not to specify ``spatial_corr_mat``, but instead the correlation matrices
    at the receiver and transmitter, using the ``rx_corr_mat`` and ``tx_corr_mat``
    parameters, respectively.
    This can be useful when single polarized antennas are simulated, and it is also
    more computationally efficient.
    This is equivalent to setting ``spatial_corr_mat`` to :

    .. math::
        \mathbf{R} = \mathbf{R}_{\text{rx}} \otimes \mathbf{R}_{\text{tx}}

    where :math:`\mathbf{R}_{\text{rx}}` is the correlation matrix at the receiver
    ``rx_corr_mat`` and  :math:`\mathbf{R}_{\text{tx}}` the correlation matrix at
    the transmitter ``tx_corr_mat``.

    Example
    --------

    The following code snippet shows how to setup a TDL channel model assuming
    an OFDM waveform:

    >>> tdl = TDL(model = "A",
    ...           delay_spread = 300e-9,
    ...           carrier_frequency = 3.5e9,
    ...           min_speed = 0.0,
    ...           max_speed = 3.0)
    >>>
    >>> channel = OFDMChannel(channel_model = tdl,
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
        TDL model to use. Must be one of "A", "B", "C", "D", "E", "A30", "B100", or "C300".

    delay_spread : float
        RMS delay spread [s].
        For the "A30", "B100", and "C300" models, the delay spread must be set
        to 10ns, 100ns, and 300ns, respectively.

    carrier_frequency : float
        Carrier frequency [Hz]

    num_sinusoids : int
        Number of sinusoids for the sum-of-sinusoids model. Defaults to 20.

    los_angle_of_arrival : float
        Angle-of-arrival for LoS path [radian]. Only used with LoS models.
        Defaults to :math:`\pi/4`.

    min_speed : float
        Minimum speed [m/s]. Defaults to 0.

    max_speed : None or float
        Maximum speed [m/s]. If set to `None`,
        then ``max_speed`` takes the same value as ``min_speed``.
        Defaults to `None`.

    num_rx_ant : int
        Number of receive antennas.
        Defaults to 1.

    num_tx_ant : int
        Number of transmit antennas.
        Defaults to 1.

    spatial_corr_mat : [num_rx_ant*num_tx_ant,num_rx_ant*num_tx_ant], tf.complex or `None`
        Spatial correlation matrix.
        If not set to `None`, then ``rx_corr_mat`` and ``tx_corr_mat`` are ignored and
        this matrix is used for spatial correlation.
        If set to `None` and ``rx_corr_mat`` and ``tx_corr_mat`` are also set to `None`,
        then no correlation is applied.
        Defaults to `None`.

    rx_corr_mat : [num_rx_ant,num_rx_ant], tf.complex or `None`
        Spatial correlation matrix for the receiver.
        If set to `None` and ``spatial_corr_mat`` is also set to `None`, then no receive
        correlation is applied.
        Defaults to `None`.

    tx_corr_mat : [num_tx_ant,num_tx_ant], tf.complex or `None`
        Spatial correlation matrix for the transmitter.
        If set to `None` and ``spatial_corr_mat`` is also set to `None`, then no transmit
        correlation is applied.
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
    a : [batch size, num_rx = 1, num_rx_ant = 1, num_tx = 1, num_tx_ant = 1, num_paths, num_time_steps], tf.complex
        Path coefficients

    tau : [batch size, num_rx = 1, num_tx = 1, num_paths], tf.float
        Path delays [s]

    """

    def __init__(   self,
                    model,
                    delay_spread,
                    carrier_frequency,
                    num_sinusoids=20,
                    los_angle_of_arrival=PI/4.,
                    min_speed=0.,
                    max_speed=None,
                    num_rx_ant=1,
                    num_tx_ant=1,
                    spatial_corr_mat=None,
                    rx_corr_mat=None,
                    tx_corr_mat=None,
                    dtype=tf.complex64):

        assert dtype.is_complex, "dtype must be a complex datatype"
        self._dtype = dtype
        real_dtype = dtype.real_dtype
        self._real_dtype = real_dtype

        # Set the file from which to load the model
        assert model in ('A', 'B', 'C', 'D', 'E', 'A30', 'B100', 'C300'),\
            "Invalid TDL model"
        if model == 'A':
            parameters_fname = "TDL-A.json"
        elif model == 'B':
            parameters_fname = "TDL-B.json"
        elif model == 'C':
            parameters_fname = "TDL-C.json"
        elif model == 'D':
            parameters_fname = "TDL-D.json"
        elif model == 'E':
            parameters_fname = "TDL-E.json"
        elif model == 'A30':
            parameters_fname = "TDL-A30.json"
            if delay_spread != 30e-9:
                print("Warning: Delay spread is set to 30ns with this model")
                delay_spread = 30e-9
        elif model == 'B100':
            parameters_fname = "TDL-B100.json"
            if delay_spread != 100e-9:
                print("Warning: Delay spread is set to 100ns with this model")
                delay_spread = 100e-9
        elif model == 'C300':
            parameters_fname = "TDL-C300.json"
            if delay_spread != 300e-9:
                print("Warning: Delay spread is set to 300ns with this model")
                delay_spread = 300e-9

        # Load model parameters
        self._load_parameters(parameters_fname)

        self._num_rx_ant = num_rx_ant
        self._num_tx_ant = num_tx_ant
        self._carrier_frequency = tf.constant(carrier_frequency, real_dtype)
        self._num_sinusoids = tf.constant(num_sinusoids, tf.int32)
        self._los_angle_of_arrival = tf.constant(   los_angle_of_arrival,
                                                    real_dtype)
        self._delay_spread = tf.constant(delay_spread, real_dtype)
        self._min_speed = tf.constant(min_speed, real_dtype)
        if max_speed is None:
            self._max_speed = self._min_speed
        else:
            assert max_speed >= min_speed, \
                "min_speed cannot be larger than max_speed"
            self._max_speed = tf.constant(max_speed, real_dtype)

        # Pre-compute maximum and minimum Doppler shifts
        self._min_doppler = self._compute_doppler(self._min_speed)
        self._max_doppler = self._compute_doppler(self._max_speed)

        # Precompute average angles of arrivals for each sinusoid
        alpha_const = 2.*PI/num_sinusoids * \
                      tf.range(1., self._num_sinusoids+1, 1., dtype=real_dtype)
        self._alpha_const = tf.reshape( alpha_const,
                                        [   1, # batch size
                                            1, # num rx
                                            1, # num rx ant
                                            1, # num tx
                                            1, # num tx ant
                                            1, # num clusters
                                            1, # num time steps
                                            num_sinusoids])

        # Precompute square root of spatial covariance matrices
        if spatial_corr_mat is not None:
            spatial_corr_mat = tf.cast(spatial_corr_mat, self._dtype)
            spatial_corr_mat_sqrt = matrix_sqrt(spatial_corr_mat)
            spatial_corr_mat_sqrt = expand_to_rank(spatial_corr_mat_sqrt, 7, 0)
            self._spatial_corr_mat_sqrt = spatial_corr_mat_sqrt
        else:
            self._spatial_corr_mat_sqrt = None
            if rx_corr_mat is not None:
                rx_corr_mat = tf.cast(rx_corr_mat, self._dtype)
                rx_corr_mat_sqrt = matrix_sqrt(rx_corr_mat)
                rx_corr_mat_sqrt = expand_to_rank(rx_corr_mat_sqrt, 7, 0)
                self._rx_corr_mat_sqrt = rx_corr_mat_sqrt
            else:
                self._rx_corr_mat_sqrt = None
            if tx_corr_mat is not None:
                tx_corr_mat = tf.cast(tx_corr_mat, self._dtype)
                tx_corr_mat_sqrt = matrix_sqrt(tx_corr_mat)
                tx_corr_mat_sqrt = expand_to_rank(tx_corr_mat_sqrt, 7, 0)
                self._tx_corr_mat_sqrt = tx_corr_mat_sqrt
            else:
                self._tx_corr_mat_sqrt = None

    @property
    def num_clusters(self):
        r"""Number of paths (:math:`M`)"""
        return self._num_clusters

    @property
    def los(self):
        r"""`True` if this is a LoS model. `False` otherwise."""
        return self._los

    @property
    def k_factor(self):
        r"""K-factor in linear scale. Only available with LoS models."""
        assert self._los, "This property is only available for LoS models"
        return tf.math.real(self._los_power/self._mean_powers[0])

    @property
    def delays(self):
        r"""Path delays [s]"""
        if self._scale_delays:
            return self._delays*self._delay_spread
        else:
            return self._delays*1e-9 # ns to s

    @property
    def mean_powers(self):
        r"""Path powers in linear scale"""
        if self._los:
            mean_powers = tf.concat([self._mean_powers[:1] + self._los_power,
                                      self._mean_powers[1:]], axis=0)
        else:
            mean_powers = self._mean_powers
        return tf.math.real(mean_powers)

    @property
    def mean_power_los(self):
        r"""LoS component power in linear scale.
        Only available with LoS models."""
        assert self._los, "This property is only available for LoS models"
        return tf.math.real(self._los_power)

    @property
    def delay_spread(self):
        r"""RMS delay spread [s]"""
        return self._delay_spread

    @delay_spread.setter
    def delay_spread(self, value):
        if self._scale_delays:
            self._delay_spread = value
        else:
            print("Warning: The delay spread cannot be set with this model")

    def __call__(self, batch_size, num_time_steps, sampling_frequency):

        # Time steps
        sample_times = tf.range(num_time_steps, dtype=self._real_dtype)\
            /sampling_frequency
        sample_times = tf.expand_dims(insert_dims(sample_times, 6, 0), -1)

        # Generate random maximum Doppler shifts for each sample
        # The Doppler shift is different for each TX-RX link, but shared by
        # all RX ant and TX ant couple for a given link.
        doppler = tf.random.uniform([   batch_size,
                                        1, # num rx
                                        1, # num rx ant
                                        1, # num tx
                                        1, # num tx ant
                                        1, # num clusters
                                        1, # num time steps
                                        1], # num sinusoids
                                        self._min_doppler,
                                        self._max_doppler,
                                        self._real_dtype)

        # Eq. (7) in the paper [TDL] (see class docstring)
        # The angle of arrival is different for each TX-RX link.
        theta = tf.random.uniform([ batch_size,
                                    1, # num rx
                                    1, # 1 RX antenna
                                    1, # num tx
                                    1, # 1 TX antenna
                                    self._num_clusters,
                                    1, # num time steps
                                    self._num_sinusoids],
                                    -PI/tf.cast(self._num_sinusoids,
                                                self._real_dtype),
                                    PI/tf.cast( self._num_sinusoids,
                                                self._real_dtype),
                                    self._real_dtype)
        alpha = self._alpha_const + theta

        # Eq. (6a)-(6c) in the paper [TDL] (see class docstring)
        phi = tf.random.uniform([   batch_size,
                                    1, # 1 RX
                                    self._num_rx_ant, # 1 RX antenna
                                    1, # 1 TX
                                    self._num_tx_ant, # 1 TX antenna
                                    self._num_clusters,
                                    1, # Phase shift is shared by all time steps
                                    self._num_sinusoids],
                                    -PI,
                                    PI,
                                    self._real_dtype)

        argument = doppler * sample_times * tf.cos(alpha) + phi

        # Eq. (6a) in the paper [SoS]
        h = tf.complex(tf.cos(argument), tf.sin(argument))
        normalization_factor = 1./tf.sqrt(  tf.cast(self._num_sinusoids,
                                            self._real_dtype))
        h = tf.complex(normalization_factor, tf.constant(0., self._real_dtype))\
            *tf.reduce_sum(h, axis=-1)

        # Scaling by average power
        mean_powers = tf.expand_dims(insert_dims(self._mean_powers, 5, 0), -1)
        h = tf.sqrt(mean_powers)*h

        # Add specular component to first tap Eq. (11) in [SoS] if LoS
        if self._los:
            # The first tap follows a Rician
            # distribution

            # Specular component phase shift
            phi_0 = tf.random.uniform([ batch_size,
                                        1, # num rx
                                        1, # 1 RX antenna
                                        1, # num tx
                                        1, # 1 TX antenna
                                        1, # only the first tap is concerned
                                        1], # Shared by all time steps
                                        -PI,
                                        PI,
                                        self._real_dtype)
            # Remove the sinusoids dim
            doppler = tf.squeeze(doppler, axis=-1)
            sample_times = tf.squeeze(sample_times, axis=-1)
            arg_spec = doppler*sample_times*tf.cos(self._los_angle_of_arrival)\
                    + phi_0
            h_spec = tf.complex(tf.cos(arg_spec), tf.sin(arg_spec))

            # Update the first tap with the specular component
            h = tf.concat([ h_spec*tf.sqrt(self._los_power) + h[:,:,:,:,:,:1,:],
                            h[:,:,:,:,:,1:,:]],
                            axis=5) # Path dims

        # Delays
        if self._scale_delays:
            delays = self._delays*self._delay_spread
        else:
            delays = self._delays*1e-9 # ns to s
        delays = insert_dims(delays, 3, 0)
        delays = tf.tile(delays, [batch_size, 1, 1, 1])

        # Apply spatial correlation if required
        if self._spatial_corr_mat_sqrt is not None:
            h = tf.transpose(h, [0,1,3,5,6,2,4]) # [..., num_rx_ant, num_tx_ant]
            #h = flatten_dims(h, 2, tf.rank(h)-2)  # [..., num_rx_ant*num_tx_ant]
            h = flatten_last_dims(h, 2) # [..., num_rx_ant*num_tx_ant]
            h = tf.expand_dims(h, axis=-1) # [..., num_rx_ant*num_tx_ant, 1]
            h = tf.matmul(self._spatial_corr_mat_sqrt, h)
            h = tf.squeeze(h, axis=-1)
            h = split_dim(h, [self._num_rx_ant, self._num_tx_ant],
                            tf.rank(h)-1)  # [..., num_rx_ant, num_tx_ant]
            h = tf.transpose(h, [0,1,5,2,6,3,4])
        else:
            if ( (self._rx_corr_mat_sqrt is not None)
                    or (self._tx_corr_mat_sqrt is not None) ):
                h = tf.transpose(h, [0,1,3,5,6,2,4])
                if self._rx_corr_mat_sqrt is not None:
                    h = tf.matmul(self._rx_corr_mat_sqrt, h)
                if self._tx_corr_mat_sqrt is not None:
                    h = tf.matmul(h, self._tx_corr_mat_sqrt)
                h = tf.transpose(h, [0,1,5,2,6,3,4])

        # Stop gadients to avoid useless backpropagation
        h = tf.stop_gradient(h)
        delays = tf.stop_gradient(delays)

        return h, delays

    ###########################################
    # Internal utility functions
    ###########################################

    def _compute_doppler(self, speed):
        r"""Compute the maximum radian Doppler frequency [Hz] for a given
        speed [m/s].

        The maximum radian Doppler frequency :math:`\omega_d` is calculated
        as:

        .. math::
            \omega_d = 2\pi  \frac{v}{c} f_c

        where :math:`v` [m/s] is the speed of the receiver relative to the
        transmitter, :math:`c` [m/s] is the speed of light and,
        :math:`f_c` [Hz] the carrier frequency.

        Input
        ------
        speed : float
            Speed [m/s]

        Output
        --------
        doppler_shift : float
            Doppler shift [Hz]
        """
        return 2.*PI*speed/SPEED_OF_LIGHT*self._carrier_frequency

    def _load_parameters(self, fname):
        r"""Load parameters of a TDL model.

        The model parameters are stored as JSON files with the following keys:
        * los : boolean that indicates if the model is a LoS model
        * num_clusters : integer corresponding to the number of clusters (paths)
        * delays : List of path delays in ascending order normalized by the RMS
            delay spread
        * powers : List of path powers in dB scale

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

        source = files(models).joinpath(fname)
        # pylint: disable=unspecified-encoding
        with open(source) as parameter_file:
            params = json.load(parameter_file)

        # LoS scenario ?
        self._los = bool(params['los'])

        # Scale the delays
        self._scale_delays = bool(params['scale_delays'])

        # Loading cluster delays and mean powers
        self._num_clusters = tf.constant(params['num_clusters'], tf.int32)

        # Retrieve power and delays
        delays = tf.constant(params['delays'], self._real_dtype)
        mean_powers = np.power(10.0, np.array(params['powers'])/10.0)
        mean_powers = tf.constant(mean_powers, self._dtype)

        if self._los:
            # The power of the specular component of the first path is stored
            # separately
            self._los_power = mean_powers[0]
            mean_powers = mean_powers[1:]
            # The first two paths have 0 delays as they correspond to the
            # specular and reflected components of the first path.
            # We need to keep only one.
            delays = delays[1:]

        # Normalize the PDP
        if self._los:
            norm_factor = tf.reduce_sum(mean_powers) + self._los_power
            self._los_power = self._los_power / norm_factor
            mean_powers = mean_powers / norm_factor
        else:
            norm_factor = tf.reduce_sum(mean_powers)
            mean_powers = mean_powers / norm_factor

        self._delays = delays
        self._mean_powers = mean_powers
