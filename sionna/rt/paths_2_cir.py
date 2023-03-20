#
# SPDX-FileCopyrightText: Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""
Keras layer that computes channel impulse responses (CIR) from paths.
"""

import tensorflow as tf
from tensorflow.keras.layers import Layer

from .scene import Scene
from .utils import dot
from .utils import r_hat, theta_phi_from_unit_vec, rotation_matrix, theta_hat,\
                    phi_hat
from sionna.utils import insert_dims, expand_to_rank, flatten_dims
from .antenna_array import AntennaArray
from .transmitter import Transmitter
from .receiver import Receiver
from sionna.constants import SPEED_OF_LIGHT, PI


class Paths2CIR(Layer):
    # pylint: disable=line-too-long
    r"""Paths2CIR(sampling_frequency, tx_velocities=None, rx_velocities=None, num_time_steps=1, scene=None, carrier_frequency=None, tx_array=None, rx_array=None, transmitters=None, receivers=None, reverse_direction=False, normalize_delays=True, dtype=tf.complex64)

    Layer transforming propagation paths into channel impulse responses (CIRs)

    This layer transforms propagation paths that are described by their
    transfer matrices :math:`\mathbf{T}_i`, delays :math:`\tau_i`,
    angles of departure (AoDs) :math:`(\theta_{\text{T},i}, \varphi_{\text{T},i})`,
    and angles of arrival (AoAs) :math:`(\theta_{\text{R},i}, \varphi_{\text{R},i})`
    into channel impulse responses according to :eq:`H_final` and :eq:`h_final2`.
    To this end, it requires the configuration of a ``carrier_frequency``,
    a ``tx_array`` and ``rx_array`` as instances of :class:`~sionna.rt.AntennaArray`,
    as well as a list of ``transmitters`` and ``receivers`` whose elements are
    instances of :class:`~sionna.rt.Transmitter` and :class:`~sionna.rt.Receiver`,
    respectively. Alternatively, a :class:`~sionna.rt.Scene` can be provided that
    includes all of this information.

    In order to understand what this layer does in detail, consider an
    arbitrary path :math:`i` between two antennas of an arbitrary transmitter
    and receiver. We ignore here the indices of the transmitter and receiver
    as well as the respective antenna indices.

    Given the AoDs :math:`(\theta_{\text{T},i}, \varphi_{\text{T},i})`,
    the orientation of the transmitter :math:`(\alpha_{\text{T}}, \beta_{\text{T}}, \gamma_{\text{T}})`,
    as well as the antenna pattern of the transmitting antenna :math:`\mathbf{C}'_\text{T}(\theta', \varphi')`
    in its local coordinate system (LCS), one can first compute the AoDs :math:`(\theta'_{\text{T},i}, \varphi'_{\text{T},i})` in the
    local coordinate system (LCS) of the transmitter according to :eq:`theta_phi_prime` and then compute
    the antenna response in the global coordinate system (GCS) :math:`\mathbf{C}_\text{T}(\theta_{\text{T},i}, \varphi_{\text{T},i})` as in :eq:`F_prime_2_F`,
    using the rotation matrix :math:`\mathbf{R}=\mathbf{R}(\alpha_{\text{T}}, \beta_{\text{T}}, \gamma_{\text{T}})`.

    The antenna response of the receiver in the GCS :math:`\mathbf{C}_\text{R}(\theta_{\text{R},i}, \varphi_{\text{R},i})` can be computed in an analog fashion
    using the AoAs :math:`(\theta_{\text{R},i}, \varphi_{\text{R},i})`, the orientation
    of the receiver :math:`(\alpha_{\text{R}}, \beta_{\text{R}}, \gamma_{\text{R}})`, and
    the antenna pattern of the receiving antenna :math:`\mathbf{C}'_\text{R}(\theta', \varphi')`
    in its LCS.

    One then obtains the complex-valued path coefficient :math:`a_i` as :eq:`H_final`:

    .. math::

        a_i = \frac{\lambda}{4\pi} \mathbf{C}_\text{R}(\theta_{\text{R},i}, \varphi_{\text{R},i})^{\mathsf{H}}\mathbf{T}_i \mathbf{C}_\text{T}(\theta_{\text{T},i}, \varphi_{\text{T},i}).

    For dual-polarized antennas, a separate channel coefficient for each
    polarization direction is computed by applying the corresponding antenna pattern.
    In this case, the output dimensions `num_rx_ant`/`num_tx_ant` are twice as large as
    the input dimensions `rx_array_size`/`tx_array_size` (see :attr:`~sionna.rt.AntennaArray.num_ant` and
    :attr:`~sionna.rt.AntennaArray.array_size`).

    Next, time evolution of the channel coefficient is added by computing the
    Doppler shift due to movements of the transmitter and receiver. If we denote by
    :math:`\mathbf{v}_{\text{T}}\in\mathbb{R}^3` and :math:`\mathbf{v}_{\text{R}}\in\mathbb{R}^3`
    the velocity vectors of the transmitter and receiver, respectively, the Doppler shifts are computed as

    .. math::

        f_{\text{T}, i} &= \frac{\hat{\mathbf{r}}(\theta_{\text{T},i}, \varphi_{\text{T},i})^\mathsf{T}\mathbf{v}_{\text{T}}}{\lambda}\qquad \text{[Hz]}\\
        f_{\text{R}, i} &= \frac{\hat{\mathbf{r}}(\theta_{\text{R},i}, \varphi_{\text{R},i})^\mathsf{T}\mathbf{v}_{\text{R}}}{\lambda}\qquad \text{[Hz]}

    which will lead to the time-dependent path coefficient

    .. math ::

        a_i(t) = a_i e^{j2\pi(f_{\text{T}, i}+f_{\text{R}, i})t}.

    Note that this model is only valid as long as the AoDs, AoAs, and path delay do not change.

    In case that the dimensions of the inputs due not specify different AoDs, AoAs, or delays for each antenna pair but only
    for each pair of transmitters and receivers, a "synthetic" array is simulated by adding additional phase shifts that depend on the
    antenna position relative to the position of the transmitter (receiver) as well as the AoDs (AoAs).

    Let us denote by :math:`\mathbf{d}_\text{T}` and :math:`\mathbf{d}_\text{R}` the relative positions (with respect to
    the positions of the transmitter/receiver) of the pair of antennas
    for which the channel impulse response shall be computed. These can be accessed through the antenna array's property
    :attr:`~sionna.rt.AntennaArray.positions`. Using a plane-wave assumption, the resulting phase shifts
    from these displacements can be computed as

    .. math::

        p_{\text{T}, i} &= \frac{2\pi}{\lambda}\hat{\mathbf{r}}(\theta_{\text{T},i}, \varphi_{\text{T},i})^\mathsf{T} \mathbf{d}_\text{T}\\
        p_{\text{R}, i} &= \frac{2\pi}{\lambda}\hat{\mathbf{r}}(\theta_{\text{R},i}, \varphi_{\text{R},i})^\mathsf{T} \mathbf{d}_\text{R}.

    The final expression for the path coefficient is

    .. math::

        a_i(t) =  a_i e^{j(p_{\text{T}, i} + p_{\text{R}, i})} e^{j2\pi(f_{\text{T}, i}+f_{\text{R}, i})t}.

    These computations are carried out for the channels between all pair of antennas of all transmitters and receivers.
    The time-dependent path coefficients are then sampled at the specified ``sampling_frequency`` for ``num_time_steps``
    samples starting at :math:`t=0`.

    Lastly, if ``normalize_delays`` is `True`, the delays :math:`\tau_i` are normalized
    such that the delay of the shortest path between any pair of antennas of a transmitter and receiver
    equals zero (ignoring antenna indices here):

    .. math::

        \tau_i = \tau_i -\min_j \tau_j \quad \forall i.

    This step is essentially equivalent to assuming perfect synchronization of all radio devices.

    Example
    -------
    .. code-block:: Python

        import sionna
        from sionna.rt import load_scene, Camera, Transmitter, Receiver, PlanarArray, Paths2CIR

        # Load example scene
        scene = load_scene(sionna.rt.scene.munich)

        # Configure antenna array for all transmitters
        scene.tx_array = PlanarArray(num_rows=1,
                                  num_cols=1,
                                  vertical_spacing=0.5,
                                  horizontal_spacing=0.5,
                                  pattern="tr38901",
                                  polarization="V")

        # Configure antenna array for all receivers
        scene.rx_array = PlanarArray(num_rows=1,
                                  num_cols=1,
                                  vertical_spacing=0.5,
                                  horizontal_spacing=0.5,
                                  pattern="dipole",
                                  polarization="V")

        # Create transmitter
        tx = Transmitter(name="tx",
                      position=[8.5,21,27],
                      orientation=[0,0,0])
        scene.add(tx)

        # Create a receiver
        rx = Receiver(name="rx",
                   position=[45,90,1.5],
                   orientation=[0,0,0])
        scene.add(rx)

        # TX points towards RX
        tx.look_at(rx)

        # Compute paths
        paths = scene.compute_paths()

        # Transform paths into channel impulse responses
        p2c = Paths2CIR(sampling_frequency=1e6, scene=scene)
        a, tau = p2c(paths.as_tuple())

        # Visualize channel impulse response
        plt.figure()
        plt.stem(np.squeeze(tau)/1e-9, -20*np.log10(np.squeeze(np.abs(a))));
        plt.xlabel(r"$\tau$ [ns]")
        plt.ylabel(r"Path loss [dB]")
        plt.title("Channel impulse response")

    .. figure:: ../figures/cir_visualization.png
        :align: center


    Parameters
    ----------
    sampling_frequency : float
        Frequency [Hz] at which the channel impulse response is sampled

    tx_velocities : [batch_size, num_tx, 3] or broadcastable, tf.float | `None`
        Velocity vectors :math:`(v_\text{x}, v_\text{y}, v_\text{z})` of all
        transmitters [m/s]. Defaults to `None` (i.e., no mobility).

    rx_velocities : [batch_size, num_tx, 3] or broadcastable, tf.float | `None`
        Velocity vectors :math:`(v_\text{x}, v_\text{y}, v_\text{z})` of all
        receivers [m/s]. Defaults to `None` (i.e., no mobility).

    num_time_steps : int
        Number of time steps.
        Defaults to 1.

    scene : :class:`~sionna.rt.Scene` | `None`
        A scene whose configuration matches the provided inputs.
        Defaults to `None`. In this case, the subsequent arguments
        must be set.

    carrier_frequency : float | `None`
        Carrier frequency [Hz].
        Only used if ``scene`` is `None`.
        Defaults to `None`.

    tx_array : :class:`~sionna.rt.AntennaArray` | `None`
        The antenna array used by all transmitters.
        Its property :attr:`~sionna.rt.AntennaArray.array_size`
        will define ``tx_array_size``.
        Only used if ``scene`` is `None`.
        Defaults to `None`.

    rx_array : :class:`~sionna.rt.AntennaArray` | `None`
        The antenna array used by all receivers.
        Its property :attr:`~sionna.rt.AntennaArray.array_size`
        will define ``rx_array_size``.
        Only used if ``scene`` is `None`.
        Defaults to `None`.

    transmitters : list, :class:`~sionna.rt.Transmitter` | `None`
        List of all transmitters.
        Its length defines ``num_tx``.
        Only used if ``scene`` is `None`.
        Defaults to `None`.

    receivers : list, :class:`~sionna.rt.Receiver` | `None`
        List of all receivers.
        Its length defines ``num_rx``.
        Only used if ``scene`` is `None`.
        Defaults to `None`.

    reverse_direction : bool
        If `True`, the outputs ``a`` and ``tau`` will be transposed such that
        ``num_tx`` and ``num_tx_ant`` are swapped with ``num_rx`` and
        ``num_rx_ant``, respectively.
        This is a useful feature to generate uplink CIRs from downlink paths.
        Defaults to `False`.

    normalize_delays: bool
        If `True`, the delays ``tau`` are normalized such that the shortest path
        between any pair of antennas belonging to a transmitter and receiver
        has zero delay.
        Defaults to `True`.

    dtype : tf.complex64 | tf.complex128
        Datatype for all computations, inputs, and outputs.
        Defaults to `tf.complex64`.

    Input
    -----
    (mat_t, tau, theta_t, phi_t, theta_r, phi_r) :
        Tuple:

    mat_t : [batch_size, num_rx, rx_array_size, num_tx, tx_array_size, max_num_paths, 2, 2] or [batch_size, num_rx, num_tx, max_num_paths, 2, 2], tf.complex
        Transfer matrices :math:`\mathbf{T}_i` as defined in :eq:`T_tilde`.
        These are 2x2 complex-valued matrices modeling the transformation
        of the vertically and horizontally polarized field components along
        the paths.
        If the shape of this tensor
        is `[1, num_rx, num_tx, max_num_paths, 2, 2]`, the responses for the
        individual antenna elements are synthetically computed.
        If there are less than `max_num_path` valid paths between a
        transmit and receive antenna, irrelevant elements must be
        filled with zeros.

    tau : [batch_size, num_rx, rx_array_size, num_tx, tx_array_size, max_num_paths] or [batch_size, num_rx, num_tx, max_num_paths], tf.float
        Propagation delay of each path [s].
        If the shape of this tensor
        is `[1, num_rx, num_tx, max_num_paths]`, the delays for the
        individual antenna elements are assumed to be equal.
        If there are less than `max_num_path` valid paths between a
        transmit and receive antenna, the irrelevant elements must be
        filled with -1.

    theta_t : [batch_size, num_rx, rx_array_size, num_tx, tx_array_size, max_num_paths] or [batch_size, num_rx, num_tx, max_num_paths], tf.float
        Zenith  angles of departure :math:`\theta_{\text{T},i}` [rad].
        If the shape of this tensor
        is `[1, num_rx, num_tx, max_num_paths]`, the angles for the
        individual antenna elements are assumed to be equal.
        If there are less than `max_num_path` valid paths between a
        transmit and receive antenna, the irrelevant elements must be
        filled with zeros.

    phi_t : [batch_size, num_rx, rx_array_size, num_tx, tx_array_size, max_num_paths] or [batch_size, num_rx, num_tx, max_num_paths], tf.float
        Azimuth angles of departure :math:`\varphi_{\text{T},i}` [rad].
        See description of ``theta_t``.

    theta_r : [batch_size, num_rx, rx_array_size, num_tx, tx_array_size, max_num_paths] or [batch_size, num_rx, num_tx, max_num_paths], tf.float
        Zenith angles of arrival :math:`\theta_{\text{R},i}` [rad].
        See description of ``theta_t``.

    phi_r : [batch_size, num_rx, rx_array_size, num_tx, tx_array_size, max_num_paths] or [batch_size, num_rx, num_tx, max_num_paths], tf.float
        Azimuth angles of arrival :math:`\varphi_{\text{T},i}` [rad].
        See description of ``theta_t``.

    Output
    ------
    a : [batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, max_num_paths, num_time_steps], tf.complex
        Path coefficients. The dimensions ``num_rx_ant`` and ``num_tx_ant`` are
        respectively defined by the property :attr:`~sionna.rt.AntennaArray.num_ant`
        of the ``rx_array`` and ``tx_array``.
        If ``reverse_direction`` is `True`,  ``num_tx`` and ``num_tx_ant`` are swapped with ``num_rx`` and
        ``num_rx_ant``, respectively.

    tau : [batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, max_num_paths] or [batch size, num_rx, num_tx, max_num_paths], tf.float
        Path delays [s]. The dimensions ``num_rx_ant`` and ``num_tx_ant`` are
        respectively defined by the property :attr:`~sionna.rt.AntennaArray.num_ant`
        of the ``rx_array`` and ``tx_array``. If the rank of the input ``tau`` is equal to four,
        the output dimensions are [batch size, num_rx, num_tx, max_num_paths].
        If ``reverse_direction`` is `True`,  ``num_tx`` and ``num_tx_ant`` are swapped with ``num_rx`` and
        ``num_rx_ant``, respectively.
    """
    def __init__(self, sampling_frequency,
                       tx_velocities=None,
                       rx_velocities=None,
                       num_time_steps=1,
                       scene=None,
                       carrier_frequency=None,
                       tx_array=None,
                       rx_array=None,
                       transmitters=None,
                       receivers=None,
                       reverse_direction=False,
                       normalize_delays=True,
                       dtype=tf.complex64):

        if dtype not in (tf.complex64, tf.complex128):
            raise ValueError("`dtype` must be tf.complex64 or tf.complex64`")

        if scene is not None:
            if not isinstance(scene, Scene):
                msg = "If `scene` is not `None`, it must be an instance of "\
                      "`Scene`"
                raise ValueError(msg)
            # If scene is set, ignore the other parameters of __init__ and use
            # the ones of the scene
            self._tx_array = scene.tx_array
            self._rx_array = scene.rx_array
            self._wavelength = scene.wavelength
            self._transmitters = scene.transmitters.values()
            self._receivers = scene.receivers.values()

        else:
            if not isinstance(tx_array, AntennaArray):
                msg = "`tx_array` must be set to an instance of `AntennaArray`"\
                      " if `scene` is not set"
                raise ValueError(msg)
            self._tx_array = tx_array

            if not isinstance(rx_array, AntennaArray):
                msg = "`rx_array` must be set to an instance of `AntennaArray`"\
                      " if `scene` is not set"
                raise ValueError(msg)
            self._rx_array = rx_array

            if carrier_frequency is None:
                raise ValueError("`carrier_frequency` must be provided")
            if carrier_frequency <= 0.0:
                raise ValueError("Wavelength must be positive")
            self._wavelength = tf.cast(SPEED_OF_LIGHT/carrier_frequency,
                                       dtype.real_dtype)

            tx_valid = True
            if isinstance(transmitters, list):
                for tx in transmitters:
                    if not isinstance(tx, Transmitter):
                        tx_valid = False
                        break
            else:
                tx_valid = False
            if tx_valid:
                self._transmitters = transmitters
            else:
                msg = "`transmitters` must be a list of `Transmitter` "\
                        "if `scene` is not set"
                raise ValueError(msg)

            rx_valid = True
            if isinstance(receivers, list):
                for rx in receivers:
                    if not isinstance(rx, Receiver):
                        rx_valid = False
                        break
            else:
                rx_valid = False
            if rx_valid:
                self._receivers = receivers
            else:
                msg = "`receivers` must be a list of `Transmitter` "\
                        "if `scene` is not set"
                raise ValueError(msg)

        sampling_frequency = tf.cast(sampling_frequency, dtype.real_dtype)
        if sampling_frequency <= 0.0:
            raise ValueError("The sampling frequency must be positive")
        self._sampling_frequency = sampling_frequency

        if tx_velocities is None:
            tx_velocities = [0.,0.,0.]
        tx_velocities = tf.cast(tx_velocities, dtype.real_dtype)
        tx_velocities = expand_to_rank(tx_velocities, 3, 0)
        if tx_velocities.shape[2] != 3:
            raise ValueError("Last dimension of `tx_velocities` must equal 3")
        self._tx_velocities = tx_velocities

        if rx_velocities is None:
            rx_velocities = [0.,0.,0.]
        rx_velocities = tf.cast(rx_velocities, dtype.real_dtype)
        rx_velocities = expand_to_rank(rx_velocities, 3, 0)
        if rx_velocities.shape[2] != 3:
            raise ValueError("Last dimension of `rx_velocities` must equal 3")
        self._rx_velocities = rx_velocities

        num_time_steps = tf.cast(num_time_steps, tf.int32)
        if num_time_steps <= 0:
            msg = "The number of time samples must a positive integer"
            raise ValueError(msg)
        self._num_time_steps = num_time_steps

        if not isinstance(reverse_direction, bool):
            msg = "`reverse_direction` must be bool"
            raise ValueError(msg)
        self._reverse_direction = reverse_direction

        if not isinstance(normalize_delays, bool):
            msg = "`normalize_delays` must be bool"
            raise ValueError(msg)
        self._normalize_delays = normalize_delays

        super().__init__(dtype=dtype)

    def build(self, input_shape):

        # Check the inputs' shape to make sure they match what was set at
        # instantiation.

        mat_t = input_shape[0]

        num_rx = len(self._receivers)
        num_tx = len(self._transmitters)
        rx_array_size = self._rx_array.positions.shape[0]
        tx_array_size = self._tx_array.positions.shape[0]


        # Detects if synthetic arrays are used or not according to the rank
        # of the input
        if mat_t.rank == 6:
            self._synthetic_array = True
            max_num_paths = mat_t[3]
            expected_in_shape = [num_rx, num_tx, max_num_paths]
            expected_rank = 4
            using = "using"
        elif mat_t.rank == 8:
            self._synthetic_array = False
            max_num_paths = mat_t[5]
            expected_in_shape = [num_rx, rx_array_size, num_tx, tx_array_size,
                                    max_num_paths]
            expected_rank = 6
            using = "not using"
        else:
            msg = "Rank of `mat_t` must be either 6 when using synthetic"\
                  " arrays or 8 when not using synthetic arrays"
            raise ValueError(msg)

        # Check the rank of input tensors (except `mat_t`)
        if not all(v.rank == expected_rank for v in input_shape[1:]):
            msg  = f"Rank of all inputs except `mat_t` must be {expected_rank}"\
                    f" when {using} synthetic arrays"
            raise ValueError(msg)

        # Check that the inner dimensions of the inputs are as expected
        if not all(v[1:] == expected_in_shape for v in input_shape[1:]):
            msg  = f"Inner shape of all inputs except `mat_t` must be"\
                   f" {expected_in_shape} when {using} synthetic arrays"
            raise ValueError(msg)

        # Check the shape of `mat_t`
        expected_in_shape += [2,2]
        if not mat_t[1:] == expected_in_shape:
            msg  = f"Inner shape of `mat_t` must be"\
                   f" {expected_in_shape} when {using} synthetic arrays"
            raise ValueError(msg)

    def call(self, inputs):

        # mat_t : [batch_dim, num_rx, rx_array_size, num_tx, tx_array_size,
        # ... max_num_paths, 2, 2]
        # or
        # [batch_dim, num_rx, num_tx, max_num_paths, 2, 2], tf.complex

        # tau : [batch_dim, num_rx, rx_array_size, num_tx, tx_array_size,
        # ... max_num_paths]
        # or
        # [batch_dim, num_rx, num_tx, max_num_paths], tf.float

        # theta_t : [batch_dim, num_rx, rx_array_size, num_tx, tx_array_size,
        # ... max_num_paths]
        # or
        # [batch_dim, num_rx, num_tx, max_num_paths], tf.float

        # phi_t : [batch_dim, num_rx, rx_array_size, num_tx, tx_array_size,
        # ... max_num_paths]
        # or
        # [batch_dim, num_rx, num_tx, max_num_paths], tf.float

        # theta_r : [batch_dim, num_rx, rx_array_size, num_tx, tx_array_size,
        # ... max_num_paths]
        # or
        # [batch_dim, num_rx, num_tx, max_num_paths], tf.float

        # phi_r : [batch_dim, num_rx, rx_array_size, num_tx, tx_array_size,
        # ... max_num_paths]
        # or
        # [batch_dim, num_rx, num_tx, max_num_paths], tf.float
        mat_t, tau, theta_t, phi_t, theta_r, phi_r = inputs

        # Apply multiplication by wavelength/4pi
        # mat_t : [batch_dim, num_rx, rx_array_size, num_tx, tx_array_size,
        # ... max_num_paths, 2, 2]
        # or
        # [batch_dim, num_rx, num_tx, max_num_paths, 2, 2]
        cst = tf.cast(self._wavelength/(4.*PI), self._dtype)
        h = cst*mat_t

        dtype = tf.as_dtype(self._dtype)
        zeror = tf.zeros((), dtype.real_dtype)
        two_pi = tf.cast(2.*PI, dtype.real_dtype)

        ######################################################
        # If using synthetic array, applies the phase shift
        # due to the antenna array
        ######################################################

        # Rotated position of the TX and RX antenna elements
        # [num_tx, tx_array_size, 3]
        tx_rel_ant_pos = [self._tx_array.rotated_positions(tx.orientation)
                            for tx in self._transmitters]
        tx_rel_ant_pos = tf.stack(tx_rel_ant_pos, axis=0)
        # [num_rx, rx_array_size, 3]
        rx_rel_ant_pos = [self._rx_array.rotated_positions(rx.orientation)
                            for rx in self._receivers]
        rx_rel_ant_pos = tf.stack(rx_rel_ant_pos, axis=0)

        # Normalized wave vectors for rx and tx
        # These vectors are pointing away from the array.
        # [batch_dim, num_rx, num_tx, max_num_paths, 3]
        # or [batch_dim, num_rx, rx_array_size, num_tx, tx_array_size,
        #       ... max_num_paths, 3]
        k_r = r_hat(theta_r, phi_r)
        # [batch_dim, num_rx, num_tx, max_num_paths, 3]
        # or [batch_dim, num_rx, rx_array_size, num_tx, tx_array_size,
        #       ... max_num_paths, 3]
        k_t = r_hat(theta_t, phi_t)

        if self._synthetic_array:
            # Expand dims for broadcasting with antennas
            # [batch_dim, num_rx, 1, num_tx, 1, max_num_paths, 3]
            k_r = tf.expand_dims(tf.expand_dims(k_r, axis=2), axis=4)
            k_t = tf.expand_dims(tf.expand_dims(k_t, axis=2), axis=4)
            # Compute the synthetic phase shifts due to the antenna array
            # Transmitter side
            # Expand for broadcasting with receiver, receive antennas,
            # paths, and batch
            # [1, 1, 1, num_tx, tx_array_size, 3]
            tx_rel_ant_pos = insert_dims(tx_rel_ant_pos, 3, axis=0)
            # [1, 1, 1, num_tx, tx_array_size, 1, 3]
            tx_rel_ant_pos = tf.expand_dims(tx_rel_ant_pos, axis=5)
            # [batch_dim, num_rx, 1, num_tx, tx_array_size, max_num_paths]
            tx_phase_shifts = dot(tx_rel_ant_pos, k_t)
            # Receiver side
            # Expand for broadcasting with transmitter, transmit antennas,
            # paths, and batch
            # [1, num_rx, rx_array_size, 3]
            rx_rel_ant_pos = tf.expand_dims(rx_rel_ant_pos, axis=0)
            # [1, num_rx, rx_array_size, 1, 1, 1, 3]
            rx_rel_ant_pos = insert_dims(rx_rel_ant_pos, 3, axis=3)
            # [batch_dim, num_rx, rx_array_size, num_tx, 1, max_num_paths]
            rx_phase_shifts = dot(rx_rel_ant_pos, k_r)
            # Total phase shift
            # [batch_dim, num_rx, rx_array_size, num_tx, tx_array_size,
            #   ... max_num_paths]
            phase_shifts = rx_phase_shifts + tx_phase_shifts
            phase_shifts = two_pi*phase_shifts/self._wavelength
            # Apply the phase shifts
            # Expand field for broadcasting with antennas
            # [batch_dim, num_rx, 1, num_tx, 1, max_num_paths, 2, 2]
            h = tf.expand_dims(tf.expand_dims(h, axis=2), axis=4)
            # Expand phase shifts for broadcasting with transfer matrix
            # [batch_dim, num_rx, rx_array_size, num_tx, tx_array_size,
            #   ... max_num_paths, 1, 1]
            phase_shifts = expand_to_rank(phase_shifts, tf.rank(h), 6)
            # Broadcast is not supported by TF for such high rank tensors.
            # We therefore do it manually
            # [batch_dim, num_rx, rx_array_size, num_tx, tx_array_size,
            #   ... max_num_paths, 2, 2]
            h = tf.tile(h, [1, 1, phase_shifts.shape[2], 1,
                            phase_shifts.shape[4], 1, 1, 1])
            # [batch_dim, num_rx, rx_array_size, num_tx, tx_array_size,
            #   ... max_num_paths, 2, 2]
            h = h*tf.exp(tf.complex(tf.zeros_like(phase_shifts), phase_shifts))

        ######################################################
        # Compute and apply antenna patterns
        ######################################################

        # Expand angles of arrival and departure for broadcasting with antennas
        # if using synthetic arrays
        if self._synthetic_array:
            # [batch_dim, num_rx, 1, num_tx, 1, max_num_paths]
            theta_t = tf.expand_dims(tf.expand_dims(theta_t,axis=2), axis=4)
            # [batch_dim, num_rx, 1, num_tx, 1, max_num_paths]
            phi_t = tf.expand_dims(tf.expand_dims(phi_t, axis=2), axis=4)
            # [batch_dim, num_rx, 1, num_tx, 1, max_num_paths]
            theta_r = tf.expand_dims(tf.expand_dims(theta_r,axis=2), axis=4)
            # [batch_dim, num_rx, 1, num_tx, 1, max_num_paths]
            phi_r = tf.expand_dims(tf.expand_dims(phi_r, axis=2), axis=4)

        # Rotation matrices for transmitters
        # [num_tx, 3, 3]
        tx_rot_mat = [rotation_matrix(tx.orientation)
                      for tx in self._transmitters]
        tx_rot_mat = tf.stack(tx_rot_mat, axis=0)
        # Rotation matrices for receivers
        # [num_rx, 3, 3]
        rx_rot_mat = [rotation_matrix(rx.orientation)
                      for rx in self._receivers]
        rx_rot_mat = tf.stack(rx_rot_mat, axis=0)

        # Normalized wave transmit vector in the local coordinate system of
        # the transmitters
        # [1, 1, 1, num_tx, 1, 1, 3, 3]
        tx_rot_mat = expand_to_rank(insert_dims(tx_rot_mat, 3, 0), 8, 4)
        # [batch_dim, num_rx, 1, num_tx, 1, max_num_paths, 3]
        # or
        # [batch_dim, num_rx, rx_array_size, num_tx, tx_array_size,
        #   ... max_num_paths, 3]
        k_prime_t = tf.linalg.matvec(tx_rot_mat, k_t, transpose_a=True)

        # Normalized wave receiver vector in the local coordinate system of
        # the receivers
        # [1, num_rx, 1, 1, 1, 1, 3, 3]
        rx_rot_mat = expand_to_rank(tf.expand_dims(rx_rot_mat, axis=0), 8, 2)
        # [batch_dim, num_rx, 1, num_tx, 1, max_num_paths, 3]
        # or
        # [batch_dim, num_rx, rx_array_size, num_tx, tx_array_size,
        #   ... max_num_paths, 3]
        k_prime_r = tf.linalg.matvec(rx_rot_mat, k_r, transpose_a=True)

        # Angles of departure in the local coordinate system of the
        # transmitter
        # [batch_dim, num_rx, 1, num_tx, 1, max_num_paths]
        # or
        # [batch_dim, num_rx, rx_array_size, num_tx, tx_array_size,
        #   ... max_num_paths]
        theta_prime_t, phi_prime_t = theta_phi_from_unit_vec(k_prime_t)

        # Angles of arrival in the local coordinate system of the
        # receivers
        # [batch_dim, num_rx, 1, num_tx, 1, max_num_paths]
        # or
        # [batch_dim, num_rx, rx_array_size, num_tx, tx_array_size,
        #   ... max_num_paths]
        theta_prime_r, phi_prime_r = theta_phi_from_unit_vec(k_prime_r)

        # Spherical global frame vectors for tx and rx
        # [batch_dim, num_rx, rx_array_size, num_tx, tx_array_size,
        #   ... max_num_paths, 3]
        #  or
        # [batch_dim, num_rx, 1, num_tx, 1, max_num_paths, 3]
        theta_hat_t = theta_hat(theta_t, phi_t)
        # [batch_dim, num_rx, rx_array_size, num_tx, tx_array_size,
        #   ... max_num_paths, 3]
        # or
        # [batch_dim, num_rx, 1, num_tx, 1, max_num_paths, 3]
        phi_hat_t = phi_hat(phi_t)
        # [batch_dim, num_rx, rx_array_size, num_tx, tx_array_size,
        #   ... max_num_paths, 3]
        # or
        # [batch_dim, num_rx, 1, num_tx, 1, max_num_paths, 3]
        theta_hat_r = theta_hat(theta_r, phi_r)
        # [batch_dim, num_rx, rx_array_size, num_tx, tx_array_size,
        #   ... max_num_paths, 3]
        # or
        # [batch_dim, num_rx, 1, num_tx, 1, max_num_paths, 3]
        phi_hat_r = phi_hat(phi_r)

        # Spherical local frame vectors for tx and rx
        # [batch_dim, num_rx, rx_array_size, num_tx, tx_array_size,
        #   ... max_num_paths, 3]
        # or
        # [batch_dim, num_rx, 1, num_tx, 1, max_num_paths, 3]
        theta_hat_prime_t = theta_hat(theta_prime_t, phi_prime_t)
        # [batch_dim, num_rx, rx_array_size, num_tx, tx_array_size,
        #   ... max_num_paths, 3]
        # or
        # [batch_dim, num_rx, 1, num_tx, 1, max_num_paths, 3]
        phi_hat_prime_t = phi_hat(phi_prime_t)
        # [batch_dim, num_rx, rx_array_size, num_tx, tx_array_size,
        #   ... max_num_paths, 3]
        # or
        # [batch_dim, num_rx, 1, num_tx, 1, max_num_paths, 3]
        theta_hat_prime_r = theta_hat(theta_prime_r, phi_prime_r)
        # [batch_dim, num_rx, rx_array_size, num_tx, tx_array_size,
        #   ... max_num_paths, 3]
        # or
        # [batch_dim, num_rx, 1, num_tx, 1, max_num_paths, 3]
        phi_hat_prime_r = phi_hat(phi_prime_r)

        # Rotation matrix for going from the spherical LCS to the spherical GCS
        # For transmitters
        # [batch_dim, num_rx, rx_array_size, num_tx, tx_array_size,
        #   ... max_num_paths]
        # or
        # [batch_dim, num_rx, 1, num_tx, 1, max_num_paths]
        tx_lcs2gcs_11 = dot(theta_hat_t,
                            tf.linalg.matvec(tx_rot_mat, theta_hat_prime_t))
        # [batch_dim, num_rx, rx_array_size, num_tx, tx_array_size,
        #   ... max_num_paths]
        # or
        # [batch_dim, num_rx, 1, num_tx, 1, max_num_paths]
        tx_lcs2gcs_12 = dot(theta_hat_t,
                            tf.linalg.matvec(tx_rot_mat, phi_hat_prime_t))
        # [batch_dim, num_rx, rx_array_size, num_tx, tx_array_size,
        #   ... max_num_paths]
        #  or
        # [batch_dim, num_rx, 1, num_tx, 1, max_num_paths]
        tx_lcs2gcs_21 = dot(phi_hat_t,
                            tf.linalg.matvec(tx_rot_mat, theta_hat_prime_t))
        # [batch_dim, num_rx, rx_array_size, num_tx, tx_array_size,
        #   ... max_num_paths]
        # or
        # [batch_dim, num_rx, 1, num_tx, 1, max_num_paths]
        tx_lcs2gcs_22 = dot(phi_hat_t,
                            tf.linalg.matvec(tx_rot_mat, phi_hat_prime_t))
        # [batch_dim, num_rx, rx_array_size, num_tx, tx_array_size,
        #   ... max_num_paths]
        # or
        # [batch_dim, num_rx, 1, num_tx, 1, max_num_paths, 2, 2]
        tx_lcs2gcs = tf.stack(
                    [tf.stack([tx_lcs2gcs_11, tx_lcs2gcs_12], axis=-1),
                     tf.stack([tx_lcs2gcs_21, tx_lcs2gcs_22], axis=-1)],
                    axis=-2)
        tx_lcs2gcs = tf.complex(tx_lcs2gcs, tf.zeros_like(tx_lcs2gcs))
        # For receivers
        # [batch_dim, num_rx, rx_array_size, num_tx, tx_array_size,
        #   ... max_num_paths]
        # or
        # [batch_dim, num_rx, 1, num_tx, 1, max_num_paths]
        rx_lcs2gcs_11 = dot(theta_hat_r,
                            tf.linalg.matvec(rx_rot_mat, theta_hat_prime_r))
        # [batch_dim, num_rx, rx_array_size, num_tx, tx_array_size,
        #   ... max_num_paths]
        # or
        # [batch_dim, num_rx, 1, num_tx, 1, max_num_paths]
        rx_lcs2gcs_12 = dot(theta_hat_r,
                            tf.linalg.matvec(rx_rot_mat, phi_hat_prime_r))
        # [batch_dim, num_rx, rx_array_size, num_tx, tx_array_size,
        #   ... max_num_paths]
        #  or
        # [batch_dim, num_rx, 1, num_tx, 1, max_num_paths]
        rx_lcs2gcs_21 = dot(phi_hat_r,
                            tf.linalg.matvec(rx_rot_mat, theta_hat_prime_r))
        # [batch_dim, num_rx, rx_array_size, num_tx, tx_array_size,
        #   ... max_num_paths]
        #  or
        # [batch_dim, num_rx, 1, num_tx, 1, max_num_paths]
        rx_lcs2gcs_22 = dot(phi_hat_r,
                            tf.linalg.matvec(rx_rot_mat, phi_hat_prime_r))
        # [batch_dim, num_rx, rx_array_size, num_tx, tx_array_size,
        #   ... max_num_paths, 2, 2]
        # or
        # [batch_dim, num_rx, 1, num_tx, 1, max_num_paths, 2, 2]
        rx_lcs2gcs = tf.stack(
                    [tf.stack([rx_lcs2gcs_11, rx_lcs2gcs_12], axis=-1),
                     tf.stack([rx_lcs2gcs_21, rx_lcs2gcs_22], axis=-1)],
                    axis=-2)
        rx_lcs2gcs = tf.complex(rx_lcs2gcs, tf.zeros_like(rx_lcs2gcs))

        # List of antenna patterns (callables)
        tx_patterns = self._tx_array.antenna.patterns
        rx_patterns = self._rx_array.antenna.patterns

        # Number of patterns for rx and tx
        num_rx_patterns = len(rx_patterns)
        num_tx_patterns = len(tx_patterns)

        tx_ant_fields_hat = []
        for pattern in tx_patterns:
            # [batch_dim, num_rx, rx_array_size, num_tx, tx_array_size,
            #   ... max_num_paths, 2]
            # or
            # [batch_dim, num_rx, 1, num_tx, 1, max_num_paths, 2]
            tx_ant_f = tf.stack(pattern(theta_prime_t, phi_prime_t), axis=-1)
            tx_ant_fields_hat.append(tx_ant_f)

        rx_ant_fields_hat = []
        for pattern in rx_patterns:
            # [batch_dim, num_rx, rx_array_size, num_tx, tx_array_size,
            #   ... max_num_paths, 2]
            # or
            # [batch_dim, num_rx, 1, num_tx, 1, max_num_paths, 2]
            rx_ant_f = tf.stack(pattern(theta_prime_r, phi_prime_r), axis=-1)
            rx_ant_fields_hat.append(rx_ant_f)

        # Stacking the patterns, corresponding to different polarization
        # directions, as an additional dimension
        # [batch_dim, num_rx, num_rx_patterns, rx_array_size, num_tx,
        #   ... tx_array_size, max_num_paths, 2]
        # or
        # [batch_dim, num_rx, num_rx_patterns, 1, num_tx, 1, max_num_paths, 2]
        rx_ant_fields_hat = tf.stack(rx_ant_fields_hat, axis=2)
        # Expand for broadcasting with tx polarization
        # [batch_dim, num_rx, num_rx_patterns, rx_array_size, num_tx, 1,
        #   ... tx_array_size, max_num_paths, 2]
        # or
        # [batch_dim, num_rx, num_rx_patterns, 1, num_tx, 1, 1,
        #   ... max_num_paths, 2]
        rx_ant_fields_hat = tf.expand_dims(rx_ant_fields_hat, axis=5)

        # Stacking the patterns, corresponding to different polarization
        # [batch_dim, num_rx, rx_array_size, num_tx, num_tx_patterns,
        #   ... tx_array_size, max_num_paths, 2]
        # or
        # [batch_dim, num_rx, 1, num_tx, num_tx_patterns, 1, max_num_paths, 2]
        tx_ant_fields_hat = tf.stack(tx_ant_fields_hat, axis=4)
        # Expand for broadcasting with rx polarization
        # [batch_dim, num_rx, 1, rx_array_size, num_tx, num_tx_patterns,
        #   ... tx_array_size, max_num_paths, 2]
        # or
        # [batch_dim, num_rx, 1, 1, num_tx, num_tx_patterns, 1, max_num_paths,2]
        tx_ant_fields_hat = tf.expand_dims(tx_ant_fields_hat, axis=2)

        # Antenna patterns to spherical global coordinate system
        # Expand to broadcast with antenna patterns
        rx_lcs2gcs = tf.expand_dims(tf.expand_dims(rx_lcs2gcs, axis=2), axis=5)
        # [batch_dim, num_rx, num_rx_patterns, rx_array_size, num_tx, 1,
        #   ... tx_array_size, max_num_paths, 2]
        # or
        # [batch_dim, num_rx, num_rx_patterns, 1, num_tx, 1, 1,
        #   ... max_num_paths, 2]
        rx_ant_fields = tf.linalg.matvec(rx_lcs2gcs, rx_ant_fields_hat)
        # Expand to broadcast with antenna patterns
        tx_lcs2gcs = tf.expand_dims(tf.expand_dims(tx_lcs2gcs, axis=2), axis=5)
        # [batch_dim, num_rx, num_rx_patterns, rx_array_size, num_tx, 1,
        #   ... tx_array_size, max_num_paths, 2]
        # or
        # [batch_dim, num_rx, num_rx_patterns, 1, num_tx, 1, 1,
        #   ... max_num_paths, 2]
        tx_ant_fields = tf.linalg.matvec(tx_lcs2gcs, tx_ant_fields_hat)

        # Expand the field to broadcast with the antenna patterns
        # [batch_dim, num_rx, 1, rx_array_size, num_tx, 1, tx_array_size,
        #   ... max_num_paths, 2, 2]
        h = tf.expand_dims(tf.expand_dims(h, axis=2), axis=5)

        # [batch_dim, num_rx, num_rx_patterns, rx_array_size, num_tx,
        #   ... num_tx_patterns, tx_array_size, max_num_paths]
        h = tf.linalg.matvec(h, tx_ant_fields)
        h = dot(rx_ant_fields, h)

        ######################################################
        # Reshaping to match Sionna shapes convention
        ######################################################

        # Reshape as expected to merge antenna and antenna patterns into one
        # dimension, as expected by Sionna
        # [batch_dim, num_rx, num_rx_ant = num_rx_patterns*rx_array_size,
        #   num_tx, num_tx_ant = num_tx_patterns*tx_array_size, max_num_paths]
        h = flatten_dims(flatten_dims(h, 2, 2), 2, 4)

        # Normalize path delays such that the first path between any
        # pair of antennas of a transmitter and receiver arrives at tau=0
        if self._normalize_delays:
            if self._synthetic_array:
                tau -= tf.reduce_min(tf.abs(tau), axis=3, keepdims=True)
            else:
                tau -= tf.reduce_min(tf.abs(tau), axis=[2,4,5], keepdims=True)

        # Tile delays to handle dual-polarized antennas
        if not self._synthetic_array:
            # Add extra dimensions to handle multiple antenna patterns, i.e.,
            # polarizations
            # [batch_dim, num_rx, 1, rx_array_size, num_tx, 1,
            #   ... tx_array_size, max_num_paths]
            tau = tf.expand_dims(tf.expand_dims(tau, axis=2), axis=5)
            # Tile delays, as all polarization angles share the same delay
            # [batch_dim, num_rx, num_rx_patterns, rx_array_size, num_tx,
            #   ... num_tx_patterns, tx_array_size, max_num_paths]
            tau = tf.tile(tau, [1, 1, num_rx_patterns, 1, 1,
                                num_tx_patterns, 1, 1])
            # Reshape as expected to merge antenna and antenna patterns into one
            # dimension, as expected by Sionna
            # [batch_dim, num_rx, num_rx_ant = num_rx_patterns*num_rx_ant,
            #   ... num_tx, num_tx_ant = num_tx_patterns*tx_array_size,
            #   ... max_num_paths]
            tau = flatten_dims(flatten_dims(tau, 2, 2), 2, 4)

        ######################################################
        # Compute and apply Doppler shift
        ######################################################

        # Expand rank of the speed vector for broadcasting with k_r
        # [batch_dim, 1, 1, num_tx, 1, 1, 3]
        tx_velocities = insert_dims(insert_dims(self._tx_velocities, 2, 1),2, 4)
        # [batch_dim, num_rx, 1, 1, 1, 1, 3]
        rx_velocities = insert_dims(self._rx_velocities, 4, 1)

        # Generate time steps
        # [num_time_steps]
        ts = tf.range(self._num_time_steps, dtype=dtype.real_dtype)
        ts = ts / self._sampling_frequency

        # Compute the Doppler shift
        # [batch_dim, num_rx, num_rx_ant, num_tx, num_tx_ant, max_num_paths]
        # or
        # [batch_dim, num_rx, 1, num_tx, 1, max_num_paths]
        tx_ds = two_pi*dot(tx_velocities, k_t)/self._wavelength
        rx_ds = two_pi*dot(rx_velocities, k_r)/self._wavelength
        ds = tx_ds + rx_ds
        # Expand for the time sample dimension
        # [batch_dim, num_rx, num_rx_ant, num_tx, num_tx_ant, max_num_paths, 1]
        # or
        # [batch_dim, num_rx, 1, num_tx, 1, max_num_paths, 1]
        ds = tf.expand_dims(ds, axis=-1)
        # Expand time steps for broadcasting
        # [1, 1, 1, 1, 1, 1, num_time_steps]
        ts = expand_to_rank(ts, tf.rank(ds), 0)
        # [batch_dim, num_rx, num_rx_ant, num_tx, num_tx_ant, max_num_paths,
        #   num_time_steps]
        # or
        # [batch_dim, num_rx, 1, num_tx, 1, max_num_paths, num_time_steps]
        ds = ds*ts
        exp_ds = tf.exp(tf.complex(zeror, ds))
        if not self._synthetic_array:
            # Tile to handle dual-polarized antennas
            # Add extra dimensions to handle multiple antenna patterns, i.e.,
            # polarizations
            # [batch_dim, num_rx, 1, num_rx_ant, num_tx, 1, num_tx_ant,
            #   ... max_num_paths, num_time_steps]
            exp_ds = tf.expand_dims(tf.expand_dims(exp_ds, axis=2), axis=5)
            # Tile as all polarization angles share the same delay
            # [batch_dim, num_rx, num_rx_patterns, num_rx_ant, num_tx,
            #   ... num_tx_patterns, num_tx_ant, max_num_paths, num_time_steps]
            exp_ds = tf.tile(exp_ds, [1, 1, num_rx_patterns, 1, 1,
                                      num_tx_patterns, 1, 1, 1])
            # Reshape as expected
            # [batch_dim, num_rx, num_rx_patterns*num_rx_ant, num_tx,
            #   num_tx_patterns*um_tx_ant, max_num_paths, num_time_steps]
            exp_ds = flatten_dims(flatten_dims(exp_ds, 2, 2), 2, 4)

        # Apply Doppler shift
        # Expand with time dimension
        # [batch_dim, num_rx, num_rx_ant, num_tx, num_tx_ant, max_num_paths, 1]
        h = tf.expand_dims(h, axis=-1)
        if self._synthetic_array:
            # Broadcast is not supported by TF for such high rank tensors.
            # We therefore do it manually
            # [batch_dim, num_rx, num_rx_ant, num_tx, num_tx_ant, max_num_paths,
            #   num_time_steps]
            h = tf.tile(h, [1, 1, 1, 1, 1, 1, exp_ds.shape[6]])
        # [batch_dim, num_rx,  num_rx_ant, num_tx, num_tx_ant, max_num_paths,
        #   num_time_steps]
        h = h*exp_ds

        if self._reverse_direction:
            # swap num_rx against num_tx and num_rx_ant against num_tx_ant
            h = tf.transpose(h, perm=[0,3,4,1,2,5,6])
            if tf.rank(tau)==4:
                tau = tf.transpose(tau, perm=[0,2,1,3])
            else:
                tau = tf.transpose(tau, perm=[0,3,4,1,2,5])

        return h, tau
