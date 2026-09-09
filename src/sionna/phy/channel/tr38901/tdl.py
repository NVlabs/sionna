#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Tapped delay line (TDL) channel model from 3GPP TR38.901 specification"""

from typing import Optional, Tuple

import numpy as np
import torch

from sionna.phy import PI, SPEED_OF_LIGHT, config
from sionna.phy.utils import insert_dims, expand_to_rank, split_dim, flatten_last_dims, rand
from sionna.phy.channel import ChannelModel

from . import models

_TDL_PARAMETER_FILES = {
    "A": ("TDL-A.json", None),
    "B": ("TDL-B.json", None),
    "C": ("TDL-C.json", None),
    "D": ("TDL-D.json", None),
    "E": ("TDL-E.json", None),
    "A10": ("TDL-A10.json", 10e-9),
    "A30": ("TDL-A30.json", 30e-9),
    "B100": ("TDL-B100.json", 100e-9),
    "C60": ("TDL-C60.json", 60e-9),
    "C300": ("TDL-C300.json", 300e-9),
    "D10": ("TDL-D10.json", 10e-9),
    "D30": ("TDL-D30.json", 30e-9),
}


class TDL(ChannelModel):
    # pylint: disable=line-too-long
    r"""
    Tapped delay line (TDL) channel model from the 3GPP :cite:p:`TR38901V1920` specification

    The power delay profiles (PDPs) are normalized to have a total energy of one.

    Two families of delay profiles are available. They differ in how the tap
    delays are obtained.

    The *scalable* profiles ``"A"`` to ``"E"`` from TR 38.901 tabulate tap
    delays that are normalized to an RMS delay spread of one. The delays used
    in the simulation are these normalized values multiplied by
    ``delay_spread``, so a single profile shape describes short and long delay
    spreads alike, and ``delay_spread`` is what represents the propagation
    scenario. Typical values are tabulated under Notes below.

    The *fixed-delay* profiles ``"A10"`` to ``"D30"`` from Annex B.2.1 of
    :cite:p:`TS38101-4` tabulate absolute tap delays in nanoseconds and are
    defined for UE conformance and performance testing. Their RMS delay spread
    is part of the profile definition and is stated in nanoseconds by the model
    name, for example 30 ns for ``"A30"``. Since the tap delays are absolute,
    they are used as tabulated and ``delay_spread`` scales nothing. It can
    therefore be omitted for these profiles; a value that differs from the one
    of the profile is overridden at construction, with a message printed, and
    later assignments to :attr:`delay_spread` are ignored. Reading the property
    returns the delay spread of the profile.
    These profiles are normally referenced together with
    a maximum Doppler frequency, as listed in the Doppler combination tables
    below.

    .. note::

        The scalable TR 38.901 models apply from 0.5 GHz to 100 GHz and for
        system bandwidths of at most 2 GHz. These applicability limits are not
        enforced by this class.

    Channel coefficients are generated using a sum-of-sinusoids model :cite:p:`SoS`.
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
    (see, e.g., Annex G.2.3.2.1 :cite:p:`TS38141-1`):

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

    :param model: TDL model to use. Must be one of ``"A"``, ``"B"``,
        ``"C"``, ``"D"``, ``"E"``, ``"A10"``, ``"A30"``, ``"B100"``,
        ``"C60"``, ``"C300"``, ``"D10"``, or ``"D30"``. The models
        ``"A"`` to ``"E"`` are the scalable TR 38.901 TDL profiles. The
        models ``"A10"``, ``"A30"``, ``"B100"``, ``"C60"``, ``"C300"``,
        ``"D10"``, and ``"D30"`` are the fixed-delay
        Annex B.2.1 delay profiles from :cite:p:`TS38101-4`.
    :param delay_spread: RMS delay spread [s]. Required for the scalable
        profiles ``"A"`` to ``"E"``, whose normalized tap delays it scales. The
        fixed-delay TS 38.101-4 models tabulate absolute tap delays, so this
        parameter can be omitted for them; it then takes the delay spread of
        the profile, which is ``10 ns`` for ``"A10"`` and ``"D10"``, ``30 ns``
        for ``"A30"`` and ``"D30"``, ``60 ns`` for ``"C60"``, ``100 ns`` for
        ``"B100"``, and ``300 ns`` for ``"C300"``. A value that differs from
        the one of the profile is discarded and a message is printed.
    :param carrier_frequency: Carrier frequency [Hz]
    :param num_sinusoids: Number of sinusoids for the sum-of-sinusoids model. Defaults to 20.
    :param los_angle_of_arrival: Angle-of-arrival for LoS path [radian]. Only used with LoS models.
        Defaults to ``arccos(0.7)``, placing the LoS Doppler peak at 0.7 times
        the maximum Doppler shift as specified by
        :cite:p:`TR38901V1920`.
    :param min_speed: Minimum speed [m/s]. Defaults to 0.0.
    :param max_speed: Maximum speed [m/s]. If set to `None`,
        then ``max_speed`` takes the same value as ``min_speed``.
    :param num_rx_ant: Number of receive antennas. Defaults to 1.
    :param num_tx_ant: Number of transmit antennas. Defaults to 1.
    :param spatial_corr_mat: Spatial correlation matrix of shape
        [num_rx_ant*num_tx_ant, num_rx_ant*num_tx_ant].
        If not set to `None`, then ``rx_corr_mat`` and ``tx_corr_mat`` are ignored and
        this matrix is used for spatial correlation.
        If set to `None` and ``rx_corr_mat`` and ``tx_corr_mat`` are also set to `None`,
        then no correlation is applied.
    :param rx_corr_mat: Spatial correlation matrix for the receiver of shape
        [num_rx_ant, num_rx_ant].
        If set to `None` and ``spatial_corr_mat`` is also set to `None`, then no receive
        correlation is applied.
    :param tx_corr_mat: Spatial correlation matrix for the transmitter of shape
        [num_tx_ant, num_tx_ant].
        If set to `None` and ``spatial_corr_mat`` is also set to `None`, then no transmit
        correlation is applied.
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`, :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for computation (e.g., 'cpu', 'cuda:0').
        If `None`, :attr:`~sionna.phy.config.Config.device` is used.
    :param spec_version: Version of the TR 38.901 parameter tables to use.
        Supported values are ``"16.1"`` and ``"19.2"``. Defaults to
        ``"19.2"``. This selector applies only to the scalable TDL-A through
        TDL-E profiles, not to the fixed TS 38.101-4 profiles.

    :input batch_size: `int`.
        Batch size.

    :input num_time_steps: `int`.
        Number of time steps.

    :input sampling_frequency: `float`.
        Sampling frequency [Hz].

    :output a: [batch size, num_rx=1, num_rx_ant, num_tx=1, num_tx_ant, num_paths, num_time_steps], `torch.complex`.
        Path coefficients.

    :output tau: [batch size, num_rx=1, num_tx=1, num_paths], `torch.float`.
        Path delays [s].

    .. rubric:: Examples

    The following code snippet shows how to set up a TDL channel model and
    generate channel impulse responses:

    .. code-block:: python

        import torch
        from sionna.phy.channel.tr38901 import TDL

        device = "cuda:0" if torch.cuda.is_available() else "cpu"

        channel_model = TDL(model="A",
                            delay_spread=300e-9,
                            carrier_frequency=3.5e9,
                            min_speed=0.0,
                            max_speed=3.0,
                            num_rx_ant=2,
                            num_tx_ant=1,
                            device=device)

        h, tau = channel_model(batch_size=32,
                               num_time_steps=14,
                               sampling_frequency=30.72e6)

    .. rubric:: Available Model Options

    +------------+------------------+---------------------+---------------------+
    | ``model``  | Source           | Delay handling      | LoS component       |
    +============+==================+=====================+=====================+
    | ``"A"``    | TR 38.901        | Scaled by           | No                  |
    |            |                  | ``delay_spread``    |                     |
    +------------+------------------+---------------------+---------------------+
    | ``"B"``    | TR 38.901        | Scaled by           | No                  |
    |            |                  | ``delay_spread``    |                     |
    +------------+------------------+---------------------+---------------------+
    | ``"C"``    | TR 38.901        | Scaled by           | No                  |
    |            |                  | ``delay_spread``    |                     |
    +------------+------------------+---------------------+---------------------+
    | ``"D"``    | TR 38.901        | Scaled by           | Yes                 |
    |            |                  | ``delay_spread``    |                     |
    +------------+------------------+---------------------+---------------------+
    | ``"E"``    | TR 38.901        | Scaled by           | Yes                 |
    |            |                  | ``delay_spread``    |                     |
    +------------+------------------+---------------------+---------------------+
    | ``"A10"``  | TS 38.101-4      | Fixed, 10 ns RMS    | No                  |
    +------------+------------------+---------------------+---------------------+
    | ``"A30"``  | TS 38.101-4      | Fixed, 30 ns RMS    | No                  |
    +------------+------------------+---------------------+---------------------+
    | ``"B100"`` | TS 38.101-4      | Fixed, 100 ns RMS   | No                  |
    +------------+------------------+---------------------+---------------------+
    | ``"C60"``  | TS 38.101-4      | Fixed, 60 ns RMS    | No                  |
    +------------+------------------+---------------------+---------------------+
    | ``"C300"`` | TS 38.101-4      | Fixed, 300 ns RMS   | No                  |
    +------------+------------------+---------------------+---------------------+
    | ``"D10"``  | TS 38.101-4      | Fixed, 10 ns RMS    | Yes                 |
    +------------+------------------+---------------------+---------------------+
    | ``"D30"``  | TS 38.101-4      | Fixed, 30 ns RMS    | Yes                 |
    +------------+------------------+---------------------+---------------------+

    The fixed-delay resources contain the profiles standardized by TS 38.101-4
    V19.2.2. Their release and frequency-range availability is:

    .. list-table:: Fixed-delay profile availability
       :header-rows: 1

       * - Specification
         - FR1 profiles
         - FR2 profiles
         - Restriction
       * - TS 38.101-4 V16.1
         - ``"A30"``, ``"B100"``, ``"C300"``
         - ``"A30"``, ``"C60"``
         - None
       * - Added by TS 38.101-4 V19.2.2
         - ``"D30"``
         - ``"D30"``, ``"A10"``, ``"D10"``
         - ``"A10"`` and ``"D10"`` apply only for channel bandwidths
           greater than 200 MHz

    The class has no channel-bandwidth argument and does not enforce these
    fixed-profile restrictions. Select a profile appropriate for the simulated
    frequency range and bandwidth.

    .. rubric:: TS 38.101-4 Doppler Combinations

    Annex B.2.2 of TS 38.101-4 defines full propagation-condition names by
    combining a fixed delay profile with a maximum Doppler frequency. These
    full names are not passed to ``model`` directly. Instead, select the delay
    profile with ``model`` and set ``min_speed`` and ``max_speed`` to realize
    the desired Doppler frequency.

    .. list-table:: FR1 channel model parameters from TS 38.101-4 Table B.2.2-1
       :header-rows: 1

       * - TS 38.101-4 name
         - ``model``
         - :math:`f_{\mathrm{D,max}}`
       * - ``TDLA30-5``
         - ``"A30"``
         - 5 Hz
       * - ``TDLA30-10``
         - ``"A30"``
         - 10 Hz
       * - ``TDLA30-20``
         - ``"A30"``
         - 20 Hz
       * - ``TDLA30-180``
         - ``"A30"``
         - 180 Hz
       * - ``TDLA30-195``
         - ``"A30"``
         - 195 Hz
       * - ``TDLA30-1400``
         - ``"A30"``
         - 1400 Hz
       * - ``TDLA30-2700``
         - ``"A30"``
         - 2700 Hz
       * - ``TDLB100-400``
         - ``"B100"``
         - 400 Hz
       * - ``TDLC300-100``
         - ``"C300"``
         - 100 Hz
       * - ``TDLC300-600``
         - ``"C300"``
         - 600 Hz
       * - ``TDLC300-1200``
         - ``"C300"``
         - 1200 Hz
       * - ``TDLD30-5``
         - ``"D30"``
         - 5 Hz

    .. list-table:: FR2 channel model parameters from TS 38.101-4 Table B.2.2-2
       :header-rows: 1

       * - TS 38.101-4 name
         - ``model``
         - :math:`f_{\mathrm{D,max}}`
       * - ``TDLA30-35``
         - ``"A30"``
         - 35 Hz
       * - ``TDLA30-75``
         - ``"A30"``
         - 75 Hz
       * - ``TDLA30-300``
         - ``"A30"``
         - 300 Hz
       * - ``TDLC60-300``
         - ``"C60"``
         - 300 Hz
       * - ``TDLD30-75``
         - ``"D30"``
         - 75 Hz

    .. rubric:: Notes

    The scalable profiles do not define a delay spread of their own, so a value
    must be chosen for ``delay_spread``. Table 7.7.3-1 of
    :cite:p:`TR38901V1920` provides example scaling parameters for this purpose,
    reproduced below. They span the range of RMS delay spreads observed in
    measurements for the typical 5G evaluation scenarios.

    +--------------------------+-------------------+
    | Model                    | Delay spread [ns] |
    +==========================+===================+
    | Very short delay spread  | :math:`10`        |
    +--------------------------+-------------------+
    | Short delay spread       | :math:`30`        |
    +--------------------------+-------------------+
    | Nominal delay spread     | :math:`100`       |
    +--------------------------+-------------------+
    | Long delay spread        | :math:`300`       |
    +--------------------------+-------------------+
    | Very long delay spread   | :math:`1000`      |
    +--------------------------+-------------------+

    Table 7.7.3-2, reproduced below, relates these values to scenarios and
    carrier frequencies, for information only. The short-delay profile is the
    median RMS delay spread of the LoS case, while the normal-delay and
    long-delay profiles are the median and the 90th percentile of the NLoS case.
    A TDL profile is not tied to a scenario: any of these delay spreads may
    occur in any scenario, but some values are more likely in some scenarios
    than in others. The table does not apply to the fixed-delay profiles, whose
    delay spread is part of the profile definition.

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
    |                        | Long delay profile   | 1148 | 1148 | 955| 841 | 786 | 720| 698 |
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

    .. note::

        TS 38.101-4 model names combine a fixed delay profile with a maximum
        Doppler frequency, e.g., ``TDLA30-10``. In this class, ``model``
        selects only the fixed delay profile, e.g., ``model="A30"``. The
        ``min_speed`` and ``max_speed`` arguments are still used and determine
        the Doppler spread through ``carrier_frequency``. To emulate a TS
        38.101-4 model with maximum Doppler frequency
        :math:`f_\mathrm{D,max}`, use
        ``max_speed = f_D_max*c/carrier_frequency``, where :math:`c` is the
        speed of light. Set ``min_speed`` to the same value for a deterministic
        speed, or to a lower value to sample speeds uniformly over an interval.
    """

    def __init__(
        self,
        model: str,
        delay_spread: Optional[float] = None,
        carrier_frequency: Optional[float] = None,
        num_sinusoids: int = 20,
        los_angle_of_arrival: float = np.arccos(0.7),
        min_speed: float = 0.,
        max_speed: Optional[float] = None,
        num_rx_ant: int = 1,
        num_tx_ant: int = 1,
        spatial_corr_mat: Optional[torch.Tensor] = None,
        rx_corr_mat: Optional[torch.Tensor] = None,
        tx_corr_mat: Optional[torch.Tensor] = None,
        precision: Optional[str] = None,
        device: Optional[str] = None,
        spec_version: str = "19.2",
    ) -> None:
        super().__init__(precision=precision, device=device)
        self._spec_version = models._validate_spec_version(spec_version)

        # carrier_frequency only defaults to None so that delay_spread, which
        # precedes it, can be omitted for the fixed-delay profiles
        if carrier_frequency is None:
            raise TypeError("carrier_frequency is required")

        # Set the file from which to load the model
        if model not in _TDL_PARAMETER_FILES:
            raise ValueError(
                f"model must be one of {list(_TDL_PARAMETER_FILES)}"
            )
        parameters_fname, fixed_delay_spread = _TDL_PARAMETER_FILES[model]
        if fixed_delay_spread is None:
            if delay_spread is None:
                raise TypeError("delay_spread is required for the scalable "
                                f"model '{model}'")
        elif delay_spread is None:
            delay_spread = fixed_delay_spread
        elif delay_spread != fixed_delay_spread:
            delay_spread_ns = int(round(fixed_delay_spread * 1e9))
            print(f"Warning: Delay spread is set to {delay_spread_ns}ns with this model")
            delay_spread = fixed_delay_spread

        # Load model parameters
        self._load_parameters(
            parameters_fname,
            fixed_profile=fixed_delay_spread is not None,
        )

        self._num_rx_ant = num_rx_ant
        self._num_tx_ant = num_tx_ant
        # Register as buffers for CUDAGraph compatibility
        self.register_buffer("_carrier_frequency", torch.tensor(carrier_frequency, dtype=self.dtype, device=self.device))
        self._num_sinusoids = num_sinusoids
        self.register_buffer("_los_angle_of_arrival", torch.tensor(los_angle_of_arrival, dtype=self.dtype, device=self.device))
        self.register_buffer("_delay_spread", torch.tensor(delay_spread, dtype=self.dtype, device=self.device))
        self.register_buffer("_min_speed", torch.tensor(min_speed, dtype=self.dtype, device=self.device))
        if max_speed is None:
            self.register_buffer("_max_speed", self._min_speed.clone())
        else:
            if not (max_speed >= min_speed):
                raise ValueError(
                    "min_speed cannot be larger than max_speed"
                )
            self.register_buffer("_max_speed", torch.tensor(max_speed, dtype=self.dtype, device=self.device))

        # Pre-compute maximum and minimum Doppler shifts
        self._min_doppler = self._compute_doppler(self._min_speed)
        self._max_doppler = self._compute_doppler(self._max_speed)

        # Precompute average angles of arrivals for each sinusoid
        alpha_const = 2. * PI / num_sinusoids * \
                      torch.arange(1., self._num_sinusoids + 1, 1., dtype=self.dtype, device=self.device)
        self._alpha_const = alpha_const.reshape(
            1,  # batch size
            1,  # num rx
            1,  # num rx ant
            1,  # num tx
            1,  # num tx ant
            1,  # num clusters
            1,  # num time steps
            num_sinusoids
        )

        # Precompute square root of spatial covariance matrices
        # Use cholesky_ex for CUDA graph compatibility
        if spatial_corr_mat is not None:
            spatial_corr_mat = torch.as_tensor(spatial_corr_mat, dtype=self.cdtype, device=self.device)
            spatial_corr_mat_sqrt, _ = torch.linalg.cholesky_ex(spatial_corr_mat, check_errors=False)
            spatial_corr_mat_sqrt = expand_to_rank(spatial_corr_mat_sqrt, 7, 0)
            self._spatial_corr_mat_sqrt = spatial_corr_mat_sqrt
        else:
            self._spatial_corr_mat_sqrt = None
            if rx_corr_mat is not None:
                rx_corr_mat = torch.as_tensor(rx_corr_mat, dtype=self.cdtype, device=self.device)
                rx_corr_mat_sqrt, _ = torch.linalg.cholesky_ex(rx_corr_mat, check_errors=False)
                rx_corr_mat_sqrt = expand_to_rank(rx_corr_mat_sqrt, 7, 0)
                self._rx_corr_mat_sqrt = rx_corr_mat_sqrt
            else:
                self._rx_corr_mat_sqrt = None
            if tx_corr_mat is not None:
                tx_corr_mat = torch.as_tensor(tx_corr_mat, dtype=self.cdtype, device=self.device)
                tx_corr_mat_sqrt, _ = torch.linalg.cholesky_ex(tx_corr_mat, check_errors=False)
                tx_corr_mat_sqrt = expand_to_rank(tx_corr_mat_sqrt, 7, 0)
                self._tx_corr_mat_sqrt = tx_corr_mat_sqrt
            else:
                self._tx_corr_mat_sqrt = None

    @property
    def num_clusters(self) -> int:
        r"""Number of paths (:math:`M`)"""
        return self._num_clusters

    @property
    def spec_version(self) -> str:
        """Version of the TR 38.901 parameter tables in use."""
        return self._spec_version

    @property
    def los(self) -> bool:
        r"""`True` if this is a LoS model. `False` otherwise."""
        return self._los

    @property
    def k_factor(self) -> torch.Tensor:
        r"""K-factor in linear scale. Only available with LoS models."""
        if not self._los:
            raise RuntimeError(
                "This property is only available for LoS models"
            )
        return torch.real(self._los_power / self._mean_powers[0])

    @property
    def delays(self) -> torch.Tensor:
        r"""Path delays [s]"""
        if self._scale_delays:
            return self._delays * self._delay_spread
        else:
            return self._delays * 1e-9  # ns to s

    @property
    def mean_powers(self) -> torch.Tensor:
        r"""Path powers in linear scale"""
        if self._los:
            mean_powers = torch.cat([self._mean_powers[:1] + self._los_power,
                                      self._mean_powers[1:]], dim=0)
        else:
            mean_powers = self._mean_powers
        return torch.real(mean_powers)

    @property
    def mean_power_los(self) -> torch.Tensor:
        r"""LoS component power in linear scale.
        Only available with LoS models."""
        if not self._los:
            raise RuntimeError(
                "This property is only available for LoS models"
            )
        return torch.real(self._los_power)

    @property
    def delay_spread(self) -> torch.Tensor:
        r"""RMS delay spread [s]"""
        return self._delay_spread

    @delay_spread.setter
    def delay_spread(self, value: float) -> None:
        if self._scale_delays:
            # Register as buffer for CUDAGraph compatibility
            self.register_buffer("_delay_spread", torch.tensor(value, dtype=self.dtype, device=self.device))
        else:
            print("Warning: The delay spread cannot be set with this model")

    def __call__(
        self,
        batch_size: int,
        num_time_steps: int,
        sampling_frequency: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Time steps
        sample_times = torch.arange(num_time_steps, dtype=self.dtype, device=self.device) \
            / sampling_frequency
        sample_times = insert_dims(sample_times, 6, 0).unsqueeze(-1)

        # Generate random maximum Doppler shifts for each sample
        # The Doppler shift is different for each TX-RX link, but shared by
        # all RX ant and TX ant couple for a given link.
        doppler = rand(
            (batch_size, 1, 1, 1, 1, 1, 1, 1),
            dtype=self.dtype,
            device=self.device,
            generator=self.torch_rng,
        ) * (self._max_doppler - self._min_doppler) + self._min_doppler

        # Eq. (7) in the paper [TDL] (see class docstring)
        # The angle of arrival is different for each TX-RX link.
        theta = rand(
            (batch_size, 1, 1, 1, 1, self._num_clusters, 1, self._num_sinusoids),
            dtype=self.dtype,
            device=self.device,
            generator=self.torch_rng,
        ) * (2 * PI / self._num_sinusoids) - PI / self._num_sinusoids
        alpha = self._alpha_const + theta

        # Eq. (6a)-(6c) in the paper [TDL] (see class docstring)
        phi = rand(
            (batch_size, 1, self._num_rx_ant, 1, self._num_tx_ant,
             self._num_clusters, 1, self._num_sinusoids),
            dtype=self.dtype,
            device=self.device,
            generator=self.torch_rng,
        ) * 2 * PI - PI

        argument = doppler * sample_times * torch.cos(alpha) + phi

        # Eq. (6a) in the paper [SoS]
        h = torch.complex(torch.cos(argument), torch.sin(argument))
        normalization_factor = 1. / torch.sqrt(torch.tensor(self._num_sinusoids, dtype=self.dtype, device=self.device))
        h = torch.complex(normalization_factor, torch.tensor(0., dtype=self.dtype, device=self.device)) \
            * h.sum(dim=-1)

        # Scaling by average power
        mean_powers = insert_dims(self._mean_powers, 5, 0).unsqueeze(-1)
        h = torch.sqrt(mean_powers) * h

        # Add specular component to first tap Eq. (11) in [SoS] if LoS
        if self._los:
            # The first tap follows a Rician
            # distribution

            # Specular component phase shift
            phi_0 = rand(
                (batch_size, 1, 1, 1, 1, 1, 1),
                dtype=self.dtype,
                device=self.device,
                generator=self.torch_rng,
            ) * 2 * PI - PI

            # Remove the sinusoids dim
            doppler = doppler.squeeze(-1)
            sample_times = sample_times.squeeze(-1)
            arg_spec = doppler * sample_times * torch.cos(self._los_angle_of_arrival) \
                + phi_0
            h_spec = torch.complex(torch.cos(arg_spec), torch.sin(arg_spec))

            # Update the first tap with the specular component
            h = torch.cat([h_spec * torch.sqrt(self._los_power) + h[:, :, :, :, :, :1, :],
                            h[:, :, :, :, :, 1:, :]],
                          dim=5)  # Path dim

        # Delays
        if self._scale_delays:
            delays = self._delays * self._delay_spread
        else:
            delays = self._delays * 1e-9  # ns to s
        delays = insert_dims(delays, 3, 0)
        delays = delays.expand(batch_size, -1, -1, -1)

        # Apply spatial correlation if required
        if self._spatial_corr_mat_sqrt is not None:
            h = h.permute(0, 1, 3, 5, 6, 2, 4)  # [..., num_rx_ant, num_tx_ant]
            h = flatten_last_dims(h, 2)  # [..., num_rx_ant*num_tx_ant]
            h = h.unsqueeze(-1)  # [..., num_rx_ant*num_tx_ant, 1]
            h = torch.matmul(self._spatial_corr_mat_sqrt, h)
            h = h.squeeze(-1)
            h = split_dim(h, [self._num_rx_ant, self._num_tx_ant],
                          h.dim() - 1)  # [..., num_rx_ant, num_tx_ant]
            h = h.permute(0, 1, 5, 2, 6, 3, 4)
        else:
            if ((self._rx_corr_mat_sqrt is not None)
                    or (self._tx_corr_mat_sqrt is not None)):
                h = h.permute(0, 1, 3, 5, 6, 2, 4)
                if self._rx_corr_mat_sqrt is not None:
                    h = torch.matmul(self._rx_corr_mat_sqrt, h)
                if self._tx_corr_mat_sqrt is not None:
                    h = torch.matmul(h, self._tx_corr_mat_sqrt.mH)
                h = h.permute(0, 1, 5, 2, 6, 3, 4)

        # Detach to avoid unnecessary gradient computation
        h = h.detach()
        delays = delays.detach()

        return h, delays

    ###########################################
    # Internal utility functions
    ###########################################

    def _compute_doppler(self, speed: torch.Tensor) -> torch.Tensor:
        r"""Compute the maximum radian Doppler frequency [Hz] for a given
        speed [m/s].

        The maximum radian Doppler frequency :math:`\omega_d` is calculated
        as:

        .. math::
            \omega_d = 2\pi  \frac{v}{c} f_c

        where :math:`v` [m/s] is the speed of the receiver relative to the
        transmitter, :math:`c` [m/s] is the speed of light and,
        :math:`f_c` [Hz] the carrier frequency.

        :param speed: Speed [m/s]

        :output doppler: Doppler shift [Hz]
        """
        return 2. * PI * speed / SPEED_OF_LIGHT * self._carrier_frequency

    def _load_parameters(self, fname: str, fixed_profile: bool) -> None:
        r"""Load parameters of a TDL model.

        The model parameters are stored as JSON files with the following keys:
        * los : boolean that indicates if the model is a LoS model
        * num_clusters : integer corresponding to the number of clusters (paths)
        * delays : List of path delays in ascending order normalized by the RMS
            delay spread
        * powers : List of path powers in dB scale

        For LoS models, the two first paths have zero delay, and are assumed
        to correspond to the specular and NLoS component, in this order.

        :param fname: File from which to load the parameters.
        :param fixed_profile: Whether ``fname`` is a fixed TS 38.101-4 profile.
        """
        if fixed_profile:
            source = models.fixed_tdl_parameter_file(fname)
        else:
            source = models.parameter_file(fname, self.spec_version)
        params = models.load_json(source)

        # LoS scenario ?
        self._los = bool(params['los'])

        # Scale the delays
        self._scale_delays = bool(params['scale_delays'])

        # Loading cluster delays and mean powers
        self._num_clusters = int(params['num_clusters'])

        # Retrieve power and delays
        delays = torch.tensor(params['delays'], dtype=self.dtype, device=self.device)
        mean_powers = np.power(10.0, np.array(params['powers']) / 10.0)
        mean_powers = torch.tensor(mean_powers, dtype=self.cdtype, device=self.device)

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
            norm_factor = torch.sum(mean_powers) + self._los_power
            self._los_power = self._los_power / norm_factor
            mean_powers = mean_powers / norm_factor
        else:
            norm_factor = torch.sum(mean_powers)
            mean_powers = mean_powers / norm_factor

        self._delays = delays
        self._mean_powers = mean_powers
