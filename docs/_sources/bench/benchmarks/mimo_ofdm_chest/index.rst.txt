MIMO-OFDM Channel Estimation
============================

This benchmark evaluates channel estimators on the uplink of a multi-user MIMO OFDM
system. The system is configured with a single cell and no inter-cell interference.
Transmission takes place between 4 single-antenna UEs and a 16-antenna BS.
It uses an OFDM resource grid consisting of 14 OFDM symbols and 624 subcarriers,
where pilots are placed on the 3rd and 12th symbol using a
:class:`~sionna.phy.ofdm.KroneckerPilotPattern`.
Channel estimators are required to interpolate across the entire resource grid.

The benchmark considers three metrics for channel estimation accuracy:

.. |br| raw:: html

   <br />

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Metric
     - Definition
   * - Normalized Mean Squared Error (NMSE)
     - Measures the average (over Monte Carlo trials) total energy of
       ``h_true - h_hat`` over receive antennas, UEs, subcarriers, and OFDM
       symbols, divided by the average total energy of ``h_true``.
   * - Bit Error Rate (BER)
     - Measures the average error rate between transmitted and decoded
       information bits.
   * - Block Error Rate (BLER)
     - Measures the average rate of wrongly decoded codewords.

While NMSE measures the raw reconstruction accuracy of a channel estimator,
BER and BLER assess the composite performance of the channel estimator with
subsequent LMMSE data detection and LDPC-based forward-error correction. Importantly,
Sionna's :class:`~sionna.phy.ofdm.LinearDetector` also requires estimates ``err_var`` of the
marginal error variances of the channel estimate ``h_hat``.

Random seeds are controled to ensure that all channel estimators are evaluated on
the same channels, transmitted bits, and noise realizations.
The same realizations of these quantities are also used across all SNR points;
the only quantity that changes between SNR points is the noise scaling.

In contrast, the data assets used for training or statistical fitting of channel
estimators are not reused across channel estimators. Every asset has a deterministic
generation recipe that is documented in the doc-page of the channel estimator to which
the asset belongs. Assets are generated with the same system configuration and channel
model that is used for the evaluation.

Data assets and weights of neural-network-based channel estimators (or statistical parameters
of model-based channel estimators) are not included in the public distribution. However,
the distribution contains routines to regenerate the data assets and retrain/refit the
models locally, if needed.

System Configuration
********************

The system is configured with the following parameters:

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Parameter
     - Value
     - Remark
   * - Channel model
     - 3GPP 38.901 :class:`~sionna.phy.channel.tr38901.UMi`
     -
   * - Carrier frequency
     - 2.6 GHz
     -
   * - Outdoor-to-indoor model
     - Low-loss
     -
   * - Pathloss
     - Off
     - Large-scale fading coefficient is 1.
   * - Shadow fading
     - Off
     - Large-scale fading coefficient is 1.
   * - Array size
     - 4 x 4 antennas (BS), |br| 1 x 1 antenna (UE)
     -
   * - Antenna spacing
     - 0.5 wavelengths
     - Vertical and horizontal.
   * - Antenna polarization
     - Vertical (single)
     - BS and UE.
   * - Antenna pattern
     - 38.901 (BS), |br| omnidirectional (UE)
     -
   * - Minimum BS-to-UE distance
     - 10 m
     -
   * - Inter-site distance
     - 200 m
     - Implies a maximum BS-to-UE distance of 115.5 m under
   * - Height
     - 10 m (BS), |br| 1.5 m (UE)
     -
   * - Indoor probability
     - 80%
     -
   * - UE velocity
     - 3 km/h
     - Upper and lower bound
   * -
     -
     -
   * - Number of UEs
     - 4
     -
   * - Streams per UE
     - 1
     -
   * - Number of subcarriers
     - 624
     -
   * - Guard carriers
     - None
     -
   * - Subcarrier spacing
     - 15 kHz
     - 10 MHz of total bandwidth
   * - Cyclic prefix length
     - 44
     - No operational meaning, channels are simulated in frequency-domain
   * - OFDM symbols per slot
     - 14
     -
   * - Constellation
     - 16QAM
     -
   * - Pilot symbol positions
     - 2, 11
     - Time-domain interpolation necessary
   * - Pilots
     - :class:`~sionna.phy.ofdm.KroneckerPilotPattern`
     - UE1's pilots are supported on subcarriers 0, 4, 8, ..., UE2's on 1, 5, 9, ..., etc.
   * - Error correction
     - :class:`~sionna.phy.fec.ldpc.LDPC5GEncoder` and :class:`~sionna.phy.fec.ldpc.LDPC5GDecoder`; three codewords per UE and stream, each with k=5990 and n=9984
     - R :math:`\approx` 0.6

Each UE and stream has 7488 data resource elements, corresponding to 29952
coded bits with 16QAM. The benchmark divides these bits evenly among three
independently encoded LDPC codewords. The resulting code rate is
5990/9984 :math:`\approx` 0.6, and the three codewords fill the resource grid
without padding. This uses Sionna's :class:`~sionna.phy.fec.ldpc.LDPC5GEncoder`
construction and rate matching, but does not implement the complete NR
transport-block CRC and code-block segmentation procedure.


Implementations
***************

Channel estimators that separate pilot-domain estimation from interpolation inherit Sionna's
:class:`~sionna.phy.ofdm.BasePilotChannelEstimator`, implement ``estimate_at_pilot_locations``, and use
an explicit interpolator. Methods that natively reconstruct complete
subcarrier vectors or the full resource grid inherit Sionna's :class:`~sionna.phy.ofdm.BaseChannelEstimator` and implement
``call(y, no)`` directly (in the former case, with internal time-domain interpolation).
Proposed estimators that do not consider interpolation and assume pilots on every subcarrier
use the pilot-domain contract with explicit interpolation.

All channel estimators are required to provide uncertainty estimates in the form of ``err_var``.
For methods from the literature where no such uncertainty estimate is specified,
and where analytic derivation is not straightforward, implementations return
``err_var = err_var_scale * ls_err_var``, where the second factor is the tensor of error variances
that would result from LS channel estimation, and the first factor is a scalar hyperparameter that
is tuned empirically.

Many channel estimators that depend on channel data, either to extract channel statistics,
or to train a neural network. For such channel estimators, corresponding data set
(henceforth referred to as `asset`) are generated using the same channel model and system configuration
as is used for the benchmark:
- Covariance assets consist of 64'000 complete system-channel realizations.
- GMM assets consist of 6,400 system-channel realizations.
- Neural-network assets consist of 6,400 and 256 system-channel realizations for training validation, respectively.

These assets are then used to generate the observation tensors in the form required by the corresponding
channel estimator (e.g., antenna vectors, subcarrier vectors, subcarrier-symbol matrices, etc.).
Observations are typically pooled across all dimensions that are not part of the observation tensor,
i.e., statistics are assumed to be identical across all dimensions that are not part of the observation tensor.

Neural networks train a separate set of weights for every SNR point at which the benchmark is evaluated.
For each SNR point, the same underlying channel asset is used.
Training runs for at most 200 epochs and uses a patience of 20 epochs without improvement.
Loss functions and optimizers are specific to the respective methods and papers.

Most channel estimators contain performance-relevant hyperparameters whose optimal choice
depends on the system and channel models. For such channel estimators, hyperparameter optimization is performed
empirically with Bayesian optimization using `Optuna <https://optuna.org>`_.
Hyperparameters are shared across all SNR points. To balance performance across the entire SNR range,
the objective function is a weighted average of the NMSE at the five SNR points -10, -5, 0, 5, and 10 dB.
To prevent the low-SNR NMSE from dominating the objective function, the weights are selected as
10^(SNR/10). Statistically fitted estimators are refitted for every Optuna trial, and neural-network
based estimators are retrained for every Optuna trial.


.. toctree::
   :hidden:
   :maxdepth: 5

   leaderboard
   estimators/index