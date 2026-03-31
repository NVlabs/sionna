Orthogonal Frequency-Division Multiplexing (OFDM)
=================================================

.. currentmodule:: sionna.phy.ofdm

This module provides layers and functions to support simulation of OFDM-based
systems. The key component is the :class:`~sionna.phy.ofdm.ResourceGrid` that
defines how data and pilot symbols are mapped onto a sequence of OFDM symbols
with a given FFT size. The resource grid can also define guard and DC carriers
which are nulled. In 4G/5G parlance, a :class:`~sionna.phy.ofdm.ResourceGrid`
would be a slot.

Once a :class:`~sionna.phy.ofdm.ResourceGrid` is defined, one can use the
:class:`~sionna.phy.ofdm.ResourceGridMapper` to map a tensor of complex-valued
data symbols onto the resource grid, prior to OFDM modulation using the
:class:`~sionna.phy.ofdm.OFDMModulator` or further processing in the
frequency domain.

The :class:`~sionna.phy.ofdm.PilotPattern` allows for a fine-grained configuration
of how transmitters send pilots for each of their streams or antennas. As the
management of pilots in multi-cell MIMO setups can quickly become complicated,
the module provides the :class:`~sionna.phy.ofdm.KroneckerPilotPattern` class
that automatically generates orthogonal pilot transmissions for all transmitters
and streams.

Additionally, the module contains blocks for channel estimation, precoding,
equalization, and detection, such as the :class:`~sionna.phy.ofdm.LSChannelEstimator`,
the :class:`~sionna.phy.ofdm.RZFPrecoder`, and the
:class:`~sionna.phy.ofdm.LMMSEEqualizer` and :class:`~sionna.phy.ofdm.LinearDetector`.
These are good starting points for the development of more advanced algorithms
and provide robust baselines for benchmarking.

.. toctree::
   :maxdepth: 2
   :hidden:

   resource_grid
   modulation
   pilot_pattern
   channel_estimation
   precoding
   equalization
   detection
