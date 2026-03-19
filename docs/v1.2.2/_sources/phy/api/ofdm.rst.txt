Orthogonal Frequency-Division Multiplexing (OFDM)
=================================================

This module provides layers and functions to support
simulation of OFDM-based systems. The key component is the
:class:`~sionna.phy.ofdm.ResourceGrid` that defines how data and pilot symbols
are mapped onto a sequence of OFDM symbols with a given FFT size. The resource
grid can also define guard and DC carriers which are nulled. In 4G/5G parlance,
a :class:`~sionna.phy.ofdm.ResourceGrid` would be a slot.
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
equalization, and detection,
such as the :class:`~sionna.phy.ofdm.LSChannelEstimator`, the
:class:`~sionna.phy.ofdm.RZFPrecoder`, and the :class:`~sionna.phy.ofdm.LMMSEEqualizer` and
:class:`~sionna.phy.ofdm.LinearDetector`.
These are good starting points for the development of more advanced algorithms
and provide robust baselines for benchmarking.

This module also provides blocks for the computation of the :class:`~sionna.phy.ofdm.PostEqualizationSINR` of a possibly precoded channel
:class:`~sionna.phy.ofdm.PrecodedChannel`. These features are useful for
:doc:`physical layer abstraction <../../sys/api/abstraction>`, e.g., using :class:`~sionna.sys.EffectiveSINR`


Resource Grid
-------------

The following code snippet shows how to setup and visualize an instance of
:class:`~sionna.phy.ofdm.ResourceGrid`:

.. code-block:: Python

   rg = ResourceGrid(num_ofdm_symbols = 14,
                     fft_size = 64,
                     subcarrier_spacing = 30e3,
                     num_tx = 1,
                     num_streams_per_tx = 1,
                     num_guard_carriers = [5, 6],
                     dc_null = True,
                     pilot_pattern = "kronecker",
                     pilot_ofdm_symbol_indices = [2, 11])
   rg.show();

.. image:: ../figures/resource_grid.png

This code creates a resource grid consisting of 14 OFDM symbols with 64
subcarriers. The first five and last six subcarriers as well as the DC
subcarriers are nulled. The second and eleventh OFDM symbol are reserved
for pilot transmissions.

Subcarriers are numbered from :math:`0` to :math:`N-1`, where :math:`N`
is the FTT size. The index :math:`0` corresponds to the lowest frequency,
which is :math:`-\frac{N}{2}\Delta_f` (for :math:`N` even) or
:math:`-\frac{N-1}{2}\Delta_f` (for :math:`N` odd), where :math:`\Delta_f`
is the subcarrier spacing which is irrelevant for the resource grid.
The index :math:`N-1` corresponds to the highest frequency,
which is :math:`(\frac{N}{2}-1)\Delta_f` (for :math:`N` even) or
:math:`\frac{N-1}{2}\Delta_f` (for :math:`N` odd).

.. autoclass:: sionna.phy.ofdm.ResourceGrid
   :members:

.. autoclass:: sionna.phy.ofdm.ResourceGridMapper
   :exclude-members: call, build
   :members:

.. autoclass:: sionna.phy.ofdm.ResourceGridDemapper
   :exclude-members: call, build
   :members:

.. autoclass:: sionna.phy.ofdm.RemoveNulledSubcarriers
   :exclude-members: call, build
   :members:


Modulation & Demodulation
-------------------------

.. autoclass:: sionna.phy.ofdm.OFDMModulator
   :exclude-members: call, build
   :members:

.. autoclass:: sionna.phy.ofdm.OFDMDemodulator
   :exclude-members: call, build
   :members:

Pilot Pattern
-------------

A :class:`~sionna.phy.ofdm.PilotPattern` defines how transmitters send pilot
sequences for each of their antennas or streams over an OFDM resource grid.
It consists of two components,
a ``mask`` and ``pilots``. The ``mask`` indicates which resource elements are
reserved for pilot transmissions by each transmitter and its respective
streams. In some cases, the number of streams is equal to the number of
transmit antennas, but this does not need to be the case, e.g., for precoded
transmissions. The ``pilots`` contains the pilot symbols that are transmitted
at the positions indicated by the ``mask``. Separating a pilot pattern into
``mask`` and ``pilots`` enables the implementation of a wide range of pilot
configurations, including trainable pilot sequences.

The following code snippet shows how to define a simple custom
:class:`~sionna.phy.ofdm.PilotPattern` for single transmitter, sending two streams
Note that ``num_effective_subcarriers`` is the number of subcarriers that
can be used for data or pilot transmissions. Due to guard
carriers or a nulled DC carrier, this number can be smaller than the
``fft_size`` of the :class:`~sionna.phy.ofdm.ResourceGrid`.

.. code-block:: Python

   num_tx = 1
   num_streams_per_tx = 2
   num_ofdm_symbols = 14
   num_effective_subcarriers = 12

   # Create a pilot mask
   mask = np.zeros([num_tx,
                    num_streams_per_tx,
                    num_ofdm_symbols,
                    num_effective_subcarriers])
   mask[0, :, [2,11], :] = 1
   num_pilot_symbols = int(np.sum(mask[0,0]))

   # Define pilot sequences
   pilots = np.zeros([num_tx,
                      num_streams_per_tx,
                      num_pilot_symbols], np.complex64)
   pilots[0, 0, 0:num_pilot_symbols:2] = (1+1j)/np.sqrt(2)
   pilots[0, 1, 1:num_pilot_symbols:2] = (1+1j)/np.sqrt(2)

   # Create a PilotPattern instance
   pp = PilotPattern(mask, pilots)

   # Visualize non-zero elements of the pilot sequence
   pp.show(show_pilot_ind=True);

.. image:: ../figures/pilot_pattern.png
.. image:: ../figures/pilot_pattern_2.png

As shown in the figures above, the pilots are mapped onto the mask from
the smallest effective subcarrier and OFDM symbol index to the highest
effective subcarrier and OFDM symbol index. Here, boths stream have 24
pilot symbols, out of which only 12 are nonzero. It is important to keep
this order of mapping in mind when designing more complex pilot sequences.

.. autoclass:: sionna.phy.ofdm.PilotPattern
   :members:

.. autoclass:: sionna.phy.ofdm.EmptyPilotPattern
   :members:

.. autoclass:: sionna.phy.ofdm.KroneckerPilotPattern
   :members:


Channel Estimation
------------------

.. autoclass:: sionna.phy.ofdm.BaseChannelEstimator
   :exclude-members: call, build
   :members:

.. autoclass:: sionna.phy.ofdm.BaseChannelInterpolator
   :exclude-members: call, build
   :members:

.. autoclass:: sionna.phy.ofdm.LSChannelEstimator
   :exclude-members: call, build, estimate_at_pilot_locations
   :members:

.. autoclass:: sionna.phy.ofdm.LinearInterpolator
   :members:

.. autoclass:: sionna.phy.ofdm.LMMSEInterpolator
   :members:

.. autoclass:: sionna.phy.ofdm.NearestNeighborInterpolator
   :members:

.. autofunction:: sionna.phy.ofdm.tdl_time_cov_mat

.. autofunction:: sionna.phy.ofdm.tdl_freq_cov_mat


Precoding
---------

.. autoclass:: sionna.phy.ofdm.RZFPrecoder
   :exclude-members: call, build
   :members:

.. autoclass:: sionna.phy.ofdm.PrecodedChannel
   :exclude-members: call, build
   :members:

.. autoclass:: sionna.phy.ofdm.CBFPrecodedChannel
   :exclude-members: call, build
   :members:

.. autoclass:: sionna.phy.ofdm.EyePrecodedChannel
   :exclude-members: call, build
   :members:
   
.. autoclass:: sionna.phy.ofdm.RZFPrecodedChannel
   :exclude-members: call, build
   :members:


Equalization
------------

.. autoclass:: sionna.phy.ofdm.OFDMEqualizer
   :exclude-members: call, build
   :members:

.. autoclass:: sionna.phy.ofdm.LMMSEEqualizer
   :exclude-members: call, build
   :members:

.. autoclass:: sionna.phy.ofdm.MFEqualizer
   :exclude-members: call, build
   :members:

.. autoclass:: sionna.phy.ofdm.ZFEqualizer
   :exclude-members: call, build
   :members:

.. autoclass:: sionna.phy.ofdm.PostEqualizationSINR
   :exclude-members: call, build
   :members:

.. autoclass:: sionna.phy.ofdm.LMMSEPostEqualizationSINR
   :exclude-members: call, build
   :members:

Detection
---------

.. autoclass:: sionna.phy.ofdm.OFDMDetector
   :exclude-members: call, build
   :members:

.. autoclass:: sionna.phy.ofdm.OFDMDetectorWithPrior
   :exclude-members: call, build
   :members:

.. autoclass:: sionna.phy.ofdm.EPDetector
   :exclude-members: call, build
   :members:

.. autoclass:: sionna.phy.ofdm.KBestDetector
   :exclude-members: call, build
   :members:

.. autoclass:: sionna.phy.ofdm.LinearDetector
   :exclude-members: call, build
   :members:

.. autoclass:: sionna.phy.ofdm.MaximumLikelihoodDetector
   :exclude-members: call, build
   :members:

.. autoclass:: sionna.phy.ofdm.MaximumLikelihoodDetectorWithPrior
   :exclude-members: call, build
   :members:

.. autoclass:: sionna.phy.ofdm.MMSEPICDetector
   :exclude-members: call, build
   :members:
