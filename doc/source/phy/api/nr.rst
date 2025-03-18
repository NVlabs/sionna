5G NR
=====

This module provides blocks and functions to support simulations of
5G NR compliant features, in particular, the physical uplink shared channel (PUSCH). It provides implementations of a subset of the physical layer functionalities as described in the 3GPP specifications [3GPP38211]_, [3GPP38212]_, and [3GPP38214]_.

The best way to discover this module's components is by having a look at the `5G NR PUSCH Tutorial <../tutorials/5G_NR_PUSCH.html>`_.

The following code snippet shows how you can make standard-compliant
simulations of the 5G NR PUSCH with a few lines of code:

.. code-block:: Python

   # Create a PUSCH configuration with default settings
   pusch_config = PUSCHConfig()

   # Instantiate a PUSCHTransmitter from the PUSCHConfig
   pusch_transmitter = PUSCHTransmitter(pusch_config)

   # Create a PUSCHReceiver using the PUSCHTransmitter
   pusch_receiver = PUSCHReceiver(pusch_transmitter)

   # AWGN channel
   channel = AWGN()

   # Simulate transmissions over the AWGN channel
   batch_size = 16
   no = 0.1 # Noise variance

   x, b = pusch_transmitter(batch_size) # Generate transmit signal and info bits

   y = channel(x, no) # Simulate channel output

   b_hat = pusch_receiver(y, no) # Recover the info bits

   # Compute BER
   print("BER:", compute_ber(b, b_hat).numpy())

The :class:`~sionna.phy.nr.PUSCHTransmitter` and :class:`~sionna.phy.nr.PUSCHReceiver` provide high-level abstractions of all required processing blocks. You can easily modify them according to your needs.

Carrier
-------

.. autoclass:: sionna.phy.nr.CarrierConfig
   :exclude-members: check_config
   :members:

Layer Mapping
-------------

.. autoclass:: sionna.phy.nr.LayerMapper
   :exclude-members: build, call
   :members:

.. autoclass:: sionna.phy.nr.LayerDemapper
   :exclude-members: build, call
   :members:

PUSCH
-----

.. autoclass:: sionna.phy.nr.PUSCHConfig
   :exclude-members: check_config, l, l_ref, l_0, l_d, l_prime, n
   :members:

.. autoclass:: sionna.phy.nr.PUSCHDMRSConfig
   :exclude-members: check_config
   :members:

.. autoclass:: sionna.phy.nr.PUSCHLSChannelEstimator
   :exclude-members: estimate_at_pilot_locations
   :members:

.. autoclass:: sionna.phy.nr.PUSCHPilotPattern
   :inherited-members:

.. autoclass:: sionna.phy.nr.PUSCHPrecoder
   :exclude-members: build, call
   :members:

.. autoclass:: sionna.phy.nr.PUSCHReceiver
   :exclude-members: build, call
   :members:

.. autoclass:: sionna.phy.nr.PUSCHTransmitter
   :exclude-members: build, call
   :members:

Transport Block
---------------

.. autoclass:: sionna.phy.nr.TBConfig
   :exclude-members: build, call
   :members:

.. autoclass:: sionna.phy.nr.TBEncoder
   :exclude-members: build, call
   :members:

.. autoclass:: sionna.phy.nr.TBDecoder
   :exclude-members: build, call
   :members:

Utils
-----

.. autofunction:: sionna.phy.nr.utils.calculate_tb_size

.. autofunction:: sionna.phy.nr.utils.generate_prng_seq

.. autofunction:: sionna.phy.nr.utils.decode_mcs_index

.. autofunction:: sionna.phy.nr.utils.calculate_num_coded_bits

.. autoclass:: sionna.phy.nr.utils.TransportBlockNR
   :members:
   :exclude-members: build, call

.. autoclass:: sionna.phy.nr.utils.CodedAWGNChannelNR
  :members:
  :exclude-members: build, call

.. autoclass:: sionna.phy.nr.utils.MCSDecoderNR
  :members:
  :exclude-members: build, call


References:
   .. [3GPP38211] 3GPP TS 38.211. "NR; Physical channels and modulation."

   .. [3GPP38212] 3GPP TS 38.212. "NR; Multiplexing and channel coding"

   .. [3GPP38214] 3GPP TS 38.214. "NR; Physical layer procedures for data."



