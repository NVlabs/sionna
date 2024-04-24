5G NR
#####
This module provides layers and functions to support simulations of
5G NR compliant features, in particular, the physical uplink shared channel (PUSCH). It provides implementations of a subset of the physical layer functionalities as described in the 3GPP specifications [3GPP38211]_, [3GPP38212]_, and [3GPP38214]_.

The best way to discover this module's components is by having a look at the `5G NR PUSCH Tutorial <../examples/5G_NR_PUSCH.html>`_.

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

   y = channel([x, no]) # Simulate channel output

   b_hat = pusch_receiver([y, no]) # Recover the info bits

   # Compute BER
   print("BER:", compute_ber(b, b_hat).numpy())

The :class:`~sionna.nr.PUSCHTransmitter` and :class:`~sionna.nr.PUSCHReceiver` provide high-level abstractions of all required processing blocks. You can easily modify them according to your needs.

Carrier
*******

CarrierConfig
-------------
.. autoclass:: sionna.nr.CarrierConfig
   :exclude-members: check_config
   :members:

Layer Mapping
*************

LayerMapper
-----------
.. autoclass:: sionna.nr.LayerMapper
   :exclude-members: build, call
   :members:

LayerDemapper
-------------
.. autoclass:: sionna.nr.LayerDemapper
   :exclude-members: build, call
   :members:


PUSCH
*****

PUSCHConfig
-----------
.. autoclass:: sionna.nr.PUSCHConfig
   :exclude-members: check_config, l, l_ref, l_0, l_d, l_prime, n
   :members:

PUSCHDMRSConfig
---------------
.. autoclass:: sionna.nr.PUSCHDMRSConfig
   :exclude-members: check_config
   :members:

PUSCHLSChannelEstimator
-----------------------
.. autoclass:: sionna.nr.PUSCHLSChannelEstimator
   :exclude-members: estimate_at_pilot_locations
   :members:

PUSCHPilotPattern
-----------------
.. autoclass:: sionna.nr.PUSCHPilotPattern
   :inherited-members:

PUSCHPrecoder
-------------
.. autoclass:: sionna.nr.PUSCHPrecoder
   :exclude-members: build, call
   :members:

PUSCHReceiver
----------------
.. autoclass:: sionna.nr.PUSCHReceiver
   :exclude-members: build, call
   :members:

PUSCHTransmitter
----------------
.. autoclass:: sionna.nr.PUSCHTransmitter
   :exclude-members: build, call
   :members:


Transport Block
***************

TBConfig
--------
.. autoclass:: sionna.nr.TBConfig
   :exclude-members:
   :members:

TBEncoder
---------
.. autoclass:: sionna.nr.TBEncoder
   :exclude-members: build, call
   :members:

TBDecoder
---------
.. autoclass:: sionna.nr.TBDecoder
   :exclude-members: build, call
   :members:

Utils
*****

calculate_tb_size
-----------------
.. autofunction:: sionna.nr.utils.calculate_tb_size

generate_prng_seq
-----------------
.. autofunction:: sionna.nr.utils.generate_prng_seq

select_mcs
----------
.. autofunction:: sionna.nr.utils.select_mcs


References:
   .. [3GPP38211] 3GPP TS 38.211. "NR; Physical channels and modulation."

   .. [3GPP38212] 3GPP TS 38.212. "NR; Multiplexing and channel coding"

   .. [3GPP38214] 3GPP TS 38.214. "NR; Physical layer procedures for data."



