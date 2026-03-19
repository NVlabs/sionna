5G NR
=====

This module provides blocks and functions to support simulations of
5G NR compliant features, in particular, the physical uplink shared channel (PUSCH). It provides implementations of a subset of the physical layer functionalities as described in the 3GPP specifications :cite:p:`3GPPTS38211`, :cite:p:`3GPPTS38212`, and :cite:p:`3GPPTS38214`.

The best way to discover this module's components is by having a look at the `5G NR PUSCH Tutorial <../tutorials/notebooks/5G_NR_PUSCH.ipynb>`_.

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

.. toctree::
   :maxdepth: 2
   :hidden:

   carrier
   layer_mapping
   pusch
   transport_block
   utils

