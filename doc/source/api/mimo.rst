Multiple-Input Multiple-Output (MIMO)
#####################################
This module provides layers and functions to support simulation of multicell
MIMO transmissions.


Stream Management
*****************

Stream management determines which transmitter is sending which stream to
which receiver. Transmitters and receivers can be user terminals or base
stations, depending on whether uplink or downlink transmissions are considered.
The :class:`~sionna.mimo.StreamManagement` class has various properties that
are needed to recover desired or interfering channel coefficients for precoding
and equalization. In order to understand how the various properties of
:class:`~sionna.mimo.StreamManagement` can be used, we recommend to have a look
at the source code of the :class:`~sionna.ofdm.LMMSEEqualizer` or
:class:`~sionna.ofdm.ZFPrecoder`.

The following code snippet shows how to configure
:class:`~sionna.mimo.StreamManagement` for a simple uplink scenario, where
four transmitters send each one stream to a receiver. Note that
:class:`~sionna.mimo.StreamManagement` is independent of the actual number of
antennas at the transmitters and receivers.

.. code-block:: Python

        num_tx = 4
        num_rx = 1
        num_streams_per_tx = 1

        # Indicate which transmitter is associated with which receiver
        # rx_tx_association[i,j] = 1 means that transmitter j sends one
        # or mutiple streams to receiver i.
        rx_tx_association = np.zeros([num_rx, num_tx])
        rx_tx_association[0,0] = 1
        rx_tx_association[0,1] = 1
        rx_tx_association[0,2] = 1
        rx_tx_association[0,3] = 1

        sm = StreamManagement(rx_tx_association, num_streams_per_tx)

.. autoclass:: sionna.mimo.StreamManagement
   :members:
   :undoc-members:


Precoding
*********

zero_forcing_precoder
---------------------
.. autofunction:: sionna.mimo.zero_forcing_precoder


Equalization
************

lmmse_equalizer
---------------
.. autofunction:: sionna.mimo.lmmse_equalizer


References:
        .. [BHS2017] Emil Björnson, Jakob Hoydis and Luca Sanguinetti (2017),
                `“Massive MIMO Networks: Spectral, Energy, and Hardware Efficiency”
                <https://massivemimobook.com>`_,
                Foundations and Trends in Signal Processing:
                Vol. 11, No. 3-4, pp 154–655.
