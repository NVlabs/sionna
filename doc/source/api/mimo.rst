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

zf_equalizer
---------------
.. autofunction:: sionna.mimo.zf_equalizer

mf_equalizer
---------------
.. autofunction:: sionna.mimo.mf_equalizer

Detection
**********

MaximumLikelihoodDetector
---------------------------------
.. autoclass:: sionna.mimo.MaximumLikelihoodDetector
   :exclude-members: call, build
   :members:

MaximumLikelihoodDetectorWithPrior
------------------------------------
.. autoclass:: sionna.mimo.MaximumLikelihoodDetectorWithPrior
   :exclude-members: call, build
   :members:

Utility Functions
*****************

complex2real_vector
-------------------
.. autofunction:: sionna.mimo.complex2real_vector

real2complex_vector
-------------------
.. autofunction:: sionna.mimo.real2complex_vector

complex2real_matrix
-------------------
.. autofunction:: sionna.mimo.complex2real_matrix

real2complex_matrix
-------------------
.. autofunction:: sionna.mimo.real2complex_matrix

complex2real_covariance
-----------------------
.. autofunction:: sionna.mimo.complex2real_covariance

real2complex_covariance
-----------------------
.. autofunction:: sionna.mimo.real2complex_covariance

complex2real_channel
--------------------
.. autofunction:: sionna.mimo.complex2real_channel

real2complex_channel
--------------------
.. autofunction:: sionna.mimo.real2complex_channel

whiten_channel
--------------
.. autofunction:: sionna.mimo.whiten_channel

References:
   .. [BHS2017] Emil Björnson, Jakob Hoydis and Luca Sanguinetti (2017),
      `“Massive MIMO Networks: Spectral, Energy, and Hardware Efficiency”
      <https://massivemimobook.com>`_,
      Foundations and Trends in Signal Processing:
      Vol. 11, No. 3-4, pp 154–655.

   .. [ProperRV] `Proper complex random variables <https://en.wikipedia.org/wiki/Complex_random_variable#Proper_complex_random_variables>`_,
      Wikipedia, accessed 11 September, 2022.
   
   .. [CovProperRV] `Covariance matrices of real and imaginary parts <https://en.wikipedia.org/wiki/Complex_random_vector#Covariance_matrices_of_real_and_imaginary_parts>`_,
      Wikipedia, accessed 11 September, 2022.

   .. [YH2015] S. Yang and L. Hanzo, `"Fifty Years of MIMO Detection: The Road to Large-Scale MIMOs"
      <https://ieeexplore.ieee.org/abstract/document/7244171>`_,
      IEEE Communications Surveys & Tutorials, vol. 17, no. 4, pp. 1941-1988, 2015.
