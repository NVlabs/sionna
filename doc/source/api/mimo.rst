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

grid_of_beams_dft_ula
---------------------
.. autofunction:: sionna.mimo.grid_of_beams_dft_ula

grid_of_beams_dft
---------------------
.. autofunction:: sionna.mimo.grid_of_beams_dft

flatten_precoding_mat
---------------------
.. autofunction:: sionna.mimo.flatten_precoding_mat

normalize_precoding_power
-------------------------
.. autofunction:: sionna.mimo.normalize_precoding_power

Equalization
************

lmmse_equalizer
---------------
.. autofunction:: sionna.mimo.lmmse_equalizer

mf_equalizer
------------
.. autofunction:: sionna.mimo.mf_equalizer

zf_equalizer
---------------
.. autofunction:: sionna.mimo.zf_equalizer


Detection
**********

EPDetector
----------
.. autoclass:: sionna.mimo.EPDetector
   :exclude-members: call, build, compute_sigma_mu, compute_v_x, compute_v_x_obs, update_lam_gam
   :members:

KBestDetector
-------------
.. autoclass:: sionna.mimo.KBestDetector
   :exclude-members: call, build
   :members:

LinearDetector
--------------
.. autoclass:: sionna.mimo.LinearDetector
   :exclude-members: call, build
   :members:

MaximumLikelihoodDetector
-------------------------
.. autoclass:: sionna.mimo.MaximumLikelihoodDetector
   :exclude-members: call, build
   :members:

MaximumLikelihoodDetectorWithPrior
----------------------------------
.. autoclass:: sionna.mimo.MaximumLikelihoodDetectorWithPrior
   :exclude-members: call, build
   :members:

MMSE-PIC
----------
.. autoclass:: sionna.mimo.MMSEPICDetector
   :exclude-members: call, build
   :members:

Utility Functions
*****************


List2LLR
--------
.. autoclass:: sionna.mimo.List2LLR
   :exclude-members: __call__
   :members:

List2LLRSimple
--------------
.. autoclass:: sionna.mimo.List2LLRSimple
   :exclude-members: call, build
   :members:

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
   .. [ProperRV] `Proper complex random variables <https://en.wikipedia.org/wiki/Complex_random_variable#Proper_complex_random_variables>`_,
      Wikipedia, accessed 11 September, 2022.

   .. [CovProperRV] `Covariance matrices of real and imaginary parts <https://en.wikipedia.org/wiki/Complex_random_vector#Covariance_matrices_of_real_and_imaginary_parts>`_,
      Wikipedia, accessed 11 September, 2022.

   .. [YH2015] S\. Yang and L\. Hanzo, `"Fifty Years of MIMO Detection: The Road to Large-Scale MIMOs"
      <https://ieeexplore.ieee.org/abstract/document/7244171>`_,
      IEEE Communications Surveys & Tutorials, vol. 17, no. 4, pp. 1941-1988, 2015.

   .. [FT2015] W\. Fu and J\. S\. Thompson, `"Performance analysis of K-best detection with adaptive modulation"
      <https://ieeexplore.ieee.org/abstract/document/7454351>`_, IEEE Int. Symp. Wirel. Commun. Sys. (ISWCS), 2015.

   .. [EP2014] J\. Céspedes, P\. M\. Olmos, M\. Sánchez-Fernández, and F\. Perez-Cruz,
      `"Expectation Propagation Detection for High-Order High-Dimensional MIMO Systems" <https://ieeexplore.ieee.org/abstract/document/6841617>`_,
      IEEE Trans. Commun., vol. 62, no. 8, pp. 2840-2849, Aug. 2014.

   .. [CST2011] C\. Studer, S\. Fateh, and D\. Seethaler,
      `"ASIC Implementation of Soft-Input Soft-Output MIMO Detection Using MMSE Parallel Interference Cancellation" <https://ieeexplore.ieee.org/abstract/document/5779722>`_,
      IEEE Journal of Solid-State Circuits, vol. 46, no. 7, pp. 1754–1765, July 2011.
