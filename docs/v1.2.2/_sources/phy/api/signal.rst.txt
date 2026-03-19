Signal
========

This module contains classes and functions for :ref:`filtering <filter>` (pulse shaping), :ref:`windowing <window>`, and :ref:`up- <upsampling>` and :ref:`downsampling <downsampling>`.
The following figure shows the different components that can be implemented using this module.

.. figure:: ../figures/signal_module.png
   :width: 75%
   :align: center

This module also contains :ref:`utility functions <utility>` for computing the (inverse) discrete Fourier transform (:ref:`FFT <fft>`/:ref:`IFFT <ifft>`), and for empirically computing the :ref:`power spectral density (PSD) <empirical_psd>` and :ref:`adjacent channel leakage ratio (ACLR) <empirical_aclr>` of a signal.

The following code snippet shows how to filter a sequence of QAM baseband symbols using a root-raised-cosine filter with a Hann window:

.. code-block:: Python

   # Create batch of QAM-16 sequences 
   batch_size = 128
   num_symbols = 1000
   num_bits_per_symbol = 4
   x = QAMSource(num_bits_per_symbol)([batch_size, num_symbols])

   # Create a root-raised-cosine filter with Hann windowing
   beta = 0.22 # Roll-off factor
   span_in_symbols = 32 # Filter span in symbols
   samples_per_symbol = 4 # Number of samples per symbol, i.e., the oversampling factor
   rrcf_hann = RootRaisedCosineFilter(span_in_symbols, samples_per_symbol, beta, window="hann")

   # Create instance of the Upsampling layer
   us = Upsampling(samples_per_symbol)

   # Upsample the baseband x
   x_us = us(x)

   # Filter the upsampled sequence
   x_rrcf = rrcf_hann(x_us)

On the receiver side, one would recover the baseband symbols as follows:

.. code-block:: Python

   # Instantiate a downsampling layer
   ds = Downsampling(samples_per_symbol, rrcf_hann.length-1, num_symbols)

   # Apply the matched filter
   x_mf = rrcf_hann(x_rrcf)
   
   # Recover the transmitted symbol sequence
   x_hat = ds(x_mf)


.. _filter:

Filters
-------

.. autoclass:: sionna.phy.signal.SincFilter
   :members: length, window, normalize, coefficients, sampling_times, show, aclr
   :exclude-members: call, build

.. autoclass:: sionna.phy.signal.RaisedCosineFilter
   :members: length, window, normalize, coefficients, sampling_times, show, aclr, beta
   :exclude-members: call, build

.. autoclass:: sionna.phy.signal.RootRaisedCosineFilter
   :members: length, window, normalize, coefficients, sampling_times, show, aclr, beta
   :exclude-members: call, build

.. autoclass:: sionna.phy.signal.CustomFilter
   :members: length, window, normalize, coefficients, sampling_times, show, aclr
   :exclude-members: call, build

.. autoclass:: sionna.phy.signal.Filter
   :members:
   :exclude-members: call, build

.. _window:

Window functions
----------------

.. autoclass:: sionna.phy.signal.HannWindow
   :members: coefficients, length, normalize, show
   :exclude-members: call, build

.. autoclass:: sionna.phy.signal.HammingWindow
   :members: coefficients, length, normalize, show
   :exclude-members: call, build

.. autoclass:: sionna.phy.signal.BlackmanWindow
   :members: coefficients, length, normalize, show
   :exclude-members: call, build

.. autoclass:: sionna.phy.signal.CustomWindow
   :members: coefficients, length, normalize, show
   :exclude-members: call, build

.. autoclass:: sionna.phy.signal.Window
   :members:
   :exclude-members: call, build

.. _utility:

Utility Functions
-----------------

.. autofunction:: sionna.phy.signal.convolve

.. _fft:

.. autofunction:: sionna.phy.signal.fft

.. _ifft:

.. autofunction:: sionna.phy.signal.ifft

.. _upsampling:

.. autoclass:: sionna.phy.signal.Upsampling
   :members:
   :exclude-members: call, build

.. _downsampling:

.. autoclass:: sionna.phy.signal.Downsampling
   :members:
   :exclude-members: call, build

.. _empirical_psd:

.. autofunction:: sionna.phy.signal.empirical_psd

.. _empirical_aclr:

.. autofunction:: sionna.phy.signal.empirical_aclr
