Channel Models
==============

This module provides channel models for simulating signal transmission through
various communication media. It supports three main categories of channels:

:doc:`wireless/index`
   Models for wireless radio channels, including additive white Gaussian noise
   (AWGN), flat fading with spatial correlation, Rayleigh block fading, and
   standardized 3GPP models (TDL, CDL, UMi, UMa, RMa) from TR 38.901. These
   models generate channel impulse responses (CIRs) that can be applied in the
   time domain or frequency domain (OFDM). The module also supports loading
   externally generated CIRs from datasets.

:doc:`optical/optical`
   Models for fiber optical communication channels. The split-step Fourier
   method (SSFM) simulates signal propagation through single-mode fibers,
   accounting for attenuation, chromatic dispersion, and Kerr nonlinearity.
   An Erbium-doped fiber amplifier (EDFA) model is provided for simulating
   lumped amplification with amplified spontaneous emission (ASE) noise.

:doc:`discrete/discrete`
   Models for channels with discrete input/output alphabets, useful for
   information-theoretic analysis and coded system simulations. Includes the
   binary symmetric channel (BSC), binary erasure channel (BEC), binary
   Z-channel, and a general binary memoryless channel. These models support
   both hard outputs and soft log-likelihood ratio (LLR) outputs.

.. toctree::
   :hidden:
   :maxdepth: 5

   wireless/index.rst
   optical/optical.rst
   discrete/discrete.rst