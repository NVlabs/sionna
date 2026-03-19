.. _ofdm-waveform:

Frequency (OFDM) Domain 
=======================

To implement the channel response assuming an OFDM waveform, it is assumed that
the power delay profiles are invariant over the duration of an OFDM symbol.
Moreover, it is assumed that the duration of the cyclic prefix (CP) equals at
least the maximum delay spread. These assumptions are common in the literature, as they
enable modeling of the channel transfer function in the frequency domain as a
single-tap channel.

For every link :math:`(u, k, v, l)` and resource element :math:`(s,n)`,
the frequency channel response is obtained by computing the Fourier transform of
the channel response at the subcarrier frequencies, i.e.,

.. math::
   \begin{aligned}
   \widehat{h}_{u, k, v, l, s, n}
      &= \int_{-\infty}^{+\infty} h_{u, k, v, l}(s,\tau) e^{-j2\pi n \Delta_f \tau} d\tau\\
      &= \sum_{m=0}^{M-1} a_{u, k, v, l, m}(s)
      e^{-j2\pi n \Delta_f \tau_{u, k, v, l, m}}
   \end{aligned}

where :math:`s` is used as time step to indicate that the channel response can
change from one OFDM symbol to the next in the event of mobility, even if it is
assumed static over the duration of an OFDM symbol.

For every receive antenna :math:`l` of every receiver :math:`v`, the
received signal :math:`y_{v, l, s, n}`` for resource element
:math:`(s, n)` is computed by

.. math::
   y_{v, l, s, n} = \sum_{u=0}^{N_{T}-1}\sum_{k=0}^{N_{TA}-1}
      \widehat{h}_{u, k, v, l, s, n} x_{u, k, s, n}
      + w_{v, l, s, n}

where :math:`x_{u, k, s, n}` is the baseband symbol transmitted by transmitter
:math:`u`` on antenna :math:`k` and resource element :math:`(s, n)`, and
:math:`w_{v, l, s, n} \sim \mathcal{CN}\left(0,N_0\right)` the additive white
Gaussian noise.

.. note::
   This model does not account for intersymbol interference (ISI) nor
   intercarrier interference (ICI). To model the ICI due to channel aging over
   the duration of an OFDM symbol or the ISI due to a delay spread exceeding the
   CP duration, one would need to simulate the channel in the time domain.
   This can be achieved by using the :class:`~sionna.phy.ofdm.OFDMModulator` and
   :class:`~sionna.phy.ofdm.OFDMDemodulator` layers, and the
   :ref:`time domain channel model <time-domain>`.
   By doing so, one performs inverse discrete Fourier transform (IDFT) on
   the transmitter side and discrete Fourier transform (DFT) on the receiver side
   on top of a single-carrier sinc-shaped waveform.
   This is equivalent to
   :ref:`simulating the channel in the frequency domain <ofdm-waveform>` if no
   ISI nor ICI is assumed, but allows the simulation of these effects in the
   event of a non-stationary channel or long delay spreads.
   Note that simulating the channel in the time domain is typically significantly
   more computationally demanding that simulating the channel in the frequency
   domain.



.. currentmodule:: sionna.phy.channel

.. autosummary::
   :toctree: .

    OFDMChannel
    GenerateOFDMChannel
    ApplyOFDMChannel
    cir_to_ofdm_channel