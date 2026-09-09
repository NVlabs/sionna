.. _time-domain:

Time Domain
===========

The model of the channel in the time domain assumes pulse shaping and receive
filtering are performed using a conventional sinc filter (see, e.g., :cite:p:`Tse`).
Using sinc for transmit and receive filtering, the discrete-time domain received
signal at time step :math:`b` is

.. math::
   y_{v, l, b} = \sum_{u=0}^{N_{T}-1}\sum_{k=0}^{N_{TA}-1}
      \sum_{\ell = L_{\text{min}}}^{L_{\text{max}}}
      \bar{h}_{u, k, v, l, b, \ell} x_{u, k, b-\ell}
      + w_{v, l, b}

where :math:`x_{u, k, b}` is the baseband symbol transmitted by transmitter
:math:`u` on antenna :math:`k` and at time step :math:`b`,
:math:`w_{v, l, b} \sim \mathcal{CN}\left(0,N_0\right)` the additive white
Gaussian noise, and :math:`\bar{h}_{u, k, v, l, b, \ell}` the channel filter tap
at time step :math:`b` and for time-lag :math:`\ell`, which is given by

.. math::
   \bar{h}_{u, k, v, l, b, \ell}
   = \sum_{m=0}^{M-1} a_{u, k, v, l, m}\left(\frac{b}{W}\right)
      \text{sinc}\left( \ell - W\tau_{u, v, m} \right).

.. note::
   The two parameters :math:`L_{\text{min}}` and :math:`L_{\text{max}}` control the smallest
   and largest time-lag for the discrete-time channel model, respectively.
   They are set when instantiating :class:`~sionna.phy.channel.TimeChannel`,
   :class:`~sionna.phy.channel.GenerateTimeChannel`, and when calling the utility
   function :func:`~sionna.phy.channel.cir_to_time_channel`.
   Because the sinc filter is neither time-limited nor causal, the discrete-time
   channel model is not causal. Therefore, ideally, one would set
   :math:`L_{\text{min}} = -\infty` and :math:`L_{\text{max}} = +\infty`.
   In practice, however, these two parameters need to be set to reasonable
   finite values. Values for these two parameters can be computed using the
   :func:`~sionna.phy.channel.time_lag_discrete_time_channel` utility function from
   a given bandwidth and maximum delay spread.
   This function returns :math:`-6` for :math:`L_{\text{min}}`. :math:`L_{\text{max}}` is computed
   from the specified bandwidth and maximum delay spread, which default value is
   :math:`3 \mu s`. These values for :math:`L_{\text{min}}` and the maximum delay spread
   were found to be valid for all the models available in Sionna when an RMS delay
   spread of 100ns is assumed.


.. currentmodule:: sionna.phy.channel

.. autosummary::
   :toctree: .

   TimeChannel
   GenerateTimeChannel
   ApplyTimeChannel
   cir_to_time_channel
   time_to_ofdm_channel