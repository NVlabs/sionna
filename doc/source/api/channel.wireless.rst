========
Wireless 
========

This module provides layers and functions that implement wireless channel models.
Models currently available include :class:`~sionna.channel.AWGN`, :ref:`flat-fading <flat-fading>` with (optional) :class:`~sionna.channel.SpatialCorrelation`, :class:`~sionna.channel.RayleighBlockFading`, as well as models from the 3rd Generation Partnership Project (3GPP) [TR38901]_: :ref:`TDL <tdl>`, :ref:`CDL <cdl>`, :ref:`UMi <umi>`, :ref:`UMa <uma>`, and :ref:`RMa <rma>`. It is also possible to :ref:`use externally generated CIRs <external-datasets>`.

Apart from :ref:`flat-fading <flat-fading>`, all of these models generate channel impulse responses (CIRs) that can then be used to
implement a channel transfer function in the :ref:`time domain <time-domain>` or
:ref:`assuming an OFDM waveform <ofdm-waveform>`.

This is achieved using the different functions, classes, and Keras layers which
operate as shown in the figures below.

.. figure:: ../figures/channel_arch_time.png
   :align: center

   Channel module architecture for time domain simulations.

.. figure:: ../figures/channel_arch_freq.png
   :align: center

   Channel module architecture for simulations assuming OFDM waveform.

A channel model generate CIRs from which channel responses in the time domain
or in the frequency domain are computed using the
:func:`~sionna.channel.cir_to_time_channel` or
:func:`~sionna.channel.cir_to_ofdm_channel` functions, respectively.
If one does not need access to the raw CIRs, the
:class:`~sionna.channel.GenerateTimeChannel` and
:class:`~sionna.channel.GenerateOFDMChannel` classes can be used to conveniently
sample CIRs and generate channel responses in the desired domain.

Once the channel responses in the time or frequency domain are computed, they
can be applied to the channel input using the
:class:`~sionna.channel.ApplyTimeChannel` or
:class:`~sionna.channel.ApplyOFDMChannel` Keras layers.

The following code snippets show how to setup and run a Rayleigh block fading
model assuming an OFDM waveform, and without accessing the CIRs or
channel responses.
This is the easiest way to setup a channel model.
Setting-up other models is done in a similar way, except for
:class:`~sionna.channel.AWGN` (see the :class:`~sionna.channel.AWGN`
class documentation).

.. code-block:: Python

   rayleigh = RayleighBlockFading(num_rx = 1,
                                  num_rx_ant = 32,
                                  num_tx = 4,
                                  num_tx_ant = 2)

   channel  = OFDMChannel(channel_model = rayleigh,
                          resource_grid = rg)

where ``rg`` is an instance of :class:`~sionna.ofdm.ResourceGrid`.

Running the channel model is done as follows:

.. code-block:: Python

   # x is the channel input
   # no is the noise variance
   y = channel([x, no])

To use the time domain representation of the channel, one can use
:class:`~sionna.channel.TimeChannel` instead of
:class:`~sionna.channel.OFDMChannel`.

If access to the channel responses is needed, one can separate their
generation from their application to the channel input by setting up the channel
model as follows:

.. code-block:: Python

   rayleigh = RayleighBlockFading(num_rx = 1,
                                  num_rx_ant = 32,
                                  num_tx = 4,
                                  num_tx_ant = 2)

   generate_channel = GenerateOFDMChannel(channel_model = rayleigh,
                                          resource_grid = rg)

   apply_channel = ApplyOFDMChannel()

where ``rg`` is an instance of :class:`~sionna.ofdm.ResourceGrid`.
Running the channel model is done as follows:

.. code-block:: Python

   # Generate a batch of channel responses
   h = generate_channel(batch_size)
   # Apply the channel
   # x is the channel input
   # no is the noise variance
   y = apply_channel([x, h, no])

Generating and applying the channel in the time domain can be achieved by using
:class:`~sionna.channel.GenerateTimeChannel` and
:class:`~sionna.channel.ApplyTimeChannel` instead of
:class:`~sionna.channel.GenerateOFDMChannel` and
:class:`~sionna.channel.ApplyOFDMChannel`, respectively.

To access the CIRs, setting up the channel can be done as follows:

.. code-block:: Python

   rayleigh = RayleighBlockFading(num_rx = 1,
                                  num_rx_ant = 32,
                                  num_tx = 4,
                                  num_tx_ant = 2)

   apply_channel = ApplyOFDMChannel()

and running the channel model as follows:

.. code-block:: Python

   cir = rayleigh(batch_size)
   h = cir_to_ofdm_channel(frequencies, *cir)
   y = apply_channel([x, h, no])

where ``frequencies`` are the subcarrier frequencies in the baseband, which can
be computed using the :func:`~sionna.channel.subcarrier_frequencies` utility
function.

Applying the channel in the time domain can be done by using
:func:`~sionna.channel.cir_to_time_channel` and
:class:`~sionna.channel.ApplyTimeChannel` instead of
:func:`~sionna.channel.cir_to_ofdm_channel` and
:class:`~sionna.channel.ApplyOFDMChannel`, respectively.

For the purpose of the present document, the following symbols apply:

+------------------------+--------------------------------------------------------------------------+
| :math:`N_T (u)`        | Number of transmitters (transmitter index)                               |
+------------------------+--------------------------------------------------------------------------+
| :math:`N_R (v)`        | Number of receivers (receiver index)                                     |
+------------------------+--------------------------------------------------------------------------+
| :math:`N_{TA} (k)`     | Number of antennas per transmitter (transmit antenna index)              |
+------------------------+--------------------------------------------------------------------------+
| :math:`N_{RA} (l)`     | Number of antennas per receiver (receive antenna index)                  |
+------------------------+--------------------------------------------------------------------------+
| :math:`N_S (s)`        | Number of OFDM symbols (OFDM symbol index)                               |
+------------------------+--------------------------------------------------------------------------+
| :math:`N_F (n)`        | Number of subcarriers (subcarrier index)                                 |
+------------------------+--------------------------------------------------------------------------+
| :math:`N_B (b)`        | Number of time samples forming the channel input (baseband symbol index) |
+------------------------+--------------------------------------------------------------------------+
| :math:`L_{\text{min}}` | Smallest time-lag for the discrete complex baseband channel              |
+------------------------+--------------------------------------------------------------------------+
| :math:`L_{\text{max}}` | Largest time-lag for the discrete complex baseband channel               |
+------------------------+--------------------------------------------------------------------------+
| :math:`M (m)`          | Number of paths (clusters) forming a power delay profile (path index)    |
+------------------------+--------------------------------------------------------------------------+
| :math:`\tau_m(t)`      | :math:`m^{th}` path (cluster) delay at time step :math:`t`               |
+------------------------+--------------------------------------------------------------------------+
| :math:`a_m(t)`         | :math:`m^{th}` path (cluster) complex coefficient at time step :math:`t` |
+------------------------+--------------------------------------------------------------------------+
| :math:`\Delta_f`       | Subcarrier spacing                                                       |
+------------------------+--------------------------------------------------------------------------+
| :math:`W`              | Bandwidth                                                                |
+------------------------+--------------------------------------------------------------------------+
| :math:`N_0`            | Noise variance                                                           |
+------------------------+--------------------------------------------------------------------------+


All transmitters are equipped with :math:`N_{TA}` antennas and all receivers
with :math:`N_{RA}` antennas.

A channel model, such as :class:`~sionna.channel.RayleighBlockFading` or
:class:`~sionna.channel.tr38901.UMi`, is used to generate for each link between
antenna :math:`k` of transmitter :math:`u` and antenna :math:`l` of receiver
:math:`v` a power delay profile
:math:`(a_{u, k, v, l, m}(t), \tau_{u, v, m}), 0 \leq m \leq M-1`.
The delays are assumed not to depend on time :math:`t`, and transmit and receive
antennas :math:`k` and :math:`l`.
Such a power delay profile corresponds to the channel impulse response

.. math::
   h_{u, k, v, l}(t,\tau) =
   \sum_{m=0}^{M-1} a_{u, k, v, l,m}(t) \delta(\tau - \tau_{u, v, m})

where :math:`\delta(\cdot)` is the Dirac delta measure.
For example, in the case of Rayleigh block fading, the power delay profiles are
time-invariant and such that for every link :math:`(u, k, v, l)`

.. math::
   \begin{align}
      M                     &= 1\\
      \tau_{u, v, 0}  &= 0\\
      a_{u, k, v, l, 0}     &\sim \mathcal{CN}(0,1).
   \end{align}

3GPP channel models use the procedure depicted in [TR38901]_ to generate power
delay profiles. With these models, the power delay profiles are time-*variant*
in the event of mobility.

AWGN
=====

.. autoclass:: sionna.channel.AWGN
   :members:
   :exclude-members: call, build

.. _flat-fading:

Flat-fading channel
===================

FlatFadingChannel
------------------

.. autoclass:: sionna.channel.FlatFadingChannel
   :members:
   :exclude-members: call, build

GenerateFlatFadingChannel
--------------------------

.. autoclass:: sionna.channel.GenerateFlatFadingChannel
   :members:

ApplyFlatFadingChannel
-----------------------

.. autoclass:: sionna.channel.ApplyFlatFadingChannel
   :members:
   :exclude-members: call, build

SpatialCorrelation
--------------------

.. autoclass:: sionna.channel.SpatialCorrelation
   :members:

KroneckerModel
----------------

.. autoclass:: sionna.channel.KroneckerModel
   :members:

PerColumnModel
----------------

.. autoclass:: sionna.channel.PerColumnModel
   :members:


.. _time-domain:

Channel model interface
=========================

.. autoclass:: sionna.channel.ChannelModel
   :members:

Time domain channel
====================

The model of the channel in the time domain assumes pulse shaping and receive
filtering are performed using a conventional sinc filter (see, e.g., [Tse]_).
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
   They are set when instantiating :class:`~sionna.channel.TimeChannel`,
   :class:`~sionna.channel.GenerateTimeChannel`, and when calling the utility
   function :func:`~sionna.channel.cir_to_time_channel`.
   Because the sinc filter is neither time-limited nor causal, the discrete-time
   channel model is not causal. Therefore, ideally, one would set
   :math:`L_{\text{min}} = -\infty` and :math:`L_{\text{max}} = +\infty`.
   In practice, however, these two parameters need to be set to reasonable
   finite values. Values for these two parameters can be computed using the
   :func:`~sionna.channel.time_lag_discrete_time_channel` utility function from
   a given bandwidth and maximum delay spread.
   This function returns :math:`-6` for :math:`L_{\text{min}}`. :math:`L_{\text{max}}` is computed
   from the specified bandwidth and maximum delay spread, which default value is
   :math:`3 \mu s`. These values for :math:`L_{\text{min}}` and the maximum delay spread
   were found to be valid for all the models available in Sionna when an RMS delay
   spread of 100ns is assumed.

TimeChannel
------------

.. autoclass:: sionna.channel.TimeChannel
   :members:
   :exclude-members: call, build

GenerateTimeChannel
--------------------

.. autoclass:: sionna.channel.GenerateTimeChannel
   :members:
   :exclude-members: call, build

ApplyTimeChannel
-----------------

.. autoclass:: sionna.channel.ApplyTimeChannel
   :members:
   :exclude-members: call, build

cir_to_time_channel
--------------------

.. autofunction:: sionna.channel.cir_to_time_channel

time_to_ofdm_channel
--------------------

.. autofunction:: sionna.channel.time_to_ofdm_channel

.. _ofdm-waveform:

Channel with OFDM waveform
===========================

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
   \begin{align}
   \widehat{h}_{u, k, v, l, s, n}
      &= \int_{-\infty}^{+\infty} h_{u, k, v, l}(s,\tau) e^{-j2\pi n \Delta_f \tau} d\tau\\
      &= \sum_{m=0}^{M-1} a_{u, k, v, l, m}(s)
      e^{-j2\pi n \Delta_f \tau_{u, k, v, l, m}}
   \end{align}

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
   This can be achieved by using the :class:`~sionna.ofdm.OFDMModulator` and
   :class:`~sionna.ofdm.OFDMDemodulator` layers, and the
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

OFDMChannel
-------------

.. autoclass:: sionna.channel.OFDMChannel
   :members:
   :exclude-members: call, build

GenerateOFDMChannel
---------------------

.. autoclass:: sionna.channel.GenerateOFDMChannel
   :members:
   :exclude-members: call, build

ApplyOFDMChannel
-----------------

.. autoclass:: sionna.channel.ApplyOFDMChannel
   :members:
   :exclude-members: call, build

cir_to_ofdm_channel
--------------------

.. autofunction:: sionna.channel.cir_to_ofdm_channel

Rayleigh block fading
======================

.. autoclass:: sionna.channel.RayleighBlockFading
   :members:
   :exclude-members: call, build

3GPP 38.901 channel models
===========================

The submodule ``tr38901`` implements 3GPP channel models from [TR38901]_.

The :ref:`CDL <cdl>`, :ref:`UMi <umi>`, :ref:`UMa <uma>`, and :ref:`RMa <rma>`
models require setting-up antenna models for the transmitters and
receivers. This is achieved using the
:class:`~sionna.channel.tr38901.PanelArray` class.

The :ref:`UMi <umi>`, :ref:`UMa <uma>`, and :ref:`RMa <rma>` models require
setting-up a network topology, specifying, e.g., the user terminals (UTs) and
base stations (BSs) locations, the UTs velocities, etc.
:ref:`Utility functions <utility-functions>` are available to help laying out
complex topologies or to quickly setup simple but widely used topologies.

PanelArray
-----------

.. autoclass:: sionna.channel.tr38901.PanelArray
   :members:

Antenna
--------

.. autoclass:: sionna.channel.tr38901.Antenna
   :members:

AntennaArray
-------------

.. autoclass:: sionna.channel.tr38901.AntennaArray
   :members:

.. _tdl:

Tapped delay line (TDL)
------------------------

.. autoclass:: sionna.channel.tr38901.TDL
   :members:
   :exclude-members: __call__

.. _cdl:

Clustered delay line (CDL)
---------------------------

.. autoclass:: sionna.channel.tr38901.CDL
   :members:
   :exclude-members: __call__

.. _umi:

Urban microcell (UMi)
----------------------

.. autoclass:: sionna.channel.tr38901.UMi
   :members:
   :exclude-members: __call__
   :inherited-members:

.. _uma:

Urban macrocell (UMa)
-----------------------

.. autoclass:: sionna.channel.tr38901.UMa
   :members:
   :exclude-members: __call__
   :inherited-members:

.. _rma:

Rural macrocell (RMa)
-----------------------

.. autoclass:: sionna.channel.tr38901.RMa
   :members:
   :exclude-members: __call__
   :inherited-members:

.. _external-datasets:

External datasets
==================

.. autoclass:: sionna.channel.CIRDataset
   :members:
   :exclude-members: __call__
   :inherited-members:

.. _utility-functions:

Utility functions
===================

subcarrier_frequencies
------------------------

.. autofunction:: sionna.channel.subcarrier_frequencies

time_lag_discrete_time_channel
-------------------------------

.. autofunction:: sionna.channel.time_lag_discrete_time_channel

deg_2_rad
----------

.. autofunction:: sionna.channel.deg_2_rad

rad_2_deg
----------

.. autofunction:: sionna.channel.rad_2_deg

wrap_angle_0_360
-----------------

.. autofunction:: sionna.channel.wrap_angle_0_360

drop_uts_in_sector
-------------------

.. autofunction:: sionna.channel.drop_uts_in_sector

relocate_uts
-------------

.. autofunction:: sionna.channel.relocate_uts

set_3gpp_scenario_parameters
------------------------------

.. autofunction:: sionna.channel.set_3gpp_scenario_parameters

gen_single_sector_topology
---------------------------------

.. autofunction:: sionna.channel.gen_single_sector_topology

gen_single_sector_topology_interferers
-------------------------------------------------

.. autofunction:: sionna.channel.gen_single_sector_topology_interferers

exp_corr_mat
-------------

.. autofunction:: sionna.channel.exp_corr_mat

one_ring_corr_mat
-------------------

.. autofunction:: sionna.channel.one_ring_corr_mat


References:
   .. [TR38901] 3GPP TR 38.901,
      "Study on channel model for frequencies from 0.5 to 100 GHz", Release 16.1

   .. [TS38141-1] 3GPP TS 38.141-1
      "Base Station (BS) conformance testing Part 1: Conducted conformance testing",
      Release 17

   .. [Tse] D\. Tse and P\. Viswanath, “Fundamentals of wireless communication“,
      Cambridge University Press, 2005.

   .. [SoS] C\. Xiao, Y\. R\. Zheng and N\. C\. Beaulieu, "Novel Sum-of-Sinusoids Simulation Models for Rayleigh and Rician Fading Channels," in IEEE Transactions on Wireless Communications, vol. 5, no. 12, pp. 3667-3679, December 2006, doi: 10.1109/TWC.2006.256990.

   .. [MAL2018] R\. K\. Mallik,
         "The exponential correlation matrix: Eigen-analysis and
         applications", IEEE Trans. Wireless Commun., vol. 17, no. 7,
         pp. 4690-4705, Jul. 2018.

   .. [BHS2017] E\. Björnson, J\. Hoydis, L\. Sanguinetti (2017),
         `“Massive MIMO Networks: Spectral, Energy, and Hardware Efficiency”
         <https://massivemimobook.com>`_,
         Foundations and Trends in Signal Processing:
         Vol. 11, No. 3-4, pp 154–655.
