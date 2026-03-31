Wireless
========

This module provides blocks and functions that implement wireless channel models.
Models currently available include :class:`~sionna.phy.channel.AWGN`, :ref:`flat-fading <flat-fading>` with (optional) :class:`~sionna.phy.channel.SpatialCorrelation`, :class:`~sionna.phy.channel.RayleighBlockFading`, as well as models from the 3rd Generation Partnership Project (3GPP) :cite:p:`TR38901`: :ref:`TDL <tdl>`, :ref:`CDL <cdl>`, :ref:`UMi <umi>`, :ref:`UMa <uma>`, and :ref:`RMa <rma>`. It is also possible to :ref:`use externally generated CIRs <external-datasets>`.

Apart from :ref:`flat-fading <flat-fading>`, all of these models generate channel impulse responses (CIRs) that can then be used to
implement a channel transfer function in the :ref:`time domain <time-domain>` or
:ref:`assuming an OFDM waveform <ofdm-waveform>`.

This is achieved using the different functions, classes, and blocks which
operate as shown in the figures below.

.. figure:: ../../../figures/channel_arch_time.png
   :align: center

   Channel module architecture for time domain simulations.

.. figure:: ../../../figures/channel_arch_freq.png
   :align: center

   Channel module architecture for simulations assuming OFDM waveform.

A channel model generate CIRs from which channel responses in the time domain
or in the frequency domain are computed using the
:func:`~sionna.phy.channel.cir_to_time_channel` or
:func:`~sionna.phy.channel.cir_to_ofdm_channel` functions, respectively.
If one does not need access to the raw CIRs, the
:class:`~sionna.phy.channel.GenerateTimeChannel` and
:class:`~sionna.phy.channel.GenerateOFDMChannel` classes can be used to conveniently
sample CIRs and generate channel responses in the desired domain.

Once the channel responses in the time or frequency domain are computed, they
can be applied to the channel input using the
:class:`~sionna.phy.channel.ApplyTimeChannel` or
:class:`~sionna.phy.channel.ApplyOFDMChannel` blocks.

The following code snippets show how to setup and run a Rayleigh block fading
model assuming an OFDM waveform, and without accessing the CIRs or
channel responses.
This is the easiest way to setup a channel model.
Setting-up other models is done in a similar way, except for
:class:`~sionna.phy.channel.AWGN` (see the :class:`~sionna.phy.channel.AWGN`
class documentation).

.. code-block:: Python

   rayleigh = RayleighBlockFading(num_rx = 1,
                                  num_rx_ant = 32,
                                  num_tx = 4,
                                  num_tx_ant = 2)

   channel  = OFDMChannel(channel_model = rayleigh,
                          resource_grid = rg)

where ``rg`` is an instance of :class:`~sionna.phy.ofdm.ResourceGrid`.

Running the channel model is done as follows:

.. code-block:: Python

   # x is the channel input
   # no is the noise variance
   y = channel(x, no)

To use the time domain representation of the channel, one can use
:class:`~sionna.phy.channel.TimeChannel` instead of
:class:`~sionna.phy.channel.OFDMChannel`.

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

where ``rg`` is an instance of :class:`~sionna.phy.ofdm.ResourceGrid`.
Running the channel model is done as follows:

.. code-block:: Python

   # Generate a batch of channel responses
   h = generate_channel(batch_size)
   # Apply the channel
   # x is the channel input
   # no is the noise variance
   y = apply_channel(x, h, no)

Generating and applying the channel in the time domain can be achieved by using
:class:`~sionna.phy.channel.GenerateTimeChannel` and
:class:`~sionna.phy.channel.ApplyTimeChannel` instead of
:class:`~sionna.phy.channel.GenerateOFDMChannel` and
:class:`~sionna.phy.channel.ApplyOFDMChannel`, respectively.

To access the CIRs, setting up the channel can be done as follows:

.. code-block:: Python

   rayleigh = RayleighBlockFading(num_rx = 1,
                                  num_rx_ant = 32,
                                  num_tx = 4,
                                  num_tx_ant = 2)

   apply_channel = ApplyOFDMChannel()

and running the channel model as follows:

.. code-block:: Python

   cir = rayleigh(batch_size, num_time_steps, sampling_frequency)
   h = cir_to_ofdm_channel(frequencies, *cir)
   y = apply_channel(x, h, no)

where ``frequencies`` are the subcarrier frequencies in the baseband, which can
be computed using the :func:`~sionna.phy.channel.subcarrier_frequencies` utility
function.

Applying the channel in the time domain can be done by using
:func:`~sionna.phy.channel.cir_to_time_channel` and
:class:`~sionna.phy.channel.ApplyTimeChannel` instead of
:func:`~sionna.phy.channel.cir_to_ofdm_channel` and
:class:`~sionna.phy.channel.ApplyOFDMChannel`, respectively.

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

A channel model, such as :class:`~sionna.phy.channel.RayleighBlockFading` or
:class:`~sionna.phy.channel.tr38901.UMi`, is used to generate for each link between
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
   \begin{aligned}
      M                     &= 1\\
      \tau_{u, v, 0}  &= 0\\
      a_{u, k, v, l, 0}     &\sim \mathcal{CN}(0,1).
   \end{aligned}

3GPP channel models use the procedure depicted in :cite:p:`TR38901` to generate power
delay profiles. With these models, the power delay profiles are time-*variant*
in the event of mobility.




.. currentmodule:: sionna.phy.channel

.. toctree::
   :maxdepth: 3
   :hidden:

   3gpp
   awgn
   cir_dataset
   channel_model
   flat_fading
   frequency_domain
   rayleigh_block_fading
   time_domain
   utils
