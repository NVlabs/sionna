=======
Optical
=======

This module provides layers and functions that implement channel models for (fiber) optical communications.
The currently only available model is the split-step Fourier method (:class:`~sionna.channel.SSFM`, for dual- and
single-polarization) that can be combined with an Erbium-doped amplifier (:class:`~sionna.channel.EDFA`).

The following code snippets show how to setup and simulate the transmission
over a single-mode fiber (SMF) by using the split-step Fourier method.

.. code-block:: Python

      # init fiber
      span = sionna.channel.optical.SSFM(
                                    alpha=0.046,
                                    beta_2=-21.67,
                                    f_c=193.55e12,
                                    gamma=1.27,
                                    length=80,
                                    n_ssfm=200,
                                    n_sp=1.0,
                                    t_norm=1e-12,
                                    with_amplification=False,
                                    with_attenuation=True,
                                    with_dispersion=True,
                                    with_nonlinearity=True,
                                    dtype=tf.complex64)
      # init amplifier
      amplifier = sionna.channel.optical.EDFA(
                                    g=4.0,
                                    f=2.0,
                                    f_c=193.55e12,
                                    dt=1.0e-12)

      @tf.function
      def simulate_transmission(x, n_span):
            y = x
            # simulate n_span fiber spans
            for _ in range(n_span):
                  # simulate single span
                  y = span(y)
                  # simulate amplifier
                  y = amplifier(y)

            return y


Running the channel model is done as follows:

.. code-block:: Python

      # x is the optical input signal, n_span the number of spans
      y = simulate_transmission(x, n_span)


For further details, the tutorial `"Optical Channel with Lumped Amplification" <../examples/Optical_Lumped_Amplification_Channel.html>`_  provides more sophisticated examples of how to use this module.

For the purpose of the present document, the following symbols apply:

+---------------------------------+-----------------------------------------------------------------------------+
| :math:`T_\text{norm}`           | Time normalization for the SSFM in :math:`(\text{s})`                       |
+---------------------------------+-----------------------------------------------------------------------------+
| :math:`L_\text{norm}`           | Distance normalization the for SSFM in :math:`(\text{m})`                   |
+---------------------------------+-----------------------------------------------------------------------------+
| :math:`W`                       | Bandwidth                                                                   |
+---------------------------------+-----------------------------------------------------------------------------+
| :math:`\alpha`                  | Attenuation coefficient in :math:`(1/L_\text{norm})`                        |
+---------------------------------+-----------------------------------------------------------------------------+
| :math:`\beta_2`                 | Group velocity dispersion coeff. in :math:`(T_\text{norm}^2/L_\text{norm})` |
+---------------------------------+-----------------------------------------------------------------------------+
| :math:`f_\mathrm{c}`            | Carrier frequency in  :math:`\text{(Hz)}`                                   |
+---------------------------------+-----------------------------------------------------------------------------+
| :math:`\gamma`                  | Nonlinearity coefficient in :math:`(1/L_\text{norm}/\text{W})`              |
+---------------------------------+-----------------------------------------------------------------------------+
| :math:`\ell`                    | Fiber length in :math:`(L_\text{norm})`                                     |
+---------------------------------+-----------------------------------------------------------------------------+
| :math:`h`                       | Planck constant                                                             |
+---------------------------------+-----------------------------------------------------------------------------+
| :math:`N_\mathrm{SSFM}`         | Number of SSFM simulation steps                                             |
+---------------------------------+-----------------------------------------------------------------------------+
| :math:`n_\mathrm{sp}`           | Spontaneous emission factor of Raman amplification                          |
+---------------------------------+-----------------------------------------------------------------------------+
| :math:`\Delta_t`                | Normalized simulation time step in :math:`(T_\text{norm})`                  |
+---------------------------------+-----------------------------------------------------------------------------+
| :math:`\Delta_z`                | Normalized simulation step size in :math:`(L_\text{norm})`                  |
+---------------------------------+-----------------------------------------------------------------------------+
| :math:`G`                       | Amplifier gain                                                              |
+---------------------------------+-----------------------------------------------------------------------------+
| :math:`F`                       | Amplifier's noise figure                                                    |
+---------------------------------+-----------------------------------------------------------------------------+
| :math:`\rho_\text{ASE}`         | Noise spectral density                                                      |
+---------------------------------+-----------------------------------------------------------------------------+
| :math:`P`                       | Signal power                                                                |
+---------------------------------+-----------------------------------------------------------------------------+
| :math:`\hat{D}`                 | Linear SSFM operator [A2012]_                                               |
+---------------------------------+-----------------------------------------------------------------------------+
| :math:`\hat{N}`                 | Non-linear SSFM operator [A2012]_                                           |
+---------------------------------+-----------------------------------------------------------------------------+
| :math:`f_\textrm{sim}`          | Simulation bandwidth                                                        |
+---------------------------------+-----------------------------------------------------------------------------+


**Remark:** Depending on the exact simulation parameters, the SSFM algorithm may require ``dtype=tf.complex128`` for accurate simulation results. However, this may increase the simulation complexity significantly.

Split-step Fourier method
=========================

.. autoclass:: sionna.channel.SSFM
   :members:
   :exclude-members: call, build

Erbium-doped fiber amplifier
============================

.. autoclass:: sionna.channel.EDFA
   :members:
   :exclude-members: call, build

Utility functions
=================

time_frequency_vector
---------------------

.. autofunction:: sionna.channel.utils.time_frequency_vector


References:
   .. [HT1973] R\. H\. Hardin and F\. D\. Tappert,
         "Applications of the Split-Step Fourier Method to the Numerical Solution of Nonlinear and Variable Coefficient Wave Equations.",
         SIAM Review Chronicles, Vol. 15, No. 2, Part 1, p 423, 1973.

   .. [FMF1976] J\. A\. Fleck, J\. R\. Morris, and M\. D\. Feit,
         "Time-dependent Propagation of High Energy Laser Beams Through the Atmosphere",
         Appl. Phys., Vol. 10, pp 129–160, 1976.

   .. [MFFP2009] N\. J\. Muga, M\. C\. Fugihara, M\. F\. S\. Ferreira, and A\. N\. Pinto,
         "ASE Noise Simulation in Raman Amplification Systems",
         Conftele, 2009.

   .. [A2012] G\. P\. Agrawal,
         "Fiber-optic Communication Systems",
         4th ed. Wiley Series in Microwave and Optical Engineering 222. New York: Wiley, 2010.

   .. [EKWFG2010] R\. J\. Essiambre, G\. Kramer, P\. J\. Winzer, G\. J\. Foschini, and B\. Goebel,
         "Capacity Limits of Optical Fiber Networks",
         Journal of Lightwave Technology 28, No. 4, 2010.

   .. [BGT2000] D\. M\. Baney, P\. Gallion, and R\. S\. Tucker,
         "Theory and Measurement Techniques for the Noise Figure of Optical Amplifiers",
         Optical Fiber Technology 6, No. 2, 2000.

   .. [GD1991] C\. R\. Giles, and E\. Desurvire,
         "Modeling Erbium-Doped Fiber Amplifiers",
         Journal of Lightwave Technology 9, No. 2, 1991.

   .. [WMC1991] P\. K\. A\. Wai, C\. R\. Menyuk, and H\. H\. Chen,
         "Stability of Solitons in Randomly Varying Birefringent Fibers",
         Optics Letters, No. 16, 1991.
