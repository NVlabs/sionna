Optical
=======
This module provides layers and functions that implement channel models for (fiber) optical communications.
The currently only available model is the split-step Fourier method (:class:`~sionna.phy.channel.SSFM`, for dual- and
single-polarization) that can be combined with an Erbium-doped amplifier (:class:`~sionna.phy.channel.EDFA`).

The following code snippets show how to setup and simulate the transmission
over a single-mode fiber (SMF) by using the split-step Fourier method.

.. code-block:: Python

      # init fiber
      span = sionna.phy.channel.optical.SSFM(
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
                                    dtype=torch.complex64)
      # init amplifier
      amplifier = sionna.phy.channel.optical.EDFA(
                                    g=4.0,
                                    f=2.0,
                                    f_c=193.55e12,
                                    dt=1.0e-12)

      @torch.compile
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


For further details, the tutorial `"Optical Channel with Lumped Amplification" <../../../tutorials/notebooks/Optical_Lumped_Amplification_Channel.ipynb>`_  provides more sophisticated examples of how to use this module.

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
| :math:`\hat{D}`                 | Linear SSFM operator (Agrawal)                                              |
+---------------------------------+-----------------------------------------------------------------------------+
| :math:`\hat{N}`                 | Non-linear SSFM operator (Agrawal)                                          |
+---------------------------------+-----------------------------------------------------------------------------+
| :math:`f_\textrm{sim}`          | Simulation bandwidth                                                        |
+---------------------------------+-----------------------------------------------------------------------------+

See :cite:p:`A2012` for the definition of the linear and non-linear SSFM operators.

**Remark:** Depending on the exact simulation parameters, the SSFM algorithm may require ``dtype=torch.complex128`` for accurate simulation results. However, this may increase the simulation complexity significantly.

.. currentmodule:: sionna.phy.channel

.. autosummary::
   :toctree: .

   SSFM
   EDFA
   utils.time_frequency_vector

