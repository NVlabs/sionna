PHY Abstraction
===============

.. figure:: ../figures/phy_abs_api.png
   :align: center
   :width: 100%


| The physical layer (PHY) abstraction method follows a two-step approach:

1. The signal-to-interference-plus-noise
   ratio (SINR) that a single codeword experiences across multiple streams,
   computed via :class:`~sionna.phy.ofdm.PostEqualizationSINR`, is 
   aggregated into a single *effective* SINR value. 
   The effective SINR is chosen so that, if all subcarriers and streams experienced
   it uniformly, the resulting block error rate (BLER) would remain
   approximately the same.
2. The effective SINR is then mapped to a BLER value via
   precomputed tables, based on the code block size.

The transport BLER (TBLER) can be finally computed as the probability that at
least one of the code blocks in the transport block is not correctly received. 

For a usage example of PHY abstraction in Sionna, refer
to the `Physical Layer Abstraction notebook
<../tutorials/PHY_Abstraction.html>`_.

Next, we formally describe the general principle of effective SINR mapping (ESM)
and the exponential ESM (EESM) model. 

| We assume the presence of multiple channel "links" :math:`i=1,\dots,N`, each
  characterized by its own :math:`\mathrm{SINR}_{i}`. 
  In principle, different codeword symbols can be transmitted on the same link,
  meaning they experience the same SINR.
| Let :math:`I(x)` measure the "quality" of a link with SINR value :math:`x`,
  the exact interpretation of which will be discussed later. 
| The effective SINR, :math:`\mathrm{SINR}_{\text{eff}}`, is defined
  as the SINR of a single-link channel whose quality matches the average
  quality of the multi-link channel:  
    
    .. math::
        I(\mathrm{SINR}_{\text{eff}}) = \frac{1}{N} \sum_{i=1}^N I(\mathrm{SINR}_{i})
        
    .. math::
        \Rightarrow \ \ \mathrm{SINR}_{\text{eff}} = I^{-1} \left( \frac{1}{N} \sum_{i=1}^N I(\mathrm{SINR}_{i}) \right)
    
| The form of the quality measure :math:`I` depends on the selected ESM method. 
| In the **exponential ESM (EESM)** model, the link
  quality is defined as:
  
.. math::
    I^{\mathrm{EESM}}(x) := \exp(-x/\beta).

Thus, the corresponding effective SINR can be expressed as: 

.. math::
    :label: EESM

    \mathrm{SINR}_{\mathrm{eff}}^{\mathrm{EESM}} = -\beta \log \left( \frac{1}{N} \sum_{i=1}^N e^{-\mathrm{SINR}_i/\beta} \right).

| In the following we outline the derivation of this expression, assuming the
  transmission of BPSK (:math:`\pm 1`) modulated codewords :math:`u^A` and
  :math:`u^B`, with a Hamming distance of :math:`d`.

**Single-link channel.** In the basic case with one link (:math:`N=1`), each codeword
symbol experiences the same channel gain :math:`\rho` and complex noise power
:math:`N_0`, resulting in the received *real* signal: 

.. math::
    y^{k}_j = \sqrt{\rho} u^{k}_j + w_j, \quad k\in\{A,B\}, \ \forall\, j

| where :math:`j` indexes the symbols and :math:`w_j \sim
  \mathcal{N}(0, N_0/2)` is additive real noise. Hence, the SNR (as well as the SINR, since
  interference is not considered) is :math:`\rho /N_0`.
| Codeword :math:`u^A` is incorrectly decoded as :math:`u^B` when the noise
  projected along the direction :math:`u^A-u^B` exceeds the half distance
  between the two codewords, equal to :math:`\sqrt{d\rho}`.
  Hence, the pairwise error probability :math:`P^{N=1}(u^A \rightarrow u^B)` can
  be expressed as:

.. math::
    :label: pairwise

    \begin{align}
        P^{N=1}\left(u^A \rightarrow u^B\right) = & \, \Pr\left( \xi \sqrt{N_0/2} >
        \sqrt{d\rho} \right), \quad \xi\sim \mathcal{N}(0,1) \\
        = & \, Q\left( \sqrt{2 d \, \mathrm{SINR}} \right) \\
        \le & \, e^{-d\, \mathrm{SINR}}
    \end{align}
  
| where :math:`Q(x)` is the tail distribution function of the standard normal
  distribution and the inequality stems from the Chernoff bound :math:`Q(x) \le
  e^{-x^2/2}`, for all :math:`x`.

| **Two-link channel.** We now assume that each symbol is transmitted through
  channel link 1 or
  2 with probabilities :math:`p_1` and :math:`p_2`, respectively. Link
  :math:`i=1,2` is characterized by its channel gain :math:`\sqrt{\rho_i}`.  
| Consider two received noiseless codewords :math:`u^A, u^B` where
  :math:`\ell_1=\ell` and :math:`\ell_2=d-\ell` symbols experience channel 1 and
  2, respectively. Then, their half distance is :math:`\sqrt{\ell_1 \rho_1 +
  \ell_2 \rho_2}` and the conditioned pairwise error probability equals: 

.. math::
    :label: conditioned

    \begin{align}
        P^{N=2}\left(u^A \rightarrow u^B | \ell_1,\ell_2\right) = & \, \Pr\left( \sqrt{N_0/2} \xi > \sqrt{\ell_1 \rho_1 + \ell_2 \rho_2} \right) \\
        = & \, Q\left( \sqrt{2\ell_1 \, \mathrm{SINR}_1 + 2\ell_2 \, \mathrm{SINR}_2 } \right).
    \end{align}

To obtain the pairwise
codeword error probability, we average expression :eq:`conditioned` across all
:math:`(\ell_1, \ell_2)` events: 

.. math::
    :label: N2

    \begin{align}
        P^{N=2}\left(u^A \rightarrow u^B\right) & \, = \sum_{\ell=0}^d {d \choose \ell} p_1^{\ell} \, p_2^{d-\ell} \, P^{N=2}\left(u^A \rightarrow u^B | \ell_1=\ell,\ell_2=d-\ell\right) \\
        \le & \, \sum_{\ell=0}^d {d \choose \ell} \left(p_1 e^{-\mathrm{SINR}_1}\right)^\ell \left( p_2 e^{-\mathrm{SINR}_2} \right)^{d-\ell} \\
        = & \, \left( p_1 e^{-\mathrm{SINR}_1}  + p_2 e^{-\mathrm{SINR}_2}\right)^d
    \end{align}

where the inequality stems again from the Chernoff bound. 

**Multi-link channel.** Expression :eq:`N2` extends to a multi-link channel
(:math:`N\ge 2`) as follows:

.. math::
    :label: pr_multistate

    \begin{align}
        P^{N}\left(u^A \rightarrow u^B\right) \le \, \left( \sum_{i=1}^N p_i e^{-\mathrm{SINR}_i} \right)^d.
    \end{align}

**EESM expression.** By equating the multi-link pairwise error probability bound :eq:`pr_multistate` with
the analogous single-link expression :eq:`pairwise`, we recognize that the
multi-link channel is analogous to a single-link channel with SINR: 

.. math::
    \mathrm{SINR}_{\mathrm{eff}}^{\mathrm{EESM}} := -\log \left( \sum_{i=1}^N p_i e^{-\mathrm{SINR}_i} \right)

| where :math:`\mathrm{SINR}_{\mathrm{eff}}^{\mathrm{EESM}}` is the *effective* SINR for the
  multi-link channel under the EESM model.
| If we further assume that all links are equiprobable, i.e., :math:`p_i=1/N`
  for all :math:`i`, then we obtain expression :eq:`EESM` with :math:`\beta=1`.

Note that the introduction of parameter :math:`\beta` in :eq:`EESM` is useful
to adapt the EESM formula to different modulation and coding schemes (MCS),
since the argument above holds for BPSK modulation only. Hence, :math:`\beta` shall
depend on the used MCS, as shown in [5GLENA]_.


.. autoclass:: sionna.sys.EffectiveSINR
  :members:
  :exclude-members: call, build

.. autoclass:: sionna.sys.EESM
  :members:
  :exclude-members: call, build

.. autoclass:: sionna.sys.PHYAbstraction
    :members:
    :exclude-members: call, build


References:
  .. [5GLENA] S. Lagen, K. Wanuga, H. Elkotby, S. Goyal, N. Patriciello, L.
    Giupponi. `"New radio physical layer abstraction for
    system-level simulations of 5G networks" <https://arxiv.org/abs/2001.10309>`_. IEEE International
    Conference on Communications (ICC), 2020  