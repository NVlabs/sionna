EXIT Analysis
=============

.. currentmodule:: sionna.phy.fec.utils
.. autosummary::
   :toctree: .

   plot_exit_chart
   get_exit_analytic
   plot_trajectory

The LDPC BP decoder allows to track the internal information flow (`extrinsic information`) during decoding via callbacks. This can be plotted in so-called EXIT Charts :cite:p:`tenBrinkEXIT` to visualize the decoding convergence.

.. image:: ../../../figures/exit.png

This short code snippet shows how to generate and plot EXIT charts:

.. code-block:: Python

   # parameters
   ebno_db = 2.5 # simulation SNR
   batch_size = 10000
   num_bits_per_symbol = 2 # QPSK
   num_iter = 20 # number of decoding iterations

   pcm_id = 4 # decide which parity check matrix should be used (0-2: BCH; 3: (3,6)-LDPC 4: LDPC 802.11n
   pcm, k, n , coderate = load_parity_check_examples(pcm_id, verbose=True)

   noise_var = ebnodb2no(ebno_db=ebno_db,
                         num_bits_per_symbol=num_bits_per_symbol,
                         coderate=coderate)

   # init callbacks for tracking of EXIT charts
   cb_exit_vn = EXITCallback(num_iter)
   cb_exit_cn = EXITCallback(num_iter)

   # init components
   decoder = LDPCBPDecoder(pcm,
                           hard_out=False,
                           cn_update="boxplus",
                           num_iter=num_iter,
                           v2c_callbacks=[cb_exit_vn,], # register callbacks
                           c2v_callbacks=[cb_exit_cn,],) # register callbacks

   # generates fake llrs as if the all-zero codeword was transmitted over an AWNG channel with BPSK modulation
   llr_source = GaussianPriorSource()


   # generate fake LLRs (Gaussian approximation)
   # Remark: the EXIT callbacks require all-zero codeword simulations
   llr_ch = llr_source([batch_size, n], noise_var)

   # simulate free running decoder (for EXIT trajectory)
   decoder(llr_ch)

   # calculate analytical EXIT characteristics
   # Hint: these curves assume asymptotic code length, i.e., may become inaccurate in the short length regime
   Ia, Iev, Iec = get_exit_analytic(pcm, ebno_db)

   # and plot the analytical exit curves

   plt = plot_exit_chart(Ia, Iev, Iec)

   # and add simulated trajectory (requires "track_exit=True")
   plot_trajectory(plt, cb_exit_vn.mi.numpy(), cb_exit_cn.mi.numpy(), ebno_db)

Remark: for rate-matched 5G LDPC codes, the EXIT approximation becomes
inaccurate due to the rate-matching and the very specific structure of the
code.

