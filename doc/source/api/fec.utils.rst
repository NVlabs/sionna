Utility Functions
=================

This module provides utility functions for the FEC package. It also provides serval functions to simplify EXIT analysis of iterative receivers.

(Binary) Linear Codes
***********************

Several functions are provided to convert parity-check matrices into generator matrices and vice versa. Please note that currently only binary codes are supported.

.. code-block:: Python

   # load example parity-check matrix
   pcm, k, n, coderate = load_parity_check_examples(pcm_id=3)

Note that many research projects provide their parity-check matrices in the  `alist` format [MacKay]_ (e.g., see [UniKL]_). The follwing code snippet provides an example of how to import an external LDPC parity-check matrix from an `alist` file and how to set-up an encoder/decoder.

.. code-block:: Python

   # load external example parity-check matrix in alist format
   al = load_alist(path=filename)
   pcm, k, n, coderate = alist2mat(al)

   # the linear encoder can be directly initialized with a parity-check matrix
   encoder = LinearEncoder(pcm, is_pcm=True)

   # initalize BP decoder for the given parity-check matrix
   decoder = LDPCBPDecoder(pcm, num_iter=20)

   # and run simulation with random information bits
   no = 1.
   batch_size = 10
   num_bits_per_symbol = 2

   source = BinarySource()
   mapper = Mapper("qam", num_bits_per_symbol)
   channel = AWGN()
   demapper = Demapper("app", "qam", num_bits_per_symbol)

   u = source([batch_size, k])
   c = encoder(u)
   x = mapper(c)
   y = channel([x, no])
   llr = demapper([y, no])
   c_hat = decoder(llr)

load_parity_check_examples
--------------------------
.. autofunction:: sionna.fec.utils.load_parity_check_examples

alist2mat
---------
.. autofunction:: sionna.fec.utils.alist2mat

load_alist
----------
.. autofunction:: sionna.fec.utils.load_alist

generate_reg_ldpc
-----------------
.. autofunction:: sionna.fec.utils.generate_reg_ldpc

make_systematic
---------------
.. autofunction:: sionna.fec.utils.make_systematic

gm2pcm
------
.. autofunction:: sionna.fec.utils.gm2pcm

pcm2gm
------
.. autofunction:: sionna.fec.utils.pcm2gm

verify_gm_pcm
-------------
.. autofunction:: sionna.fec.utils.verify_gm_pcm


EXIT Analysis
*************

The LDPC BP decoder allows to track the internal information flow (`extrinsic information`) during decoding. This can be plotted in so-called EXIT Charts [tenBrinkEXIT]_ to visualize the decoding convergence.

.. image:: ../figures/exit.png

This short code snippet shows how to generate and plot EXIT charts:

.. code-block:: Python

   # parameters
   ebno_db = 2.5 # simulation SNR
   batch_size = 10000
   num_bits_per_symbol = 2 # QPSK

   pcm_id = 4 # decide which parity check matrix should be used (0-2: BCH; 3: (3,6)-LDPC 4: LDPC 802.11n
   pcm, k, n , coderate = load_parity_check_examples(pcm_id, verbose=True)

   noise_var = ebnodb2no(ebno_db=ebno_db,
                         num_bits_per_symbol=num_bits_per_symbol,
                         coderate=coderate)

   # init components
   decoder = LDPCBPDecoder(pcm,
                           hard_out=False,
                           cn_type="boxplus",
                           trainable=False,
                           track_exit=True, # if activated, the decoder stores the outgoing extrinsic mutual information per iteration
                           num_iter=20)

   # generates fake llrs as if the all-zero codeword was transmitted over an AWNG channel with BPSK modulation
   llr_source = GaussianPriorSource()


   # generate fake LLRs (Gaussian approximation)
   llr = llr_source([[batch_size, n], noise_var])

   # simulate free running decoder (for EXIT trajectory)
   decoder(llr)

   # calculate analytical EXIT characteristics
   # Hint: these curves assume asymptotic code length, i.e., may become inaccurate in the short length regime
   Ia, Iev, Iec = get_exit_analytic(pcm, ebno_db)

   # and plot the analytical exit curves

   plt = plot_exit_chart(Ia, Iev, Iec)

   # and add simulated trajectory (requires "track_exit=True")
   plot_trajectory(plt, decoder.ie_v, decoder.ie_c, ebno_db)

Remark: for rate-matched 5G LDPC codes, the EXIT approximation becomes
inaccurate due to the rate-matching and the very specific structure of the
code.

plot_exit_chart
---------------
.. autofunction:: sionna.fec.utils.plot_exit_chart

get_exit_analytic
-----------------
.. autofunction:: sionna.fec.utils.get_exit_analytic

plot_trajectory
---------------
.. autofunction:: sionna.fec.utils.plot_trajectory


Miscellaneous
*************

GaussianPriorSource
-------------------
.. autoclass:: sionna.fec.utils.GaussianPriorSource

bin2int
-------
.. autofunction:: sionna.fec.utils.bin2int

int2bin
-------
.. autofunction:: sionna.fec.utils.int2bin

bin2int_tf
----------
.. autofunction:: sionna.fec.utils.bin2int_tf

int2bin_tf
----------
.. autofunction:: sionna.fec.utils.int2bin_tf

int_mod_2
---------
.. autofunction:: sionna.fec.utils.int_mod_2

llr2mi
------
.. autofunction:: sionna.fec.utils.llr2mi

j_fun
-----
.. autofunction:: sionna.fec.utils.j_fun

j_fun_inv
---------
.. autofunction:: sionna.fec.utils.j_fun_inv

j_fun_tf
--------
.. autofunction:: sionna.fec.utils.j_fun_tf

j_fun_inv_tf
------------
.. autofunction:: sionna.fec.utils.j_fun_inv_tf


References:
   .. [tenBrinkEXIT] S. ten Brink, “Convergence Behavior of Iteratively
      Decoded Parallel Concatenated Codes,” IEEE Transactions on
      Communications, vol. 49, no. 10, pp. 1727-1737, 2001.

   .. [Brannstrom] F. Brannstrom, L. K. Rasmussen, and A. J. Grant,
      “Convergence analysis and optimal scheduling for multiple
      concatenated codes,” IEEE Trans. Inform. Theory, vol. 51, no. 9,
      pp. 3354–3364, 2005.

   .. [Hagenauer] J. Hagenauer, “The Turbo Principle in Mobile
      Communications,” in Proc. IEEE Int. Symp. Inf. Theory and its Appl.
      (ISITA), 2002.

   .. [tenBrink] S. ten Brink, G. Kramer, and A. Ashikhmin, “Design of
      low-density parity-check codes for modulation and detection,” IEEE
      Trans. Commun., vol. 52, no. 4, pp. 670–678, Apr. 2004.

   .. [MacKay] http://www.inference.org.uk/mackay/codes/alist.html

   .. [UniKL] https://www.uni-kl.de/en/channel-codes/
