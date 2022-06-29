Polar Codes
###########

The Polar code module supports 5G-compliant Polar codes and includes successive cancellation (SC), successive cancellation list (SCL), and belief propagation (BP) decoding.

The module supports rate-matching and CRC-aided decoding.
Further, Reed-Muller (RM) code design is available and can be used in combination with the Polar encoding/decoding algorithms.

The following code snippets show how to setup and run a rate-matched 5G compliant Polar encoder and a corresponding successive cancellation list (SCL) decoder.

First, we need to create instances of :class:`~sionna.fec.polar.encoding.Polar5GEncoder` and :class:`~sionna.fec.polar.decoding.Polar5GDecoder`:


.. code-block:: Python

   encoder = Polar5GEncoder(k          = 100, # number of information bits (input)
                            n          = 200) # number of codeword bits (output)


   decoder = Polar5GDecoder(encoder    = encoder, # connect the Polar decoder to the encoder
                            dec_type   = "SCL", # can be also "SC" or "BP"
                            list_size  = 8)

Now, the encoder and decoder can be used by:

.. code-block:: Python

   # --- encoder ---
   # u contains the information bits to be encoded and has shape [...,k].
   # c contains the polar encoded codewords and has shape [...,n].
   c = encoder(u)

   # --- decoder ---
   # llr contains the log-likelihood ratios from the demapper and has shape [...,n].
   # u_hat contains the estimated information bits and has shape [...,k].
   u_hat = decoder(llr)


Polar Encoding
**************

Polar5GEncoder
--------------
.. autoclass:: sionna.fec.polar.encoding.Polar5GEncoder
   :members:
   :exclude-members: call, build

PolarEncoder
------------
.. autoclass:: sionna.fec.polar.encoding.PolarEncoder
   :members:
   :exclude-members: call, build


Polar Decoding
**************

Polar5GDecoder
--------------
.. autoclass:: sionna.fec.polar.decoding.Polar5GDecoder
   :members:
   :exclude-members: call, build

PolarSCDecoder
--------------
.. autoclass:: sionna.fec.polar.decoding.PolarSCDecoder
   :members:
   :exclude-members: call, build

PolarSCLDecoder
---------------
.. autoclass:: sionna.fec.polar.decoding.PolarSCLDecoder
   :members:
   :exclude-members: call, build

PolarBPDecoder
--------------
.. autoclass:: sionna.fec.polar.decoding.PolarBPDecoder
   :members:
   :exclude-members: call, build

Polar Utility Functions
***********************

generate_5g_ranking
-------------------
.. autofunction:: sionna.fec.polar.utils.generate_5g_ranking

generate_polar_transform_mat
----------------------------
.. autofunction:: sionna.fec.polar.utils.generate_polar_transform_mat

generate_rm_code
----------------
.. autofunction:: sionna.fec.polar.utils.generate_rm_code

generate_dense_polar
--------------------
.. autofunction:: sionna.fec.polar.utils.generate_dense_polar

References:
   .. [3GPPTS38212] ETSI 3GPP TS 38.212 "5G NR Multiplexing and channel
      coding", v.16.5.0, 2021-03.

   .. [Bioglio_Design] V. Bioglio, C. Condo, I. Land, "Design of
      Polar Codes in 5G New Radio," IEEE Communications Surveys &
      Tutorials, 2020. Online availabe https://arxiv.org/pdf/1804.04389.pdf

   .. [Hui_ChannelCoding] D. Hui, S. Sandberg, Y. Blankenship, M.
      Andersson, L. Grosjean "Channel coding in 5G new radio: A
      Tutorial Overview and Performance Comparison with 4G LTE," IEEE
      Vehicular Technology Magazine, 2018.

   .. [Arikan_Polar] E. Arikan, "Channel polarization: A method for
      constructing capacity-achieving codes for symmetric
      binary-input memoryless channels," IEEE Trans. on Information
      Theory, 2009.

   .. [Gross_Fast_SCL] Seyyed Ali Hashemi, Carlo Condo, and Warren J.
      Gross, "Fast and Flexible Successive-cancellation List Decoders
      for Polar Codes." IEEE Trans. on Signal Processing, 2017.

   .. [Tal_SCL] Ido Tal and Alexander Vardy, "List Decoding of Polar
      Codes." IEEE Trans Inf Theory, 2015.

   .. [Stimming_LLR] Alexios Balatsoukas-Stimming, Mani Bastani Parizi,
      Andreas Burg, "LLR-Based Successive Cancellation List Decoding
      of Polar Codes." IEEE Trans Signal Processing, 2015.

   .. [Hashemi_SSCL] Seyyed Ali Hashemi, Carlo Condo, and Warren J.
      Gross, "Simplified Successive-Cancellation List Decoding
      of Polar Codes." IEEE ISIT, 2016.

   .. [Cammerer_Hybrid_SCL] Sebastian Cammerer, Benedikt Leible, Matthias
      Stahl, Jakob Hoydis, and Stephan ten Brink, "Combining Belief
      Propagation and Successive Cancellation List Decoding of Polar
      Codes on a GPU Platform," IEEE ICASSP, 2017.

   .. [Arikan_BP] E. Arikan, “A Performance Comparison of Polar Codes and
      Reed-Muller Codes,” IEEE Commun. Lett., vol. 12, no. 6, pp.
      447-449, Jun. 2008.

   .. [Forney_Graphs] G. D. Forney, “Codes on graphs: normal realizations,”
      IEEE Trans. Inform. Theory, vol. 47, no. 2, pp. 520-548, Feb. 2001.

   .. [Ebada_Design] M. Ebada, S. Cammerer, A. Elkelesh and S. ten Brink,
      “Deep Learning-based Polar Code Design”, Annual Allerton
      Conference on Communication, Control, and Computing, 2019.

   .. [Goala_LP] N. Goela, S. Korada, M. Gastpar, "On LP decoding of Polar
        Codes," IEEE ITW 2010.
