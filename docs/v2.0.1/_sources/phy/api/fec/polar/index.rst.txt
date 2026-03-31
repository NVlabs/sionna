Polar Codes
===========

The Polar code module supports 5G-compliant Polar codes and includes successive
cancellation (SC), successive cancellation list (SCL), and belief propagation
(BP) decoding. The module supports rate-matching and CRC-aided decoding. Further, Reed-Muller (RM) code design is available and can be used in combination with the Polar encoding/decoding algorithms.

The following code snippets show how to setup and run a rate-matched 5G compliant Polar encoder and a corresponding successive cancellation list (SCL) decoder.

First, we need to create instances of :class:`~sionna.phy.fec.polar.encoding.Polar5GEncoder` and :class:`~sionna.phy.fec.polar.decoding.Polar5GDecoder`:


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


.. currentmodule:: sionna.phy.fec.polar

.. autosummary::
   :toctree: .

   Polar5GEncoder
   PolarEncoder
   Polar5GDecoder
   PolarSCDecoder
   PolarSCLDecoder
   PolarBPDecoder


.. toctree::
   :hidden:
   :maxdepth: 3

   utils
