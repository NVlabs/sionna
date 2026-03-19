Low-Density Parity-Check (LDPC)
===============================

The low-density parity-check (LDPC) code module supports 5G compliant LDPC codes and allows iterative belief propagation (BP) decoding.
Further, the module supports rate-matching for 5G and provides a generic linear
encoder.

The following code snippets show how to setup and run a rate-matched 5G compliant LDPC encoder and a corresponding belief propagation (BP) decoder.

First, we need to create instances of :class:`~sionna.phy.fec.ldpc.encoding.LDPC5GEncoder` and :class:`~sionna.phy.fec.ldpc.decoding.LDPC5GDecoder`:

.. code-block:: Python

   encoder = LDPC5GEncoder(k                 = 100, # number of information bits (input)
                           n                 = 200) # number of codeword bits (output)


   decoder = LDPC5GDecoder(encoder           = encoder,
                           num_iter          = 20, # number of BP iterations
                           return_infobits   = True)

Now, the encoder and decoder can be used by:

.. code-block:: Python

   # --- encoder ---
   # u contains the information bits to be encoded and has shape [...,k].
   # c contains the encoded codewords and has shape [...,n].
   c = encoder(u)

   # --- decoder ---
   # llr contains the log-likelihood ratios from the demapper and has shape [...,n].
   # u_hat contains the estimated information bits and has shape [...,k].
   u_hat = decoder(llr)

.. currentmodule:: sionna.phy.fec.ldpc

.. autosummary::
   :toctree: .

   LDPC5GEncoder
   LDPCBPDecoder
   LDPC5GDecoder

.. toctree::
   :hidden:
   :maxdepth: 3

   node_update_functions
   decoder_callbacks