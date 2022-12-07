Low-Density Parity-Check (LDPC)
===============================

The low-density parity-check (LDPC) code module supports 5G compliant LDPC codes and allows iterative belief propagation (BP) decoding.
Further, the module supports rate-matching for 5G and provides a generic linear encoder.

The following code snippets show how to setup and run a rate-matched 5G compliant LDPC encoder and a corresponding belief propagation (BP) decoder.

First, we need to create instances of :class:`~sionna.fec.ldpc.encoding.LDPC5GEncoder` and :class:`~sionna.fec.ldpc.decoding.LDPC5GDecoder`:

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
   # c contains the polar encoded codewords and has shape [...,n].
   c = encoder(u)

   # --- decoder ---
   # llr contains the log-likelihood ratios from the demapper and has shape [...,n].
   # u_hat contains the estimated information bits and has shape [...,k].
   u_hat = decoder(llr)

LDPC Encoder
************

LDPC5GEncoder
-------------

.. autoclass:: sionna.fec.ldpc.encoding.LDPC5GEncoder
   :members:
   :exclude-members: call, build

LDPC Decoder
************

LDPCBPDecoder
-------------
.. autoclass:: sionna.fec.ldpc.decoding.LDPCBPDecoder
   :members:
   :exclude-members: call, build

LDPC5GDecoder
-------------
.. autoclass:: sionna.fec.ldpc.decoding.LDPC5GDecoder
   :members:
   :exclude-members: call, build


References:
   .. [Pfister] J. Hou, P. H. Siegel, L. B. Milstein, and H. D. Pfister,
      “Capacity-approaching bandwidth-efficient coded modulation schemes
      based on low-density parity-check codes,” IEEE Trans. Inf. Theory,
      Sep. 2003.

   .. [3GPPTS38212_LDPC] ETSI 3GPP TS 38.212 "5G NR Multiplexing and channel
      coding", v.16.5.0, 2021-03.

   .. [Ryan] W. Ryan, "An Introduction to LDPC codes", CRC Handbook for
      Coding and Signal Processing for Recording Systems, 2004.

   .. [TF_ragged] https://www.tensorflow.org/guide/ragged_tensor

   .. [Richardson] T. Richardson and S. Kudekar. "Design of low-density
      parity-check codes for 5G new radio," IEEE Communications
      Magazine 56.3, 2018.

   .. [Nachmani] E. Nachmani, Y. Be'ery, and D. Burshtein. "Learning to
      decode linear codes using deep learning," IEEE Annual Allerton
      Conference on Communication, Control, and Computing (Allerton),
      2016.

   .. [Cammerer] S. Cammerer, M. Ebada, A. Elkelesh, and S. ten Brink.
      "Sparse graphs for belief propagation decoding of polar codes."
      IEEE International Symposium on Information Theory (ISIT), 2018.
