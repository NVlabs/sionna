Linear Codes
############

This package provides generic support for binary linear block codes.

For encoding, a universal :class:`~sionna.fec.linear.LinearEncoder` is available and can be initialized with either a generator or parity-check matrix. The matrix must be binary and of full rank.

For decoding, :class:`~sionna.fec.linear.OSDecoder` implements the
ordered-statistics decoding (OSD) algorithm [Fossorier]_ which provides close to
maximum-likelihood (ML) estimates for a sufficiently large order of the decoder.
Please note that OSD is highly complex and not feasible for all code lengths.

*Remark:* As this package provides support for generic encoding and decoding
(including Polar and LDPC codes), it cannot rely on code specific
optimizations. To benefit from an optimized decoder and keep the complexity as low as possible, please use the code specific enc-/decoders whenever available.

The encoder and decoder can be set up as follows:

.. code-block:: Python

   pcm, k, n, coderate = load_parity_check_examples(pcm_id=1) # load example code

   # or directly import an external parity-check matrix in alist format
   al = load_alist(path=filename)
   pcm, k, n, coderate = alist2mat(al)

   # encoder can be directly initialized with the parity-check matrix
   encoder = LinearEncoder(enc_mat=pcm, is_pcm=True)

   # decoder can be initialized with generator or parity-check matrix
   decoder = OSDecoder(pcm, t=4, is_pcm=True) # t is the OSD order

   # or instantiated from a specific encoder
   decoder = OSDecoder(encoder=encoder, t=4) # t is the OSD order

We can now run the encoder and decoder:

.. code-block:: Python

   # u contains the information bits to be encoded and has shape [...,k].
   # c contains codeword bits and has shape [...,n]
   c = encoder(u)

   # after transmission LLRs must be calculated with a demapper
   # let's assume the resulting llr_ch has shape [...,n]
   c_hat = decoder(llr_ch)


Encoder
*******

LinearEncoder
-------------
.. autoclass:: sionna.fec.linear.LinearEncoder
   :members:
   :exclude-members: call, build

AllZeroEncoder
--------------
.. autoclass:: sionna.fec.linear.AllZeroEncoder
   :members:
   :exclude-members: call, build

Decoder
*******

OSDecoder
---------
.. autoclass:: sionna.fec.linear.OSDecoder
   :members:
   :exclude-members: call, build

References:
   .. [Fossorier] M. Fossorier, S. Lin, "Soft-Decision Decoding of Linear
                  Block Codes Based on Ordered Statistics", IEEE Trans. Inf.
                  Theory, vol. 41, no.5, 1995.

   .. [Stimming_LLR_OSD] A.Balatsoukas-Stimming, M. Parizi, A. Burg,
                         "LLR-Based Successive Cancellation List Decoding
                         of Polar Codes." IEEE Trans Signal Processing, 2015.
