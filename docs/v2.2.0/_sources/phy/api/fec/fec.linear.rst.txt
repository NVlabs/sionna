Linear Codes
============

This package provides generic support for binary linear block codes.

.. currentmodule:: sionna.phy.fec.linear

.. autosummary::
   :toctree: .

   LinearEncoder
   OSDecoder

For encoding, a universal :class:`~sionna.phy.fec.linear.LinearEncoder` is available and can be initialized with either a generator or parity-check matrix. The matrix must be binary and of full rank.

For decoding, :class:`~sionna.phy.fec.linear.OSDecoder` implements the
ordered-statistics decoding (OSD) algorithm :cite:p:`Fossorier` which provides close to
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
