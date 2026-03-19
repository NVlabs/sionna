Low-Density Parity-Check (LDPC)
===============================

The low-density parity-check (LDPC) code module supports 5G compliant LDPC codes and allows iterative belief propagation (BP) decoding.
Further, the module supports rate-matching for 5G and provides a generic linear encoder.

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

.. autoclass:: sionna.phy.fec.ldpc.encoding.LDPC5GEncoder
   :members:
   :exclude-members: call, build

.. autoclass:: sionna.phy.fec.ldpc.decoding.LDPCBPDecoder
   :members:
   :exclude-members: call, build

.. autoclass:: sionna.phy.fec.ldpc.decoding.LDPC5GDecoder
   :members:
   :exclude-members: call, build

Node Update Functions
---------------------

.. autofunction:: sionna.phy.fec.ldpc.decoding.vn_update_sum

.. autofunction:: sionna.phy.fec.ldpc.decoding.cn_update_minsum

.. autofunction:: sionna.phy.fec.ldpc.decoding.cn_update_offset_minsum

.. autofunction:: sionna.phy.fec.ldpc.decoding.cn_update_phi

.. autofunction:: sionna.phy.fec.ldpc.decoding.cn_update_tanh


Decoder Callbacks
-----------------

The :class:`~sionna.phy.fec.ldpc.encoding.LDPCBPDecoder` and
:class:`~sionna.phy.fec.ldpc.decoding.LDPC5GDecoder` have the possibility to
register callbacks that are executed after each iteration. This allows to
customize the behavior of the decoder (for example to implement weighted BP
[Nachmani]_) or to track the decoding process.

A simple example to track the decoder statistics is given in the following example

.. code-block:: Python

   num_iter = 10

   # init decoder stats module
   dec_stats = DecoderStatisticsCallback(num_iter)

   encoder = LDPC5GEncoder(k = 100, # number of information bits (input)
                           n = 200) # number of codeword bits (output)

   decoder = LDPC5GDecoder(encoder           = encoder,
                           num_iter          = num_iter, # number of BP iterations
                           return_infobits   = True,
                           c2v_callbacks     = [dec_stats,]) # register stats callback

   source = GaussianPriorSource()

   # generate LLRs
   noise_var = 0.1
   batch_size = 1000
   llr_ch = source([batch_size, encoder.n], noise_var)

   # and run decoder (this can be also a loop)
   decoder(llr_ch)

   # and print statistics
   print("Avg. iterations:", dec_stats.avg_number_iterations.numpy())
   print("Success rate after n iterations:", dec_stats.success_rate.numpy())

   >> Avg. iterations: 5.404
   >> Success rate after n iterations: [0.258 0.235 0.637 0.638 0.638 0.638 0.638 0.638 0.638 0.638]



.. autofunction:: sionna.phy.fec.ldpc.utils.DecoderStatisticsCallback

.. autofunction:: sionna.phy.fec.ldpc.utils.EXITCallback

.. autofunction:: sionna.phy.fec.ldpc.utils.WeightedBPCallback

References:

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

   .. [Chen] J. Chen, et al. "Reduced-complexity Decoding of LDPC Codes."
      IEEE Transactions on Communications, vol. 53, no. 8, Aug. 2005.
