Decoder Callbacks
-----------------

.. currentmodule:: sionna.phy.fec.ldpc

.. autosummary::
   :toctree: .

   DecoderStatisticsCallback
   EXITCallback
   WeightedBPCallback


The :class:`~sionna.phy.fec.ldpc.encoding.LDPCBPDecoder` and
:class:`~sionna.phy.fec.ldpc.decoding.LDPC5GDecoder` have the possibility to
register callbacks that are executed after each iteration. This allows to
customize the behavior of the decoder (for example to implement weighted BP
:cite:p:`Nachmani`) or to track the decoding process.

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
