(Binary) Linear Codes
=====================

.. currentmodule:: sionna.phy.fec.utils
.. autosummary::
   :toctree: .

   load_parity_check_examples
   alist2mat
   load_alist
   generate_reg_ldpc
   make_systematic
   gm2pcm
   pcm2gm
   verify_gm_pcm

Several functions are provided to convert parity-check matrices into generator
matrices and vice versa. Please note that currently only binary codes are
supported.

.. code-block:: Python

   # load example parity-check matrix
   pcm, k, n, coderate = load_parity_check_examples(pcm_id=3)

Note that many research projects provide their parity-check matrices in the  `alist` format :cite:p:`MacKay` (e.g., see :cite:p:`UniKL`). The follwing code snippet provides an example of how to import an external LDPC parity-check matrix from an `alist` file and how to set-up an encoder/decoder.

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
   y = channel(x, no)
   llr = demapper(y, no)
   c_hat = decoder(llr)
