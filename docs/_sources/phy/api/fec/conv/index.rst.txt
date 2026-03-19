Convolutional Codes
-------------------

.. currentmodule:: sionna.phy.fec.conv

.. autosummary::
   :toctree: .

   ConvEncoder
   ViterbiDecoder
   BCJRDecoder


This module supports encoding of convolutional codes and provides layers for Viterbi :cite:p:`Viterbi` and BCJR :cite:p:`BCJR` decoding.

While the :class:`~sionna.phy.fec.conv.decoding.ViterbiDecoder` decoding algorithm produces maximum likelihood *sequence* estimates, the :class:`~sionna.phy.fec.conv.decoding.BCJRDecoder` produces the maximum a posterior (MAP) bit-estimates.

The following code snippet shows how to set up a rate-1/2, constraint-length-3 :class:`~sionna.phy.fec.conv.encoding.ConvEncoder` in two alternate ways and a corresponding :class:`~sionna.phy.fec.conv.decoding.ViterbiDecoder` or :class:`~sionna.phy.fec.conv.decoding.BCJRDecoder`. You can find further examples in the `Channel Coding Tutorial Notebook <../tutorials/5G_Channel_Coding_Polar_vs_LDPC_Codes.html>`_.

Setting-up:

.. code-block:: Python

   encoder = ConvEncoder(rate=1/2, # rate of the desired code
                         constraint_length=3) # constraint length of the code
   # or
   encoder = ConvEncoder(gen_poly=['101', '111']) # or polynomial can be used as input directly

   # --- Viterbi decoding ---
   decoder = ViterbiDecoder(gen_poly=encoder.gen_poly) # polynomial used in encoder
   # or just reference to the encoder
   decoder = ViterbiDecoder(encoder=encoder) # the code parameters are infered from the encoder

   # --- or BCJR decoding ---
   decoder = BCJRDecoder(gen_poly=encoder.gen_poly, algorithm="map") # polynomial used in encoder

   # or just reference to the encoder
   decoder = BCJRDecoder(encoder=encoder, algorithm="map") # the code parameters are infered from the encoder


Running the encoder / decoder:

.. code-block:: Python

   # --- encoder ---
   # u contains the information bits to be encoded and has shape [...,k].
   # c contains the convolutional encoded codewords and has shape [...,n].
   c = encoder(u)

   # --- decoder ---
   # y contains the de-mapped received codeword from channel and has shape [...,n].
   # u_hat contains the estimated information bits and has shape [...,k].
   u_hat = decoder(y)

.. toctree::
   :hidden:
   :maxdepth: 3

   utils
