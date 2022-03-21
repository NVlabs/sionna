Convolutional Codes
###################

This module supports encoding and Viterbi decoding for convolutional codes.

The following code snippet shows how to setup a rate-1/2, constraint-length-3 encoder in two alternate ways and a corresponding Viterbi decoder.

Setting-up:

.. code-block:: Python

   encoder = ConvEncoder(rate=1/2, # rate of the desired code
                         constraint_length=3) # constraint length of the code
   or
   encoder = ConvEncoder(gen_poly=['101', '111']) # or polynomial can be input directly


   decoder = ViterbiDecoder(gen_poly=encoder.gen_poly, # polynomial used in encoder
                            method="soft_llr") # can be "soft" or "hard"


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


Convolutional Encoding
**********************

.. autoclass:: sionna.fec.conv.ConvEncoder
   :members:
   :exclude-members: call, build


Viterbi Decoding
****************

.. autoclass:: sionna.fec.conv.ViterbiDecoder
   :members:
   :exclude-members: call, build


Convolutional Code Utility Functions
************************************


Trellis
-------
.. autofunction:: sionna.fec.conv.utils.Trellis


polynomial_selector
-------------------
.. autofunction:: sionna.fec.conv.utils.polynomial_selector


References:
   .. [Moon] Todd. K. Moon, "Error correction coding: Mathematical
      methods and algorithms"
