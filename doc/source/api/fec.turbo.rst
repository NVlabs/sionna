Turbo Codes
###########

This module supports encoding and decoding of Turbo codes [Berrou]_, e.g., as
used in the LTE wireless standard. The convolutional component encoders and
decoders are composed of the :class:`~sionna.fec.conv.encoding.ConvEncoder` and
:class:`~sionna.fec.conv.decoding.BCJRDecoder` layers, respectively.

Please note that various notations are used in literature to represent the
generator polynomials for the underlying convolutional codes. For simplicity,
:class:`~sionna.fec.turbo.encoding.TurboEncoder` only accepts the binary
format, i.e., `10011`, for the generator polynomial which corresponds to the
polynomial :math:`1 + D^3 + D^4`.

The following code snippet shows how to set-up a rate-1/3, constraint-length-4 :class:`~sionna.fec.turbo.encoding.TurboEncoder` and the corresponding :class:`~sionna.fec.turbo.decoding.TurboDecoder`.
You can find further examples in the `Channel Coding Tutorial Notebook <../examples/5G_Channel_Coding_Polar_vs_LDPC_Codes.html>`_.

Setting-up:

.. code-block:: Python

   encoder = TurboEncoder(constraint_length=4, # Desired constraint length of the polynomials
                          rate=1/3,  # Desired rate of Turbo code
                          terminate=True) # Terminate the constituent convolutional encoders to all-zero state
   # or
   encoder = TurboEncoder(gen_poly=gen_poly, # Generator polynomials to use in the underlying convolutional encoders
                          rate=1/3, # Rate of the desired Turbo code
                          terminate=False) # Do not terminate the constituent convolutional encoders

   # the decoder can be initialized with a reference to the encoder
   decoder = TurboDecoder(encoder,
                          num_iter=6, # Number of iterations between component BCJR decoders
                          algorithm="map", # can be also "maxlog"
                          hard_out=True) # hard_decide output

Running the encoder / decoder:

.. code-block:: Python

   # --- encoder ---
   # u contains the information bits to be encoded and has shape [...,k].
   # c contains the turbo encoded codewords and has shape [...,n], where n=k/rate when terminate is False.
   c = encoder(u)

   # --- decoder ---
   # llr contains the log-likelihood ratio values from the de-mapper and has shape [...,n].
   # u_hat contains the estimated information bits and has shape [...,k].
   u_hat = decoder(llr)


Turbo Encoding
**************

.. autoclass:: sionna.fec.turbo.TurboEncoder
   :members:
   :exclude-members: call, build


Turbo Decoding
**************

.. autoclass:: sionna.fec.turbo.TurboDecoder
   :members:
   :exclude-members: call, build

Turbo Utility Functions
***********************

TurboTermination
----------------
.. autoclass:: sionna.fec.turbo.TurboTermination
   :members:
   :exclude-members: call, build

polynomial_selector
-------------------
.. autofunction:: sionna.fec.turbo.utils.polynomial_selector

puncture_pattern
----------------
.. autofunction:: sionna.fec.turbo.utils.puncture_pattern

References:
   .. [Berrou] C\. Berrou, A\. Glavieux, P\. Thitimajshima, "Near Shannon limit error-correcting coding and decoding: Turbo-codes", IEEE ICC, 1993.

   .. [3GPPTS36212_Turbo] ETSI 3GPP TS 36.212 "Evolved Universal Terrestrial
      Radio Access (EUTRA); Multiplexing and channel coding", v.15.3.0, 2018-09.

