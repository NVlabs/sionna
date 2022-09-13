Interleaving
============

The interleaver module allows to permute tensors with either pseudo-random permutations or by row/column swapping.

To simplify distributed graph execution (e.g., by running interleaver and deinterleaver in a different sub-graph/device), the interleavers are implemented stateless. Thus, the internal seed cannot be updated on runtime and does not change after the initialization. However, if required, an explicit random seed can be passed as additional input to the interleaver/deinterleaver pair when calling the layer.

The following code snippet shows how to setup and use an instance of the interleaver:

.. code-block:: Python

   # set-up system
   interleaver = RandomInterleaver(seed=1234, # an explicit seed can be provided
                                   keep_batch_constant=False, # if True, all samples in the batch are permuted with the same pattern
                                   axis=-1) # axis which shall be permuted

   deinterleaver = Deinterleaver(interleaver=interleaver) # connect interleaver and deinterleaver

   # --- simplified usage with fixed seed ---
   # c has arbitrary shape (rank>=2)
   c_int = interleaver(c)
   # call deinterleaver to reconstruct the original order
   c_deint = deinterleaver(c_int)

   # --- advanced usage ---
   # provide explicit seed if a new random seed should be used for each call
   s = tf.random.uniform((), minval=0, maxval=12345678, dtype=tf.int32)

   c_int = interleaver([c, s])
   c_deint = deinterleaver([c_int, s])


Interleaver
***********

RowColumnInterleaver
--------------------
.. autoclass:: sionna.fec.interleaving.RowColumnInterleaver
   :members:
   :exclude-members: call, build

RandomInterleaver
-----------------
.. autoclass:: sionna.fec.interleaving.RandomInterleaver
   :members:
   :exclude-members: call, build

Turbo3GPPInterleaver
--------------------
.. autoclass:: sionna.fec.interleaving.Turbo3GPPInterleaver
   :members:
   :exclude-members: call, build


Deinterleaver
*************
.. autoclass:: sionna.fec.interleaving.Deinterleaver
   :members:
   :exclude-members: call, build


References:

   .. [3GPPTS36212_I] ETSI 3GPP TS 36.212 "Evolved Universal Terrestrial
      Radio Access (EUTRA); Multiplexing and channel coding", v.15.3.0, 2018-09.

