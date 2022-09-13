Utility Functions
#################

The utilities sub-package of the Sionna library contains many convenience
functions as well as extensions to existing TensorFlow functions.


Metrics
*******

BitErrorRate
------------
.. autoclass:: sionna.utils.BitErrorRate
   :exclude-members: result, reset_states, update_state

BitwiseMutualInformation
------------------------
.. autoclass:: sionna.utils.BitwiseMutualInformation
   :exclude-members: result, reset_states, update_state

compute_ber
-----------
.. autofunction:: sionna.utils.compute_ber

compute_bler
------------
.. autofunction:: sionna.utils.compute_bler

compute_ser
-----------
.. autofunction:: sionna.utils.compute_ser

count_errors
------------
.. autofunction:: sionna.utils.count_errors

count_block_errors
------------------
.. autofunction:: sionna.utils.count_block_errors



Tensors
*******
expand_to_rank
--------------
.. autofunction:: sionna.utils.expand_to_rank

flatten_dims
------------
.. autofunction:: sionna.utils.flatten_dims

flatten_last_dims
-----------------
.. autofunction:: sionna.utils.flatten_last_dims

insert_dims
-----------
.. autofunction:: sionna.utils.insert_dims

split_dims
----------
.. autofunction:: sionna.utils.split_dim

matrix_sqrt
-----------
.. autofunction:: sionna.utils.matrix_sqrt

matrix_sqrt_inv
---------------
.. autofunction:: sionna.utils.matrix_sqrt_inv

matrix_inv
----------
.. autofunction:: sionna.utils.matrix_inv

matrix_pinv
-----------
.. autofunction:: sionna.utils.matrix_pinv

Miscellaneous
*************

BinarySource
------------
.. autoclass:: sionna.utils.BinarySource

SymbolSource
------------
.. autoclass:: sionna.utils.SymbolSource

QAMSource
---------
.. autoclass:: sionna.utils.QAMSource

PAMSource
---------
.. autoclass:: sionna.utils.PAMSource

PlotBER
-------
.. autoclass:: sionna.utils.plotting.PlotBER
   :members:

sim_ber
-------
.. autofunction:: sionna.utils.sim_ber

ebnodb2no
---------
.. autofunction:: sionna.utils.ebnodb2no

hard_decisions
--------------
.. autofunction:: sionna.utils.hard_decisions

plot_ber
--------
.. autofunction:: sionna.utils.plotting.plot_ber

complex_normal
--------------
.. autofunction:: sionna.utils.complex_normal

log2
----
.. autofunction:: sionna.utils.log2

log10
-----
.. autofunction:: sionna.utils.log10
