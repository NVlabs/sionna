Utility Functions
=================

Linear Algebra
--------------
.. autofunction:: sionna.phy.utils.inv_cholesky

.. autofunction:: sionna.phy.utils.matrix_pinv

Metrics
-------

.. autofunction:: sionna.phy.utils.compute_ber

.. autofunction:: sionna.phy.utils.compute_bler

.. autofunction:: sionna.phy.utils.compute_ser

.. autofunction:: sionna.phy.utils.count_block_errors

.. autofunction:: sionna.phy.utils.count_errors

Miscellaneous
-------------

.. autofunction:: sionna.phy.utils.dbm_to_watt

.. autoclass:: sionna.phy.utils.db_to_lin
  :members:

.. autoclass:: sionna.phy.utils.DeepUpdateDict
  :members: deep_update

.. autofunction:: sionna.phy.utils.dict_keys_to_int

.. autofunction:: sionna.phy.utils.ebnodb2no

.. autofunction:: sionna.phy.utils.complex_normal

.. autofunction:: sionna.phy.utils.hard_decisions

.. autoclass:: sionna.phy.utils.Interpolate
   :members:

.. autoclass:: sionna.phy.utils.lin_to_db
   :members:

.. autofunction:: sionna.phy.utils.log2

.. autofunction:: sionna.phy.utils.log10

.. autoclass:: sionna.phy.utils.MCSDecoder
   :members:
   :exclude-members: call, build

.. autofunction:: sionna.phy.utils.scalar_to_shaped_tensor

.. autofunction:: sionna.phy.utils.sim_ber

.. autoclass:: sionna.phy.utils.SingleLinkChannel
   :members:
   :exclude-members: call, build

.. autoclass:: sionna.phy.utils.SplineGriddataInterpolation
   :members:

.. autofunction:: sionna.phy.utils.to_list

.. autoclass:: sionna.phy.utils.TransportBlock
   :members:
   :exclude-members: call, build

.. autofunction:: sionna.phy.utils.watt_to_dbm

Numerics
--------

.. autofunction:: sionna.phy.utils.bisection_method


Plotting
--------

.. autofunction:: sionna.phy.utils.plotting.plot_ber

.. autoclass:: sionna.phy.utils.plotting.PlotBER


Tensors
-------
.. autofunction:: sionna.phy.utils.expand_to_rank

.. autofunction:: sionna.phy.utils.flatten_dims

.. autofunction:: sionna.phy.utils.flatten_last_dims

.. autofunction:: sionna.phy.utils.insert_dims

.. autofunction:: sionna.phy.utils.split_dim

.. autofunction:: sionna.phy.utils.diag_part_axis

.. autofunction:: sionna.phy.utils.flatten_multi_index

.. autofunction:: sionna.phy.utils.gather_from_batched_indices

.. autofunction:: sionna.phy.utils.tensor_values_are_in_set

.. autofunction:: sionna.phy.utils.enumerate_indices