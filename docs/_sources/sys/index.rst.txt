System Level (SYS)
==================

This package provides differentiable system-level simulation functionalities for multi-cell networks.

It is based on a :doc:`physical layer abstraction <api/abstraction>` that
computes the block error rate (BLER) from the
:class:`~sionna.phy.ofdm.PostEqualizationSINR`. It further includes Layer-2
functionalities, such as
:doc:`link adaption (LA)<api/link_adaptation>` for adaptive modulation and coding scheme
(MCS) selection, downlink and uplink :doc:`power control<api/power_control>`, 
and :doc:`user scheduling<api/scheduling>`.
Base stations can be placed on a :doc:`spiral hexagonal<api/topology>` grid,
where wraparound is used for pathloss computation.

.. figure:: figures/sionna_sys.png
   :align: center
   :width: 100%

A good starting point for Sionna SYS is the available
:doc:`tutorials <tutorials>` page.

.. toctree::
   :hidden:
   :maxdepth: 6

   tutorials
   api/sys.rst
