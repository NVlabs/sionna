Power Control
=============

.. currentmodule:: sionna.sys

.. figure:: ../figures/power_control_api.png
   :align: center
   :width: 100%

Upon scheduling users to available resources, the transmission power must be
adjusted to ensure that users receive the desired
quality of service while minimizing interference to other users.

In the uplink, the user terminal typically aims at (partially) compensating for
the pathloss to reach a target received power at the base station.
In the downlink, the base station distributes the available power budget across
users according to a certain fairness criterion.

For an example of how to adjust transmission power in Sionna, refer
to the `Power Control notebook <../tutorials/notebooks/Power_Control.ipynb>`_.

.. autosummary::
   :toctree: .

   open_loop_uplink_power_control
   downlink_fair_power_control
