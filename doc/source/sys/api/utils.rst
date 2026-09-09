Utils
=====

.. currentmodule:: sionna.sys

Set of utility functions for Sionna SYS.

General Utilities
-----------------

.. autosummary::
   :toctree: .

   is_scheduled_in_slot
   get_pathloss
   spread_across_subcarriers

Metrics for 3GPP Calibration
----------------------------

The following utilities compute system-level metrics used for 3GPP TR 38.901
calibration :cite:p:`TR38901V160100`. They are needed to evaluate Phase 1
link-budget metrics such as coupling loss, geometry SIR, and geometry SINR,
and Phase 2 wideband interference metrics. These quantities are computed from
the large-scale path gains or coupling losses of all candidate serving and
interfering cells.

.. autosummary::
   :toctree: .

   coupling_loss_db
   received_power_dbm
   serving_indices
   geometry_sir_db
   geometry_sinr_db
   wideband_sir_db
