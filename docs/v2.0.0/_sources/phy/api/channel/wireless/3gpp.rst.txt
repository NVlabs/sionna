3GPP 38.901
===========

The submodule ``tr38901`` implements 3GPP channel models from :cite:p:`TR38901`.

The :class:`~sionna.phy.channel.tr38901.CDL`, :class:`~sionna.phy.channel.tr38901.UMi`,
:class:`~sionna.phy.channel.tr38901.UMa`, and :class:`~sionna.phy.channel.tr38901.RMa`
models require setting-up antenna models for the transmitters and
receivers. This is achieved using the
:class:`~sionna.phy.channel.tr38901.PanelArray` class.

The :class:`~sionna.phy.channel.tr38901.UMi`,
:class:`~sionna.phy.channel.tr38901.UMa`, and :class:`~sionna.phy.channel.tr38901.RMa`
models require setting-up a network topology, specifying, e.g., the user terminals (UTs) and
base stations (BSs) locations, the UTs velocities, etc.
:doc:`Utility functions <utils>` are available to help laying out
complex topologies or to quickly setup simple but widely used topologies.


.. currentmodule:: sionna.phy.channel.tr38901

.. autosummary::
   :toctree: .

   PanelArray
   Antenna
   AntennaArray
   TDL
   CDL
   UMi
   UMa
   RMa

.. _tdl:
.. _cdl:
.. _umi:
.. _uma:
.. _rma:
.. _utility-functions:
.. _sionna.phy.channel.tr38901.CDL:
.. _sionna.phy.channel.tr38901.UMi:
.. _sionna.phy.channel.tr38901.UMa:
.. _sionna.phy.channel.tr38901.RMa:
.. _sionna.phy.channel.tr38901.PanelArray:
.. _sionna.phy.channel.tr38901.AntennaArray:
.. _sionna.phy.channel.tr38901.Antenna:
.. _sionna.phy.channel.tr38901.TDL:

See the class list above for the CDL, UMi, UMa, RMa, PanelArray, AntennaArray, Antenna, and TDL API documentation.

