Antenna Patterns
================

We refer the user to the section ":ref:`far_field`" for various useful
definitions and background on antenna modeling. An
:class:`~sionna.rt.AntennaPattern` can be single- or
dual-polarized and might have for each polarization direction a possibly
different pattern. One can think of a dual-polarized pattern as two colocated
linearly polarized antennas.

Mathematically, an antenna pattern is defined as a function :math:`f:(\theta,\varphi)\mapsto (C_\theta(\theta, \varphi), C_\varphi(\theta, \varphi))` that maps a pair of zenith and azimuth angles to zenith and azimuth pattern values.

.. autoclass:: sionna.rt.AntennaPattern
    :members:
.. autoclass:: sionna.rt.PolarizedAntennaPattern
    :members:
    :inherited-members:

.. _v_pattern:

Vertically Polarized Antenna Pattern Functions
----------------------------------------------
.. autofunction:: sionna.rt.antenna_pattern.v_iso_pattern
.. autofunction:: sionna.rt.antenna_pattern.v_dipole_pattern
.. autofunction:: sionna.rt.antenna_pattern.v_hw_dipole_pattern
.. autofunction:: sionna.rt.antenna_pattern.v_tr38901_pattern


Polarization Models
-------------------
.. autofunction:: sionna.rt.antenna_pattern.polarization_model_tr38901_1
.. autofunction:: sionna.rt.antenna_pattern.polarization_model_tr38901_2


Utility Functions
-----------------
.. autofunction:: sionna.rt.antenna_pattern.antenna_pattern_to_world_implicit
.. autofunction:: sionna.rt.antenna_pattern.complex2real_antenna_pattern
.. autofunction:: sionna.rt.register_antenna_pattern
.. autofunction:: sionna.rt.register_polarization
.. autofunction:: sionna.rt.register_polarization_model

References:
   .. [Balanis97] A\. Balanis, "Antenna Theory: Analysis and Design," 2nd Edition, John Wiley & Sons, 1997.
   .. [TR38901] 3GPP TR 38.901, "`Study on channel model for frequencies from 0.5
    to 100 GHz <https://portal.3gpp.org/desktopmodules/Specifications/SpecificationDetails.aspx?specificationId=3173>`_", Release 18.0
