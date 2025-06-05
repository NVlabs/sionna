Radio Map Solvers
====================

A radio map solver computes a :doc:`radio map <radio_maps>` for a given :class:`~sionna.rt.Scene` and
for every :class:`~sionna.rt.Transmitter`.
Sionna provides a radio map solver (:class:`~sionna.rt.RadioMapSolver`) which currently
supports specular reflection (including specular chains), diffuse reflection,
and refraction. It computes a path gain map, from which a received signal
strength (RSS) map or a signal to interference plus noise ratio (SINR) map can be computed.

.. autoclass:: sionna.rt.RadioMapSolver
   :members:
   :special-members: __call__
