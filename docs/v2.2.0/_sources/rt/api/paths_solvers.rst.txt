Path Solvers
============

A path solver computes the propagation :class:`~sionna.rt.Paths`
for a given :class:`~sionna.rt.Scene`.
This includes tracing the paths and computing the corresponding channel
coefficients, delays, and angles of departure and arrival.
Sionna provides a path solver (:class:`~sionna.rt.PathSolver`) which currently
supports specular reflections and diffuse reflections,
as well as refractions (or transmissions).

.. autoclass:: sionna.rt.PathSolver
   :members:
   :special-members: __call__
