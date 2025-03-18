Radio Map
===========

A radio map describes a metric, such as path gain, received signal strength
(RSS), or signal-to-interference-plus-noise ratio (SINR) for a specific transmitter at every point on a plane.
In other words, for a given transmitter, it associates every point on a surface
with the channel gain, RSS, or SINR, that a receiver with
a specific orientation would observe at this point. A radio map is not uniquely defined as it depends on
the transmit and receive arrays and their respective antenna patterns, the transmitter and receiver orientations,
as well as transmit precoding and receive combining vectors. Moreover, a radio map is not continuous but discrete
because the plane is quantized into small rectangular bins.

In Sionna, radio maps are computed using a :doc:`radio map solver <radio_map_solvers>` which returns an instance of
:class:`~sionna.rt.RadioMap`. They can be visualized by providing them either as arguments to the functions
:meth:`~sionna.rt.Scene.render`, :meth:`~sionna.rt.Scene.render_to_file`, and :meth:`~sionna.rt.Scene.preview`,
or by using the class method :meth:`~sionna.rt.RadioMap.show`.

A very useful feature is :meth:`~sionna.rt.RadioMap.sample_positions` which allows sampling
of random positions within the scene that have sufficient path gain, RSS, or SINR from a specific transmitter.

.. autoclass:: sionna.rt.RadioMap
    :members:
    :exclude-members: finalize, measurement_plane, to_world, add, transmitter_radio_map