Radio Maps
==========

A radio map describes a metric, such as path gain, received signal strength
(RSS), or signal-to-interference-plus-noise ratio (SINR) for a specific transmitter at every point on a measurement surface.
In other words, for a given transmitter, it associates every point on a measurement surface
with the channel gain, RSS, or SINR, that a receiver equipped with a dual-polarized isotropic antenna would observe at this point.
A radio map is not uniquely defined as it depends on the transmit array and antenna pattern, the transmitter orientation,
as well as the transmit precoding vector. Moreover, a radio map is not continuous but discrete because the measurement surface is quantized into small planar bins.

In Sionna, radio maps are generated using a :doc:`radio map solver <radio_map_solvers>`, which returns an instance of a specialized class derived from the abstract base class :class:`~sionna.rt.RadioMap`. The built-in radio map solver can compute a radio map for either of the following:

* A rectangular measurement plane grid, subdivided into equal-sized rectangular cells. In this case, an instance of :class:`~sionna.rt.PlanarRadioMap` is returned.
* A mesh, where each triangle of the mesh serves as a bin of the radio map. Here, an instance of :class:`~sionna.rt.MeshRadioMap` is returned.

Radio maps can be visualized by passing them as arguments to the functions :meth:`~sionna.rt.Scene.render`, :meth:`~sionna.rt.Scene.render_to_file`, or :meth:`~sionna.rt.Scene.preview`. Additionally, :class:`~sionna.rt.PlanarRadioMap` features a class method :meth:`~sionna.rt.PlanarRadioMap.show`.

A very useful feature is :meth:`~sionna.rt.RadioMap.sample_positions` which allows sampling
of random positions within the scene that have sufficient path gain, RSS, or SINR from a specific transmitter.

.. autoclass:: sionna.rt.PlanarRadioMap
    :members:
    :exclude-members: finalize, add, to_world

.. autoclass:: sionna.rt.MeshRadioMap
    :members:
    :exclude-members: finalize, add

.. autoclass:: sionna.rt.RadioMap
    :members:
    :exclude-members: finalize, add
