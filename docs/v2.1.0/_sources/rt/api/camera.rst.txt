Cameras
=======

A :class:`~sionna.rt.Camera` defines a position and view direction
for rendering the scene.

.. figure:: ../figures/camera.png
   :align: center

A new camera can be instantiated as follows:

.. code-block:: python

   scene = load_scene(rt.scene.munich)
   cam = Camera(position=(200., 0.0, 50.))
   cam.look_at((0.0,0.0,0.0))
   scene.render(cam)

.. figure:: ../figures/camera_example.png
   :align: center

.. autoclass:: sionna.rt.Camera
    :members:
    :exclude-members: world_to_angles, world_to_position, world_transform
