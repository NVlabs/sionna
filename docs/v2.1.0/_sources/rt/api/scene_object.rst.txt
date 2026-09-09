Scene Objects
=============

A scene is made of scene objects. Examples include cars, trees,
buildings, furniture, etc.
A scene object is characterized by its geometry and material (:class:`~sionna.rt.RadioMaterial`)
and implemented as an instance of the :class:`~sionna.rt.SceneObject` class.

Scene objects are uniquely identified by their name.
To access a scene object, the :meth:`~sionna.rt.Scene.get` method of
:class:`~sionna.rt.Scene` may be used.
For example, the following code snippet shows how to load a scene and list its scene objects:

.. code-block:: python

   scene = load_scene(sionna.rt.scene.munich)
   print(scene.objects)

To select an object, e.g., named `"Schrannenhalle-itu_metal"`, you can run:

.. code-block:: python

   my_object = scene.get("Schrannenhalle-itu_metal")

You can then set the :class:`~sionna.rt.RadioMaterial`
of ``my_object`` as follows:

.. code-block:: python

   my_object.radio_material = "itu_wood"

.. autoclass:: sionna.rt.SceneObject
   :members:
   :inherited-members:
