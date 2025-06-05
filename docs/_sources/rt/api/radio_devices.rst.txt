Radio Devices
=============

A radio device refers to a :class:`~sionna.rt.Transmitter` or :class:`~sionna.rt.Receiver` equipped
with an :class:`~sionna.rt.AntennaArray` as specified by the :class:`~sionna.rt.Scene`'s properties
:attr:`~sionna.rt.Scene.tx_array` and :attr:`~sionna.rt.Scene.rx_array`, respectively.

The following code snippet shows how to instantiate a :class:`~sionna.rt.Transmitter`
equipped with a :math:`4 \times 2` :class:`~sionna.rt.PlanarArray` with cross-polarized isotropic antennas:

.. code-block:: python

    from sionna.rt import load_scene, PlanarArray, Transmitter
    scene = load_scene()
    scene.tx_array = PlanarArray(num_rows=4,
                                num_cols=2,
                                pattern="iso",
                                polarization="cross")
    tx = Transmitter(name="tx",
                    position=(0,0,0),
                    power_dbm=22)
    scene.add(tx)

The position :math:`(x,y,z)` and orientation :math:`(\alpha, \beta, \gamma)` of a radio device
can be freely configured. The latter is specified through three angles corresponding to a 3D
rotation as defined in :eq:`rotation`.

.. code-block:: python

   from sionna.rt import load_scene, Transmitter
   scene = load_scene()
   scene.add(Transmitter(name="tx", position=(0,0,0)))
   tx = scene.get("tx")
   tx.position=(10,20,30)
   tx.orientation=(0.3,0,0.1)

Radio devices need to be explicitly added to the scene using the scene's method :meth:`~sionna.rt.Scene.add`
and can be removed from it using :meth:`~sionna.rt.Scene.remove`:

.. code-block:: python

   from sionna.rt import load_scene, Transmitter
   scene = load_scene()
   scene.add(Transmitter(name="tx", position=(0,0,0)))
   scene.remove("tx")

.. autoclass:: sionna.rt.RadioDevice
    :members:

.. autoclass:: sionna.rt.Receiver
    :members:

.. autoclass:: sionna.rt.Transmitter
    :members:
