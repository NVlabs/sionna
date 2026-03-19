
.. _dev_understanding_paths:

Understanding the Paths Object
##############################
Sionna RT uses a `PathSolver
<https://nvlabs.github.io/sionna/rt/api/paths_solvers.html#sionna.rt.PathsSolver>`_
to compute propagation `Paths
<https://nvlabs.github.io/sionna/rt/api/paths.html>`_ between the antennas of
transmitters and receivers in a scene. The goal of this developer guide is to
explain the properties of the ``Paths`` object in detail. Let's start with a
short code snippet that computes propagation paths between a transmitter and receiver:

.. code-block:: python

    # Imports
    import sionna.rt
    from sionna.rt import load_scene, PlanarArray, Transmitter, Receiver, \
                          PathSolver

    # Load scene
    scene = load_scene(sionna.rt.scene.munich, merge_shapes=False)

    # Configure TX/RX antenna array
    scene.tx_array = PlanarArray(num_rows=2,
                                 num_cols=1,
                                 pattern="iso",
                                 polarization="V")
    scene.rx_array = scene.tx_array

    # Create TX/RX
    scene.add(Transmitter(name="tx", position=[8.5,21,27]))
    scene.add(Receiver(name="rx", position=[44,95,1.5]))

    # Compute propagation paths
    p_solver  = PathSolver()

    # without a synthetic array
    paths = p_solver(scene=scene, max_depth=3, synthetic_array=False)

    # with a synthetic array
    paths_syn = p_solver(scene=scene, max_depth=3, synthetic_array=True)

Depending on the boolean argument ``synthetic_array`` used during the call of the
path solver above, a source/target is either a transmit/receive antenna or a point
located at the center of an antenna array. In our example, there are two sources
and targets (one for each antenna of the array) if ``synthetic_array`` is
`False`. There is a single source and target (one for each radio device) if ``synthetic_array`` is
`True`. This can be seen from the properties of the paths objects as shown below:

.. code-block:: python

    # Show sources/targets
    print("Source coordinates: \n", paths.sources)
    print("Target coordinates: \n", paths.targets)

    # Show sources/targets with `synthetic_array`
    print("Source coordinates (synthetic array): \n", paths_syn.sources)
    print("Target coordinates (synthetic array): \n", paths_syn.targets)

::

    Source coordinates:
    [[8.5, 21, 27.0214],
    [8.5, 21, 26.9786]]
    Target coordinates:
    [[44, 95, 1.52141],
    [44, 95, 1.47859]]
    Source coordinates (synthetic array):
    [[8.5, 21, 27]]
    Target coordinates (synthetic array):
    [[44, 95, 1.5]]

Apart from the paths coefficients ``paths.a`` and delays ``paths.tau``, the paths instance stores a lot of
side information about the propagation paths, such as angles of arrival and
departure, Doppler shifts, interaction types, ids of intersected objects, and
coordinates of intersection points (i.e., vertices).

.. code-block:: python

    # Let us inspect a specific path in detail
    path_idx = 4 # Try out other values in the range [0, 14]

    # For a detailed overview of the dimensions of all properties, have a look at the API documentation
    print(f"\n--- Detailed results for path {path_idx} ---")
    print(f"Channel coefficient: {paths.a[0].numpy()[0,0,0,0,path_idx] + 1j*paths.a[1].numpy()[0,0,0,0,path_idx]}")
    print(f"Propagation delay: {paths.tau[0,0,0,0,path_idx].numpy()*1e6:.5f} us")
    print(f"Zenith angle of departure: {paths.theta_t.numpy()[0,0,0,0,path_idx]:.4f} rad")
    print(f"Azimuth angle of departure: {paths.phi_t.numpy()[0,0,0,0,path_idx]:.4f} rad")
    print(f"Zenith angle of arrival: {paths.theta_r.numpy()[0,0,0,0,path_idx]:.4f} rad")
    print(f"Azimuth angle of arrival: {paths.phi_r.numpy()[0,0,0,0,path_idx]:.4f} rad")
    print(f"Doppler shift: {paths.doppler.numpy()[0,0,0,0,path_idx]:.4f} Hz")

::

    --- Detailed results for path 4 ---
    Channel coefficient: (-1.0185684914176818e-05-9.316545401816256e-06j)
    Propagation delay: 0.60660 us
    Zenith angle of departure: 1.7115 rad
    Azimuth angle of departure: 0.1612 rad
    Zenith angle of arrival: 1.4110 rad
    Azimuth angle of arrival: -0.9712 rad
    Doppler shift: 0.0000 Hz

.. code-block:: python

    # Show the interactions undergone by all paths:
    # 0 - No interaction, 1 - Specular reflection, 2 - Diffuse reflection, 4 - Refraction
    # Note that diffuse reflections are turned off by default.
    # Shape [max_depth, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths]
    print("Interactions: \n", paths.interactions.numpy()[:,0,0,0,0,:])

    print("Number of paths: ", paths.interactions.shape[-1])

::

    Interactions:
    [[4 1 0 1 1 1 1 1 4 4 1 1 1 1 1]
    [1 0 0 1 0 1 1 0 1 1 0 1 1 0 1]
    [4 0 0 1 0 0 0 0 4 4 0 0 1 0 0]]
    Number of paths:  15

We can see that there are in total 15 propagation paths (number of columns) with a maximum number of three interactions (number of rows). There is for example a paths consisting of a refraction (4), followed by a specular reflection (1), and another refraction (4). The line-of-sight (LoS) path has no interactions with the scene, i.e., [0,0,0].

The coordinates for every interaction as well as the corresponding object ids
can be extracted in the following way:

.. code-block:: python

    # Object ids for the selected path
    print("Object IDs: \n", paths.objects.numpy()[:,0,0,0,0,path_idx])

    # Coordinates of interaction points of the selected path
    print("Vertices: \n", paths.vertices.numpy()[:,0,0,0,0,path_idx])

::

    Object IDs:
    [      2364 4294967295 4294967295]
    Vertices:
    [[42.107708 91.055534  0.      ]
    [ 0.        0.        0.      ]
    [ 0.        0.        0.      ]

Note that the second and third object ids equal 4294967295, indicating an
`invalid shape
<https://nvlabs.github.io/sionna/rt/api/paths.html#sionna.rt.constants.INVALID_SHAPE>`_.
This happens because the path under consideration has only a depth of one,
consisting of a single specular reflection before the receiver is reached.

We can recover a `SceneObject <https://nvlabs.github.io/sionna/rt/api/scene_object.html>`_ from its id in the following way:

.. code-block:: python

    for obj in scene.objects.values():
        if obj.object_id == paths.objects.numpy()[0,0,0,0,0,0]:
            break

A scene object enriches a `Mitsuba shape <https://mitsuba.readthedocs.io/en/stable/src/api_reference.html#mitsuba.Shape>`_ with additional properties.
However, all of the currently implemented algorithms assume that a scene object
is constructed from a `Mitsuba mesh <https://mitsuba.readthedocs.io/en/stable/src/api_reference.html#mitsuba.Mesh>`_
(which inherits from Mitsuba shape). A mesh is defined by a set of triangles
(also called faces or primitives), which is not the case for a shape, which could be, e.g., defined as a sphere of a
certain radius.

We can access the shape of a scene object via the property
``SceneObject.mi_shape``:

.. code-block:: python

    print(obj.mi_shape)

::


    PLYMesh[
     name = "element_231-itu_marble.ply",
     bbox = BoundingBox3f[
        min = [41.2902, 113.299, 0],
        max = [109.21, 145.203, 17.7352]
     ],
     vertex_count = 30,
     vertices = [360 B of vertex data],
     face_count = 30,
     faces = [360 B of face data],
     face_normals = 1
    ]

The ``Paths.primitives`` property provides the ids of the faces of the ``mi_shape``
of scene object that paths intersect. One can recover the normal vectors of the
primitives in the following way:

.. code-block:: python

    obj.mi_shape.face_normal(paths.primitives.numpy()[:,0,0,0,0,path_idx])

::

    [[-0.316442, -0.948612, 0],
    [nan, nan, nan],
    [nan, nan, nan]]

In our example, the path only intersects a single object and there is hence also
only a single normal vector of interest.

In some cases, there is a different number of valid paths between different
pairs of sources and targets.
An example would be a link between two antenna arrays that is partially occluded,
so that only some antennas have a line-of-sight connection.
Which paths are valid can be seen from the property ```Paths.valid```:

.. code-block:: python

    paths.valid

::

    [[[[[True, True, True, True, True, True, True, True, True, True, True, True, True, True, True],
        [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]]],
    [[[True, True, True, True, True, True, True, True, True, True, True, True, True, True, True],
        [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]]]]]

This tensor can be useful to mask invalid paths in computations.

The path instance can be used to compute the channel frequency response
`Paths.cfr() <https://nvlabs.github.io/sionna/rt/api/paths.html#sionna.rt.Paths.cfr>`_, the baseband equivalent response `Paths.cir() <https://nvlabs.github.io/sionna/rt/api/paths.html#sionna.rt.Paths.cir>`_, or the discrete complex-baseband
equivalent response `Paths.taps() <https://nvlabs.github.io/sionna/rt/api/paths.html#sionna.rt.Paths.taps>`_. How these functions are used is described
in detail in the tutorial `Introduction to Sionna RT <https://nvlabs.github.io/sionna/rt/tutorials/introduction.html>`_.
