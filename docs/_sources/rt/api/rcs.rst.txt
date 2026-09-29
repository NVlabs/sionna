Radar Cross-Section (RCS)
=========================

Sensing targets (:class:`~sionna.rt.rcs.SensingTarget`) are scene objects
that are modelled by their :class:`~sionna.rt.rcs.ScatteringModel`
rather than by their mesh and radio material. Such sensing targets are
typically used to model objects that are sensed in a radar or integrated sensing and communication (ISAC) scenario,
e.g., a car, a drone, or a pedestrian.

A :class:`~sionna.rt.rcs.ScatteringModel` consists of a collection of scattering
points, provided in the local coordinate system of the sensing target. Every scattering point carries a radar cross-section (RCS) and a
cross-polarization matrix (CPM), both implemented as callables that are evaluated for a
pair of incident and scattered directions. The RCS gives the cross-section
:math:`\sigma` [:math:`\text{m}^2`] of the point scatters, whereas the CPM gives
the complex :math:`2 \times 2` matrix :math:`\mathbf{W}` describing how the
point transforms the polarization of the incident field. Together, they define
the Jones matrix :math:`\mathbf{J} = \sqrt{\sigma}\mathbf{W}` of the transfer
function, so both are required for every scattering point. 

Custom RCS and CPM callables can be used directly, or registered under a name with
:func:`~sionna.rt.rcs.register_rcs` and :func:`~sionna.rt.rcs.register_cpm`
so that they can be referred to by that name. Sionna provides
:class:`~sionna.rt.rcs.ConstantRCS` and :class:`~sionna.rt.rcs.ConstantCPM`,
which do not depend on the incident and scattered directions, as well as the
models of the 3GPP specifications described in `3GPP TR 38.901 Models`_. A
:class:`~sionna.rt.rcs.ConstantCPM` without a cross-polarization ratio
describes a scattering point which does not depolarize.

The :class:`~sionna.rt.rcs.RCSSolver` computes the propagation
paths that connect the transmitters to the receivers of a scene through the
scattering points of its sensing targets. Paths that do not interact with a
sensing target are not computed by this solver; use
:class:`~sionna.rt.PathSolver` to compute them.

The sensing targets are traced as absorbing objects, so that a target casts a
radio shadow on the other targets and on the legs of the computed paths. The
only exception is that a target does not occlude the scattering points it
contains, i.e., the ones which lie within its bounding box, as the scattering
response of a target is entirely described by its scattering model: a leg
which starts from such a point is only tested for occlusion once it has left
that bounding box. A scattering point placed outside of that box is occluded
by its own target as it is by any other object of the scene.

The following code snippet shows how to add a sensing target with a single
scattering point to a scene, and how to compute the paths scattered by it:

.. code-block:: python

    import mitsuba as mi
    import sionna
    from sionna.rt import load_scene, PlanarArray, Receiver, Transmitter
    from sionna.rt.rcs import (ConstantCPM, ConstantRCS, RCSSolver,
                               ScatteringModel, SensingTarget)

    scene = load_scene(sionna.rt.scene.simple_street_canyon)

    scene.tx_array = PlanarArray(num_rows=1, num_cols=1, pattern="iso",
                                 polarization="V")
    scene.rx_array = scene.tx_array
    scene.add(Transmitter(name="tx", position=[-32,-9,25]))
    scene.add(Receiver(name="rx", position=[-32,11,31]))

    # Sensing target shaped as a cuboid, with a single scattering point 2m
    # below it whose cross-section is 1 square meter and which does not
    # depolarize
    model = ScatteringModel([0,0,-2], rcs=ConstantRCS(sigma=1.),
                            cpm=ConstantCPM())
    target = SensingTarget(name="st", scattering_model=model,
                           length=4., width=2., height=1.5,
                           position=[-16,-10,60])
    scene.add(target)
    solver = RCSSolver()
    paths = solver(scene)

.. autoclass:: sionna.rt.rcs.RCSSolver
    :members:
    :special-members: __call__


Sensing Targets
---------------

.. autoclass:: sionna.rt.rcs.SensingTarget
    :members:

.. autoclass:: sionna.rt.rcs.ConstantRCSSensingTarget
    :members:

.. autoclass:: sionna.rt.rcs.ScatteringModel
    :members:

.. autoclass:: sionna.rt.rcs.ScatteringPoints
    :members:


Radar Cross-Sections
--------------------

.. autoclass:: sionna.rt.rcs.ConstantRCS
    :members:
    :special-members: __call__

.. autofunction:: sionna.rt.rcs.register_rcs

.. autofunction:: sionna.rt.rcs.unregister_rcs

.. autofunction:: sionna.rt.rcs.get_rcs


Cross-Polarization Matrices
---------------------------

.. autoclass:: sionna.rt.rcs.ConstantCPM
    :members:
    :special-members: __call__

.. autofunction:: sionna.rt.rcs.register_cpm

.. autofunction:: sionna.rt.rcs.unregister_cpm

.. autofunction:: sionna.rt.rcs.get_cpm


3GPP TR 38.901 Models
---------------------

Sionna implements the sensing target models specified in :cite:`TR38901_RT`,
clause 7.9.2. The type of target (``object_type``) and the RCS model
(``model_type``) determine both the number of scattering points and the
parameters of every point, which are taken from Tables 7.9.2.1-1 to 7.9.2.1-7.
The supported combinations are listed in the documentation of
:class:`~sionna.rt.rcs.TR38901ScatteringModel`.

Every scattering point depolarizes according to the cross-polarization matrix
of clause 7.9.2.2, whose cross-polarized entries are set by the XPR of
Table 7.9.2.2-1. That matrix is taken as eq. 7.9.2-5 defines it, i.e., with
co-polarized entries of unit modulus rather than normalized, so the tabulated
cross-section is the co-polarized one and the depolarized power adds to it.

The models have three random components, all of which are disabled by default,
so that a scattering point evaluates deterministically unless they are enabled:

.. list-table::
    :header-rows: 1

    * - Flag
      - Random component
    * - ``random_sigma_s``
      - The log-normal RCS component :math:`\sigma_S` of clause 7.9.2.1, which
        is otherwise fixed to 1, exactly its linear mean per eq. 7.9.2-1
    * - ``random_xpr``
      - The XPR of eq. 7.9.2-5, which is otherwise fixed to its tabulated mean
        in dB
    * - ``random_phases``
      - The four initial phases of eq. 7.9.2-5, which are otherwise zero, so
        that a scattering point does not randomize the phase of a path

Two runs of the solver on the same
scene with the same seed therefore give every path the same draws, and
exchanging the two directions leaves the cross-section unchanged and transposes
the cross-polarization matrix, as clause 7.9.4 requires for monostatic sensing.
:class:`~sionna.rt.rcs.TR38901RCS` and :class:`~sionna.rt.rcs.TR38901CPM`
detail the draws and the limits of their reproducibility.

:class:`~sionna.rt.rcs.TR38901SensingTarget` is the entry point of these
models: it builds its own :class:`~sionna.rt.rcs.TR38901ScatteringModel`,
which in turn equips every scattering point with a
:class:`~sionna.rt.rcs.TR38901RCS` and a
:class:`~sionna.rt.rcs.TR38901CPM` callable, and carries the three flags. A
target following the specifications is therefore created as follows:

.. code-block:: python

    from sionna.rt.rcs import TR38901SensingTarget

    target = TR38901SensingTarget(name="st", object_type="vehicle-multi-sp")

    # Draw the random components of the specifications
    target.scattering_model.random_sigma_s = True
    target.scattering_model.random_xpr = True
    target.scattering_model.random_phases = True

.. autoclass:: sionna.rt.rcs.TR38901SensingTarget
    :members:

.. autoclass:: sionna.rt.rcs.TR38901ScatteringModel
    :members:

.. autoclass:: sionna.rt.rcs.TR38901RCS
    :members:
    :special-members: __call__

.. autoclass:: sionna.rt.rcs.TR38901CPM
    :members:
    :special-members: __call__
