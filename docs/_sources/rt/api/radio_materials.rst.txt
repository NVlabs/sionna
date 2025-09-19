Radio Materials
================

A radio material defines how an object scatters incident radio waves.
It implements all necessary components to simulate the interaction between
radio waves and objects composed of specific materials.

The base class :class:`~sionna.rt.RadioMaterialBase` provides an interface
for implementing arbitrary radio materials and can be used to implemented
custom materials, as detailed in the :ref:`Developer Guide <dev_custom_radio_materials>`.

The :class:`~sionna.rt.RadioMaterial` class implements the model described in the
`Primer on Electromagnetics <../em_primer.html>`_, and supports specular and diffuse
reflection as well as refraction. The details of this model are provided thereafter.

Similarly to scene objects (:class:`~sionna.rt.SceneObject`), all radio
materials are uniquely identified by their name.
For example, specifying that a scene object named `"wall"` is made of the
material named `"itu-brick"` is done as follows:

.. code-block:: python

   obj = scene.get("wall") # obj is a SceneObject
   obj.radio_material = "itu-brick" # "wall" is made of "itu-brick"

**Radio materials provided with Sionna**

The class :class:`~sionna.rt.RadioMaterial` implements the model described in the
`Primer on Electromagnetics <../em_primer.html>`_. Following this model, a radio material consists of the
real-valued relative permittivity :math:`\varepsilon_r`, the conductivity :math:`\sigma`,
and the relative  permeability :math:`\mu_r`.
For more details, see :eq:`epsilon`, :eq:`mu`, :eq:`eta`.
These quantities can possibly depend on the frequency of the incident radio
wave. Note that this model only allows non-magnetic materials with :math:`\mu_r=1`.

A radio material has also a thickness :math:`d` associated with it which impacts the computation of
the reflected and refracted fields (see :eq:`fresnel_slab`). For example, a wall
is modeled in Sionna RT as a planar surface whose radio material describes the
desired thickness.

Additionally, a :class:`~sionna.rt.RadioMaterial` can have an effective roughness (ER)
associated with it, leading to diffuse reflections (see, e.g., [Degli-Esposti11]_).
The ER model requires a scattering coefficient :math:`S\in[0,1]` :eq:`scattering_coefficient`,
a cross-polarization discrimination coefficient :math:`K_x` :eq:`xpd`, as well as a scattering pattern
:math:`f_\text{s}(\hat{\mathbf{k}}_\text{i}, \hat{\mathbf{k}}_\text{s})` :eq:`lambertian_model`--:eq:`backscattering_model`, such as the
:class:`~sionna.rt.LambertianPattern` or :class:`~sionna.rt.DirectivePattern`. The meaning of
these parameters is explained in `Scattering <../em_primer.html#scattering>`_.

Sionna provides the
:ref:`ITU models of several materials <provided-materials>` whose properties
are automatically updated according to the configured :attr:`~sionna.rt.Scene.frequency`.

.. _provided-materials:

Through the :class:`~sionna.rt.ITURadioMaterial` class, Sionna provides the models of all of the materials
defined in the ITU-R P.2040-3 recommendation [ITU_R_2040_3]_. These models are based on curve fitting to
measurement results and assume non-ionized and non-magnetic materials
(:math:`\mu_r = 1`).
Frequency dependence is modeled by

.. math::
   \begin{align}
      \varepsilon_r &= a f_{\text{GHz}}^b\\
      \sigma &= c f_{\text{GHz}}^d
   \end{align}

where :math:`f_{\text{GHz}}` is the frequency in GHz, and the constants
:math:`a`, :math:`b`, :math:`c`, and :math:`d` characterize the material.
The table below provides their values which are used in Sionna
(from [ITU_R_2040_3]_).
Note that the relative permittivity :math:`\varepsilon_r` and
conductivity :math:`\sigma` of all materials are updated automatically when
the frequency is set through the scene's property :class:`~sionna.rt.Scene.frequency`.
Moreover, by default, the scattering coefficient, :math:`S`, of these materials is set to
0, leading to no diffuse reflection.

+---------------------------+------------------------------------+--------------------------+-----------------------+
| Material type             | Real part of relative permittivity | Conductivity [S/m]       | Frequency range (GHz) |
+                           +-------------------+----------------+---------------+----------+                       +
|                           | a                 | b              | c             | d        |                       |
+---------------------------+-------------------+----------------+---------------+----------+-----------------------+
| vacuum                    | 1                 | 0              | 0             | 0        | 0.001 -- 100          |
+---------------------------+-------------------+----------------+---------------+----------+-----------------------+
| concrete                  | 5.24              | 0              | 0.0462        | 0.7822   | 1 -- 100              |
+---------------------------+-------------------+----------------+---------------+----------+-----------------------+
| brick                     | 3.91              | 0              | 0.0238        | 0.16     | 1 -- 40               |
+---------------------------+-------------------+----------------+---------------+----------+-----------------------+
| plasterboard              | 2.73              | 0              | 0.0085        | 0.9395   | 1 -- 100              |
+---------------------------+-------------------+----------------+---------------+----------+-----------------------+
| wood                      | 1.99              | 0              | 0.0047        | 1.0718   | 0.001 -- 100          |
+---------------------------+-------------------+----------------+---------------+----------+-----------------------+
| glass                     | 6.31              | 0              | 0.0036        | 1.3394   | 0.1 -- 100            |
+                           +-------------------+----------------+---------------+----------+-----------------------+
|                           | 5.79              | 0              | 0.0004        | 1.658    | 220 -- 450            |
+---------------------------+-------------------+----------------+---------------+----------+-----------------------+
| ceiling_board             | 1.48              | 0              | 0.0011        | 1.0750   | 1 -- 100              |
+                           +-------------------+----------------+---------------+----------+-----------------------+
|                           | 1.52              | 0              | 0.0029        | 1.029    | 220 -- 450            |
+---------------------------+-------------------+----------------+---------------+----------+-----------------------+
| chipboard                 | 2.58              | 0              | 0.0217        | 0.7800   | 1 -- 100              |
+---------------------------+-------------------+----------------+---------------+----------+-----------------------+
| plywood                   | 2.71              | 0              | 0.33          | 0        | 1 -- 40               |
+---------------------------+-------------------+----------------+---------------+----------+-----------------------+
| marble                    | 7.074             | 0              | 0.0055        | 0.9262   | 1 -- 60               |
+---------------------------+-------------------+----------------+---------------+----------+-----------------------+
| floorboard                | 3.66              | 0              | 0.0044        | 1.3515   | 50 -- 100             |
+---------------------------+-------------------+----------------+---------------+----------+-----------------------+
| metal                     | 1                 | 0              | :math:`10^7`  | 0        | 1 -- 100              |
+---------------------------+-------------------+----------------+---------------+----------+-----------------------+
| very_dry_ground           | 3                 | 0              | 0.00015       | 2.52     | 1 -- 10               |
+---------------------------+-------------------+----------------+---------------+----------+-----------------------+
| medium_dry_ground         | 15                | -0.1           | 0.035         | 1.63     | 1 -- 10               |
+---------------------------+-------------------+----------------+---------------+----------+-----------------------+
| wet_ground                | 30                | -0.4           | 0.15          | 1.30     | 1 -- 10               |
+---------------------------+-------------------+----------------+---------------+----------+-----------------------+


.. autoclass:: sionna.rt.RadioMaterialBase
    :members:
    :exclude-members: is_used, add_object, remove_object

.. autoclass:: sionna.rt.RadioMaterial
    :members:
    :exclude-members: traverse, to_string, scene, sample, eval, pdf

.. autoclass:: sionna.rt.ITURadioMaterial
    :members:


Scattering Patterns
-------------------

.. autoclass:: sionna.rt.ScatteringPattern
    :members:
    :special-members: __call__

.. autoclass:: sionna.rt.LambertianPattern
    :members: show
    :special-members: __call__

.. autoclass:: sionna.rt.DirectivePattern
    :members: show, alpha_r
    :special-members: __call__

.. autoclass:: sionna.rt.BackscatteringPattern
    :members: show, alpha_r, alpha_i, lambda_
    :special-members: __call__

.. autofunction:: sionna.rt.register_scattering_pattern

References:
    .. [Degli-Esposti11] V\. Degli-Esposti et al., "`Analysis and Modeling on co- and Cross-Polarized Urban Radio Propagation for Dual-Polarized MIMO Wireless Systems <https://ieeexplore.ieee.org/abstract/document/5979177>`_", IEEE Trans. Antennas Propag, vol. 59, no. 11,  pp.4247-4256, Nov. 2011.
    .. [ITU_R_2040_3] Recommendation ITU-R P.2040-3, "`Effects of building materials and structures on radiowave propagation above about 100 MHz <https://www.itu.int/rec/R-REC-P.2040-3-202308-I/en>`_"
