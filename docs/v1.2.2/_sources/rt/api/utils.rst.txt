Utility Functions
=================

Complex-valued tensors
----------------------
.. autofunction:: sionna.rt.utils.cpx_abs
.. autofunction:: sionna.rt.utils.cpx_abs_square
.. autofunction:: sionna.rt.utils.cpx_add
.. autofunction:: sionna.rt.utils.cpx_convert
.. autofunction:: sionna.rt.utils.cpx_div
.. autofunction:: sionna.rt.utils.cpx_exp
.. autofunction:: sionna.rt.utils.cpx_mul
.. autofunction:: sionna.rt.utils.cpx_sqrt
.. autofunction:: sionna.rt.utils.cpx_sub


Electromagnetics
----------------
.. autofunction:: sionna.rt.utils.complex_relative_permittivity
.. autofunction:: sionna.rt.utils.fresnel
.. autofunction:: sionna.rt.utils.f_utd
.. autofunction:: sionna.rt.utils.fresnel_reflection_coefficients_simplified
.. autofunction:: sionna.rt.utils.itu_coefficients_single_layer_slab

Geometry
--------
.. autofunction:: sionna.rt.utils.phi_hat
.. autofunction:: sionna.rt.utils.theta_hat
.. autofunction:: sionna.rt.utils.r_hat
.. autofunction:: sionna.rt.utils.theta_phi_from_unit_vec
.. autofunction:: sionna.rt.utils.rotation_matrix

Jones calculus
--------------
.. autofunction:: sionna.rt.utils.implicit_basis_vector
.. autofunction:: sionna.rt.utils.jones_matrix_rotator
.. autofunction:: sionna.rt.utils.jones_matrix_rotator_flip_forward
.. autofunction:: sionna.rt.utils.to_world_jones_rotator
.. autofunction:: sionna.rt.utils.jones_matrix_to_world_implicit
.. autofunction:: sionna.rt.utils.jones_vec_dot

Meshes
-------
.. autofunction:: sionna.rt.utils.load_mesh
.. autofunction:: sionna.rt.utils.transform_mesh

Miscellaneous
-------------
.. autofunction:: sionna.rt.utils.complex_sqrt
.. autofunction:: sionna.rt.utils.dbm_to_watt
.. autofunction:: sionna.rt.utils.isclose
.. autofunction:: sionna.rt.utils.log10
.. autofunction:: sionna.rt.utils.sigmoid
.. autofunction:: sionna.rt.utils.sinc
.. autofunction:: sionna.rt.utils.subcarrier_frequencies
.. autofunction:: sionna.rt.utils.watt_to_dbm


Ray tracing
-----------
.. autofunction:: sionna.rt.utils.fibonacci_lattice
.. autofunction:: sionna.rt.utils.spawn_ray_from_sources
.. autofunction:: sionna.rt.utils.offset_p
.. autofunction:: sionna.rt.utils.spawn_ray_towards
.. autofunction:: sionna.rt.utils.spawn_ray_to


References:
   .. [ITU_R_2040_3] Recommendation ITU-R P.2040-3, "`Effects of building materials
        and structures on radiowave propagation above about 100 MHz
        <https://www.itu.int/rec/R-REC-P.2040-3-202308-I/en>`_"
   .. [ITU_R_P_526_15] Recommendation ITU-R P.526-15, "`Propagation by
    diffraction <https://www.itu.int/rec/R-REC-P.526-15-201910-I/en>`_"
   .. [TR38901] 3GPP TR 38.901, "`Study on channel model for frequencies from 0.5
    to 100 GHz <https://portal.3gpp.org/desktopmodules/Specifications/SpecificationDetails.aspx?specificationId=3173>`_", Release 18.0
