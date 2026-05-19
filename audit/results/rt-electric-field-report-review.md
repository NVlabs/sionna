# RT Electric Field Report Review

Status: started.

## Scope

This workstream reviews the Sionna RT technical report itself before using it as
ground truth for implementation comparison. The highest-priority topic is the
correctness of electric-field computation.

## Review Order

1. Review report equations, sign conventions, coordinate conventions, units, and
   assumptions.
2. Record report-level technical errors, ambiguities, or missing definitions.
3. Use the reviewed report as ground truth for implementation comparison.
4. Map each electric-field equation/convention to source code and tests.

## Report Topics To Check

- [ ] Coordinate systems, spherical angles, and polarization bases.
- [ ] Electric-field initialization from transmitter antenna patterns.
- [ ] Field transport along ray segments, including phase and path loss.
- [ ] Reflection Jones matrices and basis transforms.
- [ ] Refraction/transmission Jones matrices and slab/thickness model.
- [ ] Diffuse scattering model and field normalization.
- [ ] Diffraction model, Keller cone geometry, diffraction coefficient, and
      lit-region handling.
- [ ] Synthetic-array phase shifts and approximation domain.
- [ ] Receiver antenna projection and final path coefficient definition.
- [ ] Doppler phase evolution and sign conventions.
- [ ] Radio-map field/path-gain estimator and non-coherent assumptions.
- [ ] Units: wavelength, impedance, power, field strength, RSS, path gain.

## Implementation Files To Map

- [ ] `ext/sionna-rt/src/sionna/rt/path_solvers/field_calculator.py`
- [ ] `ext/sionna-rt/src/sionna/rt/path_solvers/paths.py`
- [ ] `ext/sionna-rt/src/sionna/rt/path_solvers/paths_buffer.py`
- [ ] `ext/sionna-rt/src/sionna/rt/path_solvers/path_solver.py`
- [ ] `ext/sionna-rt/src/sionna/rt/radio_map_solvers/radio_map_solver.py`
- [ ] `ext/sionna-rt/src/sionna/rt/utils/electromagnetics.py`
- [ ] `ext/sionna-rt/src/sionna/rt/utils/jones.py`
- [ ] `ext/sionna-rt/src/sionna/rt/utils/geometry.py`
- [ ] `ext/sionna-rt/src/sionna/rt/utils/ray_tracing.py`
- [ ] `ext/sionna-rt/src/sionna/rt/utils/wedges.py`
- [ ] `ext/sionna-rt/src/sionna/rt/antenna_pattern.py`
- [ ] `ext/sionna-rt/src/sionna/rt/antenna_array.py`
- [ ] `ext/sionna-rt/src/sionna/rt/radio_materials/radio_material.py`
- [ ] `ext/sionna-rt/src/sionna/rt/radio_materials/scattering_pattern.py`

## Confirmed Report Issues

None recorded yet.

## Ambiguities To Resolve

None recorded yet.

## Code Mapping Notes

Not started.

