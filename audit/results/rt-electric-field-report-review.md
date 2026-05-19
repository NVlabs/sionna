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

## Candidate Report Issues To Verify

These items are not yet final findings. Each must be checked against the
derivation, implementation, and tests before it is promoted to the consolidated
report.

| ID | Location | Candidate issue | Verification task |
| --- | --- | --- | --- |
| RT-EF-RPT-001 | `sections/primer_em.tex`, `eq:rodrigues_matrix` around line 92 | Rodrigues' formula uses `sin(theta)` and `cos(theta)` but then defines `theta = a^T b`. The formula requires `theta = arccos(a^T b)` or equivalent direct use of sine/cosine from the dot product. | Check `utils/geometry.py`/related rotation helpers and all report uses of this rotation. Decide whether this is only a report typo or a code/report mismatch. |
| RT-EF-RPT-002 | `sections/primer_em.tex` around line 132 | The speed-of-light discussion says the curly-bracket factor "vanishes" for non-conducting materials. With `sigma = 0`, the factor evaluates to one, not zero. | Classify as documentation typo unless the formula or material implementation uses a different convention. |
| RT-EF-RPT-003 | `sections/primer_em.tex` around line 140 | The time-averaged Poynting vector is written as `1/2 Re{E x H}`. For complex phasors the standard expression is `1/2 Re{E x H*}`. The report's next equality with `||E||^2` appears to rely on conjugation. | Check phasor convention throughout the report and implementation. Verify whether power normalization, scattering, and radio-map path gain implicitly use the conjugated form. |
| RT-EF-RPT-004 | `sections/primer_em.tex`, `eq:V_Rmulti` around line 303 | The summation index is `n`, but path-dependent quantities use subscript `i`, followed by text saying all path-dependent quantities carry `i`. | Classify as notation typo after checking that later equations and code use a consistent path index. |
| RT-EF-RPT-005 | `sections/path_solver.tex`, Algorithm `Electric field computation`, around line 634 | Final coefficient uses `E_n^T E_n^rx`, while introductory/path-coefficient formulas use a Hermitian receive pattern. | Inspect `field_calculator.py`, antenna pattern evaluation, and receive projection. Determine whether `E_n^rx` is pre-conjugated or whether the report/code should use a Hermitian inner product. |
| RT-EF-RPT-006 | `sections/diffraction_first_order.tex` around lines 14-18 | The projected target basis vector appears to subtract `(e^T s)e` from `t`; mathematically this is suspicious because the projection of `t` onto the edge would normally use `(e^T t)e`. | Compare with the implemented first-order diffraction point computation and derive the expected expression for an arbitrary edge coordinate system. |
| RT-EF-RPT-007 | `sections/diffraction_first_order.tex` around lines 1-8 | Text says the edge passes through the origin, denoted by `o`, but uses `v = x e`. If `o` is not the zero vector, the expression would require `v = o + x e`. | Verify whether the annex intentionally translates the coordinate system to the edge origin. If so, note the wording as a clarity issue only. |
| RT-EF-RPT-008 | `sections/diffraction_radio_map_weighting_factor.tex` around lines 17-24 | The text introduces `hat{n}` as the measurement-cell normal, while equations use `n`. | Check whether `n` is consistently a unit normal in the derivation and implementation; otherwise normalization may affect the weighting factor. |
| RT-EF-RPT-009 | `sections/radio_map_solver.tex`, `eq:radio-map-solver-ci-all-2` around line 130 | The change of variables uses `r^2 / cos(theta)`. For surfaces with arbitrary normal orientation, the area factor may need `abs(cos(theta))` or a convention guaranteeing positive cosine. | Check implementation and scene/grid normal conventions. Determine whether sign handling is documented and tested. |

## Documentation Quality Issues To Track

- `sections/primer_em.tex` around line 359: "The incoming wave is must be..."
- `sections/primer_em.tex`: spelling/grammar issues such as "recomended",
  "defintitions", "coefficents", "When a rays hits", and "where me have
  omitted" should be collected during the docs pass.

## Electric-Field Equation And Convention Checklist

The following report labels must be mapped to implementation and tests during
the code audit:

- Coordinate and basis conventions: `eq:spherical_vecs`, `eq:rotation`,
  `eq:theta_phi_prime`, `eq:F_prime_2_F`, `eq:rodrigues_matrix`.
- Phasor, power, and antenna normalization: `eq:epsilon`, `eq:mu`, `eq:eta`,
  `eq:F`, `eq:S_spherical`, `eq:G`, `eq:Gdir`, `eq:C`, `eq:E_T`, `eq:P_T`,
  `eq:S_R`, `eq:A_R`, `eq:A_dir`, `eq:P_R`, `eq:V_R`.
- Path coefficient and channel response: `eq:E_R`, `eq:V_Rmulti`, `eq:H`,
  `eq:T_tilde`, `eq:H_final`, `eq:h_final2`, `eq:h_b`,
  `alg:path-solver-electric-field`.
- Reflection/refraction basis and Fresnel/slab model:
  `eq:fresnel_in_vectors`, `eq:W`, `eq:fresnel_out_vectors`,
  `eq:reflected_refracted_vectors`, `eq:fresnel`, `eq:fresnel_vac`,
  `eq:fresnel_slab`, `eq:q_fresnel_slab`.
- Diffraction and scattering: `eq:diffraction-matrix`,
  `eq:diffraction-cos`, `eq:R`, `eq:scattered_field`, `eq:xpd`,
  `eq:lambertian_model`, `eq:directive_model`, `eq:backscattering_model`.
- Path-solver energy/probability and arrays: `eq:path-solver-int-dist`,
  `eq:path-solver-used-dist-event`, `eq:path-solver-array-response-vector`.
- Radio-map field/path-gain estimators: `eq:radio-map-solver-path-gain`,
  `eq:radio-map-solver-w`, `eq:radio-map-solver-w-refl-refr`,
  `eq:radio-map-solver-ci-all-2`, `eq:radio-map-solver-ci-all-2-estimator`,
  `eq:radio-map-solver-ci-D`, `eq:radio-map-solver-ci-D-2`,
  `eq:radio-map-solver-ci-D-3`, `eq:radio-map-solver-ci-D-est`.
- Diffraction radio-map reparameterization:
  `eq:diffraction-radio-map-weighting-factor-reparametrization`,
  `eq:diffraction-radio-map-weighting-factor-kv-s`,
  `eq:diffraction-radio-map-weighting-factor-gamma`,
  `eq:diffraction-radio-map-weighting-factor-beta-0`.

## Ambiguities To Resolve

- Whether the report consistently uses transpose to mean bilinear product and
  Hermitian transpose only where conjugation is required, or whether some
  electric-field inner products omit conjugation accidentally.
- Whether all normals in area/weighting factors are guaranteed to be oriented
  such that cosine terms are nonnegative.
- Whether diffraction equations assume a translated coordinate system or an
  arbitrary edge point `o`.

## Code Mapping Notes

Not started.
