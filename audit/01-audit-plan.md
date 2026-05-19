# Hierarchical Audit Plan

This plan is intentionally broad. It has been adjusted for the clarified scope
in `04-scope-decisions.md`: this audit produces a report only, preserves the
public API, assumes current tests and notebooks pass, and treats the Sionna RT
technical report as normative.

The Sionna source code must not be modified during the audit. Any code,
test, tutorial, or documentation changes are deferred to a later stage after
the report is reviewed.

The deepest passes are RT, PHY, and SYS. Documentation and tests are audited
after those core implementation reviews, with a focus on whether docs and tests
adequately capture the audited behavior.

## Phase 0 - Alignment

Goal: agree on the audit bar before doing expensive work.

Checks:

- Define release target, supported platforms, supported GPUs, and Python ranges.
- Confirm report-only deliverable boundaries.
- Decide how deep mathematical validation should go.
- Confirm API preservation constraints.
- Record private CI/release coverage that is not visible in this checkout.
- Record that private CI targets Ubuntu with GPU and CUDA installed.
- Define when targeted test/doc/notebook commands are worth running despite the
  expensive full-suite runtime.

Outputs:

- Finalized scope.
- Priority order for audit tracks.
- Evidence standard for findings.

## Phase 1 - Baseline Reproducibility

Goal: establish a lightweight local baseline and rely on private CI guarantees
for full-suite success.

Checks:

- Inspect environment and packaging assumptions for:
  - main package with RT,
  - main package without RT if applicable,
  - RT standalone,
  - docs,
  - tests.
- Verify source install commands from README and docs.
- Build wheels and inspect contents.
- Import smoke tests:
  - `import sionna`
  - `import sionna.phy`
  - `import sionna.sys`
  - `import sionna.rt`
- Run targeted test subsets only when needed to reproduce or disprove a finding.
- Avoid full notebook reruns unless explicitly approved or needed for a
  specific reproducible issue.
- Optionally run lightweight docs checks if they are cheap in the current
  environment.

Outputs:

- Lightweight reproducibility matrix.
- Known environment assumptions.
- Installation/doc command corrections.
- A clear list of commands deliberately not run because private CI/notebook
  guarantees already cover them.

## Phase 2 - Static And Structural Audit

Goal: find low-cost correctness and maintainability issues before domain-deep
review, and collect implementation simplification opportunities.

### 2.1 Packaging And Dependency Metadata

Checks:

- Main package and RT package wheel contents.
- Namespace package behavior for `sionna.rt`.
- Version synchronization between main and RT.
- `sionna-no-rt` documentation vs actual package source.
- Minimum Python and dependency versions.
- Package-data inclusion and exclusion rules.
- Editable install behavior.
- Import-time side effects and backend initialization.
- Compatibility with CPU-only operation.

### 2.2 Public API And Compatibility

Checks:

- Public exports from `__init__.py` files.
- API naming consistency.
- Deprecated parameter behavior.
- Error messages and input validation.
- Backward compatibility promises from docs and prior versions.
- Examples in docstrings.
- Recommendations must preserve the public API.

### 2.3 Code Quality And Maintainability

Checks:

- Large files and modules with mixed responsibilities.
- Repeated shape, dtype, or validation logic.
- Hidden global state.
- Mutable default or shared-state risks.
- Exception hygiene.
- Type annotations where they add clarity.
- TODOs and partially implemented areas.
- Opportunities to simplify implementation without changing API behavior.
- Opportunities to reduce memory traffic, allocations, duplicate computation,
  or Python overhead.

### 2.4 Security And Supply Chain

Checks:

- Dependency pins and lower bounds.
- Package data with executable or untrusted file formats.
- Notebook download scripts and external assets.
- XML, mesh, path, and scene loading behavior.
- CI supply-chain hardening.
- CodeQL scope and gaps.

## Phase 3 - Technical Correctness Audit

Goal: validate the science, numerics, API contracts, and differentiability claims.

For each algorithmic area, compare implementation against the equations,
docstrings, tests, and cited references that are already part of the repository.
For Sionna RT, compare implementation against the technical report section by
section and flag any behavioral mismatch, equation mismatch, undocumented
approximation, or test gap.

For Sionna RT, electric-field computation is the highest-priority technical
topic. The audit order is:

1. Review the technical report itself for technical errors.
2. Record any report issues or ambiguities.
3. Treat the reviewed report as ground truth.
4. Compare implementation to the reviewed report equation by equation.

The expected mode is line-by-line source inspection for RT, PHY, and SYS. Use
summaries only after the file-level analysis has been completed.

Use `audit/checklists/` as the coverage tracker. The audit proceeds
file-by-file, class-by-class, function-by-function, and method-by-method. No
workstream is complete until its checklist coverage is either checked off or an
explicit reason is recorded for deferring an item.

For targeted execution, GPU is the preferred path because it is generally faster
for this project. CPU remains a supported compatibility path and should be used
when auditing CPU-specific behavior.

In addition to correctness, each deep workstream should record implementation
improvement opportunities. These should be separated from bugs. Examples:
simpler control flow, clearer invariants, reduced intermediate storage, better
vectorization, fewer conversions, cheaper indexing, more robust numerical forms,
or clearer separation between geometry, EM, and API layers.

### 3.1 PHY Core, Precision, Device, RNG

Checks:

- `Object` and `Block` conversion semantics.
- Lazy build behavior under eager execution and `torch.compile`.
- Device selection and errors on unavailable devices.
- RNG reproducibility across Python, NumPy, PyTorch, CPU, and CUDA.
- Single vs double precision consistency.
- Complex dtype propagation.
- Gradient behavior through blocks where expected.

### 3.2 PHY Channel Models

Checks:

- AWGN, flat fading, time and OFDM channel shape contracts.
- Time/frequency domain normalization conventions.
- TR 38.901 parameter tables and random-variable correlations.
- UMi, UMa, RMa, CDL, TDL edge cases.
- Antenna pattern convention consistency.
- Optical fiber and EDFA equations, units, and numerical ranges.
- CIR dataset loading and validation.

### 3.3 PHY FEC

Checks:

- CRC polynomial definitions and bit ordering.
- Interleaver/deinterleaver inverse properties.
- LDPC encoder/decoder standard compliance and edge cases.
- Polar encoder/decoder frozen-bit logic, rate matching, and unsupported cases.
- Turbo and convolutional code trellis correctness.
- Differentiable decoder behavior where claimed.
- Hard/soft decision conventions and dtype handling.

### 3.4 PHY MIMO, OFDM, And NR

Checks:

- Resource grid indexing and pilot placement.
- OFDM modulator/demodulator normalization and cyclic prefix handling.
- Channel estimation interpolation and error variance conventions.
- Equalization and detection algorithms vs references.
- MIMO stream management, precoding, and tensor shape contracts.
- NR PUSCH, DMRS, transport-block, MCS, and layer-mapping compliance.
- End-to-end differentiability and compile behavior.

### 3.5 SYS

Checks:

- Effective SINR formulas and assumptions.
- BLER table usage and interpolation.
- Link adaptation decisions and boundary MCS cases.
- Power-control equations and units.
- Scheduling fairness and tie-breaking.
- Topology geometry, wraparound, and pathloss conventions.
- Integration contracts with PHY and RT outputs.

### 3.6 RT Path Solver

Checks:

- Electric-field transport and accumulation through LoS, reflection,
  refraction, diffuse reflection, and diffraction.
- Polarization basis transforms and Jones-matrix conventions.
- Phase, delay, path length, wavelength, and sign conventions.
- Antenna pattern integration into the field calculation.
- Units and normalization of field amplitudes, path coefficients, and received
  power.
- Candidate generation against the technical report.
- Equation-to-code traceability for SBR, image method, path coefficients,
  delays, Doppler shifts, and diffraction handling.
- SBR ray sampling, interaction sampling, and energy normalization.
- Image-method candidate refinement.
- LoS, reflection, refraction, diffuse reflection, and diffraction paths.
- First-order diffraction logic and lit-region handling.
- Synthetic vs non-synthetic array behavior.
- Doppler shift formulas.
- Hash collision behavior and practical collision probability.
- Differentiability boundaries and detached geometry rationale.
- Scalability with source and target counts.

### 3.7 RT Radio Map Solver

Checks:

- Monte Carlo estimators against the technical report.
- Equation-to-code traceability for planar maps, mesh maps, diffraction maps,
  weighting factors, and synthetic-array weighting.
- Planar and mesh radio-map cell definitions.
- Non-coherent gain interpretation and docs.
- Synthetic transmit-array approximation.
- Diffraction radio-map weighting.
- Russian roulette and gain-threshold unbiasedness.
- Memory scaling with cells and samples.
- Reproducibility with seeds.
- Numerical stability near edges, grazing incidence, and small cells.

### 3.8 RT Scene, Materials, Geometry, Rendering

Checks:

- Scene loading, XML/mesh handling, and material assignment.
- Radio-material physical units and parameter validation.
- ITU material models.
- Scattering pattern normalization.
- Wedge detection and edge geometry.
- Camera, preview, and renderer correctness.
- Interactive visualization dependencies.
- Variant switching and material reload TODO.

### 3.9 Cross-Package Integration

Checks:

- RT paths to PHY channel interfaces.
- Tutorials that combine RT and PHY/SYS.
- Tensor framework interoperability documented by RT.
- Unit conventions across PHY, SYS, and RT.
- Namespace package behavior when main and RT versions diverge.

## Phase 4 - Quality Gates

Goal: make quality repeatable.

Checks:

- Define test tiers:
  - smoke,
  - CPU unit,
  - GPU unit,
  - compile,
  - RT CPU,
  - RT GPU,
  - stochastic/statistical,
  - docs,
  - notebooks,
  - performance.
- Assess private CI coverage from available information and recommend missing
  quality gates where evidence shows a gap.
- Identify flaky tests and tests with hardware assumptions.
- Check coverage gaps by package.
- Add golden-reference or property-based tests where high value.
- Establish benchmark baselines for expensive kernels and RT solvers.
- Validate docs link integrity and notebook freshness by inspection or targeted
  commands, not by default full notebook reruns.

## Phase 5 - Findings And Hardening Backlog

Goal: convert audit work into durable improvements.

Outputs:

- Findings grouped by severity and area.
- Minimal repros for bugs.
- API-preserving recommendations.
- Implementation improvement opportunities, separated from correctness bugs.
- Longer-term roadmap for structural improvements.
- Documentation correction list.
- Test coverage plan.
- CI hardening plan.
- Release checklist.

## Severity Model

- Critical: can produce materially wrong scientific results, silent data
  corruption, import/install failure for supported users, or severe security
  exposure.
- High: likely correctness bug, major documentation contradiction, broken
  supported workflow, severe performance regression, or missing release gate.
- Medium: edge-case bug, confusing API behavior, incomplete tests for important
  behavior, maintainability issue with real risk.
- Low: typo, minor docs drift, local cleanup, non-blocking refactor.

## Required Finding Evidence

Each finding should include:

- title,
- severity,
- affected component,
- file and line references,
- reproduction or reasoning,
- expected behavior,
- actual behavior,
- suggested fix,
- tests or docs to add,
- uncertainty and open questions.
