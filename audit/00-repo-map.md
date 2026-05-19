# Repository Map

This file records the initial repository exploration. It is a map for planning,
not a final audit finding list.

## Package Boundaries

### Main `sionna` package

- Package root: `src/sionna`
- Public modules:
  - `sionna.phy`: PyTorch-based physical-layer simulation.
  - `sionna.sys`: PyTorch-based system-level simulation.
  - `sionna.rt`: lazy-imported namespace expected to be provided by the
    `sionna-rt` package.
- `pyproject.toml` excludes `sionna.rt*` from the main package and declares
  `sionna-rt` as a dependency.
- Version source: `src/sionna/__init__.py`, currently `2.0.1`.

### Sionna RT package

- Submodule root: `ext/sionna-rt`
- Package root: `ext/sionna-rt/src/sionna/rt`
- Version source: `ext/sionna-rt/src/sionna/rt/__init__.py`, currently `2.0.1`.
- Dependencies include Mitsuba `3.8.0` and Dr.Jit `1.3.1`.
- Import side effect: if no Mitsuba variant is set, RT tries
  `cuda_ad_mono_polarized` and falls back to `llvm_ad_mono_polarized`.

## Source Inventory

Approximate Python line counts by major area:

| Area | Lines |
| --- | ---: |
| PHY channel | 13,124 |
| PHY FEC | 12,959 |
| PHY OFDM | 6,740 |
| PHY NR | 5,026 |
| SYS | 4,622 |
| RT path solvers | 4,571 |
| PHY MIMO | 3,480 |
| PHY utils | 3,067 |
| RT radio map solvers | 2,980 |
| RT utils | 2,542 |
| PHY core and mapping support | 2,284 |
| RT radio materials | 2,198 |
| PHY signal | 1,837 |
| RT other core modules | 5,839 |

Largest individual implementation files include:

- `src/sionna/phy/ofdm/channel_estimation.py`
- `src/sionna/phy/fec/polar/decoding.py`
- `src/sionna/phy/channel/utils.py`
- `src/sionna/phy/mapping.py`
- `src/sionna/phy/fec/ldpc/decoding.py`
- `src/sionna/phy/mimo/detection.py`
- `src/sionna/sys/topology.py`
- `ext/sionna-rt/src/sionna/rt/scene.py`
- `ext/sionna-rt/src/sionna/rt/path_solvers/paths.py`
- `ext/sionna-rt/src/sionna/rt/radio_map_solvers/radio_map_solver.py`

## Main Functional Areas

### PHY

- Core execution model: `Object`, `Block`, `config`.
- Precision and device policy: global config plus per-object overrides.
- RNG policy: Python, NumPy, and device-specific PyTorch generators.
- Signal processing: filters, windows, upsampling/downsampling, convolution.
- Mapping: constellations, mapping/demapping, sources, BER/BLER utilities.
- Channel models:
  - AWGN, flat fading, time-domain and OFDM-domain channel application.
  - TR 38.901 models: CDL, TDL, UMi, UMa, RMa, LSP, rays, antenna models.
  - Optical channel models: fiber and EDFA.
  - CIR dataset support.
- FEC:
  - CRC, interleaving, scrambling.
  - Linear, convolutional, LDPC, polar, and turbo coding.
- MIMO and OFDM:
  - Detection, equalization, precoding, stream management.
  - Resource grids, pilot patterns, channel estimation, modulation.
- NR:
  - Carrier, PUSCH configs, DMRS, precoding, receiver/transmitter, transport
    block encoder/decoder.

### SYS

- Physical-layer abstraction and effective SINR.
- Link adaptation.
- Power control.
- Scheduling.
- Topology and wraparound pathloss utilities.

### RT

- Scene loading and scene object management.
- Antenna arrays and antenna patterns.
- Radio devices: transmitters and receivers.
- Radio materials and scattering patterns.
- Path solver:
  - Shooting and bouncing candidate generation.
  - Image-method processing.
  - Field calculation.
  - Paths buffer and path output representation.
- Radio-map solver:
  - Planar and mesh radio maps.
  - SBR-based Monte Carlo estimators.
  - First-order diffraction support.
- Rendering, preview, cameras, and scene utilities.

## Test Inventory

- Main tests: `test/unit`
- RT tests: `ext/sionna-rt/test/unit`
- Total: 91 `test_*.py` files, about 42k lines.
- Main pytest config:
  - default device is GPU (`--device=gpu`),
  - supports `--device=cpu`, `--device=all`,
  - resets RNG seed to 42 per test,
  - includes aggressive CUDA memory cleanup,
  - parametrizes precision over `single` and `double`,
  - parametrizes selected compile modes.
- RT pytest config:
  - supports `--cpu`,
  - sets Mitsuba variants to GPU plus CPU fallback by default.

Important audit points:

- Determine what tests are expected to pass on CPU-only machines.
- Determine which tests are slow, stochastic, GPU-specific, or hardware-bound.
- Confirm whether `torch.compile` paths are systematically tested.
- Confirm whether RT tests cover all technical-report algorithm cases or mostly
  structural/unit behavior.

## Documentation Inventory

Main documentation:

- `doc/source/index.rst`
- `doc/source/installation.rst`
- `doc/source/phy`
- `doc/source/sys`
- `doc/source/rt` symlink to `../../ext/sionna-rt/doc/source`
- custom Sphinx extensions under `doc/source/_ext`
- notebook links through symlinks:
  - `doc/source/phy/tutorials/notebooks -> tutorials/phy`
  - `doc/source/sys/tutorials/notebooks -> tutorials/sys`

RT documentation:

- `ext/sionna-rt/doc/source/index.rst`
- `ext/sionna-rt/doc/source/em_primer.rst`
- `ext/sionna-rt/doc/source/api`
- `ext/sionna-rt/doc/source/developer`

Sionna RT technical report:

- HTML: https://nvlabs.github.io/sionna/rt/tech-report/index.html
- PDF/arXiv link exposed by the report: https://arxiv.org/abs/2504.21719
- Main sections:
  - essential concepts and terminology,
  - path solver,
  - radio map solver,
  - electromagnetics primer,
  - importance sampling,
  - first-order diffraction,
  - weighting factor for diffraction radio maps.

Technical-report limitations to audit against:

- Initial and scattered ray direction sampling could benefit from importance
  sampling.
- Path solver scaling with many targets and many sources has known limitations.
- Path solver supports first-order diffraction only.
- Paths with both diffraction and diffuse reflection are not supported.
- High-order diffraction is not supported.
- Specular-chain hash collisions are assumed negligible rather than fully
  resolved.
- Radio maps are non-coherent and therefore do not capture fast fading.
- RT radio maps support only synthetic transmit arrays, whose approximation is
  valid only when array dimensions are small relative to distances to targets
  and scatterers.
- Radio-map memory scales with number of cells, even if compute does not.

## Packaging And Release Surface

- Main install docs mention:
  - `pip install sionna`
  - `pip install sionna-rt`
  - `pip install sionna-no-rt`
  - source install with `pip install ext/sionna-rt/ .` followed by `pip install .`
- RT source install docs mention `pip install .` from the RT submodule root.
- Audit should verify:
  - main and RT wheel contents,
  - namespace package behavior,
  - editable install behavior,
  - dependency version compatibility,
  - whether `sionna-no-rt` packaging is represented in this repo or elsewhere,
  - source install command correctness and duplicate install behavior.

## Visible Automation

Visible workflows:

- `.github/workflows/codeql-analysis.yml`
- `.github/workflows/dco-check.yml`
- equivalent workflows in `ext/sionna-rt/.github/workflows`

No visible workflow was found for:

- unit tests,
- docs build,
- notebook execution,
- linting,
- package build validation,
- dependency vulnerability scanning,
- release artifacts.

This may be intentional if CI is private, but it is a major clarification point.

## Early Risk Hotspots

- Import side effects and backend selection for RT.
- `torch.compile` compatibility around lazy build and input conversion in
  `Block`.
- Global singleton config behavior across tests, devices, RNGs, and parallelism.
- Numerical stability and precision consistency in double vs single precision.
- Complex dtype conversions and real-valued representations in RT.
- Stochastic Monte Carlo estimators and reproducibility thresholds.
- Domain-reference correctness for 3GPP, NR, FEC, optical, EM, and diffraction
  algorithms.
- Large bundled data and package-data boundaries.
- Documentation/tutorial drift because notebooks are not executed during docs
  build.
- Lack of visible automated test/doc/lint release gates.

