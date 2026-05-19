# Sionna Technical Audit Report

Status: draft scaffold.

## Executive Summary

To be written after workstream findings are consolidated.

## Scope

This is a report-only audit of Sionna PHY, Sionna SYS, and Sionna RT. The audit
preserves the public API and does not produce patches in this stage.

The deepest inspection is assigned to:

- Sionna RT,
- Sionna PHY,
- Sionna SYS.

Documentation and tests are audited after the implementation review, with the
goal of checking whether they accurately document and validate the behavior
found in code.

## Assumptions

- Private CI runs all tests across the Python versions allowed by
  `pyproject.toml`.
- Private CI targets Ubuntu with GPU and CUDA installed.
- CPU-only operation is supported.
- Current tests pass.
- Current notebooks run cleanly.
- The Sionna RT technical report is normative for RT behavior.
- Expensive full-suite and notebook execution is not part of the default audit
  workflow.

## Method

The audit uses line-by-line implementation review for RT, PHY, and SYS. Findings
must be backed by source references, implementation reasoning, doc/test
traceability, or a targeted reproducer.

For Sionna RT, the audit maps the technical report to code section by section:

- report claims,
- equations,
- implementation files,
- tests,
- documentation,
- limitations and approximations.

## Workstreams

### Sionna RT

RT electric-field correctness is the highest-priority RT topic. The technical
report must first be reviewed for technical errors, and then the reviewed report
is used as ground truth for the implementation comparison.

#### Electric Field Report Review

Result file: `audit/results/rt-electric-field-report-review.md`

#### Path Solver

Result file: `audit/results/rt-path-solver.md`

#### Radio Maps

Result file: `audit/results/rt-radio-maps.md`

#### Scene, Materials, Geometry, Rendering

Result file: `audit/results/rt-scene-materials.md`

### Sionna PHY

#### Core, Precision, Device, RNG

Result file: `audit/results/phy-core.md`

#### Channel Models

Result file: `audit/results/phy-channel.md`

#### FEC

Result file: `audit/results/phy-fec.md`

#### MIMO, OFDM, NR

Result file: `audit/results/phy-mimo-ofdm-nr.md`

### Sionna SYS

Result file: `audit/results/sys.md`

### Documentation And Tutorials

Result file: `audit/results/docs-tutorials.md`

### Tests And Coverage

Result file: `audit/results/tests-coverage.md`

### Supporting Workstreams

- Packaging: `audit/results/packaging.md`
- CI/process: `audit/results/ci-process.md`
- Performance/scalability: `audit/results/performance.md`
- Security/supply chain: `audit/results/security-supply-chain.md`

## Prioritized Findings

To be written after workstream findings are consolidated.

## Hardening Roadmap

To be written after workstream findings are consolidated.

## Implementation Improvement Opportunities

To be written after workstream findings are consolidated. This section should
separate performance, memory, simplification, maintainability, and numerical
robustness opportunities from confirmed correctness findings.
