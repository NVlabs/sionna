# Sionna Technical Audit Workspace

This directory is the working area for a structured technical audit of the
Sionna repository, including Sionna PHY, Sionna SYS, and the Sionna RT submodule
under `ext/sionna-rt`.

The goal is not a shallow lint pass. The audit should produce an
evidence-backed report with concrete hardening recommendations that improve
correctness, maintainability, reproducibility, documentation, and release
confidence. Code changes are out of scope for this audit stage.

## Clarified Scope

- Deliverable: report only. Acting on findings may happen in a later stage.
- The Sionna source code must not be modified during this audit. This includes
  `src/`, `ext/sionna-rt/src/`, tests, tutorials, and project documentation.
  Code changes belong to a later implementation stage.
- API behavior must not change. Recommendations should be API-preserving unless
  a finding explicitly explains why an existing API behavior is incorrect.
- Private CI exists and runs all tests across multiple Python versions.
- Private CI targets Ubuntu with GPU and CUDA installed.
- All tests and notebooks are assumed to pass in the current version.
- CPU-only usage is a first-class supported workflow.
- The Sionna RT technical report is normative for Sionna RT behavior.
- The RT audit must check whether implementation, documentation, and tests
  match the report, including the LaTeX equations and stated limitations.
- The audit should also identify implementation improvement opportunities:
  simplifications, efficiency improvements, memory reductions, clearer
  abstractions, and maintainability improvements. These are report
  recommendations only, not code changes during this stage.
- Expensive full-suite and notebook reruns should be avoided unless a targeted
  reproducer is needed for a specific finding.

## Current Snapshot

- Main Git repository: `sionna`, branch `main`, clean working tree before this
  audit scaffold was added.
- Submodule: `ext/sionna-rt` at `19c48d6c9ff2549b7a6fcda465f01c532e0bd220`
  (`v2.0.1`).
- Main package: `sionna==2.0.1`, Python `>=3.11`, PyTorch `>=2.9.1`, depends on
  `sionna-rt`.
- RT package: `sionna-rt==2.0.1`, Python `>=3.10`, depends on Mitsuba `3.8.0`
  and Dr.Jit `1.3.1`.
- Source scale: about 71k lines of Python across main and RT source trees.
- Tests scale: about 42k lines of Python tests, 91 `test_*.py` files.
- Documentation: Sphinx docs with PHY and SYS docs in the main repo, RT docs
  symlinked from `doc/source/rt` to `ext/sionna-rt/doc/source`.
- Tutorials: 36 notebooks across PHY, SYS, and RT.
- Visible GitHub workflows: CodeQL and DCO checks. No unit-test, doc-build, or
  notebook execution workflow was visible in this checkout.

## Audit Files

- `00-repo-map.md`: repository inventory, package boundaries, documentation
  layout, test layout, and early risk hotspots.
- `01-audit-plan.md`: hierarchical audit plan with phases, tracks, and concrete
  checks.
- `02-parallel-execution-model.md`: proposed workstream split for later
  parallel agents and subagents.
- `03-questions.md`: clarification questions for deciding depth, priorities,
  deliverables, and constraints.
- `04-scope-decisions.md`: answered scope decisions and their implications for
  how the audit should be executed.
- `checklists/`: generated function-by-function, class-by-class, file-by-file
  review checklists for source and tests.
- `report.md`: consolidated report skeleton that will collect the workstream
  findings.
- `references/README.md`: canonical reference links, including the Sionna RT
  technical report PDF and TeX source.
- `results/README.md`: per-workstream report output directory.
- `templates/finding.md`: suggested format for findings so results stay
  comparable across agents.

## Initial High-Signal Observations

- Packaging has a deliberate split: `sionna` excludes `sionna.rt*`, while
  `sionna-rt` provides `sionna.rt` as a namespace package. This needs explicit
  audit coverage for editable installs, wheel builds, import behavior, version
  skew, and source install instructions.
- The main docs build pulls RT docs through a symlink and custom Sphinx config.
  This is powerful but fragile enough to deserve dedicated doc-build and link
  validation.
- The tutorial execution script has an empty active notebook list in this
  checkout, but notebooks are expected to run cleanly in the current version.
  The audit should focus on docs/tutorial drift risks and targeted examples
  rather than full notebook execution.
- Test configuration defaults to GPU for Sionna PHY/SYS tests, with memory
  cleanup logic and optional CPU mode. Private CI runs all tests across multiple
  Python versions, so the audit should inspect test adequacy and coverage rather
  than treating local full-suite execution as a baseline task.
- The Sionna RT technical report identifies algorithmic limitations that should
  become first-class audit tracks: sample efficiency, scaling with many
  sources/targets, first-order diffraction only, no mixed diffraction plus
  diffuse paths, synthetic-array approximation limits, and hash collision
  behavior.

## Working Rule

During this audit, write only audit artifacts under `audit/` and local
environment files needed for setup. Do not modify Sionna implementation,
tests, tutorials, or documentation.

Every audit finding should be tied to at least one of:

- a reproducible failure or minimal example,
- a source location and reasoning chain,
- a failed or missing test,
- a mismatch between docs, code, tests, and the Sionna RT technical report,
- a mismatch between an equation and its implementation,
- a clear release/process risk.

Every improvement recommendation should explain:

- current implementation complexity or cost,
- why the proposed direction is simpler, faster, safer, or easier to maintain,
- expected impact,
- risk to numerical behavior or API compatibility,
- tests or benchmarks needed before implementation.

For RT, PHY, and SYS, the expected depth is line-by-line technical inspection of
the implementation and its tests/docs. The audit should not settle for a broad
overview scan.

The generated checklists in `audit/checklists/` are the coverage control. A file,
class, function, or method should only be checked off after it has been
technically reviewed. The checklist is not a substitute for the report; it is a
guard against accidentally skipping implementation units.
