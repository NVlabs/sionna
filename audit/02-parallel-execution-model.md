# Parallel Execution Model

This file defines how later agents or subagents can work in parallel without
duplicating effort or stepping on each other.

## Coordination Rules

- Each workstream owns an audit area and writes only to its assigned audit
  output file.
- Code patches are out of scope for this audit stage.
- Do not modify Sionna source code, tests, tutorials, or project documentation.
  Audit agents may only write audit artifacts unless explicitly redirected in a
  later implementation stage.
- Findings must use `templates/finding.md`.
- Agents should prefer evidence over impressions.
- Agents should separate:
  - confirmed bugs,
  - suspected risks,
  - documentation gaps,
  - test gaps,
  - improvement ideas.
- Agents should not run expensive full-suite commands unless assigned.
- Agents should assume the current full test suite and notebooks pass; targeted
  commands are for reproducing or clarifying specific findings.
- Recommendations must preserve public API behavior.
- RT, PHY, and SYS workstreams are expected to perform line-by-line technical
  inspection, not only module-level sampling.
- Workstreams must use `audit/checklists/` as their coverage tracker.
- GPU is the preferred target for targeted test commands unless the task is
  explicitly about CPU behavior.

## Suggested Workstreams

| ID | Workstream | Main Read Scope | Suggested Output |
| --- | --- | --- | --- |
| A0 | Audit coordination | `audit/`, all summaries | `audit/results/00-executive-summary.md` |
| A1 | Packaging and installs | `pyproject.toml`, `README.md`, `doc/source/installation.rst`, `ext/sionna-rt/pyproject.toml` | `audit/results/packaging.md` |
| A2 | CI and process | `.github`, contribution docs, test configs | `audit/results/ci-process.md` |
| A3 | PHY core | `src/sionna/phy/{config.py,object.py,block.py}`, base tests | `audit/results/phy-core.md` |
| A4 | PHY channel | `src/sionna/phy/channel`, `test/unit/channel` | `audit/results/phy-channel.md` |
| A5 | PHY FEC | `src/sionna/phy/fec`, `test/unit/fec` | `audit/results/phy-fec.md` |
| A6 | PHY MIMO/OFDM/NR | `src/sionna/phy/{mimo,ofdm,nr}`, matching tests | `audit/results/phy-mimo-ofdm-nr.md` |
| A7 | SYS | `src/sionna/sys`, `test/unit/sys` | `audit/results/sys.md` |
| A8 | RT path solver | `ext/sionna-rt/src/sionna/rt/path_solvers`, RT report path-solver section | `audit/results/rt-path-solver.md` |
| A9 | RT radio maps | `ext/sionna-rt/src/sionna/rt/radio_map_solvers`, RT report radio-map section | `audit/results/rt-radio-maps.md` |
| A10 | RT scene/materials/geometry/rendering | `ext/sionna-rt/src/sionna/rt` outside path/map solvers | `audit/results/rt-scene-materials.md` |
| A8a | RT electric field report review | Sionna RT technical report, EM primer, field calculation files | `audit/results/rt-electric-field-report-review.md` |
| A11 | Docs and tutorials | `doc`, `tutorials`, `ext/sionna-rt/doc`, `ext/sionna-rt/tutorials` | `audit/results/docs-tutorials.md` |
| A12 | Tests and coverage | `test`, `ext/sionna-rt/test` | `audit/results/tests-coverage.md` |
| A13 | Performance and scalability | large kernels, RT solvers, benchmark scripts | `audit/results/performance.md` |
| A14 | Security and supply chain | deps, loaders, notebooks, packaging, workflows | `audit/results/security-supply-chain.md` |

Priority order:

1. RT, PHY, and SYS deep technical audit.
   - Within RT, electric-field correctness comes first.
2. Documentation and tests, driven by findings from the implementation audit.
3. Packaging, CI/process, performance, and security/supply chain.

## Workstream Brief Template

Use this when spawning an agent:

```text
You are assigned workstream <ID>: <name>.

Goal:
- Audit <area> for technical correctness, maintainability, docs/test gaps, and
  release risk.

Read scope:
- <paths>

Primary references:
- <docs, tests, technical report sections>

Do:
- Produce evidence-backed findings.
- Include file and line references.
- Distinguish confirmed bugs from risks and questions.
- For RT, PHY, and SYS, inspect the assigned source line by line and report the
  files covered.
- Update or reference the relevant checklist items as coverage evidence.
- Suggest minimal tests, documentation changes, or API-preserving code changes
  as report recommendations only.
- Record opportunities to make implementations faster, simpler, clearer, or
  more memory-efficient, and keep them separate from correctness findings.

Do not:
- Edit code.
- Edit tests, tutorials, docs, package metadata, or CI workflows.
- Re-run another workstream's scope except for integration points.
- Run full notebook or full test suites unless explicitly assigned.
- Make broad style recommendations without concrete risk.

Output:
- Write or return content for <audit/results/file.md>.
```

## Dependency Graph

Start first:

- A1 Packaging and installs
- A2 CI and process
- A3 PHY core
- A8 RT path solver
- A9 RT radio maps
- A11 Docs and tutorials

Start after initial findings from A3:

- A4 PHY channel
- A5 PHY FEC
- A6 PHY MIMO/OFDM/NR
- A7 SYS

Start after A1, A2, and domain tracks have initial notes:

- A12 Tests and coverage
- A13 Performance and scalability
- A14 Security and supply chain

Final integration:

- A0 Audit coordination consolidates duplicate findings, resolves severity, and
  produces a prioritized hardening roadmap.

## Output Structure For Results

Each workstream result should use:

```text
# <Workstream Name>

## Scope

## Commands Run

## Confirmed Findings

## Risks And Gaps

## Documentation Issues

## Test Gaps

## Implementation Improvement Opportunities

## Suggested Patches

## Open Questions
```

Each implementation workstream should also include:

```text
## Files Reviewed Line By Line

## Equations Or Algorithms Checked

## Code/Test/Doc Traceability
```

## Patch Strategy

If a later stage moves from audit to implementation:

- Assign one worker per disjoint source area.
- Keep docs and tests in the same worker only when tightly coupled to the code
  change.
- Run targeted tests before broader tests.
- Use one integration pass to normalize style, update docs, and check package
  behavior.
