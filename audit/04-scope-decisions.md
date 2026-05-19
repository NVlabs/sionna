# Scope Decisions

This file records user-provided decisions that constrain the audit.

## Decisions

1. The deliverable is an audit report only. No patches should be produced during
   this stage.
2. Acting on audit findings may happen in a second stage.
3. The Sionna source code must not be modified during this audit. Code, test,
   tutorial, documentation, package metadata, and CI changes are deferred to a
   later implementation stage.
4. Private CI exists and runs all tests on different Python versions.
5. The Python versions are those expressed by `pyproject.toml`: Sionna requires
   Python `>=3.11`; Sionna RT requires Python `>=3.10`.
6. Private CI targets Ubuntu with GPU and CUDA installed.
7. All tests must pass.
8. CPU-only usage is a supported workflow.
9. The Sionna RT technical report is normative for Sionna RT behavior.
10. The audit should check whether Sionna RT implements what the report says and
   flag any differences.
11. Algorithms should be checked for technical correctness, including whether
   equations match the implementation and whether there are mistakes or bugs.
12. The public API should not change.
13. Rerunning notebooks is expensive. Current tests pass and all notebooks run
    cleanly in the current version.
14. The final output should be a consolidated Markdown report organized by
    workstreams.
15. Findings should be kept in Markdown only.
16. RT, PHY, and SYS should receive the deepest passes.
17. Documentation and tests should be reviewed after RT, PHY, and SYS.
18. The expected depth is line-by-line technical correctness inspection, not a
    quick overview scan.
19. GPU is generally the preferred execution target for audit commands and
    targeted tests because tests run much faster there. CPU remains supported
    and should be checked for CPU-specific behavior.
20. For Sionna RT, electric-field correctness is the highest-priority technical
    audit topic.
21. The Sionna RT technical report must itself be reviewed first for technical
    errors. After that review, the report, with any identified corrections, is
    the ground truth for comparing the implementation.
22. The audit should include ideas for improving implementations, making them
    more efficient, simpler, clearer, or easier to maintain. These are
    recommendations only and do not authorize source changes during this audit.

## Implications For The Audit

- Work is read-only unless explicitly redirected in a later stage.
- Sionna implementation, test, tutorial, documentation, package metadata, and CI
  files are read-only during this audit.
- Findings should be framed as report findings and recommendations, not patches.
- Full-suite execution and full notebook reruns are not a default task. Use
  targeted commands only when they materially improve evidence for a finding.
- Because tests pass, the audit should emphasize whether the tests are adequate
  and whether important behavior is missing coverage.
- Because the API must remain stable, suggested fixes should be API-preserving.
- For Sionna RT, every major technical-report section should be mapped to:
  - relevant implementation files,
  - tests,
  - docs,
  - limitations and approximations,
  - equation-to-code correspondence.
- For the electric-field audit, first review the report equations and derivation
  for internal consistency, sign conventions, units, polarization bases, Jones
  matrices, path coefficients, diffraction/scattering terms, and normalization.
  Then compare implementation against the reviewed report.
- Differences between the report and implementation are findings even if the
  current tests pass.
- Documentation findings should include cases where limitations, assumptions,
  or approximation domains are missing or unclear.
- Each deep implementation workstream should explicitly list files reviewed
  line by line so audit coverage is visible.
- The consolidated report should preserve workstream boundaries while still
  producing one prioritized overall conclusion.
- Generated checklists under `audit/checklists/` are the primary coverage guard
  for file/function/class/method review.
- Improvement ideas should be recorded separately from correctness findings and
  should include expected benefit, implementation risk, and validation needed.

## Remaining Open Questions

These questions are still useful before large parallel work begins:

1. Should the line-by-line implementation review begin with Sionna RT path
   solving, Sionna RT radio maps, or all RT modules in dependency order?
2. Should the consolidated report include an explicit per-file coverage table?
