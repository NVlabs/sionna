# Clarification Questions

These questions should shape the audit scope before we spend significant time
on expensive verification.

## Goals And Deliverables

1. Is the primary goal to produce an audit report, patches, or both?
2. Should findings be prioritized for an upcoming release, or for long-term
   technical hardening?
3. Do you want the final output to read like an internal engineering audit, a
   public-facing quality report, or a release-readiness checklist?
4. Should we include speculative improvement ideas, or only evidence-backed
   issues?
5. What severity threshold should trigger immediate code changes during this
   session?
6. Should we preserve API behavior unless clearly broken, or are compatibility
   changes acceptable?

## Supported Users And Platforms

7. Who are the most important users: researchers, students, internal NVIDIA
   teams, commercial users, or all of these?
8. Which operating systems are officially supported beyond the Ubuntu 24.04
   recommendation?
9. Is CPU-only usage a first-class supported workflow for PHY, SYS, and RT?
10. Which GPU generations and CUDA versions should be considered supported?
11. Are Apple Silicon, Windows, or WSL expected to work?
12. Are JAX and TensorFlow interop for RT still intended to be first-class,
    best-effort, or legacy compatibility?

## Release And CI Context

13. Is there private CI outside this repository that runs unit tests, docs, and
    notebooks?
14. Which tests currently gate merges?
15. Which tests currently gate releases?
16. Are docs built in CI, and are broken links checked?
17. Are notebooks executed automatically before release?
18. Are wheels built and smoke-tested in CI?
19. Is dependency vulnerability scanning handled elsewhere?
20. Is there a formal release checklist?

## Technical Correctness Depth

21. For PHY/SYS, should we validate equations against standards and papers, or
    mostly inspect implementation and tests?
22. For 3GPP and NR functionality, which specification versions should be used
    as the source of truth?
23. For FEC, do you want full standard conformance checks or targeted edge-case
    review?
24. For optical models, which references should be considered authoritative?
25. For SYS abstraction and BLER tables, are the tables generated elsewhere, and
    can we inspect that pipeline?
26. Should differentiability be tested systematically for all differentiable
    claims, or only for key examples?

## Sionna RT And Technical Report

27. Should the RT technical report be treated as normative for code behavior?
28. Are report limitations intended to become user-facing documentation warnings?
29. Do you want us to look for code/report mismatches section by section?
30. Should we attempt numerical reproduction of figures or examples from the RT
    technical report?
31. How much should we focus on current limitations such as hash collisions,
    first-order diffraction, source/target scaling, and synthetic-array
    approximation?
32. Are there internal validation scenes, measured datasets, or reference traces
    available for RT?
33. Should performance benchmarks be compared to claims in the technical report?

## Documentation And Tutorials

34. Should docs prioritize first-time users, expert researchers, or API reference
    completeness?
35. Are all tutorials expected to run with the current `main` branch?
36. Should notebook outputs be committed, stripped, or regenerated for releases?
37. Are Colab links expected to work for every tutorial?
38. Should docs include more limitations and numerical assumptions, even when it
    makes pages longer?
39. Should public examples avoid GPU-only assumptions?

## Test Strategy

40. Is test runtime a constraint for local contributors?
41. What is the acceptable runtime for smoke, unit, full, and release test tiers?
42. Are stochastic tests allowed to use statistical tolerances that occasionally
    fail, or should all tests be deterministic?
43. Should we add property-based tests for shape, dtype, precision, and inverse
    relationships?
44. Should long-running tests and benchmarks live in pytest, separate scripts,
    or CI-only workflows?
45. Should we audit coverage quantitatively, or focus on high-risk code paths?

## Code Improvement Boundaries

46. Are refactors acceptable if they reduce real risk but change internals?
47. Should we introduce new dependencies for linting, type checking, coverage,
    or docs checks?
48. Should style recommendations align with pylint, ruff, mypy, pyright, or no
    new tooling?
49. Are type annotations a desired direction for public APIs?
50. Are there areas considered too sensitive to refactor right now?

## Audit Logistics

51. Do you want parallel agents to work read-only first, or can some workers
    start producing patches immediately?
52. Should each workstream produce a short report, a detailed report, or both?
53. Do you want a single prioritized backlog at the end, or separate backlogs by
    package?
54. Should we track findings in Markdown only, or generate machine-readable
    JSON/YAML as well?
55. Should we keep all audit artifacts in the repo, or are some meant to be
    temporary scratch files?

