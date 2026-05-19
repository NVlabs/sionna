# Audit Checklists

These checklists are generated from the Python AST and are intended to force the audit to proceed function by function, class by class, and file by file.

Regenerate with:

```bash
python3 audit/scripts/generate_source_checklists.py
```

## Source Workstreams

| Checklist | Files | Classes | Functions | Methods |
| --- | ---: | ---: | ---: | ---: |
| [RT Path Solvers](rt-path-solvers.md) | 8 | 9 | 0 | 113 |
| [RT Radio Map Solvers](rt-radio-map-solvers.md) | 5 | 4 | 0 | 56 |
| [RT Core, Scene, Materials, Geometry, Rendering, Utilities](rt-core-scene-materials-utils.md) | 37 | 27 | 91 | 227 |
| [PHY Core, Mapping, Signal, Utilities](phy-core-mapping-signal-utils.md) | 21 | 37 | 52 | 153 |
| [PHY Channel](phy-channel.md) | 35 | 47 | 20 | 334 |
| [PHY FEC](phy-fec.md) | 29 | 31 | 33 | 320 |
| [PHY MIMO, OFDM, NR](phy-mimo-ofdm-nr.md) | 30 | 61 | 27 | 337 |
| [SYS](sys.md) | 8 | 8 | 9 | 90 |

## Test Workstreams

| Checklist | Files | Classes | Functions | Methods | Test items |
| --- | ---: | ---: | ---: | ---: | ---: |
| [Main Test Suite](tests-main.md) | 82 | 301 | 131 | 1417 | 1415 |
| [RT Test Suite](tests-rt.md) | 17 | 0 | 132 | 0 | 80 |

## Coverage Guard

All source files under `src/sionna` and `ext/sionna-rt/src/sionna` are assigned to a checklist.
