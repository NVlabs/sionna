# TS 38.101-4 V19.2.2 Fixed TDL Profile Data

This is the canonical resource directory for the fixed-delay TDL profiles from
TS 38.101-4 V19.2.2 Annex B.2.1. These resources are independent of the
TR 38.901 `spec_version`, which selects only scalable TDL/CDL and scenario
parameter tables.

| Specification | FR1 profiles | FR2 profiles | Restriction |
| --- | --- | --- | --- |
| TS 38.101-4 V16.1 | A30, B100, C300 | A30, C60 | None |
| Added by TS 38.101-4 V19.2.2 | D30 | D30, A10, D10 | A10 and D10 apply only for channel bandwidths greater than 200 MHz |

The TDL API does not receive a channel bandwidth and therefore does not enforce
these restrictions. Callers must select the fixed profile appropriate for the
simulated frequency range and bandwidth.
