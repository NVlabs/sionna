# TR 38.901 V19.2 Model Data

This directory contains the parameter tables used by the TR 38.901 V19.2
channel models.

The UMi, UMa, RMa, and InF files contain the V19.2 parameter tables. The
Indoor-Office, CDL, and scalable TDL data are unchanged from V16.1, but remain
versioned TR 38.901 resources.

Fixed-delay TDL profiles are not selected by the TR `spec_version`. The
canonical TS 38.101-4 V19.2.2 files are kept once under
`../ts_38_101_4_v19_2_2`. TS 38.101-4 V16.1 defined A30, B100, and C300 for
FR1, and A30 and C60 for FR2. V19.2.2 added D30 for both FR1 and FR2, plus A10
and D10 for FR2; A10 and D10 apply only for channel bandwidths greater than
200 MHz.
