# TR 38.901 V16.1 Model Data

This directory contains the parameter tables used by the TR 38.901 V16.1
channel models.

The UMi, UMa, CDL, and scalable TDL files are the V16.1 tables. RMa uses the
corrected Rural Macro LSP and O2I tables. InH uses the V16.1 Indoor-Office
tables; these differ from the older assumptions used by the Release-14-era
calibration reference curves.

InF is included because TR 38.901 V16.1 contains the indoor-factory model.

Fixed-delay TDL profiles are not TR 38.901 versioned resources. They are kept
once under `../ts_38_101_4_v19_2_2` and are selected independently of the TR
`spec_version`. TS 38.101-4 V16.1 defined A30, B100, and C300 for FR1, and A30
and C60 for FR2. TS 38.101-4 V19.2.2 added D30 for both FR1 and FR2, plus A10
and D10 for FR2; A10 and D10 apply only for channel bandwidths greater than
200 MHz.
