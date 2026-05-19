# Audit TODO

This file tracks items for the user to provide or decide before the deep
line-by-line audit begins.

## Reference Drop Location

Place PDFs, Markdown, TeX/source archives, or internal notes under:

```text
audit/references/local/
```

Suggested folder structure:

```text
audit/references/local/
  standards/
    3gpp/
    itu/
  sionna-rt/
  phy-fec/
  phy-mimo-ofdm-channel/
  sys/
  optical/
```

These files are treated as local audit material. They should not be assumed to
be committed unless explicitly requested.

## Highest Priority References

### Standards

- [ ] 3GPP TR 38.901, Release 16.1 / v16.01.00:
      *Study on channel model for frequencies from 0.5 to 100 GHz*
- [ ] 3GPP TS 38.211 v16.2.0:
      *NR; Physical channels and modulation*
- [ ] 3GPP TS 38.212 v16.5.0:
      *5G NR Multiplexing and channel coding*
- [ ] 3GPP TS 38.213 v16.2.0:
      *NR; Physical layer procedures for control*
- [ ] 3GPP TS 38.214 v16.2.0:
      *NR; Physical layer procedures for data*
- [ ] 3GPP TS 36.212 v15.3.0:
      *E-UTRA; Multiplexing and channel coding*
- [ ] 3GPP TS 38.141-1 Release 17 / v17.17.00:
      *Base Station conformance testing Part 1*

### Sionna RT

- [ ] *Sionna RT: Technical Report*, arXiv:2504.21719, PDF
- [ ] *Sionna RT: Technical Report*, arXiv:2504.21719, TeX/source if available
- [ ] ITU-R P.2040-3:
      *Effects of building materials and structures on radiowave propagation
      above about 100 MHz*
- [ ] ITU-R P.526-15:
      *Propagation by diffraction*
- [ ] C. A. Balanis, *Advanced Engineering Electromagnetics*, 2012
- [ ] C. A. Balanis, *Antenna Theory: Analysis and Design*, 2nd ed., 1997
- [ ] McNamara, Pistorius, Malherbe,
      *Introduction to the Uniform Geometrical Theory of Diffraction*, 1990
- [ ] Keller, *Geometrical Theory of Diffraction*, 1962
- [ ] Kouyoumjian,
      *A uniform geometrical theory of diffraction for an edge in a perfectly
      conducting surface*, 1974
- [ ] Luebbers,
      *Finite conductivity uniform GTD versus knife edge diffraction in
      prediction of propagation path loss*, 1984
- [ ] METIS Deliverable D1.4: *METIS Channel Models*, 2015

### PHY / FEC

- [ ] Arikan, *Channel polarization*, 2009
- [ ] Arikan, *A Performance Comparison of Polar Codes and Reed-Muller Codes*,
      2008
- [ ] Tal and Vardy, *List Decoding of Polar Codes*, 2015
- [ ] Balatsoukas-Stimming et al.,
      *LLR-Based Successive Cancellation List Decoding of Polar Codes*, 2015
- [ ] Hashemi et al.,
      *Simplified Successive-Cancellation List Decoding of Polar Codes*, 2016
- [ ] Hashemi et al.,
      *Fast and Flexible Successive-cancellation List Decoders for Polar Codes*,
      2017
- [ ] Bioglio et al., *Design of Polar Codes in 5G New Radio*, 2020
- [ ] Richardson and Kudekar,
      *Design of low-density parity-check codes for 5G new radio*, 2018
- [ ] Ryan, *An Introduction to LDPC Codes*, 2004
- [ ] Chen et al., *Reduced-complexity Decoding of LDPC Codes*, 2005
- [ ] Bahl, Cocke, Jelinek, Raviv,
      *Optimal Decoding of Linear Codes for Minimizing Symbol Error Rate*, 1974
- [ ] Viterbi,
      *Error bounds for convolutional codes and an asymptotically optimum
      decoding algorithm*, 1967
- [ ] Berrou et al.,
      *Near Shannon limit error-correcting coding and decoding: Turbo-codes*,
      1993
- [ ] ten Brink,
      *Convergence Behavior of Iteratively Decoded Parallel Concatenated Codes*,
      2001
- [ ] ten Brink, Kramer, Ashikhmin,
      *Design of low-density parity-check codes for modulation and detection*,
      2004

### PHY MIMO / OFDM / Channel

- [ ] Tse and Viswanath, *Fundamentals of Wireless Communication*, 2005
- [ ] Bjornson, Hoydis, Sanguinetti, *Massive MIMO Networks*, 2017
- [ ] Yang and Hanzo, *Fifty Years of MIMO Detection*, 2015
- [ ] Cespedes et al.,
      *Expectation Propagation Detection for High-Order High-Dimensional MIMO
      Systems*, 2014
- [ ] Studer et al.,
      *ASIC Implementation of Soft-Input Soft-Output MIMO Detection Using MMSE
      Parallel Interference Cancellation*, 2011
- [ ] Xiao, Zheng, Beaulieu,
      *Novel Sum-of-Sinusoids Simulation Models for Rayleigh and Rician Fading
      Channels*, 2006
- [ ] Mallik,
      *The exponential correlation matrix: Eigen-analysis and applications*,
      2018

### SYS

- [ ] Lagen et al.,
      *New radio physical layer abstraction for system-level simulations of 5G
      networks*, 2020
- [ ] Jalali et al.,
      *Data throughput of CDMA-HDR a high efficiency-high data rate personal
      communication wireless system*, 2000
- [ ] Mo and Walrand,
      *Fair end-to-end window-based congestion control*, 2000
- [ ] Pedersen et al.,
      *Frequency domain scheduling for OFDMA with limited and noisy channel
      feedback*, 2007
- [ ] Sampath et al.,
      *On setting reverse link target SIR in a CDMA system*, 1997
- [ ] Any internal document or script describing BLER table generation and
      EESM/MIESM calibration

### Optical

- [ ] Agrawal, *Fiber-optic Communication Systems*, 4th ed., 2010
- [ ] Baney, Gallion, Tucker,
      *Theory and Measurement Techniques for the Noise Figure of Optical
      Amplifiers*, 2000
- [ ] Giles and Desurvire,
      *Modeling Erbium-Doped Fiber Amplifiers*, 1991
- [ ] Essiambre et al., *Capacity Limits of Optical Fiber Networks*, 2010
- [ ] Fleck, Morris, Feit,
      *Time-dependent Propagation of High Energy Laser Beams Through the
      Atmosphere*, 1976
- [ ] Hardin and Tappert,
      *Applications of the Split-Step Fourier Method to the Numerical Solution
      of Nonlinear and Variable Coefficient Wave Equations*, 1973
- [ ] Wai, Menyuk, Chen,
      *Stability of Solitons in Randomly Varying Birefringent Fibers*, 1991

## Remaining Decisions

- [ ] Should the line-by-line implementation review begin with Sionna RT path
      solving, Sionna RT radio maps, or all RT modules in dependency order?
- [ ] Should the consolidated report include an explicit per-file coverage
      table?

## Sionna RT Electric-Field Audit Priority

- [x] Record that electric-field correctness is the highest-priority Sionna RT
      audit topic.
- [x] Record that the Sionna RT technical report must be reviewed first for
      technical errors.
- [ ] Review the Sionna RT technical report field equations for internal
      consistency before comparing code.
- [ ] Map every electric-field equation and convention from the reviewed report
      to implementation files and tests.

## Audit Output Requirements

- [x] Include implementation improvement opportunities in the audit report.
- [x] Keep improvement opportunities separate from correctness bugs.
- [x] Treat improvements as report recommendations only, not permission to edit
      source code during this audit.

## Environment Setup Notes

- [x] Work on a separate Git branch named `audit`.
- [x] Create a local `.venv` for this workspace.
- [x] Configure VS Code/Cursor to use `.venv/bin/python`.
- [x] Treat GPU as the preferred execution target for audit commands and
      targeted tests unless CPU behavior is under review.
- [x] Record that Sionna source code, tests, tutorials, docs, package metadata,
      and CI workflows must not be modified during this audit.
