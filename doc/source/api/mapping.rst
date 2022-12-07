Mapping
#######

This module contains classes and functions related to mapping
of bits to constellation symbols and demapping of soft-symbols
to log-likelihood ratios (LLRs). The key components are the
:class:`~sionna.mapping.Constellation`, :class:`~sionna.mapping.Mapper`,
and :class:`~sionna.mapping.Demapper`. A :class:`~sionna.mapping.Constellation`
can be made trainable to enable learning of geometric shaping.

Constellations
**************

Constellation
-------------
.. autoclass:: sionna.mapping.Constellation
   :exclude-members: call, build
   :members:

qam
---
.. autofunction:: sionna.mapping.qam

pam
---
.. autofunction:: sionna.mapping.pam

pam_gray
--------
.. autofunction:: sionna.mapping.pam_gray

Mapper
******
.. autoclass:: sionna.mapping.Mapper
   :exclude-members: call, build
   :members:

Demapping
*********

Demapper
--------
.. autoclass:: sionna.mapping.Demapper
   :exclude-members: call, build
   :members:

DemapperWithPrior
-----------------
.. autoclass:: sionna.mapping.DemapperWithPrior
   :exclude-members: call, build
   :members:

SymbolDemapper
--------------
.. autoclass:: sionna.mapping.SymbolDemapper
   :exclude-members: call, build
   :members:

SymbolDemapperWithPrior
-----------------------
.. autoclass:: sionna.mapping.SymbolDemapperWithPrior
   :exclude-members: call, build
   :members:

Utility Functions
*****************

SymbolLogits2LLRs
------------------
.. autoclass:: sionna.mapping.SymbolLogits2LLRs
   :exclude-members: call, build
   :members:

LLRs2SymbolLogits
------------------
.. autoclass:: sionna.mapping.LLRs2SymbolLogits
   :exclude-members: call, build
   :members:

SymbolLogits2LLRsWithPrior
---------------------------
.. autoclass:: sionna.mapping.SymbolLogits2LLRsWithPrior
   :exclude-members: call, build
   :members:

SymbolLogits2Moments
----------------------
.. autoclass:: sionna.mapping.SymbolLogits2Moments
   :exclude-members: call, build
   :members:

SymbolInds2Bits
---------------
.. autoclass:: sionna.mapping.SymbolInds2Bits
   :exclude-members: call, build
   :members:

PAM2QAM
-------
.. autoclass:: sionna.mapping.PAM2QAM
   :exclude-members: call, build
   :members:

QAM2PAM
-------
.. autoclass:: sionna.mapping.QAM2PAM
   :exclude-members: call, build
   :members:

References:
   .. [3GPPTS38211] ETSI TS 38.211 "5G NR Physical channels and modulation", V16.2.0, Jul. 2020
      https://www.3gpp.org/ftp/Specs/archive/38_series/38.211/38211-h00.zip
