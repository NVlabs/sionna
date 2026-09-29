Mapping
=======

.. currentmodule:: sionna.phy.mapping

This module provides components for mapping bits to constellation symbols and
demapping received symbols back to bit log-likelihood ratios (LLRs).
The :class:`Constellation` class defines symbol constellations (QAM, PAM, or custom),
while :class:`Mapper` and :class:`Demapper` perform the forward and inverse operations.
For symbol-level soft demapping, see :class:`SymbolDemapper`.
Additional :doc:`utils` include source blocks and conversion functions between
different representations.

.. toctree::
   :maxdepth: 2
   :hidden:

   Constellation
   Mapper
   Demapper
   SymbolDemapper
   utils