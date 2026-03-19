Physical Layer (PHY)
====================

This package provides a differentiable link-level simulator.

It seamlessly integrates multiple communication system elements, including
:doc:`forward error correction (FEC) <api/fec>`,
:doc:`multiple input multiple output (MIMO) systems <api/mimo>`,
:doc:`orthogonal frequency division multiplexing (OFDM) <api/ofdm>`, and a range
of :doc:`wireless <api/channel.wireless>` and :doc:`optical <api/channel.optical>`
channel models. It also supports simulation of some :doc:`5G NR <api/nr>` compliant features.

The best way to get started is by going through some of the :doc:`Tutorials <tutorials>`.

Advanced users may want to consult the :doc:`Developer Guides <developer/developer>`
for a deeper understanding of the inner workings of Sionna PHY and to learn how to
extended it with custom physical layer algorithms.

.. toctree::
   :hidden:
   :maxdepth: 6


   tutorials
   api/phy.rst
   developer/developer
