.. _link_adaptation:

Link Adaptation Algorithms
==========================

.. figure:: ../../../doc/source/figs/tutorial_la_overview.png
   :align: center
   :width: 600px
   :alt: Link Adaptation Overview

   Link adaptation is called by the scheduling algorithm and takes HARQ feedback, effective SINR, and scheduling decisions as inputs to determine the optimal MCS.

Link adaptation is a fundamental technique in wireless communication systems that dynamically selects the optimal modulation and coding scheme (MCS) to maximize throughput while keeping the block error rate (BLER) close to a target value. The goal is to utilize the available channel capacity as efficiently as possible without exceeding reliability constraints that could impact higher-layer communication and latency requirements.

The link adaptation function is typically called by the scheduling algorithm, which determines allocation of user data to physical resource blocks (PRBs). For example, the OAI gNB's MAC scheduler applies a proportional fairness (PF) scheduler that balances throughput and fairness among users. The commonly used target BLER value is 10%, which provides a good balance between throughput and reliability.

Link adaptation algorithms typically obtain ACK/NACK feedback from HARQ (Hybrid Automatic Repeat Request) and adjust the MCS selection accordingly. In 5G NR, the frame structure can also be configured to include channel state information reference signals (CSI-RS) in the downlink, allowing the user equipment (UE) to report a channel quality indicator (CQI) index in a subsequent uplink transmission. Although important, the CQI report is typically delayed, outdated, and reported in a quantized form (with values ranging from 0 to 15), limiting its utility for accurate link adaptation.

.. figure:: ../../../doc/source/figs/tutorial_la_results.jpg
   :align: center
   :width: 600px
   :alt: Link Adaptation Result Preview

   MCS comparison of various link adaptation algorithms for an abrupt channel quality increase at time t=0s. Faster adaptation to higher spectral efficiency ultimately results in higher throughput and more resource efficient transmission (see [SALADPaper]_ for details).

In this tutorial, you will learn:

* How link adaptation works in the OAI gNB MAC scheduler
* How to develop and load MAC layer plugins using the `OAI Shared Library Loader <https://github.com/OPENAIRINTERFACE/openairinterface5g/blob/develop/common/utils/DOC/loader.md>`_
* How to implement Outer Loop Link Adaptation (OLLA) and advanced variants
* How to collect and analyze link adaptation statistics from a running system

This tutorial demonstrates several link adaptation algorithm variants:

- **OAI's original Link Adaptation (OAI-LA)**: The default algorithm implemented in the OpenAirInterface (OAI) gNB MAC scheduler that uses a sampling-based MCS selection.
- **Outer Loop Link Adaptation (OLLA)**: The industry-standard algorithm that maintains an adaptive SINR offset based on HARQ feedback. This tutorial starts with a simple OLLA implementation that requires minimal adaptation of the OAI gNB MAC and operates on scheduling statistics.
- **Advanced OLLA** *(recommended)*: Enhanced version with a per-UE HARQ feedback history list and a sigmoid-fit ILLA (Inner Loop Link Adaptation) that interpolates BLER across MCS and code block size.

The MAC plugin architecture introduced here is not limited to link adaptation — it can be used for any MAC layer feature that can be implemented as a dynamically loaded library.

For a simulation-based introduction to link adaptation, see the `Sionna Link Adaptation Tutorial <https://nvlabs.github.io/sionna/sys/tutorials/notebooks/LinkAdaptation.html>`_. For further details on the development of more advanced link adaptation algorithms, including results from real-world over-the-air experiments, we refer to [SALADPaper]_.

.. toctree::
    :maxdepth: 1

    oai-la/oai-la.rst
    olla/olla.rst
    usage.rst

References
----------

.. [SALADPaper] R. Wiesmayr, L. Maggi, S. Cammerer, J. Hoydis, F. Aït Aoudia, and A. Keller, `"SALAD: Self-Adaptive Link Adaptation," <https://arxiv.org/pdf/2510.05784>`_ arXiv preprint arXiv:2510.05784, 2024.
