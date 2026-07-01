OAI's Link Adaptation Algorithm
===============================

The original OpenAirInterface link adaptation algorithm (OAI-LA) implements a
sampling-based approach for the MCS selection.
This baseline implementation serves as a foundation for more advanced algorithms
and demonstrates the basic principles of link adaptation.
OAI-LA samples different MCS configurations, performing single-step adjustments of the MCS index based on observed BLER statistics.
It uses CQI reports to determine the maximum MCS index that can be used to achieve the target BLER.

Algorithm Overview
------------------

The OAI-LA algorithm operates on a simple threshold-based principle:

- *BLER Monitoring*: Tracks the block error rate over a sliding window
- *Threshold Comparison*: Compares current BLER against upper and lower thresholds
- *MCS Adjustment*:

  - Increments MCS if BLER < lower threshold (requires at least 4 DL
    transmissions since the last update)
  - Decrements MCS if BLER > upper threshold, with a floor at MCS 6
  - Decrements MCS if fewer than 4 DL transmissions were scheduled
    (inactivity guard, with a floor at MCS 9)
- *Stability Constraints*: Updates the MCS at most every ``BLER_UPDATE_FRAME``
  frames (10 by default) to avoid reacting to transient statistics
- *Channel Quality Indicator (CQI) Feedback*: Uses CQI reports from CSI-RS to determine the maximum allowable MCS, enabling rapid MCS reduction when channel conditions degrade

The algorithm estimates the average BLER via an exponentially weighted moving average:

.. math::

   \text{BLER}_{avg} = \alpha \cdot \text{BLER}_{avg} + (1-\alpha) \cdot \text{BLER}_{window}

where :math:`\text{BLER}_{window}` is the recent BLER, which is computed from
the changes of the scheduling statistics since the last link adaptation update
(by default, every 10 frames), and :math:`\alpha = 0.9` is the smoothing factor.
This is implemented in ``link_adaptation_log.c``:

.. literalinclude:: link_adaptation_log.c
   :language: c
   :start-after: // START marker-bler-ewma
   :end-before: // END marker-bler-ewma


The main MCS selection logic is implemented in the ``link_adaptation_get_mcs_from_bler`` function:

.. literalinclude:: link_adaptation_orig.c
   :language: c
   :start-after: int new_mcs = old_mcs
   :end-before: bler_stats->last_frame = frame;


MAC Plugin Infrastructure
-------------------------

We leverage OAI's `dynamic library loading mechanism <https://github.com/OPENAIRINTERFACE/openairinterface5g/blob/develop/common/utils/DOC/loader.md>`_ to support multiple algorithm variants and incremental builds. This enables runtime selection of different link adaptation algorithms without recompilation.

Plugin Interface
~~~~~~~~~~~~~~~~

The plugin system defines a standard interface for link adaptation modules in ``link_adaptation_extern.h``:

.. literalinclude:: link_adaptation_extern.h
   :language: c
   :start-after: #include "link_adaptation_defs.h"
   :end-before: #endif

Plugin Loading and Unloading
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The MAC plugin infrastructure automatically loads the appropriate link adaptation library in ``mac_plugins.c``:

.. literalinclude:: mac_plugins.c
   :language: c
   :start-after: // START marker-plugins
   :end-before: // END marker-plugins

See the :ref:`usage` section for available plugin variants and configuration details.
