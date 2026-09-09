.. _usage:

Configuration and Usage
=======================

The link adaptation plugins are compiled automatically as part of the gNB Docker image build (``make build-gnb``). No additional build steps are required.

Selecting a Link Adaptation Variant
------------------------------------

A plugin variant is loaded by setting ``GNB_EXTRA_OPTIONS`` in the ``.env`` file for your configuration (e.g., ``config/rfsim/.env`` or ``config/b200/.env``):

.. code-block:: bash

   # OAI's original link adaptation algorithm (default, no plugin needed)
   # (leave GNB_EXTRA_OPTIONS unset or comment it out)

   # Simple OLLA — scheduling statistics-based adaptation
   GNB_EXTRA_OPTIONS="--loader.link_adaptation.shlibversion _olla"

   # Advanced OLLA with HARQ feedback history (recommended)
   GNB_EXTRA_OPTIONS="--loader.link_adaptation.shlibversion _mcs_hist_olla"

   # Basic logging — OAI-LA with CSV data collection
   GNB_EXTRA_OPTIONS="--loader.link_adaptation.shlibversion _log"

   # Advanced logging — OAI-LA with HARQ history and CSV data collection
   GNB_EXTRA_OPTIONS="--loader.link_adaptation.shlibversion _mcs_hist_log"

The OLLA variants rely on pre-computed data bundled in ``plugins/link_adaptation/data/AWGN_results/``: simple OLLA reads raw BLER-vs-SNR tables (one ``mcs*_awgn_5G.csv`` per MCS), while advanced OLLA reads sigmoid-fit parameters from ``bler_sigma_fit.csv``. Both are mounted automatically via the ``plugins`` volume — no manual configuration is needed.

Start the system and verify the plugin loaded:

.. code-block:: bash

   ./scripts/start_system.sh rfsim   # or b200 for over-the-air
   docker compose logs oai-gnb | grep "link_adaptation"

You should see ``[LOADER] library liblink_adaptation_olla.so successfully loaded``.

Data Collection
---------------

The logging and OLLA variants write CSV statistics files to ``/opt/oai-gnb/plugins/link_adaptation/`` inside the container, which maps to ``plugins/link_adaptation/`` on the host via the ``plugins`` bind mount. The exact columns depend on the variant: ``_log`` and ``_mcs_hist_log`` write BLER-window / scheduling counters, while the OLLA variants additionally record reported and effective SNR plus per-UE ACK/NACK counts. The ``_mcs_hist_*`` variants include an RNTI column.

Evaluation
----------

To evaluate and compare link adaptation algorithms under realistic channel conditions, the :ref:`channel_emulation` tutorial can be used. It allows controlled, reproducible experiments with site-specific channel models and provides live MCS and BLER visualization via the RIC stats server.
