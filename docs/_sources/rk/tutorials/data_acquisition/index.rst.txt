.. _data_acquisition:

Plugins & Data Acquisition
==========================

.. figure:: ../../figs/tutorial_capture_overview.png
   :align: center
   :width: 600px
   :alt: Data capture overview

   Schematic overview of the data capture plugin for the 5G NR PHY layer. MIMO aspects omitted for simplicity.

The Sionna Research Kit uses the OpenAirInterface (OAI) plugin system [OAILib]_ to integrate custom code. This tutorial shows how to capture real-world 5G signals (IQ samples) using a plugin that replaces the demapper function. The captured dataset can be used for training in the :ref:`neural_demapper` tutorial.


Quick Start: Data Capture Plugin
--------------------------------

Let's start with how to use the data capture plugin.
The next section will then explain the technical background and how to create your own plugin.

Create log files:

.. code-block:: bash

    mkdir -p plugins/data_acquisition/logs
    cd plugins/data_acquisition/logs
    touch demapper_in.txt demapper_out.txt
    chmod 666 demapper_in.txt demapper_out.txt

The plugin folder is automatically mounted to the gNB container.
This allows you to read/write the log files from the host system and can also be used to pass configuration files or models to the plugin (see :ref:`neural_demapper` tutorial).

You can enable the plugin by setting the environment variable in the ``.env`` file (e.g., ``config/b200/.env``) or by passing the option to the executable:

.. code-block:: bash

    GNB_EXTRA_OPTIONS="--loader.demapper.shlibversion _capture"

This loads the ``demapper_capture.so`` shared library. The main benefit is that the plugin can be loaded and unloaded dynamically, which allows for a flexible integration of custom code in the OAI stack. For example, you can now also load the :ref:`neural_demapper` plugin as alternative demapper without recompiling the gNB.

Start the gNB.

.. code-block:: bash

    ./scripts/start_system.sh rfsim

The plugin will be loaded and the data will be captured.
You should then see entries in the log files ``demapper_in.txt`` and ``demapper_out.txt``.

Data Format
-----------

Both files use a simple text format with a header followed by symbol data:

.. code-block:: text

    0.000000001         # Time source resolution
    1373853.185968662   # Timestamp
    QPSK                # Modulation scheme
    96                  # Number of symbols
    177 -179            # Data values (2 columns for QPSK, 4 for 16-QAM, etc.)
    -179 176
    ...

- ``demapper_in.txt``: Input symbols as (Real, Imag) pairs
- ``demapper_out.txt``: Output LLRs (2 per symbol for QPSK, 4 for 16-QAM, etc.)

See the :ref:`neural_demapper` tutorial for an example of loading this data in Python.


Writing Your Own Plugin
-----------------------

The following section explains the technical details on how to create your own plugin and integrate it into the OAI stack.


Step 1: Define the Plugin Interface
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Each plugin type needs an interface definition. The demapper interface is defined in `plugins/data_acquisition/src/nr_demapper_extern.h <https://github.com/NVlabs/sionna-rk/blob/main/plugins/data_acquisition/src/nr_demapper_extern.h>`_:

.. literalinclude:: ../../../../plugins/data_acquisition/src/nr_demapper_extern.h
   :language: c
   :start-after: START marker-plugin-extern
   :end-before: END marker-plugin-extern

The interface contains function pointers for:

- ``init``: Called once at startup
- ``init_thread``: Called for each worker thread in the thread pool
- ``shutdown``: Called at cleanup
- ``compute_llr``: The main function that replaces the original OAI demapper function


Step 2: Implement the Plugin Functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Your plugin must export functions matching the above interface. Here's the capture plugin's initialization and shutdown functions from `plugins/data_acquisition/src/nr_demapper_capture.c <https://github.com/NVlabs/sionna-rk/blob/main/plugins/data_acquisition/src/nr_demapper_capture.c>`_:

.. literalinclude:: ../../../../plugins/data_acquisition/src/nr_demapper_capture.c
   :language: c
   :start-after: START marker-capture-init
   :end-before: END marker-capture-init

.. literalinclude:: ../../../../plugins/data_acquisition/src/nr_demapper_capture.c
   :language: c
   :start-after: START marker-capture-shutdown
   :end-before: END marker-capture-shutdown

The main processing function is implemented in `plugins/data_acquisition/src/nr_demapper_capture.c <https://github.com/NVlabs/sionna-rk/blob/main/plugins/data_acquisition/src/nr_demapper_capture.c>`_:


.. literalinclude:: ../../../../plugins/data_acquisition/src/nr_demapper_capture.c
   :language: c
   :start-after: START marker-capture-compute-llr
   :end-before: END marker-capture-compute-llr

For simplicity, we only implement the QPSK and 16-QAM demapper functions, but extensions are straightforward.


Step 3: Create the Plugin Loader
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In order to load the plugin, we need to create a loader function that maps function names to symbols in your shared library. This is done in `plugins/data_acquisition/src/nr_demapper_load.c <https://github.com/NVlabs/sionna-rk/blob/main/plugins/data_acquisition/src/nr_demapper_load.c>`_:

.. literalinclude:: ../../../../plugins/data_acquisition/src/nr_demapper_load.c
   :language: c
   :start-after: START marker-capture-load
   :end-before: END marker-capture-load


Step 4: Register the Plugin
^^^^^^^^^^^^^^^^^^^^^^^^^^^


Add your plugin to the central plugin system in ``plugins/common/src/plugins.c``:

.. literalinclude:: ../../../../plugins/common/src/plugins.c
   :language: c
   :start-after: START marker-plugins
   :end-before: END marker-plugins

Note that the :ref:`accelerated_ldpc` tutorial is not registered here as it is a separate OAI plugin that uses the existing OAI loader independently of the Sionna Research Kit.

Step 5: Hook into OAI Code
^^^^^^^^^^^^^^^^^^^^^^^^^^

Finally, the OAI code must call your plugin instead of the original function. We patch the function `nr_ulsch_llr_computation.c <https://github.com/OPENAIRINTERFACE/openairinterface5g/blob/develop/openair1/PHY/NR_TRANSPORT/nr_ulsch_llr_computation.c>`_ in the OpenAirInterface codebase to add this hook:

.. literalinclude:: ../../../../ext/openairinterface5g/openair1/PHY/NR_TRANSPORT/nr_ulsch_llr_computation.c
   :language: c
   :start-after: START marker-compute-llr-start
   :end-before: END marker-compute-llr-end

If no plugin is loaded, the original implementation is used. This is the case when the ``--loader.demapper.shlibversion`` parameter is not set or when the plugin is not loaded. Note that we have therefore renamed the original function to ``nr_ulsch_compute_llr_default``.


Step 6: Add CMake Build Rules
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Each plugin needs a ``CMakeLists.txt`` that registers the loader with the build system, builds the plugin as a shared library, and adds it as a dependency of the gNB target. The following is an example for the data capture plugin:

.. code-block:: cmake

    # Register loader source with parent build
    set(PLUGINS_SRC ${PLUGINS_SRC} ${CMAKE_CURRENT_SOURCE_DIR}/src/nr_demapper_load.c PARENT_SCOPE)

    # Build plugin as shared library
    add_library(demapper_capture MODULE
        ${OPENAIR1_DIR}/PHY/NR_TRANSPORT/nr_ulsch_llr_computation.c
        src/nr_demapper_capture.c
    )
    target_link_libraries(demapper_capture PRIVATE pthread)

    # Build with gNB
    add_dependencies(nr-softmodem demapper_capture)

Then add your subdirectory to `plugins/CMakeLists.txt <https://github.com/NVlabs/sionna-rk/blob/main/plugins/CMakeLists.txt>`_:

.. code-block:: cmake

    add_subdirectory(data_acquisition)


Summary
-------

After re-building the gNB, you can now use the plugin by setting the ``--loader.demapper.shlibversion`` parameter to ``_capture``.

Though plugins add implementation overhead, the advantage is that they can be loaded dynamically, allowing you to rebuild the plugin without recompiling the entire gNB. This also makes it easier to compare different implementations.

The :ref:`neural_demapper`, the :ref:`neural_receiver`, and the :ref:`accelerated_ldpc` tutorials are implemented as plugins and can be used as a drop-in replacement for the original OAI functions.

References
----------

.. [OAILib] `OAI Shared Library Loader <https://github.com/OPENAIRINTERFACE/openairinterface5g/blob/develop/common/utils/DOC/loader.md>`_
