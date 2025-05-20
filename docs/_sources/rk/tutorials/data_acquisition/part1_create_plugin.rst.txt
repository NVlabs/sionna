Part 1: Create a Plugin
=======================

A plugin mechanism in OpenAirInterface will allow us to isolate functionality originally implemented in the main source code and modularize it, so that multiple implementations and extensions to the functionality can be done. Using dynamic libraries, we will be able to select across multiple modules implementing the functionality based on the use case needs.

To keep the implementation of new functions flexible, we use OpenAirInterface's dynamic module loader [OAILib]_ which allows to load different implementations at runtime. This requires creating a plugin, but also adding the required hooks or entry points in the original code base to initialize, make them available and use the plugin. Further, modifications to the build system and the container images created are needed. 

The global modifications are already integrated in the patched OpenAirInterface code base. We will focus on the tasks required to add a plugin in this added infrastructure.

For this tutorial, we will show the procedure how the QAM demapper is modularized. The first part will describe the different parts of the plugin, and the second part will extend it with a second version that captures data from the selected function. Further, the :ref:`neural_demapper` in the next tutorial will be implemented as a plugin and can be used as a drop-in replacement for the existing OAI demapper.

.. note::
    The current plugin mechanism works with L1 code. Other parts of the OAI code, like L2, require some modifications and are still work in progress.

Architecture Overview
---------------------

There is a set of global changes to the OpenAirInterface needed to compile and expose the plugins in the code base. These changes involve exposing the plugin headers to the codebase, connecting the initialization functions, as well as changes to the build system. These changes are not covered by the tutorial.

The global files `plugins.h`, `plugins.c` and `CMakeLists.txt` represent the entry points of this infrastructure that are modified when adding a new plugin.

Plugin specific files cover the following parts:

* **Type definitions**. Data types and function signatures exposed by the plugin to OAI.
* **Public interface**. Set of functions exposed by the dynamic library, as well as the global entry point of the interface.
* **Init functions**. Functions called during the initialization of OAI that loads the dynamic library and populates the global entry point with the corresponding function pointers. Additional functions for proper deinitialization as well as additional setups.
* **Plugin entry point**. A globally accessible structure (a singleton) that contains the function pointers to the functions of the plugin to be used inside the OAI codebase.
* **Plugin code**. The code that implements the plugin functionality.

==============  ====================================================================
File            Description
==============  ====================================================================
plugins.h       Header entry point, all plugins headers added here
--------------  --------------------------------------------------------------------
plugins.c       Plugin entry point, wires loading, unloading and init of all plugins
--------------  --------------------------------------------------------------------
_defs.h         Types definitions and function signatures
--------------  --------------------------------------------------------------------
_extern.h       Public interface of the plugin
--------------  --------------------------------------------------------------------
_load.c         Load / Unload / Init functions
--------------  --------------------------------------------------------------------
.c              Plugin compilation units
--------------  --------------------------------------------------------------------
CMakeLists.txt  CMake build files
==============  ====================================================================

Select Functions
----------------

We focus on modularizing the demapper function ``nr_ulsch_compute_llr`` into a separate module. This function, found in ``openair1/PHY/NR_TRANSPORT/nr_ulsch_llr_computation.c``, handles the demapping of received QAM symbols to log-likelihood ratios (LLRs) based on the modulation scheme used. It has a clear and well-defined interface, making it an ideal candidate for modularization. The function's signature will serve as the basis for our plugin's interface requirements.

.. code-block:: c
    :caption: Functions to extract / nr_ulsch_llr_computation.c
    :name: nr_ulsch_llr_computation.c
    :linenos:

    void nr_ulsch_compute_llr(int32_t *rxdataF_comp,
                            int32_t *ul_ch_mag,
                            int32_t *ul_ch_magb,
                            int32_t *ul_ch_magc,
                            int16_t *ulsch_llr,
                            uint32_t nb_re,
                            uint8_t  symbol,
                            uint8_t  mod_order)
    {
        switch(mod_order){
            case 2:
            nr_ulsch_qpsk_llr(rxdataF_comp,
                                ulsch_llr,
                                nb_re,
                                symbol);
            break;
            case 4:
            nr_ulsch_16qam_llr(rxdataF_comp,
                                ul_ch_mag,
                                ulsch_llr,
                                nb_re,
                                symbol);
            break;
            case 6:
            nr_ulsch_64qam_llr(rxdataF_comp,
                            ul_ch_mag,
                            ul_ch_magb,
                            ulsch_llr,
                            nb_re,
                            symbol);
            break;
            case 8:
            nr_ulsch_256qam_llr(rxdataF_comp,
                                ul_ch_mag,
                                ul_ch_magb,
                                ul_ch_magc,
                                ulsch_llr,
                                nb_re,
                                symbol);
            break;
            default:
            AssertFatal(1==0,"nr_ulsch_compute_llr: invalid Qm value, symbol = %d, Qm = %d\n",symbol, mod_order);
            break;
        }
    }


Define Module Interface
-----------------------

Next, we define the module interface that specifies the function we want to extend, along with additional functions for module initialization and cleanup during loading and unloading.

.. literalinclude:: nr_demapper_extern.h
   :caption: nr_demapper_extern.h
   :language: cpp
   :linenos:
   :start-after: START marker-plugin-extern
   :end-before: END marker-plugin-extern

The structure ``demapper_interface_t`` defines the functions that each dynamically loaded library will export, making them available for dynamic linking at runtime. The global instance ``demapper_interface`` is the actual mapping loaded and active in the system - this serves as the entry point that makes the module available throughout the codebase. The functions ``load_demapper_lib`` and ``free_demapper_lib`` are called during initialization to select and load the library, utilizing the module loader functionality.

The actual function signatures are defined in a separate file ``nr_demapper_defs.h`` given as:

.. literalinclude:: nr_demapper_defs.h
   :caption: nr_demapper_defs.h
   :language: cpp
   :linenos:
   :start-after: START marker-plugin-defs
   :end-before: END marker-plugin-defs


Loading the Module
------------------

Loading the module happens during the initialization of the application. We leverage OAI's config utility, which provides flexibility in specifying the libraries to load through configuration options. This can be done in either a file or as command line arguments.

The load function performs dynamic linking of the selected module and retrieves the function pointers that the module exports. These function pointers are then mapped to the corresponding elements of the global entry point interface, making them available throughout the application.

The unload function handles the graceful shutdown of the module according to the module's own cleanup specifications.

.. literalinclude:: nr_demapper_load.c
   :caption: nr_demapper_load.c
   :language: c
   :linenos:
   :start-after: START marker-plugin-load
   :end-before: END marker-plugin-load

To integrate these functions into the executables, we need to make the module interface visible and initialize it when the application starts. We accomplish this by adding them to the two global entry points:

- ``tutorials/plugins.h``: Contains the required module interface headers
- ``tutorials/plugins.c``: Implements the initialization calls

These files provide a single, clean entry point for module integration. They are incorporated into both the gNB and UE executables of OpenAirInterface, as well as the build system. Additional plugins only need to added to these two files to be exposed to OAI.

.. literalinclude:: plugins.h
   :caption: plugins.h
   :language: c
   :linenos:
   :start-after: START marker-plugins
   :end-before: END marker-plugins

.. literalinclude:: plugins.c
   :caption: plugins.c
   :language: c
   :linenos:
   :start-after: START marker-plugins
   :end-before: END marker-plugins


Use Module Functions
--------------------

At this point, the module is loaded and its interface is available in the OpenAirInterface code. The next step is to use this interface. The demapper function selects between several modulation schemes and calls the corresponding implementation. Since we only want to accelerate specific modulation schemes, we will rename the original function ``nr_ulsch_compute_llr`` to ``nr_ulsch_compute_llr_default`` and implement a wrapper that falls back to this default implementation when the module does not handle a particular case.

.. code-block:: c
    :caption: Wrapper / nr_ulsch_llr_computation.c
    :name: nr_ulsch_llr_wrapper
    :linenos:

    // will include the definition for demapper_interface singleton
    #include <tutorials/plugins.h>

    void nr_ulsch_compute_llr(int32_t *rxdataF_comp,
                            int32_t *ul_ch_mag,
                            int32_t *ul_ch_magb,
                            int32_t *ul_ch_magc,
                            int16_t *ulsch_llr,
                            uint32_t nb_re,
                            uint8_t  symbol,
                            uint8_t  mod_order)
    {
        int handled = demapper_interface.compute_llr(rxdataF_comp,
                                                    ul_ch_mag,
                                                    ul_ch_magb,
                                                    ul_ch_magc,
                                                    ulsch_llr,
                                                    nb_re,
                                                    symbol,
                                                    mod_order);
        if (!handled)
            nr_ulsch_compute_llr_default(rxdataF_comp,
                                        ul_ch_mag,
                                        ul_ch_magb,
                                        ul_ch_magc,
                                        ulsch_llr,
                                        nb_re,
                                        symbol,
                                        mod_order);
    }


Module Implementation
---------------------

Now we can implement the actual functionality of the module. The implementation consists of two parts: a header file defining the interface and a source file containing its actual implementation. In this example, we take a minimal approach where the module simply returns zero, indicating that it does not handle any cases itself. Instead, it relies on the wrapper function to call the default behavior. This design pattern helps reduce code duplication while allowing the plugin to gradually take over specific cases as needed, with unhandled cases automatically falling back to the original implementation.

.. literalinclude:: nr_demapper_orig.c
   :caption: nr_demapper_orig.c
   :language: cpp
   :linenos:
   :start-after: START marker-plugin-orig
   :end-before: END marker-plugin-orig


Compiling
---------

Finally, we need to compile our new plugin. This is done by creating a new ``CMakeLists.txt`` file in our module directory and including it from the main ``tutorials/CMakeLists.txt`` file. This setup allows CMake to properly build and link our plugin as a shared library. A minimal version looks as follows:

.. literalinclude:: CMakeLists.txt
   :language: CMake
   :linenos:
   :start-after: START marker-plugin-cmake
   :end-before: END marker-plugin-cmake


Incremental Builds
------------------

Rebuilding the full ``ran-build-cuda`` OAI image every time to rebuild only the module is too time consuming. You can however use the image to rebuild it quickly.

.. code-block:: bash

    # Create temp directory
    mkdir /tmp/demapper

    # Launch a container, mounting the tutorials directory with the newer code.
    # If you made changes to openairinterface5g code, you also need to mount those.
    # for example, for openair1, add -v ext/openairinterface5g/openair1:/oai-ran/openair1
    docker run -v /tmp/demapper:/mnt -v ./tutorials:/oai-ran/tutorials --rm -it ran-build-cuda:latest bash

    # Inside the container
    cmake -S . -B cmake_targets/ran_build/build/ -GNinja
    cmake --build cmake_targets/ran_build/build/ --target demapper
    cmake --build cmake_targets/ran_build/build/ --target demapper_orig

    # Check updated libraries
    ls -lh cmake_targets/ran_build/build/libdemapper*

    # Copy resulting libraries to host
    cp cmake_targets/ran_build/build/libdemapper* /mnt/


Container Changes
-----------------

As a final step, you need to modify the gNB and UE Dockerfiles to include the newly created libraries in your containers. Here are the corresponding lines for the demapper:

.. code-block::
    :caption: gNB Docker file
    :name: Dockerfile.gnB.ubuntu22

    # ...
    COPY --from=gnb-build \
        /oai-ran/cmake_targets/ran_build/build/libdemapper*.so \
    # ...
    ldd /usr/local/lib/liboai_eth_transpro.so \
        /usr/local/lib/libdemapper*.so \
    # ...

.. code-block::
    :caption: NR-UE Docker file
    :name: Dockerfile.nrUE.ubuntu22

    # ...
    COPY --from=nr-ue-build \
        /oai-ran/cmake_targets/ran_build/build/libdemapper*.so \
    # ...
    ldd /usr/local/lib/liboai_eth_transpro.so \
        /usr/local/lib/libdemapper*.so \
    # ...


We can now rebuild the source tree with the demapper functions modularized into a dynamically linked library. If multiple modules are available, they can be selected at runtime using ``--loader.demapper.shlibversion _orig`` for the original version. Alternatively, you can add the following line to the ``.env`` file in your configuration directory:

.. code-block::
    :name: .env

    GNB_EXTRA_OPTIONS="--loader.demapper.shlibversion _orig"

When the library is loaded, you will see a message like this in the gNB logs:

.. code-block::

    [LOADER] library libdemapper_orig.so successfully loaded




