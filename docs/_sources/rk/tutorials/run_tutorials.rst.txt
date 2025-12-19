.. _run_tutorials:

Running the Tutorials
=====================

This is quick reference guide to run the precompiled tutorials. For in-depth explanations, see the individual :ref:`tutorials` pages.

Tutorials are implemented as plugins which can be enabled by setting the corresponding environment variables in the ``config/<setup-type>/.env`` file.
The following sections assume that the system is started in the ``rfsim`` configuration without connected hardware.

After setting the environment variable, the system can be started via:

.. code-block:: bash

    # Start the system (rfsim or other configs in config/)
    ./scripts/start_system.sh rfsim

    # Stop the system
    ./scripts/stop_system.sh

    # Check running containers
    docker compose ps

    # View gNB logs
    docker compose logs -f oai-gnb


GPU-Accelerated LDPC
--------------------

Enable the CUDA-accelerated LDPC decoder by updating your configuration (``config/rfsim/.env`` file).

.. code-block:: bash

    GNB_EXTRA_OPTIONS=--loader.ldpc.shlibversion _cuda

Start the system:

.. code-block:: bash

    ./scripts/start_system.sh rfsim

Verify the plugin is loaded by checking the gNB logs:

.. code-block:: text

    [LOADER] library libldpc_cuda.so successfully loaded

Demapper Capture Plugin
-----------------------

To capture IQ samples and LLRs using the capture plugin, create the log files in the ``plugins/data_acquisition/logs`` directory:

.. code-block:: bash

    mkdir -p plugins/data_acquisition/logs
    cd plugins/data_acquisition/logs
    touch demapper_in.txt demapper_out.txt
    chmod 666 demapper_in.txt demapper_out.txt

The results will be written into these files.

Set the environment variable and start the system:

.. code-block:: bash

    GNB_EXTRA_OPTIONS=--loader.demapper.shlibversion _capture

Start the system:

.. code-block:: bash

    ./scripts/start_system.sh rfsim

Verify the plugin is loaded by checking the gNB logs:

.. code-block:: text

    [LOADER] library libdemapper_capture.so successfully loaded

Inspect the captured data via:

.. code-block:: bash

    cat plugins/data_acquisition/logs/demapper_in.txt
    # Output: timestamps, modulation, IQ values...

    cat plugins/data_acquisition/logs/demapper_out.txt
    # Output: timestamps, modulation, LLR values...

TensorRT Neural Demapper
------------------------

Build the TensorRT engine using the ``plugins/neural_demapper/scripts/build-trt-plans.sh`` script. This is done automatically during installation of the Sionna Research Kit.

Run the neural demapper inference using TensorRT by setting the environment variable:

.. code-block:: bash

    # we limit the MCS indices to 10 in order to stay within the 16-QAM modulation order
    GNB_EXTRA_OPTIONS=--loader.demapper.shlibversion _trt --MACRLCs.[0].dl_max_mcs 10 --MACRLCs.[0].ul_max_mcs 10

It will automatically load the TRT engine as defined in ``plugins/neural_demapper/config/demapper_trt.config``.

Start the system:

.. code-block:: bash

    ./scripts/start_system.sh rfsim

Verify the plugin is loaded by checking the gNB logs:

.. code-block:: text

    [LOADER] library libdemapper_trt.so successfully loaded
    Initializing TRT demapper (TID 20)
    Initializing TRT runtime 20

Neural Receiver
---------------

Build the TensorRT engine using the ``plugins/neural_receiver/scripts/build-trt-plans.sh`` script. This is automatically done during installation of the Sionna Research Kit.

Run the neural receiver inference using TensorRT by setting the environment variable:

.. code-block:: bash

    # we limit the MCS indices to 10 in order to stay within the 16-QAM modulation order
    GNB_EXTRA_OPTIONS=--loader.receiver.shlibversion _trt --MACRLCs.[0].dl_max_mcs 10 --MACRLCs.[0].ul_max_mcs 10

Start the system:

.. code-block:: bash

    ./scripts/start_system.sh rfsim

Verify the plugin is loaded by checking the gNB logs:

.. code-block:: text

    [LOADER] library libreceiver_trt.so successfully loaded
    Initializing TRT receiver (TID 20)
    Initializing TRT runtime 20

If the receiver is running, you can also see the live statistics in the gNB logs. Note that this requires traffic to be scheduled on the PUSCH, i.e., run iperf3 on the UE side.
