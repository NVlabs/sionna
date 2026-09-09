.. _run_tutorials:

Running the Tutorials
=====================

This is a quick reference guide to run the precompiled tutorials. For in-depth explanations, see the individual :ref:`tutorials` pages.

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

.. note::

   The build script auto-detects the installed TensorRT version. On TensorRT 10 and earlier it builds the engine directly with ``trtexec --fp16``. On TensorRT 11+, where strongly-typed networks are the default and the ``--fp16`` build flag was removed, the script first converts the ONNX model to FP16 with `NVIDIA ModelOpt <https://github.com/NVIDIA/TensorRT-Model-Optimizer>`_ (``python3 -m modelopt.onnx.autocast``) and then builds the engine from the FP16 ONNX. ModelOpt is installed via ``nvidia-modelopt[onnx]`` in ``requirements.txt`` (set up by ``scripts/configure-system.dgx-spark.sh``); the build script activates the ``env`` virtual environment automatically.

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

.. note::

   As with the neural demapper, the build script auto-detects the TensorRT version and, on TensorRT 11+, converts the ONNX model to FP16 with NVIDIA ModelOpt before building the engine. The receiver conversion additionally generates a small calibration sample (``scripts/make_calibration_data.py``) for the ModelOpt reference run and keeps the ``nr_preprocessing`` subgraph in FP32 to preserve integer-exact tensor-shape arithmetic.

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

Link Adaptation
---------------

Replace the default MCS selection algorithm in the MAC scheduler with an :ref:`link_adaptation` plugin. Enable by setting the environment variable (``config/rfsim/.env`` file):

.. code-block:: bash

    # Simple OLLA
    GNB_EXTRA_OPTIONS="--loader.link_adaptation.shlibversion _olla"

    # Advanced OLLA with MCS history
    GNB_EXTRA_OPTIONS="--loader.link_adaptation.shlibversion _mcs_hist_olla"

Start the system:

.. code-block:: bash

    ./scripts/start_system.sh rfsim

Verify the plugin is loaded by checking the gNB logs:

.. code-block:: text

    [LOADER] library liblink_adaptation_olla.so successfully loaded

See :ref:`link_adaptation` for all available variants and configuration details.

Channel Emulation
-----------------

Enable the :ref:`channel_emulation` by setting the environment variable. Use the file-based mode for pre-computed CIRs or the ZMQ mode for interactive use with the SionnaRT GUI:

.. code-block:: bash

    # File-based CIR (pass-through, no distortion)
    GNB_EXTRA_OPTIONS="--cir-folder /opt/oai-gnb/plugins/channel_emulation/data/pass_through_cir"

    # ZMQ-based CIR (interactive, use with SionnaRT GUI)
    GNB_EXTRA_OPTIONS="--cir-zmq-num-taps 48"

Start the system:

.. code-block:: bash

    ./scripts/start_system.sh rfsim

Verify the plugin is loaded by checking the gNB logs:

.. code-block:: text

    [LOADER] library libchn_emu.so successfully loaded
    Channel Emulator initialized

The :ref:`ric_xapps` stats server can be used to visualize UE statistics (MCS, BLER) alongside the channel emulation. It is automatically started and available on port 5555. It is also integrated into the SionnaRT GUI.

The channel emulator works best with the `SionnaRT GUI <https://github.com/NVlabs/sionna-rt-gui>`_ to generate and export CIRs. It is automatically installed when you install the requirements.txt file, and can be started via:

.. code-block:: bash

    sionna-rt-gui --priority --config spark_rfsim.yaml # or spark_quectel.yaml

The GUI uses `NVIDIA MPS <https://docs.nvidia.com/deploy/mps/index.html>`_, which is required when real-time ray tracing runs concurrently with other CUDA plugins on the GPU (see `NVIDIA MPS <https://docs.nvidia.com/deploy/mps/index.html>`_). The active thread percentage can be configured via the ``MPS_ACTIVE_THREAD_PCT`` environment variable (default: 40%). The scripts `./scripts/start_mps.sh` and `./scripts/stop_mps.sh` can be used to start (and stop) MPS before starting the gNB and the GUI.
