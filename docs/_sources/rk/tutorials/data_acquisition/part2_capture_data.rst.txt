

Part 2: Capture Data
====================

In this second part, we will extend the demapper plugin module to capture both inputs and outputs of the demapper to files. The captured data can then be used for analysis, training, and testing of a neural demapper implementation. We focus specifically on capturing data for QPSK and 16-QAM modulation schemes, while allowing the default implementation to handle other modulation formats. Rather than reimplementing the QPSK and 16-QAM demapping algorithms, we leverage the existing OpenAirInterface implementations within our plugin.

The following sections detail the modifications required for data capture, including timestamping and multi-threading support, as well as instructions for accessing the collected data from within the containers.

Adding New Plugin Variant
-------------------------

The full code of the capture plugin is in `tutorials/neural_demapper/nr_demapper_capture.c <https://github.com/NVlabs/sionna-rk/blob/main/tutorials/neural_demapper/nr_demapper_capture.c>`_, and also included at the end of this tutorial. To compile this additional variant, we only need to add the following to the ``CMakeLists.txt`` file. The rest is handled by the already present demapper plugin. If you installed the tutorials via the provided installation scripts, this is already included.

.. literalinclude:: CMakeLists.txt
   :language: CMake
   :linenos:
   :start-after: START marker-plugin-cmake-capture
   :end-before: END marker-plugin-cmake-capture

Using New Plugin Variant
------------------------

After compiling the new variant, you can use it by passing the option ``--loader.demapper.shlibversion _capture`` to the executable. Alternatively, you can add the following line to the ``.env`` file in your configuration directory:

.. code-block::
    :name: .env

    GNB_EXTRA_OPTIONS="--loader.demapper.shlibversion _capture"

To verify that the capture variant is being used, check the gNB logs for the following message:

.. code-block::

    [LOADER] library libdemapper_capture.so successfully loaded


Access Captured Files
---------------------

The module captures inputs and outputs to two files: ``demapper_in.txt`` and ``demapper_out.txt``. To access these capture files from the container, you need to mount them to your host system. This can be done by adding volume mappings in your ``docker-compose.override.yaml`` file in the corresponding `config` folder:

.. code-block::

    cp config/common/docker-compose.override.yaml.template config/b200_arm64/docker-compose.override.yaml

And edit the file accordingly:

.. code-block::
    :caption: Mount capture files
    :name: docker-compose.override.yaml

    services:
    oai-gnb:
        volumes:
        - ../../logs/demapper_in.txt:/opt/oai-gnb/demapper_in.txt
        - ../../logs/demapper_out.txt:/opt/oai-gnb/demapper_out.txt

Create the files before running the container:

.. code-block::

    # Create the logs directory
    mkdir -p logs

    # Create the files
    touch logs/demapper_in.txt
    touch logs/demapper_out.txt

    # Make the files writable by the container
    chmod 666 logs/demapper_in.txt
    chmod 666 logs/demapper_out.txt

Note that the host filepaths can be arbitrary, as long as they are accessible from the host system. The files must exist before running the container.

Capture Format
--------------

The captured data is stored in two text files: one containing the input symbols to the demapper and another containing the corresponding demapped output values. This simple text-based format allows for easy inspection and post-processing of the captured data.

The input file format is as follows:

.. literalinclude:: nr_demapper_capture.c
   :language: c
   :linenos:
   :caption: input file format
   :start-after: START marker-capture-input-format
   :end-before: END marker-capture-input-format

.. code-block::
    :caption: Sample input format / demapper_in.txt
    :name: demapper_input.txt

    0.000000001         # Time source resolution (sec.nanosec)
    1373853.185968662   # Timestamp (sec.nanosec)
    QPSK                # Modulation scheme (QPSK or QAM16)
    96                  # Number of symbols (resource elements)
    177 -179            # Input symbols (real, imag)
    -179 176
    -180 -177
    176 178
    -177 -177
    -177 177

And the output file format is as follows:

.. literalinclude:: nr_demapper_capture.c
   :language: c
   :linenos:
   :caption: output file format
   :start-after: START marker-capture-output-format
   :end-before: END marker-capture-output-format

With the following sample output format:

.. code-block::
    :caption: Sample output format / demapper_out.txt
    :name: demapper_output.txt

    0.000000001         # Time source resolution (sec.nanosec)
    1373853.185968662   # Timestamp (sec.nanosec)
    QPSK                # Modulation scheme (QPSK or QAM16)
    96                  # Number of symbols (resource elements)
    22 -23              # Output symbols (llr1, llr2)
    -23 22
    -23 -23
    22 22
    -23 -23
    -23 22

An example to read the input and output files with Python is provided in the :ref:`neural_demapper` tutorial.

Add Timestamps
--------------

Including timestamps in the data capture enables precise analysis of the data stream by improving synchronization and sequence ordering of data blocks. Here are the key considerations when adding timestamps:

**Resolution**: Each file includes the system's clock resolution to handle platforms with varying timing precision.

**Monotonicity**: Rapid captures can result in identical timestamps. To prevent this, we use ``CLOCK_MONOTONIC`` as the clock source, which guarantees strictly increasing timestamps.

**Regularity**: We capture timestamps at the module's entry point directly before processing begins. This ensures the timestamp closely reflects the actual processing time, unaffected by processing duration or write operations.

Account for Multi-Threading
---------------------------

Since this is a multi-threaded application, we need to handle concurrent file writes carefully. Multiple threads will attempt to write data simultaneously, which could lead to corrupted output. Although the data processing is complete by the time we write to files, we still need to synchronize the actual file operations. The solution is straightforward - we use a mutex to ensure only one thread can write at a time. We initialize the mutex in our initialization functions and protect all file write operations with mutex locks:

.. code-block:: c
    :caption: Mutex for file data writes
    :name: capture-mutex-wrapper

    static pthread_mutex_t capture_lock = PTHREAD_MUTEX_INITIALIZER;

    // ...

    void capture_function( ... )
    {
        pthread_mutex_lock( &capture_lock );

        // write to files

        pthread_mutex_unlock( &capture_lock );
    }


Final Source Code
-----------------

The original demapping function remains active while the capture function runs in parallel, allowing data collection without impacting the normal operation of the connection. The complete implementation of the capture plugin is shown below.

.. literalinclude:: nr_demapper_capture.c
   :caption: nr_demapper_capture.c
   :language: c
   :linenos:
   :start-after: START marker-capture-full
   :end-before: END marker-capture-full
