.. _run_tutorials:

Running the Tutorials
=====================

This page serves as a quick reference guide for testing the precompiled tutorials. For each tutorial, it provides a concise list of useful commands, summarizes the required configuration changes, and shows the expected gNB log output. For more detailed explanations, we refer to each individual :ref:`tutorials` description.

Command Cheat-sheet
-------------------

Try the tutorials in the rfsimulator mode via:

.. code-block::

    # start a configuration
    ./scripts/start_system.sh rfsim_arm64   # or any other config in configs/

    # stop a configuration
    ./scripts/stop_system.sh rfsim_arm64

    # list of containers running
    cd config/rfsim_arm64
    docker compose ps

    # check gnb logs
    docker compose logs oai-gnb     # add -f to keep log open

    # watch CPU core load
    jtop -p 3

    # watch GPU load
    jtop -p 2

GPU-Accelerated LDPC
--------------------

In the `.env` file, add the following line:

.. code-block::

    GNB_EXTRA_OPTIONS=--loader.ldpc.shlibversion _cuda --thread-pool 6,7,8,9,10,11

And run the configuration:

.. code-block::

    $ ./scripts/start_system.sh rfsim_arm64

    Starting 5G Core network
    [+] Running 7/7
    ✔ Network oai-traffic-net  Created                                                                                 0.0s
    ✔ Network oai-public-net   Created                                                                                 0.1s
    ✔ Container oai-mysql      Started                                                                                 0.3s
    ✔ Container oai-amf        Started                                                                                 0.4s
    ✔ Container oai-smf        Started                                                                                 0.6s
    ✔ Container oai-upf        Started                                                                                 0.9s
    ✔ Container oai-ext-dn     Started                                                                                 1.2s
    Waiting for oai-mysql to be healthy (Timeout: 60s)...
    oai-mysql is ready!
    Waiting for oai-amf to be healthy (Timeout: 60s)...
    oai-amf is ready!
    Waiting for oai-smf to be healthy (Timeout: 60s)...
    oai-smf is ready!
    Waiting for oai-upf to be healthy (Timeout: 60s)...
    oai-upf is ready!
    Waiting for oai-ext-dn to be healthy (Timeout: 60s)...
    oai-ext-dn is ready!
    All services are up and healthy!
    Starting gNB
    [+] Running 6/6
    ✔ Container oai-mysql   Running                                                                                    0.0s
    ✔ Container oai-amf     Running                                                                                    0.0s
    ✔ Container oai-smf     Running                                                                                    0.0s
    ✔ Container oai-upf     Running                                                                                    0.0s
    ✔ Container oai-ext-dn  Running                                                                                    0.0s
    ✔ Container oai-gnb     Started                                                                                    0.5s
    Waiting for oai-gnb to be healthy (Timeout: 60s)...
    oai-gnb is ready!
    gNB ready to connect
    Starting nr-ue
    [+] Running 7/7
    ✔ Container oai-mysql   Running                                                                                    0.0s
    ✔ Container oai-amf     Running                                                                                    0.0s
    ✔ Container oai-smf     Running                                                                                    0.0s
    ✔ Container oai-upf     Running                                                                                    0.0s
    ✔ Container oai-ext-dn  Running                                                                                    0.0s
    ✔ Container oai-gnb     Running                                                                                    0.0s
    ✔ Container oai-nr-ue   Started                                                                                    0.5s
    Waiting for oai-nr-ue to be healthy (Timeout: 60s)...
    oai-nr-ue is ready!
    5G network is ready to connect!


Inspect oai-gnb output:

.. code-block::

    docker logs -f oai-gnb

    > ...
    > oai-gnb  | [LOADER] library libldpc_cuda.so successfully loaded
    > ...


Demapper Plugin
---------------

In the `.env` file, add the following line:

.. code-block::

    GNB_EXTRA_OPTIONS=--loader.demapper.shlibversion _orig


Inspect oai-gnb output via:

.. code-block::

    docker logs -f oai-gnb

    > ...
    > oai-gnb  | [LOADER] library libdemapper_orig.so successfully loaded
    > ...

Data Capture Plugin
-------------------

In the `.env` file, add the following line:

.. code-block::

    GNB_EXTRA_OPTIONS=--loader.demapper.shlibversion _capture --thread-pool 6

In the `docker-compose.override.yaml` file, add the following:

.. code-block::
    :caption: docker-compose.override.yaml
    :linenos:

    services:
        oai-gnb:
            volumes:
            ##### Data capture tutorial; ensure that .txt files exist and have the right permissions (666)
            - ../../logs/demapper_in.txt:/opt/oai-gnb/demapper_in.txt
            - ../../logs/demapper_out.txt:/opt/oai-gnb/demapper_out.txt

Create the files before starting the containers:

.. code-block::

    # Create the logs directory
    mkdir -p logs

    # Create the files
    touch logs/demapper_in.txt
    touch logs/demapper_out.txt

    # Make the files writable by the container
    chmod 666 logs/demapper_in.txt
    chmod 666 logs/demapper_out.txt

Inspect oai-gnb output:

.. code-block::

    docker logs -f oai-gnb

    > ...
    > oai-gnb  | [LOADER] library libdemapper_capture.so successfully loaded
    > ...

You can access the data captured after running the system:

.. code-block::

    cat logs/demapper_in.txt

    0.000000001     # time resolution of system
    19412.279803840 # timestamp
    QPSK            # modulation format
    96              # number of I/Q symbols to follow
    -21 -6          # I/Q values
    48 7            # number of RE in the PDCCH
    0 -29           # I/Q values
    -36 116
    -17 37
    -30 34

.. code-block::

    cat logs/demapper_out.txt

    0.000000001     # time resolution of system
    19412.279803840 # timestamp
    QPSK            # modulation format
    96              # number of LLRs to follow
    -3 -1           # LLRs (demapper's output)
    6 0
    0 -4
    -5 14
    -3 4
    -4 4

TensorRT Neural Demapper
------------------------

In the `.env` file, add the following line:

.. code-block::

    GNB_EXTRA_OPTIONS=--loader.demapper.shlibversion _trt --MACRLCs.[0].dl_max_mcs 10 --MACRLCs.[0].ul_max_mcs 10 --thread-pool 6,7,8,9,10,11

In the `docker-compose.override.yaml` file, add the following:

.. code-block::

    services:
        oai-gnb:
            volumes:
            ##### neural demapper tutorial; mount weights and trtengine config
            - ../../tutorials/neural_demapper/models/:/opt/oai-gnb/models
            - ./demapper_trt.config:/opt/oai-gnb/demapper_trt.config

Inspect oai-gnb output:

.. code-block::

    docker logs -f oai-gnb

    > ...
    > oai-gnb  | [LOADER] library libdemapper_trt.so successfully loaded
    > oai-gnb  | Initializing TRT demapper (TID 20)
    > oai-gnb  | Initializing TRT runtime 20
    > oai-gnb  | Loading TRT engine models/neural_demapper.2xfloat16.plan (normalized inputs: 1)
    > ...
