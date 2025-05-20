.. _ue_emulator:

Software-defined End-to-End 5G Network
======================================

In this tutorial, we will show how the entire end-to-end 5G network can be simulated using software defined user equipment (UE). This allows for the evaluation of novel --- non-standard compliant --- algorithms and protocols.
This enables you to test and prototype two-sided network functions such as AI/ML-based CSI feedback compression or even custom constellations for `pilotless communications <https://arxiv.org/pdf/2009.05261>`_.

Ensure that you have already build the nrUE container `oai-nr-ue` (see :ref:`OAI`).

.. note:: This tutorial currently only works in the `rfsimulator mode <https://gitlab.eurecom.fr/oai/openairinterface5g/-/blob/develop/radio/rfsimulator/README.md>`_.
    In this mode, the UE is connected to the gNB via a simulated radio interface without the need for actual RF hardware. This allows for the simulation of a fully software defined network.


Run the gNB
-----------

Before connecting the UE, the gNB needs to be ready to connect.
As we are using the `rfsimulator mode <https://gitlab.eurecom.fr/oai/openairinterface5g/-/blob/develop/radio/rfsimulator/README.md>`_, we need to start the gNB using a slightly different configuration file.

.. code-block::

    ./scripts/start_system.sh rfsim_arm64

The main difference in the configuration files is that the USRP is replaced by the simulated channel interface.

Check that the gNB is running correctly

.. code-block::

    docker logs -f oai-gnb

During the initialization procedure, the gNB provides the required UE parameters

.. code-block::

    136387.122571 [PHY] A (nr_common_signal_proced:92) Command line parameters for OAI UE: -C 3319680000 -r 106 --numerology 1 --ssb 516

Ensure this is properly set as ``UE_EXTRA_OPTIONS`` in the `.env` file in the `configs/rfsim_arm64` directory.
These parameters depend on the choice of the configuration file of the gNB.

As the rfsimulator does not run in real-time mode, one can simulate a larger number of PRBs than in the case of the USRP.

Run the UE
-----------

We are now ready to connect the UE to the gNB.

Instead of using a real sim-card the UE can be configured via the
`configs/common/nrue.uicc.conf file. In case you modify the IMSI, ensure it is registered in the the `oai_db.sql`. Otherwise, the UE will not be recognized by the 5G core network.

Ensure that the UE configuration in the `docker-compose.yaml` file is correctly configured. In particular, the additional ``UE_EXTRA_OPTIONS`` parameters in the `.env` file must be correctly aligned with gNB config.
For the above example with 106 PRBs, the ``UE_EXTRA_OPTIONS`` should be

.. code-block::

    UE_EXTRA_OPTIONS=-r 106 --numerology 1 -C 3319680000


Note that the `start_system.sh` script will automatically start the UE when the gNB is running. Otherwise, you can start the UE with

.. code-block::

    cd configs/rfsim_arm64/
    docker compose up -d oai-nr-ue

    # and shutdown the UE
    docker compose stop oai-nr-ue

Verify that the UE is running correctly

.. code-block::

    docker logs -f oai-nr-ue

You should now see the UE connected to the gNB.


Test performance
----------------

Verify that an IP address was assigned

.. code-block::

    docker exec -ti oai-nr-ue ifconfig

This should show a network interface with IP `12.1.1.2`.

Ping an external network

.. code-block::

    docker exec -ti oai-nr-ue ping -I oaitun_ue1 google.com

You can access the current UE stats via

.. code-block::

    docker exec -ti oai-nr-ue cat nrL1_UE_stats-0.log

Or run an end-to-end speed test via

.. code-block::

    docker exec -d oai-ext-dn iperf3 -s

    # Running uplink test (UE to gNB)
    docker exec -ti oai-nr-ue iperf3 -u -t 10 -i 1 -b 5M -B 12.1.1.2 -c 192.168.72.135

    # Running downlink test (gNB to UE)
    docker exec -ti oai-nr-ue iperf3 -u -t 10 -i 1 -b 5M -B 12.1.1.2 -c 192.168.72.135 -R

You can now also run multiple UEs by adding more instances of ``oai-nr-ue`` in the `docker-compose.yaml` file to simulate a multi-user scenario.
Further, the setup can be used to evaluate new algorithms that require a modified transmitter such as the idea of custom constellations for `pilotless communications <https://arxiv.org/pdf/2009.05261>`_.

