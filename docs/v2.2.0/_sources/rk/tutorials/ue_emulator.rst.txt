.. _ue_emulator:

Software-defined End-to-End 5G Network
======================================

In this tutorial, we will show how the entire end-to-end 5G network can be simulated using software defined user equipment (UE). This allows for the evaluation of novel --- non-standard compliant --- algorithms and protocols.
Such a setup enables you to test and prototype two-sided network functions such as AI/ML-based CSI feedback compression or even custom constellations for `pilotless communications <https://arxiv.org/pdf/2009.05261>`_.

Ensure that you have already built the UE container `oai-nr-ue` (see :ref:`OAI`).

.. note:: For end-to-end RF experiments, this tutorial requires a
    second Jetson device with another USRP connected to it. Alternatively, you can use two USRPs connected to the same Jetson. However, end-to-end simulations can be done by using OAI's `rfsimulator mode <https://gitlab.eurecom.fr/oai/openairinterface5g/-/blob/develop/radio/rfsimulator/README.md>`_.
    In this mode, the UE is connected to the gNB via a simulated RF interface.

Run the gNB
-----------

Before connecting the UE, the gNB needs to be ready to connect.

.. code-block::

    # start the gNB with USRP connected
    ./scripts/start_system.sh b200

    # or start the gNB in rfsimulator mode
    ./scripts/start_system.sh rfsim

Check that the gNB is running correctly

.. code-block::

    docker logs -f oai-gnb

During the initialization procedure, the gNB provides the required UE parameters

.. code-block::

    136387.122571 [PHY] A (nr_common_signal_proced:92) Command line parameters for OAI UE: -C 3319680000 -r 106 --numerology 1 --ssb 516

These parameters depend on the choice of the configuration file of the gNB.

Run the UE
-----------

Set the above UE parameters as ``UE_EXTRA_OPTIONS`` in the `.env` file in the corresponding `config` directory. For the above example with 106 PRBs, the ``UE_EXTRA_OPTIONS`` should be

.. code-block::

    UE_EXTRA_OPTIONS=-r 106 --numerology 1 -C 3319680000

For RF based experiments, we recommend to use the 24 or 51 PRB configurations.

Also set the ``USRP_SERIAL_UE`` to the serial number of the USRP connected to the UE. For cable-based experiments, the two USRPs must be connected as shown in :numref:`figure_system_setup2`.

Instead of using a real sim-card the UE can be configured via the
`config/common/nrue.uicc.conf` file. In case you modify the IMSI, ensure it is registered in the `oai_db.sql`. Otherwise, the UE will not be recognized by the 5G core network.

Note that the `start_system.sh` script will automatically start the UE when the gNB is running. Otherwise, you can start the UE with

.. code-block::

    # load environment variables
    set -a
    source config/b200/.env
    set +a

    # start the UE container
    cd config/common/
    docker compose up -d oai-nr-ue

    # and shutdown the UE
    docker compose stop oai-nr-ue

For RF experiments, this needs to be done on the second Jetson device.

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

Testing in RF Simulator Mode
----------------------------

The OpenAirInterface (OAI) RF simulator enables testing without access to physical radio hardware. This tutorial summarizes the basic usage for the RF simulator. For further details, see the following resources:

* `OAI RF-Simulator Guide <https://gitlab.eurecom.fr/oai/openairinterface5g/-/blob/develop/radio/rfsimulator/README.md>`_
* `Channel Modeling Guide <https://github.com/OPENAIRINTERFACE/openairinterface5g/blob/develop/openair1/SIMULATION/TOOLS/DOC/channel_simulation.md>`_
* `Telnet Usage Guide <https://github.com/OPENAIRINTERFACE/openairinterface5g/blob/develop/common/utils/telnetsrv/DOC/telnetusage.md>`_

Basic Configuration
-------------------

When launching the gNB container, include these parameters in the ``USE_ADDITIONAL_OPTIONS`` parameter of the ``docker-compose.yaml`` file:

.. code-block:: text

   --rfsimulator.options chanmod
   --telnetsrv

On the UE side, activate the RF-Simulator with these parameters:

.. code-block:: text

   --rfsimulator.options chanmod
   --rfsimulator.serveraddr 192.168.71.140 # <gNB_IP_ADDRESS>

You can find an example configuration file in the ``config/rfsim`` folder.

Dynamic Re-configuration
------------------------

The RF simulator supports runtime configuration through Telnet.
See `Telnet Usage Guide <https://github.com/OPENAIRINTERFACE/openairinterface5g/blob/develop/common/utils/telnetsrv/DOC/telnetusage.md>`_ for details.


To control the downlink channel (on the UE):


.. code-block:: bash

   # Connect to UE
   telnet 192.168.71.150 9090

   # View current settings
   channelmod show current

   # View available profiles
   channelmod show predef

   # Set noise power
   channelmod modify 0 noise_power_dB -10

To control the uplink channel (on the gNB), connect to gNB and configure uplink:

.. code-block:: bash

   # Connect to gNB
   telnet 192.168.71.140 9090

   # Set noise power
   channelmod modify 1 noise_power_dB -10


.. note::
   Changes take effect immediately. No restart is required.
