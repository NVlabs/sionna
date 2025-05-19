Using RF Simulator Mode
=======================

The OpenAirInterface (OAI) RF simulator enables testing without access to physical radio hardware. This tutorial summarizes the basic usage for the RF simulator. For further details, see the the following resources:

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

You can find an example configuration file in the ``configs/rfsim_arm64`` folder.

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
