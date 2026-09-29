.. _quectel:

Quectel Modem Setup
===================

This guide covers the configuration and testing of the Quectel 5G modem.
We have tested this guide using a Quectel RM520N-GL 5G modem, other models might work as well but are not tested.

Basic Configuration
-------------------

The modem can be controlled through :ref:`at-commands` or the `ModemManager`.
In case of the `ModemManager`, the following commands can be used to control the modem:

.. code-block:: bash

   # List available modems
   mmcli -L

   # Check modem state
   mmcli -m 0

   # Enable modem
   mmcli -m 0 --enable

   # Disable modem
   mmcli -m 0 --disable

Configure the network connection using the `NetworkManager`:

.. code-block:: bash

   # Create connection for OAI network
   nmcli c add type gsm ifname cdc-wdm0 con-name oai apn oai connection.autoconnect yes

   # Show connection status
   nmcli

   # List all connections
   nmcli connection show

   # Show device details
   nmcli device show


.. note::
   The 5G network is not running yet. Thus, no connection is established.
   The connection will be established when the 5G network is started.

In case of an Ubuntu system, the Quectel QConnectManager is highly recommended.


.. _at-commands:

AT Command Reference
--------------------

A more detailed control of the modem can be achieved using AT commands.
This is not required for the Sionna Research Kit, but can be useful for debugging and more complex experiments.

You can send the AT commands using a serial interface at 115200 bps using *Minicom* or similar. On a Windows machine, this can be done using PuTTY.

Command Syntax
^^^^^^^^^^^^^^

========= ==========================================
Operation Format
========= ==========================================
Execute   AT<command>
Execute   AT<command>=param1,param2,...,paramN
Read      AT<command>?
Test      AT<command>=?
Set       AT<command><n>
Set       AT<command>=value1,value2,...,valueN
========= ==========================================

Basic Commands
^^^^^^^^^^^^^^

=================== =================================================
Command             Description
=================== =================================================
AT                  Test startup
ATEn                Terminal echo (n=0 disable, n=1 enable)
ATI                 Display device information
AT+GMI              Get manufacturer information
AT+GMR              Get manufacturer revision
AT+CIMI             Display SIM IMSI
AT+GSN              Display SIM IMEI
AT&F                Restore factory settings
=================== =================================================

Modem Commands
^^^^^^^^^^^^^^

==================================== =====================================
Command                              Description
==================================== =====================================
AT+CGDCONT                           Set PDP context parameters
AT+CGACT                             Activate PDP context
AT+QCAINFO                           Get carrier aggregation info
AT+QNWINFO                           Get network information
AT+COPS?                             Get network operator
AT+COPS=?                            Get network operator list
AT+QSPN                              Get operator name
AT+CCLK?                             Get network time
AT+CFUN                              Enable/disable modem functionality
AT+GSQ                               Get signal strength
AT+QNWPREFCFG=”nr5g_band”            Display configured 5G bands
AT+QNWPREFCFG=”mode_pref”,nr5g       Set mode to 5G NR SA
AT+QNWPREFCFG=”nr5g_disable_mode”,0  Enable 5G NR operations
AT+CGPADDR=1                         Get UE IP address
AT+QPING                             Ping IP address
==================================== =====================================


Additional Resources
--------------------

* `Quectel RM520N Guide <https://blog.zero-iee.com/en/posts/quectel-rm520n-and-telit-fn990a28-5g-modems-on-raspberrypi-os/>`_



