.. _sim:

SIM Card Programming
====================

To connect commercial-off-the-shelf (COTS) user equipment (UE), a SIM card needs to be programmed with specific parameters. This guide walks through programming SIM cards using the `Open Cells <https://open-cells.com>`_ UICC programming tool, which allows configuring the necessary authentication and identification parameters for network access.

Prerequisites
-------------

You will need the `UICC software and a USB SIM card programmer <https://open-cells.com/index.php/uiccsim-programing/>`_ from Open Cells. Ensure ``make`` and ``gcc`` are installed.

UICC Software Setup
-------------------

The following steps can be done on the Ubuntu host machine.

Download `UICC v3.3 <https://open-cells.com/d5138782a8739209ec5760865b1e53b0/uicc-v3.3.tgz>`_ programming tool:

.. code-block:: bash

   wget https://open-cells.com/d5138782a8739209ec5760865b1e53b0/uicc-v3.3.tgz

Extract and compile:

.. code-block:: bash

   # Extract package
   tar xzf uicc-v3.3.tgz

   # Compile
   make clean
   make program_uicc
   make program_uicc_pcsc

Programming SIM Card
--------------------

Insert the SIM card and connect the USB programmer.
Read current values:

.. code-block:: bash

   sudo ./program_uicc --adm 1

Example output:

.. code-block:: text

   Existing values in USIM
   ICCID: 89330061100000000831
   WARNING: iccid luhn encoding of last digit not done
   USIM IMSI: 208920100001831
   PLMN selector: : 0x02f8297c
   Operator Control PLMN selector: : 0x02f8297c
   Home PLMN selector: : 0x02f8297c
   USIM MSISDN: 00000831
   USIM Service Provider Name: open cells
   No ADM code of 8 figures, can't program the UICC

Now let's program the SIM card with the network parameters.

.. note::
   The parameters (``key``, ``opc``, ``spn``) must match your core network configuration, particularly the Access Management Function (AMF) settings. These values should align with the core network setup defined in the AMF. In OAI tutorials, these parameters are typically configured in an SQL file that initializes the network. Make sure the ``key``, ``opc``, and ``spn`` values match your network configuration, and verify that the IMSI number is registered in your OAI database (``config/common/oai_db.sql``).

In this example, 262 represents Germany, and 99 is unassigned. These parameters are used for a test UE in the OAI tutorial setup.

.. code-block:: bash

   sudo ./program_uicc --adm 12345678 \
       --imsi 262990100016069 \
       --isdn 00000001 \
       --acc 0001 \
       --key fec86ba6eb707ed08905757b1bb44b8f \
       --opc C42449363BBAD02B66D16BC975D77CC1 \
       --spn "OpenAirInterface" \
       --authenticate


Example output:

.. code-block:: text

   Setting new values

   Reading UICC values after uploading new values
   ICCID: 89330061100000000831
   USIM IMSI: 262990100016069
   PLMN selector: : 0x02f8997c
   USIM MSISDN: 00000001
   USIM Service Provider Name: OpenAirInterface
   Succeeded to authenticate with SQN: 352
   Set HSS SQN value as: 384


Additional Resources
--------------------

* `UICC Programming Guide <https://open-cells.com/index.php/uiccsim-programing/>`_
* `PLMN Information <https://en.wikipedia.org/wiki/Public_land_mobile_network>`_
* `Mobile Network Codes <https://en.wikipedia.org/wiki/Mobile_country_code>`_
