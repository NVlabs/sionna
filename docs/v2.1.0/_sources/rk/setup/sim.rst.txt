.. _sim:

SIM Card Programming
====================

To connect commercial-off-the-shelf (COTS) user equipment (UE), a SIM card needs to be programmed with specific authentication and identification parameters for network access.

We recommend using `pySim <https://github.com/osmocom/pysim>`_ with a PC/SC-compatible reader and a programmable SIM, USIM, or ISIM card (for example, Sysmocom cards). The legacy Open Cells UICC programming tool is documented below for reference, but Open Cells has discontinued sales of its SIM cards and programmers, and the tool is no longer actively maintained.

.. note::
   The parameters (``key``, ``opc``, ``spn``, IMSI, and so on) must match your core network configuration, particularly the Access Management Function (AMF) settings. These values should align with the core network setup defined in the AMF. In OAI tutorials, these parameters are typically configured in an SQL file that initializes the network. Make sure the ``key``, ``opc``, and ``spn`` values match your network configuration, and verify that the IMSI number is registered in your OAI database (``config/common/oai_db.sql``).

.. _sim-pysim:

Programming with pySim
----------------------

`pySim <https://github.com/osmocom/pysim>`_ is an open-source tool for reading and programming programmable SIM, USIM, and ISIM cards. It works with standard PC/SC card readers and third-party programmable cards, and has been successfully used with Sysmocom ISIM cards using the same network parameters as in this guide. See the `pySim wiki <https://osmocom.org/projects/pysim/wiki>`_ for further documentation.

Prerequisites
~~~~~~~~~~~~~

You will need:

* A programmable SIM, USIM, or ISIM card (for example, from `Sysmocom <https://www.sysmocom.de/products/lab/sysmousim/>`_)
* A PC/SC-compatible card reader
* Python 3 and the pySim dependencies

pySim Installation
~~~~~~~~~~~~~~~~~~~~

The following steps can be done on the Ubuntu host machine.

Install system dependencies and clone pySim:

.. code-block:: bash

   sudo apt-get install --no-install-recommends \
       pcscd libpcsclite-dev \
       python3 python3-setuptools python3-pycryptodome python3-pyscard python3-pip \
       pcsc-tools

   git clone https://gitea.osmocom.org/sim-card/pysim.git
   cd pysim
   pip3 install --user -r requirements.txt

Ensure the PC/SC daemon is running and detect your reader:

.. code-block:: bash

   sudo systemctl start pcscd
   pcsc_scan

Note the reader index (for example, ``0``) for use with ``pySim-shell``.

Programming SIM Card
~~~~~~~~~~~~~~~~~~~~

Insert the SIM card into the reader, then start the interactive shell:

.. code-block:: bash

   ./pySim-shell.py -p 0

Replace ``0`` with the reader index reported by ``pcsc_scan`` if needed.

Verify the administrator PIN (ADM). The default ADM varies by card vendor; the example below uses ``12345678``, but sysmocom provides a custom ADM per card:

.. code-block:: text

   pySIM-shell (00:MF)> verify_adm 12345678

Read the current card contents (optional):

.. code-block:: text

   pySIM-shell (00:MF)> select MF
   pySIM-shell (00:MF)> select ADF.USIM
   pySIM-shell (00:MF/ADF.USIM)> select EF.IMSI
   pySIM-shell (00:MF/ADF.USIM/EF.IMSI)> read_binary_decoded

Now program the SIM card with the network parameters. In this example, ``262`` represents Germany and ``99`` is unassigned. These parameters are used for a test UE in the OAI tutorial setup.

Set the IMSI:

.. code-block:: text

   pySIM-shell (00:MF)> select MF/ADF.USIM/EF.IMSI
   pySIM-shell (00:MF/ADF.USIM/EF.IMSI)> edit_binary_decoded

In the editor, set ``imsi`` to ``262990100016069``, save, and exit.

Set the authentication key and OPc (Milenage algorithm):

.. code-block:: text

   pySIM-shell (00:MF)> select MF/ADF.USIM/EF.USIM_AUTH_KEY
   pySIM-shell (00:MF/ADF.USIM/EF.USIM_AUTH_KEY)> edit_binary_decoded

In the editor, set the contents to (adapt key and op_opc to match your network configuration):

.. code-block:: json

   {
     "cfg": {
       "only_4bytes_res_in_3g": false,
       "sres_deriv_func_in_2g": 1,
       "use_opc_instead_of_op": true,
       "algorithm": "milenage"
     },
     "key": "fec86ba6eb707ed08905757b1bb44b8f",
     "op_opc": "c42449363bbad02b66d16bc975d77cc1"
   }

Set the home PLMN to match the IMSI:

.. code-block:: text

   pySIM-shell (00:MF)> select MF/ADF.USIM
   pySIM-shell (00:MF/ADF.USIM)> activate_file EF.EHPLMN
   pySIM-shell (00:MF/ADF.USIM)> select EF.EHPLMN
   pySIM-shell (00:MF/ADF.USIM/EF.EHPLMN)> edit_binary_decoded

In the editor, set ``mcc`` to ``262`` and ``mnc`` to ``99``, save, and exit.

Set the MSISDN, access control class, and service provider name:

.. code-block:: text

   pySIM-shell (00:MF)> select MF/ADF.USIM/EF.MSISDN
   pySIM-shell (00:MF/ADF.USIM/EF.MSISDN)> read_records_decoded
   pySIM-shell (00:MF/ADF.USIM/EF.MSISDN)> edit_record_decoded 1

In the editor, set the MSISDN number to ``00000001``, save, and exit. This is the whole record:

.. code-block:: json

   {
      "alpha_id": "",
      "len_of_bcd": 5,
      "ton_npi": {
         "ext": true,
         "type_of_number": "unknown",
         "numbering_plan_id": "isdn_e164"
      },
      "dialing_nr": "00000001"
   }

.. code-block:: text

   pySIM-shell (00:MF)> select MF/ADF.USIM/EF.ACC
   pySIM-shell (00:MF/ADF.USIM/EF.ACC)> edit_binary_decoded

In the editor, enable the access control class 1 (``ACC1``) and disable the rest, save, and exit.

.. code-block:: text

   pySIM-shell (00:MF)> select MF/ADF.USIM/EF.SPN
   pySIM-shell (00:MF/ADF.USIM/EF.SPN)> edit_binary_decoded

In the editor, set the service provider name to ``OpenAirInterface``, save, and exit.

Verify the programmed values:

.. code-block:: text

   pySIM-shell (00:MF)> select MF/ADF.USIM/EF.IMSI
   pySIM-shell (00:MF/ADF.USIM/EF.IMSI)> read_binary_decoded
   pySIM-shell (00:MF)> select MF/ADF.USIM/EF.USIM_AUTH_KEY
   pySIM-shell (00:MF/ADF.USIM/EF.USIM_AUTH_KEY)> read_binary_decoded
   pySIM-shell (00:MF)> select MF/ADF.USIM/EF.SPN
   pySIM-shell (00:MF/ADF.USIM/EF.SPN)> read_binary_decoded

Programming with Open Cells UICC (Deprecated)
---------------------------------------------

.. warning::
   Open Cells has discontinued sales of its SIM cards and USB programmers. The UICC programming tool was primarily tested with Open Cells hardware and may not work reliably with PC/SC readers or third-party cards. This workflow is retained for reference only; use :ref:`Programming with pySim <sim-pysim>` instead.

The following walks through programming SIM cards using the `Open Cells <https://open-cells.com>`_ UICC programming tool.

Prerequisites
~~~~~~~~~~~~~

You will need the `UICC software and a USB SIM card programmer <https://open-cells.com/index.php/uiccsim-programing/>`_ from Open Cells. Ensure ``make`` and ``gcc`` are installed.

UICC Software Setup
~~~~~~~~~~~~~~~~~~~

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
~~~~~~~~~~~~~~~~~~~~

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

* `pySim wiki <https://osmocom.org/projects/pysim/wiki>`_
* `pySim repository <https://github.com/osmocom/pysim>`_
* `pySim user manual <https://downloads.osmocom.org/docs/pysim/master/html/index.html>`_
* `UICC Programming Guide <https://open-cells.com/index.php/uiccsim-programing/>`_ (deprecated)
* `PLMN Information <https://en.wikipedia.org/wiki/Public_land_mobile_network>`_
* `Mobile Network Codes <https://en.wikipedia.org/wiki/Mobile_country_code>`_
