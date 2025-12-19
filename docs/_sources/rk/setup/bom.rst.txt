.. _bom:

Bill of Materials
=================

This is a summary of the hardware components required to assemble the Sionna Research Kit. The exact choice of the RF components such as the cables is not critical and other components might work as well. You can find a more detailed list of components in the `Ettus OAI reference architecture <https://kb.ettus.com/OAI_Reference_Architecture_for_5G_and_6G_Research_with_USRP>`_.

Computing Platforms
^^^^^^^^^^^^^^^^^^^
.. note::
   The NVIDIA DGX Spark is the recommended computing platform for the Sionna Research Kit. The DGX Spark provides hardware accelerated ray tracing for real-time channel emulation and a ConnectX-7 NIC.

The supported options for the computing platform are:

* `NVIDIA DGX Spark <https://www.nvidia.com/en-us/products/workstations/dgx-spark/>`_
* `NVIDIA Jetson AGX Thor <https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-thor/>`_
* Legacy `NVIDIA Jetson AGX Orin <https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/>`_:
      * M.2 NVMe SSD for additional storage
      * An Ubuntu 22.04 or 24.04 host system is required to flash the Jetson Orin

An optional `NVIDIA Jetson Orin Nano Super Developer Kit <https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/nano-super-developer-kit/>`_ can be added as a second device to run the software-defined UE as described in the :ref:`ue_emulator` tutorial. Note that, this setup requires an additional USRP.

RF Components
^^^^^^^^^^^^^
* `Ettus USRP B210 Kit <https://www.ettus.com/all-products/ub210-kit/>`_ (other USRP models might work as well)
* USB 3.0 cable (included)
* SMA cables:
    * `8x SMA-SMA cable 50cm`
    * `4x PigTail Cable SMA - MHF4, 15cm` (required for the Quectel modem)

* RF attenuators:
    * `1x 40dB attenuator BW-S40W2+ <https://www.minicircuits.com/WebStore/dashboard.html?model=BW-S40W2%2B>`_ or `1x 20dB attenuator BW-S20W2+ <https://www.minicircuits.com/WebStore/dashboard.html?model=BW-S20W2%2B>`_

* RF splitters:
    * `1x 4-way splitter ZN4PD1-63HP-S+ <https://www.minicircuits.com/WebStore/dashboard.html?model=ZN4PD1-63HP-S%2B>`_
    * `1x 2-way splitter ZN2PD2-50-S+ <https://www.minicircuits.com/WebStore/dashboard.html?model=ZN2PD2-50-S%2B&srsltid=AfmBOorsLzwGiXtiTsLyj6kIT3yjKRAr9dGwimSUsAJIGe6aWL-WKOYa>`_

.. note::
   The Ettus B205/206mini USRPs can be used as well, but they only support single antenna experiments, i.e., no MIMO experiments are possible.

User Equipment
^^^^^^^^^^^^^^
* `Quectel 5G RM520N-GL 5G Module <https://www.quectel.com/product/5g-rm520n-series/>`_
* `M.2 to USB adapter <https://shop.tekmodul.de/M-2-USB-modem-carrier-board-standard-edition-p541645344>`_
* `Programmable SIM cards <https://open-cells.com/index.php/sim-cards/>`_
* `SIM card programmer <https://open-cells.com/index.php/sim-cards/>`_
* Host PC with USB 3.0 port; ideally with Ubuntu 22.04 or 24.04

Optional Components
^^^^^^^^^^^^^^^^^^^
* RF shielding box for over-the-air testing (Faraday cage)
* Additional antennas for over-the-air testing
* Spectrum analyzer for signal analysis and debugging
* Second `Ettus USRP B210 <https://www.ettus.com/all-products/ub210-kit/>`_ for software-defined user equipment (see :ref:`ue_emulator`)

