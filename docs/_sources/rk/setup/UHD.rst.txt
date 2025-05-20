.. _uhd:

USRP Driver Installation
========================

We now need to install the Ettus UHD driver to properly access the USRP devices.
These need to be re-build for the Jetson platform.

Key Resources
-------------

The installation guide aims to be self-contained. However, the following resources are useful for troubleshooting and developers working with the Ettus UHD driver:

* `Getting Started Guide <https://kb.ettus.com/B200/B210/B200mini/B205mini_Getting_Started_Guides>`_
* `Build Guide <https://files.ettus.com/manual/page_build_guide.html>`_
* `USB & Transport Guide <https://files.ettus.com/manual/page_transport.html>`_
* `Verification Guide <https://kb.ettus.com/Verifying_the_Operation_of_the_USRP_Using_UHD_and_GNU_Radio>`_

.. note::
   This is also achieved using the script ``scripts/install-usrp.sh``

Prerequisites
-------------

We need to install the following packages to build the UHD driver.

.. code-block:: bash

   sudo apt-get install \
       autoconf automake build-essential ccache cmake cpufrequtils \
       doxygen ethtool g++ git inetutils-tools libboost-all-dev \
       libncurses5 libncurses5-dev libusb-1.0-0 libusb-1.0-0-dev \
       libusb-dev python3-dev python3-mako python3-numpy python3-requests \
       python3-scipy python3-setuptools python3-ruamel.yaml ninja-build


.. note::
   DPDK support is disabled by default.


Building UHD
------------

For arm64, we need to build the UHD driver from source.

Get source code:

.. code-block:: bash

   git clone https://github.com/EttusResearch/uhd.git
   cd uhd/host

Configure and build:

.. code-block:: bash

   mkdir build && cd build
   cmake -DCMAKE_POLICY_DEFAULT_CMD0167=NEW -GNinja ..
   ninja
   ninja test
   sudo ninja install

Download FPGA images:

.. code-block:: bash

   sudo /usr/local/lib/uhd/utils/uhd_images_downloader.py

Post-build Configuration
------------------------

We now need to access the configuration files to properly configure the USRP devices and grant access to the USB devices.

Re-build the linker cache:

.. code-block:: bash

   sudo ldconfig

Set up USB permissions for non-root users:

.. code-block:: bash

   sudo cp /usr/local/lib/uhd/utils/uhd-usrp.rules /etc/udev/rules.d/
   sudo udevadm control --reload-rules
   sudo udevadm trigger

Testing Installation
--------------------

We can now verify the installation and access the USRP serial number.

Check USB device detection:

.. code-block:: bash

   lsusb

   Bus 001 Device 001: ID 1d6b:0002 Linux Foundation 2.0 root hub
   ...
   Bus 001 Device 067: ID 2500:0020 Ettus Research LLC USRP B210

Search for UHD devices:

.. code-block:: bash

   uhd_find_devices

   [INFO] [UHD] linux; GNU C++ version 14.1.1 20240720; Boost_108300; UHD_4.7.0.0-0-ga5ed1872
   --------------------------------------------------
   -- UHD Device 0
   --------------------------------------------------
   Device Address:
     serial: 32FCXXX
     name: MyB210
     product: B210
     type: b200

.. note::
   Please notice the serial number. It will be used to identify the USRP device in the 5G stack.

Get device details:

.. code-block:: bash

   sudo uhd_usrp_probe

   [INFO] [UHD] linux; GNU C++ version 14.1.1 20240720; Boost_108300; UHD_4.7.0.0-0-ga5ed1872
   [INFO] [B200] Loading firmware image: /usr/local/share/uhd/images/usrp_b200_fw.hex...
   ...
   [INFO] [B200] Actually got clock rate 16.000000 MHz.
   ...
   |       Device: B-Series Device
   |     _____________________________________________________
   |    /
   |   |       Mboard: B210
   |   |   serial: 32FCXXX
   |   |   name: MyB210
   |   |   product: 2
   |   |   revision: 4
   |   |   FW Version: 8.0
   |   |   FPGA Version: 16.0
   |   |
   |   |   Time sources:  none, internal, external, gpsdo
   |   |   Clock sources: internal, external, gpsdo
   |   |   Sensors: ref_locked
   ...

And run a final performance test:

.. code-block:: bash

   /usr/local/lib/uhd/examples/benchmark_rate --rx_rate 10e6 --tx_rate 10e6

   [INFO] [UHD] linux; GNU C++ version 14.1.1 20240720; Boost_108300; UHD_4.7.0.0-0-ga5ed1872
   ...
   Benchmark rate summary:
   Num received samples:     100235043
   Num dropped samples:      0
   Num overruns detected:    0
   Num transmitted samples:  100062000
   Num sequence errors (Tx): 0
   Num sequence errors (Rx): 0
   Num underruns detected:   0
   Num late commands:        0
   Num timeouts (Tx):        0
   Num timeouts (Rx):        0

   Done!
