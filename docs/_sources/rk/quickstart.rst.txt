.. _quickstart:

Quickstart
==========

By following this guide, you can set-up your private software-defined 5G network in an afternoon using the Sionna Research Kit. The system allows you to connect commercial off-the-shelf (COTS) user equipment (UE), either via cable or over-the-air.

.. warning::
    Ensure compliance with local RF spectrum regulations before conducting over-the-air experiments.

.. note::
   If you have already configured your Jetson system with Jetpack R36.3 and if you are experienced with OAI, the shortest path to get started is:

   .. code-block:: bash

      # Checkout the Github repository
      git clone https://github.com/NVlabs/sionna-rk.git
      cd sionna-rk

      # Prepare system
      make prepare-system

      # Reboot
      sudo reboot

      # Build Sionna RK images and configs
      make sionna-rk

      # You can now start the end-to-end system in the rf-simulator mode
      ./scripts/start_system rfsim_arm64

      # Or connect your USRP and run real transmissions.
      # Note that you need to modify the .env file in the
      # config/b200_arm64/ directory to set your USRP serial number
      ./scripts/start_system b200_arm64

What follows is a more detailed sequence of steps, including configuration details.

Hardware Requirements
---------------------

.. _figure_system_setup:

.. figure:: figs/system_setup_diagram.png
   :align: center
   :alt: Overview of the deployed setup.
   :width: 520px
   :name: system_overview

   Overview of the deployed setup. See `Ettus OAI reference architecture <https://kb.ettus.com/OAI_Reference_Architecture_for_5G_and_6G_Research_with_USRP>`_ for details.

:numref:`figure_system_setup` shows the setup of the Sionna Research Kit consisting of a USRP, a Quectel modem, and a Jetson AGX Orin. In the following, we will guide you through the steps to setup the system. The following components are required to run the Sionna Research Kit. Please refer to the :ref:`bom` for detailed specifications:

* `NVIDIA Jetson AGX Orin <https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/>`_ with an NVMe SSD
* `Ettus USRP B210/B205mini <https://www.ettus.com/product-categories/usrp-bus-series/>`_
* `Quectel RM520N-GLAA module <https://www.quectel.com/product/5g-rm520n-series/>`_
* `Programmable SIM cards <https://open-cells.com/index.php/sim-cards/>`_
* Ubuntu host machine for flashing of the Jetson and configuring the Quectel modem
* RF cables, splitters/combiners, attenuators, and/or antennas

Connect the components as shown in :numref:`figure_system_setup`.
This means that the Jetson board is connected to the USRP and the Quectel modem is connected to the host machine. Connect the UE and the USRP via RF cables and corresponding splitters/combiners. Note that an attenuator in-between the USRP and the UE is strongly recommended to protect the USRP from high power levels.


Step 1: Jetson Setup
--------------------

Per default, the Jetson board does not have the latest Jetpack version pre-installed. For performance reasons, we recommend using the NVMe SSD of the Jetson board.

Flashing the Jetson board requires an external Ubuntu host machine (no virtual machine supported). Note that this requires Ubuntu 20.04 or 22.04 while Ubuntu 24.04 is not supported.

The following code must be run on the host machine.
Set environment variables:

.. code-block:: bash

   # Set values for r36.3 and the AGX Orin development kit 64gb
   export L4T_RELEASE_PACKAGE=jetson_linux_r36.3.0_aarch64.tbz2
   export SAMPLE_FS_PACKAGE=tegra_linux_sample-root-filesystem_r36.3.0_aarch64.tbz2
   export BOARD=jetson-agx-orin-devkit

`Flash the Jetson <https://docs.nvidia.com/jetson/archives/r36.3/DeveloperGuide/IN/QuickStart.html#to-flash-the-jetson-developer-kit-operating-software>`_ from the Ubuntu host machine:

.. code-block:: bash

   # Prepare files
   mkdir jetson-flash && cd jetson-flash

   # Download and extract L4T R36.3
   wget https://developer.nvidia.com/downloads/embedded/l4t/r36_release_v3.0/release/jetson_linux_r36.3.0_aarch64.tbz2
   wget https://developer.nvidia.com/downloads/embedded/l4t/r36_release_v3.0/release/tegra_linux_sample-root-filesystem_r36.3.0_aarch64.tbz2

   # Extract the files
   tar xf ${L4T_RELEASE_PACKAGE}
   sudo tar xpf ${SAMPLE_FS_PACKAGE} -C Linux_for_Tegra/rootfs/
   cd Linux_for_Tegra/

   # Run scripts
   sudo ./tools/l4t_flash_prerequisites.sh
   sudo ./apply_binaries.sh

   # Connect Jetson to host machine via USB-C cable
   # Active the Recovery mode of the Jetson:
   # Press and hold recovery button, press start button, release recovery button

   # Flash to NVMe SSD
   sudo ./tools/kernel_flash/l4t_initrd_flash.sh --external-device nvme0n1p1 \
   -c tools/kernel_flash/flash_l4t_t234_nvme.xml \
   --showlogs --network usb0 jetson-agx-orin-devkit external

   # Or flash to integrated eMMC (if no NVMe SSD is available)
   sudo ./flash.sh jetson-agx-orin-devkit internal


You can now boot the Jetson and run the following steps on the Jetson board to install the prerequisites:

.. code-block:: bash

   sudo apt update
   sudo apt install -y git

Clone the repository:

.. code-block:: bash

   git clone https://github.com/NVlabs/sionna-rk.git
   cd sionna-rk

.. note::
   The following steps can also be invoked with:

   .. code-block:: bash

      make prepare-system


Configure the system:

.. code-block:: bash

   ./scripts/configure-system.sh

Build custom kernel with SCTP support (see :ref:`kernel`):

.. code-block:: bash

   ./scripts/build-custom-kernel.sh
   ./scripts/install-custom-kernel.sh
   sudo reboot

Install Sionna and TensorFlow with GPU support:

.. code-block:: bash

   # You can install the TF GPU package via
   python3 -m pip install --user --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v60 tensorflow==2.15.0+nv24.05

   # Install sionna (without ray tracing dependencies)
   python3 -m pip install --user sionna-no-rt


Step 2: USRP Setup
------------------


Install UHD drivers and verify the USRP connection:

.. code-block:: bash

   # Run install script
   ./scripts/install-usrp.sh

   # Verify connection
   uhd_find_devices
   uhd_usrp_probe

.. note::
    Make note of your USRP's serial number - you'll need it later for configuration.

.. note::
   Sometimes OAI get confused if the USRP firmware is not loaded. In that case, run one of the uhd utilities to load the default firmware in the device, and retry.

Step 3: UE Setup
----------------

The next step is to program the SIM card (see :ref:`sim`).
Connect the SIM card programmer, download the ``program_uicc`` tool from `here <https://open-cells.com/index.php/sim-cards/>`_ and run the following commands:

.. code-block:: bash

   # Note that the IMSI must be registered in /config/common/oai_db.sql
   sudo ./program_uicc --adm 12345678 --imsi 262990100016069 \
       --key fec86ba6eb707ed08905757b1bb44b8f \
       --opc C42449363BBAD02B66D16BC975D77CC1

Insert SIM card into the Quectel modem and configure the modem on the host machine (see :ref:`quectel`):

.. code-block:: bash

   sudo mmcli -m 0 --enable
   sudo nmcli c add type gsm ifname cdc-wdm0 con-name oai apn oai connection.autoconnect yes

Step 4: Deploy 5G Stack
-----------------------

.. _figure_5g_stack:

.. figure:: figs/5g_stack.png
   :align: center
   :alt: 5G Stack Overview

   Overview of the deployed 5G end-to-end stack with IP adresses and interfaces of each container. Figure from `OpenAirInterface <https://gitlab.eurecom.fr/oai/openairinterface5g/-/blob/develop/ci-scripts/yaml_files/5g_rfsimulator/README.md#2-deploy-containers>`_.

:numref:`figure_5g_stack` shows the block diagram of the complete system (see `OpenAirInterface5G guide <https://gitlab.eurecom.fr/oai/openairinterface5g/-/blob/develop/ci-scripts/yaml_files/5g_rfsimulator/README.md#2-deploy-containers>`_ for more details). The 5G stack is deployed as a set of Docker containers. The following steps build and deploy the core network components as well as the RAN components:

.. note::
   The following steps can also be invoked with:

   .. code-block:: bash

      make sionna-rk

.. code-block:: bash

   # Pull and patch OAI Core 5G network, build containers
   ./scripts/quickstart-cn5g.sh

   # Pull, patch and build OAI containers
   ./scripts/quickstart-oai.sh

   # Generate config files
   ./scripts/generate-configs.sh

   # Update TensorRT plans for current platform
   ./tutorials/neural_demapper/build-trt-plans.sh

You need to configure the environment using your individual configuration:

.. code-block:: bash

   cp config/common/.env.template config/b200_arm64/.env

Edit `.env` file in `config/b200_arm64/` and set the following parameters:

   * Set your USRP serial number
   * Select the configuration file for desired number of PRBs (default is 24, equals 8.64MHz bandwidth)

And finally, you can start the system:

.. code-block:: bash

   # For real hardware setup using the USRP
   ./scripts/start_system.sh b200_arm64

   # Or for RF simulations without connecting to real hardware
   ./scripts/start_system.sh rfsim_arm64


Monitor the system:

.. code-block:: bash

   # Show running containers
   docker ps -a

   # View gNB logs
   docker logs -f oai-gnb

The docker containers should be all in a "healthy" state and the gNB log should
indicate that the UE is successfully connected (``in-sync``).

Have Your First Call
--------------------

Congratulations, your system is now running and ready for your own experiments!

Verify connectivity on the host machine (using the Quectel modem):

.. code-block:: bash

   # Check connection status
   nmcli connection show

   # Check that ip address is assigned to wwan0
   ip addr show wwan0

   # Test internet connectivity through the 5G tunnel
   ping -I wwan0 google.com

This has been successfully tested on Raspberry Pi OS, for other distributions the Quectel QConnectManager might need to be installed manually.

Monitor system load on the Jetson:

.. code-block:: bash

   jtop


And run performance tests:

.. code-block:: bash

   # Start iperf3 server in Docker container
   docker exec -d oai-ext-dn iperf3 -s

   # On the client (UE); you need to install iperf3 on the host machine
   # Downlink test
   iperf3 -u -t 10 -i 1 -b 1M -B 12.1.1.2 -c 192.168.72.135 -R

   # Uplink test
   iperf3 -u -t 10 -i 1 -b 1M -B 12.1.1.2 -c 192.168.72.135

   # Change 1M to the desired throughput in Mbit/s

You can now have your first call over your private 5G network!

We hope that you have enjoyed this quickstart guide! For inspiration and as blueprint for your own experiments, you can now try the following precompiled tutorials:

* Explore the :ref:`accelerated_ldpc` tutorial and learn about accelerated RAN
* Learn about :ref:`data_acquisition` to generate training data for your own AI models
* Discover the toolchain from training in Sionna to the :ref:`neural_demapper` in a real 5G network
* Run a :ref:`ue_emulator`

Check the :ref:`tutorials` page for more info.

For a detailed configuration and troubleshooting, see the :ref:`setup` guide or visit the `Github Discussions <https://github.com/NVLabs/sionna/discussions>`_.
