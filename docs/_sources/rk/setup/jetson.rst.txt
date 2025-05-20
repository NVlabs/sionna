Jetson Setup
============

This guide covers the required steps to set up a new `NVIDIA Jetson AGX Orin <https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/>`_. The Jetson runs `Jetson Linux <https://developer.nvidia.com/embedded/jetson-linux>`_, an Ubuntu-based distribution with drivers and utilities optimized for the Jetson hardware.

The installation guide aims to be self-contained. However, the following resources are generally useful for developers working with the NVIDIA Jetson platform:

* `Getting Started Guide <https://developer.nvidia.com/embedded/learn/get-started-jetson-agx-orin-devkit>`_
* `User Guide <https://developer.nvidia.com/embedded/learn/jetson-agx-orin-devkit-user-guide/index.html>`_
* `Developer Guide <https://docs.nvidia.com/jetson/archives/r36.3/DeveloperGuide/index.html>`_

OS Installation
---------------

There are three ways to install/upgrade the Jetson OS. We recommend using the pre-built image.

1. Pre-built Image (Recommended)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. note::
   Requires a host system with Ubuntu 20.04 or 22.04. Will not work on other systems or virtual machines. Ubuntu 24.04 is not supported.


The following code snippet downloads the pre-built image and extracts the packages. It needs to be run on a host system with Ubuntu 20.04 or 22.04.

Export environment variables:

.. code-block:: bash

   # Set values for r36.3 and the AGX Orin development kit 64gb
   export L4T_RELEASE_PACKAGE=jetson_linux_r36.3.0_aarch64.tbz2
   export SAMPLE_FS_PACKAGE=tegra_linux_sample-root-filesystem_r36.3.0_aarch64.tbz2
   export BOARD=jetson-agx-orin-devkit

.. code-block:: bash

   # Download files (may need to authenticate)
   wget https://developer.nvidia.com/downloads/embedded/l4t/r36_release_v3.0/release/jetson_linux_r36.3.0_aarch64.tbz2
   wget https://developer.nvidia.com/downloads/embedded/l4t/r36_release_v3.0/release/tegra_linux_sample-root-filesystem_r36.3.0_aarch64.tbz2


   # Prepare files
   mkdir jetson-flash && cd jetson-flash
   tar xf ../${L4T_RELEASE_PACKAGE}
   sudo tar xpf ../${SAMPLE_FS_PACKAGE} -C Linux_for_Tegra/rootfs/
   cd Linux_for_Tegra/
   sudo ./tools/l4t_flash_prerequisites.sh
   sudo ./apply_binaries.sh

   # Set Jetson in recovery mode. 
   # For the AGX:
   # 1. Ensure that the developer kit is powered off.
   # 2. Press and hold down the Force Recovery button.
   # 3. Press, then release the Power button.
   # 4. Release the Force Recovery button.

   # Connect Jetson to host machine via USB-C cable

   # Flash to eMMC
   sudo ./flash.sh jetson-agx-orin-devkit internal

   # Flash to NVMe
   sudo ./tools/kernel_flash/l4t_initrd_flash.sh --external-device nvme0n1p1 \
   -c tools/kernel_flash/flash_l4t_t234_nvme.xml \
   --showlogs --network usb0 jetson-agx-orin-devkit external

   # Flash to SD card
   sudo ./tools/kernel_flash/l4t_initrd_flash.sh --external-device mmcblk0p1 \
   -c tools/kernel_flash/flash_l4t_t234_nvme.xml \
   --showlogs --network usb0 jetson-agx-orin-devkit external

2. SDK Manager
^^^^^^^^^^^^^^

Download and install the `SDK Manager <https://developer.nvidia.com/sdk-manager>`_ and follow the `SDK Manager Documentation <https://docs.nvidia.com/sdk-manager/install-with-sdkm-jetson/index.html>`_.

.. note::
   These instructions require an Ubuntu Host 20.04 or 22.04, or a CentOS/RedHat system. They will not work on other systems or virtual machines unless you do not use USB connections or tunnel them properly to the VM. The software disconnects multiple times during flashing, which can cause issues on many VMs.



3. SDK Manager Docker Container
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For advanced users, the SDK Manager is also available as a Docker container. See the `Container Documentation <https://docs.nvidia.com/sdk-manager/docker-containers/index.html>`_.

Post-Installation Setup
-----------------------

You should now have a Jetson OS installed on your system. After re-booting, the following commands must be executed on the Jetson device.

Update packages:

.. code-block:: bash

   sudo apt update
   sudo apt dist-upgrade
   sudo apt install git-core git cmake build-essential bc libssl-dev python3-pip

Download the Sionna Research Kit:

.. code-block:: bash


   cd ~ # We assume sionna-rk is cloned in the home directory
   git clone https://github.com/NVlabs/sionna-rk.git


Install and configure Docker:

.. code-block:: bash

   sudo apt install docker.io

   # Replace $USER with the username to add, if not the one logged in
   sudo usermod -aG docker $USER
   # Log out and log in again for changes to take effect

Install the monitoring tool `jetson-stats <https://github.com/rbonghi/jetson_stats>`_:

.. code-block:: bash

   sudo apt install python3 python3-pip
   sudo python3 -m pip install jetson-stats

   # Show system stats
   jtop

Power Mode from the Command Line
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

By default, the Jetson module ships in power mode 2, using maximum power of 30W with only 8 of 12 cores enabled. Modes are defined in `/etc/nvpmodel.conf`. The file is a symlink to the actual board definition in `/etc/nvpmodel/*`. Useful commands are listed below.

Query the current power mode:

.. code-block:: bash

   sudo nvpmodel -q
   NV Power Mode: MODE_30W
   2

Switch to different power modes:

.. code-block:: bash

   sudo nvpmodel -m 0 # Remove power limits
   sudo nvpmodel -m 1 # 15W
   sudo nvpmodel -m 2 # 30W
   sudo nvpmodel -m 3 # 50W

To make these changes persistent across reboots, modify ``/etc/nvpmodel.conf`` to set the default power mode, and create ``/etc/default/cpufrequtils`` to set the CPU governor. This can also be done as follows:

.. code-block:: bash

   sudo sed -i 's|< PM_CONFIG DEFAULT=2 >|< PM_CONFIG DEFAULT=0 >|' /etc/nvpmodel.conf
   echo 'GOVERNOR="performance"' | tee /etc/default/cpufrequtils


Version Information
-------------------

Check OS version:

.. code-block:: bash

   cat /etc/lsb-release
   DISTRIB_ID=Ubuntu
   DISTRIB_RELEASE=22.04
   DISTRIB_CODENAME=jammy
   DISTRIB_DESCRIPTION="Ubuntu 22.04.4 LTS"

Check Jetson Linux & JetPack version:

.. code-block:: bash

   cat /etc/nv_tegra_release
   # R36 (release), REVISION: 3.0, GCID: 36923193, BOARD: generic, EABI: aarch64, DATE: Fri Jul 19 23:24:25 UTC 2024
   # KERNEL_VARIANT: oot
   TARGET_USERSPACE_LIB_DIR=nvidia
   TARGET_USERSPACE_LIB_DIR_PATH=usr/lib/aarch64-linux-gnu/nvidia
