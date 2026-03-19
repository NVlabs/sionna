.. _jetson_orin:

Jetson AGX Orin Setup
=====================

This guide covers the required steps to set up an `NVIDIA Jetson AGX Orin <https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/>`_. The Jetson runs NVIDIA Jetson Linux, an Ubuntu-based distribution with drivers and utilities optimized for the Jetson hardware.

The installation guide aims to be self-contained. However, the following resources are generally useful for developers working with the NVIDIA Jetson Orin platform:

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


The following code snippet downloads the pre-built image and extracts the packages. **It needs to be run on a host system with Ubuntu 20.04 or 22.04.**

Export environment variables:

.. code-block:: bash

   # Set values for r36.4.4 and the AGX Orin development kit 64gb
   export L4T_RELEASE_PACKAGE=jetson_linux_r36.4.4_aarch64.tbz2
   export SAMPLE_FS_PACKAGE=tegra_linux_sample-root-filesystem_r36.4.4_aarch64.tbz2
   export BOARD=jetson-agx-orin-devkit


.. code-block:: bash

   # Download files (may need to authenticate)
   wget https://developer.nvidia.com/downloads/embedded/l4t/r36_release_v4.4/release/jetson_linux_r36.4.4_aarch64.tbz2
   wget https://developer.nvidia.com/downloads/embedded/l4t/r36_release_v4.4/release/tegra_linux_sample-root-filesystem_r36.4.4_aarch64.tbz2


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

For advanced users, the SDK Manager is also available as a Docker container. See the `Container Documentation <https://docs.nvidia.com/sdk-manager/docker-containers/index.html>`_ for details.

Post-Installation Setup
-----------------------

.. note::
   The following steps can also be executed via:

   .. code-block:: bash

      ./scripts/configure-system.agx-orin.sh

Update packages and install dependencies:

.. code-block:: bash

   sudo apt update
   sudo apt dist-upgrade -y
   sudo apt install -y apt-utils coreutils git-core git cmake build-essential \
       bc libssl-dev python3 python3-pip ninja-build ca-certificates curl pandoc

Download the Sionna Research Kit:

.. code-block:: bash

   cd ~ # We assume sionna-rk is cloned in the home directory
   git clone --recurse-submodules https://github.com/NVlabs/sionna-rk.git


Docker Installation
^^^^^^^^^^^^^^^^^^^

Install Docker from the official Docker repository:

.. code-block:: bash

   # Add Docker's official GPG key
   sudo install -m 0755 -d /etc/apt/keyrings
   sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
   sudo chmod a+r /etc/apt/keyrings/docker.asc

   # Add the repository to Apt sources
   echo \
     "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
     $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}") stable" | \
     sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

   # Install Docker, plugins, and NVIDIA container runtime
   sudo apt update
   sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin nvidia-container

   # Add user to docker group
   sudo usermod -aG docker $USER
   # Log out and log in again for changes to take effect

Configure Docker service:

.. code-block:: bash

   # Create Docker service override
   sudo mkdir -p /etc/systemd/system/docker.service.d
   sudo tee /etc/systemd/system/docker.service.d/override.conf <<EOF
   [Service]
   Environment="DOCKER_INSECURE_NO_IPTABLES_RAW=1"
   EOF

   # Restart Docker
   sudo systemctl daemon-reload
   sudo systemctl restart docker

Set the following environment variables on your profile:

.. code-block:: bash

   export SRK_PLATFORM="AGX Orin"
   export SRK_THREAD_POOL="6,7,8,9,10,11"
   export SRK_UE_THREAD_POOL="4,5"


TensorRT Installation
^^^^^^^^^^^^^^^^^^^^^

Install TensorRT and monitoring tools:

.. code-block:: bash

   sudo apt install -y cuda-toolkit nvidia-l4t-dla-compiler tensorrt

   # Add trtexec alias for convenience
   echo 'alias trtexec=/usr/src/tensorrt/bin/trtexec' >> ~/.bash_aliases

   # Install jetson-stats monitoring tool
   sudo python3 -m pip install -U jetson-stats

   # Show system stats
   jtop


Quectel Modem Compatibility
^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you want to connect a Quectel modem via USB to the Jetson AGX Orin, you need to build a custom kernel with SCTP support and the ``qmi_wwan`` kernel module. Note that this is only needed if the Orin acts as user equipment (UE).

This can be automatically done by running the following command:

.. code-block:: bash

   ./scripts/build-custom-kernel.sh
   ./scripts/install-custom-kernel.sh

This will build and install the custom kernel (see :ref:`kernel` for details). Reboot the system for the changes to take effect.


Orin Nano
---------

The Jetson Orin Nano can also be used with the Sionna Research Kit. The setup is similar to the AGX Orin, but with different configuration values due to its 6-core CPU (vs 12 on AGX Orin).

Use the Orin Nano configuration script:

.. code-block:: bash

   ./scripts/configure-system.orin-nano.sh

Key differences from AGX Orin:

* **Power mode**: Uses mode 2 (25W) instead of mode 0 (unlimited)
* **Thread pools**: ``SRK_THREAD_POOL="2,3,4,5"`` and ``SRK_UE_THREAD_POOL="2,3,4,5"``


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
