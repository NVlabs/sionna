.. _jetson_thor:

Jetson AGX Thor Setup
=====================

This guide covers the required steps to set up an `NVIDIA Jetson AGX Thor <https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-thor/>`_. The Jetson runs NVIDIA Jetson Linux, an Ubuntu-based distribution with drivers and utilities optimized for the Jetson hardware.

The installation guide aims to be self-contained. However, the `Jetson Linux Developer Guide <https://docs.nvidia.com/jetson/archives/r38.2.1/DeveloperGuide/index.html#>`_ is a good reference for further details.


Post-Installation Setup
-----------------------

.. note::
   The following steps can also be executed via:

   .. code-block:: bash

      ./scripts/configure-system.agx-thor.sh

Update packages and install dependencies:

.. code-block:: bash

   sudo apt update
   sudo apt dist-upgrade -y
   sudo apt install -y apt-utils coreutils git-core git cmake build-essential \
       bc libssl-dev python3 python3-pip ninja-build ca-certificates curl \
       pandoc nvidia-jetpack

Configure CUDA paths:

.. code-block:: bash

   echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
   echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
   source ~/.bashrc

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

   # Install Docker and plugins
   sudo apt update
   sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

   # Add user to docker group
   sudo usermod -aG docker $USER
   # Log out and log in again for changes to take effect


NVIDIA Container Toolkit
^^^^^^^^^^^^^^^^^^^^^^^^

Install the NVIDIA Container Toolkit for GPU support in Docker:

.. code-block:: bash

   # Add NVIDIA Container Toolkit repository
   curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
       sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
   curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
       sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
       sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

   # Install the toolkit
   sudo apt update
   sudo apt install -y libnvidia-container-tools libnvidia-container1 \
       nvidia-container-toolkit nvidia-container-toolkit-base

   # Configure Docker runtime
   sudo nvidia-ctk runtime configure --runtime=docker

Configure Docker service for Thor:

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

Set the following environment variables:

.. code-block:: bash

   export SRK_PLATFORM="AGX Thor"
   export SRK_THREAD_POOL="9,10,11,12,13"
   export SRK_UE_THREAD_POOL="4,5"

TensorRT Installation
^^^^^^^^^^^^^^^^^^^^^

Install TensorRT and monitoring tools:

.. code-block:: bash

   sudo apt install -y cuda-toolkit tensorrt

   # Add trtexec alias for convenience
   echo 'alias trtexec=/usr/src/tensorrt/bin/trtexec' >> ~/.bash_aliases


Quectel Modem Compatibility
^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you want to connect a Quectel modem via USB to the Jetson AGX Thor, you need to build a custom kernel with the ``qmi_wwan`` kernel module. Note that this is only needed if the Thor acts as user equipment (UE).

This can be automatically done by running the following command:

.. code-block:: bash

   ./scripts/build-custom-kernel.sh
   ./scripts/install-custom-kernel.sh

This will build and install the custom kernel (see :ref:`kernel` for details). Reboot the system for the changes to take effect.


Version Information
-------------------

Check OS version:

.. code-block:: bash

   cat /etc/lsb-release
   DISTRIB_ID=Ubuntu
   DISTRIB_RELEASE=24.04
   DISTRIB_CODENAME=noble
   DISTRIB_DESCRIPTION="Ubuntu 24.04.3 LTS"

Check Jetson Linux & JetPack version:

.. code-block:: bash

   cat /etc/nv_tegra_release
   # R38 (release), REVISION: 2.1, GCID: 42061081, BOARD: generic, EABI: aarch64, DATE: Wed Sep 10 19:49:31 UTC 2025
   TARGET_USERSPACE_LIB_DIR=nvidia
   TARGET_USERSPACE_LIB_DIR_PATH=usr/lib/aarch64-linux-gnu/nvidia
