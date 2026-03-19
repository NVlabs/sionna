.. _spark:

DGX Spark Setup
===============

This guide covers the required steps to set up an `NVIDIA DGX Spark <https://www.nvidia.com/en-us/products/workstations/dgx-spark/>`_. The DGX Spark runs DGX OS, an Ubuntu-based distribution with drivers and utilities optimized for the DGX Spark hardware.

The installation guide aims to be self-contained. However, the `DGX Spark User Guide <https://docs.nvidia.com/dgx/dgx-spark/index.html>`_ is a good reference for further details.


Post-Installation Setup
-----------------------

.. note::
   The following steps can also be executed via:

   .. code-block:: bash

      ./scripts/configure-system.dgx-spark.sh

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

   # Install Docker, plugins, and NVIDIA container toolkit
   sudo apt update
   sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin nvidia-container-toolkit

   # Add user to docker group
   sudo usermod -aG docker $USER
   # Log out and log in again for changes to take effect

Configure Docker runtime for GPU support:

.. code-block:: bash

   sudo nvidia-ctk runtime configure --runtime=docker

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

   export SRK_PLATFORM="DGX Spark"
   export SRK_THREAD_POOL="15,16,17,18,19"
   export SRK_UE_THREAD_POOL="4,5"


TensorRT Installation
^^^^^^^^^^^^^^^^^^^^^

Install TensorRT and monitoring tools:

.. code-block:: bash

   sudo apt install -y cuda-toolkit tensorrt nvtop

   # Add trtexec alias for convenience
   echo 'alias trtexec=/usr/src/tensorrt/bin/trtexec' >> ~/.bash_aliases


Version Information
-------------------

Check OS version:

.. code-block:: bash

   cat /etc/lsb-release
   DISTRIB_ID=Ubuntu
   DISTRIB_RELEASE=24.04
   DISTRIB_CODENAME=noble
   DISTRIB_DESCRIPTION="Ubuntu 24.04.3 LTS"

Check DGX OS version:

.. code-block:: bash

   cat /etc/dgx-release
   DGX_NAME="DGX Spark"
   DGX_PRETTY_NAME="NVIDIA DGX Spark"
   DGX_SWBUILD_DATE="2025-09-10-13-50-03"
   DGX_SWBUILD_VERSION="7.2.3"
   DGX_COMMIT_ID="833b4a7"
   DGX_PLATFORM="DGX Server for KVM"
   DGX_SERIAL_NUMBER="XXXXXXXXXXXX"

   DGX_OTA_VERSION="7.3.1"
   DGX_OTA_DATE="Wed Nov 19 05:05:30 PM CET 2025"
