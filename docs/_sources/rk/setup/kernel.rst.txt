.. _kernel:

Custom Linux Kernel
===================

The 5G core network requires SCTP (Stream Control Transmission Protocol) support in the Linux kernel. By default, this protocol is not enabled in the Jetson Linux kernel. This guide walks through building a custom Linux kernel for the Jetson platform that includes SCTP network protocol support and other required features.

This guide is based on the `Jetson Linux Kernel Configuration Guide <https://docs.nvidia.com/jetson/l4t/index.html#page/Tegra%20Linux%20Driver%20Package%20Source%20Code%20Guide/kernel_config.html>`_. The following instructions deviate from the original in a few points:

* We do not use the Bootlin toolchain (11.3) but use the provided Ubuntu GCC compiler (11.4) instead, since we are not cross-compiling
* We add the SCTP network protocol as a module
* We enable advanced routing in the kernel
* We do not recompile the DTBs (Device Tree Blobs). The system's device tree remains unchanged
* We backup the original installation; we do not override it

.. note::
   These steps are summarized in the scripts ``scripts/build-custom-kernel.sh`` and ``scripts/install-custom-kernel.sh``. Realtime kernels are not supported by the scripts at the moment.

.. note::
   It is highly recommended NOT to perform these steps as root. Use sudo or root only when necessary for specific steps.

Prerequisites
-------------

Building the kernel requires the following software:

* Git
* Build Tools
* GCC
* OpenSSL development library

Install these required packages via:

.. code-block:: bash

   sudo apt update
   sudo apt install git-core build-essential bc libssl-dev

Source Code
-----------

Download the source packages (`Driver Package Source Code <https://developer.nvidia.com/downloads/embedded/l4t/r36_release_v3.0/sources/public_sources.tbz2>`_) from the `Jetson Linux Release 36.3 page <https://developer.nvidia.com/embedded/jetson-linux-r363>`_:

.. code-block:: bash

   mkdir l4t && cd l4t
   wget https://developer.nvidia.com/downloads/embedded/l4t/r36_release_v3.0/sources/public_sources.tbz2

Extract the packages:

   .. code-block:: bash

      # Extract source packages
      tar xf public_sources.tbz2

      # Go into source directory
      cd Linux_for_Tegra/source

      # Expand required sources (kernel, OOT modules, display driver)
      tar xf kernel_src.tbz2
      tar xf kernel_oot_modules_src.tbz2
      tar xf nvidia_kernel_display_driver_source.tbz2

Kernel Configuration
--------------------

The preferred approach is to replace `defconfig` with a modified version.

.. code-block:: bash

   # Assuming sionna-rk is cloned in the home directory
   cp sionna-rk/l4t/kernel/defconfig source/kernel/kernel-jammy-src/arch/arm64/configs/defconfig

Otherwise, configure the kernel to enable the SCTP module in the kernel configuration or edit the file directly to add or modify the following lines:

.. code-block:: bash

   CONFIG_IP_ADVANCED_ROUTER=y
   CONFIG_IP_MULTIPLE_TABLES=y
   CONFIG_INET_SCTP_DIAG=m
   CONFIG_IP_SCTP=m
   CONFIG_NETFILTER_XT_MATCH_SCTP=m
   CONFIG_NF_CT_PROTO_SCTP=y
   CONFIG_SCTP_COOKIE_HMAC_MD5=y
   CONFIG_SCTP_DEFAULT_COOKIE_HMAC_MD5=y

The key changes in our config enable:

* SCTP network protocol support
* Advanced routing capabilities
* Real-time extensions (optional)

Building the Kernel
-------------------

We now need to re-build the kernel and install the modules.
To further reduce the latency induced by the scheduler, we also build a real-time kernel that can be used optionally.

If you want to build both regular and realtime kernels, compile, install and test the non-realtime version first, then compile, install and test the realtime version. Or use separate directories for the realtime and non-realtime kernel compilation.

Standard Build
^^^^^^^^^^^^^^

.. code-block:: bash

   # Navigate to kernel source directory
   cd ~/l4t/Linux_for_Tegra/source

   # Build the kernel
   make -C kernel

   # Create temporary root filesystem target directories
   mkdir -p ~/l4t/Linux_for_Tegra/rootfs/boot
   mkdir -p ~/l4t/Linux_for_Tegra/rootfs/kernel

   # Set the installation target for modules
   export INSTALL_MOD_PATH=~/l4t/Linux_for_Tegra/rootfs

   # Install kernel in temporary rootfs
   sudo -E make install -C kernel

   # Navigate to source directory
   cd ~/l4t/Linux_for_Tegra/source

   # Build modules (including OOT modules)
   export KERNEL_HEADERS=$PWD/kernel/kernel-jammy-src
   make modules

   # Install modules in temporary rootfs
   sudo -E make modules_install

Real-time Build (Optional)
^^^^^^^^^^^^^^^^^^^^^^^^^^

For real-time support:

.. code-block:: bash

   # Navigate to kernel source directory
   cd ~/l4t/Linux_for_Tegra/source

   # Enable realtime extensions
   ./generic_rt_build.sh "enable"

   # Build the kernel
   make -C kernel

   # Create temporary root filesystem target directories
   mkdir -p ~/l4t/Linux_for_Tegra/rootfs/boot
   mkdir -p ~/l4t/Linux_for_Tegra/rootfs/kernel

   # Set the installation target for modules
   export INSTALL_MOD_PATH=~/l4t/Linux_for_Tegra/rootfs

   # Tell modules to ignore realtime settings
   export IGNORE_PREEMPT_RT_PRESENCE=1

   # Install kernel in temporary rootfs
   sudo -E make install -C kernel

   # Navigate to source directory
   cd ~/l4t/Linux_for_Tegra/source

   # Build modules (including OOT modules)
   export KERNEL_HEADERS=$PWD/kernel/kernel-jammy-src
   make modules

   # Install modules in temporary rootfs
   sudo -E make modules_install


.. note::
   Double check that advanced routing and SCTP modules are properly configured after enabling the realtime kernel, but before building the Docker images.

Installing Kernel Image and Modules
-----------------------------------

.. warning::
   Back up your existing kernel before proceeding. Package updates may overwrite custom kernels.

The goal is to have ``Image``, ``initrd`` and kernel modules in two or three different versions installed: original, custom with SCTP, and optionally custom realtime with SCTP. We use in the following builds the postfixes ``.original``, ``.new`` and ``.rt``.

Standard Build
^^^^^^^^^^^^^^

.. code-block:: bash

   # Backup existing setup (kernel, initrd, and modules)
   cd /boot
   sudo cp Image Image.original
   sudo cp initrd initrd.original
   sudo cp initrd.img-5.15.136-tegra initrd.img-5.15.136-tegra.original
   sudo cp -r /lib/modules/5.15.136-tegra /lib/modules/5.15.136-tegra.original

   # Copy new kernel
   sudo cp ~/l4t/Linux_for_Tegra/rootfs/boot/Image /boot/Image.new

   # Copy new modules
   sudo cp -r ~/l4t/Linux_for_Tegra/rootfs/lib/modules/5.15.136-tegra /lib/modules/5.15.136-tegra.new

   # Create symlinks
   cd /boot
   sudo rm Image
   sudo ln -s Image.new Image

   cd /lib/modules
   sudo rm -rf /lib/modules/5.15.136-tegra
   sudo ln -s 5.15.136-tegra.new 5.15.136-tegra

   # Build new initrd
   cd /boot
   sudo nv-update-initrd
   sudo mv /boot/initrd initrd.new
   sudo ln -s initrd.new initrd
   # this is a backup of the created initrd you just updated
   sudo cp /boot/initrd /boot/initrd.img-5.15.136-tegra.new

Real-time Build (Optional)
^^^^^^^^^^^^^^^^^^^^^^^^^^

Note that because the stock realtime kernel is not installed by default, the backups are from the original kernel, and the modules do not override the stock modules.

.. code-block:: bash

   # Backup existing setup (kernel, initrd, and modules)
   cd /boot
   sudo cp Image Image.rt.original
   sudo cp initrd initrd.rt.original

   # Copy new kernel
   sudo cp ~/l4t/Linux_for_Tegra/rootfs/boot/Image /boot/Image.rt

   # Copy new modules
   sudo cp -r ~/l4t/Linux_for_Tegra/rootfs/lib/modules/5.15.136-rt-tegra /lib/modules/5.15.136-rt-tegra

   # Create symlinks
   cd /boot
   sudo rm Image
   sudo ln -s Image.rt Image

   # Build new initrd
   cd /boot
   sudo nv-update-initrd
   sudo mv /boot/initrd initrd.rt
   sudo ln -s initrd.rt initrd
   # this is a backup of the created initrd you just updated
   sudo cp /boot/initrd.rt /boot/initrd.img-5.15.136-rt-tegra

Configure Boot Sequence
-----------------------

This is equivalent to editing *grub.conf* on a standard system and allows you to boot the system to the old kernel if necessary.

.. note::
   This example assumes the system uses an NVMe SSD (root=/dev/nvme0n1p1). Adjust as needed for your setup.

.. code-block:: bash

   cd /boot/extlinux

   # Backup current configuration
   cp extlinux.conf extlinux.conf.original

Edit the original `extlinux.conf` to add another entry. Here is an example configuration. Change the ``DEFAULT`` entry to th  e label you want to use and reboot:

.. code-block:: bash

   TIMEOUT 30
   DEFAULT primary

   MENU TITLE L4T boot options

   LABEL primary
         MENU LABEL primary kernel
         LINUX /boot/Image.new
         INITRD /boot/initrd.new
         APPEND ${cbootargs} root=/dev/nvme0n1p1 rw rootwait rootfstype=ext4 mminit_loglevel=4 console=ttyTCU0,115200 console=ttyAMA0,115200 firmware_class.path=/etc/firmware fbcon=map:0 net.ifnames=0 nospectre_bhb video=efifb:off console=tty0 nv-auto-config

   LABEL realtime
         MENU LABEL realtime kernel
         LINUX /boot/Image.rt
         INITRD /boot/initrd.rt
         APPEND ${cbootargs} root=/dev/nvme0n1p1 rw rootwait rootfstype=ext4 mminit_loglevel=4 console=ttyTCU0,115200 console=ttyAMA0,115200 firmware_class.path=/etc/firmware fbcon=map:0 net.ifnames=0 nospectre_bhb video=efifb:off console=tty0 nv-auto-config

   LABEL original
         MENU LABEL original precompiled kernel
         LINUX /boot/Image.original
         INITRD /boot/initrd.original
         APPEND ${cbootargs} root=/dev/nvme0n1p1 rw rootwait rootfstype=ext4 mminit_loglevel=4 console=ttyTCU0,115200 console=ttyAMA0,115200 firmware_class.path=/etc/firmware fbcon=map:0 net.ifnames=0 nospectre_bhb video=efifb:off console=tty0 nv-auto-config


Reboot for the changes to take effect.


And finally, verify that the modules are correctly loaded:

.. code-block:: bash

   # List loaded modules
   lsmod

   # Probe and load a module (including dependencies)
   sudo modprobe sctp

   # Inspect a module
   modinfo sctp
