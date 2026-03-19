.. _kernel:

Custom Jetson Linux Kernel
==========================

.. note::
   The custom kernel is only required for NVIDIA Jetson platforms (AGX Orin,
   Orin Nano). The AGX Thor requires a custom kernel only if connecting the Quectel modem as user equipment (``qmi_wwan``). The DGX Spark does not require kernel modifications.

The 5G core network requires SCTP (Stream Control Transmission Protocol) support in the Linux kernel. Additionally, connecting a Quectel modem via USB requires the ``qmi_wwan`` kernel module. By default, these are not enabled in the Jetson Linux kernel. This guide walks through building a custom Linux kernel that includes these features.

This guide is based on the `Jetson Linux Kernel Configuration Guide <https://docs.nvidia.com/jetson/l4t/index.html#page/Tegra%20Linux%20Driver%20Package%20Source%20Code%20Guide/kernel_config.html>`_. The following instructions deviate from the original in a few points:

* We do not use the Bootlin toolchain (11.3) but use the provided Ubuntu GCC compiler instead, since we are not cross-compiling
* We add the SCTP network protocol and USB modem drivers as modules
* We enable advanced routing in the kernel
* We do not recompile the DTBs (Device Tree Blobs). The system's device tree remains unchanged
* We backup the original installation; we do not override it

.. note::
   It is highly recommended NOT to perform these steps as root. Use sudo only when necessary for specific steps.


Automated Build (Recommended)
-----------------------------

The easiest way to build and install the custom kernel is via the provided scripts:

.. code-block:: bash

   ./scripts/build-custom-kernel.sh
   ./scripts/install-custom-kernel.sh
   sudo reboot

The scripts automatically:

* Detect the installed L4T version
* Download the correct kernel sources
* Apply the required configuration options
* Build and install the kernel and modules
* Update the initrd and boot configuration

.. note::
   The scripts skip execution if the required modules (``sctp`` and ``qmi_wwan``) are already present. Use ``--force`` to rebuild anyway.


Manual Build
------------

For advanced users who want more control over the build process, the following sections describe the manual steps.

Prerequisites
^^^^^^^^^^^^^

Install the required build tools:

.. code-block:: bash

   sudo apt update
   sudo apt install -y git-core build-essential flex bison bc kmod libssl-dev

Source Code
^^^^^^^^^^^

Download and extract the kernel source packages. The exact URL depends on your L4T version:

.. code-block:: bash

   # Create build directory
   mkdir -p ext/l4t && cd ext/l4t

   # Download sources (example for L4T 36.4.3)
   wget https://developer.nvidia.com/downloads/embedded/l4t/r36_release_v4.3/sources/public_sources.tbz2

   # Extract source packages
   tar xf public_sources.tbz2
   cd Linux_for_Tegra/source

   # Expand required sources
   tar xf kernel_src.tbz2
   tar xf kernel_oot_modules_src.tbz2
   tar xf nvidia_kernel_display_driver_source.tbz2


Kernel Configuration
^^^^^^^^^^^^^^^^^^^^

The kernel configuration options are stored in ``l4t/kernel/config.options``:

.. code-block:: bash

   CONFIG_IP_ADVANCED_ROUTER=y
   CONFIG_IP_MULTIPLE_TABLES=y
   CONFIG_INET_SCTP_DIAG=m
   CONFIG_IP_SCTP=m
   CONFIG_NETFILTER_XT_MATCH_SCTP=m
   CONFIG_NF_CT_PROTO_SCTP=y
   CONFIG_SCTP_COOKIE_HMAC_MD5=y
   CONFIG_SCTP_DEFAULT_COOKIE_HMAC_MD5=y
   CONFIG_USB_NET_DRIVERS=y
   CONFIG_USB_NET_CDCETHER=m
   CONFIG_USB_NET_CDC_NCM=m
   CONFIG_USB_NET_CDC_MBIM=m
   CONFIG_USB_NET_QMI_WWAN=m
   CONFIG_USB_WDM=m
   CONFIG_USB_SERIAL_WWAN=m

These enable:

* **SCTP protocol**: Required for 5G core network communication
* **Advanced routing**: Required for proper network configuration
* **USB network drivers**: Required for Quectel modem connectivity (``qmi_wwan``)

To apply these options manually, navigate to the kernel source directory and use the kernel config scripts:

.. code-block:: bash

   cd Linux_for_Tegra/source/kernel/kernel-jammy-src  # or kernel-noble for Thor

   # Start from default config
   make mrproper
   make defconfig

   # Apply each option (example)
   ./scripts/config --file .config --set-val CONFIG_IP_SCTP m
   # ... repeat for other options

   # Resolve dependencies
   make olddefconfig

Building the Kernel
^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   cd Linux_for_Tegra/source

   # Build the kernel
   make -C kernel

   # Create temporary rootfs directories
   mkdir -p Linux_for_Tegra/rootfs/boot
   mkdir -p Linux_for_Tegra/rootfs/kernel

   # Set module installation path
   export INSTALL_MOD_PATH=$PWD/../rootfs

   # Install kernel
   sudo -E make install -C kernel

   # Set kernel headers path (adjust for your Ubuntu version)
   export KERNEL_HEADERS=$PWD/kernel/kernel-jammy-src  # or kernel-noble

   # For Thor (Noble), also set:
   export kernel_name=noble

   # Build and install modules
   make modules
   sudo -E make modules_install

Installing the Kernel
^^^^^^^^^^^^^^^^^^^^^

.. warning::
   Back up your existing kernel before proceeding. Package updates may overwrite custom kernels.

.. code-block:: bash

   # Backup existing kernel and modules
   sudo cp /boot/Image /boot/Image.original
   sudo cp /boot/initrd /boot/initrd.original
   sudo cp -r /lib/modules/$(uname -r) /lib/modules/$(uname -r).original

   # Install new kernel
   sudo cp Linux_for_Tegra/rootfs/boot/Image /boot/Image

   # Install new modules
   sudo cp -r Linux_for_Tegra/rootfs/lib/modules/$(uname -r) /lib/modules/

   # Update initrd and boot configuration
   sudo nv-update-initrd
   sudo nv-update-extlinux generic

Boot Configuration (Optional)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For more control over boot options, you can manually edit the extlinux configuration. This is equivalent to editing ``grub.conf`` on a standard system and allows you to boot the system to the old kernel if necessary.

.. note::
   This example assumes the system uses an NVMe SSD (root=/dev/nvme0n1p1). Adjust as needed for your setup.

.. code-block:: bash

   cd /boot/extlinux

   # Backup current configuration
   sudo cp extlinux.conf extlinux.conf.original

Edit ``extlinux.conf`` to add boot entries. Example configuration:

.. code-block:: bash

   TIMEOUT 30
   DEFAULT primary

   MENU TITLE L4T boot options

   LABEL primary
         MENU LABEL primary kernel
         LINUX /boot/Image
         INITRD /boot/initrd
         APPEND ${cbootargs} root=/dev/nvme0n1p1 rw rootwait rootfstype=ext4 mminit_loglevel=4 console=ttyTCU0,115200 console=ttyAMA0,115200 firmware_class.path=/etc/firmware fbcon=map:0 net.ifnames=0 nospectre_bhb video=efifb:off console=tty0 nv-auto-config

   LABEL original
         MENU LABEL original precompiled kernel
         LINUX /boot/Image.original
         INITRD /boot/initrd.original
         APPEND ${cbootargs} root=/dev/nvme0n1p1 rw rootwait rootfstype=ext4 mminit_loglevel=4 console=ttyTCU0,115200 console=ttyAMA0,115200 firmware_class.path=/etc/firmware fbcon=map:0 net.ifnames=0 nospectre_bhb video=efifb:off console=tty0 nv-auto-config

Change the ``DEFAULT`` entry to the label you want to use and reboot.


Verification
------------

After rebooting, verify the modules are available:

.. code-block:: bash

   # Check for SCTP module
   modinfo sctp

   # Check for Quectel modem driver
   modinfo qmi_wwan

   # Load modules
   sudo modprobe sctp
   sudo modprobe qmi_wwan

   # List loaded modules
   lsmod | grep -E "sctp|qmi"
