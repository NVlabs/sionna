========================
install-custom-kernel.sh
========================

SYNOPSIS
========

.. program:: install-custom-kernel.sh

.. code-block:: bash

    install-custom-kernel.sh
        [-h|--help]
        [--dry-run]
        [--verbose]
        [--source <path>]
        [--dest <path>]
        [--backup-postfix <postfix>]
        [--kernel-version <version>]

DESCRIPTION
===========

Installs a previously compiled Linux Kernel in the system. It requires the kernel and modules build directories, copies them to the right locations in the system, and modifies the boot system to use them.

The script backs up the existing kernel and modules before installing the new ones. It then updates the initrd and boot configuration using ``nv-update-initrd`` and ``nv-update-extlinux``.

This script is only supported on Tegra platforms (Jetson). It is not needed on DGX Spark.

This script requires elevated privileges. It will prompt if sudo requires it.

OPTIONS
=======

.. option:: --source <path>

    Specify the source directory for the Linux for Tegra build. Default is ``ext/l4t``.

.. option:: --dest <path>

    Specify the destination root directory for installation. Default is ``/``.

.. option:: --backup-postfix <postfix>

    Postfix to use when backing up existing kernel files. Default is ``original``.

.. option:: --kernel-version <version>

    Specify the kernel version to install. Default is the currently running kernel version.

.. option:: --dry-run

    Print the commands to execute, but do not perform the operations.

.. option:: --verbose

    Enable verbose output. Print the commands being executed.

.. option:: -h, --help

    Display help message and exit.

EXAMPLES
========

.. code-block:: bash

    ./install-custom-kernel.sh
    ./install-custom-kernel.sh --dry-run
    ./install-custom-kernel.sh --verbose --source ./ext/l4t
    ./install-custom-kernel.sh --backup-postfix backup

SEE ALSO
========

:doc:`build-custom-kernel </scripts/build-custom-kernel>`
