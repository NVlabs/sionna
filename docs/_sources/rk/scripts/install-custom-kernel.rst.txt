========================
install-custom-kernel.sh
========================

SYNOPSIS
========

.. program:: install-custom-kernel

.. code-block:: bash

    install-custom-kernel.sh 
        [-h|--help] 
        [--dry-run] 
        [--verbose] 
        [--source <path>]

DESCRIPTION
===========

Installs a previously compiled Linux Kernel in the System. It requires the kernel and modules build directories, copies them to the right locations in the system, and modifies the boot system to use them.

The script currently does not support the real-time kernel.

This script requires elevated privileges. It will prompt if sudo requires it.

OPTIONS
=======

.. option:: --source <path>

    Specify the source directory for the Linux for Tegra build. Default is 'ext/l4t'.

.. option:: -h, --help

    Display help message and exit.

.. option:: --dry-run

    Print the commands to execute, but do not perform the operations. Default is to execute.

.. option:: --verbose

    Enable verbose output. Print the commands being executed. Default is not to print them.


EXAMPLES
========

.. code-block:: bash

    install-custom-kernel.sh
    install-custom-kernel.sh --dry-run
    install-custom-kernel.sh --verbose --source ./ext/l4t


SEE ALSO
========

:doc:`build-custom-kernel </scripts/build-custom-kernel>`
