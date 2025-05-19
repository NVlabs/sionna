======================
build-custom-kernel.sh
======================

SYNOPSIS
========

.. program:: build-custom-kernel

.. code-block:: bash

    build-custom-kernel
        [-h|--help]
        [--dry-run]
        [--verbose]
        [--clean]
        [--source <path>]
        <destination_path>

DESCRIPTION
===========

Download the Linux for Tegra source code and customize the kernel sources for the requirements of the Sionna Research Kit. Build the kernel and its modules. The build should be followed by the install-custom-kernel script.

The script currently does not support the real-time kernel.

This script requires elevated privileges for certain tasks (install software dependencies and module installation). It will prompt if sudo requires it.

OPTIONS
=======

.. option:: <destination_path>

    Specify the destination directory for the Linux for Tegra build. Code will be expanded and patched here. Default is 'ext/l4t'.

.. option:: --source <path>

    Specify the root directory of Sionna Research Kit. This will be used to locate the required defconfig for the Kernel. Default is the current directory.

.. option:: -h, --help

    Display help message and exit.

.. option:: --dry-run

    Print the commands to execute, but do not perform the operations. Default is to execute.

.. option:: --verbose

    Enable verbose output. Print the commands being executed. Default is not to print them.

.. option:: --clean

    Remove the destination directory before starting, ensures this is a clean build. If the destination exists and the flag is not specified, the script will abort.

EXAMPLES
========

.. code-block:: bash

    ./build-custom-kernel.sh
    ./build-custom-kernel.sh --dry-run
    ./build-custom-kernel.sh --clean --verbose --source . ./ext/l4t

SEE ALSO
========

:doc:`install-custom-kernel </scripts/install-custom-kernel>`
