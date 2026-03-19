======================
build-custom-kernel.sh
======================

SYNOPSIS
========

.. program:: build-custom-kernel.sh

.. code-block:: bash

    build-custom-kernel.sh
        [-h|--help]
        [--dry-run]
        [--verbose]
        [--clean]
        [--force]
        [--l4t-version <version>]
        [--source <path>]
        <destination_path>

DESCRIPTION
===========

Download the Linux for Tegra source code and customize the kernel sources for the requirements of the Sionna Research Kit. Build the kernel and its modules. The build should be followed by the install-custom-kernel script.

The script auto-detects the installed L4T version and downloads the corresponding kernel sources. It applies the configuration options from ``l4t/kernel/config.options`` to enable SCTP support and USB modem drivers.

The script skips execution if the required kernel modules (``sctp`` and ``qmi_wwan``) are already present.

This script requires elevated privileges for certain tasks (install software dependencies and module installation). It will prompt if sudo requires it.

OPTIONS
=======

.. option:: <destination_path>

    Specify the destination directory for the Linux for Tegra build. Code will be expanded and patched here. Default is ``ext/l4t``.

.. option:: --source <path>

    Specify the root directory of Sionna Research Kit. This will be used to locate the required kernel config options. Default is the current directory.

.. option:: --l4t-version <version>

    Specify the L4T version to use (e.g., ``36.4.3``). Default is to auto-detect from installed packages.

.. option:: --force

    Force rebuild even if the required modules are already present.

.. option:: --clean

    Remove the destination directory before starting, ensures this is a clean build. If the destination exists and the flag is not specified, the script will abort.

.. option:: --dry-run

    Print the commands to execute, but do not perform the operations.

.. option:: --verbose

    Enable verbose output. Print the commands being executed.

.. option:: -h, --help

    Display help message and exit.

EXAMPLES
========

.. code-block:: bash

    ./build-custom-kernel.sh
    ./build-custom-kernel.sh --dry-run
    ./build-custom-kernel.sh --clean --verbose ./ext/l4t
    ./build-custom-kernel.sh --force --l4t-version 36.4.3

SEE ALSO
========

:doc:`install-custom-kernel </scripts/install-custom-kernel>`
