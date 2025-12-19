========================
install-usrp.sh
========================

SYNOPSIS
========

.. program:: install-usrp.sh

.. code-block:: bash

    install-usrp.sh
        [-h|--help]
        [--force]
        [--verbose]
        [--dry-run]

DESCRIPTION
===========

Installs the UHD (USRP Hardware Driver) drivers required to use Ettus USRP devices.

The script clones the UHD repository, builds it from source, installs the firmware images, and configures udev rules for non-root access.

The script skips execution if UHD is already installed (``uhd_find_devices`` is found in PATH).

This script requires elevated privileges. It will prompt if sudo requires it.

OPTIONS
=======

.. option:: --force

    Force installation even if UHD is already installed.

.. option:: --verbose

    Enable verbose output. Print the commands being executed.

.. option:: --dry-run

    Print the commands to execute, but do not perform the operations.

.. option:: -h, --help

    Display help message and exit.

EXAMPLES
========

.. code-block:: bash

    ./install-usrp.sh
    ./install-usrp.sh --force
    ./install-usrp.sh --dry-run --verbose

SEE ALSO
========

:doc:`configure-system </scripts/configure-system>`
