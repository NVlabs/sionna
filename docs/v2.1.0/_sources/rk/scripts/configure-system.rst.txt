======================
configure-system.sh
======================

SYNOPSIS
========

.. program:: configure-system.sh

.. code-block:: bash

    configure-system.sh
        [-h|--help]
        [--force-platform <platform>]
        [--verbose]
        [--dry-run]

DESCRIPTION
===========

Performs several configuration steps required for the proper operation of the system. The script auto-detects the platform and runs the appropriate platform-specific configuration script.

Supported platforms:

* ``agx-orin`` - NVIDIA Jetson AGX Orin Developer Kit
* ``agx-thor`` - NVIDIA Jetson AGX Thor Developer Kit
* ``orin-nano`` - NVIDIA Jetson Orin Nano Super Developer Kit
* ``dgx-spark`` - NVIDIA DGX Spark

The script installs software dependencies, configures Docker, and sets power modes as described in the platform setup guides.

This script requires elevated privileges for certain operations. You need to log out and log back in for Docker group changes to take effect.

OPTIONS
=======

.. option:: --force-platform <platform>

    Override auto-detection and force configuration for a specific platform. Valid values: ``agx-orin``, ``agx-thor``, ``orin-nano``, ``dgx-spark``.

.. option:: --verbose

    Enable verbose output. Print the commands being executed.

.. option:: --dry-run

    Print the commands to execute, but do not perform the operations.

.. option:: -h, --help

    Display help message and exit.

EXAMPLES
========

.. code-block:: bash

    ./configure-system.sh
    ./configure-system.sh --force-platform dgx-spark
    ./configure-system.sh --dry-run --verbose
