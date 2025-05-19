======================
configure-system.sh
======================

SYNOPSIS
========

.. program:: configure-system.sh

.. code-block:: bash

    configure-system.sh

DESCRIPTION
===========

Performs several configuration steps required for the proper operation of the system. Installs software dependencies and changes power modes as described in the Sionna Research Kit documentation.

This script requires elevated privileges, should be executed with sudo or as root. You need to log out and log back in for changes to take effect. This script is required only once, executing it multiple times has no effect.

This is required before the system quickstart scripts are invoked.

EXAMPLES
========

.. code-block:: bash

    sudo configure-system.sh
