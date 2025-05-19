======================
start-system.sh
======================

SYNOPSIS
========

.. program:: start-system.sh

.. code-block:: bash

    start-system.sh [config-name]

DESCRIPTION
===========

Start a set of Docker containers using the configuration files defined in [config-name], from the configs/ directory. If no configuration name is given, uses b200_arm64.

OPTIONS
=======

.. option:: config-name

    Use the files in the directory configs/<config-name> to start and configure the Docker containers.

EXAMPLES
========

.. code-block:: bash

    start-system.sh
    start-system.sh rfsim_arm64

SEE ALSO
========

:doc:`stop-system.sh </scripts/stop-system>`
