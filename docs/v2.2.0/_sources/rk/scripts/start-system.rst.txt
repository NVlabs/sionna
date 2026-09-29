======================
start_system.sh
======================

SYNOPSIS
========

.. program:: start_system.sh

.. code-block:: bash

    start_system.sh [config-name]

DESCRIPTION
===========

Start a set of Docker containers using the configuration files defined in [config-name], from the config/ directory. The script starts the 5G core network components (MySQL, AMF, SMF, UPF, ext-DN), the nearRT-RIC, and then the gNB. If the configuration name contains "rfsim", it also starts the software UE.

OPTIONS
=======

.. option:: config-name

    Use the files in the directory config/<config-name> to start and configure the Docker containers. Default is ``rfsim``.

EXAMPLES
========

.. code-block:: bash

    start_system.sh
    start_system.sh rfsim
    start_system.sh b200

SEE ALSO
========

:doc:`stop_system.sh <stop-system>`
