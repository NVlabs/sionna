======================
stop-system.sh
======================

SYNOPSIS
========

.. program:: stop-system.sh

.. code-block:: bash

    stop-system.sh [config-name]

DESCRIPTION
===========

Stop running Docker containers based on the layout of the Docker Compose file for the specified configuration. If no configuration is provided, use ``b200_arm64``.

OPTIONS
=======

.. option:: config-name

    The name of the configuration in the ``configs`` directory. Defaults to ``b200_arm64``.

EXAMPLES
========

.. code-block:: bash

    stop-system.sh
    stop-system.sh rfsim_arm64

SEE ALSO
========

:doc:`start-system.sh </scripts/start-system>`
