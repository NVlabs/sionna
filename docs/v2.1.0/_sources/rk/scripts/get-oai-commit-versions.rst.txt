==========================
get-oai-commit-versions.sh
==========================

SYNOPSIS
========

.. program:: get-oai-commit-versions.sh


.. code-block:: bash

    get-oai-commit-versions.sh <openairinterface5g_dir>

DESCRIPTION
===========

Print the commit hashes of the repository and its submodules, to easily track the exact versions being used.

OPTIONS
=======

.. option:: <openairinterface5g_dir>

    The root directory of the OpenAirInterface repository. Required.

EXAMPLES
========

.. code-block:: bash

    get-oai-commit-versions.sh ./ext/openairinterface5g
