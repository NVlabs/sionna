===============================
get-oai-cn5g-commit-versions.sh
===============================

SYNOPSIS
========

.. program:: get-oai-cn5g-commit-versions.sh


.. code-block:: bash

    get-oai-cn5g-commit-versions.sh <oai-cn5g-fed_dir>

DESCRIPTION
===========

Print the commit hashes of the repository and its submodules recursively, to easily track the exact versions being used.

OPTIONS
=======

.. option:: <oai-cn5g-fed_dir>

    The root directory of the Core Network repository. Required.

EXAMPLES
========

.. code-block:: bash

    get-oai-cn5g-commit-versions.sh ./ext/oai-cn5g-fed
