========================
get-oai-changed-files.sh
========================

SYNOPSIS
========

.. program:: get-oai-changed-files.sh

.. code-block:: bash

    get-oai-changed-files.sh
        [-h|--help]
        --source <source_directory>
        --dest <destination_directory>
        [--patch-file <patch_file>]
        [--binary]

DESCRIPTION
===========

Collects changed files in the OpenAirInterface source tree and copies them to a separate file tree in the destination directory. Optionally, it creates a patch file of the changes.

When used together with patch-oai.sh and quickstart-oai.sh, they provide bidirectional support for changes in- and out- of tree, and a simplified deployment of the changes over a patch file usable by the quickstart-oai.sh script.

OPTIONS
=======

.. option:: --source <source_directory>

    The OpenAirInterface repository from where the changes are to be extracted.

.. option:: --dest <destination_directory>

    The destination directory where to copy files changed.

.. option:: -h, --help

    Display help message and exit.

.. option:: --patch-file <patch_file>

    Write a patch file (in git diff format) to <patch_file>.

.. option:: --binary

    Include binary files in the patch output.

EXAMPLES
========

.. code-block:: bash

    get-oai-changed-files.sh --source ./ext/openairinterface5g --dest ./oai-files
    get-oai-changed-files.sh --source ./ext/openairinterface5g --dest ./oai-files --patch-file ./patches/openairinterface5g.patch

SEE ALSO
========

:doc:`quickstart-oai.sh </scripts/quickstart-oai>`
