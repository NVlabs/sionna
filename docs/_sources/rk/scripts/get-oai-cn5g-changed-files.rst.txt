=============================
get-oai-cn5g-changed-files.sh
=============================

SYNOPSIS
========

.. program:: get-oai-cn5g-changed-files.sh

.. code-block:: bash

    get-oai-cn5g-changed-files.sh
        [-h|--help]
        --source <source_directory>
        --dest <destination_directory>
        [--patch-file <patch_file>]

DESCRIPTION
===========

Collects changed files in the Core Network source tree and copies them to a separate file tree in the destination directory. Optionally, it creates a patch file of the changes.

When used together with patch-oai-cn5g.sh and quickstart-cn5g.sh, they provide bidirectional support for changes in- and out- of tree, and a simplified deployment of the changes over a patch file usable by the quickstart-cn5g.sh script.

OPTIONS
=======

.. option:: --source <source_directory>

    The Core Network repository from where the changes are to be extracted.

.. option:: --dest <destination_directory>

    The destination directory where to copy files changed.

.. option:: -h, --help

    Display help message and exit.

.. option:: --patch-file <patch_file>

    Write a patch file (in git diff format) to <patch_file>.

EXAMPLES
========

.. code-block:: bash

    get-oai-changed-files.sh --source ./ext/oai-cn5g-fed --dest ./oai-cn5g-files
    get-oai-changed-files.sh --source ./ext/oai-cn5g-fed --dest ./oai-cn5g-files --patch-file ./patches/oai-cn5g.patch

SEE ALSO
========

:doc:`quickstart-oai.sh </scripts/quickstart-cn5g>`
