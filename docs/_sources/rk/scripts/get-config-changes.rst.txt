======================
get-config-changes.sh
======================

SYNOPSIS
========

.. program:: get-config-changes.sh

.. code-block:: bash

    get-config-changes.sh
        [-h|--help]
        [--rk-dir <kit-rootdir>]
        --source <source_directory>
        --dest <destination-for-patches>

DESCRIPTION
===========

Takes a configuration directory and creates a set of patches to capture the changes. It requires the configuration directory to have been initialized with the init-nested-repos when initially generated.

This is script intended to track changes in configuration files and build the initial patches. It is not required in regular use cases.

You can track your own changes using git without the need for the nested repos.

OPTIONS
=======

.. option:: --rk-dir <kit-rootdir>

    Specify the base directory of the Sionna Research Kit. Default is the current directory.

.. option:: --source <source_directory>

    The config directory with your configuration files. Required.

.. option:: --dest <destination-for-patches>

    Destination directory where to write the patch files. Required.

.. option:: -h, --help

    Display help message and exit.


EXAMPLES
========

.. code-block:: bash

    ./generate-configs.sh --rk-dir . --source ./config --dest ./patches/config

FILES
=====

*<rk-dir>/patches/config/config-list.txt*
    List of configurations to create. Each one is a subdirectory in <config-dir>.

*<rk-dir>/patches/config/config-mappings.txt*
    File mappings per config, from OAI files to the subdirectory.

SEE ALSO
========

:doc:`generate-configs.sh </scripts/generate-configs>`, :doc:`quickstart-cn5g.sh </scripts/quickstart-cn5g>`, :doc:`start-system.sh </scripts/start-system>`
