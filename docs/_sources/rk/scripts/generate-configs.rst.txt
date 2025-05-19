======================
generate-configs.sh
======================

SYNOPSIS
========

.. program:: generate-configs.sh

.. code-block:: bash

    SCRIPT_NAME {MANDATORY_ARG} [OPTIONAL_ARG...]
    generate-configs.sh 
        [-h|--help] 
        [--clean] 
        [--no-patching] 
        [--init-nested-repos] 
        --rk-dir <kit-rootdir> 
        --oai-dir <openairinterface5g_dir> 
        --dest <config-dir>

DESCRIPTION
===========

Initialize the configuration files based on settings on the OpenAirInterface repository and apply patches from the Sionna Research Kit as required by the configuration. Write the resulting configurations in the config directory.

OPTIONS
=======

.. option:: --rk-dir <kit-rootdir>

    Specify the base directory of the Sionna Research Kit. Default is the current directory.

.. option:: --oai-dir <openairinterface5g_dir>

    Specify the base directory of OpenAirInterface. Used to source base files. Default is 'ext/openairinterface5g'.

.. option:: --dest <config-dir>

    Destination directory where to create the configuration files. Default is '<kit-rootdir>/configs'.

.. option:: -h, --help

    Display help message and exit.

.. option:: --clean

    Remove the configuration directory before writing the files. This will remove all customizations in the directory.

.. option:: --no-patching

    Do not apply the Sionna Research Kit patches to the configurations, only copy base files from the OpenAirInterface repository.

.. option:: --init-nested-repos

    Create a nested git repository for each configuration subdirectory. This allow to track individual changes based from the OAI repository. This is used internally and is not needed in common use cases.

EXAMPLES
========

.. code-block:: bash

    ./generate-configs.sh --clean
    ./generate-configs.sh --clean --rk-dir . --dest ./clean-config-files
    ./generate-configs.sh --clean --rk-dir . --oai-dir ./ext/openairinterface5g --dest ./clean-config-files
    ./generate-configs.sh --clean --rk-dir . --oai-dir ./ext/openairinterface5g --dest ./base-config-files-only --no-patching

FILES
=====

*<rk-dir>/patches/configs/config-list.txt*
    List of configurations to create. Each one is a subdirectory in <configs-dir>.

*<rk-dir>/patches/configs/config-mappings.txt*
    File mappings per config, from OAI files to the subdirectory.

*<rk-dir>/patches/configs/config.\*.patch*
    Patch files for each configuration.

SEE ALSO
========

:doc:`quickstart-cn5g.sh </scripts/quickstart-cn5g>`, :doc:`start-system.sh </scripts/start-system>`
