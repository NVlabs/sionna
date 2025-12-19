======================
quickstart-oai.sh
======================

SYNOPSIS
========

.. program:: quickstart-oai.sh

.. code-block:: bash

    quickstart-oai.sh
        [-h|--help]
        [--clean]
        [--debug]
        [--no-build]
        [--tag <tagname>]
        [--oai-version <version>]
        [--ci]
        [--source <kit-rootdir>]
        [--dest <openairinterface5g_dir>]

DESCRIPTION
===========

Start from scratch and perform the steps needed to end with a set of working Docker images for OpenAirInterface. The images can then be started using one of the provided configurations using the start-system script.

The script clones the OpenAirInterface repository at the specified version, applies patches from the Sionna Research Kit, and calls Docker to build the images.

OPTIONS
=======

.. option:: --source <kit-rootdir>

    Specify the root directory of Sionna Research Kit. This will be used to locate the required patches for OpenAirInterface. Default is the current directory.

.. option:: --dest <openairinterface5g_dir>

    Specify the destination directory for the OpenAirInterface code. Code will be cloned and patched here. Default is ``ext/openairinterface5g``.

.. option:: --oai-version <version>

    Specify the OAI version/branch to checkout. Default is ``2025.w34``.

.. option:: --tag <tagname>

    Use <tagname> for the created Docker images. Default is ``latest``.

.. option:: --clean

    Remove the OpenAirInterface directory before proceeding. If the directory exists and the flag is not specified, the script will abort.

.. option:: --no-build

    Skip the build step of the Docker images. Only clone and patch the repository.

.. option:: --debug

    Enable debug output during the build process.

.. option:: --ci

    Enable CI mode for automated builds.

.. option:: -h, --help

    Display help message and exit.

EXAMPLES
========

.. code-block:: bash

    ./quickstart-oai.sh
    ./quickstart-oai.sh --clean
    ./quickstart-oai.sh --clean --tag experimental
    ./quickstart-oai.sh --oai-version 2025.w34 --no-build

SEE ALSO
========

:doc:`start-system.sh </scripts/start-system>`, :doc:`build-oai-images.sh </scripts/build-oai-images>`
