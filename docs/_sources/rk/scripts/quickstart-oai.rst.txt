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
        [--use-existing]
        [--debug]
        [--no-build]
        [--tag <tagname>]
        [--oai-version <version>]
        [--ci]
        [--source <kit-rootdir>]
        [--dest <openairinterface5g_dir>]

DESCRIPTION
===========

Perform the steps needed to end with a set of working Docker images for OpenAirInterface. The images can then be started using one of the provided configurations using the start-system script.

By default, the script clones the OpenAirInterface repository at the specified version, applies patches from the Sionna Research Kit, and calls Docker to build the images. It can instead build from an existing OpenAirInterface checkout.

OPTIONS
=======

.. option:: --source <kit-rootdir>

    Specify the root directory of Sionna Research Kit. This will be used to locate the required patches for OpenAirInterface. Default is the current directory.

.. option:: --dest <openairinterface5g_dir>

    Specify the destination directory for the OpenAirInterface code. Code will be cloned and patched here. Default is ``ext/openairinterface5g``.

.. option:: --use-existing

    Use the existing OpenAirInterface checkout at ``--dest`` instead of cloning and patching a repository. The destination must be a valid OpenAirInterface Git checkout. This option cannot be used with ``--clean``. The ``--oai-version`` option does not apply when using an existing checkout.

.. option:: --oai-version <version>

    Specify the OAI version/branch to checkout. Default is ``2025.w34``.

.. option:: --tag <tagname>

    Use <tagname> for the created Docker images. Default is ``latest``.

.. option:: --clean

    Remove the OpenAirInterface directory before proceeding, then clone and patch a fresh checkout. This option cannot be used with ``--use-existing``. If the directory exists and neither option is specified, the script will abort.

.. option:: --no-build

    Skip the build step of the Docker images.

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
    ./quickstart-oai.sh --use-existing --dest ./ext/openairinterface5g

SEE ALSO
========

:doc:`start_system.sh <start-system>`, :doc:`build-oai-images.sh <build-oai-images>`
