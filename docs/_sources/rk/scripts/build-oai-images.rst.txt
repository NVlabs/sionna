======================
build-oai-images.sh
======================

SYNOPSIS
========

.. program:: build-oai-images.sh

.. code-block:: bash

    build-oai-images.sh
        [-h|--help]
        [--debug]
        [--tag <tagname>]
        [--no-cache]
        [--ci]
        [--force-platform <platform>]
        <openairinterface5g_dir>

DESCRIPTION
===========

This is a wrapper script that selects the Dockerfile images from the OpenAirInterface directory and builds the requested Docker images based on the architecture, and tags the resulting images accordingly.

It generates images for ran-base, ran-build, gNB (oai-gnb), and UE (oai-nr-ue).

CUDA images will have the -cuda postfix: ran-base-cuda, ran-build-cuda, oai-gnb-cuda, oai-nr-ue-cuda.

This script does not support cross-compilation (images must be generated on the same platform as they target).

OPTIONS
=======

.. option:: <openairinterface5g_dir>

    The directory containing the OpenAirInterface source code. Must be specified.

.. option:: --tag <tagname>

    Build the images and tag them using <tagname>. Default is ``latest``.

.. option:: --debug

    Enable verbose build output (``--progress plain``).

.. option:: --no-cache

    Build images without using Docker cache.

.. option:: --ci

    Enable CI mode for automated builds.

.. option:: --force-platform <platform>

    Force a specific platform configuration for the build.

.. option:: -h, --help

    Display help message and exit.

EXAMPLES
========

.. code-block:: bash

    ./build-oai-images.sh ./ext/openairinterface5g
    ./build-oai-images.sh --tag experimental ./ext/openairinterface5g
    ./build-oai-images.sh --debug --no-cache ./ext/openairinterface5g

SEE ALSO
========

:doc:`quickstart-oai.sh </scripts/quickstart-oai>`, :doc:`start-system.sh </scripts/start-system>`
