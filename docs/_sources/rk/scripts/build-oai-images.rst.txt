======================
build-oai-images.sh
======================

SYNOPSIS
========

.. program:: build-oai-images.sh

.. code-block:: bash

    build-oai-images.sh 
        [-h|--help] 
        [--tag <tagname>] 
        [--arch (x86|arm64|cuda)] 
        <openairinterface5g_dir>

DESCRIPTION
===========

This is a wrapper script that selects the Dockerfile images from the OpenAirInterface directory and builds the requested Docker images based on the architecture, and tag the resulting image accordingly.

It generates images for ran-base (ran-base), ran-build (ran-build), gNB (oai-gnb) and UE (oai-nr-ue).

CUDA images will have the -cuda postfix: ran-base-cuda, ran-build-cuda, oai-gnb-cuda, oai-nr-ue-cuda.

This script does not support cross-compilation (images must be generated in the same platform as they target).

OPTIONS
=======

.. option:: <openairinterface5g_dir>

    The directory containing the OpenAirInterface source code. Must be specified.

.. option:: -h, --help

    Display help message and exit.

.. option:: --tag <tagname>

    Build the images and tag them using <tagname>. Default is 'latest'.

.. option:: --arch (x86|arm64|cuda)

    Select the target architecture. 'x86' will select regular Dockerfiles. 'arm64' will target CPU-only, ARM64 platforms of the Sionna Research Kit. 'cuda' will target GPU-accelerated, ARM64 platforms. Default is 'cuda'

EXAMPLES
========

.. code-block:: bash

    build-oai-images.sh ./ext/openairinterface5g
    build-oai-images.sh --arch arm64 ./ext/openairinterface5g
    build-oai-images.sh --arch cuda --tag experimental ./ext/openairinterface5g

SEE ALSO
========

:doc:`quickstart-oai.sh </scripts/quickstart-oai>` ,  :doc:`start-system.sh </scripts/start-system>`
