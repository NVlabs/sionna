======================
build-cn5g-images.sh
======================

SYNOPSIS
========

.. program:: build-cn5g-images.sh

.. code-block:: bash

    build-cn5g-images.sh
        [-h|--help]
        [-d|--debug]
        [--tag <tagname>]
        <oai-cn5g_dir>

DESCRIPTION
===========

This is a wrapper script to build the Docker images of the 5G Core Network, optionally tagging them as required.

This will create the following images: oai-amf, oai-smf, oai-nrf, oai-ausf, oai-udm, oai-udr, oai-nssf, oai-upf, and trf-gen-cn5g.

This script does not support cross-compilation (images must be generated on the same platform as they target).

OPTIONS
=======

.. option:: <oai-cn5g_dir>

    The directory containing the Core Network source code. Must be specified.

.. option:: --tag <tagname>

    Build the images and tag them using <tagname>. Default is ``v2.0.1``.

.. option:: -d, --debug

    Enable verbose build output (``--progress plain``).

.. option:: -h, --help

    Display help message and exit.

EXAMPLES
========

.. code-block:: bash

    ./build-cn5g-images.sh ./ext/oai-cn5g-fed
    ./build-cn5g-images.sh --tag latest ./ext/oai-cn5g-fed
    ./build-cn5g-images.sh --debug ./ext/oai-cn5g-fed

SEE ALSO
========

:doc:`quickstart-cn5g.sh </scripts/quickstart-cn5g>`, :doc:`start-system.sh </scripts/start-system>`
