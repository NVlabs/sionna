======================
build-cn5g-images.sh
======================

SYNOPSIS
========

.. program:: build-cn5g-images.sh

.. code-block:: bash

    build-cn5g-images [-h|--help] [--tag <tagname>] <oai-cn5g_dir>

DESCRIPTION
===========

This is a wrapper script to build the Docker images of the Core Network, optionally tagging them as required.

This will create the following images: oai-amf, oai-smf, oai-nrf, oai-ausf, oai-udm, oai-udr, oai-nssf, oai-upf and trf-gen-cn5g.

This script does not support cross-compilation (images must be generated in the same platform as they target).

OPTIONS
=======

.. option:: <oai-cn5g_dir>

    The directory containing the Core Network source code. Must be specified.

.. option:: -h, --help

    Display help message and exit.

.. option:: --tag <tagname>

    Build the images and tag them using <tagname>. Default is 'latest'.

EXAMPLES
========

.. code-block:: bash

    build-cn5g-images.sh ./ext/oai-cn5g-fed

SEE ALSO
========

:doc:`quickstart-cn5g.sh </scripts/quickstart-cn5g>` ,  :doc:`start-system.sh </scripts/start-system>`
