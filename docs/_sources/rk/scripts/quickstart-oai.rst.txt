======================
quickstart-oai.sh
======================

SYNOPSIS
========

.. program:: quickstart-oai.sh

.. code-block:: bash

    quickstart-oai.sh 
        [--arch (x86|arm64|cuda)] 
        [--clean] 
        [-h|--help] 
        [--no-build] 
        [--tag <tagname>] 
        --source <kit-rootdir> 
        --dest <openairinterface5g_dir>

DESCRIPTION
===========

Start from scratch and performs the steps needed to end with a set of working Docker images for OpenAirInterface. The images can then be started using one of the provided configurations using the start-system script.

The script clones the OpenAirInterface repository, applies patches if needed, and calls Docker buildx to build the images. It uses buil-oai-images for the build.

OPTIONS
=======

.. option:: --source <kit-rootdir>

    Specify the root directory of Sionna Research Kit. This will be used to locate the required patches for OpenAirInterface. Default is the current directory.

.. option:: --dest <openairinterface5g_dir>

    Specify the destination directory for the OpenAirInterface code. Code will be clone and patched here. Default is 'ext/openairinterface5g'.

.. option:: --clean

    Remove the OpenAirInterface directory before proceeding. If the directory exists and the flag is not specified, the script will abort.

.. option:: --no-build

    Skip the build step of the Docker images. Default is to build the images.

.. option:: --tag <tagname>

    Use <tagname> for the created Docker images. Default is 'latest'.

.. option:: --arch (x86 | arm64 | cuda)

    The variant of the Docker images to build. 'x86' is unpatched OAI intended for x86_64 tests. 'arm64' is for CPU-only images compatible. 'cuda' includes support for GPU acceleration. Default is to parse the system architecture.

EXAMPLES
========

.. code-block:: bash

    quickstart-oai.sh 
    quickstart-oai.sh --clean --source . 
    quickstart-oai.sh --clean --arch cuda --source . --dest ext/oai

SEE ALSO
========

:doc:`start-system.sh </scripts/start-system>` , :doc:`build-oai-images.sh </scripts/build-oai-images>`

