======================
quickstart-cn5g.sh
======================

SYNOPSIS
========

.. program:: quickstart-cn5g.sh

.. code-block:: bash

    quickstart-cn5g.sh 
        [--arch (x86|arm64)] 
        [--clean] 
        [-h|--help] 
        [--no-build] 
        [--tag <tagname>] 
        --source <kit-rootdir> 
        --dest <oai-cn5g-fed_dir>

DESCRIPTION
===========

Start from scratch and performs the steps needed to end with a set of working Docker images for the Core Network. The images can then be started using one of the provided configurations using the start-system script.

The script clones the OpenAirInterface Core Network Federated repository (currently v2.0.1), applies patches if needed, and calls Docker buildx to build the images. It uses buil-cn5g-images for the build.

OPTIONS
=======

.. option:: --source <kit-rootdir>

    Specify the root directory of Sionna Research Kit. This will be used to locate the required patches for the Core Network. Default is the current directory.

.. option:: --dest <oai-cn5g-fed_dir>

    Specify the destination directory for the Core Network code. Code will be clone and patched here. Default is 'ext/oai-cn5g-fed'.

.. option:: --clean

    Remove the Core Network directory before proceeding. If the directory exists and the flag is not specified, the script will abort.

.. option:: --no-build

    Skip the build step of the Docker images. Default is to build the images.

.. option:: --tag <tagname>

    Use <tagname> for the created Docker images. Default is 'latest'.

.. option:: --arch (x86 | arm64)

    The variant of the Docker images to build. 'x86' is unpatched OAI intended for x86_64 tests. 'arm64' is for CPU-only images compatible. Default is to parse the system architecture.

EXAMPLES
========

.. code-block:: bash

    quickstart-cn5g.sh 
    quickstart-cn5g.sh --clean --source . 
    quickstart-cn5g.sh --clean --arch arm64 --source . --dest ext/oai

SEE ALSO
========

:doc:`start-system.sh </scripts/start-system>` , :doc:`build-cn5g-images.sh </scripts/build-cn5g-images>`
