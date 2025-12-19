.. _quickstart-cn5g:

======================
quickstart-cn5g.sh
======================

SYNOPSIS
========

.. program:: quickstart-cn5g.sh

.. code-block:: bash

    quickstart-cn5g.sh
        [-h|--help]
        [--arch (x86|arm64)]
        [--branch <branchname>]
        [--clean]
        [--debug]
        [--no-build]
        [--tag <tagname>]
        [--source <kit-rootdir>]
        [--dest <oai-cn5g-fed_dir>]

DESCRIPTION
===========

Start from scratch and perform the steps needed to end with a set of working Docker images for the 5G Core Network. The images can then be started using one of the provided configurations using the start-system script.

The script clones the OpenAirInterface Core Network Federated repository, applies patches if needed, and calls Docker to build the images.

OPTIONS
=======

.. option:: --source <kit-rootdir>

    Specify the root directory of Sionna Research Kit. This will be used to locate the required patches for the Core Network. Default is the current directory.

.. option:: --dest <oai-cn5g-fed_dir>

    Specify the destination directory for the Core Network code. Code will be cloned and patched here. Default is ``ext/oai-cn5g-fed``.

.. option:: --branch <branchname>

    Specify the branch/version to checkout. Default is ``v2.1.0-1.2``.

.. option:: --tag <tagname>

    Use <tagname> for the created Docker images. Default is the branch name.

.. option:: --arch (x86|arm64)

    The variant of the Docker images to build. ``x86`` targets x86_64 systems. ``arm64`` targets ARM64 platforms. Default is to auto-detect from system architecture.

.. option:: --clean

    Remove the Core Network directory before proceeding. If the directory exists and the flag is not specified, the script will abort.

.. option:: --no-build

    Skip the build step of the Docker images. Only clone and patch the repository.

.. option:: --debug

    Enable debug output during the build process.

.. option:: -h, --help

    Display help message and exit.

EXAMPLES
========

.. code-block:: bash

    ./quickstart-cn5g.sh
    ./quickstart-cn5g.sh --clean
    ./quickstart-cn5g.sh --clean --arch arm64
    ./quickstart-cn5g.sh --branch v2.1.0-1.2 --tag latest

SEE ALSO
========

:doc:`start-system.sh </scripts/start-system>`, :doc:`build-cn5g-images.sh </scripts/build-cn5g-images>`
