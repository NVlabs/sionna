.. _OAI:

OpenAirInterface Setup
======================

.. _fig_5g_stack:
.. figure:: ../figs/5g_stack.png
   :align: center
   :alt: 5G Stack Overview

   Overview of the deployed 5G end-to-end stack with IP adresses and interfaces of each container. Figure from `OpenAirInterface <https://gitlab.eurecom.fr/oai/openairinterface5g/-/blob/develop/ci-scripts/yaml_files/5g_rfsimulator/README.md#2-deploy-containers>`_.

The 5G stack is deployed as a set of Docker containers. The following steps will deploy the core network components and the RAN components. :numref:`fig_5g_stack` shows the block diagram of the complete system including the network interfaces and IP addresses of each container (see `OpenAirInterface5G <https://gitlab.eurecom.fr/oai/openairinterface5g/-/blob/develop/ci-scripts/yaml_files/5g_rfsimulator/README.md#2-deploy-containers>`_ for a detailed introduction).

Required Components
-------------------

As visualized in :numref:`fig_5g_stack`, the following components are required to run the complete 5G stack:

* Radio Access Network (RAN)
* Core Network (CN)
* Optional: User Equipment (UE)
* Testing tools

Core Network
^^^^^^^^^^^^

The following components are part of the `OAI 5G core network <https://gitlab.eurecom.fr/oai/cn5g/oai-cn5g-smf>`_:

* ``oai-amf`` - Access and Mobility Management Function
* ``oai-smf`` - Session Management Function
* ``oai-upf`` - User Plane Function
* ``oai-nrf`` - Network Repository Function
* ``oai-udr`` - Unified Data Repository
* ``oai-udm`` - Unified Data Management
* ``oai-ausf`` - Authentication Server Function

See the `OAI documentation <https://gitlab.eurecom.fr/oai/cn5g/oai-cn5g-fed/-/tree/master/docs>`_ for further details.

RAN Components
^^^^^^^^^^^^^^
* ``oai-gnb`` - gNodeB implementation
* ``oai-nr-ue`` - 5G NR UE implementation
* ``trf-gen-cn5g`` - Traffic generator (only for testing)


TLDR
----

The whole process of getting the source code, patching, creating the configuration files and building the docker images is automated with two scripts: ``quickstart-oai.sh`` for OAI images and ``quickstart-cn5g.sh`` for the 5G Core Network images. The rest of the document goes over the different steps performed by these scripts in detail.

.. code-block:: bash

   scripts/quickstart-oai.sh
   scripts/quickstart-cn5g.sh
   scripts/generate-configs.sh --rk-dir . --oai-dir ext/openairinterface5g --dest ./configs

.. _build_oai:

Getting the Source Code
-----------------------

.. note::
   Start with a clean checkout to avoid build issues from residual CMake files.

Clone this specific version of OAI:

.. code-block:: bash

   # This should match git commit 46a1d2a621
   git clone --branch 2024.w34 https://gitlab.eurecom.fr/oai/openairinterface5g.git ext/openairinterface5g

And the OAI 5G Core Network sources:

.. code-block:: bash

   git clone --branch v2.0.1 https://gitlab.eurecom.fr/oai/cn5g/oai-cn5g-fed.git ext/oai-cn5g-fed

   cd ext/oai-cn5g-fed
   ./scripts/syncComponents.sh \
      --nrf-branch "v2.0.1" \
      --amf-branch "v2.0.1" \
      --smf-branch "v2.0.1" \
      --upf-branch "v2.0.1" \
      --ausf-branch "v2.0.1" \
      --udm-branch "v2.0.1" \
      --udr-branch "v2.0.1" \
      --upf-vpp-branch "v2.0.1" \
      --nssf-branch "v2.0.1" \
      --nef-branch "v2.0.1" \
      --pcf-branch "v2.0.1"
   cd ../..

Patching
--------

Changes required are deployed as patches applied to the existing repositories of OAI and the core network. To apply the patches, do:

.. code-block:: bash

   # Apply OAI patches
   cd ext/openairinterface5g
   git apply < ../../patches/openairinterface5g.patch

   # Install tutorials
   git apply < ../../patches/tutorials.patch
   cd ../..

   # Apply OAI CN patches
   scripts/patch-oai-cn5g.sh --patch ./patches/oai-cn5g.patch --dest ext/oai-cn5g-fed

If you would like to explore the modified files, you can extract them with:

.. code-block:: bash

   # extract OAI changed files
   scripts/get-oai-changed-files.sh --source ext/openairinterface5g --dest files/openairinterface5g

   # extract OAI CN changed files
   scripts/get-oai-cn5g-changed-files.sh --source ext/oai-cn5g-fed --dest files/oai-cn5g

Create configuration files
--------------------------

Configuration files are only required after the images are built, but are created from OAI base files and therefore require a source tree. They are recreated as:

.. code-block:: bash

   # create configuration files
   scripts/generate-configs.sh --rk-dir . --oai-dir ext/openairinterface5g --dest ./configs

Building OAI RAN Images
-----------------------

.. note::
   Start with a clean checkout to avoid build issues from residual CMake files.

   You need a Docker version 23 or newer with BuildKit/BuildX enabled by default, which supports multi-stage image definitions.

   To display the complete build process output (useful for debugging), add ``--progress plain`` to the docker build command.

The following subsection are equivalent to running the following script. The script can invoked multiple times and be used to rebuild the images as needed when doing development and experiments.

.. code-block:: bash

   # use --arch and --tag if needed.
   scripts/build-oai-images.sh ext/openairinterface5g

Similarly, you can rebuild the Core Network images with the following script, but this should not be needed unless you update the repos:

.. code-block:: bash

   # use --tag if needed.
   scripts/build-cn5g-images.sh ext/oai-cn5g-fed

Building for aarch64
^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   cd ext/openairinterface5g

   docker build --target ran-base --tag ran-base:latest --file docker/Dockerfile.base.ubuntu22.aarch64 .
   docker build --target ran-build --tag ran-build:latest --file docker/Dockerfile.build.ubuntu22.aarch64 .

   docker build --target oai-gnb --tag oai-gnb:latest --file docker/Dockerfile.gNB.ubuntu22.aarch64 .
   docker build --target oai-nr-ue --tag oai-nr-ue:latest --file docker/Dockerfile.nrUE.ubuntu22.aarch64 .

.. _build_oai_cuda:

Building for aarch64 with CUDA support
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   cd ext/openairinterface5g

   docker build --target ran-base-cuda --tag ran-base-cuda:latest --file docker/Dockerfile.base.cuda.aarch64 .
   docker build --target ran-build-cuda --tag ran-build-cuda:latest --file docker/Dockerfile.build.cuda.aarch64 .

   docker build --target oai-gnb-cuda --tag oai-gnb-cuda:latest --file docker/Dockerfile.gNB.cuda.aarch64 .
   docker build --target oai-nr-ue-cuda --tag oai-nr-ue-cuda:latest --file docker/Dockerfile.nrUE.cuda.aarch64 .


Building OAI 5G Core Images
---------------------------

See `OAI Core Build Image Guide <https://gitlab.eurecom.fr/oai/cn5g/oai-cn5g-fed/-/blob/master/docs/BUILD_IMAGES.md>`_ for details.

A summary of the commands to build the Docker images is given below:

.. code-block:: bash

   # Clone the main repository
   git clone --branch v2.0.1 https://gitlab.eurecom.fr/oai/cn5g/oai-cn5g-fed.git

   # Fetch submodules (specific versions can be set if needed)
   cd ext/oai-cn5g-fed
   ./scripts/syncComponents.sh \
      --nrf-branch "v2.0.1" \
      --amf-branch "v2.0.1" \
      --smf-branch "v2.0.1" \
      --upf-branch "v2.0.1" \
      --ausf-branch "v2.0.1" \
      --udm-branch "v2.0.1" \
      --udr-branch "v2.0.1" \
      --upf-vpp-branch "v2.0.1" \
      --nssf-branch "v2.0.1" \
      --nef-branch "v2.0.1" \
      --pcf-branch "v2.0.1"

   # Build Docker images

   # AMF
   docker build --target oai-amf --tag oai-amf:latest --file component/oai-amf/docker/Dockerfile.amf.ubuntu --build-arg BASE_IMAGE=ubuntu:22.04 component/oai-amf

   # SMF
   docker build --target oai-smf --tag oai-smf:latest --file component/oai-smf/docker/Dockerfile.smf.ubuntu --build-arg BASE_IMAGE=ubuntu:22.04 component/oai-smf
   # Alternatively, add CPR_FORCE_USE_SYSTEM_CURL flag
   docker build --target oai-smf --tag oai-smf:latest --file component/oai-smf/docker/Dockerfile.smf.ubuntu --build-arg BASE_IMAGE=ubuntu:22.04 --build-arg CPR_FORCE_USE_SYSTEM_CURL=ON component/oai-smf

   # NRF
   docker build --target oai-nrf --tag oai-nrf:latest --file component/oai-nrf/docker/Dockerfile.nrf.ubuntu --build-arg BASE_IMAGE=ubuntu:22.04 component/oai-nrf

   # AUSF
   docker build --target oai-ausf --tag oai-ausf:latest --file component/oai-ausf/docker/Dockerfile.ausf.ubuntu --build-arg BASE_IMAGE=ubuntu:22.04 component/oai-ausf

   # UDM
   docker build --target oai-udm --tag oai-udm:latest --file component/oai-udm/docker/Dockerfile.udm.ubuntu --build-arg BASE_IMAGE=ubuntu:22.04 component/oai-udm

   # UDR
   docker build --target oai-udr --tag oai-udr:latest --file component/oai-udr/docker/Dockerfile.udr.ubuntu --build-arg BASE_IMAGE=ubuntu:22.04 component/oai-udr

   # NSSF
   docker build --target oai-nssf --tag oai-nssf:latest --file component/oai-nssf/docker/Dockerfile.nssf.ubuntu --build-arg BASE_IMAGE=ubuntu:22.04 component/oai-nssf

   # UPF
   docker build --target oai-upf --tag oai-upf:latest --file component/oai-upf/docker/Dockerfile.upf.ubuntu --build-arg BASE_IMAGE=ubuntu:22.04 component/oai-upf
   docker build --target oai-upf-vpp --tag oai-upf-vpp:latest --file component/oai-upf-vpp/docker/Dockerfile.upf-vpp.ubuntu --build-arg BASE_IMAGE=ubuntu:22.04 component/oai-upf-vpp

   # Build traffic generator
   cd ci-scripts
   docker build --target trf-gen-cn5g --tag trf-gen-cn5g:latest --file Dockerfile.traffic.generator.ubuntu .

Building Components Manually
----------------------------

The AMF can be built manually with the following commands:

.. code-block:: bash

   git clone --recurse-submodules https://gitlab.eurecom.fr/oai/cn5g/oai-cn5g-amf.git
   cd oai-cn5g-amf
   docker build --target oai-amf --tag oai-amf:latest --file docker/Dockerfile.amf.ubuntu .

The SMF can be built manually with the following commands:

.. code-block:: bash

   git clone --recurse-submodules https://gitlab.eurecom.fr/oai/cn5g/oai-cn5g-smf.git
   cd oai-cn5g-smf
   docker build --target oai-smf --tag oai-smf:latest --file docker/Dockerfile.smf.ubuntu .


In case the UDR container:

.. code-block:: bash

   docker build --target oai-udr --tag oai-udr:latest --file component/oai-udr/docker/Dockerfile.udr.ubuntu --build-arg BASE_IMAGE=ubuntu:22.04 component/oai-udr

In the case of the UPF, we need to rollback some submodules to a specific commit first:

.. code-block:: bash

   # Download the kernel source
   mkdir component/oai-upf/build/ext && pushd $_
   wget https://cdn.kernel.org/pub/linux/kernel/v5.x/linux-5.15.tar.gz
   popd

   # Set versions to match known working code
   pushd $(pwd)
   cd component/oai-upf
   git checkout 93cab8f
   cd src/common-src
   git checkout 588f79a
   popd

   # Apply patches
   cp sionna-rk/oai-cn5g/oai-upf/build/scripts/build_upf component/oai-upf/build/scripts/build_upf
   cp sionna-rk/oai-cn5g/oai-upf/build/scripts/build_helper.upf component/oai-upf/build/scripts/build_helper.upf
   cp sionna-rk/oai-cn5g/oai-upf/docker/Dockerfile.upf.ubuntu component/oai-upf/docker/Dockerfile.upf.ubuntu

   docker build --target oai-upf --tag oai-upf:latest --file component/oai-upf/docker/Dockerfile.upf.ubuntu --build-arg BASE_IMAGE=ubuntu:22.04 component/oai-upf

You should now have all relevant Docker images to run the complete 5G stack.

