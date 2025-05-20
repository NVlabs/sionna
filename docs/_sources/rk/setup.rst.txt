.. _setup:

Setup
=====

This guide provides detailed instructions for setting up the Sionna Research Kit. It covers the hardware and software preparation required to run the kit as shown in :numref:`figure_system_overview`, including build instructions for the OpenAirInterface 5G software stack on arm64 platforms. If you need to deploy custom algorithms, these build instructions will be essential. For a jump-start, please refer to :ref:`quickstart` guide.

.. _figure_system_overview:

.. figure:: figs/system_overview.png
   :align: center
   :width: 600px
   :alt: 5G Stack Overview

   Schematic overview of the Sionna Research Kit using the Jetson AGX Orin platform and the OpenAirInterface 5G software stack.

The Sionna Research Kit is designed to run on the `NVIDIA Jetson AGX Orin platform <https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/>`_, which combines ARM CPUs with integrated GPU acceleration to enable efficient edge AI and accelerated 5G applications. While most tutorials can run on x86 systems with NVIDIA RTX GPUs, the Jetson platform's unified memory architecture makes it ideal for real-time applications by enabling seamless inline acceleration without the need for expensive memory copies.
In the case of x86 systems, we recommend Ubuntu 24.04 LTS with the latest NVIDIA drivers and Docker installation.


Platform Preparation
--------------------

.. _figure_system_setup2:

.. figure:: figs/system_setup_diagram.png
   :align: center
   :alt: Overview of the deployed setup.
   :width: 540px
   :name: system_overview2

   Overview of the deployed setup. See `Ettus OAI reference architecture <https://kb.ettus.com/OAI_Reference_Architecture_for_5G_and_6G_Research_with_USRP>`_ for details.


:numref:`figure_system_setup2` shows the setup of the Sionna Research Kit consisting of a USRP, a Quectel modem, and a Jetson AGX Orin. Please note that RF cables, splitters/combiners, attenuators, and/or antennas are required to connect the components. See the :ref:`bom` for detailed hardware recommendations.

The following steps will guide you through the detailed setup of the system:

.. toctree::
   :maxdepth: 1

   setup/bom
   setup/jetson
   setup/kernel
   setup/perf
   setup/UHD
   setup/sim
   setup/quectel

Software Configuration
----------------------

The Sionna Research Kit is built on the `OpenAirInterface 5G stack <https://openairinterface.org/>`_ and provides patches to ensure compatibility with the Jetson platform. If you plan to deploy your own custom algorithms, you need to rebuild the containers from source which requires a few careful changes to ensure compatibility with the arm64 platform.

The following steps are required to setup the software components:

.. toctree::
   :maxdepth: 1

   setup/OAI
   setup/5g_config
   setup/rfsimulator
   setup/sionna


Your First Call
---------------

You can now run your first data transmission. Please ensure that your system is wired as shown in :numref:`figure_system_setup2`. To verify that the system is working as expected, we will perform a few performance tests and monitor the system load. This is detailed in the following steps:

.. toctree::
   :maxdepth: 1

   setup/perf_test


Congratulations - your system is now ready! You can now have your first call over your private 5G network.

For inspiration and as blueprint for your own experiments, you can try the following tutorials:

* Explore the :ref:`accelerated_ldpc` tutorial and learn about the accelerated RAN
* Learn about :ref:`data_acquisition` to generate training data for your own AI/ML models
* Discover the toolchain from training in Sionna to the :ref:`neural_demapper` in a real 5G network
* Run a :ref:`ue_emulator`


Documentation of Scripts
------------------------

The remainder of this guide provides a reference for the scripts available in the `scripts` directory.

Setup Scripts
^^^^^^^^^^^^^

.. toctree::
   :maxdepth: 1

   scripts/quickstart-oai
   scripts/quickstart-cn5g
   scripts/configure-system
   scripts/install-usrp

Linux Kernel Customization
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. toctree::
   :maxdepth: 1

   scripts/build-custom-kernel
   scripts/install-custom-kernel

Configuration Files
^^^^^^^^^^^^^^^^^^^

.. toctree::
   :maxdepth: 1

   scripts/start-system
   scripts/stop-system
   scripts/generate-configs

Building Docker Images
^^^^^^^^^^^^^^^^^^^^^^^

.. toctree::
   :maxdepth: 1

   scripts/build-cn5g-images
   scripts/build-oai-images

Development / Patch tracking
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. toctree::
   :maxdepth: 1

   scripts/get-config-changes
   scripts/get-oai-changed-files
   scripts/get-oai-cn5g-changed-files
   scripts/get-oai-commit-versions
   scripts/get-oai-cn5g-commit-versions
