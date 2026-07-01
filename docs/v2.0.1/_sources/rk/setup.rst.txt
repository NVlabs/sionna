.. _setup:

Setup
=====

This guide provides detailed instructions for setting up the Sionna Research Kit. It covers the hardware and software preparation required to run the kit as shown in :numref:`figure_system_overview`, including build instructions for the OpenAirInterface 5G software stack on arm64 platforms. If you need to deploy custom algorithms, these build instructions will be essential. For a jump-start, please refer to :ref:`quickstart` guide.

.. _figure_system_overview:

.. figure:: figs/system_overview.png
   :align: center
   :width: 600px
   :alt: 5G Stack Overview

   Schematic overview of the Sionna Research Kit using the NVIDIA DGX Spark platform and the OpenAirInterface 5G software stack.

The Sionna Research Kit is designed to run on an `NVIDIA DGX Spark <https://www.nvidia.com/en-us/products/workstations/dgx-spark/>`_, which combines ARM CPUs with integrated GPU acceleration to enable efficient edge AI and accelerated 5G applications. While most tutorials can run on x86 systems with NVIDIA RTX GPUs, the DGX Spark's unified memory architecture makes it ideal for real-time applications by enabling seamless inline acceleration without the need for expensive memory copies.
In the case of x86 systems, we recommend Ubuntu 24.04 LTS with the latest NVIDIA drivers and Docker installation.

.. toctree::
   :maxdepth: 2

   setup/platform_preparation
   setup/software_configuration
   setup/first_call
   setup/scripts_reference
   setup/system_upgrade
