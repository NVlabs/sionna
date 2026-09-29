.. _software_configuration:

Software Configuration
======================

The Sionna Research Kit is built on the `OpenAirInterface 5G stack <https://openairinterface.org/>`_ and provides patches to ensure compatibility with the DGX Spark platform. If you plan to deploy your own custom algorithms, you need to rebuild the containers from source which requires a few careful changes to ensure compatibility with the arm64 platform.

The following steps are required to setup the software components:

.. toctree::
   :maxdepth: 1

   OAI
   5g_config
   sionna
