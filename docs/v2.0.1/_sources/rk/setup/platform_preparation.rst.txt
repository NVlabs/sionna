.. _platform_preparation:

Platform Preparation
====================

.. _figure_system_setup2:

.. figure:: ../figs/system_setup_diagram.png
   :align: center
   :alt: Overview of the deployed setup.
   :width: 540px
   :name: system_overview2

   Overview of the deployed setup. See `Ettus OAI reference architecture <https://kb.ettus.com/OAI_Reference_Architecture_for_5G_and_6G_Research_with_USRP>`_ for details.


:numref:`figure_system_setup2` shows the setup of the Sionna Research Kit consisting of a USRP, a Quectel modem, and an NVIDIA DGX Spark. Please note that RF cables, splitters/combiners, attenuators, and/or antennas are required to connect the components. See the :ref:`bom` for detailed hardware recommendations.

The following steps will guide you through the detailed setup of the system:

.. toctree::
   :maxdepth: 1

   bom
   spark
   jetson_thor
   jetson_orin
   kernel
   perf
   UHD
   x410
   sim
   rfsimulator
   quectel
