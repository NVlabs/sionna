.. _data_acquisition:

Plugins & Data Acquisition
==========================

.. figure:: ../../figs/tutorial_capture_overview.png
   :align: center
   :width: 600px
   :alt: Data capture overview

   Schematic overview of the 5G NR PUSCH. Note that this is a simplified view showing only the relevant components for the following tutorial. For simplicity, MIMO aspects are not shown.


This tutorial explains how to capture and record real-world 5G signals using the Sionna Research Kit. It summarizes the OAI dynamic module loader [OAILib]_ and explains how to generate custom plugins to replace existing functions. This simplifies the integration of custom code - such as data capturing or a neural demapper - in the OAI stack. In preparation of the :ref:`neural_demapper` tutorial, we focus on capturing complex-valued IQ symbols before and after demapping.

The advantage of plugins is that they can be loaded and unloaded dynamically, which allows for a flexible integration of custom code in the OAI stack.


The tutorial is structured as follows:

.. toctree::
    :maxdepth: 1

    part1_create_plugin.rst
    part2_capture_data.rst

.. _references_data_acquisition:

References
----------

.. [OAILib] `OAI Shared Library Loader <https://github.com/OPENAIRINTERFACE/openairinterface5g/blob/develop/common/utils/DOC/loader.md>`_
