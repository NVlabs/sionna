=================
Research Kit (RK)
=================
.. include:: <isonum.txt>

The `NVIDIA Sionna Research Kit <https://github.com/NVlabs/sionna-rk>`_, powered by the `NVIDIA Jetson Platform <https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/>`_, is the pioneering solution that integrates in-line AI/ML accelerated computing with the adaptability of a software-defined Radio Access Network (RAN). Leveraging `OpenAirInterface <https://openairinterface.org>`_, it has `O-RAN <https://www.o-ran.org/>`_-compliant interfaces, providing extensive research opportunities—from 5G NR and O-RAN over real-world data acquisition to the deployment of cutting-edge `AI-RAN <https://ai-ran.org/>`_ algorithms for 6G.

Created by the team behind `Sionna <https://github.com/NVlabs/sionna>`_, the
Sionna Research Kit features textbook-quality tutorials. In just an afternoon,
you will connect commercial 5G equipment to a network using your own
customizable transceiver algorithms. Conducting AI-RAN experiments, whether simulated, cabled,
or over-the-air, has never been more accessible and more affordable.

.. image:: figs/sionna_overview.png
   :alt: System overview
   :width: 800px
   :align: center

|

.. rubric:: Interoperable with Sionna

Real-world data captured with the Sionna Research Kit can be used to train
machine learning models developed with Sionna. These models can then be deployed on
the Sionna Research Kit using the `NVIDIA TensorRT framework
<https://developer.nvidia.com/tensorrt>`_. The rapid prototyping of complex
communication systems and their real-time deployment have never been simpler.


.. rubric:: Democratizing AI-RAN Research

The Sionna Research Kit, with its low barrier to entry, is an excellent tool for
onboarding and educating the next generation of telecommunications engineers.
You can easily run your own private 5G network.  In addition, the NVIDIA
platform, with its GPU-accelerated libraries combined with unified memory,
provides a powerful environment for experimenting with AI-RAN algorithms. Beyond
TensorRT, the platform supports CUDA for creating high-performance
GPU-accelerated applications. Based on open-source software, researchers across
academia and industry can develop and test new algorithms within a real-world 5G
NR system and beyond.


.. rubric:: Hardware

We recommend the `NVIDIA Jetson AGX Orin
<https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/>`_
and `USRPs from Ettus Research <https://www.ettus.com/products/>`_ which are commonly
used in universities and research institutions. Refer to the :ref:`bom` for more
details. Upon release, `NVIDIA DGX Spark
<https://www.nvidia.com/en-us/products/workstations/dgx-spark/>`_ will be
supported, too.

The Jetson AGX Orin platform is ideal for experimenting with AI acceleration and deploying novel AI-RAN algorithms - even the code and tutorials of the Sionna Research Kit have been developed on the Jetson platform.

.. rubric:: Table of Contents

.. toctree::
   :maxdepth: 2

   quickstart
   setup
   tutorials
   Get the Code <https://github.com/NVlabs/sionna-rk>


.. rubric:: License and Citation

The software available under this repository is governed by the Apache 2.0 license as found in the `LICENSE <https://github.com/NVlabs/sionna-rk/blob/main/LICENSE>`_ file.

In connection with your use of this software, you may receive links to third-party technology, and your use of third-party technology may be subject to third-party terms, privacy statements or practices. NVIDIA is not responsible for the terms, privacy statements or practices of third parties. You acknowledge and agree that it is your sole responsibility to obtain any additional third-party licenses required to make, have made, use, have used, sell, import, and offer for sale products or services that include or incorporate any third-party technology. NVIDIA does not grant to you under the project license any necessary patent or other rights, including standard essential patent rights, with respect to any such third-party technology.

If you use this software, please cite it as:

.. code:: bibtex

   @software{sionna-rk,
    title = {Sionna Research Kit},
    author = {Cammerer, Sebastian, and Marcus, Guillermo and Zirr, Tobias and Hoydis, Jakob and {Ait Aoudia}, Fayçal and Wiesmayr, Reinhard and Maggi, Lorenzo and Nimier-David, Merlin and Keller, Alexander},
    note = {https://nvlabs.github.io/sionna/rk/index.html},
    year = {2025},
    version = {1.0.0}
   }
