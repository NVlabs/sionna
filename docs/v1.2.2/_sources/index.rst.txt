======
Sionna
======
.. include:: <isonum.txt>

`Sionna <https://github.com/NVlabs/sionna>`_\ |trade| is a hardware-accelerated differentiable open-source library for research
on communication systems. It is composed of the following modules:

- `Sionna RT <rt/index.html>`_: A lightning-fast stand-alone ray tracer for radio propagation modeling

- `Sionna PHY <phy/index.html>`_: A link-level simulator for wireless and optical communication systems

- `Sionna SYS <sys/index.html>`_: System-level simulation functionalities based on physical-layer abstraction

The core principles of Sionna are modularity, extensibility, and differentiability.

Every building block is an independent module that can be easily tested,
understood, and modified according to your needs. The documentation is complete
and includes references. Similar to constructing a deep neural network by
stacking different layers, complex communication system architectures can be
rapidly prototyped by connecting the desired blocks.

`Sionna PHY <phy/index.html>`_ and `Sionna SYS <sys/index.html>`_ are written in `Tensorflow <https://www.tensorflow.org>`_, while `Sionna RT <rt/index.html>`_ is built on top of `Mitsuba 3 <https://mitsuba.readthedocs.io/en/stable/>`_ and `Dr.Jit <https://drjit.readthedocs.io/en/stable/>`_. These frameworks provide automatic differentiation and can backpropagate gradients through an entire system. This is the key enabler for gradient-based optimization and machine learning, especially the integration of neural networks.

NVIDIA GPU acceleration provides orders-of-magnitude faster simulation, enabling
the interactive exploration of such systems, for example, in `Jupyter notebooks <https://jupyter.org/>`_ that can be run on cloud services such as `Google Colab <https://colab.research.google.com>`_. If no GPU is available, Sionna will run on the CPU.

The `Sionna Research Kit (SRK) <rk/index.html>`_ allows to deploy trained AI/ML components in a real software-defined 5G NR radio access network (RAN). It is based on the `OpenAirInterface <https://gitlab.eurecom.fr/oai/openairinterface5g>`_ project and is powered by the `NVIDIA Jetson AGX Thor platform <https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-thor/>`_.

Sionna is developed, continuously extended, and used by NVIDIA to drive 5G and 6G research.

.. toctree::
   :maxdepth: 7
   :hidden:

   installation
   rt/index
   phy/index
   sys/index
   rk/index
   made_with_sionna
   Discussions <https://github.com/NVlabs/sionna/discussions>
   Report an Issue <https://github.com/NVlabs/sionna/issues>
   Contribute <https://github.com/NVlabs/sionna/pulls>
   citation

.. toctree::
   :name: old-docs
   :caption: Older Versions
   :maxdepth: 1
   :hidden:

   v0.19.2 <https://jhoydis.github.io/sionna-0.19.2-doc>


License
*******
Sionna is Apache-2.0 licensed, as found in the `LICENSE <https://github.com/nvlabs/sionna/blob/main/LICENSE>`_ file.
