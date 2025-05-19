.. _neural_demapper:

Integration of a Neural Demapper
================================

.. figure:: ../../figs/tutorial_nn_demapper_overview.png
   :align: center
   :width: 600px
   :alt: Neural Demapper Overview

In this tutorial, we will integrate a neural network-based demapper into the signal processing pipeline of the physical uplink shared channel (PUSCH). We use `NVIDIA TensorRT <https://developer.nvidia.com/tensorrt>`_ for the accelerated inference and train the neural network using `Sionna <https://nvlabs.github.io/sionna/>`_. Further, training with real world data can be realized by leveraging the previous :ref:`data_acquisition` tutorial.

Note that main purpose of this tutorial is to show the efficient integration of neural components into the 5G stack. The block error-rate performance gains are not significant for this simple example. However, similar to the :ref:`accelerated_ldpc` tutorial, the integration requires careful consideration of the memory transfer patterns between the CPU and the GPU to keep the latency as low as possible.

In this tutorial, we will learn:

* How to export the trained model from Sionna to TensorRT
* How to integrate the TensorRT engine into the 5G stack
* CUDA graphs for latency reductions


The tutorial is split into two parts:

.. toctree::
   :maxdepth: 1

    Part 1: Neural Demapper Training and TensorRT Export<Neural_demapper.ipynb>
    Part 2: GPU-Accelerated Inference<accelerated_inference.rst>

References
----------

.. [Gadiyar2023] R. Gadiyar, L. Kundu, J. Boccuzzi, `"Building Software-Defined, High-Performance, and Efficient vRAN Requires Programmable Inline Acceleration" <https://developer.nvidia.com/blog/building-software-defined-high-performance-and-efficient-vran-requires-programmable-inline-acceleration>`_
      NVIDIA Developer Blog, 2023.

.. [Kundu2023] L. Kundu, et al., `"Hardware Acceleration for Open Radio Access Networks: A Contemporary Overview," <https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10310082>`_
       IEEE Communications Magazine, vol. 62, no. 9, pp. 160-167, 2023.

.. [Gray2019] `Getting Started with CUDA Graphs <https://developer.nvidia.com/blog/cuda-graphs/>`_.

.. [Schibisch2018] S. Schibisch et al., `"Online Label Recovery for Deep Learning-based Communication through Error Correcting Codes," <https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8491189>`_
      IEEE ISWCS, 2018.
