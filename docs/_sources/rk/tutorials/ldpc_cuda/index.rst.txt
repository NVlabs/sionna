.. _accelerated_ldpc:

GPU-Accelerated LDPC Decoding
=============================

.. figure:: ../../figs/tutorial_ldpc_overview.png
   :align: center
   :width: 700px
   :alt: LDPC Acceleration Overview

   Schematic overview of the 5G NR PUSCH. Note that this is a simplified view showing only the relevant components for the following tutorials and uses the Sionna naming convention. For simplicity, the HARQ process and MIMO aspects are not shown.

In this tutorial, we will show how the low-density parity-check (LDPC) decoder of the physical layer can be accelerated using CUDA.
As wireless communications is a latency critical application, we will also discuss the performance implications of different memory sharing patterns between the CPU and the GPU. The unified CPU/GPU architecture of the Jetson Orin platform allows for efficient `inline acceleration <https://developer.nvidia.com/blog/building-software-defined-high-performance-and-efficient-vran-requires-programmable-inline-acceleration/>`_ without the need of explicit memory transfers.
For a more detailed discussion on different acceleration techniques, we refer the interested reader to [Kundu2023B]_.

You will learn:

* How to accelerate the LDPC decoder using CUDA
* The conceptual difference between inline and lookaside acceleration
* Different memory transfer patterns and their performance implications

.. toctree::
    :maxdepth: 1

    Part 1: Background & Python Reference Implementation<LDPC_background.ipynb>
    Part 2: CUDA Implementation<ldpc_cuda.rst>


.. note::

   The source code of this tutorial can be found in `tutorials/ldpc_cuda/runtime/ldpc_decoder.cu <https://github.com/NVlabs/sionna-rk/blob/main/tutorials/ldpc_cuda/runtime/ldpc_decoder.cu>`_.


References
----------

.. [Romani2020] L. Romani, `"Hardware Acceleration of 5G LDPC using Datacenter-class FPGAs," <https://webthesis.biblio.polito.it/16046/1/tesi.pdf>`_
      Politecnico di Torino, 2020.

.. [Pretti2005] M. Pretti, `"A Message Passing Algorithm with Damping,"`
      J. Statist. Mech.: Theory Practice, p. 11008, Nov. 2005.

.. [Nachmani2016] E. Nachmani, Y. Be'ery and D. Burshtein, `"Learning to Decode Linear Codes Using Deep Learning," <https://arxiv.org/pdf/1607.04793>`_
      IEEE Annual Allerton Conference on Communication, Control, and Computing, pp. 341-346., 2016.

.. [Kundu2023B] L. Kundu, et al., `"Hardware Acceleration for Open Radio Access Networks: A Contemporary Overview," <https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10310082>`_
      IEEE Communications Magazine, vol. 62, no. 9, pp. 160-167, 2023.

.. [Chen2005] J. Chen, et al., `"Reduced-complexity Decoding of LDPC Codes,"`
      IEEE Transactions on Communications, vol. 53, no. 8, Aug. 2005.

.. [Richardson2018] T. Richardson and S. Kudekar, `"Design of Low-density Parity-check Codes for 5G New Radio,"`
      IEEE Communications Magazine, vol. 56, no. 3, pp. 20-27, 2018.

.. [CUDA2024] `NVIDIA CUDA Programming Guide <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html>`_

.. [Tegra2024] `CUDA for Tegra Memory Model <https://docs.nvidia.com/cuda/cuda-for-tegra-appnote/index.html>`_

