==================
"Made with Sionna"
==================

We love to see how Sionna is used by other researchers! For this reason, you find below links to papers whose authors have also published Sionna-based simulation code.

List of Projects
----------------

If you want your paper and code be listed here, please send an email to `sionna@nvidia.com <mailto:sionna@nvidia.com>`_ with links to the paper (e.g., `arXiv <https://arxiv.org>`_) and code repository (e.g., `GitHub <https://github.com>`_).

Graph Neural Networks for Enhanced Decoding of Quantum LDPC Codes
*****************************************************************
.. made-with-sionna::
    :title: Graph Neural Networks for Enhanced Decoding of Quantum LDPC Codes
    :authors: Anqi Gong, Sebastian Cammerer, Joseph M. Renes
    :year: 2023
    :version: 0.15
    :link_arxiv: https://arxiv.org/abs/2310.17758
    :link_github: https://github.com/gongaa/Feedback-GNN
    :abstract: In this work, we propose a fully differentiable iterative decoder for quantum low-density parity-check (LDPC) codes. The proposed algorithm is composed of classical belief propagation (BP) decoding stages and intermediate graph neural network (GNN) layers. Both component decoders are defined over the same sparse decoding graph enabling a seamless integration and scalability to large codes. The core idea is to use the GNN component between consecutive BP runs, so that the knowledge from the previous BP run, if stuck in a local minima caused by trapping sets or short cycles in the decoding graph, can be leveraged to better initialize the next BP run. By doing so, the proposed decoder can learn to compensate for sub-optimal BP decoding graphs that result from the design constraints of quantum LDPC codes. Since the entire decoder remains differentiable, gradient descent-based training is possible. We compare the error rate performance of the proposed decoder against various post-processing methods such as random perturbation, enhanced feedback, augmentation, and ordered-statistics decoding (OSD) and show that a carefully designed training process lowers the error-floor significantly. As a result, our proposed decoder outperforms the former three methods using significantly fewer post-processing attempts.

Sionna RT: Differentiable Ray Tracing for Radio Propagation Modeling
********************************************************************
.. made-with-sionna::
    :title: Sionna RT: Differentiable Ray Tracing for Radio Propagation Modeling
    :authors: Jakob Hoydis, Fayçal Aït Aoudia, Sebastian Cammerer, Merlin Nimier-David, Nikolaus Binder, Guillermo Marcus, Alexander Keller
    :year: 2023
    :version: 0.16
    :link_arxiv: https://arxiv.org/abs/2303.11103
    :link_github: https://github.com/NVlabs/diff-rt
    :link_colab: https://colab.research.google.com/github/NVlabs/diff-rt/blob/master/Learning_Materials.ipynb
    :abstract: Sionna is a GPU-accelerated open-source library for link-level simulations based on TensorFlow. Its latest release (v0.14) integrates a differentiable ray tracer (RT) for the simulation of radio wave propagation. This unique feature allows for the computation of gradients of the channel impulse response and other related quantities with respect to many system  and environment parameters, such as material properties, antenna patterns, array geometries, as well as transmitter and receiver orientations and positions. In this paper, we outline the key components of Sionna RT and showcase example applications such as learning of radio materials and optimizing transmitter orientations by gradient descent. While classic ray tracing is a crucial tool for 6G research topics like reconfigurable intelligent surfaces, integrated sensing and communications, as well as user localization, differentiable ray tracing is a key enabler for many novel and exciting research directions, for example, digital twins.


DUIDD: Deep-Unfolded Interleaved Detection and Decoding for MIMO Wireless Systems
*********************************************************************************
.. made-with-sionna::
    :title: DUIDD: Deep-Unfolded Interleaved Detection and Decoding for MIMO Wireless Systems
    :authors: Reinhard Wiesmayr, Chris Dick, Jakob Hoydis, Christoph Studer
    :year: 2022
    :version: 0.11
    :link_arxiv: https://arxiv.org/abs/2212.07816
    :link_github: https://github.com/IIP-Group/DUIDD
    :abstract: Iterative detection and decoding (IDD) is known to achieve near-capacity performance in multi-antenna wireless systems. We propose deep-unfolded interleaved detection and decoding (DUIDD), a new paradigm that reduces the complexity of IDD while achieving even lower error rates. DUIDD interleaves the inner stages of the data detector and channel decoder, which expedites convergence and reduces complexity. Furthermore, DUIDD applies deep unfolding to automatically optimize algorithmic hyperparameters, soft-information exchange, message damping, and state forwarding. We demonstrate the efficacy of DUIDD using NVIDIA's Sionna link-level simulator in a 5G-near multi-user MIMO-OFDM wireless system with a novel low-complexity soft-input soft-output data detector, an optimized low-density parity-check decoder, and channel vectors from a commercial ray-tracer. Our results show that DUIDD outperforms classical IDD both in terms of block error rate and computational complexity.

Bit Error and Block Error Rate Training for ML-Assisted Communication
*********************************************************************
.. made-with-sionna::
    :title: Bit Error and Block Error Rate Training for ML-Assisted Communication
    :authors: Reinhard Wiesmayr, Gian Marti, Chris Dick, Haochuan Song, Christoph Studer
    :year: 2022
    :version: 0.11
    :link_arxiv: https://arxiv.org/pdf/2210.14103.pdf
    :link_github: https://github.com/IIP-Group/BLER_Training
    :abstract: Even though machine learning (ML) techniques are being
               widely used in communications, the question of how to train
               communication systems has received surprisingly little
               attention. In this paper, we show that the commonly used binary
               cross-entropy (BCE) loss is a sensible choice in uncoded
               systems, e.g., for training ML-assisted data detectors, but may
               not be optimal in coded systems. We propose new loss functions
               targeted at minimizing the block error rate and SNR deweighting,
               a novel method that trains communication systems for optimal
               performance over a range of signal-to-noise ratios. The utility
               of the proposed loss functions as well as of SNR deweighting is
               shown through simulations in NVIDIA Sionna.

GNNs for Channel Decoding
*************************
.. made-with-sionna::
    :title: Graph Neural Networks for Channel Decoding
    :authors: Sebastian Cammerer, Jakob Hoydis, Fayçal Aït Aoudia, Alexander Keller
    :year: 2022
    :version: 0.11
    :link_arxiv: https://arxiv.org/pdf/2207.14742.pdf
    :link_github: https://github.com/NVlabs/gnn-decoder
    :link_colab: https://colab.research.google.com/github/NVlabs/gnn-decoder/blob/master/GNN_decoder_standalone.ipynb
    :abstract: We propose a fully differentiable graph neural network (GNN)-based architecture for channel decoding and showcase competitive decoding performance for various coding schemes, such as low-density parity-check (LDPC) and BCH codes. The idea is to let a neural network (NN) learn a generalized message passing algorithm over a given graph that represents the forward error correction code structure by replacing node and edge message updates with trainable functions.

DL-based Synchronization of NB-IoT
**********************************
.. made-with-sionna::
    :title: Deep Learning-Based Synchronization for Uplink NB-IoT
    :authors: Fayçal Aït Aoudia, Jakob Hoydis, Sebastian Cammerer, Matthijs Van Keirsbilck, Alexander Keller
    :year: 2022
    :version: 0.11
    :link_arxiv: https://arxiv.org/pdf/2205.10805.pdf
    :link_github: https://github.com/NVlabs/nprach_synch
    :abstract: We propose a neural network (NN)-based algorithm for device detection and time of arrival (ToA) and carrier frequency offset (CFO) estimation for the narrowband physical random-access channel (NPRACH) of narrowband internet of things (NB-IoT). The introduced NN architecture leverages residual convolutional networks as well as knowledge of the preamble structure of the 5G New Radio (5G NR) specifications.


