==================
"Made with Sionna"
==================

We love to see how Sionna is used by other researchers! For this reason, you find below links to papers whose authors have also published Sionna-based simulation code.

List of Projects
----------------

If you want your paper and code be listed here, please send an email to `sionna@nvidia.com <mailto:sionna@nvidia.com>`_ with links to the paper (e.g., `arXiv <https://arxiv.org>`_) and code repository (e.g., `GitHub <https://github.com>`_).

DUIDD: Deep-Unfolded Interleaved Detection and Decoding for MIMO Wireless Systems
*********************************************************************************
.. made-with-sionna::
    :title: Bit Error and Block Error Rate Training for ML-Assisted Communication
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


