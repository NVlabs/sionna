==================
"Made with Sionna"
==================

We love to see how Sionna is used by other researchers! For this reason, you find below links to papers whose authors have also published Sionna-based simulation code.

List of Projects
----------------

If you want your paper and code be listed here, please send an email to `sionna@nvidia.com <mailto:sionna@nvidia.com>`_ with links to the paper (e.g., `arXiv <https://arxiv.org>`_) and code repository (e.g., `GitHub <https://github.com>`_).


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
