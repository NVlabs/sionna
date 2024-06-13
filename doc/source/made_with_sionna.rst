==================
"Made with Sionna"
==================

We love to see how Sionna is used by other researchers! For this reason, you find below links to papers/projects whose authors have published Sionna-based simulation code.

If you want your paper/project and code be listed here, please send an email to `sionna@nvidia.com <mailto:sionna@nvidia.com>`_ with links to the paper (e.g., `arXiv <https://arxiv.org>`_) and code repository (e.g., `GitHub <https://github.com>`_).

.. made-with-sionna::
    :title: BostonTwin: the Boston Digital Twin for Ray-Tracing in 6G Networks
    :authors: Paolo Testolina, Michele Polese, Pedram Johari, Tommaso Melodia
    :year: March 2024
    :version: 0.16
    :link_arxiv: https://arxiv.org/abs/2403.12289
    :link_github: https://github.com/wineslab/boston_twin
    :abstract: Digital twins are now a staple of wireless networks design and evolution. Creating an accurate digital copy of a real system offers numerous opportunities to study and analyze its performance and issues. It also allows designing and testing new solutions in a risk-free environment, and applying them back to the real system after validation. A candidate technology that will heavily rely on digital twins for design and deployment is 6G, which promises robust and ubiquitous networks for eXtended Reality (XR) and immersive communications solutions. In this paper, we present BostonTwin, a dataset that merges a high-fidelity 3D model of the city of Boston, MA, with the existing geospatial data on cellular base stations deployments, in a ray-tracing-ready format. Thus, BostonTwin enables not only the instantaneous rendering and programmatic access to the building models, but it also allows for an accurate representation of the electromagnetic propagation environment in the real-world city of Boston. The level of detail and accuracy of this characterization is crucial to designing 6G networks that can support the strict requirements of sensitive and high-bandwidth applications, such as XR and immersive communication.

.. made-with-sionna::
    :title: Integrating Pre-Trained Language Model with Physical Layer Communications
    :authors: Ju-Hyung Lee, Dong-Ho Lee, Joohan Lee, Jay Pujara
    :year: February 2024
    :version: 0.16
    :link_arxiv: https://arxiv.org/abs/2402.11656
    :link_github: https://github.com/abman23/on-device-ai-comm
    :abstract: The burgeoning field of on-device AI communication, where devices exchange information directly through embedded foundation models, such as language models (LMs), requires robust, efficient, and generalizable communication frameworks. However, integrating these frameworks with existing wireless systems and effectively managing noise and bit errors pose significant challenges. In this work, we introduce a practical on-device AI communication framework, integrated with physical layer (PHY) communication functions, demonstrated through its performance on a link-level simulator. Our framework incorporates end-to-end training with channel noise to enhance resilience, incorporates vector quantized variational autoencoders (VQ-VAE) for efficient and robust communication, and utilizes pre-trained encoder-decoder transformers for improved generalization capabilities. Simulations, across various communication scenarios, reveal that our framework achieves a 50% reduction in transmission size while demonstrating substantial generalization ability and noise robustness under standardized 3GPP channel models.

.. made-with-sionna::
    :title: OpenStreetMap to Sionna Scene in Python
    :authors: Manoj Kumar Joshi
    :year: January 2024
    :version: 0.15
    :link_github: https://github.com/manoj-kumar-joshi/sionna_osm_scene
    :abstract: This Jupyter notebook shows how to create a Sionna scene (Mitsuba format) in Python code from OpenStreetMap data. Buildings are extruded and meshes for roads are created in a region specified by the user. It is an alternative to the Blender-based workflow presented <a href="https://youtu.be/7xHLDxUaQ7c">in this video</a>.

.. made-with-sionna::
    :title: Learning radio environments by differentiable ray tracing
    :authors: Jakob Hoydis, Fayçal Aït Aoudia, Sebastian Cammerer, Florian Euchner, Merlin Nimier-David, Stephan ten Brink, Alexander Keller
    :year: 2023
    :version: 0.16
    :link_arxiv: https://arxiv.org/abs/2311.18558
    :link_github: https://github.com/NVlabs/diff-rt-calibration
    :abstract: Ray tracing (RT) is instrumental in 6G research in order to generate spatially-consistent and environment-specific channel impulse responses(CIRs). While acquiring accurate scene geometries is now relatively straightforward, determining material characteristics requires precise calibration using channel measurements. We therefore introduce a novel gradient-based calibration method, complemented by differentiable parametrizations of material properties, scattering and antenna patterns. Our method seamlessly integrates with differentiable ray tracers that enable the computation of derivatives of CIRs with respect to these parameters. Essentially, we approach field computation as a large computational graph wherein parameters are trainable akin to weights of a neural network (NN). We have validated our method using both synthetic data and real-world indoor channel measurements, employing a distributed multiple-input multiple-output (MIMO) channel sounder.

.. made-with-sionna::
    :title: A Scalable and Generalizable Pathloss Map Prediction
    :authors: Ju-Hyung Lee, Andreas F. Molisch
    :year: December 2023
    :version: 0.16
    :link_arxiv: https://arxiv.org/abs/2312.03950
    :link_github: https://github.com/abman23/pmnet-sionna-rt
    :abstract: Large-scale channel prediction, i.e., estimation of the pathloss from geographical/morphological/building maps, is an essential component of wireless network planning. Ray tracing (RT)-based methods have been widely used for many years, but they require significant computational effort that may become prohibitive with the increased network densification and/or use of higher frequencies in B5G/6G systems. In this paper, we propose a data-driven, model-free pathloss map prediction (PMP) method, called PMNet. PMNet uses a supervised learning approach: it is trained on a limited amount of RT (or channel measurement) data and map data. Once trained, PMNet can predict pathloss over location with high accuracy (an RMSE level of 10−2) in a few milliseconds. We further extend PMNet by employing transfer learning (TL). TL allows PMNet to learn a new network scenario quickly (x5.6 faster training) and efficiently (using x4.5 less data) by transferring knowledge from a pre-trained model, while retaining accuracy. Our results demonstrate that PMNet is a scalable and generalizable ML-based PMP method, showing its potential to be used in several network optimization applications.

.. made-with-sionna::
    :title: Graph Neural Networks for Enhanced Decoding of Quantum LDPC Codes
    :authors: Anqi Gong, Sebastian Cammerer, Joseph M. Renes
    :year: 2023
    :version: 0.15
    :link_arxiv: https://arxiv.org/abs/2310.17758
    :link_github: https://github.com/gongaa/Feedback-GNN
    :abstract: In this work, we propose a fully differentiable iterative decoder for quantum low-density parity-check (LDPC) codes. The proposed algorithm is composed of classical belief propagation (BP) decoding stages and intermediate graph neural network (GNN) layers. Both component decoders are defined over the same sparse decoding graph enabling a seamless integration and scalability to large codes. The core idea is to use the GNN component between consecutive BP runs, so that the knowledge from the previous BP run, if stuck in a local minima caused by trapping sets or short cycles in the decoding graph, can be leveraged to better initialize the next BP run. By doing so, the proposed decoder can learn to compensate for sub-optimal BP decoding graphs that result from the design constraints of quantum LDPC codes. Since the entire decoder remains differentiable, gradient descent-based training is possible. We compare the error rate performance of the proposed decoder against various post-processing methods such as random perturbation, enhanced feedback, augmentation, and ordered-statistics decoding (OSD) and show that a carefully designed training process lowers the error-floor significantly. As a result, our proposed decoder outperforms the former three methods using significantly fewer post-processing attempts.

.. made-with-sionna::
    :title: Sionna RT: Differentiable Ray Tracing for Radio Propagation Modeling
    :authors: Jakob Hoydis, Fayçal Aït Aoudia, Sebastian Cammerer, Merlin Nimier-David, Nikolaus Binder, Guillermo Marcus, Alexander Keller
    :year: 2023
    :version: 0.16
    :link_arxiv: https://arxiv.org/abs/2303.11103
    :link_github: https://github.com/NVlabs/diff-rt
    :link_colab: https://colab.research.google.com/github/NVlabs/diff-rt/blob/master/Learning_Materials.ipynb
    :abstract: Sionna is a GPU-accelerated open-source library for link-level simulations based on TensorFlow. Its latest release (v0.14) integrates a differentiable ray tracer (RT) for the simulation of radio wave propagation. This unique feature allows for the computation of gradients of the channel impulse response and other related quantities with respect to many system  and environment parameters, such as material properties, antenna patterns, array geometries, as well as transmitter and receiver orientations and positions. In this paper, we outline the key components of Sionna RT and showcase example applications such as learning of radio materials and optimizing transmitter orientations by gradient descent. While classic ray tracing is a crucial tool for 6G research topics like reconfigurable intelligent surfaces, integrated sensing and communications, as well as user localization, differentiable ray tracing is a key enabler for many novel and exciting research directions, for example, digital twins.

.. made-with-sionna::
    :title: DUIDD: Deep-Unfolded Interleaved Detection and Decoding for MIMO Wireless Systems
    :authors: Reinhard Wiesmayr, Chris Dick, Jakob Hoydis, Christoph Studer
    :year: 2022
    :version: 0.11
    :link_arxiv: https://arxiv.org/abs/2212.07816
    :link_github: https://github.com/IIP-Group/DUIDD
    :abstract: Iterative detection and decoding (IDD) is known to achieve near-capacity performance in multi-antenna wireless systems. We propose deep-unfolded interleaved detection and decoding (DUIDD), a new paradigm that reduces the complexity of IDD while achieving even lower error rates. DUIDD interleaves the inner stages of the data detector and channel decoder, which expedites convergence and reduces complexity. Furthermore, DUIDD applies deep unfolding to automatically optimize algorithmic hyperparameters, soft-information exchange, message damping, and state forwarding. We demonstrate the efficacy of DUIDD using NVIDIA's Sionna link-level simulator in a 5G-near multi-user MIMO-OFDM wireless system with a novel low-complexity soft-input soft-output data detector, an optimized low-density parity-check decoder, and channel vectors from a commercial ray-tracer. Our results show that DUIDD outperforms classical IDD both in terms of block error rate and computational complexity.

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

.. made-with-sionna::
    :title: Graph Neural Networks for Channel Decoding
    :authors: Sebastian Cammerer, Jakob Hoydis, Fayçal Aït Aoudia, Alexander Keller
    :year: 2022
    :version: 0.11
    :link_arxiv: https://arxiv.org/pdf/2207.14742.pdf
    :link_github: https://github.com/NVlabs/gnn-decoder
    :link_colab: https://colab.research.google.com/github/NVlabs/gnn-decoder/blob/master/GNN_decoder_standalone.ipynb
    :abstract: We propose a fully differentiable graph neural network (GNN)-based architecture for channel decoding and showcase competitive decoding performance for various coding schemes, such as low-density parity-check (LDPC) and BCH codes. The idea is to let a neural network (NN) learn a generalized message passing algorithm over a given graph that represents the forward error correction code structure by replacing node and edge message updates with trainable functions.

.. made-with-sionna::
    :title: Deep Learning-Based Synchronization for Uplink NB-IoT
    :authors: Fayçal Aït Aoudia, Jakob Hoydis, Sebastian Cammerer, Matthijs Van Keirsbilck, Alexander Keller
    :year: 2022
    :version: 0.11
    :link_arxiv: https://arxiv.org/pdf/2205.10805.pdf
    :link_github: https://github.com/NVlabs/nprach_synch
    :abstract: We propose a neural network (NN)-based algorithm for device detection and time of arrival (ToA) and carrier frequency offset (CFO) estimation for the narrowband physical random-access channel (NPRACH) of narrowband internet of things (NB-IoT). The introduced NN architecture leverages residual convolutional networks as well as knowledge of the preamble structure of the 5G New Radio (5G NR) specifications.
