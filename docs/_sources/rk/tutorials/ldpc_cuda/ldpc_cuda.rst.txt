.. _accelerated_ldpc_part2:

Part 2: CUDA Implementation
===========================

.. _fig_acceleration:
.. figure:: ../../figs/inline_vs_lookaside.png
   :align: center
   :width: 500px
   :alt: Inline vs Lookaside Acceleration

   Inline and lookaside in the gNB DU processing pipeline. The figure is from [Kundu2023B]_. Note that the DGX Spark and Jetson AGX Thor platform use a unified memory architecture, which can be seen as a hybrid of the inline and lookaside memory management.

The second part of this tutorial focuses on the implementation of the LDPC decoder using CUDA and explains the common pitfalls when offloading compute-intensive functions to the GPU. Further, we show how GPU acceleration offloading of the LDPC decoding can be integrated into the OAI stack.

Function offloading to dedicated accelerators requires careful consideration of the data flow with respect to the underlying memory architecture. This is especially critical for wireless communications applications, where strict latency requirements in the receiver processing pipeline necessitate efficient memory management when utilizing dedicated signal processing accelerators.

As shown in :numref:`fig_acceleration`, one needs to distinguish between *inline* and *lookaside* acceleration. While lookaside acceleration moves data between the CPU and the hardware accelerator, inline acceleration avoids such data movement by applying the entire processing pipeline on the hardware accelerator.

Strictly speaking, DGX Spark still moves data between the CPU and GPU. However, the overhead is significantly reduced compared to traditional split-memory architectures as DGX Spark shares the same physical memory between CPU and GPU. This has implications for the caching behavior and requires a careful implementation of the CUDA kernels to avoid performance degradation. Nevertheless, we will consider it as inline acceleration for the following discussion.

For further details on CUDA, we refer to the `NVIDIA CUDA Programming Guide <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html>`_ [CUDA2024]_ and for the Jetson platform to the `CUDA for Tegra Memory Model <https://docs.nvidia.com/cuda/cuda-for-tegra-appnote/index.html>`_ [Tegra2024]_.

Overview
--------

.. _fig_ldpc_kernel:
.. figure:: ../../figs/tutorial_ldpc_kernel.png
   :align: center
   :width: 500px
   :alt: LDPC Acceleration Overview

   Overview of CUDA implementation of the LDPC BP decoding algorithm.

The CUDA implementation can be found in `plugins/ldpc_cuda/src/runtime/ldpc_decoder.cu <https://github.com/NVlabs/sionna-rk/blob/main/plugins/ldpc_cuda/src/runtime/ldpc_decoder.cu>`_. The core decoding algorithm is implemented in the *update_cn_kernel(.)* and *update_vn_kernel(.)* functions. Both kernels are iteratively called and perform the check node (CN) and variable node (VN) updates, respectively. The decoder stops when the maximum number of iterations is reached. An additional early stopping condition could also be applied to reduce the average number of iterations.

The *pack_bits_kernel(.)* kernel maps the soft-values to hard-decided bits and packs them into a more compact byte-representation which is required for the OAI processing pipeline.

CUDA Integration in OAI
-----------------------

The LDPC decoder is implemented as `shared library <https://github.com/OPENAIRINTERFACE/openairinterface5g/blob/develop/openair1/PHY/CODING/DOC/LDPCImplementation.md>`_ that can be loaded using the `OAI shared library loader <https://gitlab.eurecom.fr/oai/openairinterface5g/blob/develop/common/utils/DOC/loader.md>`_.
The implementation is in ``plugins/ldpc_cuda/src/runtime/ldpc_decoder.cu``. After modifying it, rebuild the Docker images:

.. code-block:: bash

   ./scripts/build-oai-images.sh


Running the Decoder
-------------------

The CUDA-based decoder can be used as a drop-in replacement for the existing decoder implementations. It can be loaded when running the gNB via the following ``GNB_EXTRA_OPTIONS`` in the ``.env`` file of the config folder.

.. code-block:: bash

   GNB_EXTRA_OPTIONS=--loader.ldpc.shlibversion _cuda

We strongly recommend to additionally assign dedicated CPU cores to PHY-layer processing via the *thread-pool* option. This assigns the cores 5-9 to the PHY layer thread pool. Note that the lower CPU cores are assigned to handle the USRP-related tasks such as time synchronization.

You can now start the gNB with the CUDA-based decoder by running

.. code-block:: bash

   scripts/start_system.sh rfsim # replace with b200 or other config folder

The GPU load can be monitored via

.. code-block:: bash

   nvidia-smi  # or jtop on Jetson devices

Congratulations! You have now successfully accelerated the LDPC decoder using CUDA.

Check the gNB logs for the CUDA decoder:

.. code-block:: bash

   docker logs oai-gnb

You should see the CUDA decoder being used (requires iperf3 traffic to trigger the decoder):

.. code-block:: bash

   > ...
   > [NR_PHY] I {L1_rx_thread} CUDA LDPC decoder:    123.55 us (    96.54 us / seg)
   > ...

Implementation Aspects
----------------------

The following sections focus on various technical aspects of the CUDA implementation and the performance implications of different memory transfer patterns.

For debugging and profiling, please refer to the tutorial on :ref:`debugging`.

Memory Management
^^^^^^^^^^^^^^^^^

In order to offload compute-intensive processing to the GPU, data needs to be shared between the host (CPU) and the device (GPU). We can leverage the shared system memory architecture of DGX Spark and Jetson platforms to avoid the bottleneck of costly memory transfers on traditional split-memory platforms.

In fact, we can avoid the overhead of any complex API calls and memory transition operations by allocating page-locked memory on the host using ``cudaHostAlloc``. To make input LLRs visible to the GPU, we then use a simple ``memcpy()`` operation to copy inputs from the 5G stack over into such a page-locked buffer, where unified memory addressing allows direct reads and writes both on the host and the device. For output bits, we first synchronize the parallel CUDA command stream and then simply use ``memcpy()`` to copy packed bits from the shared page-locked buffer into the 5G stack output buffer.

This simplicity is enabled by the unified memory architecture. On DGX Spark (Grace Blackwell), hardware cache coherency via NVLink-C2C ensures efficient data sharing. On Jetson platforms, the cache semantics [Tegra2024]_ require host caches to be active while device memory caching is disabled on page-locked buffers. For optimal performance on both platforms, we ensure maximally coalesced read-once and write-once memory access patterns (addressing consecutive memory in consecutive threads).

Traditional memory transfer patterns designed for split-memory architectures can also be used on DGX Spark, but can lead to higher overheads depending on the exact architecture and generation. Explicit memory transfer calls such as ``cudaMemcpyAsync(inputs..., cudaMemcpyHostToDevice, stream)`` and ``cudaMemcpyAsync(outputs..., cudaMemcpyDeviceToHost, stream)`` incur additional API overheads and may depend on availability of additional copy engines; explicit transitioning of managed pageable memory allocated by ``cudaMallocManaged()`` by ``cudaStreamAttachMemAsync()`` may also require excessive API overhead for small buffer sizes. Therefore, we instead use the patterns described above to optimize for shared system memory and low latency, particularly since input and output buffers are typically small in the real-time 5G stack.

.. note::

   On the DGX Spark architecture, ``cudaMallocManaged()`` can be even more efficient for some use cases. For an example implementation, see the :ref:`neural_receiver` tutorial.

For comparison, we show both variants side-by-side in the following code, where the latency-optimized code path is the one with ``USE_UNIFIED_MEMORY`` defined.
We copy the input LLRs from the host to the device memory in the *ldpc_decoder.cu* file:

.. literalinclude:: ldpc_decoder.cu
   :language: cpp
   :linenos:
   :start-after: START marker-copy-input

.. code-block:: cpp
   :linenos:

       // copy input LLRs from the host to the device memory
       cudaMemcpyAsync(context.llr_in_buffer, p_in, memorySize_llr_in, cudaMemcpyHostToDevice, context.stream);


After decoding, we make the output bits available via a host-side copy in parallel to an asynchronous syndrome check on the device:

.. code-block:: cpp
   :linenos:

       // pack LDPC output bits on the device
       int pack_thread_blocks = (block_length + 127) / 128;
       pack_decoded_bit_kernel<<<pack_thread_blocks, 128, 0, context.stream>>>(context.dev_llr_out, context.dev_bits_out, block_length);

      // allow CPU access of output bits while computing syndrome
   #ifdef USE_UNIFIED_MEMORY
       cudaStreamSynchronize(context.stream);

       // ... schedule syndrome or CRC check ...

       // while syndrome computations are running on the device, copy output bits to 5G stack output buffer
       memcpy(p_out, context.dev_bits_out, memorySize_bits_out);
   #else
       cudaCheck(cudaMemcpyAsync(p_out, context.dev_bits_out, memorySize_bits_out, cudaMemcpyDeviceToHost, context.stream));

       // ... schedule syndrome or CRC check ...

       // allow CPU access of output bits and syndrome
       cudaStreamSynchronize(context.stream);
   #endif

Kernel Optimization
^^^^^^^^^^^^^^^^^^^

The core idea of CUDA programming is to parallelize the execution of the kernel on the GPU. The kernel is executed by a grid of blocks, each containing a number of independent threads. This level of parallelism allows for significant speedups compared to the serial execution on the CPU.

For a detailed introduction to CUDA programming, we refer to the `CUDA Programming Guide <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html>`_.
Conceptually, a CUDA kernel is defined as follows

.. code-block:: C

   // Kernel definition: __global__ indicates that the function is a CUDA kernel
   __global__ void my_kernel(float *data, int N) {
      // Each thread processes one element of the array

      // We calculate the global thread index from the block and thread indices
      // idx is unique for each thread
      int idx = threadIdx.x + blockIdx.x * blockDim.x;
      if (idx < N) {
         // process the idx-th element of the array
         data[idx] = ...;
      }
   }

This kernel can now be launched by specifying its grid and block dimensions via

.. code-block:: C

   // Launch the kernel
   // <<<1, 32>>> specifies the grid and block sizes
   my_kernel<<<1, 32>>>(d_data, N);

For the case of LDPC decoding, the decoder can be parallelized over the number of variable (VN) and check node (CN) updates, respectively.
An overview of the CUDA implementation is given in :numref:`fig_ldpc_kernel`. The VN update kernel is given as

.. literalinclude:: ldpc_decoder.cu
   :language: cpp
   :linenos:
   :start-after: START marker-vnp-kernel
   :end-before: END marker-vnp-kernel

As the OAI processing pipeline uses multiple threads, we need to ensure proper multi-threading support in the CUDA kernel.
This is done via a thread specific context that is passed to the kernel.
This ensures that each thread operates on its own CUDA stream and, thus, can be executed in parallel without interference.

.. literalinclude:: ldpc_decoder.cu
   :language: cpp
   :linenos:
   :start-after: START marker-thread-context
   :end-before: END marker-thread-context


Further, the decoder uses clipping values for the extrinsic messages and the VN accumulator.

.. literalinclude:: ldpc_decoder.cu
   :language: cpp
   :linenos:
   :start-at: add the channel LLRs
   :end-at: llr_accumulator_t(msg_sum);

.. literalinclude:: ldpc_decoder.cu
   :language: cpp
   :linenos:
   :start-at: clip msg magnitudes to MAX_LLR_VALUE
   :end-before: END marker-vnp-clipping

Following the same principles, the CN update kernel is given as

.. literalinclude:: ldpc_decoder.cu
   :language: cpp
   :linenos:
   :start-after: START marker-cnp-kernel
   :end-before: END marker-cnp-kernel

Note that the choice of the clipping values is critical for the performance of the decoder, in particular as the decoder uses mostly *signed char* variables. This keeps the memory footprint low and can be configured via the following macros

.. literalinclude:: ldpc_decoder.cu
   :language: cpp
   :linenos:
   :start-after: START marker-dtypes
   :end-before: END marker-dtypes

For a given lifting factor `Z` and a number of rows `num_rows` and columns `num_cols` given from the BG selection, the following grid and block dimensions are used

.. code-block:: cpp
   :linenos:

   dim3 threads(256);
   // check node update
   dim3 blocks_cn(blocks_for(bg.num_rows * Z, threads.x));
   // VN update
   dim3 blocks_vn(blocks_for(bg.num_cols * Z, threads.x));

.. literalinclude:: ldpc_decoder.cu
   :language: cpp
   :linenos:
   :start-after: START marker-grid-blocks-util
   :end-before: END marker-grid-blocks-util

Note that the grid and block dimensions are architecture dependent and should be tuned for the specific GPU.

After decoding, the output is hard-decided and packed into a byte-array.

.. literalinclude:: ldpc_decoder.cu
   :language: cpp
   :linenos:
   :start-after: START marker-pack-bits
   :end-before: END marker-pack-bits


.. note::

   The decoder returns the number of iterations and declares a decoding failure by returning *max_num_iter* if the CRC or the syndrome check fails.


Testing
-------

Unit tests verify the CUDA decoder against Sionna's reference implementation using `nanobind <https://nanobind.readthedocs.io/>`_ for Python-CUDA binding. The tests cover multiple code rates for both base graphs (BG1 and BG2).

.. code-block:: bash

   cd plugins/ldpc_cuda
   pytest tests/unit/

The integration test runs the full 5G stack with the CUDA decoder and verifies end-to-end operation via iperf3 traffic.

.. code-block:: bash

   cd plugins/ldpc_cuda/tests/integration
   ./run_integration.sh

See ``plugins/ldpc_cuda/tests/`` for the full test suite.


Outlook - Weighted Belief Propagation
-------------------------------------

An extension of *classical* belief propagation is the *weighted* belief propagation [Nachmani2016]_ using gradient descent to optimize trainable parameters during message passing. An example implementation can be found in the Sionna tutorial on `Weighted Belief Propagation <https://nvlabs.github.io/sionna/examples/Weighted_BP_Algorithm.html>`_.


When looking at the current implementation of the CN update, we can already see a damping factor of 3/4 applied to each outgoing CN message [Pretti2005]_.

.. literalinclude:: ldpc_decoder.cu
   :language: cpp
   :linenos:
   :start-after: START marker-damping-factor
   :end-before: END marker-damping-factor

.. literalinclude:: ldpc_decoder.cu
   :language: cpp
   :linenos:
   :start-after: START marker-cnp-damping
   :end-before: END marker-cnp-damping

One could now implement a weighted CN update by simply replacing the damping factor with a trainable parameter from the weighted BP tutorial.

We hope you enjoyed this tutorial. You are now well equipped to accelerate other compute intensive parts of the 5G stack as well by following the same principles.
