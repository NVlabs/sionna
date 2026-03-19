.. _accelerated_inference:

Part 2: GPU-Accelerated inference
=================================

.. figure:: ../../figs/tutorial_nn_demapper_overview.png
   :align: center
   :width: 600px
   :alt: Neural Demapper Overview

We now discuss how to integrate the TensorRT engine for inference into the OAI stack. To keep the inference latency as low as possible [Gadiyar2023]_, [Kundu2023]_, we use CUDA graphs [Gray2019]_ to launch the TensorRT inference engine.

You will learn about:

* How to accelerate the neural demapper using TensorRT
* How to pre- and post-process input and output data using CUDA
* How to use CUDA graphs for latency reductions

For details on efficient memory management when offloading compute-intensive functions to the GPU, we refer to the :ref:`accelerated_ldpc` tutorial.


Demapper Implementation Overview
--------------------------------

The neural demapper is implemented in Tensorflow and exported to TensorRT, the source code of the inference logic can be found in `plugins/neural_demapper/src/runtime/trt_demapper.cpp <https://github.com/NVlabs/sionna-rk/blob/main/plugins/neural_demapper/src/runtime/trt_demapper.cpp>`_. The implementation will be explained in the following sections.

The TRT demapper receives noisy input symbols from the OpenAirInterface stack via the function ``trt_demapper_decode()``, which chunks a given array of symbols into batches of maximum size ``MAX_BLOCK_LEN`` and then calls ``trt_demapper_decode_block()`` to carry out the actual inference on each batch. To leverage data-parallel execution on the GPU, inference is performed for batches of symbols and multiple threads in parallel. The output of the neural demapper is passed back in the form of ``num_bits_per_symbol`` LLRs per input symbol.

To run the TensorRT inference engine on the given ``int16_t``-quantized data, we dequantize input symbols to half-precision floating-point format on the GPU using a data-parallel CUDA kernel (see ``norm_int16_symbols_to_float16()``), and re-quantize output LLRs using another CUDA kernel (see ``float16_llrs_to_int16()``).

.. note::

   The full implementation can be found in `plugins/neural_demapper/src/runtime/data_processing.cu <https://github.com/NVlabs/sionna-rk/blob/main/plugins/neural_demapper/src/runtime/data_processing.cu>`_ and `plugins/neural_demapper/src/runtime/trt_demapper.cpp <https://github.com/NVlabs/sionna-rk/blob/main/plugins/neural_demapper/src/runtime/trt_demapper.cpp>`_.

Setting up the TensorRT Inference Engine
----------------------------------------

To be compatible with the multi-threaded OpenAirInterface implementation, we load the neural demapper network into a TensorRT ``ICudaEngine`` once and share the inference engine between multiple ``IExecutionContext`` objects which are created per worker thread. We store the global and per-thread state, respectively, as follows:

.. code-block:: cpp
   :linenos:

   #include "NvInfer.h"
   #include <cuda_fp16.h>

   // global
   static IRuntime* runtime = nullptr;
   static ICudaEngine* engine = nullptr;

   // per thread
   struct TRTContext {
       cudaStream_t default_stream = 0;     // asynchronous CUDA command stream
       IExecutionContext* trt = nullptr;    // TensorRT execution context
       void* prealloc_memory = nullptr;     // memory block for temporary per-inference data
       __half* input_buffer = nullptr;      // device-side network inputs after CUDA pre-processing
       __half* output_buffer = nullptr;     // device-side network output before CUDA pre-processing

       int16_t* symbol_buffer = nullptr;    // write-through buffer for symbols written by CPU and read by GPU
       int16_t* magnitude_buffer = nullptr; // write-through buffer for magnitude estimates written by CPU and read by GPU
       int16_t* llr_buffer = nullptr;       // host-cached buffer for llr estimates written by GPU and read by CPU

       // list of thread contexts for shutdown
       TRTContext* next_initialized_context = nullptr;
   };
   static __thread TRTContext thread_context = { };


We call the following global initialization routine on program startup:

.. code-block:: cpp
   :linenos:

   static char const* trt_weight_file = "models/neural_demapper_qam16_2.plan"; // Training result / trtexec output
   static bool trt_normalized_inputs = true;

   extern "C" TRTContext* trt_demapper_init() {
       if (runtime)  // lazy, global
           return &trt_demapper_init_context();

       printf("Initializing TRT runtime\n");
       runtime = createInferRuntime(logger);
       printf("Loading TRT engine %s (normalized inputs: %d)\n", trt_weight_file, trt_normalized_inputs);
       std::vector<char> modelData = readModelFromFile(trt_weight_file);
       engine = runtime->deserializeCudaEngine(modelData.data(), modelData.size());

       return &trt_demapper_init_context();
   }

   // Utilities

   std::vector<char> readModelFromFile(char const* filepath) {
       std::vector<char> bytes;
       FILE* f = fopen(filepath, "rb");
       if (!f) {
           logger.log(Logger::Severity::kERROR, filepath);
           return bytes;
       }
       fseek(f, 0, SEEK_END);
       bytes.resize((size_t) ftell(f));
       fseek(f, 0, SEEK_SET);
       if (bytes.size() != fread(bytes.data(), 1, bytes.size(), f))
           logger.log(Logger::Severity::kWARNING, filepath);
       fclose(f);
       return bytes;
   }

   struct Logger : public ILogger
   {
       void log(Severity severity, const char* msg) noexcept override
       {
           // suppress info-level messages
           if (severity <= Severity::kWARNING)
               printf("TRT %s: %s\n", severity == Severity::kWARNING ? "WARNING" : "ERROR", msg);
       }
   };
   static Logger logger;

On startup of each worker thread, we initialize the per-thread contexts as follows:

.. code-block:: cpp
   :linenos:

   TRTContext& trt_demapper_init_context() {
       auto& context = thread_context;
       if (context.trt) // lazy
           return context;

       printf("Initializing TRT context (TID %d)\n", (int) gettid());

       // create execution context with its own pre-allocated temporary memory attached
       context.trt = engine->createExecutionContextWithoutDeviceMemory();
       size_t preallocSize = engine->getDeviceMemorySize();
       CHECK_CUDA(cudaMalloc(&context.prealloc_memory, preallocSize));
       context.trt->setDeviceMemory(context.prealloc_memory);

       // create own asynchronous CUDA command stream for this thread
       CHECK_CUDA(cudaStreamCreateWithFlags(&context.default_stream, cudaStreamNonBlocking));

       // allocate neural network input and output buffers (device access memory)
       cudaMalloc((void**) &context.input_buffer, sizeof(*context.input_buffer) * 4 * MAX_BLOCK_LEN);
       cudaMalloc((void**) &context.output_buffer, sizeof(*context.output_buffer) * MAX_BITS_PER_SYMBOL * MAX_BLOCK_LEN);

       // OAI decoder input buffers that can be written and read with unified addressing from CPU and GPU, respectively
       // note: GPU reads are uncached, but read-once coalesced
       cudaHostAlloc((void**) &context.symbol_buffer, sizeof(*context.symbol_buffer) * 2 * MAX_BLOCK_LEN, cudaHostAllocMapped | cudaHostAllocWriteCombined);
       cudaHostAlloc((void**) &context.magnitude_buffer, sizeof(*context.magnitude_buffer) * 2 * MAX_BLOCK_LEN, cudaHostAllocMapped | cudaHostAllocWriteCombined);
       // OAI decoder output buffers that can be written and read with unified addressing from GPU and CPU, respectively
       // note: GPU writes are uncached, but write-once coalesced
       cudaHostAlloc((void**) &context.llr_buffer, sizeof(*context.llr_buffer) * MAX_BITS_PER_SYMBOL * MAX_BLOCK_LEN, cudaHostAllocMapped);

       // keep track of active thread contexts for shutdown
       TRTContext* self = &context;
       __atomic_exchange(&initialized_thread_contexts, &self, &self->next_initialized_context, __ATOMIC_ACQ_REL);

       return context;
   }


Running Batched Inference
-------------------------

If decoder symbols are already available in half-precision floating-point format, running the TensorRT inference engine is as simple as performing one call to enqueue the corresponding inference commands on the asynchronous CUDA command stream of the calling thread's context:

.. code-block:: cpp
   :linenos:

   void trt_demapper_run(TRTContext* context, cudaStream_t stream, __half const* inputs, size_t numInputs, size_t numInputComponents, __half* outputs) {
       if (stream == 0)
           stream = context->default_stream;

       context.trt->setTensorAddress("y", (void*) inputs);
       context.trt->setInputShape("y", Dims2(numInputs, numInputComponents));
       context.trt->setTensorAddress("output_1", outputs);
       context.trt->enqueueV3(stream);
   }


Converting Data Types between Host and Device
---------------------------------------------

In the OAI 5G stack, received symbols come in from the host side in quantized ``int16_t`` format, together with a channel magnitude estimate.
In order to convert inputs to half-precision floating-point format, we first copy the symbols to a pinned memory buffer ``mapped_symbols`` that resides in unified addressable memory, and then run a CUDA kernel for dequantization and normalization on the GPU.
After inference, the conversion back to quantized LLRs follows the same pattern, first a CUDA kernel quantizes the half-precision floating-point inference outputs, then the quantized data written by the GPU is read by the CPU using the unified addressable memory buffer ``mapped_outputs``. Note that the CUDA command stream runs asynchronously, therefore it needs to be synchronized with the calling thread before accessing the output data.

.. code-block:: cpp
   :linenos:

   extern "C" void trt_demapper_decode_block(TRTContext* context_, cudaStream_t stream, int16_t const* in_symbols, int16_t const* in_mags, size_t num_symbols,
                                             __half const *mapped_symbols, __half const *mapped_mags, size_t num_batch_symbols,
                                             int16_t* outputs, uint32_t num_bits_per_symbol, __half* mapped_outputs) {
       auto& context = *context_;

       memcpy((void*) mapped_symbols, in_symbols, sizeof(*in_symbols) * 2 * num_symbols);
       memcpy((void*) mapped_mags, in_mags, sizeof(*in_mags) * 2 * num_symbols);

       size_t num_in_components;
       if (trt_normalized_inputs) {
           norm_int16_symbols_to_float16(stream, mapped_symbols, mapped_mags, num_batch_symbols,
                                         (uint16_t*) context.input_buffer, 1);
           num_in_components = 2;
       }
       else {
           [...]
           num_in_components = 4;
       }

       trt_demapper_run(&context, stream, recording ? nullptr : context.input_buffer, block_size, num_in_components, recording ? nullptr : context.output_buffer);

       float16_llrs_to_int16(stream, (uint16_t const*) context.output_buffer, num_batch_symbols,
                             mapped_outputs, num_bits_per_symbol);

       CHECK_CUDA(cudaStreamSynchronize(stream));
       memcpy(outputs, mapped_outputs, sizeof(*outputs) * num_bits_per_symbol * num_symbols);
   }

The CUDA kernel for normalization runs in a straight-forward 1D CUDA grid, reading the tuples of ``int16_t``-quantized components that make up each complex value in a coalesced (consecutive) way, as one ``int32_t`` value each. Then, the symbol values are normalized with respect to the magnitude values and again written in a coalesced way, fusing each complex symbol into one ``__half2`` value:

.. literalinclude:: ../../../../plugins/neural_demapper/src/runtime/data_processing.cu
   :language: cpp
   :linenos:
   :start-after: START marker-normalize-symbols
   :end-before: END marker-normalize-symbols

.. code-block:: cpp
   :linenos:

   inline __host__ __device__ int blocks_for(uint32_t elements, int block_size) {
       return int( uint32_t(elements + (block_size-1)) / uint32_t(block_size) );
   }

The CUDA kernel for re-quantization of output LLRs works analogously, converting half-precision floating-point LLR tuples to quantized ``int16_t`` values by fixed-point scaling and rounding:

.. literalinclude:: ../../../../plugins/neural_demapper/src/runtime/data_processing.cu
   :language: cpp
   :linenos:
   :start-after: START marker-quantize-llrs
   :end-before: END marker-quantize-llrs


Demapper Integration in OAI
---------------------------

.. note::

   Ensure that you have built the TRTengine in the first part of the tutorial.


In order to mount the TensorRT models and config files, you need to extend the ``config/common/docker-compose.override.yaml`` file:

.. code-block:: yaml
   :linenos:

   services:
      oai-gnb:
          volumes:
          - ../../plugins/neural_demapper/models/:/opt/oai-gnb/models
          - ../../plugins/neural_demapper/config/demapper_trt.config:/opt/oai-gnb/demapper_trt.config

Pre-trained models are available in `plugins/neural_demapper/models <https://github.com/NVlabs/sionna-rk/blob/main/plugins/neural_demapper/models>`_.

The TRT config file format has the following schema:

.. code-block::

   <trt_engine_file:string> # file name of the TensorRT engine
   <trt_normalized_inputs:int> # flag to indicate if the inputs are normalized

For example, the following config file will use the TensorRT engine `models/neural_demapper.2xfloat16.plan <https://github.com/NVlabs/sionna-rk/blob/main/plugins/neural_demapper/models/neural_demapper.2xfloat16.onnx>`_ and normalize the inputs:

.. code-block::

   model/neural_demapper.2xfloat16.plan
   1


Running the Demapper
--------------------

The neural demapper is implemented as shared library (see :ref:`data_acquisition`) which can be loaded using the OAI shared library loader. The demapper can now be used as a drop-in replacement for the QAM-16 default implementation. The demapper can be loaded when running the gNB via the following ``GNB_EXTRA_OPTIONS`` in the ``config/<config_name>/.env`` file of the config folder.

.. code-block:: bash

   GNB_EXTRA_OPTIONS=--loader.demapper.shlibversion _trt --MACRLCs.[0].dl_max_mcs 10 --MACRLCs.[0].ul_max_mcs 10

We limit the MCS indices to 10 in order to stay within the QAM-16 modulation order.

Congratulations! You have now successfully implemented demapping using a neural network.

You can track the GPU load via

.. code-block:: bash

   # on DGX Spark
   $ nvidia-smi

   # on Jetson
   $ jtop


Implementation Aspects
----------------------

In the following section, we focus on various technical aspects of the CUDA implementation and the performance implications of different memory transfer patterns and command scheduling optimization.


Memory Management
^^^^^^^^^^^^^^^^^

Similar to the :ref:`accelerated_ldpc` tutorial, we use the shared system memory architecture to avoid the bottleneck of costly memory transfers on traditional split-memory platforms.

As previously covered in the :ref:`accelerated_ldpc` tutorial, optimizing memory operations is essential for real-time performance. For the neural demapper implementation, we use the same efficient approach of page-locked memory (via `cudaHostAlloc()`) to enable direct GPU-CPU memory sharing. This allows for simple `memcpy()` operations instead of complex memory management calls, with host caching enabled for CPU access while device caching is disabled for direct memory access. This approach is particularly well-suited for the small buffer sizes used in neural demapping, avoiding the overhead of traditional GPU memory management methods like `cudaMemcpyAsync()` or `cudaMallocManaged()`.

For comparison, we show both variants side-by-side in the following inference code, where the latency-optimized code path is the one with ``USE_UNIFIED_MEMORY`` defined:

.. code-block:: cpp
   :linenos:

   extern "C" void trt_demapper_decode_block(TRTContext* context_, cudaStream_t stream, int16_t const* in_symbols, int16_t const* in_mags, size_t num_symbols,
                                             __half const *mapped_symbols, __half const *mapped_mags, size_t num_batch_symbols,
                                             int16_t* outputs, uint32_t num_bits_per_symbol, __half* mapped_outputs) {
       auto& context = *context_;

   #if defined(USE_UNIFIED_MEMORY)
       memcpy((void*) mapped_symbols, in_symbols, sizeof(*in_symbols) * 2 * num_symbols);
       memcpy((void*) mapped_mags, in_mags, sizeof(*in_mags) * 2 * num_symbols);
   #else
       cudaMemcpyAsync((void*) mapped_symbols, in_symbols, sizeof(*in_symbols) * 2 * num_symbols, cudaMemcpyHostToDevice, stream);
       cudaMemcpyAsync((void*) mapped_mags, in_mags, sizeof(*in_mags) * 2 * num_symbols, cudaMemcpyHostToDevice, stream);
   #endif

       size_t num_in_components;
       if (trt_normalized_inputs) {
           norm_int16_symbols_to_float16(stream, mapped_symbols, mapped_mags, num_batch_symbols,
                                         (uint16_t*) context.input_buffer, 1);
           num_in_components = 2;
       }
       else {
           [...]
           num_in_components = 4;
       }

       trt_demapper_run(&context, stream, recording ? nullptr : context.input_buffer, block_size, num_in_components, recording ? nullptr : context.output_buffer);

       float16_llrs_to_int16(stream, (uint16_t const*) context.output_buffer, num_batch_symbols,
                             mapped_outputs, num_bits_per_symbol);

   #if defined(USE_UNIFIED_MEMORY)
       // note: synchronize the asynchronous command queue before accessing from the host
       CHECK_CUDA(cudaStreamSynchronize(stream));
       memcpy(outputs, mapped_outputs, sizeof(*outputs) * num_bits_per_symbol * num_symbols);
   #else
       cudaMemcpyAsync(outputs, mapped_outputs, sizeof(*outputs) * num_bits_per_symbol * num_symbols, cudaMemcpyDeviceToHost, stream);
       // note: synchronize after the asynchronous command queue has executed the copy to host
       CHECK_CUDA(cudaStreamSynchronize(stream));
   #endif
   }


CUDA Graph Optimization
^^^^^^^^^^^^^^^^^^^^^^^

CUDA command graph APIs [Gray2019]_ were introduced to frontload the overhead of scheduling repetitive sequences of compute kernels on the GPU, allowing pre-recorded, pre-optimized command sequences, such as in our case neural network inference, to be scheduled by a single API call. Thus, latency can be reduced further, focussing runtime spending on the actual computations rather than on dynamic command scheduling. We pre-record CUDA graphs including demapper inference, data pre-processing, and post-processing, for two different batch sizes, one for common small batches and one for the maximum expected parallel batch size.

Command graphs are pre-recorded per thread due to the individual intermediate storage buffers used. We run the recording at the end of thread context initialization as introduced above, for each batch size running one *size-0* inference to trigger any kind of lazy runtime allocations, and another inference on dummy inputs for the actual recording:

.. literalinclude:: ../../../../plugins/neural_demapper/src/runtime/trt_demapper.cpp
   :language: cpp
   :linenos:
   :start-after: START marker-record-graph
   :end-before: END marker-record-graph

To extend the function ``trt_demapper_decode_block()`` with CUDA graph recording and execution, we introduce the following code paths when ``USE_GRAPHS`` is defined:

.. code-block:: cpp
   :linenos:

   extern "C" void trt_demapper_decode_block(TRTContext* context_, cudaStream_t stream, int16_t const* in_symbols, int16_t const* in_mags, size_t num_symbols,
                                             int16_t const *mapped_symbols, int16_t const *mapped_mags, size_t num_batch_symbols,
                                             int16_t* outputs, uint32_t num_bits_per_symbol, int16_t* mapped_outputs) {
       auto& context = *context_;

       uint32_t block_size = num_batch_symbols > OPT_BLOCK_LEN ? MAX_BLOCK_LEN : OPT_BLOCK_LEN;
       cudaGraph_t& graph = block_size == OPT_BLOCK_LEN ? context.graph_opt : context.graph_max;
       cudaGraphExec_t& graphCtx = block_size == OPT_BLOCK_LEN ? context.record_opt : context.record_max;

       if (num_symbols > 0) {
           memcpy((void*) mapped_symbols, in_symbols, sizeof(*in_symbols) * 2 * num_symbols);
           memcpy((void*) mapped_mags, in_mags, sizeof(*in_mags) * 2 * num_symbols);
       }

       // graph capture
       if (!graph) {
           bool recording = false;
   #ifdef USE_GRAPHS
           // allow pre-allocation before recording
           if (num_symbols > 0) {
               // in pre-recording phase
               CHECK_CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed));
               num_batch_symbols = block_size;
               recording = true;
           }
           // else: pre-allocation phase
   #endif

           size_t num_in_components;
           if (trt_normalized_inputs) {
               norm_int16_symbols_to_float16(stream, mapped_symbols, mapped_mags, num_batch_symbols,
                                             (uint16_t*) context.input_buffer, 1);
               num_in_components = 2;
           }
           else {
               int16_symbols_to_float16(stream, mapped_symbols, num_batch_symbols,
                                        (uint16_t*) context.input_buffer, 2);
               int16_symbols_to_float16(stream, mapped_mags, num_batch_symbols,
                                        (uint16_t*) context.input_buffer + 2, 2);
               num_in_components = 4;
           }

           trt_demapper_run(&context, stream, recording ? nullptr : context.input_buffer, block_size, num_in_components, recording ? nullptr : context.output_buffer);

           float16_llrs_to_int16(stream, (uint16_t const*) context.output_buffer, num_batch_symbols,
                                 mapped_outputs, num_bits_per_symbol);

   #ifdef USE_GRAPHS
           if (num_symbols > 0) {
               // in pre-recording phase
               CHECK_CUDA(cudaStreamEndCapture(stream, &graph));
               printf("Recorded CUDA graph (TID %d), stream %llX\n", (int) gettid(), (unsigned long long) stream);
           }
   #endif
       }

   #ifdef USE_GRAPHS
       if (graph && !graphCtx) {
           // in pre-recording phase
           CHECK_CUDA(cudaGraphInstantiate(&graphCtx, graph, 0));
       }
       else if (num_symbols > 0) {
           // in runtime inference, run pre-recorded graph
           cudaGraphLaunch(graphCtx, stream);
       }
   #endif

       CHECK_CUDA(cudaStreamSynchronize(stream));
       memcpy(outputs, mapped_outputs, sizeof(*outputs) * num_bits_per_symbol * num_symbols);
   }


.. note::

    Note that CUDA graphs are currently only enabled on DGX Spark. Jetson platform support is disabled pending investigation of stability issues.

Unit tests
----------

We have implemented unit tests using `pytest` to allow testing individual parts of the implementation outside of the full 5G stack.
The unit tests can be found in `plugins/neural_demapper/tests/ <https://github.com/NVlabs/sionna-rk/blob/main/plugins/neural_demapper/tests/>`_.
The unit tests use `nanobind` to call the TensorRT and CUDA modules from Python and to test against Python-based reference implementations. For more details on how to use nanobind, please refer to the `nanobind documentation <https://nanobind.readthedocs.io/en/latest/>`_.


Outlook
-------

This was a first tutorial on accelerating neural network inference using TensorRT and CUDA graphs. The neural demapper itself is a simple network and the focus was on the integration rather than the actual error rate performance.

You are now able to deploy your own neural networks using this tutorial as a blueprint. An interesting starting point could be the :ref:`neural_receiver` tutorial, which provides a 5G compliant implementation of a neural receiver and already provides a TensorRT export of the trained network.
