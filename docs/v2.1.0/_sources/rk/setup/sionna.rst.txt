.. _sionna:

Sionna Installation
===================

Sionna is only required for development purposes. It is not required to run the OpenAirInterface stack.

.. note::
   Sionna 2.0 requires PyTorch. See the platform-specific sections below.


Virtual Environment
-------------------

We recommend using a virtual environment to isolate Python dependencies:

.. code-block:: bash

   python3 -m venv ~/.venv/sionna-rk
   source ~/.venv/sionna-rk/bin/activate

To activate the environment automatically, add to your ``~/.profile``:

.. code-block:: bash

   echo 'source ~/.venv/sionna-rk/bin/activate' >> ~/.profile


DGX Spark
---------

On DGX Spark, install all requirements (including PyTorch and Sionna):

.. code-block:: bash

   pip install -r requirements.txt


Jetson Thor
-----------

On Jetson Thor, install all requirements:

.. code-block:: bash

   pip install -r requirements_thor.txt


Jetson AGX Orin & Orin Nano
---------------------------

On Jetson Orin platforms, install all requirements:

.. code-block:: bash

   pip install -r requirements_orin.txt

.. note::
   For Jetson platforms, NVIDIA provides pre-built PyTorch wheels via the
   `Jetson PyTorch containers <https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform-release-notes/pytorch-jetson-rel.html>`_.
   Refer to the NVIDIA documentation for the latest compatible PyTorch version for your JetPack release.


TensorRT Python Bindings
------------------------

To access system TensorRT bindings in the virtual environment:

On AGX Orin:

.. code-block:: bash

   echo 'export PYTHONPATH=$PYTHONPATH:/usr/lib/python3.10/dist-packages' >> ~/.profile
   source ~/.profile

On AGX Thor:

.. code-block:: bash

   echo 'export PYTHONPATH=$PYTHONPATH:/usr/lib/python3.12/dist-packages' >> ~/.profile
   echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.profile
   source ~/.profile

Verification
------------

Verify that PyTorch detects the GPU:

.. code-block:: python

   import torch
   print(torch.__version__)
   print(torch.cuda.is_available())
   print(torch.cuda.get_device_name(0))

Expected output (version string, CUDA availability, device name):

.. code-block:: text

   2.x.x
   True
   NVIDIA <device name>
