.. _sionna:

Sionna Installation
===================

Sionna is only required for development purposes. It is not required to run the OpenAirInterface stack.

.. note::
   TensorFlow installation varies by platform. See the platform-specific sections below.


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

On DGX Spark, install TensorFlow and all requirements:

.. code-block:: bash

   pip install tensorflow
   pip install -r requirements.txt


Jetson Thor
-----------

On Jetson Thor, install TensorFlow and all requirements:

.. code-block:: bash

   pip install tensorflow
   pip install -r requirements_thor.txt


Jetson AGX Orin & Orin Nano
---------------------------

On Jetson Orin platforms, TensorFlow requires NVIDIA's pre-built wheels:

.. code-block:: bash

   python3 -m pip install --user --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v60 tensorflow==2.16.1+nv24.07

   pip install -r requirements_orin.txt


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

Verify that TensorFlow detects the GPU:

.. code-block:: python

   import tensorflow as tf
   print(tf.__version__)
   print(tf.config.list_physical_devices('GPU'))

Expected output:

.. code-block:: text

   [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
