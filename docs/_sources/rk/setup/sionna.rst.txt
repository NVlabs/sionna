.. _sionna:

Installing Sionna
=================

Running TensorFlow with GPU support on the NVIDIA Jetson platform requires to install a specific version of TensorFlow.

.. code-block:: bash

    # you can install the TF GPU package via
    python3 -m pip install --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v60 tensorflow==2.15.0+nv24.05

    # verify that TF uses the GPU
    import tensorflow as tf
    print(tf.__version__)
    print(tf.config.list_physical_devices('GPU'))

You should now see the GPU in the TensorFlow output:

.. code-block:: bash

    [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]

Next, we install Sionna without ray tracing:

.. code-block:: bash

    # Install Sionna without ray tracing dependencies
    pip install sionna-no-rt

If you need ray tracing, you need to build and install `Mitsuba <https://www.mitsuba-renderer.org/>`_  separately. Unfortunately, Mitsuba is not yet available as pip package for aarch64.

Additional packages are required for the tutorials:

.. code-block:: bash

    # Install additional packages
    pip install -r requirements.txt

