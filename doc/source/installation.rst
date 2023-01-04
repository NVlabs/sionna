Installation
############

Sionna requires `Python <https://www.python.org/>`_ and `Tensorflow <https://www.tensorflow.org/>`_.
In order to run the tutorial notebooks on your machine, you also need `Jupyter <https://jupyter.org/>`_.
You can alternatively test them on `Google Colab <https://colab.research.google.com/github/nvlabs/sionna/blob/main/examples/Discover_Sionna.ipynb>`_.
Although not necessary, we recommend running Sionna in a `Docker container <https://www.docker.com>`_.

.. note::
    Sionna requires `TensorFlow 2.7-2.11 <https://www.tensorflow.org/install>`_ and Python 3.6-3.9.
    We recommend Ubuntu 20.04.

    We refer to the `TensorFlow GPU support tutorial <https://www.tensorflow.org/install/gpu>`_ for GPU support and the required driver setup.

Installation using pip
----------------------
We recommend to do this within a `virtual environment <https://docs.python.org/3/tutorial/venv.html>`_,
e.g., using `conda <https://docs.conda.io>`_. On macOS, you need to install `tensorflow-macos <https://github.com/apple/tensorflow_macos>`_ first.




1.) Install the package

.. code-block:: bash

    pip install sionna


2.) Test the installation in Python

.. code-block:: bash

    python

.. code-block:: python

    >>> import sionna
    >>> print(sionna.__version__)
    0.12.1

3.) Once Sionna is installed, you can run the `Sionna "Hello, World!" example <https://nvlabs.github.io/sionna/examples/Hello_World.html>`_, have a look at the `quick start guide <https://nvlabs.github.io/sionna/quickstart.html>`_, or at the `tutorials <https://nvlabs.github.io/sionna/tutorials.html>`_.

For a local installation, the `JupyterLab Desktop <https://github.com/jupyterlab/jupyterlab-desktop>`_ application can be used. This directly includes the Python installation and configuration.


Docker-based Installation
-------------------------

1.) Make sure that you have Docker `installed <https://docs.docker.com/engine/install/ubuntu/>`_ on your system. On Ubuntu 20.04, you can run for example

.. code-block:: bash

    sudo apt install docker.io

Ensure that your user belongs to the `docker` group (see `Docker post-installation <https://docs.docker.com/engine/install/linux-postinstall/>`_).

.. code-block:: bash

    sudo usermod -aG docker $USER

Log out and re-login to load updated group memberships.

For GPU support on Linux, you need to install the `NVIDIA Container Toolkit <https://github.com/NVIDIA/nvidia-docker>`_.

2.) Build the Sionna Docker image. From within the Sionna directory, run:

.. code-block:: bash

    make docker

3.) Run the Docker image with GPU support

.. code-block:: bash

    make run-docker gpus=all

or without GPU:

.. code-block:: bash

    make run-docker

This will immediately launch a Docker image with Sionna installed, running Jupyter on port 8888.

4.) Browse through the example notebook by connecting to `http://127.0.0.1:8888 <http://127.0.0.1:8888>`_ in your browser.


Installation from source
------------------------

We recommend to do this within a `virtual environment <https://docs.python.org/3/tutorial/venv.html>`_,
e.g., using `conda <https://docs.conda.io>`_.

1.) Clone this repository and execute from within its root folder:

.. code-block:: bash

    make install


2.) Test the installation in Python

.. code-block:: bash

    python

.. code-block:: python

    >>> import sionna
    >>> print(sionna.__version__)
    0.12.1
