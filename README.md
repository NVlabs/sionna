<!--
SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->
# Sionna: An Open-Source Library for Next-Generation Physical Layer Research

Sionna&trade; is an open-source Python library for link-level simulations of digital communication systems built on top of the open-source software library [TensorFlow](https://www.tensorflow.org) for machine learning.

The official documentation can be found [here](https://nvlabs.github.io/sionna/).

## Installation
Sionna requires [Python](https://www.python.org/) and [Tensorflow](https://www.tensorflow.org/).
In order to run the tutorial notebooks on your machine, you also need [Jupyter](https://jupyter.org/).
You can alternatively test them on [Google Colab](https://colab.research.google.com/).
Although not necessary, we recommend running Sionna in a [Docker container](https://www.docker.com).

Sionna requires [TensorFlow 2.7-2.11](https://www.tensorflow.org/install) and Python 3.6-3.9. We recommend Ubuntu 20.04. TensorFlow 2.6 still works but is not recommended because of known, unpatched CVEs.

We refer to the [TensorFlow GPU support tutorial](https://www.tensorflow.org/install/gpu) for GPU support and the required driver setup.

### Installation using pip

We recommend to do this within a [virtual environment](https://docs.python.org/3/tutorial/venv.html), e.g., using [conda](https://docs.conda.io).
On macOS, you need to install [tensorflow-macos](https://github.com/apple/tensorflow_macos) first.

1.) Install the package
```
    pip install sionna
```

2.) Test the installation in Python
```
    python
```
```
    >>> import sionna
    >>> print(sionna.__version__)
    0.12.1
```

3.) Once Sionna is installed, you can run the [Sionna "Hello, World!" example](https://nvlabs.github.io/sionna/examples/Hello_World.html), have a look at the [quick start guide](https://nvlabs.github.io/sionna/quickstart.html), or at the [tutorials](https://nvlabs.github.io/sionna/tutorials.html).

The example notebooks can be opened and executed with [Jupyter](https://jupyter.org/).

For a local installation, the [JupyterLab Desktop](https://github.com/jupyterlab/jupyterlab-desktop) application can be used which also includes the Python installation.

### Docker-based installation

1.) Make sure that you have [Docker](<https://docs.docker.com/engine/install/ubuntu/>) installed on your system. On Ubuntu 20.04, you can run for example

```
    sudo apt install docker.io
```

Ensure that your user belongs to the `docker` group (see [Docker post-installation](<https://docs.docker.com/engine/install/linux-postinstall/>))

```
    sudo usermod -aG docker $USER
```
Log out and re-login to load updated group memberships.

For GPU support on Linux, you need to install the [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker).


2.) Build the Sionna Docker image. From within the Sionna directory, run

```
    make docker
```

3.) Run the Docker image with GPU support

```
    make run-docker gpus=all
```
or without GPU:
```
    make run-docker
```

This will immediately launch a Docker image with Sionna installed, running Jupyter on port 8888.

4.) Browse through the example notebooks by connecting to [http://127.0.0.1:8888](http://127.0.0.1:8888) in your browser.

### Installation from source

We recommend to do this within a [virtual environment](https://docs.python.org/3/tutorial/venv.html), e.g., using [conda](https://docs.conda.io).

1.) Clone this repository and execute from within its root folder
```
    make install
```
2.) Test the installation in Python
```
    >>> import sionna
    >>> print(sionna.__version__)
    0.12.1
```

## License and Citation

Sionna is Apache-2.0 licensed, as found in the [LICENSE](https://github.com/nvlabs/sionna/blob/main/LICENSE) file.

If you use this software, please cite it as:
```bibtex
@article{sionna,
    title = {Sionna: An Open-Source Library for Next-Generation Physical Layer Research},
    author = {Hoydis, Jakob and Cammerer, Sebastian and {Ait Aoudia}, Fayçal and Vem, Avinash and Binder, Nikolaus and Marcus, Guillermo and Keller, Alexander},
    year = {2022},
    month = {Mar.},
    journal = {arXiv preprint},
    online = {https://arxiv.org/abs/2203.11854}
}
```
