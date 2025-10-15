<!--
SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->
# Sionna: An Open-Source Library for Research on Communication Systems

Sionna&trade; is an open-source Python-based library for research on
communication systems.

The official documentation can be found
[here](https://nvlabs.github.io/sionna/).

It is composed of the following packages:

- [Sionna RT](https://nvlabs.github.io/sionna/rt/index.html) -
    A lightning-fast stand-alone ray tracer for radio propagation modeling

- [Sionna PHY](https://nvlabs.github.io/sionna/phy/index.html) -
    A link-level simulator for wireless and optical communication systems

- [Sionna SYS](https://nvlabs.github.io/sionna/sys/index.html) -
    A system-level simulator based on physical-layer abstraction

# Installation
Sionna PHY and Sionna SYS require [Python 3.10-3.12](https://www.python.org/) and [TensorFlow 2.14-2.19](https://www.tensorflow.org/install). We recommend Ubuntu 24.04. Earlier versions of TensorFlow may still work but are not recommended. We refer to the [TensorFlow GPU support tutorial](https://www.tensorflow.org/install/gpu) for GPU support and the required driver setup.

Sionna RT has the same requirements as [Mitsuba
3](https://github.com/mitsuba-renderer/mitsuba3) and we refer to its
[installation guide](https://mitsuba.readthedocs.io/en/stable/) for further
information. To run Sionna RT on CPU, [LLVM](https://llvm.org) is required by
[Dr.Jit](https://drjit.readthedocs.io/en/stable/). Please check the
[installation instructions for the LLVM
backend](https://drjit.readthedocs.io/en/latest/what.html#backends). The source
code of Sionna RT is located in a separate [GitHub repository](https://github.com/NVlabs/sionna-rt).

If you want to run the tutorial notebooks on your machine, you also need
[JupyterLab](https://jupyter.org/). You can alternatively test them on [Google
Colab](https://colab.research.google.com/). Although not necessary, we recommend
running Sionna in a [Docker container](https://www.docker.com) and/or [Python virtual
enviroment](https://docs.python.org/3/library/venv.html).

## Installation via pip
The recommended way to install Sionna is via pip:
```
pip install sionna
```

If you want to install only Sionna RT, run:
```
pip install sionna-rt
```

You can install Sionna without the RT package via
```
pip install sionna-no-rt
```

## Installation from source
1. Clone the repository with all submodules:
    ```
    git clone --recursive https://github.com/NVlabs/sionna
    ```
    If you have already cloned the repository but forgot to set the `--recursive`
    flag, you can correct this via:
    ```
    git submodule update --init --recursive --remote
    ```
2. Install Sionna (including Sionna RT) by running the following command from within the repository's
   root folder:
    ```
    pip install ext/sionna-rt/ .
    pip install .
    ```

## Testing
First, you need to install the test requirements by executing the
following command from the repository's root directory:

```
pip install '.[test]'
```

The unit tests can then be executed by running ``pytest`` from within the
``test`` folder.

## Building the Documentation
Install the requirements for building the documentation by running the following
command from the repository's root directory:

```
pip install '.[doc]'
```

You might need to install [pandoc](https://pandoc.org) manually.

You can then build the documentation by executing ``make html`` from within the ``doc`` folder.

The documentation can then be served by any web server, e.g.,

```
python -m http.server --dir build/html
```

## For Developers

Development requirements can be installed by executing from the repository's root directory:

```
pip install '.[dev]'
```

Linting of the code can be achieved by running ```pylint src/``` from the repository's root directory.

## License and Citation

Sionna is Apache-2.0 licensed, as found in the [LICENSE](https://github.com/nvlabs/sionna/blob/main/LICENSE) file.

If you use this software, please cite it as:
```bibtex
@software{sionna,
 title = {Sionna},
 author = {Hoydis, Jakob and Cammerer, Sebastian and {Ait Aoudia}, Fayçal and Nimier-David, Merlin and Maggi, Lorenzo and Marcus, Guillermo and Vem, Avinash and Keller, Alexander},
 note = {https://nvlabs.github.io/sionna/},
 year = {2022},
 version = {1.2.1}
}
```
