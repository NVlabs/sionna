#
# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0#
"""Global Sionna PHY Configuration"""

import random
import numpy as np
import tensorflow as tf

# Mapping from precision to dtypes
dtypes = {
    'single' : {
        'tf' : {
            'cdtype' : tf.complex64,
            'rdtype' : tf.float32
        },
        'np' : {
            'cdtype' : np.complex64,
            'rdtype' : np.float32
        }
    },
    'double' : {
        'tf' : {
            'cdtype' : tf.complex128,
            'rdtype' : tf.float64
        },
        'np' : {
            'cdtype' : np.complex128,
            'rdtype' : np.float64
        }
    }
}

class Config():
    """Sionna PHY Configuration Class

    This singleton class is used to define global configuration variables
    and random number generators that can be accessed from all modules
    and functions. It is instantiated immediately and its properties can be
    accessed as :code:`sionna.phy.config.desired_property`.
    """

    # This object is a singleton
    _instance = None
    def __new__(cls):
        if cls._instance is None:
            instance = object.__new__(cls)
            cls._instance = instance
        return cls._instance

    def __init__(self):
        self._seed = None
        self._py_rng = None
        self._np_rng = None
        self._tf_rng = None
        self._precision = None

        # Set default properties
        self.precision = 'single'

    @property
    def py_rng(self):
        """
        `random.Random` : Python random number generator

        .. code-block:: python

            from sionna.phy import config
            config.seed = 42 # Set seed for deterministic results

            # Use generator instead of random
            int = config.py_rng.randint(0, 10)
        """
        if self._py_rng is None:
            self._py_rng = random.Random()
        return self._py_rng

    @property
    def np_rng(self):
        """
        `np.random.Generator` : NumPy random number generator

        .. code-block:: python

            from sionna.phy import config
            config.seed = 42 # Set seed for deterministic results

            # Use generator instead of np.random
            noise = config.np_rng.normal(size=[4])
        """
        if self._np_rng is None:
            self._np_rng = np.random.default_rng()
        return self._np_rng

    @property
    def tf_rng(self):
        """
        `tf.random.Generator` : TensorFlow random number generator

        .. code-block:: python

            from sionna.phy import config
            config.seed = 42 # Set seed for deterministic results

            # Use generator instead of tf.random
            noise = config.tf_rng.normal([4])
        """
        if self._tf_rng  is None:
            self._tf_rng = tf.random.Generator.from_non_deterministic_state()
        return self._tf_rng

    @property
    def seed(self):
        # pylint: disable=line-too-long
        """
        `None` (default) | `int` : Get/set seed for all random number generators

        All random number generators used internally by Sionna
        can be configured with a common seed to ensure reproducability
        of results. It defaults to `None` which implies that a random
        seed will be used and results are non-deterministic.

        .. code-block:: python

            # This code will lead to deterministic results
            from sionna.phy import config
            from sionna.phy.mapping import BinarySource
            config.seed = 42
            print(BinarySource()([10]))

        .. code-block:: console

            tf.Tensor([0. 1. 1. 1. 1. 0. 1. 0. 1. 0.], shape=(10,), dtype=float32)
        """
        return self._seed

    @seed.setter
    def seed(self, seed):
        # Store seed
        if seed is not None:
            seed = int(seed)
        self._seed = seed

        #TensorFlow
        self.tf_rng.reset_from_seed(seed)

        # Python
        self.py_rng.seed(seed)

        # NumPy
        self._np_rng = np.random.default_rng(seed)

    @property
    def precision(self):
        # pylint: disable=line-too-long
        """
        "single" (default) | "double" : Default precision used for all computations

        The "single" option represents real-valued floating-point numbers
        using 32 bits, whereas the "double" option uses 64 bits.
        For complex-valued data types, each component of the complex number
        (real and imaginary parts) uses either 32 bits (for "single")
        or 64 bits (for "double").
        """
        return self._precision

    @precision.setter
    def precision(self, v):
        if v not in ["single", "double"]:
            raise ValueError("Precision must be ``single`` or ``double``.")
        self._precision = v

    @property
    def np_rdtype(self):
        """
        `np.dtype` : Default NumPy dtype for real floating point numbers
        """
        return dtypes[self.precision]['np']['rdtype']

    @property
    def np_cdtype(self):
        """
        `np.dtype` : Default NumPy dtype for complex floating point numbers
        """
        return dtypes[self.precision]['np']['cdtype']

    @property
    def tf_rdtype(self):
        """
        `tf.dtype` : Default TensorFlow dtype for real floating point numbers
        """
        return dtypes[self.precision]['tf']['rdtype']

    @property
    def tf_cdtype(self):
        """
        `tf.dtype` : Default TensorFlow dtype for complex floating point numbers
        """
        return dtypes[self.precision]['tf']['cdtype']

config = Config()
