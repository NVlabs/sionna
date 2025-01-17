#
# SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Global Sionna configuration"""
import random
import numpy as np
import tensorflow as tf

class Config():
    """Sionna configuration class

    This singleton class is used to define global configuration variables
    and random number generators that can be accessed from all modules
    and functions. It is instantiated immediately and its properties can be
    accessed as "sionna.config.desired_property".
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
        self._xla_compat = None

        # Set default properties
        self.xla_compat = False

    @property
    def py_rng(self):
        """
        random.Random() : Python random number generator

        .. code-block:: python

            import sionna
            sionna.config.seed = 42 # Set seed for deterministic results

            # Use generator instead of random
            int = sionna.config.py_rng.randint(0, 10)
        """
        if self._py_rng is None:
            self._py_rng = random.Random()
        return self._py_rng

    @property
    def np_rng(self):
        """
        np.random.Generator : NumPy random number generator

        .. code-block:: python

            import sionna
            sionna.config.seed = 42 # Set seed for deterministic results

            # Use generator instead of np.random
            noise = sionna.config.np_rng.normal(size=[4])
        """
        if self._np_rng is None:
            self._np_rng = np.random.default_rng()
        return self._np_rng

    @property
    def tf_rng(self):
        """
        tf.random.Generator : TensorFlow random number generator

        .. code-block:: python

            import sionna
            sionna.config.seed = 42 # Set seed for deterministic results

            # Use generator instead of tf.random
            noise = sionna.config.tf_rng.normal([4])
        """
        if self._tf_rng  is None:
            self._tf_rng = tf.random.Generator.from_non_deterministic_state()
        return self._tf_rng

    @property
    def seed(self):
        # pylint: disable=line-too-long
        """Get/set seed for all random number generators

        All random number generators used internally by Sionna
        can be configured with a common seed to ensure reproducability
        of results. It defaults to `None` which implies that a random
        seed will be used and results are non-deterministic.

        .. code-block:: python

            # This code will lead to deterministic results
            import sionna
            sionna.config.seed = 42
            print(sionna.utils.BinarySource()([10]))

        .. code-block:: console

            tf.Tensor([0. 1. 1. 1. 1. 0. 1. 0. 1. 0.], shape=(10,), dtype=float32)

        :type: int
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
    def xla_compat(self):
        """Ensure that functions execute in an XLA compatible way.

        Not all TensorFlow ops support the three execution modes for
        all dtypes: Eager, Graph, and Graph with XLA. For this reason,
        some functions are implemented differently depending on the
        execution mode. As it is currently impossible to programmatically
        determine if a function is executed in Graph or Graph with XLA mode,
        the ``xla_compat`` property can be used to indicate which execution
        mode is desired. Note that most functions will work in all execution
        modes independently of the value of this property.

        This property can be used like this:

        .. code-block:: python

            import sionna
            sionna.config.xla_compat=True
            @tf.function(jit_compile=True)
            def func()
                # Implementation

            func()

        :type: bool
        """
        return self._xla_compat

    @xla_compat.setter
    def xla_compat(self, value):
        self._xla_compat = bool(value)
        if self._xla_compat:
            msg = "XLA can lead to reduced numerical precision." \
                  + " Use with care."
            print(msg)

config = Config()
