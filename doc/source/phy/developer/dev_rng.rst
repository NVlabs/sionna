.. _dev_rng:

Random number generation
========================

In order to make your simulations reproducible, it is important to configure a
random seed which makes your code deterministic. When Sionna is loaded, it
instantiates random number generators (RNGs) for `Python
<https://docs.python.org/3/library/random.html#alternative-generator>`_,
`NumPy <https://numpy.org/doc/stable/reference/random/generator.html>`_, and
`TensorFlow <https://www.tensorflow.org/api_docs/python/tf/random/Generator>`_. You
can then set a single seed which will make all of your
results deterministic, as long as only these RNGs are used. In the cell below,
you can see how this seed is set and how the different RNGs can be used in your
code. All of Sionna PHY's built-in functions realy on these RNGs.

.. code-block:: python

    from sionna.phy import config
    config.seed = 40

    # Python RNG - use instead of
    # import random
    # random.randint(0, 10)
    print(config.py_rng.randint(0,10))

    # NumPy RNG - use instead of
    # import numpy as np
    # np.random.randint(0, 10)
    print(config.np_rng.integers(0,10))

    # TensorFlow RNG - use instead of
    # import tensorflow as tf
    # tf.random.uniform(shape=[1], minval=0, maxval=10, dtype=tf.int32)
    print(config.tf_rng.uniform(shape=[1], minval=0, maxval=10, dtype=tf.int32))

.. code-block:: console

    7
    5
    tf.Tensor([2], shape=(1,), dtype=int32)