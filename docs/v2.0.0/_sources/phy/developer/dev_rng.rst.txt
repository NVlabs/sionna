.. _dev_rng:

Random number generation
========================

In order to make your simulations reproducible, it is important to configure a
random :attr:`~sionna.phy.config.Config.seed` which makes your code deterministic. When Sionna is loaded, the
:data:`~sionna.phy.config.Config` singleton instantiates random number generators (RNGs) for `Python
<https://docs.python.org/3/library/random.html#alternative-generator>`_,
`NumPy <https://numpy.org/doc/stable/reference/random/generator.html>`_, and
`PyTorch <https://pytorch.org/docs/stable/generated/torch.Generator.html>`_. You
can then set a single seed which will make all of your
results deterministic, as long as only these RNGs are used. In the cell below,
you can see how :attr:`~sionna.phy.config.Config.seed` is set and how
:attr:`~sionna.phy.config.Config.py_rng`,
:attr:`~sionna.phy.config.Config.np_rng`, and :meth:`~sionna.phy.config.Config.torch_rng` can be used in your
code. All of Sionna PHY's built-in functions rely on these RNGs.

.. code-block:: python

    import torch
    from sionna.phy import config
    config.seed = 40

    # Python RNG - use instead of
    # import random
    # random.randint(0, 10)
    print(config.py_rng.randint(0, 10))

    # NumPy RNG - use instead of
    # import numpy as np
    # np.random.randint(0, 10)
    print(config.np_rng.integers(0, 10))

    # PyTorch RNG - use instead of
    # torch.randint(0, 10, (1,))
    print(torch.randint(0, 10, (1,), generator=config.torch_rng(), device=config.device))

.. code-block:: console

    7
    5
    tensor([7])

For code that uses :torch:`torch.compile`, use the compile-aware utilities in the
:doc:`utility functions <../api/utils/index>` section of the PHY API (e.g. :func:`~sionna.phy.utils.randint`,
:func:`~sionna.phy.utils.normal`) and pass ``generator=config.torch_rng()``
in eager mode; they automatically switch to the global RNG when compiled.
