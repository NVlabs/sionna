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
:func:`~sionna.phy.utils.normal`). Their output defaults to ``config.device``.
When passing an explicit generator in eager mode, pass the matching device as
well:

.. code-block:: python

    from sionna.phy.utils import normal

    noise = normal(
        [4],
        device=config.device,
        generator=config.torch_rng(config.device),
    )

The compile-aware helpers automatically switch to the seeded global RNG when
compiled because explicit generators cannot be captured in the graph.

What ``config.seed`` guarantees
-------------------------------

Setting :attr:`~sionna.phy.config.Config.seed` reinitializes Sionna's configured
Python, NumPy, and per-device PyTorch generators. Public stochastic behaviour in
**Sionna PHY** and **Sionna SYS** that goes through those generators is then
reproducible across runs that:

- use the same seed,
- use the same code path (eager vs compiled; see below),
- and do not draw from other process-global RNGs
  (``random``, ``np.random``, or unseeded ``torch.*`` calls without a
  ``generator``).

Typical examples that follow this contract include AWGN / Rayleigh channel
draws, discrete-channel bit flips, OFDM Kronecker pilot symbols,
:class:`~sionna.phy.channel.CIRDataset` shuffle order,
:class:`~sionna.sys.HexGrid` UT placement, and FEC scramblers /
interleavers that take a seed or use the configured generators.

What it does **not** guarantee
------------------------------

- **Eager vs compiled equality.** In eager mode the compile-aware helpers use
  ``config.torch_rng(device)``. Under ``torch.compile`` they fall back to the
  seeded *global* device RNG. Each mode is individually reproducible after
  ``config.seed = ...``, but the two modes are not required to produce
  bitwise-identical samples for the same seed.
- **Multi-worker ``DataLoader``s.** Sharing one ``config.py_rng`` across
  worker processes is not a supported reproducibility model. Keep
  ``num_workers=0`` (the Sionna default for :class:`~sionna.phy.channel.CIRDataset`)
  or give each worker an explicit local generator.
- **Sionna RT preview cosmetics.** Default radio-material colours and other
  preview-only randomness in the RT submodule are outside
  ``sionna.phy.config.seed``. Electromagnetic sampling in RT uses Mitsuba
  samplers with their own explicit seeds.
- **Unrelated global Torch state.** Code that calls ``torch.rand`` /
  ``torch.randn`` without passing ``generator=config.torch_rng(...)`` (or the
  compile-aware wrappers) is not controlled by ``config.seed`` in eager mode.
