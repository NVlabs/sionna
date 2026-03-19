.. _dev_matrix_inversion:

Matrix inversion
================

Many signal processing algorithms are described using inverses and square roots
of matrices. However, their actual computation is very rarely required and one
should resort to alternatives in practical implementations instead. Below, we will
describe two representative examples.

Solving linear systems
----------------------
One frequently needs to compute equations of the form

.. math::

    \mathbf{x} = \mathbf{H}^{-1} \mathbf{y}

and would be tempted to implement this equation in the following way:

.. code-block:: python

    import torch

    # Create random example
    x_ = torch.randn(10, 1)
    h = torch.randn(10, 10)
    y = h @ x_

    # Solve via matrix inversion
    h_inv = torch.linalg.inv(h)
    x = h_inv @ y

A much more stable and efficient implementation avoids the inverse computation
and solves the following linear system instead

.. math::

    \mathbf{H}\mathbf{x} = \mathbf{y}

which looks in code like this:

.. code-block:: python

    # Solve as linear system
    x = torch.linalg.solve(h, y)

When :math:`\mathbf{H}` is a Hermitian positive-definite matrix, we can leverage
the `Cholesky
decomposition <https://en.wikipedia.org/wiki/Cholesky_decomposition>`_ :math:`\mathbf{H}=\mathbf{L}\mathbf{L}^{\mathsf{H}}`, where :math:`\mathbf{L}` is
a lower-triangular matrix, for an
even faster implementation. Throughout Sionna we use
:torch:`torch.linalg.cholesky_ex` with ``check_errors=False`` and two
:torch:`torch.linalg.solve_triangular` calls (instead of :torch:`torch.cholesky_solve`),
which avoids synchronization points and is preferred for CUDA graph compatibility:

.. code-block:: python

    # Solve via Cholesky decomposition (pattern used in Sionna)
    l, _ = torch.linalg.cholesky_ex(h, check_errors=False)
    y_temp = torch.linalg.solve_triangular(l, y, upper=False)
    x = torch.linalg.solve_triangular(l.mH, y_temp, upper=True)

For one-off scripts, the simpler :torch:`torch.linalg.cholesky` and
:torch:`torch.cholesky_solve` are also valid. Ready-to-use utilities built on
these patterns include :func:`~sionna.phy.utils.inv_cholesky` and
:func:`~sionna.phy.utils.matrix_pinv`.

Correlated random vectors
-------------------------
Assume that we need to generate a correlated random vector with a
given covariance matrix, i.e.,

.. math::

    \mathbf{x} = \mathbf{R}^{\frac12} \mathbf{w}

where :math:`\mathbf{w}\sim\mathcal{CN(\mathbf{0},\mathbf{I})}` and
:math:`\mathbf{R}` is known. One should avoid the explicit computation of the matrix
square root here and rather leverage the Cholesky decomposition for a
numerically stable and efficient implementation. We can compute :math:`\mathbf{R}=\mathbf{L}\mathbf{L}^{\mathsf{H}}` and
then generate :math:`\mathbf{x}=\mathbf{L}\mathbf{w}`, which can be
implemented as follows:

.. code-block:: python

    # Create covariance matrix
    r = torch.tensor([[1.0, 0.5, 0.25],
                      [0.5, 1.0, 0.5],
                      [0.25, 0.5, 1.0]])

    # Cholesky decomposition
    l = torch.linalg.cholesky(r)

    # Create batch of correlated random vectors
    w = torch.randn(100, 3)
    x = w @ l.T

It also happens, that one needs to whiten a correlated noise vector

.. math::

    \mathbf{w} = \mathbf{R}^{-\frac12}\mathbf{x}

where :math:`\mathbf{x}` is random with covariance matrix :math:`\mathbf{R} =
\mathbb{E}[\mathbf{x}\mathbf{x}^\mathsf{H}]`. Rather than computing
:math:`\mathbf{R}^{-\frac12}`, it is sufficient to compute
:math:`\mathbf{L}^{-1}`, which can be
achieved by solving the linear system :math:`\mathbf{L} \mathbf{X} = \mathbf{I}`,
exploiting the triangular structure of the Cholesky factor
:math:`\mathbf{L}`:

.. code-block:: python

    # Create covariance matrix
    r = torch.tensor([[1.0, 0.5, 0.25],
                      [0.5, 1.0, 0.5],
                      [0.25, 0.5, 1.0]])

    # Cholesky decomposition
    l = torch.linalg.cholesky(r)

    # Inverse of L
    l_inv = torch.linalg.solve_triangular(l, torch.eye(3), upper=False)
