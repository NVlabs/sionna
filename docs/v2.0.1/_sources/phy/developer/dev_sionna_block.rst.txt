.. _dev_sionna_block:

Sionna Block and Object
=======================

All of Sionna PHY's components inherit from the Sionna :class:`~sionna.phy.object.Object`
class.

A Sionna :class:`~sionna.phy.object.Object` is instantiated with an optional
:attr:`~sionna.phy.object.Object.precision` argument from which it
derives complex- and real-valued data types which can be accessed via the
properties :attr:`~sionna.phy.object.Object.cdtype` and :attr:`~sionna.phy.object.Object.dtype`, respectively:

.. code-block:: python

    from sionna.phy import Object
    obj = Object(precision="single")
    print(obj.cdtype)
    print(obj.dtype)

.. code-block:: console

    torch.complex64
    torch.float32

If the :attr:`~sionna.phy.object.Object.precision` argument is not provided,
:class:`~sionna.phy.object.Object` instances use the
global :attr:`~sionna.phy.config.Config.precision` parameter of the
:data:`~sionna.phy.config.config` singleton, as shown next:

.. code-block:: python

    from sionna.phy import config
    from sionna.phy import Object
    config.precision = "double"  # Set global precision
    obj = Object()
    print(obj.cdtype)
    print(obj.dtype)

.. code-block:: console

    torch.complex128
    torch.float64

Understanding Sionna Blocks
---------------------------

Sionna :class:`~sionna.phy.block.Block`\ s inherit from :class:`~sionna.phy.object.Object` and are used to implement most of Sionna's components.
To get an understanding of their features, let us implement a simple custom
:class:`~sionna.phy.block.Block`. Every :class:`~sionna.phy.block.Block` must implement the method :meth:`~sionna.phy.block.Block.call` which can take arbitrary
arguments and keyword arguments. It is important to understand that all tensor arguments
are cast to the :class:`~sionna.phy.block.Block`'s internal :attr:`~sionna.phy.object.Object.precision`. The
following code snippet demonstrates this behavior:

.. code-block:: python

    import torch
    from sionna.phy import config
    from sionna.phy import Block
    config.precision = "double"

    class MyBlock(Block):
        def call(self, x, y=None):
            print(x.dtype)
            if y is not None:
                print(y.dtype)

    my_block = MyBlock()
    x = torch.tensor([3.], dtype=torch.float32)
    y = torch.complex(torch.tensor(2.), torch.tensor(3.))
    my_block(x, y)

.. code-block:: console

    torch.float64
    torch.complex128

As the internal :attr:`~sionna.phy.object.Object.precision` of all
:class:`~sionna.phy.block.Block`\ s was set via the global
:attr:`~sionna.phy.config.Config.precision` flag to double precision,
the inputs ``x`` and ``y`` were cast to the corresponding dtypes prior to executing
the :class:`~sionna.phy.block.Block`'s :meth:`~sionna.phy.block.Block.call` method. Note that only floating data types are cast, as can be
seen from the following example:

.. code-block:: python

    from sionna.phy import Block

    class MyBlock(Block):
        def call(self, x):
            print(type(x))

    my_block = MyBlock()
    my_block(3)

.. code-block:: console

    <class 'int'>

The reason for this behavior is that we sometimes need to pass non-tensor
arguments (e.g. shapes or indices) so that control flow can be traced correctly
when using :torch:`torch.compile`.

In many cases, a :class:`~sionna.phy.block.Block` requires some initialization that depends on the shapes of
its inputs. The first time a :class:`~sionna.phy.block.Block` is called, it executes the :meth:`~sionna.phy.block.Block.build` method
which receives the shapes of all arguments and keyword arguments. The next
example demonstrates this feature:

.. code-block:: python

    import numpy as np
    import torch
    from sionna.phy import config
    from sionna.phy import Block
    config.precision = "double"

    class MyBlock(Block):
        def build(self, *args, **kwargs):
            self.x_shape = args[0]
            self.y_shape = kwargs["y"]

        def call(self, x, y=None):
            print(self.x_shape)
            print(x.dtype)
            print(self.y_shape)

    my_block = MyBlock()
    my_block(np.array([3., 3.]), y=torch.zeros([10, 12]))

.. code-block:: console

    (2,)
    torch.float64
    (10, 12)

Note that the argument ``x`` was provided as a NumPy array which is
converted to a PyTorch tensor within the :class:`~sionna.phy.block.Block`. This is in contrast to the
example above, where an integer input was left unchanged. For a detailed
understanding of type conversions within :class:`~sionna.phy.block.Block`\ s, see the method
:meth:`~sionna.phy.object.Object._convert` in the :class:`~sionna.phy.object.Object` class.
