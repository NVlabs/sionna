.. _dev_sionna_block:

Sionna Block and Object
=======================

All of Sionna PHY's components inherit from the Sionna `Object
<https://nvlabs.github.io/sionna/phy/api/developers.html#sionna.phy.Object>`_
class.

A Sionna ``Object`` is instantiated with an optional ``precision`` argument from which it
derives complex- and real-valued data types which can be accessed via the
properties ``cdtype`` and ``rdtype``, respectively:

.. code-block:: python

    from sionna.phy import Object
    obj = Object(precision="single")
    print(obj.cdtype)
    print(obj.rdtype)

.. code-block:: python

    <dtype: 'complex64'>
    <dtype: 'float32'>

If the ``precision`` argument is not provided, ``Objects`` use the
global ``config.precision`` parameter, as shown next:

.. code-block:: python

    from sionna.phy import config
    from sionna.phy import Object
    config.precision = "double" # Set global precision
    obj = Object()
    print(obj.cdtype)
    print(obj.rdtype)

.. code-block:: python

    <dtype: 'complex128'>
    <dtype: 'float64'>

Understanding Sionna Blocks
---------------------------

Sionna ``Blocks`` inherit from ``Objects`` and are used to implement most of Sionna's components.
To get an understanding of their features, let us implement a simple custom
``Block``. Every ``Block`` must implement the method ``call`` which can take arbitray
arguments and keyword arguments. It is important to understand that all tensor arguments
are cast to the ``Block``'s internal ``precision``. The
following code snippet demonstrates this behavior:

.. code-block:: python

    import tensorflow as tf
    from sionna.phy import config
    from sionna.phy import Block
    config.precision = "double"

    class MyBlock(Block):
        def call(self, x, y=None):
            print(x.dtype)
            if y is not None:
                print(y.dtype)

    my_block = MyBlock()
    x = tf.constant([3], dtype=tf.float32)
    y = tf.complex(2., 3.)
    my_block(x, y)

.. code-block:: console

    <dtype: 'float64'>
    <dtype: 'complex128'>

As the internal precision of all ``Blocks`` was set via the global ``precision``
flag to double preceision,
the inputs ``x`` and ``y`` were cast to the corresponding dtypes prior to executing
the Block's ``call`` method. Note that only floating data types are cast, as can be
seen from the following example:

.. code-block:: python

    class MyBlock(Block):
        def call(self, x):
            print(type(x))

    my_block = MyBlock()
    my_block(3)

.. code-block:: console

    <class 'int'>

The reason for this behavior is that we sometimes need to pass non-tensor
arguments to a function so that algorithms can be unrolled during the creation
of the computation graph.

In many cases, a ``Block`` require some initialization that requires the shapes of
its inputs. The first time a ``Block`` is called, it executes the ``build`` method
which provides the shapes of all arguments and keyword arguments. The next
example demonstrates this feature:

.. code-block:: python

    import numpy as np
    import tensorflow as tf
    from sionna.phy import Block

    class MyBlock(Block):
        def build(self, *args, **kwargs):
            self.x_shape = args[0]
            self.y_shape = kwargs["y"]

        def call(self, x, y=None):
            print(self.x_shape)
            print(x.dtype)
            print(self.y_shape)

    my_block = MyBlock()
    my_block(np.array([3, 3]), y=tf.zeros([10, 12]))

.. code-block:: console

    (2,)
    <dtype: 'float64'>
    (10, 12)

Note that the argument ``x`` was provided as NumPy array which is
converted to a TensorFlow tensor within the ``Block``. This is in contrast to the
example above, where this did not happen for an integer input. For a detailed
understanding of type conversions within ``Blocks``, we refer to the source code of
the class method ``Block._convert_to_tensor``.