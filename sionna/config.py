#
# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Global Sionna configuration"""

class Config():
    """The Sionna configuration class.

    This class is used to define global configuration variables
    that can be accessed from all modules and functions. It
    is instantiated in ``sionna.__init__()`` and its properties can be
    accessed as ``sionna.config.desired_property``.
    """
    def __init__(self):
        self.xla_compat = False

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
