
#
# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0#
"""Definition of Sionna PHY Object and Block classes"""

from abc import ABC
from abc import abstractmethod
import tensorflow as tf
import numpy as np
from .config import config, dtypes

class Object(ABC):
    """Abstract class for Sionna PHY objects

    Parameters
    ----------
    precision : `None` (default) | "single" | "double"
        Precision used for internal calculations.
        If set to `None`, the default
        :attr:`~sionna.phy.config.Config.precision` is used.
    """

    # pylint: disable=unused-argument
    def __init__(self, *args, precision=None, **kwargs):
        if precision is None:
            self._precision = config.precision
        elif precision in ['single', 'double']:
            self._precision = precision
        else:
            raise ValueError("'precision' must be 'single' or 'double'")

    @property
    def precision(self):
        """
        `str`, "single" | "double" : Precision used for all compuations
        """
        return self._precision

    @property
    def cdtype(self):
        """
        `tf.complex` : Type for complex floating point numbers
        """
        return dtypes[self.precision]['tf']['cdtype']

    @property
    def rdtype(self):
        """
        `tf.float` : Type for real floating point numbers
        """
        return dtypes[self.precision]['tf']['rdtype']

    def _cast_or_check_precision(self, v):
        """Cast tensor to internal precision or check
           if a variable has the right precision
        """
        # Check correct dtype for Variables
        if isinstance(v, tf.Variable):
            if v.dtype.is_complex:
                if v.dtype != self.cdtype:
                    msg = f"Wrong dtype. Expected {self.cdtype}" + \
                          f", got {v.dtype}"
                    raise ValueError(msg)
            elif v.dtype.is_floating:
                if v.dtype != self.rdtype:
                    msg = f"Wrong dtype. Expected {self.cdtype}" + \
                          f", got {v.dtype}"
                    raise ValueError("Wrong dtype")

        # Cast tensors to the correct dtype
        else:
            if not isinstance(v, tf.Tensor):
                v = tf.convert_to_tensor(v)
            if v.dtype.is_complex:
                v = tf.cast(v, self.cdtype)
            else:
                v = tf.cast(v, self.rdtype)

        return v

class Block(Object):
    """Abstract class for Sionna PHY processing blocks

    Parameters
    ----------
    precision : `None` (default) | "single" | "double"
        Precision used for internal calculations and outputs.
        If set to `None`, the default
        :attr:`~sionna.phy.config.Config.precision` is used.
    """
    # pylint: disable=unused-argument
    def __init__(self, *args, precision=None, **kwargs):
        super().__init__(precision=precision, **kwargs)

        # Boolean flag indicating if the block's build function has been called
        # This will prevent rebuilding the block in Eager mode each time it is
        # called.
        self._built = False

    @property
    def built(self):
        """
        `bool` : Indicates if the blocks' build function was called
        """
        return self._built

    def build(self, *arg_shapes, **kwarg_shapes):
        """
        Method to (optionally) initialize the block based on the inputs' shapes
        """
        pass

    @abstractmethod
    def call(self, *args, **kwargs):
        """
        Abstract call method with arbitrary arguments and keyword
        arguments
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def _convert_to_tensor(self, v):
        """Casts floating or complex tensors to the block's precision"""
        if isinstance(v, np.ndarray):
            v = tf.convert_to_tensor(v)
        if isinstance(v, tf.Tensor):
            if v.dtype.is_floating:
                v = tf.cast(v, self.rdtype)
            elif v.dtype.is_complex:
                v = tf.cast(v, self.cdtype)
        return v

    def _get_shape(self, v):
        """Converts an input to the corresponding TensorShape"""
        try :
            v = tf.convert_to_tensor(v)
        except (TypeError, ValueError):
            pass
        if hasattr(v, "shape"):
            return tf.TensorShape(v.shape)
        else:
            return tf.TensorShape([])

    def __call__(self, *args, **kwargs):

        args, kwargs = tf.nest.map_structure(self._convert_to_tensor,
                                             [args, kwargs])
        with tf.init_scope(): # pylint: disable=not-context-manager
            if not self._built:
                shapes =  tf.nest.map_structure(self._get_shape,
                                             [args, kwargs])
                self.build(*shapes[0], **shapes[1])
                self._built = True

        return self.call(*args, **kwargs)
