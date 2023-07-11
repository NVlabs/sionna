#
# SPDX-FileCopyrightText: Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Layer for discrete channel models"""

import tensorflow as tf
from tensorflow.keras.layers import Layer
from sionna.utils import expand_to_rank

class BinaryMemorylessChannel(Layer):
    # pylint: disable=line-too-long
    r"""BinaryMemorylessChannel(return_llrs=False, bipolar_input=False, llr_max=100., dtype=tf.float32, **kwargs)

    Discrete binary memory less channel with (possibly) asymmetric bit flipping
    probabilities.

    Inputs bits are flipped with probability :math:`p_\text{b,0}` and
    :math:`p_\text{b,1}`, respectively.

    ..  figure:: ../figures/BMC_channel.png
        :align: center

    This layer supports binary inputs (:math:`x \in \{0, 1\}`) and `bipolar`
    inputs (:math:`x \in \{-1, 1\}`).

    If activated, the channel directly returns log-likelihood ratios (LLRs)
    defined as

    .. math::
        \ell =
        \begin{cases}
            \operatorname{log} \frac{p_{b,1}}{1-p_{b,0}}, \qquad \text{if} \, y=0 \\
            \operatorname{log} \frac{1-p_{b,1}}{p_{b,0}}, \qquad \text{if} \, y=1 \\
        \end{cases}

    The error probability :math:`p_\text{b}` can be either scalar or a
    tensor (broadcastable to the shape of the input). This allows
    different erasure probabilities per bit position. In any case, its last
    dimension must be of length 2 and is interpreted as :math:`p_\text{b,0}` and
    :math:`p_\text{b,1}`.

    This class inherits from the Keras `Layer` class and can be used as layer in
    a Keras model.

    Parameters
    ----------

    return_llrs: bool
        Defaults to `False`. If `True`, the layer returns log-likelihood ratios
        instead of binary values based on ``pb``.

    bipolar_input : bool, False
        Defaults to `False`. If `True`, the expected input is given as
        :math:`\{-1,1\}` instead of :math:`\{0,1\}`.

    llr_max: tf.float
        Defaults to 100. Defines the clipping value of the LLRs.

    dtype : tf.DType
        Defines the datatype for internal calculations and the output
        dtype. Defaults to `tf.float32`.

    Input
    -----
    (x, pb) :
        Tuple:

    x : [...,n], tf.float32
        Input sequence to the channel consisting of binary values :math:`\{0,1\}
        ` or :math:`\{-1,1\}`, respectively.

    pb : [...,2], tf.float32
        Error probability. Can be a tuple of two scalars or of any
        shape that can be broadcasted to the shape of ``x``. It has an
        additional last dimension which is interpreted as :math:`p_\text{b,0}`
        and :math:`p_\text{b,1}`.

    Output
    -------
        : [...,n], tf.float32
            Output sequence of same length as the input ``x``. If
            ``return_llrs`` is `False`, the output is ternary where a `-1` and
            `0` indicate an erasure for the binary and bipolar input,
            respectively.
    """

    def __init__(self, return_llrs=False, bipolar_input=False, llr_max=100.,dtype=tf.float32, **kwargs):

        super().__init__(dtype=dtype,**kwargs)

        assert isinstance(return_llrs, bool), "return_llrs must be bool."
        self._return_llrs = return_llrs

        assert isinstance(bipolar_input, bool), "bipolar_input must be bool."
        self._bipolar_input = bipolar_input

        assert llr_max>=0., "llr_max must be a positive scalar value."
        self._llr_max = tf.cast(llr_max, dtype=self.dtype)

        if self._return_llrs:
            assert dtype in (tf.float16, tf.float32, tf.float64),\
                "LLR outputs require non-integer dtypes."
        else:
            if self._bipolar_input:
                assert dtype in (tf.float16, tf.float32, tf.float64,
                    tf.int8, tf.int16, tf.int32, tf.int64),\
                    "Only, signed dtypes are supported for bipolar inputs."
            else:
                assert dtype in (tf.float16, tf.float32, tf.float64,
                    tf.uint8, tf.uint16, tf.uint32, tf.uint64,
                    tf.int8, tf.int16, tf.int32, tf.int64),\
                    "Only, real-valued dtypes are supported."

        self._check_input = True # check input for consistency (i.e., binary)

        self._eps = 1e-9 # small additional term for numerical stability
        self._temperature = tf.constant(0.1, tf.float32) # for Gumble-softmax

    #########################################
    # Public methods and properties
    #########################################

    @property
    def llr_max(self):
        """Maximum value used for LLR calculations."""
        return self._llr_max

    @llr_max.setter
    def llr_max(self, value):
        """Maximum value used for LLR calculations."""
        assert value>=0, 'llr_max cannot be negative.'
        self._llr_max = tf.cast(value, dtype=tf.float32)

    @property
    def temperature(self):
        """Temperature for Gumble-softmax trick."""
        return self._temperature

    @temperature.setter
    def temperature(self, value):
        """Temperature for Gumble-softmax trick."""
        assert value>=0, 'temperature cannot be negative.'
        self._temperature = tf.cast(value, dtype=tf.float32)

    #########################
    # Utility methods
    #########################

    def _check_inputs(self, x):
        """Check input x for consistency, i.e., verify
        that all values are binary of bipolar values."""
        x = tf.cast(x, tf.float32)
        if self._check_input:
            if self._bipolar_input: # allow -1 and 1 for bipolar inputs
                values = (tf.constant(-1, x.dtype),tf.constant(1, x.dtype))
            else: # allow 0,1 for binary input
                values = (tf.constant(0, x.dtype),tf.constant(1, x.dtype))
            tf.debugging.assert_equal(
                tf.reduce_min(tf.cast(tf.logical_or(tf.equal(x, values[0]),
                                    tf.equal(x, values[1])), x.dtype)),
                tf.constant(1, x.dtype),
                "Input must be binary.")
            # input datatype consistency should be only evaluated once
            self._check_input = False

    @tf.custom_gradient
    def _custom_xor(self, a, b):
        """Straight through estimator for XOR."""
        def grad(upstream):
            """identity in backward direction"""
            return upstream, upstream
        # xor in forward path
        # use module for "exotic" dtypes
        if self.dtype in (tf.uint8, tf.uint16, tf.uint32, tf.uint64, tf.int8, tf.int16, tf.int32, tf.int64):
            z = tf.math.mod(a+b, tf.constant(2, self.dtype))
        else: # use abs for float dtypes
            z = tf.abs(a - b)

        return z, grad

    @tf.custom_gradient
    def _ste_binarizer(self, x):
        """Straight through binarizer to quantize bits to int values."""
        def grad(upstream):
            """identity in backward direction"""
            return upstream
        # hard-decide in forward path
        z = tf.where(x<.5, 0., 1.)
        return z, grad

    def _sample_errors(self, pb, shape):
        """Samples binary error vector with given error probability e.
        This function is based on the Gumble-softmax "trick" to keep the
        sampling differentiable."""

        # this implementation follows https://arxiv.org/pdf/1611.01144v5.pdf
        # and https://arxiv.org/pdf/1906.07748.pdf

        u1 = tf.random.uniform(shape=shape,
                               minval=0.,
                               maxval=1.,
                               dtype=tf.float32)
        u2 = tf.random.uniform(shape=shape,
                               minval=0.,
                               maxval=1.,
                               dtype=tf.float32)
        u = tf.stack((u1, u2), axis=-1)

        # sample Gumble distribution
        q = - tf.math.log(- tf.math.log(u + self._eps) + self._eps)
        p = tf.stack((pb,1-pb), axis=-1)
        p = expand_to_rank(p, tf.rank(q), axis=0)
        p = tf.broadcast_to(p, q.shape)
        a = (tf.math.log(p + self._eps) + q) / self._temperature

        # apply softmax
        e_cat = tf.nn.softmax(a)

        # binarize final values via straight-through estimator
        return self._ste_binarizer(e_cat[...,0]) # only take first class

    #########################
    # Keras layer functions
    #########################

    def build(self, input_shapes):
        """Verify correct input shapes"""

        pb_shapes = input_shapes[1]
        # allow tuple of scalars as alternative input
        if isinstance(pb_shapes, (tuple, list)):
            if not len(pb_shapes)==2:
                raise ValueError("Last dim of pb must be of length 2.")
        else:
            if len(pb_shapes)>0:
                if not pb_shapes[-1]==2:
                    raise ValueError("Last dim of pb must be of length 2.")
            else:
                raise ValueError("Last dim of pb must be of length 2.")

    def call(self, inputs):
        """Apply discrete binary memoryless channel to inputs."""

        x, pb = inputs

        # allow pb to be a tuple of two scalars
        if isinstance(pb, (tuple, list)):
            pb0 = pb[0]
            pb1 = pb[1]
        else:
            pb0 = pb[...,0]
            pb1 = pb[...,1]

        # clip for numerical stability
        pb0 = tf.cast(pb0, tf.float32) # Gumble requires float dtypes
        pb1 = tf.cast(pb1, tf.float32) # Gumble requires float dtypes
        pb0 = tf.clip_by_value(pb0, 0., 1.)
        pb1 = tf.clip_by_value(pb1, 0., 1.)

        # check x for consistency (binary, bipolar)
        self._check_inputs(x)

        e0 = self._sample_errors(pb0, x.shape)
        e1 = self._sample_errors(pb1, x.shape)

        if self._bipolar_input:
            neutral_element = tf.constant(-1, dtype=x.dtype)
        else:
            neutral_element = tf.constant(0, dtype=x.dtype)

        # mask e0 and e1 with input such that e0 only applies where x==0
        e = tf.where(x==neutral_element, e0, e1)
        e = tf.cast(e, x.dtype)

        if self._bipolar_input:
            # flip signs for bipolar case
            y = x * (-2*e + 1)
        else:
            # XOR for binary case
            y = self._custom_xor(x, e)

        # if LLRs should be returned
        if self._return_llrs:
            if not self._bipolar_input:
                y = 2 * y - 1 # transform to bipolar

            # Remark: Sionna uses the logit definition log[p(x=1)/p(x=0)]
            y0 = - (tf.math.log(pb1 + self._eps)
                   - tf.math.log(1 - pb0 - self._eps))
            y1 = (tf.math.log(1 - pb1 - self._eps)
                  - tf.math.log(pb0 + self._eps))
            # multiply by y to keep gradient
            y = tf.cast(tf.where(y==1, y1, y0), dtype=y.dtype) * y
            # and clip output llrs
            y = tf.clip_by_value(y, -self._llr_max, self._llr_max)

        return y

class BinarySymmetricChannel(BinaryMemorylessChannel):
    # pylint: disable=line-too-long
    r"""BinarySymmetricChannel(return_llrs=False, bipolar_input=False, llr_max=100., dtype=tf.float32, **kwargs)

    Discrete binary symmetric channel which randomly flips bits with probability
    :math:`p_\text{b}`.

    ..  figure:: ../figures/BSC_channel.png
        :align: center

    This layer supports binary inputs (:math:`x \in \{0, 1\}`) and `bipolar`
    inputs (:math:`x \in \{-1, 1\}`).

    If activated, the channel directly returns log-likelihood ratios (LLRs)
    defined as

    .. math::
        \ell =
        \begin{cases}
            \operatorname{log} \frac{p_{b}}{1-p_{b}}, \qquad \text{if}\, y=0 \\
            \operatorname{log} \frac{1-p_{b}}{p_{b}}, \qquad \text{if}\, y=1 \\
        \end{cases}
    where :math:`y` denotes the binary output of the channel.

    The bit flipping probability :math:`p_\text{b}` can be either a scalar or  a
    tensor (broadcastable to the shape of the input). This allows
    different bit flipping probabilities per bit position.

    This class inherits from the Keras `Layer` class and can be used as layer in
    a Keras model.

    Parameters
    ----------

    return_llrs: bool
        Defaults to `False`. If `True`, the layer returns log-likelihood ratios
        instead of binary values based on ``pb``.

    bipolar_input : bool, False
        Defaults to `False`. If `True`, the expected input is given as {-1,1}
        instead of {0,1}.

    llr_max: tf.float
        Defaults to 100. Defines the clipping value of the LLRs.

    dtype : tf.DType
        Defines the datatype for internal calculations and the output
        dtype. Defaults to `tf.float32`.

    Input
    -----
    (x, pb) :
        Tuple:

    x : [...,n], tf.float32
        Input sequence to the channel.

    pb : tf.float32
        Bit flipping probability. Can be a scalar or of any shape that
        can be broadcasted to the shape of ``x``.

    Output
    -------
        : [...,n], tf.float32
            Output sequence of same length as the input ``x``. If
            ``return_llrs`` is `False`, the output is binary and otherwise
            soft-values are returned.
    """

    def __init__(self, return_llrs=False, bipolar_input=False, llr_max=100., dtype=tf.float32, **kwargs):

        super().__init__(return_llrs=return_llrs,
                         bipolar_input=bipolar_input,
                         llr_max=llr_max,
                         dtype=dtype,
                         **kwargs)

    #########################
    # Keras layer functions
    #########################

    def build(self, input_shapes):
        """Verify correct input shapes"""
        pass # nothing to verify here

    def call(self, inputs):
        """Apply discrete binary symmetric channel, i.e., randomly flip
        bits with probability pb."""

        x, pb = inputs

        # the BSC is implemented by calling the DMC with symmetric pb
        pb = tf.cast(pb, x.dtype)
        pb = tf.stack((pb, pb), axis=-1)
        y = super().call((x, pb))

        return y

class BinaryZChannel(BinaryMemorylessChannel):
    # pylint: disable=line-too-long
    r"""BinaryZChannel(return_llrs=False, bipolar_input=False, llr_max=100., dtype=tf.float32, **kwargs)

    Layer that implements the binary Z-channel.

    In the Z-channel, transmission errors only occur for the transmission of
    second input element (i.e., if a `1` is transmitted) with error probability
    probability :math:`p_\text{b}` but the first element is always correctly
    received.

    ..  figure:: ../figures/Z_channel.png
        :align: center


    This layer supports binary inputs (:math:`x \in \{0, 1\}`) and `bipolar`
    inputs (:math:`x \in \{-1, 1\}`).

    If activated, the channel directly returns log-likelihood ratios (LLRs)
    defined as

    .. math::
        \ell =
        \begin{cases}
            \operatorname{log} \left( p_b \right), \qquad \text{if} \, y=0 \\
            \infty, \qquad \qquad \text{if} \, y=1 \\
        \end{cases}
    assuming equal probable inputs :math:`P(X=0) = P(X=1) = 0.5`.

    The error probability :math:`p_\text{b}` can be either a scalar or a
    tensor (broadcastable to the shape of the input). This allows
    different error probabilities per bit position.

    This class inherits from the Keras `Layer` class and can be used as layer in
    a Keras model.

    Parameters
    ----------

    return_llrs: bool
        Defaults to `False`. If `True`, the layer returns log-likelihood ratios
        instead of binary values based on ``pb``.

    bipolar_input : bool, False
        Defaults to `False`. If True, the expected input is given as {-1,1}
        instead of {0,1}.

    llr_max: tf.float
        Defaults to 100. Defines the clipping value of the LLRs.

    dtype : tf.DType
        Defines the datatype for internal calculations and the output
        dtype. Defaults to `tf.float32`.

    Input
    -----
    (x, pb) :
        Tuple:

    x : [...,n], tf.float32
        Input sequence to the channel.

    pb : tf.float32
        Error probability. Can be a scalar or of any shape that can be
        broadcasted to the shape of ``x``.

    Output
    -------
        : [...,n], tf.float32
            Output sequence of same length as the input ``x``. If
            ``return_llrs`` is `False`, the output is binary and otherwise
            soft-values are returned.
    """

    def __init__(self, return_llrs=False, bipolar_input=False, llr_max=100.,dtype=tf.float32, **kwargs):

        super().__init__(return_llrs=return_llrs,
                         bipolar_input=bipolar_input,
                         llr_max=llr_max,
                         dtype=dtype,
                         **kwargs)

    #########################
    # Keras layer functions
    #########################

    def build(self, input_shapes):
        """Verify correct input shapes"""
        pass # nothing to verify here

    def call(self, inputs):
        """Apply discrete binary symmetric channel, i.e., randomly flip
        bits with probability pb."""

        x, pb = inputs

        # the Z is implemented by calling the DMC with p(1|0)=0
        pb = tf.cast(pb, x.dtype)
        pb = tf.stack((tf.zeros_like(pb), pb), axis=-1)
        y = super().call((x, pb))

        return y


class BinaryErasureChannel(BinaryMemorylessChannel):
    # pylint: disable=line-too-long
    r"""BinaryErasureChannel(return_llrs=False, bipolar_input=False, llr_max=100., dtype=tf.float32, **kwargs)

    Binary erasure channel (BEC) where a bit is either correctly received
    or erased.

    In the binary erasure channel, bits are always correctly received or erased
    with erasure probability :math:`p_\text{b}`.

    ..  figure:: ../figures/BEC_channel.png
        :align: center

    This layer supports binary inputs (:math:`x \in \{0, 1\}`) and `bipolar`
    inputs (:math:`x \in \{-1, 1\}`).

    If activated, the channel directly returns log-likelihood ratios (LLRs)
    defined as

    .. math::
        \ell =
        \begin{cases}
            -\infty, \qquad \text{if} \, y=0 \\
            0, \qquad \quad \,\, \text{if} \, y=? \\
            \infty, \qquad \quad \text{if} \, y=1 \\
        \end{cases}

    The erasure probability :math:`p_\text{b}` can be either a scalar or a
    tensor (broadcastable to the shape of the input). This allows
    different erasure probabilities per bit position.

    Please note that the output of the BEC is ternary. Hereby, `-1` indicates an
    erasure for the binary configuration and `0` for the bipolar mode,
    respectively.

    This class inherits from the Keras `Layer` class and can be used as layer in
    a Keras model.

    Parameters
    ----------

    return_llrs: bool
        Defaults to `False`. If `True`, the layer returns log-likelihood ratios
        instead of binary values based on ``pb``.

    bipolar_input : bool, False
        Defaults to `False`. If `True`, the expected input is given as {-1,1}
        instead of {0,1}.

    llr_max: tf.float
        Defaults to 100. Defines the clipping value of the LLRs.

    dtype : tf.DType
        Defines the datatype for internal calculations and the output
        dtype. Defaults to `tf.float32`.

    Input
    -----
    (x, pb) :
        Tuple:

    x : [...,n], tf.float32
        Input sequence to the channel.

    pb : tf.float32
        Erasure probability. Can be a scalar or of any shape that can be
        broadcasted to the shape of ``x``.

    Output
    -------
        : [...,n], tf.float32
            Output sequence of same length as the input ``x``. If
            ``return_llrs`` is `False`, the output is ternary where each `-1`
            and each `0` indicate an erasure for the binary and bipolar input,
            respectively.
    """

    def __init__(self, return_llrs=False, bipolar_input=False, llr_max=100.,dtype=tf.float32, **kwargs):

        super().__init__(return_llrs=return_llrs,
                         bipolar_input=bipolar_input,
                         llr_max=llr_max,
                         dtype=dtype,
                         **kwargs)

        # also exclude uints, as -1 indicator for erasures does not exist
        assert dtype in (tf.float16, tf.float32, tf.float64,
                tf.int8, tf.int16, tf.int32, tf.int64),\
                "Unsigned integers are currently not supported."

    #########################
    # Keras layer functions
    #########################

    def build(self, input_shapes):
        """Verify correct input shapes"""
        pass # nothing to verify here

    def call(self, inputs):
        """Apply erasure channel to inputs."""

        x, pb = inputs

        # clip for numerical stability
        pb = tf.cast(pb, tf.float32) # Gumble requires float dtypes
        pb = tf.clip_by_value(pb, 0., 1.)

        # check x for consistency (binary, bipolar)
        self._check_inputs(x)

        # sample erasure pattern
        e = self._sample_errors(pb, x.shape)

        # if LLRs should be returned
        # remark: the Sionna logit definition is llr = log[p(x=1)/p(x=0)]
        if self._return_llrs:
            if not self._bipolar_input:
                x = 2 * x -1
            x *= tf.cast(self._llr_max, x.dtype) # calculate llrs

            # erase positions by setting llrs to 0
            y = tf.where(e==1, tf.constant(0, x.dtype), x)
        else: # ternary outputs
            # the erasure indicator depends on the operation mode
            if self._bipolar_input:
                erased_element = tf.constant(0, dtype=x.dtype)
            else:
                erased_element = tf.constant(-1, dtype=x.dtype)

            y = tf.where(e==0, x, erased_element)
        return y
