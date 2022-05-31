#
# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Classes for the simulation of flat-fading channels"""

import tensorflow as tf
from sionna.channel import AWGN
from sionna.utils import complex_normal

class GenerateFlatFadingChannel():
    # pylint: disable=line-too-long
    r"""Generates tensors of flat-fading channel realizations.

    This class generates batches of random flat-fading channel matrices.
    A spatial correlation can be applied.

    Parameters
    ----------
    num_tx_ant : int
        Number of transmit antennas.

    num_rx_ant : int
        Number of receive antennas.

    spatial_corr : SpatialCorrelation, None
        An instance of :class:`~sionna.channel.SpatialCorrelation` or `None`.
        Defaults to `None`.

    dtype : tf.complex64, tf.complex128
        The dtype of the output. Defaults to `tf.complex64`.

    Input
    -----
    batch_size : int
        The batch size, i.e., the number of channel matrices to generate.

    Output
    ------
    h : [batch_size, num_rx_ant, num_tx_ant], ``dtype``
        Batch of random flat fading channel matrices.

    """
    def __init__(self, num_tx_ant, num_rx_ant, spatial_corr=None, dtype=tf.complex64, **kwargs):
        super().__init__(**kwargs)
        self._num_tx_ant = num_tx_ant
        self._num_rx_ant = num_rx_ant
        self._dtype = dtype
        self.spatial_corr = spatial_corr

    @property
    def spatial_corr(self):
        """The :class:`~sionna.channel.SpatialCorrelation` to be used."""
        return self._spatial_corr

    @spatial_corr.setter
    def spatial_corr(self, value):
        self._spatial_corr = value

    def __call__(self, batch_size):
        # Generate standard complex Gaussian matrices
        shape = [batch_size, self._num_rx_ant, self._num_tx_ant]
        h = complex_normal(shape, dtype=self._dtype)

        # Apply spatial correlation
        if self.spatial_corr is not None:
            h = self.spatial_corr(h)

        return h

class ApplyFlatFadingChannel(tf.keras.layers.Layer):
    # pylint: disable=line-too-long
    r"""ApplyFlatFadingChannel(add_awgn=True, dtype=tf.complex64, **kwargs)

    Applies given channel matrices to a vector input and adds AWGN.

    This class applies a given tensor of flat-fading channel matrices
    to an input tensor. AWGN noise can be optionally added.
    Mathematically, for channel matrices
    :math:`\mathbf{H}\in\mathbb{C}^{M\times K}`
    and input :math:`\mathbf{x}\in\mathbb{C}^{K}`, the output is

    .. math::

        \mathbf{y} = \mathbf{H}\mathbf{x} + \mathbf{n}

    where :math:`\mathbf{n}\in\mathbb{C}^{M}\sim\mathcal{CN}(0, N_o\mathbf{I})`
    is an AWGN vector that is optionally added.


    Parameters
    ----------
    add_awgn: bool
        Indicates if AWGN noise should be added to the output.
        Defaults to `True`.

    dtype : tf.complex64, tf.complex128
        The dtype of the output. Defaults to `tf.complex64`.

    Input
    -----
    (x, h, no) :
        Tuple:

    x : [batch_size, num_tx_ant], tf.complex
        Tensor of transmit vectors.

    h : [batch_size, num_rx_ant, num_tx_ant], tf.complex
        Tensor of channel realizations. Will be broadcast to the
        dimensions of ``x`` if needed.

    no : Scalar or Tensor, tf.float
        The noise power ``no`` is per complex dimension.
        Only required if ``add_awgn==True``.
        Will be broadcast to the shape of ``y``.
        For more details, see :class:`~sionna.channel.AWGN`.

    Output
    ------
    y : [batch_size, num_rx_ant, num_tx_ant], ``dtype``
        Channel output.
    """
    def __init__(self, add_awgn=True, dtype=tf.complex64, **kwargs):
        super().__init__(trainable=False, dtype=dtype, **kwargs)
        self._add_awgn = add_awgn

    def build(self, input_shape): #pylint: disable=unused-argument
        if self._add_awgn:
            self._awgn = AWGN(dtype=self.dtype)

    def call(self, inputs):
        if self._add_awgn:
            x, h, no = inputs
        else:
            x, h = inputs

        x = tf.expand_dims(x, axis=-1)
        y = tf.matmul(h, x)
        y = tf.squeeze(y, axis=-1)

        if self._add_awgn:
            y = self._awgn((y, no))

        return y

class FlatFadingChannel(tf.keras.layers.Layer):
    # pylint: disable=line-too-long
    r"""FlatFadingChannel(num_tx_ant, num_rx_ant, spatial_corr=None, add_awgn=True, return_channel=False, dtype=tf.complex64, **kwargs)

    Applies random channel matrices to a vector input and adds AWGN.

    This class combines :class:`~sionna.channel.GenerateFlatFadingChannel` and
    :class:`~sionna.channel.ApplyFlatFadingChannel` and computes the output of
    a flat-fading channel with AWGN.

    For a given batch of input vectors :math:`\mathbf{x}\in\mathbb{C}^{K}`,
    the output is

    .. math::

        \mathbf{y} = \mathbf{H}\mathbf{x} + \mathbf{n}

    where :math:`\mathbf{H}\in\mathbb{C}^{M\times K}` are randomly generated
    flat-fading channel matrices and
    :math:`\mathbf{n}\in\mathbb{C}^{M}\sim\mathcal{CN}(0, N_o\mathbf{I})`
    is an AWGN vector that is optionally added.

    A :class:`~sionna.channel.SpatialCorrelation` can be configured and the
    channel realizations optionally returned. This is useful to simulate
    receiver algorithms with perfect channel knowledge.

    Parameters
    ----------
    num_tx_ant : int
        Number of transmit antennas.

    num_rx_ant : int
        Number of receive antennas.

    spatial_corr : SpatialCorrelation, None
        An instance of :class:`~sionna.channel.SpatialCorrelation` or `None`.
        Defaults to `None`.

    add_awgn: bool
        Indicates if AWGN noise should be added to the output.
        Defaults to `True`.

    return_channel: bool
        Indicates if the channel realizations should be returned.
        Defaults  to `False`.

    dtype : tf.complex64, tf.complex128
        The dtype of the output. Defaults to `tf.complex64`.

    Input
    -----
    (x, no) :
        Tuple or Tensor:

    x : [batch_size, num_tx_ant], tf.complex
        Tensor of transmit vectors.

    no : Scalar of Tensor, tf.float
        The noise power ``no`` is per complex dimension.
        Only required if ``add_awgn==True``.
        Will be broadcast to the dimensions of the channel output if needed.
        For more details, see :class:`~sionna.channel.AWGN`.

    Output
    ------
    (y, h) :
        Tuple or Tensor:

    y : [batch_size, num_rx_ant, num_tx_ant], ``dtype``
        Channel output.

    h : [batch_size, num_rx_ant, num_tx_ant], ``dtype``
        Channel realizations. Will only be returned if
        ``return_channel==True``.
    """
    def __init__(self,
                 num_tx_ant,
                 num_rx_ant,
                 spatial_corr=None,
                 add_awgn=True,
                 return_channel=False,
                 dtype=tf.complex64,
                 **kwargs):
        super().__init__(trainable=False, dtype=dtype, **kwargs)
        self._num_tx_ant = num_tx_ant
        self._num_rx_ant = num_rx_ant
        self._add_awgn = add_awgn
        self._return_channel = return_channel
        self._gen_chn = GenerateFlatFadingChannel(self._num_tx_ant,
                                                  self._num_rx_ant,
                                                  spatial_corr,
                                                  dtype=dtype)
        self._app_chn = ApplyFlatFadingChannel(add_awgn=add_awgn, dtype=dtype)

    @property
    def spatial_corr(self):
        """The :class:`~sionna.channel.SpatialCorrelation` to be used."""
        return self._gen_chn.spatial_corr

    @spatial_corr.setter
    def spatial_corr(self, value):
        self._gen_chn.spatial_corr = value

    @property
    def generate(self):
        """Calls the internal :class:`GenerateFlatFadingChannel`."""
        return self._gen_chn

    @property
    def apply(self):
        """Calls the internal :class:`ApplyFlatFadingChannel`."""
        return self._app_chn

    def call(self, inputs):
        if self._add_awgn:
            x, no = inputs
        else:
            x = inputs

        # Generate a batch of channel realizations
        batch_size = tf.shape(x)[0]
        h = self._gen_chn(batch_size)

        # Apply the channel to the input
        if self._add_awgn:
            y = self._app_chn([x, h, no])
        else:
            y = self._app_chn([x, h])

        if self._return_channel:
            return y, h
        else:
            return y
