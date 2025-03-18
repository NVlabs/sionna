#
# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0#
"""Classes for the simulation of flat-fading channels"""

import tensorflow as tf
from sionna.phy.channel import AWGN
from sionna.phy.utils import complex_normal
from sionna.phy import Block

class GenerateFlatFadingChannel(Block):
    # pylint: disable=line-too-long
    r"""Generates tensors of flat-fading channel realizations

    This class generates batches of random flat-fading channel matrices.
    A spatial correlation can be applied.

    Parameters
    ----------
    num_tx_ant : `int`
        Number of transmit antennas

    num_rx_ant : `int`
        Number of receive antennas

    spatial_corr : `None` (default) | :class:`~sionna.phy.channel.SpatialCorrelation`
        Spatial correlation to be applied

    precision : `None` (default) | "single" | "double"
        Precision used for internal calculations and outputs.
        If set to `None`,
        :attr:`~sionna.phy.config.Config.precision` is used.

    Input
    -----
    batch_size : `int`
        Number of channel matrices to generate

    Output
    ------
    h : [batch_size, num_rx_ant, num_tx_ant], `tf.complex`
        Batch of random flat fading channel matrices

    """
    def __init__(self, num_tx_ant, num_rx_ant, spatial_corr=None, precision=None, **kwargs):
        super().__init__(precision=precision, **kwargs)
        self._num_tx_ant = num_tx_ant
        self._num_rx_ant = num_rx_ant
        self.spatial_corr = spatial_corr

    @property
    def spatial_corr(self):
        """
        :class:`~sionna.phy.channel.SpatialCorrelation` : Get/set spatial 
            correlation to be applied
        """
        return self._spatial_corr

    @spatial_corr.setter
    def spatial_corr(self, value):
        self._spatial_corr = value

    def call(self, batch_size):
        # Generate standard complex Gaussian matrices
        shape = [batch_size, self._num_rx_ant, self._num_tx_ant]
        h = complex_normal(shape, precision=self.precision)

        # Apply spatial correlation
        if self.spatial_corr is not None:
            h = self.spatial_corr(h)

        return h

class ApplyFlatFadingChannel(Block):
    # pylint: disable=line-too-long
    r"""
    Applies given channel matrices to a vector input and adds AWGN

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
    precision : `None` (default) | "single" | "double"
        Precision used for internal calculations and outputs.
        If set to `None`,
        :attr:`~sionna.phy.config.Config.precision` is used.

    Input
    -----
    x : [batch_size, num_tx_ant], `tf.complex`
        Transmit vectors

    h : [batch_size, num_rx_ant, num_tx_ant], `tf.complex`
        Channel realizations. Will be broadcast to the
        dimensions of ``x`` if needed.

    no : `None` (default) | Tensor, `tf.float`
        (Optional) noise power ``no`` per complex dimension.
        Will be broadcast to the shape of ``y``.
        For more details, see :class:`~sionna.phy.channel.AWGN`.

    Output
    ------
    y : [batch_size, num_rx_ant], `tf.complex`
        Channel output
    """
    def __init__(self, precision=None, **kwargs):
        super().__init__(precision=precision, **kwargs)
        self._awgn = AWGN(precision=self.precision)

    def call(self, x, h, no=None):
        x = tf.expand_dims(x, axis=-1)
        y = tf.matmul(h, x)
        y = tf.squeeze(y, axis=-1)

        if no is not None:
            y = self._awgn(y, no)

        return y

class FlatFadingChannel(Block):
    # pylint: disable=line-too-long
    r"""
    Applies random channel matrices to a vector input and adds AWGN

    This class combines :class:`~sionna.phy.channel.GenerateFlatFadingChannel` and
    :class:`~sionna.phy.channel.ApplyFlatFadingChannel` and computes the output of
    a flat-fading channel with AWGN.

    For a given batch of input vectors :math:`\mathbf{x}\in\mathbb{C}^{K}`,
    the output is

    .. math::

        \mathbf{y} = \mathbf{H}\mathbf{x} + \mathbf{n}

    where :math:`\mathbf{H}\in\mathbb{C}^{M\times K}` are randomly generated
    flat-fading channel matrices and
    :math:`\mathbf{n}\in\mathbb{C}^{M}\sim\mathcal{CN}(0, N_o\mathbf{I})`
    is an AWGN vector that is optionally added.

    A :class:`~sionna.phy.channel.SpatialCorrelation` can be configured and the
    channel realizations optionally returned. This is useful to simulate
    receiver algorithms with perfect channel knowledge.

    Parameters
    ----------
    num_tx_ant : `int`
        Number of transmit antennas

    num_rx_ant : `int`
        Number of receive antennas

    spatial_corr : `None` (default) | :class:`~sionna.phy.channel.SpatialCorrelation`
        Spatial correlation to be applied

    return_channel: `bool`, (default `False`)
        Indicates if the channel realizations should be returned

    precision : `None` (default) | "single" | "double"
        Precision used for internal calculations and outputs.
        If set to `None`,
        :attr:`~sionna.phy.config.Config.precision` is used.

    Input
    -----
    x : [batch_size, num_tx_ant], `tf.complex`
        Tensor of transmit vectors

    no : `None` (default) | Tensor, `tf.float`
        (Optional) noise power ``no`` per complex dimension.
        Will be broadcast to the shape of ``y``.
        For more details, see :class:`~sionna.phy.channel.AWGN`.

    Output
    ------
    y : [batch_size, num_rx_ant], `tf.complex`
        Channel output

    h : [batch_size, num_rx_ant, num_tx_ant], `tf.complex`
        Channel realizations. Will only be returned if
        ``return_channel==True``.
    """
    def __init__(self,
                 num_tx_ant,
                 num_rx_ant,
                 spatial_corr=None,
                 return_channel=False,
                 precision=None,
                 **kwargs):
        super().__init__(precision=precision, **kwargs)
        self._num_tx_ant = num_tx_ant
        self._num_rx_ant = num_rx_ant
        self._return_channel = return_channel
        self._gen_chn = GenerateFlatFadingChannel(self._num_tx_ant,
                                                  self._num_rx_ant,
                                                  spatial_corr,
                                                  precision=precision)
        self._app_chn = ApplyFlatFadingChannel(precision=precision)

    @property
    def spatial_corr(self):
        """
        :class:`~sionna.phy.channel.SpatialCorrelation` : Get/set spatial 
            correlation to be applied
        """
        return self._gen_chn.spatial_corr

    @spatial_corr.setter
    def spatial_corr(self, value):
        self._gen_chn.spatial_corr = value

    @property
    def generate(self):
        """Calls the internal :class:`GenerateFlatFadingChannel`"""
        return self._gen_chn

    @property
    def apply(self):
        """Calls the internal :class:`ApplyFlatFadingChannel`"""
        return self._app_chn

    def call(self, x, no=None):
        # Generate a batch of channel realizations
        batch_size = tf.shape(x)[0]
        h = self._gen_chn(batch_size)

        # Apply the channel to the input
        y = self._app_chn(x, h, no)

        if self._return_channel:
            return y, h
        else:
            return y
