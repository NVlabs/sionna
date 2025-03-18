#
# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0#
"""Class definition and functions related to OFDM transmit precoding"""

from abc import abstractmethod
import tensorflow as tf
import sionna
from sionna.phy import Block
from sionna.phy.utils import flatten_dims, expand_to_rank
from sionna.phy.mimo import rzf_precoder,\
                            rzf_precoding_matrix, cbf_precoding_matrix
from sionna.phy.ofdm import RemoveNulledSubcarriers

class RZFPrecoder(Block):
    # pylint: disable=line-too-long
    r"""
    Regularized zero-forcing (RZF) precoding for multi-antenna transmissions

    This block precodes a tensor containing OFDM resource grids using
    the :meth:`~sionna.phy.mimo.rzf_precoder`. For every
    transmitter, the channels to all intended receivers are gathered
    into a channel matrix, based on the which the precoding matrix
    is computed and the input tensor is precoded. The block also outputs
    optionally the effective channel after precoding for each stream.

    Parameters
    ----------
    resource_grid : :class:`~sionna.phy.ofdm.ResourceGrid`
        ResourceGrid to be sued

    stream_management : :class:`~sionna.phy.mimo.StreamManagement`
        StreamManagement to be used

    return_effective_channel : `bool`, (default `False`)
        Indicates if the effective channel after precoding should be returned

    precision : `None` (default) | "single" | "double"
        Precision used for internal calculations and outputs.
        If set to `None`,
        :attr:`~sionna.phy.config.Config.precision` is used.

    Input
    -----
    x : [batch_size, num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size], `tf.complex`
        Resource grids to be precoded.

    h : [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_ofdm_symbols, fft_size], `tf.complex`
        Channel knowledge based on which the precoding is computed

    alpha : `0.` (default) | [batch_size, num_tx, num_ofdm_symbols, fft_size] (or broadcastable), `float`
        Regularization parameter for RZF precoding. If set to `0`, RZF is equivalent
        to ZF precoding.

    Output
    ------
    x_precoded : [batch_size, num_tx, num_tx_ant, num_ofdm_symbols, fft_size], `tf.complex`
        Precoded resource grids

    h_eff : [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols num_effective_subcarriers], `tf.complex`
        Only returned if ``return_effective_channel=True``.
        The effectice channels for all streams after precoding. Can be used to
        simulate perfect channel state information (CSI) at the receivers.
        Nulled subcarriers are automatically removed to be compliant with the
        behavior of a channel estimator.
    """
    def __init__(self,
                 resource_grid,
                 stream_management,
                 return_effective_channel=False,
                 precision=None,
                 **kwargs):
        super().__init__(precision=precision, **kwargs)
        assert isinstance(resource_grid, sionna.phy.ofdm.ResourceGrid)
        assert isinstance(stream_management, sionna.phy.mimo.StreamManagement)
        self._resource_grid = resource_grid
        self._stream_management = stream_management
        self._return_effective_channel = return_effective_channel
        self._remove_nulled_scs = RemoveNulledSubcarriers(self._resource_grid)

    def _compute_effective_channel(self, h, g):
        """Compute effective channel after precoding"""

        # Input dimensions:
        # h: [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant,...
        #     ..., num_ofdm_symbols, fft_size]
        # g: [batch_size, num_tx, num_ofdm_symbols, fft_size, num_tx_ant,
        #     ..., num_streams_per_tx]

        # Transpose h to shape:
        # [batch_size, num_rx, num_tx, num_ofdm_symbols, fft_size, num_rx_ant,...
        #  ..., num_tx_ant]
        h = tf.transpose(h, [0, 1, 3, 5, 6, 2, 4])
        h = tf.cast(h, g.dtype)

        # Add one dummy dimension to g to be broadcastable to h:
        # [batch_size, 1, num_tx, num_ofdm_symbols, fft_size, num_tx_ant,...
        #  ..., num_streams_per_tx]
        g = tf.expand_dims(g, 1)

        # Compute post precoding channel:
        # [batch_size, num_rx, num_tx, num_ofdm_symbols, fft_size, num_rx_ant,...
        #  ..., num_streams_per_tx]
        h_eff = tf.matmul(h, g)

        # Permute dimensions to common format of channel tensors:
        # [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx,...
        #  ..., num_ofdm_symbols, fft_size]
        h_eff = tf.transpose(h_eff, [0, 1, 5, 2, 6, 3, 4])

        # Remove nulled subcarriers:
        # [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx,...
        #  ..., num_ofdm_symbols num_effective_subcarriers]
        h_eff = self._remove_nulled_scs(h_eff)

        return h_eff

    def call(self, x, h, alpha=0.):

        # x has shape
        # [batch_size, num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size]
        #
        # h has shape
        # [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_ofdm_symbols...
        # ..., fft_size]

        ###
        ### Transformations to bring h and x in the desired shapes
        ###

        # Transpose x:
        #[batch_size, num_tx, num_ofdm_symbols, fft_size, num_streams_per_tx]
        x_precoded = tf.transpose(x, [0, 1, 3, 4, 2])
        x_precoded = tf.cast(x_precoded, self.cdtype)

        # Transpose h:
        # [num_tx, num_rx, num_rx_ant, num_tx_ant, num_ofdm_symbols,...
        #  ..., fft_size, batch_size]
        h_pc = tf.transpose(h, [3, 1, 2, 4, 5, 6, 0])

        # Gather desired channel for precoding:
        # [num_tx, num_rx_per_tx, num_rx_ant, num_tx_ant, num_ofdm_symbols,...
        #  ..., fft_size, batch_size]
        h_pc_desired = tf.gather(h_pc, self._stream_management.precoding_ind,
                                 axis=1, batch_dims=1)

        # Flatten dims 2,3:
        # [num_tx, num_rx_per_tx * num_rx_ant, num_tx_ant, num_ofdm_symbols,...
        #  ..., fft_size, batch_size]
        h_pc_desired = flatten_dims(h_pc_desired, 2, axis=1)

        # Transpose:
        # [batch_size, num_tx, num_ofdm_symbols, fft_size,...
        #  ..., num_streams_per_tx, num_tx_ant]
        h_pc_desired = tf.transpose(h_pc_desired, [5, 0, 3, 4, 1, 2])
        h_pc_desired = tf.cast(h_pc_desired, self.cdtype)

        ###
        ### ZF precoding
        ###
        #[batch_size, num_tx, num_ofdm_symbols, fft_size]
        alpha = tf.cast(alpha, self.rdtype)
        alpha = expand_to_rank(alpha, 4, axis=0)
        x_precoded, g = rzf_precoder(x_precoded,
                                     h_pc_desired,
                                     alpha=alpha,
                                     return_precoding_matrix=True)

        # Transpose output to desired shape:
        #[batch_size, num_tx, num_tx_ant, num_ofdm_symbols, fft_size]
        x_precoded = tf.transpose(x_precoded, [0, 1, 4, 2, 3])

        if self._return_effective_channel:
            h_eff = self._compute_effective_channel(h, g)
            return (x_precoded, h_eff)
        else:
            return x_precoded

class PrecodedChannel(Block):
    # pylint: disable=line-too-long
    r"""
    Abstract base class to compute the effective channel after precoding

    Its output can be used to compute the :class:`~sionna.phy.ofdm.PostEqualizationSINR`.

    Let
    :math:`\mathbf{H}_{i,j}\in\mathbb{C}^{\text{num_rx_ant}\times\text{num_tx_ant}}`
    be the channel matrix between transmitter :math:`j`
    and receiver :math:`i` and let
    :math:`\mathbf{G}_{j}\in\mathbb{C}^{\text{num_tx_ant}\times\text{num_streams_per_tx}}`
    be the precoding matrix of transmitter :math:`j`. 

    The effective channel :math:`\widetilde{\mathbf{H}}_{i,j}\in\mathbb{C}^{\text{num_rx_ant}\times\text{num_streams_per_tx}}`
    after precoding is given by

    .. math::
        :label: effective_precoded_channel

        \widetilde{\mathbf{H}}_{i,j} = \mathbf{H}_{i,j}\mathbf{G}_{j}\mathop{\text{diag}}(\sqrt{p_{j,1}},...,\sqrt{p_{j,\text{num_streams_per_tx}}})

    where :math:`p_{j,s}` is the transmit power of stream :math:`s` of transmitter :math:`j`.

    Parameters
    ----------
    resource_grid : :class:`~sionna.phy.ofdm.ResourceGrid`
        ResourceGrid to be used

    stream_management : :class:`~sionna.phy.mimo.StreamManagement`
        StreamManagement to be used

    precision : `None` (default) | "single" | "double"
        Precision used for internal calculations and outputs.
        If set to `None`,
        :attr:`~sionna.phy.config.Config.precision` is used.

    Input
    -----
    h : [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_ofdm_symbols, fft_size], `tf.complex`
        Actual channel realizations

    tx_power : [batch_size, num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size] (or first n dims), `tf.float32`
        Power of each stream for each transmitter

    h_hat : `None` (default) | [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_ofdm_symbols, fft_size], `tf.complex`
        Channel knowledge based on which the precoding is computed. If set to `None`,
        the actual channel realizations are used.

    Output
    ------
    h_eff : [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols num_effective_subcarriers], `tf.complex`
        The effective channel after precoding. Nulled subcarriers are
        automatically removed.
    """
    def __init__(self,
                 resource_grid,
                 stream_management,
                 precision=None,
                 **kwargs):
        super().__init__(precision=precision, **kwargs)
        assert isinstance(resource_grid, sionna.phy.ofdm.ResourceGrid)
        assert isinstance(stream_management, sionna.phy.mimo.StreamManagement)
        self._resource_grid = resource_grid
        self._stream_management = stream_management
        self._remove_nulled_scs = RemoveNulledSubcarriers(self._resource_grid)

    def get_desired_channels(self, h_hat):
        # pylint: disable=line-too-long
        r"""
        Get the desired channels for precoding

        Input
        -----
        h_hat : [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_ofdm_symbols, fft_size], `tf.complex`
            Channel knowledge based on which the precoding is computed

        Output
        ------
        h_pc_desired : [batch_size, num_tx, num_ofdm_symbols, fft_size, num_streams_per_tx, num_tx_ant], `tf.complex`
            Desired channels for precoding
        """
        # h_hat has shape
        # [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_ofdm_symbols...
        # ..., fft_size]

        # Transpose:
        # [num_tx, num_rx, num_rx_ant, num_tx_ant, num_ofdm_symbols,...
        #  ..., fft_size, batch_size]
        h_pc_desired = tf.transpose(h_hat, [3, 1, 2, 4, 5, 6, 0])

        # Gather desired channel for precoding:
        # [num_tx, num_rx_per_tx, num_rx_ant, num_tx_ant, num_ofdm_symbols,...
        #  ..., fft_size, batch_size]
        h_pc_desired = tf.gather(h_pc_desired,
                                 self._stream_management.precoding_ind,
                                 axis=1, batch_dims=1)
        # Flatten dims 1,2:
        # [num_tx, num_rx_per_tx * num_rx_ant, num_tx_ant, num_ofdm_symbols,...
        #  ..., fft_size, batch_size]
        h_pc_desired = flatten_dims(h_pc_desired, 2, axis=1)

        # Transpose:
        # [batch_size, num_tx, num_ofdm_symbols, fft_size,...
        #  ..., num_streams_per_tx, num_tx_ant]
        h_pc_desired = tf.transpose(h_pc_desired, [5, 0, 3, 4, 1, 2])

        num_streams_per_tx = self._stream_management.num_streams_per_tx

        # Check if number of streams per tx matches the channel dimensions
        if h_pc_desired.shape[-2] != num_streams_per_tx:
            msg = "The required number of streams per transmitter" \
                  + " does not match the channel dimensions"
            raise ValueError(msg)


        return h_pc_desired

    def compute_effective_channel(self, h, g):
        # pylint: disable=line-too-long
        r"""Compute effective channel after precoding

        Input
        -----
        h : [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_ofdm_symbols, fft_size], `tf.complex`
            Actual channel realizations

        g : [batch_size, num_tx, num_ofdm_symbols, fft_size, num_tx_ant, num_streams_per_tx], `tf.complex`
            Precoding matrix
        Output
        ------
        h_eff : [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols num_effective_subcarriers], `tf.complex`
            The effective channel after precoding. Nulled subcarriers are
            automatically removed.
        """
        # Input dimensions:
        # h: [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant,...
        #     ..., num_ofdm_symbols, fft_size]
        # g: [batch_size, num_tx, num_ofdm_symbols, fft_size, num_tx_ant,
        #     ..., num_streams_per_tx]

        # Transpose h to shape:
        # [batch_size, num_rx, num_tx, num_ofdm_symbols, fft_size, num_rx_ant,...
        #  ..., num_tx_ant]
        h = tf.transpose(h, [0, 1, 3, 5, 6, 2, 4])
        h = tf.cast(h, g.dtype)

        # Add one dummy dimension to g to be broadcastable to h:
        # [batch_size, 1, num_tx, num_ofdm_symbols, fft_size, num_tx_ant,...
        #  ..., num_streams_per_tx]
        g = tf.expand_dims(g, 1)

        # Compute post precoding channel:
        # [batch_size, num_rx, num_tx, num_ofdm_symbols, fft_size, num_rx_ant,...
        #  ..., num_streams_per_tx]
        h_eff = tf.matmul(h, g)

        # Permute dimensions to common format of channel tensors:
        # [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx,...
        #  ..., num_ofdm_symbols, fft_size]
        h_eff = tf.transpose(h_eff, [0, 1, 5, 2, 6, 3, 4])

        # Remove nulled subcarriers:
        # [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx,...
        #  ..., num_ofdm_symbols num_effective_subcarriers]
        h_eff = self._remove_nulled_scs(h_eff)

        return h_eff

    def apply_tx_power(self, g, tx_power):
        r"""Apply transmit power to precoding vectors

        Input
        -----
        g : [batch_size, num_tx, num_ofdm_symbols, fft_size, num_tx_ant, num_streams_per_tx], `tf.complex`
            Precoding vectors

        tx_power : [batch_size, num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size] (or first n dims), `tf.float32`
            Power of each stream for each transmitter
        """
        # [batch_size, num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size,
        # ...num_streams_per_tx]
        tx_power = expand_to_rank(tx_power, 6, axis=-1)
        # [batch_size, num_tx, num_ofdm_symbols, fft_size, num_streams_per_tx]
        tx_power = tf.transpose(tx_power, [0, 1, 3, 4, 5, 2])
        tx_power = tf.broadcast_to(tx_power, tf.shape(g))

        # Apply tx power to precoding matrix
        g = tf.cast(tf.sqrt(tx_power), self.cdtype) * g

        return g

    @abstractmethod
    def call(self, h, tx_power, h_hat=None, **kwargs):
        pass

class RZFPrecodedChannel(PrecodedChannel):
    # pylint: disable=line-too-long
    r"""
    Compute the effective channel after RZF precoding

    The precoding matrices are obtained from :func:`~sionna.phy.mimo.rzf_precoding_matrix`.

    Parameters
    ----------
    resource_grid : :class:`~sionna.phy.ofdm.ResourceGrid`
        ResourceGrid to be used

    stream_management : :class:`~sionna.phy.mimo.StreamManagement`
        StreamManagement to be used 

    precision : `None` (default) | "single" | "double"
        Precision used for internal calculations and outputs.
        If set to `None`,
        :attr:`~sionna.phy.config.Config.precision` is used.

    Input
    -----
    h : [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_ofdm_symbols, fft_size], `tf.complex`
        Actual channel realizations

    tx_power : [batch_size, num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size] (or first n dims), `tf.float32`
        Power of each stream for each transmitter

    h_hat : `None` (default) | [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_ofdm_symbols, fft_size], `tf.complex`
        Channel knowledge based on which the precoding is computed. If set to `None`,
        the actual channel realizations are used.

    alpha : `0.` (default) | [batch_size, num_tx, num_ofdm_symbols, fft_size] (or first n dims), `float`
        Regularization parameter for RZF precoding. If set to `0`, RZF is equivalent
        to ZF precoding.

    Output
    ------
    h_eff : [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols, num_effective_subcarriers], `tf.complex`
        The effective channel after precoding. Nulled subcarriers are
        automatically removed.
    """
    def call(self, h, tx_power, h_hat=None, alpha=0.):
        """
        Compute the effective channel after precoding
        """
        if h_hat is None:
            h_hat = h

        # Get desired channels for precoding
        # [batch_size, num_tx, num_ofdm_symbols, fft_size,
        #  ..., num_streams_per_tx, num_tx_ant]
        h_pc_desired = self.get_desired_channels(h_hat)

        # Compute precoding matrix
        #[batch_size, num_tx, num_ofdm_symbols, fft_size]
        alpha = tf.cast(alpha, self.rdtype)
        alpha = expand_to_rank(alpha, 4, axis=-1)
        alpha = tf.broadcast_to(alpha, tf.shape(h_pc_desired)[:4])

        # [batch_size, num_tx, num_ofdm_symbols, fft_size,
        #  ..., num_tx_ant,num_streams_per_tx]
        g = rzf_precoding_matrix(h_pc_desired,
                                 alpha,
                                 precision=self.precision)
        # Apply transmit power to precoding matrix
        g = self.apply_tx_power(g, tx_power)

        # Compute effective channel
        h_eff = self.compute_effective_channel(h, g)

        return h_eff

class CBFPrecodedChannel(PrecodedChannel):
    # pylint: disable=line-too-long
    r"""
    Compute the effective channel after conjugate beamforming (CBF) precoding

    The precoding matrices are obtained from :func:`~sionna.phy.mimo.cbf_precoding_matrix`.

    Parameters
    ----------
    resource_grid : :class:`~sionna.phy.ofdm.ResourceGrid`
        ResourceGrid to be used

    stream_management : :class:`~sionna.phy.mimo.StreamManagement`
        StreamManagement to be used

    precision : `None` (default) | "single" | "double"
        Precision used for internal calculations and outputs.
        If set to `None`,
        :attr:`~sionna.phy.config.Config.precision` is used.

    Input
    -----
    h : [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_ofdm_symbols, fft_size], `tf.complex`
        Actual channel realizations

    tx_power : [batch_size, num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size] (or first n dims), `tf.float32`
        Power of each stream for each transmitter

    h_hat : `None` (default) | [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_ofdm_symbols, fft_size], `tf.complex`
        Channel knowledge based on which the precoding is computed. If set to `None`,
        the actual channel realizations are used.

    Output
    ------
    h_eff : [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols, num_effective_subcarriers], `tf.complex`
        The effective channel after precoding. Nulled subcarriers are
        automatically removed.
    """
    def call(self, h, tx_power, h_hat=None):
        """
        Compute the effective channel after precoding
        """
        if h_hat is None:
            h_hat = h

        # Get desired channels for precoding
        # [batch_size, num_tx, num_ofdm_symbols, fft_size,
        #  ..., num_streams_per_tx, num_tx_ant]
        h_pc_desired = self.get_desired_channels(h_hat)

        # Compute precoding matrix
        # [batch_size, num_tx, num_ofdm_symbols, fft_size,
        #  ..., num_tx_ant,num_streams_per_tx]
        g = cbf_precoding_matrix(h_pc_desired,
                                 precision=self.precision)

        # Apply transmit power to precoding matrix
        g = self.apply_tx_power(g, tx_power)

        # Compute effective channel
        h_eff = self.compute_effective_channel(h, g)

        return h_eff


class EyePrecodedChannel(PrecodedChannel):
    # pylint: disable=line-too-long
    r"""
    Compute the effective channel after power allocation without precoding, i.e.,
    the identity matrix precoder is used

    Parameters
    ----------
    resource_grid : :class:`~sionna.phy.ofdm.ResourceGrid`
        ResourceGrid to be used

    stream_management : :class:`~sionna.phy.mimo.StreamManagement`
        StreamManagement to be used

    precision : `None` (default) | "single" | "double"
        Precision used for internal calculations and outputs.
        If set to `None`,
        :attr:`~sionna.phy.config.Config.precision` is used.

    Input
    -----
    h : [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_ofdm_symbols, fft_size], `tf.complex`
        Actual channel realizations

    tx_power : [batch_size, num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size] (or broadcastable), `tf.float32`
        Power of each stream for each transmitter. Also a lower-rank tensor is
        accepted if it is broadcastable to the requested shape.

    Output
    ------
    h_eff : [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols, num_effective_subcarriers], `tf.complex`
        The effective channel after power allocation. Nulled subcarriers are
        automatically removed.
    """
    def call(self, h, tx_power):
        """
        Compute the effective channel after precoding
        """
        batch_size, _, _, num_tx, num_tx_ant, num_ofdm_symbols, fft_size = h.shape

        # Compute identity precoding matrix
        # [batch_size, num_tx, num_ofdm_symbols, fft_size,
        #  ..., num_tx_ant, num_streams_per_tx=num_tx_ant]
        g = tf.eye(num_tx_ant,
                   batch_shape=[batch_size, num_tx, num_ofdm_symbols, fft_size],
                   dtype=self.cdtype)

        # Apply transmit power to precoding matrix
        g = self.apply_tx_power(g, tx_power)

        # Compute effective channel
        h_eff = self.compute_effective_channel(h, g)

        return h_eff
