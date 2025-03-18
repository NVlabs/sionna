#
# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0#
"""Class definition and functions related to OFDM channel equalization"""

from abc import abstractmethod
import tensorflow as tf

import sionna
from sionna.phy import Block
from sionna.phy.utils import flatten_dims, split_dim, flatten_last_dims,\
                             expand_to_rank, inv_cholesky
from sionna.phy.mimo import lmmse_equalizer, zf_equalizer, mf_equalizer,\
                            lmmse_matrix
from sionna.phy.ofdm import RemoveNulledSubcarriers

class OFDMEqualizer(Block):
    # pylint: disable=line-too-long
    r"""
    Block that wraps a MIMO equalizer for use with the OFDM waveform

    The parameter ``equalizer`` is a callable (e.g., a function) that
    implements a MIMO equalization algorithm for arbitrary batch dimensions.

    This class pre-processes the received resource grid ``y`` and channel
    estimate ``h_hat``, and computes for each receiver the
    noise-plus-interference covariance matrix according to the OFDM and stream
    configuration provided by the ``resource_grid`` and
    ``stream_management``, which also accounts for the channel
    estimation error variance ``err_var``. These quantities serve as input
    to the equalization algorithm that is implemented by the callable ``equalizer``.
    This block computes soft-symbol estimates together with effective noise
    variances for all streams which can, e.g., be used by a
    :class:`~sionna.phy.mapping.Demapper` to obtain LLRs.

    Note
    -----
    The callable ``equalizer`` must take three inputs:

    * **y** ([...,num_rx_ant], tf.complex) -- 1+D tensor containing the received signals.
    * **h** ([...,num_rx_ant,num_streams_per_rx], tf.complex) -- 2+D tensor containing the channel matrices.
    * **s** ([...,num_rx_ant,num_rx_ant], tf.complex) -- 2+D tensor containing the noise-plus-interference covariance matrices.

    It must generate two outputs:

    * **x_hat** ([...,num_streams_per_rx], tf.complex) -- 1+D tensor representing the estimated symbol vectors.
    * **no_eff** (tf.float) -- Tensor of the same shape as ``x_hat`` containing the effective noise variance estimates.

    Parameters
    ----------
    equalizer : `Callable`
        Callable object (e.g., a function) that implements a MIMO equalization
        algorithm for arbitrary batch dimensions

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
    y : [batch_size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size], `tf.complex`
        Received OFDM resource grid after cyclic prefix removal and FFT

    h_hat : [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols, num_effective_subcarriers], `tf.complex`
        Channel estimates for all streams from all transmitters

    err_var : [Broadcastable to shape of ``h_hat``], `tf.float`
        Variance of the channel estimation error

    no : [batch_size, num_rx, num_rx_ant] (or only the first n dims), `tf.float`
        Variance of the AWGN

    Output
    ------
    x_hat : [batch_size, num_tx, num_streams, num_data_symbols], `tf.complex`
        Estimated symbols

    no_eff : [batch_size, num_tx, num_streams, num_data_symbols], `tf.float`
        Effective noise variance for each estimated symbol
    """
    def __init__(self,
                 equalizer,
                 resource_grid,
                 stream_management,
                 precision=None,
                 **kwargs):
        super().__init__(precision=precision, **kwargs)
        assert callable(equalizer)
        assert isinstance(resource_grid, sionna.phy.ofdm.ResourceGrid)
        assert isinstance(stream_management, sionna.phy.mimo.StreamManagement)
        self._equalizer = equalizer
        self._resource_grid = resource_grid
        self._stream_management = stream_management
        self._removed_nulled_scs = RemoveNulledSubcarriers(self._resource_grid)

        # Precompute indices to extract data symbols
        mask = resource_grid.pilot_pattern.mask
        num_data_symbols = resource_grid.pilot_pattern.num_data_symbols
        data_ind = tf.argsort(flatten_last_dims(mask), direction="ASCENDING")
        self._data_ind = data_ind[...,:num_data_symbols]

    def call(self, y, h_hat, err_var, no):

        # y has shape:
        # [batch_size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size]

        # h_hat has shape:
        # [batch_size, num_rx, num_rx_ant, num_tx, num_streams,...
        #  ..., num_ofdm_symbols, num_effective_subcarriers]

        # err_var has a shape that is broadcastable to h_hat

        # no has shape [batch_size, num_rx, num_rx_ant]
        # or just the first n dimensions of this

        # Remove nulled subcarriers from y (guards, dc). New shape:
        # [batch_size, num_rx, num_rx_ant, ...
        #  ..., num_ofdm_symbols, num_effective_subcarriers]
        y_eff = self._removed_nulled_scs(y)

        ####################################################
        ### Prepare the observation y for MIMO detection ###
        ####################################################
        # Transpose y_eff to put num_rx_ant last. New shape:
        # [batch_size, num_rx, num_ofdm_symbols,...
        #  ..., num_effective_subcarriers, num_rx_ant]
        y_dt = tf.transpose(y_eff, [0, 1, 3, 4, 2])
        y_dt = tf.cast(y_dt, self.cdtype)

        ##############################################
        ### Prepare the err_var for MIMO detection ###
        ##############################################
        # New shape is:
        # [batch_size, num_rx, num_ofdm_symbols,...
        #  ..., num_effective_subcarriers, num_rx_ant, num_tx*num_streams]
        err_var_dt = tf.broadcast_to(err_var, tf.shape(h_hat))
        err_var_dt = tf.transpose(err_var_dt, [0, 1, 5, 6, 2, 3, 4])
        err_var_dt = flatten_last_dims(err_var_dt, 2)
        err_var_dt = tf.cast(err_var_dt, self.cdtype)

        ###############################
        ### Construct MIMO channels ###
        ###############################

        # Reshape h_hat for the construction of desired/interfering channels:
        # [num_rx, num_tx, num_streams_per_tx, batch_size, num_rx_ant, ,...
        #  ..., num_ofdm_symbols, num_effective_subcarriers]
        perm = [1, 3, 4, 0, 2, 5, 6]
        h_dt = tf.transpose(h_hat, perm)

        # Flatten first tthree dimensions:
        # [num_rx*num_tx*num_streams_per_tx, batch_size, num_rx_ant, ...
        #  ..., num_ofdm_symbols, num_effective_subcarriers]
        h_dt = flatten_dims(h_dt, 3, 0)

        # Gather desired and undesired channels
        ind_desired = self._stream_management.detection_desired_ind
        ind_undesired = self._stream_management.detection_undesired_ind
        h_dt_desired = tf.gather(h_dt, ind_desired, axis=0)
        h_dt_undesired = tf.gather(h_dt, ind_undesired, axis=0)

        # Split first dimension to separate RX and TX:
        # [num_rx, num_streams_per_rx, batch_size, num_rx_ant, ...
        #  ..., num_ofdm_symbols, num_effective_subcarriers]
        h_dt_desired = split_dim(h_dt_desired,
                                 [self._stream_management.num_rx,
                                  self._stream_management.num_streams_per_rx],
                                 0)
        h_dt_undesired = split_dim(h_dt_undesired,
                                   [self._stream_management.num_rx, -1], 0)

        # Permutate dims to
        # [batch_size, num_rx, num_ofdm_symbols, num_effective_subcarriers,..
        #  ..., num_rx_ant, num_streams_per_rx(num_Interfering_streams_per_rx)]
        perm = [2, 0, 4, 5, 3, 1]
        h_dt_desired = tf.transpose(h_dt_desired, perm)
        h_dt_desired = tf.cast(h_dt_desired, self.cdtype)
        h_dt_undesired = tf.transpose(h_dt_undesired, perm)

        ##################################
        ### Prepare the noise variance ###
        ##################################
        # no is first broadcast to [batch_size, num_rx, num_rx_ant]
        # then the rank is expanded to that of y
        # then it is transposed like y to the final shape
        # [batch_size, num_rx, num_ofdm_symbols,...
        #  ..., num_effective_subcarriers, num_rx_ant]
        no_dt = expand_to_rank(no, 3, -1)
        no_dt = tf.broadcast_to(no_dt, tf.shape(y)[:3])
        no_dt = expand_to_rank(no_dt, tf.rank(y), -1)
        no_dt = tf.transpose(no_dt, [0,1,3,4,2])
        no_dt = tf.cast(no_dt, self.cdtype)

        ##################################################
        ### Compute the interference covariance matrix ###
        ##################################################
        # Covariance of undesired transmitters
        s_inf = tf.matmul(h_dt_undesired, h_dt_undesired, adjoint_b=True)

        #Thermal noise
        s_no = tf.linalg.diag(no_dt)

        # Channel estimation errors
        # As we have only error variance information for each element,
        # we simply sum them across transmitters and build a
        # diagonal covariance matrix from this
        s_csi = tf.linalg.diag(tf.reduce_sum(err_var_dt, -1))

        # Final covariance matrix
        s = s_inf + s_no + s_csi
        s = tf.cast(s, self.cdtype)

        ############################################################
        ### Compute symbol estimate and effective noise variance ###
        ############################################################
        # [batch_size, num_rx, num_ofdm_symbols, num_effective_subcarriers,...
        #  ..., num_stream_per_rx]
        x_hat, no_eff = self._equalizer(y_dt, h_dt_desired, s)

        ################################################
        ### Extract data symbols for all detected TX ###
        ################################################
        # Transpose tensor to shape
        # [num_rx, num_streams_per_rx, num_ofdm_symbols,...
        #  ..., num_effective_subcarriers, batch_size]
        x_hat = tf.transpose(x_hat, [1, 4, 2, 3, 0])
        no_eff = tf.transpose(no_eff, [1, 4, 2, 3, 0])

        # Merge num_rx amd num_streams_per_rx
        # [num_rx * num_streams_per_rx, num_ofdm_symbols,...
        #  ...,num_effective_subcarriers, batch_size]
        x_hat = flatten_dims(x_hat, 2, 0)
        no_eff = flatten_dims(no_eff, 2, 0)

        # Put first dimension into the right ordering
        stream_ind = self._stream_management.stream_ind
        x_hat = tf.gather(x_hat, stream_ind, axis=0)
        no_eff = tf.gather(no_eff, stream_ind, axis=0)

        # Reshape first dimensions to [num_tx, num_streams] so that
        # we can compared to the way the streams were created.
        # [num_tx, num_streams, num_ofdm_symbols, num_effective_subcarriers,...
        #  ..., batch_size]
        num_streams = self._stream_management.num_streams_per_tx
        num_tx = self._stream_management.num_tx
        x_hat = split_dim(x_hat, [num_tx, num_streams], 0)
        no_eff = split_dim(no_eff, [num_tx, num_streams], 0)

        # Flatten resource grid dimensions
        # [num_tx, num_streams, num_ofdm_symbols*num_effective_subcarriers,...
        #  ..., batch_size]
        x_hat = flatten_dims(x_hat, 2, 2)
        no_eff = flatten_dims(no_eff, 2, 2)

        # Broadcast no_eff to the shape of x_hat
        no_eff = tf.broadcast_to(no_eff, tf.shape(x_hat))

        # Gather data symbols
        # [num_tx, num_streams, num_data_symbols, batch_size]
        x_hat = tf.gather(x_hat, self._data_ind, batch_dims=2, axis=2)
        no_eff = tf.gather(no_eff, self._data_ind, batch_dims=2, axis=2)

        # Put batch_dim first
        # [batch_size, num_tx, num_streams, num_data_symbols]
        x_hat = tf.transpose(x_hat, [3, 0, 1, 2])
        no_eff = tf.transpose(no_eff, [3, 0, 1, 2])

        return x_hat, no_eff

class LMMSEEqualizer(OFDMEqualizer):
    # pylint: disable=line-too-long
    """
    LMMSE equalization for OFDM MIMO transmissions

    This block computes linear minimum mean squared error (LMMSE) equalization
    for OFDM MIMO transmissions. The OFDM and stream configuration are provided
    by a :class:`~sionna.phy.ofdm.ResourceGrid` and
    :class:`~sionna.phy.mimo.StreamManagement` instance, respectively. The
    detection algorithm is the :meth:`~sionna.phy.mimo.lmmse_equalizer`. The block
    computes soft-symbol estimates together with effective noise variances
    for all streams which can, e.g., be used by a
    :class:`~sionna.phy.mapping.Demapper` to obtain LLRs.

    Parameters
    ----------
    resource_grid : :class:`~sionna.phy.ofdm.ResourceGrid`
        ResourceGrid to be used

    stream_management : :class:`~sionna.phy.mimo.StreamManagement`
        StreamManagement to be used

    whiten_interference : `bool`, (default `True`)
        If `True`, the interference is first whitened before equalization.
        In this case, an alternative expression for the receive filter is used which
        can be numerically more stable.

    precision : `None` (default) | "single" | "double"
        Precision used for internal calculations and outputs.
        If set to `None`,
        :attr:`~sionna.phy.config.Config.precision` is used.

    Input
    -----
    y : [batch_size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size], `tf.complex`
        Received OFDM resource grid after cyclic prefix removal and FFT

    h_hat : [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols, num_effective_subcarriers], `tf.complex`
        Channel estimates for all streams from all transmitters

    err_var : [Broadcastable to shape of ``h_hat``], `tf.float`
        Variance of the channel estimation error

    no : [batch_size, num_rx, num_rx_ant] (or only the first n dims), `tf.float`
        Variance of the AWGN

    Output
    ------
    x_hat : [batch_size, num_tx, num_streams, num_data_symbols], `tf.complex`
        Estimated symbols

    no_eff : [batch_size, num_tx, num_streams, num_data_symbols], `tf.float`
        Effective noise variance for each estimated symbol
    """
    def __init__(self,
                 resource_grid,
                 stream_management,
                 whiten_interference=True,
                 precision=None,
                 **kwargs):

        def equalizer(y, h, s):
            return lmmse_equalizer(y, h, s, whiten_interference)

        super().__init__(equalizer=equalizer,
                         resource_grid=resource_grid,
                         stream_management=stream_management,
                         precision=precision, **kwargs)

class ZFEqualizer(OFDMEqualizer):
    # pylint: disable=line-too-long
    """
    ZF equalization for OFDM MIMO transmissions

    This block computes zero-forcing (ZF) equalization
    for OFDM MIMO transmissions. The OFDM and stream configuration are provided
    by a :class:`~sionna.phy.ofdm.ResourceGrid` and
    :class:`~sionna.phy.mimo.StreamManagement` instance, respectively. The
    detection algorithm is the :meth:`~sionna.phy.mimo.zf_equalizer`. The block
    computes soft-symbol estimates together with effective noise variances
    for all streams which can, e.g., be used by a
    :class:`~sionna.phy.mapping.Demapper` to obtain LLRs.

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
    y : [batch_size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size], `tf.complex`
        Received OFDM resource grid after cyclic prefix removal and FFT

    h_hat : [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols, num_effective_subcarriers], `tf.complex`
        Channel estimates for all streams from all transmitters

    err_var : [Broadcastable to shape of ``h_hat``], `tf.float`
        Variance of the channel estimation error

    no : [batch_size, num_rx, num_rx_ant] (or only the first n dims), `tf.float`
        Variance of the AWGN

    Output
    ------
    x_hat : [batch_size, num_tx, num_streams, num_data_symbols], `tf.complex`
        Estimated symbols

    no_eff : [batch_size, num_tx, num_streams, num_data_symbols], `tf.float`
        Effective noise variance for each estimated symbol
    """
    def __init__(self,
                 resource_grid,
                 stream_management,
                 precision=None,
                 **kwargs):
        super().__init__(equalizer=zf_equalizer,
                         resource_grid=resource_grid,
                         stream_management=stream_management,
                         precision=precision, **kwargs)

class MFEqualizer(OFDMEqualizer):
    # pylint: disable=line-too-long
    """
    MF equalization for OFDM MIMO transmissions

    This block computes matched filter (MF) equalization
    for OFDM MIMO transmissions. The OFDM and stream configuration are provided
    by a :class:`~sionna.phy.ofdm.ResourceGrid` and
    :class:`~sionna.phy.mimo.StreamManagement` instance, respectively. The
    detection algorithm is the :meth:`~sionna.phy.mimo.mf_equalizer`. The block
    computes soft-symbol estimates together with effective noise variances
    for all streams which can, e.g., be used by a
    :class:`~sionna.phy.mapping.Demapper` to obtain LLRs.

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
    y : [batch_size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size], `tf.complex`
        Received OFDM resource grid after cyclic prefix removal and FFT

    h_hat : [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols, num_effective_subcarriers], `tf.complex`
        Channel estimates for all streams from all transmitters

    err_var : [Broadcastable to shape of ``h_hat``], `tf.float`
        Variance of the channel estimation error

    no : [batch_size, num_rx, num_rx_ant] (or only the first n dims), `tf.float`
        Variance of the AWGN

    Output
    ------
    x_hat : [batch_size, num_tx, num_streams, num_data_symbols], `tf.complex`
        Estimated symbols

    no_eff : [batch_size, num_tx, num_streams, num_data_symbols], `tf.float`
        Effective noise variance for each estimated symbol
    """
    def __init__(self,
                 resource_grid,
                 stream_management,
                 precision=None,
                 **kwargs):
        super().__init__(equalizer=mf_equalizer,
                         resource_grid=resource_grid,
                         stream_management=stream_management,
                         precision=precision, **kwargs)

class PostEqualizationSINR(Block):
    # pylint: disable=line-too-long
    r"""
    Abstract block that computes the SINR after equalization

    This function computes the post-equalization SINR for every transmitted
    stream from the :class:`~sionna.phy.ofdm.PrecodedChannel`.
    A stream goes from a specific transmitter to a specific
    receiver and is characterized by a precoding vector and an
    equalization vector.

    Every transmitter is equipped with `num_tx_ant` antennas and every receiver
    is equipped with `num_rx_ant` antennas. All transmitters send the same number
    of streams :math:`S`. A transmitter can allocate different power to different streams.

    Let
    :math:`\mathbf{H}_{i,j}\in\mathbb{C}^{\text{num_rx_ant}\times\text{num_tx_ant}}`
    be the complex channel matrix between receiver :math:`i` and transmitter
    :math:`j`. We denote by
    :math:`\mathbf{g}_{j_,s}\in\mathbb{C}^{\text{num_tx_ant}}` the precoding
    vector
    for stream :math:`s` sent by transmitter :math:`j`.
    Then, the received signal at receiver :math:`i` can be expressed as:

    .. math::
        \mathbf{y}_i = \sum_{j,s} \mathbf{H}_{i,j} \mathbf{g}_{j,s} \sqrt{p_{j,s}} x_{j,s} + \mathbf{n}_{i} 

    where :math:`x_{j,s}` and :math:`p_{j,s}` are the unit-power transmit symbol
    and associated transmission power for stream :math:`s`, respectively, and
    :math:`\mathbf{n}_{i}` is the additive noise, distributed as
    :math:`\mathcal{C}\mathcal{N}(0,\sigma^2 \mathbf{I})`.

    By stacking the precoding vectors into a matrix :math:`\mathbf{G}_j=\left[\mathbf{g}_{j,1}, \ldots, \mathbf{g}_{j,S}\right]`,
    and using the definition of the precoded channel :math:`\widetilde{\mathbf{H}}_{i,j}` in
    :eq:`effective_precoded_channel`, the received signal can be rewritten as:

    .. math::
        \mathbf{y}_i = \sum_j \widetilde{\mathbf{H}}_{i,j} \mathop{\text{diag}}(x_{j,1},...,x_{j,S}) + \mathbf{n}_{i}

    Next, let :math:`\mathbf{f}_{i,j,s} \in\mathbb{C}^{\text{num_rx_ant}}`
    be the equalization vector for stream :math:`s` of transmitter :math:`j`,
    applied by the intended receiver :math:`i`. Then, the useful signal power for stream :math:`s` of transmitter :math:`j` is:

    .. math::
        u_{i,j,s} = p_{j,s} \left| \mathbf{f}_{i,j,s}^\mathsf{H} \mathbf{H}_{i,j} \mathbf{g}_{j, s} \right|^2.

    We assume that the transmitted symbols :math:`x_{j,s}` are uncorrelated among each
    other. Then, the interference power for this stream can be written
    as: 

    .. math::
        v_{i,j,s} = \sum_{(j',s') \ne (j,s)} p_{j',s'} \left| \mathbf{f}_{i,j,s}^\mathsf{H} \mathbf{H}_{i,j'} \mathbf{g}_{j', s'} \right|^2.

    The post-equalization noise power can be expressed as:

    .. math::
        n_{i,j,s} = \sigma^2 \| \mathbf{f}_{i,j,s} \|^2.

    With these definitions, the SINR for this stream which is finally computed as:

    .. math::
        \mathrm{SINR}_{i,j,s} = \frac{u_{i,j,s}}{v_{i,j,s} + n_{i,j,s}}.

    Note, that the intended receiver :math:`i` for a particular stream
    :math:`(j,s)` is defined by the :class:`~sionna.phy.mimo.StreamManagement`
    object.


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
    h_eff : [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols, num_effective_subcarriers], `tf.complex`
        Effective channel after precoding as defined in :eq:`effective_precoded_channel`

    no : [batch_size, num_rx, num_rx_ant, num_ofdm_symbols, num_effective_subcarriers] (or only the first n dims), `tf.float`
        Noise variance

    h_eff_hat : `None` (default) | [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols, num_effective_subcarriers], `tf.complex`
        Estimated effective channel after precoding. If set to `None`,
        the actual channel realizations are used.

    Output
    ------
    sinr : [batch_size, num_ofdm_symbols, num_effective_subcarriers, num_rx, num_streams_per_rx], `tf.float`
        SINR after equalization
    """
    def __init__(self,
                 resource_grid,
                 stream_management,
                 precision=None,
                 **kwargs):
        super().__init__(precision=precision, **kwargs)
        self._resource_grid = resource_grid
        self._stream_management = stream_management

    def get_per_rx_channels(self, h_eff):
        # pylint: disable=line-too-long
        r""" Extract desired and undesired channels for each receiver

        Input
        -----
        h_eff : [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols, num_effective_subcarriers], `tf.complex`, `tf.complex`
            Effective precoded channel. Can be estimated or true.

        Output
        ------
        h_eff_desired : [batch_size, num_rx, num_ofdm_symbols, num_effective_subcarriers, num_rx_ant, num_streams_per_rx], `tf.complex`
            Desired effective channels

        h_eff_undesired : [batch_size, num_rx, num_ofdm_symbols, num_effective_subcarriers, num_rx_ant, num_interfering_streams_per_rx], `tf.complex`
            Undesired effective channels

        """
        # Reshape h_eff for the construction of desired/interfering channels:
        # [num_rx, num_tx, num_streams_per_tx, batch_size, num_rx_ant, ,...
        #  ..., num_ofdm_symbols, num_effective_subcarriers]
        perm = [1, 3, 4, 0, 2, 5, 6]
        h_eff = tf.transpose(h_eff, perm)

        # Flatten first three dimensions:
        # [num_rx*num_tx*num_streams_per_tx, batch_size, num_rx_ant, ...
        #  ..., num_ofdm_symbols, num_effective_subcarriers]
        h_eff = flatten_dims(h_eff, 3, 0)

        # Gather desired and undesired channels
        ind_desired = self._stream_management.detection_desired_ind
        ind_undesired = self._stream_management.detection_undesired_ind
        h_eff_desired = tf.gather(h_eff, ind_desired, axis=0)
        h_eff_undesired = tf.gather(h_eff, ind_undesired, axis=0)

        # Split first dimension to separate RX and TX:
        # [num_rx, num_streams_per_rx, batch_size, num_rx_ant, ...
        #  ..., num_ofdm_symbols, num_effective_subcarriers]
        h_eff_desired = split_dim(h_eff_desired,
                                 [self._stream_management.num_rx,
                                  self._stream_management.num_streams_per_rx],
                                 0)
        h_eff_undesired = split_dim(h_eff_undesired,
                                   [self._stream_management.num_rx, -1], 0)

        # Permutate dims to
        # [batch_size, num_rx, num_ofdm_symbols, num_effective_subcarriers,..
        #  ..., num_rx_ant, num_streams_per_rx(num_interfering_streams_per_rx)]
        perm = [2, 0, 4, 5, 3, 1]
        h_eff_desired = tf.transpose(h_eff_desired, perm)
        h_eff_undesired = tf.transpose(h_eff_undesired, perm)

        return h_eff_desired, h_eff_undesired

    def compute_interference_covariance_matrix(self, no=None, h_eff_undesired=None):
        # pylint: disable=line-too-long
        r"""Compute the interference covariance matrix

        Input
        -----
        no : `None` (default) | [batch_size, num_rx, num_ofdm_symbols, num_effective_subcarriers, num_rx_ant], `tf.float`
            Noise variance

        h_eff_undesired : `None` (default) | [batch_size, num_rx, num_ofdm_symbols, num_effective_subcarriers, num_rx_ant, num_interfering_streams_per_rx], `tf.complex`
            Undesired effective channels. If set to `None`, the actual channel realizations are used.

        Output
        ------
        s : [batch_size, num_rx, num_ofdm_symbols, num_effective_subcarriers, num_rx_ant, num_rx_ant], `tf.complex`
            Interference covariance matrix
        """
        s_no = 0.
        if no is not None:
            # Diagonal matrix
            no = tf.cast(no, self.cdtype)
            s_no = tf.linalg.diag(no)

        s_inf = 0.
        if h_eff_undesired is not None:
            s_inf = tf.matmul(h_eff_undesired, h_eff_undesired, adjoint_b=True)

        s = s_no + s_inf

        return s

    def compute_desired_signal_power(self, h_eff_desired, f):
        # pylint: disable=line-too-long
        r""" Compute the desired signal power

        Input
        -----
        h_eff_desired : [batch_size, num_rx, num_ofdm_symbols, num_effective_subcarriers, num_rx_ant, num_streams_per_rx], `tf.complex`
            Desired effective channels

        f : [batch_size, num_rx, num_ofdm_symbols, num_effective_subcarriers, num_streams_per_rx, num_rx_ant], `tf.complex`
            Receive combining vectors

        Output
        ------
        signal_power : [batch_size, num_rx, num_ofdm_symbols, num_effective_subcarriers, num_streams_per_rx], `tf.float`
            Desired signal power
        """
        signal_power = tf.einsum('...mn,...nm->...m', f, h_eff_desired)
        signal_power = tf.abs(signal_power)**2
        return signal_power

    def compute_total_power(self, h_eff_desired, h_eff_undesired, f):
        """
        Compute the total power from all transmitters

        Input
        -----
        h_eff_desired : [batch_size, num_rx, num_ofdm_symbols, num_effective_subcarriers, num_rx_ant, num_streams_per_rx], `tf.complex`
            Desired effective channels

        h_eff_undesired : [batch_size, num_rx, num_ofdm_symbols, num_effective_subcarriers, num_rx_ant, num_interfering_streams_per_rx], `tf.complex`
            Undesired effective channels

        f : [batch_size, num_rx, num_ofdm_symbols, num_effective_subcarriers,
        num_streams_per_rx, num_rx_ant], `tf.complex`
            Receive combining vectors

        Output
        ------
        total_power : [batch_size, num_rx, num_ofdm_symbols, num_effective_subcarriers, 1], `tf.float`
            Total power
        """
        h_eff = tf.concat([h_eff_desired, h_eff_undesired], axis=-1)
        total_power = tf.abs(tf.matmul(f, h_eff))**2
        total_power = tf.reduce_sum(total_power, axis=-1)
        return total_power

    def compute_noise_power(self, no, f):
        # pylint: disable=line-too-long
        r""" Compute the noise power

        Input
        -----
        no : [batch_size, num_rx, num_ofdm_symbols, num_effective_subcarriers, num_rx_ant], `tf.float`
            Noise variance

        f : [batch_size, num_rx, num_ofdm_symbols, num_effective_subcarriers, num_streams_per_rx, num_rx_ant], `tf.complex`
            Receive combining vectors
        """
        no = tf.expand_dims(tf.math.real(no), axis=-2)
        noise_power = tf.reduce_sum(tf.abs(f)**2 * no, axis=-1)
        return noise_power

    def compute_sinr(self, h_eff_desired, h_eff_undesired, no, f):
        # pylint: disable=line-too-long
        r""" Compute the SINR

        Input
        -----
        h_eff_desired : [batch_size, num_rx, num_ofdm_symbols, num_effective_subcarriers, num_rx_ant, num_streams_per_rx], `tf.complex`
            Desired effective channels

        h_eff_undesired : [batch_size, num_rx, num_ofdm_symbols, num_effective_subcarriers, num_rx_ant, num_interfering_streams_per_rx], `tf.complex`
            Undesired effective channels

        no : [batch_size, num_rx, num_ofdm_symbols, num_effective_subcarriers, num_rx_ant], `tf.float`
            Noise variance

        f : [batch_size, num_rx, num_ofdm_symbols, num_effective_subcarriers, num_rx_ant, num_streams_per_rx], `tf.complex`
            Equalization matrix

        Output
        ------
        sinr : [batch_size, num_ofdm_symbols, num_effective_subcarriers, num_rx, num_streams_per_rx], `tf.float`
            Post-equalization SINR
        """
        signal_power = self.compute_desired_signal_power(h_eff_desired, f)
        total_power = self.compute_total_power(h_eff_desired,h_eff_undesired,f)
        # For numerical stability, avoid negative values
        interference_power = tf.maximum(total_power - signal_power, tf.cast(0, self.rdtype))
        noise_power = self.compute_noise_power(no, f)
        sinr = tf.math.divide_no_nan(signal_power,
                                     interference_power + noise_power)

        # Reshape to desired dimensions
        sinr = tf.transpose(sinr, [0, 2, 3, 1, 4])
        return sinr

    @abstractmethod
    def call(self, h_eff, no, h_eff_hat=None):
        pass

class LMMSEPostEqualizationSINR(PostEqualizationSINR):
    # pylint: disable=line-too-long
    r"""
    Block that computes the SINR after LMMSE equalization

    The equalization matrix is the one computed by
    :meth:`~sionna.phy.mimo.lmmse_matrix`.

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
    h_eff : [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols, num_effective_subcarriers], `tf.complex`
        Effective channel after precoding as defined in :eq:`effective_precoded_channel`

    no : [batch_size, num_rx, num_rx_ant, num_ofdm_symbols, num_effective_subcarriers] (or only the first n dims), `tf.float`
        Noise variance

    h_eff_hat : `None` (default) | [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols, num_effective_subcarriers], `tf.complex`
        Estimated effective channel after precoding. If set to `None`,
        the actual channel realizations are used.

    interference_whitening : `bool` (default=True)
        If set to `True`, also the interference from undesired streams (e.g.,
        from other cells) is whitened

    Output
    ------
    sinr : [batch_size, num_ofdm_symbols, num_effective_subcarriers, num_rx, num_streams_per_rx], `tf.float`
        SINR after equalization
    """
    def call(self, h_eff, no, h_eff_hat=None, interference_whitening=True):
        if h_eff_hat is None:
            h_eff_hat = h_eff

        #  Ensure that noise variance has the right dimensions
        no = expand_to_rank(no, 5, -1)
        no = tf.broadcast_to(no, [tf.shape(h_eff)[0],
                                  h_eff.shape[1],
                                  h_eff.shape[2],
                                  h_eff.shape[5],
                                  h_eff.shape[6]])
        # [batch_size, num_rx, num_ofdm_symbols, num_effective_subcarriers,...
        #  ... num_rx_ant]
        no = tf.transpose(no, [0, 1, 3, 4, 2])

        # Get estimated desired and undesired channels
        h_eff_desired, h_eff_undesired = self.get_per_rx_channels(h_eff_hat)

        # Compute estimated interference covariance matrix
        if interference_whitening:
            s = self.compute_interference_covariance_matrix(
                                no=no,
                                h_eff_undesired=h_eff_undesired)
        else:
            s = self.compute_interference_covariance_matrix(
                    no=no)

        # Whiten channels
        l_inv = inv_cholesky(s) # Compute whitening matrix
        h_eff_desired = tf.matmul(l_inv, h_eff_desired)
        h_eff_undesired = tf.matmul(l_inv, h_eff_undesired)

        # Compute equalization matrix
        f = lmmse_matrix(h_eff_desired, precision=self.precision)

        # Compute SINR
        sinr = self.compute_sinr(h_eff_desired, h_eff_undesired,
                                 tf.ones_like(no), f)

        return sinr
