#
# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Class definition and functions related to OFDM channel equalization"""

import tensorflow as tf
import sionna as sn
from sionna.utils import flatten_dims, split_dim, flatten_last_dims, expand_to_rank
from sionna.ofdm import RemoveNulledSubcarriers

# pylint: disable=line-too-long
class MaximumLikelihoodDetectorWithPrior(sn.mimo.MaximumLikelihoodDetectorWithPrior):
    # pylint: disable=line-too-long
    r"""MaximumLikelihoodDetectorWithPrior(output, demapping_method, resource_grid, stream_management, constellation_type=None, num_bits_per_symbol=None, constellation=None, hard_out=False, dtype=tf.complex64, **kwargs)

    Maximum-likelihood (ML) detection for OFDM MIMO transmissions, assuming prior
    knowledge of the bits or constellation points is available.

    This layer implements maximum-likelihood (ML) detection
    for OFDM MIMO transmissions assuming prior knowledge on the transmitted data is available.
    Both ML detection of symbols or bits with either
    soft- or hard-decisions are supported. The OFDM and stream configuration are provided
    by a :class:`~sionna.ofdm.ResourceGrid` and
    :class:`~sionna.mimo.StreamManagement` instance, respectively. The
    actual detector is an instance of :class:`~sionna.mimo.MaximumLikelihoodDetectorWithPrior`.

    Parameters
    ----------
    output : One of ["bit", "symbol"], str
        The type of output, either LLRs on bits or logits on constellation symbols.

    demapping_method : One of ["app", "maxlog"], str
        The demapping method used.

    resource_grid : ResourceGrid
        An instance of :class:`~sionna.ofdm.ResourceGrid`.

    stream_management : StreamManagement
        An instance of :class:`~sionna.mimo.StreamManagement`.

    constellation_type : One of ["qam", "pam", "custom"], str
        For "custom", an instance of :class:`~sionna.mapping.Constellation`
        must be provided.

    num_bits_per_symbol : int
        The number of bits per constellation symbol, e.g., 4 for QAM16.
        Only required for ``constellation_type`` in ["qam", "pam"].

    constellation : Constellation
        An instance of :class:`~sionna.mapping.Constellation` or `None`.
        In the latter case, ``constellation_type``
        and ``num_bits_per_symbol`` must be provided.

    hard_out : bool
        If `True`, the detector computes hard-decided bit values or
        constellation point indices instead of soft-values.
        Defaults to `False`.

    dtype : One of [tf.complex64, tf.complex128] tf.DType (dtype)
        The dtype of `y`. Defaults to tf.complex64.
        The output dtype is the corresponding real dtype (tf.float32 or tf.float64).

    Input
    ------
    (y, h_hat, prior, err_var, no) :
        Tuple:

    y : [batch_size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size], tf.complex
        The received OFDM resource grid after cyclic prefix removal and FFT.

    h_hat : [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols, num_effective_subcarriers], tf.complex
        The channel estimates for all streams from all transmitters.

    prior : [batch_size, num_tx, num_streams, num_data_symbols x num_bits_per_symbol] or [batch_size, num_tx, num_streams, num_data_symbols, num_points], tf.float
        Prior of the transmitted signals.
        If ``output`` equals "bit", LLRs of the transmitted bits are expected.
        If ``output`` equals "symbol", logits of the transmitted constellation points are expected.

    err_var : [Broadcastable to shape of ``h_hat``], tf.float
        The variance of the channel estimation error.

    no : [batch_size, num_rx, num_rx_ant] (or only the first n dims), tf.float
        The variance of the AWGN noise.

    Output
    ------
    One of:

    : [batch_size, num_tx, num_streams, num_data_symbols*num_bits_per_symbol], tf.float
        LLRs or hard-decisions for every bit of every stream, if ``output`` equals `"bit"`.

    : [batch_size, num_tx, num_streams, num_data_symbols, num_points], tf.float or [batch_size, num_tx, num_streams, num_data_symbols], tf.int
        Logits or hard-decisions for constellation symbols for every stream, if ``output`` equals `"symbol"`.
        Hard-decisions correspond to the symbol indices.

    Note
    ----
    If you want to use this layer in Graph mode with XLA, i.e., within
    a function that is decorated with ``@tf.function(jit_compile=True)``,
    you must set ``sionna.Config.xla_compat=true``.
    See :py:attr:`~sionna.Config.xla_compat`.
    """

    def __init__(self,
                 output,
                 demapping_method,
                 resource_grid,
                 stream_management,
                 constellation_type=None,
                 num_bits_per_symbol=None,
                 constellation=None,
                 hard_out=False,
                 dtype=tf.complex64,
                 **kwargs):

        self._resource_grid = resource_grid
        self._stream_management = stream_management
        self._removed_nulled_scs = RemoveNulledSubcarriers(self._resource_grid)
        self._output = output

        # Precompute indices to extract data symbols
        mask = resource_grid.pilot_pattern.mask
        num_data_symbols = resource_grid.pilot_pattern.num_data_symbols
        data_ind = tf.argsort(flatten_last_dims(mask), direction="ASCENDING")
        self._data_ind = data_ind[...,:num_data_symbols]

        # Precompute indices to map priors to a resource grid
        rg_type = resource_grid.build_type_grid()
        self._data_ind_scatter = tf.where(rg_type==0)

        # Initializing maximum-likelihood baseclass
        super().__init__(output=output,
                         demapping_method=demapping_method,
                         k = stream_management.num_streams_per_rx,
                         constellation_type=constellation_type,
                         num_bits_per_symbol=num_bits_per_symbol,
                         constellation=constellation,
                         hard_out=hard_out,
                         dtype=dtype,
                         **kwargs)

    def call(self, inputs):
        y, h_hat, prior, err_var, no = inputs
        # y has shape:
        # [batch_size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size]

        # h_hat has shape:
        # [batch_size, num_rx, num_rx_ant, num_tx, num_streams,...
        #  ..., num_ofdm_symbols, num_effective_subcarriers]

        # prior has shape
        #  [batch_size, num_tx, num_streams,...
        #   ... num_data_symbols x num_bits_per_symbol]
        #       if ``output`` equals "bit"
        #  [batch_size, num_tx, num_streams, num_data_symbols, num_points]
        #       if ``output`` equals "symbol"

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
        y_dt = tf.cast(y_dt, self._dtype)

        ##############################################
        ### Prepare the err_var for MIMO detection ###
        ##############################################
        # New shape is:
        # [batch_size, num_rx, num_ofdm_symbols,...
        #  ..., num_effective_subcarriers, num_rx_ant, num_tx*num_streams]
        err_var_dt = tf.broadcast_to(err_var, tf.shape(h_hat))
        err_var_dt = tf.transpose(err_var_dt, [0, 1, 5, 6, 2, 3, 4])
        err_var_dt = flatten_last_dims(err_var_dt, 2)
        err_var_dt = tf.cast(err_var_dt, self._dtype)

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
                                        [self._stream_management.num_rx, -1], 0)
        h_dt_undesired = split_dim(h_dt_undesired,
                                        [self._stream_management.num_rx, -1], 0)

        # Permutate dims to
        # [batch_size, num_rx, num_ofdm_symbols, num_effective_subcarriers,..
        #  ..., num_rx_ant, num_streams_per_rx(num_Interfering_streams_per_rx)]
        perm = [2, 0, 4, 5, 3, 1]
        h_dt_desired = tf.transpose(h_dt_desired, perm)
        h_dt_desired = tf.cast(h_dt_desired, self._dtype)
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
        no_dt = tf.cast(no_dt, self._dtype)

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
        s = tf.cast(s, self._dtype)

        #########################
        ### Prepare the prior ###
        #########################
        # [batch_size, num_tx, num_streams_per_tx, num_data_symbols,
        #   ... num_bits_per_symbol/num_points]
        if self._output == 'bit':
            prior = split_dim(  prior,
                                [   self._resource_grid.num_data_symbols,
                                    self._constellation.num_bits_per_symbol],
                                3)
        # Create a zero template for the prior
        # [num_tx, num_streams_per_tx, num_ofdm_symbols,...
        #   ... num_effective_subcarriers, num_bits_per_symbol/num_points,
        #   ... batch_size]
        template = tf.zeros([   self._resource_grid.num_tx,
                                self._resource_grid.num_streams_per_tx,
                                self._resource_grid.num_ofdm_symbols,
                                self._resource_grid.num_effective_subcarriers,
                                tf.shape(prior)[-1],
                                tf.shape(prior)[0]],
                            tf.as_dtype(self._dtype).real_dtype)
        # [num_tx, num_streams_per_tx, num_data_symbols,
        #   ... num_bits_per_symbol/num_points, batch_size]
        prior = tf.transpose(prior, [1, 2, 3, 4, 0])
        # [num_tx, num_streams_per_tx, num_ofdm_symbols,...
        #   ... num_effective_subcarriers, num_bits_per_symbol/num_points,...
        #   ... batch_size]
        prior = flatten_dims(prior, 3, 0)
        prior = tf.tensor_scatter_nd_update(template, self._data_ind_scatter,
                                                prior)
        # [batch_size, num_ofdm_symbols, num_effective_subcarriers,...
        #  num_tx*num_streams_per_tx, num_bits_per_symbol/num_points]
        prior = tf.transpose(prior, [5, 2, 3, 0, 1, 4])
        prior = flatten_dims(prior, 2, 3)
        # Add the receive antenna dimension for broadcasting
        # [batch_size, num_rx, num_ofdm_symbols, num_effective_subcarriers,...
        #  num_tx*num_streams_per_tx, num_bits_per_symbol/num_points]
        prior = tf.tile(tf.expand_dims(prior, axis=1),
                        [1, tf.shape(y_dt)[1], 1, 1, 1, 1])

        #################################
        ### Maximum-likelihood detection
        #################################
        z = super().call([y_dt,h_dt_desired,prior,s])

        ##############################################
        ### Extract data symbols for all detected TX
        ##############################################
        # Transpose tensor to shape
        # [num_rx, num_streams_per_rx, num_ofdm_symbols,
        #    num_effective_subcarriers, num_bits_per_symbol or num_points,
        #       batch_size]
        z = tf.transpose(z, [1, 4, 2, 3, 5, 0])

        # Merge num_rx amd num_streams_per_rx
        # [num_rx * num_streams_per_rx, num_ofdm_symbols,
        #    num_effective_subcarriers, num_bits_per_symbol or num_points,
        #   batch_size]
        z = flatten_dims(z, 2, 0)

        # Put first dimension into the right ordering
        stream_ind = self._stream_management.stream_ind
        z = tf.gather(z, stream_ind, axis=0)

        # Reshape first dimensions to [num_tx, num_streams] so that
        # we can compare to the way the streams were created.
        # [num_tx, num_streams, num_ofdm_symbols, num_effective_subcarriers,
        #     num_bits_per_symbol or num_points, batch_size]
        num_streams = self._stream_management.num_streams_per_tx
        num_tx = self._stream_management.num_tx
        z = split_dim(z, [num_tx, num_streams], 0)

        # Flatten resource grid dimensions
        # [num_tx, num_streams, num_ofdm_symbols*num_effective_subcarrier,
        #    num_bits_per_symbol or num_points, batch_size]
        z = flatten_dims(z, 2, 2)

        # Gather data symbols
        # [num_tx, num_streams, num_data_symbols,
        #    num_bits_per_symbol or num_points, batch_size]
        z = tf.gather(z, self._data_ind, batch_dims=2, axis=2)

        # Put batch_dim first
        # [batch_size, num_tx, num_streams,
        #     num_data_symbols, num_bits_per_symbol or num_points]
        z = tf.transpose(z, [4, 0, 1, 2, 3])

        # Reshape LLRs to
        # [batch_size, num_tx, num_streams,
        #     n = num_data_symbols*num_bits_per_symbol]
        # if output is LLRs on bits
        if self._output == 'bit':
            z = flatten_dims(z, 2, 3)

        return z

class MaximumLikelihoodDetector(MaximumLikelihoodDetectorWithPrior):
    # pylint: disable=line-too-long
    r"""MaximumLikelihoodDetector(output, demapping_method, resource_grid, stream_management, constellation_type=None, num_bits_per_symbol=None, constellation=None, hard_out=False, dtype=tf.complex64, **kwargs)

    Maximum-likelihood (ML) detection for OFDM MIMO transmissions.

    This layer implements maximum-likelihood (ML) detection
    for OFDM MIMO transmissions. Both ML detection of symbols or bits with either
    soft- or hard-decisions are supported. The OFDM and stream configuration are provided
    by a :class:`~sionna.ofdm.ResourceGrid` and
    :class:`~sionna.mimo.StreamManagement` instance, respectively. The
    actual detector is an instance of :class:`~sionna.mimo.MaximumLikelihoodDetector`.

    Parameters
    ----------
    output : One of ["bit", "symbol"], str
        The type of output, either LLRs on bits or logits on constellation symbols.

    demapping_method : One of ["app", "maxlog"], str
        The demapping method used.

    resource_grid : ResourceGrid
        An instance of :class:`~sionna.ofdm.ResourceGrid`.

    stream_management : StreamManagement
        An instance of :class:`~sionna.mimo.StreamManagement`.

    constellation_type : One of ["qam", "pam", "custom"], str
        For "custom", an instance of :class:`~sionna.mapping.Constellation`
        must be provided.

    num_bits_per_symbol : int
        The number of bits per constellation symbol, e.g., 4 for QAM16.
        Only required for ``constellation_type`` in ["qam", "pam"].

    constellation : Constellation
        An instance of :class:`~sionna.mapping.Constellation` or `None`.
        In the latter case, ``constellation_type``
        and ``num_bits_per_symbol`` must be provided.

    hard_out : bool
        If `True`, the detector computes hard-decided bit values or
        constellation point indices instead of soft-values.
        Defaults to `False`.

    dtype : One of [tf.complex64, tf.complex128] tf.DType (dtype)
        The dtype of `y`. Defaults to tf.complex64.
        The output dtype is the corresponding real dtype (tf.float32 or tf.float64).

    Input
    ------
    (y, h_hat, err_var, no) :
        Tuple:

    y : [batch_size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size], tf.complex
        The received OFDM resource grid after cyclic prefix removal and FFT.

    h_hat : [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols, num_effective_subcarriers], tf.complex
        The channel estimates for all streams from all transmitters.

    err_var : [Broadcastable to shape of ``h_hat``], tf.float
        The variance of the channel estimation error.

    no : [batch_size, num_rx, num_rx_ant] (or only the first n dims), tf.float
        The variance of the AWGN noise.

    Output
    ------
    One of:

    : [batch_size, num_tx, num_streams, num_data_symbols*num_bits_per_symbol], tf.float
        LLRs or hard-decisions for every bit of every stream, if ``output`` equals `"bit"`.

    : [batch_size, num_tx, num_streams, num_data_symbols, num_points], tf.float or [batch_size, num_tx, num_streams, num_data_symbols], tf.int
        Logits or hard-decisions for constellation symbols for every stream, if ``output`` equals `"symbol"`.
        Hard-decisions correspond to the symbol indices.

    Note
    ----
    If you want to use this layer in Graph mode with XLA, i.e., within
    a function that is decorated with ``@tf.function(jit_compile=True)``,
    you must set ``sionna.Config.xla_compat=true``.
    See :py:attr:`~sionna.Config.xla_compat`.
    """

    def __init__(self,
                 output,
                 demapping_method,
                 resource_grid,
                 stream_management,
                 constellation_type=None,
                 num_bits_per_symbol=None,
                 constellation=None,
                 hard_out=False,
                 dtype=tf.complex64,
                 **kwargs):

        super().__init__(output=output,
                         demapping_method=demapping_method,
                         resource_grid=resource_grid,
                         stream_management=stream_management,
                         constellation_type=constellation_type,
                         num_bits_per_symbol=num_bits_per_symbol,
                         constellation=constellation,
                         hard_out=hard_out,
                         dtype=dtype,
                         **kwargs)

    def call(self, inputs):
        y, h_hat, err_var, no = inputs

        batch_size = tf.shape(y)[0]
        num_data_symbols = self._resource_grid.num_data_symbols
        num_bits_per_symbol = self._constellation.num_bits_per_symbol
        num_points = 2**num_bits_per_symbol
        real_dtype = tf.as_dtype(self._dtype).real_dtype

        # Prior shape
        if self._output == "bit":
            dim_data = [num_data_symbols*num_bits_per_symbol]
        else:
            dim_data = [num_data_symbols, num_points]
        prior_shape = tf.concat([[  batch_size,
                                    self._resource_grid.num_tx,
                                    self._resource_grid.num_streams_per_tx],
                                    dim_data], axis=0)

        # Build null-prior
        prior = tf.zeros(prior_shape, real_dtype)

        # Call
        return super().call([y, h_hat, prior, err_var, no])
