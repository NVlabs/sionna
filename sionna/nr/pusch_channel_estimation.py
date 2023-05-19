#
# SPDX-FileCopyrightText: Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""PUSCH Channel Estimation for the nr (5G) sub-package of the Sionna library.
"""
import tensorflow as tf
from tensorflow.keras.layers import Layer
from sionna.ofdm import LSChannelEstimator
from sionna.utils import expand_to_rank, split_dim

class PUSCHLSChannelEstimator(LSChannelEstimator, Layer):
    # pylint: disable=line-too-long
    r"""LSChannelEstimator(resource_grid, dmrs_length, dmrs_additional_position, num_cdm_groups_without_data, interpolation_type="nn", interpolator=None, dtype=tf.complex64, **kwargs)

    Layer implementing least-squares (LS) channel estimation for NR PUSCH Transmissions.

    After LS channel estimation at the pilot positions, the channel estimates
    and error variances are interpolated accross the entire resource grid using
    a specified interpolation function.

    The implementation is similar to that of :class:`~sionna.ofdm.LSChannelEstimator`.
    However, it additional takes into account the separation of streams in the same CDM group
    as defined in :class:`~sionna.nr.PUSCHDMRSConfig`. This is done through
    frequency and time averaging of adjacent LS channel estimates.

    Parameters
    ----------
    resource_grid : ResourceGrid
        An instance of :class:`~sionna.ofdm.ResourceGrid`

    dmrs_length : int, [1,2]
        Length of DMRS symbols. See :class:`~sionna.nr.PUSCHDMRSConfig`.

    dmrs_additional_position : int, [0,1,2,3]
        Number of additional DMRS symbols.
        See :class:`~sionna.nr.PUSCHDMRSConfig`.

    num_cdm_groups_without_data : int, [1,2,3]
        Number of CDM groups masked for data transmissions.
        See :class:`~sionna.nr.PUSCHDMRSConfig`.

    interpolation_type : One of ["nn", "lin", "lin_time_avg"], string
        The interpolation method to be used.
        It is ignored if ``interpolator`` is not `None`.
        Available options are :class:`~sionna.ofdm.NearestNeighborInterpolator` (`"nn`")
        or :class:`~sionna.ofdm.LinearInterpolator` without (`"lin"`) or with
        averaging across OFDM symbols (`"lin_time_avg"`).
        Defaults to "nn".

    interpolator : BaseChannelInterpolator
        An instance of :class:`~sionna.ofdm.BaseChannelInterpolator`,
        such as :class:`~sionna.ofdm.LMMSEInterpolator`,
        or `None`. In the latter case, the interpolator specified
        by ``interpolation_type`` is used.
        Otherwise, the ``interpolator`` is used and ``interpolation_type``
        is ignored.
        Defaults to `None`.

    dtype : tf.Dtype
        Datatype for internal calculations and the output dtype.
        Defaults to `tf.complex64`.

    Input
    -----
    (y, no) :
        Tuple:

    y : [batch_size, num_rx, num_rx_ant, num_ofdm_symbols,fft_size], tf.complex
        Observed resource grid

    no : [batch_size, num_rx, num_rx_ant] or only the first n>=0 dims, tf.float
        Variance of the AWGN

    Output
    ------
    h_ls : [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols,fft_size], tf.complex
        Channel estimates across the entire resource grid for all
        transmitters and streams

    err_var : Same shape as ``h_ls``, tf.float
        Channel estimation error variance across the entire resource grid
        for all transmitters and streams
    """
    def __init__(self,
                 resource_grid,
                 dmrs_length,
                 dmrs_additional_position,
                 num_cdm_groups_without_data,
                 interpolation_type="nn",
                 interpolator=None,
                 dtype=tf.complex64,
                 **kwargs):
        super().__init__(resource_grid,
                         interpolation_type,
                         interpolator,
                         dtype, **kwargs)

        self._dmrs_length = dmrs_length
        self._dmrs_additional_position = dmrs_additional_position
        self._num_cdm_groups_without_data = num_cdm_groups_without_data

        # Number of DMRS OFDM symbols
        self._num_dmrs_syms = self._dmrs_length \
                              * (self._dmrs_additional_position+1)

        # Number of pilot symbols per DMRS OFDM symbol
        # Some pilot symbols can be zero (for masking)
        self._num_pilots_per_dmrs_sym = int(
                    self._pilot_pattern.pilots.shape[-1]/self._num_dmrs_syms)

    def estimate_at_pilot_locations(self, y_pilots, no):
        # y_pilots : [batch_size, num_rx, num_rx_ant, num_tx, num_streams,
        #               num_pilot_symbols], tf.complex
        #     The observed signals for the pilot-carrying resource elements.

        # no : [batch_size, num_rx, num_rx_ant] or only the first n>=0 dims,
        #   tf.float
        #     The variance of the AWGN.

        # Compute LS channel estimates
        # Note: Some might be Inf because pilots=0, but we do not care
        # as only the valid estimates will be considered during interpolation.
        # We do a save division to replace Inf by 0.
        # Broadcasting from pilots here is automatic since pilots have shape
        # [num_tx, num_streams, num_pilot_symbols]
        h_ls = tf.math.divide_no_nan(y_pilots, self._pilot_pattern.pilots)
        h_ls_shape = tf.shape(h_ls)

        # Compute error variance and broadcast to the same shape as h_ls
        # Expand rank of no for broadcasting
        no = expand_to_rank(no, tf.rank(h_ls), -1)

        # Expand rank of pilots for broadcasting
        pilots = expand_to_rank(self._pilot_pattern.pilots, tf.rank(h_ls), 0)

        # Compute error variance, broadcastable to the shape of h_ls
        err_var = tf.math.divide_no_nan(no, tf.abs(pilots)**2)

        # In order to deal with CDM, we need to do (optional) time and
        # frequency averaging of the LS estimates
        h_hat = h_ls

        # (Optional) Time-averaging across adjacent DMRS OFDM symbols
        if self._dmrs_length==2:
            # Reshape last dim to [num_dmrs_syms, num_pilots_per_dmrs_sym]
            h_hat = split_dim(h_hat, [self._num_dmrs_syms,
                                      self._num_pilots_per_dmrs_sym], 5)

            # Average adjacent DMRS symbols in time domain
            h_hat = (h_hat[...,0::2,:]+h_hat[...,1::2,:]) \
                     / tf.cast(2, h_hat.dtype)
            h_hat = tf.repeat(h_hat, 2, axis=-2)
            h_hat = tf.reshape(h_hat, h_ls_shape)

            # The error variance gets reduced by a factor of two
            err_var /= tf.cast(2, err_var.dtype)

        # Frequency-averaging between adjacent channel estimates

        # Compute number of elements across which frequency averaging should
        # be done. This includes the zeroed elements.
        n = 2*self._num_cdm_groups_without_data
        k = int(h_hat.shape[-1]/n) # Second dimension

        # Reshape last dimension to [k, n]
        h_hat = split_dim(h_hat, [k, n], 5)
        cond = tf.abs(h_hat)>0 # Mask for irrelevant channel estimates
        h_hat = tf.reduce_sum(h_hat, axis=-1, keepdims=True) \
                / tf.cast(2,h_hat.dtype)
        h_hat = tf.repeat(h_hat, n, axis=-1)
        h_hat = tf.where(cond, h_hat, 0) # Mask irrelevant channel estimates
        h_hat = tf.reshape(h_hat, h_ls_shape)

        # The error variance gets reduced by a factor of two
        err_var /= tf.cast(2, err_var.dtype)

        return h_hat, err_var
