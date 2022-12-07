#
# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Functions related to OFDM channel estimation"""

import tensorflow as tf
from tensorflow.keras.layers import Layer
import numpy as np
from sionna.channel.tr38901 import models
from sionna.utils import flatten_last_dims, expand_to_rank, matrix_inv
from sionna.ofdm import ResourceGrid, RemoveNulledSubcarriers
from sionna import PI, SPEED_OF_LIGHT
from scipy.special import jv
import itertools
from abc import ABC, abstractmethod
import json
from importlib_resources import files

class BaseChannelEstimator(ABC, Layer):
    # pylint: disable=line-too-long
    r"""BaseChannelEstimator(resource_grid, interpolation_type="nn", interpolator=None, dtype=tf.complex64, **kwargs)

    Abstract layer for implementing an OFDM channel estimator.

    Any layer that implements an OFDM channel estimator must implement this
    class and its
    :meth:`~sionna.ofdm.BaseChannelEstimator.estimate_at_pilot_locations`
    abstract method.

    This class extracts the pilots from the received resource grid ``y``, calls
    the :meth:`~sionna.ofdm.BaseChannelEstimator.estimate_at_pilot_locations`
    method to estimate the channel for the pilot-carrying resource elements,
    and then interpolates the channel to compute channel estimates for the
    data-carrying resouce elements using the interpolation method specified by
    ``interpolation_type`` or the ``interpolator`` object.

    Parameters
    ----------
    resource_grid : ResourceGrid
        An instance of :class:`~sionna.ofdm.ResourceGrid`.

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
        or `None`. In the latter case, the interpolator specfied
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
    h_hat : [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols,fft_size], tf.complex
        Channel estimates accross the entire resource grid for all
        transmitters and streams

    err_var : Same shape as ``h_hat``, tf.float
        Channel estimation error variance accross the entire resource grid
        for all transmitters and streams
    """
    def __init__(self, resource_grid, interpolation_type="nn", interpolator=None, dtype=tf.complex64, **kwargs):
        super().__init__(dtype=dtype, **kwargs)

        assert isinstance(resource_grid, ResourceGrid),\
            "You must provide a valid instance of ResourceGrid."
        self._pilot_pattern = resource_grid.pilot_pattern
        self._removed_nulled_scs = RemoveNulledSubcarriers(resource_grid)

        assert interpolation_type in ["nn","lin","lin_time_avg",None], \
            "Unsupported `interpolation_type`"
        self._interpolation_type = interpolation_type

        if interpolator is not None:
            assert isinstance(interpolator, BaseChannelInterpolator), \
        "`interpolator` must implement the BaseChannelInterpolator interface"
            self._interpol = interpolator
        elif self._interpolation_type == "nn":
            self._interpol = NearestNeighborInterpolator(self._pilot_pattern)
        elif self._interpolation_type == "lin":
            self._interpol = LinearInterpolator(self._pilot_pattern)
        elif self._interpolation_type == "lin_time_avg":
            self._interpol = LinearInterpolator(self._pilot_pattern,
                                                time_avg=True)

        # Precompute indices to gather received pilot signals
        num_pilot_symbols = self._pilot_pattern.num_pilot_symbols
        mask = flatten_last_dims(self._pilot_pattern.mask)
        pilot_ind = tf.argsort(mask, axis=-1, direction="DESCENDING")
        self._pilot_ind = pilot_ind[...,:num_pilot_symbols]

    @abstractmethod
    def estimate_at_pilot_locations(self, y_pilots, no):
        """
        Estimates the channel for the pilot-carrying resource elements.

        This is an abstract method that must be implemented by a concrete
        OFDM channel estimator that implement this class.

        Input
        -----
        y_pilots : [batch_size, num_rx, num_rx_ant, num_tx, num_streams, num_pilot_symbols], tf.complex
            Observed signals for the pilot-carrying resource elements

        no : [batch_size, num_rx, num_rx_ant] or only the first n>=0 dims, tf.float
            Variance of the AWGN

        Output
        ------
        h_hat : [batch_size, num_rx, num_rx_ant, num_tx, num_streams, num_pilot_symbols], tf.complex
            Channel estimates for the pilot-carrying resource elements

        err_var : Same shape as ``h_hat``, tf.float
            Channel estimation error variance for the pilot-carrying
            resource elements
        """
        pass

    def call(self, inputs):

        y, no = inputs

        # y has shape:
        # [batch_size, num_rx, num_rx_ant, num_ofdm_symbols,..
        # ... fft_size]
        #
        # no can have shapes [], [batch_size], [batch_size, num_rx]
        # or [batch_size, num_rx, num_rx_ant]

        # Removed nulled subcarriers (guards, dc)
        y_eff = self._removed_nulled_scs(y)

        # Flatten the resource grid for pilot extraction
        # New shape: [...,num_ofdm_symbols*num_effective_subcarriers]
        y_eff_flat = flatten_last_dims(y_eff)

        # Gather pilots along the last dimensions
        # Resulting shape: y_eff_flat.shape[:-1] + pilot_ind.shape, i.e.:
        # [batch_size, num_rx, num_rx_ant, num_tx, num_streams,...
        #  ..., num_pilot_symbols]
        y_pilots = tf.gather(y_eff_flat, self._pilot_ind, axis=-1)

        # Compute LS channel estimates
        # Note: Some might be Inf because pilots=0, but we do not care
        # as only the valid estimates will be considered during interpolation.
        # We do a save division to replace Inf by 0.
        # Broadcasting from pilots here is automatic since pilots have shape
        # [num_tx, num_streams, num_pilot_symbols]
        h_hat, err_var = self.estimate_at_pilot_locations(y_pilots, no)

        # Interpolate channel estimates over the resource grid
        if self._interpolation_type is not None:
            h_hat, err_var = self._interpol(h_hat, err_var)
            err_var = tf.maximum(err_var, tf.cast(0, err_var.dtype))

        return h_hat, err_var


class LSChannelEstimator(BaseChannelEstimator, Layer):
    # pylint: disable=line-too-long
    r"""LSChannelEstimator(resource_grid, interpolation_type="nn", interpolator=None, dtype=tf.complex64, **kwargs)

    Layer implementing least-squares (LS) channel estimation for OFDM MIMO systems.

    After LS channel estimation at the pilot positions, the channel estimates
    and error variances are interpolated accross the entire resource grid using
    a specified interpolation function.

    For simplicity, the underlying algorithm is described for a vectorized observation,
    where we have a nonzero pilot for all elements to be estimated.
    The actual implementation works on a full OFDM resource grid with sparse
    pilot patterns. The following model is assumed:

    .. math::

        \mathbf{y} = \mathbf{h}\odot\mathbf{p} + \mathbf{n}

    where :math:`\mathbf{y}\in\mathbb{C}^{M}` is the received signal vector,
    :math:`\mathbf{p}\in\mathbb{C}^M` is the vector of pilot symbols,
    :math:`\mathbf{h}\in\mathbb{C}^{M}` is the channel vector to be estimated,
    and :math:`\mathbf{n}\in\mathbb{C}^M` is a zero-mean noise vector whose
    elements have variance :math:`N_0`. The operator :math:`\odot` denotes
    element-wise multiplication.

    The channel estimate :math:`\hat{\mathbf{h}}` and error variances
    :math:`\sigma^2_i`, :math:`i=0,\dots,M-1`, are computed as

    .. math::

        \hat{\mathbf{h}} &= \mathbf{y} \odot
                           \frac{\mathbf{p}^\star}{\left|\mathbf{p}\right|^2}
                         = \mathbf{h} + \tilde{\mathbf{h}}\\
             \sigma^2_i &= \mathbb{E}\left[\tilde{h}_i \tilde{h}_i^\star \right]
                         = \frac{N_0}{\left|p_i\right|^2}.

    The channel estimates and error variances are then interpolated accross
    the entire resource grid.

    Parameters
    ----------
    resource_grid : ResourceGrid
        An instance of :class:`~sionna.ofdm.ResourceGrid`.

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
        or `None`. In the latter case, the interpolator specfied
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
        Channel estimates accross the entire resource grid for all
        transmitters and streams

    err_var : Same shape as ``h_ls``, tf.float
        Channel estimation error variance accross the entire resource grid
        for all transmitters and streams
    """

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

        # Compute error variance and broadcast to the same shape as h_ls
        # Expand rank of no for broadcasting
        no = expand_to_rank(no, tf.rank(h_ls), -1)

        # Expand rank of pilots for broadcasting
        pilots = expand_to_rank(self._pilot_pattern.pilots, tf.rank(h_ls), 0)

        # Compute error variance, broadcastable to the shape of h_ls
        err_var = tf.math.divide_no_nan(no, tf.abs(pilots)**2)

        return h_ls, err_var


class BaseChannelInterpolator(ABC):
    # pylint: disable=line-too-long
    r"""BaseChannelInterpolator()

    Abstract layer for implementing an OFDM channel interpolator.

    Any layer that implements an OFDM channel interpolator must implement this
    callable class.

    A channel interpolator is used by an OFDM channel estimator
    (:class:`~sionna.ofdm.BaseChannelEstimator`) to compute channel estimates
    for the data-carrying resource elements from the channel estimates for the
    pilot-carrying resource elements.

    Input
    -----
    h_hat : [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_pilot_symbols], tf.complex
        Channel estimates for the pilot-carrying resource elements

    err_var : [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_pilot_symbols], tf.complex
        Channel estimation error variances for the pilot-carrying resource elements

    Output
    ------
    h_hat : [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size], tf.complex
        Channel estimates accross the entire resource grid for all
        transmitters and streams

    err_var : Same shape as ``h_hat``, tf.float
        Channel estimation error variance accross the entire resource grid
        for all transmitters and streams
    """

    @abstractmethod
    def __call__(self, h_hat, err_var):
        pass


class NearestNeighborInterpolator(BaseChannelInterpolator):
    # pylint: disable=line-too-long
    r"""NearestNeighborInterpolator(pilot_pattern)

    Nearest-neighbor channel estimate interpolation on a resource grid.

    This class assigns to each element of an OFDM resource grid one of
    ``num_pilots`` provided channel estimates and error
    variances according to the nearest neighbor method. It is assumed
    that the measurements were taken at the nonzero positions of a
    :class:`~sionna.ofdm.PilotPattern`.

    The figure below shows how four channel estimates are interpolated
    accross a resource grid. Grey fields indicate measurement positions
    while the colored regions show which resource elements are assigned
    to the same measurement value.

    .. image:: ../figures/nearest_neighbor_interpolation.png

    Parameters
    ----------
    pilot_pattern : PilotPattern
        An instance of :class:`~sionna.ofdm.PilotPattern`

    Input
    -----
    h_hat : [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_pilot_symbols], tf.complex
        Channel estimates for the pilot-carrying resource elements

    err_var : [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_pilot_symbols], tf.complex
        Channel estimation error variances for the pilot-carrying resource elements

    Output
    ------
    h_hat : [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size], tf.complex
        Channel estimates accross the entire resource grid for all
        transmitters and streams

    err_var : Same shape as ``h_hat``, tf.float
        Channel estimation error variances accross the entire resource grid
        for all transmitters and streams
    """
    def __init__(self, pilot_pattern):
        super().__init__()

        assert(pilot_pattern.num_pilot_symbols>0),\
            """The pilot pattern cannot be empty"""

        # Reshape mask to shape [-1,num_ofdm_symbols,num_effective_subcarriers]
        mask = np.array(pilot_pattern.mask)
        mask_shape = mask.shape # Store to reconstruct the original shape
        mask = np.reshape(mask, [-1] + list(mask_shape[-2:]))

        # Reshape the pilots to shape [-1, num_pilot_symbols]
        pilots = pilot_pattern.pilots
        pilots = np.reshape(pilots, [-1] + [pilots.shape[-1]])

        max_num_zero_pilots = np.max(np.sum(np.abs(pilots)==0, -1))
        assert max_num_zero_pilots<pilots.shape[-1],\
            """Each pilot sequence must have at least one nonzero entry"""

        # Compute gather indices for nearest neighbor interpolation
        gather_ind = np.zeros_like(mask, dtype=np.int32)
        for a in range(gather_ind.shape[0]): # For each pilot pattern...
            i_p, j_p = np.where(mask[a]) # ...determine the pilot indices

            for i in range(mask_shape[-2]): # Iterate over...
                for j in range(mask_shape[-1]): # ... all resource elements

                    # Compute Manhattan distance to all pilot positions
                    d = np.abs(i-i_p) + np.abs(j-j_p)

                    # Set the distance at all pilot positions with zero energy
                    # equal to the maximum possible distance
                    d[np.abs(pilots[a])==0] = np.sum(mask_shape[-2:])

                    # Find the pilot index with the shortest distance...
                    ind = np.argmin(d)

                    # ... and store it in the index tensor
                    gather_ind[a, i, j] = ind

        # Reshape to the original shape of the mask, i.e.:
        # [num_tx, num_streams_per_tx, num_ofdm_symbols,...
        #  ..., num_effective_subcarriers]
        self._gather_ind = tf.reshape(gather_ind, mask_shape)

    def _interpolate(self, inputs):
        # inputs has shape:
        # [k, l, m, num_tx, num_streams_per_tx, num_pilots]

        # Transpose inputs to bring batch_dims for gather last. New shape:
        # [num_tx, num_streams_per_tx, num_pilots, k, l, m]
        perm = tf.roll(tf.range(tf.rank(inputs)), -3, 0)
        inputs = tf.transpose(inputs, perm)

        # Interpolate through gather. Shape:
        # [num_tx, num_streams_per_tx, num_ofdm_symbols,
        #  ..., num_effective_subcarriers, k, l, m]
        outputs = tf.gather(inputs, self._gather_ind, 2, batch_dims=2)

        # Transpose outputs to bring batch_dims first again. New shape:
        # [k, l, m, num_tx, num_streams_per_tx,...
        #  ..., num_ofdm_symbols, num_effective_subcarriers]
        perm = tf.roll(tf.range(tf.rank(outputs)), 3, 0)
        outputs = tf.transpose(outputs, perm)

        return outputs

    def __call__(self, h_hat, err_var):

        h_hat = self._interpolate(h_hat)
        err_var = self._interpolate(err_var)
        return h_hat, err_var


class LinearInterpolator(BaseChannelInterpolator):
    # pylint: disable=line-too-long
    r"""LinearInterpolator(pilot_pattern, time_avg=False)

    Linear channel estimate interpolation on a resource grid.

    This class computes for each element of an OFDM resource grid
    a channel estimate based on ``num_pilots`` provided channel estimates and
    error variances through linear interpolation.
    It is assumed that the measurements were taken at the nonzero positions
    of a :class:`~sionna.ofdm.PilotPattern`.

    The interpolation is done first across sub-carriers and then
    across OFDM symbols.

    Parameters
    ----------
    pilot_pattern : PilotPattern
        An instance of :class:`~sionna.ofdm.PilotPattern`

    time_avg : bool
        If enabled, measurements will be averaged across OFDM symbols
        (i.e., time). This is useful for channels that do not vary
        substantially over the duration of an OFDM frame. Defaults to `False`.

    Input
    -----
    h_hat : [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_pilot_symbols], tf.complex
        Channel estimates for the pilot-carrying resource elements

    err_var : [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_pilot_symbols], tf.complex
        Channel estimation error variances for the pilot-carrying resource elements

    Output
    ------
    h_hat : [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size], tf.complex
        Channel estimates accross the entire resource grid for all
        transmitters and streams

    err_var : Same shape as ``h_hat``, tf.float
        Channel estimation error variances accross the entire resource grid
        for all transmitters and streams
    """
    def __init__(self, pilot_pattern, time_avg=False):
        super().__init__()

        assert(pilot_pattern.num_pilot_symbols>0),\
            """The pilot pattern cannot be empty"""

        self._time_avg = time_avg

        # Reshape mask to shape [-1,num_ofdm_symbols,num_effective_subcarriers]
        mask = np.array(pilot_pattern.mask)
        mask_shape = mask.shape # Store to reconstruct the original shape
        mask = np.reshape(mask, [-1] + list(mask_shape[-2:]))

        # Reshape the pilots to shape [-1, num_pilot_symbols]
        pilots = pilot_pattern.pilots
        pilots = np.reshape(pilots, [-1] + [pilots.shape[-1]])

        max_num_zero_pilots = np.max(np.sum(np.abs(pilots)==0, -1))
        assert max_num_zero_pilots<pilots.shape[-1],\
            """Each pilot sequence must have at least one nonzero entry"""

        # Create actual pilot patterns for each stream over the resource grid
        z = np.zeros_like(mask, dtype=pilots.dtype)
        for a in range(z.shape[0]):
            z[a][np.where(mask[a])] = pilots[a]

        # Linear interpolation works as follows:
        # We compute for each resource element (RE)
        # x_0 : The x-value (i.e., sub-carrier index or OFDM symbol) at which
        #       the first channel measurement was taken
        # x_1 : The x-value (i.e., sub-carrier index or OFDM symbol) at which
        #       the second channel measurement was taken
        # y_0 : The first channel estimate
        # y_1 : The second channel estimate
        # x   : The x-value (i.e., sub-carrier index or OFDM symbol)
        #
        # The linearly interpolated value y is then given as:
        # y = (x-x_0) * (y_1-y_0) / (x_1-x_0) + y_0
        #
        # The following code pre-computes various quantities and indices
        # that are needed to compute x_0, x_1, y_0, y_1, x for frequency- and
        # time-domain interpolation.

        ##
        ## Frequency-domain interpolation
        ##
        self._x_freq = tf.cast(expand_to_rank(tf.range(0, mask.shape[-1]),
                                              7,
                                              axis=0),
                               pilots.dtype)

        # Permutation indices to shift batch_dims last during gather
        self._perm_fwd_freq = tf.roll(tf.range(6), -3, 0)

        x_0_freq = np.zeros_like(mask, np.int32)
        x_1_freq = np.zeros_like(mask, np.int32)

        # Set REs of OFDM symbols without any pilot equal to -1 (dummy value)
        x_0_freq[np.sum(np.abs(z), axis=-1)==0] = -1
        x_1_freq[np.sum(np.abs(z), axis=-1)==0] = -1

        y_0_freq_ind = np.copy(x_0_freq) # Indices used to gather estimates
        y_1_freq_ind = np.copy(x_1_freq) # Indices used to gather estimates

        # For each stream
        for a in range(z.shape[0]):

            pilot_count = 0 # Counts the number of non-zero pilots

            # Indices of non-zero pilots within the pilots vector
            pilot_ind = np.where(np.abs(pilots[a]))[0]

            # Go through all OFDM symbols
            for i in range(x_0_freq.shape[1]):

                # Indices of non-zero pilots within the OFDM symbol
                pilot_ind_ofdm = np.where(np.abs(z[a][i]))[0]

                # If OFDM symbol contains only one non-zero pilot
                if len(pilot_ind_ofdm)==1:
                    # Set the indices of the first and second pilot to the same
                    # value for all REs of the OFDM symbol
                    x_0_freq[a][i] = pilot_ind_ofdm[0]
                    x_1_freq[a][i] = pilot_ind_ofdm[0]
                    y_0_freq_ind[a,i] = pilot_ind[pilot_count]
                    y_1_freq_ind[a,i] = pilot_ind[pilot_count]

                # If OFDM symbol contains two or more pilots
                elif len(pilot_ind_ofdm)>=2:
                    x0 = 0
                    x1 = 1

                    # Go through all resource elements of this OFDM symbol
                    for j in range(x_0_freq.shape[2]):
                        x_0_freq[a,i,j] = pilot_ind_ofdm[x0]
                        x_1_freq[a,i,j] = pilot_ind_ofdm[x1]
                        y_0_freq_ind[a,i,j] = pilot_ind[pilot_count + x0]
                        y_1_freq_ind[a,i,j] = pilot_ind[pilot_count + x1]
                        if j==pilot_ind_ofdm[x1] and x1<len(pilot_ind_ofdm)-1:
                            x0 = x1
                            x1 += 1

                pilot_count += len(pilot_ind_ofdm)

        x_0_freq = np.reshape(x_0_freq, mask_shape)
        x_1_freq = np.reshape(x_1_freq, mask_shape)
        x_0_freq = expand_to_rank(x_0_freq, 7, axis=0)
        x_1_freq = expand_to_rank(x_1_freq, 7, axis=0)
        self._x_0_freq = tf.cast(x_0_freq, pilots.dtype)
        self._x_1_freq = tf.cast(x_1_freq, pilots.dtype)

        # We add +1 here to shift all indices as the input will be padded
        # at the beginning with 0, (i.e., the dummy index -1 will become 0).
        self._y_0_freq_ind = np.reshape(y_0_freq_ind, mask_shape)+1
        self._y_1_freq_ind = np.reshape(y_1_freq_ind, mask_shape)+1

        ##
        ## Time-domain interpolation
        ##
        self._x_time = tf.expand_dims(tf.range(0, mask.shape[-2]), -1)
        self._x_time = tf.cast(expand_to_rank(self._x_time, 7, axis=0),
                               dtype=pilots.dtype)

        # Indices used to gather estimates
        self._perm_fwd_time = tf.roll(tf.range(7), -3, 0)

        y_0_time_ind = np.zeros(z.shape[:2], np.int32) # Gather indices
        y_1_time_ind = np.zeros(z.shape[:2], np.int32) # Gather indices

        # For each stream
        for a in range(z.shape[0]):

            # Indices of OFDM symbols for which channel estimates were computed
            ofdm_ind = np.where(np.sum(np.abs(z[a]), axis=-1))[0]

            # Only one OFDM symbol with pilots
            if len(ofdm_ind)==1:
                y_0_time_ind[a] = ofdm_ind[0]
                y_1_time_ind[a] = ofdm_ind[0]

            # Two or more OFDM symbols with pilots
            elif len(ofdm_ind)>=2:
                x0 = 0
                x1 = 1
                for i in range(z.shape[1]):
                    y_0_time_ind[a,i] = ofdm_ind[x0]
                    y_1_time_ind[a,i] = ofdm_ind[x1]
                    if i==ofdm_ind[x1] and x1<len(ofdm_ind)-1:
                        x0 = x1
                        x1 += 1

        self._y_0_time_ind = np.reshape(y_0_time_ind, mask_shape[:-1])
        self._y_1_time_ind = np.reshape(y_1_time_ind, mask_shape[:-1])

        self._x_0_time = expand_to_rank(tf.expand_dims(self._y_0_time_ind, -1),
                                                       7, axis=0)
        self._x_0_time = tf.cast(self._x_0_time, dtype=pilots.dtype)
        self._x_1_time = expand_to_rank(tf.expand_dims(self._y_1_time_ind, -1),
                                                       7, axis=0)
        self._x_1_time = tf.cast(self._x_1_time, dtype=pilots.dtype)

        #
        # Other precomputed values
        #
        # Undo permutation of batch_dims for gather
        self._perm_bwd = tf.roll(tf.range(7), 3, 0)

        # Padding for the inputs
        pad = np.zeros([6, 2], np.int32)
        pad[-1, 0] = 1
        self._pad = pad

        # Number of ofdm symbols carrying at least one pilot.
        # Used for time-averaging (optional)
        n = np.sum(np.abs(np.reshape(z, mask_shape)), axis=-1, keepdims=True)
        n = np.sum(n>0, axis=-2, keepdims=True)
        self._num_pilot_ofdm_symbols = expand_to_rank(n, 7, axis=0)


    def _interpolate_1d(self, inputs, x, x0, x1, y0_ind, y1_ind):
        # Gather the right values for y0 and y1
        y0 = tf.gather(inputs, y0_ind, axis=2, batch_dims=2)
        y1 = tf.gather(inputs, y1_ind, axis=2, batch_dims=2)

        # Undo the permutation of the inputs
        y0 = tf.transpose(y0, self._perm_bwd)
        y1 = tf.transpose(y1, self._perm_bwd)

        # Compute linear interpolation
        slope = tf.math.divide_no_nan(y1-y0, tf.cast(x1-x0, dtype=y0.dtype))
        return tf.cast(x-x0, dtype=y0.dtype)*slope + y0

    def _interpolate(self, inputs):
        #
        # Prepare inputs
        #
        # inputs has shape:
        # [k, l, m, num_tx, num_streams_per_tx, num_pilots]

        # Pad the inputs with a leading 0.
        # All undefined channel estimates will get this value.
        inputs = tf.pad(inputs, self._pad, constant_values=0)

        # Transpose inputs to bring batch_dims for gather last. New shape:
        # [num_tx, num_streams_per_tx, 1+num_pilots, k, l, m]
        inputs = tf.transpose(inputs, self._perm_fwd_freq)

        #
        # Frequency-domain interpolation
        #
        # h_hat_freq has shape:
        # [k, l, m, num_tx, num_streams_per_tx, num_ofdm_symbols,...
        #  ...num_effective_subcarriers]
        h_hat_freq = self._interpolate_1d(inputs,
                                          self._x_freq,
                                          self._x_0_freq,
                                          self._x_1_freq,
                                          self._y_0_freq_ind,
                                          self._y_1_freq_ind)
        #
        # Time-domain interpolation
        #

        # Time-domain averaging (optional)
        if self._time_avg:
            num_ofdm_symbols = h_hat_freq.shape[-2]
            h_hat_freq = tf.reduce_sum(h_hat_freq, axis=-2, keepdims=True)
            h_hat_freq /= tf.cast(self._num_pilot_ofdm_symbols,h_hat_freq.dtype)
            h_hat_freq = tf.repeat(h_hat_freq, [num_ofdm_symbols], axis=-2)

        # Transpose h_hat_freq to bring batch_dims for gather last. New shape:
        # [num_tx, num_streams_per_tx, num_ofdm_symbols,...
        #  ...num_effective_subcarriers, k, l, m]
        h_hat_time = tf.transpose(h_hat_freq, self._perm_fwd_time)

        # h_hat_time has shape:
        # [k, l, m, num_tx, num_streams_per_tx, num_ofdm_symbols,...
        #  ...num_effective_subcarriers]
        h_hat_time = self._interpolate_1d(h_hat_time,
                                          self._x_time,
                                          self._x_0_time,
                                          self._x_1_time,
                                          self._y_0_time_ind,
                                          self._y_1_time_ind)

        return h_hat_time

    def __call__(self, h_hat, err_var):

        h_hat = self._interpolate(h_hat)
        err_var = self._interpolate(err_var)
        return h_hat, err_var


class LMMSEInterpolator1D:
    # pylint: disable=line-too-long
    r"""LMMSEInterpolator1D(pilot_mask, cov_mat)

    This class performs the linear interpolation across the inner dimension of the input ``h_hat``.

    The two inner dimensions of the input ``h_hat`` form a matrix :math:`\hat{\mathbf{H}} \in \mathbb{C}^{N \times M}`.
    LMMSE interpolation is performed across the inner dimension as follows:

    .. math::
        \tilde{\mathbf{h}}_n = \mathbf{A}_n \hat{\mathbf{h}}_n

    where :math:`1 \leq n \leq N` and :math:`\hat{\mathbf{h}}_n` is
    the :math:`n^{\text{th}}` (transposed) row of :math:`\hat{\mathbf{H}}`.
    :math:`\mathbf{A}_n` is the :math:`M \times M` interpolation LMMSE matrix:

    .. math::
        \mathbf{A}_n = \mathbf{R} \mathbf{\Pi}_n \left( \mathbf{\Pi}_n^\intercal \mathbf{R} \mathbf{\Pi}_n + \tilde{\mathbf{\Sigma}}_n \right)^{-1} \mathbf{\Pi}_n^\intercal.

    where :math:`\mathbf{R}` is the :math:`M \times M` covariance matrix across the inner dimension of the quantity which is estimated,
    :math:`\mathbf{\Pi}_n` the :math:`M \times K_n` matrix that spreads :math:`K_n`
    values to a vector of size :math:`M` according to the ``pilot_mask`` for the :math:`n^{\text{th}}` row,
    and :math:`\tilde{\mathbf{\Sigma}}_n \in \mathbb{R}^{K_n \times K_n}` is the regularized channel estimation error covariance.
    The :math:`i^{\text{th}}`` diagonal element of :math:`\tilde{\mathbf{\Sigma}}_n` is such that:

    .. math::

        \left[ \tilde{\mathbf{\Sigma}}_n \right]_{i,i} = \text{max} \left\{  \right\}

     built from ``err_var`` and assumed to be diagonal.

    The returned channel estimates are

    .. math::
        \begin{bmatrix}
            {\tilde{\mathbf{h}}_1}^\intercal\\
            \vdots\\
            {\tilde{\mathbf{h}}_N}^\intercal
        \end{bmatrix}.

    The returned channel estimation error variances are the diaginal coefficients of

    .. math::
        \text{diag} \left( \mathbf{R} - \mathbf{A}_n \mathbf{\Xi}_n \mathbf{R} \right), 1 \leq n \leq N

    where :math:`\mathbf{\Xi}_n` is the diagonal matrix of size :math:`M \times M` that zeros the
    columns corresponding to rows not carrying any pilots.
    Note that interpolation is not performed for rows not carrying any pilots.

    **Remark**: The interpolation matrix differs across rows as different
    rows may carry pilots on different elements and/or have different
    estimation error variances.

    Parameters
    ----------
    pilot_mask : [:math:`N`, :math:`M`] : int
        Mask indicating the allocation of resource elements.
        0 : Data,
        1 : Pilot,
        2 : Not used,

    cov_mat : [:math:`M`, :math:`M`], tf.complex
        Covariance matrix of the channel across the inner dimension.

    last_step : bool
        Set to `True` if this is the last interpolation step.
        Otherwise, set to `False`.
        If `True`, the the output is scaled to ensure its variance is as expected
        by the following interpolation step.

    Input
    -----
    h_hat : [batch_size, num_rx, num_rx_ant, num_tx, :math:`N`, :math:`M`], tf.complex
        Channel estimates.

    err_var : [batch_size, num_rx, num_rx_ant, num_tx, :math:`N`, :math:`M`], tf.complex
        Channel estimation error variances.

    Output
    ------
    h_hat : [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, :math:`N`, :math:`M`], tf.complex
        Channel estimates interpolated across the inner dimension.

    err_var : Same shape as ``h_hat``, tf.float
        The channel estimation error variances of the interpolated channel estimates.
    """

    def __init__(self, pilot_mask, cov_mat, last_step):

        self._cdtype = cov_mat.dtype
        assert self._cdtype in (tf.complex64, tf.complex128),\
            "`cov_mat` dtype must be one of tf.complex64 or tf.complex128"
        self._rdtype = self._cdtype.real_dtype
        self._rzero = tf.constant(0.0, self._rdtype)

        # Interpolation is performed along the inner dimension of
        # the resource grid, which may be either the subcarriers
        # or the OFDM symbols dimension.
        # This dimension is referred to as the inner dimension.
        # The other dimension of the resource grid is referred to
        # as the outer dimension.

        # Size of the inner dimension.
        inner_dim_size = tf.shape(pilot_mask)[-1]
        self._inner_dim_size = inner_dim_size

        # Size of the outer dimension.
        outer_dim_size = tf.shape(pilot_mask)[-2]
        self._outer_dim_size = outer_dim_size

        self._cov_mat = cov_mat
        self._last_step = last_step

        # Computation of the interpolation matrix is done solving the
        # least-square problem:
        #
        # X = min_Z |AZ - B|_F^2
        #
        # where A = (\Pi_T R \Pi + S) and
        # B = R \Pi
        # where R is the channel covariance matrix, S the error
        # diagonal covariance matrix, and \Pi the matrix that spreads the pilots
        # according to the pilot pattern along the inner axis.

        # Extracting the locations of pilots from the pilot mask
        num_tx = tf.shape(pilot_mask)[0]
        num_streams_per_tx = tf.shape(pilot_mask)[1]

        # List of indices of pilots in the inner dimension for every
        # transmit antenna, stream, and outer dimension element.
        pilot_indices = []
        # Maximum number of pilots carried by an inner dimension.
        max_num_pil = 0
        # Indices used to add the error variance to the diagonal
        # elements of the covariance matrix restricted
        # to the elements carrying pilots.
        # These matrices are computed below.
        add_err_var_indices = np.zeros([num_tx, num_streams_per_tx,
                                        outer_dim_size, inner_dim_size, 5], int)
        for tx in range(num_tx):
            pilot_indices.append([])
            for st in range(num_streams_per_tx):
                pilot_indices[-1].append([])
                for oi in range(outer_dim_size):
                    pilot_indices[-1][-1].append([])
                    num_pil = 0 # Number of pilots on this outer dim
                    for ii in range(inner_dim_size):
                        # Check if this RE is carrying a pilot
                        # for this stream
                        if pilot_mask[tx,st,oi,ii] == 0:
                            continue
                        if pilot_mask[tx,st,oi,ii] == 1:
                            pilot_indices[tx][st][oi].append(ii)
                            indices = [tx, st, oi, num_pil, num_pil]
                            add_err_var_indices[tx, st, oi, ii] = indices
                            num_pil += 1
                    if num_pil > max_num_pil:
                        max_num_pil = num_pil
        # [num_tx, num_streams_per_tx, outer_dim_size, inner_dim_size, 5]
        self._add_err_var_indices = tf.cast(add_err_var_indices, tf.int32)

        # Different subcarriers/symbols may carry a different number of pilots.
        # To handle such cases, we create a tensor of square matrices of
        # size the maximum number of pilots carried by an inner dimension
        # and zero-padding is used to handle axes with less pilots than the
        # maximum value. The obtained structure is:
        #
        # |B 0|
        # |0 0|
        #
        pil_cov_mat = np.zeros([num_tx, num_streams_per_tx, outer_dim_size,
                                max_num_pil, max_num_pil], complex)
        for tx,st,oi in itertools.product(range(num_tx),
                                          range(num_streams_per_tx),
                                          range(outer_dim_size)):
            pil_ind = pilot_indices[tx][st][oi]
            num_pil = len(pil_ind)
            tmp = np.take(cov_mat, pil_ind, axis=0)
            pil_cov_mat_ = np.take(tmp, pil_ind, axis=1)
            pil_cov_mat[tx,st,oi,:num_pil,:num_pil] = pil_cov_mat_
        # [num_tx, num_streams_per_tx, outer_dim_size, max_num_pil, max_num_pil]
        self._pil_cov_mat = tf.constant(pil_cov_mat, self._cdtype)

        # Pre-compute the covariance matrix with only the columns corresponding
        # to pilots.
        b_mat = np.zeros([num_tx, num_streams_per_tx, outer_dim_size,
                                max_num_pil, inner_dim_size], complex)
        for tx,st,oi in itertools.product(range(num_tx),
                                          range(num_streams_per_tx),
                                          range(outer_dim_size)):
            pil_ind = pilot_indices[tx][st][oi]
            num_pil = len(pil_ind)
            b_mat_ = np.take(cov_mat, pil_ind, axis=0)
            b_mat[tx,st,oi,:num_pil,:] = b_mat_
        self._b_mat = tf.constant(b_mat, self._cdtype)

        # Indices used to fill with zeros the columns of the interpolation
        # matrix not corresponding to zeros.
        # The results is a matrix of size inner_dim_size x inner_dim_size
        # where rows and columns not correspondong to pilots are set to zero.
        pil_loc = np.zeros([num_tx, num_streams_per_tx, outer_dim_size,
                            inner_dim_size, max_num_pil, 5], dtype=int)
        for tx,st,oi,p,ii in itertools.product(range(num_tx),
                                                range(num_streams_per_tx),
                                                range(outer_dim_size),
                                                range(max_num_pil),
                                                range(inner_dim_size)):
            if p >= len(pilot_indices[tx][st][oi]):
                # An extra dummy subcarrier is added to push there padding
                # identity matrix
                pil_loc[tx, st, oi, ii, p] = [tx, st, oi,
                                              inner_dim_size,
                                              inner_dim_size]
            else:
                pil_loc[tx, st, oi, ii, p] = [tx, st, oi,
                                              ii,
                                              pilot_indices[tx][st][oi][p]]
        self._pil_loc = tf.cast(pil_loc, tf.int32)

        # Covariance matrix for each stream with only the row corresponding
        # to a pilot carrying RE not set to 0.
        # This is required to compute the estimation error variances.
        err_var_mat = np.zeros([num_tx, num_streams_per_tx, outer_dim_size,
                inner_dim_size, inner_dim_size], complex)
        for tx,st,oi in itertools.product(range(num_tx),
                                          range(num_streams_per_tx),
                                          range(outer_dim_size)):
            pil_ind = pilot_indices[tx][st][oi]
            mask = np.zeros([inner_dim_size], complex)
            mask[pil_ind] = 1.0
            mask = np.expand_dims(mask, axis=1)
            err_var_mat[tx,st,oi] = cov_mat*mask
        self._err_var_mat = tf.constant(err_var_mat, self._cdtype)

    def __call__(self, h_hat, err_var):

        # h_hat : [batch_size, num_rx, num_rx_ant, num_tx,
        #          num_streams_per_tx, outer_dim_size, inner_dim_size]
        # err_var : [batch_size, num_rx, num_rx_ant, num_tx,
        #          num_streams_per_tx, outer_dim_size, inner_dim_size]

        batch_size = tf.shape(h_hat)[0]
        num_rx = tf.shape(h_hat)[1]
        num_rx_ant = tf.shape(h_hat)[2]
        num_tx = tf.shape(h_hat)[3]
        num_tx_stream = tf.shape(h_hat)[4]
        outer_dim_size = self._outer_dim_size
        inner_dim_size = self._inner_dim_size

        #####################################
        # Compute the interpolation matrix
        #####################################

        # Computation of the interpolation matrix is done solving the
        # least-square problem:
        #
        # X = min_Z |AZ - B|_F^2
        #
        # where A = (\Pi_T R \Pi + S) and
        # B = R \Pi
        # where R is the channel covariance matrix, S the error
        # diagonal covariance matrix, and \Pi the matrix that spreads the pilots
        # according to the pilot pattern along the inner axis.

        #
        # Computing A
        #

        # Covariance matrices restricted to pilot locations
        # [num_tx, num_streams_per_tx, outer_dim_size, max_num_pil, max_num_pil]
        pil_cov_mat = self._pil_cov_mat

        # Adding batch, receive, and receive antennas dimensions to the
        # covariance matrices restricted to pilot locations and to the
        # regularization values
        # [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx,
        #  outer_dim_size, max_num_pil, max_num_pil]
        pil_cov_mat = expand_to_rank(pil_cov_mat, 8, 0)
        pil_cov_mat = tf.tile(pil_cov_mat, [batch_size, num_rx, num_rx_ant,
                                                     1, 1, 1, 1, 1])

        # Adding the noise variance to the covariance matrices restricted to
        # pilots
        # [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx,
        #  outer_dim_size, max_num_pil, max_num_pil]
        pil_cov_mat_ = tf.transpose(pil_cov_mat, [3, 4, 5, 6, 7, 0, 1, 2])
        err_var_ = tf.complex(err_var, self._rzero)
        err_var_ = tf.transpose(err_var_, [3, 4, 5, 6, 0, 1, 2])
        a_mat = tf.tensor_scatter_nd_add(pil_cov_mat_,
                                        self._add_err_var_indices, err_var_)
        a_mat = tf.transpose(a_mat, [5, 6, 7, 0, 1, 2, 3, 4])

        #
        # Computing B
        #

        # B is pre-computed as it only depend on the channel covariance and
        # pilot pattern.
        # [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx,
        #  outer_dim_size, max_num_pil, inner_dim_size]
        b_mat = self._b_mat
        b_mat = expand_to_rank(b_mat, 8, 0)
        b_mat = tf.tile(b_mat, [batch_size, num_rx, num_rx_ant,
                                1, 1, 1, 1, 1])

        #
        # Computing the interpolation matrix
        #

        # Using lstsq to compute the columns of the interpolation matrix
        # corresponding to pilots.
        # [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx,
        #  outer_dim_size, inner_dim_size, max_num_pil]
        ext_mat = tf.linalg.lstsq(a_mat, b_mat, fast=False)
        ext_mat = tf.transpose(ext_mat, [0,1,2,3,4,5,7,6], conjugate=True)

        # Filling with zeros the columns not corresponding to pilots.
        # An extra dummy outer dim is added to scatter there the coefficients
        # of the identity matrix used for padding.
        # This dummy dim is then removed.
        # [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx,
        #  outer_dim_size, inner_dim_size, inner_dim_size]
        ext_mat = tf.transpose(ext_mat, [3, 4, 5, 6, 7, 0, 1, 2])
        ext_mat = tf.scatter_nd(self._pil_loc, ext_mat,
                                            [num_tx, num_tx_stream,
                                             outer_dim_size,
                                             inner_dim_size+1,
                                             inner_dim_size+1,
                                             batch_size, num_rx, num_rx_ant])
        ext_mat = tf.transpose(ext_mat, [5, 6, 7, 0, 1, 2, 3, 4])
        ext_mat = ext_mat[...,:inner_dim_size,:inner_dim_size]

        ################################################
        # Apply interpolation over the inner dimension
        ################################################

        # [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx,
        #  outer_dim_size, inner_dim_size]
        h_hat = tf.expand_dims(h_hat, axis=-1)
        h_hat = tf.matmul(ext_mat, h_hat)
        h_hat = tf.squeeze(h_hat, axis=-1)

        ##############################
        # Compute the error variances
        ##############################

        # Keep track of the previous estimation error variances for later use
        # [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx,
        #  outer_dim_size, inner_dim_size]
        err_var_old = err_var

        # [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx,
        #  outer_dim_size, inner_dim_size]
        cov_mat = expand_to_rank(self._cov_mat, 8, 0)
        err_var = tf.linalg.diag_part(cov_mat)
        err_var_mat = expand_to_rank(self._err_var_mat, 8, 0)
        err_var_mat = tf.transpose(err_var_mat, [0, 1, 2, 3, 4, 5, 7, 6])
        err_var = err_var - tf.reduce_sum(ext_mat*err_var_mat, axis=-1)
        err_var = tf.math.real(err_var)
        err_var = tf.maximum(err_var, self._rzero)

        #####################################
        # If this is *not* the last
        # interpolation step, scales the
        # input `h_hat` to ensure
        # it has the variance expected by the
        # next interpolation step.
        #
        # The error variance also `err_var`
        # is updated accordingly.
        #####################################
        if not self._last_step:
            #
            # Variance of h_hat
            #
            # Conjugate transpose of LMMSE matrix
            # [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx,
            #  outer_dim_size, inner_dim_size, inner_dim_size]
            ext_mat_h = tf.transpose(ext_mat, [0, 1, 2, 3, 4, 5, 7, 6],
                                     conjugate=True)
            # First part of the estimate covariance
            # [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx,
            #  outer_dim_size, inner_dim_size, inner_dim_size]
            h_hat_var_1 = tf.matmul(cov_mat, ext_mat_h)
            h_hat_var_1 = tf.transpose(h_hat_var_1, [0, 1, 2, 3, 4, 5, 7, 6])
            # [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx,
            #  outer_dim_size, inner_dim_size]
            h_hat_var_1 = tf.reduce_sum(ext_mat*h_hat_var_1, axis=-1)
            # Second part of the estimate covariance
            # [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx,
            #  outer_dim_size, inner_dim_size]
            err_var_old_c = tf.complex(err_var_old, self._rzero)
            err_var_old_c = tf.expand_dims(err_var_old_c, axis=-1)
            h_hat_var_2 = err_var_old_c*ext_mat_h
            h_hat_var_2 = tf.transpose(h_hat_var_2, [0, 1, 2, 3, 4, 5, 7, 6])
            h_hat_var_2 = tf.reduce_sum(ext_mat*h_hat_var_2, axis=-1)
            # Variance of h_hat
            # [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx,
            #  outer_dim_size, inner_dim_size]
            h_hat_var = h_hat_var_1 + h_hat_var_2
            # Scaling factor
            # [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx,
            #  outer_dim_size, inner_dim_size]
            err_var_c = tf.complex(err_var, self._rzero)
            h_var = tf.linalg.diag_part(cov_mat)
            s = tf.math.divide_no_nan(2.*h_var, h_hat_var + h_var - err_var_c)
            # Apply scaling to estimate
            # [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx,
            #  outer_dim_size, inner_dim_size]
            h_hat = s*h_hat
            # Updated variance
            # [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx,
            #  outer_dim_size, inner_dim_size]
            err_var = s*(s-1.)*h_hat_var + (1.-s)*h_var + s*err_var_c
            err_var = tf.math.real(err_var)
            err_var = tf.maximum(err_var, self._rzero)

        return h_hat, err_var

class SpatialChannelFilter:
    # pylint: disable=line-too-long
    r"""SpatialChannelFilter(cov_mat, last_step)

    Implements linear minimum mean square error (LMMSE) smoothing.

    We consider the following model:

    .. math::

        \mathbf{y} = \mathbf{h} + \mathbf{n}

    where :math:`\mathbf{y}\in\mathbb{C}^{M}` is the received signal vector,
    :math:`\mathbf{h}\in\mathbb{C}^{M}` is the channel vector to be estimated
    with covariance matrix
    :math:`\mathbb{E}\left[ \mathbf{h} \mathbf{h}^{\mathsf{H}} \right] = \mathbf{R}`,
    and :math:`\mathbf{n}\in\mathbb{C}^M` is a zero-mean noise vector whose
    elements have variance :math:`N_0`.

    The channel estimate :math:`\hat{\mathbf{h}}` is computed as

    .. math::

        \hat{\mathbf{h}} &= \mathbf{A} \mathbf{y}

    where

    .. math::

        \mathbf{A} = \mathbf{R} \left( \mathbf{R} + N_0 \mathbf{I}_M \right)^{-1}

    where :math:`\mathbf{I}_M` is the :math:`M \times M` identity matrix.
    The estimation error is:

    .. math::

        \tilde{h} = \mathbf{h} - \hat{\mathbf{h}}

    The error variances

    .. math::

             \sigma^2_i = \mathbb{E}\left[\tilde{h}_i \tilde{h}_i^\star \right], 0 \leq i \leq M-1

    are the diagonal elements of

    .. math::

        \mathbb{E}\left[\mathbf{\tilde{h}} \mathbf{\tilde{h}}^{\mathsf{H}} \right] = \mathbf{R} - \mathbf{A}\mathbf{R}.


    Note
    ----
    If you want to use this function in Graph mode with XLA, i.e., within
    a function that is decorated with ``@tf.function(jit_compile=True)``,
    you must set ``sionna.Config.xla_compat=true``.
    See :py:attr:`~sionna.Config.xla_compat`.

    Parameters
    ----------
    cov_mat : [num_rx_ant, num_rx_ant], tf.complex
        Spatial covariance matrix of the channel

    last_step : bool
        Set to `True` if this is the last interpolation step.
        Otherwise, set to `False`.
        If `True`, the the output is scaled to ensure its variance is as expected
        by the following interpolation step.

    Input
    -----
    h_hat : [batch_size, num_rx, num_tx, num_streams_per_tx, num_ofdm_symbols, num_subcarriers, num_rx_ant], tf.complex
        Channel estimates.

    err_var : [batch_size, num_rx, num_tx, num_streams_per_tx, num_ofdm_symbols, num_subcarriers, num_rx_ant], tf.float
        Channel estimation error variances.

    Output
    ------
    h_hat : [batch_size, num_rx, num_tx, num_streams_per_tx, num_ofdm_symbols, num_subcarriers, num_rx_ant], tf.complex
        Channel estimates smoothed accross the spatial dimension

    err_var : [batch_size, num_rx, num_tx, num_streams_per_tx, num_ofdm_symbols, num_subcarriers, num_rx_ant], tf.float
        The channel estimation error variances of the smoothed channel estimates.
    """

    def __init__(self, cov_mat, last_step):
        self._rzero = tf.zeros((), cov_mat.dtype.real_dtype)
        self._cov_mat = cov_mat
        self._last_step = last_step

        # Indices for adding a tensor of vectors [..., num_rx_ant] to the
        # diagonal of a tensor of matrices [..., num_rx_ant, num_rx_ant]
        num_rx_ant = cov_mat.shape[0]
        add_diag_indices = [[rxa, rxa] for rxa in range(num_rx_ant)]
        self._add_diag_indices = tf.cast(add_diag_indices, tf.int32)

    def __call__(self, h_hat, err_var):
        # h_hat : [batch_size, num_rx, num_tx, num_streams_per_tx,
        #           num_ofdm_symbols, num_subcarriers, num_rx_ant]
        # err_var : [batch_size, num_rx, num_tx, num_streams_per_tx,
        #           num_ofdm_symbols, num_subcarriers, num_rx_ant]

        # [..., num_rx_ant]
        err_var = tf.complex(err_var, self._rzero)
        # Keep track of the previous estimation error variances for later use
        err_var_old = err_var

        # [num_rx_ant, num_rx_ant]
        cov_mat = self._cov_mat
        cov_mat_t = tf.transpose(cov_mat)
        num_rx_ant = tf.shape(cov_mat)[0]

        ##########################################
        # Compute LMMSE matrix
        ##########################################

        # [..., num_rx_ant, num_rx_ant]
        cov_mat = expand_to_rank(cov_mat, tf.rank(err_var)+1, axis=0)

        # Adding the error variances to the diagonal
        # [..., num_rx_ant, num_rx_ant]
        lmmse_mat = tf.broadcast_to(cov_mat, tf.concat([tf.shape(err_var),
                                                        [num_rx_ant]], axis=0))
        # [num_rx_ant, ...]
        err_var_ = tf.transpose(err_var, [6, 0, 1, 2, 3, 4, 5])
        # [num_rx_ant, num_rx_ant, ...]
        lmmse_mat = tf.transpose(lmmse_mat, [6, 7, 0, 1, 2, 3, 4, 5])
        lmmse_mat = tf.tensor_scatter_nd_add(lmmse_mat,
                                            self._add_diag_indices, err_var_)
        # [..., num_rx_ant, num_rx_ant]
        lmmse_mat = tf.transpose(lmmse_mat, [2, 3, 4, 5, 6, 7, 0, 1])

        # [..., num_rx_ant, num_rx_ant]
        lmmse_mat = matrix_inv(lmmse_mat)
        lmmse_mat = tf.matmul(cov_mat, lmmse_mat)

        ##########################################
        # Apply smoothing
        ##########################################

        # [..., num_rx_ant, 1]
        h_hat = tf.expand_dims(h_hat, axis=-1)
        # [..., num_rx_ant]
        h_hat = tf.squeeze(tf.matmul(lmmse_mat, h_hat), axis=-1)

        ##########################################
        # Compute the estimation error variances
        ##########################################

        # [..., num_rx_ant, num_rx_ant]
        cov_mat_t = expand_to_rank(cov_mat_t, tf.rank(lmmse_mat), axis=0)
        # [..., num_rx_ant]
        err_var = tf.reduce_sum(cov_mat_t*lmmse_mat, axis=-1)
        # [..., num_rx_ant]
        err_var = tf.linalg.diag_part(cov_mat) - err_var
        err_var = tf.math.real(err_var)
        err_var = tf.maximum(err_var, self._rzero)

        ##########################################
        # If this is *not* the last
        # interpolation step, scales the
        # input `h_hat` to ensure
        # it has the variance expected by the
        # next interpolation step.
        #
        # The error variance also `err_var`
        # is updated accordingly.
        ##########################################
        if not self._last_step:
            #
            # Variance of h_hat
            #
            # Conjugate transpose of the LMMSE matrix
            # [..., num_rx_ant, num_rx_ant]
            lmmse_mat_h = tf.transpose(lmmse_mat, [0, 1, 2, 3, 4, 5, 7, 6],
                                        conjugate=True)
            # First part of the estimate covariance
            # [..., num_rx_ant, num_rx_ant]
            h_hat_var_1 = tf.matmul(cov_mat, lmmse_mat_h)
            h_hat_var_1 = tf.transpose(h_hat_var_1, [0, 1, 2, 3, 4, 5, 7, 6])
            # [..., num_rx_ant]
            h_hat_var_1 = tf.reduce_sum(lmmse_mat*h_hat_var_1, axis=-1)
            # Second part of the estimate covariance
            # [..., num_rx_ant, 1]
            err_var_old = tf.expand_dims(err_var_old, axis=-1)
            # [..., num_rx_ant, num_rx_ant]
            h_hat_var_2 = err_var_old*lmmse_mat_h
            # [..., num_rx_ant, num_rx_ant]
            h_hat_var_2 = tf.transpose(h_hat_var_2, [0, 1, 2, 3, 4, 5, 7, 6])
            # [..., num_rx_ant]
            h_hat_var_2 = tf.reduce_sum(lmmse_mat*h_hat_var_2, axis=-1)
            # Variance of h_hat
            # [..., num_rx_ant]
            h_hat_var = h_hat_var_1 + h_hat_var_2
            # Scaling factor
            # [..., num_rx_ant]
            err_var_c = tf.complex(err_var, self._rzero)
            h_var = tf.linalg.diag_part(cov_mat)
            s = tf.math.divide_no_nan(2.*h_var, h_hat_var + h_var - err_var_c)
            # Apply scaling to estimate
            # [..., num_rx_ant]
            h_hat = s*h_hat
            # Updated variance
            # [..., num_rx_ant]
            err_var = s*(s-1.)*h_hat_var + (1.-s)*h_var + s*err_var_c
            err_var = tf.math.real(err_var)
            err_var = tf.maximum(err_var, self._rzero)

        return h_hat, err_var


class LMMSEInterpolator(BaseChannelInterpolator):
    # pylint: disable=line-too-long
    r"""LMMSEInterpolator(pilot_pattern, cov_mat_time, cov_mat_freq, cov_mat_space=None, order='t-f')

    LMMSE interpolation on a resource grid with optional spatial smoothing.

    This class computes for each element of an OFDM resource grid
    a channel estimate and error variance
    through linear minimum mean square error (LMMSE) interpolation/smoothing.
    It is assumed that the measurements were taken at the nonzero positions
    of a :class:`~sionna.ofdm.PilotPattern`.

    Depending on the value of ``order``, the interpolation is carried out
    accross time (t), i.e., OFDM symbols, frequency (f), i.e., subcarriers,
    and optionally space (s), i.e., receive antennas, in any desired order.

    For simplicity, we describe the underlying algorithm assuming that interpolation
    across the sub-carriers is performed first, followed by interpolation across
    OFDM symbols, and finally by spatial smoothing across receive
    antennas.
    The algorithm is similar if interpolation and/or smoothing are performed in
    a different order.
    For clarity, antenna indices are omitted when describing frequency and time
    interpolation, as the same process is applied to all the antennas.

    The input ``h_hat`` is first reshaped to a resource grid
    :math:`\hat{\mathbf{H}} \in \mathbb{C}^{N \times M}`, by scattering the channel
    estimates at pilot locations according to the ``pilot_pattern``. :math:`N`
    denotes the number of OFDM symbols and :math:`M` the number of sub-carriers.

    The first pass consists in interpolating across the sub-carriers:

    .. math::
        \hat{\mathbf{h}}_n^{(1)} = \mathbf{A}_n \hat{\mathbf{h}}_n

    where :math:`1 \leq n \leq N` is the OFDM symbol index and :math:`\hat{\mathbf{h}}_n` is
    the :math:`n^{\text{th}}` (transposed) row of :math:`\hat{\mathbf{H}}`.
    :math:`\mathbf{A}_n` is the :math:`M \times M` matrix such that:

    .. math::
        \mathbf{A}_n = \bar{\mathbf{A}}_n \mathbf{\Pi}_n^\intercal

    where

    .. math::
        \bar{\mathbf{A}}_n = \underset{\mathbf{Z} \in \mathbb{C}^{M \times K_n}}{\text{argmin}} \left\lVert \mathbf{Z}\left( \mathbf{\Pi}_n^\intercal \mathbf{R^{(f)}} \mathbf{\Pi}_n + \mathbf{\Sigma}_n \right) - \mathbf{R^{(f)}} \mathbf{\Pi}_n \right\rVert_{\text{F}}^2

    and :math:`\mathbf{R^{(f)}}` is the :math:`M \times M` channel frequency covariance matrix,
    :math:`\mathbf{\Pi}_n` the :math:`M \times K_n` matrix that spreads :math:`K_n`
    values to a vector of size :math:`M` according to the ``pilot_pattern`` for the :math:`n^{\text{th}}` OFDM symbol,
    and :math:`\mathbf{\Sigma}_n \in \mathbb{R}^{K_n \times K_n}` is the channel estimation error covariance built from
    ``err_var`` and assumed to be diagonal.
    Computation of :math:`\bar{\mathbf{A}}_n` is done using an algorithm based on complete orthogonal decomposition.
    This is done to avoid matrix inversion for badly conditioned covariance matrices.

    The channel estimation error variances after the first interpolation pass are computed as

    .. math::
        \mathbf{\Sigma}^{(1)}_n = \text{diag} \left( \mathbf{R^{(f)}} - \mathbf{A}_n \mathbf{\Xi}_n \mathbf{R^{(f)}} \right)

    where :math:`\mathbf{\Xi}_n` is the diagonal matrix of size :math:`M \times M` that zeros the
    columns corresponding to sub-carriers not carrying any pilots.
    Note that interpolation is not performed for OFDM symbols which do not carry pilots.

    **Remark**: The interpolation matrix differs across OFDM symbols as different
    OFDM symbols may carry pilots on different sub-carriers and/or have different
    estimation error variances.

    Scaling of the estimates is then performed to ensure that their
    variances match the ones expected by the next interpolation step, and the error variances are updated accordingly:

    .. math::
        \begin{align}
            \left[\hat{\mathbf{h}}_n^{(2)}\right]_m &= s_{n,m} \left[\hat{\mathbf{h}}_n^{(1)}\right]_m\\
            \left[\mathbf{\Sigma}^{(2)}_n\right]_{m,m}  &= s_{n,m}\left( s_{n,m}-1 \right) \left[\hat{\mathbf{\Sigma}}^{(1)}_n\right]_{m,m} + \left( 1 - s_{n,m} \right) \left[\mathbf{R^{(f)}}\right]_{m,m} + s_{n,m} \left[\mathbf{\Sigma}^{(1)}_n\right]_{m,m}
        \end{align}

    where the scaling factor :math:`s_{n,m}` is such that:


    .. math::
        \mathbb{E} \left\{ \left\lvert s_{n,m} \left[\hat{\mathbf{h}}_n^{(1)}\right]_m \right\rvert^2 \right\} = \left[\mathbf{R^{(f)}}\right]_{m,m} +  \mathbb{E} \left\{ \left\lvert s_{n,m} \left[\hat{\mathbf{h}}^{(1)}_n\right]_m - \left[\mathbf{h}_n\right]_m \right\rvert^2 \right\}

    which leads to:

    .. math::
        \begin{align}
            s_{n,m} &= \frac{2 \left[\mathbf{R^{(f)}}\right]_{m,m}}{\left[\mathbf{R^{(f)}}\right]_{m,m} - \left[\mathbf{\Sigma}^{(1)}_n\right]_{m,m} + \left[\hat{\mathbf{\Sigma}}^{(1)}_n\right]_{m,m}}\\
            \hat{\mathbf{\Sigma}}^{(1)}_n &= \mathbf{A}_n \mathbf{R^{(f)}} \mathbf{A}_n^{\mathrm{H}}.
        \end{align}

    The second pass consists in interpolating across the OFDM symbols:

    .. math::
        \hat{\mathbf{h}}_m^{(3)} = \mathbf{B}_m \tilde{\mathbf{h}}^{(2)}_m

    where :math:`1 \leq m \leq M` is the sub-carrier index and :math:`\tilde{\mathbf{h}}^{(2)}_m` is
    the :math:`m^{\text{th}}` column of

    .. math::
        \hat{\mathbf{H}}^{(2)} = \begin{bmatrix}
                                    {\hat{\mathbf{h}}_1^{(2)}}^\intercal\\
                                    \vdots\\
                                    {\hat{\mathbf{h}}_N^{(2)}}^\intercal
                                 \end{bmatrix}

    and :math:`\mathbf{B}_m` is the :math:`N \times N` interpolation LMMSE matrix:

    .. math::
        \mathbf{B}_m = \bar{\mathbf{B}}_m \tilde{\mathbf{\Pi}}_m^\intercal

    where

    .. math::
        \bar{\mathbf{B}}_m = \underset{\mathbf{Z} \in \mathbb{C}^{N \times L_m}}{\text{argmin}} \left\lVert \mathbf{Z} \left( \tilde{\mathbf{\Pi}}_m^\intercal \mathbf{R^{(t)}}\tilde{\mathbf{\Pi}}_m + \tilde{\mathbf{\Sigma}}^{(2)}_m \right) -  \mathbf{R^{(t)}}\tilde{\mathbf{\Pi}}_m \right\rVert_{\text{F}}^2

    where :math:`\mathbf{R^{(t)}}` is the :math:`N \times N` channel time covariance matrix,
    :math:`\tilde{\mathbf{\Pi}}_m` the :math:`N \times L_m` matrix that spreads :math:`L_m`
    values to a vector of size :math:`N` according to the ``pilot_pattern`` for the :math:`m^{\text{th}}` sub-carrier,
    and :math:`\tilde{\mathbf{\Sigma}}^{(2)}_m \in \mathbb{R}^{L_m \times L_m}` is the diagonal matrix of channel estimation error variances
    built by gathering the error variances from (:math:`\mathbf{\Sigma}^{(2)}_1,\dots,\mathbf{\Sigma}^{(2)}_N`) corresponding
    to resource elements carried by the :math:`m^{\text{th}}` sub-carrier.
    Computation of :math:`\bar{\mathbf{B}}_m` is done using an algorithm based on complete orthogonal decomposition.
    This is done to avoid matrix inversion for badly conditioned covariance matrices.

    The resulting channel estimate for the resource grid is

    .. math::
        \hat{\mathbf{H}}^{(3)} = \left[ \hat{\mathbf{h}}_1^{(3)} \dots \hat{\mathbf{h}}_M^{(3)} \right]

    The resulting channel estimation error variances are the diagonal coefficients of the matrices

    .. math::
        \mathbf{\Sigma}^{(3)}_m = \mathbf{R^{(t)}} - \mathbf{B}_m \tilde{\mathbf{\Xi}}_m \mathbf{R^{(t)}}, 1 \leq m \leq M

    where :math:`\tilde{\mathbf{\Xi}}_m` is the diagonal matrix of size :math:`N \times N` that zeros the
    columns corresponding to OFDM symbols not carrying any pilots.

    **Remark**: The interpolation matrix differs across sub-carriers as different
    sub-carriers may have different estimation error variances computed by the first
    pass.
    However, all sub-carriers carry at least one channel estimate as a result of
    the first pass, ensuring that a channel estimate is computed for all the resource
    elements after the second pass.

    **Remark:** LMMSE interpolation requires knowledge of the time and frequency
    covariance matrices of the channel. The notebook `OFDM MIMO Channel Estimation and Detection <../examples/OFDM_MIMO_Detection.ipynb>`_ shows how to estimate
    such matrices for arbitrary channel models.
    Moreover, the functions :func:`~sionna.ofdm.tdl_time_cov_mat`
    and :func:`~sionna.ofdm.tdl_freq_cov_mat` compute the expected time and frequency
    covariance matrices, respectively, for the :class:`~sionna.channel.tr38901.TDL` channel models.

    Scaling of the estimates is then performed to ensure that their
    variances match the ones expected by the next smoothing step, and the
    error variances are updated accordingly:

    .. math::
        \begin{align}
            \left[\hat{\mathbf{h}}_m^{(4)}\right]_n &= \gamma_{m,n} \left[\hat{\mathbf{h}}_m^{(3)}\right]_n\\
            \left[\mathbf{\Sigma}^{(4)}_m\right]_{n,n}  &= \gamma_{m,n}\left( \gamma_{m,n}-1 \right) \left[\hat{\mathbf{\Sigma}}^{(3)}_m\right]_{n,n} + \left( 1 - \gamma_{m,n} \right) \left[\mathbf{R^{(t)}}\right]_{n,n} + \gamma_{m,n} \left[\mathbf{\Sigma}^{(3)}_n\right]_{m,m}
        \end{align}

    where:

    .. math::
        \begin{align}
            \gamma_{m,n} &= \frac{2 \left[\mathbf{R^{(t)}}\right]_{n,n}}{\left[\mathbf{R^{(t)}}\right]_{n,n} - \left[\mathbf{\Sigma}^{(3)}_m\right]_{n,n} + \left[\hat{\mathbf{\Sigma}}^{(3)}_n\right]_{m,m}}\\
            \hat{\mathbf{\Sigma}}^{(3)}_m &= \mathbf{B}_m \mathbf{R^{(t)}} \mathbf{B}_m^{\mathrm{H}}
        \end{align}

    Finally, a spatial smoothing step is applied to every resource element carrying
    a channel estimate.
    For clarity, we drop the resource element indexing :math:`(n,m)`.
    We denote by :math:`L` the number of receive antennas, and by
    :math:`\mathbf{R^{(s)}}\in\mathbb{C}^{L \times L}` the spatial covariance matrix.

    LMMSE spatial smoothing consists in the following computations:

    .. math::
        \hat{\mathbf{h}}^{(5)} = \mathbf{C} \hat{\mathbf{h}}^{(4)}

    where

    .. math::
        \mathbf{C} = \mathbf{R^{(s)}} \left( \mathbf{R^{(s)}} + \mathbf{\Sigma}^{(4)} \right)^{-1}.

    The estimation error variances are the digonal coefficients of

    .. math::
        \mathbf{\Sigma}^{(5)} = \mathbf{R^{(s)}} - \mathbf{C}\mathbf{R^{(s)}}

    The smoothed channel estimate :math:`\hat{\mathbf{h}}^{(5)}` and corresponding
    error variances :math:`\text{diag}\left( \mathbf{\Sigma}^{(5)} \right)` are
    returned for every resource element :math:`(m,n)`.

    **Remark:** No scaling is performed after the last interpolation or smoothing
    step.

    **Remark:** All passes assume that the estimation error covariance matrix
    (:math:`\mathbf{\Sigma}`, :math:`\tilde{\mathbf{\Sigma}}^{(2)}`, or :math:`\tilde{\mathbf{\Sigma}}^{(4)}`) is diagonal, which
    may not be accurate. When this assumption does not hold, this interpolator is only
    an approximation of LMMSE interpolation.

    **Remark:** The order in which frequency interpolation, temporal
    interpolation, and, optionally, spatial smoothing are applied, is controlled using the
    ``order`` parameter.

    Note
    ----
    This layer does not support graph mode with XLA.

    Parameters
    ----------
    pilot_pattern : PilotPattern
        An instance of :class:`~sionna.ofdm.PilotPattern`

    cov_mat_time : [num_ofdm_symbols, num_ofdm_symbols], tf.complex
        Time covariance matrix of the channel

    cov_mat_freq : [fft_size, fft_size], tf.complex
        Frequency covariance matrix of the channel

    cov_time_space : [num_rx_ant, num_rx_ant], tf.complex
        Spatial covariance matrix of the channel.
        Defaults to `None`.
        Only required if spatial smoothing is requested (see ``order``).

    order : str
        Order in which to perform interpolation and optional smoothing.
        For example, ``"t-f-s"`` means that interpolation across the OFDM symbols
        is performed first (``"t"``: time), followed by interpolation across the
        sub-carriers (``"f"``: frequency), and finally smoothing across the
        receive antennas (``"s"``: space).
        Similarly, ``"f-t"`` means interpolation across the sub-carriers followed
        by interpolation across the OFDM symbols and no spatial smoothing.
        The spatial covariance matrix (``cov_time_space``) is only required when
        spatial smoothing is requested.
        Time and frequency interpolation are not optional to ensure that a channel
        estimate is computed for all resource elements.

    Input
    -----
    h_hat : [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_pilot_symbols], tf.complex
        Channel estimates for the pilot-carrying resource elements

    err_var : [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_pilot_symbols], tf.complex
        Channel estimation error variances for the pilot-carrying resource elements

    Output
    ------
    h_hat : [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size], tf.complex
        Channel estimates accross the entire resource grid for all
        transmitters and streams

    err_var : Same shape as ``h_hat``, tf.float
        Channel estimation error variances accross the entire resource grid
        for all transmitters and streams
    """

    def __init__(self, pilot_pattern, cov_mat_time, cov_mat_freq,
                    cov_mat_space=None, order='t-f'):

        # Check the specified order
        order = order.split('-')
        assert 2 <= len(order) <= 3, "Invalid order for interpolation."
        spatial_smoothing = False
        freq_smoothing = False
        time_smoothing = False
        for o in order:
            assert o in ('s', 'f', 't'), f"Uknown dimension {o}"
            if o == 's':
                assert not spatial_smoothing,\
                    "Spatial smoothing can be specified at most once"
                spatial_smoothing = True
            elif o == 't':
                assert not time_smoothing,\
                    "Temporal interpolation can be specified once only"
                time_smoothing = True
            elif o == 'f':
                assert not freq_smoothing,\
                    "Frequency interpolation can be specified once only"
                freq_smoothing = True
        if spatial_smoothing:
            assert cov_mat_space is not None,\
                "A spatial covariance matrix is required for spatial smoothing"
        assert freq_smoothing, "Frequency interpolation is required"
        assert time_smoothing, "Time interpolation is required"

        self._order = order
        self._num_ofdm_symbols = pilot_pattern.num_ofdm_symbols
        self._num_effective_subcarriers =pilot_pattern.num_effective_subcarriers

        # Build pilot masks for every stream
        pilot_mask = self._build_pilot_mask(pilot_pattern)

        # Build indices for mapping channel estimates and
        # error variances that are given as input to a
        # resource grid
        num_pilots = pilot_pattern.pilots.shape[2]
        inputs_to_rg_indices = self._build_inputs2rg_indices(pilot_mask,
                                                             num_pilots)
        self._inputs_to_rg_indices = tf.cast(inputs_to_rg_indices, tf.int32)

        # 1D interpolator according to requested order
        # Interpolation is always performed along the inner dimension.
        interpolators = []
        # Masks for masking error variances that were not updated
        err_var_masks = []
        for i, o in enumerate(order):
            # Is it the last one?
            last_step = (i == len(order)-1)
            # Frequency
            if o == "f":
                interpolator = LMMSEInterpolator1D(pilot_mask, cov_mat_freq,
                                                        last_step=last_step)
                pilot_mask = self._update_pilot_mask_interp(pilot_mask)
                err_var_mask = tf.cast(pilot_mask == 1,
                                        cov_mat_freq.dtype.real_dtype)
            # Time
            elif o == 't':
                pilot_mask = tf.transpose(pilot_mask, [0, 1, 3, 2])
                interpolator = LMMSEInterpolator1D(pilot_mask, cov_mat_time,
                                                        last_step=last_step)
                pilot_mask = self._update_pilot_mask_interp(pilot_mask)
                pilot_mask = tf.transpose(pilot_mask, [0, 1, 3, 2])
                err_var_mask = tf.cast(pilot_mask == 1,
                                            cov_mat_freq.dtype.real_dtype)
            # Space
            elif o == 's':
                interpolator = SpatialChannelFilter(cov_mat_space,
                                                    last_step=last_step)
                err_var_mask = tf.cast(pilot_mask == 1,
                                            cov_mat_freq.dtype.real_dtype)
            interpolators.append(interpolator)
            err_var_masks.append(err_var_mask)
        self._interpolators = interpolators
        self._err_var_masks = err_var_masks

    def _build_pilot_mask(self, pilot_pattern):
        """
        Build for every transmitter and stream a pilot mask indicating
        which REs are allocated to pilots, data, or not used.
        # 0 -> Data
        # 1 -> Pilot
        # 2 -> Not used
        """

        mask = pilot_pattern.mask
        pilots = pilot_pattern.pilots
        num_tx = mask.shape[0]
        num_streams_per_tx = mask.shape[1]
        num_ofdm_symbols = mask.shape[2]
        num_effective_subcarriers = mask.shape[3]

        pilot_mask = np.zeros([num_tx, num_streams_per_tx, num_ofdm_symbols,
                                num_effective_subcarriers], int)
        for tx,st in itertools.product( range(num_tx),
                                        range(num_streams_per_tx)):
            pil_index = 0
            for sb,sc in itertools.product( range(num_ofdm_symbols),
                                            range(num_effective_subcarriers)):
                if mask[tx,st,sb,sc] == 1:
                    if np.abs(pilots[tx,st,pil_index]) > 0.0:
                        pilot_mask[tx,st,sb,sc] = 1
                    else:
                        pilot_mask[tx,st,sb,sc] = 2
                    pil_index += 1

        return pilot_mask

    def _build_inputs2rg_indices(self, pilot_mask, num_pilots):
        """
        Builds indices for mapping channel estimates and
        error variances that are given as input to a
        resource grid
        """

        num_tx = pilot_mask.shape[0]
        num_streams_per_tx = pilot_mask.shape[1]
        num_ofdm_symbols = pilot_mask.shape[2]
        num_effective_subcarriers = pilot_mask.shape[3]

        inputs_to_rg_indices = np.zeros([num_tx, num_streams_per_tx,
                                         num_pilots, 4], int)
        for tx,st in itertools.product( range(num_tx),
                                        range(num_streams_per_tx)):
            pil_index = 0 # Pilot index for this stream
            for sb,sc in itertools.product( range(num_ofdm_symbols),
                                            range(num_effective_subcarriers)):
                if pilot_mask[tx,st,sb,sc] == 0:
                    continue
                if pilot_mask[tx,st,sb,sc] == 1:
                    inputs_to_rg_indices[tx, st, pil_index] = [tx, st, sb, sc]
                pil_index += 1

        return inputs_to_rg_indices

    def _update_pilot_mask_interp(self, pilot_mask):
        """
        Update the pilot mask to label the resource elements for which the
        channel was interpolated.
        """

        interpolated = np.any(pilot_mask == 1, axis=-1, keepdims=True)
        pilot_mask = np.where(interpolated, 1, pilot_mask)

        return pilot_mask

    def __call__(self, h_hat, err_var):

        # h_hat : [batch_size, num_rx, num_rx_ant, num_tx,
        #          num_streams_per_tx, num_pilots]
        # err_var : [batch_size, num_rx, num_rx_ant, num_tx,
        #          num_streams_per_tx, num_pilots]

        batch_size = tf.shape(h_hat)[0]
        num_rx = tf.shape(h_hat)[1]
        num_rx_ant = tf.shape(h_hat)[2]
        num_tx = tf.shape(h_hat)[3]
        num_tx_stream = tf.shape(h_hat)[4]
        num_ofdm_symbols = self._num_ofdm_symbols
        num_effective_subcarriers = self._num_effective_subcarriers

        # For some estimator, err_var might not have the same shape
        # as h_hat
        err_var = tf.broadcast_to(err_var, tf.shape(h_hat))

        # Mapping the channel estimates and error variances to a resource grid
        # all : [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx,
        #           num_ofdm_symbols, num_effective_subcarriers]
        h_hat = tf.transpose(h_hat, [3, 4, 5, 0, 1, 2])
        err_var = tf.transpose(err_var, [3, 4, 5, 0, 1, 2])
        h_hat = tf.scatter_nd(self._inputs_to_rg_indices, h_hat,
                                            [num_tx, num_tx_stream,
                                             num_ofdm_symbols,
                                             num_effective_subcarriers,
                                             batch_size, num_rx, num_rx_ant])
        err_var = tf.scatter_nd(self._inputs_to_rg_indices, err_var,
                                            [num_tx, num_tx_stream,
                                             num_ofdm_symbols,
                                             num_effective_subcarriers,
                                             batch_size, num_rx, num_rx_ant])
        h_hat = tf.transpose(h_hat, [4, 5, 6, 0, 1, 2, 3])
        err_var = tf.transpose(err_var, [4, 5, 6, 0, 1, 2, 3])

        # Interpolation
        # Performed according to the requested order. Transpose are used as
        # 1D interpolation is performed along the inner axis.
        items = zip(self._order, self._interpolators, self._err_var_masks)
        for o,interp,err_var_mask in items:
            # Frequency
            if o == 'f':
                # [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx,
                #           num_ofdm_symbols, num_effective_subcarriers]
                h_hat, err_var = interp(h_hat, err_var)
                err_var_mask = expand_to_rank(err_var_mask, tf.rank(err_var), 0)
                err_var = err_var*err_var_mask
            # Time
            elif o == 't':
                # [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx,
                #           num_effective_subcarriers, num_ofdm_symbols]
                h_hat = tf.transpose(h_hat, [0, 1, 2, 3, 4, 6, 5])
                err_var = tf.transpose(err_var, [0, 1, 2, 3, 4, 6, 5])
                h_hat, err_var = interp(h_hat, err_var)
                # [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx,
                #           num_ofdm_symbols, num_effective_subcarriers]
                h_hat = tf.transpose(h_hat, [0, 1, 2, 3, 4, 6, 5])
                err_var = tf.transpose(err_var, [0, 1, 2, 3, 4, 6, 5])
                err_var_mask = expand_to_rank(err_var_mask, tf.rank(err_var), 0)
                err_var = err_var*err_var_mask
            # Space
            elif o == 's':
                # [batch_size, num_rx, num_tx, num_streams_per_tx,
                #      num_ofdm_symbols, num_effective_subcarriers, num_rx_ant]
                h_hat = tf.transpose(h_hat, [0, 1, 3, 4, 5, 6, 2])
                err_var = tf.transpose(err_var, [0, 1, 3, 4, 5, 6, 2])
                h_hat, err_var = interp(h_hat, err_var)
                # [batch_size, num_rx, num_tx, num_streams_per_tx,
                #      num_ofdm_symbols, num_effective_subcarriers, num_rx_ant]
                h_hat = tf.transpose(h_hat, [0, 1, 6, 2, 3, 4, 5])
                err_var = tf.transpose(err_var, [0, 1, 6, 2, 3, 4, 5])
                err_var_mask = expand_to_rank(err_var_mask, tf.rank(err_var), 0)
                err_var = err_var*err_var_mask

        return h_hat, err_var

#######################################################
# Utilities
#######################################################

def tdl_freq_cov_mat(model, subcarrier_spacing, fft_size, delay_spread,
                        dtype=tf.complex64):
    # pylint: disable=line-too-long
    r"""
    Computes the frequency covariance matrix of a
    :class:`~sionna.channel.tr38901.TDL` channel model.

    The channel frequency covariance matrix :math:`\mathbf{R}^{(f)}` of a TDL channel model is

    .. math::
        \mathbf{R}^{(f)}_{u,v} = \sum_{\ell=1}^L P_\ell e^{-j 2 \pi \tau_\ell \Delta_f (u-v)}, 1 \leq u,v \leq M

    where :math:`M` is the FFT size, :math:`L` is the number of paths for the selected TDL model,
    :math:`P_\ell` and :math:`\tau_\ell` are the average power and delay for the
    :math:`\ell^{\text{th}}` path, respectively, and :math:`\Delta_f` is the sub-carrier spacing.

    Input
    ------
    model : str
        TDL model for which to return the covariance matrix.
        Should be one of "A", "B", "C", "D", or "E".

    subcarrier_spacing : float
        Sub-carrier spacing [Hz]

    fft_size : float
        FFT size

    delay_spread : float
        Delay spread [s]

    dtype : tf.DType
        Datatype to use for the output.
        Should be one of `tf.complex64` or `tf.complex128`.
        Defaults to `tf.complex64`.

    Output
    ------
        cov_mat : [fft_size, fft_size], tf.complex
            Channel frequency covariance matrix
    """

    assert dtype in (tf.complex64, tf.complex128),\
        "The `dtype` should be a complex datatype"

    #
    # Load the power delay profile
    #

    # Set the file from which to load the model
    assert model in ('A', 'B', 'C', 'D', 'E'), "Invalid TDL model"
    if model == 'A':
        parameters_fname = "TDL-A.json"
    elif model == 'B':
        parameters_fname = "TDL-B.json"
    elif model == 'C':
        parameters_fname = "TDL-C.json"
    elif model == 'D':
        parameters_fname = "TDL-D.json"
    elif model == 'E':
        parameters_fname = "TDL-E.json"
    source = files(models).joinpath(parameters_fname)
    # pylint: disable=unspecified-encoding
    with open(source) as parameter_file:
        params = json.load(parameter_file)
    # LoS scenario ?
    los = bool(params['los'])
    # Retrieve power and delays
    delays = np.array(params['delays'])*delay_spread
    mean_powers = np.power(10.0, np.array(params['powers'])/10.0)

    if los:
        # Add the power of the specular and non-specular component of
        # the first path
        mean_powers[0] = mean_powers[0] + mean_powers[1]
        mean_powers = np.concatenate([mean_powers[:1], mean_powers[2:]], axis=0)
        # The first two paths have 0 delays as they correspond to the
        # specular and reflected components of the first path.
        delays = delays[1:]

    # Normalize the PDP
    norm_factor = np.sum(mean_powers)
    mean_powers = mean_powers / norm_factor

    #
    # Build frequency covariance matrix
    #

    n = np.arange(fft_size)
    p = -2.*np.pi*subcarrier_spacing*n
    p = np.expand_dims(p, axis=0)
    delays = np.expand_dims(delays, axis=1)
    p = p*delays
    p = np.exp(1j*p)
    p = np.expand_dims(p, axis=-1)
    cov_mat = np.matmul(p, np.transpose(np.conj(p), [0, 2, 1]))
    mean_powers = np.expand_dims(mean_powers, axis=(1,2))
    cov_mat = np.sum(mean_powers*cov_mat, axis=0)

    return tf.cast(cov_mat, dtype)

def tdl_time_cov_mat(model, speed, carrier_frequency, ofdm_symbol_duration,
        num_ofdm_symbols, los_angle_of_arrival=PI/4., dtype=tf.complex64):
    # pylint: disable=line-too-long
    r"""
    Computes the time covariance matrix of a
    :class:`~sionna.channel.tr38901.TDL` channel model.

    For non-line-of-sight (NLoS) model, the channel time covariance matrix
    :math:`\mathbf{R^{(t)}}` of a TDL channel model is

    .. math::
        \mathbf{R^{(t)}}_{u,v} = J_0 \left( \nu \Delta_t \left( u-v \right) \right)

    where :math:`J_0` is the zero-order Bessel function of the first kind,
    :math:`\Delta_t` the duration of an OFDM symbol, and :math:`\nu` the Doppler
    spread defined by

    .. math::
        \nu = 2 \pi \frac{v}{c} f_c

    where :math:`v` is the movement speed, :math:`c` the speed of light, and
    :math:`f_c` the carrier frequency.

    For line-of-sight (LoS) channel models, the channel time covariance matrix
    is

    .. math::
        \mathbf{R^{(t)}}_{u,v} = P_{\text{NLoS}} J_0 \left( \nu \Delta_t \left( u-v \right) \right) + P_{\text{LoS}}e^{j \nu \Delta_t \left( u-v \right) \cos{\alpha_{\text{LoS}}}}

    where :math:`\alpha_{\text{LoS}}` is the angle-of-arrival for the LoS path,
    :math:`P_{\text{NLoS}}` the total power of NLoS paths, and
    :math:`P_{\text{LoS}}` the power of the LoS path. The power delay profile
    is assumed to have unit power, i.e., :math:`P_{\text{NLoS}} + P_{\text{LoS}} = 1`.

    Input
    ------
    model : str
        TDL model for which to return the covariance matrix.
        Should be one of "A", "B", "C", "D", or "E".

    speed : float
        Speed [m/s]

    carrier_frequency : float
        Carrier frequency [Hz]

    ofdm_symbol_duration : float
        Duration of an OFDM symbol [s]

    num_ofdm_symbols : int
        Number of OFDM symbols

    los_angle_of_arrival : float
        Angle-of-arrival for LoS path [radian]. Only used with LoS models.
        Defaults to :math:`\pi/4`.

    dtype : tf.DType
        Datatype to use for the output.
        Should be one of `tf.complex64` or `tf.complex128`.
        Defaults to `tf.complex64`.

    Output
    ------
        cov_mat : [num_ofdm_symbols, num_ofdm_symbols], tf.complex
            Channel time covariance matrix
    """

    # Doppler spread
    doppler_spread = 2.*PI*speed/SPEED_OF_LIGHT*carrier_frequency

    #
    # Load the power delay profile
    #

    # Set the file from which to load the model
    assert model in ('A', 'B', 'C', 'D', 'E'), "Invalid TDL model"
    if model == 'A':
        parameters_fname = "TDL-A.json"
    elif model == 'B':
        parameters_fname = "TDL-B.json"
    elif model == 'C':
        parameters_fname = "TDL-C.json"
    elif model == 'D':
        parameters_fname = "TDL-D.json"
    elif model == 'E':
        parameters_fname = "TDL-E.json"
    source = files(models).joinpath(parameters_fname)
    # pylint: disable=unspecified-encoding
    with open(source) as parameter_file:
        params = json.load(parameter_file)
    # LoS scenario ?
    los = bool(params['los'])
    # Retrieve power and delays
    mean_powers = np.power(10.0, np.array(params['powers'])/10.0)

    # Normalize the PDP
    norm_factor = np.sum(mean_powers)
    mean_powers = mean_powers / norm_factor

    if los:
        los_power = mean_powers[0]
        nlos_power = np.sum(mean_powers[1:])
    else:
        nlos_power = np.sum(mean_powers)

    #
    # Build time covariance matrix
    #

    indices = np.arange(num_ofdm_symbols)
    s1 = np.expand_dims(indices, axis=1)
    s2 = np.expand_dims(indices, axis=0)
    exp = doppler_spread*ofdm_symbol_duration*(s1-s2)
    cov_mat_nlos = jv(0.0, exp)*nlos_power
    if los:
        cov_mat_los = np.exp(1j*exp*np.cos(los_angle_of_arrival))*los_power
        cov_mat = cov_mat_nlos+cov_mat_los
    else:
        cov_mat = cov_mat_nlos

    return tf.cast(cov_mat, dtype)
