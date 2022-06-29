#
# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Functions related to OFDM channel estimation"""

import tensorflow as tf
from tensorflow.keras.layers import Layer
import numpy as np
from sionna.utils import flatten_last_dims, expand_to_rank
from sionna.ofdm import ResourceGrid, RemoveNulledSubcarriers


class LSChannelEstimator(Layer):
    # pylint: disable=line-too-long
    r"""LSChannelEstimator(resource_grid, interpolation_type="nn", dtype=tf.complex64, **kwargs)

    Layer implementing least-squares (LS) channel estimation for OFDM MIMO systems.

    After LS channel estimation at the pilot positions, the channel estimates
    and error variances are interpolated accross the entire resource grid using
    a specified interpolation function.

    For simplicity, we describe the underlying algorithm for a vectorized observation,
    where we have a nonzero pilot for all elements to be estimated.
    The actually implementation works on a full OFDM resource grid with sparse
    pilot patterns. We consider the following model:

    .. math::

        \mathbf{y} = \mathbf{h}\odot\mathbf{p} + \mathbf{n}

    where :math:`\mathbf{y}\in\mathbb{C}^{M}` is the received signal vector,
    :math:`\mathbf{p}\in\mathbb{C}^M` is the vector of pilot symbols,
    :math:`\mathbf{H}\in\mathbb{C}^{M}` is the channel vector to be estimated,
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
        The interpolation to be used. Currently only the
        :class:`~sionna.ofdm.NearestNeighborInterpolator` (`"nn`") and
        :class:`~sionna.ofdm.LinearInterpolator`
        without (`"lin"`) and with averaging across OFDM
        symbols (`"lin_time_avg"`) are supported.

    dtype : tf.Dtype
        Datatype for internal calculations and the output dtype.
        Defaults to `tf.complex64`.

    Input
    -----
    (y, no) :
        Tuple:

    y : [batch_size, num_rx, num_rx_ant, num_ofdm_symbols,fft_size], tf.complex
        The observed signals.

    no : [batch_size, num_rx, num_rx_ant] or only the first n>=0 dims, tf.float
        The variance of the AWGN.

    Output
    ------
    h_ls : [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols,fft_size], tf.complex
        The channel estimates accross the entire resource grid for all
        transmitters and streams.

    err_var : Same shape as ``h_ls``, tf.float
        The channel estimation error variance accross the entire resource grid
        for all transmitters and streams.
    """
    def __init__(self, resource_grid, interpolation_type="nn", dtype=tf.complex64, **kwargs):
        super().__init__(dtype=dtype, **kwargs)
        assert isinstance(resource_grid, ResourceGrid),\
            "You must provide a valid instance of ResourceGrid."
        self._pilot_pattern = resource_grid.pilot_pattern
        self._removed_nulled_scs = RemoveNulledSubcarriers(resource_grid)

        assert interpolation_type in ["nn", "lin", "lin_time_avg", None], \
            "Unsupported `interpolation_type`"
        self._interpolation_type = interpolation_type
        if self._interpolation_type=="nn":
            self._interpol = NearestNeighborInterpolator(self._pilot_pattern)
        if self._interpolation_type=="lin":
            self._interpol = LinearInterpolator(self._pilot_pattern)
        if self._interpolation_type=="lin_time_avg":
            self._interpol = LinearInterpolator(self._pilot_pattern, time_avg=True)

        # Precompute indices to gather received pilot signals
        num_pilot_symbols = self._pilot_pattern.num_pilot_symbols
        mask = flatten_last_dims(self._pilot_pattern.mask)
        pilot_ind = tf.argsort(mask, axis=-1, direction="DESCENDING")
        self._pilot_ind = pilot_ind[...,:num_pilot_symbols]

    def call(self, inputs):
        """
        y_ has shape
        [batch_size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size]
        no has a shape that can be broadcast to the shape [num_tx, num_streams,]
        """
        # y has shape:
        # [batch_size, num_rx, num_rx_ant, num_ofdm_symbols,..
        # ... fft_size]
        #
        # no can have shapes [], [batch_size], [batch_size, num_rx]
        # or [batch_size, num_rx, num_rx_ant]
        y, no = inputs

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
        h_ls = tf.math.divide_no_nan(y_pilots, self._pilot_pattern.pilots)

        # Compute error variance and broadcast to the same shape as h_ls
        # Expand rank of no for broadcasting
        no = expand_to_rank(no, tf.rank(h_ls), -1)

        # Expand rank of pilots for broadcasting
        pilots = expand_to_rank(self._pilot_pattern.pilots, tf.rank(h_ls), 0)

        # Compute error variance, broadcastable to the shape of h_ls
        err_var = tf.math.divide_no_nan(no, tf.abs(pilots)**2)

        # Interpolate channel estimates over the resource grid
        if self._interpolation_type is not None:
            h_ls = self._interpol(h_ls)
            err_var = tf.maximum(self._interpol(err_var),
                                 tf.cast(0, err_var.dtype))

        return h_ls, err_var

class NearestNeighborInterpolator():
    # pylint: disable=line-too-long
    """Nearest-neighbor channel estimate interpolation on a resource grid.

    This class assigns to each element of an OFDM resource grid one of
    ``num_pilots`` provided measurements, e.g., channel estimates or error
    variances, according to the nearest neighbor method. It is assumed
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
        An instance of :class:`~sionna.ofdm.PilotPattern`.

    Input
    -----
    : [k, l ,m, num_tx, num_streams_per_tx, num_pilot_symbols], tf.DType
        Tensor of quantities to be interpolated according to
        a :class:`~sionna.ofdm.PilotPattern`. This can be channel estimates
        as well as the related error variances.

    Output
    ------
    : [k, l, m, num_tx, num_streams_per_tx, num_ofdm_symbols, num_effective_subcarriers], tf.complex
        The interpolated input tensor.
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

    def __call__(self, inputs):
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


class LinearInterpolator():
    # pylint: disable=line-too-long
    r"""Linear channel estimate interpolation on a resource grid.

    This class computes for each element of an OFDM resource grid
    a channel estimate based on ``num_pilots`` provided measurements,
    e.g., channel estimates or error variances, through linear interpolation.
    It is assumed that the measurements were taken at the nonzero positions
    of a :class:`~sionna.ofdm.PilotPattern`.

    The interpolation is done first across sub-carriers and then
    across OFDM symbols.

    Parameters
    ----------
    pilot_pattern : PilotPattern
        An instance of :class:`~sionna.ofdm.PilotPattern`.

    time_avg : bool
        If enabled, measurements will be averaged across OFDM symbols
        (i.e., time). This is useful for channels that do not vary
        substantially over the duration of an OFDM frame. Defaults to `False`.

    Input
    -----
    : [k, l ,m, num_tx, num_streams_per_tx, num_pilot_symbols], tf.DType
        Tensor of quantities to be interpolated according to
        a :class:`~sionna.ofdm.PilotPattern`. This can be channel estimates
        as well as the related error variances.

    Output
    ------
    : [k, l, m, num_tx, num_streams_per_tx, num_ofdm_symbols, num_effective_subcarriers], tf.complex
        The interpolated input tensor.
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


    def _interpolate(self, inputs, x, x0, x1, y0_ind, y1_ind):
        # Gather the right values for y0 and y1
        y0 = tf.gather(inputs, y0_ind, axis=2, batch_dims=2)
        y1 = tf.gather(inputs, y1_ind, axis=2, batch_dims=2)

        # Undo the permutation of the inputs
        y0 = tf.transpose(y0, self._perm_bwd)
        y1 = tf.transpose(y1, self._perm_bwd)

        # Compute linear interpolation
        slope = tf.math.divide_no_nan(y1-y0, tf.cast(x1-x0, dtype=y0.dtype))
        return tf.cast(x-x0, dtype=y0.dtype)*slope + y0

    def __call__(self, inputs):
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
        h_hat_freq = self._interpolate(inputs,
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
            h_hat_freq /= tf.cast(self._num_pilot_ofdm_symbols, h_hat_freq.dtype)
            h_hat_freq = tf.repeat(h_hat_freq, [num_ofdm_symbols], axis=-2)

        # Transpose h_hat_freq to bring batch_dims for gather last. New shape:
        # [num_tx, num_streams_per_tx, num_ofdm_symbols,...
        #  ...num_effective_subcarriers, k, l, m]
        h_hat_time = tf.transpose(h_hat_freq, self._perm_fwd_time)

        # h_hat_time has shape:
        # [k, l, m, num_tx, num_streams_per_tx, num_ofdm_symbols,...
        #  ...num_effective_subcarriers]
        h_hat_time = self._interpolate(h_hat_time,
                                        self._x_time,
                                        self._x_0_time,
                                        self._x_1_time,
                                        self._y_0_time_ind,
                                        self._y_1_time_ind)

        return h_hat_time
