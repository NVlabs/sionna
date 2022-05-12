#
# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Class definition and functions related to the resource grid"""

import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Layer

from .pilot_pattern import PilotPattern, EmptyPilotPattern, KroneckerPilotPattern # pylint: disable=line-too-long
from sionna.utils import flatten_last_dims, flatten_dims, split_dim
import matplotlib.pyplot as plt
from matplotlib import colors


class ResourceGrid():
    # pylint: disable=line-too-long
    r"""Defines a `ResourceGrid` spanning multiple OFDM symbols and subcarriers.

    Parameters
    ----------
        num_ofdm_symbols : int
            Number of OFDM symbols.

        fft_size : int
            FFT size (, i.e., the number of subcarriers).

        subcarrier_spacing : float
            The subcarrier spacing in Hz.

        num_tx : int
            Number of transmitters.

        num_streams_per_tx : int
            Number of streams per transmitter.

        cyclic_prefix_length : int
            Length of the cyclic prefix.

        num_guard_carriers : int
            List of two integers defining the number of guardcarriers at the
            left and right side of the resource grid.

        dc_null : bool
            Indicates if the DC carrier is nulled or not.

        pilot_pattern : One of [None, "kronecker", "empty", PilotPattern]
            An instance of :class:`~sionna.ofdm.PilotPattern`, a string
            shorthand for the :class:`~sionna.ofdm.KroneckerPilotPattern`
            or :class:`~sionna.ofdm.EmptyPilotPattern`, or `None`.
            Defaults to `None` which is equivalent to `"empty"`.

        pilot_ofdm_symbol_indices : List, int
            List of indices of OFDM symbols reserved for pilot transmissions.
            Only needed if ``pilot_pattern="kronecker"``. Defaults to `None`.

        dtype : tf.Dtype
            Defines the datatype for internal calculations and the output
            dtype. Defaults to `tf.complex64`.
    """
    def __init__(self,
                 num_ofdm_symbols,
                 fft_size,
                 subcarrier_spacing,
                 num_tx=1,
                 num_streams_per_tx=1,
                 cyclic_prefix_length=0,
                 num_guard_carriers=(0,0),
                 dc_null=False,
                 pilot_pattern=None,
                 pilot_ofdm_symbol_indices=None,
                 dtype=tf.complex64):
        super().__init__()
        self._dtype = dtype
        self._num_ofdm_symbols = num_ofdm_symbols
        self._fft_size = fft_size
        self._subcarrier_spacing = subcarrier_spacing
        self._cyclic_prefix_length = cyclic_prefix_length
        self._num_tx = num_tx
        self._num_streams_per_tx = num_streams_per_tx
        self._num_guard_carriers = np.array(num_guard_carriers)
        self._dc_null = dc_null
        self._pilot_ofdm_symbol_indices = pilot_ofdm_symbol_indices
        self.pilot_pattern = pilot_pattern
        self._check_settings()

    @property
    def cyclic_prefix_length(self):
        """Length of the cyclic prefix."""
        return self._cyclic_prefix_length

    @property
    def num_tx(self):
        """Number of transmitters."""
        return self._num_tx

    @property
    def num_streams_per_tx(self):
        """Number of streams  per transmitter."""
        return self._num_streams_per_tx

    @property
    def num_ofdm_symbols(self):
        """The number of OFDM symbols of the resource grid."""
        return self._num_ofdm_symbols

    @property
    def num_resource_elements(self):
        """Number of resource elements."""
        return self._fft_size*self._num_ofdm_symbols

    @property
    def num_effective_subcarriers(self):
        """Number of subcarriers used for data and pilot transmissions."""
        n = self._fft_size - self._dc_null - np.sum(self._num_guard_carriers)
        return n

    @property
    def effective_subcarrier_ind(self):
        """Returns the indices of the effective subcarriers."""
        num_gc = self._num_guard_carriers
        sc_ind = range(num_gc[0], self.fft_size-num_gc[1])
        if self.dc_null:
            sc_ind = np.delete(sc_ind, self.dc_ind-num_gc[0])
        return sc_ind

    @property
    def num_data_symbols(self):
        """Number of resource elements used for data transmissions."""
        n = self.num_effective_subcarriers * self._num_ofdm_symbols - \
               self.num_pilot_symbols
        return tf.cast(n, tf.int32)

    @property
    def num_pilot_symbols(self):
        """Number of resource elements used for pilot symbols."""
        return self.pilot_pattern.num_pilot_symbols

    @property
    def num_zero_symbols(self):
        """Number of empty resource elements."""
        n = (self._fft_size-self.num_effective_subcarriers) * \
               self._num_ofdm_symbols
        return tf.cast(n, tf.int32)

    @property
    def num_guard_carriers(self):
        """Number of left and right guard carriers."""
        return self._num_guard_carriers

    @property
    def dc_ind(self):
        """Index of the DC subcarrier.

        If ``fft_size`` is odd, the index is (``fft_size``-1)/2.
        If ``fft_size`` is even, the index is ``fft_size``/2.
        """
        return int(self._fft_size/2 - (self._fft_size%2==1)/2)

    @property
    def fft_size(self):
        """The FFT size."""
        return self._fft_size

    @property
    def subcarrier_spacing(self):
        """The subcarrier spacing [Hz]."""
        return self._subcarrier_spacing

    @property
    def ofdm_symbol_duration(self):
        """Duration of an OFDM symbol with cyclic prefix [s]."""
        return (1. + self.cyclic_prefix_length/self.fft_size) \
                / self.subcarrier_spacing

    @property
    def bandwidth(self):
        """The occupied bandwidth [Hz]: ``fft_size*subcarrier_spacing``."""
        return self.fft_size*self.subcarrier_spacing

    @property
    def num_time_samples(self):
        """The number of time-domain samples occupied by the resource grid."""
        return (self.fft_size + self.cyclic_prefix_length) \
                * self._num_ofdm_symbols

    @property
    def dc_null(self):
        """Indicates if the DC carriers is nulled or not."""
        return self._dc_null

    @property
    def pilot_pattern(self):
        """The used PilotPattern."""
        return self._pilot_pattern

    @pilot_pattern.setter
    def pilot_pattern(self, value):
        if value is None:
            value = EmptyPilotPattern(self._num_tx,
                                      self._num_streams_per_tx,
                                      self._num_ofdm_symbols,
                                      self.num_effective_subcarriers,
                                      dtype=self._dtype)
        elif isinstance(value, PilotPattern):
            pass
        elif isinstance(value, str):
            assert value in ["kronecker", "empty"],\
                "Unknown pilot pattern"
            if value=="empty":
                value = EmptyPilotPattern(self._num_tx,
                                      self._num_streams_per_tx,
                                      self._num_ofdm_symbols,
                                      self.num_effective_subcarriers,
                                      dtype=self._dtype)
            elif value=="kronecker":
                assert self._pilot_ofdm_symbol_indices is not None,\
                    "You must provide pilot_ofdm_symbol_indices."
                value = KroneckerPilotPattern(self,
                        self._pilot_ofdm_symbol_indices, dtype=self._dtype)
        else:
            raise ValueError("Unsupported pilot_pattern")
        self._pilot_pattern = value

    def _check_settings(self):
        """Validate that all properties define a valid resource grid"""
        assert self._num_ofdm_symbols > 0, \
            "`num_ofdm_symbols` must be positive`."
        assert self._fft_size > 0, \
            "`fft_size` must be positive`."
        assert self._cyclic_prefix_length>=0, \
            "`cyclic_prefix_length must be nonnegative."
        assert self._cyclic_prefix_length<=self._fft_size, \
            "`cyclic_prefix_length cannot be longer than `fft_size`."
        assert self._num_tx > 0, \
            "`num_tx` must be positive`."
        assert self._num_streams_per_tx > 0, \
            "`num_streams_per_tx` must be positive`."
        assert len(self._num_guard_carriers)==2, \
            "`num_guard_carriers` must have two elements."
        assert np.all(np.greater_equal(self._num_guard_carriers, 0)), \
            "`num_guard_carriers` must have nonnegative entries."
        assert np.sum(self._num_guard_carriers)<=self._fft_size-self._dc_null,\
            "Total number of guardcarriers cannot be larger than `fft_size`."
        assert self._dtype in [tf.complex64, tf.complex128], \
            "dtype must be tf.complex64 or tf.complex128"
        return True

    def build_type_grid(self):
        """Returns a tensor indicating the type of each resource element.

        Resource elements can be one of

        - 0 : Data symbol
        - 1 : Pilot symbol
        - 2 : Guard carrier symbol
        - 3 : DC carrier symbol

        Output
        ------
        : [num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size], tf.int32
            Tensor indicating for each transmitter and stream the type of
            the resource elements of the corresponding resource grid.
            The type can be one of [0,1,2,3] as explained above.
        """
        shape = [self._num_tx, self._num_streams_per_tx, self._num_ofdm_symbols]
        gc_l = 2*tf.ones(shape+[self._num_guard_carriers[0]], tf.int32)
        gc_r = 2*tf.ones(shape+[self._num_guard_carriers[1]], tf.int32)
        dc   = 3*tf.ones(shape + [tf.cast(self._dc_null, tf.int32)], tf.int32)
        mask = self.pilot_pattern.mask
        split_ind = self.dc_ind-self._num_guard_carriers[0]
        rg_type = tf.concat([gc_l,                 # Left Guards
                             mask[...,:split_ind], # Data & pilots
                             dc,                   # DC
                             mask[...,split_ind:], # Data & pilots
                             gc_r], -1)            # Right guards
        return rg_type

    def show(self, tx_ind=0, tx_stream_ind=0):
        """Visualizes the resource grid for a specific transmitter and antenna.

        Input
        -----
        tx_ind : int
            Indicates the transmitter index.

        tx_stream_ind : int
            Indicates the index of the stream.

        Output
        ------
        : `matplotlib.figure`
            A handle to a matplot figure object.
        """
        fig = plt.figure(figsize=(10,4))
        data = self.build_type_grid()[tx_ind, tx_stream_ind]
        cmap = colors.ListedColormap([[60/256,8/256,72/256],
                              [45/256,91/256,128/256],
                              [45/256,172/256,111/256],
                              [250/256,228/256,62/256]])
        bounds=[0,1,2,3,4]
        norm = colors.BoundaryNorm(bounds, cmap.N)
        img = plt.imshow(data, interpolation="nearest",
                         origin="lower", cmap=cmap, norm=norm,
                         aspect="auto")
        cbar = plt.colorbar(img, ticks=[0.5, 1.5, 2.5,3.5],
                            orientation="vertical", shrink=0.8)
        cbar.set_ticklabels(["Data", "Pilot", "Guard carrier", "DC carrier"])
        plt.title("OFDM Resource Grid")
        plt.ylabel("OFDM Symbol")
        plt.xlabel("Subcarrier Index")

        return fig

class ResourceGridMapper(Layer):
    # pylint: disable=line-too-long
    r"""ResourceGridMapper(resource_grid, dtype=tf.complex64, **kwargs)

    Maps a tensor of modulated data symbols to a ResourceGrid.

    This layer takes as input a tensor of modulated data symbols
    and maps them together with pilot symbols onto an
    OFDM :class:`~sionna.ofdm.ResourceGrid`. The output can be
    converted to a time-domain signal with the
    :class:`~sionna.ofdm.Modulator` or further processed in the
    frequency domain.

    Parameters
    ----------
    resource_grid : ResourceGrid
        An instance of :class:`~sionna.ofdm.ResourceGrid`.

    dtype : tf.Dtype
        Datatype for internal calculations and the output dtype.
        Defaults to `tf.complex64`.

    Input
    -----
    : [batch_size, num_tx, num_streams_per_tx, num_data_symbols], tf.complex
        The modulated data symbols to be mapped onto the resource grid.

    Output
    ------
    : [batch_size, num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size], tf.complex
        The full OFDM resource grid in the frequency domain.
    """
    def __init__(self, resource_grid, dtype=tf.complex64, **kwargs):
        super().__init__(dtype=dtype, **kwargs)
        self._resource_grid = resource_grid

    def build(self, input_shape): # pylint: disable=unused-argument
        """Precompute a tensor of shape
        [num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size]
        which is prefilled with pilots and stores indices
        to scatter data symbols.
        """
        rg_type = self._resource_grid.build_type_grid()
        pilot_ind = tf.where(rg_type==1)
        pilots = flatten_last_dims(self._resource_grid.pilot_pattern.pilots, 3)
        self._template = tf.scatter_nd(pilot_ind, pilots, rg_type.shape)
        self._template = tf.expand_dims(self._template, -1)
        self._data_ind = tf.where(rg_type==0)

    def call(self, inputs):
        # Broadcast the resource grid template to batch_size
        batch_size = tf.shape(inputs)[0]
        new_shape = tf.concat([tf.shape(self._template)[:-1], [batch_size]], 0)
        template = tf.broadcast_to(self._template, new_shape)

        # Flatten the inputs and put batch_dim last for scatter update
        inputs = tf.transpose(flatten_last_dims(inputs, 3))
        rg = tf.tensor_scatter_nd_update(template, self._data_ind, inputs)
        rg = tf.transpose(rg, [4, 0, 1, 2, 3])

        return rg

class ResourceGridDemapper(Layer):
    # pylint: disable=line-too-long
    r"""ResourceGridDemapper(resource_grid, stream_management, dtype=tf.complex64, **kwargs)

    Extracts data-carrying resource elements from a resource grid.

    This layer takes as input an OFDM :class:`~sionna.ofdm.ResourceGrid` and
    extracts the data-carrying resource elements. In other words, it implements
    the reverse operation of :class:`~sionna.ofdm.ResourceGridMapper`.

    Parameters
    ----------
    resource_grid : ResourceGrid
        An instance of :class:`~sionna.ofdm.ResourceGrid`.

    stream_management : StreamManagement
        An instance of :class:`~sionna.mimo.StreamManagement`.

    dtype : tf.Dtype
        Datatype for internal calculations and the output dtype.
        Defaults to `tf.complex64`.

    Input
    -----
    : [batch_size, num_rx, num_streams_per_rx, num_ofdm_symbols, fft_size, data_dim]
        The full OFDM resource grid in the frequency domain.
        The last dimension `data_dim` is optional. If `data_dim`
        is used, it refers to the dimensionality of the data that should be
        demapped to individual streams. An example would be LLRs.

    Output
    ------
    : [batch_size, num_rx, num_streams_per_rx, num_data_symbols, data_dim]
        The data that were mapped into the resource grid.
        The last dimension `data_dim` is only returned if it was used for the
        input.
    """
    def __init__(self,
                 resource_grid,
                 stream_management,
                 dtype=tf.complex64,
                 **kwargs):
        super().__init__(dtype=dtype, **kwargs)
        self._stream_management = stream_management
        self._resource_grid = resource_grid

        # Precompute indices to extract data symbols
        mask = resource_grid.pilot_pattern.mask
        num_data_symbols = resource_grid.pilot_pattern.num_data_symbols
        data_ind = tf.argsort(flatten_last_dims(mask), direction="ASCENDING")
        self._data_ind = data_ind[...,:num_data_symbols]

    def call(self, y): # pylint: disable=arguments-renamed

        # y has shape
        # [batch_size, num_rx, num_streams_per_rx, num_ofdm_symbols,...
        # ..., fft_size, data_dim]

        # If data_dim is not provided, add a dummy dimension
        if len(y.shape)==5:
            y = tf.expand_dims(y, -1)

        # Remove nulled subcarriers from y (guards, dc). New shape:
        # [batch_size, num_rx, num_rx_ant, ...
        #  ..., num_ofdm_symbols, num_effective_subcarriers, data dim]
        y = tf.gather(y, self._resource_grid.effective_subcarrier_ind, axis=-2)

        # Transpose tensor to shape
        # [num_rx, num_streams_per_rx, num_ofdm_symbols,...
        #  ..., num_effective_subcarriers, data_dim, batch_size]
        y = tf.transpose(y, [1, 2, 3, 4, 5, 0])

        # Merge num_rx amd num_streams_per_rx
        # [num_rx * num_streams_per_rx, num_ofdm_symbols,...
        #  ...,num_effective_subcarriers, data_dim, batch_size]
        y = flatten_dims(y, 2, 0)

        # Put first dimension into the right ordering
        stream_ind = self._stream_management.stream_ind
        y = tf.gather(y, stream_ind, axis=0)

        # Reshape first dimensions to [num_tx, num_streams] so that
        # we can compared to the way the streams were created.
        # [num_tx, num_streams, num_ofdm_symbols, num_effective_subcarriers,...
        #  ..., data_dim, batch_size]
        num_streams = self._stream_management.num_streams_per_tx
        num_tx = self._stream_management.num_tx
        y = split_dim(y, [num_tx, num_streams], 0)

        # Flatten resource grid dimensions
        # [num_tx, num_streams, num_ofdm_symbols*num_effective_subcarriers,...
        #  ..., data_dim, batch_size]
        y = flatten_dims(y, 2, 2)

        # Gather data symbols
        # [num_tx, num_streams, num_data_symbols, data_dim, batch_size]
        y = tf.gather(y, self._data_ind, batch_dims=2, axis=2)

        # Put batch_dim first
        # [batch_size, num_tx, num_streams, num_data_symbols]
        y = tf.transpose(y, [4, 0, 1, 2, 3])

        # Squeeze data_dim
        if y.shape[-1]==1:
            y = tf.squeeze(y, -1)

        return y

class RemoveNulledSubcarriers(Layer):
    # pylint: disable=line-too-long
    r"""RemoveNulledSubcarriers(resource_grid, **kwargs)

    Removes nulled guard and/or DC subcarriers from a resource grid.

    Parameters
    ----------
    resource_grid : ResourceGrid
        An instance of :class:`~sionna.ofdm.ResourceGrid`.

    Input
    -----
    : [batch_size, num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size], tf.complex64
        Full resource grid.

    Output
    ------
    : [batch_size, num_tx, num_streams_per_tx, num_ofdm_symbols, num_effective_subcarriers], tf.complex64
        Resource grid without nulled subcarriers.
    """
    def __init__(self, resource_grid, **kwargs):
        self._sc_ind = resource_grid.effective_subcarrier_ind
        super().__init__(**kwargs)

    def call(self, inputs):
        return tf.gather(inputs, self._sc_ind, axis=-1)
