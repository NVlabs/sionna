#
# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Class definition and functions related to pilot patterns"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from sionna.utils import QAMSource


class PilotPattern():
    # pylint: disable=line-too-long
    r"""Class defining a pilot pattern for an OFDM ResourceGrid.

    This class defines a pilot pattern object that is used to configure
    an OFDM :class:`~sionna.ofdm.ResourceGrid`.

    Parameters
    ----------
    mask : [num_tx, num_streams_per_tx, num_ofdm_symbols, num_effective_subcarriers], bool
        Tensor indicating resource elements that are reserved for pilot transmissions.

    pilots : [num_tx, num_txt_ant, num_pilots], tf.complex
        The pilot symbols to be mapped onto the ``mask``.

    trainable : bool
        Indicates if ``pilots`` is a trainable `Variable`.
        Defaults to `False`.

    normalize : bool
        Indicates if the ``pilots`` should be normalized to an average
        energy of one across the last dimension. This can be useful to
        ensure that trainable ``pilots`` have a finite energy.
        Defaults to `False`.

    dtype : tf.Dtype
        Defines the datatype for internal calculations and the output
        dtype. Defaults to `tf.complex64`.
    """
    def __init__(self, mask, pilots, trainable=False, normalize=False,
                 dtype=tf.complex64):
        super().__init__()
        self._dtype = dtype
        self._mask = tf.cast(mask, tf.int32)
        self._pilots = tf.Variable(tf.cast(pilots, self._dtype), trainable)
        self.normalize = normalize
        self._check_settings()

    @property
    def num_tx(self):
        """The number of transmitters."""
        return self._mask.shape[0]

    @property
    def num_streams_per_tx(self):
        """The number of streams per transmitter."""
        return self._mask.shape[1]

    @ property
    def num_ofdm_symbols(self):
        """The number of OFDM symbols."""
        return self._mask.shape[2]

    @ property
    def num_effective_subcarriers(self):
        """The number of effectvie subcarriers."""
        return self._mask.shape[3]

    @property
    def num_pilot_symbols(self):
        """Number of pilot symbols per transmit antenna."""
        return tf.shape(self._pilots)[-1]

    @property
    def num_data_symbols(self):
        """ Number of data symbols per transmit antenna."""
        return tf.shape(self._mask)[-1]*tf.shape(self._mask)[-2] - \
               self.num_pilot_symbols

    @property
    def normalize(self):
        """Indicates if the pilots are normalized or not."""
        return self._normalize

    @normalize.setter
    def normalize(self, value):
        self._normalize = tf.cast(value, tf.bool)

    @property
    def mask(self):
        """The mask of the pilot pattern."""
        return self._mask

    @property
    def pilots(self):
        """Returns the possibly normalized tensor of pilot symbols."""
        def norm_pilots():
            scale = tf.abs(self._pilots)**2
            scale = 1/tf.sqrt(tf.reduce_mean(scale, axis=-1, keepdims=True))
            scale = tf.cast(scale, self._dtype)
            return scale*self._pilots

        return tf.cond(self.normalize, norm_pilots, lambda: self._pilots)

    def _check_settings(self):
        """Validate that all properties define a valid pilot pattern."""

        assert tf.rank(self._mask)==4, "`mask` must have four dimensions."
        assert tf.rank(self._pilots)==3, "`pilots` must have three dimensions."
        assert np.array_equal(self._mask.shape[:2], self._pilots.shape[:2]), \
            "The first two dimensions of `mask` and `pilots` must be equal."

        num_pilots = tf.reduce_sum(self._mask, axis=(-2,-1))
        assert tf.reduce_min(num_pilots)==tf.reduce_max(num_pilots), \
            """The number of nonzero elements in the masks for all transmitters
            and antennas must be identical."""

        assert self.num_pilot_symbols==tf.reduce_max(num_pilots), \
            """The shape of the last dimension of `pilots` must equal
            the number of non-zero entries within the last two
            dimensions of `mask`."""

        return True

    def show(self, tx_ind=None, stream_ind=None, show_pilot_ind=False):
        """Visualizes the non-zero pilots for some transmitters and streams.

        The function only produces correct results for non-overlapping
        pilot sequences. In order to inspect overlapping sequences, one can
        simply call the function with different transmitter and stream indices.

        Input
        -----
        tx_ind : list, int
            Indicates the indices of transmitters to be included.
            Defaults to `None`, i.e., all transmitters included.

        stream_ind : list, int
            Indicates the indices of streams to be included.
            Defaults to `None`, i.e., all streams included.

        show_pilot_ind : bool
            Indicates if the indices of the pilot symbols should be shhown.

        Output
        ------
        : matplotlib.figure.Figure
            A handle to a matplot figure object.
        """
        mask = self.mask.numpy()
        pilots = self.pilots.numpy()

        if tx_ind is None:
            tx_ind = range(0, self.num_tx)
        elif not isinstance(tx_ind, list):
            tx_ind = [tx_ind]

        if stream_ind is None:
            stream_ind = range(0, self.num_streams_per_tx)
        elif not isinstance(stream_ind, list):
            stream_ind = [stream_ind]

        legend = ["Data"]
        q = np.zeros_like(mask[0,0])
        for i in tx_ind:
            for j in stream_ind:
                legend.append(f"TX {i}, Stream {j}")
                m = mask[i,j]
                p = np.sign(np.abs(pilots[i,j]))*(i*len(stream_ind)+j+1)
                m[np.where(m)] = p
                q += m

        fig = plt.figure(figsize=(10,4))
        plt.title("Non-Zero Pilots")
        plt.ylabel("OFDM Symbol")
        plt.xlabel("Effective Subcarrier Index")
        cmap = plt.cm.tab20c
        b = np.arange(0, np.max(q)+2)
        norm = colors.BoundaryNorm(b, cmap.N)
        im = plt.imshow(q, origin="lower", aspect="auto", norm=norm, cmap=cmap)
        cbar = plt.colorbar(im)
        cbar.set_ticks(b[:-1]+0.5)
        cbar.set_ticklabels(legend)

        if show_pilot_ind:
            for t in tx_ind:
                for k in stream_ind:
                    c = 0
                    for i in range(self.num_ofdm_symbols):
                        for j in range(self.num_effective_subcarriers):
                            if self.mask[t,k,i,j]==1:
                                if np.abs(pilots[t,k,c])>0:
                                    plt.annotate(c, [j,i])
                                c+=1

        return fig

class EmptyPilotPattern(PilotPattern):
    """Creates an empty pilot pattern.

    Generates a instance of :class:`~sionna.ofdm.PilotPattern` with
    an empty ``mask`` and ``pilots``.

    Parameters
    ----------
    num_tx : int
        Number of transmitters.

    num_streams_per_tx : int
        Number of streams per transmitter.

    num_ofdm_symbols : int
        Number of OFDM symbols.

    num_effective_subcarriers : int
        Number of effective subcarriers
        that are available for the transmission of data and pilots.
        Note that this number is generally smaller than the ``fft_size``
        due to nulled subcarriers.

    dtype : tf.Dtype
        Defines the datatype for internal calculations and the output
        dtype. Defaults to `tf.complex64`.
    """
    def __init__(self,
                 num_tx,
                 num_streams_per_tx,
                 num_ofdm_symbols,
                 num_effective_subcarriers,
                 dtype=tf.complex64):

        assert num_tx > 0, \
            "`num_tx` must be positive`."
        assert num_streams_per_tx > 0, \
            "`num_streams_per_tx` must be positive`."
        assert num_ofdm_symbols > 0, \
            "`num_ofdm_symbols` must be positive`."
        assert num_effective_subcarriers > 0, \
            "`num_effective_subcarriers` must be positive`."

        shape = [num_tx, num_streams_per_tx, num_ofdm_symbols,
                      num_effective_subcarriers]
        mask = tf.zeros(shape, tf.bool)
        pilots = tf.zeros(shape[:2]+[0], dtype)
        super().__init__(mask, pilots, trainable=False, normalize=False)

class KroneckerPilotPattern(PilotPattern):
    """Simple orthogonal pilot pattern with Kronecker structure.

    This function generates an instance of :class:`~sionna.ofdm.PilotPattern`
    that allocates non-overlapping pilot sequences for all transmitters and
    streams on specified OFDM symbols. As the same pilot sequences are reused
    across those OFDM symbols, the resulting pilot pattern has a frequency-time
    Kronecker structure. This structure enables a very efficient implementation
    of the LMMSE channel estimator. Each pilot sequence is constructed from
    randomly drawn QPSK constellation points.

    Parameters
    ----------
    resource_grid : ResourceGrid
        An instance of a :class:`~sionna.ofdm.ResourceGrid`.

    pilot_ofdm_symbol_indices : list, int
        List of integers defining the OFDM symbol indices that are reserved
        for pilots.

    normalize : bool
        Indicates if the ``pilots`` should be normalized to an average
        energy of one across the last dimension.
        Defaults to `True`.

    seed : int
        Seed for the generation of the pilot sequence. Different seed values
        lead to different sequences. Defaults to 0.

    dtype : tf.Dtype
        Defines the datatype for internal calculations and the output
        dtype. Defaults to `tf.complex64`.

    Note
    ----
    It is required that the ``resource_grid``'s property
    ``num_effective_subcarriers`` is an
    integer multiple of ``num_tx * num_streams_per_tx``. This condition is
    required to ensure that all transmitters and transmit antennas get
    non-overlapping pilot sequences. For a large number of antennas and/or
    transmitters, the pilot pattern becomes very sparse in the frequency
    domain.

    Examples
    --------
    >>> rg = ResourceGrid(num_ofdm_symbols=14,
    ...                   fft_size=64,
    ...                   subcarrier_spacing = 30e3,
    ...                   num_tx=4,
    ...                   num_streams_per_tx=2,
    ...                   pilot_pattern = "kronecker",
    ...                   pilot_ofdm_symbol_indices = [2, 11])
    >>> rg.pilot_pattern.show();

    .. image:: ../figures/kronecker_pilot_pattern.png

    """
    def __init__(self,
                 resource_grid,
                 pilot_ofdm_symbol_indices,
                 normalize=True,
                 seed=0,
                 dtype=tf.complex64):

        num_tx = resource_grid.num_tx
        num_streams_per_tx = resource_grid.num_streams_per_tx
        num_ofdm_symbols = resource_grid.num_ofdm_symbols
        num_effective_subcarriers = resource_grid.num_effective_subcarriers
        self._dtype = dtype

        # Number of OFDM symbols carrying pilots
        num_pilot_symbols = len(pilot_ofdm_symbol_indices)

        # Compute the total number of required orthogonal sequences
        num_seq = num_tx*num_streams_per_tx

        # Compute the length of a pilot sequence
        num_pilots = num_pilot_symbols*num_effective_subcarriers/num_seq
        assert num_pilots%1==0, \
            """`num_effective_subcarriers` must be an integer multiple of
            `num_tx`*`num_streams_per_tx`."""

        # Number of pilots per OFDM symbol
        num_pilots_per_symbol = int(num_pilots/num_pilot_symbols)

        # Prepare empty mask and pilots
        shape = [num_tx, num_streams_per_tx,
                 num_ofdm_symbols,num_effective_subcarriers]
        mask = np.zeros(shape, bool)
        shape[2] = num_pilot_symbols
        pilots = np.zeros(shape, np.complex64)

        # Populate all selected OFDM symbols in the mask
        mask[..., pilot_ofdm_symbol_indices, :] = True

        # Populate the pilots with random QPSK symbols
        qam_source = QAMSource(2, seed=seed, dtype=self._dtype)
        for i in range(num_tx):
            for j in range(num_streams_per_tx):
                # Generate random QPSK symbols
                p = qam_source([1,1,num_pilot_symbols,num_pilots_per_symbol])

                # Place pilots spaced by num_seq to avoid overlap
                pilots[i,j,:,i*num_streams_per_tx+j::num_seq] = p

        # Reshape the pilots tensor
        pilots = np.reshape(pilots, [num_tx, num_streams_per_tx, -1])

        super().__init__(mask, pilots, trainable=False,
                         normalize=normalize, dtype=self._dtype)
