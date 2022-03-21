#
# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"Classeds and functions related to stream management in MIMO systems"

import numpy as np


class StreamManagement():
    """Class for management of streams in multi-cell MIMO networks.

    Parameters
    ----------
    rx_tx_association : [num_rx, num_tx], np.int
        A binary NumPy array where ``rx_tx_association[i,j]=1`` means
        that receiver `i` gets one ore multiple streams from
        transmitter `j`.

    num_streams_per_tx : int
        Indicates the number of streams that are transmitted by each
        transmitter.

    Note
    ----
    Several symmetry constraints on ``rx_tx_association`` are imposed
    to ensure efficient processing. All row sums and all column sums
    must be equal, i.e., all receivers have the same number of associated
    transmitters and all transmitters have the same number of associated
    receivers. It is also assumed that all transmitters send the same
    number of streams ``num_streams_per_tx``.
    """
    def __init__(self,
                 rx_tx_association,
                 num_streams_per_tx):

        super().__init__()
        self._num_streams_per_tx = int(num_streams_per_tx)
        self.rx_tx_association = rx_tx_association

    @property
    def rx_tx_association(self):
        """Association between receivers and transmitters.

        A binary NumPy array of shape `[num_rx, num_tx]`,
        where ``rx_tx_association[i,j]=1`` means that receiver `i`
        gets one ore multiple streams from transmitter `j`.
        """
        return self._rx_tx_association

    @property
    def num_rx(self):
        "Number of receivers."
        return self._num_rx

    @property
    def num_tx(self):
        "Number of transmitters."
        return self._num_tx

    @property
    def num_streams_per_tx(self):
        "Number of streams per transmitter."
        return self._num_streams_per_tx

    @property
    def num_streams_per_rx(self):
        "Number of streams transmitted to each receiver."
        return int(self.num_tx*self.num_streams_per_tx/self.num_rx)

    @property
    def num_interfering_streams_per_rx(self):
        "Number of interfering streams received at each eceiver."
        return int(self.num_tx*self.num_streams_per_tx
                   - self.num_streams_per_rx)

    @property
    def num_tx_per_rx(self):
        "Number of transmitters communicating with a receiver."
        return self._num_tx_per_rx

    @property
    def num_rx_per_tx(self):
        "Number of receivers communicating with a transmitter."
        return self._num_rx_per_tx

    @property
    def precoding_ind(self):
        """Indices needed to gather channels for precoding.

        A NumPy array of shape `[num_tx, num_rx_per_tx]`,
        where ``precoding_ind[i,:]`` contains the indices of the
        receivers to which transmitter `i` is sending streams.
        """
        return self._precoding_ind

    @property
    def stream_association(self):
        """Association between receivers, transmitters, and streams.

        A binary NumPy array of shape
        `[num_rx, num_tx, num_streams_per_tx]`, where
        ``stream_association[i,j,k]=1`` means that receiver `i` gets
        the `k` th stream from transmitter `j`.
        """
        return self._stream_association

    @property
    def detection_desired_ind(self):
        """Indices needed to gather desired channels for receive processing.

        A NumPy array of shape `[num_rx*num_streams_per_rx]` that
        can be used to gather desired channels from the flattened
        channel tensor of shape
        `[...,num_rx, num_tx, num_streams_per_tx,...]`.
        The result of the gather operation can be reshaped to
        `[...,num_rx, num_streams_per_rx,...]`.
        """
        return self._detection_desired_ind

    @property
    def detection_undesired_ind(self):
        """Indices needed to gather undesired channels for receive processing.

        A NumPy array of shape `[num_rx*num_streams_per_rx]` that
        can be used to gather undesired channels from the flattened
        channel tensor of shape `[...,num_rx, num_tx, num_streams_per_tx,...]`.
        The result of the gather operation can be reshaped to
        `[...,num_rx, num_interfering_streams_per_rx,...]`.
        """
        return self._detection_undesired_ind

    @property
    def tx_stream_ids(self):
        """Mapping of streams to transmitters.

        A NumPy array of shape `[num_tx, num_streams_per_tx]`.
        Streams are numbered from 0,1,... and assiged to transmitters in
        increasing order, i.e., transmitter 0 gets the first
        `num_streams_per_tx` and so on.
        """
        return self._tx_stream_ids

    @property
    def rx_stream_ids(self):
        """Mapping of streams to receivers.

        A Numpy array of shape `[num_rx, num_streams_per_rx]`.
        This array is obtained from ``tx_stream_ids`` together with
        the ``rx_tx_association``. ``rx_stream_ids[i,:]`` contains
        the indices of streams that are supposed to be decoded by receiver `i`.
        """
        return self._rx_stream_ids

    @property
    def stream_ind(self):
        """Indices needed to gather received streams in the correct order.

        A NumPy array of shape `[num_rx*num_streams_per_rx]` that can be
        used to gather streams from the flattened tensor of received streams
        of shape `[...,num_rx, num_streams_per_rx,...]`. The result of the
        gather operation is then reshaped to
        `[...,num_tx, num_streams_per_tx,...]`.
        """
        return self._stream_ind

    @rx_tx_association.setter
    def rx_tx_association(self, rx_tx_association):
        """Sets the rx_tx_association and derives related properties. """

        # Make sure that rx_tx_association is a binary NumPy array
        rx_tx_association = np.array(rx_tx_association, np.int32)
        assert all(x in [0,1] for x in np.nditer(rx_tx_association)), \
            "All elements of `stream_association` must be 0 or 1."

        # Obtain num_rx, num_tx from stream_association shape
        self._num_rx, self._num_tx = np.shape(rx_tx_association)

        # Each receiver must be associated with the same number of transmitters
        num_tx_per_rx = np.sum(rx_tx_association, 1)
        assert np.min(num_tx_per_rx) == np.max(num_tx_per_rx), \
            """Each receiver needs to be associated with the same number
               of transmitters."""
        self._num_tx_per_rx = num_tx_per_rx[0]

        # Each transmitter must be associated with the same number of receivers
        num_rx_per_tx = np.sum(rx_tx_association, 0)
        assert np.min(num_rx_per_tx) == np.max(num_rx_per_tx), \
            """Each transmitter needs to be associated with the same number
               of receivers."""
        self._num_rx_per_tx = num_rx_per_tx[0]

        self._rx_tx_association = rx_tx_association

        # Compute indices for precoding
        self._precoding_ind = np.zeros([self.num_tx, self.num_rx_per_tx],
                                        np.int32)
        for i in range(self.num_tx):
            self._precoding_ind[i,:] = np.where(self.rx_tx_association[:,i])[0]

        # Construct the stream association matrix
        # The element [i,j,k]=1 indicates that receiver i, get the kth stream
        # from transmitter j.
        stream_association = np.zeros(
            [self.num_rx, self.num_tx, self.num_streams_per_tx], np.int32)
        n_streams = np.min([self.num_streams_per_rx, self.num_streams_per_tx])
        tmp = np.ones([n_streams])
        for j in range(self.num_tx):
            c = 0
            for i in range(self.num_rx):
                # If receiver i gets anything from transmitter j
                if rx_tx_association[i,j]:
                    stream_association[i,j,c:c+self.num_streams_per_rx] = tmp
                    c += self.num_streams_per_rx
        self._stream_association = stream_association

        # Get indices of desired and undesired channel coefficients from
        # the flattened stream_association. These indices can be used by
        # a receiver to gather channels of desired and undesired streams.
        self._detection_desired_ind = \
                 np.where(np.reshape(stream_association, [-1])==1)[0]

        self._detection_undesired_ind = \
                 np.where(np.reshape(stream_association, [-1])==0)[0]

        # We number streams from 0,1,... and assign them to the TX
        # TX 0 gets the first num_streams_per_tx and so on:
        self._tx_stream_ids = np.reshape(
                    np.arange(0, self.num_tx*self.num_streams_per_tx),
                    [self.num_tx, self.num_streams_per_tx])

        # We now compute the stream_ids for each receiver
        self._rx_stream_ids = np.zeros([self.num_rx, self.num_streams_per_rx],
                                        np.int32)
        for i in range(self.num_rx):
            c = []
            for j in range(self.num_tx):
                # If receiver i gets anything from transmitter j
                if rx_tx_association[i,j]:
                    tmp = np.where(stream_association[i,j])[0]
                    tmp += j*self.num_streams_per_tx
                    c += list(tmp)
            self._rx_stream_ids[i,:] = c

        # Get indices to bring received streams back to the right order in
        # which they were transmitted.
        self._stream_ind = np.argsort(np.reshape(self._rx_stream_ids, [-1]))
