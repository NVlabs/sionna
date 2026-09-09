#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Classes and functions related to stream management in MIMO systems."""

import numpy as np

__all__ = ["StreamManagement"]


class StreamManagement:
    r"""Class for management of streams in multi-cell MIMO networks.

    Stream management determines which transmitter is sending which stream to
    which receiver. Transmitters and receivers can be user terminals or base
    stations, depending on whether uplink or downlink transmissions are considered.
    This class has various properties that are needed to recover desired or
    interfering channel coefficients for precoding and equalization. In order to
    understand how the various properties can be used, we recommend to have a look
    at the source code of the :class:`~sionna.phy.ofdm.LMMSEEqualizer` or
    :class:`~sionna.phy.ofdm.RZFPrecoder`.

    :param rx_tx_association: A binary NumPy array of shape [num_rx, num_tx]
        where ``rx_tx_association[i,j]=1`` means that receiver `i` gets
        one or multiple streams from transmitter `j`.
    :param num_streams_per_tx: Indicates the number of streams that are
        transmitted by each transmitter.

    .. rubric:: Notes

    Several symmetry constraints on ``rx_tx_association`` are imposed
    to ensure efficient processing. All row sums and all column sums
    must be equal, i.e., all receivers have the same number of associated
    transmitters and all transmitters have the same number of associated
    receivers. It is also assumed that all transmitters send the same
    number of streams ``num_streams_per_tx``. Every transmitted stream is
    assigned to exactly one receiver. A transmitter's streams are divided
    uniformly among its associated receivers in increasing receiver-index
    order. Therefore, ``num_streams_per_tx`` must be divisible by the number
    of receivers associated with each transmitter.
    :class:`~sionna.phy.mimo.StreamManagement` is independent of the actual
    number of antennas at the transmitters and receivers.

    .. rubric:: Examples

    The following code snippet shows how to configure
    :class:`~sionna.phy.mimo.StreamManagement` for a simple uplink scenario,
    where four transmitters send each one stream to a receiver:

    .. code-block:: python

        import numpy as np
        from sionna.phy.mimo import StreamManagement

        num_tx = 4
        num_rx = 1
        num_streams_per_tx = 1

        # Indicate which transmitter is associated with which receiver
        # rx_tx_association[i,j] = 1 means that transmitter j sends one
        # or multiple streams to receiver i.
        rx_tx_association = np.zeros([num_rx, num_tx])
        rx_tx_association[0,0] = 1
        rx_tx_association[0,1] = 1
        rx_tx_association[0,2] = 1
        rx_tx_association[0,3] = 1

        sm = StreamManagement(rx_tx_association, num_streams_per_tx)
    """

    def __init__(
        self,
        rx_tx_association: np.ndarray,
        num_streams_per_tx: int,
    ) -> None:
        self._num_streams_per_tx = int(num_streams_per_tx)
        self.rx_tx_association = rx_tx_association

    @property
    def rx_tx_association(self) -> np.ndarray:
        """Association between receivers and transmitters.

        A binary NumPy array of shape `[num_rx, num_tx]`,
        where ``rx_tx_association[i,j]=1`` means that receiver `i`
        gets one or multiple streams from transmitter `j`.
        """
        return self._rx_tx_association

    @property
    def num_rx(self) -> int:
        """Number of receivers."""
        return self._num_rx

    @property
    def num_tx(self) -> int:
        """Number of transmitters."""
        return self._num_tx

    @property
    def num_streams_per_tx(self) -> int:
        """Number of streams per transmitter."""
        return self._num_streams_per_tx

    @property
    def num_streams_per_rx(self) -> int:
        """Number of streams transmitted to each receiver."""
        return int(self.num_tx * self.num_streams_per_tx / self.num_rx)

    @property
    def num_interfering_streams_per_rx(self) -> int:
        """Number of interfering streams received at each receiver."""
        return int(self.num_tx * self.num_streams_per_tx - self.num_streams_per_rx)

    @property
    def num_tx_per_rx(self) -> int:
        """Number of transmitters communicating with a receiver."""
        return self._num_tx_per_rx

    @property
    def num_rx_per_tx(self) -> int:
        """Number of receivers communicating with a transmitter."""
        return self._num_rx_per_tx

    @property
    def precoding_ind(self) -> np.ndarray:
        """Indices needed to gather channels for precoding.

        A NumPy array of shape `[num_tx, num_rx_per_tx]`,
        where ``precoding_ind[i,:]`` contains the indices of the
        receivers to which transmitter `i` is sending streams.
        """
        return self._precoding_ind

    @property
    def stream_association(self) -> np.ndarray:
        """Association between receivers, transmitters, and streams.

        A binary NumPy array of shape
        `[num_rx, num_tx, num_streams_per_tx]`, where
        ``stream_association[i,j,k]=1`` means that receiver `i` gets
        the `k` th stream from transmitter `j`.
        """
        return self._stream_association

    @property
    def detection_desired_ind(self) -> np.ndarray:
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
    def detection_undesired_ind(self) -> np.ndarray:
        """Indices needed to gather undesired channels for receive processing.

        A NumPy array of shape
        `[num_rx*num_interfering_streams_per_rx]` that
        can be used to gather undesired channels from the flattened
        channel tensor of shape `[...,num_rx, num_tx, num_streams_per_tx,...]`.
        The result of the gather operation can be reshaped to
        `[...,num_rx, num_interfering_streams_per_rx,...]`.
        """
        return self._detection_undesired_ind

    @property
    def tx_stream_ids(self) -> np.ndarray:
        """Mapping of streams to transmitters.

        A NumPy array of shape `[num_tx, num_streams_per_tx]`.
        Streams are numbered from 0,1,... and assigned to transmitters in
        increasing order, i.e., transmitter 0 gets the first
        `num_streams_per_tx` and so on.
        """
        return self._tx_stream_ids

    @property
    def rx_stream_ids(self) -> np.ndarray:
        """Mapping of streams to receivers.

        A NumPy array of shape `[num_rx, num_streams_per_rx]`.
        This array is obtained from ``tx_stream_ids`` together with
        the ``rx_tx_association``. ``rx_stream_ids[i,:]`` contains
        the indices of streams that are supposed to be decoded by receiver `i`.
        """
        return self._rx_stream_ids

    @property
    def stream_ind(self) -> np.ndarray:
        """Indices needed to gather received streams in the correct order.

        A NumPy array of shape `[num_rx*num_streams_per_rx]` that can be
        used to gather streams from the flattened tensor of received streams
        of shape `[...,num_rx, num_streams_per_rx,...]`. The result of the
        gather operation is then reshaped to
        `[...,num_tx, num_streams_per_tx,...]`.
        """
        return self._stream_ind

    @rx_tx_association.setter
    def rx_tx_association(self, rx_tx_association: np.ndarray) -> None:
        """Sets the rx_tx_association and derives related properties."""

        # Make sure that rx_tx_association is a binary NumPy array
        rx_tx_association = np.array(rx_tx_association, np.int32)
        if not all(x in [0, 1] for x in np.nditer(rx_tx_association)):
            raise ValueError(
                "All elements of `rx_tx_association` must be 0 or 1"
            )

        # Obtain num_rx, num_tx from rx_tx_association shape
        self._num_rx, self._num_tx = np.shape(rx_tx_association)

        # Each receiver must be associated with the same number of transmitters
        num_tx_per_rx = np.sum(rx_tx_association, 1)
        if np.min(num_tx_per_rx) != np.max(num_tx_per_rx):
            raise ValueError(
                "Each receiver needs to be associated with the same number "
                "of transmitters."
            )
        self._num_tx_per_rx = num_tx_per_rx[0]

        # Each transmitter must be associated with the same number of receivers
        num_rx_per_tx = np.sum(rx_tx_association, 0)
        if np.min(num_rx_per_tx) != np.max(num_rx_per_tx):
            raise ValueError(
                "Each transmitter needs to be associated with the same "
                "number of receivers."
            )
        self._num_rx_per_tx = num_rx_per_tx[0]
        if self._num_rx_per_tx == 0:
            raise ValueError(
                "Each transmitter must be associated with at least one "
                "receiver."
            )
        if self.num_streams_per_tx % self.num_rx_per_tx != 0:
            raise ValueError(
                "`num_streams_per_tx` must be divisible by the number of "
                "receivers associated with each transmitter."
            )
        num_streams_per_link = self.num_streams_per_tx // self.num_rx_per_tx

        self._rx_tx_association = rx_tx_association

        # Compute indices for precoding
        self._precoding_ind = np.zeros(
            [self.num_tx, self.num_rx_per_tx], np.int32
        )
        for i in range(self.num_tx):
            self._precoding_ind[i, :] = np.where(self.rx_tx_association[:, i])[0]

        # Construct the stream association matrix
        # The element [i,j,k]=1 indicates that receiver i gets the kth stream
        # from transmitter j.
        stream_association = np.zeros(
            [self.num_rx, self.num_tx, self.num_streams_per_tx], np.int32
        )
        tmp = np.ones([num_streams_per_link], np.int32)
        for j in range(self.num_tx):
            c = 0
            for i in range(self.num_rx):
                # If receiver i gets anything from transmitter j
                if rx_tx_association[i, j]:
                    stream_association[
                        i, j, c : c + num_streams_per_link
                    ] = tmp
                    c += num_streams_per_link
        self._stream_association = stream_association

        # Get indices of desired and undesired channel coefficients from
        # the flattened stream_association. These indices can be used by
        # a receiver to gather channels of desired and undesired streams.
        self._detection_desired_ind = np.where(
            np.reshape(stream_association, [-1]) == 1
        )[0]

        self._detection_undesired_ind = np.where(
            np.reshape(stream_association, [-1]) == 0
        )[0]

        # We number streams from 0,1,... and assign them to the TX
        # TX 0 gets the first num_streams_per_tx and so on:
        self._tx_stream_ids = np.reshape(
            np.arange(0, self.num_tx * self.num_streams_per_tx),
            [self.num_tx, self.num_streams_per_tx],
        )

        # We now compute the stream_ids for each receiver
        self._rx_stream_ids = np.zeros(
            [self.num_rx, self.num_streams_per_rx], np.int32
        )
        for i in range(self.num_rx):
            c = []
            for j in range(self.num_tx):
                # If receiver i gets anything from transmitter j
                if rx_tx_association[i, j]:
                    tmp = np.where(stream_association[i, j])[0]
                    tmp = tmp + j * self.num_streams_per_tx
                    c += list(tmp)
            self._rx_stream_ids[i, :] = c

        # Get indices to bring received streams back to the right order in
        # which they were transmitted.
        self._stream_ind = np.argsort(np.reshape(self._rx_stream_ids, [-1]))

