#
# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Class for simulating Rayleigh block fading"""

import tensorflow as tf

from . import ChannelModel

class RayleighBlockFading(ChannelModel):
    # pylint: disable=line-too-long
    r"""RayleighBlockFading(num_rx, num_rx_ant, num_tx, num_tx_ant, dtype=tf.complex64)

    Generate channel impulse responses corresponding to a Rayleigh block
    fading channel model.

    The channel impulse responses generated are formed of a single path with
    zero delay and a normally distributed fading coefficient.
    All time steps of a batch example share the same channel coefficient
    (block fading).

    This class can be used in conjunction with the classes that simulate the
    channel response in time or frequency domain, i.e.,
    :class:`~sionna.channel.OFDMChannel`,
    :class:`~sionna.channel.TimeChannel`,
    :class:`~sionna.channel.GenerateOFDMChannel`,
    :class:`~sionna.channel.ApplyOFDMChannel`,
    :class:`~sionna.channel.GenerateTimeChannel`,
    :class:`~sionna.channel.ApplyTimeChannel`.

    Parameters
    ----------

    num_rx : int
        Number of receivers (:math:`N_R`)

    num_rx_ant : int
        Number of antennas per receiver (:math:`N_{RA}`)

    num_tx : int
        Number of transmitters (:math:`N_T`)

    num_tx_ant : int
        Number of antennas per transmitter (:math:`N_{TA}`)

    dtype : tf.DType
        Complex datatype to use for internal processing and output.
        Defaults to `tf.complex64`.

    Input
    -----
    batch_size : int
        Batch size

    num_time_steps : int
        Number of time steps

    Output
    -------
    a : [batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths = 1, num_time_steps], tf.complex
        Path coefficients

    tau : [batch size, num_rx, num_tx, num_paths = 1], tf.float
        Path delays [s]
    """

    def __init__(   self,
                    num_rx,
                    num_rx_ant,
                    num_tx,
                    num_tx_ant,
                    dtype=tf.complex64):

        assert dtype.is_complex, "'dtype' must be complex type"
        self._dtype = dtype

        # We don't set these attributes as private so that the user can update
        # them
        self.num_tx = num_tx
        self.num_tx_ant = num_tx_ant
        self.num_rx = num_rx
        self.num_rx_ant = num_rx_ant

    def __call__(self,  batch_size, num_time_steps, sampling_frequency=None):

        # Delays
        # Single path with zero delay
        delays = tf.zeros([ batch_size,
                            self.num_rx,
                            self.num_tx,
                            1], # Single path
                            dtype=self._dtype.real_dtype)

        # Fading coefficients
        std = tf.cast(tf.sqrt(0.5), dtype=self._dtype.real_dtype)
        h_real = tf.random.normal(shape=[   batch_size,
                                            self.num_rx,
                                            self.num_rx_ant,
                                            self.num_tx,
                                            self.num_tx_ant,
                                            1, # One path
                                            1], # Same response over the block
                                            stddev=std,
                                            dtype = self._dtype.real_dtype)
        h_img = tf.random.normal(shape=[    batch_size,
                                            self.num_rx,
                                            self.num_rx_ant,
                                            self.num_tx,
                                            self.num_tx_ant,
                                            1, # One cluster
                                            1], # Same response over the block
                                            stddev=std,
                                            dtype = self._dtype.real_dtype)
        h = tf.complex(h_real, h_img)
        # Tile the response over the block
        h = tf.tile(h, [1, 1, 1, 1, 1, 1, num_time_steps])
        return h, delays
