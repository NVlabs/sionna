#
# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0#
"""Abstract class proving an interface for channel models"""

from abc import abstractmethod
from sionna.phy.block import Object

class ChannelModel(Object):
    # pylint: disable=line-too-long
    r"""
    Abstract class that defines an interface for channel models

    Any channel model which generates channel impulse responses
    must implement this interface.
    All the channel models available in Sionna,
    such as :class:`~sionna.phy.channel.RayleighBlockFading`
    or :class:`~sionna.phy.channel.tr38901.TDL`, implement this interface.

    *Remark:* Some channel models only require a subset of the input parameters.

    Parameters
    ----------
    precision : `None` (default) | "single" | "double"
        Precision used for internal calculations and outputs.
        If set to `None`,
        :attr:`~sionna.phy.config.Config.precision` is used.

    Input
    -----
    batch_size : `int`
        Batch size

    num_time_steps : `int`
        Number of time steps

    sampling_frequency : `float`
        Sampling frequency [Hz]

    Output
    -------
    a : [batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_steps], `tf.complex`
        Path coefficients

    tau : [batch size, num_rx, num_tx, num_paths], `tf.float`
        Path delays [s]
    """
    def __init__(self, precision=None, **kwargs):
        super().__init__(precision=precision, **kwargs)

    @abstractmethod
    def __call__(self,  batch_size, num_time_steps, sampling_frequency):

        return NotImplemented
