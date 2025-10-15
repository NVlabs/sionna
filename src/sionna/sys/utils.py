#
# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0#
"""
Utility functions for Sionna SYS
"""

import tensorflow as tf
from sionna.phy.utils import config, dtypes, \
    insert_dims, tensor_values_are_in_set


def is_scheduled_in_slot(sinr=None,
                         num_allocated_re=None):
    # pylint: disable=line-too-long
    r"""
    Determines whether a user is scheduled in a slot

    Input
    -----
    sinr : [..., num_ofdm_symbols, num_subcarriers, num_ut, num_streams_per_ut], `tf.float` | `None` (default)
        SINR for each OFDM symbol, subcarrier, user, and stream. 
        If `None`, then ``num_allocated_re`` is required.

    num_allocated_re : [..., num_ut], `tf.float` | `None` (default)
        Number of allocated resources (streams/REs/PRBs etc.) per user. 
        If `None`, then ``sinr`` is required.

    Output
    ------
    is_scheduled : [..., num_ut] : `tf.bool`
        Whether a user is scheduled in a slot
    """

    tf.debugging.assert_equal(
        (sinr is not None) ^
        (num_allocated_re is not None),
        True,
        message="Either 'sinr' or "
        "'sinr_eff' is required as input")

    if sinr is not None:
        return tf.reduce_sum(sinr, axis=[-4, -3, -1]) > \
            tf.cast(0., sinr.dtype)
    else:
        return num_allocated_re > tf.cast(0., num_allocated_re.dtype)


def get_pathloss(h_freq,
                 rx_tx_association=None,
                 precision=None):
    # pylint: disable=line-too-long
    r"""
    Computes the pathloss for each receiver-transmitter pair and, if the
    receiver-transmitter association is provided, the pathloss between each
    user and the associated base station

    Input
    -----

    h_freq : [..., num_rx, num_rx_ant, num_tx, num_tx_ant, num_ofdm_symbols, num_subcarriers], `tf.complex` | `None` (default: `None`)
        OFDM channel matrix

    rx_tx_association : [num_rx, num_tx], `tf.int32`, `None` (default)
        Its :math:`(i,j)` element is 1 if receiver :math:`i` is attached to
        transmitter :math:`j`, 0 otherwise

    precision : `None` (default) | "single" | "double"
        Precision used for internal calculations and outputs.
        If set to `None`,
        :attr:`~sionna.phy.config.Config.precision` is used.

    Output
    ------
    pathloss_all_pairs : [..., num_rx, num_tx, num_ofdm_symbols], `tf.float`
        Pathloss for each RX-TX pair and across OFDM symbols

    pathloss_serving_tx : [..., num_ut, num_ofdm_symbols], `tf.float`
        Pathloss between each user and the associated base station. Only computed
        if ``rx_tx_association`` is provided as input.
    """
    if precision is None:
        rdtype = config.tf_rdtype
    else:
        rdtype = dtypes[precision]["tf"]["rdtype"]

    batch_size = h_freq.shape[:-6]
    lbs = len(batch_size)
    num_ofdm_symbols = h_freq.shape[-2]

    # Compute RX power
    # [..., num_rx, num_rx_ant, num_tx, num_tx_ant, num_ofdm_symbols, num_subcarriers]
    rx_power = tf.cast(tf.abs(h_freq)**2, rdtype)

    # Average across TX/RX antennas and subcarriers
    # [..., num_rx, num_tx, num_ofdm_symbols]
    rx_power = tf.reduce_mean(rx_power, axis=[-1, -3, -5])

    # Get pathloss
    # [..., num_rx, num_tx, num_ofdm_symbols]
    pathloss_all_pairs = tf.cast(1., rdtype) / rx_power

    if rx_tx_association is None:
        pathloss_serving_tx = None
    else:
        tf.debugging.assert_equal(
            tensor_values_are_in_set(rx_tx_association, [0, 1]),
            True,
            message="rx_tx_association must contain binary values")

        # Number of UTs
        num_ut = tf.reduce_sum(rx_tx_association)

        # Extract pathloss for serving TX only, for each RX
        rx_tx_association = rx_tx_association == 1
        # [batch_size, num_rx, num_tx]
        rx_tx_association = insert_dims(
            rx_tx_association, lbs, axis=0)
        rx_tx_association = tf.tile(
            rx_tx_association, batch_size + [1, 1])

        # [batch_size, num_rx, num_tx, num_ofdm_symbols]
        rx_tx_association = insert_dims(
            rx_tx_association, 1, axis=-1)
        rx_tx_association = tf.tile(
            rx_tx_association, [1]*(lbs + 2) + [num_ofdm_symbols])

        # [num_ut*prod(batch_size), num_ofdm_symbols]
        pathloss_serving_tx = tf.gather_nd(
            pathloss_all_pairs, tf.where(rx_tx_association))

        # [batch_size, num_ut, num_ofdm_symbols]
        pathloss_serving_tx = tf.reshape(
            pathloss_serving_tx, list(batch_size) + [num_ut, num_ofdm_symbols])

    return pathloss_all_pairs, pathloss_serving_tx


def spread_across_subcarriers(tx_power_per_ut,
                              is_scheduled,
                              num_tx=None,
                              precision=None):
    # pylint: disable=line-too-long
    r"""
    Distributes the power uniformly across all allocated subcarriers
    and streams for each user

    Input
    -----
    tx_power_per_ut : [..., num_ofdm_sym, num_ut], `tf.float`
        Transmit power [W] for each user

    is_scheduled : [..., num_ofdm_sym, num_subcarriers, num_ut, num_streams_per_ut], `tf.bool`
        Whether a user is scheduled on a given subcarrier and stream

    num_tx : `int` | `None` (default)
        Number of transmitters. If `None`, it is set to `num_ut`, as in uplink.

    precision : `None` (default) | "single" | "double"
        Precision used for internal calculations and outputs.
        If set to `None`,
        :attr:`~sionna.phy.config.Config.precision` is used.

    Output
    ------
    tx_power : [..., num_tx, num_streams_per_tx, num_ofdm_sym, num_subcarriers], `tf.float`
        Transmit power [W] for each user, across subcarriers, streams,
        and OFDM symbols
    """
    if precision is None:
        rdtype = config.tf_rdtype
    else:
        rdtype = dtypes[precision]["tf"]["rdtype"]

    tx_power_per_ut = tf.cast(tx_power_per_ut, rdtype)
    num_ofdm_sym, num_subcarriers, num_ut, num_streams_per_ut = is_scheduled.shape[-4:]
    lbs = len(is_scheduled.shape) - 4

    if num_tx is None:
        num_tx = num_ut

    # [..., num_ofdm_sym, num_ut, num_subcarriers, num_streams_per_ut]
    is_scheduled = tf.transpose(is_scheduled,
                                list(range(lbs)) + [lbs, lbs+2, lbs+1, lbs+3])

    # [..., num_ofdm_sym, num_ut, 1, 1]
    tx_power = insert_dims(tx_power_per_ut, 2, axis=-1)
    # Tile to [..., num_ofdm_sym, num_ut, num_subcarriers, num_streams_per_ut]
    tx_power = tf.tile(tx_power,
                       [1]*(lbs+2) + [num_subcarriers, num_streams_per_ut])
    # Mask according to scheduling decisions
    # [..., num_ofdm_sym, num_ut, num_subcarriers, num_streams_per_ut]
    tx_power = tf.where(is_scheduled,
                        tx_power,
                        tf.cast(0., rdtype))

    # N. allocated resources per user
    # [..., num_ofdm_sym, num_ut]
    num_allocated_re = tf.reduce_sum(
        tf.cast(is_scheduled, tf.int32), axis=[-2, -1])
    # [..., num_ofdm_sym, num_ut, 1, 1]
    num_allocated_re = insert_dims(num_allocated_re, 2, axis=-1)

    # Spread power equally across streams for each user
    # [..., num_ofdm_sym, num_ut, num_subcarriers, num_streams_per_ut]
    tx_power = tf.where(
        num_allocated_re > tf.cast(0, tf.int32),
        tx_power / tf.cast(num_allocated_re, rdtype),
        tf.cast(0, rdtype))

    # [..., num_ut, num_streams_per_ut, num_ofdm_sym, num_subcarriers]
    tx_power = tf.transpose(tx_power,
                            list(range(lbs)) + [lbs+1, lbs+3, lbs, lbs+2])

    # [..., num_tx, num_streams_per_tx, num_ofdm_sym, num_subcarriers]
    tx_power = tf.reshape(tx_power,
                          list(tx_power.shape[:-4]) + [num_tx, -1, num_ofdm_sym, num_subcarriers])
    return tx_power
