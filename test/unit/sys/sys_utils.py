#
# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0#

import os
import tensorflow as tf
import numpy as np

from sionna.phy.mimo import StreamManagement
from sionna.phy import config, dtypes, Block
from sionna.phy.utils import db_to_lin, lin_to_db, sample_bernoulli
from sionna.sys import PHYAbstraction, InnerLoopLinkAdaptation, \
    OuterLoopLinkAdaptation, get_pathloss, open_loop_uplink_power_control, \
    downlink_fair_power_control
from sionna.sys.utils import spread_across_subcarriers, insert_dims


def compute_sinr_numpy(num_streams_per_rx,
                       num_streams_per_tx,
                       num_rx_per_tx,
                       num_tx,
                       thermal_noise_power,
                       equalizer_sel,
                       precoding_sel,
                       rx_tx_association_sel,
                       stream_association_sel,
                       channel_sel,
                       tx_power_sel,
                       link_type):
    r"""
    Numpy implementation of the per-stream SINR computation for a single RX
    """
    signal_per_stream = np.zeros(num_streams_per_rx)
    interference_per_stream = np.zeros(num_streams_per_rx)
    noise_per_stream = np.zeros(num_streams_per_rx)

    for s in range(num_streams_per_rx):
        noise_per_stream[s] = thermal_noise_power * \
            sum(np.power(abs(equalizer_sel[s, :]), 2))

    if link_type == 'DL':

        # TX attached to RX rx_sel
        tx_sel = np.where(rx_tx_association_sel == 1)[0][0]

        # stream for user to stream for TX
        s_rx_sel_to_tx = np.where(
            stream_association_sel[tx_sel, :] == 1)[0]

        # TX
        for tx in range(num_tx):
            H_b = channel_sel[tx, ::]
            # RX to which TX b transmits to
            for rx in range(num_rx_per_tx):
                # stream of user u, that TX b transmits to
                for s_rx_tx in range(num_streams_per_rx):
                    s_tx = rx * num_streams_per_rx + s_rx_tx
                    precoding_vector = precoding_sel[tx,
                                                     :, s_tx][:, np.newaxis]
                    H_b_precoded = np.matmul(
                        H_b, precoding_vector)
                    # stream in reception
                    for s_rx in range(num_streams_per_rx):
                        # received signal
                        y = np.matmul(
                            equalizer_sel[s_rx, :], H_b_precoded)[0]
                        # signal power
                        y2 = abs(y)**2 * \
                            float(tx_power_sel[tx, s_tx])

                        is_intended_for_rx_sel = (
                            stream_association_sel[tx, s_tx] == 1)
                        is_intended_for_s_rx = (
                            s_rx_sel_to_tx[s_rx] == s_tx)

                        if is_intended_for_rx_sel & is_intended_for_s_rx:
                            signal_per_stream[s_rx] += y2
                        else:
                            interference_per_stream[s_rx] += y2

    elif link_type == 'UL':

        # TX's attached to RX rx_sel
        tx_sel = np.where(rx_tx_association_sel == 1)[0]

        # TX
        for tx in range(num_tx):
            H_b = channel_sel[tx, ::]
            # stream in transmission
            for s_tx in range(num_streams_per_tx):
                precoding_vector = precoding_sel[tx,
                                                 :, s_tx][:, np.newaxis]
                H_b_precoded = np.matmul(
                    H_b, precoding_vector)
                # stream in reception
                for s_rx in range(num_streams_per_rx):
                    # received signal
                    y = np.matmul(
                        equalizer_sel[s_rx, :], H_b_precoded)[0]
                    # signal power
                    y2 = abs(y)**2 * float(tx_power_sel[tx, s_tx])

                    is_intended_user = (tx in tx_sel)
                    # is_intended_stream = (s_rx == ((tx%num_tx_per_rx) * num_streams_per_tx + s_tx))
                    if is_intended_user:
                        is_intended_stream_and_user = (s_rx == (list(tx_sel).index(tx) *
                                                                num_streams_per_tx + s_tx))
                    else:
                        is_intended_stream_and_user = False
                    if is_intended_stream_and_user:
                        signal_per_stream[s_rx] += y2
                    else:
                        interference_per_stream[s_rx] += y2

    # Compute SINR while setting 0/0 to 0
    ind0 = (signal_per_stream == 0) & (
        (interference_per_stream + noise_per_stream) == 0)
    sinr = np.zeros(signal_per_stream.shape)
    sinr[ind0] = 0
    sinr[~ind0] = signal_per_stream[~ind0] / \
        (interference_per_stream[~ind0] + noise_per_stream[~ind0])
    return sinr


def wraparound_dist_np(grid, point):
    """ non-TensorFlow wraparound distance function between a point and a
    hexagon center within a spiral hexagonal grid """
    dist_to_bs = []
    for cell in grid.grid:
        # (x,y) coordinates of the centers of the 6 neighbors + current hexagon
        center_neighbors = np.array(
            [[grid.grid[cell].coord_euclid[i] + d[i] for i in [0, 1]] + [grid.cell_height]
             for d in (grid._mirror_displacements_euclid)])

        # distance between point and centers of hexagons having the same
        # relative coordinates in neighboring cells
        dist_point_neighbors = [np.linalg.norm(
            point - c) for c in center_neighbors]
        dist_to_bs.append(min(dist_point_neighbors))
    return dist_to_bs


class MAC(Block):
    r"""
    OLLA + PHY abstraction + HARQ feedback Sionna Block
    """

    def __init__(self,
                 sinr_eff_init,
                 bler_target,
                 olla_delta_up,
                 precision=None):
        super().__init__(precision=precision)

        self._phy_abs = PHYAbstraction(precision=precision)

        batch_size = sinr_eff_init.shape[:-1]
        num_ut = sinr_eff_init.shape[-1]
        self._olla = OuterLoopLinkAdaptation(
            self._phy_abs,
            num_ut,
            batch_size=batch_size,
            sinr_eff_init=sinr_eff_init,
            bler_target=bler_target,
            delta_up=olla_delta_up)

    @tf.function(jit_compile=True)
    def call(self,
             num_allocated_re,
             harq_feedback,
             sinr_eff_true,
             sinr_eff_feedback):

        # Link Adaptation
        mcs_index = self._olla(
            num_allocated_re,
            harq_feedback=harq_feedback,
            sinr_eff=sinr_eff_feedback)

        # Generate BLER and HARQ feedback via PHYAbstraction
        _, harq_feedback, _, tbler, _ = self._phy_abs(
            mcs_index,
            sinr_eff=sinr_eff_true,
            num_allocated_re=num_allocated_re)

        return mcs_index, tbler, harq_feedback, self._olla.offset


class SINREffFeedback(Block):
    r"""
    Generate SINR evolution and feedback
    """

    def __init__(self,
                 shape,
                 prob_feedback=.5,
                 bounds=(5, 21),
                 precision=None):
        super().__init__(precision=precision)
        self._bounds = tf.constant(bounds, dtype=self.rdtype)
        self._prob_feedback = prob_feedback
        self.true_val_db = config.tf_rng.uniform(shape=shape,
                                                 minval=self._bounds[0],
                                                 maxval=self._bounds[1],
                                                 dtype=self.rdtype)

    def call(self):
        self.true_val_db = self.true_val_db + config.tf_rng.uniform(
            shape=self.true_val_db.shape,
            minval=-1,
            maxval=1,
            dtype=self.rdtype)
        self.true_val_db = tf.maximum(self.true_val_db, self._bounds[0])
        self.true_val_db = tf.minimum(self.true_val_db, self._bounds[1])
        p = sample_bernoulli(shape=self.true_val_db.shape,
                             p=self._prob_feedback)
        sinr_feedback = tf.where(p,
                                 db_to_lin(self.true_val_db,
                                           precision=self.precision),
                                 tf.cast(0., self.rdtype))
        return sinr_feedback


def gen_num_allocated_re(prob_being_scheduled,
                         shape,
                         bounds):
    """
    Generate random number of allocated streams
    """
    num_allocated_re = config.tf_rng.uniform(
        shape=shape,
        minval=bounds[0], maxval=bounds[1], dtype=tf.int32)
    p = sample_bernoulli(shape, p=prob_being_scheduled)
    num_allocated_re = tf.where(
        p, num_allocated_re, tf.cast(0, tf.int32))
    return num_allocated_re


def get_stream_management(direction,
                          num_rx,
                          num_tx,
                          num_streams_per_ut):
    """
    Instantiate a StreamManagement object.
    It determines which data streams are intended for each receiver
    """
    if direction == 'downlink':
        num_ut_per_sector = int(num_rx / num_tx)
        num_streams_per_tx = num_streams_per_ut * num_ut_per_sector

        # RX-TX association matrix
        rx_tx_association = np.zeros([num_rx, num_tx])
        idx = np.array([[i1, i2] for i2 in range(num_tx) for i1 in
                        np.arange(i2*num_ut_per_sector, (i2+1)*num_ut_per_sector)])
        rx_tx_association[idx[:, 0], idx[:, 1]] = 1

    else:
        num_ut_per_sector = int(num_tx / num_rx)
        num_streams_per_tx = num_streams_per_ut

        # RX-TX association matrix
        rx_tx_association = np.zeros([num_rx, num_tx])
        idx = np.array([[i1, i2] for i1 in range(num_rx) for i2 in
                        np.arange(i1*num_ut_per_sector, (i1+1)*num_ut_per_sector)])
        rx_tx_association[idx[:, 0], idx[:, 1]] = 1

    stream_management = StreamManagement(rx_tx_association, num_streams_per_tx)
    return stream_management, num_streams_per_tx


@tf.function(jit_compile=True)
def get_pathloss_xla(a,
                     rx_tx_association=None,
                     precision=None):
    return get_pathloss(a,
                        rx_tx_association=rx_tx_association,
                        precision=precision)


@tf.function(jit_compile=True)
def open_loop_uplink_power_control_xla(
        pathloss,
        num_allocated_re,
        alpha=1.,
        p0_dbm=-90.,
        precision=None):
    return open_loop_uplink_power_control(
        pathloss,
        num_allocated_re,
        alpha=alpha,
        p0_dbm=p0_dbm,
        precision=precision)


@tf.function(jit_compile=True)
def downlink_fair_power_control_xla(
        pathloss,
        interference_plus_noise,
        num_resources,
        bs_max_power_dbm,
        fairness=0.,
        return_lagrangian=False,
        precision=None,
        **kwargs):
    return downlink_fair_power_control(
        pathloss,
        interference_plus_noise,
        num_resources,
        bs_max_power_dbm=bs_max_power_dbm,
        fairness=fairness,
        return_lagrangian=return_lagrangian,
        precision=precision,
        **kwargs)


@tf.function(jit_compile=True)
def spread_across_subcarriers_xla(tx_power_per_ut,
                                  is_scheduled,
                                  precision=None):
    return spread_across_subcarriers(tx_power_per_ut,
                                     is_scheduled,
                                     precision=precision)


def pf_scheduler_multislot(pf_sched, rate_achievable_avg, num_slots):
    """
    PF scheduler loop over slots
    """
    batch_size = pf_sched._batch_size
    num_ut = pf_sched._num_ut
    num_freq_res = pf_sched._num_freq_res
    num_time_samples = pf_sched._num_ofdm_sym
    num_streams = pf_sched._num_streams_per_ut

    # Average achievable rate
    # [batch_size, num_time_samples, num_freq_res, num_ut]
    rate_achievable_avg = insert_dims(rate_achievable_avg, 2, axis=-2)
    rate_achievable_avg = tf.tile(rate_achievable_avg,
                                  [1] * len(batch_size) + [num_time_samples, num_freq_res, 1])
    # Rate achieved over last slot
    rate_last_slot = tf.zeros(batch_size + [num_ut])

    hist = {'is_scheduled': tf.TensorArray(size=num_slots,
                                           element_shape=batch_size +
                                           [num_time_samples, num_freq_res,
                                            num_ut, num_streams],
                                           dtype=tf.int32),
            'rate_achievable': tf.TensorArray(size=num_slots,
                                              element_shape=batch_size +
                                              [num_time_samples,
                                                  num_freq_res, num_ut],
                                              dtype=rate_achievable_avg.dtype)
            }

    def body(slot, rate_last_slot, hist):
        # Update achievable rate according to an autoregressive model
        rate_achievable_curr = rate_achievable_avg + \
            config.tf_rng.uniform(rate_achievable_avg.shape,
                                  dtype=rate_achievable_avg.dtype,
                                  minval=-2, maxval=2)
        rate_achievable_curr = tf.maximum(
            rate_achievable_curr, tf.cast(0, rate_achievable_avg.dtype))

        # Schedule UTs on available resources
        # [batch_size, num_time_samples, num_freq_res, num_ut, num_streams]
        is_scheduled = pf_sched(rate_last_slot, rate_achievable_curr)
        # [batch_size, num_time_samples, num_freq_res, num_ut]
        is_scheduled_ut = tf.reduce_all(is_scheduled, axis=-1)

        # Achieved rate
        rate_last_slot = tf.reduce_sum(tf.cast(is_scheduled_ut, rate_achievable_avg.dtype) *
                                       rate_achievable_curr, axis=[-3, -2])

        # Record history
        hist['is_scheduled'] = hist['is_scheduled'].write(
            slot, tf.cast(is_scheduled, tf.int32))
        hist['rate_achievable'] = hist['rate_achievable'].write(
            slot, rate_achievable_curr)

        slot += 1

        return [slot, rate_last_slot, hist]

    slot = 0
    *_, hist = tf.while_loop(
        lambda slot, *_: slot < num_slots,
        body,
        [slot, rate_last_slot, hist]
    )

    for key in hist:
        hist[key] = hist[key].stack()
    return hist


@tf.function(jit_compile=True)
def pf_scheduler_multislot_xla(pf_sched, rate_achievable_avg, num_slots):
    return pf_scheduler_multislot(pf_sched, rate_achievable_avg, num_slots)
