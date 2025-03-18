#
# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0#
"""
Scheduling algorithms for Sionna SYS
"""


import tensorflow as tf
from sionna.phy import Block
from sionna.phy.utils import insert_dims


class PFSchedulerSUMIMO(Block):
    # pylint: disable=line-too-long
    r""" 
    Schedules users according to a proportional fairness (PF) metric in a
    single-user (SU) multiple-input multiple-output (MIMO) system, i.e., at most
    one user is scheduled per time-frequency resource.

    Fixing the time slot :math:`t`, :math:`\tilde{R}_t(u,i)` is
    the :emphasis:`achievable` rate for user :math:`u` on the time-frequency
    resource :math:`i` during the current slot.
    Let :math:`T_{t-1}(u)` denote the throughput :emphasis:`achieved` by user
    :math:`u` up to and including slot :math:`t-1`.
    Resource :math:`i` is assigned to the user with the highest PF metric,
    as defined in [Jalali00]_:

    .. math::
        \operatorname{argmax}_{u} \frac{\tilde{R}_{t}(u,i)}{T_{t-1}(u)}.

    All streams within a scheduled resource element are assigned to the selected user.
    
    Let :math:`R_t(u)` be the rate achieved by user :math:`u` in slot :math:`t`. 
    The throughput :math:`T` by each user :math:`u` is updated via
    geometric discounting: 

    .. math::
        T_t(u) = \beta \, T_{t-1}(u) + (1-\beta) \, R_t(u)

    where :math:`\beta\in(0,1)` is the discount factor.

    Parameters
    ----------

    num_ut : `int`
        Number of user terminals

    num_freq_res : `int`
        Number of available frequency resources. 
        A frequency resource is the smallest frequency unit that can be
        allocated to a user, typically a physical resource block (PRB).

    num_ofdm_sym : `int`
        Number of OFDM symbols in a slot

    batch_size : `list` | `int` | `None` (default)
        Batch size or shape. It can account for multiple sectors in
        which scheduling is performed simultaneously. If `None`, the batch size
        is set to [].

    num_streams_per_ut : `int` (default: 1)
        Number of streams per user

    beta : `float` (default: 0.98)
        Discount factor for computing the time-averaged achieved rate. Must be
        within (0,1).

    precision : `None` (default) | "single" | "double"
        Precision used for internal calculations and outputs.
        If set to `None`,
        :attr:`~sionna.phy.config.Config.precision` is used.

    Input
    -----

    rate_last_slot : [batch_size, num_ut]
        Rate achieved by each user in the last slot

    rate_achievable_curr_slot : [batch_size, num_ofdm_sym, num_freq_res, num_ut], `tf.float`
        Achievable rate for each user across the OFDM grid in the
        current slot

    Output
    ------

    is_scheduled: [batch_size, num_ofdm_sym, num_freq_res, num_ut, num_streams_per_ut], `tf.bool`
        Whether a user is scheduled for transmission for each available resource
    """
    def __init__(self,
                 num_ut,
                 num_freq_res,
                 num_ofdm_sym,
                 batch_size=None,
                 num_streams_per_ut=1,
                 beta=.98,
                 precision=None):
        super().__init__(precision=precision)

        if batch_size is None:
            batch_size = []
        elif (not isinstance(batch_size, list)) and \
                (isinstance(batch_size, int) or (len(batch_size) == 0)):
            batch_size = [batch_size]
        self._batch_size = batch_size
        self._num_ut = int(num_ut)
        self._num_freq_res = int(num_freq_res)
        self._num_ofdm_sym = int(num_ofdm_sym)
        self._num_streams_per_ut = int(num_streams_per_ut)
        self.beta = beta
        self._rate_achieved_past = tf.Variable(
            tf.cast(tf.fill(list(batch_size) + [num_ut], 1), self.rdtype))
        self._pf_metric = tf.Variable(
            tf.zeros(list(batch_size) +
                     [num_ofdm_sym, num_freq_res, num_ut],
                     self.rdtype))

    @property
    def rate_achieved_past(self):
        r"""
        [batch_size, num_ut], `tf.float` (read-only) : :math:`\beta`-discounted time-averaged
        achieved rate for each user
        """
        return self._rate_achieved_past

    @property
    def pf_metric(self):
        r"""
        [batch_size, num_ofdm_sym, num_freq_res, num_ut], `tf.float` (read-only) : Proportional 
        fairness (PF) metric in the last slot 
        """
        return self._pf_metric

    @property
    def beta(self):
        r"""
        `float`: Get/set the discount factor for computing the time-averaged
        achieved rate. Must be within (0,1).
        """
        return self._beta

    @beta.setter
    def beta(self, value):
        tf.debugging.assert_equal(
            0. < value < 1.,
            True,
            message="Discount factor 'beta' must be within (0;1)")
        self._beta = tf.cast(value, self.rdtype)

    def call(self,
             rate_last_slot,
             rate_achievable_curr_slot):

        # ------------------------ #
        # Validate and cast inputs #
        # ------------------------ #
        tf.debugging.assert_equal(
            rate_last_slot.shape,
            self._batch_size + [self._num_ut],
            message="Inconsistent 'rate_last_slot' shape")

        tf.debugging.assert_equal(
            rate_achievable_curr_slot.shape,
            self._batch_size + [self._num_ofdm_sym,
                                self._num_freq_res,
                                self._num_ut],
            message="Inconsistent 'rate_achievable_curr_slot' shape")

        # [batch_size, num_ut]
        rate_last_slot = tf.cast(rate_last_slot, self.rdtype)
        # [batch_size, num_ofdm_sym, num_ut, num_freq_res]
        rate_achievable_curr_slot = tf.cast(rate_achievable_curr_slot,
                                            self.rdtype)

        # ---------------------------- #
        # Update average achieved rate #
        # ---------------------------- #
        # [batch_size, num_ut]
        self._rate_achieved_past.assign(
            self.beta * self._rate_achieved_past +
            (1 - self.beta) * rate_last_slot)
        # [batch_size, 1, 1, num_ut]
        rate_achieved_past = insert_dims(self._rate_achieved_past, 2, axis=-2)

        # ----------------- #
        # Compute PF metric #
        # ----------------- #
        # [batch_size, num_ofdm_sym, num_freq_res, num_ut]
        self._pf_metric.assign(
            rate_achievable_curr_slot / rate_achieved_past)

        # ------------ #
        # Schedule UTs #
        # ------------ #
        # Assign each time/frequency resource to the user with highest PF metric
        # [batch_size, num_ofdm_sym, num_freq_res]
        scheduled_ut = tf.argmax(self._pf_metric, axis=-1)
        # [batch_size, num_ofdm_sym, num_freq_res, num_ut]
        is_scheduled = tf.one_hot(scheduled_ut, depth=self._num_ut)
        # [batch_size, num_ofdm_sym, num_freq_res, num_ut, 1]
        is_scheduled = tf.expand_dims(is_scheduled, axis=-1)
        # [batch_size, num_ofdm_sym, num_freq_res, num_ut, num_streams]
        is_scheduled = tf.tile(is_scheduled, [1]*(3 + len(self._batch_size)) +
                               [self._num_streams_per_ut])

        return tf.cast(is_scheduled, tf.bool)
