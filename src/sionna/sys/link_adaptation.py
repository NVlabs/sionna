# pylint: disable=line-too-long
#
# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0#
"""
Link adaptation for Sionna SYS
"""

import tensorflow as tf
from sionna.phy import Block
from sionna.phy.utils import find_true_position, insert_dims, \
    scalar_to_shaped_tensor, tensor_values_are_in_set, \
    lin_to_db, db_to_lin
from sionna.sys.utils import is_scheduled_in_slot


class InnerLoopLinkAdaptation(Block):
    # pylint: disable=line-too-long
    r"""
    Class for inner loop link adaptation (ILLA). It computes the highest
    available modulation and coding scheme (MCS) whose 
    associated transport block error rate (TBLER) does not exceed the specified 
    ``bler_target``:

    .. math::

        \max \left\{ \text{MCS}: \ \text{TBLER}(\text{MCS}, \text{SINR}_{\text{eff}}) \le \text{BLER}_{\text{target}} \right\}

    where :math:`\text{SINR}_{\text{eff}}` is the effective SINR value provided
    as input.
    If no such MCS exists, the lowest available MCS index is returned. If a user
    is not scheduled, ``fill_mcs_value`` is returned.

    Parameters
    ----------

    phy_abstraction : :class:`~sionna.sys.PHYAbstraction`
        An instance of :class:`~sionna.sys.PHYAbstraction`

    bler_target : `float` (default: 0.1)
        BLER target

    fill_mcs_value : `int` (default: 0)
        MCS value assigned to non-scheduled users

    Input
    -----

    sinr : [..., num_ofdm_symbols, num_subcarriers, num_ut, num_streams_per_ut], `tf.float` | `None` (default)
        SINR for each OFDM symbol, subcarrier, user and stream. 
        If `None`, then ``sinr_eff`` and ``num_allocated_re`` are both
        required. 

    sinr_eff : [..., num_ut], `tf.float` | `None` (default)
        Estimated effective SINR for each user. 
        If `None`, then ``sinr`` is required.

    num_allocated_re : [..., num_ut], `tf.int32` | `None` (default)
        Number of allocated resources in a slot, computed across OFDM symbols,
        subcarriers and streams, for each user.
        If `None`, then ``sinr`` is required.

    mcs_table_index : [..., num_ut], `tf.int32` | `int` (default: 1)
        MCS table index for each user. For further details, refer to the 
        :ref:`mcs_table_cat_note`.

    mcs_category : [..., num_ut], `tf.int32` | `int` (default: 0)
        MCS table category for each user. For further details, refer to the 
        :ref:`mcs_table_cat_note`.

    return_lowest_available_mcs : `bool` (default: `False`)
        If `True`, the lowest MCS available in ``phy_abstraction`` BLER tables
        is returned for each user. Only used for internal purposes.

    Output
    ------

    mcs_index : [..., num_ut]
        Highest available MCS whose BLER does not exceed the target, or the
        lowest available MCS if no such MCS exists, for each user

    Example
    -------

    .. code-block:: python

        from sionna.sys import PHYAbstraction, InnerLoopLinkAdaptation

        bler_target = 0.1

        # Initialize the PHY abstraction object
        phy_abs = PHYAbstraction()

        # Initialize the ILLA object
        illa = InnerLoopLinkAdaptation(phy_abs,
                                    bler_target=0.1)

        # Effective SINR for each user
        sinr_eff = tf.Variable([0.1, 10, 100])
        # N. allocated resource elements for each user
        num_allocated_re = tf.Variable([20, 30, 30])

        # Compute the MCS index for each user
        mcs_index = illa(sinr_eff=sinr_eff,
                        num_allocated_re=num_allocated_re,
                        mcs_table_index=1,
                        mcs_category=0)
        print('Selected MCS index =', mcs_index.numpy())
        % Selected MCS index = [ 3 16 27]
    """

    def __init__(self,
                 phy_abstraction,
                 bler_target=0.1,
                 fill_mcs_value=0):

        super().__init__(precision=phy_abstraction.precision)
        self._phy_abstraction = phy_abstraction
        self._fill_mcs_value = tf.cast(fill_mcs_value, tf.int32)
        self._bler_target = tf.Variable(tf.cast(bler_target, self.rdtype))

    @property
    def bler_target(self):
        r"""
        `tf.float` : Get/set the BLER target for each user
        """
        return self._bler_target

    @bler_target.setter
    def bler_target(self, value):
        self._bler_target.assign(tf.cast(value, self.rdtype))

    def call(self,
             sinr=None,
             sinr_eff=None,
             num_allocated_re=None,
             mcs_table_index=1,
             mcs_category=0,
             return_lowest_available_mcs=False,
             **kwargs):

        tf.debugging.assert_equal(
            (sinr is not None) ^
            ((sinr_eff is not None) and (num_allocated_re is not None)),
            True,
            message="Either 'sinr' or "
            "('sinr_eff','num_allocated_re') is required as input")

        # N. available MCS indices
        num_mcs = self._phy_abstraction.bler_table_interp.shape[2]

        # Check which UTs are scheduled
        ut_is_scheduled = is_scheduled_in_slot(
            sinr=sinr,
            num_allocated_re=num_allocated_re)

        # Cast and reshape inputs
        if sinr is not None:
            sinr = tf.cast(sinr, self.rdtype)
            batch_dims = sinr.shape[:-4]
            num_ut = sinr.shape[-2]
        else:
            sinr_eff = tf.cast(sinr_eff, self.rdtype)
            batch_dims = sinr_eff.shape[:-1]
            num_ut = sinr_eff.shape[-1]

        # ----------------------- #
        # Tile across MCS indices #
        # ----------------------- #
        # [..., num_mcs, num_ut]
        mcs_index_all = tf.range(num_mcs, dtype=tf.int32)
        mcs_index_all = insert_dims(mcs_index_all, len(batch_dims),
                                    axis=0)[..., tf.newaxis]
        mcs_index_all = tf.tile(mcs_index_all, batch_dims + [1, num_ut])

        # [..., num_mcs, num_ut]
        mcs_table_index = scalar_to_shaped_tensor(
            mcs_table_index, tf.int32, batch_dims + [num_ut])
        mcs_table_index = tf.tile(
            tf.expand_dims(mcs_table_index, axis=-2),
            [1]*(len(mcs_table_index.shape)-1) + [num_mcs, 1])

        # [..., num_mcs, num_ut]
        mcs_category = scalar_to_shaped_tensor(
            mcs_category, tf.int32, batch_dims + [num_ut])
        mcs_category = tf.tile(
            tf.expand_dims(mcs_category, axis=-2),
            [1]*(len(mcs_category.shape)-1) + [num_mcs, 1])

        if num_allocated_re is not None:
            # [..., num_ut]
            num_allocated_re = tf.cast(num_allocated_re, tf.int32)
            num_allocated_re = tf.expand_dims(
                num_allocated_re, axis=-2)
            num_allocated_re = tf.tile(num_allocated_re,
                                       [1]*len(batch_dims) + [num_mcs, 1])

        # -------------- #
        # Effective SINR #
        # -------------- #
        # Expand across all possible MCS indices
        if sinr is not None:
            # [..., num_mcs, num_ofdm_symbols, num_subcarriers, num_ut, num_streams_per_ut]
            sinr = tf.expand_dims(sinr, axis=-5)
            sinr = tf.tile(
                sinr, [1]*len(batch_dims) + [num_mcs] + [1]*4)
        else:
            # [..., num_mcs, num_ut]
            sinr_eff = tf.expand_dims(sinr_eff, axis=-2)
            sinr_eff = tf.tile(sinr_eff,
                               [1]*len(batch_dims) + [num_mcs, 1])

        # ----- #
        # TBLER #
        # ----- #
        # [..., num_mcs, num_ut]
        *_, tbler_per_mcs, _ = self._phy_abstraction(
            mcs_index_all,
            sinr=sinr,
            sinr_eff=sinr_eff,
            num_allocated_re=num_allocated_re,
            mcs_table_index=mcs_table_index,
            mcs_category=mcs_category,
            check_mcs_index_validity=False)

        # ---------- #
        # Select MCS #
        # ---------- #
        # Check that each user has at least one available MCS
        num_unavailable_mcs = tf.reduce_sum(
            tf.cast(tbler_per_mcs > 1, tf.int32), axis=-2)
        tf.debugging.assert_equal(
            tf.reduce_any(num_unavailable_mcs == num_mcs),
            False,
            message='No MCS index available for some users')

        # Find the highest MCS with BLER <= bler_target
        # If no such MCS is found, then returns -1
        # [..., num_ut]
        mcs_index = find_true_position(
            tbler_per_mcs <= self.bler_target,
            side='last',
            axis=-2)

        # Lowest available MCS
        # [..., num_ut]
        lowest_available_mcs = find_true_position(
            (tbler_per_mcs >= 0) & (tbler_per_mcs <= 1),
            side='first',
            axis=-2)

        # If all MCS have BLER > bler_target, select lowest available MCS
        mcs_index = tf.where(mcs_index != -1,
                             mcs_index,
                             lowest_available_mcs)

        # A non-scheduled user receives MCS=_fill_mcs_value
        mcs_index = tf.where(ut_is_scheduled,
                             mcs_index,
                             self._fill_mcs_value)

        if return_lowest_available_mcs:
            return mcs_index, lowest_available_mcs
        return mcs_index


class OuterLoopLinkAdaptation(Block):
    # pylint: disable=line-too-long
    r"""
    Class for outer-loop link adaptation (OLLA). 
    The modulation and coding scheme (MCS) index for a user is determined as the
    highest index whose corresponding transport block error rate (TBLER) remains
    below the specified ``bler_target``. 
    The SINR value used for TBLER computation is given by the last effective
    SINR feedback, :math:`\text{SINR}_{\text{eff}}` [dB], reduced by an offset
    value, :math:`\Delta_{\mathrm{offset}}`: 

    .. math::

        \max \left\{ \text{MCS}: \ \text{TBLER}(\text{MCS}, \text{SINR}_{\text{eff}}-\Delta_{\text{offset}}) \le \text{BLER}_{\text{target}} \right\}

    The value of :math:`\Delta_{\text{offset}}` is adjusted depending on the HARQ feedback [Pedersen05]_:

    .. math::

        \Delta_{\mathrm{offset}} = \left\{
        \begin{array}{l}
            \Delta_{\mathrm{offset}} - \Delta_{\mathrm{down}} \quad \text{if HARQ=ACK} \\
            \Delta_{\mathrm{offset}} + \Delta_{\mathrm{up}} \quad \text{if HARQ=NACK}
        \end{array}
        \right.

    where the relationship between
    :math:`\Delta_{\mathrm{up}}` and :math:`\Delta_{\mathrm{down}}` is given by
    [Sampath97]_: 

    .. math::
        \frac{\Delta_{\mathrm{up}}}{\Delta_{\mathrm{down}}} = \frac{1 - \mathrm{BLER}_{\mathrm{target}}}{\mathrm{BLER}_{\mathrm{target}}}.

    Parameters
    ----------

    phy_abstraction : :class:`~sionna.sys.PHYAbstraction`
        An instance of :class:`~sionna.sys.PHYAbstraction`

    num_ut : `int`
        Number of user terminals

    bler_target : `float` (default: 0.1)
        BLER target value, within 0 and 1

    delta_up : `float` (default: 1.)
        Increment applied to the SINR offset [dB] when a NACK is received for a
        user 

    batch_size : `list` | `int` | `None` (default)
        Batch size or shape. It accounts for multiple users for which link
        adaptation is performed simultaneously. If `None`, the batch size is
        set to [].

    sinr_eff_init : [..., num_ut], `tf.float` | `float` (default: 1.)
        Initial value of effective SINR for each user. Non-positive values are
        treated as missing and replaced by ``sinr_eff_init_fill``.
        If `float`, the same value is assigned to all users.

    sinr_eff_init_fill : `float` (default: 1.)
        Value replacing non-positive ``sinr_eff_init`` values

    offset_min : `float` (default: -20.)
        Minimum SINR [dB] offset value

    offset_max : `float` (default: 20.)
        Maximum SINR [dB] offset value

    Input
    -----

    num_allocated_re : [..., num_ut], `tf.int32`
        Number of allocated resources in the upcoming slot, computed across OFDM
        symbols, subcarriers and streams, for each user

    harq_feedback : [..., num_ut], -1 | 0 | 1
        If 0 (1, resp.), then a NACK (ACK, resp.) is received. If -1, feedback
        is missing.

    sinr_eff : [..., num_ut], `tf.float` | `None` (default)
        Estimated effective SINR for each user. Non-positive values are treated as
        missing. 

    mcs_table_index : [..., num_ut], `tf.int32` | `int` (default: 1)
        MCS table index for each user. For further details, refer to the 
        :ref:`mcs_table_cat_note`.

    mcs_category : [..., num_ut], `tf.int32` | `int` (default: 0)
        MCS table category for each user. For further details, refer to the 
        :ref:`mcs_table_cat_note`.

    Output
    ------

    mcs_index : [..., num_ut]
        Selected MCS index for each user
    """

    def __init__(self,
                 phy_abstraction,
                 num_ut,
                 bler_target=0.1,
                 delta_up=1.,
                 batch_size=None,
                 sinr_eff_init=1.,
                 sinr_eff_init_fill=1.,
                 offset_min=-20.,
                 offset_max=20.):

        super().__init__(precision=phy_abstraction.precision)
        tf.debugging.assert_non_negative(
            sinr_eff_init_fill,
            message="'sinr_eff_init_fill' must be positive")

        if batch_size is None:
            batch_size = []
        elif (not isinstance(batch_size, list)) and \
                (isinstance(batch_size, int) or (len(batch_size) == 0)):
            batch_size = [batch_size]

        self._batch_size = batch_size
        self._num_ut = num_ut
        self._phy_abstraction = phy_abstraction
        self._illa = InnerLoopLinkAdaptation(phy_abstraction,
                                             bler_target=bler_target)

        self._bler_target = tf.Variable(tf.cast(bler_target, self.rdtype))
        self._delta_up = tf.Variable(tf.cast(delta_up, self.rdtype))
        self._delta_down = tf.Variable(self._get_delta_down())
        self._sinr_eff_db_last = None
        self._offset = None

        self._offset_min = tf.Variable(tf.cast(offset_min, self.rdtype))
        self._offset_max = tf.Variable(tf.cast(offset_max, self.rdtype))

        sinr_eff_init = scalar_to_shaped_tensor(
            sinr_eff_init,
            self.rdtype,
            self._batch_size + [self._num_ut])
        sinr_eff_init_fill = tf.cast(sinr_eff_init_fill, self.rdtype)
        # Convert effective SINR to dB and fill N/A values (<=0)
        self._sinr_eff_db_last = tf.Variable(
            tf.where(
                sinr_eff_init > 0,
                lin_to_db(sinr_eff_init, precision=self.precision),
                lin_to_db(sinr_eff_init_fill, precision=self.precision)))
        # Reset SINR offset to 0
        self._offset = tf.Variable(
            tf.zeros(sinr_eff_init.shape, dtype=self.rdtype))

    def _get_delta_down(self):
        return tf.cast(
            self.delta_up * self.bler_target / (1 - self.bler_target),
            self.rdtype)

    def reset(self,
              sinr_eff_init=1.,
              sinr_eff_init_fill=.1):
        """
        Resets the values of ``sinr_eff_db_last`` and ``offset``
        """
        sinr_eff_init = scalar_to_shaped_tensor(
            sinr_eff_init,
            self.rdtype,
            self._batch_size + [self._num_ut])
        sinr_eff_init_fill = tf.cast(sinr_eff_init_fill, self.rdtype)
        # Convert effective SINR to dB and fill N/A values (<=0)
        self._sinr_eff_db_last.assign(
            tf.where(
                sinr_eff_init > 0,
                lin_to_db(sinr_eff_init, precision=self.precision),
                lin_to_db(sinr_eff_init_fill, precision=self.precision)))
        # Reset SINR offset to 0
        self._offset.assign(
            tf.zeros(sinr_eff_init.shape, dtype=self.rdtype))

    @property
    def offset(self):
        r"""
        [..., num_ut], `tf.float` (read-only) : Effective SINR [dB] offset for each user
        """
        return self._offset

    @property
    def offset_max(self):
        """
        `tf.float` : Get/set the maximum ``offset`` value
        """
        return self._offset_max

    @offset_max.setter
    def offset_max(self, value):
        self._offset_max.assign(tf.cast(value, self.rdtype))

    @property
    def offset_min(self):
        """
        `tf.float` : Get/set the minimum ``offset`` value
        """
        return self._offset_min

    @offset_min.setter
    def offset_min(self, value):
        self._offset_min.assign(tf.cast(value, self.rdtype))

    @property
    def bler_target(self):
        r"""
        `tf.float` : Get/set the BLER target for each user
        """
        return self._bler_target

    @bler_target.setter
    def bler_target(self, value):
        self._bler_target.assign(tf.cast(value, self.rdtype))
        self._delta_down.assign(self._get_delta_down())

    @property
    def sinr_eff_db_last(self):
        r"""
        [..., num_ut], `tf.float` : Get/set the last observed effective SINR
        [dB] value for each user
        """
        return self._sinr_eff_db_last

    @sinr_eff_db_last.setter
    def sinr_eff_db_last(self, value):
        self._sinr_eff_db_last.assign(tf.cast(value, self.rdtype))

    @property
    def delta_down(self):
        r"""
        `float` (read-only) : Decrement applied to the SINR offset when an
        ACK is received for a user. Computed as ``delta_up * bler_target
        / (1 - bler_target)``.
        """
        return self._delta_down

    @property
    def delta_up(self):
        r"""
        `float` : Get/set the increment applied to the SINR offset when a
        NACK is received for a user
        """
        return self._delta_up

    @delta_up.setter
    def delta_up(self, value):
        tf.debugging.assert_positive(
            value,
            message="'delta_up' must be positive")
        self._delta_up.assign(tf.cast(value, self.rdtype))
        self._delta_down.assign(self._get_delta_down())

    def call(self,
             num_allocated_re,
             harq_feedback=None,
             sinr_eff=None,
             mcs_table_index=1,
             mcs_category=0):

        tf.debugging.assert_equal(
            tensor_values_are_in_set(harq_feedback, [-1, 0, 1]),
            True,
            message="'harq_feedback' must contain values in "
                    "[-1 (N/A), 0 (NACK), 1 (ACK)]")

        # Cast and reshape inputs
        shape = num_allocated_re.shape
        if harq_feedback is None:
            harq_feedback = tf.cast(tf.fill(shape, -1), tf.int32)
        else:
            harq_feedback = tf.cast(harq_feedback, tf.int32)
        if sinr_eff is None:
            sinr_eff = tf.zeros(shape, dtype=self.rdtype)
        else:
            sinr_eff = tf.cast(sinr_eff, self.rdtype)
        num_allocated_re = tf.cast(num_allocated_re, tf.int32)
        mcs_table_index = scalar_to_shaped_tensor(mcs_table_index,
                                                  tf.int32,
                                                  shape)
        mcs_category = scalar_to_shaped_tensor(mcs_category,
                                               tf.int32,
                                               shape)

        # ---------------------------- #
        # Update effective SINR offset #
        # ---------------------------- #
        self._offset.assign(
            tf.where(harq_feedback == 1,
                     self._offset - self._delta_down,
                     tf.where(harq_feedback == 0,
                              self._offset + self._delta_up,
                              self._offset)))

        # Project offset to [offset_min; offset_max]
        self._offset.assign(tf.maximum(self._offset, self._offset_min))
        self._offset.assign(tf.minimum(self._offset, self._offset_max))

        # ----------------------------------- #
        # Update last observed effective SINR #
        # ----------------------------------- #
        self.sinr_eff_db_last = tf.where(sinr_eff > 0,
                                         lin_to_db(sinr_eff,
                                                   precision=self.precision),
                                         self._sinr_eff_db_last)

        # -------------------------- #
        # Offset SINR and apply ILLA #
        # -------------------------- #
        sinr_eff_offset = db_to_lin(self._sinr_eff_db_last - self._offset,
                                    precision=self.precision)
        mcs_index = self._illa(
            sinr_eff=sinr_eff_offset,
            num_allocated_re=num_allocated_re,
            mcs_table_index=mcs_table_index,
            mcs_category=mcs_category,
            return_lowest_available_mcs=False)

        return mcs_index
