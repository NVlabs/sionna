# pylint: disable=line-too-long, too-many-arguments, too-many-positional-arguments
#
# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0#
"""
Transmit power control for Sionna SYS
"""

import tensorflow as tf
from sionna.phy import dtypes, config
from sionna.phy.utils import scalar_to_shaped_tensor, lin_to_db, \
    dbm_to_watt, bisection_method


def open_loop_uplink_power_control(pathloss,
                                   num_allocated_subcarriers,
                                   alpha=1.,
                                   p0_dbm=-90.,
                                   ut_max_power_dbm=26.,
                                   precision=None):
    r"""
    Implements an open-loop uplink power control procedure inspired by 3GPP TS
    38.213, Section 7.1.1 [3GPP38213]_. 

    For each user, the uplink transmission power :math:`P^{\mathrm{UL}}`
    is computed as:

    .. math::
        P^{\mathrm{UL}} = \min \{ P_0 + \alpha PL + 10 \log_{10}(\mathrm{\#PRB}), \ P^{\mathrm{max}}\} \quad [\mathrm{dBm}]

    where :math:`P^{\mathrm{max}}` is the maximum power, :math:`P_0` [dBm] is
    the target received power per Physical Resource Block (PRB), :math:`PL` is
    the pathloss and :math:`\alpha\in [0,1]` is the pathloss compensation factor. 

    Note that if :math:`\alpha=1`, the pathloss is fully compensated and the
    power per PRB received by the base station equals :math:`P_0` [dBm], assuming
    :math:`P^{\mathrm{max}}` is not reached. Lower values of :math:`\alpha` can help
    reducing interference caused to neighboring cells. 

    With respect to 3GPP TS 38.213, additional factors such as
    closed-loop control and transport format adjustments are here ignored

    Input
    -----

    pathloss : [..., num_ut], `tf.float`
        Pathloss for each user relative to the serving base station, in linear scale

    num_allocated_subcarriers : [..., num_ut]
        Number of allocated subcarriers for each user

    alpha : [..., num_ut], `tf.float` | `float` (default: 1.0)
        Pathloss compensation factor. If a `float`, the same value is
        applied to all users.

    p0_dbm : [..., num_ut], `tf.float` | `float` (default: -90.0)
        Target received power per PRB. If a `float`, the same value is
        applied to all users.

    ut_max_power_dbm : [..., num_ut], `tf.float` | `float` (default: 26.0)
        Maximum transmit power [dBm] for each user. If a `float`, the same
        value is applied to all users.

    precision : `None` (default) | "single" | "double"
        Precision used for internal calculations and outputs.
        If set to `None`,
        :attr:`~sionna.phy.config.Config.precision` is used.

    Output
    ------

    tx_power_per_ut : [..., num_ut], `tf.float`
        Uplink transmit power [W] for each user, across subcarriers, streams
        and time steps

    Example
    -------
    .. code-block:: Python

        import matplotlib.pyplot as plt
        from sionna.sys import open_loop_uplink_power_control
        from sionna.phy import config
        from sionna.phy.utils import db_to_lin, watt_to_dbm

        # N. users
        num_ut = 100
        # Max tx power per UT
        ut_max_power_dbm = 26  # [dBm]
        # Pathloss [dB]
        pathloss_db = config.tf_rng.uniform([num_ut], minval=80, maxval=120)
        # N. allocated subcarriers per UT
        num_allocated_subcarriers = tf.fill([num_ut], 40)
        # Parameters (pathloss compensation factor, reference rx power)
        alpha_p0 = [(1, -90), (.8, -75)]

        for alpha, p0, in alpha_p0:
            # Power allocation
            tx_power_per_ut = open_loop_uplink_power_control(db_to_lin(pathloss_db),
                                        num_allocated_subcarriers=num_allocated_subcarriers,
                                        alpha=alpha,
                                        p0_dbm=p0,
                                        ut_max_power_dbm=ut_max_power_dbm)
            # Plot CDF of tx power
            plt.ecdf(watt_to_dbm(tx_power_per_ut), label=fr'$\alpha$={alpha}, $P_0$={p0} dBm')
        # Plot max UT power
        plt.plot([ut_max_power_dbm]*2, [0, 1], 'k--', label='max UT power')

        plt.legend()
        plt.grid()
        plt.xlabel('Tx power [dBm]')
        plt.ylabel('Cumulative density function')
        plt.title('Uplink tx power distribution')
        plt.show()

    .. figure:: ../figures/ulpc.png
        :align: center
        :width: 80%
    """
    if precision is None:
        rdtype = config.tf_rdtype
    else:
        rdtype = dtypes[precision]["tf"]["rdtype"]

    tf.debugging.assert_shapes([(pathloss, num_allocated_subcarriers.shape)],
                               message="Inconsistent input shapes")

    # [..., num_ut]
    pathloss_db = lin_to_db(pathloss, precision=precision)

    # Cast inputs
    # [..., num_ut] | float
    alpha = tf.cast(alpha, rdtype)
    # [..., num_ut] | float
    p0_dbm = tf.cast(p0_dbm, rdtype)
    # [..., num_ut] | float
    ut_max_power_dbm = tf.cast(ut_max_power_dbm, rdtype)

    # N. allocated PRBs
    # [..., num_ut]
    num_allocated_prb = tf.cast(tf.math.ceil(num_allocated_subcarriers / 12),
                                tf.int32)

    # Uplink power per user [Watt] via 3GPP TS 38.213
    # [..., num_ut]
    tx_power_per_ut = tf.where(
        num_allocated_prb > 0,
        dbm_to_watt(
            p0_dbm + alpha * pathloss_db + lin_to_db(num_allocated_prb,
                                                     precision=precision),
            precision=precision),
        tf.cast(0., rdtype))

    # Limit power to max_power_dbm
    tx_power_per_ut = tf.minimum(tx_power_per_ut,
                                 dbm_to_watt(ut_max_power_dbm,
                                             precision=precision))

    return tx_power_per_ut


def downlink_fair_power_control(pathloss,
                                interference_plus_noise,
                                num_allocated_re,
                                bs_max_power_dbm=56.,
                                guaranteed_power_ratio=0.5,
                                fairness=0.,
                                return_lagrangian=False,
                                precision=None,
                                **kwargs):
    r"""
    Allocates the downlink transmit power fairly across all users served by a
    single base station (BS)

    The maximum BS transmit power :math:`\overline{P}` is distributed across users
    by solving the following optimization problem:

    .. math:: 

        \begin{align}
            \mathbf{p}^* = \operatorname{argmax}_{\mathbf{p}} & \, \sum_{u=1}^{U} g^{(f)} \big( r_u \log( 1 + p_u q_u) \big) \\
            \mathrm{s.t.} & \, \sum_{u=1}^U r_u p_u = \overline{P} \\
            & r_u p_u \ge \rho \frac{\overline{P}}{U} , \quad \forall \, u=1,\dots,U
        \end{align}

    where :math:`q_u` represents the estimated channel quality, defined as the
    ratio between the channel gain (being the inverse of pathloss) and the
    interference plus noise ratio,  
    :math:`r_u>0` denotes the number of allocated resources, and
    :math:`p_u` is the per-resource allocated power, for every user :math:`u`.

    The parameter :math:`\rho\in[0;1]` denotes the guaranteed power ratio; if
    set to 1, the power is distributed uniformly across all users.

    The fairness function :math:`g^{(f)}` is defined as in [MoWalrand]_: 

    .. math::

        \begin{align}
            g^{(f)}(x) = \left\{ 
            \begin{array}{l}
                \log(x), \quad f=1 \\
                \frac{x^{1-f}}{1-f}, \quad \forall\, f>0, \ f\ne 1.
            \end{array} \right.
        \end{align}

    When the fairness parameter :math:`f=0`, the sum of utilities :math:`\log( 1
    + p_u q_u)` is maximized, leading to a waterfilling-like solution
    (see, e.g., [Tse]_).
    As :math:`f` increases, the allocation becomes increasingly
    egalitarian. The case :math:`f=1` maximizes proportional fairness; 
    as :math:`f\to \infty`, the solution approaches a max-min
    allocation. 

    For optimal power allocations :math:`p^*_u>\frac{\overline{P}}{U r_u}`, the
    Karush-Kuhn-Tucker (KKT) conditions can be expressed as:

    .. math::

        \big[ r_u \log (1+p^*_u q_u) \big]^f (1+p^*_u q_u) = q_u \mu^{-1}, \quad \forall \, u

    where :math:`\mu` is the Lagrangian multiplier associated with the
    constraint on the total transmit power.

    This function returns the optimal power allocation :math:`r_u p_u^*` and the
    corresponding utility :math:`r_u \log( 1 + p^*_u q_u)`, for each user
    :math:`u=1,\dots,U`. 
    If ``return_lagrangian`` is `True`, :math:`\mu^{-1}` is returned, too.

    Input
    -----

    pathloss : [..., num_ut], `tf.float`
        Pathloss for each user in linear scale

    interference_plus_noise : [..., num_ut], `tf.float` | `float`
        Interference plus noise [Watt] for each user. If `float`, the same
        value is assigned to all users.

    num_allocated_re : [..., num_ut], `tf.int32` | `int`
        Number of allocated resources to each user. If `int`, the same
        value is assigned to all users.

    bs_max_power_dbm : [...], `tf.float` | `float` (default: 56.)
        Maximum transmit power for the base station [dBm]. If `float`, the
        same value is assigned to all batches.

    guaranteed_power_ratio : `float` (default: 0.2)
        The power allocated to a user is guaranteed to exceed a portion
        ``guaranteed_power_ratio`` of ``bs_max_power_dbm`` divided by the number
        of scheduled users. Must be within [0;1].

    fairness : `float` (default: 0.)
        Fairness parameter. If 0, the sum of utilities is
        maximized; when 1, proportional fairness is achieved. As ``fairness``
        increases, the optimal allocation approaches a max-min one.

    return_lagrangian : `bool` (default: `False`)
        If `True`, the inverse of the optimal Lagrangian multiplier
        ``mu_inv_star`` is returned 

    precision : `None` (default) | "single" | "double"
        Precision used for internal calculations and outputs.
        If set to `None`,
        :attr:`~sionna.phy.config.Config.precision` is used.

    kwargs : `dict`
        Additional inputs for
        :func:`~sionna.phy.utils.bisection_method` used to compute the optimal
        power allocation, such as ``eps_x``, ``eps_y``, ``max_n_iter``,
        ``step_expand`` 

    Output
    ------

    tx_power : [..., num_ut], `tf.float`
        Optimal downlink power allocation :math:`p_u^*` [Watt] for each user
        :math:`u` 

    utility : [..., num_ut], `tf.float`
        Optimal utility for each user, computed as :math:`r_u \log( 1 + p^*_u
        q_u)` for user :math:`u`

    mu_inv_star : [...], `tf.float`
        Inverse of the optimal Lagrangian multiplier :math:`\mu` associated with the
        total power constraint. Only returned if ``return_lagrangian`` is `True`.

    Example
    -------

    .. code-block:: Python

        import tensorflow as tf
        import numpy as np
        import matplotlib.pyplot as plt
        from sionna.phy import config
        from sionna.phy.utils import db_to_lin, dbm_to_watt
        from sionna.sys import downlink_fair_power_control
        config.seed = 45

        # Evaluate the impact of 'fairness' and 'guaranteed_power_ratio'
        # parameters on the DL power allocation and utility

        # Guaranteed power ratios
        guaranteed_power_ratio_vec = [0, .35, .7]

        # Fairness parameters
        fairness_vec = [0, 1, 2, 5]

        # N. users
        num_ut = 30

        # BS tx power
        bs_max_power_dbm = 56
        max_power_bs = dbm_to_watt(bs_max_power_dbm)

        # Interference plus noise
        interference_plus_noise = 5e-10  # [W]

        # Generate random pathloss
        pathloss_db = config.tf_rng.uniform(
            [num_ut], minval=70, maxval=110)  # [dB]
        pathloss = db_to_lin(pathloss_db)

        # Channel quality
        cq = 1 / (pathloss * interference_plus_noise)

        fig, axs = plt.subplots(3, len(guaranteed_power_ratio_vec),
                                figsize=(3.5*len(guaranteed_power_ratio_vec), 8),
                                sharey='row')
        fig.subplots_adjust(top=0.8)
        for ax in axs.flatten():
            ax.yaxis.set_tick_params(labelbottom=True)
            ax.grid()
            ax.set_xlabel(r'User terminal $u$', fontsize=12)

        # Show channel quality in decreasing order
        ind_sort = np.argsort(cq)[::-1]
        axs[0, 1].plot(10*np.log10(cq)[ind_sort], '.-')
        axs[0, 1].set_ylabel(r'$q_u$ [dB]', fontsize=12)
        axs[0, 1].set_title('Channel quality')

        for ii, guar_ratio in enumerate(guaranteed_power_ratio_vec):
            # Guaranteed power for each user
            guaranteed_power = guar_ratio * max_power_bs / num_ut

            for fairness in fairness_vec:
                # DL fair power allocation
                tx_power, utility = downlink_fair_power_control(
                    pathloss,
                    interference_plus_noise=interference_plus_noise,
                    num_allocated_re=1,
                    bs_max_power_dbm=bs_max_power_dbm,
                    guaranteed_power_ratio=guar_ratio,
                    fairness=fairness)

                # Show utility
                axs[2, ii].plot(utility.numpy()[ind_sort], '.-',
                            label=f'fairness = {fairness}')
                # Show transmit power
                axs[1, ii].plot(tx_power.numpy()[ind_sort], '.-',
                            label=f'fairness = {fairness}')

            axs[1, ii].plot([0, num_ut-1], [guaranteed_power]*2, '--k', label='guaranteed power')
            axs[1, ii].set_ylabel(r'Power $r_u p^*_u$ [W]', fontsize=12)
            axs[1, ii].legend(fontsize=9)
            axs[2, ii].set_ylabel(r'Utility $r_u \log(1+p^*_u q_u)$', fontsize=12)
            axs[1, ii].set_title(f'Guaranteed power ratio = {guar_ratio}')

        fig.suptitle('Downlink fair power control', y=.98, fontsize=18)
        fig.tight_layout()
        fig.delaxes(axs[0, 0])
        fig.delaxes(axs[0, 2])
        plt.show()

    .. figure:: ../figures/fair_DL_tx_power.png
        :align: center
    """
    # ------------------- #
    # Auxiliary functions #
    # ------------------- #
    def kkt_fun(p,
                mu_inv,
                fairness,
                cq,
                num_resources):
        r"""
        Computes the Karush-Kuhn-Tucker (KKT) function, that must be 0 when the
        solution is optimal
        """
        one = tf.cast(1, rdtype)
        fairness = tf.cast(fairness, rdtype)
        if fairness == 0:
            return cq * mu_inv[..., tf.newaxis] - (one + p * cq)
        else:
            log_pow = tf.pow(num_resources *
                             tf.math.log(one + p * cq), fairness)
            return cq * mu_inv[..., tf.newaxis] - log_pow * (one + p * cq)

    def get_p_star_mu(mu_inv,
                      fairness,
                      cq,
                      num_resources):
        """
        Computes the optimal power allocation given a certain (non-optimal, in
        general) inverse Lagrangian multiplier ``mu_inv``
        """
        # [..., num_ut]
        if fairness == 0:
            p_star_mu = tf.maximum(
                mu_inv[..., tf.newaxis] - tf.pow(cq, -1),
                p_left)
        else:
            p_star_mu, _ = bisection_method(  # pylint: disable=unbalanced-tuple-unpacking
                kkt_fun,
                p_left,
                p_right,
                expand_to_right=False,
                expand_to_left=False,
                regula_falsi=False,
                mu_inv=mu_inv,
                fairness=fairness,
                cq=cq,
                num_resources=num_resources,
                precision=precision,
                **kwargs)
        return p_star_mu

    def constraint_slackness(mu_inv,
                             fairness,
                             cq,
                             num_resources,
                             max_power_bs):
        """
        Computes the amount of unused power for the given Lagrangian multiplier
        """
        p_star_mu = get_p_star_mu(mu_inv,
                                  fairness,
                                  cq,
                                  num_resources)
        # [...]
        slackness = max_power_bs - \
            tf.reduce_sum(num_resources * p_star_mu, axis=-1)
        return slackness

    # ----------- #
    # Cast inputs #
    # ----------- #
    batch_size, num_ut = pathloss.shape[:-1], pathloss.shape[-1]
    if precision is None:
        rdtype = config.tf_rdtype
    else:
        rdtype = dtypes[precision]["tf"]["rdtype"]
    fairness = tf.cast(fairness, rdtype)
    pathloss = tf.cast(pathloss, rdtype)
    num_allocated_re = scalar_to_shaped_tensor(
        num_allocated_re, rdtype, batch_size + [num_ut])
    interference_plus_noise = tf.cast(interference_plus_noise, rdtype)
    guaranteed_power_ratio = tf.cast(guaranteed_power_ratio, rdtype)
    max_power_bs = dbm_to_watt(bs_max_power_dbm, precision=precision)
    max_power_bs = scalar_to_shaped_tensor(
        max_power_bs, rdtype, batch_size)

    # Allocate zero power budget to batches with no resources
    max_power_bs = tf.where(tf.reduce_sum(num_allocated_re, axis=-1) > 0,
                            max_power_bs,
                            tf.cast(0, rdtype))

    # ------------ #
    # Check inputs #
    # ------------ #
    tf.debugging.assert_non_negative(
        fairness,
        message="fairness parameter must be non-negative")
    tf.debugging.assert_non_negative(
        guaranteed_power_ratio,
        message="guaranteed_power_ratio must be in [0;1]")
    tf.debugging.assert_less_equal(
        guaranteed_power_ratio, tf.cast(1., rdtype),
        message="guaranteed_power_ratio must be in [0;1]")

    # ----------------- #
    # Search boundaries #
    # ----------------- #
    # Minimum power per user (total across resources)
    # [...]
    num_scheduled_uts = tf.reduce_sum(
        tf.cast(num_allocated_re > 0, rdtype), axis=-1)
    p_left = guaranteed_power_ratio * max_power_bs / num_scheduled_uts

    # Minimum power per user (for one resource)
    # [..., num_ut]
    p_left = tf.expand_dims(p_left, axis=-1)
    p_left = p_left / num_allocated_re
    p_left = tf.where(num_allocated_re == 0,
                      tf.cast(0., rdtype),
                      p_left)

    # Maximum power per user (one resource)
    p_right = max_power_bs[..., tf.newaxis] / num_allocated_re
    p_right = tf.where(num_allocated_re == 0,
                       tf.cast(0., rdtype),
                       p_right)

    # (Soft) min/max value of Lagrangian multiplier inverse mu^-1
    # Represents the "water level" for fairness=0
    # [...]
    mu_inv_left = tf.fill(batch_size, tf.cast(0., rdtype))
    mu_inv_right = tf.fill(batch_size, tf.cast(1000., rdtype))

    # ------------------------------------------ #
    # (Inverse of) optimal Lagrangian multiplier #
    # ------------------------------------------ #
    # Channel quality
    cq = 1 / (pathloss * interference_plus_noise)
    # [...]
    # pylint: disable=unbalanced-tuple-unpacking
    mu_inv_star, _ = bisection_method(
        constraint_slackness,
        mu_inv_left,
        mu_inv_right,
        expand_to_right=True,
        expand_to_left=False,
        regula_falsi=False,
        fairness=fairness,
        cq=cq,
        num_resources=num_allocated_re,
        max_power_bs=max_power_bs,
        precision=precision,
        **kwargs)

    # ---------------------- #
    # Optimal transmit power #
    # ---------------------- #
    # [..., num_ut]
    tx_power = get_p_star_mu(mu_inv_star,
                             fairness,
                             cq,
                             num_allocated_re)

    # Compute total power across resources
    tx_power = tx_power * num_allocated_re

    # ---------------- #
    # Achieved utility #
    # ---------------- #
    utility = num_allocated_re * \
        tf.math.log(tf.cast(1, rdtype) + tx_power * cq)

    if return_lagrangian:
        return tx_power, utility, mu_inv_star
    else:
        return tx_power, utility
