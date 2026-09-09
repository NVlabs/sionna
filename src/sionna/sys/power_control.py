# pylint: disable=line-too-long, too-many-arguments, too-many-positional-arguments
#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""
Transmit power control for Sionna SYS
"""

from typing import Optional, Tuple, Union

import torch

from sionna._validation import check_scalar_range, check_tensor_range
from sionna.phy import dtypes, config
from sionna.phy.config import Precision
from sionna.phy.utils import (
    scalar_to_shaped_tensor,
    lin_to_db,
    dbm_to_watt,
    bisection_method,
)

__all__ = ["open_loop_uplink_power_control", "downlink_fair_power_control"]


def open_loop_uplink_power_control(
    pathloss: torch.Tensor,
    num_allocated_subcarriers: torch.Tensor,
    alpha: Union[float, torch.Tensor] = 1.0,
    p0_dbm: Union[float, torch.Tensor] = -90.0,
    ut_max_power_dbm: Union[float, torch.Tensor] = 26.0,
    precision: Optional[Precision] = None,
) -> torch.Tensor:
    r"""
    Implements an open-loop uplink power control procedure inspired by 3GPP TS
    38.213, Section 7.1.1 :cite:p:`3GPP38213`.

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
    closed-loop control and transport format adjustments are here ignored.

    :param pathloss: Pathloss for each user relative to the serving base station,
        in linear scale. Shape: [..., num_ut].
    :param num_allocated_subcarriers: Number of allocated subcarriers for each user.
        Shape: [..., num_ut].
    :param alpha: Pathloss compensation factor. If a `float`, the same value is
        applied to all users. Shape: [..., num_ut] or scalar. Defaults to 1.0.
    :param p0_dbm: Target received power per PRB [dBm]. If a `float`, the same
        value is applied to all users. Shape: [..., num_ut] or scalar.
        Defaults to -90.0.
    :param ut_max_power_dbm: Maximum transmit power [dBm] for each user. If a
        `float`, the same value is applied to all users.
        Shape: [..., num_ut] or scalar. Defaults to 26.0.
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`, :attr:`~sionna.phy.config.Config.precision` is used.

    :output tx_power_per_ut: [..., num_ut], `torch.float`.
        Uplink transmit power [W] for each user, across subcarriers, streams
        and time steps.

    .. rubric:: Examples

    .. code-block:: python

        import torch
        import matplotlib.pyplot as plt
        from sionna.sys import open_loop_uplink_power_control
        from sionna.phy import config
        from sionna.phy.utils import db_to_lin, watt_to_dbm

        # N. users
        num_ut = 100
        # Max tx power per UT
        ut_max_power_dbm = 26  # [dBm]
        # Pathloss [dB]
        pathloss_db = config.rng.uniform(80, 120, size=(num_ut,))
        # N. allocated subcarriers per UT
        num_allocated_subcarriers = torch.full([num_ut], 40)
        # Parameters (pathloss compensation factor, reference rx power)
        alpha_p0 = [(1, -90), (.8, -75)]

        for alpha, p0 in alpha_p0:
            # Power allocation
            tx_power_per_ut = open_loop_uplink_power_control(
                db_to_lin(pathloss_db),
                num_allocated_subcarriers=num_allocated_subcarriers,
                alpha=alpha,
                p0_dbm=p0,
                ut_max_power_dbm=ut_max_power_dbm)
            # Plot CDF of tx power
            plt.ecdf(watt_to_dbm(tx_power_per_ut).cpu().numpy(),
                     label=fr'$\alpha$={alpha}, $P_0$={p0} dBm')
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
        rdtype = config.dtype
    else:
        rdtype = dtypes[precision]["torch"]["dtype"]

    # Ensure shapes match
    if pathloss.shape != num_allocated_subcarriers.shape:
        raise ValueError(
            "Inconsistent input shapes: 'pathloss' and "
            "'num_allocated_subcarriers' must have the same shape"
        )

    # [..., num_ut]
    pathloss_db = lin_to_db(pathloss, precision=precision)

    # Cast inputs - use .to() for existing tensors to avoid new_tensor under torch.compile
    dev = pathloss.device
    if isinstance(alpha, torch.Tensor):
        alpha = alpha.to(dtype=rdtype, device=dev)
    else:
        alpha = torch.as_tensor(alpha, dtype=rdtype, device=dev)
    if isinstance(p0_dbm, torch.Tensor):
        p0_dbm = p0_dbm.to(dtype=rdtype, device=dev)
    else:
        p0_dbm = torch.as_tensor(p0_dbm, dtype=rdtype, device=dev)
    if isinstance(ut_max_power_dbm, torch.Tensor):
        ut_max_power_dbm = ut_max_power_dbm.to(dtype=rdtype, device=dev)
    else:
        ut_max_power_dbm = torch.as_tensor(ut_max_power_dbm, dtype=rdtype, device=dev)

    # N. allocated PRBs - ensure same device as pathloss
    # [..., num_ut]
    num_allocated_subcarriers = num_allocated_subcarriers.to(device=pathloss.device)
    num_allocated_prb = torch.ceil(num_allocated_subcarriers.to(rdtype) / 12).to(torch.int32)

    # Uplink power per user [Watt] via 3GPP TS 38.213
    # [..., num_ut]
    tx_power_per_ut = torch.where(
        num_allocated_prb > 0,
        dbm_to_watt(
            p0_dbm + alpha * pathloss_db + lin_to_db(num_allocated_prb, precision=precision),
            precision=precision),
        torch.zeros((), dtype=rdtype, device=pathloss.device))

    # Limit power to max_power_dbm
    tx_power_per_ut = torch.minimum(
        tx_power_per_ut,
        dbm_to_watt(ut_max_power_dbm, precision=precision))

    return tx_power_per_ut


def downlink_fair_power_control(
    pathloss: torch.Tensor,
    interference_plus_noise: Union[float, torch.Tensor],
    num_allocated_re: Union[int, torch.Tensor],
    bs_max_power_dbm: Union[float, torch.Tensor] = 56.0,
    guaranteed_power_ratio: float = 0.5,
    fairness: float = 0.0,
    return_lagrangian: bool = False,
    precision: Optional[Precision] = None,
    **kwargs,
) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    r"""
    Allocates the downlink transmit power fairly across all users served by a
    single base station (BS).

    The maximum BS transmit power :math:`\overline{P}` is distributed across users
    by solving the following optimization problem:

    .. math::

        \begin{aligned}
            \mathbf{p}^* = \operatorname{argmax}_{\mathbf{p}} & \, \sum_{u=1}^{U} g^{(f)} \big( r_u \log( 1 + p_u q_u) \big) \\
            \mathrm{s.t.} & \, \sum_{u=1}^U r_u p_u = \overline{P} \\
            & r_u p_u \ge \rho \frac{\overline{P}}{U} , \quad \forall \, u=1,\dots,U
        \end{aligned}

    where :math:`q_u` represents the estimated channel quality, defined as the
    ratio between the channel gain (being the inverse of pathloss) and the
    interference plus noise ratio,
    :math:`r_u>0` denotes the number of allocated resources, and
    :math:`p_u` is the per-resource allocated power, for every user :math:`u`.

    The parameter :math:`\rho\in[0;1]` denotes the guaranteed power ratio; if
    set to 1, the power is distributed uniformly across all users.

    The fairness function :math:`g^{(f)}` is defined as in :cite:p:`MoWalrand`:

    .. math::

        \begin{aligned}
            g^{(f)}(x) = \left\{
            \begin{array}{l}
                \log(x), \quad f=1 \\
                \frac{x^{1-f}}{1-f}, \quad \forall\, f>0, \ f\ne 1.
            \end{array} \right.
        \end{aligned}

    When the fairness parameter :math:`f=0`, the sum of utilities :math:`\log( 1
    + p_u q_u)` is maximized, leading to a waterfilling-like solution
    (see, e.g., :cite:p:`Tse`).
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

    :param pathloss: Pathloss for each user in linear scale.
        Shape: [..., num_ut].
    :param interference_plus_noise: Interference plus noise [Watt] for each
        user. If `float`, the same value is assigned to all users.
        Shape: [..., num_ut] or scalar.
    :param num_allocated_re: Number of allocated resources to each user.
        If `int`, the same value is assigned to all users.
        Shape: [..., num_ut] or scalar.
    :param bs_max_power_dbm: Maximum transmit power for the base station [dBm].
        If `float`, the same value is assigned to all batches. Defaults to 56.0.
    :param guaranteed_power_ratio: The power allocated to a user is guaranteed to
        exceed a portion ``guaranteed_power_ratio`` of ``bs_max_power_dbm``
        divided by the number of scheduled users. Must be within [0;1].
        Defaults to 0.5.
    :param fairness: Fairness parameter. If 0, the sum of utilities is
        maximized; when 1, proportional fairness is achieved. As ``fairness``
        increases, the optimal allocation approaches a max-min one.
        Defaults to 0.0.
    :param return_lagrangian: If `True`, the inverse of the optimal Lagrangian
        multiplier ``mu_inv_star`` is returned. Defaults to `False`.
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`, :attr:`~sionna.phy.config.Config.precision` is used.
    :param kwargs: Additional inputs for
        :func:`~sionna.phy.utils.bisection_method` used to compute the optimal
        power allocation, such as ``eps_x``, ``eps_y``, ``max_n_iter``,
        ``step_expand``.

    :output tx_power: [..., num_ut], `torch.float`.
        Optimal total downlink power allocation :math:`r_u p_u^*` [Watt] for
        each user :math:`u`.
    :output utility: [..., num_ut], `torch.float`.
        Optimal utility for each user, computed as :math:`r_u \log( 1 + p^*_u
        q_u)` for user :math:`u`.
    :output mu_inv_star: [...], `torch.float`.
        Inverse of the optimal Lagrangian multiplier :math:`\mu` associated with the
        total power constraint. Only returned if ``return_lagrangian`` is `True`.

    .. rubric:: Examples

    .. code-block:: python

        import torch
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
        pathloss_db = config.rng.uniform(70, 110, size=(num_ut,))  # [dB]
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
        ind_sort = np.argsort(cq.cpu().numpy())[::-1]
        axs[0, 1].plot(10*np.log10(cq.cpu().numpy())[ind_sort], '.-')
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
                axs[2, ii].plot(utility.cpu().numpy()[ind_sort], '.-',
                            label=f'fairness = {fairness}')
                # Show transmit power
                axs[1, ii].plot(tx_power.cpu().numpy()[ind_sort], '.-',
                            label=f'fairness = {fairness}')

            axs[1, ii].plot([0, num_ut-1], [guaranteed_power]*2, '--k',
                            label='guaranteed power')
            axs[1, ii].set_ylabel(r'Power $r_u p^*_u$ [W]', fontsize=12)
            axs[1, ii].legend(fontsize=9)
            axs[2, ii].set_ylabel(
                r'Utility $r_u \log(1+p^*_u q_u)$', fontsize=12)
            axs[1, ii].set_title(
                f'Guaranteed power ratio = {guar_ratio}')

        fig.suptitle('Downlink fair power control', y=.98, fontsize=18)
        fig.tight_layout()
        fig.delaxes(axs[0, 0])
        fig.delaxes(axs[0, 2])
        plt.show()

    .. figure:: ../figures/fair_DL_tx_power.png
        :align: center
    """
    # Get precision dtype
    if precision is None:
        rdtype = config.dtype
    else:
        rdtype = dtypes[precision]["torch"]["dtype"]

    batch_shape = list(pathloss.shape[:-1])
    num_ut = pathloss.shape[-1]

    # Helper to create scalar tensors on the same device as pathloss
    def _scalar(val):
        return torch.as_tensor(val, dtype=rdtype, device=pathloss.device)

    # ------------------- #
    # Auxiliary functions #
    # ------------------- #
    def kkt_fun(p, mu_inv, fairness_val, cq, num_resources):
        r"""
        Computes the Karush-Kuhn-Tucker (KKT) function, that must be 0 when the
        solution is optimal
        """
        one = _scalar(1.0)
        fairness_t = _scalar(fairness_val)
        if fairness_val == 0:
            return cq * mu_inv.unsqueeze(-1) - (one + p * cq)
        else:
            log_pow = torch.pow(num_resources * torch.log(one + p * cq), fairness_t)
            return cq * mu_inv.unsqueeze(-1) - log_pow * (one + p * cq)

    def get_p_star_mu(mu_inv, fairness_val, cq, num_resources):
        """
        Computes the optimal power allocation given a certain (non-optimal, in
        general) inverse Lagrangian multiplier ``mu_inv``
        """
        # [..., num_ut]
        if fairness_val == 0:
            p_star_mu = torch.maximum(
                mu_inv.unsqueeze(-1) - torch.pow(cq, -1),
                p_left)
        else:
            p_star_mu, _ = bisection_method(
                kkt_fun,
                p_left,
                p_right,
                expand_to_right=False,
                expand_to_left=False,
                regula_falsi=False,
                mu_inv=mu_inv,
                fairness_val=fairness_val,
                cq=cq,
                num_resources=num_resources,
                precision=precision,
                **kwargs)
        return p_star_mu

    def constraint_slackness(mu_inv, fairness_val, cq, num_resources, max_power_bs_val):
        """
        Computes the amount of unused power for the given Lagrangian multiplier
        """
        p_star_mu = get_p_star_mu(mu_inv, fairness_val, cq, num_resources)
        # [...]
        slackness = max_power_bs_val - torch.sum(num_resources * p_star_mu, dim=-1)
        return slackness

    # ----------- #
    # Cast inputs #
    # ----------- #
    # Convert fairness to Python float to avoid torch.compile graph breaks from .item()
    if isinstance(fairness, torch.Tensor):
        fairness = float(fairness.item())
    else:
        fairness = float(fairness)
    if isinstance(guaranteed_power_ratio, torch.Tensor):
        check_tensor_range(
            guaranteed_power_ratio,
            name="guaranteed_power_ratio",
            minimum=0,
            maximum=1,
            message="guaranteed_power_ratio must be in [0, 1]",
        )
    else:
        check_scalar_range(
            guaranteed_power_ratio,
            name="guaranteed_power_ratio",
            minimum=0,
            maximum=1,
            message="guaranteed_power_ratio must be in [0, 1]",
        )
    pathloss = pathloss.to(dtype=rdtype)
    num_allocated_re = scalar_to_shaped_tensor(
        num_allocated_re, rdtype, batch_shape + [num_ut], device=pathloss.device)
    if not isinstance(interference_plus_noise, torch.Tensor):
        interference_plus_noise = _scalar(interference_plus_noise)
    else:
        interference_plus_noise = interference_plus_noise.to(dtype=rdtype, device=pathloss.device)
    guaranteed_power_ratio = _scalar(guaranteed_power_ratio)
    max_power_bs = dbm_to_watt(bs_max_power_dbm, precision=precision, device=pathloss.device)
    max_power_bs = scalar_to_shaped_tensor(max_power_bs, rdtype, batch_shape, device=pathloss.device)

    # Allocate zero power budget to batches with no resources
    max_power_bs = torch.where(
        torch.sum(num_allocated_re, dim=-1) > 0,
        max_power_bs,
        _scalar(0.0))

    # ------------ #
    # Check inputs #
    # ------------ #
    if not (fairness >= 0):
        raise ValueError("fairness parameter must be non-negative")

    # ----------------- #
    # Search boundaries #
    # ----------------- #
    # Minimum power per user (total across resources)
    # [...]
    num_scheduled_uts = torch.sum(
        (num_allocated_re > 0).to(rdtype), dim=-1)
    p_left = guaranteed_power_ratio * max_power_bs / num_scheduled_uts

    # Minimum power per user (for one resource)
    # [..., num_ut]
    p_left = p_left.unsqueeze(-1)
    p_left = p_left / num_allocated_re
    p_left = torch.where(
        num_allocated_re == 0,
        _scalar(0.0),
        p_left)

    # Maximum power per user (one resource)
    p_right = max_power_bs.unsqueeze(-1) / num_allocated_re
    p_right = torch.where(
        num_allocated_re == 0,
        _scalar(0.0),
        p_right)

    # (Soft) min/max value of Lagrangian multiplier inverse mu^-1
    # Represents the "water level" for fairness=0
    # [...]
    mu_inv_left = torch.full(batch_shape, 0.0, dtype=rdtype, device=pathloss.device)
    mu_inv_right = torch.full(batch_shape, 1000.0, dtype=rdtype, device=pathloss.device)

    # ------------------------------------------ #
    # (Inverse of) optimal Lagrangian multiplier #
    # ------------------------------------------ #
    # Channel quality
    cq = 1 / (pathloss * interference_plus_noise)
    # [...]
    mu_inv_star, _ = bisection_method(
        constraint_slackness,
        mu_inv_left,
        mu_inv_right,
        expand_to_right=True,
        expand_to_left=False,
        regula_falsi=False,
        fairness_val=fairness,
        cq=cq,
        num_resources=num_allocated_re,
        max_power_bs_val=max_power_bs,
        precision=precision,
        **kwargs)

    # ---------------------- #
    # Optimal transmit power #
    # ---------------------- #
    # [..., num_ut]
    tx_power_per_re = get_p_star_mu(
        mu_inv_star, fairness, cq, num_allocated_re
    )

    # Compute total power across resources
    tx_power = tx_power_per_re * num_allocated_re

    # ---------------- #
    # Achieved utility #
    # ---------------- #
    utility = num_allocated_re * torch.log(
        _scalar(1.0) + tx_power_per_re * cq
    )

    if return_lagrangian:
        return tx_power, utility, mu_inv_star
    else:
        return tx_power, utility
