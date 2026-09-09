#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Numerical methods for Sionna PHY and SYS"""

from typing import Callable, Optional, Tuple, Union

import torch

from sionna._validation import check_one_of, check_tensor_all
from sionna.phy import config, dtypes

__all__ = ["expand_bound", "bisection_method"]


def expand_bound(
    f: Callable,
    bound: torch.Tensor,
    side: str,
    step_expand: float = 2.0,
    max_n_iter: int = 100,
    precision: Optional[str] = None,
    device: Optional[str] = None,
    **kwargs,
) -> torch.Tensor:
    r"""
    Expands the left (right, respectively) search interval end point until the
    function ``f`` becomes positive (negative, resp.)

    :param f: Generic function handle that takes batched inputs and returns batched outputs.
        Applies a different decreasing univariate function to each of its inputs.
        Must accept input batches of the same shape as ``left`` and ``right``.
    :param bound: [...], `torch.float`.
        Left (if ``side`` is 'left') or right (if ``side`` is 'right') end point
        of the initial search interval, for each batch.
    :param side: 'left' | 'right'.
        See ``bound``.
    :param step_expand: Geometric progression factor at which the bound is expanded. Must be
        higher than 1.
    :param max_n_iter: Maximum number of iterations.
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`,
        :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for computation. If `None`, the device of ``bound``
        is used when it is a tensor, and
        :attr:`~sionna.phy.config.Config.device` otherwise.
    :param kwargs: Additional arguments for function ``f``.

    :output bound: [...], `torch.float`.
        Final value of expanded bound.

    .. rubric:: Examples

    .. code-block:: python

        import torch
        from sionna.phy.utils.numerics import expand_bound

        # Define a decreasing univariate function of x
        def f(x):
            return -x + 1.0

        # Expand right bound until f becomes negative
        bound = torch.tensor(0.5)
        result = expand_bound(f, bound, side='right', step_expand=2.0)
        print(result)
    """
    if precision is None:
        dtype = config.dtype
    else:
        dtype = dtypes[precision]["torch"]["dtype"]

    # Explicit device wins, then the device of the input tensor, then config
    if device is None:
        if isinstance(bound, torch.Tensor):
            device = bound.device
        else:
            device = config.device

    if not isinstance(bound, torch.Tensor):
        bound = torch.tensor(bound, dtype=dtype, device=device)
    else:
        bound = bound.to(dtype=dtype, device=device)

    if not isinstance(step_expand, torch.Tensor):
        step_expand = torch.tensor(step_expand, dtype=dtype, device=device)
    else:
        step_expand = step_expand.to(dtype=dtype, device=device)

    # Validate inputs
    check_one_of(
        side,
        ("left", "right"),
        name="side",
        message="side must be 'left' or 'right'",
    )
    check_tensor_all(
        step_expand > 1,
        name="step_expand",
        message="`step_expand` must be > 1",
    )

    # Initialize left and right bounds for search intervals
    if side == "left":
        for i in range(max_n_iter):
            f_val = f(bound, **kwargs)
            # Check if any bound needs expansion (f < 0)
            condition = f_val < 0
            if not torch.any(condition):
                break

            # Update bounds where f < 0
            # bound = bound - step_expand^i
            step = torch.pow(torch.abs(step_expand), i)
            bound = torch.where(condition, bound - step, bound)

        check_tensor_all(
            f(bound, **kwargs) >= 0,
            name="bound",
            message=(
                "Root cannot be bracketed on the left; increase "
                "`step_expand` or `max_n_iter`"
            ),
            error_type=RuntimeError,
        )
    else:
        for i in range(max_n_iter):
            f_val = f(bound, **kwargs)
            # Check if any bound needs expansion (f > 0)
            condition = f_val > 0
            if not torch.any(condition):
                break

            # Update bounds where f > 0
            # bound = bound + step_expand^i
            step = torch.pow(torch.abs(step_expand), i)
            bound = torch.where(condition, bound + step, bound)

        check_tensor_all(
            f(bound, **kwargs) <= 0,
            name="bound",
            message=(
                "Root cannot be bracketed on the right; increase "
                "`step_expand` or `max_n_iter`"
            ),
            error_type=RuntimeError,
        )

    return bound


def bisection_method(
    f: Callable,
    left: torch.Tensor,
    right: torch.Tensor,
    regula_falsi: bool = False,
    expand_to_left: bool = True,
    expand_to_right: bool = True,
    step_expand: float = 2.0,
    eps_x: float = 1e-5,
    eps_y: float = 1e-4,
    max_n_iter: int = 100,
    return_brackets: bool = False,
    precision: Optional[str] = None,
    device: Optional[str] = None,
    **kwargs,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    r"""
    Implements the classic bisection method for estimating the root of batches of decreasing
    univariate functions

    :param f: Generic function handle that takes batched inputs and returns batched outputs.
        Applies a different decreasing univariate function to each of its inputs.
        Must accept input batches of the same shape as ``left`` and ``right``.
    :param left: [...], `torch.float`.
        Left end point of the initial search interval, for each batch.
        The root is guessed to be contained within [``left``, ``right``].
    :param right: [...], `torch.float`.
        Right end point of the initial search interval, for each batch.
    :param regula_falsi: If `True`, then the `regula falsi` method is employed to determine the
        next root guess. This guess is computed as the x-intercept of the line
        passing through the two points formed by the function evaluated at the
        current search interval endpoints.
        Else, the next root guess is computed as the middle point of the current
        search interval. Defaults to `False`.
    :param expand_to_left: If `True` and ``f(left)`` is negative, then ``left`` is decreased by a
        geometric progression of ``step_expand`` until ``f`` becomes positive,
        for each batch.
        If `False`, then ``left`` is not decreased. Defaults to `True`.
    :param expand_to_right: If `True` and ``f(right)`` is positive, then ``right`` is increased by a
        geometric progression of ``step_expand`` until ``f`` becomes negative,
        for each batch.
        If `False`, then ``right`` is not increased. Defaults to `True`.
    :param step_expand: See ``expand_to_left`` and ``expand_to_right``. Defaults to 2.0.
    :param eps_x: Convergence criterion. Search terminates after ``max_n_iter`` iterations
        or if, for each batch, either the search interval length is smaller than
        ``eps_x`` or the function absolute value is smaller than ``eps_y``. Defaults to 1e-5.
    :param eps_y: Convergence criterion. See ``eps_x``. Defaults to 1e-4.
    :param max_n_iter: Maximum number of iterations. Defaults to 100.
    :param return_brackets: If `True`, the final values of search interval ``left`` and ``right``
        end point are returned. Defaults to `False`.
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`,
        :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for computation. If `None`, the device of ``left``
        (or ``right``) is used when either is a tensor, and
        :attr:`~sionna.phy.config.Config.device` otherwise.
    :param kwargs: Additional arguments for function ``f``.

    :output x_opt: [...], `torch.float`.
        Estimated roots of the input batch of functions ``f``.

    :output f_opt: [...], `torch.float`.
        Value of function ``f`` evaluated at ``x_opt``.

    :output left: [...], `torch.float`.
        Final value of left end points of the search intervals.
        Only returned if ``return_brackets`` is `True`.

    :output right: [...], `torch.float`.
        Final value of right end points of the search intervals.
        Only returned if ``return_brackets`` is `True`.

    .. rubric:: Examples

    .. code-block:: python

        import torch
        from sionna.phy.utils import bisection_method

        # Define a decreasing univariate function of x
        def f(x, a):
            return - torch.pow(x - a, 3)

        # Initial search interval
        left, right = 0., 2.
        # Input parameter for function a
        a = 3

        # Perform bisection method
        x_opt, _ = bisection_method(f, left, right, eps_x=1e-4, eps_y=0, a=a)
        print(x_opt.numpy())
        # 2.9999084
    """
    if precision is None:
        dtype = config.dtype
    else:
        dtype = dtypes[precision]["torch"]["dtype"]

    # Explicit device wins, then the device of an input tensor, then config
    if device is None:
        if isinstance(left, torch.Tensor):
            device = left.device
        elif isinstance(right, torch.Tensor):
            device = right.device
        else:
            device = config.device

    if not isinstance(left, torch.Tensor):
        left = torch.tensor(left, dtype=dtype, device=device)
    else:
        left = left.to(dtype=dtype, device=device)

    if not isinstance(right, torch.Tensor):
        right = torch.tensor(right, dtype=dtype, device=device)
    else:
        right = right.to(dtype=dtype, device=device)

    eps_x = torch.tensor(eps_x, dtype=dtype, device=device)

    check_tensor_all(
        left <= right,
        name="left",
        message="`left` must be <= `right`",
    )

    # -------------------------- #
    # Expand (or not) end points #
    # -------------------------- #
    if expand_to_left:
        # Decrease left bracket until f gets positive
        left = expand_bound(
            f,
            left,
            "left",
            step_expand=step_expand,
            max_n_iter=max_n_iter,
            precision=precision,
            device=device,
            **kwargs,
        )
    else:
        left = torch.where(f(right, **kwargs) > 0, right, left)

    if expand_to_right:
        # Increase left bracket until f gets negative
        right = expand_bound(
            f,
            right,
            "right",
            step_expand=step_expand,
            max_n_iter=max_n_iter,
            precision=precision,
            device=device,
            **kwargs,
        )
    else:
        right = torch.where(f(left, **kwargs) < 0, left, right)

    # -------------- #
    # Initialization #
    # -------------- #
    def get_x_next(left, right):
        """Computes the next guess of the function root"""
        if regula_falsi:
            # Regula falsi:
            # Compute x-intercept of function evaluated at current end points
            f_left = f(left, **kwargs)
            f_right = f(right, **kwargs)
            x_next = torch.where(
                right > left,
                (left * f_right - right * f_left) / (f_right - f_left),
                left,
            )
        else:
            # Compute middle point of search interval
            x_next = (left + right) / 2
        return x_next

    x_next = get_x_next(left, right)
    f_next = f(x_next, **kwargs)

    # -------------- #
    # Bisection loop #
    # -------------- #
    for _ in range(max_n_iter):
        # Convergence criterion
        # Condition 1: Interval length is small enough
        stop_cond1 = torch.abs(right - left) < eps_x

        # Condition 2: Function value is small enough
        stop_cond2 = torch.abs(f_next) < eps_y

        if torch.all(stop_cond1 | stop_cond2):
            break

        # Update left and right bounds

        # Next guess
        x_next = get_x_next(left, right)
        f_next = f(x_next, **kwargs)

        # If f_next >= 0, then shrink interval to the right
        left = torch.where(f_next >= 0, x_next, left)

        # If f_next <= 0, then shrink interval to the left
        right = torch.where(f_next <= 0, x_next, right)

    if return_brackets:
        return x_next, f_next, left, right
    else:
        return x_next, f_next
