# pylint: disable=line-too-long
#
# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0#
"""Numerical methods for Sionna PHY and SYS"""

import tensorflow as tf
from sionna.phy import config, dtypes


def expand_bound(f,
                 bound,
                 side,
                 step_expand=2.,
                 max_n_iter=100,
                 precision=None,
                 **kwargs):
    r"""
    Expands the left (right, respectively) search interval end point until the
    function ``f`` becomes positive (negative, resp.)

    Input
    -----

    f : `callable`
        Generic function handle that takes batched inputs and returns batched outputs.
        Applies a different decreasing univariate function to each of its inputs.
        Must accept input batches of the same shape as ``left`` and ``right``.

    bound : [...], `tf.float`
        Left (if ``side`` is 'left') or right (if ``side`` is 'right') end point
        of the initial search interval, for each batch

    side : 'left' | 'right'
        See ``bound``

    step_expand : `float`
        Geometric progression factor at which the bound is expanded. Must be
        higher than 1.

    max_n_iter : `int`
        Maximum number of iterations

    precision : `None` (default) | "single" | "double"
        Precision used for internal calculations and outputs.
        If set to `None`,
        :attr:`~sionna.phy.config.Config.precision` is used.

    kwargs : `dict`
        Additional arguments for function ``f``

    Output
    ------

    bound : [...], `tf.float`
        Final value of expanded bound
    """
    if precision is None:
        rdtype = config.tf_rdtype
    else:
        rdtype = dtypes[precision]["tf"]["rdtype"]

    # Cast inputs
    bound = tf.cast(bound, rdtype)
    step_expand = tf.cast(step_expand, rdtype)

    # Validate inputs
    tf.debugging.assert_equal(
        side in ['left', 'right'], True,
        message="side must be 'left' or 'right'")

    tf.debugging.assert_greater(step_expand, tf.cast(1., rdtype),
                                message="step_expand must be > 1")

    # Initialize left and right bounds for search intervals
    if side == 'left':
        bound = tf.while_loop(
            lambda bound, _: tf.reduce_any(f(bound, **kwargs) < 0),
            lambda bound, ii: [tf.where(f(bound, **kwargs) < 0,
                                        bound -
                                        tf.pow(tf.abs(step_expand),
                                               tf.cast(ii, rdtype)),
                                        bound),
                               ii + 1],
            [bound, 0],
            maximum_iterations=max_n_iter)[0]

        tf.debugging.assert_equal(
            tf.reduce_all(f(bound, **kwargs) >= 0),
            True,
            message="Root cannot be found. Please either increase "
            "'step_expand' or 'max_n_iter'")
    else:
        bound = tf.while_loop(
            lambda bound, _: tf.reduce_any(f(bound, **kwargs) > 0),
            lambda bound, ii: [tf.where(f(bound, **kwargs) > 0,
                                        bound +
                                        tf.pow(tf.abs(step_expand),
                                               tf.cast(ii, rdtype)),
                                        bound),
                               ii + 1],
            [bound, 0],
            maximum_iterations=max_n_iter)[0]

        tf.debugging.assert_equal(
            tf.reduce_all(f(bound, **kwargs) <= 0),
            True,
            message="Root cannot be found. Please either increase "
            "'step_expand' or 'max_n_iter'")
    return bound


def bisection_method(f,
                     left,
                     right,
                     regula_falsi=False,
                     expand_to_left=True,
                     expand_to_right=True,
                     step_expand=2.,
                     eps_x=1e-5,
                     eps_y=1e-4,
                     max_n_iter=100,
                     return_brackets=False,
                     precision=None,
                     **kwargs):
    r"""
    Implements the classic bisection method for estimating the root of batches of decreasing
    univariate functions

    Input
    -----

    f : `callable`
        Generic function handle that takes batched inputs and returns batched outputs.
        Applies a different decreasing univariate function to each of its inputs.
        Must accept input batches of the same shape as ``left`` and ``right``.

    left : [...], `tf.float`
        Left end point of the initial search interval, for each batch.
        The root is guessed to be contained within [``left``, ``right``].

    right : [...], `tf.float`
        Right end point of the initial search interval, for each batch

    regula_falsi : `bool` (default: `False`)
        If `True`, then the `regula falsi` method is employed to determine the
        next root guess. This guess is computed as the x-intercept of the line
        passing through the two points formed by the function evaluated at the
        current search interval endpoints.  
        Else, the next root guess is computed as the middle point of the current
        search interval.

    expand_to_left : `bool` (default: `True`)
        If `True` and ``f(left)`` is negative, then ``left`` is decreased by a
        geometric progression of ``step_expand`` until ``f`` becomes positive,
        for each batch. 
        If `False`, then ``left`` is not decreased.

    expand_to_right : `bool` (default: `True`)
        If `True` and ``f(left)`` is positive, then ``right`` is increased by a
        geometric progression of ``step_expand`` until ``f`` becomes negative,
        for each batch.
        If `False`, then ``right`` is not increased.

    step_expand : `float` (default: 2.)
        See ``expand_to_left`` and ``expand_to_right``

    eps_x : `float` (default: 1e-4)
        Convergence criterion. Search terminates after ``max_n_iter`` iterations
        or if, for each batch, either the search interval length is smaller than
        ``eps_x`` or the function absolute value is smaller than ``eps_y``.

    eps_y : `float` (default: 1e-4)
        Convergence criterion. See ``eps_x``.

    max_n_iter : `int` (default: 1000)
        Maximum number of iterations

    return_brackets : `bool` (default: `False`)
        If `True`, the final values of search interval ``left`` and ``right``
        end point are returned 

    precision : `None` (default) | "single" | "double"
        Precision used for internal calculations and outputs.
        If set to `None`,
        :attr:`~sionna.phy.config.Config.precision` is used.

    kwargs : `dict`
        Additional arguments for function ``f``

    Output
    ------

    x_opt : [...], `tf.float`
        Estimated roots of the input batch of functions ``f``

    f_opt : [...], `tf.float`
        Value of function ``f`` evaluated at ``x_opt``

    left : [...], `tf.float`
        Final value of left end points of the search intervals. 
        Only returned if ``return_brackets`` is `True`. 

    right : [...], `tf.float`
        Final value of right end points of the search intervals. 
        Only returned if ``return_brackets`` is `True`.


    Example
    -------

    .. code-block:: Python

        import tensorflow as tf
        from sionna.phy.utils import bisection_method

        # Define a decreasing univariate function of x
        def f(x, a):
            return - tf.math.pow(x - a, 3)

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
        rdtype = config.tf_rdtype
    else:
        rdtype = dtypes[precision]["tf"]["rdtype"]

    # Validate inputs
    tf.debugging.assert_less_equal(
        left,
        right,
        message="bound_left must be <= bound_right")

    # Cast inputs
    left = tf.cast(left, rdtype)
    right = tf.cast(right, rdtype)
    eps_x = tf.cast(eps_x, rdtype)

    # -------------------------- #
    # Expand (or not) end points #
    # -------------------------- #
    if expand_to_left:
        # Decrease left bracket until f gets positive
        left = expand_bound(f,
                            left,
                            'left',
                            step_expand=step_expand,
                            max_n_iter=max_n_iter,
                            precision=precision,
                            **kwargs)
    else:
        left = tf.where(f(right, **kwargs) > 0,
                        right,
                        left)

    if expand_to_right:
        # Increase left bracket until f gets negative
        right = expand_bound(f,
                             right,
                             'right',
                             step_expand=step_expand,
                             max_n_iter=max_n_iter,
                             precision=precision,
                             **kwargs)
    else:
        right = tf.where(f(left, **kwargs) < 0,
                         left,
                         right)

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
            x_next = tf.where(
                right > left,
                (left * f_right - right * f_left) / (f_right - f_left),
                left)
        else:
            # Compute middle point of search interval
            x_next = (left + right) / tf.cast(2, rdtype)
        return x_next

    x_next = get_x_next(left, right)
    f_next = f(x_next, **kwargs)

    # -------------- #
    # Bisection loop #
    # -------------- #
    # pylint: disable=unused-argument
    def cond_bisection(left, right, x_next, f_next):
        """Convergence criterion (If True, search continues)"""
        # Condition 1: Interval length is small enough
        stop_cond1 = tf.abs(right - left) < eps_x

        # Condition 2: Function value is small enough
        stop_cond2 = tf.abs(f_next) < eps_y

        return not tf.reduce_all(stop_cond1 | stop_cond2)

    def body_bisection(left, right, x_next, f_next):
        """Bisection body: Update left and right bounds"""
        # Next guess
        x_next = get_x_next(left, right)
        f_next = f(x_next, **kwargs)

        # If f_next >= 0, then shrink interval to the right
        left = tf.where(f_next >= 0,
                        x_next,
                        left)

        # If f_next <= 0, then shrink interval to the left
        right = tf.where(f_next <= 0,
                         x_next,
                         right)
        return [left, right, x_next, f_next]

    # Perform bisection method
    left, right, x_opt, f_opt = tf.while_loop(
        cond_bisection,
        body_bisection,
        [left, right, x_next, f_next],
        maximum_iterations=max_n_iter)

    if return_brackets:
        return x_opt, f_opt, left, right
    else:
        return x_opt, f_opt
