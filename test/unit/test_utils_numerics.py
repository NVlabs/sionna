#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import torch
from sionna.phy.utils import bisection_method, expand_bound
from sionna.phy import config


def bisection_method_compile(
    f,
    left,
    right,
    regula_falsi=False,
    expand_to_left=True,
    expand_to_right=True,
    step_expand=2.0,
    eps_x=1e-5,
    eps_y=1e-5,
    max_n_iter=100,
    return_brackets=False,
    precision=None,
    **kwargs,
):
    return bisection_method(
        f,
        left,
        right,
        regula_falsi=regula_falsi,
        expand_to_left=expand_to_left,
        expand_to_right=expand_to_right,
        step_expand=step_expand,
        eps_x=eps_x,
        eps_y=eps_y,
        max_n_iter=max_n_iter,
        return_brackets=return_brackets,
        precision=precision,
        **kwargs,
    )


# Compile the wrapper function
bisection_method_compiled = torch.compile(bisection_method_compile)


def test_tensor_contracts_compile_as_fullgraph(device):
    """Tensor contracts must not add graph breaks to the numerical helpers."""

    def f(x):
        return 1.0 - x

    @torch.compile(fullgraph=True)
    def compiled(bound, left, right, step_expand):
        expanded = expand_bound(
            f,
            bound,
            "left",
            step_expand=step_expand,
            max_n_iter=0,
        )
        root, _ = bisection_method(
            f,
            left,
            right,
            expand_to_left=False,
            expand_to_right=False,
            max_n_iter=0,
        )
        return expanded, root

    args = (
        torch.tensor(0.5, device=device),
        torch.tensor(0.0, device=device),
        torch.tensor(2.0, device=device),
        torch.tensor(2.0, device=device),
    )
    actual = compiled(*args)
    expected = (
        expand_bound(f, args[0], "left", step_expand=args[3], max_n_iter=0),
        bisection_method(
            f,
            args[1],
            args[2],
            expand_to_left=False,
            expand_to_right=False,
            max_n_iter=0,
        )[0],
    )
    torch.testing.assert_close(actual, expected)


def test_bisection_method(device, precision):
    """
    Validate bisection_method with batched inputs, testing both compiled and
    non-compiled versions, with and without bracketing expansion.
    """
    if precision == "single":
        dtype = torch.float32
    else:
        dtype = torch.float64

    # Set global config device as bisection_method uses it
    # We save the old device to restore it later
    old_device = config.device
    config.device = device

    try:

        def f1(x, a, b):
            return a - b * x

        def f2(x, a, b):
            return -torch.abs(b) * torch.pow(x - a, 3)

        batch_size = (50, 50)

        # Parameters for f1
        a1 = torch.rand(batch_size, device=device, dtype=dtype) * 1.5 - 0.5  # [-0.5, 1]
        b1 = torch.rand(batch_size, device=device, dtype=dtype)  # [0, 1]

        # Parameters for f2
        a2 = (
            torch.rand(batch_size, device=device, dtype=dtype) * 2.0 - 0.5
        )  # [-0.5, 1.5]
        b2 = torch.rand(batch_size, device=device, dtype=dtype) * 9.9 + 0.1  # [0.1, 10]

        par = {f1: {"a": a1, "b": b1}, f2: {"a": a2, "b": b2}}

        left = torch.zeros(batch_size, device=device, dtype=dtype)
        right = torch.ones(batch_size, device=device, dtype=dtype)
        eps_x = 1e-4
        eps_y = 1e-5

        # ------------------------ #
        # W/O bracketing expansion #
        # ------------------------ #
        for f in [f1, f2]:
            for fun in [bisection_method, bisection_method_compiled]:
                for regula_falsi in [False, True]:
                    x_opt, _, left_opt, right_opt = fun(
                        f,
                        left,
                        right,
                        eps_x=eps_x,
                        eps_y=eps_y,
                        expand_to_left=False,
                        expand_to_right=False,
                        max_n_iter=20000,
                        regula_falsi=regula_falsi,
                        return_brackets=True,
                        precision=precision,
                        a=par[f]["a"],
                        b=par[f]["b"],
                    )

                    # If f(right) >= 0, then must return right
                    condition_right = f(right, par[f]["a"], par[f]["b"]) >= 0
                    if torch.any(condition_right):
                        assert torch.all(
                            x_opt[condition_right] == right[condition_right]
                        ), "Failed checking f(right) >= 0 condition"

                    # If f(left) <= 0, then must return left
                    condition_left = f(left, par[f]["a"], par[f]["b"]) <= 0
                    if torch.any(condition_left):
                        assert torch.all(
                            x_opt[condition_left] == left[condition_left]
                        ), "Failed checking f(left) <= 0 condition"

                    # Else, it returns a point within the bounds with f roughly 0
                    condition_middle = (f(left, par[f]["a"], par[f]["b"]) > 0) & (
                        f(right, par[f]["a"], par[f]["b"]) < 0
                    )

                    if torch.any(condition_middle):
                        f_x_opt_middle = f(
                            x_opt[condition_middle],
                            par[f]["a"][condition_middle],
                            par[f]["b"][condition_middle],
                        )
                        right_opt_middle = right_opt[condition_middle]
                        left_opt_middle = left_opt[condition_middle]

                        # Check convergence
                        converged = (torch.abs(f_x_opt_middle) < eps_y) | (
                            torch.abs(right_opt_middle - left_opt_middle) < eps_x
                        )
                        assert torch.all(
                            converged
                        ), "Failed convergence check w/o expansion"

        # ------------------------- #
        # WITH bracketing expansion #
        # ------------------------- #
        for f in [f1, f2]:
            for fun in [bisection_method, bisection_method_compiled]:
                for regula_falsi in [False, True]:
                    x_opt, _, left_opt, right_opt = fun(
                        f,
                        left,
                        right,
                        eps_x=eps_x,
                        eps_y=eps_y,
                        expand_to_left=True,
                        expand_to_right=True,
                        max_n_iter=20000,
                        regula_falsi=regula_falsi,
                        return_brackets=True,
                        precision=precision,
                        a=par[f]["a"],
                        b=par[f]["b"],
                    )

                    # All roots must be found
                    f_x_opt = f(x_opt, par[f]["a"], par[f]["b"])
                    if f == f1:
                        root = par[f1]["a"] / par[f1]["b"]
                    else:
                        root = par[f2]["a"]

                    # Check convergence
                    converged = (torch.abs(f_x_opt) < eps_y) | (
                        torch.abs(x_opt - root) < eps_x
                    )

                    assert torch.all(
                        converged
                    ), f"Failed convergence check with expansion for {f.__name__}, regula_falsi={regula_falsi}"

    finally:
        config.device = old_device


def test_expand_bound(device, precision):
    """
    Validate expand_bound for both left and right expansion.
    """
    if precision == "single":
        dtype = torch.float32
    else:
        dtype = torch.float64

    # Set global config device
    old_device = config.device
    config.device = device

    try:
        batch_size = (100,)

        # Test 1: Expand to left
        # f(x) = a - x. Decreasing. Root at a.
        # We start at x > a (e.g. 1.0 vs -5.0), so f(x) = a - x < 0.
        # We want to expand left until f(x) > 0.
        def f1(x, a):
            return a - x

        # a in [-10, 0]
        a = torch.rand(batch_size, device=device, dtype=dtype) * 10.0 - 10.0

        # Initial bound = 1.0 (to the right of a)
        bound_init = torch.ones(batch_size, device=device, dtype=dtype)

        # Verify initial condition f(bound) < 0 (negative region)
        assert torch.all(f1(bound_init, a) < 0)

        bound_final = expand_bound(
            f1, bound_init, side="left", step_expand=2.0, precision=precision, a=a
        )

        # Should now be >= 0 (crossed root to the left)
        assert torch.all(f1(bound_final, a) >= 0)
        assert torch.all(bound_final < bound_init)  # Should have moved left

        # Test 2: Expand to right
        # f(x) = a - x. Decreasing. Root at a.
        # We start at x < a (e.g. 0.0 vs 5.0), so f(x) = a - x > 0.
        # We want to expand right until f(x) < 0.
        def f2(x, a):
            return a - x

        # a in [1, 10]
        a = torch.rand(batch_size, device=device, dtype=dtype) * 9.0 + 1.0

        # Initial bound = 0.0 (to the left of a)
        bound_init = torch.zeros(batch_size, device=device, dtype=dtype)

        # Verify initial condition f(bound) > 0 (positive region)
        assert torch.all(f2(bound_init, a) > 0)

        bound_final = expand_bound(
            f2, bound_init, side="right", step_expand=2.0, precision=precision, a=a
        )

        # Should now be <= 0 (crossed root to the right)
        assert torch.all(f2(bound_final, a) <= 0)
        assert torch.all(bound_final > bound_init)  # Should have moved right

    finally:
        config.device = old_device


def test_bisection_docstring_example(device, precision):
    """
    Test the example from the bisection_method docstring to ensure
    API matches documentation.
    """
    old_device = config.device
    config.device = device

    try:
        # Define a decreasing univariate function of x
        def f(x, a):
            return -torch.pow(x - a, 3)

        # Initial search interval
        left, right = 0.0, 2.0
        # Input parameter for function a
        a = 3

        # Perform bisection method
        x_opt, _ = bisection_method(f, left, right, eps_x=1e-4, eps_y=0, a=a)

        # Expected result from docstring: 2.9999084
        assert torch.abs(x_opt - 3.0) < 1e-3

    finally:
        config.device = old_device


def test_bisection_scalar_inputs(device, precision):
    """
    Verify that bisection_method works with scalar (non-tensor) inputs.
    """
    old_device = config.device
    config.device = device

    try:
        # Simple linear function: f(x) = 1 - x. Root at x = 1.
        def f(x):
            return 1.0 - x

        # Scalar inputs
        left = 0.0
        right = 2.0

        x_opt, f_opt = bisection_method(
            f, left, right, eps_x=1e-6, eps_y=1e-6, max_n_iter=100, precision=precision
        )

        # Check root found
        assert torch.abs(x_opt - 1.0) < 1e-5
        assert torch.abs(f_opt) < 1e-5

    finally:
        config.device = old_device


def test_explicit_device_overrides_input_device(device, precision):
    """Explicit placement moves tensor bounds and all returned values."""
    def f(x):
        return 1.0 - x

    bound = expand_bound(
        f,
        torch.tensor(0.0, device="cpu"),
        side="right",
        precision=precision,
        device=device,
    )
    x_opt, f_opt = bisection_method(
        f,
        torch.tensor(0.0, device="cpu"),
        torch.tensor(2.0, device="cpu"),
        precision=precision,
        device=device,
    )

    for value in (bound, x_opt, f_opt):
        assert value.device == torch.device(device)


def test_bisection_return_brackets_false(device, precision):
    """
    Verify that return_brackets=False returns only (x_opt, f_opt).
    """
    if precision == "single":
        dtype = torch.float32
    else:
        dtype = torch.float64

    old_device = config.device
    config.device = device

    try:

        def f(x):
            return 1.0 - x

        left = torch.tensor([0.0], device=device, dtype=dtype)
        right = torch.tensor([2.0], device=device, dtype=dtype)

        result = bisection_method(
            f, left, right, return_brackets=False, precision=precision
        )

        # Should return tuple of 2 elements
        assert isinstance(result, tuple)
        assert len(result) == 2

        x_opt, f_opt = result
        assert torch.abs(x_opt - 1.0) < 1e-4
        assert torch.abs(f_opt) < 1e-4

    finally:
        config.device = old_device


def test_bisection_output_dtype(device, precision):
    """
    Verify that output dtype matches the precision parameter.
    """
    if precision == "single":
        expected_dtype = torch.float32
    else:
        expected_dtype = torch.float64

    old_device = config.device
    config.device = device

    try:

        def f(x):
            return 1.0 - x

        left = 0.0
        right = 2.0

        x_opt, f_opt = bisection_method(f, left, right, precision=precision)

        assert x_opt.dtype == expected_dtype
        assert f_opt.dtype == expected_dtype

    finally:
        config.device = old_device


def test_expand_bound_no_expansion_needed(device, precision):
    """
    Test expand_bound when the initial bound already satisfies the condition.
    """
    if precision == "single":
        dtype = torch.float32
    else:
        dtype = torch.float64

    old_device = config.device
    config.device = device

    try:
        # f(x) = 1 - x. Decreasing. Root at x = 1.
        def f(x):
            return 1.0 - x

        # Left expansion: need f(bound) >= 0.
        # f(0) = 1 > 0. Already satisfied.
        bound_init = torch.tensor([0.0], device=device, dtype=dtype)
        bound_final = expand_bound(f, bound_init, side="left", precision=precision)

        # Should not have moved
        assert torch.allclose(bound_final, bound_init)

        # Right expansion: need f(bound) <= 0.
        # f(2) = -1 < 0. Already satisfied.
        bound_init = torch.tensor([2.0], device=device, dtype=dtype)
        bound_final = expand_bound(f, bound_init, side="right", precision=precision)

        # Should not have moved
        assert torch.allclose(bound_final, bound_init)

    finally:
        config.device = old_device


def test_expand_bound_different_step_expand(device, precision):
    """
    Test expand_bound with different step_expand values.
    """
    if precision == "single":
        dtype = torch.float32
    else:
        dtype = torch.float64

    old_device = config.device
    config.device = device

    try:
        # f(x) = -10 - x. Decreasing. Root at x = -10.
        def f(x):
            return -10.0 - x

        # Initial bound at 0. f(0) = -10 < 0.
        # Need to expand left until f >= 0 (i.e., x <= -10).
        bound_init = torch.tensor([0.0], device=device, dtype=dtype)

        # Test with step_expand=3.0
        bound_final = expand_bound(
            f, bound_init, side="left", step_expand=3.0, precision=precision
        )

        assert torch.all(f(bound_final) >= 0)
        assert torch.all(bound_final <= -10.0)

        # Test with step_expand=1.5 (smaller steps)
        bound_final_slow = expand_bound(
            f, bound_init, side="left", step_expand=1.5, precision=precision
        )

        assert torch.all(f(bound_final_slow) >= 0)
        assert torch.all(bound_final_slow <= -10.0)

    finally:
        config.device = old_device


def test_bisection_immediate_convergence(device, precision):
    """
    Test bisection when root is exactly at the midpoint (immediate convergence).
    """
    if precision == "single":
        dtype = torch.float32
    else:
        dtype = torch.float64

    old_device = config.device
    config.device = device

    try:
        # f(x) = 1 - x. Root at x = 1.
        # If left=0, right=2, midpoint is 1 = root.
        def f(x):
            return 1.0 - x

        left = torch.tensor([0.0], device=device, dtype=dtype)
        right = torch.tensor([2.0], device=device, dtype=dtype)

        x_opt, f_opt = bisection_method(
            f,
            left,
            right,
            eps_x=1e-6,
            eps_y=1e-6,
            expand_to_left=False,
            expand_to_right=False,
            precision=precision,
        )

        # Should find exact root (or very close)
        assert torch.abs(x_opt - 1.0) < 1e-5
        assert torch.abs(f_opt) < 1e-5

    finally:
        config.device = old_device
