#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Tests for Sionna's private shared validation helpers."""

import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest
import torch

from sionna._validation import (
    check_binary,
    check_instance,
    check_one_of,
    check_scalar_range,
    check_sequence_of,
    check_tensor,
    check_tensor_all,
    check_tensor_range,
    check_tensor_values_in,
)


SOURCE_ROOT = Path(__file__).resolve().parents[2] / "src"


def _subprocess_env() -> dict[str, str]:
    """Create an environment that imports this checkout first."""
    env = os.environ.copy()
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        f"{SOURCE_ROOT}{os.pathsep}{existing}" if existing else str(SOURCE_ROOT)
    )
    return env


class _Base:
    """Base type for instance-check tests."""


class _Child(_Base):
    """Child type for instance-check tests."""


def test_check_instance_and_exact_type():
    """Instance checks support subclasses and explicit exact-type checks."""
    value = _Child()
    check_instance(value, _Base, name="value")
    check_instance(value, _Child, name="value", exact=True)

    with pytest.raises(
        TypeError,
        match=r"`value` must have type _Base; got _Child\.",
    ):
        check_instance(value, _Base, name="value", exact=True)


def test_check_instance_message_override():
    """Static helpers preserve caller-provided domain context."""
    with pytest.raises(TypeError, match="custom type contract"):
        check_instance(
            object(),
            _Base,
            name="value",
            message="custom type contract",
        )


def test_check_one_of():
    """Python membership checks produce deterministic messages."""
    check_one_of("left", ("left", "right"), name="side")
    with pytest.raises(
        ValueError,
        match=r"`side` must be one of \('left', 'right'\); got 'center'\.",
    ):
        check_one_of("center", ("left", "right"), name="side")
    with pytest.raises(ValueError, match="choices must not be empty"):
        check_one_of("left", (), name="side")
    with pytest.raises(TypeError, match="ordered sequence"):
        check_one_of("left", {"left", "right"}, name="side")
    with pytest.raises(TypeError, match="does not accept tensors"):
        check_one_of(torch.tensor(1), (0, 1), name="value")


@pytest.mark.parametrize(
    ("value", "kwargs", "raises"),
    [
        (0.0, {"minimum": 0.0}, False),
        (0.0, {"minimum": 0.0, "lower_inclusive": False}, True),
        (1.0, {"maximum": 1.0}, False),
        (1.0, {"maximum": 1.0, "upper_inclusive": False}, True),
        (0.5, {"minimum": 0.0, "maximum": 1.0}, False),
        (float("nan"), {"minimum": 0.0}, True),
    ],
)
def test_check_scalar_range(value, kwargs, raises):
    """Scalar ranges support one-sided and open/closed intervals."""
    if raises:
        with pytest.raises(ValueError):
            check_scalar_range(value, name="value", **kwargs)
    else:
        check_scalar_range(value, name="value", **kwargs)


def test_check_scalar_range_rejects_invalid_helper_use():
    """Scalar range metadata errors are explicit."""
    with pytest.raises(ValueError, match="minimum or maximum"):
        check_scalar_range(1, name="value")
    with pytest.raises(TypeError, match="does not accept tensors"):
        check_scalar_range(torch.tensor(1), name="value", minimum=0)


def test_check_sequence_of():
    """Sequence checks cover container, item type, and length."""
    check_sequence_of([_Child(), _Child()], _Base, name="values", length=2)
    check_sequence_of((_Child(),), _Base, name="values", min_length=1)

    with pytest.raises(TypeError, match=r"`values` must be a list, tuple"):
        check_sequence_of({_Child()}, _Base, name="values")
    with pytest.raises(ValueError, match="exactly 2 items"):
        check_sequence_of([_Child()], _Base, name="values", length=2)
    with pytest.raises(TypeError, match="item 1"):
        check_sequence_of([_Child(), object()], _Base, name="values")


def test_check_sequence_of_rejects_invalid_helper_use():
    """Conflicting or negative length constraints are rejected."""
    with pytest.raises(ValueError, match="mutually exclusive"):
        check_sequence_of(
            [],
            object,
            name="values",
            length=0,
            min_length=0,
        )
    with pytest.raises(ValueError, match="non-negative"):
        check_sequence_of([], object, name="values", length=-1)


@pytest.mark.parametrize("error_type", [TypeError, ValueError, RuntimeError])
def test_check_tensor_eager_exception(error_type, device):
    """Eager tensor checks preserve Sionna's exception classes and messages."""
    check_tensor(
        torch.tensor(True, dtype=torch.bool, device=device),
        "tensor contract",
        error_type=error_type,
    )
    with pytest.raises(error_type, match="tensor contract"):
        check_tensor(
            torch.tensor(False, dtype=torch.bool, device=device),
            "tensor contract",
            error_type=error_type,
        )


def test_check_tensor_rejects_invalid_helper_use(device):
    """The primitive requires a scalar Boolean tensor and supported exception."""
    with pytest.raises(TypeError, match="torch.Tensor"):
        check_tensor(True, "contract")
    with pytest.raises(ValueError, match="scalar Boolean"):
        check_tensor(torch.tensor(1, device=device), "contract")
    with pytest.raises(ValueError, match="scalar Boolean"):
        check_tensor(
            torch.tensor([True, False], device=device),
            "contract",
        )
    with pytest.raises(TypeError, match="error_type"):
        check_tensor(
            torch.tensor(True, device=device),
            "contract",
            error_type=NotImplementedError,
        )


def test_tensor_conveniences_eager(device):
    """Common tensor predicates share eager exception behavior."""
    values = torch.tensor([0.0, 0.5, 1.0], device=device)
    check_tensor_all(values >= 0, name="values")
    check_tensor_range(
        values,
        name="values",
        minimum=0.0,
        maximum=1.0,
    )
    check_tensor_values_in(
        torch.tensor([0, 2, 4], device=device),
        (0, 2, 4),
        name="values",
    )
    check_binary(torch.tensor([0, 1, 1], device=device), name="bits")
    check_binary(
        torch.tensor([-1, 1, -1], device=device),
        name="symbols",
        bipolar=True,
    )

    with pytest.raises(ValueError, match="values"):
        check_tensor_range(
            values,
            name="values",
            minimum=0.0,
            maximum=1.0,
            lower_inclusive=False,
        )
    with pytest.raises(ValueError, match="category"):
        check_tensor_values_in(
            torch.tensor([0, 2], device=device),
            (0, 1),
            name="category",
        )
    with pytest.raises(ValueError, match="bits"):
        check_binary(torch.tensor([0, 2], device=device), name="bits")


def test_tensor_conveniences_empty_and_nan(device):
    """Empty predicates are vacuously true while NaN fails bounded ranges."""
    empty = torch.empty(0, device=device)
    check_tensor_range(empty, name="empty", minimum=0.0)
    check_tensor_values_in(empty, (0.0, 1.0), name="empty")

    with pytest.raises(ValueError, match="values"):
        check_tensor_range(
            torch.tensor([float("nan")], device=device),
            name="values",
            minimum=0.0,
        )


def test_tensor_conveniences_reject_invalid_helper_use(device):
    """Tensor convenience metadata failures remain ordinary Python errors."""
    with pytest.raises(ValueError, match="dtype torch.bool"):
        check_tensor_all(torch.ones(2, device=device), name="values")
    with pytest.raises(ValueError, match="minimum or maximum"):
        check_tensor_range(torch.ones(2, device=device), name="values")
    with pytest.raises(ValueError, match="choices must not be empty"):
        check_tensor_values_in(
            torch.ones(2, device=device),
            (),
            name="values",
        )
    with pytest.raises(TypeError, match="ordered sequence"):
        check_tensor_values_in(
            torch.ones(2, device=device),
            {0, 1},
            name="values",
        )


@pytest.mark.parametrize("backend", ["eager", "inductor"])
def test_tensor_check_cold_compiles_fullgraph(backend, device):
    """A valid first invocation compiles as one full graph."""

    @torch.compile(backend=backend, fullgraph=True)
    def checked(value):
        check_tensor_range(
            value,
            name="value",
            minimum=0.0,
            maximum=1.0,
        )
        return value.square()

    value = torch.tensor([0.0, 0.5, 1.0], device=device)
    torch.testing.assert_close(checked(value), value.square())


def test_tensor_check_dynamic_shapes_use_one_graph(device):
    """One Dynamo graph handles multiple valid tensor shapes."""
    graph_count = 0

    def counting_backend(graph_module, _example_inputs):
        nonlocal graph_count
        graph_count += 1
        return graph_module.forward

    @torch.compile(
        backend=counting_backend,
        fullgraph=True,
        dynamic=True,
    )
    def checked(value):
        check_binary(value, name="value")
        check_tensor_range(
            value,
            name="value",
            minimum=-value.shape[0],
            maximum=value.shape[1],
            message="symbolic range contract",
        )
        return value + 1

    for shape in ((2, 3), (5, 7)):
        value = torch.zeros(shape, device=device)
        assert checked(value).shape == shape
    assert graph_count == 1


def test_eager_tensor_check_survives_optimized_python(device):
    """A real ``python -O`` interpreter still raises the eager exception."""
    script = textwrap.dedent(
        f"""
        import sys
        import torch
        from sionna._validation import check_tensor

        if sys.flags.optimize == 0:
            raise SystemExit(10)
        try:
            check_tensor(
                torch.tensor(False, device={device!r}),
                "optimized tensor contract",
            )
        except ValueError as error:
            if str(error) == "optimized tensor contract":
                raise SystemExit(0)
        raise SystemExit(11)
        """
    )
    result = subprocess.run(
        [sys.executable, "-O", "-c", script],
        check=False,
        capture_output=True,
        text=True,
        env=_subprocess_env(),
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("backend", ["eager", "inductor"])
@pytest.mark.parametrize("warm_first", [False, True])
def test_invalid_compiled_tensor_check_is_isolated(
    backend,
    warm_first,
    device,
    tmp_path,
):
    """Cold and warm invalid GPU inputs assert in a disposable process."""
    if not device.startswith("cuda"):
        pytest.skip("Device-assert isolation requires a CUDA device")
    cache_dir = tmp_path / f"torchinductor-{backend}-{warm_first}"
    script = textwrap.dedent(
        f"""
        import os
        import torch
        from sionna._validation import check_tensor_range

        @torch.compile(backend={backend!r}, fullgraph=True)
        def checked(value):
            check_tensor_range(
                value,
                name="value",
                minimum=0.0,
                message="compiled tensor contract",
            )
            return value.square()

        try:
            if {warm_first!r}:
                checked(torch.tensor([1.0, 2.0], device={device!r}))
                torch.cuda.synchronize()
            checked(torch.tensor([1.0, -1.0], device={device!r}))
            torch.cuda.synchronize()
        except BaseException as error:
            print(type(error).__name__, str(error), flush=True)
            os._exit(17)
        print("compiled tensor contract was not enforced", flush=True)
        os._exit(0)
        """
    )
    env = _subprocess_env()
    env["CUDA_LAUNCH_BLOCKING"] = "1"
    env["TORCHINDUCTOR_CACHE_DIR"] = str(cache_dir)
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    output = result.stdout + result.stderr
    assert result.returncode != 0, output
    assert "compiled tensor contract" in output
