#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Private validation helpers shared by Sionna PHY and SYS."""

from collections.abc import Sequence
from typing import Any, Optional

import torch


__all__: list[str] = []

_ERROR_TYPES = (TypeError, ValueError, RuntimeError)


def _message(override: Optional[str], fallback: str) -> str:
    """Return a caller-provided message or a generated fallback."""
    return fallback if override is None else override


def _type_names(expected: type | tuple[type, ...]) -> str:
    """Format one or more expected Python types."""
    expected_types = expected if isinstance(expected, tuple) else (expected,)
    return ", ".join(expected_type.__name__ for expected_type in expected_types)


def _range_message(
    name: str,
    minimum: Any,
    maximum: Any,
    lower_inclusive: bool,
    upper_inclusive: bool,
    *,
    tensor: bool,
) -> str:
    """Build a deterministic scalar or tensor range message."""
    subject = f"`{name}` must contain values" if tensor else f"`{name}` must be"
    if minimum is not None and maximum is not None:
        left = "[" if lower_inclusive else "("
        right = "]" if upper_inclusive else ")"
        return f"{subject} in {left}{minimum}, {maximum}{right}."
    if minimum is not None:
        operator = ">=" if lower_inclusive else ">"
        return f"{subject} {operator} {minimum}."
    operator = "<=" if upper_inclusive else "<"
    return f"{subject} {operator} {maximum}."


def _check_error_type(error_type: type[Exception]) -> None:
    """Reject exception classes outside Sionna's validation contract."""
    if error_type not in _ERROR_TYPES:
        raise TypeError(
            "error_type must be TypeError, ValueError, or RuntimeError"
        )


def check_tensor(
    condition: torch.Tensor,
    message: str,
    *,
    error_type: type[Exception] = ValueError,
) -> None:
    """Validate a scalar Boolean tensor in eager and compiled execution.

    Eager execution raises ``error_type``. During ``torch.compile``, the
    condition is emitted as a device assertion to avoid a Dynamo graph break.
    Compiled failures therefore do not preserve the eager exception type.
    """
    _check_error_type(error_type)
    if not isinstance(condition, torch.Tensor):
        raise TypeError("condition must be a torch.Tensor")
    if condition.dtype != torch.bool or condition.numel() != 1:
        raise ValueError("condition must be a scalar Boolean tensor")

    if torch.compiler.is_compiling():
        # pylint: disable-next=protected-access
        torch._assert_async(condition, message)
    elif not bool(condition):
        raise error_type(message)


def check_tensor_all(
    predicate: torch.Tensor,
    *,
    name: str,
    message: Optional[str] = None,
    error_type: type[Exception] = ValueError,
) -> None:
    """Require every element of a Boolean tensor predicate to be true."""
    if not isinstance(predicate, torch.Tensor):
        raise TypeError("predicate must be a torch.Tensor")
    if predicate.dtype != torch.bool:
        raise ValueError("predicate must have dtype torch.bool")
    check_tensor(
        torch.all(predicate),
        _message(message, f"`{name}` contains invalid values."),
        error_type=error_type,
    )


def check_tensor_range(
    value: torch.Tensor,
    *,
    name: str,
    minimum: Any = None,
    maximum: Any = None,
    lower_inclusive: bool = True,
    upper_inclusive: bool = True,
    message: Optional[str] = None,
    error_type: type[Exception] = ValueError,
) -> None:
    """Require every tensor element to lie within the requested interval."""
    if not isinstance(value, torch.Tensor):
        raise TypeError("value must be a torch.Tensor")
    if minimum is None and maximum is None:
        raise ValueError("minimum or maximum must be provided")

    predicate = torch.ones_like(value, dtype=torch.bool)
    if minimum is not None:
        predicate &= value >= minimum if lower_inclusive else value > minimum
    if maximum is not None:
        predicate &= value <= maximum if upper_inclusive else value < maximum

    if message is None:
        message = _range_message(
            name,
            minimum,
            maximum,
            lower_inclusive,
            upper_inclusive,
            tensor=True,
        )
    check_tensor_all(
        predicate,
        name=name,
        message=message,
        error_type=error_type,
    )


def check_tensor_values_in(
    value: torch.Tensor,
    choices: Sequence[Any],
    *,
    name: str,
    message: Optional[str] = None,
    error_type: type[Exception] = ValueError,
) -> None:
    """Require every tensor element to equal one of the static choices."""
    if not isinstance(value, torch.Tensor):
        raise TypeError("value must be a torch.Tensor")
    if not isinstance(choices, Sequence):
        raise TypeError("choices must be an ordered sequence")
    choices = tuple(choices)
    if not choices:
        raise ValueError("choices must not be empty")

    predicate = value == choices[0]
    for choice in choices[1:]:
        predicate |= value == choice

    if message is None:
        message = f"`{name}` must contain only values from {choices}."
    check_tensor_all(
        predicate,
        name=name,
        message=message,
        error_type=error_type,
    )


def check_binary(
    value: torch.Tensor,
    *,
    name: str,
    bipolar: bool = False,
    message: Optional[str] = None,
    error_type: type[Exception] = ValueError,
) -> None:
    """Require tensor values to be binary or bipolar."""
    choices = (-1, 1) if bipolar else (0, 1)
    default = (
        f"`{name}` must contain only -1 and 1."
        if bipolar
        else f"`{name}` must contain only 0 and 1."
    )
    check_tensor_values_in(
        value,
        choices,
        name=name,
        message=_message(message, default),
        error_type=error_type,
    )


def check_instance(
    value: Any,
    expected: type | tuple[type, ...],
    *,
    name: str,
    exact: bool = False,
    message: Optional[str] = None,
) -> None:
    """Require a Python value to have one of the expected types."""
    expected_types = expected if isinstance(expected, tuple) else (expected,)
    valid = (
        type(value) in expected_types
        if exact
        else isinstance(value, expected_types)
    )
    if not valid:
        relation = "have type" if exact else "be an instance of"
        raise TypeError(
            _message(
                message,
                f"`{name}` must {relation} {_type_names(expected_types)}; "
                f"got {type(value).__name__}.",
            )
        )


def check_one_of(
    value: Any,
    choices: Sequence[Any],
    *,
    name: str,
    message: Optional[str] = None,
) -> None:
    """Require a Python value to equal one of the static choices."""
    if isinstance(value, torch.Tensor):
        raise TypeError(
            "check_one_of does not accept tensors; use check_tensor_values_in"
        )
    if not isinstance(choices, Sequence):
        raise TypeError("choices must be an ordered sequence")
    if len(choices) == 0:
        raise ValueError("choices must not be empty")
    if value not in choices:
        if message is None:
            message = f"`{name}` must be one of {tuple(choices)}; got {value!r}."
        raise ValueError(
            message
        )


def check_scalar_range(
    value: Any,
    *,
    name: str,
    minimum: Any = None,
    maximum: Any = None,
    lower_inclusive: bool = True,
    upper_inclusive: bool = True,
    message: Optional[str] = None,
) -> None:
    """Require a non-tensor scalar to lie within the requested interval."""
    if isinstance(value, torch.Tensor):
        raise TypeError(
            "check_scalar_range does not accept tensors; use check_tensor_range"
        )
    if minimum is None and maximum is None:
        raise ValueError("minimum or maximum must be provided")

    valid = True
    if minimum is not None:
        valid &= value >= minimum if lower_inclusive else value > minimum
    if maximum is not None:
        valid &= value <= maximum if upper_inclusive else value < maximum
    if not valid:
        raise ValueError(
            _message(
                message,
                _range_message(
                    name,
                    minimum,
                    maximum,
                    lower_inclusive,
                    upper_inclusive,
                    tensor=False,
                ),
            )
        )


def check_sequence_of(
    value: Any,
    item_type: type | tuple[type, ...],
    *,
    name: str,
    sequence_type: type | tuple[type, ...] = (list, tuple),
    length: Optional[int] = None,
    min_length: Optional[int] = None,
    message: Optional[str] = None,
) -> None:
    """Require a sequence container, item type, and optional length."""
    if length is not None and min_length is not None:
        raise ValueError("length and min_length are mutually exclusive")
    if length is not None and length < 0:
        raise ValueError("length must be non-negative")
    if min_length is not None and min_length < 0:
        raise ValueError("min_length must be non-negative")

    if not isinstance(value, sequence_type):
        raise TypeError(
            _message(
                message,
                f"`{name}` must be a {_type_names(sequence_type)}; "
                f"got {type(value).__name__}.",
            )
        )
    if length is not None and len(value) != length:
        raise ValueError(
            _message(
                message,
                f"`{name}` must contain exactly {length} items; "
                f"got {len(value)}.",
            )
        )
    if min_length is not None and len(value) < min_length:
        raise ValueError(
            _message(
                message,
                f"`{name}` must contain at least {min_length} items; "
                f"got {len(value)}.",
            )
        )

    for index, item in enumerate(value):
        if not isinstance(item, item_type):
            raise TypeError(
                _message(
                    message,
                    f"`{name}` item {index} must be an instance of "
                    f"{_type_names(item_type)}; got {type(item).__name__}.",
                )
            )
