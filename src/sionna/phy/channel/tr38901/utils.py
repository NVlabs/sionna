#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Internal utilities for 3GPP TR 38.901 channel models."""

import torch


def update_topology_buffer(
    module: torch.nn.Module,
    name: str,
    value: torch.Tensor,
    shape_error_hint: str = "",
) -> None:
    """Update or register a topology-dependent tensor buffer."""

    existing = getattr(module, name, None)
    if existing is not None and name in module._buffers:
        if existing.shape != value.shape:
            message = (
                f"Cannot change shape of '{name}'. "
                f"Expected {existing.shape}, got {value.shape}."
            )
            if shape_error_hint:
                message = f"{message} {shape_error_hint}"
            raise RuntimeError(message)
        existing.copy_(value)
        return

    if hasattr(module, name) and name not in module._buffers:
        delattr(module, name)

    if torch.compiler.is_compiling():
        raise RuntimeError(
            f"Cannot initialize buffer '{name}' inside torch.compile. "
            "Run one eager warmup first or pre-allocate topology tensors."
        )
    module.register_buffer(name, value)
