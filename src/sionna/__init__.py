#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Sionna Library."""

import importlib
import pkgutil
from types import ModuleType

__version__ = "2.1.0"

# Extend __path__ to include sionna namespace packages (e.g., sionna-rt)
__path__ = pkgutil.extend_path(__path__, __name__)


# pylint: disable=invalid-name
def __getattr__(name: str) -> ModuleType:
    if name in ["phy", "sys", "rt"]:
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__} has no attribute {name}")
