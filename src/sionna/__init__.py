#
# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0#
"""Sionna Library"""

import importlib

__version__ = "1.2.1"

# pylint: disable=invalid-name
def __getattr__(name):
    if name in ["phy", "sys", "rt"]:
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__} has no attribute {name}")
