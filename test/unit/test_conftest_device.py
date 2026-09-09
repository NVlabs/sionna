#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Unit tests for test/conftest.py device-option resolution."""

from conftest import resolve_device_option


def test_resolve_device_auto_prefers_gpu_when_cuda_available():
    assert resolve_device_option("auto", cuda_available=True) == "gpu"


def test_resolve_device_auto_falls_back_to_cpu_without_cuda():
    assert resolve_device_option("auto", cuda_available=False) == "cpu"


def test_resolve_device_explicit_values_unchanged():
    assert resolve_device_option("cpu", cuda_available=False) == "cpu"
    assert resolve_device_option("gpu", cuda_available=True) == "gpu"
    assert resolve_device_option("all", cuda_available=True) == "all"
