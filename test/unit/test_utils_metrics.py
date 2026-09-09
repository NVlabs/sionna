#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
import pytest
import torch
from sionna.phy.utils import metrics
from sionna.phy.config import dtypes


@pytest.mark.parametrize(
    "metric_name",
    [
        "compute_ber",
        "compute_ser",
        "compute_bler",
        "count_errors",
        "count_block_errors",
    ],
)
def test_metrics_reject_mismatched_shapes(metric_name, device):
    """Metrics never broadcast estimates across incompatible shapes."""
    metric = getattr(metrics, metric_name)
    values = torch.zeros((2, 1), device=device)
    estimates = torch.zeros((2, 3), device=device)
    with pytest.raises(ValueError, match="same shape"):
        metric(values, estimates)


@pytest.mark.parametrize(
    "metric_name",
    [
        "compute_ber",
        "compute_ser",
        "compute_bler",
        "count_errors",
        "count_block_errors",
    ],
)
def test_metrics_reject_empty_inputs(metric_name, device):
    """Metrics do not report NaN or zero for an empty observation."""
    metric = getattr(metrics, metric_name)
    values = torch.empty((0, 4), device=device)
    with pytest.raises(ValueError, match="must not be empty"):
        metric(values, values.clone())


def test_compute_ber(device, precision):
    """Test compute_ber function."""

    # Check return dtype
    b = torch.zeros((100,), device=device)
    b_hat = torch.zeros((100,), device=device)
    ber = metrics.compute_ber(b, b_hat, precision)
    assert ber.dtype == dtypes[precision]["torch"]["dtype"]
    assert ber.device.type == torch.device(device).type

    # Check correctness
    # Case 1: No errors
    b = torch.tensor([0, 1, 0, 1], device=device, dtype=torch.float32)
    b_hat = torch.tensor([0, 1, 0, 1], device=device, dtype=torch.float32)
    ber = metrics.compute_ber(b, b_hat, precision)
    assert ber == 0.0

    # Case 2: All errors
    b_hat = torch.tensor([1, 0, 1, 0], device=device, dtype=torch.float32)
    ber = metrics.compute_ber(b, b_hat, precision)
    assert ber == 1.0

    # Case 3: Docstring example (50% errors)
    b_hat = torch.tensor([0, 1, 1, 0], device=device, dtype=torch.float32)
    ber = metrics.compute_ber(b, b_hat, precision)
    assert ber == 0.5

    # Case 4: Different shapes (broadcasting) - although function doc says "same shape",
    # usually elementwise ops support broadcasting, but let's stick to simple cases or check docs.
    # Doc says "A tensor like b".

    # Case 5: Higher dimensions
    b = torch.zeros((10, 10), device=device)
    b_hat = torch.zeros((10, 10), device=device)
    b_hat[0, :] = 1  # 10 errors out of 100
    ber = metrics.compute_ber(b, b_hat, precision)
    assert torch.isclose(ber, torch.tensor(0.1, dtype=ber.dtype))


def test_compute_ser(device, precision):
    """Test compute_ser function."""

    # Check return dtype
    s = torch.zeros((100,), device=device, dtype=torch.int32)
    s_hat = torch.zeros((100,), device=device, dtype=torch.int32)
    ser = metrics.compute_ser(s, s_hat, precision)
    assert ser.dtype == dtypes[precision]["torch"]["dtype"]
    assert ser.device.type == torch.device(device).type

    # Check correctness
    # Case 1: No errors
    s = torch.tensor([0, 1, 2, 3], device=device, dtype=torch.int32)
    s_hat = torch.tensor([0, 1, 2, 3], device=device, dtype=torch.int32)
    ser = metrics.compute_ser(s, s_hat, precision)
    assert ser == 0.0

    # Case 2: All errors
    s_hat = torch.tensor([1, 2, 3, 0], device=device, dtype=torch.int32)
    ser = metrics.compute_ser(s, s_hat, precision)
    assert ser == 1.0

    # Case 3: Docstring example (50% errors)
    s_hat = torch.tensor([0, 1, 3, 2], device=device, dtype=torch.int32)
    ser = metrics.compute_ser(s, s_hat, precision)
    assert ser == 0.5


def test_compute_bler(device, precision):
    """Test compute_bler function."""

    # Check return dtype
    b = torch.zeros((10, 10), device=device)
    b_hat = torch.zeros((10, 10), device=device)
    bler = metrics.compute_bler(b, b_hat, precision)
    assert bler.dtype == dtypes[precision]["torch"]["dtype"]
    assert bler.device.type == torch.device(device).type

    # Check correctness
    # Case 1: No errors
    b = torch.zeros((2, 5), device=device)  # 2 blocks of length 5
    b_hat = torch.zeros((2, 5), device=device)
    bler = metrics.compute_bler(b, b_hat, precision)
    assert bler == 0.0

    # Case 2: One error in one block
    b_hat[0, 0] = 1
    # Block 0 is error, Block 1 is correct. BLER = 0.5
    bler = metrics.compute_bler(b, b_hat, precision)
    assert bler == 0.5

    # Case 3: Errors in all blocks
    b_hat[1, 0] = 1
    bler = metrics.compute_bler(b, b_hat, precision)
    assert bler == 1.0

    # Case 4: Multiple errors in same block don't double count
    b_hat[0, 1] = 1
    bler = metrics.compute_bler(b, b_hat, precision)
    assert bler == 1.0

    # Docstring example: error only in second block
    b = torch.tensor([[0, 1], [1, 0]], device=device)
    b_hat = torch.tensor([[0, 1], [1, 1]], device=device)
    bler = metrics.compute_bler(b, b_hat, precision)
    assert bler == 0.5


def test_count_errors(device):
    """Test count_errors function."""

    # Check correctness
    b = torch.tensor([0, 1, 0, 1], device=device, dtype=torch.float32)

    # Case 1: No errors
    b_hat = torch.tensor([0, 1, 0, 1], device=device, dtype=torch.float32)
    errors = metrics.count_errors(b, b_hat)
    assert errors == 0
    assert errors.dtype == torch.int64

    # Case 2: All errors
    b_hat = torch.tensor([1, 0, 1, 0], device=device, dtype=torch.float32)
    errors = metrics.count_errors(b, b_hat)
    assert errors == 4

    # Case 3: Docstring example (2 errors)
    b_hat = torch.tensor([0, 1, 1, 0], device=device, dtype=torch.float32)
    errors = metrics.count_errors(b, b_hat)
    assert errors == 2


def test_count_block_errors(device):
    """Test count_block_errors function."""

    # Check correctness
    b = torch.zeros((2, 5), device=device)  # 2 blocks of length 5

    # Case 1: No errors
    b_hat = torch.zeros((2, 5), device=device)
    errors = metrics.count_block_errors(b, b_hat)
    assert errors == 0
    assert errors.dtype == torch.int64

    # Case 2: One block error
    b_hat[0, 0] = 1
    errors = metrics.count_block_errors(b, b_hat)
    assert errors == 1

    # Case 3: Two block errors
    b_hat[1, 0] = 1
    errors = metrics.count_block_errors(b, b_hat)
    assert errors == 2

    # Case 4: Multiple errors in same block don't double count
    b_hat[0, 1] = 1
    errors = metrics.count_block_errors(b, b_hat)
    assert errors == 2

    # Docstring example: only the second block is in error
    b_small = torch.tensor([[0, 1], [1, 0]], device=device)
    b_hat_small = torch.tensor([[0, 1], [1, 1]], device=device)
    errors = metrics.count_block_errors(b_small, b_hat_small)
    assert errors == 1
