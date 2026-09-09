#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Tests for LDPC decoder utility callbacks."""

import torch

from sionna.phy.fec.ldpc.utils import EXITCallback


def test_exit_callback_uses_stable_accumulator_dtypes(device):
    """EXIT statistics use wide sums and integer sample counters."""
    callback = EXITCallback(num_iter=2, device=device)

    assert callback._mi.dtype == torch.float64
    assert callback._num_samples.dtype == torch.int64
    assert callback._mi.device == torch.device(device)
    assert callback._num_samples.device == torch.device(device)

    msg = torch.tensor([[-1.0, 0.5, 0.0]], device=device)
    result = callback(msg, 0)

    assert result is msg
    assert callback._num_samples[0].item() == 1
    assert callback.mi.dtype == torch.float64
