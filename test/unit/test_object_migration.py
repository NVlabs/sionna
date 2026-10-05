#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Regression tests for issue #1167.

Verifies ``.to()`` syncs the Sionna logical device and plain tensor
attributes. Uses ``meta`` so tests run without GPU.
"""
import torch

from sionna.phy import Block, Object
from sionna.phy.channel import AWGN


class _PlainAndBuffer(Block):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._plain = torch.ones(4)
        self.register_buffer("_buf", torch.ones(4))

    def call(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs


def test_to_syncs_logical_device():
    obj = Object(device="cpu")
    obj.to("meta")
    assert obj.device == "meta"


def test_awgn_syncs_logical_device():
    awgn = AWGN(device="cpu")
    awgn.to("meta")
    assert awgn.device == "meta"


def test_to_migrates_plain_tensors_and_buffers():
    block = _PlainAndBuffer(device="cpu")
    block.to("meta")
    assert block.device == "meta"
    assert block._plain.device == torch.device("meta")
    assert block._buf.device == torch.device("meta")


def test_nested_child_syncs_via_plain_wrapper():
    class Wrapper(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.awgn = AWGN(device="cpu")

    wrapper = Wrapper()
    wrapper.to("meta")
    assert wrapper.awgn.device == "meta"


def test_dtype_only_migration_preserves_device():
    obj = Object(device="cpu")
    obj.to("meta")
    obj.to(torch.float64)
    assert obj.device == "meta"
    obj.float()
    assert obj.device == "meta"
