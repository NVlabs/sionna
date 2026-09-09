#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Light config.seed reproducibility checks across public PHY entry points."""

import torch

from sionna.phy import config
from sionna.phy.channel import AWGN, BinarySymmetricChannel, RayleighBlockFading
from sionna.phy.utils import complex_normal, normal


def _run_twice(seed, fn):
    """Reset config.seed and invoke ``fn`` twice; return both results."""
    config.seed = seed
    first = fn()
    config.seed = seed
    second = fn()
    return first, second


class TestConfigSeedPublicEntryPoints:
    """Representative PHY draws must follow config.seed in eager mode.

    The root ``set_seed`` autouse fixture resets ``config.seed`` before each
    test, so these methods do not restore the prior seed themselves.
    """

    def test_complex_normal_and_normal(self, device):
        """Compile-aware noise helpers honor config.seed."""

        def draw():
            return (
                complex_normal([32], device=device),
                normal([32], device=device),
            )

        (c0, n0), (c1, n1) = _run_twice(123, draw)
        assert torch.equal(c0, c1)
        assert torch.equal(n0, n1)

        config.seed = 456
        c2, n2 = draw()
        assert not torch.equal(c0, c2)
        assert not torch.equal(n0, n2)

    def test_awgn(self, device):
        """AWGN noise draws follow config.seed."""
        x = torch.zeros(8, 4, dtype=torch.complex64, device=device)

        def draw():
            return AWGN(device=device)(x, 0.1)

        y0, y1 = _run_twice(123, draw)
        assert torch.equal(y0, y1)

        config.seed = 456
        y2 = draw()
        assert not torch.equal(y0, y2)

    def test_binary_symmetric_channel(self, device):
        """BSC bit flips follow config.seed."""
        x = torch.zeros(256, dtype=torch.float32, device=device)

        def draw():
            return BinarySymmetricChannel(device=device)(x, 0.25)

        y0, y1 = _run_twice(123, draw)
        assert torch.equal(y0, y1)

        config.seed = 456
        y2 = draw()
        assert not torch.equal(y0, y2)

    def test_rayleigh_block_fading(self, device):
        """Rayleigh block fading coefficients follow config.seed."""

        def draw():
            model = RayleighBlockFading(
                num_rx=1,
                num_rx_ant=2,
                num_tx=1,
                num_tx_ant=2,
                device=device,
            )
            a, _tau = model(batch_size=4, num_time_steps=8)
            return a

        a0, a1 = _run_twice(123, draw)
        assert torch.equal(a0, a1)

        config.seed = 456
        a2 = draw()
        assert not torch.equal(a0, a2)
