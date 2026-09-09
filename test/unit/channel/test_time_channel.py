#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Tests for the time domain channel components"""

import torch
import pytest

from sionna.phy.channel import (
    GenerateTimeChannel,
    ApplyTimeChannel,
    TimeChannel,
    RayleighBlockFading,
)


class TestGenerateTimeChannel:
    """Tests for the GenerateTimeChannel class"""

    def test_output_shape(self, device, precision):
        """Verify output shape matches expected dimensions"""
        num_rx = 2
        num_rx_ant = 4
        num_tx = 3
        num_tx_ant = 2
        batch_size = 16
        num_time_samples = 100
        l_min = -6
        l_max = 20
        l_tot = l_max - l_min + 1

        channel_model = RayleighBlockFading(
            num_rx=num_rx,
            num_rx_ant=num_rx_ant,
            num_tx=num_tx,
            num_tx_ant=num_tx_ant,
            precision=precision,
            device=device,
        )

        gen_channel = GenerateTimeChannel(
            channel_model,
            bandwidth=1e6,
            num_time_samples=num_time_samples,
            l_min=l_min,
            l_max=l_max,
            precision=precision,
            device=device,
        )

        h_time = gen_channel(batch_size)

        expected_shape = torch.Size(
            [
                batch_size,
                num_rx,
                num_rx_ant,
                num_tx,
                num_tx_ant,
                num_time_samples + l_max - l_min,
                l_tot,
            ]
        )
        assert h_time.shape == expected_shape

    def test_output_dtype(self, device, precision):
        """Verify output dtype matches precision setting"""
        channel_model = RayleighBlockFading(
            num_rx=1,
            num_rx_ant=2,
            num_tx=1,
            num_tx_ant=4,
            precision=precision,
            device=device,
        )

        gen_channel = GenerateTimeChannel(
            channel_model,
            bandwidth=1e6,
            num_time_samples=100,
            l_min=-6,
            l_max=20,
            precision=precision,
            device=device,
        )

        h_time = gen_channel(batch_size=32)

        expected_cdtype = torch.complex64 if precision == "single" else torch.complex128
        assert h_time.dtype == expected_cdtype

    def test_output_device(self, device):
        """Verify outputs are on the correct device"""
        channel_model = RayleighBlockFading(
            num_rx=1, num_rx_ant=2, num_tx=1, num_tx_ant=4, device=device
        )

        gen_channel = GenerateTimeChannel(
            channel_model,
            bandwidth=1e6,
            num_time_samples=100,
            l_min=-6,
            l_max=20,
            device=device,
        )

        h_time = gen_channel(batch_size=32)
        assert h_time.device == torch.device(device)

    def test_properties(self, device):
        """Verify properties return correct values"""
        channel_model = RayleighBlockFading(
            num_rx=1, num_rx_ant=2, num_tx=1, num_tx_ant=4, device=device
        )

        l_min = -6
        l_max = 20
        bandwidth = 1e6
        num_time_samples = 100

        gen_channel = GenerateTimeChannel(
            channel_model,
            bandwidth=bandwidth,
            num_time_samples=num_time_samples,
            l_min=l_min,
            l_max=l_max,
            device=device,
        )

        assert gen_channel.l_min == l_min
        assert gen_channel.l_max == l_max
        assert gen_channel.l_tot == l_max - l_min + 1
        assert gen_channel.bandwidth == bandwidth
        assert gen_channel.num_time_samples == num_time_samples

    def test_channel_normalization(self, device):
        """Verify channel normalization produces unit average energy per time step"""
        channel_model = RayleighBlockFading(
            num_rx=1,
            num_rx_ant=1,
            num_tx=1,
            num_tx_ant=1,
            precision="double",
            device=device,
        )

        gen_channel = GenerateTimeChannel(
            channel_model,
            bandwidth=1e6,
            num_time_samples=100,
            l_min=-6,
            l_max=20,
            normalize_channel=True,
            precision="double",
            device=device,
        )

        # Generate many samples
        h_time = gen_channel(batch_size=1000)

        # Compute average energy per time step (sum over taps, mean over time and batch)
        energy_per_time = h_time.abs().square().sum(dim=-1).mean()

        # Should be approximately 1.0
        assert abs(energy_per_time.item() - 1.0) < 0.1

    def test_docstring_example(self, device):
        """Test the example from the docstring"""
        channel_model = RayleighBlockFading(
            num_rx=1, num_rx_ant=2, num_tx=1, num_tx_ant=4, device=device
        )
        gen_channel = GenerateTimeChannel(
            channel_model,
            bandwidth=1e6,
            num_time_samples=100,
            l_min=-6,
            l_max=20,
            device=device,
        )
        h_time = gen_channel(batch_size=32)
        assert h_time.shape == torch.Size([32, 1, 2, 1, 4, 126, 27])


class TestApplyTimeChannel:
    """Tests for the ApplyTimeChannel class"""

    def test_output_shape(self, device, precision):
        """Verify output shape matches expected dimensions"""
        num_time_samples = 100
        l_tot = 27
        batch_size = 32
        num_tx, num_tx_ant = 1, 4
        num_rx, num_rx_ant = 1, 2

        apply_channel = ApplyTimeChannel(
            num_time_samples=num_time_samples,
            l_tot=l_tot,
            precision=precision,
            device=device,
        )

        cdtype = torch.complex64 if precision == "single" else torch.complex128
        x = torch.randn(
            batch_size,
            num_tx,
            num_tx_ant,
            num_time_samples,
            dtype=cdtype,
            device=device,
        )
        h_time = torch.randn(
            batch_size,
            num_rx,
            num_rx_ant,
            num_tx,
            num_tx_ant,
            num_time_samples + l_tot - 1,
            l_tot,
            dtype=cdtype,
            device=device,
        )

        y = apply_channel(x, h_time)

        expected_shape = torch.Size(
            [batch_size, num_rx, num_rx_ant, num_time_samples + l_tot - 1]
        )
        assert y.shape == expected_shape

    def test_output_dtype(self, device, precision):
        """Verify output dtype matches input dtype"""
        num_time_samples = 100
        l_tot = 27

        apply_channel = ApplyTimeChannel(
            num_time_samples=num_time_samples,
            l_tot=l_tot,
            precision=precision,
            device=device,
        )

        cdtype = torch.complex64 if precision == "single" else torch.complex128
        x = torch.randn(32, 1, 4, num_time_samples, dtype=cdtype, device=device)
        h_time = torch.randn(
            32,
            1,
            2,
            1,
            4,
            num_time_samples + l_tot - 1,
            l_tot,
            dtype=cdtype,
            device=device,
        )

        y = apply_channel(x, h_time)
        assert y.dtype == cdtype

    def test_output_device(self, device):
        """Verify outputs are on the correct device"""
        num_time_samples = 100
        l_tot = 27

        apply_channel = ApplyTimeChannel(
            num_time_samples=num_time_samples, l_tot=l_tot, device=device
        )

        x = torch.randn(
            32, 1, 4, num_time_samples, dtype=torch.complex64, device=device
        )
        h_time = torch.randn(
            32,
            1,
            2,
            1,
            4,
            num_time_samples + l_tot - 1,
            l_tot,
            dtype=torch.complex64,
            device=device,
        )

        y = apply_channel(x, h_time)
        assert y.device == torch.device(device)

    def test_with_noise(self, device, precision):
        """Verify that noise is added when no is provided"""
        num_time_samples = 100
        l_tot = 27

        apply_channel = ApplyTimeChannel(
            num_time_samples=num_time_samples,
            l_tot=l_tot,
            precision=precision,
            device=device,
        )

        cdtype = torch.complex64 if precision == "single" else torch.complex128
        x = torch.randn(32, 1, 4, num_time_samples, dtype=cdtype, device=device)
        h_time = torch.randn(
            32,
            1,
            2,
            1,
            4,
            num_time_samples + l_tot - 1,
            l_tot,
            dtype=cdtype,
            device=device,
        )

        # Without noise
        y_no_noise = apply_channel(x, h_time)

        # With noise
        y_with_noise = apply_channel(x, h_time, no=0.1)

        # Results should be different
        assert not torch.allclose(y_no_noise, y_with_noise)

    def test_without_noise(self, device):
        """Verify that no noise is added when no is None"""
        num_time_samples = 100
        l_tot = 27

        apply_channel = ApplyTimeChannel(
            num_time_samples=num_time_samples, l_tot=l_tot, device=device
        )

        x = torch.randn(
            32, 1, 4, num_time_samples, dtype=torch.complex64, device=device
        )
        h_time = torch.randn(
            32,
            1,
            2,
            1,
            4,
            num_time_samples + l_tot - 1,
            l_tot,
            dtype=torch.complex64,
            device=device,
        )

        # Two runs without noise should give same result
        import sionna.phy

        sionna.phy.config.seed = 42
        y1 = apply_channel(x, h_time, no=None)
        sionna.phy.config.seed = 42
        y2 = apply_channel(x, h_time, no=None)

        assert torch.allclose(y1, y2)

    def test_properties(self, device):
        """Verify properties return correct values"""
        num_time_samples = 100
        l_tot = 27

        apply_channel = ApplyTimeChannel(
            num_time_samples=num_time_samples, l_tot=l_tot, device=device
        )

        assert apply_channel.num_time_samples == num_time_samples
        assert apply_channel.l_tot == l_tot

    @pytest.mark.parametrize(
        "num_time_samples,l_tot",
        [(1, 1), (2, 3), (10, 4), (17, 7), (64, 1), (127, 16), (1000, 33)],
    )
    def test_gather_matrix(self, device, num_time_samples, l_tot):
        """Verify the gather matrix has the expected Toeplitz structure.

        ``_g`` indexes a zero-padded input, so entry ``num_time_samples``
        addresses the padding element. Row ``i`` must read
        ``[i, i-1, ..., i-l_tot+1]``, clamped to the padding index wherever the
        index would be negative.
        """
        apply_channel = ApplyTimeChannel(
            num_time_samples=num_time_samples, l_tot=l_tot, device=device
        )
        g = apply_channel._g

        assert g.dtype == torch.int64
        assert tuple(g.shape) == (num_time_samples + l_tot - 1, l_tot)

        i = torch.arange(num_time_samples + l_tot - 1, device=g.device).unsqueeze(1)
        j = torch.arange(l_tot, device=g.device).unsqueeze(0)
        expected = torch.where(i - j < 0, num_time_samples, i - j)
        expected = torch.minimum(
            expected, torch.tensor(num_time_samples, device=g.device)
        )
        assert torch.equal(g.cpu(), expected.cpu())

    def test_impulse_response(self, device):
        """Verify impulse response produces expected output"""
        num_time_samples = 10
        l_tot = 3

        apply_channel = ApplyTimeChannel(
            num_time_samples=num_time_samples,
            l_tot=l_tot,
            precision="double",
            device=device,
        )

        # Create impulse input at time 0
        x = torch.zeros(
            1, 1, 1, num_time_samples, dtype=torch.complex128, device=device
        )
        x[0, 0, 0, 0] = 1.0

        # Create simple channel with known taps
        h_time = torch.zeros(
            1,
            1,
            1,
            1,
            1,
            num_time_samples + l_tot - 1,
            l_tot,
            dtype=torch.complex128,
            device=device,
        )
        # Set channel taps [1, 2, 3] for all time steps
        h_time[..., 0] = 1.0
        h_time[..., 1] = 2.0
        h_time[..., 2] = 3.0

        y = apply_channel(x, h_time)

        # Output should start with [1, 2, 3, 0, 0, ...]
        assert torch.isclose(
            y[0, 0, 0, 0], torch.tensor(1.0, dtype=torch.complex128, device=device)
        )
        assert torch.isclose(
            y[0, 0, 0, 1], torch.tensor(2.0, dtype=torch.complex128, device=device)
        )
        assert torch.isclose(
            y[0, 0, 0, 2], torch.tensor(3.0, dtype=torch.complex128, device=device)
        )

    def test_docstring_example(self, device):
        """Test the example from the docstring"""
        num_time_samples = 100
        l_tot = 27
        batch_size = 32
        num_tx, num_tx_ant = 1, 4
        num_rx, num_rx_ant = 1, 2

        apply_channel = ApplyTimeChannel(
            num_time_samples=num_time_samples, l_tot=l_tot, device=device
        )

        x = torch.randn(
            batch_size,
            num_tx,
            num_tx_ant,
            num_time_samples,
            dtype=torch.complex64,
            device=device,
        )
        h_time = torch.randn(
            batch_size,
            num_rx,
            num_rx_ant,
            num_tx,
            num_tx_ant,
            num_time_samples + l_tot - 1,
            l_tot,
            dtype=torch.complex64,
            device=device,
        )
        y = apply_channel(x, h_time)
        assert y.shape == torch.Size([32, 1, 2, 126])


class TestTimeChannel:
    """Tests for the TimeChannel class"""

    def test_output_shape(self, device, precision):
        """Verify output shape matches expected dimensions"""
        num_rx = 2
        num_rx_ant = 4
        num_tx = 3
        num_tx_ant = 2
        batch_size = 16
        num_time_samples = 100
        l_min = -6
        l_max = 20

        channel_model = RayleighBlockFading(
            num_rx=num_rx,
            num_rx_ant=num_rx_ant,
            num_tx=num_tx,
            num_tx_ant=num_tx_ant,
            precision=precision,
            device=device,
        )

        channel = TimeChannel(
            channel_model,
            bandwidth=1e6,
            num_time_samples=num_time_samples,
            l_min=l_min,
            l_max=l_max,
            precision=precision,
            device=device,
        )

        cdtype = torch.complex64 if precision == "single" else torch.complex128
        x = torch.randn(
            batch_size,
            num_tx,
            num_tx_ant,
            num_time_samples,
            dtype=cdtype,
            device=device,
        )

        y = channel(x)

        expected_shape = torch.Size(
            [batch_size, num_rx, num_rx_ant, num_time_samples + l_max - l_min]
        )
        assert y.shape == expected_shape

    def test_return_channel(self, device, precision):
        """Verify return_channel option returns channel response"""
        num_rx = 1
        num_rx_ant = 2
        num_tx = 1
        num_tx_ant = 4
        batch_size = 32
        num_time_samples = 100
        l_min = -6
        l_max = 20
        l_tot = l_max - l_min + 1

        channel_model = RayleighBlockFading(
            num_rx=num_rx,
            num_rx_ant=num_rx_ant,
            num_tx=num_tx,
            num_tx_ant=num_tx_ant,
            precision=precision,
            device=device,
        )

        channel = TimeChannel(
            channel_model,
            bandwidth=1e6,
            num_time_samples=num_time_samples,
            l_min=l_min,
            l_max=l_max,
            return_channel=True,
            precision=precision,
            device=device,
        )

        cdtype = torch.complex64 if precision == "single" else torch.complex128
        x = torch.randn(
            batch_size,
            num_tx,
            num_tx_ant,
            num_time_samples,
            dtype=cdtype,
            device=device,
        )

        result = channel(x)

        # Should return tuple when return_channel=True
        assert isinstance(result, tuple)
        assert len(result) == 2

        y, h_time = result

        expected_y_shape = torch.Size(
            [batch_size, num_rx, num_rx_ant, num_time_samples + l_max - l_min]
        )
        expected_h_shape = torch.Size(
            [
                batch_size,
                num_rx,
                num_rx_ant,
                num_tx,
                num_tx_ant,
                num_time_samples + l_max - l_min,
                l_tot,
            ]
        )

        assert y.shape == expected_y_shape
        assert h_time.shape == expected_h_shape

    def test_output_dtype(self, device, precision):
        """Verify output dtype matches precision setting"""
        channel_model = RayleighBlockFading(
            num_rx=1,
            num_rx_ant=2,
            num_tx=1,
            num_tx_ant=4,
            precision=precision,
            device=device,
        )

        channel = TimeChannel(
            channel_model,
            bandwidth=1e6,
            num_time_samples=100,
            precision=precision,
            device=device,
        )

        cdtype = torch.complex64 if precision == "single" else torch.complex128
        x = torch.randn(32, 1, 4, 100, dtype=cdtype, device=device)

        y = channel(x)
        assert y.dtype == cdtype

    def test_output_device(self, device):
        """Verify outputs are on the correct device"""
        channel_model = RayleighBlockFading(
            num_rx=1, num_rx_ant=2, num_tx=1, num_tx_ant=4, device=device
        )

        channel = TimeChannel(
            channel_model, bandwidth=1e6, num_time_samples=100, device=device
        )

        x = torch.randn(32, 1, 4, 100, dtype=torch.complex64, device=device)
        y = channel(x)

        assert y.device == torch.device(device)

    def test_with_noise(self, device, precision):
        """Verify noise is added when no is provided"""
        channel_model = RayleighBlockFading(
            num_rx=1,
            num_rx_ant=2,
            num_tx=1,
            num_tx_ant=4,
            precision=precision,
            device=device,
        )

        channel = TimeChannel(
            channel_model,
            bandwidth=1e6,
            num_time_samples=100,
            precision=precision,
            device=device,
        )

        cdtype = torch.complex64 if precision == "single" else torch.complex128
        x = torch.randn(32, 1, 4, 100, dtype=cdtype, device=device)

        import sionna.phy

        sionna.phy.config.seed = 42
        y_no_noise = channel(x)

        sionna.phy.config.seed = 42
        y_with_noise = channel(x, no=0.1)

        # Results should be different due to noise
        assert not torch.allclose(y_no_noise, y_with_noise)

    def test_properties(self, device):
        """Verify properties return correct values"""
        channel_model = RayleighBlockFading(
            num_rx=1, num_rx_ant=2, num_tx=1, num_tx_ant=4, device=device
        )

        l_min = -6
        l_max = 20
        bandwidth = 1e6
        num_time_samples = 100

        channel = TimeChannel(
            channel_model,
            bandwidth=bandwidth,
            num_time_samples=num_time_samples,
            l_min=l_min,
            l_max=l_max,
            device=device,
        )

        assert channel.l_min == l_min
        assert channel.l_max == l_max
        assert channel.l_tot == l_max - l_min + 1
        assert channel.bandwidth == bandwidth
        assert channel.num_time_samples == num_time_samples

    def test_default_l_min_l_max(self, device):
        """Verify default l_min and l_max are computed correctly"""
        channel_model = RayleighBlockFading(
            num_rx=1, num_rx_ant=2, num_tx=1, num_tx_ant=4, device=device
        )

        bandwidth = 20e6
        maximum_delay_spread = 3e-6

        channel = TimeChannel(
            channel_model,
            bandwidth=bandwidth,
            num_time_samples=100,
            maximum_delay_spread=maximum_delay_spread,
            device=device,
        )

        # Default l_min should be -6
        assert channel.l_min == -6
        # Default l_max should be ceil(bandwidth * maximum_delay_spread) + 6
        import math

        expected_l_max = int(math.ceil(maximum_delay_spread * bandwidth) + 6)
        assert channel.l_max == expected_l_max

    def test_docstring_example(self, device):
        """Test the example from the docstring"""
        channel_model = RayleighBlockFading(
            num_rx=1, num_rx_ant=2, num_tx=1, num_tx_ant=4, device=device
        )
        channel = TimeChannel(
            channel_model,
            bandwidth=1e6,
            num_time_samples=100,
            l_min=-6,
            l_max=20,
            return_channel=True,
            device=device,
        )

        x = torch.randn(32, 1, 4, 100, dtype=torch.complex64, device=device)
        y, h_time = channel(x)
        assert y.shape == torch.Size([32, 1, 2, 126])
        assert h_time.shape == torch.Size([32, 1, 2, 1, 4, 126, 27])


class TestTimeChannelCompiled:
    """Tests for TimeChannel with torch.compile"""

    def test_compiled_apply_time_channel(self, device):
        """Verify ApplyTimeChannel works with torch.compile"""
        num_time_samples = 100
        l_tot = 27

        apply_channel = ApplyTimeChannel(
            num_time_samples=num_time_samples, l_tot=l_tot, device=device
        )

        @torch.compile
        def apply_func(x, h_time):
            return apply_channel(x, h_time)

        x = torch.randn(8, 1, 4, num_time_samples, dtype=torch.complex64, device=device)
        h_time = torch.randn(
            8,
            1,
            2,
            1,
            4,
            num_time_samples + l_tot - 1,
            l_tot,
            dtype=torch.complex64,
            device=device,
        )

        y = apply_func(x, h_time)
        assert y.shape == torch.Size([8, 1, 2, 126])

    def test_compiled_time_channel(self, device):
        """Verify TimeChannel works with torch.compile"""
        channel_model = RayleighBlockFading(
            num_rx=1, num_rx_ant=2, num_tx=1, num_tx_ant=4, device=device
        )
        channel = TimeChannel(
            channel_model,
            bandwidth=1e6,
            num_time_samples=100,
            l_min=-6,
            l_max=20,
            device=device,
        )

        @torch.compile
        def channel_func(x):
            return channel(x)

        x = torch.randn(8, 1, 4, 100, dtype=torch.complex64, device=device)
        y = channel_func(x)

        assert y.shape == torch.Size([8, 1, 2, 126])
