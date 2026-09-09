#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Tests for the CIRDataset channel model"""

import torch

from sionna.phy import config
from sionna.phy.channel import CIRDataset, TimeChannel
from sionna.phy.channel.cir_dataset import _CIRIterableDataset


class SimpleGenerator:
    """Simple generator for testing that yields random CIR samples."""

    def __init__(
        self,
        num_rx: int,
        num_rx_ant: int,
        num_tx: int,
        num_tx_ant: int,
        num_paths: int,
        num_time_steps: int,
        num_samples: int = 100,
        cdtype: torch.dtype = torch.complex64,
        dtype: torch.dtype = torch.float32,
    ):
        self.num_rx = num_rx
        self.num_rx_ant = num_rx_ant
        self.num_tx = num_tx
        self.num_tx_ant = num_tx_ant
        self.num_paths = num_paths
        self.num_time_steps = num_time_steps
        self.num_samples = num_samples
        self.cdtype = cdtype
        self.dtype = dtype

    def __call__(self):
        for _ in range(self.num_samples):
            a = torch.randn(
                self.num_rx,
                self.num_rx_ant,
                self.num_tx,
                self.num_tx_ant,
                self.num_paths,
                self.num_time_steps,
                dtype=self.cdtype,
            )
            tau = (
                torch.rand(self.num_rx, self.num_tx, self.num_paths, dtype=self.dtype)
                * 1e-6
            )
            yield a, tau


class TestCIRDataset:
    """Tests for the CIRDataset class"""

    def test_output_shape(self, device, precision):
        """Verify output shapes for path coefficients and delays"""
        num_rx = 2
        num_rx_ant = 4
        num_tx = 3
        num_tx_ant = 2
        num_paths = 5
        num_time_steps = 10
        batch_size = 16

        cdtype = torch.complex64 if precision == "single" else torch.complex128
        dtype = torch.float32 if precision == "single" else torch.float64

        generator = SimpleGenerator(
            num_rx,
            num_rx_ant,
            num_tx,
            num_tx_ant,
            num_paths,
            num_time_steps,
            num_samples=200,
            cdtype=cdtype,
            dtype=dtype,
        )

        channel = CIRDataset(
            generator,
            batch_size,
            num_rx,
            num_rx_ant,
            num_tx,
            num_tx_ant,
            num_paths,
            num_time_steps,
            precision=precision,
            device=device,
        )

        a, tau = channel()

        expected_a_shape = torch.Size(
            [
                batch_size,
                num_rx,
                num_rx_ant,
                num_tx,
                num_tx_ant,
                num_paths,
                num_time_steps,
            ]
        )
        expected_tau_shape = torch.Size([batch_size, num_rx, num_tx, num_paths])

        assert a.shape == expected_a_shape
        assert tau.shape == expected_tau_shape

    def test_output_dtype(self, device, precision):
        """Verify output dtypes match precision setting"""
        num_rx = 1
        num_rx_ant = 2
        num_tx = 1
        num_tx_ant = 4
        num_paths = 3
        num_time_steps = 8
        batch_size = 8

        generator = SimpleGenerator(
            num_rx,
            num_rx_ant,
            num_tx,
            num_tx_ant,
            num_paths,
            num_time_steps,
            num_samples=100,
        )

        channel = CIRDataset(
            generator,
            batch_size,
            num_rx,
            num_rx_ant,
            num_tx,
            num_tx_ant,
            num_paths,
            num_time_steps,
            precision=precision,
            device=device,
        )

        a, tau = channel()

        expected_cdtype = torch.complex64 if precision == "single" else torch.complex128
        expected_dtype = torch.float32 if precision == "single" else torch.float64

        assert a.dtype == expected_cdtype
        assert tau.dtype == expected_dtype

    def test_output_device(self, device):
        """Verify outputs are on the correct device"""
        num_rx = 1
        num_rx_ant = 2
        num_tx = 1
        num_tx_ant = 4
        num_paths = 3
        num_time_steps = 8
        batch_size = 8

        generator = SimpleGenerator(
            num_rx,
            num_rx_ant,
            num_tx,
            num_tx_ant,
            num_paths,
            num_time_steps,
            num_samples=100,
        )

        channel = CIRDataset(
            generator,
            batch_size,
            num_rx,
            num_rx_ant,
            num_tx,
            num_tx_ant,
            num_paths,
            num_time_steps,
            device=device,
        )

        a, tau = channel()

        assert a.device == torch.device(device)
        assert tau.device == torch.device(device)

    def test_batch_size_property(self, device):
        """Verify dynamic batch_size changes work correctly"""
        num_rx = 1
        num_rx_ant = 2
        num_tx = 1
        num_tx_ant = 4
        num_paths = 3
        num_time_steps = 8
        initial_batch_size = 8
        new_batch_size = 16

        generator = SimpleGenerator(
            num_rx,
            num_rx_ant,
            num_tx,
            num_tx_ant,
            num_paths,
            num_time_steps,
            num_samples=200,
        )

        channel = CIRDataset(
            generator,
            initial_batch_size,
            num_rx,
            num_rx_ant,
            num_tx,
            num_tx_ant,
            num_paths,
            num_time_steps,
            device=device,
        )

        # Check initial batch size
        assert channel.batch_size == initial_batch_size
        a, _ = channel()
        assert a.shape[0] == initial_batch_size

        # Change batch size
        channel.batch_size = new_batch_size
        assert channel.batch_size == new_batch_size
        a, _ = channel()
        assert a.shape[0] == new_batch_size

    def test_multiple_iterations(self, device):
        """Verify dataset can be sampled multiple times"""
        num_rx = 1
        num_rx_ant = 2
        num_tx = 1
        num_tx_ant = 4
        num_paths = 3
        num_time_steps = 8
        batch_size = 8

        generator = SimpleGenerator(
            num_rx,
            num_rx_ant,
            num_tx,
            num_tx_ant,
            num_paths,
            num_time_steps,
            num_samples=100,
        )

        channel = CIRDataset(
            generator,
            batch_size,
            num_rx,
            num_rx_ant,
            num_tx,
            num_tx_ant,
            num_paths,
            num_time_steps,
            device=device,
        )

        # Sample multiple times - should not raise
        for _ in range(20):
            a, _ = channel()
            assert a.shape[0] == batch_size

    def test_properties(self, device):
        """Verify that all properties return correct values"""
        num_rx = 2
        num_rx_ant = 4
        num_tx = 3
        num_tx_ant = 5
        num_paths = 6
        num_time_steps = 10
        batch_size = 8

        generator = SimpleGenerator(
            num_rx,
            num_rx_ant,
            num_tx,
            num_tx_ant,
            num_paths,
            num_time_steps,
        )

        channel = CIRDataset(
            generator,
            batch_size,
            num_rx,
            num_rx_ant,
            num_tx,
            num_tx_ant,
            num_paths,
            num_time_steps,
            device=device,
        )

        assert channel.num_rx == num_rx
        assert channel.num_rx_ant == num_rx_ant
        assert channel.num_tx == num_tx
        assert channel.num_tx_ant == num_tx_ant
        assert channel.num_paths == num_paths
        assert channel.num_time_steps == num_time_steps
        assert channel.batch_size == batch_size

    def test_call_parameters_ignored(self, device):
        """Verify that call parameters are accepted but ignored"""
        num_rx = 1
        num_rx_ant = 2
        num_tx = 1
        num_tx_ant = 4
        num_paths = 3
        num_time_steps = 8
        batch_size = 8

        generator = SimpleGenerator(
            num_rx,
            num_rx_ant,
            num_tx,
            num_tx_ant,
            num_paths,
            num_time_steps,
            num_samples=100,
        )

        channel = CIRDataset(
            generator,
            batch_size,
            num_rx,
            num_rx_ant,
            num_tx,
            num_tx_ant,
            num_paths,
            num_time_steps,
            device=device,
        )

        # Should not raise an error when parameters are provided
        a1, _ = channel(batch_size=64, num_time_steps=100, sampling_frequency=15e3)
        a2, _ = channel()

        # Shapes should match the configured values, not the call parameters
        assert a1.shape[0] == batch_size
        assert a2.shape[0] == batch_size
        assert a1.shape[-1] == num_time_steps
        assert a2.shape[-1] == num_time_steps


class TestCIRDatasetWithTimeChannel:
    """Integration tests for CIRDataset with TimeChannel"""

    def test_time_channel_output_shape(self, device, precision):
        """Verify TimeChannel output shapes when using CIRDataset"""
        num_rx = 1
        num_rx_ant = 2
        num_tx = 1
        num_tx_ant = 4
        num_paths = 3
        batch_size = 32
        num_time_samples = 100
        l_min = -6
        l_max = 20
        l_tot = l_max - l_min + 1
        num_time_steps = num_time_samples + l_tot - 1

        cdtype = torch.complex64 if precision == "single" else torch.complex128
        dtype = torch.float32 if precision == "single" else torch.float64

        generator = SimpleGenerator(
            num_rx,
            num_rx_ant,
            num_tx,
            num_tx_ant,
            num_paths,
            num_time_steps,
            num_samples=200,
            cdtype=cdtype,
            dtype=dtype,
        )

        channel_model = CIRDataset(
            generator,
            batch_size,
            num_rx,
            num_rx_ant,
            num_tx,
            num_tx_ant,
            num_paths,
            num_time_steps,
            precision=precision,
            device=device,
        )

        time_channel = TimeChannel(
            channel_model,
            bandwidth=1e6,
            num_time_samples=num_time_samples,
            l_min=l_min,
            l_max=l_max,
            return_channel=True,
            precision=precision,
            device=device,
        )

        x = torch.randn(
            batch_size,
            num_tx,
            num_tx_ant,
            num_time_samples,
            dtype=cdtype,
            device=device,
        )
        y, h_time = time_channel(x)

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

    def test_time_channel_output_dtype(self, device, precision):
        """Verify TimeChannel output dtypes match precision"""
        num_rx = 1
        num_rx_ant = 2
        num_tx = 1
        num_tx_ant = 4
        num_paths = 3
        batch_size = 32
        num_time_samples = 100
        l_min = -6
        l_max = 20
        l_tot = l_max - l_min + 1
        num_time_steps = num_time_samples + l_tot - 1

        cdtype = torch.complex64 if precision == "single" else torch.complex128
        dtype = torch.float32 if precision == "single" else torch.float64

        generator = SimpleGenerator(
            num_rx,
            num_rx_ant,
            num_tx,
            num_tx_ant,
            num_paths,
            num_time_steps,
            num_samples=200,
            cdtype=cdtype,
            dtype=dtype,
        )

        channel_model = CIRDataset(
            generator,
            batch_size,
            num_rx,
            num_rx_ant,
            num_tx,
            num_tx_ant,
            num_paths,
            num_time_steps,
            precision=precision,
            device=device,
        )

        time_channel = TimeChannel(
            channel_model,
            bandwidth=1e6,
            num_time_samples=num_time_samples,
            l_min=l_min,
            l_max=l_max,
            return_channel=True,
            precision=precision,
            device=device,
        )

        x = torch.randn(
            batch_size,
            num_tx,
            num_tx_ant,
            num_time_samples,
            dtype=cdtype,
            device=device,
        )
        y, h_time = time_channel(x)

        assert y.dtype == cdtype
        assert h_time.dtype == cdtype


class TestCIRDatasetRNG:
    """CIRDataset shuffle order must follow sionna.phy.config.seed."""

    def test_shuffle_reproducible_under_config_seed(self, device):
        """Resetting config.seed reproduces the shuffle-buffer order."""

        class IndexedGenerator:
            """Yields identifiable samples so shuffle order is observable."""

            def __init__(self, num_samples: int = 8):
                self.num_samples = num_samples

            def __call__(self):
                for i in range(self.num_samples):
                    a = torch.full((1, 1, 1, 1, 1, 1), float(i), dtype=torch.complex64)
                    tau = torch.full((1, 1, 1), float(i), dtype=torch.float32)
                    yield a, tau

        def collect_order():
            dataset = _CIRIterableDataset(IndexedGenerator(), buffer_size=8)
            return [int(a.real.item()) for a, _ in dataset]

        config.seed = 123
        first = collect_order()
        config.seed = 123
        second = collect_order()
        config.seed = 456
        third = collect_order()

        assert first == second
        assert first != third
