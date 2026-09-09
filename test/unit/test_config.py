#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
import copy

import pytest
import torch
import numpy as np
from sionna.phy import config, dtypes
from sionna.phy.config import Config


class TestConfigSingleton:
    """Tests for Config singleton pattern."""

    def test_singleton_same_instance(self):
        """Test that Config() always returns the same instance."""
        config1 = Config()
        config2 = Config()
        assert config1 is config2

    def test_global_config_is_singleton(self):
        """Test that the global config is the singleton instance."""
        assert config is Config()


class TestConfigPrecision:
    """Tests for precision settings."""

    @pytest.fixture(autouse=True)
    def restore_precision(self):
        """`config` is a singleton, so these assignments outlive the test."""
        original = config.precision
        yield
        config.precision = original

    def test_precision_single(self):
        """Test single precision settings."""
        config.precision = "single"
        assert config.precision == "single"
        assert config.dtype == torch.float32
        assert config.cdtype == torch.complex64
        assert config.np_dtype == np.float32
        assert config.np_cdtype == np.complex64

    def test_precision_double(self):
        """Test double precision settings."""
        config.precision = "double"
        assert config.precision == "double"
        assert config.dtype == torch.float64
        assert config.cdtype == torch.complex128
        assert config.np_dtype == np.float64
        assert config.np_cdtype == np.complex128

    def test_precision_invalid(self):
        """Test that invalid precision raises ValueError."""
        with pytest.raises(ValueError, match="Precision must be"):
            config.precision = "invalid"

    def test_precision_parametrized(self, precision):
        """Test precision with parametrized fixture."""
        config.precision = precision
        assert config.np_dtype == dtypes[precision]["np"]["dtype"]
        assert config.np_cdtype == dtypes[precision]["np"]["cdtype"]
        assert config.dtype == dtypes[precision]["torch"]["dtype"]
        assert config.cdtype == dtypes[precision]["torch"]["cdtype"]


class TestConfigDevice:
    """Tests for device settings."""

    def test_device_setter(self, device):
        """Test that the device is set correctly."""
        config.device = device
        assert config.device == device

    def test_device_setter_invalid(self):
        """Test that setting an invalid device raises an error."""
        with pytest.raises(ValueError):
            config.device = "invalid_device"

    def test_available_devices_contains_cpu(self):
        """Test that CPU is always in available devices."""
        assert "cpu" in config.available_devices

    def test_available_devices_cuda_format(self):
        """Test that CUDA devices have correct format."""
        for device in config.available_devices:
            if device != "cpu":
                assert device.startswith("cuda:")

    def test_device_default_selection(self):
        """Test that default device selection works."""
        config.device = None  # Reset to default
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            assert config.device == "cuda:0"
        else:
            assert config.device == "cpu"


class TestConfigSeed:
    """Tests for seed and RNG settings."""

    def test_seed_reproducibility(self):
        """Test that the seed is set correctly for all random number generators."""
        config.seed = 12345

        # Sample random values
        py_rnd = config.py_rng.random()
        np_rnd = config.np_rng.random()
        torch_rnds = {}
        for device in config.available_devices:
            torch_rnds[device] = torch.rand(
                1, device=device, generator=config.torch_rng(device)
            ).item()

        # Reset seed to same value
        config.seed = 12345

        # Values should be identical
        assert py_rnd == config.py_rng.random()
        assert np_rnd == config.np_rng.random()
        for device in config.available_devices:
            assert (
                torch.rand(1, device=device, generator=config.torch_rng(device)).item()
                == torch_rnds[device]
            )

    def test_different_seeds_produce_different_values(self):
        """Test that different seeds produce different random values."""
        config.seed = 111
        val1 = config.np_rng.random()

        config.seed = 222
        val2 = config.np_rng.random()

        assert val1 != val2

    def test_seed_none_allows_random_values(self):
        """Test that seed=None allows non-reproducible random values."""
        config.seed = None

        # Sample multiple times - values should generally differ
        # (statistically extremely unlikely to be the same)
        values = [config.np_rng.random() for _ in range(10)]
        assert len(set(values)) > 1  # At least some values should differ

    @pytest.mark.parametrize(
        "invalid_seed, exception",
        [
            ("abc", TypeError),
            (1.5, TypeError),
            (True, TypeError),
            (-1, ValueError),
            (2**64, ValueError),
        ],
    )
    def test_invalid_seed_assignment_is_transactional(
        self, invalid_seed, exception
    ):
        """Rejected seeds must not mutate any existing RNG stream."""
        config.seed = 123
        py_rng = config.py_rng
        np_rng = config.np_rng
        torch_rngs = {
            device: config.torch_rng(device) for device in config.available_devices
        }

        # Advance every explicit stream before taking its state snapshot.
        py_rng.random()
        np_rng.random()
        for device, generator in torch_rngs.items():
            torch.rand(1, device=device, generator=generator)

        py_state = py_rng.getstate()
        np_state = copy.deepcopy(np_rng.bit_generator.state)
        torch_states = {
            device: generator.get_state().clone()
            for device, generator in torch_rngs.items()
        }
        default_cpu_state = torch.default_generator.get_state().clone()
        default_cuda_states = [
            generator.get_state().clone()
            for generator in torch.cuda.default_generators
        ]

        with pytest.raises(exception):
            config.seed = invalid_seed

        assert config.seed == 123
        assert config.py_rng is py_rng
        assert config.np_rng is np_rng
        assert config.py_rng.getstate() == py_state
        np.testing.assert_equal(config.np_rng.bit_generator.state, np_state)
        for device, generator in torch_rngs.items():
            assert config.torch_rng(device) is generator
            assert torch.equal(generator.get_state(), torch_states[device])
        assert torch.equal(torch.default_generator.get_state(), default_cpu_state)
        for generator, state in zip(
            torch.cuda.default_generators, default_cuda_states
        ):
            assert torch.equal(generator.get_state(), state)

    @pytest.mark.parametrize("seed", [np.int32(321), np.int64(321)])
    def test_numpy_integer_seed(self, seed):
        """Preserve support for NumPy integer seeds."""
        config.seed = seed
        values = config.np_rng.random(4).tolist()

        config.seed = 321
        assert config.np_rng.random(4).tolist() == values

    def test_maximum_seed_wraps_device_offsets(self):
        """The full uint64 seed range remains valid on multiple devices."""
        seed = 2**64 - 1
        config.seed = seed

        for offset, device in enumerate(config.available_devices):
            expected = (seed + offset) % 2**64
            assert config.torch_rng(device).initial_seed() == expected
            if device == "cpu":
                assert torch.default_generator.initial_seed() == expected
            else:
                device_idx = int(device.split(":")[1])
                assert (
                    torch.cuda.default_generators[device_idx].initial_seed()
                    == expected
                )

    def test_device_specific_seed_offsets(self):
        """Test that different devices get different random streams with same base seed."""
        if len(config.available_devices) < 2:
            pytest.skip("Need at least 2 devices to test device-specific seeds")

        config.seed = 42

        # Get first random value from each device
        device_values = {}
        for device in config.available_devices:
            device_values[device] = torch.rand(
                1, device=device, generator=config.torch_rng(device)
            ).item()

        # Values from different devices should be different
        # (since each device gets seed + device_index)
        unique_values = set(device_values.values())
        assert len(unique_values) == len(
            device_values
        ), "Each device should produce different random streams"

    def test_torch_rng_per_device(self, device):
        """Test that torch_rng returns correct generator for each device."""
        config.seed = 123

        rng = config.torch_rng(device)
        assert rng.device.type == device.split(":")[0]

        # Should be reproducible
        val1 = torch.rand(1, device=device, generator=config.torch_rng(device)).item()
        config.seed = 123
        val2 = torch.rand(1, device=device, generator=config.torch_rng(device)).item()
        assert val1 == val2

    def test_torch_rng_default_device(self):
        """Test that torch_rng() without argument uses config.device."""
        config.seed = 456
        config.device = "cpu"

        rng = config.torch_rng()
        assert rng.device.type == "cpu"

    def test_default_generator_seeding(self):
        """Test that default generators are seeded for compiled mode compatibility."""
        config.seed = 789

        # The default CPU generator should be seeded
        val1 = torch.rand(1).item()

        config.seed = 789
        val2 = torch.rand(1).item()

        # Default generator should produce same values with same seed
        assert val1 == val2

    def test_py_rng_reproducibility(self):
        """Test Python RNG reproducibility."""
        config.seed = 100
        values1 = [config.py_rng.random() for _ in range(5)]

        config.seed = 100
        values2 = [config.py_rng.random() for _ in range(5)]

        assert values1 == values2

    def test_np_rng_reproducibility(self):
        """Test NumPy RNG reproducibility."""
        config.seed = 200
        values1 = config.np_rng.random(10).tolist()

        config.seed = 200
        values2 = config.np_rng.random(10).tolist()

        assert values1 == values2

    def test_rng_independence(self):
        """Test that different RNG types are independent."""
        config.seed = 300

        # Sample from one RNG
        _ = config.py_rng.random()
        _ = config.py_rng.random()

        # NumPy RNG should still be at its initial state
        config.seed = 300
        np_val1 = config.np_rng.random()

        config.seed = 300
        _ = config.py_rng.random()  # Advance py_rng
        np_val2 = config.np_rng.random()

        assert np_val1 == np_val2


class TestConfigIntegration:
    """Integration tests for Config."""

    def test_seed_reset_reinitializes_all_rngs(self):
        """Test that setting seed reinitializes all RNGs."""
        config.seed = 500

        # Advance all RNGs
        _ = config.py_rng.random()
        _ = config.np_rng.random()
        for device in config.available_devices:
            _ = torch.rand(1, device=device, generator=config.torch_rng(device))

        # Reset seed
        config.seed = 500

        # All RNGs should be back to initial state - sample values
        py_val = config.py_rng.random()
        np_val = config.np_rng.random()
        torch_vals = {
            device: torch.rand(
                1, device=device, generator=config.torch_rng(device)
            ).item()
            for device in config.available_devices
        }

        # Reset again and verify all produce the same values
        config.seed = 500
        assert config.py_rng.random() == py_val
        assert config.np_rng.random() == np_val
        for device in config.available_devices:
            assert (
                torch.rand(1, device=device, generator=config.torch_rng(device)).item()
                == torch_vals[device]
            )