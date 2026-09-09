#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Unit tests for ResourceGrid and related classes"""

import numpy as np
import pytest
import torch

from sionna.phy.mapping import QAMSource
from sionna.phy.mimo import StreamManagement
from sionna.phy.ofdm import (
    OFDMDemodulator,
    OFDMModulator,
    RemoveNulledSubcarriers,
    ResourceGrid,
    ResourceGridDemapper,
    ResourceGridMapper,
)


class TestResourceGrid:
    """Tests for the ResourceGrid class"""

    def test_basic_creation(self, device, precision):
        """Test basic ResourceGrid creation"""
        rg = ResourceGrid(
            num_ofdm_symbols=14,
            fft_size=64,
            subcarrier_spacing=30e3,
            precision=precision,
            device=device,
        )

        assert rg.num_ofdm_symbols == 14
        assert rg.fft_size == 64
        assert rg.subcarrier_spacing == 30e3
        assert rg.num_tx == 1
        assert rg.num_streams_per_tx == 1
        assert rg.cyclic_prefix_length == 0

    def test_with_guard_carriers(self, device, precision):
        """Test ResourceGrid with guard carriers"""
        rg = ResourceGrid(
            num_ofdm_symbols=14,
            fft_size=64,
            subcarrier_spacing=30e3,
            num_guard_carriers=(5, 5),
            precision=precision,
            device=device,
        )

        assert rg.num_effective_subcarriers == 64 - 10
        assert len(rg.effective_subcarrier_ind) == 54

    def test_with_dc_null(self, device, precision):
        """Test ResourceGrid with DC nulling"""
        rg = ResourceGrid(
            num_ofdm_symbols=14,
            fft_size=64,
            subcarrier_spacing=30e3,
            dc_null=True,
            precision=precision,
            device=device,
        )

        assert rg.num_effective_subcarriers == 63
        assert rg.dc_null is True

    def test_with_pilots(self, device, precision):
        """Test ResourceGrid with pilot pattern"""
        rg = ResourceGrid(
            num_ofdm_symbols=14,
            fft_size=64,
            subcarrier_spacing=30e3,
            num_tx=2,
            num_streams_per_tx=2,
            pilot_pattern="kronecker",
            pilot_ofdm_symbol_indices=[2, 11],
            precision=precision,
            device=device,
        )

        assert rg.num_pilot_symbols > 0
        assert rg.num_data_symbols < rg.num_effective_subcarriers * 14

    def test_properties(self, device, precision):
        """Test computed properties"""
        rg = ResourceGrid(
            num_ofdm_symbols=14,
            fft_size=64,
            subcarrier_spacing=30e3,
            cyclic_prefix_length=16,
            precision=precision,
            device=device,
        )

        assert rg.num_resource_elements == 14 * 64
        assert rg.bandwidth == 64 * 30e3
        assert rg.num_time_samples == (64 + 16) * 14
        assert rg.ofdm_symbol_duration == (1 + 16/64) / 30e3

    def test_build_type_grid(self, device, precision):
        """Test build_type_grid returns correct tensor"""
        rg = ResourceGrid(
            num_ofdm_symbols=14,
            fft_size=64,
            subcarrier_spacing=30e3,
            num_guard_carriers=(5, 5),
            dc_null=True,
            precision=precision,
            device=device,
        )

        type_grid = rg.build_type_grid()
        assert type_grid.shape == torch.Size([1, 1, 14, 64])

        # Check guard carriers are marked correctly
        assert torch.all(type_grid[..., :5] == 2)  # Left guards
        assert torch.all(type_grid[..., -5:] == 2)  # Right guards

    @pytest.mark.parametrize(
        ("fft_size", "guards", "expected_indices", "expected_types"),
        [
            (8, (4, 0), [5, 6, 7], [2, 2, 2, 2, 3, 0, 0, 0]),
            (8, (0, 3), [0, 1, 2, 3], [0, 0, 0, 0, 3, 2, 2, 2]),
            (7, (3, 0), [4, 5, 6], [2, 2, 2, 3, 0, 0, 0]),
            (7, (0, 3), [0, 1, 2], [0, 0, 0, 3, 2, 2, 2]),
        ],
    )
    def test_guard_carriers_adjacent_to_dc(
        self,
        fft_size,
        guards,
        expected_indices,
        expected_types,
        device,
        precision,
    ):
        """Guard regions may end immediately next to the nulled DC carrier."""
        rg = ResourceGrid(
            num_ofdm_symbols=1,
            fft_size=fft_size,
            subcarrier_spacing=30e3,
            num_guard_carriers=guards,
            dc_null=True,
            precision=precision,
            device=device,
        )

        np.testing.assert_array_equal(
            rg.effective_subcarrier_ind, expected_indices
        )
        assert rg.num_effective_subcarriers == len(expected_indices)
        torch.testing.assert_close(
            rg.build_type_grid()[0, 0, 0],
            torch.tensor(expected_types, dtype=torch.int32, device=device),
        )

    @pytest.mark.parametrize("guards", [(5, 0), (6, 0), (0, 4), (0, 5)])
    def test_guard_carriers_cannot_include_dc(
        self, guards, device, precision
    ):
        """A separately nulled DC carrier cannot also be a guard carrier."""
        with pytest.raises(ValueError, match="include the DC subcarrier"):
            ResourceGrid(
                num_ofdm_symbols=1,
                fft_size=8,
                subcarrier_spacing=30e3,
                num_guard_carriers=guards,
                dc_null=True,
                precision=precision,
                device=device,
            )

    def test_resource_grid_requires_effective_subcarrier(
        self, device, precision
    ):
        """Validation runs before pilot-pattern construction."""
        with pytest.raises(ValueError, match="at least one effective"):
            ResourceGrid(
                num_ofdm_symbols=1,
                fft_size=8,
                subcarrier_spacing=30e3,
                num_guard_carriers=(4, 3),
                dc_null=True,
                precision=precision,
                device=device,
            )

    def test_invalid_params(self, device, precision):
        """Test that invalid parameters raise errors"""
        with pytest.raises(ValueError):
            ResourceGrid(
                num_ofdm_symbols=0,
                fft_size=64,
                subcarrier_spacing=30e3,
                precision=precision,
                device=device,
            )

        with pytest.raises(ValueError):
            ResourceGrid(
                num_ofdm_symbols=14,
                fft_size=64,
                subcarrier_spacing=30e3,
                cyclic_prefix_length=100,  # Larger than fft_size
                precision=precision,
                device=device,
            )


class TestResourceGridMapper:
    """Tests for the ResourceGridMapper class"""

    def test_basic_mapping(self, device, precision):
        """Test basic data mapping to resource grid"""
        rg = ResourceGrid(
            num_ofdm_symbols=14,
            fft_size=64,
            subcarrier_spacing=30e3,
            precision=precision,
            device=device,
        )
        mapper = ResourceGridMapper(rg, precision=precision, device=device)
        qam_source = QAMSource(4, precision=precision, device=device)

        x = qam_source([32, 1, 1, rg.num_data_symbols])
        rg_mapped = mapper(x)

        assert rg_mapped.shape == torch.Size([32, 1, 1, 14, 64])

    def test_mapping_with_pilots(self, device, precision):
        """Test mapping with pilot symbols"""
        rg = ResourceGrid(
            num_ofdm_symbols=14,
            fft_size=64,
            subcarrier_spacing=30e3,
            num_tx=2,
            num_streams_per_tx=2,
            pilot_pattern="kronecker",
            pilot_ofdm_symbol_indices=[2, 11],
            precision=precision,
            device=device,
        )
        mapper = ResourceGridMapper(rg, precision=precision, device=device)
        qam_source = QAMSource(4, precision=precision, device=device)

        x = qam_source([32, 2, 2, rg.num_data_symbols])
        rg_mapped = mapper(x)

        assert rg_mapped.shape == torch.Size([32, 2, 2, 14, 64])


class TestResourceGridDemapper:
    """Tests for the ResourceGridDemapper class"""

    def test_various_params(self, device, precision):
        """Test demapper with various parameters (data_dim dimension omitted)"""
        fft_size = 72
        qam_source = QAMSource(4, precision=precision, device=device)

        for cp_length in [0, 1, 12, fft_size]:
            for num_tx in [1, 2, 3, 8]:
                for num_streams_per_tx in [1, 2, 3]:
                    rg = ResourceGrid(
                        num_ofdm_symbols=14,
                        fft_size=fft_size,
                        subcarrier_spacing=30e3,
                        num_tx=num_tx,
                        num_streams_per_tx=num_streams_per_tx,
                        cyclic_prefix_length=cp_length,
                        precision=precision,
                        device=device,
                    )
                    sm = StreamManagement(
                        np.ones([1, rg.num_tx]), rg.num_streams_per_tx
                    )
                    rg_mapper = ResourceGridMapper(
                        rg, precision=precision, device=device
                    )
                    rg_demapper = ResourceGridDemapper(
                        rg, sm, precision=precision, device=device
                    )
                    modulator = OFDMModulator(
                        rg.cyclic_prefix_length, precision=precision, device=device
                    )
                    demodulator = OFDMDemodulator(
                        rg.fft_size, 0, rg.cyclic_prefix_length,
                        precision=precision, device=device
                    )

                    x = qam_source([128, rg.num_tx, rg.num_streams_per_tx,
                                   rg.num_data_symbols])
                    x_rg = rg_mapper(x)
                    x_time = modulator(x_rg)
                    y = demodulator(x_time)
                    x_hat = rg_demapper(y)

                    err = torch.max(torch.abs(x - x_hat)).item()
                    assert err < 1e-5

    def test_data_dim(self, device, precision):
        """Test demapper with data_dim dimension provided"""
        fft_size = 72
        qam_source = QAMSource(4, precision=precision, device=device)

        for cp_length in [0, 12]:
            for num_tx in [1, 2]:
                for num_streams_per_tx in [1, 2]:
                    rg = ResourceGrid(
                        num_ofdm_symbols=14,
                        fft_size=fft_size,
                        subcarrier_spacing=30e3,
                        num_tx=num_tx,
                        num_streams_per_tx=num_streams_per_tx,
                        cyclic_prefix_length=cp_length,
                        precision=precision,
                        device=device,
                    )
                    sm = StreamManagement(
                        np.ones([1, rg.num_tx]), rg.num_streams_per_tx
                    )
                    rg_mapper = ResourceGridMapper(
                        rg, precision=precision, device=device
                    )
                    rg_demapper = ResourceGridDemapper(
                        rg, sm, precision=precision, device=device
                    )
                    modulator = OFDMModulator(
                        rg.cyclic_prefix_length, precision=precision, device=device
                    )
                    demodulator = OFDMDemodulator(
                        rg.fft_size, 0, rg.cyclic_prefix_length,
                        precision=precision, device=device
                    )

                    x = qam_source([64, rg.num_tx, rg.num_streams_per_tx,
                                   rg.num_data_symbols])
                    x_rg = rg_mapper(x)
                    x_time = modulator(x_rg)
                    y = demodulator(x_time)

                    # Stack inputs to simulate the data_dim dimension
                    y = torch.stack([y, y, y], dim=-1)
                    x_hat = rg_demapper(y)
                    x_stacked = torch.stack([x, x, x], dim=-1)

                    err = torch.max(torch.abs(x_stacked - x_hat)).item()
                    assert err < 1e-5


class TestRemoveNulledSubcarriers:
    """Tests for the RemoveNulledSubcarriers class"""

    def test_remove_guard_carriers(self, device, precision):
        """Test removal of guard carriers"""
        rg = ResourceGrid(
            num_ofdm_symbols=14,
            fft_size=64,
            subcarrier_spacing=30e3,
            num_guard_carriers=(5, 5),
            precision=precision,
            device=device,
        )
        remover = RemoveNulledSubcarriers(rg, precision=precision, device=device)

        x = torch.randn(
            32, 1, 1, 14, 64,
            dtype=torch.complex64 if precision == "single" else torch.complex128,
            device=device
        )
        y = remover(x)

        assert y.shape == torch.Size([32, 1, 1, 14, 54])

    def test_remove_dc_carrier(self, device, precision):
        """Test removal of DC carrier"""
        rg = ResourceGrid(
            num_ofdm_symbols=14,
            fft_size=64,
            subcarrier_spacing=30e3,
            dc_null=True,
            precision=precision,
            device=device,
        )
        remover = RemoveNulledSubcarriers(rg, precision=precision, device=device)

        x = torch.randn(
            32, 1, 1, 14, 64,
            dtype=torch.complex64 if precision == "single" else torch.complex128,
            device=device
        )
        y = remover(x)

        assert y.shape == torch.Size([32, 1, 1, 14, 63])

    def test_remove_all_nulled(self, device, precision):
        """Test removal of all nulled subcarriers"""
        rg = ResourceGrid(
            num_ofdm_symbols=14,
            fft_size=64,
            subcarrier_spacing=30e3,
            num_guard_carriers=(5, 5),
            dc_null=True,
            precision=precision,
            device=device,
        )
        remover = RemoveNulledSubcarriers(rg, precision=precision, device=device)

        x = torch.randn(
            32, 1, 1, 14, 64,
            dtype=torch.complex64 if precision == "single" else torch.complex128,
            device=device
        )
        y = remover(x)

        assert y.shape == torch.Size([32, 1, 1, 14, 53])


class TestResourceGridCompile:
    """Tests for torch.compile compatibility"""

    def test_mapper_compile(self, device, precision, mode):
        """Test that ResourceGridMapper works with torch.compile"""
        if device == "cpu" and mode != "default":
            pytest.skip("Only default mode supported on CPU")

        rg = ResourceGrid(
            num_ofdm_symbols=14,
            fft_size=64,
            subcarrier_spacing=30e3,
            precision=precision,
            device=device,
        )
        mapper = ResourceGridMapper(rg, precision=precision, device=device)
        compiled_mapper = torch.compile(mapper, mode=mode)

        qam_source = QAMSource(4, precision=precision, device=device)
        x = qam_source([32, 1, 1, rg.num_data_symbols])

        rg_compiled = compiled_mapper(x)

        # Create fresh mapper for comparison
        mapper2 = ResourceGridMapper(rg, precision=precision, device=device)
        rg_ref = mapper2(x)

        assert torch.allclose(rg_compiled, rg_ref, atol=1e-5)

    def test_demapper_compile(self, device, precision, mode):
        """Test that ResourceGridDemapper works with torch.compile"""
        if device == "cpu" and mode != "default":
            pytest.skip("Only default mode supported on CPU")

        rg = ResourceGrid(
            num_ofdm_symbols=14,
            fft_size=64,
            subcarrier_spacing=30e3,
            precision=precision,
            device=device,
        )
        sm = StreamManagement(np.ones([1, 1]), 1)
        mapper = ResourceGridMapper(rg, precision=precision, device=device)
        demapper = ResourceGridDemapper(rg, sm, precision=precision, device=device)
        compiled_demapper = torch.compile(demapper, mode=mode)

        qam_source = QAMSource(4, precision=precision, device=device)
        x = qam_source([32, 1, 1, rg.num_data_symbols])
        rg_mapped = mapper(x)

        x_hat_compiled = compiled_demapper(rg_mapped)

        # Create fresh demapper for comparison
        demapper2 = ResourceGridDemapper(rg, sm, precision=precision, device=device)
        x_hat_ref = demapper2(rg_mapped)

        assert torch.allclose(x_hat_compiled, x_hat_ref, atol=1e-5)

