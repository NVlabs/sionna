#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Unit tests for pilot pattern classes"""

import numpy as np
import pytest
import torch

from sionna.phy import config
from sionna.phy.ofdm import (
    EmptyPilotPattern,
    KroneckerPilotPattern,
    PilotPattern,
    ResourceGrid,
)


class TestPilotPattern:
    """Tests for the PilotPattern base class"""

    def test_basic_creation(self, device, precision):
        """Test basic PilotPattern creation with valid inputs"""
        num_tx = 2
        num_streams = 2
        num_ofdm_symbols = 14
        num_eff_subcarriers = 64

        # Create mask and pilots
        mask = np.zeros([num_tx, num_streams, num_ofdm_symbols, num_eff_subcarriers])
        mask[:, :, 0, :] = 1  # First OFDM symbol for pilots
        num_pilots = int(mask.sum() / (num_tx * num_streams))
        pilots = np.ones([num_tx, num_streams, num_pilots], dtype=np.complex64)

        pp = PilotPattern(
            mask, pilots, normalize=False, precision=precision, device=device
        )

        assert pp.num_tx == num_tx
        assert pp.num_streams_per_tx == num_streams
        assert pp.num_ofdm_symbols == num_ofdm_symbols
        assert pp.num_effective_subcarriers == num_eff_subcarriers
        assert pp.num_pilot_symbols == num_pilots
        assert pp.mask.device.type == device.split(":")[0]

    def test_pilot_normalization(self, device, precision):
        """Test that pilot normalization works correctly"""
        mask = np.zeros([1, 1, 4, 8])
        mask[0, 0, 0, :4] = 1
        pilots = np.array([[[2 + 2j, 1 + 1j, 3 + 3j, 0.5 + 0.5j]]], dtype=np.complex64)

        pp = PilotPattern(
            mask, pilots, normalize=True, precision=precision, device=device
        )

        # Check that normalized pilots have unit average energy
        norm_pilots = pp.pilots
        avg_energy = norm_pilots.abs().square().mean()
        assert torch.allclose(
            avg_energy,
            torch.tensor(1.0, dtype=avg_energy.dtype, device=device),
            atol=1e-5
        )

    def test_mask_pilots_shape_mismatch(self, device, precision):
        """Test that mismatched mask and pilots shapes raise error"""
        mask = np.zeros([2, 2, 4, 8])
        mask[:, :, 0, :] = 1
        # Wrong shape for pilots
        pilots = np.ones([1, 1, 8], dtype=np.complex64)

        with pytest.raises(ValueError):
            PilotPattern(mask, pilots, precision=precision, device=device)

    def test_check_settings_mask_wrong_rank(self, device, precision):
        """Test that mask with wrong rank (not 4) raises error"""
        # mask does not have rank 4
        mask = np.zeros([1, 10], bool)
        pilots = np.zeros([1, 10, 20], np.complex64)

        with pytest.raises(ValueError):
            PilotPattern(mask, pilots, precision=precision, device=device)

    def test_check_settings_pilots_wrong_rank(self, device, precision):
        """Test that pilots with wrong rank (not 3) raises error"""
        # pilots does not have rank 3
        mask = np.zeros([4, 2, 10, 46], bool)
        pilots = np.zeros([1, 10, 20, 2], np.complex64)

        with pytest.raises(ValueError):
            PilotPattern(mask, pilots, precision=precision, device=device)

    def test_check_settings_dimension_mismatch(self, device, precision):
        """Test that pilots and mask with different first two dimensions raise error"""
        mask = np.zeros([1, 2, 14, 64], bool)
        mask[0, 0, 0, :] = True
        mask[0, 1, 1, :] = True
        num_pilots = int(np.max(np.sum(mask, (-2, -1))))
        # Wrong second dimension (3 instead of 2)
        pilots = np.zeros([1, 3, num_pilots], np.complex64)

        with pytest.raises(ValueError):
            PilotPattern(mask, pilots, precision=precision, device=device)

    def test_check_settings_inconsistent_true_counts(self, device, precision):
        """Test that mask with inconsistent True counts across TX/streams raises error"""
        mask = np.zeros([1, 2, 14, 64], bool)
        mask[0, 0, 0, :] = True  # 64 pilots
        mask[0, 1, 1:3, :] = True  # 128 pilots - different count!
        num_pilots = int(np.max(np.sum(mask, (-2, -1))))
        pilots = np.zeros([1, 2, num_pilots], np.complex64)

        with pytest.raises(ValueError):
            PilotPattern(mask, pilots, precision=precision, device=device)

    def test_check_settings_wrong_pilots_last_dim(self, device, precision):
        """Test that pilots with wrong last dimension raises error"""
        mask = np.zeros([1, 2, 14, 64], bool)
        mask[0, 0, 0, :] = True
        mask[0, 1, 1, :] = True
        num_pilots = int(np.max(np.sum(mask, (-2, -1))))
        # Wrong last dimension (num_pilots+1 instead of num_pilots)
        pilots = np.zeros([1, 2, num_pilots + 1], np.complex64)

        with pytest.raises(ValueError):
            PilotPattern(mask, pilots, precision=precision, device=device)

    def test_properties_basic(self, device, precision):
        """Test num_pilot_symbols and num_data_symbols properties"""
        mask = np.zeros([1, 2, 14, 64], bool)
        mask[0, 0, 0, :] = True
        mask[0, 1, 1, :] = True
        num_pilots = int(np.max(np.sum(mask, (-2, -1))))
        pilots = np.zeros([1, 2, num_pilots], np.complex64)
        pp = PilotPattern(mask, pilots, precision=precision, device=device)

        assert pp.num_pilot_symbols == 64
        assert pp.num_data_symbols == 13 * 64

    def test_properties_multiple_pilot_symbols(self, device, precision):
        """Test properties with multiple OFDM symbols for pilots"""
        mask = np.zeros([1, 2, 14, 64], bool)
        mask[0, 0, :2, :] = True  # First two OFDM symbols
        mask[0, 1, 1:3, :] = True  # Second and third OFDM symbols
        num_pilots = int(np.max(np.sum(mask, (-2, -1))))
        pilots = np.zeros([1, 2, num_pilots], np.complex64)
        pp = PilotPattern(mask, pilots, precision=precision, device=device)

        assert pp.num_pilot_symbols == 128
        assert pp.num_data_symbols == 12 * 64


class TestEmptyPilotPattern:
    """Tests for the EmptyPilotPattern class"""

    def test_empty_pattern_creation(self, device, precision):
        """Test EmptyPilotPattern creates valid empty pattern"""
        num_tx = 2
        num_streams = 2
        num_ofdm_symbols = 14
        num_eff_subcarriers = 64

        pp = EmptyPilotPattern(
            num_tx, num_streams, num_ofdm_symbols, num_eff_subcarriers,
            precision=precision, device=device
        )

        assert pp.num_tx == num_tx
        assert pp.num_streams_per_tx == num_streams
        assert pp.num_ofdm_symbols == num_ofdm_symbols
        assert pp.num_effective_subcarriers == num_eff_subcarriers
        assert pp.num_pilot_symbols == 0
        assert pp.num_data_symbols == num_ofdm_symbols * num_eff_subcarriers
        assert pp.mask.sum() == 0

    def test_empty_pattern_invalid_params(self, device, precision):
        """Test EmptyPilotPattern with invalid parameters"""
        with pytest.raises(ValueError):
            EmptyPilotPattern(0, 1, 14, 64, precision=precision, device=device)

        with pytest.raises(ValueError):
            EmptyPilotPattern(1, 0, 14, 64, precision=precision, device=device)

    def test_empty_pattern_nan_dimension(self, device, precision):
        """Test EmptyPilotPattern rejects a NaN dimension"""
        with pytest.raises(ValueError, match="positive"):
            EmptyPilotPattern(
                float("nan"), 1, 14, 64, precision=precision, device=device
            )


class TestKroneckerPilotPattern:
    """Tests for the KroneckerPilotPattern class"""

    def test_kronecker_pattern_creation(self, device, precision):
        """Test KroneckerPilotPattern creates valid orthogonal pattern"""
        rg = ResourceGrid(
            num_ofdm_symbols=14,
            fft_size=64,
            subcarrier_spacing=30e3,
            num_tx=4,
            num_streams_per_tx=2,
            precision=precision,
            device=device,
        )

        pp = KroneckerPilotPattern(
            rg, [2, 11], precision=precision, device=device
        )

        assert pp.num_tx == 4
        assert pp.num_streams_per_tx == 2
        # Check that pilots are on the correct OFDM symbols
        assert pp.mask[:, :, 2, :].sum() > 0
        assert pp.mask[:, :, 11, :].sum() > 0
        # Other symbols should have no pilots
        assert pp.mask[:, :, 0, :].sum() == 0

    def test_kronecker_orthogonality(self, device, precision):
        """Test that Kronecker pilots are orthogonal across streams"""
        rg = ResourceGrid(
            num_ofdm_symbols=14,
            fft_size=64,
            subcarrier_spacing=30e3,
            num_tx=2,
            num_streams_per_tx=2,
            precision=precision,
            device=device,
        )

        pp = KroneckerPilotPattern(
            rg, [2], normalize=False, precision=precision, device=device
        )

        # Check that different TX/stream combinations have non-overlapping pilots
        pilots = pp.pilots.cpu().numpy()
        for i in range(2):
            for j in range(2):
                p_ij = pilots[i, j]
                for k in range(2):
                    for l in range(2):
                        if i != k or j != l:
                            p_kl = pilots[k, l]
                            # Non-zero elements should not overlap
                            overlap = np.sum((p_ij != 0) & (p_kl != 0))
                            assert overlap == 0

    def test_kronecker_with_resource_grid_shorthand(self, device, precision):
        """Test creating KroneckerPilotPattern via ResourceGrid shorthand"""
        rg = ResourceGrid(
            num_ofdm_symbols=14,
            fft_size=64,
            subcarrier_spacing=30e3,
            num_tx=4,
            num_streams_per_tx=2,
            pilot_pattern="kronecker",
            pilot_ofdm_symbol_indices=[2, 11],
            precision=precision,
            device=device,
        )

        assert isinstance(rg.pilot_pattern, KroneckerPilotPattern)
        assert rg.num_pilot_symbols > 0

    def test_kronecker_uses_sionna_rng(self, device, precision):
        """The global Sionna seed controls random pilot generation."""
        rg = ResourceGrid(
            num_ofdm_symbols=14,
            fft_size=64,
            subcarrier_spacing=30e3,
            num_tx=2,
            num_streams_per_tx=2,
            precision=precision,
            device=device,
        )
        previous_seed = config.seed
        try:
            config.seed = 123
            pilots_a = KroneckerPilotPattern(
                rg, [2, 11], precision=precision, device=device
            ).pilots
            config.seed = 123
            pilots_b = KroneckerPilotPattern(
                rg, [2, 11], precision=precision, device=device
            ).pilots
            config.seed = 456
            pilots_c = KroneckerPilotPattern(
                rg, [2, 11], precision=precision, device=device
            ).pilots
        finally:
            config.seed = previous_seed

        assert torch.equal(pilots_a, pilots_b)
        assert not torch.equal(pilots_a, pilots_c)


class TestPilotPatternCompile:
    """Tests for torch.compile compatibility"""

    def test_pilot_pattern_compile(self, device, precision, mode):
        """Test that PilotPattern works with torch.compile"""
        if device == "cpu" and mode != "default":
            pytest.skip("Only default mode supported on CPU")

        num_tx = 2
        num_streams = 1
        num_ofdm_symbols = 14
        num_eff_subcarriers = 64

        mask = np.zeros([num_tx, num_streams, num_ofdm_symbols, num_eff_subcarriers])
        mask[:, :, 0, :8] = 1
        num_pilots = 8
        pilots = np.ones([num_tx, num_streams, num_pilots], dtype=np.complex64)

        pp = PilotPattern(
            mask, pilots, normalize=True, precision=precision, device=device
        )

        # Access pilots property which involves computation
        pilots_out = pp.pilots
        assert pilots_out.shape == torch.Size([num_tx, num_streams, num_pilots])

