#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Tests for Upsampling block"""

import pytest
import torch

from sionna.phy import dtypes
from sionna.phy.signal import Upsampling


class TestUpsampling:
    """Tests for the Upsampling class"""

    @pytest.mark.parametrize("samples_per_symbol", [0, -1, -4])
    def test_invalid_samples_per_symbol_rejected(self, device, samples_per_symbol):
        """A non-positive upsampling factor must be reported, not applied.

        Zero silently produced an empty tensor and negative values failed deep
        inside ``torch.nn.functional.pad``.
        """
        with pytest.raises(ValueError, match="samples_per_symbol"):
            Upsampling(samples_per_symbol=samples_per_symbol, device=device)

    def test_first_dimension_axis_supported(self, device):
        """``axis=0`` must work; the implementation is dimension-agnostic."""
        x = torch.arange(12.0, device=device).reshape(3, 4)
        y = Upsampling(samples_per_symbol=2, axis=0, device=device)(x)

        assert list(y.shape) == [6, 4]
        assert torch.equal(y[0], x[0])
        assert torch.all(y[1] == 0)

    def test_shape_default_axis(self, device, precision):
        """Test the output shape with default axis (-1)"""
        rdtype = dtypes[precision]["torch"]["dtype"]
        samples_per_symbol = 4

        upsampler = Upsampling(
            samples_per_symbol=samples_per_symbol, precision=precision, device=device
        )

        x = torch.randn(32, 100, dtype=rdtype, device=device)
        y = upsampler(x)

        expected_shape = [32, 100 * samples_per_symbol]
        assert list(y.shape) == expected_shape

    def test_shape_different_axis(self, device, precision):
        """Test the output shape with different axis values"""
        rdtype = dtypes[precision]["torch"]["dtype"]
        samples_per_symbol = 4

        # Test with axis=1 (upsampling the second dimension)
        upsampler = Upsampling(
            samples_per_symbol=samples_per_symbol,
            axis=1,
            precision=precision,
            device=device,
        )

        x = torch.randn(32, 50, 100, dtype=rdtype, device=device)
        y = upsampler(x)

        expected_shape = [32, 50 * samples_per_symbol, 100]
        assert list(y.shape) == expected_shape

    def test_dtype_preservation_real(self, device, precision):
        """Test that real dtype is preserved"""
        rdtype = dtypes[precision]["torch"]["dtype"]
        upsampler = Upsampling(samples_per_symbol=4, precision=precision, device=device)

        x = torch.randn(32, 100, dtype=rdtype, device=device)
        y = upsampler(x)

        assert y.dtype == rdtype

    def test_dtype_preservation_complex(self, device, precision):
        """Test that complex dtype is preserved"""
        cdtype = dtypes[precision]["torch"]["cdtype"]
        upsampler = Upsampling(samples_per_symbol=4, precision=precision, device=device)

        x = torch.randn(32, 100, dtype=cdtype, device=device)
        y = upsampler(x)

        assert y.dtype == cdtype

    def test_zero_insertion(self, device, precision):
        """Test that zeros are correctly inserted between samples"""
        rdtype = dtypes[precision]["torch"]["dtype"]
        samples_per_symbol = 4

        upsampler = Upsampling(
            samples_per_symbol=samples_per_symbol, precision=precision, device=device
        )

        # Create a simple input
        x = torch.tensor([1.0, 2.0, 3.0], dtype=rdtype, device=device)
        y = upsampler(x)

        # Expected: [1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0]
        expected = torch.tensor(
            [1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0],
            dtype=rdtype,
            device=device,
        )

        assert torch.allclose(y, expected)

    def test_upsampling_factor_1(self, device, precision):
        """Test that upsampling factor of 1 returns the input unchanged"""
        rdtype = dtypes[precision]["torch"]["dtype"]

        upsampler = Upsampling(samples_per_symbol=1, precision=precision, device=device)

        x = torch.randn(32, 100, dtype=rdtype, device=device)
        y = upsampler(x)

        assert torch.allclose(y, x)

    def test_batched_input(self, device, precision):
        """Test with multi-dimensional batched input"""
        rdtype = dtypes[precision]["torch"]["dtype"]
        samples_per_symbol = 3

        upsampler = Upsampling(
            samples_per_symbol=samples_per_symbol, precision=precision, device=device
        )

        x = torch.randn(8, 16, 24, 50, dtype=rdtype, device=device)
        y = upsampler(x)

        expected_shape = [8, 16, 24, 50 * samples_per_symbol]
        assert list(y.shape) == expected_shape

    def test_gradient_flow(self, device):
        """Test that gradients flow through the upsampling operation"""
        rdtype = dtypes["double"]["torch"]["dtype"]

        upsampler = Upsampling(samples_per_symbol=4, precision="double", device=device)

        x = torch.randn(32, 100, dtype=rdtype, device=device, requires_grad=True)
        y = upsampler(x)
        loss = y.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape
        # Due to zero-insertion, each input element contributes once to the output
        assert torch.allclose(x.grad, torch.ones_like(x))
