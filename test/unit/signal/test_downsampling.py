#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Tests for Downsampling block"""

import pytest
import torch

from sionna.phy import dtypes
from sionna.phy.signal import Downsampling


class TestDownsampling:
    """Tests for the Downsampling class"""

    @pytest.mark.parametrize("samples_per_symbol", [0, -1, -4])
    def test_invalid_samples_per_symbol_rejected(self, device, samples_per_symbol):
        """A non-positive downsampling factor must name the offending parameter."""
        with pytest.raises(ValueError, match="samples_per_symbol"):
            Downsampling(samples_per_symbol=samples_per_symbol, device=device)

    @pytest.mark.parametrize("offset", [-1, -4])
    def test_negative_offset_rejected(self, device, offset):
        """A negative offset must not be reinterpreted as counting from the end.

        ``call()`` slices ``x[..., offset::samples_per_symbol]``, so a negative
        offset used to select samples from the end of the signal and return
        them without any error.
        """
        with pytest.raises(ValueError, match="offset"):
            Downsampling(samples_per_symbol=2, offset=offset, device=device)

    @pytest.mark.parametrize("num_symbols", [-1, -10])
    def test_negative_num_symbols_rejected(self, device, num_symbols):
        """A negative num_symbols must not silently truncate from the end."""
        with pytest.raises(ValueError, match="num_symbols"):
            Downsampling(
                samples_per_symbol=2, num_symbols=num_symbols, device=device
            )

    def test_zero_num_symbols_accepted(self, device):
        """Zero is a degenerate but well-defined request for no symbols."""
        x = torch.arange(8.0, device=device)
        y = Downsampling(samples_per_symbol=2, num_symbols=0, device=device)(x)
        assert list(y.shape) == [0]

    def test_first_dimension_axis_supported(self, device):
        """``axis=0`` must work; the implementation is dimension-agnostic."""
        x = torch.arange(12.0, device=device).reshape(4, 3)
        y = Downsampling(samples_per_symbol=2, axis=0, device=device)(x)

        assert list(y.shape) == [2, 3]
        assert torch.equal(y[0], x[0])
        assert torch.equal(y[1], x[2])

    def test_shape_default_axis(self, device, precision):
        """Test the output shape with default axis (-1)"""
        rdtype = dtypes[precision]["torch"]["dtype"]
        samples_per_symbol = 4

        downsampler = Downsampling(
            samples_per_symbol=samples_per_symbol, precision=precision, device=device
        )

        x = torch.randn(32, 400, dtype=rdtype, device=device)
        y = downsampler(x)

        expected_shape = [32, 400 // samples_per_symbol]
        assert list(y.shape) == expected_shape

    def test_shape_different_axis(self, device, precision):
        """Test the output shape with different axis values"""
        rdtype = dtypes[precision]["torch"]["dtype"]
        samples_per_symbol = 4

        # Test with axis=1 (downsampling the second dimension)
        downsampler = Downsampling(
            samples_per_symbol=samples_per_symbol,
            axis=1,
            precision=precision,
            device=device,
        )

        x = torch.randn(32, 200, 100, dtype=rdtype, device=device)
        y = downsampler(x)

        expected_shape = [32, 200 // samples_per_symbol, 100]
        assert list(y.shape) == expected_shape

    def test_dtype_preservation_real(self, device, precision):
        """Test that real dtype is preserved"""
        rdtype = dtypes[precision]["torch"]["dtype"]
        downsampler = Downsampling(
            samples_per_symbol=4, precision=precision, device=device
        )

        x = torch.randn(32, 400, dtype=rdtype, device=device)
        y = downsampler(x)

        assert y.dtype == rdtype

    def test_dtype_preservation_complex(self, device, precision):
        """Test that complex dtype is preserved"""
        cdtype = dtypes[precision]["torch"]["cdtype"]
        downsampler = Downsampling(
            samples_per_symbol=4, precision=precision, device=device
        )

        x = torch.randn(32, 400, dtype=cdtype, device=device)
        y = downsampler(x)

        assert y.dtype == cdtype

    def test_sample_selection(self, device, precision):
        """Test that correct samples are selected"""
        rdtype = dtypes[precision]["torch"]["dtype"]
        samples_per_symbol = 4

        downsampler = Downsampling(
            samples_per_symbol=samples_per_symbol,
            offset=0,
            precision=precision,
            device=device,
        )

        # Create input where we know the expected output
        x = torch.arange(12, dtype=rdtype, device=device)
        y = downsampler(x)

        # Expected: [0, 4, 8]
        expected = torch.tensor([0.0, 4.0, 8.0], dtype=rdtype, device=device)
        assert torch.allclose(y, expected)

    def test_offset(self, device, precision):
        """Test that offset is correctly applied"""
        rdtype = dtypes[precision]["torch"]["dtype"]
        samples_per_symbol = 4

        downsampler = Downsampling(
            samples_per_symbol=samples_per_symbol,
            offset=2,
            precision=precision,
            device=device,
        )

        x = torch.arange(16, dtype=rdtype, device=device)
        y = downsampler(x)

        # Expected: [2, 6, 10, 14] (starting from offset 2, step 4)
        # (16-2)//4 = 3, but we can fit 4 samples: indices 2, 6, 10, 14
        expected = torch.tensor([2.0, 6.0, 10.0, 14.0], dtype=rdtype, device=device)
        assert torch.allclose(y, expected)

    def test_num_symbols(self, device, precision):
        """Test that num_symbols limits the output length"""
        rdtype = dtypes[precision]["torch"]["dtype"]
        samples_per_symbol = 4

        downsampler = Downsampling(
            samples_per_symbol=samples_per_symbol,
            num_symbols=2,
            precision=precision,
            device=device,
        )

        x = torch.arange(16, dtype=rdtype, device=device)
        y = downsampler(x)

        # Without num_symbols, would get [0, 4, 8, 12], but limited to 2
        expected = torch.tensor([0.0, 4.0], dtype=rdtype, device=device)
        assert torch.allclose(y, expected)

    def test_offset_and_num_symbols(self, device, precision):
        """Test combined offset and num_symbols"""
        rdtype = dtypes[precision]["torch"]["dtype"]
        samples_per_symbol = 4

        downsampler = Downsampling(
            samples_per_symbol=samples_per_symbol,
            offset=1,
            num_symbols=2,
            precision=precision,
            device=device,
        )

        x = torch.arange(20, dtype=rdtype, device=device)
        y = downsampler(x)

        # Starting from 1, step 4: [1, 5, 9, 13, 17], but limited to 2
        expected = torch.tensor([1.0, 5.0], dtype=rdtype, device=device)
        assert torch.allclose(y, expected)

    def test_downsampling_factor_1(self, device, precision):
        """Test that downsampling factor of 1 returns the input unchanged"""
        rdtype = dtypes[precision]["torch"]["dtype"]

        downsampler = Downsampling(
            samples_per_symbol=1, precision=precision, device=device
        )

        x = torch.randn(32, 100, dtype=rdtype, device=device)
        y = downsampler(x)

        assert torch.allclose(y, x)

    def test_batched_input(self, device, precision):
        """Test with multi-dimensional batched input"""
        rdtype = dtypes[precision]["torch"]["dtype"]
        samples_per_symbol = 3

        downsampler = Downsampling(
            samples_per_symbol=samples_per_symbol, precision=precision, device=device
        )

        x = torch.randn(8, 16, 24, 150, dtype=rdtype, device=device)
        y = downsampler(x)

        expected_shape = [8, 16, 24, 150 // samples_per_symbol]
        assert list(y.shape) == expected_shape

    def test_gradient_flow(self, device):
        """Test that gradients flow through the downsampling operation"""
        rdtype = dtypes["double"]["torch"]["dtype"]
        samples_per_symbol = 4

        downsampler = Downsampling(
            samples_per_symbol=samples_per_symbol, precision="double", device=device
        )

        x = torch.randn(32, 400, dtype=rdtype, device=device, requires_grad=True)
        y = downsampler(x)
        loss = y.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape

        # Gradients should only flow to the samples that were selected
        # Check that gradients at sample positions are 1 and others are 0
        for i in range(400):
            if i % samples_per_symbol == 0:
                assert x.grad[0, i].item() == 1.0
            else:
                assert x.grad[0, i].item() == 0.0

    def test_upsampling_downsampling_roundtrip(self, device, precision):
        """Test that upsampling followed by downsampling recovers the original signal"""
        from sionna.phy.signal import Upsampling

        rdtype = dtypes[precision]["torch"]["dtype"]
        samples_per_symbol = 4

        upsampler = Upsampling(
            samples_per_symbol=samples_per_symbol, precision=precision, device=device
        )
        downsampler = Downsampling(
            samples_per_symbol=samples_per_symbol, precision=precision, device=device
        )

        x = torch.randn(32, 100, dtype=rdtype, device=device)
        y = upsampler(x)
        x_recovered = downsampler(y)

        assert torch.allclose(x, x_recovered)
