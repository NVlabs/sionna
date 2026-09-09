#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Tests for window functions: Window, CustomWindow, HannWindow, HammingWindow, BlackmanWindow"""

import numpy as np
import pytest
import torch

from sionna.phy import dtypes
from sionna.phy.signal import CustomWindow, HannWindow, HammingWindow, BlackmanWindow


class TestCustomWindow:
    """Tests for the CustomWindow class"""

    @pytest.mark.parametrize("inp_complex", [False, True])
    def test_dtype(self, precision, device, inp_complex):
        """Test the output dtype for different input and window dtypes"""
        rdtype = dtypes[precision]["torch"]["dtype"]
        cdtype = dtypes[precision]["torch"]["cdtype"]

        if inp_complex:
            inp = torch.randn(64, 100, dtype=cdtype, device=device)
        else:
            inp = torch.randn(64, 100, dtype=rdtype, device=device)

        win_coeff = torch.randn(100, dtype=rdtype, device=device)
        window = CustomWindow(
            coefficients=win_coeff, precision=precision, device=device
        )
        out = window(inp)

        if inp_complex:
            out_dtype = cdtype
        else:
            out_dtype = rdtype
        assert out.dtype == out_dtype

    def test_shape(self, device):
        """Test the output shape"""
        input_shape = [64, 16, 24, 100]
        window_length = input_shape[-1]
        inp = torch.randn(input_shape, device=device)
        win_coeff = torch.randn(window_length, device=device)
        window = CustomWindow(win_coeff, device=device)
        out = window(inp)
        assert list(out.shape) == input_shape

    @pytest.mark.parametrize("window_length", [1, 2, 3, 8, 200])
    def test_length_mismatch_rejected(self, device, window_length):
        """A window must match the input length it is applied to.

        Length one is the important case: it is the only length that
        broadcasts, so it used to pass silently and return the input
        unchanged instead of reporting the mismatch.
        """
        inp = torch.randn([4, 100], device=device)
        window = CustomWindow(
            torch.ones(window_length, device=device), device=device
        )

        with pytest.raises(ValueError, match="must match the window length"):
            window(inp)

    def test_matching_length_accepted(self, device):
        """The documented case must keep working, including length one."""
        for length in [1, 8, 100]:
            window = CustomWindow(
                torch.ones(length, device=device), normalize=False, device=device
            )
            inp = torch.randn([4, length], device=device)
            assert list(window(inp).shape) == [4, length]

    @pytest.mark.parametrize("inp_complex", [False, True])
    def test_computation(self, device, inp_complex):
        """Test the calculation"""
        batch_size = 64
        input_length = 100
        rdtype = dtypes["double"]["torch"]["dtype"]
        cdtype = dtypes["double"]["torch"]["cdtype"]

        if inp_complex:
            inp = torch.randn(batch_size, input_length, dtype=cdtype, device=device)
        else:
            inp = torch.randn(batch_size, input_length, dtype=rdtype, device=device)

        coefficients = torch.randn(input_length, dtype=rdtype, device=device)
        window = CustomWindow(
            coefficients=coefficients, precision="double", device=device
        )
        out = window(inp)

        # Reference: element-wise multiplication
        out_ref = coefficients.unsqueeze(0) * inp
        max_err = torch.max(torch.abs(out - out_ref)).item()
        assert max_err <= 1e-10

    def test_normalization(self, device):
        """Test the normalization"""
        win_length = 128
        rdtype = dtypes["double"]["torch"]["dtype"]
        coeff = torch.randn(win_length, dtype=rdtype, device=device)
        window = CustomWindow(coeff, normalize=True, precision="double", device=device)
        inp = torch.ones(win_length, dtype=rdtype, device=device)
        out = window(inp)
        mean_power = torch.mean(torch.abs(out) ** 2).item()
        assert abs(mean_power - 1.0) <= 1e-6

    def test_trainable(self, device):
        """Test gradient computation"""
        batch_size = 64
        win_length = 128
        rdtype = dtypes["single"]["torch"]["dtype"]
        cdtype = dtypes["single"]["torch"]["cdtype"]

        inp = torch.randn(batch_size, win_length, dtype=cdtype, device=device)
        coeff = torch.randn(win_length, dtype=rdtype, device=device, requires_grad=True)

        # Trainable
        window = CustomWindow(coeff, precision="single", device=device)
        out = window(inp)
        loss = torch.mean(torch.abs(out) ** 2)
        loss.backward()

        assert coeff.grad is not None
        assert coeff.grad.shape == coeff.shape
        assert coeff.grad.sum().item() != 0


class TestHannWindow:
    """Tests for the HannWindow class"""

    def test_shape(self, device, precision):
        """Test the output shape"""
        input_shape = [32, 64]
        inp = torch.randn(input_shape, device=device)
        window = HannWindow(precision=precision, device=device)
        out = window(inp)
        assert list(out.shape) == input_shape

    def test_coefficients(self, device, precision):
        """Test that Hann window coefficients are computed correctly"""
        N = 64
        rdtype = dtypes[precision]["torch"]["dtype"]
        inp = torch.ones(N, dtype=rdtype, device=device)
        window = HannWindow(precision=precision, device=device)
        out = window(inp)

        # Expected Hann window: sin^2(pi * n / N)
        n = np.arange(N)
        expected = np.square(np.sin(np.pi * n / N))

        tol = 1e-5 if precision == "single" else 1e-10
        max_err = np.max(np.abs(out.cpu().numpy() - expected))
        assert max_err < tol

    def test_complex_input(self, device, precision):
        """Test that window works with complex input"""
        cdtype = dtypes[precision]["torch"]["cdtype"]
        inp = torch.randn(32, 64, dtype=cdtype, device=device)
        window = HannWindow(precision=precision, device=device)
        out = window(inp)
        assert out.is_complex()
        assert out.dtype == cdtype


class TestHammingWindow:
    """Tests for the HammingWindow class"""

    def test_shape(self, device, precision):
        """Test the output shape"""
        input_shape = [32, 64]
        inp = torch.randn(input_shape, device=device)
        window = HammingWindow(precision=precision, device=device)
        out = window(inp)
        assert list(out.shape) == input_shape

    def test_coefficients(self, device, precision):
        """Test that Hamming window coefficients are computed correctly"""
        N = 64
        rdtype = dtypes[precision]["torch"]["dtype"]
        inp = torch.ones(N, dtype=rdtype, device=device)
        window = HammingWindow(precision=precision, device=device)
        out = window(inp)

        # Expected Hamming window: a0 - (1-a0) * cos(2*pi*n/N)
        n = np.arange(N)
        a0 = 25.0 / 46.0
        a1 = 1.0 - a0
        expected = a0 - a1 * np.cos(2.0 * np.pi * n / N)

        tol = 1e-5 if precision == "single" else 1e-10
        max_err = np.max(np.abs(out.cpu().numpy() - expected))
        assert max_err < tol


class TestBlackmanWindow:
    """Tests for the BlackmanWindow class"""

    def test_shape(self, device, precision):
        """Test the output shape"""
        input_shape = [32, 64]
        inp = torch.randn(input_shape, device=device)
        window = BlackmanWindow(precision=precision, device=device)
        out = window(inp)
        assert list(out.shape) == input_shape

    def test_coefficients(self, device, precision):
        """Test that Blackman window coefficients are computed correctly"""
        N = 64
        rdtype = dtypes[precision]["torch"]["dtype"]
        inp = torch.ones(N, dtype=rdtype, device=device)
        window = BlackmanWindow(precision=precision, device=device)
        out = window(inp)

        # Expected Blackman window
        n = np.arange(N)
        a0 = 7938.0 / 18608.0
        a1 = 9240.0 / 18608.0
        a2 = 1430.0 / 18608.0
        expected = (
            a0 - a1 * np.cos(2.0 * np.pi * n / N) + a2 * np.cos(4.0 * np.pi * n / N)
        )

        tol = 1e-5 if precision == "single" else 1e-10
        max_err = np.max(np.abs(out.cpu().numpy() - expected))
        assert max_err < tol

    def test_normalization(self, device, precision):
        """Test the normalization feature"""
        N = 128
        rdtype = dtypes[precision]["torch"]["dtype"]
        inp = torch.ones(N, dtype=rdtype, device=device)
        window = BlackmanWindow(normalize=True, precision=precision, device=device)
        out = window(inp)

        # With normalization, mean power should be 1
        mean_power = torch.mean(out**2).item()
        tol = 1e-5 if precision == "single" else 1e-10
        assert abs(mean_power - 1.0) < tol
