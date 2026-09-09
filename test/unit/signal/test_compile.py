#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Tests for torch.compile compatibility of signal module blocks"""

import pytest
import torch

from sionna.phy import dtypes
from sionna.phy.signal import (
    Upsampling,
    Downsampling,
    HannWindow,
    HammingWindow,
    BlackmanWindow,
    CustomWindow,
    RootRaisedCosineFilter,
    RaisedCosineFilter,
    SincFilter,
    CustomFilter,
    convolve,
    fft,
    ifft,
)


class TestCompileUpsampling:
    """Tests for torch.compile compatibility of Upsampling"""

    @pytest.mark.parametrize("mode", ["default", "reduce-overhead"])
    def test_compile_upsampling(self, device, precision, mode):
        """Test that Upsampling works with torch.compile"""
        rdtype = dtypes[precision]["torch"]["dtype"]
        upsampler = Upsampling(samples_per_symbol=4, precision=precision, device=device)

        # Compile the block
        compiled_upsampler = torch.compile(upsampler, mode=mode)

        x = torch.randn(32, 100, dtype=rdtype, device=device)

        # Run both versions
        y_eager = upsampler(x)
        y_compiled = compiled_upsampler(x)

        assert y_compiled.shape == y_eager.shape
        assert torch.allclose(y_compiled, y_eager, atol=1e-5)


class TestCompileDownsampling:
    """Tests for torch.compile compatibility of Downsampling"""

    @pytest.mark.parametrize("mode", ["default", "reduce-overhead"])
    def test_compile_downsampling(self, device, precision, mode):
        """Test that Downsampling works with torch.compile"""
        rdtype = dtypes[precision]["torch"]["dtype"]
        downsampler = Downsampling(
            samples_per_symbol=4, offset=2, precision=precision, device=device
        )

        # Compile the block
        compiled_downsampler = torch.compile(downsampler, mode=mode)

        x = torch.randn(32, 400, dtype=rdtype, device=device)

        # Run both versions
        y_eager = downsampler(x)
        y_compiled = compiled_downsampler(x)

        assert y_compiled.shape == y_eager.shape
        assert torch.allclose(y_compiled, y_eager, atol=1e-5)


class TestCompileWindows:
    """Tests for torch.compile compatibility of Window classes"""

    @pytest.mark.parametrize("mode", ["default", "reduce-overhead"])
    @pytest.mark.parametrize(
        "window_class", [HannWindow, HammingWindow, BlackmanWindow]
    )
    def test_compile_builtin_windows(self, device, precision, mode, window_class):
        """Test that built-in window classes work with torch.compile"""
        rdtype = dtypes[precision]["torch"]["dtype"]
        window = window_class(precision=precision, device=device)
        reference_window = window_class(precision=precision, device=device)

        # Compile and invoke this instance while it is still cold.
        compiled_window = torch.compile(window, mode=mode)

        x = torch.randn(32, 64, dtype=rdtype, device=device)

        y_compiled = compiled_window(x)
        y_eager = reference_window(x)

        assert window.built
        assert y_compiled.shape == y_eager.shape
        assert torch.allclose(y_compiled, y_eager, atol=1e-5)

    @pytest.mark.parametrize("mode", ["default", "reduce-overhead"])
    def test_compile_custom_window(self, device, precision, mode):
        """Test that CustomWindow works with torch.compile"""
        rdtype = dtypes[precision]["torch"]["dtype"]
        coefficients = torch.randn(64, dtype=rdtype, device=device)
        window = CustomWindow(
            coefficients=coefficients, precision=precision, device=device
        )

        # Compile the block
        compiled_window = torch.compile(window, mode=mode)

        x = torch.randn(32, 64, dtype=rdtype, device=device)

        # Run both versions
        y_eager = window(x)
        y_compiled = compiled_window(x)

        assert y_compiled.shape == y_eager.shape
        assert torch.allclose(y_compiled, y_eager, atol=1e-5)


class TestCompileFilters:
    """Tests for torch.compile compatibility of Filter classes"""

    @pytest.mark.parametrize("mode", ["default", "reduce-overhead"])
    def test_compile_rrc_filter(self, device, precision, mode):
        """Test that RootRaisedCosineFilter works with torch.compile"""
        rdtype = dtypes[precision]["torch"]["dtype"]
        rrc = RootRaisedCosineFilter(
            span_in_symbols=8,
            samples_per_symbol=4,
            beta=0.35,
            precision=precision,
            device=device,
        )

        # Compile the block
        compiled_rrc = torch.compile(rrc, mode=mode)

        x = torch.randn(32, 100, dtype=rdtype, device=device)

        # Run both versions
        y_eager = rrc(x, padding="same")
        y_compiled = compiled_rrc(x, padding="same")

        assert y_compiled.shape == y_eager.shape
        assert torch.allclose(y_compiled, y_eager, atol=1e-5)

    @pytest.mark.parametrize("mode", ["default", "reduce-overhead"])
    def test_compile_rc_filter(self, device, precision, mode):
        """Test that RaisedCosineFilter works with torch.compile"""
        rdtype = dtypes[precision]["torch"]["dtype"]
        rc = RaisedCosineFilter(
            span_in_symbols=8,
            samples_per_symbol=4,
            beta=0.35,
            precision=precision,
            device=device,
        )

        # Compile the block
        compiled_rc = torch.compile(rc, mode=mode)

        x = torch.randn(32, 100, dtype=rdtype, device=device)

        # Run both versions
        y_eager = rc(x, padding="same")
        y_compiled = compiled_rc(x, padding="same")

        assert y_compiled.shape == y_eager.shape
        assert torch.allclose(y_compiled, y_eager, atol=1e-5)

    @pytest.mark.parametrize("mode", ["default", "reduce-overhead"])
    def test_compile_sinc_filter(self, device, precision, mode):
        """Test that SincFilter works with torch.compile"""
        rdtype = dtypes[precision]["torch"]["dtype"]
        sinc = SincFilter(
            span_in_symbols=8,
            samples_per_symbol=4,
            precision=precision,
            device=device,
        )

        # Compile the block
        compiled_sinc = torch.compile(sinc, mode=mode)

        x = torch.randn(32, 100, dtype=rdtype, device=device)

        # Run both versions
        y_eager = sinc(x, padding="same")
        y_compiled = compiled_sinc(x, padding="same")

        assert y_compiled.shape == y_eager.shape
        assert torch.allclose(y_compiled, y_eager, atol=1e-5)

    @pytest.mark.parametrize("mode", ["default", "reduce-overhead"])
    def test_compile_custom_filter(self, device, precision, mode):
        """Test that CustomFilter works with torch.compile"""
        rdtype = dtypes[precision]["torch"]["dtype"]
        coefficients = torch.randn(33, dtype=rdtype, device=device)
        filt = CustomFilter(
            samples_per_symbol=4,
            coefficients=coefficients,
            precision=precision,
            device=device,
        )

        # Compile the block
        compiled_filt = torch.compile(filt, mode=mode)

        x = torch.randn(32, 100, dtype=rdtype, device=device)

        # Run both versions
        y_eager = filt(x, padding="same")
        y_compiled = compiled_filt(x, padding="same")

        assert y_compiled.shape == y_eager.shape
        assert torch.allclose(y_compiled, y_eager, atol=1e-5)


class TestCompileFunctions:
    """Tests for torch.compile compatibility of signal utility functions"""

    @pytest.mark.parametrize("mode", ["default", "reduce-overhead"])
    @pytest.mark.parametrize("padding", ["valid", "same", "full"])
    def test_compile_convolve(self, device, precision, mode, padding):
        """Test that convolve works with torch.compile"""
        rdtype = dtypes[precision]["torch"]["dtype"]

        def convolve_fn(inp, ker):
            return convolve(inp, ker, padding=padding, precision=precision)

        compiled_convolve = torch.compile(convolve_fn, mode=mode)

        inp = torch.randn(32, 100, dtype=rdtype, device=device)
        ker = torch.randn(11, dtype=rdtype, device=device)

        # Run both versions
        y_eager = convolve_fn(inp, ker)
        y_compiled = compiled_convolve(inp, ker)

        assert y_compiled.shape == y_eager.shape
        assert torch.allclose(y_compiled, y_eager, atol=1e-5)

    @pytest.mark.parametrize("mode", ["default", "reduce-overhead"])
    def test_compile_fft(self, device, precision, mode):
        """Test that fft works with torch.compile"""
        cdtype = dtypes[precision]["torch"]["cdtype"]

        def fft_fn(x):
            return fft(x, precision=precision)

        compiled_fft = torch.compile(fft_fn, mode=mode)

        x = torch.randn(32, 64, dtype=cdtype, device=device)

        # Run both versions
        y_eager = fft_fn(x)
        y_compiled = compiled_fft(x)

        assert y_compiled.shape == y_eager.shape
        assert torch.allclose(y_compiled, y_eager, atol=1e-5)

    @pytest.mark.parametrize("mode", ["default", "reduce-overhead"])
    def test_compile_ifft(self, device, precision, mode):
        """Test that ifft works with torch.compile"""
        cdtype = dtypes[precision]["torch"]["cdtype"]

        def ifft_fn(x):
            return ifft(x, precision=precision)

        compiled_ifft = torch.compile(ifft_fn, mode=mode)

        x = torch.randn(32, 64, dtype=cdtype, device=device)

        # Run both versions
        y_eager = ifft_fn(x)
        y_compiled = compiled_ifft(x)

        assert y_compiled.shape == y_eager.shape
        assert torch.allclose(y_compiled, y_eager, atol=1e-5)
