#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Tests for signal utility functions: convolve, fft, ifft, empirical_psd, empirical_aclr"""

import numpy as np
import pytest
import torch

from sionna.phy import dtypes
from sionna.phy.signal import convolve, fft, ifft, empirical_psd, empirical_aclr


class TestConvolve:
    """Tests for the convolve function"""

    @pytest.mark.parametrize("inp_complex", [False, True])
    @pytest.mark.parametrize("ker_complex", [False, True])
    @pytest.mark.parametrize("padding", ["valid", "same", "full"])
    def test_dtype(self, precision, device, inp_complex, ker_complex, padding):
        """Test the output dtype for all possible combinations of input dtypes"""
        rdtype = dtypes[precision]["torch"]["dtype"]
        cdtype = dtypes[precision]["torch"]["cdtype"]

        if inp_complex:
            inp = torch.randn(64, 100, dtype=cdtype, device=device)
        else:
            inp = torch.randn(64, 100, dtype=rdtype, device=device)

        if ker_complex:
            ker = torch.randn(10, dtype=cdtype, device=device)
        else:
            ker = torch.randn(10, dtype=rdtype, device=device)

        if inp_complex or ker_complex:
            out_dtype = cdtype
        else:
            out_dtype = rdtype

        out = convolve(inp, ker, padding=padding, precision=precision)
        assert out.dtype == out_dtype

    def test_shape(self, device):
        """Test the output shape for all padding modes and for even and odd kernel lengths"""
        # Even kernel length
        input_shape = [64, 16, 24, 100]
        kernel_length = 8
        inp = torch.randn(input_shape, device=device)
        ker = torch.randn(kernel_length, device=device)

        # 'valid' padding
        out = convolve(inp, ker, "valid")
        out_shape = input_shape[:-1] + [input_shape[-1] - kernel_length + 1]
        assert list(out.shape) == out_shape

        # 'same' padding
        out = convolve(inp, ker, "same")
        out_shape = input_shape
        assert list(out.shape) == out_shape

        # 'full' padding
        out = convolve(inp, ker, "full")
        out_shape = input_shape[:-1] + [input_shape[-1] + kernel_length - 1]
        assert list(out.shape) == out_shape

        # Odd kernel length
        kernel_length = 5
        ker = torch.randn(kernel_length, device=device)

        # 'valid' padding
        out = convolve(inp, ker, "valid")
        out_shape = input_shape[:-1] + [input_shape[-1] - kernel_length + 1]
        assert list(out.shape) == out_shape

        # 'same' padding
        out = convolve(inp, ker, "same")
        out_shape = input_shape
        assert list(out.shape) == out_shape

        # 'full' padding
        out = convolve(inp, ker, "full")
        out_shape = input_shape[:-1] + [input_shape[-1] + kernel_length - 1]
        assert list(out.shape) == out_shape

    @pytest.mark.parametrize("inp_complex", [False, True])
    @pytest.mark.parametrize("ker_complex", [False, True])
    @pytest.mark.parametrize("padding", ["valid", "same", "full"])
    @pytest.mark.parametrize("kernel_size", [1, 2, 5, 8, 100])
    def test_computation(self, device, inp_complex, ker_complex, padding, kernel_size):
        """Test the convolution calculation against np.convolve()"""
        input_length = 100
        cdtype = dtypes["double"]["torch"]["cdtype"]
        rdtype = dtypes["double"]["torch"]["dtype"]

        if inp_complex:
            inp = torch.randn(1, input_length, dtype=cdtype, device=device)
        else:
            inp = torch.randn(1, input_length, dtype=rdtype, device=device)

        if ker_complex:
            ker = torch.randn(kernel_size, dtype=cdtype, device=device)
        else:
            ker = torch.randn(kernel_size, dtype=rdtype, device=device)

        out = convolve(inp, ker, padding, precision="double")
        out_ref = np.convolve(inp.cpu().numpy()[0], ker.cpu().numpy(), mode=padding)
        max_err = np.max(np.abs(out.cpu().numpy()[0] - out_ref))
        assert max_err <= 1e-10


class TestFFT:
    """Tests for the fft function"""

    def test_normalization(self, device, precision):
        """Test that fft followed by ifft returns the original signal (within tolerance)"""
        cdtype = dtypes[precision]["torch"]["cdtype"]
        x = torch.randn(32, 64, dtype=cdtype, device=device)

        X = fft(x, precision=precision)
        x_reconstructed = ifft(X, precision=precision)

        if precision == "double":
            tol = 1e-12
        else:
            tol = 1e-5

        max_err = torch.max(torch.abs(x - x_reconstructed)).item()
        assert max_err < tol

    def test_shape(self, device, precision):
        """Test that fft preserves shape"""
        cdtype = dtypes[precision]["torch"]["cdtype"]
        x = torch.randn(32, 64, dtype=cdtype, device=device)

        X = fft(x, precision=precision)
        assert X.shape == x.shape

    def test_axis(self, device, precision):
        """Test that fft works correctly along different axes"""
        cdtype = dtypes[precision]["torch"]["cdtype"]
        x = torch.randn(8, 16, 32, dtype=cdtype, device=device)

        # Default axis=-1
        X1 = fft(x, axis=-1, precision=precision)
        assert X1.shape == x.shape

        # axis=0
        X2 = fft(x, axis=0, precision=precision)
        assert X2.shape == x.shape

        # axis=1
        X3 = fft(x, axis=1, precision=precision)
        assert X3.shape == x.shape

    def test_scaling(self, device, precision):
        """Test that the FFT is normalized by 1/sqrt(N)"""
        cdtype = dtypes[precision]["torch"]["cdtype"]
        N = 64
        x = torch.ones(N, dtype=cdtype, device=device)

        X = fft(x, precision=precision)

        # For a constant input, only the DC component should be non-zero
        # The DC component should be sqrt(N) due to the 1/sqrt(N) normalization
        expected_dc = np.sqrt(N)
        if precision == "double":
            tol = 1e-12
        else:
            tol = 1e-5

        assert abs(X[0].abs().item() - expected_dc) < tol


class TestIFFT:
    """Tests for the ifft function"""

    def test_shape(self, device, precision):
        """Test that ifft preserves shape"""
        cdtype = dtypes[precision]["torch"]["cdtype"]
        X = torch.randn(32, 64, dtype=cdtype, device=device)

        x = ifft(X, precision=precision)
        assert x.shape == X.shape

    def test_axis(self, device, precision):
        """Test that ifft works correctly along different axes"""
        cdtype = dtypes[precision]["torch"]["cdtype"]
        X = torch.randn(8, 16, 32, dtype=cdtype, device=device)

        # Default axis=-1
        x1 = ifft(X, axis=-1, precision=precision)
        assert x1.shape == X.shape

        # axis=0
        x2 = ifft(X, axis=0, precision=precision)
        assert x2.shape == X.shape


class TestEmpiricalPSD:
    """Tests for the empirical_psd function"""

    def test_output_shape(self, device, precision):
        """Test that empirical_psd returns correct shapes"""
        cdtype = dtypes[precision]["torch"]["cdtype"]
        N = 256
        x = torch.randn(100, N, dtype=cdtype, device=device)

        freqs, psd = empirical_psd(x, show=False, precision=precision)

        assert freqs.shape == (N,)
        assert psd.shape == (N,)

    def test_frequency_range(self, device, precision):
        """Test that frequency range is correct for different oversampling factors"""
        cdtype = dtypes[precision]["torch"]["cdtype"]
        x = torch.randn(100, 256, dtype=cdtype, device=device)

        # Oversampling = 1
        freqs, _ = empirical_psd(x, show=False, oversampling=1.0, precision=precision)
        assert freqs[0].item() == pytest.approx(-0.5, abs=1e-5)
        assert freqs[-1].item() == pytest.approx(0.5, abs=1e-5)

        # Oversampling = 2
        freqs, _ = empirical_psd(x, show=False, oversampling=2.0, precision=precision)
        assert freqs[0].item() == pytest.approx(-1.0, abs=1e-5)
        assert freqs[-1].item() == pytest.approx(1.0, abs=1e-5)


class TestEmpiricalACLR:
    """Tests for the empirical_aclr function"""

    def test_output_scalar(self, device, precision):
        """Test that empirical_aclr returns a scalar"""
        cdtype = dtypes[precision]["torch"]["cdtype"]
        x = torch.randn(100, 256, dtype=cdtype, device=device)

        aclr = empirical_aclr(x, precision=precision)
        assert aclr.dim() == 0  # Scalar tensor

    def test_aclr_positive(self, device, precision):
        """Test that ACLR is positive"""
        cdtype = dtypes[precision]["torch"]["cdtype"]
        x = torch.randn(100, 256, dtype=cdtype, device=device)

        aclr = empirical_aclr(x, oversampling=2.0, precision=precision)
        assert aclr.item() > 0

    def test_fullgraph_compile(self, device, precision):
        """A valid ACLR computation must not synchronize a tensor to Python."""
        cdtype = dtypes[precision]["torch"]["cdtype"]
        x = torch.randn(16, 128, dtype=cdtype, device=device)

        def aclr_fn(signal):
            return empirical_aclr(
                signal, oversampling=2.0, precision=precision
            )

        expected = aclr_fn(x)
        actual = torch.compile(aclr_fn, fullgraph=True)(x)

        torch.testing.assert_close(actual, expected)

    @pytest.mark.parametrize("oversampling", [0.0, -1.0, -2.5])
    def test_invalid_oversampling_rejected(self, device, oversampling):
        """A non-positive oversampling factor must be reported.

        The frequency axis spans ``[-0.5*oversampling, 0.5*oversampling]``, so a
        negative factor reverses it and a zero factor collapses it, in both
        cases returning a plausible-looking number.
        """
        x = torch.randn(16, 128, dtype=torch.complex64, device=device)

        with pytest.raises(ValueError, match="oversampling"):
            empirical_aclr(x, oversampling=oversampling)

    @pytest.mark.parametrize("f_min,f_max", [(0.5, -0.5), (0.1, 0.1), (1.0, 0.0)])
    def test_swapped_or_empty_band_rejected(self, device, f_min, f_max):
        """A swapped or degenerate in-band must be reported, not divided by."""
        x = torch.randn(16, 128, dtype=torch.complex64, device=device)

        with pytest.raises(ValueError, match="f_min"):
            empirical_aclr(x, oversampling=2.0, f_min=f_min, f_max=f_max)

    def test_band_outside_spectrum_rejected(self, device):
        """An in-band that does not overlap the spectrum must be reported."""
        x = torch.randn(16, 128, dtype=torch.complex64, device=device)

        with pytest.raises(ValueError, match="lies outside the spectrum"):
            empirical_aclr(x, oversampling=2.0, f_min=5.0, f_max=9.0)

    def test_band_narrower_than_resolution_rejected(self, device):
        """An in-band selecting no frequency bin must be reported."""
        x = torch.randn(16, 128, dtype=torch.complex64, device=device)

        with pytest.raises(ValueError, match="no frequency bin"):
            empirical_aclr(x, oversampling=2.0, f_min=0.0, f_max=1e-9)

    def test_empty_out_of_band_is_zero(self, device):
        """An empty out-of-band is legitimate and yields an ACLR of zero.

        This is the default configuration: at ``oversampling=1`` the spectrum
        spans exactly the in-band, so there is no out-of-band power.
        """
        x = torch.randn(16, 128, dtype=torch.complex64, device=device)

        assert empirical_aclr(x).item() == 0.0
        assert empirical_aclr(x, f_min=-9.0, f_max=9.0).item() == 0.0
