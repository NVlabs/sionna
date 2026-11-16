#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Tests for sionna.phy.mimo.utils module."""

import pytest
import numpy as np
import torch

from sionna.phy import config
from sionna.phy.mimo.utils import (
    complex2real_vector,
    real2complex_vector,
    complex2real_matrix,
    real2complex_matrix,
    complex2real_covariance,
    real2complex_covariance,
    complex2real_channel,
    real2complex_channel,
    whiten_channel,
    List2LLRSimple,
)
from sionna.phy.utils import matrix_pinv


def exp_corr_mat(a: float, n: int, dtype=torch.complex64) -> torch.Tensor:
    """Generate exponential correlation matrix."""
    row = torch.pow(torch.tensor(a, dtype=dtype.to_real()), torch.arange(n, dtype=dtype.to_real()))
    mat = torch.zeros((n, n), dtype=dtype)
    for i in range(n):
        for j in range(n):
            mat[i, j] = row[abs(i - j)]
    return mat


class TestComplex2RealVector:
    """Tests for complex2real_vector and real2complex_vector."""

    @pytest.mark.parametrize("shape", [[1], [20, 1], [30, 20], [30, 20, 40]])
    def test_vector_transformation_roundtrip(self, shape, device, precision):
        """Test that complex2real and real2complex are inverses."""
        cdtype = torch.complex64 if precision == "single" else torch.complex128
        rdtype = torch.float32 if precision == "single" else torch.float64

        # Generate random complex vector
        z = torch.complex(
            torch.randn(shape, dtype=rdtype, device=device),
            torch.randn(shape, dtype=rdtype, device=device),
        )
        x = z.real
        y = z.imag

        # complex2real transformation
        zr = complex2real_vector(z)
        x_, y_ = zr.chunk(2, dim=-1)
        assert torch.allclose(x, x_)
        assert torch.allclose(y, y_)

        # real2complex transformation (roundtrip)
        zc = real2complex_vector(zr)
        assert torch.allclose(z, zc)

    def test_simple_example(self, device):
        """Test with a simple known example."""
        z = torch.complex(
            torch.tensor([1.0, 2.0], device=device),
            torch.tensor([3.0, 4.0], device=device),
        )
        zr = complex2real_vector(z)
        expected = torch.tensor([1.0, 2.0, 3.0, 4.0], device=device)
        assert torch.allclose(zr, expected)


class TestComplex2RealMatrix:
    """Tests for complex2real_matrix and real2complex_matrix."""

    @pytest.mark.parametrize(
        "shape",
        [[1, 1], [20, 1], [1, 20], [30, 20], [30, 20, 40], [12, 45, 64, 42]],
    )
    def test_matrix_transformation_roundtrip(self, shape, device, precision):
        """Test that complex2real and real2complex matrix transforms are inverses."""
        rdtype = torch.float32 if precision == "single" else torch.float64

        # Generate random complex matrix
        h = torch.complex(
            torch.randn(shape, dtype=rdtype, device=device),
            torch.randn(shape, dtype=rdtype, device=device),
        )
        h_r = h.real
        h_i = h.imag

        # complex2real transformation
        hr = complex2real_matrix(h)

        # Verify structure
        assert torch.allclose(h_r, hr[..., : shape[-2], : shape[-1]])
        assert torch.allclose(h_r, hr[..., shape[-2] :, shape[-1] :])
        assert torch.allclose(h_i, hr[..., shape[-2] :, : shape[-1]])
        assert torch.allclose(-h_i, hr[..., : shape[-2], shape[-1] :])

        # real2complex transformation (roundtrip)
        hc = real2complex_matrix(hr)
        assert torch.allclose(h, hc)


class TestComplex2RealCovariance:
    """Tests for complex2real_covariance and real2complex_covariance."""

    @pytest.mark.parametrize("n", [1, 2, 5, 13])
    @pytest.mark.parametrize("batch_shape", [[1], [5, 30], [4, 5, 10]])
    def test_covariance_transformation_roundtrip(self, n, batch_shape, device, precision):
        """Test that covariance transformations are inverses."""
        cdtype = torch.complex64 if precision == "single" else torch.complex128
        rdtype = torch.float32 if precision == "single" else torch.float64

        # Generate exponential correlation matrix
        a = torch.rand(batch_shape, dtype=rdtype, device=device)
        r = torch.stack(
            [exp_corr_mat(a.flatten()[i].item(), n, cdtype).to(device)
             for i in range(a.numel())],
            dim=0,
        ).reshape(*batch_shape, n, n)

        r_r = r.real / 2
        r_i = r.imag / 2

        # complex2real transformation
        rr = complex2real_covariance(r)
        assert torch.allclose(r_r, rr[..., :n, :n], atol=1e-5)
        assert torch.allclose(r_r, rr[..., n:, n:], atol=1e-5)
        assert torch.allclose(r_i, rr[..., n:, :n], atol=1e-5)
        assert torch.allclose(-r_i, rr[..., :n, n:], atol=1e-5)

        # real2complex transformation (roundtrip)
        rc = real2complex_covariance(rr)
        assert torch.allclose(r, rc, atol=1e-5)


class TestCovarianceStatistics:
    """Test that the statistics of the real-valued equivalent random vector match."""

    def test_covariance_statistics(self, device, precision):
        """Test that statistics of real-valued equivalent vector match target covariance."""
        if precision == "single":
            pytest.skip("Requires double precision for statistical accuracy")

        cdtype = torch.complex128
        rdtype = torch.float64

        batch_size = 100000
        num_batches = 10
        n = 8
        a = 0.8

        # Generate exponential correlation matrix
        r = exp_corr_mat(a, n, cdtype).to(device)
        rr = complex2real_covariance(r.unsqueeze(0)).squeeze(0)
        r_12 = torch.linalg.cholesky(r)

        r_hat = torch.zeros_like(rr)
        for _ in range(num_batches):
            # Generate correlated complex samples
            real_part = torch.randn(n, batch_size, dtype=rdtype, device=device)
            imag_part = torch.randn(n, batch_size, dtype=rdtype, device=device)
            w = torch.complex(real_part, imag_part) / np.sqrt(2)
            w = r_12 @ w  # [n, batch_size]
            w = w.T  # [batch_size, n]

            # Transform to real
            wr = complex2real_vector(w)  # [batch_size, 2n]

            # Estimate covariance
            r_batch = (wr.T @ wr) / batch_size
            r_hat = r_hat + r_batch / num_batches

        # Check that estimated covariance matches the real covariance
        max_err = (rr - r_hat).abs().max()
        assert max_err < 1e-2, f"Max covariance error {max_err} too large"


class TestComplex2RealChannel:
    """Tests for complex2real_channel and real2complex_channel."""

    def test_channel_transformation_roundtrip(self, device, precision):
        """Test that channel transformations are inverses."""
        rdtype = torch.float32 if precision == "single" else torch.float64
        cdtype = torch.complex64 if precision == "single" else torch.complex128

        batch_size = 10
        num_rx = 8
        num_tx = 4

        y = torch.complex(
            torch.randn(batch_size, num_rx, dtype=rdtype, device=device),
            torch.randn(batch_size, num_rx, dtype=rdtype, device=device),
        )
        h = torch.complex(
            torch.randn(batch_size, num_rx, num_tx, dtype=rdtype, device=device),
            torch.randn(batch_size, num_rx, num_tx, dtype=rdtype, device=device),
        )
        s = torch.eye(num_rx, dtype=cdtype, device=device).unsqueeze(0).expand(batch_size, -1, -1)

        # Transform to real
        yr, hr, sr = complex2real_channel(y, h, s)

        # Check shapes
        assert yr.shape == (batch_size, 2 * num_rx)
        assert hr.shape == (batch_size, 2 * num_rx, 2 * num_tx)
        assert sr.shape == (batch_size, 2 * num_rx, 2 * num_rx)

        # Transform back
        yc, hc, sc = real2complex_channel(yr, hr, sr)
        assert torch.allclose(y, yc, atol=1e-5)
        assert torch.allclose(h, hc, atol=1e-5)
        assert torch.allclose(s, sc, atol=1e-5)


class TestWhitenChannel:
    """Tests for whiten_channel."""

    @pytest.mark.parametrize("num_rx", [2, 4])
    def test_receiver_frame_preserves_noise_weighted_distances(
        self, device, precision, num_rx
    ):
        """Nonunitary receiver coordinates must transport noise covariance too."""
        cdtype = torch.complex64 if precision == "single" else torch.complex128
        rdtype = torch.float32 if precision == "single" else torch.float64

        real = torch.arange(
            1, num_rx * 2 + 1, dtype=rdtype, device=device
        ).reshape(num_rx, 2)
        h = torch.complex(
            real / (2 * num_rx), torch.flip(real, [0]) / (3 * num_rx)
        ).to(cdtype)

        c = torch.eye(num_rx, dtype=cdtype, device=device)
        c[0, 1] = 0.25 + 0.1j
        c[-1, 0] = -0.13 + 0.08j
        s = c @ c.mH

        w = torch.diag(
            torch.linspace(0.8, 1.4, num_rx, dtype=rdtype, device=device)
        ).to(cdtype)
        w[0, 1] = 0.2 + 0.1j
        w[-1, 0] = -0.1 + 0.05j
        w2 = torch.eye(num_rx, dtype=cdtype, device=device)
        w2[0, -1] = -0.08 + 0.12j
        w2[-1, -1] = 1.1 + 0.15j

        symbols = torch.tensor(
            [[1 + 0j, 1j], [-1 + 0j, 1j], [1j, -1j], [-1j, -1]],
            dtype=cdtype,
            device=device,
        )
        noise = torch.complex(
            torch.linspace(0.07, 0.17, num_rx, dtype=rdtype, device=device),
            torch.linspace(-0.05, 0.11, num_rx, dtype=rdtype, device=device),
        )
        y = h @ symbols[0] + noise

        def distances(yw, hw):
            residual = yw.unsqueeze(0) - symbols @ hw.T
            return residual.abs().square().sum(dim=-1)

        yw0, hw0 = whiten_channel(y, h, s, return_s=False)
        d0 = distances(yw0, hw0)
        gram0 = hw0.mH @ hw0
        tol = 7e-5 if precision == "single" else 1e-11

        # Independent arithmetic path: compare whitening against direct
        # covariance solves rather than using whitening as its own oracle.
        residual = y.unsqueeze(0) - symbols @ h.T
        direct_d = (
            residual.conj() * torch.linalg.solve(s, residual.T).T
        ).sum(dim=-1).real
        direct_gram = h.mH @ torch.linalg.solve(s, h)
        torch.testing.assert_close(d0, direct_d, rtol=0, atol=tol)
        torch.testing.assert_close(gram0, direct_gram, rtol=0, atol=tol)

        for transform in (w, w2 @ w):
            yp = transform @ y
            hp = transform @ h
            sp = transform @ s @ transform.mH
            yw, hw = whiten_channel(yp, hp, sp, return_s=False)

            # Recompute the transformed-frame observables independently too.
            residual_p = yp.unsqueeze(0) - symbols @ hp.T
            direct_d_p = (
                residual_p.conj() * torch.linalg.solve(sp, residual_p.T).T
            ).sum(dim=-1).real
            direct_gram_p = hp.mH @ torch.linalg.solve(sp, hp)

            torch.testing.assert_close(
                distances(yw, hw), direct_d_p, rtol=0, atol=tol
            )
            torch.testing.assert_close(
                hw.mH @ hw, direct_gram_p, rtol=0, atol=tol
            )
            torch.testing.assert_close(direct_d_p, d0, rtol=0, atol=tol)
            torch.testing.assert_close(direct_gram_p, gram0, rtol=0, atol=tol)

        # Negative control: a transformed receiver signal with an untransformed
        # covariance is a different physical noise model, not a change of frame.
        yw_bad, hw_bad = whiten_channel(w @ y, w @ h, s, return_s=False)
        assert (distances(yw_bad, hw_bad) - d0).abs().max() > 0.01

    def test_whiten_channel_identity_covariance(self, device, precision):
        """Test whitening with identity covariance (should be no-op on covariance)."""
        rdtype = torch.float32 if precision == "single" else torch.float64
        cdtype = torch.complex64 if precision == "single" else torch.complex128

        batch_size = 5
        num_rx = 4
        num_tx = 2

        y = torch.complex(
            torch.randn(batch_size, num_rx, dtype=rdtype, device=device),
            torch.randn(batch_size, num_rx, dtype=rdtype, device=device),
        )
        h = torch.complex(
            torch.randn(batch_size, num_rx, num_tx, dtype=rdtype, device=device),
            torch.randn(batch_size, num_rx, num_tx, dtype=rdtype, device=device),
        )
        s = torch.eye(num_rx, dtype=cdtype, device=device).unsqueeze(0).expand(batch_size, -1, -1).clone()

        yw, hw, sw = whiten_channel(y, h, s)

        # With identity covariance, y and h should be unchanged
        assert torch.allclose(y, yw, atol=1e-5)
        assert torch.allclose(h, hw, atol=1e-5)
        # sw should be identity
        assert torch.allclose(
            sw,
            torch.eye(num_rx, dtype=cdtype, device=device).unsqueeze(0).expand(batch_size, -1, -1),
            atol=1e-5,
        )

    def test_whiten_channel_return_s_false(self, device, precision):
        """Test whitening with return_s=False."""
        rdtype = torch.float32 if precision == "single" else torch.float64
        cdtype = torch.complex64 if precision == "single" else torch.complex128

        y = torch.complex(
            torch.randn(4, dtype=rdtype, device=device),
            torch.randn(4, dtype=rdtype, device=device),
        )
        h = torch.complex(
            torch.randn(4, 2, dtype=rdtype, device=device),
            torch.randn(4, 2, dtype=rdtype, device=device),
        )
        s = torch.eye(4, dtype=cdtype, device=device)

        result = whiten_channel(y, h, s, return_s=False)
        assert len(result) == 2

    def test_whiten_channel_symbol_recovery(self, device, precision):
        """Test that whitened channel can be used to recover symbols."""
        if precision == "single":
            pytest.skip("Requires double precision for numerical stability")

        rdtype = torch.float64
        cdtype = torch.complex128

        num_rx = 8
        num_tx = 4
        batch_size = 100

        # Random channel
        h = torch.complex(
            torch.randn(batch_size, num_rx, num_tx, dtype=rdtype, device=device),
            torch.randn(batch_size, num_rx, num_tx, dtype=rdtype, device=device),
        )

        # Random symbols (QPSK-like)
        x = torch.complex(
            torch.randn(batch_size, num_tx, dtype=rdtype, device=device),
            torch.randn(batch_size, num_tx, dtype=rdtype, device=device),
        )

        # Noise-free transmission
        y = (h @ x.unsqueeze(-1)).squeeze(-1)

        # Arbitrary covariance
        s = exp_corr_mat(0.5, num_rx, cdtype).to(device) + torch.eye(num_rx, dtype=cdtype, device=device)
        s = s.unsqueeze(0).expand(batch_size, -1, -1)

        # Whiten and recover
        yw, hw, sw = whiten_channel(y, h, s)
        xw = (matrix_pinv(hw) @ yw.unsqueeze(-1)).squeeze(-1)

        # Should recover x exactly for noise-free case
        assert torch.allclose(x, xw, atol=1e-5)

    def test_whiten_channel_noise_covariance(self, device, precision):
        """Test that noise covariance is correctly whitened."""
        if precision == "single":
            pytest.skip("Requires double precision for statistical accuracy")

        rdtype = torch.float64
        cdtype = torch.complex128

        num_rx = 8
        num_tx = 4
        batch_size = 50000
        num_batches = 10

        # Channel correlation matrix
        r = exp_corr_mat(0.8, num_rx, cdtype).to(device)
        r_12 = torch.linalg.cholesky(r)

        # Noise correlation matrix
        s = exp_corr_mat(0.5, num_rx, cdtype).to(device) + torch.eye(num_rx, dtype=cdtype, device=device)
        s_12 = torch.linalg.cholesky(s)

        err_accum = torch.zeros((num_rx, num_rx), dtype=cdtype, device=device)

        for _ in range(num_batches):
            # Generate random symbols (QPSK-like)
            x = torch.complex(
                torch.randn(batch_size, num_tx, dtype=rdtype, device=device),
                torch.randn(batch_size, num_tx, dtype=rdtype, device=device),
            )
            x = x / x.abs().clamp(min=1e-10)  # Normalize

            # Generate correlated channel
            h_white = torch.complex(
                torch.randn(batch_size, num_rx, num_tx, dtype=rdtype, device=device),
                torch.randn(batch_size, num_rx, num_tx, dtype=rdtype, device=device),
            ) / np.sqrt(2)
            h = r_12.unsqueeze(0) @ h_white

            # Generate correlated noise
            w_white = torch.complex(
                torch.randn(batch_size, num_rx, dtype=rdtype, device=device),
                torch.randn(batch_size, num_rx, dtype=rdtype, device=device),
            ) / np.sqrt(2)
            w = (s_12 @ w_white.unsqueeze(-1)).squeeze(-1)

            # Received signal
            hx = (h @ x.unsqueeze(-1)).squeeze(-1)
            y = hx + w

            # Whiten channel
            yw, hw, sw = whiten_channel(y, h, s.unsqueeze(0).expand(batch_size, -1, -1))

            # Compute recovered noise
            hwx = (hw @ x.unsqueeze(-1)).squeeze(-1)
            ww = yw - hwx

            # Estimate noise covariance
            err = (ww.mH @ ww) / batch_size - sw[0]
            err_accum = err_accum + err / num_batches

        # Check that noise covariance after whitening is close to identity
        max_err = err_accum.abs().max()
        assert max_err < 0.02, f"Max noise covariance error {max_err} too large"


class TestList2LLRSimple:
    """Tests for List2LLRSimple."""

    def test_list2llr_simple_shape(self, device, precision):
        """Test that List2LLRSimple produces correct output shape."""
        num_bits_per_symbol = 4
        num_streams = 2
        num_paths = 16
        batch_size = 8

        list2llr = List2LLRSimple(
            num_bits_per_symbol=num_bits_per_symbol,
            precision=precision,
            device=device,
        )

        rdtype = torch.float32 if precision == "single" else torch.float64
        cdtype = torch.complex64 if precision == "single" else torch.complex128

        y = torch.complex(
            torch.randn(batch_size, num_streams, dtype=rdtype, device=device),
            torch.randn(batch_size, num_streams, dtype=rdtype, device=device),
        )
        r = torch.complex(
            torch.randn(batch_size, num_streams, num_streams, dtype=rdtype, device=device),
            torch.randn(batch_size, num_streams, num_streams, dtype=rdtype, device=device),
        )
        dists = torch.rand(batch_size, num_paths, dtype=rdtype, device=device)
        path_inds = torch.randint(
            0, 2**num_bits_per_symbol, (batch_size, num_paths, num_streams),
            device=device, dtype=torch.int32
        )
        path_syms = torch.complex(
            torch.randn(batch_size, num_paths, num_streams, dtype=rdtype, device=device),
            torch.randn(batch_size, num_paths, num_streams, dtype=rdtype, device=device),
        )

        llr = list2llr(y, r, dists, path_inds, path_syms)

        assert llr.shape == (batch_size, num_streams, num_bits_per_symbol)

    def test_list2llr_simple_clipping(self, device, precision):
        """Test that LLRs are clipped correctly."""
        num_bits_per_symbol = 2
        llr_clip_val = 10.0

        list2llr = List2LLRSimple(
            num_bits_per_symbol=num_bits_per_symbol,
            llr_clip_val=llr_clip_val,
            precision=precision,
            device=device,
        )

        rdtype = torch.float32 if precision == "single" else torch.float64

        # Create paths with only one symbol index (all 0s)
        # This should result in extreme LLRs that get clipped
        batch_size = 4
        num_paths = 4
        num_streams = 1

        y = torch.randn(batch_size, num_streams, dtype=rdtype, device=device)
        r = torch.randn(batch_size, num_streams, num_streams, dtype=rdtype, device=device)
        dists = torch.rand(batch_size, num_paths, dtype=rdtype, device=device)
        path_inds = torch.zeros(batch_size, num_paths, num_streams, device=device, dtype=torch.int32)
        path_syms = torch.randn(batch_size, num_paths, num_streams, dtype=rdtype, device=device)

        llr = list2llr(y, r, dists, path_inds, path_syms)

        # All LLRs should be within clip bounds
        assert llr.abs().max() <= llr_clip_val

    def test_list2llr_simple_property(self, device, precision):
        """Test llr_clip_val property getter/setter."""
        list2llr = List2LLRSimple(
            num_bits_per_symbol=4,
            llr_clip_val=20.0,
            precision=precision,
            device=device,
        )

        assert list2llr.llr_clip_val == 20.0
        list2llr.llr_clip_val = 15.0
        assert list2llr.llr_clip_val == 15.0

