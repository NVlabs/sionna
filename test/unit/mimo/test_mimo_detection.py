#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Tests for sionna.phy.mimo.detection module."""

import pytest
import numpy as np
import torch

from sionna.phy.mimo.detection import (
    LinearDetector,
    MaximumLikelihoodDetector,
    KBestDetector,
    EPDetector,
    MMSEPICDetector,
)
from sionna.phy.mapping import Constellation
from sionna.phy.utils import complex_normal


class TestLinearDetector:
    """Tests for LinearDetector class."""

    @pytest.mark.parametrize("equalizer", ["lmmse", "zf", "mf"])
    @pytest.mark.parametrize("output", ["bit", "symbol"])
    def test_linear_detector_shape(self, device, precision, equalizer, output):
        """Test that LinearDetector produces correct output shape."""
        cdtype = torch.complex64 if precision == "single" else torch.complex128

        batch_size = 4
        num_rx = 8
        num_streams = 4
        num_bits_per_symbol = 4

        detector = LinearDetector(
            equalizer=equalizer,
            output=output,
            demapping_method="app",
            constellation_type="qam",
            num_bits_per_symbol=num_bits_per_symbol,
            precision=precision,
            device=device,
        )

        y = complex_normal((batch_size, num_rx), precision=precision, device=device)
        h = complex_normal((batch_size, num_rx, num_streams), precision=precision, device=device)
        s = torch.eye(num_rx, dtype=cdtype, device=device).unsqueeze(0).expand(batch_size, -1, -1)

        z = detector(y, h, s)

        if output == "bit":
            assert z.shape == (batch_size, num_streams, num_bits_per_symbol)
        else:
            num_points = 2 ** num_bits_per_symbol
            assert z.shape == (batch_size, num_streams, num_points)

    def test_linear_detector_hard_out(self, device, precision):
        """Test LinearDetector with hard output."""
        cdtype = torch.complex64 if precision == "single" else torch.complex128

        batch_size = 4
        num_rx = 6
        num_streams = 2
        num_bits_per_symbol = 2

        detector = LinearDetector(
            equalizer="lmmse",
            output="bit",
            demapping_method="app",
            constellation_type="qam",
            num_bits_per_symbol=num_bits_per_symbol,
            hard_out=True,
            precision=precision,
            device=device,
        )

        y = complex_normal((batch_size, num_rx), precision=precision, device=device)
        h = complex_normal((batch_size, num_rx, num_streams), precision=precision, device=device)
        s = torch.eye(num_rx, dtype=cdtype, device=device).unsqueeze(0).expand(batch_size, -1, -1)

        z = detector(y, h, s)

        # Hard output should be 0 or 1
        assert z.shape == (batch_size, num_streams, num_bits_per_symbol)
        assert ((z == 0) | (z == 1)).all()

    def test_invalid_equalizer(self, device):
        """Test that an unsupported equalizer is rejected."""
        with pytest.raises(ValueError, match="equalizer"):
            LinearDetector(
                equalizer="invalid",
                output="bit",
                demapping_method="app",
                constellation_type="qam",
                num_bits_per_symbol=2,
                device=device,
            )

    def test_invalid_output(self, device):
        """Test that an unsupported output is rejected."""
        with pytest.raises(ValueError, match="output"):
            LinearDetector(
                equalizer="lmmse",
                output=1,
                demapping_method="app",
                constellation_type="qam",
                num_bits_per_symbol=2,
                device=device,
            )


class TestMaximumLikelihoodDetector:
    """Tests for MaximumLikelihoodDetector class."""

    @pytest.mark.parametrize("output", ["bit", "symbol"])
    @pytest.mark.parametrize("demapping_method", ["app", "maxlog"])
    def test_ml_detector_shape(self, device, precision, output, demapping_method):
        """Test that ML detector produces correct output shape."""
        cdtype = torch.complex64 if precision == "single" else torch.complex128

        batch_size = 4
        num_rx = 4
        num_streams = 2
        num_bits_per_symbol = 2

        detector = MaximumLikelihoodDetector(
            output=output,
            demapping_method=demapping_method,
            num_streams=num_streams,
            constellation_type="qam",
            num_bits_per_symbol=num_bits_per_symbol,
            precision=precision,
            device=device,
        )

        y = complex_normal((batch_size, num_rx), precision=precision, device=device)
        h = complex_normal((batch_size, num_rx, num_streams), precision=precision, device=device)
        s = torch.eye(num_rx, dtype=cdtype, device=device).unsqueeze(0).expand(batch_size, -1, -1)

        z = detector(y, h, s)

        if output == "bit":
            assert z.shape == (batch_size, num_streams, num_bits_per_symbol)
        else:
            num_points = 2 ** num_bits_per_symbol
            assert z.shape == (batch_size, num_streams, num_points)

    def test_ml_detector_hard_out_symbol(self, device, precision):
        """Test ML detector with hard symbol output."""
        cdtype = torch.complex64 if precision == "single" else torch.complex128

        batch_size = 4
        num_rx = 4
        num_streams = 2
        num_bits_per_symbol = 2

        detector = MaximumLikelihoodDetector(
            output="symbol",
            demapping_method="maxlog",
            num_streams=num_streams,
            constellation_type="qam",
            num_bits_per_symbol=num_bits_per_symbol,
            hard_out=True,
            precision=precision,
            device=device,
        )

        y = complex_normal((batch_size, num_rx), precision=precision, device=device)
        h = complex_normal((batch_size, num_rx, num_streams), precision=precision, device=device)
        s = torch.eye(num_rx, dtype=cdtype, device=device).unsqueeze(0).expand(batch_size, -1, -1)

        z = detector(y, h, s)

        assert z.shape == (batch_size, num_streams)
        assert z.dtype == torch.int32


class TestKBestDetector:
    """Tests for KBestDetector class."""

    def test_kbest_wrong_parameters(self, device, precision):
        """Test that wrong parameters raise errors."""
        with pytest.raises(ValueError):
            # Neither constellation nor constellation_type
            KBestDetector("bit", 4, 16, precision=precision, device=device)

        with pytest.raises(ValueError):
            # Missing num_bits_per_symbol
            KBestDetector("bit", 4, 16, constellation_type="qam", precision=precision, device=device)

        with pytest.raises(ValueError):
            # Missing constellation_type
            KBestDetector("bit", 4, 16, num_bits_per_symbol=4, precision=precision, device=device)

    @pytest.mark.parametrize("use_real_rep", [True, False])
    def test_kbest_detector_shape(self, device, precision, use_real_rep):
        """Test that KBest detector produces correct output shape."""
        cdtype = torch.complex64 if precision == "single" else torch.complex128

        batch_size = 4
        num_rx = 6
        num_streams = 2
        num_bits_per_symbol = 4
        k = 16

        detector = KBestDetector(
            output="bit",
            num_streams=num_streams,
            k=k,
            constellation_type="qam",
            num_bits_per_symbol=num_bits_per_symbol,
            use_real_rep=use_real_rep,
            precision=precision,
            device=device,
        )

        y = complex_normal((batch_size, num_rx), precision=precision, device=device)
        h = complex_normal((batch_size, num_rx, num_streams), precision=precision, device=device)
        s = torch.eye(num_rx, dtype=cdtype, device=device).unsqueeze(0).expand(batch_size, -1, -1)

        llr = detector(y, h, s)

        assert llr.shape == (batch_size, num_streams, num_bits_per_symbol)

    def test_kbest_detector_hard_out(self, device, precision):
        """Test KBest detector with hard output."""
        cdtype = torch.complex64 if precision == "single" else torch.complex128

        batch_size = 4
        num_rx = 6
        num_streams = 2
        num_bits_per_symbol = 4
        k = 16

        detector = KBestDetector(
            output="bit",
            num_streams=num_streams,
            k=k,
            constellation_type="qam",
            num_bits_per_symbol=num_bits_per_symbol,
            hard_out=True,
            precision=precision,
            device=device,
        )

        y = complex_normal((batch_size, num_rx), precision=precision, device=device)
        h = complex_normal((batch_size, num_rx, num_streams), precision=precision, device=device)
        s = torch.eye(num_rx, dtype=cdtype, device=device).unsqueeze(0).expand(batch_size, -1, -1)

        bits = detector(y, h, s)

        assert bits.shape == (batch_size, num_streams, num_bits_per_symbol)
        assert ((bits == 0) | (bits == 1)).all()

    @pytest.mark.parametrize("use_real_rep", [True, False])
    @pytest.mark.parametrize(
        ("output", "hard_out", "expected_shape"),
        [
            ("bit", False, (2, 2)),
            ("bit", True, (2, 2)),
            ("symbol", True, (2,)),
        ],
    )
    def test_kbest_detector_unbatched(
        self,
        device,
        precision,
        use_real_rep,
        output,
        hard_out,
        expected_shape,
    ):
        """KBest supports the documented input shapes without a batch axis."""
        cdtype = torch.complex64 if precision == "single" else torch.complex128
        detector = KBestDetector(
            output=output,
            num_streams=2,
            k=16,
            constellation_type="qam",
            num_bits_per_symbol=2,
            hard_out=hard_out,
            use_real_rep=use_real_rep,
            precision=precision,
            device=device,
        )
        y = complex_normal((4,), precision=precision, device=device)
        h = complex_normal((4, 2), precision=precision, device=device)
        s = torch.eye(4, dtype=cdtype, device=device)

        result = detector(y, h, s)

        assert result.shape == expected_shape

    @pytest.mark.gpu
    def test_kbest_detector_unbatched_compiles(self, device):
        """Compiled KBest preserves unbatched output rank and values."""
        if not device.startswith("cuda"):
            pytest.skip("This compile regression requires a CUDA device")

        detector = KBestDetector(
            output="bit",
            num_streams=2,
            k=4,
            constellation_type="qam",
            num_bits_per_symbol=2,
            device=device,
        )
        y = complex_normal((4,), device=device)
        h = complex_normal((4, 2), device=device)
        s = torch.eye(4, dtype=torch.complex64, device=device)
        expected = detector(y, h, s)

        actual = torch.compile(detector)(y, h, s)

        assert actual.shape == (2, 2)
        torch.testing.assert_close(actual, expected)

    @pytest.mark.gpu
    def test_kbest_reduce_overhead_cuda_graph(self, device):
        """Test KBest through CUDA graph recording and replay."""
        if not device.startswith("cuda"):
            pytest.skip("CUDA graph capture requires a CUDA device")

        batch_size = 4
        num_rx = 6
        num_streams = 2

        detector = KBestDetector(
            output="bit",
            num_streams=num_streams,
            k=16,
            constellation_type="qam",
            num_bits_per_symbol=4,
            device=device,
        )

        y = complex_normal((batch_size, num_rx), device=device)
        h = complex_normal((batch_size, num_rx, num_streams), device=device)
        s = torch.eye(num_rx, dtype=torch.complex64, device=device)
        s = s.unsqueeze(0).expand(batch_size, -1, -1)

        compiled_detector = torch.compile(detector, mode="reduce-overhead")
        outputs = []
        for _ in range(3):
            outputs.append(compiled_detector(y, h, s).clone())
            torch.cuda.synchronize(device)

        for output in outputs:
            assert torch.isfinite(output).all()
            torch.testing.assert_close(output, outputs[0])

    def test_kbest_detector_k_warning(self, device, precision):
        """Test that a warning is issued when k is too large."""
        with pytest.warns(UserWarning, match="larger than the possible maximum"):
            KBestDetector(
                output="bit",
                num_streams=2,
                k=10000,  # Too large for 2 streams with 4 bits per symbol
                constellation_type="qam",
                num_bits_per_symbol=4,
                precision=precision,
                device=device,
            )


class TestEPDetector:
    """Tests for EPDetector class."""

    @pytest.mark.parametrize("output", ["bit", "symbol"])
    def test_ep_detector_shape(self, device, precision, output):
        """Test that EP detector produces correct output shape."""
        cdtype = torch.complex64 if precision == "single" else torch.complex128

        batch_size = 4
        num_rx = 6
        num_streams = 2
        num_bits_per_symbol = 4

        detector = EPDetector(
            output=output,
            num_bits_per_symbol=num_bits_per_symbol,
            l=5,
            precision=precision,
            device=device,
        )

        y = complex_normal((batch_size, num_rx), precision=precision, device=device)
        h = complex_normal((batch_size, num_rx, num_streams), precision=precision, device=device)
        s = torch.eye(num_rx, dtype=cdtype, device=device).unsqueeze(0).expand(batch_size, -1, -1)

        z = detector(y, h, s)

        if output == "bit":
            assert z.shape == (batch_size, num_streams, num_bits_per_symbol)
        else:
            num_points = 2 ** num_bits_per_symbol
            assert z.shape == (batch_size, num_streams, num_points)

    def test_ep_detector_hard_out(self, device, precision):
        """Test EP detector with hard output."""
        cdtype = torch.complex64 if precision == "single" else torch.complex128

        batch_size = 4
        num_rx = 6
        num_streams = 2
        num_bits_per_symbol = 4

        detector = EPDetector(
            output="symbol",
            num_bits_per_symbol=num_bits_per_symbol,
            hard_out=True,
            l=3,
            precision=precision,
            device=device,
        )

        y = complex_normal((batch_size, num_rx), precision=precision, device=device)
        h = complex_normal((batch_size, num_rx, num_streams), precision=precision, device=device)
        s = torch.eye(num_rx, dtype=cdtype, device=device).unsqueeze(0).expand(batch_size, -1, -1)

        z = detector(y, h, s)

        assert z.shape == (batch_size, num_streams)
        assert z.dtype == torch.int32


class TestMMSEPICDetector:
    """Tests for MMSEPICDetector class."""

    @pytest.mark.parametrize("output", ["bit", "symbol"])
    @pytest.mark.parametrize("num_iter", [1, 3])
    def test_mmse_pic_detector_shape(self, device, precision, output, num_iter):
        """Test that MMSE PIC detector produces correct output shape."""
        cdtype = torch.complex64 if precision == "single" else torch.complex128
        rdtype = torch.float32 if precision == "single" else torch.float64

        batch_size = 4
        num_rx = 6
        num_streams = 2
        num_bits_per_symbol = 4

        detector = MMSEPICDetector(
            output=output,
            demapping_method="maxlog",
            num_iter=num_iter,
            constellation_type="qam",
            num_bits_per_symbol=num_bits_per_symbol,
            precision=precision,
            device=device,
        )

        y = complex_normal((batch_size, num_rx), precision=precision, device=device)
        h = complex_normal((batch_size, num_rx, num_streams), precision=precision, device=device)
        s = torch.eye(num_rx, dtype=cdtype, device=device).unsqueeze(0).expand(batch_size, -1, -1)

        if output == "bit":
            prior = torch.zeros(batch_size, num_streams, num_bits_per_symbol, dtype=rdtype, device=device)
        else:
            num_points = 2 ** num_bits_per_symbol
            prior = torch.zeros(batch_size, num_streams, num_points, dtype=rdtype, device=device)

        z = detector(y, h, s, prior)

        if output == "bit":
            assert z.shape == (batch_size, num_streams, num_bits_per_symbol)
        else:
            num_points = 2 ** num_bits_per_symbol
            assert z.shape == (batch_size, num_streams, num_points)

    def test_mmse_pic_detector_hard_out(self, device, precision):
        """Test MMSE PIC detector with hard output."""
        cdtype = torch.complex64 if precision == "single" else torch.complex128
        rdtype = torch.float32 if precision == "single" else torch.float64

        batch_size = 4
        num_rx = 6
        num_streams = 2
        num_bits_per_symbol = 4

        detector = MMSEPICDetector(
            output="bit",
            demapping_method="maxlog",
            num_iter=2,
            constellation_type="qam",
            num_bits_per_symbol=num_bits_per_symbol,
            hard_out=True,
            precision=precision,
            device=device,
        )

        y = complex_normal((batch_size, num_rx), precision=precision, device=device)
        h = complex_normal((batch_size, num_rx, num_streams), precision=precision, device=device)
        s = torch.eye(num_rx, dtype=cdtype, device=device).unsqueeze(0).expand(batch_size, -1, -1)
        prior = torch.zeros(batch_size, num_streams, num_bits_per_symbol, dtype=rdtype, device=device)

        z = detector(y, h, s, prior)

        assert z.shape == (batch_size, num_streams, num_bits_per_symbol)
        assert ((z == 0) | (z == 1)).all()


class TestKBestZeroNoise:
    """Tests for KBestDetector on zero/near-zero noise channels (adapted from TF tests)."""

    @pytest.mark.parametrize("num_bits_per_symbol", [2, 4])
    @pytest.mark.parametrize("use_real_rep", [True, False])
    def test_kbest_symbol_errors_zero_noise(self, device, precision, num_bits_per_symbol, use_real_rep):
        """Test that we get no symbol errors on noise-free channel."""
        cdtype = torch.complex64 if precision == "single" else torch.complex128
        rdtype = torch.float32 if precision == "single" else torch.float64

        num_streams = 3
        num_rx = 7
        batch_size = 50
        k = 64

        # Create detector
        detector = KBestDetector(
            output="symbol",
            num_streams=num_streams,
            k=k,
            constellation_type="qam",
            num_bits_per_symbol=num_bits_per_symbol,
            use_real_rep=use_real_rep,
            hard_out=True,
            precision=precision,
            device=device,
        )

        # Generate random symbols
        from sionna.phy.mapping import Constellation
        const = Constellation("qam", num_bits_per_symbol, precision=precision, device=device)
        points = const()
        num_points = len(points)
        x_ind = torch.randint(0, num_points, (batch_size, num_streams), device=device)
        x = points[x_ind]

        # Random channel
        h = complex_normal((batch_size, num_rx, num_streams), precision=precision, device=device)

        # Noise-free transmission
        y = (h @ x.unsqueeze(-1)).squeeze(-1)

        # Very small noise covariance
        s = torch.eye(num_rx, dtype=cdtype, device=device) * 1e-9
        s = s.unsqueeze(0).expand(batch_size, -1, -1)

        # Detect
        x_ind_hat = detector(y, h, s)

        # Should have zero symbol errors
        ser = (x_ind != x_ind_hat).float().mean()
        assert ser == 0.0, f"Symbol error rate should be 0, got {ser}"


class TestEPZeroNoise:
    """Tests for EPDetector on zero/near-zero noise channels (adapted from TF tests)."""

    @pytest.mark.parametrize("num_bits_per_symbol", [2, 4])
    def test_ep_symbol_errors_zero_noise(self, device, precision, num_bits_per_symbol):
        """Test that we get no symbol errors on noise-free channel."""
        cdtype = torch.complex64 if precision == "single" else torch.complex128

        num_streams = 3
        num_rx = 7
        batch_size = 50

        # Create detector
        detector = EPDetector(
            output="symbol",
            num_bits_per_symbol=num_bits_per_symbol,
            hard_out=True,
            l=10,
            precision=precision,
            device=device,
        )

        # Generate random symbols
        from sionna.phy.mapping import Constellation
        const = Constellation("qam", num_bits_per_symbol, precision=precision, device=device)
        points = const()
        num_points = len(points)
        x_ind = torch.randint(0, num_points, (batch_size, num_streams), device=device)
        x = points[x_ind]

        # Random channel
        h = complex_normal((batch_size, num_rx, num_streams), precision=precision, device=device)

        # Noise-free transmission
        y = (h @ x.unsqueeze(-1)).squeeze(-1)

        # Very small noise covariance
        s = torch.eye(num_rx, dtype=cdtype, device=device) * 1e-4
        s = s.unsqueeze(0).expand(batch_size, -1, -1)

        # Detect
        x_ind_hat = detector(y, h, s)

        # Should have zero symbol errors
        ser = (x_ind != x_ind_hat).float().mean()
        assert ser == 0.0, f"Symbol error rate should be 0, got {ser}"


class TestDetectorConsistency:
    """Tests for consistency between different detectors."""

    def test_ml_kbest_consistency(self, device, precision):
        """Test that KBest with full search matches ML detector."""
        cdtype = torch.complex64 if precision == "single" else torch.complex128
        atol = 1e-3 if precision == "single" else 1e-6

        batch_size = 4
        num_rx = 4
        num_streams = 2
        num_bits_per_symbol = 2  # QPSK
        k = 4 ** 2  # Full search for 2 streams with QPSK

        ml_detector = MaximumLikelihoodDetector(
            output="bit",
            demapping_method="maxlog",
            num_streams=num_streams,
            constellation_type="qam",
            num_bits_per_symbol=num_bits_per_symbol,
            precision=precision,
            device=device,
        )

        kbest_detector = KBestDetector(
            output="bit",
            num_streams=num_streams,
            k=k,
            constellation_type="qam",
            num_bits_per_symbol=num_bits_per_symbol,
            precision=precision,
            device=device,
        )

        y = complex_normal((batch_size, num_rx), precision=precision, device=device)
        h = complex_normal((batch_size, num_rx, num_streams), precision=precision, device=device)
        s = torch.eye(num_rx, dtype=cdtype, device=device).unsqueeze(0).expand(batch_size, -1, -1)

        llr_ml = ml_detector(y, h, s)
        llr_kbest = kbest_detector(y, h, s)

        # KBest with full search should give similar results to ML
        # Note: exact match is not expected due to different algorithms
        # but the signs should generally agree
        sign_agreement = ((llr_ml > 0) == (llr_kbest > 0)).float().mean()
        assert sign_agreement > 0.8  # At least 80% agreement

    @pytest.mark.parametrize("num_bits_per_symbol", [2, 4])
    def test_kbest_llr_matches_ml(self, device, precision, num_bits_per_symbol):
        """Test that KBest with full search produces LLRs matching ML detector.

        Note: This test uses use_real_rep=False because the complex and real
        representations use different algorithms that produce equivalent but not
        identical LLR values at high SNR due to numerical scaling differences.
        """
        cdtype = torch.complex64 if precision == "single" else torch.complex128
        atol = 1e-4 if precision == "single" else 1e-4

        num_streams = 2
        num_rx = 6
        batch_size = 20
        k = (2 ** num_bits_per_symbol) ** num_streams  # Full search

        ml_detector = MaximumLikelihoodDetector(
            output="bit",
            demapping_method="maxlog",
            num_streams=num_streams,
            constellation_type="qam",
            num_bits_per_symbol=num_bits_per_symbol,
            precision=precision,
            device=device,
        )

        kbest_detector = KBestDetector(
            output="bit",
            num_streams=num_streams,
            k=k,
            constellation_type="qam",
            num_bits_per_symbol=num_bits_per_symbol,
            use_real_rep=False,  # Must use complex rep for exact match with ML
            precision=precision,
            device=device,
        )
        # Disable LLR clipping for comparison
        kbest_detector.list2llr.llr_clip_val = float('inf')

        # Random channel with some noise
        y = complex_normal((batch_size, num_rx), precision=precision, device=device)
        h = complex_normal((batch_size, num_rx, num_streams), precision=precision, device=device)
        no = 0.1
        s = torch.eye(num_rx, dtype=cdtype, device=device) * no
        s = s.unsqueeze(0).expand(batch_size, -1, -1)

        llr_ml = ml_detector(y, h, s)
        llr_kbest = kbest_detector(y, h, s)

        # LLRs should be close
        assert torch.allclose(llr_ml, llr_kbest, atol=atol)

