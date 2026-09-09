#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Tests for discrete channel models"""

import numpy as np
import pytest
import torch

from sionna.phy import dtypes
from sionna.phy.channel import (
    BinaryErasureChannel,
    BinaryMemorylessChannel,
    BinarySymmetricChannel,
    BinaryZChannel,
)
from sionna.phy.mapping import BinarySource


class TestDiscreteChannels:
    """General tests for all discrete channels."""

    def test_dtypes_float32_binary(self, device):
        """Test float32 input for binary mode without LLRs."""
        channels = [
            (BinarySymmetricChannel, 0.1),
            (BinaryZChannel, 0.1),
            (BinaryMemorylessChannel, (0.1, 0.1)),
        ]

        source = BinarySource(device=device, precision="single")
        for channel_cls, pb in channels:
            ch = channel_cls(
                bipolar_input=False,
                return_llrs=False,
                precision="single",
                device=device,
            )
            x = source((10, 11))
            y = ch(x, pb)
            assert y.dtype == torch.float32

    def test_dtypes_float64_binary(self, device):
        """Test float64 input for binary mode without LLRs."""
        channels = [
            (BinarySymmetricChannel, 0.1),
            (BinaryZChannel, 0.1),
            (BinaryMemorylessChannel, (0.1, 0.1)),
        ]

        source = BinarySource(device=device, precision="double")
        for channel_cls, pb in channels:
            ch = channel_cls(
                bipolar_input=False,
                return_llrs=False,
                precision="double",
                device=device,
            )
            x = source((10, 11))
            y = ch(x, pb)
            assert y.dtype == torch.float64

    def test_dtypes_int_binary(self, device):
        """Test integer dtypes for binary mode without LLRs."""
        int_dtypes = [torch.int8, torch.int16, torch.int32, torch.int64]

        channels = [
            (BinarySymmetricChannel, 0.1),
            (BinaryZChannel, 0.1),
            (BinaryMemorylessChannel, (0.1, 0.1)),
        ]

        source = BinarySource(device=device)
        for channel_cls, pb in channels:
            ch = channel_cls(
                bipolar_input=False,
                return_llrs=False,
                device=device,
            )
            for dt in int_dtypes:
                x = source((10, 11)).to(dtype=dt)
                y = ch(x, pb)
                assert y.dtype == dt

    def test_dtypes_uint8_binary(self, device):
        """Test uint8 dtype for binary mode without LLRs (not BEC)."""
        channels = [
            (BinarySymmetricChannel, 0.1),
            (BinaryZChannel, 0.1),
            (BinaryMemorylessChannel, (0.1, 0.1)),
        ]

        source = BinarySource(device=device)
        for channel_cls, pb in channels:
            ch = channel_cls(
                bipolar_input=False,
                return_llrs=False,
                device=device,
            )
            x = source((10, 11)).to(dtype=torch.uint8)
            y = ch(x, pb)
            assert y.dtype == torch.uint8

    def test_dtypes_bec_signed_only(self, device):
        """Test that BEC supports signed dtypes for binary mode without LLRs."""
        valid_dtypes = [
            torch.float32,
            torch.float64,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        ]

        source = BinarySource(device=device)
        for dt in valid_dtypes:
            precision = "double" if dt == torch.float64 else "single"
            ch = BinaryErasureChannel(
                bipolar_input=False,
                return_llrs=False,
                precision=precision,
                device=device,
            )
            x = source((10, 11)).to(dtype=dt, device=device)
            y = ch(x, 0.1)
            assert y.dtype == dt

    def test_dtypes_bipolar_signed(self, device):
        """Test that signed dtypes are supported for bipolar mode without LLRs."""
        signed_dtypes = [torch.int8, torch.int16, torch.int32, torch.int64]

        channels = [
            (BinarySymmetricChannel, 0.1),
            (BinaryZChannel, 0.1),
            (BinaryErasureChannel, 0.1),
            (BinaryMemorylessChannel, (0.1, 0.1)),
        ]

        source = BinarySource(device=device)
        for channel_cls, pb in channels:
            ch = channel_cls(
                bipolar_input=True,
                return_llrs=False,
                device=device,
            )
            for dt in signed_dtypes:
                x = source((10, 11))
                x = (2 * x - 1).to(dtype=dt, device=device)
                y = ch(x, pb)
                assert y.dtype == dt

    def test_dtypes_with_llr(self, device, precision):
        """Test that only float dtypes are supported when returning LLRs."""
        dtype = dtypes[precision]["torch"]["dtype"]

        channels = [
            (BinarySymmetricChannel, lambda: 0.1),
            (BinaryZChannel, lambda: 0.1),
            (BinaryErasureChannel, lambda: 0.1),
            (BinaryMemorylessChannel, lambda: (0.1, 0.1)),
        ]

        source = BinarySource(device=device, precision=precision)
        for channel_cls, get_pb in channels:
            for bipolar in [True, False]:
                ch = channel_cls(
                    bipolar_input=bipolar,
                    return_llrs=True,
                    precision=precision,
                    device=device,
                )

                x = source((10, 11))
                if bipolar:
                    x = 2 * x - 1
                pb = get_pb()
                y = ch(x, pb)
                assert y.dtype == dtype

    def test_llrs_monte_carlo(self, device):
        """Test LLR output against Monte Carlo based estimation."""
        num_samples = int(1e6)
        source = BinarySource(device=device, precision="double")
        channel = BinaryMemorylessChannel(
            return_llrs=False,
            bipolar_input=False,
            llr_max=20.0,
            device=device,
            precision="double",
        )
        channel_ref = BinaryMemorylessChannel(
            return_llrs=True,
            bipolar_input=False,
            llr_max=20.0,
            device=device,
            precision="double",
        )

        # Test different error probabilities
        pbs = [
            (0.0, 0.0),
            (0.1, 0.1),
            (0.5, 0.5),
            (0.99, 0.99),
            (0.1, 0.4),
            (0.0, 0.5),
            (0.01, 0.99),
        ]

        for pb in pbs:
            x = source((num_samples,))
            y = channel(x, pb)
            y_ref = channel_ref(x, pb)

            x_np = x.cpu().numpy()
            y_np = y.cpu().numpy()

            trans_mat = np.zeros((2, 2))
            for i in range(num_samples):
                trans_mat[int(x_np[i]), int(y_np[i])] += 1

            trans_mat /= num_samples

            # Calculate LLRs based on simulated probabilities
            eps = 1e-20
            l_0 = -(np.log(trans_mat[0, 0] + eps) - np.log(trans_mat[1, 0] + eps))
            l_1 = -(np.log(trans_mat[0, 1] + eps) - np.log(trans_mat[1, 1] + eps))

            # Remove nans from div-by-zeros
            l_0 = np.nan_to_num(l_0)
            l_1 = np.nan_to_num(l_1)

            # Clipping
            l_0 = np.clip(l_0, -20.0, 20.0)
            l_1 = np.clip(l_1, -20.0, 20.0)

            # Allow tolerance due to Monte Carlo sampling
            c1 = np.isclose(
                np.minimum(l_0, l_1),
                y_ref.min().cpu().numpy(),
                rtol=0.01,
                atol=0.1,
            )
            c2 = np.isclose(
                np.maximum(l_0, l_1),
                y_ref.max().cpu().numpy(),
                rtol=0.01,
                atol=0.1,
            )

            assert c1, f"LLR min mismatch for pb={pb}"
            assert c2, f"LLR max mismatch for pb={pb}"

    def test_gradient(self, device):
        """Test that channel is differentiable w.r.t pb."""
        bs = 10000
        channel = BinarySymmetricChannel(device=device)
        x = torch.zeros(bs, device=device)

        # Randomly initialized variable
        pb = torch.tensor(0.1, requires_grad=True, device=device)
        optimizer = torch.optim.Adam([pb], lr=0.01)

        # Approximate a target error rate via SGD
        target_ber = 0.4

        for _ in range(100):
            optimizer.zero_grad()
            y = channel(x, pb)
            loss = (y.mean() - target_ber) ** 2
            loss.backward()
            optimizer.step()

        assert np.isclose(pb.detach().cpu().numpy(), target_ber, rtol=0.1, atol=0.01)

    def test_pb_statistics_binary(self, device):
        """Test for correct error statistics with binary input."""
        bs = int(1e5)

        # Test different broadcastable shapes
        pbs = [
            (0.2, 0.4),
            (0.95, 0.2),
            (1.0, 0.0),
        ]

        source = BinarySource(device=device)
        for pb in pbs:
            channel = BinaryMemorylessChannel(bipolar_input=False, device=device)
            x = source((bs, 2, 4))
            y = channel(x, pb)

            # Count errors
            e = torch.where(x != y, 1.0, 0.0)

            # Evaluate x=0 and x=1 separately
            e0 = torch.where(x == 0, e, torch.zeros_like(e))
            e1 = torch.where(x != 0, e, torch.zeros_like(e))

            # Get per bit position BER
            ber0 = e0.sum(dim=0) / (x == 0).float().sum(dim=0)
            ber1 = e1.sum(dim=0) / (x != 0).float().sum(dim=0)

            # Allow mismatch due to Monte Carlo sampling
            assert torch.all(torch.abs(ber0 - pb[0]) < 0.01).item()
            assert torch.all(torch.abs(ber1 - pb[1]) < 0.01).item()

    def test_pb_statistics_bipolar(self, device):
        """Test for correct error statistics with bipolar input."""
        bs = int(1e5)

        pbs = [
            (0.2, 0.4),
            (0.95, 0.2),
        ]

        source = BinarySource(device=device)
        for pb in pbs:
            channel = BinaryMemorylessChannel(bipolar_input=True, device=device)
            x = source((bs, 2, 4))
            x = 2 * x - 1  # Convert to bipolar
            y = channel(x, pb)

            # Count errors
            e = torch.where(x != y, 1.0, 0.0)

            # Evaluate x=-1 and x=1 separately
            e0 = torch.where(x == -1, e, torch.zeros_like(e))
            e1 = torch.where(x != -1, e, torch.zeros_like(e))

            # Get per bit position BER
            ber0 = e0.sum(dim=0) / (x == -1).float().sum(dim=0)
            ber1 = e1.sum(dim=0) / (x != -1).float().sum(dim=0)

            # Allow mismatch due to Monte Carlo sampling
            assert torch.all(torch.abs(ber0 - pb[0]) < 0.01).item()
            assert torch.all(torch.abs(ber1 - pb[1]) < 0.01).item()

    def test_output_shape(self, device, precision):
        """Test that output shape matches input shape."""
        source = BinarySource(device=device, precision=precision)

        channels = [
            (BinarySymmetricChannel, 0.1),
            (BinaryZChannel, 0.1),
            (BinaryErasureChannel, 0.1),
            (BinaryMemorylessChannel, (0.1, 0.1)),
        ]

        for channel_cls, pb in channels:
            ch = channel_cls(precision=precision, device=device)
            x = source((32, 64, 16))
            y = ch(x, pb)
            assert y.shape == x.shape

    def test_device_placement(self, device, precision):
        """Test that output is on the correct device."""
        source = BinarySource(device=device, precision=precision)

        channels = [
            (BinarySymmetricChannel, 0.1),
            (BinaryZChannel, 0.1),
            (BinaryErasureChannel, 0.1),
            (BinaryMemorylessChannel, (0.1, 0.1)),
        ]

        for channel_cls, pb in channels:
            ch = channel_cls(precision=precision, device=device)
            x = source((32, 16))
            y = ch(x, pb)
            assert y.device == x.device


class TestBinarySymmetricChannel:
    """Tests specific to BinarySymmetricChannel."""

    def test_docstring_example(self, device):
        """Test that the example from the docstring works correctly."""
        channel = BinarySymmetricChannel(device=device)
        x = torch.randint(0, 2, (10, 100), dtype=torch.float32, device=device)
        pb = 0.1
        y = channel(x, pb)
        assert y.shape == torch.Size([10, 100])

    def test_symmetric_error_rate(self, device):
        """Test that error rate is symmetric for 0s and 1s."""
        bs = int(1e5)
        pb = 0.2
        source = BinarySource(device=device)
        channel = BinarySymmetricChannel(device=device)

        x = source((bs,))
        y = channel(x, pb)

        # Error rate for 0s
        mask0 = x == 0
        er0 = (y[mask0] != 0).float().mean()

        # Error rate for 1s
        mask1 = x == 1
        er1 = (y[mask1] != 1).float().mean()

        # Both should be close to pb
        assert torch.abs(er0 - pb) < 0.01
        assert torch.abs(er1 - pb) < 0.01


class TestBinaryZChannel:
    """Tests specific to BinaryZChannel."""

    def test_docstring_example(self, device):
        """Test that the example from the docstring works correctly."""
        channel = BinaryZChannel(device=device)
        x = torch.randint(0, 2, (10, 100), dtype=torch.float32, device=device)
        pb = 0.1
        y = channel(x, pb)
        assert y.shape == torch.Size([10, 100])

    def test_zero_never_flipped(self, device):
        """Test that zeros are never flipped in Z-channel."""
        bs = int(1e5)
        pb = 0.5
        channel = BinaryZChannel(device=device)

        x = torch.zeros(bs, device=device)
        y = channel(x, pb)

        # All outputs should remain 0
        assert (y == 0).all()

    def test_one_flipping_rate(self, device):
        """Test that ones are flipped at the correct rate."""
        bs = int(1e5)
        pb = 0.3
        channel = BinaryZChannel(device=device)

        x = torch.ones(bs, device=device)
        y = channel(x, pb)

        # Error rate should be close to pb
        er = (y == 0).float().mean()
        assert torch.abs(er - pb) < 0.01


class TestBinaryErasureChannel:
    """Tests specific to BinaryErasureChannel."""

    def test_docstring_example(self, device):
        """Test that the example from the docstring works correctly."""
        channel = BinaryErasureChannel(device=device)
        x = torch.randint(0, 2, (10, 100), dtype=torch.float32, device=device)
        pb = 0.1
        y = channel(x, pb)
        assert y.shape == torch.Size([10, 100])

    def test_erasure_statistics_scalar(self, device):
        """Test erasure rate matches specified probability for scalar pb."""
        bs = int(1e5)
        source = BinarySource(device=device)

        pbs = [0.2, 0.5, 0.95]

        for pb in pbs:
            for bipolar in [True, False]:
                channel = BinaryErasureChannel(bipolar_input=bipolar, device=device)
                x = source((bs,))
                if bipolar:
                    x = 2 * x - 1
                y = channel(x, pb)

                # Erasure indicator
                erased_element = 0.0 if bipolar else -1.0
                erasure_rate = (y == erased_element).float().mean()

                assert torch.abs(erasure_rate - pb) < 0.01

    def test_erasure_statistics_broadcastable(self, device):
        """Test erasure rate for broadcastable pb shapes."""
        bs = int(1e5)
        source = BinarySource(device=device)

        # Test different broadcastable shapes
        pbs = [
            torch.tensor([0.1, 0.1, 0.1, 0.8], device=device),
            torch.tensor([[0.2, 0.3, 0.1, 0.0], [0.2, 0.7, 0.2, 0.3]], device=device),
        ]

        for pb in pbs:
            for bipolar in [True, False]:
                channel = BinaryErasureChannel(bipolar_input=bipolar, device=device)
                x = source((bs, 2, 4))
                if bipolar:
                    x = 2 * x - 1
                y = channel(x, pb)

                erased_element = 0.0 if bipolar else -1.0
                e = torch.where(y == erased_element, 1.0, 0.0)
                ber = e.mean(dim=0)

                assert torch.all(torch.abs(ber - pb) < 0.01).item()

    def test_no_bit_flipping(self, device):
        """Test that BEC only erases, never flips bits."""
        bs = int(1e4)
        pb = 0.5
        source = BinarySource(device=device)
        channel = BinaryErasureChannel(device=device)

        x = source((bs,))
        y = channel(x, pb)

        # For non-erased positions, bits should match original
        non_erased = y != -1
        assert (x[non_erased] == y[non_erased]).all()


class TestBinaryMemorylessChannel:
    """Tests specific to BinaryMemorylessChannel."""

    def test_docstring_example(self, device):
        """Test that the example from the docstring works correctly."""
        channel = BinaryMemorylessChannel(device=device)
        x = torch.randint(0, 2, (10, 100), dtype=torch.float32, device=device)
        pb = (0.1, 0.2)
        y = channel(x, pb)
        assert y.shape == torch.Size([10, 100])

    def test_asymmetric_error_rates(self, device):
        """Test asymmetric error rates for 0s and 1s."""
        bs = int(1e5)
        pb0, pb1 = 0.1, 0.4
        source = BinarySource(device=device)
        channel = BinaryMemorylessChannel(device=device)

        x = source((bs,))
        y = channel(x, (pb0, pb1))

        # Error rate for 0s
        mask0 = x == 0
        er0 = (y[mask0] != 0).float().mean()

        # Error rate for 1s
        mask1 = x == 1
        er1 = (y[mask1] != 1).float().mean()

        assert torch.abs(er0 - pb0) < 0.01
        assert torch.abs(er1 - pb1) < 0.01

    def test_llr_max_property(self, device):
        """Test llr_max property getter and setter."""
        channel = BinaryMemorylessChannel(llr_max=50.0, device=device)
        llr_max: torch.Tensor = channel.llr_max
        assert llr_max.item() == 50.0

        channel.llr_max = 75.0
        llr_max = channel.llr_max
        assert llr_max.item() == 75.0

    def test_temperature_property(self, device):
        """Test temperature property getter and setter."""
        channel = BinaryMemorylessChannel(device=device)
        temp: torch.Tensor = channel.temperature
        assert temp.item() == pytest.approx(0.1)

        channel.temperature = 0.5
        temp = channel.temperature
        assert temp.item() == 0.5


class TestEdgeCases:
    """Test edge cases for discrete channels."""

    def test_zero_error_probability(self, device):
        """Test that zero error probability results in no errors."""
        bs = int(1e4)
        source = BinarySource(device=device)

        for channel_cls in [BinarySymmetricChannel, BinaryZChannel]:
            channel = channel_cls(device=device)
            x = source((bs,))
            y = channel(x, 0.0)

            # No errors expected
            assert (x == y).all()

    def test_one_error_probability_bsc(self, device):
        """Test BSC with probability 1 flips all bits."""
        bs = int(1e4)
        source = BinarySource(device=device)
        channel = BinarySymmetricChannel(device=device)

        x = source((bs,))
        y = channel(x, 1.0)

        # All bits should be flipped
        assert ((x == 0) & (y == 1)).sum() + ((x == 1) & (y == 0)).sum() == bs

    def test_one_error_probability_z_channel(self, device):
        """Test Z-channel with probability 1 flips all 1s."""
        bs = int(1e4)
        channel = BinaryZChannel(device=device)

        # All ones
        x = torch.ones(bs, device=device)
        y = channel(x, 1.0)

        # All should become 0
        assert (y == 0).all()

    def test_one_erasure_probability(self, device):
        """Test BEC with probability 1 erases all bits."""
        bs = int(1e4)
        source = BinarySource(device=device)
        channel = BinaryErasureChannel(device=device)

        x = source((bs,))
        y = channel(x, 1.0)

        # All should be erased (-1)
        assert (y == -1).all()

    def test_invalid_llr_max(self, device):
        """Test that negative llr_max raises error."""
        with pytest.raises(ValueError):
            BinaryMemorylessChannel(llr_max=-1.0, device=device)

    def test_invalid_temperature(self, device):
        """Test that negative temperature raises error."""
        channel = BinaryMemorylessChannel(device=device)
        with pytest.raises(ValueError):
            channel.temperature = -0.1

    def test_invalid_input_binary_check(self, device):
        """Test that non-binary input raises error."""
        channel = BinarySymmetricChannel(bipolar_input=False, device=device)
        x = torch.tensor([0.0, 0.5, 1.0], device=device)  # 0.5 is invalid
        with pytest.raises(ValueError):
            channel(x, 0.1)

    def test_invalid_input_bipolar_check(self, device):
        """Test that non-bipolar input raises error."""
        channel = BinarySymmetricChannel(bipolar_input=True, device=device)
        x = torch.tensor([-1.0, 0.0, 1.0], device=device)  # 0 is invalid
        with pytest.raises(ValueError):
            channel(x, 0.1)


class TestCompileMode:
    """Test torch.compile compatibility."""

    @pytest.mark.skipif(
        not hasattr(torch, "compile"), reason="torch.compile not available"
    )
    def test_bsc_compile(self, device, precision):
        """Test BinarySymmetricChannel with torch.compile."""
        source = BinarySource(device=device, precision=precision)
        channel = BinarySymmetricChannel(
            precision=precision,
            device=device,
        )

        @torch.compile(fullgraph=True, dynamic=True)
        def run_channel(x, pb):
            return channel(x, pb)

        for shape in ((100, 50), (80, 40)):
            x = source(shape)
            y = run_channel(x, 0.1)
            assert y.shape == x.shape

    @pytest.mark.skipif(
        not hasattr(torch, "compile"), reason="torch.compile not available"
    )
    def test_bec_compile(self, device, precision):
        """Test BinaryErasureChannel with torch.compile."""
        source = BinarySource(device=device, precision=precision)
        channel = BinaryErasureChannel(
            precision=precision,
            device=device,
        )

        @torch.compile(fullgraph=True, dynamic=True)
        def run_channel(x, pb):
            return channel(x, pb)

        for shape in ((100, 50), (80, 40)):
            x = source(shape)
            y = run_channel(x, 0.1)
            assert y.shape == x.shape


class TestBinaryMemorylessChannelContracts:
    """Regression tests for BMC pb validation and LLR endpoint numerics."""

    def test_pb_shape_vector_of_length_2(self, device):
        """Documented pb shape [2] must be accepted."""
        channel = BinaryMemorylessChannel(device=device)
        x = torch.zeros(8, device=device)
        pb = torch.tensor([0.1, 0.2], device=device)
        y = channel(x, pb)
        assert y.shape == x.shape

    def test_pb_shape_batch_n_2(self, device):
        """Broadcastable pb with trailing dim 2 must be accepted."""
        channel = BinaryMemorylessChannel(device=device)
        x = torch.zeros(2, 4, device=device)
        pb = torch.tensor(
            [[[0.1, 0.2], [0.05, 0.15], [0.2, 0.1], [0.0, 0.3]],
             [[0.3, 0.1], [0.1, 0.1], [0.25, 0.25], [0.4, 0.05]]],
            device=device,
        )
        y = channel(x, pb)
        assert y.shape == x.shape

    def test_pb_rejects_wrong_last_dim(self, device):
        """Malformed pb with last dim != 2 must raise ValueError."""
        channel = BinaryMemorylessChannel(device=device)
        x = torch.zeros(2, device=device)
        pb = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], device=device)
        with pytest.raises(ValueError, match="Last dimension of pb"):
            channel(x, pb)

    def test_pb_rejects_trailing_singleton(self, device):
        """pb ending in a singleton last dim must raise ValueError."""
        channel = BinaryMemorylessChannel(device=device)
        x = torch.zeros(2, device=device)
        pb = torch.tensor([[[0.1], [0.2]]], device=device)
        with pytest.raises(ValueError, match="Last dimension of pb"):
            channel(x, pb)

    def test_llr_endpoints_are_finite(self, device, precision):
        """LLRs at pb endpoints must be finite and clipped to llr_max."""
        llr_max = 20.0
        channel = BinaryMemorylessChannel(
            return_llrs=True,
            llr_max=llr_max,
            precision=precision,
            device=device,
        )
        x = torch.tensor(
            [0.0, 1.0], dtype=channel.dtype, device=device
        )
        for pb in (
            (0.0, 0.0),
            (1.0, 1.0),
            (0.0, 1.0),
            (1.0, 0.0),
            torch.tensor([0.0, 1.0], dtype=channel.dtype, device=device),
            torch.tensor([1.0, 0.0], dtype=channel.dtype, device=device),
        ):
            y = channel(x, pb)
            assert torch.isfinite(y).all(), f"non-finite LLRs for pb={pb}"
            assert (y.abs() <= llr_max + 1e-5).all()
            # Intermediate logs must also stay finite; finiteness must not
            # rely only on the final +/- llr_max clip.
            assert not torch.isinf(y).any()
