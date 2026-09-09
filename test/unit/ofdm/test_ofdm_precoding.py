#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Unit tests for OFDM Precoding classes."""

import numpy as np
import pytest
import torch

from sionna.phy.ofdm import (
    ResourceGrid,
    RZFPrecoder,
    RZFPrecodedChannel,
    CBFPrecodedChannel,
    EyePrecodedChannel,
)
from sionna.phy.mimo import StreamManagement, rzf_precoding_matrix
from sionna.phy.channel import (
    RayleighBlockFading,
    cir_to_ofdm_channel,
    subcarrier_frequencies,
)
from sionna.phy.utils import expand_to_rank


def create_test_setup(
    num_tx=2,
    num_rx_per_tx=2,
    num_streams_per_rx=2,
    device="cpu",
    precision="single",
):
    """Create a common test setup for precoding tests."""
    num_rx = num_tx * num_rx_per_tx
    num_rx_ant = num_streams_per_rx
    num_streams_per_tx = num_rx_per_tx * num_streams_per_rx
    num_tx_ant = num_streams_per_tx * 2

    rx_tx_association = np.zeros((num_rx, num_tx), dtype=np.int32)
    for j in range(num_tx):
        rx_tx_association[j * num_rx_per_tx : (j + 1) * num_rx_per_tx, j] = 1

    sm = StreamManagement(rx_tx_association, num_streams_per_tx)
    rg = ResourceGrid(
        num_ofdm_symbols=14,
        fft_size=64,
        subcarrier_spacing=15e3,
        num_tx=num_tx,
        num_streams_per_tx=num_streams_per_tx,
        precision=precision,
        device=device,
    )

    return {
        "sm": sm,
        "rg": rg,
        "num_tx": num_tx,
        "num_rx": num_rx,
        "num_rx_ant": num_rx_ant,
        "num_tx_ant": num_tx_ant,
        "num_streams_per_tx": num_streams_per_tx,
        "rx_tx_association": rx_tx_association,
    }


class TestRZFPrecoder:
    """Tests for the RZFPrecoder class."""

    def test_output_shapes(self, device, precision):
        """Test that RZFPrecoder produces correct output shapes."""
        setup = create_test_setup(device=device, precision=precision)
        sm, rg = setup["sm"], setup["rg"]
        num_tx = setup["num_tx"]
        num_rx = setup["num_rx"]
        num_rx_ant = setup["num_rx_ant"]
        num_tx_ant = setup["num_tx_ant"]
        num_streams_per_tx = setup["num_streams_per_tx"]

        precoder = RZFPrecoder(
            rg, sm, return_effective_channel=False,
            precision=precision, device=device
        )

        batch_size = 16
        cdtype = torch.complex64 if precision == "single" else torch.complex128

        x = torch.randn(
            batch_size, num_tx, num_streams_per_tx, 14, 64,
            dtype=cdtype, device=device
        )
        h = torch.randn(
            batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, 14, 64,
            dtype=cdtype, device=device
        )

        x_precoded = precoder(x, h, alpha=0.1)

        # Check output shape
        expected_shape = (batch_size, num_tx, num_tx_ant, 14, 64)
        assert x_precoded.shape == expected_shape

    def test_output_shapes_with_effective_channel(self, device, precision):
        """Test RZFPrecoder with effective channel output."""
        setup = create_test_setup(device=device, precision=precision)
        sm, rg = setup["sm"], setup["rg"]
        num_tx = setup["num_tx"]
        num_rx = setup["num_rx"]
        num_rx_ant = setup["num_rx_ant"]
        num_tx_ant = setup["num_tx_ant"]
        num_streams_per_tx = setup["num_streams_per_tx"]

        precoder = RZFPrecoder(
            rg, sm, return_effective_channel=True,
            precision=precision, device=device
        )

        batch_size = 16
        cdtype = torch.complex64 if precision == "single" else torch.complex128

        x = torch.randn(
            batch_size, num_tx, num_streams_per_tx, 14, 64,
            dtype=cdtype, device=device
        )
        h = torch.randn(
            batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, 14, 64,
            dtype=cdtype, device=device
        )

        x_precoded, h_eff = precoder(x, h, alpha=0.1)

        # Check output shapes
        assert x_precoded.shape == (batch_size, num_tx, num_tx_ant, 14, 64)
        # h_eff shape: [batch, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols, num_effective_subcarriers]
        assert h_eff.shape[0] == batch_size
        assert h_eff.shape[1] == num_rx
        assert h_eff.shape[2] == num_rx_ant
        assert h_eff.shape[3] == num_tx
        assert h_eff.shape[4] == num_streams_per_tx
        assert h_eff.shape[5] == 14

    def test_zf_precoding_alpha_zero(self, device, precision):
        """Test that alpha=0 gives ZF precoding (same as RZF with alpha=0)."""
        setup = create_test_setup(device=device, precision=precision)
        sm, rg = setup["sm"], setup["rg"]

        precoder = RZFPrecoder(rg, sm, precision=precision, device=device)

        batch_size = 8
        cdtype = torch.complex64 if precision == "single" else torch.complex128

        x = torch.randn(
            batch_size, setup["num_tx"], setup["num_streams_per_tx"], 14, 64,
            dtype=cdtype, device=device
        )
        h = torch.randn(
            batch_size, setup["num_rx"], setup["num_rx_ant"], setup["num_tx"],
            setup["num_tx_ant"], 14, 64, dtype=cdtype, device=device
        )

        # Should not raise any errors
        x_precoded = precoder(x, h, alpha=0.0)
        assert not torch.isnan(x_precoded).any()


class TestRZFPrecodedChannel:
    """Tests for the RZFPrecodedChannel class."""

    def test_against_alternative_implementation(self, device, precision):
        """Test RZFPrecodedChannel against an alternative manual implementation."""
        num_rx_per_tx = 2
        num_streams_per_rx = 2
        num_rx_ant = num_streams_per_rx
        num_tx = 2
        num_rx = num_rx_per_tx * num_tx
        num_streams_per_tx = num_rx_per_tx * num_streams_per_rx
        num_tx_ant = num_streams_per_tx * 2

        rx_tx_association = np.zeros((num_rx, num_tx), dtype=np.int32)
        for j in range(num_tx):
            rx_tx_association[j * num_rx_per_tx : (j + 1) * num_rx_per_tx, j] = 1

        sm = StreamManagement(rx_tx_association, num_streams_per_tx)
        rg = ResourceGrid(
            num_ofdm_symbols=14,
            fft_size=64,
            subcarrier_spacing=15e3,
            num_tx=num_tx,
            num_streams_per_tx=num_streams_per_tx,
            precision=precision,
            device=device,
        )

        channel = RayleighBlockFading(
            num_rx=num_rx,
            num_rx_ant=num_rx_ant,
            num_tx=num_tx,
            num_tx_ant=num_tx_ant,
            precision=precision,
            device=device,
        )

        batch_size = 32
        rdtype = torch.float32 if precision == "single" else torch.float64

        torch.manual_seed(42)
        tx_power = torch.rand(
            batch_size, num_tx, num_streams_per_tx, rg.num_ofdm_symbols, rg.fft_size,
            dtype=rdtype, device=device
        )
        alpha = torch.rand(batch_size, num_tx, 1, 1, dtype=rdtype, device=device)

        cir = channel(batch_size=batch_size, num_time_steps=rg.num_ofdm_symbols)
        frequencies = subcarrier_frequencies(
            rg.fft_size, rg.subcarrier_spacing, precision=precision, device=device
        )
        h = cir_to_ofdm_channel(frequencies, *cir)

        precoded_channel = RZFPrecodedChannel(
            resource_grid=rg, stream_management=sm,
            precision=precision, device=device
        )
        h_eff = precoded_channel(h, tx_power=tx_power, alpha=alpha)

        # Verify against alternative implementation
        for j in range(num_tx):
            rx_ind = np.where(rx_tx_association[:, j])[0]
            h_des = h[:, rx_ind, :, j : j + 1, :, :, :]
            h_des = h_des.squeeze(3)
            h_des = h_des.reshape(
                batch_size, -1, h.shape[4], rg.num_ofdm_symbols, rg.fft_size
            )
            h_des = h_des.permute(0, 3, 4, 1, 2)

            g = rzf_precoding_matrix(h_des, alpha=alpha[:, j], precision=precision)

            # Apply power
            power = tx_power[:, j, :]
            power = power.permute(0, 2, 3, 1)
            power = expand_to_rank(power, g.dim(), axis=-2)
            g = power.sqrt().to(dtype=g.dtype) * g

            for i in range(num_rx):
                h_i_j = h[:, i, :, j, :, :, :]
                h_i_j = h_i_j.permute(0, 3, 4, 1, 2)
                h_eff_i_j = h_i_j @ g

                q = h_eff[:, i, :, j, :]
                q = q.permute(0, 3, 4, 1, 2)

                atol = 1e-5 if precision == "single" else 1e-10
                assert torch.allclose(q, h_eff_i_j, atol=atol)

    def test_output_shape(self, device, precision):
        """Test that RZFPrecodedChannel produces correct output shapes."""
        setup = create_test_setup(device=device, precision=precision)
        sm, rg = setup["sm"], setup["rg"]

        precoded_channel = RZFPrecodedChannel(
            rg, sm, precision=precision, device=device
        )

        batch_size = 16
        cdtype = torch.complex64 if precision == "single" else torch.complex128
        rdtype = torch.float32 if precision == "single" else torch.float64

        h = torch.randn(
            batch_size, setup["num_rx"], setup["num_rx_ant"], setup["num_tx"],
            setup["num_tx_ant"], 14, 64, dtype=cdtype, device=device
        )
        tx_power = torch.rand(
            batch_size, setup["num_tx"], setup["num_streams_per_tx"], 14, 64,
            dtype=rdtype, device=device
        )

        h_eff = precoded_channel(h, tx_power, alpha=0.1)

        # Check dimensions (last dim is num_effective_subcarriers)
        assert h_eff.shape[0] == batch_size
        assert h_eff.shape[1] == setup["num_rx"]
        assert h_eff.shape[2] == setup["num_rx_ant"]
        assert h_eff.shape[3] == setup["num_tx"]
        assert h_eff.shape[4] == setup["num_streams_per_tx"]
        assert h_eff.shape[5] == 14

    def test_with_separate_h_hat(self, device, precision):
        """Test RZFPrecodedChannel with separate h_hat for precoding computation."""
        setup = create_test_setup(device=device, precision=precision)
        sm, rg = setup["sm"], setup["rg"]

        precoded_channel = RZFPrecodedChannel(
            rg, sm, precision=precision, device=device
        )

        batch_size = 8
        cdtype = torch.complex64 if precision == "single" else torch.complex128
        rdtype = torch.float32 if precision == "single" else torch.float64

        h = torch.randn(
            batch_size, setup["num_rx"], setup["num_rx_ant"], setup["num_tx"],
            setup["num_tx_ant"], 14, 64, dtype=cdtype, device=device
        )
        h_hat = torch.randn_like(h)  # Different channel estimate
        tx_power = torch.rand(
            batch_size, setup["num_tx"], setup["num_streams_per_tx"], 14, 64,
            dtype=rdtype, device=device
        )

        h_eff = precoded_channel(h, tx_power, h_hat=h_hat, alpha=0.1)

        # Should produce valid output
        assert not torch.isnan(h_eff).any()
        assert h_eff.shape[0] == batch_size


class TestCBFPrecodedChannel:
    """Tests for the CBFPrecodedChannel class."""

    def test_output_shape(self, device, precision):
        """Test that CBFPrecodedChannel produces correct output shapes."""
        setup = create_test_setup(device=device, precision=precision)
        sm, rg = setup["sm"], setup["rg"]

        precoded_channel = CBFPrecodedChannel(
            rg, sm, precision=precision, device=device
        )

        batch_size = 16
        cdtype = torch.complex64 if precision == "single" else torch.complex128
        rdtype = torch.float32 if precision == "single" else torch.float64

        h = torch.randn(
            batch_size, setup["num_rx"], setup["num_rx_ant"], setup["num_tx"],
            setup["num_tx_ant"], 14, 64, dtype=cdtype, device=device
        )
        tx_power = torch.rand(
            batch_size, setup["num_tx"], setup["num_streams_per_tx"], 14, 64,
            dtype=rdtype, device=device
        )

        h_eff = precoded_channel(h, tx_power)

        assert h_eff.shape[0] == batch_size
        assert h_eff.shape[1] == setup["num_rx"]
        assert h_eff.shape[2] == setup["num_rx_ant"]
        assert h_eff.shape[3] == setup["num_tx"]
        assert h_eff.shape[4] == setup["num_streams_per_tx"]
        assert h_eff.shape[5] == 14

    def test_with_h_hat(self, device, precision):
        """Test CBFPrecodedChannel with separate h_hat."""
        setup = create_test_setup(device=device, precision=precision)
        sm, rg = setup["sm"], setup["rg"]

        precoded_channel = CBFPrecodedChannel(
            rg, sm, precision=precision, device=device
        )

        batch_size = 8
        cdtype = torch.complex64 if precision == "single" else torch.complex128
        rdtype = torch.float32 if precision == "single" else torch.float64

        h = torch.randn(
            batch_size, setup["num_rx"], setup["num_rx_ant"], setup["num_tx"],
            setup["num_tx_ant"], 14, 64, dtype=cdtype, device=device
        )
        h_hat = torch.randn_like(h)
        tx_power = torch.rand(
            batch_size, setup["num_tx"], setup["num_streams_per_tx"], 14, 64,
            dtype=rdtype, device=device
        )

        h_eff = precoded_channel(h, tx_power, h_hat=h_hat)

        assert not torch.isnan(h_eff).any()


class TestEyePrecodedChannel:
    """Tests for the EyePrecodedChannel class."""

    def test_output_shape(self, device, precision):
        """Test that EyePrecodedChannel produces correct output shapes."""
        # For identity precoder, num_streams_per_tx must equal num_tx_ant
        num_tx = 2
        num_rx = 2
        num_tx_ant = 4
        num_rx_ant = 2
        num_streams_per_tx = num_tx_ant  # Must equal num_tx_ant

        rx_tx_association = np.eye(num_rx, num_tx, dtype=np.int32)
        sm = StreamManagement(rx_tx_association, num_streams_per_tx)
        rg = ResourceGrid(
            num_ofdm_symbols=14,
            fft_size=64,
            subcarrier_spacing=15e3,
            num_tx=num_tx,
            num_streams_per_tx=num_streams_per_tx,
            precision=precision,
            device=device,
        )

        precoded_channel = EyePrecodedChannel(
            rg, sm, precision=precision, device=device
        )

        batch_size = 16
        cdtype = torch.complex64 if precision == "single" else torch.complex128
        rdtype = torch.float32 if precision == "single" else torch.float64

        h = torch.randn(
            batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, 14, 64,
            dtype=cdtype, device=device
        )
        tx_power = torch.rand(
            batch_size, num_tx, num_streams_per_tx, 14, 64,
            dtype=rdtype, device=device
        )

        h_eff = precoded_channel(h, tx_power)

        assert h_eff.shape[0] == batch_size
        assert h_eff.shape[1] == num_rx
        assert h_eff.shape[2] == num_rx_ant
        assert h_eff.shape[3] == num_tx
        assert h_eff.shape[4] == num_streams_per_tx
        assert h_eff.shape[5] == 14

    def test_identity_precoding_preserves_channel_structure(self, device, precision):
        """Test that identity precoding with unit power preserves channel structure."""
        num_tx = 1
        num_rx = 1
        num_tx_ant = 2
        num_rx_ant = 2
        num_streams_per_tx = num_tx_ant

        rx_tx_association = np.array([[1]], dtype=np.int32)
        sm = StreamManagement(rx_tx_association, num_streams_per_tx)
        rg = ResourceGrid(
            num_ofdm_symbols=14,
            fft_size=64,
            subcarrier_spacing=15e3,
            num_tx=num_tx,
            num_streams_per_tx=num_streams_per_tx,
            precision=precision,
            device=device,
        )

        precoded_channel = EyePrecodedChannel(
            rg, sm, precision=precision, device=device
        )

        batch_size = 4
        cdtype = torch.complex64 if precision == "single" else torch.complex128
        rdtype = torch.float32 if precision == "single" else torch.float64

        h = torch.randn(
            batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, 14, 64,
            dtype=cdtype, device=device
        )
        # Unit power
        tx_power = torch.ones(
            batch_size, num_tx, num_streams_per_tx, 14, 64,
            dtype=rdtype, device=device
        )

        h_eff = precoded_channel(h, tx_power)

        # With unit power and identity precoding, effective channel should equal original
        # (after removing nulled subcarriers)
        # Since there are no guard carriers, h_eff should be very close to h
        atol = 1e-5 if precision == "single" else 1e-10
        assert torch.allclose(h_eff, h, atol=atol)

    def test_broadcastable_tx_power(self, device, precision):
        """Test EyePrecodedChannel with broadcastable tx_power."""
        num_tx = 2
        num_rx = 2
        num_tx_ant = 4
        num_rx_ant = 2
        num_streams_per_tx = num_tx_ant

        rx_tx_association = np.eye(num_rx, num_tx, dtype=np.int32)
        sm = StreamManagement(rx_tx_association, num_streams_per_tx)
        rg = ResourceGrid(
            num_ofdm_symbols=14,
            fft_size=64,
            subcarrier_spacing=15e3,
            num_tx=num_tx,
            num_streams_per_tx=num_streams_per_tx,
            precision=precision,
            device=device,
        )

        precoded_channel = EyePrecodedChannel(
            rg, sm, precision=precision, device=device
        )

        batch_size = 8
        cdtype = torch.complex64 if precision == "single" else torch.complex128
        rdtype = torch.float32 if precision == "single" else torch.float64

        h = torch.randn(
            batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, 14, 64,
            dtype=cdtype, device=device
        )
        # Lower-rank tx_power (broadcastable) - shape [1, num_tx, num_streams_per_tx, 1, 1]
        # to properly broadcast to [batch_size, num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size]
        tx_power = torch.rand(
            1, num_tx, num_streams_per_tx, 1, 1,
            dtype=rdtype, device=device
        )

        h_eff = precoded_channel(h, tx_power)
        assert not torch.isnan(h_eff).any()


class TestPrecodingCompile:
    """Tests for torch.compile compatibility of precoding classes."""

    def test_rzf_precoder_compile(self, device, precision, mode):
        """Test that RZFPrecoder works with torch.compile."""
        if device == "cpu" and mode != "default":
            pytest.skip("Only default mode supported on CPU")

        setup = create_test_setup(device=device, precision=precision)
        sm, rg = setup["sm"], setup["rg"]

        precoder = RZFPrecoder(rg, sm, precision=precision, device=device)
        compiled_precoder = torch.compile(precoder, mode=mode)

        batch_size = 8
        cdtype = torch.complex64 if precision == "single" else torch.complex128

        x = torch.randn(
            batch_size, setup["num_tx"], setup["num_streams_per_tx"], 14, 64,
            dtype=cdtype, device=device
        )
        h = torch.randn(
            batch_size, setup["num_rx"], setup["num_rx_ant"], setup["num_tx"],
            setup["num_tx_ant"], 14, 64, dtype=cdtype, device=device
        )

        # Run compiled version
        x_precoded_compiled = compiled_precoder(x, h, alpha=0.1)

        # Run non-compiled version
        precoder2 = RZFPrecoder(rg, sm, precision=precision, device=device)
        x_precoded = precoder2(x, h, alpha=0.1)

        atol = 1e-4 if precision == "single" else 1e-8
        assert torch.allclose(x_precoded, x_precoded_compiled, atol=atol)

    def test_rzf_precoded_channel_compile(self, device, precision, mode):
        """Test that RZFPrecodedChannel works with torch.compile."""
        if device == "cpu" and mode != "default":
            pytest.skip("Only default mode supported on CPU")

        setup = create_test_setup(device=device, precision=precision)
        sm, rg = setup["sm"], setup["rg"]

        precoded_channel = RZFPrecodedChannel(
            rg, sm, precision=precision, device=device
        )
        compiled_channel = torch.compile(precoded_channel, mode=mode)

        batch_size = 8
        cdtype = torch.complex64 if precision == "single" else torch.complex128
        rdtype = torch.float32 if precision == "single" else torch.float64

        h = torch.randn(
            batch_size, setup["num_rx"], setup["num_rx_ant"], setup["num_tx"],
            setup["num_tx_ant"], 14, 64, dtype=cdtype, device=device
        )
        tx_power = torch.rand(
            batch_size, setup["num_tx"], setup["num_streams_per_tx"], 14, 64,
            dtype=rdtype, device=device
        )

        try:
            h_eff_compiled = compiled_channel(h, tx_power, alpha=0.1)
        except RuntimeError as error:
            torch_version = torch.__version__.partition("+")[0]
            error_type = type(error)
            if (
                device.startswith("cuda")
                and torch_version == "2.11.0"
                and error_type.__module__ == "torch._inductor.exc"
                and error_type.__name__ == "InductorError"
                and "KeyError: 'complex" in str(error)
            ):
                pytest.xfail(
                    "PyTorch #185396: Inductor cannot generate a Triton "
                    "signature for complex tensors"
                )
            raise

        precoded_channel2 = RZFPrecodedChannel(
            rg, sm, precision=precision, device=device
        )
        h_eff = precoded_channel2(h, tx_power, alpha=0.1)

        atol = 1e-4 if precision == "single" else 1e-8
        assert torch.allclose(h_eff, h_eff_compiled, atol=atol)

    def test_cbf_precoded_channel_compile(self, device, precision, mode):
        """Test that CBFPrecodedChannel works with torch.compile."""
        if device == "cpu" and mode != "default":
            pytest.skip("Only default mode supported on CPU")

        setup = create_test_setup(device=device, precision=precision)
        sm, rg = setup["sm"], setup["rg"]

        precoded_channel = CBFPrecodedChannel(
            rg, sm, precision=precision, device=device
        )
        compiled_channel = torch.compile(precoded_channel, mode=mode)

        batch_size = 8
        cdtype = torch.complex64 if precision == "single" else torch.complex128
        rdtype = torch.float32 if precision == "single" else torch.float64

        h = torch.randn(
            batch_size, setup["num_rx"], setup["num_rx_ant"], setup["num_tx"],
            setup["num_tx_ant"], 14, 64, dtype=cdtype, device=device
        )
        tx_power = torch.rand(
            batch_size, setup["num_tx"], setup["num_streams_per_tx"], 14, 64,
            dtype=rdtype, device=device
        )

        h_eff_compiled = compiled_channel(h, tx_power)

        precoded_channel2 = CBFPrecodedChannel(
            rg, sm, precision=precision, device=device
        )
        h_eff = precoded_channel2(h, tx_power)

        atol = 1e-4 if precision == "single" else 1e-8
        assert torch.allclose(h_eff, h_eff_compiled, atol=atol)

    def test_eye_precoded_channel_compile(self, device, precision, mode):
        """Test that EyePrecodedChannel works with torch.compile."""
        if device == "cpu" and mode != "default":
            pytest.skip("Only default mode supported on CPU")

        num_tx = 2
        num_rx = 2
        num_tx_ant = 4
        num_rx_ant = 2
        num_streams_per_tx = num_tx_ant

        rx_tx_association = np.eye(num_rx, num_tx, dtype=np.int32)
        sm = StreamManagement(rx_tx_association, num_streams_per_tx)
        rg = ResourceGrid(
            num_ofdm_symbols=14,
            fft_size=64,
            subcarrier_spacing=15e3,
            num_tx=num_tx,
            num_streams_per_tx=num_streams_per_tx,
            precision=precision,
            device=device,
        )

        precoded_channel = EyePrecodedChannel(
            rg, sm, precision=precision, device=device
        )
        compiled_channel = torch.compile(precoded_channel, mode=mode)

        batch_size = 8
        cdtype = torch.complex64 if precision == "single" else torch.complex128
        rdtype = torch.float32 if precision == "single" else torch.float64

        h = torch.randn(
            batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, 14, 64,
            dtype=cdtype, device=device
        )
        tx_power = torch.rand(
            batch_size, num_tx, num_streams_per_tx, 14, 64,
            dtype=rdtype, device=device
        )

        h_eff_compiled = compiled_channel(h, tx_power)

        precoded_channel2 = EyePrecodedChannel(
            rg, sm, precision=precision, device=device
        )
        h_eff = precoded_channel2(h, tx_power)

        atol = 1e-4 if precision == "single" else 1e-8
        assert torch.allclose(h_eff, h_eff_compiled, atol=atol)
