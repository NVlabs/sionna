#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Unit tests for OFDM Modulator"""

import numpy as np
import pytest
import torch

from sionna.phy.mapping import QAMSource
from sionna.phy.ofdm import OFDMModulator


class TestOFDMModulator:
    """Tests for the OFDMModulator class"""

    def test_cyclic_prefixes(self, device, precision):
        """Test that cyclic prefix is correctly implemented"""
        batch_size = 64
        fft_size = 72
        num_ofdm_symbols = 14
        qam_source = QAMSource(4, precision=precision, device=device)

        for cp_length in [1, 12, 36, fft_size]:
            modulator = OFDMModulator(
                cp_length, precision=precision, device=device
            )
            x = qam_source([batch_size, num_ofdm_symbols, fft_size])
            x_time = modulator(x)
            x_time = x_time.reshape(batch_size, num_ofdm_symbols, -1)

            # Check that CP equals the end of the symbol
            assert torch.allclose(
                x_time[..., :cp_length],
                x_time[..., -cp_length:],
                atol=1e-5
            )

    def test_cyclic_prefix_too_large(self, device, precision):
        """Test that CP larger than FFT size raises error"""
        fft_size = 72
        cp_length = fft_size + 1
        modulator = OFDMModulator(
            cp_length, precision=precision, device=device
        )
        qam_source = QAMSource(4, precision=precision, device=device)
        x = qam_source([64, 14, fft_size])

        with pytest.raises(ValueError):
            modulator(x)

    def test_variable_cyclic_prefixes(self, device, precision):
        """Test per-OFDM symbol cyclic prefix length"""
        batch_size = 64
        fft_size = 72
        num_ofdm_symbols = fft_size
        qam_source = QAMSource(4, precision=precision, device=device)

        cp_lengths = np.arange(fft_size)
        modulator = OFDMModulator(
            cp_lengths, precision=precision, device=device
        )
        x = qam_source([batch_size, num_ofdm_symbols, fft_size])
        x_time = modulator(x)

        # Verify CP for each symbol
        start = 0
        for i in range(num_ofdm_symbols):
            end = start + cp_lengths[i] + fft_size
            x_sym = x_time[..., start:end]
            if cp_lengths[i] > 0:
                assert torch.allclose(
                    x_sym[..., :cp_lengths[i]],
                    x_sym[..., -cp_lengths[i]:],
                    atol=1e-5
                )
            start = end

    def test_higher_dimensions(self, device, precision):
        """Test modulator with higher dimensional inputs"""
        batch_size = [64, 12, 6]
        fft_size = 72
        num_ofdm_symbols = 14
        qam_source = QAMSource(4, precision=precision, device=device)

        for cp_length in [1, 12, 36]:
            modulator = OFDMModulator(
                cp_length, precision=precision, device=device
            )
            x = qam_source(batch_size + [num_ofdm_symbols, fft_size])
            x_time = modulator(x)
            x_time = x_time.reshape(*batch_size, num_ofdm_symbols, -1)

            assert torch.allclose(
                x_time[..., :cp_length],
                x_time[..., -cp_length:],
                atol=1e-5
            )

    def test_variable_cyclic_prefixes_higher_dimensions(self, device, precision):
        """Test per-OFDM symbol CP length with multi-dimensional batch size"""
        batch_size = [64, 12, 6]
        fft_size = 72
        num_ofdm_symbols = fft_size
        qam_source = QAMSource(4, precision=precision, device=device)

        cp_lengths = np.arange(fft_size)
        modulator = OFDMModulator(
            cp_lengths, precision=precision, device=device
        )
        x = qam_source(batch_size + [num_ofdm_symbols, fft_size])
        x_time = modulator(x)

        start = 0
        for i in range(num_ofdm_symbols):
            end = start + cp_lengths[i] + fft_size
            x_sym = x_time[..., start:end]
            if cp_lengths[i] > 0:
                assert torch.allclose(
                    x_sym[..., :cp_lengths[i]],
                    x_sym[..., -cp_lengths[i]:],
                    atol=1e-5
                )
            start = end

    def test_output_length(self, device, precision):
        """Test that output has correct length"""
        batch_size = 32
        fft_size = 64
        num_ofdm_symbols = 14
        cp_length = 16
        qam_source = QAMSource(4, precision=precision, device=device)

        modulator = OFDMModulator(
            cp_length, precision=precision, device=device
        )
        x = qam_source([batch_size, num_ofdm_symbols, fft_size])
        x_time = modulator(x)

        expected_length = num_ofdm_symbols * (fft_size + cp_length)
        assert x_time.shape[-1] == expected_length

    def test_zero_cyclic_prefix(self, device, precision):
        """Test modulator with zero cyclic prefix"""
        batch_size = 32
        fft_size = 64
        num_ofdm_symbols = 14
        qam_source = QAMSource(4, precision=precision, device=device)

        modulator = OFDMModulator(0, precision=precision, device=device)
        x = qam_source([batch_size, num_ofdm_symbols, fft_size])
        x_time = modulator(x)

        expected_length = num_ofdm_symbols * fft_size
        assert x_time.shape[-1] == expected_length


class TestOFDMModulatorCompile:
    """Tests for torch.compile compatibility"""

    def test_modulator_compile(self, device, precision, mode):
        """Test that OFDMModulator works with torch.compile"""
        if device == "cpu" and mode != "default":
            pytest.skip("Only default mode supported on CPU")

        batch_size = 32
        fft_size = 64
        num_ofdm_symbols = 14
        cp_length = 16

        modulator = OFDMModulator(
            cp_length, precision=precision, device=device
        )
        compiled_modulator = torch.compile(modulator, mode=mode)

        qam_source = QAMSource(4, precision=precision, device=device)
        x = qam_source([batch_size, num_ofdm_symbols, fft_size])

        # Run compiled version
        x_time_compiled = compiled_modulator(x)

        # Run non-compiled version for comparison
        modulator2 = OFDMModulator(
            cp_length, precision=precision, device=device
        )
        x_time = modulator2(x)

        assert torch.allclose(x_time, x_time_compiled, atol=1e-5)

    def test_first_call_compiles_as_fullgraph(self, device, precision):
        """Lazy-build CP validation must not split the first compiled call."""
        fft_size = 64
        num_ofdm_symbols = 14
        cp_length = 16
        x = QAMSource(4, precision=precision, device=device)(
            [4, num_ofdm_symbols, fft_size]
        )

        modulator = OFDMModulator(
            cp_length, precision=precision, device=device
        )
        compiled_modulator = torch.compile(modulator, fullgraph=True)
        actual = compiled_modulator(x)

        expected = OFDMModulator(
            cp_length, precision=precision, device=device
        )(x)
        torch.testing.assert_close(actual, expected)

