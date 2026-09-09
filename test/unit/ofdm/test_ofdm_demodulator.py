#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Unit tests for OFDM Demodulator"""

import numpy as np
import pytest
import torch

from sionna.phy.mapping import QAMSource
from sionna.phy.ofdm import OFDMDemodulator, OFDMModulator


class TestOFDMDemodulator:
    """Tests for the OFDMDemodulator class"""

    def test_cyclic_prefixes(self, device, precision):
        """Test demodulation with various cyclic prefix lengths"""
        batch_size = 64
        fft_size = 72
        num_ofdm_symbols = 14
        qam_source = QAMSource(4, precision=precision, device=device)

        for cp_length in [0, 1, 12, 36, fft_size]:
            modulator = OFDMModulator(
                cp_length, precision=precision, device=device
            )
            demodulator = OFDMDemodulator(
                fft_size, 0, cp_length, precision=precision, device=device
            )
            x = qam_source([batch_size, num_ofdm_symbols, fft_size])
            x_time = modulator(x)
            x_hat = demodulator(x_time)

            assert torch.max(torch.abs(x - x_hat)) < 1e-5

    def test_higher_dimensions(self, device, precision):
        """Test demodulator with higher dimensional inputs"""
        batch_size = [64, 12, 6]
        fft_size = 72
        num_ofdm_symbols = 14
        qam_source = QAMSource(4, precision=precision, device=device)

        for cp_length in [1, 12, 36]:
            modulator = OFDMModulator(
                cp_length, precision=precision, device=device
            )
            demodulator = OFDMDemodulator(
                fft_size, 0, cp_length, precision=precision, device=device
            )
            x = qam_source(batch_size + [num_ofdm_symbols, fft_size])
            x_time = modulator(x)
            x_hat = demodulator(x_time)

            assert torch.max(torch.abs(x - x_hat)) < 1e-5

    def test_overlapping_input(self, device, precision):
        """Test demodulator handles extra trailing samples"""
        batch_size = 64
        fft_size = 72
        num_ofdm_symbols = 14
        qam_source = QAMSource(4, precision=precision, device=device)

        for cp_length in [0, 12]:
            modulator = OFDMModulator(
                cp_length, precision=precision, device=device
            )
            demodulator = OFDMDemodulator(
                fft_size, 0, cp_length, precision=precision, device=device
            )
            x = qam_source([batch_size, num_ofdm_symbols, fft_size])
            x_time = modulator(x)

            # Add extra trailing samples
            x_time = torch.cat([x_time, x_time[..., :10]], dim=-1)
            x_hat = demodulator(x_time)

            assert torch.max(torch.abs(x - x_hat)) < 1e-5

    def test_variable_cyclic_prefixes(self, device, precision):
        """Test demodulation with variable cyclic prefix lengths per symbol"""
        batch_size = 32
        fft_size = 72
        num_ofdm_symbols = fft_size
        qam_source = QAMSource(4, precision=precision, device=device)

        cp_lengths = np.arange(fft_size)
        modulator = OFDMModulator(
            cp_lengths, precision=precision, device=device
        )
        demodulator = OFDMDemodulator(
            fft_size, 0, cp_lengths, precision=precision, device=device
        )

        x = qam_source([batch_size, num_ofdm_symbols, fft_size])
        x_time = modulator(x)
        x_hat = demodulator(x_time)

        assert torch.max(torch.abs(x - x_hat)) < 1e-5

    def test_l_min_phase_compensation(self, device, precision):
        """Test that l_min phase compensation works correctly"""
        batch_size = 32
        fft_size = 64
        num_ofdm_symbols = 14
        cp_length = 16
        qam_source = QAMSource(4, precision=precision, device=device)

        modulator = OFDMModulator(
            cp_length, precision=precision, device=device
        )

        # Test with l_min = 0
        demodulator_0 = OFDMDemodulator(
            fft_size, 0, cp_length, precision=precision, device=device
        )
        x = qam_source([batch_size, num_ofdm_symbols, fft_size])
        x_time = modulator(x)
        x_hat_0 = demodulator_0(x_time)

        assert torch.max(torch.abs(x - x_hat_0)) < 1e-5

    def test_output_shape(self, device, precision):
        """Test that output has correct shape"""
        batch_size = 32
        fft_size = 64
        num_ofdm_symbols = 14
        cp_length = 16
        qam_source = QAMSource(4, precision=precision, device=device)

        modulator = OFDMModulator(
            cp_length, precision=precision, device=device
        )
        demodulator = OFDMDemodulator(
            fft_size, 0, cp_length, precision=precision, device=device
        )

        x = qam_source([batch_size, num_ofdm_symbols, fft_size])
        x_time = modulator(x)
        x_hat = demodulator(x_time)

        assert x_hat.shape == x.shape


class TestOFDMDemodulatorCompile:
    """Tests for torch.compile compatibility"""

    def test_trailing_samples_compile_without_graph_breaks(
        self, device, precision
    ):
        """Test trailing-sample removal compiles as a single graph."""
        fft_size = 64
        cp_length = 16
        modulator = OFDMModulator(
            cp_length, precision=precision, device=device
        )
        demodulator = OFDMDemodulator(
            fft_size, 0, cp_length, precision=precision, device=device
        )
        compiled_demodulator = torch.compile(demodulator, fullgraph=True)

        qam_source = QAMSource(4, precision=precision, device=device)
        x = qam_source([4, 14, fft_size])
        x_time = modulator(x)
        x_time = torch.cat([x_time, torch.zeros_like(x_time[..., :1])], dim=-1)

        x_hat = compiled_demodulator(x_time)
        torch.testing.assert_close(x_hat, x)

    def test_demodulator_compile(self, device, precision, mode):
        """Test that OFDMDemodulator works with torch.compile"""
        if device == "cpu" and mode != "default":
            pytest.skip("Only default mode supported on CPU")

        batch_size = 32
        fft_size = 64
        num_ofdm_symbols = 14
        cp_length = 16

        modulator = OFDMModulator(
            cp_length, precision=precision, device=device
        )
        demodulator = OFDMDemodulator(
            fft_size, 0, cp_length, precision=precision, device=device
        )
        compiled_demodulator = torch.compile(demodulator, mode=mode)

        qam_source = QAMSource(4, precision=precision, device=device)
        x = qam_source([batch_size, num_ofdm_symbols, fft_size])
        x_time = modulator(x)

        # Run compiled version
        x_hat_compiled = compiled_demodulator(x_time)

        # Run non-compiled version for comparison
        demodulator2 = OFDMDemodulator(
            fft_size, 0, cp_length, precision=precision, device=device
        )
        x_hat = demodulator2(x_time)

        assert torch.allclose(x_hat, x_hat_compiled, atol=1e-5)


class TestOFDMModDemod:
    """End-to-end tests for modulator and demodulator"""

    def test_end_to_end(self, device, precision):
        """E2E test verifying that all shapes can be properly inferred"""
        fft_size = 72
        num_ofdm_symbols = 14
        qam_source = QAMSource(4, precision=precision, device=device)

        for cp_length in [0, 1, 5, 12]:
            for padding in [0, 1, 5, 71]:
                modulator = OFDMModulator(
                    cp_length, precision=precision, device=device
                )
                demodulator = OFDMDemodulator(
                    fft_size, 0, cp_length, precision=precision, device=device
                )

                x_rg = qam_source([128, 1, 1, num_ofdm_symbols, fft_size])
                x_time = modulator(x_rg)

                # Add padding
                if padding > 0:
                    pad = torch.zeros_like(x_time)[..., :padding]
                    x_time = torch.cat([x_time, pad], dim=-1)

                x_f = demodulator(x_time)
                assert x_f.shape == torch.Size(
                    [128, 1, 1, num_ofdm_symbols, fft_size]
                )

    def test_end_to_end_variable_cp(self, device, precision):
        """E2E test with variable cyclic prefix lengths"""
        fft_size = 72
        num_ofdm_symbols = 72
        qam_source = QAMSource(4, precision=precision, device=device)

        cp_lengths = np.arange(fft_size)
        for padding in [0, 1, 5]:
            modulator = OFDMModulator(
                cp_lengths, precision=precision, device=device
            )
            demodulator = OFDMDemodulator(
                fft_size, 0, cp_lengths, precision=precision, device=device
            )

            x_rg = qam_source([32, 1, 1, num_ofdm_symbols, fft_size])
            x_time = modulator(x_rg)

            if padding > 0:
                pad = torch.zeros_like(x_time)[..., :padding]
                x_time = torch.cat([x_time, pad], dim=-1)

            x_f = demodulator(x_time)
            assert x_f.shape == torch.Size(
                [32, 1, 1, num_ofdm_symbols, fft_size]
            )
