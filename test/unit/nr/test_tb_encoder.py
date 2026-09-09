#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Tests for TBEncoder and TBDecoder."""

import os

import pytest
import numpy as np
import torch

from sionna.phy.nr import TBEncoder, TBDecoder
from sionna.phy.mapping import BinarySource


# Module-relative path: CI runs pytest from test/, so a repo-root-relative
# path like test/unit/nr/tb_refs/ never resolves there.
REF_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tb_refs")


class TestTBEncoder:
    """Tests for TBEncoder."""

    def test_reference(self):
        """Test against reference implementation."""
        if not os.path.isdir(REF_PATH):
            pytest.skip("Reference files not found")

        files = sorted(
            fi for fi in os.listdir(REF_PATH) if fi.endswith(".npz")
        )
        if not files:
            pytest.skip("No reference files found")

        # Test all reference files
        for fn in files:
            data = np.load(os.path.join(REF_PATH, fn))
            u_ref = data["u_ref"]
            c_ref = data["c_ref"]
            n_id = data["n_id"]
            n_rnti = data["n_rnti"]
            target_coderate = data["coderate"]
            num_bits_per_symbol = data["num_bits_per_symbol"]
            num_layers = data["num_layers"]
            num_coded_bits = c_ref.shape[1]
            tb_size = u_ref.shape[1]

            encoder = TBEncoder(
                num_coded_bits=int(num_coded_bits),
                target_tb_size=int(tb_size),
                target_coderate=float(target_coderate),
                num_bits_per_symbol=int(num_bits_per_symbol),
                num_layers=int(num_layers),
                n_rnti=int(n_rnti),
                n_id=int(n_id),
                channel_type="PUSCH",
                codeword_index=0,
                use_scrambler=True,
            )

            decoder = TBDecoder(encoder, cn_update="minsum")

            u_tensor = torch.tensor(u_ref, dtype=encoder.dtype)
            c = encoder(u_tensor)
            u_hat, _ = decoder(2 * c - 1)

            np.testing.assert_array_equal(c.cpu().numpy(), c_ref)
            np.testing.assert_array_equal(u_hat.cpu().numpy(), u_ref)

    def test_multi_stream(self):
        """Test that n_rnti and n_id can be provided as list."""
        n_rnti = [224, 42, 1, 1337, 45666, 2333, 2133]
        n_id = [42, 123, 0, 3, 32, 456, 875]

        bs = 10
        tb_size = 50000
        num_coded_bits = 100000
        target_coderate = tb_size / num_coded_bits
        num_bits_per_symbol = 4
        num_layers = 2

        encoder = TBEncoder(
            target_tb_size=tb_size,
            num_coded_bits=num_coded_bits,
            target_coderate=target_coderate,
            num_bits_per_symbol=num_bits_per_symbol,
            num_layers=num_layers,
            n_rnti=n_rnti,
            n_id=n_id,
            channel_type="PUSCH",
            codeword_index=0,
            use_scrambler=True,
        )

        decoder = TBDecoder(encoder)
        source = BinarySource()

        u = source([bs, len(n_rnti), encoder.k])
        c = encoder(u)
        u_hat, _ = decoder(2 * c - 1)

        np.testing.assert_array_equal(u.cpu().numpy(), u_hat.cpu().numpy())

        # Verify against individual encoders
        c_ref = np.zeros_like(c.cpu().numpy())
        for idx, (nr, ni) in enumerate(zip(n_rnti, n_id)):
            encoder_single = TBEncoder(
                target_tb_size=tb_size,
                num_coded_bits=num_coded_bits,
                target_coderate=target_coderate,
                num_bits_per_symbol=num_bits_per_symbol,
                num_layers=num_layers,
                n_rnti=nr,
                n_id=ni,
                channel_type="PUSCH",
                codeword_index=0,
                use_scrambler=True,
            )
            c_ref[:, idx, :] = encoder_single(u[:, idx, :]).cpu().numpy()

        np.testing.assert_array_equal(c.cpu().numpy(), c_ref)

    def test_basic_encoding(self):
        """Test basic encoding functionality."""
        encoder = TBEncoder(
            target_tb_size=1000,
            num_coded_bits=2000,
            target_coderate=0.5,
            num_bits_per_symbol=4,
            num_layers=1,
            n_rnti=1,
            n_id=1,
        )

        # Check properties
        assert encoder.k <= 1000
        assert encoder.n == 2000
        assert encoder.coderate <= 0.6  # Allow some slack due to quantization

        # Encode random bits
        source = BinarySource()
        bits = source([10, encoder.k])
        coded = encoder(bits)

        assert coded.shape == (10, encoder.n)
        assert coded.dtype == encoder.dtype

    def test_output_shape_with_layers(self):
        """Test output shape with multiple layers."""
        encoder = TBEncoder(
            target_tb_size=500,
            num_coded_bits=1000,
            target_coderate=0.5,
            num_bits_per_symbol=4,
            num_layers=2,
            n_rnti=1,
            n_id=1,
        )

        source = BinarySource()
        bits = source([5, encoder.k])
        coded = encoder(bits)

        assert coded.shape[-1] == encoder.n

    @pytest.mark.parametrize("num_bits_per_symbol", [2, 4, 6])
    def test_different_modulation_orders(self, num_bits_per_symbol):
        """Test with different modulation orders."""
        # Use num_coded_bits divisible by all modulation orders (LCM of 2, 4, 6 = 12)
        encoder = TBEncoder(
            target_tb_size=800,
            num_coded_bits=1596,  # Divisible by 2, 4, and 6 (actually 1596/6=266)
            target_coderate=0.5,
            num_bits_per_symbol=num_bits_per_symbol,
            num_layers=1,
            n_rnti=1,
            n_id=1,
        )

        source = BinarySource()
        bits = source([4, encoder.k])
        coded = encoder(bits)

        assert coded.shape[-1] == encoder.n


class TestTBDecoder:
    """Tests for TBDecoder."""

    def test_basic_decoding(self):
        """Test basic decoding with perfect LLRs."""
        encoder = TBEncoder(
            target_tb_size=500,
            num_coded_bits=1000,
            target_coderate=0.5,
            num_bits_per_symbol=4,
            num_layers=1,
            n_rnti=1,
            n_id=1,
            use_scrambler=False,  # Disable for simpler testing
        )
        decoder = TBDecoder(encoder, num_bp_iter=10)

        # Check properties
        assert decoder.k == encoder.tb_size
        assert decoder.n == encoder.n

        # Generate bits and encode
        source = BinarySource()
        bits = source([5, encoder.k])
        coded = encoder(bits)

        # Create perfect LLRs (high confidence)
        llr = 10.0 * (2.0 * coded - 1.0)

        # Decode
        bits_hat, crc_status = decoder(llr)

        assert bits_hat.shape == bits.shape
        assert crc_status.shape == (5,)

    def test_encode_decode_cycle(self):
        """Test that encode-decode recovers original bits with no noise."""
        encoder = TBEncoder(
            target_tb_size=300,
            num_coded_bits=600,
            target_coderate=0.5,
            num_bits_per_symbol=4,
            num_layers=1,
            n_rnti=1,
            n_id=1,
            use_scrambler=False,
        )
        decoder = TBDecoder(encoder, num_bp_iter=20)

        source = BinarySource()
        bits = source([2, encoder.k])
        coded = encoder(bits)

        # High SNR LLRs
        llr = 20.0 * (2.0 * coded - 1.0)

        bits_hat, crc_status = decoder(llr)

        # All CRCs should pass
        assert crc_status.all()

        # Bits should match
        np.testing.assert_array_equal(bits.cpu().numpy(), bits_hat.cpu().numpy())


class TestTBEncoderDecoder:
    """Integration tests for TBEncoder and TBDecoder."""

    @pytest.mark.parametrize("num_layers", [1, 2])
    def test_with_multiple_layers(self, num_layers):
        """Test encode-decode with multiple layers."""
        encoder = TBEncoder(
            target_tb_size=400,
            num_coded_bits=800,
            target_coderate=0.5,
            num_bits_per_symbol=4,
            num_layers=num_layers,
            n_rnti=1,
            n_id=1,
            use_scrambler=False,
        )
        decoder = TBDecoder(encoder, num_bp_iter=15)

        source = BinarySource()
        bits = source([3, encoder.k])
        coded = encoder(bits)
        llr = 15.0 * (2.0 * coded - 1.0)
        bits_hat, crc_status = decoder(llr)

        # Should decode correctly at high SNR
        assert crc_status.all()

    def test_with_scrambling(self):
        """Test encode-decode with scrambling enabled."""
        encoder = TBEncoder(
            target_tb_size=400,
            num_coded_bits=800,
            target_coderate=0.5,
            num_bits_per_symbol=4,
            num_layers=1,
            n_rnti=12345,
            n_id=100,
            use_scrambler=True,
        )
        decoder = TBDecoder(encoder, num_bp_iter=15)

        source = BinarySource()
        bits = source([2, encoder.k])
        coded = encoder(bits)
        llr = 15.0 * (2.0 * coded - 1.0)
        bits_hat, crc_status = decoder(llr)

        assert crc_status.all()
        np.testing.assert_array_equal(bits.cpu().numpy(), bits_hat.cpu().numpy())

    def test_identity_multiple_scenarios(self):
        """Test that receiver can recover info bits in various scenarios."""
        source = BinarySource()

        # Test parameters covering:
        # 1.) Single CB segmentation
        # 2.) Long CB / multiple CWs
        # 3.) Deactivated scrambler
        # 4.) N-dimensional inputs

        bs_list = [[10], [10], [10]]
        tb_sizes = [6656, 984, 984]
        num_coded_bits_list = [13440, 2880, 2880]
        num_bits_per_symbols = [4, 2, 2]
        num_layers_list = [1, 2, 4]
        n_rntis = [1337, 1337, 1337]
        sc_ids = [1, 2, 42]
        use_scramblers = [True, False, True]

        for i in range(len(tb_sizes)):
            encoder = TBEncoder(
                target_tb_size=tb_sizes[i],
                num_coded_bits=num_coded_bits_list[i],
                target_coderate=tb_sizes[i] / num_coded_bits_list[i],
                num_bits_per_symbol=num_bits_per_symbols[i],
                num_layers=num_layers_list[i],
                n_rnti=n_rntis[i],
                n_id=sc_ids[i],
                channel_type="PUSCH",
                use_scrambler=use_scramblers[i],
            )

            decoder = TBDecoder(encoder, num_bp_iter=10, cn_update="minsum")

            u = source(bs_list[i] + [encoder.k])
            c = encoder(u)
            llr_ch = 2 * c - 1  # Apply BPSK
            u_hat, crc_status = decoder(llr_ch)

            # All info bits can be recovered
            np.testing.assert_array_equal(u.cpu().numpy(), u_hat.cpu().numpy())
            # All CRC checks are valid
            assert crc_status.all()

    def test_crc_detection(self):
        """Test that CRC detects the correct erroneous positions."""
        source = BinarySource()
        bs = 10

        encoder = TBEncoder(
            target_tb_size=10000,
            num_coded_bits=20000,
            target_coderate=0.5,
            num_bits_per_symbol=4,
            n_rnti=12367,
            n_id=312,
            use_scrambler=True,
        )

        decoder = TBDecoder(encoder, num_bp_iter=20, cn_update="minsum")

        u = source([bs, encoder.k])
        c = encoder(u)
        llr_ch = 2 * c - 1  # Apply BPSK

        # Destroy TB at batch index 7
        err_pos = 7
        llr_ch_np = llr_ch.cpu().numpy()
        llr_ch_np[err_pos, 500:590] = -10  # Overwrite some LLR positions

        llr_ch_corrupt = torch.tensor(llr_ch_np, dtype=llr_ch.dtype, device=llr_ch.device)
        u_hat, crc_status = decoder(llr_ch_corrupt)

        # All CRC checks are correct except at err_pos
        crc_status_np = crc_status.cpu().numpy()
        crc_status_ref = np.ones_like(crc_status_np)
        crc_status_ref[err_pos] = 0

        np.testing.assert_array_equal(crc_status_np, crc_status_ref)

    def test_torch_compile(self):
        """Test torch.compile works with encoder/decoder."""
        source = BinarySource()
        bs = 10

        encoder = TBEncoder(
            target_tb_size=10000,
            num_coded_bits=20000,
            target_coderate=0.5,
            num_bits_per_symbol=4,
            n_rnti=12367,
            n_id=312,
            use_scrambler=True,
        )

        decoder = TBDecoder(encoder, num_bp_iter=10, cn_update="minsum")

        # Compile decoder
        compiled_decoder = torch.compile(decoder)

        u = source([bs, encoder.k])
        c = encoder(u)
        llr_ch = 2 * c - 1

        x = compiled_decoder(llr_ch)
        assert x[0].shape[0] == bs

        # Change batch_size
        u2 = source([2 * bs, encoder.k])
        c2 = encoder(u2)
        llr_ch2 = 2 * c2 - 1
        x2 = compiled_decoder(llr_ch2)
        assert x2[0].shape[0] == 2 * bs

