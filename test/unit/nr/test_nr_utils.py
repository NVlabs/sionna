#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Tests for sionna.phy.nr.utils functions."""

import pytest
import numpy as np
import torch

from sionna.phy.nr.utils import (
    decode_mcs_index,
    generate_prng_seq,
    calculate_tb_size,
    calculate_num_coded_bits,
    calculate_codeword_bits,
    MCSDecoderNR,
)
from .utils import calculate_tb_size_numpy, decode_mcs_index_numpy


class TestGeneratePrngSeq:
    """Tests for the pseudo-random sequence generator."""

    def test_invalid_length_negative(self):
        """Test rejection of negative length."""
        with pytest.raises(ValueError):
            generate_prng_seq(-1, 10)

    def test_invalid_c_init_negative(self):
        """Test rejection of negative c_init."""
        with pytest.raises(ValueError):
            generate_prng_seq(10, -1)

    def test_invalid_c_init_too_large(self):
        """Test rejection of c_init >= 2^31."""
        with pytest.raises(ValueError, match=r"2\^31"):
            generate_prng_seq(100, 2**31)
        with pytest.raises(ValueError, match=r"2\^31"):
            generate_prng_seq(100, 2**32 - 1)

    def test_c_init_max_accepted(self):
        """Boundary: 2^31-1 is accepted and differs from 0."""
        s_max = generate_prng_seq(64, 2**31 - 1)
        s0 = generate_prng_seq(64, 0)
        assert s_max.shape == (64,)
        assert not np.array_equal(s_max, s0)

    def test_reference_sequence(self):
        """Test against reference example from 3GPP."""
        n_rnti = 20001
        n_id = 41
        c_init = n_rnti * 2**15 + n_id
        length = 100

        s_ref = np.array([
            0., 1., 1., 1., 1., 0., 0., 0., 0., 1., 0., 1., 0.,
            1., 1., 1., 0., 0., 0., 1., 1., 1., 0., 0., 0., 1.,
            1., 0., 0., 1., 1., 1., 0., 1., 0., 0., 1., 1., 1.,
            0., 1., 0., 0., 0., 0., 0., 1., 1., 1., 0., 1., 1.,
            0., 1., 1., 0., 0., 0., 1., 0., 0., 1., 0., 0., 1.,
            0., 0., 0., 0., 0., 0., 1., 1., 1., 0., 1., 0., 0.,
            1., 1., 0., 1., 1., 1., 0., 0., 0., 0., 0., 1., 0.,
            1., 1., 1., 1., 1., 1., 1., 0., 0.
        ])

        s = generate_prng_seq(length, c_init)
        np.testing.assert_array_equal(s, s_ref)

    def test_different_c_init_different_sequence(self):
        """Test that different c_init produces different sequence."""
        n_rnti = 20001
        n_id = 41
        c_init = n_rnti * 2**15 + n_id
        length = 100

        s1 = generate_prng_seq(length, c_init)
        s2 = generate_prng_seq(length, c_init + 1)

        assert not np.array_equal(s1, s2)


class TestDecodeMcsIndex:
    """Tests for MCS index decoding."""

    def test_pdsch_table1(self):
        """Test PDSCH MCS table 1 (Table 5.1.3.1-1)."""
        qs = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 6,
              6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6]
        rs = [120, 157, 193, 251, 308, 379, 449, 526, 602, 679,
              340, 378, 434, 490, 553, 616, 658, 438, 466, 517,
              567, 616, 666, 719, 772, 822, 873, 910, 948]

        for idx, q in enumerate(qs):
            m, r = decode_mcs_index(mcs_index=idx, table_index=1, is_pusch=False)
            assert m.item() == q
            assert r.item() == pytest.approx(rs[idx] / 1024)

    def test_pdsch_table2(self):
        """Test PDSCH MCS table 2 (Table 5.1.3.1-2)."""
        qs = [2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 6, 6, 6, 6, 6, 6, 6,
              6, 6, 8, 8, 8, 8, 8, 8, 8, 8]
        rs = [120, 193, 308, 449, 602, 378, 434, 490, 553, 616,
              658, 466, 517, 567, 616, 666, 719, 772, 822, 873,
              682.5, 711, 754, 797, 841, 885, 916.5, 948]

        for idx, q in enumerate(qs):
            m, r = decode_mcs_index(mcs_index=idx, table_index=2, is_pusch=False)
            assert m.item() == q
            assert r.item() == pytest.approx(rs[idx] / 1024)

    def test_pusch_without_precoding_table1(self):
        """Test PUSCH without transform precoding (Table 5.1.3.1-1)."""
        qs = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 6,
              6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6]
        rs = [120, 157, 193, 251, 308, 379, 449, 526, 602, 679,
              340, 378, 434, 490, 553, 616, 658, 438, 466, 517,
              567, 616, 666, 719, 772, 822, 873, 910, 948]

        for idx, q in enumerate(qs):
            m, r = decode_mcs_index(
                mcs_index=idx, table_index=1, is_pusch=True,
                transform_precoding=False)
            assert m.item() == q
            assert r.item() == pytest.approx(rs[idx] / 1024)

    def test_pusch_with_precoding_table1_pi2bpsk_false(self):
        """Test PUSCH with transform precoding Table 6.1.4.1-1 (q=2)."""
        qs = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 6,
              6, 6, 6, 6, 6, 6, 6, 6, 6, 6]
        rs = [120, 157, 193, 251, 308, 379, 449, 526, 602, 679,
              340, 378, 434, 490, 553, 616, 658, 466, 517,
              567, 616, 666, 719, 772, 822, 873, 910, 948]

        for idx, q in enumerate(qs):
            m, r = decode_mcs_index(
                mcs_index=idx, table_index=1, is_pusch=True,
                transform_precoding=True, pi2bpsk=False)
            assert m.item() == q
            assert r.item() == pytest.approx(rs[idx] / 1024)

    def test_pusch_with_precoding_table1_pi2bpsk_true(self):
        """Test PUSCH with transform precoding Table 6.1.4.1-1 (q=1)."""
        qs = [1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 6,
              6, 6, 6, 6, 6, 6, 6, 6, 6, 6]
        rs = [240, 314, 193, 251, 308, 379, 449, 526, 602, 679,
              340, 378, 434, 490, 553, 616, 658, 466, 517,
              567, 616, 666, 719, 772, 822, 873, 910, 948]

        for idx, q in enumerate(qs):
            m, r = decode_mcs_index(
                mcs_index=idx, table_index=1, is_pusch=True,
                transform_precoding=True, pi2bpsk=True)
            assert m.item() == q
            assert r.item() == pytest.approx(rs[idx] / 1024)

    def test_invalid_mcs_index_raises(self):
        """Test that invalid MCS index raises error."""
        with pytest.raises(ValueError):
            decode_mcs_index(mcs_index=29, table_index=1)


class TestCalculateTbSize:
    """Tests for transport block size calculation."""

    @pytest.mark.parametrize("mcs_index", [0, 4, 16, 20, 27])
    @pytest.mark.parametrize("num_layers", [1, 2])
    @pytest.mark.parametrize("num_prbs", [1, 20, 100])
    def test_tb_size_consistency(self, mcs_index, num_layers, num_prbs):
        """Test TB size calculation produces consistent results."""
        q, r = decode_mcs_index(mcs_index, table_index=2)
        # Convert tensors to Python scalars
        q_val = q.item()
        r_val = r.item()
        num_ofdm_symbols = 14
        num_dmrs_per_prb = 12

        result = calculate_tb_size(
            target_coderate=r_val,
            modulation_order=q_val,
            num_layers=num_layers,
            num_prbs=num_prbs,
            num_ofdm_symbols=num_ofdm_symbols,
            num_dmrs_per_prb=num_dmrs_per_prb,
            verbose=False,
        )

        tb_size, cb_size, num_cbs, tb_crc_length, cb_crc_length, cw_length = result
        # Convert tensor results to Python scalars
        tb_size = int(tb_size)
        cb_size = int(cb_size)
        num_cbs = int(num_cbs)
        tb_crc_length = int(tb_crc_length)
        cb_crc_length = int(cb_crc_length)

        # TB size must equal number of CB bits (+CRC overhead)
        assert tb_size == num_cbs * (cb_size - cb_crc_length) - tb_crc_length

        # Individual cw length for each cb
        assert num_cbs == len(cw_length)

        # Single cw TB has no CB CRC
        if num_cbs == 1:
            assert cb_crc_length == 0
        else:
            assert cb_crc_length == 24

        # TB CRC is 16 or 24
        if tb_size > 3824:
            assert tb_crc_length == 24
        else:
            assert tb_crc_length == 16

    def test_tb_size_vs_numpy(self):
        """Validate calculate_tb_size against NumPy reference."""
        q, r = decode_mcs_index(10, table_index=1)
        # Convert tensors to Python scalars
        q_val = q.item()
        r_val = r.item()

        result = calculate_tb_size(
            target_coderate=r_val,
            modulation_order=q_val,
            num_layers=1,
            num_prbs=50,
            num_ofdm_symbols=14,
            num_dmrs_per_prb=12,
            verbose=False,
        )
        tb_size, cb_size, num_cbs, tb_crc_length, cb_crc_length, cw_length = result
        # Convert tensor results to Python scalars
        tb_size = int(tb_size)
        cb_size = int(cb_size)
        num_cbs = int(num_cbs)
        tb_crc_length = int(tb_crc_length)
        cb_crc_length = int(cb_crc_length)
        if isinstance(cw_length, torch.Tensor):
            cw_length = cw_length.cpu().numpy()

        # Compare against numpy version
        result_np = calculate_tb_size_numpy(
            modulation_order=q_val,
            target_coderate=r_val,
            num_layers=1,
            num_prbs=50,
            num_ofdm_symbols=14,
            num_dmrs_per_prb=12,
            verbose=False,
        )

        assert tb_size == result_np[0]
        assert cb_size == result_np[1]
        assert num_cbs == result_np[2]
        assert tb_crc_length == result_np[3]
        assert cb_crc_length == result_np[4]
        np.testing.assert_array_equal(cw_length, result_np[5])


class TestDecodeMcsIndexAgainstNumpy:
    """Test decode_mcs_index against NumPy reference implementation."""

    @pytest.mark.parametrize("mcs_index", range(0, 27))
    @pytest.mark.parametrize("table_index", [1, 2])
    @pytest.mark.parametrize("is_pusch", [True, False])
    def test_vs_numpy(self, mcs_index, table_index, is_pusch):
        """Compare PyTorch version against NumPy reference."""
        # PyTorch version
        m_torch, r_torch = decode_mcs_index(
            mcs_index=mcs_index,
            table_index=table_index,
            is_pusch=is_pusch,
            transform_precoding=False,
        )

        # NumPy reference
        channel_type = "PUSCH" if is_pusch else "PDSCH"
        m_np, r_np = decode_mcs_index_numpy(
            mcs_index=mcs_index,
            table_index=table_index,
            channel_type=channel_type,
            transform_precoding=False,
        )

        assert m_torch.item() == m_np
        assert r_torch.item() == pytest.approx(r_np)


class TestMCSDecoderNRDefaultsAndValidation:
    """MCSDecoderNR default alignment and category/index contracts."""

    def test_default_matches_decode_mcs_index(self):
        """Block default transform_precoding matches the function default."""
        decoder = MCSDecoderNR()
        mo_b, r_b = decoder(mcs_index=17, mcs_table_index=1, mcs_category=0)
        mo_f, r_f = decode_mcs_index(
            17, table_index=1, is_pusch=True, transform_precoding=False
        )
        assert mo_b.item() == mo_f.item()
        assert r_b.item() == pytest.approx(r_f.item())

    def test_explicit_transform_precoding_true(self):
        decoder = MCSDecoderNR()
        mo_b, r_b = decoder(
            mcs_index=17,
            mcs_table_index=1,
            mcs_category=0,
            transform_precoding=True,
        )
        mo_f, r_f = decode_mcs_index(
            17, table_index=1, is_pusch=True, transform_precoding=True
        )
        assert mo_b.item() == mo_f.item()
        assert r_b.item() == pytest.approx(r_f.item())

    def test_invalid_category_raises(self):
        decoder = MCSDecoderNR()
        for cat in (-1, 2):
            with pytest.raises(ValueError, match="mcs_category"):
                decoder(mcs_index=10, mcs_table_index=1, mcs_category=cat)

    def test_non_integer_mcs_index_raises(self):
        with pytest.raises(ValueError, match="integer"):
            decode_mcs_index(1.9, table_index=1)
        with pytest.raises(ValueError, match="integer"):
            decode_mcs_index(np.float64(1.9), table_index=1)
        with pytest.raises(ValueError, match="integer"):
            decode_mcs_index(torch.tensor([1.2, 2.0]), table_index=1)

    def test_integer_valued_float_accepted(self):
        mo, r = decode_mcs_index(1.0, table_index=1)
        mo_i, r_i = decode_mcs_index(1, table_index=1)
        assert mo.item() == mo_i.item()
        assert r.item() == pytest.approx(r_i.item())

    def test_numpy_integer_scalar_accepted(self):
        mo, r = decode_mcs_index(np.int64(5), table_index=1)
        mo_i, r_i = decode_mcs_index(5, table_index=1)
        assert mo.item() == mo_i.item()
        assert r.item() == pytest.approx(r_i.item())

    def test_compiled_fullgraph(self, device):
        """Tensor category and index checks must not break the compiled graph."""
        decoder = MCSDecoderNR(device=device)

        @torch.compile(fullgraph=True)
        def decode(mcs_index, table_index, category):
            return decoder(mcs_index, table_index, category)

        mcs_index = torch.tensor(
            [[3.0, 8.0, 14.0]], dtype=torch.float32, device=device
        )
        table_index = torch.tensor(1, device=device)
        category = torch.tensor(0, device=device)
        actual = decode(mcs_index, table_index, category)
        expected = decoder(mcs_index, table_index, category)
        torch.testing.assert_close(actual, expected)


class TestCalculateTbSizeVectorized:
    """Vectorized calculate_tb_size without precomputed num_coded_bits."""

    def test_vector_without_num_coded_bits(self):
        result = calculate_tb_size(
            modulation_order=torch.tensor([4, 4]),
            target_coderate=torch.tensor([0.5, 0.5]),
            num_prbs=torch.tensor([10, 20]),
            num_ofdm_symbols=torch.tensor([14, 14]),
            num_dmrs_per_prb=torch.tensor([12, 12]),
        )
        tb_size = result[0]
        assert tb_size.shape == (2,)

        # Match scalar path elementwise
        for i, n_prb in enumerate([10, 20]):
            scalar = calculate_tb_size(
                modulation_order=4,
                target_coderate=0.5,
                num_prbs=n_prb,
                num_ofdm_symbols=14,
                num_dmrs_per_prb=12,
            )
            assert int(tb_size[i]) == int(scalar[0])

    def test_broadcast_scalar_layers(self):
        result = calculate_tb_size(
            modulation_order=torch.tensor([4, 6]),
            target_coderate=torch.tensor([0.5, 0.6]),
            num_prbs=torch.tensor([8, 16]),
            num_ofdm_symbols=14,
            num_dmrs_per_prb=12,
            num_layers=2,
        )
        assert result[0].shape == (2,)

    def test_compiled_fullgraph(self, device):
        """Grid-parameter validation must not break the compiled graph."""

        @torch.compile(fullgraph=True)
        def calculate(q, rate, num_prbs, num_symbols, num_dmrs):
            return calculate_tb_size(
                modulation_order=q,
                target_coderate=rate,
                num_prbs=num_prbs,
                num_ofdm_symbols=num_symbols,
                num_dmrs_per_prb=num_dmrs,
                return_cw_length=False,
            )

        args = (
            torch.tensor([4, 6], device=device),
            torch.tensor([0.5, 0.6], device=device),
            torch.tensor([8, 16], device=device),
            torch.tensor([14, 14], device=device),
            torch.tensor([12, 12], device=device),
        )
        actual = calculate(*args)
        expected = calculate_tb_size(
            modulation_order=args[0],
            target_coderate=args[1],
            num_prbs=args[2],
            num_ofdm_symbols=args[3],
            num_dmrs_per_prb=args[4],
            return_cw_length=False,
        )
        torch.testing.assert_close(actual, expected)


class TestCodedBitHelpers:
    """TBS-capped vs uncapped rate-matching bit budgets."""

    def test_cap_distinguishes_tbs_and_codeword(self):
        # 14 symbols, 0 DMRS, 0 overhead -> 168 REs/PRB (> 156)
        capped = calculate_num_coded_bits(4, 1, 14, 0, 1)
        uncapped = calculate_codeword_bits(4, 1, 14, 0, 1)
        assert capped == 4 * 156
        assert uncapped == 4 * 168
        assert uncapped > capped

    def test_below_cap_helpers_agree(self):
        # 14 symbols, 12 DMRS -> 156 REs exactly at the cap boundary after
        # overhead; with more DMRS the helpers must match.
        capped = calculate_num_coded_bits(4, 10, 14, 24, 1)
        uncapped = calculate_codeword_bits(4, 10, 14, 24, 1)
        assert capped == uncapped

