#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Unit tests for LDPC HARQ-IR support (encoder and decoder)."""

import pytest
import torch

from sionna.phy.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder
from sionna.phy.mapping import BinarySource
from sionna.phy.fec.utils import GaussianPriorSource
from sionna.phy.utils import ebnodb2no


# =========================================================================
# Encoder tests
# =========================================================================

class TestLDPC5GEncoderHARQ:
    """Tests for HARQ-related functionality of LDPC5GEncoder."""

    # --- Backward compatibility ------------------------------------------

    @pytest.mark.parametrize("k,r", [(100, 0.5), (50, 0.34), (500, 0.75)])
    def test_rv0_matches_standard(self, device, k, r):
        """Encoding with rv=[0] must reproduce the standard (non-HARQ)
        output."""
        n = int(k / r)
        u = BinarySource(device=device)([10, k])
        enc = LDPC5GEncoder(k, n, device=device)

        c_std = enc(u)
        c_harq = enc(u, rv=[0])

        assert c_harq.shape == (10, 1, n)
        assert torch.equal(c_std, c_harq[:, 0, :])

    # --- Output shapes ---------------------------------------------------

    @pytest.mark.parametrize("rv_list", [[0], [0, 2], [0, 2, 3, 1], [0, 0]])
    def test_output_shape(self, device, rv_list):
        """Output shape must be [bs, len(rv), n]."""
        k, n, bs = 100, 200, 8
        enc = LDPC5GEncoder(k, n, device=device)
        u = BinarySource(device=device)([bs, k])

        c = enc(u, rv=rv_list)
        assert c.shape == (bs, len(rv_list), n)

    @pytest.mark.parametrize(
        "shape", [[10, 100], [2, 5, 100], [1, 3, 4, 100]]
    )
    def test_multi_dim_input(self, device, shape):
        """HARQ encoding must support arbitrary leading dimensions."""
        k, n = 100, 200
        enc = LDPC5GEncoder(k, n, device=device)
        u = BinarySource(device=device)(shape)

        c = enc(u, rv=[0, 2])
        expected = list(shape[:-1]) + [2, n]
        assert list(c.shape) == expected

    # --- Correctness -----------------------------------------------------

    def test_all_zero_codeword(self, device):
        """All-zero input must give all-zero codeword for every RV."""
        k, n = 100, 200
        enc = LDPC5GEncoder(k, n, device=device)
        u = torch.zeros(4, k, device=device)

        c = enc(u, rv=[0, 1, 2, 3])
        assert torch.equal(c, torch.zeros_like(c))

    @pytest.mark.parametrize("num_bits_per_symbol", [2, 4])
    def test_output_interleaver_per_rv(self, device, num_bits_per_symbol):
        """Output interleaver is applied independently per RV and can be
        undone."""
        k, n, bs = 100, 200, 5
        u = BinarySource(device=device)([bs, k])

        enc_plain = LDPC5GEncoder(k, n, device=device)
        enc_int = LDPC5GEncoder(
            k, n, num_bits_per_symbol=num_bits_per_symbol, device=device,
        )

        c_plain = enc_plain(u, rv=[0, 2])
        c_int = enc_int(u, rv=[0, 2])

        c_deint = c_int[:, :, enc_int.out_int_inv]
        assert torch.equal(c_plain, c_deint)

    # --- Error handling --------------------------------------------------

    def test_invalid_rv_raises(self, device):
        """Invalid RV index must raise ValueError."""
        enc = LDPC5GEncoder(k=100, n=200, device=device)
        u = torch.zeros(1, 100, device=device)

        with pytest.raises(ValueError, match="Invalid RV index"):
            enc(u, rv=[5])

    def test_empty_rv_raises(self, device):
        """Empty rv list must raise ValueError."""
        enc = LDPC5GEncoder(k=100, n=200, device=device)
        u = torch.zeros(1, 100, device=device)

        with pytest.raises(ValueError, match="non-empty"):
            enc(u, rv=[])

    # --- Properties ------------------------------------------------------

    def test_encoder_properties(self):
        """Verify HARQ-related encoder properties."""
        enc = LDPC5GEncoder(k=100, n=200)

        assert enc.k_filler == enc.k_ldpc - enc.k
        assert enc.n_cb == enc.n_ldpc - 2 * enc.z
        assert enc.n_cb_comp == enc.n_cb - enc.k_filler
        assert isinstance(enc.rv_starts, list)
        assert len(enc.rv_starts) == 4
        assert enc.rv_starts[0] == 0


# =========================================================================
# Decoder tests
# =========================================================================

class TestLDPC5GDecoderHARQ:
    """Tests for HARQ-related functionality of LDPC5GDecoder."""

    # --- Construction ----------------------------------------------------

    def test_pruning_disabled_for_harq(self, device):
        """Pruning must be silently disabled when harq_mode=True."""
        enc = LDPC5GEncoder(k=100, n=200, device=device)
        dec = LDPC5GDecoder(
            enc, prune_pcm=True, harq_mode=True, device=device,
        )
        assert dec._nb_pruned_nodes == 0

    # --- Backward compatibility ------------------------------------------

    def test_standard_path_unchanged(self, device):
        """Decoder without harq_mode must produce the same output as
        the HARQ decoder defaulting to rv=[0] (both with pruning off)."""
        k, n, bs = 100, 200, 10
        llr = GaussianPriorSource(device=device)([bs, n], 0.5)

        enc = LDPC5GEncoder(k, n, device=device)
        dec_std = LDPC5GDecoder(
            enc, prune_pcm=False, num_iter=5, hard_out=False, device=device,
        )
        dec_harq = LDPC5GDecoder(
            enc, num_iter=5, hard_out=False, harq_mode=True, device=device,
        )

        out_std = dec_std(llr)
        out_harq = dec_harq(llr)
        assert torch.allclose(out_std, out_harq, atol=1e-5)

    # --- Noiseless decoding --------------------------------------------

    @pytest.mark.parametrize(
        "k,r", [(100, 0.5), (50, 0.34), (200, 0.75)]
    )
    def test_roundtrip_perfect_llrs(self, device, k, r):
        """Perfect (noiseless) LLRs must decode to the original bits for
        various RV combinations."""
        n = int(k / r)
        bs = 10
        u = BinarySource(device=device)([bs, k])

        enc = LDPC5GEncoder(k, n, device=device)
        dec = LDPC5GDecoder(enc, num_iter=20, harq_mode=True, device=device)

        for rv_list in [[0], [0, 2], [0, 2, 3, 1]]:
            c = enc(u, rv=rv_list)
            llr = 2.0 * (2.0 * c - 1.0)
            u_hat = dec(llr, rv=rv_list)
            assert torch.equal(u, u_hat), (
                f"Roundtrip failed for rv={rv_list}, k={k}, n={n}"
            )

    def test_roundtrip_duplicate_rv(self, device):
        """Chase combining (duplicate RV) must decode correctly."""
        k, n, bs = 100, 200, 10
        u = BinarySource(device=device)([bs, k])

        enc = LDPC5GEncoder(k, n, device=device)
        dec = LDPC5GDecoder(enc, num_iter=20, harq_mode=True, device=device)

        c = enc(u, rv=[0, 0])
        llr = 2.0 * (2.0 * c - 1.0)
        u_hat = dec(llr, rv=[0, 0])
        assert torch.equal(u, u_hat)

    # --- Output shape and modes ------------------------------------------

    @pytest.mark.parametrize("return_infobits", [True, False])
    def test_return_infobits(self, device, return_infobits):
        """Both return_infobits modes must work in HARQ."""
        k, n, bs = 100, 200, 5
        u = BinarySource(device=device)([bs, k])

        enc = LDPC5GEncoder(k, n, device=device)
        dec = LDPC5GDecoder(
            enc, num_iter=20, return_infobits=return_infobits,
            harq_mode=True, device=device,
        )

        c = enc(u, rv=[0, 2])
        out = dec(2.0 * (2.0 * c - 1.0), rv=[0, 2])

        expected_last = k if return_infobits else n
        assert out.shape == (bs, expected_last)

    # --- Error handling --------------------------------------------------

    def test_invalid_rv_raises(self, device):
        """Decoder must reject the same invalid RV values as the encoder."""
        enc = LDPC5GEncoder(k=100, n=200, device=device)
        dec = LDPC5GDecoder(enc, harq_mode=True, device=device)
        llr = torch.zeros(1, 200, device=device)

        with pytest.raises(ValueError, match="Invalid RV index"):
            dec(llr, rv=[-1])
        with pytest.raises(ValueError, match="Invalid RV index"):
            dec(llr, rv=[4])
        with pytest.raises(ValueError, match="non-empty"):
            dec(llr, rv=[])

    def test_rv_dim_mismatch_raises(self, device):
        """Mismatched rv dimension must raise ValueError."""
        k, n = 100, 200
        enc = LDPC5GEncoder(k, n, device=device)
        dec = LDPC5GDecoder(enc, harq_mode=True, device=device)

        llr = torch.zeros(5, 3, n, device=device)
        with pytest.raises(ValueError, match="must equal len\\(rv\\)"):
            dec(llr, rv=[0, 2])  # rv has 2 but dim -2 is 3

    # --- Noisy performance -----------------------------------------------

    def test_more_rvs_improves_ber(self, device):
        """Adding a second RV at moderate noise must not degrade BER."""
        k, n, bs = 100, 300, 200
        u = BinarySource(device=device)([bs, k])
        sigma = 1.5

        enc = LDPC5GEncoder(k, n, device=device)
        dec = LDPC5GDecoder(enc, num_iter=20, harq_mode=True, device=device)

        torch.manual_seed(42)
        c1 = enc(u, rv=[0])
        y1 = (2.0 * c1 - 1.0) + sigma * torch.randn_like(c1)
        u_hat1 = dec(2.0 * y1 / sigma**2, rv=[0])
        ber1 = (u != u_hat1).float().mean()

        torch.manual_seed(42)
        c2 = enc(u, rv=[0, 2])
        y2 = (2.0 * c2 - 1.0) + sigma * torch.randn_like(c2)
        u_hat2 = dec(2.0 * y2 / sigma**2, rv=[0, 2])
        ber2 = (u != u_hat2).float().mean()

        assert ber2 <= ber1 + 1e-6, (
            f"Two-RV BER ({ber2:.4f}) should not exceed single-RV "
            f"({ber1:.4f})"
        )

    def test_two_rvs_vs_lower_rate_noisy(self, device):
        """Under noise, HARQ rv=[0,1] and standard rate k/(2n) should
        achieve comparable BER (both transmit 2n bits for k info bits)."""
        k, n, bs = 100, 200, 500
        u = BinarySource(device=device)([bs, k])
        sigma = 1.0

        enc = LDPC5GEncoder(k, n, device=device)
        dec = LDPC5GDecoder(enc, num_iter=20, harq_mode=True, device=device)
        torch.manual_seed(0)
        c_rvs = enc(u, rv=[0, 1])
        y = (2.0 * c_rvs - 1.0) + sigma * torch.randn_like(c_rvs)
        u_hat_harq = dec(2.0 * y / sigma**2, rv=[0, 1])
        ber_harq = (u != u_hat_harq).float().mean()

        enc2 = LDPC5GEncoder(k, 2 * n, device=device)
        dec2 = LDPC5GDecoder(enc2, num_iter=20, device=device)
        torch.manual_seed(0)
        c_std = enc2(u)
        y2 = (2.0 * c_std - 1.0) + sigma * torch.randn_like(c_std)
        u_hat_std = dec2(2.0 * y2 / sigma**2)
        ber_std = (u != u_hat_std).float().mean()

        assert abs(ber_harq - ber_std) < 0.05, (
            f"BER gap too large: HARQ={ber_harq:.4f}, std={ber_std:.4f}"
        )

    def test_ber_improves_with_each_rv(self, device):
        """BER must monotonically improve as RVs are added: rv=[0] worse
        than [0,1] worse than [0,1,2] worse than [0,1,2,3]."""
        k, n, bs = 100, 300, 1000
        u = BinarySource(device=device)([bs, k])
        sigma = 1.2

        enc = LDPC5GEncoder(k, n, device=device)
        dec = LDPC5GDecoder(enc, num_iter=20, harq_mode=True, device=device)

        rv_configs = [[0], [0, 1], [0, 1, 2], [0, 1, 2, 3]]
        bers = []

        for rv_list in rv_configs:
            torch.manual_seed(99)
            c = enc(u, rv=rv_list)
            x = 2.0 * c - 1.0
            y = x + sigma * torch.randn_like(x)
            llr = 2.0 * y / sigma**2
            u_hat = dec(llr, rv=rv_list)
            bers.append((u != u_hat).float().mean().item())

        for i in range(len(bers) - 1):
            assert bers[i + 1] <= bers[i] + 1e-6, (
                f"BER did not improve: rv_configs[{i}]={bers[i]:.4f} "
                f"vs rv_configs[{i+1}]={bers[i+1]:.4f}"
            )

    def test_chase_combining_3db_gain(self, device):
        """Chase combining (rv=[0,0]) should behave like a ~3 dB SNR gain.

        Sending the same RV twice through independent AWGN channels and
        combining the LLRs doubles the effective SNR (= +3 dB).  We
        verify that a single RV at ``ebno + 3 dB`` achieves similar BER
        to two copies of RV 0 at ``ebno``.
        """
        k, n, bs = 100, 300, 2000
        source = BinarySource(device=device)
        u = source([bs, k])
        coderate = k / n

        enc = LDPC5GEncoder(k, n, device=device)
        dec = LDPC5GDecoder(enc, num_iter=20, harq_mode=True, device=device)

        ebno_db = 1.5
        no = float(ebnodb2no(ebno_db, num_bits_per_symbol=1, coderate=coderate))
        no_3db = float(ebnodb2no(ebno_db + 3.0, num_bits_per_symbol=1, coderate=coderate))

        # --- Single RV at ebno + 3 dB ---
        torch.manual_seed(7)
        c = enc(u, rv=[0])
        x = 2.0 * c - 1.0
        y = x + (no_3db ** 0.5) * torch.randn_like(x)
        u_hat_1rv = dec(2.0 * y / no_3db, rv=[0])
        ber_1rv_3db = (u != u_hat_1rv).float().mean()

        # --- Chase combining (rv=[0,0]) at ebno, independent noise ---
        torch.manual_seed(7)
        c2 = enc(u, rv=[0, 0])
        x2 = 2.0 * c2 - 1.0
        y2 = x2 + (no ** 0.5) * torch.randn_like(x2)
        llr2 = 2.0 * y2 / no
        u_hat_chase = dec(llr2, rv=[0, 0])
        ber_chase = (u != u_hat_chase).float().mean()

        # Both see ~same effective SNR; BER should be comparable
        assert abs(ber_chase - ber_1rv_3db) < 0.03, (
            f"Chase combining BER ({ber_chase:.4f}) should be close to "
            f"single-RV at +3dB ({ber_1rv_3db:.4f})"
        )


# =========================================================================
# Spec conformance
# =========================================================================

class TestHARQSpecConformance:
    """Verify RV start positions and compressed-buffer logic against
    TS 38.212."""

    @pytest.mark.parametrize(
        "k,n,bg",
        [
            (100, 200, "bg1"),
            (500, 1000, "bg1"),
            (2000, 4000, "bg1"),
            (50, 150, "bg2"),
            (200, 600, "bg2"),
            (500, 1500, "bg2"),
        ],
    )
    def test_rv_start_positions(self, k, n, bg):
        """RV start positions must match TS 38.212 Table 5.4.2.1-2.

        With I_LBRM=0, k0 = coeff * Z for the standard base-graph
        column coefficients.
        """
        enc = LDPC5GEncoder(k, n, bg=bg)
        z = enc.z

        if bg == "bg1":
            expected = [0, 17 * z, 33 * z, 56 * z]
        else:
            expected = [0, 13 * z, 25 * z, 43 * z]

        assert enc.rv_starts == expected

    @pytest.mark.parametrize("k,n", [(100, 200), (200, 600)])
    def test_k0_comp_identity_before_filler(self, k, n):
        """k0 values before the filler region must be unchanged by
        _k0_comp."""
        enc = LDPC5GEncoder(k, n)
        filler_start = k - 2 * enc.z
        for k0 in range(0, min(filler_start, 10)):
            assert enc._k0_comp(k0) == k0

    @pytest.mark.parametrize("k,n", [(100, 200), (200, 600)])
    def test_k0_comp_after_filler(self, k, n):
        """k0 values beyond the filler region must be shifted down by
        k_filler."""
        enc = LDPC5GEncoder(k, n)
        filler_end = k - 2 * enc.z + enc.k_filler
        k0 = filler_end + 5
        assert enc._k0_comp(k0) == k0 - enc.k_filler

    @pytest.mark.parametrize("k,n", [(100, 200), (200, 600)])
    def test_compressed_buffer_consistency(self, k, n):
        """n_cb_comp must equal n_cb minus filler bits."""
        enc = LDPC5GEncoder(k, n)
        assert enc.n_cb_comp == enc.n_cb - enc.k_filler
        assert enc.n_cb == enc.n_ldpc - 2 * enc.z
