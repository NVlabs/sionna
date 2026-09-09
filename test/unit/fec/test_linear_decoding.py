#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Unit tests for sionna.phy.fec.linear.OSDecoder."""

import numpy as np
import pytest
import scipy as sp
import torch

from sionna.phy import Block
from sionna.phy.fec.linear import LinearEncoder, OSDecoder
from sionna.phy.fec.utils import GaussianPriorSource, load_parity_check_examples, pcm2gm
from sionna.phy.utils import ebnodb2no, sim_ber
from sionna.phy.channel.awgn import AWGN
from sionna.phy.mapping import Mapper, Demapper, BinarySource


class SystemModel(Block):
    """System model for channel coding BER simulations."""

    def __init__(
        self,
        encoder: LinearEncoder,
        decoder: OSDecoder,
        precision: str = None,
        device: str = None,
    ):
        super().__init__(precision=precision, device=device)

        self.source = BinarySource(precision=precision, device=device)
        self.channel = AWGN(precision=precision, device=device)
        self.mapper = Mapper("pam", 1, precision=precision, device=device)
        self.demapper = Demapper("app", "pam", 1, precision=precision, device=device)

        self.decoder = decoder
        self.encoder = encoder
        self._coderate = encoder.k / encoder.n

    @property
    def coderate(self) -> float:
        """Code rate."""
        return self._coderate

    def call(self, batch_size: int, ebno_db: float) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the system model."""
        no = ebnodb2no(ebno_db, coderate=self.coderate, num_bits_per_symbol=1)

        b = self.source([batch_size, self.encoder.k])
        c = self.encoder(b)
        x = self.mapper(c)
        y = self.channel(x, no)
        llr_ch = self.demapper(y, no)
        c_hat = self.decoder(llr_ch)
        return c, c_hat


class TestOSDecoder:
    """Tests for the OSDecoder class."""

    def test_numerical_stability(self, device):
        """Test numerical stability of the decoder for large and small LLRs."""
        bs = 100
        pcm, k, _, _ = load_parity_check_examples(1)
        enc = LinearEncoder(pcm, is_pcm=True, device=device)
        dec = OSDecoder(pcm, is_pcm=True, device=device)
        source = BinarySource(device=device)

        u = source((bs, k))
        c = enc(u)

        # Very large LLRs (decoder clips internally at 100)
        llr_ch = 1000.0 * (2 * c - 1)
        c_hat = dec(llr_ch)
        assert torch.equal(c_hat, c)

        # Very small LLRs (but still correct)
        llr_ch = 0.0001 * (2 * c - 1)
        c_hat = dec(llr_ch)
        assert torch.equal(c_hat, c)

    @pytest.mark.parametrize("n, t", [
        (10, 1), (10, 2), (10, 3), (10, 4), (10, 5),
        (45, 1), (45, 2), (45, 3), (45, 4), (45, 5),
        (100, 1), (100, 2), (100, 3),
        (250, 1), (250, 2), (250, 3),
    ])
    def test_error_patterns(self, n, t):
        """Test that _num_error_patterns() returns correct values."""
        pcm, _, _, _ = load_parity_check_examples(0)
        dec = OSDecoder(pcm, is_pcm=True)

        num_eps = dec._num_error_patterns(n, t)
        num_eps_ref = sp.special.comb(n, t, exact=True, repetition=False)
        assert num_eps == num_eps_ref

        ep = dec._gen_error_patterns(n, t)
        num_com = dec._num_error_patterns(n, t)
        assert num_com == len(ep), "Number of error patterns does not match."

    def test_dtype(self, device, precision):
        """Test support for variable precisions."""
        pcm, _, n, _ = load_parity_check_examples(1, verbose=False)
        gm = pcm2gm(pcm)

        shape = [100, n]
        source = GaussianPriorSource(device=device)

        # Only floating point is supported
        dt_map = {"single": torch.float32, "double": torch.float64}

        dec = OSDecoder(gm, precision=precision, device=device)
        llr_ch = source(shape, no=0.1)
        c = dec(llr_ch)
        # Output dtype must match precision
        assert c.dtype == dt_map[precision]

    def test_input_consistency(self, device):
        """Test against inconsistent inputs."""
        pcm_id = 2
        pcm, k, n, _ = load_parity_check_examples(pcm_id)
        bs = 20
        dec = OSDecoder(pcm, is_pcm=True, device=device)

        # Valid input
        dec(torch.zeros(bs, n, device=device))

        # Batch dimension is flexible
        dec(torch.zeros(bs + 1, n, device=device))

        # Test for invalid input shape
        with pytest.raises(ValueError, match="Last dimension must be of size"):
            dec(torch.zeros(bs, n + 1, device=device))

    def test_non_binary_matrix_gm(self):
        """Test that decoder raises error for non-binary generator matrix."""
        pcm, _, _, _ = load_parity_check_examples(2)
        pcm_modified = np.copy(pcm)
        pcm_modified[1, 2] = 2

        # Test for non-binary matrix (interpreted as gm)
        with pytest.raises(ValueError, match="must be binary"):
            OSDecoder(pcm_modified)

    def test_non_binary_matrix_pcm(self):
        """Test that decoder raises error for non-binary parity-check matrix."""
        pcm, _, _, _ = load_parity_check_examples(2)
        pcm_modified = np.copy(pcm)
        pcm_modified[3, 27] = 2

        # Test for non-binary matrix (as pcm)
        with pytest.raises(ValueError, match="must be binary"):
            OSDecoder(pcm_modified, is_pcm=True)

    def test_torch_compile(self, device):
        """Test that torch.compile works as expected."""
        pcm, _, n, _ = load_parity_check_examples(2)
        bs = 20
        dec = OSDecoder(pcm, is_pcm=True, device=device)
        source = GaussianPriorSource(device=device)

        u = source([bs, n], no=0.1)

        # Test with torch.compile
        compiled_dec = torch.compile(dec)
        c = compiled_dec(u)
        assert c.shape == (bs, n)

    @pytest.mark.parametrize("shape_prefix", [
        [],
        [10, 20, 30],
        [1, 40],
        [10, 2, 3, 4, 3],
    ])
    def test_multi_dimensional(self, device, shape_prefix):
        """Test against arbitrary input shapes.

        The decoder should only operate on axis=-1.
        """
        pcm_id = 3
        pcm, _, n, _ = load_parity_check_examples(pcm_id)
        s = shape_prefix + [n]
        dec = OSDecoder(pcm, is_pcm=True, t=2, device=device)
        source = GaussianPriorSource(device=device)

        llr = source(s, no=0.2)
        llr_ref = llr.reshape(-1, n)

        c = dec(llr)
        c_ref = dec(llr_ref)

        c_ref = c_ref.reshape(shape_prefix + [n])
        assert torch.equal(c, c_ref)

    @pytest.mark.parametrize(
        "pcm_id, t, snrs_ref, blers_ref, batch_size,"
        " num_target_block_errors, max_mc_iter, compile_mode",
        [
            pytest.param(
                0, 2,
                torch.linspace(0, 5, 6),
                np.array([1.832e-01, 1.253e-01, 7.047e-02,
                          2.899e-02, 1.252e-02, 4.371e-03]),
                1000, 10000, 500, None,
                id="hamming_7_4",
            ),
            pytest.param(
                1, 4,
                torch.tensor([0, 1.5, 3.0, 4]),
                # Converged reference (~3000 block errors/point), averaged
                # over torch 2.12 and 2.13 which agree within MC noise.
                np.array([7.08e-01, 2.68e-01, 2.61e-02, 1.94e-03]),
                200, 1000, 1000, "default",
                id="bch_63_45",
            ),
        ],
    )
    def test_reference(
        self, device, pcm_id, t, snrs_ref, blers_ref,
        batch_size, num_target_block_errors, max_mc_iter, compile_mode,
    ):
        """Test against reference ML results."""
        pcm, k, n, coderate = load_parity_check_examples(pcm_id)
        encoder = LinearEncoder(pcm, is_pcm=True, device=device)
        decoder = OSDecoder(encoder=encoder, t=t, device=device)

        model = SystemModel(encoder, decoder, device=device)

        sim_kwargs = dict(
            ebno_dbs=snrs_ref,
            batch_size=batch_size,
            max_mc_iter=max_mc_iter,
            num_target_block_errors=num_target_block_errors,
            device=device,
        )
        if compile_mode is not None:
            sim_kwargs["compile_mode"] = compile_mode

        _, bler = sim_ber(model, **sim_kwargs)
        bler_np = bler.cpu().numpy()
        assert np.all(np.isclose(bler_np, blers_ref, rtol=0.2))

    def test_properties(self):
        """Test that decoder properties return correct values."""
        pcm_id = 0  # (7,4) Hamming code
        pcm, k, n, coderate = load_parity_check_examples(pcm_id)
        dec = OSDecoder(pcm, is_pcm=True, t=2)

        assert dec.k == k
        assert dec.n == n
        assert dec.t == 2
        assert dec.gm.shape == (k, n)

    def test_docstring_example(self):
        """Verify the docstring example works correctly."""
        # Load (7,4) Hamming code
        pcm, k, n, _ = load_parity_check_examples(0)
        encoder = LinearEncoder(pcm, is_pcm=True)
        decoder = OSDecoder(encoder=encoder, t=2)

        # Generate random codeword and add noise
        u = torch.randint(0, 2, (10, k), dtype=torch.float32)
        c = encoder(u)
        llr_ch = 2.0 * (2.0 * c - 1.0)  # Perfect LLRs
        c_hat = decoder(llr_ch)
        assert torch.equal(c, c_hat)

    def test_encoder_initialization(self, device):
        """Test that decoder can be initialized from encoder."""
        pcm, k, n, _ = load_parity_check_examples(0)
        encoder = LinearEncoder(pcm, is_pcm=True, device=device)
        decoder = OSDecoder(encoder=encoder, t=1, device=device)

        assert decoder.k == encoder.k
        assert decoder.n == encoder.n

    def test_invalid_t_type(self):
        """Test that non-integer t raises TypeError."""
        pcm, _, _, _ = load_parity_check_examples(0)
        with pytest.raises(TypeError, match="t must be int"):
            OSDecoder(pcm, is_pcm=True, t=1.5)

    def test_missing_enc_mat_and_encoder(self):
        """Test that ValueError is raised when both enc_mat and encoder are None."""
        with pytest.raises(ValueError, match="enc_mat cannot be None"):
            OSDecoder()

    def test_invalid_is_pcm_type(self):
        """Test that non-boolean is_pcm raises TypeError."""
        pcm, _, _, _ = load_parity_check_examples(0)
        with pytest.raises(TypeError, match="is_pcm must be bool"):
            OSDecoder(pcm, is_pcm="yes")

