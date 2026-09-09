#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Unit tests for sionna.phy.fec.ldpc.encoding.LDPC5GEncoder."""

import os
import re
import numpy as np
import pytest
import torch

from sionna.phy import config
from sionna.phy.fec.ldpc import LDPC5GEncoder
from sionna.phy.mapping import BinarySource


class TestLDPC5GEncoder:
    """Tests for the LDPC5GEncoder class."""

    def test_invalid_inputs(self):
        """Test against invalid values of n and k."""
        param_invalid = [
            [-1, 10],
            [10, -3],
            ["a", 10],
            [3, "10"],
            [10, 9],
            [8500, 10000],
            [5000, 30000],
        ]  # (k, n)
        for p in param_invalid:
            with pytest.raises(BaseException):
                LDPC5GEncoder(p[0], p[1])

        param_valid = [
            [12, 20],
            [12, 30],
            [1000, 1566],
            [364, 1013],
            [948, 1024],
            [36, 100],
            [12, 18],
            [8448, 10000],
        ]  # (k, n)
        for p in param_valid:
            LDPC5GEncoder(p[0], p[1])

    @pytest.mark.parametrize("k", [12, 20, 100, 1234, 2000])
    @pytest.mark.parametrize("r", [0.34, 0.5, 0.7, 0.85])
    def test_output_dim(self, device, k, r):
        """Test that output dimensions are correct and all-zero input yields
        all-zero output.
        """
        n = int(k / r)
        if k > 3840 and r < 1 / 3:
            pytest.skip("Range not officially supported")

        enc = LDPC5GEncoder(k, n, device=device)
        bs = 10

        # Test for correct dimensions
        u = torch.zeros(bs, k, device=device)
        c = enc(u)
        assert c.shape[-1] == n

        # Test for all-zero codeword
        c_hat = torch.zeros(bs, n, device=device)
        assert torch.equal(c, c_hat)

    @pytest.mark.parametrize("k", [12, 100, 500])
    @pytest.mark.parametrize("r", [0.5, 0.75])
    def test_systematic_part(self, device, k, r):
        """Test that systematic part (excluding first 2z pos) is preserved."""
        n = int(k / r)
        enc = LDPC5GEncoder(k, n, device=device)
        source = BinarySource(device=device)
        bs = 10

        z = enc._z
        u = source([bs, k])
        c = enc(u)

        # Systematic part: u[2*z:] should equal c[:k-2*z]
        assert torch.equal(u[:, 2 * z :], c[:, : k - 2 * z])

    def test_non_binary_input_raises(self, device):
        """Binary checks stay enabled after a valid call; check_input=False skips them."""
        k, n, bs = 100, 200, 20
        enc = LDPC5GEncoder(k, n, device=device)
        u = torch.zeros(bs, k, device=device)
        enc(u)
        u[13, 37] = 2
        with pytest.raises(ValueError, match="Input must be binary"):
            enc(u)

        enc_off = LDPC5GEncoder(k, n, device=device, check_input=False)
        assert enc_off(u).shape[-1] == n

    def test_dim_mismatch(self, device):
        """Test that encoder raises error for inconsistent input dimensions."""
        k = 100
        n = 200
        bs = 20
        enc = LDPC5GEncoder(k + 1, n, device=device)

        # Test for wrong last dimension
        with pytest.raises(ValueError, match="Last dimension must be of length"):
            enc(torch.zeros(bs, k, device=device))

    @pytest.mark.parametrize(
        "shape",
        [[100], [10, 20, 30, 100], [1, 40, 100], [10, 2, 3, 4, 3, 100]],
    )
    def test_multi_dimensional(self, device, shape):
        """Test against arbitrary input shapes."""
        k = 100
        n = 200
        enc = LDPC5GEncoder(k, n, device=device)
        source = BinarySource(device=device)

        u = source(shape)
        u_ref = u.reshape(-1, k)

        c = enc(u)  # encode with shape
        c_ref = enc(u_ref)  # encode as 2-D array

        expected_shape = list(shape)
        expected_shape[-1] = n
        c_ref = c_ref.reshape(expected_shape)
        assert torch.equal(c, c_ref)

    def test_torch_compile(self, device):
        """Test that torch.compile works as expected."""
        k = 50
        n = 100
        bs = 10
        enc = LDPC5GEncoder(k, n, device=device)
        source = BinarySource(device=device)

        u = source([bs, k])

        # Test with torch.compile
        compiled_enc = torch.compile(enc, fullgraph=True)
        c = compiled_enc(u)
        assert c.shape == (bs, n)

        # Run again to test tracing stability
        c2 = compiled_enc(u)
        assert torch.equal(c, c2)

    @pytest.mark.parametrize("mode", ["default", "reduce-overhead"])
    def test_torch_compile_modes(self, device, mode):
        """Test that torch.compile works with different compilation modes."""
        if device == "cpu" and mode == "reduce-overhead":
            pytest.skip("reduce-overhead mode can be slow on CPU")

        k = 50
        n = 100
        bs = 10
        enc = LDPC5GEncoder(k, n, device=device)
        source = BinarySource(device=device)

        u = source([bs, k])

        # Reference without compilation
        c_ref = enc(u)

        # Compile and run
        compiled_enc = torch.compile(enc, mode=mode)
        c_compiled = compiled_enc(u)

        assert torch.equal(c_ref, c_compiled)

    def test_dtypes_flexible(self, device, precision):
        """Test that encoder supports variable dtypes and yields same result."""
        dt_supported = (
            torch.float16,
            torch.float32,
            torch.float64,
            torch.int32,
            torch.int64,
        )

        k = 100
        n = 200
        bs = 20
        enc_ref = LDPC5GEncoder(k, n, precision="single", device=device)
        source = BinarySource(device=device)

        u = source([bs, k])
        c_ref = enc_ref(u)

        for dt in dt_supported:
            enc = LDPC5GEncoder(k, n, precision=precision, device=device)
            u_dt = u.to(dt)
            c = enc(u_dt)

            c_32 = c.to(torch.float32)
            assert torch.equal(c_ref, c_32)

    def test_ldpc_interleaver(self):
        """Test that LDPC output interleaver pattern is correct."""
        enc = LDPC5GEncoder(k=12, n=20)

        params = [[12, 4], [100, 2], [80, 8]]
        for n, m in params:
            s, s_inv = enc.generate_out_int(n, m)

            idx = np.arange(n)
            idx_p = idx[s]
            idx_pp = idx_p[s_inv]

            # Test that interleaved vector is not the same
            assert not np.array_equal(idx, idx_p)
            # Test that interleaver can be inverted
            assert np.array_equal(idx, idx_pp)

        # Test that for m=1 no interleaving happens
        m = 1
        for n in [10, 100, 1000]:
            s, s_inv = enc.generate_out_int(n, m)
            idx = np.arange(n)
            assert np.array_equal(idx, s)
            assert np.array_equal(idx, s_inv)

    def test_properties(self):
        """Test that encoder properties return correct values."""
        k = 100
        n = 200
        enc = LDPC5GEncoder(k, n)

        assert enc.k == k
        assert enc.n == n
        assert enc.coderate == k / n
        assert enc.pcm is not None
        assert enc.z > 0
        assert enc.k_ldpc >= k
        assert enc.n_ldpc >= n

    def test_docstring_example(self):
        """Verify the docstring example works correctly."""
        # Create encoder for k=100 information bits and n=200 codeword bits
        encoder = LDPC5GEncoder(k=100, n=200)

        # Generate random information bits
        u = torch.randint(0, 2, (10, 100), dtype=torch.float32)
        c = encoder(u)
        assert c.shape == torch.Size([10, 200])

    @pytest.mark.parametrize("num_bits_per_symbol", [2, 4, 8])
    def test_output_interleaver(self, device, num_bits_per_symbol):
        """Test that output interleaver is correctly applied."""
        k = 100
        n = 200  # Must be divisible by num_bits_per_symbol
        bs = 10

        enc_no_int = LDPC5GEncoder(k, n, device=device)
        enc_with_int = LDPC5GEncoder(
            k, n, num_bits_per_symbol=num_bits_per_symbol, device=device
        )
        source = BinarySource(device=device)

        u = source([bs, k])
        c_no_int = enc_no_int(u)
        c_with_int = enc_with_int(u)

        # Codewords should be different (interleaved)
        assert not torch.equal(c_no_int, c_with_int)

        # But should be the same after de-interleaving
        c_deint = c_with_int[:, enc_with_int.out_int_inv]
        assert torch.equal(c_no_int, c_deint)

    @pytest.mark.parametrize("k", [12, 100, 500, 2000])
    @pytest.mark.parametrize("r", [0.34, 0.5, 0.75, 0.9])
    def test_parity_check(self, device, k, r):
        """Test that encoded codewords satisfy parity-check equations."""
        n = int(k / r)
        if k > 3840 and r < 1 / 3:
            pytest.skip("Range not officially supported")

        enc = LDPC5GEncoder(k, n, device=device)
        source = BinarySource(device=device)
        bs = 10

        u = source([bs, k])
        c = enc(u)

        # Get the full LDPC codeword (before rate-matching)
        # For this test we verify that k systematic bits are preserved
        z = enc._z

        # First 2*z bits are punctured, so c[0:k-2*z] should equal u[2*z:]
        assert torch.equal(c[:, : k - 2 * z], u[:, 2 * z :])

    def test_example_matrices(self, device):
        """Test encoding against reference generator matrices.
        
        Loads generator matrices from test/codes/ldpc/ and verifies that the
        encoder produces the same codewords as direct matrix multiplication.
        """
        bs = 10
        
        # Find the test codes directory
        # Try relative path from test file location
        test_dir = os.path.dirname(os.path.abspath(__file__))
        ref_path = os.path.join(test_dir, "..", "..", "codes", "ldpc")
        
        if not os.path.exists(ref_path):
            pytest.skip("Reference matrices not found")

        # Get all generator matrix files
        filenames = [f for f in os.listdir(ref_path) if f.endswith("_G.npy")]

        # Identify all k and n parameters from filenames
        params = []
        for s in filenames:
            m = re.match(r"k(\d+)_n(\d+)_G\.npy", s)
            if m is not None:
                params.append([int(m.group(1)), int(m.group(2))])

        assert len(params) > 0, "No reference matrices found"

        source = BinarySource(device=device)

        for p in params:
            k = int(p[0])
            n = int(p[1])

            # Load sparse generator matrix
            gm_sp = np.array(
                np.load(
                    os.path.join(ref_path, f"k{k}_n{n}_G.npy"),
                    allow_pickle=True,
                )
            )

            # Convert sparse format to dense matrix
            gm = np.zeros([k, n])
            for i in range(len(gm_sp[0, :])):
                c_idx = gm_sp[0, i]
                r_idx = gm_sp[1, i]
                gm[c_idx - 1, r_idx - 1] = 1

            gm_t = torch.tensor(gm, dtype=torch.float32, device=device)

            u = source([bs, k])
            enc = LDPC5GEncoder(k, n, device=device)
            c = enc(u)

            # Direct encoding via matrix multiplication
            c_ref = torch.matmul(u.unsqueeze(1), gm_t)
            c_ref = torch.fmod(c_ref, 2)
            c_ref = c_ref.squeeze(1)

            assert torch.equal(c, c_ref), f"not equal for k={k}, n={n}"

