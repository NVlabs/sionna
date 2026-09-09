#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Unit tests for sionna.phy.fec.polar.encoding."""

import os
import numpy as np
import pytest
import torch

from sionna.phy.fec.polar.encoding import PolarEncoder, Polar5GEncoder
from sionna.phy.fec.polar.utils import (
    generate_5g_ranking,
    generate_polar_transform_mat,
)
from sionna.phy.mapping import BinarySource


# Get reference data directory
test_dir = os.path.dirname(os.path.abspath(__file__))
ref_path = os.path.join(test_dir, "..", "..", "codes", "polar")


class TestPolarEncoder:
    """Tests for the PolarEncoder class."""

    @pytest.mark.parametrize(
        "k,n,k_des,n_des",
        [
            (-1, 10, 1, 32),
            (10, -3, 10, 32),
            ("1.0", 10, 1, 32),
            (3, "10.", 3, 32),
            (10, 9, 10, 32),
        ],
    )
    def test_invalid_inputs(self, k, n, k_des, n_des):
        """Test against invalid values of n and k."""
        frozen_pos, _ = generate_5g_ranking(k_des, n_des)
        with pytest.raises(BaseException):
            PolarEncoder(frozen_pos, n)

    @pytest.mark.parametrize(
        "k,n",
        [(0, 32), (10, 32), (32, 32), (100, 256), (123, 1024), (1024, 1024)],
    )
    def test_valid_inputs(self, k, n):
        """Test that valid shapes are accepted."""
        frozen_pos, _ = generate_5g_ranking(k, n)
        enc = PolarEncoder(frozen_pos, n)
        assert enc.k == k
        assert enc.n == n

    @pytest.mark.parametrize(
        "k,n",
        [(1, 32), (10, 32), (32, 32), (100, 256), (123, 1024), (1024, 1024)],
    )
    def test_output_dim(self, device, k, n):
        """Test that output dims are correct (=n) and output is all-zero
        codeword for all-zero input."""
        bs = 10
        frozen_pos, _ = generate_5g_ranking(k, n)
        enc = PolarEncoder(frozen_pos, n, device=device)
        u = torch.zeros([bs, k], device=device)
        c = enc(u)
        assert c.shape[-1] == n

        # Also check that all-zero input yields all-zero output
        c_hat = torch.zeros([bs, n], device=device)
        assert torch.equal(c, c_hat)

    def test_multi_dimensional(self, device):
        """Test against multi-dimensional shapes."""
        k = 120
        n = 256

        frozen_pos, _ = generate_5g_ranking(k, n)
        source = BinarySource(device=device)
        enc = PolarEncoder(frozen_pos, n, device=device)

        b = source([100, k])
        b_res = b.reshape([4, 5, 5, k])

        # Encode 2D Tensor
        c = enc(b)
        # Encode 4D Tensor
        c_res = enc(b_res)
        # And reshape to 2D shape
        c_res = c_res.reshape([100, n])
        # Both version should yield same result
        assert torch.equal(c, c_res)

    def test_torch_compile(self, device):
        """Test that torch.compile works as expected."""
        bs = 10
        k = 100
        n = 128
        source = BinarySource(device=device)
        frozen_pos, _ = generate_5g_ranking(k, n)
        enc = PolarEncoder(frozen_pos, n, device=device)

        @torch.compile(fullgraph=True)
        def run_graph(u):
            return enc(u)

        u = source([bs, k])
        x = run_graph(u)
        assert x.shape == (bs, n)

        # Execute the graph twice
        x = run_graph(u)
        assert x.shape == (bs, n)

        # And change batch_size
        u = source([bs + 1, k])
        x = run_graph(u)
        assert x.shape == (bs + 1, n)

    def test_non_binary_input_raises(self, device):
        """Binary checks stay enabled after a valid call; check_input=False skips them."""
        k, n, bs = 100, 256, 10
        frozen_pos, _ = generate_5g_ranking(k, n)
        enc = PolarEncoder(frozen_pos, n, device=device)
        u = torch.zeros(bs, k, device=device)
        enc(u)
        u[3, 7] = 2
        with pytest.raises(ValueError, match="Input must be binary"):
            enc(u)

        enc_off = PolarEncoder(frozen_pos, n, device=device, check_input=False)
        assert enc_off(u).shape[-1] == n

    def test_ref_implementation(self, device):
        """Test channel rankings against reference implementation based on
        polar transform matrix."""
        bs = 10
        k = 12
        n = 32

        frozen_pos, info_pos = generate_5g_ranking(k, n)
        source = BinarySource(device=device)
        enc = PolarEncoder(frozen_pos, n, device=device)

        b = source([bs, k])
        u = torch.zeros([bs, n], device=device)
        u[:, info_pos] = b

        # Call reference implementation
        gen_mat = generate_polar_transform_mat(int(np.log2(n)))
        gen_mat = torch.tensor(gen_mat, dtype=torch.float32, device=device)
        gen_mat = gen_mat.unsqueeze(0)

        u_exp = u.unsqueeze(1)
        c_ref = torch.matmul(u_exp, gen_mat)
        c_ref = torch.fmod(c_ref, 2)
        c_ref = c_ref.squeeze(1)

        # And run encoder (to be tested)
        c = enc(b)

        assert torch.equal(c.float(), c_ref)

    def test_batch(self, device):
        """Test that all samples in batch yield same output (for same input)."""
        bs = 100
        k = 120
        n = 256

        frozen_pos, _ = generate_5g_ranking(k, n)
        source = BinarySource(device=device)
        enc = PolarEncoder(frozen_pos, n, device=device)

        b = source([1, 15, k])
        b_rep = b.repeat([bs, 1, 1])
        c = enc(b_rep)

        for i in range(bs):
            assert torch.equal(c[0, :, :], c[i, :, :])

    @pytest.mark.parametrize(
        "dtype",
        [torch.float16, torch.float32, torch.float64, torch.int8, torch.int32],
    )
    def test_dtypes_flexible(self, device, dtype):
        """Test that encoder supports variable dtypes and yields same result."""
        bs = 10
        k = 32
        n = 64

        source = BinarySource(device=device)
        frozen_pos, _ = generate_5g_ranking(k, n)
        enc_ref = PolarEncoder(frozen_pos, n, device=device)

        u = source([bs, k])
        c_ref = enc_ref(u)

        enc = PolarEncoder(frozen_pos, n, device=device)
        u_dt = u.to(dtype)
        c = enc(u_dt)

        c_32 = c.to(torch.float32)

        assert torch.equal(c_ref.float(), c_32)


class TestPolar5GEncoder:
    """Tests for the Polar5GEncoder class including rate-matching.

    The layer inherits from PolarEncoder, thus many basic tests are
    already covered by the previous testcases.
    """

    @pytest.mark.parametrize(
        "k,n",
        [
            (-1, 30),
            (12, -3),
            ("12.", 30),
            (3, "10."),
            (10, 9),
            (10, 32),
            (10, 10),
            (1014, 1040),
            (1000, 1100),
            (100, 110),
        ],
    )
    def test_invalid_inputs(self, k, n):
        """Test against invalid values of n and k according to 38.212."""
        with pytest.raises((BaseException, ValueError)):
            Polar5GEncoder(k, n)

    @pytest.mark.parametrize(
        "k,n",
        [(12, 32), (20, 32), (100, 256), (243, 1024), (1013, 1088)],
    )
    def test_output_dim(self, device, k, n):
        """Test that output dims are correct (=n) and output is all-zero
        codeword for all-zero inputs."""
        bs = 10
        enc = Polar5GEncoder(k, n, device=device)
        u = torch.zeros([bs, k], device=device)
        c = enc(u)
        assert c.shape[-1] == n
        # Also check that all-zero input yields all-zero output
        c_hat = torch.zeros_like(c)
        assert torch.equal(c, c_hat)

    def test_non_binary_input_raises(self, device):
        """Binary checks stay enabled after a valid call; check_input=False skips them."""
        k, n, bs = 100, 200, 10
        enc = Polar5GEncoder(k, n, device=device)
        u = torch.zeros(bs, k, device=device)
        enc(u)
        u[3, 7] = 2
        with pytest.raises(ValueError, match="Input must be binary"):
            enc(u)

        enc_off = Polar5GEncoder(k, n, device=device, check_input=False)
        assert enc_off(u).shape[-1] == n

    def test_multi_dimensional(self, device):
        """Test against arbitrary shapes."""
        k = 56
        n = 240

        source = BinarySource(device=device)
        enc = Polar5GEncoder(k, n, device=device)

        b = source([100, k])
        b_res = b.reshape([4, 5, 5, k])

        # Encode 2D Tensor
        c = enc(b)
        # Encode 4D Tensor
        c_res = enc(b_res)
        # And reshape to 2D shape
        c_res = c_res.reshape([100, n])
        # Both version should yield same result
        assert torch.equal(c, c_res)

    def test_torch_compile(self, device):
        """Test that torch.compile works as expected."""
        bs = 10
        k = 100
        n = 135
        source = BinarySource(device=device)
        enc = Polar5GEncoder(k, n, device=device)

        @torch.compile(fullgraph=True)
        def run_graph(u):
            return enc(u)

        u = source([bs, k])
        x = run_graph(u)
        assert x.shape == (bs, n)

        # Execute the graph twice
        x = run_graph(u)
        assert x.shape == (bs, n)

        # And change batch_size
        u = source([bs + 1, k])
        x = run_graph(u)
        assert x.shape == (bs + 1, n)

    def test_batch(self, device):
        """Test that all samples in batch yield same output (for same input)."""
        bs = 100
        k = 120
        n = 253

        source = BinarySource(device=device)
        enc = Polar5GEncoder(k, n, device=device)

        b = source([1, 15, k])
        b_rep = b.repeat([bs, 1, 1])

        c = enc(b_rep)

        for i in range(bs):
            assert torch.equal(c[0, :, :], c[i, :, :])

    @pytest.mark.parametrize(
        "filename",
        [
            "E45_k30_K41",
            "E70_k32_K43",
            "E127_k29_K40",
            "E1023_k400_K411",
            "E70_k28_K39",
        ],
    )
    def test_ref_implementation(self, device, filename):
        """Test against pre-generated test cases.

        The test-cases include CRC-encoding and rate-matching and
        cover puncturing, shortening and repetition coding.
        """
        if not os.path.exists(ref_path):
            pytest.skip("Reference data not found")

        u_path = os.path.join(ref_path, filename + "_u.npy")
        c_path = os.path.join(ref_path, filename + "_c.npy")

        if not os.path.exists(u_path) or not os.path.exists(c_path):
            pytest.skip(f"Reference data for {filename} not found")

        u = np.load(u_path)
        u = torch.tensor(u, dtype=torch.float32, device=device)
        c_ref = np.load(c_path)

        k = u.shape[1]
        n = c_ref.shape[1]

        enc = Polar5GEncoder(k, n, device=device)
        c = enc(u)

        assert np.array_equal(c.cpu().numpy(), c_ref)

    @pytest.mark.parametrize(
        "dtype",
        [torch.float16, torch.float32, torch.float64, torch.int8, torch.int32],
    )
    def test_dtypes_flexible(self, device, dtype):
        """Test that encoder supports variable dtypes and yields same result."""
        bs = 10
        k = 32
        n = 64

        source = BinarySource(device=device)
        enc_ref = Polar5GEncoder(k, n, device=device)

        u = source([bs, k])
        c_ref = enc_ref(u)

        enc = Polar5GEncoder(k, n, device=device)
        u_dt = u.to(dtype)
        c = enc(u_dt)

        c_32 = c.to(torch.float32)

        assert torch.equal(c_ref.float(), c_32)

    @pytest.mark.parametrize("channel_type", ["uplink", "downlink"])
    def test_channel_types(self, device, channel_type):
        """Test that both uplink and downlink channel types work."""
        if channel_type == "uplink":
            k, n = 50, 100
        else:  # downlink has different constraints
            k, n = 50, 150

        source = BinarySource(device=device)
        enc = Polar5GEncoder(k, n, channel_type=channel_type, device=device)

        u = source([10, k])
        c = enc(u)

        assert c.shape == (10, n)

    def test_properties(self, device):
        """Test that properties return correct values."""
        k, n = 100, 200
        enc = Polar5GEncoder(k, n, device=device)

        assert enc.k == k
        assert enc.n == n
        assert enc.k_target == k
        assert enc.n_target == n
        assert enc.k_polar >= k  # Includes CRC bits
        assert enc.n_polar >= n  # Mother code might be larger
        assert enc.enc_crc is not None



