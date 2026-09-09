#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Unit tests for sionna.phy.fec.polar.decoding."""

import os
import numpy as np
import pytest
import torch

from sionna.phy.fec.polar.encoding import PolarEncoder, Polar5GEncoder
from sionna.phy.fec.polar.decoding import (
    PolarSCDecoder,
    PolarSCLDecoder,
    PolarBPDecoder,
    Polar5GDecoder,
)
from sionna.phy.fec.polar.utils import generate_5g_ranking
from sionna.phy.fec.crc import CRCEncoder
from sionna.phy.fec.utils import GaussianPriorSource
from sionna.phy.mapping import BinarySource


# Get reference data directory
test_dir = os.path.dirname(os.path.abspath(__file__))
ref_path = os.path.join(test_dir, "..", "..", "codes", "polar")


class TestPolarSCDecoder:
    """Tests for the PolarSCDecoder class."""

    def test_invalid_inputs(self):
        """Test against invalid values of n and frozen_pos."""
        # frozen vec too long
        n = 32
        frozen_pos = np.arange(n + 1)
        with pytest.raises(BaseException):
            PolarSCDecoder(frozen_pos, n)

        # n not a pow of 2
        n = 32
        k = 12
        frozen_pos, _ = generate_5g_ranking(k, n)
        with pytest.raises(BaseException):
            PolarSCDecoder(frozen_pos, n + 1)

    def test_valid_inputs(self):
        """Test that valid shapes are accepted."""
        param_valid = [
            (0, 32),
            (10, 32),
            (32, 32),
            (100, 256),
            (123, 1024),
            (1024, 1024),
        ]

        for k, n in param_valid:
            frozen_pos, _ = generate_5g_ranking(k, n)
            dec = PolarSCDecoder(frozen_pos, n)
            assert dec.k == k
            assert dec.n == n

    @pytest.mark.parametrize(
        "k,n",
        [(1, 32), (10, 32), (32, 32), (100, 256), (123, 1024), (1024, 1024)],
    )
    def test_output_dim(self, device, k, n):
        """Test that output dims are correct (=k) and output equals all-zero
        codeword for all-zero LLR input."""
        bs = 10
        frozen_pos, _ = generate_5g_ranking(k, n)
        dec = PolarSCDecoder(frozen_pos, n, device=device)
        # all-zero with BPSK (no noise); logits
        c = -10.0 * torch.ones([bs, n], device=device)
        u = dec(c)
        assert u.shape[-1] == k
        # Also check that all-zero input yields all-zero output
        u_hat = torch.zeros([bs, k], device=device)
        assert torch.equal(u, u_hat)

    def test_numerical_stab(self, device):
        """Test for numerical stability (no nan or infty as output)."""
        bs = 10
        param_valid = [(1, 32), (10, 32), (32, 32), (100, 256)]
        source = GaussianPriorSource(device=device)

        for k, n in param_valid:
            frozen_pos, _ = generate_5g_ranking(k, n)
            dec = PolarSCDecoder(frozen_pos, n, device=device)

            # Case 1: extremely large inputs
            c = source([bs, n], 0.0001)
            u1 = dec(c)
            assert not torch.any(torch.isnan(u1))
            assert not torch.any(torch.isinf(u1))

            # Case 2: zero llr input
            c = torch.zeros([bs, n], device=device)
            u2 = dec(c)
            assert not torch.any(torch.isnan(u2))
            assert not torch.any(torch.isinf(u2))

    @pytest.mark.parametrize(
        "k,n",
        [(1, 32), (10, 32), (32, 32), (100, 256), (123, 1024), (1024, 1024)],
    )
    def test_identity(self, device, k, n):
        """Test that info bits can be recovered if no noise is added."""
        bs = 10

        source = BinarySource(device=device)

        frozen_pos, _ = generate_5g_ranking(k, n)
        enc = PolarEncoder(frozen_pos, n, device=device)
        dec = PolarSCDecoder(frozen_pos, n, device=device)

        u = source([bs, k])
        c = enc(u)
        llr_ch = 20.0 * (2.0 * c - 1)  # Demod BPSK without noise
        u_hat = dec(llr_ch)

        assert torch.equal(u, u_hat)

    def test_multi_dimensional(self, device):
        """Test against arbitrary shapes."""
        k = 120
        n = 256

        frozen_pos, _ = generate_5g_ranking(k, n)
        source = BinarySource(device=device)
        dec = PolarSCDecoder(frozen_pos, n, device=device)

        b = source([100, n])
        b_res = b.reshape([4, 5, 5, n])

        # Decode 2D Tensor
        c = dec(b)
        # Decode 4D Tensor
        c_res = dec(b_res)
        # And reshape to 2D shape
        c_res = c_res.reshape([100, k])
        # Both version should yield same result
        assert torch.equal(c, c_res)

    def test_batch(self, device):
        """Test that all samples in batch yield same output (for same input)."""
        bs = 100
        k = 120
        n = 256

        frozen_pos, _ = generate_5g_ranking(k, n)
        source = BinarySource(device=device)
        dec = PolarSCDecoder(frozen_pos, n, device=device)

        b = source([1, 15, n])
        b_rep = b.repeat([bs, 1, 1])

        c = dec(b_rep)

        for i in range(bs):
            assert torch.equal(c[0, :, :], c[i, :, :])

    def test_torch_compile(self, device):
        """Test that torch.compile works as expected."""
        bs = 10
        k = 100
        n = 128
        source = BinarySource(device=device)
        frozen_pos, _ = generate_5g_ranking(k, n)
        dec = PolarSCDecoder(frozen_pos, n, device=device)

        @torch.compile
        def run_graph(u):
            return dec(u)

        u = source([bs, n])
        x = run_graph(u)
        assert x.shape == (bs, k)

        # Execute the graph twice
        x = run_graph(u)
        assert x.shape == (bs, k)

    def test_ref_implementation(self, device):
        """Test against pre-calculated results from internal implementation."""
        if not os.path.exists(ref_path):
            pytest.skip("Reference data not found")

        filenames = ["P_128_37", "P_128_110", "P_256_128"]

        for f in filenames:
            a_path = os.path.join(ref_path, f + "_Avec.npy")
            l_path = os.path.join(ref_path, f + "_Lch.npy")
            u_path = os.path.join(ref_path, f + "_uhat.npy")

            if not all(os.path.exists(p) for p in [a_path, l_path, u_path]):
                pytest.skip(f"Reference data for {f} not found")

            A = np.load(a_path)
            llr_ch = np.load(l_path)
            u_hat = np.load(u_path)
            frozen_pos = np.array(np.where(A == 0)[0])
            info_pos = np.array(np.where(A == 1)[0])

            n = len(frozen_pos) + len(info_pos)

            dec = PolarSCDecoder(frozen_pos, n, device=device)
            l_in = -1.0 * llr_ch  # logits
            l_in_t = torch.tensor(l_in, dtype=torch.float32, device=device)
            u_hat_tf = dec(l_in_t)

            # The output should be equal to the reference
            assert np.array_equal(u_hat_tf.cpu().numpy(), u_hat)

    @pytest.mark.parametrize("precision", ["single", "double"])
    def test_dtype_flexible(self, device, precision):
        """Test that output_dtype can be flexible."""
        batch_size = 100
        k = 30
        n = 64
        source = GaussianPriorSource(device=device)
        frozen_pos, _ = generate_5g_ranking(k, n)

        dtypes_supported = {"single": torch.float32, "double": torch.float64}
        dt_out = dtypes_supported[precision]

        llr = source([batch_size, n], 0.5)

        dec = PolarSCDecoder(frozen_pos, n, precision=precision, device=device)
        x = dec(llr)
        assert x.dtype == dt_out


class TestPolarSCLDecoder:
    """Tests for the PolarSCLDecoder class."""

    def test_invalid_inputs(self):
        """Test against invalid values of n and frozen_pos."""
        # frozen vec too long
        n = 32
        frozen_pos = np.arange(n + 1)
        with pytest.raises(BaseException):
            PolarSCLDecoder(frozen_pos, n)

        # n not a pow of 2
        n = 32
        k = 12
        frozen_pos, _ = generate_5g_ranking(k, n)
        with pytest.raises(BaseException):
            PolarSCLDecoder(frozen_pos, n + 1)

        # list_size not a pow of 2
        with pytest.raises(ValueError):
            PolarSCLDecoder(frozen_pos, n, list_size=3)

        # return_crc_status without crc_degree
        with pytest.raises(ValueError):
            PolarSCLDecoder(frozen_pos, n, return_crc_status=True)

    def test_valid_inputs(self):
        """Test that valid shapes are accepted."""
        param_valid = [
            (0, 32),
            (10, 32),
            (32, 32),
            (100, 256),
        ]

        for k, n in param_valid:
            frozen_pos, _ = generate_5g_ranking(k, n)
            dec = PolarSCLDecoder(frozen_pos, n)
            assert dec.k == k
            assert dec.n == n

    @pytest.mark.parametrize("k,n", [(1, 32), (10, 32), (32, 32)])
    def test_output_dim(self, device, k, n):
        """Test that output dims are correct and output is all-zero codeword."""
        bs = 10
        frozen_pos, _ = generate_5g_ranking(k, n)
        dec = PolarSCLDecoder(frozen_pos, n, device=device)
        # all-zero with BPSK (no noise); logits
        c = -10.0 * torch.ones([bs, n], device=device)
        u = dec(c)
        assert u.shape[-1] == k
        u_hat = torch.zeros([bs, k], device=device)
        assert torch.equal(u, u_hat)

    def test_numerical_stab(self, device):
        """Test for numerical stability (no nan or infty as output)."""
        bs = 10
        param_valid = [(1, 32), (10, 32), (32, 32)]
        source = GaussianPriorSource(device=device)

        for k, n in param_valid:
            frozen_pos, _ = generate_5g_ranking(k, n)
            dec = PolarSCLDecoder(frozen_pos, n, device=device)

            # Case 1: extremely large inputs
            c = source([bs, n], 0.0001)
            u1 = dec(c)
            assert not torch.any(torch.isnan(u1))
            assert not torch.any(torch.isinf(u1))

            # Case 2: zero input
            c = torch.zeros([bs, n], device=device)
            u2 = dec(c)
            assert not torch.any(torch.isnan(u2))
            assert not torch.any(torch.isinf(u2))

    @pytest.mark.parametrize("k,n", [(1, 32), (10, 32), (32, 32)])
    def test_identity(self, device, k, n):
        """Test that info bits can be recovered if no noise is added."""
        bs = 10

        source = BinarySource(device=device)

        frozen_pos, _ = generate_5g_ranking(k, n)
        enc = PolarEncoder(frozen_pos, n, device=device)
        dec = PolarSCLDecoder(frozen_pos, n, device=device)

        u = source([bs, k])
        c = enc(u)
        llr_ch = 200.0 * (2.0 * c - 1)  # Demod BPSK without noise
        u_hat = dec(llr_ch)

        assert torch.equal(u, u_hat)

    def test_multi_dimensional(self, device):
        """Test against multi-dimensional input shapes."""
        k = 64
        n = 128

        frozen_pos, _ = generate_5g_ranking(k, n)
        source = BinarySource(device=device)
        dec = PolarSCLDecoder(frozen_pos, n, device=device)

        b = source([100, n])
        b_res = b.reshape([4, 5, 5, n])

        # Decode 2D Tensor
        c = dec(b)
        # Decode 4D Tensor
        c_res = dec(b_res)
        # And reshape to 2D shape
        c_res = c_res.reshape([100, k])
        # Both versions should yield same result
        assert torch.equal(c, c_res)

    def test_batch(self, device):
        """Test that all samples in batch yield same output (for same input)."""
        bs = 100
        k = 32
        n = 64

        frozen_pos, _ = generate_5g_ranking(k, n)
        source = BinarySource(device=device)
        dec = PolarSCLDecoder(frozen_pos, n, device=device)

        b = source([1, 4, n])
        b_rep = b.repeat([bs, 1, 1])

        c = dec(b_rep)

        for i in range(bs):
            assert torch.equal(c[0, :, :], c[i, :, :])

    @pytest.mark.parametrize("list_size", [1, 2, 8, 16])
    def test_list_sizes(self, device, list_size):
        """Test that different list sizes work."""
        bs = 10
        k = 16
        n = 32

        frozen_pos, _ = generate_5g_ranking(k, n)
        source = BinarySource(device=device)
        enc = PolarEncoder(frozen_pos, n, device=device)
        dec = PolarSCLDecoder(
            frozen_pos, n, list_size=list_size, device=device
        )

        u = source([bs, k])
        c = enc(u)
        llr_ch = 200.0 * (2.0 * c - 1)
        u_hat = dec(llr_ch)

        assert torch.equal(u, u_hat)

    def test_with_crc(self, device):
        """Test SCL decoding with CRC."""
        bs = 10
        k = 32
        n = 64
        crc_degree = "CRC11"

        frozen_pos, _ = generate_5g_ranking(k, n)
        source = BinarySource(device=device)
        enc = PolarEncoder(frozen_pos, n, device=device)
        enc_crc = CRCEncoder(crc_degree, device=device)
        k_crc = enc_crc.crc_length

        u = source([bs, k - k_crc])
        u_crc = enc_crc(u)
        c = enc(u_crc)
        llr_ch = 200.0 * (2.0 * c - 1)

        dec = PolarSCLDecoder(
            frozen_pos, n, crc_degree=crc_degree, device=device
        )
        u_hat = dec(llr_ch)

        assert torch.equal(u_crc, u_hat)

    @pytest.mark.parametrize("precision", ["single", "double"])
    def test_dtype_flexible(self, device, precision):
        """Test that output dtype is variable."""
        batch_size = 100
        k = 30
        n = 64
        source = GaussianPriorSource(device=device)
        frozen_pos, _ = generate_5g_ranking(k, n)

        dtypes_supported = {"single": torch.float32, "double": torch.float64}
        dt_out = dtypes_supported[precision]

        llr = source([batch_size, n], 0.5)

        dec = PolarSCLDecoder(
            frozen_pos, n, precision=precision, device=device
        )
        x = dec(llr)
        assert x.dtype == dt_out

    def test_return_crc_scl(self, device):
        """Test that correct CRC status is returned."""
        k = 32
        n = 64
        bs = 100

        source = BinarySource(device=device)

        frozen_pos, _ = generate_5g_ranking(k, n)
        enc = PolarEncoder(frozen_pos, n, device=device)
        crc_enc = CRCEncoder("CRC11", device=device)
        dec = PolarSCLDecoder(
            frozen_pos, n, crc_degree="CRC11",
            return_crc_status=True, device=device
        )
        k_crc = crc_enc.crc_length

        u = source((bs, 3, k - k_crc))
        u_crc = crc_enc(u)
        c = enc(u_crc)
        llr_ch = 20.0 * (2.0 * c - 1)  # No noise
        u_hat, crc_status = dec(llr_ch)

        # Without noise, CRC should always be valid
        assert crc_status.all()
        assert torch.equal(u_crc, u_hat)

    def test_torch_compile(self, device):
        """Test that torch.compile works for the tape-based SCL decoder."""
        bs = 10
        k = 100
        n = 128
        source = BinarySource(device=device)
        frozen_pos, _ = generate_5g_ranking(k, n)
        dec = PolarSCLDecoder(frozen_pos, n, device=device)

        @torch.compile
        def run_graph(u):
            return dec(u)

        u = source([bs, n])
        x = run_graph(u)
        assert x.shape == (bs, k)

        # Execute the graph twice to exercise the cached compiled artifact
        x = run_graph(u)
        assert x.shape == (bs, k)


class TestPolarBPDecoder:
    """Tests for the PolarBPDecoder class."""

    def test_invalid_inputs(self):
        """Test against invalid values of n and frozen_pos."""
        # frozen vec too long
        n = 32
        frozen_pos = np.arange(n + 1)
        with pytest.raises(BaseException):
            PolarBPDecoder(frozen_pos, n)

        # n not a pow of 2
        n = 32
        k = 12
        frozen_pos, _ = generate_5g_ranking(k, n)
        with pytest.raises(BaseException):
            PolarBPDecoder(frozen_pos, n + 1)

    def test_num_iter_setter(self, device):
        """Setter must reject the same invalid values as the constructor."""
        frozen_pos, _ = generate_5g_ranking(12, 32)
        dec = PolarBPDecoder(frozen_pos, 32, num_iter=1, device=device)
        with pytest.raises(ValueError, match="positive"):
            dec.num_iter = 0
        with pytest.raises(TypeError, match="integer"):
            dec.num_iter = 1.5
        dec.num_iter = 4
        assert dec.num_iter == 4

    def test_valid_inputs(self):
        """Test that valid shapes are accepted."""
        param_valid = [
            (0, 32),
            (10, 32),
            (32, 32),
            (100, 256),
            (123, 1024),
            (1024, 1024),
        ]

        for k, n in param_valid:
            frozen_pos, _ = generate_5g_ranking(k, n)
            dec = PolarBPDecoder(frozen_pos, n)
            assert dec.k == k
            assert dec.n == n

    @pytest.mark.parametrize(
        "k,n", [(1, 32), (10, 32), (32, 32), (100, 256), (123, 1024)]
    )
    @pytest.mark.parametrize("hard_out", [True, False])
    def test_output_dim(self, device, k, n, hard_out):
        """Test that output dims are correct and output is all-zero codeword."""
        bs = 10
        frozen_pos, _ = generate_5g_ranking(k, n)
        dec = PolarBPDecoder(frozen_pos, n, hard_out=hard_out, device=device)
        # all-zero with BPSK (no noise); logits
        c = -10.0 * torch.ones([bs, n], device=device)
        u = dec(c)
        assert u.shape[-1] == k
        if hard_out:
            u_hat = torch.zeros([bs, k], device=device)
            assert torch.equal(u, u_hat)

    @pytest.mark.parametrize(
        "k,n", [(1, 32), (10, 32), (32, 32), (100, 256), (123, 1024)]
    )
    def test_identity(self, device, k, n):
        """Test that info bits can be recovered if no noise is added."""
        bs = 10

        source = BinarySource(device=device)
        frozen_pos, _ = generate_5g_ranking(k, n)
        enc = PolarEncoder(frozen_pos, n, device=device)
        dec = PolarBPDecoder(frozen_pos, n, device=device)

        u = source([bs, k])
        c = enc(u)
        llr_ch = 20.0 * (2.0 * c - 1)  # Demod BPSK without noise
        u_hat = dec(llr_ch)

        assert torch.equal(u, u_hat)

    def test_multi_dimensional(self, device):
        """Test against arbitrary shapes."""
        k = 120
        n = 256

        frozen_pos, _ = generate_5g_ranking(k, n)
        source = BinarySource(device=device)
        dec = PolarBPDecoder(frozen_pos, n, device=device)

        shapes = [[4, 5, 5], []]
        for s_ in shapes:
            s = s_.copy()
            bs = int(np.prod(s)) if len(s) > 0 else 1
            b = source([bs, n])
            s.append(n)
            if len(s) > 1:
                b_res = b.reshape(s)
            else:
                b_res = b

            # Decode 2D Tensor
            c = dec(b)
            # Decode 4D Tensor
            c_res = dec(b_res)
            # And reshape to 2D shape
            c_res = c_res.reshape([bs, k])
            # Both versions should yield same result
            assert torch.equal(c, c_res)

    def test_batch(self, device):
        """Test that all samples in batch yield same output (for same input)."""
        bs = 100
        k = 120
        n = 256

        frozen_pos, _ = generate_5g_ranking(k, n)
        source = BinarySource(device=device)
        dec = PolarBPDecoder(frozen_pos, n, device=device)

        b = source([1, 15, n])
        b_rep = b.repeat([bs, 1, 1])

        c = dec(b_rep)

        for i in range(bs):
            assert torch.equal(c[0, :, :], c[i, :, :])

    def test_numerics(self, device):
        """Test for numerical stability with large llrs and many iterations."""
        bs = 100
        k = 120
        n = 256
        num_iter = 200

        for hard_out in [False, True]:
            frozen_pos, _ = generate_5g_ranking(k, n)
            source = GaussianPriorSource(device=device)
            dec = PolarBPDecoder(
                frozen_pos, n, hard_out=hard_out, num_iter=num_iter, device=device
            )

            b = source([bs, n], 0.001)  # Very large llrs

            c = dec(b)

            # All values are finite (not nan and not inf)
            assert torch.all(torch.isfinite(c))

    @pytest.mark.parametrize("precision", ["single", "double"])
    def test_dtype_flexible(self, device, precision):
        """Test that output dtype can be variable."""
        batch_size = 100
        k = 30
        n = 64
        source = GaussianPriorSource(device=device)
        frozen_pos, _ = generate_5g_ranking(k, n)

        dtypes_supported = {"single": torch.float32, "double": torch.float64}
        dt_out = dtypes_supported[precision]

        llr = source([batch_size, n], 0.5)

        dec = PolarBPDecoder(frozen_pos, n, precision=precision, device=device)
        x = dec(llr)
        assert x.dtype == dt_out

    def test_ref_implementation(self, device):
        """Test against NumPy reference implementation.

        Tests both hard and soft output.
        """

        def boxplus_np(x, y, llr_max):
            """Check node update (boxplus) for LLRs in numpy."""
            x_in = np.maximum(np.minimum(x, llr_max), -llr_max)
            y_in = np.maximum(np.minimum(y, llr_max), -llr_max)
            llr_out = np.log(1 + np.exp(x_in + y_in))
            llr_out -= np.log(np.exp(x_in) + np.exp(y_in))
            return llr_out

        def decode_bp_np(llr_ch_np, n_iter, frozen_pos, info_pos, llr_max):
            """NumPy reference BP decoder."""
            n = llr_ch_np.shape[-1]
            bs = llr_ch_np.shape[0]
            n_stages = int(np.log2(n))

            msg_r = np.zeros([bs, n_stages + 1, n])
            msg_l = np.zeros([bs, n_stages + 1, n])

            # Init llr_ch
            msg_l[:, n_stages, :] = -1 * llr_ch_np

            # Init frozen positions with infty
            msg_r[:, 0, frozen_pos] = llr_max

            # Decode
            for _ in range(n_iter):
                # Update r messages
                for s in range(n_stages):
                    ind_range = np.arange(int(n / 2))
                    ind_1 = ind_range * 2 - np.mod(ind_range, 2**s)
                    ind_2 = ind_1 + 2**s

                    l1_in = msg_l[:, s + 1, ind_1]
                    l2_in = msg_l[:, s + 1, ind_2]
                    r1_in = msg_r[:, s, ind_1]
                    r2_in = msg_r[:, s, ind_2]

                    msg_r[:, s + 1, ind_1] = boxplus_np(r1_in, l2_in + r2_in, llr_max)
                    msg_r[:, s + 1, ind_2] = boxplus_np(r1_in, l1_in, llr_max) + r2_in

                # Update l messages
                for s in range(n_stages - 1, -1, -1):
                    ind_range = np.arange(int(n / 2))
                    ind_1 = ind_range * 2 - np.mod(ind_range, 2**s)
                    ind_2 = ind_1 + 2**s

                    l1_in = msg_l[:, s + 1, ind_1]
                    l2_in = msg_l[:, s + 1, ind_2]
                    r1_in = msg_r[:, s, ind_1]
                    r2_in = msg_r[:, s, ind_2]

                    msg_l[:, s, ind_1] = boxplus_np(l1_in, l2_in + r2_in, llr_max)
                    msg_l[:, s, ind_2] = boxplus_np(r1_in, l1_in, llr_max) + l2_in

            # Recover u_hat
            u_hat_soft = msg_l[:, 0, info_pos]
            u_hat = 0.5 * (1 - np.sign(u_hat_soft))
            return u_hat, u_hat_soft

        # Test parameters
        noise_var = 0.3
        num_iters = [5, 10, 20]
        llr_max = 19.3
        bs = 100
        n = 128
        k = 64
        frozen_pos, info_pos = generate_5g_ranking(k, n)

        for num_iter in num_iters:
            source = GaussianPriorSource(device=device)
            llr_ch = source([bs, n], noise_var)

            # Decode with PyTorch implementation
            dec_bp = PolarBPDecoder(
                frozen_pos, n, hard_out=True, num_iter=num_iter, device=device
            )
            dec_bp_soft = PolarBPDecoder(
                frozen_pos, n, hard_out=False, num_iter=num_iter, device=device
            )

            u_hat_bp = dec_bp(llr_ch).cpu().numpy()
            u_hat_bp_soft = dec_bp_soft(llr_ch).cpu().numpy()

            # Decode with NumPy reference
            llr_ch_np = llr_ch.cpu().numpy()
            u_hat_ref, u_hat_ref_soft = decode_bp_np(
                llr_ch_np, num_iter, frozen_pos, info_pos, llr_max
            )

            # The hard output should be equal to the reference
            assert np.array_equal(u_hat_bp, u_hat_ref)
            # The soft output should be close (with sign flip for logits)
            assert np.allclose(
                -u_hat_bp_soft, u_hat_ref_soft, rtol=5e-2, atol=5e-3
            )


class TestPolar5GDecoder:
    """Tests for the Polar5GDecoder class."""

    def test_invalid_inputs(self):
        """Test against invalid input values."""
        enc = Polar5GEncoder(40, 60)
        with pytest.raises(BaseException):
            Polar5GDecoder(enc, dec_type=1)
        with pytest.raises(BaseException):
            Polar5GDecoder(enc, dec_type="ABC")
        with pytest.raises(BaseException):
            Polar5GDecoder("SC")

    @pytest.mark.parametrize(
        "k,n", [(12, 20), (20, 44), (100, 257), (123, 897)]
    )
    @pytest.mark.parametrize("dec_type", ["SC", "SCL", "BP"])
    def test_identity_uplink(self, device, k, n, dec_type):
        """Test that info bits can be recovered for uplink scenario."""
        bs = 10

        source = BinarySource(device=device)
        enc = Polar5GEncoder(k, n, channel_type="uplink", device=device)
        dec = Polar5GDecoder(enc, dec_type=dec_type, device=device)

        u = source([bs, k])
        c = enc(u)
        assert c.shape[-1] == n
        llr_ch = 20.0 * (2.0 * c - 1)  # Demod BPSK without noise
        u_hat = dec(llr_ch)

        assert torch.equal(u, u_hat)

    @pytest.mark.parametrize("k,n", [(1, 25), (20, 44), (140, 576)])
    @pytest.mark.parametrize("dec_type", ["SC", "SCL", "BP"])
    def test_identity_downlink(self, device, k, n, dec_type):
        """Test that info bits can be recovered for downlink scenario."""
        bs = 10

        source = BinarySource(device=device)
        enc = Polar5GEncoder(k, n, channel_type="downlink", device=device)
        dec = Polar5GDecoder(enc, dec_type=dec_type, device=device)

        u = source([bs, k])
        c = enc(u)
        assert c.shape[-1] == n
        llr_ch = 20.0 * (2.0 * c - 1)  # Demod BPSK without noise
        u_hat = dec(llr_ch)

        assert torch.equal(u, u_hat)

    def test_multi_dimensional(self, device):
        """Test against arbitrary shapes."""
        k = 120
        n = 237

        source = BinarySource(device=device)

        for dec_type in ["SC", "SCL", "BP"]:
            for ch_type in ["uplink"]:  # downlink has k<=140 constraint
                enc = Polar5GEncoder(k, n, channel_type=ch_type, device=device)
                dec = Polar5GDecoder(enc, dec_type=dec_type, device=device)

                b = source([100, n])
                b_res = b.reshape([4, 5, 5, n])

                # Decode 2D Tensor
                c = dec(b)
                # Decode 4D Tensor
                c_res = dec(b_res)
                # And reshape to 2D shape
                c_res = c_res.reshape([100, k])
                # Both versions should yield same result
                assert torch.equal(c, c_res)

    def test_batch(self, device):
        """Test that all samples in batch yield same output (for same input)."""
        bs = 100
        k = 95
        n = 145

        enc = Polar5GEncoder(k, n, device=device)
        source = GaussianPriorSource(device=device)

        for dec_type in ["SC", "SCL", "BP"]:
            dec = Polar5GDecoder(enc, dec_type=dec_type, device=device)

            llr = source([1, 4, n], 0.5)
            llr_rep = llr.repeat([bs, 1, 1])

            c = dec(llr_rep)

            for i in range(bs):
                assert torch.equal(c[0, :, :], c[i, :, :])

    def test_torch_compile(self, device):
        """Test that torch.compile works for the supported decoders."""
        bs = 10
        k = 45
        n = 67
        enc = Polar5GEncoder(k, n, device=device)
        source = GaussianPriorSource(device=device)

        for dec_type in ["SC", "SCL", "BP"]:
            dec = Polar5GDecoder(enc, dec_type=dec_type, device=device)

            @torch.compile
            def run_graph(u):
                return dec(u)

            u = source([bs, n], 0.5)
            x = run_graph(u)
            assert x.shape == (bs, k)

            # Execute the graph twice
            x = run_graph(u)
            assert x.shape == (bs, k)

    @pytest.mark.parametrize("precision", ["single", "double"])
    def test_dtype_flexible(self, device, precision):
        """Test that output dtype can be variable."""
        batch_size = 100
        k = 30
        n = 64
        source = GaussianPriorSource(device=device)
        enc = Polar5GEncoder(k, n, device=device)

        dtypes_supported = {"single": torch.float32, "double": torch.float64}
        dt_out = dtypes_supported[precision]

        llr = source([batch_size, n], 0.5)

        dec = Polar5GDecoder(enc, precision=precision, device=device)
        x = dec(llr)
        assert x.dtype == dt_out

    @pytest.mark.parametrize("channel_type", ["uplink", "downlink"])
    def test_return_crc(self, device, channel_type):
        """Test that correct CRC status is returned."""
        if channel_type == "uplink":
            k, n = 32, 64
        else:
            k, n = 32, 80  # Downlink needs more space for CRC24

        bs = 100

        source = BinarySource(device=device)

        enc = Polar5GEncoder(k, n, channel_type=channel_type, device=device)
        dec = Polar5GDecoder(enc, "SCL", return_crc_status=True, device=device)

        u = source((bs, 3, k))
        c = enc(u)
        llr_ch = 20.0 * (2.0 * c - 1)  # No noise
        u_hat, crc_status = dec(llr_ch)

        # Without noise, CRC should always be valid
        assert crc_status.all()
        assert torch.equal(u, u_hat)

