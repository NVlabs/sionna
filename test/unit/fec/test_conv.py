#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Unit tests for sionna.phy.fec.conv module."""

import os

import numpy as np
import pytest
import torch

from sionna.phy import config
from sionna.phy.fec.conv import (
    ConvEncoder,
    ViterbiDecoder,
    BCJRDecoder,
    polynomial_selector,
    Trellis,
)
from sionna.phy.fec.utils import GaussianPriorSource
from sionna.phy.mapping import BinarySource
from sionna.phy.utils.misc import ebnodb2no


# Get test data directory
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
REF_PATH = os.path.join(TEST_DIR, "..", "..", "codes", "conv")


class TestPolynomialSelector:
    """Tests for the polynomial_selector function."""

    @pytest.mark.parametrize("rate", [1/2, 1/3])
    @pytest.mark.parametrize("cs", [3, 4, 5, 6, 7, 8])
    def test_valid_rates_and_constraint_lengths(self, rate, cs):
        """Test that valid rate/constraint_length combinations return polynomials."""
        gen_poly = polynomial_selector(rate, cs)
        assert isinstance(gen_poly, tuple)
        assert all(isinstance(p, str) for p in gen_poly)
        assert len(gen_poly[0]) == cs

    def test_invalid_rate(self):
        """Test that invalid rates raise ValueError."""
        with pytest.raises(ValueError):
            polynomial_selector(rate=0.25, constraint_length=5)

    def test_invalid_constraint_length(self):
        """Test that invalid constraint lengths raise ValueError."""
        with pytest.raises(ValueError):
            polynomial_selector(rate=0.5, constraint_length=2)
        with pytest.raises(ValueError):
            polynomial_selector(rate=0.5, constraint_length=9)

    def test_constraint_length_type(self):
        """Test that non-integer constraint_length raises TypeError."""
        with pytest.raises(TypeError):
            polynomial_selector(rate=0.5, constraint_length=5.0)


class TestTrellis:
    """Tests for the Trellis class."""

    def test_trellis_creation(self, device):
        """Test basic trellis creation."""
        gen_poly = ('101', '111')
        trellis = Trellis(gen_poly, device=device)

        assert trellis.ns == 4  # 2^(3-1)
        assert trellis.conv_n == 2
        assert trellis.conv_k == 1
        assert trellis.to_nodes.shape == (4, 2)
        assert trellis.from_nodes.shape == (4, 2)
        assert trellis.op_mat.shape == (4, 4)

    def test_trellis_device_movement(self, device):
        """Test moving trellis between devices."""
        gen_poly = ('101', '111')
        trellis = Trellis(gen_poly, device="cpu")
        trellis.to(device)

        assert trellis.to_nodes.device.type == device.split(":")[0]


class TestConvEncoder:
    """Tests for the ConvEncoder class."""

    @pytest.mark.parametrize("rate", [1/2, 1/3])
    @pytest.mark.parametrize("k", [10, 20, 50, 100])
    @pytest.mark.parametrize("rsc,terminate",
                             [(False, False), (True, False), (False, True)])
    def test_output_dim(self, device, rate, k, rsc, terminate):
        """Test that output dimensions are correct and all-zero input yields
        all-zero output.
        """
        bs = 10
        mu = 4
        enc = ConvEncoder(rate=rate, constraint_length=mu+1, rsc=rsc,
                          terminate=terminate, device=device)

        n_expected = int(k / rate)
        if terminate:
            n_expected += int(mu / rate)

        u = torch.zeros(bs, k, device=device)
        c = enc(u)
        assert c.shape[-1] == n_expected

        c_hat = torch.zeros(bs, n_expected, device=device)
        assert torch.equal(c, c_hat)

    def test_invalid_inputs(self):
        """Test that invalid rate values and constraint lengths raise errors."""
        rate_invalid = [0.2, 0.45, 0.01]
        rate_valid = [1/3, 1/2]

        constraint_length_invalid = [2, 9, 0]
        constraint_length_valid = [3, 4, 5, 6, 7, 8]

        for rate in rate_valid:
            for mu in constraint_length_invalid:
                with pytest.raises(ValueError):
                    ConvEncoder(rate=rate, constraint_length=mu)

        for rate in rate_invalid:
            for mu in constraint_length_valid:
                with pytest.raises(ValueError):
                    ConvEncoder(rate=rate, constraint_length=mu)

    @pytest.mark.parametrize("gen_poly", [['101', '111'], ('101', '111')])
    @pytest.mark.parametrize("rsc", [False, True])
    def test_polynomial_input(self, device, gen_poly, rsc):
        """Test that different formats of input polynomials are accepted."""
        bs = 10
        k = 100
        n = 200
        u = torch.zeros(bs, k, device=device)

        enc = ConvEncoder(gen_poly=gen_poly, rsc=rsc, device=device)
        c = enc(u)
        assert c.shape[-1] == n
        c_hat = torch.zeros(bs, n, device=device)
        assert torch.equal(c, c_hat)

    def test_invalid_polynomials(self):
        """Test that invalid generator polynomials raise errors."""
        # Different lengths
        with pytest.raises(ValueError):
            ConvEncoder(gen_poly=['1001', '111'])

        # Non-binary characters
        with pytest.raises(ValueError):
            ConvEncoder(gen_poly=['1211', '1101'])

        # Non-string elements
        with pytest.raises(TypeError):
            ConvEncoder(gen_poly=['1001', 111])

    @pytest.mark.parametrize("terminate", [False, True])
    @pytest.mark.parametrize("shape", [[4, 5, 5], []])
    def test_multi_dimensional(self, device, terminate, shape):
        """Test against arbitrary input shapes."""
        k = 120
        rate = 1/2
        mu = 4

        source = BinarySource(device=device)
        enc = ConvEncoder(rate=rate, constraint_length=mu+1,
                          terminate=terminate, device=device)

        n = int(k / rate)
        if terminate:
            n += int(mu / rate)

        s = shape.copy()
        bs = int(np.prod(s)) if s else 1
        b = source([bs, k])
        if s:
            s.append(k)
            b_res = b.reshape(s)
        else:
            b_res = b.reshape(k)

        c = enc(b)
        c_res = enc(b_res)

        expected_shape = list(b_res.shape[:-1]) + [n]
        assert list(c_res.shape) == expected_shape

        if s:
            c_res_flat = c_res.reshape(bs, n)
        else:
            c_res_flat = c_res.reshape(1, n)
        assert torch.equal(c, c_res_flat)

    @pytest.mark.parametrize("gen_poly,gen_str,rate,mu", [
        (['101', '111'], 'conv_rate_half_57_', 1/2, 3),
        (['1101', '1111'], 'conv_rate_half_6474_', None, None),
        (['101', '111', '111'], 'conv_rate_onethird_577_', 1/3, 3),
        (['101', '111', '111', '111'], 'conv_rate_onefourth_5777_', None, None),
    ])
    def test_ref_implementation(self, device, gen_poly, gen_str, rate, mu):
        """Test against pre-encoded codewords from reference implementation."""
        if not os.path.exists(REF_PATH):
            pytest.skip("Reference data not found")

        enc = ConvEncoder(gen_poly=gen_poly, device=device)

        u = np.load(os.path.join(REF_PATH, gen_str + 'ref_u.npy'))
        cref = np.load(os.path.join(REF_PATH, gen_str + 'ref_x.npy'))

        u_t = torch.tensor(u, dtype=torch.float32, device=device)
        c = enc(u_t)
        assert np.array_equal(c.cpu().numpy(), cref)

        if rate is not None:
            enc2 = ConvEncoder(rate=rate, constraint_length=mu,
                               device=device)
            c2 = enc2(u_t)
            assert np.array_equal(c2.cpu().numpy(), cref)

    @pytest.mark.parametrize("terminate,cl", [(False, 8), (True, 7)])
    def test_batch(self, device, terminate, cl):
        """Test that all samples in batch yield same output for same input."""
        bs = 100
        k = 117

        source = BinarySource(device=device)
        enc = ConvEncoder(rate=0.5, constraint_length=cl,
                          terminate=terminate, device=device)

        b = source([1, 15, k])
        b_rep = b.repeat(bs, 1, 1)
        c = enc(b_rep)

        for i in range(bs):
            assert torch.equal(c[0, :, :], c[i, :, :])

    @pytest.mark.parametrize("dt", [torch.float16, torch.float32,
                                     torch.float64, torch.int32, torch.int64])
    def test_dtypes_flexible(self, device, dt):
        """Test that encoder supports variable dtypes."""
        bs = 10
        k = 32

        source = BinarySource(device=device)
        enc_ref = ConvEncoder(rate=0.5, constraint_length=7, rsc=True,
                              precision="single", device=device)
        u = source([bs, k])
        c_ref = enc_ref(u)

        enc = ConvEncoder(rate=0.5, constraint_length=7, rsc=True,
                          precision="single", device=device)
        u_dt = u.to(dt)
        c = enc(u_dt)

        c_32 = c.to(torch.float32)
        assert torch.equal(c_ref, c_32)

    def test_torch_compile(self, device):
        """Test that torch.compile works as expected."""
        bs = 10
        k = 100

        source = BinarySource(device=device)
        enc = ConvEncoder(rate=0.5, constraint_length=7, device=device)

        u = source([bs, k])

        # Reference without compile
        c_ref = enc(u)

        # Compile and run
        compiled_enc = torch.compile(enc)
        c = compiled_enc(u)

        # Execute twice
        c2 = compiled_enc(u)

        assert torch.equal(c_ref, c)
        assert torch.equal(c, c2)

    def test_docstring_example(self):
        """Verify the docstring example works correctly."""
        encoder = ConvEncoder(rate=0.5, constraint_length=5)
        u = torch.randint(0, 2, (10, 100), dtype=torch.float32)
        c = encoder(u)
        assert c.shape == torch.Size([10, 200])


class TestViterbiDecoder:
    """Tests for the ViterbiDecoder class."""

    @pytest.mark.parametrize("rate", [1/2, 1/3])
    @pytest.mark.parametrize("k", [10, 22, 40])
    @pytest.mark.parametrize("rsc,terminate,cl",
                             [(False, False, 5), (True, False, 3),
                              (False, True, 4)])
    def test_output_dim(self, device, rate, k, rsc, terminate, cl):
        """Test that output dims are correct and all-zero codeword is decoded."""
        bs = 10
        muterm = 3
        dec = ViterbiDecoder(rate=rate, constraint_length=cl, rsc=rsc,
                             terminate=terminate, device=device)

        n = int(k / rate)
        if terminate:
            n += int(muterm / rate)

        c = -10. * torch.ones(bs, n, device=device)
        u = dec(c)

        assert u.shape[-1] == k
        u_hat = torch.zeros(bs, k, device=device)
        assert torch.equal(u, u_hat)

    def test_dynamic_shapes(self, device):
        """A reused decoder must rebuild when the codeword length changes."""
        dec = ViterbiDecoder(
            rate=1/2, constraint_length=5, terminate=False, device=device
        )
        u20 = dec(torch.zeros(2, 20, device=device))
        u40 = dec(torch.zeros(2, 40, device=device))
        assert u20.shape == (2, 10)
        assert u40.shape == (2, 20)

    @pytest.mark.parametrize("rate", [1/2, 1/3])
    @pytest.mark.parametrize("k", [10, 20, 60])
    def test_numerical_stability(self, device, rate, k):
        """Test for numerical stability (no nan or infinity)."""
        bs = 10
        source = GaussianPriorSource(device=device)
        n = int(k / rate)
        dec = ViterbiDecoder(rate=rate, constraint_length=5, device=device)

        c = source([bs, n], no=0.0001)
        u1 = dec(c)
        assert not torch.any(torch.isnan(u1))
        assert not torch.any(torch.isinf(u1))

        c = torch.zeros(bs, n, device=device)
        u2 = dec(c)
        assert not torch.any(torch.isnan(u2))
        assert not torch.any(torch.isinf(u2))

    @pytest.mark.parametrize("r", [1/3, 1/2])
    @pytest.mark.parametrize("cs", [3, 4, 5, 6])
    def test_init(self, device, r, cs):
        """Test different init methods lead to same result."""
        bs = 10
        n = 120
        no = 0.1
        source = GaussianPriorSource(device=device)
        enc = ConvEncoder(rate=r, constraint_length=cs, device=device)

        dec1 = ViterbiDecoder(gen_poly=enc.gen_poly, device=device)
        dec2 = ViterbiDecoder(rate=r, constraint_length=cs, device=device)

        llr = source([bs, n], no=no)
        x_hat1 = dec1(llr)
        x_hat2 = dec2(llr)
        assert torch.equal(x_hat1, x_hat2)

    @pytest.mark.parametrize("rate", [1/2, 1/3])
    @pytest.mark.parametrize("mu", [3, 8])
    def test_identity(self, device, rate, mu):
        """Test that info bits can be recovered if no noise is added."""
        bs = 10
        k = 35
        source = BinarySource(device=device)
        u = source([bs, k])

        enc = ConvEncoder(rate=rate, constraint_length=mu, device=device)
        cw = enc(u)
        code_syms = 20. * (2. * cw - 1)
        u_hat = ViterbiDecoder(
            gen_poly=enc.gen_poly, method='soft_llr', device=device
        )(code_syms)
        assert torch.equal(u, u_hat)

        u_hat_hard = ViterbiDecoder(
            gen_poly=enc.gen_poly, method='hard', device=device
        )(cw)
        assert torch.equal(u, u_hat_hard)

        enc_rsc = ConvEncoder(rate=rate, constraint_length=mu, rsc=True,
                              device=device)
        cw_rsc = enc_rsc(u)
        code_syms_rsc = 20. * (2. * cw_rsc - 1)
        u_hat_rsc = ViterbiDecoder(
            gen_poly=enc_rsc.gen_poly, method='soft_llr', rsc=True,
            device=device
        )(code_syms_rsc)
        assert torch.equal(u, u_hat_rsc)

    @pytest.mark.parametrize("terminate", [True, False])
    @pytest.mark.parametrize("shape", [[4, 5, 5], []])
    def test_multi_dimensional(self, device, terminate, shape):
        """Test against arbitrary input shapes."""
        k = 100
        rate = 1/2
        mu = 3
        source = BinarySource(device=device)

        dec = ViterbiDecoder(rate=rate, constraint_length=mu+1,
                             terminate=terminate, device=device)

        n = int(k / rate)
        if terminate:
            n += int(mu / rate)

        s = shape.copy()
        bs = int(np.prod(s)) if s else 1
        b = source([bs, n])

        if s:
            s.append(n)
            b_res = b.reshape(s)
        else:
            b_res = b.reshape(n)

        c = dec(b)
        c_res = dec(b_res)

        expected_shape = list(b_res.shape[:-1]) + [k]
        assert list(c_res.shape) == expected_shape

    def test_batch(self, device):
        """Test that all samples in batch yield same output for same input."""
        bs = 100
        n = 240

        source = GaussianPriorSource(device=device)
        dec = ViterbiDecoder(rate=1/2, constraint_length=3, device=device)

        b = source([1, n], no=1)
        b_rep = b.repeat(bs, 1)

        c = dec(b_rep)

        for i in range(bs):
            assert torch.equal(c[0, :], c[i, :])

    @pytest.mark.parametrize("gen_poly,gen_str", [
        (['101', '111'], 'conv_rate_half_57_'),
        (['1101', '1111'], 'conv_rate_half_6474_'),
        (['101', '111', '111'], 'conv_rate_onethird_577_'),
        (['101', '111', '111', '111'], 'conv_rate_onefourth_5777_'),
    ])
    def test_ref_implementation(self, device, gen_poly, gen_str):
        """Test against pre-decoded results from reference implementation."""
        if not os.path.exists(REF_PATH):
            pytest.skip("Reference data not found")

        dec = ViterbiDecoder(gen_poly=gen_poly, method='soft_llr',
                             device=device)

        yref = np.load(os.path.join(REF_PATH, gen_str + 'ref_y.npy'))
        uhat_ref = np.load(os.path.join(REF_PATH, gen_str + 'ref_uhat.npy'))

        no = ebnodb2no(torch.tensor(4.95), num_bits_per_symbol=2,
                       coderate=1.)
        yref_soft = torch.tensor(
            2 * yref / no.item(), dtype=torch.float32, device=device
        )
        uhat = dec(yref_soft)
        assert np.array_equal(uhat_ref, uhat.cpu().numpy())

    @pytest.mark.parametrize("p,dt_out",
                             [("single", torch.float32),
                              ("double", torch.float64)])
    @pytest.mark.parametrize("dt_in", [torch.float32, torch.float64])
    def test_dtype_flexible(self, device, p, dt_out, dt_in):
        """Test that output dtype can be flexible."""
        batch_size = 100
        n = 64
        source = GaussianPriorSource(device=device)

        llr = source([batch_size, n], no=0.5).to(dt_in)
        dec = ViterbiDecoder(rate=1/2, constraint_length=3,
                             precision=p, device=device)
        x = dec(llr)
        assert x.dtype == dt_out

    def test_torch_compile(self, device):
        """Test that torch.compile works."""
        bs = 10
        n = 128
        source = BinarySource(device=device)
        dec = ViterbiDecoder(rate=1/2, constraint_length=5, device=device)

        u = source([bs, n])
        x_ref = dec(u)

        compiled_dec = torch.compile(dec)
        x = compiled_dec(u)
        x2 = compiled_dec(u)

        assert torch.equal(x_ref, x)
        assert torch.equal(x, x2)

        # Test with different batch size (same n)
        u3 = source([bs + 1, n])
        x3 = compiled_dec(u3)
        assert x3.shape[0] == bs + 1

    def test_docstring_example(self):
        """Verify the docstring example works correctly."""
        decoder = ViterbiDecoder(rate=0.5, constraint_length=5)
        llr = torch.randn(10, 200)
        u_hat = decoder(llr)
        assert u_hat.shape == torch.Size([10, 100])


class TestBCJRDecoder:
    """Tests for the BCJRDecoder class."""

    @pytest.mark.parametrize("rate", [1/2, 1/3])
    @pytest.mark.parametrize("k", [10, 45])
    @pytest.mark.parametrize("rsc,terminate,cl",
                             [(False, False, 5), (True, False, 3),
                              (False, True, 6)])
    def test_output_dim(self, device, rate, k, rsc, terminate, cl):
        """Test that output dims are correct and all-zero codeword is decoded."""
        bs = 10
        muterm = 5
        dec = BCJRDecoder(rate=rate, constraint_length=cl, rsc=rsc,
                          terminate=terminate, device=device)

        n = int(k / rate)
        if terminate:
            n += int(muterm / rate)

        c = -10. * torch.ones(bs, n, device=device)
        u1 = dec(c)

        assert u1.shape[-1] == k
        u_hat = torch.zeros(bs, k, device=device)
        assert torch.equal(u1, u_hat)

    @pytest.mark.parametrize("rate", [1/2, 1/3])
    @pytest.mark.parametrize("k", [22, 55])
    def test_numerical_stability(self, device, rate, k):
        """Test for numerical stability (no nan or infinity)."""
        bs = 10
        source = GaussianPriorSource(device=device)
        n = int(k / rate)
        dec = BCJRDecoder(rate=rate, constraint_length=5, device=device)

        c = source([bs, n], no=0.0001)
        u1 = dec(c)
        assert not torch.any(torch.isnan(u1))
        assert not torch.any(torch.isinf(u1))

        c = torch.zeros(bs, n, device=device)
        u2 = dec(c)
        assert not torch.any(torch.isnan(u2))
        assert not torch.any(torch.isinf(u2))

    @pytest.mark.parametrize("r", [1/3, 1/2])
    @pytest.mark.parametrize("cs", [3, 4, 5, 6])
    def test_init(self, device, r, cs):
        """Test different init methods lead to same result."""
        bs = 10
        n = 120
        no = 0.1
        source = GaussianPriorSource(device=device)
        enc = ConvEncoder(rate=r, constraint_length=cs, device=device)

        dec1 = BCJRDecoder(gen_poly=enc.gen_poly, device=device)
        dec2 = BCJRDecoder(rate=r, constraint_length=cs, device=device)

        llr = source([bs, n], no=no)
        x_hat1 = dec1(llr)
        x_hat2 = dec2(llr)
        assert torch.equal(x_hat1, x_hat2)

    @pytest.mark.parametrize("rate", [1/2, 1/3])
    @pytest.mark.parametrize("mu", [3, 8])
    @pytest.mark.parametrize("alg", ["map", "log", "maxlog"])
    def test_identity(self, device, rate, mu, alg):
        """Test that info bits can be recovered if no noise is added."""
        bs = 5
        k = 40
        source = BinarySource(device=device)
        u = source([bs, k])

        enc = ConvEncoder(rate=rate, constraint_length=mu, device=device)
        cw = enc(u)
        code_syms = 20. * (2. * cw - 1)
        u_hat = BCJRDecoder(
            gen_poly=enc.gen_poly, algorithm=alg, device=device
        )(code_syms)
        assert torch.equal(u, u_hat)

        enc_rsc = ConvEncoder(rate=rate, constraint_length=mu, rsc=True,
                              device=device)
        cw_rsc = enc_rsc(u)
        code_syms_rsc = 20. * (2. * cw_rsc - 1)
        u_hat_rsc = BCJRDecoder(
            gen_poly=enc_rsc.gen_poly, algorithm=alg, rsc=True,
            device=device
        )(code_syms_rsc)
        assert torch.equal(u, u_hat_rsc)

    @pytest.mark.parametrize("alg", ["map", "log", "maxlog"])
    def test_algorithms(self, device, alg):
        """Test all three BCJR algorithms produce valid output."""
        bs = 10
        k = 40
        n = 80
        source = GaussianPriorSource(device=device)
        llr = source([bs, n], no=0.5)

        dec = BCJRDecoder(rate=1/2, constraint_length=4, algorithm=alg,
                          device=device)
        u = dec(llr)
        assert u.shape == (bs, k)
        assert not torch.any(torch.isnan(u))

    @pytest.mark.parametrize("terminate", [True, False])
    @pytest.mark.parametrize("shape", [[4, 5, 5], []])
    def test_multi_dimensional(self, device, terminate, shape):
        """Test against arbitrary input shapes."""
        k = 40
        rate = 1/2
        mu = 3
        source = BinarySource(device=device)

        dec = BCJRDecoder(rate=rate, constraint_length=mu+1,
                          terminate=terminate, device=device)

        n = int(k / rate)
        if terminate:
            n += int(mu / rate)

        s = shape.copy()
        bs = int(np.prod(s)) if s else 1
        b = source([bs, n])

        if s:
            s.append(n)
            b_res = b.reshape(s)
        else:
            b_res = b.reshape(n)

        c = dec(b)
        c_res = dec(b_res)

        expected_shape = list(b_res.shape[:-1]) + [k]
        assert list(c_res.shape) == expected_shape

    def test_batch(self, device):
        """Test that all samples in batch yield same output for same input."""
        bs = 100
        n = 240

        source = GaussianPriorSource(device=device)
        dec = BCJRDecoder(rate=1/2, constraint_length=3, device=device)

        b = source([1, n], no=1)
        b_rep = b.repeat(bs, 1)

        c = dec(b_rep)

        for i in range(bs):
            assert torch.equal(c[0, :], c[i, :])

    @pytest.mark.parametrize("gen_poly,gen_str", [
        (['101', '111'], 'conv_rate_half_57_'),
        (['1101', '1111'], 'conv_rate_half_6474_'),
        (['101', '111', '111'], 'conv_rate_onethird_577_'),
        (['101', '111', '111', '111'], 'conv_rate_onefourth_5777_'),
    ])
    def test_ref_implementation(self, device, gen_poly, gen_str):
        """Test against pre-decoded results from reference implementation."""
        if not os.path.exists(REF_PATH):
            pytest.skip("Reference data not found")

        dec = BCJRDecoder(gen_poly=gen_poly, device=device)

        yref = np.load(os.path.join(REF_PATH, gen_str + 'ref_y.npy'))
        uhat_ref = np.load(os.path.join(REF_PATH, gen_str + 'ref_uhat.npy'))

        yref_soft = torch.tensor(
            0.5 * (yref + 1), dtype=torch.float32, device=device
        )
        uhat = dec(yref_soft)
        assert np.array_equal(uhat_ref, uhat.cpu().numpy())

    @pytest.mark.parametrize("p,dt_out",
                             [("single", torch.float32),
                              ("double", torch.float64)])
    @pytest.mark.parametrize("dt_in", [torch.float32, torch.float64])
    def test_dtype_flexible(self, device, p, dt_out, dt_in):
        """Test that output dtype can be flexible."""
        batch_size = 100
        n = 64
        source = GaussianPriorSource(device=device)

        llr = source([batch_size, n], no=0.5).to(dt_in)
        dec = BCJRDecoder(rate=1/2, constraint_length=3,
                          precision=p, device=device)
        x = dec(llr)
        assert x.dtype == dt_out

    def test_torch_compile(self, device):
        """Test that torch.compile works (runs without error).

        Note: Unlike ViterbiDecoder, BCJRDecoder's iterative forward-backward
        algorithm may produce slightly different results when compiled due to
        tracing behavior. This test verifies the compiled version runs and
        produces valid output shapes, matching the original TF test behavior.
        """
        bs = 10
        n = 128
        k = n // 2  # rate=1/2
        source = BinarySource(device=device)
        dec = BCJRDecoder(rate=1/2, constraint_length=5, device=device)

        u = source([bs, n])

        # Compile and run
        compiled_dec = torch.compile(dec)
        x = compiled_dec(u)

        # Verify output shape is correct
        assert x.shape == (bs, k)

        # Execute twice to test graph reuse
        x2 = compiled_dec(u)
        assert x2.shape == (bs, k)

        # Test with different batch size (same n)
        u3 = source([bs + 1, n])
        x3 = compiled_dec(u3)
        assert x3.shape == (bs + 1, k)

    def test_hard_vs_soft_output(self, device):
        """Test hard vs soft output modes."""
        bs = 10
        n = 120

        source = GaussianPriorSource(device=device)
        llr = source([bs, n], no=0.5)

        dec_hard = BCJRDecoder(rate=1/2, constraint_length=4, hard_out=True,
                               device=device)
        dec_soft = BCJRDecoder(rate=1/2, constraint_length=4, hard_out=False,
                               device=device)

        u_hard = dec_hard(llr)
        u_soft = dec_soft(llr)

        # Hard output should be binary
        assert torch.all((u_hard == 0) | (u_hard == 1))

        # Soft output should be LLRs (not binary)
        assert not torch.all((u_soft == 0) | (u_soft == 1))

        # Hard decision of soft should match hard output
        u_soft_hard = (u_soft > 0).to(u_hard.dtype)
        assert torch.equal(u_hard, u_soft_hard)

    def test_docstring_example(self):
        """Verify the docstring example works correctly."""
        decoder = BCJRDecoder(rate=0.5, constraint_length=5)
        llr = torch.randn(10, 200)
        u_hat = decoder(llr)
        assert u_hat.shape == torch.Size([10, 100])

    def test_encoder_input(self, device):
        """Test that decoder can be initialized from encoder."""
        enc = ConvEncoder(rate=0.5, constraint_length=5, rsc=True,
                          terminate=True, device=device)
        dec = BCJRDecoder(encoder=enc, device=device)

        assert dec.gen_poly == enc.gen_poly
        assert dec.terminate == enc.terminate

    @pytest.mark.parametrize("n", [64, 128, 256])
    def test_dynamic_shapes(self, device, n):
        """Test for different codeword lengths."""
        enc = ConvEncoder(
            gen_poly=('1101', '1011'), rate=1/2, terminate=False,
            device=device
        )
        dec = BCJRDecoder(encoder=enc, device=device)

        llr_ch = torch.zeros(1, n, device=device)
        u_hat = dec(llr_ch)

        expected_k = n // 2
        assert u_hat.shape == (1, expected_k)

