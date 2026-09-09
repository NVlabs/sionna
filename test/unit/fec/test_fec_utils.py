#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Unit tests for sionna.phy.fec.utils module."""

import os
import numpy as np
import pytest
import torch

from sionna.phy import config
from sionna.phy.fec.utils import (
    GaussianPriorSource,
    llr2mi,
    j_fun,
    j_fun_inv,
    load_parity_check_examples,
    bin2int,
    int2bin,
    alist2mat,
    load_alist,
    make_systematic,
    gm2pcm,
    pcm2gm,
    verify_gm_pcm,
    generate_reg_ldpc,
    int_mod_2,
)

# Get path to test codes directory
current_dir = os.path.dirname(os.path.abspath(__file__))
test_dir = os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir))


# =============================================================================
# Tests for j_fun and j_fun_inv
# =============================================================================


class TestJFun:
    """Tests for j_fun and j_fun_inv functions."""

    def test_j_fun_inv_roundtrip(self):
        """Test that J(inv_j(x))==x for PyTorch implementation."""
        x = np.arange(0.01, 20, 0.1)
        x_t = torch.tensor(x, dtype=torch.float32)
        y = j_fun(x_t)
        z = j_fun_inv(y)
        assert np.allclose(x, z.numpy(), rtol=0.001)

    def test_j_fun_output_range(self):
        """Verify j_fun outputs are in valid mutual information range [0, 1]."""
        mu = torch.linspace(0.01, 100, 100)
        mi = j_fun(mu)
        assert torch.all(mi >= 0)
        assert torch.all(mi <= 1)

    def test_j_fun_inv_output_clipped(self):
        """Verify j_fun_inv output is clipped to maximum of 20."""
        mi = torch.tensor([0.9999, 1.0])
        mu = j_fun_inv(mi)
        assert torch.all(mu <= 20)

    def test_j_fun_numerical_stability(self):
        """Test j_fun handles extreme input values gracefully."""
        mu_extreme = torch.tensor([1e-15, 1e10])
        mi = j_fun(mu_extreme)
        assert not torch.any(torch.isnan(mi))
        assert not torch.any(torch.isinf(mi))

    def test_j_fun_inv_numerical_stability(self):
        """Test j_fun_inv handles extreme input values gracefully."""
        mi_extreme = torch.tensor([1e-15, 1.0 - 1e-10])
        mu = j_fun_inv(mi_extreme)
        assert not torch.any(torch.isnan(mu))
        assert not torch.any(torch.isinf(mu))

    def test_j_fun_docstring_example(self):
        """Verify the docstring example works correctly."""
        mu = torch.tensor([0.1, 1.0, 5.0])
        mi = j_fun(mu)
        assert mi.shape == torch.Size([3])
        assert torch.all(mi >= 0) and torch.all(mi <= 1)

    def test_j_fun_inv_docstring_example(self):
        """Verify the docstring example works correctly."""
        mi = torch.tensor([0.1, 0.5, 0.9])
        mu = j_fun_inv(mi)
        assert mu.shape == torch.Size([3])
        assert torch.all(mu >= 0)


# =============================================================================
# Tests for GaussianPriorSource
# =============================================================================


class TestGaussianPriorSource:
    """Tests for GaussianPriorSource class."""

    def test_mutual_information(self, device):
        """Test that Gaussian priors have the correct mutual information.

        Indirectly, also validates llr2mi function.
        """
        num_samples = [100000]
        s = GaussianPriorSource(device=device)
        mi = np.arange(0.01, 0.99, 0.01)
        ia_hat = np.zeros_like(mi)
        for i, mii in enumerate(mi):
            x = s(num_samples, mi=mii)
            ia_hat[i] = llr2mi(x).cpu().numpy()
        # This is a Monte Carlo sim and Gaussian approx; tolerance is high
        assert np.allclose(mi, ia_hat, atol=1e-2)

    def test_sigma_ch(self, device):
        """Test that Gaussian priors have correct sigma_ch.

        The standard_dev of the generated LLRs must be:
        sigma_llr^2 = 4 / sigma_ch^2
        and
        mu_llr = sigma_llr^2 / 2
        """
        num_samples = [100000]
        s = GaussianPriorSource(device=device)
        sigma_ch = np.arange(0.3, 5, 0.1)

        sigma_target = np.sqrt(4 * sigma_ch ** (-2))
        mu_target = sigma_target ** (2) / 2

        sigma_hat = np.zeros_like(sigma_ch)
        mu_hat = np.zeros_like(sigma_ch)

        for i, sigma in enumerate(sigma_ch):
            x = s(num_samples, no=sigma**2)
            x_centered = x - x.mean()
            sigma_hat[i] = torch.sqrt(torch.mean(x_centered**2)).cpu().numpy()
            mu_hat[i] = x.mean().cpu().numpy()

        # This is a Monte Carlo sim and approximated; tolerance is high
        assert np.allclose(sigma_target, sigma_hat, atol=1e-1)
        # -1.* due to logits vs llrs
        assert np.allclose(mu_target, -1.0 * mu_hat, atol=1e-1)

    @pytest.mark.parametrize("shape", [[100], [10, 20], [5, 10, 15]])
    def test_output_shape(self, device, shape):
        """Verify output shape matches requested shape."""
        s = GaussianPriorSource(device=device)
        x = s(shape, no=1.0)
        assert list(x.shape) == shape

    def test_requires_no_or_mi(self, device):
        """Verify ValueError when neither no nor mi is provided."""
        s = GaussianPriorSource(device=device)
        with pytest.raises(ValueError, match="Either no or mi must be provided"):
            s([100])

    def test_docstring_example(self):
        """Verify the docstring example works correctly."""
        source = GaussianPriorSource()
        llrs = source([1000], no=1.0)
        assert llrs.shape == torch.Size([1000])


# =============================================================================
# Tests for llr2mi
# =============================================================================


class TestLlr2mi:
    """Tests for llr2mi function."""

    def test_dtype_validation(self):
        """Verify TypeError for non-float dtype."""
        llr = torch.tensor([1, 0, 1], dtype=torch.int32)
        with pytest.raises(TypeError, match="real-valued float"):
            llr2mi(llr)

    def test_reduce_dims_true(self, device):
        """Verify reduce_dims=True returns scalar."""
        llr = torch.randn(100, 50, device=device)
        mi = llr2mi(llr, reduce_dims=True)
        assert mi.dim() == 0

    def test_reduce_dims_false(self, device):
        """Verify reduce_dims=False averages only last dimension."""
        llr = torch.randn(10, 20, 30, device=device)
        mi = llr2mi(llr, reduce_dims=False)
        assert mi.shape == torch.Size([10, 20])

    def test_with_sign_sequence(self, device):
        """Verify llr2mi works with sign sequence adjustment."""
        # Generate proper LLRs from GaussianPriorSource
        source = GaussianPriorSource(device=device)
        llr = source([1000], no=1.0)
        s = torch.ones_like(llr)
        # With s=1 (all-zero codeword), llr2mi should work properly
        mi = llr2mi(llr, s=s)
        # MI should be in valid range when using proper LLR distribution
        assert mi.item() >= -0.1  # Allow small numerical tolerance

    def test_docstring_example(self, device):
        """Verify llr2mi computes mutual information from proper Gaussian LLRs."""
        # Use GaussianPriorSource to generate proper LLRs
        source = GaussianPriorSource(device=device)
        llr = source([10000], no=1.0)
        mi = llr2mi(llr)
        # MI should be positive for properly generated LLRs
        assert mi.item() >= 0  # MI is non-negative for valid input


# =============================================================================
# Tests for bin2int
# =============================================================================


class TestBin2Int:
    """Tests for bin2int function with both list and tensor inputs."""

    @pytest.mark.parametrize("bits, expected", [
        ([1, 0, 1], 5),
        ([1], 1),
        ([0], 0),
        ([1, 1, 1, 1], 15),
        ([0, 1, 0, 1, 1, 1, 0], 46),
    ])
    def test_list_predefined_cases(self, bits, expected):
        """Test bin2int with Python list inputs."""
        x = bin2int(bits)
        assert x == expected

    def test_list_empty_returns_none(self):
        """Verify empty list returns None."""
        assert bin2int([]) is None

    @pytest.mark.parametrize("bits, expected", [
        ([1, 0, 1], 5),
        ([1], 1),
        ([0], 0),
        ([1, 1, 1, 1], 15),
        ([0, 1, 0, 1, 1, 1, 0], 46),
    ])
    def test_tensor_predefined_cases(self, device, bits, expected):
        """Test bin2int with tensor inputs."""
        arr = torch.tensor(bits, device=device)
        x = bin2int(arr).item()
        assert x == expected

    def test_tensor_batched_input(self, device):
        """Verify bin2int works with batched tensor input."""
        arr = torch.tensor([[1, 0, 1], [0, 1, 1]], device=device)
        result = bin2int(arr)
        expected = torch.tensor([5, 3], device=device)
        assert torch.equal(result, expected)

    @pytest.mark.parametrize("bits", [
        [1, 0, 1], [0], [1], [1, 1, 1, 1], [0, 1, 0, 1, 1, 1, 0],
    ])
    def test_list_and_tensor_agree(self, bits):
        """Verify list and tensor paths produce the same result."""
        list_result = bin2int(bits)
        tensor_result = bin2int(torch.tensor(bits)).item()
        assert list_result == tensor_result

    def test_docstring_example_list(self):
        """Verify the list docstring example works correctly."""
        result = bin2int([1, 0, 1])
        assert result == 5

    def test_docstring_example_tensor(self):
        """Verify the tensor docstring example works correctly."""
        result = bin2int(torch.tensor([[1, 0, 1], [0, 1, 1]]))
        expected = torch.tensor([5, 3])
        assert torch.equal(result, expected)


# =============================================================================
# Tests for int2bin
# =============================================================================


class TestInt2Bin:
    """Tests for int2bin function with both int and tensor inputs."""

    @pytest.mark.parametrize("num, length, expected", [
        (5, 3, [1, 0, 1]),
        (1, 1, [1]),
        (1, 2, [0, 1]),
        (15, 4, [1, 1, 1, 1]),
        (46, 7, [0, 1, 0, 1, 1, 1, 0]),
    ])
    def test_list_predefined_cases(self, num, length, expected):
        """Test int2bin with Python int inputs."""
        x = int2bin(num, length)
        assert x == expected

    def test_list_truncation(self):
        """Verify truncation when length is smaller than needed (int input)."""
        result = int2bin(12, 3)
        assert result == [1, 0, 0]

    def test_list_zero_length(self):
        """Verify zero length returns empty list."""
        result = int2bin(5, 0)
        assert result == []

    def test_list_negative_input_raises(self):
        """Verify negative int input raises ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            int2bin(-1, 4)

    @pytest.mark.parametrize("num, length, expected", [
        (5, 3, [1, 0, 1]),
        (1, 1, [1]),
        (1, 2, [0, 1]),
        (15, 4, [1, 1, 1, 1]),
        (46, 7, [0, 1, 0, 1, 1, 1, 0]),
        (13, 3, [1, 0, 1]),
    ])
    def test_tensor_predefined_cases(self, device, num, length, expected):
        """Test int2bin with tensor inputs."""
        ints = torch.tensor(num, device=device)
        x = int2bin(ints, length).cpu().numpy()
        assert np.array_equal(x, expected)

    def test_tensor_zero_length(self, device):
        """Verify zero length returns empty tensor."""
        ints = torch.tensor(6, device=device)
        result = int2bin(ints, 0)
        assert result.shape[-1] == 0

    def test_tensor_batched_input(self, device):
        """Verify int2bin works with batched tensor input."""
        ints = torch.tensor([6, 12], device=device)
        result = int2bin(ints, 4)
        expected = torch.tensor([[0, 1, 1, 0], [1, 1, 0, 0]], device=device)
        assert torch.equal(result, expected)

    @pytest.mark.parametrize("num, length", [
        (5, 3),
        (1, 1),
        (1, 2),
        (15, 4),
        (46, 7),
    ])
    def test_list_and_tensor_agree(self, num, length):
        """Verify int and tensor paths produce the same result."""
        list_result = int2bin(num, length)
        tensor_result = int2bin(torch.tensor(num), length).tolist()
        assert list_result == tensor_result

    def test_docstring_example_int(self):
        """Verify the int docstring example works correctly."""
        result = int2bin(5, 4)
        assert result == [0, 1, 0, 1]

    def test_docstring_example_tensor(self):
        """Verify the tensor docstring example works correctly."""
        result = int2bin(torch.tensor([5, 12]), 4)
        expected = torch.tensor([[0, 1, 0, 1], [1, 1, 0, 0]])
        assert torch.equal(result, expected)


# =============================================================================
# Tests for alist2mat and load_alist
# =============================================================================


class TestAlist:
    """Tests for alist2mat and load_alist functions."""

    def test_explicit_example(self):
        """Test alist2mat with explicit (7,4) Hamming code example."""
        # (7,4) Hamming code
        alist = [
            [7, 3],
            [3, 4],
            [1, 1, 1, 2, 2, 2, 3],
            [4, 4, 4],
            [1, 0, 0],
            [2, 0, 0],
            [3, 0, 0],
            [1, 2, 0],
            [1, 3, 0],
            [2, 3, 0],
            [1, 2, 3],
            [1, 4, 5, 7],
            [2, 4, 6, 7],
            [3, 5, 6, 7],
        ]

        pcm, k, n, r = alist2mat(alist, verbose=False)

        # test for valid code parameters
        assert k == 4
        assert n == 7
        assert r == 4 / 7
        assert len(pcm) == 3
        assert len(pcm[0]) == 7

        pcm_true = [
            [1, 0, 0, 1, 1, 0, 1],
            [0, 1, 0, 1, 0, 1, 1],
            [0, 0, 1, 0, 1, 1, 1],
        ]
        assert np.array_equal(pcm_true, pcm)

    def test_load_alist_file(self):
        """Test loading alist from file."""
        path = os.path.join(test_dir, "codes", "ldpc", "wimax_576_0.5.alist")

        # Skip if file doesn't exist
        if not os.path.exists(path):
            pytest.skip(f"Test file not found: {path}")

        # load file
        alist = load_alist(path)

        # convert to full pcm
        pcm, k, n, r = alist2mat(alist, verbose=False)

        # check parameters for consistency
        assert k == 288
        assert n == 576
        assert r == 0.5
        assert len(pcm) == n - k
        assert len(pcm[0]) == n


# =============================================================================
# Tests for verify_gm_pcm
# =============================================================================


class TestVerifyGmPcm:
    """Tests for verify_gm_pcm function."""

    def test_invalid_pcm_shape(self):
        """Test that verify_gm_pcm identifies invalid shapes."""
        n = 20
        k = 12
        gm = np.zeros((n, k))
        pcm = np.zeros((n, k))
        with pytest.raises(ValueError):
            verify_gm_pcm(gm, pcm)

    def test_invalid_gm_shape(self):
        """Test that verify_gm_pcm identifies invalid gm shapes."""
        n = 20
        k = 12
        gm = np.zeros((n, n - k))
        pcm = np.zeros((n, k))
        with pytest.raises(ValueError):
            verify_gm_pcm(gm, pcm)

    def test_correct_pair_passes(self):
        """Verify correct gm/pcm pair passes verification."""
        pcm_id = 0
        pcm, k, n, _ = load_parity_check_examples(pcm_id=pcm_id)
        gm = pcm2gm(pcm)
        assert verify_gm_pcm(gm, pcm)

    def test_incorrect_pair_fails(self):
        """Verify incorrect pair does not pass verification."""
        pcm_id = 3  # use longer matrix (as it requires column swaps)
        pcm, k, n, _ = load_parity_check_examples(pcm_id=pcm_id)
        gm = pcm2gm(pcm)
        assert verify_gm_pcm(gm, pcm)
        gm_sys, _ = make_systematic(gm)
        assert not verify_gm_pcm(gm_sys, pcm)

    def test_nonbinary_input_raises(self):
        """Test nonbinary input raises assertion."""
        pcm_id = 0
        pcm, k, n, _ = load_parity_check_examples(pcm_id=pcm_id)
        gm = pcm2gm(pcm)
        gm_nonbin = np.copy(gm)
        pcm_nonbin = np.copy(pcm)
        gm_nonbin[0, 0] = 2  # make elements non-binary
        pcm_nonbin[0, 0] = 2  # make elements non-binary
        with pytest.raises(ValueError):
            verify_gm_pcm(gm_nonbin, pcm)
        with pytest.raises(ValueError):
            verify_gm_pcm(gm, pcm_nonbin)


# =============================================================================
# Tests for pcm2gm
# =============================================================================


class TestPcm2Gm:
    """Tests for pcm2gm function."""

    @pytest.mark.parametrize("pcm_id", [0, 1, 2, 3, 4])
    def test_consistency(self, pcm_id):
        """Test pcm2gm function for consistency."""
        pcm, _, _, _ = load_parity_check_examples(pcm_id=pcm_id)
        gm = pcm2gm(pcm)
        assert verify_gm_pcm(gm, pcm)

    def test_manual_test_case(self):
        """Additional manual test case (see PR #236)."""
        pcm = np.array(
            [
                [1, 0, 0, 0, 0, 1, 1],
                [0, 1, 0, 0, 1, 0, 1],
                [0, 0, 1, 0, 1, 0, 0],
                [0, 0, 0, 1, 0, 1, 0],
            ]
        )
        gm = pcm2gm(pcm, verify_results=False)
        assert verify_gm_pcm(gm, pcm)


# =============================================================================
# Tests for gm2pcm
# =============================================================================


class TestGm2Pcm:
    """Tests for gm2pcm function."""

    @pytest.mark.parametrize("pcm_id", [0, 1, 2, 3, 4])
    def test_consistency(self, pcm_id):
        """Test gm2pcm function for consistency."""
        gm, _, _, _ = load_parity_check_examples(pcm_id=pcm_id)
        pcm = gm2pcm(gm)
        assert verify_gm_pcm(gm, pcm)


# =============================================================================
# Tests for load_parity_check_examples
# =============================================================================


class TestLoadParityCheck:
    """Tests for load_parity_check_examples function."""

    @pytest.mark.parametrize("pcm_id", [0, 1, 2, 3, 4])
    def test_code_parameters(self, pcm_id):
        """Test that code parameters are correct."""
        pcm, k, n, r = load_parity_check_examples(pcm_id, verbose=False)

        n_pcm = pcm.shape[1]
        k_pcm = n_pcm - pcm.shape[0]
        assert k == k_pcm
        assert n == n_pcm
        assert r == k_pcm / n_pcm

        assert ((pcm == 0) | (pcm == 1)).all()

    def test_docstring_example(self):
        """Verify the docstring example works correctly."""
        pcm, k, n, coderate = load_parity_check_examples(0)
        assert n == 7
        assert k == 4
        assert np.isclose(coderate, 4 / 7)

    def test_codes_load_without_pickle(self):
        """Built-in example codes must load without pickle."""
        from importlib_resources import files, as_file
        from sionna.phy.fec.ldpc import codes

        source = files(codes).joinpath("example_codes.npz")
        with as_file(source) as path:
            with np.load(path, allow_pickle=False) as pcms:
                assert set(pcms.files) == {f"pcm_{i}" for i in range(5)}


# =============================================================================
# Tests for make_systematic
# =============================================================================


class TestMakeSystematic:
    """Tests for make_systematic function."""

    @pytest.mark.filterwarnings("ignore: All-zero column")
    @pytest.mark.parametrize("pcm_id", [0, 1, 2, 3, 4])
    @pytest.mark.parametrize("is_pcm", [False, True])
    def test_shapes_and_identity(self, pcm_id, is_pcm):
        """Test that shapes do not change and that identity matrix is found."""
        pcm, k, n, _ = load_parity_check_examples(pcm_id=pcm_id)
        m = n - k
        pcm_sys, _ = make_systematic(np.array(pcm), is_pcm=is_pcm)

        assert pcm.shape[0] == pcm_sys.shape[0]
        assert pcm.shape[1] == pcm_sys.shape[1]

        if is_pcm:
            assert np.array_equal(np.eye(m), pcm_sys[:, -m:])
        else:
            assert np.array_equal(np.eye(m), pcm_sys[:, :m])

    def test_non_full_rank_raises(self):
        """Test that non full rank raises ValueError and emits warning."""
        pcm_id = 0
        pcm, _, _, _ = load_parity_check_examples(pcm_id=pcm_id)
        pcm[1, :] = pcm[0, :]  # overwrite one row (non-full rank)

        # The all-zero column warning should be raised before the ValueError
        with pytest.warns(UserWarning, match="All-zero column"):
            with pytest.raises(ValueError):
                make_systematic(pcm, is_pcm=True)

    def test_all_zero_and_all_one_inputs(self):
        """Test with all-zero and all-one inputs raises ValueError with warning."""
        k = 13
        n = 20
        for is_pcm in (False, True):
            mat = np.zeros((k, n))
            # All-zero matrix triggers warning (for is_pcm=True) before ValueError
            if is_pcm:
                with pytest.warns(UserWarning, match="All-zero column"):
                    with pytest.raises(ValueError):
                        make_systematic(mat, is_pcm=is_pcm)
            else:
                with pytest.raises(ValueError):
                    make_systematic(mat, is_pcm=is_pcm)

            mat = np.ones((k, n))
            with pytest.raises(ValueError):
                make_systematic(mat, is_pcm=is_pcm)


# =============================================================================
# Tests for generate_reg_ldpc
# =============================================================================


class TestGenerateRegLdpc:
    """Tests for generate_reg_ldpc function."""

    @pytest.mark.parametrize("v, c, n_des", [
        (3, 6, 100),
        (1, 10, 1000),
        (3, 6, 10000),
        (2, 7, 703),
    ])
    def test_ldpc_generation(self, v, c, n_des):
        """Test LDPC generator function."""
        pcm, k, n, r = generate_reg_ldpc(v, c, n_des, verbose=False)

        assert r == k / n
        assert (np.sum(pcm, axis=0) == v).all()
        assert (np.sum(pcm, axis=1) == c).all()
        assert n >= n_des
        assert pcm.shape[0] == n - k
        assert pcm.shape[1] == n


# =============================================================================
# Tests for int_mod_2
# =============================================================================


class TestIntMod2:
    """Tests for int_mod_2 function."""

    def test_int_inputs(self, device):
        """Test modulo 2 operation with integer inputs."""
        s = [10, 20, 30]
        x = torch.randint(-2**30, 2**30, s, dtype=torch.int32, device=device)

        y = int_mod_2(x)
        y_ref = torch.fmod(x.to(torch.float64).abs(), 2.0)
        assert torch.equal(y, y_ref.to(torch.int32))

    def test_float_inputs(self, device):
        """Test modulo 2 operation with float inputs."""
        s = [10, 20, 30]
        x = torch.rand(s, dtype=torch.float32, device=device) * 2000 - 1000

        y = int_mod_2(x)
        # model implicit cast
        x_ = torch.abs(torch.round(x))
        y_ref = torch.fmod(x_, 2)
        assert torch.allclose(y, y_ref)

    def test_docstring_example(self):
        """Verify the docstring example works correctly."""
        x = torch.tensor([0, 1, 2, 3, 4, 5])
        result = int_mod_2(x)
        expected = torch.tensor([0, 1, 0, 1, 0, 1])
        assert torch.equal(result, expected)

