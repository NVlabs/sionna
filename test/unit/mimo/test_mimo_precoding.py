#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Tests for sionna.phy.mimo.precoding module."""

import pytest
import numpy as np
import torch

from sionna.phy.mimo.precoding import (
    rzf_precoding_matrix,
    cbf_precoding_matrix,
    rzf_precoder,
    grid_of_beams_dft_ula,
    grid_of_beams_dft,
    flatten_precoding_mat,
    normalize_precoding_power,
)
from sionna.phy.utils import complex_normal


class TestRZFPrecodingMatrix:
    """Tests for rzf_precoding_matrix function."""

    def test_rzf_precoding_matrix_shape(self, device, precision):
        """Test that RZF precoding matrix has correct shape."""
        cdtype = torch.complex64 if precision == "single" else torch.complex128

        k = 4  # num_users
        m = 8  # num_antennas

        h = complex_normal((k, m), precision=precision, device=device)
        g = rzf_precoding_matrix(h, precision=precision)

        assert g.shape == (m, k)

    def test_rzf_precoding_matrix_batched(self, device, precision):
        """Test RZF precoding matrix with batch dimensions."""
        cdtype = torch.complex64 if precision == "single" else torch.complex128

        batch_size = 10
        k = 4
        m = 8

        h = complex_normal((batch_size, k, m), precision=precision, device=device)
        g = rzf_precoding_matrix(h, precision=precision)

        assert g.shape == (batch_size, m, k)

    def test_rzf_precoding_matrix_unit_norm_columns(self, device, precision):
        """Test that RZF precoding matrix has unit-norm columns."""
        cdtype = torch.complex64 if precision == "single" else torch.complex128
        atol = 1e-5 if precision == "single" else 1e-10

        k = 4
        m = 8

        h = complex_normal((k, m), precision=precision, device=device)
        g = rzf_precoding_matrix(h, precision=precision)

        # Each column should have unit norm
        column_norms = (g.abs() ** 2).sum(dim=-2).sqrt()
        assert torch.allclose(column_norms, torch.ones_like(column_norms), atol=atol)

    def test_rzf_precoder_diagonalizes_channel(self, device, precision):
        """Test that RZF precoder diagonalizes the channel without regularization."""
        if precision == "single":
            pytest.skip("Requires double precision for numerical stability")

        cdtype = torch.complex128
        atol = 1e-6

        k = 4
        m = 8

        h = complex_normal((k, m), precision=precision, device=device)
        g = rzf_precoding_matrix(h, alpha=0.0, precision=precision)

        # H @ G should be diagonal (with unit diagonal for ZF)
        hg = h @ g
        # Extract off-diagonal elements
        mask = ~torch.eye(k, dtype=torch.bool, device=device)
        off_diag = hg[mask]

        # Off-diagonal elements should be near zero
        assert torch.allclose(off_diag, torch.zeros_like(off_diag), atol=atol)


class TestCBFPrecodingMatrix:
    """Tests for cbf_precoding_matrix function."""

    def test_cbf_precoding_matrix_shape(self, device, precision):
        """Test that CBF precoding matrix has correct shape."""
        cdtype = torch.complex64 if precision == "single" else torch.complex128

        k = 4
        m = 8

        h = complex_normal((k, m), precision=precision, device=device)
        g = cbf_precoding_matrix(h, precision=precision)

        assert g.shape == (m, k)

    def test_cbf_precoding_matrix_unit_norm_columns(self, device, precision):
        """Test that CBF precoding matrix has unit-norm columns."""
        cdtype = torch.complex64 if precision == "single" else torch.complex128
        atol = 1e-5 if precision == "single" else 1e-10

        k = 4
        m = 8

        h = complex_normal((k, m), precision=precision, device=device)
        g = cbf_precoding_matrix(h, precision=precision)

        # Each column should have unit norm
        column_norms = (g.abs() ** 2).sum(dim=-2).sqrt()
        assert torch.allclose(column_norms, torch.ones_like(column_norms), atol=atol)


class TestRZFPrecoder:
    """Tests for rzf_precoder function."""

    def test_rzf_precoder_shape(self, device, precision):
        """Test that RZF precoder produces correct output shape."""
        cdtype = torch.complex64 if precision == "single" else torch.complex128

        k = 4
        m = 8

        x = complex_normal((k,), precision=precision, device=device)
        h = complex_normal((k, m), precision=precision, device=device)

        x_precoded = rzf_precoder(x, h, precision=precision)

        assert x_precoded.shape == (m,)

    def test_rzf_precoder_batched(self, device, precision):
        """Test RZF precoder with batch dimensions."""
        cdtype = torch.complex64 if precision == "single" else torch.complex128

        batch_size = 10
        k = 4
        m = 8

        x = complex_normal((batch_size, k), precision=precision, device=device)
        h = complex_normal((batch_size, k, m), precision=precision, device=device)

        x_precoded = rzf_precoder(x, h, precision=precision)

        assert x_precoded.shape == (batch_size, m)

    def test_rzf_precoder_return_matrix(self, device, precision):
        """Test RZF precoder with return_precoding_matrix=True."""
        cdtype = torch.complex64 if precision == "single" else torch.complex128

        k = 4
        m = 8

        x = complex_normal((k,), precision=precision, device=device)
        h = complex_normal((k, m), precision=precision, device=device)

        x_precoded, g = rzf_precoder(x, h, return_precoding_matrix=True, precision=precision)

        assert x_precoded.shape == (m,)
        assert g.shape == (m, k)


class TestGridOfBeamsDFTULA:
    """Tests for grid_of_beams_dft_ula function."""

    def test_gob_ula_shape(self, device, precision):
        """Test that GoB ULA has correct shape."""
        num_ant = 8
        oversmpl = 2

        gob = grid_of_beams_dft_ula(
            num_ant, oversmpl, precision=precision, device=device
        )

        assert gob.shape == (num_ant * oversmpl, num_ant)
        assert gob.device == torch.device(device)

    def test_gob_ula_orthogonality(self, device, precision):
        """Test that beams in the DFT GoB are orthogonal."""
        atol = 1e-5 if precision == "single" else 1e-10

        num_ant = 8
        gob = grid_of_beams_dft_ula(num_ant, precision=precision)

        # gob @ gob^H should be identity
        prod = gob @ gob.mH

        expected = torch.eye(num_ant, dtype=prod.dtype, device=prod.device)
        assert torch.allclose(prod, expected, atol=atol)

    def test_gob_ula_unit_power_beams(self, device, precision):
        """Test that each beam has unit power."""
        atol = 1e-5 if precision == "single" else 1e-10

        num_ant = 8
        oversmpl = 2

        gob = grid_of_beams_dft_ula(num_ant, oversmpl, precision=precision)

        # Each row should have unit power
        row_power = (gob.abs() ** 2).sum(dim=-1)
        expected = torch.ones(num_ant * oversmpl, device=row_power.device, dtype=row_power.dtype)
        assert torch.allclose(row_power, expected, atol=atol)


class TestGridOfBeamsDFT:
    """Tests for grid_of_beams_dft function."""

    def test_gob_shape(self, device, precision):
        """Test that GoB for URA has correct shape."""
        num_ant_v = 4
        num_ant_h = 8

        gob = grid_of_beams_dft(
            num_ant_v, num_ant_h, precision=precision, device=device
        )

        assert gob.shape == (num_ant_v, num_ant_h, num_ant_v * num_ant_h)
        assert gob.device == torch.device(device)

    def test_gob_with_oversampling(self, device, precision):
        """Test GoB with oversampling."""
        num_ant_v = 2
        num_ant_h = 4
        oversmpl_v = 2
        oversmpl_h = 2

        gob = grid_of_beams_dft(num_ant_v, num_ant_h, oversmpl_v, oversmpl_h, precision=precision)

        assert gob.shape == (
            num_ant_v * oversmpl_v,
            num_ant_h * oversmpl_h,
            num_ant_v * num_ant_h,
        )

    def test_gob_orthogonality(self, device, precision):
        """Test that beams in the DFT GoB are orthogonal."""
        atol = 1e-3 if precision == "single" else 1e-6

        num_rows = 2
        num_cols = 6

        gob = grid_of_beams_dft(num_rows, num_cols, precision=precision)

        # Flatten the first two dimensions
        gob1 = gob.reshape(num_rows * num_cols, num_rows * num_cols)

        # Product should be close to identity
        prod = (gob1 @ gob1.mH).abs()
        expected = torch.eye(num_rows * num_cols, device=prod.device, dtype=prod.dtype)

        assert torch.allclose(prod, expected, atol=atol)


class TestFlattenPrecodingMat:
    """Tests for flatten_precoding_mat function."""

    def test_flatten_by_column(self, device, precision):
        """Test flattening by column."""
        cdtype = torch.complex64 if precision == "single" else torch.complex128

        mat = torch.arange(6, dtype=cdtype.to_real(), device=device).reshape(2, 3)
        mat = torch.complex(mat, torch.zeros_like(mat))

        vec = flatten_precoding_mat(mat, by_column=True)

        # By column: [0, 3, 1, 4, 2, 5]
        assert vec.shape == (6,)
        expected = torch.tensor([0, 3, 1, 4, 2, 5], dtype=cdtype, device=device)
        assert torch.allclose(vec, expected)

    def test_flatten_by_row(self, device, precision):
        """Test flattening by row."""
        cdtype = torch.complex64 if precision == "single" else torch.complex128

        mat = torch.arange(6, dtype=cdtype.to_real(), device=device).reshape(2, 3)
        mat = torch.complex(mat, torch.zeros_like(mat))

        vec = flatten_precoding_mat(mat, by_column=False)

        # By row: [0, 1, 2, 3, 4, 5]
        assert vec.shape == (6,)
        expected = torch.tensor([0, 1, 2, 3, 4, 5], dtype=cdtype, device=device)
        assert torch.allclose(vec, expected)

    def test_flatten_batched(self, device, precision):
        """Test flattening with batch dimensions."""
        cdtype = torch.complex64 if precision == "single" else torch.complex128

        batch_size = 5
        mat = complex_normal((batch_size, 4, 8), precision=precision, device=device)

        vec = flatten_precoding_mat(mat)

        assert vec.shape == (batch_size, 32)


class TestNormalizePrecodingPower:
    """Tests for normalize_precoding_power function."""

    def test_normalize_to_unit_power(self, device, precision):
        """Test normalization to unit power."""
        cdtype = torch.complex64 if precision == "single" else torch.complex128
        atol = 1e-5 if precision == "single" else 1e-10

        vec = complex_normal((4, 8), precision=precision, device=device) * 5  # Non-unit power

        vec_norm = normalize_precoding_power(vec, precision=precision)

        # Each row should have unit power
        row_power = (vec_norm.abs() ** 2).sum(dim=-1)
        expected = torch.ones(4, device=device, dtype=row_power.dtype)
        assert torch.allclose(row_power, expected, atol=atol)

    def test_normalize_to_custom_power(self, device, precision):
        """Test normalization to custom power levels."""
        cdtype = torch.complex64 if precision == "single" else torch.complex128
        atol = 1e-5 if precision == "single" else 1e-10

        vec = complex_normal((3, 8), precision=precision, device=device)
        tx_power_list = [1.0, 2.0, 0.5]

        vec_norm = normalize_precoding_power(vec, tx_power_list, precision=precision)

        # Each row should have specified power
        row_power = (vec_norm.abs() ** 2).sum(dim=-1)
        expected = torch.tensor(
            tx_power_list, device=device, dtype=row_power.dtype
        )
        assert torch.allclose(row_power, expected, atol=atol)

    def test_negative_custom_power_raises(self, device, precision):
        """Transmit powers must be nonnegative."""
        vec = complex_normal((2, 8), precision=precision, device=device)

        with pytest.raises(ValueError, match="nonnegative"):
            normalize_precoding_power(
                vec, [1.0, -0.5], precision=precision
            )

    def test_zero_norm_vector_raises(self, device, precision):
        """A zero-norm precoding vector has no normalized representation."""
        vec = complex_normal((2, 8), precision=precision, device=device)
        vec[1] = 0

        with pytest.raises(ValueError, match="zero norm"):
            normalize_precoding_power(vec, precision=precision)

    @pytest.mark.gpu
    def test_normalization_compiles_as_fullgraph(self, device):
        """Zero-norm validation must not split valid compiled calls."""
        if not device.startswith("cuda"):
            pytest.skip("This compile regression requires a CUDA device")

        normalize = torch.compile(normalize_precoding_power, fullgraph=True)
        vec = complex_normal((2, 8), device=device)
        actual = normalize(vec)
        expected = normalize_precoding_power(vec)
        torch.testing.assert_close(actual, expected)

    def test_normalize_1d_input(self, device, precision):
        """Test normalization with 1D input."""
        cdtype = torch.complex64 if precision == "single" else torch.complex128
        atol = 1e-5 if precision == "single" else 1e-10

        vec = complex_normal((8,), precision=precision, device=device) * 3

        vec_norm = normalize_precoding_power(vec, precision=precision)

        # Should have unit power
        power = (vec_norm.abs() ** 2).sum()
        assert torch.allclose(power, torch.ones_like(power), atol=atol)


class TestPrecodingCompilation:
    """Tests for torch.compile compatibility."""

    def test_rzf_precoder_compiles(self, device, precision):
        """Test that RZF precoder can be compiled."""
        cdtype = torch.complex64 if precision == "single" else torch.complex128

        k = 4
        m = 8

        x = complex_normal((k,), precision=precision, device=device)
        h = complex_normal((k, m), precision=precision, device=device)

        compiled_fn = torch.compile(rzf_precoder)
        x_compiled = compiled_fn(x, h, precision=precision)
        x_orig = rzf_precoder(x, h, precision=precision)

        assert torch.allclose(x_compiled, x_orig, atol=1e-5)

