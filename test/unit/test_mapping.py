#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Unit tests for sionna.phy.mapping module."""

import numpy as np
import pytest
import torch
from scipy.special import logsumexp, softmax

from sionna.phy import dtypes
from sionna.phy.mapping import (
    pam_gray,
    qam,
    pam,
    Constellation,
    Mapper,
    Demapper,
    SymbolDemapper,
    SymbolLogits2LLRs,
    LLRs2SymbolLogits,
    SymbolLogits2Moments,
    SymbolInds2Bits,
    QAM2PAM,
    PAM2QAM,
    BinarySource,
    SymbolSource,
    QAMSource,
    PAMSource,
)


# =============================================================================
# Reference implementations for testing
# =============================================================================


def bpsk(b):
    """Reference BPSK formula from 5G standard."""
    return 1 - 2 * b[0]


def pam4_ref(b):
    """Reference 4-PAM formula from 5G standard."""
    return (1 - 2 * b[0]) * (2 - (1 - 2 * b[1]))


def pam8_ref(b):
    """Reference 8-PAM formula from 5G standard."""
    return (1 - 2 * b[0]) * (4 - (1 - 2 * b[1]) * (2 - (1 - 2 * b[2])))


def pam16_ref(b):
    """Reference 16-PAM formula from 5G standard."""
    return (1 - 2 * b[0]) * (
        8 - (1 - 2 * b[1]) * (4 - (1 - 2 * b[2]) * (2 - (1 - 2 * b[3])))
    )


# =============================================================================
# Tests for pam_gray function
# =============================================================================


class TestPamGray:
    """Tests for pam_gray function."""

    def test_against_5g_formulas(self):
        """Verify pam_gray produces correct results compared to 5G standard formulas."""
        for n, comp in enumerate([bpsk, pam4_ref, pam8_ref, pam16_ref]):
            for i in range(0, 2 ** (n + 1)):
                b = np.array(list(np.binary_repr(i, n + 1)), dtype=np.int32)
                assert pam_gray(b) == comp(b)

    def test_docstring_example(self):
        """Verify the docstring example works correctly."""
        b = np.array([1, 0])
        result = pam_gray(b)
        assert result == -1


# =============================================================================
# Tests for pam function
# =============================================================================


class TestPam:
    """Tests for pam constellation generator function."""

    def test_against_5g_formulas(self):
        """Verify PAM constellations match 5G standard formulas."""
        for n, comp in enumerate([bpsk, pam4_ref, pam8_ref, pam16_ref]):
            num_bits_per_symbol = n + 1
            c = pam(num_bits_per_symbol, normalize=False)
            for i in range(0, 2**num_bits_per_symbol):
                b = np.array(
                    list(np.binary_repr(i, num_bits_per_symbol)), dtype=np.int32
                )
                assert np.equal(c[i], comp(b))

    def test_normalization(self):
        """Verify normalized PAM constellations have unit energy."""
        for num_bits_per_symbol in range(1, 9):
            c = pam(num_bits_per_symbol, normalize=True)
            assert np.isclose(np.mean(np.abs(c) ** 2), 1.0, atol=1e-4)

    def test_output_shape(self):
        """Verify output has correct shape."""
        for num_bits in [1, 2, 3, 4]:
            c = pam(num_bits)
            assert c.shape == (2**num_bits,)

    def test_invalid_num_bits(self):
        """Verify error is raised for invalid num_bits_per_symbol."""
        with pytest.raises(ValueError):
            pam(0)
        with pytest.raises(ValueError):
            pam(-1)

    def test_docstring_example(self):
        """Verify the docstring example works correctly."""
        constellation = pam(2)
        assert constellation.shape == (4,)


# =============================================================================
# Tests for qam function
# =============================================================================


class TestQam:
    """Tests for qam constellation generator function."""

    def test_against_5g_formulas(self):
        """Verify QAM constellations match 5G standard formulas."""
        for n, pam_ref in enumerate([bpsk, pam4_ref, pam8_ref, pam16_ref]):
            num_bits_per_symbol = 2 * (n + 1)
            c = qam(num_bits_per_symbol, normalize=False)
            for i in range(0, 2**num_bits_per_symbol):
                b = np.array(list(np.binary_repr(i, 2 * (n + 1))), dtype=np.int32)
                expected = pam_ref(b[0::2]) + 1j * pam_ref(b[1::2])
                assert np.equal(c[i], expected)

    def test_normalization(self):
        """Verify normalized QAM constellations have unit energy."""
        for num_bits_per_symbol in [2, 4, 6, 8, 10, 12, 14]:
            c = qam(num_bits_per_symbol, normalize=True)
            assert np.isclose(np.mean(np.abs(c) ** 2), 1.0, atol=1e-4)

    def test_output_shape(self):
        """Verify output has correct shape."""
        for num_bits in [2, 4, 6, 8]:
            c = qam(num_bits)
            assert c.shape == (2**num_bits,)

    def test_invalid_num_bits(self):
        """Verify error is raised for invalid num_bits_per_symbol."""
        with pytest.raises(ValueError):
            qam(0)
        with pytest.raises(ValueError):
            qam(3)  # Must be even
        with pytest.raises(ValueError):
            qam(5)  # Must be even

    def test_invalid_normalize(self):
        """Verify error is raised for a non-boolean normalize."""
        with pytest.raises(TypeError, match="normalize"):
            qam(4, normalize=1)

    def test_docstring_example(self):
        """Verify the docstring example works correctly."""
        constellation = qam(4)
        assert constellation.shape == (16,)


# =============================================================================
# Tests for Constellation class
# =============================================================================


class TestConstellation:
    """Tests for Constellation class."""

    def test_invalid_constellation_type(self, device):
        """Verify error for invalid constellation type."""
        with pytest.raises(ValueError):
            Constellation("invalid_type", 2, device=device)

    def test_invalid_num_bits_per_symbol(self, device):
        """Verify error for invalid num_bits_per_symbol."""
        with pytest.raises(ValueError):
            Constellation("custom", 0, device=device)
        with pytest.raises(ValueError):
            Constellation("qam", 0, device=device)
        with pytest.raises(ValueError):
            Constellation("qam", 3, device=device)  # Must be even for QAM

    def test_custom_requires_points(self, device):
        """Verify custom constellation requires points."""
        with pytest.raises(ValueError):
            Constellation("custom", 3, device=device)

    def test_points_only_for_custom(self, device):
        """Verify points can only be provided for custom constellations."""
        points = np.zeros([4])
        with pytest.raises(ValueError):
            Constellation("qam", 2, points=points, device=device)

    def test_wrong_points_shape(self, device):
        """Verify error for wrong points shape."""
        points = np.zeros([7])  # Should be 8 for 3 bits
        with pytest.raises(ValueError):
            Constellation("custom", 3, points=points, device=device)

    def test_output_dimensions(self, device):
        """Verify output has correct dimensions."""
        for num_bits_per_symbol in [2, 4, 6, 8]:
            c = Constellation("qam", num_bits_per_symbol, device=device)
            points = c()
            assert points.shape == torch.Size([2**num_bits_per_symbol])

    def test_output_dimensions_custom(self, device):
        """Verify custom constellation output dimensions."""
        for num_bits_per_symbol in range(1, 8):
            points_np = np.random.randn(
                2**num_bits_per_symbol
            ) + 1j * np.random.randn(2**num_bits_per_symbol)
            c = Constellation(
                "custom", num_bits_per_symbol, points=points_np, device=device
            )
            assert c().shape == torch.Size([2**num_bits_per_symbol])

    def test_normalization_qam(self, device):
        """Verify QAM constellations are normalized by default."""
        for num_bits in [2, 4, 6, 8]:
            c = Constellation("qam", num_bits, device=device)
            points = c()
            energy = (points.abs() ** 2).mean().item()
            assert np.isclose(energy, 1.0, atol=1e-5)

    def test_normalization_pam(self, device):
        """Verify PAM constellations are normalized by default."""
        for num_bits in [1, 2, 3, 4]:
            c = Constellation("pam", num_bits, device=device)
            points = c()
            energy = (points.abs() ** 2).mean().item()
            assert np.isclose(energy, 1.0, atol=1e-5)

    def test_custom_normalization_and_centering(self, device):
        """Verify custom constellation normalization and centering."""
        points_np = np.array([1 + 1j, 2 + 2j, 3 + 3j, 4 + 4j])

        # With normalize and center
        c = Constellation(
            "custom", 2, points=points_np, normalize=True, center=True, device=device
        )
        output = c()
        assert np.isclose(output.mean().abs().item(), 0.0, atol=1e-5)
        assert np.isclose((output.abs() ** 2).mean().item(), 1.0, atol=1e-5)

    def test_dtype(self, precision, device):
        """Verify output dtype matches requested precision."""
        c = Constellation("qam", 4, precision=precision, device=device)
        expected_dtype = dtypes[precision]["torch"]["cdtype"]
        assert c().dtype == expected_dtype

    def test_docstring_example(self):
        """Verify the docstring example works correctly."""
        const = Constellation("qam", 4)
        points = const()
        assert points.shape == torch.Size([16])


# =============================================================================
# Tests for Mapper class
# =============================================================================


class TestMapper:
    """Tests for Mapper class."""

    def test_output_dimensions(self, device):
        """Verify mapper output has correct dimensions."""
        num_bits_per_symbol = 4
        batch_size = 100
        num_symbols = 100
        mapper = Mapper("qam", num_bits_per_symbol, device=device)
        bits = torch.randint(
            0, 2, (batch_size, num_symbols * num_bits_per_symbol), device=device
        ).float()

        x = mapper(bits)
        assert x.shape == torch.Size([batch_size, num_symbols])

    def test_output_dimensions_multidim(self, device):
        """Verify mapper works with multi-dimensional inputs."""
        num_bits_per_symbol = 4
        mapper = Mapper("qam", num_bits_per_symbol, device=device)

        bits = torch.randint(
            0, 2, (10, 2, 3, 100 * num_bits_per_symbol), device=device
        ).float()
        x = mapper(bits)
        assert x.shape == torch.Size([10, 2, 3, 100])

    def test_mapping_correctness(self, device):
        """Verify bits are correctly mapped to constellation symbols."""
        num_bits_per_symbol = 8
        mapper = Mapper("qam", num_bits_per_symbol, device=device)

        # Create all possible bit combinations
        b = np.zeros([2**num_bits_per_symbol, num_bits_per_symbol])
        for i in range(0, 2**num_bits_per_symbol):
            b[i] = np.array(
                list(np.binary_repr(i, num_bits_per_symbol)), dtype=np.int32
            )

        bits = torch.tensor(b, dtype=torch.float32, device=device)
        x = mapper(bits)
        constellation = mapper.constellation()

        for i, s in enumerate(x):
            assert torch.allclose(s, constellation[i])

    def test_return_indices(self, device):
        """Verify mapper can return symbol indices."""
        mapper = Mapper("qam", 4, return_indices=True, device=device)
        bits = torch.randint(0, 2, (100, 3, 400), device=device).float()
        x, ind = mapper(bits)

        assert x.shape == torch.Size([100, 3, 100])
        assert ind.shape == torch.Size([100, 3, 100])
        assert ind.dtype == torch.int32

    def test_dtype(self, precision, device):
        """Verify output dtype matches requested precision."""
        mapper = Mapper("qam", 4, precision=precision, device=device)
        bits = torch.randint(0, 2, (10, 40), device=device).float()
        x = mapper(bits)
        expected_dtype = dtypes[precision]["torch"]["cdtype"]
        assert x.dtype == expected_dtype

    @pytest.mark.parametrize("shape", [(), (5,)])
    def test_invalid_input_length(self, device, shape):
        """Reject inputs without a complete final constellation label."""
        mapper = Mapper("qam", 4, device=device)
        bits = torch.zeros(shape, device=device)

        with pytest.raises(ValueError, match="last dimension"):
            mapper(bits)

    def test_input_length_check_under_compilation(self, device):
        """Keep the shape contract for valid and invalid compiled inputs."""
        torch._dynamo.reset()
        mapper = torch.compile(Mapper("qam", 4, device=device), fullgraph=True)
        bits = torch.randint(0, 2, (8, 40), device=device).float()
        assert mapper(bits).shape == (8, 10)

        # Python exceptions cannot be represented in a full graph. Verify the
        # ordinary compiled path falls back and preserves the public error.
        torch._dynamo.reset()
        checked_mapper = torch.compile(Mapper("qam", 4, device=device))
        with pytest.raises(ValueError, match="last dimension"):
            checked_mapper(torch.zeros(8, 41, device=device))

    def test_docstring_example(self):
        """Verify the docstring example works correctly."""
        mapper = Mapper("qam", 4)
        bits = torch.randint(0, 2, (10, 100))
        symbols = mapper(bits)
        assert symbols.shape == torch.Size([10, 25])


# =============================================================================
# Tests for Demapper class
# =============================================================================


class TestDemapper:
    """Tests for Demapper class."""

    def test_invalid_method(self, device):
        """Verify error for invalid demapping method."""
        with pytest.raises(ValueError):
            Demapper("invalid_method", "qam", 4, device=device)

    def test_output_dimensions(self, device):
        """Verify demapper output has correct dimensions."""
        for num_bits_per_symbol in [2, 4, 6]:
            mapper = Mapper("qam", num_bits_per_symbol, device=device)
            demapper = Demapper("app", "qam", num_bits_per_symbol, device=device)

            batch_size = 99
            dim1 = 10
            dim2 = 12
            bits = torch.randint(
                0, 2, (batch_size, dim1, dim2 * num_bits_per_symbol), device=device
            ).float()
            x = mapper(bits)
            llr = demapper(x, torch.tensor(1.0, device=device))

            assert llr.shape == torch.Size(
                [batch_size, dim1, dim2 * num_bits_per_symbol]
            )

    def test_app_demapping(self, device):
        """Verify APP demapping produces correct LLRs."""
        num_bits_per_symbol = 6
        mapper = Mapper("qam", num_bits_per_symbol, device=device)
        demapper = Demapper("app", "qam", num_bits_per_symbol, device=device)

        bits = torch.randint(
            0, 2, (10, 10 * num_bits_per_symbol), device=device
        ).float()
        x = mapper(bits)
        no = torch.rand(x.shape, device=device) * 99 + 0.01  # Random noise variance

        llr = demapper(x, no)
        points = demapper.constellation.points.cpu().numpy()
        c0 = demapper._logits2llrs._c0.cpu().numpy()
        c1 = demapper._logits2llrs._c1.cpu().numpy()

        # Verify against reference implementation
        for batch in range(x.shape[0]):
            for i, y in enumerate(x[batch].cpu().numpy()):
                dist = np.abs(y - points) ** 2
                exp = -dist / no[batch, i].cpu().numpy()

                llr_ref = logsumexp(np.take(exp, c1), axis=0) - logsumexp(
                    np.take(exp, c0), axis=0
                )
                llr_target = (
                    llr[batch, i * num_bits_per_symbol : (i + 1) * num_bits_per_symbol]
                    .cpu()
                    .numpy()
                )
                assert np.allclose(llr_ref, llr_target, atol=1e-5)

    def test_maxlog_demapping(self, device):
        """Verify maxlog demapping produces correct LLRs."""
        num_bits_per_symbol = 6
        mapper = Mapper("qam", num_bits_per_symbol, device=device)
        demapper = Demapper("maxlog", "qam", num_bits_per_symbol, device=device)

        bits = torch.randint(
            0, 2, (10, 10 * num_bits_per_symbol), device=device
        ).float()
        x = mapper(bits)
        no = torch.rand(x.shape, device=device) * 99 + 0.01

        llr = demapper(x, no)
        points = demapper.constellation.points.cpu().numpy()
        c0 = demapper._logits2llrs._c0.cpu().numpy()
        c1 = demapper._logits2llrs._c1.cpu().numpy()

        for batch in range(x.shape[0]):
            for i, y in enumerate(x[batch].cpu().numpy()):
                dist = np.abs(y - points) ** 2
                exp = -dist / no[batch, i].cpu().numpy()

                llr_ref = np.max(np.take(exp, c1), axis=0) - np.max(
                    np.take(exp, c0), axis=0
                )
                llr_target = (
                    llr[batch, i * num_bits_per_symbol : (i + 1) * num_bits_per_symbol]
                    .cpu()
                    .numpy()
                )
                assert np.allclose(llr_ref, llr_target, atol=1e-5)

    def test_broadcastable_noise_variance(self, device):
        """Verify demapper works with broadcastable noise variance."""
        num_bits_per_symbol = 4
        mapper = Mapper("qam", num_bits_per_symbol, device=device)
        demapper = Demapper("app", "qam", num_bits_per_symbol, device=device)

        bits = torch.randint(
            0, 2, (100, 10 * num_bits_per_symbol), device=device
        ).float()
        x = mapper(bits)
        no = torch.rand(1, 10, device=device) * 99 + 0.01  # Broadcastable

        llr = demapper(x, no)
        assert llr.shape == torch.Size([100, 10 * num_bits_per_symbol])

    def test_dtype(self, precision, device):
        """Verify output dtype matches requested precision."""
        demapper = Demapper("app", "qam", 4, precision=precision, device=device)
        y = torch.randn(
            10, 10, dtype=dtypes[precision]["torch"]["cdtype"], device=device
        )
        no = torch.tensor(0.1, dtype=dtypes[precision]["torch"]["dtype"], device=device)
        llr = demapper(y, no)
        assert llr.dtype == dtypes[precision]["torch"]["dtype"]


# =============================================================================
# Tests for Demapper with prior
# =============================================================================


class TestDemapperWithPrior:
    """Tests for Demapper class with prior information."""

    def test_app_demapping_with_prior(self, device):
        """Verify APP demapping with prior produces correct LLRs."""
        num_bits_per_symbol = 6
        num_points = 2**num_bits_per_symbol
        mapper = Mapper("qam", num_bits_per_symbol, device=device)
        demapper = Demapper("app", "qam", num_bits_per_symbol, device=device)

        bits = torch.randint(
            0, 2, (10, 10 * num_bits_per_symbol), device=device
        ).float()
        x = mapper(bits)
        no = torch.rand(x.shape, device=device) * 99 + 0.01
        prior = torch.randn(10, 10, num_bits_per_symbol, device=device)

        llr = demapper(x, no, prior=prior)

        # Compute reference
        points = demapper.constellation.points.cpu().numpy()
        c0 = demapper._logits2llrs._c0.cpu().numpy()
        c1 = demapper._logits2llrs._c1.cpu().numpy()

        # Binary labels
        a = np.zeros([num_points, num_bits_per_symbol])
        for i in range(num_points):
            a[i, :] = np.array(
                list(np.binary_repr(i, num_bits_per_symbol)), dtype=np.int32
            )
        a = 2 * a - 1

        prior_np = prior.cpu().numpy()
        ps_exp = a[None, None, :, :] * prior_np[:, :, None, :]
        ps_exp = ps_exp - np.log(1 + np.exp(ps_exp))
        ps_exp = np.sum(ps_exp, axis=-1)

        for batch in range(x.shape[0]):
            for i, y in enumerate(x[batch].cpu().numpy()):
                dist = np.abs(y - points) ** 2
                exp = -dist / no[batch, i].cpu().numpy()
                ps_exp_ = ps_exp[batch, i]

                llr_ref = logsumexp(
                    np.take(exp, c1) + np.take(ps_exp_, c1), axis=0
                ) - logsumexp(np.take(exp, c0) + np.take(ps_exp_, c0), axis=0)
                llr_target = (
                    llr[batch, i * num_bits_per_symbol : (i + 1) * num_bits_per_symbol]
                    .cpu()
                    .numpy()
                )
                assert np.allclose(llr_ref, llr_target, atol=1e-5)


# =============================================================================
# Tests for SymbolDemapper class
# =============================================================================


class TestSymbolDemapper:
    """Tests for SymbolDemapper class."""

    def test_output_dimensions(self, device):
        """Verify symbol demapper output has correct dimensions."""
        for num_bits_per_symbol in [2, 4, 6]:
            num_points = 2**num_bits_per_symbol
            mapper = Mapper("qam", num_bits_per_symbol, device=device)
            demapper = SymbolDemapper("qam", num_bits_per_symbol, device=device)

            batch_size = 32
            dim1 = 10
            dim2 = 12
            bits = torch.randint(
                0, 2, (batch_size, dim1, dim2 * num_bits_per_symbol), device=device
            ).float()
            x = mapper(bits)
            logits = demapper(x, torch.tensor(1.0, device=device))

            assert logits.shape == torch.Size([batch_size, dim1, dim2, num_points])

    def test_soft_output(self, device):
        """Verify symbol demapper produces correct log-probabilities."""
        num_bits_per_symbol = 6
        constellation = Constellation("qam", num_bits_per_symbol, device=device)
        mapper = Mapper(constellation=constellation, device=device)
        demapper = SymbolDemapper(constellation=constellation, device=device)

        bits = torch.randint(
            0, 2, (10, 10 * num_bits_per_symbol), device=device
        ).float()
        x = mapper(bits)
        no = torch.rand(x.shape, device=device) * 99 + 0.01

        logits = demapper(x, no)
        points = constellation().cpu().numpy()

        for batch in range(x.shape[0]):
            for i, y in enumerate(x[batch].cpu().numpy()):
                dist = np.abs(y - points) ** 2
                exp = -dist / no[batch, i].cpu().numpy()

                logits_ref = exp - np.log(np.sum(np.exp(exp)))
                logits_target = logits[batch, i].cpu().numpy()
                assert np.allclose(logits_ref, logits_target, atol=1e-5)

    def test_hard_output(self, device):
        """Verify hard output returns correct indices."""
        mapper = Mapper("qam", 4, device=device)
        demapper = SymbolDemapper("qam", 4, hard_out=True, device=device)

        bits = torch.randint(0, 2, (10, 40), device=device).float()
        x = mapper(bits)
        hard = demapper(x, torch.tensor(0.0001, device=device))

        assert hard.dtype == torch.int32

# =============================================================================
# Tests shared by Demapper and SymbolDemapper
# =============================================================================


@pytest.mark.parametrize(
    "demapper_class, args",
    [(Demapper, ("app",)), (SymbolDemapper, ())],
)
def test_zero_and_sub_tiny_noise_are_clamped(
    demapper_class, args, precision, device
):
    """Clamp both demappers without introducing device-bound module state."""
    rdtype = dtypes[precision]["torch"]["dtype"]
    constellation = Constellation(
        "qam", 6, precision=precision, device=device
    )
    demapper = demapper_class(
        *args, constellation=constellation, precision=precision, device=device
    )
    y = constellation()[:1]
    tiny = torch.finfo(rdtype).tiny
    expected = demapper(y, torch.tensor(tiny, dtype=rdtype, device=device))

    for no in (0.0, tiny / 2):
        actual = demapper(y, torch.tensor(no, dtype=rdtype, device=device))
        assert not torch.isnan(actual).any()
        assert torch.equal(actual, expected)


# =============================================================================
# Tests for SymbolLogits2LLRs class
# =============================================================================


class TestSymbolLogits2LLRs:
    """Tests for SymbolLogits2LLRs class."""

    def test_invalid_method(self):
        """Verify error for invalid method."""
        with pytest.raises(ValueError):
            SymbolLogits2LLRs("invalid", 4)

    def test_output_dimensions(self, device):
        """Verify output has correct dimensions."""
        for num_bits_per_symbol in [1, 2, 4, 6]:
            converter = SymbolLogits2LLRs("app", num_bits_per_symbol, device=device)
            batch_size = 99
            dim1 = 10
            dim2 = 12
            logits = torch.randn(
                batch_size, dim1, dim2, 2**num_bits_per_symbol, device=device
            )
            llr = converter(logits)

            assert llr.shape == torch.Size(
                [batch_size, dim1, dim2, num_bits_per_symbol]
            )

    def test_app_llr_calculation(self, device):
        """Verify APP method produces correct LLRs."""
        num_bits_per_symbol = 6
        converter_app = SymbolLogits2LLRs("app", num_bits_per_symbol, device=device)

        logits = torch.randn(10, 10, 2**num_bits_per_symbol, device=device) * 20

        llr_app = converter_app(logits)

        c0 = converter_app._c0.cpu().numpy()
        c1 = converter_app._c1.cpu().numpy()

        for b in range(logits.shape[0]):
            for i in range(logits.shape[1]):
                logits_np = logits[b, i].cpu().numpy()
                llr_ref = logsumexp(np.take(logits_np, c1), axis=0) - logsumexp(
                    np.take(logits_np, c0), axis=0
                )
                assert np.allclose(llr_app[b, i].cpu().numpy(), llr_ref, atol=1e-5)

    def test_maxlog_llr_calculation(self, device):
        """Verify maxlog method produces correct LLRs."""
        num_bits_per_symbol = 6
        converter_maxlog = SymbolLogits2LLRs(
            "maxlog", num_bits_per_symbol, device=device
        )

        logits = torch.randn(10, 10, 2**num_bits_per_symbol, device=device) * 20

        llr_maxlog = converter_maxlog(logits)

        c0 = converter_maxlog._c0.cpu().numpy()
        c1 = converter_maxlog._c1.cpu().numpy()

        for b in range(logits.shape[0]):
            for i in range(logits.shape[1]):
                logits_np = logits[b, i].cpu().numpy()
                llr_ref = np.max(np.take(logits_np, c1), axis=0) - np.max(
                    np.take(logits_np, c0), axis=0
                )
                assert np.allclose(llr_maxlog[b, i].cpu().numpy(), llr_ref, atol=1e-5)

    def test_hard_output(self, device):
        """Verify hard output produces binary decisions."""
        converter = SymbolLogits2LLRs("app", 4, hard_out=True, device=device)
        logits = torch.randn(10, 10, 16, device=device)
        hard = converter(logits)

        # Hard decisions should be 0 or 1
        assert torch.all((hard == 0) | (hard == 1))

    def test_docstring_example(self):
        """Verify the docstring example works correctly."""
        converter = SymbolLogits2LLRs("app", 4)
        logits = torch.randn(10, 25, 16)
        llr = converter(logits)
        assert llr.shape == torch.Size([10, 25, 4])


# =============================================================================
# Tests for LLRs2SymbolLogits class
# =============================================================================


class TestLLRs2SymbolLogits:
    """Tests for LLRs2SymbolLogits class."""

    def test_output_dimensions(self, device):
        """Verify output has correct dimensions."""
        for num_bits_per_symbol in [1, 2, 4, 6]:
            converter = LLRs2SymbolLogits(num_bits_per_symbol, device=device)
            batch_size = 99
            dim1 = 10
            dim2 = 12
            llrs = torch.randn(
                batch_size, dim1, dim2, num_bits_per_symbol, device=device
            )
            logits = converter(llrs)

            assert logits.shape == torch.Size(
                [batch_size, dim1, dim2, 2**num_bits_per_symbol]
            )

    def test_logits_calculation(self, device):
        """Verify logits are computed correctly."""

        def sigmoid(x):
            return 1.0 / (1.0 + np.exp(-x))

        num_bits_per_symbol = 6
        converter = LLRs2SymbolLogits(num_bits_per_symbol, device=device)
        llrs = torch.randn(10, 10, num_bits_per_symbol, device=device) * 20

        logits = converter(llrs)
        a = converter._a.cpu().numpy()

        for b in range(llrs.shape[0]):
            for i in range(llrs.shape[1]):
                logits_ref = np.sum(
                    np.log(sigmoid(a * llrs[b, i].cpu().numpy())), axis=1
                )
                assert np.allclose(logits[b, i].cpu().numpy(), logits_ref, atol=1e-5)

    def test_hard_output(self, device):
        """Verify hard output returns indices."""
        converter = LLRs2SymbolLogits(4, hard_out=True, device=device)
        llrs = torch.randn(10, 10, 4, device=device)
        hard = converter(llrs)

        assert hard.dtype == torch.int32
        assert hard.shape == torch.Size([10, 10])

    def test_docstring_example(self):
        """Verify the docstring example works correctly."""
        converter = LLRs2SymbolLogits(4)
        llr = torch.randn(10, 25, 4)
        logits = converter(llr)
        assert logits.shape == torch.Size([10, 25, 16])


# =============================================================================
# Tests for SymbolLogits2Moments class
# =============================================================================


class TestSymbolLogits2Moments:
    """Tests for SymbolLogits2Moments class."""

    def test_output_dimensions(self, device):
        """Verify output has correct dimensions."""
        for num_bits_per_symbol in [2, 4, 6]:
            converter = SymbolLogits2Moments("qam", num_bits_per_symbol, device=device)
            batch_size = 99
            dim1 = 10
            dim2 = 12
            logits = torch.randn(
                batch_size, dim1, dim2, 2**num_bits_per_symbol, device=device
            )
            mean, var = converter(logits)

            assert mean.shape == torch.Size([batch_size, dim1, dim2])
            assert var.shape == torch.Size([batch_size, dim1, dim2])

    def test_moments_calculation(self, device):
        """Verify moments are computed correctly."""
        num_bits_per_symbol = 6
        constellation = Constellation("qam", num_bits_per_symbol, device=device)
        converter = SymbolLogits2Moments(constellation=constellation, device=device)
        points = constellation().cpu().numpy()

        logits = torch.randn(10, 2**num_bits_per_symbol, device=device) * 20
        mean, var = converter(logits)

        for i in range(logits.shape[0]):
            p = softmax(logits[i].cpu().numpy())
            mean_ref = np.sum(p * points)
            var_ref = np.sum(p * np.square(np.abs(points - mean_ref)))

            assert np.allclose(mean[i].cpu().numpy(), mean_ref, atol=1e-5)
            assert np.allclose(var[i].cpu().numpy(), var_ref, atol=1e-5)

    def test_docstring_example(self):
        """Verify the docstring example works correctly."""
        converter = SymbolLogits2Moments("qam", 4)
        logits = torch.randn(10, 25, 16)
        mean, var = converter(logits)
        assert mean.shape == torch.Size([10, 25])
        assert var.shape == torch.Size([10, 25])


# =============================================================================
# Tests for SymbolInds2Bits class
# =============================================================================


class TestSymbolInds2Bits:
    """Tests for SymbolInds2Bits class."""

    def test_output_shape(self, device):
        """Verify output has correct shape."""
        converter = SymbolInds2Bits(4, device=device)
        indices = torch.tensor([0, 5, 10, 15], device=device)
        bits = converter(indices)
        assert bits.shape == torch.Size([4, 4])

    def test_conversion_correctness(self, device):
        """Verify indices are correctly converted to bits."""
        converter = SymbolInds2Bits(4, device=device)

        # Test all indices for 4-bit symbols
        for i in range(16):
            indices = torch.tensor([i], device=device)
            bits = converter(indices)
            expected = list(np.binary_repr(i, 4))
            expected = [float(b) for b in expected]
            assert bits[0].tolist() == expected

    def test_docstring_example(self):
        """Verify the docstring example works correctly."""
        converter = SymbolInds2Bits(4)
        indices = torch.tensor([0, 5, 10, 15])
        bits = converter(indices)
        assert bits.shape == torch.Size([4, 4])


# =============================================================================
# Tests shared by QAM2PAM and PAM2QAM
# =============================================================================


@pytest.mark.parametrize("converter", [QAM2PAM, PAM2QAM])
class TestSquareQamBitWidth:
    """Tests for the bit-width contract shared by QAM2PAM and PAM2QAM."""

    @pytest.mark.parametrize("num_bits_per_symbol", [0, -2, 1, 3])
    def test_invalid_qam_width(self, converter, num_bits_per_symbol):
        """Reject non-positive and odd square-QAM widths."""
        with pytest.raises(ValueError, match="positive even integer"):
            converter(num_bits_per_symbol)

    @pytest.mark.parametrize("num_bits_per_symbol", [True, 4.0, "4"])
    def test_non_integer_qam_width(self, converter, num_bits_per_symbol):
        """Reject values that are not integer bit widths."""
        with pytest.raises(TypeError, match="must be an integer"):
            converter(num_bits_per_symbol)

    @pytest.mark.parametrize("num_bits_per_symbol", [np.int32(2), np.int64(4)])
    def test_numpy_integer_qam_width(self, converter, num_bits_per_symbol):
        """Preserve support for NumPy integer widths."""
        converter(num_bits_per_symbol)


# =============================================================================
# Tests for QAM2PAM class
# =============================================================================


class TestQAM2PAM:
    """Tests for QAM2PAM class."""

    def test_conversion(self, device):
        """Verify QAM to PAM index conversion is correct."""
        converter = QAM2PAM(4, device=device)

        # Test a few known conversions for 16-QAM
        # Index 0 -> bits 0000 -> PAM1=00, PAM2=00 -> indices 0, 0
        ind_pam1, ind_pam2 = converter(torch.tensor([0], device=device))
        assert ind_pam1.item() == 0
        assert ind_pam2.item() == 0

        # Index 15 -> bits 1111 -> PAM1=11, PAM2=11 -> indices 3, 3
        ind_pam1, ind_pam2 = converter(torch.tensor([15], device=device))
        assert ind_pam1.item() == 3
        assert ind_pam2.item() == 3

    def test_output_shape(self, device):
        """Verify output shapes are correct."""
        converter = QAM2PAM(4, device=device)
        indices = torch.tensor([0, 5, 10, 15], device=device)
        ind_pam1, ind_pam2 = converter(indices)

        assert ind_pam1.shape == torch.Size([4])
        assert ind_pam2.shape == torch.Size([4])


# =============================================================================
# Tests for PAM2QAM class
# =============================================================================


class TestPAM2QAM:
    """Tests for PAM2QAM class."""

    def test_hard_conversion(self, device):
        """Verify PAM to QAM hard index conversion is correct."""
        converter = PAM2QAM(4, hard_in_out=True, device=device)

        # Test round-trip with QAM2PAM
        qam2pam = QAM2PAM(4, device=device)
        original = torch.tensor([0, 5, 10, 15], device=device)

        ind_pam1, ind_pam2 = qam2pam(original)
        recovered = converter(ind_pam1, ind_pam2)

        assert torch.equal(original, recovered)

    def test_soft_conversion(self, device):
        """Verify PAM to QAM soft logits conversion."""
        converter = PAM2QAM(4, hard_in_out=False, device=device)

        pam1_logits = torch.randn(2, 10, 4, device=device)
        pam2_logits = torch.randn(2, 10, 4, device=device)

        qam_logits = converter(pam1_logits, pam2_logits)

        assert qam_logits.shape == torch.Size([2, 10, 16])


# =============================================================================
# Tests for BinarySource class
# =============================================================================


class TestBinarySource:
    """Tests for BinarySource class."""

    def test_output_shape(self, device):
        """Verify output has correct shape."""
        source = BinarySource(device=device)
        bits = source([10, 100])
        assert bits.shape == torch.Size([10, 100])

    def test_output_values(self, device):
        """Verify output contains only 0s and 1s."""
        source = BinarySource(device=device)
        bits = source([100, 100])
        assert torch.all((bits == 0) | (bits == 1))

    def test_dtype(self, precision, device):
        """Verify output dtype matches requested precision."""
        source = BinarySource(precision=precision, device=device)
        bits = source([10, 10])
        expected_dtype = dtypes[precision]["torch"]["dtype"]
        assert bits.dtype == expected_dtype

    def test_docstring_example(self):
        """Verify the docstring example works correctly."""
        source = BinarySource()
        bits = source([10, 100])
        assert bits.shape == torch.Size([10, 100])


# =============================================================================
# Tests for SymbolSource class
# =============================================================================


class TestSymbolSource:
    """Tests for SymbolSource class."""

    def test_output_shape(self, device):
        """Verify output has correct shape."""
        source = SymbolSource("qam", 4, device=device)
        symbols = source([10, 100])
        assert symbols.shape == torch.Size([10, 100])

    def test_return_indices(self, device):
        """Verify indices can be returned."""
        source = SymbolSource("qam", 4, return_indices=True, device=device)
        result = source([10, 100])

        assert len(result) == 2
        assert result[0].shape == torch.Size([10, 100])
        assert result[1].shape == torch.Size([10, 100])

    def test_return_bits(self, device):
        """Verify bits can be returned."""
        source = SymbolSource("qam", 4, return_bits=True, device=device)
        result = source([10, 100])

        assert len(result) == 2
        assert result[0].shape == torch.Size([10, 100])
        assert result[1].shape == torch.Size([10, 400])

    def test_return_indices_and_bits(self, device):
        """Verify both indices and bits can be returned."""
        source = SymbolSource(
            "qam", 4, return_indices=True, return_bits=True, device=device
        )
        result = source([10, 100])

        assert len(result) == 3
        assert result[0].shape == torch.Size([10, 100])  # symbols
        assert result[1].shape == torch.Size([10, 100])  # indices
        assert result[2].shape == torch.Size([10, 400])  # bits

    def test_tensor_shape_is_not_mutated(self, device):
        """A tensor-valued shape must remain unchanged after a call."""
        shape = torch.tensor([10, 100], device=device)
        expected_shape = shape.clone()
        source = SymbolSource("qam", 4, return_bits=True, device=device)

        symbols, bits = source(shape)

        assert torch.equal(shape, expected_shape)
        assert symbols.shape == torch.Size([10, 100])
        assert bits.shape == torch.Size([10, 400])

    @pytest.mark.parametrize("shape", [[], (), torch.Size([])])
    def test_empty_shape_is_rejected(self, shape, device):
        """An empty shape has no symbol dimension and must be rejected."""
        source = SymbolSource("qam", 4, return_bits=True, device=device)

        with pytest.raises(ValueError, match="at least one dimension"):
            source(shape)

    @pytest.mark.parametrize("shape", [[10, 100], [10, 1], [100]])
    @pytest.mark.parametrize(
        "source, constellation_type, num_bits_per_symbol",
        [
            (QAMSource, "qam", 4),
            (PAMSource, "pam", 2),
        ],
    )
    def test_returned_bits_match_demapper(
        self, source, constellation_type, num_bits_per_symbol, shape, device
    ):
        """Returned bits have the demapper's shape and ordering."""
        symbol_source = source(
            num_bits_per_symbol, return_bits=True, device=device
        )
        demapper = Demapper(
            "app",
            constellation_type,
            num_bits_per_symbol,
            hard_out=True,
            device=device,
        )

        symbols, bits = symbol_source(shape)
        recovered_bits = demapper(
            symbols, torch.tensor(1e-7, device=device)
        )

        assert symbols.shape == torch.Size(shape)
        assert bits.shape == torch.Size(
            shape[:-1] + [shape[-1] * num_bits_per_symbol]
        )
        assert bits.shape == recovered_bits.shape
        assert torch.equal(bits, recovered_bits)

    def test_docstring_example(self):
        """Verify the docstring example works correctly."""
        source = SymbolSource("qam", 4)
        symbols = source([10, 100])
        assert symbols.shape == torch.Size([10, 100])


# =============================================================================
# Tests for QAMSource class
# =============================================================================


class TestQAMSource:
    """Tests for QAMSource class."""

    def test_output_shape(self, device):
        """Verify output has correct shape."""
        source = QAMSource(4, device=device)
        symbols = source([10, 100])
        assert symbols.shape == torch.Size([10, 100])

    def test_dtype(self, precision, device):
        """Verify output dtype matches requested precision."""
        source = QAMSource(4, precision=precision, device=device)
        symbols = source([10, 10])
        expected_dtype = dtypes[precision]["torch"]["cdtype"]
        assert symbols.dtype == expected_dtype

    def test_docstring_example(self):
        """Verify the docstring example works correctly."""
        source = QAMSource(4)
        symbols = source([10, 100])
        assert symbols.shape == torch.Size([10, 100])


# =============================================================================
# Tests for PAMSource class
# =============================================================================


class TestPAMSource:
    """Tests for PAMSource class."""

    def test_output_shape(self, device):
        """Verify output has correct shape."""
        source = PAMSource(2, device=device)
        symbols = source([10, 100])
        assert symbols.shape == torch.Size([10, 100])

    def test_docstring_example(self):
        """Verify the docstring example works correctly."""
        source = PAMSource(2)
        symbols = source([10, 100])
        assert symbols.shape == torch.Size([10, 100])


# =============================================================================
# Tests for torch.compile compatibility
# =============================================================================


class TestCompileCompatibility:
    """Tests for torch.compile compatibility."""

    def test_mapper_compile(self, device):
        """Verify Mapper works with torch.compile."""
        if device.startswith("cuda"):
            pytest.skip("torch.compile on CUDA requires ptxas toolchain")

        mapper = Mapper("qam", 4, device=device)

        @torch.compile
        def run(bits):
            return mapper(bits)

        bits = torch.randint(0, 2, (10, 40), device=device).float()
        result = run(bits)
        assert result.shape == torch.Size([10, 10])

    def test_demapper_compile(self, device):
        """Verify Demapper works with torch.compile."""
        if device.startswith("cuda"):
            pytest.skip("torch.compile on CUDA requires ptxas toolchain")

        demapper = Demapper("app", "qam", 4, device=device)

        @torch.compile
        def run(y, no):
            return demapper(y, no)

        y = torch.randn(10, 10, dtype=torch.complex64, device=device)
        no = torch.tensor(0.1, device=device)
        result = run(y, no)
        assert result.shape == torch.Size([10, 40])
