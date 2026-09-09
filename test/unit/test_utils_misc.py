#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Unit tests for sionna.phy.utils.misc module."""

import numpy as np
import pytest
import torch

from sionna.phy import dtypes
from sionna.phy.utils.misc import (
    complex_normal,
    lin_to_db,
    db_to_lin,
    watt_to_dbm,
    dbm_to_watt,
    ebnodb2no,
    hard_decisions,
    sample_bernoulli,
    sim_ber,
    to_list,
    dict_keys_to_int,
    scalar_to_shaped_tensor,
    DeepUpdateDict,
    SplineGriddataInterpolation,
    SingleLinkChannel,
)


# =============================================================================
# Helper class for sim_ber testing
# =============================================================================


class MockErrorGenerator:
    """Utility class to emulate monte-carlo simulation with predefined errors.

    This class generates deterministic error patterns for testing sim_ber.
    """

    def __init__(self, num_errors_per_snr: list, shape: tuple, device: str = "cpu"):
        """
        :param num_errors_per_snr: List of error counts, one per SNR point.
        :param shape: Shape of the output tensors [batch, bits_per_block].
        :param device: Device for output tensors.
        """
        self.shape = shape
        self.errors_per_snr = num_errors_per_snr
        self.device = device
        self.call_count = 0

    def reset(self):
        """Reset internal call counter."""
        self.call_count = 0

    def __call__(self, batch_size: int, ebno_db: torch.Tensor):
        """Generate b and b_hat with a predetermined number of errors.

        The inputs are ignored but required for sim_ber interface.
        """
        num_errors = self.errors_per_snr[self.call_count]
        self.call_count += 1

        # Create ground truth (all zeros)
        b = torch.zeros(self.shape, dtype=torch.float32, device=self.device)

        # Create estimate with predetermined number of errors
        b_hat = torch.zeros(self.shape, dtype=torch.float32, device=self.device)
        flat_b_hat = b_hat.flatten()

        # Distribute errors across the tensor
        num_elements = flat_b_hat.numel()
        error_indices = torch.randperm(num_elements, device=self.device)[:num_errors]
        flat_b_hat[error_indices] = 1.0
        b_hat = flat_b_hat.reshape(self.shape)

        return b, b_hat


# =============================================================================
# Tests for complex_normal
# =============================================================================


class TestComplexNormal:
    """Tests for complex_normal function."""

    def test_variance(self, device):
        """Verify that generated samples have the correct variance."""
        shape = [1000000]
        variances = [0.5, 1.0, 2.3, 10.0]

        for var in variances:
            x = complex_normal(shape, var=var, device=device)
            actual_var = torch.var(x.real).item() + torch.var(x.imag).item()
            assert np.isclose(
                var, actual_var, rtol=1e-2
            ), f"Expected variance {var}, got {actual_var}"

    def test_default_variance(self, device):
        """Verify default variance is 1.0."""
        shape = [1000000]
        x = complex_normal(shape, device=device)
        actual_var = torch.var(x.real).item() + torch.var(x.imag).item()
        assert np.isclose(1.0, actual_var, rtol=1e-2)

    def test_real_imag_balanced(self, device):
        """Verify real and imaginary parts have equal variance."""
        shape = [1000000]
        x = complex_normal(shape, var=2.0, device=device)
        var_real = torch.var(x.real).item()
        var_imag = torch.var(x.imag).item()
        assert np.isclose(var_real, var_imag, rtol=1e-2)

    def test_precision(self, precision, device):
        """Verify output dtype matches requested precision."""
        x = complex_normal([100], precision=precision, device=device)
        expected_dtype = dtypes[precision]["torch"]["cdtype"]
        assert x.dtype == expected_dtype

    def test_shape(self, device):
        """Verify output shape matches requested shape."""
        shapes = [[100], [7, 8, 5], [4, 5, 67, 8]]
        for shape in shapes:
            x = complex_normal(shape, device=device)
            assert list(x.shape) == shape

    def test_docstring_example(self):
        """Verify the docstring example works correctly."""
        x = complex_normal([2, 3], var=2.0)
        assert x.shape == torch.Size([2, 3])


# =============================================================================
# Tests for lin_to_db and db_to_lin
# =============================================================================


class TestLinToDb:
    """Tests for lin_to_db function."""

    def test_known_values(self):
        """Verify conversion for known values."""
        test_cases = [
            (1.0, 0.0),
            (10.0, 10.0),
            (100.0, 20.0),
            (1000.0, 30.0),
            (0.1, -10.0),
            (0.01, -20.0),
        ]
        for lin, db in test_cases:
            result = lin_to_db(torch.tensor(lin))
            assert np.isclose(
                result.item(), db, rtol=1e-5
            ), f"lin_to_db({lin}) = {result.item()}, expected {db}"

    def test_accepts_python_float(self):
        """Verify function accepts Python float input."""
        result = lin_to_db(100.0)
        assert np.isclose(result.item(), 20.0, rtol=1e-5)

    def test_precision(self, precision):
        """Verify output dtype matches requested precision."""
        result = lin_to_db(torch.tensor(10.0), precision=precision)
        expected_dtype = dtypes[precision]["torch"]["dtype"]
        assert result.dtype == expected_dtype

    def test_docstring_example(self):
        """Verify the docstring example works correctly."""
        x = torch.tensor(100.0)
        assert np.isclose(lin_to_db(x).item(), 20.0, rtol=1e-5)


class TestDbToLin:
    """Tests for db_to_lin function."""

    def test_known_values(self):
        """Verify conversion for known values."""
        test_cases = [
            (0.0, 1.0),
            (10.0, 10.0),
            (20.0, 100.0),
            (30.0, 1000.0),
            (-10.0, 0.1),
            (-20.0, 0.01),
        ]
        for db, lin in test_cases:
            result = db_to_lin(torch.tensor(db))
            assert np.isclose(
                result.item(), lin, rtol=1e-5
            ), f"db_to_lin({db}) = {result.item()}, expected {lin}"

    def test_accepts_python_float(self):
        """Verify function accepts Python float input."""
        result = db_to_lin(20.0)
        assert np.isclose(result.item(), 100.0, rtol=1e-5)

    def test_precision(self, precision):
        """Verify output dtype matches requested precision."""
        result = db_to_lin(torch.tensor(10.0), precision=precision)
        expected_dtype = dtypes[precision]["torch"]["dtype"]
        assert result.dtype == expected_dtype

    def test_docstring_example(self):
        """Verify the docstring example works correctly."""
        x = torch.tensor(20.0)
        assert np.isclose(db_to_lin(x).item(), 100.0, rtol=1e-5)


class TestLinDbRoundTrip:
    """Tests for lin_to_db and db_to_lin round-trip conversion."""

    def test_round_trip_lin_db_lin(self):
        """Verify lin -> dB -> lin round-trip."""
        values = [0.001, 0.1, 1.0, 10.0, 100.0, 1000.0]
        for v in values:
            result = db_to_lin(lin_to_db(torch.tensor(v)))
            assert np.isclose(result.item(), v, rtol=1e-5)

    def test_round_trip_db_lin_db(self):
        """Verify dB -> lin -> dB round-trip."""
        values = [-30.0, -10.0, 0.0, 10.0, 30.0]
        for v in values:
            result = lin_to_db(db_to_lin(torch.tensor(v)))
            assert np.isclose(result.item(), v, rtol=1e-5)


# =============================================================================
# Tests for watt_to_dbm and dbm_to_watt
# =============================================================================


class TestWattToDbm:
    """Tests for watt_to_dbm function."""

    def test_known_values(self):
        """Verify conversion for known values."""
        test_cases = [
            (1.0, 30.0),  # 1 W = 30 dBm
            (0.001, 0.0),  # 1 mW = 0 dBm
            (0.01, 10.0),  # 10 mW = 10 dBm
            (0.1, 20.0),  # 100 mW = 20 dBm
        ]
        for watt, dbm in test_cases:
            result = watt_to_dbm(torch.tensor(watt))
            assert np.isclose(result.item(), dbm, rtol=1e-5, atol=1e-6)

    def test_docstring_example(self):
        """Verify the docstring example works correctly."""
        x = torch.tensor(1.0)
        assert np.isclose(watt_to_dbm(x).item(), 30.0, rtol=1e-5)


class TestDbmToWatt:
    """Tests for dbm_to_watt function."""

    def test_known_values(self):
        """Verify conversion for known values."""
        test_cases = [
            (30.0, 1.0),  # 30 dBm = 1 W
            (0.0, 0.001),  # 0 dBm = 1 mW
            (10.0, 0.01),  # 10 dBm = 10 mW
            (20.0, 0.1),  # 20 dBm = 100 mW
        ]
        for dbm, watt in test_cases:
            result = dbm_to_watt(torch.tensor(dbm))
            assert np.isclose(result.item(), watt, rtol=1e-5)

    def test_docstring_example(self):
        """Verify the docstring example works correctly."""
        x = torch.tensor(30.0)
        assert np.isclose(dbm_to_watt(x).item(), 1.0, rtol=1e-5)


class TestWattDbmRoundTrip:
    """Tests for watt_to_dbm and dbm_to_watt round-trip conversion."""

    def test_round_trip(self):
        """Verify round-trip conversion."""
        watts = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]
        for w in watts:
            result = dbm_to_watt(watt_to_dbm(torch.tensor(w)))
            assert np.isclose(result.item(), w, rtol=1e-5)


# =============================================================================
# Tests for ebnodb2no
# =============================================================================


class TestEbnoDb2No:
    """Tests for ebnodb2no function."""

    def test_basic_computation(self, device):
        """Verify basic computation of noise variance."""
        # For Eb/No = 10 dB, coderate = 0.5, 4 bits/symbol:
        # ebno = 10, no = 1 / (10 * 0.5 * 4) = 0.05
        no = ebnodb2no(10.0, num_bits_per_symbol=4, coderate=0.5, device=device)
        assert np.isclose(no.item(), 0.05, rtol=1e-5)
        assert no.device.type == device.split(":")[0]

    def test_tensor_input(self, device):
        """Verify function works with tensor input."""
        ebno_db = torch.tensor(10.0, device=device)
        no = ebnodb2no(ebno_db, num_bits_per_symbol=4, coderate=0.5)
        assert np.isclose(no.item(), 0.05, rtol=1e-5)

    def test_varying_coderate(self, device):
        """Verify noise variance scales inversely with coderate."""
        no_half = ebnodb2no(10.0, num_bits_per_symbol=4, coderate=0.5, device=device)
        no_full = ebnodb2no(10.0, num_bits_per_symbol=4, coderate=1.0, device=device)
        assert np.isclose(no_half.item() / no_full.item(), 2.0, rtol=1e-5)

    def test_varying_bits_per_symbol(self, device):
        """Verify noise variance scales inversely with bits per symbol."""
        no_2 = ebnodb2no(10.0, num_bits_per_symbol=2, coderate=0.5, device=device)
        no_4 = ebnodb2no(10.0, num_bits_per_symbol=4, coderate=0.5, device=device)
        assert np.isclose(no_2.item() / no_4.item(), 2.0, rtol=1e-5)

    @pytest.mark.parametrize("num_bits_per_symbol", [0, -1])
    def test_nonpositive_num_bits_per_symbol(self, num_bits_per_symbol):
        """The modulation order must be positive."""
        with pytest.raises(ValueError, match="positive"):
            ebnodb2no(10.0, num_bits_per_symbol, 0.5)

    @pytest.mark.parametrize("num_bits_per_symbol", [1.5, True])
    def test_nonintegral_num_bits_per_symbol(self, num_bits_per_symbol):
        """The modulation order must be an integer, excluding booleans."""
        with pytest.raises(TypeError, match="integer"):
            ebnodb2no(10.0, num_bits_per_symbol, 0.5)

    @pytest.mark.parametrize("coderate", [0.0, -0.1, 1.1, np.nan, np.inf])
    def test_invalid_coderate(self, coderate):
        """Coderates must be finite and lie within (0, 1]."""
        with pytest.raises(ValueError, match=r"within \(0, 1\]"):
            ebnodb2no(10.0, 4, coderate)


# =============================================================================
# Tests for hard_decisions
# =============================================================================


class TestHardDecisions:
    """Tests for hard_decisions function."""

    def test_positive_values(self):
        """Verify positive LLRs map to 1."""
        llr = torch.tensor([0.1, 1.0, 10.0, 100.0])
        result = hard_decisions(llr)
        assert torch.all(result == 1.0)

    def test_negative_values(self):
        """Verify negative LLRs map to 0."""
        llr = torch.tensor([-0.1, -1.0, -10.0, -100.0])
        result = hard_decisions(llr)
        assert torch.all(result == 0.0)

    def test_zero_maps_to_zero(self):
        """Verify zero LLR maps to 0 (nonpositive)."""
        llr = torch.tensor([0.0])
        result = hard_decisions(llr)
        assert result.item() == 0.0

    def test_preserves_dtype(self):
        """Verify output dtype matches input dtype."""
        for dtype in [torch.float32, torch.float64]:
            llr = torch.tensor([1.0, -1.0], dtype=dtype)
            result = hard_decisions(llr)
            assert result.dtype == dtype

    def test_docstring_example(self):
        """Verify the docstring example works correctly."""
        llr = torch.tensor([-1.5, 0.0, 2.3, -0.1])
        result = hard_decisions(llr)
        expected = torch.tensor([0.0, 0.0, 1.0, 0.0])
        assert torch.allclose(result, expected)


# =============================================================================
# Tests for sample_bernoulli
# =============================================================================


class TestSampleBernoulli:
    """Tests for sample_bernoulli function."""

    def test_probability_zero(self, device):
        """Verify p=0 produces all False."""
        samples = sample_bernoulli([1000], p=0.0, device=device)
        assert samples.sum().item() == 0

    def test_probability_one(self, device):
        """Verify p=1 produces all True."""
        samples = sample_bernoulli([1000], p=1.0, device=device)
        assert samples.sum().item() == 1000

    def test_probability_half(self, device):
        """Verify p=0.5 produces approximately 50% True."""
        samples = sample_bernoulli([100000], p=0.5, device=device)
        proportion = samples.sum().item() / 100000
        assert np.isclose(proportion, 0.5, atol=0.01)

    def test_shape(self, device):
        """Verify output shape matches requested shape."""
        shapes = [[100], [10, 20], [5, 10, 20]]
        for shape in shapes:
            samples = sample_bernoulli(shape, p=0.5, device=device)
            assert list(samples.shape) == shape

    def test_returns_boolean(self, device):
        """Verify output is boolean tensor."""
        samples = sample_bernoulli([100], p=0.5, device=device)
        assert samples.dtype == torch.bool

    @pytest.mark.parametrize("p", [-0.1, 1.1, np.nan, np.inf])
    def test_invalid_scalar_probability(self, p, device):
        """Scalar probabilities must be finite and within [0, 1]."""
        with pytest.raises(ValueError, match=r"within \[0, 1\]"):
            sample_bernoulli([10], p=p, device=device)

    def test_invalid_sequence_probability(self, device):
        """Every value in a host-side probability sequence is validated."""
        with pytest.raises(ValueError, match=r"within \[0, 1\]"):
            sample_bernoulli([2], p=[0.2, -0.1], device=device)

    def test_tensor_probability_is_not_validated(self, device):
        """Reading back a device tensor would stall the host, so skip it."""
        p = torch.tensor([0.2, -0.1], device=device)
        assert sample_bernoulli([2], p=p, device=device).shape == (2,)

    @pytest.mark.parametrize("as_tensor", [False, True])
    def test_compiled(self, as_tensor, device):
        """Validation must not force a graph break or a data-dependent guard."""

        @torch.compile(fullgraph=True)
        def sample(p):
            return sample_bernoulli([32], p=p, device=device)

        p = torch.tensor(0.5, device=device) if as_tensor else 0.5
        result = sample(p)
        assert result.shape == (32,)
        assert result.dtype == torch.bool


# =============================================================================
# Tests for sim_ber
# =============================================================================


class TestSimBer:
    """Exhaustive tests for sim_ber function."""

    def test_correct_ber_calculation(self, device):
        """Verify BER is correctly calculated from predetermined errors."""
        shape = (500, 200)  # 100,000 bits total
        errors_per_snr = [1000, 400, 200, 100, 10, 0]
        expected_ber = np.array(errors_per_snr) / np.prod(shape)

        mock = MockErrorGenerator(errors_per_snr, shape, device=device)
        ebno_dbs = torch.zeros(len(errors_per_snr))

        ber, _ = sim_ber(
            mock,
            ebno_dbs,
            batch_size=1,
            max_mc_iter=1,
            early_stop=False,
            verbose=False,
            device=device,
        )

        assert np.allclose(ber.cpu().numpy(), expected_ber, rtol=1e-5)

    def test_early_stop_on_no_errors(self, device):
        """Verify simulation stops early when no errors occur."""
        shape = (100, 100)
        # Third SNR point has 0 errors - should trigger early stop
        errors_per_snr = [100, 50, 0, 10, 5]

        mock = MockErrorGenerator(errors_per_snr, shape, device=device)
        ebno_dbs = torch.zeros(len(errors_per_snr))

        ber, _ = sim_ber(
            mock,
            ebno_dbs,
            batch_size=1,
            max_mc_iter=1,
            early_stop=True,
            verbose=False,
            device=device,
        )

        # Only first 3 SNR points should be simulated
        assert mock.call_count == 3
        # Remaining points should have BER=0
        assert ber[3].item() == 0.0
        assert ber[4].item() == 0.0

    def test_early_stop_disabled(self, device):
        """Verify all SNR points are simulated when early_stop=False."""
        shape = (100, 100)
        errors_per_snr = [100, 50, 0, 10, 5]

        mock = MockErrorGenerator(errors_per_snr, shape, device=device)
        ebno_dbs = torch.zeros(len(errors_per_snr))

        ber, _ = sim_ber(
            mock,
            ebno_dbs,
            batch_size=1,
            max_mc_iter=1,
            early_stop=False,
            verbose=False,
            device=device,
        )

        # All SNR points should be simulated
        assert mock.call_count == len(errors_per_snr)
        # BER at third point should be 0
        assert ber[2].item() == 0.0
        # But 4th and 5th should have non-zero BER
        assert ber[3].item() > 0
        assert ber[4].item() > 0

    def test_max_mc_iter_accumulation(self, device):
        """Verify errors accumulate correctly across multiple MC iterations."""
        shape = (100, 100)  # 10,000 bits
        # Each iteration returns 100 errors
        errors_per_iter = [100, 100, 100]  # 3 iterations

        mock = MockErrorGenerator(errors_per_iter, shape, device=device)
        ebno_dbs = torch.zeros(1)

        ber, _ = sim_ber(
            mock,
            ebno_dbs,
            batch_size=1,
            max_mc_iter=3,
            early_stop=False,
            verbose=False,
            device=device,
        )

        # Total errors = 300, total bits = 30,000
        expected_ber = 300 / 30000
        assert np.isclose(ber[0].item(), expected_ber, rtol=1e-5)

    def test_num_target_bit_errors_stopping(self, device):
        """Verify simulation stops when target bit errors is reached."""
        shape = (100, 100)  # 10,000 bits
        # Each iteration returns 500 errors, need 1000 to stop
        errors_per_iter = [500, 500, 500, 500, 500]

        mock = MockErrorGenerator(errors_per_iter, shape, device=device)
        ebno_dbs = torch.zeros(1)

        ber, _ = sim_ber(
            mock,
            ebno_dbs,
            batch_size=1,
            max_mc_iter=5,
            num_target_bit_errors=1000,
            early_stop=False,
            verbose=False,
            device=device,
        )

        # Should stop after 2 iterations (1000 errors reached)
        assert mock.call_count == 2

    def test_num_target_block_errors_stopping(self, device):
        """Verify simulation stops when target block errors is reached."""
        shape = (100, 100)  # 100 blocks
        # Each block with any error counts as block error
        # With 100 random errors in 100 blocks, expect ~63% block errors
        errors_per_iter = [100, 100, 100, 100, 100]

        mock = MockErrorGenerator(errors_per_iter, shape, device=device)
        ebno_dbs = torch.zeros(1)

        sim_ber(
            mock,
            ebno_dbs,
            batch_size=1,
            max_mc_iter=5,
            num_target_block_errors=50,
            early_stop=False,
            verbose=False,
            device=device,
        )

        # Should stop before all 5 iterations
        assert mock.call_count < 5

    def test_target_ber_early_stop(self, device):
        """Verify simulation stops when target BER is reached."""
        shape = (1000, 100)  # 100,000 bits
        # Decreasing error counts to simulate improving channel
        errors_per_snr = [5000, 1000, 100, 10]
        # Target BER of 0.001 should be reached at SNR point 3

        mock = MockErrorGenerator(errors_per_snr, shape, device=device)
        ebno_dbs = torch.zeros(len(errors_per_snr))

        sim_ber(
            mock,
            ebno_dbs,
            batch_size=1,
            max_mc_iter=1,
            target_ber=0.001,  # Stop when BER < 0.001
            early_stop=True,
            verbose=False,
            device=device,
        )

        # Should stop at SNR point 3 (BER = 100/100000 = 0.001)
        # or point 4 (BER = 10/100000 = 0.0001 < target)
        assert mock.call_count <= 4

    def test_soft_estimates_mode(self, device):
        """Verify soft estimates are converted to hard decisions."""

        def soft_mc_fun(batch_size, ebno_db):  # noqa: ARG001
            b = torch.zeros((batch_size, 100), device=device)
            # Return soft estimates (LLRs): positive = bit 1, negative = bit 0
            b_hat_soft = torch.randn((batch_size, 100), device=device)
            return b, b_hat_soft

        ebno_dbs = torch.zeros(1)

        # With soft_estimates=True, should apply hard_decisions internally
        ber, _ = sim_ber(
            soft_mc_fun,
            ebno_dbs,
            batch_size=100,
            max_mc_iter=1,
            soft_estimates=True,
            early_stop=False,
            verbose=False,
            device=device,
        )

        # BER should be approximately 0.5 (random LLRs)
        assert 0.3 < ber[0].item() < 0.7

    def test_callback_continue(self, device):
        """Verify callback with CALLBACK_CONTINUE doesn't affect simulation."""
        shape = (100, 100)
        errors_per_snr = [100, 50, 25]
        call_count = [0]

        def callback(
            mc_iter, snr_idx, ebno_dbs, bit_errors, block_errors, nb_bits, nb_blocks
        ):  # noqa: ARG001
            call_count[0] += 1
            return sim_ber.CALLBACK_CONTINUE

        mock = MockErrorGenerator(errors_per_snr, shape, device=device)
        ebno_dbs = torch.zeros(len(errors_per_snr))

        sim_ber(
            mock,
            ebno_dbs,
            batch_size=1,
            max_mc_iter=1,
            early_stop=False,
            verbose=False,
            callback=callback,
            device=device,
        )

        # Callback should be called once per iteration
        assert call_count[0] == len(errors_per_snr)

    def test_callback_next_snr(self, device):
        """Verify CALLBACK_NEXT_SNR skips to next SNR point."""
        shape = (100, 100)
        errors_per_iter = [100] * 10  # Would normally run 10 iterations

        call_count = [0]

        def callback(
            mc_iter, snr_idx, ebno_dbs, bit_errors, block_errors, nb_bits, nb_blocks
        ):  # noqa: ARG001
            call_count[0] += 1
            # Skip to next SNR after 2 iterations
            if mc_iter >= 1:
                return sim_ber.CALLBACK_NEXT_SNR
            return sim_ber.CALLBACK_CONTINUE

        mock = MockErrorGenerator(errors_per_iter, shape, device=device)
        ebno_dbs = torch.zeros(1)

        sim_ber(
            mock,
            ebno_dbs,
            batch_size=1,
            max_mc_iter=10,
            early_stop=False,
            verbose=False,
            callback=callback,
            device=device,
        )

        # Should stop after 2 iterations due to callback
        assert call_count[0] == 2

    def test_callback_stop(self, device):
        """Verify CALLBACK_STOP terminates entire simulation."""
        shape = (100, 100)
        errors_per_snr = [100, 50, 25, 10, 5]

        def callback(
            mc_iter, snr_idx, ebno_dbs, bit_errors, block_errors, nb_bits, nb_blocks
        ):  # noqa: ARG001
            # Stop after SNR point 2
            if snr_idx >= 2:
                return sim_ber.CALLBACK_STOP
            return sim_ber.CALLBACK_CONTINUE

        mock = MockErrorGenerator(errors_per_snr, shape, device=device)
        ebno_dbs = torch.zeros(len(errors_per_snr))

        sim_ber(
            mock,
            ebno_dbs,
            batch_size=1,
            max_mc_iter=1,
            early_stop=False,
            verbose=False,
            callback=callback,
            device=device,
        )

        # Should stop after 3 SNR points
        assert mock.call_count == 3

    def test_empty_ebno_dbs(self, device):
        """Verify handling of empty SNR list."""

        def dummy_mc_fun(batch_size, ebno_db):  # noqa: ARG001
            return torch.zeros(10), torch.zeros(10)

        ebno_dbs = torch.tensor([])

        ber, bler = sim_ber(
            dummy_mc_fun,
            ebno_dbs,
            batch_size=1,
            max_mc_iter=1,
            early_stop=False,
            verbose=False,
            device=device,
        )

        assert len(ber) == 0
        assert len(bler) == 0

    def test_ebno_dbs_as_python_list(self, device):
        """Verify sim_ber accepts Python list for ebno_dbs."""
        shape = (100, 100)
        errors_per_snr = [100, 50, 25]
        expected_ber = np.array(errors_per_snr) / np.prod(shape)

        mock = MockErrorGenerator(errors_per_snr, shape, device=device)
        # Pass ebno_dbs as a Python list instead of tensor
        ebno_dbs = [0.0, 5.0, 10.0]

        ber, _ = sim_ber(
            mock,
            ebno_dbs,
            batch_size=1,
            max_mc_iter=1,
            early_stop=False,
            verbose=False,
            device=device,
        )

        assert np.allclose(ber.cpu().numpy(), expected_ber, rtol=1e-5)

    def test_input_validation_early_stop(self):
        """Verify TypeError for non-bool early_stop."""

        def dummy_mc_fun(batch_size, ebno_db):  # noqa: ARG001
            return torch.zeros(10), torch.zeros(10)

        with pytest.raises(TypeError, match="early_stop must be bool"):
            sim_ber(
                dummy_mc_fun, torch.zeros(1), 1, 1, early_stop="true", verbose=False
            )

    def test_input_validation_soft_estimates(self):
        """Verify TypeError for non-bool soft_estimates."""

        def dummy_mc_fun(batch_size, ebno_db):  # noqa: ARG001
            return torch.zeros(10), torch.zeros(10)

        with pytest.raises(TypeError, match="soft_estimates must be bool"):
            sim_ber(dummy_mc_fun, torch.zeros(1), 1, 1, soft_estimates=1, verbose=False)

    def test_input_validation_verbose(self):
        """Verify TypeError for non-bool verbose."""

        def dummy_mc_fun(batch_size, ebno_db):  # noqa: ARG001
            return torch.zeros(10), torch.zeros(10)

        with pytest.raises(TypeError, match="verbose must be bool"):
            sim_ber(dummy_mc_fun, torch.zeros(1), 1, 1, verbose="yes")

    @pytest.mark.parametrize(
        ("argument", "value", "error"),
        [
            ("batch_size", 0, ValueError),
            ("batch_size", -1, ValueError),
            ("batch_size", 1.5, TypeError),
            ("batch_size", True, TypeError),
            ("max_mc_iter", 0, ValueError),
            ("max_mc_iter", -1, ValueError),
            ("max_mc_iter", 1.5, TypeError),
            ("max_mc_iter", True, TypeError),
        ],
    )
    def test_positive_sample_count_validation(self, argument, value, error):
        """Zero-sample simulations fail before invoking the callback."""

        def dummy_mc_fun(batch_size, ebno_db):  # noqa: ARG001
            return torch.zeros(10), torch.zeros(10)

        kwargs = {"batch_size": 1, "max_mc_iter": 1, argument: value}
        with pytest.raises(error, match=argument):
            sim_ber(
                dummy_mc_fun,
                torch.zeros(1),
                verbose=False,
                **kwargs,
            )

    @pytest.mark.parametrize(
        "outputs",
        [
            (torch.empty(1, 0), torch.empty(1, 0)),
            (torch.empty(0, 10), torch.empty(0, 10)),
            (torch.zeros(2, 4), torch.zeros(2, 5)),
        ],
    )
    def test_invalid_mc_fun_output_shape(self, outputs):
        """Empty and mismatched callback outputs fail explicitly."""

        def mc_fun(batch_size, ebno_db):  # noqa: ARG001
            return outputs

        with pytest.raises(ValueError, match="mc_fun"):
            sim_ber(
                mc_fun,
                torch.zeros(1),
                batch_size=1,
                max_mc_iter=1,
                verbose=False,
            )

    def test_precision_output(self, precision, device):
        """Verify output tensors have correct precision."""
        shape = (100, 100)
        errors_per_snr = [100]

        mock = MockErrorGenerator(errors_per_snr, shape, device=device)
        ebno_dbs = torch.zeros(1)

        ber, bler = sim_ber(
            mock,
            ebno_dbs,
            batch_size=1,
            max_mc_iter=1,
            early_stop=False,
            verbose=False,
            precision=precision,
            device=device,
        )

        expected_dtype = dtypes[precision]["torch"]["dtype"]
        assert ber.dtype == expected_dtype
        assert bler.dtype == expected_dtype

    def test_compile_mode(self, mode, device):
        """Verify sim_ber works with different torch.compile modes."""
        shape = (100, 100)
        errors_per_snr = [100, 50, 25]
        expected_ber = np.array(errors_per_snr) / np.prod(shape)

        mock = MockErrorGenerator(errors_per_snr, shape, device=device)
        ebno_dbs = torch.zeros(len(errors_per_snr))

        ber, _ = sim_ber(
            mock,
            ebno_dbs,
            batch_size=1,
            max_mc_iter=1,
            early_stop=False,
            verbose=False,
            compile_mode=mode,
            device=device,
        )

        # BER should be correctly calculated regardless of compile mode
        assert np.allclose(ber.cpu().numpy(), expected_ber, rtol=1e-5)

    def test_compile_mode_none(self, device):
        """Verify sim_ber works without compilation (compile_mode=None)."""
        shape = (100, 100)
        errors_per_snr = [100]

        mock = MockErrorGenerator(errors_per_snr, shape, device=device)
        ebno_dbs = torch.zeros(1)

        ber, _ = sim_ber(
            mock,
            ebno_dbs,
            batch_size=1,
            max_mc_iter=1,
            early_stop=False,
            verbose=False,
            compile_mode=None,  # Explicit None
            device=device,
        )

        expected_ber = 100 / np.prod(shape)
        assert np.isclose(ber[0].item(), expected_ber, rtol=1e-5)

    def test_compile_mode_invalid_type(self):
        """Verify TypeError for non-string compile_mode."""

        def dummy_mc_fun(batch_size, ebno_db):  # noqa: ARG001
            return torch.zeros(10), torch.zeros(10)

        with pytest.raises(TypeError, match="compile_mode must be str or None"):
            sim_ber(
                dummy_mc_fun,
                torch.zeros(1),
                1,
                1,
                compile_mode=123,  # Invalid type
                verbose=False,
            )


# =============================================================================
# Tests for to_list
# =============================================================================


class TestToList:
    """Tests for to_list function."""

    def test_none_returns_none(self):
        """Verify None input returns None."""
        assert to_list(None) is None

    def test_int_returns_list(self):
        """Verify integer is wrapped in list."""
        assert to_list(1) == [1]

    def test_float_returns_list(self):
        """Verify float is wrapped in list."""
        assert to_list(3.14) == [3.14]

    def test_list_returns_list(self):
        """Verify list returns same list."""
        assert to_list([1, 2, 3]) == [1, 2, 3]

    def test_string_returns_list(self):
        """Verify string is wrapped in list (not split)."""
        assert to_list("abc") == ["abc"]

    def test_numpy_array_returns_list(self):
        """Verify numpy array is converted to list."""
        result = to_list(np.array([1, 2, 3]))
        assert result == [1, 2, 3]

    def test_docstring_examples(self):
        """Verify the docstring examples work correctly."""
        assert to_list(5) == [5]
        assert to_list([1, 2, 3]) == [1, 2, 3]
        assert to_list("hello") == ["hello"]


# =============================================================================
# Tests for dict_keys_to_int
# =============================================================================


class TestDictKeysToInt:
    """Tests for dict_keys_to_int function."""

    def test_string_int_keys_converted(self):
        """Verify string integer keys are converted to int."""
        result = dict_keys_to_int({"1": "a", "2": "b"})
        assert result == {1: "a", 2: "b"}

    def test_non_int_keys_preserved(self):
        """Verify non-integer string keys are preserved."""
        result = dict_keys_to_int({"abc": 1, "4.5": 2})
        assert result == {"abc": 1, "4.5": 2}

    def test_mixed_keys(self):
        """Verify mixed keys are handled correctly."""
        result = dict_keys_to_int({"1": "a", "abc": "b", 3: "c"})
        assert result == {1: "a", "abc": "b", 3: "c"}

    def test_non_dict_returns_unchanged(self):
        """Verify non-dict input is returned unchanged."""
        assert dict_keys_to_int([1, 2, 3]) == [1, 2, 3]
        assert dict_keys_to_int("hello") == "hello"

    def test_docstring_example(self):
        """Verify the docstring example works correctly."""
        dict_in = {"1": {"2": [45, "3"]}, "4.3": 6, "d": [5, "87"]}
        result = dict_keys_to_int(dict_in)
        assert result == {1: {"2": [45, "3"]}, "4.3": 6, "d": [5, "87"]}


# =============================================================================
# Tests for scalar_to_shaped_tensor
# =============================================================================


class TestScalarToShapedTensor:
    """Tests for scalar_to_shaped_tensor function."""

    def test_from_int(self, device):
        """Verify integer scalar creates filled tensor."""
        result = scalar_to_shaped_tensor(5, torch.float32, [2, 3], device=device)
        assert result.shape == torch.Size([2, 3])
        assert torch.all(result == 5.0)
        assert result.dtype == torch.float32

    def test_from_float(self, device):
        """Verify float scalar creates filled tensor."""
        result = scalar_to_shaped_tensor(3.14, torch.float64, [3, 4], device=device)
        assert result.shape == torch.Size([3, 4])
        expected = torch.full([3, 4], 3.14, dtype=torch.float64, device=device)
        assert torch.allclose(result, expected)
        assert result.dtype == torch.float64

    def test_from_bool(self, device):
        """Verify bool scalar creates filled tensor."""
        result = scalar_to_shaped_tensor(True, torch.float32, [2, 2], device=device)
        assert result.shape == torch.Size([2, 2])
        assert torch.all(result == 1.0)

    def test_from_0d_tensor(self, device):
        """Verify 0-dimensional tensor creates filled tensor."""
        inp = torch.tensor(7.0, device=device)
        result = scalar_to_shaped_tensor(inp, torch.float32, [2, 3], device=device)
        assert result.shape == torch.Size([2, 3])
        assert torch.all(result == 7.0)

    def test_from_shaped_tensor_matching(self, device):
        """Verify shaped tensor with matching shape is cast to dtype."""
        inp = torch.ones([2, 3], dtype=torch.float32, device=device)
        result = scalar_to_shaped_tensor(inp, torch.float64, [2, 3], device=device)
        assert result.shape == torch.Size([2, 3])
        assert result.dtype == torch.float64
        assert torch.all(result == 1.0)

    def test_from_shaped_tensor_mismatched_shape_raises(self, device):
        """Verify mismatched shape raises a value error."""
        inp = torch.ones([2, 4], device=device)
        with pytest.raises(ValueError, match="Inconsistent shape"):
            scalar_to_shaped_tensor(inp, torch.float32, [2, 3], device=device)

    def test_docstring_examples(self, device):
        """Verify the docstring examples work correctly."""
        # From scalar
        t = scalar_to_shaped_tensor(5.0, torch.float32, [2, 3], device=device)
        expected = torch.tensor([[5.0, 5.0, 5.0], [5.0, 5.0, 5.0]], device=device)
        assert torch.allclose(t, expected)

        # From tensor
        t = scalar_to_shaped_tensor(
            torch.ones(2, 3, device=device), torch.float64, [2, 3], device=device
        )
        assert t.dtype == torch.float64


# =============================================================================
# Tests for DeepUpdateDict
# =============================================================================


class TestDeepUpdateDict:
    """Tests for DeepUpdateDict class."""

    def test_merge_without_conflicts(self):
        """Verify merging dicts without key conflicts."""
        dict1 = DeepUpdateDict({"a": 1, "b": {"b1": 10, "b2": 20}})
        dict_delta = {"c": -2, "b": {"b3": 30}}
        dict1.deep_update(dict_delta)
        assert dict1 == {"a": 1, "b": {"b1": 10, "b2": 20, "b3": 30}, "c": -2}

    def test_handle_key_conflicts(self):
        """Verify new values overwrite old values on conflict."""
        dict1 = DeepUpdateDict({"a": 1, "b": {"b1": 10, "b2": 20}})
        dict_delta = {"a": -2, "b": {"b1": {"f": 3, "g": 4}}}
        dict1.deep_update(dict_delta)
        assert dict1 == {"a": -2, "b": {"b1": {"f": 3, "g": 4}, "b2": 20}}

    def test_stop_at_keys(self):
        """Verify stop_at_keys replaces entire subtree."""
        dict1 = DeepUpdateDict({"a": 1, "b": {"b1": 10, "b2": 20}})
        dict_delta = {"a": -2, "b": {"b1": {"f": 3, "g": 4}}}
        dict1.deep_update(dict_delta, stop_at_keys="b")
        # At key 'b', entire subtree is replaced
        assert dict1 == {"a": -2, "b": {"b1": {"f": 3, "g": 4}}}

    def test_nested_dict_conflict(self):
        """Verify nested dict conflicts are handled correctly."""
        dict1 = DeepUpdateDict({"a": {3: 50}, "b": {"b1": 10, "b2": 20}})
        dict_delta = {"a": -2, "b": {"b1": {"f": 3, "g": 4}}}
        dict1.deep_update(dict_delta)
        assert dict1 == {"a": -2, "b": {"b1": {"f": 3, "g": 4}, "b2": 20}}

    def test_docstring_examples(self):
        """Verify the docstring examples work correctly."""
        # Example 1: Merge without conflicts
        dict1 = DeepUpdateDict({"a": 1, "b": {"b1": 10, "b2": 20}})
        dict_delta1 = {"c": -2, "b": {"b3": 30}}
        dict1.deep_update(dict_delta1)
        assert dict1 == {"a": 1, "b": {"b1": 10, "b2": 20, "b3": 30}, "c": -2}

        # Example 2: Handle key conflicts
        dict2 = DeepUpdateDict({"a": 1, "b": {"b1": 10, "b2": 20}})
        dict_delta2 = {"a": -2, "b": {"b1": {"f": 3, "g": 4}}}
        dict2.deep_update(dict_delta2)
        assert dict2 == {"a": -2, "b": {"b1": {"f": 3, "g": 4}, "b2": 20}}


# =============================================================================
# Tests for SplineGriddataInterpolation
# =============================================================================


class TestSplineGriddataInterpolation:
    """Tests for SplineGriddataInterpolation class."""

    def test_struct_basic(self):
        """Verify basic structured interpolation."""
        interp = SplineGriddataInterpolation()

        x = np.array([0.0, 1.0, 2.0, 3.0])
        y = np.array([0.0, 1.0, 2.0])
        z = np.array(
            [
                [1.0, 2.0, 3.0],
                [2.0, 3.0, 4.0],
                [3.0, 4.0, 5.0],
                [4.0, 5.0, 6.0],
            ]
        )

        x_interp = np.array([0.5, 1.5, 2.5])
        y_interp = np.array([0.5, 1.5])

        result = interp.struct(z, x, y, x_interp, y_interp, spline_degree=1)

        assert result.shape == (3, 2)
        # Values should be interpolated
        assert np.all(result > 0)

    def test_struct_too_few_points_raises(self):
        """Verify error when too few points for interpolation."""
        interp = SplineGriddataInterpolation()

        x = np.array([0.0])  # Only 1 point, need at least 2 for degree=1
        y = np.array([0.0, 1.0])
        z = np.array([[1.0, 2.0]])

        with pytest.raises(ValueError, match="Too few points"):
            interp.struct(z, x, y, x, y, spline_degree=1)

    def test_struct_all_zero(self):
        """An all-zero BLER table interpolates to exact zeros."""
        interp = SplineGriddataInterpolation()
        x = np.array([0.0, 1.0])
        y = np.array([0.0, 1.0])
        result = interp.struct(
            np.zeros((2, 2)),
            x,
            y,
            np.array([0.0, 0.5, 1.0]),
            np.array([0.0, 0.5, 1.0]),
        )
        assert result.shape == (3, 3)
        assert np.array_equal(result, np.zeros((3, 3)))

    def test_struct_near_all_zero(self):
        """A table with one nonzero sample still uses spline interpolation."""
        interp = SplineGriddataInterpolation()
        x = np.array([0.0, 1.0])
        y = np.array([0.0, 1.0])
        z = np.array([[0.0, 0.0], [0.0, 0.5]])
        result = interp.struct(z, x, y, x, y)
        assert np.all(np.isfinite(result))
        assert np.isclose(result[-1, -1], 0.5)

    def test_unstruct_basic(self):
        """Verify basic unstructured interpolation."""
        interp = SplineGriddataInterpolation()

        # Scattered points
        x = np.array([0.0, 1.0, 0.0, 1.0])
        y = np.array([0.0, 0.0, 1.0, 1.0])
        z = np.array([1.0, 2.0, 3.0, 4.0])

        x_interp = np.array([0.0, 0.5, 1.0])
        y_interp = np.array([0.0, 0.5, 1.0])

        result = interp.unstruct(z, x, y, x_interp, y_interp, griddata_method="linear")

        assert result.shape == (3, 3)


# =============================================================================
# Tests for SingleLinkChannel
# =============================================================================


class TestSingleLinkChannel:
    """Tests for SingleLinkChannel class."""

    def test_properties_initialization(self, device):
        """Verify properties are correctly initialized."""

        # Create a concrete subclass for testing
        class ConcreteSingleLinkChannel(SingleLinkChannel):
            def call(self, batch_size, ebno_db):
                return torch.zeros(batch_size), torch.zeros(batch_size)

        channel = ConcreteSingleLinkChannel(
            num_bits_per_symbol=4,
            num_info_bits=1024,
            target_coderate=0.5,
            device=device,
        )

        assert channel.num_bits_per_symbol == 4
        assert channel.num_info_bits == 1024
        assert channel.target_coderate == 0.5

    def test_num_coded_bits_calculation(self, device):
        """Verify num_coded_bits is correctly calculated."""

        class ConcreteSingleLinkChannel(SingleLinkChannel):
            def call(self, batch_size, ebno_db):
                return torch.zeros(batch_size), torch.zeros(batch_size)

        channel = ConcreteSingleLinkChannel(
            num_bits_per_symbol=4,
            num_info_bits=1024,
            target_coderate=0.5,
            device=device,
        )

        # 1024 / 0.5 = 2048, which is a multiple of 4
        assert channel.num_coded_bits == 2048

    def test_num_coded_bits_rounded_up(self, device):
        """Verify num_coded_bits is rounded up to multiple of bits_per_symbol."""

        class ConcreteSingleLinkChannel(SingleLinkChannel):
            def call(self, batch_size, ebno_db):
                return torch.zeros(batch_size), torch.zeros(batch_size)

        channel = ConcreteSingleLinkChannel(
            num_bits_per_symbol=6,
            num_info_bits=100,
            target_coderate=0.5,
            device=device,
        )

        # 100 / 0.5 = 200, ceil(200/6)*6 = 204
        assert channel.num_coded_bits == 204

    def test_invalid_num_bits_per_symbol(self, device):
        """Verify value error for non-positive num_bits_per_symbol."""

        class ConcreteSingleLinkChannel(SingleLinkChannel):
            def call(self, batch_size, ebno_db):
                return torch.zeros(batch_size), torch.zeros(batch_size)

        with pytest.raises(ValueError, match="positive integer"):
            ConcreteSingleLinkChannel(
                num_bits_per_symbol=0,
                num_info_bits=1024,
                target_coderate=0.5,
                device=device,
            )

    def test_invalid_target_coderate(self, device):
        """Verify an explicit error for out-of-range target_coderate."""

        class ConcreteSingleLinkChannel(SingleLinkChannel):
            def call(self, batch_size, ebno_db):
                return torch.zeros(batch_size), torch.zeros(batch_size)

        with pytest.raises(ValueError, match="within"):
            ConcreteSingleLinkChannel(
                num_bits_per_symbol=4,
                num_info_bits=1024,
                target_coderate=1.5,  # Invalid: > 1
                device=device,
            )

    @pytest.mark.parametrize("value", [0.0, -0.1, 1.1, np.nan, np.inf])
    def test_target_coderate_boundaries_preserve_state(self, value, device):
        """Rejected updates do not leave the channel in an invalid state."""

        class ConcreteSingleLinkChannel(SingleLinkChannel):
            def call(self, batch_size, ebno_db):
                return torch.zeros(batch_size), torch.zeros(batch_size)

        channel = ConcreteSingleLinkChannel(
            num_bits_per_symbol=4,
            num_info_bits=100,
            target_coderate=0.5,
            device=device,
        )
        with pytest.raises(ValueError, match="within"):
            channel.target_coderate = value
        assert channel.target_coderate == 0.5
        assert channel.num_coded_bits == 200

    def test_target_coderate_valid_endpoints(self, device):
        """The upper endpoint and a small positive coderate are valid."""

        class ConcreteSingleLinkChannel(SingleLinkChannel):
            def call(self, batch_size, ebno_db):
                return torch.zeros(batch_size), torch.zeros(batch_size)

        channel = ConcreteSingleLinkChannel(4, 100, 1.0, device=device)
        assert channel.num_coded_bits == 100
        channel.target_coderate = 0.01
        assert channel.num_coded_bits == 10_000

    def test_property_setters_update_num_coded_bits(self, device):
        """Verify changing properties updates num_coded_bits."""

        class ConcreteSingleLinkChannel(SingleLinkChannel):
            def call(self, batch_size, ebno_db):
                return torch.zeros(batch_size), torch.zeros(batch_size)

        channel = ConcreteSingleLinkChannel(
            num_bits_per_symbol=4,
            num_info_bits=1024,
            target_coderate=0.5,
            device=device,
        )

        assert channel.num_coded_bits == 2048

        # Update coderate
        channel.target_coderate = 0.25
        # 1024 / 0.25 = 4096
        assert channel.num_coded_bits == 4096
