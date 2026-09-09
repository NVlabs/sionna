#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Unit tests for optical channel components (SSFM and EDFA)."""

import numpy as np
import pytest
import torch

from sionna.phy import H
from sionna.phy.channel import utils
from sionna.phy.channel.optical import SSFM, EDFA


# Reference data for SSFM validation (from numerical simulation)
# This is the expected output magnitude for a Gaussian pulse propagating
# through a fiber with dispersion and nonlinearity
U_REF = np.array(
    [
        4.7173e-08,
        4.4381e-08,
        3.8966e-08,
        3.1258e-08,
        2.174e-08,
        1.1112e-08,
        3.435e-09,
        1.2823e-08,
        2.3379e-08,
        3.2709e-08,
        4.0165e-08,
        4.5286e-08,
        4.7768e-08,
        4.7472e-08,
        4.4436e-08,
        3.888e-08,
        3.1224e-08,
        2.2178e-08,
        1.3321e-08,
        1.0368e-08,
        1.7341e-08,
        2.6729e-08,
        3.5422e-08,
        4.2403e-08,
        4.7121e-08,
        4.9256e-08,
        4.8689e-08,
        4.5498e-08,
        3.9978e-08,
        3.27e-08,
        2.4708e-08,
        1.8249e-08,
        1.749e-08,
        2.3231e-08,
        3.137e-08,
        3.924e-08,
        4.5625e-08,
        4.988e-08,
        5.1662e-08,
        5.0867e-08,
        4.7624e-08,
        4.2319e-08,
        3.5683e-08,
        2.9021e-08,
        2.4584e-08,
        2.4933e-08,
        2.994e-08,
        3.7027e-08,
        4.4061e-08,
        4.9811e-08,
        5.3587e-08,
        5.5033e-08,
        5.407e-08,
        5.0879e-08,
        4.5938e-08,
        4.0108e-08,
        3.4775e-08,
        3.1848e-08,
        3.2834e-08,
        3.7348e-08,
        4.3587e-08,
        4.9847e-08,
        5.4978e-08,
        5.8291e-08,
        5.9441e-08,
        5.8377e-08,
        5.5335e-08,
        5.0867e-08,
        4.59e-08,
        4.1762e-08,
        3.9947e-08,
        4.1339e-08,
        4.5491e-08,
        5.1049e-08,
        5.6623e-08,
        6.1181e-08,
        6.4073e-08,
        6.4979e-08,
        6.3884e-08,
        6.1071e-08,
        5.7143e-08,
        5.3029e-08,
        4.9915e-08,
        4.8938e-08,
        5.0607e-08,
        5.4481e-08,
        5.9483e-08,
        6.4465e-08,
        6.8516e-08,
        7.1043e-08,
    ]
)


class TestEDFA:
    """Tests for EDFA (Erbium-Doped Fiber Amplifier) class."""

    def test_edfa_gain_batch(self, precision):
        """Verify that EDFA correctly applies gain to input signal."""
        F = 0
        G = 4.0
        f_c = 193.55e12
        dt = 1e-12
        amplifier = EDFA(G, F, f_c, dt, False, precision=precision)
        shape = (10, 10, 10000)

        dtype = torch.float64 if precision == "double" else torch.float32

        x = torch.complex(
            (1.0 / np.sqrt(2.0)) * torch.ones(shape, dtype=dtype),
            (1.0 / np.sqrt(2.0)) * torch.ones(shape, dtype=dtype),
        )
        y = amplifier(x)
        p = np.mean(np.mean(np.abs(y.cpu().numpy()) ** 2.0, axis=-1))
        assert np.abs(G - p) <= 1e-5, "Incorrect EDFA gain in batch processing"

    def test_edfa_gain_batch_dual_polarized(self, precision):
        """Verify EDFA gain with dual polarization."""
        F = 0
        G = 4.0
        f_c = 193.55e12
        dt = 1e-12
        amplifier = EDFA(G, F, f_c, dt, True, precision=precision)
        shape = (100, 2, 10000)

        dtype = torch.float64 if precision == "double" else torch.float32

        x = np.sqrt(1.0 / 2.0) * torch.complex(
            (1.0 / np.sqrt(2.0)) * torch.ones(shape, dtype=dtype),
            (1.0 / np.sqrt(2.0)) * torch.ones(shape, dtype=dtype),
        )
        y = amplifier(x)
        p = np.mean(np.mean(np.sum(np.abs(y.cpu().numpy()) ** 2.0, axis=-2), axis=-1))
        assert np.abs(G - p) <= 1e-5, "Incorrect EDFA gain for dual polarization"

    def test_edfa_noise(self, precision):
        """Verify EDFA adds correct ASE noise power."""
        cdtype = torch.complex128 if precision == "double" else torch.complex64

        F = 10 ** (6 / 10)
        G = 2.0
        f_c = 193.55e12
        dt = 1e-12
        n_sp = F / 2.0 * G / (G - 1.0)
        rho_n_ASE = n_sp * (G - 1.0) * H * f_c
        P_n_ASE = 2.0 * rho_n_ASE * (1.0 / dt)

        amplifier = EDFA(G, F, f_c, dt, False, precision=precision)
        x = torch.zeros((100, 10, 1000), dtype=cdtype)
        y = amplifier(x)
        sigma_n_ASE_square = np.mean(np.var(y.cpu().numpy(), axis=-1))

        assert (
            np.abs((P_n_ASE - sigma_n_ASE_square) / P_n_ASE) <= 1e-2
        ), "Incorrect EDFA noise"

    @pytest.mark.parametrize("compile_mode", ["default", None])
    def test_edfa_compiled(self, precision, compile_mode):
        """Test EDFA with torch.compile."""
        if compile_mode is None:
            pytest.skip("Skipping non-compiled test (already covered)")

        dtype = torch.float64 if precision == "double" else torch.float32

        F = 0.0
        G = 4.0
        f_c = 193.55e12
        dt = 1e-12
        amplifier = EDFA(G, F, f_c, dt, False, precision=precision)

        if compile_mode is not None:
            try:
                amplifier_fn = torch.compile(amplifier, mode=compile_mode)
            except (RuntimeError, TypeError):
                pytest.skip("torch.compile not available")
        else:
            amplifier_fn = amplifier

        shape = (10, 10, 1000)
        x = torch.complex(
            (1.0 / np.sqrt(2.0)) * torch.ones(shape, dtype=dtype),
            (1.0 / np.sqrt(2.0)) * torch.ones(shape, dtype=dtype),
        )
        y = amplifier_fn(x)
        p = np.mean(np.mean(np.abs(y.cpu().numpy()) ** 2.0, axis=-1))
        assert np.abs(G - p) <= 1e-5, "Incorrect EDFA gain in compiled mode"


class TestSSFM:
    """Tests for SSFM (Split-Step Fourier Method) class."""

    @pytest.fixture
    def gaussian_pulse_setup(self, precision):
        """Create Gaussian pulse for SSFM testing."""
        dtype = torch.float64 if precision == "double" else torch.float32
        cdtype = torch.complex128 if precision == "double" else torch.complex64

        T = 100  # time window (period)
        N = 2**12  # number of points
        dt = T / N  # timestep
        t, _ = utils.time_frequency_vector(N, dt, precision=precision)

        P_0 = 1.0  # Peak power
        T_0 = 1.0  # Temporal scaling

        u_0 = np.sqrt(P_0) * np.exp(-((t.cpu().numpy() / T_0) ** 2.0) / 2.0)
        u_0 = torch.complex(
            torch.tensor(u_0, dtype=dtype),
            torch.zeros_like(torch.tensor(u_0, dtype=dtype)),
        )

        return {
            "u_0": u_0,
            "dt": dt,
            "precision": precision,
            "dtype": dtype,
            "cdtype": cdtype,
        }

    def test_ssfm_reference(self, precision):
        """Verify SSFM output matches reference for Gaussian pulse."""
        dtype = torch.float64 if precision == "double" else torch.float32

        T = 100
        N = 2**12
        dt = T / N
        t, _ = utils.time_frequency_vector(N, dt, precision=precision)

        P_0 = 1.0
        T_0 = 1.0

        u_0 = np.sqrt(P_0) * np.exp(-((t.cpu().numpy() / T_0) ** 2.0) / 2.0)
        u_0 = torch.complex(
            torch.tensor(u_0, dtype=dtype),
            torch.zeros(N, dtype=dtype),
        )

        ssfm = SSFM(
            alpha=0.0,
            beta_2=1.0,
            f_c=193.55e12,
            gamma=1.0,
            half_window_length=0,
            length=5.0,
            n_ssfm=2000,
            sample_duration=dt,
            with_amplification=False,
            with_attenuation=False,
            with_dispersion=True,
            with_nonlinearity=True,
            precision=precision,
            t_norm=1e-12,
            n_sp=1.0,
        )

        u = ssfm(u_0)

        # Compare with reference (only first 90 points for brevity)
        u_out = np.abs(u.cpu().numpy().flatten())[:90]
        assert np.mean((u_out - U_REF[:90]) ** 2) <= 1e-7, "Incorrect SSFM output"

    def test_adaptive_ssfm_reference(self, precision):
        """Verify adaptive SSFM output matches reference."""
        dtype = torch.float64 if precision == "double" else torch.float32

        T = 100
        N = 2**12
        dt = T / N
        t, _ = utils.time_frequency_vector(N, dt, precision=precision)

        P_0 = 1.0
        T_0 = 1.0

        u_0 = np.sqrt(P_0) * np.exp(-((t.cpu().numpy() / T_0) ** 2.0) / 2.0)
        u_0 = torch.complex(
            torch.tensor(u_0, dtype=dtype),
            torch.zeros(N, dtype=dtype),
        )

        ssfm = SSFM(
            alpha=0.0,
            beta_2=1.0,
            f_c=193.55e12,
            gamma=1.0,
            half_window_length=0,
            length=5.0,
            n_ssfm="adaptive",
            sample_duration=dt,
            with_amplification=False,
            with_attenuation=False,
            with_dispersion=True,
            with_nonlinearity=True,
            precision=precision,
            t_norm=1e-12,
            n_sp=1.0,
            phase_inc=1e-3,
        )

        u = ssfm(u_0)

        u_out = np.abs(u.cpu().numpy().flatten())[:90]
        assert np.mean((u_out - U_REF[:90]) ** 2) <= 1e-7, "Incorrect adaptive SSFM"

    def test_ssfm_batch(self, precision):
        """Verify SSFM works correctly with batched inputs."""
        dtype = torch.float64 if precision == "double" else torch.float32

        T = 100
        N = 2**12
        dt = T / N
        t, _ = utils.time_frequency_vector(N, dt, precision=precision)

        P_0 = 1.0
        T_0 = 1.0

        u_0 = np.sqrt(P_0) * np.exp(-((t.cpu().numpy() / T_0) ** 2.0) / 2.0)
        u_0 = torch.complex(
            torch.tensor(u_0, dtype=dtype),
            torch.zeros(N, dtype=dtype),
        )

        # Batch the input: [2, 2, N]
        u_0 = u_0.unsqueeze(0).unsqueeze(0).expand(2, 2, -1).clone()

        ssfm = SSFM(
            alpha=0.0,
            beta_2=1.0,
            f_c=193.55e12,
            gamma=1.0,
            half_window_length=0,
            length=5.0,
            n_ssfm=2000,
            sample_duration=dt,
            with_amplification=True,
            with_attenuation=False,
            with_dispersion=True,
            with_nonlinearity=True,
            precision=precision,
            t_norm=1e-12,
            n_sp=0.0,
        )

        u = ssfm(u_0)

        # Compare each batch element
        u_ref = np.tile(U_REF[:90], (2, 2, 1))
        u_out = np.abs(u.cpu().numpy())[..., :90]
        assert np.mean((u_out - u_ref) ** 2) <= 1e-7, "Incorrect SSFM batch processing"

    def test_ssfm_dual_polarized(self, precision):
        """Verify SSFM Manakov equation for dual polarization."""
        dtype = torch.float64 if precision == "double" else torch.float32

        T = 100
        N = 2**12
        dt = T / N
        t, _ = utils.time_frequency_vector(N, dt, precision=precision)

        # Distribute power uniformly over both polarizations
        # Account for 8/9 factor of Manakov equation
        P_0 = 9 / 8 * 0.5
        T_0 = 1.0

        u_0 = np.sqrt(P_0) * np.exp(-((t.cpu().numpy() / T_0) ** 2.0) / 2.0)
        u_0 = torch.complex(
            torch.tensor(u_0, dtype=dtype),
            torch.zeros(N, dtype=dtype),
        )

        # Shape: [3, 2, N] where 2 is polarization dimension
        u_0 = u_0.unsqueeze(0).unsqueeze(0).expand(3, 2, -1).clone()

        ssfm = SSFM(
            alpha=0.0,
            beta_2=1.0,
            f_c=193.55e12,
            gamma=1.0,
            half_window_length=0,
            length=5.0,
            n_ssfm=2000,
            sample_duration=dt,
            with_amplification=False,
            with_attenuation=False,
            with_dispersion=True,
            with_nonlinearity=True,
            with_manakov=True,
            precision=precision,
            t_norm=1e-12,
            n_sp=0.0,
        )

        # Remove Manakov factor and geometrically add both polarizations
        u = ssfm(u_0) * np.sqrt(8 / 9)
        u = torch.norm(u, dim=-2, keepdim=True)

        u_ref = np.tile(U_REF[:90], (3, 1, 1))
        u_out = np.abs(u.cpu().numpy())[..., :90]
        assert (
            np.mean((u_out - u_ref) ** 2) <= 1e-7
        ), "Incorrect SSFM dual polarized batch processing"

    def test_ssfm_amplification(self, precision):
        """Verify SSFM with Raman amplification compensates attenuation."""
        dtype = torch.float64 if precision == "double" else torch.float32

        T = 100
        N = 2**12
        dt = T / N
        t, _ = utils.time_frequency_vector(N, dt, precision=precision)

        P_0 = 1.0
        T_0 = 1.0

        u_0 = np.sqrt(P_0) * np.exp(-((t.cpu().numpy() / T_0) ** 2.0) / 2.0)
        u_0 = torch.complex(
            torch.tensor(u_0, dtype=dtype),
            torch.zeros(N, dtype=dtype),
        )

        ssfm = SSFM(
            alpha=0.046,
            beta_2=1.0,
            f_c=193.55e12,
            gamma=1.0,
            half_window_length=0,
            length=5.0,
            n_ssfm=2000,
            sample_duration=dt,
            with_amplification=True,
            with_attenuation=True,
            with_dispersion=False,
            with_nonlinearity=False,
            precision=precision,
            t_norm=1e-12,
            n_sp=0.0,
        )

        u = ssfm(u_0)

        # With amplification compensating attenuation, output should equal input
        # Use precision-dependent tolerance (float32 has ~7 decimal digits)
        tol = 1e-7 if precision == "single" else 1e-10
        assert (
            np.mean((np.abs(u.cpu().numpy()) - np.abs(u_0.cpu().numpy())) ** 2) <= tol
        ), "Incorrect SSFM amplification"

    def test_ssfm_amplification_noise(self, precision):
        """Verify SSFM Raman amplification adds correct noise power."""
        cdtype = torch.complex128 if precision == "double" else torch.complex64

        T = 100
        N = 2**12
        dt = T / N
        n_sp = 1.0
        f_c = 193.55e12
        alpha = 0.046
        length = 5.0
        t_norm = 1e-12

        rho_n = H * f_c * alpha * length * n_sp
        p_n_ase = rho_n / dt / t_norm

        ssfm = SSFM(
            alpha=alpha,
            beta_2=1.0,
            f_c=f_c,
            gamma=1.0,
            half_window_length=0,
            length=5.0,
            n_ssfm=2000,
            sample_duration=dt,
            with_amplification=True,
            with_attenuation=True,
            with_dispersion=False,
            with_nonlinearity=False,
            precision=precision,
            t_norm=t_norm,
            n_sp=n_sp,
        )

        u_0 = torch.zeros((100, 10, 2000), dtype=cdtype)
        u = ssfm(u_0)

        sigma_n_ASE_square = np.mean(np.var(u.cpu().numpy(), axis=-1))
        assert (
            np.abs((p_n_ase - sigma_n_ASE_square) / p_n_ase) <= 1e-2
        ), "Incorrect SSFM amplification noise"

    def test_ssfm_amplification_noise_dual_polarized(self, precision):
        """Verify SSFM noise power for dual polarization."""
        cdtype = torch.complex128 if precision == "double" else torch.complex64

        T = 100
        N = 2**12
        dt = T / N
        n_sp = 1.0
        f_c = 193.55e12
        alpha = 0.046
        length = 5.0
        t_norm = 1e-12

        rho_n = H * f_c * alpha * length * n_sp
        p_n_ase = rho_n / dt / t_norm

        ssfm = SSFM(
            alpha=alpha,
            beta_2=1.0,
            f_c=f_c,
            gamma=1.0,
            half_window_length=0,
            length=5.0,
            n_ssfm=2000,
            sample_duration=dt,
            with_amplification=True,
            with_attenuation=True,
            with_dispersion=False,
            with_nonlinearity=False,
            with_manakov=True,
            precision=precision,
            t_norm=t_norm,
            n_sp=n_sp,
        )

        u_0 = torch.zeros((100, 2, 2000), dtype=cdtype)
        u = ssfm(u_0)

        sigma_n_ASE_square = np.mean(np.var(u.cpu().numpy(), axis=-1))
        # For dual polarization, noise is split between polarizations
        assert (
            np.abs((0.5 * p_n_ase - sigma_n_ASE_square) / (0.5 * p_n_ase)) <= 1e-2
        ), "Incorrect SSFM amplification noise for dual polarization"

    @pytest.mark.parametrize("compile_mode", ["default", None])
    def test_ssfm_compiled(self, precision, compile_mode):
        """Test SSFM with torch.compile."""
        if compile_mode is None:
            pytest.skip("Skipping non-compiled test (already covered)")

        dtype = torch.float64 if precision == "double" else torch.float32

        T = 100
        N = 2**10  # Smaller for faster compilation
        dt = T / N
        t, _ = utils.time_frequency_vector(N, dt, precision=precision)

        P_0 = 1.0
        T_0 = 1.0

        u_0 = np.sqrt(P_0) * np.exp(-((t.cpu().numpy() / T_0) ** 2.0) / 2.0)
        u_0 = torch.complex(
            torch.tensor(u_0, dtype=dtype),
            torch.zeros(N, dtype=dtype),
        )

        ssfm = SSFM(
            alpha=0.0,
            beta_2=1.0,
            f_c=193.55e12,
            gamma=1.0,
            half_window_length=0,
            length=5.0,
            n_ssfm=100,
            sample_duration=dt,
            with_amplification=False,
            with_attenuation=False,
            with_dispersion=True,
            with_nonlinearity=True,
            precision=precision,
            t_norm=1e-12,
            n_sp=1.0,
        )

        try:
            ssfm_fn = torch.compile(ssfm, mode=compile_mode)
            u = ssfm_fn(u_0)
            # Just verify it runs without error
            assert u.shape == u_0.shape
        except (RuntimeError, TypeError) as e:
            pytest.skip(f"torch.compile not available: {e}")

    def test_ssfm_compiled_has_no_graph_breaks(self, precision):
        """Check that SSFM captures as a single graph.

        Reading a scalar constant back out of a buffer with ``.item()`` inside
        ``call()`` splits the graph at every invocation, so the frequency-vector
        setup must not depend on a tensor-to-Python conversion.
        """
        if not hasattr(torch, "compile"):
            pytest.skip("torch.compile is not available")

        import torch._dynamo as dynamo
        from torch._dynamo.utils import counters

        dtype = torch.float64 if precision == "double" else torch.float32
        n = 2**8
        dt = 100.0 / n

        ssfm = SSFM(
            alpha=0.0,
            beta_2=1.0,
            f_c=193.55e12,
            gamma=1.0,
            half_window_length=0,
            length=5.0,
            n_ssfm=4,
            sample_duration=dt,
            with_amplification=False,
            with_attenuation=False,
            with_dispersion=True,
            with_nonlinearity=True,
            precision=precision,
            t_norm=1e-12,
            n_sp=1.0,
        )

        x = torch.complex(torch.zeros(n, dtype=dtype), torch.zeros(n, dtype=dtype))
        x[n // 2] = 1.0

        dynamo.reset()
        counters.clear()
        try:
            u = torch.compile(ssfm)(x)
        except (RuntimeError, TypeError) as e:
            pytest.skip(f"torch.compile not available: {e}")

        assert u.shape == x.shape
        assert not dict(counters.get("graph_break", {}))

    def test_cached_sample_duration_matches_buffer(self, precision):
        """Check the cached sample duration agrees with the buffer.

        ``call()`` passes the cached scalar to ``time_frequency_vector()``,
        which derives the frequency spacing from it in double arithmetic. A
        cache taken from the constructor argument rather than from the buffer
        therefore shifts the frequency grid at single precision, where the
        buffer holds a rounded value. ``0.337095`` is not exactly representable
        in `float32`, so the two disagree unless the cache is read off the
        buffer.
        """
        ssfm = SSFM(sample_duration=0.337095, precision=precision)
        assert ssfm._sample_duration_value == ssfm._sample_duration.item()

    def test_derived_state_updated_after_state_load(self, precision):
        """Check state loading refreshes values derived from buffers."""
        source = torch.nn.Sequential(
            SSFM(
                alpha=0.03,
                length=5.0,
                n_sp=2.0,
                sample_duration=0.337095,
                t_norm=2e-12,
                n_ssfm=1,
                with_attenuation=False,
                with_nonlinearity=False,
                precision=precision,
            )
        )
        target = torch.nn.Sequential(
            SSFM(
                alpha=0.07,
                length=8.0,
                n_sp=3.0,
                sample_duration=0.5,
                t_norm=1e-12,
                n_ssfm=1,
                with_attenuation=False,
                with_nonlinearity=False,
                precision=precision,
            )
        )

        target.load_state_dict(source.state_dict())

        source_ssfm = source[0]
        target_ssfm = target[0]
        assert (
            getattr(target_ssfm, "_sample_duration_value")
            == getattr(source_ssfm, "_sample_duration_value")
        )
        for name in ("_dz", "_rho_n", "_p_n_ase"):
            torch.testing.assert_close(
                getattr(target_ssfm, name),
                getattr(source_ssfm, name),
                rtol=0.0,
                atol=0.0,
            )

        dtype = torch.float64 if precision == "double" else torch.float32
        x = torch.complex(torch.zeros(256, dtype=dtype), torch.zeros(256, dtype=dtype))
        x[128] = 1.0
        torch.testing.assert_close(target(x), source(x), rtol=0.0, atol=0.0)


class TestDocstringExamples:
    """Test that docstring examples work correctly."""

    def test_edfa_example(self):
        """Verify EDFA docstring example produces expected output shape."""
        edfa = EDFA(
            g=4.0,
            f=2.0,
            f_c=193.55e12,
            dt=1.0e-12,
            with_dual_polarization=False,
        )

        x = torch.randn(10, 100, dtype=torch.complex64)
        y = edfa(x)
        assert y.shape == torch.Size([10, 100])

    def test_ssfm_example(self):
        """Verify SSFM docstring example produces expected output shape."""
        ssfm = SSFM(
            alpha=0.046,
            beta_2=-21.67,
            f_c=193.55e12,
            gamma=1.27,
            half_window_length=100,
            length=80,
            n_ssfm=200,
            n_sp=1.0,
            t_norm=1e-12,
            with_amplification=False,
            with_attenuation=True,
            with_dispersion=True,
            with_manakov=False,
            with_nonlinearity=True,
        )

        x = torch.randn(10, 1024, dtype=torch.complex64)
        y = ssfm(x)
        assert y.shape == torch.Size([10, 1024])
