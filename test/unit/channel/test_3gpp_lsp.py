#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Tests for 3GPP TR 38.901 LSP classes"""

import numpy as np
import pytest
import torch
from scipy.stats import kstest, norm

from sionna.phy.channel import tr38901
from sionna.phy.channel.tr38901 import LSP, LSPGenerator

from channel_test_utils import (
    corr_dist_asd,
    corr_dist_asa,
    corr_dist_ds,
    corr_dist_k,
    corr_dist_sf,
    corr_dist_zsa,
    corr_dist_zsd,
    cross_corr,
    generate_random_bool,
    generate_random_loc,
    limited_normal,
    log10ASD,
    log10ASA,
    log10DS,
    log10K_dB,
    log10SF_dB,
    log10ZSA,
    log10ZSD,
    los_probability,
    zod_offset,
)


def _winner_spatial_correlation(
    cross_correlation, internal_names, correlation_by_name
):
    """Return output correlations for the WINNER II separated procedure."""
    standard_names = [
        name
        for name in ("sf", "k", "ds", "asd", "asa", "zsd", "zsa")
        if name in internal_names
    ]
    standard_indices = [internal_names.index(name) for name in standard_names]
    correlation = cross_correlation[np.ix_(standard_indices, standard_indices)]
    factor = np.linalg.cholesky(correlation)
    output = {}
    for row, name in enumerate(standard_names):
        output[name] = sum(
            factor[row, column] ** 2 * correlation_by_name[latent_name]
            for column, latent_name in enumerate(standard_names)
        )
    return output


class TestLSP:
    """Tests for the LSP data class"""

    def test_lsp_instantiation(self, device, precision):
        """Test that LSP can be instantiated with the expected shapes"""
        batch_size = 10
        num_tx = 3
        num_rx = 5

        # Create random tensors with expected shapes
        ds = torch.rand(batch_size, num_tx, num_rx, device=device)
        asd = torch.rand(batch_size, num_tx, num_rx, device=device)
        asa = torch.rand(batch_size, num_tx, num_rx, device=device)
        sf = torch.rand(batch_size, num_tx, num_rx, device=device)
        k_factor = torch.rand(batch_size, num_tx, num_rx, device=device)
        zsa = torch.rand(batch_size, num_tx, num_rx, device=device)
        zsd = torch.rand(batch_size, num_tx, num_rx, device=device)

        lsp = LSP(
            ds=ds,
            asd=asd,
            asa=asa,
            sf=sf,
            k_factor=k_factor,
            zsa=zsa,
            zsd=zsd,
        )

        # Verify all attributes are correctly assigned
        assert torch.equal(lsp.ds, ds)
        assert torch.equal(lsp.asd, asd)
        assert torch.equal(lsp.asa, asa)
        assert torch.equal(lsp.sf, sf)
        assert torch.equal(lsp.k_factor, k_factor)
        assert torch.equal(lsp.zsa, zsa)
        assert torch.equal(lsp.zsd, zsd)

    def test_lsp_ds_positive(self, device, precision):
        """Test that delay spread is positive"""
        batch_size = 5
        num_tx = 2
        num_rx = 3

        # Create positive values
        ds = torch.abs(torch.randn(batch_size, num_tx, num_rx, device=device)) + 1e-9
        asd = torch.rand(batch_size, num_tx, num_rx, device=device)
        asa = torch.rand(batch_size, num_tx, num_rx, device=device)
        sf = torch.rand(batch_size, num_tx, num_rx, device=device)
        k_factor = torch.rand(batch_size, num_tx, num_rx, device=device)
        zsa = torch.rand(batch_size, num_tx, num_rx, device=device)
        zsd = torch.rand(batch_size, num_tx, num_rx, device=device)

        lsp = LSP(
            ds=ds,
            asd=asd,
            asa=asa,
            sf=sf,
            k_factor=k_factor,
            zsa=zsa,
            zsd=zsd,
        )

        assert torch.all(lsp.ds > 0)

    def test_lsp_angle_spreads_bounded(self, device, precision):
        """Test that angle spreads are bounded (degrees)"""
        batch_size = 5
        num_tx = 2
        num_rx = 3

        # Create bounded angle spread values (0 to 104 degrees for azimuth, 0 to 52 for zenith)
        ds = torch.rand(batch_size, num_tx, num_rx, device=device) * 1e-6
        asd = torch.rand(batch_size, num_tx, num_rx, device=device) * 104
        asa = torch.rand(batch_size, num_tx, num_rx, device=device) * 104
        sf = torch.rand(batch_size, num_tx, num_rx, device=device)
        k_factor = torch.rand(batch_size, num_tx, num_rx, device=device)
        zsa = torch.rand(batch_size, num_tx, num_rx, device=device) * 52
        zsd = torch.rand(batch_size, num_tx, num_rx, device=device) * 52

        lsp = LSP(
            ds=ds,
            asd=asd,
            asa=asa,
            sf=sf,
            k_factor=k_factor,
            zsa=zsa,
            zsd=zsd,
        )

        # Check azimuth angle spreads are bounded by 104 degrees
        assert torch.all(lsp.asd <= 104)
        assert torch.all(lsp.asa <= 104)

        # Check zenith angle spreads are bounded by 52 degrees
        assert torch.all(lsp.zsa <= 52)
        assert torch.all(lsp.zsd <= 52)

    def test_lsp_k_factor_positive(self, device, precision):
        """Test that K-factor is positive (linear scale)"""
        batch_size = 5
        num_tx = 2
        num_rx = 3

        ds = torch.rand(batch_size, num_tx, num_rx, device=device) * 1e-6
        asd = torch.rand(batch_size, num_tx, num_rx, device=device) * 50
        asa = torch.rand(batch_size, num_tx, num_rx, device=device) * 50
        sf = torch.rand(batch_size, num_tx, num_rx, device=device)
        # K-factor should be positive (linear scale)
        k_factor = torch.abs(torch.randn(batch_size, num_tx, num_rx, device=device)) + 0.1
        zsa = torch.rand(batch_size, num_tx, num_rx, device=device) * 30
        zsd = torch.rand(batch_size, num_tx, num_rx, device=device) * 30

        lsp = LSP(
            ds=ds,
            asd=asd,
            asa=asa,
            sf=sf,
            k_factor=k_factor,
            zsa=zsa,
            zsd=zsd,
        )

        assert torch.all(lsp.k_factor > 0)

    def test_lsp_sf_positive(self, device, precision):
        """Test that shadow fading is positive (linear scale)"""
        batch_size = 5
        num_tx = 2
        num_rx = 3

        ds = torch.rand(batch_size, num_tx, num_rx, device=device) * 1e-6
        asd = torch.rand(batch_size, num_tx, num_rx, device=device) * 50
        asa = torch.rand(batch_size, num_tx, num_rx, device=device) * 50
        # SF should be positive (linear scale)
        sf = torch.abs(torch.randn(batch_size, num_tx, num_rx, device=device)) + 0.1
        k_factor = torch.rand(batch_size, num_tx, num_rx, device=device)
        zsa = torch.rand(batch_size, num_tx, num_rx, device=device) * 30
        zsd = torch.rand(batch_size, num_tx, num_rx, device=device) * 30

        lsp = LSP(
            ds=ds,
            asd=asd,
            asa=asa,
            sf=sf,
            k_factor=k_factor,
            zsa=zsa,
            zsd=zsd,
        )

        assert torch.all(lsp.sf > 0)

    def test_lsp_shapes_consistent(self, device, precision):
        """Test that all LSP attributes have consistent shapes"""
        batch_size = 8
        num_tx = 4
        num_rx = 6

        ds = torch.rand(batch_size, num_tx, num_rx, device=device)
        asd = torch.rand(batch_size, num_tx, num_rx, device=device)
        asa = torch.rand(batch_size, num_tx, num_rx, device=device)
        sf = torch.rand(batch_size, num_tx, num_rx, device=device)
        k_factor = torch.rand(batch_size, num_tx, num_rx, device=device)
        zsa = torch.rand(batch_size, num_tx, num_rx, device=device)
        zsd = torch.rand(batch_size, num_tx, num_rx, device=device)

        lsp = LSP(
            ds=ds,
            asd=asd,
            asa=asa,
            sf=sf,
            k_factor=k_factor,
            zsa=zsa,
            zsd=zsd,
        )

        expected_shape = (batch_size, num_tx, num_rx)

        assert lsp.ds.shape == expected_shape
        assert lsp.asd.shape == expected_shape
        assert lsp.asa.shape == expected_shape
        assert lsp.sf.shape == expected_shape
        assert lsp.k_factor.shape == expected_shape
        assert lsp.zsa.shape == expected_shape
        assert lsp.zsd.shape == expected_shape


class TestSystemLevelChannelLSP:
    """Tests for public LSP sampling from system-level channels."""

    @staticmethod
    def _make_umi(device, precision):
        carrier_frequency = 3.5e9
        bs_array = tr38901.PanelArray(
            num_rows_per_panel=1,
            num_cols_per_panel=1,
            polarization="single",
            polarization_type="V",
            antenna_pattern="omni",
            carrier_frequency=carrier_frequency,
            precision=precision,
            device=device,
        )
        ut_array = tr38901.PanelArray(
            num_rows_per_panel=1,
            num_cols_per_panel=1,
            polarization="single",
            polarization_type="V",
            antenna_pattern="omni",
            carrier_frequency=carrier_frequency,
            precision=precision,
            device=device,
        )
        return tr38901.UMi(
            carrier_frequency=carrier_frequency,
            o2i_model="low",
            ut_array=ut_array,
            bs_array=bs_array,
            direction="downlink",
            precision=precision,
            device=device,
        )

    def test_sample_lsp_requires_topology(self, device, precision):
        """Test that LSP sampling requires a configured topology."""
        channel = self._make_umi(device, precision)
        with pytest.raises(RuntimeError, match="set_topology"):
            channel.sample_lsp()

    def test_sample_lsp_shape(self, device, precision):
        """Test that sampled LSPs match the configured topology shape."""
        channel = self._make_umi(device, precision)
        dtype = torch.float64 if precision == "double" else torch.float32
        batch_size = 2
        num_ut = 3

        ut_x = torch.tensor([20.0, 25.0, 30.0], dtype=dtype, device=device)
        ut_loc = torch.stack(
            [ut_x, torch.zeros_like(ut_x), torch.full_like(ut_x, 1.5)], dim=-1
        )
        ut_loc = ut_loc.unsqueeze(0).expand(batch_size, -1, -1).clone()
        channel.set_topology(
            ut_loc=ut_loc,
            bs_loc=torch.tensor(
                [[[0.0, 0.0, 10.0]], [[0.0, 0.0, 10.0]]],
                dtype=dtype,
                device=device,
            ),
            ut_orientations=torch.zeros(batch_size, num_ut, 3,
                                        dtype=dtype, device=device),
            bs_orientations=torch.zeros(batch_size, 1, 3,
                                        dtype=dtype, device=device),
            ut_velocities=torch.zeros(batch_size, num_ut, 3,
                                      dtype=dtype, device=device),
            in_state=torch.zeros(batch_size, num_ut, dtype=torch.bool,
                                 device=device),
            los=True,
        )

        lsp = channel.sample_lsp()
        assert lsp.ds.shape == (batch_size, 1, num_ut)
        assert lsp.asd.shape == (batch_size, 1, num_ut)
        assert lsp.asa.shape == (batch_size, 1, num_ut)
        assert lsp.sf.shape == (batch_size, 1, num_ut)
        assert lsp.k_factor.shape == (batch_size, 1, num_ut)
        assert lsp.zsa.shape == (batch_size, 1, num_ut)
        assert lsp.zsd.shape == (batch_size, 1, num_ut)
        assert lsp.pathloss.shape == (batch_size, 1, num_ut)

    def test_cached_lsp_reuses_its_pathloss(
        self, device, precision, monkeypatch
    ):
        """Check a cached LSP realization includes the penetration draw."""
        channel = self._make_umi(device, precision)
        dtype = torch.float64 if precision == "double" else torch.float32
        channel.set_topology(
            ut_loc=torch.tensor(
                [[[20.0, 0.0, 1.5]]], dtype=dtype, device=device
            ),
            bs_loc=torch.tensor(
                [[[0.0, 0.0, 10.0]]], dtype=dtype, device=device
            ),
            ut_orientations=torch.zeros(1, 1, 3, dtype=dtype, device=device),
            bs_orientations=torch.zeros(1, 1, 3, dtype=dtype, device=device),
            ut_velocities=torch.zeros(1, 1, 3, dtype=dtype, device=device),
            in_state=torch.ones(1, 1, dtype=torch.bool, device=device),
            los=False,
        )
        assert channel._lsp.pathloss is not None

        def fail_resample():
            raise AssertionError("cached channel call resampled path loss")

        monkeypatch.setattr(channel._lsp_sampler, "sample_pathloss", fail_resample)
        channel(num_time_samples=1, sampling_frequency=1e6)


class TestLSPGenerator:
    """Tests for LSPGenerator matching TensorFlow implementation tests."""

    # Test configuration
    CARRIER_FREQUENCY = 3.5e9  # Hz
    H_UT = 1.5  # Height of UTs
    H_BS = 35.0  # Height of BSs
    BATCH_SIZE = 100000  # Large batch for statistical tests
    NB_UT = 5  # Number of UTs for spatial correlation tests

    # Test thresholds
    MAX_ERR_KS = 1e-2  # Maximum KS statistic for distribution tests
    MAX_ERR_CROSS_CORR = 3e-2  # Maximum error for cross-correlation
    MAX_ERR_SPAT_CORR = 3e-2  # Maximum error for spatial correlation
    MAX_ERR_LOS_PROB = 1e-2  # Maximum error for LoS probability
    MAX_ERR_ZOD_OFFSET = 1e-2  # Maximum error for ZOD offset
    MAX_ERR_PATHLOSS_MEAN = 1.0  # Maximum error for pathloss mean
    MAX_ERR_PATHLOSS_STD = 1e-1  # Maximum error for pathloss std

    @pytest.mark.parametrize("spec_version", ["16.1", "19.2"])
    @pytest.mark.parametrize(
        "window_type,mean", [("ordinary", 9.0), ("metallized", 20.0)]
    )
    def test_rma_car_penetration_is_ut_specific_and_shared_across_bs(
        self,
        device,
        precision,
        monkeypatch,
        spec_version,
        window_type,
        mean,
    ):
        """Check the exact Section 7.4.3.2 car penetration formula."""
        import sionna.phy.channel.tr38901.lsp as lsp_module

        dtype = torch.float32 if precision == "single" else torch.float64
        fc = 3.5e9
        bs_array = tr38901.PanelArray(
            num_rows_per_panel=1,
            num_cols_per_panel=1,
            polarization="single",
            polarization_type="V",
            antenna_pattern="omni",
            carrier_frequency=fc,
            precision=precision,
            device=device,
        )
        ut_array = tr38901.PanelArray(
            num_rows_per_panel=1,
            num_cols_per_panel=1,
            polarization="single",
            polarization_type="V",
            antenna_pattern="omni",
            carrier_frequency=fc,
            precision=precision,
            device=device,
        )
        scenario = tr38901.RMaScenario(
            fc,
            ut_array,
            bs_array,
            "downlink",
            spec_version=spec_version,
            car_window_type=window_type,
            precision=precision,
            device=device,
        )
        scenario.set_topology(
            ut_loc=torch.tensor(
                [[[100.0, 0.0, 1.5], [120.0, 0.0, 1.5]]],
                dtype=dtype,
                device=device,
            ),
            bs_loc=torch.tensor(
                [
                    [
                        [0.0, 0.0, 35.0],
                        [200.0, 0.0, 35.0],
                        [0.0, 200.0, 35.0],
                    ]
                ],
                dtype=dtype,
                device=device,
            ),
            ut_orientations=torch.zeros(1, 2, 3, dtype=dtype, device=device),
            bs_orientations=torch.zeros(1, 3, 3, dtype=dtype, device=device),
            ut_velocities=torch.zeros(1, 2, 3, dtype=dtype, device=device),
            in_state=torch.zeros(1, 2, dtype=torch.bool, device=device),
            in_car=torch.tensor([[True, False]], device=device),
            los=True,
        )

        observed_shapes = []

        def fake_normal(shape, dtype, device, generator=None):
            del generator
            observed_shapes.append(tuple(shape))
            return torch.tensor(
                [[[0.25, -0.75]]], dtype=dtype, device=device
            )

        monkeypatch.setattr(lsp_module, "normal", fake_normal)
        sampler = LSPGenerator(scenario)
        car_loss = sampler._car_penetration_loss()

        expected = torch.tensor(
            [[[mean + 1.25, 0.0]]], dtype=dtype, device=device
        ).expand(1, 3, 2)
        assert observed_shapes == [(1, 1, 2)]
        torch.testing.assert_close(car_loss, expected)
        torch.testing.assert_close(car_loss[:, 0], car_loss[:, 1])
        torch.testing.assert_close(car_loss[:, 0], car_loss[:, 2])

        monkeypatch.setattr(
            sampler, "_o2i_low_loss", lambda: torch.zeros_like(car_loss)
        )
        total = sampler.sample_pathloss()
        torch.testing.assert_close(total, scenario.basic_pathloss + expected)

    @pytest.mark.parametrize("o2i_model", ["low", "high"])
    def test_o2i_random_component_is_shared_across_bs(
        self, device, precision, monkeypatch, o2i_model
    ):
        """Test O2I random loss is sampled once per UT, not per link."""

        import sionna.phy.channel.tr38901.lsp as lsp_module

        fc = 6e9
        dtype = torch.float32 if precision == "single" else torch.float64
        bs_array = tr38901.PanelArray(
            num_rows_per_panel=1,
            num_cols_per_panel=1,
            polarization="single",
            polarization_type="V",
            antenna_pattern="omni",
            carrier_frequency=fc,
            precision=precision,
            device=device,
        )
        ut_array = tr38901.PanelArray(
            num_rows_per_panel=1,
            num_cols_per_panel=1,
            polarization="single",
            polarization_type="V",
            antenna_pattern="omni",
            carrier_frequency=fc,
            precision=precision,
            device=device,
        )
        scenario = tr38901.UMiScenario(
            fc,
            o2i_model,
            ut_array,
            bs_array,
            "downlink",
            precision=precision,
            device=device,
        )
        scenario.set_topology(
            ut_loc=torch.tensor(
                [[[100.0, 20.0, 1.5], [150.0, 40.0, 1.5]]],
                dtype=dtype,
                device=device,
            ),
            bs_loc=torch.tensor(
                [
                    [
                        [0.0, 0.0, 10.0],
                        [100.0, 0.0, 10.0],
                        [0.0, 100.0, 10.0],
                    ]
                ],
                dtype=dtype,
                device=device,
            ),
            ut_orientations=torch.zeros(1, 2, 3, dtype=dtype, device=device),
            bs_orientations=torch.zeros(1, 3, 3, dtype=dtype, device=device),
            ut_velocities=torch.zeros(1, 2, 3, dtype=dtype, device=device),
            in_state=torch.ones(1, 2, dtype=torch.bool, device=device),
            los=False,
        )

        observed_shapes = []

        def fake_normal(shape, dtype, device, generator=None):
            del generator
            observed_shapes.append(tuple(shape))
            return torch.tensor(
                [[[0.25, -0.75]]], dtype=dtype, device=device
            )

        monkeypatch.setattr(lsp_module, "normal", fake_normal)
        lsp_sampler = LSPGenerator(scenario)
        lsp_sampler.topology_updated_callback()
        o2i_loss = getattr(lsp_sampler, f"_o2i_{o2i_model}_loss")()

        assert observed_shapes == [(1, 1, 2)]
        assert o2i_loss.shape == (1, 3, 2)
        torch.testing.assert_close(o2i_loss[:, 0, :], o2i_loss[:, 1, :])
        torch.testing.assert_close(o2i_loss[:, 0, :], o2i_loss[:, 2, :])

    def test_o2i_random_component_uses_10m_spatial_correlation(
        self, device, precision
    ):
        """Check Section 7.4.3.1 random loss has 10 m correlation."""
        fc = 6e9
        dtype = torch.float32 if precision == "single" else torch.float64
        batch_size = 12000
        bs_array = tr38901.PanelArray(
            num_rows_per_panel=1,
            num_cols_per_panel=1,
            polarization="single",
            polarization_type="V",
            antenna_pattern="omni",
            carrier_frequency=fc,
            precision=precision,
            device=device,
        )
        ut_array = tr38901.PanelArray(
            num_rows_per_panel=1,
            num_cols_per_panel=1,
            polarization="single",
            polarization_type="V",
            antenna_pattern="omni",
            carrier_frequency=fc,
            precision=precision,
            device=device,
        )
        scenario = tr38901.UMiScenario(
            fc,
            "low",
            ut_array,
            bs_array,
            "downlink",
            precision=precision,
            device=device,
        )
        scenario.set_spatial_consistency_enabled(True)
        ut_loc = torch.tensor(
            [[100.0, 0.0, 1.5], [110.0, 0.0, 1.5]],
            dtype=dtype,
            device=device,
        ).unsqueeze(0).expand(batch_size, -1, -1).clone()
        scenario.set_topology(
            ut_loc=ut_loc,
            bs_loc=torch.tensor(
                [[[0.0, 0.0, 10.0]]], dtype=dtype, device=device
            ).expand(batch_size, -1, -1).clone(),
            ut_orientations=torch.zeros(
                batch_size, 2, 3, dtype=dtype, device=device
            ),
            bs_orientations=torch.zeros(
                batch_size, 1, 3, dtype=dtype, device=device
            ),
            ut_velocities=torch.zeros(
                batch_size, 2, 3, dtype=dtype, device=device
            ),
            in_state=torch.ones(
                batch_size, 2, dtype=torch.bool, device=device
            ),
            los=False,
        )
        sampler = LSPGenerator(scenario)
        sampler.topology_updated_callback()
        samples = sampler._sample_o2i_penetration_random(1.0)[:, 0]
        measured = np.corrcoef(samples.cpu().numpy().T)[0, 1]
        assert measured == pytest.approx(np.exp(-1.0), abs=0.035)

    @pytest.mark.parametrize("o2i_model", ["low", "high"])
    def test_below_6ghz_o2i_uses_legacy_compatibility_model(
        self, device, precision, monkeypatch, o2i_model
    ):
        """Check Table 7.4.3-3 wall loss and zero random component."""

        import sionna.phy.channel.tr38901.lsp as lsp_module

        fc = 3.5e9
        dtype = torch.float32 if precision == "single" else torch.float64
        bs_array = tr38901.PanelArray(
            num_rows_per_panel=1,
            num_cols_per_panel=1,
            polarization="single",
            polarization_type="V",
            antenna_pattern="omni",
            carrier_frequency=fc,
            precision=precision,
            device=device,
        )
        ut_array = tr38901.PanelArray(
            num_rows_per_panel=1,
            num_cols_per_panel=1,
            polarization="single",
            polarization_type="V",
            antenna_pattern="omni",
            carrier_frequency=fc,
            precision=precision,
            device=device,
        )
        scenario = tr38901.UMiScenario(
            fc,
            o2i_model,
            ut_array,
            bs_array,
            "downlink",
            precision=precision,
            device=device,
        )
        scenario.set_topology(
            ut_loc=torch.tensor(
                [[[100.0, 0.0, 1.5], [120.0, 0.0, 1.5]]],
                dtype=dtype,
                device=device,
            ),
            bs_loc=torch.tensor(
                [[[0.0, 0.0, 10.0]]], dtype=dtype, device=device
            ),
            ut_orientations=torch.zeros(1, 2, 3, dtype=dtype, device=device),
            bs_orientations=torch.zeros(1, 1, 3, dtype=dtype, device=device),
            ut_velocities=torch.zeros(1, 2, 3, dtype=dtype, device=device),
            in_state=torch.tensor([[True, False]], device=device),
            los=False,
            distance_2d_in=torch.tensor(
                [[4.0, 8.0]], dtype=dtype, device=device
            ),
        )

        def fail_normal(*_args, **_kwargs):
            raise AssertionError("legacy O2I model must not sample random loss")

        monkeypatch.setattr(lsp_module, "normal", fail_normal)
        sampler = LSPGenerator(scenario)
        loss = getattr(sampler, f"_o2i_{o2i_model}_loss")()
        expected = torch.tensor(
            [[[22.0, 0.0]]], dtype=dtype, device=device
        )
        torch.testing.assert_close(loss, expected)

    def test_lsp_shared_for_co_sited_bs(self, device, precision):
        """Test that LSP realizations are shared by co-sited sectors."""
        fc = self.CARRIER_FREQUENCY
        dtype = torch.float32 if precision == "single" else torch.float64

        bs_array = tr38901.PanelArray(
            num_rows_per_panel=1,
            num_cols_per_panel=1,
            polarization="single",
            polarization_type="V",
            antenna_pattern="omni",
            carrier_frequency=fc,
            precision=precision,
            device=device,
        )
        ut_array = tr38901.PanelArray(
            num_rows_per_panel=1,
            num_cols_per_panel=1,
            polarization="single",
            polarization_type="V",
            antenna_pattern="omni",
            carrier_frequency=fc,
            precision=precision,
            device=device,
        )
        scenario = tr38901.UMiScenario(
            fc, "low", ut_array, bs_array, "downlink", precision=precision, device=device
        )
        scenario.set_topology(
            ut_loc=torch.tensor([[[100.0, 0.0, 1.5]]], dtype=dtype, device=device),
            bs_loc=torch.tensor(
                [[[0.0, 0.0, 10.0], [0.0, 0.0, 10.0], [50.0, 0.0, 10.0]]],
                dtype=dtype,
                device=device,
            ),
            ut_orientations=torch.zeros(1, 1, 3, dtype=dtype, device=device),
            bs_orientations=torch.zeros(1, 3, 3, dtype=dtype, device=device),
            ut_velocities=torch.zeros(1, 1, 3, dtype=dtype, device=device),
            in_state=torch.tensor([[False]], device=device),
            los=True,
        )

        lsp_sampler = LSPGenerator(scenario)
        lsp_sampler.topology_updated_callback()
        lsp = lsp_sampler()

        ungrouped_differs = False
        for value in (lsp.ds, lsp.asd, lsp.asa, lsp.sf, lsp.k_factor, lsp.zsa, lsp.zsd):
            torch.testing.assert_close(value[:, 0, :], value[:, 1, :])
            ungrouped_differs = ungrouped_differs or not torch.equal(
                value[:, 0, :], value[:, 2, :]
            )
        assert ungrouped_differs

    @pytest.fixture(scope="class")
    @classmethod
    def lsp_samples(cls, request):
        """Sample LSPs from all channel models for testing.

        Uses the device specified by --device flag.
        """
        device_option = request.config.getoption("--device", default="gpu")
        if device_option == "cpu":
            device = "cpu"
        elif device_option == "gpu" and torch.cuda.is_available():
            device = "cuda:0"
        elif device_option == "all" and torch.cuda.is_available():
            device = "cuda:0"  # Use GPU for class-scoped fixture when "all"
        else:
            device = "cpu"
        batch_size = cls.BATCH_SIZE
        nb_bs = 1
        nb_ut = cls.NB_UT
        fc = cls.CARRIER_FREQUENCY
        h_ut = cls.H_UT
        h_bs = cls.H_BS
        dtype = torch.float64

        # Create antenna arrays
        bs_array = tr38901.PanelArray(
            num_rows_per_panel=2,
            num_cols_per_panel=2,
            polarization="dual",
            polarization_type="VH",
            antenna_pattern="38.901",
            carrier_frequency=fc,
            precision="double",
            device=device,
        )
        ut_array = tr38901.PanelArray(
            num_rows_per_panel=1,
            num_cols_per_panel=1,
            polarization="dual",
            polarization_type="VH",
            antenna_pattern="38.901",
            carrier_frequency=fc,
            precision="double",
            device=device,
        )

        # Generate shared topology
        ut_orientations = torch.zeros(batch_size, nb_ut, dtype=dtype, device=device)
        bs_orientations = torch.zeros(batch_size, nb_bs, dtype=dtype, device=device)
        ut_velocities = torch.zeros(batch_size, nb_ut, dtype=dtype, device=device)

        ut_loc = generate_random_loc(
            batch_size, nb_ut, (100, 2000), (100, 2000), (h_ut, h_ut),
            share_loc=True, dtype=dtype, device=device
        )
        bs_loc = generate_random_loc(
            batch_size, nb_bs, (0, 100), (0, 100), (h_bs, h_bs),
            share_loc=True, dtype=dtype, device=device
        )

        samples = {}

        # Test all scenarios
        for model_name, scenario_cls in [
            ("rma", tr38901.RMaScenario),
            ("umi", tr38901.UMiScenario),
            ("uma", tr38901.UMaScenario),
        ]:
            samples[model_name] = {}

            # Create scenario
            if model_name == "rma":
                scenario = scenario_cls(
                    fc,
                    ut_array,
                    bs_array,
                    "uplink",
                    precision="double",
                    device=device,
                    spec_version="16.1",
                )
            else:
                scenario = scenario_cls(
                    fc,
                    "low",
                    ut_array,
                    bs_array,
                    "uplink",
                    precision="double",
                    device=device,
                    spec_version="16.1",
                )

            lsp_sampler = LSPGenerator(scenario)

            # LoS
            in_state = generate_random_bool(batch_size, nb_ut, 0.0, device=device)
            topology_kwargs = {}
            if model_name == "rma":
                topology_kwargs["in_car"] = torch.zeros_like(in_state)
            scenario.set_topology(
                ut_loc, bs_loc, ut_orientations, bs_orientations,
                ut_velocities, in_state, los=True, **topology_kwargs
            )
            lsp_sampler.topology_updated_callback()
            samples[model_name]["los"] = {
                "lsp": lsp_sampler(),
                "zod_offset": scenario.zod_offset.cpu().numpy().copy(),
                "pathloss": lsp_sampler.sample_pathloss()[:, 0, :].cpu().numpy().copy(),
            }

            # NLoS
            in_state = generate_random_bool(batch_size, nb_ut, 0.0, device=device)
            scenario.set_topology(
                ut_loc, bs_loc, ut_orientations, bs_orientations,
                ut_velocities, in_state, los=False
            )
            lsp_sampler.topology_updated_callback()
            samples[model_name]["nlos"] = {
                "lsp": lsp_sampler(),
                "zod_offset": scenario.zod_offset.cpu().numpy().copy(),
                "pathloss": lsp_sampler.sample_pathloss()[:, 0, :].cpu().numpy().copy(),
            }

            # O2I
            in_state = generate_random_bool(batch_size, nb_ut, 1.0, device=device)
            scenario.set_topology(
                ut_loc, bs_loc, ut_orientations, bs_orientations,
                ut_velocities, in_state, los=False
            )
            lsp_sampler.topology_updated_callback()
            samples[model_name]["o2i"] = {
                "lsp": lsp_sampler(),
                "zod_offset": scenario.zod_offset.cpu().numpy().copy(),
                "pathloss": lsp_sampler.sample_pathloss()[:, 0, :].cpu().numpy().copy(),
            }

            # Store scenario info — use .copy() to decouple from in-place-updated buffers
            samples[model_name]["los_prob"] = scenario.los_probability.cpu().numpy().copy()
            samples[model_name]["d_2d"] = scenario.distance_2d.cpu().numpy().copy()
            samples[model_name]["d_2d_ut"] = scenario.matrix_ut_distance_2d.cpu().numpy().copy()
            samples[model_name]["d_2d_out"] = scenario.distance_2d_out.cpu().numpy().copy()
            samples[model_name]["d_3d"] = scenario.distance_3d[0, 0, :].cpu().numpy().copy()
            if model_name == "rma":
                samples[model_name]["w"] = scenario.average_street_width
                samples[model_name]["h"] = scenario.average_building_height

        return samples

    @pytest.mark.parametrize("model", ["rma", "umi", "uma"])
    @pytest.mark.parametrize("submodel", ["los", "nlos", "o2i"])
    def test_ds_distribution(self, lsp_samples, model, submodel):
        """Test the distribution of LSP DS (delay spread)."""
        lsp = lsp_samples[model][submodel]["lsp"]
        samples = lsp.ds[:, 0, 0].cpu().numpy()
        samples = np.log10(samples)

        mu, std = log10DS(model, submodel, self.CARRIER_FREQUENCY)
        D, _ = kstest(samples, norm.cdf, args=(mu, std))

        assert D <= self.MAX_ERR_KS, f"{model}:{submodel} DS distribution failed"

    @pytest.mark.parametrize("model", ["rma", "umi", "uma"])
    @pytest.mark.parametrize("submodel", ["los", "nlos", "o2i"])
    def test_asa_distribution(self, lsp_samples, model, submodel):
        """Test the distribution of LSP ASA (azimuth angle spread of arrival)."""
        lsp = lsp_samples[model][submodel]["lsp"]
        samples = lsp.asa[:, 0, 0].cpu().numpy()
        samples = np.log10(samples)

        mu, std = log10ASA(model, submodel, self.CARRIER_FREQUENCY)
        a = -np.inf
        b = (np.log10(104) - mu) / std
        samples_ref = limited_normal(self.BATCH_SIZE, a, b, mu, std)

        # Check maximum value is not exceeded
        maxval = np.max(samples)
        assert maxval <= np.log10(104), f"{model}:{submodel} ASA exceeds max"

        # KS test on continuous part
        samples = samples[samples < np.log10(104)]
        samples_ref = samples_ref[samples_ref < np.log10(104)]
        D, _ = kstest(samples, samples_ref)
        assert D <= self.MAX_ERR_KS, f"{model}:{submodel} ASA distribution failed"

    @pytest.mark.parametrize("model", ["rma", "umi", "uma"])
    @pytest.mark.parametrize("submodel", ["los", "nlos", "o2i"])
    def test_asd_distribution(self, lsp_samples, model, submodel):
        """Test the distribution of LSP ASD (azimuth angle spread of departure)."""
        lsp = lsp_samples[model][submodel]["lsp"]
        samples = lsp.asd[:, 0, 0].cpu().numpy()
        samples = np.log10(samples)

        mu, std = log10ASD(model, submodel, self.CARRIER_FREQUENCY)
        a = -np.inf
        b = (np.log10(104) - mu) / std
        samples_ref = limited_normal(self.BATCH_SIZE, a, b, mu, std)

        # Check maximum value is not exceeded
        maxval = np.max(samples)
        assert maxval <= np.log10(104), f"{model}:{submodel} ASD exceeds max"

        # KS test on continuous part
        samples = samples[samples < np.log10(104)]
        samples_ref = samples_ref[samples_ref < np.log10(104)]
        D, _ = kstest(samples, samples_ref)
        assert D <= self.MAX_ERR_KS, f"{model}:{submodel} ASD distribution failed"

    @pytest.mark.parametrize("model", ["rma", "umi", "uma"])
    @pytest.mark.parametrize("submodel", ["los", "nlos", "o2i"])
    def test_zsa_distribution(self, lsp_samples, model, submodel):
        """Test the distribution of LSP ZSA (zenith angle spread of arrival)."""
        lsp = lsp_samples[model][submodel]["lsp"]
        samples = lsp.zsa[:, 0, 0].cpu().numpy()
        samples = np.log10(samples)

        mu, std = log10ZSA(model, submodel, self.CARRIER_FREQUENCY)
        a = -np.inf
        b = (np.log10(52) - mu) / std
        samples_ref = limited_normal(self.BATCH_SIZE, a, b, mu, std)

        # Check maximum value is not exceeded
        maxval = np.max(samples)
        assert maxval <= np.log10(52), f"{model}:{submodel} ZSA exceeds max"

        # KS test on continuous part
        samples = samples[samples < np.log10(52)]
        samples_ref = samples_ref[samples_ref < np.log10(52)]
        D, _ = kstest(samples, samples_ref)
        assert D <= self.MAX_ERR_KS, f"{model}:{submodel} ZSA distribution failed"

    @pytest.mark.parametrize("model", ["rma", "umi", "uma"])
    @pytest.mark.parametrize("submodel", ["los", "nlos", "o2i"])
    def test_zsd_distribution(self, lsp_samples, model, submodel):
        """Test the distribution of LSP ZSD (zenith angle spread of departure)."""
        lsp = lsp_samples[model][submodel]["lsp"]
        d_2d = lsp_samples[model]["d_2d"][0, 0, 0]
        samples = lsp.zsd[:, 0, 0].cpu().numpy()
        samples = np.log10(samples)

        mu, std = log10ZSD(
            model, submodel, d_2d, self.CARRIER_FREQUENCY, self.H_BS, self.H_UT
        )
        a = -np.inf
        b = (np.log10(52) - mu) / std
        samples_ref = limited_normal(self.BATCH_SIZE, a, b, mu, std)

        # Check maximum value is not exceeded
        maxval = np.max(samples)
        assert maxval <= np.log10(52), f"{model}:{submodel} ZSD exceeds max"

        # KS test on continuous part
        samples = samples[samples < np.log10(52)]
        samples_ref = samples_ref[samples_ref < np.log10(52)]
        D, _ = kstest(samples, samples_ref)
        assert D <= self.MAX_ERR_KS, f"{model}:{submodel} ZSD distribution failed"

    @pytest.mark.parametrize("model", ["rma", "umi", "uma"])
    @pytest.mark.parametrize("submodel", ["los", "nlos", "o2i"])
    def test_sf_distribution(self, lsp_samples, model, submodel):
        """Test the distribution of LSP SF (shadow fading)."""
        lsp = lsp_samples[model][submodel]["lsp"]
        d_2d = lsp_samples[model]["d_2d"][0, 0, 0]
        samples = lsp.sf[:, 0, 0].cpu().numpy()
        samples = 10.0 * np.log10(samples)  # Convert to dB

        mu, std = log10SF_dB(
            model, submodel, d_2d, self.CARRIER_FREQUENCY, self.H_BS, self.H_UT
        )
        D, _ = kstest(samples, norm.cdf, args=(mu, std))
        assert D <= self.MAX_ERR_KS, f"{model}:{submodel} SF distribution failed"

    @pytest.mark.parametrize("model", ["rma", "umi", "uma"])
    def test_k_factor_distribution(self, lsp_samples, model):
        """Test the distribution of LSP K-factor (LoS only)."""
        lsp = lsp_samples[model]["los"]["lsp"]
        samples = lsp.k_factor[:, 0, 0].cpu().numpy()
        samples = 10.0 * np.log10(samples)  # Convert to dB

        mu, std = log10K_dB(model, "los")
        D, _ = kstest(samples, norm.cdf, args=(mu, std))
        assert D <= self.MAX_ERR_KS, f"{model}:los K-factor distribution failed"

    @pytest.mark.parametrize("model", ["rma", "umi", "uma"])
    @pytest.mark.parametrize("submodel", ["los", "nlos", "o2i"])
    def test_cross_correlation(self, lsp_samples, model, submodel):
        """Test the LSP cross-correlation matrix."""
        lsp = lsp_samples[model][submodel]["lsp"]
        for ut_index in range(lsp.ds.shape[-1]):
            lsp_list = [
                np.log10(lsp.ds[:, 0, ut_index].cpu().numpy()),
                np.log10(lsp.asd[:, 0, ut_index].cpu().numpy()),
                np.log10(lsp.asa[:, 0, ut_index].cpu().numpy()),
                np.log10(lsp.sf[:, 0, ut_index].cpu().numpy()),
            ]
            if submodel == "los":
                lsp_list.append(
                    np.log10(lsp.k_factor[:, 0, ut_index].cpu().numpy())
                )
            lsp_list.extend(
                [
                    np.log10(lsp.zsa[:, 0, ut_index].cpu().numpy()),
                    np.log10(lsp.zsd[:, 0, ut_index].cpu().numpy()),
                ]
            )
            measured = np.corrcoef(np.stack(lsp_list, axis=-1).T)
            max_err = np.max(np.abs(cross_corr(model, submodel) - measured))
            assert max_err <= self.MAX_ERR_CROSS_CORR, (
                f"{model}:{submodel} UT {ut_index} cross-correlation failed"
            )

    @pytest.mark.parametrize("model", ["rma", "umi", "uma"])
    @pytest.mark.parametrize("submodel", ["los", "nlos", "o2i"])
    def test_spatial_correlation(self, lsp_samples, model, submodel):
        """Test the spatial correlation of LSPs."""
        lsp = lsp_samples[model][submodel]["lsp"]
        d_2d_ut = lsp_samples[model]["d_2d_ut"][0, 0]

        # Get measured correlations
        ds_samples = np.log10(lsp.ds[:, 0, :].cpu().numpy())
        asd_samples = np.log10(lsp.asd[:, 0, :].cpu().numpy())
        asa_samples = np.log10(lsp.asa[:, 0, :].cpu().numpy())
        sf_samples = np.log10(lsp.sf[:, 0, :].cpu().numpy())
        if submodel == "los":
            k_samples = np.log10(lsp.k_factor[:, 0, :].cpu().numpy())
        zsa_samples = np.log10(lsp.zsa[:, 0, :].cpu().numpy())
        zsd_samples = np.log10(lsp.zsd[:, 0, :].cpu().numpy())

        C_ds_measured = np.corrcoef(ds_samples.T)[0]
        C_asd_measured = np.corrcoef(asd_samples.T)[0]
        C_asa_measured = np.corrcoef(asa_samples.T)[0]
        C_sf_measured = np.corrcoef(sf_samples.T)[0]
        if submodel == "los":
            C_k_measured = np.corrcoef(k_samples.T)[0]
        C_zsa_measured = np.corrcoef(zsa_samples.T)[0]
        C_zsd_measured = np.corrcoef(zsd_samples.T)[0]

        correlation_by_name = {
            "ds": np.exp(-d_2d_ut / corr_dist_ds(model, submodel)),
            "asd": np.exp(-d_2d_ut / corr_dist_asd(model, submodel)),
            "asa": np.exp(-d_2d_ut / corr_dist_asa(model, submodel)),
            "sf": np.exp(-d_2d_ut / corr_dist_sf(model, submodel)),
            "zsa": np.exp(-d_2d_ut / corr_dist_zsa(model, submodel)),
            "zsd": np.exp(-d_2d_ut / corr_dist_zsd(model, submodel)),
        }
        if submodel == "los":
            correlation_by_name["k"] = np.exp(
                -d_2d_ut / corr_dist_k(model, submodel)
            )
        names = ["ds", "asd", "asa", "sf"]
        measured = {
            "ds": C_ds_measured,
            "asd": C_asd_measured,
            "asa": C_asa_measured,
            "sf": C_sf_measured,
            "zsa": C_zsa_measured,
            "zsd": C_zsd_measured,
        }
        if submodel == "los":
            names.append("k")
            measured["k"] = C_k_measured
        names.extend(["zsa", "zsd"])
        expected = _winner_spatial_correlation(
            cross_corr(model, submodel), names, correlation_by_name
        )
        for name, measured_correlation in measured.items():
            assert (
                np.max(np.abs(measured_correlation - expected[name]))
                <= self.MAX_ERR_SPAT_CORR
            ), f"{model}:{submodel} {name} spatial correlation failed"

    @pytest.mark.parametrize("model", ["rma", "umi", "uma"])
    def test_los_probability(self, lsp_samples, model):
        """Test LoS probability calculation."""
        d_2d_out = lsp_samples[model]["d_2d_out"]
        los_prob_ref = los_probability(model, d_2d_out, self.H_UT)
        los_prob = lsp_samples[model]["los_prob"]
        max_err = np.max(np.abs(los_prob_ref - los_prob))

        assert max_err <= self.MAX_ERR_LOS_PROB, f"{model} LoS probability failed"

    @pytest.mark.parametrize("model", ["rma", "umi", "uma"])
    @pytest.mark.parametrize("submodel", ["los", "nlos", "o2i"])
    def test_zod_offset(self, lsp_samples, model, submodel):
        """Test ZOD offset calculation."""
        d_2d = lsp_samples[model]["d_2d"]
        samples = lsp_samples[model][submodel]["zod_offset"]
        samples_ref = zod_offset(model, submodel, self.CARRIER_FREQUENCY, d_2d, self.H_UT)
        max_err = np.max(np.abs(samples - samples_ref))

        assert max_err <= self.MAX_ERR_ZOD_OFFSET, f"{model}:{submodel} ZOD offset failed"


_INDOOR_LSP_CROSS_CORR = {
    ("inh", "los"): np.array(
        [
            [1.0, 0.6, 0.8, -0.8, -0.5, 0.2, 0.1],
            [0.6, 1.0, 0.4, -0.4, 0.0, 0.0, 0.5],
            [0.8, 0.4, 1.0, -0.5, 0.0, 0.5, 0.0],
            [-0.8, -0.4, -0.5, 1.0, 0.5, 0.3, 0.2],
            [-0.5, 0.0, 0.0, 0.5, 1.0, 0.1, 0.0],
            [0.2, 0.0, 0.5, 0.3, 0.1, 1.0, 0.0],
            [0.1, 0.5, 0.0, 0.2, 0.0, 0.0, 1.0],
        ]
    ),
    ("inh", "nlos"): np.array(
        [
            [1.0, 0.4, 0.0, -0.5, -0.06, -0.27],
            [0.4, 1.0, 0.0, 0.0, 0.23, 0.35],
            [0.0, 0.0, 1.0, -0.4, 0.43, -0.08],
            [-0.5, 0.0, -0.4, 1.0, 0.0, 0.0],
            [-0.06, 0.23, 0.43, 0.0, 1.0, 0.42],
            [-0.27, 0.35, -0.08, 0.0, 0.42, 1.0],
        ]
    ),
    ("inf", "los"): np.array(
        [
            [1.0, 0.0, 0.0, 0.0, -0.7, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, -0.5, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [-0.7, -0.5, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ]
    ),
    ("inf", "nlos"): np.eye(6),
}

_INDOOR_LSP_CORR_DIST = {
    ("inh", "los"): {
        "ds": 8.0,
        "asd": 7.0,
        "asa": 5.0,
        "sf": 10.0,
        "k": 4.0,
        "zsa": 4.0,
        "zsd": 4.0,
    },
    ("inh", "nlos"): {
        "ds": 5.0,
        "asd": 3.0,
        "asa": 3.0,
        "sf": 6.0,
        "zsa": 4.0,
        "zsd": 4.0,
    },
    ("inf", "los"): {
        "ds": 10.0,
        "asd": 10.0,
        "asa": 10.0,
        "sf": 10.0,
        "k": 10.0,
        "zsa": 10.0,
        "zsd": 10.0,
    },
    ("inf", "nlos"): {
        "ds": 10.0,
        "asd": 10.0,
        "asa": 10.0,
        "sf": 10.0,
        "zsa": 10.0,
        "zsd": 10.0,
    },
}


class TestIndoorLSPGeneratorCorrelations:
    """Tests for LSP correlations of indoor-office and indoor-factory models."""

    CARRIER_FREQUENCY = 3.5e9
    BATCH_SIZE = 100000
    NB_UT = 5
    MAX_ERR_CROSS_CORR = 3e-2
    # The final public LSP values include the TR 38.901 angle-spread caps.
    MAX_ERR_SPAT_CORR = 4e-2

    @pytest.fixture(scope="class")
    @classmethod
    def lsp_samples(cls, request):
        """Sample InH and InF LSPs for statistical correlation tests."""
        device_option = request.config.getoption("--device", default="gpu")
        if device_option == "cpu":
            device = "cpu"
        elif device_option == "gpu" and torch.cuda.is_available():
            device = "cuda:0"
        elif device_option == "all" and torch.cuda.is_available():
            device = "cuda:0"
        else:
            device = "cpu"

        batch_size = cls.BATCH_SIZE
        nb_bs = 1
        nb_ut = cls.NB_UT
        fc = cls.CARRIER_FREQUENCY
        dtype = torch.float64

        bs_array = tr38901.PanelArray(
            num_rows_per_panel=2,
            num_cols_per_panel=2,
            polarization="dual",
            polarization_type="VH",
            antenna_pattern="38.901",
            carrier_frequency=fc,
            precision="double",
            device=device,
        )
        ut_array = tr38901.PanelArray(
            num_rows_per_panel=1,
            num_cols_per_panel=1,
            polarization="dual",
            polarization_type="VH",
            antenna_pattern="38.901",
            carrier_frequency=fc,
            precision="double",
            device=device,
        )

        ut_x = 10.0 + torch.arange(nb_ut, dtype=dtype, device=device)
        ut_loc = torch.stack(
            [
                ut_x,
                torch.zeros_like(ut_x),
                torch.full_like(ut_x, 1.5),
            ],
            dim=-1,
        ).reshape(1, nb_ut, 3).expand(batch_size, -1, -1).clone()
        bs_loc = torch.tensor(
            [[[0.0, 0.0, 8.0]]], dtype=dtype, device=device
        ).expand(batch_size, nb_bs, -1).clone()
        ut_orientations = torch.zeros(
            batch_size, nb_ut, 3, dtype=dtype, device=device
        )
        bs_orientations = torch.zeros(
            batch_size, nb_bs, 3, dtype=dtype, device=device
        )
        ut_velocities = torch.zeros(
            batch_size, nb_ut, 3, dtype=dtype, device=device
        )
        in_state = torch.ones(batch_size, nb_ut, dtype=torch.bool, device=device)

        samples = {}
        for model_name, scenario in (
            (
                "inh",
                tr38901.InHScenario(
                    fc,
                    "open",
                    ut_array,
                    bs_array,
                    "uplink",
                    precision="double",
                    device=device,
                    spec_version="16.1",
                ),
            ),
            (
                "inf",
                tr38901.InFScenario(
                    fc,
                    "SH",
                    ut_array,
                    bs_array,
                    "uplink",
                    precision="double",
                    device=device,
                    spec_version="16.1",
                ),
            ),
        ):
            samples[model_name] = {}
            lsp_sampler = LSPGenerator(scenario)
            for submodel, los in (("los", True), ("nlos", False)):
                scenario.set_topology(
                    ut_loc,
                    bs_loc,
                    ut_orientations,
                    bs_orientations,
                    ut_velocities,
                    in_state,
                    los=los,
                )
                lsp_sampler.topology_updated_callback()
                samples[model_name][submodel] = {
                    "lsp": lsp_sampler(),
                    "d_2d_ut": (
                        scenario.matrix_ut_distance_2d[0].cpu().numpy().copy()
                    ),
                }

        return samples

    @pytest.mark.parametrize("model", ["inh", "inf"])
    @pytest.mark.parametrize("submodel", ["los", "nlos"])
    def test_cross_correlation(self, lsp_samples, model, submodel):
        """Test InH/InF LSP cross-correlation matrices."""
        lsp = lsp_samples[model][submodel]["lsp"]
        reference = _INDOOR_LSP_CROSS_CORR[(model, submodel)]
        for ut_index in range(lsp.ds.shape[-1]):
            lsp_list = [
                np.log10(lsp.ds[:, 0, ut_index].cpu().numpy()),
                np.log10(lsp.asd[:, 0, ut_index].cpu().numpy()),
                np.log10(lsp.asa[:, 0, ut_index].cpu().numpy()),
                np.log10(lsp.sf[:, 0, ut_index].cpu().numpy()),
            ]
            if submodel == "los":
                lsp_list.append(
                    np.log10(lsp.k_factor[:, 0, ut_index].cpu().numpy())
                )
            lsp_list.extend(
                [
                    np.log10(lsp.zsa[:, 0, ut_index].cpu().numpy()),
                    np.log10(lsp.zsd[:, 0, ut_index].cpu().numpy()),
                ]
            )
            measured = np.corrcoef(np.stack(lsp_list, axis=-1).T)
            max_err = np.max(np.abs(reference - measured))
            assert max_err <= self.MAX_ERR_CROSS_CORR, (
                f"{model}:{submodel} UT {ut_index} cross-correlation failed"
            )

    @pytest.mark.parametrize("model", ["inh", "inf"])
    @pytest.mark.parametrize("submodel", ["los", "nlos"])
    def test_spatial_correlation(self, lsp_samples, model, submodel):
        """Test InH/InF spatial correlation of each LSP across UTs."""
        lsp = lsp_samples[model][submodel]["lsp"]
        d_2d_ut = lsp_samples[model][submodel]["d_2d_ut"][0]
        corr_dist = _INDOOR_LSP_CORR_DIST[(model, submodel)]

        measured = {
            "ds": np.corrcoef(np.log10(lsp.ds[:, 0, :].cpu().numpy()).T)[0],
            "asd": np.corrcoef(np.log10(lsp.asd[:, 0, :].cpu().numpy()).T)[0],
            "asa": np.corrcoef(np.log10(lsp.asa[:, 0, :].cpu().numpy()).T)[0],
            "sf": np.corrcoef(np.log10(lsp.sf[:, 0, :].cpu().numpy()).T)[0],
            "zsa": np.corrcoef(np.log10(lsp.zsa[:, 0, :].cpu().numpy()).T)[0],
            "zsd": np.corrcoef(np.log10(lsp.zsd[:, 0, :].cpu().numpy()).T)[0],
        }
        if submodel == "los":
            measured["k"] = np.corrcoef(
                np.log10(lsp.k_factor[:, 0, :].cpu().numpy()).T
            )[0]

        correlation_by_name = {
            name: np.exp(-d_2d_ut / distance)
            for name, distance in corr_dist.items()
        }
        names = ["ds", "asd", "asa", "sf"]
        if submodel == "los":
            names.append("k")
        names.extend(["zsa", "zsd"])
        expected = _winner_spatial_correlation(
            _INDOOR_LSP_CROSS_CORR[(model, submodel)],
            names,
            correlation_by_name,
        )
        for name, measured_corr in measured.items():
            assert (
                np.max(np.abs(measured_corr - expected[name]))
                <= self.MAX_ERR_SPAT_CORR
            ), f"{model}:{submodel} {name} spatial correlation failed"
