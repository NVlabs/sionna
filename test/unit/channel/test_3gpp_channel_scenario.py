#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Tests for 3GPP TR 38.901 system level channel scenarios (RMa, UMa, UMi)"""

import warnings

import numpy as np
import pytest
import torch

from sionna.phy.channel import GenerateOFDMChannel, GenerateTimeChannel
from sionna.phy.channel.tr38901 import (
    InF,
    InFScenario,
    InH,
    InHScenario,
    LSPGenerator,
    PanelArray,
    RaysGenerator,
    RMaScenario,
    RMa,
    UMaScenario,
    UMa,
    UMiScenario,
    UMi,
)
from sionna.phy.channel.tr38901 import models
from channel_test_utils import generate_random_loc, generate_random_bool


# Test configuration
BATCH_SIZE = 16
CARRIER_FREQUENCY = 3.5e9  # Hz
MAX_ERR = 1e-2
H_UT = 1.5
H_BS = 10.0
NB_BS = 3
NB_UT = 10


def create_arrays(fc, device, precision="single"):
    """Create UT and BS panel arrays for testing."""
    bs_array = PanelArray(
        num_rows_per_panel=2,
        num_cols_per_panel=2,
        polarization="dual",
        polarization_type="VH",
        antenna_pattern="38.901",
        carrier_frequency=fc,
        precision=precision,
        device=device,
    )
    ut_array = PanelArray(
        num_rows_per_panel=1,
        num_cols_per_panel=1,
        polarization="dual",
        polarization_type="VH",
        antenna_pattern="38.901",
        carrier_frequency=fc,
        precision=precision,
        device=device,
    )
    return ut_array, bs_array


def set_single_link_topology(scenario, device, precision, h_bs=25.0,
                             h_ut=1.5, d_2d=100.0, indoor=False, los=True,
                             distance_2d_in=None):
    """Set one deterministic BS-UT link for parameter-table tests."""
    dtype = torch.float32 if precision == "single" else torch.float64
    scenario.set_topology(
        ut_loc=torch.tensor([[[d_2d, 0.0, h_ut]]], dtype=dtype, device=device),
        bs_loc=torch.tensor([[[0.0, 0.0, h_bs]]], dtype=dtype, device=device),
        ut_orientations=torch.zeros(1, 1, 3, dtype=dtype, device=device),
        bs_orientations=torch.zeros(1, 1, 3, dtype=dtype, device=device),
        ut_velocities=torch.zeros(1, 1, 3, dtype=dtype, device=device),
        in_state=torch.tensor([[indoor]], dtype=torch.bool, device=device),
        los=los,
        distance_2d_in=distance_2d_in,
    )


class _CompatibilityResourceGrid:
    """Minimal resource grid for system-channel wrapper coverage."""

    num_ofdm_symbols = 2
    fft_size = 4
    subcarrier_spacing = 15e3
    cyclic_prefix_length = 1

    @property
    def ofdm_symbol_duration(self):
        return (
            1.0 + self.cyclic_prefix_length / self.fft_size
        ) / self.subcarrier_spacing


class TestSystemLevelChannelCallCompatibility:
    """Tests for the generic channel-model call compatibility interface."""

    def test_call_forms_and_wrappers(self, device):
        """Test canonical, compatibility, wrapper, and error call forms."""

        ut_array, bs_array = create_arrays(CARRIER_FREQUENCY, device)
        channel = UMi(
            carrier_frequency=CARRIER_FREQUENCY,
            o2i_model="low",
            ut_array=ut_array,
            bs_array=bs_array,
            direction="downlink",
            device=device,
        )
        set_single_link_topology(channel, device, "single", h_bs=10.0)

        h, _ = channel(3, 1e6)
        assert h.shape[-1] == 3

        h, _ = channel(99, 3, 1e6)
        assert h.shape[-1] == 3

        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            h, _ = channel(99, 3, batch_size_compat=1e6)
        assert h.shape[-1] == 3

        with pytest.raises(TypeError, match="unexpected keyword argument 'foo'"):
            channel(99, 3, foo=1e6)
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            channel(3, 1e6, unknown_keyword=True)

        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            h_freq = GenerateOFDMChannel(
                channel,
                _CompatibilityResourceGrid(),
                device=device,
            )(batch_size=99)
            h_time = GenerateTimeChannel(
                channel,
                bandwidth=1e6,
                num_time_samples=3,
                l_min=0,
                l_max=1,
                device=device,
            )(batch_size=99)
        assert h_freq.shape[-2:] == (2, 4)
        assert h_time.shape[-2:] == (4, 2)


class TestSystemLevelChannelTopologyReset:
    """Tests for complete topology-cache invalidation."""

    def test_reset_clears_scenario_lsp_ray_and_blockage_caches(
        self, device, precision
    ):
        """Materialize and clear every layer of topology-dependent state."""

        dtype = torch.float32 if precision == "single" else torch.float64
        ut_array, bs_array = create_arrays(CARRIER_FREQUENCY, device, precision)
        channel = UMi(
            carrier_frequency=CARRIER_FREQUENCY,
            o2i_model="low",
            ut_array=ut_array,
            bs_array=bs_array,
            direction="downlink",
            enable_spatial_consistency=True,
            enable_blockage=True,
            blockage_self_blocking="portrait",
            precision=precision,
            device=device,
        )
        channel.set_topology(
            ut_loc=torch.tensor(
                [[[100.0, 0.0, 1.5], [120.0, 20.0, 1.5]]],
                dtype=dtype,
                device=device,
            ),
            bs_loc=torch.tensor(
                [[[0.0, 0.0, 10.0]]], dtype=dtype, device=device
            ),
            ut_orientations=torch.zeros(1, 2, 3, dtype=dtype, device=device),
            bs_orientations=torch.zeros(1, 1, 3, dtype=dtype, device=device),
            ut_velocities=torch.zeros(1, 2, 3, dtype=dtype, device=device),
            in_state=torch.zeros(1, 2, dtype=torch.bool, device=device),
            los=True,
        )
        channel(num_time_samples=1, sampling_frequency=1e6)

        scenario_cache_names = (
            "_ut_loc",
            "_bs_loc",
            "_bs_virtual_loc",
            "_bs_site_ids",
            "_bs_site_representatives",
            "_ut_orientations",
            "_bs_orientations",
            "_ut_velocities",
            "_in_state",
            "_ut_spatial_region_ids",
            "_distance_2d",
            "_distance_3d",
            "_raw_distance_2d_in",
            "_distance_2d_in",
            "_distance_2d_out",
            "_distance_3d_in",
            "_distance_3d_out",
            "_matrix_ut_distance_2d",
            "_los_aod",
            "_los_aoa",
            "_los_zod",
            "_los_zoa",
            "_outdoor_los",
            "_los",
            "_lsp_log_mean",
            "_lsp_log_std",
            "_zod_offset",
            "_pl_b",
        )
        lsp_cache_names = (
            "_cross_lsp_correlation_matrix_sqrt",
            "_spatial_lsp_correlation_matrix_sqrt",
        )
        ray_tensor_cache_names = (
            "_cluster_mask",
            "_spatial_consistency_matrix_sqrt",
            "_spatial_sign_consistency_matrix_sqrt",
        )
        blockage_cache_names = (
            "_matrix_sqrt",
            "_blocker_phi",
            "_blocker_x",
            "_blocker_y",
        )
        scenario = channel._scenario
        lsp_sampler = channel._lsp_sampler
        ray_sampler = channel._ray_sampler
        blockage_model = ray_sampler._blockage_model

        assert all(hasattr(scenario, name) for name in scenario_cache_names)
        assert all(hasattr(lsp_sampler, name) for name in lsp_cache_names)
        assert all(
            isinstance(getattr(ray_sampler, name), torch.Tensor)
            for name in ray_tensor_cache_names
        )
        assert blockage_model is not None
        assert all(
            isinstance(getattr(blockage_model, name), torch.Tensor)
            for name in blockage_cache_names
        )
        assert channel._lsp is not None
        assert channel._set_topology_called

        channel.reset_topology()

        assert not any(hasattr(scenario, name) for name in scenario_cache_names)
        assert scenario.spatial_consistency_track_ids is None
        assert scenario._requested_los is None
        assert not any(hasattr(lsp_sampler, name) for name in lsp_cache_names)
        assert not hasattr(ray_sampler, "_cluster_mask")
        assert ray_sampler._spatial_consistency_matrix_sqrt is None
        assert ray_sampler._spatial_sign_consistency_matrix_sqrt is None
        assert all(
            getattr(blockage_model, name) is None
            and name not in blockage_model._buffers
            for name in blockage_cache_names
        )
        assert channel._lsp is None
        assert not channel._set_topology_called

    def test_reset_clears_optional_state_before_shape_change(
        self, device, precision
    ):
        """Do not reuse track IDs or forced LOS after a topology reset."""
        dtype = torch.float32 if precision == "single" else torch.float64
        ut_array, bs_array = create_arrays(CARRIER_FREQUENCY, device, precision)
        channel = UMi(
            carrier_frequency=CARRIER_FREQUENCY,
            o2i_model="low",
            ut_array=ut_array,
            bs_array=bs_array,
            direction="downlink",
            enable_spatial_consistency=True,
            precision=precision,
            device=device,
        )
        channel.set_topology(
            ut_loc=torch.tensor(
                [[[50.0, 0.0, 1.5], [70.0, 0.0, 1.5]]],
                dtype=dtype,
                device=device,
            ),
            bs_loc=torch.tensor(
                [[[0.0, 0.0, 10.0]]], dtype=dtype, device=device
            ),
            ut_orientations=torch.zeros(1, 2, 3, dtype=dtype, device=device),
            bs_orientations=torch.zeros(1, 1, 3, dtype=dtype, device=device),
            ut_velocities=torch.zeros(1, 2, 3, dtype=dtype, device=device),
            in_state=torch.zeros(1, 2, dtype=torch.bool, device=device),
            los=torch.tensor([[[True, False]]], device=device),
            spatial_consistency_track_ids=torch.tensor(
                [[4, 4]], dtype=torch.int64, device=device
            ),
        )

        channel.reset_topology()
        channel.set_topology(
            ut_loc=torch.tensor(
                [[[50.0, 0.0, 1.5],
                  [70.0, 0.0, 1.5],
                  [90.0, 0.0, 1.5]]],
                dtype=dtype,
                device=device,
            ),
            bs_loc=torch.tensor(
                [[[0.0, 0.0, 10.0]]], dtype=dtype, device=device
            ),
            ut_orientations=torch.zeros(1, 3, 3, dtype=dtype, device=device),
            bs_orientations=torch.zeros(1, 1, 3, dtype=dtype, device=device),
            ut_velocities=torch.zeros(1, 3, 3, dtype=dtype, device=device),
            in_state=torch.zeros(1, 3, dtype=torch.bool, device=device),
        )

        assert channel._scenario.los.shape == (1, 1, 3)
        assert channel._scenario.spatial_consistency_track_ids is None
        assert channel._scenario._requested_los is None

    def test_preallocation_accepts_per_ut_virtual_bs_and_track_ids(
        self, device, precision
    ):
        """Preallocate every documented topology tensor shape."""
        dtype = torch.float32 if precision == "single" else torch.float64
        ut_array, bs_array = create_arrays(CARRIER_FREQUENCY, device, precision)
        scenario = UMiScenario(
            carrier_frequency=CARRIER_FREQUENCY,
            o2i_model="low",
            ut_array=ut_array,
            bs_array=bs_array,
            direction="downlink",
            precision=precision,
            device=device,
        )
        scenario.allocate_topology_tensors(batch_size=1, num_bs=3, num_ut=2)
        bs_loc = torch.tensor(
            [[[0.0, 0.0, 10.0], [200.0, 0.0, 10.0], [0.0, 200.0, 10.0]]],
            dtype=dtype,
            device=device,
        )
        bs_virtual_loc = bs_loc.unsqueeze(2).expand(-1, -1, 2, -1).clone()
        track_ids = torch.tensor([[7, 7]], dtype=torch.int64, device=device)
        scenario.set_topology(
            ut_loc=torch.tensor(
                [[[50.0, 0.0, 1.5], [60.0, 5.0, 1.5]]],
                dtype=dtype,
                device=device,
            ),
            bs_loc=bs_loc,
            ut_orientations=torch.zeros(1, 2, 3, dtype=dtype, device=device),
            bs_orientations=torch.zeros(1, 3, 3, dtype=dtype, device=device),
            ut_velocities=torch.zeros(1, 2, 3, dtype=dtype, device=device),
            in_state=torch.zeros(1, 2, dtype=torch.bool, device=device),
            los=True,
            bs_virtual_loc=bs_virtual_loc,
            spatial_consistency_track_ids=track_ids,
        )

        assert scenario.bs_virtual_loc.shape == (1, 3, 2, 3)
        torch.testing.assert_close(scenario.spatial_consistency_track_ids, track_ids)


class TestSpecVersion:
    """Tests for versioned TR 38.901 parameter tables."""

    def test_model_resource_paths_are_versioned(self):
        """Test table resources are resolved from versioned folders."""
        v16_path = models.parameter_file("UMi_LoS.json", "16.1")
        v19_path = models.parameter_file("UMi_LoS.json", "19.2")

        assert str(v16_path).endswith("models/v16_1/UMi_LoS.json")
        assert str(v19_path).endswith("models/v19_2/UMi_LoS.json")
        assert v16_path.is_file()
        assert v19_path.is_file()

    @pytest.mark.parametrize(
        "version",
        [
            None,
            16.1,
            "14.0",
            "16",
            "16.0",
            "16.1.0",
            "19",
            "19.0",
            "19.2.0",
            "v16.1",
            "V19.2",
            " 19.2 ",
        ],
    )
    def test_noncanonical_spec_versions_raise(self, version):
        """Test only exact supported release labels are accepted."""
        with pytest.raises(ValueError, match="16.1.*19.2"):
            models.parameter_file("UMi_LoS.json", version)

    def test_default_spec_version_is_v19_2(self, device, precision):
        """Test the latest parameter-table version is the default."""
        ut_array, bs_array = create_arrays(CARRIER_FREQUENCY, device, precision)
        scenario = UMiScenario(
            carrier_frequency=CARRIER_FREQUENCY,
            o2i_model="low",
            ut_array=ut_array,
            bs_array=bs_array,
            direction="downlink",
            precision=precision,
            device=device,
        )
        assert scenario.spec_version == "19.2"

    def test_invalid_spec_version_raises(self, device, precision):
        """Test unsupported TR 38.901 versions are rejected."""
        ut_array, bs_array = create_arrays(CARRIER_FREQUENCY, device, precision)
        with pytest.raises(ValueError, match="spec_version"):
            UMiScenario(
                carrier_frequency=CARRIER_FREQUENCY,
                o2i_model="low",
                ut_array=ut_array,
                bs_array=bs_array,
                direction="downlink",
                spec_version="18.0",
                precision=precision,
                device=device,
            )

    def test_v19_2_umi_los_parameters(self, device, precision):
        """Test UMi LOS table values updated by Rel-19."""
        fc = 6e9
        ut_array, bs_array = create_arrays(fc, device, precision)
        scenario = UMiScenario(
            carrier_frequency=fc,
            o2i_model="low",
            ut_array=ut_array,
            bs_array=bs_array,
            direction="downlink",
            spec_version="19.2",
            precision=precision,
            device=device,
        )
        set_single_link_topology(scenario, device, precision, h_bs=10.0,
                                 indoor=False, los=True)

        log_fc = np.log10(1.0 + 6.0)
        assert scenario.spec_version == "19.2"
        assert scenario.get_param("muDS")[0, 0, 0].item() == pytest.approx(
            -0.18*log_fc - 7.28
        )
        assert scenario.get_param("sigmaASD")[0, 0, 0].item() == pytest.approx(
            0.08*log_fc + 0.29
        )
        assert scenario.get_param("muASA")[0, 0, 0].item() == pytest.approx(
            -0.07*log_fc + 1.66
        )

    def test_v19_2_umi_nlos_parameters(self, device, precision):
        """Test UMi NLOS table values updated by Rel-19."""
        fc = 6e9
        ut_array, bs_array = create_arrays(fc, device, precision)
        scenario = UMiScenario(
            carrier_frequency=fc,
            o2i_model="low",
            ut_array=ut_array,
            bs_array=bs_array,
            direction="downlink",
            spec_version="19.2",
            precision=precision,
            device=device,
        )
        set_single_link_topology(scenario, device, precision, h_bs=10.0,
                                 indoor=False, los=False)

        log_fc = np.log10(1.0 + 6.0)
        assert scenario.get_param("muDS")[0, 0, 0].item() == pytest.approx(
            -0.22*log_fc - 6.87
        )
        assert scenario.get_param("sigmaZSA")[0, 0, 0].item() == pytest.approx(
            -0.05*log_fc + 0.35
        )

    def test_v19_2_uma_parameters(self, device, precision):
        """Test UMa LOS/NLOS/O2I table values updated by Rel-19."""
        fc = 6e9
        ut_array, bs_array = create_arrays(fc, device, precision)
        scenario = UMaScenario(
            carrier_frequency=fc,
            o2i_model="low",
            ut_array=ut_array,
            bs_array=bs_array,
            direction="downlink",
            spec_version="19.2",
            precision=precision,
            device=device,
        )

        set_single_link_topology(scenario, device, precision, h_bs=25.0,
                                 indoor=False, los=True)
        log_fc = np.log10(6.0)
        assert scenario.get_param("muDS")[0, 0, 0].item() == pytest.approx(
            -7.067 - 0.0794*log_fc
        )
        assert scenario.get_param("sigmaDS")[0, 0, 0].item() == pytest.approx(
            0.57 + 0.026*log_fc
        )
        assert scenario.get_param("cASD")[0, 0, 0].item() == pytest.approx(3.58)

        scenario.reset_topology()
        set_single_link_topology(scenario, device, precision, h_bs=25.0,
                                 indoor=False, los=False)
        assert scenario.get_param("muASD")[0, 0, 0].item() == pytest.approx(1.09)
        assert scenario.get_param("sigmaASD")[0, 0, 0].item() == pytest.approx(0.44)
        assert scenario.get_param("cASD")[0, 0, 0].item() == pytest.approx(1.8)

        scenario.reset_topology()
        set_single_link_topology(scenario, device, precision, h_bs=25.0,
                                 indoor=True, los=True)
        assert scenario.get_param("muASD")[0, 0, 0].item() == pytest.approx(0.58)
        assert scenario.get_param("sigmaASD")[0, 0, 0].item() == pytest.approx(0.7)

    def test_v19_2_rma_parameters_and_o2i_state(self, device, precision):
        """Test RMa Rel-19 ZSA/ZSD and dedicated O2I state handling."""
        fc = 6e9
        ut_array, bs_array = create_arrays(fc, device, precision)
        scenario = RMaScenario(
            carrier_frequency=fc,
            ut_array=ut_array,
            bs_array=bs_array,
            direction="downlink",
            spec_version="19.2",
            precision=precision,
            device=device,
        )

        set_single_link_topology(scenario, device, precision, h_bs=35.0,
                                 h_ut=1.5, d_2d=100.0, indoor=False, los=True)
        expected_los_zsd = max(-1.0, -0.17*0.1 + 0.22)
        assert scenario.lsp_log_mean[0, 0, 0, 5].item() == pytest.approx(0.47)
        assert scenario.lsp_log_mean[0, 0, 0, 6].item() == pytest.approx(
            expected_los_zsd
        )
        assert scenario.lsp_log_std[0, 0, 0, 5].item() == pytest.approx(0.40)
        assert scenario.lsp_log_std[0, 0, 0, 6].item() == pytest.approx(0.34)

        scenario.reset_topology()
        set_single_link_topology(scenario, device, precision, h_bs=35.0,
                                 h_ut=1.5, d_2d=100.0, indoor=False, los=False)
        expected_nlos_zsd = max(-1.0, -0.19*0.1 + 0.28)
        expected_zod_offset = np.degrees(
            np.arctan(31.5/100.0) - np.arctan(33.5/100.0)
        )
        zod_atol = 2e-6 if precision == "single" else 1e-12
        assert scenario.lsp_log_mean[0, 0, 0, 5].item() == pytest.approx(0.58)
        assert scenario.lsp_log_mean[0, 0, 0, 6].item() == pytest.approx(
            expected_nlos_zsd
        )
        assert scenario.lsp_log_std[0, 0, 0, 5].item() == pytest.approx(0.37)
        assert scenario.lsp_log_std[0, 0, 0, 6].item() == pytest.approx(0.30)
        assert scenario.zod_offset[0, 0, 0].item() == pytest.approx(
            expected_zod_offset, abs=zod_atol
        )

        scenario.reset_topology()
        set_single_link_topology(scenario, device, precision, h_bs=35.0,
                                 h_ut=1.5, d_2d=100.0, indoor=True, los=True)
        assert scenario.los[0, 0, 0].item() is False
        assert scenario.outdoor_los[0, 0, 0].item() is True
        assert scenario.get_param("muDSc")[0, 0, 0].item() == pytest.approx(-7.47)
        assert scenario.get_param("muASD")[0, 0, 0].item() == pytest.approx(0.67)
        assert scenario.lsp_log_mean[0, 0, 0, 6].item() == pytest.approx(
            expected_nlos_zsd
        )
        assert scenario.zod_offset[0, 0, 0].item() == pytest.approx(
            expected_zod_offset, abs=zod_atol
        )

    def test_v16_1_rma_parameters_and_o2i_state(self, device, precision):
        """Test RMa V16.1 corrected ZSA/ZSD and dedicated O2I state handling."""
        fc = 6e9
        ut_array, bs_array = create_arrays(fc, device, precision)
        scenario = RMaScenario(
            carrier_frequency=fc,
            ut_array=ut_array,
            bs_array=bs_array,
            direction="downlink",
            spec_version="16.1",
            precision=precision,
            device=device,
        )

        set_single_link_topology(scenario, device, precision, h_bs=35.0,
                                 h_ut=1.5, d_2d=100.0, indoor=False, los=True)
        expected_los_zsd = max(-1.0, -0.17*0.1 + 0.22)
        assert scenario.lsp_log_mean[0, 0, 0, 5].item() == pytest.approx(0.47)
        assert scenario.lsp_log_mean[0, 0, 0, 6].item() == pytest.approx(
            expected_los_zsd
        )
        assert scenario.lsp_log_std[0, 0, 0, 5].item() == pytest.approx(0.40)
        assert scenario.lsp_log_std[0, 0, 0, 6].item() == pytest.approx(0.34)

        scenario.reset_topology()
        set_single_link_topology(scenario, device, precision, h_bs=35.0,
                                 h_ut=1.5, d_2d=100.0, indoor=False, los=False)
        expected_nlos_zsd = max(-1.0, -0.19*0.1 + 0.28)
        expected_zod_offset = np.degrees(
            np.arctan(31.5/100.0) - np.arctan(33.5/100.0)
        )
        zod_atol = 2e-6 if precision == "single" else 1e-12
        assert scenario.lsp_log_mean[0, 0, 0, 5].item() == pytest.approx(0.58)
        assert scenario.lsp_log_mean[0, 0, 0, 6].item() == pytest.approx(
            expected_nlos_zsd
        )
        assert scenario.lsp_log_std[0, 0, 0, 5].item() == pytest.approx(0.37)
        assert scenario.lsp_log_std[0, 0, 0, 6].item() == pytest.approx(0.30)
        assert scenario.zod_offset[0, 0, 0].item() == pytest.approx(
            expected_zod_offset, abs=zod_atol
        )

        scenario.reset_topology()
        set_single_link_topology(scenario, device, precision, h_bs=35.0,
                                 h_ut=1.5, d_2d=100.0, indoor=True, los=True)
        assert scenario.los[0, 0, 0].item() is False
        assert scenario.outdoor_los[0, 0, 0].item() is True
        assert scenario.get_param("muDSc")[0, 0, 0].item() == pytest.approx(-7.47)
        assert scenario.get_param("muASD")[0, 0, 0].item() == pytest.approx(0.67)
        assert scenario.lsp_log_mean[0, 0, 0, 6].item() == pytest.approx(
            expected_nlos_zsd
        )
        assert scenario.zod_offset[0, 0, 0].item() == pytest.approx(
            expected_zod_offset, abs=zod_atol
        )

    @pytest.mark.parametrize("spec_version", ["16.1", "19.2"])
    def test_inh_parameters(self, device, precision, spec_version):
        """Test V16.1/V19.2 Indoor-Office table values."""
        fc = 6e9
        ut_array, bs_array = create_arrays(fc, device, precision)
        scenario = InHScenario(
            carrier_frequency=fc,
            indoor_scenario="open",
            ut_array=ut_array,
            bs_array=bs_array,
            direction="downlink",
            spec_version=spec_version,
            precision=precision,
            device=device,
        )

        set_single_link_topology(scenario, device, precision, h_bs=3.0,
                                 h_ut=1.0, d_2d=10.0, indoor=True, los=True)
        log_fc = np.log10(1.0 + 6.0)
        assert scenario.spec_version == spec_version
        assert scenario.get_param("muDS")[0, 0, 0].item() == pytest.approx(
            -0.01*log_fc - 7.692
        )
        assert scenario.get_param("sigmaDS")[0, 0, 0].item() == pytest.approx(0.18)
        assert scenario.get_param("muK")[0, 0, 0].item() == pytest.approx(7.0)
        assert scenario.get_param("muXPR")[0, 0, 0].item() == pytest.approx(11.0)
        assert scenario.get_param("numClusters")[0, 0, 0].item() == pytest.approx(15)
        assert scenario.get_param("rTau")[0, 0, 0].item() == pytest.approx(3.6)
        assert scenario.get_param("cDS")[0, 0, 0].item() == pytest.approx(3.91)
        assert scenario.get_param("cASA")[0, 0, 0].item() == pytest.approx(8.0)
        assert scenario.lsp_log_mean[0, 0, 0, 6].item() == pytest.approx(
            -1.43*log_fc + 2.228
        )
        assert scenario.lsp_log_std[0, 0, 0, 6].item() == pytest.approx(
            0.13*log_fc + 0.30
        )

        scenario.reset_topology()
        set_single_link_topology(scenario, device, precision, h_bs=3.0,
                                 h_ut=1.0, d_2d=10.0, indoor=True, los=False)
        assert scenario.get_param("muDS")[0, 0, 0].item() == pytest.approx(
            -0.28*log_fc - 7.173
        )
        assert scenario.get_param("sigmaASD")[0, 0, 0].item() == pytest.approx(0.25)
        assert scenario.get_param("muXPR")[0, 0, 0].item() == pytest.approx(10.0)
        assert scenario.get_param("numClusters")[0, 0, 0].item() == pytest.approx(19)
        assert scenario.get_param("cASA")[0, 0, 0].item() == pytest.approx(11.0)
        assert scenario.lsp_log_mean[0, 0, 0, 6].item() == pytest.approx(1.08)
        assert scenario.lsp_log_std[0, 0, 0, 6].item() == pytest.approx(0.36)


class TestTopologyLosState:
    """Tests for deterministic per-link LoS state control."""

    def test_tensor_los_state(self, device, precision):
        """Test explicit per-link LoS/NLoS tensor support."""

        dtype = torch.float32 if precision == "single" else torch.float64
        ut_array, bs_array = create_arrays(CARRIER_FREQUENCY, device, precision)
        scenario = UMiScenario(
            carrier_frequency=CARRIER_FREQUENCY,
            o2i_model="low",
            ut_array=ut_array,
            bs_array=bs_array,
            direction="downlink",
            precision=precision,
            device=device,
        )
        ut_loc = torch.tensor(
            [
                [[50.0, 0.0, 1.5], [80.0, 0.0, 1.5], [120.0, 0.0, 1.5]],
                [[50.0, 10.0, 1.5], [80.0, 10.0, 1.5], [120.0, 10.0, 1.5]],
            ],
            dtype=dtype,
            device=device,
        )
        bs_loc = torch.tensor(
            [
                [[0.0, 0.0, 10.0], [200.0, 0.0, 10.0]],
                [[0.0, 0.0, 10.0], [200.0, 0.0, 10.0]],
            ],
            dtype=dtype,
            device=device,
        )
        in_state = torch.tensor(
            [[False, True, False], [False, False, True]],
            dtype=torch.bool,
            device=device,
        )
        requested_los = torch.tensor(
            [
                [[True, True, False], [False, True, True]],
                [[False, True, True], [True, False, True]],
            ],
            dtype=torch.bool,
            device=device,
        )
        scenario.set_topology(
            ut_loc=ut_loc,
            bs_loc=bs_loc,
            ut_orientations=torch.zeros(2, 3, 3, dtype=dtype, device=device),
            bs_orientations=torch.zeros(2, 2, 3, dtype=dtype, device=device),
            ut_velocities=torch.zeros(2, 3, 3, dtype=dtype, device=device),
            in_state=in_state,
            los=requested_los,
        )

        expected = requested_los & (~in_state.unsqueeze(1))
        torch.testing.assert_close(scenario.outdoor_los, requested_los)
        torch.testing.assert_close(scenario.los, expected)

    def test_o2i_uses_outdoor_los_for_pathloss_and_zsd(
        self, device, precision
    ):
        """Keep O2I channel NLoS while using the outdoor LoS condition."""
        dtype = torch.float32 if precision == "single" else torch.float64
        fc = 3.5e9
        ut_array, bs_array = create_arrays(fc, device, precision)
        scenario = UMiScenario(
            carrier_frequency=fc,
            o2i_model="low",
            ut_array=ut_array,
            bs_array=bs_array,
            direction="downlink",
            precision=precision,
            device=device,
        )
        indoor_distance = torch.tensor([[5.0]], dtype=dtype, device=device)
        set_single_link_topology(
            scenario,
            device,
            precision,
            h_bs=10.0,
            d_2d=100.0,
            indoor=True,
            los=True,
            distance_2d_in=indoor_distance,
        )

        d_3d = np.sqrt(100.0**2 + (10.0 - 1.5) ** 2)
        expected_pathloss = (
            32.4 + 21.0*np.log10(d_3d) + 20.0*np.log10(fc/1e9)
        )
        expected_zsd = max(
            -0.21,
            -14.8*(100.0/1000.0) + 0.01*abs(1.5 - 10.0) + 0.83,
        )
        assert not scenario.los.item()
        assert scenario.outdoor_los.item()
        assert scenario.basic_pathloss.item() == pytest.approx(
            expected_pathloss, abs=1e-4
        )
        assert scenario.lsp_log_mean[..., 6].item() == pytest.approx(
            expected_zsd, abs=1e-5
        )
        assert scenario.zod_offset.item() == pytest.approx(0.0)

    def test_tensor_los_state_without_batch_dimension(self, device, precision):
        """Test a [num_bs, num_ut] LoS tensor is broadcast over the batch."""

        dtype = torch.float32 if precision == "single" else torch.float64
        ut_array, bs_array = create_arrays(CARRIER_FREQUENCY, device, precision)
        scenario = InHScenario(
            carrier_frequency=CARRIER_FREQUENCY,
            indoor_scenario="open",
            ut_array=ut_array,
            bs_array=bs_array,
            direction="downlink",
            precision=precision,
            device=device,
        )
        requested_los = torch.tensor(
            [[True, False], [False, True]],
            dtype=torch.bool,
            device=device,
        )
        scenario.set_topology(
            ut_loc=torch.tensor(
                [
                    [[10.0, 0.0, 1.0], [20.0, 0.0, 1.0]],
                    [[10.0, 5.0, 1.0], [20.0, 5.0, 1.0]],
                ],
                dtype=dtype,
                device=device,
            ),
            bs_loc=torch.tensor(
                [
                    [[0.0, 0.0, 3.0], [40.0, 0.0, 3.0]],
                    [[0.0, 0.0, 3.0], [40.0, 0.0, 3.0]],
                ],
                dtype=dtype,
                device=device,
            ),
            ut_orientations=torch.zeros(2, 2, 3, dtype=dtype, device=device),
            bs_orientations=torch.zeros(2, 2, 3, dtype=dtype, device=device),
            ut_velocities=torch.zeros(2, 2, 3, dtype=dtype, device=device),
            in_state=torch.ones(2, 2, dtype=torch.bool, device=device),
            los=requested_los,
        )

        expected = requested_los.unsqueeze(0).expand(2, -1, -1)
        torch.testing.assert_close(scenario.los, expected)

    def test_invalid_tensor_los_shape_raises(self, device, precision):
        """Test explicit LoS tensors must match the configured topology."""

        dtype = torch.float32 if precision == "single" else torch.float64
        ut_array, bs_array = create_arrays(CARRIER_FREQUENCY, device, precision)
        scenario = UMiScenario(
            carrier_frequency=CARRIER_FREQUENCY,
            o2i_model="low",
            ut_array=ut_array,
            bs_array=bs_array,
            direction="downlink",
            precision=precision,
            device=device,
        )
        with pytest.raises(ValueError, match="los.*shape"):
            scenario.set_topology(
                ut_loc=torch.tensor(
                    [[[50.0, 0.0, 1.5]]], dtype=dtype, device=device
                ),
                bs_loc=torch.tensor(
                    [[[0.0, 0.0, 10.0]]], dtype=dtype, device=device
                ),
                ut_orientations=torch.zeros(1, 1, 3, dtype=dtype, device=device),
                bs_orientations=torch.zeros(1, 1, 3, dtype=dtype, device=device),
                ut_velocities=torch.zeros(1, 1, 3, dtype=dtype, device=device),
                in_state=torch.zeros(1, 1, dtype=torch.bool, device=device),
                los=torch.ones(1, 1, 2, dtype=torch.bool, device=device),
            )


class TestInHScenario:
    """Tests for InHScenario"""

    def _make_scenario(self, device, precision, indoor_scenario="open"):
        ut_array, bs_array = create_arrays(CARRIER_FREQUENCY, device, precision)
        return InHScenario(
            carrier_frequency=CARRIER_FREQUENCY,
            indoor_scenario=indoor_scenario,
            ut_array=ut_array,
            bs_array=bs_array,
            direction="downlink",
            precision=precision,
            device=device,
        )

    def test_instantiation(self, device, precision):
        """Test InHScenario can be instantiated"""
        scenario = self._make_scenario(device, precision)
        assert scenario is not None
        assert scenario.indoor_scenario == "open"
        assert scenario.carrier_frequency.item() == pytest.approx(CARRIER_FREQUENCY)

    @pytest.mark.parametrize(
        "indoor_scenario, distances, expected",
        [
            (
                "open",
                [1.0, 10.0, 60.0],
                [
                    1.0,
                    np.exp(-(10.0 - 5.0) / 70.8),
                    np.exp(-(60.0 - 49.0) / 211.7) * 0.54,
                ],
            ),
            (
                "mixed",
                [1.0, 2.0, 10.0],
                [
                    1.0,
                    np.exp(-(2.0 - 1.2) / 4.7),
                    np.exp(-(10.0 - 6.5) / 32.6) * 0.32,
                ],
            ),
        ],
    )
    def test_los_probability(
        self, device, precision, indoor_scenario, distances, expected
    ):
        """Test InH open/mixed office LoS probabilities."""
        scenario = self._make_scenario(device, precision, indoor_scenario)
        dtype = torch.float32 if precision == "single" else torch.float64
        ut_loc = torch.tensor(
            [[[d, 0.0, 1.0] for d in distances]], dtype=dtype, device=device
        )
        bs_loc = torch.tensor([[[0.0, 0.0, 3.0]]], dtype=dtype, device=device)
        scenario.set_topology(
            ut_loc=ut_loc,
            bs_loc=bs_loc,
            ut_orientations=torch.zeros(1, len(distances), 3, dtype=dtype, device=device),
            bs_orientations=torch.zeros(1, 1, 3, dtype=dtype, device=device),
            ut_velocities=torch.zeros(1, len(distances), 3, dtype=dtype, device=device),
            in_state=torch.ones(1, len(distances), dtype=torch.bool, device=device),
            los=True,
        )

        np.testing.assert_allclose(
            scenario.los_probability.cpu().numpy()[0, 0],
            np.asarray(expected),
            rtol=1e-5,
            atol=1e-6,
        )

    @pytest.mark.parametrize("los", [True, False])
    def test_pathloss_no_o2i_penetration(self, device, precision, los):
        """Test InH pathloss does not add O2I penetration for indoor UTs."""
        fc = 6e9
        ut_array, bs_array = create_arrays(fc, device, precision)
        scenario = InHScenario(
            carrier_frequency=fc,
            indoor_scenario="open",
            ut_array=ut_array,
            bs_array=bs_array,
            direction="downlink",
            precision=precision,
            device=device,
        )
        dtype = torch.float32 if precision == "single" else torch.float64
        d_2d = 10.0
        h_bs = 3.0
        h_ut = 1.0
        d_3d = np.sqrt(d_2d**2 + (h_bs - h_ut) ** 2)
        scenario.set_topology(
            ut_loc=torch.tensor([[[d_2d, 0.0, h_ut]]], dtype=dtype, device=device),
            bs_loc=torch.tensor([[[0.0, 0.0, h_bs]]], dtype=dtype, device=device),
            ut_orientations=torch.zeros(1, 1, 3, dtype=dtype, device=device),
            bs_orientations=torch.zeros(1, 1, 3, dtype=dtype, device=device),
            ut_velocities=torch.zeros(1, 1, 3, dtype=dtype, device=device),
            in_state=torch.ones(1, 1, dtype=torch.bool, device=device),
            los=los,
        )

        pl_los = 32.4 + 17.3 * np.log10(d_3d) + 20.0 * np.log10(fc / 1e9)
        pl_nlos_prime = (
            38.3 * np.log10(d_3d) + 17.30 + 24.9 * np.log10(fc / 1e9)
        )
        expected = pl_los if los else max(pl_los, pl_nlos_prime)
        pathloss = LSPGenerator(scenario).sample_pathloss()
        assert pathloss[0, 0, 0].item() == pytest.approx(expected, abs=1e-4)

    def test_channel_default_indoor_state(self, device, precision):
        """Test public InH channel assumes indoor UTs if in_state is omitted."""
        fc = CARRIER_FREQUENCY
        ut_array, bs_array = create_arrays(fc, device, precision)
        channel = InH(
            carrier_frequency=fc,
            ut_array=ut_array,
            bs_array=bs_array,
            direction="downlink",
            precision=precision,
            device=device,
        )
        dtype = torch.float32 if precision == "single" else torch.float64
        channel.set_topology(
            ut_loc=torch.tensor([[[10.0, 0.0, 1.0]]], dtype=dtype, device=device),
            bs_loc=torch.tensor([[[0.0, 0.0, 3.0]]], dtype=dtype, device=device),
            ut_orientations=torch.zeros(1, 1, 3, dtype=dtype, device=device),
            bs_orientations=torch.zeros(1, 1, 3, dtype=dtype, device=device),
            ut_velocities=torch.zeros(1, 1, 3, dtype=dtype, device=device),
            los=True,
        )

        assert torch.all(channel._scenario.indoor)
        a, tau = channel(num_time_samples=4, sampling_frequency=1e6)
        assert a.shape[-1] == 4
        assert tau.shape[-1] == 23

    def test_indoor_links_use_inh_zenith_model(self, device, precision):
        """Test InH indoor LoS links use geometric ZoA instead of O2I ZoA."""
        scenario = self._make_scenario(device, precision)
        dtype = torch.float32 if precision == "single" else torch.float64
        scenario.set_topology(
            ut_loc=torch.tensor([[[10.0, 0.0, 1.0]]], dtype=dtype, device=device),
            bs_loc=torch.tensor([[[0.0, 0.0, 3.0]]], dtype=dtype, device=device),
            ut_orientations=torch.zeros(1, 1, 3, dtype=dtype, device=device),
            bs_orientations=torch.zeros(1, 1, 3, dtype=dtype, device=device),
            ut_velocities=torch.zeros(1, 1, 3, dtype=dtype, device=device),
            in_state=torch.ones(1, 1, dtype=torch.bool, device=device),
            los=True,
        )

        powers = torch.full(
            (1, 1, 1, scenario.num_clusters_max),
            0.01,
            dtype=dtype,
            device=device,
        )
        powers[..., 0] = 1.0
        spread = torch.full((1, 1, 1), 10.0, dtype=dtype, device=device)
        k_factor = torch.full((1, 1, 1), 10.0, dtype=dtype, device=device)
        zoa = RaysGenerator(scenario)._zenith_angles(
            spread, k_factor, powers, "zoa"
        )

        first_cluster_center = 0.5 * (zoa[0, 0, 0, 0, 0] + zoa[0, 0, 0, 0, 1])
        assert first_cluster_center.item() == pytest.approx(
            scenario.los_zoa[0, 0, 0].item(), abs=1e-5
        )
        assert first_cluster_center.item() != pytest.approx(90.0, abs=1.0)


class TestInFScenario:
    """Tests for InFScenario."""

    def _make_scenario(self, device, precision, factory_scenario="SH",
                       fc=CARRIER_FREQUENCY):
        ut_array, bs_array = create_arrays(fc, device, precision)
        return InFScenario(
            carrier_frequency=fc,
            factory_scenario=factory_scenario,
            ut_array=ut_array,
            bs_array=bs_array,
            direction="downlink",
            precision=precision,
            device=device,
        )

    def test_instantiation(self, device, precision):
        """Test InFScenario can be instantiated."""
        scenario = self._make_scenario(device, precision)
        assert scenario.factory_scenario == "sh"
        assert scenario.spec_version == "19.2"
        assert scenario.hall_dimensions.shape == (3,)

    def test_v16_1_instantiation(self, device, precision):
        """Test InFScenario accepts TR 38.901 V16.1 parameter tables."""
        scenario = self._make_scenario(device, precision)
        scenario = InFScenario(
            carrier_frequency=CARRIER_FREQUENCY,
            factory_scenario="SH",
            ut_array=scenario.ut_array,
            bs_array=scenario.bs_array,
            direction="downlink",
            spec_version="16.1",
            precision=precision,
            device=device,
        )
        assert scenario.factory_scenario == "sh"
        assert scenario.spec_version == "16.1"

    def test_rejects_removed_version(self, device, precision):
        """Test removed parameter-table versions are rejected."""
        ut_array, bs_array = create_arrays(CARRIER_FREQUENCY, device, precision)
        with pytest.raises(ValueError, match="16.1.*19.2"):
            InFScenario(
                carrier_frequency=CARRIER_FREQUENCY,
                factory_scenario="SH",
                ut_array=ut_array,
                bs_array=bs_array,
                direction="downlink",
                spec_version="14.0",
                precision=precision,
                device=device,
            )

    @pytest.mark.parametrize(
        "factory_scenario,distance,h_bs,h_ut,expected",
        [
            ("SL", 10.0, 1.5, 1.5, np.exp(-10.0/(-10.0/np.log1p(-0.2)))),
            ("DL", 10.0, 1.5, 1.5, np.exp(-10.0/(-2.0/np.log1p(-0.6)))),
            (
                "SH",
                10.0,
                8.0,
                1.5,
                np.exp(-10.0/((-10.0/np.log1p(-0.2))*(8.0-1.5)/(2.0-1.5))),
            ),
            (
                "DH",
                10.0,
                8.0,
                1.5,
                np.exp(-10.0/((-2.0/np.log1p(-0.6))*(8.0-1.5)/(6.0-1.5))),
            ),
            ("HH", 10.0, 8.0, 8.0, 1.0),
        ],
    )
    def test_los_probability(self, device, precision, factory_scenario,
                             distance, h_bs, h_ut, expected):
        """Test InF LOS probabilities from Table 7.4.2-1."""
        scenario = self._make_scenario(device, precision, factory_scenario)
        set_single_link_topology(
            scenario, device, precision, h_bs=h_bs, h_ut=h_ut,
            d_2d=distance, indoor=True, los="random",
        )

        assert scenario.los_probability[0, 0, 0].item() == pytest.approx(
            expected, rel=1e-5
        )

    def test_hh_rejects_tensor_forced_nlos(self, device, precision):
        """Reject per-link forced NLoS states for the LOS-only HH model."""
        scenario = self._make_scenario(device, precision, "HH")
        forced_nlos = torch.zeros(1, 1, 1, dtype=torch.bool, device=device)
        with pytest.raises(ValueError, match="LOS-only.*forced NLOS"):
            scenario.set_topology(los=forced_nlos)

    @pytest.mark.parametrize(
        "factory_scenario,los",
        [
            ("SL", True),
            ("SL", False),
            ("DL", False),
            ("SH", False),
            ("DH", False),
        ],
    )
    def test_pathloss_models(self, device, precision, factory_scenario, los):
        """Test InF pathloss formulas from Table 7.4.1-1."""
        fc = 3.5e9
        scenario = self._make_scenario(device, precision, factory_scenario, fc)
        d_2d = 20.0
        h_bs = 8.0
        h_ut = 1.5
        d_3d = np.sqrt(d_2d**2 + (h_bs-h_ut)**2)
        set_single_link_topology(
            scenario, device, precision, h_bs=h_bs, h_ut=h_ut,
            d_2d=d_2d, indoor=True, los=los,
        )

        fc_ghz = fc/1e9
        pl_los = 31.84 + 21.50*np.log10(d_3d) + 19.00*np.log10(fc_ghz)
        pl_sl = 33.0 + 25.5*np.log10(d_3d) + 20.0*np.log10(fc_ghz)
        pl_dl = 18.6 + 35.7*np.log10(d_3d) + 20.0*np.log10(fc_ghz)
        pl_sh = 32.4 + 23.0*np.log10(d_3d) + 20.0*np.log10(fc_ghz)
        pl_dh = 33.63 + 21.9*np.log10(d_3d) + 20.0*np.log10(fc_ghz)
        nlos = {
            "SL": max(pl_los, pl_sl),
            "DL": max(pl_los, pl_sl, pl_dl),
            "SH": max(pl_los, pl_sh),
            "DH": max(pl_los, pl_dh),
        }[factory_scenario]
        expected = pl_los if los else nlos

        pathloss = LSPGenerator(scenario).sample_pathloss()
        assert pathloss[0, 0, 0].item() == pytest.approx(expected, abs=1e-4)

    @pytest.mark.parametrize("los", [True, False])
    def test_lsp_parameters(self, device, precision, los):
        """Test InF LSP means and standard deviations from Table 7.5-6."""
        fc = 3.5e9
        scenario = self._make_scenario(device, precision, "SH", fc)
        set_single_link_topology(
            scenario, device, precision, h_bs=8.0, h_ut=1.5,
            d_2d=20.0, indoor=True, los=los,
        )

        length, width, height = (300.0, 150.0, 10.0)
        volume = length*width*height
        surface = 2.0*(length*width + length*height + width*height)
        ratio = volume/surface
        log_fc = np.log10(1.0 + fc/1e9)
        expected_mu_ds = (
            np.log10(26.0*ratio + 14.0) - 9.35 if los
            else np.log10(30.0*ratio + 32.0) - 9.44
        )
        expected_mu_zsa = -0.2*log_fc + 1.5 if los else -0.13*log_fc + 1.45
        expected_sigma_sf = 0.43 if los else 0.59

        assert scenario.lsp_log_mean[0, 0, 0, 0].item() == pytest.approx(
            expected_mu_ds
        )
        assert scenario.lsp_log_mean[0, 0, 0, 5].item() == pytest.approx(
            expected_mu_zsa
        )
        assert scenario.lsp_log_mean[0, 0, 0, 6].item() == pytest.approx(
            1.35 if los else 1.2
        )
        assert scenario.lsp_log_std[0, 0, 0, 3].item() == pytest.approx(
            expected_sigma_sf
        )
        assert scenario.zod_offset[0, 0, 0].item() == pytest.approx(0.0)

    def test_channel_default_indoor_state(self, device, precision):
        """Test public InF channel assumes indoor UTs if in_state is omitted."""
        fc = CARRIER_FREQUENCY
        ut_array, bs_array = create_arrays(fc, device, precision)
        channel = InF(
            carrier_frequency=fc,
            ut_array=ut_array,
            bs_array=bs_array,
            direction="downlink",
            factory_scenario="SH",
            precision=precision,
            device=device,
        )
        dtype = torch.float32 if precision == "single" else torch.float64
        channel.set_topology(
            ut_loc=torch.tensor([[[10.0, 0.0, 1.5]]], dtype=dtype, device=device),
            bs_loc=torch.tensor([[[0.0, 0.0, 8.0]]], dtype=dtype, device=device),
            ut_orientations=torch.zeros(1, 1, 3, dtype=dtype, device=device),
            bs_orientations=torch.zeros(1, 1, 3, dtype=dtype, device=device),
            ut_velocities=torch.zeros(1, 1, 3, dtype=dtype, device=device),
            los=True,
        )

        assert torch.all(channel._scenario.indoor)
        a, tau = channel(num_time_samples=4, sampling_frequency=1e6)
        assert a.shape[-1] == 4
        assert tau.shape[-1] == 29


class TestRMaScenario:
    """Tests for RMaScenario"""

    def test_instantiation(self, device, precision):
        """Test RMaScenario can be instantiated"""
        ut_array, bs_array = create_arrays(CARRIER_FREQUENCY, device, precision)
        scenario = RMaScenario(
            carrier_frequency=CARRIER_FREQUENCY,
            ut_array=ut_array,
            bs_array=bs_array,
            direction="uplink",
            precision=precision,
            device=device,
        )
        assert scenario is not None
        assert scenario.carrier_frequency.item() == pytest.approx(CARRIER_FREQUENCY)

    def test_distance_calculation(self, device, precision):
        """Test distance calculations (2D and 3D)"""
        ut_array, bs_array = create_arrays(CARRIER_FREQUENCY, device, precision)
        scenario = RMaScenario(
            carrier_frequency=CARRIER_FREQUENCY,
            ut_array=ut_array,
            bs_array=bs_array,
            direction="uplink",
            precision=precision,
            device=device,
        )

        dtype = torch.float32 if precision == "single" else torch.float64

        # Generate random locations
        ut_loc = generate_random_loc(
            BATCH_SIZE, NB_UT, (100, 2000), (100, 2000), (H_UT, H_UT),
            dtype=dtype, device=device
        )
        bs_loc = generate_random_loc(
            BATCH_SIZE, NB_BS, (0, 100), (0, 100), (H_BS, H_BS),
            dtype=dtype, device=device
        )
        ut_orientations = torch.zeros(BATCH_SIZE, NB_UT, 3, dtype=dtype, device=device)
        bs_orientations = torch.zeros(BATCH_SIZE, NB_BS, 3, dtype=dtype, device=device)
        ut_velocities = torch.zeros(BATCH_SIZE, NB_UT, 3, dtype=dtype, device=device)
        in_state = generate_random_bool(BATCH_SIZE, NB_UT, 0.5, device=device)

        scenario.set_topology(
            ut_loc, bs_loc, ut_orientations, bs_orientations, ut_velocities, in_state
        )

        # Get distances
        d_3d = scenario.distance_3d
        d_3d_in = scenario.distance_3d_in
        d_3d_out = scenario.distance_3d_out
        d_2d = scenario.distance_2d
        d_2d_in = scenario.distance_2d_in
        d_2d_out = scenario.distance_2d_out

        # Verify total 3D distances
        ut_loc_expanded = ut_loc.unsqueeze(1)
        bs_loc_expanded = bs_loc.unsqueeze(2)
        d_3d_ref = torch.sqrt(((ut_loc_expanded - bs_loc_expanded) ** 2).sum(dim=3))
        max_err = torch.max(torch.abs(d_3d - d_3d_ref) / d_3d_ref)
        assert max_err <= MAX_ERR

        # Verify 3D indoor + outdoor = total
        max_err = torch.max(torch.abs(d_3d - d_3d_in - d_3d_out) / d_3d)
        assert max_err <= MAX_ERR

        # Verify total 2D distances
        d_2d_ref = torch.sqrt(
            ((ut_loc_expanded[:, :, :, :2] - bs_loc_expanded[:, :, :, :2]) ** 2).sum(dim=3)
        )
        max_err = torch.max(torch.abs(d_2d - d_2d_ref) / d_2d_ref)
        assert max_err <= MAX_ERR

        # Verify 2D indoor + outdoor = total
        max_err = torch.max(torch.abs(d_2d - d_2d_in - d_2d_out) / d_2d)
        assert max_err <= MAX_ERR

    def test_get_param(self, device, precision):
        """Test the get_param() function retrieves correct values"""
        ut_array, bs_array = create_arrays(CARRIER_FREQUENCY, device, precision)
        scenario = RMaScenario(
            carrier_frequency=CARRIER_FREQUENCY,
            ut_array=ut_array,
            bs_array=bs_array,
            direction="uplink",
            precision=precision,
            device=device,
        )

        dtype = torch.float32 if precision == "single" else torch.float64

        ut_loc = generate_random_loc(
            BATCH_SIZE, NB_UT, (100, 2000), (100, 2000), (H_UT, H_UT),
            dtype=dtype, device=device
        )
        bs_loc = generate_random_loc(
            BATCH_SIZE, NB_BS, (0, 100), (0, 100), (H_BS, H_BS),
            dtype=dtype, device=device
        )
        ut_orientations = torch.zeros(BATCH_SIZE, NB_UT, 3, dtype=dtype, device=device)
        bs_orientations = torch.zeros(BATCH_SIZE, NB_BS, 3, dtype=dtype, device=device)
        ut_velocities = torch.zeros(BATCH_SIZE, NB_UT, 3, dtype=dtype, device=device)
        in_state = generate_random_bool(BATCH_SIZE, NB_UT, 0.5, device=device)

        scenario.set_topology(
            ut_loc, bs_loc, ut_orientations, bs_orientations, ut_velocities, in_state
        )

        # Test that muDSc is correctly extracted (RMa-specific values)
        param_tensor = scenario.get_param("muDSc")

        # Build reference tensor
        los = scenario.los.cpu().numpy()
        indoor = scenario.indoor.cpu().numpy()[:, None, :]

        param_tensor_ref = np.zeros([BATCH_SIZE, NB_BS, NB_UT])
        param_tensor_ref[np.where(los)] = -7.49  # LoS value
        param_tensor_ref[np.where(~los)] = -7.43  # NLoS value
        param_tensor_ref[np.where(np.broadcast_to(indoor, los.shape))] = -7.47

        max_err = np.max(np.abs(param_tensor.cpu().numpy() - param_tensor_ref))
        assert max_err <= 1e-5

    def test_los_probability_bounds(self, device, precision):
        """Test that LoS probability is between 0 and 1"""
        ut_array, bs_array = create_arrays(CARRIER_FREQUENCY, device, precision)
        scenario = RMaScenario(
            carrier_frequency=CARRIER_FREQUENCY,
            ut_array=ut_array,
            bs_array=bs_array,
            direction="uplink",
            precision=precision,
            device=device,
        )

        dtype = torch.float32 if precision == "single" else torch.float64

        ut_loc = generate_random_loc(
            BATCH_SIZE, NB_UT, (100, 2000), (100, 2000), (H_UT, H_UT),
            dtype=dtype, device=device
        )
        bs_loc = generate_random_loc(
            BATCH_SIZE, NB_BS, (0, 100), (0, 100), (H_BS, H_BS),
            dtype=dtype, device=device
        )
        ut_orientations = torch.zeros(BATCH_SIZE, NB_UT, 3, dtype=dtype, device=device)
        bs_orientations = torch.zeros(BATCH_SIZE, NB_BS, 3, dtype=dtype, device=device)
        ut_velocities = torch.zeros(BATCH_SIZE, NB_UT, 3, dtype=dtype, device=device)
        in_state = generate_random_bool(BATCH_SIZE, NB_UT, 0.5, device=device)

        scenario.set_topology(
            ut_loc, bs_loc, ut_orientations, bs_orientations, ut_velocities, in_state
        )

        los_prob = scenario.los_probability
        assert torch.all(los_prob >= 0.0)
        assert torch.all(los_prob <= 1.0)


class TestUMaScenario:
    """Tests for UMaScenario"""

    def test_instantiation(self, device, precision):
        """Test UMaScenario can be instantiated"""
        ut_array, bs_array = create_arrays(CARRIER_FREQUENCY, device, precision)
        scenario = UMaScenario(
            carrier_frequency=CARRIER_FREQUENCY,
            o2i_model="low",
            ut_array=ut_array,
            bs_array=bs_array,
            direction="uplink",
            precision=precision,
            device=device,
        )
        assert scenario is not None
        assert scenario.carrier_frequency.item() == pytest.approx(CARRIER_FREQUENCY)

    def test_o2i_model_validation(self, device, precision):
        """Test that o2i_model must be 'low' or 'high'"""
        ut_array, bs_array = create_arrays(CARRIER_FREQUENCY, device, precision)

        with pytest.raises(ValueError):
            UMaScenario(
                carrier_frequency=CARRIER_FREQUENCY,
                o2i_model="invalid",
                ut_array=ut_array,
                bs_array=bs_array,
                direction="uplink",
                precision=precision,
                device=device,
            )

    def test_environment_height_uses_discrete_uniform_support(
        self, device, precision, monkeypatch
    ):
        """Test UMa hE sampling uses 1 m or the specified 3 m grid."""

        import sionna.phy.channel.tr38901.uma_scenario as uma_module

        dtype = torch.float32 if precision == "single" else torch.float64
        ut_array, bs_array = create_arrays(CARRIER_FREQUENCY, device, precision)
        scenario = UMaScenario(
            carrier_frequency=CARRIER_FREQUENCY,
            o2i_model="low",
            ut_array=ut_array,
            bs_array=bs_array,
            direction="downlink",
            precision=precision,
            device=device,
        )
        scenario.set_topology(
            ut_loc=torch.tensor(
                [[
                    [100.0, 0.0, 22.5],
                    [200.0, 0.0, 22.5],
                    [300.0, 0.0, 22.5],
                    [400.0, 0.0, 22.5],
                ]],
                dtype=dtype,
                device=device,
            ),
            bs_loc=torch.tensor(
                [[[0.0, 0.0, 25.0], [0.0, 100.0, 25.0]]],
                dtype=dtype,
                device=device,
            ),
            ut_orientations=torch.zeros(1, 4, 3, dtype=dtype, device=device),
            bs_orientations=torch.zeros(1, 2, 3, dtype=dtype, device=device),
            ut_velocities=torch.zeros(1, 4, 3, dtype=dtype, device=device),
            in_state=torch.zeros(1, 4, dtype=torch.bool, device=device),
            los=True,
        )

        branch_draw = torch.tensor(
            [[[0.0, 1.0, 1.0, 1.0], [1.0, 1.0, 0.0, 1.0]]],
            dtype=dtype,
            device=device,
        )
        discrete_draw = torch.tensor(
            [[[0.0, 0.24, 0.26, 0.99], [0.74, 0.5, 0.1, 0.9]]],
            dtype=dtype,
            device=device,
        )
        draws = iter((branch_draw, discrete_draw))

        def fake_rand(shape, dtype, device, generator=None):
            del generator
            value = next(draws)
            assert tuple(value.shape) == tuple(shape)
            return value.to(dtype=dtype, device=device)

        monkeypatch.setattr(uma_module, "rand", fake_rand)
        h_e = scenario._sample_environment_height(
            scenario.distance_2d,
            scenario.h_ut.unsqueeze(1),
        )
        expected = torch.tensor(
            [[[1.0, 12.0, 15.0, 21.0], [18.0, 18.0, 1.0, 21.0]]],
            dtype=dtype,
            device=device,
        )

        torch.testing.assert_close(h_e, expected, rtol=0.0, atol=0.0)
        with pytest.raises(StopIteration):
            next(draws)

    def test_environment_height_is_shared_by_site(self, device, precision):
        """Check the UMa effective environment belongs to a BS site."""
        dtype = torch.float32 if precision == "single" else torch.float64
        ut_array, bs_array = create_arrays(CARRIER_FREQUENCY, device, precision)
        scenario = UMaScenario(
            CARRIER_FREQUENCY,
            "low",
            ut_array,
            bs_array,
            "downlink",
            precision=precision,
            device=device,
        )
        scenario.set_topology(
            ut_loc=torch.tensor(
                [[[100.0, 0.0, 22.5], [300.0, 0.0, 22.5]]],
                dtype=dtype,
                device=device,
            ),
            bs_loc=torch.tensor(
                [[[0.0, 0.0, 25.0], [0.0, 0.0, 25.0]]],
                dtype=dtype,
                device=device,
            ),
            ut_orientations=torch.zeros(1, 2, 3, dtype=dtype, device=device),
            bs_orientations=torch.zeros(1, 2, 3, dtype=dtype, device=device),
            ut_velocities=torch.zeros(1, 2, 3, dtype=dtype, device=device),
            in_state=torch.zeros(1, 2, dtype=torch.bool, device=device),
            los=True,
        )

        h_e = scenario._sample_environment_height(
            scenario.distance_2d, scenario.h_ut.unsqueeze(1)
        )
        torch.testing.assert_close(h_e[:, 0], h_e[:, 1])

    def test_indoor_floors_infer_distinct_spatial_regions(
        self, device, precision
    ):
        """Check indoor UT heights infer the TR 38.901 floor grid."""
        dtype = torch.float32 if precision == "single" else torch.float64
        ut_array, bs_array = create_arrays(CARRIER_FREQUENCY, device, precision)
        scenario = UMaScenario(
            CARRIER_FREQUENCY,
            "low",
            ut_array,
            bs_array,
            "downlink",
            precision=precision,
            device=device,
        )
        scenario.set_topology(
            ut_loc=torch.tensor(
                [[[100.0, 0.0, 1.5], [100.0, 0.0, 4.5]]],
                dtype=dtype,
                device=device,
            ),
            bs_loc=torch.tensor(
                [[[0.0, 0.0, 25.0]]], dtype=dtype, device=device
            ),
            ut_orientations=torch.zeros(1, 2, 3, dtype=dtype, device=device),
            bs_orientations=torch.zeros(1, 1, 3, dtype=dtype, device=device),
            ut_velocities=torch.zeros(1, 2, 3, dtype=dtype, device=device),
            in_state=torch.ones(1, 2, dtype=torch.bool, device=device),
            los=False,
        )
        expected = torch.tensor([[0, 1]], dtype=torch.int64, device=device)
        assert torch.equal(scenario.ut_spatial_region_ids, expected)

    def test_los_probability_bounds(self, device, precision):
        """Test that LoS probability is between 0 and 1"""
        ut_array, bs_array = create_arrays(CARRIER_FREQUENCY, device, precision)
        scenario = UMaScenario(
            carrier_frequency=CARRIER_FREQUENCY,
            o2i_model="low",
            ut_array=ut_array,
            bs_array=bs_array,
            direction="uplink",
            precision=precision,
            device=device,
        )

        dtype = torch.float32 if precision == "single" else torch.float64

        ut_loc = generate_random_loc(
            BATCH_SIZE, NB_UT, (100, 2000), (100, 2000), (H_UT, H_UT),
            dtype=dtype, device=device
        )
        bs_loc = generate_random_loc(
            BATCH_SIZE, NB_BS, (0, 100), (0, 100), (H_BS, H_BS),
            dtype=dtype, device=device
        )
        ut_orientations = torch.zeros(BATCH_SIZE, NB_UT, 3, dtype=dtype, device=device)
        bs_orientations = torch.zeros(BATCH_SIZE, NB_BS, 3, dtype=dtype, device=device)
        ut_velocities = torch.zeros(BATCH_SIZE, NB_UT, 3, dtype=dtype, device=device)
        in_state = generate_random_bool(BATCH_SIZE, NB_UT, 0.5, device=device)

        scenario.set_topology(
            ut_loc, bs_loc, ut_orientations, bs_orientations, ut_velocities, in_state
        )

        los_prob = scenario.los_probability
        assert torch.all(los_prob >= 0.0)
        assert torch.all(los_prob <= 1.0)

    def test_co_sited_bs_detected_from_equal_positions(self, device, precision):
        """Test that BSs with identical positions share a site representative."""
        ut_array, bs_array = create_arrays(CARRIER_FREQUENCY, device, precision)
        scenario = UMaScenario(
            carrier_frequency=CARRIER_FREQUENCY,
            o2i_model="low",
            ut_array=ut_array,
            bs_array=bs_array,
            direction="downlink",
            precision=precision,
            device=device,
        )

        dtype = torch.float32 if precision == "single" else torch.float64
        zeros_ut = torch.zeros(1, 1, 3, dtype=dtype, device=device)
        zeros_bs = torch.zeros(1, 3, 3, dtype=dtype, device=device)
        scenario.set_topology(
            ut_loc=torch.tensor([[[100.0, 0.0, 1.5]]], dtype=dtype, device=device),
            bs_loc=torch.tensor(
                [[[0.0, 0.0, 25.0], [0.0, 0.0, 25.0], [50.0, 0.0, 25.0]]],
                dtype=dtype,
                device=device,
            ),
            ut_orientations=zeros_ut,
            bs_orientations=zeros_bs,
            ut_velocities=zeros_ut,
            in_state=torch.tensor([[False]], device=device),
            los=True,
        )

        expected = torch.tensor([[0, 0, 2]], dtype=torch.int64, device=device)
        assert torch.equal(scenario.bs_site_representatives, expected)

    def test_near_duplicate_bs_locations_warn_without_site_ids(
        self, device, precision
    ):
        """Test that nearly co-sited BSs require explicit site identifiers."""
        ut_array, bs_array = create_arrays(CARRIER_FREQUENCY, device, precision)
        scenario = UMaScenario(
            carrier_frequency=CARRIER_FREQUENCY,
            o2i_model="low",
            ut_array=ut_array,
            bs_array=bs_array,
            direction="downlink",
            precision=precision,
            device=device,
        )

        dtype = torch.float32 if precision == "single" else torch.float64
        eps = torch.finfo(dtype).eps
        zeros_ut = torch.zeros(1, 1, 3, dtype=dtype, device=device)
        zeros_bs = torch.zeros(1, 2, 3, dtype=dtype, device=device)
        with pytest.warns(UserWarning, match="bs_site_ids"):
            scenario.set_topology(
                ut_loc=torch.tensor(
                    [[[100.0, 0.0, 1.5]]], dtype=dtype, device=device
                ),
                bs_loc=torch.tensor(
                    [[[0.0, 0.0, 25.0], [4.0 * eps, 0.0, 25.0]]],
                    dtype=dtype,
                    device=device,
                ),
                ut_orientations=zeros_ut,
                bs_orientations=zeros_bs,
                ut_velocities=zeros_ut,
                in_state=torch.tensor([[False]], device=device),
                los=True,
            )

        expected = torch.tensor([[0, 1]], dtype=torch.int64, device=device)
        assert torch.equal(scenario.bs_site_representatives, expected)

    def test_legacy_indoor_distance_is_site_link_specific(
        self, device, precision
    ):
        """Check Table 7.4.3-3 distance ownership below 6 GHz."""
        ut_array, bs_array = create_arrays(CARRIER_FREQUENCY, device, precision)
        scenario = UMaScenario(
            carrier_frequency=CARRIER_FREQUENCY,
            o2i_model="low",
            ut_array=ut_array,
            bs_array=bs_array,
            direction="downlink",
            precision=precision,
            device=device,
        )

        dtype = torch.float32 if precision == "single" else torch.float64
        zeros_ut = torch.zeros(1, 2, 3, dtype=dtype, device=device)
        zeros_bs = torch.zeros(1, 3, 3, dtype=dtype, device=device)
        scenario.set_topology(
            ut_loc=torch.tensor(
                [[[100.0, 0.0, 1.5], [120.0, 0.0, 1.5]]],
                dtype=dtype,
                device=device,
            ),
            bs_loc=torch.tensor(
                [[[0.0, 0.0, 25.0], [0.0, 0.0, 25.0], [50.0, 0.0, 25.0]]],
                dtype=dtype,
                device=device,
            ),
            ut_orientations=zeros_ut,
            bs_orientations=zeros_bs,
            ut_velocities=zeros_ut,
            in_state=torch.tensor([[True, True]], device=device),
            los=False,
            distance_2d_in=torch.tensor(
                [[[5.0, 12.0], [20.0, 20.0], [8.0, 3.0]]],
                dtype=dtype,
                device=device,
            ),
        )

        torch.testing.assert_close(
            scenario.distance_2d_in[:, 0, :], scenario.distance_2d_in[:, 1, :]
        )
        torch.testing.assert_close(
            scenario.distance_2d_in[:, 0, :],
            torch.tensor([[5.0, 12.0]], dtype=dtype, device=device),
        )
        torch.testing.assert_close(
            scenario.distance_2d_in[:, 2, :],
            torch.tensor([[8.0, 3.0]], dtype=dtype, device=device),
        )

    def test_modern_indoor_distance_is_ut_specific(self, device, precision):
        """Check Table 7.4.3-2 distance ownership from 6 GHz onward."""
        fc = 6e9
        ut_array, bs_array = create_arrays(fc, device, precision)
        scenario = UMaScenario(
            fc,
            "low",
            ut_array,
            bs_array,
            "downlink",
            precision=precision,
            device=device,
        )
        dtype = torch.float32 if precision == "single" else torch.float64
        scenario.set_topology(
            ut_loc=torch.tensor(
                [[[100.0, 0.0, 1.5], [120.0, 0.0, 1.5]]],
                dtype=dtype,
                device=device,
            ),
            bs_loc=torch.tensor(
                [[[0.0, 0.0, 25.0], [50.0, 0.0, 25.0]]],
                dtype=dtype,
                device=device,
            ),
            ut_orientations=torch.zeros(1, 2, 3, dtype=dtype, device=device),
            bs_orientations=torch.zeros(1, 2, 3, dtype=dtype, device=device),
            ut_velocities=torch.zeros(1, 2, 3, dtype=dtype, device=device),
            in_state=torch.ones(1, 2, dtype=torch.bool, device=device),
            los=False,
            distance_2d_in=torch.tensor(
                [[5.0, 12.0]], dtype=dtype, device=device
            ),
        )
        torch.testing.assert_close(
            scenario.distance_2d_in[:, 0], scenario.distance_2d_in[:, 1]
        )
        original = scenario.distance_2d_in.clone()
        scenario.set_topology(los=True)
        torch.testing.assert_close(scenario.distance_2d_in, original)

    def test_in_state_requires_boolean_dtype_in_allocation_modes(
        self, device, precision
    ):
        """Check eager and preallocated topology setup reject integer masks."""
        ut_array, bs_array = create_arrays(CARRIER_FREQUENCY, device, precision)
        dtype = torch.float32 if precision == "single" else torch.float64
        topology = dict(
            ut_loc=torch.tensor(
                [[[100.0, 0.0, 1.5]]], dtype=dtype, device=device
            ),
            bs_loc=torch.tensor(
                [[[0.0, 0.0, 10.0]]], dtype=dtype, device=device
            ),
            ut_orientations=torch.zeros(1, 1, 3, dtype=dtype, device=device),
            bs_orientations=torch.zeros(1, 1, 3, dtype=dtype, device=device),
            ut_velocities=torch.zeros(1, 1, 3, dtype=dtype, device=device),
            in_state=torch.ones(1, 1, dtype=torch.int64, device=device),
            los=True,
        )
        for preallocate in (False, True):
            scenario = UMiScenario(
                CARRIER_FREQUENCY,
                "low",
                ut_array,
                bs_array,
                "downlink",
                precision=precision,
                device=device,
            )
            if preallocate:
                scenario.allocate_topology_tensors(1, 1, 1)
            with pytest.raises(TypeError, match="in_state"):
                scenario.set_topology(**topology)

    @pytest.mark.parametrize("scenario_name", ["inh", "inf"])
    def test_native_indoor_distance_contract(
        self, device, precision, scenario_name
    ):
        """Check native indoor links use total distance without NaNs."""
        ut_array, bs_array = create_arrays(CARRIER_FREQUENCY, device, precision)
        if scenario_name == "inh":
            scenario = InHScenario(
                CARRIER_FREQUENCY,
                "open",
                ut_array,
                bs_array,
                "downlink",
                precision=precision,
                device=device,
            )
            h_bs, h_ut = 3.0, 1.0
        else:
            scenario = InFScenario(
                CARRIER_FREQUENCY,
                "SH",
                ut_array,
                bs_array,
                "downlink",
                precision=precision,
                device=device,
            )
            h_bs, h_ut = 8.0, 1.5
        dtype = torch.float32 if precision == "single" else torch.float64
        common = dict(
            ut_loc=torch.tensor(
                [[[0.0, 0.0, h_ut]]], dtype=dtype, device=device
            ),
            bs_loc=torch.tensor(
                [[[0.0, 0.0, h_bs]]], dtype=dtype, device=device
            ),
            ut_orientations=torch.zeros(1, 1, 3, dtype=dtype, device=device),
            bs_orientations=torch.zeros(1, 1, 3, dtype=dtype, device=device),
            ut_velocities=torch.zeros(1, 1, 3, dtype=dtype, device=device),
            in_state=torch.ones(1, 1, dtype=torch.bool, device=device),
            los=True,
        )
        scenario.set_topology(**common)
        torch.testing.assert_close(scenario.distance_2d_in, scenario.distance_2d)
        torch.testing.assert_close(scenario.distance_3d_in, scenario.distance_3d)
        torch.testing.assert_close(
            scenario.distance_2d_out, torch.zeros_like(scenario.distance_2d_out)
        )
        torch.testing.assert_close(
            scenario.distance_3d_out, torch.zeros_like(scenario.distance_3d_out)
        )
        assert torch.all(torch.isfinite(scenario.distance_3d_in))

        invalid = dict(common)
        invalid["in_state"] = torch.zeros(
            1, 1, dtype=torch.bool, device=device
        )
        scenario.reset_topology()
        with pytest.raises(ValueError, match="require every UT to be indoor"):
            scenario.set_topology(**invalid)

        scenario.reset_topology()
        with pytest.raises(ValueError, match="not applicable"):
            scenario.set_topology(
                **common,
                distance_2d_in=torch.ones(1, 1, dtype=dtype, device=device),
            )


class TestUMiScenario:
    """Tests for UMiScenario"""

    def test_instantiation(self, device, precision):
        """Test UMiScenario can be instantiated"""
        ut_array, bs_array = create_arrays(CARRIER_FREQUENCY, device, precision)
        scenario = UMiScenario(
            carrier_frequency=CARRIER_FREQUENCY,
            o2i_model="low",
            ut_array=ut_array,
            bs_array=bs_array,
            direction="uplink",
            precision=precision,
            device=device,
        )
        assert scenario is not None
        assert scenario.carrier_frequency.item() == pytest.approx(CARRIER_FREQUENCY)

    def test_los_probability_bounds(self, device, precision):
        """Test that LoS probability is between 0 and 1"""
        ut_array, bs_array = create_arrays(CARRIER_FREQUENCY, device, precision)
        scenario = UMiScenario(
            carrier_frequency=CARRIER_FREQUENCY,
            o2i_model="high",
            ut_array=ut_array,
            bs_array=bs_array,
            direction="uplink",
            precision=precision,
            device=device,
        )

        dtype = torch.float32 if precision == "single" else torch.float64

        ut_loc = generate_random_loc(
            BATCH_SIZE, NB_UT, (100, 2000), (100, 2000), (H_UT, H_UT),
            dtype=dtype, device=device
        )
        bs_loc = generate_random_loc(
            BATCH_SIZE, NB_BS, (0, 100), (0, 100), (H_BS, H_BS),
            dtype=dtype, device=device
        )
        ut_orientations = torch.zeros(BATCH_SIZE, NB_UT, 3, dtype=dtype, device=device)
        bs_orientations = torch.zeros(BATCH_SIZE, NB_BS, 3, dtype=dtype, device=device)
        ut_velocities = torch.zeros(BATCH_SIZE, NB_UT, 3, dtype=dtype, device=device)
        in_state = generate_random_bool(BATCH_SIZE, NB_UT, 0.5, device=device)

        scenario.set_topology(
            ut_loc, bs_loc, ut_orientations, bs_orientations, ut_velocities, in_state
        )

        los_prob = scenario.los_probability
        assert torch.all(los_prob >= 0.0)
        assert torch.all(los_prob <= 1.0)

    def test_los_zoa_points_from_ut_to_bs(self, device, precision):
        """Test LoS ZOA is folded to the UT-to-BS zenith direction."""
        dtype = torch.float32 if precision == "single" else torch.float64
        ut_array, bs_array = create_arrays(CARRIER_FREQUENCY, device, precision)
        scenario = UMiScenario(
            carrier_frequency=CARRIER_FREQUENCY,
            o2i_model="low",
            ut_array=ut_array,
            bs_array=bs_array,
            direction="downlink",
            precision=precision,
            device=device,
        )

        bs_loc = torch.tensor([[[0.0, 0.0, 10.0]]], dtype=dtype, device=device)
        ut_loc = torch.tensor([[[100.0, 0.0, 1.5]]], dtype=dtype, device=device)
        zeros_bs = torch.zeros(1, 1, 3, dtype=dtype, device=device)
        zeros_ut = torch.zeros(1, 1, 3, dtype=dtype, device=device)

        scenario.set_topology(
            ut_loc,
            bs_loc,
            zeros_ut,
            zeros_bs,
            zeros_ut,
            torch.zeros(1, 1, dtype=torch.bool, device=device),
            los=True,
        )

        delta = ut_loc[:, None, :, :] - bs_loc[:, :, None, :]
        distance_2d = torch.linalg.norm(delta[..., :2], dim=-1)
        expected_zod = torch.rad2deg(torch.atan2(distance_2d, delta[..., 2]))
        expected_zoa = torch.rad2deg(torch.atan2(distance_2d, -delta[..., 2]))

        torch.testing.assert_close(scenario.los_zod, expected_zod)
        torch.testing.assert_close(scenario.los_zoa, expected_zoa)
        assert torch.all(scenario.los_zoa >= 0.0)
        assert torch.all(scenario.los_zoa <= 180.0)


class TestRMaChannel:
    """Tests for RMa channel model"""

    @staticmethod
    def _scenario(device, precision, **kwargs):
        ut_array, bs_array = create_arrays(
            CARRIER_FREQUENCY, device, precision
        )
        return RMaScenario(
            carrier_frequency=CARRIER_FREQUENCY,
            ut_array=ut_array,
            bs_array=bs_array,
            direction="downlink",
            precision=precision,
            device=device,
            **kwargs,
        )

    @staticmethod
    def _topology(device, precision, in_state):
        dtype = torch.float32 if precision == "single" else torch.float64
        batch_size, num_ut = in_state.shape
        ut_x = torch.arange(
            100.0,
            100.0 + 20.0 * num_ut,
            20.0,
            dtype=dtype,
            device=device,
        )
        ut_loc = torch.stack(
            [
                ut_x,
                torch.zeros_like(ut_x),
                torch.full_like(ut_x, 1.5),
            ],
            dim=-1,
        ).unsqueeze(0).expand(batch_size, -1, -1).clone()
        return dict(
            ut_loc=ut_loc,
            bs_loc=torch.tensor(
                [[[0.0, 0.0, 35.0]]], dtype=dtype, device=device
            ).expand(batch_size, -1, -1).clone(),
            ut_orientations=torch.zeros(
                batch_size, num_ut, 3, dtype=dtype, device=device
            ),
            bs_orientations=torch.zeros(
                batch_size, 1, 3, dtype=dtype, device=device
            ),
            ut_velocities=torch.zeros(
                batch_size, num_ut, 3, dtype=dtype, device=device
            ),
            in_state=in_state,
            los=True,
        )

    def test_instantiation(self, device, precision):
        """Test RMa channel can be instantiated"""
        ut_array, bs_array = create_arrays(CARRIER_FREQUENCY, device, precision)
        channel = RMa(
            carrier_frequency=CARRIER_FREQUENCY,
            ut_array=ut_array,
            bs_array=bs_array,
            direction="uplink",
            precision=precision,
            device=device,
        )
        assert channel is not None

    @pytest.mark.parametrize("preallocate", [False, True])
    def test_default_population_maps_non_indoor_uts_to_cars(
        self, device, precision, preallocate
    ):
        """Check the Table 7.2-3 default indoor/in-car population."""
        scenario = self._scenario(device, precision)
        in_state = torch.tensor(
            [[True, False, False]], dtype=torch.bool, device=device
        )
        if preallocate:
            scenario.allocate_topology_tensors(1, 1, 3)
        scenario.set_topology(**self._topology(device, precision, in_state))

        torch.testing.assert_close(scenario.in_car, ~in_state)
        assert not torch.any(scenario.in_car & scenario.indoor)
        assert not scenario.los[0, 0, 0]
        assert torch.all(scenario.los[0, 0, 1:])
        torch.testing.assert_close(
            scenario.distance_2d_in[0, 0, 1:],
            torch.zeros_like(scenario.distance_2d_in[0, 0, 1:]),
        )

        original = scenario.in_car.clone()
        scenario.set_topology(los=False)
        torch.testing.assert_close(scenario.in_car, original)

        updated_in_state = torch.tensor(
            [[False, True, False]], dtype=torch.bool, device=device
        )
        scenario.set_topology(in_state=updated_in_state)
        torch.testing.assert_close(scenario.in_car, ~updated_in_state)

    def test_explicit_in_car_mask_and_validation(self, device, precision):
        """Check pedestrian outdoor selection and invalid car masks."""
        in_state = torch.tensor(
            [[False, False]], dtype=torch.bool, device=device
        )
        topology = self._topology(device, precision, in_state)
        scenario = self._scenario(device, precision)
        updated = scenario.set_topology(
            **topology,
            in_car=torch.tensor([[False, True]], device=device),
        )
        assert updated
        torch.testing.assert_close(
            scenario.in_car,
            torch.tensor([[False, True]], device=device),
        )

        with pytest.raises(TypeError, match="in_car"):
            self._scenario(device, precision).set_topology(
                **topology,
                in_car=torch.zeros(1, 2, dtype=torch.int64, device=device),
            )
        with pytest.raises(ValueError, match="shape"):
            self._scenario(device, precision).set_topology(
                **topology,
                in_car=torch.zeros(2, dtype=torch.bool, device=device),
            )

        overlapping = self._topology(
            device,
            precision,
            torch.tensor([[True, False]], device=device),
        )
        with pytest.raises(ValueError, match="both indoor and in-car"):
            self._scenario(device, precision).set_topology(
                **overlapping,
                in_car=torch.tensor([[True, False]], device=device),
            )

        original_indoor = scenario.indoor.clone()
        original_in_car = scenario.in_car.clone()
        with pytest.raises(ValueError, match="both indoor and in-car"):
            scenario.set_topology(
                in_state=torch.tensor([[False, True]], device=device)
            )
        torch.testing.assert_close(scenario.indoor, original_indoor)
        torch.testing.assert_close(scenario.in_car, original_in_car)

    def test_compiled_inferred_car_mask_follows_indoor_state(self, device):
        """Check repeated compiled topology updates refresh inferred cars."""
        if not hasattr(torch, "compile"):
            pytest.skip("torch.compile is not available")

        scenario = self._scenario(device, "single")
        scenario.allocate_topology_tensors(1, 1, 3)
        first_in_state = torch.tensor(
            [[True, False, False]], dtype=torch.bool, device=device
        )
        topology = self._topology(device, "single", first_in_state)
        del topology["in_state"]

        def update(in_state):
            scenario.set_topology(**topology, in_state=in_state)
            return scenario.in_car.clone()

        compiled_update = torch.compile(update, backend="eager")
        torch.testing.assert_close(
            compiled_update(first_in_state), ~first_in_state
        )

        second_in_state = torch.tensor(
            [[False, True, False]], dtype=torch.bool, device=device
        )
        torch.testing.assert_close(
            compiled_update(second_in_state), ~second_in_state
        )

    def test_in_car_state_resets_across_shape_changes(
        self, device, precision
    ):
        """Check RMa car state participates in reset and preallocation."""
        scenario = self._scenario(device, precision)
        in_state = torch.tensor([[False, False]], device=device)
        scenario.set_topology(
            **self._topology(device, precision, in_state),
            in_car=torch.tensor([[False, True]], device=device),
        )
        assert scenario.in_car.shape == (1, 2)

        scenario.reset_topology()
        assert scenario.in_car is None
        scenario.allocate_topology_tensors(1, 1, 3)
        new_in_state = torch.tensor(
            [[False, True, False]], device=device
        )
        scenario.set_topology(
            **self._topology(device, precision, new_in_state)
        )
        torch.testing.assert_close(scenario.in_car, ~new_in_state)

    @pytest.mark.parametrize("window_type", ["ordinary", "metallized"])
    def test_car_window_type_validation(
        self, device, precision, window_type
    ):
        """Check supported RMa car-window selections and means."""
        scenario = self._scenario(
            device, precision, car_window_type=window_type
        )
        expected = 9.0 if window_type == "ordinary" else 20.0
        assert scenario.car_window_type == window_type
        assert scenario.car_penetration_loss_mean.item() == pytest.approx(
            expected
        )

        with pytest.raises(ValueError, match="car_window_type"):
            self._scenario(device, precision, car_window_type="unsupported")

    def test_channel_generation(self, device, precision):
        """Test channel impulse response generation"""
        dtype = torch.float32 if precision == "single" else torch.float64

        ut_array, bs_array = create_arrays(CARRIER_FREQUENCY, device, precision)
        channel = RMa(
            carrier_frequency=CARRIER_FREQUENCY,
            ut_array=ut_array,
            bs_array=bs_array,
            direction="uplink",
            precision=precision,
            device=device,
        )

        # Set up topology
        ut_loc = generate_random_loc(
            BATCH_SIZE, NB_UT, (100, 2000), (100, 2000), (H_UT, H_UT),
            dtype=dtype, device=device
        )
        bs_loc = generate_random_loc(
            BATCH_SIZE, NB_BS, (0, 100), (0, 100), (H_BS, H_BS),
            dtype=dtype, device=device
        )
        ut_orientations = torch.zeros(BATCH_SIZE, NB_UT, 3, dtype=dtype, device=device)
        bs_orientations = torch.zeros(BATCH_SIZE, NB_BS, 3, dtype=dtype, device=device)
        ut_velocities = torch.zeros(BATCH_SIZE, NB_UT, 3, dtype=dtype, device=device)
        in_state = generate_random_bool(BATCH_SIZE, NB_UT, 0.5, device=device)

        channel.set_topology(
            ut_loc, bs_loc, ut_orientations, bs_orientations, ut_velocities, in_state
        )

        # Generate channel
        num_time_samples = 10
        sampling_frequency = 1e6
        result = channel(num_time_samples, sampling_frequency)
        h, delays = result[0], result[1]

        # Check output shapes
        assert h.dim() == 7  # [batch, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time]
        assert h.shape[0] == BATCH_SIZE
        assert delays.dim() == 4  # [batch, num_rx, num_tx, num_paths]
        assert delays.shape[0] == BATCH_SIZE


class TestUMaChannel:
    """Tests for UMa channel model"""

    def test_instantiation(self, device, precision):
        """Test UMa channel can be instantiated"""
        ut_array, bs_array = create_arrays(CARRIER_FREQUENCY, device, precision)
        channel = UMa(
            carrier_frequency=CARRIER_FREQUENCY,
            o2i_model="low",
            ut_array=ut_array,
            bs_array=bs_array,
            direction="uplink",
            precision=precision,
            device=device,
        )
        assert channel is not None

    def test_channel_generation(self, device, precision):
        """Test channel impulse response generation"""
        dtype = torch.float32 if precision == "single" else torch.float64

        ut_array, bs_array = create_arrays(CARRIER_FREQUENCY, device, precision)
        channel = UMa(
            carrier_frequency=CARRIER_FREQUENCY,
            o2i_model="low",
            ut_array=ut_array,
            bs_array=bs_array,
            direction="downlink",
            precision=precision,
            device=device,
        )

        # Set up topology
        ut_loc = generate_random_loc(
            BATCH_SIZE, NB_UT, (100, 2000), (100, 2000), (H_UT, H_UT),
            dtype=dtype, device=device
        )
        bs_loc = generate_random_loc(
            BATCH_SIZE, NB_BS, (0, 100), (0, 100), (H_BS, H_BS),
            dtype=dtype, device=device
        )
        ut_orientations = torch.zeros(BATCH_SIZE, NB_UT, 3, dtype=dtype, device=device)
        bs_orientations = torch.zeros(BATCH_SIZE, NB_BS, 3, dtype=dtype, device=device)
        ut_velocities = torch.zeros(BATCH_SIZE, NB_UT, 3, dtype=dtype, device=device)
        in_state = generate_random_bool(BATCH_SIZE, NB_UT, 0.5, device=device)

        channel.set_topology(
            ut_loc, bs_loc, ut_orientations, bs_orientations, ut_velocities, in_state
        )

        # Generate channel
        num_time_samples = 10
        sampling_frequency = 1e6
        result = channel(num_time_samples, sampling_frequency)
        h, delays = result[0], result[1]

        # Check output shapes
        assert h.dim() == 7
        assert h.shape[0] == BATCH_SIZE
        assert delays.dim() == 4
        assert delays.shape[0] == BATCH_SIZE


class TestUMiChannel:
    """Tests for UMi channel model"""

    def test_instantiation(self, device, precision):
        """Test UMi channel can be instantiated"""
        ut_array, bs_array = create_arrays(CARRIER_FREQUENCY, device, precision)
        channel = UMi(
            carrier_frequency=CARRIER_FREQUENCY,
            o2i_model="high",
            ut_array=ut_array,
            bs_array=bs_array,
            direction="uplink",
            precision=precision,
            device=device,
        )
        assert channel is not None

    def test_channel_generation(self, device, precision):
        """Test channel impulse response generation"""
        ut_array, bs_array = create_arrays(CARRIER_FREQUENCY, device, precision)
        channel = UMi(
            carrier_frequency=CARRIER_FREQUENCY,
            o2i_model="low",
            ut_array=ut_array,
            bs_array=bs_array,
            direction="uplink",
            precision=precision,
            device=device,
        )

        dtype = torch.float32 if precision == "single" else torch.float64

        # Set up topology
        ut_loc = generate_random_loc(
            BATCH_SIZE, NB_UT, (100, 2000), (100, 2000), (H_UT, H_UT),
            dtype=dtype, device=device
        )
        bs_loc = generate_random_loc(
            BATCH_SIZE, NB_BS, (0, 100), (0, 100), (H_BS, H_BS),
            dtype=dtype, device=device
        )
        ut_orientations = torch.zeros(BATCH_SIZE, NB_UT, 3, dtype=dtype, device=device)
        bs_orientations = torch.zeros(BATCH_SIZE, NB_BS, 3, dtype=dtype, device=device)
        ut_velocities = torch.zeros(BATCH_SIZE, NB_UT, 3, dtype=dtype, device=device)
        in_state = generate_random_bool(BATCH_SIZE, NB_UT, 0.5, device=device)

        channel.set_topology(
            ut_loc, bs_loc, ut_orientations, bs_orientations, ut_velocities, in_state
        )

        # Generate channel
        num_time_samples = 10
        sampling_frequency = 1e6
        result = channel(num_time_samples, sampling_frequency)
        h, delays = result[0], result[1]

        # Check output shapes
        assert h.dim() == 7
        assert h.shape[0] == BATCH_SIZE
        assert delays.dim() == 4
        assert delays.shape[0] == BATCH_SIZE


class TestAllScenarios:
    """Cross-scenario tests"""

    @pytest.mark.parametrize("scenario_class,kwargs", [
        (RMaScenario, {}),
        (UMaScenario, {"o2i_model": "low"}),
        (UMiScenario, {"o2i_model": "low"}),
    ])
    def test_lsp_shapes(self, device, precision, scenario_class, kwargs):
        """Test LSP shapes are correct for all scenarios"""
        ut_array, bs_array = create_arrays(CARRIER_FREQUENCY, device, precision)
        scenario = scenario_class(
            carrier_frequency=CARRIER_FREQUENCY,
            ut_array=ut_array,
            bs_array=bs_array,
            direction="uplink",
            precision=precision,
            device=device,
            **kwargs,
        )

        dtype = torch.float32 if precision == "single" else torch.float64

        ut_loc = generate_random_loc(
            BATCH_SIZE, NB_UT, (100, 2000), (100, 2000), (H_UT, H_UT),
            dtype=dtype, device=device
        )
        bs_loc = generate_random_loc(
            BATCH_SIZE, NB_BS, (0, 100), (0, 100), (H_BS, H_BS),
            dtype=dtype, device=device
        )
        ut_orientations = torch.zeros(BATCH_SIZE, NB_UT, 3, dtype=dtype, device=device)
        bs_orientations = torch.zeros(BATCH_SIZE, NB_BS, 3, dtype=dtype, device=device)
        ut_velocities = torch.zeros(BATCH_SIZE, NB_UT, 3, dtype=dtype, device=device)
        in_state = generate_random_bool(BATCH_SIZE, NB_UT, 0.5, device=device)

        scenario.set_topology(
            ut_loc, bs_loc, ut_orientations, bs_orientations, ut_velocities, in_state
        )

        # Check LSP mean and std shapes
        assert scenario.lsp_log_mean.shape == (BATCH_SIZE, NB_BS, NB_UT, 7)
        assert scenario.lsp_log_std.shape == (BATCH_SIZE, NB_BS, NB_UT, 7)

    @pytest.mark.parametrize("direction", ["uplink", "downlink"])
    def test_direction_validation(self, device, precision, direction):
        """Test that direction parameter works for both values"""
        ut_array, bs_array = create_arrays(CARRIER_FREQUENCY, device, precision)
        scenario = RMaScenario(
            carrier_frequency=CARRIER_FREQUENCY,
            ut_array=ut_array,
            bs_array=bs_array,
            direction=direction,
            precision=precision,
            device=device,
        )
        assert scenario.direction == direction

    def test_invalid_direction(self, device, precision):
        """Test that invalid direction raises error"""
        ut_array, bs_array = create_arrays(CARRIER_FREQUENCY, device, precision)
        with pytest.raises(ValueError):
            RMaScenario(
                carrier_frequency=CARRIER_FREQUENCY,
                ut_array=ut_array,
                bs_array=bs_array,
                direction="invalid",
                precision=precision,
                device=device,
            )

    def test_los_random_resamples_after_forced_state(self, device, precision):
        """Test explicit stochastic LoS state after a forced LoS state."""
        ut_array, bs_array = create_arrays(CARRIER_FREQUENCY, device, precision)
        scenario = UMiScenario(
            carrier_frequency=CARRIER_FREQUENCY,
            o2i_model="low",
            ut_array=ut_array,
            bs_array=bs_array,
            direction="downlink",
            precision=precision,
            device=device,
        )
        dtype = torch.float32 if precision == "single" else torch.float64
        ut_loc = torch.tensor([[[20.0, 0.0, 1.5],
                                [40.0, 0.0, 1.5]]], dtype=dtype, device=device)
        bs_loc = torch.tensor([[[0.0, 0.0, 10.0]]], dtype=dtype, device=device)
        ut_orientations = torch.zeros(1, 2, 3, dtype=dtype, device=device)
        bs_orientations = torch.zeros(1, 1, 3, dtype=dtype, device=device)
        ut_velocities = torch.zeros(1, 2, 3, dtype=dtype, device=device)
        in_state = torch.zeros(1, 2, dtype=torch.bool, device=device)

        scenario.set_topology(
            ut_loc, bs_loc, ut_orientations, bs_orientations, ut_velocities,
            in_state, los=True
        )
        assert scenario._requested_los is True
        assert torch.all(scenario.los)

        scenario.set_topology(los=None)
        assert scenario._requested_los is True
        assert torch.all(scenario.los)

        scenario.set_topology(los="random")
        assert scenario._requested_los is None
        assert scenario.los.shape == (1, 1, 2)

        with pytest.raises(ValueError):
            scenario.set_topology(los="stochastic")
