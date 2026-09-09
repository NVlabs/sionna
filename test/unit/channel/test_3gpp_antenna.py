#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Tests for 3GPP TR 38.901 antenna classes"""

import numpy as np
import matplotlib.pyplot as plt
import pytest
import torch

from sionna.phy import PI, SPEED_OF_LIGHT, dtypes
from sionna.phy.channel.tr38901 import (
    AntennaElement,
    AntennaPanel,
    PanelArray,
    Antenna,
    AntennaArray,
    HandheldUTArray,
    UMi,
    ChannelCoefficientsGenerator,
    Topology,
)


class TestAntennaElement:
    """Tests for the AntennaElement class"""

    def test_omni_pattern_returns_ones(self, device, precision):
        """Test that omnidirectional pattern returns 1.0 for all angles"""
        ant = AntennaElement(pattern="omni", precision=precision, device=device)

        theta = torch.linspace(0.01, PI - 0.01, 10, device=device)
        phi = torch.linspace(-PI + 0.01, PI - 0.01, 10, device=device)

        pattern = ant._radiation_pattern_omni(theta, phi)

        assert torch.allclose(pattern, torch.ones_like(pattern))

    def test_38901_pattern_peak_at_boresight(self, device, precision):
        """Test that 38.901 pattern has maximum at boresight (theta=pi/2, phi=0)"""
        ant = AntennaElement(pattern="38.901", precision=precision, device=device)

        # Boresight direction
        theta_boresight = torch.tensor([PI / 2], device=device)
        phi_boresight = torch.tensor([0.0], device=device)
        pattern_boresight = ant._radiation_pattern_38901(theta_boresight, phi_boresight)

        # Off-boresight direction
        theta_off = torch.tensor([PI / 4], device=device)
        phi_off = torch.tensor([PI / 4], device=device)
        pattern_off = ant._radiation_pattern_38901(theta_off, phi_off)

        assert pattern_boresight > pattern_off

    def test_field_polarization_vertical(self, device, precision):
        """Test that vertical polarization (slant_angle=0) gives f_phi=0"""
        ant = AntennaElement(pattern="omni", slant_angle=0.0, precision=precision, device=device)

        theta = torch.tensor([PI / 2], device=device)
        phi = torch.tensor([0.0], device=device)
        f_theta, f_phi = ant.field(theta, phi)

        # For slant_angle=0, f_phi should be 0
        assert torch.allclose(f_phi, torch.zeros_like(f_phi), atol=1e-6)
        # f_theta should be non-zero (equal to 1 for omni pattern)
        assert torch.allclose(f_theta, torch.ones_like(f_theta), atol=1e-6)

    def test_field_polarization_horizontal(self, device, precision):
        """Test that horizontal polarization (slant_angle=pi/2) gives f_theta=0"""
        dtype = dtypes[precision]["torch"]["dtype"]
        ant = AntennaElement(pattern="omni", slant_angle=PI / 2, precision=precision, device=device)

        theta = torch.tensor([PI / 2], dtype=dtype, device=device)
        phi = torch.tensor([0.0], dtype=dtype, device=device)
        f_theta, f_phi = ant.field(theta, phi)

        # For slant_angle=pi/2, f_theta should be ~0
        assert torch.allclose(f_theta, torch.zeros_like(f_theta), atol=1e-6)
        # f_phi should be non-zero (equal to 1 for omni pattern)
        assert torch.allclose(f_phi, torch.ones_like(f_phi), atol=1e-6)

    def test_output_dtype(self, device, precision):
        """Test that output matches configured precision"""
        dtype = dtypes[precision]["torch"]["dtype"]
        ant = AntennaElement(pattern="omni", precision=precision, device=device)

        theta = torch.tensor([PI / 2], dtype=dtype, device=device)
        phi = torch.tensor([0.0], dtype=dtype, device=device)
        f_theta, f_phi = ant.field(theta, phi)

        assert f_theta.dtype == dtype
        assert f_phi.dtype == dtype
        assert f_theta.device == torch.device(device)

    def test_pattern_property(self, device, precision):
        """Test that pattern property returns the correct value"""
        ant_omni = AntennaElement(pattern="omni", precision=precision, device=device)
        ant_38901 = AntennaElement(pattern="38.901", precision=precision, device=device)
        ant_handheld = AntennaElement(pattern="38.901-handheld", precision=precision, device=device)

        assert ant_omni.pattern == "omni"
        assert ant_38901.pattern == "38.901"
        assert ant_handheld.pattern == "38.901-handheld"

    def test_38901_handheld_pattern_from_table(self, device, precision):
        """Test the TR 38.901 Table 7.3-2 handheld UT pattern values"""
        dtype = dtypes[precision]["torch"]["dtype"]
        ant = AntennaElement(pattern="38.901-handheld",
                             precision=precision,
                             device=device)

        theta_deg = torch.tensor([90.0, 90.0, 27.5, 27.5, 90.0, 90.0],
                                 dtype=dtype,
                                 device=device)
        phi_deg = torch.tensor([0.0, 62.5, 0.0, 62.5, 125.0, 180.0],
                               dtype=dtype,
                               device=device)
        expected_gain_db = torch.tensor([5.3, 2.3, 2.3, -0.7, -6.7, -17.2],
                                        dtype=dtype,
                                        device=device)

        theta = theta_deg * PI / 180
        phi = phi_deg * PI / 180
        f_theta, f_phi = ant.field(theta, phi)
        gain_db = 10 * torch.log10(f_theta**2 + f_phi**2)

        assert torch.allclose(gain_db, expected_gain_db,
                              rtol=0.0, atol=1e-5)


class TestHandheldUTArray:
    """Tests for the TR 38.901 handheld UT antenna array"""

    def test_invalid_polarization_type_preserves_context(self, device):
        """Validation errors identify the selected polarization mode."""
        with pytest.raises(ValueError, match="single polarization"):
            HandheldUTArray(
                carrier_frequency=7e9,
                polarization="single",
                polarization_type="cross",
                device=device,
            )
        with pytest.raises(ValueError, match="dual polarization"):
            HandheldUTArray(
                carrier_frequency=7e9,
                polarization="dual",
                polarization_type="V",
                device=device,
            )

    def test_four_corner_candidate_positions(self, device, precision):
        """Test the Figure 7.3-2 four-corner candidate subset"""
        dtype = dtypes[precision]["torch"]["dtype"]
        array = HandheldUTArray(
            carrier_frequency=7e9,
            polarization="single",
            antenna_locations="tr38901-4",
            precision=precision,
            device=device,
        )

        expected = torch.tensor(
            [
                [-0.075, -0.035, 0.0],
                [-0.075, 0.035, 0.0],
                [0.075, -0.035, 0.0],
                [0.075, 0.035, 0.0],
            ],
            dtype=dtype,
            device=device,
        )
        assert array.antenna_locations == (1, 7, 3, 5)
        assert array.num_ant == 4
        assert torch.allclose(array.ant_pos, expected, atol=1e-7)

    def test_candidate_position_numbering_matches_spec(self, device, precision):
        """Test the Figure 7.3-2 candidate numbering."""
        dtype = dtypes[precision]["torch"]["dtype"]
        array = HandheldUTArray(
            carrier_frequency=7e9,
            polarization="single",
            antenna_locations="tr38901",
            precision=precision,
            device=device,
        )

        expected = torch.tensor(
            [
                [-0.075, -0.035, 0.0],
                [0.0, -0.035, 0.0],
                [0.075, -0.035, 0.0],
                [0.075, 0.0, 0.0],
                [0.075, 0.035, 0.0],
                [0.0, 0.035, 0.0],
                [-0.075, 0.035, 0.0],
                [-0.075, 0.0, 0.0],
            ],
            dtype=dtype,
            device=device,
        )

        assert array.antenna_locations == tuple(range(1, 9))
        assert torch.allclose(array.ant_pos, expected, atol=1e-7)

    def test_dual_polarization_indices_and_positions(self, device, precision):
        """Test dual-polarized handheld arrays duplicate candidate positions"""
        array = HandheldUTArray(
            carrier_frequency=15e9,
            polarization="dual",
            antenna_locations=(1, 2, 3),
            precision=precision,
            device=device,
        )

        assert array.num_ant == 6
        assert torch.equal(array.ant_ind_pol1,
                           torch.tensor([0, 1, 2], device=device))
        assert torch.equal(array.ant_ind_pol2,
                           torch.tensor([3, 4, 5], device=device))
        assert torch.allclose(array.ant_pos[:3], array.ant_pos[3:])
        assert torch.equal(
            array.port_field_component,
            torch.tensor([0, 0, 0, 1, 1, 1], device=device),
        )

    def test_candidate_boresight_and_polarization_axes(self, device, precision):
        """Test Clause 7.3 center-to-candidate and tangential directions."""
        dtype = dtypes[precision]["torch"]["dtype"]
        array = HandheldUTArray(
            carrier_frequency=7e9,
            polarization="single",
            antenna_locations="tr38901",
            precision=precision,
            device=device,
        )

        boresight = array.port_basis[:, :, 0]
        polarization_axis = -array.port_basis[:, :, 2]
        candidate_direction = array.ant_pos / torch.linalg.norm(
            array.ant_pos, dim=-1, keepdim=True
        )

        assert torch.allclose(boresight, candidate_direction, atol=1e-6)
        assert torch.allclose(boresight[:, 2],
                              torch.zeros(8, dtype=dtype, device=device),
                              atol=1e-6)
        assert torch.allclose(polarization_axis[:, 2],
                              torch.zeros(8, dtype=dtype, device=device),
                              atol=1e-6)
        assert torch.allclose(torch.sum(boresight*polarization_axis, dim=-1),
                              torch.zeros(8, dtype=dtype, device=device),
                              atol=1e-6)

        device_normal = torch.tensor([0.0, 0.0, 1.0],
                                     dtype=dtype,
                                     device=device)
        expected_polarization_axis = torch.linalg.cross(
            boresight, device_normal.expand_as(boresight), dim=-1
        )
        expected_polarization_axis = expected_polarization_axis \
            / torch.linalg.norm(expected_polarization_axis,
                                dim=-1,
                                keepdim=True)
        assert torch.allclose(polarization_axis,
                              expected_polarization_axis,
                              atol=1e-6)

    def test_dual_cross_polarization_axes_match_spec(self, device, precision):
        """Test Clause 7.3 dual-field 45 degree polarization rotation."""
        dtype = dtypes[precision]["torch"]["dtype"]
        array = HandheldUTArray(
            carrier_frequency=15e9,
            polarization="dual",
            polarization_type="cross",
            antenna_locations="tr38901",
            precision=precision,
            device=device,
        )

        boresight = array.port_basis[:8, :, 0]
        pol1 = -array.port_basis[:8, :, 2]
        pol2 = array.port_basis[8:, :, 1]
        single_pol = torch.linalg.cross(
            boresight,
            torch.tensor([0.0, 0.0, 1.0],
                         dtype=dtype,
                         device=device).expand_as(boresight),
            dim=-1,
        )
        single_pol = single_pol / torch.linalg.norm(
            single_pol, dim=-1, keepdim=True
        )
        orthogonal_pol = torch.linalg.cross(boresight, single_pol, dim=-1)

        expected_projection = torch.full([8],
                                         1/np.sqrt(2),
                                         dtype=dtype,
                                         device=device)
        assert torch.allclose(torch.sum(pol1*single_pol, dim=-1),
                              expected_projection,
                              atol=1e-6)
        assert torch.allclose(torch.sum(pol1*orthogonal_pol, dim=-1),
                              expected_projection,
                              atol=1e-6)
        assert torch.allclose(torch.sum(pol1*pol2, dim=-1),
                              torch.zeros(8, dtype=dtype, device=device),
                              atol=1e-6)
        assert torch.allclose(torch.sum(pol2*boresight, dim=-1),
                              torch.zeros(8, dtype=dtype, device=device),
                              atol=1e-6)

    def test_dual_vh_polarization_axes_are_orthogonal(self, device, precision):
        """Test VH ports use orthogonal local field components."""
        dtype = dtypes[precision]["torch"]["dtype"]
        array = HandheldUTArray(
            carrier_frequency=15e9,
            polarization="dual",
            polarization_type="VH",
            antenna_locations="tr38901",
            precision=precision,
            device=device,
        )

        boresight = array.port_basis[:8, :, 0]
        pol1 = -array.port_basis[:8, :, 2]
        pol2 = array.port_basis[8:, :, 1]
        assert torch.allclose(torch.sum(pol1*pol2, dim=-1),
                              torch.zeros(8, dtype=dtype, device=device),
                              atol=1e-6)
        assert torch.allclose(torch.sum(pol2*boresight, dim=-1),
                              torch.zeros(8, dtype=dtype, device=device),
                              atol=1e-6)

    def test_device_size_uses_depth_and_width(self, device, precision):
        """Test the handheld device dimensions follow TR 38.901 axes."""
        dtype = dtypes[precision]["torch"]["dtype"]
        array = HandheldUTArray(
            carrier_frequency=7e9,
            polarization="single",
            antenna_locations=(2,),
            device_depth=0.2,
            device_width=0.1,
            precision=precision,
            device=device,
        )

        expected = torch.tensor([[0.0, -0.05, 0.0]],
                                dtype=dtype,
                                device=device)
        assert torch.allclose(array.device_size,
                              torch.tensor([0.2, 0.1],
                                           dtype=dtype,
                                           device=device))
        assert torch.allclose(array.ant_pos, expected, atol=1e-7)

    def test_element_field_has_table_power_at_candidate_boresight(self, device, precision):
        """Test per-candidate field rotation and Table 7.3-2 gain"""
        dtype = dtypes[precision]["torch"]["dtype"]
        array = HandheldUTArray(
            carrier_frequency=7e9,
            polarization="single",
            antenna_locations=(6,),
            precision=precision,
            device=device,
        )

        theta = torch.tensor([PI/2], dtype=dtype, device=device)
        phi = torch.tensor([PI/2], dtype=dtype, device=device)
        field = array.element_field(theta, phi)
        power = torch.sum(field**2, dim=-1)
        expected = torch.tensor(10**(5.3/10), dtype=dtype, device=device)

        assert field.shape == (1, 1, 2)
        assert torch.allclose(power.squeeze(), expected, rtol=1e-5)

    def test_element_field_applies_port_power_offsets(self, device, precision):
        """Test optional antenna-imbalance attenuation"""
        dtype = dtypes[precision]["torch"]["dtype"]
        array = HandheldUTArray(
            carrier_frequency=7e9,
            polarization="single",
            antenna_locations=(6,),
            port_power_offsets_db=[3.0],
            precision=precision,
            device=device,
        )

        theta = torch.tensor([PI/2], dtype=dtype, device=device)
        phi = torch.tensor([PI/2], dtype=dtype, device=device)
        field = array.element_field(theta, phi)
        power = torch.sum(field**2, dim=-1)
        expected = torch.tensor(10**((5.3 - 3.0)/10),
                                dtype=dtype,
                                device=device)

        assert torch.allclose(power.squeeze(), expected, rtol=1e-5)

    def test_channel_coefficients_use_handheld_element_fields(self, device, precision):
        """Test that channel coefficients accept per-port handheld fields"""
        dtype = dtypes[precision]["torch"]["dtype"]
        tx_array = Antenna(
            polarization="single",
            polarization_type="H",
            antenna_pattern="omni",
            carrier_frequency=7e9,
            precision=precision,
            device=device,
        )
        rx_array = HandheldUTArray(
            carrier_frequency=7e9,
            polarization="single",
            antenna_locations=(6,),
            precision=precision,
            device=device,
        )
        gen = ChannelCoefficientsGenerator(
            carrier_frequency=7e9,
            tx_array=tx_array,
            rx_array=rx_array,
            subclustering=False,
            precision=precision,
            device=device,
        )

        zeros = torch.zeros([1, 1, 3], dtype=dtype, device=device)
        link = torch.zeros([1, 1, 1], dtype=dtype, device=device)
        topology = Topology(
            velocities=zeros,
            moving_end="rx",
            los_aoa=link,
            los_aod=link,
            los_zoa=link,
            los_zod=link,
            los=torch.ones([1, 1, 1], dtype=torch.bool, device=device),
            distance_3d=torch.ones([1, 1, 1], dtype=dtype, device=device),
            tx_orientations=zeros,
            rx_orientations=zeros,
        )

        aoa = torch.tensor([[[[[PI/2]]]]], dtype=dtype, device=device)
        aod = torch.tensor([[[[[0.0]]]]], dtype=dtype, device=device)
        zoa = torch.tensor([[[[[PI/2]]]]], dtype=dtype, device=device)
        zod = torch.tensor([[[[[PI/2]]]]], dtype=dtype, device=device)
        h_phase = torch.eye(2, dtype=dtypes[precision]["torch"]["cdtype"],
                            device=device).reshape(1, 1, 1, 1, 1, 2, 2)

        h_field = gen._step_11_field_matrix(
            topology, aoa, aod, zoa, zod, h_phase
        )

        assert h_field.shape == (1, 1, 1, 1, 1, 1, 1)
        assert torch.abs(h_field).squeeze() > 0.0

    def test_channel_coefficients_rotate_handheld_ut_orientation(
        self, device, precision
    ):
        """Test UT orientation rotates handheld positions and field response."""
        dtype = dtypes[precision]["torch"]["dtype"]
        cdtype = dtypes[precision]["torch"]["cdtype"]
        tx_array = Antenna(
            polarization="single",
            polarization_type="H",
            antenna_pattern="omni",
            carrier_frequency=7e9,
            precision=precision,
            device=device,
        )
        rx_array = HandheldUTArray(
            carrier_frequency=7e9,
            polarization="single",
            antenna_locations=(6,),
            precision=precision,
            device=device,
        )
        gen = ChannelCoefficientsGenerator(
            carrier_frequency=7e9,
            tx_array=tx_array,
            rx_array=rx_array,
            subclustering=False,
            precision=precision,
            device=device,
        )

        zeros = torch.zeros([1, 1, 3], dtype=dtype, device=device)
        link = torch.zeros([1, 1, 1], dtype=dtype, device=device)

        def topology(rx_orientations):
            return Topology(
                velocities=zeros,
                moving_end="rx",
                los_aoa=link,
                los_aod=link,
                los_zoa=link,
                los_zod=link,
                los=torch.ones([1, 1, 1], dtype=torch.bool, device=device),
                distance_3d=torch.ones([1, 1, 1],
                                       dtype=dtype,
                                       device=device),
                tx_orientations=zeros,
                rx_orientations=rx_orientations,
            )

        yaw_90 = torch.tensor([[[PI/2, 0.0, 0.0]]],
                              dtype=dtype,
                              device=device)
        rx_pos = gen._step_11_get_rx_antenna_positions(topology(yaw_90))
        expected_pos = torch.tensor([[[[-0.035, 0.0, 0.0]]]],
                                    dtype=dtype,
                                    device=device)
        assert torch.allclose(rx_pos, expected_pos, atol=1e-6)

        aod = torch.tensor([[[[[0.0]]]]], dtype=dtype, device=device)
        zod = torch.tensor([[[[[PI/2]]]]], dtype=dtype, device=device)
        h_phase = torch.eye(2, dtype=cdtype, device=device)
        h_phase = h_phase.reshape(1, 1, 1, 1, 1, 2, 2)

        def field(rx_orientations, aoa, zoa):
            aoa = torch.tensor([[[[[aoa]]]]], dtype=dtype, device=device)
            zoa = torch.tensor([[[[[zoa]]]]], dtype=dtype, device=device)
            return gen._step_11_field_matrix(
                topology(rx_orientations), aoa, aod, zoa, zod, h_phase
            )

        reference = field(zeros, PI/2, PI/2)
        rotated = field(yaw_90, PI, PI/2)
        off_boresight = field(yaw_90, PI/2, PI/2)

        assert torch.allclose(rotated, reference, rtol=1e-5, atol=1e-6)
        assert torch.abs(rotated).squeeze() > torch.abs(off_boresight).squeeze()

    def test_show_methods(self, device, precision):
        """Test handheld array visualization helpers."""
        array = HandheldUTArray(
            carrier_frequency=7e9,
            polarization="dual",
            antenna_locations="tr38901-4",
            precision=precision,
            device=device,
        )

        plt.close("all")
        array.show()
        assert len(plt.get_fignums()) == 1
        ax = plt.gcf().axes[0]
        assert ax.get_title() == "Handheld UT Array"
        assert ax.get_xlabel() == "y (m)"
        assert ax.get_ylabel() == "x (m)"

        plt.close("all")
        array.show_element_radiation_pattern()
        assert len(plt.get_fignums()) == 3
        plt.close("all")

    def test_system_level_channel_accepts_handheld_ut_array(self, device, precision):
        """Test handheld arrays can be passed to system-level models."""
        bs_array = PanelArray(
            num_rows_per_panel=1,
            num_cols_per_panel=1,
            polarization="single",
            polarization_type="V",
            antenna_pattern="omni",
            carrier_frequency=7e9,
            precision=precision,
            device=device,
        )
        ut_array = HandheldUTArray(
            carrier_frequency=7e9,
            polarization="single",
            antenna_locations=(6,),
            precision=precision,
            device=device,
        )

        channel = UMi(
            carrier_frequency=7e9,
            o2i_model="low",
            ut_array=ut_array,
            bs_array=bs_array,
            direction="downlink",
            precision=precision,
            device=device,
        )

        assert channel._scenario.ut_array is ut_array
        assert channel._scenario.spec_version == "19.2"


class TestAntennaPanel:
    """Tests for the AntennaPanel class"""

    def test_antenna_positions_shape_single_pol(self, device, precision):
        """Test that antenna positions have correct shape for single polarization"""
        num_rows, num_cols = 4, 2
        panel = AntennaPanel(
            num_rows=num_rows,
            num_cols=num_cols,
            polarization="single",
            vertical_spacing=0.5,
            horizontal_spacing=0.5,
            precision=precision,
            device=device,
        )

        expected_num_ant = num_rows * num_cols
        assert panel.ant_pos.shape == (expected_num_ant, 3)

    def test_antenna_positions_shape_dual_pol(self, device, precision):
        """Test that antenna positions have correct shape for dual polarization"""
        num_rows, num_cols = 4, 2
        panel = AntennaPanel(
            num_rows=num_rows,
            num_cols=num_cols,
            polarization="dual",
            vertical_spacing=0.5,
            horizontal_spacing=0.5,
            precision=precision,
            device=device,
        )

        expected_num_ant = num_rows * num_cols * 2  # Double for dual polarization
        assert panel.ant_pos.shape == (expected_num_ant, 3)

    def test_antenna_positions_on_yz_plane(self, device, precision):
        """Test that all antennas lie on the y-z plane (x=0)"""
        panel = AntennaPanel(
            num_rows=3,
            num_cols=3,
            polarization="single",
            vertical_spacing=0.5,
            horizontal_spacing=0.5,
            precision=precision,
            device=device,
        )

        # All x coordinates should be 0
        assert torch.allclose(panel.ant_pos[:, 0], torch.zeros_like(panel.ant_pos[:, 0]))

    def test_properties(self, device, precision):
        """Test that panel properties are correctly set"""
        dtype = dtypes[precision]["torch"]["dtype"]
        panel = AntennaPanel(
            num_rows=4,
            num_cols=3,
            polarization="dual",
            vertical_spacing=0.7,
            horizontal_spacing=0.6,
            precision=precision,
            device=device,
        )

        assert panel.num_rows == 4
        assert panel.num_cols == 3
        assert panel.polarization == "dual"
        assert torch.isclose(panel.vertical_spacing, torch.tensor(0.7, dtype=dtype, device=device))
        assert torch.isclose(panel.horizontal_spacing, torch.tensor(0.6, dtype=dtype, device=device))


class TestPanelArray:
    """Tests for the PanelArray class"""

    def test_invalid_polarization_type_preserves_context(self, device):
        """Validation errors identify the selected polarization mode."""
        kwargs = {
            "num_rows_per_panel": 1,
            "num_cols_per_panel": 1,
            "antenna_pattern": "omni",
            "carrier_frequency": 3.5e9,
            "device": device,
        }
        with pytest.raises(ValueError, match="single polarization"):
            PanelArray(
                polarization="single",
                polarization_type="cross",
                **kwargs,
            )
        with pytest.raises(ValueError, match="dual polarization"):
            PanelArray(
                polarization="dual",
                polarization_type="V",
                **kwargs,
            )

    def test_num_antennas_single_panel_single_pol(self, device, precision):
        """Test total antenna count for single panel, single polarization"""
        array = PanelArray(
            num_rows_per_panel=4,
            num_cols_per_panel=4,
            polarization="single",
            polarization_type="V",
            antenna_pattern="omni",
            carrier_frequency=3.5e9,
            precision=precision,
            device=device,
        )

        assert array.num_ant == 16
        assert array.num_panels == 1
        assert array.num_panels_ant == 16

    def test_num_antennas_single_panel_dual_pol(self, device, precision):
        """Test total antenna count for single panel, dual polarization"""
        array = PanelArray(
            num_rows_per_panel=4,
            num_cols_per_panel=4,
            polarization="dual",
            polarization_type="VH",
            antenna_pattern="omni",
            carrier_frequency=3.5e9,
            precision=precision,
            device=device,
        )

        assert array.num_ant == 32  # 16 * 2 for dual polarization
        assert array.num_panels == 1
        assert array.num_panels_ant == 32

    def test_num_antennas_multiple_panels(self, device, precision):
        """Test total antenna count for multiple panels"""
        array = PanelArray(
            num_rows_per_panel=2,
            num_cols_per_panel=2,
            polarization="dual",
            polarization_type="cross",
            antenna_pattern="38.901",
            carrier_frequency=3.5e9,
            num_rows=2,
            num_cols=2,
            precision=precision,
            device=device,
        )

        # 2x2 elements per panel, dual polarization = 8 antennas per panel
        # 2x2 panels = 4 panels
        assert array.num_ant == 32
        assert array.num_panels == 4
        assert array.num_panels_ant == 8

    def test_antenna_positions_scaled_by_wavelength(self, device, precision):
        """Test that antenna positions are scaled by wavelength"""
        dtype = dtypes[precision]["torch"]["dtype"]
        carrier_frequency = 3e9
        wavelength = SPEED_OF_LIGHT / carrier_frequency

        array = PanelArray(
            num_rows_per_panel=2,
            num_cols_per_panel=2,
            polarization="single",
            polarization_type="V",
            antenna_pattern="omni",
            carrier_frequency=carrier_frequency,
            element_vertical_spacing=0.5,
            element_horizontal_spacing=0.5,
            precision=precision,
            device=device,
        )

        # Maximum spacing between antennas in wavelengths should be 0.5
        # In meters, this should be 0.5 * wavelength
        max_y_diff = array.ant_pos[:, 1].max() - array.ant_pos[:, 1].min()
        expected_max_y = 0.5 * wavelength  # 2 columns, 0.5 spacing

        assert torch.isclose(max_y_diff, torch.tensor(expected_max_y, dtype=dtype, device=device), rtol=1e-5)

    def test_polarization_indices_single(self, device, precision):
        """Test that polarization indices are correct for single polarization"""
        array = PanelArray(
            num_rows_per_panel=2,
            num_cols_per_panel=2,
            polarization="single",
            polarization_type="H",
            antenna_pattern="omni",
            carrier_frequency=3.5e9,
            precision=precision,
            device=device,
        )

        assert array.ant_ind_pol1.shape[0] == 4
        # For single polarization, pol2 should be empty
        assert array._ant_ind_pol2.numel() == 0

    def test_polarization_indices_dual(self, device, precision):
        """Test that polarization indices are correct for dual polarization"""
        array = PanelArray(
            num_rows_per_panel=2,
            num_cols_per_panel=2,
            polarization="dual",
            polarization_type="VH",
            antenna_pattern="omni",
            carrier_frequency=3.5e9,
            precision=precision,
            device=device,
        )

        # Each polarization should have half the antennas
        assert array.ant_ind_pol1.shape[0] == 4
        assert array.ant_ind_pol2.shape[0] == 4

    def test_docstring_example(self, device):
        """Test that the example from the docstring works correctly"""
        array = PanelArray(
            num_rows_per_panel=4,
            num_cols_per_panel=4,
            polarization='dual',
            polarization_type='VH',
            antenna_pattern='38.901',
            carrier_frequency=3.5e9,
            num_cols=2,
            panel_horizontal_spacing=3.,
            device=device,
        )

        # Should create without errors and have correct number of antennas
        # 4x4 elements, dual pol = 32 per panel, 2 panels = 64 total
        assert array.num_ant == 64

    def test_polarization_types(self, device, precision):
        """Test that different polarization types are handled correctly"""
        # Single V
        array_v = PanelArray(
            num_rows_per_panel=2,
            num_cols_per_panel=2,
            polarization="single",
            polarization_type="V",
            antenna_pattern="omni",
            carrier_frequency=3.5e9,
            precision=precision,
            device=device,
        )
        assert array_v.polarization_type == "V"

        # Single H
        array_h = PanelArray(
            num_rows_per_panel=2,
            num_cols_per_panel=2,
            polarization="single",
            polarization_type="H",
            antenna_pattern="omni",
            carrier_frequency=3.5e9,
            precision=precision,
            device=device,
        )
        assert array_h.polarization_type == "H"

        # Dual VH
        array_vh = PanelArray(
            num_rows_per_panel=2,
            num_cols_per_panel=2,
            polarization="dual",
            polarization_type="VH",
            antenna_pattern="omni",
            carrier_frequency=3.5e9,
            precision=precision,
            device=device,
        )
        assert array_vh.polarization_type == "VH"

        # Dual cross
        array_cross = PanelArray(
            num_rows_per_panel=2,
            num_cols_per_panel=2,
            polarization="dual",
            polarization_type="cross",
            antenna_pattern="omni",
            carrier_frequency=3.5e9,
            precision=precision,
            device=device,
        )
        assert array_cross.polarization_type == "cross"


class TestAntenna:
    """Tests for the Antenna class (single element)"""

    def test_single_element(self, device, precision):
        """Test that Antenna creates a single-element array"""
        ant = Antenna(
            polarization="single",
            polarization_type="V",
            antenna_pattern="omni",
            carrier_frequency=3.5e9,
            precision=precision,
            device=device,
        )

        assert ant.num_ant == 1
        assert ant.num_rows_per_panel == 1
        assert ant.num_cols_per_panel == 1

    def test_dual_pol_single_element(self, device, precision):
        """Test dual polarization creates two antenna elements"""
        ant = Antenna(
            polarization="dual",
            polarization_type="cross",
            antenna_pattern="38.901",
            carrier_frequency=3.5e9,
            precision=precision,
            device=device,
        )

        assert ant.num_ant == 2
        assert ant.ant_ind_pol1.shape[0] == 1
        assert ant.ant_ind_pol2.shape[0] == 1

    def test_docstring_example(self, device):
        """Test that the example from the docstring works correctly"""
        ant = Antenna(
            polarization='single',
            polarization_type='V',
            antenna_pattern='omni',
            carrier_frequency=3.5e9,
            device=device,
        )
        assert ant.num_ant == 1

    def test_inheritance(self, device, precision):
        """Test that Antenna inherits from PanelArray"""
        ant = Antenna(
            polarization="single",
            polarization_type="V",
            antenna_pattern="omni",
            carrier_frequency=3.5e9,
            precision=precision,
            device=device,
        )

        assert isinstance(ant, PanelArray)


class TestAntennaArray:
    """Tests for the AntennaArray class"""

    def test_array_dimensions(self, device, precision):
        """Test that AntennaArray has correct dimensions"""
        array = AntennaArray(
            num_rows=4,
            num_cols=4,
            polarization="dual",
            polarization_type="cross",
            antenna_pattern="38.901",
            carrier_frequency=3.5e9,
            precision=precision,
            device=device,
        )

        assert array.num_ant == 32  # 4x4 * 2 polarizations
        assert array.num_rows_per_panel == 4
        assert array.num_cols_per_panel == 4

    def test_custom_spacing(self, device, precision):
        """Test that custom element spacing is applied"""
        dtype = dtypes[precision]["torch"]["dtype"]
        array = AntennaArray(
            num_rows=2,
            num_cols=2,
            polarization="single",
            polarization_type="V",
            antenna_pattern="omni",
            carrier_frequency=3.5e9,
            vertical_spacing=0.7,
            horizontal_spacing=0.3,
            precision=precision,
            device=device,
        )

        assert torch.isclose(array.element_vertical_spacing, torch.tensor(0.7, dtype=dtype, device=device))
        assert torch.isclose(array.element_horizontal_spacing, torch.tensor(0.3, dtype=dtype, device=device))

    def test_docstring_example(self, device):
        """Test that the example from the docstring works correctly"""
        array = AntennaArray(
            num_rows=4,
            num_cols=4,
            polarization='dual',
            polarization_type='cross',
            antenna_pattern='38.901',
            carrier_frequency=3.5e9,
            device=device,
        )
        assert array.num_ant == 32

    def test_inheritance(self, device, precision):
        """Test that AntennaArray inherits from PanelArray"""
        array = AntennaArray(
            num_rows=2,
            num_cols=2,
            polarization="single",
            polarization_type="V",
            antenna_pattern="omni",
            carrier_frequency=3.5e9,
            precision=precision,
            device=device,
        )

        assert isinstance(array, PanelArray)
