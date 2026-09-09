#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Tests for 3GPP TR 38.901 ChannelCoefficientsGenerator classes"""

import numpy as np
import torch

from sionna.phy import PI
from sionna.phy.channel.tr38901 import (
    Topology,
    ChannelCoefficientsGenerator,
    PanelArray,
)


class TestTopology:
    """Tests for the Topology data class"""

    def test_topology_instantiation(self, device, precision):
        """Test that Topology can be instantiated"""
        batch_size = 10
        num_bs = 3
        num_ut = 5

        velocities = torch.randn(batch_size, num_ut, 3, device=device)
        los_aoa = torch.rand(batch_size, num_bs, num_ut, device=device) * PI
        los_aod = torch.rand(batch_size, num_bs, num_ut, device=device) * PI
        los_zoa = torch.rand(batch_size, num_bs, num_ut, device=device) * PI
        los_zod = torch.rand(batch_size, num_bs, num_ut, device=device) * PI
        los = torch.rand(batch_size, num_bs, num_ut, device=device) > 0.5
        distance_3d = torch.rand(batch_size, num_bs, num_ut, device=device) * 1000
        tx_orientations = torch.rand(batch_size, num_bs, 3, device=device) * 2 * PI
        rx_orientations = torch.rand(batch_size, num_ut, 3, device=device) * 2 * PI

        topology = Topology(
            velocities=velocities,
            moving_end="rx",
            los_aoa=los_aoa,
            los_aod=los_aod,
            los_zoa=los_zoa,
            los_zod=los_zod,
            los=los,
            distance_3d=distance_3d,
            tx_orientations=tx_orientations,
            rx_orientations=rx_orientations,
        )

        assert torch.equal(topology.velocities, velocities)
        assert topology.moving_end == "rx"
        assert torch.equal(topology.los_aoa, los_aoa)
        assert torch.equal(topology.los_aod, los_aod)
        assert torch.equal(topology.los_zoa, los_zoa)
        assert torch.equal(topology.los_zod, los_zod)
        assert torch.equal(topology.los, los)
        assert torch.equal(topology.distance_3d, distance_3d)
        assert torch.equal(topology.tx_orientations, tx_orientations)
        assert torch.equal(topology.rx_orientations, rx_orientations)

    def test_topology_moving_end_options(self, device, precision):
        """Test that Topology accepts both 'tx' and 'rx' for moving_end"""
        batch_size = 5
        num_bs = 2
        num_ut = 3

        velocities = torch.randn(batch_size, num_ut, 3, device=device)
        los_aoa = torch.rand(batch_size, num_bs, num_ut, device=device)
        los_aod = torch.rand(batch_size, num_bs, num_ut, device=device)
        los_zoa = torch.rand(batch_size, num_bs, num_ut, device=device)
        los_zod = torch.rand(batch_size, num_bs, num_ut, device=device)
        los = torch.rand(batch_size, num_bs, num_ut, device=device) > 0.5
        distance_3d = torch.rand(batch_size, num_bs, num_ut, device=device)
        tx_orientations = torch.rand(batch_size, num_bs, 3, device=device)
        rx_orientations = torch.rand(batch_size, num_ut, 3, device=device)

        # Test 'rx' moving end
        topology_rx = Topology(
            velocities=velocities,
            moving_end="rx",
            los_aoa=los_aoa,
            los_aod=los_aod,
            los_zoa=los_zoa,
            los_zod=los_zod,
            los=los,
            distance_3d=distance_3d,
            tx_orientations=tx_orientations,
            rx_orientations=rx_orientations,
        )
        assert topology_rx.moving_end == "rx"

        # Test 'tx' moving end
        velocities_tx = torch.randn(batch_size, num_bs, 3, device=device)
        topology_tx = Topology(
            velocities=velocities_tx,
            moving_end="tx",
            los_aoa=los_aoa,
            los_aod=los_aod,
            los_zoa=los_zoa,
            los_zod=los_zod,
            los=los,
            distance_3d=distance_3d,
            tx_orientations=tx_orientations,
            rx_orientations=rx_orientations,
        )
        assert topology_tx.moving_end == "tx"


class TestChannelCoefficientsGenerator:
    """Tests for the ChannelCoefficientsGenerator class"""

    CARRIER_FREQUENCY = 3.5e9  # Hz
    MAX_ERR = 1e-2

    def _create_generator(self, device, precision):
        """Create a ChannelCoefficientsGenerator for testing"""
        tx_array = PanelArray(
            num_rows_per_panel=2,
            num_cols_per_panel=2,
            polarization="dual",
            polarization_type="VH",
            antenna_pattern="38.901",
            carrier_frequency=self.CARRIER_FREQUENCY,
            precision=precision,
            device=device,
        )
        rx_array = PanelArray(
            num_rows_per_panel=1,
            num_cols_per_panel=1,
            polarization="dual",
            polarization_type="VH",
            antenna_pattern="38.901",
            carrier_frequency=self.CARRIER_FREQUENCY,
            precision=precision,
            device=device,
        )

        ccg = ChannelCoefficientsGenerator(
            carrier_frequency=self.CARRIER_FREQUENCY,
            tx_array=tx_array,
            rx_array=rx_array,
            subclustering=False,
            precision=precision,
            device=device,
        )

        return ccg, tx_array, rx_array

    def test_unit_sphere_vector(self, device, precision):
        """Test unit sphere vector computation"""
        ccg, _, _ = self._create_generator(device, precision)

        batch_size = 32
        theta = torch.randn(batch_size, device=device, dtype=ccg.dtype)
        phi = torch.randn(batch_size, device=device, dtype=ccg.dtype)

        uvec = ccg._unit_sphere_vector(theta, phi)

        # Check shape
        assert uvec.shape == (batch_size, 3, 1)

        # Check that vectors have unit norm
        norms = torch.sqrt((uvec ** 2).sum(dim=1)).squeeze()
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_unit_sphere_vector_reference(self, device, precision):
        """Test unit sphere vector against reference implementation"""
        ccg, _, _ = self._create_generator(device, precision)

        batch_size = 32
        theta = torch.randn(batch_size, device=device, dtype=ccg.dtype)
        phi = torch.randn(batch_size, device=device, dtype=ccg.dtype)

        uvec = ccg._unit_sphere_vector(theta, phi)

        # Reference implementation
        uvec_ref = torch.stack([
            torch.sin(theta) * torch.cos(phi),
            torch.sin(theta) * torch.sin(phi),
            torch.cos(theta),
        ], dim=-1).unsqueeze(-1)

        assert torch.allclose(uvec, uvec_ref, atol=1e-5)

    def test_forward_rotation_matrix(self, device, precision):
        """Test forward rotation matrix computation"""
        ccg, _, _ = self._create_generator(device, precision)

        batch_size = 32
        orientations = torch.randn(batch_size, 3, device=device, dtype=ccg.dtype)

        rot_mat = ccg._forward_rotation_matrix(orientations)

        # Check shape
        assert rot_mat.shape == (batch_size, 3, 3)

        # Check that rotation matrices are orthogonal (R^T R = I)
        identity = torch.eye(3, device=device, dtype=ccg.dtype).expand(batch_size, -1, -1)
        product = rot_mat.mT @ rot_mat
        assert torch.allclose(product, identity, atol=1e-5)

    def test_reverse_rotation_matrix(self, device, precision):
        """Test reverse rotation matrix is inverse of forward"""
        ccg, _, _ = self._create_generator(device, precision)

        batch_size = 32
        orientations = torch.randn(batch_size, 3, device=device, dtype=ccg.dtype)

        rot_fwd = ccg._forward_rotation_matrix(orientations)
        rot_rev = ccg._reverse_rotation_matrix(orientations)

        # Check that reverse is inverse of forward
        identity = torch.eye(3, device=device, dtype=ccg.dtype).expand(batch_size, -1, -1)
        product = rot_fwd @ rot_rev
        assert torch.allclose(product, identity, atol=1e-5)

    def test_gcs_to_lcs(self, device, precision):
        """Test GCS to LCS angle transformation"""
        ccg, _, _ = self._create_generator(device, precision)

        batch_size = 32
        orientations = torch.randn(batch_size, 3, device=device, dtype=ccg.dtype)
        theta = torch.rand(batch_size, device=device, dtype=ccg.dtype) * PI
        phi = (torch.rand(batch_size, device=device, dtype=ccg.dtype) * 2 - 1) * PI

        theta_prime, phi_prime = ccg._gcs_to_lcs(orientations, theta, phi)

        # Check output shapes
        assert theta_prime.shape == theta.shape
        assert phi_prime.shape == phi.shape

        # Check that zenith is in valid range
        assert torch.all(theta_prime >= 0)
        assert torch.all(theta_prime <= PI)

    def test_rot_pos(self, device, precision):
        """Test position rotation"""
        ccg, _, _ = self._create_generator(device, precision)

        batch_size = 32
        orientations = torch.randn(batch_size, 3, device=device, dtype=ccg.dtype)
        positions = torch.randn(batch_size, 3, 1, device=device, dtype=ccg.dtype)

        rotated = ccg._rot_pos(orientations, positions)

        # Check shape
        assert rotated.shape == positions.shape

        # Check that norm is preserved
        orig_norms = torch.sqrt((positions ** 2).sum(dim=1))
        rot_norms = torch.sqrt((rotated ** 2).sum(dim=1))
        assert torch.allclose(orig_norms, rot_norms, atol=1e-5)

    def test_step_10_phase_generation(self, device, precision):
        """Test that step 10 generates random phases"""
        ccg, _, _ = self._create_generator(device, precision)

        batch_size = 5
        shape = (batch_size, 2, 3, 10, 4)  # batch, tx, rx, clusters, rays
        phi = ccg._step_10(shape)

        # Check shape (adds 4 for polarization combinations)
        assert phi.shape == (*shape, 4)

        # Check phases are in valid range (-pi, pi)
        assert torch.all(phi >= -PI)
        assert torch.all(phi <= PI)

    def test_los_delay_phase_sign(self, device, precision):
        """Pin the TR 38.901 Eq. 7.5-29 LoS propagation phase sign."""
        dtype = torch.float64 if precision == "double" else torch.float32
        tx_array = PanelArray(
            num_rows_per_panel=1,
            num_cols_per_panel=1,
            polarization="single",
            polarization_type="V",
            antenna_pattern="omni",
            carrier_frequency=self.CARRIER_FREQUENCY,
            precision=precision,
            device=device,
        )
        rx_array = PanelArray(
            num_rows_per_panel=1,
            num_cols_per_panel=1,
            polarization="single",
            polarization_type="V",
            antenna_pattern="omni",
            carrier_frequency=self.CARRIER_FREQUENCY,
            precision=precision,
            device=device,
        )
        ccg = ChannelCoefficientsGenerator(
            carrier_frequency=self.CARRIER_FREQUENCY,
            tx_array=tx_array,
            rx_array=rx_array,
            subclustering=False,
            precision=precision,
            device=device,
        )
        distance = torch.stack(
            [
                torch.zeros((), dtype=dtype, device=device),
                ccg._lambda_0.to(dtype) / 4.0,
            ]
        ).reshape(1, 1, 2)
        angles = torch.full((1, 1, 2), PI / 2, dtype=dtype, device=device)
        topology = Topology(
            velocities=torch.zeros(1, 2, 3, dtype=dtype, device=device),
            moving_end="rx",
            los_aoa=angles,
            los_aod=angles,
            los_zoa=angles,
            los_zod=angles,
            los=torch.ones(1, 1, 2, dtype=torch.bool, device=device),
            distance_3d=distance,
            tx_orientations=torch.zeros(1, 1, 3, dtype=dtype, device=device),
            rx_orientations=torch.zeros(1, 2, 3, dtype=dtype, device=device),
        )
        h_los = ccg._step_11_los(
            topology,
            torch.zeros(1, dtype=dtype, device=device),
        )
        ratio = h_los[0, 0, 1, 0, 0, 0, 0] / h_los[0, 0, 0, 0, 0, 0, 0]
        expected = torch.exp(
            torch.tensor(-0.5j*np.pi, dtype=ratio.dtype, device=device)
        )
        torch.testing.assert_close(ratio, expected)

    def test_doppler_uses_direction_at_moving_end(self, device, precision):
        """Check Eqs. 7.6-45/46 use arrival or departure consistently."""
        ccg, _, _ = self._create_generator(device, precision)
        dtype = ccg.dtype
        shape = (1, 1, 1, 1, 1)
        aoa = torch.full(shape, PI / 2, dtype=dtype, device=device)
        aod = torch.zeros(shape, dtype=dtype, device=device)
        zoa = torch.full(shape, PI / 2, dtype=dtype, device=device)
        zod = torch.full(shape, PI / 2, dtype=dtype, device=device)
        sample_time = (ccg._lambda_0 / 4.0).reshape(1)

        common = dict(
            los_aoa=aoa.squeeze(-1).squeeze(-1),
            los_aod=aod.squeeze(-1).squeeze(-1),
            los_zoa=zoa.squeeze(-1).squeeze(-1),
            los_zod=zod.squeeze(-1).squeeze(-1),
            los=torch.ones(1, 1, 1, dtype=torch.bool, device=device),
            distance_3d=torch.ones(1, 1, 1, dtype=dtype, device=device),
            tx_orientations=torch.zeros(1, 1, 3, dtype=dtype, device=device),
            rx_orientations=torch.zeros(1, 1, 3, dtype=dtype, device=device),
        )
        velocity_x = torch.tensor(
            [[[1.0, 0.0, 0.0]]], dtype=dtype, device=device
        )
        tx_topology = Topology(
            velocities=velocity_x, moving_end="tx", **common
        )
        tx_doppler = ccg._step_11_doppler_matrix(
            tx_topology, aoa, aod, zoa, zod, sample_time
        )
        expected_tx = torch.tensor(1j, dtype=ccg.cdtype, device=device)
        torch.testing.assert_close(tx_doppler.squeeze(), expected_tx)

        rx_topology = Topology(
            velocities=velocity_x, moving_end="rx", **common
        )
        rx_doppler = ccg._step_11_doppler_matrix(
            rx_topology, aoa, aod, zoa, zod, sample_time
        )
        torch.testing.assert_close(
            rx_doppler.squeeze(),
            torch.ones((), dtype=ccg.cdtype, device=device),
        )

    def test_step_11_get_tx_antenna_positions(self, device, precision):
        """Test TX antenna position computation"""
        ccg, tx_array, _ = self._create_generator(device, precision)

        batch_size = 5
        num_tx = 2

        tx_orientations = torch.zeros(batch_size, num_tx, 3, device=device, dtype=ccg.dtype)
        velocities = torch.zeros(batch_size, 3, 3, device=device, dtype=ccg.dtype)
        los_aoa = torch.zeros(batch_size, num_tx, 3, device=device, dtype=ccg.dtype)
        los_aod = torch.zeros(batch_size, num_tx, 3, device=device, dtype=ccg.dtype)
        los_zoa = torch.zeros(batch_size, num_tx, 3, device=device, dtype=ccg.dtype)
        los_zod = torch.zeros(batch_size, num_tx, 3, device=device, dtype=ccg.dtype)
        los = torch.zeros(batch_size, num_tx, 3, dtype=torch.bool, device=device)
        distance_3d = torch.ones(batch_size, num_tx, 3, device=device, dtype=ccg.dtype)
        rx_orientations = torch.zeros(batch_size, 3, 3, device=device, dtype=ccg.dtype)

        topology = Topology(
            velocities=velocities,
            moving_end="rx",
            los_aoa=los_aoa,
            los_aod=los_aod,
            los_zoa=los_zoa,
            los_zod=los_zod,
            los=los,
            distance_3d=distance_3d,
            tx_orientations=tx_orientations,
            rx_orientations=rx_orientations,
        )

        tx_pos = ccg._step_11_get_tx_antenna_positions(topology)

        # Check shape: [batch, num_tx, num_tx_ant, 3]
        assert tx_pos.shape == (batch_size, num_tx, tx_array.num_ant, 3)

    def test_step_11_get_rx_antenna_positions(self, device, precision):
        """Test RX antenna position computation"""
        ccg, _, rx_array = self._create_generator(device, precision)

        batch_size = 5
        num_rx = 3

        rx_orientations = torch.zeros(batch_size, num_rx, 3, device=device, dtype=ccg.dtype)
        tx_orientations = torch.zeros(batch_size, 2, 3, device=device, dtype=ccg.dtype)
        velocities = torch.zeros(batch_size, num_rx, 3, device=device, dtype=ccg.dtype)
        los_aoa = torch.zeros(batch_size, 2, num_rx, device=device, dtype=ccg.dtype)
        los_aod = torch.zeros(batch_size, 2, num_rx, device=device, dtype=ccg.dtype)
        los_zoa = torch.zeros(batch_size, 2, num_rx, device=device, dtype=ccg.dtype)
        los_zod = torch.zeros(batch_size, 2, num_rx, device=device, dtype=ccg.dtype)
        los = torch.zeros(batch_size, 2, num_rx, dtype=torch.bool, device=device)
        distance_3d = torch.ones(batch_size, 2, num_rx, device=device, dtype=ccg.dtype)

        topology = Topology(
            velocities=velocities,
            moving_end="rx",
            los_aoa=los_aoa,
            los_aod=los_aod,
            los_zoa=los_zoa,
            los_zod=los_zod,
            los=los,
            distance_3d=distance_3d,
            tx_orientations=tx_orientations,
            rx_orientations=rx_orientations,
        )

        rx_pos = ccg._step_11_get_rx_antenna_positions(topology)

        # Check shape: [batch, num_rx, num_rx_ant, 3]
        assert rx_pos.shape == (batch_size, num_rx, rx_array.num_ant, 3)


class TestSystemLevelChannelCompiled:
    """Tests for torch.compile compatibility of system-level channel models.
    
    These tests specifically verify that the Inductor backend (which uses Triton)
    works correctly with complex tensor operations in the 3GPP channel models.
    CPU tests are skipped because Inductor falls back to eager mode on CPU
    and the specific issue (torch.tensor with complex dtype) only manifests
    with the Triton code generation.
    """

    def test_umi_compiled(self, device):
        if device == "cpu":
            import pytest
            pytest.skip("Inductor/Triton tests only relevant on GPU")
        """Verify UMi channel model works with torch.compile"""
        from sionna.phy.channel.tr38901 import PanelArray, UMi
        from sionna.phy.channel import gen_single_sector_topology

        bs_array = PanelArray(
            num_rows_per_panel=2, num_cols_per_panel=2,
            polarization='single', polarization_type='V',
            antenna_pattern='omni', carrier_frequency=3.5e9
        )
        ut_array = PanelArray(
            num_rows_per_panel=1, num_cols_per_panel=1,
            polarization='single', polarization_type='V',
            antenna_pattern='omni', carrier_frequency=3.5e9
        )

        channel_model = UMi(
            carrier_frequency=3.5e9, o2i_model='low',
            ut_array=ut_array, bs_array=bs_array,
            direction='uplink', device=device
        )

        topology = gen_single_sector_topology(4, 1, 'umi', device=device)
        channel_model.set_topology(*topology)

        @torch.compile
        def generate_channel():
            return channel_model(99, 10, 1e6)

        h, tau = generate_channel()
        assert h.shape[0] == 4  # batch size
        assert h.shape[-1] == 10  # num time samples

    def test_tdl_compiled(self, device):
        """Verify TDL channel model works with torch.compile"""
        from sionna.phy.channel.tr38901 import TDL

        tdl = TDL(
            model="A", delay_spread=100e-9, carrier_frequency=3.5e9,
            min_speed=0.0, max_speed=10.0, device=device
        )

        @torch.compile
        def generate_channel():
            return tdl(batch_size=32, num_time_steps=100, sampling_frequency=1e6)

        h, delays = generate_channel()
        assert h.shape[0] == 32  # batch size
        assert h.shape[-1] == 100  # num time steps

    def test_cdl_reduce_overhead_compiled(self, device):
        """Verify CDL works with CUDA graphs and no pre-allocation API."""
        if device == "cpu":
            import pytest
            pytest.skip("Inductor/Triton tests only relevant on GPU")
        import torch._dynamo as dynamo
        from torch._dynamo.utils import counters
        from sionna.phy.channel.tr38901 import CDL, PanelArray

        bs_array = PanelArray(
            num_rows_per_panel=1, num_cols_per_panel=1,
            polarization='single', polarization_type='V',
            antenna_pattern='omni', carrier_frequency=3.5e9
        )
        ut_array = PanelArray(
            num_rows_per_panel=1, num_cols_per_panel=1,
            polarization='single', polarization_type='V',
            antenna_pattern='omni', carrier_frequency=3.5e9
        )

        cdl = CDL(
            model="A", delay_spread=100e-9, carrier_frequency=3.5e9,
            ut_array=ut_array, bs_array=bs_array,
            min_speed=0.0, max_speed=10.0, direction='uplink', device=device
        )

        def generate_channel():
            return cdl(batch_size=2, num_time_steps=10,
                       sampling_frequency=1e6)

        dynamo.reset()
        counters.clear()
        compiled_generate_channel = torch.compile(
            generate_channel, mode="reduce-overhead"
        )
        h, delays = compiled_generate_channel()
        h, delays = compiled_generate_channel()
        torch.cuda.synchronize(torch.device(device))

        assert h.shape[0] == 2
        assert h.shape[-1] == 10
        assert not dict(counters.get("graph_break", {}))
        assert counters["inductor"]["cudagraph_skips"] == 0
