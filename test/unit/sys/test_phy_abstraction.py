#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Unit tests for sionna.sys.phy_abstraction"""

import numpy as np
import pytest
import torch

from sionna.phy import config
from sionna.phy.utils import DeepUpdateDict, random_tensor_from_values
from sionna.sys import PHYAbstraction


class TestPHYAbstraction:
    """Tests for the PHYAbstraction class."""

    def test_get_idx_from_grid_uses_object_device(self, device):
        """Direct helper calls follow the owning object's device and precision."""
        with pytest.warns(UserWarning):
            phy_abs = PHYAbstraction(
                load_bler_tables_from="",
                precision="double",
                device=device,
            )

        scalar_idx = phy_abs.get_idx_from_grid(5.0, "snr")
        tensor_idx = phy_abs.get_idx_from_grid(
            torch.tensor(5.0, dtype=torch.float32, device="cpu"), "snr"
        )

        for idx in (scalar_idx, tensor_idx):
            assert idx.dtype == torch.int32
            assert idx.device == torch.device(device)
        torch.testing.assert_close(scalar_idx, tensor_idx)

    def test_write_and_load(self, device, tmp_path):
        """Test the SNR to BER/BLER table generation."""
        sim_set_1 = {
            "category": {
                0: {"index": {1: {"MCS": [10, 24]}, 2: {"MCS": [12]}}}
            }
        }
        snr_dbs_1 = [0, 20]
        cb_sizes_1 = [50, 100, 150]
        path = tmp_path / "test.json"
        filename = str(path)

        # Start from no loaded table
        with pytest.warns(UserWarning):
            phy_abs = PHYAbstraction(load_bler_tables_from="", device=device)

        # Compute tables and save them to file
        table_1 = phy_abs.new_bler_table(
            snr_dbs_1,
            cb_sizes_1,
            sim_set_1,
            filename=filename,
            max_mc_iter=15,
            batch_size=10,
            verbose=False,
        )

        # Check that results have been written to file
        assert path.is_file(), "File was not created"
        assert path.stat().st_size > 0, "File is empty"

        # Load tables
        table_loaded = PHYAbstraction.load_table(filename)

        # Check that the two tables (dumped and loaded) are equal
        for category in table_1["category"]:
            for table_index in table_1["category"][category]["index"]:
                for mcs in table_1["category"][category]["index"][table_index]["MCS"]:
                    res_mcs = table_1["category"][category]["index"][table_index][
                        "MCS"
                    ][mcs]
                    res_mcs1 = table_loaded["category"][category]["index"][
                        table_index
                    ]["MCS"][mcs]

                    for a, b in zip(res_mcs["SNR_db"], res_mcs1["SNR_db"]):
                        assert a == b, "SNR_db mismatch"

                    for cbs in res_mcs["CBS"]:
                        res = res_mcs["CBS"][cbs]
                        res1 = res_mcs1["CBS"][cbs]

                        for a, b in zip(res["BLER"], res1["BLER"]):
                            assert a == b, "BLER mismatch"

        # Append another MCS to the existing category and table index.
        sim_set_2 = {
            "category": {0: {"index": {1: {"MCS": [11]}}}}
        }
        phy_abs.new_bler_table(
            snr_dbs_1,
            cb_sizes_1,
            sim_set_2,
            filename=filename,
            write_mode="a",
            max_mc_iter=15,
            batch_size=10,
            verbose=False,
        )
        table_appended = PHYAbstraction.load_table(filename)

        # Existing table-index and MCS subtrees must survive the append.
        assert set(table_appended["category"][0]["index"]) == {1, 2}
        assert set(
            table_appended["category"][0]["index"][1]["MCS"]
        ) == {10, 11, 24}
        assert set(
            table_appended["category"][0]["index"][2]["MCS"]
        ) == {12}
        for table_index, table in table_loaded["category"][0]["index"].items():
            for mcs, result in table["MCS"].items():
                assert (
                    table_appended["category"][0]["index"][table_index][
                        "MCS"
                    ][mcs]
                    == result
                )

    def test_bler_interpolation(self, device):
        """Validate the (CBS, SNR) -> BLER interpolation."""
        categories = [1, 1, 1]
        table_index = [1, 1, 1]
        mcs = [10, 15, 16]

        # Instantiate the PHY abstraction object
        phy_abs = PHYAbstraction(device=device)

        assert len(categories) == len(table_index)
        assert len(table_index) == len(mcs)

        for k in range(len(categories)):
            table_tmp = phy_abs.bler_table["category"][categories[k]]["index"][
                table_index[k]
            ]["MCS"][mcs[k]]

            # SNR/CBS values at which tables have been simulated
            snr_dbs_sim = table_tmp["SNR_db"]
            cb_sizes_sim = list(table_tmp["CBS"].keys())

            # Redefine the interpolation grid
            phy_abs.cbs_interp_min_max_delta = (
                cb_sizes_sim[0],
                cb_sizes_sim[-1],
                (cb_sizes_sim[1] - cb_sizes_sim[0]) // 10,
            )
            phy_abs.snr_db_interp_min_max_delta = (
                snr_dbs_sim[0],
                snr_dbs_sim[-1],
                (snr_dbs_sim[1] - snr_dbs_sim[0]) / 10,
            )

            # Interpolated table
            table_interp = phy_abs.bler_table_interp.cpu().numpy()[
                categories[k], table_index[k] - 1, mcs[k], ::
            ]

            for cbs in cb_sizes_sim:
                cbs_interp_ind = np.argmin(abs(phy_abs._cbs_interp - cbs))
                bler_sim = table_tmp["CBS"][cbs]["BLER"]
                for ii, snr in enumerate(snr_dbs_sim):
                    snr_interp_ind = np.argmin(abs(phy_abs._snr_dbs_interp - snr))

                    bler_interp = table_interp[cbs_interp_ind, snr_interp_ind]

                    # Check that interpolated value and original value coincide
                    assert abs(bler_sim[ii] - bler_interp) < 1e-2, (
                        f"BLER mismatch at CBS={cbs}, SNR={snr}"
                    )

    def test_get_bler(self, device):
        """Test get_bler method."""
        cbs_delta = 99
        assert (cbs_delta % 2) != 0, "cbs_delta must be odd"

        phy_abs = PHYAbstraction(
            cbs_interp_min_max_delta=(24, 8448, cbs_delta),
            precision="double",
            device=device,
        )

        # Check that it does not throw errors with non-tensor inputs
        bler_float = phy_abs.get_bler(
            mcs_index=10,
            mcs_table_index=1,
            mcs_category=0,  # PUSCH
            cb_size=500,
            snr_eff=10,
        )

        # Test with tensor inputs
        shape = [20, 20]
        generator = config.torch_rng(device)

        snr_db = torch.rand(shape, device=device, generator=generator) * 20
        snr = torch.pow(torch.tensor(10.0, device=device), snr_db / 10)
        table_index = random_tensor_from_values([1, 2], shape)
        mcs = random_tensor_from_values(list(range(10, 20)), shape)
        cbs = torch.randint(24, 8000, shape, dtype=torch.int32, device=device, generator=generator)
        category = random_tensor_from_values([0, 1], shape)

        bler_pt = phy_abs.get_bler(mcs, table_index, category, cbs, snr)
        bler_pt = bler_pt.cpu().numpy()

        for i1 in range(shape[0]):
            for i2 in range(shape[1]):
                table_idx = table_index[i1, i2].item() - 1
                category_idx = category[i1, i2].item()
                mcs_idx = mcs[i1, i2].item()

                cbs_ = cbs[i1, i2].item()
                cbs_idx = np.argmin(abs(phy_abs._cbs_interp - cbs_))

                snr_db_ = snr_db[i1, i2].item()
                snr_db_idx = np.argmin(abs(phy_abs._snr_dbs_interp - snr_db_))

                bler_numpy = phy_abs.bler_table_interp[
                    category_idx, table_idx, mcs_idx, cbs_idx, snr_db_idx
                ].cpu().numpy()

                assert abs(bler_pt[i1, i2] - bler_numpy) < 1e-6, (
                    f"BLER mismatch at ({i1}, {i2})"
                )

    def test_call(self, device):
        """Ensure that 'call' method of PHYAbstraction does not throw any error."""
        batch_size = 2
        num_ut = 8
        num_ofdm_symbols = 4
        num_subcarriers = 12
        num_streams_per_ut = 2

        # Generate SINR
        sinr = torch.rand(
            batch_size,
            num_ofdm_symbols,
            num_subcarriers,
            num_ut,
            num_streams_per_ut,
            device=device,
        ) * 100

        # MCS
        mcs_index = torch.randint(
            3, 10, (batch_size, num_ut), dtype=torch.int32, device=device
        )

        mcs_table_index = 1
        mcs_category = 0  # PUSCH

        # Instantiate PHYAbstraction object
        phy_abs = PHYAbstraction(precision="double", device=device)

        num_decoded_bits, harq_feedback, sinr_eff, tbler, bler = phy_abs(
            mcs_index,
            sinr=sinr,
            mcs_table_index=mcs_table_index,
            mcs_category=mcs_category,
        )

        # Basic shape checks
        assert num_decoded_bits.shape == (batch_size, num_ut)
        assert harq_feedback.shape == (batch_size, num_ut)
        assert sinr_eff.shape == (batch_size, num_ut)
        assert tbler.shape == (batch_size, num_ut)
        assert bler.shape == (batch_size, num_ut)

        # If HARQ=1 (ACK) then number of successfully decoded bits must be positive
        ack_mask = harq_feedback == 1
        if ack_mask.any():
            assert (num_decoded_bits[ack_mask] > 0).all(), (
                "ACK with zero decoded bits"
            )

    @pytest.mark.parametrize("mode", ["default", "reduce-overhead"])
    def test_compiled(self, device, mode):
        """Test that PHYAbstraction works with torch.compile."""
        if device == "cpu" and mode == "reduce-overhead":
            pytest.skip("reduce-overhead mode not well supported on CPU")

        batch_size = 2
        num_ut = 4
        num_ofdm_symbols = 2
        num_subcarriers = 12
        num_streams_per_ut = 2

        sinr = torch.rand(
            batch_size,
            num_ofdm_symbols,
            num_subcarriers,
            num_ut,
            num_streams_per_ut,
            device=device,
        ) * 100

        mcs_index = torch.randint(
            3, 10, (batch_size, num_ut), dtype=torch.int32, device=device
        )

        phy_abs = PHYAbstraction(device=device)

        compiled_call = torch.compile(phy_abs.call, mode=mode)

        # Run compiled version
        num_decoded_bits, harq_feedback, sinr_eff, tbler, bler = compiled_call(
            mcs_index, sinr=sinr, mcs_table_index=1, mcs_category=0
        )

        # Basic shape checks
        assert num_decoded_bits.shape == (batch_size, num_ut)

    def test_compiled_dynamic_num_ut(self, device):
        """Compiled PHYAbstraction must tolerate a changing num_ut (SymInt sizes).

        Regression: torch.rand(tbler.shape) failed fake-tensor propagation when
        Dynamo introduced symbolic shapes after graph breaks in gather/MCS paths.
        """
        phy_abs = PHYAbstraction(device=device)

        @torch.compile
        def step(mcs, sinr):
            return phy_abs(mcs, sinr=sinr, mcs_table_index=1, mcs_category=0)

        for num_ut in (4, 3, 5):
            sinr = torch.rand(1, 2, 12, num_ut, 1, device=device) * 100
            mcs = torch.randint(3, 10, (1, num_ut), dtype=torch.int32, device=device)
            num_decoded_bits, harq_feedback, *_ = step(mcs, sinr)
            assert num_decoded_bits.shape == (1, num_ut)
            assert harq_feedback.shape == (1, num_ut)

    def test_num_decoded_bits_matches_tb_size(self, device, monkeypatch):
        """Decoded bits on ACK equal transport-block information bits."""
        from sionna.phy.nr.utils import MCSDecoderNR, calculate_tb_size

        phy_abs = PHYAbstraction(device=device, precision="double")
        mcs_decoder = MCSDecoderNR(device=device)

        # First user has one CB; second user has many CBs.
        mcs_index = torch.tensor([3, 27], dtype=torch.int32, device=device)
        num_allocated_re = torch.tensor(
            [[256, 50000]], dtype=torch.int32, device=device
        )
        sinr_eff = torch.full(
            (1, 2),
            1e6,
            dtype=torch.float64,
            device=device,
        )
        # This test covers decoded-bit accounting, not stochastic HARQ. Force
        # successful decoding independently of the finite BLER-table SNR range.
        monkeypatch.setattr(
            phy_abs,
            "get_bler",
            lambda *args, **kwargs: torch.zeros_like(sinr_eff),
        )

        num_decoded_bits, harq, *_ = phy_abs(
            mcs_index.unsqueeze(0),
            sinr_eff=sinr_eff,
            num_allocated_re=num_allocated_re,
            mcs_table_index=1,
            mcs_category=0,
        )
        assert (harq == 1).all()

        mod_order, coderate = mcs_decoder(
            mcs_index, torch.ones_like(mcs_index), torch.zeros_like(mcs_index)
        )
        num_coded_bits = mod_order * num_allocated_re[0]
        tb_size, cb_size, num_cb, *_ = calculate_tb_size(
            mod_order,
            coderate,
            num_coded_bits=num_coded_bits,
            return_cw_length=False,
            device=device,
        )
        assert num_cb[0] == 1
        assert num_cb[1] > 1
        assert torch.equal(num_decoded_bits[0], tb_size)
        assert (num_decoded_bits[0] < num_cb * cb_size).all()

    def test_missing_bler_row_is_inf_and_nack(self, device):
        """Unavailable BLER rows stay inf and produce a deterministic NACK."""
        phy_abs = PHYAbstraction(device=device)
        num_decoded_bits, harq, _, tbler, bler = phy_abs(
            torch.tensor([0], dtype=torch.int32, device=device),
            sinr_eff=torch.tensor([1.0], device=device),
            # Produces four code blocks, catching parity-dependent inf
            # propagation through 1 - (1 - BLER) ** num_cb.
            num_allocated_re=torch.tensor(
                [50000], dtype=torch.int32, device=device
            ),
            mcs_table_index=1,
            mcs_category=0,
        )
        assert torch.isinf(bler).item()
        assert torch.isinf(tbler).item()
        assert harq.item() == 0
        assert num_decoded_bits.item() == 0

    def test_cbs_above_simulated_range_matches_boundary(self, device):
        """CBS above the largest simulated point uses the boundary curve."""
        phy_abs = PHYAbstraction(device=device)
        bler_2000 = phy_abs.get_bler(10, 1, 0, 2000, 2.0)
        bler_8448 = phy_abs.get_bler(10, 1, 0, 8448, 2.0)
        assert torch.isclose(bler_2000, bler_8448).item()
