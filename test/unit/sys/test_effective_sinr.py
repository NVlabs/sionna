#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Unit tests for sionna.sys.effective_sinr"""

import json

import numpy as np
import pytest
import torch

from sionna.phy import config
from sionna.sys import EESM


def get_sinr_eff_numpy(
    sinr_sel: np.ndarray,
    beta_sel: float,
    sinr_eff_min_db: float,
    sinr_eff_max_db: float,
) -> float:
    """Numpy version of effective SINR computation.

    :param sinr_sel: SINR values
    :param beta_sel: Beta parameter for EESM
    :param sinr_eff_min_db: Minimum effective SINR [dB]
    :param sinr_eff_max_db: Maximum effective SINR [dB]
    :return: Effective SINR
    """
    n_used_res = (sinr_sel > 0).sum()
    if n_used_res == 0:
        return 0
    sinr_exp = np.exp(-sinr_sel / beta_sel)
    sinr_exp = sinr_exp * (sinr_sel > 0)
    sinr_eff_numpy = -beta_sel * np.log(np.sum(sinr_exp) / n_used_res)

    sinr_eff_min = np.power(10, sinr_eff_min_db / 10)
    sinr_eff_max = np.power(10, sinr_eff_max_db / 10)
    if sinr_eff_numpy > sinr_eff_max:
        sinr_eff_numpy = sinr_eff_max
    if sinr_eff_numpy < sinr_eff_min:
        sinr_eff_numpy = sinr_eff_min
    return sinr_eff_numpy


class TestEffectiveSINR:
    """Tests for the EESM effective SINR computation."""

    def test_unused_resources_return_zero(self, device):
        """Unused users and streams remain zero below the clamp floor."""
        eff_sinr_obj = EESM(device=device)
        sinr = torch.zeros(1, 2, 3, 2, 2, device=device)
        mcs = torch.zeros(1, 2, dtype=torch.int32, device=device)

        for per_stream in (False, True):
            sinr_eff = eff_sinr_obj(sinr, mcs, per_stream=per_stream)
            torch.testing.assert_close(sinr_eff, torch.zeros_like(sinr_eff))

        sinr[..., 0, 0] = 1.0
        sinr_eff = eff_sinr_obj(sinr, mcs)
        assert sinr_eff[0, 0] > 0
        assert sinr_eff[0, 1] == 0

        sinr_eff_per_stream = eff_sinr_obj(sinr, mcs, per_stream=True)
        assert sinr_eff_per_stream[0, 0, 0] > 0
        assert sinr_eff_per_stream[0, 0, 1] == 0
        torch.testing.assert_close(
            sinr_eff_per_stream[0, 1],
            torch.zeros_like(sinr_eff_per_stream[0, 1]),
        )

    def test_sinr_eff_vs_numpy(self, device):
        """Check that the effective SINR computation matches its Numpy counterpart."""
        batch_size = 30
        num_ofdm_symbols = 2
        num_subcarriers = 10
        num_ut = 50
        num_streams_per_ut = 3
        sinr_eff_min_db = -40
        sinr_eff_max_db = 40
        precision = "double"
        dtype = torch.float64

        eff_sinr_obj = EESM(
            sinr_eff_min_db=sinr_eff_min_db,
            sinr_eff_max_db=sinr_eff_max_db,
            precision=precision,
            device=device,
        )

        # Generate SINR randomly
        generator = config.torch_rng(device)
        sinr = torch.rand(
            batch_size,
            num_ofdm_symbols,
            num_subcarriers,
            num_ut,
            num_streams_per_ut,
            dtype=dtype,
            device=device,
            generator=generator,
        ) * 10

        # Mask SINR on some streams
        mask = torch.randint(
            0,
            2,
            (batch_size, num_ofdm_symbols, num_subcarriers, num_ut, num_streams_per_ut),
            dtype=torch.int32,
            device=device,
            generator=generator,
        )
        sinr = sinr * mask.to(dtype)

        # Generate MCS per user
        mcs = torch.randint(
            0, 27, (batch_size, num_ut), dtype=torch.int32, device=device, generator=generator
        )

        # Table index
        table_index = torch.randint(
            1, 3, (batch_size, num_ut), dtype=torch.int32, device=device, generator=generator
        )

        for per_stream in [True, False]:
            # PyTorch version
            sinr_eff = eff_sinr_obj(
                sinr, mcs, mcs_table_index=table_index, per_stream=per_stream
            ).cpu().numpy()

            for batch in range(batch_size):
                for ut in range(num_ut):
                    mcs_sel = mcs[batch, ut].item()
                    table_idx_sel = table_index[batch, ut].item()
                    beta_sel = eff_sinr_obj.beta_tensor.cpu().numpy()[
                        table_idx_sel - 1, mcs_sel
                    ]

                    if per_stream:
                        for stream in range(num_streams_per_ut):
                            sinr_sel = sinr[batch, :, :, ut, stream].cpu().numpy()
                            # Numpy version
                            sinr_eff_numpy = get_sinr_eff_numpy(
                                sinr_sel, beta_sel, sinr_eff_min_db, sinr_eff_max_db
                            )
                            assert abs(sinr_eff[batch, ut, stream] - sinr_eff_numpy) < 1e-5, (
                                f"Mismatch at batch={batch}, ut={ut}, stream={stream}"
                            )
                    else:
                        sinr_sel = sinr[batch, :, :, ut, :].cpu().numpy()
                        # Numpy version
                        sinr_eff_numpy = get_sinr_eff_numpy(
                            sinr_sel, beta_sel, sinr_eff_min_db, sinr_eff_max_db
                        )
                        assert abs(sinr_eff[batch, ut] - sinr_eff_numpy) < 1e-5, (
                            f"Mismatch at batch={batch}, ut={ut}"
                        )

    @pytest.mark.parametrize("mode", ["default", "reduce-overhead"])
    def test_compiled(self, device, mode):
        """Test that EESM works with torch.compile."""
        if device == "cpu" and mode == "reduce-overhead":
            pytest.skip("reduce-overhead mode not well supported on CPU")

        batch_size = 4
        num_ofdm_symbols = 2
        num_subcarriers = 10
        num_ut = 8
        num_streams_per_ut = 2
        precision = "single"
        dtype = torch.float32

        eff_sinr_obj = EESM(precision=precision, device=device)

        compiled_call = torch.compile(eff_sinr_obj.call, mode=mode)

        # Generate inputs
        sinr = torch.rand(
            batch_size,
            num_ofdm_symbols,
            num_subcarriers,
            num_ut,
            num_streams_per_ut,
            dtype=dtype,
            device=device,
        ) * 10
        mcs = torch.randint(0, 27, (batch_size, num_ut), dtype=torch.int32, device=device)

        # Run compiled version
        sinr_eff = compiled_call(sinr, mcs, mcs_table_index=1, per_stream=False)

        # Basic shape check
        assert sinr_eff.shape == (batch_size, num_ut)

    @pytest.mark.parametrize("table_index", [0, 3, 4])
    def test_unsupported_mcs_table_index_raises(self, device, table_index):
        """Default EESM betas cover tables 1-2 only."""
        eesm = EESM(device=device)
        sinr = torch.ones(1, 2, 4, 1, 1, device=device)
        mcs = torch.zeros(1, 1, dtype=torch.int32, device=device)
        with pytest.raises(ValueError, match="mcs_table_index"):
            eesm(sinr, mcs, mcs_table_index=table_index)

    @pytest.mark.parametrize(
        "betas",
        [[], [1.0, 0.0], [1.0, -1.0], [1.0, float("inf")], [1.0, "bad"]],
    )
    def test_invalid_beta_table_raises(self, device, tmp_path, betas):
        """Beta rows must be non-empty and finite positive."""
        path = tmp_path / "betas.json"
        path.write_text(json.dumps({"index": {"1": betas}}))
        with pytest.raises(ValueError, match="non-empty|finite positive"):
            EESM(load_beta_table_from=str(path), device=device)

    def test_default_beta_table_contract(self, device):
        """Default beta table has expected indices and ignores category."""
        eesm = EESM(device=device)
        assert set(eesm.beta_table["index"]) == {1, 2}
        assert {idx: len(row) for idx, row in eesm.beta_table["index"].items()} == {
            1: 29,
            2: 28,
        }
        assert eesm.validate_beta_table() is True

        sinr = torch.ones(1, 2, 4, 1, 1, device=device)
        mcs = torch.zeros(1, 1, dtype=torch.int32, device=device)
        category_zero = eesm(sinr, mcs, mcs_table_index=1, mcs_category=0)
        category_one = eesm(sinr, mcs, mcs_table_index=1, mcs_category=1)
        assert torch.equal(category_zero, category_one)

    def test_noncontiguous_custom_beta_indices(self, device, tmp_path):
        """Only explicitly loaded table indices are accepted."""
        path = tmp_path / "betas.json"
        path.write_text(
            json.dumps({"index": {"1": [1.0], "3": [2.0]}})
        )
        eesm = EESM(load_beta_table_from=str(path), device=device)
        sinr = torch.ones(1, 2, 4, 1, 1, device=device)
        mcs = torch.zeros(1, 1, dtype=torch.int32, device=device)

        assert eesm(sinr, mcs, mcs_table_index=3).shape == (1, 1)
        with pytest.raises(ValueError, match="mcs_table_index"):
            eesm(sinr, mcs, mcs_table_index=2)
