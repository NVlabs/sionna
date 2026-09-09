#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Tests for TR 38.901 and TS 38.101-4 model resources."""

import zipfile

import torch

from sionna.phy.channel.tr38901 import (
    CDL,
    PanelArray,
    TDL,
    UMiScenario,
    models,
)
from sionna.phy.ofdm import tdl_freq_cov_mat, tdl_time_cov_mat


_FIXED_TDL_FILES = {
    "TDL-A10.json",
    "TDL-A30.json",
    "TDL-B100.json",
    "TDL-C60.json",
    "TDL-C300.json",
    "TDL-D10.json",
    "TDL-D30.json",
}


def test_fixed_tdl_resources_are_canonical():
    """Fixed TS profiles exist once and outside both TR version directories."""
    for filename in _FIXED_TDL_FILES:
        assert models.fixed_tdl_parameter_file(filename).is_file()
        assert not models.parameter_file(filename, "16.1").is_file()
        assert not models.parameter_file(filename, "19.2").is_file()


def test_model_consumers_support_zip_backed_resources(tmp_path, monkeypatch):
    """TDL, CDL, and covariance helpers load archive-backed Traversables."""
    resources = {
        "tr/TDL-A.json": models.parameter_file("TDL-A.json").read_bytes(),
        "tr/CDL-A.json": models.parameter_file("CDL-A.json").read_bytes(),
        "tr/UMi_LoS.json": models.parameter_file("UMi_LoS.json").read_bytes(),
        "tr/UMi_NLoS.json": models.parameter_file("UMi_NLoS.json").read_bytes(),
        "tr/UMi_O2I.json": models.parameter_file("UMi_O2I.json").read_bytes(),
        "fixed/TDL-A30.json": models.fixed_tdl_parameter_file(
            "TDL-A30.json"
        ).read_bytes(),
    }
    archive_path = tmp_path / "models.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        for filename, data in resources.items():
            archive.writestr(filename, data)

    tr_requests = []
    fixed_requests = []
    with zipfile.ZipFile(archive_path) as archive:
        root = zipfile.Path(archive)

        def parameter_file(filename, spec_version="19.2"):
            tr_requests.append((filename, spec_version))
            return root.joinpath("tr", filename)

        def fixed_tdl_parameter_file(filename):
            fixed_requests.append(filename)
            return root.joinpath("fixed", filename)

        monkeypatch.setattr(models, "parameter_file", parameter_file)
        monkeypatch.setattr(
            models, "fixed_tdl_parameter_file", fixed_tdl_parameter_file
        )

        scalable_tdl = TDL("A", 100e-9, 3.5e9, device="cpu")
        fixed_tdl_v16 = TDL(
            "A30", 30e-9, 3.5e9, spec_version="16.1", device="cpu"
        )
        fixed_tdl_v19 = TDL(
            "A30", 30e-9, 3.5e9, spec_version="19.2", device="cpu"
        )

        array = PanelArray(
            num_rows_per_panel=1,
            num_cols_per_panel=1,
            polarization="single",
            polarization_type="V",
            antenna_pattern="omni",
            carrier_frequency=3.5e9,
            device="cpu",
        )
        cdl = CDL("A", 100e-9, 3.5e9, array, array, device="cpu")
        scenario = UMiScenario(
            3.5e9,
            "low",
            array,
            array,
            "downlink",
            device="cpu",
        )
        freq_cov = tdl_freq_cov_mat("A", 30e3, 4, 100e-9)
        time_cov = tdl_time_cov_mat("A", 3.0, 3.5e9, 1e-3, 4)

    torch.testing.assert_close(fixed_tdl_v16.delays, fixed_tdl_v19.delays)
    torch.testing.assert_close(
        fixed_tdl_v16.mean_powers, fixed_tdl_v19.mean_powers
    )
    assert scalable_tdl.num_clusters == 23
    assert cdl._num_clusters == 23
    assert scenario.num_clusters_los == 12
    assert freq_cov.shape == (4, 4)
    assert time_cov.shape == (4, 4)
    assert fixed_requests == ["TDL-A30.json", "TDL-A30.json"]
    assert all(filename != "TDL-A30.json" for filename, _ in tr_requests)
