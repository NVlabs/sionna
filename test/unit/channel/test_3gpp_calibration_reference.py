#
# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION &
# AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Sanity checks for bundled TR 38.901 calibration reference curves."""

import json
import math
from pathlib import Path

import pytest


REFERENCE_JSON = (
    Path(__file__).parent / "tr38901_calibration_results" / "reference_curves.json"
)
MANIFEST_JSON = REFERENCE_JSON.with_name("manifest.json")


def _assert_finite(values, allow_none=False):
    assert all(
        (allow_none and value is None)
        or (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
        )
        for value in values
    )


def _assert_monotonic(values, allow_none=False):
    _assert_finite(values, allow_none=allow_none)
    values = [value for value in values if value is not None]
    assert all(a <= b for a, b in zip(values, values[1:]))


def test_finite_validation_rejects_malformed_values():
    """Allow declared missing samples, but reject malformed numeric data."""
    _assert_finite([1.0, None], allow_none=True)
    with pytest.raises(AssertionError):
        _assert_finite([1.0, "invalid"], allow_none=True)
    with pytest.raises(AssertionError):
        _assert_finite([1.0, math.nan])


def test_reference_cdf_curves_are_valid():
    """Check CDF references do not contain duplicated extraction rows."""
    data = json.loads(REFERENCE_JSON.read_text())
    for phase in (
        "blockage_model_a",
        "spatial_consistency_metric1_2",
        "spatial_consistency_config2_metric1_2",
    ):
        for entry in data[phase].values():
            for curve in entry["metrics"].values():
                assert len(curve["cdf"]) == 101
                assert len(curve["x"]) == 101
                assert len(curve["percentiles"]) == 101
                assert curve["cdf"][0] == 0.0
                assert curve["cdf"][-1] == 100.0
                _assert_monotonic(curve["cdf"])
                _assert_monotonic(curve["x"])
                for individual_curve in curve["individual_curves"].values():
                    assert len(individual_curve["cdf"]) == 101
                    assert len(individual_curve["x"]) == 101
                    assert individual_curve["cdf"][0] == 0.0
                    assert individual_curve["cdf"][-1] == 100.0
                    _assert_monotonic(individual_curve["cdf"])
                    _assert_monotonic(individual_curve["x"], allow_none=True)


def test_reference_provenance_matches_repository_evidence():
    """Require explicit provenance gaps instead of fabricated source details."""

    data = json.loads(REFERENCE_JSON.read_text())
    metadata = data["metadata"]
    provenance = metadata["provenance"]
    phases = set(data) - {"metadata"}

    assert set(metadata["reference_standard_by_phase"]) == phases
    assert set(metadata["source_files"]) <= phases
    assert provenance["source_workbook_filenames_by_phase"] == metadata["source_files"]
    assert provenance["source_contribution_ids"] == metadata["source_contributions"]
    assert provenance["source_workbook_binaries_bundled"] is False
    assert provenance["retrieval_urls_bundled"] is False
    assert provenance["source_checksums_bundled"] is False
    assert provenance["extraction_tooling_bundled"] is False


def test_historical_path_gain_references_are_truthfully_labelled():
    """Negative workbook curves must never masquerade as positive loss."""

    data = json.loads(REFERENCE_JSON.read_text())
    for phase, phase_data in data.items():
        if phase == "metadata":
            continue
        for entry in phase_data.values():
            metrics = entry.get("metrics", {})
            assert "coupling_loss" not in metrics
            curve = metrics.get("historical_path_gain")
            if curve is None:
                continue
            assert curve["label"] == "Historical Path Gain [dB]"
            assert max(curve["x"]) < 0.0


def test_spatial_consistency_config2_reference_is_static_subset():
    """Check the bundled Config2 reference contains only static CDF metrics."""
    data = json.loads(REFERENCE_JSON.read_text())
    entry = data["spatial_consistency_config2_metric1_2"]["UMi-30GHz"]
    assert entry["sheet_name"] == "Config2-ProcA"
    assert set(entry["metrics"]) == {
        "historical_path_gain",
        "historical_wideband_sir",
    }
    sir = entry["metrics"]["historical_wideband_sir"]
    assert sir["label"] == "Historical Wideband SIR [dB]"
    for curve in entry["metrics"].values():
        assert curve["n_companies"] == 5


def test_incompatible_inf_reference_is_not_bundled():
    """Do not attach the 15/25 m contribution curves to normative 10 m InF."""

    data = json.loads(REFERENCE_JSON.read_text())

    assert not any(phase.startswith("inf") for phase in data if phase != "metadata")
    assert not any(
        phase.startswith("inf") for phase in data["metadata"]["source_files"]
    )


def test_committed_manifest_describes_complete_schema_2_bundle():
    """Keep the committed generated bundle complete and current."""

    manifest = json.loads(MANIFEST_JSON.read_text())

    assert manifest["spec_version"] == "16.1"
    assert manifest["calibration_schema_version"] == 2
    assert "bundle_status" not in manifest
    assert "required_calibration_schema_version" not in manifest
    assert len(manifest["runs"]) == 46
    assert all(
        run["spec_version"] == "16.1"
        and run["calibration_schema_version"] == 2
        for run in manifest["runs"].values()
    )


def test_reference_line_curves_are_valid():
    """Check spatial consistency line references use the intended distance grid."""
    data = json.loads(REFERENCE_JSON.read_text())
    for entry in data["spatial_consistency_metric3_6"].values():
        for curve in entry["metrics"].values():
            assert len(curve["x"]) == 131
            assert len(curve["y"]) == 131
            assert curve["x"][0] == 0.0
            assert curve["x"][-1] == 130.0
            _assert_monotonic(curve["x"])
            _assert_finite(curve["y"])
            for individual_curve in curve["individual_curves"].values():
                assert len(individual_curve["x"]) == 131
                assert len(individual_curve["y"]) == 131
                assert individual_curve["x"][0] == 0.0
                assert individual_curve["x"][-1] == 130.0
                _assert_monotonic(individual_curve["x"])
                _assert_finite(individual_curve["y"], allow_none=True)
