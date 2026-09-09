#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Regressions against generated calibration snapshots, not independent truth."""

import gc
import json
from pathlib import Path

import numpy as np
import pytest
import torch

import tr38901_calibration as calibration


RESULTS_DIR = Path(__file__).parent / "tr38901_calibration_results" / "cdfs"
REGRESSION_SEED = 20260713
SPEC_VERSION = "16.1"
CDF_PERCENTILES = np.asarray([10.0, 25.0, 50.0, 75.0, 90.0])

CDF_CASES = [
    pytest.param("uma_config1", "UMa_30GHz_config1.json", id="uma-config1"),
    pytest.param("inh_config1", "InH_30GHz_config1.json", id="inh-config1"),
    pytest.param("umi_config2", "UMi_30GHz_config2.json", id="umi-config2"),
    pytest.param("inf_sl", "InF-SL_3.5GHz_inf.json", id="inf-sl"),
    pytest.param(
        "spatial_metric1_2",
        "UMi_30GHz_spatial_consistency_metric1_2.json",
        id="spatial-consistency-cdf",
    ),
    pytest.param(
        "blockage_model_a",
        "UMi_30GHz_blockage_model_a.json",
        id="blockage-model-a",
    ),
]


@pytest.fixture(autouse=True)
def _release_calibration_memory():
    """Release large calibration tensors after every regression case."""

    yield
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _load_snapshot(filename: str, require_current_schema: bool = True) -> dict:
    data = json.loads((RESULTS_DIR / filename).read_text())
    assert data["metadata"]["spec_version"] == SPEC_VERSION
    if (
        require_current_schema
        and data["metadata"].get("calibration_schema_version")
        != calibration.CALIBRATION_SCHEMA_VERSION
    ):
        pytest.skip(
            f"Generated snapshot {filename} requires calibration-bundle regeneration"
        )
    return data


def _run_phase2(scenario: str, device: str, config_id: int = 1):
    cfg = calibration._get_phase2_config(scenario, 30.0, config_id)
    return calibration._run_phase2_batches(
        cfg,
        num_batches=2,
        batch_size=1,
        num_ut_per_sector=1,
        seed=REGRESSION_SEED,
        precision="single",
        devices=[device],
        spec_version=SPEC_VERSION,
    )


def _run_cdf_case(case: str, device: str):
    if case == "uma_config1":
        return _run_phase2("UMa", device)
    if case == "inh_config1":
        return _run_phase2("InH", device)
    if case == "umi_config2":
        return _run_phase2("UMi", device, config_id=2)
    if case == "inf_sl":
        cfg = calibration._get_inf_config("InF-SL", 3.5)
        return calibration._run_inf_batches(
            cfg,
            num_batches=1,
            batch_size=1,
            num_ut_per_bs=30,
            ut_chunk_size=64,
            seed=REGRESSION_SEED,
            precision="single",
            devices=[device],
            spec_version=SPEC_VERSION,
        )
    if case == "spatial_metric1_2":
        return calibration._run_spatial_metric1_2(
            num_ut_per_sector=30,
            num_drops=1,
            ut_chunk_size=128,
            carrier_frequency=30e9,
            precision="single",
            devices=[device],
            seed=REGRESSION_SEED,
            spec_version=SPEC_VERSION,
        )
    if case == "blockage_model_a":
        cfg = calibration._get_phase2_config("UMi", 30.0, 1)
        return calibration._run_phase2_batches(
            cfg,
            num_batches=1,
            batch_size=1,
            num_ut_per_sector=2,
            seed=REGRESSION_SEED,
            precision="single",
            devices=[device],
            spec_version=SPEC_VERSION,
            phase_name="blockage_model_a",
            enable_blockage=True,
            blockage_self_blocking="landscape",
        )
    raise ValueError(f"Unknown calibration regression case: {case}")


def _assert_cdf_matches_snapshot(result, snapshot: dict) -> None:
    """Compare fresh samples with a previously generated empirical snapshot."""

    assert result.scenario == snapshot["metadata"]["scenario"]
    assert result.fc_ghz == snapshot["metadata"]["frequency_ghz"]
    assert result.phase == snapshot["metadata"]["phase"]
    assert set(result.metrics) == set(snapshot["metrics"])

    failures = []
    expected_cdf = CDF_PERCENTILES / 100.0
    for metric, values in result.metrics.items():
        samples = np.asarray(values, dtype=np.float64)
        assert np.all(np.isfinite(samples)), f"{metric} produced non-finite samples"
        assert samples.size >= 50, f"{metric} produced too few samples"

        curve = snapshot["metrics"][metric]
        snapshot_percentiles = np.asarray(curve["percentiles"], dtype=np.float64)
        snapshot_x = np.asarray(curve["x"], dtype=np.float64)
        assert np.all(np.isfinite(snapshot_percentiles)), (
            f"{metric} snapshot percentiles contain non-finite values"
        )
        assert np.all(np.isfinite(snapshot_x)), (
            f"{metric} snapshot curve contains non-finite values"
        )
        thresholds = np.interp(
            CDF_PERCENTILES,
            snapshot_percentiles,
            snapshot_x,
        )
        observed_cdf = np.asarray(
            [np.mean(samples <= threshold) for threshold in thresholds]
        )
        deviations = np.abs(observed_cdf - expected_cdf)

        # This is intentionally a broad behavioral guard. The finite-sample
        # term permits independent random streams on different platforms.
        tolerance = max(0.15, 2.0 / np.sqrt(samples.size))
        worst = int(np.argmax(deviations))
        if deviations[worst] > tolerance:
            failures.append(
                f"{metric} at P{CDF_PERCENTILES[worst]:g}: observed CDF "
                f"{observed_cdf[worst]:.3f}, expected "
                f"{expected_cdf[worst]:.3f}, deviation "
                f"{deviations[worst]:.3f} > {tolerance:.3f} "
                f"({samples.size} samples)"
            )

    assert not failures, "Calibration CDF regression:\n" + "\n".join(failures)


@pytest.mark.parametrize("case,snapshot_filename", CDF_CASES)
def test_reduced_calibration_cdfs_match_generated_snapshots(
    case, snapshot_filename, device
):
    """Detect distribution shifts from a prior generated V16.1 snapshot."""

    snapshot = _load_snapshot(snapshot_filename)
    result = _run_cdf_case(case, device)
    _assert_cdf_matches_snapshot(result, snapshot)


def test_blockage_model_b_curve_matches_generated_snapshot(device):
    """Detect material changes from the generated Model B curve snapshot."""

    result = calibration._run_blockage_model_b(
        num_realizations=3,
        seed=REGRESSION_SEED,
        precision="single",
        devices=[device],
        spec_version=SPEC_VERSION,
    )
    snapshot = _load_snapshot(
        "UMi_30GHz_blockage_model_b.json", require_current_schema=False
    )

    curve = snapshot["metrics"]["relative_rsrp_db"]
    assert np.all(np.isfinite(result.x_values))
    assert np.all(np.isfinite(result.metrics["relative_rsrp_db"]))
    assert np.all(np.isfinite(curve["x"]))
    assert np.all(np.isfinite(curve["y"]))
    assert np.allclose(result.x_values, curve["x"])
    delta = result.metrics["relative_rsrp_db"] - np.asarray(curve["y"])
    max_error = float(np.max(np.abs(delta)))
    rmse = float(np.sqrt(np.mean(delta**2)))

    assert max_error <= 3.0, f"Blockage Model B max curve error: {max_error:.3f} dB"
    assert rmse <= 1.25, f"Blockage Model B curve RMSE: {rmse:.3f} dB"


def test_spatial_consistency_curves_match_generated_snapshot(device):
    """Detect material changes from generated spatial-correlation snapshots."""

    distances = np.asarray([0.0, 5.0, 10.0, 30.0, 60.0, 100.0])
    snapshot = _load_snapshot("UMi_30GHz_spatial_consistency_metric3_6.json")
    assert snapshot["metrics"]["los_state_corr"]["y"][0] > 0.5, (
        "The committed LOS/NLOS-state snapshot has lost short-distance "
        "spatial correlation"
    )
    result = calibration._run_spatial_metric3_6(
        distances_m=distances,
        num_ut_per_sector=100,
        num_drops=8,
        carrier_frequency=30e9,
        precision="single",
        devices=[device],
        seed=REGRESSION_SEED,
        spec_version=SPEC_VERSION,
    )
    failures = []
    for metric, values in result.metrics.items():
        curve = snapshot["metrics"][metric]
        assert np.all(np.isfinite(values)), f"{metric} produced non-finite values"
        assert np.all(np.isfinite(curve["x"])), (
            f"{metric} snapshot x values contain non-finite values"
        )
        assert np.all(np.isfinite(curve["y"])), (
            f"{metric} snapshot y values contain non-finite values"
        )
        expected = np.interp(distances, curve["x"], curve["y"])
        delta = np.asarray(values) - expected
        max_error = float(np.max(np.abs(delta)))
        rmse = float(np.sqrt(np.mean(delta**2)))
        if max_error > 0.4 or rmse > 0.2:
            failures.append(f"{metric}: max error {max_error:.3f}, RMSE {rmse:.3f}")

    assert not failures, "Spatial-consistency regression:\n" + "\n".join(failures)
