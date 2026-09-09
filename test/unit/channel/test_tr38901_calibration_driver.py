#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Focused tests for calibration-driver workload provenance."""

import json
import warnings
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import tr38901_calibration as calibration


def _fake_result(seed, *, phase="inf", metadata=None):
    samples = np.asarray([seed, seed + 0.5], dtype=np.float64)
    return calibration.CalibrationResult(
        scenario="InF-SL",
        fc_ghz=3.5,
        phase=phase,
        metrics={"coupling_loss": samples},
        serving_bs=np.zeros(2, dtype=np.int64),
        in_state=np.ones(2, dtype=bool),
        o2i_is_high_loss=np.zeros(2, dtype=bool),
        metadata=metadata,
    )


def test_cdf_writer_rejects_non_finite_samples(tmp_path):
    """Do not silently discard invalid calibration samples."""
    result = _fake_result(1)
    result.metrics["coupling_loss"][1] = np.nan

    with pytest.raises(ValueError, match="non-finite"):
        calibration._write_cdf_json(tmp_path, result, [0.0, 100.0])


def test_line_writer_rejects_non_finite_values(tmp_path):
    """Do not serialize non-standard JSON NaN/Infinity values."""
    result = calibration.LineCalibrationResult(
        scenario="UMi",
        fc_ghz=30.0,
        phase="blockage_model_b",
        x_label="distance",
        x_values=np.asarray([0.0, 1.0]),
        metrics={"relative_rsrp_db": np.asarray([0.0, np.inf])},
        metadata={},
    )

    with pytest.raises(ValueError, match="non-finite"):
        calibration._write_line_json(tmp_path, result)


def test_rma_topology_uses_fixed_ut_height():
    """Keep the generic calibration topology helper valid for RMa."""
    cfg = calibration.Phase1Config(
        scenario="RMa",
        fc_ghz=0.7,
        isd_m=1732.0,
        bs_height_m=35.0,
        min_bs_ut_dist_m=35.0,
        indoor_probability=0.5,
        tx_power_dbm=49.0,
        bandwidth_hz=20e6,
    )

    topology = calibration._make_topology(
        cfg,
        batch_size=1,
        num_ut_per_sector=1,
        seed=1234,
        precision="single",
        device="cpu",
    )

    assert np.all(topology.ut_loc[..., 2].cpu().numpy() == 1.5)


def test_default_devices_uses_all_visible_cuda_devices(monkeypatch):
    """The calibration default must use every available CUDA device."""

    monkeypatch.setattr(calibration.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(calibration.torch.cuda, "device_count", lambda: 3)

    assert calibration._default_devices() == ["cuda:0", "cuda:1", "cuda:2"]


def test_serving_link_values_use_positive_coupling_loss():
    """Path gain and coupling loss must be distinct, opposite-sign metrics."""

    path_gain = torch.tensor([[-90.0, -80.0, -100.0]])
    all_loss, serving, serving_loss, historical_gain = calibration._serving_link_values(
        path_gain
    )

    assert torch.equal(all_loss, torch.tensor([[90.0, 80.0, 100.0]]))
    assert serving.tolist() == [1]
    assert serving_loss.tolist() == [80.0]
    assert historical_gain.tolist() == [-80.0]


def test_config2_wideband_metric_includes_noise():
    """Config2 stores SINR and labels the interference-only result historical."""

    cfg = calibration._get_phase2_config("UMi", 30.0, 1)
    coupling_loss = torch.tensor([[80.0, 83.0]], dtype=torch.float64)
    serving = torch.tensor([0])
    metrics = calibration._spatial_wideband_metrics(
        coupling_loss,
        cfg,
        serving,
        include_historical_sir=True,
    )

    desired_mw = 10.0 ** ((cfg.tx_power_dbm - 80.0) / 10.0)
    interference_mw = 10.0 ** ((cfg.tx_power_dbm - 83.0) / 10.0)
    noise_dbm = -174.0 + 10.0 * np.log10(cfg.bandwidth_hz) + cfg.noise_figure_db
    noise_mw = 10.0 ** (noise_dbm / 10.0)
    expected_sinr = 10.0 * np.log10(desired_mw / (interference_mw + noise_mw))

    assert set(metrics) == {"wideband_sinr", "historical_wideband_sir"}
    assert metrics["wideband_sinr"].item() == pytest.approx(expected_sinr)
    assert metrics["historical_wideband_sir"].item() == pytest.approx(3.0)
    assert metrics["wideband_sinr"].item() < metrics["historical_wideband_sir"].item()


@pytest.mark.parametrize(
    "scenario,dimensions",
    [
        ("InF-SL", (120.0, 60.0, 10.0)),
        ("InF-DL", (300.0, 150.0, 10.0)),
        ("InF-SH", (300.0, 150.0, 10.0)),
        ("InF-DH", (120.0, 60.0, 10.0)),
    ],
)
def test_inf_profile_uses_normative_geometry(scenario, dimensions):
    """The InF profile always uses the normative V16.1 10 m hall."""

    cfg = calibration._get_inf_config(scenario, 3.5)

    assert cfg.phase == "inf"
    assert cfg.profile == calibration.INF_NORMATIVE_PROFILE
    assert calibration._inf_calibration_hall_dimensions(cfg) == dimensions


def test_inf_topology_metadata_is_validated_without_unpack_warning():
    """The calibration consumes checked InF geometry metadata."""

    cfg = calibration._get_inf_config("InF-SL", 3.5)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        topology = calibration._make_inf_topology(
            cfg,
            batch_size=1,
            num_ut_per_bs=1,
            precision="single",
            device="cpu",
        )

    assert topology.ut_loc.shape == (1, 18, 3)


def test_lsp_ut_slice_preserves_one_correlated_realization():
    """Chunking slices every field from the same full-topology LSP sample."""

    fields = [
        torch.arange(24, dtype=torch.float32).reshape(1, 2, 12) + offset
        for offset in range(7)
    ]
    pathloss = torch.arange(24, dtype=torch.float32).reshape(1, 2, 12) + 100.0
    lsp = calibration.LSP(*fields, pathloss=pathloss)

    sliced = calibration._slice_lsp_by_ut(lsp, 3, 7)

    for name in (
        "ds",
        "asd",
        "asa",
        "sf",
        "k_factor",
        "zsa",
        "zsd",
        "pathloss",
    ):
        assert torch.equal(getattr(sliced, name), getattr(lsp, name)[:, :, 3:7])


def test_path_gain_uses_pathloss_sampled_with_the_lsp(monkeypatch):
    """Attachment SF and path loss must come from one sampler realization."""

    topology = calibration.TopologyBundle(
        ut_loc=torch.zeros(1, 1, 3),
        bs_loc=torch.zeros(1, 1, 3),
        ut_orientations=torch.zeros(1, 1, 3),
        bs_orientations=torch.zeros(1, 1, 3),
        ut_velocities=torch.zeros(1, 1, 3),
        in_state=torch.ones(1, 1, dtype=torch.bool),
        distance_2d_in=None,
        los=None,
        bs_virtual_loc=torch.zeros(1, 1, 1, 3),
        bs_site_ids=torch.zeros(1, dtype=torch.int64),
        site_positions=torch.zeros(1, 2),
    )
    ones = torch.ones(1, 1, 1)
    lsp = calibration.LSP(
        ones,
        ones,
        ones,
        sf=10.0 * ones,
        k_factor=ones,
        zsa=ones,
        zsd=ones,
        pathloss=90.0 * ones,
    )
    scenario = SimpleNamespace(
        los=torch.ones(1, 1, 1, dtype=torch.bool),
        basic_pathloss=90.0 * ones,
    )

    def sampler():
        return lsp

    monkeypatch.setattr(
        calibration,
        "_make_scenario_and_sampler",
        lambda *args, **kwargs: (scenario, sampler),
    )

    path_gain, _high_loss, details = calibration._sample_path_gain_all_bs(
        calibration._get_phase1_config("InH", 30.0),
        topology,
        seed=1,
        precision="single",
        device="cpu",
        spec_version="16.1",
        include_antenna_gain=False,
        return_details=True,
    )

    assert path_gain.item() == pytest.approx(-80.0)
    assert details.lsp is lsp
    assert details.shadow_fading_db.item() == pytest.approx(10.0)
    assert details.o2i_loss_db.item() == pytest.approx(0.0)


def test_phase2_attachment_and_spreads_reuse_the_same_lsp(monkeypatch):
    """The LSP carrying attachment SF must also drive serving-cell rays."""

    topology = calibration.TopologyBundle(
        ut_loc=torch.zeros(1, 1, 3),
        bs_loc=torch.zeros(1, 2, 3),
        ut_orientations=torch.zeros(1, 1, 3),
        bs_orientations=torch.zeros(1, 2, 3),
        ut_velocities=torch.zeros(1, 1, 3),
        in_state=torch.zeros(1, 1, dtype=torch.bool),
        distance_2d_in=None,
        los=None,
        bs_virtual_loc=torch.zeros(1, 2, 1, 3),
        bs_site_ids=torch.arange(2),
        site_positions=torch.zeros(2, 2),
    )
    lsp_fields = [torch.ones(1, 2, 1) * value for value in range(1, 8)]
    shared_lsp = calibration.LSP(*lsp_fields)
    details = calibration.LargeScaleDetails(
        outdoor_los=torch.ones(1, 2, 1, dtype=torch.bool),
        basic_outdoor_pathloss_db=torch.zeros(1, 2, 1),
        shadow_fading_db=torch.zeros(1, 2, 1),
        o2i_loss_db=torch.zeros(1, 2, 1),
        lsp=shared_lsp,
    )
    observed = []

    class FakeChannel:
        def set_topology(self, *args, **kwargs):
            self._lsp = object()

        def __call__(self, **kwargs):
            observed.append(self._lsp)
            return torch.zeros(1), torch.zeros(1), object()

    channel = FakeChannel()
    monkeypatch.setattr(calibration, "_make_topology", lambda *args, **kwargs: topology)
    monkeypatch.setattr(calibration, "_set_phase2_ut_orientations", lambda *args: None)
    monkeypatch.setattr(
        calibration,
        "_sample_path_gain_all_bs",
        lambda *args, **kwargs: (
            torch.tensor([[[-80.0], [-90.0]]]),
            torch.zeros(1, 1, dtype=torch.bool),
            details,
        ),
    )
    monkeypatch.setattr(
        calibration,
        "_build_phase2_arrays",
        lambda *args, **kwargs: (object(), object()),
    )
    monkeypatch.setattr(
        calibration, "_make_phase2_channel", lambda *args, **kwargs: channel
    )
    monkeypatch.setattr(
        calibration, "_phase2_port_mapping", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(calibration, "_port_channels", lambda values, _weights: values)
    monkeypatch.setattr(
        calibration, "_apply_large_scale", lambda values, _path_gain: values
    )
    monkeypatch.setattr(
        calibration,
        "_port0_path_gain_db",
        lambda _values: torch.tensor([[[-80.0, -90.0]]]),
    )

    def spread_metrics(observed_channel, _rays, serving):
        observed.append(observed_channel._lsp)
        assert serving.tolist() == [0]
        return {
            key: np.zeros(1)
            for key in ("delay_spread_ns", "asd_deg", "zsd_deg", "asa_deg", "zsa_deg")
        }

    monkeypatch.setattr(calibration, "_phase2_spread_metrics", spread_metrics)

    result = calibration._run_phase2_once(
        calibration._get_phase2_config("UMi", 30.0, 1),
        batch_size=1,
        num_ut_per_sector=1,
        seed=1,
        precision="single",
        device="cpu",
        spec_version="16.1",
    )

    assert observed == [shared_lsp, shared_lsp]
    assert result.metrics["coupling_loss"].tolist() == [80.0]
    assert result.metrics["historical_path_gain"].tolist() == [-80.0]


def test_spatial_attachment_lsp_enables_spatial_correlation(monkeypatch):
    """The full LSP sampled before chunking must retain spatial conditioning."""

    topology = (
        torch.zeros(1, 1, 3),
        torch.zeros(1, 1, 3),
        torch.zeros(1, 1, 3),
        torch.zeros(1, 1, 3),
        torch.zeros(1, 1, 3),
        torch.zeros(1, 1, dtype=torch.bool),
        None,
        torch.zeros(1, 1, 1, 3),
        torch.zeros(1, dtype=torch.int64),
        torch.zeros(1, 1),
    )

    class SamplingObserved(Exception):
        pass

    def sample_path_gain(*args, **kwargs):
        assert kwargs["return_details"] is True
        assert kwargs["enable_spatial_consistency"] is True
        raise SamplingObserved

    monkeypatch.setattr(
        calibration, "_spatial_metric1_2_topology", lambda *args, **kwargs: topology
    )
    monkeypatch.setattr(calibration, "_sample_path_gain_all_bs", sample_path_gain)

    with pytest.raises(SamplingObserved):
        calibration._run_spatial_metric1_2_drop(
            num_ut_per_sector=1,
            ut_chunk_size=1,
            carrier_frequency=30e9,
            precision="single",
            device="cpu",
            seed=1,
            spec_version="16.1",
            indoor=True,
            include_historical_sir=False,
            num_rings=1,
            retain_central_site_only=True,
            retain_dropped_central_site_only=False,
        )


def test_static_request_assignment_is_reproducible():
    """Request-to-device assignment must not depend on completion order."""

    requests = [("UMi", float(index), "phase1") for index in range(7)]
    assignments = calibration._static_request_assignments(
        requests, ["cuda:0", "cuda:1", "cuda:2"]
    )

    assert [[index for index, _ in group] for group in assignments] == [
        [0, 3, 6],
        [1, 4],
        [2, 5],
    ]


def test_los_metric_reads_realized_outdoor_state_for_o2i_links():
    """LOS-state calibration preserves the outdoor state of O2I links."""

    channel = SimpleNamespace(
        _scenario=SimpleNamespace(
            los=torch.tensor([[[False, False, False]]]),
            outdoor_los=torch.tensor([[[True, False, True]]]),
        )
    )

    observed = calibration._channel_los_status(channel, torch.float64)

    assert observed.dtype == torch.float64
    assert observed.tolist() == [1.0, 0.0, 1.0]


def test_config2_phase_requests_normative_sinr_and_historical_sir(monkeypatch):
    """The Config2 entry point must select the corrected metric policy."""

    observed = {}

    def run_common(*args, **kwargs):
        observed.update(kwargs)
        return object()

    monkeypatch.setattr(calibration, "_run_spatial_metric1_2_common", run_common)

    result = calibration._run_spatial_config2_metric1_2(
        num_ut_per_sector=1,
        num_drops=1,
        ut_chunk_size=1,
        carrier_frequency=30e9,
        precision="single",
        devices=["cpu"],
        seed=1,
        spec_version="16.1",
    )

    assert result is not None
    assert observed["include_historical_sir"] is True
    assert "wideband_sinr includes thermal noise" in observed["metadata_note"]


@pytest.mark.parametrize(
    "invalid_args,error",
    [
        (["--batch-size", "0"], "--batch-size must be positive"),
        (
            ["--cdf-percentiles", "50", "40"],
            "--cdf-percentiles values must be strictly increasing",
        ),
        (
            [
                "--phases",
                "spatial_consistency_metric1_2",
                "--scenarios",
                "UMi",
                "--frequencies-ghz",
                "29",
            ],
            "defined only for UMi at 30 GHz",
        ),
    ],
)
def test_cli_preflight_finishes_before_cleaning(
    tmp_path, monkeypatch, invalid_args, error
):
    """Invalid invocations cannot create or clean the requested output path."""

    output_dir = tmp_path / "calibration-output"
    monkeypatch.setattr(
        calibration,
        "_clean_output_dir",
        lambda _path: pytest.fail("cleaning ran before preflight completed"),
    )

    with pytest.raises(ValueError, match=error):
        calibration.main(
            [
                *invalid_args,
                "--devices",
                "cpu",
                "--output-dir",
                str(output_dir),
                "--no-plot",
                "--clean",
            ]
        )

    assert not output_dir.exists()


def test_reference_provenance_validation_precedes_cleaning(tmp_path, monkeypatch):
    """An under-specified reference file cannot trigger destructive cleanup."""

    output_dir = tmp_path / "calibration-output"
    reference_json = tmp_path / "reference.json"
    reference_json.write_text(json.dumps({"metadata": {}}))
    monkeypatch.setattr(
        calibration,
        "_clean_output_dir",
        lambda _path: pytest.fail("cleaning ran before provenance validation"),
    )

    with pytest.raises(ValueError, match="must record source_files"):
        calibration.main(
            [
                "--devices",
                "cpu",
                "--output-dir",
                str(output_dir),
                "--reference-json",
                str(reference_json),
                "--no-plot",
                "--clean",
            ]
        )

    assert not output_dir.exists()


def test_inf_batches_record_effective_workload(monkeypatch):
    """The per-result metadata must disambiguate InF batch grouping."""

    observed_seeds = []

    def run_once(_cfg, **kwargs):
        observed_seeds.append(kwargs["seed"])
        return _fake_result(
            kwargs["seed"],
            metadata={"num_bs": 18, "ut_chunk_size": 256},
        )

    monkeypatch.setattr(calibration, "_run_inf_once", run_once)
    monkeypatch.setattr(calibration.torch.cuda, "is_available", lambda: False)

    result = calibration._run_inf_batches(
        cfg=object(),
        num_batches=5,
        batch_size=1,
        num_ut_per_bs=90,
        ut_chunk_size=256,
        seed=1234,
        precision="single",
        devices=["cuda:0"],
        spec_version=calibration.CALIBRATION_SPEC_VERSION,
    )

    assert result.num_samples == 10
    assert observed_seeds == [1234, 1235, 1236, 1237, 1238]
    assert result.metadata == {
        "num_bs": 18,
        "ut_chunk_size": 256,
        "num_batches": 5,
        "batch_size": 1,
        "num_ut_per_bs": 90,
        "effective_num_topology_drops": 5,
    }


def test_phase1_batches_record_effective_workload(monkeypatch):
    """Phase 1 records batch grouping and the effective drop count."""

    observed = []

    def run_once(_cfg, **kwargs):
        observed.append((kwargs["seed"], kwargs["device"]))
        return _fake_result(kwargs["seed"], phase="phase1")

    monkeypatch.setattr(calibration, "_run_phase1_once", run_once)
    monkeypatch.setattr(calibration.torch.cuda, "is_available", lambda: False)

    result = calibration._run_phase1_batches(
        cfg=object(),
        num_batches=3,
        batch_size=2,
        num_ut_per_sector=16,
        seed=1234,
        precision="single",
        devices=["cuda:0", "cuda:1"],
        spec_version=calibration.CALIBRATION_SPEC_VERSION,
    )

    assert result.num_samples == 6
    assert observed == [(1234, "cuda:0"), (1235, "cuda:1"), (1236, "cuda:0")]
    assert result.metadata == {
        "num_batches": 3,
        "batch_size": 2,
        "num_ut_per_sector": 16,
        "effective_num_topology_drops": 6,
    }


def test_phase2_batches_record_effective_workload(monkeypatch):
    """Phase 2 records batch grouping and forwards phase options."""

    observed = []

    def run_once(_cfg, **kwargs):
        observed.append(kwargs)
        return _fake_result(kwargs["seed"], phase="blockage_model_a")

    monkeypatch.setattr(calibration, "_run_phase2_once", run_once)
    monkeypatch.setattr(calibration.torch.cuda, "is_available", lambda: False)

    result = calibration._run_phase2_batches(
        cfg=object(),
        num_batches=2,
        batch_size=4,
        num_ut_per_sector=1,
        seed=7,
        precision="single",
        devices=["cuda:0"],
        phase_name="blockage_model_a",
        enable_blockage=True,
        blockage_self_blocking="portrait",
        spec_version=calibration.CALIBRATION_SPEC_VERSION,
    )

    assert result.num_samples == 4
    assert [kwargs["seed"] for kwargs in observed] == [7, 8]
    assert all(kwargs["phase_name"] == "blockage_model_a" for kwargs in observed)
    assert all(kwargs["enable_blockage"] for kwargs in observed)
    assert all(kwargs["blockage_self_blocking"] == "portrait" for kwargs in observed)
    assert result.metadata == {
        "num_batches": 2,
        "batch_size": 4,
        "num_ut_per_sector": 1,
        "effective_num_topology_drops": 8,
    }


def test_existing_runs_retain_workload_metadata(tmp_path):
    """Indexing an existing CDF must not discard its workload provenance."""

    cdf_dir = tmp_path / "cdfs"
    cdf_dir.mkdir()
    path = cdf_dir / "InF-SL_3.5GHz_inf.json"
    path.write_text(
        json.dumps(
            {
                "metadata": {
                    "scenario": "InF-SL",
                    "frequency_ghz": 3.5,
                    "phase": "inf",
                    "num_samples": 8100,
                    "num_batches": 5,
                    "batch_size": 1,
                    "effective_num_topology_drops": 5,
                    "num_bs": 18,
                    "num_ut_per_bs": 90,
                    "ut_chunk_size": 256,
                },
                "metrics": {"coupling_loss": {}},
            }
        )
    )

    run = calibration._existing_cdf_runs(tmp_path)[path.stem]
    assert run["workload"] == {
        "num_batches": 5,
        "batch_size": 1,
        "effective_num_topology_drops": 5,
        "num_ut_per_bs": 90,
        "num_bs": 18,
        "ut_chunk_size": 256,
    }
