#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Generate TR 38.901 calibration curve data.

This script is intentionally not named ``test_*.py`` and is therefore not
collected by pytest. It is a manually runnable statistical calibration script.
The Phase 1, Phase 2 Config 1, and Phase 2 Config 2 runs cover the UMi, UMa,
and InH indoor-office scenarios using the TR 38.901 V16.1 model. The bundled
baseline reference curves are from the Release-14-era calibration campaign in
Section 7.8. The blockage and spatial-consistency references are likewise from
the V14 additional-feature calibrations in Tables 7.8-5 and 7.8-6. The
normative ``inf`` run uses the V16.1 Table 7.8-7 10 m hall. All reference
curves are bundled as static JSON data extracted from named 3GPP calibration
workbooks.

The script is self-contained within the Sionna repository. It generates JSON
files with empirical CDF curves for Phase 1, Phase 2 Config 1, Phase 2
Config 2, InF, the supported blockage additional-feature calibrations, and
the spatial-consistency Config1 metric-1/2, Config1 metric-3/6, and Config2
static metric-1/2 calibrations for UMi at 30 GHz. For blockage Model B and
spatial-consistency metric 3/6, it generates line curves. It does not compute
error metrics. By default, it also
regenerates figures that overlay generated curves with the bundled 3GPP
reference curves when reference curves are available.

Generated ``coupling_loss`` values are positive losses. Negative values kept
for comparison with historical workbook curves are stored separately as
``historical_path_gain``. Table 7.8-5 Config2 stores noise-inclusive
``wideband_sinr``; its contribution-era interference-only curve is retained
only as ``historical_wideband_sir``.

Important InH and InF reference notes
-------------------------------------

The bundled InH reference curves predate the final V16.1 Indoor-Office
parameter tables. In particular, the ASA, ASD, and ZSA distributions and
several related cluster parameters changed significantly. A V16.1 InH run is
therefore not expected to overlay those angular-spread reference curves.

The public R1-1909704 InF calibration curves use 25 m sparse-clutter halls,
15 m dense-clutter halls, and pre-final delay-spread assumptions. They are not
bundled or overlaid as direct references for the final V16.1 Table 7.8-7
10 m calibration geometry.

Baseline-only example from the Sionna repository root (the three baseline
phases are also the ``--phases`` default):

.. code-block:: bash

   python test/unit/channel/tr38901_calibration.py \\
       --phases phase1 config1 config2 \\
       --phase1-num-batches 100 \\
       --phase2-num-batches 100

Baseline-only example from the Sionna ``test`` folder:

.. code-block:: bash

   python unit/channel/tr38901_calibration.py \\
       --phases phase1 config1 config2 \\
       --phase1-num-batches 100 \\
       --phase2-num-batches 100

Full-bundle example from the Sionna repository root using the approved
coherent workload:

.. code-block:: bash

   python test/unit/channel/tr38901_calibration.py \\
       --phases phase1 config1 config2 inf \\
         blockage_model_a blockage_model_b \\
         spatial_consistency_metric1_2 \\
         spatial_consistency_metric3_6 \\
         spatial_consistency_config2_metric1_2 \\
       --phase1-num-batches 100 \\
       --phase2-num-batches 100 \\
       --batch-size 1 \\
       --phase1-uts-per-sector 16 \\
       --phase2-uts-per-sector 1 \\
       --spatial-metric1-2-num-ut-per-sector 20 \\
       --spatial-metric1-2-num-drops 100 \\
       --spatial-metric1-2-ut-chunk-size 128 \\
       --spatial-metric3-6-num-ut-per-sector 200 \\
       --spatial-metric3-6-num-drops 20 \\
       --inf-num-batches 5 \\
       --inf-uts-per-bs 90 \\
       --inf-ut-chunk-size 256 \\
       --seed 1234 \\
       --precision single \\
       --output-dir /tmp/tr38901-full \\
       --clean

This command is expected to create 46 result JSON files under ``cdfs/``,
46 manifest runs under ``runs`` in ``manifest.json``, and 18 figures. The
explicit spatial-consistency and InF values above define the approved
full-bundle workload; they do not change the CLI defaults documented below.

Each result JSON stores its applicable workload provenance in its
``metadata`` object, for example batch/drop counts, batch size, UT count,
and chunk size. The manifest mirrors those per-result fields under
``runs.<key>.workload`` so that runs with different workloads remain
individually reproducible. ``num_samples`` remains a separate result and
manifest field.

By default, the script uses all visible CUDA devices if CUDA is available and
``cpu`` otherwise. Results are written to
``test/unit/channel/tr38901_calibration_results`` under the Sionna repository
root unless ``--output-dir`` is provided. With multiple devices, independent
scenario/frequency/phase runs execute concurrently in one spawned process per
device. Request index ``i`` is statically assigned to device index
``i % num_devices``; completion timing cannot change that assignment. The
parent process alone writes result files and plots. Each result records its
assigned device because deterministic Sionna random streams use device-specific
seed offsets.

The script sets :attr:`sionna.phy.config.Config.device`,
:attr:`sionna.phy.config.Config.precision`, and
:attr:`sionna.phy.config.Config.seed` for each run. This is intentional for a
standalone calibration driver and should not be interpreted as a thread-safe
library pattern.

Arguments
---------

``--scenarios``
    Scenarios to simulate. Default: ``UMi UMa InH``.
``--frequencies-ghz``
    Carrier frequencies in GHz. If omitted, defaults are selected per phase:
    ``6 30 70`` for Phase 1, ``6 30 60 70`` for Phase 2 Config 1/2, and
    ``3.5 28`` for InF, and ``30`` for the blockage and
    spatial-consistency additional-feature calibrations.
``--phases``
    Calibration phases to generate. Use ``phase1``, ``config1``, and/or
    ``config2`` for the baseline calibration. Use ``blockage_model_a`` or
    ``blockage_model_b`` for the TR 38.901 Table 7.8-6 blockage
    additional-feature calibrations. Use
    ``spatial_consistency_metric1_2`` or ``spatial_consistency_metric3_6``
    for the TR 38.901 Table 7.8-5 Config1 spatial-consistency calibrations.
    Use ``spatial_consistency_config2_metric1_2`` for the Table 7.8-5
    Config2 static CDF metrics. The dynamic Config2 varying-rate metrics are
    intentionally not generated. Use ``inf`` for the normative TR 38.901
    V16.1 Table 7.8-7 10 m hall.
    Default: the three baseline phases.
``--phase1-num-batches``
    Number of independent Phase 1 topology drops per scenario and frequency.
    Default: ``100``.
``--phase2-num-batches``
    Number of independent Phase 2 topology drops per scenario, frequency, and
    configuration. For blockage Model B, this is the number of independent
    CDL-E ray-coupling realizations averaged in linear RSRP. Default: ``100``.
``--batch-size``
    Number of independent drops processed together. Default: ``1``.
``--phase1-uts-per-sector``
    Number of UTs per sector for Phase 1 drops. InH UTs are still dropped
    uniformly over the full indoor-office room. Default: ``16``.
``--phase2-uts-per-sector``
    Number of UTs per sector for Phase 2 drops. InH UTs are still dropped
    uniformly over the full indoor-office room. Default: ``1``.
``--spatial-metric1-2-num-ut-per-sector``
    Number of UTs per sector for one spatial-consistency metric-1/2 drop.
    Default: ``20``.
``--spatial-metric1-2-num-drops``
    Number of independent spatial-consistency metric-1/2 drops. Default:
    ``10``.
``--spatial-metric1-2-ut-chunk-size``
    Maximum number of UTs processed together for metric 1/2. Default:
    ``256``.
``--spatial-metric3-6-num-ut-per-sector``
    Number of UTs per sector for one spatial-consistency metric-3/6 drop.
    Default: ``200``.
``--spatial-metric3-6-num-drops``
    Number of independent spatial-consistency metric-3/6 drops. Default:
    ``10``.
``--spatial-metric3-6-distances-m``
    Distance-bin lower edges in meters for metric 3/6. Default: ``0 1 ... 130``.
``--inf-num-batches``
    Number of independent InF topology drops per sub-scenario and frequency.
    Default: ``1``.
``--inf-uts-per-bs``
    Number of UTs per InF BS for each topology drop. Table 7.8-7 uses 18 BSs,
    so the default ``90`` gives 1620 UTs per drop.
``--inf-ut-chunk-size``
    Maximum number of InF UTs processed together for fast-fading spread
    metrics. Default: ``256``.
``--seed``
    Base random seed. Default: ``1234``.
``--precision``
    Sionna precision, either ``single`` or ``double``. Default: ``single``.
``--devices``
    Devices used concurrently, with one spawned worker process per device.
    Requests are assigned statically by request index modulo device count.
    Default: all visible CUDA devices if available, otherwise ``cpu``.
``--cdf-percentiles``
    Percentiles, in percent, at which CDF values are stored. Default:
    ``0 1 2 ... 100``.
``--output-dir``
    Directory for generated CDF JSON files, figures, and the manifest. Default:
    ``test/unit/channel/tr38901_calibration_results`` under the Sionna
    repository root.
``--reference-json``
    JSON file containing 3GPP reference curves. Default:
    ``test/unit/channel/tr38901_calibration_results/reference_curves.json``.
``--no-plot``
    Do not regenerate figures after writing CDF JSON files. Default: disabled.
``--clean``
    Remove generated files in the output directory before running. The bundled
    reference-curve JSON is preserved. Default: disabled.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
import importlib.util
import json
import math
import multiprocessing
from pathlib import Path
import shutil
import sys
from typing import Sequence

import numpy as np
import torch


SCRIPT_PATH = Path(__file__).resolve()
DEFAULT_OUTPUT_DIR = SCRIPT_PATH.parent / "tr38901_calibration_results"
DEFAULT_REFERENCE_JSON = DEFAULT_OUTPUT_DIR / "reference_curves.json"
PLOT_SCRIPT = DEFAULT_OUTPUT_DIR / "plot_tr38901_calibration.py"
DEFAULT_SCENARIOS = ["UMi", "UMa", "InH"]
DEFAULT_INF_SCENARIOS = ["InF-SL", "InF-DL", "InF-SH", "InF-DH"]
INF_NORMATIVE_PHASE = "inf"
INF_NORMATIVE_PROFILE = "tr38901_v16.1_table_7.8-7"
DEFAULT_FREQUENCIES_BY_PHASE = {
    "phase1": [6.0, 30.0, 70.0],
    "config1": [6.0, 30.0, 60.0, 70.0],
    "config2": [6.0, 30.0, 60.0, 70.0],
    "inf": [3.5, 28.0],
    "blockage_model_a": [30.0],
    "blockage_model_b": [30.0],
    "spatial_consistency_metric1_2": [30.0],
    "spatial_consistency_metric3_6": [30.0],
    "spatial_consistency_config2_metric1_2": [30.0],
}
DEFAULT_PHASES = ["phase1", "config1", "config2"]
PHASE_CHOICES = [
    *DEFAULT_PHASES,
    "inf",
    "blockage_model_a",
    "blockage_model_b",
    "spatial_consistency_metric1_2",
    "spatial_consistency_metric3_6",
    "spatial_consistency_config2_metric1_2",
]
DEFAULT_PERCENTILES = [float(p) for p in range(101)]
DEFAULT_SPATIAL_METRIC3_6_DISTANCES = [float(d) for d in range(131)]
CALIBRATION_SPEC_VERSION = "16.1"
CALIBRATION_SCHEMA_VERSION = 2
PHASE_STANDARDS = {
    "phase1": "3GPP TR 38.901 v16.1.0 Section 7.8 calibration",
    "config1": "3GPP TR 38.901 v16.1.0 Section 7.8 calibration",
    "config2": "3GPP TR 38.901 v16.1.0 Section 7.8 calibration",
    "inf": "3GPP TR 38.901 v16.1.0 Table 7.8-7 indoor-factory calibration",
    "blockage_model_a": (
        "3GPP TR 38.901 v16.1.0 Table 7.8-6 additional-feature blockage calibration"
    ),
    "blockage_model_b": (
        "3GPP TR 38.901 v16.1.0 Table 7.8-6 additional-feature blockage calibration"
    ),
    "spatial_consistency_metric1_2": (
        "3GPP TR 38.901 v16.1.0 Table 7.8-5 additional-feature "
        "spatial-consistency calibration"
    ),
    "spatial_consistency_metric3_6": (
        "3GPP TR 38.901 v16.1.0 Table 7.8-5 additional-feature "
        "spatial-consistency calibration"
    ),
    "spatial_consistency_config2_metric1_2": (
        "3GPP TR 38.901 v16.1.0 Table 7.8-5 additional-feature "
        "spatial-consistency Config2 static calibration"
    ),
}

INH_REFERENCE_NOTE = (
    "The bundled InH reference curves come from the Release-14-era "
    "calibration campaign. The final V16.1 Indoor-Office ASA, ASD, ZSA, and "
    "related cluster parameters differ significantly, so offsets in those "
    "curves are expected."
)

INF_REFERENCE_NOTE = (
    "The public R1-1909704 InF curves use 15/25 m halls and pre-final "
    "delay-spread assumptions, so they are not bundled or overlaid as direct "
    "references for the normative V16.1 10 m geometry."
)

INF_SCENARIO_ALIASES = {
    "sl": "SL",
    "inf-sl": "SL",
    "sparse-low": "SL",
    "sparse-clutter-low-bs": "SL",
    "dl": "DL",
    "inf-dl": "DL",
    "dense-low": "DL",
    "dense-clutter-low-bs": "DL",
    "sh": "SH",
    "inf-sh": "SH",
    "sparse-high": "SH",
    "sparse-clutter-high-bs": "SH",
    "dh": "DH",
    "inf-dh": "DH",
    "dense-high": "DH",
    "dense-clutter-high-bs": "DH",
}

PARSER = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.RawDescriptionHelpFormatter,
)
PARSER.add_argument(
    "--scenarios",
    nargs="+",
    default=DEFAULT_SCENARIOS,
    metavar="SCENARIO",
    help="Scenarios to simulate. Default: UMi UMa InH.",
)
PARSER.add_argument(
    "--frequencies-ghz",
    nargs="+",
    type=float,
    default=None,
    metavar="FC",
    help=(
        "Carrier frequencies in GHz. If omitted, defaults are selected per "
        "phase: phase1 uses 6 30 70; config1/config2 use 6 30 60 70; "
        "InF uses 3.5 28; blockage_model_a/blockage_model_b and "
        "spatial-consistency phases use 30."
    ),
)
PARSER.add_argument(
    "--phases",
    nargs="+",
    choices=PHASE_CHOICES,
    default=DEFAULT_PHASES,
    metavar="PHASE",
    help=(
        "Calibration phases to generate. Default: phase1 config1 config2. "
        "Additional supported phases: inf blockage_model_a blockage_model_b "
        "spatial_consistency_metric1_2 "
        "spatial_consistency_metric3_6 "
        "spatial_consistency_config2_metric1_2."
    ),
)
PARSER.add_argument(
    "--phase1-num-batches",
    type=int,
    default=100,
    metavar="N",
    help=(
        "Number of independent Phase 1 topology drops per scenario and "
        "frequency. Default: 100."
    ),
)
PARSER.add_argument(
    "--phase2-num-batches",
    type=int,
    default=100,
    metavar="N",
    help=(
        "Number of independent Phase 2 topology drops per scenario, "
        "frequency, and configuration. For blockage_model_b, this is the "
        "number of independent CDL-E ray-coupling realizations averaged in "
        "linear RSRP. Default: 100."
    ),
)
PARSER.add_argument(
    "--batch-size",
    type=int,
    default=1,
    metavar="N",
    help="Number of independent drops processed together. Default: 1.",
)
PARSER.add_argument(
    "--phase1-uts-per-sector",
    type=int,
    default=16,
    metavar="N",
    help=(
        "Number of UTs per sector for Phase 1 drops. InH UTs are still "
        "dropped uniformly over the full indoor-office room. Default: 16."
    ),
)
PARSER.add_argument(
    "--phase2-uts-per-sector",
    type=int,
    default=1,
    metavar="N",
    help=(
        "Number of UTs per sector for Phase 2 drops. InH UTs are still "
        "dropped uniformly over the full indoor-office room. Default: 1."
    ),
)
PARSER.add_argument(
    "--spatial-metric1-2-num-ut-per-sector",
    type=int,
    default=20,
    metavar="N",
    help=(
        "Number of UTs per sector for one spatial-consistency metric-1/2 "
        "drop. Default: 20."
    ),
)
PARSER.add_argument(
    "--spatial-metric1-2-num-drops",
    type=int,
    default=10,
    metavar="N",
    help=("Number of independent spatial-consistency metric-1/2 drops. Default: 10."),
)
PARSER.add_argument(
    "--spatial-metric1-2-ut-chunk-size",
    type=int,
    default=256,
    metavar="N",
    help=(
        "Maximum number of UTs processed together for "
        "spatial-consistency metric 1/2. Default: 256."
    ),
)
PARSER.add_argument(
    "--spatial-metric3-6-num-ut-per-sector",
    type=int,
    default=200,
    metavar="N",
    help=(
        "Number of UTs per sector for one spatial-consistency metric-3/6 "
        "drop. Default: 200."
    ),
)
PARSER.add_argument(
    "--spatial-metric3-6-num-drops",
    type=int,
    default=10,
    metavar="N",
    help=("Number of independent spatial-consistency metric-3/6 drops. Default: 10."),
)
PARSER.add_argument(
    "--spatial-metric3-6-distances-m",
    nargs="+",
    type=float,
    default=DEFAULT_SPATIAL_METRIC3_6_DISTANCES,
    metavar="D",
    help=(
        "Distance-bin lower edges in meters for spatial-consistency "
        "metric 3/6. Default: 0 1 ... 130."
    ),
)
PARSER.add_argument(
    "--inf-num-batches",
    type=int,
    default=1,
    metavar="N",
    help=(
        "Number of independent InF topology drops per sub-scenario and "
        "frequency. Default: 1."
    ),
)
PARSER.add_argument(
    "--inf-uts-per-bs",
    type=int,
    default=90,
    metavar="N",
    help=(
        "Number of UTs per InF BS for each topology drop. Table 7.8-7 uses "
        "18 BSs, so the default 90 gives 1620 UTs per drop."
    ),
)
PARSER.add_argument(
    "--inf-ut-chunk-size",
    type=int,
    default=256,
    metavar="N",
    help=(
        "Maximum number of InF UTs processed together for fast-fading "
        "spread metrics. Default: 256."
    ),
)
PARSER.add_argument(
    "--seed",
    type=int,
    default=1234,
    metavar="N",
    help="Base random seed. Default: 1234.",
)
PARSER.add_argument(
    "--precision",
    choices=["single", "double"],
    default="single",
    help="Sionna precision. Default: single.",
)
PARSER.add_argument(
    "--devices",
    nargs="+",
    default=None,
    metavar="DEVICE",
    help=(
        "Devices used concurrently, with one spawned worker per device. "
        "Default: all visible CUDA devices if available, otherwise cpu."
    ),
)
PARSER.add_argument(
    "--cdf-percentiles",
    nargs="+",
    type=float,
    default=DEFAULT_PERCENTILES,
    metavar="P",
    help="Percentiles, in percent, at which CDF values are stored. Default: 0 1 ... 100.",
)
PARSER.add_argument(
    "--output-dir",
    type=Path,
    default=DEFAULT_OUTPUT_DIR,
    metavar="PATH",
    help=(
        "Directory for generated CDF JSON files, figures, and the manifest. Default: "
        "test/unit/channel/tr38901_calibration_results under the Sionna "
        "repository root."
    ),
)
PARSER.add_argument(
    "--reference-json",
    type=Path,
    default=DEFAULT_REFERENCE_JSON,
    metavar="PATH",
    help=(
        "JSON file containing 3GPP reference curves. Default: "
        "test/unit/channel/tr38901_calibration_results/reference_curves.json."
    ),
)
PARSER.add_argument(
    "--no-plot",
    action="store_true",
    help="Do not regenerate figures after writing CDF JSON files.",
)
PARSER.add_argument(
    "--clean",
    action="store_true",
    help="Remove generated files in the output directory before running.",
)


def _find_sionna_root() -> Path:
    """Find the nearest parent containing the Sionna source tree."""

    for parent in SCRIPT_PATH.parents:
        if (parent / "src" / "sionna").exists():
            return parent
    raise RuntimeError("Could not locate the Sionna repository root")


SIONNA_ROOT = _find_sionna_root()
SIONNA_SRC = SIONNA_ROOT / "src"
if str(SIONNA_SRC) not in sys.path:
    sys.path.insert(0, str(SIONNA_SRC))

from sionna.phy import SPEED_OF_LIGHT, config as sionna_config  # noqa: E402
from sionna.phy.channel.tr38901 import (  # noqa: E402
    BlockageModelB,
    CDL,
    ChannelCoefficientsGenerator,
    InH,
    InHScenario,
    InF,
    InFScenario,
    LSP,
    LSPGenerator,
    PanelArray,
    Rays,
    Topology,
    UMa,
    UMaScenario,
    UMi,
    UMiScenario,
    angular_spreads_from_rays,
    delay_spread_from_rays,
    prb_singular_values,
)
from sionna.phy.channel.utils import deg_2_rad, rad_2_deg  # noqa: E402
from sionna.sys import (  # noqa: E402
    gen_tr38901_indoor_factory_topology,
    gen_tr38901_indoor_office_topology,
    gen_tr38901_multicell_topology,
    geometry_sinr_db,
    geometry_sir_db,
    serving_indices,
    wideband_sir_db,
)


METRIC_LABELS = {
    "coupling_loss": "Coupling Loss [dB]",
    "historical_path_gain": "Historical Path Gain [dB]",
    "geometry_sinr": "Geometry SINR [dB]",
    "geometry_sir": "Geometry SIR [dB]",
    "wideband_sir": "Wideband SIR [dB]",
    "wideband_sinr": "Wideband SINR [dB]",
    "historical_wideband_sir": "Historical Wideband SIR [dB]",
    "delay_spread_ns": "Delay Spread [ns]",
    "asd_deg": "ASD [deg]",
    "zsd_deg": "ZSD [deg]",
    "asa_deg": "ASA [deg]",
    "zsa_deg": "ZSA [deg]",
    "sv1_db": "1st Singular Value [dB]",
    "sv2_db": "2nd Singular Value [dB]",
    "sv_ratio_db": "SV Ratio [dB]",
    "rsrp_db": "RSRP [dB]",
    "relative_rsrp_db": "Relative RSRP [dB]",
    "third_cluster_delay_corr": "Third-cluster Delay Correlation",
    "third_cluster_aoa_corr": "Third-cluster AOA Correlation",
    "los_state_corr": "LOS/NLOS-state Correlation",
    "channel_response_corr": "Channel-response Correlation",
}


@dataclass(frozen=True)
class Phase1Config:
    """Large-scale calibration parameters from TR 38.901 V16.1 Table 7.8-1."""

    scenario: str
    fc_ghz: float
    isd_m: float
    bs_height_m: float
    min_bs_ut_dist_m: float
    indoor_probability: float
    tx_power_dbm: float
    bandwidth_hz: float
    noise_figure_db: float = 9.0
    num_rings: int = 2
    bs_electrical_downtilt_deg: float = 102.0
    bs_array_rows: int = 10
    bs_array_vertical_spacing_lambda: float = 0.5
    room_length_m: float | None = None
    room_width_m: float | None = None

    @property
    def scenario_lower(self) -> str:
        return self.scenario.lower()


@dataclass(frozen=True)
class Phase2Config(Phase1Config):
    """Full-channel calibration parameters from TR 38.901 V16.1 Table 7.8-2."""

    config_id: int = 1
    subcarrier_spacing_hz: float = 15e3
    prb_num_subcarriers: int = 12

    @property
    def phase_name(self) -> str:
        return f"config{self.config_id}"


@dataclass(frozen=True)
class InFCalibrationConfig:
    """Indoor-factory calibration parameters from TR 38.901 Table 7.8-7."""

    scenario: str
    factory_scenario: str
    fc_ghz: float
    phase: str
    profile: str
    tx_power_dbm: float = 30.0
    bandwidth_hz: float = 100e6
    noise_figure_db: float = 9.0


@dataclass(frozen=True)
class TopologyBundle:
    """Topology tensors for one TR 38.901 calibration drop."""

    ut_loc: torch.Tensor
    bs_loc: torch.Tensor
    ut_orientations: torch.Tensor
    bs_orientations: torch.Tensor
    ut_velocities: torch.Tensor
    in_state: torch.Tensor
    distance_2d_in: torch.Tensor | None
    los: None
    bs_virtual_loc: torch.Tensor
    bs_site_ids: torch.Tensor
    site_positions: torch.Tensor

    @property
    def num_ut(self) -> int:
        return self.ut_loc.shape[1]


@dataclass(frozen=True)
class LargeScaleDetails:
    """Intermediate quantities needed to align Phase 2 fast fading drops."""

    outdoor_los: torch.Tensor
    basic_outdoor_pathloss_db: torch.Tensor
    shadow_fading_db: torch.Tensor
    o2i_loss_db: torch.Tensor
    lsp: LSP


@dataclass(frozen=True)
class CalibrationResult:
    """Metric samples for one scenario, frequency, and phase."""

    scenario: str
    fc_ghz: float
    phase: str
    metrics: dict[str, np.ndarray]
    serving_bs: np.ndarray
    in_state: np.ndarray
    o2i_is_high_loss: np.ndarray
    metadata: dict | None = None

    @property
    def num_samples(self) -> int:
        first = next(iter(self.metrics.values()))
        return int(first.size)


@dataclass(frozen=True)
class LineCalibrationResult:
    """Line-curve metrics for one deterministic calibration setup."""

    scenario: str
    fc_ghz: float
    phase: str
    x_label: str
    x_values: np.ndarray
    metrics: dict[str, np.ndarray]
    metadata: dict

    @property
    def num_samples(self) -> int:
        return int(self.x_values.size)


def _default_devices() -> list[str]:
    """Return all visible CUDA devices, or CPU when CUDA is unavailable."""

    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        return [f"cuda:{index}" for index in range(torch.cuda.device_count())]
    return ["cpu"]


def _resolve_output_dir(path: Path) -> Path:
    """Resolve ``path`` relative to the current working directory."""

    if path.is_absolute():
        return path
    return (Path.cwd() / path).resolve()


def _manifest_path(path: Path, output_dir: Path) -> str:
    """Return a stable path string for generated manifests."""

    try:
        return str(path.relative_to(output_dir))
    except ValueError:
        return str(path)


def _clean_output_dir(output_dir: Path) -> None:
    """Remove generated CDF data while preserving bundled reference data."""

    for child in (output_dir / "cdfs", output_dir / "figures", output_dir / "samples"):
        if child.exists():
            shutil.rmtree(child)
    for filename in ("manifest.json", "full_calibration_summary.json"):
        path = output_dir / filename
        if path.exists():
            path.unlink()


def _load_plotter():
    """Load the bundled calibration plotting helper."""

    spec = importlib.util.spec_from_file_location(
        "sionna_tr38901_calibration_plotter", PLOT_SCRIPT
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load calibration plotter: {PLOT_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.plot_calibration_results


def _frequencies_for_phase(
    phase: str, requested_frequencies_ghz: Sequence[float] | None
) -> list[float]:
    """Return the frequency list for one calibration phase."""

    if requested_frequencies_ghz is not None:
        return [float(f) for f in requested_frequencies_ghz]
    return list(DEFAULT_FREQUENCIES_BY_PHASE[phase])


def _scenarios_for_phase(phase: str, requested_scenarios: Sequence[str]) -> list[str]:
    """Return the scenario list for one calibration phase."""

    if phase == INF_NORMATIVE_PHASE:
        scenarios = []
        for scenario in requested_scenarios:
            key = scenario.strip().lower().replace("_", "-").replace(" ", "-")
            if key in INF_SCENARIO_ALIASES:
                scenarios.append(f"InF-{INF_SCENARIO_ALIASES[key]}")
        if scenarios:
            return list(dict.fromkeys(scenarios))
        if list(requested_scenarios) == DEFAULT_SCENARIOS:
            return list(DEFAULT_INF_SCENARIOS)
        raise ValueError(
            f"The {phase} calibration is defined only for InF-SL, InF-DL, "
            "InF-SH, and InF-DH."
        )

    umi_only_phases = (
        "blockage_model_a",
        "blockage_model_b",
        "spatial_consistency_metric1_2",
        "spatial_consistency_metric3_6",
        "spatial_consistency_config2_metric1_2",
    )
    if phase in umi_only_phases:
        scenarios = [
            scenario
            for scenario in requested_scenarios
            if scenario.strip().lower()
            in ("umi", "umi-street canyon", "umi-street-canyon")
        ]
        if not scenarios:
            raise ValueError(
                f"The {phase} calibration is defined only for UMi. "
                "Include UMi in --scenarios or omit --scenarios."
            )
        return ["UMi"]
    return list(requested_scenarios)


def _get_phase1_config(scenario: str, fc_ghz: float) -> Phase1Config:
    """Return the TR 38.901 V16.1 Phase 1 calibration config."""

    scenario_key = scenario.strip().lower()
    is_high_frequency = fc_ghz >= 30.0
    bandwidth_hz = 100e6 if is_high_frequency else 20e6

    if scenario_key in ("umi", "umi-street canyon", "umi-street-canyon"):
        return Phase1Config(
            scenario="UMi",
            fc_ghz=float(fc_ghz),
            isd_m=200.0,
            bs_height_m=10.0,
            min_bs_ut_dist_m=10.0,
            indoor_probability=0.8,
            tx_power_dbm=35.0 if is_high_frequency else 44.0,
            bandwidth_hz=bandwidth_hz,
        )

    if scenario_key == "uma":
        return Phase1Config(
            scenario="UMa",
            fc_ghz=float(fc_ghz),
            isd_m=500.0,
            bs_height_m=25.0,
            min_bs_ut_dist_m=35.0,
            indoor_probability=0.8,
            tx_power_dbm=35.0 if is_high_frequency else 49.0,
            bandwidth_hz=bandwidth_hz,
        )

    if scenario_key in ("inh", "indoor", "indoor-office", "inh-open"):
        return Phase1Config(
            scenario="InH",
            fc_ghz=float(fc_ghz),
            isd_m=20.0,
            bs_height_m=3.0,
            min_bs_ut_dist_m=0.0,
            indoor_probability=1.0,
            tx_power_dbm=24.0,
            bandwidth_hz=bandwidth_hz,
            bs_electrical_downtilt_deg=110.0,
            room_length_m=120.0,
            room_width_m=50.0,
        )

    raise ValueError(f"Unsupported calibration scenario: {scenario!r}")


def _get_phase2_config(scenario: str, fc_ghz: float, config_id: int) -> Phase2Config:
    """Return the TR 38.901 V16.1 Phase 2 calibration config."""

    if config_id not in (1, 2):
        raise ValueError(f"Phase 2 config_id must be 1 or 2, got {config_id!r}")
    base = _get_phase1_config(scenario, fc_ghz)
    return Phase2Config(**asdict(base), config_id=config_id)


def _get_inf_config(
    scenario: str,
    fc_ghz: float,
) -> InFCalibrationConfig:
    """Return the normative indoor-factory calibration config."""

    key = scenario.strip().lower().replace("_", "-").replace(" ", "-")
    if key not in INF_SCENARIO_ALIASES:
        raise ValueError(
            "The InF calibration scenario must be one of InF-SL, InF-DL, "
            "InF-SH, or InF-DH."
        )
    factory_scenario = INF_SCENARIO_ALIASES[key]
    return InFCalibrationConfig(
        scenario=f"InF-{factory_scenario}",
        factory_scenario=factory_scenario,
        fc_ghz=float(fc_ghz),
        phase=INF_NORMATIVE_PHASE,
        profile=INF_NORMATIVE_PROFILE,
    )


def _inf_calibration_hall_dimensions(
    cfg: InFCalibrationConfig,
) -> tuple[float, float, float]:
    """Return the normative TR 38.901 V16.1 hall dimensions."""

    if cfg.factory_scenario in ("SL", "DH"):
        length = 120.0
        width = 60.0
    else:
        length = 300.0
        width = 150.0
    return length, width, 10.0


def _inf_profile_note() -> str:
    """Describe the normative InF geometry and reference limitation."""

    return (
        "Normative TR 38.901 V16.1 Table 7.8-7 InF calibration using the "
        "10 m hall height. "
        + INF_REFERENCE_NOTE
    )


@contextmanager
def _scoped_global_config(precision: str, device: str):
    """Apply ``precision`` and ``device`` globally, restoring them on exit.

    Callers of this module import it as a test helper, so leaking the global
    settings would change the device and precision of every later test.
    """

    previous_precision = sionna_config.precision
    previous_device = sionna_config.device
    sionna_config.precision = precision
    sionna_config.device = device
    try:
        yield
    finally:
        sionna_config.precision = previous_precision
        sionna_config.device = previous_device


def _make_topology(
    cfg: Phase1Config,
    batch_size: int,
    num_ut_per_sector: int,
    seed: int,
    precision: str,
    device: str,
) -> TopologyBundle:
    """Generate the TR 38.901 calibration topology."""

    with _scoped_global_config(precision, device):
        sionna_config.seed = seed

        if cfg.scenario_lower == "inh":
            return _make_inh_topology(
                cfg,
                batch_size=batch_size,
                num_ut_per_sector=num_ut_per_sector,
                precision=precision,
                device=device,
            )

        topology, site_positions = gen_tr38901_multicell_topology(
            cfg.scenario_lower,
            batch_size=batch_size,
            num_ut_per_sector=num_ut_per_sector,
            carrier_frequency=cfg.fc_ghz * 1e9,
            num_rings=cfg.num_rings,
            isd=cfg.isd_m,
            bs_height=cfg.bs_height_m,
            min_bs_ut_dist=cfg.min_bs_ut_dist_m,
            indoor_probability=cfg.indoor_probability,
            return_site_positions=True,
            precision=precision,
            device=device,
        )
    (
        ut_loc,
        bs_loc,
        ut_orientations,
        bs_orientations,
        ut_velocities,
        in_state,
        los,
        bs_virtual_loc,
        bs_site_ids,
        _spatial_consistency_track_ids,
        distance_2d_in,
    ) = topology
    return TopologyBundle(
        ut_loc=ut_loc,
        bs_loc=bs_loc,
        ut_orientations=ut_orientations,
        bs_orientations=bs_orientations,
        ut_velocities=ut_velocities,
        in_state=in_state,
        distance_2d_in=distance_2d_in,
        los=los,
        bs_virtual_loc=bs_virtual_loc,
        bs_site_ids=bs_site_ids,
        site_positions=site_positions,
    )


def _torch_dtype(precision: str) -> torch.dtype:
    """Return the real torch dtype for the selected Sionna precision."""

    return torch.float64 if precision == "double" else torch.float32


def _make_inh_topology(
    cfg: Phase1Config,
    batch_size: int,
    num_ut_per_sector: int,
    precision: str,
    device: str,
) -> TopologyBundle:
    """Generate the TR 38.901 indoor-office calibration topology.

    The InH layout follows Figure 7.2-1 and Table 7.8-1 of TR 38.901:
    12 ceiling-mounted BS locations in a 120 m by 50 m room, spaced by 20 m,
    with three co-sited sectors at each BS location. UTs are uniformly dropped
    over the full room.
    """

    topology, site_positions = gen_tr38901_indoor_office_topology(
        batch_size=batch_size,
        num_ut_per_sector=num_ut_per_sector,
        room_length=float(cfg.room_length_m),
        room_width=float(cfg.room_width_m),
        isd=cfg.isd_m,
        bs_height=cfg.bs_height_m,
        ut_height=1.0,
        min_bs_ut_dist=cfg.min_bs_ut_dist_m,
        return_site_positions=True,
        precision=precision,
        device=device,
    )
    (
        ut_loc,
        bs_loc,
        ut_orientations,
        bs_orientations,
        ut_velocities,
        in_state,
        los,
        bs_virtual_loc,
        bs_site_ids,
    ) = topology

    return TopologyBundle(
        ut_loc=ut_loc,
        bs_loc=bs_loc,
        ut_orientations=ut_orientations,
        bs_orientations=bs_orientations,
        ut_velocities=ut_velocities,
        in_state=in_state,
        distance_2d_in=None,
        los=los,
        bs_virtual_loc=bs_virtual_loc,
        bs_site_ids=bs_site_ids,
        site_positions=site_positions,
    )


def _make_inf_topology(
    cfg: InFCalibrationConfig,
    batch_size: int,
    num_ut_per_bs: int,
    precision: str,
    device: str,
) -> TopologyBundle:
    """Generate the TR 38.901 indoor-factory calibration topology."""

    if num_ut_per_bs <= 0:
        raise ValueError("--inf-uts-per-bs must be positive")
    num_bs = 18
    hall_length, hall_width, hall_height = _inf_calibration_hall_dimensions(cfg)
    topology, site_positions = gen_tr38901_indoor_factory_topology(
        cfg.factory_scenario,
        batch_size=batch_size,
        num_ut=num_bs * num_ut_per_bs,
        hall_length=hall_length,
        hall_width=hall_width,
        hall_height=hall_height,
        min_bs_ut_dist=1.0,
        return_site_positions=True,
        precision=precision,
        device=device,
    )
    expected_hall_dimensions = _inf_calibration_hall_dimensions(cfg)
    if topology.factory_scenario != cfg.factory_scenario:
        raise ValueError(
            "Generated InF topology has factory_scenario="
            f"{topology.factory_scenario!r}; expected "
            f"{cfg.factory_scenario!r}"
        )
    if topology.hall_dimensions != expected_hall_dimensions:
        raise ValueError(
            "Generated InF topology has hall_dimensions="
            f"{topology.hall_dimensions}; expected "
            f"{expected_hall_dimensions}"
        )

    # Metadata was checked explicitly above; slicing avoids the warning that
    # guards unvalidated direct unpacking by external callers.
    (
        ut_loc,
        bs_loc,
        ut_orientations,
        bs_orientations,
        ut_velocities,
        in_state,
        los,
        bs_virtual_loc,
        bs_site_ids,
    ) = topology[:]

    return TopologyBundle(
        ut_loc=ut_loc,
        bs_loc=bs_loc,
        ut_orientations=ut_orientations,
        bs_orientations=bs_orientations,
        ut_velocities=ut_velocities,
        in_state=in_state,
        distance_2d_in=None,
        los=los,
        bs_virtual_loc=bs_virtual_loc,
        bs_site_ids=bs_site_ids,
        site_positions=site_positions,
    )


def _simple_arrays(fc_hz: float, precision: str, device: str):
    """Return omni single-element UT and BS arrays for Phase 1."""

    bs_array = PanelArray(
        num_rows_per_panel=1,
        num_cols_per_panel=1,
        polarization="single",
        polarization_type="V",
        antenna_pattern="omni",
        carrier_frequency=fc_hz,
        precision=precision,
        device=device,
    )
    ut_array = PanelArray(
        num_rows_per_panel=1,
        num_cols_per_panel=1,
        polarization="single",
        polarization_type="V",
        antenna_pattern="omni",
        carrier_frequency=fc_hz,
        precision=precision,
        device=device,
    )
    return ut_array, bs_array


def _make_inf_scenario_and_sampler(
    cfg: InFCalibrationConfig,
    topology: TopologyBundle,
    precision: str,
    device: str,
    spec_version: str,
):
    """Create an indoor-factory scenario and LSP sampler."""

    fc_hz = cfg.fc_ghz * 1e9
    ut_array, bs_array = _simple_arrays(fc_hz, precision, device)
    scenario = InFScenario(
        carrier_frequency=fc_hz,
        factory_scenario=cfg.factory_scenario,
        ut_array=ut_array,
        bs_array=bs_array,
        direction="downlink",
        hall_dimensions=_inf_calibration_hall_dimensions(cfg),
        enable_pathloss=True,
        enable_shadow_fading=True,
        precision=precision,
        device=device,
        spec_version=spec_version,
    )
    scenario.set_topology(
        topology.ut_loc,
        topology.bs_loc,
        topology.ut_orientations,
        topology.bs_orientations,
        topology.ut_velocities,
        topology.in_state,
        los=None,
        bs_virtual_loc=topology.bs_virtual_loc,
        bs_site_ids=topology.bs_site_ids,
        distance_2d_in=topology.distance_2d_in,
    )
    sampler = LSPGenerator(scenario)
    sampler.topology_updated_callback()
    return scenario, sampler


def _make_scenario_and_sampler(
    cfg: Phase1Config,
    topology: TopologyBundle,
    o2i_model: str,
    in_state: torch.Tensor,
    los: bool | torch.Tensor | None,
    precision: str,
    device: str,
    spec_version: str,
    enable_spatial_consistency: bool = False,
):
    """Create a system-level scenario and LSP sampler."""

    fc_hz = cfg.fc_ghz * 1e9
    ut_array, bs_array = _simple_arrays(fc_hz, precision, device)
    if cfg.scenario_lower == "inh":
        scenario = InHScenario(
            carrier_frequency=fc_hz,
            indoor_scenario="open",
            ut_array=ut_array,
            bs_array=bs_array,
            direction="downlink",
            enable_pathloss=True,
            enable_shadow_fading=True,
            precision=precision,
            device=device,
            spec_version=spec_version,
        )
    else:
        cls = UMiScenario if cfg.scenario_lower == "umi" else UMaScenario
        scenario = cls(
            carrier_frequency=fc_hz,
            o2i_model=o2i_model,
            ut_array=ut_array,
            bs_array=bs_array,
            direction="downlink",
            enable_pathloss=True,
            enable_shadow_fading=True,
            precision=precision,
            device=device,
            spec_version=spec_version,
        )
    scenario.set_spatial_consistency_enabled(enable_spatial_consistency)
    scenario.set_topology(
        topology.ut_loc,
        topology.bs_loc,
        topology.ut_orientations,
        topology.bs_orientations,
        topology.ut_velocities,
        in_state,
        los=los,
        bs_virtual_loc=topology.bs_virtual_loc,
        bs_site_ids=topology.bs_site_ids,
        distance_2d_in=topology.distance_2d_in,
    )
    sampler = LSPGenerator(scenario)
    sampler.topology_updated_callback()
    return scenario, sampler


def _phase1_bs_gain_db(
    los_zod_deg: torch.Tensor,
    los_aod_deg: torch.Tensor,
    bs_yaw_rad: torch.Tensor,
    cfg: Phase1Config,
) -> torch.Tensor:
    """Compute the Table 7.8-1 BS element and vertical DFT array gain."""

    dtype = los_zod_deg.dtype
    device = los_zod_deg.device

    theta = los_zod_deg
    yaw_deg = torch.rad2deg(bs_yaw_rad).unsqueeze(-1)
    phi = (los_aod_deg - yaw_deg + 180.0) % 360.0 - 180.0

    theta_3db = torch.tensor(65.0, dtype=dtype, device=device)
    phi_3db = torch.tensor(65.0, dtype=dtype, device=device)
    a_max = torch.tensor(30.0, dtype=dtype, device=device)
    g_e_max = torch.tensor(8.0, dtype=dtype, device=device)

    a_v = -torch.minimum(12.0 * ((theta - 90.0) / theta_3db) ** 2, a_max)
    a_h = -torch.minimum(12.0 * (phi / phi_3db) ** 2, a_max)
    element_gain = -torch.minimum(-(a_v + a_h), a_max) + g_e_max

    rows = cfg.bs_array_rows
    spacing = cfg.bs_array_vertical_spacing_lambda
    theta_rad = torch.deg2rad(theta)
    steer_rad = torch.deg2rad(
        torch.tensor(cfg.bs_electrical_downtilt_deg, dtype=dtype, device=device)
    )
    psi = 2.0 * torch.pi * spacing * (torch.cos(theta_rad) - torch.cos(steer_rad))
    numerator = torch.sin(rows * psi / 2.0)
    denominator = torch.sin(psi / 2.0)
    af = torch.where(
        torch.abs(denominator) < 1e-7,
        torch.full_like(denominator, float(rows)),
        torch.abs(numerator / denominator),
    )
    array_gain = 20.0 * torch.log10(torch.clamp(af, min=1e-30))
    array_gain = array_gain - 10.0 * np.log10(rows)
    return element_gain + array_gain


def _sample_path_gain_all_bs(
    cfg: Phase1Config,
    topology: TopologyBundle,
    seed: int,
    precision: str,
    device: str,
    spec_version: str,
    include_antenna_gain: bool,
    return_details: bool = False,
    enable_spatial_consistency: bool = False,
) -> (
    tuple[torch.Tensor, torch.Tensor]
    | tuple[torch.Tensor, torch.Tensor, LargeScaleDetails]
):
    """Sample path gain values for all BS-UT pairs."""

    sionna_config.device = device
    sionna_config.precision = precision
    sionna_config.seed = seed

    if cfg.scenario_lower == "inh":
        scenario, sampler = _make_scenario_and_sampler(
            cfg,
            topology,
            "low",
            topology.in_state,
            None,
            precision,
            device,
            spec_version,
            enable_spatial_consistency,
        )
        lsp = sampler()
        sf_db = 10.0 * torch.log10(lsp.sf)
        if lsp.pathloss is None:
            raise RuntimeError("Path loss was not sampled with the LSP realization")
        o2i_db = lsp.pathloss - scenario.basic_pathloss
        path_gain_db = -lsp.pathloss + sf_db
        high_loss_ut = torch.zeros_like(topology.in_state)

        if include_antenna_gain:
            bs_yaw = topology.bs_orientations[..., 0]
            path_gain_db = path_gain_db + _phase1_bs_gain_db(
                scenario.los_zod,
                scenario.los_aod,
                bs_yaw,
                cfg,
            )

        if return_details:
            return (
                path_gain_db,
                high_loss_ut,
                LargeScaleDetails(
                    outdoor_los=scenario.los.clone(),
                    basic_outdoor_pathloss_db=scenario.basic_pathloss,
                    shadow_fading_db=sf_db,
                    o2i_loss_db=o2i_db,
                    lsp=lsp,
                ),
            )

        return path_gain_db, high_loss_ut

    scenario_low, sampler_low = _make_scenario_and_sampler(
        cfg,
        topology,
        "low",
        topology.in_state,
        None,
        precision,
        device,
        spec_version,
        enable_spatial_consistency,
    )
    outdoor_los = scenario_low.outdoor_los.clone()
    scenario_high, sampler_high = _make_scenario_and_sampler(
        cfg,
        topology,
        "high",
        topology.in_state,
        outdoor_los,
        precision,
        device,
        spec_version,
        enable_spatial_consistency,
    )

    lsp = sampler_low()
    if lsp.pathloss is None:
        raise RuntimeError("Path loss was not sampled with the LSP realization")
    high_pathloss_db = sampler_high.sample_pathloss()
    sf_db = 10.0 * torch.log10(lsp.sf)

    generator = torch.Generator(device=device)
    generator.manual_seed(seed + 1543)
    high_loss_ut = (
        torch.rand(
            topology.in_state.shape,
            device=device,
            dtype=topology.ut_loc.dtype,
            generator=generator,
        )
        < 0.5
    ) & topology.in_state
    high_loss_link = high_loss_ut.unsqueeze(1).expand_as(lsp.pathloss)

    pathloss_db = torch.where(high_loss_link, high_pathloss_db, lsp.pathloss)
    lsp.pathloss = pathloss_db
    o2i_db = pathloss_db - scenario_low.basic_pathloss
    path_gain_db = -pathloss_db + sf_db

    if include_antenna_gain:
        bs_yaw = topology.bs_orientations[..., 0]
        path_gain_db = path_gain_db + _phase1_bs_gain_db(
            scenario_low.los_zod,
            scenario_low.los_aod,
            bs_yaw,
            cfg,
        )

    if return_details:
        return (
            path_gain_db,
            high_loss_ut,
            LargeScaleDetails(
                outdoor_los=outdoor_los,
                basic_outdoor_pathloss_db=scenario_low.basic_pathloss,
                shadow_fading_db=sf_db,
                o2i_loss_db=o2i_db,
                lsp=lsp,
            ),
        )

    return path_gain_db, high_loss_ut


def _run_phase1_once(
    cfg: Phase1Config,
    batch_size: int,
    num_ut_per_sector: int,
    seed: int,
    precision: str,
    device: str,
    spec_version: str,
) -> CalibrationResult:
    """Run one Phase 1 drop and compute calibration metrics."""

    topology = _make_topology(
        cfg,
        batch_size=batch_size,
        num_ut_per_sector=num_ut_per_sector,
        seed=seed,
        precision=precision,
        device=device,
    )
    path_gain_all, high_loss_ut = _sample_path_gain_all_bs(
        cfg,
        topology,
        seed=seed + 1009,
        precision=precision,
        device=device,
        spec_version=spec_version,
        include_antenna_gain=True,
    )

    flat_gain = path_gain_all.permute(0, 2, 1).reshape(-1, path_gain_all.shape[1])
    coupling_loss, serving, serving_loss, serving_gain = _serving_link_values(flat_gain)
    sir = geometry_sir_db(coupling_loss, serving=serving)
    sinr = geometry_sinr_db(
        coupling_loss,
        tx_power_dbm=cfg.tx_power_dbm,
        bandwidth_hz=cfg.bandwidth_hz,
        noise_figure_db=cfg.noise_figure_db,
        serving=serving,
    )

    return CalibrationResult(
        scenario=cfg.scenario,
        fc_ghz=cfg.fc_ghz,
        phase="phase1",
        metrics={
            "coupling_loss": serving_loss.detach().cpu().numpy(),
            "historical_path_gain": serving_gain.detach().cpu().numpy(),
            "geometry_sinr": sinr.detach().cpu().numpy(),
            "geometry_sir": sir.detach().cpu().numpy(),
        },
        serving_bs=serving.detach().cpu().numpy(),
        in_state=topology.in_state.detach().cpu().numpy().reshape(-1),
        o2i_is_high_loss=high_loss_ut.detach().cpu().numpy().reshape(-1),
    )


def _concat_results(results: Sequence[CalibrationResult]) -> CalibrationResult:
    """Concatenate metric samples from multiple independent drops."""

    first = results[0]
    return CalibrationResult(
        scenario=first.scenario,
        fc_ghz=first.fc_ghz,
        phase=first.phase,
        metrics={
            key: np.concatenate([result.metrics[key] for result in results])
            for key in first.metrics
        },
        serving_bs=np.concatenate([result.serving_bs for result in results]),
        in_state=np.concatenate([result.in_state for result in results]),
        o2i_is_high_loss=np.concatenate(
            [result.o2i_is_high_loss for result in results]
        ),
        metadata=first.metadata,
    )


def _with_workload_metadata(
    result: CalibrationResult,
    **workload: int,
) -> CalibrationResult:
    """Return ``result`` with explicit per-run workload provenance."""

    metadata = dict(result.metadata or {})
    metadata.update({key: int(value) for key, value in workload.items()})
    return replace(result, metadata=metadata)


def _run_phase1_batches(
    cfg: Phase1Config,
    num_batches: int,
    batch_size: int,
    num_ut_per_sector: int,
    seed: int,
    precision: str,
    devices: Sequence[str],
    spec_version: str,
) -> CalibrationResult:
    """Run multiple independent Phase 1 drops."""

    results = []
    for batch_idx in range(num_batches):
        device = devices[batch_idx % len(devices)]
        results.append(
            _run_phase1_once(
                cfg,
                batch_size=batch_size,
                num_ut_per_sector=num_ut_per_sector,
                seed=seed + batch_idx,
                precision=precision,
                device=device,
                spec_version=spec_version,
            )
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return _with_workload_metadata(
        _concat_results(results),
        num_batches=num_batches,
        batch_size=batch_size,
        num_ut_per_sector=num_ut_per_sector,
        effective_num_topology_drops=num_batches * batch_size,
    )


def _build_phase2_arrays(
    cfg: Phase2Config,
    precision: str,
    device: str,
) -> tuple[PanelArray, PanelArray]:
    """Build UT and BS arrays from TR 38.901 V16.1 Table 7.8-2."""

    fc_hz = cfg.fc_ghz * 1e9
    if cfg.config_id == 1:
        bs_array = PanelArray(
            num_rows_per_panel=4,
            num_cols_per_panel=4,
            polarization="dual",
            polarization_type="cross",
            antenna_pattern="38.901",
            carrier_frequency=fc_hz,
            num_rows=1,
            num_cols=2,
            panel_vertical_spacing=2.5,
            panel_horizontal_spacing=2.5,
            element_vertical_spacing=0.5,
            element_horizontal_spacing=0.5,
            precision=precision,
            device=device,
        )
    elif cfg.config_id == 2:
        bs_array = PanelArray(
            num_rows_per_panel=2,
            num_cols_per_panel=2,
            polarization="single",
            polarization_type="V",
            antenna_pattern="38.901",
            carrier_frequency=fc_hz,
            element_vertical_spacing=0.5,
            element_horizontal_spacing=0.5,
            precision=precision,
            device=device,
        )
    else:
        raise ValueError(f"Unsupported Phase 2 config_id: {cfg.config_id}")

    ut_array = PanelArray(
        num_rows_per_panel=1,
        num_cols_per_panel=1,
        polarization="dual",
        polarization_type="VH",
        antenna_pattern="omni",
        carrier_frequency=fc_hz,
        precision=precision,
        device=device,
    )
    return ut_array, bs_array


def _complex_dtype(precision: str) -> torch.dtype:
    """Return the complex dtype for the selected Sionna precision."""

    return torch.complex128 if precision == "double" else torch.complex64


def _steering_weights(
    ant_pos_m: torch.Tensor,
    carrier_frequency_hz: float,
    theta_deg: float,
    phi_deg: float,
) -> torch.Tensor:
    """Return conjugate array-response weights for a local steering direction."""

    dtype = ant_pos_m.dtype
    device = ant_pos_m.device
    theta = torch.deg2rad(torch.tensor(theta_deg, dtype=dtype, device=device))
    phi = torch.deg2rad(torch.tensor(phi_deg, dtype=dtype, device=device))
    direction = torch.stack(
        [
            torch.sin(theta) * torch.cos(phi),
            torch.sin(theta) * torch.sin(phi),
            torch.cos(theta),
        ]
    )
    wavelength = torch.tensor(
        SPEED_OF_LIGHT / carrier_frequency_hz, dtype=dtype, device=device
    )
    phase = -2.0 * torch.pi * (ant_pos_m / wavelength * direction).sum(dim=-1)
    return torch.exp(torch.complex(torch.zeros_like(phase), phase))


def _phase2_port_mapping(
    bs_array: PanelArray,
    cfg: Phase2Config,
    precision: str,
) -> torch.Tensor:
    """Return CRS-port to BS-element mapping weights."""

    cdtype = _complex_dtype(precision)
    device = bs_array.ant_pos.device

    if cfg.config_id == 2:
        return torch.eye(bs_array.num_ant, dtype=cdtype, device=device)

    num_elements_per_pol_panel = (
        bs_array.num_rows_per_panel * bs_array.num_cols_per_panel
    )
    weights = torch.zeros((4, bs_array.num_ant), dtype=cdtype, device=device)
    steering = _steering_weights(
        bs_array.ant_pos,
        cfg.fc_ghz * 1e9,
        theta_deg=cfg.bs_electrical_downtilt_deg,
        phi_deg=0.0,
    ).to(cdtype)

    for panel_idx in range(2):
        base = panel_idx * 2 * num_elements_per_pol_panel
        groups = (
            torch.arange(base, base + num_elements_per_pol_panel, device=device),
            torch.arange(
                base + num_elements_per_pol_panel,
                base + 2 * num_elements_per_pol_panel,
                device=device,
            ),
        )
        for pol_idx, group in enumerate(groups):
            port = panel_idx * 2 + pol_idx
            weights[port, group] = steering[group] / np.sqrt(num_elements_per_pol_panel)

    return weights


def _set_phase2_ut_orientations(topology: TopologyBundle, seed: int) -> None:
    """Apply the Phase 2 UT orientation distribution in-place."""

    generator = torch.Generator(device=topology.ut_loc.device)
    generator.manual_seed(seed + 3557)
    alpha = (
        2.0
        * torch.pi
        * torch.rand(
            topology.ut_orientations[..., 0].shape,
            dtype=topology.ut_orientations.dtype,
            device=topology.ut_orientations.device,
            generator=generator,
        )
    )
    topology.ut_orientations[..., 0] = alpha
    topology.ut_orientations[..., 1] = torch.pi / 2.0
    topology.ut_orientations[..., 2] = 0.0


def _make_phase2_channel(
    cfg: Phase2Config,
    ut_array: PanelArray,
    bs_array: PanelArray,
    precision: str,
    device: str,
    spec_version: str,
    enable_blockage: bool = False,
    blockage_self_blocking: str | None = None,
):
    """Create a Sionna full-channel model for Phase 2."""

    if cfg.scenario_lower == "inh":
        channel = InH(
            carrier_frequency=cfg.fc_ghz * 1e9,
            indoor_scenario="open",
            ut_array=ut_array,
            bs_array=bs_array,
            direction="downlink",
            enable_pathloss=False,
            enable_shadow_fading=False,
            always_generate_lsp=False,
            enable_blockage=enable_blockage,
            blockage_self_blocking=blockage_self_blocking,
            precision=precision,
            device=device,
            spec_version=spec_version,
        )
    else:
        channel_cls = UMi if cfg.scenario_lower == "umi" else UMa
        channel = channel_cls(
            carrier_frequency=cfg.fc_ghz * 1e9,
            o2i_model="low",
            ut_array=ut_array,
            bs_array=bs_array,
            direction="downlink",
            enable_pathloss=False,
            enable_shadow_fading=False,
            always_generate_lsp=False,
            enable_blockage=enable_blockage,
            blockage_self_blocking=blockage_self_blocking,
            precision=precision,
            device=device,
            spec_version=spec_version,
        )
    channel.return_rays = True
    return channel


def _port_channels(a: torch.Tensor, port_weights: torch.Tensor) -> torch.Tensor:
    """Map antenna-domain CIRs to CRS-port CIRs."""

    return torch.einsum("burstln,qt->bursqln", a, port_weights.to(dtype=a.dtype))


def _apply_large_scale(
    a: torch.Tensor,
    path_gain_no_ant_db: torch.Tensor,
) -> torch.Tensor:
    """Apply externally sampled path gain to fast-fading CIRs."""

    path_gain_rx_tx = path_gain_no_ant_db.permute(0, 2, 1)
    gain = torch.pow(
        torch.tensor(10.0, dtype=path_gain_rx_tx.dtype, device=path_gain_rx_tx.device),
        path_gain_rx_tx / 20.0,
    )
    return a * gain[:, :, None, :, None, None, None].to(dtype=a.dtype)


def _slice_lsp_by_ut(lsp: LSP, start: int, stop: int) -> LSP:
    """Return a UT slice while preserving one full-topology LSP realization."""

    return LSP(
        ds=lsp.ds[:, :, start:stop],
        asd=lsp.asd[:, :, start:stop],
        asa=lsp.asa[:, :, start:stop],
        sf=lsp.sf[:, :, start:stop],
        k_factor=lsp.k_factor[:, :, start:stop],
        zsa=lsp.zsa[:, :, start:stop],
        zsd=lsp.zsd[:, :, start:stop],
        pathloss=(None if lsp.pathloss is None else lsp.pathloss[:, :, start:stop]),
    )


def _serving_link_values(
    flat_path_gain_db: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return all-link loss, serving index, serving loss, and serving gain."""

    coupling_loss_all = -flat_path_gain_db
    serving = serving_indices(coupling_loss_all)
    serving_coupling_loss = torch.take_along_dim(
        coupling_loss_all, serving.unsqueeze(-1), dim=-1
    ).squeeze(-1)
    serving_path_gain = torch.take_along_dim(
        flat_path_gain_db, serving.unsqueeze(-1), dim=-1
    ).squeeze(-1)
    return coupling_loss_all, serving, serving_coupling_loss, serving_path_gain


def _spatial_wideband_metrics(
    coupling_loss: torch.Tensor,
    cfg: Phase2Config,
    serving: torch.Tensor,
    include_historical_sir: bool,
) -> dict[str, torch.Tensor]:
    """Compute normative SINR and, when requested, historical SIR."""

    metrics = {
        "wideband_sinr": geometry_sinr_db(
            coupling_loss,
            tx_power_dbm=cfg.tx_power_dbm,
            bandwidth_hz=cfg.bandwidth_hz,
            noise_figure_db=cfg.noise_figure_db,
            serving=serving,
        )
    }
    if include_historical_sir:
        metrics["historical_wideband_sir"] = wideband_sir_db(
            coupling_loss, serving=serving
        )
    return metrics


def _linear_to_db(value: torch.Tensor) -> torch.Tensor:
    """Convert linear values to dB."""

    floor = torch.finfo(value.real.dtype).tiny
    return 10.0 * torch.log10(torch.clamp(value.real, min=floor))


def _port0_path_gain_db(h_port_scaled: torch.Tensor) -> torch.Tensor:
    """Compute CRS-port-0 path gain for all UT/BS links."""

    h0 = h_port_scaled[:, :, :, :, 0, :, :]
    power = torch.sum(torch.abs(h0) ** 2, dim=(2, 4, 5))
    power = power / (h0.shape[2] * h0.shape[-1])
    return _linear_to_db(power)


def _raw_cluster_delay_and_aoa(
    rays: Rays,
    sort_indices: torch.Tensor,
    raw_cluster: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return delay and first-ray AOA of a raw cluster after delay sorting."""

    rank = torch.argmax((sort_indices == raw_cluster).to(torch.int64), dim=3)
    delay = torch.gather(rays.delays, dim=3, index=rank.unsqueeze(3)).squeeze(3)
    aoa_index = rank.reshape(*rank.shape, 1, 1).expand(
        *rank.shape, 1, rays.aoa.shape[-1]
    )
    aoa = torch.gather(rays.aoa, dim=3, index=aoa_index).squeeze(3)[..., 0]
    aoa = (aoa + torch.pi) % (2.0 * torch.pi) - torch.pi
    return delay, aoa


def _set_spatial_metric1_2_ut_orientations(
    ut_orientations: torch.Tensor,
    seed: int,
) -> None:
    """Apply the TR 38.901 Phase-2 UT orientation distribution in-place."""

    generator = torch.Generator(device=ut_orientations.device)
    generator.manual_seed(seed + 3557)
    ut_orientations[..., 0] = (
        2.0
        * torch.pi
        * torch.rand(
            ut_orientations[..., 0].shape,
            dtype=ut_orientations.dtype,
            device=ut_orientations.device,
            generator=generator,
        )
    )
    ut_orientations[..., 1] = torch.pi / 2.0
    ut_orientations[..., 2] = 0.0


def _set_spatial_metric1_2_ut_velocities(
    ut_velocities: torch.Tensor,
    seed: int,
) -> None:
    """Apply the Table 7.8-5 Config2 random movement directions."""

    generator = torch.Generator(device=ut_velocities.device)
    generator.manual_seed(seed + 4567)
    speed_mps = torch.tensor(
        30.0 / 3.6,
        dtype=ut_velocities.dtype,
        device=ut_velocities.device,
    )
    direction = (
        2.0
        * torch.pi
        * torch.rand(
            ut_velocities[..., 0].shape,
            dtype=ut_velocities.dtype,
            device=ut_velocities.device,
            generator=generator,
        )
    )
    ut_velocities[..., 0] = speed_mps * torch.cos(direction)
    ut_velocities[..., 1] = speed_mps * torch.sin(direction)
    ut_velocities[..., 2] = 0.0


def _spatial_metric1_2_topology(
    num_ut_per_sector: int,
    carrier_frequency: float,
    precision: str,
    device: str,
    seed: int,
    indoor: bool,
    num_rings: int,
) -> tuple[torch.Tensor, ...]:
    """Generate the UMi topology used for spatial-consistency metric 1/2."""

    sionna_config.seed = seed
    topology = gen_tr38901_multicell_topology(
        "umi",
        batch_size=1,
        num_ut_per_sector=num_ut_per_sector,
        carrier_frequency=carrier_frequency,
        num_rings=num_rings,
        isd=200.0,
        bs_height=10.0,
        min_bs_ut_dist=0.0,
        indoor_probability=1.0 if indoor else 0.0,
        apply_tr36873_indoor_heights=False,
        enforce_indoor_distance=False,
        precision=precision,
        device=device,
    )
    (
        ut_loc,
        bs_loc,
        ut_orientations,
        bs_orientations,
        ut_velocities,
        in_state,
        los,
        bs_virtual_loc,
        bs_site_ids,
        _spatial_consistency_track_ids,
        distance_2d_in,
    ) = topology
    _set_spatial_metric1_2_ut_orientations(ut_orientations, seed)
    if not indoor:
        _set_spatial_metric1_2_ut_velocities(ut_velocities, seed)
    return (
        ut_loc,
        bs_loc,
        ut_orientations,
        bs_orientations,
        ut_velocities,
        in_state,
        los,
        bs_virtual_loc,
        bs_site_ids,
        distance_2d_in,
    )


def _make_spatial_metric1_2_channel(
    carrier_frequency: float,
    precision: str,
    device: str,
    spec_version: str,
):
    """Create the Phase-2 Config-1 UMi channel used by metric 1/2."""

    cfg = _get_phase2_config("UMi", carrier_frequency / 1e9, 1)
    ut_array, bs_array = _build_phase2_arrays(cfg, precision, device)
    channel = UMi(
        carrier_frequency=carrier_frequency,
        o2i_model="low",
        ut_array=ut_array,
        bs_array=bs_array,
        direction="downlink",
        enable_pathloss=False,
        enable_shadow_fading=False,
        always_generate_lsp=False,
        enable_spatial_consistency=True,
        precision=precision,
        device=device,
        spec_version=spec_version,
    )
    port_weights = _phase2_port_mapping(bs_array, cfg, precision)
    return channel, cfg, port_weights


def _run_spatial_metric1_2_drop(
    num_ut_per_sector: int,
    ut_chunk_size: int,
    carrier_frequency: float,
    precision: str,
    device: str,
    seed: int,
    spec_version: str,
    indoor: bool,
    include_historical_sir: bool,
    num_rings: int,
    retain_central_site_only: bool,
    retain_dropped_central_site_only: bool,
) -> dict[str, torch.Tensor]:
    """Run one TR 38.901 Table 7.8-5 metric-1/2 drop."""

    sionna_config.device = device
    sionna_config.precision = precision
    sionna_config.seed = seed

    if ut_chunk_size <= 0:
        raise ValueError("--spatial-metric1-2-ut-chunk-size must be positive")

    topology = _spatial_metric1_2_topology(
        num_ut_per_sector,
        carrier_frequency,
        precision,
        device,
        seed,
        indoor=indoor,
        num_rings=num_rings,
    )
    (
        ut_loc,
        bs_loc,
        ut_orientations,
        bs_orientations,
        ut_velocities,
        in_state,
        los,
        bs_virtual_loc,
        bs_site_ids,
        distance_2d_in,
    ) = topology
    if retain_dropped_central_site_only:
        num_central_ut = 3 * num_ut_per_sector
        ut_loc = ut_loc[:, :num_central_ut, :]
        ut_orientations = ut_orientations[:, :num_central_ut, :]
        ut_velocities = ut_velocities[:, :num_central_ut, :]
        in_state = in_state[:, :num_central_ut]
        distance_2d_in = distance_2d_in[..., :num_central_ut]
        bs_virtual_loc = bs_virtual_loc[:, :, :num_central_ut, :]

    num_ut = ut_loc.shape[1]
    phase2_cfg = _get_phase2_config("UMi", carrier_frequency / 1e9, 1)
    topology_bundle = TopologyBundle(
        ut_loc=ut_loc,
        bs_loc=bs_loc,
        ut_orientations=ut_orientations,
        bs_orientations=bs_orientations,
        ut_velocities=ut_velocities,
        in_state=in_state,
        distance_2d_in=distance_2d_in,
        los=None,
        bs_virtual_loc=bs_virtual_loc,
        bs_site_ids=bs_site_ids,
        site_positions=torch.empty(0, 2, dtype=ut_loc.dtype, device=ut_loc.device),
    )
    path_gain_no_ant, _high_loss_ut, details = _sample_path_gain_all_bs(
        phase2_cfg,
        topology_bundle,
        seed=seed + 1009,
        precision=precision,
        device=device,
        spec_version=spec_version,
        include_antenna_gain=False,
        return_details=True,
        enable_spatial_consistency=True,
    )

    metric_chunks: dict[str, list[torch.Tensor]] = {
        "coupling_loss": [],
        "historical_path_gain": [],
        "wideband_sinr": [],
    }
    if include_historical_sir:
        metric_chunks["historical_wideband_sir"] = []
    attached = 0
    for start in range(0, num_ut, ut_chunk_size):
        stop = min(start + ut_chunk_size, num_ut)
        chunk_topology = (
            ut_loc[:, start:stop, :],
            bs_loc,
            ut_orientations[:, start:stop, :],
            bs_orientations,
            ut_velocities[:, start:stop, :],
            in_state[:, start:stop],
            los,
            bs_virtual_loc[:, :, start:stop, :],
            bs_site_ids,
            distance_2d_in[..., start:stop],
        )
        channel, phase2_cfg, port_weights = _make_spatial_metric1_2_channel(
            carrier_frequency, precision, device, spec_version
        )
        channel.set_topology(
            chunk_topology[0],
            chunk_topology[1],
            chunk_topology[2],
            chunk_topology[3],
            chunk_topology[4],
            chunk_topology[5],
            los=details.outdoor_los[:, :, start:stop],
            bs_virtual_loc=chunk_topology[7],
            bs_site_ids=chunk_topology[8],
            distance_2d_in=chunk_topology[9],
        )
        channel._lsp = _slice_lsp_by_ut(details.lsp, start, stop)
        a, _ = channel(
            num_time_samples=1,
            sampling_frequency=phase2_cfg.bandwidth_hz,
        )
        a = _apply_large_scale(a, path_gain_no_ant[:, :, start:stop])
        h_port = _port_channels(a, port_weights)
        path_gain_all = _port0_path_gain_db(h_port)

        flat_gain = path_gain_all.reshape(-1, path_gain_all.shape[-1])
        coupling_loss, serving, serving_loss, serving_gain = _serving_link_values(
            flat_gain
        )
        interference_metrics = _spatial_wideband_metrics(
            coupling_loss,
            phase2_cfg,
            serving,
            include_historical_sir,
        )

        chunk_ut = stop - start
        batch_index = torch.arange(
            ut_loc.shape[0], device=ut_loc.device
        ).repeat_interleave(chunk_ut)
        ut_index = torch.arange(chunk_ut, device=ut_loc.device).repeat(ut_loc.shape[0])
        serving_virtual_xy = chunk_topology[7][batch_index, serving, ut_index, :2]
        if retain_central_site_only:
            sample_mask = torch.linalg.norm(serving_virtual_xy, dim=-1) < 1e-4
        else:
            sample_mask = torch.ones_like(serving, dtype=torch.bool)

        metric_chunks["coupling_loss"].append(serving_loss[sample_mask].detach().cpu())
        metric_chunks["historical_path_gain"].append(
            serving_gain[sample_mask].detach().cpu()
        )
        for key, values in interference_metrics.items():
            metric_chunks[key].append(values[sample_mask].detach().cpu())
        attached += int(sample_mask.sum().detach().cpu())
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return {
        **{key: torch.cat(chunks) for key, chunks in metric_chunks.items()},
        "num_attached_to_central_site": torch.tensor(attached),
        "num_ut": torch.tensor(num_ut),
    }


def _run_spatial_metric1_2_common(
    num_ut_per_sector: int,
    num_drops: int,
    ut_chunk_size: int,
    carrier_frequency: float,
    precision: str,
    devices: Sequence[str],
    seed: int,
    spec_version: str,
    phase: str,
    indoor: bool,
    include_historical_sir: bool,
    num_rings: int,
    retain_central_site_only: bool,
    retain_dropped_central_site_only: bool,
    metadata_note: str,
) -> CalibrationResult:
    """Run the TR 38.901 Table 7.8-5 metric-1/2 CDF calibration."""

    if not math.isclose(carrier_frequency, 30e9):
        raise ValueError("The spatial-consistency metric-1/2 reference is for 30 GHz.")
    if num_ut_per_sector <= 0:
        raise ValueError("--spatial-metric1-2-num-ut-per-sector must be positive")
    if num_drops <= 0:
        raise ValueError("--spatial-metric1-2-num-drops must be positive")

    metric_samples: dict[str, list[np.ndarray]] = {
        "coupling_loss": [],
        "historical_path_gain": [],
        "wideband_sinr": [],
    }
    if include_historical_sir:
        metric_samples["historical_wideband_sir"] = []
    attached = 0
    total_ut = 0
    for drop_idx in range(num_drops):
        device = devices[drop_idx % len(devices)]
        result = _run_spatial_metric1_2_drop(
            num_ut_per_sector,
            ut_chunk_size,
            carrier_frequency,
            precision,
            device,
            seed + 1009 * (drop_idx + 1),
            spec_version,
            indoor=indoor,
            include_historical_sir=include_historical_sir,
            num_rings=num_rings,
            retain_central_site_only=retain_central_site_only,
            retain_dropped_central_site_only=retain_dropped_central_site_only,
        )
        for key in metric_samples:
            metric_samples[key].append(result[key].detach().cpu().numpy())
        attached += int(result["num_attached_to_central_site"].detach().cpu())
        total_ut += int(result["num_ut"].detach().cpu())
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    metrics = {key: np.concatenate(samples) for key, samples in metric_samples.items()}
    coupling_loss = metrics["coupling_loss"]
    return CalibrationResult(
        scenario="UMi",
        fc_ghz=30.0,
        phase=phase,
        metrics=metrics,
        serving_bs=np.zeros(coupling_loss.shape, dtype=np.int64),
        in_state=np.full(coupling_loss.shape, indoor, dtype=bool),
        o2i_is_high_loss=np.zeros(coupling_loss.shape, dtype=bool),
        metadata={
            "num_drops": int(num_drops),
            "num_ut_per_sector": int(num_ut_per_sector),
            "ut_chunk_size": int(ut_chunk_size),
            "num_attached_to_central_site": int(attached),
            "num_ut_total": int(total_ut),
            "high_loss_fraction_among_indoor": None,
            "serving_lsp_realization": "shared_correlated_full_topology",
            "note": metadata_note,
        },
    )


def _run_spatial_metric1_2(
    num_ut_per_sector: int,
    num_drops: int,
    ut_chunk_size: int,
    carrier_frequency: float,
    precision: str,
    devices: Sequence[str],
    seed: int,
    spec_version: str,
) -> CalibrationResult:
    """Run the TR 38.901 Table 7.8-5 Config1 metric-1/2 CDF calibration."""

    return _run_spatial_metric1_2_common(
        num_ut_per_sector,
        num_drops,
        ut_chunk_size,
        carrier_frequency,
        precision,
        devices,
        seed,
        spec_version,
        phase="spatial_consistency_metric1_2",
        indoor=True,
        include_historical_sir=False,
        num_rings=1,
        retain_central_site_only=True,
        retain_dropped_central_site_only=False,
        metadata_note=(
            "TR 38.901 Table 7.8-5 Config1 metric 1/2: all UTs are indoor, "
            "spatial consistency is enabled, CRS port 0 is formed from the "
            "Phase-2 Config-1 port mapping, and samples are retained for UTs "
            "served by the central wrapped site."
        ),
    )


def _run_spatial_config2_metric1_2(
    num_ut_per_sector: int,
    num_drops: int,
    ut_chunk_size: int,
    carrier_frequency: float,
    precision: str,
    devices: Sequence[str],
    seed: int,
    spec_version: str,
) -> CalibrationResult:
    """Run the TR 38.901 Table 7.8-5 Config2 static CDF calibration."""

    return _run_spatial_metric1_2_common(
        num_ut_per_sector,
        num_drops,
        ut_chunk_size,
        carrier_frequency,
        precision,
        devices,
        seed,
        spec_version,
        phase="spatial_consistency_config2_metric1_2",
        indoor=False,
        include_historical_sir=True,
        num_rings=1,
        retain_central_site_only=False,
        retain_dropped_central_site_only=True,
        metadata_note=(
            "TR 38.901 Table 7.8-5 Config2 metric 1/2: all UTs are outdoor, "
            "spatial consistency is enabled, CRS port 0 is formed from the "
            "Phase-2 Config-1 port mapping, and samples are retained for the "
            "central-cell dropped UTs in a wrapped network drop. The dynamic "
            "Config2 varying-rate metrics are intentionally not generated. "
            "wideband_sinr includes thermal noise as required by Config2; "
            "the contribution-era interference-only curve is retained only "
            "as historical_wideband_sir."
        ),
    )


def _spatial_metric3_6_base_topology(
    num_ut_per_sector: int,
    carrier_frequency: float,
    precision: str,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate the one-site UMi UT drop used by metric 3/6."""

    topology = gen_tr38901_multicell_topology(
        "umi",
        batch_size=1,
        num_ut_per_sector=num_ut_per_sector,
        carrier_frequency=carrier_frequency,
        num_rings=0,
        isd=200.0,
        bs_height=10.0,
        min_bs_ut_dist=0.0,
        indoor_probability=0.0,
        apply_tr36873_indoor_heights=False,
        precision=precision,
        device=device,
    )
    (
        ut_loc,
        bs_loc,
        _ut_orientations,
        bs_orientations,
        _ut_velocities,
        _in_state,
        _los,
        _bs_virtual_loc,
        _bs_site_ids,
        _spatial_consistency_track_ids,
        _distance_2d_in,
    ) = topology
    return ut_loc[0], bs_loc[:, :1, :], bs_orientations[:, :1, :]


def _spatial_metric3_6_drop_topology(
    base_ut_loc: torch.Tensor,
    bs_loc: torch.Tensor,
    bs_orientations: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """Build a one-site, one-drop topology for metric 3/6."""

    dtype = base_ut_loc.dtype
    device = base_ut_loc.device
    num_ut = base_ut_loc.shape[0]
    batch_size = 1
    ut_loc = base_ut_loc.reshape(1, num_ut, 3).clone()
    ut_orientations = torch.zeros(batch_size, num_ut, 3, dtype=dtype, device=device)
    ut_velocities = torch.zeros(batch_size, num_ut, 3, dtype=dtype, device=device)
    in_state = torch.ones(batch_size, num_ut, dtype=torch.bool, device=device)
    bs_virtual_loc = bs_loc.unsqueeze(2).expand(-1, -1, num_ut, -1).clone()
    bs_site_ids = torch.zeros(1, dtype=torch.int64, device=device)
    return (
        ut_loc,
        bs_loc,
        ut_orientations,
        bs_orientations,
        ut_velocities,
        in_state,
        None,
        bs_virtual_loc,
        bs_site_ids,
    )


def _make_spatial_metric3_6_channel(
    carrier_frequency: float,
    precision: str,
    device: str,
    spec_version: str,
):
    """Create the Phase-2 Config-1 UMi channel used by metric 3/6."""

    cfg = _get_phase2_config("UMi", carrier_frequency / 1e9, 1)
    ut_array, bs_array = _build_phase2_arrays(cfg, precision, device)
    channel = UMi(
        carrier_frequency=carrier_frequency,
        o2i_model="low",
        ut_array=ut_array,
        bs_array=bs_array,
        direction="downlink",
        enable_pathloss=False,
        enable_shadow_fading=False,
        always_generate_lsp=False,
        enable_spatial_consistency=True,
        precision=precision,
        device=device,
        spec_version=spec_version,
    )
    channel.return_rays = True
    port_weights = _phase2_port_mapping(bs_array, cfg, precision)
    return channel, cfg, port_weights


def _spatial_metric3_6_ordered_pair_moments(
    values: torch.Tensor,
    pair_indices: tuple[torch.Tensor, torch.Tensor],
    pair_mask: torch.Tensor,
    complex_values: bool = False,
) -> dict:
    """Raw moments for a Pearson coefficient over ordered UT pairs."""

    i, j = pair_indices
    i = i[pair_mask]
    j = j[pair_mask]
    x = torch.cat([values[i], values[j]], dim=0)
    y = torch.cat([values[j], values[i]], dim=0)
    if complex_values:
        return {
            "count": int(x.numel()),
            "sum_x": torch.sum(x).detach().cpu().item(),
            "sum_y": torch.sum(y).detach().cpu().item(),
            "sum_xx": torch.sum(torch.abs(x) ** 2).detach().cpu().item(),
            "sum_yy": torch.sum(torch.abs(y) ** 2).detach().cpu().item(),
            "sum_xy": torch.sum(x * torch.conj(y)).detach().cpu().item(),
        }
    return {
        "count": int(x.numel()),
        "sum_x": torch.sum(x).detach().cpu().item(),
        "sum_y": torch.sum(y).detach().cpu().item(),
        "sum_xx": torch.sum(x * x).detach().cpu().item(),
        "sum_yy": torch.sum(y * y).detach().cpu().item(),
        "sum_xy": torch.sum(x * y).detach().cpu().item(),
    }


def _channel_los_status(channel, dtype: torch.dtype) -> torch.Tensor:
    """Return the realized outdoor LOS state used by the channel model."""

    return channel._scenario.outdoor_los[0, 0].to(dtype=dtype)


def _corr_from_moments(moments: dict, complex_values: bool = False) -> float:
    """Compute a Pearson coefficient from accumulated raw moments."""

    count = moments["count"]
    if count == 0:
        return float("nan")
    if complex_values:
        mean_x = moments["sum_x"] / count
        mean_y = moments["sum_y"] / count
        cov = moments["sum_xy"] / count - mean_x * mean_y.conjugate()
        var_x = moments["sum_xx"] / count - abs(mean_x) ** 2
        var_y = moments["sum_yy"] / count - abs(mean_y) ** 2
        denom = math.sqrt(max(var_x.real, 1e-30) * max(var_y.real, 1e-30))
        return abs(cov / denom)
    mean_x = moments["sum_x"] / count
    mean_y = moments["sum_y"] / count
    cov = moments["sum_xy"] / count - mean_x * mean_y
    var_x = moments["sum_xx"] / count - mean_x * mean_x
    var_y = moments["sum_yy"] / count - mean_y * mean_y
    denom = math.sqrt(max(var_x, 1e-30) * max(var_y, 1e-30))
    return cov / denom


def _add_moments(accumulator: dict, idx: int, moments: dict) -> None:
    """Add raw pair moments to an accumulator."""

    for key, value in moments.items():
        accumulator[key][idx] += value


def _run_spatial_metric3_6_drop(
    distances_m: list[float],
    num_ut_per_sector: int,
    carrier_frequency: float,
    precision: str,
    device: str,
    seed: int,
    spec_version: str,
) -> tuple[list[dict[str, float]], int, int]:
    """Run one one-cell metric-3/6 drop."""

    sionna_config.device = device
    sionna_config.precision = precision
    sionna_config.seed = seed
    base_ut_loc, bs_loc, bs_orientations = _spatial_metric3_6_base_topology(
        num_ut_per_sector, carrier_frequency, precision, device
    )
    channel, phase2_cfg, port_weights = _make_spatial_metric3_6_channel(
        carrier_frequency, precision, device, spec_version
    )
    topology = _spatial_metric3_6_drop_topology(base_ut_loc, bs_loc, bs_orientations)
    channel.set_topology(*topology)

    a, tau, rays = channel(
        num_time_samples=1,
        sampling_frequency=phase2_cfg.bandwidth_hz,
    )
    scenario = channel._scenario
    third = min(2, scenario.num_clusters_max - 1)
    assert rays.cluster_sort_indices is not None
    delay, aoa = _raw_cluster_delay_and_aoa(rays, rays.cluster_sort_indices, third)
    delay = delay[0, 0]
    aoa = aoa[0, 0]

    port_h = _port_channels(a, port_weights)
    path_h = port_h[0, :, 0, 0, 0, :, 0]
    path_tau = tau[0, :, 0, :]
    frequency_hz = phase2_cfg.subcarrier_spacing_hz
    phase = torch.exp(
        torch.complex(
            torch.zeros_like(path_tau),
            -2.0 * torch.pi * path_tau * frequency_hz,
        )
    ).to(dtype=path_h.dtype)
    channel_response = torch.sum(path_h * phase, dim=-1)

    ut_loc = topology[0]
    xy = ut_loc[0, :, :2]
    pair_distance = torch.linalg.norm(xy.unsqueeze(0) - xy.unsqueeze(1), dim=-1)
    pair_indices = torch.triu_indices(
        xy.shape[0], xy.shape[0], offset=1, device=xy.device
    )
    pair_distances = pair_distance[pair_indices[0], pair_indices[1]]
    los_status = _channel_los_status(channel, base_ut_loc.dtype)

    drop_results = []
    for distance_m in distances_m:
        pair_mask = (pair_distances >= distance_m) & (pair_distances < distance_m + 1.0)
        num_pairs = int(pair_mask.sum().detach().cpu().item())
        if num_pairs == 0:
            drop_results.append(
                {
                    "num_pairs": 0,
                    "third_cluster_delay_corr": None,
                    "third_cluster_aoa_corr": None,
                    "los_state_corr": None,
                    "channel_response_corr": None,
                }
            )
            continue
        drop_results.append(
            {
                "num_pairs": num_pairs,
                "third_cluster_delay_corr": _spatial_metric3_6_ordered_pair_moments(
                    delay, pair_indices, pair_mask
                ),
                "third_cluster_aoa_corr": _spatial_metric3_6_ordered_pair_moments(
                    aoa, pair_indices, pair_mask
                ),
                "los_state_corr": _spatial_metric3_6_ordered_pair_moments(
                    los_status, pair_indices, pair_mask
                ),
                "channel_response_corr": _spatial_metric3_6_ordered_pair_moments(
                    channel_response,
                    pair_indices,
                    pair_mask,
                    complex_values=True,
                ),
            }
        )
    return drop_results, int(base_ut_loc.shape[0]), int(pair_indices.shape[1])


def _run_spatial_metric3_6(
    distances_m: list[float],
    num_ut_per_sector: int,
    num_drops: int,
    carrier_frequency: float,
    precision: str,
    devices: Sequence[str],
    seed: int,
    spec_version: str,
) -> LineCalibrationResult:
    """Run the TR 38.901 Table 7.8-5 metric-3/6 line calibration."""

    if not math.isclose(carrier_frequency, 30e9):
        raise ValueError("The spatial-consistency metric-3/6 reference is for 30 GHz.")
    if num_ut_per_sector <= 0:
        raise ValueError("--spatial-metric3-6-num-ut-per-sector must be positive")
    if num_drops <= 0:
        raise ValueError("--spatial-metric3-6-num-drops must be positive")

    metric_keys = (
        "third_cluster_delay_corr",
        "third_cluster_aoa_corr",
        "los_state_corr",
        "channel_response_corr",
    )
    moment_names = ("count", "sum_x", "sum_y", "sum_xx", "sum_yy", "sum_xy")
    moments = {
        key: {
            name: [
                0j
                if key == "channel_response_corr"
                and name in ("sum_x", "sum_y", "sum_xy")
                else 0.0
                for _ in distances_m
            ]
            for name in moment_names
        }
        for key in metric_keys
    }
    counts = [0 for _ in distances_m]
    num_ut = 0
    num_pairs_per_drop = 0
    for drop_idx in range(num_drops):
        device = devices[drop_idx % len(devices)]
        drop_results, num_ut, num_pairs_per_drop = _run_spatial_metric3_6_drop(
            distances_m,
            num_ut_per_sector,
            carrier_frequency,
            precision,
            device,
            seed + 1009 * (drop_idx + 1),
            spec_version,
        )
        for idx, result in enumerate(drop_results):
            if result["num_pairs"] == 0:
                continue
            counts[idx] += result["num_pairs"]
            for key in metric_keys:
                _add_moments(moments[key], idx, result[key])
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    curves = {
        key: np.asarray(
            [
                _corr_from_moments(
                    {name: moments[key][name][idx] for name in moment_names},
                    complex_values=key == "channel_response_corr",
                )
                for idx in range(len(distances_m))
            ],
            dtype=np.float64,
        )
        for key in metric_keys
    }
    return LineCalibrationResult(
        scenario="UMi",
        fc_ghz=30.0,
        phase="spatial_consistency_metric3_6",
        x_label="Distance [m]",
        x_values=np.asarray(distances_m, dtype=np.float64),
        metrics=curves,
        metadata={
            "num_drops": int(num_drops),
            "num_ut_per_sector": int(num_ut_per_sector),
            "num_ut": int(num_ut),
            "num_pairs_per_drop": int(num_pairs_per_drop),
            "num_pairs_per_bin": [int(v) for v in counts],
            "note": (
                "TR 38.901 Table 7.8-5 metric 3/6: all UTs are indoor, "
                "ordered UT pairs are binned by 2D distance intervals "
                "[d,d+1) m. Thus the point plotted at d=0 uses distinct UT "
                "pairs in [0,1) m, not self-correlation at exact co-location. "
                "Exact co-location is correlated by construction. Raw cluster "
                "3 is selected after delay sorting, and the channel-response "
                "curve uses Phase-2 Config-1 CRS port 0 at the first non-DC "
                "OFDM subcarrier."
            ),
        },
    )


def _phase2_spread_metrics(
    channel, rays, serving: torch.Tensor
) -> dict[str, np.ndarray]:
    """Compute Phase 2 Config 1 delay and angular-spread metrics."""

    ds_ns = delay_spread_from_rays(
        rays, channel._lsp, channel._scenario, serving=serving
    ).reshape(-1)
    spreads = angular_spreads_from_rays(
        rays, channel._lsp, channel._scenario, serving=serving
    )
    return {
        "delay_spread_ns": (ds_ns * 1e9).detach().cpu().numpy(),
        "asd_deg": torch.rad2deg(spreads["asd"].reshape(-1)).detach().cpu().numpy(),
        "zsd_deg": torch.rad2deg(spreads["zsd"].reshape(-1)).detach().cpu().numpy(),
        "asa_deg": torch.rad2deg(spreads["asa"].reshape(-1)).detach().cpu().numpy(),
        "zsa_deg": torch.rad2deg(spreads["zsa"].reshape(-1)).detach().cpu().numpy(),
    }


def _sample_inf_path_gain_all_bs(
    cfg: InFCalibrationConfig,
    topology: TopologyBundle,
    seed: int,
    precision: str,
    device: str,
    spec_version: str,
) -> tuple[torch.Tensor, LargeScaleDetails]:
    """Sample InF path gains for all BS-UT pairs."""

    sionna_config.device = device
    sionna_config.precision = precision
    sionna_config.seed = seed

    scenario, sampler = _make_inf_scenario_and_sampler(
        cfg, topology, precision, device, spec_version
    )
    lsp = sampler()
    sf_db = 10.0 * torch.log10(lsp.sf)
    if lsp.pathloss is None:
        raise RuntimeError("Path loss was not sampled with the LSP realization")
    path_gain_db = -lsp.pathloss + sf_db
    return (
        path_gain_db,
        LargeScaleDetails(
            outdoor_los=scenario.los.clone(),
            basic_outdoor_pathloss_db=scenario.basic_pathloss,
            shadow_fading_db=sf_db,
            o2i_loss_db=lsp.pathloss - scenario.basic_pathloss,
            lsp=lsp,
        ),
    )


def _run_inf_spread_metrics(
    cfg: InFCalibrationConfig,
    topology: TopologyBundle,
    los_state: torch.Tensor,
    lsp: LSP,
    serving: torch.Tensor,
    ut_chunk_size: int,
    seed: int,
    precision: str,
    device: str,
    spec_version: str,
) -> dict[str, np.ndarray]:
    """Compute InF delay and angular-spread metrics in UT chunks."""

    if ut_chunk_size <= 0:
        raise ValueError("--inf-ut-chunk-size must be positive")

    batch_size = topology.ut_loc.shape[0]
    num_ut = topology.ut_loc.shape[1]
    serving_by_ut = serving.reshape(batch_size, num_ut)
    metrics: dict[str, list[np.ndarray]] = {
        "delay_spread_ns": [],
        "asd_deg": [],
        "zsd_deg": [],
        "asa_deg": [],
        "zsa_deg": [],
    }

    for start in range(0, num_ut, ut_chunk_size):
        stop = min(start + ut_chunk_size, num_ut)
        sionna_config.seed = seed + start
        fc_hz = cfg.fc_ghz * 1e9
        ut_array, bs_array = _simple_arrays(fc_hz, precision, device)
        channel = InF(
            carrier_frequency=fc_hz,
            ut_array=ut_array,
            bs_array=bs_array,
            direction="downlink",
            factory_scenario=cfg.factory_scenario,
            hall_dimensions=_inf_calibration_hall_dimensions(cfg),
            enable_pathloss=False,
            enable_shadow_fading=False,
            always_generate_lsp=False,
            precision=precision,
            device=device,
            spec_version=spec_version,
        )
        channel.return_rays = True
        channel.set_topology(
            topology.ut_loc[:, start:stop, :],
            topology.bs_loc,
            topology.ut_orientations[:, start:stop, :],
            topology.bs_orientations,
            topology.ut_velocities[:, start:stop, :],
            topology.in_state[:, start:stop],
            los=los_state[:, :, start:stop],
            bs_virtual_loc=topology.bs_virtual_loc[:, :, start:stop, :],
            bs_site_ids=topology.bs_site_ids,
        )
        channel._lsp = _slice_lsp_by_ut(lsp, start, stop)
        _a, _tau, rays = channel(
            num_time_samples=1,
            sampling_frequency=cfg.bandwidth_hz,
        )
        chunk_metrics = _phase2_spread_metrics(
            channel,
            rays,
            serving_by_ut[:, start:stop].reshape(-1),
        )
        for key, values in chunk_metrics.items():
            metrics[key].append(values)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return {key: np.concatenate(values) for key, values in metrics.items()}


def _run_inf_once(
    cfg: InFCalibrationConfig,
    batch_size: int,
    num_ut_per_bs: int,
    ut_chunk_size: int,
    seed: int,
    precision: str,
    device: str,
    spec_version: str,
) -> CalibrationResult:
    """Run one indoor-factory calibration drop."""

    sionna_config.device = device
    sionna_config.precision = precision
    sionna_config.seed = seed

    topology = _make_inf_topology(
        cfg,
        batch_size=batch_size,
        num_ut_per_bs=num_ut_per_bs,
        precision=precision,
        device=device,
    )
    path_gain_all, details = _sample_inf_path_gain_all_bs(
        cfg,
        topology,
        seed=seed + 1009,
        precision=precision,
        device=device,
        spec_version=spec_version,
    )

    flat_gain = path_gain_all.permute(0, 2, 1).reshape(-1, path_gain_all.shape[1])
    coupling_loss, serving, serving_loss, serving_gain = _serving_link_values(flat_gain)
    sir = geometry_sir_db(coupling_loss, serving=serving)
    sinr = geometry_sinr_db(
        coupling_loss,
        tx_power_dbm=cfg.tx_power_dbm,
        bandwidth_hz=cfg.bandwidth_hz,
        noise_figure_db=cfg.noise_figure_db,
        serving=serving,
    )

    metrics = {
        "coupling_loss": serving_loss.detach().cpu().numpy(),
        "historical_path_gain": serving_gain.detach().cpu().numpy(),
        "geometry_sinr": sinr.detach().cpu().numpy(),
        "geometry_sir": sir.detach().cpu().numpy(),
    }
    metrics.update(
        _run_inf_spread_metrics(
            cfg,
            topology,
            details.outdoor_los,
            details.lsp,
            serving,
            ut_chunk_size,
            seed=seed + 2003,
            precision=precision,
            device=device,
            spec_version=spec_version,
        )
    )

    return CalibrationResult(
        scenario=cfg.scenario,
        fc_ghz=cfg.fc_ghz,
        phase=cfg.phase,
        metrics=metrics,
        serving_bs=serving.detach().cpu().numpy(),
        in_state=topology.in_state.detach().cpu().numpy().reshape(-1),
        o2i_is_high_loss=np.zeros(topology.in_state.numel(), dtype=bool),
        metadata={
            "factory_scenario": cfg.factory_scenario,
            "calibration_profile": cfg.profile,
            "hall_dimensions_m": list(_inf_calibration_hall_dimensions(cfg)),
            "num_bs": int(topology.bs_loc.shape[1]),
            "num_ut_per_bs": int(num_ut_per_bs),
            "ut_chunk_size": int(ut_chunk_size),
            "tx_power_dbm": float(cfg.tx_power_dbm),
            "bandwidth_hz": float(cfg.bandwidth_hz),
            "noise_figure_db": float(cfg.noise_figure_db),
            "reference_geometry_compatible": False,
            "serving_lsp_realization": "shared_correlated_full_topology",
            "note": _inf_profile_note(),
        },
    )


def _run_inf_batches(
    cfg: InFCalibrationConfig,
    num_batches: int,
    batch_size: int,
    num_ut_per_bs: int,
    ut_chunk_size: int,
    seed: int,
    precision: str,
    devices: Sequence[str],
    spec_version: str,
) -> CalibrationResult:
    """Run multiple independent indoor-factory calibration drops."""

    if num_batches <= 0:
        raise ValueError("--inf-num-batches must be positive")

    results = []
    for batch_idx in range(num_batches):
        device = devices[batch_idx % len(devices)]
        results.append(
            _run_inf_once(
                cfg,
                batch_size=batch_size,
                num_ut_per_bs=num_ut_per_bs,
                ut_chunk_size=ut_chunk_size,
                seed=seed + batch_idx,
                precision=precision,
                device=device,
                spec_version=spec_version,
            )
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return _with_workload_metadata(
        _concat_results(results),
        num_batches=num_batches,
        batch_size=batch_size,
        num_ut_per_bs=num_ut_per_bs,
        effective_num_topology_drops=num_batches * batch_size,
    )


def _phase2_singular_metrics(
    h_port_raw: torch.Tensor,
    tau: torch.Tensor,
    cfg: Phase2Config,
    serving: torch.Tensor,
) -> dict[str, np.ndarray]:
    """Compute Phase 2 Config 2 PRB singular-value metrics."""

    num_ut = h_port_raw.shape[1]
    values = []
    for flat_idx, bs_idx in enumerate(serving.detach().cpu().tolist()):
        batch_idx = flat_idx // num_ut
        ut_idx = flat_idx % num_ut
        h = h_port_raw[batch_idx, ut_idx, :, bs_idx, :, :, 0]
        delays = tau[batch_idx, ut_idx, bs_idx]
        sv = prb_singular_values(
            h,
            delays,
            carrier_frequency=cfg.fc_ghz * 1e9,
            subcarrier_spacing=cfg.subcarrier_spacing_hz,
            prb_num_subcarriers=cfg.prb_num_subcarriers,
        )
        values.append(torch.stack([sv[0], sv[1], sv[0] - sv[1]]))

    stacked = torch.stack(values, dim=0)
    return {
        "sv1_db": stacked[:, 0].detach().cpu().numpy(),
        "sv2_db": stacked[:, 1].detach().cpu().numpy(),
        "sv_ratio_db": stacked[:, 2].detach().cpu().numpy(),
    }


def _run_phase2_once(
    cfg: Phase2Config,
    batch_size: int,
    num_ut_per_sector: int,
    seed: int,
    precision: str,
    device: str,
    spec_version: str,
    phase_name: str | None = None,
    enable_blockage: bool = False,
    blockage_self_blocking: str | None = None,
) -> CalibrationResult:
    """Run one Phase 2 drop and compute calibration metrics."""

    sionna_config.device = device
    sionna_config.precision = precision
    sionna_config.seed = seed

    topology = _make_topology(
        cfg,
        batch_size=batch_size,
        num_ut_per_sector=num_ut_per_sector,
        seed=seed,
        precision=precision,
        device=device,
    )
    _set_phase2_ut_orientations(topology, seed)

    path_gain_no_ant, high_loss_ut, details = _sample_path_gain_all_bs(
        cfg,
        topology,
        seed=seed + 1009,
        precision=precision,
        device=device,
        spec_version=spec_version,
        include_antenna_gain=False,
        return_details=True,
    )

    sionna_config.seed = seed + 2003
    ut_array, bs_array = _build_phase2_arrays(cfg, precision=precision, device=device)
    channel = _make_phase2_channel(
        cfg,
        ut_array,
        bs_array,
        precision,
        device,
        spec_version,
        enable_blockage=enable_blockage,
        blockage_self_blocking=blockage_self_blocking,
    )
    channel.set_topology(
        topology.ut_loc,
        topology.bs_loc,
        topology.ut_orientations,
        topology.bs_orientations,
        topology.ut_velocities,
        topology.in_state,
        los=details.outdoor_los,
        bs_virtual_loc=topology.bs_virtual_loc,
        bs_site_ids=topology.bs_site_ids,
        distance_2d_in=topology.distance_2d_in,
    )
    channel._lsp = details.lsp

    a, tau, rays = channel(num_time_samples=1, sampling_frequency=cfg.bandwidth_hz)
    port_weights = _phase2_port_mapping(bs_array, cfg, precision=precision)
    h_port_raw = _port_channels(a, port_weights)
    h_scaled = _apply_large_scale(a, path_gain_no_ant)
    h_port_scaled = _port_channels(h_scaled, port_weights)

    path_gain_all = _port0_path_gain_db(h_port_scaled)
    flat_gain = path_gain_all.reshape(-1, path_gain_all.shape[-1])
    coupling_loss, serving, serving_loss, serving_gain = _serving_link_values(flat_gain)
    sir = wideband_sir_db(coupling_loss, serving=serving)
    sinr = geometry_sinr_db(
        coupling_loss,
        tx_power_dbm=cfg.tx_power_dbm,
        bandwidth_hz=cfg.bandwidth_hz,
        noise_figure_db=cfg.noise_figure_db,
        serving=serving,
    )

    metrics = {
        "coupling_loss": serving_loss.detach().cpu().numpy(),
        "historical_path_gain": serving_gain.detach().cpu().numpy(),
        "wideband_sir": sir.detach().cpu().numpy(),
    }
    if cfg.config_id == 1:
        metrics.update(_phase2_spread_metrics(channel, rays, serving))
    else:
        metrics.update(_phase2_singular_metrics(h_port_raw, tau, cfg, serving))

    if phase_name == "blockage_model_a":
        metrics = {
            "coupling_loss": metrics["coupling_loss"],
            "historical_path_gain": metrics["historical_path_gain"],
            "wideband_sinr": sinr.detach().cpu().numpy(),
            "asa_deg": metrics["asa_deg"],
        }

    return CalibrationResult(
        scenario=cfg.scenario,
        fc_ghz=cfg.fc_ghz,
        phase=cfg.phase_name if phase_name is None else phase_name,
        metrics=metrics,
        serving_bs=serving.detach().cpu().numpy(),
        in_state=topology.in_state.detach().cpu().numpy().reshape(-1),
        o2i_is_high_loss=high_loss_ut.detach().cpu().numpy().reshape(-1),
        metadata={
            "serving_lsp_realization": "shared_correlated_full_topology",
        },
    )


def _run_phase2_batches(
    cfg: Phase2Config,
    num_batches: int,
    batch_size: int,
    num_ut_per_sector: int,
    seed: int,
    precision: str,
    devices: Sequence[str],
    spec_version: str,
    phase_name: str | None = None,
    enable_blockage: bool = False,
    blockage_self_blocking: str | None = None,
) -> CalibrationResult:
    """Run multiple independent Phase 2 drops."""

    results = []
    for batch_idx in range(num_batches):
        device = devices[batch_idx % len(devices)]
        results.append(
            _run_phase2_once(
                cfg,
                batch_size=batch_size,
                num_ut_per_sector=num_ut_per_sector,
                seed=seed + batch_idx,
                precision=precision,
                device=device,
                spec_version=spec_version,
                phase_name=phase_name,
                enable_blockage=enable_blockage,
                blockage_self_blocking=blockage_self_blocking,
            )
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return _with_workload_metadata(
        _concat_results(results),
        num_batches=num_batches,
        batch_size=batch_size,
        num_ut_per_sector=num_ut_per_sector,
        effective_num_topology_drops=num_batches * batch_size,
    )


def _make_blockage_model_b_geometry(
    precision: str,
    device: str,
    spec_version: str,
) -> tuple[UMiScenario, Topology, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create the TR 38.901 Table 7.8-6 CDL-E blockage geometry."""

    fc_hz = 30e9
    dtype = _torch_dtype(precision)
    ut_y = torch.arange(21, dtype=dtype, device=device)
    batch_size = ut_y.numel()

    cfg = _get_phase2_config("UMi", 30.0, 1)
    ut_array, bs_array = _build_phase2_arrays(cfg, precision=precision, device=device)
    ut_loc = torch.stack(
        [
            torch.full_like(ut_y, 100.0),
            ut_y,
            torch.full_like(ut_y, 1.5),
        ],
        dim=-1,
    ).unsqueeze(1)
    bs_loc = (
        torch.tensor([[[0.0, 0.0, 30.0]]], dtype=dtype, device=device)
        .expand(batch_size, 1, 3)
        .clone()
    )
    ut_orientations = torch.zeros(batch_size, 1, 3, dtype=dtype, device=device)
    bs_orientations = torch.zeros(batch_size, 1, 3, dtype=dtype, device=device)
    ut_velocities = torch.zeros(batch_size, 1, 3, dtype=dtype, device=device)
    in_state = torch.zeros(batch_size, 1, dtype=torch.bool, device=device)

    scenario = UMiScenario(
        carrier_frequency=fc_hz,
        o2i_model="low",
        ut_array=ut_array,
        bs_array=bs_array,
        direction="downlink",
        enable_pathloss=False,
        enable_shadow_fading=False,
        precision=precision,
        device=device,
        spec_version=spec_version,
    )
    scenario.set_topology(
        ut_loc,
        bs_loc,
        ut_orientations,
        bs_orientations,
        ut_velocities,
        in_state,
        los=True,
    )

    topology = Topology(
        velocities=ut_velocities,
        moving_end="rx",
        los_aoa=deg_2_rad(scenario.los_aoa),
        los_aod=deg_2_rad(scenario.los_aod),
        los_zoa=deg_2_rad(scenario.los_zoa),
        los_zod=deg_2_rad(scenario.los_zod),
        los=torch.ones(batch_size, 1, 1, dtype=torch.bool, device=device),
        distance_3d=scenario.distance_3d,
        tx_orientations=bs_orientations,
        rx_orientations=ut_orientations,
    )
    return scenario, topology, ut_y, ut_array, bs_array


def _expand_first_batch(tensor: torch.Tensor, batch_size: int) -> torch.Tensor:
    """Expand a single-link CDL tensor to all UT positions."""

    return tensor.expand(batch_size, *tensor.shape[1:])


def _translated_cdl_e_rays(
    cdl: CDL,
    scenario: UMiScenario,
    batch_size: int,
) -> Rays:
    """Create CDL-E rays translated to the BS-UT direct path."""

    base = cdl._create_rays(1)
    target_aoa = deg_2_rad(scenario.los_aoa).reshape(batch_size, 1, 1, 1, 1)
    target_aod = deg_2_rad(scenario.los_aod).reshape(batch_size, 1, 1, 1, 1)
    target_zoa = deg_2_rad(scenario.los_zoa).reshape(batch_size, 1, 1, 1, 1)
    target_zod = deg_2_rad(scenario.los_zod).reshape(batch_size, 1, 1, 1, 1)

    return Rays(
        delays=_expand_first_batch(base.delays, batch_size),
        powers=_expand_first_batch(base.powers, batch_size),
        aoa=_expand_first_batch(base.aoa, batch_size) + (target_aoa - cdl._los_aoa),
        aod=_expand_first_batch(base.aod, batch_size) + (target_aod - cdl._los_aod),
        zoa=_expand_first_batch(base.zoa, batch_size) + (target_zoa - cdl._los_zoa),
        zod=_expand_first_batch(base.zod, batch_size) + (target_zod - cdl._los_zod),
        xpr=_expand_first_batch(base.xpr, batch_size),
    )


def _run_blockage_model_b_once(
    seed: int,
    precision: str,
    device: str,
    spec_version: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Run one CDL-E ray-coupling realization for blockage Model B."""

    sionna_config.device = device
    sionna_config.precision = precision
    sionna_config.seed = seed

    scenario, topology, ut_y, ut_array, bs_array = _make_blockage_model_b_geometry(
        precision=precision,
        device=device,
        spec_version=spec_version,
    )
    batch_size = ut_y.numel()
    dtype = _torch_dtype(precision)
    fc_hz = 30e9

    cdl = CDL(
        model="E",
        delay_spread=100e-9,
        carrier_frequency=fc_hz,
        ut_array=ut_array,
        bs_array=bs_array,
        direction="downlink",
        ut_orientation=torch.zeros(batch_size, 3, dtype=dtype, device=device),
        bs_orientation=torch.zeros(batch_size, 3, dtype=dtype, device=device),
        ut_velocity=torch.zeros(batch_size, 3, dtype=dtype, device=device),
        precision=precision,
        device=device,
        spec_version=spec_version,
    )
    rays = _translated_cdl_e_rays(cdl, scenario, batch_size)

    # Table 7.8-6 places the screen at z=1.5 m. The calibration reference
    # curve follows the interpretation that this is the lower-edge centre.
    # Sionna's public Model B API uses the physical screen centre from
    # Section 7.6.4.2, hence the conversion to z=1.5+h/2.
    blocker_height_m = 10.0
    blocker_width_m = 2.0
    blocker_center = torch.tensor(
        [[80.0, 10.0, 1.5 + 0.5 * blocker_height_m]],
        dtype=dtype,
        device=device,
    )
    blockage = BlockageModelB(
        scenario,
        blocker_centers=blocker_center,
        blocker_widths=torch.tensor([blocker_width_m], dtype=dtype, device=device),
        blocker_heights=torch.tensor([blocker_height_m], dtype=dtype, device=device),
        precision=precision,
        device=device,
    )
    ray_loss, los_loss = blockage(
        rad_2_deg(rays.aoa),
        rad_2_deg(rays.zoa),
        scenario.los_aoa,
        scenario.los_zoa,
    )
    rays.blockage_loss_db = ray_loss
    rays.los_blockage_loss_db = los_loss
    rays.blockage_loss_applied_to_powers = False

    generator = ChannelCoefficientsGenerator(
        carrier_frequency=fc_hz,
        tx_array=bs_array,
        rx_array=ut_array,
        subclustering=False,
        precision=precision,
        device=device,
    )
    h, _ = generator(
        num_time_samples=1,
        sampling_frequency=1.0,
        k_factor=cdl._get_k_factor(batch_size),
        rays=rays,
        topology=topology,
    )
    h = h.permute(0, 2, 4, 1, 5, 3, 6)
    port_weights = _phase2_port_mapping(
        bs_array,
        _get_phase2_config("UMi", 30.0, 1),
        precision=precision,
    )
    h_port = _port_channels(h, port_weights)
    h_port0 = h_port[:, :, :, :, 0, :, :]
    rsrp_linear = torch.sum(torch.abs(h_port0) ** 2, dim=(2, 4, 5))
    rsrp_linear = rsrp_linear / (h_port0.shape[2] * h_port0.shape[-1])

    return (
        ut_y.detach().cpu().numpy(),
        rsrp_linear.reshape(-1).detach().cpu().numpy(),
    )


def _run_blockage_model_b(
    num_realizations: int,
    seed: int,
    precision: str,
    devices: Sequence[str],
    spec_version: str,
) -> LineCalibrationResult:
    """Run the TR 38.901 Table 7.8-6 blockage Model B calibration."""

    if num_realizations <= 0:
        raise ValueError("blockage_model_b requires at least one realization")

    ut_y = None
    accumulated = None
    for idx in range(num_realizations):
        device = devices[idx % len(devices)]
        current_y, rsrp_linear = _run_blockage_model_b_once(
            seed=seed + idx,
            precision=precision,
            device=device,
            spec_version=spec_version,
        )
        ut_y = current_y if ut_y is None else ut_y
        accumulated = (
            rsrp_linear.copy() if accumulated is None else accumulated + rsrp_linear
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    rsrp_linear = accumulated / float(num_realizations)
    rsrp_db = 10.0 * np.log10(np.maximum(rsrp_linear, np.finfo(np.float64).tiny))
    relative_rsrp_db = rsrp_db - rsrp_db[0]

    return LineCalibrationResult(
        scenario="UMi",
        fc_ghz=30.0,
        phase="blockage_model_b",
        x_label="UT y-position [m]",
        x_values=ut_y,
        metrics={
            "rsrp_db": rsrp_db,
            "relative_rsrp_db": relative_rsrp_db,
        },
        metadata={
            "num_cdl_e_realizations": int(num_realizations),
            "cdl_model": "CDL-E",
            "delay_spread_s": 100e-9,
            "bs_position_m": [0.0, 0.0, 30.0],
            "ut_positions_m": "[(100, y, 1.5) for y in 0..20]",
            "blocker_table_position_m": [80.0, 10.0, 1.5],
            "blocker_center_m": [80.0, 10.0, 6.5],
            "blocker_width_m": 2.0,
            "blocker_height_m": 10.0,
            "note": (
                "The Table 7.8-6 screen z coordinate is converted to the "
                "physical centre expected by BlockageModelB by adding h/2. "
                "The RSRP curve is computed from the coherent CIR after "
                "per-ray amplitude blockage losses, averaged in linear power "
                "over CDL-E ray-coupling realizations. This matches the "
                "bundled 3GPP Model B reference curve convention."
            ),
        },
    )


def _finite_array(values: np.ndarray, name: str) -> np.ndarray:
    """Return a non-empty finite float array or raise a diagnostic error."""

    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ValueError(f"`{name}` must not be empty")
    finite = np.isfinite(array)
    if not np.all(finite):
        invalid = int(array.size - np.count_nonzero(finite))
        raise ValueError(f"`{name}` contains {invalid} non-finite value(s)")
    return array


def _cdf_curve(values: np.ndarray, percentiles: Sequence[float]) -> dict:
    """Convert finite samples to JSON-serializable CDF curve data."""

    samples = _finite_array(values, "CDF samples")
    curve = [float(v) for v in np.percentile(samples, percentiles)]
    return {
        "label": "",
        "percentiles": [float(p) for p in percentiles],
        "cdf": [float(p) for p in percentiles],
        "x": curve,
        "num_samples": int(samples.size),
    }


def _write_cdf_json(
    output_dir: Path,
    result: CalibrationResult,
    percentiles: Sequence[float],
) -> Path:
    """Write one scenario/frequency/phase CDF JSON file."""

    key = f"{result.scenario}_{result.fc_ghz:g}GHz_{result.phase}"
    metrics = {}
    for metric_key, values in result.metrics.items():
        curve = _cdf_curve(values, percentiles)
        curve["label"] = METRIC_LABELS[metric_key]
        metrics[metric_key] = curve
    metric_conventions = {
        "coupling_loss": "Positive serving-link loss in dB.",
        "historical_path_gain": (
            "Negative serving-link path gain retained for comparison with "
            "historical workbook curves."
        ),
        "historical_wideband_sir": (
            "Interference-only contribution metric retained separately from "
            "noise-inclusive wideband_sinr."
        ),
    }

    data = {
        "metadata": {
            "standard": PHASE_STANDARDS[result.phase],
            "generator": "test/unit/channel/tr38901_calibration.py",
            "scenario": result.scenario,
            "frequency_ghz": float(result.fc_ghz),
            "phase": result.phase,
            "calibration_schema_version": CALIBRATION_SCHEMA_VERSION,
            "num_samples": result.num_samples,
            "indoor_fraction": float(np.mean(result.in_state)),
            "high_loss_fraction_among_indoor": float(
                np.mean(result.o2i_is_high_loss[result.in_state])
            )
            if np.any(result.in_state)
            else 0.0,
            "metric_conventions": {
                key: value
                for key, value in metric_conventions.items()
                if key in metrics
            },
        },
        "metrics": metrics,
    }
    if result.metadata:
        data["metadata"].update(result.metadata)

    path = output_dir / "cdfs" / f"{key}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False))
    return path


def _write_line_json(output_dir: Path, result: LineCalibrationResult) -> Path:
    """Write one scenario/frequency/phase line-curve JSON file."""

    key = f"{result.scenario}_{result.fc_ghz:g}GHz_{result.phase}"
    x_values = _finite_array(result.x_values, "line-curve x values")
    metrics = {}
    for metric_key, values in result.metrics.items():
        metric_values = _finite_array(values, f"line-curve metric {metric_key}")
        if metric_values.size != x_values.size:
            raise ValueError(
                f"Line-curve metric `{metric_key}` has {metric_values.size} "
                f"values for {x_values.size} x coordinates"
            )
        metrics[metric_key] = {
            "label": METRIC_LABELS[metric_key],
            "x_label": result.x_label,
            "x": [float(v) for v in x_values],
            "y": [float(v) for v in metric_values],
            "num_samples": int(metric_values.size),
        }

    data = {
        "metadata": {
            "standard": PHASE_STANDARDS[result.phase],
            "generator": "test/unit/channel/tr38901_calibration.py",
            "scenario": result.scenario,
            "frequency_ghz": float(result.fc_ghz),
            "phase": result.phase,
            "calibration_schema_version": CALIBRATION_SCHEMA_VERSION,
            "curve_type": "line",
            "num_samples": result.num_samples,
            **result.metadata,
        },
        "metrics": metrics,
    }

    path = output_dir / "cdfs" / f"{key}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False))
    return path


WORKLOAD_METADATA_FIELDS = (
    "num_batches",
    "batch_size",
    "effective_num_topology_drops",
    "num_ut_per_sector",
    "num_ut_per_bs",
    "num_bs",
    "num_drops",
    "ut_chunk_size",
    "num_cdl_e_realizations",
)


def _workload_metadata(metadata: dict) -> dict:
    """Extract workload provenance stored with one generated result."""

    return {
        field: metadata[field]
        for field in WORKLOAD_METADATA_FIELDS
        if field in metadata
    }


def _with_spec_metadata(
    result: CalibrationResult | LineCalibrationResult,
    spec_version: str,
) -> CalibrationResult | LineCalibrationResult:
    """Return ``result`` with explicit spec-version provenance."""

    metadata = dict(result.metadata or {})
    metadata["spec_version"] = spec_version
    metadata["calibration_schema_version"] = CALIBRATION_SCHEMA_VERSION
    return replace(result, metadata=metadata)


def _existing_cdf_runs(output_dir: Path) -> dict:
    """Return manifest entries for existing CDF JSON files."""

    runs = {}
    cdf_dir = output_dir / "cdfs"
    if not cdf_dir.exists():
        return runs
    for path in sorted(cdf_dir.glob("*.json")):
        data = json.loads(path.read_text())
        metadata = data.get("metadata", {})
        runs[path.stem] = {
            "scenario": metadata.get("scenario"),
            "frequency_ghz": metadata.get("frequency_ghz"),
            "phase": metadata.get("phase"),
            "curve_type": metadata.get("curve_type", "cdf"),
            "num_samples": metadata.get("num_samples"),
            "file": str(path.relative_to(output_dir)),
            "metrics": list(data.get("metrics", {})),
            "spec_version": metadata.get("spec_version"),
            "calibration_schema_version": metadata.get("calibration_schema_version"),
            "device": metadata.get("device"),
            "workload": _workload_metadata(metadata),
        }
    return runs


def _run_one(
    scenario: str,
    fc_ghz: float,
    phase: str,
    args: argparse.Namespace,
    devices: Sequence[str],
) -> CalibrationResult | LineCalibrationResult:
    """Run one selected calibration phase."""

    spec_version = CALIBRATION_SPEC_VERSION

    if phase == "phase1":
        cfg = _get_phase1_config(scenario, fc_ghz)
        return _run_phase1_batches(
            cfg,
            num_batches=args.phase1_num_batches,
            batch_size=args.batch_size,
            num_ut_per_sector=args.phase1_uts_per_sector,
            seed=args.seed,
            precision=args.precision,
            devices=devices,
            spec_version=spec_version,
        )

    if phase == "spatial_consistency_metric1_2":
        if scenario.strip().lower() not in (
            "umi",
            "umi-street canyon",
            "umi-street-canyon",
        ):
            raise ValueError(
                "The spatial_consistency_metric1_2 calibration is defined only for UMi."
            )
        if not np.isclose(float(fc_ghz), 30.0):
            raise ValueError(
                "The spatial_consistency_metric1_2 calibration is defined at 30 GHz."
            )
        return _run_spatial_metric1_2(
            num_ut_per_sector=args.spatial_metric1_2_num_ut_per_sector,
            num_drops=args.spatial_metric1_2_num_drops,
            ut_chunk_size=args.spatial_metric1_2_ut_chunk_size,
            carrier_frequency=30e9,
            precision=args.precision,
            devices=devices,
            seed=args.seed,
            spec_version=spec_version,
        )

    if phase == "spatial_consistency_metric3_6":
        if scenario.strip().lower() not in (
            "umi",
            "umi-street canyon",
            "umi-street-canyon",
        ):
            raise ValueError(
                "The spatial_consistency_metric3_6 calibration is defined only for UMi."
            )
        if not np.isclose(float(fc_ghz), 30.0):
            raise ValueError(
                "The spatial_consistency_metric3_6 calibration is defined at 30 GHz."
            )
        return _run_spatial_metric3_6(
            distances_m=[float(v) for v in args.spatial_metric3_6_distances_m],
            num_ut_per_sector=args.spatial_metric3_6_num_ut_per_sector,
            num_drops=args.spatial_metric3_6_num_drops,
            carrier_frequency=30e9,
            precision=args.precision,
            devices=devices,
            seed=args.seed,
            spec_version=spec_version,
        )

    if phase == "spatial_consistency_config2_metric1_2":
        if scenario.strip().lower() not in (
            "umi",
            "umi-street canyon",
            "umi-street-canyon",
        ):
            raise ValueError(
                "The spatial_consistency_config2_metric1_2 calibration is "
                "defined only for UMi."
            )
        if not np.isclose(float(fc_ghz), 30.0):
            raise ValueError(
                "The spatial_consistency_config2_metric1_2 calibration is "
                "defined at 30 GHz."
            )
        return _run_spatial_config2_metric1_2(
            num_ut_per_sector=args.spatial_metric1_2_num_ut_per_sector,
            num_drops=args.spatial_metric1_2_num_drops,
            ut_chunk_size=args.spatial_metric1_2_ut_chunk_size,
            carrier_frequency=30e9,
            precision=args.precision,
            devices=devices,
            seed=args.seed,
            spec_version=spec_version,
        )

    if phase == "blockage_model_a":
        if scenario.strip().lower() not in (
            "umi",
            "umi-street canyon",
            "umi-street-canyon",
        ):
            raise ValueError(
                "The blockage_model_a calibration is defined only for UMi."
            )
        if not np.isclose(float(fc_ghz), 30.0):
            raise ValueError("The blockage_model_a calibration is defined at 30 GHz.")
        cfg = _get_phase2_config("UMi", 30.0, 1)
        return _run_phase2_batches(
            cfg,
            num_batches=args.phase2_num_batches,
            batch_size=args.batch_size,
            num_ut_per_sector=args.phase2_uts_per_sector,
            seed=args.seed,
            precision=args.precision,
            devices=devices,
            spec_version=spec_version,
            phase_name="blockage_model_a",
            enable_blockage=True,
            blockage_self_blocking="landscape",
        )

    if phase == "blockage_model_b":
        if scenario.strip().lower() not in (
            "umi",
            "umi-street canyon",
            "umi-street-canyon",
        ):
            raise ValueError(
                "The blockage_model_b calibration is defined only for UMi."
            )
        if not np.isclose(float(fc_ghz), 30.0):
            raise ValueError("The blockage_model_b calibration is defined at 30 GHz.")
        return _run_blockage_model_b(
            num_realizations=args.phase2_num_batches,
            seed=args.seed,
            precision=args.precision,
            devices=devices,
            spec_version=spec_version,
        )

    if phase == INF_NORMATIVE_PHASE:
        if float(fc_ghz) not in (3.5, 28.0):
            raise ValueError(
                f"The {phase} calibration is defined at 3.5 GHz and 28 GHz."
            )
        cfg = _get_inf_config(scenario, fc_ghz)
        return _run_inf_batches(
            cfg,
            num_batches=args.inf_num_batches,
            batch_size=args.batch_size,
            num_ut_per_bs=args.inf_uts_per_bs,
            ut_chunk_size=args.inf_ut_chunk_size,
            seed=args.seed,
            precision=args.precision,
            devices=devices,
            spec_version=spec_version,
        )

    config_id = 1 if phase == "config1" else 2
    cfg = _get_phase2_config(scenario, fc_ghz, config_id)
    return _run_phase2_batches(
        cfg,
        num_batches=args.phase2_num_batches,
        batch_size=args.batch_size,
        num_ut_per_sector=args.phase2_uts_per_sector,
        seed=args.seed,
        precision=args.precision,
        devices=devices,
        spec_version=spec_version,
    )


def _calibration_requests(args: argparse.Namespace) -> list[tuple[str, float, str]]:
    """Return selected scenario/frequency/phase runs in manifest order."""

    requests = []
    for phase in args.phases:
        phase_scenarios = _scenarios_for_phase(phase, args.scenarios)
        phase_frequencies = _frequencies_for_phase(phase, args.frequencies_ghz)
        requests.extend(
            (scenario, fc_ghz, phase)
            for scenario in phase_scenarios
            for fc_ghz in phase_frequencies
        )
    return requests


def _validated_devices(requested_devices: Sequence[str]) -> list[str]:
    """Validate and canonicalize devices before any output-side effects."""

    if not requested_devices:
        raise ValueError("--devices must contain at least one device")

    devices = []
    for value in requested_devices:
        try:
            parsed = torch.device(value)
        except (RuntimeError, TypeError) as err:
            raise ValueError(f"Invalid calibration device {value!r}") from err

        if parsed.type == "cuda":
            if not torch.cuda.is_available():
                raise ValueError(f"CUDA device requested but unavailable: {value!r}")
            index = 0 if parsed.index is None else parsed.index
            if index < 0 or index >= torch.cuda.device_count():
                raise ValueError(
                    f"CUDA device index {index} is outside the visible range "
                    f"[0, {torch.cuda.device_count()})"
                )
            canonical = f"cuda:{index}"
        elif parsed.type == "cpu":
            canonical = "cpu"
        else:
            raise ValueError(
                f"Unsupported calibration device type {parsed.type!r}; use cpu or cuda"
            )
        if canonical not in devices:
            devices.append(canonical)
    return devices


def _validate_strictly_increasing_finite(
    values: Sequence[float],
    option: str,
    minimum: float,
    maximum: float | None = None,
) -> None:
    """Validate a non-empty finite numeric grid."""

    if not values:
        raise ValueError(f"{option} must not be empty")
    numeric = [float(value) for value in values]
    if not all(math.isfinite(value) for value in numeric):
        raise ValueError(f"{option} must contain only finite values")
    if any(value < minimum for value in numeric):
        raise ValueError(f"{option} values must be at least {minimum:g}")
    if maximum is not None and any(value > maximum for value in numeric):
        raise ValueError(f"{option} values must be at most {maximum:g}")
    if any(left >= right for left, right in zip(numeric, numeric[1:])):
        raise ValueError(f"{option} values must be strictly increasing")


def _validate_request_definition(request: tuple[str, float, str]) -> None:
    """Validate one request without constructing a channel or topology."""

    scenario, fc_ghz, phase = request
    if not math.isfinite(float(fc_ghz)) or float(fc_ghz) <= 0.0:
        raise ValueError("--frequencies-ghz values must be finite and positive")

    if phase == "phase1":
        _get_phase1_config(scenario, fc_ghz)
    elif phase in ("config1", "config2"):
        _get_phase2_config(scenario, fc_ghz, 1 if phase == "config1" else 2)
    elif phase == INF_NORMATIVE_PHASE:
        if float(fc_ghz) not in (3.5, 28.0):
            raise ValueError(f"The {phase} calibration is defined at 3.5 and 28 GHz")
        _get_inf_config(scenario, fc_ghz)
    else:
        if scenario != "UMi" or not np.isclose(float(fc_ghz), 30.0):
            raise ValueError(
                f"The {phase} calibration is defined only for UMi at 30 GHz"
            )


def _validate_reference_json(reference_json: Path) -> dict:
    """Validate that reference data are readable before a long run."""

    if not reference_json.is_file():
        raise ValueError(f"Reference JSON does not exist: {reference_json}")
    try:
        reference = json.loads(reference_json.read_text())
    except (OSError, json.JSONDecodeError) as err:
        raise ValueError(f"Reference JSON is not readable: {reference_json}") from err
    if not isinstance(reference, dict) or "metadata" not in reference:
        raise ValueError("Reference JSON must contain a top-level metadata object")
    metadata = reference["metadata"]
    if not isinstance(metadata, dict):
        raise ValueError("Reference JSON metadata must be an object")
    phase_keys = set(reference) - {"metadata"}
    source_files = metadata.get("source_files")
    contribution_ids = metadata.get("source_contributions")
    standards = metadata.get("reference_standard_by_phase")
    provenance = metadata.get("provenance")
    if not isinstance(source_files, dict) or not source_files:
        raise ValueError("Reference metadata must record source_files")
    if not set(source_files) <= phase_keys:
        raise ValueError("Reference source_files contain an unknown phase")
    if not isinstance(contribution_ids, list) or not all(
        isinstance(value, str) and value for value in contribution_ids
    ):
        raise ValueError("Reference metadata must record source_contributions")
    if not isinstance(standards, dict) or set(standards) != phase_keys:
        raise ValueError("Reference standards must cover every reference phase")
    if not isinstance(provenance, dict):
        raise ValueError("Reference metadata must contain provenance")
    if provenance.get("source_workbook_filenames_by_phase") != source_files:
        raise ValueError("Reference provenance source filenames are inconsistent")
    if provenance.get("source_contribution_ids") != contribution_ids:
        raise ValueError("Reference provenance contribution IDs are inconsistent")
    for field in (
        "source_workbook_binaries_bundled",
        "retrieval_urls_bundled",
        "source_checksums_bundled",
        "extraction_tooling_bundled",
    ):
        if not isinstance(provenance.get(field), bool):
            raise ValueError(f"Reference provenance field {field!r} must be boolean")
    return reference


def _preflight_calibration(
    args: argparse.Namespace,
    devices: Sequence[str],
    output_dir: Path,
    reference_json: Path,
) -> list[tuple[str, float, str]]:
    """Validate a complete invocation before creating or cleaning output."""

    positive_options = {
        "--phase1-num-batches": args.phase1_num_batches,
        "--phase2-num-batches": args.phase2_num_batches,
        "--batch-size": args.batch_size,
        "--phase1-uts-per-sector": args.phase1_uts_per_sector,
        "--phase2-uts-per-sector": args.phase2_uts_per_sector,
        "--spatial-metric1-2-num-ut-per-sector": (
            args.spatial_metric1_2_num_ut_per_sector
        ),
        "--spatial-metric1-2-num-drops": args.spatial_metric1_2_num_drops,
        "--spatial-metric1-2-ut-chunk-size": (args.spatial_metric1_2_ut_chunk_size),
        "--spatial-metric3-6-num-ut-per-sector": (
            args.spatial_metric3_6_num_ut_per_sector
        ),
        "--spatial-metric3-6-num-drops": args.spatial_metric3_6_num_drops,
        "--inf-num-batches": args.inf_num_batches,
        "--inf-uts-per-bs": args.inf_uts_per_bs,
        "--inf-ut-chunk-size": args.inf_ut_chunk_size,
    }
    for option, value in positive_options.items():
        if int(value) <= 0:
            raise ValueError(f"{option} must be positive")

    _validate_strictly_increasing_finite(
        args.cdf_percentiles,
        "--cdf-percentiles",
        minimum=0.0,
        maximum=100.0,
    )
    _validate_strictly_increasing_finite(
        args.spatial_metric3_6_distances_m,
        "--spatial-metric3-6-distances-m",
        minimum=0.0,
    )
    if args.frequencies_ghz is not None:
        _validate_strictly_increasing_finite(
            args.frequencies_ghz,
            "--frequencies-ghz",
            minimum=0.0,
        )
        if any(float(value) <= 0.0 for value in args.frequencies_ghz):
            raise ValueError("--frequencies-ghz values must be positive")

    if not devices:
        raise ValueError("At least one validated calibration device is required")
    requests = _calibration_requests(args)
    if not requests:
        raise ValueError("The selected arguments produce no calibration requests")
    for request in requests:
        _validate_request_definition(request)
    if len(set(requests)) != len(requests):
        raise ValueError(
            "The selected arguments produce duplicate calibration requests"
        )

    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError(f"Output path is not a directory: {output_dir}")
    _validate_reference_json(reference_json)
    if args.clean:
        for child in ("cdfs", "figures", "samples"):
            try:
                reference_json.relative_to(output_dir / child)
            except ValueError:
                continue
            raise ValueError(
                f"--reference-json would be removed by --clean: {reference_json}"
            )
    if not args.no_plot:
        if not PLOT_SCRIPT.is_file():
            raise ValueError(f"Calibration plotter does not exist: {PLOT_SCRIPT}")
        if importlib.util.find_spec("matplotlib") is None:
            raise ValueError("matplotlib is required unless --no-plot is used")
        _load_plotter()
    return requests


def _run_request(
    request: tuple[str, float, str],
    args: argparse.Namespace,
    device: str,
) -> CalibrationResult | LineCalibrationResult:
    """Run one calibration request on one worker-local device."""

    scenario, fc_ghz, phase = request
    result = _run_one(scenario, fc_ghz, phase, args, [device])
    result = _with_spec_metadata(result, CALIBRATION_SPEC_VERSION)
    metadata = dict(result.metadata or {})
    metadata["device"] = device
    return replace(result, metadata=metadata)


def _static_request_assignments(
    requests: Sequence[tuple[str, float, str]],
    devices: Sequence[str],
) -> list[list[tuple[int, tuple[str, float, str]]]]:
    """Assign request indices to devices by stable round-robin partitioning."""

    if not devices:
        raise ValueError("At least one calibration device is required")
    if not requests:
        return []
    worker_count = min(len(devices), len(requests))
    return [
        [
            (request_index, request)
            for request_index, request in enumerate(requests)
            if request_index % worker_count == worker_index
        ]
        for worker_index in range(worker_count)
    ]


def _run_assigned_requests(
    assignments: Sequence[tuple[int, tuple[str, float, str]]],
    args: argparse.Namespace,
    device: str,
) -> list[tuple[int, CalibrationResult | LineCalibrationResult]]:
    """Run one worker's statically assigned requests in request order."""

    return [
        (request_index, _run_request(request, args, device))
        for request_index, request in assignments
    ]


def _run_requests(
    requests: Sequence[tuple[str, float, str]],
    args: argparse.Namespace,
    devices: Sequence[str],
) -> list[CalibrationResult | LineCalibrationResult]:
    """Run requests serially or concurrently across device-bound workers."""

    if len(devices) == 1 or len(requests) <= 1:
        return [_run_request(request, args, devices[0]) for request in requests]

    assignments = _static_request_assignments(requests, devices)
    worker_count = len(assignments)
    context = multiprocessing.get_context("spawn")
    executors = [
        ProcessPoolExecutor(max_workers=1, mp_context=context)
        for _ in range(worker_count)
    ]
    try:
        results: list[CalibrationResult | LineCalibrationResult | None] = [None] * len(
            requests
        )
        futures = [
            executor.submit(
                _run_assigned_requests,
                assignments[worker_index],
                args,
                devices[worker_index],
            )
            for worker_index, executor in enumerate(executors)
        ]
        for future in futures:
            for result_index, result in future.result():
                results[result_index] = result

        if any(result is None for result in results):
            raise RuntimeError("A calibration worker returned no result")
        return [result for result in results if result is not None]
    finally:
        for executor in executors:
            executor.shutdown(wait=True, cancel_futures=True)


def main(argv: list[str] | None = None) -> None:
    """Run the calibration CDF generation workflow."""

    args = PARSER.parse_args(argv)
    requested_devices = _default_devices() if args.devices is None else args.devices
    devices = _validated_devices(requested_devices)
    output_dir = _resolve_output_dir(args.output_dir)
    reference_json = _resolve_output_dir(args.reference_json)
    requests = _preflight_calibration(args, devices, output_dir, reference_json)
    reference_metadata = json.loads(reference_json.read_text())["metadata"]
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.clean:
        _clean_output_dir(output_dir)

    manifest = {
        "standard": "3GPP TR 38.901 V16.1 calibration",
        "baseline_reference_standard": (
            "3GPP TR 38.901 v14.0.0 Section 7.8 baseline calibration"
        ),
        "additional_feature_reference_standard": (
            "3GPP TR 38.901 v14.0.0 Tables 7.8-5 and 7.8-6 "
            "additional-feature calibration workbooks"
        ),
        "normative_indoor_factory_standard": (
            "3GPP TR 38.901 v16.1.0 Table 7.8-7 indoor-factory calibration"
        ),
        "indoor_factory_reference_note": INF_REFERENCE_NOTE,
        "indoor_hotspot_reference_note": INH_REFERENCE_NOTE,
        "reference_standard_by_phase": reference_metadata.get(
            "reference_standard_by_phase", {}
        ),
        "reference_provenance": reference_metadata.get("provenance", {}),
        "reference_value_conventions": reference_metadata.get("value_conventions", {}),
        "generator": "test/unit/channel/tr38901_calibration.py",
        "scenarios": args.scenarios,
        "frequencies_ghz": (
            None
            if args.frequencies_ghz is None
            else [float(f) for f in args.frequencies_ghz]
        ),
        "default_frequencies_by_phase": DEFAULT_FREQUENCIES_BY_PHASE,
        "phases": args.phases,
        "spec_version": CALIBRATION_SPEC_VERSION,
        "calibration_schema_version": CALIBRATION_SCHEMA_VERSION,
        "phase1_num_batches": int(args.phase1_num_batches),
        "phase2_num_batches": int(args.phase2_num_batches),
        "batch_size": int(args.batch_size),
        "phase1_uts_per_sector": int(args.phase1_uts_per_sector),
        "phase2_uts_per_sector": int(args.phase2_uts_per_sector),
        "spatial_metric1_2_num_ut_per_sector": int(
            args.spatial_metric1_2_num_ut_per_sector
        ),
        "spatial_metric1_2_num_drops": int(args.spatial_metric1_2_num_drops),
        "spatial_metric1_2_ut_chunk_size": int(args.spatial_metric1_2_ut_chunk_size),
        "spatial_metric3_6_num_ut_per_sector": int(
            args.spatial_metric3_6_num_ut_per_sector
        ),
        "spatial_metric3_6_num_drops": int(args.spatial_metric3_6_num_drops),
        "spatial_metric3_6_distances_m": [
            float(v) for v in args.spatial_metric3_6_distances_m
        ],
        "inf_num_batches": int(args.inf_num_batches),
        "inf_uts_per_bs": int(args.inf_uts_per_bs),
        "inf_ut_chunk_size": int(args.inf_ut_chunk_size),
        "seed": int(args.seed),
        "precision": args.precision,
        "devices": devices,
        "device_assignment": "static_request_index_modulo_device_count",
        "cdf_percentiles": [float(p) for p in args.cdf_percentiles],
        "reference_json": _manifest_path(reference_json, output_dir),
        "figures_dir": _manifest_path(output_dir / "figures", output_dir),
        "plot": not args.no_plot,
        "runs": _existing_cdf_runs(output_dir),
    }

    if len(devices) > 1 and len(requests) > 1:
        print(
            f"Running {len(requests)} calibration runs concurrently on "
            f"{', '.join(devices)}"
        )
    results = _run_requests(requests, args, devices)
    for result in results:
        if isinstance(result, LineCalibrationResult):
            path = _write_line_json(output_dir, result)
        else:
            path = _write_cdf_json(output_dir, result, args.cdf_percentiles)
        key = f"{result.scenario}_{result.fc_ghz:g}GHz_{result.phase}"
        manifest["runs"][key] = {
            "scenario": result.scenario,
            "frequency_ghz": float(result.fc_ghz),
            "phase": result.phase,
            "curve_type": "line"
            if isinstance(result, LineCalibrationResult)
            else "cdf",
            "num_samples": result.num_samples,
            "file": str(path.relative_to(output_dir)),
            "metrics": list(result.metrics),
            "spec_version": CALIBRATION_SPEC_VERSION,
            "calibration_schema_version": CALIBRATION_SCHEMA_VERSION,
            "device": (result.metadata or {}).get("device"),
            "workload": _workload_metadata(result.metadata or {}),
        }
        print(f"Wrote {path} ({result.num_samples} samples)")

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {manifest_path}")

    if not args.no_plot:
        plot_calibration_results = _load_plotter()
        output_paths = plot_calibration_results(
            input_dir=output_dir / "cdfs",
            reference_json=reference_json,
            output_dir=output_dir / "figures",
            scenarios=args.scenarios,
            frequencies_ghz=args.frequencies_ghz,
            phases=args.phases,
            clean=False,
        )
        for output_path in output_paths:
            print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
