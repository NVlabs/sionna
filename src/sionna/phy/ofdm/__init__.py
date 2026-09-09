#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""OFDM Module of the Sionna PHY Package"""

from .pilot_pattern import PilotPattern, EmptyPilotPattern, KroneckerPilotPattern
from .modulator import OFDMModulator
from .demodulator import OFDMDemodulator
from .resource_grid import (
    ResourceGrid,
    ResourceGridMapper,
    ResourceGridDemapper,
    RemoveNulledSubcarriers,
)
from .detection import (
    OFDMDetector,
    OFDMDetectorWithPrior,
    MaximumLikelihoodDetector,
    MaximumLikelihoodDetectorWithPrior,
    LinearDetector,
    KBestDetector,
    EPDetector,
    MMSEPICDetector,
)
from .equalization import (
    OFDMEqualizer,
    LMMSEEqualizer,
    ZFEqualizer,
    MFEqualizer,
    PostEqualizationSINR,
    LMMSEPostEqualizationSINR,
)
from .channel_estimation import (
    BaseChannelEstimator,
    BasePilotChannelEstimator,
    LSChannelEstimator,
    LMMSEChannelEstimator,
    BaseChannelInterpolator,
    NearestNeighborInterpolator,
    LinearInterpolator,
    tdl_freq_cov_mat,
    tdl_time_cov_mat,
)
from .precoding import (
    RZFPrecoder,
    PrecodedChannel,
    RZFPrecodedChannel,
    CBFPrecodedChannel,
    EyePrecodedChannel,
)
