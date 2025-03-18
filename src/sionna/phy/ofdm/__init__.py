#
# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0#
"""OFDM Module of the Sionna PHY Package"""

from .resource_grid import ResourceGrid, RemoveNulledSubcarriers,\
                           ResourceGridMapper, ResourceGridDemapper
from .pilot_pattern import PilotPattern, EmptyPilotPattern,\
                           KroneckerPilotPattern
from .modulator import OFDMModulator
from .demodulator import OFDMDemodulator
from .channel_estimation import LSChannelEstimator,\
                                NearestNeighborInterpolator,\
                                LinearInterpolator, LMMSEInterpolator,\
                                BaseChannelEstimator, BaseChannelInterpolator,\
                                tdl_freq_cov_mat, tdl_time_cov_mat
from .equalization import OFDMEqualizer, LMMSEEqualizer, ZFEqualizer,\
                          MFEqualizer, PostEqualizationSINR,\
                          LMMSEPostEqualizationSINR
from .detection import OFDMDetector, OFDMDetectorWithPrior,\
                       MaximumLikelihoodDetector,\
                       MaximumLikelihoodDetectorWithPrior, LinearDetector,\
                       KBestDetector, EPDetector, MMSEPICDetector
from .precoding import RZFPrecoder, PrecodedChannel, RZFPrecodedChannel,\
                       CBFPrecodedChannel, EyePrecodedChannel
