#
# SPDX-FileCopyrightText: Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""
Ray tracing module of Sionna.
"""


###########################################
# Configuring Mitsuba variant
###########################################

import tensorflow as tf
import mitsuba as mi

# If at least one GPU is detected, the CUDA variant is used.
# Otherwise, the LLVM variant is used.
# Note: LLVM is required for execution on CPU.
# Note: If multiple GPUs are visible, the first one is used.
gpus = tf.config.list_physical_devices('GPU')
if len(gpus) > 0:
    mi.set_variant('cuda_ad_rgb')
else:
    mi.set_variant('llvm_ad_rgb')

############################################
# Objects available to the user
############################################
# pylint: disable=wrong-import-position
from .scene import load_scene, Scene
from .camera import Camera
from .antenna import Antenna, compute_gain, visualize, iso_pattern,\
                     dipole_pattern, hw_dipole_pattern, tr38901_pattern
from .antenna_array import AntennaArray, PlanarArray
from .radio_material import RadioMaterial
from .scene_object import SceneObject
from .transmitter import Transmitter
from .receiver import Receiver
from .paths_2_cir import Paths2CIR
from .paths import Paths
from .coverage_map import CoverageMap
from .utils import rotation_matrix, rotate, theta_phi_from_unit_vec,\
                  r_hat, theta_hat, phi_hat, cross, dot,\
                  normalize
