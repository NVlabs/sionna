#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Constants for the Sionna PHY Package."""

from scipy.constants import Boltzmann, Planck, epsilon_0, pi, speed_of_light

__all__ = [
    "ALPHA_MAX",
    "BOLTZMANN_CONSTANT",
    "DIELECTRIC_PERMITTIVITY_VACUUM",
    "H",
    "PI",
    "SPEED_OF_LIGHT",
]

ALPHA_MAX = 32  # Maximum value
BOLTZMANN_CONSTANT = Boltzmann  # J/K
DIELECTRIC_PERMITTIVITY_VACUUM = epsilon_0  # F/m
H = Planck  # J/Hz
PI = pi
SPEED_OF_LIGHT = speed_of_light  # m/s
