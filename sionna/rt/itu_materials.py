#
# SPDX-FileCopyrightText: Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""
Instantiates a set of radio materials for the scene objects.
These materials are from Table 3 of the Recommendation ITU-R P.2040-2.
"""

import numpy as np
from .radio_material import RadioMaterial
from . import scene

def instantiate_itu_materials(dtype):
    #########################################
    # Vacuum (~ air)
    #########################################

    def vacuum_properties(f_hz): # pylint: disable=unused-argument
        return (1.0, 0.0)

    rm = RadioMaterial("vacuum",
                       frequency_update_callback=vacuum_properties,
                       dtype=dtype)
    scene.Scene().add(rm)


    #########################################
    # Concrete
    #########################################

    def concrete_properties(f_hz):
        f_ghz = f_hz / 1e9
        if f_ghz < 1.0 or f_ghz > 100.:
            return (-1.0, -1.0)

        relative_permittivity = 5.24
        conductivity = 0.0462*np.power(f_ghz, 0.7822)
        return (relative_permittivity, conductivity)

    # Materials parameters will be updated when the frequency is set
    rm = RadioMaterial("itu_concrete",
                      frequency_update_callback=concrete_properties,
                      dtype=dtype)
    scene.Scene().add(rm)

    ##########################################
    # Brick
    ##########################################

    def brick_properties(f_hz):
        f_ghz = f_hz / 1e9
        if f_ghz < 1.0 or f_ghz > 40.:
            return (-1.0, -1.0)

        relative_permittivity = 3.91
        conductivity = 0.0238*np.power(f_ghz, 0.16)
        return (relative_permittivity, conductivity)

    # Materials parameters will be updated when the frequency is set
    rm = RadioMaterial("itu_brick",
                       frequency_update_callback=brick_properties,
                       dtype=dtype)
    scene.Scene().add(rm)

    #########################################
    # Plasterboard
    #########################################

    def plasterboard_properties(f_hz):
        f_ghz = f_hz / 1e9
        if f_ghz < 1.0 or f_ghz > 100.:
            return (-1.0, -1.0)


        relative_permittivity = 2.73
        conductivity = 0.0085*np.power(f_ghz, 0.9395)
        return (relative_permittivity, conductivity)

    # Materials parameters will be updated when the frequency is set
    rm = RadioMaterial("itu_plasterboard",
                       frequency_update_callback=plasterboard_properties,
                       dtype=dtype)
    scene.Scene().add(rm)

    #########################################
    # Wood
    #########################################

    def wood_properties(f_hz):
        f_ghz = f_hz / 1e9
        if f_ghz < 0.001 or f_ghz > 100.:
            return (-1.0, -1.0)

        relative_permittivity = 1.99
        conductivity = 0.0047*np.power(f_ghz, 1.0718)
        return (relative_permittivity, conductivity)

    # Materials parameters will be updated when the frequency is set
    rm = RadioMaterial("itu_wood",
                       frequency_update_callback=wood_properties,
                       dtype=dtype)
    scene.Scene().add(rm)

    #########################################
    # Glass
    #########################################

    def glass_properties(f_hz):
        f_ghz = f_hz / 1e9
        if 0.1 <= f_ghz <= 100.:
            relative_permittivity = 6.31
            conductivity = 0.0036*np.power(f_ghz, 1.3394)
            return (relative_permittivity, conductivity)
        elif 220. <= f_ghz <= 450.:
            relative_permittivity = 5.79
            conductivity = 0.0004*np.power(f_ghz, 1.658)
            return (relative_permittivity, conductivity)
        else:
            return (-1.0, -1.0)

    # Materials parameters will be updated when the frequency is set
    rm = RadioMaterial("itu_glass",
                       frequency_update_callback=glass_properties,
                       dtype=dtype)
    scene.Scene().add(rm)

    #########################################
    # Ceiling board
    #########################################

    def ceiling_board_properties(f_hz):
        f_ghz = f_hz / 1e9
        if 1. <= f_ghz <= 100.:
            relative_permittivity = 1.48
            conductivity = 0.0011*np.power(f_ghz, 1.0750)
            return (relative_permittivity, conductivity)
        elif 220. <= f_ghz <= 450.:
            relative_permittivity = 1.52
            conductivity = 0.0029*np.power(f_ghz, 1.029)
            return (relative_permittivity, conductivity)
        else:
            return (-1.0, -1.0)

    # Materials parameters will be updated when the frequency is set
    rm = RadioMaterial("itu_ceiling_board",
                       frequency_update_callback=ceiling_board_properties,
                       dtype=dtype)
    scene.Scene().add(rm)

    #########################################
    # Chipboard
    #########################################

    def chipboard_properties(f_hz):
        f_ghz = f_hz / 1e9
        if f_ghz < 1.0 or f_ghz > 100.0:
            return (-1.0, -1.0)

        relative_permittivity = 2.58
        conductivity = 0.0217*np.power(f_ghz, 0.7800)
        return (relative_permittivity, conductivity)

    # Materials parameters will be updated when the frequency is set
    rm = RadioMaterial("itu_chipboard",
                       frequency_update_callback=chipboard_properties,
                       dtype=dtype)
    scene.Scene().add(rm)

    #########################################
    # Plywood
    #########################################

    def plywood_properties(f_hz):
        f_ghz = f_hz / 1e9
        if f_ghz < 1.0 or f_ghz > 40.0:
            return (-1.0, -1.0)

        relative_permittivity = 2.71
        conductivity = 0.33
        return (relative_permittivity, conductivity)

    # Materials parameters will be updated when the frequency is set
    rm = RadioMaterial("itu_plywood",
                       frequency_update_callback=plywood_properties,
                       dtype=dtype)
    scene.Scene().add(rm)

    #########################################
    # Marble
    #########################################

    def marble_properties(f_hz):
        f_ghz = f_hz / 1e9
        if f_ghz < 1.0 or f_ghz > 60.0:
            return (-1.0, -1.0)

        relative_permittivity = 7.074
        conductivity = 0.0055*np.power(f_ghz, 0.9262)
        return (relative_permittivity, conductivity)

    # Materials parameters will be updated when the frequency is set
    rm = RadioMaterial("itu_marble",
                       frequency_update_callback=marble_properties,
                       dtype=dtype)
    scene.Scene().add(rm)

    #########################################
    # Floorboard
    #########################################

    def floorboard_properties(f_hz):
        f_ghz = f_hz / 1e9
        if f_ghz < 50.0 or f_ghz > 100.0:
            return (-1.0, -1.0)

        relative_permittivity = 3.66
        conductivity = 0.0044*np.power(f_ghz, 1.3515)
        return (relative_permittivity, conductivity)

    # Materials parameters will be updated when the frequency is set
    rm = RadioMaterial("itu_floorboard",
                       frequency_update_callback=floorboard_properties,
                       dtype=dtype)
    scene.Scene().add(rm)

    #########################################
    # Metal
    #########################################

    def metal_properties(f_hz):
        f_ghz = f_hz / 1e9
        if f_ghz < 1.0 or f_ghz > 100.0:
            return (-1.0, -1.0)

        relative_permittivity = 1.0
        conductivity = 1e7
        return (relative_permittivity, conductivity)

    # Materials parameters will be updated when the frequency is set
    rm = RadioMaterial("itu_metal",
                       frequency_update_callback=metal_properties,
                       dtype=dtype)
    scene.Scene().add(rm)

    #########################################
    # Very dry ground
    #########################################

    def very_dry_ground_properties(f_hz):
        f_ghz = f_hz / 1e9
        if f_ghz < 1.0 or f_ghz > 10.0:
            return (-1.0, -1.0)

        relative_permittivity = 3.0
        conductivity = 0.00015*np.power(f_ghz, 2.52)
        return (relative_permittivity, conductivity)

    # Materials parameters will be updated when the frequency is set
    rm = RadioMaterial("itu_very_dry_ground",
                       frequency_update_callback=very_dry_ground_properties,
                       dtype=dtype)
    scene.Scene().add(rm)

    #########################################
    # Medium dry ground
    #########################################

    def medium_dry_ground_properties(f_hz):
        f_ghz = f_hz / 1e9
        if f_ghz < 1.0 or f_ghz > 10.0:
            return (-1.0, -1.0)

        relative_permittivity = 15.0*np.power(f_ghz, -0.1)
        conductivity = 0.035*np.power(f_ghz, 1.63)
        return (relative_permittivity, conductivity)

    # Materials parameters will be updated when the frequency is set
    rm = RadioMaterial("itu_medium_dry_ground",
                       frequency_update_callback=medium_dry_ground_properties,
                       dtype=dtype)
    scene.Scene().add(rm)

    #########################################
    # Wet ground
    #########################################

    def wet_ground_properties(f_hz):
        f_ghz = f_hz / 1e9
        if f_ghz < 1.0 or f_ghz > 10.0:
            return (-1.0, -1.0)

        relative_permittivity = 30.0*np.power(f_ghz, -0.4)
        conductivity = 0.15*np.power(f_ghz, 1.30)
        return (relative_permittivity, conductivity)

    # Materials parameters will be updated when the frequency is set
    rm = RadioMaterial("itu_wet_ground",
                       frequency_update_callback=wet_ground_properties,
                       dtype=dtype)
    scene.Scene().add(rm)
