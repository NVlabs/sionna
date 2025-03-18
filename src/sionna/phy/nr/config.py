#
# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0#
"""Abstract class for configuration of the NR (5G) module of Sionna PHY"""

from abc import ABC
import copy
import numpy as np

class Config(ABC):
    # pylint: disable=line-too-long
    """Abstract configuration class for the nr (5G) sub-package of Sionna PHY
    """
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if key in dir(self):
                setattr(self, key, value)

    def _ifndef(self, name, value):
        if not hasattr(self, f"_{name}"):
            setattr(self, f"_{name}", value)

    def clone(self, deep=True):
        """Returns a copy of the Config object

        Input
        -----
        deep : `bool`, (default `True`)
            If `True`, a deep copy will be returned.
        """
        if deep:
            return copy.deepcopy(self)
        else:
            return copy.copy(self)

    def check_config(self):
        pass

    def show(self):
        """Print all properties of a configuration"""
        self.check_config()
        print(self._name)
        print("="*len(self._name))
        for a in dir(self):
            val = getattr(self, a)
            if a[0]!="_" and a not in ["show", "name", "check_config", \
                                       "check_config_precoded", "clone", \
                                       "c_init", "dmrs", "tb", "carrier"]:
                if a in ["dmrs_grid", "dmrs_grid_precoded", "dmrs_mask", "n"]:
                    print(f"{a} : shape {np.array(val).shape}")
                else:
                    print(f"{a} : {val}")
        print("\r")
