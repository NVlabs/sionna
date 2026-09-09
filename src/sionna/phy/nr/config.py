#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Abstract class for configuration of the NR (5G) module of Sionna PHY."""

from abc import ABC
import copy
import numpy as np


__all__ = ["Config"]


class Config(ABC):
    """Abstract configuration class for the NR (5G) sub-package of Sionna PHY.

    All configurable properties can be provided as keyword arguments during
    initialization or changed later.
    """

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if key not in dir(self):
                raise TypeError(
                    f"{type(self).__name__} got an unexpected keyword "
                    f"argument '{key}'"
                )
            setattr(self, key, value)

    def _ifndef(self, name: str, value) -> None:
        """Set a default value for an attribute if it doesn't exist."""
        if not hasattr(self, f"_{name}"):
            setattr(self, f"_{name}", value)

    def clone(self, deep: bool = True) -> "Config":
        """Returns a copy of the Config object.

        :param deep: If `True`, a deep copy will be returned.

        :output config: Copy of the configuration.
        """
        if deep:
            return copy.deepcopy(self)
        else:
            return copy.copy(self)

    def check_config(self) -> None:
        """Validates the configuration. Override in subclasses."""
        pass

    def show(self) -> None:
        """Print all properties of a configuration."""
        self.check_config()
        print(self._name)
        print("=" * len(self._name))
        for a in dir(self):
            val = getattr(self, a)
            if a[0] != "_" and a not in [
                "show",
                "name",
                "check_config",
                "check_config_precoded",
                "clone",
                "c_init",
                "dmrs",
                "tb",
                "carrier",
            ]:
                if a in ["dmrs_grid", "dmrs_grid_precoded", "dmrs_mask", "n"]:
                    print(f"{a} : shape {np.array(val).shape}")
                else:
                    print(f"{a} : {val}")
        print("\r")

