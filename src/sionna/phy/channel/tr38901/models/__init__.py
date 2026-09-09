#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""3GPP channel-model parameter resources."""

import json
from importlib_resources import files
from importlib_resources.abc import Traversable

__all__ = [
    "fixed_tdl_parameter_file",
    "load_json",
    "parameter_file",
]

_VERSION_DIRS = {
    "16.1": "v16_1",
    "19.2": "v19_2",
}

_FIXED_TDL_DIR = "ts_38_101_4_v19_2_2"

def _validate_spec_version(spec_version: str) -> str:
    """Validate an exact supported TR 38.901 version label."""

    if not isinstance(spec_version, str) or spec_version not in _VERSION_DIRS:
        raise ValueError("spec_version must be '16.1' or '19.2'")
    return spec_version


def parameter_file(filename: str, spec_version: str = "19.2"):
    """Return a versioned TR 38.901 parameter resource.

    This function resolves scalable TDL/CDL and system-level scenario data
    under the selected TR 38.901 resource directory. Fixed-delay TS 38.101-4
    TDL profiles are deliberately stored separately and are not resolved by
    this function.

    The return value implements the ``importlib.resources`` ``Traversable``
    interface, so callers should use methods such as ``open()`` or
    ``read_bytes()`` instead of assuming a filesystem path. Resource existence
    is not checked by this function.

    :param filename: Resource name relative to the selected version directory.
    :param spec_version: Exact supported version label, either ``"16.1"`` or
        ``"19.2"``.

    :output resource: Traversable package resource for ``filename``.

    :raises ValueError: If ``spec_version`` is unsupported.
    """

    version = _validate_spec_version(spec_version)
    return files(__name__).joinpath(_VERSION_DIRS[version], filename)


def fixed_tdl_parameter_file(filename: str):
    """Return a fixed-profile TS 38.101-4 V19.2.2 resource.

    These canonical fixed-delay resources are independent of the TR 38.901
    ``spec_version`` selector. The caller remains responsible for the profile's
    frequency-range and channel-bandwidth applicability.
    """

    return files(__name__).joinpath(_FIXED_TDL_DIR, filename)


def load_json(source: Traversable) -> dict:
    """Load JSON from a package resource, including archive-backed resources.

    :param source: Traversable package resource containing JSON data.

    :output data: Decoded JSON object.
    """

    with source.open("r", encoding="utf-8") as parameter_file:
        return json.load(parameter_file)
