#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Definition of Sionna Object."""

import random
from typing import Any, Optional
import torch
import numpy as np
from .config import config, dtypes, Precision

__all__ = ["Object"]


class Object(torch.nn.Module):
    """Base class for Sionna PHY objects.

    :param precision: Floating-point precision ('single' or 'double') to be used within the block.
        If `None`, :attr:`~sionna.phy.config.Config.precision` is used.
        Defaults to `None`.
    :param device: Device for computation (e.g., 'cpu', 'cuda:0') to be used within the block.
        If `None`, :attr:`~sionna.phy.config.Config.device` is used.
        Defaults to `None`.
    """

    def __init__(
        self,
        *args: Any,
        precision: Optional[str] = None,
        device: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the object."""
        # Initialize nn.Module first
        super().__init__()

        if precision is not None and precision not in dtypes:
            raise ValueError(f"Invalid precision: {precision}")
        if device is not None and device not in config.available_devices:
            raise ValueError(f"Invalid device: {device}")
        self._precision: Precision = (
            config.precision if precision is None else precision
        )
        # Use _device_str to avoid conflict with nn.Module internals
        self._device_str: str = config.device if device is None else device

    @property
    def dtype(self) -> torch.dtype:
        """Get the PyTorch real-valued dtype based on the current precision."""
        return dtypes[self.precision]["torch"]["dtype"]

    @property
    def cdtype(self) -> torch.dtype:
        """Get the PyTorch complex-valued dtype based on the current precision."""
        return dtypes[self.precision]["torch"]["cdtype"]

    @property
    def np_dtype(self) -> type:
        """Get the NumPy real-valued dtype based on the current precision."""
        return dtypes[self.precision]["np"]["dtype"]

    @property
    def np_cdtype(self) -> type:
        """Get the NumPy complex-valued dtype based on the current precision."""
        return dtypes[self.precision]["np"]["cdtype"]

    @property
    def precision(self) -> Precision:
        """Get the floating-point precision ('single' or 'double')."""
        return self._precision

    @property
    def device(self) -> str:
        """Get the device for computation (e.g., 'cpu', 'cuda:0')."""
        return self._device_str

    @property
    def torch_rng(self) -> torch.Generator:
        """Get the PyTorch random number generator for the object's device."""
        return config.torch_rng(self.device)

    @property
    def np_rng(self) -> np.random.Generator:
        """Get the NumPy random number generator."""
        return config.np_rng

    @property
    def py_rng(self) -> random.Random:
        """Get the Python random number generator."""
        return config.py_rng

    def _convert(self, v: Any) -> Any:
        # None stays None
        if v is None:
            return None

        # Handle recursion for lists/tuples/dicts
        if isinstance(v, (list, tuple)):
            return type(v)(self._convert(x) for x in v)
        if isinstance(v, dict):
            return {k: self._convert(val) for k, val in v.items()}

        # Strings and ints stay as-is (ints often used for shapes/indices)
        if isinstance(v, (str, int)):
            return v

        # Convert floats and complex to tensors (data values)
        # Also convert numpy arrays and other array-like objects
        if not isinstance(v, torch.Tensor):
            v = torch.as_tensor(v, device=self._device_str)

        # Determine target dtype
        if v.is_complex():
            target_dtype = self.cdtype
        elif v.is_floating_point():
            target_dtype = self.dtype
        else:
            # Keep integer/boolean dtypes unchanged
            target_dtype = v.dtype

        # Only call .to() if conversion is needed
        if v.device != torch.device(self.device) or v.dtype != target_dtype:
            v = v.to(device=self.device, dtype=target_dtype)

        return v

    def _get_shape(self, v):
        """Extracts shape tuple if available."""
        # Handle recursion
        if isinstance(v, (list, tuple)):
            return type(v)(self._get_shape(x) for x in v)
        if isinstance(v, dict):
            return {k: self._get_shape(val) for k, val in v.items()}

        if hasattr(v, "shape"):
            return tuple(v.shape)
        return ()

    def _check_device(self, name: str, child: "Object") -> None:
        """Reject a sub-object that lives on a different device.

        Tensors created by the child stay on the child's device, so the
        mismatch would otherwise surface much later as a bare PyTorch error
        that names two devices but not the object responsible. Precision is
        deliberately not checked here because mixed-precision composition can
        be intentional and does not create cross-device runtime errors.
        """
        # Nothing to compare against until Object.__init__ has run.
        parent_device = self.__dict__.get("_device_str")
        child_device = child.__dict__.get("_device_str")
        if parent_device is None or child_device is None:
            return
        if child_device != parent_device:
            attr = name.lstrip("_")
            raise ValueError(
                f"{type(self).__name__} is on device {parent_device!r}, but its "
                f"{attr} ({type(child).__name__}) is on device {child_device!r}. "
                f"Build both on the same device, e.g. "
                f"{type(child).__name__}(..., device={parent_device!r})."
            )

    def _check_contained_devices(
        self, name: str, value: Any, seen: Optional[set[int]] = None
    ) -> None:
        """Recursively validate Object instances held by common containers."""
        if isinstance(value, Object):
            self._check_device(name, value)
            return
        container_types = (
            dict,
            list,
            tuple,
            set,
            torch.nn.ModuleDict,
            torch.nn.ModuleList,
        )
        if not isinstance(value, container_types):
            return
        if seen is None:
            seen = set()
        value_id = id(value)
        if value_id in seen:
            return
        seen.add(value_id)
        if isinstance(value, (dict, torch.nn.ModuleDict)):
            for key, child in value.items():
                self._check_contained_devices(f"{name}[{key!r}]", child, seen)
        else:
            for index, child in enumerate(value):
                self._check_contained_devices(f"{name}[{index}]", child, seen)

    def __setattr__(self, name: str, value: Any) -> None:
        """Override to ensure property setters are called even for nn.Module values.

        PyTorch's nn.Module.__setattr__ intercepts nn.Module assignments and
        registers them in _modules, bypassing property setters. This override
        checks if there's a property descriptor with a setter on the class and
        uses it instead.
        """
        self._check_contained_devices(name, value)
        cls = type(self)
        descriptor = getattr(cls, name, None)
        if (
            descriptor is not None
            and isinstance(descriptor, property)
            and descriptor.fset is not None
        ):
            # Use the property setter directly
            descriptor.fset(self, value)
        else:
            # Fall back to nn.Module's default behavior
            super().__setattr__(name, value)
