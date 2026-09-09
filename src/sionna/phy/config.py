#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Global Sionna PHY Configuration"""

from __future__ import annotations
import random
from numbers import Integral
from typing import Literal
import numpy as np
import torch

__all__ = ["Config", "config", "dtypes", "Precision"]

_SEED_MODULUS = 1 << 64

# Type aliases
Precision = Literal["single", "double"]

# Mapping from precision to dtypes
dtypes: dict[str, dict[str, dict[str, type]]] = {
    "single": {
        "torch": {"cdtype": torch.complex64, "dtype": torch.float32},
        "np": {"cdtype": np.complex64, "dtype": np.float32},
    },
    "double": {
        "torch": {"cdtype": torch.complex128, "dtype": torch.float64},
        "np": {"cdtype": np.complex128, "dtype": np.float64},
    },
}


class Config:
    """Sionna PHY Configuration Class

    This singleton class is used to define global configuration variables
    and random number generators that can be accessed from all modules
    and functions. It is instantiated immediately and its properties can be
    accessed as :code:`sionna.phy.config.desired_property`.
    """

    _instance: Config | None = None

    def __new__(cls) -> Config:
        """Create or return the singleton Config instance."""
        if cls._instance is None:
            instance = object.__new__(cls)
            cls._instance = instance
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        """Initialize the Config instance with default values."""
        if self._initialized:
            return
        self._initialized: bool = True

        # Initialize private properties
        self._precision: Precision | None = None
        self._device: str | None = None
        self._seed: int | None = None
        self._py_rng: random.Random | None = None
        self._np_rng: np.random.Generator | None = None
        self._torch_rngs: dict[str, torch.Generator] = {}
        self._np_dtype: type | None = None
        self._np_cdtype: type | None = None
        self._dtype: torch.dtype | None = None
        self._cdtype: torch.dtype | None = None

        # Set default property values
        self.precision = "single"
        self.device = None
        self.seed = None

    @property
    def py_rng(self) -> random.Random:
        """`random.Random` : Python random number generator

        Example
        -------

        .. code-block:: python

            from sionna.phy import config
            config.seed = 42 # Set seed for deterministic results

            # Use generator instead of random
            val = config.py_rng.randint(0, 10)
        """
        if self._py_rng is None:
            self._py_rng = random.Random(self.seed)
        return self._py_rng

    @property
    def np_rng(self) -> np.random.Generator:
        """`np.random.Generator` : NumPy random number generator

        Example
        -------

        .. code-block:: python

            from sionna.phy import config
            config.seed = 42 # Set seed for deterministic results

            # Use generator instead of np.random
            noise = config.np_rng.normal(size=[4])
        """
        if self._np_rng is None:
            self._np_rng = np.random.default_rng(self.seed)
        return self._np_rng

    def torch_rng(self, device: str | None = None) -> torch.Generator:
        """`torch.Generator` : PyTorch random number generator for the specified device

        :param device: Device name (e.g., ``'cpu'``, ``'cuda:0'``).
            If `None`, :attr:`~sionna.phy.config.Config.device` is used.

        Example
        -------

        .. code-block:: python

            import torch
            from sionna.phy import config
            config.seed = 42 # Set seed for deterministic results

            # Explicit generators must match the output device
            noise = torch.randn(
                [4],
                device=config.device,
                generator=config.torch_rng(config.device),
            )
        """
        if device is None:
            device = self.device
        return self._torch_rngs[device]

    def _reset_rngs(self) -> None:
        """Reset all random number generators."""
        # Uses device-specific seed offsets (seed + device_index) to ensure
        # different devices produce different random streams. This applies to
        # both the explicit generators (used in eager mode) and the default
        # CUDA generators (used in compiled mode via global RNG).
        self._py_rng = None
        self._np_rng = None
        self._torch_rngs = {}

        # Initialize CUDA to populate default_generators if available
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            torch.cuda.init()

        for i, device in enumerate(self.available_devices):
            self._torch_rngs[device] = torch.Generator(device=device)

            if self._seed is None:
                # Random seeding - each device gets a random seed
                self._torch_rngs[device].seed()
                # Also seed default generators randomly
                if device == "cpu":
                    torch.default_generator.seed()
                elif device.startswith("cuda:"):
                    device_idx = int(device.split(":")[1])
                    # Check if default generators are initialized and index is valid
                    if torch.cuda.is_available() and device_idx < len(
                        torch.cuda.default_generators
                    ):
                        torch.cuda.default_generators[device_idx].seed()
            else:
                # Deterministic seeding with device-specific offset
                # This ensures different devices produce different streams
                device_seed = (self._seed + i) % _SEED_MODULUS
                self._torch_rngs[device].manual_seed(device_seed)
                # Also seed default generators for compiled mode
                if device == "cpu":
                    torch.default_generator.manual_seed(device_seed)
                elif device.startswith("cuda:"):
                    device_idx = int(device.split(":")[1])
                    # Check if default generators are initialized and index is valid
                    if torch.cuda.is_available() and device_idx < len(
                        torch.cuda.default_generators
                    ):
                        torch.cuda.default_generators[device_idx].manual_seed(
                            device_seed
                        )

    @property
    def seed(self) -> int | None:
        """`None` (default) | `int` : Get/set seed for all random number generators

        All random number generators used internally by Sionna
        can be configured with a common seed to ensure reproducibility
        of results. It defaults to `None` which implies that a random
        seed will be used and results are non-deterministic. Integer seeds
        must be in the interval :math:`[0, 2^{64}-1]`.

        Example
        -------

        .. code-block:: python

            # This code will lead to deterministic results
            from sionna.phy import config
            from sionna.phy.mapping import BinarySource
            config.seed = 42
            print(BinarySource()([10]))
        """
        return self._seed

    @seed.setter
    def seed(self, seed: int | None) -> None:
        if seed is not None:
            if isinstance(seed, bool) or not isinstance(seed, Integral):
                raise TypeError("seed must be an integer or None")
            seed = int(seed)
            if not 0 <= seed < _SEED_MODULUS:
                raise ValueError("seed must be in the range [0, 2**64 - 1]")

        self._seed = seed
        self._reset_rngs()

    @property
    def device(self) -> str:
        """`str` : Get/set the device for computation (e.g., ``'cpu'``, ``'cuda:0'``)"""
        return self._device

    @device.setter
    def device(self, v: str | None) -> None:
        # Set default device if None
        if v is None:
            if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                v = "cuda:0"
            else:
                v = "cpu"

        # If device is already set to the desired value, do nothing
        if self._device == v:
            return

        # Raise error if device is invalid
        if v not in self.available_devices:
            raise ValueError(f"Invalid device: {v}")

        # Set device value
        self._device = v

    @property
    def np_dtype(self) -> np.dtype:
        """`np.dtype` : Default NumPy dtype for real floating point numbers"""
        return dtypes[self.precision]["np"]["dtype"]

    @property
    def np_cdtype(self) -> np.dtype:
        """`np.dtype` : Default NumPy dtype for complex floating point numbers"""
        return dtypes[self.precision]["np"]["cdtype"]

    @property
    def dtype(self) -> torch.dtype:
        """`torch.dtype` : Default PyTorch dtype for real floating point numbers"""
        return dtypes[self.precision]["torch"]["dtype"]

    @property
    def cdtype(self) -> torch.dtype:
        """`torch.dtype` : Default PyTorch dtype for complex floating point numbers"""
        return dtypes[self.precision]["torch"]["cdtype"]

    @property
    def precision(self) -> Precision:
        """``"single"`` (default) | ``"double"`` : Default precision used for all computations

        The ``"single"`` option represents real-valued floating-point numbers
        using 32 bits, whereas the ``"double"`` option uses 64 bits.
        For complex-valued data types, each component of the complex number
        (real and imaginary parts) uses either 32 bits (for ``"single"``)
        or 64 bits (for ``"double"``).
        """
        return self._precision

    @precision.setter
    def precision(self, v: Precision) -> None:
        if v not in dtypes:
            raise ValueError("Precision must be 'single' or 'double'.")
        self._precision = v

    @property
    def available_devices(self) -> list[str]:
        """`list` of `str` : List of available compute devices"""
        devices = ["cpu"]
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                devices.append(f"cuda:{i}")
        return devices


config = Config()
