#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Smart random number generation utilities for torch.compile compatibility.

These functions automatically switch between using a generator (for reproducibility
in eager mode) and global RNG state (for graph fusion in compiled mode).
Outputs default to :attr:`sionna.phy.config.Config.device`, and an omitted
generator defaults to :meth:`sionna.phy.config.Config.torch_rng` for that device,
so :attr:`sionna.phy.config.Config.seed` controls them. An explicit generator
must belong to the resolved output device.

Note: For proper multi-device reproducibility, use `sionna.phy.config.seed` instead
of `torch.manual_seed()`. The config seeds each device's default generator with a
device-specific offset, ensuring different devices produce different random streams.
"""

import math
from typing import Optional, Sequence, Union
import torch

from sionna.phy.config import config, dtypes, Precision

__all__ = ["randint", "rand", "uniform", "normal", "complex_normal"]


def _resolve_device(
    device: Optional[Union[str, torch.device]],
) -> Union[str, torch.device]:
    """Resolve an omitted device to Sionna's configured device."""
    return config.device if device is None else device


def _resolve_generator(
    generator: Optional[torch.Generator],
    device: Union[str, torch.device],
) -> Optional[torch.Generator]:
    """Resolve an omitted generator to Sionna's generator for ``device``.

    Returns `None` while compiling, where a generator cannot be captured in
    the graph, so an explicit one is ignored and the global RNG is used.
    """
    if torch.compiler.is_compiling():
        return None
    if generator is not None:
        return generator
    resolved = torch.device(device)
    if resolved.type == "cuda" and resolved.index is None:
        resolved = torch.device("cuda", torch.cuda.current_device())
    return config.torch_rng(str(resolved))


def _shape_template(
    size: Sequence[int],
    dtype: torch.dtype,
    device: Union[str, torch.device],
) -> torch.Tensor:
    """Create an expanded scalar template that supports symbolic dimensions."""
    return torch.empty((), dtype=dtype, device=device).expand(tuple(size))


def randint(
    low: int,
    high: int,
    size: Sequence[int],
    *,
    dtype: Optional[torch.dtype] = None,
    device: Optional[Union[str, torch.device]] = None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Generate random integer tensor, compile-aware.

    In eager mode, uses the provided generator for reproducibility.
    In compiled mode, uses global RNG state for graph fusion.

    :param low: Minimum value (inclusive).
    :param high: Maximum value (exclusive).
    :param size: Shape of the output tensor.
    :param dtype: Data type of the output tensor.
    :param device: Device for the output tensor. If `None`,
        :attr:`~sionna.phy.config.Config.device` is used.
    :param generator: Random number generator. If `None`,
        :meth:`~sionna.phy.config.Config.torch_rng` for the output device is
        used. Ignored in compiled mode, which uses the global RNG.

    :output samples: Tensor with random integer values.
    """
    device = _resolve_device(device)
    generator = _resolve_generator(generator, device)
    if torch.compiler.is_compiling():
        output_dtype = torch.int64 if dtype is None else dtype
        template = _shape_template(size, output_dtype, device)
        return torch.randint_like(template, low, high, dtype=output_dtype)
    return torch.randint(
        low, high, size, dtype=dtype, device=device, generator=generator
    )


def rand(
    size: Sequence[int],
    *,
    dtype: Optional[torch.dtype] = None,
    device: Optional[Union[str, torch.device]] = None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Generate random uniform tensor [0, 1), compile-aware.

    In eager mode, uses the provided generator for reproducibility.
    In compiled mode, uses global RNG state for graph fusion.

    :param size: Shape of the output tensor.
    :param dtype: Data type of the output tensor.
    :param device: Device for the output tensor. If `None`,
        :attr:`~sionna.phy.config.Config.device` is used.
    :param generator: Random number generator. If `None`,
        :meth:`~sionna.phy.config.Config.torch_rng` for the output device is
        used. Ignored in compiled mode, which uses the global RNG.

    :output samples: Tensor with random uniform values.
    """
    device = _resolve_device(device)
    generator = _resolve_generator(generator, device)
    if torch.compiler.is_compiling():
        output_dtype = torch.get_default_dtype() if dtype is None else dtype
        template = _shape_template(size, output_dtype, device)
        return torch.rand_like(template)
    return torch.rand(size, dtype=dtype, device=device, generator=generator)


def uniform(
    size: Sequence[int],
    *,
    low: float = 0.0,
    high: float = 1.0,
    dtype: Optional[torch.dtype] = None,
    device: Optional[Union[str, torch.device]] = None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Generate random uniform tensor in [low, high), compile-aware.

    In eager mode, uses the provided generator for reproducibility.
    In compiled mode, uses global RNG state for graph fusion.

    :param size: Shape of the output tensor.
    :param low: Lower bound (inclusive). Defaults to 0.0.
    :param high: Upper bound (exclusive). Defaults to 1.0.
    :param dtype: Data type of the output tensor.
    :param device: Device for the output tensor. If `None`,
        :attr:`~sionna.phy.config.Config.device` is used.
    :param generator: Random number generator. If `None`,
        :meth:`~sionna.phy.config.Config.torch_rng` for the output device is
        used. Ignored in compiled mode, which uses the global RNG.

    :output samples: Tensor with random uniform values in [low, high).
    """
    result = rand(size, dtype=dtype, device=device, generator=generator)
    if low != 0.0 or high != 1.0:
        result = result * (high - low) + low
    return result


def normal(
    size: Sequence[int],
    *,
    mean: float = 0.0,
    std: float = 1.0,
    dtype: Optional[torch.dtype] = None,
    device: Optional[Union[str, torch.device]] = None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Generate random normal tensor, compile-aware.

    In eager mode, uses the provided generator for reproducibility.
    In compiled mode, uses torch.randn (which uses the Graph RNG) to ensure
    proper synchronization with other random operations like randint.

    Note: torch.normal uses a different RNG stream than torch.randn under
    torch.compile, which can cause training issues. Using torch.randn with
    scaling ensures consistent RNG behavior.

    :param size: Shape of the output tensor.
    :param mean: Mean of the distribution. Defaults to 0.0.
    :param std: Standard deviation of the distribution. Defaults to 1.0.
    :param dtype: Data type of the output tensor.
    :param device: Device for the output tensor. If `None`,
        :attr:`~sionna.phy.config.Config.device` is used.
    :param generator: Random number generator. If `None`,
        :meth:`~sionna.phy.config.Config.torch_rng` for the output device is
        used. Ignored in compiled mode, which uses the global RNG.

    :output samples: Tensor with random normal values.
    """
    device = _resolve_device(device)
    generator = _resolve_generator(generator, device)
    if torch.compiler.is_compiling():
        # Use the Graph RNG (same as randint), then apply mean/std.
        # This avoids the RNG desynchronization issue with torch.normal.
        output_dtype = torch.get_default_dtype() if dtype is None else dtype
        template = _shape_template(size, output_dtype, device)
        result = torch.randn_like(template)
        if std != 1.0:
            result = result * std
        if mean != 0.0:
            result = result + mean
        return result
    else:
        return torch.normal(
            mean=mean,
            std=std,
            size=size,
            dtype=dtype,
            device=device,
            generator=generator,
        )


def complex_normal(
    shape: Sequence[int],
    var: float = 1.0,
    *,
    precision: Optional[Precision] = None,
    device: Optional[Union[str, torch.device]] = None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Generate a complex normal random tensor, compile-aware.

    Generates circularly symmetric complex Gaussian random variables with total
    variance ``var`` (i.e., variance ``var/2`` per real and imaginary component).

    In eager mode, uses the provided generator for reproducibility.
    In compiled mode, uses global RNG state for graph fusion.

    :param shape: Shape of the output tensor.
    :param var: Total variance, finite and non-negative. Defaults to 1.0.
    :param precision: Precision used for the output tensor.
        If set to `None`, :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for the output tensor. If `None`,
        :attr:`~sionna.phy.config.Config.device` is used.
    :param generator: Random number generator. If `None`,
        :meth:`~sionna.phy.config.Config.torch_rng` for the output device is
        used. Ignored in compiled mode, which uses the global RNG.

    :output samples: Complex tensor with complex normal values.

    .. rubric:: Examples

    .. code-block:: python

        from sionna.phy.utils import complex_normal

        x = complex_normal([2, 3], var=2.0)
        print(x.shape)
        # torch.Size([2, 3])
    """
    if not torch.compiler.is_compiling():
        if not math.isfinite(var) or var < 0.0:
            raise ValueError("var must be a finite non-negative number.")

    # Determine dtype from precision
    if precision is None:
        dtype = config.dtype
    else:
        dtype = dtypes[precision]["torch"]["dtype"]

    device = _resolve_device(device)
    generator = _resolve_generator(generator, device)

    # Generate real and imaginary parts with half the total variance each.
    std = math.sqrt(var / 2.0)
    real = normal(
        shape, std=std, dtype=dtype, device=device, generator=generator
    )
    imag = normal(
        shape, std=std, dtype=dtype, device=device, generator=generator
    )
    return torch.complex(real, imag)
