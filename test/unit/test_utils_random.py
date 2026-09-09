#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Unit tests for compile-aware random utilities."""

import inspect

import pytest
import torch

from sionna.phy import config
from sionna.phy.config import dtypes
from sionna.phy.utils import (
    complex_normal,
    normal,
    rand,
    randint,
    sample_bernoulli,
    uniform,
)
from sionna.phy.utils.misc import complex_normal as misc_complex_normal
from sionna.phy.utils.random import complex_normal as random_complex_normal


def test_complex_normal_is_single_public_function():
    """All supported import paths resolve to the canonical implementation."""
    assert complex_normal is random_complex_normal
    assert misc_complex_normal is random_complex_normal
    assert list(inspect.signature(complex_normal).parameters) == [
        "shape",
        "var",
        "precision",
        "device",
        "generator",
    ]


def test_complex_normal_rejects_size_keyword():
    """The accidental size keyword is not part of the public contract."""
    with pytest.raises(TypeError, match="unexpected keyword argument 'size'"):
        complex_normal(size=[2, 3])


@pytest.mark.parametrize(
    "factory",
    [
        lambda: randint(0, 4, [16]),
        lambda: rand([16]),
        lambda: uniform([16], low=-2.0, high=3.0),
        lambda: normal([16]),
        lambda: complex_normal([16]),
    ],
)
def test_default_device_follows_config(factory, device):
    """Omitted devices resolve to config.device for every helper."""
    assert factory().device == torch.device(device)


def test_explicit_cpu_overrides_gpu_config(device):
    """An explicit device takes precedence over config.device."""
    assert rand([4], device="cpu").device.type == "cpu"
    assert complex_normal([4], device="cpu").device.type == "cpu"


def test_config_generator_with_omitted_device(device):
    """The documented config generator pattern resolves a matching device."""
    samples = rand([8], generator=config.torch_rng())
    assert samples.device == torch.device(device)


@pytest.mark.parametrize("factory", [rand, normal])
def test_explicit_generator_is_reproducible(factory, device):
    """Eager helpers honor an explicit, matching generator."""
    generator_a = torch.Generator(device=device).manual_seed(123)
    generator_b = torch.Generator(device=device).manual_seed(123)
    samples_a = factory([128], device=device, generator=generator_a)
    samples_b = factory([128], device=device, generator=generator_b)
    assert torch.equal(samples_a, samples_b)


def test_complex_normal_generator_and_variance(device):
    """The canonical function combines generator and variance support."""
    generator_a = torch.Generator(device=device).manual_seed(123)
    generator_b = torch.Generator(device=device).manual_seed(123)
    samples_a = complex_normal(
        [200_000], var=2.0, device=device, generator=generator_a
    )
    samples_b = complex_normal(
        [200_000], var=2.0, device=device, generator=generator_b
    )
    assert torch.equal(samples_a, samples_b)
    assert torch.isclose(
        torch.mean(torch.abs(samples_a) ** 2),
        torch.tensor(2.0, device=device),
        rtol=0.02,
    )


@pytest.mark.parametrize("var", [-1.0, float("nan"), float("inf")])
def test_complex_normal_rejects_invalid_var(var, device):
    """An invalid variance is named instead of surfacing as a math error."""
    with pytest.raises(ValueError, match="var"):
        complex_normal([4], var=var, device=device)


@pytest.mark.parametrize(
    "factory",
    [
        lambda **kw: randint(0, 1_000, [8], **kw),
        lambda **kw: rand([8], **kw),
        lambda **kw: uniform([8], low=-2.0, high=3.0, **kw),
        lambda **kw: normal([8], **kw),
        lambda **kw: complex_normal([8], **kw),
    ],
)
def test_omitted_generator_defaults_to_config_generator(factory, device):
    """An omitted generator draws from config.torch_rng, not the global RNG."""
    config.seed = 123
    expected = factory(device=device)

    # Unrelated global RNG consumption does not shift the output.
    config.seed = 123
    torch.randn(1_000, device=device)
    assert torch.equal(factory(device=device), expected)

    # Omitting the generator matches passing the configured one explicitly.
    config.seed = 123
    assert torch.equal(
        factory(device=device, generator=config.torch_rng(device)), expected
    )


def test_complex_normal_precision(precision, device):
    """The configured precision controls the complex output dtype."""
    samples = complex_normal([8], precision=precision, device=device)
    assert samples.dtype == dtypes[precision]["torch"]["cdtype"]


def test_mismatched_generator_fails(device):
    """PyTorch reports an explicit generator/output-device mismatch."""
    if not torch.cuda.is_available():
        pytest.skip("A mismatch needs a second device to be detectable")
    # A CPU generator matches a CPU output, so pick the other device.
    other = "cpu" if torch.device(device).type == "cuda" else "cuda:0"
    generator = torch.Generator(device=other).manual_seed(123)
    with pytest.raises(RuntimeError, match="generator"):
        rand([4], device=device, generator=generator)


def test_ranges_and_dtypes(device):
    """The basic distribution and dtype contracts are preserved."""
    integers = randint(2, 7, [1_000], dtype=torch.int32, device=device)
    samples = uniform(
        [1_000], low=-3.0, high=-1.0, dtype=torch.float64, device=device
    )
    assert integers.dtype == torch.int32
    assert torch.all((integers >= 2) & (integers < 7))
    assert samples.dtype == torch.float64
    assert torch.all((samples >= -3.0) & (samples < -1.0))


@pytest.mark.parametrize("with_generator", [False, True])
def test_helpers_compile_fullgraph(with_generator, device):
    """All helpers share one full graph; an explicit generator is ignored.

    A `Generator` cannot be traced, so passing one must not break compilation.
    """
    generator = (
        torch.Generator(device=device).manual_seed(7) if with_generator else None
    )

    @torch.compile(fullgraph=True)
    def sample_all():
        return (
            randint(0, 4, [8], generator=generator),
            rand([8], generator=generator),
            uniform([8], generator=generator),
            normal([8], generator=generator),
            complex_normal([8], var=2.0, generator=generator),
        )

    outputs = sample_all()
    assert all(output.shape == (8,) for output in outputs)
    assert all(output.device == torch.device(device) for output in outputs)
    assert outputs[-1].is_complex()


def test_helpers_compile_with_symbolic_sizes(device):
    """Random helpers must accept dimensions derived from compiled inputs."""

    @torch.compile(fullgraph=True, dynamic=True)
    def sample_all(reference):
        shape = reference.shape
        return (
            randint(0, 4, shape, device=reference.device),
            rand(shape, device=reference.device),
            uniform(shape, device=reference.device),
            normal(shape, device=reference.device),
            complex_normal(shape, device=reference.device),
            sample_bernoulli(shape, 0.5, device=reference.device),
        )

    for shape in ((2, 3), (5, 4)):
        outputs = sample_all(torch.empty(shape, device=device))
        assert all(output.shape == shape for output in outputs)
