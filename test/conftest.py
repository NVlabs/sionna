#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""
Pytest configuration for Sionna test suite.

GPU Memory Management Tips for Running Full Test Suite:
--------------------------------------------------------
When running many GPU tests sequentially, CUDA memory can become fragmented,
leading to OOM errors even when individual tests pass. Options to mitigate:

1. Run with periodic garbage collection (default, configurable):
   pytest --gc-interval=25  # More aggressive cleanup every 25 tests

2. Run tests in smaller batches by test directory:
   pytest test/unit/channel/ && pytest test/unit/fec/ && ...

3. Use pytest-forked for maximum isolation (each test in separate process):
   pip install pytest-forked
   pytest --forked

4. Reduce parallelism if using pytest-xdist:
   pytest -n 1  # Single worker to reduce memory pressure

5. Set PYTORCH_ALLOC_CONF for better memory management:
   export PYTORCH_ALLOC_CONF=expandable_segments:True
   (This is set automatically by conftest.py if not already set)
"""

import gc
import os
import pytest
import sys
import torch


def resolve_device_option(
    device_option: str, *, cuda_available: bool | None = None
) -> str:
    """Resolve ``--device`` to ``cpu``, ``gpu``, or ``all``.

    ``auto`` becomes ``gpu`` when CUDA is available, otherwise ``cpu``.
    """
    if cuda_available is None:
        cuda_available = torch.cuda.is_available()
    if device_option == "auto":
        return "gpu" if cuda_available else "cpu"
    return device_option


def pytest_addoption(parser):
    """Add command line options for device selection."""
    parser.addoption(
        "--device",
        action="store",
        default="auto",
        choices=["auto", "cpu", "gpu", "all"],
        help=(
            "Device to run tests on: auto, cpu, gpu, or all "
            "(default: auto = gpu if CUDA is available, else cpu)"
        ),
    )
    parser.addoption(
        "--gc-interval",
        action="store",
        default=50,
        type=int,
        help="Perform aggressive garbage collection every N tests (default: 50)",
    )


def pytest_configure(config) -> None:
    # Register custom markers
    config.addinivalue_line("markers", "gpu: mark test as GPU-only")

    # Configure PyTorch CUDA memory allocator for better memory management
    # during long test runs. This helps reduce memory fragmentation.
    if (
        "PYTORCH_CUDA_ALLOC_CONF" not in os.environ
        and "PYTORCH_ALLOC_CONF" not in os.environ
    ):
        # expandable_segments helps reduce fragmentation by allowing
        # the allocator to release memory back to the system more easily
        os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

    # Resolve --device before importing sionna. Never bake cuda:0 into the
    # environment on a CPU-only host.
    device_option = resolve_device_option(config.getoption("--device"))
    config.option.device = device_option

    if device_option == "gpu" and not torch.cuda.is_available():
        pytest.exit(
            "CUDA is not available. Re-run with --device=cpu (or omit --device "
            "to auto-select cpu on this host).",
            returncode=1,
        )

    # Set SIONNA_DEVICE env var BEFORE importing sionna
    # This controls the default device for all tests
    if device_option == "cpu":
        os.environ["SIONNA_DEVICE"] = "cpu"
    elif device_option == "gpu":
        os.environ["SIONNA_DEVICE"] = "cuda:0"
    # Note: sionna.phy.config should read SIONNA_DEVICE if set
    # "--device=all" leaves the env unset; per-test fixtures set config.device.

    # Add test subdirectories to path for direct imports of test utilities
    test_dir = os.path.dirname(os.path.abspath(__file__))
    for subdir in ["unit/channel", "unit/sys"]:
        path = os.path.join(test_dir, subdir)
        if path not in sys.path:
            sys.path.insert(0, path)

    # Add src directory to path for sionna imports
    src_dir = os.path.join(os.path.dirname(test_dir), "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    import sionna

    # Also set config.device directly after import
    if device_option == "cpu":
        sionna.phy.config.device = "cpu"
    elif device_option == "gpu":
        sionna.phy.config.device = "cuda:0"


@pytest.fixture(autouse=True)
def set_seed():
    """Seed each test and undo any global config it leaves behind.

    Tests that mutate `config.device` or `config.precision` without restoring
    them would otherwise change the device and precision of every later test.
    """
    import sionna.phy

    config = sionna.phy.config
    config.seed = 42
    original_device = config.device
    original_precision = config.precision
    yield
    config.device = original_device
    config.precision = original_precision


def _clear_compile_cache():
    """Clear torch.compile caches which can hold GPU memory."""
    try:
        torch._dynamo.reset()
    except Exception:
        pass  # Ignore if dynamo is not available or reset fails


def clear_all_gpu_memory():
    """Helper to aggressively clear GPU memory on all available CUDA devices."""
    # Force Python garbage collection first (multiple passes for cyclic refs)
    gc.collect()
    gc.collect()

    if torch.cuda.is_available():
        for device_id in range(torch.cuda.device_count()):
            with torch.cuda.device(device_id):
                # Empty the CUDA cache
                torch.cuda.empty_cache()
                # Synchronize to ensure all operations are complete
                torch.cuda.synchronize()
                # Reset memory stats to help with fragmentation tracking
                torch.cuda.reset_peak_memory_stats()

    # Additional gc pass to catch any lingering references
    gc.collect()


def pytest_runtest_teardown(item, nextitem):
    """
    Hook that runs after each test teardown.
    If the next test is in a different class or module (or there is no next test),
    perform aggressive memory cleanup to free class/module-scoped fixture data.
    """
    current_class = getattr(item, "cls", None)
    next_class = getattr(nextitem, "cls", None) if nextitem else None
    current_module = getattr(item, "module", None)
    next_module = getattr(nextitem, "module", None) if nextitem else None

    # If we're switching classes, modules, or finishing, do aggressive cleanup
    if current_class != next_class or current_module != next_module:
        # Clear torch.compile caches
        _clear_compile_cache()
        # Clear GPU memory
        clear_all_gpu_memory()


def pytest_generate_tests(metafunc):
    """Generate test parameters based on --device option."""
    if "device" in metafunc.fixturenames:
        # Already resolved from "auto" in pytest_configure; "gpu" without CUDA
        # exits there, so this only sees cpu / gpu-with-CUDA / all.
        device_option = metafunc.config.getoption("--device")

        if device_option == "cpu":
            devices = ["cpu"]
        elif device_option == "gpu":
            devices = ["cuda:0"]
        else:  # "all"
            devices = ["cpu"]
            if torch.cuda.is_available():
                devices.append("cuda:0")

        # Indirect so the `device` fixture below runs and restores
        # `config.device` afterwards; without it the params bypass the fixture.
        metafunc.parametrize("device", devices, indirect=True)


@pytest.fixture
def device(request):
    """
    Fixture that provides the device for testing.
    The actual parametrization is done by pytest_generate_tests.
    Also sets the global config.device for the duration of the test.
    """
    from sionna.phy import config

    device = request.param
    if device not in config.available_devices:
        pytest.skip(f"Device {device} not available")

    # Set global config device for this test
    original_device = config.device
    config.device = device
    yield device
    config.device = original_device


# List of precisions to test
PRECISIONS = ["single", "double"]


@pytest.fixture(params=PRECISIONS)
def precision(request):
    """
    Fixture that parametrizes tests over precisions.
    """
    return request.param


# Compilation modes for torch.compile tests. max-autotune is intentionally
# omitted from the shared fixture: it dominates suite runtime while only a
# subset of compile tests need it. Opt in per-test with an explicit
# @pytest.mark.parametrize("mode", [...]) when coverage is required.
MODES = ["default", "reduce-overhead"]


@pytest.fixture(params=MODES)
def mode(request):
    """
    Fixture that parametrizes tests over compilation modes.
    """
    return request.param
