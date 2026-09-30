#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Compare rate-changing FIR blocks with the existing Sionna block chains.

Run from the repository root after installing Sionna, for example::

    PYTHONPATH=src python test/benchmarks/test_rate_change_fir_benchmark.py --device cpu
    pytest -s test/benchmarks/test_rate_change_fir_benchmark.py --device=cpu

This is a manual benchmark. Timings depend on hardware, input shape, and
PyTorch settings; it deliberately has no CI pass/fail threshold.
"""

import argparse
import statistics
import time

import torch

from sionna.phy.signal import (
    CustomFilter,
    DecimatingFIR,
    Downsampling,
    InterpolatingFIR,
    Upsampling,
)


def measure(fn, device: torch.device, warmup: int, repeats: int) -> float:
    """Return median forward time in milliseconds, synchronizing CUDA."""

    def synchronize() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    with torch.inference_mode():
        for _ in range(warmup):
            fn()
        synchronize()
        timings = []
        for _ in range(repeats):
            synchronize()
            start = time.perf_counter()
            fn()
            synchronize()
            timings.append((time.perf_counter() - start) * 1000)
    return statistics.median(timings)


def run_benchmark(args: argparse.Namespace) -> None:
    """Print side-by-side medians for the two rate-changing operations."""
    if min(args.batch, args.length, args.sps, args.repeats, args.threads) < 1:
        raise ValueError("batch, length, sps, repeats, and threads must be positive")
    if args.warmup < 0:
        raise ValueError("warmup must be non-negative")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable")

    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    torch.manual_seed(123)
    x = torch.randn(args.batch, args.length, device=device)
    taps = torch.randn(8 * args.sps + 1, device=device)
    old_filter = CustomFilter(args.sps, taps, normalize=False, device=args.device)
    old_up = Upsampling(args.sps, device=args.device)
    old_down = Downsampling(args.sps, offset=args.sps - 1, device=args.device)
    new_up = InterpolatingFIR(args.sps, taps, device=args.device)
    new_down = DecimatingFIR(args.sps, taps, offset=args.sps - 1, device=args.device)

    cases = (
        ("interpolation", lambda: old_filter(old_up(x)), lambda: new_up(x)),
        ("decimation", lambda: old_down(old_filter(x)), lambda: new_down(x)),
    )
    print(
        f"device={device} batch={args.batch} length={args.length} "
        f"factor={args.sps} taps={taps.numel()} threads={args.threads} "
        f"warmup={args.warmup} repeats={args.repeats}"
    )
    print("case             old median ms   new median ms   speedup")
    for name, old_fn, new_fn in cases:
        old_ms = measure(old_fn, device, args.warmup, args.repeats)
        new_ms = measure(new_fn, device, args.warmup, args.repeats)
        print(f"{name:16} {old_ms:13.3f} {new_ms:15.3f} {old_ms / new_ms:8.2f}x")


def test_rate_change_fir_benchmark() -> None:
    """Opt-in pytest entry point; timings are informative, not assertions."""
    args = argparse.Namespace(
        device="cpu",
        batch=32,
        length=2048,
        sps=4,
        warmup=10,
        repeats=100,
        threads=1,
    )
    previous_threads = torch.get_num_threads()
    try:
        run_benchmark(args)
    finally:
        torch.set_num_threads(previous_threads)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--length", type=int, default=2048)
    parser.add_argument("--sps", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--threads", type=int, default=1)
    run_benchmark(parser.parse_args())


if __name__ == "__main__":
    main()
