#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Cold-compiled invalid-input checks for validation-helper consumers."""

import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest
import torch


SOURCE_ROOT = Path(__file__).resolve().parents[2] / "src"
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Compiled device-assert isolation requires a CUDA device",
)


CASES = [
    pytest.param(
        """
        from sionna.phy.utils import flatten_multi_index
        @torch.compile(fullgraph=True)
        def checked(value):
            return flatten_multi_index(value, [2, 2])
        """,
        'checked(torch.tensor([[-1, 0]], device="cuda:0"))',
        "`indices` must be non-negative",
        id="flatten-multi-index",
    ),
    pytest.param(
        """
        from sionna.phy.nr.utils import MCSDecoderNR
        decoder = MCSDecoderNR(device="cuda:0")
        @torch.compile(fullgraph=True)
        def checked(value):
            return decoder(value, 1, 0)
        """,
        'checked(torch.tensor([-1], device="cuda:0"))',
        "MCS index cannot be negative",
        id="mcs-decoder",
    ),
    pytest.param(
        """
        from sionna.phy.nr.utils import calculate_tb_size
        @torch.compile(fullgraph=True)
        def checked(num_prbs):
            return calculate_tb_size(
                torch.tensor([4], device="cuda:0"),
                torch.tensor([0.5], device="cuda:0"),
                num_prbs=num_prbs,
                num_ofdm_symbols=torch.tensor([14], device="cuda:0"),
                num_dmrs_per_prb=torch.tensor([12], device="cuda:0"),
                return_cw_length=False,
            )
        """,
        'checked(torch.tensor([0], device="cuda:0"))',
        "num_prbs must be in [1, 275]",
        id="tb-size",
    ),
    pytest.param(
        """
        from sionna.phy.channel import BinarySymmetricChannel
        channel = BinarySymmetricChannel(device="cuda:0")
        @torch.compile(fullgraph=True)
        def checked(value):
            return channel(value, 0.1)
        """,
        'checked(torch.tensor([0.0, 0.5, 1.0], device="cuda:0"))',
        "Input must be binary",
        id="binary-channel",
    ),
    pytest.param(
        """
        from sionna.phy.fec.polar import PolarEncoder
        from sionna.phy.fec.polar.utils import generate_5g_ranking
        frozen, _ = generate_5g_ranking(16, 32)
        encoder = PolarEncoder(frozen, 32, device="cuda:0")
        @torch.compile(fullgraph=True)
        def checked(value):
            return encoder(value)
        """,
        (
            'checked(torch.tensor([[0.0, 1.0, 0.5, 1.0] * 4], '
            'device="cuda:0"))'
        ),
        "Input must be binary",
        id="polar-encoder",
    ),
    pytest.param(
        """
        from sionna.phy.ofdm import OFDMModulator
        modulator = OFDMModulator(128, device="cuda:0")
        compiled = torch.compile(modulator, fullgraph=True)
        """,
        'compiled(torch.zeros(1, 2, 64, dtype=torch.complex64, device="cuda:0"))',
        "`cyclic_prefix_length` cannot be larger than `fft_size`",
        id="ofdm-modulator",
    ),
    pytest.param(
        """
        from sionna.phy.mimo import normalize_precoding_power
        checked = torch.compile(normalize_precoding_power, fullgraph=True)
        """,
        'checked(torch.zeros(1, 8, dtype=torch.complex64, device="cuda:0"))',
        "zero norm",
        id="precoding-power",
    ),
    pytest.param(
        """
        from sionna.sys import geometry_sinr_db
        @torch.compile(fullgraph=True)
        def checked(bandwidth):
            return geometry_sinr_db(
                torch.tensor([[100.0, 110.0]], device="cuda:0"),
                46.0,
                bandwidth,
                9.0,
            )
        """,
        'checked(torch.tensor(-1.0, device="cuda:0"))',
        "`bandwidth_hz` must contain finite, positive values",
        id="geometry-sinr",
    ),
    pytest.param(
        """
        from sionna.sys import get_pathloss
        @torch.compile(fullgraph=True)
        def checked(association):
            channel = torch.ones(
                1, 2, 1, 2, 1, 2, 4,
                dtype=torch.complex64,
                device="cuda:0",
            )
            return get_pathloss(channel, association)
        """,
        'checked(torch.tensor([[1, 2], [0, 1]]))',
        "rx_tx_association must contain binary values",
        id="pathloss-association",
    ),
    pytest.param(
        """
        from sionna.sys import HexGrid
        grid = HexGrid(isd=50, num_rings=1, device="cuda:0")
        @torch.compile(fullgraph=True, dynamic=True)
        def checked(reference):
            return grid.call(
                reference.shape[0],
                reference.shape[1],
                20,
                min_ut_height=2,
                max_ut_height=1,
            )
        """,
        'checked(torch.empty(2, 3, device="cuda:0"))',
        "max_ut_height must be >= min_ut_height",
        id="hex-grid",
    ),
]


@pytest.mark.parametrize(("setup", "invocation", "message"), CASES)
def test_invalid_cold_compiled_consumer(
    setup,
    invocation,
    message,
    tmp_path,
):
    """Each compiled consumer enforces its tensor contract in isolation."""
    script = "\n".join(
        [
            "import os",
            "import torch",
            textwrap.dedent(setup).strip(),
            "",
            "try:",
            f"    {invocation}",
            "    torch.cuda.synchronize()",
            "except BaseException as error:",
            "    print(type(error).__name__, str(error), flush=True)",
            "    os._exit(17)",
            'print("compiled tensor contract was not enforced", flush=True)',
            "os._exit(0)",
        ]
    )
    env = os.environ.copy()
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        f"{SOURCE_ROOT}{os.pathsep}{existing}" if existing else str(SOURCE_ROOT)
    )
    env["CUDA_LAUNCH_BLOCKING"] = "1"
    env["TORCHINDUCTOR_CACHE_DIR"] = str(tmp_path / "torchinductor")

    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    output = result.stdout + result.stderr
    assert result.returncode != 0, output
    assert message in output
