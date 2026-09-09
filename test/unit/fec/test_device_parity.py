#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""CPU/GPU parity tests for 5G FEC blocks."""

import pytest
import torch

from sionna.phy.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder
from sionna.phy.fec.polar import Polar5GEncoder, Polar5GDecoder
from sionna.phy.mapping import BinarySource


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cpu_gpu_parity_5g():
    """Tiny 5G LDPC and Polar encode/decode must match on CPU and GPU."""
    gpu = "cuda:0"
    source = BinarySource(device="cpu")

    k_ldpc, n_ldpc, bs = 50, 100, 4
    bits_ldpc = source([bs, k_ldpc])
    enc_ldpc_cpu = LDPC5GEncoder(k_ldpc, n_ldpc, device="cpu")
    enc_ldpc_gpu = LDPC5GEncoder(k_ldpc, n_ldpc, device=gpu)
    c_ldpc_cpu = enc_ldpc_cpu(bits_ldpc)
    c_ldpc_gpu = enc_ldpc_gpu(bits_ldpc.to(gpu))
    assert torch.equal(c_ldpc_cpu, c_ldpc_gpu.cpu())

    llr_ldpc = 10.0 * (2.0 * c_ldpc_cpu - 1.0)
    u_ldpc_cpu = LDPC5GDecoder(
        enc_ldpc_cpu, num_iter=10, device="cpu"
    )(llr_ldpc)
    u_ldpc_gpu = LDPC5GDecoder(
        enc_ldpc_gpu, num_iter=10, device=gpu
    )(llr_ldpc.to(gpu))
    assert torch.equal(u_ldpc_cpu, bits_ldpc)
    assert torch.equal(u_ldpc_gpu.cpu(), bits_ldpc)

    k_polar, n_polar = 20, 32
    bits_polar = source([bs, k_polar])
    enc_polar_cpu = Polar5GEncoder(k_polar, n_polar, device="cpu")
    enc_polar_gpu = Polar5GEncoder(k_polar, n_polar, device=gpu)
    c_polar_cpu = enc_polar_cpu(bits_polar)
    c_polar_gpu = enc_polar_gpu(bits_polar.to(gpu))
    assert torch.equal(c_polar_cpu, c_polar_gpu.cpu())

    llr_polar = 20.0 * (2.0 * c_polar_cpu - 1.0)
    u_polar_cpu = Polar5GDecoder(
        enc_polar_cpu, dec_type="SCL", list_size=8, device="cpu"
    )(llr_polar)
    u_polar_gpu = Polar5GDecoder(
        enc_polar_gpu, dec_type="SCL", list_size=8, device=gpu
    )(llr_polar.to(gpu))
    assert torch.equal(u_polar_cpu, bits_polar)
    assert torch.equal(u_polar_gpu.cpu(), bits_polar)
