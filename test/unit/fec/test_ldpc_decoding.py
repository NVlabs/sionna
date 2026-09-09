#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Unit tests for sionna.phy.fec.ldpc.decoding."""

import numpy as np
import pytest
import scipy as sp
import torch

from sionna.phy import config
from sionna.phy.fec.ldpc import (
    LDPC5GEncoder,
    LDPC5GDecoder,
    LDPCBPDecoder,
    cn_update_minsum,
    cn_update_phi,
    cn_update_tanh,
    cn_update_offset_minsum,
    vn_update_sum,
    EXITCallback,
)
from sionna.phy.fec.utils import (
    GaussianPriorSource,
    llr2mi,
    load_parity_check_examples,
)
from sionna.phy.fec.linear import LinearEncoder
from sionna.phy.utils import hard_decisions, sim_ber, ebnodb2no
from sionna.phy.mapping import BinarySource
from sionna.phy.channel import AWGN


CN_UPDATES = ["minsum", "boxplus", "boxplus-phi", "offset-minsum"]


def test_exit_callback_compiles_fullgraph(device):
    """Padded-message handling must not branch on tensor data."""
    callback = EXITCallback(num_iter=2, device=device)

    @torch.compile(fullgraph=True)
    def apply(msg):
        return callback(msg, 0)

    msg = torch.tensor([0.0, 1.0, -2.0], device=device)
    torch.testing.assert_close(apply(msg), msg)
    torch.testing.assert_close(apply(torch.zeros_like(msg)), torch.zeros_like(msg))
    actual = callback.mi[0]
    expected = (llr2mi(-msg[msg != 0]) / 2.0).to(actual.dtype)
    torch.testing.assert_close(actual, expected)


#############################
# Testcases for LDPCBPDecoder
#############################


class TestLDPCBPDecoder:
    """Tests for the LDPCBPDecoder class."""

    @pytest.mark.parametrize("r", [0.5, 0.75])
    @pytest.mark.parametrize("n", [64, 100])
    def test_pcm_consistency(self, r, n):
        """Test against correct pcm formats.

        Parity-check matrix is only allowed to contain binary values.
        """
        k = int(n * r)

        # Raise error if PCM contains other elements than 0,1
        pcm = config.np_rng.uniform(0, 2, [n - k, n]).astype(int)
        # Set a random position to 2 (invalid)
        idx = config.np_rng.uniform(0, n - k, [2]).astype(int)
        pcm[idx[0], idx[1]] = 2
        with pytest.raises(ValueError, match="PC matrix must be binary"):
            LDPCBPDecoder(pcm)

        # Raise error if input shape does not match PCM dim
        pcm = config.np_rng.uniform(0, 2, [n - k, n]).astype(int)
        dec = LDPCBPDecoder(pcm)
        llr = torch.randn(10, n + 1)
        with pytest.raises((AssertionError, ValueError)):
            dec(llr)

    @pytest.mark.parametrize("pcm_id", [0, 1, 2])
    @pytest.mark.parametrize("num_iter", [0, 1, 10])
    def test_message_passing(self, device, pcm_id, num_iter):
        """Test that message passing works correctly using identity functions."""
        pcm, k, n, r = load_parity_check_examples(pcm_id=pcm_id)
        dec = LDPCBPDecoder(
            pcm,
            cn_update="identity",
            vn_update="identity",
            hard_out=False,
            num_iter=num_iter,
            llr_max=100000,
            return_state=True,
            device=device,
        )

        # Feed node indices as inputs (to test correct message passing)
        llr_ch = torch.arange(n, dtype=torch.float32, device=device)
        llr_ch = llr_ch.unsqueeze(0)

        y, msg_v2c = dec(llr_ch)

        # Normalize y by node degree (as VN nodes marginalize)
        vn_degree = np.sum(pcm, axis=0)
        vn_degree_t = torch.tensor(vn_degree, dtype=torch.float32, device=device)

        if num_iter > 0:
            y_ = y / (vn_degree_t + 1)  # +1 as we also marginalize over llr_ch
        else:
            y_ = y

        assert torch.allclose(llr_ch, y_, atol=0.01)

    @pytest.mark.parametrize("cn_update", CN_UPDATES)
    @pytest.mark.parametrize("num_iter", [0, 1, 10])
    def test_torch_compile(self, device, cn_update, num_iter):
        """Test that decoder supports torch.compile."""
        pcm, k, n, r = load_parity_check_examples(pcm_id=0)
        dec = LDPCBPDecoder(
            pcm,
            cn_update=cn_update,
            vn_update="sum",
            hard_out=False,
            num_iter=num_iter,
            device=device,
        )
        source = GaussianPriorSource(device=device)

        llr_ch = source([10, n], 0.1)

        # Reference without compilation
        y_ref = dec(llr_ch)

        # Test with torch.compile
        compiled_dec = torch.compile(dec)
        y_compiled = compiled_dec(llr_ch)

        # Use relaxed tolerance as torch.compile may reorder floating-point
        # operations (e.g., FMA, fused log/exp), causing small numerical
        # differences especially for transcendental-heavy CN updates like
        # boxplus-phi.
        assert torch.allclose(y_ref, y_compiled, atol=1e-3)

    @pytest.mark.parametrize("shape", [[], [2, 3], [2, 3, 4, 5]])
    @pytest.mark.parametrize("pcm_id", [0, 1])
    def test_batch_and_multidimension(self, device, shape, pcm_id):
        """Test that batches and multi-dimensional shapes are properly handled."""
        pcm, k, n, r = load_parity_check_examples(pcm_id=pcm_id)
        dec = LDPCBPDecoder(pcm, num_iter=10, hard_out=False, device=device)
        source = GaussianPriorSource(device=device)

        shape_with_n = shape + [n]
        llr_ch = source(shape_with_n, 0.1)
        y = dec(llr_ch)

        # Reshape before decoding
        y_ref_ = dec(llr_ch.reshape(-1, n))
        # Restore shape after decoding
        y_ref = y_ref_.reshape(shape_with_n)

        assert torch.allclose(y, y_ref, rtol=0.001, atol=0.001)

    @pytest.mark.parametrize("dt_in", [torch.float32, torch.float64])
    @pytest.mark.parametrize("prec", ["single", "double"])
    def test_dtypes(self, device, prec, dt_in):
        """Test different precisions."""
        pcm, k, n, r = load_parity_check_examples(pcm_id=0)
        dec = LDPCBPDecoder(
            pcm,
            hard_out=False,
            precision=prec,
            return_state=True,
            device=device,
        )

        llr_ch = torch.zeros(10, n, dtype=dt_in, device=device)
        y, v2c_msg = dec(llr_ch)

        if prec == "single":
            assert y.dtype == torch.float32
            assert v2c_msg.dtype == torch.float32
        else:
            assert y.dtype == torch.float64
            assert v2c_msg.dtype == torch.float64

    @pytest.mark.parametrize("pcm_id", [0, 1])
    @pytest.mark.parametrize("num_iter", [1, 10])
    def test_internal_state(self, device, pcm_id, num_iter):
        """Test that internal state is correctly returned.

        Run the decoder 1 x num_iter and num_iter x 1 and compare results.
        """
        pcm, k, n, r = load_parity_check_examples(pcm_id=pcm_id)
        source = GaussianPriorSource(device=device)
        bs = 10

        dec_ref = LDPCBPDecoder(
            pcm, hard_out=False, num_iter=num_iter, device=device
        )
        dec = LDPCBPDecoder(
            pcm, hard_out=False, return_state=True, num_iter=1, device=device
        )

        llr_ch = source([bs, n], 0.1)

        # Run reference decoder with num_iter iterations
        y_ref = dec_ref(llr_ch)

        # Run decoder num_iter times with 1 iteration
        msg_v2c = None
        for i in range(num_iter):
            y, msg_v2c = dec(llr_ch, msg_v2c=msg_v2c)

        assert torch.allclose(y, y_ref, atol=1e-5)

        # Also test that num_iter can be passed during call
        y2, _ = dec(llr_ch, num_iter=num_iter)
        assert torch.allclose(y2, y_ref, atol=1e-5)

    def test_gradient(self, device):
        """Test that gradients are accessible and not None."""
        pcm, k, n, r = load_parity_check_examples(1)
        bs = 10

        dec = LDPCBPDecoder(pcm, num_iter=2, hard_out=False, device=device)
        x = torch.ones((bs, n), dtype=torch.float32, device=device, requires_grad=True)

        y = dec(x)
        loss = y.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.all(x.grad == 0)

    @pytest.mark.parametrize("cn_update", CN_UPDATES)
    @pytest.mark.parametrize("num_iter", [0, 1, 10])
    def test_all_erasure(self, device, cn_update, num_iter):
        """Test that all-erasure (llr_ch=0) yields exact 0 outputs.

        This tests against biases in the decoder.
        """
        pcm, k, n, _ = load_parity_check_examples(2)
        bs = 10
        dec = LDPCBPDecoder(
            pcm, cn_update=cn_update, hard_out=False, num_iter=num_iter, device=device
        )
        llr_ch = torch.zeros((bs, n), dtype=torch.float32, device=device)

        y = dec(llr_ch)
        assert torch.equal(llr_ch, y)

    @pytest.mark.parametrize("num_iter", [0, 10])
    def test_hard_output(self, device, num_iter):
        """Test hard-out flag yields hard-decided output."""
        pcm, k, n, _ = load_parity_check_examples(2)
        bs = 10
        source = GaussianPriorSource(device=device)
        dec = LDPCBPDecoder(pcm, hard_out=True, num_iter=num_iter, device=device)

        llr_ch = source([bs, n], 0.1)
        y = dec(llr_ch)

        # Only binary values are allowed
        y_np = y.cpu().numpy()
        assert np.array_equal(y_np, y_np.astype(bool))

    def test_sparse(self, device):
        """Test that parity-check matrix can be scipy.sparse mat."""
        pcm, k, n, _ = load_parity_check_examples(2)
        source = GaussianPriorSource(device=device)
        bs = 10
        num_iter = 10

        # Generate sparse parity-check matrices
        pcm_csc = sp.sparse.csc_matrix(pcm)
        pcm_csr = sp.sparse.csr_matrix(pcm)

        # Instantiate decoders with different pcm datatypes
        dec = LDPCBPDecoder(pcm, num_iter=num_iter, device=device)
        dec_csc = LDPCBPDecoder(pcm_csc, num_iter=num_iter, device=device)
        dec_csr = LDPCBPDecoder(pcm_csr, num_iter=num_iter, device=device)

        llr = source([bs, n], 0.9)

        # Decode with each decoder
        res = dec(llr)
        res_csc = dec_csc(llr)
        res_csr = dec_csr(llr)

        # Results must be the same
        assert torch.allclose(res, res_csc)
        assert torch.allclose(res, res_csr)

    @pytest.mark.parametrize("cn_update", CN_UPDATES)
    @pytest.mark.parametrize("pcm_id", [1, 2, 3])
    def test_e2e_ldpc(self, device, pcm_id, cn_update):
        """End-to-end test of LDPC coding scheme using a linear encoder."""
        no = 0.3
        num_iter = 20
        bs = 10

        pcm, k, n, _ = load_parity_check_examples(pcm_id)
        source = BinarySource(device=device)
        channel = AWGN(device=device)

        encoder = LinearEncoder(pcm, is_pcm=True, device=device)
        dec = LDPCBPDecoder(
            pcm, num_iter=num_iter, cn_update=cn_update, device=device
        )

        bits = source([bs, k])
        c = encoder(bits)
        x_bpsk = 2 * c - 1  # BPSK mapping
        x_bpsk = x_bpsk.to(torch.complex64)
        y = channel(x_bpsk, no)

        # LLR computation
        llr_ch = torch.real(2 / no**2 * y)
        c_hat = dec(llr_ch)

        # Test that transmitted codeword is correctly recovered
        assert torch.equal(c, c_hat)

        # Check that there was at least one transmission error
        c_hat_no_coding = hard_decisions(llr_ch)
        assert not torch.equal(c, c_hat_no_coding)

    @pytest.mark.parametrize("num_iter", [0, 1, 10])
    @pytest.mark.parametrize("llr_max", [0, 5, 100])
    def test_llr_max(self, device, llr_max, num_iter):
        """Test that llr_max is correctly applied."""
        pcm, k, n, _ = load_parity_check_examples(1)
        bs = 10
        dec = LDPCBPDecoder(
            pcm,
            num_iter=num_iter,
            hard_out=False,
            llr_max=llr_max,
            return_state=True,
            device=device,
        )

        # Generate large random inputs
        llr_ch = 2 * llr_max * torch.randn(bs, n, device=device)
        y, msg = dec(llr_ch)

        # Check that no larger value than llr_max exists
        assert y.abs().max() <= llr_max + 1e-5
        assert msg.abs().max() <= llr_max + 1e-5

    def test_scheduling(self, device):
        """Test different schedulings."""
        # Checks are independent
        pcm = np.array([[1, 1, 1, 0, 0, 0], [0, 0, 0, 1, 1, 1]])
        n = pcm.shape[1]

        cns1 = "flooding"
        cns2 = np.stack([[0], [1]], axis=0)  # layered
        cns3 = np.stack([[0], [0]], axis=0)  # update only CN0

        y_out = []
        for cns in [cns1, cns2, cns3]:
            dec = LDPCBPDecoder(
                pcm,
                num_iter=10,
                hard_out=False,
                vn_update="sum",
                cn_update="minsum",
                cn_schedule=cns,
                llr_max=100000,
                device=device,
            )

            x = torch.arange(n, dtype=torch.float32, device=device)
            y = dec(x)
            y_out.append(y)

        # Layered and flooding should be the same output (nodes are independent)
        assert torch.allclose(y_out[0], y_out[1])

        # Output should be different if only first CN is updated
        assert not torch.allclose(y_out[0], y_out[2])


#####################################
# Testcases for node update functions
#####################################


def _gen_padded_tensor(node_degrees, batch_size, device):
    """Generate a padded tensor for testing node update functions."""
    num_nodes = len(node_degrees)
    max_degree = max(node_degrees)

    msg = torch.zeros(batch_size, num_nodes, max_degree, device=device)
    mask = torch.zeros(num_nodes, max_degree, device=device)

    for node_idx, degree in enumerate(node_degrees):
        msg[:, node_idx, :degree] = torch.randn(batch_size, degree, device=device)
        mask[node_idx, :degree] = 1.0

    return msg, mask


class TestNodeUpdateFunctions:
    """Tests for VN and CN update functions."""

    @pytest.mark.parametrize("llr_clipping", [5, 20, 100, None])
    def test_vn_update_sum(self, device, llr_clipping):
        """Test VN update against reference implementation."""
        bs = 100
        node_degrees = [3, 4, 5, 6, 7]
        num_nodes = len(node_degrees)

        msg_c2v, mask = _gen_padded_tensor(node_degrees, bs, device)
        llr_ch = 0.1 * torch.randn(bs, num_nodes, device=device)

        msg_v2c, x_tot = vn_update_sum(msg_c2v, mask, llr_ch, llr_clipping)

        # NumPy reference: msg_np is [bs, num_nodes, max_degree]
        msg_np = msg_c2v.cpu().numpy()
        llr_np = llr_ch.cpu().numpy()
        mask_np = mask.cpu().numpy()

        x_tot_ref = np.zeros((bs, num_nodes))
        x_e_ref = np.zeros_like(msg_np)

        for node_idx in range(num_nodes):
            degree = int(mask_np[node_idx].sum())
            x_in = msg_np[:, node_idx, :degree]  # [bs, degree]
            x_tot_ref[:, node_idx] = np.sum(x_in, axis=1) + llr_np[:, node_idx]
            for i in range(degree):
                x_e_ref[:, node_idx, i] = x_tot_ref[:, node_idx] - x_in[:, i]

        if llr_clipping is not None:
            x_e_ref = np.clip(x_e_ref, -llr_clipping, llr_clipping)
            x_tot_ref = np.clip(x_tot_ref, -llr_clipping, llr_clipping)

        # mask [num_nodes, max_degree] broadcasts with [bs, num_nodes, max_degree]
        x_e_ref = x_e_ref * mask_np[np.newaxis, :, :]

        assert np.allclose(
            x_tot.cpu().numpy(), x_tot_ref, rtol=0.001, atol=0.001
        )
        assert np.allclose(
            msg_v2c.cpu().numpy(), x_e_ref, rtol=0.001, atol=0.001
        )

    @pytest.mark.parametrize("llr_clipping", [5, 20, 100, None])
    def test_cn_update_minsum(self, device, llr_clipping):
        """Test minsum CN update against reference implementation."""
        bs = 100
        node_degrees = [3, 4, 5, 6, 7]

        msg_v2c, mask = _gen_padded_tensor(node_degrees, bs, device)
        msg_c2v = cn_update_minsum(msg_v2c, mask, llr_clipping)

        # NumPy reference: msg_np is [bs, num_nodes, max_degree]
        msg_np = msg_v2c.cpu().numpy()
        mask_np = mask.cpu().numpy()

        msg_c2v_ref = np.zeros_like(msg_np)
        for node_idx, degree in enumerate(node_degrees):
            x_in = msg_np[:, node_idx, :degree]  # [bs, degree]
            sign_out = (
                np.prod(np.sign(x_in), axis=1, keepdims=True) * np.sign(x_in)
            )  # [bs, degree]
            x_abs = np.abs(x_in)

            for i in range(degree):
                cur_min = np.inf
                for j in range(degree):
                    if i != j:
                        cur_min = np.minimum(cur_min, x_abs[:, j])
                msg_c2v_ref[:, node_idx, i] = cur_min * sign_out[:, i]

        if llr_clipping is not None:
            msg_c2v_ref = np.clip(msg_c2v_ref, -llr_clipping, llr_clipping)

        msg_c2v_ref = msg_c2v_ref * mask_np[np.newaxis, :, :]

        assert np.allclose(
            msg_c2v.cpu().numpy(), msg_c2v_ref, rtol=0.001, atol=0.001
        )

    @pytest.mark.parametrize("llr_clipping", [5, 20, 100, None])
    @pytest.mark.parametrize("offset", [0, 0.5, 1.0])
    def test_cn_update_offset_minsum(self, device, llr_clipping, offset):
        """Test offset minsum CN update against reference implementation."""
        bs = 100
        node_degrees = [3, 4, 5, 6, 7]

        msg_v2c, mask = _gen_padded_tensor(node_degrees, bs, device)
        msg_c2v = cn_update_offset_minsum(msg_v2c, mask, llr_clipping, offset=offset)

        # NumPy reference: msg_np is [bs, num_nodes, max_degree]
        msg_np = msg_v2c.cpu().numpy()
        mask_np = mask.cpu().numpy()

        msg_c2v_ref = np.zeros_like(msg_np)
        for node_idx, degree in enumerate(node_degrees):
            x_in = msg_np[:, node_idx, :degree]  # [bs, degree]
            sign_out = (
                np.prod(np.sign(x_in), axis=1, keepdims=True) * np.sign(x_in)
            )
            x_abs = np.abs(x_in)

            for i in range(degree):
                cur_min = np.inf
                for j in range(degree):
                    if i != j:
                        cur_min = np.minimum(cur_min, x_abs[:, j])
                msg_c2v_ref[:, node_idx, i] = np.maximum(cur_min - offset, 0) * sign_out[:, i]

        if llr_clipping is not None:
            msg_c2v_ref = np.clip(msg_c2v_ref, -llr_clipping, llr_clipping)

        msg_c2v_ref = msg_c2v_ref * mask_np[np.newaxis, :, :]

        assert np.allclose(
            msg_c2v.cpu().numpy(), msg_c2v_ref, rtol=0.001, atol=0.001
        )

    @pytest.mark.parametrize("llr_clipping", [5, 20, 100, None])
    def test_cn_update_boxplus(self, device, llr_clipping):
        """Test boxplus CN update (tanh) against reference implementation."""
        bs = 100
        node_degrees = [3, 4, 5, 6, 7]

        msg_v2c, mask = _gen_padded_tensor(node_degrees, bs, device)
        msg_c2v = cn_update_tanh(msg_v2c, mask, llr_clipping)

        # NumPy reference: msg_np is [bs, num_nodes, max_degree]
        msg_np = msg_v2c.cpu().numpy()
        mask_np = mask.cpu().numpy()

        msg_c2v_ref = np.zeros_like(msg_np)
        for node_idx, degree in enumerate(node_degrees):
            x_in = msg_np[:, node_idx, :degree]  # [bs, degree]
            for i in range(degree):
                v = np.ones(bs)
                for j in range(degree):
                    if i != j:
                        v *= np.tanh(x_in[:, j] / 2)
                msg_c2v_ref[:, node_idx, i] = 2 * np.arctanh(np.clip(v, -0.9999999, 0.9999999))

        if llr_clipping is not None:
            msg_c2v_ref = np.clip(msg_c2v_ref, -llr_clipping, llr_clipping)

        msg_c2v_ref = msg_c2v_ref * mask_np[np.newaxis, :, :]

        assert np.allclose(
            msg_c2v.cpu().numpy(), msg_c2v_ref, rtol=0.01, atol=0.01
        )

    @pytest.mark.parametrize("llr_clipping", [5, 20, 100, None])
    def test_cn_update_boxplus_phi(self, device, llr_clipping):
        """Test boxplus-phi CN update against reference implementation."""
        bs = 100
        node_degrees = [3, 4, 5, 6, 7]

        msg_v2c, mask = _gen_padded_tensor(node_degrees, bs, device)
        msg_c2v = cn_update_phi(msg_v2c, mask, llr_clipping)

        def _phi(x):
            x = np.clip(np.abs(x), 1e-10, 20)
            return np.log(np.exp(x) + 1) - np.log(np.exp(x) - 1)

        # NumPy reference: msg_np is [bs, num_nodes, max_degree]
        msg_np = msg_v2c.cpu().numpy()
        mask_np = mask.cpu().numpy()

        msg_c2v_ref = np.zeros_like(msg_np)
        for node_idx, degree in enumerate(node_degrees):
            x_in = msg_np[:, node_idx, :degree]  # [bs, degree]
            for i in range(degree):
                v = np.zeros(bs)
                s = np.ones(bs)
                for j in range(degree):
                    if i != j:
                        sign_j = np.sign(x_in[:, j])
                        sign_j = np.where(sign_j == 0, 1.0, sign_j)
                        s *= sign_j
                        v += _phi(np.abs(x_in[:, j]))
                msg_c2v_ref[:, node_idx, i] = s * _phi(v)

        if llr_clipping is not None:
            msg_c2v_ref = np.clip(msg_c2v_ref, -llr_clipping, llr_clipping)

        msg_c2v_ref = msg_c2v_ref * mask_np[np.newaxis, :, :]

        assert np.allclose(
            msg_c2v.cpu().numpy(), msg_c2v_ref, rtol=0.01, atol=0.01
        )


#############################
# Testcases for LDPC5GDecoder
#############################


class TestLDPC5GDecoder:
    """Tests for the LDPC5GDecoder class."""

    @pytest.mark.parametrize("cn_update", CN_UPDATES)
    @pytest.mark.parametrize("return_infobits", [False, True])
    @pytest.mark.parametrize("num_iter", [0, 1, 10])
    def test_torch_compile(self, device, cn_update, num_iter, return_infobits):
        """Test that decoder supports torch.compile."""
        k = 34
        n = 89
        bs = 10

        encoder = LDPC5GEncoder(k, n, device=device)
        decoder = LDPC5GDecoder(
            encoder,
            num_iter=num_iter,
            cn_update=cn_update,
            return_infobits=return_infobits,
            device=device,
        )
        source = GaussianPriorSource(device=device)

        llr_ch = source([bs, n], 0.1)

        # Reference without compilation
        y_ref = decoder(llr_ch)

        # Test with torch.compile
        compiled_dec = torch.compile(decoder)
        y_compiled = compiled_dec(llr_ch)

        # Use relaxed tolerance as torch.compile may reorder floating-point
        # operations (e.g., FMA, fused log/exp), causing small numerical
        # differences especially for transcendental-heavy CN updates like
        # boxplus-phi.
        assert torch.allclose(y_ref, y_compiled, atol=1e-3)

    @pytest.mark.parametrize("mode", ["default", "reduce-overhead"])
    def test_torch_compile_modes(self, device, mode):
        """Test that torch.compile works with different compilation modes."""
        if device == "cpu" and mode == "reduce-overhead":
            pytest.skip("reduce-overhead mode can be slow on CPU")

        k = 34
        n = 89
        bs = 10

        encoder = LDPC5GEncoder(k, n, device=device)
        decoder = LDPC5GDecoder(encoder, num_iter=5, device=device)
        source = GaussianPriorSource(device=device)

        llr_ch = source([bs, n], 0.1)

        # Reference without compilation
        y_ref = decoder(llr_ch)

        # Compile and run
        compiled_dec = torch.compile(decoder, mode=mode)
        y_compiled = compiled_dec(llr_ch)

        assert torch.allclose(y_ref, y_compiled, atol=1e-3)

    @pytest.mark.parametrize("k", [100, 400, 800, 2000])
    @pytest.mark.parametrize("r", [0.34, 0.5, 0.75, 0.9])
    def test_pruning_5g(self, device, k, r):
        """Test degree-1 VN pruning."""
        bs = 10
        n = int(k / r)

        source = GaussianPriorSource(device=device)

        enc = LDPC5GEncoder(k, n, device=device)
        dec = LDPC5GDecoder(
            enc, prune_pcm=True, hard_out=False, num_iter=10, device=device
        )
        dec_ref = LDPC5GDecoder(
            enc, prune_pcm=False, hard_out=False, num_iter=10, device=device
        )

        llr = source([bs, n], 0.5)
        x = dec(llr)
        x_ref = dec_ref(llr)

        # Allow small difference as iterative error can accumulate
        diff = torch.abs(x - x_ref).mean().item()
        assert diff < 5e-2

    @pytest.mark.parametrize("parameters", [[12, 25], [20, 65], [45, 63], [12, 59], [500, 1000]])
    def test_scheduling_pruning_5g(self, device, parameters):
        """Test layered scheduling for 5G code.

        Test that pruning of the pcm does not mess up the CN update schedule.
        """
        k, n = parameters
        enc = LDPC5GEncoder(k, n, device=device)

        retval = []
        for p in [False, True]:
            dec = LDPC5GDecoder(
                enc,
                cn_schedule="layered",
                num_iter=5,
                return_infobits=False,
                hard_out=False,
                llr_max=10000,
                cn_update="minsum",
                prune_pcm=p,
                device=device,
            )

            x = torch.arange(n, dtype=torch.float32, device=device)
            y = -dec(-x)
            retval.append(y)

        assert torch.allclose(retval[0], retval[1])

    def test_scheduling_5g(self, device):
        """Test layered scheduling for 5G code.

        Test against the rule of thumb that layered decoding requires approx 50%
        of the iterations for similar results.
        Note that correct scheduling was already tested in the BP decoder.
        """
        ebno_db = torch.arange(0, 3, 0.5, device=device)
        k = 200
        n = 400
        source = BinarySource(device=device)
        enc = LDPC5GEncoder(k=k, n=n, device=device)
        cn_update = "boxplus"

        bler = []
        for cns, num_iter in zip(["layered", "flooding"], [8, 16]):
            dec = LDPC5GDecoder(
                enc,
                num_iter=num_iter,
                cn_update=cn_update,
                cn_schedule=cns,
                device=device,
            )

            def run_graph(batch_size, ebno_db, _dec=dec, _device=device):
                no = ebnodb2no(ebno_db, 2, k / n, device=_device)
                b = source((batch_size, k))
                c = enc(b)
                x = 2 * c - 1  # BPSK
                y = x + torch.sqrt(no) * torch.randn(
                    batch_size, n, device=_device, dtype=x.dtype
                )
                llr_ch = 2 * y / no
                return b, _dec(llr_ch)

            _, bler_ = sim_ber(
                run_graph,
                ebno_dbs=ebno_db,
                max_mc_iter=10,
                num_target_block_errors=100,
                target_bler=1e-3,
                batch_size=10000,
                soft_estimates=False,
                early_stop=True,
                verbose=False,
                device=device,
            )
            bler.append(bler_)

        # Verify that BLERs are similar; allow rtol as this is only a rule of thumb
        # and BLERs are in the log-domain, i.e. a factor 2x is still ok
        # Use larger tolerance due to statistical variation
        assert torch.allclose(bler[0], bler[1], rtol=1.0, atol=0.01)

    def test_gradient_5g(self, device):
        """Test that gradients are accessible and not None."""
        k = 20
        n = 50
        bs = 10

        enc = LDPC5GEncoder(k, n, device=device)
        dec = LDPC5GDecoder(enc, num_iter=2, hard_out=False, device=device)
        x = torch.ones((bs, n), dtype=torch.float32, device=device, requires_grad=True)

        y = dec(x)
        loss = y.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.all(x.grad == 0)

    @pytest.mark.parametrize(
        "parameters", [[12, 20, 1], [200, 250, 2], [345, 544, 4], [231, 808, 8]]
    )
    def test_output_interleaver_5g(self, device, parameters):
        """Test output interleaver."""
        k, n, m = parameters
        bs = 10

        source = BinarySource(device=device)
        enc_ref = LDPC5GEncoder(k, n, device=device)
        enc = LDPC5GEncoder(k, n, m, device=device)
        dec_ref = LDPC5GDecoder(enc_ref, cn_update="minsum", device=device)
        dec = LDPC5GDecoder(enc, cn_update="minsum", device=device)
        dec_cw = LDPC5GDecoder(
            enc, cn_update="minsum", return_infobits=False, device=device
        )

        u = source([bs, k])
        c = enc(u)
        c_ref = enc_ref(u)

        # Emulate tx (no noise/scaling due to minsum required)
        y = 2 * c - 1
        y_ref = 2 * c_ref - 1

        u_hat = dec(y)
        c_hat = dec_cw(y)
        u_hat_ref = dec_ref(y_ref)

        assert torch.equal(u_hat, u_hat_ref)

        # Also verify that codeword is correctly returned
        assert torch.equal(c_hat, c)

        # And verify that c and c_ref are different for m>1
        if m > 1:
            assert not torch.equal(c, c_ref)

    @pytest.mark.parametrize("cn_update", ["boxplus-phi", "minsum"])
    @pytest.mark.parametrize("n", [100, 500])
    @pytest.mark.parametrize("r", [0.5, 0.75])
    def test_e2e_ldpc_5g(self, device, r, n, cn_update):
        """Test end-to-end LDPC coding scheme with 5G NR Encoder.

        Uses high SNR (noiseless) to ensure reliable decoding.
        """
        num_iter = 20
        bs = 5
        k = int(r * n)

        source = BinarySource(device=device)
        encoder = LDPC5GEncoder(k, n, device=device)
        decoder = LDPC5GDecoder(
            encoder, num_iter=num_iter, cn_update=cn_update, device=device
        )

        bits = source([bs, k])
        c = encoder(bits)

        # Use noiseless perfect LLRs for reliable testing
        llr_ch = 10 * (2 * c - 1)  # Perfect logits
        b_hat = decoder(llr_ch)

        # Test that transmitted info bits are correctly recovered
        assert torch.equal(bits, b_hat)

    @pytest.mark.parametrize("cn_update", CN_UPDATES)
    @pytest.mark.parametrize("dt_in", [torch.float32, torch.float64])
    @pytest.mark.parametrize("prec", ["single", "double"])
    @pytest.mark.parametrize("return_infobits", [False, True])
    def test_dtypes_5g(self, device, dt_in, prec, cn_update, return_infobits):
        """Test different precisions."""
        k = 50
        n = 100
        bs = 10

        encoder = LDPC5GEncoder(k, n, device=device)
        decoder = LDPC5GDecoder(
            encoder,
            cn_update=cn_update,
            precision=prec,
            return_infobits=return_infobits,
            return_state=True,
            device=device,
        )

        llr_ch = torch.zeros(bs, n, dtype=dt_in, device=device)
        y, v2c_msg = decoder(llr_ch)

        if prec == "single":
            assert y.dtype == torch.float32
            assert v2c_msg.dtype == torch.float32
        else:
            assert y.dtype == torch.float64
            assert v2c_msg.dtype == torch.float64

    @pytest.mark.parametrize("num_iter", [1, 10])
    def test_internal_state_5g(self, device, num_iter):
        """Test that internal state is correctly returned."""
        k = 50
        n = 100
        bs = 10

        source = GaussianPriorSource(device=device)
        encoder = LDPC5GEncoder(k, n, device=device)
        decoder_ref = LDPC5GDecoder(
            encoder,
            return_infobits=False,
            hard_out=False,
            return_state=False,
            num_iter=num_iter,
            device=device,
        )
        decoder = LDPC5GDecoder(
            encoder,
            return_infobits=False,
            hard_out=False,
            return_state=True,
            num_iter=1,
            device=device,
        )

        llr_ch = source([bs, n], 0.1)

        # Run reference decoder with num_iter iterations
        y_ref = decoder_ref(llr_ch)

        # Run decoder num_iter times with 1 iteration
        msg_v2c = None
        for i in range(num_iter):
            y, msg_v2c = decoder(llr_ch, msg_v2c=msg_v2c)

        assert torch.allclose(y, y_ref, atol=1e-5)

        # Also test that num_iter can be passed during call
        y2, _ = decoder(llr_ch, num_iter=num_iter)
        assert torch.allclose(y2, y_ref, atol=1e-5)

    @pytest.mark.parametrize("cn_update", CN_UPDATES)
    @pytest.mark.parametrize("num_iter", [0, 1, 10])
    def test_all_erasure_5g(self, device, cn_update, num_iter):
        """Test that all-erasure (llr_ch=0) yields exact 0 outputs."""
        k = 75
        n = 150
        bs = 10

        encoder = LDPC5GEncoder(k, n, device=device)
        decoder = LDPC5GDecoder(
            encoder,
            cn_update=cn_update,
            return_infobits=False,
            hard_out=False,
            num_iter=num_iter,
            device=device,
        )

        llr_ch = torch.zeros((bs, n), dtype=torch.float32, device=device)
        y = decoder(llr_ch)

        assert torch.equal(llr_ch, y)

    @pytest.mark.parametrize("num_iter", [0, 10])
    def test_hard_output_5g(self, device, num_iter):
        """Test hard-out flag yields hard-decided output."""
        k = 75
        n = 150
        bs = 10

        source = GaussianPriorSource(device=device)
        encoder = LDPC5GEncoder(k, n, device=device)
        decoder = LDPC5GDecoder(
            encoder,
            hard_out=True,
            num_iter=num_iter,
            return_infobits=False,
            device=device,
        )

        llr_ch = source([bs, n], 0.1)
        y = decoder(llr_ch)

        # Only binary values are allowed
        y_np = y.cpu().numpy()
        assert np.array_equal(y_np, y_np.astype(bool))

    @pytest.mark.parametrize("num_iter", [0, 1, 10])
    @pytest.mark.parametrize("llr_max", [0, 5, 100])
    def test_llr_max_5g(self, device, llr_max, num_iter):
        """Test that llr_max is correctly applied."""
        k = 12
        n = 20
        bs = 10

        encoder = LDPC5GEncoder(k, n, device=device)
        decoder = LDPC5GDecoder(
            encoder,
            hard_out=False,
            num_iter=num_iter,
            return_infobits=False,
            return_state=True,
            llr_max=llr_max,
            device=device,
        )

        llr_ch = 2 * llr_max * torch.randn(bs, n, device=device)
        y, msg = decoder(llr_ch)

        assert y.abs().max() <= llr_max + 1e-5
        assert msg.abs().max() <= llr_max + 1e-5

    @pytest.mark.parametrize("cn_update", CN_UPDATES)
    def test_llr_max_none_roundtrip(self, device, cn_update):
        """llr_max=None must decode noiseless codewords for every CN update."""
        k, n, bs = 100, 200, 8
        enc = LDPC5GEncoder(k, n, device=device)
        bits = BinarySource(device=device)([bs, k])
        c = enc(bits)
        llr_ch = 10.0 * (2.0 * c - 1.0)

        dec_clip = LDPC5GDecoder(
            enc, num_iter=20, cn_update=cn_update, device=device
        )
        dec_none = LDPC5GDecoder(
            enc, num_iter=20, cn_update=cn_update, llr_max=None, device=device
        )
        u_clip = dec_clip(llr_ch)
        u_none = dec_none(llr_ch)
        assert torch.equal(u_clip, bits)
        assert torch.equal(u_none, bits)

    @pytest.mark.parametrize("cn_update", CN_UPDATES)
    def test_llr_max_none_harq_roundtrip(self, device, cn_update):
        """HARQ with llr_max=None must match clipped decoding on perfect LLRs."""
        k, n, bs = 100, 200, 8
        enc = LDPC5GEncoder(k, n, device=device)
        bits = BinarySource(device=device)([bs, k])
        rv = [0, 2]
        c = enc(bits, rv=rv)
        llr_ch = 10.0 * (2.0 * c - 1.0)

        dec_clip = LDPC5GDecoder(
            enc, num_iter=20, cn_update=cn_update, harq_mode=True, device=device
        )
        dec_none = LDPC5GDecoder(
            enc,
            num_iter=20,
            cn_update=cn_update,
            harq_mode=True,
            llr_max=None,
            device=device,
        )
        u_clip = dec_clip(llr_ch, rv=rv)
        u_none = dec_none(llr_ch, rv=rv)
        assert torch.equal(u_clip, bits)
        assert torch.equal(u_none, bits)

    @pytest.mark.parametrize("cn_update", CN_UPDATES)
    def test_llr_max_none_unclipped_and_finite(self, device, cn_update):
        """llr_max=None must not clip messages and must stay finite."""
        k, n, bs = 50, 100, 4
        enc = LDPC5GEncoder(k, n, device=device)
        dec = LDPC5GDecoder(
            enc,
            cn_update=cn_update,
            hard_out=False,
            num_iter=5,
            return_infobits=False,
            return_state=True,
            llr_max=None,
            device=device,
        )
        llr_ch = 50.0 * torch.randn(bs, n, device=device)
        y, msg = dec(llr_ch)
        assert torch.isfinite(y).all()
        assert torch.isfinite(msg).all()
        assert y.abs().max() > 20.0

    @pytest.mark.parametrize(
        "k,n,ebno_db",
        [
            (100, 200, 5.0),
            (200, 300, 4.5),
            (80, 240, 3.5),
        ],
    )
    def test_llr_max_none_bler(self, device, k, n, ebno_db):
        """Same noisy observations: unclipped must recover every codeword
        that clipped decoding recovers."""
        bs = 128
        torch.manual_seed(0)
        enc = LDPC5GEncoder(k, n, device=device)
        bits = BinarySource(device=device)([bs, k])
        c = enc(bits)
        no = ebnodb2no(ebno_db, num_bits_per_symbol=1, coderate=k / n)
        x = (2.0 * c - 1.0).to(torch.complex64)
        y = AWGN(device=device)(x, no)
        llr_ch = 2.0 * torch.real(y) / no

        dec_clip = LDPC5GDecoder(enc, num_iter=20, device=device)
        dec_none = LDPC5GDecoder(
            enc, num_iter=20, llr_max=None, device=device
        )
        u_clip = dec_clip(llr_ch)
        u_none = dec_none(llr_ch)
        ok_clip = ~torch.any(u_clip != bits, dim=-1)
        ok_none = ~torch.any(u_none != bits, dim=-1)

        assert ok_clip.any(), "clipped decoder failed every codeword"
        assert torch.all(ok_none | ~ok_clip), (
            "unclipped decoder lost a codeword that clipped decoding recovered"
        )

    @pytest.mark.parametrize("shape", [[], [2, 3], [2, 3, 4, 5]])
    def test_batch_and_multidimension_5g(self, device, shape):
        """Test that batches and multi-dimensional shapes are handled."""
        k = 100
        n = 200
        num_iter = 5

        encoder = LDPC5GEncoder(k, n, device=device)
        decoder = LDPC5GDecoder(
            encoder, hard_out=False, num_iter=num_iter, return_infobits=False, device=device
        )

        source = GaussianPriorSource(device=device)
        shape_with_n = shape + [n]
        llr_ch = source(shape_with_n, 0.1)
        y = decoder(llr_ch)

        # Reshape before decoding
        y_ref_ = decoder(llr_ch.reshape(-1, n))
        # Restore shape after decoding
        y_ref = y_ref_.reshape(shape_with_n)

        assert torch.allclose(y, y_ref, rtol=0.001, atol=0.001)

    @pytest.mark.parametrize(
        "parameters",
        [
            [64, 128],
            [64, 180],
            [167, 201],
            [439, 800],
            [948, 1024],
            [3893, 7940],
        ],
    )
    def test_rate_matching_5g(self, device, parameters):
        """Test that if return_infobits==False, the full codeword is returned.

        We test this for zero iterations to see if all internal reshapes
        are correctly recovered before returning the estimate.
        """
        k, n = parameters
        bs = 10

        enc = LDPC5GEncoder(k, n, device=device)
        dec = LDPC5GDecoder(
            enc, hard_out=False, return_infobits=False, num_iter=0, device=device
        )

        llr = torch.randn(bs, n, device=device) + 4.2

        # Check if return after 0 iterations equals input
        c_hat = dec(llr)
        assert torch.allclose(c_hat, llr, atol=1e-5)

