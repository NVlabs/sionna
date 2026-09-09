#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Utility functions for LDPC decoding."""

from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import torch

from sionna.phy.object import Object


__all__ = [
    "EXITCallback",
    "DecoderStatisticsCallback",
    "WeightedBPCallback",
]


class EXITCallback(Object):
    # pylint: disable=line-too-long
    """Callback for the LDPCBPDecoder to track EXIT statistics.

    Can be registered as ``c2v_callbacks`` or ``v2c_callbacks`` in the
    :class:`~sionna.phy.fec.ldpc.decoding.LDPCBPDecoder` and the
    :class:`~sionna.phy.fec.ldpc.decoding.LDPC5GDecoder`.

    This callback requires all-zero codeword simulations.

    :param num_iter: Maximum number of decoding iterations.
    :param device: Device for computation (e.g., 'cpu', 'cuda:0').

    :input msg: [batch_size, num_vns, max_degree], `torch.float`.
        The v2c or c2v messages.

    :input it: `int`.
        Current number of decoding iterations.

    :output msg: `torch.float`.
        Same as ``msg``.

    .. rubric:: Examples


    .. code-block:: python

        from sionna.phy.fec.ldpc import LDPCBPDecoder
        from sionna.phy.fec.ldpc.utils import EXITCallback

        # Create callback
        exit_cb = EXITCallback(num_iter=20)

        # Create decoder with callback
        decoder = LDPCBPDecoder(pcm, v2c_callbacks=[exit_cb])

        # After decoding, access mutual information
        mi = exit_cb.mi
    """

    def __init__(
        self,
        num_iter: int,
        device: Optional[str] = None,
    ):
        super().__init__(device=device)
        # Accumulators are sized for the accumulation, not for the simulation
        # precision, matching DecoderStatisticsCallback below.
        self.register_buffer(
            "_mi",
            torch.zeros(num_iter + 1, dtype=torch.float64, device=self.device),
        )
        self.register_buffer(
            "_num_samples",
            torch.zeros(num_iter + 1, dtype=torch.int64, device=self.device),
        )

    @property
    def mi(self) -> torch.Tensor:
        """Mutual information after each iteration"""
        return self._mi / self._num_samples.to(torch.float64)

    def __call__(
        self,
        msg: torch.Tensor,
        it: int,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """Process messages and update EXIT statistics."""
        # Flatten messages and compute MI (exclude padded values)
        msg_flat = msg.reshape(-1)
        nonzero_mask = msg_flat != 0
        log_term = torch.log2(
            1.0 + torch.exp(torch.clamp(-msg_flat, min=-100.0, max=100.0))
        )
        num_values = nonzero_mask.sum()
        mean_term = (
            log_term * nonzero_mask.to(log_term.dtype)
        ).sum() / num_values.clamp_min(1).to(log_term.dtype)
        mi_val = torch.where(
            num_values > 0,
            1.0 - mean_term,
            torch.zeros((), dtype=log_term.dtype, device=log_term.device),
        )
        self._mi[it] = self._mi[it] + mi_val
        self._num_samples[it] = self._num_samples[it] + 1
        return msg


class DecoderStatisticsCallback(Object):
    """Callback for the LDPCBPDecoder to track decoder statistics.

    Can be registered as ``c2v_callbacks`` in the
    :class:`~sionna.phy.fec.ldpc.decoding.LDPCBPDecoder` and the
    :class:`~sionna.phy.fec.ldpc.decoding.LDPC5GDecoder`.

    Remark: the decoding statistics are based on CN convergence, i.e.,
    successful decoding is assumed if all check nodes are fulfilled.
    This overestimates the success-rate as it includes cases where the decoder
    converges to the wrong codeword.

    :param num_iter: Maximum number of decoding iterations.
    :param device: Device for computation (e.g., 'cpu', 'cuda:0').

    :input msg: [batch_size, num_vns, max_degree], `torch.float`.
        v2c messages.

    :input it: `int`.
        Current number of decoding iterations.

    :output msg: `torch.float`.
        Same as ``msg``.

    .. rubric:: Examples


    .. code-block:: python

        from sionna.phy.fec.ldpc import LDPCBPDecoder
        from sionna.phy.fec.ldpc.utils import DecoderStatisticsCallback

        # Create callback
        stats_cb = DecoderStatisticsCallback(num_iter=20)

        # Create decoder with callback
        decoder = LDPCBPDecoder(pcm, c2v_callbacks=[stats_cb])

        # After decoding, access statistics
        print(stats_cb.success_rate)
        print(stats_cb.avg_number_iterations)
    """

    def __init__(
        self,
        num_iter: int,
        device: Optional[str] = None,
    ):
        super().__init__(device=device)
        self._num_iter = num_iter
        self.register_buffer("_num_samples", torch.zeros(num_iter, dtype=torch.int64, device=self.device))
        self.register_buffer("_decoded_samples", torch.zeros(
            num_iter, dtype=torch.int64, device=self.device
        ))

    @property
    def num_samples(self) -> torch.Tensor:
        """Total number of processed codewords"""
        return self._num_samples

    @property
    def num_decoded_cws(self) -> torch.Tensor:
        """Number of decoded codewords after each iteration"""
        return self._decoded_samples

    @property
    def success_rate(self) -> torch.Tensor:
        """Success rate after each iteration"""
        succ = self._decoded_samples.to(torch.float64)
        num_samples = self._num_samples.to(torch.float64)
        return succ / num_samples

    @property
    def avg_number_iterations(self) -> torch.Tensor:
        """Average number of decoding iterations"""
        num_decoded = self._decoded_samples.to(torch.float64)
        num_samples = self._num_samples.to(torch.float64)

        num_active = num_samples - num_decoded
        total_iters = num_active.sum()
        avg_iter = total_iters / num_samples[0]
        return avg_iter

    def reset_stats(self) -> None:
        """Reset internal statistics"""
        self.register_buffer("_num_samples", torch.zeros(
            self._num_iter, dtype=torch.int64, device=self.device
        ))
        self.register_buffer("_decoded_samples", torch.zeros(
            self._num_iter, dtype=torch.int64, device=self.device
        ))

    def __call__(
        self,
        msg: torch.Tensor,
        it: int,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """Process messages and update decoder statistics."""
        # msg shape: [batch_size, num_nodes, max_degree]
        sign_val = torch.sign(msg)
        sign_val = torch.where(sign_val == 0, torch.ones_like(sign_val), sign_val)

        sign_node = sign_val.prod(dim=2)  # [bs, num_nodes]

        node_success = sign_node > 0  # [bs, num_nodes]
        cw_success = node_success.all(dim=1)  # [bs]

        num_decoded = cw_success.sum().to(torch.int64)
        batch_size = msg.shape[0]

        # Update statistics
        if it < self._num_iter:
            self._num_samples[it] = self._num_samples[it] + batch_size
            self._decoded_samples[it] = self._decoded_samples[it] + num_decoded

        return msg


class WeightedBPCallback(Object):
    # pylint: disable=line-too-long
    r"""Callback for the LDPCBPDecoder to enable weighted BP :cite:p:`Nachmani`.

    The BP decoder is fully differentiable and can be made trainable
    by following the concept of *weighted BP* :cite:p:`Nachmani` leading to

    .. math::
        y_{j \to i} = 2 \operatorname{tanh}^{-1} \left( \prod_{i' \in \mathcal{N}(j) \setminus i} \operatorname{tanh} \left( \frac{\textcolor{red}{w_{i' \to j}} \cdot x_{i' \to j}}{2} \right) \right)

    where :math:`w_{i \to j}` denotes the trainable weight of message
    :math:`x_{i \to j}`.
    Please note that the training of some check node types may be not supported.

    Can be registered as ``c2v_callbacks`` and ``v2c_callbacks`` in the
    :class:`~sionna.phy.fec.ldpc.decoding.LDPCBPDecoder` and the
    :class:`~sionna.phy.fec.ldpc.decoding.LDPC5GDecoder`.

    :param num_edges: Number of edges in the decoding graph.
    :param pcm: Optional parity-check matrix. If provided, enables weighted BP
        in padded message format used by the decoder.
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`, :py:attr:`~sionna.phy.config.precision` is used.
    :param device: Device for computation (e.g., 'cpu', 'cuda:0').

    :input msg: [batch_size, num_vns, max_degree], `torch.float`.
        v2c messages.

    :output msg: `torch.float`.
        Same as ``msg``.

    .. rubric:: Examples


    .. code-block:: python

        from sionna.phy.fec.ldpc import LDPCBPDecoder
        from sionna.phy.fec.ldpc.utils import WeightedBPCallback
        import numpy as np

        # Create a simple parity-check matrix
        pcm = np.array([[1, 1, 0, 1], [0, 1, 1, 1]])

        # Create callback with trainable weights
        weighted_cb = WeightedBPCallback(num_edges=np.sum(pcm), pcm=pcm)

        # Create decoder with callback
        decoder = LDPCBPDecoder(pcm, v2c_callbacks=[weighted_cb])

        # Access trainable weights
        print(weighted_cb.weights)
    """

    def __init__(
        self,
        num_edges: int,
        pcm: Optional[Union[np.ndarray, sp.spmatrix]] = None,
        precision: Optional[str] = None,
        device: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(precision=precision, device=device, **kwargs)

        # Note: Using nn.Parameter instead of register_buffer since requires_grad=True
        self._edge_weights = torch.nn.Parameter(torch.ones(
            num_edges, dtype=self.dtype, device=self.device
        ))

        # Build indices for padded format if PCM is provided
        self._has_pcm = pcm is not None
        if self._has_pcm:
            self._build_padded_indices(pcm)

    def _build_padded_indices(
        self, pcm: Union[np.ndarray, sp.spmatrix]
    ) -> None:
        """Build index arrays for mapping flat edge weights to padded format"""
        # Convert to sparse if needed
        if isinstance(pcm, np.ndarray):
            pcm_sparse = sp.csr_matrix(pcm)
        else:
            pcm_sparse = pcm

        # Get edge indices (same logic as in LDPCBPDecoder)
        cn_idx, vn_idx, _ = sp.find(pcm_sparse)

        # Sort by VN index (for VN-padded format)
        idx_vn_sorted = np.argsort(vn_idx)
        vn_idx_sorted = vn_idx[idx_vn_sorted]

        # Sort by CN index (for CN-padded format)
        idx_cn_sorted = np.argsort(cn_idx)

        num_vns = pcm.shape[1]
        num_cns = pcm.shape[0]

        # Compute row splits for VN perspective
        vn_row_splits = np.zeros(num_vns + 1, dtype=np.int32)
        for i in vn_idx_sorted:
            vn_row_splits[i + 1] += 1
        vn_row_splits = np.cumsum(vn_row_splits)

        # Compute row splits for CN perspective
        cn_idx_sorted = cn_idx[idx_cn_sorted]
        cn_row_splits = np.zeros(num_cns + 1, dtype=np.int32)
        for i in cn_idx_sorted:
            cn_row_splits[i + 1] += 1
        cn_row_splits = np.cumsum(cn_row_splits)

        # Compute max degrees
        vn_degrees = np.diff(vn_row_splits)
        cn_degrees = np.diff(cn_row_splits)
        max_vn_degree = int(vn_degrees.max()) if len(vn_degrees) > 0 else 0
        max_cn_degree = int(cn_degrees.max()) if len(cn_degrees) > 0 else 0

        # Build VN gather index: maps (vn, position) -> edge_index
        vn_gather_idx = np.zeros((num_vns, max_vn_degree), dtype=np.int32)
        for vn in range(num_vns):
            start = vn_row_splits[vn]
            end = vn_row_splits[vn + 1]
            degree = end - start
            if degree > 0:
                # idx_vn_sorted gives original edge indices sorted by VN
                vn_gather_idx[vn, :degree] = idx_vn_sorted[start:end]

        # Build CN gather index: maps (cn, position) -> edge_index
        cn_gather_idx = np.zeros((num_cns, max_cn_degree), dtype=np.int32)
        for cn in range(num_cns):
            start = cn_row_splits[cn]
            end = cn_row_splits[cn + 1]
            degree = end - start
            if degree > 0:
                cn_gather_idx[cn, :degree] = idx_cn_sorted[start:end]

        # Register as buffers
        self.register_buffer(
            "_vn_gather_idx",
            torch.tensor(vn_gather_idx, dtype=torch.int32, device=self.device)
        )
        self.register_buffer(
            "_cn_gather_idx",
            torch.tensor(cn_gather_idx, dtype=torch.int32, device=self.device)
        )
        self._num_vns = num_vns
        self._num_cns = num_cns
        self._max_vn_degree = max_vn_degree
        self._max_cn_degree = max_cn_degree

    @property
    def weights(self) -> torch.Tensor:
        """Trainable edge weights"""
        return self._edge_weights

    def show_weights(self, size: float = 7) -> None:
        """Show histogram of trainable weights.

        :param size: Figure size of the matplotlib figure.
        """
        plt.figure(figsize=(size, size))
        plt.hist(self._edge_weights.detach().cpu().numpy(), density=True, bins=20, align="mid")
        plt.xlabel("weight value")
        plt.ylabel("density")
        plt.grid(True, which="both", axis="both")
        plt.title("Weight Distribution")

    def __call__(
        self,
        msg: torch.Tensor,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """Multiply messages with trainable weights for weighted BP."""
        if msg.dim() == 2:
            # Flat format [batch_size, num_edges]
            msg = msg * self._edge_weights  # broadcasts [bs, num_edges] * [num_edges]
        elif msg.dim() == 3 and self._has_pcm:
            # Padded format [batch_size, num_nodes, max_degree]
            num_nodes = msg.shape[1]
            max_degree = msg.shape[2]

            if num_nodes == self._num_vns and max_degree == self._max_vn_degree:
                gather_idx = self._vn_gather_idx
            elif num_nodes == self._num_cns and max_degree == self._max_cn_degree:
                gather_idx = self._cn_gather_idx
            else:
                return msg

            weights_padded = self._edge_weights[gather_idx]  # [num_nodes, max_degree]

            # [bs, num_nodes, max_degree] * [1, num_nodes, max_degree]
            msg = msg * weights_padded.unsqueeze(0)

        return msg


