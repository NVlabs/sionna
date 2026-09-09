#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Blocks for channel decoding and utility functions."""

from typing import Callable, List, Optional, Tuple, Union
import types

import numpy as np
import scipy as sp
import torch

from sionna.phy import Block
from sionna.phy.fec.ldpc.encoding import LDPC5GEncoder


__all__ = [
    "LDPCBPDecoder",
    "LDPC5GDecoder",
    "vn_update_sum",
    "vn_node_update_identity",
    "cn_update_tanh",
    "cn_update_phi",
    "cn_update_minsum",
    "cn_update_offset_minsum",
    "cn_node_update_identity",
]


class LDPCBPDecoder(Block):
    # pylint: disable=line-too-long
    r"""Iterative belief propagation decoder for low-density parity-check (LDPC)
    codes and other codes on graphs.

    This class defines a generic belief propagation decoder for decoding
    with arbitrary parity-check matrices. It can be used to iteratively
    estimate/recover the transmitted codeword (or information bits) based on the
    LLR-values of the received noisy codeword observation.

    Per default, the decoder implements the flooding message passing algorithm
    :cite:p:`Ryan`, i.e., all nodes are updated in a parallel fashion. Different check
    node update functions are available:

    (1) `boxplus`

        .. math::
            y_{j \to i} = 2 \operatorname{tanh}^{-1} \left( \prod_{i' \in \mathcal{N}(j) \setminus i} \operatorname{tanh} \left( \frac{x_{i' \to j}}{2} \right) \right)

    (2) `boxplus-phi`

        .. math::
            y_{j \to i} = \alpha_{j \to i} \cdot \phi \left( \sum_{i' \in \mathcal{N}(j) \setminus i} \phi \left( |x_{i' \to j}|\right) \right)

        with :math:`\phi(x)=-\operatorname{log}\left(\operatorname{tanh}\left(\frac{x}{2}\right)\right)`

    (3) `minsum`

        .. math::
            \qquad y_{j \to i} = \alpha_{j \to i} \cdot \min_{i' \in \mathcal{N}(j) \setminus i} \left(|x_{i' \to j}|\right)

    (4) `offset-minsum`

    .. math::
            \qquad y_{j \to i} = \alpha_{j \to i} \cdot \max \left( \min_{i' \in \mathcal{N}(j) \setminus i} \left(|x_{i' \to j}| \right)-\beta , 0\right)

    where :math:`\beta=0.5` and :math:`y_{j \to i}` denotes the message
    from check node (CN) *j* to variable node (VN) *i* and :math:`x_{i \to j}`
    from VN *i* to CN *j*, respectively. Further, :math:`\mathcal{N}(j)`
    denotes all indices of connected VNs to CN *j* and

    .. math::
        \alpha_{j \to i} = \prod_{i' \in \mathcal{N}(j) \setminus i} \operatorname{sign}(x_{i' \to j})

    is the sign of the outgoing message. For further details we refer to
    :cite:p:`Ryan` and :cite:p:`Chen` for offset corrected minsum.

    Note that for full 5G 3GPP NR compatibility, the correct puncturing and
    shortening patterns must be applied (cf. :cite:p:`Richardson` for details), this
    can be done by :class:`~sionna.phy.fec.ldpc.encoding.LDPC5GEncoder` and
    :class:`~sionna.phy.fec.ldpc.decoding.LDPC5GDecoder`, respectively.

    If required, the decoder can be made trainable and is fully differentiable
    by following the concept of *weighted BP* :cite:p:`Nachmani`. For this, custom
    callbacks can be registered that scale the messages during decoding. Please
    see the corresponding tutorial notebook for details.

    For numerical stability, the decoder applies LLR clipping of +/- ``llr_max``
    to the input LLRs.

    :param pcm: An ndarray of shape `[n-k, n]` defining the parity-check
        matrix consisting only of `0` or `1` entries. Can also be of type
        `scipy.sparse.csr_matrix` or `scipy.sparse.csc_matrix`.
    :param cn_update: Check node update rule to be used as described above.
        One of "boxplus-phi" (default), "boxplus", "minsum", "offset-minsum",
        "identity", or a callable.
        If a callable is provided, it will be used instead as CN update.
        The input of the function is a tensor of v2c messages of shape
        `[batch_size, num_cns, max_degree]` with a mask of shape
        `[num_cns, max_degree]`.
    :param vn_update: Variable node update rule to be used.
        One of "sum" (default), "identity", or a callable.
        If a callable is provided, it will be used instead as VN update.
        The input of the function is a tensor of c2v messages of shape
        `[batch_size, num_vns, max_degree]` with a mask of shape
        `[num_vns, max_degree]`.
    :param cn_schedule: Defines the CN update scheduling per BP iteration.
        Can be either "flooding" to update all nodes in parallel (recommended)
        or a 2D tensor of shape `[num_update_steps, num_active_nodes]` where
        each row defines the node indices to be updated per subiteration.
        In this case each BP iteration runs ``num_update_steps``
        subiterations, thus the decoder's level of parallelization is lower
        and usually the decoding throughput decreases.
    :param hard_out: If `True`, the decoder provides hard-decided codeword
        bits instead of soft-values.
    :param num_iter: Defining the number of decoder iterations (due to
        batching, no early stopping used at the moment!).
    :param llr_max: Internal clipping value for all internal messages. If
        `None`, no clipping is applied.
    :param v2c_callbacks: Each callable will be executed after each VN update
        with the following arguments ``msg_vn``, ``it``, ``x_hat``, where
        ``msg_vn`` are the v2c messages as tensor of shape
        `[batch_size, num_vns, max_degree]`, ``x_hat`` is the current
        estimate of each VN of shape `[batch_size, num_vns]`, and ``it`` is
        the current iteration counter.
        It must return an updated version of ``msg_vn`` of same shape.
    :param c2v_callbacks: Each callable will be executed after each CN update
        with the following arguments ``msg_cn`` and ``it`` where ``msg_cn``
        are the c2v messages as tensor of shape
        `[batch_size, num_cns, max_degree]` and ``it`` is the current
        iteration counter.
        It must return an updated version of ``msg_cn`` of same shape.
    :param return_state: If `True`, the internal VN messages ``msg_vn`` from
        the last decoding iteration are returned, and ``msg_vn`` or `None`
        needs to be given as a second input when calling the decoder.
        This can be used for iterative demapping and decoding.
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`, :py:attr:`~sionna.phy.config.precision` is used.
    :param device: Device for computation (e.g., 'cpu', 'cuda:0').

    :input llr_ch: [..., n], `torch.float`.
        Tensor containing the channel logits/llr values.

    :input msg_v2c: `None` | [batch_size, num_edges], `torch.float`.
        Tensor of VN messages representing the internal decoder state.
        Required only if the decoder shall use its previous internal state,
        e.g., for iterative detection and decoding (IDD) schemes.

    :output x_hat: [..., n], `torch.float`.
        Tensor of same shape as ``llr_ch`` containing
        bit-wise soft-estimates (or hard-decided bit-values) of all
        codeword bits.

    :output msg_v2c: [batch_size, num_edges], `torch.float`.
        Tensor of VN messages representing the internal decoder state.
        Returned only if ``return_state`` is set to `True`.

    .. rubric:: Notes

    As decoding input logits :math:`\operatorname{log} \frac{p(x=1)}{p(x=0)}`
    are assumed for compatibility with the learning framework, but internally
    log-likelihood ratios (LLRs) with definition
    :math:`\operatorname{log} \frac{p(x=0)}{p(x=1)}` are used.

    The decoder is not (particularly) optimized for quasi-cyclic (QC) LDPC
    codes and, thus, supports arbitrary parity-check matrices.

    .. rubric:: Examples


    .. code-block:: python

        import torch
        from sionna.phy.fec.utils import load_parity_check_examples
        from sionna.phy.fec.ldpc import LDPCBPDecoder

        # Load (7,4) Hamming code
        pcm, k, n, _ = load_parity_check_examples(0)
        decoder = LDPCBPDecoder(pcm, num_iter=10)

        # Decode random LLRs
        llr_ch = torch.randn(100, n) * 2.0
        c_hat = decoder(llr_ch)
        print(c_hat.shape)
        # torch.Size([100, 7])
    """

    def __init__(
        self,
        pcm: Union[np.ndarray, sp.sparse.csr_matrix, sp.sparse.csc_matrix],
        cn_update: Union[str, Callable] = "boxplus-phi",
        vn_update: Union[str, Callable] = "sum",
        cn_schedule: Union[str, np.ndarray, torch.Tensor] = "flooding",
        hard_out: bool = True,
        num_iter: int = 20,
        llr_max: Optional[float] = 20.0,
        v2c_callbacks: Optional[List[Callable]] = None,
        c2v_callbacks: Optional[List[Callable]] = None,
        return_state: bool = False,
        precision: Optional[str] = None,
        device: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(precision=precision, device=device, **kwargs)

        # Check inputs for consistency
        if not isinstance(hard_out, bool):
            raise TypeError("hard_out must be bool.")
        if not isinstance(num_iter, int):
            raise TypeError("num_iter must be int.")
        if num_iter < 0:
            raise ValueError("num_iter cannot be negative.")
        if not isinstance(return_state, bool):
            raise TypeError("return_state must be bool.")

        if isinstance(pcm, np.ndarray):
            if not np.array_equal(pcm, pcm.astype(bool)):
                raise ValueError("PC matrix must be binary.")
        elif isinstance(pcm, sp.sparse.csr_matrix):
            if not np.array_equal(pcm.data, pcm.data.astype(bool)):
                raise ValueError("PC matrix must be binary.")
        elif isinstance(pcm, sp.sparse.csc_matrix):
            if not np.array_equal(pcm.data, pcm.data.astype(bool)):
                raise ValueError("PC matrix must be binary.")
        else:
            raise TypeError("Unsupported dtype of pcm.")

        # Deprecation warning for cn_type
        if "cn_type" in kwargs:
            raise TypeError("'cn_type' is deprecated; use 'cn_update' instead.")

        # Init decoder parameters
        self._pcm = pcm
        self._hard_out = hard_out
        self._num_iter = num_iter
        self._return_state = return_state

        self._num_cns = pcm.shape[0]  # total number of check nodes
        self._num_vns = pcm.shape[1]  # total number of variable nodes

        # Internal value for LLR clipping
        if llr_max is not None and not isinstance(llr_max, (int, float)):
            raise TypeError("llr_max must be int or float.")
        self._llr_max = float(llr_max) if llr_max is not None else None

        if v2c_callbacks is None:
            self._v2c_callbacks = []
        else:
            if isinstance(v2c_callbacks, (list, tuple)):
                self._v2c_callbacks = list(v2c_callbacks)
            elif isinstance(v2c_callbacks, types.FunctionType):
                self._v2c_callbacks = [v2c_callbacks]
            else:
                raise TypeError("v2c_callbacks must be a list of callables.")

        if c2v_callbacks is None:
            self._c2v_callbacks = []
        else:
            if isinstance(c2v_callbacks, (list, tuple)):
                self._c2v_callbacks = list(c2v_callbacks)
            elif isinstance(c2v_callbacks, types.FunctionType):
                self._c2v_callbacks = [c2v_callbacks]
            else:
                raise TypeError("c2v_callbacks must be a list of callables.")

        # Make pcm sparse first if ndarray is provided
        if isinstance(pcm, np.ndarray):
            pcm_sparse = sp.sparse.csr_matrix(pcm)
        else:
            pcm_sparse = pcm

        # Assign all edges to CN and VN nodes, respectively
        cn_idx, vn_idx, _ = sp.sparse.find(pcm_sparse)

        # Sort indices explicitly (scipy.sparse.find changed from column to
        # row sorting in scipy>=1.11)
        idx = np.argsort(vn_idx)
        self._cn_idx = cn_idx[idx]
        self._vn_idx = vn_idx[idx]

        # Number of edges equals number of non-zero elements in PCM
        self._num_edges = len(self._vn_idx)

        # Pre-load the CN function
        if cn_update == "boxplus":
            self._cn_update = cn_update_tanh
        elif cn_update == "boxplus-phi":
            self._cn_update = cn_update_phi
        elif cn_update in ("minsum", "min"):
            self._cn_update = cn_update_minsum
        elif cn_update == "offset-minsum":
            self._cn_update = cn_update_offset_minsum
        elif cn_update == "identity":
            self._cn_update = cn_node_update_identity
        elif callable(cn_update):
            self._cn_update = cn_update
        else:
            raise TypeError("Provided cn_update not supported.")

        # Pre-load the VN function
        if vn_update == "sum":
            self._vn_update = vn_update_sum
        elif vn_update == "identity":
            self._vn_update = vn_node_update_identity
        elif callable(vn_update):
            self._vn_update = vn_update
        else:
            raise TypeError("Provided vn_update not supported.")

        ######################
        # Init graph structure
        ######################

        # Handle scheduling
        # Register as buffers for CUDAGraph compatibility
        if isinstance(cn_schedule, str) and cn_schedule == "flooding":
            self._scheduling = "flooding"
            self.register_buffer(
                "_cn_schedule",
                torch.arange(
                    self._num_cns, dtype=torch.int32, device=self.device
                ).unsqueeze(0),
            )
        elif isinstance(cn_schedule, (np.ndarray, torch.Tensor)):
            if isinstance(cn_schedule, np.ndarray):
                cn_schedule = torch.tensor(
                    cn_schedule, dtype=torch.int32, device=self.device
                )
            else:
                cn_schedule = cn_schedule.to(dtype=torch.int32, device=self.device)
            self._scheduling = "custom"
            if len(cn_schedule.shape) != 2:
                raise ValueError("cn_schedule must be of rank 2.")
            if cn_schedule.max() >= self._num_cns:
                raise ValueError(
                    "cn_schedule can only contain values smaller than num_cns."
                )
            if cn_schedule.min() < 0:
                raise ValueError("cn_schedule cannot contain negative values.")
            self.register_buffer("_cn_schedule", cn_schedule)
        else:
            raise ValueError("cn_schedule can be 'flooding' or an array of ints.")

        # Build index arrays for message permutation
        # Permutation index to rearrange edge messages into CN perspective
        v2c_perm = np.argsort(self._cn_idx)
        # And the inverse operation
        v2c_perm_inv = np.argsort(v2c_perm)

        self.register_buffer(
            "_v2c_perm", torch.tensor(v2c_perm, dtype=torch.int32, device=self.device)
        )
        self.register_buffer(
            "_v2c_perm_inv",
            torch.tensor(v2c_perm_inv, dtype=torch.int32, device=self.device),
        )
        self.register_buffer(
            "_vn_idx_t",
            torch.tensor(self._vn_idx, dtype=torch.int32, device=self.device),
        )
        self.register_buffer(
            "_cn_idx_t",
            torch.tensor(self._cn_idx, dtype=torch.int32, device=self.device),
        )

        # Compute row splits for CN perspective (after v2c_perm)
        cn_idx_sorted = self._cn_idx[v2c_perm]
        cn_row_splits = self._compute_row_splits(cn_idx_sorted, self._num_cns)
        self.register_buffer(
            "_cn_row_splits",
            torch.tensor(cn_row_splits, dtype=torch.int32, device=self.device),
        )

        # Compute row splits for VN perspective
        vn_row_splits = self._compute_row_splits(self._vn_idx, self._num_vns)
        self.register_buffer(
            "_vn_row_splits",
            torch.tensor(vn_row_splits, dtype=torch.int32, device=self.device),
        )

        # Compute max degrees for padding
        cn_degrees = np.diff(cn_row_splits)
        vn_degrees = np.diff(vn_row_splits)
        self._max_cn_degree = int(cn_degrees.max()) if len(cn_degrees) > 0 else 0
        self._max_vn_degree = int(vn_degrees.max()) if len(vn_degrees) > 0 else 0

        # Build padded index arrays for vectorized operations
        self._cn_gather_idx, self._cn_mask = self._build_padded_indices(
            cn_idx_sorted, cn_row_splits, self._num_cns, self._max_cn_degree
        )
        self._vn_gather_idx, self._vn_mask = self._build_padded_indices(
            self._vn_idx, vn_row_splits, self._num_vns, self._max_vn_degree
        )

        # Build scatter indices for CN update
        # This maps from padded CN messages back to edge format
        self._cn_scatter_idx = self._build_scatter_indices(
            cn_row_splits, self._num_cns, self._max_cn_degree
        )

        # Build scatter indices for VN update
        self._vn_scatter_idx = self._build_scatter_indices(
            vn_row_splits, self._num_vns, self._max_vn_degree
        )

        # Precompute valid position indices for scatter operations (avoids dynamic shapes)
        # For CN: which positions in the flattened padded array are valid
        cn_valid_positions, cn_valid_edge_idx = self._build_valid_scatter_indices(
            cn_row_splits, self._num_cns, self._max_cn_degree
        )
        self.register_buffer("_cn_valid_positions", cn_valid_positions)
        self.register_buffer("_cn_valid_edge_idx", cn_valid_edge_idx)

        # For VN: which positions in the flattened padded array are valid
        vn_valid_positions, vn_valid_edge_idx = self._build_valid_scatter_indices(
            vn_row_splits, self._num_vns, self._max_vn_degree
        )
        self.register_buffer("_vn_valid_positions", vn_valid_positions)
        self.register_buffer("_vn_valid_edge_idx", vn_valid_edge_idx)

        # Precompute per-subiteration index arrays for custom scheduling
        if self._scheduling == "custom":
            self._build_custom_schedule_indices(cn_row_splits, v2c_perm)

    def _compute_row_splits(self, idx: np.ndarray, num_nodes: int) -> np.ndarray:
        """Compute row splits from sorted indices."""
        row_splits = np.zeros(num_nodes + 1, dtype=np.int32)
        for i in idx:
            row_splits[i + 1] += 1
        row_splits = np.cumsum(row_splits)
        return row_splits

    def _build_padded_indices(
        self, idx: np.ndarray, row_splits: np.ndarray, num_nodes: int, max_degree: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build padded gather indices and mask for vectorized operations."""
        # Create padded index array
        gather_idx = np.zeros((num_nodes, max_degree), dtype=np.int32)
        mask = np.zeros((num_nodes, max_degree), dtype=np.float32)

        for node in range(num_nodes):
            start = row_splits[node]
            end = row_splits[node + 1]
            degree = end - start
            if degree > 0:
                gather_idx[node, :degree] = np.arange(start, end)
                mask[node, :degree] = 1.0

        return (
            torch.tensor(gather_idx, dtype=torch.int32, device=self.device),
            torch.tensor(mask, dtype=self.dtype, device=self.device),
        )

    def _build_scatter_indices(
        self, row_splits: np.ndarray, num_nodes: int, max_degree: int
    ) -> torch.Tensor:
        """Build scatter indices for converting padded format back to flat."""
        scatter_idx = np.zeros((num_nodes, max_degree), dtype=np.int32)
        for node in range(num_nodes):
            start = row_splits[node]
            end = row_splits[node + 1]
            degree = end - start
            if degree > 0:
                scatter_idx[node, :degree] = np.arange(start, end)
        return torch.tensor(scatter_idx, dtype=torch.int32, device=self.device)

    def _build_valid_scatter_indices(
        self, row_splits: np.ndarray, num_nodes: int, max_degree: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build precomputed valid position and edge indices for scatter operations.

        This precomputes which positions in the flattened padded format are valid
        and their corresponding edge indices, avoiding dynamic shape operations
        during forward pass.

        :param row_splits: Row splits array for the nodes.
        :param num_nodes: Number of nodes.
        :param max_degree: Maximum degree (padding size).

        :output valid_positions: [num_edges] indices into flattened padded
            array.

        :output valid_edge_idx: [num_edges] corresponding edge indices.
        """
        valid_positions = []
        valid_edge_idx = []

        for node in range(num_nodes):
            start = row_splits[node]
            end = row_splits[node + 1]
            degree = end - start
            for d in range(degree):
                # Position in flattened padded array: node * max_degree + d
                flat_pos = node * max_degree + d
                # Corresponding edge index
                edge_idx = start + d
                valid_positions.append(flat_pos)
                valid_edge_idx.append(edge_idx)

        return (
            torch.tensor(valid_positions, dtype=torch.int32, device=self.device),
            torch.tensor(valid_edge_idx, dtype=torch.int32, device=self.device),
        )

    def _build_custom_schedule_indices(
        self, cn_row_splits: np.ndarray, v2c_perm: np.ndarray
    ) -> None:
        """Build index arrays for custom/layered CN scheduling.

        For each sub-iteration, we need indices to:
        1. Gather messages for only the active CNs
        2. Scatter updated messages back to the correct edge positions

        v2c_perm[cn_order_pos] gives the VN-order index of the edge at
        CN-order position cn_order_pos.
        """
        num_sub_iters = self._cn_schedule.shape[0]
        self._schedule_gather_idx = []
        self._schedule_cn_mask = []
        self._schedule_edge_idx = []  # Edge indices in VN order for scatter update
        self._schedule_valid_positions = (
            []
        )  # Precomputed valid positions (avoids dynamic shapes)

        cn_schedule_np = self._cn_schedule.cpu().numpy()

        for j in range(num_sub_iters):
            active_cns = cn_schedule_np[j]
            num_active = len(active_cns)

            # Compute max degree for active CNs in this sub-iteration
            active_degrees = []
            for cn in active_cns:
                start = cn_row_splits[cn]
                end = cn_row_splits[cn + 1]
                active_degrees.append(end - start)
            max_active_degree = max(active_degrees) if active_degrees else 0

            # Build gather indices for active CNs
            gather_idx = np.zeros((num_active, max_active_degree), dtype=np.int32)
            mask = np.zeros((num_active, max_active_degree), dtype=np.float32)
            edge_indices = []  # Edge indices in VN order
            valid_positions = []  # Valid positions in flattened padded array

            for i, cn in enumerate(active_cns):
                start = cn_row_splits[cn]
                end = cn_row_splits[cn + 1]
                degree = end - start
                if degree > 0:
                    gather_idx[i, :degree] = np.arange(start, end)
                    mask[i, :degree] = 1.0
                    # Edge positions in CN order
                    edge_cn_order = np.arange(start, end)
                    # v2c_perm[cn_pos] gives the VN-order index of edge at cn_pos
                    edge_vn_order = v2c_perm[edge_cn_order]
                    edge_indices.extend(edge_vn_order.tolist())
                    # Precompute valid positions in flattened format
                    for d in range(degree):
                        valid_positions.append(i * max_active_degree + d)

            self._schedule_gather_idx.append(
                torch.tensor(gather_idx, dtype=torch.int32, device=self.device)
            )
            self._schedule_cn_mask.append(
                torch.tensor(mask, dtype=self.dtype, device=self.device)
            )
            self._schedule_edge_idx.append(
                torch.tensor(edge_indices, dtype=torch.int32, device=self.device)
            )
            self._schedule_valid_positions.append(
                torch.tensor(valid_positions, dtype=torch.int32, device=self.device)
            )

    ###############################
    # Public methods and properties
    ###############################

    @property
    def pcm(self) -> Union[np.ndarray, sp.sparse.csr_matrix]:
        """Parity-check matrix of LDPC code."""
        return self._pcm

    @property
    def num_cns(self) -> int:
        """Number of check nodes."""
        return self._num_cns

    @property
    def num_vns(self) -> int:
        """Number of variable nodes."""
        return self._num_vns

    @property
    def n(self) -> int:
        """Codeword length."""
        return self._num_vns

    @property
    def coderate(self) -> float:
        """Coderate assuming independent parity checks."""
        return (self._num_vns - self._num_cns) / self._num_vns

    @property
    def num_edges(self) -> int:
        """Number of edges in decoding graph."""
        return self._num_edges

    @property
    def num_iter(self) -> int:
        """Number of decoding iterations."""
        return self._num_iter

    @num_iter.setter
    def num_iter(self, num_iter: int) -> None:
        """Set number of decoding iterations."""
        if not isinstance(num_iter, int):
            raise TypeError("num_iter must be int.")
        if num_iter < 0:
            raise ValueError("num_iter cannot be negative.")
        self._num_iter = num_iter

    @property
    def llr_max(self) -> Optional[float]:
        """Max LLR value used for internal calculations."""
        return self._llr_max

    @llr_max.setter
    def llr_max(self, value: float) -> None:
        """Set max LLR value."""
        if value is not None and value < 0:
            raise ValueError("llr_max cannot be negative.")
        self._llr_max = float(value) if value is not None else None

    @property
    def return_state(self) -> bool:
        """Return internal decoder state for IDD schemes."""
        return self._return_state

    #########################
    # Decoding functions
    #########################

    def _gather_to_cn(self, msg_v2c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Gather v2c messages to CN perspective with padding.

        :param msg_v2c: [batch_size, num_edges] tensor of v2c messages.

        :output msg_cn: [batch_size, num_cns, max_cn_degree] tensor of CN
            messages.

        :output mask: [num_cns, max_cn_degree] mask tensor.
        """
        msg_cn_flat = msg_v2c[:, self._v2c_perm]  # [bs, num_edges]
        msg_cn = msg_cn_flat[
            :, self._cn_gather_idx
        ]  # [bs, num_cns, max_cn_degree]

        # mask [num_cns, max_cn_degree] broadcasts with [bs, num_cns, max_deg]
        msg_cn = msg_cn * self._cn_mask

        return msg_cn, self._cn_mask

    def _scatter_from_cn(
        self, msg_cn: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Scatter CN messages back to flat edge format.

        :param msg_cn: [batch_size, num_cns, max_cn_degree] tensor.
        :param mask: [num_cns, max_cn_degree] mask tensor (unused, kept for API).

        :output msg_c2v: [batch_size, num_edges] tensor in VN order.
        """
        batch_size = msg_cn.shape[0]

        msg_flat = msg_cn.reshape(
            batch_size, -1
        )  # [bs, num_cns * max_cn_degree]

        msg_c2v = torch.zeros(
            batch_size, self._num_edges, dtype=msg_cn.dtype, device=msg_cn.device
        )

        valid_msg = msg_flat[:, self._cn_valid_positions]  # [bs, num_edges]
        msg_c2v[:, self._cn_valid_edge_idx] = valid_msg

        msg_c2v = msg_c2v[:, self._v2c_perm_inv]

        return msg_c2v

    def _gather_to_vn(self, msg_c2v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Gather c2v messages to VN perspective with padding.

        :param msg_c2v: [batch_size, num_edges] tensor of c2v messages (VN order).

        :output msg_vn: [batch_size, num_vns, max_vn_degree] padded VN messages.

        :output mask: [num_vns, max_vn_degree] VN mask tensor.
        """
        msg_vn = msg_c2v[
            :, self._vn_gather_idx
        ]  # [bs, num_vns, max_vn_degree]

        # mask [num_vns, max_vn_degree] broadcasts with [bs, num_vns, max_deg]
        msg_vn = msg_vn * self._vn_mask

        return msg_vn, self._vn_mask

    def _scatter_from_vn(
        self, msg_vn: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Scatter VN messages back to flat edge format.

        :param msg_vn: [batch_size, num_vns, max_vn_degree] tensor.
        :param mask: [num_vns, max_vn_degree] mask tensor (unused, kept for API).

        :output msg_v2c: [batch_size, num_edges] tensor.
        """
        batch_size = msg_vn.shape[0]

        msg_flat = msg_vn.reshape(batch_size, -1)

        msg_v2c = torch.zeros(
            batch_size, self._num_edges, dtype=msg_vn.dtype, device=msg_vn.device
        )

        valid_msg = msg_flat[:, self._vn_valid_positions]  # [bs, num_edges]
        msg_v2c[:, self._vn_valid_edge_idx] = valid_msg

        return msg_v2c

    def _bp_iter(
        self,
        msg_v2c: torch.Tensor,
        msg_c2v: torch.Tensor,
        llr_ch: torch.Tensor,
        x_hat: torch.Tensor,
        it: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Main decoding iteration.

        :param msg_v2c: [batch_size, num_edges] v2c messages.
        :param msg_c2v: [batch_size, num_edges] c2v messages.
        :param llr_ch: [batch_size, num_vns] channel LLRs.
        :param x_hat: [batch_size, num_vns] current estimate.
        :param it: Current iteration number.

        :output msg_v2c: Updated v2c messages.

        :output msg_c2v: Updated c2v messages.

        :output x_hat: Updated VN estimates.
        """
        # Process all sub-iterations
        for j in range(self._cn_schedule.shape[0]):
            if self._scheduling == "flooding":
                # Flooding: update all CNs in parallel
                # Gather messages to CN perspective
                msg_cn, cn_mask = self._gather_to_cn(msg_v2c)

                # Apply CN update
                msg_cn_out = self._cn_update(msg_cn, cn_mask, self._llr_max)

                # Apply CN callbacks
                for cb in self._c2v_callbacks:
                    msg_cn_out = cb(msg_cn_out, it)

                # Scatter back to edge format
                msg_c2v = self._scatter_from_cn(msg_cn_out, cn_mask)
            else:
                # Custom/layered scheduling: update only active CNs
                msg_c2v = self._bp_iter_custom_cn(msg_v2c, msg_c2v, j, it)

            # Gather messages to VN perspective
            msg_vn, vn_mask = self._gather_to_vn(msg_c2v)

            # Apply VN update
            msg_vn_out, x_hat = self._vn_update(msg_vn, vn_mask, llr_ch, self._llr_max)

            # Apply VN callbacks
            for cb in self._v2c_callbacks:
                msg_vn_out = cb(msg_vn_out, it + 1, x_hat)

            # Scatter back to edge format
            msg_v2c = self._scatter_from_vn(msg_vn_out, vn_mask)

        return msg_v2c, msg_c2v, x_hat

    def _bp_iter_custom_cn(
        self,
        msg_v2c: torch.Tensor,
        msg_c2v: torch.Tensor,
        sub_iter: int,
        it: int,
    ) -> torch.Tensor:
        """Process CN update for custom scheduling (only active CNs).

        :param msg_v2c: [batch_size, num_edges] v2c messages in VN order.
        :param msg_c2v: [batch_size, num_edges] current c2v messages.
        :param sub_iter: Sub-iteration index (which CNs to update).
        :param it: Current iteration number.

        :output msg_c2v: Updated c2v messages.
        """
        batch_size = msg_v2c.shape[0]
        gather_idx = self._schedule_gather_idx[sub_iter]
        cn_mask = self._schedule_cn_mask[sub_iter]
        edge_idx = self._schedule_edge_idx[sub_iter]
        valid_positions = self._schedule_valid_positions[sub_iter]

        msg_v2c_cn_order = msg_v2c[:, self._v2c_perm]

        msg_cn = msg_v2c_cn_order[
            :, gather_idx
        ]  # [bs, num_active_cns, max_degree]

        # mask [num_active_cns, max_degree] broadcasts with [bs, ...]
        msg_cn = msg_cn * cn_mask

        msg_cn_out = self._cn_update(msg_cn, cn_mask, self._llr_max)

        for cb in self._c2v_callbacks:
            msg_cn_out = cb(msg_cn_out, it)

        msg_flat = msg_cn_out.reshape(batch_size, -1)
        valid_msg = msg_flat[:, valid_positions]

        msg_c2v = msg_c2v.clone()
        msg_c2v[:, edge_idx] = valid_msg

        return msg_c2v

    def build(self, input_shape: tuple, **kwargs) -> None:
        """Build block and validate input shape."""
        if input_shape[-1] != self._num_vns:
            raise ValueError("Last dimension must be of length n.")

    def call(
        self,
        llr_ch: torch.Tensor,
        /,
        *,
        num_iter: Optional[int] = None,
        msg_v2c: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Iterative BP decoding function.

        :param llr_ch: Channel LLRs of shape [..., n].
        :param num_iter: Number of iterations. If `None`, uses default.
        :param msg_v2c: Initial v2c messages for IDD schemes.

        :output x_hat: Decoded bits of shape [..., n].

        :output msg_v2c: Decoder state. Returned only if ``return_state``
            is `True`.
        """
        if num_iter is None:
            num_iter = self._num_iter

        # Clip LLRs for numerical stability
        if self._llr_max is not None:
            llr_ch = llr_ch.clamp(-self._llr_max, self._llr_max)

        # Reshape to support multi-dimensional inputs
        llr_ch_shape = list(llr_ch.shape)
        new_shape = [-1, self._num_vns]
        llr_ch_reshaped = llr_ch.reshape(new_shape)  # [batch_size, num_vns]

        # Logits are converted into "true" LLRs as usually done in literature
        llr_ch_reshaped = llr_ch_reshaped * -1.0

        # Initialize v2c messages
        if msg_v2c is None:
            msg_v2c = llr_ch_reshaped[:, self._vn_idx_t]  # [bs, num_edges]
        else:
            msg_v2c = msg_v2c * -1.0  # invert sign due to logit definition

        # Messages from CN perspective; initialized to zero
        msg_c2v = torch.zeros_like(msg_v2c)

        # Apply VN callbacks before first iteration
        if self._v2c_callbacks:
            msg_vn, vn_mask = self._gather_to_vn(msg_v2c)
            for cb in self._v2c_callbacks:
                msg_vn = cb(msg_vn, 0, llr_ch_reshaped)
            msg_v2c = self._scatter_from_vn(msg_vn, vn_mask)

        # Initialize x_hat
        x_hat = llr_ch_reshaped

        # Main decoding loop
        for it in range(num_iter):
            msg_v2c, msg_c2v, x_hat = self._bp_iter(
                msg_v2c, msg_c2v, llr_ch_reshaped, x_hat, it
            )

        if self._hard_out:
            # Hard decide decoder output
            x_hat = (x_hat <= 0).to(self.dtype)
        else:
            x_hat = x_hat * -1.0  # convert LLRs back into logits

        # Reshape to match original input dimensions
        output_shape = llr_ch_shape.copy()
        x_reshaped = x_hat.reshape(output_shape)

        if not self._return_state:
            return x_reshaped
        else:
            msg_v2c = msg_v2c * -1.0  # invert sign due to logit definition
            return x_reshaped, msg_v2c


#######################
# Node update functions
#######################


def vn_node_update_identity(
    msg_c2v: torch.Tensor,
    mask: torch.Tensor,
    llr_ch: torch.Tensor,
    llr_clipping: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # pylint: disable=line-too-long
    r"""Dummy variable node update function for testing.

    Behaves as an identity function and can be used for testing and debugging
    of message passing decoding.

    Marginalizes input messages and returns them as second output.

    :param msg_c2v: Tensor of shape `[batch_size, num_nodes, max_degree]`
        representing c2v messages.
    :param mask: Tensor of shape `[num_nodes, max_degree]` indicating valid
        edges.
    :param llr_ch: Tensor of shape `[batch_size, num_nodes]` containing the
        channel LLRs.
    :param llr_clipping: Clipping value used for internal processing. If
        `None`, no internal clipping is applied.
    """
    x_tot = msg_c2v.sum(dim=2) + llr_ch  # [bs, num_nodes]

    return msg_c2v, x_tot


def vn_update_sum(
    msg_c2v: torch.Tensor,
    mask: torch.Tensor,
    llr_ch: torch.Tensor,
    llr_clipping: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # pylint: disable=line-too-long
    r"""Variable node update function implementing the `sum` update.

    This function implements the (extrinsic) variable node update
    function. It takes the sum over all incoming messages ``msg`` excluding
    the intrinsic (= outgoing) message itself.

    Additionally, the channel LLR ``llr_ch`` is considered in each variable
    node.

    :param msg_c2v: Tensor of shape `[batch_size, num_nodes, max_degree]`
        representing c2v messages.
    :param mask: Tensor of shape `[num_nodes, max_degree]` indicating valid
        edges.
    :param llr_ch: Tensor of shape `[batch_size, num_nodes]` containing the
        channel LLRs.
    :param llr_clipping: Clipping value used for internal processing. If
        `None`, no internal clipping is applied.
    """
    x = msg_c2v.sum(dim=2)  # [bs, num_nodes]
    x_tot = x + llr_ch

    # Extrinsic message: total - intrinsic
    x_e = x_tot.unsqueeze(2) - msg_c2v  # [bs, num_nodes, max_degree]

    # mask [num_nodes, max_degree] broadcasts with [bs, num_nodes, max_degree]
    x_e = x_e * mask

    if llr_clipping is not None:
        x_e = x_e.clamp(-llr_clipping, llr_clipping)
        x_tot = x_tot.clamp(-llr_clipping, llr_clipping)

    return x_e, x_tot


def cn_node_update_identity(
    msg_v2c: torch.Tensor,
    mask: torch.Tensor,
    llr_clipping: Optional[float] = None,
) -> torch.Tensor:
    # pylint: disable=line-too-long
    r"""Dummy function that returns the first tensor without any processing.

    Used for testing and debugging of message passing decoding.

    :param msg_v2c: Tensor of shape `[batch_size, num_nodes, max_degree]`
        representing v2c messages.
    :param mask: Tensor of shape `[num_nodes, max_degree]` indicating valid
        edges.
    :param llr_clipping: Clipping value (unused).
    """
    return msg_v2c


def cn_update_offset_minsum(
    msg_v2c: torch.Tensor,
    mask: torch.Tensor,
    llr_clipping: Optional[float] = None,
    offset: float = 0.5,
) -> torch.Tensor:
    # pylint: disable=line-too-long
    r"""Check node update function implementing the offset corrected minsum.

    The function implements

    .. math::
            \qquad y_{j \to i} = \alpha_{j \to i} \cdot \max \left( \min_{i' \in \mathcal{N}(j) \setminus i} \left(|x_{i' \to j}| \right)-\beta , 0\right)

    where :math:`\beta=0.5` and :math:`y_{j \to i}` denotes the message from
    check node (CN) *j* to variable node (VN) *i* and :math:`x_{i \to j}` from
    VN *i* to CN *j*, respectively. Further, :math:`\mathcal{N}(j)` denotes
    all indices of connected VNs to CN *j* and

    .. math::
        \alpha_{j \to i} = \prod_{i' \in \mathcal{N}(j) \setminus i} \operatorname{sign}(x_{i' \to j})

    is the sign of the outgoing message. For further details we refer to
    :cite:p:`Chen`.

    :param msg_v2c: Tensor of shape `[batch_size, num_nodes, max_degree]`
        representing v2c messages.
    :param mask: Tensor of shape `[num_nodes, max_degree]` indicating valid
        edges.
    :param llr_clipping: Clipping value used for internal processing. If
        `None`, no internal clipping is applied.
    :param offset: Offset value to be subtracted from each outgoing message.
    """
    large_val = 1e6

    # mask [num_nodes, max_degree] broadcasts with [bs, num_nodes, max_degree]
    inv_mask = 1.0 - mask

    sign_val = torch.sign(msg_v2c)
    sign_val = torch.where(sign_val == 0, torch.ones_like(sign_val), sign_val)

    # For padded positions, set sign to 1 (neutral for product)
    sign_val = sign_val * mask + inv_mask

    sign_node = sign_val.prod(dim=2, keepdim=True)  # [bs, num_nodes, 1]

    # Extrinsic sign: total sign / intrinsic sign
    sign_out = sign_node * sign_val  # [bs, num_nodes, max_degree]

    msg_abs = torch.abs(msg_v2c)

    # For padded positions, set to large value so they don't affect min
    msg_abs_masked = msg_abs * mask + large_val * inv_mask

    min_val, _ = msg_abs_masked.min(dim=2, keepdim=True)  # [bs, num_nodes, 1]

    msg_min1 = (msg_abs_masked - min_val) * mask + large_val * inv_mask

    is_min_position = msg_min1 == 0

    num_min_positions = (is_min_position * mask).sum(dim=2, keepdim=True)

    double_min = (num_min_positions > 1).to(msg_v2c.dtype)

    msg_for_second_min = torch.where(
        is_min_position, torch.full_like(msg_min1, large_val), msg_min1
    )

    min_val_2, _ = msg_for_second_min.min(dim=2, keepdim=True)
    min_val_2 = min_val_2 + min_val

    min_val_e = (1 - double_min) * min_val_2 + double_min * min_val

    msg_e = torch.where(is_min_position, min_val_e, min_val)

    msg_e = torch.clamp(msg_e - offset, min=0)

    msg_e = msg_e * mask

    msg_out = sign_out * msg_e

    if llr_clipping is not None:
        msg_out = msg_out.clamp(-llr_clipping, llr_clipping)

    return msg_out


def cn_update_minsum(
    msg_v2c: torch.Tensor,
    mask: torch.Tensor,
    llr_clipping: Optional[float] = None,
) -> torch.Tensor:
    # pylint: disable=line-too-long
    r"""Check node update function implementing the `minsum` update.

    The function implements

    .. math::
            \qquad y_{j \to i} = \alpha_{j \to i} \cdot \min_{i' \in \mathcal{N}(j) \setminus i} \left(|x_{i' \to j}|\right)

    where :math:`y_{j \to i}` denotes the message from check node (CN) *j* to
    variable node (VN) *i* and :math:`x_{i \to j}` from VN *i* to CN *j*,
    respectively. Further, :math:`\mathcal{N}(j)` denotes all indices of
    connected VNs to CN *j* and

    .. math::
        \alpha_{j \to i} = \prod_{i' \in \mathcal{N}(j) \setminus i} \operatorname{sign}(x_{i' \to j})

    is the sign of the outgoing message. For further details we refer to
    :cite:p:`Ryan` and :cite:p:`Chen`.

    :param msg_v2c: Tensor of shape `[batch_size, num_nodes, max_degree]`
        representing v2c messages.
    :param mask: Tensor of shape `[num_nodes, max_degree]` indicating valid
        edges.
    :param llr_clipping: Clipping value used for internal processing. If
        `None`, no internal clipping is applied.
    """
    return cn_update_offset_minsum(msg_v2c, mask, llr_clipping, offset=0.0)


def cn_update_tanh(
    msg_v2c: torch.Tensor,
    mask: torch.Tensor,
    llr_clipping: Optional[float] = None,
) -> torch.Tensor:
    # pylint: disable=line-too-long
    r"""Check node update function implementing the `boxplus` operation.

    This function implements the (extrinsic) check node update
    function. It calculates the boxplus function over all incoming messages
    "msg" excluding the intrinsic (=outgoing) message itself.
    The exact boxplus function is implemented by using the tanh function.

    The function implements

    .. math::
            y_{j \to i} = 2 \operatorname{tanh}^{-1} \left( \prod_{i' \in \mathcal{N}(j) \setminus i} \operatorname{tanh} \left( \frac{x_{i' \to j}}{2} \right) \right)

    where :math:`y_{j \to i}` denotes the message from check node (CN) *j* to
    variable node (VN) *i* and :math:`x_{i \to j}` from VN *i* to CN *j*,
    respectively. Further, :math:`\mathcal{N}(j)` denotes all indices of
    connected VNs to CN *j* and

    .. math::
        \alpha_{j \to i} = \prod_{i' \in \mathcal{N}(j) \setminus i} \operatorname{sign}(x_{i' \to j})

    is the sign of the outgoing message. For further details we refer to
    :cite:p:`Ryan`.

    Note that for numerical stability clipping can be applied.

    :param msg_v2c: Tensor of shape `[batch_size, num_nodes, max_degree]`
        representing v2c messages.
    :param mask: Tensor of shape `[num_nodes, max_degree]` indicating valid
        edges.
    :param llr_clipping: Clipping value used for internal processing. If
        `None`, no internal clipping is applied.
    """
    atanh_clip_value = 1 - 1e-7

    # mask [num_nodes, max_degree] broadcasts with [bs, num_nodes, max_degree]
    inv_mask = 1.0 - mask

    msg = msg_v2c / 2
    msg = torch.tanh(msg)

    msg = torch.where(msg == 0, torch.full_like(msg, 1e-12), msg)

    # For padded positions, set to 1 (neutral for product)
    msg = msg * mask + inv_mask

    msg_prod = msg.prod(dim=2, keepdim=True)  # [bs, num_nodes, 1]

    msg_recip = 1.0 / msg
    msg_e = msg_recip * msg_prod

    msg_e = torch.where(torch.abs(msg_e) < 1e-7, torch.zeros_like(msg_e), msg_e)

    msg_e = msg_e.clamp(-atanh_clip_value, atanh_clip_value)

    msg_out = 2 * torch.atanh(msg_e)

    msg_out = msg_out * mask

    if llr_clipping is not None:
        msg_out = msg_out.clamp(-llr_clipping, llr_clipping)

    return msg_out


def cn_update_phi(
    msg_v2c: torch.Tensor,
    mask: torch.Tensor,
    llr_clipping: Optional[float] = None,
) -> torch.Tensor:
    # pylint: disable=line-too-long
    r"""Check node update function implementing the `boxplus` operation.

    This function implements the (extrinsic) check node update function
    based on the numerically more stable `"_phi"` function (cf. :cite:p:`Ryan`).
    It calculates the boxplus function over all incoming messages ``msg``
    excluding the intrinsic (=outgoing) message itself.
    The exact boxplus function is implemented by using the `"_phi"` function
    as in :cite:p:`Ryan`.

    The function implements

    .. math::
            y_{j \to i} = \alpha_{j \to i} \cdot \phi \left( \sum_{i' \in \mathcal{N}(j) \setminus i} \phi \left( |x_{i' \to j}|\right) \right)

    where :math:`\phi(x)=-\operatorname{log}\left(\operatorname{tanh}\left(\frac{x}{2}\right)\right)`
    and :math:`y_{j \to i}` denotes the message from check node
    (CN) *j* to variable node (VN) *i* and :math:`x_{i \to j}` from VN *i* to
    CN *j*, respectively. Further, :math:`\mathcal{N}(j)` denotes all indices
    of connected VNs to CN *j* and

    .. math::
        \alpha_{j \to i} = \prod_{i' \in \mathcal{N}(j) \setminus i} \operatorname{sign}(x_{i' \to j})

    is the sign of the outgoing message. For further details we refer to
    :cite:p:`Ryan`.

    Note that for numerical stability clipping can be applied.

    :param msg_v2c: Tensor of shape `[batch_size, num_nodes, max_degree]`
        representing v2c messages.
    :param mask: Tensor of shape `[num_nodes, max_degree]` indicating valid
        edges.
    :param llr_clipping: Clipping value used for internal processing. If
        `None`, no internal clipping is applied.
    """

    def _phi(x: torch.Tensor) -> torch.Tensor:
        r"""Implements :math:`\phi(x)=-\operatorname{log}\left(\operatorname{tanh}\left(\frac{x}{2}\right)\right)`."""
        if x.dtype == torch.float32:
            x = x.clamp(min=8.5e-8, max=16.635532)
        elif x.dtype == torch.float64:
            x = x.clamp(min=1e-12, max=28.324079)
        else:
            x = x.clamp(min=1e-7, max=20.0)

        return torch.log(torch.exp(x) + 1) - torch.log(torch.exp(x) - 1)

    # mask [num_nodes, max_degree] broadcasts with [bs, num_nodes, max_degree]
    inv_mask = 1.0 - mask

    sign_val = torch.sign(msg_v2c)
    sign_val = torch.where(sign_val == 0, torch.ones_like(sign_val), sign_val)
    sign_val = sign_val * mask + inv_mask

    sign_node = sign_val.prod(dim=2, keepdim=True)  # [bs, num_nodes, 1]

    sign_out = sign_val * sign_node

    msg_abs = torch.abs(msg_v2c)

    msg_phi = _phi(msg_abs)

    # For padded positions, set to 0 (neutral for sum)
    msg_phi = msg_phi * mask

    msg_sum = msg_phi.sum(dim=2, keepdim=True)  # [bs, num_nodes, 1]

    # Extrinsic: total sum - intrinsic
    msg_e = msg_sum - msg_phi

    msg_e = _phi(msg_e)

    msg_out = sign_out * msg_e

    msg_out = msg_out * mask

    if llr_clipping is not None:
        msg_out = msg_out.clamp(-llr_clipping, llr_clipping)

    return msg_out


class LDPC5GDecoder(LDPCBPDecoder):
    # pylint: disable=line-too-long
    r"""Iterative belief propagation decoder for 5G NR LDPC codes.

    Inherits from :class:`~sionna.phy.fec.ldpc.decoding.LDPCBPDecoder` and
    provides a wrapper for 5G compatibility, i.e., automatically handles
    rate-matching according to :cite:p:`3GPPTS38212`.

    Note that for full 5G 3GPP NR compatibility, the correct puncturing and
    shortening patterns must be applied and, thus, the encoder object is
    required as input.

    If required the decoder can be made trainable and is differentiable
    (the training of some check node types may be not supported) following the
    concept of "weighted BP" :cite:p:`Nachmani`.

    When ``harq_mode=True``, the decoder supports HARQ-IR decoding.
    Pass ``rv`` to :meth:`call` together with stacked LLRs of shape
    ``[..., num_rvs, n]`` to combine multiple redundancy versions
    before BP decoding.  PCM pruning is automatically disabled in
    HARQ mode.

    :param encoder: An instance of
        :class:`~sionna.phy.fec.ldpc.encoding.LDPC5GEncoder` containing the
        correct code parameters.
    :param cn_update: Check node update rule to be used as described above.
        One of "boxplus-phi" (default), "boxplus", "minsum", "offset-minsum",
        "identity", or a callable.
        If a callable is provided, it will be used instead as CN update.
        The input of the function is a tensor of v2c messages of shape
        `[batch_size, num_cns, max_degree]` with a mask of shape
        `[num_cns, max_degree]`.
    :param vn_update: Variable node update rule to be used.
        One of "sum" (default), "identity", or a callable.
        If a callable is provided, it will be used instead as VN update.
        The input of the function is a tensor of c2v messages of shape
        `[batch_size, num_vns, max_degree]` with a mask of shape
        `[num_vns, max_degree]`.
    :param cn_schedule: Defines the CN update scheduling per BP iteration.
        Can be either "flooding" to update all nodes in parallel (recommended)
        or "layered" to sequentially update all CNs in the same lifting group
        together or a 2D tensor of shape
        `[num_update_steps, num_active_nodes]` where each row defines the
        node indices to be updated per subiteration. In this case each BP
        iteration runs ``num_update_steps`` subiterations, thus the decoder's
        level of parallelization is lower and usually the decoding throughput
        decreases.
    :param hard_out: If `True`, the decoder provides hard-decided codeword
        bits instead of soft-values.
    :param return_infobits: If `True`, only the `k` info bits (soft or
        hard-decided) are returned. Otherwise all `n` positions are returned.
    :param prune_pcm: If `True`, all punctured degree-1 VNs and connected
        check nodes are removed from the decoding graph (see :cite:p:`Cammerer` for
        details). Besides numerical differences, this should yield the same
        decoding result but improves the decoding throughput and reduces the
        memory footprint.  Automatically set to `False` when
        ``harq_mode=True``.
    :param num_iter: Defining the number of decoder iterations (due to
        batching, no early stopping used at the moment!).
    :param llr_max: Internal clipping value for all internal messages. If
        `None`, no clipping is applied. Filler bits from 5G shortening
        still use a finite LLR of ``1e4``.
    :param v2c_callbacks: Each callable will be executed after each VN update
        with the following arguments ``msg_vn``, ``it``, ``x_hat``, where
        ``msg_vn`` are the v2c messages as tensor of shape
        `[batch_size, num_vns, max_degree]`, ``x_hat`` is the current
        estimate of each VN of shape `[batch_size, num_vns]`, and ``it`` is
        the current iteration counter.
        It must return an updated version of ``msg_vn`` of same shape.
    :param c2v_callbacks: Each callable will be executed after each CN update
        with the following arguments ``msg_cn`` and ``it`` where ``msg_cn``
        are the c2v messages as tensor of shape
        `[batch_size, num_cns, max_degree]` and ``it`` is the current
        iteration counter.
        It must return an updated version of ``msg_cn`` of same shape.
    :param return_state: If `True`, the internal VN messages ``msg_vn`` from
        the last decoding iteration are returned, and ``msg_vn`` or `None`
        needs to be given as a second input when calling the decoder.
        This can be used for iterative demapping and decoding.
    :param harq_mode: If `True`, the decoder operates in HARQ-IR mode.
        PCM pruning is automatically disabled, and ``rv`` can be passed
        to :meth:`call` to combine multiple redundancy versions.
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`, :py:attr:`~sionna.phy.config.precision` is used.
    :param device: Device for computation (e.g., 'cpu', 'cuda:0').

    :input llr_ch: [..., n] or [..., num_rvs, n], `torch.float`.
        Tensor containing the channel logits/llr values.
        When ``rv`` is given, the second-to-last dimension must equal
        ``len(rv)``.

    :input rv: `None` | list of int.
        Redundancy versions corresponding to the stacked LLRs.
        If `None`, standard single-transmission decoding is used.

    :input msg_v2c: `None` | [batch_size, num_edges], `torch.float`.
        Tensor of VN messages representing the internal decoder state.
        Required only if the decoder shall use its previous internal state,
        e.g., for iterative detection and decoding (IDD) schemes.

    :output x_hat: [..., n] or [..., k], `torch.float`.
        Tensor of same shape as ``llr_ch`` containing
        bit-wise soft-estimates (or hard-decided bit-values) of all
        `n` codeword bits or only the `k` information bits if
        ``return_infobits`` is `True`.

    :output msg_v2c: [batch_size, num_edges], `torch.float`.
        Tensor of VN messages representing the internal decoder state.
        Returned only if ``return_state`` is set to `True`.
        Remark: always returns entire decoder state, even if
        ``return_infobits`` is `True`.

    .. rubric:: Notes

    As decoding input logits :math:`\operatorname{log} \frac{p(x=1)}{p(x=0)}`
    are assumed for compatibility with the learning framework, but internally
    LLRs with definition :math:`\operatorname{log} \frac{p(x=0)}{p(x=1)}` are
    used.

    The decoder is not (particularly) optimized for Quasi-cyclic (QC) LDPC
    codes and, thus, supports arbitrary parity-check matrices.

    The batch-dimension is shifted to the last dimension during decoding to
    avoid a performance degradation caused by a severe indexing overhead.

    .. rubric:: Examples


    .. code-block:: python

        import torch
        from sionna.phy.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder

        # Create encoder and decoder
        encoder = LDPC5GEncoder(k=100, n=200)
        decoder = LDPC5GDecoder(encoder, num_iter=20)

        # Encode and decode
        u = torch.randint(0, 2, (10, 100), dtype=torch.float32)
        c = encoder(u)
        llr_ch = 2.0 * (2.0 * c - 1.0)  # Perfect LLRs
        u_hat = decoder(llr_ch)
        print(torch.equal(u, u_hat))
        # True

        # HARQ example: combine two redundancy versions
        dec_harq = LDPC5GDecoder(encoder, num_iter=20, harq_mode=True)
        c_rvs = encoder(u, rv=[0, 2])              # [10, 2, 200]
        llr_rvs = 2.0 * (2.0 * c_rvs - 1.0)
        u_hat = dec_harq(llr_rvs, rv=[0, 2])       # [10, 100]
    """

    # Known-zero filler LLR when ``llr_max`` is None (no message clipping).
    _FILLER_LLR = 1e4

    def __init__(
        self,
        encoder: LDPC5GEncoder,
        cn_update: Union[str, Callable] = "boxplus-phi",
        vn_update: Union[str, Callable] = "sum",
        cn_schedule: Union[str, np.ndarray, torch.Tensor] = "flooding",
        hard_out: bool = True,
        return_infobits: bool = True,
        num_iter: int = 20,
        llr_max: Optional[float] = 20.0,
        v2c_callbacks: Optional[List[Callable]] = None,
        c2v_callbacks: Optional[List[Callable]] = None,
        prune_pcm: bool = True,
        return_state: bool = False,
        harq_mode: bool = False,
        precision: Optional[str] = None,
        device: Optional[str] = None,
        **kwargs,
    ):
        # Needs the 5G Encoder to access all 5G parameters
        if not isinstance(encoder, LDPC5GEncoder):
            raise TypeError("encoder must be of class LDPC5GEncoder.")

        # Store encoder reference (will be assigned after super().__init__())
        _encoder_ref = encoder
        pcm = encoder.pcm

        if not isinstance(return_infobits, bool):
            raise TypeError("return_infobits must be bool.")
        self._return_infobits = return_infobits

        if not isinstance(return_state, bool):
            raise TypeError("return_state must be bool.")

        if not isinstance(harq_mode, bool):
            raise TypeError("harq_mode must be bool.")
        self._harq_mode = harq_mode

        # Deprecation warning for cn_type
        if "cn_type" in kwargs:
            raise TypeError("'cn_type' is deprecated; use 'cn_update' instead.")

        # Prune punctured degree-1 VNs and connected CNs
        if not isinstance(prune_pcm, bool):
            raise TypeError("prune_pcm must be bool.")

        # Pruning must be disabled in HARQ mode
        if harq_mode:
            prune_pcm = False
        self._prune_pcm = prune_pcm

        if prune_pcm:
            # Find index of first position with only degree-1 VN
            dv = np.sum(pcm, axis=0)  # VN degree
            last_pos = encoder.n_ldpc
            for idx in range(encoder.n_ldpc - 1, 0, -1):
                if dv[0, idx] == 1:
                    last_pos = idx
                else:
                    break

            # Number of filler bits
            k_filler = encoder.k_ldpc - encoder.k

            # Number of punctured bits
            nb_punc_bits = (encoder.n_ldpc - k_filler) - encoder.n - 2 * encoder.z

            # If layered decoding is used, quantize number of punctured bits
            # to a multiple of z
            if cn_schedule == "layered":
                nb_punc_bits = int(np.floor(nb_punc_bits / encoder.z) * encoder.z)

            # Effective codeword length after pruning of vn-1 nodes
            self._n_pruned = int(np.maximum(last_pos, encoder._n_ldpc - nb_punc_bits))
            self._nb_pruned_nodes = encoder._n_ldpc - self._n_pruned

            if self._nb_pruned_nodes < 0:
                raise ArithmeticError(
                    "Internal error: number of pruned nodes must be positive."
                )
            # Remove last CNs and VNs from pcm
            if self._nb_pruned_nodes > 0:
                pcm = pcm[: -self._nb_pruned_nodes, : -self._nb_pruned_nodes]
        else:
            self._nb_pruned_nodes = 0
            self._n_pruned = encoder._n_ldpc

        # Handle layered scheduling
        if cn_schedule == "layered":
            z = encoder.z
            num_blocks = int(pcm.shape[0] / z)
            cn_schedule_list = []
            for i in range(num_blocks):
                cn_schedule_list.append(np.arange(z) + i * z)
            cn_schedule = np.stack(cn_schedule_list, axis=0)

        super().__init__(
            pcm,
            cn_update=cn_update,
            vn_update=vn_update,
            cn_schedule=cn_schedule,
            hard_out=hard_out,
            num_iter=num_iter,
            llr_max=llr_max,
            v2c_callbacks=v2c_callbacks,
            c2v_callbacks=c2v_callbacks,
            return_state=return_state,
            precision=precision,
            device=device,
            **kwargs,
        )

        # Assign encoder after super().__init__() for nn.Module compatibility
        self._encoder = _encoder_ref

    ###############################
    # Public methods and properties
    ###############################

    @property
    def encoder(self) -> LDPC5GEncoder:
        """LDPC Encoder used for rate-matching/recovery."""
        return self._encoder

    ########################
    # Sionna block functions
    ########################

    def build(self, input_shape: tuple, **kwargs) -> None:
        """Build block and check input dimensions."""
        if input_shape[-1] != self.encoder.n:
            raise ValueError("Last dimension must be of length n.")

        self._old_shape_5g = input_shape

    def call(
        self,
        llr_ch: torch.Tensor,
        /,
        *,
        rv: Optional[List[int]] = None,
        num_iter: Optional[int] = None,
        msg_v2c: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Iterative BP decoding function and rate-recovery.

        :param llr_ch: Channel LLRs of shape ``[..., n]`` (standard) or
            ``[..., num_rvs, n]`` (HARQ).
        :param rv: List of RV indices matching the stacked LLRs.
            If `None`, standard single-RV decoding is used.  In HARQ
            mode (``harq_mode=True``), defaults to ``[0]``.
        :param num_iter: Number of iterations. If `None`, uses default.
        :param msg_v2c: Initial v2c messages for IDD schemes.

        :output x_hat: Decoded bits of shape ``[..., n]`` or ``[..., k]``.

        :output msg_v2c: Decoder state. Returned only if ``return_state``
            is `True`.
        """
        llr_ch_shape = list(llr_ch.shape)
        input_device = llr_ch.device
        n = self.encoder.n
        k = self.encoder.k
        z = self.encoder.z
        k_filler = self.encoder.k_filler
        nb_pruned = self._nb_pruned_nodes
        buf_len = self.encoder.n_cb_comp - nb_pruned

        # --- Build compressed RM buffer -----------------------------------
        if self._harq_mode:
            if rv is None:
                rv = [0]
            rv = list(rv)
            num_rvs = len(rv)

            if num_rvs > 1:
                if llr_ch_shape[-2] != num_rvs:
                    raise ValueError(
                        f"Second-to-last dimension of llr_ch "
                        f"({llr_ch_shape[-2]}) must equal len(rv) "
                        f"({num_rvs})."
                    )
                llr_ch_flat = llr_ch.reshape(-1, num_rvs, n)
                leading_shape = llr_ch_shape[:-2]
            else:
                has_rv_dim = (len(llr_ch_shape) >= 3
                              and llr_ch_shape[-2] == 1)
                leading_shape = (llr_ch_shape[:-2] if has_rv_dim
                                 else llr_ch_shape[:-1])
                llr_ch_flat = llr_ch.reshape(-1, n).unsqueeze(1)
            batch_size = llr_ch_flat.shape[0]

            # Accumulate multiple RVs via pad + roll
            starts = self.encoder.get_start_positions_comp(rv)
            pad_len = buf_len - n
            llr_buf = torch.zeros(
                batch_size, buf_len, dtype=self.dtype, device=input_device,
            )
            for rv_idx in range(num_rvs):
                llr_rv = llr_ch_flat[:, rv_idx, :]
                if self._encoder.num_bits_per_symbol is not None:
                    llr_rv = llr_rv[:, self._encoder.out_int_inv]
                llr_buf = llr_buf + torch.nn.functional.pad(
                    llr_rv.to(self.dtype), (0, pad_len)
                ).roll(starts[rv_idx], dims=1)
        else:
            leading_shape = llr_ch_shape[:-1]
            llr_ch_reshaped = llr_ch.reshape(-1, n)
            batch_size = llr_ch_reshaped.shape[0]

            if self._encoder.num_bits_per_symbol is not None:
                llr_ch_reshaped = llr_ch_reshaped[
                    :, self._encoder.out_int_inv
                ]
            llr_buf = torch.nn.functional.pad(
                llr_ch_reshaped, (0, buf_len - n)
            )

        # --- Reassemble full n_ldpc vector ---------------------------------
        n_sys_rm = k - 2 * z
        filler_llr = (
            self._FILLER_LLR if self._llr_max is None else self._llr_max
        )
        llr_5g = torch.cat(
            [
                torch.zeros(batch_size, 2 * z,
                            dtype=self.dtype, device=input_device),
                llr_buf[:, :n_sys_rm],
                -filler_llr * torch.ones(batch_size, k_filler,
                                            dtype=self.dtype,
                                            device=input_device),
                llr_buf[:, n_sys_rm:],
            ],
            dim=1,
        )

        # --- BP decoding --------------------------------------------------
        output = super().call(llr_5g, num_iter=num_iter, msg_v2c=msg_v2c)

        if self._return_state:
            x_hat, msg_v2c_out = output
        else:
            x_hat = output

        # --- Output formatting --------------------------------------------
        if self._return_infobits:
            u_hat = x_hat[:, :k]
            output_shape = leading_shape + [k]
            output_shape[0] = -1
            u_reshaped = u_hat.reshape(output_shape)
            if self._return_state:
                return u_reshaped, msg_v2c_out
            return u_reshaped

        x = x_hat.reshape(batch_size, self._n_pruned)
        x_no_filler1 = x[:, :k]
        x_no_filler2 = x[:, self.encoder.k_ldpc:self._n_pruned]
        x_no_filler = torch.cat([x_no_filler1, x_no_filler2], dim=1)
        x_short = x_no_filler[:, 2 * z:2 * z + n]

        if self._encoder.num_bits_per_symbol is not None:
            x_short = x_short[:, self._encoder.out_int]

        output_shape = leading_shape + [n]
        output_shape[0] = -1
        x_short = x_short.reshape(output_shape)
        if self._return_state:
            return x_short, msg_v2c_out
        return x_short
