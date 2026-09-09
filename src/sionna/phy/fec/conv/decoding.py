#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Convolutional code Viterbi and BCJR decoding."""

from typing import Optional, Tuple, Union
import warnings

import numpy as np
import torch

from sionna.phy import Block
from sionna.phy.fec.utils import int2bin, int_mod_2
from sionna.phy.fec.conv.utils import resolve_gen_poly, Trellis


__all__ = ["ViterbiDecoder", "BCJRDecoder"]


class ViterbiDecoder(Block):
    r"""Applies Viterbi decoding to a sequence of noisy codeword bits.

    Implements the Viterbi decoding algorithm :cite:p:`Viterbi` that returns an
    estimate of the information bits for a noisy convolutional codeword.
    Takes as input either LLR values (``method`` = ``'soft_llr'``) or hard
    bit values (``method`` = ``'hard'``) and returns a hard decided estimation
    of the information bits.

    :param encoder: If ``encoder`` is provided as input, the following input
        parameters are not required and will be ignored: ``gen_poly``,
        ``rate``, ``constraint_length``, ``rsc``, ``terminate``. They will be
        inferred from the ``encoder`` object itself. If `None`, the above
        parameters must be provided explicitly.
    :param gen_poly: Tuple of strings with each string being a 0,1 sequence.
        If `None`, ``rate`` and ``constraint_length`` must be provided.
    :param rate: Valid values are 1/3 and 0.5. Only required if ``gen_poly``
        is `None`.
    :param constraint_length: Valid values are between 3 and 8 inclusive.
        Only required if ``gen_poly`` is `None`.
    :param rsc: Boolean flag indicating whether the encoder is
        recursive-systematic for given generator polynomials.
        `True` indicates encoder is recursive-systematic.
        `False` indicates encoder is feed-forward non-systematic.
        Defaults to `False`.
    :param terminate: Boolean flag indicating whether the codeword is
        terminated.
        `True` indicates codeword is terminated to all-zero state.
        `False` indicates codeword is not terminated.
        Defaults to `False`.
    :param method: Valid values are ``'soft_llr'`` or ``'hard'``. In computing
        path metrics, ``'soft_llr'`` expects channel LLRs as input.
        ``'hard'`` assumes a binary symmetric channel (BSC) with 0/1 values
        as inputs. In case of ``'hard'``, inputs will be quantized to 0/1
        values.
    :param return_info_bits: Boolean flag indicating whether only the
        information bits or all codeword bits are returned. Defaults to `True`.
    :param precision: Precision used for internal calculations and outputs.
        If `None`, :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for computation (e.g., 'cpu', 'cuda:0').
        If `None`, :attr:`~sionna.phy.config.Config.device` is used.

    :input inputs: [..., n], `torch.float`.
        Tensor containing the (noisy) channel output symbols where ``n``
        denotes the codeword length.

    :output output: [..., rate \* n], `torch.float`.
        Binary tensor containing the estimates of the information bit tensor.

    .. rubric:: Notes

    A full implementation of the decoder rather than a windowed approach
    is used. For a given codeword of duration ``T``, the path metric is
    computed from time ``0`` to ``T`` and the path with optimal metric at
    time ``T`` is selected. The optimal path is then traced back from ``T``
    to ``0`` to output the estimate of the information bit vector used to
    encode. For larger codewords, note that the current method is sub-optimal
    in terms of memory utilization and latency.
    This method is also excluded from ``torch.compile`` using
    ``@torch.compiler.disable`` because the Viterbi algorithm's inherently
    sequential structure (forward pass, traceback, output extraction) causes
    extremely long compilation times due to loop unrolling.

    .. rubric:: Examples


    .. code-block:: python

        from sionna.phy.fec.conv import ViterbiDecoder

        decoder = ViterbiDecoder(rate=0.5, constraint_length=5)
        llr = torch.randn(10, 200)  # Received LLRs
        u_hat = decoder(llr)
        print(u_hat.shape)
        # torch.Size([10, 100])
    """

    def __init__(
        self,
        *,
        encoder: Optional["ConvEncoder"] = None,
        gen_poly: Optional[Tuple[str, ...]] = None,
        rate: float = 1/2,
        constraint_length: int = 3,
        rsc: bool = False,
        terminate: bool = False,
        method: str = 'soft_llr',
        return_info_bits: bool = True,
        precision: Optional[str] = None,
        device: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(precision=precision, device=device, **kwargs)

        if encoder is not None:
            self._gen_poly = encoder.gen_poly
            self._trellis = encoder.trellis
            self._terminate = encoder.terminate
            if self._trellis.device != self.device:
                self._trellis.to(self.device)
        else:
            self._gen_poly = resolve_gen_poly(gen_poly, rate,
                                              constraint_length)
            self._trellis = Trellis(self.gen_poly, rsc=rsc, device=self.device)
            self._terminate = terminate

        self._coderate_desired = 1 / len(self.gen_poly)
        self._mu = self._trellis.mu

        if method not in ('soft_llr', 'hard'):
            raise ValueError("method must be 'soft_llr' or 'hard'.")

        # conv_k denotes number of input bit streams
        # Can only be 1 in current implementation
        self._conv_k = self._trellis.conv_k

        # conv_n denotes number of output bits for conv_k input bits
        self._conv_n = self._trellis.conv_n

        # For conv codes, the code dimensions are unknown during initialization
        self._k = None
        self._n = None
        self._num_syms = None

        self._ni = 2**self._conv_k
        self._no = 2**self._conv_n
        self._ns = self._trellis.ns

        self._method = method
        self._return_info_bits = return_info_bits

        # If i->j state transition emits symbol k, gather with ipst_op_idx
        # gathers (i,k) element from input in row j.
        self._ipst_op_idx = None

        # Pre-computed output bit patterns for branch metric calculation
        # Register buffer placeholder for CUDAGraph compatibility
        self.register_buffer("_op_bits", None)

    @property
    def gen_poly(self) -> Tuple[str, ...]:
        """Generator polynomial used by the encoder."""
        return self._gen_poly

    @property
    def coderate(self) -> float:
        """Rate of the code used in the encoder."""
        if self.terminate and self._n is None:
            warnings.warn(
                "Due to termination, the true coderate is lower "
                "than the returned design rate. "
                "The exact true rate is dependent on the value of n and "
                "hence cannot be computed before the first call().",
                RuntimeWarning,
                stacklevel=2,
            )
            self._coderate = self._coderate_desired
        elif self.terminate and self._n is not None:
            k = self._coderate_desired * self._n - self._mu
            self._coderate = k / self._n
        else:
            self._coderate = self._coderate_desired
        return self._coderate

    @property
    def trellis(self) -> Trellis:
        """Trellis object used during encoding."""
        return self._trellis

    @property
    def terminate(self) -> bool:
        """Indicates if the encoder is terminated during codeword generation."""
        return self._terminate

    @property
    def k(self) -> Optional[int]:
        """Number of information bits per codeword."""
        if self._k is None:
            warnings.warn(
                "The value of k cannot be computed before the first call().",
                RuntimeWarning,
                stacklevel=2,
            )
        return self._k

    @property
    def n(self) -> Optional[int]:
        """Number of codeword bits."""
        if self._n is None:
            warnings.warn(
                "The value of n cannot be computed before the first call().",
                RuntimeWarning,
                stacklevel=2,
            )
        return self._n

    def _mask_by_tonode(self) -> torch.Tensor:
        """Creates index matrix for gathering by to-node.

        Returns Ns x Ni x 2 index matrix. When applied as gather index on a
        Ns x num_ops matrix ((i,j) denoting metric for prev_st=i and output=j)
        the output is matrix sorted by next_state. Row i in output
        denotes the 2 possible metrics for transition to state i.
        """
        cnst = self._ns * self._ni
        from_nodes_vec = self._trellis.from_nodes.reshape(cnst)
        op_idx = self._trellis.op_by_tonode.reshape(cnst)
        st_op_idx = torch.stack([from_nodes_vec, op_idx], dim=-1)
        st_op_idx = st_op_idx.reshape(self._ns, self._ni, 2)
        return st_op_idx

    def _bmcalc(self, y: torch.Tensor) -> torch.Tensor:
        """Calculate branch metrics for a given noisy codeword tensor.

        For each time period t, computes the distance of symbol vector y[t]
        from each possible output symbol. The distance metric is L2 distance
        if decoder parameter method is 'soft'. The distance metric is L1
        distance if parameter method is 'hard'.
        """
        batch_size = y.shape[0]
        # Reshape y to [bs, num_syms, conv_n]
        y_reshaped = y.reshape(batch_size, -1, self._conv_n)
        num_syms = y_reshaped.shape[1]

        # op_bits: [no, conv_n] - pre-computed in build()
        # Expand for broadcasting: [1, 1, no, conv_n]
        op_bits_exp = self._op_bits.unsqueeze(0).unsqueeze(0)

        # y_reshaped: [bs, num_syms, 1, conv_n]
        y_exp = y_reshaped.unsqueeze(2)

        if self._method == 'soft_llr':
            op_mat_sign = 1 - 2. * op_bits_exp  # [1, 1, no, conv_n]
            llr_sign = -1. * y_exp * op_mat_sign  # [bs, num_syms, no, conv_n]
            # Sum of LLR*(sign of bit) for each symbol
            bm = llr_sign.sum(dim=-1)  # [bs, num_syms, no]
            bm = bm.permute(0, 2, 1)  # [bs, no, num_syms]
        else:  # method == 'hard'
            diffabs = torch.abs(y_exp - op_bits_exp)  # [bs, num_syms, no, conv_n]
            # Manhattan distance of symbols
            bm = diffabs.sum(dim=-1)  # [bs, num_syms, no]
            bm = bm.permute(0, 2, 1)  # [bs, no, num_syms]

        return bm

    def _update_fwd(
        self,
        init_cm: torch.Tensor,
        bm_mat: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass computing cumulative metrics and traceback states."""
        batch_size = init_cm.shape[0]

        cm_list = []
        tb_list = []

        # Pre-compute gather indices for vectorized gather-by-tonode
        # _ipst_op_idx[to_st, inp_idx, :] = (from_st, op_sym)
        from_st_idx = self._ipst_op_idx[:, :, 0]  # [ns, ni]
        op_sym_idx = self._ipst_op_idx[:, :, 1]  # [ns, ni]

        prev_cm = init_cm
        for sym in range(self._num_syms):
            metrics_t = bm_mat[..., sym]  # [bs, no]

            # Ns x No matrix - (s,j) is path_metric at state s with
            # transition op=j
            sum_metric = prev_cm.unsqueeze(2) + metrics_t.unsqueeze(1)
            # [bs, ns, no]

            # Vectorized gather by to-node using advanced indexing
            # sum_metric_bytonode[b, to_st, inp_idx] = sum_metric[b, from_st, op_sym]
            sum_metric_bytonode = sum_metric[:, from_st_idx, op_sym_idx]

            # Get minimum metric and corresponding predecessor index
            tb_state_idx = sum_metric_bytonode.argmin(dim=2)  # [bs, ns]

            # Vectorized: get the actual from-states for traceback
            # from_nodes[to_st, :] gives possible predecessors for each to_st
            # tb_state_idx[:, to_st] selects which predecessor (0 or 1)
            # Result: tb_states[b, to_st] = from_nodes[to_st, tb_state_idx[b, to_st]]
            from_nodes_exp = self._trellis.from_nodes.unsqueeze(0).expand(
                batch_size, -1, -1)  # [bs, ns, ni]
            tb_states = from_nodes_exp.gather(
                2, tb_state_idx.unsqueeze(2).long()
            ).squeeze(2).to(torch.int32)

            cum_t = sum_metric_bytonode.min(dim=2).values

            cm_list.append(cum_t)
            tb_list.append(tb_states)

            prev_cm = cum_t

        cm = torch.stack(cm_list, dim=-1)  # [bs, ns, num_syms]
        tb = torch.stack(tb_list, dim=-1)  # [bs, ns, num_syms]
        return cm, tb

    def _optimal_path(
        self,
        cm_: torch.Tensor,
        tb_: torch.Tensor,
    ) -> torch.Tensor:
        """Compute optimal path (state at each time t) given cm_ & tb_.

        :param cm_: Cumulative metrics for each state at time t [bs, ns, T]
        :param tb_: Traceback state for each state at time t [bs, ns, T]

        :output opt_path: Optimal path of shape [bs, T]
        """
        batch_size = cm_.shape[0]
        num_syms = tb_.shape[-1]

        optst_list = [None] * num_syms
        if self._terminate:
            opt_term_state = torch.zeros(batch_size, dtype=torch.int32,
                                         device=self.device)
        else:
            opt_term_state = cm_[:, :, -1].argmin(dim=1).to(torch.int32)
        optst_list[num_syms - 1] = opt_term_state

        for sym in range(num_syms - 1, 0, -1):
            opt_st = optst_list[sym]
            # Get the traceback state for each batch element
            opt_st_tminus1 = tb_[:, :, sym].gather(
                1, opt_st.unsqueeze(1).long()
            ).squeeze(1).to(torch.int32)
            optst_list[sym - 1] = opt_st_tminus1

        return torch.stack(optst_list, dim=1)  # [bs, num_syms]

    def _op_bits_path(
        self,
        paths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Given a path, compute the input bit stream that results in the path.

        Used in call() where the input is optimal path (seq of states) such
        as the path returned by _optimal_path.
        """
        paths = paths.to(torch.int32)
        batch_size = paths.shape[0]
        num_transitions = paths.shape[-1] - 1

        ip_bits_list = []
        dec_syms_list = []
        ni = self._trellis.ni
        ip_sym_mask = torch.arange(ni, device=self.device).unsqueeze(0)

        for sym in range(1, paths.shape[-1]):
            prev_st = paths[:, sym - 1]
            curr_st = paths[:, sym]

            # Get output symbol for transition
            dec_ = self._trellis.op_mat[prev_st, curr_st]
            dec_syms_list.append(dec_)

            # Find which input bit caused the transition
            # to_nodes[prev_st] gives the 2 possible next states
            to_states = self._trellis.to_nodes[prev_st]  # [bs, ni]
            match_st = (to_states == curr_st.unsqueeze(1))  # [bs, ni]

            # Get the input bit (0 or 1) that matches
            ip_bit = (match_st * ip_sym_mask).sum(dim=-1)
            ip_bits_list.append(ip_bit)

        ip_bit_vec_est = torch.stack(ip_bits_list, dim=1)
        ip_sym_vec_est = torch.stack(dec_syms_list, dim=1)

        return ip_bit_vec_est, ip_sym_vec_est

    def build(self, input_shape: torch.Size):
        """Build block and check dimensions."""
        self._n = input_shape[-1]

        divisible = self._n % self._conv_n
        if divisible != 0:
            raise ValueError('Length of codeword should be divisible by '
                             'number of output bits per symbol.')

        self._num_syms = int(self._n * self._coderate_desired)

        self._num_term_syms = self._mu if self.terminate else 0
        self._k = self._num_syms - self._num_term_syms

        # Build index mask
        self._ipst_op_idx = self._mask_by_tonode()

        # Pre-compute output bit patterns for branch metric calculation
        # Shape: [no, conv_n]
        op_bits = np.stack(
            [int2bin(op, self._conv_n) for op in range(self._no)]
        )
        # Register as buffer for CUDAGraph compatibility
        self.register_buffer("_op_bits", torch.tensor(op_bits, dtype=self.dtype,
                                     device=self.device))

        # Move trellis to correct device if needed
        if self._trellis.device != self.device:
            self._trellis.to(self.device)

    @torch.compiler.disable
    def call(self, inputs: torch.Tensor, /) -> torch.Tensor:
        """Viterbi decoding function.

        :param inputs: Noisy codeword tensor of shape [..., n] where ``n`` is
            the codeword length. All leading dimensions are treated as batch
            dimensions.

        :output output: Decoded information bits of shape [..., k] if
            ``return_info_bits`` is `True`, otherwise [..., n].
        """
        LARGEDIST = 2.**20

        # Allow different codeword lengths in eager mode
        if self._n is None or inputs.shape[-1] != self._n:
            self.build(inputs.shape)

        if self._method == 'hard':
            # Ensure binary values
            inputs = int_mod_2(inputs)
        elif self._method == 'soft_llr':
            inputs = -1. * inputs

        output_shape = list(inputs.shape)
        y_resh = inputs.reshape(-1, self._n)
        if self._return_info_bits:
            output_shape[-1] = self._k
        else:
            output_shape[-1] = self._n

        batch_size = y_resh.shape[0]

        # Branch metrics matrix for a given y
        bm_mat = self._bmcalc(y_resh)

        init_cm = torch.full((self._ns,), LARGEDIST, dtype=self.dtype,
                             device=self.device)
        init_cm[0] = 0.0
        prev_cm = init_cm.unsqueeze(0).expand(batch_size, -1).clone()

        # Forward pass computing cumulative metrics and traceback
        cm, tb = self._update_fwd(prev_cm, bm_mat)

        zero_st = torch.zeros((batch_size, 1), dtype=torch.int32,
                              device=self.device)
        opt_path = self._optimal_path(cm, tb)
        opt_path = torch.cat((zero_st, opt_path), dim=1)

        msghat, cwhat = self._op_bits_path(opt_path)

        if self._return_info_bits:
            msghat = msghat[..., :self._k]
            output = msghat.to(self.dtype)
        else:
            output = cwhat.to(self.dtype)

        output_reshaped = output.reshape(output_shape)
        return output_reshaped


class BCJRDecoder(Block):
    r"""Applies BCJR decoding to a sequence of noisy codeword bits.

    Implements the BCJR decoding algorithm :cite:p:`BCJR` that returns an
    estimate of the information bits for a noisy convolutional codeword.
    Takes as input channel LLRs and optional a priori LLRs.
    Returns an estimate of the information bits, either output LLRs
    (``hard_out`` = `False`) or hard decoded bits (``hard_out`` = `True`),
    respectively.

    :param encoder: If ``encoder`` is provided as input, the following input
        parameters are not required and will be ignored: ``gen_poly``,
        ``rate``, ``constraint_length``, ``rsc``, ``terminate``. They will be
        inferred from the ``encoder`` object itself. If `None`, the above
        parameters must be provided explicitly.
    :param gen_poly: Tuple of strings with each string being a 0,1 sequence.
        If `None`, ``rate`` and ``constraint_length`` must be provided.
    :param rate: Valid values are 1/3 and 1/2. Only required if ``gen_poly``
        is `None`.
    :param constraint_length: Valid values are between 3 and 8 inclusive.
        Only required if ``gen_poly`` is `None`.
    :param rsc: Boolean flag indicating whether the encoder is
        recursive-systematic for given generator polynomials.
        `True` indicates encoder is recursive-systematic.
        `False` indicates encoder is feed-forward non-systematic.
        Defaults to `False`.
    :param terminate: Boolean flag indicating whether the codeword is
        terminated.
        `True` indicates codeword is terminated to all-zero state.
        `False` indicates codeword is not terminated.
        Defaults to `False`.
    :param hard_out: Boolean flag indicating whether to output hard or soft
        decisions on the decoded information vector.
        `True` implies a hard-decoded information vector of 0/1's as output.
        `False` implies output is decoded LLRs of the information.
        Defaults to `True`.
    :param algorithm: Indicates the implemented BCJR algorithm,
        where ``'map'`` denotes the exact MAP algorithm, ``'log'`` indicates
        the exact MAP implementation but in log-domain, and ``'maxlog'``
        indicates the approximated MAP implementation in log-domain where
        :math:`\log(e^{a}+e^{b}) \sim \max(a,b)`.
    :param precision: Precision used for internal calculations and outputs.
        If `None`, :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for computation (e.g., 'cpu', 'cuda:0').
        If `None`, :attr:`~sionna.phy.config.Config.device` is used.

    :input llr_ch: [..., n], `torch.float`.
        Tensor containing the (noisy) channel LLRs, where ``n`` denotes the
        codeword length.

    :input llr_a: [..., k], `None` (default) | `torch.float`.
        Tensor containing the a priori information of each information bit.
        Implicitly assumed to be 0 if only ``llr_ch`` is provided.

    :output msghat: `torch.float`.
        Tensor of shape ``[..., coderate*n]`` containing the estimates of the
        information bit tensor.

    .. rubric:: Examples


    .. code-block:: python

        from sionna.phy.fec.conv import BCJRDecoder

        decoder = BCJRDecoder(rate=0.5, constraint_length=5)
        llr = torch.randn(10, 200)  # Received LLRs
        u_hat = decoder(llr)
        print(u_hat.shape)
        # torch.Size([10, 100])
    """

    def __init__(
        self,
        encoder: Optional["ConvEncoder"] = None,
        gen_poly: Optional[Tuple[str, ...]] = None,
        rate: float = 1/2,
        constraint_length: int = 3,
        rsc: bool = False,
        terminate: bool = False,
        hard_out: bool = True,
        algorithm: str = 'map',
        precision: Optional[str] = None,
        device: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(precision=precision, device=device, **kwargs)

        if encoder is not None:
            self._gen_poly = encoder.gen_poly
            self._trellis = encoder.trellis
            self._terminate = encoder.terminate
            if self._trellis.device != self.device:
                self._trellis.to(self.device)
        else:
            self._gen_poly = resolve_gen_poly(gen_poly, rate,
                                              constraint_length)
            self._trellis = Trellis(self.gen_poly, rsc=rsc, device=self.device)
            self._terminate = terminate

        valid_algorithms = ['map', 'log', 'maxlog']
        if algorithm not in valid_algorithms:
            raise ValueError("algorithm must be one of map, log or maxlog")

        self._coderate_desired = 1 / len(self._gen_poly)
        self._mu = len(self._gen_poly[0]) - 1

        self._num_term_bits = None
        self._num_term_syms = None

        # conv_k denotes number of input bit streams
        # Can only be 1 in current implementation
        self._conv_k = self._trellis.conv_k
        if self._conv_k != 1:
            raise NotImplementedError("Only conv_k=1 currently supported.")

        self._mu = self._trellis.mu
        # conv_n denotes number of output bits for conv_k input bits
        self._conv_n = self._trellis.conv_n

        # Length of Info-bit vector
        self._k = None
        # Length of codeword, including termination bits
        self._n = None
        # Number of encoding periods or state transitions
        self._num_syms = None

        self._ni = 2**self._conv_k
        self._no = 2**self._conv_n
        self._ns = self._trellis.ns

        self._hard_out = hard_out
        self._algorithm = algorithm

        self._ipst_op_idx = None
        self._ipst_ip_idx = None

        # Pre-computed output bit patterns for branch metric calculation
        # Register buffer placeholder for CUDAGraph compatibility
        self.register_buffer("_op_bits", None)

    @property
    def gen_poly(self) -> Tuple[str, ...]:
        """Generator polynomial used by the encoder."""
        return self._gen_poly

    @property
    def coderate(self) -> float:
        """Rate of the code used in the encoder."""
        if self.terminate and self._n is None:
            warnings.warn(
                "Due to termination, the true coderate is lower "
                "than the returned design rate. "
                "The exact true rate is dependent on the value of n and "
                "hence cannot be computed before the first call().",
                RuntimeWarning,
                stacklevel=2,
            )
            self._coderate = self._coderate_desired
        elif self.terminate and self._n is not None:
            k = self._coderate_desired * self._n - self._mu
            self._coderate = k / self._n
        else:
            self._coderate = self._coderate_desired
        return self._coderate

    @property
    def trellis(self) -> Trellis:
        """Trellis object used during encoding."""
        return self._trellis

    @property
    def terminate(self) -> bool:
        """Indicates if the encoder is terminated during codeword generation."""
        return self._terminate

    @property
    def k(self) -> Optional[int]:
        """Number of information bits per codeword."""
        if self._k is None:
            warnings.warn(
                "The value of k cannot be computed before the first call().",
                RuntimeWarning,
                stacklevel=2,
            )
        return self._k

    @property
    def n(self) -> Optional[int]:
        """Number of codeword bits."""
        if self._n is None:
            warnings.warn(
                "The value of n cannot be computed before the first call().",
                RuntimeWarning,
                stacklevel=2,
            )
        return self._n

    def _mask_by_tonode(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Creates index matrices for gathering by to-node.

        Assume i->j a valid state transition given info-bit b & emits symbol k.
        Returns following two _ns x _ni x 2 matrices:
        - st_op_idx: jth row contains (i,k) tuples
        - st_ip_idx: jth row contains (i,b) tuples

        When applied as gather on a _ns x _no matrix, the output is
        matrix sorted by next_state.
        """
        cnst = self._ns * self._ni
        from_nodes_vec = self._trellis.from_nodes.reshape(cnst)
        op_idx = self._trellis.op_by_tonode.reshape(cnst)
        st_op_idx = torch.stack([from_nodes_vec, op_idx], dim=-1)
        st_op_idx = st_op_idx.reshape(self._ns, self._ni, 2)

        ip_idx = self._trellis.ip_by_tonode.reshape(cnst)
        st_ip_idx = torch.stack([from_nodes_vec, ip_idx], dim=-1)
        st_ip_idx = st_ip_idx.reshape(self._ns, self._ni, 2)

        return st_op_idx, st_ip_idx

    def _bmcalc(self, llr_in: torch.Tensor) -> torch.Tensor:
        """Calculate branch gamma metrics for a given noisy codeword tensor.

        For each time period t, computes the 'distance' of symbol
        vector y[t] from each possible output symbol.
        """
        batch_size = llr_in.shape[0]
        # Reshape llr_in to [bs, num_syms, conv_n]
        llr_reshaped = llr_in.reshape(batch_size, -1, self._conv_n)

        # op_bits: [no, conv_n] - pre-computed in build()
        # Expand for broadcasting: [1, 1, no, conv_n]
        op_bits_exp = self._op_bits.unsqueeze(0).unsqueeze(0)
        op_mat_sign = 1. - 2. * op_bits_exp

        # llr_reshaped: [bs, num_syms, 1, conv_n]
        llr_exp = llr_reshaped.unsqueeze(2)

        llr_sign = llr_exp * op_mat_sign  # [bs, num_syms, no, conv_n]
        half_llr_sign = 0.5 * llr_sign

        if self._algorithm in ['log', 'maxlog']:
            bm = half_llr_sign.sum(dim=-1)  # [bs, num_syms, no]
        else:
            bm = torch.exp(half_llr_sign.sum(dim=-1))

        bm = bm.permute(0, 2, 1).contiguous()  # [bs, no, num_syms]
        return bm

    def _initialize(
        self,
        llr_ch: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize alpha and beta tensors."""
        batch_size = llr_ch.shape[0]

        if self._algorithm in ['log', 'maxlog']:
            init_vals = (float('-inf'), 0.0)
        else:
            init_vals = (0.0, 1.0)

        alpha_init = torch.full((self._ns,), init_vals[0], dtype=self.dtype,
                                device=self.device)
        alpha_init[0] = init_vals[1]

        if not self._terminate:
            eq_prob = 1. / self._ns
            if self._algorithm in ['log', 'maxlog']:
                eq_prob = np.log(eq_prob)
            beta_init = torch.full((self._ns,), eq_prob, dtype=self.dtype,
                                   device=self.device)
        else:
            beta_init = alpha_init.clone()

        alpha_init = alpha_init.unsqueeze(0).expand(batch_size, -1).clone()
        beta_init = beta_init.unsqueeze(0).expand(batch_size, -1).clone()

        return alpha_init, beta_init

    def _update_fwd(
        self,
        alph_init: torch.Tensor,
        bm_mat: torch.Tensor,
        llr: torch.Tensor,
    ) -> torch.Tensor:
        """Run forward update from time t=0 to t=k-1.

        At each time t, computes alpha_t using alpha_t-1 and gamma_t.
        Returns tensor of alpha values [bs, ns, num_syms+1].
        """
        batch_size = alph_init.shape[0]
        alph_list = [alph_init]
        alph_prev = alph_init

        # op_by_fromnode[from_st, input] = output symbol
        op_mask = self._trellis.op_by_fromnode  # [ns, ni]

        ipbit_mat = torch.arange(self._ni, device=self.device).unsqueeze(0) \
            .unsqueeze(0).expand(batch_size, self._ns, -1).contiguous()  # [bs, ns, ni]
        ipbitsign_mat = 1. - 2. * ipbit_mat.to(self.dtype)

        # Pre-compute gather indices for vectorized gather-by-tonode
        # _ipst_ip_idx[to_st, inp_idx, :] = (from_st, inp_bit)
        # We need to gather alph_gam_prod[:, from_st, inp_bit] for all (to_st, inp_idx)
        from_st_idx = self._ipst_ip_idx[:, :, 0].contiguous()  # [ns, ni]
        inp_bit_idx = self._ipst_ip_idx[:, :, 1].contiguous()  # [ns, ni]

        for t in range(self._num_syms):
            bm_t = bm_mat[..., t].contiguous()  # [bs, no]
            llr_t = 0.5 * llr[..., t].unsqueeze(1).unsqueeze(2)  # [bs, 1, 1]

            # bm_byfromst[bs, from_st, input] = bm_t[bs, op_mask[from_st, input]]
            bm_byfromst = bm_t[:, op_mask].contiguous()  # [bs, ns, ni]

            signed_half_llr = llr_t * ipbitsign_mat  # [bs, ns, ni]

            if self._algorithm in ['log', 'maxlog']:
                llr_byfromst = signed_half_llr
                gamma_byfromst = llr_byfromst + bm_byfromst
                alph_gam_prod = (gamma_byfromst + alph_prev.unsqueeze(2)).contiguous()
            else:
                llr_byfromst = torch.exp(signed_half_llr)
                gamma_byfromst = llr_byfromst * bm_byfromst
                alph_gam_prod = (gamma_byfromst * alph_prev.unsqueeze(2)).contiguous()

            # Vectorized gather by to-node using advanced indexing
            # alph_gam_prod: [bs, ns, ni] indexed by [from_st_idx, inp_bit_idx]
            # Result: alphgam_bytost[b, to_st, inp_idx] = alph_gam_prod[b, from_st_idx[to_st, inp_idx], inp_bit_idx[to_st, inp_idx]]
            alphgam_bytost = alph_gam_prod[:, from_st_idx, inp_bit_idx].contiguous()

            if self._algorithm == 'map':
                alph_t = alphgam_bytost.sum(dim=-1)
                alph_t_sum = alph_t.sum(dim=-1, keepdim=True)
                alph_t = alph_t / alph_t_sum
            elif self._algorithm == 'log':
                alph_t = torch.logsumexp(alphgam_bytost, dim=-1)
            else:  # maxlog
                alph_t = alphgam_bytost.max(dim=-1).values

            alph_prev = alph_t
            alph_list.append(alph_t)

        return torch.stack(alph_list, dim=-1)  # [bs, ns, num_syms+1]

    def _update_bwd(
        self,
        beta_init: torch.Tensor,
        bm_mat: torch.Tensor,
        llr: torch.Tensor,
        alpha_ta: torch.Tensor,
    ) -> torch.Tensor:
        """Run backward update from time t=k-1 to t=0.

        At each time t, computes beta_t-1 using beta_t and gamma_t.
        Returns LLRs for information bits for t=0,1,...,k-1.
        """
        batch_size = beta_init.shape[0]
        beta_next = beta_init

        llr_op_list = [None] * self._num_syms

        # op_mask[from_st, input] = output symbol
        op_mask = self._trellis.op_by_fromnode  # [ns, ni]
        tonode_mask = self._trellis.to_nodes  # [ns, ni]

        ipbit_mat = torch.arange(self._ni, device=self.device).unsqueeze(0) \
            .unsqueeze(0).expand(batch_size, self._ns, -1).contiguous()  # [bs, ns, ni]
        ipbitsign_mat = 1. - 2. * ipbit_mat.to(self.dtype)

        for t in range(self._num_syms - 1, -1, -1):
            bm_t = bm_mat[..., t].contiguous()  # [bs, no]
            llr_t = 0.5 * llr[..., t].unsqueeze(1).unsqueeze(2)  # [bs, 1, 1]
            signed_half_llr = llr_t * ipbitsign_mat

            bm_byfromst = bm_t[:, op_mask].contiguous()  # [bs, ns, ni]

            if self._algorithm in ['log', 'maxlog']:
                llr_byfromst = signed_half_llr
                gamma_byfromst = (llr_byfromst + bm_byfromst).contiguous()
            else:
                llr_byfromst = torch.exp(signed_half_llr)
                gamma_byfromst = (llr_byfromst * bm_byfromst).contiguous()

            # beta_bytonode[bs, from_st, input] = beta_next[bs, to_nodes[from_st, input]]
            beta_bytonode = beta_next[:, tonode_mask].contiguous()  # [bs, ns, ni]

            if self._algorithm not in ['log', 'maxlog']:
                beta_gam_prod = gamma_byfromst * beta_bytonode
                beta_t = beta_gam_prod.sum(dim=-1)
                beta_t_sum = beta_t.sum(dim=-1, keepdim=True)
                beta_t = beta_t / beta_t_sum
            elif self._algorithm == 'log':
                beta_gam_prod = gamma_byfromst + beta_bytonode
                beta_t = torch.logsumexp(beta_gam_prod, dim=-1)
            else:  # maxlog
                beta_gam_prod = gamma_byfromst + beta_bytonode
                beta_t = beta_gam_prod.max(dim=-1).values

            alph_t = alpha_ta[..., t].contiguous()  # [bs, ns]

            if self._algorithm not in ['log', 'maxlog']:
                llr_op_t0 = alph_t * gamma_byfromst[..., 0].contiguous() * beta_bytonode[..., 0].contiguous()
                llr_op_t1 = alph_t * gamma_byfromst[..., 1].contiguous() * beta_bytonode[..., 1].contiguous()
                llr_op_t = torch.log(
                    llr_op_t0.sum(dim=-1) / llr_op_t1.sum(dim=-1)
                )
            else:
                llr_op_t0 = alph_t + gamma_byfromst[..., 0].contiguous() + beta_bytonode[..., 0].contiguous()
                llr_op_t1 = alph_t + gamma_byfromst[..., 1].contiguous() + beta_bytonode[..., 1].contiguous()
                if self._algorithm == 'log':
                    llr_op_t = torch.logsumexp(llr_op_t0, dim=-1) - \
                               torch.logsumexp(llr_op_t1, dim=-1)
                else:  # maxlog
                    llr_op_t = llr_op_t0.max(dim=-1).values - \
                               llr_op_t1.max(dim=-1).values

            llr_op_list[t] = llr_op_t
            beta_next = beta_t

        return torch.stack(llr_op_list, dim=-1)  # [bs, num_syms]

    def build(self, llr_ch_shape: torch.Size, **kwargs):
        """Build block and check dimensions."""
        self._n = llr_ch_shape[-1]
        self._num_syms = int(self._n * self._coderate_desired)

        self._num_term_syms = self._mu if self._terminate else 0
        self._num_term_bits = int(self._num_term_syms / self._coderate_desired)

        self._k = self._num_syms - self._num_term_syms

        # Build index masks
        self._ipst_op_idx, self._ipst_ip_idx = self._mask_by_tonode()

        # Pre-compute output bit patterns for branch metric calculation
        # Shape: [no, conv_n]
        op_bits = np.stack(
            [int2bin(op, self._conv_n) for op in range(self._no)]
        )
        # Register as buffer for CUDAGraph compatibility
        self.register_buffer("_op_bits", torch.tensor(op_bits, dtype=self.dtype,
                                     device=self.device))

        # Move trellis to correct device if needed
        if self._trellis.device != self.device:
            self._trellis.to(self.device)

    @torch.compiler.disable
    def call(
        self,
        llr_ch: torch.Tensor,
        /,
        *,
        llr_a: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """BCJR decoding function.

        :param llr_ch: Noisy channel LLR tensor of shape [..., n] where ``n``
            is the codeword length. All leading dimensions are treated as
            batch dimensions.
        :param llr_a: Optional a priori LLR tensor of shape [..., k] where
            ``k`` is the number of information bits. Implicitly assumed to be
            0 if not provided.

        :output msghat: Decoded information bits (or LLRs) of shape
            [..., k].
        """
        output_shape = list(llr_ch.shape)
        input_device = llr_ch.device

        # Allow different codeword lengths in eager mode
        # Also ensure build() is called (needed for torch.compile compatibility)
        if self._n is None or output_shape[-1] != self._n:
            self._built = False
            if torch.compiler.is_compiling():
                # During compilation trace, we need concrete values
                torch.compiler.disable(self.build)(llr_ch.shape)
            else:
                self.build(llr_ch.shape)
            self._built = True

        # Move module to input device if needed (for torch.compile compatibility)
        if self._op_bits is not None and self._op_bits.device != input_device:
            self.to(input_device)

        output_shape[-1] = self._k
        llr_ch = llr_ch.reshape(-1, self._n)
        batch_size = llr_ch.shape[0]

        if llr_a is None:
            llr_a = torch.zeros(
                batch_size, self._num_syms,
                dtype=self.dtype, device=input_device
            )
        else:
            llr_a = llr_a.reshape(-1, self._num_syms)

        # Internally, we use more common LLR definition log(p(x=0)/p(x=1))
        llr_ch = -1. * llr_ch
        llr_a = -1. * llr_a

        # Branch metrics matrix for a given y
        bm_mat = self._bmcalc(llr_ch)
        alpha_init, beta_init = self._initialize(llr_ch)

        alph_ta = self._update_fwd(alpha_init, bm_mat, llr_a)
        llr_op = self._update_bwd(beta_init, bm_mat, llr_a, alph_ta)

        # Revert LLR definition
        msghat = -1. * llr_op[..., :self._k]

        if self._hard_out:
            msghat = (msghat > 0.0).to(self.dtype)

        msghat_reshaped = msghat.reshape(output_shape)
        return msghat_reshaped

