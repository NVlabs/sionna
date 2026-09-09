#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Blocks for Polar decoding such as successive cancellation (SC),
successive cancellation list (SCL) and iterative belief propagation (BP)
decoding."""

from typing import NamedTuple, Optional, Tuple, Union
import enum
import numbers
import warnings
import numpy as np
import torch
from torch.nn.functional import softplus

from sionna.phy import Block
from sionna.phy.fec.crc import CRCDecoder, CRCEncoder
from sionna.phy.fec.polar.encoding import Polar5GEncoder
from sionna.phy.fec.polar.utils import _is_pow2


__all__ = [
    "PolarSCDecoder",
    "PolarSCLDecoder",
    "PolarBPDecoder",
    "Polar5GDecoder",
]

# Workaround for an upstream inductor C++ codegen bug on torch <= 2.11.
# When the inner SCL / BP decode loop is fused with surrounding ops
# (e.g. rate matching in ``Polar5GDecoder``), inductor produces invalid
# C++ — ``decltype(scalar)::blendv`` on int8 for SCL (pytorch#178148),
# or a gcc ICE on the very large fused BP kernel. The SCL codegen bug is
# fixed in torch >= 2.12 (see also pytorch#180212). We force a dynamo
# graph break around the inner SCL loop only on CPU instances of
# affected torch versions; CUDA (Triton) and torch >= 2.12 keep the full
# fused compile. Eager execution is never affected.
#

_NEEDS_CPU_COMPILE_BREAK = tuple(
    int(p) for p in torch.__version__.split("+", 1)[0].split(".")[:2]
) < (2, 12)
_CPU_COMPILE_BREAK_WARNED = [False]


def _install_cpu_compile_break(decoder, attr_name):
    """Wrap a bound decode method with ``torch.compiler.disable`` so
    that, when traced by ``torch.compile``, dynamo graph-breaks here
    and runs the inner loop eagerly. Eager calls go straight through
    to the original method. A one-shot warning is emitted on the
    first compiled invocation so users on old torch learn that
    compile-mode throughput is reduced and that upgrading to
    torch >= 2.12 restores full fused-compile performance."""
    disabled = torch.compiler.disable(recursive=False)(getattr(decoder, attr_name))

    def _wrapper(*args, **kwargs):
        if torch.compiler.is_compiling() and not _CPU_COMPILE_BREAK_WARNED[0]:
            _CPU_COMPILE_BREAK_WARNED[0] = True
            warnings.warn(
                "torch < 2.12 contains an inductor C++ codegen bug that "
                "breaks torch.compile of the SCL / BP inner decode loop "
                "on CPU. A dynamo graph break is being inserted around "
                "the inner loop as a workaround; the rest of the decoder "
                "still compiles, but compile-mode throughput is reduced. "
                "Upgrade to torch >= 2.12 to recover full fused-compile "
                "performance.",
                RuntimeWarning,
                stacklevel=3,
            )
        return disabled(*args, **kwargs)

    setattr(decoder, attr_name, _wrapper)


class _Op(enum.IntEnum):
    """Opcodes for the SCL decoder tape."""
    F = 0            # check-node: push LLRs to left child
    G = 1            # variable-node: push LLRs to right child
    COMBINE = 2      # Polar XOR combine hard decisions upward
    LEAF_FROZEN = 3  # single frozen bit (forced to 0)
    LEAF_INFO = 4    # single info bit: fork paths, prune to L
    R0 = 5           # fast-SCL: whole sub-tree is frozen (rate 0)
    REP = 6          # fast-SCL: repetition code (only last bit info)
    R1 = 7           # fast-SCL: whole sub-tree is info (rate 1), M=1 flip


class _TapeEntry(NamedTuple):
    """A single SCL decoder instruction.

    ``op``, ``stage``, ``off`` and ``length`` describe the Polar-tree
    region the op acts on. ``slot_a``, ``slot_b`` and ``slot_c`` are
    pre-computed workspace offsets whose meaning depends on ``op``;
    construct entries through the ``_make_*`` factories on
    :class:`PolarSCLDecoder` to keep the conventions self-documenting.

    Per-opcode conventions:

    * ``F``           — ``slot_a = llr_src``, ``slot_b = llr_dst``.
    * ``G``           — ``slot_a = llr_src``, ``slot_b = llr_dst``,
      ``slot_c = u_src``.
    * ``COMBINE``     — ``slot_a = u_src``, ``slot_b = u_dst``.
    * ``LEAF_FROZEN`` — ``slot_a = llr_src``.
    * ``LEAF_INFO``   — ``slot_a = llr_src``.
    * ``R0``          — ``slot_a = llr_src``, ``slot_b = u_dst``
      (zero when ``stage == n_stages``).
    * ``REP``         — ``slot_a = llr_src``, ``slot_b = u_dst``.
    * ``R1``          — ``slot_a = llr_src``, ``slot_b = u_dst``
      (zero when ``stage == n_stages``).
    """
    op: int
    stage: int
    off: int
    length: int
    slot_a: int = 0
    slot_b: int = 0
    slot_c: int = 0


class PolarSCDecoder(Block):
    """Successive cancellation (SC) decoder :cite:p:`Arikan_Polar` for Polar codes
    and Polar-like codes.

    :param frozen_pos: Array of `int` defining the ``n-k`` indices of the
        frozen positions.
    :param n: Defining the codeword length.
    :param precision: Precision used for internal calculations and outputs.
        If `None`, :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for computation (e.g., 'cpu', 'cuda:0').
        If `None`, :attr:`~sionna.phy.config.Config.device` is used.

    :input llr_ch: [..., n], `torch.float`.
        Tensor containing the channel LLR values (as logits).

    :output u_hat: [..., k], `torch.float`.
        Tensor containing hard-decided estimations of all ``k``
        information bits.

    .. rubric:: Notes

    This block implements the SC decoder as described in
    :cite:p:`Arikan_Polar`. However, the implementation follows the `recursive
    tree` :cite:p:`Gross_Fast_SCL` terminology and combines nodes for increased
    throughputs without changing the outcome of the algorithm.

    As commonly done, we assume frozen bits are set to `0`. Please note
    that - although its practical relevance is only little - setting frozen
    bits to `1` may result in `affine` codes instead of linear code as the
    `all-zero` codeword is not necessarily part of the code any more.

    .. rubric:: Examples


    .. code-block:: python

        import torch
        from sionna.phy.fec.polar import PolarSCDecoder, PolarEncoder
        from sionna.phy.fec.polar.utils import generate_5g_ranking

        k, n = 100, 256
        frozen_pos, _ = generate_5g_ranking(k, n)
        encoder = PolarEncoder(frozen_pos, n)
        decoder = PolarSCDecoder(frozen_pos, n)

        bits = torch.randint(0, 2, (10, k), dtype=torch.float32)
        codewords = encoder(bits)
        llr_ch = 20.0 * (2.0 * codewords - 1)  # BPSK without noise
        decoded = decoder(llr_ch)
        print(torch.equal(bits, decoded))
        # True
    """

    def __init__(
        self,
        frozen_pos: np.ndarray,
        n: int,
        *,
        precision: Optional[str] = None,
        device: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(precision=precision, device=device, **kwargs)

        if not isinstance(n, numbers.Number):
            raise TypeError("n must be a number.")
        n = int(n)

        if not np.issubdtype(frozen_pos.dtype, int):
            raise TypeError("frozen_pos contains non int.")
        if len(frozen_pos) > n:
            msg = "Num. of elements in frozen_pos cannot be greater than n."
            raise ValueError(msg)
        if not _is_pow2(n):
            raise ValueError("n must be a power of 2.")

        # Store internal attributes
        self._n = n
        self._frozen_pos = frozen_pos
        self._k = self._n - len(self._frozen_pos)
        self._info_pos = np.setdiff1d(np.arange(self._n), self._frozen_pos)
        if self._k != len(self._info_pos):
            msg = "Internal error: invalid info_pos generated."
            raise ArithmeticError(msg)

        # Register info_pos as buffer for torch.compile compatibility
        self.register_buffer(
            "_info_pos_t",
            torch.tensor(self._info_pos, dtype=torch.int64, device=self.device),
        )

        self._llr_max = 30.0  # Internal max LLR value
        # Create a frozen bit vector for simpler encoding
        self._frozen_ind = np.zeros(self._n)
        self._frozen_ind[self._frozen_pos] = 1

        # Register frozen indicator as tensor buffer for torch.compile compatibility
        self.register_buffer(
            "_frozen_ind_t",
            torch.tensor(self._frozen_ind, dtype=self.dtype, device=self.device),
        )

        # Enable graph pruning
        self._use_fast_sc = False

    @property
    def n(self) -> int:
        """Codeword length."""
        return self._n

    @property
    def k(self) -> int:
        """Number of information bits."""
        return self._k

    @property
    def frozen_pos(self) -> np.ndarray:
        """Frozen positions for Polar decoding."""
        return self._frozen_pos

    @property
    def info_pos(self) -> np.ndarray:
        """Information bit positions for Polar encoding."""
        return self._info_pos

    @property
    def llr_max(self) -> float:
        """Maximum LLR value for internal calculations."""
        return self._llr_max

    def _cn_op(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Check-node update (boxplus) for LLR inputs.

        Operations are performed element-wise.

        See :cite:p:`Stimming_LLR` and :cite:p:`Hashemi_SSCL` for detailed equations.
        """
        x_in = torch.clamp(x, min=-self._llr_max, max=self._llr_max)
        y_in = torch.clamp(y, min=-self._llr_max, max=self._llr_max)

        # Avoid division for numerical stability
        llr_out = torch.log(1 + torch.exp(x_in + y_in))
        llr_out = llr_out - torch.log(torch.exp(x_in) + torch.exp(y_in))

        return llr_out

    def _vn_op(
        self, x: torch.Tensor, y: torch.Tensor, u_hat: torch.Tensor
    ) -> torch.Tensor:
        """VN update for LLR inputs."""
        return (1 - 2 * u_hat) * x + y

    def _polar_decode_sc(
        self, llr_ch: torch.Tensor, frozen_ind: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Recursive SC decoding function.

        Recursively branch decoding tree and split into decoding of `upper`
        and `lower` path until reaching a leaf node.

        The function returns the u_hat decisions at stage `0` and the bit
        decisions of the intermediate stage `s` (i.e., the re-encoded
        version of `u_hat` until the current stage `s`).

        This decoder parallelizes over the batch-dimension, i.e., the tree
        is processed for all samples in the batch in parallel. This yields a
        higher throughput, but does not improve the latency.
        """
        # Calculate current codeword length
        n = frozen_ind.shape[0]

        # Branch if leaf is not reached yet
        if n > 1:
            if self._use_fast_sc:
                if frozen_ind.sum() == n:
                    u_hat = torch.zeros_like(llr_ch)
                    return u_hat, u_hat

            llr_ch1 = llr_ch[..., 0 : int(n / 2)]
            llr_ch2 = llr_ch[..., int(n / 2) :]
            frozen_ind1 = frozen_ind[0 : int(n / 2)]
            frozen_ind2 = frozen_ind[int(n / 2) :]

            # Upper path
            x_llr1_in = self._cn_op(llr_ch1, llr_ch2)

            # Call the decoding function (with upper half)
            u_hat1, u_hat1_up = self._polar_decode_sc(x_llr1_in, frozen_ind1)

            # Lower path
            x_llr2_in = self._vn_op(llr_ch1, llr_ch2, u_hat1_up)
            # Call the decoding function again (with lower half)
            u_hat2, u_hat2_up = self._polar_decode_sc(x_llr2_in, frozen_ind2)

            # Combine u_hat from both branches
            u_hat = torch.cat([u_hat1, u_hat2], -1)

            # Calculate re-encoded version of u_hat at current stage
            u_hat1_up_int = u_hat1_up.to(torch.int8)
            u_hat2_up_int = u_hat2_up.to(torch.int8)
            u_hat1_up_int = torch.bitwise_xor(u_hat1_up_int, u_hat2_up_int)
            u_hat1_up = u_hat1_up_int.to(self.dtype)
            u_hat_up = torch.cat([u_hat1_up, u_hat2_up], -1)

        else:  # If leaf is reached perform basic decoding op (=decision)
            # Use tensor operations to avoid CUDA graph breaks
            # frozen_ind is a 1-element tensor at this point
            is_frozen = frozen_ind[0] == 1  # Tensor comparison

            # Compute frozen case: u_hat = 0
            frozen_result = torch.zeros_like(llr_ch)

            # Compute non-frozen case: hard decision
            decision_result = 0.5 * (1.0 - torch.sign(llr_ch))
            # Handle exact 0 LLRs (u_hat = 0.5) by setting to 1
            decision_result = torch.where(
                decision_result == 0.5,
                torch.ones_like(decision_result),
                decision_result,
            )

            # Branchless selection using torch.where
            u_hat = torch.where(is_frozen, frozen_result, decision_result)
            u_hat_up = u_hat
        return u_hat, u_hat_up

    def build(self, input_shape: Tuple[int, ...]) -> None:
        """Check if shape of input is invalid."""
        if input_shape[-1] != self._n:
            raise ValueError("Invalid input shape.")

    def call(self, llr_ch: torch.Tensor) -> torch.Tensor:
        """Successive cancellation (SC) decoding function.

        Performs successive cancellation decoding and returns the estimated
        information bits.

        :param llr_ch: Tensor of shape `[..., n]` containing the
            channel LLR values (as logits).

        :output u_hat: Tensor of shape `[..., k]` containing hard-decided
            estimations of all ``k`` information bits.

        Note: This function recursively unrolls the SC decoding tree, thus,
        for larger values of ``n`` building the decoding graph can become
        time consuming.
        """
        # Reshape inputs to [-1, n]
        input_shape = llr_ch.shape
        new_shape = (-1, self._n)
        llr_ch = llr_ch.reshape(new_shape)

        llr_ch = -1.0 * llr_ch  # Logits are converted into "true" llrs

        # Decode
        u_hat_n, _ = self._polar_decode_sc(llr_ch, self._frozen_ind_t)

        # Recover the k information bit positions using pre-registered buffer
        u_hat = u_hat_n[:, self._info_pos_t]

        # Reconstruct input shape
        output_shape = list(input_shape[:-1]) + [self.k]
        u_hat_reshape = u_hat.reshape(output_shape)
        return u_hat_reshape


class PolarSCLDecoder(Block):
    # pylint: disable=line-too-long
    """Fast successive cancellation list (SCL) decoder for Polar and Polar-like
    codes :cite:p:`Tal_SCL` :cite:p:`Hashemi_SSCL`. Rate-1 nodes use a
    single-flip (``M=1``) shortcut rather than exact list decoding; see Notes.

    :param frozen_pos: Array of `int` defining the ``n-k`` indices of the
        frozen positions.
    :param n: Defining the codeword length.
    :param list_size: Defining the list size ``L`` of the decoder. Must
        be a power of 2.
    :param crc_degree: Defining the CRC polynomial to be used. Can be any
        value from `{CRC24A, CRC24B, CRC24C, CRC16, CRC11, CRC6}`.
    :param ind_iil_inv: If not `None`, the sequence is used as inverse
        input-bit interleaver before the CRC is evaluated. This only
        affects CRC evaluation; the output sequence is not permuted.
    :param return_crc_status: If `True`, the decoder additionally returns
        the CRC status indicating if a codeword was (most likely)
        correctly recovered. This is only available if ``crc_degree``
        is not `None`.
    :param precision: Precision used for internal calculations and outputs.
        If `None`, :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for computation (e.g., 'cpu', 'cuda:0').
        If `None`, :attr:`~sionna.phy.config.Config.device` is used.

    :input llr_ch: [..., n], `torch.float`.
        Tensor containing the channel LLR values (as logits).

    :output b_hat: [..., k], `torch.float`.
        Binary tensor containing hard-decided estimations of all `k`
        information bits.

    :output crc_status: [...], `torch.bool`.
        CRC status indicating if a codeword was (most likely) correctly
        recovered. This is only returned if ``return_crc_status`` is `True`.
        Note that false positives are possible.

    .. rubric:: Notes

    This block implements the successive cancellation list (SCL) decoder
    as described in :cite:p:`Tal_SCL` using LLR-domain message updates
    :cite:p:`Stimming_LLR`. At construction time, the decoder performs a
    single depth-first traversal of the Polar decoding tree and generates
    a fixed sequence of message-passing instructions (the decoder `tape`).
    The traversal also applies the fast-SCL node shortcuts of
    :cite:p:`Hashemi_SSCL`, which collapse common sub-trees into single
    ops to avoid descending all the way to per-bit leaves.

    Each tape entry is therefore one of ``F`` (check-node update), ``G``
    (variable-node update), ``COMBINE`` (upward XOR combine), a single
    frozen or info leaf decision, or one of the fast-SCL node shortcuts
    ``R0`` / ``REP`` / ``R1``. Decoding replays this tape over shared
    workspace tensors, so the sequence of launched kernels does not
    depend on the channel input and is compatible with `torch.compile`
    and CUDA graph capture.

    The fast-SCL shortcuts cover three special sub-tree shapes: an
    all-frozen sub-tree is replaced by a single ``R0`` op, a repetition
    sub-tree (all leaves frozen except the last) by a single ``REP``
    op, and a rate-1 sub-tree (no frozen leaves) by a single ``R1``
    op. ``R1`` uses the single-flip approximation `(M=1)`, i.e., it
    keeps two alternatives per surviving path: the maximum-likelihood
    codeword and the codeword with the least-reliable bit flipped.

    Only the currently active region of the Polar tree is materialized per
    path: LLRs occupy a flat buffer of size ``n - 1`` indexed by stage,
    hard decisions for stages ``1 .. n_stages - 1`` occupy ``2n - 4``
    slots, and stage-0 leaf decisions live in a persistent ``[B, 2L, n]``
    tensor.
    The per-op workspace offsets are pre-computed during tape construction
    so that decoding issues plain tensor slices without any per-op address
    arithmetic.

    With CRC-aided decoding, the decoder selects the surviving path
    with the smallest path metric *after* adding a large CRC-failure
    penalty to every path that does not check. If no path passes the
    CRC, the path with the smallest underlying metric is returned and
    its ``crc_status`` is `False`.

    As commonly done, we assume frozen bits are set to `0`. Please note
    that - although its practical relevance is only little - setting
    frozen bits to `1` may result in `affine` codes instead of a linear
    code as the all-zero codeword is not necessarily part of the code
    any more.

    .. rubric:: Examples


    .. code-block:: python

        import torch
        from sionna.phy.fec.polar import PolarSCLDecoder, PolarEncoder
        from sionna.phy.fec.polar.utils import generate_5g_ranking

        k, n = 100, 256
        frozen_pos, _ = generate_5g_ranking(k, n)
        encoder = PolarEncoder(frozen_pos, n)
        decoder = PolarSCLDecoder(frozen_pos, n, list_size=8)

        bits = torch.randint(0, 2, (10, k), dtype=torch.float32)
        codewords = encoder(bits)
        llr_ch = 20.0 * (2.0 * codewords - 1)  # BPSK without noise
        decoded = decoder(llr_ch)
        print(torch.equal(bits, decoded))
        # True
    """

    def __init__(
        self,
        frozen_pos: np.ndarray,
        n: int,
        list_size: int = 8,
        crc_degree: Optional[str] = None,
        ind_iil_inv: Optional[np.ndarray] = None,
        return_crc_status: bool = False,
        *,
        precision: Optional[str] = None,
        device: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(precision=precision, device=device, **kwargs)

        # ---- validate scalar inputs --------------------------------------
        if not isinstance(n, numbers.Number):
            raise TypeError("n must be a number.")
        n = int(n)
        if not isinstance(list_size, int):
            raise TypeError("list_size must be integer.")
        if not isinstance(return_crc_status, bool):
            raise TypeError("return_crc_status must be bool.")
        if not np.issubdtype(frozen_pos.dtype, int):
            raise TypeError("frozen_pos contains non int.")
        if len(frozen_pos) > n:
            raise ValueError(
                "Num. of elements in frozen_pos cannot be greater than n."
            )
        if not _is_pow2(n):
            raise ValueError("n must be a power of 2.")
        if n < 2:
            raise ValueError("n must be at least 2.")
        if not _is_pow2(list_size):
            raise ValueError("list_size must be a power of 2.")

        # ---- store immutable code / decoder parameters -------------------
        self._n = n
        self._frozen_pos = frozen_pos
        self._k = n - len(frozen_pos)
        self._list_size = list_size
        self._info_pos = np.setdiff1d(np.arange(n), frozen_pos)
        self._llr_max = 30.0
        if self._k != len(self._info_pos):
            raise ArithmeticError("Internal error: invalid info_pos generated.")

        self._frozen_ind = np.zeros(n, dtype=np.int8)
        self._frozen_ind[frozen_pos] = 1
        self._n_stages = int(np.log2(n))

        self.register_buffer(
            "_info_pos_t",
            torch.tensor(self._info_pos, dtype=torch.int64, device=self.device),
        )

        # ---- CRC setup ---------------------------------------------------
        if crc_degree is not None:
            self._use_crc = True
            self._crc_encoder = CRCEncoder(
                crc_degree, precision=precision, device=device
            )
            self._crc_decoder = CRCDecoder(
                self._crc_encoder, precision=precision, device=device
            )
            self._k_crc = self._crc_decoder.encoder.crc_length
        else:
            self._use_crc = False
            self._k_crc = 0
        if self._k < self._k_crc:
            raise ValueError("Value of k is too small for given CRC_degree.")

        if (crc_degree is None) and return_crc_status:
            raise ValueError("Returning CRC status requires given crc_degree.")
        self._return_crc_status = return_crc_status

        # ---- optional CRC input-bit de-interleaver -----------------------
        if ind_iil_inv is not None:
            if ind_iil_inv.shape[0] != self._k:
                raise ValueError("ind_iil_inv must be of length k.")
            self._iil = True
            self.register_buffer(
                "_ind_iil_inv_t",
                torch.tensor(ind_iil_inv, dtype=torch.int64,
                             device=self.device),
            )
        else:
            self._iil = False

        # ---- build tape --------------------------------------------------
        # The decoder uses two small 1-D scratch buffers per path:
        #   * ``llr``  holds intermediate LLRs            (size ``n - 1``).
        #   * ``uhat`` holds intermediate bit decisions   (size ``2n - 4``).
        # ``_slot_llr[s]`` / ``_slot_uhat[s]`` give the offset at which
        # stage ``s`` lives inside the corresponding buffer; the tape
        # entries built next bake these offsets in so that the decode
        # loop only needs plain tensor slicing.
        # Stages ``0 .. n_stages - 1`` are stored in ``llr`` at offset
        # ``2^s - 1``. Stage ``n_stages`` is the channel input and is
        # read directly from ``llr_ch`` by the decode loop, so it has
        # no slot in ``llr``. We use ``None`` as a sentinel for that
        # stage rather than the buffer-size value ``n - 1``: any tape
        # entry that accidentally indexes ``llr`` with this slot will
        # raise ``TypeError`` on ``None + offset`` instead of silently
        # producing an out-of-bounds (and empty) slice.
        self._slot_llr = [(1 << s) - 1 for s in range(self._n_stages)]
        self._slot_llr.append(None)
        self._llr_flat_size = max((1 << self._n_stages) - 1, 1)
        self._slot_uhat = [0] + [
            (1 << (s + 1)) - 4 for s in range(1, self._n_stages + 1)
        ]
        self._uhat_flat_size = max((1 << (self._n_stages + 1)) - 4, 1)
        self._tape = self._build_tape()

        # Initial path-metric vector. Only slots 0 and L are valid
        # (these are the seeds of the u=0 and u=1 halves of the first
        # info-leaf split); the remaining 2L-2 slots are seeded with
        # ``+inf`` so that the first topk deterministically discards
        # them regardless of how much real paths have accumulated.
        #
        # Note: the list reaches full ``L``-path utilisation only after
        # ``ceil(log2(L))`` info-leaf forks have been consumed, since
        # each fork doubles the number of finite-metric paths from the
        # initial pair. For codes whose first ``log2(L)`` info bits land
        # in fast-SCL ``R1`` sub-trees (which fork only once via the
        # single-flip approximation rather than per bit), full diversity
        # is reached even later. This warm-up does not affect
        # correctness — discarded ``+inf`` slots never compete with real
        # paths — but it slightly reduces effective list diversity over
        # the first few decoded bits.
        pm_init = np.full(2 * list_size, np.inf, dtype=np.float64)
        pm_init[0] = 0.0
        pm_init[list_size] = 0.0
        self.register_buffer(
            "_pm_init",
            torch.tensor(pm_init, dtype=self.dtype, device=self.device),
        )

        # On affected torch versions on CPU, force a graph break around
        # the inner SCL tape replay so it isn't fused with surrounding
        # ops by inductor's CPU codegen (see ``_NEEDS_CPU_COMPILE_BREAK``
        # at the top of this module). CUDA / newer torch versions skip
        # this and keep the full fused compile.
        if _NEEDS_CPU_COMPILE_BREAK and torch.device(self.device).type == "cpu":
            _install_cpu_compile_break(self, "_decode")

    # ------------------------------------------------------------------
    # Read-only properties
    # ------------------------------------------------------------------
    @property
    def n(self):
        """Codeword length."""
        return self._n

    @property
    def k(self):
        """Number of information bits (including any outer CRC)."""
        return self._k

    @property
    def k_crc(self):
        """Length of the outer CRC (0 if none)."""
        return self._k_crc

    @property
    def frozen_pos(self):
        """Frozen positions for Polar decoding."""
        return self._frozen_pos

    @property
    def info_pos(self):
        """Information bit positions for Polar encoding."""
        return self._info_pos

    @property
    def llr_max(self):
        """Internal LLR clipping magnitude."""
        return self._llr_max

    @property
    def list_size(self):
        """Decoder list size ``L``."""
        return self._list_size

    # ==================================================================
    # Tape construction
    # ==================================================================
    @staticmethod
    def _make_F(stage, off, length, llr_src, llr_dst):
        """``F`` op — check-node update at an internal tree node.

        Pushes LLRs from the parent stage down to the left child as
        ``boxplus(x, y)``, where ``x`` and ``y`` are the two halves of
        the parent-stage LLR vector.
        """
        return _TapeEntry(_Op.F, stage, off, length, llr_src, llr_dst)

    @staticmethod
    def _make_G(stage, off, length, llr_src, llr_dst, u_src):
        """``G`` op — variable-node update at an internal tree node.

        Pushes LLRs from the parent stage down to the right child,
        conditioned on the left child's hard decision ``u``: returns
        ``x + y`` if ``u == 0`` and ``y - x`` if ``u == 1``.
        """
        return _TapeEntry(_Op.G, stage, off, length, llr_src, llr_dst, u_src)

    @staticmethod
    def _make_combine(stage, off, length, u_src, u_dst):
        """``COMBINE`` op — propagate hard decisions upward.

        Once both children of an internal node have produced their hard
        decisions ``ul`` (left) and ``ur`` (right), this writes the
        parent-stage decision pair ``(ul ^ ur, ur)`` so the next
        ``G`` / ``COMBINE`` further up the tree can read it.
        """
        return _TapeEntry(_Op.COMBINE, stage, off, length, u_src, u_dst)

    def _make_leaf_frozen(self, off, llr_src):
        """``LEAF_FROZEN`` op — frozen bit (decoded as ``0``).

        Updates the path metric with the cost of the LLR disagreeing
        with the forced ``u = 0`` decision; emits no bit decision
        because frozen bits are known to both encoder and decoder.

        ``slot_a`` is forced to 0 by convention: the F/G chain has
        already deposited the stage-0 LLR for bit ``off`` at offset 0
        before this op runs. The assertion guards against tape
        construction errors that would otherwise be silent.
        """
        if not 0 <= off < self._n:
            raise RuntimeError(
                f"LEAF_FROZEN off={off} out of range [0, {self._n}); "
                "the decoder tape is invalid."
            )
        return _TapeEntry(_Op.LEAF_FROZEN, 0, off, 1, llr_src)

    @staticmethod
    def _make_leaf_info(off, llr_src):
        """``LEAF_INFO`` op — information bit, splits each path.

        Forks every surviving path into two candidates (``u = 0`` and
        ``u = 1``), updates each metric, then prunes back to the
        ``L`` smallest-metric paths.
        """
        return _TapeEntry(_Op.LEAF_INFO, 0, off, 1, llr_src)

    @staticmethod
    def _make_R0(stage, off, length, llr_src, u_dst):
        """``R0`` op — rate-0 (all-frozen) sub-tree shortcut.

        A whole sub-tree whose leaves are all frozen decodes to the
        all-zero codeword. Adds the LLR-disagreement cost over the
        block to every path in one step (replaces the entire
        ``F``/``G``/``COMBINE``/leaf chain).
        """
        return _TapeEntry(_Op.R0, stage, off, length, llr_src, u_dst)

    @staticmethod
    def _make_REP(stage, off, length, llr_src, u_dst):
        """``REP`` op — repetition sub-tree shortcut.

        A sub-tree where every leaf is frozen except the last is a
        repetition code: the only two valid codewords are all-zero
        and all-one. Splits each path into those two hypotheses,
        updates the metrics in one step, then prunes to ``L``.
        """
        return _TapeEntry(_Op.REP, stage, off, length, llr_src, u_dst)

    @staticmethod
    def _make_R1(stage, off, length, llr_src, u_dst):
        """``R1`` op — rate-1 (no-frozen) sub-tree shortcut, ``M = 1`` flip.

        A sub-tree with no frozen leaves is unconstrained: the
        maximum-likelihood codeword-stage decision is just
        ``sign(LLR)``. To still benefit from the list, we keep one
        alternative per path that flips the least-reliable bit
        (Hashemi *et al.*'s single-flip approximation).
        """
        return _TapeEntry(_Op.R1, stage, off, length, llr_src, u_dst)

    def _build_tape(self):
        """Generate the SCL instruction sequence.

        Performs a depth-first traversal of the Polar decoding tree.
        At each internal node the pattern is
        ``F -> left subtree -> G -> right subtree -> COMBINE``. Leaves
        are either frozen or information bits. Fast-SCL shortcuts
        collapse all-frozen sub-trees into ``R0``, rate-1 sub-trees
        into ``R1``, and repetition sub-trees (all leaves frozen
        except the last) into ``REP``.

        Every emitted instruction is annotated with the workspace
        offsets it reads/writes so the decoder issues plain tensor
        slices without per-op address arithmetic at run time.
        """
        frozen = self._frozen_ind
        n_stages = self._n_stages
        sl = self._slot_llr
        su = self._slot_uhat
        tape = []

        def u_dst_for(stage, off):
            """Where this sub-tree writes its hard decisions for the
            enclosing COMBINE to read at ``stage``. Returns 0 for
            outer-stage sub-trees, whose result is never read upward."""
            if stage >= n_stages:
                return 0
            pos_in_parent = off & ((1 << (stage + 1)) - 1)
            return su[stage] + pos_in_parent

        def visit(stage, off, length):
            """Recursively generate tape ops for the sub-tree at
            ``[off, off + length)`` rooted at ``stage``.

            If the sub-tree matches one of the fast-SCL patterns
            (all-frozen, rate-1, or repetition) it is collapsed into a
            single ``R0`` / ``R1`` / ``REP`` op. Otherwise the standard
            ``F`` -> left subtree -> ``G`` -> right subtree -> ``COMBINE``
            sequence is generated, recursing on each half. Length-1
            sub-trees terminate the recursion at a frozen or info leaf.
            """
            if length == 1:
                if frozen[off]:
                    tape.append(self._make_leaf_frozen(off, sl[0]))
                else:
                    tape.append(self._make_leaf_info(off, sl[0]))
                return
            sub = frozen[off:off + length]
            n_frozen = int(sub.sum())
            if n_frozen == length:
                tape.append(self._make_R0(
                    stage, off, length, sl[stage],
                    u_dst_for(stage, off),
                ))
                return
            if n_frozen == 0:
                tape.append(self._make_R1(
                    stage, off, length, sl[stage],
                    u_dst_for(stage, off),
                ))
                return
            if n_frozen == length - 1 and sub[-1] == 0:
                tape.append(self._make_REP(
                    stage, off, length, sl[stage],
                    u_dst_for(stage, off),
                ))
                return
            half = length // 2
            # F: read parent-stage LLRs, write child-stage left half.
            tape.append(self._make_F(
                stage, off, length, sl[stage], sl[stage - 1]
            ))
            visit(stage - 1, off, half)
            # G reads the left-child hard decisions at the child stage.
            pos_in_child = off & (length - 1)
            u_left = su[stage - 1] + pos_in_child
            tape.append(self._make_G(
                stage, off, length, sl[stage], sl[stage - 1], u_left
            ))
            visit(stage - 1, off + half, half)
            # COMBINE merges the two child halves; skipped at the outer
            # stage because the caller only reads stage-0 bits.
            if stage < n_stages:
                pos_in_parent = off & ((1 << (stage + 1)) - 1)
                tape.append(self._make_combine(
                    stage, off, length, u_left, su[stage] + pos_in_parent
                ))

        visit(n_stages, 0, self._n)
        return tuple(tape)

    # ==================================================================
    # Vectorized per-op primitives (over [B, 2L, ...])
    # ==================================================================
    def _cn(self, x, y):
        """Check-node update (boxplus) for LLR inputs.

        Closed-form stable identity (see :cite:p:`Stimming_LLR`,
        :cite:p:`Hashemi_SSCL`):

        .. math::

            \\text{boxplus}(x, y) = \\text{sign}(x)\\,\\text{sign}(y)
                \\, \\min(|x|, |y|)
                + \\text{softplus}(-|x + y|)
                - \\text{softplus}(-|x - y|).

        Inputs are clamped to ``[-llr_max, llr_max]`` so that VN
        outputs (which can grow up to ``2 * llr_max``) are bounded
        before they re-enter the next ``F`` op.
        """
        x_c = torch.clamp(x, -self._llr_max, self._llr_max)
        y_c = torch.clamp(y, -self._llr_max, self._llr_max)
        return (
            torch.sign(x_c) * torch.sign(y_c)
            * torch.minimum(x_c.abs(), y_c.abs())
            + softplus(-(x_c + y_c).abs())
            - softplus(-(x_c - y_c).abs())
        )

    @staticmethod
    def _vn(x, y, u):
        """Variable-node update for LLR inputs.

        Returns ``x + y`` if the left-child hard decision ``u`` is 0,
        and ``y - x`` if ``u`` is 1.
        """
        return (1.0 - 2.0 * u) * x + y

    @staticmethod
    def _polar_transform_inplace(u_hat, length):
        """Apply the size-``length`` Polar butterfly transform in place.

        Used by the rate-1 op to map codeword-stage hard decisions
        ``c_hat`` back to the stage-0 information bits ``u_hat``;
        the Polar transform is its own inverse over `GF(2)`.

        ``u_hat`` is modified in place. The same tensor is also
        returned so the function can be chained, but callers should
        not rely on the return value to obtain a fresh copy.
        """
        shape = u_hat.shape
        prefix = shape[:-1]
        stride = 1
        while stride < length:
            v = u_hat.view(*prefix, length // (2 * stride), 2, stride)
            v[..., 0, :] ^= v[..., 1, :]
            stride *= 2
        return u_hat

    # ------------------------------------------------------------------
    # Path pruning (topk + tile)
    # ------------------------------------------------------------------
    def _topk_tile(self, pm, llr, uhat, bits):
        """Prune ``2L`` candidate paths back to the best ``L`` and re-tile.

        After every info-leaf / ``REP`` / ``R1`` op each of the ``L``
        surviving paths has been split into two candidates, leaving
        ``2L`` paths along the list dimension. This helper:

        1. picks the ``L`` paths with the smallest path metric, and
        2. duplicates that selection back up to ``2L`` (slots ``0..L-1``
           and ``L..2L-1`` carry the same ``L`` survivors), so the next
           info-leaf can fork the top half into ``u = 0`` and the bottom
           half into ``u = 1`` without further bookkeeping.

        ``pm``, ``llr``, ``uhat`` and ``bits`` are gathered along the
        list dimension with the same index, so each path keeps a
        consistent view of its workspace state. Returns the gathered
        ``(pm, llr, uhat, bits)``.
        """
        B = pm.shape[0]
        L = self._list_size
        # Pre-doubling the L indices to 2L (via ``repeat(1, 2)``) lets
        # us produce the tiled survivor layout in a single ``gather``
        # call instead of separate ``gather`` + ``concat`` ops.
        _, top = torch.topk(-pm, L, dim=-1)
        idx = top.repeat(1, 2)
        pm = torch.gather(pm, 1, idx)
        llr = torch.gather(
            llr, 1, idx.view(B, 2 * L, 1).expand(-1, -1, llr.shape[2])
        )
        uhat = torch.gather(
            uhat, 1, idx.view(B, 2 * L, 1).expand(-1, -1, uhat.shape[2])
        )
        bits = torch.gather(
            bits, 1, idx.view(B, 2 * L, 1).expand(-1, -1, bits.shape[2])
        )
        return pm, llr, uhat, bits

    # ==================================================================
    # Main decode
    # ==================================================================
    def _decode(self, llr_ch):
        """SCL decoding on the packed workspace.

        Replays the decoder tape over the following per-path tensors:

          * ``llr[B, 2L, n - 1]``    — LLRs for stages ``s < n_stages``.
            Stage ``s`` occupies ``2^s`` slots starting at offset
            ``2^s - 1``. Channel LLRs (stage ``n_stages``) are read from
            ``llr_ch`` directly and are not copied into this buffer.
          * ``uhat[B, 2L, 2n - 4]``  — intermediate hard decisions for
            stages ``1 .. n_stages - 1``. The slot for stage ``n_stages``
            is unused because the outer ``COMBINE`` is skipped.
          * ``bits[B, 2L, n]``       — persistent stage-0 leaf decisions.

        Channel LLRs are assumed pre-clamped (see ``call``); ``_cn``
        is the only place that clamps after that. Inputs to other ops
        are bounded by ``2 * llr_max`` (immediately after a VN), and
        ``softplus`` handles those magnitudes precisely.

        Returns ``(bits, pm)`` sorted in ascending order of ``pm``,
        so index 0 is the surviving path with the smallest metric.
        ``bits.shape = [B, 2L, n]``.
        """
        B = llr_ch.shape[0]
        L = self._list_size
        n = self._n
        n_stages = self._n_stages
        dev, dt = llr_ch.device, llr_ch.dtype

        llr = torch.zeros(B, 2 * L, self._llr_flat_size, dtype=dt, device=dev)
        uhat = torch.zeros(
            B, 2 * L, self._uhat_flat_size, dtype=torch.int8, device=dev
        )
        bits = torch.zeros(B, 2 * L, n, dtype=torch.int8, device=dev)
        pm = self._pm_init.to(dtype=dt, device=dev).expand(B, -1).clone()

        for entry in self._tape:
            op = entry.op
            stage, off, length = entry.stage, entry.off, entry.length
            half = length // 2

            if op == _Op.F:
                src, dst = entry.slot_a, entry.slot_b
                if stage == n_stages:
                    # Add size-1 path axis so channel LLRs broadcast over 2L.
                    x = llr_ch[:, off:off + half].unsqueeze(1)
                    y = llr_ch[:, off + half:off + length].unsqueeze(1)
                else:
                    x = llr[:, :, src:src + half]
                    y = llr[:, :, src + half:src + length]
                llr[:, :, dst:dst + half] = self._cn(x, y)

            elif op == _Op.G:
                src, dst, u_src = entry.slot_a, entry.slot_b, entry.slot_c
                if stage == n_stages:
                    # Add size-1 path axis so channel LLRs broadcast over 2L.
                    x = llr_ch[:, off:off + half].unsqueeze(1)
                    y = llr_ch[:, off + half:off + length].unsqueeze(1)
                else:
                    x = llr[:, :, src:src + half]
                    y = llr[:, :, src + half:src + length]
                # Left-child hard decisions live in `bits` when the
                # child is a leaf stage (stage==1), otherwise in `uhat`.
                if stage == 1:
                    u = bits[:, :, off:off + half].to(dt)
                else:
                    u = uhat[:, :, u_src:u_src + half].to(dt)
                llr[:, :, dst:dst + half] = self._vn(x, y, u)

            elif op == _Op.COMBINE:
                u_src, u_dst = entry.slot_a, entry.slot_b
                if stage == 1:
                    ul = bits[:, :, off:off + half]
                    ur = bits[:, :, off + half:off + length]
                else:
                    ul = uhat[:, :, u_src:u_src + half]
                    ur = uhat[:, :, u_src + half:u_src + length]
                uhat[:, :, u_dst:u_dst + half] = ul ^ ur
                uhat[:, :, u_dst + half:u_dst + length] = ur

            elif op == _Op.LEAF_FROZEN:
                l_val = llr[:, :, entry.slot_a]
                pm.add_(softplus(-l_val))

            elif op == _Op.LEAF_INFO:
                # Top L paths: u=0 (bits already 0). Bottom L: u=1.
                bits[:, L:, off] = 1
                l_val = llr[:, :, entry.slot_a]
                u = bits[:, :, off].to(dt)
                pm.add_(softplus(-(1.0 - 2.0 * u) * l_val))
                pm, llr, uhat, bits = self._topk_tile(pm, llr, uhat, bits)

            elif op == _Op.R0:
                if stage == n_stages:
                    # Add size-1 path axis so channel LLRs broadcast over 2L.
                    l_val = llr_ch[:, off:off + length].unsqueeze(1)
                else:
                    src = entry.slot_a
                    l_val = llr[:, :, src:src + length]
                pm.add_(softplus(-l_val).sum(dim=-1))
                # R0 skips the internal COMBINE chain. The path-pruning
                # gather can reorder rows of `uhat`, so we explicitly
                # zero this region for every path before the parent
                # COMBINE reads it.
                if stage < n_stages:
                    u_dst = entry.slot_b
                    uhat[:, :, u_dst:u_dst + length] = 0

            elif op == _Op.REP:
                if stage == n_stages:
                    # Whole code is a single REP subtree — channel LLRs
                    # are shared across all paths, so broadcast to 2L
                    # before splitting into the two hypotheses.
                    l_val = (
                        llr_ch[:, off:off + length]
                        .unsqueeze(1)
                        .expand(B, 2 * L, length)
                    )
                else:
                    src = entry.slot_a
                    l_val = llr[:, :, src:src + length]
                llr_pm = torch.cat(
                    [l_val[:, :L, :], -l_val[:, L:, :]], dim=1
                )
                pm.add_(softplus(-llr_pm).sum(dim=-1))
                u_dst = entry.slot_b
                if stage < n_stages:
                    uhat[:, :L, u_dst:u_dst + length] = 0
                    uhat[:, L:, u_dst:u_dst + length] = 1
                bits[:, L:, off + length - 1] = 1
                pm, llr, uhat, bits = self._topk_tile(pm, llr, uhat, bits)

            elif op == _Op.R1:
                # Rate-1 sub-tree. ML codeword-stage decision is the
                # sign of the LLRs; the M=1 flip alternative inverts
                # the least-reliable bit on the bottom-L paths.
                if stage == n_stages:
                    l_val = (
                        llr_ch[:, off:off + length]
                        .unsqueeze(1)
                        .expand(B, 2 * L, length)
                    )
                else:
                    src = entry.slot_a
                    l_val = llr[:, :, src:src + length]
                abs_l = l_val.abs()
                pm_add = softplus(-abs_l).sum(dim=-1)
                delta, j = abs_l.min(dim=-1)
                c_hat = (l_val < 0).to(torch.int8)
                flip = torch.zeros_like(c_hat)
                flip[:, L:].scatter_(2, j[:, L:].unsqueeze(-1), 1)
                c_hat = c_hat ^ flip
                pm_add[:, L:] = pm_add[:, L:] + delta[:, L:]
                pm.add_(pm_add)
                # Parent COMBINE reads c_hat at stage s. We must store
                # c_hat into uhat *before* mutating it via the Polar
                # transform on the way to `bits`.
                if stage < n_stages:
                    u_dst = entry.slot_b
                    uhat[:, :, u_dst:u_dst + length] = c_hat
                self._polar_transform_inplace(c_hat, length)
                bits[:, :, off:off + length] = c_hat
                pm, llr, uhat, bits = self._topk_tile(pm, llr, uhat, bits)
            else:
                raise RuntimeError(f"Unknown tape op {op!r}")

        ind = torch.argsort(pm, dim=-1)
        pm = torch.gather(pm, 1, ind)
        bits = torch.gather(bits, 1, ind.view(B, 2 * L, 1).expand(-1, -1, n))
        return bits, pm

    # ==================================================================
    # Public API
    # ==================================================================
    def build(self, input_shape):
        if input_shape[-1] != self._n:
            raise ValueError("Invalid input shape.")

    def call(self, llr_ch):
        input_shape = llr_ch.shape
        llr_ch = llr_ch.reshape(-1, self._n)
        # Convert input logits to LLRs and clamp once at ingest. All
        # downstream ops rely on bounded inputs to ``_cn`` and on
        # ``softplus`` for the leaf/R0/R1/REP path-metric updates,
        # which are precise for inputs of magnitude up to ~2*llr_max.
        llr_ch = torch.clamp(-llr_ch, -self._llr_max, self._llr_max)

        bits, msg_pm = self._decode(llr_ch)  # [B, 2L, n], [B, 2L]

        if self._use_crc:
            u_hat_list = bits[:, :, self._info_pos_t].to(llr_ch.dtype)
            if self._iil:
                u_hat_list_crc = u_hat_list[:, :, self._ind_iil_inv_t]
            else:
                u_hat_list_crc = u_hat_list
            # ``CRCDecoder`` returns ``crc_valid`` with shape
            # [B, 2L, 1]; the trailing singleton is squeezed for the
            # path-metric penalty and again before reshaping the
            # public crc_status output.
            _, crc_valid = self._crc_decoder(u_hat_list_crc)
            pm_penalty = (1.0 - crc_valid.float()) * self._llr_max * self._k
            msg_pm = msg_pm + pm_penalty.squeeze(-1)

        cand_ind = torch.argmin(msg_pm, dim=-1)
        batch_indices = torch.arange(bits.shape[0], device=bits.device)
        c_hat = bits[batch_indices, cand_ind, :].to(llr_ch.dtype)
        u_hat = c_hat[:, self._info_pos_t]

        output_shape = list(input_shape[:-1]) + [self._k]
        u_hat_reshape = u_hat.reshape(output_shape)

        if self._return_crc_status:
            crc_status = crc_valid[batch_indices, cand_ind].squeeze(-1)
            crc_status = crc_status.reshape(list(input_shape[:-1]))
            return u_hat_reshape, crc_status
        return u_hat_reshape

class PolarBPDecoder(Block):
    # pylint: disable=line-too-long
    """Belief propagation (BP) decoder for Polar codes :cite:p:`Arikan_Polar` and
    Polar-like codes based on :cite:p:`Arikan_BP` and :cite:p:`Forney_Graphs`.

    :param frozen_pos: Array of `int` defining the ``n-k`` indices of the
        frozen positions.
    :param n: Defining the codeword length.
    :param num_iter: Defining the number of decoder iterations (no early
        stopping used at the moment).
    :param hard_out: If `True`, the decoder provides hard-decided
        information bits instead of soft-values.
    :param precision: Precision used for internal calculations and outputs.
        If `None`, :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for computation (e.g., 'cpu', 'cuda:0').
        If `None`, :attr:`~sionna.phy.config.Config.device` is used.

    :input llr_ch: [..., n], `torch.float`.
        Tensor containing the channel logits/llr values.

    :output u_hat: [..., k], `torch.float`.
        Tensor containing bit-wise soft-estimates (or hard-decided
        bit-values) of all ``k`` information bits.

    .. rubric:: Notes

    This decoder is fully differentiable and, thus, well-suited for
    gradient descent-based learning tasks such as `learned code design`
    :cite:p:`Ebada_Design`.

    As commonly done, we assume frozen bits are set to `0`. Please note
    that - although its practical relevance is only little - setting frozen
    bits to `1` may result in `affine` codes instead of linear code as the
    `all-zero` codeword is not necessarily part of the code any more.

    .. rubric:: Examples


    .. code-block:: python

        import torch
        from sionna.phy.fec.polar import PolarBPDecoder, PolarEncoder
        from sionna.phy.fec.polar.utils import generate_5g_ranking

        k, n = 100, 256
        frozen_pos, _ = generate_5g_ranking(k, n)
        encoder = PolarEncoder(frozen_pos, n)
        decoder = PolarBPDecoder(frozen_pos, n, num_iter=20)

        bits = torch.randint(0, 2, (10, k), dtype=torch.float32)
        codewords = encoder(bits)
        llr_ch = 20.0 * (2.0 * codewords - 1)  # BPSK without noise
        decoded = decoder(llr_ch)
        print(torch.equal(bits, decoded))
        # True
    """

    def __init__(
        self,
        frozen_pos: np.ndarray,
        n: int,
        num_iter: int = 20,
        hard_out: bool = True,
        *,
        precision: Optional[str] = None,
        device: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(precision=precision, device=device, **kwargs)

        if not isinstance(n, numbers.Number):
            raise TypeError("n must be a number.")
        n = int(n)
        if not np.issubdtype(frozen_pos.dtype, int):
            raise TypeError("frozen_pos contains non int.")
        if len(frozen_pos) > n:
            msg = "Num. of elements in frozen_pos cannot be greater than n."
            raise ValueError(msg)
        if not _is_pow2(n):
            raise ValueError("n must be a power of 2.")

        if not isinstance(hard_out, bool):
            raise TypeError("hard_out must be boolean.")

        # Store internal attributes
        self._n = n
        self._frozen_pos = frozen_pos
        self._k = self._n - len(self._frozen_pos)
        self._info_pos = np.setdiff1d(np.arange(self._n), self._frozen_pos)
        if self._k != len(self._info_pos):
            raise ArithmeticError("Internal error: invalid info_pos generated.")

        # Register info/frozen positions as buffers for torch.compile
        # compatibility (avoid numpy indexing inside the hot loop).
        self.register_buffer(
            "_info_pos_t",
            torch.tensor(self._info_pos, dtype=torch.int64, device=self.device),
        )
        self.register_buffer(
            "_frozen_pos_t",
            torch.tensor(self._frozen_pos, dtype=torch.int64,
                         device=self.device),
        )

        if not isinstance(num_iter, int):
            raise TypeError("num_iter must be integer.")
        if num_iter <= 0:
            raise ValueError("num_iter must be a positive value.")
        self._num_iter = num_iter

        self._llr_max = 19.3
        self._hard_out = hard_out

        self._n_stages = int(np.log2(self._n))

        # Pre-compute the per-stage butterfly index patterns once so the
        # decode loop only performs tensor ops (no numpy, no allocation).
        ind_range = np.arange(self._n // 2)
        stage_ind_1 = []
        stage_ind_2 = []
        stage_ind_inv = []
        for ind_s in range(self._n_stages):
            ind_1 = ind_range * 2 - np.mod(ind_range, 2**ind_s)
            ind_2 = ind_1 + 2**ind_s
            ind_inv = np.argsort(np.concatenate([ind_1, ind_2], axis=0))
            stage_ind_1.append(ind_1)
            stage_ind_2.append(ind_2)
            stage_ind_inv.append(ind_inv)
        self.register_buffer(
            "_stage_ind_1",
            torch.tensor(np.stack(stage_ind_1), dtype=torch.int64,
                         device=self.device),
        )
        self.register_buffer(
            "_stage_ind_2",
            torch.tensor(np.stack(stage_ind_2), dtype=torch.int64,
                         device=self.device),
        )
        self.register_buffer(
            "_stage_ind_inv",
            torch.tensor(np.stack(stage_ind_inv), dtype=torch.int64,
                         device=self.device),
        )

        # See ``_NEEDS_CPU_COMPILE_BREAK`` at the top of this module.
        if _NEEDS_CPU_COMPILE_BREAK and torch.device(self.device).type == "cpu":
            _install_cpu_compile_break(self, "_decode_bp")

    @property
    def n(self) -> int:
        """Codeword length."""
        return self._n

    @property
    def k(self) -> int:
        """Number of information bits."""
        return self._k

    @property
    def frozen_pos(self) -> np.ndarray:
        """Frozen positions for Polar decoding."""
        return self._frozen_pos

    @property
    def info_pos(self) -> np.ndarray:
        """Information bit positions for Polar encoding."""
        return self._info_pos

    @property
    def llr_max(self) -> float:
        """Maximum LLR value for internal calculations."""
        return self._llr_max

    @property
    def num_iter(self) -> int:
        """Number of decoding iterations."""
        return self._num_iter

    @num_iter.setter
    def num_iter(self, num_iter: int) -> None:
        """Number of decoding iterations."""
        if not isinstance(num_iter, int):
            raise TypeError("num_iter must be integer.")
        if num_iter <= 0:
            raise ValueError("num_iter must be a positive value.")
        self._num_iter = num_iter

    @property
    def hard_out(self) -> bool:
        """Indicates if decoder hard-decides outputs."""
        return self._hard_out

    @hard_out.setter
    def hard_out(self, hard_out: bool) -> None:
        if not isinstance(hard_out, bool):
            raise TypeError("hard_out must be bool.")
        self._hard_out = hard_out

    def _boxplus(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Check-node update (boxplus) for LLR inputs."""
        x_in = torch.clamp(x, min=-self._llr_max, max=self._llr_max)
        y_in = torch.clamp(y, min=-self._llr_max, max=self._llr_max)

        llr_out = torch.log(1 + torch.exp(x_in + y_in))
        llr_out = llr_out - torch.log(torch.exp(x_in) + torch.exp(y_in))

        return llr_out

    def _bp_iter_graph_break(self) -> None:
        """Break the compile graph between BP iterations on CPU.

        Without this, dynamo unrolls all ``num_iter`` sweeps into one
        inductor kernel that can exceed GCC's size limits and ICE during
        C++ codegen (see pytorch#180212).
        """
        if (
            torch.compiler.is_compiling()
            and torch.device(self.device).type == "cpu"
        ):
            torch._dynamo.graph_break()

    def _bp_single_iteration(
        self,
        ind_it: int,
        llr_ch: torch.Tensor,
        msg_l_prev,
        msg_r_in: torch.Tensor,
        zeros_half: torch.Tensor,
    ):
        """One forward/backward BP sweep.

        Returns ``(msg_l_iter, msg_r_iter)`` — per-stage message tensors
        for this iteration.
        """
        msg_l_iter = [None] * (self._n_stages + 1)
        msg_r_iter = [None] * (self._n_stages + 1)

        # Update left-to-right messages
        for ind_s in range(self._n_stages):
            ind_1 = self._stage_ind_1[ind_s]
            ind_2 = self._stage_ind_2[ind_s]
            ind_inv = self._stage_ind_inv[ind_s]

            if ind_s == self._n_stages - 1:
                l1_in = llr_ch[:, ind_1]
                l2_in = llr_ch[:, ind_2]
            elif ind_it == 0:
                l1_in = zeros_half
                l2_in = zeros_half
            else:
                l_in = msg_l_prev[ind_s + 1]
                l1_in = l_in[:, ind_1]
                l2_in = l_in[:, ind_2]

            if ind_s == 0:
                r1_in = msg_r_in[:, ind_1]
                r2_in = msg_r_in[:, ind_2]
            else:
                r_in = msg_r_iter[ind_s]
                r1_in = r_in[:, ind_1]
                r2_in = r_in[:, ind_2]

            r1_out = self._boxplus(r1_in, l2_in + r2_in)
            r2_out = self._boxplus(r1_in, l1_in) + r2_in

            r_out = torch.cat([r1_out, r2_out], 1)
            r_out = r_out[:, ind_inv]
            msg_r_iter[ind_s + 1] = r_out

        # Update right-to-left messages
        for ind_s in range(self._n_stages - 1, -1, -1):
            ind_1 = self._stage_ind_1[ind_s]
            ind_2 = self._stage_ind_2[ind_s]
            ind_inv = self._stage_ind_inv[ind_s]

            if ind_s == self._n_stages - 1:
                l1_in = llr_ch[:, ind_1]
                l2_in = llr_ch[:, ind_2]
            else:
                l_in = msg_l_iter[ind_s + 1]
                l1_in = l_in[:, ind_1]
                l2_in = l_in[:, ind_2]

            if ind_s == 0:
                r1_in = msg_r_in[:, ind_1]
                r2_in = msg_r_in[:, ind_2]
            else:
                r_in = msg_r_iter[ind_s]
                r1_in = r_in[:, ind_1]
                r2_in = r_in[:, ind_2]

            l1_out = self._boxplus(l1_in, l2_in + r2_in)
            l2_out = self._boxplus(r1_in, l1_in) + l2_in

            l_out = torch.cat([l1_out, l2_out], 1)
            l_out = l_out[:, ind_inv]
            msg_l_iter[ind_s] = l_out

        return msg_l_iter, msg_r_iter

    def _decode_bp(
        self, llr_ch: torch.Tensor, num_iter: int
    ) -> torch.Tensor:
        """Iterative BP decoding function with LLR-values."""
        bs = llr_ch.shape[0]
        device = llr_ch.device

        msg_l = [[None] * (self._n_stages + 1) for _ in range(num_iter)]
        msg_r = [[None] * (self._n_stages + 1) for _ in range(num_iter)]

        msg_r_in = torch.zeros((bs, self._n), dtype=self.dtype, device=device)
        msg_r_in[:, self._frozen_pos_t] = self._llr_max

        zeros_half = torch.zeros((bs, self._n // 2), dtype=self.dtype,
                                 device=device)

        msg_l_prev = None
        for ind_it in range(num_iter):
            msg_l_iter, msg_r_iter = self._bp_single_iteration(
                ind_it, llr_ch, msg_l_prev, msg_r_in, zeros_half
            )
            msg_l[ind_it] = msg_l_iter
            msg_r[ind_it] = msg_r_iter
            msg_l_prev = msg_l_iter
            if ind_it + 1 < num_iter:
                self._bp_iter_graph_break()

        u_hat = msg_l[num_iter - 1][0][:, self._info_pos_t]

        if self._hard_out:
            u_hat = torch.where(
                u_hat > 0,
                torch.zeros_like(u_hat),
                torch.ones_like(u_hat),
            )
        else:
            u_hat = -1.0 * u_hat  # Re-transform to logits

        return u_hat

    def build(self, input_shape: Tuple[int, ...]) -> None:
        """Build and check if shape of input is invalid."""
        if input_shape[-1] != self._n:
            raise ValueError("Invalid input shape.")

    def call(self, llr_ch: torch.Tensor) -> torch.Tensor:
        """Iterative BP decoding function.

        This function performs ``num_iter`` belief propagation decoding
        iterations and returns the estimated information bits.

        :param llr_ch: Tensor of shape `[..., n]` containing the
            channel logits/llr values.

        :output u_hat: Tensor of shape `[..., k]` containing bit-wise
            soft-estimates (or hard-decided bit-values) of all ``k``
            information bits.

        Note: This function recursively unrolls the BP decoding graph,
        thus, for larger values of ``n`` or more iterations, building the
        decoding graph can become time and memory consuming.
        """
        # Reshape inputs to [-1, n]
        input_shape = llr_ch.shape
        new_shape = (-1, self._n)
        llr_ch = llr_ch.reshape(new_shape)

        llr_ch = -1.0 * llr_ch  # Logits to LLRs

        # Decode
        u_hat = self._decode_bp(llr_ch, self._num_iter)

        # Reconstruct input shape
        output_shape = list(input_shape[:-1]) + [self.k]
        u_hat_reshape = u_hat.reshape(output_shape)
        return u_hat_reshape


class Polar5GDecoder(Block):
    # pylint: disable=line-too-long
    """Wrapper for 5G NR Polar decoding including rate-recovery and CRC
    removal. Matches :class:`~sionna.phy.fec.polar.encoding.Polar5GEncoder`,
    including the downlink (`DCI`) deviations described there.

    :param enc_polar: Instance of the
        :class:`~sionna.phy.fec.polar.encoding.Polar5GEncoder` used for
        encoding including rate-matching.
    :param dec_type: Defining the decoder to be used. Must be one of
        `{"SC", "SCL", "BP"}`.
    :param list_size: Defining the list size iff list-decoding is used.
        Only required for ``dec_type`` `"SCL"`.
    :param num_iter: Defining the number of BP iterations. Only required
        for ``dec_type`` `"BP"`.
    :param return_crc_status: If `True`, the decoder additionally returns
        the CRC status indicating if a codeword was (most likely) correctly
        recovered.
    :param precision: Precision used for internal calculations and outputs.
        If `None`, :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for computation (e.g., 'cpu', 'cuda:0').
        If `None`, :attr:`~sionna.phy.config.Config.device` is used.

    :input llr_ch: [..., n], `torch.float`.
        Tensor containing the channel logits/llr values.

    :output b_hat: [..., k], `torch.float`.
        Binary tensor containing hard-decided estimations of all `k`
        information bits.

    :output crc_status: [...], `torch.bool`.
        CRC status indicating if a codeword was (most likely) correctly
        recovered. This is only returned if ``return_crc_status`` is `True`.
        Note that false positives are possible.

    .. rubric:: Notes

    This block supports the uplink and downlink Polar rate-matching scheme
    without `codeword segmentation`.

    Although the decoding `list size` is not provided by 3GPP
    :cite:p:`3GPPTS38212`, the consortium has agreed on a `list size` of 8 for
    the 5G decoding reference curves :cite:p:`Bioglio_Design`.
    ``dec_type="SCL"`` uses
    :class:`~sionna.phy.fec.polar.decoding.PolarSCLDecoder`, including the
    rate-1 single-flip approximation.

    All list-decoders apply `CRC-aided` decoding, however, the non-list
    decoders (`"SC"` and `"BP"`) cannot materialize the CRC leading to an
    effective rate-loss.

    .. rubric:: Examples


    .. code-block:: python

        import torch
        from sionna.phy.fec.polar import Polar5GEncoder, Polar5GDecoder

        k, n = 100, 200
        encoder = Polar5GEncoder(k, n)
        decoder = Polar5GDecoder(encoder, dec_type="SCL", list_size=8)

        bits = torch.randint(0, 2, (10, k), dtype=torch.float32)
        codewords = encoder(bits)
        llr_ch = 20.0 * (2.0 * codewords - 1)  # BPSK without noise
        decoded = decoder(llr_ch)
        print(torch.equal(bits, decoded))
        # True
    """

    def __init__(
        self,
        enc_polar: Polar5GEncoder,
        dec_type: str = "SC",
        list_size: int = 8,
        num_iter: int = 20,
        return_crc_status: bool = False,
        *,
        precision: Optional[str] = None,
        device: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(precision=precision, device=device, **kwargs)

        if not isinstance(enc_polar, Polar5GEncoder):
            raise TypeError("enc_polar must be Polar5GEncoder.")
        if not isinstance(dec_type, str):
            raise TypeError("dec_type must be str.")

        # Store internal attributes
        self._n_target = enc_polar.n_target
        self._k_target = enc_polar.k_target
        self._n_polar = enc_polar.n_polar
        self._k_polar = enc_polar.k_polar
        self._k_crc = enc_polar.enc_crc.crc_length
        self._bil = enc_polar.channel_type == "uplink"
        self._iil = enc_polar.channel_type == "downlink"
        self._llr_max = 100
        self._enc_polar = enc_polar
        self._dec_type = dec_type

        # Initialize the de-interleaver patterns
        self._init_interleavers()

        # Initialize decoder
        if dec_type == "SC":
            warnings.warn(
                "5G Polar codes use an integrated CRC that cannot be "
                "materialized with SC decoding and, thus, causes a "
                "degraded performance. Please consider SCL decoding "
                "instead.",
                UserWarning,
                stacklevel=2,
            )
            self._polar_dec = PolarSCDecoder(
                self._enc_polar.frozen_pos,
                self._n_polar,
                precision=precision,
                device=device,
            )
        elif dec_type == "SCL":
            self._polar_dec = PolarSCLDecoder(
                self._enc_polar.frozen_pos,
                self._n_polar,
                crc_degree=self._enc_polar.enc_crc.crc_degree,
                list_size=list_size,
                ind_iil_inv=self.ind_iil_inv,
                precision=precision,
                device=device,
            )
        elif dec_type == "BP":
            warnings.warn(
                "5G Polar codes use an integrated CRC that cannot be "
                "materialized with BP decoding and, thus, causes a "
                "degraded performance. Please consider SCL decoding "
                "instead.",
                UserWarning,
                stacklevel=2,
            )
            if not isinstance(num_iter, int):
                raise TypeError("num_iter must be int.")
            if num_iter <= 0:
                raise ValueError("num_iter must be positive.")
            self._num_iter = num_iter
            self._polar_dec = PolarBPDecoder(
                self._enc_polar.frozen_pos,
                self._n_polar,
                num_iter=num_iter,
                hard_out=True,
                precision=precision,
                device=device,
            )
        else:
            raise ValueError("Unknown value for dec_type.")

        if not isinstance(return_crc_status, bool):
            raise TypeError("return_crc_status must be bool.")

        self._return_crc_status = return_crc_status
        if self._return_crc_status:
            if dec_type == "SCL":
                self._dec_crc = self._polar_dec._crc_decoder
            else:
                self._dec_crc = CRCDecoder(
                    self._enc_polar.enc_crc,
                    precision=precision,
                    device=device,
                )

    @property
    def k_target(self) -> int:
        """Number of information bits including rate-matching."""
        return self._k_target

    @property
    def n_target(self) -> int:
        """Codeword length including rate-matching."""
        return self._n_target

    @property
    def k_polar(self) -> int:
        """Number of information bits of mother Polar code."""
        return self._k_polar

    @property
    def n_polar(self) -> int:
        """Codeword length of mother Polar code."""
        return self._n_polar

    @property
    def llr_max(self) -> float:
        """Maximum LLR value for internal calculations."""
        return self._llr_max

    @property
    def dec_type(self) -> str:
        """Decoder type used for decoding as str."""
        return self._dec_type

    @property
    def polar_dec(self):
        """Decoder instance used for decoding."""
        return self._polar_dec

    def _init_interleavers(self) -> None:
        """Initialize inverse interleaver patterns for rate-recovery."""
        # Channel interleaver
        ind_ch_int = self._enc_polar.channel_interleaver(
            np.arange(self._n_target)
        )
        self.ind_ch_int_inv = np.argsort(ind_ch_int)

        # Sub-block interleaver
        ind_sub_int = self._enc_polar.subblock_interleaving(
            np.arange(self._n_polar)
        )
        self.ind_sub_int_inv = np.argsort(ind_sub_int)

        # Input bit interleaver
        if self._iil:
            self.ind_iil_inv = np.argsort(
                self._enc_polar.input_interleaver(np.arange(self._k_polar))
            )
        else:
            self.ind_iil_inv = None

        # Register as buffers for torch.compile compatibility
        self.register_buffer(
            "_ind_ch_int_inv_t",
            torch.tensor(
                self.ind_ch_int_inv, dtype=torch.int32, device=self.device
            ),
        )
        self.register_buffer(
            "_ind_sub_int_inv_t",
            torch.tensor(
                self.ind_sub_int_inv, dtype=torch.int32, device=self.device
            ),
        )
        if self._iil:
            self.register_buffer(
                "_ind_iil_inv_t",
                torch.tensor(
                    self.ind_iil_inv, dtype=torch.int32, device=self.device
                ),
            )
        else:
            self._ind_iil_inv_t = None

    def build(self, input_shape: Tuple[int, ...]) -> None:
        """Build and check if shape of input is invalid."""
        if input_shape[-1] != self._n_target:
            raise ValueError("Invalid input shape.")

    def call(
        self, llr_ch: torch.Tensor
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Polar decoding and rate-recovery for uplink 5G Polar codes.

        :param llr_ch: Tensor of shape `[..., n]` containing the
            channel logits/llr values.

        :output b_hat: Tensor of shape `[..., k]` containing hard-decided
            estimates of all ``k`` information bits.

        :output crc_status: CRC status. Returned only if
            ``return_crc_status`` is `True`.
        """
        input_shape = llr_ch.shape
        new_shape = (-1, self._n_target)
        llr_ch = llr_ch.reshape(new_shape)

        # 1.) Undo channel interleaving
        if self._bil:
            llr_deint = llr_ch[:, self._ind_ch_int_inv_t]
        else:
            llr_deint = llr_ch

        # 2.) Remove puncturing, shortening, repetition
        if self._n_target >= self._n_polar:
            # Repetition coding
            n_rep = self._n_target - self._n_polar
            llr_1 = llr_deint[:, :n_rep]
            llr_2 = llr_deint[:, n_rep : self._n_polar]
            llr_3 = llr_deint[:, self._n_polar :]
            llr_dematched = torch.cat([llr_1 + llr_3, llr_2], 1)
        else:
            if self._k_polar / self._n_target <= 7 / 16:
                # Puncturing
                llr_zero = torch.zeros(
                    (llr_deint.shape[0], self._n_polar - self._n_target),
                    dtype=self.dtype,
                    device=llr_deint.device,
                )
                llr_dematched = torch.cat([llr_zero, llr_deint], 1)
            else:
                # Shortening
                llr_infty = (
                    -self._llr_max
                    * torch.ones(
                        (llr_deint.shape[0], self._n_polar - self._n_target),
                        dtype=self.dtype,
                        device=llr_deint.device,
                    )
                )
                llr_dematched = torch.cat([llr_deint, llr_infty], 1)

        # 3.) Remove subblock interleaving
        llr_dec = llr_dematched[:, self._ind_sub_int_inv_t]

        # 4.) Run main decoder
        u_hat_crc = self._polar_dec(llr_dec)

        # 5.) Remove input bit interleaving for downlink channels only
        if self._ind_iil_inv_t is not None:
            u_hat_crc = u_hat_crc[:, self._ind_iil_inv_t]

        # 6.) Evaluate or remove CRC (and PC)
        if self._return_crc_status:
            u_hat, crc_status = self._dec_crc(u_hat_crc)
        else:
            u_hat = u_hat_crc[:, : -self._k_crc]

        # Reconstruct input shape
        output_shape = list(input_shape[:-1]) + [self._k_target]
        u_hat_reshape = u_hat.reshape(output_shape)
        u_hat_reshape = u_hat_reshape.to(self.dtype)

        if self._return_crc_status:
            output_shape_crc = list(input_shape[:-1])
            crc_status = crc_status.reshape(output_shape_crc)
            return u_hat_reshape, crc_status
        else:
            return u_hat_reshape

