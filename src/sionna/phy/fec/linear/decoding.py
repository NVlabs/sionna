#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Blocks for decoding of linear codes."""

from typing import Optional
import itertools
import warnings

import numpy as np
import scipy as sp
import torch

from sionna.phy import Block
from sionna.phy.fec.coding import pcm2gm, make_systematic
from sionna.phy.fec.utils import int_mod_2
from sionna.phy.utils import hard_decisions

__all__ = ["OSDecoder"]


class OSDecoder(Block):
    # pylint: disable=line-too-long
    r"""Ordered statistics decoding (OSD) for binary, linear block codes.

    This block implements the OSD algorithm as proposed in :cite:p:`Fossorier` and,
    thereby, approximates maximum likelihood decoding for a sufficiently large
    order :math:`t`. The algorithm works for arbitrary linear block codes, but
    has a high computational complexity for long codes.

    The algorithm consists of the following steps:

        1. Sort LLRs according to their reliability and apply the same column
        permutation to the generator matrix.

        2. Bring the permuted generator matrix into its systematic form
        (so-called *most-reliable basis*).

        3. Hard-decide and re-encode the :math:`k` most reliable bits and
        discard the remaining :math:`n-k` received positions.

        4. Generate all possible error patterns up to :math:`t` errors in the
        :math:`k` most reliable positions find the most likely codeword within
        these candidates.

    This implementation of the OSD algorithm uses the LLR-based distance metric
    from :cite:p:`Stimming_LLR` which simplifies the handling of higher-order
    modulation schemes.

    :param enc_mat: Binary generator matrix of shape `[k, n]`. If ``is_pcm`` is
        `True`, ``enc_mat`` is interpreted as parity-check matrix of shape
        `[n-k, n]`.
    :param t: Order of the OSD algorithm.
    :param is_pcm: If `True`, ``enc_mat`` is interpreted as parity-check matrix.
    :param encoder: Sionna block that implements a FEC encoder.
        If not `None`, ``enc_mat`` will be ignored and the code as specified by
        the encoder is used to initialize OSD.
    :param precision: Precision used for internal calculations and outputs.
        If `None`, :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for computation (e.g., 'cpu', 'cuda:0').
        If `None`, :attr:`~sionna.phy.config.Config.device` is used.

    :input llr_ch: [..., n], `torch.float`.
        Tensor containing the channel logits/llr values.

    :output c_hat: [..., n], `torch.float`.
        Tensor of same shape as ``llr_ch`` containing binary hard-decisions
        of all codeword bits.

    .. rubric:: Notes

    OS decoding is of high complexity and is only feasible for small values of
    :math:`t` as :math:`{n \choose t}` patterns must be evaluated. The
    advantage of OSD is that it works for arbitrary linear block codes and
    provides an estimate of the expected ML performance for sufficiently large
    :math:`t`. However, for some code families, more efficient decoding
    algorithms with close to ML performance exist which can exploit certain
    code specific properties. Examples of such decoders are the
    :class:`~sionna.phy.fec.conv.ViterbiDecoder` algorithm for convolutional codes
    or the :class:`~sionna.phy.fec.polar.decoding.PolarSCLDecoder` for Polar codes
    (for a sufficiently large list size).

    It is recommended to run the decoder with ``torch.compile()`` as it
    significantly reduces the memory complexity (typically 4-5x reduction)
    and improves execution speed (typically 7x or more). Without compilation,
    the decoder materializes large intermediate tensors of shape
    ``[batch_size, num_patterns, n]`` where ``num_patterns`` can be very
    large for higher values of ``t``.

    .. rubric:: Examples


    .. code-block:: python

        import torch
        from sionna.phy.fec.utils import load_parity_check_examples
        from sionna.phy.fec.linear import LinearEncoder, OSDecoder

        # Load (7,4) Hamming code
        pcm, k, n, _ = load_parity_check_examples(0)
        encoder = LinearEncoder(pcm, is_pcm=True)
        decoder = OSDecoder(encoder=encoder, t=2)

        # Generate random codeword and add noise
        u = torch.randint(0, 2, (10, k), dtype=torch.float32)
        c = encoder(u)
        llr_ch = 2.0 * (2.0 * c - 1.0)  # Perfect LLRs
        c_hat = decoder(llr_ch)
        print(torch.equal(c, c_hat))
        # True
    """

    def __init__(
        self,
        enc_mat: Optional[np.ndarray] = None,
        t: int = 0,
        is_pcm: bool = False,
        encoder: Optional[Block] = None,
        precision: Optional[str] = None,
        device: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(precision=precision, device=device, **kwargs)

        if not isinstance(is_pcm, bool):
            raise TypeError("is_pcm must be bool.")

        self._llr_max = 100.0  # internal clipping value for llrs

        if enc_mat is not None:
            # Check that gm is binary
            if isinstance(enc_mat, np.ndarray):
                if not np.array_equal(enc_mat, enc_mat.astype(bool)):
                    raise ValueError("enc_mat must be binary.")
            elif isinstance(enc_mat, sp.sparse.csr_matrix):
                if not np.array_equal(enc_mat.data, enc_mat.data.astype(bool)):
                    raise ValueError("enc_mat must be binary.")
            elif isinstance(enc_mat, sp.sparse.csc_matrix):
                if not np.array_equal(enc_mat.data, enc_mat.data.astype(bool)):
                    raise ValueError("enc_mat must be binary.")
            else:
                raise TypeError("Unsupported dtype of enc_mat.")

        if int(t) != t:
            raise TypeError("t must be int.")
        self._t = int(t)

        if encoder is not None:
            # Test that encoder is already initialized (relevant for conv codes)
            if encoder.k is None:
                raise ValueError(
                    "It seems as if the encoder is not "
                    "initialized or has no attribute k."
                )
            # Encode identity matrix to get k basis vectors of the code
            u = torch.eye(encoder.k, dtype=self.dtype, device=self.device)
            u = u.unsqueeze(0)
            # Encode and remove batch_dim
            gm = encoder(u).squeeze(0).to(self.dtype)
            self._gm = gm
        else:
            if enc_mat is None:
                raise ValueError("enc_mat cannot be None if no encoder is provided.")
            if is_pcm:
                gm = pcm2gm(enc_mat)
            else:
                # Check if gm is of full rank (raise error otherwise)
                make_systematic(enc_mat)
                gm = enc_mat
            # Register as buffer for CUDAGraph compatibility
            self.register_buffer("_gm", torch.tensor(gm, dtype=self.dtype, device=self.device))

        self._k = self._gm.shape[0]
        self._n = self._gm.shape[1]

        # Init error patterns
        num_patterns = self._num_error_patterns(self._n, self._t)

        # Storage/computational complexity scales with n
        num_symbols = num_patterns * self._n
        if num_symbols > 1e9:  # empirically found to be a good trade-off
            warnings.warn(
                f"Required memory complexity is large for the "
                f"given code parameters and t={t}. Please consider small "
                f"batch-sizes to keep the inference complexity small and "
                f"activate torch.compile() if possible.",
                UserWarning,
                stacklevel=2,
            )
        if num_symbols > 1e11:  # empirically found to be a good trade-off
            raise ResourceWarning(
                "Due to its high complexity, OSD is not "
                "feasible for the selected parameters. "
                "Please consider using a smaller value for t."
            )

        # Pre-compute all error patterns
        self._err_patterns = []
        for t_i in range(1, t + 1):
            self._err_patterns.append(self._gen_error_patterns(self._k, t_i))

    @property
    def gm(self) -> torch.Tensor:
        """Generator matrix of the code."""
        return self._gm

    @property
    def n(self) -> int:
        """Codeword length."""
        return self._n

    @property
    def k(self) -> int:
        """Number of information bits per codeword."""
        return self._k

    @property
    def t(self) -> int:
        """Order of the OSD algorithm."""
        return self._t

    def _num_error_patterns(self, n: int, t: int) -> int:
        r"""Returns number of possible error patterns for t errors in n
        positions, i.e., calculates :math:`{n \choose t}`.

        :param n: Length of vector.
        :param t: Number of errors.

        :output num_patterns: Number of error patterns.
        """
        return sp.special.comb(n, t, exact=True, repetition=False)

    def _gen_error_patterns(self, n: int, t: int) -> torch.Tensor:
        r"""Returns tensor of all possible error patterns for t errors in n
        positions.

        :param n: Length of vector.
        :param t: Number of errors.

        :output err_patterns: Tensor of shape [`num_patterns`, `t`] where
            `num_patterns` = :math:`{n \choose t}`, containing the `t` error
            indices.
        """
        err_patterns = list(itertools.combinations(range(n), t))
        return torch.tensor(err_patterns, dtype=torch.long, device=self.device)

    def _get_dist(self, llr: torch.Tensor, c_hat: torch.Tensor) -> torch.Tensor:
        """Distance function used for ML candidate selection.

        Currently, the distance metric from Polar decoding :cite:p:`Stimming_LLR`
        literature is implemented.

        :param llr: Received llrs of the channel observations of shape [bs, n].
        :param c_hat: Candidate codewords for which the distance to ``llr``
            shall be evaluated of shape [bs, num_cand, n].

        :output d: Distance between ``llr`` and ``c_hat`` for each of the
            `num_cand` codeword candidates of shape [bs, num_cand].

        Reference
        ---------
        [Stimming_LLR] Alexios Balatsoukas-Stimming, Mani Bastani Parizi,
        Andreas Burg, "LLR-Based Successive Cancellation List Decoding
        of Polar Codes." IEEE Trans Signal Processing, 2015.
        """
        # Broadcast llr to all codeword candidates
        llr = llr.unsqueeze(1)
        llr_sign = llr * (-2.0 * c_hat + 1.0)  # apply BPSK mapping

        d = torch.log1p(torch.exp(llr_sign))
        return d.mean(dim=2)

    def _find_min_dist(
        self,
        llr_ch: torch.Tensor,
        ep: torch.Tensor,
        gm_mrb: torch.Tensor,
        c: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""Find error pattern which leads to minimum distance.

        Uses chunked processing when memory requirements exceed threshold to
        prevent OOM errors for large numbers of error patterns.

        :param llr_ch: Channel observations as llrs after mrb sorting of shape
            [bs, n].
        :param ep: Tensor of shape [`num_patterns`, `t`] where
            `num_patterns` = :math:`{n \choose t}`, containing the `t` error
            indices.
        :param gm_mrb: Most reliable basis for each batch example of shape
            [bs, k, n].
        :param c: Most reliable base codeword of shape [bs, n].

        :output d_best: Distance of shape [bs] for the most likely codeword
            after testing all ``ep`` error patterns.

        :output c_hat: Codeword of shape [bs, n] for the most likely
            codeword after testing all ``ep`` error patterns.
        """
        num_patterns = ep.shape[0]
        t_val = ep.shape[1]
        bs = llr_ch.shape[0]

        # Estimate memory for full computation: e tensor [bs, num_patterns, t, n]
        bytes_per_element = 4 if llr_ch.dtype == torch.float32 else 8
        estimated_memory = bs * num_patterns * t_val * self._n * bytes_per_element

        # Use chunking if estimated memory exceeds 1 GB
        memory_threshold = 1 * 1024 * 1024 * 1024  # 1 GB

        if estimated_memory <= memory_threshold:
            # Small enough - process all at once (best for torch.compile)
            e = gm_mrb[:, ep, :]  # [bs, num_patterns, t, n]
            e = e.sum(dim=2)  # [bs, num_patterns, n]
            e = e + c.unsqueeze(1)
            c_cand = int_mod_2(e)  # [bs, num_patterns, n]
            d = self._get_dist(llr_ch, c_cand)  # [bs, num_patterns]
            idx = d.argmin(dim=1)  # [bs]
            batch_range = torch.arange(bs, device=self.device)
            c_hat = c_cand[batch_range, idx]  # [bs, n]
            d_best = d[batch_range, idx]  # [bs]
            return d_best, c_hat

        # Large case - process in chunks to avoid OOM
        # Target ~256 MB per chunk for the e tensor
        target_chunk_memory = 256 * 1024 * 1024
        chunk_size = max(1, target_chunk_memory // (bs * t_val * self._n * bytes_per_element))
        chunk_size = min(chunk_size, num_patterns)

        # Initialize best distance and codeword
        d_best = torch.full((bs,), float("inf"), dtype=llr_ch.dtype, device=self.device)
        c_hat_best = c.clone()

        # Process error patterns in chunks
        for start in range(0, num_patterns, chunk_size):
            end = min(start + chunk_size, num_patterns)
            ep_chunk = ep[start:end]

            # Generate test candidates for this chunk
            e = gm_mrb[:, ep_chunk, :]  # [bs, chunk_size, t, n]
            e = e.sum(dim=2)  # [bs, chunk_size, n]
            e = e + c.unsqueeze(1)
            c_cand = int_mod_2(e)  # [bs, chunk_size, n]

            # Calculate distance for each candidate
            d = self._get_dist(llr_ch, c_cand)  # [bs, chunk_size]

            # Find best in this chunk
            d_min_chunk, idx_chunk = d.min(dim=1)  # [bs]

            # Update best if this chunk has better candidates
            improved = d_min_chunk < d_best
            batch_range = torch.arange(bs, device=self.device)
            c_hat_chunk = c_cand[batch_range, idx_chunk]  # [bs, n]
            c_hat_best = torch.where(improved.unsqueeze(1), c_hat_chunk, c_hat_best)
            d_best = torch.where(improved, d_min_chunk, d_best)

        return d_best, c_hat_best

    def _find_mrb(
        self, gm: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Find most reliable basis for all generator matrices in batch.

        :param gm: Generator matrix for each batch example of shape [bs, k, n].

        :output gm_mrb: Tensor of shape [bs, k, n] containing the most
            reliable basis in systematic form for each batch example.

        :output idx_sort: Tensor of shape [bs, n] containing the indices of
            column permutations applied during mrb calculation.
        """
        bs = gm.shape[0]
        k = self._k
        n = self._n

        # Storage for pivot positions
        idx_pivot = torch.zeros((bs, k), dtype=torch.long, device=self.device)

        # Bring gm in systematic form (by so-called pivot method)
        gm = gm.clone()
        for idx_c in range(k):
            # Find pivot (i.e., first pos with index 1)
            idx_p = gm[:, idx_c, :].argmax(dim=-1)  # [bs]

            # Store pivot position
            idx_pivot[:, idx_c] = idx_p

            # And eliminate the column in all other rows
            # Get the column at pivot position for each batch
            batch_range = torch.arange(bs, device=self.device)
            r = gm[batch_range, :, idx_p]  # [bs, k]

            # Ignore idx_c row itself by setting to zero
            r[:, idx_c] = 0

            # Mask is zero at all rows where pivot position of this row is zero
            mask = r.unsqueeze(-1)  # [bs, k, 1]
            gm_off = gm[:, idx_c, :].unsqueeze(1)  # [bs, 1, n]

            # Update all rows in parallel (binary operations)
            gm = int_mod_2(gm + mask * gm_off)

        # Find non-pivot positions (i.e., all indices that are not part of idx_pivot)
        # Add large offset to pivot indices and sorting gives the indices of interest
        idx_range = torch.arange(n, device=self.device).unsqueeze(0).expand(bs, -1)

        # Large value to be added to irrelevant indices
        updates = n * torch.ones((bs, k), dtype=torch.long, device=self.device)

        # Create scatter indices
        batch_idx = torch.arange(bs, device=self.device).unsqueeze(1).expand(-1, k)

        # Add large value to pivot positions
        idx = idx_range.clone()
        idx[batch_idx, idx_pivot] = idx[batch_idx, idx_pivot] + updates

        # Sort and slice first n-k indices (equals parity positions)
        sorted_idx = idx.argsort(dim=1)
        idx_parity = sorted_idx[:, : n - k]  # [bs, n-k]

        idx_sort = torch.cat([idx_pivot, idx_parity], dim=1)  # [bs, n]

        # Permute gm according to indices idx_sort
        batch_idx = torch.arange(bs, device=self.device).view(-1, 1, 1)
        row_idx = torch.arange(k, device=self.device).view(1, -1, 1)
        idx_sort_expanded = idx_sort.unsqueeze(1).expand(-1, k, -1)
        gm = gm[batch_idx, row_idx, idx_sort_expanded]

        return gm, idx_sort

    def build(self, input_shape: tuple) -> None:
        """Check for valid input shapes."""
        if input_shape[-1] != self._n:
            raise ValueError(f"Last dimension must be of size n={self._n}.")

    def call(self, llr_ch: torch.Tensor, /) -> torch.Tensor:
        r"""Applies ordered statistic decoding to inputs.

        Remark: the decoder is implemented with llr definition
        llr = p(x=1)/p(x=0).

        :param llr_ch: Channel LLRs of shape [..., n].

        :output c_hat: Hard decisions of shape [..., n].
        """
        # Validate input shape
        if llr_ch.shape[-1] != self._n:
            raise ValueError(f"Last dimension must be of size n={self._n}.")

        # Flatten batch-dim
        input_shape = llr_ch.shape
        llr_ch = llr_ch.reshape(-1, self._n)
        bs = llr_ch.shape[0]

        # Clip inputs
        llr_ch = llr_ch.clamp(-self._llr_max, self._llr_max)

        # Step 1: sort LLRs
        idx_sort = llr_ch.abs().argsort(dim=-1, descending=True)

        # Permute gm per batch sample individually
        gm = self._gm.unsqueeze(0).expand(bs, -1, -1)

        # Gather columns according to idx_sort
        batch_idx = torch.arange(bs, device=self.device).view(-1, 1, 1)
        row_idx = torch.arange(self._k, device=self.device).view(1, -1, 1)
        idx_sort_expanded = idx_sort.unsqueeze(1).expand(-1, self._k, -1)
        gm_sort = gm[batch_idx, row_idx, idx_sort_expanded]

        # Step 2: Find most reliable basis (MRB)
        gm_mrb, idx_mrb = self._find_mrb(gm_sort)

        # Apply corresponding mrb permutations
        batch_range = torch.arange(bs, device=self.device).unsqueeze(1)
        idx_sort = idx_sort.gather(1, idx_mrb)
        llr_sort = llr_ch.gather(1, idx_sort)

        # Find inverse permutation for final output
        idx_sort_inv = idx_sort.argsort(dim=1)

        # Hard-decide k most reliable positions and encode
        u_hd = hard_decisions(llr_sort[:, : self._k])
        u_hd = u_hd.unsqueeze(1)  # [bs, 1, k]
        c = torch.matmul(u_hd, gm_mrb).squeeze(1)  # [bs, n]
        c = int_mod_2(c)

        # And search for most likely pattern
        # _get_dist expects a list of candidates, thus expand_dims to [bs, 1, n]
        d_best = self._get_dist(llr_sort, c.unsqueeze(1))
        d_best = d_best.squeeze(1)  # [bs]
        c_hat_best = c

        # Known in advance - can be unrolled
        for ep in self._err_patterns:
            # Compute distance for all candidate codewords
            d, c_hat = self._find_min_dist(llr_sort, ep, gm_mrb, c)

            # Select most likely candidate
            mask = (d < d_best).unsqueeze(1)  # [bs, 1]
            c_hat_best = torch.where(mask, c_hat, c_hat_best)
            d_best = torch.where(d < d_best, d, d_best)

        # Undo permutations for final codeword
        c_hat_best = c_hat_best.gather(1, idx_sort_inv)

        # Restore input shape
        c_hat = c_hat_best.reshape(input_shape)

        return c_hat
