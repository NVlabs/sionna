#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Classes and functions related to MIMO channel detection."""

import warnings
from typing import Optional, Callable, Union
import numpy as np
import torch

from sionna.phy.block import Block
from sionna.phy.config import config, dtypes, Precision
from sionna.phy.utils import (
    expand_to_rank,
    flatten_last_dims,
    flatten_dims,
    split_dim,
    insert_dims,
    hard_decisions,
)
from sionna.phy.mapping import (
    Constellation,
    SymbolLogits2LLRs,
    LLRs2SymbolLogits,
    PAM2QAM,
    Demapper,
    SymbolDemapper,
    SymbolInds2Bits,
    SymbolLogits2Moments,
)
from sionna.phy.mimo.utils import (
    complex2real_channel,
    whiten_channel,
    List2LLR,
    List2LLRSimple,
    complex2real_matrix,
    complex2real_vector,
    real2complex_vector,
)
from sionna.phy.mimo.equalization import lmmse_equalizer, zf_equalizer, mf_equalizer

__all__ = [
    "LinearDetector",
    "MaximumLikelihoodDetector",
    "KBestDetector",
    "EPDetector",
    "MMSEPICDetector",
]


class LinearDetector(Block):
    r"""
    Convenience class that combines an equalizer,
    such as :func:`~sionna.phy.mimo.lmmse_equalizer`, and a
    :class:`~sionna.phy.mapping.Demapper`.

    :param equalizer: The equalizer to be used. Either one of the existing
        equalizers :func:`~sionna.phy.mimo.lmmse_equalizer`,
        :func:`~sionna.phy.mimo.zf_equalizer`, or
        :func:`~sionna.phy.mimo.mf_equalizer` can be used (specified as
        ``"lmmse"``, ``"zf"``, or ``"mf"``), or a custom equalizer callable
        provided that has the same input/output specification.
    :param output: Type of output, either ``"bit"`` for LLRs on bits or
        ``"symbol"`` for logits on constellation symbols
    :param demapping_method: Demapping method, either ``"app"`` or ``"maxlog"``
    :param constellation_type: Constellation type, one of ``"qam"``,
        ``"pam"``, or ``"custom"``. For ``"custom"``, an instance of
        :class:`~sionna.phy.mapping.Constellation` must be provided.
    :param num_bits_per_symbol: Number of bits per constellation symbol,
        e.g., 4 for QAM16.
        Only required for ``constellation_type`` in [``"qam"``, ``"pam"``].
    :param constellation: An instance of
        :class:`~sionna.phy.mapping.Constellation` or `None`. If `None`,
        ``constellation_type`` and ``num_bits_per_symbol`` must be provided.
    :param hard_out: If `True`, the detector computes hard-decided bit values
        or constellation point indices instead of soft-values.
        Defaults to `False`.
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`, :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for computations

    :input y: [...,M], `torch.complex`. Received signals.
    :input h: [...,M,num_streams], `torch.complex`. Channel matrices.
    :input s: [...,M,M], `torch.complex`. Noise covariance matrices.

    One of:

    :output llr: [..., num_streams, num_bits_per_symbol], `torch.float`.
        LLRs or hard-decisions for every bit of every stream,
        if ``output`` equals ``"bit"``.
    :output logits: [..., num_streams, num_points], `torch.float` or [..., num_streams], `torch.int32`.
        Logits or hard-decisions for constellation symbols for every stream,
        if ``output`` equals ``"symbol"``.
        Hard-decisions correspond to the symbol indices.

    .. rubric:: Examples

    .. code-block:: python

        detector = LinearDetector(
            equalizer="lmmse",
            output="bit",
            demapping_method="app",
            constellation_type="qam",
            num_bits_per_symbol=4
        )
        llr = detector(y, h, s)
    """

    def __init__(
        self,
        equalizer: Union[str, Callable],
        output: str,
        demapping_method: str,
        constellation_type: Optional[str] = None,
        num_bits_per_symbol: Optional[int] = None,
        constellation: Optional[Constellation] = None,
        hard_out: bool = False,
        precision: Optional[Precision] = None,
        device: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(precision=precision, device=device, **kwargs)
        self._output = output
        self._hard_out = hard_out

        # Determine the equalizer to use
        if isinstance(equalizer, str):
            if equalizer not in ("lmmse", "zf", "mf"):
                raise ValueError(
                    "equalizer must be one of: 'lmmse', 'zf', 'mf'"
                )
            if equalizer == "lmmse":
                self._equalizer = lmmse_equalizer
            elif equalizer == "zf":
                self._equalizer = zf_equalizer
            else:
                self._equalizer = mf_equalizer
        else:
            self._equalizer = equalizer

        if output not in ("bit", "symbol"):
            raise ValueError("output must be one of: 'bit', 'symbol'")
        if demapping_method not in ("app", "maxlog"):
            raise ValueError(
                "demapping_method must be one of: 'app', 'maxlog'"
            )

        self._constellation = Constellation.check_or_create(
            constellation_type=constellation_type,
            num_bits_per_symbol=num_bits_per_symbol,
            constellation=constellation,
            precision=precision,
            device=device,
        )

        # Determine the demapper to use
        if output == "bit":
            self._demapper = Demapper(
                demapping_method,
                constellation=self._constellation,
                hard_out=hard_out,
                precision=precision,
                device=device,
            )
        else:
            self._demapper = SymbolDemapper(
                constellation=self._constellation,
                hard_out=hard_out,
                precision=precision,
                device=device,
            )

    def call(self, y: torch.Tensor, h: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        x_hat, no_eff = self._equalizer(y, h, s, precision=self.precision)
        z = self._demapper(x_hat, no_eff)

        # Reshape to the expected output shape
        num_streams = h.shape[-1]
        if self._output == "bit":
            num_bits_per_symbol = self._constellation.num_bits_per_symbol
            z = split_dim(z, [num_streams, num_bits_per_symbol], z.dim() - 1)

        return z


class MaximumLikelihoodDetector(Block):
    r"""
    MIMO maximum-likelihood (ML) detector.

    This block implements MIMO maximum-likelihood (ML) detection assuming the
    following channel model:

    .. math::
        \mathbf{y} = \mathbf{H}\mathbf{x} + \mathbf{n}

    where :math:`\mathbf{y}\in\mathbb{C}^M` is the received signal vector,
    :math:`\mathbf{x}\in\mathcal{C}^K` is the vector of transmitted symbols which
    are uniformly and independently drawn from the constellation
    :math:`\mathcal{C}`,
    :math:`\mathbf{H}\in\mathbb{C}^{M\times K}` is the known channel matrix,
    and :math:`\mathbf{n}\in\mathbb{C}^M` is a complex Gaussian noise vector.
    It is assumed that
    :math:`\mathbb{E}\left[\mathbf{n}\right]=\mathbf{0}` and
    :math:`\mathbb{E}\left[\mathbf{n}\mathbf{n}^{\mathsf{H}}\right]=\mathbf{S}`,
    where :math:`\mathbf{S}` has full rank.
    Optionally, prior information of the transmitted signal
    :math:`\mathbf{x}` can be provided, either as LLRs on the bits mapped
    onto :math:`\mathbf{x}` or as logits on the individual constellation
    points forming :math:`\mathbf{x}`.

    Prior to demapping, the received signal is whitened:

    .. math::
        \tilde{\mathbf{y}} &= \mathbf{S}^{-\frac{1}{2}}\mathbf{y}\\
        &=  \mathbf{S}^{-\frac{1}{2}}\mathbf{H}\mathbf{x} + \mathbf{S}^{-\frac{1}{2}}\mathbf{n}\\
        &= \tilde{\mathbf{H}}\mathbf{x} + \tilde{\mathbf{n}}

    The block can compute ML detection of symbols or bits with either
    soft- or hard-decisions. Note that decisions are computed symbol-/bit-wise
    and not jointly for the entire vector :math:`\textbf{x}` (or the
    underlying vector of bits).

    **ML detection of bits:**

    Soft-decisions on bits are called log-likelihood ratios (LLR).
    With the "app" demapping method, the LLR for the :math:`i\text{th}` bit
    of the :math:`k\text{th}` user is then computed according to

    .. math::
        \begin{aligned}
            LLR(k,i)&= \ln\left(\frac{\Pr\left(b_{k,i}=1\lvert \mathbf{y},\mathbf{H}\right)}{\Pr\left(b_{k,i}=0\lvert \mathbf{y},\mathbf{H}\right)}\right)\\
                    &=\ln\left(\frac{
                    \sum_{\mathbf{x}\in\mathcal{C}_{k,i,1}} \exp\left(
                        -\left\lVert\tilde{\mathbf{y}}-\tilde{\mathbf{H}}\mathbf{x}\right\rVert^2
                        \right) \Pr\left( \mathbf{x} \right)
                    }{
                    \sum_{\mathbf{x}\in\mathcal{C}_{k,i,0}} \exp\left(
                        -\left\lVert\tilde{\mathbf{y}}-\tilde{\mathbf{H}}\mathbf{x}\right\rVert^2
                        \right) \Pr\left( \mathbf{x} \right)
                    }\right)
        \end{aligned}

    where :math:`\mathcal{C}_{k,i,1}` and :math:`\mathcal{C}_{k,i,0}` are the
    sets of vectors of constellation points for which the :math:`i\text{th}` bit
    of the :math:`k\text{th}` user is equal to 1 and 0, respectively.
    :math:`\Pr\left( \mathbf{x} \right)` is the prior distribution of the
    vector of constellation points :math:`\mathbf{x}`. Assuming that the
    constellation points and bit levels are independent, it is computed from
    the prior of the bits according to

    .. math::
        \Pr\left( \mathbf{x} \right) = \prod_{k=1}^K \prod_{i=1}^{I} \sigma \left( LLR_p(k,i) \right)

    where :math:`LLR_p(k,i)` is the prior knowledge of the
    :math:`i\text{th}` bit of the :math:`k\text{th}` user given as an LLR
    and which is set to :math:`0` if no prior knowledge is assumed to be
    available, and :math:`\sigma\left(\cdot\right)` is the sigmoid function.
    The definition of the LLR has been chosen such that it is equivalent with
    that of logit. This is different from many textbooks in communications,
    where the LLR is defined as
    :math:`LLR(k,i) = \ln\left(\frac{\Pr\left(b_{k,i}=0\lvert \mathbf{y},\mathbf{H}\right)}{\Pr\left(b_{k,i}=1\lvert \mathbf{y},\mathbf{H}\right)}\right)`.

    With the "maxlog" demapping method, the LLR for the :math:`i\text{th}` bit
    of the :math:`k\text{th}` user is approximated like

    .. math::
        \begin{aligned}
            LLR(k,i) \approx&\ln\left(\frac{
                \max_{\mathbf{x}\in\mathcal{C}_{k,i,1}} \left( \exp\left(
                    -\left\lVert\tilde{\mathbf{y}}-\tilde{\mathbf{H}}\mathbf{x}\right\rVert^2
                    \right) \Pr\left( \mathbf{x} \right) \right)
                }{
                \max_{\mathbf{x}\in\mathcal{C}_{k,i,0}} \left( \exp\left(
                    -\left\lVert\tilde{\mathbf{y}}-\tilde{\mathbf{H}}\mathbf{x}\right\rVert^2
                    \right) \Pr\left( \mathbf{x} \right) \right)
                }\right)\\
                = &\min_{\mathbf{x}\in\mathcal{C}_{k,i,0}} \left( \left\lVert\tilde{\mathbf{y}}-\tilde{\mathbf{H}}\mathbf{x}\right\rVert^2 - \ln \left(\Pr\left( \mathbf{x} \right) \right) \right) -
                    \min_{\mathbf{x}\in\mathcal{C}_{k,i,1}} \left( \left\lVert\tilde{\mathbf{y}}-\tilde{\mathbf{H}}\mathbf{x}\right\rVert^2 - \ln \left( \Pr\left( \mathbf{x} \right) \right) \right).
            \end{aligned}

    **ML detection of symbols:**

    Soft-decisions on symbols are called logits (i.e., unnormalized
    log-probability).

    With the "app" demapping method, the logit for the
    constellation point :math:`c \in \mathcal{C}` of the
    :math:`k\text{th}` user is computed according to

    .. math::
        \begin{aligned}
            \text{logit}(k,c) &= \ln\left(\sum_{\mathbf{x} : x_k = c} \exp\left(
                        -\left\lVert\tilde{\mathbf{y}}-\tilde{\mathbf{H}}\mathbf{x}\right\rVert^2
                        \right)\Pr\left( \mathbf{x} \right)\right).
        \end{aligned}

    With the "maxlog" demapping method, the logit for the constellation
    point :math:`c \in \mathcal{C}` of the :math:`k\text{th}` user is
    approximated like

    .. math::
        \text{logit}(k,c) \approx \max_{\mathbf{x} : x_k = c} \left(
                -\left\lVert\tilde{\mathbf{y}}-\tilde{\mathbf{H}}\mathbf{x}\right\rVert^2 + \ln \left( \Pr\left( \mathbf{x} \right) \right)
                \right).

    When hard decisions are requested, this block returns for the
    :math:`k` th stream

    .. math::
        \hat{c}_k = \underset{c \in \mathcal{C}}{\text{argmax}} \left( \sum_{\mathbf{x} : x_k = c} \exp\left(
                        -\left\lVert\tilde{\mathbf{y}}-\tilde{\mathbf{H}}\mathbf{x}\right\rVert^2
                        \right)\Pr\left( \mathbf{x} \right) \right)

    where :math:`\mathcal{C}` is the set of constellation points.

    :param output: Type of output, either ``"bit"`` for LLRs on bits or
        ``"symbol"`` for logits on constellation symbols
    :param demapping_method: Demapping method, either ``"app"`` or
        ``"maxlog"``
    :param num_streams: Number of transmitted streams
    :param constellation_type: Constellation type, one of ``"qam"``,
        ``"pam"``, or ``"custom"``. For ``"custom"``, an instance of
        :class:`~sionna.phy.mapping.Constellation` must be provided.
    :param num_bits_per_symbol: Number of bits per constellation symbol,
        e.g., 4 for QAM16.
        Only required for ``constellation_type`` in [``"qam"``, ``"pam"``].
    :param constellation: An instance of
        :class:`~sionna.phy.mapping.Constellation` or `None`. If `None`,
        ``constellation_type`` and ``num_bits_per_symbol`` must be provided.
    :param hard_out: If `True`, the detector computes hard-decided bit values
        or constellation point indices instead of soft-values.
        Defaults to `False`.
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`, :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for computations

    :input y: [...,M], `torch.complex`. Received signals.
    :input h: [...,M,num_streams], `torch.complex`. Channel matrices.
    :input s: [...,M,M], `torch.complex`. Noise covariance matrices.
    :input prior: `None` (default) | [...,num_streams,num_bits_per_symbol] or [...,num_streams,num_points], `torch.float`.
        Prior of the transmitted signals.
        If ``output`` equals ``"bit"``, then LLRs of the transmitted bits are expected.
        If ``output`` equals ``"symbol"``, then logits of the transmitted constellation points are expected.

    One of:

    :output llr: [..., num_streams, num_bits_per_symbol], `torch.float`.
        LLRs or hard-decisions for every bit of every stream,
        if ``output`` equals ``"bit"``.
    :output logits: [..., num_streams, num_points], `torch.float` or [..., num_streams], `torch.int32`.
        Logits or hard-decisions for constellation symbols for every stream,
        if ``output`` equals ``"symbol"``.
        Hard-decisions correspond to the symbol indices.

    .. rubric:: Examples

    .. code-block:: python

        detector = MaximumLikelihoodDetector(
            output="bit",
            demapping_method="maxlog",
            num_streams=2,
            constellation_type="qam",
            num_bits_per_symbol=4
        )
        llr = detector(y, h, s)
    """

    def __init__(
        self,
        output: str,
        demapping_method: str,
        num_streams: int,
        constellation_type: Optional[str] = None,
        num_bits_per_symbol: Optional[int] = None,
        constellation: Optional[Constellation] = None,
        hard_out: bool = False,
        precision: Optional[Precision] = None,
        device: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(precision=precision, device=device, **kwargs)

        if output not in ("bit", "symbol"):
            raise ValueError("output must be one of: 'bit', 'symbol'")
        if demapping_method not in ("app", "maxlog"):
            raise ValueError(
                "demapping_method must be one of: 'app', 'maxlog'"
            )

        self._output = output
        self._demapping_method = demapping_method
        self._hard_out = hard_out

        # Determine the reduce function for LLR computation
        if self._demapping_method == "app":
            self._reduce = torch.logsumexp
        else:
            self._reduce = lambda x, dim: x.max(dim=dim).values

        # Create constellation object
        self._constellation = Constellation.check_or_create(
            constellation_type=constellation_type,
            num_bits_per_symbol=num_bits_per_symbol,
            constellation=constellation,
            precision=precision,
            device=device,
        )

        # Build lookup tables
        vecs, vecs_ind, c = self._build_vecs(num_streams)
        self._vecs = torch.as_tensor(vecs, dtype=self.cdtype, device=self.device)
        self._vecs_ind = torch.as_tensor(
            vecs_ind, dtype=torch.int64, device=self.device
        )
        self._c = torch.as_tensor(c, dtype=torch.int64, device=self.device)

        if output == "bit":
            num_bits = self._constellation.num_bits_per_symbol
            self._logits2llr = SymbolLogits2LLRs(
                method=demapping_method,
                num_bits_per_symbol=num_bits,
                hard_out=hard_out,
                precision=precision,
                device=device,
                **kwargs,
            )
            self._llrs2logits = LLRs2SymbolLogits(
                num_bits_per_symbol=num_bits,
                hard_out=False,
                precision=precision,
                device=device,
                **kwargs,
            )

    @property
    def constellation(self) -> Constellation:
        """The constellation used by the detector."""
        return self._constellation

    def _build_vecs(self, num_streams: int):
        """Build list of all possible transmitted vectors and symbol indices."""
        points = self._constellation().cpu().numpy()
        num_points = len(points)

        def _build_vecs_(n):
            if n == 1:
                vecs = np.expand_dims(points, axis=1)
                vecs_ind = np.expand_dims(np.arange(num_points), axis=1)
            else:
                v, vi = _build_vecs_(n - 1)
                vecs = []
                vecs_ind = []
                for i, p in enumerate(points):
                    vecs.append(
                        np.concatenate([np.full([v.shape[0], 1], p), v], axis=1)
                    )
                    vecs_ind.append(
                        np.concatenate([np.full([v.shape[0], 1], i), vi], axis=1)
                    )
                vecs = np.concatenate(vecs, axis=0)
                vecs_ind = np.concatenate(vecs_ind, axis=0)
            return vecs, vecs_ind

        vecs, vecs_ind = _build_vecs_(num_streams)

        tx_ind = np.arange(num_streams)
        tx_ind = np.expand_dims(tx_ind, axis=0)
        tx_ind = np.tile(tx_ind, [vecs_ind.shape[0], 1])
        vecs_ind = np.stack([tx_ind, vecs_ind], axis=-1)

        # Compute symbol indices for every stream
        c = []
        for p in points:
            c_ = []
            for j in range(num_streams):
                c_.append(np.where(vecs[:, j] == p)[0])
            c_ = np.stack(c_, axis=-1)
            c.append(c_)
        c = np.stack(c, axis=-1)

        return vecs, vecs_ind, c

    def call(
        self,
        y: torch.Tensor,
        h: torch.Tensor,
        s: torch.Tensor,
        prior: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # If operating on bits, compute prior on symbols from the prior on bits
        if prior is not None and self._output == "bit":
            prior = self._llrs2logits(prior)

        # Whiten channel
        y, h = whiten_channel(y, h, s, return_s=False)

        # Add extra dims for broadcasting
        h = h.unsqueeze(-3)  # [..., 1, M, K]
        y = y.unsqueeze(-2)  # [..., 1, M]

        # Reshape list of all possible vectors
        vecs = self._vecs.unsqueeze(-1)  # [num_vecs, K, 1]
        vecs = expand_to_rank(vecs, h.dim(), 0)

        # Compute exponents
        diff = y - (h @ vecs).squeeze(-1)
        exponents = -(diff.abs() ** 2).sum(dim=-1)  # [..., num_vecs]

        # Add prior
        if prior is not None:
            prior = expand_to_rank(prior, exponents.dim(), axis=0)
            prior_rank = prior.dim()
            transpose_ind = [prior_rank - 2, prior_rank - 1] + list(
                range(prior_rank - 2)
            )
            prior = prior.permute(transpose_ind)
            # Gather prior values
            prior = prior[self._vecs_ind[..., 0], self._vecs_ind[..., 1]]
            transpose_ind = list(range(2, prior_rank)) + [0, 1]
            prior = prior.permute(transpose_ind)
            prior = prior.sum(dim=-1)
            exponents = exponents + prior

        # Gather exponents for all symbols
        exp = exponents.index_select(-1, self._c.flatten()).reshape(
            *exponents.shape[:-1], *self._c.shape
        )

        # Compute logits on constellation points
        logits = self._reduce(exp, dim=-3)

        if self._output == "bit":
            return self._logits2llr(logits)
        else:
            if self._hard_out:
                return logits.argmax(dim=-1).to(torch.int32)
            else:
                return logits


class KBestDetector(Block):
    r"""
    MIMO K-Best detector.

    This block implements K-Best MIMO detection as described
    in (Eq. 4-5) :cite:p:`FT2015`. It can either generate hard decisions (for symbols
    or bits) or compute LLRs.

    The algorithm operates in either the complex or real-valued domain.
    Although both options produce identical results, the former has the
    advantage that it can be applied to arbitrary non-QAM constellations. It
    also reduces the number of streams (or depth) by a factor of two.

    The way soft-outputs (i.e., LLRs) are computed is determined by the
    ``list2llr`` function. The default solution
    :class:`~sionna.phy.mimo.List2LLRSimple` assigns a predetermined
    value to all LLRs without counter-hypothesis.

    This block assumes the following channel model:

    .. math::
        \mathbf{y} = \mathbf{H}\mathbf{x} + \mathbf{n}

    where :math:`\mathbf{y}\in\mathbb{C}^M` is the received signal vector,
    :math:`\mathbf{x}\in\mathcal{C}^S` is the vector of transmitted symbols
    which are uniformly and independently drawn from the constellation
    :math:`\mathcal{C}`,
    :math:`\mathbf{H}\in\mathbb{C}^{M\times S}` is the known channel matrix,
    and :math:`\mathbf{n}\in\mathbb{C}^M` is a complex Gaussian noise vector.
    It is assumed that
    :math:`\mathbb{E}\left[\mathbf{n}\right]=\mathbf{0}` and
    :math:`\mathbb{E}\left[\mathbf{n}\mathbf{n}^{\mathsf{H}}\right]=\mathbf{S}`,
    where :math:`\mathbf{S}` has full rank.

    In a first optional step, the channel model is converted to its
    real-valued equivalent, see
    :func:`~sionna.phy.mimo.complex2real_channel`. We assume in the sequel
    the complex-valued representation. Then, the channel is whitened using
    :func:`~sionna.phy.mimo.whiten_channel`:

    .. math::
        \tilde{\mathbf{y}} &= \mathbf{S}^{-\frac{1}{2}}\mathbf{y}\\
        &=  \mathbf{S}^{-\frac{1}{2}}\mathbf{H}\mathbf{x} + \mathbf{S}^{-\frac{1}{2}}\mathbf{n}\\
        &= \tilde{\mathbf{H}}\mathbf{x} + \tilde{\mathbf{n}}.

    Next, the columns of :math:`\tilde{\mathbf{H}}` are sorted according
    to their norm in descending order. Then, the QR decomposition of the
    resulting channel matrix is computed:

    .. math::
        \tilde{\mathbf{H}} = \mathbf{Q}\mathbf{R}

    where :math:`\mathbf{Q}\in\mathbb{C}^{M\times S}` is unitary and
    :math:`\mathbf{R}\in\mathbb{C}^{S\times S}` is upper-triangular.
    The channel outputs are then pre-multiplied by
    :math:`\mathbf{Q}^{\mathsf{H}}`.
    This leads to the final channel model on which the K-Best detection
    algorithm operates:

    .. math::
        \bar{\mathbf{y}} = \mathbf{R}\bar{\mathbf{x}} + \bar{\mathbf{n}}

    where :math:`\bar{\mathbf{y}}\in\mathbb{C}^S`,
    :math:`\bar{\mathbf{x}}\in\mathbb{C}^S`, and
    :math:`\bar{\mathbf{n}}\in\mathbb{C}^S`
    with :math:`\mathbb{E}\left[\bar{\mathbf{n}}\right]=\mathbf{0}` and
    :math:`\mathbb{E}\left[\bar{\mathbf{n}}\bar{\mathbf{n}}^{\mathsf{H}}\right]=\mathbf{I}`.

    **LLR Computation**

    The K-Best algorithm produces :math:`K` candidate solutions
    :math:`\bar{\mathbf{x}}_k\in\mathcal{C}^S`
    and their associated distance metrics
    :math:`d_k=\lVert \bar{\mathbf{y}} - \mathbf{R}\bar{\mathbf{x}}_k \rVert^2`
    for :math:`k=1,\dots,K`. If the real-valued channel representation is
    used, the distance metrics are scaled by 0.5 to account for the reduced
    noise power in each complex dimension.
    A hard-decision is simply the candidate with the shortest distance.
    Various ways to compute LLRs from this list (and possibly
    additional side-information) are possible. The (sub-optimal) default
    solution is :class:`~sionna.phy.mimo.List2LLRSimple`. Custom solutions
    can be provided.

    :param output: Type of output, either ``"bit"`` for LLRs on bits or
        ``"symbol"`` for logits on constellation symbols
    :param num_streams: Number of transmitted streams
    :param k: Number of paths to keep. Cannot be larger than the
        number of constellation points to the power of the number of streams.
    :param constellation_type: Constellation type, one of ``"qam"``,
        ``"pam"``, or ``"custom"``. For ``"custom"``, an instance of
        :class:`~sionna.phy.mapping.Constellation` must be provided.
    :param num_bits_per_symbol: Number of bits per constellation symbol,
        e.g., 4 for QAM16.
        Only required for ``constellation_type`` in [``"qam"``, ``"pam"``].
    :param constellation: An instance of
        :class:`~sionna.phy.mapping.Constellation` or `None`. If `None`,
        ``constellation_type`` and ``num_bits_per_symbol`` must be provided.
    :param hard_out: If `True`, the detector computes hard-decided bit values
        or constellation point indices instead of soft-values.
        Defaults to `False`.
    :param use_real_rep: If `True`, the detector uses the real-valued
        equivalent representation of the channel. Note that this only works
        with a QAM constellation. Defaults to `False`.
    :param list2llr: The function to be used to compute LLRs from a list of
        candidate solutions. If `None`, the default solution
        :class:`~sionna.phy.mimo.List2LLRSimple` is used.
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`, :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for computations

    :input y: [...,M], `torch.complex`. Received signals.
    :input h: [...,M,num_streams], `torch.complex`. Channel matrices.
    :input s: [...,M,M], `torch.complex`. Noise covariance matrices.

    One of:

    :output llr: [..., num_streams, num_bits_per_symbol], `torch.float`.
        LLRs or hard-decisions for every bit of every stream,
        if ``output`` equals ``"bit"``.
    :output logits: [..., num_streams, num_points], `torch.float` or [..., num_streams], `torch.int32`.
        Logits or hard-decisions for constellation symbols for every stream,
        if ``output`` equals ``"symbol"``.
        Hard-decisions correspond to the symbol indices.

    .. rubric:: Examples

    .. code-block:: python

        detector = KBestDetector(
            output="bit",
            num_streams=2,
            k=16,
            constellation_type="qam",
            num_bits_per_symbol=4
        )
        llr = detector(y, h, s)
    """

    def __init__(
        self,
        output: str,
        num_streams: int,
        k: int,
        constellation_type: Optional[str] = None,
        num_bits_per_symbol: Optional[int] = None,
        constellation: Optional[Constellation] = None,
        hard_out: bool = False,
        use_real_rep: bool = False,
        list2llr: Optional[List2LLR] = None,
        precision: Optional[Precision] = None,
        device: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(precision=precision, device=device, **kwargs)

        if output not in ("bit", "symbol"):
            raise ValueError("output must be one of: 'bit', 'symbol'")

        err_msg = "You must provide either constellation or constellation_type and num_bits_per_symbol."
        if constellation is None:
            if constellation_type is None or num_bits_per_symbol is None:
                raise ValueError(err_msg)
        else:
            if constellation_type is not None or num_bits_per_symbol is not None:
                raise ValueError(err_msg)

        if constellation is not None:
            if constellation.precision != self.precision:
                raise ValueError("Constellation has wrong precision.")

        self._output = output
        self._hard_out = hard_out
        self._use_real_rep = use_real_rep

        if self._use_real_rep:
            err_msg = "Only QAM can be used for the real-valued representation"
            if constellation_type is not None:
                if constellation_type != "qam":
                    raise ValueError(err_msg)
            else:
                if constellation._constellation_type != "qam":
                    raise ValueError(err_msg)

            self._num_streams = 2 * num_streams

            if num_bits_per_symbol is None:
                n = constellation.num_bits_per_symbol // 2
                self._num_bits_per_symbol = n
            else:
                self._num_bits_per_symbol = num_bits_per_symbol // 2

            c = Constellation(
                "pam",
                self._num_bits_per_symbol,
                normalize=False,
                precision=precision,
                device=device,
            )
            c._points = c._points / (torch.std(c._points).item() * np.sqrt(2))
            self._constellation = c.points.real.to(self.dtype)

            self._pam2qam = PAM2QAM(
                2 * self._num_bits_per_symbol, precision=precision, device=device
            )

        else:
            self._num_streams = num_streams
            c = Constellation.check_or_create(
                constellation_type=constellation_type,
                num_bits_per_symbol=num_bits_per_symbol,
                constellation=constellation,
                precision=precision,
                device=device,
            )
            self._constellation = c()
            self._num_bits_per_symbol = c.num_bits_per_symbol

        self._num_symbols = self._constellation.shape[0]
        self._k = min(k, self._num_symbols**self._num_streams)
        if self._k < k:
            msg = f"KBestDetector: The provided value of k={k} is larger than the possible maximum. It has been set to k={self._k}."
            warnings.warn(msg, UserWarning, stacklevel=2)

        # Compute the number of previous paths each layer needs to consider
        num_paths = [1]
        for l in range(1, self._num_streams + 1):
            num_paths.append(min(self._k, self._num_symbols**l))
        self._num_paths = num_paths

        # Precompute indices for tensor scatter updates
        indices = np.zeros(
            [self._num_streams, self._k * self._num_streams, 2], np.int32
        )
        for l in range(0, self._num_streams):
            ind = np.zeros([num_paths[l + 1], self._num_streams])
            ind[:, : l + 1] = 1
            ind = np.stack(np.where(ind), -1)
            indices[l, : ind.shape[0], : ind.shape[1]] = ind
        self._indices = torch.tensor(indices, dtype=torch.int64, device=self.device)

        # Precompute symbol patterns for each layer to avoid recreating in loop
        # Symbol pattern: [k, num_symbols] -> tiled constellation
        self._sym_pattern = (
            self._constellation.reshape(1, -1).expand(self._k, -1).reshape(-1)
        )
        self._ind_pattern = torch.arange(self._num_symbols, device=self.device).repeat(
            self._k
        )

        if self._output == "bit":
            if not self._hard_out:
                if list2llr is None:
                    self.list2llr = List2LLRSimple(
                        self._num_bits_per_symbol, precision=precision, device=device
                    )
                else:
                    self.list2llr = list2llr
            else:
                if self._use_real_rep:
                    n = 2 * self._num_bits_per_symbol
                else:
                    n = self._num_bits_per_symbol
                self._symbolinds2bits = SymbolInds2Bits(
                    n, precision=precision, device=device
                )
        else:
            if not self._hard_out:
                raise ValueError(
                    "Soft-symbols are not supported for this detector."
                )

    @property
    def list2llr(self) -> List2LLR:
        """Set/get the function to compute LLRs from candidate solutions."""
        return self._list2llr

    @list2llr.setter
    def list2llr(self, value: List2LLR) -> None:
        if not isinstance(value, List2LLR):
            raise TypeError("`list2llr` must be an instance of List2LLR.")
        self._list2llr = value

    def _preprocessing(self, y, h, s):
        if self._use_real_rep:
            y, h, s = complex2real_channel(y, h, s)

        y, h = whiten_channel(y, h, s, return_s=False)

        # Order columns of H by decreasing norm
        h_norm = (h.abs() ** 2).sum(dim=-2)
        column_order = h_norm.argsort(dim=-1, descending=True)
        # Gather columns
        h = torch.gather(h, -1, column_order.unsqueeze(-2).expand_as(h))

        # Use Cholesky decomposition instead of QR for better performance
        # QR: H = QR, y' = Q^H y
        # Cholesky: H^H H = L L^H, R = L^H, y' = L^{-1} H^H y
        # This is ~60x faster than QR for batched small matrices
        g = h.mH @ h  # Gram matrix [batch, K, K]
        hty = (h.mH @ y.unsqueeze(-1)).squeeze(-1)  # [batch, K]

        # Avoid the checked Cholesky operation, which cannot be captured by
        # CUDA graphs. As in other Sionna linear-algebra paths, the status is
        # intentionally ignored.
        L, _ = torch.linalg.cholesky_ex(g, check_errors=False)

        # R = L^H (upper triangular, same structure as QR's R)
        r = L.mH

        # Solve L y' = H^H y for y' (triangular solve)
        y = torch.linalg.solve_triangular(L, hty.unsqueeze(-1), upper=False).squeeze(-1)

        return y, r, column_order

    def _select_best_paths(self, dists, path_syms, path_inds, k):
        """Select k best paths based on distances.

        Args:
            dists: [batch_size, num_candidates]
            path_syms: [batch_size, num_candidates, stream+1]
            path_inds: [batch_size, num_candidates, stream+1]
            k: number of paths to keep

        Returns:
            Updated dists, path_syms, path_inds with shape [batch_size, k, ...]
        """
        k = min(path_syms.shape[1], k)

        # Get k paths with shortest distance
        _, ind = torch.topk(-dists, k=k, dim=-1, sorted=True)
        dists = torch.gather(dists, 1, ind)
        path_syms = torch.gather(
            path_syms, 1, ind.unsqueeze(-1).expand(-1, -1, path_syms.shape[-1])
        )
        path_inds = torch.gather(
            path_inds, 1, ind.unsqueeze(-1).expand(-1, -1, path_inds.shape[-1])
        )

        return dists, path_syms, path_inds

    def _next_layer(self, y, r, dists, path_syms, path_inds, stream: int):
        """Process one layer of the K-Best algorithm.

        This implementation uses a memory-efficient approach that computes distances
        using broadcasting BEFORE expanding paths, then only materializes selected
        paths AFTER top-k selection. This reduces memory usage by ~2x compared to
        the standard approach of expanding all path candidates first.
        """
        batch_size = y.shape[0]
        stream_ind = self._num_streams - 1 - stream
        num_paths = self._num_paths[stream]
        num_paths_next = self._num_paths[stream + 1]

        # Extract relevant values from input tensors (views, no copy)
        curr_dists = dists[:, :num_paths]  # [batch, num_paths]
        curr_path_syms = path_syms[:, :num_paths, :stream]  # [batch, num_paths, stream]
        curr_path_inds = path_inds[:, :num_paths, :stream]  # [batch, num_paths, stream]

        # Get channel coefficients for this layer
        # r_row has shape [batch, stream+1] after flip, where:
        #   - r_row[:, :-1] are coefficients for existing symbols
        #   - r_row[:, -1] is the diagonal coefficient for the new symbol
        y_s = y[:, stream_ind]  # [batch]
        r_row = r[:, stream_ind, stream_ind:].flip(-1)  # [batch, stream+1]

        # Compute partial contribution from existing symbols (no expansion needed)
        if stream > 0:
            r_existing = r_row[:, :-1].unsqueeze(1)  # [batch, 1, stream]
            partial = (r_existing * curr_path_syms).sum(dim=-1)  # [batch, num_paths]
        else:
            partial = torch.zeros(
                batch_size, num_paths, dtype=self.dtype, device=self.device
            )

        # For each new symbol candidate, compute full residual using broadcasting
        # This avoids materializing [batch, num_paths * num_symbols, stream] tensors
        r_diag = r_row[:, -1:]  # [batch, 1] - diagonal coefficient for new symbol
        new_contrib = r_diag * self._constellation.unsqueeze(0)  # [batch, num_symbols]

        # Compute residuals for all (path, symbol) combinations
        # y_s: [batch] -> [batch, 1, 1]
        # partial: [batch, num_paths] -> [batch, num_paths, 1]
        # new_contrib: [batch, num_symbols] -> [batch, 1, num_symbols]
        # Result: [batch, num_paths, num_symbols]
        residuals = (
            y_s.unsqueeze(-1).unsqueeze(-1)
            - partial.unsqueeze(-1)
            - new_contrib.unsqueeze(1)
        )
        deltas = residuals.abs().square()  # [batch, num_paths, num_symbols]

        # Compute distances for all (path, symbol) combinations
        all_dists = curr_dists.unsqueeze(-1) + deltas  # [batch, num_paths, num_symbols]

        # Select top-k from the flattened view
        all_dists_flat = all_dists.view(
            batch_size, -1
        )  # [batch, num_paths * num_symbols]
        _, topk_idx = torch.topk(-all_dists_flat, k=num_paths_next, dim=-1, sorted=True)

        # Decode indices to (path_idx, symbol_idx)
        path_idx = topk_idx // self._num_symbols  # Which of the original paths
        sym_idx = topk_idx % self._num_symbols  # Which new symbol

        # Gather selected distances
        sel_dists = torch.gather(all_dists_flat, 1, topk_idx)  # [batch, num_paths_next]

        # Create new symbols for selected paths
        new_syms = self._constellation[sym_idx].unsqueeze(
            -1
        )  # [batch, num_paths_next, 1]
        new_inds = sym_idx.unsqueeze(-1)  # [batch, num_paths_next, 1]

        # Gather selected paths (only materialize selected paths, not all candidates)
        if stream > 0:
            sel_path_syms = torch.gather(
                curr_path_syms, 1, path_idx.unsqueeze(-1).expand(-1, -1, stream)
            )  # [batch, num_paths_next, stream]
            sel_path_inds = torch.gather(
                curr_path_inds, 1, path_idx.unsqueeze(-1).expand(-1, -1, stream)
            )  # [batch, num_paths_next, stream]
            # Append new symbols to selected paths
            sel_path_syms = torch.cat([sel_path_syms, new_syms], dim=-1)
            sel_path_inds = torch.cat([sel_path_inds, new_inds], dim=-1)
        else:
            # First layer: no previous symbols, just use new symbols directly
            sel_path_syms = new_syms  # [batch, num_paths_next, 1]
            sel_path_inds = new_inds  # [batch, num_paths_next, 1]

        return sel_dists, sel_path_syms, sel_path_inds

    def _unsort(self, column_order, tensor, transpose=True):
        unsort_inds = column_order.argsort(dim=-1)
        if transpose:
            tensor = tensor.transpose(-1, -2)
        tensor = torch.gather(tensor, -2, unsort_inds.unsqueeze(-1).expand_as(tensor))
        if transpose:
            tensor = tensor.transpose(-1, -2)
        return tensor

    def build(self, *input_shapes):
        if not (input_shapes[1][-2] >= input_shapes[1][-1]):
            raise ValueError(
                "The number of receive antennas cannot be smaller than the "
                "number of streams"
            )

    @staticmethod
    def _restore_batch_dims(
        tensor: torch.Tensor, batch_shape: list[int]
    ) -> torch.Tensor:
        """Restore the leading dimensions flattened or added by ``call``."""
        if len(batch_shape) == 0:
            return tensor.squeeze(0)
        if len(batch_shape) > 1:
            return split_dim(tensor, batch_shape, 0)
        return tensor

    def call(self, y: torch.Tensor, h: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        # Flatten batch dimensions
        batch_shape = list(y.shape[:-1])
        num_batch_dims = len(batch_shape)
        if num_batch_dims == 0:
            y = y.unsqueeze(0)
            h = h.unsqueeze(0)
            s = s.unsqueeze(0)
        elif num_batch_dims > 1:
            y = flatten_dims(y, num_batch_dims, 0)
            h = flatten_dims(h, num_batch_dims, 0)
            s = flatten_dims(s, num_batch_dims, 0)

        y, r, column_order = self._preprocessing(y, h, s)
        batch_size = y.shape[0]

        # Initialize state for K-Best search
        dists = torch.zeros(batch_size, self._k, dtype=self.dtype, device=self.device)
        path_syms = torch.zeros(
            batch_size,
            self._k,
            self._num_streams,
            dtype=self.cdtype,
            device=self.device,
        )
        path_inds = torch.zeros(
            batch_size,
            self._k,
            self._num_streams,
            dtype=torch.int64,
            device=self.device,
        )

        # Sequential K-Best algorithm
        for stream in range(self._num_streams):
            dists, path_syms, path_inds = self._next_layer(
                y, r, dists, path_syms, path_inds, stream
            )

        # Reverse order as detection started with the last symbol first
        path_syms = path_syms.flip(-1)
        path_inds = path_inds.flip(-1)

        if self._hard_out:
            path_inds = self._unsort(column_order, path_inds)
            hard_dec = path_inds[:, 0, :]

            if self._use_real_rep:
                hard_dec = self._pam2qam(
                    hard_dec[..., : self._num_streams // 2],
                    hard_dec[..., self._num_streams // 2 :],
                )

            if self._output == "bit":
                hard_dec = self._symbolinds2bits(hard_dec)

            return self._restore_batch_dims(hard_dec, batch_shape)

        else:
            if self._use_real_rep:
                llr = self.list2llr(y, r, dists, path_inds.to(torch.int32), path_syms)
                llr = self._unsort(column_order, llr, transpose=False)

                llr1 = llr[:, : self._num_streams // 2]
                llr2 = llr[:, self._num_streams // 2 :]
                llr1 = llr1.unsqueeze(-1)
                llr2 = llr2.unsqueeze(-1)
                llr = torch.cat([llr1, llr2], -1)
                llr = llr.reshape(
                    -1, self._num_streams // 2, 2 * self._num_bits_per_symbol
                )
            else:
                llr = self.list2llr(y, r, dists, path_inds.to(torch.int32), path_syms)
                llr = self._unsort(column_order, llr, transpose=False)

            return self._restore_batch_dims(llr, batch_shape)


class EPDetector(Block):
    r"""
    MIMO Expectation Propagation (EP) detector.

    This block implements Expectation Propagation (EP) MIMO detection as
    described in :cite:p:`EP2014`. It can generate hard- or soft-decisions for
    symbols or bits.

    This block assumes the following channel model:

    .. math::
        \mathbf{y} = \mathbf{H}\mathbf{x} + \mathbf{n}

    where :math:`\mathbf{y}\in\mathbb{C}^M` is the received signal vector,
    :math:`\mathbf{x}\in\mathcal{C}^S` is the vector of transmitted symbols
    which are uniformly and independently drawn from the constellation
    :math:`\mathcal{C}`,
    :math:`\mathbf{H}\in\mathbb{C}^{M\times S}` is the known channel matrix,
    and :math:`\mathbf{n}\in\mathbb{C}^M` is a complex Gaussian noise vector.
    It is assumed that
    :math:`\mathbb{E}\left[\mathbf{n}\right]=\mathbf{0}` and
    :math:`\mathbb{E}\left[\mathbf{n}\mathbf{n}^{\mathsf{H}}\right]=\mathbf{S}`,
    where :math:`\mathbf{S}` has full rank.

    The channel model is first whitened using
    :func:`~sionna.phy.mimo.whiten_channel` and then converted to its
    real-valued equivalent, see
    :func:`~sionna.phy.mimo.complex2real_channel`, prior to MIMO detection.

    The computation of LLRs is done by converting the symbol logits
    that naturally arise in the algorithm to LLRs using
    :func:`~sionna.phy.mapping.PAM2QAM`. Custom conversions of symbol logits
    to LLRs can be implemented by using the soft-symbol output.

    The detector is currently restricted to QAM constellations.

    :param output: Type of output, either ``"bit"`` for LLRs on bits or
        ``"symbol"`` for logits on constellation symbols
    :param num_bits_per_symbol: Number of bits per QAM constellation symbol,
        e.g., 4 for QAM16
    :param hard_out: If `True`, the detector computes hard-decided bit values
        or constellation point indices instead of soft-values.
        Defaults to `False`.
    :param l: Number of iterations. Defaults to 10.
    :param beta: Parameter :math:`\beta\in[0,1]` for update smoothing.
        Defaults to 0.9.
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`, :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for computations

    :input y: [...,M], `torch.complex`. Received signals.
    :input h: [...,M,num_streams], `torch.complex`. Channel matrices.
    :input s: [...,M,M], `torch.complex`. Noise covariance matrices.

    One of:

    :output llr: [..., num_streams, num_bits_per_symbol], `torch.float`.
        LLRs or hard-decisions for every bit of every stream,
        if ``output`` equals ``"bit"``.
    :output logits: [..., num_streams, num_points], `torch.float` or [..., num_streams], `torch.int32`.
        Logits or hard-decisions for constellation symbols for every stream,
        if ``output`` equals ``"symbol"``.
        Hard-decisions correspond to the symbol indices.

    .. rubric:: Examples

    .. code-block:: python

        detector = EPDetector(
            output="bit",
            num_bits_per_symbol=4,
            l=10
        )
        llr = detector(y, h, s)
    """

    def __init__(
        self,
        output: str,
        num_bits_per_symbol: int,
        hard_out: bool = False,
        l: int = 10,
        beta: float = 0.9,
        precision: Optional[Precision] = None,
        device: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(precision=precision, device=device, **kwargs)

        if self.precision == "single":
            self._prec = 1e-6
        else:
            self._prec = 1e-12

        if output not in ("bit", "symbol"):
            raise ValueError("output must be one of: 'bit', 'symbol'")
        self._output = output
        self._hard_out = hard_out

        if self._output == "symbol":
            self._pam2qam = PAM2QAM(
                num_bits_per_symbol, hard_out, precision=precision, device=device
            )
        else:
            self._symbollogits2llrs = SymbolLogits2LLRs(
                "maxlog",
                num_bits_per_symbol // 2,
                hard_out=hard_out,
                precision=precision,
                device=device,
            )

        if not (l >= 1):
            raise ValueError("l must be a positive integer")
        self._l = l

        if not 0.0 <= beta <= 1.0:
            raise ValueError("beta must be in [0,1]")
        self._beta = beta

        self._num_bits_per_symbol = num_bits_per_symbol // 2
        points = Constellation(
            "pam", int(self._num_bits_per_symbol), precision=precision, device=device
        )()
        self._points = (points / np.sqrt(2.0)).real.to(self.dtype)
        self._es = torch.tensor(
            self._points.var().item(), dtype=self.dtype, device=self.device
        )

        # Pre-compute scalar for noise
        self._no = torch.tensor(0.5, dtype=self.dtype, device=self.device)

    def compute_sigma_mu(self, h_t_h, h_t_y, no, lam, gam):
        """Eq. (28) and Eq. (29)."""
        lam = torch.diag_embed(lam)
        gam = gam.unsqueeze(-1)

        # Use inv_ex with check_errors=False for CUDA graph compatibility
        sigma, _ = torch.linalg.inv_ex(h_t_h + no * lam, check_errors=False)
        mu = (sigma @ (h_t_y + no * gam)).squeeze(-1)
        sigma = sigma * no
        sigma = torch.diagonal(sigma, dim1=-2, dim2=-1)

        return sigma, mu

    def compute_v_x_obs(self, sigma, mu, lam, gam):
        """Eq. (31) and Eq. (32)."""
        v_obs = torch.clamp(1 / (1 / sigma - lam), min=self._prec)
        x_obs = v_obs * (mu / sigma - gam)
        return v_obs, x_obs

    def compute_v_x(self, v_obs, x_obs):
        """Eq. (33)."""
        x_obs = x_obs.unsqueeze(-1)
        v_obs = v_obs.unsqueeze(-1)

        points = expand_to_rank(self._points, x_obs.dim(), axis=0)
        logits = -((x_obs - points) ** 2) / (2.0 * v_obs)
        pmf = torch.softmax(logits, dim=-1)

        x = (points * pmf).sum(dim=-1, keepdim=True)
        v = ((points - x) ** 2 * pmf).sum(dim=-1)
        v = torch.clamp(v, min=self._prec)
        x = x.squeeze(-1)

        return v, x, logits

    def update_lam_gam(self, v, v_obs, x, x_obs, lam, gam):
        """Eq. (35), Eq. (36), Eq. (37), and Eq. (38)."""
        lam_old = lam
        gam_old = gam

        lam = 1 / v - 1 / v_obs
        gam = x / v - x_obs / v_obs

        lam_new = torch.where(lam < 0, lam_old, lam)
        gam_new = torch.where(lam < 0, gam_old, gam)

        lam_damp = (1 - self._beta) * lam_new + self._beta * lam_old
        gam_damp = (1 - self._beta) * gam_new + self._beta * gam_old

        return lam_damp, gam_damp

    def call(self, y: torch.Tensor, h: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        batch_shape = list(y.shape[:-1])
        num_batch_dims = len(batch_shape)
        if num_batch_dims > 1:
            y = flatten_dims(y, num_batch_dims, 0)
            h = flatten_dims(h, num_batch_dims, 0)
            s = flatten_dims(s, num_batch_dims, 0)

        n_t = h.shape[-1]

        y, h, s = whiten_channel(y, h, s)
        y, h, s = complex2real_channel(y, h, s)

        # Initialize EP iteration variables
        batch_dims = y.shape[:-1]
        n_t_r = h.shape[-1]  # 2 * num_streams after real conversion
        gam = torch.zeros(*batch_dims, n_t_r, dtype=self.dtype, device=self.device)
        lam = (
            torch.ones(*batch_dims, n_t_r, dtype=self.dtype, device=self.device)
            / self._es
        )

        h_t_h = h.mT @ h
        y = y.unsqueeze(-1)
        h_t_y = h.mT @ y
        no = expand_to_rank(self._no, h.dim(), axis=-1)

        for _ in range(self._l):
            sigma, mu = self.compute_sigma_mu(h_t_h, h_t_y, no, lam, gam)
            v_obs, x_obs = self.compute_v_x_obs(sigma, mu, lam, gam)
            v, x, logits = self.compute_v_x(v_obs, x_obs)
            lam, gam = self.update_lam_gam(v, v_obs, x, x_obs, lam, gam)

        pam1_logits = logits[..., :n_t, :]
        pam2_logits = logits[..., n_t:, :]

        if self._output == "symbol" and self._hard_out:
            pam1_ind = pam1_logits.argmax(dim=-1).to(torch.int32)
            pam2_ind = pam2_logits.argmax(dim=-1).to(torch.int32)
            qam_ind = self._pam2qam(pam1_ind, pam2_ind).to(torch.int32)

            if num_batch_dims > 1:
                qam_ind = split_dim(qam_ind, batch_shape, 0)
            return qam_ind

        elif self._output == "symbol" and not self._hard_out:
            qam_logits = self._pam2qam(pam1_logits, pam2_logits)

            if num_batch_dims > 1:
                qam_logits = split_dim(qam_logits, batch_shape, 0)
            return qam_logits

        elif self._output == "bit":
            llr1 = self._symbollogits2llrs(pam1_logits)
            llr2 = self._symbollogits2llrs(pam2_logits)

            llr = torch.stack([llr1, llr2], -1)
            llr = flatten_last_dims(llr)

            if num_batch_dims > 1:
                llr = split_dim(llr, batch_shape, 0)
            return llr


class MMSEPICDetector(Block):
    r"""
    Minimum mean square error (MMSE) with parallel interference cancellation
    (PIC) detector.

    This block implements the MMSE PIC detector, as proposed in :cite:p:`CST2011`.
    For ``num_iter``>1, this implementation performs MMSE PIC self-iterations.
    MMSE PIC self-iterations can be understood as a concatenation of MMSE PIC
    detectors from :cite:p:`CST2011`, which forward intrinsic LLRs to the next
    self-iteration.

    Compared to :cite:p:`CST2011`, this implementation also accepts priors on the
    constellation symbols as an alternative to priors on the bits.

    This block assumes the following channel model:

    .. math::
        \mathbf{y} = \mathbf{H}\mathbf{x} + \mathbf{n}

    where :math:`\mathbf{y}\in\mathbb{C}^M` is the received signal vector,
    :math:`\mathbf{x}\in\mathcal{C}^S` is the vector of transmitted symbols
    which are uniformly and independently drawn from the constellation
    :math:`\mathcal{C}`,
    :math:`\mathbf{H}\in\mathbb{C}^{M\times S}` is the known channel matrix,
    and :math:`\mathbf{n}\in\mathbb{C}^M` is a complex Gaussian noise vector.
    It is assumed that
    :math:`\mathbb{E}\left[\mathbf{n}\right]=\mathbf{0}` and
    :math:`\mathbb{E}\left[\mathbf{n}\mathbf{n}^{\mathsf{H}}\right]=\mathbf{S}`,
    where :math:`\mathbf{S}` has full rank.

    The algorithm starts by computing the soft symbols
    :math:`\bar{x}_s=\mathbb{E}\left[ x_s \right]` and
    variances :math:`v_s=\mathbb{E}\left[ |e_s|^2\right]` from the priors,
    where :math:`e_s = x_s - \bar{x}_s`, for all :math:`s=1,\dots,S`.

    Next, for each stream, the interference caused by all other streams is
    cancelled from the observation :math:`\mathbf{y}`, leading to

    .. math::
        \hat{\mathbf{y}}_s = \mathbf{y} - \sum_{j\neq s} \mathbf{h}_j x_j = \mathbf{h}_s x_s + \tilde{\mathbf{n}}_s,\quad s=1,\dots,S

    where
    :math:`\tilde{\mathbf{n}}_s=\sum_{j\neq s} \mathbf{h}_j e_j + \mathbf{n}`.

    Then, a linear MMSE filter :math:`\mathbf{w}_s` is computed to reduce the
    residual noise for each observation :math:`\hat{\mathbf{y}}_s`, which is
    given as

    .. math::
        \mathbf{w}_s = \mathbf{h}_s^{\mathsf{H}}\left( \mathbf{H} \mathbf{D}_s\mathbf{H}^{\mathsf{H}} +\mathbf{S} \right)^{-1}

    where :math:`\mathbf{D}_s \in \mathbb{C}^{S\times S}` is diagonal with
    entries

    .. math::
        \left[\mathbf{D}_s\right]_{i,i} = \begin{cases}
                                            v_i & i\neq s \\
                                            1 & i=s.
                                          \end{cases}

    The filtered observations

    .. math::
        \tilde{z}_s = \mathbf{w}_s^{\mathsf{H}} \hat{\mathbf{y}}_s = \tilde{\mu}_s x_s + \mathbf{w}_s^{\mathsf{H}}\tilde{\mathbf{n}}_s

    where :math:`\tilde{\mu}_s=\mathbf{w}_s^{\mathsf{H}} \mathbf{h}_s`, are
    then demapped to either symbol logits or LLRs, assuming that the remaining
    noise is Gaussian with variance

    .. math::
        \nu_s^2 = \mathop{\text{Var}}\left[\tilde{z}_s\right] = \mathbf{w}_s^{\mathsf{H}} \left(\sum_{j\neq s} \mathbf{h}_j \mathbf{h}_j^{\mathsf{H}} v_j +\mathbf{S} \right)\mathbf{w}_s.

    The resulting soft-symbols can then be used for the next self-iteration of
    the algorithm.

    Note that this algorithm can be substantially simplified as described in
    :cite:p:`CST2011` to avoid the computation of different matrix inverses for each
    stream. This is the version which is implemented.

    :param output: Type of output, either ``"bit"`` for LLRs on bits or
        ``"symbol"`` for logits on constellation symbols
    :param demapping_method: Demapping method, either ``"maxlog"``
        (default) or ``"app"``
    :param num_iter: Number of MMSE PIC iterations. Defaults to 1.
    :param constellation_type: Constellation type, one of ``"qam"``,
        ``"pam"``, or ``"custom"``. For ``"custom"``, an instance of
        :class:`~sionna.phy.mapping.Constellation` must be provided.
    :param num_bits_per_symbol: Number of bits per constellation symbol,
        e.g., 4 for QAM16.
        Only required for ``constellation_type`` in [``"qam"``, ``"pam"``].
    :param constellation: An instance of
        :class:`~sionna.phy.mapping.Constellation` or `None`. If `None`,
        ``constellation_type`` and ``num_bits_per_symbol`` must be provided.
    :param hard_out: If `True`, the detector computes hard-decided bit values
        or constellation point indices instead of soft-values.
        Defaults to `False`.
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`, :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for computations

    :input y: [...,M], `torch.complex`. Received signals.
    :input h: [...,M,num_streams], `torch.complex`. Channel matrices.
    :input s: [...,M,M], `torch.complex`. Noise covariance matrices.
    :input prior: [...,num_streams,num_bits_per_symbol] or [...,num_streams,num_points], `torch.float`.
        Prior of the transmitted signals.
        If ``output`` equals ``"bit"``, then LLRs of the transmitted bits are expected.
        If ``output`` equals ``"symbol"``, then logits of the transmitted constellation points are expected.

    One of:

    :output llr: [..., num_streams, num_bits_per_symbol], `torch.float`.
        LLRs or hard-decisions for every bit of every stream,
        if ``output`` equals ``"bit"``.
    :output logits: [..., num_streams, num_points], `torch.float` or [..., num_streams], `torch.int32`.
        Logits or hard-decisions for constellation symbols for every stream,
        if ``output`` equals ``"symbol"``.
        Hard-decisions correspond to the symbol indices.

    .. rubric:: Examples

    .. code-block:: python

        detector = MMSEPICDetector(
            output="bit",
            demapping_method="maxlog",
            num_iter=3,
            constellation_type="qam",
            num_bits_per_symbol=4
        )
        llr = detector(y, h, s, prior)
    """

    def __init__(
        self,
        output: str,
        demapping_method: str = "maxlog",
        num_iter: int = 1,
        constellation_type: Optional[str] = None,
        num_bits_per_symbol: Optional[int] = None,
        constellation: Optional[Constellation] = None,
        hard_out: bool = False,
        precision: Optional[Precision] = None,
        device: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(precision=precision, device=device, **kwargs)

        if not isinstance(num_iter, int):
            raise TypeError("num_iter must be an integer")
        if output not in ("bit", "symbol"):
            raise ValueError("output must be one of: 'bit', 'symbol'")
        if demapping_method not in ("app", "maxlog"):
            raise ValueError(
                "demapping_method must be one of: 'app', 'maxlog'"
            )

        self._num_iter = num_iter
        self._output = output
        self._epsilon = 1e-4
        self._demapping_method = demapping_method
        self._hard_out = hard_out

        self._constellation = Constellation.check_or_create(
            constellation_type=constellation_type,
            num_bits_per_symbol=num_bits_per_symbol,
            constellation=constellation,
            precision=precision,
            device=device,
        )

        self._llr_2_symbol_logits = LLRs2SymbolLogits(
            self._constellation.num_bits_per_symbol, precision=precision, device=device
        )

        if self._output == "symbol":
            self._llr_2_symbol_logits_output = LLRs2SymbolLogits(
                self._constellation.num_bits_per_symbol,
                precision=precision,
                device=device,
                hard_out=hard_out,
            )
            self._symbol_logits_2_llrs = SymbolLogits2LLRs(
                method=demapping_method,
                num_bits_per_symbol=self._constellation.num_bits_per_symbol,
                precision=precision,
                device=device,
            )

        self._symbol_logits_2_moments = SymbolLogits2Moments(
            constellation=self._constellation, precision=precision, device=device
        )

        self._bit_demapper = Demapper(
            demapping_method=demapping_method,
            constellation=self._constellation,
            precision=precision,
            device=device,
        )

    def call(
        self,
        y: torch.Tensor,
        h: torch.Tensor,
        s: torch.Tensor,
        prior: torch.Tensor,
    ) -> torch.Tensor:
        # Whiten channel
        y, h = whiten_channel(y, h, s, return_s=False)

        # Matched filtering
        y_mf = insert_dims((h.mH @ y.unsqueeze(-1)).squeeze(-1), num_dims=1, axis=-1)

        # Gram matrix
        g = h.mH @ h

        # Real-valued domain for numerical stability
        hr = complex2real_matrix(h)
        gr = hr.mT @ hr

        # Compute a priori LLRs
        if (self._output == "symbol") and (
            prior.shape[-1] == self._constellation.num_points
        ):
            llr_a = self._symbol_logits_2_llrs(prior)
        else:
            llr_a = prior

        llr_shape = llr_a.shape

        def mmse_pic_self_iteration(llr_d, llr_a):
            llr_a = llr_d

            x_logits = self._llr_2_symbol_logits(llr_a)
            x_hat, var_x = self._symbol_logits_2_moments(x_logits)

            # Parallel interference cancellation
            y_mf_pic = (
                y_mf
                + g * insert_dims(x_hat, num_dims=1, axis=-2)
                - g @ insert_dims(x_hat, num_dims=1, axis=-1)
            )

            # Stack variances for real-valued domain
            var_x = torch.cat([var_x, var_x], dim=-1).real.to(self.dtype)
            var_x_row_vec = insert_dims(var_x, num_dims=1, axis=-2)

            # Compute a = G_r * diag(var_x) + I
            i = expand_to_rank(
                torch.eye(gr.shape[-1], dtype=gr.dtype, device=gr.device), gr.dim(), 0
            )
            a = gr * var_x_row_vec + i

            # Use inv_ex with check_errors=False for CUDA graph compatibility
            a_inv, _ = torch.linalg.inv_ex(a, check_errors=False)

            mu = (a_inv * gr.mT).sum(dim=-1)

            y_mf_pic_trans = y_mf_pic.mT
            y_mf_pic_trans = complex2real_vector(y_mf_pic_trans)
            y_mf_pic_trans = torch.cat([y_mf_pic_trans, y_mf_pic_trans], dim=-2)

            x_hat = real2complex_vector(
                (a_inv * y_mf_pic_trans).sum(dim=-1) / mu.to(a_inv.dtype)
            )

            var_x = mu / torch.clamp(1 - var_x * mu, min=self._epsilon)
            var_x, _ = var_x.chunk(2, dim=-1)

            no_eff = 1.0 / var_x

            llr_d = self._bit_demapper(x_hat, no_eff, llr_a).reshape(llr_shape)

            return llr_d, llr_a

        null_prior = torch.zeros(llr_shape, dtype=self.dtype, device=self.device)
        llr_d = llr_a
        llr_a = null_prior

        for _ in range(self._num_iter):
            llr_d, llr_a = mmse_pic_self_iteration(llr_d, llr_a)

        llr_e = llr_d - llr_a

        if self._output == "symbol":
            out = self._llr_2_symbol_logits_output(llr_e)
        else:
            out = llr_e
            if self._hard_out:
                out = hard_decisions(out)

        return out
