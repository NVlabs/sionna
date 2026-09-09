#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Convolutional code encoding."""

from typing import Optional, Tuple, Union
import warnings

import torch

from sionna.phy import Block
from sionna.phy.fec.utils import bin2int, int2bin
from sionna.phy.fec.conv.utils import resolve_gen_poly, Trellis


__all__ = ["ConvEncoder"]


class ConvEncoder(Block):
    r"""Encodes an information binary tensor to a convolutional codeword.

    Currently, only generator polynomials for codes of rate=1/n for n=2,3,4,...
    are allowed.

    :param gen_poly: Sequence of strings with each string being a 0,1 sequence.
        If `None`, ``rate`` and ``constraint_length`` must be provided.
    :param rate: Valid values are 1/3 and 0.5. Only required if ``gen_poly``
        is `None`.
    :param constraint_length: Valid values are between 3 and 8 inclusive.
        Only required if ``gen_poly`` is `None`.
    :param rsc: Boolean flag indicating whether the Trellis generated is
        recursive systematic or not. If `True`, the encoder is
        recursive-systematic. In this case first polynomial in ``gen_poly``
        is used as the feedback polynomial. Defaults to `False`.
    :param terminate: Encoder is terminated to all zero state if `True`.
        If terminated, the true rate of the code is slightly lower than
        ``rate``.
    :param precision: Precision used for internal calculations and outputs.
        If `None`, :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for computation (e.g., 'cpu', 'cuda:0').
        If `None`, :attr:`~sionna.phy.config.Config.device` is used.

    :input bits: [..., k], `torch.float`.
        Binary tensor containing the information bits where ``k`` is the
        information length.

    :output cw: [..., k/rate], `torch.float`.
        Binary tensor containing the encoded codeword for the given input
        information tensor where ``rate`` is
        :math:`\frac{1}{\textrm{len}\left(\textrm{gen\_poly}\right)}`
        (if ``gen_poly`` is provided).

    .. rubric:: Notes

    The generator polynomials from :cite:p:`Moon` are available for various
    rate and constraint lengths. To select them, use the ``rate`` and
    ``constraint_length`` arguments.

    In addition, polynomials for any non-recursive convolutional encoder
    can be given as input via ``gen_poly`` argument. Currently, only
    polynomials with rate=1/n are supported. When the ``gen_poly`` argument
    is given, the ``rate`` and ``constraint_length`` arguments are ignored.

    Various notations are used in the literature to represent the generator
    polynomials for convolutional codes. In :cite:p:`Moon`, the octal digits
    format is primarily used. In the octal format, the generator polynomial
    `10011` corresponds to 46. Another widely used format
    is decimal notation with MSB. In this notation, polynomial `10011`
    corresponds to 19. For simplicity, the
    :class:`~sionna.phy.fec.conv.ConvEncoder` only accepts the bit
    format i.e. `10011` as ``gen_poly`` argument.

    Also note that ``constraint_length`` and ``memory`` are two different
    terms often used to denote the strength of a convolutional code. In this
    sub-package, we use ``constraint_length``. For example, the
    polynomial `10011` has a ``constraint_length`` of 5, however its
    ``memory`` is only 4.

    When ``terminate`` is `True`, the true rate of the convolutional
    code is slightly lower than ``rate``. It equals
    :math:`\frac{r*k}{k+\mu}` where `r` denotes ``rate`` and
    :math:`\mu` is ``constraint_length`` - 1. For example when
    ``terminate`` is `True`, ``k=100``,
    :math:`\mu=4` and ``rate`` =0.5, true rate equals
    :math:`\frac{0.5*100}{104}=0.481`.

    .. rubric:: Examples


    .. code-block:: python

        from sionna.phy.fec.conv import ConvEncoder

        encoder = ConvEncoder(rate=0.5, constraint_length=5)
        u = torch.randint(0, 2, (10, 100), dtype=torch.float32)
        c = encoder(u)
        print(c.shape)
        # torch.Size([10, 200])
    """

    def __init__(
        self,
        gen_poly: Optional[Tuple[str, ...]] = None,
        rate: float = 1/2,
        constraint_length: int = 3,
        rsc: bool = False,
        terminate: bool = False,
        precision: Optional[str] = None,
        device: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(precision=precision, device=device, **kwargs)

        self._gen_poly = resolve_gen_poly(gen_poly, rate, constraint_length)

        self._rsc = rsc
        self._terminate = terminate

        self._coderate_desired = 1 / len(self.gen_poly)
        # Differs when terminate is True
        self._coderate = self._coderate_desired

        self._trellis = Trellis(self.gen_poly, rsc=self._rsc, device=self.device)
        self._mu = self._trellis.mu

        # conv_k denotes number of input bit streams.
        # Only 1 allowed in current implementation
        self._conv_k = self._trellis.conv_k

        # conv_n denotes number of output bits for conv_k input bits
        self._conv_n = self._trellis.conv_n

        self._ni = 2**self._conv_k
        self._no = 2**self._conv_n
        self._ns = self._trellis.ns

        # For conv codes, the code dimensions are unknown during initialization
        self._k = None
        self._n = None
        self._num_syms = None

    @property
    def gen_poly(self) -> Tuple[str, ...]:
        """Generator polynomial used by the encoder"""
        return self._gen_poly

    @property
    def coderate(self) -> float:
        """Rate of the code used in the encoder"""
        if self.terminate and self._k is None:
            warnings.warn(
                "Due to termination, the true coderate is lower "
                "than the returned design rate. "
                "The exact true rate is dependent on the value of k and "
                "hence cannot be computed before the first call().",
                RuntimeWarning,
                stacklevel=2,
            )
        elif self.terminate and self._k is not None:
            term_factor = self._k / (self._k + self._mu)
            self._coderate = self._coderate_desired * term_factor
        return self._coderate

    @property
    def trellis(self) -> Trellis:
        """Trellis object used during encoding"""
        return self._trellis

    @property
    def terminate(self) -> bool:
        """Indicates if the convolutional encoder is terminated"""
        return self._terminate

    @property
    def k(self) -> Optional[int]:
        """Number of information bits per codeword"""
        if self._k is None:
            warnings.warn(
                "The value of k cannot be computed before the first call().",
                RuntimeWarning,
                stacklevel=2,
            )
        return self._k

    @property
    def n(self) -> Optional[int]:
        """Number of codeword bits"""
        if self._n is None:
            warnings.warn(
                "The value of n cannot be computed before the first call().",
                RuntimeWarning,
                stacklevel=2,
            )
        return self._n

    def build(self, input_shape: torch.Size):
        """Build block and check dimensions.

        :param input_shape: Shape of input tensor (..., k)
        """
        self._k = input_shape[-1]
        self._n = int(self._k / self._coderate_desired)
        if self._terminate:
            self._n += int(self._mu / self._coderate_desired)

        # num_syms denotes number of encoding periods or state transitions.
        # Different from _k when _conv_k > 1.
        self._num_syms = int(self._k // self._conv_k)

        # Move trellis to correct device if needed
        if self._trellis.device != self.device:
            self._trellis.to(self.device)

    @torch.compiler.disable
    def call(self, bits: torch.Tensor, /) -> torch.Tensor:
        r"""Convolutional code encoding function.

        :param bits: Binary tensor of shape [..., k] containing the
            information bits where ``k`` is the information length.

        .. rubric:: Notes

        This method uses ``@torch.compiler.disable`` because the encoding
        loop iterates over information bits, causing extremely long
        compilation times with ``torch.compile``.

        .. rubric:: Examples


        .. code-block:: python

            from sionna.phy.fec.conv import ConvEncoder

            encoder = ConvEncoder(rate=0.5, constraint_length=5)
            u = torch.randint(0, 2, (10, 100), dtype=torch.float32)
            c = encoder(u)
            print(c.shape)
            # torch.Size([10, 200])
        """
        # Check if rebuild is needed
        if bits.shape[-1] != self._k:
            self._built = False
            self.build(bits.shape)
            self._built = True

        # Cast internally to int32 to enable bitshift operations
        msg = bits.to(torch.int32)
        output_shape = list(msg.shape)
        output_shape[-1] = self._n

        msg_reshaped = msg.reshape(-1, self._k)
        batch_size = msg_reshaped.shape[0]
        term_syms = int(self._mu) if self._terminate else 0

        prev_st = torch.zeros(batch_size, dtype=torch.int32, device=self.device)
        cw_parts = []

        idx_offset = torch.arange(self._conv_k, device=self.device)
        for idx in range(0, self._k, self._conv_k):
            # Get message bits at current index
            msg_bits_idx = msg_reshaped[:, idx:idx + self._conv_k]
            msg_idx = bin2int(msg_bits_idx)

            # State transition: to_nodes[prev_st, msg_idx]
            new_st = self._trellis.to_nodes[prev_st, msg_idx]

            # Output symbol: op_mat[prev_st, new_st]
            idx_syms = self._trellis.op_mat[prev_st, new_st]
            idx_bits = int2bin(idx_syms, self._conv_n)
            cw_parts.append(idx_bits)
            prev_st = new_st

        cw = torch.cat(cw_parts, dim=1)

        # Termination
        if self._terminate:
            term_parts = []
            if self._rsc:
                fb_poly = torch.tensor(
                    [int(x) for x in self.gen_poly[0][1:]],
                    dtype=torch.int32,
                    device=self.device
                )

            for idx in range(0, term_syms, self._conv_k):
                prev_st_bits = int2bin(prev_st, self._mu)
                if self._rsc:
                    # Compute feedback bit
                    msg_idx = (prev_st_bits * fb_poly).sum(dim=-1)
                    msg_idx = int2bin(msg_idx, 1).squeeze(-1)
                else:
                    msg_idx = torch.zeros(batch_size, dtype=torch.int32,
                                          device=self.device)

                new_st = self._trellis.to_nodes[prev_st, msg_idx]
                idx_syms = self._trellis.op_mat[prev_st, new_st]
                idx_bits = int2bin(idx_syms, self._conv_n)
                term_parts.append(idx_bits)
                prev_st = new_st

            if term_parts:
                term_bits = torch.cat(term_parts, dim=1)
                cw = torch.cat([cw, term_bits], dim=-1)

        cw = cw.to(self.dtype)
        cw_reshaped = cw.reshape(output_shape)

        return cw_reshaped



