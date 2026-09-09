#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Blocks for cyclic redundancy checks (CRC) and utility functions."""

from typing import Optional, Tuple
import warnings
import numpy as np
import torch

from sionna.phy import Block
from sionna.phy.fec.utils import int_mod_2


__all__ = ["CRCEncoder", "CRCDecoder"]


class CRCEncoder(Block):
    """Adds a Cyclic Redundancy Check (CRC) to the input sequence.

    The CRC polynomials from Sec. 5.1 in :cite:p:`3GPPTS38212` are available:
    `{CRC24A, CRC24B, CRC24C, CRC16, CRC11, CRC6}`.

    :param crc_degree: Defines the CRC polynomial to be used. Can be any
        value from `{CRC24A, CRC24B, CRC24C, CRC16, CRC11, CRC6}`.
    :param k: Optional number of input bits. If specified, the generator
        matrix is pre-built during initialization, which is required for
        ``torch.compile`` compatibility. If not specified, the matrix is
        built lazily on first call.
    :param precision: Precision used for internal calculations and outputs.
        If `None`, :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for computation (e.g., 'cpu', 'cuda:0').
        If `None`, :attr:`~sionna.phy.config.Config.device` is used.

    :input bits: [..., k], `torch.float`.
        Binary tensor of arbitrary shape where the last dimension is
        `[..., k]`.

    :output x_crc: [..., k + crc_length], `torch.float`.
        Binary tensor containing CRC-encoded bits of the same shape as
        ``bits`` except the last dimension changes to
        `[..., k + crc_length]`.

    .. rubric:: Notes

    For performance enhancements, a generator-matrix-based implementation is
    used for fixed `k` instead of the more common shift register-based
    operations. Thus, the encoder must trigger an (internal) rebuild if `k`
    changes.

    .. rubric:: Examples


    .. code-block:: python

        import torch
        from sionna.phy.fec.crc import CRCEncoder

        encoder = CRCEncoder("CRC24A")
        bits = torch.randint(0, 2, (10, 100), dtype=torch.float32)
        encoded = encoder(bits)
        print(encoded.shape)
        # torch.Size([10, 124])
    """

    def __init__(
        self,
        crc_degree: str,
        *,
        k: Optional[int] = None,
        precision: Optional[str] = None,
        device: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(precision=precision, device=device, **kwargs)

        if not isinstance(crc_degree, str):
            raise TypeError("crc_degree must be a string.")
        self._crc_degree = crc_degree

        # Init 5G CRC polynomial
        self._crc_pol, self._crc_length = self._select_crc_pol(self._crc_degree)

        self._k: Optional[int] = None
        self._n: Optional[int] = None
        # Register buffer placeholder for CUDAGraph compatibility
        self.register_buffer("_g_mat_crc", None)
        self._fixed_k: bool = False  # True if k was pre-specified

        # Pre-build generator matrix if k is specified
        # (required for torch.compile compatibility)
        if k is not None:
            self.build((k,))
            self._built = True  # Mark as built to skip lazy init in __call__
            self._fixed_k = True

    @property
    def crc_degree(self) -> str:
        """CRC degree as string."""
        return self._crc_degree

    @property
    def crc_length(self) -> int:
        """Length of CRC. Equals number of CRC parity bits."""
        return self._crc_length

    @property
    def crc_pol(self) -> np.ndarray:
        """CRC polynomial in binary representation."""
        return self._crc_pol

    @property
    def k(self) -> Optional[int]:
        """Number of information bits per codeword."""
        if self._k is None:
            warnings.warn(
                "CRC encoder is not initialized yet. "
                "Input dimensions are unknown.",
                RuntimeWarning,
                stacklevel=2,
            )
        return self._k

    @property
    def n(self) -> Optional[int]:
        """Number of codeword bits after CRC encoding."""
        if self._n is None:
            warnings.warn(
                "CRC encoder is not initialized yet. "
                "Output dimensions are unknown.",
                RuntimeWarning,
                stacklevel=2,
            )
        return self._n

    def _select_crc_pol(self, crc_degree: str) -> Tuple[np.ndarray, int]:
        """Select 5G CRC polynomial according to Sec. 5.1 :cite:p:`3GPPTS38212`."""
        if crc_degree == "CRC24A":
            crc_length = 24
            crc_coeffs = [24, 23, 18, 17, 14, 11, 10, 7, 6, 5, 4, 3, 1, 0]
        elif crc_degree == "CRC24B":
            crc_length = 24
            crc_coeffs = [24, 23, 6, 5, 1, 0]
        elif crc_degree == "CRC24C":
            crc_length = 24
            crc_coeffs = [24, 23, 21, 20, 17, 15, 13, 12, 8, 4, 2, 1, 0]
        elif crc_degree == "CRC16":
            crc_length = 16
            crc_coeffs = [16, 12, 5, 0]
        elif crc_degree == "CRC11":
            crc_length = 11
            crc_coeffs = [11, 10, 9, 5, 0]
        elif crc_degree == "CRC6":
            crc_length = 6
            crc_coeffs = [6, 5, 0]
        else:
            raise ValueError("Invalid CRC Polynomial")

        # Invert array (MSB instead of LSB)
        crc_pol_inv = np.zeros(crc_length + 1)
        crc_pol_inv[[crc_length - c for c in crc_coeffs]] = 1

        return crc_pol_inv.astype(int), crc_length

    def _gen_crc_mat(self, k: int, pol_crc: np.ndarray) -> np.ndarray:
        """Build (dense) generator matrix for CRC parity bits.

        The principle idea is to treat the CRC as systematic linear code, i.e.,
        the generator matrix can be composed out of `k` linear independent
        (valid) codewords. For this, we CRC encode all `k` unit-vectors
        `[0,...1,...,0]` and build the generator matrix.
        To avoid `O(k^2)` complexity, we start with the last unit vector
        given as `[0,...,0,1]` and can generate the result for next vector
        `[0,...,1,0]` via another polynomial division of the remainder from the
        previous result. This allows to successively build the generator matrix
        at linear complexity `O(k)`.
        """
        crc_length = len(pol_crc) - 1
        g_mat = np.zeros([k, crc_length])

        x_crc = np.zeros(crc_length, dtype=int)
        x_crc[0] = 1
        for i in range(k):
            # Shift by one position
            x_crc = np.concatenate([x_crc, [0]])
            if x_crc[0] == 1:
                x_crc = np.bitwise_xor(x_crc, pol_crc)
            x_crc = x_crc[1:]
            g_mat[k - i - 1, :] = x_crc

        return g_mat

    @torch.compiler.disable
    def build(self, input_shape: Tuple[int, ...]) -> None:
        """Build the generator matrix.

        The CRC is always added to the last dimension of the input.

        Note: For torch.compile compatibility, use the ``k`` parameter in
        ``__init__`` to pre-build the generator matrix.
        """
        k = input_shape[-1]
        if k is None:
            raise ValueError("Shape of last dimension cannot be None.")
        g_mat_crc = self._gen_crc_mat(k, self.crc_pol)
        # Register as buffer for CUDAGraph compatibility
        self.register_buffer("_g_mat_crc", torch.tensor(
            g_mat_crc, dtype=self.dtype, device=self.device
        ))

        self._k = k
        self._n = k + g_mat_crc.shape[1]

    def call(self, bits: torch.Tensor) -> torch.Tensor:
        """Cyclic Redundancy Check (CRC) encoding function.

        This function adds the CRC parity bits to ``bits``.

        :param bits: Binary tensor of arbitrary shape `[..., k]`.

        :output x_out: CRC-encoded bits of shape
            `[..., k + crc_length]`.
        """
        # For torch.compile compatibility: if k was pre-specified, skip all
        # dynamic checks and just use the pre-built matrix
        if not self._fixed_k:
            # Dynamic mode: rebuild if needed
            input_k = bits.shape[-1]
            if self._g_mat_crc is None or input_k != self._k:
                self.build(tuple(bits.shape))

        # Note: as the code is systematic, we only encode the CRC positions
        # Thus, the generator matrix is non-sparse and a "full" matrix
        # multiplication is probably the fastest implementation
        x_exp = bits.unsqueeze(-2)  # row vector of shape [..., 1, k]

        # Matrix multiplication for CRC bits
        x_crc = torch.matmul(x_exp.to(self.dtype), self._g_mat_crc)

        # Take modulo 2 of x_crc
        x_crc = x_crc.to(torch.int32)
        x_crc = int_mod_2(x_crc)
        # Cast back to original dtype
        x_crc = x_crc.to(x_exp.dtype)

        x_conc = torch.cat([x_exp, x_crc], dim=-1)
        x_out = x_conc.squeeze(-2)

        return x_out


class CRCDecoder(Block):
    """Allows Cyclic Redundancy Check (CRC) verification and removes parity bits.

    The CRC polynomials from Sec. 5.1 in :cite:p:`3GPPTS38212` are available:
    `{CRC24A, CRC24B, CRC24C, CRC16, CRC11, CRC6}`.

    :param crc_encoder: An instance of
        :class:`~sionna.phy.fec.crc.CRCEncoder` associated with the
        CRCDecoder.
    :param precision: Precision used for internal calculations and outputs.
        If `None`, :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for computation (e.g., 'cpu', 'cuda:0').
        If `None`, :attr:`~sionna.phy.config.Config.device` is used.

    :input x_crc: [..., k + crc_length], `torch.float`.
        Binary tensor containing the CRC-encoded bits (the last
        `crc_length` bits are parity bits).

    :output bits: [..., k], `torch.float`.
        Binary tensor containing the information bit sequence without CRC
        parity bits.

    :output crc_valid: [..., 1], `torch.bool`.
        Boolean tensor containing the result of the CRC check per codeword.

    .. rubric:: Examples


    .. code-block:: python

        import torch
        from sionna.phy.fec.crc import CRCEncoder, CRCDecoder

        encoder = CRCEncoder("CRC24A")
        decoder = CRCDecoder(encoder)

        bits = torch.randint(0, 2, (10, 100), dtype=torch.float32)
        encoded = encoder(bits)
        decoded, crc_valid = decoder(encoded)
        print(decoded.shape, crc_valid.all())
        # torch.Size([10, 100]) tensor(True)
    """

    def __init__(
        self,
        crc_encoder: CRCEncoder,
        *,
        precision: Optional[str] = None,
        device: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(precision=precision, device=device, **kwargs)

        if not isinstance(crc_encoder, CRCEncoder):
            raise TypeError("crc_encoder must be a CRCEncoder instance.")
        self._encoder = crc_encoder

        # To detect changing input dimensions
        self._bit_shape: Optional[Tuple[int, ...]] = None

    @property
    def crc_degree(self) -> str:
        """CRC degree as string."""
        return self._encoder.crc_degree

    @property
    def encoder(self) -> CRCEncoder:
        """CRC Encoder used for internal validation."""
        return self._encoder

    def build(self, input_shape: Tuple[int, ...]) -> None:
        """Check shapes."""
        self._bit_shape = input_shape
        if input_shape[-1] < self._encoder.crc_length:
            raise ValueError(
                "Input length must be greater than or equal to the CRC length."
            )

    def call(
        self, x_crc: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Cyclic Redundancy Check (CRC) verification function.

        This function verifies the CRC of ``x_crc``. It returns the result of
        the CRC validation and removes parity bits from ``x_crc``.

        :param x_crc: Binary tensor of arbitrary shape
            `[..., k + crc_length]`.

        :output x_info: Information bits without CRC parity bits of shape
            `[..., k]`.

        :output crc_valid: Result of the CRC validation for each codeword
            of shape `[..., 1]`.
        """
        if self._bit_shape is None or x_crc.shape[-1] != self._bit_shape[-1]:
            self.build(tuple(x_crc.shape))

        # Extract information bits and received CRC parity bits
        x_info = x_crc[..., : -self._encoder.crc_length]
        x_parity_received = x_crc[..., -self._encoder.crc_length :]

        # Re-encode information bits to compute expected CRC parity
        x_parity_computed = self._encoder(x_info)[..., -self._encoder.crc_length :]

        # Cast output to desired precision as encoder can have a different
        # precision
        x_parity_computed = x_parity_computed.to(self.dtype)

        # Compare received parity with computed parity
        # CRC is valid if all parity bits match
        crc_valid = (x_parity_received == x_parity_computed).all(dim=-1, keepdim=True)

        return x_info, crc_valid

