#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Blocks for scrambling, descrambling and utility functions."""

from typing import Optional, Sequence, Union
import warnings
import torch

from sionna._validation import check_binary
from sionna.phy import config, Block
from sionna.phy.utils import expand_to_rank


__all__ = ["Scrambler", "TB5GScrambler", "Descrambler"]


class Scrambler(Block):
    r"""Randomly flips the state/sign of a sequence of bits or LLRs,
    respectively.

    :param seed: Defines the initial state of the pseudo random generator
        to generate the scrambling sequence. If `None`, a random integer
        will be generated. Only used when ``keep_state`` is `True`.
    :param keep_batch_constant: If `True`, all samples in the batch are
        scrambled with the same scrambling sequence. Otherwise, per sample a
        random sequence is generated.
    :param sequence: If provided, the seed will be ignored and the explicit
        scrambling sequence is used. Must be an array of 0s and 1s. Shape
        must be broadcastable to ``x``.
    :param binary: Indicates whether bit-sequence should be flipped
        (i.e., binary operations are performed) or the signs should be
        flipped (i.e., soft-value/LLR domain-based).
    :param keep_state: Indicates whether the scrambling sequence should be
        kept constant.
    :param precision: Precision used for internal calculations and outputs.
        If `None`, :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for computation (e.g., 'cpu', 'cuda:0').
        If `None`, :attr:`~sionna.phy.config.Config.device` is used.

    :input x: torch.Tensor.
        Tensor of arbitrary shape.

    :input seed: `None` | `int`.
        An integer defining the state of the random number generator.
        If explicitly given, the global internal seed is replaced by this
        seed. Can be used to realize random scrambler/descrambler pairs
        (call with same random seed).

    :input binary: `None` | `bool`.
        Overrules the init parameter ``binary`` if explicitly given.
        Indicates whether bit-sequence should be flipped
        (i.e., binary operations are performed) or the signs should be
        flipped (i.e., soft-value/LLR domain-based).

    :output x_out: torch.Tensor.
        Tensor of same shape as ``x``.

    .. rubric:: Notes

    For inverse scrambling, the same scrambler can be reused (as the values
    are flipped again, i.e., result in the original state). However,
    ``keep_state`` must be set to `True` as a new sequence would be generated
    otherwise.

    The scrambler block is stateless, i.e., the seed is either random
    during each call or must be explicitly provided during init/call.
    If the seed is provided in the init function, this fixed seed is used
    for all calls. However, an explicit seed can be provided during
    the call function to realize true random states.

    Scrambling is typically used to ensure equal likely 0 and 1 for
    sources with unequal bit probabilities. As we have a perfect source in
    the simulations, this is not required. However, for all-zero codeword
    simulations and higher-order modulation, so-called "channel-adaptation"
    :cite:p:`Pfister03` is required.

    .. rubric:: Examples


    .. code-block:: python

        import torch
        from sionna.phy.fec.scrambling import Scrambler

        scrambler = Scrambler(seed=42, keep_state=True)
        bits = torch.randint(0, 2, (10, 100), dtype=torch.float32)
        scrambled = scrambler(bits)
        unscrambled = scrambler(scrambled)  # Re-use for descrambling
        assert torch.allclose(bits, unscrambled)
    """

    def __init__(
        self,
        seed: Optional[int] = None,
        keep_batch_constant: bool = False,
        binary: bool = True,
        sequence: Optional[torch.Tensor] = None,
        keep_state: bool = True,
        *,
        precision: Optional[str] = None,
        device: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(precision=precision, device=device, **kwargs)

        if not isinstance(keep_batch_constant, bool):
            raise TypeError("keep_batch_constant must be bool.")
        self._keep_batch_constant = keep_batch_constant

        if seed is not None:
            if sequence is not None:
                warnings.warn(
                    "Explicit scrambling sequence provided. "
                    "Seed will be ignored.",
                    UserWarning,
                    stacklevel=2,
                )
            if not isinstance(seed, int):
                raise TypeError("seed must be int.")
        else:
            seed = int(config.np_rng.integers(0, 2**31 - 1))

        if not isinstance(binary, bool):
            raise TypeError("binary must be bool.")
        self._binary = binary

        if not isinstance(keep_state, bool):
            raise TypeError("keep_state must be bool.")
        self._keep_state = keep_state

        # If keep_state==True this seed is used to generate scrambling sequences
        self._seed = seed

        # If an explicit sequence is provided the above parameters will be ignored
        self._sequence: Optional[torch.Tensor] = None
        if sequence is not None:
            sequence = sequence.to(dtype=self.dtype, device=self.device)
            check_binary(
                sequence,
                name="sequence",
                message="Scrambling sequence must be binary.",
            )
            self._sequence = sequence

    @property
    def seed(self) -> int:
        """Seed used to generate random sequence."""
        return self._seed

    @property
    def keep_state(self) -> bool:
        """Indicates if new random sequences are used per call."""
        return self._keep_state

    @property
    def sequence(self) -> Optional[torch.Tensor]:
        """Explicit scrambling sequence if provided."""
        return self._sequence

    def _generate_scrambling(
        self, input_shape: torch.Size, seed: int
    ) -> torch.Tensor:
        r"""Generates a random sequence of `0`\ s and `1`\ s that can be used
        to initialize a scrambler and updates the internal attributes."""
        # Create a generator with the given seed
        gen = torch.Generator(device=self.device)
        gen.manual_seed(seed)

        if self._keep_batch_constant:
            input_shape_no_bs = input_shape[1:]
            seq = torch.randint(
                0,
                2,
                input_shape_no_bs,
                generator=gen,
                device=self.device,
                dtype=torch.int32,
            )
            # Expand batch dim so it can be broadcasted
            seq = seq.unsqueeze(0)
        else:
            seq = torch.randint(
                0,
                2,
                input_shape,
                generator=gen,
                device=self.device,
                dtype=torch.int32,
            )

        return seq.to(self.dtype)

    def call(
        self,
        x: torch.Tensor,
        seed: Optional[int] = None,
        binary: Optional[bool] = None,
    ) -> torch.Tensor:
        """Scrambling function.

        This function returns the scrambled version of ``x``.

        :param x: Tensor of arbitrary shape.
        :param seed: An integer defining the state of the random number
            generator. If explicitly given, the global internal seed is
            replaced by this seed. Can be used to realize random
            scrambler/descrambler pairs (call with same random seed).
        :param binary: Overrules the init parameter ``binary`` if explicitly
            given. Indicates whether bit-sequence should be flipped (i.e.,
            binary operations are performed) or the signs should be flipped
            (i.e., soft-value/LLR domain-based).
        """
        if binary is None:
            binary = self._binary
        else:
            if not isinstance(binary, bool):
                raise TypeError("binary must be bool.")

        input_shape = x.shape
        input_dtype = x.dtype
        x = x.to(self.dtype)

        # Determine seed to use
        if seed is not None:
            use_seed = seed
        elif self._keep_state:
            use_seed = self._seed
        else:
            # Generate new seed for each call
            use_seed = int(config.np_rng.integers(0, 2**31 - 1))

        # Apply sequence if explicit sequence is provided
        if self._sequence is not None:
            rand_seq = self._sequence
        else:
            rand_seq = self._generate_scrambling(input_shape, use_seed)

        if binary:
            # Flip bits by subtraction and map -1 to 1 via abs(.) operator
            x_out = torch.abs(x - rand_seq)
        else:
            rand_seq_bipol = -2 * rand_seq + 1
            x_out = x * rand_seq_bipol

        return x_out.to(input_dtype)


class TB5GScrambler(Block):
    r"""5G NR Scrambler for PUSCH and PDSCH channel.

    Implements the pseudo-random bit scrambling as defined in
    :cite:p:`3GPPTS38211` Sec. 6.3.1.1 for the "PUSCH" channel and in
    Sec. 7.3.1.1 for the "PDSCH" channel.

    Only for the "PDSCH" channel, the scrambler can be configured for two
    codeword transmission mode. Hereby, ``codeword_index`` corresponds to the
    index of the codeword to be scrambled.

    If ``n_rnti`` is a list of ints, the scrambler assumes that the second
    last axis contains ``len(n_rnti)`` elements. This allows independent
    scrambling for multiple independent streams.

    :param n_rnti: RNTI identifier provided by higher layer. Defaults to 1
        and must be in range `[0, 65535]`. If a list is provided, every list
        element defines a scrambling sequence for multiple independent streams.
    :param n_id: Scrambling ID related to cell id and provided by higher
        layer. Defaults to 1 and must be in range `[0, 1023]`. If a list is
        provided, every list element defines a scrambling sequence for
        multiple independent streams.
    :param binary: Indicates whether bit-sequence should be flipped
        (i.e., binary operations are performed) or the signs should be
        flipped (i.e., soft-value/LLR domain-based).
    :param channel_type: Can be either ``'PUSCH'`` or ``'PDSCH'``.
    :param codeword_index: Scrambler can be configured for two codeword
        transmission. ``codeword_index`` can be either 0 or 1.
    :param precision: Precision used for internal calculations and outputs.
        If `None`, :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for computation (e.g., 'cpu', 'cuda:0').
        If `None`, :attr:`~sionna.phy.config.Config.device` is used.

    :input x: torch.Tensor.
        Tensor of arbitrary shape. If ``n_rnti`` and ``n_id`` are a list,
        it is assumed that ``x`` has shape ``[..., num_streams, n]``
        where ``num_streams = len(n_rnti)``.

    :input binary: `None` | `bool`.
        Overrules the init parameter ``binary`` if explicitly given.
        Indicates whether bit-sequence should be flipped
        (i.e., binary operations are performed) or the signs should be
        flipped (i.e., soft-value/LLR domain-based).

    :output x_out: torch.Tensor.
        Tensor of same shape as ``x``.

    .. rubric:: Notes

    The parameters radio network temporary identifier (RNTI) ``n_rnti`` and
    the datascrambling ID ``n_id`` are usually provided by the higher layer
    protocols.

    For inverse scrambling, the same scrambler can be reused (as the values
    are flipped again, i.e., result in the original state).

    .. rubric:: Examples


    .. code-block:: python

        import torch
        from sionna.phy.fec.scrambling import TB5GScrambler

        scrambler = TB5GScrambler(n_rnti=1, n_id=1)
        bits = torch.randint(0, 2, (10, 100), dtype=torch.float32)
        scrambled = scrambler(bits)
        unscrambled = scrambler(scrambled)  # Re-use for descrambling
        assert torch.allclose(bits, unscrambled)
    """

    def __init__(
        self,
        n_rnti: Union[int, Sequence[int]] = 1,
        n_id: Union[int, Sequence[int]] = 1,
        binary: bool = True,
        channel_type: str = "PUSCH",
        codeword_index: int = 0,
        *,
        precision: Optional[str] = None,
        device: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(precision=precision, device=device, **kwargs)

        if not isinstance(binary, bool):
            raise TypeError("binary must be bool.")
        self._binary = binary

        if channel_type not in ("PDSCH", "PUSCH"):
            raise TypeError("Unsupported channel_type.")

        if codeword_index not in (0, 1):
            raise ValueError("codeword_index must be 0 or 1.")

        self._input_shape: Optional[torch.Size] = None

        # Allow list input for independent multi-stream scrambling
        if isinstance(n_rnti, (list, tuple)):
            if not isinstance(n_id, (list, tuple)):
                raise TypeError("n_id must be a list of same length as n_rnti.")

            if len(n_rnti) != len(n_id):
                raise ValueError("n_rnti and n_id must be of same length.")

            self._multi_stream = True
            n_rnti = list(n_rnti)
            n_id = list(n_id)
        else:
            n_rnti = [n_rnti]
            n_id = [n_id]
            self._multi_stream = False

        # Check all entries for consistency
        for idx, (nr, ni) in enumerate(zip(n_rnti, n_id)):
            if nr % 1 != 0:
                raise ValueError("n_rnti must be integer.")
            if nr not in range(2**16):
                raise ValueError("n_rnti must be in [0, 65535].")
            n_rnti[idx] = int(nr)

            if ni % 1 != 0:
                raise ValueError("n_id must be integer.")
            if ni not in range(2**10):
                raise ValueError("n_id must be in [0, 1023].")
            n_id[idx] = int(ni)

        self._c_init = []
        if channel_type == "PUSCH":
            # Defined in 6.3.1.1 in 38.211
            for nr, ni in zip(n_rnti, n_id):
                self._c_init.append(nr * 2**15 + ni)
        elif channel_type == "PDSCH":
            # Defined in 7.3.1.1 in 38.211
            for nr, ni in zip(n_rnti, n_id):
                self._c_init.append(nr * 2**15 + codeword_index * 2**14 + ni)

        self._sequence: Optional[torch.Tensor] = None

    @property
    def keep_state(self) -> bool:
        """Required for descrambler, is always `True` for the TB5GScrambler."""
        return True

    @torch.compiler.disable
    def _generate_scrambling(self, input_shape: torch.Size) -> torch.Tensor:
        r"""Returns random sequence of `0`\ s and `1`\ s following
        :cite:p:`3GPPTS38211`.

        Note: This method is decorated with ``@torch.compiler.disable``
        because it calls ``generate_prng_seq`` which uses NumPy operations
        that cannot be traced by ``torch.compile``.
        """
        # Lazy import to avoid circular dependency
        from sionna.phy.nr.utils import generate_prng_seq

        seq = generate_prng_seq(input_shape[-1], self._c_init[0])
        seq = torch.tensor(seq, dtype=self.dtype, device=self.device)
        seq = expand_to_rank(seq, len(input_shape), axis=0)

        if self._multi_stream:
            for c in self._c_init[1:]:
                s = generate_prng_seq(input_shape[-1], c)
                s = torch.tensor(s, dtype=self.dtype, device=self.device)
                s = expand_to_rank(s, len(input_shape), axis=0)
                seq = torch.cat([seq, s], dim=-2)

        return seq

    def build(self, input_shape: torch.Size, **kwargs) -> None:
        """Initialize pseudo-random scrambling sequence."""
        # kwargs may contain 'binary' from call, which we ignore here
        self._input_shape = input_shape

        # In multi-stream mode, axis=-2 must have dimension=len(c_init)
        if self._multi_stream:
            if input_shape[-2] != len(self._c_init):
                raise ValueError(
                    "Dimension of axis=-2 must be equal to len(n_rnti)."
                )

        self._sequence = self._generate_scrambling(input_shape)

    def call(
        self,
        x: torch.Tensor,
        binary: Optional[bool] = None,
    ) -> torch.Tensor:
        """Scrambling function.

        This function returns the scrambled version of ``x``.

        :param x: Tensor of arbitrary shape.
        :param binary: Overrules the init parameter ``binary`` if explicitly
            given.
        """
        if binary is None:
            binary = self._binary
        else:
            if not isinstance(binary, bool):
                raise TypeError("binary must be bool.")

        if self._input_shape is None or x.shape[-1] != self._input_shape[-1]:
            self.build(x.shape)

        input_dtype = x.dtype
        x = x.to(self.dtype)

        if binary:
            # Flip bits by subtraction and map -1 to 1 via abs(.) operator
            x_out = torch.abs(x - self._sequence)
        else:
            rand_seq_bipol = -2 * self._sequence + 1
            x_out = x * rand_seq_bipol

        return x_out.to(input_dtype)


class Descrambler(Block):
    r"""Descrambler for a given scrambler.

    :param scrambler: Associated
        :class:`~sionna.phy.fec.scrambling.Scrambler` or
        :class:`~sionna.phy.fec.scrambling.TB5GScrambler` instance which
        should be descrambled.
    :param binary: Indicates whether bit-sequence should be flipped (i.e.,
        binary operations are performed) or the signs should be flipped
        (i.e., soft-value/LLR domain-based).
    :param precision: Precision used for internal calculations and outputs.
        If `None`, uses same precision as associated scrambler.
    :param device: Device for computation (e.g., 'cpu', 'cuda:0').
        If `None`, uses same device as associated scrambler.

    :input x: torch.Tensor.
        Tensor of arbitrary shape.

    :input seed: `int`.
        An integer defining the state of the random number generator.
        If explicitly given, the global internal seed is replaced by this
        seed. Can be used to realize random scrambler/descrambler pairs
        (call with same random seed).

    :output x_out: torch.Tensor.
        Tensor of same shape as ``x``.

    .. rubric:: Examples


    .. code-block:: python

        import torch
        from sionna.phy.fec.scrambling import Scrambler, Descrambler

        scrambler = Scrambler(seed=42, keep_state=True)
        descrambler = Descrambler(scrambler, binary=False)

        llrs = torch.randn(10, 100)
        scrambled = scrambler(llrs, binary=False)
        unscrambled = descrambler(scrambled)
        assert torch.allclose(llrs, unscrambled)
    """

    def __init__(
        self,
        scrambler: Union[Scrambler, TB5GScrambler],
        binary: bool = True,
        *,
        precision: Optional[str] = None,
        device: Optional[str] = None,
        **kwargs,
    ):
        if not isinstance(scrambler, (Scrambler, TB5GScrambler)):
            raise TypeError("scrambler must be an instance of Scrambler.")

        # If precision is None, use same precision as associated scrambler
        if precision is None:
            precision = scrambler.precision
        if device is None:
            device = scrambler.device

        super().__init__(precision=precision, device=device, **kwargs)

        # Must assign scrambler after super().__init__() since it's an nn.Module
        self._scrambler = scrambler

        if not isinstance(binary, bool):
            raise TypeError("binary must be bool.")
        self._binary = binary

        if self._scrambler.keep_state is False:
            warnings.warn(
                "Scrambler uses random sequences that cannot be "
                "accessed by descrambler. Please use keep_state=True and "
                "provide explicit random seed as input to call function.",
                UserWarning,
                stacklevel=2,
            )

        if self._scrambler.precision != self.precision:
            warnings.warn(
                "Scrambler and descrambler are using different precision. "
                "This will cause an internal implicit cast.",
                UserWarning,
                stacklevel=2,
            )

    @property
    def scrambler(self) -> Union[Scrambler, TB5GScrambler]:
        """Associated scrambler instance."""
        return self._scrambler

    def call(
        self,
        x: torch.Tensor,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """Descrambling function.

        This function returns the descrambled version of ``x``.

        :param x: Tensor of arbitrary shape.
        :param seed: An integer defining the state of the random number
            generator. If explicitly given, the global internal seed is
            replaced by this seed. Can be used to realize random
            scrambler/descrambler pairs (must be called with same random
            seed).
        """
        input_dt = x.dtype
        x = x.to(self.dtype)

        if isinstance(self._scrambler, Scrambler):
            if seed is not None:
                s = seed
            else:
                s = self._scrambler.seed  # Use seed from associated scrambler
            x_out = self._scrambler(x, seed=s, binary=self._binary)
        elif isinstance(self._scrambler, TB5GScrambler):
            x_out = self._scrambler(x, binary=self._binary)
        else:
            raise TypeError("Unknown Scrambler type.")

        return x_out.to(input_dt)

