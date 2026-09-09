#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Blocks for interleaving and utility functions."""

from typing import Optional, Union
import warnings
import numpy as np
import torch
from importlib_resources import files, as_file

from sionna._validation import check_instance
from sionna.phy import config, Block

__all__ = [
    "RowColumnInterleaver",
    "RandomInterleaver",
    "Turbo3GPPInterleaver",
    "Deinterleaver",
]


class RowColumnInterleaver(Block):
    r"""Interleaves a sequence of inputs via row/column swapping.

    :param row_depth: The row depth, i.e., how many values per row can be
        stored.
    :param axis: The dimension that should be interleaved.
    :param inverse: If `True`, the inverse permutation is performed.
    :param precision: Precision used for internal calculations and outputs.
        If `None`, :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for computation (e.g., 'cpu', 'cuda:0').
        If `None`, :attr:`~sionna.phy.config.Config.device` is used.

    :input x: torch.Tensor.
        Tensor of arbitrary shape and arbitrary dtype.

    :output x_int: torch.Tensor.
        Tensor of same shape and dtype as ``x``.

    .. rubric:: Notes

    If the sequence length is not a multiple of ``row_depth``, additional
    filler bits are used for the last row that will be removed internally.
    However, for the last positions the interleaving distance may be
    slightly degraded.

    .. rubric:: Examples


    .. code-block:: python

        import torch
        from sionna.phy.fec.interleaving import RowColumnInterleaver

        interleaver = RowColumnInterleaver(row_depth=4)
        x = torch.arange(12).reshape(1, 12).float()
        y = interleaver(x)
        print(y)
        # tensor([[ 0.,  4.,  8.,  1.,  5.,  9.,  2.,  6., 10.,  3.,  7., 11.]])
    """

    def __init__(
        self,
        row_depth: int,
        axis: int = -1,
        inverse: bool = False,
        precision: Optional[str] = None,
        device: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(precision=precision, device=device, **kwargs)

        check_instance(axis, int, name="axis", message="axis must be int.")
        self._axis = axis

        check_instance(
            row_depth,
            int,
            name="row_depth",
            message="row_depth must be int.",
        )
        self._row_depth = row_depth

        check_instance(
            inverse,
            bool,
            name="inverse",
            message="inverse must be bool.",
        )
        self._inverse = inverse

        # Permutation sequences initialized during build
        self._perm_seq: Optional[torch.Tensor] = None
        self._perm_seq_inv: Optional[torch.Tensor] = None

        # Required for associated deinterleaver
        self._keep_state = True

    @property
    def axis(self) -> int:
        """Axis to be permuted."""
        return self._axis

    @property
    def row_depth(self) -> int:
        """Row depth of the row-column interleaver."""
        return self._row_depth

    @property
    def perm_seq(self) -> Optional[torch.Tensor]:
        """Permutation sequence."""
        return self._perm_seq

    @property
    def perm_seq_inv(self) -> Optional[torch.Tensor]:
        """Inverse permutation sequence."""
        return self._perm_seq_inv

    @property
    def keep_state(self) -> bool:
        """Row-column interleaver always uses the same internal state."""
        return True

    def _generate_perm_rc(
        self, n_seq: int, r_depth: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generates a row/column permutation to initialize an RC-interleaver.

        If required, last positions use filler positions.

        :param n_seq: Sequence length to interleave.
        :param r_depth: Depth of the interleaver.

        :output perm_seq: Forward permutation tensor.

        :output perm_seq_inv: Inverse permutation tensor.
        """
        # Round to next multiple of r_depth
        n = int(np.ceil(n_seq / r_depth) * r_depth)
        nb_rows = n // r_depth

        ind = torch.arange(n, dtype=torch.int64, device=self.device)

        # Rearrange in row/column format
        ind_rc = ind.reshape(nb_rows, -1)

        # Interleave via row/column swapping (transpose)
        ind_cr = ind_rc.t()

        # Read out indices in column/row ordering
        perm_seq_filler = ind_cr.reshape(-1)

        # Remove filler positions
        mask = perm_seq_filler < n_seq
        perm_seq = perm_seq_filler[mask]
        perm_seq_inv = torch.argsort(perm_seq)

        return perm_seq, perm_seq_inv

    def build(self, input_shape: tuple) -> None:
        """Build block and check dimensions.

        :param input_shape: Shape of input tensor.
        """
        if self._axis >= len(input_shape) or self._axis < -len(input_shape):
            raise ValueError("Axis does not match input shape.")

        # Normalize negative axis
        axis = self._axis if self._axis >= 0 else len(input_shape) + self._axis

        # Interleaver can't build pattern for dynamic shapes
        if input_shape[axis] is None:
            raise ValueError("Permutation axis cannot be None (dynamic).")

        # Generate permutation patterns
        p, pi = self._generate_perm_rc(input_shape[axis], self._row_depth)
        self._perm_seq = p
        self._perm_seq_inv = pi

    def call(
        self,
        x: torch.Tensor,
        /,
        *,
        inverse: Optional[bool] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Interleaving function.

        This function returns the permuted version of ``x``.

        :param x: Tensor of arbitrary shape.
        :param inverse: If provided, overrides the init parameter.

        :output x_int: Interleaved tensor of same shape as ``x``.
        """
        input_shape = x.shape

        # Normalize axis
        axis = self._axis if self._axis >= 0 else len(input_shape) + self._axis

        # Re-init if shape has changed
        if self._perm_seq is None or x.shape[axis] != self._perm_seq.shape[0]:
            self._built = False
            self.build(x.shape)
            self._built = True

        # Use internal value if not explicitly provided
        if inverse is None:
            inverse = self._inverse

        # Ensure permutation is on the same device as input
        perm = self._perm_seq_inv if inverse else self._perm_seq
        if perm.device != x.device:
            perm = perm.to(x.device)

        x_int = torch.index_select(x, axis, perm)
        return x_int


class RandomInterleaver(Block):
    r"""Random interleaver permuting a sequence of input symbols.

    :param seed: Integer defining the random seed used if ``keep_state`` is
        `True`.
    :param keep_batch_constant: If `True`, each sample in the batch uses the
        same permutation. Otherwise, unique permutations per batch sample
        are generated (slower).
    :param inverse: If `True`, the inverse permutation is performed.
    :param keep_state: If `True`, the permutation is fixed for multiple calls
        (defined by ``seed`` attribute).
    :param axis: The dimension that should be interleaved.
        First dimension (``axis=0``) is not allowed.
    :param precision: Precision used for internal calculations and outputs.
        If `None`, :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for computation (e.g., 'cpu', 'cuda:0').
        If `None`, :attr:`~sionna.phy.config.Config.device` is used.

    :input x: torch.Tensor.
        Tensor of arbitrary shape and dtype.

    :input seed: `int`.
        An integer defining the state of the random number
        generator. If explicitly given, the global internal seed is
        replaced by this seed. Can be used to realize random
        interleaver/deinterleaver pairs (call with same random seed).

    :output x_int: torch.Tensor.
        Tensor of same shape and dtype as the input ``x``.

    .. rubric:: Notes

    The interleaver block is stateless, i.e., the seed is either random
    during each call or must be explicitly provided during init/call.

    This is NOT the 5G interleaver sequence.

    .. rubric:: Examples


    .. code-block:: python

        import torch
        from sionna.phy.fec.interleaving import RandomInterleaver

        interleaver = RandomInterleaver(seed=42, keep_state=True)
        x = torch.arange(10).reshape(1, 10).float()
        y = interleaver(x)
        print(y)
    """

    def __init__(
        self,
        seed: Optional[int] = None,
        keep_batch_constant: bool = True,
        inverse: bool = False,
        keep_state: bool = True,
        axis: int = -1,
        precision: Optional[str] = None,
        device: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(precision=precision, device=device, **kwargs)

        check_instance(
            keep_batch_constant,
            bool,
            name="keep_batch_constant",
            message="keep_batch_constant must be bool.",
        )
        self._keep_batch_constant = keep_batch_constant

        check_instance(axis, int, name="axis", message="axis must be int.")
        self._axis = axis

        if seed is not None:
            check_instance(seed, int, name="seed", message="seed must be int.")
        else:
            # Generate random seed if no value is provided
            seed = self.py_rng.randint(0, 2**31 - 1)

        self._seed = seed

        check_instance(
            inverse,
            bool,
            name="inverse",
            message="inverse must be boolean.",
        )
        self._inverse = inverse

        check_instance(
            keep_state,
            bool,
            name="keep_state",
            message="keep_state must be boolean.",
        )
        self._keep_state = keep_state

        if self._keep_state is False and self._inverse is True:
            warnings.warn(
                "keep_state=False and, thus, a new realization of "
                "the interleaver is generated during each call. Thus, "
                "the inverse interleaver does not correspond to a previous "
                "interleaver call.",
                UserWarning,
                stacklevel=2,
            )

    @property
    def seed(self) -> int:
        """Seed to generate random sequence."""
        return self._seed

    @property
    def axis(self) -> int:
        """Axis to be permuted."""
        return self._axis

    @property
    def keep_state(self) -> bool:
        """Generate new random seed per call."""
        return self._keep_state

    def find_s_min(
        self, seed: int, seq_length: int, s_min_stop: int = 0
    ) -> int:
        r"""Find :math:`S` parameter such that :math:`\pi(i)-\pi(j)>S` for all
        :math:`i-j<S`. This can be used to find optimized interleaver patterns.

        ``s_min_stop`` is an additional stopping condition, i.e., stop if
        current :math:`S` is already smaller than ``s_min_stop``.

        :param seed: Seed to draw random permutation that shall be analyzed.
        :param seq_length: Length of permutation sequence to be analyzed.
        :param s_min_stop: Enables early stop if already current s_min <
            ``s_min_stop``.

        :output s_min: The S-parameter for the given ``seed``.
        """
        check_instance(seed, int, name="seed", message="seed must be int.")
        check_instance(
            seq_length,
            int,
            name="seq_length",
            message="seq_length must be int.",
        )
        check_instance(
            s_min_stop,
            int,
            name="s_min_stop",
            message="s_min_stop must be int.",
        )

        perm_seq = self._generate_perm_full(seed, seq_length, batch_size=1)
        perm_seq = perm_seq.squeeze(0).cpu().numpy()

        s_min = seq_length
        for i in range(len(perm_seq)):
            for j in range(-s_min, s_min, 1):
                if j == 0:
                    continue
                if 0 <= i + j < seq_length:
                    d = np.abs(perm_seq[i] - perm_seq[i + j])
                    if d <= np.abs(j):
                        s_min = min(s_min, np.abs(j))
                    if d < s_min and np.abs(j) < s_min:
                        s_min = min(s_min, d)
            # Early stop
            if s_min <= s_min_stop:
                break

        return int(s_min)

    def _generate_perm_full(
        self,
        seed: int,
        seq_length: int,
        batch_size: int,
        inverse: bool = False,
        target_device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Generates a random permutation for the interleaver.

        :param seed: Seed for the random number generator.
        :param seq_length: Length of the sequence to be permuted.
        :param batch_size: Number of independent permutations.
        :param inverse: If `True`, the inverse permutation is generated.
        :param target_device: Device for the output tensor.

        :output perm_seq: Permutation tensor of shape
            ``[batch_size, seq_length]``.
        """
        device = target_device if target_device is not None else self.device
        # Generator must be on CPU for CUDA tensors
        gen_device = "cpu" if str(device).startswith("cuda") else device
        gen = torch.Generator(device=gen_device)
        gen.manual_seed(seed)

        rand_seq = torch.rand(
            batch_size,
            seq_length,
            generator=gen,
            device=gen_device,
            dtype=torch.float32,
        )

        perm_seq = torch.argsort(rand_seq, dim=-1)

        if inverse:
            perm_seq = torch.argsort(perm_seq, dim=-1)

        # Move to target device if needed
        if perm_seq.device != device:
            perm_seq = perm_seq.to(device)

        return perm_seq

    def build(self, input_shape: tuple, **kwargs) -> None:
        """Build block and check consistency of dimensions.

        :param input_shape: Shape of input tensor.
        """
        if self._axis >= len(input_shape) or self._axis < -len(input_shape):
            raise ValueError("Axis does not match input shape.")

    def call(
        self,
        x: torch.Tensor,
        /,
        *,
        seed: Optional[int] = None,
        inverse: Optional[bool] = None,
    ) -> torch.Tensor:
        """Interleaving function.

        This function returns the permuted version of ``x``.

        :param x: Tensor of arbitrary shape.
        :param seed: An integer defining the state of the random number
            generator. If explicitly given, the global internal seed is
            replaced by this seed. Can be used to realize random
            interleaver/deinterleaver pairs (call with same random seed).
        :param inverse: If provided, overrides the init parameter.

        :output x_int: Interleaved tensor of same shape as ``x``.

        :raises TypeError: If ``inverse`` is not `None` or `bool`.
        :raises ValueError: If inverse interleaving is requested with
            ``keep_state=False`` without explicitly providing the seed.

        .. rubric:: Notes

        In case of inverse interleaving (e.g., at the receiver),
        ``keep_state`` should be `True` as otherwise a new permutation is
        generated and the output is not equal to the original sequence.
        Alternatively, an explicit seed must be provided as function
        argument.
        """
        input_shape = x.shape

        if inverse is None:
            inverse = self._inverse
        else:
            check_instance(
                inverse,
                bool,
                name="inverse",
                message="inverse must be bool.",
            )

        # Determine seed to use
        if seed is not None:
            use_seed = seed
        elif self._keep_state:
            use_seed = self._seed
        else:
            if inverse:
                raise ValueError(
                    "Inverse interleaving not possible for "
                    "random seeds per call (keep_state=False) without "
                    "explicitly providing the seed as inputs."
                )
            # Generate new seed for each call
            use_seed = self.py_rng.randint(0, 2**31 - 1)

        # Normalize axis
        axis = self._axis if self._axis >= 0 else len(input_shape) + self._axis

        # Select batch size for permutation generation
        if self._keep_batch_constant:
            batch_size = 1
        else:
            # Special case: no batch dim
            if len(x.shape) == 1:
                batch_size = 1
            else:
                batch_size = x.shape[0]

        perm_seq = self._generate_perm_full(
            use_seed, x.shape[axis], batch_size, inverse, target_device=x.device
        )

        if self._keep_batch_constant:
            # Broadcast single sequence over complete batch
            perm_seq = perm_seq.squeeze(0)
            x_int = torch.index_select(x, axis, perm_seq)
        elif len(x.shape) == 1:
            # Special case: no batch dim
            perm_seq = perm_seq.squeeze(0)
            x_int = torch.index_select(x, axis, perm_seq)
        elif len(x.shape) == 2:
            # 2D case: batch x seq_len - use gather directly
            # perm_seq is [batch_size, seq_len]
            x_int = torch.gather(x, axis, perm_seq)
        else:
            # Per-batch permutation using gather for higher dims
            # Move axis to position 1 for gather
            x_t = x.movedim(axis, 1)
            # perm_seq is [batch_size, seq_len], expand to match x_t shape
            perm_expanded = perm_seq
            for _ in range(len(x_t.shape) - 2):
                perm_expanded = perm_expanded.unsqueeze(-1)
            perm_expanded = perm_expanded.expand(-1, -1, *x_t.shape[2:])
            x_int = torch.gather(x_t, 1, perm_expanded)
            x_int = x_int.movedim(1, axis)

        return x_int


class Turbo3GPPInterleaver(Block):
    """Interleaver for 3GPP Turbo codes.

    Interleaver as used in the 3GPP Turbo codes :cite:p:`3GPPTS36212` and, thus,
    the maximum length is given as 6144 elements (only for the dimension as
    specified by ``axis``).

    :param inverse: If `True`, the inverse permutation is performed.
    :param axis: The dimension that should be interleaved.
        First dimension (``axis=0``) is not allowed.
    :param precision: Precision used for internal calculations and outputs.
        If `None`, :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for computation (e.g., 'cpu', 'cuda:0').
        If `None`, :attr:`~sionna.phy.config.Config.device` is used.

    :input x: torch.Tensor.
        2+D tensor of arbitrary shape and dtype.

    :output x_int: torch.Tensor.
        2+D tensor of same shape and dtype as the input ``x``.

    .. rubric:: Notes

    Note that this implementation slightly deviates from the 3GPP
    standard :cite:p:`3GPPTS36212` in a sense that zero-padding is introduced
    for cases when the exact interleaver length is not supported by the
    standard.

    .. rubric:: Examples


    .. code-block:: python

        import torch
        from sionna.phy.fec.interleaving import Turbo3GPPInterleaver

        interleaver = Turbo3GPPInterleaver()
        x = torch.arange(40).reshape(1, 40).float()
        y = interleaver(x)
        print(y.shape)
        # torch.Size([1, 40])
    """

    def __init__(
        self,
        inverse: bool = False,
        axis: int = -1,
        precision: Optional[str] = None,
        device: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(precision=precision, device=device, **kwargs)

        check_instance(axis, int, name="axis", message="axis must be int.")
        self._axis = axis
        self._keep_state = True  # Required for deinterleaver

        check_instance(
            inverse,
            bool,
            name="inverse",
            message="inverse must be boolean.",
        )
        self._inverse = inverse

        # Load interleaver patterns as defined in the 3GPP standard
        self.coeffs_dict: dict[int, tuple[int, int]] = {}
        # Import coeffs from turbo module
        from sionna.phy.fec.turbo import coeffs

        source = files(coeffs).joinpath("turbo_coeffs.csv")
        with as_file(source) as coeffs_file:
            csv_reader = np.genfromtxt(coeffs_file, delimiter=",")
            for line_count, row in enumerate(csv_reader):
                if line_count > 0:  # Ignore header line
                    self.coeffs_dict[int(row[1])] = (int(row[2]), int(row[3]))

    @property
    def axis(self) -> int:
        """Axis to be permuted."""
        return self._axis

    @property
    def keep_state(self) -> bool:
        """Always `True` for the Turbo3GPP interleaver."""
        return self._keep_state

    def _generate_perm_full(
        self,
        frame_size: int,
        inverse: bool = False,
        target_device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Generates a permutation for the 3GPP Turbo interleaver.

        :param frame_size: Length of the sequence to be permuted.
        :param inverse: If `True`, the inverse permutation is generated.
        :param target_device: Device for the output tensor.

        :output perm_seq: Permutation tensor of shape ``[frame_size]``.
        """
        device = target_device if target_device is not None else self.device
        k = frame_size
        if k not in self.coeffs_dict:
            geqk_sizes = sorted([x for x in self.coeffs_dict if x >= k])
            if len(geqk_sizes) == 0:
                raise ValueError(
                    "Input frame size too large for 3GPP Turbo Interleaver."
                )
            else:
                k = geqk_sizes[0]

        f1, f2 = self.coeffs_dict[k]
        perm_seq = [(f1 * i + f2 * (i**2)) % k for i in range(k)]

        if frame_size < k:
            perm_seq = [x for x in perm_seq if x < frame_size]

        perm_seq = torch.tensor(perm_seq, dtype=torch.int64, device=device)

        if inverse:
            perm_seq = torch.argsort(perm_seq)

        return perm_seq

    def build(self, input_shape: tuple) -> None:
        """Build block and check consistency of dimensions.

        :param input_shape: Shape of input tensor.
        """
        if self._axis >= len(input_shape) or self._axis < -len(input_shape):
            raise ValueError("Axis does not match input shape.")

        # Normalize axis
        axis = self._axis if self._axis >= 0 else len(input_shape) + self._axis
        frame_size = input_shape[axis]

        if frame_size >= 6145:
            raise ValueError(
                "3GPP Turbo Interleaver is defined for block lengths up to 6144."
            )

    def call(
        self,
        x: torch.Tensor,
        /,
        *,
        inverse: Optional[bool] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Interleaving function.

        This function returns the permuted version of ``x``.

        :param x: Tensor of arbitrary shape.
        :param inverse: If provided, overrides the init parameter.

        :output x_int: Interleaved tensor of same shape as ``x``.
        """
        input_shape = x.shape

        # Normalize axis
        axis = self._axis if self._axis >= 0 else len(input_shape) + self._axis
        frame_size = input_shape[axis]

        if inverse is None:
            inverse = self._inverse

        perm_seq = self._generate_perm_full(frame_size, inverse, target_device=x.device)
        x_int = torch.index_select(x, axis, perm_seq)

        return x_int


class Deinterleaver(Block):
    """Deinterleaver that reverts the interleaver for a given input sequence.

    :param interleaver: Associated interleaver which shall be deinterleaved
        by this block. Can be either
        :class:`~sionna.phy.fec.interleaving.RandomInterleaver`,
        :class:`~sionna.phy.fec.interleaving.RowColumnInterleaver`, or
        :class:`~sionna.phy.fec.interleaving.Turbo3GPPInterleaver`.
    :param precision: Precision used for internal calculations and outputs.
        If `None`, inherits from ``interleaver``.
    :param device: Device for computation (e.g., 'cpu', 'cuda:0').
        If `None`, inherits from ``interleaver``.

    :input x: torch.Tensor.
        2+D tensor of arbitrary shape.

    :input seed: `int`.
        An integer defining the state of the random number
        generator. If explicitly given, the global internal seed is
        replaced by this seed. Can be used to realize random
        interleaver/deinterleaver pairs (call with same random seed).

    :output x_out: torch.Tensor.
        2+D tensor of same shape and dtype as the input ``x``.

    .. rubric:: Notes

    This block provides a wrapper of the inverse interleaver function.

    .. rubric:: Examples


    .. code-block:: python

        import torch
        from sionna.phy.fec.interleaving import RandomInterleaver, Deinterleaver

        interleaver = RandomInterleaver(seed=42, keep_state=True)
        deinterleaver = Deinterleaver(interleaver)

        x = torch.arange(10).reshape(1, 10).float()
        y = interleaver(x)
        z = deinterleaver(y)
        print(torch.allclose(x, z))
        # True
    """

    def __init__(
        self,
        interleaver: Union[
            RandomInterleaver, RowColumnInterleaver, Turbo3GPPInterleaver
        ],
        precision: Optional[str] = None,
        device: Optional[str] = None,
        **kwargs,
    ):
        if not isinstance(
            interleaver,
            (RandomInterleaver, RowColumnInterleaver, Turbo3GPPInterleaver),
        ):
            raise ValueError(
                "interleaver is not a valid interleaver instance."
            )

        # If precision/device is None, use same as associated interleaver
        if precision is None:
            precision = interleaver.precision
        if device is None:
            device = interleaver.device

        super().__init__(precision=precision, device=device, **kwargs)
        
        # Assign interleaver after super().__init__() for nn.Module compatibility
        self._interleaver = interleaver

        if self._interleaver.keep_state is False:
            warnings.warn(
                "Deinterleaver requires interleaver to have "
                "keep_state=True or to explicitly provide the seed as inputs.",
                UserWarning,
                stacklevel=2,
            )

    @property
    def interleaver(
        self,
    ) -> Union[RandomInterleaver, RowColumnInterleaver, Turbo3GPPInterleaver]:
        """Associated interleaver instance."""
        return self._interleaver

    def build(self, input_shape: tuple) -> None:
        """Build block."""
        pass

    def call(
        self,
        x: torch.Tensor,
        /,
        *,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """Deinterleaving function.

        This function returns the permuted version of ``x``.

        :param x: Tensor of arbitrary shape.
        :param seed: An integer defining the state of the random number
            generator. If explicitly given, the global internal seed is
            replaced by this seed. Can be used to realize random
            interleaver/deinterleaver pairs (call with same random seed).

        :output x_out: Deinterleaved tensor of same shape as ``x``.
        """
        input_dtype = x.dtype

        if isinstance(self._interleaver, RandomInterleaver):
            x_out = self._interleaver(x, seed=seed, inverse=True)
        else:
            x_out = self._interleaver(x, inverse=True)

        # Cast to original dtype to avoid different dtypes
        return x_out.to(input_dtype)

