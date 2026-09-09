#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Blocks for discrete channel models"""

from typing import Optional, Tuple, Union

import torch

from sionna._validation import check_binary
from sionna.phy import Block
from sionna.phy.config import Precision
from sionna.phy.utils import expand_to_rank, rand

__all__ = [
    "BinaryMemorylessChannel",
    "BinarySymmetricChannel",
    "BinaryZChannel",
    "BinaryErasureChannel",
]


class _STEBinarizer(torch.autograd.Function):
    """Straight-through estimator for binarization."""

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        """Hard decision in forward pass."""
        return torch.where(x < 0.5, 0.0, 1.0)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        """Identity in backward pass."""
        return grad_output


class _CustomXOR(torch.autograd.Function):
    """Straight-through estimator for XOR operation."""

    @staticmethod
    def forward(ctx, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """XOR operation in forward pass."""
        if a.dtype in (
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        ):
            z = torch.remainder(a + b, 2)
        else:
            z = torch.abs(a - b)
        return z

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Identity for both inputs in backward pass."""
        return grad_output, grad_output


class BinaryMemorylessChannel(Block):
    r"""
    Discrete binary memoryless channel with (possibly) asymmetric bit flipping
    probabilities.

    Input bits are flipped with probability :math:`p_\text{b,0}` and
    :math:`p_\text{b,1}`, respectively.

    ..  figure:: /phy/figures/BMC_channel.png
        :align: center

    This block supports binary inputs (:math:`x \in \{0, 1\}`) and `bipolar`
    inputs (:math:`x \in \{-1, 1\}`).

    If activated, the channel directly returns log-likelihood ratios (LLRs)
    defined as

    .. math::
        \ell =
        \begin{cases}
            \operatorname{log} \frac{p_{b,1}}{1-p_{b,0}}, \qquad \text{if} \, y=0 \\
            \operatorname{log} \frac{1-p_{b,1}}{p_{b,0}}, \qquad \text{if} \, y=1 \\
        \end{cases}

    Probabilities at the closed endpoints :math:`\{0,1\}` are accepted. For
    LLR outputs they are clamped away from the endpoints before the logarithm
    so that the LLRs remain finite and then saturate at ``llr_max``.

    The error probability :math:`p_\text{b}` can be a length-2 sequence of
    scalars or a tensor whose last dimension has length 2 (broadcastable to
    the shape of the input). This allows different flip probabilities per bit
    position. The last dimension is interpreted as :math:`p_\text{b,0}` and
    :math:`p_\text{b,1}`.

    :param return_llrs: If `True`, the layer returns log-likelihood ratios
        instead of binary values based on ``pb``. Defaults to `False`.
    :param bipolar_input: If `True`, the expected input is given as
        :math:`\{-1,1\}` instead of :math:`\{0,1\}`. Defaults to `False`.
    :param llr_max: Clipping value of the LLRs. Defaults to 100.
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`, :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for computation. If `None`,
        :attr:`~sionna.phy.config.Config.device` is used.

    :input x: [...,n], `torch.Tensor`.
        Input sequence to the channel consisting of binary values
        :math:`\{0,1\}` or :math:`\{-1,1\}`, respectively.

    :input pb: [...,2], `torch.Tensor`.
        Error probability. Can be a tuple of two scalars or a tensor of any
        shape that can be broadcast to ``x`` with a last dimension of length
        2, interpreted as :math:`p_\text{b,0}` and :math:`p_\text{b,1}`.

    :output y: [...,n], `torch.Tensor`.
        Output sequence of the same length as ``x``. Binary (or bipolar)
        symbols when ``return_llrs`` is `False`, or log-likelihood ratios
        when ``return_llrs`` is `True`.

    .. rubric:: Examples

    .. code-block:: python

        import torch
        from sionna.phy.channel import BinaryMemorylessChannel

        channel = BinaryMemorylessChannel()
        x = torch.randint(0, 2, (10, 100), dtype=torch.float32)
        pb = (0.1, 0.2)  # p(flip|0) = 0.1, p(flip|1) = 0.2
        y = channel(x, pb)
        print(y.shape)
        # torch.Size([10, 100])
    """

    def __init__(
        self,
        return_llrs: bool = False,
        bipolar_input: bool = False,
        llr_max: float = 100.0,
        precision: Optional[Precision] = None,
        device: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(precision=precision, device=device, **kwargs)

        if not isinstance(return_llrs, bool):
            raise TypeError("return_llrs must be bool.")
        self._return_llrs = return_llrs

        if not isinstance(bipolar_input, bool):
            raise TypeError("bipolar_input must be bool.")
        self._bipolar_input = bipolar_input

        if llr_max < 0.0:
            raise ValueError("llr_max must be a non-negative value.")
        # Register as buffer for CUDAGraph compatibility
        self.register_buffer("_llr_max", torch.tensor(llr_max, dtype=self.dtype, device=self.device))

        self._check_input = True  # check input for consistency (i.e., binary)
        self._eps = 1e-9  # small additional term for numerical stability
        self.register_buffer("_temperature", torch.tensor(0.1, dtype=self.dtype, device=self.device))

    @property
    def llr_max(self) -> torch.Tensor:
        """Get/set maximum value used for LLR calculations."""
        return self._llr_max

    @llr_max.setter
    def llr_max(self, value: float) -> None:
        if value < 0:
            raise ValueError("llr_max cannot be negative.")
        # Register as buffer for CUDAGraph compatibility
        self.register_buffer("_llr_max", torch.tensor(value, dtype=self.dtype, device=self.device))

    @property
    def temperature(self) -> torch.Tensor:
        """Get/set temperature for Gumbel-softmax trick."""
        return self._temperature

    @temperature.setter
    def temperature(self, value: float) -> None:
        if value < 0:
            raise ValueError("temperature cannot be negative.")
        # Register as buffer for CUDAGraph compatibility
        self.register_buffer("_temperature", torch.tensor(value, dtype=self.dtype, device=self.device))

    def _check_inputs(self, x: torch.Tensor) -> None:
        """Validate binary inputs once eagerly and on every compiled call."""
        if torch.compiler.is_compiling() or self._check_input:
            check_binary(
                x.to(self.dtype),
                name="x",
                bipolar=self._bipolar_input,
                message="Input must be binary.",
            )
        if not torch.compiler.is_compiling():
            self._check_input = False

    def _check_dtype(self, x: torch.Tensor, allow_uint: bool = True) -> None:
        """Check input dtype for consistency with parameters."""
        float_dtypes = (torch.float32, torch.float64)
        signed_int_dtypes = (torch.int8, torch.int16, torch.int32, torch.int64)
        unsigned_int_dtypes = (torch.uint8,)

        if self._return_llrs:
            if x.dtype not in float_dtypes:
                raise TypeError("LLR outputs require float dtypes.")
        else:
            if self._bipolar_input:
                valid_dtypes = float_dtypes + signed_int_dtypes
                if x.dtype not in valid_dtypes:
                    raise TypeError(
                        "Only signed dtypes are supported for bipolar inputs."
                    )
            else:
                valid_dtypes = float_dtypes + signed_int_dtypes + unsigned_int_dtypes
                if x.dtype not in valid_dtypes:
                    raise TypeError("Only real-valued dtypes are supported.")

            if not allow_uint:
                if x.dtype in unsigned_int_dtypes:
                    raise TypeError("Only signed dtypes supported.")

    def _sample_errors(self, pb: torch.Tensor, shape: torch.Size) -> torch.Tensor:
        """Sample binary error vector with given error probability.

        This function is based on the Gumbel-softmax trick to keep the
        sampling differentiable.
        """
        # Implementation follows https://arxiv.org/pdf/1611.01144v5.pdf
        # and https://arxiv.org/pdf/1906.07748.pdf
        # Uses smart rand that switches to global RNG in compiled mode

        u1 = rand(
            shape, dtype=pb.dtype, device=self.device, generator=self.torch_rng
        )
        u2 = rand(
            shape, dtype=pb.dtype, device=self.device, generator=self.torch_rng
        )
        u = torch.stack((u1, u2), dim=-1)

        # Sample Gumbel distribution
        eps = torch.tensor(self._eps, dtype=pb.dtype, device=self.device)
        temp = self._temperature.to(pb.dtype)
        q = -torch.log(-torch.log(u + eps) + eps)

        p = torch.stack((pb, 1 - pb), dim=-1)
        p = expand_to_rank(p, q.dim(), axis=0)
        p = p.broadcast_to(q.shape)

        a = (torch.log(p + eps) + q) / temp

        # Apply softmax
        e_cat = torch.nn.functional.softmax(a, dim=-1)

        # Binarize final values via straight-through estimator
        return _STEBinarizer.apply(e_cat[..., 0])

    def build(self, *input_shapes) -> None:
        """No shape validation in ``build``; ``pb`` is checked in ``call``."""

    def _validate_pb(
        self, pb: Union[Tuple[float, float], torch.Tensor]
    ) -> None:
        """Validate ``pb`` against the documented public contract."""
        if isinstance(pb, (tuple, list)):
            if len(pb) != 2:
                raise ValueError(
                    "pb as a sequence must contain exactly two probabilities "
                    "(p_b,0 and p_b,1)."
                )
            return
        if not isinstance(pb, torch.Tensor):
            raise TypeError(
                "pb must be a torch.Tensor or a length-2 sequence of "
                "probabilities."
            )
        if pb.ndim == 0 or pb.shape[-1] != 2:
            raise ValueError(
                "Last dimension of pb must have length 2 "
                f"(got shape {tuple(pb.shape)})."
            )

    def call(
        self,
        x: torch.Tensor,
        pb: Union[Tuple[float, float], torch.Tensor],
    ) -> torch.Tensor:
        """Apply discrete binary memoryless channel to inputs."""
        # Check input dtype for consistency with parameters
        self._check_dtype(x)
        self._validate_pb(pb)

        # Allow pb to be a tuple of two scalars
        if isinstance(pb, (tuple, list)):
            pb0 = pb[0]
            pb1 = pb[1]
        else:
            pb0 = pb[..., 0]
            pb1 = pb[..., 1]

        # Convert to tensor and clip for numerical stability
        if not isinstance(pb0, torch.Tensor):
            pb0 = torch.tensor(pb0, dtype=self.dtype, device=self.device)
        else:
            pb0 = pb0.to(dtype=self.dtype, device=self.device)
        if not isinstance(pb1, torch.Tensor):
            pb1 = torch.tensor(pb1, dtype=self.dtype, device=self.device)
        else:
            pb1 = pb1.to(dtype=self.dtype, device=self.device)

        pb0 = pb0.clamp(0.0, 1.0)
        pb1 = pb1.clamp(0.0, 1.0)

        # Check x for consistency (binary, bipolar)
        self._check_inputs(x)

        e0 = self._sample_errors(pb0, x.shape)
        e1 = self._sample_errors(pb1, x.shape)

        if self._bipolar_input:
            neutral_element = torch.tensor(-1, dtype=x.dtype, device=self.device)
        else:
            neutral_element = torch.tensor(0, dtype=x.dtype, device=self.device)

        # Mask e0 and e1 with input such that e0 only applies where x==0
        e = torch.where(x == neutral_element, e0, e1)
        e = e.to(x.dtype)

        if self._bipolar_input:
            # Flip signs for bipolar case
            y = x * (-2 * e + 1)
        else:
            # XOR for binary case
            y = _CustomXOR.apply(x, e)

        # If LLRs should be returned
        if self._return_llrs:
            if not self._bipolar_input:
                y = 2 * y - 1  # transform to bipolar

            # Remark: Sionna uses the logit definition log[p(x=1)/p(x=0)]
            # Clamp away from {0,1} using the working precision's machine eps
            # so both endpoints keep strictly positive log arguments (1e-9 is
            # below float32 eps and would make 1-eps round back to 1.0).
            # LLRs then saturate and are clipped to +/- llr_max.
            eps = torch.finfo(pb0.dtype).eps
            pb0_llr = pb0.clamp(eps, 1.0 - eps)
            pb1_llr = pb1.clamp(eps, 1.0 - eps)
            y0 = -(torch.log(pb1_llr) - torch.log(1.0 - pb0_llr))
            y1 = torch.log(1.0 - pb1_llr) - torch.log(pb0_llr)

            # Multiply by y to keep gradient
            y = torch.where(y == 1, y1, y0).to(y.dtype) * y

            # Clip output LLRs
            llr_max = self._llr_max.to(y.dtype)
            y = y.clamp(-llr_max, llr_max)

        return y


class BinarySymmetricChannel(BinaryMemorylessChannel):
    r"""
    Discrete binary symmetric channel which randomly flips bits with
    probability :math:`p_\text{b}`.

    ..  figure:: /phy/figures/BSC_channel.png
        :align: center

    This layer supports binary inputs (:math:`x \in \{0, 1\}`) and `bipolar`
    inputs (:math:`x \in \{-1, 1\}`).

    If activated, the channel directly returns log-likelihood ratios (LLRs)
    defined as

    .. math::
        \ell =
        \begin{cases}
            \operatorname{log} \frac{p_{b}}{1-p_{b}}, \qquad \text{if}\, y=0 \\
            \operatorname{log} \frac{1-p_{b}}{p_{b}}, \qquad \text{if}\, y=1 \\
        \end{cases}

    where :math:`y` denotes the binary output of the channel.

    The bit flipping probability :math:`p_\text{b}` can be either a scalar or
    a tensor (broadcastable to the shape of the input). This allows
    different bit flipping probabilities per bit position.

    :param return_llrs: If `True`, the layer returns log-likelihood ratios
        instead of binary values based on ``pb``. Defaults to `False`.
    :param bipolar_input: If `True`, the expected input is given as
        :math:`\{-1,1\}` instead of :math:`\{0,1\}`. Defaults to `False`.
    :param llr_max: Clipping value of the LLRs. Defaults to 100.
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`, :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for computation. If `None`,
        :attr:`~sionna.phy.config.Config.device` is used.

    :input x: [...,n], `torch.Tensor`.
        Input sequence to the channel.

    :input pb: `torch.Tensor` or `float`.
        Bit flipping probability. Can be a scalar or of any shape that
        can be broadcasted to the shape of ``x``.

    :output y: [...,n], `torch.Tensor`.
        Output sequence of the same length as ``x``. Binary (or bipolar)
        symbols when ``return_llrs`` is `False`, or log-likelihood ratios
        when ``return_llrs`` is `True`. Endpoint flip probabilities are
        accepted; LLR outputs remain finite and saturate at ``llr_max``.

    .. rubric:: Examples

    .. code-block:: python

        import torch
        from sionna.phy.channel import BinarySymmetricChannel

        channel = BinarySymmetricChannel()
        x = torch.randint(0, 2, (10, 100), dtype=torch.float32)
        pb = 0.1  # bit flip probability
        y = channel(x, pb)
        print(y.shape)
        # torch.Size([10, 100])
    """

    def __init__(
        self,
        return_llrs: bool = False,
        bipolar_input: bool = False,
        llr_max: float = 100.0,
        precision: Optional[Precision] = None,
        device: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            return_llrs=return_llrs,
            bipolar_input=bipolar_input,
            llr_max=llr_max,
            precision=precision,
            device=device,
            **kwargs,
        )

    def build(self, *input_shapes) -> None:
        """No shape validation needed for BSC."""

    def call(
        self,
        x: torch.Tensor,
        pb: Union[float, torch.Tensor],
    ) -> torch.Tensor:
        """Apply discrete binary symmetric channel, i.e., randomly flip
        bits with probability ``pb``."""
        # BSC is implemented by calling the base class with symmetric pb
        if not isinstance(pb, torch.Tensor):
            pb = torch.tensor(pb, dtype=x.dtype, device=self.device)
        else:
            pb = pb.to(dtype=x.dtype, device=self.device)

        pb_stacked = torch.stack((pb, pb), dim=-1)
        return super().call(x, pb_stacked)


class BinaryZChannel(BinaryMemorylessChannel):
    r"""
    Block that implements the binary Z-channel.

    In the Z-channel, transmission errors only occur for the transmission of
    second input element (i.e., if a `1` is transmitted) with error probability
    :math:`p_\text{b}` but the first element is always correctly received.

    ..  figure:: /phy/figures/Z_channel.png
        :align: center

    This block supports binary inputs (:math:`x \in \{0, 1\}`) and `bipolar`
    inputs (:math:`x \in \{-1, 1\}`).

    If activated, the channel directly returns log-likelihood ratios (LLRs)
    defined as

    .. math::
        \ell =
        \begin{cases}
            \operatorname{log} \left( p_b \right), \qquad \text{if} \, y=0 \\
            \infty, \qquad \qquad \text{if} \, y=1 \\
        \end{cases}

    assuming equal probable inputs :math:`P(X=0) = P(X=1) = 0.5`.

    The error probability :math:`p_\text{b}` can be either a scalar or a
    tensor (broadcastable to the shape of the input). This allows
    different error probabilities per bit position.

    :param return_llrs: If `True`, the layer returns log-likelihood ratios
        instead of binary values based on ``pb``. Defaults to `False`.
    :param bipolar_input: If `True`, the expected input is given as
        :math:`\{-1,1\}` instead of :math:`\{0,1\}`. Defaults to `False`.
    :param llr_max: Clipping value of the LLRs. Defaults to 100.
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`, :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for computation. If `None`,
        :attr:`~sionna.phy.config.Config.device` is used.

    :input x: [...,n], `torch.Tensor`.
        Input sequence to the channel.

    :input pb: `torch.Tensor` or `float`.
        Error probability. Can be a scalar or of any shape that can be
        broadcasted to the shape of ``x``.

    :output y: [...,n], `torch.Tensor`.
        Output sequence of same length as the input ``x``. If
        ``return_llrs`` is `False`, the output is binary and otherwise
        soft-values are returned.

    .. rubric:: Examples

    .. code-block:: python

        import torch
        from sionna.phy.channel import BinaryZChannel

        channel = BinaryZChannel()
        x = torch.randint(0, 2, (10, 100), dtype=torch.float32)
        pb = 0.1  # error probability for 1->0
        y = channel(x, pb)
        print(y.shape)
        # torch.Size([10, 100])
    """

    def __init__(
        self,
        return_llrs: bool = False,
        bipolar_input: bool = False,
        llr_max: float = 100.0,
        precision: Optional[Precision] = None,
        device: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            return_llrs=return_llrs,
            bipolar_input=bipolar_input,
            llr_max=llr_max,
            precision=precision,
            device=device,
            **kwargs,
        )

    def build(self, *input_shapes) -> None:
        """No shape validation needed for Z-channel."""

    def call(
        self,
        x: torch.Tensor,
        pb: Union[float, torch.Tensor],
    ) -> torch.Tensor:
        """Apply binary Z-channel, i.e., randomly flip 1s to 0s with
        probability ``pb``."""
        # Z-channel is implemented by calling the base class with p(1|0)=0
        if not isinstance(pb, torch.Tensor):
            pb = torch.tensor(pb, dtype=x.dtype, device=self.device)
        else:
            pb = pb.to(dtype=x.dtype, device=self.device)

        pb_stacked = torch.stack((torch.zeros_like(pb), pb), dim=-1)
        return super().call(x, pb_stacked)


class BinaryErasureChannel(BinaryMemorylessChannel):
    r"""
    Binary erasure channel (BEC) where a bit is either correctly received
    or erased.

    In the binary erasure channel, bits are always correctly received or erased
    with erasure probability :math:`p_\text{b}`.

    ..  figure:: /phy/figures/BEC_channel.png
        :align: center

    This block supports binary inputs (:math:`x \in \{0, 1\}`) and `bipolar`
    inputs (:math:`x \in \{-1, 1\}`).

    If activated, the channel directly returns log-likelihood ratios (LLRs)
    defined as

    .. math::
        \ell =
        \begin{cases}
            -\infty, \qquad \text{if} \, y=0 \\
            0, \qquad \quad \,\, \text{if} \, y=? \\
            \infty, \qquad \quad \text{if} \, y=1 \\
        \end{cases}

    The erasure probability :math:`p_\text{b}` can be either a scalar or a
    tensor (broadcastable to the shape of the input). This allows
    different erasure probabilities per bit position.

    Please note that the output of the BEC is ternary. Hereby, `-1` indicates an
    erasure for the binary configuration and `0` for the bipolar mode,
    respectively.

    :param return_llrs: If `True`, the layer returns log-likelihood ratios
        instead of binary values based on ``pb``. Defaults to `False`.
    :param bipolar_input: If `True`, the expected input is given as
        :math:`\{-1,1\}` instead of :math:`\{0,1\}`. Defaults to `False`.
    :param llr_max: Clipping value of the LLRs. Defaults to 100.
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`, :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for computation. If `None`,
        :attr:`~sionna.phy.config.Config.device` is used.

    :input x: [...,n], `torch.Tensor`.
        Input sequence to the channel.

    :input pb: `torch.Tensor` or `float`.
        Erasure probability. Can be a scalar or of any shape that can be
        broadcasted to the shape of ``x``.

    :output y: [...,n], `torch.Tensor`.
        Output sequence of same length as the input ``x``. If
        ``return_llrs`` is `False`, the output is ternary where each `-1`
        and each `0` indicate an erasure for the binary and bipolar input,
        respectively.

    .. rubric:: Examples

    .. code-block:: python

        import torch
        from sionna.phy.channel import BinaryErasureChannel

        channel = BinaryErasureChannel()
        x = torch.randint(0, 2, (10, 100), dtype=torch.float32)
        pb = 0.1  # erasure probability
        y = channel(x, pb)
        print(y.shape)
        # torch.Size([10, 100])
    """

    def __init__(
        self,
        return_llrs: bool = False,
        bipolar_input: bool = False,
        llr_max: float = 100.0,
        precision: Optional[Precision] = None,
        device: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            return_llrs=return_llrs,
            bipolar_input=bipolar_input,
            llr_max=llr_max,
            precision=precision,
            device=device,
            **kwargs,
        )

    def build(self, *input_shapes) -> None:
        """No shape validation needed for BEC."""

    def call(
        self,
        x: torch.Tensor,
        pb: Union[float, torch.Tensor],
    ) -> torch.Tensor:
        """Apply erasure channel to inputs."""
        # Check input dtype for consistency with parameters
        self._check_dtype(x, allow_uint=False)

        # Convert to tensor and clip for numerical stability
        if not isinstance(pb, torch.Tensor):
            pb = torch.tensor(pb, dtype=self.dtype, device=self.device)
        else:
            pb = pb.to(dtype=self.dtype, device=self.device)

        pb = pb.clamp(0.0, 1.0)

        # Check x for consistency (binary, bipolar)
        self._check_inputs(x)

        # Sample erasure pattern
        e = self._sample_errors(pb, x.shape)

        # If LLRs should be returned
        # Remark: the Sionna logit definition is llr = log[p(x=1)/p(x=0)]
        if self._return_llrs:
            if not self._bipolar_input:
                x = 2 * x - 1
            x = x * self._llr_max.to(x.dtype)  # calculate LLRs

            # Erase positions by setting LLRs to 0
            y = torch.where(
                e == 1, torch.tensor(0, dtype=x.dtype, device=self.device), x
            )
        else:  # ternary outputs
            # The erasure indicator depends on the operation mode
            if self._bipolar_input:
                erased_element = torch.tensor(0, dtype=x.dtype, device=self.device)
            else:
                erased_element = torch.tensor(-1, dtype=x.dtype, device=self.device)

            y = torch.where(e == 0, x, erased_element)

        return y
