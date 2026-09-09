#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Utility functions and blocks for the FEC package."""

from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import warnings

from sionna.phy import Block
from sionna.phy.utils import normal


__all__ = [
    "GaussianPriorSource",
    "llr2mi",
    "j_fun",
    "j_fun_inv",
    "bin2int",
    "int2bin",
    "int_mod_2",
    # Re-exported from plotting
    "plot_trajectory",
    "plot_exit_chart",
    "get_exit_analytic",
    # Re-exported from coding
    "load_parity_check_examples",
    "alist2mat",
    "load_alist",
    "make_systematic",
    "gm2pcm",
    "pcm2gm",
    "verify_gm_pcm",
    "generate_reg_ldpc",
]

_PLOTTING_NAMES = {"plot_trajectory", "plot_exit_chart", "get_exit_analytic"}
_CODING_NAMES = {
    "load_parity_check_examples", "alist2mat", "load_alist",
    "make_systematic", "gm2pcm", "pcm2gm", "verify_gm_pcm",
    "generate_reg_ldpc",
}


def __getattr__(name):
    """Lazy re-exports from plotting and coding submodules."""
    if name in _PLOTTING_NAMES:
        from sionna.phy.fec import plotting
        return getattr(plotting, name)
    if name in _CODING_NAMES:
        from sionna.phy.fec import coding
        return getattr(coding, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


class GaussianPriorSource(Block):
    r"""Generates synthetic Log-Likelihood Ratios (LLRs) for Gaussian channels.

    Generates synthetic Log-Likelihood Ratios (LLRs) as if an all-zero codeword
    was transmitted over a Binary Additive White Gaussian Noise (Bi-AWGN)
    channel. The LLRs are generated based on either the noise variance ``no``
    or mutual information. If mutual information is used, it represents the
    information associated with a binary random variable observed through an
    AWGN channel.

    The generated LLRs follow a Gaussian distribution with parameters:

    .. math::
        \sigma_{\text{llr}}^2 = \frac{4}{\sigma_\text{ch}^2}

    .. math::
        \mu_{\text{llr}} = \frac{\sigma_\text{llr}^2}{2}

    where :math:`\sigma_\text{ch}^2` is the noise variance specified by ``no``.

    .. note::
        ``no`` is the variance of the real-valued noise in the Bi-AWGN channel.
        This differs from :class:`~sionna.phy.channel.AWGN`, where ``no`` is the
        total variance of a complex-valued noise sample and each real dimension
        has variance ``no/2``. To match the LLR distribution obtained from
        BPSK transmission over :class:`~sionna.phy.channel.AWGN` with noise
        variance ``no``, use ``no/2`` for this source.

    If the mutual information is provided as input, the J-function as described
    in :cite:p:`Brannstrom` is used to relate the mutual information to the
    corresponding LLR distribution.

    :param precision: Precision used for internal calculations and outputs.
        If `None`, :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for computation (e.g., 'cpu', 'cuda:0').
        If `None`, :attr:`~sionna.phy.config.Config.device` is used.

    :input output_shape: `list` of `int` or `torch.Size`.
        Shape of the generated LLR tensor.

    :input no: `None` (default) | `float`.
        Scalar defining the real-valued noise variance for the synthetic
        Bi-AWGN channel.

    :input mi: `None` (default) | `float`.
        Scalar defining the mutual information for the synthetic AWGN channel.
        Only used if ``no`` is `None`.

    :output llr: `torch.Tensor`.
        Tensor with shape defined by ``output_shape``.

    .. rubric:: Examples


    .. code-block:: python

        from sionna.phy.fec.utils import GaussianPriorSource

        source = GaussianPriorSource()
        llrs = source([1000], no=1.0)
        print(llrs.shape)
        # torch.Size([1000])
    """

    def __init__(
        self,
        *,
        precision: Optional[str] = None,
        device: Optional[str] = None,
        **kwargs
    ):
        super().__init__(precision=precision, device=device, **kwargs)

    def call(
        self,
        output_shape: Union[List[int], Tuple[int, ...], torch.Size],
        no: Optional[float] = None,
        mi: Optional[float] = None,
    ) -> torch.Tensor:
        """Generate Gaussian distributed LLRs as if the all-zero codeword
        was transmitted over a Bi-AWGN channel.
        """
        if no is None:
            if mi is None:
                raise ValueError("Either no or mi must be provided.")
            # clip mi to range (0, 1)
            mi_t = torch.as_tensor(mi, dtype=self.dtype, device=self.device)
            mi_t = mi_t.clamp(min=1e-7, max=1.0)
            mu_llr = j_fun_inv(mi_t)
            sigma_llr = torch.sqrt(2 * mu_llr)
        else:
            # noise_var must be positive
            no_t = torch.as_tensor(no, dtype=self.dtype, device=self.device)
            no_t = no_t.clamp(min=1e-7)
            sigma_llr = torch.sqrt(4 / no_t)
            mu_llr = sigma_llr**2 / 2

        # generate LLRs with Gaussian approximation (BPSK, all-zero cw)
        # Use negative mean as we generate logits with definition p(b=1)/p(b=0)
        # Uses smart randn that switches to global RNG in compiled mode
        llr = normal(
            output_shape,
            dtype=self.dtype,
            device=self.device,
            generator=self.torch_rng,
        ) * sigma_llr + (-1.0 * mu_llr)
        return llr


def llr2mi(
    llr: torch.Tensor,
    s: Optional[torch.Tensor] = None,
    reduce_dims: bool = True,
) -> torch.Tensor:
    r"""Approximates the mutual information based on Log-Likelihood Ratios (LLRs).

    This function approximates the mutual information for a given set of ``llr``
    values, assuming an `all-zero codeword` transmission as derived in
    :cite:p:`Hagenauer`:

    .. math::

        I \approx 1 - \sum \operatorname{log_2} \left( 1 + \operatorname{e}^{-\text{llr}} \right)

    The approximation relies on the `symmetry condition`:

    .. math::

        p(\text{llr}|x=0) = p(\text{llr}|x=1) \cdot \operatorname{exp}(\text{llr})

    For cases where the transmitted codeword is not all-zero, this method
    requires knowledge of the original bit sequence ``s`` to adjust the LLR
    signs accordingly, simulating an all-zero codeword transmission.

    Note that the LLRs are defined as :math:`\frac{p(x=1)}{p(x=0)}`, which
    reverses the sign compared to the solution in :cite:p:`Hagenauer`.

    :param llr: Tensor of arbitrary shape containing LLR values.
    :param s: Tensor of the same shape as ``llr`` representing the signs of the
        transmitted sequence (assuming BPSK), with values of +/-1.
    :param reduce_dims: If `True`, reduces all dimensions and returns a scalar.
        If `False`, averages only over the last dimension.

    :output mi: Approximated mutual information. Scalar if ``reduce_dims``
        is `True`, otherwise tensor with the same shape as ``llr`` except
        for the last dimension, which is removed.

    .. rubric:: Examples


    .. code-block:: python

        import torch
        from sionna.phy.fec.utils import llr2mi

        llr = torch.randn(1000) * 2.0
        mi = llr2mi(llr)
        print(mi.item())
    """
    if not llr.is_floating_point():
        raise TypeError("Dtype of llr must be a real-valued float.")

    if s is not None:
        # ensure compatible types
        s = s.to(llr.dtype)
        # scramble sign as if all-zero cw was transmitted
        llr_zero = s * llr
    else:
        llr_zero = llr

    # clip for numerical stability
    llr_zero = llr_zero.clamp(min=-100.0, max=100.0)

    x = torch.log2(1.0 + torch.exp(llr_zero))
    if reduce_dims:
        x = 1.0 - x.mean()
    else:
        x = 1.0 - x.mean(dim=-1)
    return x


def j_fun(mu: torch.Tensor) -> torch.Tensor:
    r"""Computes the `J-function`.

    The `J-function` relates mutual information to the mean of
    Gaussian-distributed Log-Likelihood Ratios (LLRs) using the Gaussian
    approximation. This function implements the approximation proposed in
    :cite:p:`Brannstrom`:

    .. math::

        J(\mu) \approx \left( 1 - 2^{H_\text{1}(2\mu)^{H_\text{2}}}\right)^{H_\text{3}}

    where :math:`\mu` represents the mean of the LLR distribution, and the
    constants are defined as :math:`H_\text{1}=0.3073`,
    :math:`H_\text{2}=0.8935`, and :math:`H_\text{3}=1.1064`.

    Input values are clipped to [1e-10, 1000] for numerical stability.

    :param mu: Tensor of arbitrary shape, representing the mean of the
        LLR values.

    :output mi: Tensor of the same shape and dtype as ``mu``, containing
        the calculated mutual information values.

    .. rubric:: Examples


    .. code-block:: python

        import torch
        from sionna.phy.fec.utils import j_fun

        mu = torch.tensor([0.1, 1.0, 5.0])
        mi = j_fun(mu)
        print(mi)
    """
    # input must be positive for numerical stability
    mu = mu.clamp(min=1e-10, max=1000)

    h1 = 0.3073
    h2 = 0.8935
    h3 = 1.1064
    mi = (1 - 2 ** (-h1 * (2 * mu) ** h2)) ** h3
    return mi


def j_fun_inv(mi: torch.Tensor) -> torch.Tensor:
    r"""Computes the inverse of the `J-function`.

    The `J-function` relates mutual information to the mean of
    Gaussian-distributed Log-Likelihood Ratios (LLRs) using the Gaussian
    approximation. This function computes the inverse `J-function` based on the
    approximation proposed in :cite:p:`Brannstrom`:

    .. math::

        J(\mu) \approx \left( 1 - 2^{H_\text{1}(2\mu)^{H_\text{2}}}\right)^{H_\text{3}}

    where :math:`\mu` is the mean of the LLR distribution, and constants are
    defined as :math:`H_\text{1}=0.3073`, :math:`H_\text{2}=0.8935`, and
    :math:`H_\text{3}=1.1064`.

    Input values are clipped to [1e-10, 1] for numerical stability.
    The output is clipped to a maximum LLR of 20.

    :param mi: Tensor of arbitrary shape, representing mutual information
        values.

    :output mu: Tensor of the same shape and dtype as ``mi``, containing
        the computed mean values of the LLR distribution.

    .. rubric:: Examples


    .. code-block:: python

        import torch
        from sionna.phy.fec.utils import j_fun_inv

        mi = torch.tensor([0.1, 0.5, 0.9])
        mu = j_fun_inv(mi)
        print(mu)
    """
    # input must be positive for numerical stability
    mi = mi.clamp(min=1e-10, max=1.0)

    h1 = 0.3073
    h2 = 0.8935
    h3 = 1.1064
    mu = 0.5 * ((-1 / h1) * torch.log2(1 - mi ** (1 / h3))) ** (1 / h2)
    return mu.clamp(max=20)  # clip the output to mu_max=20


def bin2int(arr: Union[List[int], torch.Tensor]) -> Union[int, None, torch.Tensor]:
    """Converts a binary array or tensor to its integer representation.

    Interprets the binary representation from most significant to least
    significant bit. For example, ``[1, 0, 1]`` is converted to ``5``.

    Accepts both plain Python lists and PyTorch tensors. When a list is
    provided, returns a plain Python `int` (or `None` if empty). When a
    tensor is provided, returns a tensor with the last dimension reduced.

    :param arr: An iterable of binary values (0's and 1's), or a
        `torch.Tensor` with binary values along the last dimension.

    :output result: The integer representation. A plain `int` if ``arr``
        is a list, `None` if the list is empty, or a `torch.Tensor` if
        ``arr`` is a tensor.

    .. rubric:: Examples


    .. code-block:: python

        from sionna.phy.fec.utils import bin2int

        # From a Python list
        result = bin2int([1, 0, 1])
        print(result)
        # 5

        # From a PyTorch tensor
        import torch
        result = bin2int(torch.tensor([[1, 0, 1], [0, 1, 1]]))
        print(result)
        # tensor([5, 3])
    """
    if isinstance(arr, torch.Tensor):
        len_ = arr.shape[-1]
        shifts = torch.arange(len_ - 1, -1, -1, device=arr.device)
        arr_int = arr.to(torch.int32)
        return (arr_int << shifts).sum(dim=-1)
    else:
        if len(arr) == 0:
            return None
        return int("".join([str(x) for x in arr]), 2)


def int2bin(num: Union[int, torch.Tensor],
            length: int) -> Union[List[int], torch.Tensor]:
    """Converts an integer or integer tensor to binary representation.

    When ``num`` is a plain Python `int`, returns a list of 0's and 1's padded
    to the specified ``length``. When ``num`` is a `torch.Tensor`, returns a
    tensor with an additional dimension of size ``length`` at the end.

    Both ``num`` (or all elements) and ``length`` must be non-negative.

    :param num: The integer(s) to convert. Either a non-negative Python
        `int` or a `torch.Tensor` of integers.
    :param length: The desired bit length of the binary representation.
        Must be non-negative.

    :output result: A list of 0's and 1's if ``num`` is an `int`, or a
        `torch.Tensor` with an extra trailing dimension of size ``length``.

    .. rubric:: Examples


    .. code-block:: python

        from sionna.phy.fec.utils import int2bin

        # From a Python int
        result = int2bin(5, 4)
        print(result)
        # [0, 1, 0, 1]

        # From a PyTorch tensor
        import torch
        result = int2bin(torch.tensor([5, 12]), 4)
        print(result)
        # tensor([[0, 1, 0, 1],
        #         [1, 1, 0, 0]])
    """
    if length < 0:
        raise ValueError("length should be non-negative.")

    if isinstance(num, torch.Tensor):
        if length == 0:
            return torch.zeros(
                num.shape + (0,), dtype=torch.int32, device=num.device
            )
        shifts = torch.arange(length - 1, -1, -1, device=num.device)
        num_int = num.to(torch.int32)
        return (num_int.unsqueeze(-1) >> shifts) % 2
    else:
        if num < 0:
            raise ValueError("Input integer should be non-negative.")
        bin_ = format(num, f"0{length}b")
        return [int(x) for x in bin_[-length:]] if length else []


def int_mod_2(x: torch.Tensor) -> torch.Tensor:
    r"""Modulo 2 operation and implicit rounding for floating point inputs.

    Performs more efficient modulo-2 operation for integer inputs using bitwise
    AND. For floating inputs, applies implicit rounding before the modulo
    operation.

    :param x: Tensor to which the modulo 2 operation is applied.

    :output x_mod: Binary tensor containing the result of the modulo 2
        operation, with the same shape as ``x``.

    .. rubric:: Examples


    .. code-block:: python

        import torch
        from sionna.phy.fec.utils import int_mod_2

        x = torch.tensor([0, 1, 2, 3, 4, 5])
        result = int_mod_2(x)
        print(result)
        # tensor([0, 1, 0, 1, 0, 1])
    """
    if x.dtype in (torch.int8, torch.int16, torch.int32, torch.int64):
        x_mod = x & 1
    else:
        # round to next integer
        x_ = torch.abs(torch.round(x))
        x_mod = torch.fmod(x_, 2)
    return x_mod
