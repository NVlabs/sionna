#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Blocks implementing windowing functions"""

from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import torch

from sionna.phy import Block
from sionna.phy.config import Precision
from sionna.phy.utils import expand_to_rank

__all__ = [
    "Window",
    "CustomWindow",
    "HannWindow",
    "HammingWindow",
    "BlackmanWindow",
]


class Window(Block):
    r"""Abstract class defining a window function.

    The window function is applied through element-wise multiplication.

    The window function is real-valued. The dtype of the output is the same
    as the dtype of the input ``x`` to which the window function is applied.
    The window function and the input must have the same precision.

    :param normalize: If `True`, the window is normalized to have unit average
        power per coefficient. Defaults to `False`.
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`, :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for computation. If `None`,
        :attr:`~sionna.phy.config.Config.device` is used.

    :input x: [..., N], `torch.complex` or `torch.float`.
        The input to which the window function is applied.
        The window function is applied along the last dimension.
        The length of the last dimension ``N`` must be the same
        as the ``length`` of the window function.

    :output y: [..., N], `torch.complex` or `torch.float`.
        Output of the windowing operation.

    .. rubric:: Examples

    .. code-block:: python

        import torch
        from sionna.phy.signal import HannWindow

        window = HannWindow()
        x = torch.randn(32, 64)
        y = window(x)
        print(y.shape)
        # torch.Size([32, 64])
    """

    def __init__(
        self,
        normalize: bool = False,
        precision: Optional[Precision] = None,
        device: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(precision=precision, device=device, **kwargs)
        if not isinstance(normalize, bool):
            raise ValueError(f"normalize must be bool, got {type(normalize)}")
        self._normalize = normalize
        self._coefficients: Optional[torch.Tensor] = None

    @property
    def coefficients(self) -> torch.Tensor:
        """Raw window coefficients (before normalization)"""
        return self._coefficients

    @coefficients.setter
    def coefficients(self, v: Union[torch.Tensor, np.ndarray]) -> None:
        if not isinstance(v, torch.Tensor):
            v = torch.as_tensor(v, dtype=self.dtype, device=self.device)
        else:
            # Preserve gradient if already a tensor with requires_grad
            # Only convert if dtype or device differs
            if v.dtype != self.dtype or str(v.device) != self.device:
                v = v.to(dtype=self.dtype, device=self.device)
        self._coefficients = v

    @property
    def length(self) -> int:
        """Window length in number of samples"""
        return self.coefficients.shape[0]

    @property
    def normalize(self) -> bool:
        """If `True`, the window is normalized to have unit average power per coefficient"""
        return self._normalize

    def show(
        self,
        samples_per_symbol: int,
        domain: str = "time",
        scale: str = "lin",
    ) -> None:
        r"""Plot the window in time or frequency domain.

        For the computation of the Fourier transform, a minimum DFT size
        of 1024 is assumed which is obtained through zero padding of
        the window coefficients in the time domain.

        :param samples_per_symbol: Number of samples per symbol, i.e., the
            oversampling factor
        :param domain: Desired domain. Can be "time" or "frequency".
            Defaults to "time".
        :param scale: y-scale of the magnitude in the frequency domain.
            Can be "lin" (i.e., linear) or "db" (i.e., Decibel).
            Defaults to "lin".
        """
        if domain not in ("time", "frequency"):
            raise ValueError(
                f"domain must be 'time' or 'frequency', got {domain!r}"
            )

        # Normalize if requested
        w = self.coefficients
        if self.normalize:
            energy = torch.mean(w**2)
            w = w / torch.sqrt(energy)

        # Sampling times
        n_min = -(self.length // 2)
        n_max = n_min + self.length
        sampling_times = np.arange(n_min, n_max, dtype=np.float32)
        sampling_times /= samples_per_symbol

        w_np = w.detach().cpu().numpy()

        if domain == "time":
            plt.figure(figsize=(12, 6))
            plt.plot(sampling_times, np.real(w_np))
            plt.title("Time domain")
            plt.grid()
            plt.xlabel(r"Normalized time $(t/T)$")
            plt.ylabel(r"$w(t)$")
            plt.xlim(sampling_times[0], sampling_times[-1])
        else:
            if scale not in ("lin", "db"):
                raise ValueError(
                    f"scale must be 'lin' or 'db', got {scale!r}"
                )
            fft_size = max(1024, w.shape[-1])
            h = np.fft.fft(w_np, fft_size)
            h = np.fft.fftshift(h)
            h = np.abs(h)
            plt.figure(figsize=(12, 6))
            if scale == "db":
                h = np.maximum(h, 1e-10)
                h = 10 * np.log10(h)
                plt.ylabel(r"$|W(f)|$ (dB)")
            else:
                plt.ylabel(r"$|W(f)|$")
            f = np.linspace(-samples_per_symbol / 2, samples_per_symbol / 2, fft_size)
            plt.plot(f, h)
            plt.title("Frequency domain")
            plt.grid()
            plt.xlabel(r"Normalized frequency $(f/W)$")
            plt.xlim(f[0], f[-1])

    def call(self, x: torch.Tensor) -> torch.Tensor:
        w = self.coefficients

        # A window is defined sample-by-sample, so its length must match the
        # input. Without this check a length-one window broadcasts silently and
        # returns the input unchanged, and any other mismatch surfaces as a
        # low-level broadcasting error that names neither the window nor the
        # input.
        if x.shape[-1] != w.shape[-1]:
            raise ValueError(
                f"Input length ({x.shape[-1]}) must match the window length "
                f"({w.shape[-1]})"
            )

        # Normalize if requested
        if self.normalize:
            energy = torch.mean(w**2)
            w = w / torch.sqrt(energy)

        # Expand to the same rank as the input for broadcasting
        w = expand_to_rank(w, x.dim(), 0)

        # Apply window (w is automatically broadcast to match x's dtype)
        return w * x


class CustomWindow(Window):
    r"""Block for defining a custom window function.

    The window function is applied through element-wise multiplication.

    :param coefficients: Window coefficients with shape [N]
    :param normalize: If `True`, the window is normalized to have unit average
        power per coefficient. Defaults to `False`.
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`, :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for computation. If `None`,
        :attr:`~sionna.phy.config.Config.device` is used.

    :input x: [..., N], `torch.complex` or `torch.float`.
        Input to which the window function is applied.
        The window function is applied along the last dimension.
        The length of the last dimension ``N`` must be the same as the
        ``length`` of the window function.

    :output y: [..., N], `torch.complex` or `torch.float`.
        Output of the windowing operation.

    .. rubric:: Examples

    .. code-block:: python

        import torch
        from sionna.phy.signal import CustomWindow

        coefficients = torch.hann_window(64)
        window = CustomWindow(coefficients)
        x = torch.randn(32, 64)
        y = window(x)
        print(y.shape)
        # torch.Size([32, 64])
    """

    def __init__(
        self,
        coefficients: Union[torch.Tensor, np.ndarray],
        normalize: bool = False,
        precision: Optional[Precision] = None,
        device: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            normalize=normalize, precision=precision, device=device, **kwargs
        )
        self.coefficients = coefficients


class HannWindow(Window):
    r"""Block for defining a Hann window function.

    The window function is applied through element-wise multiplication.

    The Hann window is defined by

    .. math::
        w_n = \sin^2 \left( \frac{\pi n}{N} \right), 0 \leq n \leq N-1

    where :math:`N` is the window length.

    :param normalize: If `True`, the window is normalized to have unit average
        power per coefficient. Defaults to `False`.
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`, :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for computation. If `None`,
        :attr:`~sionna.phy.config.Config.device` is used.

    :input x: [..., N], `torch.complex` or `torch.float`.
        The input to which the window function is applied.
        The window function is applied along the last dimension.
        The length of the last dimension ``N`` must be the same
        as the ``length`` of the window function.

    :output y: [..., N], `torch.complex` or `torch.float`.
        Output of the windowing operation.

    .. rubric:: Examples

    .. code-block:: python

        import torch
        from sionna.phy.signal import HannWindow

        window = HannWindow()
        x = torch.randn(32, 64)
        y = window(x)
        print(y.shape)
        # torch.Size([32, 64])
    """

    def __init__(
        self,
        normalize: bool = False,
        precision: Optional[Precision] = None,
        device: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            normalize=normalize, precision=precision, device=device, **kwargs
        )

    def build(self, input_shape: tuple) -> None:
        length = input_shape[-1]
        n = np.arange(length)
        coefficients = np.square(np.sin(np.pi * n / length))
        self.coefficients = coefficients


class HammingWindow(Window):
    r"""Block for defining a Hamming window function.

    The window function is applied through element-wise multiplication.

    The Hamming window is defined by

    .. math::
        w_n = a_0 - (1-a_0) \cos \left( \frac{2 \pi n}{N} \right), 0 \leq n \leq N-1

    where :math:`N` is the window length and :math:`a_0 = \frac{25}{46}`.

    :param normalize: If `True`, the window is normalized to have unit average
        power per coefficient. Defaults to `False`.
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`, :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for computation. If `None`,
        :attr:`~sionna.phy.config.Config.device` is used.

    :input x: [..., N], `torch.complex` or `torch.float`.
        The input to which the window function is applied.
        The window function is applied along the last dimension.
        The length of the last dimension ``N`` must be the same
        as the ``length`` of the window function.

    :output y: [..., N], `torch.complex` or `torch.float`.
        Output of the windowing operation.

    .. rubric:: Examples

    .. code-block:: python

        import torch
        from sionna.phy.signal import HammingWindow

        window = HammingWindow()
        x = torch.randn(32, 64)
        y = window(x)
        print(y.shape)
        # torch.Size([32, 64])
    """

    def __init__(
        self,
        normalize: bool = False,
        precision: Optional[Precision] = None,
        device: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            normalize=normalize, precision=precision, device=device, **kwargs
        )

    def build(self, input_shape: tuple) -> None:
        n = input_shape[-1]
        nn = np.arange(n)
        a0 = 25.0 / 46.0
        a1 = 1.0 - a0
        coefficients = a0 - a1 * np.cos(2.0 * np.pi * nn / n)
        self.coefficients = coefficients


class BlackmanWindow(Window):
    r"""Block for defining a Blackman window function.

    The window function is applied through element-wise multiplication.

    The Blackman window is defined by

    .. math::
        w_n = a_0 - a_1 \cos \left( \frac{2 \pi n}{N} \right) + a_2 \cos \left( \frac{4 \pi n}{N} \right), 0 \leq n \leq N-1

    where :math:`N` is the window length, :math:`a_0 = \frac{7938}{18608}`,
    :math:`a_1 = \frac{9240}{18608}`, and :math:`a_2 = \frac{1430}{18608}`.

    :param normalize: If `True`, the window is normalized to have unit average
        power per coefficient. Defaults to `False`.
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`, :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for computation. If `None`,
        :attr:`~sionna.phy.config.Config.device` is used.

    :input x: [..., N], `torch.complex` or `torch.float`.
        The input to which the window function is applied.
        The window function is applied along the last dimension.
        The length of the last dimension ``N`` must be the same
        as the ``length`` of the window function.

    :output y: [..., N], `torch.complex` or `torch.float`.
        Output of the windowing operation.

    .. rubric:: Examples

    .. code-block:: python

        import torch
        from sionna.phy.signal import BlackmanWindow

        window = BlackmanWindow()
        x = torch.randn(32, 64)
        y = window(x)
        print(y.shape)
        # torch.Size([32, 64])
    """

    def __init__(
        self,
        normalize: bool = False,
        precision: Optional[Precision] = None,
        device: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            normalize=normalize, precision=precision, device=device, **kwargs
        )

    def build(self, input_shape: tuple) -> None:
        n = input_shape[-1]
        nn = np.arange(n)
        a0 = 7938.0 / 18608.0
        a1 = 9240.0 / 18608.0
        a2 = 1430.0 / 18608.0
        coefficients = (
            a0 - a1 * np.cos(2.0 * np.pi * nn / n) + a2 * np.cos(4.0 * np.pi * nn / n)
        )
        self.coefficients = coefficients
