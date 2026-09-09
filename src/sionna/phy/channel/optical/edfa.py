#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Model for an Erbium-Doped Fiber Amplifier."""

from typing import Optional

import torch

from sionna.phy import Block, H
from sionna.phy.utils import normal


__all__ = ["EDFA"]


class EDFA(Block):
    # pylint: disable=line-too-long
    r"""
    Block implementing a model of an Erbium-Doped Fiber Amplifier

    Amplifies the optical input signal by a given gain and adds
    amplified spontaneous emission (ASE) noise.

    The noise figure including the noise due to beating of signal and
    spontaneous emission is :math:`F_\mathrm{ASE,shot} =\frac{\mathrm{SNR}
    _\mathrm{in}}{\mathrm{SNR}_\mathrm{out}}`,
    where ideally the detector is limited by shot noise only, and
    :math:`\text{SNR}` is the signal-to-noise-ratio. Shot noise is
    neglected here but is required to derive the noise power of the amplifier, as
    otherwise the input SNR is infinitely large. Hence, for the input SNR,
    it follows :cite:p:`A2012` that
    :math:`\mathrm{SNR}_\mathrm{in}=\frac{P}{2hf_cW}`, where :math:`h` denotes
    Planck's constant, :math:`P` is the signal power, and :math:`W` the
    considered bandwidth.
    The output SNR is decreased by ASE noise induced by the amplification.
    Note that shot noise is applied after the amplifier and is hence not
    amplified. It results that :math:`\mathrm{SNR}_\mathrm{out}=\frac{GP}{\left
    (4\rho_\mathrm{ASE}+2hf_c\right)W}`, where :math:`G` is the
    parametrized gain.
    Hence, one can write the former equation as :math:`F_\mathrm{ASE,shot} = 2
    n_\mathrm{sp} \left(1-G^{-1}\right) + G^{-1}`.
    Dropping shot noise again results in :math:`F = 2 n_\mathrm{sp} \left(1-G^
    {-1}\right)=2 n_\mathrm{sp} \frac{G-1}{G}`.

    For a transparent link, e.g., the required gain per span is :math:`G =
    \exp\left(\alpha \ell \right)`.
    The spontaneous emission factor is :math:`n_\mathrm{sp}=\frac{F}
    {2}\frac{G}{G-1}`.
    According to :cite:p:`A2012` and :cite:p:`EKWFG2010` combined with :cite:p:`BGT2000` and :cite:p:`GD1991`,
    the noise power spectral density of the EDFA per state of
    polarization is obtained as :math:`\rho_\mathrm{ASE}^{(1)} = n_\mathrm{sp}\left
    (G-1\right) h f_c=\frac{1}{2}G F h f_c`.
    At simulation frequency :math:`f_\mathrm{sim}`, the noise has a power of
    :math:`P_\mathrm{ASE}^{(1)}=\sigma_\mathrm{n,ASE}^2=2\rho_\mathrm{ASE}^{(1)}
    \cdot f_\mathrm{sim}`,
    where the factor :math:`2` accounts for the unpolarized noise (for dual
    polarization the factor is :math:`1` per polarization).
    Here, the notation :math:`()^{(1)}` means that this is the noise introduced by a
    single EDFA.

    :param g: Amplifier gain (linear domain). Defaults to 4.0.
    :param f: Noise figure (linear domain). Defaults to 7.0.
    :param f_c: Carrier frequency :math:`f_\mathrm{c}` in :math:`(\text{Hz})`.
        Defaults to 193.55e12.
    :param dt: Time step :math:`\Delta_t` in :math:`(\text{s})`. Defaults to 1e-12.
    :param with_dual_polarization: If `True`, considers axis [-2] as x- and
        y-polarization and applies the noise per polarization. Defaults to `False`.
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`, :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for computation. If `None`,
        :attr:`~sionna.phy.config.Config.device` is used.

    :input x: Tensor, `torch.complex`.
        Optical input signal.

    :output y: Tensor (same shape as ``x``), `torch.complex`.
        Amplifier output.

    .. rubric:: Examples

    .. code-block:: python

        import torch
        from sionna.phy.channel.optical import EDFA

        edfa = EDFA(
            g=4.0,
            f=2.0,
            f_c=193.55e12,
            dt=1.0e-12,
            with_dual_polarization=False)

        # x is the optical input signal
        x = torch.randn(10, 100, dtype=torch.complex64)
        y = edfa(x)
        print(y.shape)
        # torch.Size([10, 100])
    """

    def __init__(
        self,
        g: float = 4.0,
        f: float = 7.0,
        f_c: float = 193.55e12,
        dt: float = 1e-12,
        with_dual_polarization: bool = False,
        precision: Optional[str] = None,
        device: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(precision=precision, device=device, **kwargs)

        # Register as buffers for CUDAGraph compatibility
        self.register_buffer("_g", torch.tensor(g, dtype=self.dtype, device=self.device))
        self.register_buffer("_f", torch.tensor(f, dtype=self.dtype, device=self.device))
        self.register_buffer("_f_c", torch.tensor(f_c, dtype=self.dtype, device=self.device))
        self.register_buffer("_dt", torch.tensor(dt, dtype=self.dtype, device=self.device))

        if not isinstance(with_dual_polarization, bool):
            raise TypeError("with_dual_polarization must be bool.")
        self._with_dual_polarization = with_dual_polarization

        # Spontaneous emission factor
        if self._g == 1.0:
            self.register_buffer("_n_sp", torch.tensor(0.0, dtype=self.dtype, device=self.device))
        else:
            self.register_buffer("_n_sp", self._f / 2.0 * self._g / (self._g - 1.0))

        self._rho_n_ase = (
            self._n_sp * (self._g - 1.0) * H * self._f_c
        )  # Noise density in (W/Hz)

        self._p_n_ase = 2.0 * self._rho_n_ase / self._dt  # Noise power in (W)

        if self._with_dual_polarization:
            self._p_n_ase = self._p_n_ase / 2.0

    def call(self, inputs: torch.Tensor) -> torch.Tensor:
        """Process the optical input signal through the EDFA.

        :param inputs: Optical input signal

        :output y: Amplified signal with ASE noise
        """
        if self._with_dual_polarization:
            if inputs.dim() < 2 or inputs.shape[-2] != 2:
                raise ValueError(
                    "For dual polarization, second to last dimension must be 2."
                )

        x = inputs.to(dtype=self.cdtype, device=self.device)

        # Calculate noise signal with given noise power
        # Uses smart randn that switches to global RNG in compiled mode
        noise_std = torch.sqrt(self._p_n_ase / 2.0)
        n_real = normal(
            x.shape, dtype=self.dtype, device=self.device, generator=self.torch_rng
        ) * noise_std
        n_imag = normal(
            x.shape, dtype=self.dtype, device=self.device, generator=self.torch_rng
        ) * noise_std
        n = torch.complex(n_real, n_imag)

        # Amplify signal
        x = x * torch.sqrt(self._g).to(self.cdtype)

        # Add noise signal
        y = x + n

        return y
