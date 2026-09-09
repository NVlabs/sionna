#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Split-step Fourier method to approximate the solution of the nonlinear
Schroedinger equation."""

from typing import Optional, Union

import torch

from sionna.phy import Block, H, PI
from sionna.phy.channel import utils
from sionna.phy.utils import normal


__all__ = ["SSFM"]


class SSFM(Block):
    # pylint: disable=line-too-long
    r"""
    Block implementing the split-step Fourier method (SSFM)

    The SSFM (first mentioned in :cite:p:`HT1973`) numerically solves the generalized
    nonlinear Schrödinger equation (NLSE)

    .. math::

        \frac{\partial E(t,z)}{\partial z}=-\frac{\alpha}{2} E(t,z)+j\frac{\beta_2}{2}\frac{\partial^2 E(t,z)}{\partial t^2}-j\gamma |E(t,z)|^2 E(t,z) + n(n_{\text{sp}};\,t,\,z)

    for an unpolarized (or single polarized) optical signal;
    or the Manakov equation (according to :cite:p:`WMC1991`)

    .. math::

        \frac{\partial \mathbf{E}(t,z)}{\partial z}=-\frac{\alpha}{2} \mathbf{E}(t,z)+j\frac{\beta_2}{2}\frac{\partial^2 \mathbf{E}(t,z)}{\partial t^2}-j\gamma \frac{8}{9}||\mathbf{E}(t,z)||_2^2 \mathbf{E}(t,z) + \mathbf{n}(n_{\text{sp}};\,t,\,z)

    for dual polarization, with attenuation coefficient :math:`\alpha`, group
    velocity dispersion parameters :math:`\beta_2`, and nonlinearity
    coefficient :math:`\gamma`. The noise terms :math:`n(n_{\text{sp}};\,t,\,z)`
    and :math:`\mathbf{n}(n_{\text{sp}};\,t,\,z)`, respectively, stem from
    an (optional) ideally distributed Raman amplification with
    spontaneous emission factor :math:`n_\text{sp}`. The optical signal
    :math:`E(t,\,z)` has the unit :math:`\sqrt{\text{W}}`. For the dual
    polarized case, :math:`\mathbf{E}(t,\,z)=(E_x(t,\,z), E_y(t,\,z))`
    is a vector consisting of the signal components of both polarizations.

    The symmetrized SSFM is applied according to Eq. (7) of :cite:p:`FMF1976` that
    can be written as

    .. math::

        E(z+\Delta_z,t) \approx \exp\left(\frac{\Delta_z}{2}\hat{D}\right)\exp\left(\int^{z+\Delta_z}_z \hat{N}(z')dz'\right)\exp\left(\frac{\Delta_z}{2}\hat{D}\right)E(z,\,t)

    where only the single-polarized case is shown. The integral is
    approximated by :math:`\Delta_z\hat{N}` with :math:`\hat{D}` and
    :math:`\hat{N}` denoting the linear and nonlinear SSFM operator,
    respectively :cite:p:`A2012`.

    Additionally, ideally distributed Raman amplification may be applied, which
    is implemented as in :cite:p:`MFFP2009`. Please note that the implemented
    Raman amplification currently results in a transparent fiber link. Hence,
    the introduced gain cannot be parametrized.

    The SSFM operates on normalized time :math:`T_\text{norm}`
    (e.g., :math:`T_\text{norm}=1\,\text{ps}=1\cdot 10^{-12}\,\text{s}`) and
    distance units :math:`L_\text{norm}`
    (e.g., :math:`L_\text{norm}=1\,\text{km}=1\cdot 10^{3}\,\text{m}`).
    Hence, all parameters as well as the signal itself have to be given with the
    same unit prefix for the
    same unit (e.g., always pico for time, or kilo for distance). Despite the normalization,
    the SSFM is implemented with physical
    units, which is different from the normalization, e.g., used for the
    nonlinear Fourier transform. For simulations, only :math:`T_\text{norm}` has to be
    provided.

    To avoid reflections at the signal boundaries during simulation, a Hamming
    window can be applied in each SSFM-step, whose length can be
    defined by ``half_window_length``.

    :param alpha: Attenuation coefficient :math:`\alpha` in :math:`(1/L_\text{norm})`.
        Defaults to 0.046.
    :param beta_2: Group velocity dispersion coefficient :math:`\beta_2` in
        :math:`(T_\text{norm}^2/L_\text{norm})`. Defaults to -21.67.
    :param f_c: Carrier frequency :math:`f_\mathrm{c}` in :math:`(\text{Hz})`.
        Defaults to 193.55e12.
    :param gamma: Nonlinearity coefficient :math:`\gamma` in
        :math:`(1/L_\text{norm}/\text{W})`. Defaults to 1.27.
    :param half_window_length: Half of the Hamming window length. Defaults to 0.
    :param length: Fiber length :math:`\ell` in :math:`(L_\text{norm})`.
        Defaults to 80.0.
    :param n_ssfm: Number of steps :math:`N_\mathrm{SSFM}`.
        Set to "adaptive" to use nonlinear-phase rotation to calculate
        the step widths adaptively (maximum rotation can be set in
        ``phase_inc``). Defaults to 1.
    :param n_sp: Spontaneous emission factor :math:`n_\mathrm{sp}`
        of Raman amplification. Defaults to 1.0.
    :param sample_duration: Normalized time step :math:`\Delta_t` in
        :math:`(T_\text{norm})`. Defaults to 1.0.
    :param t_norm: Time normalization :math:`T_\text{norm}` in :math:`(\text{s})`.
        Defaults to 1e-12.
    :param with_amplification: If `True`, enables ideal inline amplification
        and corresponding noise. Defaults to `False`.
    :param with_attenuation: If `True`, enables attenuation. Defaults to `True`.
    :param with_dispersion: If `True`, applies chromatic dispersion.
        Defaults to `True`.
    :param with_manakov: If `True`, considers axis [-2] as x- and y-polarization
        and calculates the nonlinear step as given by the Manakov equation.
        Defaults to `False`.
    :param with_nonlinearity: If `True`, applies Kerr nonlinearity.
        Defaults to `True`.
    :param phase_inc: Maximum nonlinear-phase rotation in rad allowed during
        simulation. To be used with ``n_ssfm`` = "adaptive". Defaults to 1e-4.
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`, :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for computation. If `None`,
        :attr:`~sionna.phy.config.Config.device` is used.

    :input x: [..., n] or [..., 2, n], `torch.complex`.
        Input signal in :math:`(\sqrt{\text{W}})`. If ``with_manakov``
        is `True`, the second last dimension is interpreted
        as x- and y-polarization, respectively.

    :output y: Tensor (same shape as ``x``), `torch.complex`.
        Channel output.

    .. rubric:: Examples

    .. code-block:: python

        import torch
        from sionna.phy.channel.optical import SSFM

        ssfm = SSFM(
            alpha=0.046,
            beta_2=-21.67,
            f_c=193.55e12,
            gamma=1.27,
            half_window_length=100,
            length=80,
            n_ssfm=200,
            n_sp=1.0,
            t_norm=1e-12,
            with_amplification=False,
            with_attenuation=True,
            with_dispersion=True,
            with_manakov=False,
            with_nonlinearity=True)

        # x is the optical input signal
        x = torch.randn(10, 1024, dtype=torch.complex64)
        y = ssfm(x)
        print(y.shape)
        # torch.Size([10, 1024])
    """

    def __init__(
        self,
        alpha: float = 0.046,
        beta_2: float = -21.67,
        f_c: float = 193.55e12,
        gamma: float = 1.27,
        half_window_length: int = 0,
        length: float = 80,
        n_ssfm: Union[int, str] = 1,
        n_sp: float = 1.0,
        sample_duration: float = 1.0,
        t_norm: float = 1e-12,
        with_amplification: bool = False,
        with_attenuation: bool = True,
        with_dispersion: bool = True,
        with_manakov: bool = False,
        with_nonlinearity: bool = True,
        phase_inc: float = 1e-4,
        precision: Optional[str] = None,
        device: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(precision=precision, device=device, **kwargs)

        # Register as buffers for CUDAGraph compatibility
        self.register_buffer("_alpha", torch.tensor(alpha, dtype=self.dtype, device=self.device))
        self.register_buffer("_beta_2", torch.tensor(beta_2, dtype=self.dtype, device=self.device))
        self.register_buffer("_f_c", torch.tensor(f_c, dtype=self.dtype, device=self.device))
        self.register_buffer("_gamma", torch.tensor(gamma, dtype=self.dtype, device=self.device))
        self._half_window_length = half_window_length
        self.register_buffer("_length", torch.tensor(length, dtype=self.dtype, device=self.device))
        self.register_buffer("_phase_inc", torch.tensor(phase_inc, dtype=self.dtype, device=self.device))

        if n_ssfm == "adaptive":
            self._n_ssfm = -1  # adaptive == -1
            self._adaptive = True
        elif isinstance(n_ssfm, int):
            if n_ssfm <= 0:
                raise ValueError("n_ssfm must be positive")
            self._n_ssfm = n_ssfm
            self._adaptive = False
        else:
            raise ValueError(
                "Unsupported parameter for n_ssfm. Either an integer or 'adaptive'."
            )

        # Register as buffers for CUDAGraph compatibility
        self.register_buffer("_n_sp", torch.tensor(n_sp, dtype=self.dtype, device=self.device))
        self.register_buffer("_t_norm", torch.tensor(t_norm, dtype=self.dtype, device=self.device))
        self.register_buffer("_sample_duration", torch.tensor(
            sample_duration, dtype=self.dtype, device=self.device
        ))

        # Booleans
        self._with_amplification = with_amplification
        self._with_attenuation = with_attenuation
        self._with_dispersion = with_dispersion
        self._with_manakov = with_manakov
        self._with_nonlinearity = with_nonlinearity

        self._update_derived_state()
        self.register_load_state_dict_post_hook(self._refresh_derived_state)

        # Pre-compute Hamming window
        if self._half_window_length > 0:
            self.register_buffer("_window", torch.hamming_window(
                2 * self._half_window_length,
                dtype=self.dtype,
                device=self.device,
            ))
        else:
            self.register_buffer("_window", None)

    def _update_derived_state(self) -> None:
        """Update values derived from registered buffers."""
        # Only used for constant step width
        if not self._adaptive:
            self._dz = self._length / self._n_ssfm

        # `time_frequency_vector()` takes a float, and reading it back with
        # `.item()` inside `call()` forces a Dynamo graph break under
        # `torch.compile`. Read it off the buffer so it carries the rounding of
        # `self.dtype`, which affects the frequency spacing at single precision.
        self._sample_duration_value = self._sample_duration.item()

        self._rho_n = H * self._f_c * self._alpha * self._length * self._n_sp
        self._p_n_ase = self._rho_n / self._sample_duration / self._t_norm
        if self._with_manakov:
            self._p_n_ase = self._p_n_ase / 2.0

    def _refresh_derived_state(self, module, incompatible_keys) -> None:
        """Refresh values derived from buffers after loading state."""
        del module, incompatible_keys
        self._update_derived_state()

    def _apply_linear_operator(
        self,
        q: torch.Tensor,
        dz: torch.Tensor,
        frequency_vector: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the linear operator (dispersion and attenuation/amplification)."""
        # Chromatic dispersion
        if self._with_dispersion:
            dispersion_phase = (
                -self._beta_2 / 2.0 * dz * (2.0 * PI * frequency_vector) ** 2
            )
            dispersion = torch.exp(
                torch.complex(
                    torch.zeros_like(dispersion_phase),
                    dispersion_phase,
                )
            )
            dispersion = torch.fft.fftshift(dispersion, dim=-1)
            q = torch.fft.ifft(torch.fft.fft(q) * dispersion)

        # Attenuation
        if self._with_attenuation:
            q = q * torch.exp(-self._alpha / 2.0 * dz).to(self.cdtype)

        # Amplification (Raman)
        if self._with_amplification:
            q = q * torch.exp(self._alpha / 2.0 * dz).to(self.cdtype)

        return q

    def _apply_noise(self, q: torch.Tensor, dz: torch.Tensor) -> torch.Tensor:
        """Apply noise due to Raman amplification."""
        if self._with_amplification:
            step_noise = self._p_n_ase * dz / self._length / 2.0
            noise_std = torch.sqrt(step_noise)
            # Uses smart randn that switches to global RNG in compiled mode
            q_n_real = normal(
                q.shape, dtype=self.dtype, device=self.device, generator=self.torch_rng
            ) * noise_std
            q_n_imag = normal(
                q.shape, dtype=self.dtype, device=self.device, generator=self.torch_rng
            ) * noise_std
            q_n = torch.complex(q_n_real, q_n_imag)
            q = q + q_n

        return q

    def _apply_nonlinear_operator(
        self,
        q: torch.Tensor,
        dz: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the nonlinear operator (Kerr effect)."""
        if self._with_nonlinearity:
            if self._with_manakov:
                # Sum over polarizations
                power = (q.abs() ** 2).sum(dim=-2, keepdim=True)
                phase = 8.0 / 9.0 * power * self._gamma * (-dz.real)
                q = q * torch.exp(torch.complex(torch.zeros_like(phase), phase))
            else:
                power = q.abs() ** 2
                phase = power * self._gamma * (-dz.real)
                q = q * torch.exp(torch.complex(torch.zeros_like(phase), phase))

        return q

    def _calculate_step_width(
        self, q: torch.Tensor, remaining_length: torch.Tensor
    ) -> torch.Tensor:
        """Calculate adaptive step width based on maximum power."""
        max_power = (q.abs() ** 2).max()
        # Ensure that the exact length is reached in the end
        dz = torch.minimum(
            self._phase_inc / self._gamma / max_power,
            remaining_length,
        )
        return dz

    def _apply_window(self, q: torch.Tensor, window: torch.Tensor) -> torch.Tensor:
        """Apply windowing function."""
        return q * window

    def _build_window(self, signal_length: int) -> torch.Tensor:
        """Build the complete window for the signal length."""
        if self._half_window_length == 0 or self._window is None:
            return torch.ones(signal_length, dtype=self.cdtype, device=self.device)

        # Build window: [hamming_left | ones | hamming_right]
        window = torch.cat(
            [
                self._window[: self._half_window_length].to(self.cdtype),
                torch.ones(
                    signal_length - 2 * self._half_window_length,
                    dtype=self.cdtype,
                    device=self.device,
                ),
                self._window[self._half_window_length :].to(self.cdtype),
            ],
            dim=0,
        )
        return window

    def call(self, inputs: torch.Tensor) -> torch.Tensor:
        """Process the optical input signal through the fiber.

        :param inputs: Optical input signal

        :output x: Channel output after fiber propagation
        """
        if self._with_manakov:
            if inputs.dim() < 2 or inputs.shape[-2] != 2:
                raise ValueError(
                    "For Manakov mode, second to last dimension must be 2."
                )

        x = inputs.to(dtype=self.cdtype, device=self.device)
        input_shape = x.shape

        # Generate frequency vectors
        _, f = utils.time_frequency_vector(
            input_shape[-1],
            self._sample_duration_value,
            precision=self.precision,
            device=self.device,
        )

        # Window function calculation (depends on length of the signal)
        window = self._build_window(input_shape[-1])

        if self._adaptive:
            # Adaptive step width
            remaining_length = self._length.clone()

            while remaining_length >= 1e-3:  # Avoid numerical issues for 0
                dz = self._calculate_step_width(x, remaining_length)

                # Apply window-function
                x = self._apply_window(x, window)
                x = self._apply_linear_operator(x, dz, f)
                x = self._apply_nonlinear_operator(x, dz)
                x = self._apply_noise(x, dz)
                remaining_length = remaining_length - dz

        else:
            # Constant step size
            dz = self._dz
            dz_half = dz / 2.0

            # Symmetric SSFM
            # Start with half linear propagation
            x = self._apply_linear_operator(x, dz_half, f)

            # Proceed with N_SSFM-1 steps applying nonlinear and linear operator
            for _ in range(self._n_ssfm - 1):
                x = self._apply_window(x, window)
                x = self._apply_nonlinear_operator(x, dz)
                x = self._apply_noise(x, dz)
                x = self._apply_linear_operator(x, dz, f)

            # Final nonlinear operator
            x = self._apply_nonlinear_operator(x, dz)
            # Final noise application
            x = self._apply_noise(x, dz)
            # End with half linear propagation
            x = self._apply_linear_operator(x, dz_half, f)

        return x
