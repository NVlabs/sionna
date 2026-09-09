#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Utility functions for the channel module"""

import warnings
from typing import Optional, Tuple

import torch

from sionna._validation import check_tensor_all
from sionna.phy import PI, config, dtypes
from sionna.phy.utils import expand_to_rank, sample_bernoulli
from sionna.phy.utils.random import rand, randint

__all__ = [
    "subcarrier_frequencies",
    "time_frequency_vector",
    "time_lag_discrete_time_channel",
    "cir_to_ofdm_channel",
    "cir_to_time_channel",
    "time_to_ofdm_channel",
    "deg_2_rad",
    "rad_2_deg",
    "wrap_angle_0_360",
    "drop_uts_in_sector",
    "set_3gpp_scenario_parameters",
    "relocate_uts",
    "generate_uts_topology",
    "random_ut_properties",
    "gen_single_sector_topology",
    "gen_single_sector_topology_interferers",
    "exp_corr_mat",
    "one_ring_corr_mat",
]


def subcarrier_frequencies(
    num_subcarriers: int,
    subcarrier_spacing: float,
    precision: Optional[str] = None,
    device: Optional[str] = None,
) -> torch.Tensor:
    r"""
    Compute the baseband frequencies of ``num_subcarrier`` subcarriers spaced by
    ``subcarrier_spacing``, i.e.,

    >>> # If num_subcarrier is even:
    >>> frequencies = [-num_subcarrier/2, ..., 0, ..., num_subcarrier/2-1] * subcarrier_spacing
    >>>
    >>> # If num_subcarrier is odd:
    >>> frequencies = [-(num_subcarrier-1)/2, ..., 0, ..., (num_subcarrier-1)/2] * subcarrier_spacing

    :param num_subcarriers: Number of subcarriers
    :param subcarrier_spacing: Subcarrier spacing [Hz]
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`, :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for computation. If `None`,
        :attr:`~sionna.phy.config.Config.device` is used.

    :output frequencies: [``num_subcarrier``], `torch.float`.
        Baseband frequencies of subcarriers.

    .. rubric:: Examples

    .. code-block:: python

        from sionna.phy.channel import subcarrier_frequencies

        # 64 subcarriers with 15 kHz spacing
        freqs = subcarrier_frequencies(64, 15e3)
        print(freqs.shape)
        # torch.Size([64])
    """
    if precision is None:
        dtype = config.dtype
    else:
        dtype = dtypes[precision]["torch"]["dtype"]

    if device is None:
        device = config.device

    start = -(num_subcarriers // 2)
    if num_subcarriers % 2 == 0:
        limit = num_subcarriers // 2
    else:
        limit = num_subcarriers // 2 + 1

    frequencies = torch.arange(start, limit, dtype=dtype, device=device)
    frequencies = frequencies * subcarrier_spacing
    return frequencies


def time_frequency_vector(
    num_samples: int,
    sample_duration: float,
    precision: Optional[str] = None,
    device: Optional[str] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    Compute the time and frequency vector for a given number of samples
    and duration per sample in normalized time unit.

    >>> t = torch.linspace(-n_min, n_max, num_samples) * sample_duration
    >>> f = torch.linspace(-n_min, n_max, num_samples) * 1/(sample_duration*num_samples)

    :param num_samples: Number of samples
    :param sample_duration: Sample duration in normalized time
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`, :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for computation. If `None`,
        :attr:`~sionna.phy.config.Config.device` is used.

    :output t: [``num_samples``], `torch.float`.
        Time vector.

    :output f: [``num_samples``], `torch.float`.
        Frequency vector.

    .. rubric:: Examples

    .. code-block:: python

        from sionna.phy.channel import time_frequency_vector

        t, f = time_frequency_vector(128, 1e-6)
        print(t.shape, f.shape)
        # torch.Size([128]) torch.Size([128])
    """
    if precision is None:
        dtype = config.dtype
    else:
        dtype = dtypes[precision]["torch"]["dtype"]

    if device is None:
        device = config.device

    num_samples = int(num_samples)

    if num_samples % 2 == 0:  # if even
        n_min = -num_samples // 2
        n_max = num_samples // 2 - 1
    else:  # if odd
        n_min = -(num_samples - 1) // 2
        n_max = (num_samples + 1) // 2 - 1

    # Time vector
    indices = torch.linspace(n_min, n_max, num_samples, dtype=dtype, device=device)
    t = indices * sample_duration

    # Frequency vector
    df = 1.0 / sample_duration / num_samples
    f = indices * df

    return t, f


def time_lag_discrete_time_channel(
    bandwidth: float,
    maximum_delay_spread: float = 3e-6,
) -> Tuple[int, int]:
    r"""
    Compute the smallest and largest time-lag for the discrete complex baseband
    channel, i.e., :math:`L_{\text{min}}` and :math:`L_{\text{max}}`.

    The smallest time-lag (:math:`L_{\text{min}}`) returned is always -6, as this value
    was found small enough for all models included in Sionna.

    The largest time-lag (:math:`L_{\text{max}}`) is computed from the ``bandwidth``
    and ``maximum_delay_spread`` as follows:

    .. math::
        L_{\text{max}} = \lceil W \tau_{\text{max}} \rceil + 6

    where :math:`L_{\text{max}}` is the largest time-lag, :math:`W` the ``bandwidth``,
    and :math:`\tau_{\text{max}}` the ``maximum_delay_spread``.

    The default value for the ``maximum_delay_spread`` is 3us, which was found
    to be large enough to include most significant paths with all channel models
    included in Sionna assuming a nominal delay spread of 100ns.

    .. rubric:: Notes

    The values of :math:`L_{\text{min}}` and :math:`L_{\text{max}}` computed
    by this function are only recommended values.
    :math:`L_{\text{min}}` and :math:`L_{\text{max}}` should be set according to
    the considered channel model. For OFDM systems, one also needs to be careful
    that the effective length of the complex baseband channel is not larger than
    the cyclic prefix length.

    :param bandwidth: Bandwidth (:math:`W`) [Hz]
    :param maximum_delay_spread: Maximum delay spread [s]. Defaults to 3e-6.

    :output l_min: `int`.
        Smallest time-lag (:math:`L_{\text{min}}`) for the discrete complex
        baseband channel. Set to -6, as this value was found small enough for
        all models included in Sionna.

    :output l_max: `int`.
        Largest time-lag (:math:`L_{\text{max}}`) for the discrete complex
        baseband channel.

    .. rubric:: Examples

    .. code-block:: python

        from sionna.phy.channel import time_lag_discrete_time_channel

        l_min, l_max = time_lag_discrete_time_channel(20e6)
        print(l_min, l_max)
        # -6 66
    """
    import math

    l_min = -6
    l_max = int(math.ceil(maximum_delay_spread * bandwidth) + 6)
    return l_min, l_max


def cir_to_ofdm_channel(
    frequencies: torch.Tensor,
    a: torch.Tensor,
    tau: torch.Tensor,
    normalize: bool = False,
) -> torch.Tensor:
    r"""
    Compute the frequency response of the channel at ``frequencies``

    Given a channel impulse response
    :math:`(a_{m}, \tau_{m}), 0 \leq m \leq M-1` (inputs ``a`` and ``tau``),
    the channel frequency response for the frequency :math:`f`
    is computed as follows:

    .. math::
        \widehat{h}(f) = \sum_{m=0}^{M-1} a_{m} e^{-j2\pi f \tau_{m}}

    :param frequencies: Frequencies at which to compute the channel response,
        shape [fft_size]
    :param a: Path coefficients, shape
        [batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_steps]
    :param tau: Path delays, shape [batch size, num_rx, num_tx, num_paths] or
        [batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths]
    :param normalize: If set to `True`, the channel is normalized over the
        resource grid

    :output h_f: [batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_time_steps, fft_size], `torch.complex`.
        Channel frequency responses at ``frequencies``.

    .. rubric:: Examples

    .. code-block:: python

        import torch
        from sionna.phy.channel import cir_to_ofdm_channel, subcarrier_frequencies

        # Create dummy CIR
        batch_size, num_paths, num_time_steps = 2, 4, 10
        a = torch.randn(batch_size, 1, 1, 1, 1, num_paths, num_time_steps, dtype=torch.complex64)
        tau = torch.rand(batch_size, 1, 1, num_paths) * 1e-6

        # Compute OFDM channel
        frequencies = subcarrier_frequencies(64, 15e3)
        h_f = cir_to_ofdm_channel(frequencies, a, tau)
        print(h_f.shape)
        # torch.Size([2, 1, 1, 1, 1, 10, 64])
    """
    real_dtype = tau.dtype

    if tau.dim() == 4:
        # Expand dims to broadcast with h. Add the following dimensions:
        #  - number of rx antennas (2)
        #  - number of tx antennas (4)
        tau = tau.unsqueeze(2).unsqueeze(4)
        # Manually tile for high-rank tensor broadcasting
        tau = tau.expand(-1, -1, -1, -1, a.shape[4], -1)

    # Add a time samples dimension for broadcasting
    tau = tau.unsqueeze(6)

    # Bring all tensors to broadcastable shapes
    tau = tau.unsqueeze(-1)
    # Ensure frequencies is on the same device as tau
    frequencies = frequencies.to(device=tau.device, dtype=tau.dtype)
    frequencies = expand_to_rank(frequencies, tau.dim(), axis=0)

    # Compute the Fourier transforms of all cluster taps
    # Exponential component
    e = torch.exp(
        torch.complex(
            torch.zeros_like(tau),
            -2 * PI * frequencies * tau,
        )
    )
    # Multiply and sum over all paths without materializing the broadcast product
    h_f = torch.einsum("...pn,...puf->...nf", a, e)

    if normalize:
        # Normalization is performed such that for each batch example and
        # link the energy per resource grid is one.
        # Average over TX antennas, RX antennas, OFDM symbols and subcarriers.
        c = h_f.abs().square().mean(dim=(2, 4, 5, 6), keepdim=True)
        c = torch.complex(c.sqrt(), torch.zeros_like(c))
        h_f = torch.where(c.abs() > 0, h_f / c, h_f)

    return h_f


def cir_to_time_channel(
    bandwidth: float,
    a: torch.Tensor,
    tau: torch.Tensor,
    l_min: int,
    l_max: int,
    normalize: bool = False,
) -> torch.Tensor:
    r"""
    Compute the channel taps forming the discrete complex-baseband
    representation of the channel from the channel impulse response
    (``a``, ``tau``)

    This function assumes that a sinc filter is used for pulse shaping and receive
    filtering. Therefore, given a channel impulse response
    :math:`(a_{m}(t), \tau_{m}), 0 \leq m \leq M-1`, the channel taps
    are computed as follows:

    .. math::
        \bar{h}_{b, \ell}
        = \sum_{m=0}^{M-1} a_{m}\left(\frac{b}{W}\right)
            \text{sinc}\left( \ell - W\tau_{m} \right)

    for :math:`\ell` ranging from ``l_min`` to ``l_max``, and where :math:`W` is
    the ``bandwidth``.

    :param bandwidth: Bandwidth [Hz]
    :param a: Path coefficients, shape
        [batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_steps]
    :param tau: Path delays [s], shape [batch size, num_rx, num_tx, num_paths] or
        [batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths]
    :param l_min: Smallest time-lag for the discrete complex baseband channel
        (:math:`L_{\text{min}}`)
    :param l_max: Largest time-lag for the discrete complex baseband channel
        (:math:`L_{\text{max}}`)
    :param normalize: If set to `True`, the channel is normalized over the
        block size to ensure unit average energy per time step

    :output hm: [batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_time_steps, l_max - l_min + 1], `torch.complex`.
        Channel taps coefficients.

    .. rubric:: Examples

    .. code-block:: python

        import torch
        from sionna.phy.channel import cir_to_time_channel

        # Create dummy CIR
        batch_size, num_paths, num_time_steps = 2, 4, 10
        a = torch.randn(batch_size, 1, 1, 1, 1, num_paths, num_time_steps, dtype=torch.complex64)
        tau = torch.rand(batch_size, 1, 1, num_paths) * 1e-6

        # Compute time channel
        h_t = cir_to_time_channel(20e6, a, tau, l_min=-6, l_max=20)
        print(h_t.shape)
        # torch.Size([2, 1, 1, 1, 1, 10, 27])
    """
    real_dtype = tau.dtype

    if tau.dim() == 4:
        # Expand dims to broadcast with h. Add the following dimensions:
        #  - number of rx antennas (2)
        #  - number of tx antennas (4)
        tau = tau.unsqueeze(2).unsqueeze(4)
        # Manually tile for high-rank tensor broadcasting
        tau = tau.expand(-1, -1, -1, -1, a.shape[4], -1)

    # Add a time samples dimension for broadcasting
    tau = tau.unsqueeze(6)

    # Time lags for which to compute the channel taps
    l = torch.arange(l_min, l_max + 1, dtype=real_dtype, device=tau.device)

    # Bring tau and l to broadcastable shapes
    tau = tau.unsqueeze(-1)
    l = expand_to_rank(l, tau.dim(), axis=0)

    # sinc pulse shaping (torch.sinc is normalized: sinc(x) = sin(pi*x)/(pi*x))
    g = torch.sinc(l - tau * bandwidth)
    g = torch.complex(g, torch.zeros_like(g))
    a = a.unsqueeze(-1)

    # For every tap, sum the sinc-weighted coefficients
    hm = (a * g).sum(dim=-3)

    if normalize:
        # Normalization is performed such that for each batch example and
        # link the energy per block is one.
        # The total energy of a channel response is the sum of the squared
        # norm over the channel taps.
        # Average over block size, RX antennas, and TX antennas
        c = hm.abs().square().sum(dim=6, keepdim=True).mean(dim=(2, 4, 5), keepdim=True)
        c = torch.complex(c.sqrt(), torch.zeros_like(c))
        hm = torch.where(c.abs() > 0, hm / c, hm)

    return hm


def time_to_ofdm_channel(
    h_t: torch.Tensor,
    rg,
    l_min: int,
) -> torch.Tensor:
    r"""
    Compute the channel frequency response from the discrete complex-baseband
    channel impulse response

    Given a discrete complex-baseband channel impulse response
    :math:`\bar{h}_{b,\ell}`, for :math:`\ell` ranging from :math:`L_\text{min}\le 0`
    to :math:`L_\text{max}`, the discrete channel frequency response is computed as

    .. math::

        \hat{h}_{b,n} = \sum_{k=0}^{L_\text{max}} \bar{h}_{b,k} e^{-j \frac{2\pi kn}{N}} + \sum_{k=L_\text{min}}^{-1} \bar{h}_{b,k} e^{-j \frac{2\pi n(N+k)}{N}}, \quad n=0,\dots,N-1

    where :math:`N` is the FFT size and :math:`b` is the time step.

    This function only produces one channel frequency response per OFDM symbol, i.e.,
    only values of :math:`b` corresponding to the start of an OFDM symbol (after
    cyclic prefix removal) are considered.

    :param h_t: Tensor of discrete complex-baseband channel impulse responses,
        shape [..., num_time_steps, l_max-l_min+1]
    :param rg: Resource grid
    :param l_min: Smallest time-lag for the discrete complex baseband
        channel impulse response (:math:`L_{\text{min}}`)

    :output h_f: [..., num_ofdm_symbols, fft_size], `torch.complex`.
        Tensor of discrete complex-baseband channel frequency responses.

    .. rubric:: Notes

    Note that the result of this function is generally different from the
    output of :meth:`~sionna.phy.channel.utils.cir_to_ofdm_channel` because
    the discrete complex-baseband channel impulse response is truncated
    (see :meth:`~sionna.phy.channel.utils.cir_to_time_channel`). This effect
    can be observed in the example below.

    .. rubric:: Examples

    .. code-block:: python

        import torch
        from sionna.phy.channel import (subcarrier_frequencies,
            cir_to_ofdm_channel, cir_to_time_channel,
            time_lag_discrete_time_channel, time_to_ofdm_channel)
        from sionna.phy.channel.tr38901 import TDL
        from sionna.phy.ofdm import ResourceGrid

        # Setup resource grid and channel model
        rg = ResourceGrid(num_ofdm_symbols=1,
                          fft_size=1024,
                          subcarrier_spacing=15e3)
        tdl = TDL("A", 100e-9, 3.5e9)

        # Generate CIR
        cir = tdl(batch_size=1, num_time_steps=1, sampling_frequency=rg.bandwidth)

        # Generate OFDM channel from CIR
        frequencies = subcarrier_frequencies(rg.fft_size, rg.subcarrier_spacing)
        h_freq = cir_to_ofdm_channel(frequencies, *cir, normalize=True).squeeze()

        # Generate time channel from CIR
        l_min, l_max = time_lag_discrete_time_channel(rg.bandwidth)
        h_time = cir_to_time_channel(rg.bandwidth, *cir, l_min=l_min, l_max=l_max, normalize=True)

        # Generate OFDM channel from time channel
        h_freq_hat = time_to_ofdm_channel(h_time, rg, l_min).squeeze()
    """
    # Total length of an OFDM symbol including cyclic prefix
    ofdm_length = rg.fft_size + rg.cyclic_prefix_length

    # Downsample the impulse response to one sample per OFDM symbol
    h_t = h_t[..., rg.cyclic_prefix_length : rg.num_time_samples : ofdm_length, :]

    # Pad channel impulse response with zeros to the FFT size.
    # A longer impulse response cannot produce an fft_size frequency response
    # without truncation; reject instead of silently returning the wrong shape.
    pad_dims = rg.fft_size - h_t.shape[-1]
    if pad_dims < 0:
        raise ValueError(
            "The last dimension of h_t "
            f"(got {h_t.shape[-1]}) must not exceed rg.fft_size "
            f"({rg.fft_size})."
        )
    if pad_dims > 0:
        pad_shape = list(h_t.shape[:-1]) + [pad_dims]
        h_t = torch.cat(
            [h_t, torch.zeros(pad_shape, dtype=h_t.dtype, device=h_t.device)], dim=-1
        )

    # Circular shift of negative time lags so that the channel impulse response
    # starts with h_{b,0}
    h_t = torch.roll(h_t, shifts=l_min, dims=-1)

    # Compute FFT
    h_f = torch.fft.fft(h_t)

    # Move the zero subcarrier to the center of the spectrum
    h_f = torch.fft.fftshift(h_f, dim=-1)

    return h_f


def deg_2_rad(x: torch.Tensor) -> torch.Tensor:
    r"""
    Convert degree to radian

    :param x: Angles in degree

    :output y: `torch.float`.
        Angles ``x`` converted to radian.

    .. rubric:: Examples

    .. code-block:: python

        import torch
        from sionna.phy.channel import deg_2_rad

        angle_deg = torch.tensor([0.0, 90.0, 180.0, 360.0])
        angle_rad = deg_2_rad(angle_deg)
        print(angle_rad)
        # tensor([0.0000, 1.5708, 3.1416, 6.2832])
    """
    return x * (PI / 180.0)


def rad_2_deg(x: torch.Tensor) -> torch.Tensor:
    r"""
    Convert radian to degree

    :param x: Angles in radian

    :output y: `torch.float`.
        Angles ``x`` converted to degree.

    .. rubric:: Examples

    .. code-block:: python

        import torch
        from sionna.phy.channel import rad_2_deg

        angle_rad = torch.tensor([0.0, 1.5708, 3.1416])
        angle_deg = rad_2_deg(angle_rad)
        print(angle_deg)
        # tensor([  0.,  90., 180.])
    """
    return x * (180.0 / PI)


def wrap_angle_0_360(angle: torch.Tensor) -> torch.Tensor:
    r"""
    Wrap ``angle`` to (0,360)

    :param angle: Input angle in degrees

    :output y: `torch.float`.
        ``angle`` wrapped to (0,360).

    .. rubric:: Examples

    .. code-block:: python

        import torch
        from sionna.phy.channel import wrap_angle_0_360

        angles = torch.tensor([-90.0, 0.0, 450.0, 720.0])
        wrapped = wrap_angle_0_360(angles)
        print(wrapped)
        # tensor([270.,   0.,  90.,   0.])
    """
    return angle % 360.0


def drop_uts_in_sector(
    batch_size: int,
    num_ut: int,
    min_bs_ut_dist: float,
    isd: float,
    bs_height: float = 0.0,
    ut_height: float = 0.0,
    precision: Optional[str] = None,
    device: Optional[str] = None,
) -> torch.Tensor:
    r"""
    Sample UT locations uniformly at random within a sector

    The sector from which UTs are sampled is shown in the following figure.
    The BS is assumed to be located at the origin (0,0) of the coordinate
    system.

    .. figure:: /phy/figures/drop_uts_in_sector.png
        :align: center
        :scale: 30%

    :param batch_size: Batch size
    :param num_ut: Number of UTs to sample per batch example
    :param min_bs_ut_dist: Minimum BS-UT distance [m]
    :param isd: Inter-site distance, i.e., the distance between two adjacent
        base stations [m]
    :param bs_height: BS height, i.e., distance between the BS and the X-Y
        plane [m]. Defaults to 0.0.
    :param ut_height: UT height, i.e., distance between the UT and the X-Y
        plane [m]. Defaults to 0.0.
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`, :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for computation. If `None`,
        :attr:`~sionna.phy.config.Config.device` is used.

    :output ut_loc: [batch_size, num_ut, 2], `torch.float`.
        UT locations in the X-Y plane.
    """
    if precision is None:
        dtype = config.dtype
    else:
        dtype = dtypes[precision]["torch"]["dtype"]

    if device is None:
        device = config.device

    # Cast to dtype
    if not isinstance(min_bs_ut_dist, torch.Tensor):
        d_min = torch.tensor(min_bs_ut_dist, dtype=dtype, device=device)
    else:
        d_min = min_bs_ut_dist.to(dtype=dtype, device=device)

    bs_h = torch.tensor(bs_height, dtype=dtype, device=device)
    ut_h = torch.tensor(ut_height, dtype=dtype, device=device)

    # Force the minimum BS-UT distance >= their height difference
    d_min = torch.maximum(d_min, torch.abs(bs_h - ut_h))

    if not isinstance(isd, torch.Tensor):
        r = torch.tensor(isd * 0.5, dtype=dtype, device=device)
    else:
        r = (isd * 0.5).to(dtype=dtype, device=device)

    # Minimum squared distance between BS and UT on the X-Y plane
    r_min2 = d_min**2 - (bs_h - ut_h) ** 2

    # Angles from (-pi/6, pi/6), covering half of the sector and denoted by
    # alpha_half, are randomly sampled for all UTs.
    # Then, the maximum distance of UTs from the BS, denoted by r_max,
    # is computed for each angle.
    # For each angle, UT positions are sampled such that their *squared* from
    # the BS is uniformly sampled from the range (r_min**2, r_max**2)
    # Each UT is then randomly and uniformly pushed to a half of the sector
    # by adding either PI/6 or PI/2 to the angle alpha_half

    # Sample angles for half of the sector (which half will be decided randomly)
    generator = config.torch_rng(device)
    alpha_half = (
        rand(
            (batch_size, num_ut),
            dtype=dtype,
            device=device,
            generator=generator,
        )
        * (PI / 3.0)
        - PI / 6.0
    )

    # Maximum distance (computed on the X-Y plane) from BS to a point in
    # the sector, at each angle in alpha_half
    r_max = r / torch.cos(alpha_half)

    # To ensure the UT distribution is uniformly distributed across the
    # sector, we sample positions such that their *squared* distance from
    # the BS is uniformly distributed within (r_min**2, r_max**2)
    uniform_samples = rand(
        (batch_size, num_ut),
        dtype=dtype,
        device=device,
        generator=generator,
    )
    distance2 = r_min2 + uniform_samples * (r_max**2 - r_min2)
    distance = torch.sqrt(distance2)

    # Randomly assign the UTs to one of the two halves of the sector
    side = sample_bernoulli(
        [batch_size, num_ut], 0.5, precision=precision, device=device
    )
    side = side.to(dtype)
    side = 2.0 * side + 1.0
    alpha = alpha_half + side * PI / 6.0

    # Compute UT locations in the X-Y plane
    ut_loc = torch.stack(
        [distance * torch.cos(alpha), distance * torch.sin(alpha)], dim=-1
    )

    return ut_loc


def set_3gpp_scenario_parameters(
    scenario: str,
    min_bs_ut_dist: Optional[float] = None,
    isd: Optional[float] = None,
    bs_height: Optional[float] = None,
    min_ut_height: Optional[float] = None,
    max_ut_height: Optional[float] = None,
    indoor_probability: Optional[float] = None,
    min_ut_velocity: Optional[float] = None,
    max_ut_velocity: Optional[float] = None,
    precision: Optional[str] = None,
    device: Optional[str] = None,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    r"""
    Set valid parameters for a specified 3GPP system level ``scenario``
    (RMa, UMi, or UMa)

    If a parameter is given, then it is returned. If it is set to `None`,
    then a parameter valid according to the chosen scenario is returned
    (see :cite:p:`TR38901V1920`).
    The default minimum BS-UT distances for the UMi and UMa scenarios,
    including their calibration variants, are 10 m and 35 m, respectively,
    as specified in Table 7.8-1 of :cite:p:`TR38901V160100`.

    :param scenario: System level model scenario. One of ``"uma"``,
        ``"umi"``, ``"rma"``, ``"uma-calibration"``, or
        ``"umi-calibration"``.
    :param min_bs_ut_dist: Minimum BS-UT distance [m]
    :param isd: Inter-site distance [m]
    :param bs_height: BS elevation [m]
    :param min_ut_height: Minimum UT elevation [m]
    :param max_ut_height: Maximum UT elevation [m]
    :param indoor_probability: Probability of a UT to be indoor. For RMa, the
        remaining UTs are interpreted as in-car by
        :class:`~sionna.phy.channel.tr38901.RMa` unless an explicit ``in_car``
        mask is passed to
        :meth:`~sionna.phy.channel.tr38901.RMa.set_topology`.
    :param min_ut_velocity: Minimum UT velocity [m/s]
    :param max_ut_velocity: Maximum UT velocity [m/s]
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`, :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for computation. If `None`,
        :attr:`~sionna.phy.config.Config.device` is used.

    :output min_bs_ut_dist: `torch.float`.
        Minimum BS-UT distance [m].

    :output isd: `torch.float`.
        Inter-site distance [m].

    :output bs_height: `torch.float`.
        BS elevation [m].

    :output min_ut_height: `torch.float`.
        Minimum UT elevation [m].

    :output max_ut_height: `torch.float`.
        Maximum UT elevation [m].

    :output indoor_probability: `torch.float`.
        Probability of a UT to be indoor.

    :output min_ut_velocity: `torch.float`.
        Minimum UT velocity [m/s].

    :output max_ut_velocity: `torch.float`.
        Maximum UT velocity [m/s].

    .. rubric:: Examples

    The returned tensors can be passed to topology-generation helpers that
    require scenario-dependent defaults.

    .. code-block:: python

        import torch
        from sionna.phy.channel import (
            generate_uts_topology,
            set_3gpp_scenario_parameters,
        )

        params = set_3gpp_scenario_parameters("umi")
        (min_bs_ut_dist, isd, bs_height, min_ut_height, max_ut_height,
         indoor_probability, min_ut_velocity, max_ut_velocity) = params

        ut_loc, ut_orientations, ut_velocities, in_state = generate_uts_topology(
            batch_size=1,
            num_ut=4,
            drop_area="sector",
            cell_loc_xy=torch.zeros(1, 1, 2),
            min_bs_ut_dist=min_bs_ut_dist,
            isd=isd,
            min_ut_height=min_ut_height,
            max_ut_height=max_ut_height,
            indoor_probability=indoor_probability,
            min_ut_velocity=min_ut_velocity,
            max_ut_velocity=max_ut_velocity,
        )
    """
    if scenario not in (
        "umi",
        "uma",
        "rma",
        "umi-calibration",
        "uma-calibration",
    ):
        raise ValueError(
            "`scenario` must be one of 'umi', 'uma', 'rma', "
            "'umi-calibration', 'uma-calibration'"
        )

    if precision is None:
        dtype = config.dtype
    else:
        dtype = dtypes[precision]["torch"]["dtype"]

    if device is None:
        device = config.device

    # Default values for scenario parameters.
    # All distances and heights are in meters
    # All velocities are in meters per second.
    default_scenario_par = {
        "umi": {
            "min_bs_ut_dist": 10.0,
            "isd": 200.0,
            "bs_height": 10.0,
            "min_ut_height": 1.5,
            "max_ut_height": 1.5,
            "indoor_probability": 0.8,
            "min_ut_velocity": 0.0,
            "max_ut_velocity": 0.0,
        },
        "umi-calibration": {
            "min_bs_ut_dist": 10.0,
            "isd": 200.0,
            "bs_height": 10.0,
            "min_ut_height": 1.5,
            "max_ut_height": 1.5,
            "indoor_probability": 0.8,
            "min_ut_velocity": 3.0 / 3.6,
            "max_ut_velocity": 3.0 / 3.6,
        },
        "uma": {
            "min_bs_ut_dist": 35.0,
            "isd": 500.0,
            "bs_height": 25.0,
            "min_ut_height": 1.5,
            "max_ut_height": 1.5,
            "indoor_probability": 0.8,
            "min_ut_velocity": 0.0,
            "max_ut_velocity": 0.0,
        },
        "uma-calibration": {
            "min_bs_ut_dist": 35.0,
            "isd": 500.0,
            "bs_height": 25.0,
            "min_ut_height": 1.5,
            "max_ut_height": 1.5,
            "indoor_probability": 0.8,
            "min_ut_velocity": 3.0 / 3.6,
            "max_ut_velocity": 3.0 / 3.6,
        },
        "rma": {
            "min_bs_ut_dist": 35.0,
            "isd": 5000.0,
            "bs_height": 35.0,
            "min_ut_height": 1.5,
            "max_ut_height": 1.5,
            "indoor_probability": 0.5,
            "min_ut_velocity": 0.0,
            "max_ut_velocity": 0.0,
        },
    }

    def _get_param(param_value, param_name):
        if param_value is None:
            return torch.tensor(
                default_scenario_par[scenario][param_name], dtype=dtype, device=device
            )
        return torch.tensor(param_value, dtype=dtype, device=device)

    min_bs_ut_dist = _get_param(min_bs_ut_dist, "min_bs_ut_dist")
    isd = _get_param(isd, "isd")
    bs_height = _get_param(bs_height, "bs_height")
    min_ut_height = _get_param(min_ut_height, "min_ut_height")
    max_ut_height = _get_param(max_ut_height, "max_ut_height")
    indoor_probability = _get_param(indoor_probability, "indoor_probability")
    min_ut_velocity = _get_param(min_ut_velocity, "min_ut_velocity")
    max_ut_velocity = _get_param(max_ut_velocity, "max_ut_velocity")

    return (
        min_bs_ut_dist,
        isd,
        bs_height,
        min_ut_height,
        max_ut_height,
        indoor_probability,
        min_ut_velocity,
        max_ut_velocity,
    )


def relocate_uts(
    ut_loc: torch.Tensor,
    sector_id: torch.Tensor,
    cell_loc: torch.Tensor,
) -> torch.Tensor:
    r"""
    Relocate the UTs by rotating them into the sector with index ``sector_id``
    and transposing them to the cell centered on ``cell_loc``

    ``sector_id`` gives the index of the sector to which the UTs are
    rotated to. The picture below shows how the three sectors of a cell are
    indexed.

    .. figure:: /phy/figures/panel_array_sector_id.png
        :align: center
        :scale: 30%

        Indexing of sectors

    If ``sector_id`` is a scalar, then all UTs are relocated to the same
    sector indexed by ``sector_id``.
    If ``sector_id`` is a tensor, it should be broadcastable with
    [``batch_size``, ``num_ut``], and give the sector in which each UT or
    batch example is relocated to.

    When calling the function, ``ut_loc`` gives the locations of the UTs to
    relocate, which are all assumed to be in sector with index 0, and in the
    cell centered on the origin (0,0).

    :param ut_loc: UTs locations in the X-Y plane,
        shape [batch_size, num_ut, 2]
    :param sector_id: Indexes of the sector to which to relocate the UTs,
        broadcastable with [batch_size, num_ut]
    :param cell_loc: Center of the cell to which to transpose the UTs,
        broadcastable with [batch_size, num_ut, 2]

    :output ut_loc: [batch_size, num_ut, 2], `torch.float`.
        Relocated UTs locations in the X-Y plane.
    """
    # Expand the rank of sector_id such that it is broadcastable with
    # (batch size, num_ut)
    sector_id = sector_id.to(ut_loc.dtype)
    sector_id = expand_to_rank(sector_id, 2, 0)

    # Expand cell_loc
    cell_loc = cell_loc.to(ut_loc.dtype)
    cell_loc = expand_to_rank(cell_loc, ut_loc.dim(), 0)

    # Rotation matrix tensor, broadcastable with [batch size, num uts, 2, 2]
    rotation_angle = sector_id * 2.0 * PI / 3.0
    cos_angle = torch.cos(rotation_angle)
    sin_angle = torch.sin(rotation_angle)
    rotation_matrix = torch.stack([cos_angle, -sin_angle, sin_angle, cos_angle], dim=-1)
    new_shape = list(rotation_angle.shape) + [2, 2]
    rotation_matrix = rotation_matrix.reshape(new_shape)
    rotation_matrix = rotation_matrix.to(ut_loc.dtype)

    # Applying the rotation matrix
    ut_loc = ut_loc.unsqueeze(-1)
    ut_loc_rotated = (rotation_matrix @ ut_loc).squeeze(-1)

    # Translate to the BS location
    ut_loc_rotated_translated = ut_loc_rotated + cell_loc

    return ut_loc_rotated_translated


def generate_uts_topology(
    batch_size: int,
    num_ut: int,
    drop_area: str,
    cell_loc_xy: torch.Tensor,
    min_bs_ut_dist: torch.Tensor,
    isd: torch.Tensor,
    min_ut_height: torch.Tensor,
    max_ut_height: torch.Tensor,
    indoor_probability: torch.Tensor,
    min_ut_velocity: torch.Tensor,
    max_ut_velocity: torch.Tensor,
    precision: Optional[str] = None,
    device: Optional[str] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""
    Sample UTs location from a sector or a cell

    :param batch_size: Batch size
    :param num_ut: Number of UTs to sample per batch example
    :param drop_area: If set to ``"sector"``, UTs are sampled from the
        sector with index 0. If ``"cell"``, UTs are sampled from the
        entire cell.
    :param cell_loc_xy: Center of the cell(s), broadcastable with
        [batch_size, num_ut, 2]
    :param min_bs_ut_dist: Minimum BS-UT distance [m]
    :param isd: Inter-site distance [m]
    :param min_ut_height: Minimum UT elevation [m]
    :param max_ut_height: Maximum UT elevation [m]
    :param indoor_probability: Probability of a UT to be indoor
    :param min_ut_velocity: Minimum UT velocity [m/s]
    :param max_ut_velocity: Maximum UT velocity [m/s]
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`, :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for computation. If `None`,
        :attr:`~sionna.phy.config.Config.device` is used.

    :output ut_loc: [batch_size, num_ut, 3], `torch.float`.
        UTs locations.

    :output ut_orientations: [batch_size, num_ut, 3], `torch.float`.
        UTs orientations [radian].

    :output ut_velocities: [batch_size, num_ut, 3], `torch.float`.
        UTs velocities [m/s]. The norm is drawn uniformly at random
        between ``min_ut_velocity`` and ``max_ut_velocity``, while the
        direction is planar and drawn uniformly at random between 0 and
        :math:`2\pi`.

    :output in_state: [batch_size, num_ut], `torch.bool`.
        Indoor/outdoor state of UTs. `True` means indoor, `False` means
        outdoor.
    """
    if drop_area not in ("sector", "cell"):
        raise ValueError("Drop area must be either 'sector' or 'cell'")

    if precision is None:
        dtype = config.dtype
    else:
        dtype = dtypes[precision]["torch"]["dtype"]

    if device is None:
        device = config.device
    generator = None if torch.compiler.is_compiling() else config.torch_rng(device)

    # Randomly generating the UT locations
    if not isinstance(min_bs_ut_dist, torch.Tensor):
        min_bs_ut_dist_val = torch.tensor(min_bs_ut_dist, dtype=dtype, device=device)
    else:
        min_bs_ut_dist_val = min_bs_ut_dist

    if not isinstance(isd, torch.Tensor):
        isd_val = torch.tensor(isd, dtype=dtype, device=device)
    else:
        isd_val = isd

    ut_loc_xy = drop_uts_in_sector(
        batch_size,
        num_ut,
        min_bs_ut_dist_val,
        isd_val,
        precision=precision,
        device=device,
    )
    if drop_area == "sector":
        sectors = torch.tensor(0, dtype=torch.int32, device=device)
    else:  # 'cell'
        sectors = randint(
            0,
            3,
            (batch_size, num_ut),
            dtype=torch.int32,
            device=device,
            generator=generator,
        )
    ut_loc_xy = relocate_uts(ut_loc_xy, sectors, cell_loc_xy)

    ut_loc_z = (
        rand(
            (batch_size, num_ut, 1),
            dtype=dtype,
            device=device,
            generator=generator,
        )
        * (max_ut_height - min_ut_height)
        + min_ut_height
    )
    ut_loc = torch.cat([ut_loc_xy, ut_loc_z], dim=-1)

    # Draw random UT orientation, velocity and indoor state
    ut_orientations, ut_velocities, in_state = random_ut_properties(
        batch_size,
        num_ut,
        indoor_probability,
        min_ut_velocity,
        max_ut_velocity,
        precision=precision,
        device=device,
    )

    return ut_loc, ut_orientations, ut_velocities, in_state


def random_ut_properties(
    batch_size: int,
    num_ut: int,
    indoor_probability: torch.Tensor,
    min_ut_velocity: torch.Tensor,
    max_ut_velocity: torch.Tensor,
    precision: Optional[str] = None,
    device: Optional[str] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""
    Draw random values for UT velocity, orientation and indoor state

    :param batch_size: Batch size
    :param num_ut: Number of UTs to sample per batch example
    :param indoor_probability: Probability of a UT to be indoor
    :param min_ut_velocity: Minimum UT velocity [m/s]
    :param max_ut_velocity: Maximum UT velocity [m/s]
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`, :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for computation. If `None`,
        :attr:`~sionna.phy.config.Config.device` is used.

    :output ut_orientations: [batch_size, num_ut, 3], `torch.float`.
        UTs orientations [radian].

    :output ut_velocities: [batch_size, num_ut, 3], `torch.float`.
        UTs velocity vectors [m/s]. The norm is drawn uniformly at random
        between ``min_ut_velocity`` and ``max_ut_velocity``, while the
        direction is planar and drawn uniformly at random between 0 and
        :math:`2\pi`.

    :output in_state: [batch_size, num_ut], `torch.bool`.
        Indoor/outdoor state of UTs. `True` means indoor, `False` means
        outdoor.
    """
    if precision is None:
        dtype = config.dtype
    else:
        dtype = dtypes[precision]["torch"]["dtype"]

    if device is None:
        device = config.device
    generator = None if torch.compiler.is_compiling() else config.torch_rng(device)

    # Randomly generating the UT indoor/outdoor state
    if not isinstance(indoor_probability, torch.Tensor):
        indoor_probability_val = torch.tensor(
            indoor_probability, dtype=dtype, device=device
        )
    else:
        indoor_probability_val = indoor_probability

    in_state = sample_bernoulli(
        [batch_size, num_ut],
        indoor_probability_val,
        precision=precision,
        device=device,
    )

    # Randomly generate the UT velocities
    ut_vel_angle = (
        rand(
            (batch_size, num_ut),
            dtype=dtype,
            device=device,
            generator=generator,
        )
        * 2
        * PI
        - PI
    )
    ut_vel_norm = (
        rand(
            (batch_size, num_ut),
            dtype=dtype,
            device=device,
            generator=generator,
        )
        * (max_ut_velocity - min_ut_velocity)
        + min_ut_velocity
    )
    ut_velocities = torch.stack(
        [
            ut_vel_norm * torch.cos(ut_vel_angle),
            ut_vel_norm * torch.sin(ut_vel_angle),
            torch.zeros(batch_size, num_ut, dtype=dtype, device=device),
        ],
        dim=-1,
    )

    # Randomly generate the UT orientations
    ut_bearing = (
        rand(
            (batch_size, num_ut),
            dtype=dtype,
            device=device,
            generator=generator,
        )
        * PI
        - 0.5 * PI
    )
    ut_downtilt = (
        rand(
            (batch_size, num_ut),
            dtype=dtype,
            device=device,
            generator=generator,
        )
        * PI
        - 0.5 * PI
    )
    ut_slant = (
        rand(
            (batch_size, num_ut),
            dtype=dtype,
            device=device,
            generator=generator,
        )
        * PI
        - 0.5 * PI
    )
    ut_orientations = torch.stack([ut_bearing, ut_downtilt, ut_slant], dim=-1)

    return ut_orientations, ut_velocities, in_state


def gen_single_sector_topology(
    batch_size: int,
    num_ut: int,
    scenario: str,
    min_bs_ut_dist: Optional[float] = None,
    isd: Optional[float] = None,
    bs_height: Optional[float] = None,
    min_ut_height: Optional[float] = None,
    max_ut_height: Optional[float] = None,
    indoor_probability: Optional[float] = None,
    min_ut_velocity: Optional[float] = None,
    max_ut_velocity: Optional[float] = None,
    precision: Optional[str] = None,
    device: Optional[str] = None,
) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    r"""
    Generate a batch of topologies consisting of a single BS located at the
    origin and ``num_ut`` UTs randomly and uniformly dropped in a cell sector

    The following picture shows the sector from which UTs are sampled.

    .. figure:: /phy/figures/drop_uts_in_sector.png
        :align: center
        :scale: 30%

    UT velocity and orientation are drawn uniformly at random, whereas the BS points
    towards the center of the sector it serves.

    The drop configuration can be controlled through the optional parameters.
    Parameters set to `None` are set to valid values according to the chosen
    ``scenario`` (see :cite:p:`TR38901V1920`).

    The returned batch of topologies can be used as-is with the
    :meth:`set_topology` method of the system level models, i.e.
    :class:`~sionna.phy.channel.tr38901.UMi`,
    :class:`~sionna.phy.channel.tr38901.UMa`,
    and :class:`~sionna.phy.channel.tr38901.RMa`.

    :param batch_size: Batch size
    :param num_ut: Number of UTs to sample per batch example
    :param scenario: System level model scenario. One of ``"uma"``,
        ``"umi"``, ``"rma"``, ``"uma-calibration"``, or
        ``"umi-calibration"``.
    :param min_bs_ut_dist: Minimum BS-UT distance [m]
    :param isd: Inter-site distance [m]
    :param bs_height: BS elevation [m]
    :param min_ut_height: Minimum UT elevation [m]
    :param max_ut_height: Maximum UT elevation [m]
    :param indoor_probability: Probability of a UT to be indoor. For RMa, the
        remaining UTs are interpreted as in-car by
        :class:`~sionna.phy.channel.tr38901.RMa` unless an explicit ``in_car``
        mask is passed to
        :meth:`~sionna.phy.channel.tr38901.RMa.set_topology`.
    :param min_ut_velocity: Minimum UT velocity [m/s]
    :param max_ut_velocity: Maximum UT velocity [m/s]
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`, :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for computation. If `None`,
        :attr:`~sionna.phy.config.Config.device` is used.

    :output ut_loc: [batch_size, num_ut, 3], `torch.float`.
        UTs locations.

    :output bs_loc: [batch_size, 1, 3], `torch.float`.
        BS location. Set to (0,0,0) for all batch examples.

    :output ut_orientations: [batch_size, num_ut, 3], `torch.float`.
        UTs orientations [radian].

    :output bs_orientations: [batch_size, 1, 3], `torch.float`.
        BS orientations [radian]. Oriented towards the center of the sector.

    :output ut_velocities: [batch_size, num_ut, 3], `torch.float`.
        UTs velocities [m/s].

    :output in_state: [batch_size, num_ut], `torch.bool`.
        Indoor/outdoor state of UTs. `True` means indoor, `False` means
        outdoor. For RMa, the initial
        :meth:`~sionna.phy.channel.tr38901.RMa.set_topology` call interprets
        every `False` entry as in-car per Table 7.2-3 unless ``in_car`` is
        supplied explicitly.

    .. rubric:: Examples

    .. code-block:: python

        from sionna.phy.channel.tr38901 import PanelArray, UMi
        from sionna.phy.channel import gen_single_sector_topology

        # Create antenna arrays
        bs_array = PanelArray(num_rows_per_panel=4,
                              num_cols_per_panel=4,
                              polarization='dual',
                              polarization_type='VH',
                              antenna_pattern='38.901',
                              carrier_frequency=3.5e9)

        ut_array = PanelArray(num_rows_per_panel=1,
                              num_cols_per_panel=1,
                              polarization='single',
                              polarization_type='V',
                              antenna_pattern='omni',
                              carrier_frequency=3.5e9)

        # Create channel model
        channel_model = UMi(carrier_frequency=3.5e9,
                            o2i_model='low',
                            ut_array=ut_array,
                            bs_array=bs_array,
                            direction='uplink')

        # Generate the topology
        topology = gen_single_sector_topology(batch_size=100,
                                              num_ut=4,
                                              scenario='umi')

        # Set the topology
        ut_loc, bs_loc, ut_orientations, bs_orientations, ut_velocities, in_state = topology
        channel_model.set_topology(ut_loc,
                                   bs_loc,
                                   ut_orientations,
                                   bs_orientations,
                                   ut_velocities,
                                   in_state)
        channel_model.show_topology()

    .. image:: /phy/figures/drop_uts_in_sector_topology.png
    """
    params = set_3gpp_scenario_parameters(
        scenario,
        min_bs_ut_dist,
        isd,
        bs_height,
        min_ut_height,
        max_ut_height,
        indoor_probability,
        min_ut_velocity,
        max_ut_velocity,
        precision=precision,
        device=device,
    )
    (
        min_bs_ut_dist_t,
        isd_t,
        bs_height_t,
        min_ut_height_t,
        max_ut_height_t,
        indoor_probability_t,
        min_ut_velocity_t,
        max_ut_velocity_t,
    ) = params

    if precision is None:
        dtype = config.dtype
    else:
        dtype = dtypes[precision]["torch"]["dtype"]

    if device is None:
        device = config.device

    # Setting BS to (0,0,bs_height)
    bs_loc = torch.stack(
        [
            torch.zeros(batch_size, 1, dtype=dtype, device=device),
            torch.zeros(batch_size, 1, dtype=dtype, device=device),
            bs_height_t.expand(batch_size, 1),
        ],
        dim=-1,
    )

    # Setting the BS orientation such that it is downtilted towards the center
    # of the sector
    sector_center = (min_bs_ut_dist_t + 0.5 * isd_t) * 0.5
    bs_downtilt = 0.5 * PI - torch.atan(sector_center / bs_height_t)
    bs_yaw = torch.tensor(PI / 3.0, dtype=dtype, device=device)
    bs_orientation = torch.stack(
        [
            bs_yaw.expand(batch_size, 1),
            bs_downtilt.expand(batch_size, 1),
            torch.zeros(batch_size, 1, dtype=dtype, device=device),
        ],
        dim=-1,
    )

    # Generating the UTs
    ut_topology = generate_uts_topology(
        batch_size,
        num_ut,
        "sector",
        torch.zeros(2, dtype=dtype, device=device),
        min_bs_ut_dist_t,
        isd_t,
        min_ut_height_t,
        max_ut_height_t,
        indoor_probability_t,
        min_ut_velocity_t,
        max_ut_velocity_t,
        precision,
        device,
    )
    ut_loc, ut_orientations, ut_velocities, in_state = ut_topology

    return ut_loc, bs_loc, ut_orientations, bs_orientation, ut_velocities, in_state


def gen_single_sector_topology_interferers(
    batch_size: int,
    num_ut: int,
    num_interferer: int,
    scenario: str,
    min_bs_ut_dist: Optional[float] = None,
    isd: Optional[float] = None,
    bs_height: Optional[float] = None,
    min_ut_height: Optional[float] = None,
    max_ut_height: Optional[float] = None,
    indoor_probability: Optional[float] = None,
    min_ut_velocity: Optional[float] = None,
    max_ut_velocity: Optional[float] = None,
    precision: Optional[str] = None,
    device: Optional[str] = None,
) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    r"""
    Generate a batch of topologies consisting of a single BS located at the
    origin, ``num_ut`` UTs randomly and uniformly dropped in a cell sector, and
    ``num_interferer`` interfering UTs randomly dropped in the adjacent cells

    The following picture shows how UTs are sampled.

    .. figure:: /phy/figures/drop_uts_in_sector_interferers.png
        :align: center
        :scale: 30%

    UT velocity and orientation are drawn uniformly at random, whereas the BS
    points towards the center of the sector it serves.

    The drop configuration can be controlled through the optional parameters.
    Parameters set to `None` are set to valid values according to the chosen
    ``scenario`` (see :cite:p:`TR38901V1920`).

    The returned batch of topologies can be used as-is with the
    :meth:`set_topology` method of the system level models, i.e.
    :class:`~sionna.phy.channel.tr38901.UMi`,
    :class:`~sionna.phy.channel.tr38901.UMa`,
    and :class:`~sionna.phy.channel.tr38901.RMa`.

    In the returned ``ut_loc``, ``ut_orientations``, ``ut_velocities``, and
    ``in_state`` tensors, the first ``num_ut`` items along the axis with index
    1 correspond to the served UTs, whereas the remaining ``num_interferer``
    items correspond to the interfering UTs.

    :param batch_size: Batch size
    :param num_ut: Number of UTs to sample per batch example
    :param num_interferer: Number of interfering UTs per batch example
    :param scenario: System level model scenario. One of ``"uma"``,
        ``"umi"``, ``"rma"``, ``"uma-calibration"``, or
        ``"umi-calibration"``.
    :param min_bs_ut_dist: Minimum BS-UT distance [m]
    :param isd: Inter-site distance [m]
    :param bs_height: BS elevation [m]
    :param min_ut_height: Minimum UT elevation [m]
    :param max_ut_height: Maximum UT elevation [m]
    :param indoor_probability: Probability of a UT to be indoor. For RMa, the
        remaining UTs are interpreted as in-car by
        :class:`~sionna.phy.channel.tr38901.RMa` unless an explicit ``in_car``
        mask is passed to
        :meth:`~sionna.phy.channel.tr38901.RMa.set_topology`.
    :param min_ut_velocity: Minimum UT velocity [m/s]
    :param max_ut_velocity: Maximum UT velocity [m/s]
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`, :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for computation. If `None`,
        :attr:`~sionna.phy.config.Config.device` is used.

    :output ut_loc: [batch_size, num_ut + num_interferer, 3], `torch.float`.
        UTs locations. The first ``num_ut`` items along the axis with index
        1 correspond to the served UTs, whereas the remaining
        ``num_interferer`` items correspond to the interfering UTs.

    :output bs_loc: [batch_size, 1, 3], `torch.float`.
        BS location. Set to (0,0,0) for all batch examples.

    :output ut_orientations: [batch_size, num_ut + num_interferer, 3], `torch.float`.
        UTs orientations [radian]. The first ``num_ut`` items along the
        axis with index 1 correspond to the served UTs, whereas the
        remaining ``num_interferer`` items correspond to the interfering
        UTs.

    :output bs_orientations: [batch_size, 1, 3], `torch.float`.
        BS orientation [radian]. Oriented towards the center of the sector.

    :output ut_velocities: [batch_size, num_ut + num_interferer, 3], `torch.float`.
        UTs velocities [m/s]. The first ``num_ut`` items along the axis
        with index 1 correspond to the served UTs, whereas the remaining
        ``num_interferer`` items correspond to the interfering UTs.

    :output in_state: [batch_size, num_ut + num_interferer], `torch.bool`.
        Indoor/outdoor state of UTs. `True` means indoor, `False` means
        outdoor. For RMa, the initial
        :meth:`~sionna.phy.channel.tr38901.RMa.set_topology` call interprets
        every `False` entry as in-car per Table 7.2-3 unless ``in_car`` is
        supplied explicitly. The first ``num_ut`` items along the axis with
        index 1 correspond to the served UTs, whereas the remaining
        ``num_interferer`` items correspond to the interfering UTs.

    .. rubric:: Examples

    .. code-block:: python

        from sionna.phy.channel.tr38901 import PanelArray, UMi
        from sionna.phy.channel import gen_single_sector_topology_interferers

        # Create antenna arrays
        bs_array = PanelArray(num_rows_per_panel=4,
                              num_cols_per_panel=4,
                              polarization='dual',
                              polarization_type='VH',
                              antenna_pattern='38.901',
                              carrier_frequency=3.5e9)

        ut_array = PanelArray(num_rows_per_panel=1,
                              num_cols_per_panel=1,
                              polarization='single',
                              polarization_type='V',
                              antenna_pattern='omni',
                              carrier_frequency=3.5e9)

        # Create channel model
        channel_model = UMi(carrier_frequency=3.5e9,
                            o2i_model='low',
                            ut_array=ut_array,
                            bs_array=bs_array,
                            direction='uplink')

        # Generate the topology
        topology = gen_single_sector_topology_interferers(batch_size=100,
                                                          num_ut=4,
                                                          num_interferer=4,
                                                          scenario='umi')

        # Set the topology
        ut_loc, bs_loc, ut_orientations, bs_orientations, ut_velocities, in_state = topology
        channel_model.set_topology(ut_loc,
                                   bs_loc,
                                   ut_orientations,
                                   bs_orientations,
                                   ut_velocities,
                                   in_state)
        channel_model.show_topology()

    .. image:: /phy/figures/drop_uts_in_sector_topology_inter.png
    """
    params = set_3gpp_scenario_parameters(
        scenario,
        min_bs_ut_dist,
        isd,
        bs_height,
        min_ut_height,
        max_ut_height,
        indoor_probability,
        min_ut_velocity,
        max_ut_velocity,
        precision=precision,
        device=device,
    )
    (
        min_bs_ut_dist_t,
        isd_t,
        bs_height_t,
        min_ut_height_t,
        max_ut_height_t,
        indoor_probability_t,
        min_ut_velocity_t,
        max_ut_velocity_t,
    ) = params

    if precision is None:
        dtype = config.dtype
    else:
        dtype = dtypes[precision]["torch"]["dtype"]

    if device is None:
        device = config.device

    # Setting BS to (0,0,bs_height)
    bs_loc = torch.stack(
        [
            torch.zeros(batch_size, 1, dtype=dtype, device=device),
            torch.zeros(batch_size, 1, dtype=dtype, device=device),
            bs_height_t.expand(batch_size, 1),
        ],
        dim=-1,
    )

    # Setting the BS orientation such that it is downtilted towards the center
    # of the sector
    sector_center = (min_bs_ut_dist_t + 0.5 * isd_t) * 0.5
    bs_downtilt = 0.5 * PI - torch.atan(sector_center / bs_height_t)
    bs_yaw = torch.tensor(PI / 3.0, dtype=dtype, device=device)
    bs_orientation = torch.stack(
        [
            bs_yaw.expand(batch_size, 1),
            bs_downtilt.expand(batch_size, 1),
            torch.zeros(batch_size, 1, dtype=dtype, device=device),
        ],
        dim=-1,
    )

    # Generating the UTs located in the sector served by the BS
    ut_topology = generate_uts_topology(
        batch_size,
        num_ut,
        "sector",
        torch.zeros(2, dtype=dtype, device=device),
        min_bs_ut_dist_t,
        isd_t,
        min_ut_height_t,
        max_ut_height_t,
        indoor_probability_t,
        min_ut_velocity_t,
        max_ut_velocity_t,
        precision,
        device,
    )
    ut_loc, ut_orientations, ut_velocities, in_state = ut_topology

    # Generating the UTs located in the adjacent cells
    # Users are randomly dropped in one of the two adjacent cells
    if not isinstance(isd_t, torch.Tensor):
        isd_val = torch.tensor(isd_t, dtype=dtype, device=device)
    else:
        isd_val = isd_t

    # We need scalar value for rotation matrix construction in torch.tensor()
    # If isd_val is a tensor, we need to extract item if we are in eager mode
    # to construct inter_cell_center as a tensor from python list.
    # However, to avoid graph break, we should construct inter_cell_center purely with torch ops.

    isd_val_scalar = isd_val

    # Constructing tensor directly with torch ops to avoid list->tensor conversion with item()
    # [0.0, isd_val]
    row1 = torch.stack([torch.tensor(0.0, dtype=dtype, device=device), isd_val_scalar])

    # [isd_val * cos(PI/6), isd_val * sin(PI/6)]
    cos_pi_6 = torch.cos(torch.tensor(PI / 6.0, dtype=dtype, device=device))
    sin_pi_6 = torch.sin(torch.tensor(PI / 6.0, dtype=dtype, device=device))
    row2 = torch.stack([isd_val_scalar * cos_pi_6, isd_val_scalar * sin_pi_6])

    inter_cell_center = torch.stack([row1, row2])

    cell_index = randint(
        0,
        2,
        (batch_size, num_interferer),
        dtype=torch.int64,
        device=device,
        generator=config.torch_rng(device),
    )
    inter_cells = inter_cell_center[cell_index]

    inter_topology = generate_uts_topology(
        batch_size,
        num_interferer,
        "cell",
        inter_cells,
        min_bs_ut_dist_t,
        isd_t,
        min_ut_height_t,
        max_ut_height_t,
        indoor_probability_t,
        min_ut_velocity_t,
        max_ut_velocity_t,
        precision,
        device,
    )
    inter_loc, inter_orientations, inter_velocities, inter_in_state = inter_topology

    ut_loc = torch.cat([ut_loc, inter_loc], dim=1)
    ut_orientations = torch.cat([ut_orientations, inter_orientations], dim=1)
    ut_velocities = torch.cat([ut_velocities, inter_velocities], dim=1)
    in_state = torch.cat([in_state, inter_in_state], dim=1)

    return ut_loc, bs_loc, ut_orientations, bs_orientation, ut_velocities, in_state


def _toeplitz(col: torch.Tensor, row: torch.Tensor) -> torch.Tensor:
    """
    Construct a Toeplitz matrix from the first column and first row.

    :param col: First column of the Toeplitz matrix, shape [..., n]
    :param row: First row of the Toeplitz matrix, shape [..., m]
        (row[..., 0] should equal col[..., 0])

    :output result: Toeplitz matrix, shape [..., n, m]
    """
    n = col.shape[-1]
    m = row.shape[-1]

    # Create the values array: [row[m-1], ..., row[1], col[0], col[1], ..., col[n-1]]
    # We reverse row[1:] and concatenate with col
    row_reversed = row[..., 1:].flip(dims=[-1])
    vals = torch.cat([row_reversed, col], dim=-1)

    # Create index matrix for Toeplitz structure
    # T[i,j] = vals[m - 1 - j + i]
    i_idx = torch.arange(n, device=col.device).unsqueeze(1)  # [n, 1]
    j_idx = torch.arange(m, device=col.device).unsqueeze(0)  # [1, m]
    indices = (m - 1) - j_idx + i_idx  # [n, m]

    # Expand indices for batch dimensions
    batch_shape = col.shape[:-1]
    for _ in range(len(batch_shape)):
        indices = indices.unsqueeze(0)
    indices = indices.expand(*batch_shape, n, m)

    # Gather values using the indices
    result = torch.gather(vals.unsqueeze(-2).expand(*batch_shape, n, -1), -1, indices)

    return result


def exp_corr_mat(
    a: torch.Tensor,
    n: int,
    precision: Optional[str] = None,
    device: Optional[str] = None,
) -> torch.Tensor:
    r"""Generates exponential correlation matrices

    This function computes for every element :math:`a` of a complex-valued
    tensor :math:`\mathbf{a}` the corresponding :math:`n\times n` exponential
    correlation matrix :math:`\mathbf{R}(a,n)`, defined as (Eq. 1, :cite:p:`MAL2018`):

    .. math::
        \mathbf{R}(a,n)_{i,j} = \begin{cases}
                    1 & \text{if } i=j\\
                    a^{i-j}  & \text{if } i>j\\
                    (a^\star)^{j-i}  & \text{if } j<i, j=1,\dots,n\\
                  \end{cases}

    where :math:`|a|<1` and :math:`\mathbf{R}\in\mathbb{C}^{n\times n}`.

    :param a: Parameters :math:`a` for the exponential correlation matrices,
        shape [n_0, ..., n_k]
    :param n: Number of dimensions of the output correlation matrices
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`, :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for computation. If `None`,
        :attr:`~sionna.phy.config.Config.device` is used.

    :output R: [n_0, ..., n_k, n, n], `torch.complex`.
        Correlation matrices.

    .. rubric:: Examples

    .. code-block:: python

        import torch
        from sionna.phy.channel import exp_corr_mat

        # Single correlation matrix
        R = exp_corr_mat(torch.tensor(0.9+0.1j), 4)
        print(R.shape)
        # torch.Size([4, 4])

        # Batch of correlation matrices
        R = exp_corr_mat(torch.rand(2, 3) * 0.9, 4)
        print(R.shape)
        # torch.Size([2, 3, 4, 4])
    """
    if precision is None:
        cdtype = config.cdtype
    else:
        cdtype = dtypes[precision]["torch"]["cdtype"]

    if device is None:
        device = config.device

    # Cast to desired output dtype and expand last dimension for broadcasting
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a, dtype=cdtype, device=device)
    else:
        a = a.to(dtype=cdtype, device=device)
    a = a.unsqueeze(-1)

    check_tensor_all(
        torch.abs(a) < 1,
        name="a",
        message=(
            "The absolute value of the elements of `a` must be smaller than one"
        ),
    )

    # Vector of exponents, adapt dtype and dimensions for broadcasting
    exp = torch.arange(0, n, device=device)
    exp = exp.to(dtype=cdtype)
    exp = expand_to_rank(exp, a.dim(), 0)

    # First column of R
    col = torch.pow(a, exp)

    # For a=0, one needs to remove the resulting nans due to 0**0=nan
    cond = torch.isnan(col.real)
    col = torch.where(cond, torch.ones_like(col), col)

    # First row of R (equal to complex-conjugate of first column)
    row = torch.conj(col)

    # Build Toeplitz matrix
    r = _toeplitz(col, row)

    return r


def one_ring_corr_mat(
    phi_deg: torch.Tensor,
    num_ant: int,
    d_h: float = 0.5,
    sigma_phi_deg: float = 15.0,
    precision: Optional[str] = None,
    device: Optional[str] = None,
) -> torch.Tensor:
    r"""Generates covariance matrices from the one-ring model

    This function generates approximate covariance matrices for the
    so-called `one-ring` model (Eq. 2.24) :cite:p:`BHS2017`. A uniform
    linear array (ULA) with uniform antenna spacing is assumed. The elements
    of the covariance matrices are computed as:

    .. math::
        \mathbf{R}_{\ell,m} =
              \exp\left( j2\pi d_\text{H} (\ell -m)\sin(\varphi) \right)
              \exp\left( -\frac{\sigma_\varphi^2}{2}
              \left( 2\pi d_\text{H}(\ell -m)\cos(\varphi) \right)^2 \right)

    for :math:`\ell,m = 1,\dots, M`, where :math:`M` is the number of antennas,
    :math:`\varphi` is the angle of arrival, :math:`d_\text{H}` is the antenna
    spacing in multiples of the wavelength,
    and :math:`\sigma^2_\varphi` is the angular standard deviation.

    :param phi_deg: Azimuth angles (deg) of arrival, shape [n_0, ..., n_k]
    :param num_ant: Number of antennas
    :param d_h: Antenna spacing in multiples of the wavelength.
        Defaults to 0.5.
    :param sigma_phi_deg: Angular standard deviation (deg). Values greater
        than 15 should not be used as the approximation becomes invalid.
        Defaults to 15.0.
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`, :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for computation. If `None`,
        :attr:`~sionna.phy.config.Config.device` is used.

    :output R: [n_0, ..., n_k, num_ant, num_ant], `torch.complex`.
        Covariance matrices.

    .. rubric:: Examples

    .. code-block:: python

        import torch
        from sionna.phy.channel import one_ring_corr_mat

        # Single covariance matrix
        R = one_ring_corr_mat(torch.tensor(45.0), 4)
        print(R.shape)
        # torch.Size([4, 4])

        # Batch of covariance matrices
        R = one_ring_corr_mat(torch.rand(2, 3) * 180 - 90, 4)
        print(R.shape)
        # torch.Size([2, 3, 4, 4])
    """
    if precision is None:
        dtype = config.dtype
        cdtype = config.cdtype
    else:
        dtype = dtypes[precision]["torch"]["dtype"]
        cdtype = dtypes[precision]["torch"]["cdtype"]

    if device is None:
        device = config.device

    if sigma_phi_deg > 15:
        warnings.warn(
            "sigma_phi_deg should be smaller than 15.",
            UserWarning,
            stacklevel=2,
        )

    # Convert all inputs to radians
    if not isinstance(phi_deg, torch.Tensor):
        phi_deg = torch.tensor(phi_deg, dtype=dtype, device=device)
    else:
        phi_deg = phi_deg.to(dtype=dtype, device=device)

    sigma_phi_deg_t = torch.tensor(sigma_phi_deg, dtype=dtype, device=device)
    phi = deg_2_rad(phi_deg)
    sigma_phi = deg_2_rad(sigma_phi_deg_t)

    # Add dimensions for broadcasting
    phi = phi.unsqueeze(-1)
    sigma_phi = sigma_phi.unsqueeze(-1) if sigma_phi.dim() > 0 else sigma_phi

    # Compute first column
    c = 2 * PI * d_h
    d = c * torch.arange(0, num_ant, dtype=dtype, device=device)
    d = expand_to_rank(d, phi.dim(), 0)

    a = torch.complex(
        torch.zeros_like(d * torch.sin(phi)),
        d * torch.sin(phi),
    )
    exp_a = torch.exp(a)  # First exponential term

    b = -0.5 * (sigma_phi * d * torch.cos(phi)) ** 2
    exp_b = torch.exp(b).to(cdtype)  # Second exponential term

    col = exp_a * exp_b  # First column

    # First row is just the complex conjugate of first column
    row = torch.conj(col)

    # Build Toeplitz matrix
    r = _toeplitz(col, row)

    return r
