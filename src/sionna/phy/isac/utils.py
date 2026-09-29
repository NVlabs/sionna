#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Utility functions for integrated sensing and communication."""

import math
from typing import Optional, Union

import torch

from sionna._validation import check_one_of, check_tensor_all
from sionna.phy.config import Precision, config, dtypes
from sionna.phy.constants import PI

__all__ = [
    "steering_vectors",
    "angular_delay_doppler_spectrum",
]


def steering_vectors(
    positions: torch.Tensor,
    theta: Union[float, torch.Tensor],
    phi: Union[float, torch.Tensor],
    wavelength: Union[float, torch.Tensor],
    *,
    mode: str = "cartesian",
    precision: Optional[Precision] = None,
) -> torch.Tensor:
    r"""Generate normalized steering vectors.

    For an array with :math:`M` antennas at positions
    :math:`\mathbf{d}_m\in\mathbb{R}^3`, this function computes the normalized
    steering vector
    :math:`\mathbf{a}(\theta,\varphi)\in\mathbb{C}^M` with elements

    .. math::

        a_m(\theta,\varphi)
        = \frac{1}{\sqrt{M}}
          \exp\left(
          j\frac{2\pi}{\lambda}
          \mathbf{d}_m^{\mathsf{T}}
          \widehat{\mathbf{r}}(\theta,\varphi)
          \right),
        \quad m=1,\dots,M,

    where :math:`\lambda` is the carrier wavelength and

    .. math::

        \widehat{\mathbf{r}}(\theta,\varphi)
        =
        \begin{bmatrix}
        \sin(\theta)\cos(\varphi)\\
        \sin(\theta)\sin(\varphi)\\
        \cos(\theta)
        \end{bmatrix}.

    This is the array response used by the Sionna channel models, normalized
    to unit norm. Beamforming therefore conjugates it, as in
    :func:`angular_delay_doppler_spectrum`, and the matched transmit precoder
    for direction :math:`(\theta,\varphi)` is
    :math:`\mathbf{a}(\theta,\varphi)^*`.

    :param positions: Antenna positions :math:`\mathbf{d}_m=(x,y,z)` in
        meters, shape [num_ant, 3].
    :param theta: Zenith angles in radians. Values must lie in
        :math:`[0,\pi]`.
    :param phi: Azimuth angles in radians. Values conventionally lie in
        :math:`[-\pi,\pi]`, but any finite value is accepted because the
        steering vector is :math:`2\pi`-periodic in :math:`\varphi`.
    :param wavelength: Scalar carrier wavelength in meters.
    :param mode: If ``"cartesian"``, both inputs must be scalar or
        one-dimensional and all combinations are generated. If ``"paired"``,
        ``theta`` and ``phi`` are broadcast and interpreted as angle pairs.
        Defaults to ``"cartesian"``.
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`, :attr:`~sionna.phy.config.Config.precision` is used.

    :output a: [..., num_ant], `torch.complex`. Unit-norm steering
        vectors. For Cartesian mode, the shape is always
        [num_theta, num_phi, num_ant], where scalar angles contribute an axis
        of length one. For paired mode, the shape is the broadcast shape of
        ``theta`` and ``phi`` followed by [num_ant].

    .. rubric:: Notes

    This function only models the phase shifts caused by antenna positions.
    Antenna patterns and polarization-dependent gains are not included.
    Co-located polarization components therefore receive identical phases.

    .. rubric:: Examples

    .. code-block:: python

        import torch
        from sionna.phy.isac import steering_vectors

        positions = torch.tensor([[0., -0.025, 0.],
                                  [0.,  0.025, 0.]])

        # By default, all combinations of the two angles are generated
        theta = torch.tensor([torch.pi/3, torch.pi/2])
        phi = torch.tensor([0., torch.pi/4, torch.pi/2])
        w = steering_vectors(positions, theta, phi, wavelength=0.1)
        # w.shape = torch.Size([2, 3, 2]) # [num_theta, num_phi, num_ant]

        # Paired mode instead broadcasts the angles into direction pairs
        w = steering_vectors(positions, theta, phi[:2], wavelength=0.1,
                             mode="paired")
        # w.shape = torch.Size([2, 2]) # [num_directions, num_ant]
    """
    if not isinstance(positions, torch.Tensor):
        raise TypeError("`positions` must be a torch.Tensor.")
    if positions.dim() != 2 or positions.shape[-1] != 3:
        raise ValueError("`positions` must have shape [num_ant, 3].")
    if positions.shape[0] == 0:
        raise ValueError("`positions` must contain at least one antenna.")
    if positions.is_complex():
        raise TypeError("`positions` must be real-valued.")
    for value, name in ((theta, "theta"), (phi, "phi"),
                        (wavelength, "wavelength")):
        if isinstance(value, torch.Tensor) and value.is_complex():
            raise TypeError(f"`{name}` must be real-valued.")
    check_one_of(mode, ("paired", "cartesian"), name="mode")

    if precision is None:
        rdtype = config.dtype
    else:
        rdtype = dtypes[precision]["torch"]["dtype"]

    device = positions.device
    positions = positions.to(dtype=rdtype)
    # Validate in the input dtype so a rounded pi endpoint remains valid
    # when the requested computation precision is higher.
    theta_dtype = theta.dtype if isinstance(theta, torch.Tensor) else rdtype
    theta = torch.as_tensor(theta, dtype=theta_dtype, device=device)
    phi = torch.as_tensor(phi, dtype=rdtype, device=device)
    wavelength = torch.as_tensor(wavelength, dtype=rdtype, device=device)

    if theta.numel() == 0 or phi.numel() == 0:
        raise ValueError("`theta` and `phi` must not be empty.")
    if wavelength.dim() != 0:
        raise ValueError("`wavelength` must be scalar.")
    check_tensor_all(
        torch.isfinite(wavelength) & (wavelength > 0),
        name="wavelength",
        message="`wavelength` must be finite and strictly positive.",
    )
    check_tensor_all(
        torch.isfinite(theta) & (theta >= 0) & (theta <= PI),
        name="theta",
        message="`theta` must contain finite values in [0, pi].",
    )
    theta = theta.to(dtype=rdtype)
    check_tensor_all(
        torch.isfinite(phi),
        name="phi",
        message="`phi` must contain only finite values.",
    )

    if mode == "paired":
        try:
            theta, phi = torch.broadcast_tensors(theta, phi)
        except RuntimeError as err:
            raise ValueError(
                "`theta` and `phi` must have broadcast-compatible shapes in "
                "paired mode."
            ) from err
    else:
        if theta.dim() > 1 or phi.dim() > 1:
            raise ValueError(
                "`theta` and `phi` must be scalar or one-dimensional in "
                "cartesian mode."
            )
        theta = theta.reshape(-1, 1)
        phi = phi.reshape(1, -1)

    direction = torch.stack(
        (
            torch.sin(theta) * torch.cos(phi),
            torch.sin(theta) * torch.sin(phi),
            torch.cos(theta) * torch.ones_like(phi),
        ),
        dim=-1,
    )
    phase = (
        2 * PI / wavelength
        * torch.einsum("...c,mc->...m", direction, positions)
    )
    a = torch.exp(torch.complex(torch.zeros_like(phase), phase))
    a = a / torch.sqrt(
        torch.as_tensor(positions.shape[0], dtype=rdtype, device=device)
    )
    return a


def angular_delay_doppler_spectrum(
    h_dd: torch.Tensor,
    rx_steering_vectors: torch.Tensor,
    tx_steering_vectors: torch.Tensor,
    *,
    mode: str = "paired",
) -> torch.Tensor:
    r"""Compute a Bartlett-type angular delay-Doppler spectrum.

    Let
    :math:`\mathbf{H}_{b,r,t,q,\ell}\in\mathbb{C}^{M_\text{R}\times
    M_\text{T}}` denote the MIMO channel at Doppler bin :math:`q` and delay
    bin :math:`\ell` for transmitter :math:`t`, receiver :math:`r`, and
    arbitrary batch index :math:`b`. For receive and transmit steering vectors
    :math:`\mathbf{a}_{\text{R},r,i}` and
    :math:`\mathbf{a}_{\text{T},t,j}`, the Cartesian spectrum is

    .. math::

        P_{b,r,i,t,j,q,\ell}
        =
        \left|
        \mathbf{a}_{\text{R},r,i}^{\mathsf{H}}
        \mathbf{H}_{b,r,t,q,\ell}
        \mathbf{a}_{\text{T},t,j}^*
        \right|^2.

    Both array responses enter conjugated, which is most apparent in index
    notation,

    .. math::

        P
        =
        \left|
        \sum_{m=1}^{M_\text{R}}\sum_{n=1}^{M_\text{T}}
        a_{\text{R},m}^*\,H_{mn}\,a_{\text{T},n}^*
        \right|^2.

    The Hermitian transpose above merely reflects that
    :math:`\mathbf{a}_\text{R}` is contracted from the left, where a row
    vector is required, whereas :math:`\mathbf{a}_\text{T}` must stay a
    column.

    This conjugation pattern differs from the classical Bartlett spectrum
    :math:`\mathbf{a}^{\mathsf{H}}\mathbf{R}\mathbf{a}`, which is defined on a
    covariance matrix and therefore already carries a conjugation in its own
    outer product. A propagation channel does not: a target contributes
    :math:`\mathbf{H}=\beta\mathbf{a}_\text{R}\mathbf{a}_\text{T}^{\mathsf{T}}`
    with a plain transpose, so both scan vectors must be conjugated for the
    phases to cancel. Scanning that target then gives
    :math:`\beta\|\mathbf{a}_\text{R}\|^2\|\mathbf{a}_\text{T}\|^2` instead of
    a sum of squared phasors.

    :param h_dd: MIMO delay-Doppler channel, shape
        [..., num_rx, num_rx_ant, num_tx, num_tx_ant, num_doppler_bins,
        num_delay_bins]. The last dimension corresponds to the time lags
        returned by
        :func:`~sionna.phy.channel.ofdm_to_delay_doppler_channel`.
    :param rx_steering_vectors: Shared receive steering vectors with shape
        [num_rx_directions, num_rx_ant], or receiver-specific vectors with
        shape [num_rx, num_rx_directions, num_rx_ant]. Array responses as
        returned by :func:`steering_vectors`; they are conjugated internally.
    :param tx_steering_vectors: Shared transmit steering vectors with shape
        [num_tx_directions, num_tx_ant], or transmitter-specific vectors with
        shape [num_tx, num_tx_directions, num_tx_ant]. Array responses as
        returned by :func:`steering_vectors`; they are conjugated internally.
    :param mode: If ``"paired"``, direction indices are paired. Their counts
        must then match or one count must be one. If ``"cartesian"``, all
        receive and transmit direction combinations are evaluated. Defaults to
        ``"paired"``.

    :output spectrum: `torch.float`. Linear power spectrum. Paired mode
        returns [..., num_rx, num_tx, num_direction_pairs, num_doppler_bins,
        num_delay_bins]. Cartesian mode returns
        [..., num_rx, num_rx_directions, num_tx, num_tx_directions,
        num_doppler_bins, num_delay_bins].

    .. rubric:: Notes

    The output is an angular delay-Doppler spectrum. Converting delay to range
    depends on the sensing geometry. For a monostatic system,
    :math:`R=c\tau/2`; for a bistatic system, delay represents the total
    transmitter-target-receiver path length divided by :math:`c`.

    For the unit-norm steering vectors returned by :func:`steering_vectors`,
    no spectrum value exceeds the channel energy
    :math:`\|\mathbf{H}_{b,r,t,q,\ell}\|_\text{F}^2`. If the receive and
    transmit steering vectors each form an orthonormal basis, the spectrum
    summed over all direction pairs equals this energy. For single-antenna
    devices, the spectrum reduces to :math:`|H_{b,r,t,q,\ell}|^2`.

    The channel and both steering banks are promoted to their common
    :func:`torch.promote_types` data type, so the output precision follows the
    widest input rather than that of ``h_dd``.

    Cartesian mode carries a separate axis for the receive and transmit
    directions and therefore scales with their product. Scanning the same
    direction at both ends, as in a monostatic system, is much cheaper in
    paired mode.

    .. rubric:: Examples

    .. code-block:: python

        import torch
        from sionna.phy.isac import (angular_delay_doppler_spectrum,
            steering_vectors)

        # Half-wavelength uniform linear array with four antennas
        wavelength = 0.1
        positions = torch.zeros(4, 3)
        positions[:, 1] = torch.arange(4)*wavelength/2

        # Scan 25 azimuth directions in the horizontal plane, then flatten
        # the grid into a list of directions
        theta = torch.tensor([torch.pi/2])
        phi = torch.deg2rad(torch.linspace(-60., 60., 25))
        steering = steering_vectors(positions, theta, phi, wavelength)
        steering = steering.reshape(-1, positions.shape[0])

        # Delay-Doppler channel as returned by
        # sionna.phy.channel.ofdm_to_delay_doppler_channel, with shape
        # [batch, num_rx, num_rx_ant, num_tx, num_tx_ant, num_doppler_bins,
        # num_delay_bins]
        h_dd = torch.randn(1, 1, 4, 1, 4, 32, 64, dtype=torch.complex64)

        # Monostatic scan: pair each RX direction with the same TX direction
        spectrum = angular_delay_doppler_spectrum(h_dd, steering, steering)
        print(spectrum.shape)
        # torch.Size([1, 1, 1, 25, 32, 64])

        # Cartesian mode evaluates all RX-TX direction combinations instead
        spectrum = angular_delay_doppler_spectrum(h_dd, steering, steering,
                                                  mode="cartesian")
        print(spectrum.shape)
        # torch.Size([1, 1, 25, 1, 25, 32, 64])
    """
    if not isinstance(h_dd, torch.Tensor):
        raise TypeError("`h_dd` must be a torch.Tensor.")
    if h_dd.dim() < 6:
        raise ValueError(
            "`h_dd` must have shape [..., num_rx, num_rx_ant, num_tx, "
            "num_tx_ant, num_doppler_bins, num_delay_bins]."
        )
    if not h_dd.is_complex():
        raise TypeError("`h_dd` must be complex-valued.")
    if not isinstance(rx_steering_vectors, torch.Tensor):
        raise TypeError("`rx_steering_vectors` must be a torch.Tensor.")
    if not isinstance(tx_steering_vectors, torch.Tensor):
        raise TypeError("`tx_steering_vectors` must be a torch.Tensor.")
    check_one_of(mode, ("paired", "cartesian"), name="mode")

    num_rx = h_dd.shape[-6]
    num_rx_ant = h_dd.shape[-5]
    num_tx = h_dd.shape[-4]
    num_tx_ant = h_dd.shape[-3]

    # `einsum` requires matching data types, so promote to the widest input
    # rather than casting the steering banks down to the channel data type
    dtype = torch.promote_types(h_dd.dtype, rx_steering_vectors.dtype)
    dtype = torch.promote_types(dtype, tx_steering_vectors.dtype)
    h_dd = h_dd.to(dtype=dtype)

    rx_steering_vectors = _prepare_steering_bank(
        rx_steering_vectors,
        num_devices=num_rx,
        num_ant=num_rx_ant,
        name="rx_steering_vectors",
        dtype=dtype,
        device=h_dd.device,
    )
    tx_steering_vectors = _prepare_steering_bank(
        tx_steering_vectors,
        num_devices=num_tx,
        num_ant=num_tx_ant,
        name="tx_steering_vectors",
        dtype=dtype,
        device=h_dd.device,
    )

    # Both banks hold array responses, so beamforming conjugates them
    rx_steering_vectors = rx_steering_vectors.conj()
    tx_steering_vectors = tx_steering_vectors.conj()

    if mode == "cartesian":
        beamformed = torch.einsum(
            "rim,...rmtnql,tjn->...ritjql",
            rx_steering_vectors,
            h_dd,
            tx_steering_vectors,
        )
    else:
        num_rx_directions = rx_steering_vectors.shape[1]
        num_tx_directions = tx_steering_vectors.shape[1]
        if (
            num_rx_directions != num_tx_directions
            and num_rx_directions != 1
            and num_tx_directions != 1
        ):
            raise ValueError(
                "In paired mode, the RX and TX direction counts must match "
                "or one of them must be one."
            )
        num_directions = max(num_rx_directions, num_tx_directions)
        rx_steering_vectors = rx_steering_vectors.expand(
            num_rx, num_directions, num_rx_ant
        )
        tx_steering_vectors = tx_steering_vectors.expand(
            num_tx, num_directions, num_tx_ant
        )
        # Choose the smallest intermediate explicitly. Its common receiver,
        # transmitter, and direction axes leave only the antenna counts and
        # the number of delay-Doppler bins (including batches) to compare.
        num_bins = math.prod(h_dd.shape[:-6]) * math.prod(h_dd.shape[-2:])
        if num_bins >= max(num_rx_ant, num_tx_ant):
            paired_vectors = torch.einsum(
                "rdm,tdn->rtdmn", rx_steering_vectors, tx_steering_vectors
            )
            beamformed = torch.einsum(
                "rtdmn,...rmtnql->...rtdql", paired_vectors, h_dd
            )
        elif num_rx_ant >= num_tx_ant:
            projected = torch.einsum(
                "rdm,...rmtnql->...rtdnql", rx_steering_vectors, h_dd
            )
            beamformed = torch.einsum(
                "...rtdnql,tdn->...rtdql", projected, tx_steering_vectors
            )
        else:
            projected = torch.einsum(
                "...rmtnql,tdn->...rtdmql", h_dd, tx_steering_vectors
            )
            beamformed = torch.einsum(
                "rdm,...rtdmql->...rtdql", rx_steering_vectors, projected
            )

    return beamformed.abs().square()


def _prepare_steering_bank(
    vectors: torch.Tensor,
    *,
    num_devices: int,
    num_ant: int,
    name: str,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Validate and expand a shared or device-specific steering bank."""
    if vectors.dim() == 2:
        if vectors.shape[-1] != num_ant:
            raise ValueError(
                f"`{name}` must have {num_ant} antenna coefficients."
            )
        vectors = vectors.unsqueeze(0).expand(num_devices, -1, -1)
    elif vectors.dim() == 3:
        if vectors.shape[0] != num_devices or vectors.shape[-1] != num_ant:
            raise ValueError(
                f"`{name}` must have shape [{num_devices}, "
                f"num_directions, {num_ant}]."
            )
    else:
        raise ValueError(
            f"`{name}` must have shape [num_directions, num_ant] or "
            f"[num_devices, num_directions, num_ant]."
        )
    if vectors.shape[1] == 0:
        raise ValueError(f"`{name}` must contain at least one direction.")
    return vectors.to(dtype=dtype, device=device)
