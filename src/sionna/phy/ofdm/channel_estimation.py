#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Classes and functions related to OFDM channel estimation."""

import itertools
from abc import abstractmethod
from typing import List, Optional, Tuple, Sequence

import numpy as np
import torch
from scipy.special import jv

from sionna.phy import Block, PI, SPEED_OF_LIGHT
from sionna.phy.channel.tr38901 import models
from sionna.phy.config import config, dtypes, Precision
from sionna.phy.object import Object
from sionna.phy.ofdm import RemoveNulledSubcarriers, ResourceGrid, PilotPattern
from sionna.phy.utils import expand_to_rank, flatten_last_dims

__all__ = [
    "BaseChannelEstimator",
    "BasePilotChannelEstimator",
    "LSChannelEstimator",
    "LMMSEChannelEstimator",
    "BaseChannelInterpolator",
    "NearestNeighborInterpolator",
    "LinearInterpolator",
    "tdl_freq_cov_mat",
    "tdl_time_cov_mat",
]


class BaseChannelEstimator(Block):
    r"""Abstract block for implementing an OFDM channel estimator.

    Any block that implements an OFDM channel estimator must implement this
    class and its
    :meth:`~sionna.phy.ofdm.BaseChannelEstimator.call`
    abstract method.

    This class calls the :meth:`~sionna.phy.ofdm.BaseChannelEstimator.call`
    method to estimate the channel for the entire resource grid.

    :param resource_grid: Resource grid
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`,
        :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for tensor operations. If `None`,
        :attr:`~sionna.phy.config.Config.device` is used.

    :input y: [batch_size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size], `torch.complex`.
        Observed signals.
    :input no: [batch_size, num_rx, num_rx_ant] or only the first n>=0 dims, `torch.float`.
        Variance of the AWGN.

    :output h_hat: [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols, num_effective_subcarriers], `torch.complex`.
        Channel estimates across the entire resource grid for all
        transmitters and streams.
    :output err_var: Same shape as ``h_hat``, `torch.float`.
        Channel estimation error variance across the entire resource grid
        for all transmitters and streams.
    """

    def __init__(
        self,
        resource_grid: ResourceGrid,
        precision: Optional[Precision] = None,
        device: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(precision=precision, device=device, **kwargs)

        if not isinstance(resource_grid, ResourceGrid):
            raise TypeError(
                "You must provide a valid instance of ResourceGrid."
            )
        self._pilot_pattern = resource_grid.pilot_pattern
        self._remove_nulled_scs = RemoveNulledSubcarriers(
            resource_grid, precision=self.precision, device=self.device
        )

        # Precompute indices to gather received pilot signals
        self._pilot_ind = _pilot_ind_from_pattern(
            self._pilot_pattern, device=self.device
        )

    @abstractmethod
    def call(
        self, y: torch.Tensor, no: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Estimate the channel.

        This is an abstract method that must be implemented by a concrete
        OFDM channel estimator that implements this class.

        :param y: [batch_size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size], `torch.complex`.
            Observed signals.
        :param no: [batch_size, num_rx, num_rx_ant] or only the first n>=0 dims, `torch.float`.
            Variance of the AWGN.

        :output h_hat: [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols, num_effective_subcarriers], `torch.complex`.
            Channel estimates across the entire resource grid for all
            transmitters and streams.
        :output err_var: Same shape as ``h_hat``, `torch.float`.
            Channel estimation error variance across the entire resource grid
            for all transmitters and streams.
        """
        pass

    def _extract_pilots(self, y: torch.Tensor) -> torch.Tensor:
        # y has shape:
        # [batch_size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size]

        # Remove nulled subcarriers (guards, dc)
        y_eff = self._remove_nulled_scs(y)

        # Flatten the resource grid for pilot extraction
        # New shape: [..., num_ofdm_symbols*num_effective_subcarriers]
        y_eff_flat = flatten_last_dims(y_eff)

        # Gather pilots along the last dimension
        # Expand pilot_ind to match y_eff_flat dimensions
        pilot_ind = self._pilot_ind.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        pilot_ind = pilot_ind.expand(
            y_eff_flat.shape[0], y_eff_flat.shape[1], y_eff_flat.shape[2], -1, -1, -1
        )
        y_pilots = torch.gather(
            y_eff_flat.unsqueeze(3)
            .unsqueeze(4)
            .expand(-1, -1, -1, pilot_ind.shape[3], pilot_ind.shape[4], -1),
            -1,
            pilot_ind,
        )
        return y_pilots


class BasePilotChannelEstimator(BaseChannelEstimator):
    r"""Abstract block for implementing an OFDM channel estimator that first
    estimates the channel at the pilot locations and then obtains channel
    estimates for the entire resource grid through interpolation.

    Any block that implements this class must implement the
    :meth:`~sionna.phy.ofdm.BasePilotChannelEstimator.estimate_at_pilot_locations`
    abstract method.

    This class extracts the pilots from the received resource grid ``y``, calls
    the :meth:`~sionna.phy.ofdm.BasePilotChannelEstimator.estimate_at_pilot_locations`
    method to estimate the channel for the pilot-carrying resource elements,
    and then interpolates the channel to compute channel estimates for the
    data-carrying resource elements using the interpolation method specified by
    ``interpolation_type`` or the ``interpolator`` object.

    :param resource_grid: Resource grid
    :param interpolation_type: The interpolation method to be used.
        It is ignored if ``interpolator`` is not `None`.
        Available options are
        :class:`~sionna.phy.ofdm.NearestNeighborInterpolator` (``"nn"``),
        :class:`~sionna.phy.ofdm.LinearInterpolator` without (``"lin"``) or
        with averaging across OFDM symbols (``"lin_time_avg"``).
    :param interpolator: An instance of
        :class:`~sionna.phy.ofdm.BaseChannelInterpolator` or `None`.
        In the latter case, the interpolator specified
        by ``interpolation_type`` is used.
        Otherwise, the ``interpolator`` is used and ``interpolation_type``
        is ignored.
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`,
        :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for tensor operations. If `None`,
        :attr:`~sionna.phy.config.Config.device` is used.

    :input y: [batch_size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size], `torch.complex`.
        Observed resource grid.
    :input no: [batch_size, num_rx, num_rx_ant] or only the first n>=0 dims, `torch.float`.
        Variance of the AWGN.

    :output h_hat: [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols, num_effective_subcarriers], `torch.complex`.
        Channel estimates across the entire resource grid for all
        transmitters and streams.
    :output err_var: Same shape as ``h_hat``, `torch.float`.
        Channel estimation error variance across the entire resource grid
        for all transmitters and streams.

    .. rubric:: Examples

    .. code-block:: python

        import torch
        from sionna.phy.ofdm import ResourceGrid, LSChannelEstimator

        rg = ResourceGrid(num_ofdm_symbols=14,
                          fft_size=64,
                          subcarrier_spacing=30e3,
                          num_tx=2,
                          num_streams_per_tx=2,
                          pilot_pattern="kronecker",
                          pilot_ofdm_symbol_indices=[2, 11])

        estimator = LSChannelEstimator(rg, interpolation_type="lin")

        batch_size = 16
        y = torch.randn(batch_size, 1, 4, 14, 64, dtype=torch.complex64)
        no = torch.ones(1) * 0.1

        h_hat, err_var = estimator(y, no)
        print(h_hat.shape)
        # torch.Size([16, 1, 4, 2, 2, 14, 60])
    """

    def __init__(
        self,
        resource_grid: ResourceGrid,
        interpolation_type: str = "nn",
        interpolator: Optional["BaseChannelInterpolator"] = None,
        precision: Optional[Precision] = None,
        device: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(resource_grid, precision=precision, device=device, **kwargs)

        if interpolator is not None:
            if not isinstance(interpolator, BaseChannelInterpolator):
                raise TypeError(
                    "`interpolator` must implement the "
                    "BaseChannelInterpolator interface"
                )
            self._interpol = interpolator
        else:
            if interpolation_type not in ("nn", "lin", "lin_time_avg"):
                raise ValueError(
                    "interpolation_type must be one of: 'nn', 'lin', "
                    "'lin_time_avg'"
                )
            self._interpolation_type = interpolation_type
            if self._interpolation_type == "nn":
                self._interpol = NearestNeighborInterpolator(self._pilot_pattern, precision=self.precision, device=self.device)
            elif self._interpolation_type == "lin":
                self._interpol = LinearInterpolator(self._pilot_pattern, precision=self.precision, device=self.device)
            elif self._interpolation_type == "lin_time_avg":
                self._interpol = LinearInterpolator(self._pilot_pattern, time_avg=True, precision=self.precision, device=self.device)

    @abstractmethod
    def estimate_at_pilot_locations(
        self, y_pilots: torch.Tensor, no: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Estimate the channel for the pilot-carrying resource elements.

        This is an abstract method that must be implemented by a concrete
        OFDM channel estimator that implements this class.

        :param y_pilots: [batch_size, num_rx, num_rx_ant, num_tx, num_streams, num_pilot_symbols], `torch.complex`.
            Observed signals for the pilot-carrying resource elements.
        :param no: [batch_size, num_rx, num_rx_ant] or only the first n>=0 dims, `torch.float`.
            Variance of the AWGN.

        :output h_hat: [batch_size, num_rx, num_rx_ant, num_tx, num_streams, num_pilot_symbols], `torch.complex`.
            Channel estimates for the pilot-carrying resource elements.
        :output err_var: Same shape as ``h_hat``, `torch.float`.
            Channel estimation error variance for the pilot-carrying
            resource elements.
        """
        pass

    def call(
        self, y: torch.Tensor, no: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # y has shape:
        # [batch_size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size]
        #
        # no can have shapes [], [batch_size], [batch_size, num_rx]
        # or [batch_size, num_rx, num_rx_ant]

        # Extract pilots from the received resource grid
        y_pilots = self._extract_pilots(y)

        # Compute channel estimates at pilot locations
        h_hat, err_var = self.estimate_at_pilot_locations(y_pilots, no)

        # Interpolate channel estimates over the resource grid
        h_hat, err_var = self._interpol(h_hat, err_var)

        return h_hat, err_var


class LSChannelEstimator(BasePilotChannelEstimator):
    r"""Least-squares (LS) channel estimation for OFDM MIMO systems.

    After LS channel estimation at the pilot positions, the channel estimates
    and error variances are interpolated across the entire resource grid using
    a specified interpolation function.

    For simplicity, the underlying algorithm is described for a vectorized
    observation, where we have a nonzero pilot for all elements to be estimated.
    The actual implementation works on a full OFDM resource grid with sparse
    pilot patterns. The following model is assumed:

    .. math::

        \mathbf{y} = \mathbf{h}\odot\mathbf{p} + \mathbf{n}

    where :math:`\mathbf{y}\in\mathbb{C}^{M}` is the received signal vector,
    :math:`\mathbf{p}\in\mathbb{C}^M` is the vector of pilot symbols,
    :math:`\mathbf{h}\in\mathbb{C}^{M}` is the channel vector to be estimated,
    and :math:`\mathbf{n}\in\mathbb{C}^M` is a zero-mean noise vector whose
    elements have variance :math:`N_0`. The operator :math:`\odot` denotes
    element-wise multiplication.

    The channel estimate :math:`\hat{\mathbf{h}}` and error variances
    :math:`\sigma^2_i`, :math:`i=0,\dots,M-1`, are computed as

    .. math::

        \hat{\mathbf{h}} &= \mathbf{y} \odot
                           \frac{\mathbf{p}^\star}{\left|\mathbf{p}\right|^2}
                         = \mathbf{h} + \tilde{\mathbf{h}}\\
             \sigma^2_i &= \mathbb{E}\left[\tilde{h}_i \tilde{h}_i^\star \right]
                         = \frac{N_0}{\left|p_i\right|^2}.

    The channel estimates and error variances are then interpolated across
    the entire resource grid.

    :param resource_grid: Resource grid
    :param interpolation_type: The interpolation method to be used.
        It is ignored if ``interpolator`` is not `None`.
        Available options are
        :class:`~sionna.phy.ofdm.NearestNeighborInterpolator` (``"nn"``),
        :class:`~sionna.phy.ofdm.LinearInterpolator` without (``"lin"``) or
        with averaging across OFDM symbols (``"lin_time_avg"``).
    :param interpolator: An instance of
        :class:`~sionna.phy.ofdm.BaseChannelInterpolator` or `None`.
        In the latter case, the interpolator specified
        by ``interpolation_type`` is used.
        Otherwise, the ``interpolator`` is used and ``interpolation_type``
        is ignored.
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`,
        :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for tensor operations. If `None`,
        :attr:`~sionna.phy.config.Config.device` is used.

    :input y: [batch_size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size], `torch.complex`.
        Observed resource grid.
    :input no: [batch_size, num_rx, num_rx_ant] or only the first n>=0 dims, `torch.float`.
        Variance of the AWGN.

    :output h_ls: [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols, num_effective_subcarriers], `torch.complex`.
        Channel estimates across the entire resource grid for all
        transmitters and streams.
    :output err_var: Same shape as ``h_ls``, `torch.float`.
        Channel estimation error variance across the entire resource grid
        for all transmitters and streams.
    """

    def estimate_at_pilot_locations(
        self, y_pilots: torch.Tensor, no: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # y_pilots : [batch_size, num_rx, num_rx_ant, num_tx, num_streams,
        #               num_pilot_symbols], torch.complex
        #     The observed signals for the pilot-carrying resource elements.
        #
        # no : [batch_size, num_rx, num_rx_ant] or only the first n>=0 dims,
        #   torch.float
        #     The variance of the AWGN.
        return _estimate_ls_at_pilot_locations(y_pilots, no, self._pilot_pattern.pilots)



class LMMSEChannelEstimator(BaseChannelEstimator):
    r"""LMMSE channel estimation for MIMO OFDM systems, where LMMSE filtering
    (interpolation and smoothing) is applied across time, frequency, and (optionally) spatial
    dimensions. Here, *interpolation* refers to estimating the channel at resource elements
    without an input estimate, while *smoothing* refers to updating existing
    estimates using channel statistics and error variances. The frequency and time
    passes perform both interpolation and smoothing, whereas the optional spatial
    pass performs smoothing only. We use *filtering* as an umbrella term for both
    operations.

    Depending on the value of ``order``, the filtering is
    carried out across time (t), i.e., OFDM symbols, frequency (f), i.e.,
    subcarriers, and optionally space (s), i.e., receive antennas, in any
    desired order.

    For simplicity, we describe the underlying algorithm assuming that
    filtering across the sub-carriers is performed first,
    followed by filtering across OFDM symbols, and finally
    by smoothing across receive antennas.
    The algorithm is similar if filtering is performed in
    a different order.
    For clarity, antenna indices are omitted when describing frequency and time
    filtering, as the same process is applied to all the antennas.

    From the input ``y``, we start by extracting the observations at the non-zero
    positions of the ``PilotPattern`` and arranging them in a resource grid
    with the following model:
    :math:`\mathbf{Y} = \mathbf{H} \odot \mathbf{P} + \mathbf{N} \in \mathbb{C}^{N \times M}`,
    where :math:`\mathbf{Y}` are the observations, :math:`\mathbf{H} \in \mathbb{C}^{N \times M}`
    is the channel matrix, :math:`\mathbf{P} \in \mathbb{C}^{N \times M}` is the matrix of
    non-zero pilots, and :math:`\mathbf{N} \in \mathbb{C}^{N \times M}` is the noise matrix.
    :math:`N` denotes the number of OFDM symbols and :math:`M` the number of sub-carriers.

    A least-squares (LS) estimate and corresponding error variances are first computed as:

    .. math::

        \hat{\mathbf{H}} &= \mathbf{Y} \odot
                           \frac{\mathbf{P}^\star}{\left|\mathbf{P}\right|^2}
                         = \mathbf{H} + \tilde{\mathbf{H}}\\
             \sigma^2_{n,m} &= \mathbb{E}\left[\tilde{h}_{n,m} \tilde{h}_{n,m}^\star \right]
                         = \frac{N_0}{\left|p_{n,m}\right|^2}.

    The first pass then consists in filtering across the sub-carriers:

    .. math::
        \hat{\mathbf{h}}_n^{(1)} = \mathbf{A}_n \hat{\mathbf{h}}_n

    where :math:`1 \leq n \leq N` is the OFDM symbol index and
    :math:`\hat{\mathbf{h}}_n` is the :math:`n^{\text{th}}` (transposed) row
    of :math:`\hat{\mathbf{H}}`.
    :math:`\mathbf{A}_n` is the :math:`M \times M` matrix such that:

    .. math::
        \mathbf{A}_n = \bar{\mathbf{A}}_n \mathbf{\Pi}_n^\intercal

    where

    .. math::
        \bar{\mathbf{A}}_n = \underset{\mathbf{Z} \in \mathbb{C}^{M \times K_n}}{\text{argmin}} \left\lVert \mathbf{Z}\left( \mathbf{\Pi}_n^\intercal \mathbf{R^{(f)}} \mathbf{\Pi}_n + \mathbf{\Sigma}_n \right) - \mathbf{R^{(f)}} \mathbf{\Pi}_n \right\rVert_{\text{F}}^2

    and :math:`\mathbf{R^{(f)}}` is the :math:`M \times M` channel frequency
    covariance matrix,
    :math:`\mathbf{\Pi}_n` the :math:`M \times K_n` matrix that spreads
    :math:`K_n` values to a vector of size :math:`M` according to the
    ``pilot_pattern`` for the :math:`n^{\text{th}}` OFDM symbol,
    and :math:`\mathbf{\Sigma}_n \in \mathbb{R}^{K_n \times K_n}` is the
    channel estimation error covariance built from ``err_var`` and assumed to
    be diagonal.
    Computation of :math:`\bar{\mathbf{A}}_n` uses a diagonally loaded
    Cholesky factorization followed by triangular solves. The diagonal loading
    improves numerical stability for badly conditioned covariance matrices.

    The channel estimation error variances after the first filtering pass
    are computed as

    .. math::
        \mathbf{\Sigma}^{(1)}_n = \text{diag} \left( \mathbf{R^{(f)}} - \mathbf{A}_n \mathbf{\Xi}_n \mathbf{R^{(f)}} \right)

    where :math:`\mathbf{\Xi}_n` is the diagonal matrix of size
    :math:`M \times M` that zeros the columns corresponding to sub-carriers
    not carrying any pilots.
    Note that interpolation is not performed for OFDM symbols which do not
    carry pilots.

    **Remark**: The filter matrix differs across OFDM symbols as
    different OFDM symbols may carry pilots on different sub-carriers and/or
    have different estimation error variances.

    Scaling of the estimates is then performed to ensure that their
    variances match the ones expected by the next filtering step, and the
    error variances are updated accordingly:

    .. math::
        \begin{aligned}
            \left[\hat{\mathbf{h}}_n^{(2)}\right]_m &= s_{n,m} \left[\hat{\mathbf{h}}_n^{(1)}\right]_m\\
            \left[\mathbf{\Sigma}^{(2)}_n\right]_{m,m}  &= s_{n,m}\left( s_{n,m}-1 \right) \left[\hat{\mathbf{\Sigma}}^{(1)}_n\right]_{m,m} + \left( 1 - s_{n,m} \right) \left[\mathbf{R^{(f)}}\right]_{m,m} + s_{n,m} \left[\mathbf{\Sigma}^{(1)}_n\right]_{m,m}
        \end{aligned}

    where the scaling factor :math:`s_{n,m}` is such that:

    .. math::
        \mathbb{E} \left\{ \left\lvert s_{n,m} \left[\hat{\mathbf{h}}_n^{(1)}\right]_m \right\rvert^2 \right\} = \left[\mathbf{R^{(f)}}\right]_{m,m} +  \mathbb{E} \left\{ \left\lvert s_{n,m} \left[\hat{\mathbf{h}}^{(1)}_n\right]_m - \left[\mathbf{h}_n\right]_m \right\rvert^2 \right\}

    which leads to:

    .. math::
        \begin{aligned}
            s_{n,m} &= \frac{2 \left[\mathbf{R^{(f)}}\right]_{m,m}}{\left[\mathbf{R^{(f)}}\right]_{m,m} - \left[\mathbf{\Sigma}^{(1)}_n\right]_{m,m} + \left[\hat{\mathbf{\Sigma}}^{(1)}_n\right]_{m,m}}\\
            \hat{\mathbf{\Sigma}}^{(1)}_n &= \mathbf{A}_n \mathbf{R^{(f)}} \mathbf{A}_n^{\mathrm{H}}.
        \end{aligned}

    The second pass consists in filtering across the OFDM symbols:

    .. math::
        \hat{\mathbf{h}}_m^{(3)} = \mathbf{B}_m \tilde{\mathbf{h}}^{(2)}_m

    where :math:`1 \leq m \leq M` is the sub-carrier index and
    :math:`\tilde{\mathbf{h}}^{(2)}_m` is the :math:`m^{\text{th}}` column of

    .. math::
        \hat{\mathbf{H}}^{(2)} = \begin{bmatrix}
                                    {\hat{\mathbf{h}}_1^{(2)}}^\intercal\\
                                    \vdots\\
                                    {\hat{\mathbf{h}}_N^{(2)}}^\intercal
                                 \end{bmatrix}

    and :math:`\mathbf{B}_m` is the :math:`N \times N` LMMSE filter
    matrix:

    .. math::
        \mathbf{B}_m = \bar{\mathbf{B}}_m \tilde{\mathbf{\Pi}}_m^\intercal

    where

    .. math::
        \bar{\mathbf{B}}_m = \underset{\mathbf{Z} \in \mathbb{C}^{N \times L_m}}{\text{argmin}} \left\lVert \mathbf{Z} \left( \tilde{\mathbf{\Pi}}_m^\intercal \mathbf{R^{(t)}}\tilde{\mathbf{\Pi}}_m + \tilde{\mathbf{\Sigma}}^{(2)}_m \right) -  \mathbf{R^{(t)}}\tilde{\mathbf{\Pi}}_m \right\rVert_{\text{F}}^2

    where :math:`\mathbf{R^{(t)}}` is the :math:`N \times N` channel time
    covariance matrix,
    :math:`\tilde{\mathbf{\Pi}}_m` the :math:`N \times L_m` matrix that
    spreads :math:`L_m` values to a vector of size :math:`N` according to the
    ``pilot_pattern`` for the :math:`m^{\text{th}}` sub-carrier,
    and :math:`\tilde{\mathbf{\Sigma}}^{(2)}_m \in \mathbb{R}^{L_m \times L_m}`
    is the diagonal matrix of channel estimation error variances
    built by gathering the error variances from
    (:math:`\mathbf{\Sigma}^{(2)}_1,\dots,\mathbf{\Sigma}^{(2)}_N`)
    corresponding to resource elements carried by the :math:`m^{\text{th}}`
    sub-carrier.
    Computation of :math:`\bar{\mathbf{B}}_m` uses a diagonally loaded
    Cholesky factorization followed by triangular solves. The diagonal loading
    improves numerical stability for badly conditioned covariance matrices.

    The resulting channel estimate for the resource grid is

    .. math::
        \hat{\mathbf{H}}^{(3)} = \left[ \hat{\mathbf{h}}_1^{(3)} \dots \hat{\mathbf{h}}_M^{(3)} \right]

    The resulting channel estimation error variances are the diagonal
    coefficients of the matrices

    .. math::
        \mathbf{\Sigma}^{(3)}_m = \mathbf{R^{(t)}} - \mathbf{B}_m \tilde{\mathbf{\Xi}}_m \mathbf{R^{(t)}}, 1 \leq m \leq M

    where :math:`\tilde{\mathbf{\Xi}}_m` is the diagonal matrix of size
    :math:`N \times N` that zeros the columns corresponding to OFDM symbols
    not carrying any pilots.

    **Remark**: The filter matrix differs across sub-carriers as
    different sub-carriers may have different estimation error variances
    computed by the first pass.
    However, all sub-carriers carry at least one channel estimate as a result
    of the first pass, ensuring that a channel estimate is computed for all the
    resource elements after the second pass.

    **Remark:** LMMSE filtering requires knowledge of the time and
    frequency covariance matrices of the channel.
    The functions :func:`~sionna.phy.ofdm.tdl_time_cov_mat`
    and :func:`~sionna.phy.ofdm.tdl_freq_cov_mat` compute the expected time
    and frequency covariance matrices, respectively, for the
    :class:`~sionna.phy.channel.tr38901.TDL` channel models.

    Scaling of the estimates is then performed to ensure that their
    variances match the ones expected by the next smoothing step, and the
    error variances are updated accordingly:

    .. math::
        \begin{aligned}
            \left[\hat{\mathbf{h}}_m^{(4)}\right]_n &= \gamma_{m,n} \left[\hat{\mathbf{h}}_m^{(3)}\right]_n\\
            \left[\mathbf{\Sigma}^{(4)}_m\right]_{n,n}  &= \gamma_{m,n}\left( \gamma_{m,n}-1 \right) \left[\hat{\mathbf{\Sigma}}^{(3)}_m\right]_{n,n} + \left( 1 - \gamma_{m,n} \right) \left[\mathbf{R^{(t)}}\right]_{n,n} + \gamma_{m,n} \left[\mathbf{\Sigma}^{(3)}_n\right]_{m,m}
        \end{aligned}

    where:

    .. math::
        \begin{aligned}
            \gamma_{m,n} &= \frac{2 \left[\mathbf{R^{(t)}}\right]_{n,n}}{\left[\mathbf{R^{(t)}}\right]_{n,n} - \left[\mathbf{\Sigma}^{(3)}_m\right]_{n,n} + \left[\hat{\mathbf{\Sigma}}^{(3)}_n\right]_{m,m}}\\
            \hat{\mathbf{\Sigma}}^{(3)}_m &= \mathbf{B}_m \mathbf{R^{(t)}} \mathbf{B}_m^{\mathrm{H}}
        \end{aligned}

    Finally, if requested, a spatial smoothing step is applied to every resource element
    carrying a channel estimate.
    For clarity, we drop the resource element indexing :math:`(n,m)`.
    We denote by :math:`L` the number of receive antennas, and by
    :math:`\mathbf{R^{(s)}}\in\mathbb{C}^{L \times L}` the spatial covariance
    matrix.

    LMMSE spatial smoothing consists in the following computations:

    .. math::
        \hat{\mathbf{h}}^{(5)} = \mathbf{C} \hat{\mathbf{h}}^{(4)}

    where

    .. math::
        \mathbf{C} = \mathbf{R^{(s)}} \left( \mathbf{R^{(s)}} + \mathbf{\Sigma}^{(4)} \right)^{-1}.

    The estimation error variances are the diagonal coefficients of

    .. math::
        \mathbf{\Sigma}^{(5)} = \mathbf{R^{(s)}} - \mathbf{C}\mathbf{R^{(s)}}

    The smoothed channel estimate :math:`\hat{\mathbf{h}}^{(5)}` and
    corresponding error variances
    :math:`\text{diag}\left( \mathbf{\Sigma}^{(5)} \right)` are
    returned for every resource element :math:`(m,n)`.

    **Remark:** No scaling is performed after the last filtering step.

    **Remark:** All passes assume that the estimation error covariance matrix
    (:math:`\mathbf{\Sigma}`,
    :math:`\tilde{\mathbf{\Sigma}}^{(2)}`, or
    :math:`\tilde{\mathbf{\Sigma}}^{(4)}`) is diagonal, which
    may not be accurate. When this assumption does not hold, this estimator
    is only an approximation of LMMSE estimation.

    **Remark:** The order in which frequency filtering, temporal
    filtering, and, optionally, spatial smoothing are applied, is
    controlled using the ``order`` parameter.

    :param resource_grid: Used resource grid.
    :param cov_mat_time: Time covariance matrix of the channel.
    :param cov_mat_freq: Frequency covariance matrix of the channel.
    :param cov_mat_space: Spatial covariance matrix of the channel.
        Only required if spatial smoothing is requested (see ``order``).
    :param order: Order in which to perform filtering.
        For example, ``"t-f-s"`` means that filtering across
        the OFDM symbols is performed first (``"t"``: time), followed by
        filtering across the sub-carriers (``"f"``: frequency), and
        filtering across the receive antennas (``"s"``: space).
        Similarly, ``"f-t"`` means filtering across the sub-carriers
        followed by filtering across the OFDM symbols, and no spatial
        filtering. The spatial covariance matrix (``cov_mat_space``) is
        only required when spatial filtering is requested. Time and frequency
        filtering are not optional to ensure that a channel estimate is
        computed for all resource elements.

    :input y: [batch_size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size], `torch.complex`.
        Observed resource grid.
    :input no: [batch_size, num_rx, num_rx_ant] or only the first n>=0 dims, `torch.float`.
        Variance of the AWGN.

    :output h_hat: [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols, num_effective_subcarriers], `torch.complex`.
        Channel estimates across the entire resource grid for all
        transmitters and streams.
    :output err_var: Same shape as ``h_hat``, `torch.float`.
        Channel estimation error variances across the entire resource grid
        for all transmitters and streams.

    .. rubric:: Examples

    .. code-block:: python

        import torch
        from sionna.phy.ofdm import (
            ResourceGrid, LMMSEChannelEstimator,
            tdl_freq_cov_mat, tdl_time_cov_mat
        )

        rg = ResourceGrid(num_ofdm_symbols=14,
                          fft_size=64,
                          subcarrier_spacing=30e3,
                          num_tx=1,
                          num_streams_per_tx=1,
                          pilot_pattern="kronecker",
                          pilot_ofdm_symbol_indices=[2, 11])

        # Compute covariance matrices
        cov_mat_freq = tdl_freq_cov_mat("A", 30e3, 64, 100e-9)
        cov_mat_time = tdl_time_cov_mat("A", 3.0, 3.5e9, 35.7e-6, 14)

        # Create LMMSE channel estimator
        estimator = LMMSEChannelEstimator(
            rg, cov_mat_time, cov_mat_freq, order="f-t"
        )
    """

    def __init__(
        self,
        resource_grid: ResourceGrid,
        cov_mat_time: torch.Tensor,
        cov_mat_freq: torch.Tensor,
        cov_mat_space: Optional[torch.Tensor] = None,
        order: str = "t-f",
        precision: Optional[Precision] = None,
        device: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(resource_grid, precision=precision, device=device, **kwargs)

        # Check the specified order
        order_list = order.split("-")
        if not 2 <= len(order_list) <= 3:
            raise ValueError("Invalid order for filtering.")
        spatial_filtering = False
        freq_filtering = False
        time_filtering = False
        for o in order_list:
            if o not in ("s", "f", "t"):
                raise ValueError(f"Unknown dimension {o} in `order`")
            if o == "s":
                if spatial_filtering:
                    raise ValueError(
                        "Spatial filtering can be specified at most once"
                    )
                spatial_filtering = True
            elif o == "t":
                if time_filtering:
                    raise ValueError(
                        "Temporal filtering can be specified once only"
                    )
                time_filtering = True
            elif o == "f":
                if freq_filtering:
                    raise ValueError(
                        "Frequency filtering can be specified once only"
                    )
                freq_filtering = True

        if spatial_filtering:
            if cov_mat_space is None:
                raise ValueError(
                    "A spatial covariance matrix is required for spatial "
                    "filtering"
                )
        if not freq_filtering:
            raise ValueError("Frequency filtering is required")
        if not time_filtering:
            raise ValueError("Time filtering is required")

        self._order = order_list
        self._num_ofdm_symbols = self._pilot_pattern.num_ofdm_symbols
        self._num_effective_subcarriers = self._pilot_pattern.num_effective_subcarriers

        # Build pilot masks for every stream
        pilot_mask = self._build_pilot_mask(self._pilot_pattern)

        # Build indices for mapping channel estimates to resource grid
        num_pilots = self._pilot_pattern.pilots.shape[2]
        inputs_to_rg_indices, scatter_indices = self._build_inputs2rg_indices(
            pilot_mask, num_pilots
        )
        # Register scatter indices as tensor buffer for vectorized operations
        self.register_buffer(
            "_scatter_tx",
            torch.tensor(
                scatter_indices[:, 0], dtype=torch.int64, device=self.device
            ),
        )
        self.register_buffer(
            "_scatter_st",
            torch.tensor(
                scatter_indices[:, 1], dtype=torch.int64, device=self.device
            ),
        )
        self.register_buffer(
            "_scatter_p",
            torch.tensor(
                scatter_indices[:, 2], dtype=torch.int64, device=self.device
            ),
        )
        self.register_buffer(
            "_scatter_sb",
            torch.tensor(
                scatter_indices[:, 3], dtype=torch.int64, device=self.device
            ),
        )
        self.register_buffer(
            "_scatter_sc",
            torch.tensor(
                scatter_indices[:, 4], dtype=torch.int64, device=self.device
            ),
        )
        # Register as buffer for CUDAGraph compatibility (on self.device)
        self.register_buffer(
            "_inputs_to_rg_indices",
            torch.tensor(inputs_to_rg_indices, dtype=torch.int64, device=self.device),
        )

        # Build filters according to requested order
        filters = []
        for i, o in enumerate(order_list):
            last_step = i == len(order_list) - 1
            if o == "f":
                lmmse_filter = _LMMSEEstimator1D(
                    pilot_mask, cov_mat_freq, last_step=last_step, precision=self.precision, device=self.device
                )
                pilot_mask = self._update_pilot_mask_interp(pilot_mask)
                err_var_mask = torch.tensor(
                    pilot_mask == 1, dtype=self.dtype, device=self.device
                )
            elif o == "t":
                pilot_mask_t = np.transpose(pilot_mask, [0, 1, 3, 2])
                lmmse_filter = _LMMSEEstimator1D(
                    pilot_mask_t, cov_mat_time, last_step=last_step, precision=self.precision, device=self.device
                )
                pilot_mask = self._update_pilot_mask_interp(pilot_mask_t)
                pilot_mask = np.transpose(pilot_mask, [0, 1, 3, 2])
                err_var_mask = torch.tensor(
                    pilot_mask == 1, dtype=self.dtype, device=self.device
                )
            else:  # 's'
                lmmse_filter = _SpatialLMMSEEstimator1D(cov_mat_space, last_step=last_step, precision=self.precision, device=self.device)
                err_var_mask = torch.tensor(
                    pilot_mask == 1, dtype=self.dtype, device=self.device
                )
            filters.append(lmmse_filter)
            # Register each err_var_mask as a buffer for CUDAGraph compatibility
            self.register_buffer(f"_err_var_mask_{i}", err_var_mask)

        self._filters = filters
        # Build list from registered buffers
        self._err_var_masks = [
            getattr(self, f"_err_var_mask_{i}") for i in range(len(order_list))
        ]

    def _build_pilot_mask(self, pilot_pattern) -> np.ndarray:
        """Build pilot mask indicating which REs are pilots, data, or unused."""
        mask = pilot_pattern.mask.cpu().numpy()
        pilots = pilot_pattern.pilots.cpu().numpy()
        num_tx = mask.shape[0]
        num_streams_per_tx = mask.shape[1]
        num_ofdm_symbols = mask.shape[2]
        num_effective_subcarriers = mask.shape[3]

        pilot_mask = np.zeros(
            [num_tx, num_streams_per_tx, num_ofdm_symbols, num_effective_subcarriers],
            int,
        )
        for tx, st in itertools.product(range(num_tx), range(num_streams_per_tx)):
            pil_index = 0
            for sb, sc in itertools.product(
                range(num_ofdm_symbols), range(num_effective_subcarriers)
            ):
                if mask[tx, st, sb, sc] == 1:
                    if np.abs(pilots[tx, st, pil_index]) > 0.0:
                        pilot_mask[tx, st, sb, sc] = 1
                    else:
                        pilot_mask[tx, st, sb, sc] = 2
                    pil_index += 1

        return pilot_mask

    def _build_inputs2rg_indices(
        self, pilot_mask: np.ndarray, num_pilots: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Build indices for mapping channel estimates to a resource grid."""
        num_tx = pilot_mask.shape[0]
        num_streams_per_tx = pilot_mask.shape[1]
        num_ofdm_symbols = pilot_mask.shape[2]
        num_effective_subcarriers = pilot_mask.shape[3]

        inputs_to_rg_indices = np.zeros(
            [num_tx, num_streams_per_tx, num_pilots, 4], int
        )
        # Pre-compute scatter indices as numpy array for vectorized operations
        scatter_indices_list = []

        for tx, st in itertools.product(range(num_tx), range(num_streams_per_tx)):
            pil_index = 0
            for sb, sc in itertools.product(
                range(num_ofdm_symbols), range(num_effective_subcarriers)
            ):
                if pilot_mask[tx, st, sb, sc] == 0:
                    continue
                if pilot_mask[tx, st, sb, sc] == 1:
                    inputs_to_rg_indices[tx, st, pil_index] = [tx, st, sb, sc]
                    scatter_indices_list.append([tx, st, pil_index, sb, sc])
                pil_index += 1

        scatter_indices = np.array(scatter_indices_list, dtype=np.int64)
        return inputs_to_rg_indices, scatter_indices

    def _update_pilot_mask_interp(self, pilot_mask: np.ndarray) -> np.ndarray:
        """Update pilot mask to label interpolated resource elements."""
        interpolated = np.any(pilot_mask == 1, axis=-1, keepdims=True)
        pilot_mask = np.where(interpolated, 1, pilot_mask)
        return pilot_mask

    def _estimate_ls_at_pilot_locations(
        self, y_pilots: torch.Tensor, no: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Estimate the channel at the pilot locations using LS estimation."""
        h_ls, err_var_ls = _estimate_ls_at_pilot_locations(y_pilots, no, self._pilot_pattern.pilots)
        return h_ls, err_var_ls

    @torch.compiler.disable  # Torchinductor doesn't support complex number codegen
    def call(self, y: torch.Tensor, no: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # y has shape:
        # [batch_size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size]
        #
        # no can have shapes [], [batch_size], [batch_size, num_rx]
        # or [batch_size, num_rx, num_rx_ant]

        # Extract pilots from the received resource grid
        y_pilots = self._extract_pilots(y)

        h_ls, err_var_ls = self._estimate_ls_at_pilot_locations(y_pilots, no)

        output_shape = (
            *h_ls.shape[:-1],
            self._num_ofdm_symbols,
            self._num_effective_subcarriers,
        )
        h_hat_rg = h_ls.new_zeros(output_shape)
        err_var_rg = err_var_ls.new_zeros(output_shape)

        # Scatter values to resource grid using vectorized indexing
        # Uses pre-computed tensor indices for compile-friendly operations
        tx_idx = self._scatter_tx  # [N]
        st_idx = self._scatter_st  # [N]
        p_idx = self._scatter_p  # [N]
        sb_idx = self._scatter_sb  # [N]
        sc_idx = self._scatter_sc  # [N]

        # Vectorized scatter: gather from h_hat and scatter to h_hat_rg
        h_hat_rg[:, :, :, tx_idx, st_idx, sb_idx, sc_idx] = h_ls[
            :, :, :, tx_idx, st_idx, p_idx
        ]
        err_var_rg[:, :, :, tx_idx, st_idx, sb_idx, sc_idx] = err_var_ls[
            :, :, :, tx_idx, st_idx, p_idx
        ]

        h_hat = h_hat_rg
        err_var = err_var_rg

        # Apply filters
        for o, lmmse_filter, err_var_mask in zip(
            self._order, self._filters, self._err_var_masks
        ):
            if o == "f":
                h_hat, err_var = lmmse_filter(h_hat, err_var)
                err_var_mask = expand_to_rank(err_var_mask, err_var.dim(), 0)
                err_var = err_var * err_var_mask
            elif o == "t":
                h_hat = h_hat.permute(0, 1, 2, 3, 4, 6, 5)
                err_var = err_var.permute(0, 1, 2, 3, 4, 6, 5)
                h_hat, err_var = lmmse_filter(h_hat, err_var)
                h_hat = h_hat.permute(0, 1, 2, 3, 4, 6, 5)
                err_var = err_var.permute(0, 1, 2, 3, 4, 6, 5)
                err_var_mask = expand_to_rank(err_var_mask, err_var.dim(), 0)
                err_var = err_var * err_var_mask
            elif o == "s":
                h_hat = h_hat.permute(0, 1, 3, 4, 5, 6, 2)
                err_var = err_var.permute(0, 1, 3, 4, 5, 6, 2)
                h_hat, err_var = lmmse_filter(h_hat, err_var)
                h_hat = h_hat.permute(0, 1, 6, 2, 3, 4, 5)
                err_var = err_var.permute(0, 1, 6, 2, 3, 4, 5)
                err_var_mask = expand_to_rank(err_var_mask, err_var.dim(), 0)
                err_var = err_var * err_var_mask

        return h_hat, err_var



class BaseChannelInterpolator(Object):
    r"""Abstract class for implementing an OFDM channel interpolator.

    Any class that implements an OFDM channel interpolator must implement this
    callable class.

    A channel interpolator may be used by an OFDM channel estimator (e.g.,
    :class:`~sionna.phy.ofdm.BasePilotChannelEstimator`) to compute channel
    estimates for the data-carrying resource elements from the channel
    estimates for the pilot-carrying resource elements.

    Unless time averaging is enabled, an interpolator must preserve the input
    channel estimates and error variances at the corresponding pilot locations.

    When `time_avg` is `True`, frequency-interpolated channel estimates at
    pilot-bearing OFDM symbols are averaged and the resulting estimate is used
    for every OFDM symbol.

    :param time_avg: If `True`, measurements will be averaged across OFDM
        symbols (i.e., time). This is useful for channels that do not vary
        substantially over the duration of an OFDM frame. Defaults to `False`.
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`,
        :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for tensor operations. If `None`,
        :attr:`~sionna.phy.config.Config.device` is used.

    :input h_hat: [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_pilot_symbols], `torch.complex`.
        Channel estimates for the pilot-carrying resource elements.
    :input err_var: [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_pilot_symbols], `torch.float`.
        Channel estimation error variances for the pilot-carrying resource
        elements.

    :output h_hat: [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols, num_effective_subcarriers], `torch.complex`.
        Channel estimates across the entire resource grid for all
        transmitters and streams.
    :output err_var: Same shape as ``h_hat``, `torch.float`.
        Channel estimation error variances across the entire resource grid
        for all transmitters and streams.
    """

    def __init__(
        self,
        time_avg: bool = False,
        precision: Optional[Precision] = None,
        device: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(precision=precision, device=device, **kwargs)
        self._time_avg = time_avg

    @abstractmethod
    def __call__(
        self, h_hat: torch.Tensor, err_var: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pass


class NearestNeighborInterpolator(BaseChannelInterpolator):
    r"""Nearest-neighbor channel estimate interpolation on a resource grid.

    This class assigns to each element of an OFDM resource grid one of
    ``num_pilots`` provided channel estimates and error
    variances according to the nearest neighbor method. It is assumed
    that the measurements were taken at the nonzero positions of a
    :class:`~sionna.phy.ofdm.PilotPattern`.

    The figure below shows how four channel estimates are interpolated
    across a resource grid. Grey fields indicate measurement positions
    while the colored regions show which resource elements are assigned
    to the same measurement value.

    .. image:: ../../figures/nearest_neighbor_interpolation.png

    :param pilot_pattern: Used pilot pattern.
    :param time_avg: If `True`, measurements will be averaged across OFDM
        symbols (i.e., time). This is useful for channels that do not vary
        substantially over the duration of an OFDM frame. Defaults to `False`.
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`,
        :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for tensor operations. If `None`,
        :attr:`~sionna.phy.config.Config.device` is used.

    :input h_hat: [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_pilot_symbols], `torch.complex`.
        Channel estimates for the pilot-carrying resource elements.
    :input err_var: [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_pilot_symbols], `torch.float`.
        Channel estimation error variances for the pilot-carrying resource
        elements.

    :output h_hat: [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols, num_effective_subcarriers], `torch.complex`.
        Channel estimates across the entire resource grid for all
        transmitters and streams.
    :output err_var: Same shape as ``h_hat``, `torch.float`.
        Channel estimation error variances across the entire resource grid
        for all transmitters and streams.
    """

    def __init__(
        self,
        pilot_pattern,
        time_avg: bool = False,
        precision: Optional[Precision] = None,
        device: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(precision=precision, device=device, time_avg=time_avg, **kwargs)

        if pilot_pattern.num_pilot_symbols <= 0:
            raise ValueError("The pilot pattern cannot be empty")

        # Reshape mask to shape [-1, num_ofdm_symbols, num_effective_subcarriers]
        mask = pilot_pattern.mask.cpu().numpy()
        mask_shape = mask.shape  # Store to reconstruct the original shape
        mask = np.reshape(mask, [-1] + list(mask_shape[-2:]))

        # Reshape the pilots to shape [-1, num_pilot_symbols]
        pilots = pilot_pattern.pilots.cpu().numpy()
        pilots = np.reshape(pilots, [-1] + [pilots.shape[-1]])

        max_num_zero_pilots = np.max(np.sum(np.abs(pilots) == 0, -1))
        if max_num_zero_pilots >= pilots.shape[-1]:
            raise ValueError(
                "Each pilot sequence must have at least one nonzero entry"
            )

        # Compute gather indices for nearest neighbor interpolation
        gather_ind = np.zeros_like(mask, dtype=np.int64)
        for a in range(gather_ind.shape[0]):  # For each pilot pattern...
            i_p, j_p = np.where(mask[a])  # ...determine the pilot indices

            for i in range(mask_shape[-2]):  # Iterate over...
                for j in range(mask_shape[-1]):  # ... all resource elements
                    # Compute Manhattan distance to all pilot positions
                    d = np.abs(i - i_p) + np.abs(j - j_p)

                    # Set the distance at all pilot positions with zero energy
                    # equal to the maximum possible distance
                    d[np.abs(pilots[a]) == 0] = np.sum(mask_shape[-2:])

                    # Find the pilot index with the shortest distance
                    ind = np.argmin(d)

                    # Store it in the index tensor
                    gather_ind[a, i, j] = ind

        self.register_buffer(
            "_gather_ind",
            torch.tensor(
                np.reshape(gather_ind, mask_shape), dtype=torch.int64, device=self.device
            )
        )

        # Compute gather indices for frequency-domain interpolation (for time-averaging case)
        frequency_gather_ind = np.zeros_like(mask, dtype=np.int64)
        pilot_symbol_mask = np.zeros((*mask.shape[:-1], 1), dtype=np.bool_)

        for pattern_index in range(mask.shape[0]):
            pilot_symbol_indices, pilot_subcarrier_indices = np.where(
                mask[pattern_index]
            )
            active_pilot_indices = np.flatnonzero(
                np.abs(pilots[pattern_index]) > 0
            )

            for symbol_index in range(mask.shape[-2]):
                symbol_pilot_indices = active_pilot_indices[
                    pilot_symbol_indices[active_pilot_indices] == symbol_index
                ]
                if symbol_pilot_indices.size == 0:
                    continue

                pilot_symbol_mask[pattern_index, symbol_index, 0] = True

                subcarrier_distances = np.abs(
                    np.arange(mask.shape[-1])[:, None]
                    - pilot_subcarrier_indices[symbol_pilot_indices][None, :]
                )
                nearest_indices = np.argmin(subcarrier_distances, axis=-1)

                # These are indices into the compact pilot-estimate dimension.
                frequency_gather_ind[pattern_index, symbol_index] = (
                    symbol_pilot_indices[nearest_indices]
                )

        self.register_buffer(
            "_frequency_gather_ind",
            torch.tensor(
                np.reshape(frequency_gather_ind, mask_shape),
                dtype=torch.int64,
                device=self.device,
            ),
        )
        self.register_buffer(
            "_pilot_symbol_mask",
            torch.tensor(
                np.reshape(
                    pilot_symbol_mask,
                    (*mask_shape[:-1], 1),
                ),
                dtype=torch.bool,
                device=self.device,
            ),
        )

    def _interpolate(self, inputs: torch.Tensor) -> torch.Tensor:
        """Interpolate using nearest neighbor method."""
        # inputs: [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx,
        #          num_pilots]

        # Move batch dimensions to end
        # [num_tx, num_streams_per_tx, num_pilots, batch_size, num_rx, num_rx_ant]
        inputs = inputs.permute(3, 4, 5, 0, 1, 2)

        # Gather using indices
        # gather_ind: [num_tx, num_streams_per_tx, num_ofdm_symbols,
        #              num_effective_subcarriers]
        if self._time_avg:
            gather_ind = self._frequency_gather_ind
        else:
            gather_ind = self._gather_ind

        # Expand gather_ind to match batch dimensions
        # [num_tx, num_streams_per_tx, num_ofdm_symbols,
        #  num_effective_subcarriers, batch_size, num_rx, num_rx_ant]
        gather_ind = gather_ind.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        gather_ind = gather_ind.expand(
            -1, -1, -1, -1, inputs.shape[3], inputs.shape[4], inputs.shape[5]
        )

        # inputs: [num_tx, num_streams_per_tx, num_pilots, batch_size, num_rx,
        #          num_rx_ant]
        # Expand inputs to match output shape
        inputs = (
            inputs.unsqueeze(2)
            .unsqueeze(3)
            .expand(-1, -1, gather_ind.shape[2], gather_ind.shape[3], -1, -1, -1, -1)
        )

        # Gather along the pilots dimension
        outputs = torch.gather(inputs, 4, gather_ind.unsqueeze(4)).squeeze(4)

        # Move batch dimensions back to front
        # [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx,
        #  num_ofdm_symbols, num_effective_subcarriers]
        outputs = outputs.permute(4, 5, 6, 0, 1, 2, 3)

        # Time-average the channel estimates if enabled
        if self._time_avg:
            pilot_symbol_mask = self._pilot_symbol_mask[
                None, None, None, ...
            ]

            outputs = outputs * pilot_symbol_mask
            num_pilot_symbols = pilot_symbol_mask.sum(
                dim=-2,
                keepdim=True,
            ).to(outputs.dtype)

            outputs = outputs.sum(dim=-2, keepdim=True)
            outputs = outputs / num_pilot_symbols
            outputs = outputs.expand(
                *outputs.shape[:-2],
                self._pilot_symbol_mask.shape[-2],
                outputs.shape[-1],
            )

        return outputs

    def __call__(
        self, h_hat: torch.Tensor, err_var: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h_hat = self._interpolate(h_hat)
        err_var = self._interpolate(err_var)
        err_var = torch.clamp_min(err_var, 0.0)
        return h_hat, err_var


class LinearInterpolator(BaseChannelInterpolator):
    r"""Linear channel estimate interpolation on a resource grid.

    This class computes for each element of an OFDM resource grid
    a channel estimate based on ``num_pilots`` provided channel estimates and
    error variances through linear interpolation.
    It is assumed that the measurements were taken at the nonzero positions
    of a :class:`~sionna.phy.ofdm.PilotPattern`.

    The interpolation is done first across sub-carriers and then
    across OFDM symbols.

    .. image:: ../../figures/linear_interpolation.png

    :param pilot_pattern: Used pilot pattern
    :param time_avg: If `True`, measurements will be averaged across OFDM
        symbols (i.e., time). This is useful for channels that do not vary
        substantially over the duration of an OFDM frame. Defaults to `False`.
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`,
        :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for tensor operations. If `None`,
        :attr:`~sionna.phy.config.Config.device` is used.

    :input h_hat: [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_pilot_symbols], `torch.complex`.
        Channel estimates for the pilot-carrying resource elements.
    :input err_var: [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_pilot_symbols], `torch.float`.
        Channel estimation error variances for the pilot-carrying resource
        elements.

    :output h_hat: [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols, num_effective_subcarriers], `torch.complex`.
        Channel estimates across the entire resource grid for all
        transmitters and streams.
    :output err_var: Same shape as ``h_hat``, `torch.float`.
        Channel estimation error variances across the entire resource grid
        for all transmitters and streams.
    """

    def __init__(
        self,
        pilot_pattern,
        time_avg: bool = False,
        precision: Optional[Precision] = None,
        device: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(precision=precision, device=device, time_avg=time_avg, **kwargs)

        if pilot_pattern.num_pilot_symbols <= 0:
            raise ValueError("The pilot pattern cannot be empty")

        # Reshape mask to shape [-1, num_ofdm_symbols, num_effective_subcarriers]
        mask = pilot_pattern.mask.cpu().numpy()
        mask_shape = mask.shape  # Store to reconstruct the original shape
        mask = np.reshape(mask, [-1] + list(mask_shape[-2:]))

        # Reshape the pilots to shape [-1, num_pilot_symbols]
        pilots = pilot_pattern.pilots.cpu().numpy()
        pilots = np.reshape(pilots, [-1] + [pilots.shape[-1]])

        max_num_zero_pilots = np.max(np.sum(np.abs(pilots) == 0, -1))
        if max_num_zero_pilots >= pilots.shape[-1]:
            raise ValueError(
                "Each pilot sequence must have at least one nonzero entry"
            )

        # Create actual pilot patterns for each stream over the resource grid
        z = np.zeros_like(mask, dtype=pilots.dtype)
        for a in range(z.shape[0]):
            z[a][np.where(mask[a])] = pilots[a]

        ##
        # Frequency-domain interpolation
        ##
        # Register as buffer for CUDAGraph compatibility
        self.register_buffer(
            "_x_freq",
            torch.arange(0, mask.shape[-1], dtype=self.dtype, device=self.device),
        )

        x_0_freq = np.zeros_like(mask, np.int64)
        x_1_freq = np.zeros_like(mask, np.int64)

        # Set REs of OFDM symbols without any pilot equal to -1 (dummy value)
        x_0_freq[np.sum(np.abs(z), axis=-1) == 0] = -1
        x_1_freq[np.sum(np.abs(z), axis=-1) == 0] = -1

        y_0_freq_ind = np.copy(x_0_freq)  # Indices used to gather estimates
        y_1_freq_ind = np.copy(x_1_freq)  # Indices used to gather estimates

        # For each stream
        for a in range(z.shape[0]):
            pilot_count = 0  # Counts the number of non-zero pilots

            # Indices of non-zero pilots within the pilots vector
            pilot_ind = np.where(np.abs(pilots[a]))[0]

            # Go through all OFDM symbols
            for i in range(x_0_freq.shape[1]):
                # Indices of non-zero pilots within the OFDM symbol
                pilot_ind_ofdm = np.where(np.abs(z[a][i]))[0]

                # If OFDM symbol contains only one non-zero pilot
                if len(pilot_ind_ofdm) == 1:
                    x_0_freq[a][i] = pilot_ind_ofdm[0]
                    x_1_freq[a][i] = pilot_ind_ofdm[0]
                    y_0_freq_ind[a, i] = pilot_ind[pilot_count]
                    y_1_freq_ind[a, i] = pilot_ind[pilot_count]

                # If OFDM symbol contains two or more pilots
                elif len(pilot_ind_ofdm) >= 2:
                    x0 = 0
                    x1 = 1

                    for j in range(x_0_freq.shape[2]):
                        x_0_freq[a, i, j] = pilot_ind_ofdm[x0]
                        x_1_freq[a, i, j] = pilot_ind_ofdm[x1]
                        y_0_freq_ind[a, i, j] = pilot_ind[pilot_count + x0]
                        y_1_freq_ind[a, i, j] = pilot_ind[pilot_count + x1]
                        if j == pilot_ind_ofdm[x1] and x1 < len(pilot_ind_ofdm) - 1:
                            x0 = x1
                            x1 += 1

                pilot_count += len(pilot_ind_ofdm)

        x_0_freq = np.reshape(x_0_freq, mask_shape)
        x_1_freq = np.reshape(x_1_freq, mask_shape)
        # Register as buffers for CUDAGraph compatibility
        self.register_buffer(
            "_x_0_freq",
            torch.tensor(x_0_freq, dtype=self.dtype, device=self.device),
        )
        self.register_buffer(
            "_x_1_freq",
            torch.tensor(x_1_freq, dtype=self.dtype, device=self.device),
        )

        # We add +1 to shift all indices as the input will be padded
        # at the beginning with 0
        self.register_buffer(
            "_y_0_freq_ind",
            torch.tensor(
                np.reshape(y_0_freq_ind, mask_shape) + 1,
                dtype=torch.int64,
                device=self.device,
            ),
        )
        self.register_buffer(
            "_y_1_freq_ind",
            torch.tensor(
                np.reshape(y_1_freq_ind, mask_shape) + 1,
                dtype=torch.int64,
                device=self.device,
            ),
        )

        ##
        # Time-domain interpolation
        ##
        # Register as buffer for CUDAGraph compatibility
        self.register_buffer(
            "_x_time",
            torch.arange(
                0, mask.shape[-2], dtype=self.dtype, device=self.device
            ).unsqueeze(-1),
        )

        y_0_time_ind = np.zeros(z.shape[:2], np.int64)  # Gather indices
        y_1_time_ind = np.zeros(z.shape[:2], np.int64)  # Gather indices

        # For each stream
        for a in range(z.shape[0]):
            # Indices of OFDM symbols for which channel estimates were computed
            ofdm_ind = np.where(np.sum(np.abs(z[a]), axis=-1))[0]

            # Only one OFDM symbol with pilots
            if len(ofdm_ind) == 1:
                y_0_time_ind[a] = ofdm_ind[0]
                y_1_time_ind[a] = ofdm_ind[0]

            # Two or more OFDM symbols with pilots
            elif len(ofdm_ind) >= 2:
                x0 = 0
                x1 = 1
                for i in range(z.shape[1]):
                    y_0_time_ind[a, i] = ofdm_ind[x0]
                    y_1_time_ind[a, i] = ofdm_ind[x1]
                    if i == ofdm_ind[x1] and x1 < len(ofdm_ind) - 1:
                        x0 = x1
                        x1 += 1

        # Register as buffers for CUDAGraph compatibility
        self.register_buffer(
            "_y_0_time_ind",
            torch.tensor(
                np.reshape(y_0_time_ind, mask_shape[:-1]),
                dtype=torch.int64,
                device=self.device,
            ),
        )
        self.register_buffer(
            "_y_1_time_ind",
            torch.tensor(
                np.reshape(y_1_time_ind, mask_shape[:-1]),
                dtype=torch.int64,
                device=self.device,
            ),
        )

        self.register_buffer(
            "_x_0_time",
            self._y_0_time_ind.unsqueeze(-1).to(dtype=self.dtype),
        )
        self.register_buffer(
            "_x_1_time",
            self._y_1_time_ind.unsqueeze(-1).to(dtype=self.dtype),
        )

        # Number of OFDM symbols carrying at least one pilot
        n = np.sum(np.abs(np.reshape(z, mask_shape)), axis=-1, keepdims=True)
        n = np.sum(n > 0, axis=-2, keepdims=True)
        self.register_buffer(
            "_num_pilot_ofdm_symbols",
            torch.tensor(n, dtype=self.dtype, device=self.device),
        )

    def _interpolate_1d(
        self,
        inputs: torch.Tensor,
        x: torch.Tensor,
        x0: torch.Tensor,
        x1: torch.Tensor,
        y0_ind: torch.Tensor,
        y1_ind: torch.Tensor,
    ) -> torch.Tensor:
        """Perform 1D linear interpolation."""
        # inputs: [num_tx, num_streams_per_tx, 1+num_pilots, batch_size, num_rx,
        #          num_rx_ant]

        # Expand indices to match batch dimensions
        batch_dims = inputs.shape[3:]

        # y0_ind, y1_ind: [num_tx, num_streams_per_tx, num_ofdm_symbols,
        #                  num_effective_subcarriers]
        y0_ind_expanded = y0_ind
        y1_ind_expanded = y1_ind

        # Add batch dimensions
        for _ in batch_dims:
            y0_ind_expanded = y0_ind_expanded.unsqueeze(-1)
            y1_ind_expanded = y1_ind_expanded.unsqueeze(-1)

        y0_ind_expanded = y0_ind_expanded.expand(*y0_ind.shape, *batch_dims)
        y1_ind_expanded = y1_ind_expanded.expand(*y1_ind.shape, *batch_dims)

        # Expand inputs for gathering
        # inputs needs to match shape for gather
        inputs_expanded = (
            inputs.unsqueeze(2)
            .unsqueeze(3)
            .expand(-1, -1, y0_ind.shape[2], y0_ind.shape[3], -1, -1, -1, -1)
        )

        # Gather y0 and y1
        y0 = torch.gather(inputs_expanded, 4, y0_ind_expanded.unsqueeze(4)).squeeze(4)
        y1 = torch.gather(inputs_expanded, 4, y1_ind_expanded.unsqueeze(4)).squeeze(4)

        # Move batch dimensions back
        # [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx,
        #  num_ofdm_symbols, num_effective_subcarriers]
        y0 = y0.permute(4, 5, 6, 0, 1, 2, 3).contiguous()
        y1 = y1.permute(4, 5, 6, 0, 1, 2, 3).contiguous()

        # Compute linear interpolation
        # Expand x, x0, x1 to match output shape
        x = expand_to_rank(x, y0.dim(), 0)
        x0 = expand_to_rank(x0, y0.dim(), 0)
        x1 = expand_to_rank(x1, y0.dim(), 0)

        slope = torch.where(
            x1 != x0,
            (y1 - y0) / (x1 - x0),
            torch.zeros_like(y0),
        )

        return ((x - x0) * slope + y0).contiguous()

    def _interpolate(self, inputs: torch.Tensor) -> torch.Tensor:
        """Interpolate channel estimates across the resource grid."""
        # Pad the inputs with a leading 0
        pad = (1, 0)  # Pad last dimension
        inputs = torch.nn.functional.pad(inputs, pad)

        # Move batch dimensions to end
        # [num_tx, num_streams_per_tx, 1+num_pilots, batch_size, num_rx, num_rx_ant]
        inputs = inputs.permute(3, 4, 5, 0, 1, 2).contiguous()

        # Frequency-domain interpolation
        h_hat_freq = self._interpolate_1d(
            inputs,
            self._x_freq,
            self._x_0_freq,
            self._x_1_freq,
            self._y_0_freq_ind,
            self._y_1_freq_ind,
        )

        # Time-domain averaging (optional)
        if self._time_avg:
            num_ofdm_symbols = h_hat_freq.shape[-2]
            h_hat_freq = h_hat_freq.sum(dim=-2, keepdim=True)
            n = self._num_pilot_ofdm_symbols
            n = expand_to_rank(n, h_hat_freq.dim(), 0)
            h_hat_freq = h_hat_freq / n
            h_hat_freq = h_hat_freq.repeat(1, 1, 1, 1, 1, num_ofdm_symbols, 1)

        # Time-domain interpolation
        # Transpose: [num_tx, num_streams_per_tx, num_ofdm_symbols,
        #             num_effective_subcarriers, batch_size, num_rx, num_rx_ant]
        h_hat_time = h_hat_freq.permute(3, 4, 5, 6, 0, 1, 2).contiguous()

        # Expand for time interpolation gathering
        y_0_time_ind = self._y_0_time_ind
        y_1_time_ind = self._y_1_time_ind

        # y_0_time_ind shape: [num_tx, num_streams_per_tx, num_ofdm_symbols]
        # We need to expand to match h_hat_time shape on dims 3-6
        trailing_dims = h_hat_time.shape[3:]  # [num_eff_subcarriers, batch, rx, rx_ant]

        y_0_time_expanded = y_0_time_ind
        y_1_time_expanded = y_1_time_ind

        # Add trailing dimensions
        for _ in trailing_dims:
            y_0_time_expanded = y_0_time_expanded.unsqueeze(-1)
            y_1_time_expanded = y_1_time_expanded.unsqueeze(-1)

        # Expand to match h_hat_time shape
        y_0_time_expanded = y_0_time_expanded.expand(
            *y_0_time_ind.shape, *trailing_dims
        )
        y_1_time_expanded = y_1_time_expanded.expand(
            *y_1_time_ind.shape, *trailing_dims
        )

        # Gather y0 and y1 for time interpolation
        y0 = torch.gather(h_hat_time, 2, y_0_time_expanded)
        y1 = torch.gather(h_hat_time, 2, y_1_time_expanded)

        # Move back to standard order for output
        # [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx,
        #  num_ofdm_symbols, num_effective_subcarriers]
        y0 = y0.permute(4, 5, 6, 0, 1, 2, 3).contiguous()
        y1 = y1.permute(4, 5, 6, 0, 1, 2, 3).contiguous()

        # Linear interpolation in time
        x = self._x_time
        x0 = self._x_0_time
        x1 = self._x_1_time

        x = expand_to_rank(x, y0.dim(), 0)
        x0 = expand_to_rank(x0, y0.dim(), 0)
        x1 = expand_to_rank(x1, y0.dim(), 0)

        slope = torch.where(
            x1 != x0,
            (y1 - y0) / (x1 - x0),
            torch.zeros_like(y0),
        )

        h_hat_time = ((x - x0) * slope + y0).contiguous()

        return h_hat_time

    def __call__(
        self, h_hat: torch.Tensor, err_var: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h_hat = self._interpolate(h_hat)

        # The interpolator requires complex-valued inputs
        err_var_complex = torch.complex(err_var, torch.zeros_like(err_var))
        err_var = self._interpolate(err_var_complex)
        err_var = torch.clamp_min(err_var.real, 0.0)
        return h_hat, err_var


class _LMMSEEstimator1D(Object):
    r"""LMMSE estimation with interpolation across the inner dimension of the input.
    This estimator is used internally by :class:`LMMSEChannelEstimator`.

    The two inner dimensions of the input ``h_hat`` form a matrix
    :math:`\hat{\mathbf{H}} \in \mathbb{C}^{N \times M}`.
    LMMSE interpolation is performed across the inner dimension as follows:

    .. math::
        \tilde{\mathbf{h}}_n = \mathbf{A}_n \hat{\mathbf{h}}_n

    where :math:`1 \leq n \leq N` and :math:`\hat{\mathbf{h}}_n` is
    the :math:`n^{\text{th}}` (transposed) row of :math:`\hat{\mathbf{H}}`.
    :math:`\mathbf{A}_n` is the :math:`M \times M` interpolation LMMSE matrix:

    .. math::
        \mathbf{A}_n = \mathbf{R} \mathbf{\Pi}_n \left( \mathbf{\Pi}_n^\intercal \mathbf{R} \mathbf{\Pi}_n + \tilde{\mathbf{\Sigma}}_n \right)^{-1} \mathbf{\Pi}_n^\intercal.

    where :math:`\mathbf{R}` is the :math:`M \times M` covariance matrix across
    the inner dimension of the quantity which is estimated,
    :math:`\mathbf{\Pi}_n` the :math:`M \times K_n` matrix that spreads
    :math:`K_n` values to a vector of size :math:`M` according to the
    ``pilot_mask`` for the :math:`n^{\text{th}}` row,
    and :math:`\tilde{\mathbf{\Sigma}}_n \in \mathbb{R}^{K_n \times K_n}` is
    the regularized channel estimation error covariance.
    The :math:`i^{\text{th}}` diagonal element of
    :math:`\tilde{\mathbf{\Sigma}}_n` is such that:

    .. math::

        \left[ \tilde{\mathbf{\Sigma}}_n \right]_{i,i} = \max \left\{ \left[ \mathbf{\Sigma}_n \right]_{i,i},\; 0 \right\}

    built from ``err_var`` and assumed to be diagonal.

    The returned channel estimates are

    .. math::
        \begin{bmatrix}
            {\tilde{\mathbf{h}}_1}^\intercal\\
            \vdots\\
            {\tilde{\mathbf{h}}_N}^\intercal
        \end{bmatrix}.

    The returned channel estimation error variances are the diagonal
    coefficients of

    .. math::
        \text{diag} \left( \mathbf{R} - \mathbf{A}_n \mathbf{\Xi}_n \mathbf{R} \right), 1 \leq n \leq N

    where :math:`\mathbf{\Xi}_n` is the diagonal matrix of size
    :math:`M \times M` that zeros the columns corresponding to rows not
    carrying any pilots.
    Note that interpolation is not performed for rows not carrying any pilots.

    **Remark**: The interpolation matrix differs across rows as different
    rows may carry pilots on different elements and/or have different
    estimation error variances.

    :param pilot_mask: Mask indicating the allocation of resource elements.
        0: Data, 1: Pilot, 2: Not used.
    :param cov_mat: Covariance matrix of the channel across the inner
        dimension
    :param last_step: Set to `True` if this is the last interpolation step.
        Otherwise, set to `False`.
        If `True`, the output is scaled to ensure its variance is as expected
        by the following interpolation step.

    :input h_hat: [batch_size, num_rx, num_rx_ant, num_tx, :math:`N`, :math:`M`], `torch.complex`.
        Channel estimates.
    :input err_var: [batch_size, num_rx, num_rx_ant, num_tx, :math:`N`, :math:`M`], `torch.float`.
        Channel estimation error variances.

    :output h_hat: [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, :math:`N`, :math:`M`], `torch.complex`.
        Channel estimates interpolated across the inner dimension.
    :output err_var: Same shape as ``h_hat``, `torch.float`.
        The channel estimation error variances of the interpolated channel
        estimates.
    """

    def __init__(
        self,
        pilot_mask: np.ndarray,
        cov_mat: torch.Tensor,
        last_step: bool,
        precision: Optional[Precision] = None,
        device: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(precision=precision, device=device, **kwargs)

        # Register as buffer for CUDAGraph compatibility
        self.register_buffer(
            "_rzero", torch.tensor(0.0, dtype=self.dtype, device=self.device)
        )

        # Size of inner and outer dimensions
        inner_dim_size = pilot_mask.shape[-1]
        outer_dim_size = pilot_mask.shape[-2]
        self._inner_dim_size = inner_dim_size
        self._outer_dim_size = outer_dim_size

        # Register cov_mat as buffer for CUDAGraph compatibility
        if not cov_mat.is_complex():
            raise ValueError("`cov_mat` must be complex")
        self.register_buffer("_cov_mat", cov_mat.to(dtype=self.cdtype, device=self.device))
        self._last_step = last_step

        # Extract pilot locations
        num_tx = pilot_mask.shape[0]
        num_streams_per_tx = pilot_mask.shape[1]

        # List of indices of pilots in the inner dimension
        pilot_indices = []
        max_num_pil = 0
        add_err_var_indices = np.zeros(
            [num_tx, num_streams_per_tx, outer_dim_size, inner_dim_size, 5], int
        )
        # Pre-compute list of valid pilot positions for compile-friendly iteration
        # Each entry is (tx, st, oi, ii, pil_idx) as Python ints
        valid_pilot_positions = []

        for tx in range(num_tx):
            pilot_indices.append([])
            for st in range(num_streams_per_tx):
                pilot_indices[-1].append([])
                for oi in range(outer_dim_size):
                    pilot_indices[-1][-1].append([])
                    num_pil = 0
                    for ii in range(inner_dim_size):
                        if pilot_mask[tx, st, oi, ii] == 0:
                            continue
                        if pilot_mask[tx, st, oi, ii] == 1:
                            pilot_indices[tx][st][oi].append(ii)
                            indices = [tx, st, oi, num_pil, num_pil]
                            add_err_var_indices[tx, st, oi, ii] = indices
                            # Store valid position as Python tuple
                            valid_pilot_positions.append((tx, st, oi, ii, num_pil))
                            num_pil += 1
                    max_num_pil = max(max_num_pil, num_pil)

        # Store as Python list (not tensor) for compile-friendly iteration
        self._valid_pilot_positions = valid_pilot_positions

        # Register as buffer for CUDAGraph compatibility
        self.register_buffer(
            "_add_err_var_indices",
            torch.tensor(add_err_var_indices, dtype=torch.int64, device=self.device),
        )

        # Build pilot covariance matrix
        cov_mat_np = cov_mat.cpu().numpy()
        pil_cov_mat = np.zeros(
            [num_tx, num_streams_per_tx, outer_dim_size, max_num_pil, max_num_pil],
            complex,
        )
        for tx, st, oi in itertools.product(
            range(num_tx), range(num_streams_per_tx), range(outer_dim_size)
        ):
            pil_ind = pilot_indices[tx][st][oi]
            num_pil = len(pil_ind)
            if num_pil > 0:
                tmp = np.take(cov_mat_np, pil_ind, axis=0)
                pil_cov_mat_ = np.take(tmp, pil_ind, axis=1)
                pil_cov_mat[tx, st, oi, :num_pil, :num_pil] = pil_cov_mat_
        # Register as buffer for CUDAGraph compatibility
        self.register_buffer(
            "_pil_cov_mat",
            torch.tensor(pil_cov_mat, dtype=self.cdtype, device=self.device),
        )

        # Pre-compute B matrix
        b_mat = np.zeros(
            [num_tx, num_streams_per_tx, outer_dim_size, max_num_pil, inner_dim_size],
            complex,
        )
        for tx, st, oi in itertools.product(
            range(num_tx), range(num_streams_per_tx), range(outer_dim_size)
        ):
            pil_ind = pilot_indices[tx][st][oi]
            num_pil = len(pil_ind)
            if num_pil > 0:
                b_mat_ = np.take(cov_mat_np, pil_ind, axis=0)
                b_mat[tx, st, oi, :num_pil, :] = b_mat_
        # Register as buffer for CUDAGraph compatibility
        self.register_buffer(
            "_b_mat", torch.tensor(b_mat, dtype=self.cdtype, device=self.device)
        )

        # Indices for scatter
        pil_loc = np.zeros(
            [
                num_tx,
                num_streams_per_tx,
                outer_dim_size,
                inner_dim_size,
                max_num_pil,
                5,
            ],
            dtype=int,
        )
        for tx, st, oi, p, ii in itertools.product(
            range(num_tx),
            range(num_streams_per_tx),
            range(outer_dim_size),
            range(max_num_pil),
            range(inner_dim_size),
        ):
            if p >= len(pilot_indices[tx][st][oi]):
                pil_loc[tx, st, oi, ii, p] = [
                    tx,
                    st,
                    oi,
                    inner_dim_size,
                    inner_dim_size,
                ]
            else:
                pil_loc[tx, st, oi, ii, p] = [
                    tx,
                    st,
                    oi,
                    ii,
                    pilot_indices[tx][st][oi][p],
                ]
        # Register as buffer for CUDAGraph compatibility
        # Extract only the row/col indices (indices 3 and 4) for compile-friendly scatter
        # Shape: [num_tx, num_streams_per_tx, outer_dim_size, inner_dim_size, max_num_pil]
        self.register_buffer(
            "_pil_loc_row",
            torch.tensor(pil_loc[..., 3], dtype=torch.int64, device=self.device),
        )
        self.register_buffer(
            "_pil_loc_col",
            torch.tensor(pil_loc[..., 4], dtype=torch.int64, device=self.device),
        )

        # Error variance matrix
        err_var_mat = np.zeros(
            [
                num_tx,
                num_streams_per_tx,
                outer_dim_size,
                inner_dim_size,
                inner_dim_size,
            ],
            complex,
        )
        for tx, st, oi in itertools.product(
            range(num_tx), range(num_streams_per_tx), range(outer_dim_size)
        ):
            pil_ind = pilot_indices[tx][st][oi]
            mask = np.zeros([inner_dim_size], complex)
            mask[pil_ind] = 1.0
            mask = np.expand_dims(mask, axis=1)
            err_var_mat[tx, st, oi] = cov_mat_np * mask
        # Register as buffer for CUDAGraph compatibility
        self.register_buffer(
            "_err_var_mat",
            torch.tensor(err_var_mat, dtype=self.cdtype, device=self.device),
        )

    @torch.compiler.disable  # Complex linear algebra (Cholesky, solve_triangular) is slower when compiled
    def __call__(
        self, h_hat: torch.Tensor, err_var: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = h_hat.shape[0]
        num_rx = h_hat.shape[1]
        num_rx_ant = h_hat.shape[2]
        num_tx = h_hat.shape[3]
        num_tx_stream = h_hat.shape[4]
        outer_dim_size = self._outer_dim_size
        inner_dim_size = self._inner_dim_size

        pil_loc_row = self._pil_loc_row
        pil_loc_col = self._pil_loc_col
        rzero = self._rzero

        #####################################
        # Compute the interpolation matrix
        #####################################

        # Compute A matrix (covariance + error variance)
        # [num_tx, num_streams_per_tx, outer_dim_size, max_num_pil, max_num_pil]
        pil_cov_mat = self._pil_cov_mat
        # [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx,
        #  outer_dim_size, max_num_pil, max_num_pil]
        pil_cov_mat = expand_to_rank(pil_cov_mat, 8, 0)
        pil_cov_mat = pil_cov_mat.expand(
            batch_size, num_rx, num_rx_ant, -1, -1, -1, -1, -1
        )

        # Add error variance to diagonal using scatter
        # Transpose for scatter operation
        # [num_tx, num_streams_per_tx, outer_dim_size, max_num_pil, max_num_pil,
        #  batch_size, num_rx, num_rx_ant]
        pil_cov_mat_ = pil_cov_mat.permute(3, 4, 5, 6, 7, 0, 1, 2).clone()
        err_var_c = err_var.to(self.cdtype)
        # [num_tx, num_streams_per_tx, outer_dim_size, inner_dim_size,
        #  batch_size, num_rx, num_rx_ant]
        err_var_ = err_var_c.permute(3, 4, 5, 6, 0, 1, 2)

        # Add error variance to diagonal using pre-computed valid positions
        # (avoids data-dependent branching for torch.compile compatibility)
        for tx, st, oi, ii, pil_idx in self._valid_pilot_positions:
            pil_cov_mat_[tx, st, oi, pil_idx, pil_idx] += err_var_[tx, st, oi, ii]

        # [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx,
        #  outer_dim_size, max_num_pil, max_num_pil]
        a_mat = pil_cov_mat_.permute(5, 6, 7, 0, 1, 2, 3, 4)

        # Compute B matrix
        # [num_tx, num_streams_per_tx, outer_dim_size, max_num_pil, inner_dim_size]
        b_mat = self._b_mat
        # [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx,
        #  outer_dim_size, max_num_pil, inner_dim_size]
        b_mat = expand_to_rank(b_mat, 8, 0)
        b_mat = b_mat.expand(batch_size, num_rx, num_rx_ant, -1, -1, -1, -1, -1)

        # Solve least squares: a_mat @ ext_mat = b_mat
        # [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx,
        #  outer_dim_size, max_num_pil, inner_dim_size]
        # Use direct Cholesky solve for better numerical stability (avoids squaring
        # condition number like matrix_pinv does with Gram matrix A^H @ A)

        # Add precision-dependent regularization for numerical stability at high SNR
        # (when err_var becomes very small, the matrix can become ill-conditioned)
        eps = torch.finfo(self.dtype).eps
        rcond = eps * a_mat.shape[-1]
        diag_mean = torch.diagonal(a_mat, dim1=-2, dim2=-1).abs().mean(dim=-1, keepdim=True).unsqueeze(-1)
        reg = rcond * diag_mean * torch.eye(a_mat.shape[-1], dtype=a_mat.dtype, device=self.device)
        a_mat_reg = a_mat + reg

        # Cholesky solve: a_mat @ X = b_mat
        chol, _ = torch.linalg.cholesky_ex(a_mat_reg, check_errors=False)
        y = torch.linalg.solve_triangular(chol, b_mat, upper=False)
        ext_mat = torch.linalg.solve_triangular(chol.mH, y, upper=True)

        # Conjugate transpose
        # [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx,
        #  outer_dim_size, inner_dim_size, max_num_pil]
        ext_mat = ext_mat.mH

        # Scatter to expand columns from max_num_pil to inner_dim_size
        # Using the pil_loc indices to place values correctly
        # [num_tx, num_streams_per_tx, outer_dim_size, inner_dim_size, max_num_pil,
        #  batch_size, num_rx, num_rx_ant]
        ext_mat_t = ext_mat.permute(3, 4, 5, 6, 7, 0, 1, 2)

        # Create output tensor with extra padding row/column
        ext_mat_full = torch.zeros(
            num_tx,
            num_tx_stream,
            outer_dim_size,
            inner_dim_size + 1,
            inner_dim_size + 1,
            batch_size,
            num_rx,
            num_rx_ant,
            dtype=self.cdtype,
            device=self.device,
        )

        # Scatter values according to pilot locations using fully vectorized advanced indexing
        # This avoids Python loops which cause slow compilation and execution

        # Create broadcast-compatible index tensors for the first 3 dimensions
        tx_idx = torch.arange(num_tx, device=self.device)[:, None, None, None, None]
        st_idx = torch.arange(num_tx_stream, device=self.device)[None, :, None, None, None]
        oi_idx = torch.arange(outer_dim_size, device=self.device)[None, None, :, None, None]

        # Single vectorized scatter using advanced indexing
        # All indices broadcast to [num_tx, num_tx_stream, outer, inner, max_pil]
        # The trailing dims [batch, num_rx, num_rx_ant] are preserved automatically
        ext_mat_full[tx_idx, st_idx, oi_idx, pil_loc_row, pil_loc_col] = ext_mat_t

        # Remove padding and transpose back
        # [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx,
        #  outer_dim_size, inner_dim_size, inner_dim_size]
        ext_mat = ext_mat_full[:, :, :, :inner_dim_size, :inner_dim_size, :, :, :]
        ext_mat = ext_mat.permute(5, 6, 7, 0, 1, 2, 3, 4)

        ################################################
        # Apply interpolation over the inner dimension
        ################################################

        # [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx,
        #  outer_dim_size, inner_dim_size]
        h_hat_out = (ext_mat @ h_hat.unsqueeze(-1)).squeeze(-1)

        ##############################
        # Compute the error variances
        ##############################

        # [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx,
        #  outer_dim_size, inner_dim_size]
        cov_mat = self._cov_mat
        cov_mat = expand_to_rank(cov_mat, 8, 0)
        err_var_out = torch.diagonal(cov_mat, dim1=-2, dim2=-1)
        err_var_mat = self._err_var_mat
        err_var_mat = expand_to_rank(err_var_mat, 8, 0)
        # Transpose (NOT conjugate transpose) to swap last two dimensions
        err_var_mat_t = err_var_mat.transpose(-1, -2)
        err_var_out = err_var_out - (ext_mat * err_var_mat_t).sum(dim=-1)
        err_var_out = err_var_out.real
        err_var_out = torch.maximum(err_var_out, rzero)

        #####################################
        # If this is *not* the last
        # interpolation step, scales the
        # input `h_hat` to ensure
        # it has the variance expected by the
        # next interpolation step.
        #
        # The error variance also `err_var`
        # is updated accordingly.
        #####################################
        if not self._last_step:
            # Conjugate transpose of LMMSE matrix
            # [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx,
            #  outer_dim_size, inner_dim_size, inner_dim_size]
            ext_mat_h = ext_mat.transpose(-1, -2).conj()

            # First part of the estimate covariance
            # [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx,
            #  outer_dim_size, inner_dim_size, inner_dim_size]
            h_hat_var_1 = cov_mat @ ext_mat_h
            h_hat_var_1 = h_hat_var_1.transpose(-1, -2)
            # [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx,
            #  outer_dim_size, inner_dim_size]
            h_hat_var_1 = (ext_mat * h_hat_var_1).sum(dim=-1)

            # Second part of the estimate covariance
            # [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx,
            #  outer_dim_size, inner_dim_size]
            err_var_old_c = err_var.to(self.cdtype).unsqueeze(-1)
            h_hat_var_2 = err_var_old_c * ext_mat_h
            h_hat_var_2 = h_hat_var_2.transpose(-1, -2)
            h_hat_var_2 = (ext_mat * h_hat_var_2).sum(dim=-1)

            # Variance of h_hat
            # [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx,
            #  outer_dim_size, inner_dim_size]
            h_hat_var = h_hat_var_1 + h_hat_var_2

            # Scaling factor
            # [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx,
            #  outer_dim_size, inner_dim_size]
            err_var_c = err_var_out.to(self.cdtype)
            h_var = torch.diagonal(cov_mat, dim1=-2, dim2=-1)
            denom = h_hat_var + h_var - err_var_c
            # Use divide_no_nan equivalent
            s = torch.where(
                denom.abs() > 1e-12, 2.0 * h_var / denom, torch.zeros_like(denom)
            )

            # Apply scaling to estimate
            # [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx,
            #  outer_dim_size, inner_dim_size]
            h_hat_out = s * h_hat_out

            # Updated variance (using complex arithmetic, then take real part)
            # [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx,
            #  outer_dim_size, inner_dim_size]
            err_var_out_c = (
                s * (s - 1.0) * h_hat_var + (1.0 - s) * h_var + s * err_var_c
            )
            err_var_out = err_var_out_c.real
            err_var_out = torch.maximum(err_var_out, rzero)

        return h_hat_out, err_var_out


class _SpatialLMMSEEstimator1D(Object):
    r"""LMMSE estimation as used in the spatial dimension of :class:`LMMSEChannelEstimator`.
    The main difference is that this estimator does not use interpolation.

    The returned estimate :math:`\hat{\mathbf{h}}` is computed as

    .. math::

        \hat{\mathbf{h}} = \mathbf{A} \mathbf{y}

    where

    .. math::

        \mathbf{A} = \mathbf{R} \left( \mathbf{R} + \text{diag}(\text{err_var})\right)^{-1}

    where :math:`\text{err_var}` is the :math:`M \times M` diagonal matrix of input error variances.
    The estimation error is

    .. math::

        \tilde{h} = \mathbf{h} - \hat{\mathbf{h}},

    and the returned error variances are

    .. math::

             \sigma^2_i = \mathbb{E}\left[\tilde{h}_i \tilde{h}_i^\star \right], 0 \leq i \leq M-1

    and are the diagonal elements of

    .. math::

        \mathbb{E}\left[\mathbf{\tilde{h}} \mathbf{\tilde{h}}^{\mathsf{H}} \right] = \mathbf{R} - \mathbf{A}\mathbf{R}.

    :param cov_mat: Spatial covariance matrix of the channel
    :param last_step: Set to `True` if this is the last interpolation step.
        Otherwise, set to `False`.
        If `True`, the output is scaled to ensure its variance is as expected
        by the following interpolation step.

    :input h_hat: [batch_size, num_rx, num_tx, num_streams_per_tx, num_ofdm_symbols, num_subcarriers, num_rx_ant], `torch.complex`.
        Channel estimates.
    :input err_var: [batch_size, num_rx, num_tx, num_streams_per_tx, num_ofdm_symbols, num_subcarriers, num_rx_ant], `torch.float`.
        Channel estimation error variances.

    :output h_hat: [batch_size, num_rx, num_tx, num_streams_per_tx, num_ofdm_symbols, num_subcarriers, num_rx_ant], `torch.complex`.
        Channel estimates smoothed across the spatial dimension.
    :output err_var: [batch_size, num_rx, num_tx, num_streams_per_tx, num_ofdm_symbols, num_subcarriers, num_rx_ant], `torch.float`.
        The channel estimation error variances of the smoothed channel
        estimates.
    """

    def __init__(
        self,
        cov_mat: torch.Tensor,
        last_step: bool,
        precision: Optional[Precision] = None,
        device: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(precision=precision, device=device, **kwargs)

        # Register as buffers for CUDAGraph compatibility
        self.register_buffer(
            "_rzero", torch.zeros((), dtype=self.dtype, device=self.device)
        )
        if not cov_mat.is_complex():
            raise ValueError("`cov_mat` must be complex")
        self.register_buffer("_cov_mat", cov_mat.to(dtype=self.cdtype, device=self.device))
        self._last_step = last_step

        # Indices for adding to diagonal
        num_rx_ant = cov_mat.shape[0]
        add_diag_indices = [[rxa, rxa] for rxa in range(num_rx_ant)]
        self.register_buffer(
            "_add_diag_indices",
            torch.tensor(add_diag_indices, dtype=torch.int64, device=self.device),
        )

    def __call__(
        self, h_hat: torch.Tensor, err_var: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        cov_mat = self._cov_mat
        rzero = self._rzero


        # [num_rx_ant, num_rx_ant]
        cov_mat_t = cov_mat.T

        ##########################################
        # Compute LMMSE matrix
        ##########################################

        # [..., num_rx_ant, num_rx_ant]
        cov_mat_expanded = expand_to_rank(cov_mat, h_hat.dim() + 1, 0)

        # Adding the error variances to the diagonal
        # [..., num_rx_ant, num_rx_ant]
        lmmse_mat = cov_mat_expanded + torch.diag_embed(err_var)

        # Add precision-dependent regularization for numerical stability at high SNR
        # (when err_var becomes very small, the matrix can become ill-conditioned)
        eps = torch.finfo(self.dtype).eps
        rcond = eps * lmmse_mat.shape[-1]
        diag_mean = torch.diagonal(lmmse_mat, dim1=-2, dim2=-1).abs().mean(dim=-1, keepdim=True).unsqueeze(-1)
        reg = rcond * diag_mean * torch.eye(lmmse_mat.shape[-1], dtype=lmmse_mat.dtype, device=self.device)
        lmmse_mat = lmmse_mat + reg

        # [..., num_rx_ant, num_rx_ant]
        # Use cholesky_ex with check_errors=False and solve_triangular for better
        # CUDA graph compatibility (avoids synchronization in cholesky_solve)
        l, info = torch.linalg.cholesky_ex(lmmse_mat, check_errors=False)
        # Solve L L^H X = B via two triangular solves:
        # 1) L Y = B (lower triangular)
        # 2) L^H X = Y (upper triangular on conjugate transpose)
        y = torch.linalg.solve_triangular(l, cov_mat_expanded, upper=False, left=True)
        lmmse_mat = torch.linalg.solve_triangular(l.mH, y, upper=True, left=True)
        lmmse_mat = lmmse_mat.transpose(-1, -2).conj()

        ##########################################
        # Apply smoothing
        ##########################################

        # [..., num_rx_ant]
        h_hat = (lmmse_mat @ h_hat.unsqueeze(-1)).squeeze(-1)

        ##########################################
        # Compute the estimation error variances
        ##########################################

        # [..., num_rx_ant, num_rx_ant]
        cov_mat_t_expanded = expand_to_rank(cov_mat_t, lmmse_mat.dim(), 0)
        # [..., num_rx_ant]
        err_var_out = (cov_mat_t_expanded * lmmse_mat).sum(dim=-1)
        # [..., num_rx_ant]
        err_var_out = torch.diagonal(cov_mat_expanded, dim1=-2, dim2=-1) - err_var_out
        err_var_out = err_var_out.real
        err_var_out = torch.maximum(err_var_out, rzero)

        ##########################################
        # If this is *not* the last
        # interpolation step, scales the
        # input `h_hat` to ensure
        # it has the variance expected by the
        # next interpolation step.
        #
        # The error variance also `err_var`
        # is updated accordingly.
        ##########################################
        if not self._last_step:
            # Conjugate transpose of the LMMSE matrix
            # [..., num_rx_ant, num_rx_ant]
            lmmse_mat_h = lmmse_mat.transpose(-1, -2).conj()

            # First part of the estimate covariance
            # [..., num_rx_ant, num_rx_ant]
            h_hat_var_1 = cov_mat_expanded @ lmmse_mat_h
            h_hat_var_1 = h_hat_var_1.transpose(-1, -2)
            # [..., num_rx_ant]
            h_hat_var_1 = (lmmse_mat * h_hat_var_1).sum(dim=-1)

            # Second part of the estimate covariance
            # [..., num_rx_ant, 1]
            err_var_expanded = err_var.unsqueeze(-1)
            # [..., num_rx_ant, num_rx_ant]
            h_hat_var_2 = err_var_expanded * lmmse_mat_h
            # [..., num_rx_ant, num_rx_ant]
            h_hat_var_2 = h_hat_var_2.transpose(-1, -2)
            # [..., num_rx_ant]
            h_hat_var_2 = (lmmse_mat * h_hat_var_2).sum(dim=-1)

            # Variance of h_hat
            # [..., num_rx_ant]
            h_hat_var = h_hat_var_1 + h_hat_var_2

            # Scaling factor
            # [..., num_rx_ant]
            err_var_c = err_var_out.to(self.cdtype)
            h_var = torch.diagonal(cov_mat_expanded, dim1=-2, dim2=-1)
            denom = h_hat_var + h_var - err_var_c
            s = torch.where(
                denom.abs() > 1e-12, 2.0 * h_var / denom, torch.zeros_like(denom)
            )

            # Apply scaling to estimate
            # [..., num_rx_ant]
            h_hat = s * h_hat

            # Updated variance (using complex arithmetic, then take real part)
            # [..., num_rx_ant]
            err_var_out_c = (
                s * (s - 1.0) * h_hat_var + (1.0 - s) * h_var + s * err_var_c
            )
            err_var_out = err_var_out_c.real
            err_var_out = torch.maximum(err_var_out, rzero)

        return h_hat, err_var_out



#######################################################
# Utilities
#######################################################


def _estimate_ls_at_pilot_locations(
    y_pilots: torch.Tensor, no: torch.Tensor, pilots: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    # y_pilots : [batch_size, num_rx, num_rx_ant, num_tx, num_streams,
    #               num_pilot_symbols], torch.complex
    #     The observed signals for the pilot-carrying resource elements.
    #
    # no : [batch_size, num_rx, num_rx_ant] or only the first n>=0 dims,
    #   torch.float
    #     The variance of the AWGN.
    #
    # pilots : [num_tx, num_streams, num_pilot_symbols], torch.complex
    #     The pilots to be used for channel estimation.

    # Get pilots tensor
    pilots = pilots.to(device=y_pilots.device, dtype=y_pilots.dtype)

    # Compute LS channel estimates
    # Safe division to handle zero pilots
    pilot_mask = pilots.abs() > 0
    safe_pilots = torch.where(pilot_mask, pilots, torch.ones_like(pilots))
    h_ls = torch.where(
        pilot_mask,
        y_pilots / safe_pilots,
        torch.zeros_like(y_pilots))

    # Compute error variance and broadcast to the same shape as h_ls
    # Expand rank of no for broadcasting
    no = expand_to_rank(no, h_ls.dim(), -1)

    # Expand rank of pilots for broadcasting
    pilot_mask_expanded = expand_to_rank(pilot_mask, h_ls.dim(), 0)
    safe_pilots_expanded = expand_to_rank(safe_pilots, h_ls.dim(), 0)

    # Compute error variance, broadcastable to the shape of h_ls
    err_var = torch.where(
        pilot_mask_expanded,
        no / safe_pilots_expanded.abs().square(),
        torch.zeros_like(no),
    )

    # Broadcast err_var to match h_ls shape
    err_var = err_var.expand(h_ls.shape).clone()

    return h_ls, err_var


def tdl_freq_cov_mat(
    model: str,
    subcarrier_spacing: float,
    fft_size: int,
    delay_spread: float,
    precision: Optional[Precision] = None,
    device: Optional[str] = None,
) -> torch.Tensor:
    r"""Compute the frequency covariance matrix of a
    :class:`~sionna.phy.channel.tr38901.TDL` channel model.

    The channel frequency covariance matrix :math:`\mathbf{R}^{(f)}` of a TDL
    channel model is

    .. math::
        \mathbf{R}^{(f)}_{u,v} = \sum_{\ell=1}^L P_\ell e^{-j 2 \pi \tau_\ell \Delta_f (u-v)}, 1 \leq u,v \leq M

    where :math:`M` is the FFT size, :math:`L` is the number of paths for the
    selected TDL model, :math:`P_\ell` and :math:`\tau_\ell` are the average
    power and delay for the :math:`\ell^{\text{th}}` path, respectively, and
    :math:`\Delta_f` is the sub-carrier spacing.

    :param model: TDL model (``"A"``, ``"B"``, ``"C"``, ``"D"``, ``"E"``)
    :param subcarrier_spacing: Sub-carrier spacing [Hz]
    :param fft_size: FFT size
    :param delay_spread: Delay spread [s]
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`,
        :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for computation (e.g., ``"cpu"``, ``"cuda:0"``).
        If `None`, :attr:`~sionna.phy.config.Config.device` is used.

    .. rubric:: Examples

    .. code-block:: python

        from sionna.phy.ofdm import tdl_freq_cov_mat

        cov_mat = tdl_freq_cov_mat("A", 30e3, 64, 100e-9)
        print(cov_mat.shape)
        # torch.Size([64, 64])
    """
    precision = config.precision if precision is None else precision
    cdtype = dtypes[precision]["torch"]["cdtype"]
    device = config.device if device is None else device

    # Load the power delay profile
    if model not in ("A", "B", "C", "D", "E"):
        raise ValueError("model must be one of: 'A', 'B', 'C', 'D', 'E'")
    parameters_fname = f"TDL-{model}.json"
    source = models.parameter_file(parameters_fname)
    params = models.load_json(source)

    los = bool(params["los"])
    delays = np.array(params["delays"]) * delay_spread
    mean_powers = np.power(10.0, np.array(params["powers"]) / 10.0)

    if los:
        mean_powers[0] = mean_powers[0] + mean_powers[1]
        mean_powers = np.concatenate([mean_powers[:1], mean_powers[2:]], axis=0)
        delays = delays[1:]

    # Normalize the PDP
    norm_factor = np.sum(mean_powers)
    mean_powers = mean_powers / norm_factor

    # Build frequency covariance matrix
    n = np.arange(fft_size)
    p = -2.0 * np.pi * subcarrier_spacing * n
    p = np.expand_dims(p, axis=0)
    delays = np.expand_dims(delays, axis=1)
    p = p * delays
    p = np.exp(1j * p)
    p = np.expand_dims(p, axis=-1)
    cov_mat = np.matmul(p, np.transpose(np.conj(p), [0, 2, 1]))
    mean_powers = np.expand_dims(mean_powers, axis=(1, 2))
    cov_mat = np.sum(mean_powers * cov_mat, axis=0)

    return torch.tensor(cov_mat, dtype=cdtype, device=device)


def tdl_time_cov_mat(
    model: str,
    speed: float,
    carrier_frequency: float,
    ofdm_symbol_duration: float,
    num_ofdm_symbols: int,
    los_angle_of_arrival: float = np.arccos(0.7),
    precision: Optional[Precision] = None,
    device: Optional[str] = None,
) -> torch.Tensor:
    r"""Compute the time covariance matrix of a
    :class:`~sionna.phy.channel.tr38901.TDL` channel model.

    For non-line-of-sight (NLoS) model, the channel time covariance matrix
    :math:`\mathbf{R^{(t)}}` of a TDL channel model is

    .. math::
        \mathbf{R^{(t)}}_{u,v} = J_0 \left( \nu \Delta_t \left( u-v \right) \right)

    where :math:`J_0` is the zero-order Bessel function of the first kind,
    :math:`\Delta_t` the duration of an OFDM symbol, and :math:`\nu` the
    Doppler spread defined by

    .. math::
        \nu = 2 \pi \frac{v}{c} f_c

    where :math:`v` is the movement speed, :math:`c` the speed of light, and
    :math:`f_c` the carrier frequency.

    For line-of-sight (LoS) channel models, the channel time covariance matrix
    is

    .. math::
        \mathbf{R^{(t)}}_{u,v} = P_{\text{NLoS}} J_0 \left( \nu \Delta_t \left( u-v \right) \right) + P_{\text{LoS}}e^{j \nu \Delta_t \left( u-v \right) \cos{\alpha_{\text{LoS}}}}

    where :math:`\alpha_{\text{LoS}}` is the angle-of-arrival for the LoS
    path, :math:`P_{\text{NLoS}}` the total power of NLoS paths, and
    :math:`P_{\text{LoS}}` the power of the LoS path. The power delay profile
    is assumed to have unit power, i.e.,
    :math:`P_{\text{NLoS}} + P_{\text{LoS}} = 1`.

    :param model: TDL model (``"A"``, ``"B"``, ``"C"``, ``"D"``, ``"E"``)
    :param speed: Speed [m/s]
    :param carrier_frequency: Carrier frequency [Hz]
    :param ofdm_symbol_duration: Duration of an OFDM symbol [s]
    :param num_ofdm_symbols: Number of OFDM symbols
    :param los_angle_of_arrival: Angle-of-arrival for LoS path [radian].
        Only used with LoS models. Defaults to ``arccos(0.7)`` as specified by
        TR 38.901.
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`,
        :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for computation (e.g., ``"cpu"``, ``"cuda:0"``).
        If `None`, :attr:`~sionna.phy.config.Config.device` is used.

    .. rubric:: Examples

    .. code-block:: python

        from sionna.phy.ofdm import tdl_time_cov_mat

        cov_mat = tdl_time_cov_mat("A", 3.0, 3.5e9, 35.7e-6, 14)
        print(cov_mat.shape)
        # torch.Size([14, 14])
    """
    precision = config.precision if precision is None else precision
    cdtype = dtypes[precision]["torch"]["cdtype"]
    device = config.device if device is None else device

    # Doppler spread
    doppler_spread = 2.0 * PI * speed / SPEED_OF_LIGHT * carrier_frequency

    # Load the power delay profile
    if model not in ("A", "B", "C", "D", "E"):
        raise ValueError("model must be one of: 'A', 'B', 'C', 'D', 'E'")
    parameters_fname = f"TDL-{model}.json"
    source = models.parameter_file(parameters_fname)
    params = models.load_json(source)

    los = bool(params["los"])
    mean_powers = np.power(10.0, np.array(params["powers"]) / 10.0)

    # Normalize the PDP
    norm_factor = np.sum(mean_powers)
    mean_powers = mean_powers / norm_factor

    if los:
        los_power = mean_powers[0]
        nlos_power = np.sum(mean_powers[1:])
    else:
        nlos_power = np.sum(mean_powers)

    # Build time covariance matrix
    indices = np.arange(num_ofdm_symbols)
    s1 = np.expand_dims(indices, axis=1)
    s2 = np.expand_dims(indices, axis=0)
    exp = doppler_spread * ofdm_symbol_duration * (s1 - s2)
    cov_mat_nlos = jv(0.0, exp) * nlos_power

    if los:
        cov_mat_los = np.exp(1j * exp * np.cos(los_angle_of_arrival)) * los_power
        cov_mat = cov_mat_nlos + cov_mat_los
    else:
        cov_mat = cov_mat_nlos

    return torch.tensor(cov_mat, dtype=cdtype, device=device)


# A pilot coord is (tx, stream, symbol, subcarrier) in the effective resource
# grid after guard carriers and the DC carrier have been removed.
_PilotCoord = Tuple[int, int, int, int]

# A pilot group item is (pilot_coord, vector_index, flat_pilot_index). The
# vector index addresses one (tx, stream) compact pilot vector; the flat index
# addresses the flattened [tx, stream, pilot] compact pilot space used for
# gather/scatter.
_PilotGroupItem = Tuple[_PilotCoord, int, int]


def _pilot_ind_from_pattern(
    pilot_pattern: PilotPattern,
    device: Optional[str] = None,
) -> torch.Tensor:
    """Return Sionna-compatible compact pilot indices for ``pilot_pattern``."""
    mask = flatten_last_dims(pilot_pattern.mask)
    pilot_ind = torch.argsort(mask.float(), dim=-1, descending=True, stable=True)
    pilot_ind = pilot_ind[..., : pilot_pattern.num_pilot_symbols]
    return pilot_ind.to(device=device)


class _PilotGroupMixin:
    r"""Mixin for organizing active pilot symbols into equal-length groups.

    Pilot symbols are represented by coordinates of the form
    ``(tx, stream, symbol, subcarrier)``, where ``subcarrier`` refers to the
    effective resource grid after removing guard carriers and the DC carrier.

    The dimensions listed in ``group_over`` vary within a group. All remaining
    dimensions identify the group. For example,
    ``group_over=("subcarrier",)`` creates one group for each fixed
    transmitter, stream, and OFDM symbol, with its pilot values ordered by
    subcarrier index.

    Classes using this mixin must inherit from
    :class:`~sionna.phy.object.Object` and call
    :meth:`_init_pilot_groups` during initialization. Pilot groups are derived
    from a fixed :class:`~sionna.phy.ofdm.PilotPattern` and must all contain
    the same number of active pilot symbols.
    """

    _COORD_COLUMNS = {
        "tx": 0,
        "stream": 1,
        "symbol": 2,
        "subcarrier": 3,
    }

    @classmethod
    def _sanitize_group_dims(cls, dims: Sequence[str]) -> Tuple[str, ...]:
        sanitized: List[str] = []
        for dim in dims:
            key = str(dim)
            if key not in cls._COORD_COLUMNS:
                raise ValueError(
                    f"Unknown pilot group dimension {dim!r}; expected one of "
                    f"{sorted(cls._COORD_COLUMNS)}"
                )
            if key not in sanitized:
                sanitized.append(key)
        return tuple(sanitized)

    @staticmethod
    def _pilot_group_sort_key(
        group_item: _PilotGroupItem,
        sort_columns: Sequence[int],
    ) -> Tuple[int, ...]:
        pilot_coord, vector_index, flat_pilot_index = group_item
        physical_sort_key = tuple(pilot_coord[col] for col in sort_columns)
        return physical_sort_key + (vector_index, flat_pilot_index)

    def _init_pilot_groups(
        self,
        pilot_pattern: PilotPattern,
        *,
        group_over: Sequence[str] = ("subcarrier",),
        active_only: bool = True,
        persistent: bool = True,
    ) -> None:
        r"""Build equal-length pilot groups and register runtime buffers.

        ``group_over`` names the coordinate dimensions that vary inside one
        group. For example, ``group_over=("subcarrier",)`` creates one group
        per fixed ``(tx, stream, symbol)`` and orders the group's pilot
        points by subcarrier.

        :param pilot_pattern: Sionna pilot pattern.
        :param group_over: Coordinate dimensions that vary within one group.
            Valid entries are ``"tx"``, ``"stream"``, ``"symbol"``, and
            ``"subcarrier"``. Their order defines the within-group layout.
        :param active_only: If `True`, entries with zero pilot symbols are
            omitted.
        :param persistent: If `True`, pilot-group buffers are included in the
            module's ``state_dict``.
        """
        pilot_ind = _pilot_ind_from_pattern(
            pilot_pattern,
            device=self.device
        )

        # ``group_over`` varies inside a group; the remaining coordinates
        # define the group itself. Its order defines the within-group layout.
        group_over = self._sanitize_group_dims(group_over)

        group_over_columns = {self._COORD_COLUMNS[d] for d in group_over}
        fixed_columns = [
            col for col in range(len(self._COORD_COLUMNS))
            if col not in group_over_columns
        ]
        sort_columns = [self._COORD_COLUMNS[d] for d in group_over]

        pilot_ind = pilot_ind.detach().cpu().numpy().astype(np.int64)
        pilots = pilot_pattern.pilots.detach().cpu().numpy()

        num_tx = int(pilot_ind.shape[0])
        num_streams_per_tx = int(pilot_ind.shape[1])
        num_pilots = int(pilot_ind.shape[2])
        num_effective_subcarriers = int(pilot_pattern.num_effective_subcarriers)

        # Maps each group-defining coordinate tuple to the pilot resource
        # elements in that group. For group_over=("subcarrier",), a key is
        # (tx, stream, symbol), and its value is a list of pilot group items.
        pilot_groups_by_fixed_coords = {}
        for tx in range(num_tx):
            for stream in range(num_streams_per_tx):
                for vector_index in range(num_pilots):
                    pilot = pilots[tx, stream, vector_index]
                    if active_only and not bool(np.abs(pilot) > 0):
                        continue

                    flat_rg_index = int(pilot_ind[tx, stream, vector_index])
                    symbol = flat_rg_index // num_effective_subcarriers
                    subcarrier = flat_rg_index % num_effective_subcarriers
                    pilot_coord = (tx, stream, symbol, subcarrier)
                    fixed_coord_values = tuple(
                        pilot_coord[col] for col in fixed_columns
                    )
                    stream_offset = tx * num_streams_per_tx + stream
                    flat_pilot_index = stream_offset * num_pilots + vector_index
                    pilot_group_item = (pilot_coord, vector_index, flat_pilot_index)
                    pilot_group = pilot_groups_by_fixed_coords.setdefault(
                        fixed_coord_values, []
                    )
                    pilot_group.append(pilot_group_item)

        if not pilot_groups_by_fixed_coords:
            raise ValueError("Pilot grouping produced no groups.")

        # First sort the groups by their fixed coordinates, then sort the pilot
        # entries inside each group according to ``group_over``.
        def sort_pilot_group(group_item: _PilotGroupItem) -> Tuple[int, ...]:
            return self._pilot_group_sort_key(group_item, sort_columns)

        pilot_groups = []
        for fixed_coord_values in sorted(pilot_groups_by_fixed_coords):
            pilot_group = pilot_groups_by_fixed_coords[fixed_coord_values]
            pilot_groups.append(sorted(pilot_group, key=sort_pilot_group))

        # The registered buffers below are rectangular tensors with shape
        # [num_groups, group_len], so ragged pilot groups are rejected here.
        lengths = [len(pilot_group) for pilot_group in pilot_groups]
        unique_lengths = sorted(set(lengths))
        if len(unique_lengths) != 1:
            raise ValueError(
                "_PilotGroupMixin currently requires equal-length groups; "
                f"got group lengths {unique_lengths}."
            )

        # [num_groups, group_len, 4], with columns
        # [tx, stream, symbol, subcarrier].
        pilot_coords = torch.tensor(
            [[item[0] for item in pilot_group] for pilot_group in pilot_groups],
            dtype=torch.int64,
            device=self.device,
        )
        # [num_groups, group_len], indexing h_ls[..., tx, stream, pilot].
        vector_indices = torch.tensor(
            [[item[1] for item in pilot_group] for pilot_group in pilot_groups],
            dtype=torch.int64,
            device=self.device,
        )
        # [num_groups, group_len], indexing h_ls.flatten(start_dim=-3).
        flat_indices = torch.tensor(
            [[item[2] for item in pilot_group] for pilot_group in pilot_groups],
            dtype=torch.int64,
            device=self.device,
        )

        self._pilot_group_num_groups = int(pilot_coords.shape[0])
        self._pilot_group_len = int(pilot_coords.shape[1])
        self._pilot_group_over = group_over
        self._pilot_group_constant_coord_dims = tuple(
            dim for dim in self._COORD_COLUMNS if dim not in group_over
        )

        self.register_buffer(
            "_pilot_group_coords", pilot_coords, persistent=persistent
        )
        self.register_buffer(
            "_pilot_group_vector_indices", vector_indices, persistent=persistent
        )
        self.register_buffer(
            "_pilot_group_flat_indices", flat_indices, persistent=persistent
        )

    def _expanded_pilot_group_indices(
        self,
        leading_shape: torch.Size,
    ) -> torch.Tensor:
        r"""Return flattened group indices expanded across leading dimensions.

        :param leading_shape: Shape of the dimensions preceding the flattened
            compact pilot dimension.

        :output indices: Expanded gather/scatter indices.
        """
        indices = self._pilot_group_flat_indices.reshape(-1)
        view_shape = (1,) * len(leading_shape) + (indices.numel(),)
        return indices.reshape(view_shape).expand(*leading_shape, indices.numel())

    def _gather_pilot_groups(self, pilot_tensor: torch.Tensor) -> torch.Tensor:
        r"""Gather compact pilot values into grouped vectors.

        :param pilot_tensor: Tensor of shape
            ``[..., num_tx, num_streams_per_tx, num_pilot_symbols]``.

        :output grouped_values: Tensor of shape
            ``[..., num_groups, group_len]``.
        """
        self._check_pilot_groups_initialized()
        if pilot_tensor.dim() < 3:
            raise ValueError("`pilot_tensor` must have at least three dimensions.")

        flat = pilot_tensor.flatten(start_dim=-3)
        leading_shape = flat.shape[:-1]
        indices = self._expanded_pilot_group_indices(leading_shape)
        values = torch.gather(flat, dim=-1, index=indices)
        return values.reshape(
            *leading_shape,
            self._pilot_group_num_groups,
            self._pilot_group_len,
        )

    def _scatter_pilot_groups(
        self,
        base: torch.Tensor,
        grouped_values: torch.Tensor,
    ) -> torch.Tensor:
        r"""Return ``base`` with grouped values scattered into pilot positions.

        :param base: Tensor of shape
            ``[..., num_tx, num_streams_per_tx, num_pilot_symbols]``.
        :param grouped_values: Tensor of shape
            ``[..., num_groups, group_len]``.

        :output updated: Tensor with the same shape as ``base``.
        """
        self._check_pilot_groups_initialized()
        flat = base.flatten(start_dim=-3)
        leading_shape = flat.shape[:-1]
        indices = self._expanded_pilot_group_indices(leading_shape)
        source = grouped_values.reshape(*leading_shape, -1)
        updated = flat.scatter(dim=-1, index=indices, src=source)
        return updated.reshape_as(base)

    def _scatter_pilot_groups_(
        self,
        out: torch.Tensor,
        grouped_values: torch.Tensor,
    ) -> torch.Tensor:
        r"""In-place variant of ``_scatter_pilot_groups``.

        ``out`` must be contiguous over the final three compact-pilot
        dimensions, which is true for tensors created by ``torch.zeros_like`` or
        ``clone`` in the current estimators.

        :param out: Tensor of shape
            ``[..., num_tx, num_streams_per_tx, num_pilot_symbols]``.
        :param grouped_values: Tensor of shape
            ``[..., num_groups, group_len]``.

        :output out: The input tensor after in-place scatter.
        """
        self._check_pilot_groups_initialized()
        flat = out.view(*out.shape[:-3], -1)
        leading_shape = flat.shape[:-1]
        indices = self._expanded_pilot_group_indices(leading_shape)
        source = grouped_values.reshape(*leading_shape, -1)
        flat.scatter_(dim=-1, index=indices, src=source)
        return out

    def _pilot_group_constant_coord(
        self,
        dim: str,
    ) -> torch.Tensor:
        r"""Return per-group coordinate values for a fixed coordinate dimension.

        For example, after ``group_over=("subcarrier",)``, calling
        ``_pilot_group_constant_coord("symbol")`` returns the OFDM symbol
        index of every group. The returned tensor has shape ``[num_groups]``.

        Raises if ``dim`` varies within the groups.

        :param dim: Coordinate dimension.

        :output coords: Tensor of shape ``[num_groups]``.
        """
        self._check_pilot_groups_initialized()
        sanitized_dim = self._sanitize_group_dims((dim,))
        dim = sanitized_dim[0]
        if dim not in self._pilot_group_constant_coord_dims:
            raise ValueError(f"Pilot groups do not have a fixed {dim} coordinate.")

        col = self._COORD_COLUMNS[dim]
        return self._pilot_group_coords[:, 0, col]

    def _pilot_group_varying_coord(
        self,
        dim: str,
    ) -> torch.Tensor:
        r"""Return per-group coordinate values for a varying dimension.

        For example, after ``group_over=("subcarrier",)``, calling
        ``_pilot_group_varying_coord("subcarrier")`` returns the subcarrier
        indices inside every group. The returned tensor has shape
        ``[num_groups, group_len]``.

        Raises if ``dim`` is fixed within the groups.

        :param dim: Coordinate dimension.

        :output coords: Tensor of shape ``[num_groups, group_len]``.
        """
        self._check_pilot_groups_initialized()
        sanitized_dim = self._sanitize_group_dims((dim,))
        dim = sanitized_dim[0]
        if dim not in self._pilot_group_over:
            raise ValueError(f"Pilot groups do not have a varying {dim} coordinate.")

        col = self._COORD_COLUMNS[dim]
        return self._pilot_group_coords[:, :, col]

    def _check_pilot_groups_initialized(self) -> None:
        if not hasattr(self, "_pilot_group_flat_indices"):
            raise RuntimeError(
                "Pilot groups are not initialized. Call _init_pilot_groups() "
                "from the estimator constructor."
            )
