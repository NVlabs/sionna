#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Classes and functions related to OFDM channel equalization."""

from abc import abstractmethod
from typing import Callable, Optional, Tuple

import torch

from sionna.phy import Block
from sionna.phy.config import Precision
from sionna.phy.mimo import StreamManagement
from sionna.phy.mimo import lmmse_equalizer, zf_equalizer, mf_equalizer, lmmse_matrix
from sionna.phy.ofdm import RemoveNulledSubcarriers, ResourceGrid
from sionna.phy.utils import expand_to_rank, flatten_dims, flatten_last_dims, split_dim
from sionna.phy.utils import inv_cholesky

__all__ = [
    "OFDMEqualizer",
    "LMMSEEqualizer",
    "ZFEqualizer",
    "MFEqualizer",
    "PostEqualizationSINR",
    "LMMSEPostEqualizationSINR",
]


class OFDMEqualizer(Block):
    r"""Block that wraps a MIMO equalizer for use with the OFDM waveform.

    The parameter ``equalizer`` is a callable (e.g., a function) that
    implements a MIMO equalization algorithm for arbitrary batch dimensions.

    This class pre-processes the received resource grid ``y`` and channel
    estimate ``h_hat``, and computes for each receiver the
    noise-plus-interference covariance matrix according to the OFDM and stream
    configuration provided by the ``resource_grid`` and
    ``stream_management``, which also accounts for the channel
    estimation error variance ``err_var``. These quantities serve as input
    to the equalization algorithm that is implemented by the callable
    ``equalizer``. This block computes soft-symbol estimates together with
    effective noise variances for all streams which can, e.g., be used by a
    :class:`~sionna.phy.mapping.Demapper` to obtain LLRs.

    .. rubric:: Notes

    The callable ``equalizer`` must take three inputs:

    * **y** ([...,num_rx_ant], `torch.complex`) -- 1+D tensor containing the received signals.
    * **h** ([...,num_rx_ant,num_streams_per_rx], `torch.complex`) -- 2+D tensor containing the channel matrices.
    * **s** ([...,num_rx_ant,num_rx_ant], `torch.complex`) -- 2+D tensor containing the noise-plus-interference covariance matrices.

    It must generate two outputs:

    * **x_hat** ([...,num_streams_per_rx], `torch.complex`) -- 1+D tensor representing the estimated symbol vectors.
    * **no_eff** (`torch.float`) -- Tensor of the same shape as ``x_hat`` containing the effective noise variance estimates.

    :param equalizer: Callable object (e.g., a function) that implements a
        MIMO equalization algorithm for arbitrary batch dimensions.
    :param resource_grid: ResourceGrid to be used
    :param stream_management: StreamManagement to be used
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`, :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for tensor operations. If `None`,
        :attr:`~sionna.phy.config.Config.device` is used.

    :input y: [batch_size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size], `torch.complex`.
        Received OFDM resource grid after cyclic prefix removal and FFT.
    :input h_hat: [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols, num_effective_subcarriers], `torch.complex`.
        Channel estimates for all streams from all transmitters.
    :input err_var: [Broadcastable to shape of ``h_hat``], `torch.float`.
        Variance of the channel estimation error.
    :input no: [batch_size, num_rx, num_rx_ant] (or only the first n dims), `torch.float`.
        Variance of the AWGN.

    :output x_hat: [batch_size, num_tx, num_streams, num_data_symbols], `torch.complex`.
        Estimated symbols.
    :output no_eff: [batch_size, num_tx, num_streams, num_data_symbols], `torch.float`.
        Effective noise variance for each estimated symbol.

    .. rubric:: Examples

    .. code-block:: python

        import numpy as np
        import torch
        from sionna.phy.ofdm import ResourceGrid, LMMSEEqualizer
        from sionna.phy.mimo import StreamManagement

        rg = ResourceGrid(num_ofdm_symbols=14,
                          fft_size=64,
                          subcarrier_spacing=30e3,
                          num_tx=2,
                          num_streams_per_tx=2,
                          pilot_pattern="kronecker",
                          pilot_ofdm_symbol_indices=[2, 11])
        sm = StreamManagement(np.ones([1, 2]), 2)
        equalizer = LMMSEEqualizer(rg, sm)

        batch_size = 16
        y = torch.randn(batch_size, 1, 4, 14, 64, dtype=torch.complex64)
        h_hat = torch.randn(batch_size, 1, 4, 2, 2, 14, 60, dtype=torch.complex64)
        err_var = torch.ones(1) * 0.01
        no = torch.ones(1) * 0.1

        x_hat, no_eff = equalizer(y, h_hat, err_var, no)
        print(x_hat.shape, no_eff.shape)
        # torch.Size([16, 2, 2, 840]) torch.Size([16, 2, 2, 840])
    """

    def __init__(
        self,
        equalizer: Callable,
        resource_grid: ResourceGrid,
        stream_management: StreamManagement,
        precision: Optional[Precision] = None,
        device: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(precision=precision, device=device, **kwargs)
        if not callable(equalizer):
            raise TypeError("`equalizer` must be callable.")
        self._equalizer = equalizer
        self._resource_grid = resource_grid
        self._stream_management = stream_management
        self._removed_nulled_scs = RemoveNulledSubcarriers(
            self._resource_grid, precision=self.precision, device=self.device
        )

        # Precompute indices to extract data symbols
        # Use stable=True to ensure consistent ordering across CPU/GPU
        mask = resource_grid.pilot_pattern.mask
        num_data_symbols = resource_grid.pilot_pattern.num_data_symbols
        data_ind = torch.argsort(
            flatten_last_dims(mask), dim=-1, descending=False,
            stable=True
        )
        self._data_ind = data_ind[..., :num_data_symbols].to(device=self.device)

        # Precompute stream management indices as tensors for CUDA Graph compatibility
        self._detection_desired_ind = torch.tensor(
            stream_management.detection_desired_ind,
            dtype=torch.long,
            device=self.device
        )
        self._detection_undesired_ind = torch.tensor(
            stream_management.detection_undesired_ind,
            dtype=torch.long,
            device=self.device
        )
        self._stream_ind = torch.tensor(
            stream_management.stream_ind,
            dtype=torch.long,
            device=self.device
        )

    def call(
        self,
        y: torch.Tensor,
        h_hat: torch.Tensor,
        err_var: torch.Tensor,
        no: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Equalize the received signal.

        :param y: Received OFDM resource grid after cyclic prefix removal and
            FFT with shape
            ``[batch_size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size]``
        :param h_hat: Channel estimates for all streams from all transmitters
            with shape ``[batch_size, num_rx, num_rx_ant, num_tx,
            num_streams_per_tx, num_ofdm_symbols,
            num_effective_subcarriers]``
        :param err_var: Variance of the channel estimation error,
            broadcastable to the shape of ``h_hat``
        :param no: Variance of the AWGN with shape
            ``[batch_size, num_rx, num_rx_ant]`` or fewer dimensions
        """
        # Remove nulled subcarriers from y (guards, dc). New shape:
        # [batch_size, num_rx, num_rx_ant,
        #  num_ofdm_symbols, num_effective_subcarriers]
        y_eff = self._removed_nulled_scs(y)

        # Transpose y_eff to put num_rx_ant last. New shape:
        # [batch_size, num_rx, num_ofdm_symbols,
        #  num_effective_subcarriers, num_rx_ant]
        y_dt = y_eff.permute(0, 1, 3, 4, 2).to(self.cdtype)

        # Prepare err_var for MIMO detection
        # New shape is:
        # [batch_size, num_rx, num_ofdm_symbols,
        #  num_effective_subcarriers, num_rx_ant, num_tx*num_streams]
        if not isinstance(err_var, torch.Tensor):
            err_var = torch.as_tensor(err_var, dtype=self.dtype, device=self.device)
        err_var_dt = err_var.broadcast_to(h_hat.shape)
        err_var_dt = err_var_dt.permute(0, 1, 5, 6, 2, 3, 4)
        err_var_dt = flatten_last_dims(err_var_dt, 2).to(self.cdtype)

        # Construct MIMO channels
        # Reshape h_hat for the construction of desired/interfering channels:
        # [num_rx, num_tx, num_streams_per_tx, batch_size, num_rx_ant,
        #  num_ofdm_symbols, num_effective_subcarriers]
        perm = [1, 3, 4, 0, 2, 5, 6]
        h_dt = h_hat.permute(*perm)

        # Flatten first three dimensions:
        # [num_rx*num_tx*num_streams_per_tx, batch_size, num_rx_ant,
        #  num_ofdm_symbols, num_effective_subcarriers]
        h_dt = flatten_dims(h_dt, 3, 0)

        # Gather desired and undesired channels using precomputed tensor indices
        h_dt_desired = h_dt[self._detection_desired_ind]
        h_dt_undesired = h_dt[self._detection_undesired_ind]

        # Split first dimension to separate RX and TX:
        # [num_rx, num_streams_per_rx, batch_size, num_rx_ant,
        #  num_ofdm_symbols, num_effective_subcarriers]
        h_dt_desired = split_dim(
            h_dt_desired,
            [self._stream_management.num_rx, self._stream_management.num_streams_per_rx],
            0,
        )
        h_dt_undesired = split_dim(
            h_dt_undesired, [self._stream_management.num_rx, -1], 0
        )

        # Permutate dims to
        # [batch_size, num_rx, num_ofdm_symbols, num_effective_subcarriers,
        #  num_rx_ant, num_streams_per_rx(num_interfering_streams_per_rx)]
        perm = [2, 0, 4, 5, 3, 1]
        h_dt_desired = h_dt_desired.permute(*perm).to(self.cdtype)
        h_dt_undesired = h_dt_undesired.permute(*perm)

        # Prepare the noise variance
        # no is first broadcast to [batch_size, num_rx, num_rx_ant]
        # then the rank is expanded to that of y
        # then it is transposed like y to the final shape
        # [batch_size, num_rx, num_ofdm_symbols,
        #  num_effective_subcarriers, num_rx_ant]
        no_dt = expand_to_rank(no, 3, -1)
        no_dt = no_dt.broadcast_to(y.shape[:3])
        no_dt = expand_to_rank(no_dt, y.dim(), -1)
        no_dt = no_dt.permute(0, 1, 3, 4, 2).to(self.cdtype)

        # Compute the interference covariance matrix
        # Covariance of undesired transmitters
        s_inf = h_dt_undesired @ h_dt_undesired.mH

        # Thermal noise
        s_no = torch.diag_embed(no_dt)

        # Channel estimation errors
        # As we have only error variance information for each element,
        # we simply sum them across transmitters and build a
        # diagonal covariance matrix from this
        s_csi = torch.diag_embed(err_var_dt.sum(dim=-1))

        # Final covariance matrix
        s = (s_inf + s_no + s_csi).to(self.cdtype)

        # Compute symbol estimate and effective noise variance
        # [batch_size, num_rx, num_ofdm_symbols, num_effective_subcarriers,
        #  num_stream_per_rx]
        x_hat, no_eff = self._equalizer(y_dt, h_dt_desired, s)
        x_hat = x_hat.to(self.cdtype)
        no_eff = no_eff.to(self.dtype)

        # Extract data symbols for all detected TX
        # Transpose tensor to shape
        # [num_rx, num_streams_per_rx, num_ofdm_symbols,
        #  num_effective_subcarriers, batch_size]
        x_hat = x_hat.permute(1, 4, 2, 3, 0)
        no_eff = no_eff.permute(1, 4, 2, 3, 0)

        # Merge num_rx and num_streams_per_rx
        # [num_rx * num_streams_per_rx, num_ofdm_symbols,
        #  num_effective_subcarriers, batch_size]
        x_hat = flatten_dims(x_hat, 2, 0)
        no_eff = flatten_dims(no_eff, 2, 0)

        # Put first dimension into the right ordering using precomputed tensor indices
        x_hat = x_hat[self._stream_ind]
        no_eff = no_eff[self._stream_ind]

        # Reshape first dimensions to [num_tx, num_streams] so that
        # we can compare to the way the streams were created.
        # [num_tx, num_streams, num_ofdm_symbols, num_effective_subcarriers,
        #  batch_size]
        num_streams = self._stream_management.num_streams_per_tx
        num_tx = self._stream_management.num_tx
        x_hat = split_dim(x_hat, [num_tx, num_streams], 0)
        no_eff = split_dim(no_eff, [num_tx, num_streams], 0)

        # Flatten resource grid dimensions
        # [num_tx, num_streams, num_ofdm_symbols*num_effective_subcarriers,
        #  batch_size]
        x_hat = flatten_dims(x_hat, 2, 2)
        no_eff = flatten_dims(no_eff, 2, 2)

        # Broadcast no_eff to the shape of x_hat
        no_eff = no_eff.broadcast_to(x_hat.shape)

        # Gather data symbols
        # [num_tx, num_streams, num_data_symbols, batch_size]
        data_ind_expanded = self._data_ind.unsqueeze(-1)
        data_ind_expanded = data_ind_expanded.expand(-1, -1, -1, x_hat.shape[3])
        x_hat = torch.gather(x_hat, 2, data_ind_expanded)
        no_eff = torch.gather(no_eff, 2, data_ind_expanded)

        # Put batch_dim first
        # [batch_size, num_tx, num_streams, num_data_symbols]
        x_hat = x_hat.permute(3, 0, 1, 2)
        no_eff = no_eff.permute(3, 0, 1, 2)

        return x_hat, no_eff


class LMMSEEqualizer(OFDMEqualizer):
    r"""LMMSE equalization for OFDM MIMO transmissions.

    This block computes linear minimum mean squared error (LMMSE) equalization
    for OFDM MIMO transmissions. The OFDM and stream configuration are provided
    by a :class:`~sionna.phy.ofdm.ResourceGrid` and
    :class:`~sionna.phy.mimo.StreamManagement` instance, respectively. The
    detection algorithm is the :meth:`~sionna.phy.mimo.lmmse_equalizer`. The
    block computes soft-symbol estimates together with effective noise variances
    for all streams which can, e.g., be used by a
    :class:`~sionna.phy.mapping.Demapper` to obtain LLRs.

    :param resource_grid: ResourceGrid to be used
    :param stream_management: StreamManagement to be used
    :param whiten_interference: If `True`, the interference is first whitened
        before equalization. In this case, an alternative expression for the
        receive filter is used which can be numerically more stable.
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`, :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for tensor operations. If `None`,
        :attr:`~sionna.phy.config.Config.device` is used.

    :input y: [batch_size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size], `torch.complex`.
        Received OFDM resource grid after cyclic prefix removal and FFT.
    :input h_hat: [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols, num_effective_subcarriers], `torch.complex`.
        Channel estimates for all streams from all transmitters.
    :input err_var: [Broadcastable to shape of ``h_hat``], `torch.float`.
        Variance of the channel estimation error.
    :input no: [batch_size, num_rx, num_rx_ant] (or only the first n dims), `torch.float`.
        Variance of the AWGN.

    :output x_hat: [batch_size, num_tx, num_streams, num_data_symbols], `torch.complex`.
        Estimated symbols.
    :output no_eff: [batch_size, num_tx, num_streams, num_data_symbols], `torch.float`.
        Effective noise variance for each estimated symbol.
    """

    def __init__(
        self,
        resource_grid: ResourceGrid,
        stream_management: StreamManagement,
        whiten_interference: bool = True,
        precision: Optional[Precision] = None,
        device: Optional[str] = None,
        **kwargs,
    ) -> None:
        def equalizer(y, h, s):
            return lmmse_equalizer(
                y,
                h,
                s,
                whiten_interference=whiten_interference,
                precision=self.precision,
            )

        super().__init__(
            equalizer=equalizer,
            resource_grid=resource_grid,
            stream_management=stream_management,
            precision=precision,
            device=device,
            **kwargs,
        )


class ZFEqualizer(OFDMEqualizer):
    r"""ZF equalization for OFDM MIMO transmissions.

    This block computes zero-forcing (ZF) equalization
    for OFDM MIMO transmissions. The OFDM and stream configuration are provided
    by a :class:`~sionna.phy.ofdm.ResourceGrid` and
    :class:`~sionna.phy.mimo.StreamManagement` instance, respectively. The
    detection algorithm is the :meth:`~sionna.phy.mimo.zf_equalizer`. The block
    computes soft-symbol estimates together with effective noise variances
    for all streams which can, e.g., be used by a
    :class:`~sionna.phy.mapping.Demapper` to obtain LLRs.

    :param resource_grid: ResourceGrid to be used
    :param stream_management: StreamManagement to be used
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`, :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for tensor operations. If `None`,
        :attr:`~sionna.phy.config.Config.device` is used.

    :input y: [batch_size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size], `torch.complex`.
        Received OFDM resource grid after cyclic prefix removal and FFT.
    :input h_hat: [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols, num_effective_subcarriers], `torch.complex`.
        Channel estimates for all streams from all transmitters.
    :input err_var: [Broadcastable to shape of ``h_hat``], `torch.float`.
        Variance of the channel estimation error.
    :input no: [batch_size, num_rx, num_rx_ant] (or only the first n dims), `torch.float`.
        Variance of the AWGN.

    :output x_hat: [batch_size, num_tx, num_streams, num_data_symbols], `torch.complex`.
        Estimated symbols.
    :output no_eff: [batch_size, num_tx, num_streams, num_data_symbols], `torch.float`.
        Effective noise variance for each estimated symbol.
    """

    def __init__(
        self,
        resource_grid: ResourceGrid,
        stream_management: StreamManagement,
        precision: Optional[Precision] = None,
        device: Optional[str] = None,
        **kwargs,
    ) -> None:
        def equalizer(y, h, s):
            return zf_equalizer(y, h, s, precision=self.precision)

        super().__init__(
            equalizer=equalizer,
            resource_grid=resource_grid,
            stream_management=stream_management,
            precision=precision,
            device=device,
            **kwargs,
        )


class MFEqualizer(OFDMEqualizer):
    r"""MF equalization for OFDM MIMO transmissions.

    This block computes matched filter (MF) equalization
    for OFDM MIMO transmissions. The OFDM and stream configuration are provided
    by a :class:`~sionna.phy.ofdm.ResourceGrid` and
    :class:`~sionna.phy.mimo.StreamManagement` instance, respectively. The
    detection algorithm is the :meth:`~sionna.phy.mimo.mf_equalizer`. The block
    computes soft-symbol estimates together with effective noise variances
    for all streams which can, e.g., be used by a
    :class:`~sionna.phy.mapping.Demapper` to obtain LLRs.

    :param resource_grid: ResourceGrid to be used
    :param stream_management: StreamManagement to be used
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`, :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for tensor operations. If `None`,
        :attr:`~sionna.phy.config.Config.device` is used.

    :input y: [batch_size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size], `torch.complex`.
        Received OFDM resource grid after cyclic prefix removal and FFT.
    :input h_hat: [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols, num_effective_subcarriers], `torch.complex`.
        Channel estimates for all streams from all transmitters.
    :input err_var: [Broadcastable to shape of ``h_hat``], `torch.float`.
        Variance of the channel estimation error.
    :input no: [batch_size, num_rx, num_rx_ant] (or only the first n dims), `torch.float`.
        Variance of the AWGN.

    :output x_hat: [batch_size, num_tx, num_streams, num_data_symbols], `torch.complex`.
        Estimated symbols.
    :output no_eff: [batch_size, num_tx, num_streams, num_data_symbols], `torch.float`.
        Effective noise variance for each estimated symbol.
    """

    def __init__(
        self,
        resource_grid: ResourceGrid,
        stream_management: StreamManagement,
        precision: Optional[Precision] = None,
        device: Optional[str] = None,
        **kwargs,
    ) -> None:
        def equalizer(y, h, s):
            return mf_equalizer(y, h, s, precision=self.precision)

        super().__init__(
            equalizer=equalizer,
            resource_grid=resource_grid,
            stream_management=stream_management,
            precision=precision,
            device=device,
            **kwargs,
        )


class PostEqualizationSINR(Block):
    r"""Abstract block that computes the SINR after equalization.

    This function computes the post-equalization SINR for every transmitted
    stream from the :class:`~sionna.phy.ofdm.PrecodedChannel`.
    A stream goes from a specific transmitter to a specific
    receiver and is characterized by a precoding vector and an
    equalization vector.

    Every transmitter is equipped with `num_tx_ant` antennas and every receiver
    is equipped with `num_rx_ant` antennas. All transmitters send the same number
    of streams :math:`S`. A transmitter can allocate different power to different streams.

    Let
    :math:`\mathbf{H}_{i,j}\in\mathbb{C}^{\text{num\_rx\_ant}\times\text{num\_tx\_ant}}`
    be the complex channel matrix between receiver :math:`i` and transmitter
    :math:`j`. We denote by
    :math:`\mathbf{g}_{j_,s}\in\mathbb{C}^{\text{num\_tx\_ant}}` the precoding
    vector
    for stream :math:`s` sent by transmitter :math:`j`.
    Then, the received signal at receiver :math:`i` can be expressed as:

    .. math::
        \mathbf{y}_i = \sum_{j,s} \mathbf{H}_{i,j} \mathbf{g}_{j,s} \sqrt{p_{j,s}} x_{j,s} + \mathbf{n}_{i}

    where :math:`x_{j,s}` and :math:`p_{j,s}` are the unit-power transmit symbol
    and associated transmission power for stream :math:`s`, respectively, and
    :math:`\mathbf{n}_{i}` is the additive noise, distributed as
    :math:`\mathcal{C}\mathcal{N}(0,\sigma^2 \mathbf{I})`.

    By stacking the precoding vectors into a matrix :math:`\mathbf{G}_j=\left[\mathbf{g}_{j,1}, \ldots, \mathbf{g}_{j,S}\right]`,
    and using the definition of the precoded channel :math:`\widetilde{\mathbf{H}}_{i,j}` in
    :eq:`effective_precoded_channel`, the received signal can be rewritten as:

    .. math::
        \mathbf{y}_i = \sum_j \widetilde{\mathbf{H}}_{i,j} \mathop{\text{diag}}(x_{j,1},...,x_{j,S}) + \mathbf{n}_{i}

    Next, let :math:`\mathbf{f}_{i,j,s} \in\mathbb{C}^{\text{num\_rx\_ant}}`
    be the equalization vector for stream :math:`s` of transmitter :math:`j`,
    applied by the intended receiver :math:`i`. Then, the useful signal power for stream :math:`s` of transmitter :math:`j` is:

    .. math::
        u_{i,j,s} = p_{j,s} \left| \mathbf{f}_{i,j,s}^\mathsf{H} \mathbf{H}_{i,j} \mathbf{g}_{j, s} \right|^2.

    We assume that the transmitted symbols :math:`x_{j,s}` are uncorrelated among each
    other. Then, the interference power for this stream can be written
    as:

    .. math::
        v_{i,j,s} = \sum_{(j',s') \ne (j,s)} p_{j',s'} \left| \mathbf{f}_{i,j,s}^\mathsf{H} \mathbf{H}_{i,j'} \mathbf{g}_{j', s'} \right|^2.

    The post-equalization noise power can be expressed as:

    .. math::
        n_{i,j,s} = \sigma^2 \| \mathbf{f}_{i,j,s} \|^2.

    With these definitions, the SINR for this stream which is finally computed as:

    .. math::
        \mathrm{SINR}_{i,j,s} = \frac{u_{i,j,s}}{v_{i,j,s} + n_{i,j,s}}.

    .. rubric:: Notes

    The intended receiver :math:`i` for a particular stream
    :math:`(j,s)` is defined by the :class:`~sionna.phy.mimo.StreamManagement`
    object.

    :param resource_grid: ResourceGrid to be used.
    :param stream_management: StreamManagement to be used.
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`, :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for tensor operations. If `None`,
        :attr:`~sionna.phy.config.Config.device` is used.

    :input h_eff: [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols, num_effective_subcarriers], `torch.complex`.
        Effective channel after precoding as defined in :eq:`effective_precoded_channel`.
    :input no: [batch_size, num_rx, num_rx_ant, num_ofdm_symbols, num_effective_subcarriers] (or only the first n dims), `torch.float`.
        Noise variance.
    :input h_eff_hat: `None` (default) | [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols, num_effective_subcarriers], `torch.complex`.
        Estimated effective channel after precoding. If set to `None`,
        the actual channel realizations are used.

    :output sinr: [batch_size, num_ofdm_symbols, num_effective_subcarriers, num_rx, num_streams_per_rx], `torch.float`.
        SINR after equalization.
    """

    def __init__(
        self,
        resource_grid: ResourceGrid,
        stream_management: StreamManagement,
        precision: Optional[Precision] = None,
        device: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(precision=precision, device=device, **kwargs)
        self._resource_grid = resource_grid
        self._stream_management = stream_management

    def get_per_rx_channels(
        self, h_eff: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Extract desired and undesired channels for each receiver.

        :param h_eff: [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols, num_effective_subcarriers], `torch.complex`.
            Effective precoded channel. Can be estimated or true.

        :output h_eff_desired: [batch_size, num_rx, num_ofdm_symbols, num_effective_subcarriers, num_rx_ant, num_streams_per_rx], `torch.complex`.
            Desired effective channels.
        :output h_eff_undesired: [batch_size, num_rx, num_ofdm_symbols, num_effective_subcarriers, num_rx_ant, num_interfering_streams_per_rx], `torch.complex`.
            Undesired effective channels.
        """
        # Reshape h_eff for the construction of desired/interfering channels:
        # [num_rx, num_tx, num_streams_per_tx, batch_size, num_rx_ant,
        #  num_ofdm_symbols, num_effective_subcarriers]
        perm = [1, 3, 4, 0, 2, 5, 6]
        h_eff_t = h_eff.permute(*perm)

        # Flatten first three dimensions:
        # [num_rx*num_tx*num_streams_per_tx, batch_size, num_rx_ant,
        #  num_ofdm_symbols, num_effective_subcarriers]
        h_eff_t = flatten_dims(h_eff_t, 3, 0)

        # Gather desired and undesired channels
        ind_desired = self._stream_management.detection_desired_ind
        ind_undesired = self._stream_management.detection_undesired_ind
        h_eff_desired = h_eff_t[ind_desired]
        h_eff_undesired = h_eff_t[ind_undesired]

        # Split first dimension to separate RX and TX:
        # [num_rx, num_streams_per_rx, batch_size, num_rx_ant,
        #  num_ofdm_symbols, num_effective_subcarriers]
        h_eff_desired = split_dim(
            h_eff_desired,
            [self._stream_management.num_rx, self._stream_management.num_streams_per_rx],
            0,
        )
        h_eff_undesired = split_dim(
            h_eff_undesired, [self._stream_management.num_rx, -1], 0
        )

        # Permutate dims to
        # [batch_size, num_rx, num_ofdm_symbols, num_effective_subcarriers,
        #  num_rx_ant, num_streams_per_rx(num_interfering_streams_per_rx)]
        perm = [2, 0, 4, 5, 3, 1]
        h_eff_desired = h_eff_desired.permute(*perm)
        h_eff_undesired = h_eff_undesired.permute(*perm)

        return h_eff_desired, h_eff_undesired

    def compute_interference_covariance_matrix(
        self,
        no: Optional[torch.Tensor] = None,
        h_eff_undesired: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        r"""Compute the interference covariance matrix.

        :param no: `None` (default) | [batch_size, num_rx, num_ofdm_symbols, num_effective_subcarriers, num_rx_ant], `torch.float`.
            Noise variance.
        :param h_eff_undesired: `None` (default) | [batch_size, num_rx, num_ofdm_symbols, num_effective_subcarriers, num_rx_ant, num_interfering_streams_per_rx], `torch.complex`.
            Undesired effective channels. If set to `None`, the actual channel realizations are used.

        :output s: [batch_size, num_rx, num_ofdm_symbols, num_effective_subcarriers, num_rx_ant, num_rx_ant], `torch.complex`.
            Interference covariance matrix.
        """
        s_no = 0.0
        if no is not None:
            # Diagonal matrix
            no = no.to(self.cdtype)
            s_no = torch.diag_embed(no)

        s_inf = 0.0
        if h_eff_undesired is not None:
            s_inf = h_eff_undesired @ h_eff_undesired.mH

        s = s_no + s_inf

        return s

    def compute_desired_signal_power(
        self, h_eff_desired: torch.Tensor, f: torch.Tensor
    ) -> torch.Tensor:
        r"""Compute the desired signal power.

        :param h_eff_desired: [batch_size, num_rx, num_ofdm_symbols, num_effective_subcarriers, num_rx_ant, num_streams_per_rx], `torch.complex`.
            Desired effective channels.
        :param f: [batch_size, num_rx, num_ofdm_symbols, num_effective_subcarriers, num_streams_per_rx, num_rx_ant], `torch.complex`.
            Receive combining vectors.

        :output signal_power: [batch_size, num_rx, num_ofdm_symbols, num_effective_subcarriers, num_streams_per_rx], `torch.float`.
            Desired signal power.
        """
        signal_power = torch.einsum("...mn,...nm->...m", f, h_eff_desired)
        signal_power = signal_power.abs().square()
        return signal_power

    def compute_total_power(
        self,
        h_eff_desired: torch.Tensor,
        h_eff_undesired: torch.Tensor,
        f: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the total power from all transmitters.

        :param h_eff_desired: [batch_size, num_rx, num_ofdm_symbols, num_effective_subcarriers, num_rx_ant, num_streams_per_rx], `torch.complex`.
            Desired effective channels.
        :param h_eff_undesired: [batch_size, num_rx, num_ofdm_symbols, num_effective_subcarriers, num_rx_ant, num_interfering_streams_per_rx], `torch.complex`.
            Undesired effective channels.
        :param f: [batch_size, num_rx, num_ofdm_symbols, num_effective_subcarriers, num_streams_per_rx, num_rx_ant], `torch.complex`.
            Receive combining vectors.

        :output total_power: [batch_size, num_rx, num_ofdm_symbols, num_effective_subcarriers, 1], `torch.float`.
            Total power.
        """
        h_eff = torch.cat([h_eff_desired, h_eff_undesired], dim=-1)
        total_power = (f @ h_eff).abs().square()
        total_power = total_power.sum(dim=-1)
        return total_power

    def compute_noise_power(
        self, no: torch.Tensor, f: torch.Tensor
    ) -> torch.Tensor:
        r"""Compute the noise power.

        :param no: [batch_size, num_rx, num_ofdm_symbols, num_effective_subcarriers, num_rx_ant], `torch.float`.
            Noise variance.
        :param f: [batch_size, num_rx, num_ofdm_symbols, num_effective_subcarriers, num_streams_per_rx, num_rx_ant], `torch.complex`.
            Receive combining vectors.
        """
        no = no.real.unsqueeze(-2)
        noise_power = (f.abs().square() * no).sum(dim=-1)
        return noise_power

    def compute_sinr(
        self,
        h_eff_desired: torch.Tensor,
        h_eff_undesired: torch.Tensor,
        no: torch.Tensor,
        f: torch.Tensor,
    ) -> torch.Tensor:
        r"""Compute the SINR.

        :param h_eff_desired: [batch_size, num_rx, num_ofdm_symbols, num_effective_subcarriers, num_rx_ant, num_streams_per_rx], `torch.complex`.
            Desired effective channels.
        :param h_eff_undesired: [batch_size, num_rx, num_ofdm_symbols, num_effective_subcarriers, num_rx_ant, num_interfering_streams_per_rx], `torch.complex`.
            Undesired effective channels.
        :param no: [batch_size, num_rx, num_ofdm_symbols, num_effective_subcarriers, num_rx_ant], `torch.float`.
            Noise variance.
        :param f: [batch_size, num_rx, num_ofdm_symbols, num_effective_subcarriers, num_rx_ant, num_streams_per_rx], `torch.complex`.
            Equalization matrix.

        :output sinr: [batch_size, num_ofdm_symbols, num_effective_subcarriers, num_rx, num_streams_per_rx], `torch.float`.
            Post-equalization SINR.
        """
        signal_power = self.compute_desired_signal_power(h_eff_desired, f)
        total_power = self.compute_total_power(h_eff_desired, h_eff_undesired, f)
        # For numerical stability, avoid negative values
        interference_power = torch.maximum(
            total_power - signal_power,
            torch.zeros(1, dtype=self.dtype, device=total_power.device),
        )
        noise_power = self.compute_noise_power(no, f)
        sinr = torch.where(
            interference_power + noise_power > 0,
            signal_power / (interference_power + noise_power),
            torch.zeros_like(signal_power),
        )

        # Reshape to desired dimensions
        sinr = sinr.permute(0, 2, 3, 1, 4)
        return sinr

    @abstractmethod
    def call(
        self,
        h_eff: torch.Tensor,
        no: torch.Tensor,
        h_eff_hat: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute the post-equalization SINR."""
        pass


class LMMSEPostEqualizationSINR(PostEqualizationSINR):
    r"""Block that computes the SINR after LMMSE equalization.

    The equalization matrix is the one computed by
    :meth:`~sionna.phy.mimo.lmmse_matrix`.

    :param resource_grid: ResourceGrid to be used
    :param stream_management: StreamManagement to be used
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`, :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for tensor operations. If `None`,
        :attr:`~sionna.phy.config.Config.device` is used.

    :input h_eff: [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols, num_effective_subcarriers], `torch.complex`.
        Effective channel after precoding as defined in :eq:`effective_precoded_channel`.
    :input no: [batch_size, num_rx, num_rx_ant, num_ofdm_symbols, num_effective_subcarriers] (or only the first n dims), `torch.float`.
        Noise variance.
    :input h_eff_hat: `None` (default) | [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols, num_effective_subcarriers], `torch.complex`.
        Estimated effective channel after precoding. If set to `None`,
        the actual channel realizations are used.
    :input interference_whitening: `bool` (default=True).
        If set to `True`, also the interference from undesired streams (e.g.,
        from other cells) is whitened.

    :output sinr: [batch_size, num_ofdm_symbols, num_effective_subcarriers, num_rx, num_streams_per_rx], `torch.float`.
        SINR after equalization.

    .. rubric:: Examples

    .. code-block:: python

        import numpy as np
        import torch
        from sionna.phy.ofdm import ResourceGrid, LMMSEPostEqualizationSINR
        from sionna.phy.mimo import StreamManagement

        rg = ResourceGrid(num_ofdm_symbols=14,
                          fft_size=64,
                          subcarrier_spacing=30e3,
                          num_tx=2,
                          num_streams_per_tx=2)
        sm = StreamManagement(np.ones([1, 2]), 2)
        sinr_computer = LMMSEPostEqualizationSINR(rg, sm)

        batch_size = 16
        h_eff = torch.randn(batch_size, 1, 4, 2, 2, 14, 64, dtype=torch.complex64)
        no = torch.ones(1) * 0.1

        sinr = sinr_computer(h_eff, no)
        print(sinr.shape)
        # torch.Size([16, 14, 64, 1, 4])
    """

    def call(
        self,
        h_eff: torch.Tensor,
        no: torch.Tensor,
        h_eff_hat: Optional[torch.Tensor] = None,
        interference_whitening: bool = True,
    ) -> torch.Tensor:
        """Compute the SINR after LMMSE equalization."""
        if h_eff_hat is None:
            h_eff_hat = h_eff

        # Ensure that noise variance has the right dimensions
        no = expand_to_rank(no, 5, -1)
        no = no.broadcast_to(
            [
                h_eff.shape[0],
                h_eff.shape[1],
                h_eff.shape[2],
                h_eff.shape[5],
                h_eff.shape[6],
            ]
        )
        # [batch_size, num_rx, num_ofdm_symbols, num_effective_subcarriers,
        #  num_rx_ant]
        no = no.permute(0, 1, 3, 4, 2)

        # Get estimated desired and undesired channels
        h_eff_desired, h_eff_undesired = self.get_per_rx_channels(h_eff_hat)

        # Compute estimated interference covariance matrix
        if interference_whitening:
            s = self.compute_interference_covariance_matrix(
                no=no, h_eff_undesired=h_eff_undesired
            )
        else:
            s = self.compute_interference_covariance_matrix(no=no)

        # Whiten channels
        l_inv = inv_cholesky(s)  # Compute whitening matrix
        h_eff_desired = l_inv @ h_eff_desired
        h_eff_undesired = l_inv @ h_eff_undesired

        # Compute equalization matrix
        f = lmmse_matrix(h_eff_desired, precision=self.precision)

        # Compute SINR
        sinr = self.compute_sinr(
            h_eff_desired, h_eff_undesired, torch.ones_like(no), f
        )

        return sinr

