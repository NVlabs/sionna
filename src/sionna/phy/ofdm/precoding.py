#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Classes and functions related to OFDM transmit precoding."""

from abc import abstractmethod
from typing import Optional, Tuple, Union

import torch

import sionna.phy.ofdm
import sionna.phy.mimo
from sionna.phy import Block
from sionna.phy.config import Precision
from sionna.phy.utils import flatten_dims, expand_to_rank
from sionna.phy.mimo import rzf_precoder, rzf_precoding_matrix, cbf_precoding_matrix
from sionna.phy.ofdm import RemoveNulledSubcarriers

__all__ = [
    "RZFPrecoder",
    "PrecodedChannel",
    "RZFPrecodedChannel",
    "CBFPrecodedChannel",
    "EyePrecodedChannel",
]


class RZFPrecoder(Block):
    r"""Regularized zero-forcing (RZF) precoding for multi-antenna transmissions.

    This block precodes a tensor containing OFDM resource grids using
    the :meth:`~sionna.phy.mimo.rzf_precoder`. For every
    transmitter, the channels to all intended receivers are gathered
    into a channel matrix, based on which the precoding matrix
    is computed and the input tensor is precoded. The block also outputs
    optionally the effective channel after precoding for each stream.

    :param resource_grid: ResourceGrid to be used
    :param stream_management: StreamManagement to be used
    :param return_effective_channel: Indicates if the effective channel
        after precoding should be returned. Defaults to `False`.
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`, :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for computation.
        If `None`, :attr:`~sionna.phy.config.Config.device` is used.

    :input x: [batch_size, num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size], `torch.complex`.
        Resource grids to be precoded.
    :input h: [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_ofdm_symbols, fft_size], `torch.complex`.
        Channel knowledge based on which the precoding is computed.
    :input alpha: `0.` (default) | [batch_size, num_tx, num_ofdm_symbols, fft_size] (or broadcastable), `float`.
        Regularization parameter for RZF precoding. If set to `0`, RZF is equivalent
        to ZF precoding.

    :output x_precoded: [batch_size, num_tx, num_tx_ant, num_ofdm_symbols, fft_size], `torch.complex`.
        Precoded resource grids.
    :output h_eff: [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols, num_effective_subcarriers], `torch.complex`.
        Only returned if ``return_effective_channel=True``.
        The effective channels for all streams after precoding. Can be used to
        simulate perfect channel state information (CSI) at the receivers.
        Nulled subcarriers are automatically removed to be compliant with the
        behavior of a channel estimator.

    .. rubric:: Examples

    .. code-block:: python

        import numpy as np
        import torch
        from sionna.phy.ofdm import ResourceGrid, RZFPrecoder
        from sionna.phy.mimo import StreamManagement

        # Setup: 2 transmitters, 4 receivers (2 per TX), 2 streams per RX
        num_tx = 2
        num_rx_per_tx = 2
        num_rx = num_tx * num_rx_per_tx
        num_streams_per_rx = 2
        num_streams_per_tx = num_rx_per_tx * num_streams_per_rx

        rx_tx_association = np.zeros((num_rx, num_tx), dtype=np.int32)
        for j in range(num_tx):
            rx_tx_association[j*num_rx_per_tx:(j+1)*num_rx_per_tx, j] = 1

        sm = StreamManagement(rx_tx_association, num_streams_per_tx)
        rg = ResourceGrid(num_ofdm_symbols=14, fft_size=64,
                          subcarrier_spacing=15e3, num_tx=num_tx,
                          num_streams_per_tx=num_streams_per_tx)

        precoder = RZFPrecoder(rg, sm)

        # Create inputs
        batch_size = 16
        x = torch.randn(batch_size, num_tx, num_streams_per_tx, 14, 64,
                        dtype=torch.complex64)
        h = torch.randn(batch_size, num_rx, num_streams_per_rx, num_tx,
                        num_streams_per_tx * 2, 14, 64, dtype=torch.complex64)

        x_precoded = precoder(x, h, alpha=0.1)
        print(x_precoded.shape)
        # torch.Size([16, 2, 8, 14, 64])
    """

    def __init__(
        self,
        resource_grid: "sionna.phy.ofdm.ResourceGrid",
        stream_management: "sionna.phy.mimo.StreamManagement",
        return_effective_channel: bool = False,
        precision: Optional[Precision] = None,
        device: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(precision=precision, device=device, **kwargs)
        if not isinstance(resource_grid, sionna.phy.ofdm.ResourceGrid):
            raise TypeError("`resource_grid` must be an instance of ResourceGrid.")
        if not isinstance(stream_management, sionna.phy.mimo.StreamManagement):
            raise TypeError(
                "`stream_management` must be an instance of StreamManagement."
            )
        self._resource_grid = resource_grid
        self._stream_management = stream_management
        self._return_effective_channel = return_effective_channel
        self._remove_nulled_scs = RemoveNulledSubcarriers(
            self._resource_grid, precision=precision, device=device
        )
        
        # Convert precoding indices to tensor and register as buffer for CUDA graph compatibility
        precoding_ind_tensor = torch.tensor(
            stream_management.precoding_ind, 
            dtype=torch.int64, 
            device=self.device
        )
        self.register_buffer("_precoding_ind_tensor", precoding_ind_tensor)

    def _compute_effective_channel(
        self, h: torch.Tensor, g: torch.Tensor
    ) -> torch.Tensor:
        """Compute effective channel after precoding.

        :param h: Channel tensor with shape
            [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_ofdm_symbols, fft_size]
        :param g: Precoding matrix with shape
            [batch_size, num_tx, num_ofdm_symbols, fft_size, num_tx_ant, num_streams_per_tx]
        """
        # Transpose h to shape:
        # [batch_size, num_rx, num_tx, num_ofdm_symbols, fft_size, num_rx_ant, num_tx_ant]
        h = h.permute(0, 1, 3, 5, 6, 2, 4).to(dtype=g.dtype)

        # Compute post-precoding channel without materializing broadcasted g:
        # [batch_size, num_rx, num_tx, num_ofdm_symbols, fft_size, num_rx_ant, num_streams_per_tx]
        h_eff = torch.einsum("brtofxy,btofys->brtofxs", h, g)

        # Permute dimensions to common format of channel tensors:
        # [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size]
        h_eff = h_eff.permute(0, 1, 5, 2, 6, 3, 4)

        # Remove nulled subcarriers
        h_eff = self._remove_nulled_scs(h_eff)

        return h_eff

    def call(
        self,
        x: torch.Tensor,
        h: torch.Tensor,
        alpha: Union[float, torch.Tensor] = 0.0,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Precode the input resource grids.

        :param x: Resource grids to be precoded with shape
            [batch_size, num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size]
        :param h: Channel knowledge with shape
            [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_ofdm_symbols, fft_size]
        :param alpha: Regularization parameter for RZF precoding
        """
        # Transpose x:
        # [batch_size, num_tx, num_ofdm_symbols, fft_size, num_streams_per_tx]
        x_precoded = x.permute(0, 1, 3, 4, 2).to(dtype=self.cdtype)

        # Transpose h:
        # [num_tx, num_rx, num_rx_ant, num_tx_ant, num_ofdm_symbols, fft_size, batch_size]
        h_pc = h.permute(3, 1, 2, 4, 5, 6, 0)

        # Gather desired channel for precoding using advanced indexing
        # precoding_ind has shape [num_tx, num_rx_per_tx]
        precoding_ind = self._precoding_ind_tensor
        num_tx = h_pc.shape[0]

        # [num_tx, num_rx_per_tx, num_rx_ant, num_tx_ant, num_ofdm_symbols, fft_size, batch_size]
        h_pc_desired = torch.stack(
            [h_pc[i, precoding_ind[i]] for i in range(num_tx)]
        )

        # Flatten dims 1,2 (num_rx_per_tx, num_rx_ant):
        # [num_tx, num_rx_per_tx * num_rx_ant, num_tx_ant, num_ofdm_symbols, fft_size, batch_size]
        h_pc_desired = flatten_dims(h_pc_desired, 2, axis=1)

        # Transpose:
        # [batch_size, num_tx, num_ofdm_symbols, fft_size, num_streams_per_tx, num_tx_ant]
        h_pc_desired = h_pc_desired.permute(5, 0, 3, 4, 1, 2).to(dtype=self.cdtype)

        # RZF precoding
        alpha = torch.as_tensor(alpha, dtype=self.dtype, device=self.device)
        alpha = expand_to_rank(alpha, 4, axis=0)
        x_precoded, g = rzf_precoder(
            x_precoded,
            h_pc_desired,
            alpha=alpha,
            return_precoding_matrix=True,
            precision=self.precision,
        )

        # Transpose output to desired shape:
        # [batch_size, num_tx, num_tx_ant, num_ofdm_symbols, fft_size]
        x_precoded = x_precoded.permute(0, 1, 4, 2, 3)

        if self._return_effective_channel:
            h_eff = self._compute_effective_channel(h, g)
            return (x_precoded, h_eff)
        else:
            return x_precoded


class PrecodedChannel(Block):
    r"""Abstract base class to compute the effective channel after precoding.

    Its output can be used to compute the :class:`~sionna.phy.ofdm.PostEqualizationSINR`.

    Let
    :math:`\mathbf{H}_{i,j}\in\mathbb{C}^{\text{num\_rx\_ant}\times\text{num\_tx\_ant}}`
    be the channel matrix between transmitter :math:`j`
    and receiver :math:`i` and let
    :math:`\mathbf{G}_{j}\in\mathbb{C}^{\text{num\_tx\_ant}\times\text{num\_streams\_per\_tx}}`
    be the precoding matrix of transmitter :math:`j`.

    The effective channel
    :math:`\widetilde{\mathbf{H}}_{i,j}\in\mathbb{C}^{\text{num\_rx\_ant}\times\text{num\_streams\_per\_tx}}`
    after precoding is given by

    .. math::
        :label: effective_precoded_channel

        \widetilde{\mathbf{H}}_{i,j} = \mathbf{H}_{i,j}\mathbf{G}_{j}
        \mathop{\text{diag}}(\sqrt{p_{j,1}},...,\sqrt{p_{j,\text{num\_streams\_per\_tx}}})

    where :math:`p_{j,s}` is the transmit power of stream :math:`s` of transmitter :math:`j`.

    :param resource_grid: ResourceGrid to be used
    :param stream_management: StreamManagement to be used
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`, :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for computation.
        If `None`, :attr:`~sionna.phy.config.Config.device` is used.

    :input h: [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_ofdm_symbols, fft_size], `torch.complex`.
        Actual channel realizations.
    :input tx_power: [batch_size, num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size] (or first n dims), `torch.float`.
        Power of each stream for each transmitter.
    :input h_hat: `None` (default) | [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_ofdm_symbols, fft_size], `torch.complex`.
        Channel knowledge based on which the precoding is computed. If set to `None`,
        the actual channel realizations are used.

    :output h_eff: [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols, num_effective_subcarriers], `torch.complex`.
        The effective channel after precoding. Nulled subcarriers are
        automatically removed.
    """

    def __init__(
        self,
        resource_grid: "sionna.phy.ofdm.ResourceGrid",
        stream_management: "sionna.phy.mimo.StreamManagement",
        precision: Optional[Precision] = None,
        device: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(precision=precision, device=device, **kwargs)
        if not isinstance(resource_grid, sionna.phy.ofdm.ResourceGrid):
            raise TypeError("`resource_grid` must be an instance of ResourceGrid.")
        if not isinstance(stream_management, sionna.phy.mimo.StreamManagement):
            raise TypeError(
                "`stream_management` must be an instance of StreamManagement."
            )
        self._resource_grid = resource_grid
        self._stream_management = stream_management
        self._remove_nulled_scs = RemoveNulledSubcarriers(
            self._resource_grid, precision=precision, device=device
        )
        
        # Convert precoding indices to tensor and register as buffer for CUDA graph compatibility
        precoding_ind_tensor = torch.tensor(
            stream_management.precoding_ind, 
            dtype=torch.int64, 
            device=self.device
        )
        self.register_buffer("_precoding_ind_tensor", precoding_ind_tensor)

    def get_desired_channels(self, h_hat: torch.Tensor) -> torch.Tensor:
        r"""Get the desired channels for precoding.

        :param h_hat: Channel knowledge with shape
            [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_ofdm_symbols, fft_size]
        """
        # Transpose:
        # [num_tx, num_rx, num_rx_ant, num_tx_ant, num_ofdm_symbols, fft_size, batch_size]
        h_pc_desired = h_hat.permute(3, 1, 2, 4, 5, 6, 0)

        # Gather desired channel for precoding
        precoding_ind = self._precoding_ind_tensor
        num_tx = h_pc_desired.shape[0]

        # [num_tx, num_rx_per_tx, num_rx_ant, num_tx_ant, num_ofdm_symbols, fft_size, batch_size]
        h_pc_desired = torch.stack(
            [h_pc_desired[i, precoding_ind[i]] for i in range(num_tx)]
        )

        # Flatten dims 1,2:
        # [num_tx, num_rx_per_tx * num_rx_ant, num_tx_ant, num_ofdm_symbols, fft_size, batch_size]
        h_pc_desired = flatten_dims(h_pc_desired, 2, axis=1)

        # Transpose:
        # [batch_size, num_tx, num_ofdm_symbols, fft_size, num_streams_per_tx, num_tx_ant]
        h_pc_desired = h_pc_desired.permute(5, 0, 3, 4, 1, 2)

        num_streams_per_tx = self._stream_management.num_streams_per_tx

        # Check if number of streams per tx matches the channel dimensions
        if h_pc_desired.shape[-2] != num_streams_per_tx:
            msg = (
                "The required number of streams per transmitter"
                " does not match the channel dimensions"
            )
            raise ValueError(msg)

        return h_pc_desired

    def compute_effective_channel(
        self, h: torch.Tensor, g: torch.Tensor
    ) -> torch.Tensor:
        r"""Compute effective channel after precoding.

        :param h: Actual channel realizations with shape
            [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_ofdm_symbols, fft_size]
        :param g: Precoding matrix with shape
            [batch_size, num_tx, num_ofdm_symbols, fft_size, num_tx_ant, num_streams_per_tx]
        """
        # Transpose h to shape:
        # [batch_size, num_rx, num_tx, num_ofdm_symbols, fft_size, num_rx_ant, num_tx_ant]
        h = h.permute(0, 1, 3, 5, 6, 2, 4).to(dtype=g.dtype)

        # Compute post-precoding channel without materializing broadcasted g:
        # [batch_size, num_rx, num_tx, num_ofdm_symbols, fft_size, num_rx_ant, num_streams_per_tx]
        h_eff = torch.einsum("brtofxy,btofys->brtofxs", h, g)

        # Permute dimensions to common format of channel tensors:
        # [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size]
        h_eff = h_eff.permute(0, 1, 5, 2, 6, 3, 4)

        # Remove nulled subcarriers
        h_eff = self._remove_nulled_scs(h_eff)

        return h_eff

    def apply_tx_power(
        self, g: torch.Tensor, tx_power: torch.Tensor
    ) -> torch.Tensor:
        r"""Apply transmit power to precoding vectors.

        :param g: Precoding vectors with shape
            [batch_size, num_tx, num_ofdm_symbols, fft_size, num_tx_ant, num_streams_per_tx]
        :param tx_power: Power of each stream for each transmitter with shape
            [batch_size, num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size] (or first n dims)
        """
        # Expand tx_power to 6 dimensions
        tx_power = expand_to_rank(tx_power, 6, axis=-1)
        # [batch_size, num_tx, num_ofdm_symbols, fft_size, 1, num_streams_per_tx]
        tx_power = tx_power.permute(0, 1, 3, 4, 5, 2)
        tx_power = torch.broadcast_to(tx_power, g.shape)

        # Apply tx power to precoding matrix
        g = tx_power.sqrt().to(dtype=self.cdtype) * g

        return g

    @abstractmethod
    def call(
        self,
        h: torch.Tensor,
        tx_power: torch.Tensor,
        h_hat: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Compute the effective channel after precoding."""
        pass


class RZFPrecodedChannel(PrecodedChannel):
    r"""Compute the effective channel after RZF precoding.

    The precoding matrices are obtained from :func:`~sionna.phy.mimo.rzf_precoding_matrix`.

    :param resource_grid: ResourceGrid to be used
    :param stream_management: StreamManagement to be used
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`, :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for computation.
        If `None`, :attr:`~sionna.phy.config.Config.device` is used.

    :input h: [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_ofdm_symbols, fft_size], `torch.complex`.
        Actual channel realizations.
    :input tx_power: [batch_size, num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size] (or first n dims), `torch.float`.
        Power of each stream for each transmitter.
    :input h_hat: `None` (default) | [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_ofdm_symbols, fft_size], `torch.complex`.
        Channel knowledge based on which the precoding is computed. If set to `None`,
        the actual channel realizations are used.
    :input alpha: `0.` (default) | [batch_size, num_tx, num_ofdm_symbols, fft_size] (or first n dims), `float`.
        Regularization parameter for RZF precoding. If set to `0`, RZF is equivalent
        to ZF precoding.

    :output h_eff: [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols, num_effective_subcarriers], `torch.complex`.
        The effective channel after precoding. Nulled subcarriers are
        automatically removed.

    .. rubric:: Examples

    .. code-block:: python

        import numpy as np
        import torch
        from sionna.phy.ofdm import ResourceGrid, RZFPrecodedChannel
        from sionna.phy.mimo import StreamManagement

        num_tx = 2
        num_rx_per_tx = 2
        num_rx = num_tx * num_rx_per_tx
        num_streams_per_rx = 2
        num_streams_per_tx = num_rx_per_tx * num_streams_per_rx

        rx_tx_association = np.zeros((num_rx, num_tx), dtype=np.int32)
        for j in range(num_tx):
            rx_tx_association[j*num_rx_per_tx:(j+1)*num_rx_per_tx, j] = 1

        sm = StreamManagement(rx_tx_association, num_streams_per_tx)
        rg = ResourceGrid(num_ofdm_symbols=14, fft_size=64,
                          subcarrier_spacing=15e3, num_tx=num_tx,
                          num_streams_per_tx=num_streams_per_tx)

        precoded_channel = RZFPrecodedChannel(rg, sm)

        batch_size = 16
        h = torch.randn(batch_size, num_rx, num_streams_per_rx, num_tx,
                        num_streams_per_tx * 2, 14, 64, dtype=torch.complex64)
        tx_power = torch.rand(batch_size, num_tx, num_streams_per_tx, 14, 64)

        h_eff = precoded_channel(h, tx_power, alpha=0.1)
        print(h_eff.shape)
        # torch.Size([16, 4, 2, 2, 4, 14, 64])
    """

    def call(
        self,
        h: torch.Tensor,
        tx_power: torch.Tensor,
        h_hat: Optional[torch.Tensor] = None,
        alpha: Union[float, torch.Tensor] = 0.0,
    ) -> torch.Tensor:
        """Compute the effective channel after RZF precoding.

        :param h: Actual channel realizations
        :param tx_power: Power of each stream for each transmitter
        :param h_hat: Channel knowledge for precoding computation. If `None`, uses h.
        :param alpha: Regularization parameter for RZF precoding
        """
        if h_hat is None:
            h_hat = h

        # Get desired channels for precoding
        # [batch_size, num_tx, num_ofdm_symbols, fft_size, num_streams_per_tx, num_tx_ant]
        h_pc_desired = self.get_desired_channels(h_hat)

        # Compute precoding matrix
        alpha = torch.as_tensor(alpha, dtype=self.dtype, device=self.device)
        alpha = expand_to_rank(alpha, 4, axis=-1)
        alpha = torch.broadcast_to(alpha, h_pc_desired.shape[:4])

        # [batch_size, num_tx, num_ofdm_symbols, fft_size, num_tx_ant, num_streams_per_tx]
        g = rzf_precoding_matrix(h_pc_desired, alpha, precision=self.precision)

        # Apply transmit power to precoding matrix
        g = self.apply_tx_power(g, tx_power)

        # Compute effective channel
        h_eff = self.compute_effective_channel(h, g)

        return h_eff


class CBFPrecodedChannel(PrecodedChannel):
    r"""Compute the effective channel after conjugate beamforming (CBF) precoding.

    The precoding matrices are obtained from :func:`~sionna.phy.mimo.cbf_precoding_matrix`.

    :param resource_grid: ResourceGrid to be used
    :param stream_management: StreamManagement to be used
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`, :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for computation.
        If `None`, :attr:`~sionna.phy.config.Config.device` is used.

    :input h: [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_ofdm_symbols, fft_size], `torch.complex`.
        Actual channel realizations.
    :input tx_power: [batch_size, num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size] (or first n dims), `torch.float`.
        Power of each stream for each transmitter.
    :input h_hat: `None` (default) | [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_ofdm_symbols, fft_size], `torch.complex`.
        Channel knowledge based on which the precoding is computed. If set to `None`,
        the actual channel realizations are used.

    :output h_eff: [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols, num_effective_subcarriers], `torch.complex`.
        The effective channel after precoding. Nulled subcarriers are
        automatically removed.

    .. rubric:: Examples

    .. code-block:: python

        import numpy as np
        import torch
        from sionna.phy.ofdm import ResourceGrid, CBFPrecodedChannel
        from sionna.phy.mimo import StreamManagement

        num_tx = 2
        num_rx_per_tx = 2
        num_rx = num_tx * num_rx_per_tx
        num_streams_per_rx = 2
        num_streams_per_tx = num_rx_per_tx * num_streams_per_rx

        rx_tx_association = np.zeros((num_rx, num_tx), dtype=np.int32)
        for j in range(num_tx):
            rx_tx_association[j*num_rx_per_tx:(j+1)*num_rx_per_tx, j] = 1

        sm = StreamManagement(rx_tx_association, num_streams_per_tx)
        rg = ResourceGrid(num_ofdm_symbols=14, fft_size=64,
                          subcarrier_spacing=15e3, num_tx=num_tx,
                          num_streams_per_tx=num_streams_per_tx)

        precoded_channel = CBFPrecodedChannel(rg, sm)

        batch_size = 16
        h = torch.randn(batch_size, num_rx, num_streams_per_rx, num_tx,
                        num_streams_per_tx * 2, 14, 64, dtype=torch.complex64)
        tx_power = torch.rand(batch_size, num_tx, num_streams_per_tx, 14, 64)

        h_eff = precoded_channel(h, tx_power)
        print(h_eff.shape)
        # torch.Size([16, 4, 2, 2, 4, 14, 64])
    """

    def call(
        self,
        h: torch.Tensor,
        tx_power: torch.Tensor,
        h_hat: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute the effective channel after CBF precoding.

        :param h: Actual channel realizations
        :param tx_power: Power of each stream for each transmitter
        :param h_hat: Channel knowledge for precoding computation. If `None`, uses h.
        """
        if h_hat is None:
            h_hat = h

        # Get desired channels for precoding
        # [batch_size, num_tx, num_ofdm_symbols, fft_size, num_streams_per_tx, num_tx_ant]
        h_pc_desired = self.get_desired_channels(h_hat)

        # Compute precoding matrix
        # [batch_size, num_tx, num_ofdm_symbols, fft_size, num_tx_ant, num_streams_per_tx]
        g = cbf_precoding_matrix(h_pc_desired, precision=self.precision)

        # Apply transmit power to precoding matrix
        g = self.apply_tx_power(g, tx_power)

        # Compute effective channel
        h_eff = self.compute_effective_channel(h, g)

        return h_eff


class EyePrecodedChannel(PrecodedChannel):
    r"""Compute the effective channel after power allocation without precoding.

    The identity matrix precoder is used, meaning no spatial precoding is applied,
    only power allocation.

    :param resource_grid: ResourceGrid to be used
    :param stream_management: StreamManagement to be used
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`, :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for computation.
        If `None`, :attr:`~sionna.phy.config.Config.device` is used.

    :input h: [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_ofdm_symbols, fft_size], `torch.complex`.
        Actual channel realizations.
    :input tx_power: [batch_size, num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size] (or broadcastable), `torch.float`.
        Power of each stream for each transmitter. Also a lower-rank tensor is
        accepted if it is broadcastable to the requested shape.

    :output h_eff: [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols, num_effective_subcarriers], `torch.complex`.
        The effective channel after power allocation. Nulled subcarriers are
        automatically removed.

    .. rubric:: Examples

    .. code-block:: python

        import numpy as np
        import torch
        from sionna.phy.ofdm import ResourceGrid, EyePrecodedChannel
        from sionna.phy.mimo import StreamManagement

        num_tx = 2
        num_rx = 2
        num_tx_ant = 4
        num_streams_per_tx = num_tx_ant  # Must equal num_tx_ant for identity precoder

        rx_tx_association = np.eye(num_rx, num_tx, dtype=np.int32)
        sm = StreamManagement(rx_tx_association, num_streams_per_tx)
        rg = ResourceGrid(num_ofdm_symbols=14, fft_size=64,
                          subcarrier_spacing=15e3, num_tx=num_tx,
                          num_streams_per_tx=num_streams_per_tx)

        precoded_channel = EyePrecodedChannel(rg, sm)

        batch_size = 16
        h = torch.randn(batch_size, num_rx, 2, num_tx, num_tx_ant, 14, 64,
                        dtype=torch.complex64)
        tx_power = torch.rand(batch_size, num_tx, num_streams_per_tx, 14, 64)

        h_eff = precoded_channel(h, tx_power)
        print(h_eff.shape)
        # torch.Size([16, 2, 2, 2, 4, 14, 64])
    """

    def call(
        self,
        h: torch.Tensor,
        tx_power: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the effective channel with identity precoding.

        :param h: Actual channel realizations
        :param tx_power: Power of each stream for each transmitter
        """
        batch_size = h.shape[0]
        num_tx = h.shape[3]
        num_tx_ant = h.shape[4]
        num_ofdm_symbols = h.shape[5]
        fft_size = h.shape[6]

        # Compute identity precoding matrix
        # [batch_size, num_tx, num_ofdm_symbols, fft_size, num_tx_ant, num_streams_per_tx=num_tx_ant]
        g = torch.eye(num_tx_ant, dtype=self.cdtype, device=self.device)
        g = g.expand(batch_size, num_tx, num_ofdm_symbols, fft_size, num_tx_ant, num_tx_ant)

        # Apply transmit power to precoding matrix
        g = self.apply_tx_power(g, tx_power)

        # Compute effective channel
        h_eff = self.compute_effective_channel(h, g)

        return h_eff
