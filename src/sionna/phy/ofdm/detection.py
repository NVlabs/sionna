#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Class definition and functions related to OFDM channel detection."""

from abc import abstractmethod
from typing import Callable, Optional, Union

import torch

from sionna.phy import Block
from sionna.phy.config import Precision
from sionna.phy.mapping import Constellation
from sionna.phy.mimo import StreamManagement
from sionna.phy.mimo import (
    MaximumLikelihoodDetector as MaximumLikelihoodDetector_,
    LinearDetector as LinearDetector_,
    KBestDetector as KBestDetector_,
    EPDetector as EPDetector_,
    MMSEPICDetector as MMSEPICDetector_,
)
from sionna.phy.ofdm import RemoveNulledSubcarriers, ResourceGrid
from sionna.phy.utils import expand_to_rank, flatten_dims, flatten_last_dims, split_dim

__all__ = [
    "OFDMDetector",
    "OFDMDetectorWithPrior",
    "MaximumLikelihoodDetector",
    "MaximumLikelihoodDetectorWithPrior",
    "LinearDetector",
    "KBestDetector",
    "EPDetector",
    "MMSEPICDetector",
]


class OFDMDetector(Block):
    r"""Block that wraps a MIMO detector for use with the OFDM waveform.

    The parameter ``detector`` is a callable (e.g., a function) that
    implements a MIMO detection algorithm for arbitrary batch dimensions.

    This class pre-processes the received resource grid ``y`` and channel
    estimate ``h_hat``, and computes for each receiver the
    noise-plus-interference covariance matrix according to the OFDM and stream
    configuration provided by the ``resource_grid`` and
    ``stream_management``, which also accounts for the channel
    estimation error variance ``err_var``. These quantities serve as input to
    the detection algorithm that is implemented by ``detector``.
    Both detection of symbols or bits with either soft- or hard-decisions
    are supported.

    .. rubric:: Notes

    The callable ``detector`` must take as input a tuple
    :math:`(\mathbf{y}, \mathbf{h}, \mathbf{s})` such that:

    * **y** ([...,num_rx_ant], `torch.complex`) -- 1+D tensor containing the received signals.
    * **h** ([...,num_rx_ant,num_streams_per_rx], `torch.complex`) -- 2+D tensor containing the channel matrices.
    * **s** ([...,num_rx_ant,num_rx_ant], `torch.complex`) -- 2+D tensor containing the noise-plus-interference covariance matrices.

    It must generate one of following outputs depending on the value of
    ``output``:

    * **b_hat** ([..., num_streams_per_rx, num_bits_per_symbol], `torch.float`) -- LLRs or hard-decisions for every bit of every stream, if ``output`` equals `"bit"`.
    * **x_hat** ([..., num_streams_per_rx, num_points], `torch.float`) or ([..., num_streams_per_rx], `torch.int`) -- Logits or hard-decisions for constellation symbols for every stream, if ``output`` equals `"symbol"`. Hard-decisions correspond to the symbol indices.

    :param detector: Callable object (e.g., a function) that implements a MIMO
        detection algorithm for arbitrary batch dimensions. Either one of the
        existing detectors, e.g.,
        :class:`~sionna.phy.mimo.LinearDetector`,
        :class:`~sionna.phy.mimo.MaximumLikelihoodDetector`, or
        :class:`~sionna.phy.mimo.KBestDetector` can be used, or a custom
        detector callable provided that has the same input/output
        specification.
    :param output: Type of output, either "bit" or "symbol"
    :param resource_grid: ResourceGrid to be used
    :param stream_management: StreamManagement to be used
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`,
        :attr:`~sionna.phy.config.Config.precision` is used.
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

    :output z: One of:

        [batch_size, num_tx, num_streams, num_data_symbols\*num_bits_per_symbol], `torch.float`.
        LLRs or hard-decisions for every bit of every stream, if ``output``
        equals `"bit"`.

        [batch_size, num_tx, num_streams, num_data_symbols, num_points], `torch.float` or [batch_size, num_tx, num_streams, num_data_symbols], `torch.int32`.
        Logits or hard-decisions for constellation symbols for every stream,
        if ``output`` equals `"symbol"`.
        Hard-decisions correspond to the symbol indices.

    .. rubric:: Examples

    .. code-block:: python

        import numpy as np
        import torch
        from sionna.phy.ofdm import ResourceGrid, LinearDetector
        from sionna.phy.mimo import StreamManagement

        rg = ResourceGrid(num_ofdm_symbols=14,
                          fft_size=64,
                          subcarrier_spacing=30e3,
                          num_tx=2,
                          num_streams_per_tx=2,
                          pilot_pattern="kronecker",
                          pilot_ofdm_symbol_indices=[2, 11])
        sm = StreamManagement(np.ones([1, 2]), 2)
        detector = LinearDetector("lmmse", "bit", "app", rg, sm,
                                  constellation_type="qam",
                                  num_bits_per_symbol=4)

        batch_size = 16
        y = torch.randn(batch_size, 1, 4, 14, 64, dtype=torch.complex64)
        h_hat = torch.randn(batch_size, 1, 4, 2, 2, 14, 60,
                            dtype=torch.complex64)
        err_var = torch.ones(1) * 0.01
        no = torch.ones(1) * 0.1

        llr = detector(y, h_hat, err_var, no)
        print(llr.shape)
        # torch.Size([16, 2, 2, 3360])
    """

    def __init__(
        self,
        detector: Callable,
        output: str,
        resource_grid: ResourceGrid,
        stream_management: StreamManagement,
        precision: Optional[Precision] = None,
        device: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(precision=precision, device=device, **kwargs)
        self._detector = detector
        self._resource_grid = resource_grid
        self._stream_management = stream_management
        self._removed_nulled_scs = RemoveNulledSubcarriers(
            self._resource_grid, precision=self.precision, device=self.device
        )
        self._output = output

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
            device=self.device,
        )
        self._detection_undesired_ind = torch.tensor(
            stream_management.detection_undesired_ind,
            dtype=torch.long,
            device=self.device,
        )
        self._stream_ind = torch.tensor(
            stream_management.stream_ind, dtype=torch.long, device=self.device
        )

    def _preprocess_inputs(
        self,
        y: torch.Tensor,
        h_hat: torch.Tensor,
        err_var: torch.Tensor,
        no: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pre-process the received signal and compute the
        noise-plus-interference covariance matrix."""

        # Remove nulled subcarriers from y (guards, dc). New shape:
        # [batch_size, num_rx, num_rx_ant, num_ofdm_symbols,
        #  num_effective_subcarriers]
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
            [
                self._stream_management.num_rx,
                self._stream_management.num_streams_per_rx,
            ],
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

        return y_dt, h_dt_desired, s

    def _extract_datasymbols(self, z: torch.Tensor) -> torch.Tensor:
        """Extract data symbols for all detected TX."""

        # If output is symbols with hard decision, the rank is 5 and not 6 as
        # for other cases. The tensor rank is therefore expanded with one extra
        # dimension, which is removed later.
        rank_expanded = z.dim() < 6
        z = expand_to_rank(z, 6, -1)

        # Transpose tensor to shape
        # [num_rx, num_streams_per_rx, num_ofdm_symbols,
        #  num_effective_subcarriers, num_bits_per_symbol or num_points,
        #  batch_size]
        z = z.permute(1, 4, 2, 3, 5, 0)

        # Merge num_rx and num_streams_per_rx
        # [num_rx * num_streams_per_rx, num_ofdm_symbols,
        #  num_effective_subcarriers, num_bits_per_symbol or num_points,
        #  batch_size]
        z = flatten_dims(z, 2, 0)

        # Put first dimension into the right ordering using precomputed tensor index
        z = z[self._stream_ind]

        # Reshape first dimensions to [num_tx, num_streams] so that
        # we can compare to the way the streams were created.
        # [num_tx, num_streams, num_ofdm_symbols, num_effective_subcarriers,
        #  num_bits_per_symbol or num_points, batch_size]
        num_streams = self._stream_management.num_streams_per_tx
        num_tx = self._stream_management.num_tx
        z = split_dim(z, [num_tx, num_streams], 0)

        # Flatten resource grid dimensions
        # [num_tx, num_streams, num_ofdm_symbols*num_effective_subcarrier,
        #  num_bits_per_symbol or num_points, batch_size]
        z = flatten_dims(z, 2, 2)

        # Gather data symbols
        # [num_tx, num_streams, num_data_symbols,
        #  num_bits_per_symbol or num_points, batch_size]
        data_ind_expanded = self._data_ind.unsqueeze(-1).unsqueeze(-1)
        data_ind_expanded = data_ind_expanded.expand(-1, -1, -1, z.shape[3], z.shape[4])
        z = torch.gather(z, 2, data_ind_expanded)

        # Put batch_dim first
        # [batch_size, num_tx, num_streams,
        #  num_data_symbols, num_bits_per_symbol or num_points]
        z = z.permute(4, 0, 1, 2, 3)

        # Reshape LLRs to
        # [batch_size, num_tx, num_streams,
        #  n = num_data_symbols*num_bits_per_symbol]
        # if output is LLRs on bits
        if self._output == "bit":
            z = flatten_dims(z, 2, 3)

        # Remove dummy dimension if output is symbols with hard decision
        if rank_expanded:
            z = z.squeeze(-1)

        return z

    def call(
        self,
        y: torch.Tensor,
        h_hat: torch.Tensor,
        err_var: torch.Tensor,
        no: torch.Tensor,
    ) -> torch.Tensor:
        """Detect transmitted signals.

        :param y: Received OFDM resource grid after cyclic prefix removal
            and FFT with shape
            `[batch_size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size]`
        :param h_hat: Channel estimates for all streams from all transmitters
            with shape `[batch_size, num_rx, num_rx_ant, num_tx,
            num_streams_per_tx, num_ofdm_symbols, num_effective_subcarriers]`
        :param err_var: Variance of the channel estimation error,
            broadcastable to shape of ``h_hat``
        :param no: Variance of the AWGN with shape
            `[batch_size, num_rx, num_rx_ant]` or fewer dimensions
        """
        # Pre-process the inputs
        y_dt, h_dt_desired, s = self._preprocess_inputs(y, h_hat, err_var, no)

        # Detection
        z = self._detector(y_dt, h_dt_desired, s)

        # Extract data symbols for all detected TX
        z = self._extract_datasymbols(z)

        return z


class OFDMDetectorWithPrior(OFDMDetector):
    r"""Block that wraps a MIMO detector that assumes prior knowledge of the
    bits or constellation points is available, for use with the OFDM waveform.

    The parameter ``detector`` is a callable (e.g., a function) that
    implements a MIMO detection algorithm with prior for arbitrary batch
    dimensions.

    This class pre-processes the received resource grid ``y``, channel
    estimate ``h_hat``, and the prior information ``prior``, and computes
    for each receiver the noise-plus-interference covariance matrix according
    to the OFDM and stream configuration provided by the ``resource_grid`` and
    ``stream_management``, which also accounts for the channel
    estimation error variance ``err_var``. These quantities serve as input to
    the detection algorithm that is implemented by ``detector``.
    Both detection of symbols or bits with either soft- or hard-decisions
    are supported.

    .. rubric:: Notes

    The callable ``detector`` must take as input a tuple
    :math:`(\mathbf{y}, \mathbf{h}, \mathbf{prior}, \mathbf{s})` such that:

    * **y** ([...,num_rx_ant], `torch.complex`) -- 1+D tensor containing the received signals.
    * **h** ([...,num_rx_ant,num_streams_per_rx], `torch.complex`) -- 2+D tensor containing the channel matrices.
    * **prior** ([...,num_streams_per_rx,num_bits_per_symbol] or [...,num_streams_per_rx,num_points], `torch.float`) -- Prior for the transmitted signals. If ``output`` equals "bit", then LLRs for the transmitted bits are expected. If ``output`` equals "symbol", then logits for the transmitted constellation points are expected.
    * **s** ([...,num_rx_ant,num_rx_ant], `torch.complex`) -- 2+D tensor containing the noise-plus-interference covariance matrices.

    It must generate one of the following outputs depending on the value of
    ``output``:

    * **b_hat** ([..., num_streams_per_rx, num_bits_per_symbol], `torch.float`) -- LLRs or hard-decisions for every bit of every stream, if ``output`` equals `"bit"`.
    * **x_hat** ([..., num_streams_per_rx, num_points], `torch.float`) or ([..., num_streams_per_rx], `torch.int`) -- Logits or hard-decisions for constellation symbols for every stream, if ``output`` equals `"symbol"`. Hard-decisions correspond to the symbol indices.

    :param detector: Callable object (e.g., a function) that implements a MIMO
        detection algorithm with prior for arbitrary batch dimensions. Either
        the existing detector
        :class:`~sionna.phy.mimo.MaximumLikelihoodDetector` can be used, or a
        custom detector callable provided that has the same input/output
        specification.
    :param output: Type of output, either "bit" or "symbol"
    :param resource_grid: ResourceGrid to be used
    :param stream_management: StreamManagement to be used
    :param constellation_type: Type of constellation, `None` (default),
        "qam", "pam", or "custom".
        For "custom", an instance of
        :class:`~sionna.phy.mapping.Constellation` must be provided.
    :param num_bits_per_symbol: Number of bits per constellation symbol,
        e.g., 4 for QAM16.
        Only required for ``constellation_type`` in ["qam", "pam"].
    :param constellation: Instance of
        :class:`~sionna.phy.mapping.Constellation` or `None`.
        If `None`, ``constellation_type``
        and ``num_bits_per_symbol`` must be provided.
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`,
        :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for tensor operations. If `None`,
        :attr:`~sionna.phy.config.Config.device` is used.

    :input y: [batch_size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size], `torch.complex`.
        Received OFDM resource grid after cyclic prefix removal and FFT.
    :input h_hat: [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols, num_effective_subcarriers], `torch.complex`.
        Channel estimates for all streams from all transmitters.
    :input prior: [batch_size, num_tx, num_streams, num_data_symbols x num_bits_per_symbol] or [batch_size, num_tx, num_streams, num_data_symbols, num_points], `torch.float`.
        Prior of the transmitted signals.
        If ``output`` equals "bit", LLRs of the transmitted bits are expected.
        If ``output`` equals "symbol", logits of the transmitted constellation
        points are expected.
    :input err_var: [Broadcastable to shape of ``h_hat``], `torch.float`.
        Variance of the channel estimation error.
    :input no: [batch_size, num_rx, num_rx_ant] (or only the first n dims), `torch.float`.
        Variance of the AWGN.

    :output z: One of:

        [batch_size, num_tx, num_streams, num_data_symbols\*num_bits_per_symbol], `torch.float`.
        LLRs or hard-decisions for every bit of every stream, if ``output``
        equals `"bit"`.

        [batch_size, num_tx, num_streams, num_data_symbols, num_points], `torch.float` or [batch_size, num_tx, num_streams, num_data_symbols], `torch.int32`.
        Logits or hard-decisions for constellation symbols for every stream,
        if ``output`` equals `"symbol"`.
        Hard-decisions correspond to the symbol indices.
    """

    def __init__(
        self,
        detector: Callable,
        output: str,
        resource_grid: ResourceGrid,
        stream_management: StreamManagement,
        constellation_type: Optional[str] = None,
        num_bits_per_symbol: Optional[int] = None,
        constellation: Optional[Constellation] = None,
        precision: Optional[Precision] = None,
        device: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            detector=detector,
            output=output,
            resource_grid=resource_grid,
            stream_management=stream_management,
            precision=precision,
            device=device,
            **kwargs,
        )

        # Constellation object
        self._constellation = Constellation.check_or_create(
            constellation_type=constellation_type,
            num_bits_per_symbol=num_bits_per_symbol,
            constellation=constellation,
            precision=precision,
            device=device,
        )

        # Precompute indices to map priors to a resource grid
        rg_type = resource_grid.build_type_grid()
        # The nulled subcarriers (nulled DC and guard carriers) are removed to
        # get the correct indices of data-carrying resource elements.
        remove_nulled_sc = RemoveNulledSubcarriers(
            resource_grid, precision=self.precision, device=self.device
        )
        self.register_buffer(
            "_data_ind_scatter",
            torch.nonzero(remove_nulled_sc(rg_type) == 0).to(device=self.device),
        )

        # Store dimensions for pre-allocation
        # Prior dimension depends on output mode
        if output == "bit":
            self._prior_dim = self._constellation.num_bits_per_symbol
        else:
            self._prior_dim = self._constellation.num_points

        # Track allocated batch size for CUDAGraph compatibility
        self._allocated_batch_size = None
        self.register_buffer("_template", None)

    def build(self, y_shape, h_hat_shape, prior_shape, err_var_shape, no_shape):
        """Pre-allocate buffers based on input shapes.

        Called automatically by Block.__call__ in eager mode before tracing.
        This ensures tensor allocations happen outside the compiled graph.
        """
        batch_size = prior_shape[0]
        self.allocate_for_batch_size(batch_size)

    def allocate_for_batch_size(self, batch_size: int) -> None:
        """Pre-allocate buffers for CUDAGraph compatibility.

        :param batch_size: Batch size to allocate for
        """
        if self._allocated_batch_size == batch_size:
            return  # Already allocated

        self._allocated_batch_size = batch_size

        # Pre-allocate template buffer for prior mapping
        # [num_tx, num_streams_per_tx, num_ofdm_symbols,
        #  num_effective_subcarriers, num_bits_per_symbol/num_points, batch_size]
        self.register_buffer(
            "_template",
            torch.zeros(
                [
                    self._resource_grid.num_tx,
                    self._resource_grid.num_streams_per_tx,
                    self._resource_grid.num_ofdm_symbols,
                    self._resource_grid.num_effective_subcarriers,
                    self._prior_dim,
                    batch_size,
                ],
                dtype=self.dtype,
                device=self.device,
            ),
        )

    def call(
        self,
        y: torch.Tensor,
        h_hat: torch.Tensor,
        prior: torch.Tensor,
        err_var: torch.Tensor,
        no: torch.Tensor,
    ) -> torch.Tensor:
        """Detect transmitted signals with prior information.

        :param y: Received OFDM resource grid after cyclic prefix removal
            and FFT with shape
            `[batch_size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size]`
        :param h_hat: Channel estimates for all streams from all transmitters
            with shape `[batch_size, num_rx, num_rx_ant, num_tx,
            num_streams_per_tx, num_ofdm_symbols, num_effective_subcarriers]`
        :param prior: Prior of the transmitted signals with shape
            `[batch_size, num_tx, num_streams, num_data_symbols x
            num_bits_per_symbol]` or `[batch_size, num_tx, num_streams,
            num_data_symbols, num_points]`
        :param err_var: Variance of the channel estimation error,
            broadcastable to shape of ``h_hat``
        :param no: Variance of the AWGN with shape
            `[batch_size, num_rx, num_rx_ant]` or fewer dimensions
        """
        # Pre-process the inputs
        y_dt, h_dt_desired, s = self._preprocess_inputs(y, h_hat, err_var, no)

        # Prepare the prior
        # [batch_size, num_tx, num_streams_per_tx, num_data_symbols,
        #  num_bits_per_symbol/num_points]
        if self._output == "bit":
            prior = split_dim(
                prior,
                [
                    self._resource_grid.pilot_pattern.num_data_symbols,
                    self._constellation.num_bits_per_symbol,
                ],
                3,
            )

        # Priors (LLRs or symbol logits) are inherently real-valued.
        # Convert to real dtype if complex was passed.
        if prior.is_complex():
            prior = prior.real

        batch_size = prior.shape[0]
        num_rx = y.shape[1]

        # [num_tx, num_streams_per_tx, num_data_symbols,
        #  num_bits_per_symbol/num_points, batch_size]
        prior_t = prior.permute(1, 2, 3, 4, 0)

        # Flatten first 3 dimensions
        # [num_data_symbols, num_bits_per_symbol/num_points, batch_size]
        prior_flat = flatten_dims(prior_t, 3, 0)

        # Scatter priors to the template using vectorized indexing
        # Create a fresh tensor each call for CUDA graph compatibility
        # (in-place modification of buffers breaks CUDA graph replay)
        # [num_tx, num_streams_per_tx, num_ofdm_symbols,
        #  num_effective_subcarriers, num_bits_per_symbol/num_points, batch_size]
        template_shape = [
            self._resource_grid.num_tx,
            self._resource_grid.num_streams_per_tx,
            self._resource_grid.num_ofdm_symbols,
            self._resource_grid.num_effective_subcarriers,
            self._prior_dim,
            batch_size,
        ]
        template = torch.zeros(template_shape, dtype=prior.dtype, device=prior.device)
        # Use functional index_put for CUDA graph compatibility
        idx = self._data_ind_scatter
        template = template.index_put(
            (idx[:, 0], idx[:, 1], idx[:, 2], idx[:, 3]), prior_flat
        )

        # [batch_size, num_ofdm_symbols, num_effective_subcarriers,
        #  num_tx*num_streams_per_tx, num_bits_per_symbol/num_points]
        prior_rg = template.permute(5, 2, 3, 0, 1, 4)
        prior_rg = flatten_dims(prior_rg, 2, 3)

        # Add the receive antenna dimension for broadcasting
        # [batch_size, num_rx, num_ofdm_symbols, num_effective_subcarriers,
        #  num_tx*num_streams_per_tx, num_bits_per_symbol/num_points]
        prior_rg = prior_rg.unsqueeze(1).expand(-1, num_rx, -1, -1, -1, -1)

        # Detection with prior
        z = self._detector(y_dt, h_dt_desired, s, prior_rg)

        # Extract data symbols for all detected TX
        z = self._extract_datasymbols(z)

        return z


class MaximumLikelihoodDetector(OFDMDetector):
    r"""Maximum-likelihood (ML) detection for OFDM MIMO transmissions.

    This block implements maximum-likelihood (ML) detection
    for OFDM MIMO transmissions. Both ML detection of symbols or bits with
    either soft- or hard-decisions are supported. The OFDM and stream
    configuration are provided by a
    :class:`~sionna.phy.ofdm.ResourceGrid` and
    :class:`~sionna.phy.mimo.StreamManagement` instance, respectively. The
    actual detector is an instance of
    :class:`~sionna.phy.mimo.MaximumLikelihoodDetector`.

    :param output: Type of output, either "bit" or "symbol"
    :param demapping_method: Demapping method used, either "app" or "maxlog"
    :param resource_grid: ResourceGrid to be used
    :param stream_management: StreamManagement to be used
    :param constellation_type: Type of constellation, `None` (default),
        "qam", "pam", or "custom".
        For "custom", an instance of
        :class:`~sionna.phy.mapping.Constellation` must be provided.
    :param num_bits_per_symbol: Number of bits per constellation symbol,
        e.g., 4 for QAM16.
        Only required for ``constellation_type`` in ["qam", "pam"].
    :param constellation: Instance of
        :class:`~sionna.phy.mapping.Constellation` or `None`.
        If `None`, ``constellation_type``
        and ``num_bits_per_symbol`` must be provided.
    :param hard_out: If `True`, the detector computes hard-decided bit values
        or constellation point indices instead of soft-values.
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`,
        :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for tensor operations. If `None`,
        :attr:`~sionna.phy.config.Config.device` is used.

    :input y: [batch_size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size], `torch.complex`.
        Received OFDM resource grid after cyclic prefix removal and FFT.
    :input h_hat: [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols, num_effective_subcarriers], `torch.complex`.
        Channel estimates for all streams from all transmitters.
    :input err_var: [Broadcastable to shape of ``h_hat``], `torch.float`.
        Variance of the channel estimation error.
    :input no: [batch_size, num_rx, num_rx_ant] (or only the first n dims), `torch.float`.
        Variance of the AWGN noise.

    :output z: One of:

        [batch_size, num_tx, num_streams, num_data_symbols\*num_bits_per_symbol], `torch.float`.
        LLRs or hard-decisions for every bit of every stream, if ``output``
        equals `"bit"`.

        [batch_size, num_tx, num_streams, num_data_symbols, num_points], `torch.float` or [batch_size, num_tx, num_streams, num_data_symbols], `torch.int32`.
        Logits or hard-decisions for constellation symbols for every stream,
        if ``output`` equals `"symbol"`.
        Hard-decisions correspond to the symbol indices.
    """

    def __init__(
        self,
        output: str,
        demapping_method: str,
        resource_grid: ResourceGrid,
        stream_management: StreamManagement,
        constellation_type: Optional[str] = None,
        num_bits_per_symbol: Optional[int] = None,
        constellation: Optional[Constellation] = None,
        hard_out: bool = False,
        precision: Optional[Precision] = None,
        device: Optional[str] = None,
        **kwargs,
    ) -> None:
        # Instantiate the maximum-likelihood detector
        detector = MaximumLikelihoodDetector_(
            output=output,
            demapping_method=demapping_method,
            num_streams=stream_management.num_streams_per_rx,
            constellation_type=constellation_type,
            num_bits_per_symbol=num_bits_per_symbol,
            constellation=constellation,
            hard_out=hard_out,
            precision=precision,
            device=device,
            **kwargs,
        )

        super().__init__(
            detector=detector,
            output=output,
            resource_grid=resource_grid,
            stream_management=stream_management,
            precision=precision,
            device=device,
            **kwargs,
        )


class MaximumLikelihoodDetectorWithPrior(OFDMDetectorWithPrior):
    r"""Maximum-likelihood (ML) detection for OFDM MIMO transmissions,
    assuming prior knowledge of the bits or constellation points is available.

    This block implements maximum-likelihood (ML) detection
    for OFDM MIMO transmissions assuming prior knowledge on the transmitted
    data is available. Both ML detection of symbols or bits with either
    soft- or hard-decisions are supported. The OFDM and stream configuration
    are provided by a
    :class:`~sionna.phy.ofdm.ResourceGrid` and
    :class:`~sionna.phy.mimo.StreamManagement` instance, respectively. The
    actual detector is an instance of
    :class:`~sionna.phy.mimo.MaximumLikelihoodDetector`.

    :param output: Type of output, either "bit" or "symbol"
    :param demapping_method: Demapping method used, either "app" or "maxlog"
    :param resource_grid: ResourceGrid to be used
    :param stream_management: StreamManagement to be used
    :param constellation_type: Type of constellation, `None` (default),
        "qam", "pam", or "custom".
        For "custom", an instance of
        :class:`~sionna.phy.mapping.Constellation` must be provided.
    :param num_bits_per_symbol: Number of bits per constellation symbol,
        e.g., 4 for QAM16.
        Only required for ``constellation_type`` in ["qam", "pam"].
    :param constellation: Instance of
        :class:`~sionna.phy.mapping.Constellation` or `None`.
        If `None`, ``constellation_type``
        and ``num_bits_per_symbol`` must be provided.
    :param hard_out: If `True`, the detector computes hard-decided bit values
        or constellation point indices instead of soft-values.
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`,
        :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for tensor operations. If `None`,
        :attr:`~sionna.phy.config.Config.device` is used.

    :input y: [batch_size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size], `torch.complex`.
        Received OFDM resource grid after cyclic prefix removal and FFT.
    :input h_hat: [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols, num_effective_subcarriers], `torch.complex`.
        Channel estimates for all streams from all transmitters.
    :input prior: [batch_size, num_tx, num_streams, num_data_symbols x num_bits_per_symbol] or [batch_size, num_tx, num_streams, num_data_symbols, num_points], `torch.float`.
        Prior of the transmitted signals.
        If ``output`` equals "bit", LLRs of the transmitted bits are expected.
        If ``output`` equals "symbol", logits of the transmitted constellation
        points are expected.
    :input err_var: [Broadcastable to shape of ``h_hat``], `torch.float`.
        Variance of the channel estimation error.
    :input no: [batch_size, num_rx, num_rx_ant] (or only the first n dims), `torch.float`.
        Variance of the AWGN noise.

    :output z: One of:

        [batch_size, num_tx, num_streams, num_data_symbols\*num_bits_per_symbol], `torch.float`.
        LLRs or hard-decisions for every bit of every stream, if ``output``
        equals `"bit"`.

        [batch_size, num_tx, num_streams, num_data_symbols, num_points], `torch.float` or [batch_size, num_tx, num_streams, num_data_symbols], `torch.int32`.
        Logits or hard-decisions for constellation symbols for every stream,
        if ``output`` equals `"symbol"`.
        Hard-decisions correspond to the symbol indices.
    """

    def __init__(
        self,
        output: str,
        demapping_method: str,
        resource_grid: ResourceGrid,
        stream_management: StreamManagement,
        constellation_type: Optional[str] = None,
        num_bits_per_symbol: Optional[int] = None,
        constellation: Optional[Constellation] = None,
        hard_out: bool = False,
        precision: Optional[Precision] = None,
        device: Optional[str] = None,
        **kwargs,
    ) -> None:
        # Instantiate the maximum-likelihood detector
        detector = MaximumLikelihoodDetector_(
            output=output,
            demapping_method=demapping_method,
            num_streams=stream_management.num_streams_per_rx,
            constellation_type=constellation_type,
            num_bits_per_symbol=num_bits_per_symbol,
            constellation=constellation,
            hard_out=hard_out,
            with_prior=True,
            precision=precision,
            device=device,
            **kwargs,
        )

        super().__init__(
            detector=detector,
            output=output,
            resource_grid=resource_grid,
            stream_management=stream_management,
            constellation_type=constellation_type,
            num_bits_per_symbol=num_bits_per_symbol,
            constellation=constellation,
            precision=precision,
            device=device,
            **kwargs,
        )


class LinearDetector(OFDMDetector):
    r"""Linear detector for OFDM MIMO transmissions.

    This block wraps a MIMO linear equalizer and a
    :class:`~sionna.phy.mapping.Demapper`
    for use with the OFDM waveform.
    Both detection of symbols or bits with
    either soft- or hard-decisions are supported. The OFDM and stream
    configuration are provided by a
    :class:`~sionna.phy.ofdm.ResourceGrid` and
    :class:`~sionna.phy.mimo.StreamManagement` instance, respectively. The
    actual detector is an instance of
    :class:`~sionna.phy.mimo.LinearDetector`.

    :param equalizer: Equalizer to be used. Either one of the existing
        equalizers, e.g.,
        :func:`~sionna.phy.mimo.lmmse_equalizer`,
        :func:`~sionna.phy.mimo.zf_equalizer`, or
        :func:`~sionna.phy.mimo.mf_equalizer` can be used, or a custom
        equalizer function provided that has the same input/output
        specification.
    :param output: Type of output, either "bit" or "symbol"
    :param demapping_method: Demapping method used, either "app" or "maxlog"
    :param resource_grid: ResourceGrid to be used
    :param stream_management: StreamManagement to be used
    :param constellation_type: Type of constellation, `None` (default),
        "qam", "pam", or "custom".
        For "custom", an instance of
        :class:`~sionna.phy.mapping.Constellation` must be provided.
    :param num_bits_per_symbol: Number of bits per constellation symbol,
        e.g., 4 for QAM16.
        Only required for ``constellation_type`` in ["qam", "pam"].
    :param constellation: Instance of
        :class:`~sionna.phy.mapping.Constellation` or `None`.
        If `None`, ``constellation_type``
        and ``num_bits_per_symbol`` must be provided.
    :param hard_out: If `True`, the detector computes hard-decided bit values
        or constellation point indices instead of soft-values.
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`,
        :attr:`~sionna.phy.config.Config.precision` is used.
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

    :output z: One of:

        [batch_size, num_tx, num_streams, num_data_symbols\*num_bits_per_symbol], `torch.float`.
        LLRs or hard-decisions for every bit of every stream, if ``output``
        equals `"bit"`.

        [batch_size, num_tx, num_streams, num_data_symbols, num_points], `torch.float` or [batch_size, num_tx, num_streams, num_data_symbols], `torch.int32`.
        Logits or hard-decisions for constellation symbols for every stream,
        if ``output`` equals `"symbol"`.
        Hard-decisions correspond to the symbol indices.
    """

    def __init__(
        self,
        equalizer: Union[str, Callable],
        output: str,
        demapping_method: str,
        resource_grid: ResourceGrid,
        stream_management: StreamManagement,
        constellation_type: Optional[str] = None,
        num_bits_per_symbol: Optional[int] = None,
        constellation: Optional[Constellation] = None,
        hard_out: bool = False,
        precision: Optional[Precision] = None,
        device: Optional[str] = None,
        **kwargs,
    ) -> None:
        # Instantiate the linear detector
        detector = LinearDetector_(
            equalizer=equalizer,
            output=output,
            demapping_method=demapping_method,
            constellation_type=constellation_type,
            num_bits_per_symbol=num_bits_per_symbol,
            constellation=constellation,
            hard_out=hard_out,
            precision=precision,
            device=device,
            **kwargs,
        )

        super().__init__(
            detector=detector,
            output=output,
            resource_grid=resource_grid,
            stream_management=stream_management,
            precision=precision,
            device=device,
            **kwargs,
        )


class KBestDetector(OFDMDetector):
    r"""K-Best detector for OFDM MIMO transmissions.

    This block wraps the MIMO K-Best detector for use with the OFDM waveform.
    Both detection of symbols or bits with either soft- or hard-decisions
    are supported. The OFDM and stream configuration are provided by a
    :class:`~sionna.phy.ofdm.ResourceGrid` and
    :class:`~sionna.phy.mimo.StreamManagement` instance, respectively. The
    actual detector is an instance of
    :class:`~sionna.phy.mimo.KBestDetector`.

    :param output: Type of output, either "bit" or "symbol"
    :param num_streams: Number of transmitted streams
    :param k: Number of paths to keep. Cannot be larger than the
        number of constellation points to the power of the number of
        streams.
    :param resource_grid: ResourceGrid to be used
    :param stream_management: StreamManagement to be used
    :param constellation_type: Type of constellation, `None` (default),
        "qam", "pam", or "custom".
        For "custom", an instance of
        :class:`~sionna.phy.mapping.Constellation` must be provided.
    :param num_bits_per_symbol: Number of bits per constellation symbol,
        e.g., 4 for QAM16.
        Only required for ``constellation_type`` in ["qam", "pam"].
    :param constellation: Instance of
        :class:`~sionna.phy.mapping.Constellation` or `None`.
        If `None`, ``constellation_type``
        and ``num_bits_per_symbol`` must be provided.
    :param hard_out: If `True`, the detector computes hard-decided bit values
        or constellation point indices instead of soft-values.
    :param use_real_rep: If `True`, the detector uses the real-valued
        equivalent representation of the channel. Note that this only works
        with a QAM constellation.
    :param list2llr: The function to be used to compute LLRs from a list of
        candidate solutions.
        If `None`, the default solution
        :class:`~sionna.phy.mimo.List2LLRSimple` is used.
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`,
        :attr:`~sionna.phy.config.Config.precision` is used.
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

    :output z: One of:

        [batch_size, num_tx, num_streams, num_data_symbols\*num_bits_per_symbol], `torch.float`.
        LLRs or hard-decisions for every bit of every stream, if ``output``
        equals `"bit"`.

        [batch_size, num_tx, num_streams, num_data_symbols, num_points], `torch.float` or [batch_size, num_tx, num_streams, num_data_symbols], `torch.int32`.
        Logits or hard-decisions for constellation symbols for every stream,
        if ``output`` equals `"symbol"`.
        Hard-decisions correspond to the symbol indices.
    """

    def __init__(
        self,
        output: str,
        num_streams: int,
        k: int,
        resource_grid: ResourceGrid,
        stream_management: StreamManagement,
        constellation_type: Optional[str] = None,
        num_bits_per_symbol: Optional[int] = None,
        constellation: Optional[Constellation] = None,
        hard_out: bool = False,
        use_real_rep: bool = False,
        list2llr: Optional[Callable] = None,
        precision: Optional[Precision] = None,
        device: Optional[str] = None,
        **kwargs,
    ) -> None:
        # Instantiate the K-Best detector
        detector = KBestDetector_(
            output=output,
            num_streams=num_streams,
            k=k,
            constellation_type=constellation_type,
            num_bits_per_symbol=num_bits_per_symbol,
            constellation=constellation,
            hard_out=hard_out,
            use_real_rep=use_real_rep,
            list2llr=list2llr,
            precision=precision,
            device=device,
            **kwargs,
        )

        super().__init__(
            detector=detector,
            output=output,
            resource_grid=resource_grid,
            stream_management=stream_management,
            precision=precision,
            device=device,
            **kwargs,
        )


class EPDetector(OFDMDetector):
    r"""Expectation Propagation (EP) detector for OFDM MIMO transmissions.

    This block wraps the MIMO EP detector for use with the OFDM waveform.
    Both detection of symbols or bits with either soft- or hard-decisions
    are supported. The OFDM and stream configuration are provided by a
    :class:`~sionna.phy.ofdm.ResourceGrid` and
    :class:`~sionna.phy.mimo.StreamManagement` instance, respectively. The
    actual detector is an instance of
    :class:`~sionna.phy.mimo.EPDetector`.

    :param output: Type of output, either "bit" or "symbol"
    :param resource_grid: ResourceGrid to be used
    :param stream_management: StreamManagement to be used
    :param num_bits_per_symbol: Number of bits per constellation symbol,
        e.g., 4 for QAM16.
        Only required for ``constellation_type`` in ["qam", "pam"].
    :param hard_out: If `True`, the detector computes hard-decided bit values
        or constellation point indices instead of soft-values.
    :param l: Number of iterations
    :param beta: Parameter :math:`\beta\in[0,1]` for update smoothing
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`,
        :attr:`~sionna.phy.config.Config.precision` is used.
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

    :output z: One of:

        [batch_size, num_tx, num_streams, num_data_symbols\*num_bits_per_symbol], `torch.float`.
        LLRs or hard-decisions for every bit of every stream, if ``output``
        equals `"bit"`.

        [batch_size, num_tx, num_streams, num_data_symbols, num_points], `torch.float` or [batch_size, num_tx, num_streams, num_data_symbols], `torch.int32`.
        Logits or hard-decisions for constellation symbols for every stream,
        if ``output`` equals `"symbol"`.
        Hard-decisions correspond to the symbol indices.
    """

    def __init__(
        self,
        output: str,
        resource_grid: ResourceGrid,
        stream_management: StreamManagement,
        num_bits_per_symbol: Optional[int] = None,
        hard_out: bool = False,
        l: int = 10,
        beta: float = 0.9,
        precision: Optional[Precision] = None,
        device: Optional[str] = None,
        **kwargs,
    ) -> None:
        # Instantiate the EP detector
        detector = EPDetector_(
            output=output,
            num_bits_per_symbol=num_bits_per_symbol,
            hard_out=hard_out,
            l=l,
            beta=beta,
            precision=precision,
            device=device,
            **kwargs,
        )

        super().__init__(
            detector=detector,
            output=output,
            resource_grid=resource_grid,
            stream_management=stream_management,
            precision=precision,
            device=device,
            **kwargs,
        )


class MMSEPICDetector(OFDMDetectorWithPrior):
    r"""MMSE PIC detector for OFDM MIMO transmissions.

    This block wraps the MIMO MMSE PIC detector for use with the OFDM
    waveform.
    Both detection of symbols or bits with either soft- or hard-decisions
    are supported. The OFDM and stream configuration are provided by a
    :class:`~sionna.phy.ofdm.ResourceGrid` and
    :class:`~sionna.phy.mimo.StreamManagement` instance, respectively. The
    actual detector is an instance of
    :class:`~sionna.phy.mimo.MMSEPICDetector`.

    :param output: Type of output, either "bit" or "symbol"
    :param demapping_method: Demapping method used, either "app" or "maxlog"
    :param resource_grid: ResourceGrid to be used
    :param stream_management: StreamManagement to be used
    :param num_iter: Number of MMSE PIC iterations
    :param constellation_type: Type of constellation, `None` (default),
        "qam", "pam", or "custom".
        For "custom", an instance of
        :class:`~sionna.phy.mapping.Constellation` must be provided.
    :param num_bits_per_symbol: Number of bits per constellation symbol,
        e.g., 4 for QAM16.
        Only required for ``constellation_type`` in ["qam", "pam"].
    :param constellation: Instance of
        :class:`~sionna.phy.mapping.Constellation` or `None`.
        If `None`, ``constellation_type``
        and ``num_bits_per_symbol`` must be provided.
    :param hard_out: If `True`, the detector computes hard-decided bit values
        or constellation point indices instead of soft-values.
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`,
        :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for tensor operations. If `None`,
        :attr:`~sionna.phy.config.Config.device` is used.

    :input y: [batch_size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size], `torch.complex`.
        Received OFDM resource grid after cyclic prefix removal and FFT.
    :input h_hat: [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols, num_effective_subcarriers], `torch.complex`.
        Channel estimates for all streams from all transmitters.
    :input prior: [batch_size, num_tx, num_streams, num_data_symbols x num_bits_per_symbol] or [batch_size, num_tx, num_streams, num_data_symbols, num_points], `torch.float`.
        Prior of the transmitted signals.
        If ``output`` equals "bit", LLRs of the transmitted bits are expected.
        If ``output`` equals "symbol", logits of the transmitted constellation
        points are expected.
    :input err_var: [Broadcastable to shape of ``h_hat``], `torch.float`.
        Variance of the channel estimation error.
    :input no: [batch_size, num_rx, num_rx_ant] (or only the first n dims), `torch.float`.
        Variance of the AWGN.

    :output z: One of:

        [batch_size, num_tx, num_streams, num_data_symbols\*num_bits_per_symbol], `torch.float`.
        LLRs or hard-decisions for every bit of every stream, if ``output``
        equals `"bit"`.

        [batch_size, num_tx, num_streams, num_data_symbols, num_points], `torch.float` or [batch_size, num_tx, num_streams, num_data_symbols], `torch.int32`.
        Logits or hard-decisions for constellation symbols for every stream,
        if ``output`` equals `"symbol"`.
        Hard-decisions correspond to the symbol indices.
    """

    def __init__(
        self,
        output: str,
        demapping_method: str,
        resource_grid: ResourceGrid,
        stream_management: StreamManagement,
        num_iter: int = 1,
        constellation_type: Optional[str] = None,
        num_bits_per_symbol: Optional[int] = None,
        constellation: Optional[Constellation] = None,
        hard_out: bool = False,
        precision: Optional[Precision] = None,
        device: Optional[str] = None,
        **kwargs,
    ) -> None:
        # Instantiate the MMSE PIC detector
        detector = MMSEPICDetector_(
            output=output,
            demapping_method=demapping_method,
            num_iter=num_iter,
            constellation_type=constellation_type,
            num_bits_per_symbol=num_bits_per_symbol,
            constellation=constellation,
            hard_out=hard_out,
            precision=precision,
            device=device,
            **kwargs,
        )

        super().__init__(
            detector=detector,
            output=output,
            resource_grid=resource_grid,
            stream_management=stream_management,
            constellation_type=constellation_type,
            num_bits_per_symbol=num_bits_per_symbol,
            constellation=constellation,
            precision=precision,
            device=device,
            **kwargs,
        )
