#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Class definition and functions related to the resource grid"""

from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import colors

from sionna.phy import Block, Object
from sionna.phy.config import Precision
from sionna.phy.mimo import StreamManagement
from sionna.phy.utils import flatten_dims, flatten_last_dims, split_dim

from .pilot_pattern import EmptyPilotPattern, KroneckerPilotPattern, PilotPattern

__all__ = [
    "ResourceGrid",
    "ResourceGridMapper",
    "ResourceGridDemapper",
    "RemoveNulledSubcarriers",
]


class ResourceGrid(Object):
    r"""Defines a :class:`~sionna.phy.ofdm.ResourceGrid` spanning multiple
    OFDM symbols and subcarriers.

    A resource grid defines how data and pilot symbols are mapped onto a
    sequence of OFDM symbols with a given FFT size. The resource grid can
    also define guard and DC carriers which are nulled. In 4G/5G parlance,
    a resource grid would correspond to a slot.

    Once a :class:`~sionna.phy.ofdm.ResourceGrid` is defined, one can use
    the :class:`~sionna.phy.ofdm.ResourceGridMapper` to map a tensor of
    complex-valued data symbols onto the resource grid, prior to OFDM
    modulation using the :class:`~sionna.phy.ofdm.OFDMModulator` or
    further processing in the frequency domain.

    Subcarriers are numbered from :math:`0` to :math:`N-1`, where :math:`N`
    is the FFT size. The index :math:`0` corresponds to the lowest frequency,
    which is :math:`-\frac{N}{2}\Delta_f` (for :math:`N` even) or
    :math:`-\frac{N-1}{2}\Delta_f` (for :math:`N` odd), where
    :math:`\Delta_f` is the subcarrier spacing.
    The index :math:`N-1` corresponds to the highest frequency,
    which is :math:`(\frac{N}{2}-1)\Delta_f` (for :math:`N` even) or
    :math:`\frac{N-1}{2}\Delta_f` (for :math:`N` odd).

    :param num_ofdm_symbols: Number of OFDM symbols
    :param fft_size: FFT size (i.e., the number of subcarriers)
    :param subcarrier_spacing: Subcarrier spacing [Hz]
    :param num_tx: Number of transmitters. Defaults to `1`.
    :param num_streams_per_tx: Number of streams per transmitter.
        Defaults to `1`.
    :param cyclic_prefix_length: Length of the cyclic prefix.
        Defaults to `0`.
    :param num_guard_carriers: Tuple of two integers defining the number of
        guard carriers at the left and right side of the resource grid.
        Defaults to `(0, 0)`.
    :param dc_null: If `True`, the DC carrier is nulled. Defaults to `False`.
    :param pilot_pattern: An instance of
        :class:`~sionna.phy.ofdm.PilotPattern`, a string shorthand for the
        :class:`~sionna.phy.ofdm.KroneckerPilotPattern` or
        :class:`~sionna.phy.ofdm.EmptyPilotPattern`, or `None`.
        `None` is equivalent to ``"empty"``. Defaults to `None`.
    :param pilot_ofdm_symbol_indices: List of indices of OFDM symbols
        reserved for pilot transmissions. Only needed if
        ``pilot_pattern="kronecker"``. Defaults to `None`.
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`, :attr:`~sionna.phy.config.Config.precision` is
        used.
    :param device: Device for tensor operations. If `None`,
        :attr:`~sionna.phy.config.Config.device` is used.

    .. rubric:: Examples

    The following code snippet shows how to setup and visualize an instance
    of :class:`~sionna.phy.ofdm.ResourceGrid`:

    .. code-block:: python

        from sionna.phy.ofdm import ResourceGrid

        rg = ResourceGrid(num_ofdm_symbols=14,
                          fft_size=64,
                          subcarrier_spacing=30e3,
                          num_tx=1,
                          num_streams_per_tx=1,
                          num_guard_carriers=[5, 6],
                          dc_null=True,
                          pilot_pattern="kronecker",
                          pilot_ofdm_symbol_indices=[2, 11])
        rg.show()

    This code creates a resource grid consisting of 14 OFDM symbols with 64
    subcarriers. The first five and last six subcarriers as well as the DC
    subcarriers are nulled. The second and eleventh OFDM symbol are reserved
    for pilot transmissions.
    """

    def __init__(
        self,
        num_ofdm_symbols: int,
        fft_size: int,
        subcarrier_spacing: float,
        num_tx: int = 1,
        num_streams_per_tx: int = 1,
        cyclic_prefix_length: int = 0,
        num_guard_carriers: Tuple[int, int] = (0, 0),
        dc_null: bool = False,
        pilot_pattern: Optional[Union[str, PilotPattern]] = None,
        pilot_ofdm_symbol_indices: Optional[List[int]] = None,
        precision: Optional[Precision] = None,
        device: Optional[str] = None,
    ) -> None:
        super().__init__(precision=precision, device=device)
        self._num_ofdm_symbols = num_ofdm_symbols
        self._fft_size = fft_size
        self._subcarrier_spacing = subcarrier_spacing
        self._cyclic_prefix_length = int(cyclic_prefix_length)
        self._num_tx = num_tx
        self._num_streams_per_tx = num_streams_per_tx
        self._num_guard_carriers = np.array(num_guard_carriers)
        # Cache sum for torch.compile compatibility
        self._num_guard_carriers_sum = int(np.sum(self._num_guard_carriers))
        self._dc_null = dc_null
        self._pilot_ofdm_symbol_indices = pilot_ofdm_symbol_indices
        self._pilot_pattern: Optional[PilotPattern] = None
        self._check_settings()
        self.pilot_pattern = pilot_pattern

    @property
    def cyclic_prefix_length(self) -> int:
        """Length of the cyclic prefix"""
        return self._cyclic_prefix_length

    @property
    def num_tx(self) -> int:
        """Number of transmitters"""
        return self._num_tx

    @property
    def num_streams_per_tx(self) -> int:
        """Number of streams per transmitter"""
        return self._num_streams_per_tx

    @property
    def num_ofdm_symbols(self) -> int:
        """Number of OFDM symbols of the resource grid"""
        return self._num_ofdm_symbols

    @property
    def num_resource_elements(self) -> int:
        """Number of resource elements"""
        return self._fft_size * self._num_ofdm_symbols

    @property
    def num_effective_subcarriers(self) -> int:
        """Number of subcarriers used for data and pilot transmissions"""
        n = self._fft_size - self._dc_null - self._num_guard_carriers_sum
        return int(n)

    @property
    def effective_subcarrier_ind(self) -> np.ndarray:
        """Indices of the effective subcarriers"""
        num_gc = self._num_guard_carriers
        sc_ind = np.arange(num_gc[0], self.fft_size - num_gc[1])
        if self.dc_null:
            sc_ind = np.delete(sc_ind, self.dc_ind - num_gc[0])
        return sc_ind

    @property
    def num_data_symbols(self) -> int:
        """Number of resource elements used for data transmissions"""
        n = (
            self.num_effective_subcarriers * self._num_ofdm_symbols
            - self.num_pilot_symbols
        )
        return int(n)

    @property
    def num_pilot_symbols(self) -> int:
        """Number of resource elements used for pilot symbols"""
        return self.pilot_pattern.num_pilot_symbols

    @property
    def num_zero_symbols(self) -> int:
        """Number of empty resource elements"""
        n = (
            (self._fft_size - self.num_effective_subcarriers)
            * self._num_ofdm_symbols
        )
        return int(n)

    @property
    def num_guard_carriers(self) -> np.ndarray:
        """Number of left and right guard carriers"""
        return self._num_guard_carriers

    @property
    def dc_ind(self) -> int:
        """Index of the DC subcarrier.

        If ``fft_size`` is odd, the index is ``(fft_size-1)/2``.
        If ``fft_size`` is even, the index is ``fft_size/2``.
        """
        return int(self._fft_size / 2 - (self._fft_size % 2 == 1) / 2)

    @property
    def fft_size(self) -> int:
        """FFT size"""
        return self._fft_size

    @property
    def subcarrier_spacing(self) -> float:
        """Subcarrier spacing [Hz]"""
        return self._subcarrier_spacing

    @property
    def ofdm_symbol_duration(self) -> float:
        """Duration of an OFDM symbol with cyclic prefix [s]"""
        return (
            (1.0 + self.cyclic_prefix_length / self.fft_size)
            / self.subcarrier_spacing
        )

    @property
    def bandwidth(self) -> float:
        """Occupied bandwidth [Hz]: ``fft_size*subcarrier_spacing``"""
        return self.fft_size * self.subcarrier_spacing

    @property
    def num_time_samples(self) -> int:
        """Number of time-domain samples occupied by the resource grid"""
        return (
            (self.fft_size + self.cyclic_prefix_length)
            * self._num_ofdm_symbols
        )

    @property
    def dc_null(self) -> bool:
        """Indicates if the DC carrier is nulled or not"""
        return self._dc_null

    @property
    def pilot_pattern(self) -> PilotPattern:
        """Get/set the used :class:`~sionna.phy.ofdm.PilotPattern`"""
        # With nn.Module inheritance, submodules may be stored in _modules
        # under the property name (due to nn.Module.__setattr__ behavior)
        if "pilot_pattern" in self._modules:
            return self._modules["pilot_pattern"]
        return self._pilot_pattern

    @pilot_pattern.setter
    def pilot_pattern(
        self, value: Optional[Union[str, PilotPattern]]
    ) -> None:
        if value is None:
            value = EmptyPilotPattern(
                self._num_tx,
                self._num_streams_per_tx,
                self._num_ofdm_symbols,
                self.num_effective_subcarriers,
                precision=self.precision,
                device=self.device,
            )
        elif isinstance(value, PilotPattern):
            pass
        elif isinstance(value, str):
            if value not in ["kronecker", "empty"]:
                raise ValueError("Unknown pilot pattern")
            if value == "empty":
                value = EmptyPilotPattern(
                    self._num_tx,
                    self._num_streams_per_tx,
                    self._num_ofdm_symbols,
                    self.num_effective_subcarriers,
                    precision=self.precision,
                    device=self.device,
                )
            elif value == "kronecker":
                if self._pilot_ofdm_symbol_indices is None:
                    raise ValueError(
                        "You must provide pilot_ofdm_symbol_indices."
                    )
                value = KroneckerPilotPattern(
                    self,
                    self._pilot_ofdm_symbol_indices,
                    precision=self.precision,
                    device=self.device,
                )
        else:
            raise ValueError("Unsupported pilot_pattern")
        # When value is an nn.Module, nn.Module.__setattr__ will handle it
        # and register under "pilot_pattern". We still set _pilot_pattern
        # for consistency with property access pattern.
        self._pilot_pattern = value
        # Also register as submodule explicitly
        self._modules["pilot_pattern"] = value

    def _check_settings(self) -> bool:
        """Validate that all properties define a valid resource grid."""
        if not (self._num_ofdm_symbols > 0):
            raise ValueError("`num_ofdm_symbols` must be positive.")
        if not (self._fft_size > 0):
            raise ValueError("`fft_size` must be positive.")
        if not (self._cyclic_prefix_length >= 0):
            raise ValueError("`cyclic_prefix_length` must be nonnegative.")
        if not (self._cyclic_prefix_length <= self._fft_size):
            raise ValueError(
                "`cyclic_prefix_length` cannot be longer than `fft_size`."
            )
        if not (self._num_tx > 0):
            raise ValueError("`num_tx` must be positive.")
        if not (self._num_streams_per_tx > 0):
            raise ValueError("`num_streams_per_tx` must be positive.")
        if len(self._num_guard_carriers) != 2:
            raise ValueError("`num_guard_carriers` must have two elements.")
        if not np.all(np.greater_equal(self._num_guard_carriers, 0)):
            raise ValueError(
                "`num_guard_carriers` must have nonnegative entries."
            )
        if not (
            self._num_guard_carriers_sum <= self._fft_size - self._dc_null
        ):
            raise ValueError(
                "Total number of guard carriers cannot be larger than "
                "`fft_size`."
            )
        if self._dc_null:
            if self._num_guard_carriers[0] > self.dc_ind:
                raise ValueError(
                    "Left guard carriers cannot include the DC subcarrier."
                )
            if self._num_guard_carriers[1] > self._fft_size - self.dc_ind - 1:
                raise ValueError(
                    "Right guard carriers cannot include the DC subcarrier."
                )
        if self.num_effective_subcarriers <= 0:
            raise ValueError(
                "The resource grid must contain at least one effective "
                "subcarrier."
            )
        return True

    def build_type_grid(self) -> torch.Tensor:
        """Returns a tensor indicating the type of each resource element.

        Resource elements can be one of

        - 0 : Data symbol
        - 1 : Pilot symbol
        - 2 : Guard carrier symbol
        - 3 : DC carrier symbol

        :output rg_type: [num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size], `torch.int32`.
            Tensor indicating for each transmitter and stream the type of
            the resource elements of the corresponding resource grid.
            The type can be one of [0, 1, 2, 3] as explained above.
        """
        shape = [self._num_tx, self._num_streams_per_tx, self._num_ofdm_symbols]
        gc_l = 2 * torch.ones(
            shape + [self._num_guard_carriers[0]],
            dtype=torch.int32, device=self.device
        )
        gc_r = 2 * torch.ones(
            shape + [self._num_guard_carriers[1]],
            dtype=torch.int32, device=self.device
        )
        dc = 3 * torch.ones(
            shape + [int(self._dc_null)],
            dtype=torch.int32, device=self.device
        )
        mask = self.pilot_pattern.mask
        split_ind = self.dc_ind - self._num_guard_carriers[0]
        rg_type = torch.cat(
            [
                gc_l,                    # Left guards
                mask[..., :split_ind],   # Data & pilots
                dc,                      # DC
                mask[..., split_ind:],   # Data & pilots
                gc_r                     # Right guards
            ],
            dim=-1
        )
        return rg_type

    def show(
        self, tx_ind: int = 0, tx_stream_ind: int = 0
    ) -> plt.Figure:
        """Visualizes the resource grid for a specific transmitter and stream.

        :param tx_ind: Transmitter index
        :param tx_stream_ind: Stream index
        """
        fig = plt.figure()
        data = self.build_type_grid()[tx_ind, tx_stream_ind].cpu().numpy()
        cmap = colors.ListedColormap([
            [60/256, 8/256, 72/256],
            [45/256, 91/256, 128/256],
            [45/256, 172/256, 111/256],
            [250/256, 228/256, 62/256]
        ])
        bounds = [0, 1, 2, 3, 4]
        norm = colors.BoundaryNorm(bounds, cmap.N)
        img = plt.imshow(
            np.transpose(data),
            interpolation="nearest",
            origin="lower",
            cmap=cmap,
            norm=norm,
            aspect="auto"
        )
        cbar = plt.colorbar(
            img, ticks=[0.5, 1.5, 2.5, 3.5],
            orientation="vertical", shrink=0.8
        )
        cbar.set_ticklabels(["Data", "Pilot", "Guard carrier", "DC carrier"])
        plt.title("OFDM Resource Grid")
        plt.ylabel("Subcarrier Index")
        plt.xlabel("OFDM Symbol")
        plt.xticks(range(0, data.shape[0]))

        return fig


class ResourceGridMapper(Block):
    r"""Maps a tensor of modulated data symbols to a
    :class:`~sionna.phy.ofdm.ResourceGrid`.

    This layer takes as input a tensor of modulated data symbols
    and maps them together with pilot symbols onto an
    OFDM :class:`~sionna.phy.ofdm.ResourceGrid`. The output can be
    converted to a time-domain signal with the
    :class:`~sionna.phy.ofdm.OFDMModulator` or further processed in the
    frequency domain.

    :param resource_grid: :class:`~sionna.phy.ofdm.ResourceGrid` to be used
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`, :attr:`~sionna.phy.config.Config.precision` is
        used.
    :param device: Device for tensor operations. If `None`,
        :attr:`~sionna.phy.config.Config.device` is used.

    :input inputs: [batch_size, num_tx, num_streams_per_tx, num_data_symbols], `torch.complex`.
        Modulated data symbols to be mapped onto the resource grid.

    :output template: [batch_size, num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size], `torch.complex`.
        Full OFDM resource grid in the frequency domain.

    .. rubric:: Examples

    .. code-block:: python

        import torch
        from sionna.phy.ofdm import ResourceGrid, ResourceGridMapper
        from sionna.phy.mapping import QAMSource

        rg = ResourceGrid(num_ofdm_symbols=14,
                          fft_size=64,
                          subcarrier_spacing=30e3)
        mapper = ResourceGridMapper(rg)
        qam = QAMSource(4)

        # Generate data symbols
        x = qam([32, 1, 1, rg.num_data_symbols])
        # Map to resource grid
        rg_mapped = mapper(x)
        print(rg_mapped.shape)
        # torch.Size([32, 1, 1, 14, 64])
    """

    def __init__(
        self,
        resource_grid: ResourceGrid,
        precision: Optional[Precision] = None,
        device: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(precision=precision, device=device, **kwargs)
        self._resource_grid = resource_grid

        # Precompute a tensor of shape
        # [num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size]
        # which is prefilled with pilots and stores indices
        # to scatter data symbols.
        self._rg_type = self._resource_grid.build_type_grid()
        self.register_buffer(
            "_pilot_ind_flat",
            torch.nonzero(self._rg_type.flatten() == 1, as_tuple=False).squeeze(-1),
            persistent=False,
        )
        self.register_buffer(
            "_data_ind_flat",
            torch.nonzero(self._rg_type.flatten() == 0, as_tuple=False).squeeze(-1),
            persistent=False,
        )

    def call(self, inputs: torch.Tensor) -> torch.Tensor:
        """Map data symbols to resource grid.

        :param inputs: Modulated data symbols with shape
            `[batch_size, num_tx, num_streams_per_tx, num_data_symbols]`
        :output template: Full OFDM resource grid with shape
            `[batch_size, num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size]`
        """
        batch_size = inputs.shape[0]

        # Create a flattened resource-grid template
        rg_shape = list(self._rg_type.shape)
        pilot_template = torch.zeros(
            self._rg_type.numel(), dtype=inputs.dtype, device=inputs.device
        )

        # Map pilots onto resource grid (if any)
        if self._pilot_ind_flat.shape[0] > 0:
            pilots = flatten_last_dims(
                self._resource_grid.pilot_pattern.pilots, 3
            ).to(inputs.dtype)
            pilot_template = pilot_template.scatter(
                0, self._pilot_ind_flat, pilots
            )

        # Map data symbols onto resource grid
        # data_flat has shape [batch_size, num_tx * num_streams * num_data_symbols]
        data_flat = flatten_last_dims(inputs, 3)
        template = pilot_template.unsqueeze(0).expand(batch_size, -1)
        data_ind = self._data_ind_flat.unsqueeze(0).expand(batch_size, -1)
        template = template.scatter(1, data_ind, data_flat)

        return template.reshape([batch_size] + rg_shape)


class ResourceGridDemapper(Block):
    r"""Extracts data-carrying resource elements from a resource grid.

    This block takes as input an OFDM
    :class:`~sionna.phy.ofdm.ResourceGrid` and extracts the data-carrying
    resource elements. In other words, it implements the reverse operation
    of :class:`~sionna.phy.ofdm.ResourceGridMapper`.

    :param resource_grid: :class:`~sionna.phy.ofdm.ResourceGrid` to be used
    :param stream_management: :class:`~sionna.phy.mimo.StreamManagement`
        to be used
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`, :attr:`~sionna.phy.config.Config.precision` is
        used.
    :param device: Device for tensor operations. If `None`,
        :attr:`~sionna.phy.config.Config.device` is used.

    :input y: [batch_size, num_rx, num_streams_per_rx, num_ofdm_symbols, fft_size, data_dim], `torch.complex`.
        Full OFDM resource grid in the frequency domain.
        The last dimension ``data_dim`` is optional. If ``data_dim``
        is used, it refers to the dimensionality of the data that should be
        demapped to individual streams. An example would be LLRs.

    :output y: [batch_size, num_rx, num_streams_per_rx, num_data_symbols, data_dim], `torch.complex`.
        The data that were mapped into the resource grid.
        The last dimension ``data_dim`` is only returned if it was used for
        the input.

    .. rubric:: Examples

    .. code-block:: python

        import numpy as np
        from sionna.phy.ofdm import (ResourceGrid,
                                      ResourceGridMapper,
                                      ResourceGridDemapper)
        from sionna.phy.mimo import StreamManagement
        from sionna.phy.mapping import QAMSource

        rg = ResourceGrid(num_ofdm_symbols=14,
                          fft_size=64,
                          subcarrier_spacing=30e3)
        sm = StreamManagement(np.ones([1, 1]), 1)
        mapper = ResourceGridMapper(rg)
        demapper = ResourceGridDemapper(rg, sm)
        qam = QAMSource(4)

        x = qam([32, 1, 1, rg.num_data_symbols])
        rg_mapped = mapper(x)
        x_hat = demapper(rg_mapped)
        print(x_hat.shape)
        # torch.Size([32, 1, 1, 896])
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
        self._stream_management = stream_management
        self._resource_grid = resource_grid

        # Precompute indices to extract data symbols
        mask = resource_grid.pilot_pattern.mask
        num_data_symbols = resource_grid.pilot_pattern.num_data_symbols
        # Use stable=True to maintain relative order for equal elements
        data_ind = torch.argsort(
            flatten_last_dims(mask), dim=-1, stable=True
        )
        self.register_buffer("_data_ind", data_ind[..., :num_data_symbols])

    def call(self, y: torch.Tensor) -> torch.Tensor:
        """Extract data symbols from resource grid.

        :param y: Full OFDM resource grid with shape
            `[batch_size, num_rx, num_streams_per_rx, num_ofdm_symbols, fft_size]`
            or `[batch_size, num_rx, num_streams_per_rx, num_ofdm_symbols, fft_size, data_dim]`
        :output y: Data symbols with shape
            `[batch_size, num_rx, num_streams_per_rx, num_data_symbols]`
            or `[batch_size, num_rx, num_streams_per_rx, num_data_symbols, data_dim]`
        """
        # y has shape
        # [batch_size, num_rx, num_streams_per_rx, num_ofdm_symbols,
        #  fft_size, data_dim]

        # If data_dim is not provided, add a dummy dimension
        squeeze_last = False
        if y.dim() == 5:
            y = y.unsqueeze(-1)
            squeeze_last = True

        # Remove nulled subcarriers from y (guards, dc)
        # Shape: [batch_size, num_rx, num_rx_ant,
        #         num_ofdm_symbols, num_effective_subcarriers, data_dim]
        y = y[..., self._resource_grid.effective_subcarrier_ind, :]

        # Transpose tensor to shape
        # [num_rx, num_streams_per_rx, num_ofdm_symbols,
        #  num_effective_subcarriers, data_dim, batch_size]
        y = y.permute(1, 2, 3, 4, 5, 0)

        # Merge num_rx and num_streams_per_rx
        # [num_rx * num_streams_per_rx, num_ofdm_symbols,
        #  num_effective_subcarriers, data_dim, batch_size]
        y = flatten_dims(y, 2, 0)

        # Put first dimension into the right ordering
        stream_ind = self._stream_management.stream_ind
        y = y[stream_ind]

        # Reshape first dimensions to [num_tx, num_streams]
        num_streams = self._stream_management.num_streams_per_tx
        num_tx = self._stream_management.num_tx
        y = split_dim(y, [num_tx, num_streams], 0)

        # Flatten resource grid dimensions
        # [num_tx, num_streams, num_ofdm_symbols*num_effective_subcarriers,
        #  data_dim, batch_size]
        y = flatten_dims(y, 2, 2)

        # Gather data symbols
        # [num_tx, num_streams, num_data_symbols, data_dim, batch_size]
        # Expand _data_ind to match y dimensions
        data_ind = self._data_ind.unsqueeze(-1).unsqueeze(-1)
        data_ind = data_ind.expand(-1, -1, -1, y.shape[3], y.shape[4])
        y = torch.gather(y, 2, data_ind)

        # Put batch_dim first
        # [batch_size, num_tx, num_streams, num_data_symbols, data_dim]
        y = y.permute(4, 0, 1, 2, 3)

        # Squeeze data_dim if it was added
        if squeeze_last:
            y = y.squeeze(-1)

        return y


class RemoveNulledSubcarriers(Block):
    r"""Removes nulled guard and/or DC subcarriers from a resource grid.

    :param resource_grid: :class:`~sionna.phy.ofdm.ResourceGrid` to be used
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`, :attr:`~sionna.phy.config.Config.precision` is
        used.
    :param device: Device for tensor operations. If `None`,
        :attr:`~sionna.phy.config.Config.device` is used.

    :input inputs: [batch_size, num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size], `torch.complex`.
        Full resource grid.

    :output grid: [batch_size, num_tx, num_streams_per_tx, num_ofdm_symbols, num_effective_subcarriers], `torch.complex`.
        Resource grid without nulled subcarriers.

    .. rubric:: Examples

    .. code-block:: python

        import torch
        from sionna.phy.ofdm import ResourceGrid, RemoveNulledSubcarriers

        rg = ResourceGrid(num_ofdm_symbols=14,
                          fft_size=64,
                          subcarrier_spacing=30e3,
                          num_guard_carriers=(5, 5),
                          dc_null=True)
        remover = RemoveNulledSubcarriers(rg)

        x = torch.randn(32, 1, 1, 14, 64, dtype=torch.complex64)
        y = remover(x)
        print(y.shape)
        # torch.Size([32, 1, 1, 14, 53])
    """

    def __init__(
        self,
        resource_grid: ResourceGrid,
        precision: Optional[Precision] = None,
        device: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(precision=precision, device=device, **kwargs)
        # Register as buffer for CUDAGraph compatibility
        self.register_buffer(
            "_sc_ind",
            torch.from_numpy(
                resource_grid.effective_subcarrier_ind.astype(np.int64)
            ).to(self.device),
        )

    def call(self, inputs: torch.Tensor) -> torch.Tensor:
        """Remove nulled subcarriers from resource grid.

        :param inputs: Full resource grid with shape
            `[batch_size, num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size]`
        :output grid: Resource grid without nulled subcarriers with shape
            `[batch_size, num_tx, num_streams_per_tx, num_ofdm_symbols, num_effective_subcarriers]`
        """
        # Use index_select for torch.compile compatibility
        return torch.index_select(inputs, -1, self._sc_ind)

