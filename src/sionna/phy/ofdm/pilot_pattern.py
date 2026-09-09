#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Class definition and functions related to pilot patterns"""

from typing import List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import colors

from sionna.phy import Object
from sionna.phy.config import Precision
from sionna.phy.mapping import QAMSource

__all__ = ["PilotPattern", "EmptyPilotPattern", "KroneckerPilotPattern"]


class PilotPattern(Object):
    r"""Class defining a pilot pattern for an OFDM ResourceGrid.

    A :class:`~sionna.phy.ofdm.PilotPattern` defines how transmitters send pilot
    sequences for each of their antennas or streams over an OFDM resource grid.
    It consists of two components, a ``mask`` and ``pilots``. The ``mask``
    indicates which resource elements are reserved for pilot transmissions by
    each transmitter and its respective streams. In some cases, the number of
    streams is equal to the number of transmit antennas, but this does not need
    to be the case, e.g., for precoded transmissions. The ``pilots`` contains
    the pilot symbols that are transmitted at the positions indicated by the
    ``mask``. Separating a pilot pattern into ``mask`` and ``pilots`` enables
    the implementation of a wide range of pilot configurations, including
    trainable pilot sequences.

    The pilots are mapped onto the mask from the smallest effective subcarrier
    and OFDM symbol index to the highest effective subcarrier and OFDM symbol
    index. It is important to keep this order of mapping in mind when designing
    more complex pilot sequences.

    :param mask: Tensor indicating resource elements reserved for pilot
        transmissions with shape
        `[num_tx, num_streams_per_tx, num_ofdm_symbols, num_effective_subcarriers]`
    :param pilots: The pilot symbols to be mapped onto the ``mask`` with shape
        `[num_tx, num_streams_per_tx, num_pilots]`
    :param normalize: If `True`, the ``pilots`` are normalized to an average
        energy of one across the last dimension. Defaults to `False`.
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`, :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for tensor operations. If `None`,
        :attr:`~sionna.phy.config.Config.device` is used.

    .. rubric:: Notes

    Note that ``num_effective_subcarriers`` is the number of subcarriers that
    can be used for data or pilot transmissions. Due to guard carriers or a
    nulled DC carrier, this number can be smaller than the ``fft_size`` of
    the :class:`~sionna.phy.ofdm.ResourceGrid`.

    .. rubric:: Examples

    The following code snippet shows how to define a simple custom
    :class:`~sionna.phy.ofdm.PilotPattern` for a single transmitter sending
    two streams:

    .. code-block:: python

        import numpy as np
        from sionna.phy.ofdm import PilotPattern

        num_tx = 1
        num_streams_per_tx = 2
        num_ofdm_symbols = 14
        num_effective_subcarriers = 12

        # Create a pilot mask
        mask = np.zeros([num_tx,
                         num_streams_per_tx,
                         num_ofdm_symbols,
                         num_effective_subcarriers])
        mask[0, :, [2,11], :] = 1
        num_pilot_symbols = int(np.sum(mask[0,0]))

        # Define pilot sequences
        pilots = np.zeros([num_tx,
                           num_streams_per_tx,
                           num_pilot_symbols], np.complex64)
        pilots[0, 0, 0:num_pilot_symbols:2] = (1+1j)/np.sqrt(2)
        pilots[0, 1, 1:num_pilot_symbols:2] = (1+1j)/np.sqrt(2)

        # Create a PilotPattern instance
        pp = PilotPattern(mask, pilots)

        # Visualize non-zero elements of the pilot sequence
        pp.show(show_pilot_ind=True)
    """

    def __init__(
        self,
        mask: Union[np.ndarray, torch.Tensor],
        pilots: Union[np.ndarray, torch.Tensor],
        normalize: bool = False,
        precision: Optional[Precision] = None,
        device: Optional[str] = None,
    ) -> None:
        super().__init__(precision=precision, device=device)

        # Convert mask to tensor - register as buffer for CUDAGraph compatibility
        if isinstance(mask, np.ndarray):
            self.register_buffer("_mask", torch.tensor(mask, dtype=torch.int32, device=self.device))
        else:
            self.register_buffer("_mask", mask.to(dtype=torch.int32, device=self.device))

        # Initialize _pilots buffer placeholder (will be set by property setter)
        self.register_buffer("_pilots", None)
        self.pilots = pilots
        self.normalize = normalize
        self._check_settings()

    @property
    def num_tx(self) -> int:
        """Number of transmitters"""
        return self._mask.shape[0]

    @property
    def num_streams_per_tx(self) -> int:
        """Number of streams per transmitter"""
        return self._mask.shape[1]

    @property
    def num_ofdm_symbols(self) -> int:
        """Number of OFDM symbols"""
        return self._mask.shape[2]

    @property
    def num_effective_subcarriers(self) -> int:
        """Number of effective subcarriers"""
        return self._mask.shape[3]

    @property
    def num_pilot_symbols(self) -> int:
        """Number of pilot symbols per transmit stream"""
        return self._pilots.shape[-1]

    @property
    def num_data_symbols(self) -> int:
        """Number of data symbols per transmit stream"""
        return self._mask.shape[-1] * self._mask.shape[-2] - self.num_pilot_symbols

    @property
    def normalize(self) -> bool:
        """Get/set if the pilots are normalized or not"""
        return self._normalize

    @normalize.setter
    def normalize(self, value: bool) -> None:
        self._normalize = bool(value)

    @property
    def mask(self) -> torch.Tensor:
        """Mask of the pilot pattern with shape
        `[num_tx, num_streams_per_tx, num_ofdm_symbols, num_effective_subcarriers]`
        """
        return self._mask

    @property
    def pilots(self) -> torch.Tensor:
        """Get/set the possibly normalized tensor of pilot symbols with shape
        `[num_tx, num_streams_per_tx, num_pilots]`.

        If pilots are normalized, the normalization will be applied after new
        values for pilots have been set. If this is not the desired behavior,
        turn normalization off.
        """
        if self._normalize:
            scale = self._pilots.abs().square()
            scale = 1 / scale.mean(dim=-1, keepdim=True).sqrt()
            scale = scale.to(self.cdtype)
            return scale * self._pilots
        return self._pilots

    @pilots.setter
    def pilots(self, v: Union[np.ndarray, torch.Tensor]) -> None:
        # Register as buffer for CUDAGraph compatibility
        if isinstance(v, np.ndarray):
            self.register_buffer("_pilots", torch.tensor(v, dtype=self.cdtype, device=self.device))
        else:
            self.register_buffer("_pilots", v.to(dtype=self.cdtype, device=self.device))

    def _check_settings(self) -> bool:
        """Validate that all properties define a valid pilot pattern."""
        if self._mask.dim() != 4:
            raise ValueError("`mask` must have four dimensions.")
        if self._pilots.dim() != 3:
            raise ValueError("`pilots` must have three dimensions.")
        if list(self._mask.shape[:2]) != list(self._pilots.shape[:2]):
            raise ValueError(
                "The first two dimensions of `mask` and `pilots` must be equal."
            )

        num_pilots = self._mask.sum(dim=(-2, -1))
        if num_pilots.min() != num_pilots.max():
            raise ValueError(
                "The number of nonzero elements in the masks for all "
                "transmitters and streams must be identical."
            )

        if self.num_pilot_symbols != num_pilots.max().item():
            raise ValueError(
                "The shape of the last dimension of `pilots` must equal "
                "the number of non-zero entries within the last two "
                "dimensions of `mask`."
            )

        return True

    def show(
        self,
        tx_ind: Optional[Union[int, List[int]]] = None,
        stream_ind: Optional[Union[int, List[int]]] = None,
        show_pilot_ind: bool = False,
    ) -> List[plt.Figure]:
        """Visualizes the pilot patterns for some transmitters and streams.

        :param tx_ind: Indices of transmitters to include. If `None`, all
            transmitters are included.
        :param stream_ind: Indices of streams to include. If `None`, all
            streams are included.
        :param show_pilot_ind: If `True`, the indices of the pilot symbols
            are shown. Defaults to `False`.
        :output figs: List of matplotlib figure objects showing each the pilot
            pattern from a specific transmitter and stream
        """
        mask = self.mask.cpu().numpy()
        pilots = self.pilots.cpu().numpy()

        if tx_ind is None:
            tx_ind = list(range(self.num_tx))
        elif not isinstance(tx_ind, list):
            tx_ind = [tx_ind]

        if stream_ind is None:
            stream_ind = list(range(self.num_streams_per_tx))
        elif not isinstance(stream_ind, list):
            stream_ind = [stream_ind]

        figs = []
        for i in tx_ind:
            for j in stream_ind:
                q = np.zeros_like(mask[0, 0])
                q[np.where(mask[i, j])] = (np.abs(pilots[i, j]) == 0) + 1
                legend = ["Data", "Pilots", "Masked"]
                fig = plt.figure()
                plt.title(f"TX {i} - Stream {j}")
                plt.xlabel("OFDM Symbol")
                plt.ylabel("Subcarrier Index")
                plt.xticks(range(0, q.shape[0]))
                cmap = plt.cm.tab20c
                b = np.arange(0, 4)
                norm = colors.BoundaryNorm(b, cmap.N)
                im = plt.imshow(
                    np.transpose(q),
                    interpolation="nearest",
                    origin="lower",
                    aspect="auto",
                    norm=norm,
                    cmap=cmap,
                )
                cbar = plt.colorbar(im)
                cbar.set_ticks(b[:-1] + 0.5)
                cbar.set_ticklabels(legend)

                if show_pilot_ind:
                    c = 0
                    for t in range(self.num_ofdm_symbols):
                        for k in range(self.num_effective_subcarriers):
                            if mask[i, j][t, k]:
                                if np.abs(pilots[i, j, c]) > 0:
                                    plt.annotate(c, [t, k])
                                c += 1
                figs.append(fig)

        return figs


class EmptyPilotPattern(PilotPattern):
    """Creates an empty pilot pattern.

    Generates an instance of :class:`~sionna.phy.ofdm.PilotPattern` with
    an empty ``mask`` and ``pilots``.

    :param num_tx: Number of transmitters
    :param num_streams_per_tx: Number of streams per transmitter
    :param num_ofdm_symbols: Number of OFDM symbols
    :param num_effective_subcarriers: Number of effective subcarriers
        that are available for the transmission of data and pilots.
        Note that this number is generally smaller than the ``fft_size``
        due to nulled subcarriers.
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`, :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for tensor operations. If `None`,
        :attr:`~sionna.phy.config.Config.device` is used.
    """

    def __init__(
        self,
        num_tx: int,
        num_streams_per_tx: int,
        num_ofdm_symbols: int,
        num_effective_subcarriers: int,
        precision: Optional[Precision] = None,
        device: Optional[str] = None,
    ) -> None:
        if not (num_tx > 0):
            raise ValueError("`num_tx` must be positive.")
        if not (num_streams_per_tx > 0):
            raise ValueError("`num_streams_per_tx` must be positive.")
        if not (num_ofdm_symbols > 0):
            raise ValueError("`num_ofdm_symbols` must be positive.")
        if not (num_effective_subcarriers > 0):
            raise ValueError("`num_effective_subcarriers` must be positive.")

        shape = [num_tx, num_streams_per_tx, num_ofdm_symbols,
                 num_effective_subcarriers]
        mask = np.zeros(shape, dtype=bool)
        pilots = np.zeros(shape[:2] + [0], dtype=np.complex64)
        super().__init__(
            mask, pilots, normalize=False, precision=precision, device=device
        )


class KroneckerPilotPattern(PilotPattern):
    """Simple orthogonal pilot pattern with Kronecker structure.

    This function generates an instance of
    :class:`~sionna.phy.ofdm.PilotPattern` that allocates non-overlapping pilot
    sequences for all transmitters and streams on specified OFDM symbols.
    As the same pilot sequences are reused across those OFDM symbols, the
    resulting pilot pattern has a frequency-time Kronecker structure. This
    structure enables a very efficient implementation of the LMMSE channel
    estimator. Each pilot sequence is constructed from randomly drawn QPSK
    constellation points.

    :param resource_grid: Resource grid to be used
    :param pilot_ofdm_symbol_indices: List of integers defining the OFDM
        symbol indices that are reserved for pilots
    :param normalize: If `True`, the ``pilots`` are normalized to an average
        energy of one across the last dimension. Defaults to `True`.
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`, :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for tensor operations. If `None`,
        :attr:`~sionna.phy.config.Config.device` is used.

    .. rubric:: Notes

    It is required that the ``resource_grid``'s property
    ``num_effective_subcarriers`` is an integer multiple of
    ``num_tx * num_streams_per_tx``. This condition is required to ensure
    that all transmitters and streams get non-overlapping pilot sequences.
    For a large number of streams and/or transmitters, the pilot pattern
    becomes very sparse in the frequency domain.

    .. rubric:: Examples

    .. code-block:: python

        from sionna.phy.ofdm import ResourceGrid

        rg = ResourceGrid(num_ofdm_symbols=14,
                          fft_size=64,
                          subcarrier_spacing=30e3,
                          num_tx=4,
                          num_streams_per_tx=2,
                          pilot_pattern="kronecker",
                          pilot_ofdm_symbol_indices=[2, 11])
        rg.pilot_pattern.show()
    """

    def __init__(
        self,
        resource_grid: "ResourceGrid",  # noqa: F821
        pilot_ofdm_symbol_indices: List[int],
        normalize: bool = True,
        precision: Optional[Precision] = None,
        device: Optional[str] = None,
    ) -> None:
        num_tx = resource_grid.num_tx
        num_streams_per_tx = resource_grid.num_streams_per_tx
        num_ofdm_symbols = resource_grid.num_ofdm_symbols
        num_effective_subcarriers = resource_grid.num_effective_subcarriers

        # Number of OFDM symbols carrying pilots
        num_pilot_symbols = len(pilot_ofdm_symbol_indices)

        # Compute the total number of required orthogonal sequences
        num_seq = num_tx * num_streams_per_tx

        # Compute the length of a pilot sequence
        num_pilots = num_pilot_symbols * num_effective_subcarriers / num_seq
        if (num_pilots / num_pilot_symbols) % 1 != 0:
            raise ValueError(
                "`num_effective_subcarriers` must be an integer multiple of "
                "`num_tx`*`num_streams_per_tx`."
            )

        # Number of pilots per OFDM symbol
        num_pilots_per_symbol = int(num_pilots / num_pilot_symbols)

        # Prepare empty mask and pilots
        shape = [num_tx, num_streams_per_tx,
                 num_ofdm_symbols, num_effective_subcarriers]
        mask = np.zeros(shape, bool)
        shape[2] = num_pilot_symbols
        pilots = np.zeros(shape, np.complex64)

        # Populate all selected OFDM symbols in the mask
        mask[..., pilot_ofdm_symbol_indices, :] = True

        # Populate the pilots with random QPSK symbols
        qam_source = QAMSource(2, precision=precision, device=device)
        for i in range(num_tx):
            for j in range(num_streams_per_tx):
                # Generate random QPSK symbols
                p = qam_source([1, 1, num_pilot_symbols, num_pilots_per_symbol])
                p = p.cpu().numpy()

                # Place pilots spaced by num_seq to avoid overlap
                pilots[i, j, :, i * num_streams_per_tx + j::num_seq] = p

        # Reshape the pilots tensor
        pilots = np.reshape(pilots, [num_tx, num_streams_per_tx, -1])

        super().__init__(
            mask, pilots, normalize=normalize, precision=precision, device=device
        )

