#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Class for creating a CIR sampler, usable as a channel model, from a CIR
generator."""

from typing import Callable, Iterator, Optional, Tuple

import torch
from torch.utils.data import DataLoader, IterableDataset

from sionna.phy import config
from .channel_model import ChannelModel

__all__ = ["CIRDataset"]


class _CIRIterableDataset(IterableDataset):
    """Internal iterable dataset wrapping a CIR generator with shuffle buffer.

    :param cir_generator: Generator that yields (a, tau) tuples
    :param buffer_size: Size of the shuffle buffer
    """

    def __init__(
        self,
        cir_generator: Callable[[], Iterator[Tuple[torch.Tensor, torch.Tensor]]],
        buffer_size: int = 32,
    ) -> None:
        super().__init__()
        self._cir_generator = cir_generator
        self._buffer_size = buffer_size

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """Iterate over the dataset with shuffling via a buffer."""
        buffer = []
        for item in self._cir_generator():
            buffer.append(item)
            if len(buffer) >= self._buffer_size:
                config.py_rng.shuffle(buffer)
                while buffer:
                    yield buffer.pop()
        # Yield remaining items
        config.py_rng.shuffle(buffer)
        while buffer:
            yield buffer.pop()


class _InfiniteCIRIterableDataset(IterableDataset):
    """Infinitely repeating iterable dataset with shuffle buffer.

    :param cir_generator: Generator that yields (a, tau) tuples
    :param buffer_size: Size of the shuffle buffer
    """

    def __init__(
        self,
        cir_generator: Callable[[], Iterator[Tuple[torch.Tensor, torch.Tensor]]],
        buffer_size: int = 32,
    ) -> None:
        super().__init__()
        self._cir_generator = cir_generator
        self._buffer_size = buffer_size

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """Iterate infinitely over the dataset with shuffling via a buffer."""
        buffer = []
        while True:
            for item in self._cir_generator():
                buffer.append(item)
                if len(buffer) >= self._buffer_size:
                    config.py_rng.shuffle(buffer)
                    while buffer:
                        yield buffer.pop()
            # Yield remaining items before restarting
            config.py_rng.shuffle(buffer)
            while buffer:
                yield buffer.pop()


def _collate_fn(
    batch: list[Tuple[torch.Tensor, torch.Tensor]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Collate function for batching CIR samples."""
    a_list, tau_list = zip(*batch)
    # Convert numpy arrays to tensors if needed
    a_list = [torch.as_tensor(a) for a in a_list]
    tau_list = [torch.as_tensor(tau) for tau in tau_list]
    a = torch.stack(a_list, dim=0)
    tau = torch.stack(tau_list, dim=0)
    return a, tau


class CIRDataset(ChannelModel):
    # pylint: disable=line-too-long
    r"""
    Creates a channel model from a dataset that can be used with classes such as
    :class:`~sionna.phy.channel.TimeChannel` and :class:`~sionna.phy.channel.OFDMChannel`.
    The dataset is defined by a `generator <https://wiki.python.org/moin/Generators>`_.

    The batch size is configured when instantiating the dataset or through the
    :attr:`~sionna.phy.channel.CIRDataset.batch_size` property.
    The number of time steps (``num_time_steps``) and sampling frequency
    (``sampling_frequency``) can only be set when instantiating the dataset.
    The specified values must be in accordance with the data.

    :param cir_generator: `Generator <https://wiki.python.org/moin/Generators>`_
        that returns channel impulse responses ``(a, tau)`` where ``a`` is the
        tensor of channel coefficients of shape
        ``[num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_steps]``
        and dtype ``torch.complex``, and ``tau`` the tensor of path delays
        of shape ``[num_rx, num_tx, num_paths]`` and dtype ``torch.float``.
    :param batch_size: Batch size
    :param num_rx: Number of receivers (:math:`N_R`)
    :param num_rx_ant: Number of antennas per receiver (:math:`N_{RA}`)
    :param num_tx: Number of transmitters (:math:`N_T`)
    :param num_tx_ant: Number of antennas per transmitter (:math:`N_{TA}`)
    :param num_paths: Number of paths (:math:`M`)
    :param num_time_steps: Number of time steps
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`, :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for computation (e.g., 'cpu', 'cuda:0').
        If `None`, :attr:`~sionna.phy.config.Config.device` is used.

    :input batch_size: `int`.
        Batch size (ignored, uses the configured batch_size).

    :input num_time_steps: `int`.
        Number of time steps (ignored, uses the configured num_time_steps).

    :input sampling_frequency: `float`.
        Sampling frequency [Hz] (ignored).

    :output a: [batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_steps], ``torch.complex``.
        Path coefficients.

    :output tau: [batch size, num_rx, num_tx, num_paths], ``torch.float``.
        Path delays [s].

    .. rubric:: Examples

    The following code snippet shows how to use this class as a channel model.

    >>> my_generator = MyGenerator(...)
    >>> channel_model = sionna.phy.channel.CIRDataset(my_generator,
    ...                                           batch_size,
    ...                                           num_rx,
    ...                                           num_rx_ant,
    ...                                           num_tx,
    ...                                           num_tx_ant,
    ...                                           num_paths,
    ...                                           num_time_steps+l_tot-1)
    >>> channel = sionna.phy.channel.TimeChannel(channel_model, bandwidth, num_time_steps)

    where ``MyGenerator`` is a generator

    >>> class MyGenerator:
    ...
    ...     def __call__(self):
    ...         ...
    ...         yield a, tau

    that returns `torch.complex` path coefficients ``a`` with shape
    ``[num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_steps]``
    and `torch.float` path delays ``tau`` (in second)
    ``[num_rx, num_tx, num_paths]``.
    """

    def __init__(
        self,
        cir_generator: Callable[[], Iterator[Tuple[torch.Tensor, torch.Tensor]]],
        batch_size: int,
        num_rx: int,
        num_rx_ant: int,
        num_tx: int,
        num_tx_ant: int,
        num_paths: int,
        num_time_steps: int,
        precision: Optional[str] = None,
        device: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(precision=precision, device=device, **kwargs)

        self._cir_generator = cir_generator
        self._batch_size = batch_size
        self._num_time_steps = num_time_steps
        self._num_rx = num_rx
        self._num_rx_ant = num_rx_ant
        self._num_tx = num_tx
        self._num_tx_ant = num_tx_ant
        self._num_paths = num_paths

        # Create the infinite iterable dataset with shuffle buffer
        self._dataset = _InfiniteCIRIterableDataset(cir_generator, buffer_size=32)

        # Create DataLoader for batching
        self._create_dataloader()

    def _create_dataloader(self) -> None:
        """Create a new DataLoader with the current batch size."""
        self._dataloader = DataLoader(
            self._dataset,
            batch_size=self._batch_size,
            collate_fn=_collate_fn,
        )
        self._iter = iter(self._dataloader)

    @property
    def batch_size(self) -> int:
        """Get/set batch size"""
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value: int) -> None:
        """Set the batch size and recreate the DataLoader."""
        self._batch_size = value
        self._create_dataloader()

    @property
    def num_time_steps(self) -> int:
        """Number of time steps"""
        return self._num_time_steps

    @property
    def num_rx(self) -> int:
        """Number of receivers"""
        return self._num_rx

    @property
    def num_rx_ant(self) -> int:
        """Number of antennas per receiver"""
        return self._num_rx_ant

    @property
    def num_tx(self) -> int:
        """Number of transmitters"""
        return self._num_tx

    @property
    def num_tx_ant(self) -> int:
        """Number of antennas per transmitter"""
        return self._num_tx_ant

    @property
    def num_paths(self) -> int:
        """Number of paths"""
        return self._num_paths

    def __call__(
        self,
        batch_size: Optional[int] = None,
        num_time_steps: Optional[int] = None,
        sampling_frequency: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample a batch from the CIR dataset."""
        a, tau = next(self._iter)

        # Convert to the correct dtype and device
        a = a.to(dtype=self.cdtype, device=self.device)
        tau = tau.to(dtype=self.dtype, device=self.device)

        return a, tau
